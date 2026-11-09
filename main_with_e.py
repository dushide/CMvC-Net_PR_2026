from __future__ import print_function, division
import os
import random
from tqdm import tqdm

from model_with_e import CMvCNet
from util.clusteringPerformance import StatisticClustering
from util.utils import features_to_adj
import numpy as np
import torch
import torch.nn as nn
import time
import scipy.io as sio
from scipy.sparse.linalg import svds as sparse_svds


def train(features, epoch, labels, n_view, n_clusters, model, optimizer, scheduler,
          device, lam_sparse=1e-3, lam_orth=1e-3, lam_ent=1e-3, tsne_save_path=None):
    criterion = nn.MSELoss()
    acc_max = 0.0
    res = []

    features_dev = [torch.tensor(features[v] / 1.0, dtype=torch.float32, device=device)
                    for v in range(n_view)]

    with tqdm(total=epoch, desc="Training") as pbar:
        start_time = time.time()
        for epoch_i in range(epoch):
            model.train()
            optimizer.zero_grad()

            # Deep-unfolding: each epoch starts from the fixed init state
            Z_in = model.Z_init.detach()
            D_in = [d.detach() for d in model.D_init]
            E_in = [e.detach() for e in model.E_init]
            F_in = [f.detach() for f in model.F_init]
            Q_in = [q.detach() for q in model.Q_init]

            output_z, output_d, output_e, output_f, output_q = model(
                features_dev, Z_in, D_in, E_in, F_in, Q_in
            )

            # ---- 1. 重构损失: MSE( D_v Z, X_v )（仅共性）----
            # 只监督共性分量 D_v Z 重构 X_v，F_v Q_v 作为自然互补残差，
            # 保证共性/互补分离有意义，与原始 LargeMvC-Net 训练目标一致。
            loss_rec = torch.zeros((), device=device, dtype=torch.float32)
            for k in range(n_view):
                loss_rec = loss_rec + criterion(torch.mm(output_d[k], output_z), features_dev[k])

            # ---- 2. 稀疏正则化 ----
            loss_sparse = output_z.abs().mean()
            for k in range(n_view):
                loss_sparse = loss_sparse + output_e[k].norm(p=2, dim=1).mean()
                loss_sparse = loss_sparse + output_q[k].norm(p=2, dim=0).mean()

            # ---- 3. 互补正交损失: sum_v ||D_v^T F_v||_F^2 ----
            loss_orth = torch.zeros((), device=device, dtype=torch.float32)
            for k in range(n_view):
                loss_orth = loss_orth + torch.mm(output_d[k].t(), output_f[k]).pow(2).mean()

            # ---- 4. 熵正则化 ----
            z_softmax = torch.softmax(output_z, dim=0)
            loss_ent = -(z_softmax * (z_softmax + 1e-10).log()).sum(dim=0).mean()

            # ---- 总损失 ----
            loss = (loss_rec
                    + lam_sparse * loss_sparse
                    + lam_orth   * loss_orth
                    + lam_ent    * loss_ent)

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            # ---- 聚类: Z (m × n) → 截断 SVD → KMeans ----
            Z_np = output_z.detach().cpu().numpy()
            k_svd = min(n_clusters, Z_np.shape[0] - 1, Z_np.shape[1] - 1)
            if k_svd > 0:
                _, _, Vh = sparse_svds(Z_np, k=k_svd)
                Vh = Vh[::-1].copy()   # ascending → descending
            else:
                _, _, Vh = np.linalg.svd(Z_np, full_matrices=False)
            output_zz = Vh[:n_clusters].T   # n × c

            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(
                output_zz, labels, n_clusters
            )
            if ACC[0] > acc_max:
                acc_max = ACC[0]
                res = ["{}({})".format(item[0] * 100, item[1] * 100)
                       for item in [ACC, NMI, Purity, ARI, Fscore, Precision, Recall]]
                if tsne_save_path is not None:
                    os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)
                    sio.savemat(tsne_save_path, {
                        "rep":    output_zz.astype(np.float32),
                        "labels": labels.reshape(1, -1).astype(np.float64),
                    })

            pbar.update(1)
            print({
                "ep":    epoch_i + 1,
                "Rec":   "{:.4f}".format(float(loss_rec.detach())),
                "Sp*λ":  "{:.4f}".format(float(lam_sparse * loss_sparse.detach())),
                "Ort*λ": "{:.4f}".format(float(lam_orth   * loss_orth.detach())),
                "Ent*λ": "{:.4f}".format(float(lam_ent    * loss_ent.detach())),
                "ACC":   "{:.2f}|{:.2f}".format(ACC[0] * 100, acc_max * 100),
            })

        print(f"Time: {time.time() - start_time:.2f} S")
    return res


def main(data, args):
    SCORE = ["ACC", "NMI", "Purity", "ARI", "Fscore", "Precision", "Recall"]

    seed = args.seed
    lr = args.lr
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("Using CPU")
    elif args.device == "cpu":
        device = torch.device("cpu")
        print("Using CPU (explicitly requested)")
    else:
        device = torch.device("cuda:" + args.device)
        print(f"Using CUDA:{args.device}")

    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    (
        X, Z_init, D_init, E_init, F_init, Q_init,
        m, n_view, num_sample, labels, n_clusters, r,
    ) = features_to_adj(data, args.path + args.data_path,
                        complement_rank=args.rank, anchor=args.anchor,
                        anchor_ratio=args.anchor_ratio,
                        rank_mult=getattr(args, 'rank_mult', 1))

    print("=" * 60)
    print(f"  Dataset:    {data}")
    print(f"  Views:      {n_view}    Samples: {num_sample}    Classes: {n_clusters}")
    print(f"  Anchors:    {m}     Rank: {r}     Layers: {args.layer}")
    print(f"  LR: {args.lr}  Epoch: {args.epoch}  Seed: {args.seed}")
    print(f"  thre1: {args.thre1}  thre2: {args.thre2}  thre_q: {args.thre_q}")
    print(f"  lam_sparse: {args.lam_sparse}  lam_orth: {args.lam_orth}  lam_ent: {args.lam_ent}")
    print(f"  Device:     {device}")
    print("=" * 60)

    model = CMvCNet(
        m, n_view, args.layer, r,
        args.thre1, args.thre2, args.thre_q,
        Z_init, D_init, E_init, F_init, Q_init,
        device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=15, verbose=False, min_lr=1e-8
    )

    tsne_path = (os.path.join(args.path, "tsne", f"CMvCNet_{data}.mat")
                 if getattr(args, 'save_tsne', False) else None)
    res = train(
        X, args.epoch, labels, n_view, n_clusters,
        model, optimizer, scheduler, device,
        lam_sparse=args.lam_sparse,
        lam_orth=args.lam_orth,
        lam_ent=args.lam_ent,
        tsne_save_path=tsne_path,
    )

    print("{}:{}\n".format(data, dict(zip(SCORE, res))))
    with open(args.save_file, "a") as f:
        f.write("{}:{}\n".format(data, dict(zip(SCORE, res))))
