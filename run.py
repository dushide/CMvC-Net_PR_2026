import sys
import argparse
from main_with_e import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## Parameter setting
    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)
    parser.add_argument("--data_path", type=str, default='/data/', help="Path of datasets.")
    parser.add_argument("--save_file", type=str, default="res_para.txt", help="Path of datasets.")
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--use_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed. Default is 40.")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument("--norm", action='store_true', default=True, help="Normalize the feature.")
    parser.add_argument("--layer", type=int, default="2", help="Number of unfolding layers.")  # network layers
    parser.add_argument("--thre1", type=float, default="0.01", help="thre1")  # parameter for L1 regularization
    parser.add_argument("--thre2", type=float, default="0.01", help="thre2")  # parameter for L21 regularization
    parser.add_argument("--lr", type=float, default="0.001", help="lr") # Learning rate
    parser.add_argument("--epoch", type=int, default="200", help="epochs")
    parser.add_argument("--anchor", type=int, default=None,
                        help="Number of anchors m. Default: anchor_ratio * n_clusters (auto).")
    parser.add_argument("--anchor_ratio", type=int, default="1",
                        help="Anchor multiplier when --anchor is not set. m = anchor_ratio * n_clusters.")
    parser.add_argument("--rank", type=int, default=None, help="Complementary rank r for F_v. Default: n_clusters.")
    parser.add_argument("--thre_q", type=float, default="0.001", help="Threshold for Q_v row L2,1 shrinkage.")
    parser.add_argument("--lam_sparse", type=float, default="1.0", help="Weight for sparse regularization loss.")
    parser.add_argument("--lam_orth",   type=float, default="1.0", help="Weight for complementary orthogonality loss.")
    parser.add_argument("--lam_ent",    type=float, default="1.0", help="Weight for entropy regularization loss.")
    parser.add_argument("--rank_mult",  type=int,   default=1,
                        help="Rank multiplier: r = rank_mult * n_clusters (used when --rank is None).")
    parser.add_argument("--save_tsne", action="store_true", default=False,
                        help="Save best-epoch t-SNE representation to tsne/CMvCNet_{dataset}.mat.")
    parser.add_argument("--dataset_override", type=str, default=None,
                        help="Run a single dataset by name (used by sensitivity.py).")
    args = parser.parse_args()

    dataset = {1: "Animals", 2: "Caltech101-all", 3: "cifar10", 4: "MITIndoor"
               , 5: "Notting-Hill",  6: "YTF50", 7: "scene15", 8: "YTF100"}  # dataset list

    if args.dataset_override is not None:
        # Called from sensitivity.py: run the named dataset directly
        main(args.dataset_override, args)
    else:
        # 1 Animals, 2 Caltech101-all, 3 cifar10, 4 MITIndoor, 5 Notting-Hill, 6 YTF50
        select_dataset = list(range(1, 7))
        for i in select_dataset:
            main(dataset[i], args)

