import numpy as np
import scipy.io as scio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import h5py


def _read_h5_cell(f, dataset):
    """Read a MATLAB cell array stored as an HDF5 dataset of object references."""
    return [np.array(f[dataset[i, 0]]) for i in range(dataset.shape[0])]


def features_to_adj(datasets, path="./data/", complement_rank=None, anchor=None, anchor_ratio=1, rank_mult=1):
    print("loading {} data...".format(datasets))
    filepath = '{}{}.mat'.format(path, datasets)

    try:
        matData = scio.loadmat(filepath)
        is_h5 = False
    except NotImplementedError:
        matData = h5py.File(filepath, 'r')
        is_h5 = True

    try:
        if is_h5:
            if datasets in ["YTF50", "YTF100"]:
                X = [np.array(v).T for v in _read_h5_cell(matData, matData['fea'])]
                Y = np.array(matData['gt'])
            else:
                X = [np.array(v).T for v in _read_h5_cell(matData, matData['X'])]
                Y = np.array(matData['Y'])
        else:
            if datasets in ["YTF50", "YTF100"]:
                X = matData['fea'][0]
                Y = matData['gt']
            else:
                X = matData['X'][0]
                Y = matData['Y']
    finally:
        if is_h5:
            matData.close()

    num_view = len(X)
    labels = Y.reshape(-1,)
    num_sample = labels.shape[0]
    n_cluster = len(set(np.array(labels)))
    m = anchor if anchor is not None else n_cluster * anchor_ratio

    # Standardize each view
    XX = []
    for p in range(num_view):
        scaler = StandardScaler(with_mean=True, with_std=True)
        if not isinstance(X[p], np.ndarray):
            X[p] = np.array(X[p])
        X[p] = scaler.fit_transform(X[p].T)
        XX.append(X[p])

    XX = np.vstack(XX)
    # svds requires k < min(shape); clamp m to valid range
    k_z = min(m, XX.shape[0] - 1, XX.shape[1] - 1)
    k_z = max(1, k_z)
    if k_z < m:
        print(f"[WARN] anchor {m} clamped to {k_z} (svds limit for shape {XX.T.shape})")
        m = k_z
    U, _, _ = svds(XX.T, k=m)

    # K-means to initialize Z (m × n)
    np.random.seed(12)
    kmeans = KMeans(n_clusters=m, max_iter=100, n_init=10, random_state=12)
    idx = kmeans.fit_predict(U)

    Z = np.zeros((m, num_sample))
    for i in range(num_sample):
        Z[idx[i], i] = 1

    # Initialize D (d_v × m) and E (d_v × n) for each view via SVD Procrustes
    D, E = [], []
    for i in range(num_view):
        C = np.dot(X[i], Z.T)
        Uc, _, Vt = svd(C, full_matrices=False)
        D_i = np.dot(Uc, Vt)
        D.append(D_i)
        E.append(X[i] - np.dot(D_i, Z))

    # Complementary bases F (d_v × r) and coefficients Q (r × n)
    # F initialized from residual X_v - D_v @ Z so it captures complementary info
    if complement_rank is None:
        complement_rank = n_cluster * rank_mult
    d_list = [X[i].shape[0] for i in range(num_view)]
    r = int(min(complement_rank, m, min(d_list), num_sample))
    r = max(1, r)

    F, Q = [], []
    for i in range(num_view):
        residual = X[i] - np.dot(D[i], Z)
        k_f = min(r, residual.shape[0] - 1, residual.shape[1] - 1)
        k_f = max(1, k_f)
        Uf, _, _ = svds(residual, k=k_f)
        F_i = Uf[:, ::-1].copy()
        if k_f < r:
            F_i = np.hstack([F_i, np.zeros((F_i.shape[0], r - k_f))])
        Q_i = np.dot(F_i.T, X[i])
        F.append(F_i)
        Q.append(Q_i)

    return X, Z, D, E, F, Q, m, num_view, num_sample, labels, n_cluster, r
