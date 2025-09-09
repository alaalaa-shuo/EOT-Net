import numpy as np

def OSP(x, per):
    # x = np.expand_dims(x, axis=-1)
    D, H, W = x.shape
    xtmp = x

    X = np.reshape(xtmp, (D, H * W)).astype(np.float64)
    # D*N
    k = None
    U, Sigma, _ = np.linalg.svd(X, full_matrices=False)
    # SVD: U,sigma,V
    Sigmalst = np.diag(Sigma)
    for j in range(1, len(Sigmalst)):
        if Sigmalst[j, j] > per * Sigmalst[j - 1, j - 1]:
            k = min(j - 1, len(Sigmalst))
            break
    # 根据奇异值，找差别最大的k个奇异值

    # Orthogonal Subspace Projection
    X_mean = np.mean(X, axis=1)
    # D:表示各个维度上的均值

    # X_centered = X - np.tile(X_mean, (X.shape[0], 1))  # 对X进行中心化处理
    X_centered = X - X_mean[:, np.newaxis]
    # X_mean[:, np.newaxis]: D*1
    Cov = np.cov(X_centered.T, rowvar=False)
    eigenValue, eigenVector = np.linalg.eig(Cov)

    idx = np.argsort(eigenValue)[::-1]
    V_selected = (eigenVector[:, idx[0:k]]).T  # 选择前k个特征向量作为主成分

    Pbkg = V_selected.T @ np.linalg.pinv(
        V_selected @ V_selected.T) @ V_selected
    Ptar = np.eye(D) - Pbkg
    X_projected_bkg = Pbkg @ X
    X_projected_tar = Ptar @ X

    row_max_bkg = X_projected_bkg.max(axis=1)
    row_min_bkg = X_projected_bkg.min(axis=1)
    X_bkg = ((X_projected_bkg - row_min_bkg[:, np.newaxis]) / (
            row_max_bkg[:, np.newaxis] - row_min_bkg[:, np.newaxis] + 0.00001))
    x_bkg = np.reshape(X_bkg, (D, H, W))

    row_max_tar = X_projected_tar.max(axis=1)
    row_min_tar = X_projected_tar.min(axis=1)
    X_tar = ((X_projected_tar - row_min_tar[:, np.newaxis]) / (
            row_max_tar[:, np.newaxis] - row_min_tar[:, np.newaxis] + 0.00001))
    x_tar = np.reshape(X_tar, (D, H, W))

    return x_bkg, x_tar