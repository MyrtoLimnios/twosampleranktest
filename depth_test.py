# Functions for depth test Tukey adapted from https://github.com/GuillaumeStaermanML/DRPM

from sklearn.preprocessing import normalize
from sklearn.covariance import MinCovDet as MCD
from sklearn.decomposition import PCA
import numpy as np
from statistics import mode, multimode

def cov_matrix(X, robust=False):
    """ Compute the covariance matrix of X.
    """
    if robust:
        cov = MCD().fit(X)
        sigma = cov.covariance_
    else:
        sigma = np.cov(X.T)

    return sigma

def standardize(X, robust=False):
    """ Compute the square inverse of the covariance matrix of X.
    """
    sigma = cov_matrix(X, robust)
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(sigma)

    if (rank < n_features):
        pca = PCA(rank)
        pca.fit(X)
        X_transf= pca.fit_transform(X)
        sigma = cov_matrix(X_transf)
    else:
        X_transf = X.copy()

    u, s, _ = np.linalg.svd(sigma)
    square_inv_matrix = u / np.sqrt(s)

    return X_transf@square_inv_matrix

def sampled_sphere(n_dirs, d):
    """ Produce ndirs samples of d-dimensional uniform distribution on the
        unit sphere
    """

    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=n_dirs)

    return normalize(U)


def tukey_depth_proj(Xtrain, XYtest, n_dirs=None):
    """ Compute the score of the classical tukey depth of X w.r.t. X

    Parameters
    ----------
    X : Array of shape (n_samples, n_features)
            The training set.

    ndirs : int | None
        The number of random directions to compute the score.
        If None, the number of directions is chosen as
        n_features * 100.

    Return
    -------
    tukey_score: Array of float
        Depth score of each delement in X.
    """
    X = Xtrain
    n_samples, n_features = X.shape

    if n_dirs is None:
        n_dirs = n_features * 100

    # Simulated random directions on the unit sphere
    U = sampled_sphere(n_dirs, n_features)

    sequence = np.arange(1, n_samples + 1)
    depth = np.zeros((n_samples, n_dirs))

    # Compute projections
    proj = np.matmul(X, U.T)

    rank_matrix = np.matrix.argsort(proj, axis=0)

    for k in range(n_dirs):
        depth[rank_matrix[:, k], k] = sequence

    depth = depth / (n_samples * 1.)

    ### Optimal direction
    if len(multimode(np.argmin(depth, axis=1))) == n_dirs:
        u_star = np.sum(U, axis= 0) / (n_dirs * 1.)
        print('mean dir')

    else:
        u_star = U[mode(np.argmin(depth, axis=1))] ## vecteur minimisant le plus frequement l'échantillon


    proj_XYtest = np.matmul(XYtest, u_star.T)

    return proj_XYtest