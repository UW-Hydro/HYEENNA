import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors

K = 5
METRIC = 'chebyshev'
EPS = 1e-10


def nearest_distances(X: np.array, k: int=K) -> list:
    knn = NearestNeighbors(n_neighbors=k, metric=METRIC)
    knn.fit(X)
    d, _ = knn.kneighbors(X)
    return d[:, -1]


def marginal_neighbors(X: np.array, R: np.array) -> list:
    knn = NearestNeighbors(metric=METRIC)
    knn.fit(X)
    return np.array([len(knn.radius_neighbors(p.reshape(1, -1), r)[0][0])
                     for p, r in zip(X, R)])


def entropy(X: np.array, k: int=K) -> float:
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    assert d == 1, "Use multivariate_entropy"
    e = 2 * nearest_distances(X, k) + EPS * np.random.random(size=n)
    ent = d * np.mean(np.log(e)) + psi(n) - psi(k) + np.log(1)
    return ent

""" THIS IS WRONG THIS IS WRONG THIS IS WRONG THIS IS WRONG """
def multivariate_entropy(X1n: list, k: int=K) -> float:
    # TODO: FIXME: This only handles 1d variables
    raise NotImplementedError()
    r_arr = [nearest_distances(X, k) + EPS * np.random.random(size=len(X))
             for X in X1n]
    ent = psi(len(X1n[0])) - psi(k) + np.log(2) - np.sum(r_arr)
    return ent


def mutual_info(X: np.array, Y: np.array, k: int=K) -> float:
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    n, d = X.shape
    r = nearest_distances(np.hstack([X, Y]), k) + EPS * np.random.random(size=n)
    n_x = marginal_neighbors(X, r)
    n_y = marginal_neighbors(Y, r)
    return psi(n) + psi(k) - (1./k) - np.mean(psi(n_x+1) + psi(n_y+1))


def conditional_mutual_info(
        X: np.array, Y: np.array, Z: np.array, k: int=K) -> float:
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    n, dX = X.shape
    n, dY = Y.shape
    n, dZ = Z.shape
    xz = np.hstack([X, Z])
    yz = np.hstack([Y, Z])
    z = np.hstack([Z])
    xyz = np.hstack([X, Y, Z])
    r = nearest_distances(xyz, k) + EPS * np.random.random(size=n)
    n_xz = marginal_neighbors(xz, r)
    n_yz = marginal_neighbors(yz, r)
    n_z = marginal_neighbors(z, r)
    return psi(k) - np.mean(psi(n_xz+1) + psi(n_yz+1) - psi(n_z+1))


def transfer_entropy(X: np.array, Y: np.array, tau: int,
                     omega: int, k: int, l: int, neighbors=10, **kwargs) -> float:
    end = -(np.max([k, l]) + np.max([tau, omega]))
    x = np.array(Y[:end]).reshape(-1, 1)
    y, z = [], []
    for w in range(tau, tau+k):
        y.append(X[w:end+w])
    for w in range(omega, omega+l):
        z.append(Y[w:end+w])
    if k > 1:
        y = np.vstack(y).T
    else:
        y = np.array(y).reshape(-1, 1)
    if l > 1:
        z = np.vstack(z).T
    else:
        z = np.array(z).reshape(-1, 1)
    return conditional_mutual_info(x, y, z, k=neighbors)


def conditional_transfer_entropy(X: np.array, Y: np.array, Z: np.array,
                                 tau: int, omega: int, nu: int,
                                 k: int, l: int, m: int, neighbors=K) -> float:
    end = len(X) - (np.max([k, l, m]) + np.max([tau, omega, nu]))
    nZ, dZ = Z.shape
    y = [Y[:end]]
    nX = len(X[:end])
    y, z1, z2 = [], [], []
    for w in range(tau, tau+k):
        y.append(X[w:end+w])
    for w in range(omega, omega+l):
        z1.append(Y[w:end+w])
    for w in range(nu, nu+m):
        z2.append(Z[w:end+w].reshape(-1, nX))
    # TODO: FIXME: Make sure these are flexible
    x = np.array(x).reshape(-1, 1)
    y = np.hstack(y)
    z1 = np.hstack(z1)
    if len(z2):
        z2 = np.hstack(z2)
        z = np.hstack([np.array(z1), np.array(z2)])
    else:
        z = z1
    return conditional_mutual_info(x, y, z, k=neighbors)
