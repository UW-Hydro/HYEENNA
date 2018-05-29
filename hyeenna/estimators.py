import numpy as np
from scipy.special import spi
from sklearn.neighbors import NearestNeighbors

METRIC = 'chebyshev'
EPS = 1e-12
K = 10

def nearest_distances(X: np.array, k: int=K) -> list:
    knn = NearestNeighbors(n_neighbors=k, metric=METRIC)
    knn.fit(X)
    d, _ = knn.kneighbors(X)
    return d[:, -1]


def marginal_neighbors(X: np.array, R: np.array) -> list:
    knn = NearestNeighbors(metric=METRIC)
    knn.fit(X)
    return [len(knn.radius_neighbors(p.reshape(1,-1), r)[0][0])
            for p, r in zip(X,R)]


def entropy(X: np.array, k: int=K) -> float:
    if len(X.shape) == 1:
	X = X.reshape(-1, 1)
    n, d = X.shape
    r = nearest_distances(x, k) + eps * np.random.random(n)
    ent = d * np.log(np.mean(r)) + psi(n) - psi(k) + d * np.log(2)
    return ent


def mutual_info(X: np.array, Y: np.array, k: int=K) -> float:
    if len(X.shape) == 1:
	X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
	Y = Y.reshape(-1, 1)
    n, d = X.shape
    r = nearest_distances(np.hstack([X, Y]), k) - EPS * np.random.random(n)
    n_x = marginal_neighbors(x, r)
    n_y = marginal_neighbors(y, r)
    return psi(n) + psi(k) - (1./k) - np.mean(psi(n_x) + psi(n_y))


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
    z  = np.hstack([Z])
    xyz = np.hstack([X, Y, Z])
    r = nearest_distances(xyz, k) + EPS * np.random.random(n)
    n_xz = marginal_neighbors(xz, r)
    n_yz = marginal_neighbors(yz, r)
    n_z  = marginal_neighbors(z, r)
    return psi(k) - np.mean(psi(n_xz) + psi(n_yz) - psi(n_z))


def transfer_entropy(X: np.array, Y: np.array,
	tau: int, omega: int, k: int, l: int) -> float:
    start = np.max([k,l]) + lag
    x = [X[start:]]
    y, z = [], []
    for w in range(tau, tau+l+1):
        y.append(Y[start-tau-w:-tau-w])
    for w in range(1, k+1):
        z.append(X[1][start-omega-w:-omega-w])
    #TODO: FIXME: Make sure these are flexible
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1, l)
    z = np.array(z).reshape(-1, k)
    return conditional_mutual_info(x, y, z)


def conditional_transfer_entropy(X: np.array, Y: np.array, Z: np.array,
	tau: int, omega: int, k: int, l: int) -> float:
    start = np.max([k,l]) + lag
    x = [X[start:]]
    y, z = [], []
    for w in range(tau, tau+l+1):
        y.append(X[start-tau-w:-tau-w])
    for w in range(omega, omega+k+1):
        z.append(Y[start-omega-w:-omega-w])
        z.append(Z[start-omega-w:-omega-w])
    #TODO: FIXME: Make sure these are flexible
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1, l)
    z = np.array(z).reshape(-1, 2*k)
    return conditional_mutual_info(x, y, z)


