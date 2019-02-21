import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors

K = 5
METRIC = 'chebyshev'
EPS = 1e-10


def nearest_distances_vec(X: np.array, Y: np.array=None,
                          k: int=K, metric=METRIC) -> np.array:
    """Find vector distance to all k nearest neighbors"""
    knn = NearestNeighbors(n_neighbors=k, metric=METRIC)
    knn.fit(X)
    if Y is not None:
        _, l = knn.kneighbors(Y)
        dvec = np.array([X[i] - Y[ll[1:]] for i, ll in enumerate(l)])
    else:
        _, l = knn.kneighbors(X)
        dvec = np.array([X[i] - X[ll[1:]] for i, ll in enumerate(l)])
    return dvec


def nearest_distances(X: np.array, Y: np.array=None,
                      k: int=K, metric=METRIC) -> list:
    """Distance to the kth nearest neighbor"""
    knn = NearestNeighbors(n_neighbors=k, metric=METRIC)
    knn.fit(X)
    if Y is not None:
        d, _ = knn.kneighbors(Y)
    else:
        d, _ = knn.kneighbors(X)
    return d[:, -1]


def marginal_neighbors(X: np.array, R: np.array, metric=METRIC) -> list:
    """Number of neighbors within a certain radius"""
    knn = NearestNeighbors(metric=METRIC)
    knn.fit(X)
    return np.array([len(knn.radius_neighbors(p.reshape(1, -1), r)[0][0])
                     for p, r in zip(X, R)])


def entropy(X: np.array, k: int=K) -> float:
    """
    Computes the Shannon entropy of a random variable X using
    the KL nearest neighbor estimator.

    The formula is given by:
        $$
        \hat{H}(X) = \psi(N) - \psi(k) + \log(C_d)
        + d \langle \log(\epsilon) \rangle
        $$
    where
        - $N$ is the number of samples
        - $k$ is the number of neighbors
        - $\psi is the digamma function
        - $\rangle \cdot \rangle$ is the mean
        - $\epsilon_i$ is the 2 times the distance to the
          $k^{th}$ nearest neighbor.

    Parameters
    ----------
    X: np.array
        Sample from a random variable
    k: int, optional
        Number of neighbors to use in estimation

    Returns
    -------
    ent: float
        estimated entropy

    References
    ----------
    .. [0] Goria, M. N., Leonenko, N. N., Mergel, V. V., & Inverardi, P. L. N.
       (2005). A new class of random vector entropy estimators and its
       applications in testing statistical hypotheses. Journal of
       Nonparametric Statistics, 17(3), 277–297.
       https://doi.org/10.1080/104852504200026815
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('Data must be given as a numpy array!')
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    n, d = X.shape
    e = 2 * nearest_distances(X, k=k+1) + EPS * np.random.random(size=n)
    ent = d * np.mean(np.log(e)) + psi(n) - psi(k) + np.log(1)
    return ent


def conditional_entropy(X: np.array, Y: np.array, k: int=K) -> float:
    """
    Computes the conditional Shannon entropy of a sample of a random
    variable X given another sample of a random variable Y using an
    adaptation of the KL and KSG estimators

    Parameters
    ----------
    X: np.array
        Sample from a random variable
    Y: np.array
        Sample from a random variable
    k: int, optional
        Number of neighbors to use in estimation

    Returns
    -------
    cent: float
        estimated conditional entropy

    References
    ----------
    .. [0] Goria, M. N., Leonenko, N. N., Mergel, V. V., & Inverardi, P. L. N.
       (2005). A new class of random vector entropy estimators and its
       applications in testing statistical hypotheses. Journal of
       Nonparametric Statistics, 17(3), 277–297.
    .. [1] - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
       Estimating mutual information. Physical Review E - Statistical Physics,
       Plasmas, Fluids, and Related Interdisciplinary Topics, 69(6), 16.
       https://doi.org/10.1103/PhysRevE.69.066138
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    n, d = X.shape
    assert X.shape[0] == Y.shape[0], "{} - {}".format(X.shape, Y.shape)
    r = (nearest_distances(np.hstack([X, Y]), k=k+1)
         - EPS * np.random.random(size=n))
    n_y = marginal_neighbors(Y, r)
    return d * np.mean(np.log(2*r)) + np.mean(psi(n_y)) - psi(k)


def mutual_info(X: np.array, Y: np.array, k: int=K) -> float:
    """
    Computes the Mututal information of two random variables, X
    and Y, using the KSG nearest neighbor estimator.

    The formula is given by:
        $$
        \hat{I}(X,Y) = \psi(N) + \psi(k) - \frac{1}{k}
        - \langle \psi(n_X +1) + \psi(n_Y +1) \rangle
        $$
    where
        - $N$ is the number of samples
        - $k$ is the number of neighbors
        - $\psi is the digamma function
        - $\rangle \cdot\rangle$ is the mean
        - $\n_i$ is the number of points within the distance of
          the $k^{th}$ nearest neighbor when projected into the
          subspace spanned by $i$.

    Parameters
    ----------
    X: np.array
        A sample from a random variable
    Y: np.array
        A sample from a random variable
    k: int, optional
        Number of neighbors to use in estimation.

    Returns
    -------
    mi: float
        The mutual information

    References
    ----------
    ..  [0] - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
        Estimating mutual information. Physical Review E - Statistical Physics,
        Plasmas, Fluids, and Related Interdisciplinary Topics, 69(6), 16.
        https://doi.org/10.1103/PhysRevE.69.066138
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    n, d = X.shape
    assert X.shape[0] == Y.shape[0], "{} - {}".format(X.shape, Y.shape)
    r = (nearest_distances(np.hstack([X, Y]), k=k+1)
         - EPS * np.random.random(size=n))
    n_x = marginal_neighbors(X, r)
    n_y = marginal_neighbors(Y, r)
    return psi(n) + psi(k) - (1./k) - np.mean(psi(n_x+1) + psi(n_y+1))


def mi_local_nonuniformity_correction(X, *args, k: int=K,
                                      alpha=1.05, **kwargs):
    """
    Compute the local nonuniformity correction factor for strongly
    dependent variables. This correction is calculated based on the
    structure of the space of k-nearest neighbors. The volume of
    the hyper-rectangle of the maximum-norm bounding box for the
    k-nearest neighbor estimation is compared to that of the
    hyper-rectangle bounding the principal components of the covariance
    matrix of the k-nearest neighbor locations.

    Parameters
    ----------
    X: np.array
        A sample from a random variable
    *args: List[np.array]
        Samples from random variables
    k: int, optional
        Number of neighbors to use in estimation.
    alpha: float, optional
        Sensitivity parameter for filtering non-dependent volumes
    **kwargs: np.array
        Samples from random variables

    Returns
    -------
    lnc: float
        The correction factor to be subtracted from the mutual information

    References
    ----------
    .. [0] - Gao, S., Steeg, G. V., & Galstyan, A. (2014). Efficient
    Estimation of Mutual Information for Strongly Dependent Variables.
    Retrieved from https://arxiv.org/abs/1411.2003v3
    """
    data = [X, *args, *kwargs.values()]
    for i, d in enumerate(data):
        if len(d.shape) == 1:
            data[i] = d.reshape(-1, 1)
    assert data[0].shape == data[-1].shape
    n, d = data[0].shape
    M = (nearest_distances_vec(np.hstack(data), k=k+1)
         + EPS * np.random.random(size=(n,k, d)))

    # Compute volume of hypercube bounding r[i]
    V = np.prod(np.max(np.abs(M), axis=1), axis=-1)

    # Compute volume of PCA of neighbors
    C = np.array([np.cov(M[i].T) for i in range(M.shape[0])])
    eigvals, _ = np.linalg.eig(C)
    V_bar = np.prod(np.sqrt(eigvals), axis=-1)

    # Compute correction factor
    lnc = np.log(V_bar / V)
    mask = (V_bar / V) >= alpha
    lnc[mask] = 0
    return np.mean(lnc)


def multi_mutual_info(X: np.array, *args, k: int=K, **kwargs) -> float:
    """
    Computes the multivariate mututal information of several random
    variables using the KSG nearest neighbor estimator.

    The formula is given by:
        $$
        \hat{I}(X_1,...,X_m) = (m-1)\cdot\psi(N) + \psi(k) - \frac{m-1}{k}
        - \langle \psi(n_{X_1} +1) + ... + \psi(n_{X_m} +1) \rangle
        $$
    where
        - $N$ is the number of samples
        - $m$ is the number of variables
        - $k$ is the number of neighbors
        - $\psi is the digamma function
        - $\rangle \cdot\rangle$ is the mean
        - $\n_i$ is the number of points within the distance of
          the $k^{th}$ nearest neighbor when projected into the
          subspace spanned by $i$.

    Parameters
    ----------
    X: np.array
        A sample from a random variable
    *args: List[np.array]
        Samples from random variables
    k: int, optional
        Number of neighbors to use in estimation.
    **kwargs: np.array
        Samples from random variables

    Returns
    -------
    mi: float
        The mutual information

    References
    ----------
    ..  [0] - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
        Estimating mutual information. Physical Review E - Statistical Physics,
        Plasmas, Fluids, and Related Interdisciplinary Topics, 69(6), 16.
        https://doi.org/10.1103/PhysRevE.69.066138
    """
    data = [X, *args, *kwargs.values()]
    for i, d in enumerate(data):
        if len(d.shape) == 1:
            data[i] = d.reshape(-1, 1)
    assert data[0].shape == data[-1].shape
    n, d = data[0].shape
    r = (nearest_distances(np.hstack(data), k=k+1)
         - EPS * np.random.random(size=n))
    n_i = [marginal_neighbors(d, r) for d in data]
    return (psi(k) - (len(data)-1)/k + (len(data)-1) * psi(n)
            - np.mean(np.sum([psi(n+1) for n in n_i], axis=0)))


def conditional_mutual_info(
        X: np.array, Y: np.array, Z: np.array, k: int=K) -> float:
    """
    Compute the conditional mutual information

    Parameters
    ----------
    X: np.array
        Sample from random variable X
    Y: np.array
        Sample from random variable Y
    Z: np.array
        Sample from random variable Z
    k: int, optional
        Number of neighbors to use in estimation

    Returns
    -------
    estimated conditional mutual information

    References
    ----------
    .. [0] - Vlachos, I., & Kugiumtzis, D. (2010).
       Non-uniform state space reconstruction and coupling detection.
       https://doi.org/10.1103/PhysRevE.82.016207
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)
    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)
    nX, dX = X.shape
    nY, dY = Y.shape
    nZ, dZ = Z.shape
    assert nX == nY, "{} - {}".format(X.shape, Y.shape)
    assert nZ == nY, "{} - {}".format(Z.shape, Y.shape)
    xz = np.hstack([X, Z])
    yz = np.hstack([Y, Z])
    z = np.hstack([Z])
    xyz = np.hstack([X, Y, Z])
    r = nearest_distances(xyz, k=k+1) + EPS * np.random.random(size=nX)
    n_xz = marginal_neighbors(xz, r)
    n_yz = marginal_neighbors(yz, r)
    n_z = marginal_neighbors(z, r)
    return psi(k) - np.mean(psi(n_xz+1) + psi(n_yz+1) - psi(n_z+1))


def kl_divergence(P: np.array, Q: np.array, k: int=K):
    """
    Compute the KL divergence

    Parameters
    ----------
    P: np.array
        Sample from random variable P
    Q: np.array
        Sample from random variable Q
    k: int, optional
       Number of neighbors to use in estimation

    Returns
    -------
    estimated KL divergence D(P|Q)

    References
    ----------
    .. [0] - Wang, Q., Kulkarni, S. R., & Verdu, S. (2006). A Nearest-Neighbor
       Approach to Estimating Divergence between Continuous Random Vectors.
       In 2006 IEEE International Symposium on Information Theory.
       https://doi.org/10.1109/ISIT.2006.261842
    """
    if len(P.shape) == 1:
        P = P.reshape(-1, 1)

    if len(Q.shape) == 1:
        Q = Q.reshape(-1, 1)

    nP, dP = P.shape
    nQ, dQ = Q.shape
    assert dP == dQ, "{} - {}".format(P.shape, Q.shape)

    nu = nearest_distances(P, k=k+1, metric=METRIC)
    rho = nearest_distances(Q, P, k=k, metric=METRIC)

    div = (dP * (np.mean(np.log(nu)) - np.mean(np.log(rho)))
           + np.log(nQ / (nP-1)))
    return div


def transfer_entropy(X: np.array, Y: np.array,
                     tau: int=1, omega: int=1,
                     k: int=1, l: int=1,
                     neighbors: int=K, **kwargs) -> float:
    """
    Compute the transfer entropy from a source variable, X, to
    a target variable, Y.

    Parameters
    ----------
    X: np.array
        Source sample from a random variable X
    Y: np.array
        Target sample from a random variable Y
    tau: int (default: 1)
        Number of timestep lags for the source variable
    omega: int (default: 1)
        Number of timestep lags for the target variable conditioning
    k: int (default: 1)
        Width of window for the source variable.
    l: int (default: 1)
        Width of window for the target variable conditioning.
    neighbors: int (default: K)
        Parameter controlling the number of neighbors to use in estimation.
    **kwargs:
        Other arguments (undocumented, for internal usage)

    Returns
    -------
    transfer_entropy: float
        Computed via conditional_mutual_info

    References
    ----------
    .. [0] Schreiber, T. (2000). Measuring information transfer.
       Physical Review Letters, 85(2), 461–464.
       https://doi.org/10.1103/PhysRevLett.85.461
    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    start = np.max([k, l]) + np.max([tau, omega])
    nX, dX = X.shape
    nY, dY = Y.shape

    x = Y[start:]
    y, z = [], []

    for w in range(tau, tau+k):
        y.append(X[start-w:-w])

    for w in range(omega, omega+l):
        z.append(Y[start-w:-w])

    if k > 1:
        y = np.hstack(y)
    else:
        y = np.array(y).reshape(-1, 1)

    if l > 1:
        z = np.hstack(z)
    else:
        z = np.array(z).reshape(-1, 1)

    return conditional_mutual_info(x, y, z, k=neighbors)


def conditional_transfer_entropy(X: np.array, Y: np.array, Z: np.array,
                                 tau: int=1, omega: int=1, nu: int=1,
                                 k: int=1, l: int=1, m: int=1,
                                 neighbors: int=K, **kwargs) -> float:
    """
    Compute the transfer entropy from a source variable, X, to
    a target variable, Y, conditioned on other variables contained
    in Z.

    Parameters
    ----------
    X: np.array
        Source sample from a random variable X
    Y: np.array
        Target sample from a random variable Y
    Z: np.array
        Conditioning variable(s).
    tau: int (default: 1)
        Number of timestep lags for the source variable
    omega: int (default: 1)
        Number of timestep lags for the target variable conditioning
    nu: int (default: 1)
        Number of timestep lags for the source variable conditioning
    k: int (default: 1)
        Width of window for the source variable.
    l: int (default: 1)
        Width of window for the target variable conditioning.
    m: int (default: 1)
        Width of window for the source variable conditioning.
    neighbors: int (default: K)
        Parameter controlling the number of neighbors to use in estimation.
    **kwargs:
        Other arguments (undocumented, for internal usage)

    Returns
    -------
    conditional_transfer_entropy: float
        Computed via conditional_mutual_info

    References
    ----------
    .. [0] Schreiber, T. (2000). Measuring information transfer.
       Physical Review Letters, 85(2), 461–464.
       https://doi.org/10.1103/PhysRevLett.85.461

    """
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    if len(Z.shape) == 1:
        Z = Z.reshape(-1, 1)

    start = np.max([k, l, m]) + np.max([tau, omega, nu])
    nZ, dZ = Z.shape
    x = Y[start:]
    nX, dX = x.shape
    y, z1, z2 = [], [], []

    for w in range(tau, tau+k):
        y.append(X[start-w:-w])

    for w in range(omega, omega+l):
        z1.append(Y[start-w:-w])

    for w in range(nu, nu+m):
        z2.append(Z[start-w:-w].reshape(nX, dX))

    if k > 1:
        y = np.hstack(y)
    else:
        y = np.array(y).reshape(-1, 1)

    if l > 1:
        z1 = np.hstack(z1)
    else:
        z1 = np.array(z1).reshape(-1, 1)

    if len(z2):
        z2 = np.hstack(z2)
        z = np.hstack([np.array(z1), np.array(z2)])
    else:
        z = z1

    return conditional_mutual_info(x, y, z, k=neighbors)
