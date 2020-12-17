"""
Fuzzy C-means clustering algorithm.
"""

import numpy as np
from scipy.spatial.distance import cdist
#from .normalize_columns import normalize_columns, normalize_power_columns


def _cmeans0(data, u_old, c, m, metric):
    """
    Single step in generic fuzzy c-means clustering algorithm.
    Parameters inherited from cmeans()

    :param data:
    :param u_old:
    :param c:
    :param m:
    :param metric:
    :return:
    """

    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)

    # Choose the max value between old c-partionned matrix and a delta
    # to elimante 0 values.
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = _distance(data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    # Minimization objective function
    jm = (um * d ** 2).sum()
    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


def _distance(data, centers, metric='euclidean'):
    """
    Euclidean distance from each point to each cluster center.

    :param data: 2d array (N x Q)vData to be analyzed. There are N data points.
    :param centers: centers : 2d array (C x Q) Cluster centers. There are C clusters, with Q features.
    :param metric: string, set euclidean by default but it aceppts other
    options of ``scipy.spatial.distance.cdist``.
    :return: dist : 2d array (C x N) Euclidean distance from each point, to each cluster center.
    """

    return cdist(data, centers, metric=metric).T


def _fp_coeff(u):
    '''
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.

    :param u: 2d array (C, N) Fuzzy c-partitioned matrix;
              N = number of data points and
              C = number of clusters.
    :return: fpc : float, Fuzzy partition coefficient.
    '''

    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


def cmeans(data, c, m, error, maxiter, metric='euclidean', init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].

    Parameters

    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.

    c : Desired number of clusters or classes.

    m : Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m, normally equals to 2

    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.

    maxiter : int
        Maximum number of iterations allowed.

    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.

    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.

    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns

    cntr : 2d array, size (S, c)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.

    """
    # Setup u0 first init matrice
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)
    # softmax

    # Initialize loop parameters
    jm = np.zeros(0)

    # Number of iterations
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()

        [cntr, u, Jjm, d] = _cmeans0(data, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule Frobenius à revoir
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


def cmeans_predict(test_data, cntr_trained, m, error, maxiter, metric='euclidean', init=None,
                   seed=None):
    """
    Prediction of new data in given a trained fuzzy c-means framework [1].

    Parameters
    ----------
    test_data : 2d array, size (S, N)
        New, independent data set to be predicted based on trained c-means
        from ``cmeans``. N is the number of data sets; S is the number of
        features within each sample vector.
    cntr_trained : 2d array, size (S, c)
        Location of trained centers from prior training c-means.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (S, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.

    Returns
    -------
    u : 2d array, (S, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (S, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (S, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.


    """
    c = cntr_trained.shape[0]

    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = test_data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        [u, Jjm, d] = _cmeans_predict0(test_data, cntr_trained, u2, c, m, metric)
        jm = np.hstack((jm, Jjm))
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return u, u0, d, jm, p, fpc


def _cmeans_predict0(test_data, cntr, u_old, c, m, metric):
    """

    Parameters inherited from cmeans()

    Very similar to initial clustering, except `cntr` is not updated, thus
    the new test data are forced into known (trained) clusters.

    :param test_data:
    :param cntr:
    :param u_old:
    :param c:
    :param m:
    :param metric:
    :return:
    """


    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m
    test_data = test_data.T

    # For prediction, we do not recalculate cluster centers. The test_data is
    # forced to conform to the prior clustering.

    d = _distance(test_data, cntr, metric)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return u, jm, d


def normalize_columns(columns):
    """
    Normalize columns of matrix.

    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns

    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns / np.sum(columns, axis=0, keepdims=1)

    return normalized_columns


def normalize_power_columns(x, exponent):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.

    Parameters
    ----------
    x : 2d array (M x N)
        Matrix with columns
    n : float
        Exponent

    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**n) but safe

    """

    assert np.all(x >= 0.0)

    x = x.astype(np.float64)

    # values in range [0, 1]
    x = x / np.max(x, axis=0, keepdims=True)

    # values in range [eps, 1]
    x = np.fmax(x, np.finfo(x.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        x /= np.min(x, axis=0, keepdims=True)

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        x = x ** exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        x = x ** exponent

    result = normalize_columns(x)

    return result