import numpy as np
from scipy.optimize import minimize
# from scipy.sparse import identity, diags
from numpy import identity, diag
from scipy.interpolate import interp1d


def ddmat(x, d):
    m = x.size
    if (d == 0):
        D = identity(m)
    else:
        dx = x[d:] - x[0:(m - d)]
        V = diag(1 / dx)
        D = d * np.dot(V, np.diff(ddmat(x, d - 1), axis=0))
    return D


def rgdtsmcore(x, y, d, lam, xh):
    N = x.size
    Nh = xh.size
    idx = interp1d(xh, np.arange(Nh), kind='nearest')
    M = identity(Nh)[idx(x).astype(int), :]
    D = ddmat(xh, d)
    W = identity(N)
    U = identity(Nh-d)
    delta = np.trace(np.dot(D.conj().transpose(), D)) / np.power(Nh, (2 + d))
    A = (np.dot(M.conj().transpose(), np.dot(W, M))
         + lam/delta * np.dot(D.conj().transpose(), np.dot(U, D)))
    yh = np.linalg.solve(A,
                         np.dot(M.conj().transpose(), np.dot(W, y)))
    H = np.dot(M, np.linalg.solve(A, np.dot(M.conj().transpose(), W)))
    R = (np.dot(M, yh) - y)
    v = np.dot(R.conj().transpose(), R) / np.power(N * (1 - np.trace(H) / N),
                                                   2)
    return (yh, v)


def regdatasmooth(x, y, xh, d=2):
    maxiter = 50
    guess = 0
    f = lambda lam: rgdtsmcore(x, y, d, lam, xh)[1]
    lam = minimize(f, guess, method='Nelder-Mead', tol=1e-6,
                   options={'maxiter': maxiter, 'disp': True}).x[0]
    yh = rgdtsmcore(x, y, d, lam, xh)[0]
    return yh
