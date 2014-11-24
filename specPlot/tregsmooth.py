import numpy as np
from scipy.optimize import minimize
import scipy.sparse as spa
from scipy.interpolate import interp1d


def ddmat(x, d):
    m = x.size
    if (d == 0):
        D = spa.identity(m)
    else:
        dx = x[d:] - x[:-d]
        V = spa.spdiags(np.divide(1, dx.T), np.array([0]), m-d, m-d)
        D = d * V * spa.dia_matrix(np.diff(ddmat(x, d-1).todense(), axis=0))
    return D


def rgdtsmcore(x, y, d, lam, xh):
    if hasattr(lam, "__len__"):
        lam = lam[0]
    N = x.size
    Nh = xh.size
    idx = interp1d(xh.A1, np.arange(Nh), kind='nearest')
    M = spa.dia_matrix(np.identity(Nh)[idx(x.A1).astype(int), :])
    D = ddmat(xh, d)
    W = spa.identity(N)
    U = spa.identity(Nh-d)
    delta = (D.H * D).diagonal().sum() / np.power(Nh, (2 + d))
    A = (M.H * W * M) + lam / delta * (D.H * U * D)
    yh = np.linalg.solve(A.todense(), M.H * W * y)
    H = M * np.linalg.solve(A.todense(), (M.H * W).todense())
    R = (M * yh) - y
    v = ((R.H * R) / np.power(N * (1 - H.trace()) / N, 2))[0, 0]
    return (yh, v)


def regdatasmooth(x, y, xh, d=2):
    maxiter = 50
    guess = 0
    f = lambda lam: rgdtsmcore(x, y, d, lam, xh)[1]
    lam = minimize(f, guess, method='Nelder-Mead', tol=1e-6,
                   options={'maxiter': maxiter, 'disp': True}).x[0]
    yh = rgdtsmcore(x, y, d, lam, xh)[0]
    return yh
