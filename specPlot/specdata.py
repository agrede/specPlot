"""
Spectrometer Data Reader

Copyright (C) 2014 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import scipy.constants as codata
from scipy.interpolate import interp1d
from glob import glob


def calibration(measpath, refpath, t=1.0):
    refdata = np.genfromtxt(refpath, delimiter=",", skip_header=1)
    measdata = np.genfromtxt(measpath, delimiter=",", skip_header=2)
    ind = np.where((measdata[:, 0] >= refdata[:, 0].min()) *
                   (measdata[:, 0] <= refdata[:, 0].max()))[0]
    lam = measdata[ind, 0] * 1e-9
    ref = interp1d(1e-9 * refdata[:, 0], refdata[:, 1], kind='cubic')
    cor = (lam * t * ref(lam))/(measdata[ind, 1:].mean(axis=1)
                                * codata.h * codata.c)
    return (lam, cor, ind)


def measurement(paths, cal, ind, t=1.0):
    cal = np.atleast_2d(cal).transpose()
    Phi = np.zeros((ind.size, 0), float)
    for idx, path in enumerate(paths):
        tmp = np.genfromtxt(path, delimiter=",", skip_header=2)
        Phi = np.hstack((Phi, tmp[ind, 1:]*cal/t))
    return Phi


def find_paths(dpath):
    calpaths = sorted(glob(dpath + "c*.csv"))
    dtapaths = sorted(glob(dpath + "D*.csv"))
    return (calpaths, dtapaths)


def norm_range(lam, lmin, lmax, Phi):
    k = np.where((lam >= lmin) * (lam <= lmax))[0]
    Phim = np.amax(Phi[k, :], axis=0)
    return Phi/Phim


def integrate_flux(lam, Phi, lmin=None, lmax=None):
    pass
