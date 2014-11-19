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
from plotspec import range_filter
from datetime import datetime


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


def calibration_error(measpath, refpath, t=1.0):
    refdata = np.genfromtxt(refpath, delimiter=",", skip_header=1)
    measdata = np.genfromtxt(measpath, delimiter=",", skip_header=2)
    ind = np.where((measdata[:, 0] >= refdata[:, 0].min()) *
                   (measdata[:, 0] <= refdata[:, 0].max()))[0]
    lam = measdata[ind, 0] * 1e-9
    ref = interp1d(1e-9 * refdata[:, 0], refdata[:, 1], kind='cubic')
    cor_tmp = (lam * t * ref(lam))/(codata.h * codata.c)
    cor = cor_tmp/(measdata[ind, 1:].mean(axis=1))
    corh = cor_tmp/(measdata[ind, 1:].mean(axis=1) +
                    measdata[ind, 1:].std(axis=1))
    corl = cor_tmp/(measdata[ind, 1:].mean(axis=1) -
                    measdata[ind, 1:].std(axis=1))
    return (lam, cor, corh, corl, ind)


def measurement(paths, cal, ind, t=1.0):
    cal = np.atleast_2d(cal).transpose()
    Phi = np.zeros((ind.size, 0), float)
    fidx = []
    cidx = 0
    for idx, path in enumerate(paths):
        if (type(t) is np.ndarray):
            tint = t[idx]
        else:
            tint = t
        tmp = np.genfromtxt(path, delimiter=",", skip_header=2)
        Phi = np.hstack((Phi, tmp[ind, 1:]*cal/tint))
        fidx.append(np.array(range(cidx, Phi.shape[1])))
        cidx = Phi.shape[1]
    return (Phi, fidx)


def find_paths(dpath):
    calpaths = sorted(glob(dpath + "c*.csv"))
    dtapaths = sorted(glob(dpath + "D*.csv"))
    return (calpaths, dtapaths)


def norm_range(lam, lmin, lmax, Phi):
    k = np.where((lam >= lmin) * (lam <= lmax))[0]
    Phim = np.amax(Phi[k, :], axis=0)
    return Phi/Phim


def fix_jag(idx, Phi):
    Phic = Phi
    Phic[idx:, :] = Phi[idx:, :] * (Phi[[idx-1], :] / Phi[[idx], :])
    return Phic


def integrate_flux(lam, Phi, rngs=None):
    if (rngs is None):
        rngs = {'x': np.array([lam.min(), lam.max()]),
                'y': np.array([Phi.min(), Phi.max()])}
    (lam, Phi) = range_filter(lam, Phi, rngs)
    iPhi = np.zeros((1, Phi.shape[1]))
    for k in range(0, Phi.shape[1]):
        kn = np.where(~np.isnan(Phi[:, k]))[0]
        iPhi[0, k] = np.trapz(Phi[kn, k], lam[kn, 0])
    return iPhi


def find_times(paths, fmt="%m/%d/%Y %H:%M:%S"):
    dts = []
    for path in paths:
        with open(path, 'r') as f:
            dts.extend([datetime.strptime(d, fmt) for d
                        in f.readline()[:-1].split(',')[1:]])
    return dts
