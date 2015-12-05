"""
Spectrometer Data Reader

Copyright (C) 2014--2015 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import scipy.constants as codata
from scipy.interpolate import interp1d
from glob import glob
from plotspec import range_filter
from datetime import datetime
import re


def calibration(measpath, refpath="OOIntensityData.csv", t=1.0):
    refdata = np.genfromtxt(refpath, delimiter=",", skip_header=1)
    measdata = np.genfromtxt(measpath, delimiter=",", skip_header=2)
    ind = np.where((measdata[:, 0] >= refdata[:, 0].min()) *
                   (measdata[:, 0] <= refdata[:, 0].max()))[0]
    lam = measdata[ind, 0] * 1e-9
    ref = interp1d(1e-9 * refdata[:, 0], refdata[:, 1], kind='cubic')
    cor = (lam * t * ref(lam))/(measdata[ind, 1:].mean(axis=1)
                                * codata.h * codata.c)
    return (lam, cor, ind)


def calibration_error(measpath, refpath="OOIntensityData.csv", t=1.0):
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


def img_calibration(Xpath, Mpath, refpath="OOIntensityData.csv", t=1.0):
    refdata = np.genfromtxt(refpath, delimiter=",", skip_header=1)
    lamdata = np.genfromtxt(Xpath, delimiter=",", skip_header=1)
    measdata = np.genfromtxt(Mpath, delimiter=",", skip_header=0)
    ind = np.where((lamdata >= refdata[:, 0].min()) *
                   (lamdata <= refdata[:, 0].max()))[0]
    lam = lamdata[ind] * 1e-9
    ref = interp1d(1e-9 * refdata[:, 0], refdata[:, 1], kind='cubic')
    cor = (lam * t * ref(lam))/(measdata[:, ind] * codata.h * codata.c)
    return (lam, cor, ind)


def measurement(paths, cal, ind, t=1.0):
    cal = np.atleast_2d(cal).transpose()
    (ltmp, tmp, fidx) = cps(paths, t=t)
    Phi = tmp[ind, :]*cal
    return (Phi, fidx)


def img_measure(paths, cal, ind, t=1.0):
    tmp = img_counts(paths, t=t)
    Phi = tmp[:, ind, :]*cal
    return Phi


def img_counts(paths, t=1.0):
    tmp = np.genfromtxt(paths[0], delimiter=",")
    cnts = np.zeros((tmp.shape[0], tmp.shape[1], paths.size),
                    float)
    for idx, path in enumerate(paths):
        if (type(t) is np.ndarray):
            tint = t[idx]
        else:
            tint = t
        tmp = np.genfromtxt(path, delimiter=",")/tint
        if (tmp.shape is cnts[:, :, 0].shape):
            cnts[:, :, idx] = tmp
        else:
            print("Error in : "+path)
    return cnts


def cps(paths, t=1.0):
    tmp = np.genfromtxt(paths[0], delimiter=",", skip_header=2)
    lam = tmp[:, 0]
    cnts = np.zeros((lam.size, 0), float)
    fidx = []
    cidx = 0
    for idx, path in enumerate(paths):
        if (type(t) is np.ndarray):
            tint = t[idx]
        else:
            tint = t
        tmp = np.genfromtxt(path, delimiter=",", skip_header=2)
        if (cnts.shape[0] == tmp.shape[0]):
            cnts = np.hstack((cnts, tmp[:, 1:]/tint))
            fidx.append(np.array(range(cidx, cnts.shape[1])))
            cidx = cnts.shape[1]
        else:
            print(path+" Wrong number of rows")
    return (lam, cnts, fidx)


def find_paths(dpath):
    calpaths = np.array(sorted(glob(dpath + "c*.csv")))
    dtapaths = np.array(sorted(glob(dpath + "D*.csv")))
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
                        in f.readline()[:-2].split(',')[1:]])
    return dts


def time_diff(times):
    return np.array([(t-times[0]).total_seconds() for t in times])


def read_absorb(paths):
    DIn = np.genfromtxt(paths[0], delimiter="\t", skip_header=1)
    DIr = np.genfromtxt(paths[1], delimiter="\t", skip_header=1)
    lam = DIn[:, 0]*1e-9
    A = 1-DIn[:, 1]/DIr[:, 1]
    return (lam, A)


def read_absorbs(paths, cuton=580e-9):
    lam = np.array([])
    A = np.array([])
    pcut = None
    for idx, path in enumerate(paths):
        (tlam, tA) = read_absorb(path)
        if (type(cuton) is np.ndarray):
            tcut = cuton[idx]
        elif (idx < 1):
            tcut = cuton
        else:
            tcut = None
        if (pcut is None):
            k0 = 0
        else:
            k0 = np.where(tlam >= pcut)[0][0]
        if (tcut is None):
            kN = tlam.size
        else:
            kN = np.where(tlam >= tcut)[0][0] - 1
        lam = np.append(lam, tlam[k0:kN])
        A = np.append(A, tA[k0:kN])
        pcut = tcut
    return (lam, A)


def read_log(path, pattern):
    ptrn = re.compile(pattern)
    rtn = {}
    with open(path, 'r') as f:
        for l in f:
            m = ptrn.match(l)
            if m:
                rtn[m.group(1)] = m.groups()
    return rtn


def read_scope(paths):
    v = []
    t = np.zeros((1000, len(paths)))
    for k, pth in enumerate(paths):
        tmp = np.genfromtxt(pth, delimiter=",", skip_header=2)
        t[:, k] = tmp[:, 0]
        v.append(tmp[:, 1:])
    return (t, v)


def enum(paths):
    return [(k, p.split("/")[-1]) for k, p in enumerate(paths)]
