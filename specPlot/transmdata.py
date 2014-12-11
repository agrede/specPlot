"""
Transmission Measurement Reader

Copyright (C) 2014 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np


def rawdata(paths, mult, skip=1):
    for idx, path in enumerate(paths):
        if (type(mult) is np.ndarray()):
            m = mult[idx]
        else:
            m = mult
        tmp = np.getfromtxt(path, delimiter="\t", skip_header=skip)
        if (idx == 0):
            lam = tmp[:, 0]*1e-9
            amp = np.zeros((lam.size, paths.size), float)
            phase = np.zeros((lam.size, paths.size), float)
        amp[:, idx] = tmp[:, 1]*m
        phase[:, idx] = np.radians(tmp[:, 2])
    return (lam, amp, phase)
