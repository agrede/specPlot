"""
Spectrometer Data Plotter

Copyright (C) 2014 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as codata
import regex
from jinja2 import Environment, FileSystemLoader

LATEX_SUBS = (
    (regex.compile(r'\\'), r'\\textbackslash'),
    (regex.compile(r'([{}_#%&$])'), r'\\\1'),
    (regex.compile(r'~'), r'\~{}'),
    (regex.compile(r'\^'), r'\^{}'),
    (regex.compile(r'"'), r"''"),
    (regex.compile(r'\.\.\.+'), r'\\ldots'),
)


def escape_tex(value):
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval

texenv = Environment(loader=FileSystemLoader('./templates'))
texenv.BLOCK_START_STRING = '((*'
texenv.BLOCK_END_STRING = '*))'
texenv.VARIABLE_START_STRING = '((('
texenv.VARIABLE_END_STRING = ')))'
texenv.COMMENT_START_STRING = '((='
texenv.COMMENT_END_STRING = '=))'
texenv.filters['escape_tex'] = escape_tex


def elam(lam):
    return codata.h * codata.c / (codata.e * lam)


def data_ranges(x, major, minor):
    xmax = x.max()
    xmin = x.min()
    xrng = xmax - xmin
    step_range = np.array([0.1, 0.2, 0.5, 1])
    step_digits = np.array([-1, -1, -1, 0])
    m = np.ceil(np.log10(xrng/major))
    M = np.power(10, m)
    ks = np.where(xrng / (M * step_range) < major)[0][0]
    step = M * step_range[ks]
    step_digit = m + step_digits[ks]
    nxmax = xmax + (step - np.mod(xmax, step)) * (np.mod(xmax, step) != 0)
    nxmin = xmin - np.mod(xmin, step)
    nmxmax = nxmax
    nmxmin = nxmin
    stepm = None
    if (minor > 0):
        Mm = np.power(10, step_digit)
        ksm = np.where(step / (Mm * step_range) < minor)[0][0]
        stepm = Mm * step_range[ksm]
        if (xmax < nxmax-stepm):
            nxmax = nxmax-step
            nmxmax = xmax + (stepm - np.mod(xmax, stepm))
        if (xmin > nxmin+stepm):
            nxmin = nxmin + step
            nmxmin = xmin - np.mod(xmin, stepm)
    return (nmxmin, nmxmax, nxmin, nxmax, step, stepm)


def tosinum(xs, option):
    return ['\protect{\num[' + option + ']{%.1e}}' % x for x in xs]


def mkplot(pth, lam, data, labels, legend, rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        kl = np.where((lam >= rngs['x'][0]) * (lam <= rngs['x'][1]))[0]
        lam = np.atleast_2d(lam[kl])
        data = data[kl, :]
        kd = np.where(((data < rngs['y'][0]) + (data > rngs['y'][1])) > 0)[0]
        data[kd] = None
    if (limits is None or ticks is None):
        siopt = 'scientific-notation=fixed, fixed-exponent=0'
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(lam * 1e9,
                                                                7, 7)
        (x2min, x2max, x2jmin, x2jmax, x2step, x2stepm) = data_ranges(
            elam(lam), 7, 4)
        (ymin, ymax, yjmin, yjmax, ystep, ystepm) = data_ranges(data, 7, 7)
        limits = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        xticks = np.linspace(xjmin, xjmax,
                             np.round((xjmax - xjmin) / xstep) + 1)
        yticks = np.linspace(yjmin, yjmax,
                             np.round((yjmax - yjmin) / ystep) + 1)
        ticks = {
            'xmajor': xticks,
            'xminor': np.linspace(xmin, xmax,
                                  np.round((xmax - xmin) / xstepm) + 1),
            'xlabels': tosinum(xticks, siopt),
            'xmajor': np.linspace(xjmin, xjmax,
                                  np.round((xjmax - xjmin) / xstep) + 1),
            'xminor': np.linspace(xmin, xmax,
                                  np.round((xmax - xmin) / xstepm) + 1),
            'x2major': np.linspace(elam(x2jmax), elam(x2jmin),
                                   np.round((x2jmax - x2jmin) / x2step + 1)),
            'x2minor': np.linspace(elam(x2max), elam(x2min),
                                   np.round((x2max - x2min) / x2stepm + 1)),
            'x2labels': tosinum(
                np.linspace(x2max, x2min,
                            np.round((x2max - x2min) / x2step + 1)), siopt),
            'ymajor': yticks,
            'yminor': np.linspace(ymin, ymax,
                                  np.round((ymax - ymin) / ystepm) + 1),
            'ylabels': tosinum(yticks, siopt)
        }
    np.savetxt(pth+".csv", np.hstack((lam, data)), delimiter=',')
    template = texenv.get_template('plot.tex')
    f = open(pth+".txt", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, legend=legend))
    f.close()


def prime_factors(n):
    """Returns all the prime factors of a positive integer"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
    return factors
