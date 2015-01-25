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

texenv = Environment(autoescape=False,
                     loader=FileSystemLoader('./templates'))
texenv.block_start_string = '((*'
texenv.block_end_string = '*))'
texenv.variable_start_string = '((('
texenv.variable_end_string = ')))'
texenv.comment_start_string = '((='
texenv.comment_end_string = '=))'
texenv.filters['escape_tex'] = escape_tex


def elam(lam):
    return codata.h * codata.c / (codata.e * lam)


def data_ranges(x, major, minor):
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
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
        tmp = np.where(step / (Mm * step_range) < minor)[0]
        if (tmp.size > 0):
            ksm = tmp[0]
            stepm = Mm * step_range[ksm]
            if (xmax <= nxmax-stepm):
                nxmax = nxmax-step
                nmxmax = xmax + (stepm - np.mod(xmax, stepm))
            if (xmin >= nxmin+stepm):
                nxmin = nxmin + step
                nmxmin = xmin - np.mod(xmin, stepm)
    return (nmxmin, nmxmax, nxmin, nxmax, step, stepm)


def data_inner_ranges(x, major, minor):
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
    xrng = xmax - xmin
    step_range = np.array([0.1, 0.2, 0.5, 1])
    step_digits = np.array([-1, -1, -1, 0])
    m = np.ceil(np.log10(xrng/major))
    M = np.power(10, m)
    ks = np.where(xrng / (M * step_range) < major)[0][0]
    step = M * step_range[ks]
    step_digit = m + step_digits[ks]
    nxmax = xmax - np.mod(xmax, step)
    nxmin = xmin + (step - np.mod(xmin, step)) * (np.mod(xmin, step) != 0)
    nmxmax = nxmax
    nmxmin = nxmin
    stepm = None
    if (minor > 0):
        Mm = np.power(10, step_digit)
        tmp = np.where(step / (Mm * step_range) < minor)[0]
        if (tmp.size > 0):
            ksm = tmp[0]
            stepm = Mm * step_range[ksm]
            if (xmax >= nxmax + stepm):
                nmxmax = xmax - np.mod(xmax, stepm)
            if (xmin <= nxmin - stepm):
                nmxmin = xmin + (stepm - np.mod(xmin, stepm))
    return (nmxmin, nmxmax, nxmin, nxmax, step, stepm)


def tosinum(xs, option):
    return ['\protect{\\num[' + option + ']{%.1e}}' % x for x in xs]


def range_filter(x, y, rngs):
    kx = np.where((x >= rngs['x'][0]) * (x <= rngs['x'][1]))[0]
    x = np.atleast_2d(x[kx]).transpose()
    y = y[kx, :]
    ky = np.where(((y < rngs['y'][0]) + (y > rngs['y'][1])) > 0)[0]
    y[ky] = np.nan
    return (x, y)


def default_labels():
    return {
        'xlabel': {
            'text': 'Wavelength',
            'symbol': '\\lambda',
            'units': '\\nm'
        },
        'ylabel': {
            'text': 'Photon Flux',
            'symbol': '\\phi',
            'units': '\\arb'
        },
        'x2label': {
            'text': 'Photon Energy',
            'symbol': 'E_{\\lambda}',
            'units': '\\eV'
        }
    }


def default_labels_temp():
    tmp = default_labels()
    tmp['zlabel'] = {
        'text': 'Temperature',
        'symbol': 'T',
        'units': '\K'
    }
    return tmp


def mkplot(pth, lam, data, legend, autoscale=True, labels=default_labels(),
           rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        (lam, data) = range_filter(lam, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data)
    if (limits is None or ticks is None):
        siopt = 'scientific-notation=fixed, fixed-exponent=0'
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(lam * 1e9,
                                                                7, 7)
        (x2min, x2max, x2jmin, x2jmax, x2step, x2stepm) = data_inner_ranges(
            elam(np.array([xmin, xmax])*1e-9), 7, 7)
        (ymin, ymax, yjmin, yjmax, ystep, ystepm) = data_ranges(data, 7, 7)
        limits = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        ticks = {}
        ticks['xmajor'] = np.linspace(xjmin, xjmax,
                                      np.round((xjmax - xjmin) / xstep) + 1)
        ticks['xlabels'] = tosinum(ticks['xmajor'], siopt)
        if (xstepm is not None):
            ticks['xminor'] = np.linspace(xmin, xmax,
                                          np.round((xmax - xmin) / xstepm) + 1)
        ticks['x2major'] = elam(np.linspace(x2jmax, x2jmin,
                                            np.round((x2jmax - x2jmin) / x2step
                                                     + 1)))*1e9
        ticks['x2labels'] = tosinum(
            np.linspace(x2jmax, x2jmin,
                        np.round((x2jmax - x2jmin) / x2step + 1)), siopt)
        if (x2stepm is not None):
            ticks['x2minor'] =  elam(np.linspace(x2max, x2min,
                                                 np.round((x2max - x2min)
                                                          / x2stepm + 1)))*1e9
        ticks['ymajor'] = np.linspace(yjmin, yjmax,
                                      np.round((yjmax - yjmin) / ystep) + 1)
        ticks['ylabels'] = tosinum(ticks['ymajor'], siopt)
        if (ystepm is not None):
            ticks['yminor'] = np.linspace(ymin, ymax,
                                          np.round((ymax - ymin) / ystepm) + 1)
    np.savetxt(pth+".csv", np.hstack((lam*1e9, data)), delimiter=',')
    print(ticks['xlabels'])
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, legend=legend))
    f.close()


def mkzplot(pth, lam, data, zs, autoscale=True, labels=default_labels_temp(),
            rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        (lam, data) = range_filter(lam, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data)
    if (limits is None or ticks is None):
        siopt = 'scientific-notation=fixed, fixed-exponent=0'
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(lam * 1e9,
                                                                7, 7)
        (x2min, x2max, x2jmin, x2jmax, x2step, x2stepm) = data_inner_ranges(
            elam(np.array([xmin, xmax])*1e-9), 7, 7)
        (ymin, ymax, yjmin, yjmax, ystep, ystepm) = data_ranges(data, 7, 7)
        (zmin, zmax, zjmin, zjmax, zstep, zstepm) = data_ranges(zs, 7, 7)
        limits = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'zmin': zmin,
            'zmax': zmax
        }
        ticks = {}
        ticks['xmajor'] = np.linspace(xjmin, xjmax,
                                      np.round((xjmax - xjmin) / xstep) + 1)
        ticks['xlabels'] = tosinum(ticks['xmajor'], siopt)
        if (xstepm is not None):
            ticks['xminor'] = np.linspace(xmin, xmax,
                                          np.round((xmax - xmin) / xstepm) + 1)
        ticks['x2major'] = elam(np.linspace(x2jmax, x2jmin,
                                            np.round((x2jmax - x2jmin) / x2step
                                                     + 1)))*1e9
        ticks['x2labels'] = tosinum(
            np.linspace(x2jmax, x2jmin,
                        np.round((x2jmax - x2jmin) / x2step + 1)), siopt)
        if (x2stepm is not None):
            ticks['x2minor'] =  elam(np.linspace(x2max, x2min,
                                                 np.round((x2max - x2min)
                                                          / x2stepm + 1)))*1e9
        ticks['ymajor'] = np.linspace(yjmin, yjmax,
                                      np.round((yjmax - yjmin) / ystep) + 1)
        ticks['ylabels'] = tosinum(ticks['ymajor'], siopt)
        if (ystepm is not None):
            ticks['yminor'] = np.linspace(ymin, ymax,
                                          np.round((ymax - ymin) / ystepm) + 1)
    np.savetxt(pth+".csv", np.hstack((lam*1e9, data)), delimiter=',')
    print(ticks['xlabels'])
    template = texenv.get_template('temp.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, zs=zs))
    f.close()


def int_plot_labels(xlabel, xsymbol, xunit, xhighlight=None, hltxt=None,
                    ypos=0.5):
    lbl = {
        'xlabel': {
            'text': xlabel,
            'symbol': xsymbol,
            'units': xunit
        },
        'ylabel': {
            'text': 'Integrated Photon Flux',
            'units': '\\arb'
        }
    }
    if (xhighlight is not None):
        lbl['xhighlight'] = {}
        lbl['xhighlight']['start'] = xhighlight[0]
        lbl['xhighlight']['end'] = xhighlight[1]
        if (hltxt is not None):
            lbl['xhighlight']['lbltxt'] = hltxt
            lbl['xhighlight']['lblanch'] = [(0.2 *
                                             (xhighlight[1] - xhighlight[0])
                                             + xhighlight[0]),
                                            ypos]
    return lbl


def mkintplot(pth, x, data, legend, autoscale=True, labels=default_labels(),
              rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        (x, data) = range_filter(x, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data, axis=0)
    if (limits is None or ticks is None):
        siopt = 'scientific-notation=fixed, fixed-exponent=0'
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(x, 7, 7)
        (ymin, ymax, yjmin, yjmax, ystep, ystepm) = data_ranges(data, 7, 7)
        limits = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        ticks = {}
        ticks['xmajor'] = np.linspace(xjmin, xjmax,
                                      np.round((xjmax - xjmin) / xstep) + 1)
        ticks['xlabels'] = tosinum(ticks['xmajor'], siopt)
        if (xstepm is not None):
            ticks['xminor'] = np.linspace(xmin, xmax,
                                          np.round((xmax - xmin) / xstepm) + 1)
        ticks['ymajor'] = np.linspace(yjmin, yjmax,
                                      np.round((yjmax - yjmin) / ystep) + 1)
        ticks['ylabels'] = tosinum(ticks['ymajor'], siopt)
        if (ystepm is not None):
            ticks['yminor'] = np.linspace(ymin, ymax,
                                          np.round((ymax - ymin) / ystepm) + 1)
    np.savetxt(pth+".csv", np.hstack((x, data)), delimiter=',')
    print(ticks['xlabels'])
    template = texenv.get_template('int_flux_over.tex')
    f = open(pth+".tex", 'w')
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
