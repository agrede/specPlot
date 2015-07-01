"""
Spectrometer Data Plotter

Copyright (C) 2014--2015 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as codata
import re as regex
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
texenv.lstrip_blocks = True
texenv.trim_blocks = True


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
            if (nmxmin >= nxmin-stepm):
                nmxmin = nxmin-stepm
            if (nmxmax <= nxmax+stepm):
                nmxmax = nxmax+stepm
    return (nmxmin, nmxmax, nxmin, nxmax, step, stepm)


def data_log_ranges(x):
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
    lmax = np.ceil((np.log10(xmax)))
    lmin = np.floor((np.log10(xmin)))
    pmax = np.power(10, lmax)
    pmin = np.power(10, lmin)
    if (xmax/pmax > 0.8):
        pmax = 2*pmax
        lmax = lmax+1
    if (pmin/xmin < 2):
        pmin = 0.8*pmin
        lmin = lmin-1
    return (pmin, pmax, lmin, lmax)


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
    kx, ky = np.where(((np.isnan(y))))
    y[kx, ky] = rngs['y'][0]-1
    kx, ky = np.where(((y < rngs['y'][0]) + (y > rngs['y'][1])) > 0)
    y[kx, ky] = np.nan
    return (x, y)


def range_filter_2d(x, y, rngs):
    kx1, ky1 = np.where(((x < rngs['x'][0]) + (x > rngs['x'][1])) > 0)
    kx2, ky2 = np.where(((y < rngs['y'][0]) + (y > rngs['y'][1])) > 0)
    x[kx1, ky1] = np.nan
    y[kx1, ky1] = np.nan
    x[kx2, ky2] = np.nan
    y[kx2, ky2] = np.nan
    k = np.where((np.isnan(x).sum(axis=1) < x.shape[1]))[0]
    x = x[k, :]
    y = y[k, :]
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
            'symbol': 'h \\nu',
            'units': '\\eV'
        }
    }


def default_Elabels():
    return {
        'xlabel': {
            'text': 'Photon Energy',
            'symbol': 'h \\nu',
            'units': '\\eV'
        },
        'ylabel': {
            'text': 'Photon Flux',
            'symbol': '\\phi',
            'units': '\\arb'
        },
        'x2label': {
            'text': 'Wavelength',
            'symbol': '\\lambda',
            'units': '\\nm'
        }
    }


def default_Ezlabels():
    tmp = default_Elabels()
    tmp['zlabel'] = {
        'text': 'Temperature',
        'symbol': 'T',
        'units': '\\K'
    }
    return tmp


def default_labels_temp():
    tmp = default_labels()
    tmp['zlabel'] = {
        'text': 'Temperature',
        'symbol': 'T',
        'units': '\\K'
    }
    return tmp


def mkplot_axis(prefix, x, rtype='outer', major=7, minor=7):
    if (rtype is 'outer'):
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(x, 7, 7)
    else:
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_inner_ranges(x, 7, 7)
    ticks = {prefix+'major': '%f,%f,...,%f' % (
        xjmin, xjmin+xstep, xjmax)}
    limits = {prefix+'min': xmin, prefix+'max': xmax}
    if (xstepm is not None):
        ticks[prefix+'minor'] = '%f,%f,...,%f' % (
            xmin, xmin+xstepm, xmax)
    ticks[prefix+'labels'] = -np.round(np.log10(xstep))
    if (ticks[prefix+'labels'] < 0):
        ticks[prefix+'labels'] = 0
    return (ticks, limits)


def mkplot_special_axis(prefix, x, fun, rtype='inner', major=7, minor=7,
                        siopt='scientific-notation=fixed, fixed-exponent=0'):
    if (rtype is 'outer'):
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_ranges(x, 7, 7)
    else:
        (xmin, xmax, xjmin, xjmax, xstep, xstepm) = data_inner_ranges(x, 7, 7)
    ticks = {}
    ticks[prefix+'major'] = ",".join(
        ["%f" % fun(y) for y in np.linspace(
            xjmax, xjmin, np.round((xjmax - xjmin) / xstep + 1))])
    ticks[prefix+'labels'] = tosinum(
        np.linspace(xjmax, xjmin,
                    np.round((xjmax - xjmin) / xstep + 1)), siopt)
    if (xstepm is not None):
        ticks[prefix+'minor'] = ",".join(
            ["%f" % fun(y) for y in np.linspace(
                xmax, xmin, np.round((xmax - xmin) / xstepm + 1))])
    limits = {prefix+'min': xmin, prefix+'max': xmax}
    return (ticks, limits)


def mkplot_gen(lam, data, autoscale, rngs, limits, ticks):
    if (rngs is not None):
        (lam, data) = range_filter(lam, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data)
    if (limits is None or ticks is None):
        limits = {}
        ticks = {}
        (t_ticks, t_limits) = mkplot_axis('x', lam*1e9)
        ticks.update(t_ticks)
        limits.update(t_limits)
        (t_ticks, t_limits) = mkplot_special_axis(
            'x2',
            elam(np.array([limits['xmin'], limits['xmax']])*1e-9),
            lambda x: elam(x)*1e9)
        ticks.update(t_ticks)
        (t_ticks, t_limits) = mkplot_axis('y', data)
        ticks.update(t_ticks)
        limits.update(t_limits)
    return (limits, ticks, lam, data)


def mkplot_genE(E, data, autoscale, rngs, limits, ticks):
    if (rngs is not None):
        (E, data) = range_filter(E, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data)
    if (limits is None or ticks is None):
        limits = {}
        ticks = {}
        (t_ticks, t_limits) = mkplot_axis('x', E)
        ticks.update(t_ticks)
        limits.update(t_limits)
        (t_ticks, t_limits) = mkplot_special_axis(
            'x2',
            elam(np.array([limits['xmin'], limits['xmax']]))*1e9,
            lambda x: elam(x)*1e9)
        ticks.update(t_ticks)
        (t_ticks, t_limits) = mkplot_axis('y', data)
        ticks.update(t_ticks)
        limits.update(t_limits)
    return (limits, ticks, E, data)


def mkplot(pth, lam, data, legend, autoscale=True, labels=default_labels(),
           rngs=None, limits=None, ticks=None):
    (limits, ticks, lam, data) = mkplot_gen(
        lam, data, autoscale, rngs, limits, ticks)
    np.savetxt(pth+".csv", np.hstack((lam*1e9, data)), delimiter=',')
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, legend=legend))
    f.close()


def mkEplot(pth, E, data, legend, autoscale=True, labels=default_Elabels(),
            rngs=None, limits=None, ticks=None):
    (limits, ticks, E, data) = mkplot_genE(
        E, data, autoscale, rngs, limits, ticks)
    np.savetxt(pth+".csv", np.hstack((E, data)), delimiter=',')
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, legend=legend))
    f.close()


def mkzplot(pth, lam, data, zs, autoscale=True, labels=default_labels_temp(),
            rngs=None, limits=None, ticks=None):
    (limits, ticks, lam, data) = mkplot_gen(
        lam, data, autoscale, rngs, limits, ticks)
    (zmin, zmax, zjmin, zjmax, zstep, zstepm) = data_ranges(zs, 7, 7)
    limits['zmin'] = zmin
    limits['zmax'] = zmax
    np.savetxt(pth+".csv", np.hstack((lam*1e9, data)), delimiter=',')
    template = texenv.get_template('temp.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, zs=zs))
    f.close()


def mkEzplot(pth, E, data, zs, autoscale=True, labels=default_Ezlabels(),
             rngs=None, limits=None, ticks=None):
    (limits, ticks, E, data) = mkplot_genE(
        E, data, autoscale, rngs, limits, ticks)
    (zmin, zmax, zjmin, zjmax, zstep, zstepm) = data_ranges(zs, 7, 7)
    limits['zmin'] = zmin
    limits['zmax'] = zmax
    np.savetxt(pth+".csv", np.hstack((E, data)), delimiter=',')
    template = texenv.get_template('temp.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, zs=zs))
    f.close()


def mkzdecay(pth, x, data, zs, fit, autoscale=False,
             labels={'xlabel': {'text': 'x'}, 'ylabel': {'text': 'y'}},
             rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        if (len(x.shape) > 1 and x.shape[1] > 1):
            (x, data) = range_filter_2d(x, data, rngs)
            print("True")
        else:
            (x, data) = range_filter(x, data, rngs)
            print("False")
    if (autoscale):
        data = data / np.nanmax(data, axis=0)
    if (limits is None or ticks is None):
        limits = {}
        ticks = {}
        (t_ticks, t_limits) = mkplot_axis('x', x)
        ticks.update(t_ticks)
        limits.update(t_limits)
        (ymin, ymax, lymin, lymax) = data_log_ranges(data)
        limits.update({'ymin': ymin, 'ymax': ymax})
        ticks['ytickten'] = '%i,%i,...,%i' % (lymin, lymin+1, lymax+1)
        (zmin, zmax, zjmin, zjmax, zstep, zstepm) = data_ranges(zs, 7, 7)
        limits['zmin'] = zmin
        limits['zmax'] = zmax
    print((x.shape, data.shape))
    np.savetxt(pth+".csv", np.hstack((x, data)), delimiter=',')
    template = texenv.get_template('decayPlot.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(multiplex=(x.shape[1] > 1), limits=limits,
                        ticks=ticks, labels=labels, fit=fit, zs=zs))
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
        data = data / np.nanmax(data)
    if (limits is None or ticks is None):
        limits = {}
        ticks = {}
        (t_ticks, t_limits) = mkplot_axis('x', x)
        ticks.update(t_ticks)
        limits.update(t_limits)
        (t_ticks, t_limits) = mkplot_axis('y', data)
        ticks.update(t_ticks)
        limits.update(t_limits)
    np.savetxt(pth+".csv", np.hstack((x, data)), delimiter=',')
    template = texenv.get_template('int_flux_over.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(limits=limits,
                        ticks=ticks, labels=labels, legend=legend))
    f.close()


def mkloglogplot(pth, x, data, legend, autoscale=True, labels=default_labels(),
                 rngs=None, limits=None, ticks=None):
    if (rngs is not None):
        (x, data) = range_filter(x, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data, axis=0)
    if (limits is None or ticks is None):
        (xmin, xmax, lxmin, lxmax) = data_log_ranges(x)
        (ymin, ymax, lymin, lymax) = data_log_ranges(data)
        limits = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
        ticks = {}
        ticks['xtickten'] = '%i,%i,...,%i' % (lxmin, lxmin+1, lxmax+1)
        print((lymin, lymin+1, lymax+1))
        ticks['ytickten'] = '%i,%i,...,%i' % (lymin, lymin+1, lymax+1)
    np.savetxt(pth+".csv", np.hstack((x, data)), delimiter=',')
    template = texenv.get_template('loglogplot.tex')
    f = open(pth+".tex", 'w')
    f.write(
        template.render(multiplex=(x.shape[1] > 1), limits=limits,
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
