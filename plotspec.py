"""
Spectrometer Data Plotter.

Copyright (C) 2014--2022 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import scipy.constants as PC
import matplotlib.pyplot as plt
import re as regex
from numpy.linalg import norm
from numbers import Number
from collections.abc import Iterable
from jinja2 import Environment, FileSystemLoader
import os

LATEX_SUBS = (
    (regex.compile(r'\\'), r'\\textbackslash'),
    (regex.compile(r'([{}_#%&$])'), r'\\\1'),
    (regex.compile(r'~'), r'\~{}'),
    (regex.compile(r'\^'), r'\^{}'),
    (regex.compile(r'"'), r"''"),
    (regex.compile(r'\.\.\.+'), r'\\ldots'),
)

UNIT_SCALES: {
    'yocto': -24,
    'zepto': -21,
    'atto': -18,
    'femto': -15,
    'pico': -12,
    'nano': -9,
    'micro': -6,
    'milli': -3,
    'centi': -2,
    'deci': -1,
    'deca': 1,
    'hecto': 2,
    'kilo': 3,
    'mega': 6,
    'giga': 9,
    'tera': 12,
    'peta': 15,
    'exa': 18,
    'zetta': 21,
    'yotta': 24}

ANCHORN = [
    "east", "north east", "north",
    "north west", "west", "south west",
    "south", "south east"]
ANCHORS = np.array([
    [1., 0], [1, 1], [0, 1],
    [-1, 1], [-1, 0], [-1, -1],
    [0, -1], [1, -1]])
ANCHORS /= norm(ANCHORS, axis=1, keepdims=True)


def escape_tex(value):
    """
    Jinja filter for escaping in TeX documents.

    Parameters
    ----------
    value : str
        unsafe string.

    Returns
    -------
    newval : str
        safe string.

    """
    newval = value
    for pattern, replacement in LATEX_SUBS:
        newval = pattern.sub(replacement, newval)
    return newval


texenv = Environment(
    autoescape=False,
    loader=FileSystemLoader(
        os.path.join(os.path.dirname(__file__), "templates")))
texenv.block_start_string = '((*'
texenv.block_end_string = '*))'
texenv.variable_start_string = '((('
texenv.variable_end_string = ')))'
texenv.comment_start_string = '((='
texenv.comment_end_string = '=))'
texenv.filters['escape_tex'] = escape_tex
texenv.lstrip_blocks = True
texenv.trim_blocks = True


def first_where(expr):
    """
    Return the index of the first occurrence of non-zero expr or None if all zero.
    """
    return next((k for k, v in enumerate(expr) if v), None)


def elam(x):
    """
    Change wavelength to energy and reverse.

    Parameters
    ----------
    x : {float, array_like}
        wavelength in m or energy in eV

    Returns
    -------
    {float, array_like} : wavelength<->eV
    """
    return PC.h * PC.c / (PC.e * x)


def phie(x, phi):
    """
    Change phi num/unit wavelength to num/per unit energy and reverse.

    Parameters
    ----------
    x : array_like
        wavelength in m or energy in eV
    phi : num / unit x

    Returns
    -------
    numpy.ndarray : Change of units version of phie
    """
    return (np.atleast_2d(x/elam(x)).T*phi)


def signed_ceil(x):
    """
    Signed ceiling function (complement to trunc).

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    numpy.ndarray or scalar
        sign(x)*ceil(abs(x)).
    """
    return np.sign(x)*np.ceil(np.abs(x))


def data_minmax(xmin, xmax, step, inner):
    """
    Return floor / ceiling division depending on inner.

    Parameters
    ----------
    xmin : float
        minimum numerator
    xmax : float
        maximum numerator
    step : float
        divisor
    inner : bool
        floor / ceiling swap

    Returns
    -------
    min : float
        nearest step size away from xmin (ceiling for inner)
    max : float
        nearest step size away from xmax (floor for inner)
    """
    eps = np.finfo(type(step)).resolution
    xs = np.array([xmin, xmax])
    ys = np.zeros(2)
    xmid = xs.mean()
    tmid = step*np.trunc(xmid/(step-eps))
    if inner:
        if (xmax-xmin)/(step-eps)-1. < -eps:
            for xtst in np.arange(-1, 2)*step+tmid:
                if xmin <= xtst and xtst <= xmax:
                    return (xtst, xtst)
        else:
            ys = step*np.trunc((xs-tmid)/(step-eps))+tmid
    else:
        sngs = np.array([-1, 1])
        ys = step*signed_ceil((xs-xmid)/(step-eps))+tmid
        ys += sngs*step*(np.abs(ys-xs)-step < -eps)
    return tuple(ys)


def data_ranges(x, major, minor, inner=False):
    """
    Find data ranges.

    Parameters
    ----------
    x : numpy.ndarray
        array of data
    major : int
        maximum number of major ticks
    minor : int
        maximum number of minor ticks between majors
    inner : bool
        all ticks will be within data range

    Returns
    -------
    minormin : float
        minor min
    minormax : float
        minor max
    minorstep : float
         minor step
    majormin : float
        major min
    majormax : float
         major max
    majorstep : float
         major step
    """
    xmax = np.nanmax(x)
    xmin = np.nanmin(x)
    xrng = xmax - xmin
    step_range = np.array([0.1, 0.2, 0.5, 1])
    step_digits = np.array([-1, -1, -1, 0])
    m = int(np.ceil(np.log10(xrng/major)))
    M = 10.**m
    k = first_where(xrng / (M * step_range) <= major)
    majorstep = M * step_range[k]
    step_digit = m + step_digits[k]
    majormin, majormax = data_minmax(xmin, xmax, majorstep, inner)
    minormin = majormin
    minormax = majormax
    minorstep = None
    Mm = 10.**step_digit
    ksm = first_where(majorstep / (Mm * step_range) <= minor)
    if (minor > 0 and ksm is not None):
        minorstep = Mm * step_range[ksm]
        minormin, minormax = data_minmax(xmin, xmax, minorstep, inner)
        if not inner:
            if xmax <= majormax-minorstep:
                majormax -= majorstep
            if xmin >= majormin+minorstep:
                majormin += majorstep
    return (minormin, minormax, minorstep), (majormin, majormax, majorstep)


def data_log_ranges(xs, inner=False):
    """
    log10 minimum and maximum values.

    Parameters
    ----------
    xs : numpy.ndarray
        data for ranging
    inner : bool
        stay within x bounds (default: false)

    Returns
    -------
    minors : tuple(float, float)
        minor min, max
    lmajors : tuple(int)
        log10 major min, max
    """
    xmax = np.nanmax(xs)
    xmin = np.nanmin(xs)
    lmajors = [int(t) for t in data_minmax(
        np.log10(xmin), np.log10(xmax), 1., inner)]
    majors = [10**t for t in lmajors]
    minorsteps = [10**(t0+t1+1*inner) for t0, t1 in zip(lmajors, [0, -1])]
    minors = [t0*t1 for t0, t1 in zip(
        data_minmax(xmin/minorsteps[0], xmax/minorsteps[1], 1.0, inner),
        minorsteps)]
    if not inner:
        if minors[0]/majors[0] < 2:
            lmajors[0] -= 1
            minors[0] = 0.8*majors[0]
        else:
            minors[0] -= majors[0]
        if minors[1]/majors[1] > 0.8:
            lmajors[1] += 1
            minors[1] = 2*majors[1]
        else:
            minors[1] += 0.1*majors[1]
    return minors, lmajors


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


def make_gen_label(pfx, label, symbol, unit):
    rtn = {}
    if label:
        lbl = {'text': label}
        if symbol:
            lbl['symbol'] = symbol
        if unit:
            lbl['units'] = unit
        rtn[pfx+'label'] = lbl
    return rtn


def make_labels(xlabel, xsymbol, xunit,
                ylabel, ysymbol, yunit, x2unit=None, y2unit=None,
                zlabel=None, zsymbol=None, zunit=None):
    labels = {}
    labels.update(make_gen_label('x', xlabel, xsymbol, xunit))
    labels.update(make_gen_label('y', ylabel, ysymbol, yunit))
    if zlabel:
        labels.update(make_gen_label('z', zlabel, zsymbol, zunit))
    if x2unit:
        labels['x2label'] = {'text': "", 'units': x2unit}
    if y2unit:
        labels['y2label'] = {'text': "", 'units': y2unit}
    return labels


def mktick_fstr(step, single=False):
    """
    Make formating strings for tickmarks.

    Parameters
    ----------
    step : float
        step size.
    single : bool, optional
        make single (True) or range (False). The default is False.

    Returns
    -------
    str
        formatting string.
    prec : int
        number of decimal places.

    """
    prec = max(int(np.ceil(-np.log10(step))), 0)
    fsing = ("{:0."+str(prec)+"f}")
    if single:
        return fsing, prec
    else:
        return ",".join([s for s in [fsing]*2+["...", fsing]]), prec


def inc_arange(start, stop, step=1):
    """
    Evenly spaced data with inclusive stop.

    Wrapper around np.arange

    Parameters
    ----------
    start : Number
    stop : Number
    step : Number

    Returns
    -------
    numpy.ndarray
    """
    if type(step) is int:
        return np.arange(start, stop+step, step)
    else:
        return np.arange(start, stop+0.05*step, step)


def pgfplots_inc_arange(start, stop, step):
    """
    Return tuple for generating tikz style ranges i.e. {start,start+step,...,stop}.

    Parameters
    ----------
    start : Number
    stop : Number
    step : Number

    Returns
    -------
    tuple(Number, Number, Number)
    """
    return (start, start+step, stop+step)


def minor_log_ticks(minors, lmajors):
    """
    Make base 10 minor tick-marks array.

    Parameters
    ----------
    minors : tuple(float, float)
        min, max minor ticks
    lmajors
        min, max log10 major ticks

    Returns
    -------
    list (float)
    """
    powers = np.power(
        10., inc_arange(lmajors[0], lmajors[1], 1)).reshape((-1, 1))
    mticks = np.arange(1, 10).reshape((1, -1))
    return [x for x in (powers*mticks).ravel()
            if x >= minors[0] and x <= minors[1]]


def mkplot_axis(
        prefix, xs, ifun=None, log=False, inner=False, major=7, minor=7,
        xscale=1., x2scale=1., allticks=False):
    """
    Make axis limits and ticks.

    Parameters
    ----------
    prefix : str
        axis prefix
    xs : numpy.ndarray
        data to build axis for
    ifun : callable, optional
        Function to convert labels values to tick values
    log : bool, optional
        log scale values
    inner : bool, optional
        axis marks should be within xs values
    major : int, optional
        maximum number of major ticks
    minor : int, optional
        maximum number of minor ticks between majors
    xscale : float, optional
        scale x values by this number
    x2scale : float, optional
        scale x2 values by this number
    allticks : bool, optional
        return all ticks regardless of ifun

    Returns
    -------
    ticks : dict
        partial dictionary for template ticks argument for given axis prefix
    limits : dict
        partial dictionary for template limits argument for given axis prefix
    """
    ticks = {}
    limits = {}
    if log:
        minors, lmajors = data_log_ranges(xs, inner)
        if ifun or allticks:
            if ifun:
                ticks[prefix+'major'] = [
                    "{:f}".format(ifun(x/x2scale)*xscale)
                    for x in np.power(10., inc_arange(*lmajors))]
                ticks[prefix+'minor'] = [
                    "{:f}".format(ifun(x/x2scale)*xscale)
                    for x in minor_log_ticks(minors, lmajors)]
            else:
                ticks[prefix+'major'] = [
                    "{:f}".format(x)
                    for x in np.power(10., inc_arange(*lmajors))]
                ticks[prefix+'minor'] = [
                    "{:f}".format(x)
                    for x in minor_log_ticks(minors, lmajors)]
            ticks[prefix+'labels'] = [
                "e{:d}".format(x) for x in inc_arange(*lmajors)]
        else:
            ticks[prefix+'log'] = mktick_fstr(1)[0].format(
                *pgfplots_inc_arange(lmajors[0], lmajors[1], 1))
    else:
        minors, majors = data_ranges(xs, major, minor, inner)
        fstr, prec = mktick_fstr(majors[2], ifun or allticks)
        if ifun or allticks:
            if ifun:
                ticks[prefix+'major'] = [
                    "{:f}".format(ifun(x/x2scale)*xscale)
                    for x in inc_arange(*majors)]
                if minors[2] is not None:
                    ticks[prefix+'minor'] = [
                        "{:f}".format(ifun(x/x2scale)*xscale)
                        for x in inc_arange(*minors)]
            else:
                ticks[prefix+'major'] = [
                    "{:f}".format(x)
                    for x in inc_arange(*majors)]
                if minors[2] is not None:
                    ticks[prefix+'minor'] = [
                        "{:f}".format(x)
                        for x in inc_arange(*minors)]
            ticks[prefix+'labels'] = [
                fstr.format(np.round(x, prec)+0) for x in inc_arange(*majors)]
        else:
            ticks[prefix+'major'] = fstr.format(*pgfplots_inc_arange(*majors))
            if minors[2] is not None:
                fstrm = mktick_fstr(minors[2])[0]
                ticks[prefix+'minor'] = fstrm.format(
                    *pgfplots_inc_arange(*minors))
            ticks[prefix+'labels'] = prec
    limits = {prefix+'min': minors[0], prefix+'max': minors[1]}
    return (ticks, limits)


def mkplot_gen(xs, data, autoscale, rngs, limits, ticks,
               logx=False, logy=False,
               xfun=None, xifun=None, yfun=None, yifun=None,
               xscale=1., yscale=1., x2scale=1., y2scale=1.):
    """
    Filter and setup limits and ticks template arguments.

    Parameters
    ----------
    xs : {(N,), (N,M)} array_like
        x axis values
    data : (N,M) array_like
        y for xs points
    autoscale : bool
        divide y values by maximum value
    rngs : dict
        filter out data given {'x': [min, max], 'y': [min, max]}
    limits : dict
        existing limits template argument not to update without overwrite
    ticks : dict
        existing ticks template argument not to update without overwrite
    logx : bool, optional
        logscale x values
    logy : bool, optional
        logscale y values
    xfun : callable, optional
        function to translate x-axis to x2 axis
    xifun : callable,  optional
        function to translate x2-axis to x-axis
    yfun : callable, optional
        function to translate y-axis to y2 axis
    yifun : callable,  optional
        function to translate y2-axis to y-axis
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    x2scale : float, optional
        scale x2 values by this number
    y2scale : float, optional
        scale y2 values by this number

    Returns
    -------
    limits : dict
        updated limits template argument without overwrite
    ticks : dict
        updated ticks template argument without overwrite
    xs : array_like
        filtered xs values (nans or removes rows with all nans)
    data : array_like
        filtered data (nans or removes rows with all nans)
    """
    if (rngs is not None):
        (xs, data) = range_filter(xs, data, rngs)
    if (autoscale):
        data = data / np.nanmax(data)
    (t_ticks, t_limits) = mkplot_axis('x', xs*xscale, log=logx)
    ticks = {**t_ticks, **ticks}
    limits = {**t_limits, **limits}
    if xfun is not None:
        (t_ticks, t_limits) = mkplot_axis(
            'x2',
            x2scale*xfun(np.array([limits['xmin'], limits['xmax']])/xscale),
            xifun, log=logx, inner=True, xscale=xscale, x2scale=x2scale)
        ticks = {**t_ticks, **ticks}
    if 'ymin' in limits and 'ymax' in limits:
        (t_ticks, t_limits) = mkplot_axis(
            'y', np.array([limits['ymin'], limits['ymax']]), log=logy)
    else:
        (t_ticks, t_limits) = mkplot_axis('y', data*yscale, log=logy)
    ticks = {**t_ticks, **ticks}
    limits = {**t_limits, **limits}
    if yfun is not None:
        (t_ticks, t_limits) = mkplot_axis(
            'y2',
            y2scale*yfun(np.array([limits['ymin'], limits['ymax']])/yscale),
            yifun, log=logy, inner=True, xscale=yscale, x2scale=y2scale)
        ticks = {**t_ticks, **ticks}
    return (limits, ticks, xs, data)


def mkplot(pth, xs, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=None, zsymbol=None, zunit=None,
           title=None,
           xfun=None, xifun=None, x2unit=None,
           yfun=None, yifun=None, y2unit=None,
           autoscale=False, rngs=None, limits={}, ticks={},
           logx=False, logy=False, logz=False,
           xscale=1., yscale=1., zscale=1.,
           x2scale=1., y2scale=1.,
           linestyle=True, marker="*", additional="",
           thickstyle=True, x2shift=0.0, useLegend=None, useNodes=None):
    """
    Makeplot.

    Parameters
    ----------
    pth : str
        path without file extensions to write to
    xs : {(N,), (N,M)} array_like
        x axis values
    data : (N,M) array_like
        y for xs points
    legendorzs : (M,) array_like
        legend (node text) for each addplot or if numeric creates meshplot type
    xlabel : str
        x-axis label text
    xsymbol : str
        x-axis label symbol (in math environment)
    xunit : str
        x-axis siunitx style units
    ylabel : str
        y-axis label text
    ysymbol : str
        y-axis label symbol (in math environment)
    yunit : str
        y-axis siunitx style units
    zlabel : str, optional
        z-axis label text
    zsymbol : str, optional
        z-axis label symbol (in math environment)
    zunit : str, optional
        z-axis siunitx style units
    title : str, optional
        Title for plot
    xfun : callable, optional
        function to translate x-axis to x2 axis
    xifun : callable,  optional
        function to translate x2-axis to x-axis
    x2unit : str, optional
        upper x unit
    yfun : callable, optional
        function to translate y-axis to y2 axis
    yifun : callable,  optional
        function to translate y2-axis to y-axis
    y2unit : str, optional
        upper y unit
    autoscale : bool, optional
        divide y values by maximum value
    rngs : dict, optional
        filter out data given {'x': [min, max], 'y': [min, max]}
    limits : dict, optional
        existing limits template argument not to update without overwrite
    ticks : dict, optional
        existing ticks template argument not to update without overwrite
    logx : bool, optional
        logscale x values
    logy : bool, optional
        logscale y values
    logz : bool, optional
        logscale z values (actually changes data appropriately)
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    zscale : float, optional
        scale y values by this number
    x2scale : float, optional
        scale x2 values by this number
    y2scale : float, optional
        scale y2 values by this number
    linestyle : bool, optional
        use lines if true or markers if false
    marker : str, optional
        marker to use if linestyle=false
    additional: str, optional
        additional string that will be added to the end
    thickstyle: bool, optional
        changes style from "thick" if true (default) to "thin, mark size=1.2pt"
    x2shift: float, optional
        adds yshift equal to x2shift em (for decenders)
    """
    args = {}
    (limits, ticks, xs, data) = mkplot_gen(
        xs, data, autoscale, rngs, limits, ticks, logx, logy, xfun, xifun,
        yfun, yifun, xscale, yscale, x2scale, y2scale)
    args['linestyle'] = "no markers, solid" if linestyle else "only marks, mark="+marker
    if len(xs.shape) < 2:
        xs = xs.reshape((-1, 1))
    colorLines = isinstance(legendorzs[0], Number) and zlabel is not None
    if colorLines:
        zs = np.atleast_1d(legendorzs)
        (t_ticks, t_limits) = mkplot_axis('z', zs*zscale, log=logz)
        ticks = {**t_ticks, **ticks}
        limits = {**t_limits, **limits}
        if logz:
            limits['zmin'] = np.log10(limits['zmin'])
            limits['zmax'] = np.log10(limits['zmax'])
            args['zs'] = np.log10(zs*zscale)
        else:
            args['zs'] = zs
        if not linestyle:
            args['linestyle'] = (
                "scatter, only marks, mark=" + marker +
                ", scatter/use mapped color=" +
                "{draw=mapped color, fill=mapped color}")
    else:
        args['legend'] = legendorzs
    plotlabels = {
        "show_legend": False,
        "show_nodes": False,
        "pos_legend": "north west",
        "pos_nodes": [0.5]*len(legendorzs),
        "anch_nodes": ["east"]*len(legendorzs)}
    findnodes = False
    findnodeanchors = False
    findlegend = False
    if colorLines and not useLegend:
        pass
    elif isinstance(useLegend, str):
        plotlabels['pos_legend'] = useLegend
        plotlabels['show_legend'] = True
    elif useLegend:
        plotlabels['show_legend'] = True
        findlegend = True
    elif useNodes:
        plotlabels['show_nodes'] = True
        if isinstance(useNodes, dict):
            plotlabels['pos_nodes'] = useNodes['pos_nodes']
            plotlabels['anch_nodes'] = useNodes['anch_nodes']
        elif isinstance(useNodes, Iterable) and isinstance(useNodes[0], Number):
            plotlabels['pos_nodes'] = useNodes
            findnodeanchors = True
        else:
            findnodes = True
            findnodeanchors = True
    elif useNodes is None:
        findnodes = True
        findnodeanchors = True
        if useLegend is None:
            findlegend = True
        else:
            plotlabels['show_nodes'] = True
    args['multiplex'] = (xs.shape[0] < xs.size)
    if findnodes or findnodeanchors or findlegend:
        xns = normalize_points(
            xs*xscale,
            limits['xmin'], limits['xmax'])
        if not args['multiplex']:
            xns = xns*np.ones((1, data.shape[1]))
        yns = normalize_points(data*yscale, limits['ymin'], limits['ymax'])
        ps = np.dstack((xns, yns)).swapaxes(1, 2)
        if findnodes or findnodeanchors:
            N = ps.shape[0]
            M = ps.shape[2]
            ds = np.zeros(M)
            for m in range(M):
                if findnodes:
                    n, d = furthest_point(
                        ps[:, :, m], ps[:, :, np.arange(M) != m], bnd=0.1)
                    plotlabels['pos_nodes'][m] = n/N
                    ds[m] = d
                else:
                    n = int(plotlabels['pos_nodes'][m]*N)
                if findnodeanchors:
                    n2 = furthest_point(ps[n, :, m].reshape((1, 2))+ANCHORS*0.01, ps)[0]
                    plotlabels['anch_nodes'][m] = np.roll(ANCHORN, 4)[n2]
            if findlegend and ds.min() > 0.05:
                findlegend = False
                plotlabels['show_nodes'] = True
            elif findlegend:
                plotlabels['show_legend'] = True
        if findlegend:
            n = quadrant_counts(ps).argmin()
            plotlabels['pos_legend'] = ANCHORN[1::2][n]
    args['plotlabels'] = plotlabels
    args['axistype'] = "axis"
    if logx and logy:
        args['axistype'] = "loglogaxis"
    elif logx:
        args['axistype'] = "semilogxaxis"
    elif logy:
        args['axistype'] = "semilogyaxis"
    args['multiplex'] = (xs.shape[0] < xs.size)
    args['labels'] = make_labels(xlabel, xsymbol, xunit,
                                 ylabel, ysymbol, yunit,
                                 x2unit, y2unit,
                                 zlabel, zsymbol, zunit)
    if (
            logx and x2unit is not None and 'x2labels' in ticks and
            isinstance(ticks['x2labels'], Iterable)):
        for k, x in enumerate(ticks['x2labels'][:-1]):
            ticks['x2labels'][k] = "\\num{{{:s}}}".format(x)
    if (
            logy and y2unit is not None and 'y2labels' in ticks and
            isinstance(ticks['y2labels'], Iterable)):
        for k, x in enumerate(ticks['y2labels'][:-1]):
            ticks['y2labels'][k] = "\\num{{{:s}}}".format(x)
    if title is not None:
        args['title'] = title
    args['limits'] = limits
    args['ticks'] = ticks
    args['tickcolor'] = "black"
    args['additional'] = additional
    args['thickstyle'] = thickstyle
    args['x2shift'] = x2shift
    np.savetxt(pth+".csv", np.hstack((xs*xscale, data*yscale)), delimiter=',')
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(template.render(args))
    f.close()


def mklamplot(pth, lam, data, legendorzs,
              xlabel="Wavelength", xsymbol=None, xunit=None,
              ylabel="Photon Flux", ysymbol=None, yunit="\\arb",
              zlabel=None, zsymbol=None, zunit=None,
              x2unit=None,
              autoscale=True, title=None,
              rngs=None, limits=None, ticks=None,
              logx=False, logy=False, logz=False,
              xscale=1e9, x2scale=1., linestyle=True,
              marker="*", additional="",
              thickstyle=True, x2shift=0.0, useLegend=False, useNodes=True):
    """
    Shortcut for wavelength data.

    pth : str
        path without file extensions to write to
    lam : {(N,), (N,M)} array_like
        wavelenth axis values
    data : (N,M) array_like
        y for lam points
    legendorzs : (M,) array_like
        legend (node text) for each addplot or if numeric creates meshplot type
    xlabel : str, optional
        x-axis label text
    xsymbol : str, optional
        x-axis label symbol (in math environment)
    xunit : str, optional
        x-axis siunitx style units (will use nm if xscale=1e9, um if 1e6)
    ylabel : str, optional
        y-axis label text
    ysymbol : str, optional
        y-axis label symbol (in math environment)
    yunit : str, optional
        y-axis siunitx style units
    x2unit : str, optional
        x2-axis siunitx style units (will use eV if x2scale=1, meV if x2scale=1e3)
    autoscale : bool, optional
        divide y values by maximum value
    title : str, optional
        title for plot
    rngs : dict, optional
        filter out data given {'x': [min, max], 'y': [min, max]}
    limits : dict, optional
        existing limits template argument not to update without overwrite
    ticks : dict, optional
        existing ticks template argument not to update without overwrite
    logx : bool, optional
        logscale x values
    logy : bool, optional
        logscale y values
    logz : bool, optional
        logscale z values (actually changes data appropriately)
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    zscale : float, optional
        scale y values by this number
    linestyle : bool, optional
        use lines if true or markers if false
    marker : str, optional
        marker to use if linestyle=false
    additional: str, optional
        additional string that will be added to the end
    thickstyle: bool, optional
        changes style from "thick" if true (default) to "thin, mark size=1.2pt"
    x2shift: float, optional
        adds yshift equal to x2shift em (for decenders)

    """
    if xunit is None:
        xunit = "\\um" if xscale==1e6 else "\\nm"
    if x2unit is None:
        x2unit = "\\meV" if x2scale==1e3 else "\\eV"
    mkplot(pth, lam, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=zlabel, zsymbol=zsymbol, zunit=zunit,
           xfun=elam, xifun=elam,
           x2unit=x2unit, title=title,
           autoscale=autoscale, rngs=rngs, limits=limits, ticks=ticks,
           logx=logx, logy=logy, logz=logz,
           xscale=xscale, x2scale=x2scale, linestyle=linestyle,
           marker=marker, additional=additional,
           thickstyle=thickstyle, x2shift=x2shift, useLegend=useLegend,
           useNodes=useNodes)


def mkEplot(pth, es, data, legendorzs,
            xlabel="Photon Energy", xsymbol=None, xunit=None,
            ylabel="Photon Flux", ysymbol=None, yunit="\\arb",
            zlabel=None, zsymbol=None, zunit=None,
            x2unit=None,
            autoscale=True, title=None,
            rngs=None, limits={}, ticks={},
            logx=False, logy=False, logz=False,
            xscale=1., x2scale=1e9, linestyle=True,
            marker="*", additional="",
            thickstyle=True, x2shift=None, useLegend=False, useNodes=True):
    """
    Shortcut for energy data.

    pth : str
        path without file extensions to write to
    es : {(N,), (N,M)} array_like
        energy data in eV
    data : (N,M) array_like
        y for lam points
    legendorzs : (M,) array_like
        legend (node text) for each addplot or if numeric creates meshplot type
    xlabel : str, optional
        x-axis label text
    xsymbol : str, optional
        x-axis label symbol (in math environment)
    xunit : str, optional
        x-axis siunitx style units (will use eV if xscale=1, meV if x2scale=1e3)
    ylabel : str, optional
        y-axis label text
    ysymbol : str, optional
        y-axis label symbol (in math environment)
    yunit : str, optional
        y-axis siunitx style units
    x2unit : str, optional
        x2-axis siunitx style units (will use nm if x2scale=1e9, um if 1e6)
    autoscale : bool, optional
        divide y values by maximum value
    title : str, optional
        title for plot
    rngs : dict, optional
        filter out data given {'x': [min, max], 'y': [min, max]}
    limits : dict, optional
        existing limits template argument not to update without overwrite
    ticks : dict, optional
        existing ticks template argument not to update without overwrite
    logx : bool, optional
        logscale x values
    logy : bool, optional
        logscale y values
    logz : bool, optional
        logscale z values (actually changes data appropriately)
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    zscale : float, optional
        scale y values by this number
    linestyle : bool, optional
        use lines if true or markers if false
    marker : str, optional
        marker to use if linestyle=false
    additional: str, optional
        additional string that will be added to the end
    thickstyle: bool, optional
        changes style from "thick" if true (default) to "thin, mark size=1.2pt"
    x2shift: float, optional
        adds yshift equal to x2shift em (for decenders)

    """
    if xunit is None:
        xunit = "\\meV" if xscale==1e3 else "\\eV"
    if x2unit is None:
        x2unit = "\\um" if x2scale==1e6 else "\\nm"
    if x2shift is None:
        if x2scale == 1e6:
            x2shift=-0.5
    mkplot(pth, es, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=zlabel, zsymbol=zsymbol, zunit=zunit,
           xfun=elam, xifun=elam,
           x2unit=x2unit, title=title,
           autoscale=autoscale, rngs=rngs, limits=limits, ticks=ticks,
           logx=logx, logy=logy, logz=logz,
           xscale=xscale, x2scale=x2scale, linestyle=linestyle,
           marker=marker, additional=additional,
           thickstyle=thickstyle, x2shift=x2shift, useLegend=useLegend, useNodes=useNodes)


def mkcolorplot(pth, xs, ys, zs,
                xlabel, xsymbol, xunit,
                ylabel, ysymbol, yunit,
                zlabel, zsymbol, zunit,
                title=None,
                xfun=None, xifun=None, x2unit=None,
                yfun=None, yifun=None, y2unit=None,
                autoscale=False, limits={}, ticks={},
                logx=False, logy=False, logz=False,
                xscale=1., yscale=1., zscale=1.,
                x2scale=1., y2scale=1.,
                linestyle=True):
    """
    Makecolorplot.

    Parameters
    ----------
    pth : str
        path without file extensions to write to
    xs : {(N,), (N,M)} array_like
        x axis values
    data : (N,M) array_like
        y for xs points
    legendorzs : (M,) array_like
        legend (node text) for each addplot or if numeric creates meshplot type
    xlabel : str
        x-axis label text
    xsymbol : str
        x-axis label symbol (in math environment)
    xunit : str
        x-axis siunitx style units
    ylabel : str
        y-axis label text
    ysymbol : str
        y-axis label symbol (in math environment)
    yunit : str
        y-axis siunitx style units
    zlabel : str, optional
        z-axis label text
    zsymbol : str, optional
        z-axis label symbol (in math environment)
    zunit : str, optional
        z-axis siunitx style units
    title : str, optional
        title for plot
    xfun : callable, optional
        function to translate x-axis to x2 axis
    xifun : callable,  optional
        function to translate x2-axis to x-axis
    x2unit : str
        x2-axis siunitx style units
    yfun : callable, optional
        function to translate y-axis to y2 axis
    yifun : callable,  optional
        function to translate y2-axis to y-axis
    y2unit : str
        y2-axis siunitx style units
    autoscale : bool, optional
        divide y values by maximum value
    rngs : dict, optional
        filter out data given {'x': [min, max], 'y': [min, max]}
    limits : dict, optional
        existing limits template argument not to update without overwrite
    ticks : dict, optional
        existing ticks template argument not to update without overwrite
    logx : bool, optional
        logscale x values
    logy : bool, optional
        logscale y values
    logz : bool, optional
        logscale z values (actually changes data appropriately)
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    zscale : float, optional
        scale y values by this number
    x2scale : float, optional
        scale x2 values by this number
    linestyle : bool, optional
        use lines if true or markers if false

    """
    args = {}

    t_limits = {'xmin': np.nanmin(xs*xscale), 'xmax': np.nanmax(xs*xscale),
                'ymin': np.nanmin(ys*yscale), 'ymax': np.nanmax(ys*yscale)}
    limits = {**t_limits, **limits}
    xlims = np.array([limits['xmin'], limits['xmax']])
    ylims = np.array([limits['ymin'], limits['ymax']])
    if logz:
        t_ticks, t_limits = mkplot_axis(
            'z', zs*zscale, log=True, ifun=lambda x: np.log10(x))
    else:
        t_ticks, t_limits = mkplot_axis('z', zs*zscale, log=False, minor=0)
    ticks = {**t_ticks, **ticks}
    limits = {**t_limits, **limits}
    if logz:
        limits['zmin'] = np.log10(limits['zmin'])
        limits['zmax'] = np.log10(limits['zmax'])

    t_ticks = mkplot_axis('x', xlims, log=logx, inner=True)[0]
    ticks = {**t_ticks, **ticks}
    if xfun is not None:
        t_ticks = mkplot_axis(
            'x2', x2scale*xfun(xlims/xscale),
            xifun, log=logx, inner=True, xscale=xscale, x2scale=x2scale)[0]
        ticks = {**t_ticks, **ticks}
    t_ticks = mkplot_axis('y', ylims, log=logy, inner=True)[0]
    ticks = {**t_ticks, **ticks}
    if yfun is not None:
        t_ticks = mkplot_axis(
            'y2', y2scale*yfun(ylims/yscale),
            yifun, log=logy, inner=True, xscale=yscale, x2scale=y2scale)[0]
        ticks = {**t_ticks, **ticks}
    args['axistype'] = "axis"
    if logx and logy:
        args['axistype'] = "loglogaxis"
    elif logx:
        args['axistype'] = "semilogxaxis"
    elif logy:
        args['axistype'] = "semilogyaxis"
    args['labels'] = make_labels(xlabel, xsymbol, xunit,
                                 ylabel, ysymbol, yunit,
                                 x2unit, y2unit,
                                 zlabel, zsymbol, zunit)
    if title is not None:
        args['title'] = title
    args['limits'] = limits
    args['ticks'] = ticks
    args['tickcolor'] = "white"
    args['image'] = True
    plt.imsave(pth+".png", zs, vmin=limits['zmin'], vmax=limits['zmax'],
               origin='lower', cmap='viridis')
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(template.render(args))
    f.close()


def prime_factors(n):
    """
    Return all prime factors of a positive integer.

    Parameters
    ----------
    n : int
        integer to factor.

    Returns
    -------
    factors : list
        prime factors of n.

    """
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
    return factors


def normalize_points(xs, xmin, xmax):
    """
    Normalize points between 0 and 1.

    Parameters
    ----------
    xs : numpy.ndarray
        values to normalize.
    xmin : float
        minimum value.
    xmax : float
        maximum value.

    Returns
    -------
    numpy.ndarray
        normalized xs coordinates.

    """
    return (xs-xmin)/(xmax-xmin)


def furthest_point(ps, qs, bnd=None):
    """
    Find the point in ps that is furthest away from the closest point in qs.

    Parameters
    ----------
    ps : numpy.ndarray (N,2)
        Normalized (x,y) coordinates.
    qs : numpy.ndarray (N, 2, M)
        Normalized (x,y) coordinates.

    Returns
    -------
    kmx : int
        index of ps furthest away from all qs.
    dmx : float
        distance away.

    """
    kmx = 0
    dmx = 0.0
    for k, p0 in enumerate(ps):
        if not np.all(np.isfinite(p0)):
            continue
        dbnd = 1.0
        if bnd is not None:
            dbnd = (
                (p0 > (1.-bnd))/(1.+bnd-p0) +
                (p0 < bnd)/(bnd+p0)).sum()+1.0
        d = np.nanmin(norm(qs-p0.reshape((1, 2, 1)), axis=1))/dbnd
        if d > dmx:
            kmx = k
            dmx = d
    return kmx, dmx


def quadrant_counts(ps):
    """
    Find the number of points in each quadrant.

    Parameters
    ----------
    ps : numpy.ndarray (N,2,M)
        normalized (x, y) coordinates.

    Returns
    -------
    ns : numpy.ndarray(4,)
        number of points in ps that fall in each quadrant.

    """
    ms = np.atleast_3d([[1, 1],[-1, 1],[-1, -1],[1, -1]])
    ns = np.zeros(4, dtype=int)
    for k, m in enumerate(ms):
        ns[k] = np.nansum(((ps-0.5)*m > 0).prod(1))
    return ns


def unequal_array(xs):
    """
    Stack unequal length arrays with nans for values not specified.

    Parameters
    ----------
    xs : list (M)
        list of lists (max length N).

    Returns
    -------
    ys : numpy.ndarray (N,M)
        xs stacked.

    """
    N = max([len(x) for x in xs])
    M = len(xs)
    ys = np.nan*np.ones((N, M))
    for k, x in enumerate(xs):
        n = len(x)
        ys[:n, k] = x
    return ys