"""
Spectrometer Data Plotter

Copyright (C) 2014--2021 Alex J. Grede
GPL v3, See LICENSE.txt for details
This module is part of specPlot
"""

import numpy as np
import scipy.constants as PC
import matplotlib.pyplot as plt
import re as regex
from numbers import Number
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


def escape_tex(value):
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
    index of the first occurrence of non-zero expr or None if all zero
    """
    return next((k for k, v in enumerate(expr) if v), None)


def elam(x):
    """
    Change wavelength to energy and reverse

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
    Change phi num/unit wavelength to num/per unit energy and reverse

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


def data_minmax(xmin, xmax, step, inner):
    """
    floor / ceiling divide depending on inner

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
    if inner:
        return (step*np.ceil(xmin/step), step*np.floor(xmax/step))
    else:
        return (step*np.floor(xmin/step), step*np.ceil(xmax/step))


def data_ranges(x, major, minor, inner=False):
    """
    Find data ranges

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
    log10 minimum and maximum values

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
        np.log10(xmin), np.log10(xmax), 1, inner)]
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
                ylabel, ysymbol, yunit, x2unit,
                zlabel=None, zsymbol=None, zunit=None):
    labels = {}
    labels.update(make_gen_label('x', xlabel, xsymbol, xunit))
    labels.update(make_gen_label('y', ylabel, ysymbol, yunit))
    if zlabel:
        labels.update(make_gen_label('z', zlabel, zsymbol, zunit))
    if x2unit:
        labels['x2label'] = {'text': "", 'units': x2unit}
    return labels


def mktick_fstr(step, single=False):
    prec = max(int(np.ceil(-np.log10(step))), 0)
    fsing = ("{:0."+str(prec)+"f}")
    if single:
        return fsing, prec
    else:
        return ",".join([s for s in [fsing]*2+["...", fsing]]), prec


def inc_arange(start, stop, step=1):
    """
    np.arange with inclusive stop

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
    Returns tuple for generating tikz style ranges
    i.e. {start,start+step,...,stop}

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
    Base 10 minor tick-marks array

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
    Make axis limits and ticks

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
                "\\num{{e{:d}}}".format(x) for x in inc_arange(*lmajors)]
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
                fstr.format(x) for x in inc_arange(*majors)]
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
               fun=None, ifun=None,
               xscale=1., yscale=1., x2scale=1.):
    """
    Filter and setup limits and ticks template arguments

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
    fun : callable, optional
        function to translate x-axis to x2 axis
    ifun : callable,  optional
        function to translate x2-axis to x-axis
    xscale : float, optional
        scale x values by this number
    yscale : float, optional
        scale y values by this number
    x2scale : float, optional
        scale x2 values by this number

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
    if fun is not None:
        (t_ticks, t_limits) = mkplot_axis(
            'x2',
            x2scale*fun(np.array([limits['xmin'], limits['xmax']])/xscale),
            ifun, log=logx, inner=True, xscale=xscale, x2scale=x2scale)
        ticks = {**t_ticks, **ticks}
    (t_ticks, t_limits) = mkplot_axis('y', data*yscale, log=logy)
    ticks = {**t_ticks, **ticks}
    limits = {**t_limits, **limits}
    return (limits, ticks, xs, data)


def mkplot(pth, xs, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=None, zsymbol=None, zunit=None,
           title=None,
           fun=None, ifun=None,
           x2unit=None,
           autoscale=False, rngs=None, limits={}, ticks={},
           logx=False, logy=False, logz=False,
           xscale=1., yscale=1., zscale=1., x2scale=1.,
           linestyle=True):
    """
    Makeplot

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
    fun : callable, optional
        function to translate x-axis to x2 axis
    ifun : callable,  optional
        function to translate x2-axis to x-axis
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
    (limits, ticks, xs, data) = mkplot_gen(
        xs, data, autoscale, rngs, limits, ticks, logx, logy, fun, ifun,
        xscale, yscale, x2scale)
    if len(xs.shape) < 2:
        xs = xs.reshape((-1, 1))
    if isinstance(legendorzs[0], Number) and zlabel is not None:
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
    else:
        args['legend'] = legendorzs
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
                                 x2unit,
                                 zlabel, zsymbol, zunit)
    if title is not None:
        args['title'] = title
    args['limits'] = limits
    args['ticks'] = ticks
    args['tickcolor'] = "black"
    args['linestyle'] = "no markers, solid" if linestyle else "only marks, mark=*"
    np.savetxt(pth+".csv", np.hstack((xs*xscale, data*yscale)), delimiter=',')
    template = texenv.get_template('plot.tex')
    f = open(pth+".tex", 'w')
    f.write(template.render(args))
    f.close()


def mklamplot(pth, lam, data, legendorzs,
              xlabel="Wavelength", xsymbol=None, xunit="\\nm",
              ylabel="Photon Flux", ysymbol=None, yunit="\\arb",
              x2unit="\\eV",
              autoscale=True,
              rngs=None, limits=None, ticks=None,
              logx=False, logy=False, logz=False,
              xscale=1e9, x2scale=1., linestyle=True):
    """
    Shortcut for wavelength data

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
        x-axis siunitx style units
    ylabel : str, optional
        y-axis label text
    ysymbol : str, optional
        y-axis label symbol (in math environment)
    yunit : str, optional
        y-axis siunitx style units
    x2unit : str, optional
        x2-axis siunitx style units
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
    linestyle : bool, optional
        use lines if true or markers if false

    """
    mkplot(pth, lam, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=None, zsymbol=None, zunit=None,
           fun=elam, ifun=elam,
           x2unit=x2unit,
           autoscale=autoscale, rngs=rngs, limits=limits, ticks=ticks,
           logx=logx, logy=logy, logz=logz,
           xscale=xscale, x2scale=x2scale, linestyle=linestyle)


def mkEplot(pth, es, data, legendorzs,
            xlabel="Photon Energy", xsymbol=None, xunit="\\eV",
            ylabel="Photon Flux", ysymbol=None, yunit="\\arb",
            x2unit="\\nm",
            autoscale=True,
            rngs=None, limits={}, ticks={},
            logx=False, logy=False, logz=False,
            xscale=1., x2scale=1e9, linestyle=True):
    """
    Shortcut for wavelength data

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
        x-axis siunitx style units
    ylabel : str, optional
        y-axis label text
    ysymbol : str, optional
        y-axis label symbol (in math environment)
    yunit : str, optional
        y-axis siunitx style units
    x2unit : str, optional
        x2-axis siunitx style units
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
    linestyle : bool, optional
        use lines if true or markers if false

    """
    mkplot(pth, es, data, legendorzs,
           xlabel, xsymbol, xunit,
           ylabel, ysymbol, yunit,
           zlabel=None, zsymbol=None, zunit=None,
           fun=elam, ifun=elam,
           x2unit=x2unit,
           autoscale=autoscale, rngs=rngs, limits=limits, ticks=ticks,
           logx=logx, logy=logy, logz=logz,
           xscale=xscale, x2scale=x2scale, linestyle=linestyle)


def mkcolorplot(pth, xs, ys, zs,
                xlabel, xsymbol, xunit,
                ylabel, ysymbol, yunit,
                zlabel, zsymbol, zunit,
                title=None,
                fun=None, ifun=None,
                x2unit=None,
                autoscale=False, limits={}, ticks={},
                logx=False, logy=False, logz=False,
                xscale=1., yscale=1., zscale=1., x2scale=1.,
                linestyle=True):
    """
    Makecolorplot

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
    fun : callable, optional
        function to translate x-axis to x2 axis
    ifun : callable,  optional
        function to translate x2-axis to x-axis
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
    if fun is not None:
        t_ticks = mkplot_axis(
            'x2', x2scale*fun(xlims/xscale),
            ifun, log=logx, inner=True, xscale=xscale, x2scale=x2scale)[0]
        ticks = {**t_ticks, **ticks}
    t_ticks = mkplot_axis('y', ylims, log=logy, inner=True)[0]
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
                                 x2unit,
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
    """Returns all the prime factors of a positive integer"""
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n /= d
        d = d + 1
    return factors
