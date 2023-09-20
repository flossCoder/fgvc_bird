#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# plot_aux.py
# Copyright (C) 2022 flossCoder
# 
# plot_aux is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# plot_aux is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Thu Sep  1 12:03:11 2022

@author: flossCoder
"""

import numpy as np
from scipy.stats import sem
from statsmodels.robust.scale import mad

def plot_mean_stderr(axes, x_val, y, color = None, marker = "s"):
    """
    This function plots the mean and std err of the mean for the given y vector.

    Parameters
    ----------
    axes : matplotlib.axes._subplots.AxesSubplot
        The axes object from matplotlib.
    x_val : int
        X position of the current errorbar.
    y : list or numpy array of floats
        The value vector.
    color : string, optional
        The color used for the current errorbar. The default is None.
    marker : string, optional
        The marker used for the errorbar. The default is "s".

    Returns
    -------
    result : dict
        The results dict contains the following entries:
            mean: The mean of y.
            std_err: The standard error of the mean.
            min: The smallest value of y.
            max: The largest value of y.

    """
    result = {"mean": np.mean(y), "std_err": sem(y), "min": np.min(y), "max": np.max(y)}
    if color is None:
        axes.errorbar(x_val, result["mean"], yerr = result["std_err"], capsize = 10, ecolor = "k", fmt = marker)
    else:
        axes.errorbar(x_val, result["mean"], yerr = result["std_err"], capsize = 10, ecolor = "k", fmt = "%s%s"%(color, marker))
    axes.plot(x_val, result["min"], "ok")
    axes.plot(x_val, result["max"], "ok")
    axes.get_xaxis().set_visible(False)
    return result

def plot_median_mad(axes, x_val, y, color = None, marker = "s"):
    """
    This function plots the median and mad for the given y vector.

    Parameters
    ----------
    axes : matplotlib.axes._subplots.AxesSubplot
        The axes object from matplotlib.
    x_val : int
        X position of the current errorbar.
    y : list or numpy array of floats
        The value vector.
    color : string, optional
        The color used for the current errorbar. The default is None.
    marker : string, optional
        The marker used for the errorbar. The default is "s".

    Returns
    -------
    result : dict
        The results dict contains the following entries:
            medain: The median of y.
            mad: The mad.
            min: The smallest value of y.
            max: The largest value of y.

    """
    result = {"median": np.median(y), "mad": mad(y), "min": np.min(y), "max": np.max(y)}
    if color is None:
        axes.errorbar(x_val, result["median"], yerr = result["mad"], capsize = 10, ecolor = "k", fmt = marker)
    else:
        axes.errorbar(x_val, result["median"], yerr = result["mad"], capsize = 10, ecolor = "k", fmt = "%s%s"%(color, marker))
    axes.plot(x_val, result["min"], "ok")
    axes.plot(x_val, result["max"], "ok")
    axes.get_xaxis().set_visible(False)
    return result

def plot_hyperparameter(axes, hp_value, hp_labels, hp_min, hp_max, marker = None, color = None, loc = None, alpha = 0.3):
    """
    This function defines the hyperparameter plot.

    Parameters
    ----------
    axes : matplotlib.axes._subplots.AxesSubplot
        The axes object from matplotlib.
    hp_value : float or list of floats
        The value of the hyperparameter (in case of float) or a list of hyperparameters.
    hp_labels : string or list of strings
        The labels of the hyperparameter(s) as string or list (len(hp_labels) must be equals len(hp_value)).
    hp_min : float
        The min value of the hyperparameter for the tuning process.
    hp_max : float
        The max value of the hyperparameter for the tuning process.
    marker : string or a list of strings, optional
        A marker string of matplotlib pyplot (for details see the matpotlib docs)
        or a list of such marker strings (len(marker) must be equals len(hp_value)).
        If the marker is None, a default internal marker will be used. The default is None.
    color : string, optional
        A color string of matplotlib pyplot (for details see the matpotlib docs). The default is None.
    loc : string
        The location of the legend (for details see the matpotlib docs). The default is None.
    alpha : float, optional
        The alpha value [0, 1] describes the opacity of the hyperparameter interval bar. The default is 0.3.

    Raises
    ------
    Exception
        The exception is raised in case of an invalid input for hp_labels or marker.

    Returns
    -------
    None.

    """
    default_marker = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]
    if color is None:
        color = ""
    if type(hp_value) == type([]):
        if type(hp_labels) != type([]) or len(hp_value) != len(hp_labels):
            raise Exception("Invalid hp_labels input")
        if type(marker) == type([]) and len(hp_value) != len(marker):
            raise Exception("Invalid marker input")
        if type(marker) is None:
            marker = default_marker
        for i in range(len(hp_value)):
            axes.plot(hp_value[i], 0, marker[i] + color, label = hp_labels[i])
    else:
        axes.plot(hp_value, 0, marker + color, label = hp_labels)
    
    axes.fill_betweenx([-0.01, 0.01], [hp_min], [hp_max], color = color, alpha = alpha)
    axes.set_ylim(-0.02, 0.02)
    axes.yaxis.set_visible(False)
    if type(loc) == type(""):
        axes.legend(loc = loc)
    else:
        axes.legend()

def plot_hyperparameter_value(axes, hp_value, hp_label = None, marker = None, color = None, alpha = 1):
    """
    This function plots an additional hyperparameter value.

    Parameters
    ----------
    axes : matplotlib.axes._subplots.AxesSubplot
        The axes object from matplotlib.
    hp_value : float
        The hyperparameter value.
    hp_label : string, optional
        The string of the hyperparameter, if it is not None. The default is None.
    marker : string, optional
        A marker string of matplotlib pyplot (for details see the matpotlib docs).
        If the marker is None, a default internal marker will be used. The default is None.
    color : string, optional
        A color string of matplotlib pyplot (for details see the matpotlib docs). The default is None.
    alpha : float, optional
        The alpha value [0, 1] describes the opacity of the hyperparameter value. The default is 1.

    Returns
    -------
    None.

    """
    if color is None:
        color = ""
    if type(marker) is None:
        marker = "."
    if hp_value is None:
        axes.plot(hp_value, 0, marker + color, alpha = alpha)
    else:
        axes.plot(hp_value, 0, marker + color, label = hp_label, alpha = alpha)
