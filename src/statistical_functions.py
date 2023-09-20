#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# statistical_functions.py
# Copyright (C) 2022 flossCoder
# 
# statistical_functions is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# statistical_functions is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Feb 14 09:04:46 2022

@author: flossCoder
"""

import numpy as np

def check_epsilon(y, y_hat, epsilon = 0.01):
    """
    This function tests, if the deviation of y and y_hat is less than epsilon.

    Parameters
    ----------
    y : float
        y denotes the ground trouth.
    y_hat : float
        y_hat denotes the estimated value.
    epsilon : float, optional
        epsilon denotes the upper bound for the deviation of y and y_hat. The default is 0.01.

    Returns
    -------
    boolean
        Returns True, if the deviation of y and y_hat is less than epsilon and False otherwise.

    """
    return np.abs((y-y_hat)/y) < epsilon
