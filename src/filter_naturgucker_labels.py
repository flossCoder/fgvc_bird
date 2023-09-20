#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# filter_naturgucker_labels.py
# Copyright (C) 2022 flossCoder
# 
# filter_naturgucker_labels is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# filter_naturgucker_labels is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Jan 14 10:45:27 2022

@author: flossCoder
"""

import numpy as np
from copy import deepcopy

def filter_data(data):
    """
    This function filters the labels from naturgucker.de to eliminate some bugs.

    Parameters
    ----------
    data : numpy array
        The original input from naturgucker.de.

    Returns
    -------
    numpy array
        The filtered data from naturgucker.de.

    """
    for j in range(4):
        data[:,j] = [i.replace("[", "").replace("]", "") for i in data[:,j]]
    return(data)

def remove_problematic_categories(data):
    """
    This function removes all problematic categories from the dataset.

    Parameters
    ----------
    data : numpy array
        The original input from naturgucker.de.

    Returns
    -------
    numpy array
        The dataset from naturgucker.de after removing the problematic categories.

    """
    result = np.zeros(len(data[:,0]), dtype="bool")
    for i in range(len(result)):
        r = [any(["subsp." in data[i,j] for j in range(4)])]
        r.append(any(["?" in data[i,j] for j in range(4)]))
        r.append(any(["/" in data[i,j] for j in range(4)]))
        result[i] = any(r)
    return deepcopy(data[np.logical_not(result),:])

def remove_small_categories(data, threshold):
    """
    This function removes species, that do not contain enough samples.

    Parameters
    ----------
    data : numpy array
        The original input from naturgucker.de.
    threshold : integer
        The desired smallest number of samples for a category.

    Returns
    -------
    numpy array
        The dataset from naturgucker.de after removing the smallest categories.

    """
    [bins, counts] = np.unique(data[:,3], return_counts=True)
    remove_species = bins[counts > threshold]
    remove = np.zeros(len(data[:,0]), dtype="bool")
    for i in range(len(data[:,0])):
        remove[i] = data[i,3] in remove_species
    return deepcopy(data[remove,:])
