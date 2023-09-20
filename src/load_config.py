#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_config.py
# Copyright (C) 2022 flossCoder
# 
# load_config is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_config is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Sep  6 08:34:53 2022

@author: flossCoder
"""

import os

def convert_line_elem_numeric(elem):
    """
    This function tries to convert the given string to float.

    Parameters
    ----------
    elem : string
        The input string.

    Returns
    -------
    string or float
        The output is eather a float, in case the input string can be converted to float, or a string.

    """
    try:
        return float(elem)
    except:
        return elem.replace('"', '').replace("'", "")

def parse_config(wd, config, find_hp = True):
    """
    This function parses the parameter of a config file.

    Parameters
    ----------
    wd : string
        The basic working directory.
    config_filename : string
        The name of the config file.
    find_hp : bool, optional
        Search for keras tuner hyperparameter (start with "hp_"). The default is True.

    Returns
    -------
    result_dict : dictionary
        The dictionary contains.

    """
    f = open(os.path.join(wd, config))
    lines = f.readlines()
    f.close()
    
    result_dict = {}
    for line in lines:
        line = line.replace("\n", "").strip()
        if "=" in line and ((find_hp and "hp_" == line[:3]) or not(find_hp)):
            splitted_line = line.split("=")
            if len(splitted_line) == 2:
                splitted_line[0] = splitted_line[0].strip()
                splitted_line[1] = splitted_line[1].strip()
                result_dict[splitted_line[0]] = convert_line_elem_numeric(splitted_line[1])
    return result_dict
