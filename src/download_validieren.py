#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# download_validieren.py
# Copyright (C) 2021 flossCoder
# 
# download_validieren is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# download_validieren is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Jul 27 13:40:38 2021

@author: flossCoder
"""

import os
import sys
from bilder_download import import_voegel_csv

def validate(wd, data, directory_name = None):
    """
    This function checks, if the figures can be found and the required data structure exists.

    Parameters
    ----------
    wd : string
        The basic working directory.
    data : numpy array
        Contains all the bird species and file assignments as numpy array.
    directory_name : string
        The name of the subdirectory, where the data shall be saved, if None, save in wd.

    Returns
    -------
    None.

    """
    if directory_name != None:
        e = open(os.path.join(wd, directory_name, "missing_images"), "w")
    else:
        e = open(os.path.join(wd, "missing_images"), "w")
    for [order, family, genus, species, bird_id, bird_url] in data:
        order = order.replace(".", "_").replace("/", "__").replace(" x ?", "")
        family = family.replace(".", "_").replace("/", "__").replace(" x ?", "")
        genus = genus.replace(".", "_").replace("/", "__").replace(" x ?", "")
        species = species.replace(".", "_").replace("/", "__").replace(" x ?", "")
        order_wd = os.path.join(wd, directory_name) if directory_name != None else wd
        if order not in os.listdir(order_wd):
            e.write("order missing: %s, %s, %s, %s, %s\n"%(bird_id, order, family, genus, species))
            continue
        family_wd = os.path.join(order_wd, order)
        if family not in os.listdir(family_wd):
            e.write("family missing: %s, %s, %s, %s, %s\n"%(bird_id, order, family, genus, species))
            continue
        genus_wd = os.path.join(family_wd, family)
        if genus not in os.listdir(genus_wd):
            e.write("genus missing: %s, %s, %s, %s, %s\n"%(bird_id, order, family, genus, species))
            continue
        species_wd = os.path.join(genus_wd, genus)
        if species not in os.listdir(species_wd):
            e.write("species missing: %s, %s, %s, %s, %s\n"%(bird_id, order, family, genus, species))
            continue
        figure_wd = os.path.join(species_wd, species)
        figure_name = "%s.%s"%(bird_id, bird_url.split(".")[-1])
        if figure_name not in os.listdir(figure_wd):
            e.write("figure missing: %s, %s, %s, %s, %s\n"%(bird_id, order, family, genus, species))
            continue
    e.close()

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, filename, directory_name].

    Raises
    ------
    Exception
        The exception is raised, in case not enough parameter are passed to main.

    Returns
    -------
    None.

    """
    if len(argv) < 2:
        print("error, wrong number of parameter")
        print("wd: The basic working directory.")
        print("filename: The filename describes the name of the csv file containing the required data.")
        print("directory_name: The name of the subdirectory, where the data shall be saved, if None, save in wd.")
        raise Exception()
    elif len(argv) == 3:
        directory_name = argv[2]
    else:
        directory_name = None
    wd = argv[0]
    filename = argv[1]
    [header, data] = import_voegel_csv(wd, filename)
    validate(wd, data, directory_name)

if __name__ == "__main__":
    main(sys.argv[1:])
