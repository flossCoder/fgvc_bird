#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# bilder_download.py
# Copyright (C) 2021 flossCoder
# 
# bilder_download is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# bilder_download is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jun 28 09:37:18 2021

@author: flossCoder
"""

import numpy as np
import os
import requests
import sys
import csv
from copy import deepcopy

def import_voegel_csv(wd, filename):
    """
    This function imports the csv file containing the bird species and file assignment.

    Parameters
    ----------
    wd : string
        The working directory.
    filename : string
        The filename describes the name of the csv file containing the required data.

    Returns
    -------
    list
        [header, data].
        The header, as numpy array, if the csv file contained a header, otherwise None.
        Data contains all the bird species and file assignments as numpy array.
    """
    if ".csv" in filename:
        file = open(os.path.join(wd, filename), "r")
    else:
        file = open(os.path.join(wd, "%s.csv"%filename), "r")
    lines = file.readlines()
    file.close()
    # find the header
    if "Ordnung" in lines[0] or "Familie" in lines[0] or "Gattung" in lines[0] or "Art" in lines[0] or "BildID" in lines[0] or "Link flickr" in lines[0]:
        header = np.array([i.split("\n")[0] for i in lines[0].split("\t")])
        lines = lines[1:]
    else:
        header = None
    data = []
    for line in lines:
        data.append([i.split("\n")[0] for i in line.split("\t")])
    return [header, np.array(data)]

def aux_mkdir(wd, dirname):
    """
    This function generates the required directory, if necessary.

    Parameters
    ----------
    wd : string
        The basic working directory.
    dirname : string
        The path in the working directory to be generated, if required.

    Returns
    -------
    None.
    
    Raises
    ------
    e : OSError
        This error is raised, when the os tries to generate a new directory.

    """
    if dirname not in os.listdir(wd):
        try:
            os.mkdir(os.path.join(wd, dirname))
        except OSError as e:
            print(e)
            raise(e)

def make_ids_unique(data):
    """
    This function repairs all the duplicate ids.

    Parameters
    ----------
    data : numpy array
        Contains all the bird species and file assignments as numpy array.

    Returns
    -------
    list
        [data, change_happend].
        data contains all the bird species and file assignments as numpy array.
        change_happened = True => ids had to be renamed, otherwise not.

    """
    [unique_ids, counts] = np.unique(data[:,-2], return_counts=True)
    if np.shape(data)[0] != np.shape(unique_ids)[0]:
        for current_id in unique_ids[counts != 1]:
            count = 0
            for index in np.where(data[:,-2]==current_id)[0]:
                data[index,-2] = "%s_%s"%(current_id, str(count))
                count += 1
        return [data, True]
    else:
        return [data, False]

def download_birds(wd, data, directory_name = None):
    """
    This function performs the download of the bird figures.

    Parameters
    ----------
    wd : string
        The basic working directory.
    data : numpy array
        Contains all the bird species and file assignments as numpy array.
    directory_name : string, optional
        The name of the subdirectory, where the data shall be saved, if None, save in wd.

    Returns
    -------
    None.

    """
    if directory_name != None:
        aux_mkdir(wd, directory_name)
        d = os.path.join(wd, directory_name)
    else:
        d = wd
    ld = os.listdir(d)
    if "error_log" in ld:
        count = 1
        while "error_log_%i"%count in ld:
            count += 1
        os.rename(os.path.join(d, "error_log"), os.path.join(d, "error_log_%i"%count))
    e = open(os.path.join(d, "error_log"), "a")
    
    for [order, family, genus, species, bird_id, bird_url] in data:
        order = order.replace(".", "_").replace("/", "__").replace(" x ?", "")
        family = family.replace(".", "_").replace("/", "__").replace(" x ?", "")
        genus = genus.replace(".", "_").replace("/", "__").replace(" x ?", "")
        species = species.replace(".", "_").replace("/", "__").replace(" x ?", "")
        order_wd = os.path.join(wd, directory_name) if directory_name != None else wd
        aux_mkdir(order_wd, order)
        family_wd = os.path.join(order_wd, order)
        aux_mkdir(family_wd, family)
        genus_wd = os.path.join(family_wd, family)
        aux_mkdir(genus_wd, genus)
        species_wd = os.path.join(genus_wd, genus)
        aux_mkdir(species_wd, species)
        figure_wd = os.path.join(species_wd, species)
        if "%s.%s"%(bird_id, bird_url.split(".")[-1]) not in os.listdir(figure_wd):
            print("%s, %s, %s, %s, %s, %s"%(order, family, genus, species, bird_id, bird_url))
            try:
                picture = requests.get(bird_url, timeout=120)
                if picture.status_code == 200:
                    with open(os.path.join(figure_wd, "%s.%s"%(bird_id, bird_url.split(".")[-1])), 'wb') as f:
                        f.write(picture.content)
                else:
                    e.write("%s, %s, %s, %s; %s, %s: %s\n"%(order, family, genus, species, bird_id, bird_url, str(picture.status_code)))
            except Exception as ex:
                e.write("%s, %s, %s, %s; %s, %s: failed: %s\n"%(order, family, genus, species, bird_id, bird_url, str(ex)))
    e.close()

def remove_unnecessary_dir(wd, directory_name = None):
    """
    This function removes all unnecessary directories.

    Parameters
    ----------
    wd : string
        The basic working directory.
    directory_name : string
        The name of the subdirectory, where the data have been saved, if None, use wd.

    Returns
    -------
    list
        A list containing the informations for removed species.

    """
    if directory_name != None:
        d = os.path.join(wd, directory_name)
    else:
        d = wd
    remove = []
    for f in os.listdir(d):
        a = os.path.join(d, f)
        if os.path.isdir(a):
            # recursively check the subdirecory
            b = remove_unnecessary_dir(a)
            if b != []:
                [remove.append(i) if type(i) == type([]) else remove.append([i]) for i in b]
    # remove d, if it does not contain any files
    if len(remove) < 4:
        [i.insert(0, d.split("/")[-1]) if len(i) < 4 else i for i in remove] if remove != [] else []
    if os.listdir(d) == []:
        os.rmdir(d)
        if remove == []:
            remove = [d.split("/")[-1]]
    return remove

def clean_up(wd, filename, header, data, directory_name = None):
    """
    This function cleans the downloaded directory and the input csv.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename : string
        The filename describes the name of the csv file containing the required data.
    header : numpy array
        The header, as numpy array, if the csv file contained a header, otherwise None.
    data : numpy array
        Contains all the bird species and file assignments as numpy array.
    directory_name : string
        The name of the subdirectory, where the data have been saved, if None, use wd.

    Returns
    -------
    None.

    """
    def aux_logical_and(j, index = 3):
        """
        This function tests the logical and in the code below for remove result from remove_unnecessary_dir.

        Parameters
        ----------
        j : numpy array
            The current state of the remove line under test.
        index : int, optional
            The current column index in the data_ array. The default is 3.

        Returns
        -------
        boolean numpy array
            The lines in data_, that can be removed (True) for the current remove line.

        """
        if len(j) == 1:
            return data_[:,index]==j[-1]
        else:
            return np.logical_and(data[:,index] == j[-1], aux_logical_and(j[:-1], index - 1))
    remove = remove_unnecessary_dir(wd, directory_name)
    result = np.zeros(len(data[:,0]), dtype="bool")
    data_ = deepcopy(data)
    for i in range(np.shape(data)[0]):
        data_[i,0] = data_[i,0].replace(".", "_").replace("/", "__").replace(" x ?", "")
        data_[i,1] = data_[i,1].replace(".", "_").replace("/", "__").replace(" x ?", "")
        data_[i,2] = data_[i,2].replace(".", "_").replace("/", "__").replace(" x ?", "")
        data_[i,3] = data_[i,3].replace(".", "_").replace("/", "__").replace(" x ?", "")
    for i in remove:
        result = np.logical_or(result, aux_logical_and(i))
    export_vogel_csv(wd, "%s_removed"%filename, data[np.logical_not(result),:], header)

def export_vogel_csv(wd, filename, data, header = None):
    """
    This function exports the data as csv.

    Parameters
    ----------
    wd : string
        The directory, where the data should be saved.
    filename : string
        The filename describes the name of the csv file containing the required data.
    data : numpy array
        Contains all the bird species and file assignments as numpy array.
    header : numpy array, optional
        The header, as numpy array, if the csv file contained a header, otherwise None. The default is None.

    Returns
    -------
    None.

    """
    with open(os.path.join(wd, "%s.csv"%filename), 'w', newline='') as f:
        writer = csv.writer(f, delimiter="\t")
        if type(header) != type(None):
            writer.writerow(header)
        writer.writerows(data.tolist())

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
    download_birds(wd, data, directory_name)
    clean_up(wd, filename, header, data, directory_name)

if __name__ == "__main__":
    main(sys.argv[1:])
    #wd="/home/adolf/Dokumente/bilder_export"
    #filename="test"
    #directory_name="test"
    #remove_unnecessary_dir(wd, directory_name)
    #wd="/home/adolf/Dokumente/bilder_export"
    #filename="mini"
    #directory_name="foo"
    #[header, data] = import_voegel_csv(wd, filename)
    #download_birds(wd, data, directory_name)
