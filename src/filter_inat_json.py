#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# filter_inat_json.py
# Copyright (C) 2022 flossCoder
# 
# filter_inat_json is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# filter_inat_json is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Thu Jan 13 06:39:52 2022

@author: flossCoder
"""

import sys
from aux_io import load_json, dump_json

def filter_inat_json_train_val_2019(categories, data):
    """
    This function removes all informations for 2019 from the train or validation json, that does not belong to birds.

    Parameters
    ----------
    categories : dict
        The categories from the inat challenge.
    
    data : dict
        The data from the inat challenge, that should be filtered.

    Returns
    -------
    json object
        The filtered categories and dataset.
    """
    # figure out the categories
    print("remove categories")
    count = 0
    delete = []
    category_id = []
    while count < len(categories):
        if categories[count]["class"] != "Aves" and categories[count]["class"] != "Birds":
            delete.append(count)
            category_id.append(categories[count]["id"])
        count += 1
    # delete irrelevant categories
    for d in delete[::-1]:
        del categories[d]
    
    # figure out the categories
    print("remove categories")
    count = 0
    delete = []
    while count < len(data["categories"]):
        if data["categories"][count]["id"] not in category_id:
            delete.append(count)
        count += 1
    # delete irrelevant categories
    for d in delete[::-1]:
        del data["categories"][d]
    
    # figure out the annotations
    print("remove annotations")
    count = 0
    delete = []
    image_id = []
    while count < len(data["annotations"]):
        if data["annotations"][count]["category_id"] in category_id:
            image_id.append(data["annotations"][count]["image_id"])
            delete.append(count)
        count += 1
    # delete irrelevant categories
    for d in delete[::-1]:
        del data["annotations"][d]
    
    # figure out the images
    print("remove images")
    count = 0
    delete = []
    while count < len(data["images"]):
        if data["images"][0]["file_name"].split("/")[1] != "Aves" and data["images"][0]["file_name"].split("/")[1] != "Birds":#data["images"][count]["id"] in image_id:
            delete.append(count)
        count += 1
    # delete irrelevant images
    for d in delete[::-1]:
        del data["images"][d]
    
    return [categories, data]

def filter_inat_json_train_val(data):
    """
    This function removes all informations (except from 2019) from the train or validation json, that does not belong to birds.

    Parameters
    ----------
    data : dict
        The data from the inat challenge, that should be filtered.

    Returns
    -------
    json object
        The filtered dataset.

    """
    # figure out the categories
    print("remove categories")
    count = 0
    delete = []
    category_id = []
    while count < len(data["categories"]):
        if data["categories"][count]["supercategory"] != "Aves" and data["categories"][count]["supercategory"] != "Birds":
            delete.append(count)
            category_id.append(data["categories"][count]["id"])
        count += 1
    # delete irrelevant categories
    for d in delete[::-1]:
        del data["categories"][d]
    
    # figure out the annotations
    print("remove annotations")
    count = 0
    delete = []
    image_id = []
    while count < len(data["annotations"]):
        if data["annotations"][count]["category_id"] in category_id:
            image_id.append(data["annotations"][count]["image_id"])
            delete.append(count)
        count += 1
    # delete irrelevant categories
    for d in delete[::-1]:
        del data["annotations"][d]
    
    # figure out the images
    print("remove images")
    count = 0
    delete = []
    while count < len(data["images"]):
        if data["images"][0]["file_name"].split("/")[1] != "Aves" and data["images"][0]["file_name"].split("/")[1] != "Birds":#data["images"][count]["id"] in image_id:
            delete.append(count)
        count += 1
    # delete irrelevant images
    for d in delete[::-1]:
        del data["images"][d]
    return(data)

if __name__ == "__main__":
    wd = sys.argv[1]
    filename = sys.argv[2]
    if len(sys.argv) == 3:
        data = load_json(wd, filename)
        data = filter_inat_json_train_val(data)
        dump_json(wd, filename, data)
    elif len(sys.argv) == 4:
        categories = load_json(wd, sys.argv[3])
        data = load_json(wd, filename)
        [categories, data] = filter_inat_json_train_val_2019(categories, data)
        dump_json(wd, sys.argv[3], categories)
        dump_json(wd, filename, data)
    else:
        raise Exception("invalid number of arguments")
