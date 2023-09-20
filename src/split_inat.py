#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# split_inat.py
# Copyright (C) 2022 flossCoder
# 
# split_inat is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# split_inat is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Wed Jun 22 06:52:38 2022

@author: flossCoder
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy

from aux_io import load_json, dump_json

def generate_test_set_inat(wd, input_dir, output_dir, number_of_test_samples_per_category, train_filename = "train2017.json", train_bboxes_filename = "train_2017_bboxes.json", test_filename = "test2017.json", test_bboxes_filename = "test_2017_bboxes.json", seed = 42):
    """
    This function generates the test set from the training set of the INaturalist dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    input_dir : string
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
    number_of_test_samples_per_category : int
        The number of images in the test set per category.
    train_filename : string, optional
        The name of the json-file containing the training set. The default is "train2017.json".
    train_bboxes_filename : string, optional
        The name of the json-file containing the bounding box annotations of the training set. The default is "train_2017_bboxes.json".
    test_filename : string, optional
        The name of the json-file containing the test set. The default is "test2017.json".
    test_bboxes_filename : string, optional
        The name of the json-file containing the bounding box annotations of the test set. The default is "test_2017_bboxes.json".
    seed : int, optional
        The seed used for sklearn train_test_split guarantees reproducibility. The default is 42.

    Returns
    -------
    None.

    """
    # load the json data
    json_wd = os.path.join(wd, input_dir)
    train_data = load_json(json_wd, train_filename)
    train_bboxes = load_json(json_wd, train_bboxes_filename)
    [train_category_ids, train_counts] = np.unique([i["category_id"] for i in train_data["annotations"]], return_counts=True)
    train_data_new = deepcopy(train_data)
    train_data_new["annotations"] = []
    train_data_new["images"] = []
    train_bboxes_new = deepcopy(train_bboxes)
    train_bboxes_new["annotations"] = []
    train_bboxes_new["images"] = []
    test_data_new = deepcopy(train_data)
    test_data_new["annotations"] = []
    test_data_new["images"] = []
    test_bboxes_new = deepcopy(train_bboxes)
    test_bboxes_new["annotations"] = []
    test_bboxes_new["images"] = []
    for cid in train_category_ids:
        entries = [i for i in train_data["annotations"] if i["category_id"] == cid]
        [entries_train, entries_test, train_dummy, test_dummy] = train_test_split(entries, entries, test_size = number_of_test_samples_per_category, random_state = seed)
        # prepare saving the split into the new data structures
        image_train_ids = [e["image_id"] for e in entries_train]
        [train_data_new["annotations"].append(e) for e in entries_train]
        image_test_ids = [e["image_id"] for e in entries_test]
        [test_data_new["annotations"].append(e) for e in entries_test]
        # copy the old training images
        for i in train_data["images"]:
            if i["id"] in image_train_ids:
                train_data_new["images"].append(i)
            elif i["id"] in image_test_ids:
                test_data_new["images"].append(i)
        # copy the old training image annotations
        for i in train_data["annotations"]:
            if i["image_id"] in image_train_ids:
                train_data_new["annotations"].append(i)
            elif i["image_id"] in image_test_ids:
                test_data_new["annotations"].append(i)
        # copy the old training bounding boxes
        for i in train_bboxes["images"]:
            if i["id"] in image_train_ids:
                train_bboxes_new["images"].append(i)
            elif i["id"] in image_test_ids:
                test_bboxes_new["images"].append(i)
        # copy the old training bounding box annotations
        for i in train_bboxes["annotations"]:
            if i["image_id"] in image_train_ids:
                train_bboxes_new["annotations"].append(i)
            elif i["image_id"] in image_test_ids:
                test_bboxes_new["annotations"].append(i)
    # save the json files into output_dir
    dump_json(os.path.join(wd, output_dir), train_filename.replace("_filtered", ""), train_data_new)
    dump_json(os.path.join(wd, output_dir), train_bboxes_filename.replace("_filtered", ""), train_bboxes_new)
    dump_json(os.path.join(wd, output_dir), test_filename.replace("_filtered", ""), test_data_new)
    dump_json(os.path.join(wd, output_dir), test_bboxes_filename.replace("_filtered", ""), test_bboxes_new)
