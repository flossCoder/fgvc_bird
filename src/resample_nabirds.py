#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resample_nabirds.py
# Copyright (C) 2022 flossCoder
# 
# resample_nabirds is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resample_nabirds is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Feb 28 07:42:20 2022

@author: flossCoder
"""

import os
import sys
from shutil import copyfile
from train_test_eval_sampler import sample_dataset
from load_nabirds import load_txt_files_nabirds
from aux_functions import power

def resample_nabirds(input_wd, output_wd, bounding_boxes, classes, image_class_labels, images_file, train_test_split, test_number, val_number, min_training_samples, function_handle, function_args, bounding_boxes_fn = "bounding_boxes", classes_fn = "classes", image_class_labels_fn = "image_class_labels", images_fn = "images", train_test_split_fn = "train_test_split", seed = 42):
    """
    This function resamples the nabirds dataset.

    Parameters
    ----------
    input_wd : string
        The input working directory.
    output_wd : string
        The output working directory.
    bounding_boxes : numpy array
        The bounding box annotations.
    classes : numpy array
        The class annotations.
    image_class_labels : numpy array
        The image to class assignments.
    images_file : numpy array
        The image filename assignment.
    train_test_split : numpy array
        The train and test split.
    test_number : integer
        The number of images per category used for test.
    val_number : integer
        The number of images per category used for validation.
    min_training_samples : integer
        The minimum number of samples for training a category.
    function_handle : function handle
        The function describes how many samples should be used for a category i, where i is the number of the category in ascending sorting.
    function_args : list
        A list of arguments required for the function.
    bounding_boxes_fn : string, optional
        The input filename of the bounding box annotations. The default is "bounding_boxes".
    classes_fn : string, optional
        The input filename of the class annotations. The default is "classes".
    image_class_labels_fn : string, optional
        The input filename of the image to class assignments. The default is "image_class_labels".
    images_fn : string, optional
        The input filename of the image filename assignments. The default is "images".
    train_test_split_fn : string, optional
        The input filename of the train and test split. The default is "train_test_split".
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Returns
    -------
    None.

    """
    [train_index, test_index, val_index] = sample_dataset(image_class_labels[:,1], test_number, val_number, min_training_samples, function_handle, function_args, seed)
    index = train_index + test_index + val_index
    with open(os.path.join(output_wd, "%s.txt"%bounding_boxes_fn), "w") as f:
        for i in bounding_boxes[index]:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%classes_fn), "w") as f:
        for i in classes:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%image_class_labels_fn), "w") as f:
        for i in image_class_labels[index]:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%images_fn), "w") as f:
        for i in images_file[index]:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%train_test_split_fn), "w") as f:
        for i in train_test_split[train_index]:
            f.writelines(" ".join([i[0], "0"]) + "\n")
        for i in train_test_split[test_index]:
            f.writelines(" ".join([i[0], "1"]) + "\n")
        for i in train_test_split[val_index]:
            f.writelines(" ".join([i[0], "2"]) + "\n")
    
    aux_dir = os.path.join(output_wd, "images")
    os.mkdir(aux_dir)
    for i in images_file[index]:
        image_dir = os.path.join(output_wd, "images", i[1].split("/")[0])
        if i[1].split("/")[0] not in os.listdir(aux_dir):
            os.mkdir(image_dir)
        if i[1].split("/")[1] not in os.listdir(image_dir):
            copyfile(os.path.join(input_wd, "images", i[1]), os.path.join(image_dir, i[1].split("/")[1]))

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [input_wd, output_wd, test_number, val_number, min_training_samples, function_name, function_args...].

    Raises
    ------
    Exception
        The exception is raised, in case an invalid function name is passed.

    Returns
    -------
    None.

    """
    input_wd = argv[0]
    output_wd = argv[1]
    test_number = int(argv[2])
    val_number = int(argv[3])
    min_training_samples = int(argv[4])
    function_name = argv[5]
    function_args = [float(i) for i in argv[6:]]
    
    if function_name == "power":
        function_handle = power
    else:
        raise Exception("Invalid function name %s"%function_name)
    
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_nabirds(input_wd)
    resample_nabirds(input_wd, output_wd, bounding_boxes, classes, image_class_labels, images_file, train_test_split, test_number, val_number, min_training_samples, function_handle, function_args)

if __name__ == "__main__":
    main(sys.argv[1:])
