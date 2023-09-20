#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resample_birdsnap.py
# Copyright (C) 2022 flossCoder
# 
# resample_birdsnap is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resample_birdsnap is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Mar  4 06:44:38 2022

@author: flossCoder
"""

import os
import sys
from shutil import copyfile
import numpy as np
from train_test_eval_sampler import sample_dataset
from load_birdsnap import load_txt_files_birdsnap
from aux_functions import power

def resample_birdsnap(wd, code_dir, input_dir, output_dir, image_split_fn, images_content, success_content, test_number, val_number, min_training_samples, function_handle, function_args, success_fn = "success.txt", seed = 42):
    """
    This function resamples the birdsnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    input_dir : strinig
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
    image_split_fn : string
        The name of the file containing the split into train, test and evalutation.
    images_content : list
        The content of the images annotations.
    success_content : list
        The content of the images which have been successfully downloaded.
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
    success_fn : string, optional
        The list of images downloaded successfully. The default is "success.txt".
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Returns
    -------
    None.

    """
    index = [np.where(images_content[:,2] == i)[0][0] for i in success_content[:,1]]
    [train_index, test_index, val_index] = sample_dataset(images_content[index,3], test_number, val_number, min_training_samples, function_handle, function_args, seed)
    f = open(os.path.join(wd, output_dir, success_fn), "w")
    f_ = open(os.path.join(wd, output_dir, image_split_fn), "w")
    for i in images_content[index][train_index, 2]:
        f.writelines("\t".join(success_content[np.where(success_content[:,1] == i)[0][0]]) + "\n")
        f_.writelines("\t".join([i, "0"]) + "\n")
    for i in images_content[index][test_index, 2]:
        f.writelines("\t".join(success_content[np.where(success_content[:,1] == i)[0][0]]) + "\n")
        f_.writelines("\t".join([i, "1"]) + "\n")
    for i in images_content[index][val_index, 2]:
        f.writelines("\t".join(success_content[np.where(success_content[:,1] == i)[0][0]]) + "\n")
        f_.writelines("\t".join([i, "2"]) + "\n")
    f.close()
    f_.close()
    
    aux_dir = os.path.join(wd, output_dir, "images")
    os.mkdir(aux_dir)
    for i in images_content[index][(train_index + test_index + val_index),2]:
        image_dir = os.path.join(wd, output_dir, "images", i.split("/")[0])
        if i.split("/")[0] not in os.listdir(aux_dir):
            os.mkdir(image_dir)
        if i.split("/")[1] not in os.listdir(image_dir):
            copyfile(os.path.join(wd, input_dir, "images", i), os.path.join(image_dir, i.split("/")[1]))

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, code_dir, input_dir, output_dir, image_split_fn,

    Raises
    ------
    Exception
        The exception is raised, in case an invalid function name is passed.

    Returns
    -------
    None.

    """
    wd = argv[0]
    code_dir = argv[1]
    input_dir = argv[2]
    output_dir = argv[3]
    image_split_fn = argv[4]
    test_number = int(argv[5])
    val_number = int(argv[6])
    min_training_samples = int(argv[7])
    function_name = argv[8]
    function_args = [float(i) for i in argv[9:]]
    
    if function_name == "power":
        function_handle = power
    else:
        raise Exception("Invalid function name %s"%function_name)
    
    [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content] = load_txt_files_birdsnap(wd, code_dir, input_dir)
    resample_birdsnap(wd, code_dir, input_dir, output_dir, image_split_fn, images_content, success_content, test_number, val_number, min_training_samples, function_handle, function_args)

if __name__ == "__main__":
    main(sys.argv[1:])
