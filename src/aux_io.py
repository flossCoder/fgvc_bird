#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# aux_io.py
# Copyright (C) 2022 flossCoder
# 
# aux_io is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# aux_io is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Feb 18 08:52:41 2022

@author: flossCoder
"""

import os
import json
import numpy as np

def load_json(wd, filename):
    """
    This function loads the data.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename : string
        The name of the json file that should be loaded.

    Returns
    -------
    json object
        The data obtained from the json file.

    """
    f = open(os.path.join(wd, filename), "r")
    data = json.load(f)
    f.close()
    return(data)

def dump_json(wd, filename, data):
    """
    This function saves the filtered data.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename : string
        The name of the json file that should be loaded.
    data : dict
        The filtered data from the inat challenge.

    Returns
    -------
    None.

    """
    f = open(os.path.join(wd, "%s_filtered.json"%filename.split(".json")[0]), "w")
    data = json.dump(data, f)
    f.close()

def save_image_temp(output_wd, output_name, images_file_training, images_file_test, images_file_validation):
    """
    This function saves the image files into a temporary file.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    images_file_training : numpy array
        The image paths of the trainings set.
    images_file_test : numpy array
        The image paths of the test set.
    images_file_validation : numpy array
        The image paths of the validation set.

    Returns
    -------
    list
        [temp_filename_dir, training_image_line_index, test_image_line_index, validation_image_line_index].
        temp_filename_dir : The path and filename to the temporary file.
        training_image_line_index : The index of the temp file for the training set.
        test_image_line_index : The index of the temp file for the test set.
        validation_image_line_index : The index of the temp file for the validation set.

    """
    temp_filename_dir = os.path.join(output_wd, "%s.temp"%(output_name))
    training_image_line_index = []
    test_image_line_index = []
    validation_image_line_index = []
    with open(temp_filename_dir, "w") as f:
        if type(images_file_training) != type(None) and len(images_file_training) != 0:
            f.writelines("\n".join(images_file_training))
            training_image_line_index = np.arange(len(images_file_training))
            f.writelines("\n")
        if type(images_file_test) != type(None) and len(images_file_test) != 0:
            f.writelines("\n".join(images_file_test))
            test_image_line_index = np.arange(len(training_image_line_index), len(training_image_line_index) + len(images_file_test))
            f.writelines("\n")
        if type(images_file_validation) != type(None) and len(images_file_validation) != 0:
            f.writelines("\n".join(images_file_validation))
            validation_image_line_index = np.arange(len(images_file_test), len(images_file_test) + len(images_file_validation))
            f.writelines("\n")
    return [temp_filename_dir, training_image_line_index, test_image_line_index, validation_image_line_index]

def rem_image_temp(temp_filename_dir):
    """
    This function removes the temporary file.

    Parameters
    ----------
    temp_filename_dir : string
        The path and filename to the temporary file.

    Returns
    -------
    None.

    """
    if os.path.exists(temp_filename_dir):
        os.remove(temp_filename_dir)

def load_history(wd, filename):
    """
    This function loads the history data.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename : string
        The name of the npy file that should be loaded.

    Returns
    -------
    history : dict of numpy arrays.
        The history as a dictionary.
    """
    history = np.load(os.path.join(wd, "%s.npy"%filename), allow_pickle =
True).tolist()
    for key in history.keys():
        history[key] = np.array(history[key])
    return history

def load_categorical_prediction(output_wd, output_name, epoch = None):
    """
    This function loads the category wise results of the test prediction.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    epoch : int, optional
        The epoch for loading the model. If None, the final model is loaded. The default is None.

    Returns
    -------
    result : dict
        The results after parsing the file.

    """
    result = {}
    f = open(os.path.join(output_wd, "%s_categorical-prediction_%s.txt"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")), "r")
    lines = f.readlines()
    f.close()
    add_keys = []
    if "#" in lines[0]:
        l = lines[0].replace("# ", "").replace("\n", "")
        for i in l.split(", "):
            key, value = i.split(": ")
            result[key] = float(value)
            result["cat_" + key] = np.zeros(len(lines) - 1)
            if key not in ["score", "Top-1-Acc", "Top-5-Acc"]:
                add_keys.append(key)
        
        result["id"] = np.zeros(len(lines) - 1)
        result["counts"] = np.zeros(len(lines) - 1)
    else:
        result["id"] = np.zeros(len(lines))
        result["counts"] = np.zeros(len(lines))
        result["cat_score"] = np.zeros(len(lines))
        result["cat_Top-1-Acc"] = np.zeros(len(lines))
        result["cat_Top-5-Acc"] = np.zeros(len(lines))
    for l in range(len(lines[1:])):
        j = [float(i) for i in lines[l + 1].replace("\n", "").split(" ")]
        result["id"][l] = j[0]
        result["counts"][l] = j[1]
        result["cat_score"][l] = j[2]
        result["cat_Top-1-Acc"][l] = j[3]
        result["cat_Top-5-Acc"][l] = j[4]
        if add_keys != []:
            for k in range(len(add_keys)):
                result["cat_" + add_keys[k]][l] = j[5 + k]
    
    return result
