#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_inat.py
# Copyright (C) 2022 flossCoder
# 
# load_inat is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_inat is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Jun 28 06:49:33 2022

@author: flossCoder
"""

import os
import numpy as np
from aux_io import load_json
from aux_functions import compute_label_assignment, convert_labels_to_artificial_labels

def prepare_classification_inat(wd, json_dir, data_dir, classification_data_filename = None, labels_artificial_labels_assignment_filename = None, train_filename = "train2017.json", val_filename = "val2017.json", test_filename = "test2017.json"):
    """
    This function loads the labels and image paths of the iNat dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    json_dir : string
        The json_dir describes the sub-directory of wd containing the json files.
    data_dir : string
        The data_dir describes the sub-directory of wd containing the image files.
    classification_data_filename : string, optional
        The filename of precalculated classification data, if it exists, otherwise None is given. The default is None.
    labels_artificial_labels_assignment_filename : string, optional
        The name of the labels to artificial labels assignment, if it exists, otherwise None is given. The default is None.
    train_filename : string, optional
        The name of the json-file containing the training set. The default is "train2017.json".
    val_filename : string, optional
        The name of the json-file containing the validation set. The default is "val2017.json".
    test_filename : string, optional
        The name of the json-file containing the test set. The default is "test2017.json".

    Returns
    -------
    list
        [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation].
        images_file_training : The image paths of the trainings set.
        image_class_labels_training : The labels of the trainings set.
        images_file_test : The image paths of the test set.
        image_class_labels_test : The labels of the test set.
        images_file_validation : The image paths of the validation set.
        image_class_labels_validation : The labels of the validation set.

    """
    json_wd = os.path.join(wd, json_dir)
    if type(classification_data_filename) != type(None):
        [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation] = np.load(os.path.join(json_wd, classification_data_filename), allow_pickle = True)
        images_file_training = rename_dir_images_file(wd, data_dir, images_file_training)
        images_file_test = rename_dir_images_file(wd, data_dir, images_file_test)
        images_file_validation = rename_dir_images_file(wd, data_dir, images_file_validation)
    else:
        train_data = load_json(json_wd, train_filename)
        val_data = load_json(json_wd, val_filename)
        test_data = load_json(json_wd, test_filename)
        [images_file_training, image_class_labels_training] = calculate_auxiliary_assignment(train_data)
        images_file_training = [os.path.join(wd, data_dir, i) for i in images_file_training]
        [images_file_test, image_class_labels_test] = calculate_auxiliary_assignment(test_data)
        images_file_test = [os.path.join(wd, data_dir, i) for i in images_file_test]
        [images_file_validation, image_class_labels_validation] = calculate_auxiliary_assignment(val_data)
        images_file_validation = [os.path.join(wd, data_dir, i) for i in images_file_validation]
        if type(labels_artificial_labels_assignment_filename) != type(None):
            labels_artificial_labels_assignment = np.loadtxt(os.path.join(json_wd, labels_artificial_labels_assignment_filename))
        else:
            labels = np.unique(np.concatenate((image_class_labels_training, image_class_labels_test, image_class_labels_validation)))
            labels_artificial_labels_assignment = compute_label_assignment(labels)
        image_class_labels_training = convert_labels_to_artificial_labels(image_class_labels_training, labels_artificial_labels_assignment)
        image_class_labels_test = convert_labels_to_artificial_labels(image_class_labels_test, labels_artificial_labels_assignment)
        image_class_labels_validation = convert_labels_to_artificial_labels(image_class_labels_validation, labels_artificial_labels_assignment)
    return [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation]

def prepare_bb_estimation(wd, json_dir, data_dir, bbox_data_filename = None, classification_data_filename = None, labels_artificial_labels_assignment_filename = None, train_filename = "train2017.json", val_filename = "val2017.json", test_filename = "test2017.json", train_bboxes_filename = "train2017_bboxes.json", val_bboxes_filename = "val2017_bboxes.json", test_bboxes_filename = "test2017_bboxes.json"):
    """
    This function loads the labels, bounding boxes and image paths of the iNat dataset and aggregates the sets.

    Parameters
    ----------
    wd : string
        The basic working directory.
    json_dir : string
        The json_dir describes the sub-directory of wd containing the json files.
    data_dir : string
        The data_dir describes the sub-directory of wd containing the image files.
    bbox_data_filename : string, optional
        The filename of precalculated bounding box data, if it exists, otherwise None is given. The default is None.
    classification_data_filename : string, optional
        The filename of precalculated classification data, if it exists, otherwise None is given. The default is None.
    labels_artificial_labels_assignment_filename : string, optional
        The name of the labels to artificial labels assignment, if it exists, otherwise None is given. The default is None.
    train_filename : string, optional
        The name of the json-file containing the training set. The default is "train2017.json".
    val_filename : string, optional
        The name of the json-file containing the validation set. The default is "val2017.json".
    test_filename : string, optional
        The name of the json-file containing the test set. The default is "test2017.json".
    train_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the training set. The default is "train2017_bboxes.json".
    val_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the validation set. The default is "val2017_bboxes.json".
    test_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the test set. The default is "test2017_bboxes.json".

    Returns
    -------
    list
        [images_file, image_class_labels, bounding_boxes].
        images_file : The image path of the data set.
        image_class_labels : The labels of the data set.
        bounding_boxes : The bounding boxes of the data set.

    """
    [images_file_training, image_class_labels_training, bboxes_training, images_file_validation, image_class_labels_validation, bboxes_validation, images_file_test, image_class_labels_test, bboxes_test] = prepare_bb_estimation_train_test_val_set(wd, json_dir, data_dir, bbox_data_filename, classification_data_filename, labels_artificial_labels_assignment_filename, train_filename, val_filename, test_filename, train_bboxes_filename, val_bboxes_filename, test_bboxes_filename)
    images_file = np.concatenate((images_file_training, images_file_validation, images_file_test))
    image_class_labels = np.concatenate((image_class_labels_training, image_class_labels_validation, image_class_labels_test))
    bounding_boxes = np.concatenate((bboxes_training, bboxes_validation, bboxes_test))
    return [images_file, image_class_labels, bounding_boxes]

def prepare_bb_estimation_train_test_val_set(wd, json_dir, data_dir, bbox_data_filename = None, classification_data_filename = None, labels_artificial_labels_assignment_filename = None, train_filename = "train2017.json", val_filename = "val2017.json", test_filename = "test2017.json", train_bboxes_filename = "train2017_bboxes.json", val_bboxes_filename = "val2017_bboxes.json", test_bboxes_filename = "test2017_bboxes.json"):
    """
    This function loads the labels, bounding boxes and image paths of the iNat dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    json_dir : string
        The json_dir describes the sub-directory of wd containing the json files.
    data_dir : string
        The data_dir describes the sub-directory of wd containing the image files.
    bbox_data_filename : string, optional
        The filename of precalculated bounding box data, if it exists, otherwise None is given. The default is None.
    classification_data_filename : string, optional
        The filename of precalculated classification data, if it exists, otherwise None is given. The default is None.
    labels_artificial_labels_assignment_filename : string, optional
        The name of the labels to artificial labels assignment, if it exists, otherwise None is given. The default is None.
    train_filename : string, optional
        The name of the json-file containing the training set. The default is "train2017.json".
    val_filename : string, optional
        The name of the json-file containing the validation set. The default is "val2017.json".
    test_filename : string, optional
        The name of the json-file containing the test set. The default is "test2017.json".
    train_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the training set. The default is "train2017_bboxes.json".
    val_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the validation set. The default is "val2017_bboxes.json".
    test_bboxes_filename : string, optional
        The name of the json-file containing bounding boxes of the test set. The default is "test2017_bboxes.json".

    Returns
    -------
    list
        [images_file_training, image_class_labels_training, bboxes_training, images_file_validation, image_class_labels_validation, bboxes_validation, images_file_test, image_class_labels_test, bboxes_test]
        images_file_training : The image path of the training set.
        image_class_labels_training : The labels of the training set.
        bboxes_training : The bounding boxes of the training set.
        images_file_validation : The image path of the validation set.
        image_class_labels_validation : The labels of the validation set.
        bboxes_validation : The bounding boxes of the validation set.
        images_file_test : The image path of the test set.
        image_class_labels_test : The labels of the test set.
        bboxes_test : The bounding boxes of the test set.

    """
    json_wd = os.path.join(wd, json_dir)
    if type(bbox_data_filename) != type(None):
        [images_file_training, image_class_labels_training, bboxes_training, images_file_validation, image_class_labels_validation, bboxes_validation, images_file_test, image_class_labels_test, bboxes_test] = np.load(os.path.join(json_wd, bbox_data_filename), allow_pickle = True)
        images_file_training = rename_dir_images_file(wd, data_dir, images_file_training)
        images_file_test = rename_dir_images_file(wd, data_dir, images_file_test)
        images_file_validation = rename_dir_images_file(wd, data_dir, images_file_validation)
    else:
        train_bboxes = load_json(json_wd, train_bboxes_filename)
        val_bboxes = load_json(json_wd, val_bboxes_filename)
        test_bboxes = load_json(json_wd, test_bboxes_filename)
        [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation] = prepare_classification_inat(wd, json_dir, data_dir, classification_data_filename, labels_artificial_labels_assignment_filename, train_filename, val_filename, test_filename)
        [images_file_training, image_class_labels_training, bboxes_training] = calculate_bbox_assignment(images_file_training, train_bboxes)
        [images_file_validation, image_class_labels_validation, bboxes_validation] = calculate_bbox_assignment(images_file_validation, val_bboxes)
        [images_file_test, image_class_labels_test, bboxes_test] = calculate_bbox_assignment(images_file_test, test_bboxes)
    return [images_file_training, image_class_labels_training, bboxes_training, images_file_validation, image_class_labels_validation, bboxes_validation, images_file_test, image_class_labels_test, bboxes_test]

def rename_dir_images_file(wd, data_dir, images_file):
    """
    This function ensures, that the directory description is correct in terms of wd and data_dir.

    Parameters
    ----------
    wd : string
        The basic working directory.
    json_dir : string
        The json_dir describes the sub-directory of wd containing the json files.
    data_dir : string
        The data_dir describes the sub-directory of wd containing the image files.
    images_file : list
        The image path of the data set.

    Returns
    -------
    images_file : list
        The image path of the data set after correcting the dir.

    """
    if len(images_file) != 0 and os.path.join(wd, data_dir) not in images_file[0].split("Aves"):
        images_file = [os.path.join(wd, data_dir, "Aves", i.split("Aves")[1].strip("/")) for i in images_file]
    return images_file

def calculate_bbox_assignment(images_file, data):
    """
    This function obtains the image to label to bounding box assignment,
    where image and label are valid for the corresponding bounding box.

    Parameters
    ----------
    images_file : list
        The image path of the data set.
    data : dict
        The raw data of the bounding boxes json file.

    Returns
    -------
    list
        [images_file_new, image_class_labels_new, bboxes].
        images_file_new : The image path of the data set for the new assignment.
        image_class_labels_new : The labels of the data set for the new assignment.
        bboxes : The bounding boxes of the corresponding images.

    """
    image_names = np.array([i["file_name"].split("Aves")[-1] for i in data["images"]])
    image_id  = np.array([i["id"] for i in data["images"]])
    bbox_coordinates = np.array([i["bbox"] for i in data["annotations"]])
    bbox_category_id = np.array([i["category_id"] for i in data["annotations"]])
    bbox_image_id = np.array([i["image_id"] for i in data["annotations"]])
    images_file_new = []
    image_class_labels_new = []
    bboxes = []
    for i in range(len(images_file)):
        index = image_names == images_file[i].split("Aves")[-1]
        if any(index):
            current_id = image_id[index]
            for j in np.where(bbox_image_id == current_id)[0]:
                images_file_new.append(images_file[i])
                image_class_labels_new.append(bbox_category_id[j])
                bboxes.append(current_id.tolist() + bbox_coordinates[j].tolist())
    return [np.array(images_file_new), np.array(image_class_labels_new), np.array(bboxes)]

def calculate_auxiliary_assignment(data):
    """
    This function calculates the image file label assignment for a single subset of iNat.

    Parameters
    ----------
    data : dict
        The input subset of the dataset.

    Returns
    -------
    list
        [images_file, image_class_labels].
        images_file : The image paths of the subset.
        image_class_labels : The labels of the subset.

    """
    images_file = []
    image_class_labels = []
    aux = np.zeros((len(data["annotations"]), 2))
    count = 0
    for a in data["annotations"]:
        aux[count] = [a["image_id"], a["category_id"]]
        count += 1
    for i in data["images"]:
        image_filename = "/".join(i["file_name"].split("/")[1:])
        annotations = np.unique(aux[aux[:,0] == i["id"],1])
        for annotation in annotations:
            images_file.append(image_filename)
            image_class_labels.append(annotation)
    return [images_file, image_class_labels]
