#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_nabirds.py
# Copyright (C) 2022 flossCoder
# 
# load_nabirds is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_nabirds is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Feb 25 09:57:36 2022

@author: flossCoder
"""

import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from image_processing import resize_bounding_boxes, resize_images, normalize_bounding_boxes
from aux_functions import compute_label_assignment, convert_labels_to_artificial_labels

def load_txt_files_nabirds(wd, bounding_boxes_fn = "bounding_boxes", classes_fn = "classes", image_class_labels_fn = "image_class_labels", images_fn = "images", train_test_split_fn = "train_test_split"):
    """
    This function loads the txt files from the NABirds dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
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

    Returns
    -------
    list
        [bounding_boxes, classes, image_class_labels, images_file, train_test_split].
        bounding_boxes : The bounding box annotations.
        classes : The class annotations.
        image_class_labels : The image to class assignments.
        images_file : The image filename assignment.
        train_test_split : The train and test split.

    """
    bounding_boxes = np.loadtxt(os.path.join(wd, "%s.txt"%bounding_boxes_fn), dtype=str)
    classes = []
    f = open(os.path.join(wd, "%s.txt"%classes_fn), "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line_split = line.split(" ")
        classes.append([line_split[0], " ".join([i.replace("\n", "") for i in line_split[1:]])])
    classes = np.array(classes, dtype=str)
    image_class_labels = np.loadtxt(os.path.join(wd, "%s.txt"%image_class_labels_fn), dtype=str)
    images_file = np.loadtxt(os.path.join(wd, "%s.txt"%images_fn), dtype=str)
    train_test_split = np.loadtxt(os.path.join(wd, "%s.txt"%train_test_split_fn), dtype=str)
    return [bounding_boxes, classes, image_class_labels, images_file, train_test_split]

def load_images_nabirds(wd, images_file):
    """
    This function loads the images from the NABirds dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    images_file : numpy array
        The image filename assignment.

    Returns
    -------
    list
        [images, original_resolution]
        images_file : Each entry of the list contains an image in numpy format.
        original_resolution : Contains the original resolution of the image.

    """
    images = []
    original_resolution = []
    for i in images_file:
        image = Image.open(os.path.join(wd, "images", i[1])).convert('RGB')
        images.append(np.array(image))
        original_resolution.append(image.size)
    return [images, original_resolution]

def performe_train_test_validation_split_nabirds(bounding_boxes, image_class_labels, images_file, train_test_split):
    """
    This function splits the input data into training-, test- and validation set.

    Parameters
    ----------
    bounding_boxes : numpy array
        The bounding box annotations.
    image_class_labels : numpy array
        The image to class assignments.
    images_file : numpy array
        The image filename assignment.
    train_test_split : numpy array
        The train and test split.

    Raises
    ------
    Exception
        The exception is raised in case an invalid number accounts in train_test_split.

    Returns
    -------
    list
        [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation,
         image_class_labels_training, image_class_labels_test, image_class_labels_validation,
         images_file_training, images_file_test, images_file_validation].
        bounding_boxes_training : The bounding box annotations for training.
        bounding_boxes_test : The bounding box annotations for testing.
        bounding_boxes_validation : The bounding box annotations for validation.
        image_class_labels_training : The image to class assignments for training.
        image_class_labels_test : The image to class assignments for testing.
        image_class_labels_validation : The image to class assignments for validation.
        images_file_training : The image filename assignment for training.
        images_file_test : The image filename assignment for testing.
        images_file_validation : The image filename assignment for validation.

    """
    number_of_trainings_samples = sum(train_test_split[:,1] == "0")
    number_of_test_samples = sum(train_test_split[:,1] == "1")
    number_of_validation_samples = sum(train_test_split[:,1] == "2")
    bounding_boxes_training = np.zeros((number_of_trainings_samples, np.shape(bounding_boxes)[1]), dtype='<U65')
    bounding_boxes_test = np.zeros((number_of_test_samples, np.shape(bounding_boxes)[1]), dtype='<U65')
    bounding_boxes_validation = np.zeros((number_of_validation_samples, np.shape(bounding_boxes)[1]), dtype='<U65')
    image_class_labels_training = np.zeros((number_of_trainings_samples, np.shape(image_class_labels)[1]), dtype='<U65')
    image_class_labels_test = np.zeros((number_of_test_samples, np.shape(image_class_labels)[1]), dtype='<U65')
    image_class_labels_validation = np.zeros((number_of_validation_samples, np.shape(image_class_labels)[1]), dtype='<U65')
    images_file_training = np.zeros((number_of_trainings_samples, np.shape(images_file)[1]), dtype='<U65')
    images_file_test = np.zeros((number_of_test_samples, np.shape(images_file)[1]), dtype='<U65')
    images_file_validation = np.zeros((number_of_validation_samples, np.shape(images_file)[1]), dtype='<U65')
    trainings_count = 0
    test_count = 0
    validation_count = 0
    for i in range(np.shape(train_test_split)[0]):
        image_id = train_test_split[i,0]
        if train_test_split[i,1] == "0":
            if bounding_boxes[i,0] == image_id:
                bounding_boxes_training[trainings_count,:] = bounding_boxes[i,:]
            else:
                bounding_boxes_training[trainings_count,:] = bounding_boxes[(bounding_boxes[:,0]==image_id),:]
            if image_class_labels[i,0] == image_id:
                image_class_labels_training[trainings_count,:] = image_class_labels[i,:]
            else:
                image_class_labels_training[trainings_count,:] = image_class_labels[(image_class_labels[:,0] == image_id),:]
            if image_id == images_file[i,0]:
                images_file_training[trainings_count,:] = images_file[i,:]
            else:
                images_file_training[trainings_count,:] = images_file[(images_file[:,0] == image_id),:]
            trainings_count += 1
        elif train_test_split[i,1] == "1":
            if bounding_boxes[i,0] == image_id:
                bounding_boxes_test[test_count,:] = bounding_boxes[i,:]
            else:
                bounding_boxes_test[test_count,:] = bounding_boxes[(bounding_boxes[:,0] == image_id),:]
            if image_class_labels[i,0] == image_id:
                image_class_labels_test[test_count,:] = image_class_labels[i,:]
            else:
                image_class_labels_test[test_count,:] = image_class_labels[(image_class_labels[:,0]==image_id),:]
            if image_id == images_file[i,0]:
                images_file_test[test_count,:] = images_file[i,:]
            else:
                images_file_test[test_count,:] = images_file[(images_file[:,0] == image_id),:]
            test_count += 1
        elif train_test_split[i,1] == "2":
            if bounding_boxes[i,0] == image_id:
                bounding_boxes_validation[validation_count,:] = bounding_boxes[i,:]
            else:
                bounding_boxes_validation[validation_count,:] = bounding_boxes[(bounding_boxes[:,0] == image_id),:]
            if image_class_labels[i,0] == image_id:
                image_class_labels_validation[validation_count,:] = image_class_labels[i,:]
            else:
                image_class_labels_validation[validation_count,:] = image_class_labels[(image_class_labels[:,0] == image_id),:]
            if image_id == images_file[i,0]:
                images_file_validation[validation_count,:] = images_file[i,:]
            else:
                images_file_validation[validation_count,:] = images_file[(images_file[:,0] == image_id),:]
            validation_count += 1
        else:
            raise Exception("Invalid train test split class %i for id %i"%(int(train_test_split[i,1]), int(train_test_split[i,0])))
    return [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation]

def prepare_train_test_nabirds(wd, resolution = None):
    """
    This function prepares everything for training and testing.

    Parameters
    ----------
    wd : string
        The basic working directory.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.

    Returns
    -------
    list
        [images_training, labels_training, bb_training, images_test, labels_test, bb_test, images_validation, labels_validation, bb_validation].
        images_training : Images for training.
        labels_training : Labels for training.
        bb_training : Bounding Boxes for training.
        images_test : Images for testing.
        labels_test : Labels for testing.
        bb_test : Bounding Boxes for testing.
        images_validation : Images for validation.
        labels_validation : Labels for validation.
        bb_validation : Bounding Boxes for validation.

    """
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_nabirds(wd)
    [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation] = performe_train_test_validation_split_nabirds(bounding_boxes, image_class_labels, images_file, train_test_split)
    [images_training, original_resolution_training] = load_images_nabirds(wd, images_file_training)
    [images_test, original_resolution_test] = load_images_nabirds(wd, images_file_test)
    [images_validation, original_resolution_validation] = load_images_nabirds(wd, images_file_validation)
    bounding_boxes_training = np.array([[0] + [float(j) for j in i[1:]] for i in bounding_boxes_training])
    bounding_boxes_test = np.array([[0] + [float(j) for j in i[1:]] for i in bounding_boxes_test])
    bounding_boxes_validation = np.array([[0] + [float(j) for j in i[1:]] for i in bounding_boxes_validation])
    if resolution != None:
        images_training = resize_images(images_training, resolution)
        images_test = resize_images(images_test, resolution)
        images_validation = resize_images(images_validation, resolution)
        bounding_boxes_training = resize_bounding_boxes(bounding_boxes_training, original_resolution_training, resolution)
        bounding_boxes_test = resize_bounding_boxes(bounding_boxes_test, original_resolution_test, resolution)
        bounding_boxes_validation = resize_bounding_boxes(bounding_boxes_validation, original_resolution_validation, resolution)
        if len(bounding_boxes_training) != 0:
            bounding_boxes_training = normalize_bounding_boxes(bounding_boxes_training, resolution)
        if len(bounding_boxes_test) != 0:
            bounding_boxes_test = normalize_bounding_boxes(bounding_boxes_test, resolution)
        if len(bounding_boxes_validation) != 0:
            bounding_boxes_validation = normalize_bounding_boxes(bounding_boxes_validation, resolution)
    else:
        if len(bounding_boxes_training) != 0:
            bounding_boxes_training[:,1:] = bounding_boxes_training[:,1:]/np.array(original_resolution_training)[:,[0,1,0,1]]
        if len(bounding_boxes_test) != 0:
            bounding_boxes_test[:,1:] = bounding_boxes_test[:,1:]/np.array(original_resolution_test)[:,[0,1,0,1]]
        if len(bounding_boxes_validation) != 0:
            bounding_boxes_validation[:,1:] = bounding_boxes_validation[:,1:]/np.array(original_resolution_validation)[:,[0,1,0,1]]
    labels_training = np.array([int(i) for i in image_class_labels_training[:,1]])
    labels_test = np.array([int(i) for i in image_class_labels_test[:,1]])
    labels_validation = np.array([int(i) for i in image_class_labels_validation[:,1]])
    if len(bounding_boxes_training) != 0:
        bb_training = bounding_boxes_training[:,1:]
    else:
        bb_training = None
    if len(bounding_boxes_test) != 0:
        bb_test = bounding_boxes_test[:,1:]
    else:
        bb_test = None
    if len(bounding_boxes_validation) != 0:
        bb_validation = bounding_boxes_validation[:,1:]
    else:
        bb_validation = None
    return [images_training, labels_training, bb_training, images_test, labels_test, bb_test, images_validation, labels_validation, bb_validation]

def prepare_classification_nabirds(wd, **kwargs):
    """
    This function loads the labels and image paths of the nabirds dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    **kwargs : dict
        The kwargs are used to pass extra arguments to load_txt_files_cub.
        For details see the documentation on varargs params of load_txt_files_cub.

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
    bounding_boxes_fn = "bounding_boxes"
    classes_fn = "classes"
    image_class_labels_fn = "image_class_labels"
    images_fn = "images"
    train_test_split_fn = "train_test_split"
    if "bounding_boxes_fn" in kwargs.keys():
        bounding_boxes_fn = kwargs["bounding_boxes_fn"]
    if "classes_fn" in kwargs.keys():
        classes_fn = kwargs["classes_fn"]
    if "image_class_labels_fn" in kwargs.keys():
        image_class_labels_fn = kwargs["image_class_labels_fn"]
    if "images_fn" in kwargs.keys():
        images_fn = kwargs["images_fn"]
    if "train_test_split_fn" in kwargs.keys():
        train_test_split_fn = kwargs["train_test_split_fn"]
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_nabirds(wd, bounding_boxes_fn, classes_fn, image_class_labels_fn, images_fn, train_test_split_fn)
    [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation] = performe_train_test_validation_split_nabirds(bounding_boxes, image_class_labels, images_file, train_test_split)
    images_file_training = np.array([os.path.join(wd, "images", i[1]) for i in images_file_training])
    images_file_test = np.array([os.path.join(wd, "images", i[1]) for i in images_file_test])
    images_file_validation = np.array([os.path.join(wd, "images", i[1]) for i in images_file_validation])
    image_class_labels_training = np.array([int(i) for i in image_class_labels_training[:,1]])
    image_class_labels_test = np.array([int(i) for i in image_class_labels_test[:,1]])
    image_class_labels_validation = np.array([int(i) for i in image_class_labels_validation[:,1]])
    labels = np.unique(np.concatenate((image_class_labels_training, image_class_labels_test, image_class_labels_validation)))
    labels_artificial_labels_assignment = compute_label_assignment(labels)
    image_class_labels_training = convert_labels_to_artificial_labels(image_class_labels_training, labels_artificial_labels_assignment)
    image_class_labels_test = convert_labels_to_artificial_labels(image_class_labels_test, labels_artificial_labels_assignment)
    image_class_labels_validation = convert_labels_to_artificial_labels(image_class_labels_validation, labels_artificial_labels_assignment)
    return [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation]

def prepare_bb_estimation(wd):
    """
    This function loads the labels, bounding boxes and image paths of the nabirds dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.

    Returns
    -------
    list
        [images_file, image_class_labels, bounding_boxes, images_file_original].
        images_file : The image path of the data set.
        image_class_labels The labels of the data set.
        bounding_boxes : The bounding boxes of the data set.
        images_file_original : The original version of the images_file.

    """
    [bounding_boxes, classes, image_class_labels, images_file_original, train_test_split] = load_txt_files_nabirds(wd)
    images_file = np.array([os.path.join(wd, "images", i) for i in images_file_original[:,1]])
    image_class_labels = image_class_labels[:,1].astype(int)
    labels_artificial_labels_assignment = compute_label_assignment(np.unique(image_class_labels))
    image_class_labels = convert_labels_to_artificial_labels(image_class_labels, labels_artificial_labels_assignment)
    return [images_file, image_class_labels, bounding_boxes, images_file_original]
