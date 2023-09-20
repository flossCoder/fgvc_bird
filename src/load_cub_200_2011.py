#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_cub_200_2011.py
# Copyright (C) 2022 flossCoder
# 
# load_cub_200_2011 is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_cub_200_2011 is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jan 31 10:53:35 2022

@author: flossCoder
"""

import numpy as np
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from image_processing import resize_bounding_boxes, resize_images, normalize_bounding_boxes

def load_txt_files_cub(wd, bounding_boxes_fn = "bounding_boxes", classes_fn = "classes", image_class_labels_fn = "image_class_labels", images_fn = "images", train_test_split_fn = "train_test_split"):
    """
    This function loads the txt files from the CUB-200-2011 dataset.

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
        [bounding_boxes, classes, image_class_labels, images, train_test_split].
        bounding_boxes : The bounding box annotations.
        classes : The class annotations.
        image_class_labels : The image to class assignments.
        images_file : The image filename assignment.
        train_test_split : The train and test split.

    """
    bounding_boxes = np.loadtxt(os.path.join(wd, "%s.txt"%bounding_boxes_fn))
    classes = np.loadtxt(os.path.join(wd, "%s.txt"%classes_fn), dtype="str")
    image_class_labels = np.loadtxt(os.path.join(wd, "%s.txt"%image_class_labels_fn))
    images_file = np.loadtxt(os.path.join(wd, "%s.txt"%images_fn), dtype="str")
    train_test_split = np.loadtxt(os.path.join(wd, "%s.txt"%train_test_split_fn))
    return [bounding_boxes, classes, image_class_labels, images_file, train_test_split]

def load_images_cub(wd, images_file):
    """
    This function loads the images from the CUB-200-2011 dataset.

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
        images : Each entry of the list contains an image in numpy format.
        original_resolution : Contains the original resolution of the image.

    """
    images = []
    original_resolution = []
    for i in images_file:
        image = Image.open(os.path.join(wd, "images", i[1])).convert('RGB')
        images.append(np.array(image))
        original_resolution.append(image.size)
    return [images, original_resolution]

def performe_train_test_split_cub(bounding_boxes, image_class_labels, images_file, train_test_split):
    """
    This function splits the input data into training- and test set.

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
        [bounding_boxes_training, bounding_boxes_test,
         image_class_labels_training, image_class_labels_test,
         images_file_training, images_file_test].
        bounding_boxes_training : The bounding box annotations for training.
        bounding_boxes_test : The bounding box annotations for testing.
        image_class_labels_training : The image to class assignments for training.
        image_class_labels_test : The image to class assignments for testing.
        images_file_training : The image filename assignment for training.
        images_file_test : The image filename assignment for testing.

    """
    number_of_trainings_samples = sum(train_test_split[:,1] == 0)
    number_of_test_samples = sum(train_test_split[:,1] == 1)
    bounding_boxes_training = np.zeros((number_of_trainings_samples, np.shape(bounding_boxes)[1]))
    bounding_boxes_test = np.zeros((number_of_test_samples, np.shape(bounding_boxes)[1]))
    image_class_labels_training = np.zeros((number_of_trainings_samples, np.shape(image_class_labels)[1]))
    image_class_labels_test = np.zeros((number_of_test_samples, np.shape(image_class_labels)[1]))
    images_file_training = np.zeros((number_of_trainings_samples, np.shape(images_file)[1]), dtype = images_file.dtype)
    images_file_test = np.zeros((number_of_test_samples, np.shape(images_file)[1]), dtype = images_file.dtype)
    trainings_count = 0
    test_count = 0
    for i in range(np.shape(train_test_split)[0]):
        image_id = train_test_split[i,0]
        if train_test_split[i,1] == 0:
            if bounding_boxes[i,0] == image_id:
                bounding_boxes_training[trainings_count,:] = bounding_boxes[i,:]
            else:
                bounding_boxes_training[trainings_count,:] = bounding_boxes[(bounding_boxes[:,0]==image_id),:]
            if image_class_labels[i,0] == image_id:
                image_class_labels_training[trainings_count,:] = image_class_labels[i,:]
            else:
                image_class_labels_training[trainings_count,:] = image_class_labels[(image_class_labels[:,0]==image_id),:]
            if str(int(image_id)) == images_file[i,0]:
                images_file_training[trainings_count,:] = images_file[i,:]
            else:
                images_file_training[trainings_count,:] = images_file[(images_file[:,0]==str(int(image_id))),:]
            trainings_count += 1
        elif train_test_split[i,1] == 1:
            if bounding_boxes[i,0] == image_id:
                bounding_boxes_test[test_count,:] = bounding_boxes[i,:]
            else:
                bounding_boxes_test[test_count,:] = bounding_boxes[(bounding_boxes[:,0]==image_id),:]
            if image_class_labels[i,0] == image_id:
                image_class_labels_test[test_count,:] = image_class_labels[i,:]
            else:
                image_class_labels_test[test_count,:] = image_class_labels[(image_class_labels[:,0]==image_id),:]
            if str(int(image_id)) == images_file[i,0]:
                images_file_test[test_count,:] = images_file[i,:]
            else:
                images_file_test[test_count,:] = images_file[(images_file[:,0]==str(int(image_id))),:]
            test_count += 1
        else:
            raise Exception("Invalid train test split class %i for id %i"%(int(train_test_split[i,1]), int(train_test_split[i,0])))
    return [bounding_boxes_training, bounding_boxes_test, image_class_labels_training, image_class_labels_test, images_file_training, images_file_test]

def prepare_train_test_cub(wd, resolution = None):
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
        [images_training, labels_training, bb_training, images_test, labels_test, bb_test].
        images_training : Images for training.
        labels_training : Labels for training.
        bb_training : Bounding Boxes for training.
        images_test : Images for testing.
        labels_test : Labels for testing.
        bb_test : Bounding Boxes for testing.

    """
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_cub(wd)
    [bounding_boxes_training, bounding_boxes_test, image_class_labels_training, image_class_labels_test, images_file_training, images_file_test] = performe_train_test_split_cub(bounding_boxes, image_class_labels, images_file, train_test_split)
    [images_training, original_resolution_training] = load_images_cub(wd, images_file_training)
    [images_test, original_resolution_test] = load_images_cub(wd, images_file_test)
    if resolution != None:
        images_training = resize_images(images_training, resolution)
        images_test = resize_images(images_test, resolution)
        bounding_boxes_training = resize_bounding_boxes(bounding_boxes_training, original_resolution_training, resolution)
        bounding_boxes_training = normalize_bounding_boxes(bounding_boxes_training, resolution)
        bounding_boxes_test = resize_bounding_boxes(bounding_boxes_test, original_resolution_test, resolution)
        bounding_boxes_test = normalize_bounding_boxes(bounding_boxes_test, resolution)
    else:
        bounding_boxes_training[:,1:] = bounding_boxes_training[:,1:]/np.array(original_resolution_training)[:,[0,1,0,1]]
        bounding_boxes_test[:,1:] = bounding_boxes_test[:,1:]/np.array(original_resolution_test)[:,[0,1,0,1]]
    labels_training = image_class_labels_training[:,1]
    labels_test = image_class_labels_test[:,1]
    bb_training = bounding_boxes_training[:,1:]
    bb_test = bounding_boxes_test[:,1:]
    return [images_training, labels_training, bb_training, images_test, labels_test, bb_test]

def prepare_classification(wd, **kwargs):
    """
    This function loads the labels and image paths of the CUB 200 2011 dataset.

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
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_cub(wd, bounding_boxes_fn, classes_fn, image_class_labels_fn, images_fn, train_test_split_fn)
    [bounding_boxes_training, bounding_boxes_test, image_class_labels_training, image_class_labels_test, images_file_training, images_file_test] = performe_train_test_split_cub(bounding_boxes, image_class_labels, images_file, train_test_split)
    images_file_training = np.array([os.path.join(wd, "images", i[1]) for i in images_file_training])
    images_file_test = np.array([os.path.join(wd, "images", i[1]) for i in images_file_test])
    images_file_validation = None
    image_class_labels_training = image_class_labels_training[:,1]
    image_class_labels_test = image_class_labels_test[:,1]
    image_class_labels_validation = None
    return [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation]

def prepare_bb_estimation(wd):
    """
    This function loads the labels, bounding boxes and image paths of the CUB 200 2011 dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.

    Returns
    -------
    list
        [images_file, image_class_labels, bounding_boxes, images_file_original].
        images_file : The image path of the data set.
        image_class_labels : The labels of the data set.
        bounding_boxes : The bounding boxes of the data set.
        images_file_original : The original version of the images_file.

    """
    [bounding_boxes, classes, image_class_labels, images_file_original, train_test_split] = load_txt_files_cub(wd)
    images_file = np.array([os.path.join(wd, "images", i[1]) for i in images_file_original])
    return [images_file, image_class_labels, bounding_boxes, images_file_original]
