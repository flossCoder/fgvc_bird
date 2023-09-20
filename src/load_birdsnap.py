#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_birdsnap.py
# Copyright (C) 2022 flossCoder
# 
# load_birdsnap is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_birdsnap is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Wed Mar  2 07:41:13 2022

@author: flossCoder
"""

import numpy as np
import os
from copy import deepcopy
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from image_processing import resize_bounding_boxes, resize_images, normalize_bounding_boxes
from convert_bounding_boxes import bb_conv_lucrlc_to_lucwh

def aux_load_file(file_dir, filename):
    """
    This auxiliary function loads the files.

    Parameters
    ----------
    file_dir : string
        The directory to the file.
    filename : string
        The name of the file.

    Returns
    -------
    list
        [header, content].
        header : The header of the file.
        content : The content of the file.

    """
    f = open(os.path.join(file_dir, filename), "r")
    lines = f.readlines()
    f.close()
    header = [i.replace("\n", "") for i in lines[0].split("\t")]
    content = []
    for l in lines[1:]:
        content.append([i.replace("\n", "") for i in l.split("\t")])
    return [header, np.array(content)]

def load_txt_files_birdsnap(wd, code_dir, data_dir, image_split_fn = None, bb_fn = None, all_ims_fn = "all-ims", images_fn = "images.txt", species_fn = "species.txt", test_images_fn = "test_images.txt", success_fn = "success.txt"):
    """
    This function loads the required files from the BirdSnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    data_dir : string
        The sub-directory of the data.
    image_split_fn : string, optional
        The name of the file containing the split into train, test and evaluation. The default is None.
    bb_fn : string, optional
        The bounding box annotation. The default is None.
    all_ims_fn : string, optional
        The list of all images in the dataset. The default is "all-ims".
    images_fn : string, optional
        The annotation of the images. The default is "images.txt".
    species_fn : string, optional
        The annotation of the species. The default is "species.txt".
    test_images_fn : string, optional
        The official test set. The default is "test_images.txt".
    success_fn : string, optional
        The list of images downloaded successfully. The default is "success.txt".

    Returns
    -------
    list
        [all_ims_content, images_header, images_content,
         species_header, species_content, test_images_header,
         test_images_content, success_content, image_split_content, bb_annotation].
        all_ims_content : A list of all images.
        images_header : The header of the images annotations.
        images_content : The content of the images annotations.
        species_header : The header of the species annotations.
        species_content : The content of the species annotations.
        test_images_header : The header of the test set.
        test_images_content : The content of the test set.
        success_content : The content of the images which have been successfully downloaded.
        image_split_content : The informations for train-, test- and validation split.
        bb_content : The bounding box annotation.

    """
    f = open(os.path.join(wd, code_dir, all_ims_fn), "r")
    lines = f.readlines()
    f.close()
    all_ims_content = [i.replace("\n", "") for i in lines]
    [images_header, images_content] = aux_load_file(os.path.join(wd, code_dir), images_fn)
    [species_header, species_content] = aux_load_file(os.path.join(wd, code_dir), species_fn)
    [test_images_header, test_images_content] = aux_load_file(os.path.join(wd, code_dir), test_images_fn)
    test_images_content = [i[0] for i in test_images_content]
    f = open(os.path.join(wd, data_dir, success_fn), "r")
    lines = f.readlines()
    f.close()
    success_content = np.array([i.replace("\n", "").split("\t") for i in lines])
    if image_split_fn == None:
        image_split_content = None
    else:
        f = open(os.path.join(wd, data_dir, image_split_fn), "r")
        lines = f.readlines()
        f.close
        image_split_content = np.array([i.replace("\n", "").split("\t") for i in lines])
    bb_content = None
    if bb_fn != None:
        f = open(os.path.join(wd, data_dir, bb_fn), "r")
        lines = f.readlines()
        f.close()
        bb_content = np.array([i.replace("\n", "").split("\t") for i in lines])
    return [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content]

def load_images_birdsnap(wd, data_dir, success_content):
    """
    This function loads the images from the BirdSnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    data_dir : string
        The sub-directory of the data.
    success_content : list
        The content of the images which have been successfully downloaded.

    Returns
    -------
    list
        [images, original_resolution]
        images_file : Each entry of the list contains an image in numpy format.
        original_resolution : Contains the original resolution of the image.

    """
    images = []
    original_resolution = []
    for i in success_content:
        image = Image.open(os.path.join(wd, data_dir, "images", i[1])).convert('RGB')
        images.append(np.array(image))
        original_resolution.append(image.size)
    return [images, original_resolution]

def aggregate_bounding_boxes_birdsnap(success_content, images_content):
    """
    This function aggregates the bounding boxes and convertes them into the right format.

    Parameters
    ----------
    success_content : list
        The content of the images which have been successfully downloaded.
    images_content : list
        The content of the images annotations.

    Returns
    -------
    numpy array
        The bounding boxes in the format left upper corner / width / height.

    """
    bounding_boxes = []
    for i in range(len(success_content)):
        bounding_boxes.append(bb_conv_lucrlc_to_lucwh([success_content[i,1]] + [int(j) for j in np.transpose(images_content[images_content[:,2] == success_content[i,1], 4:8])]))
    return np.array(bounding_boxes)

def performe_train_test_validation_split_birdsnap(species_content, images_content, success_content, test_images_content, image_split_content, bb_content = None):
    """
    This function splits the input data into training-, test- and validation set.

    Parameters
    ----------
    species_content : list
        The content of the species annotations.
    images_content : list
        The content of the images annotations.
    success_content : list
        The content of the images which have been successfully downloaded.
    test_images_content : list
        The content of the test set.
    image_split_content : list
        The informations for train-, test- and validation split.
    bb_content : numpy array, optinal
        The bounding box annotation. The default is None.

    Raises
    ------
    Exception
        The exception is raised in case the image_split_content contains an invalid split int (except [0, 2]).

    Returns
    -------
    list
        [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation,
         labels_training, labels_test, labels_validation, images_file_training,
         images_file_test, images_file_validation].
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
    if type(bb_content) != type(None):
        bounding_boxes = aggregate_bounding_boxes_birdsnap(success_content, images_content)
    else:
        bounding_boxes = bb_content
    bounding_boxes_training = []
    bounding_boxes_test = []
    bounding_boxes_validation = []
    image_class_labels_training = []
    image_class_labels_test = []
    image_class_labels_validation = []
    images_file_training = []
    images_file_test = []
    images_file_validation = []
    if type(image_split_content) != type(None):
        for i in range(len(image_split_content)):
            if image_split_content[i,1] == "0":
                if type(bb_content) != type(None):
                    bounding_boxes_training.append(bounding_boxes[bounding_boxes[:,0] == image_split_content[i,0]][0])
                image_class_labels_training.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],3])
                images_file_training.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],[0,2]])
            elif image_split_content[i,1] == "1":
                if type(bb_content) != type(None):
                    bounding_boxes_test.append(bounding_boxes[bounding_boxes[:,0] == image_split_content[i,0]][0])
                image_class_labels_test.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],3])
                images_file_test.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],[0,2]])
            elif image_split_content[i,1] == "2":
                if type(bb_content) != type(None):
                    bounding_boxes_validation.append(bounding_boxes[bounding_boxes[:,0] == image_split_content[i,0]][0])
                image_class_labels_validation.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],3])
                images_file_validation.append(images_content[np.where(images_content[:,2] == image_split_content[i,0])[0][0],[0,2]])
            else:
                raise Exception("Invalid split %s for image id %s"%(image_split_content[i,1], image_split_content[i,0]))
    else:
        success_content_copy = deepcopy(success_content)
        for i in test_images_content:
            if i in success_content[:,1]:
                if type(bb_content) != type(None):
                    bounding_boxes_test.append(bounding_boxes[bounding_boxes[:,0] == i][0])
                index = np.where(images_content[:,2] == i)[0][0]
                image_class_labels_test.append(images_content[index,3])
                images_file_test.append(images_content[index,[0,2]])
                success_content_copy = np.delete(success_content_copy, np.where(success_content_copy[:,1] == i)[0][0], 0)
        for i in range(len(success_content_copy)):
            if type(bb_content) != type(None):
                bounding_boxes_training.append(bounding_boxes[bounding_boxes[:,0] == success_content_copy[i,1]][0])
            image_class_labels_training.append(images_content[np.where(images_content[:,2] == success_content_copy[i,1])[0][0],3])
            images_file_training.append(success_content_copy[i])
    return [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation]

def prepare_train_test_birdsnap(wd, code_dir, data_dir, image_split_fn, bb_fn = None, resolution = None):
    """
    This function prepares everything for training and testing.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    data_dir : string
        The sub-directory of the data.
    image_split_fn : string
        The name of the file containing the split into train, test and evaluation.
    bb_fn : string, optional
        The bounding box annotation. The default is None.
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
    [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content] = load_txt_files_birdsnap(wd, code_dir, data_dir, image_split_fn, bb_fn)
    [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation] = performe_train_test_validation_split_birdsnap(species_content, images_content, success_content, test_images_content, image_split_content, bb_content)
    [images_training, original_resolution_training] = load_images_birdsnap(wd, data_dir, images_file_training)
    [images_test, original_resolution_test] = load_images_birdsnap(wd, data_dir, images_file_test)
    [images_validation, original_resolution_validation] = load_images_birdsnap(wd, data_dir, images_file_validation)
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
    labels_training = np.array([int(i) for i in image_class_labels_training])
    labels_test = np.array([int(i) for i in image_class_labels_test])
    labels_validation = np.array([int(i) for i in image_class_labels_validation])
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

def prepare_classification_birdsnap(wd, code_dir, data_dir, image_split_fn = None, success_fn = "success.txt", test_images_fn = "test_images.txt"):
    """
    This function loads the labels and image paths of the birdsnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    data_dir : string
        The sub-directory of the data.
    image_split_fn : string, optional
        The name of the file containing the split into train, test and evaluation. The default is None.
    success_fn : string, optional
        The list of images downloaded successfully. The default is "success.txt".
    test_images_fn : string, optional
        The official test set. The default is "test_images.txt".
    
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
    bb_fn = None
    all_ims_fn = "all-ims"
    images_fn = "images.txt"
    species_fn = "species.txt"
    [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content] = load_txt_files_birdsnap(wd, code_dir, data_dir, image_split_fn, bb_fn, all_ims_fn, images_fn, species_fn, test_images_fn, success_fn)
    [bounding_boxes_training, bounding_boxes_test, bounding_boxes_validation, image_class_labels_training, image_class_labels_test, image_class_labels_validation, images_file_training, images_file_test, images_file_validation] = performe_train_test_validation_split_birdsnap(species_content, images_content, success_content, test_images_content, image_split_content, bb_content)
    images_file_training = np.array([os.path.join(wd, data_dir, "images", i[1]) for i in images_file_training])
    images_file_test = np.array([os.path.join(wd, data_dir, "images", i[1]) for i in images_file_test])
    images_file_validation = np.array([os.path.join(wd, data_dir, "images", i[1]) for i in images_file_validation])
    image_class_labels_training = np.array([int(i) for i in image_class_labels_training])
    image_class_labels_test = np.array([int(i) for i in image_class_labels_test])
    image_class_labels_validation = np.array([int(i) for i in image_class_labels_validation])
    return [images_file_training, image_class_labels_training, images_file_test, image_class_labels_test, images_file_validation, image_class_labels_validation]

def prepare_bb_estimation(wd, code_dir, data_dir, image_split_fn = None):
    """
    This function loads the labels, bounding boxes and image paths of the birdsnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    data_dir : string
        The sub-directory of the data.
    image_split_fn : string, optional
        The name of the file containing the split into train, test and evaluation. The default is None.

    Returns
    -------
    list
        [images_file, image_class_labels, bounding_boxes]
        images_file : The image path of the data set.
        image_class_labels : The labels of the data set.
        bounding_boxes : The bounding boxes of the data set.

    """
    [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content] = load_txt_files_birdsnap(wd, code_dir, data_dir, image_split_fn)
    images_file = []
    image_class_labels = []
    bounding_boxes = aggregate_bounding_boxes_birdsnap(success_content, images_content)
    for i in success_content[:,1]:
        images_file.append(os.path.join(wd, data_dir, "images", i))
        image_class_labels.append(int(images_content[images_content[:,2] == i][0,3]))
    images_file = np.array(images_file)
    image_class_labels = np.array(image_class_labels)
    return [images_file, image_class_labels, bounding_boxes]

def aux_generate_label_ids(wd, code_dir, data_dir, success_fn = "success.txt", image_split_fn = None, include_image_labels = None, images_fn = "images.txt"):
    """
    This function aggregates the labels.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    data_dir : string
        The sub-directory of the data.
        The sub-directory of the data.
    success_fn : TYPE, optional
        The list of images downloaded successfully. The default is "success.txt".
    image_split_fn : string, optional
        The name of the file containing the split into train, test and evaluation.
        If None is given all labels will be used. The default is None.
    include_image_labels : int or list of ints, optional
        This parameter states whicht label types defined in the image_split_file
        shall be included. If None is given, all will be included. The default is None.
    images_fn : string, optional
        The annotation of the images. The default is "images.txt".

    Returns
    -------
    A numpy-array containing a mapping of filenames (first column), the label (second column)
    and split (if possible, otherwise the third column is zero).

    """
    [images_header, images_content] = aux_load_file(os.path.join(wd, code_dir), images_fn)
    f = open(os.path.join(wd, data_dir, success_fn), "r")
    lines = f.readlines()
    f.close()
    success_content = np.array([i.replace("\n", "").split("\t") for i in lines])
    if type(image_split_fn) == type(None):
        image_split_content = None
        result = np.zeros([np.shape(success_content)[0], 3], dtype = images_content.dtype)
        for i in range(len(success_content)):
            result[i, 0] = success_content[i, 1]
            result[i, 1] = images_content[success_content[i, 1] == images_content[:,2], 3][0]
    else:
        f = open(os.path.join(wd, data_dir, image_split_fn), "r")
        lines = f.readlines()
        f.close
        image_split_content = np.array([i.replace("\n", "").split("\t") for i in lines])
        if type(include_image_labels) == type(None):
            result = np.zeros([np.shape(image_split_content)[0], 3], dtype = images_content.dtype)
            for i in range(len(image_split_content)):
                result[i, 0] = image_split_content[i, 0]
                result[i, 1] = images_content[image_split_content[i, 0] == images_content[:,2], 3][0]
                result[i, 2] = image_split_content[i, 1]
        else:
            include_image_labels = include_image_labels if type(include_image_labels) == type([]) else [include_image_labels]
            result = np.zeros([sum([sum(image_split_content[:, 1] == str(i)) for i in include_image_labels]), 3], dtype = images_content.dtype)
            count = 0
            for l in include_image_labels:
                for i in image_split_content[image_split_content[:, 1] == str(l), 0]:
                    result[count, 0] = i
                    result[count, 1] = images_content[i == images_content[:,2], 3][0]
                    result[count, 2] = l
                    count += 1
    return result
