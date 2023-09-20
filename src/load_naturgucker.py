#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# load_naturgucker.py
# Copyright (C) 2022 flossCoder
# 
# load_naturgucker is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# load_naturgucker is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Feb  1 09:18:04 2022

@author: flossCoder
"""

import os
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from bilder_download import import_voegel_csv
from image_processing import resize_images

def load_images_naturgucker(wd, data, directory_name = None):
    """
    This function loads the images from the naturgucker (sub-)dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    data : numpy array
        Contains all the bird species and file assignments as numpy array.
    directory_name : string, optional
        The name of the subdirectory, where the data shall be saved, if None, save in wd.

    Raises
    ------
    Exception
        The exception is raised in case an image cannot be loaded.

    Returns
    -------
    list
        [images, original_resolution]
        images_file : Each entry of the list contains an image in numpy format.
        original_resolution : Contains the original resolution of the image.

    """
    if directory_name != None and directory_name not in os.listdir(wd):
        raise Exception("Invalid directory name: %s"%directory_name)
    order_wd = os.path.join(wd, directory_name) if directory_name != None else wd
    images = []
    original_resolution = []
    for [order, family, genus, species, bird_id, bird_url] in data:
        try:
            order = order.replace(".", "_").replace("/", "__").replace(" x ?", "")
            family = family.replace(".", "_").replace("/", "__").replace(" x ?", "")
            genus = genus.replace(".", "_").replace("/", "__").replace(" x ?", "")
            species = species.replace(".", "_").replace("/", "__").replace(" x ?", "")
            figure = "%s.%s"%(bird_id, bird_url.split(".")[-1])
            family_wd = os.path.join(order_wd, order)
            genus_wd = os.path.join(family_wd, family)
            species_wd = os.path.join(genus_wd, genus)
            figure_wd = os.path.join(species_wd, species)
            image = Image.open(os.path.join(figure_wd, figure)).convert('RGB')
            images.append(image)
            original_resolution.append(image.size)
        except Exception as ex:
            if order not in os.listdir(order_wd):
                raise Exception("Order %s does not exist directory %s"%(order, order_wd))
            elif family not in os.listdir(family_wd):
                raise Exception("Family %s does not exist directory %s"%(family, family_wd))
            elif genus not in os.listdir(genus_wd):
                raise Exception("Genus %s does not exist directory %s"%(genus, genus_wd))
            elif species not in os.listdir(species_wd):
                raise Exception("Species %s does not exist directory %s"%(species, species_wd))
            elif figure not in os.listdir(figure_wd):
                raise Exception("Figure %s does not exist directory %s"%(figure, figure_wd))
            else:
                raise Exception("An unknown error occured for bird %s"%bird_id)
    return images

def prepare_train_test_naturgucker(wd, filename_training, filename_test, directory_name = None, resolution = None, data_index = 3):
    """
    This function prepares everything for training and testing.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename_training : string
        The filename describes the name of the csv file containing the required data for training.
    filename_test : string
        The filename describes the name of the csv file containing the required data for testing.
    directory_name : string, optional
        The name of the subdirectory, where the data shall be saved, if None, save in wd. The default is None.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.
    data_index : int, optional
        The column used for labeling (0 = Ordnung, 1 = Familie, 2 = Gattung, 3 = Art). The default is 3.

    Returns
    -------
    list
        [images_training, labels_training, images_test, labels_test].
        images_training : Images for training.
        labels_training : Labels for training.
        images_test : Images for testing.
        labels_test : Labels for testing.

    """
    [header_training, data_training] = import_voegel_csv(wd, filename_training)
    [header_test, data_test] = import_voegel_csv(wd, filename_test)
    images_training = load_images_naturgucker(wd, data_training, directory_name)
    images_test = load_images_naturgucker(wd, data_test, directory_name)
    if resolution != None:
        images_training = resize_images(images_training, resolution)
        images_test = resize_images(images_test, resolution)
    labels_training = data_training[:,data_index]
    labels_test = data_test[:,data_index]
    return [images_training, labels_training, images_test, labels_test]

def prepare_classification_naturgucker(wd, filename_training, filename_test = None, filename_validation = None, directory_name = None, data_index = 3):
    """
    This function loads the labels and image paths of the naturgucker.de dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename_training : string
        The filename describes the name of the csv file containing the required data for training.
    filename_test : string, optional
        The filename describes the name of the csv file containing the required data for testing. The default is None.
    filename_validation : string, optional
        The filename describes the name of the csv file containing the required data for validation. The default is None.
    directory_name : string, optional
        The name of the subdirectory, where the data shall be saved, if None, save in wd. The default is None.
    data_index : int, optional
        The column used for labeling (0 = Ordnung, 1 = Familie, 2 = Gattung, 3 = Art). The default is 3.

    Returns
    -------
    list
        [images_file_training, image_class_labels_training, ids_training, images_file_test, image_class_labels_test, ids_test, images_file_validation, image_class_labels_validation, ids_validation].
        images_file_training : The image paths of the trainings set.
        image_class_labels_training : The labels of the trainings set.
        ids_training : The image ids of the trainings set.
        images_file_test : The image paths of the test set.
        image_class_labels_test : The labels of the test set.
        ids_test : The image ids of the test set.
        images_file_validation : The image paths of the validation set.
        image_class_labels_validation : The labels of the validation set.
        ids_validation : The image ids of the validation set.

    """
    [header_training, data_training] = import_voegel_csv(wd, filename_training)
    if type(filename_test) != type(None):
        [header_test, data_test] = import_voegel_csv(wd, filename_test)
    else:
        data_test = []
    if type(filename_validation) != type(None):
        [header_validation, data_validation] = import_voegel_csv(wd, filename_test)
    else:
        data_validation = []
    if type(directory_name) == type(None):
        images_file_training = np.array([os.path.join(wd, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_training])
        images_file_test = np.array([os.path.join(wd, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_test])
        images_file_validation = np.array([os.path.join(wd, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_validation])
    else:
        images_file_training = np.array([os.path.join(wd, directory_name, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_training])
        images_file_test = np.array([os.path.join(wd, directory_name, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_test])
        images_file_validation = np.array([os.path.join(wd, directory_name, i[0].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[1].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[2].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[3].replace(".", "_").replace("/", "__").replace(" x ?", ""), "%s.%s"%(i[4].replace(".", "_").replace("/", "__").replace(" x ?", ""), i[5].replace("/", "__").replace(" x ?", "").split(".")[-1])) for i in data_validation])
    if type(filename_validation) != type(None) and type(filename_test) != type(None):
        labels = np.sort(np.unique(np.concatenate((data_training[:,data_index], data_test[:,data_index], data_validation[:,data_index]))))
        image_class_labels_validation = np.searchsorted(labels, data_validation[:,data_index])
        image_class_labels_test = np.searchsorted(labels, data_test[:,data_index])
        ids_validation = data_validation[:,4]
        ids_test = data_test[:,4]
    elif type(filename_test) != type(None):
        labels = np.sort(np.unique(np.concatenate((data_training[:,data_index], data_test[:,data_index]))))
        image_class_labels_validation = None
        image_class_labels_test = np.searchsorted(labels, data_test[:,data_index])
        ids_validation = None
        ids_test = data_test[:,4]
    elif type(filename_validation) != type(None):
        labels = np.sort(np.unique(np.concatenate((data_training[:,data_index], data_validation[:,data_index]))))
        image_class_labels_validation = np.searchsorted(labels, data_validation[:,data_index])
        image_class_labels_test = None
        ids_validation = data_validation[:,4]
        ids_test = None
    else:
        labels = np.sort(np.unique(data_training[:,data_index]))
        image_class_labels_validation = None
        image_class_labels_test = None
        ids_validation = None
        ids_test = None
    image_class_labels_training = np.searchsorted(labels, data_training[:,data_index])
    ids_training = data_training[:,4]
    return [images_file_training, image_class_labels_training, ids_training, images_file_test, image_class_labels_test, ids_test, images_file_validation, image_class_labels_validation, ids_validation]
