#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# image_processing.py
# Copyright (C) 2022 flossCoder
# 
# image_processing is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# image_processing is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Feb  4 09:43:54 2022

@author: flossCoder
"""

import numpy as np
from PIL import Image
from copy import deepcopy

def resize_images(images, resolution = None):
    """
    This function resizes the images.

    Parameters
    ----------
    images : list of numpy arrays
        Each entry of the list contains an image.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.

    Returns
    -------
    list of numpy arrays
        The resized images.

    """
    if resolution != None and type(resolution) != int and type(resolution) != float and len(resolution) == 2:
        for i in range(len(images)):
            if type(images[i]) == type(np.array([])):
                images[i] = np.array(Image.fromarray(images[i]).resize(resolution))
            else:
                images[i] = images[i].resize(resolution)
    elif resolution != None and (type(resolution) == int or type(resolution) == float):
        for i in range(len(images)):
            if type(images[i]) == type(np.array([])):
                image = Image.fromarray(images[i])
                images[i] = np.array(image.resize((int(image.size[0] * resolution), int(image.size[1] * resolution))))
            else:
                images[i] = images[i].resize((int(images[i].size[0] * resolution), int(images[i].size[1] * resolution)))
    elif resolution != None and len(resolution) == 1:
        for i in range(len(images)):
            if type(images[i]) == type(np.array([])):
                image = Image.fromarray(images[i])
                images[i] = np.array(image.resize((int(image.size[0] * resolution[0]), int(image.size[1] * resolution[0]))))
            else:
                images[i] = images[i].resize((int(images[i].size[0] * resolution[0]), int(images[i].size[1] * resolution[0])))
    return images

def resize_bounding_boxes(bounding_boxes, original_resolution, resolution = None):
    """
    This function resizes the bounding boxes.

    Parameters
    ----------
    bounding_boxes : numpy array
        The bounding box annotations.
    original_resolution : numpy array
        Contains the original resolution of the image.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.

    Returns
    -------
    bounding_boxes : numpy array
        The resized bounding box annotations.

    """
    if resolution != None and type(resolution) != int and type(resolution) != float and len(resolution) == 2:
        for i in range(len(bounding_boxes)):
            bounding_boxes[i][1] = int(bounding_boxes[i][1]*resolution[0]/original_resolution[i][0])
            bounding_boxes[i][2] = int(bounding_boxes[i][2]*resolution[1]/original_resolution[i][1])
            bounding_boxes[i][3] = int(bounding_boxes[i][3]*resolution[0]/original_resolution[i][0])
            bounding_boxes[i][4] = int(bounding_boxes[i][4]*resolution[1]/original_resolution[i][1])
    elif resolution != None and (type(resolution) == int or type(resolution) == float):
        for i in range(len(bounding_boxes)):
            bounding_boxes[i][1] = int(bounding_boxes[i][1]*resolution)
            bounding_boxes[i][2] = int(bounding_boxes[i][2]*resolution)
            bounding_boxes[i][3] = int(bounding_boxes[i][3]*resolution)
            bounding_boxes[i][4] = int(bounding_boxes[i][4]*resolution)
    elif resolution != None and len(resolution) == 1:
        for i in range(len(bounding_boxes)):
            bounding_boxes[i][1] = int(bounding_boxes[i][1]*resolution[0])
            bounding_boxes[i][2] = int(bounding_boxes[i][2]*resolution[0])
            bounding_boxes[i][3] = int(bounding_boxes[i][3]*resolution[0])
            bounding_boxes[i][4] = int(bounding_boxes[i][4]*resolution[0])
    return bounding_boxes

def normalize_bounding_boxes(bounding_boxes, resolution = None):
    """
    This function normalizes the bounding boxes.

    Parameters
    ----------
    bounding_boxes : numpy array
        The bounding box annotations.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.

    Returns
    -------
    bounding_boxes : numpy array
        The normalized bounding box annotations.

    """
    if resolution != None and type(resolution) != int and type(resolution) != float and len(resolution) == 2:
        bounding_boxes[:,1:] = bounding_boxes[:,1:]/np.array(resolution)[[0,1,0,1]]
    elif resolution != None and (type(resolution) == int or type(resolution) == float):
         bounding_boxes[:,1:] = bounding_boxes[:,1:]/resolution
    elif resolution != None and (type(resolution) == int or type(resolution) == float):
        bounding_boxes[:,1:] = bounding_boxes[:,1:]/resolution[0]
    return bounding_boxes

def coordinates_crop_image_at_bounding_box(image, bounding_box, resolution, apply_centered_zero_padding = False):
    """
    This function calculates the part of the image that has to be taken out, plus the bounding boxes.

    Parameters
    ----------
    image : PIL or numpy array
        The image.
    bounding_box : numpy array
        The bounding box annotations.
    resolution : tuple
        The resolution describes the output size of an image.
    apply_centered_zero_padding : boolean, optional
        If True enlarge the bounding box allways symmetrically, otherwise enlarge the bounding box optimally. The default is False.

    Returns
    -------
    list
        [new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h].
        new_x : x coordinate of the upper left corner of the crop box.
        new_y : y coordinate of the upper left corner of the crop box.
        new_w : widht of the crop box.
        new_h : hight of the crop box.
        bb_x : x coordinate of the upper left corner of the bounding box on the cropped image.
        bb_y : y coordinate of the upper left corner of the bounding box on the cropped image.
        bb_w : widht of the bounding box on the cropped image.
        bb_h : hight of the bounding box on the cropped image.

    """
    if type(image) == type(np.array([])):
        image = Image.fromarray(image)
    desired_relation = resolution[0] / resolution[1]
    current_relation = bounding_box[3] / bounding_box[4]
    if current_relation > desired_relation:
        # adapt box in y direction
        delta = bounding_box[3] / desired_relation - bounding_box[4]
        new_x = bounding_box[1]
        new_w = bounding_box[3]
        new_h = bounding_box[4] + np.ceil(delta)
        if apply_centered_zero_padding or (bounding_box[2] >= np.ceil(delta/2) and (image.size[1] - bounding_box[4] - bounding_box[2]) >= np.ceil(delta/2)):
            # simple case: the new box can be enlarged in both vertical directions
            new_y = bounding_box[2] - np.floor(delta/2)
        elif (image.size[1] - bounding_box[4]) > np.ceil(delta):
            # the box fits on the image, but cannot be enlarged symetrically
            if bounding_box[2] < (image.size[1] - bounding_box[2] - bounding_box[4]):
                # fit the box to the upper line
                new_y = 0
            else:
                # fit the box to the lower line
                new_y = image.size[1] - new_h - 1
        else:
            # the box does not fit on the image and must be optimally lokalized for symetrical zero padding
            new_y = np.ceil((image.size[1] - new_h) / 2)
    elif current_relation < desired_relation:
        # adapt box in x direction
        delta = bounding_box[4] * desired_relation - bounding_box[3]
        new_y = bounding_box[2]
        new_w = bounding_box[3] + np.ceil(delta)
        new_h = bounding_box[4]
        if apply_centered_zero_padding or (bounding_box[1] >= np.ceil(delta/2) and (image.size[0] - bounding_box[3] - bounding_box[1]) >= np.ceil(delta/2)):
            # simple case: the new box can be enlarged in both horizontal directions
            new_x = bounding_box[1] - np.floor(delta/2)
        elif (image.size[0] - bounding_box[3]) > np.ceil(delta):
            # the box fits on the image, but cannot be enlarged symetrically
            if bounding_box[1] < (image.size[0] - bounding_box[1] - bounding_box[3]):
                # fit the box to the left line
                new_x = 0
            else:
                # fit the box to the right line
                new_x = image.size[0] - new_w - 1
        else:
            # the box does not fit on the image and must be optimally lokalized for symetrical zero padding
            new_x = np.ceil((image.size[0] - new_w) / 2)
    else:
        # the resolution of the box fits
        new_x = bounding_box[1]
        new_y = bounding_box[2]
        new_w = bounding_box[3]
        new_h = bounding_box[4]
    bb_x = np.round((bounding_box[1] - new_x) * (resolution[0] / new_w))
    bb_y = np.round((bounding_box[2] - new_y) * (resolution[1] / new_h))
    bb_w = np.round(bounding_box[3] * (resolution[0] / new_w))
    bb_h = np.round(bounding_box[4] * (resolution[1] / new_h))
    return [new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h]

def crop_images_at_bounding_boxes(images, bounding_boxes, resolution):
    """
    This function obtains the image crops at the bounding box, where the aspect ratio is maintained.

    Parameters
    ----------
    images : list of numpy arrays
        Each entry of the list contains an image.
    bounding_boxes : numpy array
        The bounding box annotations.
    resolution : tuple
        The resolution describes the output size of an image.

    Returns
    -------
    list
        [new_images, new_bounding_boxes].
        new_images : The cropped images.
        new_bounding_boxes : The bounding box coordinates of the new images.

    """
    new_images = []
    new_bounding_boxes = []
    for i in range(len(images)):
        if type(images[i]) == type(np.array([])):
            image = Image.fromarray(images[i])
        else:
            image = images[i]
        [new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(image, bounding_boxes[i], resolution)
        new_images.append(image.crop((new_x, new_y, new_x+new_w, new_y+new_h)).resize(resolution))
        new_bounding_boxes.append([bounding_boxes[i][0]] + [bb_x, bb_y, bb_w, bb_h])
    return [new_images, new_bounding_boxes]

def generate_bounding_boxes_crops(images, bounding_boxes, resolution):
    """
    This function obtains the image crops at the bounding box, where the aspect ratio is not maintained.

    Parameters
    ----------
    images : list of numpy arrays
        Each entry of the list contains an image.
    bounding_boxes : numpy array
        The bounding box annotations.
    resolution : tuple
        The resolution describes the output size of an image.

    Returns
    -------
    list
        [new_images, new_bounding_boxes].
        new_images : The cropped images.
        new_bounding_boxes : The bounding box coordinates of the new images.

    """
    new_images = []
    new_bounding_boxes = []
    for i in range(len(images)):
        if type(images[i]) == type(np.array([])):
            image = Image.fromarray(images[i])
        else:
            image = images[i]
        i_id, bb_x, bb_y, bb_w, bb_h = bounding_boxes[i]
        new_images.append(image.crop((bb_x, bb_y, bb_x + bb_w, bb_y + bb_h)).resize(resolution))
        new_bounding_boxes.append([i_id, 0, 0] + list(resolution))
    return [new_images, new_bounding_boxes]

def generate_bounding_boxes_crops_with_zero_padding(images, bounding_boxes, resolution):
    """
    This function obtains the image crops at the bounding box, where the aspect ratio is not maintained and the unnacessary pixels are zero padded.

    Parameters
    ----------
    images : list of numpy arrays
        Each entry of the list contains an image.
    bounding_boxes : numpy array
        The bounding box annotations.
    resolution : tuple
        The resolution describes the output size of an image.

    Returns
    -------
    list
        [new_images, new_bounding_boxes].
        new_images : The cropped images.
        new_bounding_boxes : The bounding box coordinates of the new images.

    """
    new_images = []
    new_bounding_boxes = []
    for i in range(len(images)):
        if type(images[i]) == type(np.array([])):
            image = Image.fromarray(images[i])
        else:
            image = images[i]
        [new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(image, bounding_boxes[i], resolution, True)
        image = image.crop((new_x, new_y, new_x+new_w, new_y+new_h)).resize(resolution)
        image = np.array(image)
        aux = deepcopy(image[int(bb_y) : int(bb_y + bb_h), int(bb_x) : int(bb_x + bb_w)])
        image *= 0
        image[int(bb_y) : int(bb_y + bb_h), int(bb_x) : int(bb_x + bb_w)] = aux
        image = Image.fromarray(image)
        new_images.append(image)
        new_bounding_boxes.append([bounding_boxes[i][0]] + [bb_x, bb_y, bb_w, bb_h])
    return [new_images, new_bounding_boxes]

def load_images(images_files, resolution = None):
    """
    This function loads the images from the given path.

    Parameters
    ----------
    images_files : numpy array
        The paths of the images.
   resolution : tuple or float, optional
       The resolution describes the output size of an image. If resolution
       contains a tuple with two ints, they are used as resolution. If it
       contains a float, the number is considered as scaling factor preserving
       the original resolution. The default is None.

    Returns
    -------
    numpy array
        The images after loading.

    """
    current_images = []
    for i in images_files:
        image = Image.open(i).convert('RGB')
        if resolution != None:
            image = resize_images([image], resolution)[0]
        current_images.append(np.array(image))
    return np.array(current_images)

def load_images_filenames_image_temp(temp_filename_dir, image_index):
    """
    This function loads the image filenames from the temp file.

    Parameters
    ----------
    temp_filename_dir : string
        The path and filename to the temporary directory.
    image_index : numpy array
        The index of the temp file.

    Returns
    -------
    current_images_filenames : list
        The filenames of the images denoted by the image_index.

    """
    current_images_filenames = [None for i in range(len(image_index))]
    image_index_arg = np.argsort(image_index)
    sorted_image_index = image_index[image_index_arg]
    count = 0
    with open(temp_filename_dir) as f:
        for i, line in enumerate(f):
            if i == sorted_image_index[count]:
                current_images_filenames[image_index_arg[count]] = line.replace("\n", "")
                count += 1
                if count >= len(sorted_image_index):
                    break
    return current_images_filenames

def load_images_image_temp(temp_filename_dir, image_index, resolution = None):
    """
    This function loads the images.

    Parameters
    ----------
    temp_filename_dir : string
        The path and filename to the temporary directory.
    image_index : numpy array
        The index of the temp file.
    resolution : tuple or float, optional
       The resolution describes the output size of an image. If resolution
       contains a tuple with two ints, they are used as resolution. If it
       contains a float, the number is considered as scaling factor preserving
       the original resolution. The default is None.

    Returns
    -------
    numpy array
        The images after loading.

    """
    current_images = [None for i in range(len(image_index))]
    image_index_arg = np.argsort(image_index)
    sorted_image_index = image_index[image_index_arg]
    count = 0
    with open(temp_filename_dir) as f:
        for i, line in enumerate(f):
            if i == sorted_image_index[count]:
                image = Image.open(line.replace("\n", "")).convert('RGB')
                if resolution != None:
                    image = resize_images([image], resolution)[0]
                current_images[image_index_arg[count]] = np.array(image)
                count += 1
                if count >= len(sorted_image_index):
                    break
    return np.array(current_images)
