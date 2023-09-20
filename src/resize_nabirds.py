#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resize_nabirds.py
# Copyright (C) 2022 flossCoder
# 
# resize_nabirds is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resize_nabirds is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Mar  1 10:47:39 2022

@author: flossCoder
"""

import numpy as np
import os
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from image_processing import resize_images, crop_images_at_bounding_boxes, resize_bounding_boxes, generate_bounding_boxes_crops, generate_bounding_boxes_crops_with_zero_padding
from load_nabirds import load_txt_files_nabirds

def resize_nabirds(input_wd, output_wd, resolution, resize_at_bounding_box, bounding_boxes_fn = "bounding_boxes", classes_fn = "classes", image_class_labels_fn = "image_class_labels", images_fn = "images", train_test_split_fn = "train_test_split"):
    """
    This function resizes the nabirds dataset.

    Parameters
    ----------
    input_wd : string
        The input working directory.
    output_wd : string
        The output working directory.
    resolution : tuple
        Each entry of the resolution describes the output size of an image.
    resize_at_bounding_box : int
        Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
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

    Raises
    ------
    Exception
        The exception is raised in case an invalid resize_at_bounding_box parameter is given.

    Returns
    -------
    None.

    """
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_nabirds(input_wd, bounding_boxes_fn, classes_fn, image_class_labels_fn, images_fn, train_test_split_fn)
    bounding_boxes_resized = []
    aux_dir = os.path.join(output_wd, "images")
    os.mkdir(aux_dir)
    for i in range(len(images_file)):
        if i >= len(bounding_boxes) or (i < len(bounding_boxes) and bounding_boxes[i,0] != images_file[i,0]):
            j = np.where(bounding_boxes[:,0] == images_file[i,0])[0]
            if len(j) != 0:
                j = j[0]
            else:
                j = None
        elif i < len(bounding_boxes):
            j = i
        else:
            j = None
        if j is not None:
            image = Image.open(os.path.join(input_wd, "images", images_file[i,1]))
            original_resolution = image.size
            if resize_at_bounding_box == 0:
                [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bounding_boxes[j][1:]]], resolution)
                image = a[0]
                bb = b[0]
            elif resize_at_bounding_box == 1:
                [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bounding_boxes[j][1:]]], resolution)
                image = a[0]
                bb = b[0]
            elif resize_at_bounding_box == 2:
                image = resize_images([image], resolution)[0]
                bb = resize_bounding_boxes([[0] + [int(b) for b in bounding_boxes[j][1:]]], [original_resolution], resolution)
            elif resize_at_bounding_box == 3:
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bounding_boxes[j][1:]]], resolution)
                image = a[0]
                bb = b[0]
            else:
                raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
            
            if images_file[i,1].split("/")[0] not in os.listdir(aux_dir):
                os.mkdir(os.path.join(aux_dir, images_file[i,1].split("/")[0]))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(os.path.join(aux_dir, images_file[i,1]))
            bounding_boxes_resized.append([bounding_boxes[j,0]] + [str(b) for b in bb])
    
    with open(os.path.join(output_wd, "%s.txt"%bounding_boxes_fn), "w") as f:
        for i in bounding_boxes:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%classes_fn), "w") as f:
        for i in classes:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%image_class_labels_fn), "w") as f:
        for i in image_class_labels:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%images_fn), "w") as f:
        for i in images_file:
            f.writelines(" ".join(i) + "\n")
    with open(os.path.join(output_wd, "%s.txt"%train_test_split_fn), "w") as f:
        for i in train_test_split:
            f.writelines(" ".join(i) + "\n")

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [input_wd, output_wd, resolution width, resolution hight, resize_at_bounding_box, bounding_boxes_fn (optional), train_test_split_fn (optional)].
        input_wd : The input working directory.
        output_wd : The output working directory.
        resolution width: The width of the resolution.
        resolution hight: The hight of the resolution.
        resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
        bounding_boxes_fn (optional) : The input filename of the bounding box annotations. The default is "bounding_boxes".
        train_test_split_fn (optional) : The input filename of the train and test split. The default is "train_test_split".
    
    Raises
    ------
    Exception
        The exception is raised, in case not enough parameter are passed to main.

    Returns
    -------
    None.

    """
    if len(argv) != 5 and len(argv) != 7:
        print("The input parameter: [input_wd, output_wd, resolution, resize_at_bounding_box, bounding_boxes_fn (optional), train_test_split_fn (optional)].")
        print("input_wd : The input working directory.")
        print("output_wd : The output working directory.")
        print("resolution width: The width of the resolution.")
        print("resolution hight: The hight of the resolution.")
        print("resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).")
        print("bounding_boxes_fn (optional) : The input filename of the bounding box annotations. The default is 'bounding_boxes'.")
        print('train_test_split_fn (optional) : The input filename of the train and test split. The default is "train_test_split".')
        raise Exception("Wrong number of parameter")
    input_wd = argv[0]
    output_wd = argv[1]
    resolution = (int(argv[2]), int(argv[3]))
    resize_at_bounding_box = int(argv[4])
    if len(argv) == 5:
        resize_nabirds(input_wd, output_wd, resolution, resize_at_bounding_box)
    elif len(argv) == 7:
        bounding_boxes_fn = argv[5]
        train_test_split_fn = argv[6]
        resize_nabirds(input_wd, output_wd, resolution, resize_at_bounding_box, bounding_boxes_fn, "classes", "image_class_labels", "images", train_test_split_fn)

if __name__ == "__main__":
    main(sys.argv[1:])
