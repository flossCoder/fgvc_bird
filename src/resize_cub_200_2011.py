#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resize_cub_200_2011.py
# Copyright (C) 2022 flossCoder
# 
# resize_cub_200_2011 is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resize_cub_200_2011 is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jun 27 06:17:52 2022

@author: flossCoder
"""

import numpy as np
import os
import sys
from shutil import copyfile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from load_cub_200_2011 import load_txt_files_cub
from image_processing import resize_images, crop_images_at_bounding_boxes, resize_bounding_boxes, generate_bounding_boxes_crops, generate_bounding_boxes_crops_with_zero_padding

def resize_cub_200_2011(wd, input_dir, output_dir, resolution, resize_at_bounding_box, bounding_boxes_fn = "bounding_boxes", classes_fn = "classes", image_class_labels_fn = "image_class_labels", images_fn = "images", train_test_split_fn = "train_test_split"):
    """
    This function resizes the CUB 200 2011 dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    input_dir : string
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
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
    [bounding_boxes, classes, image_class_labels, images_file, train_test_split] = load_txt_files_cub(os.path.join(wd, input_dir), bounding_boxes_fn, classes_fn, image_class_labels_fn, images_fn, train_test_split_fn)
    os.mkdir(os.path.join(wd, output_dir, "images"))
    copyfile(os.path.join(wd, input_dir, "%s.txt"%classes_fn), os.path.join(wd, output_dir, "%s.txt"%classes_fn))
    copyfile(os.path.join(wd, input_dir, "%s.txt"%image_class_labels_fn), os.path.join(wd, output_dir, "%s.txt"%image_class_labels_fn))
    copyfile(os.path.join(wd, input_dir, "%s.txt"%images_fn), os.path.join(wd, output_dir, "%s.txt"%images_fn))
    copyfile(os.path.join(wd, input_dir, "%s.txt"%train_test_split_fn), os.path.join(wd, output_dir, "%s.txt"%train_test_split_fn))
    bounding_boxes_resized = []
    for i in images_file:
        image = Image.open(os.path.join(wd, input_dir, "images", i[1])).convert('RGB')
        original_resolution = image.size
        bounding_box = bounding_boxes[bounding_boxes[:,0] == int(i[0])]
        if len(bounding_box) != 0:
            bounding_box = bounding_box[0]
            if resize_at_bounding_box == 0:
                [a, b] = crop_images_at_bounding_boxes([image], [[int(b) for b in bounding_box]], resolution)
                image = a[0]
                bb = b[0]
            elif resize_at_bounding_box == 1:
                [a, b] = generate_bounding_boxes_crops([image], [[int(b) for b in bounding_box]], resolution)
                image = a[0]
                bb = b[0]
            elif resize_at_bounding_box == 2:
                image = resize_images([image], resolution)[0]
                bb = resize_bounding_boxes([[int(b) for b in bounding_box]], [original_resolution], resolution)[0]
            elif resize_at_bounding_box == 3:
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[int(b) for b in bounding_box]], resolution)
                image = a[0]
                bb = b[0]
            else:
                raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
            
            if i[1].split("/")[0] not in os.listdir(os.path.join(wd, output_dir, "images")):
                os.mkdir(os.path.join(wd, output_dir, "images", i[1].split("/")[0]))
            image.save(os.path.join(wd, output_dir, "images", i[1]))
            bounding_boxes_resized.append(bb)
    np.savetxt(os.path.join(wd, output_dir, "%s.txt"%bounding_boxes_fn), bounding_boxes_resized)

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, input_dir, output_dir, resolution width, resolution hight, resize_at_bounding_box, bounding_boxes_fn (optional)].
        wd : The basic working directory.
        input_dir : The sub-directory of the input data.
        output_dir : The sub-directory of the output data.
        resolution width: The width of the resolution.
        resolution hight: The hight of the resolution.
        resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
        bounding_boxes_fn (optional) : The input filename of the bounding box annotations. The default is 'bounding_boxes'.

    Returns
    -------
    None.

    """
    if len(argv) != 6 and len(argv) != 7:
        print("The input parameter: [wd, input_dir, output_dir, resolution, resize_at_bounding_box, bounding_boxes_fn (optional)].")
        print("wd : The basic working directory.")
        print("input_dir : The sub-directory of the input data.")
        print("output_dir : The sub-directory of the output data.")
        print("resolution width: The width of the resolution.")
        print("resolution hight: The hight of the resolution.")
        print("resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).")
        print("bounding_boxes_fn (optional) : The input filename of the bounding box annotations. The default is 'bounding_boxes'.")
        raise Exception("Wrong number of parameter")
    wd = argv[0]
    input_dir = argv[1]
    output_dir = argv[2]
    resolution = (int(argv[3]), int(argv[4]))
    resize_at_bounding_box = int(argv[5])
    
    if len(argv) == 6:
        resize_cub_200_2011(wd, input_dir, output_dir, resolution, resize_at_bounding_box)
    elif len(argv) == 7:
        bounding_boxes_fn = argv[6]
        resize_cub_200_2011(wd, input_dir, output_dir, resolution, resize_at_bounding_box, bounding_boxes_fn)

if __name__ == "__main__":
    main(sys.argv[1:])
