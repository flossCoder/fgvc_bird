#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resize_birdsnap.py
# Copyright (C) 2022 flossCoder
# 
# resize_birdsnap is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resize_birdsnap is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Mar  4 09:46:06 2022

@author: flossCoder
"""

import os
import sys
from shutil import copyfile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from image_processing import resize_images, crop_images_at_bounding_boxes, resize_bounding_boxes, generate_bounding_boxes_crops, generate_bounding_boxes_crops_with_zero_padding
from load_birdsnap import load_txt_files_birdsnap, aggregate_bounding_boxes_birdsnap

def resize_birdsnap(wd, code_dir, input_dir, output_dir, resolution, resize_at_bounding_box, image_split_fn = None, all_ims_fn = "all-ims", images_fn = "images.txt", species_fn = "species.txt", test_images_fn = "test_images.txt", success_fn = "success.txt", bb_fn = None):
    """
    This function resizes the birdsnap dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    code_dir : string
        The sub-directory of the code.
    input_dir : string
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
    resolution : tuple
        Each entry of the resolution describes the output size of an image.
    resize_at_bounding_box : int
        Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
    image_split_fn : string, optional
        The name of the file containing the split into train, test and evaluation. The default is None.
    all_ims_fn : string, optional
        The list of all images in the dataset. The default is "all-ims".
    images_fn : string, optional
        The annotation of the images. The default is "images.txt".
    species_fn : string, optional
        The annotation of the species. The default is "species.txt".
    test_images_fn : string, optional
        The official test set. The default is "test_images.txt".
    bb_fn : string, optional
        The bounding box annotation. The default is None.

    Raises
    ------
    Exception
        The exception is raised in case an invalid resize_at_bounding_box parameter is given.

    Returns
    -------
    None.

    """
    [all_ims_content, images_header, images_content, species_header, species_content, test_images_header, test_images_content, success_content, image_split_content, bb_content] = load_txt_files_birdsnap(wd, code_dir, input_dir, image_split_fn, bb_fn, all_ims_fn, images_fn, species_fn, test_images_fn, success_fn)
    if type(bb_content) == type(None):
        bounding_boxes = aggregate_bounding_boxes_birdsnap(success_content, images_content)
    else:
        bounding_boxes = bb_content
    bounding_boxes_resized = []
    aux_dir = os.path.join(wd, output_dir, "images")
    os.mkdir(aux_dir)
    for i in bounding_boxes:
        image = Image.open(os.path.join(wd, input_dir, "images", i[0])).convert("RGB")
        original_resolution = image.size
        if resize_at_bounding_box == 0:
            [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in i[1:]]], resolution)
            image = a[0]
            bb = b[0]
        elif resize_at_bounding_box == 1:
            [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in i[1:]]], resolution)
            image = a[0]
            bb = b[0]
        elif resize_at_bounding_box == 2:
            image = resize_images([image], resolution)[0]
            bb = resize_bounding_boxes([[0] + [int(b) for b in i[1:]]], [original_resolution], resolution)[0]
        elif resize_at_bounding_box == 3:
            [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in i[1:]]], resolution)
            image = a[0]
            bb = b[0]
        else:
            raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
        
        if i[0].split("/")[0] not in os.listdir(aux_dir):
            os.mkdir(os.path.join(aux_dir, i[0].split("/")[0]))
        
        image.save(os.path.join(aux_dir, i[0]))
        bounding_boxes_resized.append([i[0]] + [str(b) for b in bb[1:]])
    
    copyfile(os.path.join(wd, input_dir, success_fn), os.path.join(wd, output_dir, success_fn))
    if image_split_fn != None:
        copyfile(os.path.join(wd, input_dir, image_split_fn), os.path.join(wd, output_dir, image_split_fn))
    
    if bb_fn == None:
        f = open(os.path.join(wd, output_dir, "bounding_boxes.txt"), "w")
    else:
        f = open(os.path.join(wd, output_dir, bb_fn), "w")
    for i in bounding_boxes_resized:
        f.writelines("\t".join(i) + "\n")
    f.close()

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, code_dir, input_dir, output_dir, resolution width, resolution hight, image_split_fn, resize_at_bounding_box, bb_fn, test_images_fn].
        wd : The basic working directory.
        code_dir : The sub-directory of the code.
        input_dir : The input directory.
        output_dir : The output directory.
        resolution width: The width of the resolution.
        resolution hight: The hight of the resolution.
        image_split_fn : The name of the file containing the split into train, test and evaluation.
        resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
        test_images_fn (optional) : The official test set. The default is "test_images.txt".
        bb_fn (optional) : The bounding box annotation. The default is None.

    Raises
    ------
    Exception
        The exception is raised, in case not enough parameter are passed to main.

    Returns
    -------
    None.

    """
    if len(argv) != 8 and len(argv) != 10:
        print("The input parameter: [wd, code_dir, input_dir, output_dir, resolution, image_split_fn, resize_at_bounding_box, bb_fn, test_images_fn].")
        print("wd : The basic working directory.")
        print("code_dir : The sub-directory of the code.")
        print("input_dir : The input directory.")
        print("output_dir : The output directory.")
        print("resolution width: The width of the resolution.")
        print("resolution hight: The hight of the resolution.")
        print("image_split_fn : The name of the file containing the split into train, test and evaluation.")
        print("resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).")
        print("test_images_fn (optional) : The official test set. The default is 'test_images.txt'.")
        print("bb_fn (optional) : The bounding box annotation. The default is None.")
        raise Exception("Wrong number of parameter")
    wd = argv[0]
    code_dir = argv[1]
    input_dir = argv[2]
    output_dir = argv[3]
    resolution = (int(argv[4]), int(argv[5]))
    image_split_fn = argv[6]
    resize_at_bounding_box = int(argv[7])
    if len(argv) == 8:
        resize_birdsnap(wd, code_dir, input_dir, output_dir, resolution, resize_at_bounding_box, image_split_fn)
    elif len(argv) == 10:
        test_images_fn = argv[8]
        bb_fn = argv[9]
        resize_birdsnap(wd, code_dir, input_dir, output_dir, resolution, resize_at_bounding_box, image_split_fn, "all-ims", "images.txt", "species.txt", test_images_fn, "success.txt", bb_fn)

if __name__ == "__main__":
    main(sys.argv[1:])
