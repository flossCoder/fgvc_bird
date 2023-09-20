#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resize_naturgucker.py
# Copyright (C) 2022 flossCoder
# 
# resize_naturgucker is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resize_naturgucker is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Feb  4 10:24:12 2022

@author: flossCoder
"""

import os
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from copy import deepcopy

from image_processing import resize_images, crop_images_at_bounding_boxes, resize_bounding_boxes, generate_bounding_boxes_crops, generate_bounding_boxes_crops_with_zero_padding
from bilder_download import import_voegel_csv, aux_mkdir

def resize_naturgucker(wd, filename, resolution, resize_at_bounding_box, bounding_boxes_fn=None, input_directory_name=None, output_directory_name=None):
    """
    This function resizes all images from naturgucker and saves the results.

    Parameters
    ----------
    wd : string
        The basic working directory.
    filename : string
        The filename describes the name of the csv file containing the required data.
    resolution : tuple or float
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution.
    resize_at_bounding_box : int
        Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
    bounding_boxes_fn : string, optional
        The filename of the bounding boxes, if not None. Ensure it is not None, if resize_at_bounding_boxes is not (2). The default is None.
    input_directory_name : string, optional
        The name of the subdirectory, where the data shall be loaded from, if None, open from wd. The default is None.
    output_directory_name : string, optional
        The name of the subdirectory, where the data shall be saved, if None, save in wd. The default is None.

    Raises
    ------
    Exception
        The exception is raised in case an image cannot be loaded or in case an invalid resize_at_bounding_box parameter is given..

    Returns
    -------
    None.

    """
    [header, data] = import_voegel_csv(wd, filename)
    if input_directory_name != None and input_directory_name not in os.listdir(wd):
        raise Exception("Invalid directory name: %s"%input_directory_name)
    order_wd = os.path.join(wd, input_directory_name) if input_directory_name != None else wd
    order_wd_out = os.path.join(wd, output_directory_name) if output_directory_name != None else wd
    if bounding_boxes_fn is not None:
        bounding_boxes = np.loadtxt(os.path.join(wd, "%s.txt"%bounding_boxes_fn))
        bounding_boxes_new = deepcopy(bounding_boxes)
    else:
        bounding_boxes = None
    for i in range(len(data)):
        [order, family, genus, species, bird_id, bird_url] = data[i]
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
            aux_mkdir(order_wd_out, order)
            family_wd_out = os.path.join(order_wd_out, order)
            aux_mkdir(family_wd_out, family)
            genus_wd_out = os.path.join(family_wd_out, family)
            aux_mkdir(genus_wd_out, genus)
            species_wd_out = os.path.join(genus_wd_out, genus)
            aux_mkdir(species_wd_out, species)
            figure_wd_out = os.path.join(species_wd_out, species)
            image = Image.open(os.path.join(figure_wd, figure)).convert('RGB')
            original_resolution = image.size
            if resize_at_bounding_box == 0:
                [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bounding_boxes[i][1:]]], resolution)
                image = a[0]
                bounding_boxes_new[i][1:] = b[0][1:]
            elif resize_at_bounding_box == 1:
                [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bounding_boxes[i][1:]]], resolution)
                image = a[0]
                bounding_boxes_new[i][1:] = b[0][1:]
            elif resize_at_bounding_box == 2:
                image = resize_images([image], resolution)[0]
                if bounding_boxes_fn is not None:
                    b = resize_bounding_boxes([[0] + [int(b) for b in bounding_boxes[i][1:]]], [original_resolution], resolution)
                    bounding_boxes_new[i][1:] = b[0][1:]
            elif resize_at_bounding_box == 3:
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bounding_boxes[i][1:]]], resolution)
                image = a[0]
                bounding_boxes_new[i][1:] = b[0][1:]
            else:
                raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
            image.save(os.path.join(figure_wd_out, figure))
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
    if bounding_boxes_fn is not None:
        if resize_at_bounding_box == 0:
            bb_addon = "erw-bb-crop"
        elif resize_at_bounding_box == 1:
            bb_addon = "bb-crop"
        elif resize_at_bounding_box == 2:
            bb_addon = "skaliert"
        elif resize_at_bounding_box == 3:
            bb_addon = "bb-crop-zp"
        np.savetxt(os.path.join(order_wd_out, "%s_%s.txt"%(bounding_boxes_fn, bb_addon)), bounding_boxes_new, fmt = "%s")

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, filename, directory_name, resolution width, resolution hight, resize_at_bounding_box, bounding_boxes_fn, input_directory_name, output_directory_name].
        wd: The basic working directory.
        filename: The filename describes the name of the csv file containing the required data.
        resolution width: The width of the resolution.
        resolution hight: The hight of the resolution.
        resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
        bounding_boxes_fn : The filename of the bounding boxes, if not None. Ensure it is not None, if resize_at_bounding_boxes is not (2).
        input_directory_name: The name of the subdirectory, where the data shall be opened from, if None, open from wd.
        output_directory_name: The name of the subdirectory, where the data shall be saved, if None, save in wd.

    Raises
    ------
    Exception
        The exception is raised, in case not enough parameter are passed to main.

    Returns
    -------
    None.

    """
    if len(argv) < 5:
        print("error, wrong number of parameter")
        print("wd: The basic working directory.")
        print("filename: The filename describes the name of the csv file containing the required data.")
        print("resolution width: The width of the resolution.")
        print("resolution hight: The hight of the resolution.")
        print("resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).")
        print("bounding_boxes_fn : The filename of the bounding boxes, if not None. Ensure it is not None, if resize_at_bounding_boxes is not (2).")
        print("input_directory_name: The name of the subdirectory, where the data shall be opened from, if None, open from wd.")
        print("output_directory_name: The name of the subdirectory, where the data shall be saved, if None, save in wd.")
        raise Exception()
    elif len(argv) == 5:
        wd = argv[0]
        filename = argv[1]
        resolution = (int(argv[2]), int(argv[3]))
        resize_at_bounding_box = int(argv[4])
        bounding_boxes_fn=None
        input_directory_name=None
        output_directory_name=None
    elif len(argv) == 6:
        wd = argv[0]
        filename = argv[1]
        resolution = (int(argv[2]), int(argv[3]))
        resize_at_bounding_box = int(argv[4])
        bounding_boxes_fn = argv[5] if argv[5] != "None" else None
        input_directory_name=None
        output_directory_name=None
    elif len(argv) == 7:
        wd = argv[0]
        filename = argv[1]
        resolution = (int(argv[2]), int(argv[3]))
        resize_at_bounding_box = int(argv[4])
        bounding_boxes_fn = argv[5] if argv[5] != "None" else None
        input_directory_name=argv[6]
        output_directory_name=None
    elif len(argv) == 8:
        wd = argv[0]
        filename = argv[1]
        resolution = (int(argv[2]), int(argv[3]))
        resize_at_bounding_box = int(argv[4])
        bounding_boxes_fn = argv[5] if argv[5] != "None" else None
        input_directory_name=argv[6]
        output_directory_name=argv[7]
    
    resize_naturgucker(wd, filename, resolution, resize_at_bounding_box, bounding_boxes_fn, input_directory_name, output_directory_name)

if __name__ == "__main__":
    main(sys.argv[1:])
