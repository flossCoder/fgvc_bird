#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_crop_images_at_bounding_boxes.py
# Copyright (C) 2022 flossCoder
# 
# test_crop_images_at_bounding_boxes is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# test_crop_images_at_bounding_boxes is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Feb  8 08:51:26 2022

@author: flossCoder
"""

import os
import sys
import numpy as np
from PIL import Image

resources_dir = os.path.join(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0], "test_resources")
src_dir = os.path.join(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0], "src")

sys.path.append(src_dir)
from image_processing import coordinates_crop_image_at_bounding_box, crop_images_at_bounding_boxes
from statistical_functions import check_epsilon

bounding_boxes = np.loadtxt(os.path.join(resources_dir, "data.txt"))
images = []
for i in bounding_boxes[:,0]:
    images.append(Image.open(os.path.join(resources_dir, "P1050142_%i.JPG"%int(i))))

# define resolutions
desired_resolution = [(1129, 1000), (1129, 1000), (1129, 1000), (1129, 1000), (1129, 700), (1129, 700), (1129, 700), (1129, 700), (1129, 833), (223, 164)]

# test changes on y-coordinate
# example that works fine:

i = 0
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 1000))
if [1050.0, 625.0, 1129.0, 1000.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 1000)) or [0.0, 83.0, 1129.0,  833.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that fits to the lower line:
i = 1
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 1000))
if [234.0, 288.0, 1140.0, 1010.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 1000)) or [0.0, 152.0, 1129.0, 820.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that fits to the upper line:
i = 2
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 1000))
if [3.0, 0, 1117.0, 990.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 1000)) or [0.0, 41.0, 1129.0, 843.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that does not fit
i = 3
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 1000))
if [14.0, -56.0, 1118.0, 991.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 1000)) or [0.0, 71.0, 1129.0, 832.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# test changes on y-coordinate
# example that works fine:
i = 4
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 700))
if [943.0, 708.0, 1344.0, 833.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 700)) or [90.0, 0.0, 948.0, 700.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that fits to the right line:
i = 5
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 700))
if [313.0, 132.0, 1339.0, 830.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 700)) or [153.0, 0.0, 944.0, 700.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that fits to the left line:
i = 6
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 700))
if [0, 20.0, 1323.0, 820.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 700)) or [12.0, 0.0, 952.0, 700.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that does not fit:
i = 7
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 700))
if [-93.0, 14.0, 1331.0, 825.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 700)) or [91.0, 0.0, 948.0, 700.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# test changes on all coordinate
# example that works fine (box and resolution are identical)
i = 8
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (1129, 833))
if [1050.0, 708.0, 1129.0, 833.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (1129 / 833)) or [0.0, 0.0, 1129.0, 833.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# example that works fine (box and resolution are not identical)
i = 9
[new_x, new_y, new_w, new_h, bb_x, bb_y, bb_w, bb_h] = coordinates_crop_image_at_bounding_box(images[i], bounding_boxes[i], (223, 164))
if [14.0, 20.0, 1115.0, 820.0] != [new_x, new_y, new_w, new_h] or not check_epsilon((new_w / new_h), (223 / 164)) or [0.0, 0.0, 223.0, 164.0] != [bb_x, bb_y, bb_w, bb_h] or not check_epsilon((bounding_boxes[i,3] / bounding_boxes[i,4]), (bb_w / bb_h)):
    raise Exception("Example %i failes"%i)

# Test the generation of cropped images
new_images = []
new_bounding_boxes = []
for i in range(len(bounding_boxes[:,0])):
    [a, b] = crop_images_at_bounding_boxes([images[i]], [bounding_boxes[i]], desired_resolution[i])
    new_images.append(a[0])
    new_bounding_boxes.append(b[0])
    if any([not check_epsilon(a[0].size[k], desired_resolution[i][k]) for k in range(len(desired_resolution[i]))]):
        raise Exception("Example %i has invalid resolution"%i)
