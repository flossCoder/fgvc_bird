#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# convert_bounding_boxes.py
# Copyright (C) 2022 flossCoder
# 
# convert_bounding_boxes is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# convert_bounding_boxes is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Wed Mar  2 11:19:03 2022

@author: flossCoder
"""

def bb_conv_lucwh_to_cwh(input_bounding_box):
    """
    Convert a given bounding box in the format upper left corner plus width and height
    into the format bounding box center plus width and height.

    Parameters
    ----------
    input_bounding_box : list
        The list contains [id, x upper left corner, y upper left corner, width, height].

    Returns
    -------
    list
        [bb_id, bb_c_x, bb_c_y, bb_w, bb_h].
        bb_id : The id of the bounding box (optional).
        bb_c_x : x coordinate of the bounding box center.
        bb_c_y : y coordinate of the bounding box center.
        bb_w : Width of the bounding box.
        bb_h : Height of the bounding box.

    """
    if len(input_bounding_box) == 5:
        bb_id = input_bounding_box[0]
        bb_c_x = input_bounding_box[1] + input_bounding_box[3] / 2
        bb_c_y = input_bounding_box[2] + input_bounding_box[4] / 2
        bb_w = input_bounding_box[3]
        bb_h = input_bounding_box[4]
        return [bb_id, bb_c_x, bb_c_y, bb_w, bb_h]
    else:
        bb_c_x = input_bounding_box[0] + input_bounding_box[2] / 2
        bb_c_y = input_bounding_box[1] + input_bounding_box[3] / 2
        bb_w = input_bounding_box[2]
        bb_h = input_bounding_box[3]
        return [bb_c_x, bb_c_y, bb_w, bb_h]

def bb_conv_cwh_to_lucwh(input_bounding_box):
    """
    Convert a given bounding box in the format bounding box center plus width and height
    into the format upper left corner plus width and height.

    Parameters
    ----------
    input_bounding_box : list
        The list contains [id, x center, y center, width, height].

    Returns
    -------
    list
        [bb_id, bb_luc_x, bb_luc_y, bb_w, bb_h].
        bb_id : The id of the bounding box (optional).
        bb_luc_x : x coordinate of the upper left corner.
        bb_luc_y : y coordinate of the upper left corner.
        bb_w : Width of the bounding box.
        bb_h : Height of the bounding box.

    """
    if len(input_bounding_box) == 5:
        bb_id = input_bounding_box[0]
        bb_luc_x = input_bounding_box[1] - input_bounding_box[3] / 2
        bb_luc_y = input_bounding_box[2] - input_bounding_box[4] / 2
        bb_w = input_bounding_box[3]
        bb_h = input_bounding_box[4]
        return [bb_id, bb_luc_x, bb_luc_y, bb_w, bb_h]
    else:
        bb_luc_x = input_bounding_box[0] - input_bounding_box[2] / 2
        bb_luc_y = input_bounding_box[1] - input_bounding_box[3] / 2
        bb_w = input_bounding_box[2]
        bb_h = input_bounding_box[3]
        return [bb_luc_x, bb_luc_y, bb_w, bb_h]

def bb_conv_lucwh_to_lucrlc(input_bounding_box):
    """
    Convert a given bounding box in the format upper left corner plus width and height
    into the format upper left corner, lower right corner.

    Parameters
    ----------
    input_bounding_box : list
        The list contains [id, x upper left corner, y upper left corner, width, height].

    Returns
    -------
    list
        [bb_id, bb_luc_x, bb_luc_y, bb_rlc_x, bb_rlc_y].
        bb_id : The id of the bounding box (optional).
        bb_luc_x : x coordinate of the upper left corner.
        bb_luc_y : y coordinate of the upper left corner.
        bb_rlc_x : x coordinate of the lower right corner.
        bb_rlc_y : y coordinate of the lower right corner.

    """
    if len(input_bounding_box) == 5:
        bb_id = input_bounding_box[0]
        bb_luc_x = input_bounding_box[1]
        bb_luc_y = input_bounding_box[2]
        bb_rlc_x = input_bounding_box[1] + input_bounding_box[3]
        bb_rlc_y = input_bounding_box[2] + input_bounding_box[4]
        return [bb_id, bb_luc_x, bb_luc_y, bb_rlc_x, bb_rlc_y]
    else:
        bb_luc_x = input_bounding_box[0]
        bb_luc_y = input_bounding_box[1]
        bb_rlc_x = input_bounding_box[0] + input_bounding_box[2]
        bb_rlc_y = input_bounding_box[1] + input_bounding_box[3]
        return [bb_luc_x, bb_luc_y, bb_rlc_x, bb_rlc_y]

def bb_conv_lucrlc_to_lucwh(input_bounding_box):
    """
    Convert a given bounding box in the format upper left corner, lower right corner
    into the format upper left corner plus width and height.

    Parameters
    ----------
    input_bounding_box : list
        The list contains [id, x upper left corner, y upper left corner, x lower right corner, y lower right corner].

    Returns
    -------
    list
        [bb_id, bb_luc_x, bb_luc_y, bb_w, bb_h].
        bb_id : The id of the bounding box (optional).
        bb_luc_x : x coordinate of the upper left corner.
        bb_luc_y : y coordinate of the upper left corner.
        bb_w : Width of the bounding box.
        bb_h : Height of the bounding box.

    """
    if len(input_bounding_box) == 5:
        bb_id = input_bounding_box[0]
        bb_luc_x = input_bounding_box[1]
        bb_luc_y = input_bounding_box[2]
        bb_w = input_bounding_box[3] - input_bounding_box[1]
        bb_h = input_bounding_box[4] - input_bounding_box[2]
        return [bb_id, bb_luc_x, bb_luc_y, bb_w, bb_h]
    else:
        bb_luc_x = input_bounding_box[0]
        bb_luc_y = input_bounding_box[1]
        bb_w = input_bounding_box[2] - input_bounding_box[0]
        bb_h = input_bounding_box[3] - input_bounding_box[1]
        return [bb_luc_x, bb_luc_y, bb_w, bb_h]
