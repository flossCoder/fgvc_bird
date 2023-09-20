#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# plot_image.py
# Copyright (C) 2022 flossCoder
# 
# plot_image is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# plot_image is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Feb  1 11:16:41 2022

@author: flossCoder
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_bounding_box(bounding_box, color, show_anno = False):
    """
    This function plots the bounding box of an image.

    Parameters
    ----------
    bounding_box : numpy array
        This array contains x and y coordinates in pixels of the top left corner plus width and height in pixels.
    color : string
        The drawing color of the bounding box (must be valid matplotlib).
    show_anno : string, optional
        Draw the top left corner of the bounding box. The default is False.

    Returns
    -------
    None.

    """
    plt.plot([bounding_box[0], bounding_box[0]+bounding_box[2], bounding_box[0]+bounding_box[2], bounding_box[0], bounding_box[0]], [bounding_box[1], bounding_box[1], bounding_box[1]+bounding_box[3], bounding_box[1]+bounding_box[3], bounding_box[1]], "%s"%color)
    if show_anno:
        plt.plot(bounding_box[0], bounding_box[1], "o%s"%color)

def plot_image(image):
    """
    This function plots an image.

    Parameters
    ----------
    image : PIL or numpy array
        The image.

    Returns
    -------
    None.

    """
    if type(image) == type(np.array([])):
        plt.imshow(image)
    else:
        plt.imshow(np.array(image))

def plot_cup(image, bounding_box = None, color = "r", show_anno = False):
    """
    This function plots the image of the CUB-200-2011 dataset.

    Parameters
    ----------
    image : PIL or numpy array
        The image.
    bounding_box : TYPE, optional
        This array contains x and y coordinates in pixels of the top left corner plus width and height in pixels. The default is None.
    color : string
        The drawing color of the bounding box (must be valid matplotlib).
    show_anno : string, optional
        Draw the top left corner of the bounding box. The default is False.

    Returns
    -------
    None.

    """
    plot_image(image)
    if type(bounding_box) != type(None):
        plot_bounding_box(bounding_box[1:] if len(bounding_box) == 5 else bounding_box, color, show_anno)

def plot_nabirds(image, bounding_box = None, color = "r", show_anno = False):
    """
    This function plots the image of the NABirds dataset.

    Parameters
    ----------
    image : PIL or numpy array
        The image.
    bounding_box : TYPE, optional
        This array contains x and y coordinates in pixels of the top left corner plus width and height in pixels. The default is None.
    color : string
        The drawing color of the bounding box (must be valid matplotlib).
    show_anno : string, optional
        Draw the top left corner of the bounding box. The default is False.

    Returns
    -------
    None.

    """
    plot_image(image)
    if type(bounding_box) != type(None):
        plot_bounding_box(bounding_box[1:].astype(int) if len(bounding_box) == 5 else bounding_box.astype(int), color, show_anno)
