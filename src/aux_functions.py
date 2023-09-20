#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# aux_functions.py
# Copyright (C) 2022 flossCoder
# 
# aux_functions is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# aux_functions is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Feb 22 09:41:46 2022

@author: flossCoder
"""

import numpy as np
from scipy.interpolate import interp1d

power = lambda x, alpha, k, b: k*x**alpha+b

"""
This function convertes a dictionary into a numpy array, such that the values are ordered by the keys.

Parameters
----------
dictionary : dict
    The dictionary that shall be converted.

Returns
-------
numpy array
    The values of the dictionary ordered by the keys.

"""
dict_to_numpy_array = lambda dictionary: np.array([dictionary[i] for i in np.sort(list(dictionary.keys()))])

"""
This function convertes a dictionary into a numpy array, such that the values are ordered by the given keys.

Parameters
----------
dictionary : dict
    The dictionary that shall be converted.

ordered_keys : list
    The keys of the dictionary in a given order.

Returns
-------
numpy array
    The values of the dictionary ordered by the given keys.

"""
dict_to_numpy_array_ordered_keys = lambda dictionary, ordered_keys: np.array([dictionary[i] for i in ordered_keys])

def bbox_iou(bbox_gt, bbox_pred):
    """
    This function calculates the bounding box intersection over union (iou).

    Parameters
    ----------
    bbox_gt : numpy array
        The ground truth bounding box denotes the label from the dataset.
        The bounding box must be given in lucwh (xywh) format.
        bbox_gt can be eather one bounding box (shape = (4,)) or a list of n boxes (shape = (n,4)).
    bbox_pred : numpy array
        The predicted bounding boxes, for more details s. bbox_gt.

    Raises
    ------
    Exception
        The exception is raised in case the shapes of bbox_gt and bbox_pred differ.

    Returns
    -------
    numpy array
        The resulting intersection over union (iou) of the bounding boxes.

    """
    if np.shape(bbox_gt) != np.shape(bbox_pred):
        raise Exception("Different input shapes of ground truth %s and prediction %s"%(str(np.shape(bbox_gt)), str(np.shape(bbox_pred))))
    if len(np.shape(bbox_gt)) == 1:
        x_gt = bbox_gt[0]
        y_gt = bbox_gt[1]
        w_gt = bbox_gt[2]
        h_gt = bbox_gt[3]
        x_pred = bbox_pred[0]
        y_pred = bbox_pred[1]
        w_pred = bbox_pred[2]
        h_pred = bbox_pred[3]
        x_overlap = np.max([0, np.min([(x_gt + w_gt), (x_pred + w_pred)]) - np.max([x_gt, x_pred])])
        y_overlap = np.max([0, np.min([(y_gt + h_gt), (y_pred + h_pred)]) - np.max([y_gt, y_pred])])
    else:
        x_gt = bbox_gt[:,0]
        y_gt = bbox_gt[:,1]
        w_gt = bbox_gt[:,2]
        h_gt = bbox_gt[:,3]
        x_pred = bbox_pred[:,0]
        y_pred = bbox_pred[:,1]
        w_pred = bbox_pred[:,2]
        h_pred = bbox_pred[:,3]
        x_overlap = np.max([np.zeros(len(x_gt)), np.min([(x_gt + w_gt), (x_pred + w_pred)], axis = 0) - np.max([x_gt, x_pred], axis = 0)], axis = 0)
        y_overlap = np.max([np.zeros(len(y_gt)), np.min([(y_gt + h_gt), (y_pred + h_pred)], axis = 0) - np.max([y_gt, y_pred], axis = 0)], axis = 0)
    a_intersect = x_overlap * y_overlap
    a_gt = w_gt * h_gt
    a_pred = w_pred * h_pred
    a_union = a_gt + a_pred - a_intersect
    return a_intersect / a_union

def compute_label_assignment(labels):
    """
    This function computes the assignemt of the labels to the artificial labels required by keras.

    Parameters
    ----------
    labels : numpy array
        The original label array.

    Returns
    -------
    labels_artificial_labels_assignment : numpy array
        A LUT describing the assignment of the original labels to the artificial labels.

    """
    index = np.argsort(labels)
    index_inv = np.searchsorted(labels[index], labels)
    artificial_labels = np.arange(len(labels))
    labels_artificial_labels_assignment = np.zeros((len(labels), 2))
    labels_artificial_labels_assignment[:,0] = labels
    labels_artificial_labels_assignment[:,1] = artificial_labels[index_inv]
    return labels_artificial_labels_assignment

def convert_labels_to_artificial_labels(labels, labels_artificial_labels_assignment):
    """
    This function converts labels into artificial labels.

    Parameters
    ----------
    labels : numpy array
        The original label array.
    labels_artificial_labels_assignment : numpy array
        A LUT describing the assignment of the original labels to the artificial labels.

    Returns
    -------
    artificial_labels : numpy array
        The artificial label array.

    """
    artificial_labels = np.zeros(np.shape(labels))
    for i in range(len(labels)):
        artificial_labels[i] = labels_artificial_labels_assignment[labels_artificial_labels_assignment[:,0] == labels[i],1][0]
    return artificial_labels

def convert_artificial_labels_to_labels(artificial_labels, labels_artificial_labels_assignment):
    """
    This function converts artificial labels into labels.

    Parameters
    ----------
    artificial_labels : numpy array
        The artificial label array.
    labels_artificial_labels_assignment : numpy array
        A LUT describing the assignment of the original labels to the artificial labels.

    Returns
    -------
    labels : numpy array
        The original label array.

    """
    labels = np.zeros(np.shape(artificial_labels))
    for i in range(len(artificial_labels)):
        labels[i] = labels_artificial_labels_assignment[labels_artificial_labels_assignment[:,1] == artificial_labels[i],0][0]
    return labels

def generate_dataset_interpolated_data(counts_dataset_1, counts_dataset_2):
    """
    This function calculates the interpolated counts of dataset 1 in the bins of dataset 2.

    Parameters
    ----------
    counts_dataset_1 : numpy array
        The counts of dataset 1, make sure the number of bins is larger than the one for dataset 2.
    counts_dataset_2 : numpy array
        The counts of dataset 2, make sure the number of bins is smaller than the one for dataset 1.

    Raises
    ------
    Exception
        The exception is raised in case the number of bins of dataset 1 is less or equals to the number of bins of dataset 2.

    Returns
    -------
    new_counts_dataset_1 : numpy array
        The counts of dataset 1 interpolated into the bin space of dataset 2.

    """
    if len(counts_dataset_1) <= len(counts_dataset_2):
        raise Exception("Dataset 2 is larger or equals to dataset 1")
    
    counts_dataset_1 = np.sort(counts_dataset_1)[::-1]
    counts_dataset_2 = np.sort(counts_dataset_2)[::-1]
    
    x_dataset_1 = np.arange(len(counts_dataset_1)) / len(counts_dataset_1)
    x_dataset_2 = np.arange(len(counts_dataset_2)) / len(counts_dataset_2)
    
    f = interp1d(x_dataset_1, counts_dataset_1, kind = 'cubic')
    
    new_counts_dataset_1 = f(x_dataset_2)
    
    return new_counts_dataset_1
