#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# apply_yolo_v5.py
# Copyright (C) 2022 flossCoder
# 
# apply_yolo_v5 is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# apply_yolo_v5 is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jul 11 12:02:05 2022

@author: flossCoder
"""

import torch
import numpy as np
import pickle
import os

from image_processing import load_images_filenames_image_temp
from convert_bounding_boxes import bb_conv_cwh_to_lucwh
from aux_functions import bbox_iou

BIRD_YOLO_ID = 14

def load_yolo5(net_version = 'yolov5x', hub_dir = 'ultralytics/yolov5'):
    """
    This function loads the yolov5 pretrained net from the torch hub.

    Parameters
    ----------
    net_version : string, optional
        The size of the network. The following is available:
            yolov5n = Nano, yolov5s = Small, yolov5m = Medium, yolov5l = Large, yolov5x = XLarge.
            The accuracy depends on the size of the network. The default is 'yolov5l'.
    hub_dir : string, optional
        The type of network (usually not changed). The default is 'ultralytics/yolov5'.

    Returns
    -------
    model : models.common.AutoShape
        The cnn model loaded from torch hub.

    """
    model = torch.hub.load(hub_dir, net_version)
    return model

def calculate_bounding_boxes(model, temp_filename_dir, image_index):
    """
    This function calculates the bounding boxes of the given images.

    Parameters
    ----------
    model : models.common.AutoShape
        The cnn yolo model.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    image_index : numpy array
        The index of the temp file.

    Returns
    -------
    aggregated_bounding_boxes : list
        Each entry is a sublist representing one bounding box. Each bounding box is determined by
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
            s : Score of the bounding box (How shure is yolo, that the bounding box is good).
            c : Estimated class.
    current_results : list
        The detection objects of the images.

    """
    current_image_files = load_images_filenames_image_temp(temp_filename_dir, image_index)
    try:
        current_results = model(current_image_files)
    except:
        for i in current_image_files:
            try:
                aux = model([i])
            except:
                print("Failiure in image " + i)
        current_results = model(current_image_files)
    aggregated_bounding_boxes = []
    for i in range(len(image_index)):
        for j in current_results.xywh[i]:
            if len(j) != 0 and j[5] == BIRD_YOLO_ID:
                aux = j.tolist()
                aggregated_bounding_boxes.append([image_index[i]] + bb_conv_cwh_to_lucwh(aux[:4]) + aux[4:])
    return [aggregated_bounding_boxes, current_results.tolist()]

def find_birds(wd, output_name, model, temp_filename_dir, number_of_images, batch_size = None, pickle_objects = False):
    """
    This function generates the bounding boxes list for the images and the given model.

    Parameters
    ----------
    wd : string
        The basic working directory.
    output_name : string
        The basic name of the output filename.
    model : models.common.AutoShape
        The cnn yolo model.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    number_of_images : int
        The number of images.
    batch_size : int, optional
        The batch_size denotes the number of images send to yolo for the evaluation per batch.
        If the batch_size is None, all images will be send to yolo in one step. The default is None.
    pickle_objects : boolean, optional
        Pickle the original detection objects (True) or not (False). The default is False.

    Returns
    -------
    aggregated_bounding_boxes : list
        Each entry is a sublist representing one bounding box. Each bounding box is determined by
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
            s : Score of the bounding box (How shure is yolo, that the bounding box is good).
            c : Estimated class.

    """
    if type(batch_size) != type(None):
        aggregated_bounding_boxes = []
        detection_objects = []
        count = 0
        for i in np.arange(np.ceil(number_of_images / batch_size)):
            print("%s / %s"%(str(i), str(np.ceil(number_of_images / batch_size))))
            image_index = np.arange(count, np.min((number_of_images, (count + batch_size))))
            count += batch_size
            current_bb, current_do = calculate_bounding_boxes(model, temp_filename_dir, image_index)
            aggregated_bounding_boxes += current_bb
            if pickle_objects:
                pickle.dump(current_do, open("%s_%i.pkl"%(os.path.join(wd, output_name), i), "wb"))
    else:
        image_index = np.arange(number_of_images)
        aggregated_bounding_boxes, detection_objects = calculate_bounding_boxes(model, temp_filename_dir, image_index)
        if pickle_objects:
            pickle.dump(detection_objects, open("%s.pkl"%(os.path.join(wd, output_name)), "wb"))
    return aggregated_bounding_boxes

def aggregate_pickle(wd, output_name, number_of_images, batch_size):
    """
    This function aggregates the pickle files dumped out batchwise.

    Parameters
    ----------
    wd : string
        The basic working directory.
    output_name : string
        The basic name of the output filename.
    number_of_images : int
        The number of images.
    batch_size : int
        The batch_size denotes the number of images send to yolo for the evaluation per batch.

    Returns
    -------
    None.

    """
    detection_objects = []
    for i in np.arange(np.ceil(number_of_images / batch_size)):
        detection_objects += pickle.load(open("%s_%i.pkl"%(os.path.join(wd, output_name), i), "rb"))
    for i in np.arange(np.ceil(number_of_images / batch_size)):
        os.remove("%s_%i.pkl"%(os.path.join(wd, output_name), i))
    pickle.dump(detection_objects, open("%s.pkl"%(os.path.join(wd, output_name)), "wb"))

def post_processing_bounding_boxes(number_of_images, aggregated_bounding_boxes, choose_best_bb = True, score_threshold = -np.inf, max_boxes_per_images = np.inf, **kwargs):
    """
    This function post processes the bounding box list.

    Parameters
    ----------
    number_of_images : int
        The number of images.
    aggregated_bounding_boxes : list
        Each entry is a sublist representing one bounding box. Each bounding box is determined by
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
            s : Score of the bounding box (How shure is yolo, that the bounding box is good).
            c : Estimated class.
    choose_best_bb : boolean, optional
        If more than one bounding box exists for an image (True), choose the best one. The default is True.
    score_threshold : float, optional
        The minimal score for a bounding box to be passed. The default is -np.inf.
    max_boxes_per_images : float, optional
        The maximum number of bounding boxes per images. The default is np.inf.
    **kwargs : dict
        The kwargs can be used for optional parameters.

    Returns
    -------
    new_aggregated_bounding_boxes : list
        The filtered version of the aggregated_bounding_boxes input.
        Each entry is a sublist represents one bounding box. Each bounding box is determined by
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
            s : Score of the bounding box (How shure is yolo, that the bounding box is good).
            c : Estimated class.

    """
    if not(choose_best_bb) and not(np.isfinite(score_threshold)) and not(np.isfinite(max_boxes_per_images)):
        return aggregated_bounding_boxes
    aux = np.array(aggregated_bounding_boxes)
    new_aggregated_bounding_boxes = []
    for i in np.unique(aux[:,0]):
        current_boxes = np.where(aux[:,0] == i)[0]
        if len(current_boxes) != 0:
            if choose_best_bb: # for images with more than one bounding box choose the best one according to the score
                sorting = np.argsort(aux[current_boxes,5])
                if aux[current_boxes[sorting[-1]],5] > score_threshold: # further check, if best score is large enough
                    new_aggregated_bounding_boxes += [aux[current_boxes[sorting[-1]],:].tolist()]
            elif len(current_boxes) > max_boxes_per_images: # remove images containing to many bounding boxes
                pass
            elif np.isfinite(score_threshold): # use score threshold to remove bad bounding boxes
                new_aggregated_bounding_boxes += aux[current_boxes[aux[current_boxes,5] > score_threshold],:].tolist()
            else: # use all images
                new_aggregated_bounding_boxes += aux[current_boxes,:].tolist()
    return new_aggregated_bounding_boxes

def find_bounding_boxes_for_multiple_gt_boxes(gt_bounding_boxes, aggregated_bounding_boxes, choose_best_bb = True, score_threshold = -np.inf, max_boxes_per_images = np.inf, min_iou = -np.inf, **kwargs):
    """
    This function allows to select the bounding boxes of images, in case there are multiple ground truth bounding boxes.

    Parameters
    ----------
    gt_bounding_boxes : numpy array
        The ground truth bounding boxes contains [image_index, x, y, w, h], where
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
    aggregated_bounding_boxes : list
        Each entry is a sublist representing one bounding box. Each bounding box is determined by
            image_index : The line of the image in the temp_filename_dir.
            x : The x coordinate of the upper left corner of the bounding box.
            y : The y coordinate of the upper left corner of the bounding box.
            w : The width of the bounding box.
            h : The height of the bounding box.
            s : Score of the bounding box (How shure is yolo, that the bounding box is good).
            c : Estimated class.
    choose_best_bb : boolean, optional
        If more than one bounding box exists for an image (True), choose the best one. The default is True.
    score_threshold : float, optional
        The minimal score for a bounding box to be passed. The default is -np.inf.
    max_boxes_per_images : float, optional
        The maximum number of bounding boxes per images. The default is np.inf.
    min_iou : float, optional
        The minimal iou allowed to be chosen in case of multiple gt and predicted bounding boxes. The default is -np.inf.
    **kwargs : dict
        The kwargs can be used for optional parameters.

    Returns
    -------
    list
        [new_gt_bounding_boxes, new_aggregated_bounding_boxes].
        new_gt_bounding_boxes : The ground truth bounding boxes (lines in new_gt_bounding_boxes and new_aggregated_bounding_boxes are corresponding).
        new_aggregated_bounding_boxes : The aggregated bounding boxes.

    """
    new_gt_bounding_boxes = []
    aux = np.array(aggregated_bounding_boxes)
    new_aggregated_bounding_boxes = []
    for i in np.arange(np.max((len(np.unique(gt_bounding_boxes[:,0])), len(np.unique(aux[:,0]))))):
        current_gt_boxes = np.where(gt_bounding_boxes[:,0] == i)[0] if type(gt_bounding_boxes) != type(None) else []
        current_pred_boxes = np.where(aux[:,0] == i)[0]
        if len(current_pred_boxes) == 0:
            pass
        elif len(current_gt_boxes) == 1:
            if choose_best_bb: # for images with more than one bounding box choose the best one according to the score
                sorting = np.argsort(aux[current_pred_boxes,5])
                if aux[current_pred_boxes[sorting[-1]],5] > score_threshold: # further check, if best score is large enough
                    new_aggregated_bounding_boxes += [aux[current_pred_boxes[sorting[-1]],:].tolist()]
                    new_gt_bounding_boxes += [gt_bounding_boxes[current_gt_boxes[0]].tolist()]
            elif len(current_pred_boxes) > max_boxes_per_images: # remove images containing to many bounding boxes
                pass
            elif np.isfinite(score_threshold): # use score threshold to remove bad bounding boxes
                new_aggregated_bounding_boxes += [aux[aux[current_pred_boxes,5] > score_threshold,:].tolist()]
                new_gt_bounding_boxes += [gt_bounding_boxes[current_gt_boxes[0]].tolist()] * len([aux[aux[current_pred_boxes,5] > score_threshold,:].tolist()])
            else: # use all images
                new_aggregated_bounding_boxes += [aux[current_pred_boxes,:].tolist()]
                new_gt_bounding_boxes += [gt_bounding_boxes[current_gt_boxes[0]].tolist()] * len([aux[current_pred_boxes,:].tolist()])
        elif len(current_gt_boxes) > 1:
            iou_matrix = np.zeros((len(current_gt_boxes), len(current_pred_boxes)))
            assignment = np.zeros((len(current_gt_boxes), 3))
            for m in range(len(current_gt_boxes)):
                for n in range(len(current_pred_boxes)):
                    iou_matrix[m,n] = bbox_iou(gt_bounding_boxes[current_gt_boxes[m],-4:], aux[current_pred_boxes[n],1:5])
                n = np.argmax(iou_matrix[m,:])
                assignment[m,:] = [m,n,iou_matrix[m,n]]
            for n in range(len(current_pred_boxes)):
                n_index = assignment[:,1] == n
                if any(n_index):
                    assignment[np.where(n_index)[0][assignment[n_index,2] != max(assignment[n_index,2])],2] = -1
            for j in assignment[np.logical_and(assignment[:,2] != -1, assignment[:,2] > min_iou)]:
                new_aggregated_bounding_boxes += [aux[current_pred_boxes[int(j[1])],:].tolist()]
                new_gt_bounding_boxes += [gt_bounding_boxes[current_gt_boxes[int(j[0])]].tolist()]
    return [new_gt_bounding_boxes, new_aggregated_bounding_boxes]

def enhance_bbox_arrays(gt_bounding_boxes, pred_bounding_boxes):
    """
    This function enhances two 2D arrays, such that they have the same length and the id's in each line (column 0) are equal.

    Parameters
    ----------
    gt_bounding_boxes : numpy array
        The ground truth bounding box array.
    pred_bounding_boxes : numpy array
        The predicted bounding box array.

    Returns
    -------
    list
        [new_gt_bounding_boxes, new_pred_bounding_boxes].
        new_gt_bounding_boxes : The new ground truth bounding box array.
        new_pred_bounding_boxes : The new predicted bounding box array.

    """
    gt_index = np.argsort(gt_bounding_boxes[:,0])
    pred_index = np.argsort(pred_bounding_boxes[:,0])
    new_gt_bounding_boxes = []
    new_pred_bounding_boxes = []
    gt_count = 0
    pred_count = 0
    while pred_count < len(pred_index) and gt_count < len(gt_index):
        if gt_bounding_boxes[gt_index[gt_count],0] == pred_bounding_boxes[pred_index[pred_count],0]:
            new_gt_bounding_boxes.append(gt_bounding_boxes[gt_index[gt_count],:])
            new_pred_bounding_boxes.append(pred_bounding_boxes[pred_index[pred_count],:])
            pred_count += 1
        elif gt_bounding_boxes[gt_index[gt_count],0] > pred_bounding_boxes[pred_index[pred_count],0] and pred_count < (len(pred_index) - 1):
            pred_count += 1
        elif gt_bounding_boxes[gt_index[gt_count],0] < pred_bounding_boxes[pred_index[pred_count],0] and gt_count < (len(gt_index) - 1):
            gt_count += 1
    return np.array(new_gt_bounding_boxes), np.array(new_pred_bounding_boxes)

