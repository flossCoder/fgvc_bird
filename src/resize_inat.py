#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# resize_inat.py
# Copyright (C) 2022 flossCoder
# 
# resize_inat is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# resize_inat is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jun 20 07:02:54 2022

@author: flossCoder
"""

import os
import sys
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy

from image_processing import resize_images, crop_images_at_bounding_boxes, resize_bounding_boxes, generate_bounding_boxes_crops, generate_bounding_boxes_crops_with_zero_padding
from aux_io import load_json, dump_json

def aux_generate_output_directories(wd, input_dir, output_dir, sub_dir = None):
    """
    This function generates the directory tree of the input directory for the output recursively.

    Parameters
    ----------
    wd : string
        The basic working directory.
    input_dir : string
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
    sub_dir : string, optional
        The sub_dir denotes the current sub directory in the image directory tree. The default is None.

    Returns
    -------
    None.

    """
    aux = os.path.join(wd, input_dir) if sub_dir is None else os.path.join(wd, input_dir, sub_dir)
    for d in os.listdir(aux):
        if os.path.isdir(os.path.join(aux, d)):
            os.mkdir(os.path.join(wd, output_dir, d) if sub_dir is None else os.path.join(wd, output_dir, sub_dir, d))
            aux_generate_output_directories(wd, input_dir, output_dir, d if sub_dir is None else os.path.join(sub_dir, d))

def resize_inat(wd, json_dir, input_dir, output_dir, resolution, resize_at_bounding_box, train_filename = "train2017.json", val_filename = "val2017.json", test_filename = "test2017.json", train_bboxes_filename = "train_2017_bboxes.json", val_bboxes_filename = "val_2017_bboxes.json", test_bboxes_filename = "test_2017_bboxes.json"):
    """
    This function resizes the iNat dataset.

    Parameters
    ----------
    wd : string
        The basic working directory.
    json_dir : string
        The json_dir describes the sub-directory of wd containing the json files.
    input_dir : string
        The sub-directory of the input data.
    output_dir : string
        The sub-directory of the output data.
    resolution : tuple
        Each entry of the resolution describes the output size of an image.
    resize_at_bounding_box : int
        Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
    train_filename : string, optional
        The name of the json-file containing the training set. The default is "train2017.json".
    val_filename : string, optional
        The name of the json-file containing the validation set. The default is "val2017.json".
    test_filename : string, optional
        The name of the json-file containing the test set. The default is "test2017.json".
    train_bboxes_filename : string, optional
        The name of the json-file containing the bounding box annotations of the training set. The default is "train_2017_bboxes.json".
    val_bboxes_filename : string, optional
        The name of the json-file containing the bounding box annotations of the validation set. The default is "val_2017_bboxes.json".
    test_bboxes_filename : string, optional
        The name of the json-file containing the bounding box annotations of the test set. The default is "test_2017_bboxes.json".

    Raises
    ------
    Exception
        The exception is raised in case an invalid resize_at_bounding_box parameter is given.

    Returns
    -------
    None.

    """
    # load the json data
    json_wd = os.path.join(wd, json_dir)
    train_data = load_json(json_wd, train_filename)
    val_data = load_json(json_wd, val_filename)
    test_data = load_json(json_wd, test_filename)
    train_bboxes = load_json(json_wd, train_bboxes_filename)
    val_bboxes = load_json(json_wd, val_bboxes_filename)
    test_bboxes = load_json(json_wd, test_bboxes_filename)
    # setup directories in output
    aux_generate_output_directories(wd, input_dir, output_dir)
    
    # resize the training data
    new_train_data = deepcopy(train_data)
    new_train_data["images"] = []
    new_train_bboxes = deepcopy(train_bboxes)
    new_train_bboxes["images"] = []
    new_train_bboxes["annotations"] = []
    if resize_at_bounding_box == 0:
        for i in train_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in train_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_train_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_train_bboxes["images"].append(new_bb_image)
                new_train_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_train_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 1:
        for i in train_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in train_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_train_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_train_bboxes["images"].append(new_bb_image)
                new_train_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_train_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 2:
        for i in train_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in train_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            image = resize_images([image], resolution)[0]
            new_train_data["images"].append(deepcopy(i))
            for bbox_elem in bbox:
                bb = resize_bounding_boxes([[0] + [int(b) for b in bbox_elem["bbox"]]], [original_resolution], resolution)[0]
                new_bbox_elem = deepcopy(bbox_elem)
                new_bbox_elem["bbox"] = bb[1:]
                new_train_bboxes["annotations"].append(new_bbox_elem)
            image.save("/".join([wd] + [output_dir] + splitted[1:]))
    elif resize_at_bounding_box == 3:
        for i in train_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in train_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_train_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_train_bboxes["images"].append(new_bb_image)
                new_train_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_train_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in train_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_train_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    else:
        raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
    
    aux = train_bboxes_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_train_bboxes)
    aux = train_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_train_data)
    
    # resize the validation data
    new_val_data = deepcopy(val_data)
    new_val_data["images"] = []
    new_val_bboxes = deepcopy(val_bboxes)
    new_val_bboxes["images"] = []
    new_val_bboxes["annotations"] = []
    if resize_at_bounding_box == 0:
        for i in val_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in val_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_val_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_val_bboxes["images"].append(new_bb_image)
                new_val_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_val_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 1:
        for i in val_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in val_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_val_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_val_bboxes["images"].append(new_bb_image)
                new_val_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_val_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 2:
        for i in val_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in val_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            image = resize_images([image], resolution)[0]
            new_val_data["images"].append(deepcopy(i))
            for bbox_elem in bbox:
                bb = resize_bounding_boxes([[0] + [int(b) for b in bbox_elem["bbox"]]], [original_resolution], resolution)[0]
                new_bbox_elem = deepcopy(bbox_elem)
                new_bbox_elem["bbox"] = bb[1:]
                new_val_bboxes["annotations"].append(new_bbox_elem)
            image.save("/".join([wd] + [output_dir] + splitted[1:]))
    elif resize_at_bounding_box == 3:
        for i in val_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in val_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_val_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_val_bboxes["images"].append(new_bb_image)
                new_val_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_val_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in val_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_val_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    else:
        raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
    
    aux = val_bboxes_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_val_bboxes)
    aux = val_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_val_data)
    
    # resize the test data
    new_test_data = deepcopy(test_data)
    new_test_data["images"] = []
    new_test_bboxes = deepcopy(test_bboxes)
    new_test_bboxes["images"] = []
    new_test_bboxes["annotations"] = []
    if resize_at_bounding_box == 0:
        for i in test_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in test_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_test_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_test_bboxes["images"].append(new_bb_image)
                new_test_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = crop_images_at_bounding_boxes([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_test_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 1:
        for i in test_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in test_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_test_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_test_bboxes["images"].append(new_bb_image)
                new_test_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_test_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    elif resize_at_bounding_box == 2:
        for i in test_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in test_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            image = resize_images([image], resolution)[0]
            new_test_data["images"].append(deepcopy(i))
            for bbox_elem in bbox:
                bb = resize_bounding_boxes([[0] + [int(b) for b in bbox_elem["bbox"]]], [original_resolution], resolution)[0]
                new_bbox_elem = deepcopy(bbox_elem)
                new_bbox_elem["bbox"] = bb[1:]
                new_test_bboxes["annotations"].append(new_bbox_elem)
            image.save("/".join([wd] + [output_dir] + splitted[1:]))
    elif resize_at_bounding_box == 3:
        for i in test_data["images"]:
            splitted = i["file_name"].split("/")
            image = Image.open("/".join([wd] + [input_dir] + splitted[1:])).convert("RGB")
            bbox = [j for j in test_bboxes["annotations"] if j["image_id"] == i["id"]]
            original_resolution = image.size
            if len(bbox) == 1:
                bbox = bbox[0]
                [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox["bbox"]]], resolution)
                image = a[0]
                new_bbox_elem = deepcopy(bbox)
                new_bbox_elem["bbox"] = b[0][1:]
                new_test_bboxes["annotations"].append(new_bbox_elem)
                new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox["image_id"]][0])
                new_test_bboxes["images"].append(new_bb_image)
                new_test_data["images"].append(deepcopy(i))
                image.save("/".join([wd] + [output_dir] + splitted[1:]))
            elif len(bbox) > 1:
                for bbox_elem in bbox:
                    [a, b] = generate_bounding_boxes_crops_with_zero_padding([image], [[0] + [int(b) for b in bbox_elem["bbox"]]], resolution)
                    image = a[0]
                    new_bbox_elem = deepcopy(bbox_elem)
                    new_bbox_elem["bbox"] = b[0][1:]
                    new_test_bboxes["annotations"].append(new_bbox_elem)
                    new_bb_image = deepcopy([j for j in test_bboxes["images"] if j["id"] == bbox_elem["image_id"]][0])
                    new_bb_image["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_bboxes["images"].append(new_bb_image)
                    new_i = deepcopy(i)
                    new_i["file_name"] = "/".join(splitted[0:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]])
                    new_test_data["images"].append(new_i)
                    image.save("/".join([wd] + [output_dir] + splitted[1:-1] + [splitted[-1].split(".")[0] + "_%i."%bbox_elem["id"] + splitted[-1].split(".")[1]]))
    else:
        raise Exception("Invalid resize_at_bounding_box = %s"%str(resize_at_bounding_box))
    
    aux = test_bboxes_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_test_bboxes)
    aux = test_filename.split(".")
    dump_json(wd, aux[0] + "_" + output_dir + "." + aux[1], new_test_data)

def main(argv):
    """
    The main function executing this skript based on the shell parameter.

    Parameters
    ----------
    argv : list
        The input parameter: [wd, json_dir, input_dir, output_dir, resolution width, resolution hight, resize_at_bounding_box, train_filename, val_filename, test_filename, train_bboxes_filename, val_bboxes_filename, test_bboxes_filename].
        wd : The basic working directory.
        json_dir : The json_dir describes the sub-directory of wd containing the json files.
        input_dir : The sub-directory of the input data.
        output_dir : The sub-directory of the output data.
        resolution width: The width of the resolution.
        resolution hight: The hight of the resolution.
        resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).
        train_filename : The name of the json-file containing the training set.
        val_filename : The name of the json-file containing the validation set.
        test_filename : The name of the json-file containing the test set.
        train_bboxes_filename : The name of the json-file containing the bounding box annotations of the training set.
        val_bboxes_filename : The name of the json-file containing the bounding box annotations of the validation set.
        test_bboxes_filename : The name of the json-file containing the bounding box annotations of the test set.

    Raises
    ------
    Exception
        The exception is raised, in case not enough parameter are passed to main.

    Returns
    -------
    None.

    """
    if len(argv) != 13:
        print("The input parameter: [wd, json_dir, input_dir, output_dir, resolution, resize_at_bounding_box, train_filename, val_filename, test_filename, train_bboxes_filename, val_bboxes_filename, test_bboxes_filename].")
        print("wd : The basic working directory.")
        print("json_dir : The json_dir describes the sub-directory of wd containing the json files.")
        print("input_dir : The sub-directory of the input data.")
        print("output_dir : The sub-directory of the output data.")
        print("resolution width: The width of the resolution.")
        print("resolution hight: The hight of the resolution.")
        print("resize_at_bounding_box : Use the bounding box for resizing and cropping maintaining the aspect ratio (0), crop and resize at bounding box without maintaining the aspect ratio (1), resize whole image (2) or crop and resize at bounding box maintaining the aspect ratio by zero padding (3).")
        print("train_filename : The name of the json-file containing the training set.")
        print("val_filename : The name of the json-file containing the validation set.")
        print("test_filename : The name of the json-file containing the test set.")
        print("train_bboxes_filename : The name of the json-file containing the bounding box annotations of the training set.")
        print("val_bboxes_filename : The name of the json-file containing the bounding box annotations of the validation set.")
        print("test_bboxes_filename : The name of the json-file containing the bounding box annotations of the test set.")
        raise Exception("Wrong number of parameter")
    wd = argv[0]
    json_dir = argv[1]
    input_dir = argv[2]
    output_dir = argv[3]
    resolution = (int(argv[4]), int(argv[5]))
    resize_at_bounding_box = int(argv[6])
    train_filename = argv[7]
    val_filename = argv[8]
    test_filename = argv[9]
    train_bboxes_filename = argv[10]
    val_bboxes_filename = argv[11]
    test_bboxes_filename = argv[12]
    
    resize_inat(wd, json_dir, input_dir, output_dir, resolution, resize_at_bounding_box, train_filename, val_filename, test_filename, train_bboxes_filename, val_bboxes_filename, test_bboxes_filename)

if __name__ == "__main__":
    main(sys.argv[1:])
