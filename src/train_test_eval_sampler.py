#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train_test_eval_sampler.py
# Copyright (C) 2022 flossCoder
# 
# train_test_eval_sampler is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# train_test_eval_sampler is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Feb 21 10:20:05 2022

@author: flossCoder
"""

import numpy as np
from sklearn.model_selection import train_test_split

def generate_train_test_split(input_labels, test_number, seed = 42):
    """
    This function generates the train- and test split randomly, such that each category contains a fixed number of images.

    Parameters
    ----------
    input_labels : numpy array
        The labels of the dataset used for splitting.
    test_number : integer
        The number of images per category used for test.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Raises
    ------
    Exception
        The Exception is raised, in case any category contains less than test_number images.

    Returns
    -------
    list
        [train_index, test_index].
        train_index : The index of the data subset for training.
        test_index : The index of the data subset for testing.

    """
    [bins, counts] = np.unique(input_labels, return_counts = True)
    if any(counts <= test_number):
        raise Exception("Invalid test_number = %i, which is smaller than the smallest category with %i items"%(test_number, np.min(counts)))
    train_index = []
    test_index = []
    for i in range(len(counts)):
        current_ids = np.where(input_labels==bins[i])[0]
        x_train, x_test, y_train, y_test = train_test_split(current_ids, current_ids, test_size = test_number, random_state = seed)
        train_index = train_index + x_train.tolist()
        test_index = test_index + x_test.tolist()
    return [train_index, test_index]

def generate_train_test_eval_split(input_labels, train_frac, test_frac, val_frac, seed = 42):
    """
    This function generates the train-, test- and validation split randomly, such that the split holds for each class.

    Parameters
    ----------
    input_labels : numpy array
        The labels of the dataset used for splitting.
    train_frac : integer
        The integer fraction of 100 from all dataset items used for training.
    test_frac : integer
        The integer fraction of 100 from all dataset items used for testing.
    val_frac : integer
        The integer fraction of 100 from all dataset items used for validation.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Returns
    -------
    list
        [train_index, test_index, val_index].
        train_index : The index of the data subset for training.
        test_index : The index of the data subset for testing.
        val_index : The index of the data subset for validation.

    """
    [bins, counts, train_counts_per_category, test_counts_per_category, val_counts_per_category] = init_split(input_labels, train_frac, test_frac, val_frac)
    train_index = []
    test_index = []
    val_index = []
    for i in range(len(counts)):
        current_ids = np.where(input_labels==bins[i])[0]
        x_train, x_test, y_train, y_test = train_test_split(current_ids, current_ids, test_size = (test_frac / 100), random_state = seed)
        test_index = test_index + x_test.tolist()
        if val_frac != 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, x_train, test_size = ((val_frac * len(current_ids)) / (100 * len(x_train))), random_state = seed)
            val_index = val_index + x_val.tolist()
        train_index = train_index + x_train.tolist()
    
    return [train_index, test_index, val_index]

def sample_dataset(input_labels, test_number, val_number, min_training_samples, function_handle, function_args, seed = 42):
    """
    This function samples a dataset according to a given function.

    Parameters
    ----------
    input_labels : numpy array
        The labels of the dataset used for splitting.
    test_number : integer
        The number of images per category used for test.
    val_number : integer
        The number of images per category used for validation.
    min_training_samples : integer
        The minimum number of samples for training a category.
    function_handle : function handle
        The function describes how many samples should be used for a category i, where i is the number of the category in ascending sorting.
    function_args : list
        A list of arguments required for the function.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Raises
    ------
    Exception
        The exception is raised in case a category contains not enough samples.

    Returns
    -------
    list
        [train_index, test_index, val_index].
        train_index : The index of the data subset for training.
        test_index : The index of the data subset for testing.
        val_index : The index of the data subset for validation.

    """
    
    [bins, counts] = np.unique(input_labels, return_counts=True)
    index = np.argsort(counts)[::-1]
    bins = bins[index]
    counts = counts[index]
    train_index = []
    test_index = []
    val_index = []
    y_ = function_handle(np.arange(len(counts)), *function_args)
    train_number = np.round((y_ - np.min(y_)) / (np.max(y_) - np.min(y_)) * (np.max(counts) - test_number - val_number - min_training_samples)) + min_training_samples
    for i in range(len(counts)):
        current_ids = np.where(input_labels==bins[i])[0]
        x_train, x_test, y_train, y_test = train_test_split(current_ids, current_ids, test_size = (test_number / len(current_ids)), random_state = seed)
        test_index = test_index + x_test.tolist()
        if val_number != 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, x_train, test_size = (val_number / len(x_train)), random_state = seed)
            val_index = val_index + x_val.tolist()
        
        if train_number[i] > len(x_train):
            raise Exception("Invalid train number %s larger as number of samples %s for index %i"%(str(train_number[i]), str(x_train), i))
        elif train_number[i] < len(x_train):
            x_rem, x_train, y_rem, y_train = train_test_split(x_train, x_train, test_size = (train_number[i] / len(x_train)), random_state = seed)
        train_index = train_index + x_train.tolist()
    
    return [train_index, test_index, val_index]

def reversed_sampler(input_labels, val_number, h = 0, seed = 42):
    """
    This function adapts the reverse sampler from
    B. Zhou, Q. Cui, X.-S. Wei, und Z.-M. Chen, „BBN: Bilateral-Branch Network With Cumulative Learning for Long-Tailed Visual Recognition“, in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA, Juni 2020, S. 9716–9725. doi: 10.1109/CVPR42600.2020.00974.

    Parameters
    ----------
    input_labels : numpy array
        The labels of the dataset used for splitting.
    val_number : integer
        The number of images per category used for validation.
    h : integer, optional
        The number of head categories sampled out of the dataset. The default is 0.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility. The default is 42.

    Returns
    -------
    list
        [train_index, test_index, val_index].
        train_index : The index of the data subset for training.
        test_index : The index of the data subset for testing.
        val_index : The index of the data subset for validation.

    """
    np.random.seed(seed)
    [bins, counts] = np.unique(input_labels, return_counts=True)
    index = np.argsort(counts)[::-1]
    bins = bins[index]
    counts = counts[index]
    # calculate the reversed sampling number for the bins
    train_number = counts[h:][::-1]
    if h != 0:
        train_number = train_number * sum(counts) / sum(train_number)
        sum_counts = sum(counts)
        sum_train_number = sum(np.round(train_number))
        if sum_train_number != sum_counts:
            aux_index = np.arange(len(train_number))
            np.random.shuffle(aux_index)
            for i in aux_index:
                if sum_train_number > sum_counts:
                    train_number[i] = np.floor(train_number[i])
                elif sum_train_number < sum_counts:
                    train_number[i] = np.ceil(train_number[i])
                else:
                    train_number[i] = np.round(train_number[i])
                sum_train_number = sum(np.round(train_number))
        else:
            train_number = np.round(train_number)
        
    train_number = train_number - val_number
    train_index = []
    val_index = []
    for i in range(len(train_number)):
        current_ids = np.where(input_labels==bins[i+h])[0]
        if len(current_ids) < (train_number[i] + val_number):
            current_ids = current_ids[np.random.randint(len(current_ids), size = int(train_number[i] + val_number))]
        if val_number != 0:
            x_train, x_val, y_train, y_val = train_test_split(current_ids, current_ids, test_size = (val_number / len(current_ids)), random_state = seed)
            val_index = val_index + x_val.tolist()
            if train_number[i] != len(x_train):
                x_rem, x_train, y_rem, y_train = train_test_split(x_train, x_train, test_size = (train_number[i] / len(x_train)), random_state = seed)
        else:
            if train_number[i] != len(current_ids):
                x_rem, x_train, y_rem, y_train = train_test_split(current_ids, current_ids, test_size = (train_number[i] / len(current_ids)), random_state = seed)
            else:
                x_train = current_ids
        train_index = train_index + x_train.tolist()
    
    return [train_index, val_index]

def init_split(input_labels, train_frac, test_frac, val_frac):
    """
    This function initializes the split containing some validation checks.

    Parameters
    ----------
    input_labels : numpy array
        The labels of the dataset used for splitting.
    train_frac : integer
        The integer fraction of 100 from all dataset items used for training.
    test_frac : integer
        The integer fraction of 100 from all dataset items used for testing.
    val_frac : integer
        The integer fraction of 100 from all dataset items used for validation.

    Raises
    ------
    Exception
        The exception is raised in case the number of test or validation counts is less than 1 in any category, or the sum of test and validation counts is larger than the number of images in one category.

    Returns
    -------
    list
        [bins, counts, train_counts_per_category, test_counts_per_category, val_counts_per_category].
        bins : The unique label classes.
        counts : The counts per class.
        train_counts_per_category : The number of training samples per class.
        test_counts_per_category : The number of test samples per class.
        val_counts_per_category : The number of validation samples per class.

    """
    if (train_frac + test_frac + val_frac) != 100:
        raise Exception("Invalid split ratios")
    [bins, counts] = np.unique(input_labels, return_counts=True)
    train_counts_per_category = np.round(counts * train_frac / 100)
    test_counts_per_category = np.round(counts * test_frac / 100)
    val_counts_per_category = np.round(counts * val_frac / 100)
    for i in range(len(counts)):
        if test_frac != 0 and test_counts_per_category[i] < 1:
            raise Exception("Invalid value counts per category: %i for %s is less than %i for test"%(counts[i], bins[i], test_counts_per_category[i]))
        if val_frac != 0 and val_counts_per_category[i] < 1:
            raise Exception("Invalid value counts per category: %i for %s is less than %i for validation"%(counts[i], bins[i], val_counts_per_category[i]))
        if counts[i] < (val_counts_per_category[i]+test_counts_per_category[i]):
            raise Exception("Invalid configuration: %i for %s is less than %i for test and validation"%(counts[i], bins[i], (val_counts_per_category[i]+test_counts_per_category[i])))
    return [bins, counts, train_counts_per_category, test_counts_per_category, val_counts_per_category]
