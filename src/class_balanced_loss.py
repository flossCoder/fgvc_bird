#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# class_balanced_loss.py
# Copyright (C) 2022 flossCoder
# 
# class_balanced_loss is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# class_balanced_loss is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Mar 18 07:48:10 2022

@author: flossCoder
"""

import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np

def class_balanced_pre_factor(labels, beta = None):
    """
    This function implements the class balance factor from
    Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269. doi: 10.1109/CVPR.2019.00949.

    Parameters
    ----------
    labels : numpy array
        The labels in one-hot encoding.
    beta : float, optional
        The beta hyperparameter, with 0 <= beta <= 1 or None (beta is set to (number of classes - 1) / number of classes). The default is None.

    Returns
    -------
    tensorflow vector
        The balance pre factor is returned.

    """
    if beta == None:
        beta = (np.shape(labels)[1] - 1) / np.shape(labels)[1]
    number_of_samples = np.sum(labels, axis = 0)
    class_pre_factor = np.ones(np.shape(number_of_samples))
    class_pre_factor[number_of_samples != 0] = (1.0 - beta) / (1.0 - beta ** number_of_samples[number_of_samples != 0])
    return tf.convert_to_tensor(class_pre_factor, dtype = tf.float32)

def class_balanced_categorical_crossentropy_loss(labels, beta = None):
    """
    This is the wrapper function for the class balance categorical crossentropy from
    Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269. doi: 10.1109/CVPR.2019.00949.

    Parameters
    ----------
    labels : numpy array
        The labels in one-hot encoding.
    beta : float, optional
        The beta hyperparameter, with 0 < beta < 1 or None (beta is set to (number of classes - 1) / number of classes). The default is None.

    Returns
    -------
    class_balanced_categorical_crossentropy : function handle
        This function handle can be used with keras.

    """
    class_balanced_categorical_crossentropy_loss.beta = beta
    class_balanced_categorical_crossentropy_loss.class_pre_factor = class_balanced_pre_factor(labels, beta)
    def class_balanced_categorical_crossentropy(y_true, y_pred):
        """
        This function implements the class balance categorical crossentropy from
        Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269. doi: 10.1109/CVPR.2019.00949.

        Parameters
        ----------
        y_true : tensorflow tensor
            The true label in one-hot encoding.
        y_pred : tensorflow tensor
            The predicted label in one-hot encoding.

        Returns
        -------
        tensorflow float
            The class balanced categorical crossentropy.

        """
        pre_factor = math_ops.reduce_sum(y_true * class_balanced_categorical_crossentropy_loss.class_pre_factor, axis = 1)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pre_factor = tf.cast(pre_factor, ce.dtype)
        return (pre_factor * ce)
    return class_balanced_categorical_crossentropy
