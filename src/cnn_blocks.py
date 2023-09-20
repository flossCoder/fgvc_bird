#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# cnn_blocks.py
# Copyright (C) 2022 flossCoder
# 
# cnn_blocks is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# cnn_blocks is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Mar 29 09:14:31 2022

@author: flossCoder
"""

import tensorflow as tf

def inception_resnet_block(x, l2, filters = 5):
    """
    This function defines an inception resnet block developed from
    C. Szegedy, S. Ioffe, V. Vanhoucke, und A. Alemi, „Inception-v4, Inception-ResNet
    and the Impact of Residual Connections on Learning“, AAAI, Bd. 31, Nr. 1,
    Feb. 2017, Zugegriffen: 12. Mai 2021.
    [Online]. Verfügbar unter: https://ojs.aaai.org/index.php/AAAI/article/view/11231 Fig. 4 (right).

    Parameters
    ----------
    x : keras tensor
        The input tensor.
    l2 : float
        L2 is the regularization penalty hyperparameter.
    filters : int, optional
        The number of Conv2D filter. The default is 5.

    Returns
    -------
    x : keras tensor
        The output tensor.

    """
    branch_1 = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(x)
    branch_1 = tf.keras.layers.Conv2D(filters, (1, 3), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(branch_1)
    branch_1 = tf.keras.layers.Conv2D(filters, (3, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(branch_1)
    
    branch_2 = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(x)
    
    branch_12 = tf.keras.layers.concatenate([branch_1, branch_2], axis = 3)
    
    branch_12 = tf.keras.layers.Conv2D(filters, (1, 1), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(branch_12)
    
    x = tf.keras.layers.Add()([branch_12, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def resnet_block(x, l2, filters = 5):
    """
    This function defines a resnet block developed from
    K. He, X. Zhang, S. Ren, und J. Sun, „Identity Mappings in Deep Residual Networks“,
    in Computer Vision – ECCV 2016, Cham, 2016, S. 630–645. doi: 10.1007/978-3-319-46493-0_38 Fig. 4(a).

    Parameters
    ----------
    x : keras tensor
        The input tensor.
    l2 : float
        L2 is the regularization penalty hyperparameter.
    filters : int, optional
        The number of Conv2D filter. The default is 5.

    Returns
    -------
    x : keras tensor
        The output tensor.

    """
    x_skip = x
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding = 'same', kernel_regularizer = tf.keras.regularizers.l2(l2))(x)
    x = tf.keras.layers.BatchNormalization(axis = 3)(x)
    
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    return x
