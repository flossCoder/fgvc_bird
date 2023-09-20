#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simple_model.py
# Copyright (C) 2022 flossCoder
# 
# simple_model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# simple_model is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Feb 14 09:57:25 2022

@author: flossCoder
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import os
import numpy as np

import auxiliary_ml_functions as aux
from data_generator import ClassificationSequence, TrainingValidationSetCuiGenerator
from cnn_blocks import inception_resnet_block
import class_balanced_loss as cb_loss

def create_simple_inception_v3_model_3(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], l2 = 0.0001, learning_rate = 0.009, momentum = 0.9, cnn_trainable = False):
    """
    This function appendes an inception-resnet block to a simple inception v3
    cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    cnn_trainable : boolean, optional
        This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    x_input = tf.keras.layers.Input((image_resolution + (3,)), dtype=tf.float32)
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = cnn_trainable, arguments=dict(return_endpoints=True))
    if not(cnn_trainable):
        for i in [9, 13, 16, 26, 30, 46, 53, 54, 69, 75, 89, 114, 121, 132, 148, 152, 159, 178, 199, 200, 208, 214, 218, 230, 253, 276, 298, 313, 319, 335, 336, 337, 338, 356, 360, 365]:
            if cnn_model.variables[i] not in cnn_model.trainable_variables:
                cnn_model.trainable_variables.append(cnn_model.variables[i])
            if cnn_model.weights[i] not in cnn_model.trainable_weights:
                cnn_model.trainable_weights.append(cnn_model.weights[i])
    x = cnn_model(x_input)
    x = tf.keras.layers.SpatialDropout2D(0.5)(x['InceptionV3/Mixed_7c'])
    x = inception_resnet_block(x, l2, 2048)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(number_of_classes, activation = "softmax")(x)
    sgd_opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    model.compile(optimizer = sgd_opt, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inception_v3_model_2(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, momentum = 0.9, cnn_trainable = False):
    """
    This function creates a simple inception v3 cnn model (only the n last blocks are fine-tuned) with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    cnn_trainable : boolean, optional
        This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    x_input = tf.keras.layers.Input((image_resolution + (3,)), dtype=tf.float32)
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = cnn_trainable, arguments=dict(return_endpoints=True))
    if not(cnn_trainable):
        for i in [9, 13, 16, 26, 30, 46, 53, 54, 69, 75, 89, 114, 121, 132, 148, 152, 159, 178, 199, 200, 208, 214, 218, 230, 253, 276, 298, 313, 319, 335, 336, 337, 338, 356, 360, 365]:
            if cnn_model.variables[i] not in cnn_model.trainable_variables:
                cnn_model.trainable_variables.append(cnn_model.variables[i])
            if cnn_model.weights[i] not in cnn_model.trainable_weights:
                cnn_model.trainable_weights.append(cnn_model.weights[i])
    x = cnn_model(x_input)
    x = tf.keras.layers.Flatten()(x['InceptionV3/global_pool'])
    x = tf.keras.layers.Dense(number_of_classes, activation = "softmax")(x)
    sgd_opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    model.compile(optimizer = sgd_opt, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inception_v3_model_adam(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, momentum = 0.9, cnn_trainable = False):
    """
    This function creates a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    cnn_trainable : boolean, optional
        This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = cnn_trainable)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    adam = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(optimizer = adam, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inception_v3_model_rmsprop(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, momentum = 0.9, cnn_trainable = False):
    """
    This function creates a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    cnn_trainable : boolean, optional
        This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = cnn_trainable)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate = learning_rate, momentum = momentum)
    model.compile(optimizer = rmsprop, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_residual_simple_inception_v3_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function enhances a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    using residual connections.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    x_input = tf.keras.layers.Input((image_resolution + (3,)), dtype=tf.float32)
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = True, arguments=dict(return_endpoints=True))
    x = cnn_model(x_input)
    
    keys = ['InceptionV3/Conv2d_1a_3x3', 'InceptionV3/Conv2d_2a_3x3', 'InceptionV3/Conv2d_2b_3x3', 'InceptionV3/MaxPool_3a_3x3', 'InceptionV3/Conv2d_3b_1x1', 'InceptionV3/Conv2d_4a_3x3', 'InceptionV3/MaxPool_5a_3x3', 'InceptionV3/Mixed_5b', 'InceptionV3/Mixed_5c', 'InceptionV3/Mixed_5d', 'InceptionV3/Mixed_6a', 'InceptionV3/Mixed_6b', 'InceptionV3/Mixed_6c', 'InceptionV3/Mixed_6d', 'InceptionV3/Mixed_6e', 'InceptionV3/Mixed_7a', 'InceptionV3/Mixed_7b', 'InceptionV3/Mixed_7c']
    
    # 'InceptionV3/Mixed_7c': TensorShape([None, 8, 8, 2048])
    # 'InceptionV3/Mixed_7b': TensorShape([None, 8, 8, 2048])
    # 'InceptionV3/Mixed_7a': TensorShape([None, 8, 8, 1280])
    branch_1 = tf.keras.layers.Conv2D(2048, (1, 1), padding = "same")(x[keys[-3]])
    # 'InceptionV3/Mixed_6e': TensorShape([None, 17, 17, 768])
    # 'InceptionV3/Mixed_6d': TensorShape([None, 17, 17, 768])
    # 'InceptionV3/Mixed_6c': TensorShape([None, 17, 17, 768])
    # 'InceptionV3/Mixed_6b': TensorShape([None, 17, 17, 768])
    # 'InceptionV3/Mixed_6a': TensorShape([None, 17, 17, 768])
    branch_2 = tf.keras.layers.Add()([x[keys[-4]], x[keys[-5]], x[keys[-6]], x[keys[-7]], x[keys[-8]]])
    branch_2 = tf.keras.layers.MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = "valid")(branch_2)
    branch_2 = tf.keras.layers.Conv2D(2048, (1, 1), padding = "same")(branch_2)
    # 'InceptionV3/Mixed_5d': TensorShape([None, 35, 35, 288])
    # 'InceptionV3/Mixed_5c': TensorShape([None, 35, 35, 288])
    branch_3 = tf.keras.layers.Add()([x[keys[-9]], x[keys[-10]]])
    branch_3 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (4, 4), padding = "valid")(branch_3)
    branch_3 = tf.keras.layers.Conv2D(2048, (1, 1), padding = "same")(branch_3)
    # 'InceptionV3/Mixed_5b': TensorShape([None, 35, 35, 256])
    branch_4 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (4, 4), padding = "valid")(x[keys[-11]])
    branch_4 = tf.keras.layers.Conv2D(2048, (1, 1), padding = "same")(branch_4)
    # 'InceptionV3/MaxPool_5a_3x3': TensorShape([None, 35, 35, 192])
    branch_5 = tf.keras.layers.MaxPool2D(pool_size = (5, 5), strides = (4, 4), padding = "valid")(x[keys[-12]])
    branch_5 = tf.keras.layers.Conv2D(2048, (1, 1), padding = "same")(branch_5)
    # 'InceptionV3/Conv2d_4a_3x3': TensorShape([None, 71, 71, 192])
    # 'InceptionV3/Conv2d_3b_1x1': TensorShape([None, 73, 73, 80])
    # 'InceptionV3/MaxPool_3a_3x3': TensorShape([None, 73, 73, 64])
    # 'InceptionV3/Conv2d_2b_3x3': TensorShape([None, 147, 147, 64])
    # 'InceptionV3/Conv2d_2a_3x3': TensorShape([None, 147, 147, 32])
    # 'InceptionV3/Conv2d_1a_3x3': TensorShape([None, 149, 149, 32])
    x = tf.keras.layers.Add()([x[keys[-1]], x[keys[-2]], branch_1, branch_2, branch_3, branch_4, branch_5])
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(number_of_classes, activation = "softmax")(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[1]), (sgd_opt_dense, [model.layers[0]] + model.layers[2:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_shortened_simple_inception_v3_model_with_small_lr_fine_tuning(output_wd, output_name, cut_off_number, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a shortened version of the simple inception v3
    cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    cut_off_number : int
        The cut_off_number defines the block of the inception v3 network,
        which is used to pass out the feature vector. The number must be an integer in [0, 17].
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    x_input = tf.keras.layers.Input((image_resolution + (3,)), dtype=tf.float32)
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = True, arguments=dict(return_endpoints=True))
    x = cnn_model(x_input)
    
    keys = ['InceptionV3/Conv2d_1a_3x3', 'InceptionV3/Conv2d_2a_3x3', 'InceptionV3/Conv2d_2b_3x3', 'InceptionV3/MaxPool_3a_3x3', 'InceptionV3/Conv2d_3b_1x1', 'InceptionV3/Conv2d_4a_3x3', 'InceptionV3/MaxPool_5a_3x3', 'InceptionV3/Mixed_5b', 'InceptionV3/Mixed_5c', 'InceptionV3/Mixed_5d', 'InceptionV3/Mixed_6a', 'InceptionV3/Mixed_6b', 'InceptionV3/Mixed_6c', 'InceptionV3/Mixed_6d', 'InceptionV3/Mixed_6e', 'InceptionV3/Mixed_7a', 'InceptionV3/Mixed_7b', 'InceptionV3/Mixed_7c']
    print("Cutting Inception-V3 after block " + keys[cut_off_number])
    x = tf.keras.layers.GlobalAveragePooling2D()(x[keys[cut_off_number]])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(number_of_classes, activation = "softmax")(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x)
    
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[1]), (sgd_opt_dense, [model.layers[0]] + model.layers[2:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inception_v3_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = True)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_densenet121_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple densenet121 cnn model with weights from ImageNet.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on densenet121.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = tf.keras.applications.DenseNet121(weights = 'imagenet', include_top = False)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_densenet169_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple densenet169 cnn model with weights from ImageNet.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on densenet169.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = tf.keras.applications.DenseNet169(weights = 'imagenet', include_top = False)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_densenet201_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple densenet201 cnn model with weights from ImageNet.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on densenet201.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = tf.keras.applications.DenseNet201(weights = 'imagenet', include_top = False)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inceptionv3_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple inception v3 cnn model with weights from ImageNet.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = tf.keras.applications.InceptionV3(weights = 'imagenet', include_top = False)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_efficientNetB0_model_with_small_lr_fine_tuning(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, learning_rate_cnn = 0.0009, momentum = 0.9, momentum_cnn = 0.09):
    """
    This function creates a simple inception v3 cnn model with weights from ImageNet.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    learning_rate_cnn : float, optional
        The learning rate is a hyperparameter for fine tuning the cnn. The default is 0.0009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    momentum_cnn : float, optional
        The nesterov momentum is a hyperparameter for the cnn used during the training to increase training stability. The default is 0.09.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on EfficientNetB0.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = tf.keras.applications.EfficientNetB0(weights = 'imagenet', include_top = False)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = learning_rate_cnn, momentum = momentum_cnn, nesterov = True)
    sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def create_simple_inception_v3_model(output_wd, output_name, image_resolution, number_of_classes, save_model = True, loss = "categorical_crossentropy", metrics = ['accuracy'], learning_rate = 0.009, momentum = 0.9, cnn_trainable = False):
    """
    This function creates a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    image_resolution : tuple
        The input resolution of the images.
    number_of_classes : int
        The number of classes of the dataset.
    save_model : boolean, optional
        Flag indicating, whether to save the compiled model to file. The default is True.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    metrics : string or function handle
        The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
    learning_rate : float, optional
        The learning rate is a hyperparameter for the learning process. The default is 0.009.
    momentum : float, optional
        The nesterov momentum is a hyperparameter used during the training to increase training stability. The default is 0.9.
    cnn_trainable : boolean, optional
        This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

    Returns
    -------
    model : keras model
        The keras / tensorflow model based on inception v3.

    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = image_resolution + (3,)))
    cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = cnn_trainable)
    model.add(cnn_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(number_of_classes, activation = "softmax"))
    sgd_opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum, nesterov = True)
    model.compile(optimizer = sgd_opt, loss = loss, metrics = metrics)
    if save_model:
        aux.save_compiled_model(model, output_wd, output_name)
    return model

def train_model(model, output_wd, output_name, temp_filename_dir, training_image_line_index, labels_training, validation_image_line_index, labels_validation, batch_size, num_epochs, keras_net_application, number_of_classes = None, resolution = None, shuffel_training_samples = False, seed = None):
    """
    This function performes the training.

    Parameters
    ----------
    model : keras model
        The keras / tensorflow model based on inception v3.
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    training_image_line_index : numpy array
        The index of the temp file for the trainings set.
    labels_training : numpy array
        The labels of the trainings set.
    validation_image_line_index : numpy array
        The index of the temp file for the validation set.
    labels_validation : numpy array
        The labels of the validation set.
    batch_size : int
        The number of samples per training batch.
    num_epochs : int
        The number of training epochs.
    keras_net_application : module handle
        This module handle addresses a tf.keras.applications.MODULE module, such that keras_net_application.preprocess_input is valid.
    number_of_classes : int, optional
        The number of classes, if None, take the number of unique labels as number of classes. The default is None.
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.
    shuffel_training_samples : boolean, optional
        Shuffel training samples denotes, whether the dataset shall be shuffeled (if True) after each epoch. The default is False.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility, if it is not None. The default is None.

    Returns
    -------
    model :  keras model
        The keras / tensorflow model based on inception v3.
    history : keras history
        The history of the training.

    """
    callback = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_wd, "%s_{epoch:02d}.h5"%output_name), save_weights_only = True, save_freq = "epoch")
    history = model.fit(x = ClassificationSequence(temp_filename_dir, training_image_line_index, labels_training, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed), validation_data = ClassificationSequence(temp_filename_dir, validation_image_line_index, labels_validation, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed), epochs = num_epochs, callbacks = [callback])
    np.save(os.path.join(output_wd, "%s_hist.npy"%output_name), history.history)
    return model, history

def train_model_with_cui_sampling(model, output_wd, output_name, temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, number_of_classes = None, validation_split = 0.1, learning_rate_factor = 10., stage_split = 2, resolution = None, shuffel_training_samples = False, seed = None, split_seed = None):
    """
    This function performes the training.

    Parameters
    ----------
    model : keras model
        The keras / tensorflow model based on inception v3.
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    training_image_line_index : numpy array
        The index of the temp file for the trainings set.
    labels_training : numpy array
        The labels of the trainings set.
    validation_image_line_index : numpy array
        The index of the temp file for the validation set.
    labels_validation : numpy array
        The labels of the validation set.
    batch_size : int
        The number of samples per training batch.
    num_epochs : int
        The number of training epochs.
    keras_net_application : module handle
        This module handle addresses a tf.keras.applications.MODULE module, such that keras_net_application.preprocess_input is valid.
    number_of_classes : int, optional
        The number of classes, if None, take the number of unique labels as number of classes. The default is None.
    validation_split : float, optional
        The validation split implies the ratio between the size of the trainings set and the validation set. The default is 0.1.
    learning_rate_factor : float, optional
        The factor for decreasing the learning rate in the second phase. The default is 10.0
    stage_split : float, optional
        Stage split is the denominator for obtaining the number of epochs in the first training stage (epochs / stage_split). The default is 2.0
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.
    shuffel_training_samples : boolean, optional
        Shuffel training samples denotes, whether the dataset shall be shuffeled (if True) after each epoch. The default is False.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility, if it is not None. The default is None.
    split_seed : int, optional
        The split seed serves as a seed for the scikit-learn split function, if it is not None. The default is None.

    Returns
    -------
    model :  keras model
        The keras / tensorflow model based on inception v3.
    history : keras history
        The history of the training.

    """
    callback_first_phase = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_wd, "%s_{epoch:02d}.h5"%output_name), save_weights_only = True, save_freq = "epoch")
    main_generator = TrainingValidationSetCuiGenerator(temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, number_of_classes, validation_split, resolution, shuffel_training_samples, seed, split_seed)
    main_generator.generate_train_val_split()
    train_generator, val_generator = main_generator.get_train_val_objects()
    
    epochs_first_phase = int(np.ceil(num_epochs / stage_split))
    epochs_second_phase = num_epochs - epochs_first_phase
    
    # fit the first phase with the full and imbalanced dataset
    history = model.fit(x = train_generator, validation_data = val_generator, epochs = epochs_first_phase, callbacks = [callback_first_phase])
    
    # fit the second phase with the rebalanced dataset
    callback_second_phase = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_wd, "%s_{epoch:02d}_.h5"%output_name), save_weights_only = True, save_freq = "epoch")
    try:
        # use normal optimizer
        tf.keras.backend.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate / learning_rate_factor)
    except:
        # use multioptimizer
        for optimizer_spec in model.optimizer.optimizer_specs:
            tf.keras.backend.set_value(optimizer_spec['optimizer'].learning_rate, optimizer_spec['optimizer'].learning_rate / learning_rate_factor)
    main_generator.equilibrate_distribution()
    history_ = model.fit(x = train_generator, validation_data = val_generator, epochs = epochs_second_phase, callbacks = [callback_second_phase])
    aux.rename_files(output_wd, ["%s_%s_.h5"%(output_name, f"{epoch:02d}") for epoch in range(1, epochs_first_phase + 1)], ["%s_%s.h5"%(output_name, f"{epoch:02d}") for epoch in range(epochs_first_phase + 1, num_epochs + 1)])
    
    history = aux.join_history_objects(history, history_)
    np.save(os.path.join(output_wd, "%s_hist.npy"%output_name), history.history)
    return model, history

def train_model_with_cui_sampling_cbl(model, output_wd, output_name, temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, cbl_wrapper, beta = None, number_of_classes = None, validation_split = 0.1, learning_rate_factor = 10., stage_split = 2, resolution = None, shuffel_training_samples = False, seed = None, split_seed = None):
    """
    This function performes the training.

    Parameters
    ----------
    model : keras model
        The keras / tensorflow model based on inception v3.
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    training_image_line_index : numpy array
        The index of the temp file for the trainings set.
    labels_training : numpy array
        The labels of the trainings set.
    validation_image_line_index : numpy array
        The index of the temp file for the validation set.
    labels_validation : numpy array
        The labels of the validation set.
    batch_size : int
        The number of samples per training batch.
    num_epochs : int
        The number of training epochs.
    keras_net_application : module handle
        This module handle addresses a tf.keras.applications.MODULE module, such that keras_net_application.preprocess_input is valid.
    cbl_wrapper : function handle
        The wrapper function of the class balanced categorical crossentropy loss.
    beta : float, optional
        The beta hyperparameter, with 0 < beta < 1 or None (beta is set to (number of classes - 1) / number of classes). The default is None.
    number_of_classes : int, optional
        The number of classes, if None, take the number of unique labels as number of classes. The default is None.
    validation_split : float, optional
        The validation split implies the ratio between the size of the trainings set and the validation set. The default is 0.1.
    learning_rate_factor : float, optional
        The factor for decreasing the learning rate in the second phase. The default is 10.0
    stage_split : float, optional
        Stage split is the denominator for obtaining the number of epochs in the first training stage (epochs / stage_split). The default is 2.0
    resolution : tuple or float, optional
        The resolution describes the output size of an image. If resolution
        contains a tuple with two ints, they are used as resolution. If it
        contains a float, the number is considered as scaling factor preserving
        the original resolution. The default is None.
    shuffel_training_samples : boolean, optional
        Shuffel training samples denotes, whether the dataset shall be shuffeled (if True) after each epoch. The default is False.
    seed : integer, optional
        The seed used for numpy random to guarantee reproducibility, if it is not None. The default is None.
    split_seed : int, optional
        The split seed serves as a seed for the scikit-learn split function, if it is not None. The default is None.

    Returns
    -------
    model :  keras model
        The keras / tensorflow model based on inception v3.
    history : keras history
        The history of the training.

    """
    callback_first_phase = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_wd, "%s_{epoch:02d}.h5"%output_name), save_weights_only = True, save_freq = "epoch")
    main_generator = TrainingValidationSetCuiGenerator(temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, number_of_classes, validation_split, resolution, shuffel_training_samples, seed, split_seed)
    main_generator.generate_train_val_split()
    train_generator, val_generator = main_generator.get_train_val_objects()
    
    epochs_first_phase = int(np.ceil(num_epochs / stage_split))
    epochs_second_phase = num_epochs - epochs_first_phase
    
    # fit the first phase with the full and imbalanced dataset
    history = model.fit(x = train_generator, validation_data = val_generator, epochs = epochs_first_phase, callbacks = [callback_first_phase])
    
    # fit the second phase with the rebalanced dataset
    callback_second_phase = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(output_wd, "%s_{epoch:02d}_.h5"%output_name), save_weights_only = True, save_freq = "epoch")
    try:
        # use normal optimizer
        tf.keras.backend.set_value(model.optimizer.learning_rate, model.optimizer.learning_rate / learning_rate_factor)
    except:
        # use multioptimizer
        for optimizer_spec in model.optimizer.optimizer_specs:
            tf.keras.backend.set_value(optimizer_spec['optimizer'].learning_rate, optimizer_spec['optimizer'].learning_rate / learning_rate_factor)
    main_generator.equilibrate_distribution()
    cbl_wrapper.pre_factor = cb_loss.class_balanced_pre_factor(np.concatenate((main_generator.training_generator.images_class_labels, main_generator.validation_generator.images_class_labels)), beta)
    history_ = model.fit(x = train_generator, validation_data = val_generator, epochs = epochs_second_phase, callbacks = [callback_second_phase])
    aux.rename_files(output_wd, ["%s_%s_.h5"%(output_name, f"{epoch:02d}") for epoch in range(1, epochs_first_phase + 1)], ["%s_%s.h5"%(output_name, f"{epoch:02d}") for epoch in range(epochs_first_phase + 1, num_epochs + 1)])
    
    history = aux.join_history_objects(history, history_)
    np.save(os.path.join(output_wd, "%s_hist.npy"%output_name), history.history)
    return model, history
