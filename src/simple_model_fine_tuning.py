#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simple_model_fine_tuning.py
# Copyright (C) 2022 flossCoder
# 
# simple_model_fine_tuning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# simple_model_fine_tuning is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Thu May  5 06:43:04 2022

@author: flossCoder
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa

import auxiliary_ml_functions as aux
import class_balanced_loss as cbl
from hypermodel import AbstractHyperModel, AbstractRebalanceCuiHyperModel

class SimpleInceptionV3FineTuningModel(AbstractHyperModel):
    """
    This class defines a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    """
    def __init__(self, image_resolution, number_of_classes, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy']):
        """
        This function initializes the SimpleInceptionV3FineTuningModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.

        Returns
        -------
        None.

        """
        AbstractHyperModel.__init__(self, image_resolution, number_of_classes, loss, metrics, True)
        self.hp_learning_rate_cnn_min_value = hp_learning_rate_cnn_min_value
        self.hp_learning_rate_cnn_max_value = hp_learning_rate_cnn_max_value
        self.hp_learning_rate_cnn_sampling = hp_learning_rate_cnn_sampling
        self.hp_learning_rate_cnn_default = hp_learning_rate_cnn_default
        self.hp_momentum_cnn_min_value = hp_momentum_cnn_min_value
        self.hp_momentum_cnn_max_value = hp_momentum_cnn_max_value
        self.hp_momentum_cnn_step = hp_momentum_cnn_step
        self.hp_momentum_cnn_default = hp_momentum_cnn_default
        self.hp_learning_rate_dense_min_value = hp_learning_rate_dense_min_value
        self.hp_learning_rate_dense_max_value = hp_learning_rate_dense_max_value
        self.hp_learning_rate_dense_sampling = hp_learning_rate_dense_sampling
        self.hp_learning_rate_dense_default = hp_learning_rate_dense_default
        self.hp_momentum_dense_min_value = hp_momentum_dense_min_value
        self.hp_momentum_dense_max_value = hp_momentum_dense_max_value
        self.hp_momentum_dense_step = hp_momentum_dense_step
        self.hp_momentum_dense_default = hp_momentum_dense_default
    
    def build(self, hps):
        """
        This function creates a simple inception v3 cnn model with weights from
        "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
        Fine-Grained Categorization and Domain-Specific Transfer Learning“,
        in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
        Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
        
        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.
        
        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.
        
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = self.image_resolution))
        cnn_model, hps = self.setup_backbone_cnn(hps)
        model.add(cnn_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.number_of_classes, activation = "softmax"))
        
        hp_learning_rate_cnn = hps.Float("learning_rate_cnn", min_value = self.hp_learning_rate_cnn_min_value, max_value = self.hp_learning_rate_cnn_max_value, sampling = self.hp_learning_rate_cnn_sampling, default = self.hp_learning_rate_cnn_default)
        hp_momentum_cnn = hps.Float("momentum_cnn", min_value = self.hp_momentum_cnn_min_value, max_value = self.hp_momentum_cnn_max_value, step = self.hp_momentum_cnn_step, default = self.hp_momentum_cnn_default)
        sgd_opt_cnn = tf.keras.optimizers.SGD(learning_rate = hp_learning_rate_cnn, momentum = hp_momentum_cnn, nesterov = True)
        
        hp_learning_rate_dense = hps.Float("learning_rate_dense", min_value = self.hp_learning_rate_dense_min_value, max_value = self.hp_learning_rate_dense_max_value, sampling = self.hp_learning_rate_dense_sampling, default = self.hp_learning_rate_dense_default)
        hp_momentum_dense = hps.Float("momentum_dense", min_value = self.hp_momentum_dense_min_value, max_value = self.hp_momentum_dense_max_value, step = self.hp_momentum_dense_step, default = self.hp_momentum_dense_default)
        sgd_opt_dense = tf.keras.optimizers.SGD(learning_rate = hp_learning_rate_dense, momentum = hp_momentum_dense, nesterov = True)
        
        optimizers_layers_tuple = [(sgd_opt_cnn, model.layers[0]), (sgd_opt_dense, model.layers[1:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_layers_tuple)
        
        model.compile(optimizer = optimizer, loss = self.loss, metrics = self.metrics)
        return model
    
    def setup_backbone_cnn(self, hps):
        """
        This function sets up the backbone cnn via tensorflow_hub.
        
        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        tensorflow_hub.keras_layer.KerasLayer
            The weights and architecture of the cui inception v3.
        hps : HyperParameters
            The hyperparameters object for the model.

        """
        return hub.KerasLayer(aux.resolve_cui_dir(), trainable = True), hps

class SimpleEfficientNetB0FineTuningModel(SimpleInceptionV3FineTuningModel):
    """
    This class defines the EfficientNetB0 from
    "M. Tan und Q. Le, „EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks“,
    in International Conference on Machine Learning, Long Beach, California, Mai 2019, S. 6105–6114.
    Accessed: May 12, 2021. [Online]. Avaliable: http://proceedings.mlr.press/v97/tan19a.html."
    with weights obtained from imagenet pre training.
    Furthermore a drop out is used to change the network depth stochastically:
    "G. Huang, Y. Sun, Z. Liu, D. Sedra, und K. Q. Weinberger, „Deep Networks with Stochastic Depth“,
    in Computer Vision – ECCV 2016, Cham, 2016, S. 646–661. doi: 10.1007/978-3-319-46493-0_39."
    """
    def __init__(self, image_resolution, number_of_classes, hp_drop_connect_rate_min = 0.0, hp_drop_connect_rate_max = 1.0, hp_drop_connect_rate_step = 0.1, hp_drop_connect_rate_default = 0.2, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy']):
        """
        This function initializes the SimpleEfficientNetB0FineTuningModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        hp_drop_connect_rate_min : float, optional
            The minimum value for the hyperparameter drop connect rate of the cnn. The default is 0.0.
        hp_drop_connect_rate_max : float, optional
            The maximum value for the hyperparameter drop connect rate of the cnn. The default is 1.0.
        hp_drop_connect_rate_step : float, optional
            The step size for the hyperparameter drop connect rate of the cnn. The default is 0.1.
        hp_drop_connect_rate_default : float, optional
            The default value for the hyperparameter drop connect rate of the cnn. The default is 0.2.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.

        Returns
        -------
        None.

        """
        self.hp_drop_connect_rate_min = hp_drop_connect_rate_min
        self.hp_drop_connect_rate_max = hp_drop_connect_rate_max
        self.hp_drop_connect_rate_step = hp_drop_connect_rate_step
        self.hp_drop_connect_rate_default = hp_drop_connect_rate_default
        super().__init__(image_resolution, number_of_classes, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, loss, metrics)
    
    def setup_backbone_cnn(self, hps):
        """
        This function sets up the efficientnet b0 using a dropout rate generating a stochastic depth.

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        cnn_model : tf.keras.applications.efficientnet.EfficientNetB0
            The weights and architecture of EfficientNetB0.
        hps : HyperParameters
            The hyperparameters object for the model.

        """
        hp_drop_connect_rate = hps.Float("drop_connect_rate", min_value = self.hp_drop_connect_rate_min, max_value = self.hp_drop_connect_rate_max, step = self.hp_drop_connect_rate_step, default = self.hp_drop_connect_rate_default)
        cnn_model = tf.keras.applications.efficientnet.EfficientNetB0(weights = 'imagenet', include_top = False, drop_connect_rate = hp_drop_connect_rate)
        return cnn_model, hps

class SimpleInceptionV3ImageNetFineTuningModel(SimpleInceptionV3FineTuningModel):
    """
    This class implements an InceptionNetV3 similar to class SimpleInceptionV3FineTuningModel.
    The weights are taken from the standard keras implementation (they where pretrained on imagenet).
    """
    def __init__(self, image_resolution, number_of_classes, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy']):
        """
        This function initializes the SimpleInceptionV3ImageNetFineTuningModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.

        Returns
        -------
        None.

        """
        SimpleInceptionV3FineTuningModel.__init__(self, image_resolution, number_of_classes, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, loss, metrics)

    def setup_backbone_cnn(self, hps):
        """
        This function sets up the inception v3 backbone.

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        cnn_model : tf.keras.applications.efficientnet.EfficientNetB0
            The weights and architecture of EfficientNetB0.
        hps : HyperParameters
            The hyperparameters object for the model.

        """
        cnn_model = tf.keras.applications.inception_v3.InceptionV3(weights = 'imagenet', include_top = False)
        return cnn_model, hps

class SimpleInceptionV3FineTuningCuiSamplingModel(AbstractRebalanceCuiHyperModel, SimpleInceptionV3FineTuningModel):
    """
    This class defines a simple inception v3 cnn model with weights and sampling sheme from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    """
    def __init__(self, image_resolution, number_of_classes, hp_stage_split_min_value, hp_stage_split_max_value, hp_stage_split_step, hp_stage_split_default = 2, hp_learning_rate_factor_min_value = 10, hp_learning_rate_factor_max_value = 1000, hp_learning_rate_factor_sampling = "LOG", hp_learning_rate_factor_default = 10, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy'], cnn_trainable = False, main_generator = None):
        """
        This function initializes the SimpleInceptionV3FineTuningCuiSamplingModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        hp_stage_split_min_value : float
            The minimum value of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_max_value : float
            The maximum value of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_step : float
            The step size of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_default : float, optional
            The default value of the hyperparameter stage_split. The default is 2.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_learning_rate_factor_min_value : float, optional
            The minimum value of the hyperparameter learning_rate_factor. The default is 10.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_max_value : float, optional
            The maximum value of the hyperparameter learning_rate_factor. The default is 1000.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_sampling : string, optional
            The sampling type of the hyperparameter learning_rate_factor. The default is "LOG".
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_default : float, optional
            The default value of the hyperparameter learning_rate_factor. The default is 10.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
        cnn_trainable : boolean, optional
            This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.
        main_generator : TrainingValidationSetCuiGenerator, optional
            The main_generator is the object, which is responsible for the generation of training- and validation generator. The default is False.

        Returns
        -------
        None.

        """
        AbstractRebalanceCuiHyperModel.__init__(self, image_resolution, number_of_classes, loss, metrics, cnn_trainable, main_generator)
        SimpleInceptionV3FineTuningModel.__init__(self, image_resolution, number_of_classes, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, loss, metrics)
        self.hp_stage_split_min_value = hp_stage_split_min_value
        self.hp_stage_split_max_value = hp_stage_split_max_value
        self.hp_stage_split_step = hp_stage_split_step
        self.hp_stage_split_default = hp_stage_split_default
        self.hp_learning_rate_factor_min_value = hp_learning_rate_factor_min_value
        self.hp_learning_rate_factor_max_value = hp_learning_rate_factor_max_value
        self.hp_learning_rate_factor_sampling = hp_learning_rate_factor_sampling
        self.hp_learning_rate_factor_default = hp_learning_rate_factor_default
    
    def build(self, hps):
        """
        This function creates a simple inception v3 cnn model with weights and sampling sheme from
        "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
        Fine-Grained Categorization and Domain-Specific Transfer Learning“,
        in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
        Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.

        """
        hps = self.cui_sampling_hps(hps)
        return SimpleInceptionV3FineTuningModel.build(self, hps)
    
    def cui_sampling_hps(self, hps):
        """
        This function defines the hyperparameter operations for the cui sampling.

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        hps : HyperParameters
            The hyperparameters object for the model.

        """
        hps.Float("stage_split", min_value = self.hp_stage_split_min_value, max_value = self.hp_stage_split_max_value, step = self.hp_stage_split_step, default = self.hp_stage_split_default)
        hps.Float("learning_rate_factor", min_value = self.hp_learning_rate_factor_min_value, max_value = self.hp_learning_rate_factor_max_value, sampling = self.hp_learning_rate_factor_sampling, default = self.hp_learning_rate_factor_default)
        return hps

class SimpleInceptionV3FineTuningCBLModel(SimpleInceptionV3FineTuningModel):
    """
    This class defines a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    using the class balanced categorical crossentropy loss from
    "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based
    on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision
    and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269.
    doi: 10.1109/CVPR.2019.00949."
    """
    def __init__(self, image_resolution, number_of_classes, labels, hp_beta_min_value = 0.0001, hp_beta_max_value = 0.9999, hp_beta_sampling = 'LOG', hp_beta_default = 0.5, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, metrics = ['accuracy']):
        """
        This function initializes the SimpleInceptionV3FineTuningCBLModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        labels : numpy-array
            The given labels.
        hp_beta_min_value : float, optional
            The minimum value for the hyperparameter beta. The default is 0.0001.
        hp_beta_max_value : float, optional
            The maximum value for the hyperparameter beta. The default is 0.9999.
        hp_beta_sampling : string, optional
            The sampling type for the hyperparameter beta. The default is 'LOG'.
        hp_beta_default : float, optional
            The default value for the hyperparameter beta. The default is 0.5.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.

        Returns
        -------
        None.

        """
        SimpleInceptionV3FineTuningModel.__init__(self, image_resolution, number_of_classes, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, None, metrics)
        self.cbl_wrapper = cbl.class_balanced_categorical_crossentropy_loss
        self.labels = aux.preprocess_labels(labels, number_of_classes)
        self.hp_beta_min_value = hp_beta_min_value
        self.hp_beta_max_value = hp_beta_max_value
        self.hp_beta_sampling = hp_beta_sampling
        self.hp_beta_default = hp_beta_default
    
    def build(self, hps):
        """
        This function creates a simple inception v3 cnn model with weights from
        "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
        Fine-Grained Categorization and Domain-Specific Transfer Learning“,
        in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
        Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
        using the class balanced categorical crossentropy loss from
        "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based
        on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269.
        doi: 10.1109/CVPR.2019.00949."
        
        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.
        
        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.
        
        """
        hps = self.cbl_hps(hps)
        return SimpleInceptionV3FineTuningModel.build(self, hps)
    
    def cbl_hps(self, hps):
        """
        This function defines the hyperparameter operations for fine tuning the clb.

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.

        Returns
        -------
        hps : HyperParameters
            The hyperparameters object for the model.

        """
        hp_beta = hps.Float("beta", min_value = self.hp_beta_min_value, max_value = self.hp_beta_max_value, sampling = self.hp_beta_sampling, default = self.hp_beta_default)
        self.loss = self.cbl_wrapper(self.labels, hp_beta)
        return hps

class SimpleInceptionV3FineTuningCuiSamplingCBLModel(SimpleInceptionV3FineTuningCuiSamplingModel, SimpleInceptionV3FineTuningCBLModel):
    """
    This class defines a simple inception v3 cnn model with weights and sampling sheme from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    using the class balanced categorical crossentropy loss from
    "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based
    on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision
    and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269.
    doi: 10.1109/CVPR.2019.00949."
    """
    def __init__(self, image_resolution, number_of_classes, labels, hp_stage_split_min_value, hp_stage_split_max_value, hp_stage_split_step, hp_stage_split_default = 2, hp_learning_rate_factor_min_value = 10, hp_learning_rate_factor_max_value = 1000, hp_learning_rate_factor_sampling = "LOG", hp_learning_rate_factor_default = 10, hp_beta_min_value = 0.0001, hp_beta_max_value = 0.9999, hp_beta_sampling = 'LOG', hp_beta_default = 0.5, hp_learning_rate_cnn_min_value = 1e-6, hp_learning_rate_cnn_max_value = 1e-2, hp_learning_rate_cnn_sampling = 'LOG', hp_learning_rate_cnn_default = 1e-4, hp_momentum_cnn_min_value = 0.5, hp_momentum_cnn_max_value = 0.9, hp_momentum_cnn_step = 0.1, hp_momentum_cnn_default = 0.9, hp_learning_rate_dense_min_value = 1e-6, hp_learning_rate_dense_max_value = 1e-1, hp_learning_rate_dense_sampling = 'LOG', hp_learning_rate_dense_default = 1e-2, hp_momentum_dense_min_value = 0.5, hp_momentum_dense_max_value = 0.9, hp_momentum_dense_step = 0.1, hp_momentum_dense_default = 0.9, metrics = ['accuracy'], cnn_trainable = False, main_generator = None):
        """
        This function initializes the SimpleInceptionV3FineTuningCuiSamplingCBLModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        labels : numpy-array
            The given labels.
        hp_stage_split_min_value : float
            The minimum value of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_max_value : float
            The maximum value of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_step : float
            The step size of the hyperparameter stage_split.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_stage_split_default : float, optional
            The default value of the hyperparameter stage_split. The default is 2.
            The stage_split defines the split between first- and second trainings stage.
            The first stage takes num_epochs / stage_split epochs and second stage takes num_epochs - first stage epochs.
        hp_learning_rate_factor_min_value : float, optional
            The minimum value of the hyperparameter learning_rate_factor. The default is 10.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_max_value : float, optional
            The maximum value of the hyperparameter learning_rate_factor. The default is 1000.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_sampling : string, optional
            The sampling type of the hyperparameter learning_rate_factor. The default is "LOG".
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_learning_rate_factor_default : float, optional
            The default value of the hyperparameter learning_rate_factor. The default is 10.
            The hyperparameter learning_rate_factor denotes the change of the learning rate in the second stage in comparison to stage one.
        hp_beta_min_value : float, optional
            The minimum value for the hyperparameter beta. The default is 0.0001.
        hp_beta_max_value : float, optional
            The maximum value for the hyperparameter beta. The default is 0.9999.
        hp_beta_sampling : string, optional
            The sampling type for the hyperparameter beta. The default is 'LOG'.
        hp_beta_default : float, optional
            The default value for the hyperparameter beta. The default is 0.5.
        hp_learning_rate_cnn_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the cnn. The default is 1e-6.
        hp_learning_rate_cnn_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the cnn. The default is 1e-2.
        hp_learning_rate_cnn_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the cnn. The default is 'LOG'.
        hp_learning_rate_cnn_default : float, optional
            The default value for the hyperparameter learning rate of the cnn. The default is 1e-4.
        hp_momentum_cnn_min_value : float, optional
            The minimum value for the hyperparameter momentum of the cnn. The default is 0.5.
        hp_momentum_cnn_max_value : float, optional
            The momentum value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_momentum_cnn_step : float, optional
            The step size for the hyperparameter momentum of the cnn. The default is 0.1.
        hp_momentum_cnn_default : float, optional
            The default value for the hyperparameter momentum of the cnn. The default is 0.9.
        hp_learning_rate_dense_min_value : float, optional
            The minimum value for the hyperparameter learning rate of the dense layer. The default is 1e-6.
        hp_learning_rate_dense_max_value : float, optional
            The maximum value for the hyperparameter learning rate of the dense layer. The default is 1e-1.
        hp_learning_rate_dense_sampling : string, optional
            The sampling type for the hyperparameter learning rate of the dense layer. The default is 'LOG'.
        hp_learning_rate_dense_default : float, optional
            The default value for the hyperparameter learning rate of the dense layer. The default is 1e-2.
        hp_momentum_dense_min_value : float, optional
            The minimum value for the hyperparameter momentum of the dense layer. The default is 0.5.
        hp_momentum_dense_max_value : float, optional
            The momentum value for the hyperparameter momentum of the dense layer. The default is 0.9.
        hp_momentum_dense_step : float, optional
            The step size for the hyperparameter momentum of the dense layer. The default is 0.1.
        hp_momentum_dense_default : float, optional
            The default value for the hyperparameter momentum of the dense layer. The default is 0.9.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
        cnn_trainable : boolean, optional
            This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.
        main_generator : TrainingValidationSetCuiGenerator, optional
            The main_generator is the object, which is responsible for the generation of training- and validation generator. The default is False.

        Returns
        -------
        None.

        """
        SimpleInceptionV3FineTuningCuiSamplingModel.__init__(self, image_resolution, number_of_classes, hp_stage_split_min_value, hp_stage_split_max_value, hp_stage_split_step, hp_stage_split_default, hp_learning_rate_factor_min_value, hp_learning_rate_factor_max_value, hp_learning_rate_factor_sampling, hp_learning_rate_factor_default, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, None, metrics, cnn_trainable, main_generator)
        SimpleInceptionV3FineTuningCBLModel.__init__(self, image_resolution, number_of_classes, labels, hp_beta_min_value, hp_beta_max_value, hp_beta_sampling, hp_beta_default, hp_learning_rate_cnn_min_value, hp_learning_rate_cnn_max_value, hp_learning_rate_cnn_sampling, hp_learning_rate_cnn_default, hp_momentum_cnn_min_value, hp_momentum_cnn_max_value, hp_momentum_cnn_step, hp_momentum_cnn_default, hp_learning_rate_dense_min_value, hp_learning_rate_dense_max_value, hp_learning_rate_dense_sampling, hp_learning_rate_dense_default, hp_momentum_dense_min_value, hp_momentum_dense_max_value, hp_momentum_dense_step, hp_momentum_dense_default, metrics)
    
    def build(self, hps):
        """
        This function creates a simple inception v3 cnn model with weights and sampling sheme from
        "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
        Fine-Grained Categorization and Domain-Specific Transfer Learning“,
        in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
        Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
        using the class balanced categorical crossentropy loss from
        "Y. Cui, M. Jia, T.-Y. Lin, Y. Song, und S. Belongie, „Class-Balanced Loss Based
        on Effective Number of Samples“, in 2019 IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), Long Beach, CA, USA, Juni 2019, S. 9260–9269.
        doi: 10.1109/CVPR.2019.00949."

        Parameters
        ----------
        hps : HyperParameters
            The hyperparameters object for the model.
        
        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.

        """
        hps = SimpleInceptionV3FineTuningCuiSamplingModel.cui_sampling_hps(self, hps)
        return SimpleInceptionV3FineTuningCBLModel.build(self, hps)
