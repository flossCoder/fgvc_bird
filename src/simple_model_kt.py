#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simple_model_kt.py
# Copyright (C) 2022 flossCoder
# 
# simple_model_kt is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# simple_model_kt is distributed in the hope that it will be useful, but
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

from hypermodel import AbstractHyperModel, AbstractRebalanceCuiHyperModel
import auxiliary_ml_functions as aux
import class_balanced_loss as cbl

class SimpleInceptionV3Model(AbstractHyperModel):
    """
    This class defines a simple inception v3 cnn model with weights from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    """
    def __init__(self, image_resolution, number_of_classes, hp_learning_rate_min_value = 1e-4, hp_learning_rate_max_value = 1e-2, hp_learning_rate_sampling = 'LOG', hp_learning_rate_default = 0.009, hp_momentum_min_value = 0.5, hp_momentum_max_value = 0.9, hp_momentum_step = 0.1, hp_momentum_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy'], cnn_trainable = False):
        """
        This function initializes the SimpleInceptionV3Model.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        labels : numpy-array
            The given labels.
        hp_learning_rate_min_value : float, optional
            The minimum value for the hyperparameter learning rate. The default is 1e-4.
        hp_learning_rate_max_value : float, optional
            The maximum value for the hyperparameter learning rate. The default is 1e-2.
        hp_learning_rate_sampling : string, optional
            The sampling type for the hyperparameter learning rate. The default is 'LOG'.
        hp_learning_rate_default : float, optional
            The default value for the hyperparameter learning rate. The default is 0.009.
        hp_momentum_min_value : float, optional
            The minimum value for the hyperparameter momentum. The default is 0.5.
        hp_momentum_max_value : float, optional
            The momentum value for the hyperparameter momentum. The default is 0.9.
        hp_momentum_step : float, optional
            The step size for the hyperparameter momentum. The default is 0.1.
        hp_momentum_default : float, optional
            The default value for the hyperparameter momentum. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
        cnn_trainable : boolean, optional
            This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

        Returns
        -------
        None.

        """
        AbstractHyperModel.__init__(self, image_resolution, number_of_classes, loss, metrics, cnn_trainable)
        self.hp_learning_rate_min_value = hp_learning_rate_min_value
        self.hp_learning_rate_max_value = hp_learning_rate_max_value
        self.hp_learning_rate_sampling = hp_learning_rate_sampling
        self.hp_learning_rate_default = hp_learning_rate_default
        self.hp_momentum_min_value = hp_momentum_min_value
        self.hp_momentum_max_value = hp_momentum_max_value
        self.hp_momentum_step = hp_momentum_step
        self.hp_momentum_default = hp_momentum_default
    
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
            The best hyperparameters object for the model.
        
        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.
        
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = self.image_resolution))
        cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = self.cnn_trainable)
        model.add(cnn_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.number_of_classes, activation = "softmax"))
        hp_learning_rate = hps.Float("learning_rate", min_value = self.hp_learning_rate_min_value, max_value = self.hp_learning_rate_max_value, sampling = self.hp_learning_rate_sampling, default = self.hp_learning_rate_default)
        hp_momentum = hps.Float("momentum", min_value = self.hp_momentum_min_value, max_value = self.hp_momentum_max_value, step = self.hp_momentum_step, default = self.hp_momentum_default)
        sgd_opt = tf.keras.optimizers.SGD(learning_rate = hp_learning_rate, momentum = hp_momentum, nesterov = True)
        model.compile(optimizer = sgd_opt, loss = self.loss, metrics = self.metrics)
        return model

class SimpleInceptionV3CuiSamplingModel(AbstractRebalanceCuiHyperModel, SimpleInceptionV3Model):
    """
    This class defines a simple inception v3 cnn model with weights and sampling sheme from
    "Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale
    Fine-Grained Categorization and Domain-Specific Transfer Learning“,
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432."
    """
    def __init__(self, image_resolution, number_of_classes, hp_learning_rate_min_value = 1e-4, hp_learning_rate_max_value = 1e-2, hp_learning_rate_sampling = 'LOG', hp_learning_rate_default = 0.009, hp_momentum_min_value = 0.5, hp_momentum_max_value = 0.9, hp_momentum_step = 0.1, hp_momentum_default = 0.9, loss = "categorical_crossentropy", metrics = ['accuracy'], cnn_trainable = False, main_generator = None):
        """
        This function initializes the SimpleInceptionV3CuiSamplingModel.

        Parameters
        ----------
        image_resolution : tuple
            The input resolution of the images.
        number_of_classes : int
            The number of classes of the dataset.
        hp_learning_rate_min_value : float, optional
            The minimum value for the hyperparameter learning rate. The default is 1e-4.
        hp_learning_rate_max_value : float, optional
            The maximum value for the hyperparameter learning rate. The default is 1e-2.
        hp_learning_rate_sampling : string, optional
            The sampling type for the hyperparameter learning rate. The default is 'LOG'.
        hp_learning_rate_default : float, optional
            The default value for the hyperparameter learning rate. The default is 0.009.
        hp_momentum_min_value : float, optional
            The minimum value for the hyperparameter momentum. The default is 0.5.
        hp_momentum_max_value : float, optional
            The momentum value for the hyperparameter momentum. The default is 0.9.
        hp_momentum_step : float, optional
            The step size for the hyperparameter momentum. The default is 0.1.
        hp_momentum_default : float, optional
            The default value for the hyperparameter momentum. The default is 0.9.
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
        SimpleInceptionV3Model.__init__(self, image_resolution, number_of_classes, hp_learning_rate_min_value, hp_learning_rate_max_value, hp_learning_rate_sampling, hp_learning_rate_default, hp_momentum_min_value, hp_momentum_max_value, hp_momentum_step, hp_momentum_default, loss, metrics, cnn_trainable)

class SimpleInceptionV3ModelCBL(SimpleInceptionV3Model):
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
    def __init__(self, image_resolution, number_of_classes, labels, hp_beta_min_value = 0.0001, hp_beta_max_value = 0.9999, hp_beta_sampling = 'LOG', hp_beta_default = 0.5, hp_learning_rate_min_value = 1e-4, hp_learning_rate_max_value = 1e-2, hp_learning_rate_sampling = 'LOG', hp_learning_rate_default = 0.009, hp_momentum_min_value = 0.5, hp_momentum_max_value = 0.9, hp_momentum_step = 0.1, hp_momentum_default = 0.9, metrics = ['accuracy'], cnn_trainable = False):
        """
        This function initializes the SimpleInceptionV3ModelCBL.

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
        hp_learning_rate_min_value : float, optional
            The minimum value for the hyperparameter learning rate. The default is 1e-4.
        hp_learning_rate_max_value : float, optional
            The maximum value for the hyperparameter learning rate. The default is 1e-2.
        hp_learning_rate_sampling : string, optional
            The sampling type for the hyperparameter learning rate. The default is 'LOG'.
        hp_learning_rate_default : float, optional
            The default value for the hyperparameter learning rate. The default is 0.009.
        hp_momentum_min_value : float, optional
            The minimum value for the hyperparameter momentum. The default is 0.5.
        hp_momentum_max_value : float, optional
            The momentum value for the hyperparameter momentum. The default is 0.9.
        hp_momentum_step : float, optional
            The step size for the hyperparameter momentum. The default is 0.1.
        hp_momentum_default : float, optional
            The default value for the hyperparameter momentum. The default is 0.9.
        loss : string or function handle
            The loss is eather a string of a build in loss of keras or a function handle.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
        cnn_trainable : boolean, optional
            This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.

        Returns
        -------
        None.

        """
        SimpleInceptionV3Model.__init__(self, image_resolution, number_of_classes, hp_learning_rate_min_value, hp_learning_rate_max_value, hp_learning_rate_sampling, hp_learning_rate_default, hp_momentum_min_value, hp_momentum_max_value, hp_momentum_step, hp_momentum_default, None, metrics, cnn_trainable)
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
            The best hyperparameters object for the model.
        
        Returns
        -------
        model : keras model
            The keras / tensorflow model based on inception v3.
        
        """
        # define the cbl loss
        hp_beta = hps.Float("beta", min_value = self.hp_beta_min_value, max_value = self.hp_beta_max_value, sampling = self.hp_beta_sampling, default = self.hp_beta_default)
        self.loss = self.cbl_wrapper(self.labels, hp_beta)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape = self.image_resolution))
        cnn_model = hub.KerasLayer("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", trainable = self.cnn_trainable)
        model.add(cnn_model)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.number_of_classes, activation = "softmax"))
        hp_learning_rate = hps.Float("learning_rate", min_value = self.hp_learning_rate_min_value, max_value = self.hp_learning_rate_max_value, sampling = self.hp_learning_rate_sampling, default = self.hp_learning_rate_default)
        hp_momentum = hps.Float("momentum", min_value = self.hp_momentum_min_value, max_value = self.hp_momentum_max_value, step = self.hp_momentum_step, default = self.hp_momentum_default)
        sgd_opt = tf.keras.optimizers.SGD(learning_rate = hp_learning_rate, momentum = hp_momentum, nesterov = True)
        model.compile(optimizer = sgd_opt, loss = self.loss, metrics = self.metrics)
        return model

class SimpleInceptionV3CuiSamplingCBLModel(AbstractRebalanceCuiHyperModel, SimpleInceptionV3ModelCBL):
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
    def __init__(self, image_resolution, number_of_classes, labels, hp_beta_min_value = 0.0001, hp_beta_max_value = 0.9999, hp_beta_sampling = 'LOG', hp_beta_default = 0.5, hp_learning_rate_min_value = 1e-4, hp_learning_rate_max_value = 1e-2, hp_learning_rate_sampling = 'LOG', hp_learning_rate_default = 0.009, hp_momentum_min_value = 0.5, hp_momentum_max_value = 0.9, hp_momentum_step = 0.1, hp_momentum_default = 0.9, metrics = ['accuracy'], cnn_trainable = False, main_generator = None):
        """
        This function initializes the SimpleInceptionV3ModelCBL.

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
        hp_learning_rate_min_value : float, optional
            The minimum value for the hyperparameter learning rate. The default is 1e-4.
        hp_learning_rate_max_value : float, optional
            The maximum value for the hyperparameter learning rate. The default is 1e-2.
        hp_learning_rate_sampling : string, optional
            The sampling type for the hyperparameter learning rate. The default is 'LOG'.
        hp_learning_rate_default : float, optional
            The default value for the hyperparameter learning rate. The default is 0.009.
        hp_momentum_min_value : float, optional
            The minimum value for the hyperparameter momentum. The default is 0.5.
        hp_momentum_max_value : float, optional
            The momentum value for the hyperparameter momentum. The default is 0.9.
        hp_momentum_step : float, optional
            The step size for the hyperparameter momentum. The default is 0.1.
        hp_momentum_default : float, optional
            The default value for the hyperparameter momentum. The default is 0.9.
        metrics : string or function handle
            The metrics used for evaluation is a list containing eather strings of build in keras functions or function handles.
        cnn_trainable : boolean, optional
            This flag indicates, whether the cnn weights shall be fine tuned (True) or not (False). The default is False.
        main_generator : TrainingValidationSetCuiGenerator
            The main_generator is the object, which is responsible for the generation of training- and validation generator.

        Returns
        -------
        None.

        """
        AbstractRebalanceCuiHyperModel.__init__(self, image_resolution, number_of_classes, None, metrics, cnn_trainable, main_generator)
        SimpleInceptionV3ModelCBL.__init__(self, image_resolution, number_of_classes, labels, hp_beta_min_value, hp_beta_max_value, hp_beta_sampling, hp_beta_default, hp_learning_rate_min_value, hp_learning_rate_max_value, hp_learning_rate_sampling, hp_learning_rate_default, hp_momentum_min_value, hp_momentum_max_value, hp_momentum_step, hp_momentum_default, metrics, cnn_trainable)
