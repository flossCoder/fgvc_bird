#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# data_generator.py
# Copyright (C) 2022 flossCoder
# 
# data_generator is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# data_generator is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Fri Mar 11 06:56:28 2022

@author: flossCoder
"""

import tensorflow as tf
import numpy as np
from image_processing import load_images_image_temp
import auxiliary_ml_functions as aux
from sklearn.model_selection import train_test_split
from copy import deepcopy

class ClassificationSequence(tf.keras.utils.Sequence):
    """
    This class implements a data generator for training a classifier in keras implementing keras.utils.Sequence
    """
    def __init__(self, temp_filename_dir, images_line_index, images_class_labels, batch_size, keras_net_application, number_of_classes = None, resolution = None, shuffel_training_samples = False, seed = None, test_mode = False, **kwargs):
        """
        This function initializes the ClassificationSequence data generator.

        Parameters
        ----------number_of_classes = len(np.unique(np.concatenate((labels_training, labels_test, labels_validation))))
        temp_filename_dir : string
            The path and filename to the temporary directory.
        images_line_index : numpy array
            The line index of the images in the temp file.
        images_class_labels : numpy array
            The labels of the image set.
        batch_size : int
            The batch size denotes how many training samples shall be used during an epoch.
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
        test_mode : boolean, optional
            Indicates, whether the data generator is used for model prediction (True) or not (False). The default is False
        **kwargs : dict
            The kwargs can be used to handle data augmentation with the statement "augmentation_list":
            The augmentation list contains the augmentation operations.
            Each entry is eather a function handle or a tuple containing a
            function handle and a dictionary with kwargs passed to the augmentation function.

        Returns
        -------
        None.

        """
        self.temp_filename_dir = temp_filename_dir
        self.images_line_index = images_line_index
        self.keras_net_application = keras_net_application
        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.resolution = resolution
        self.shuffel_training_samples = shuffel_training_samples
        self.images_class_labels = aux.preprocess_labels(np.array(images_class_labels), number_of_classes)
        if "augmentation_list" in kwargs.keys():
            self.augmentation_list = kwargs["augmentation_list"]
        else:
            self.augmentation_list = None
        self.non_random_augmentation = []
        self.augmentation_task = None
        self.random_augmentation = []
        self.number_of_batches = None
        self.calculate_number_of_batches()
        self.sample_index = np.arange(len(self.images_line_index))
        if seed != None:
            np.random.seed(seed)
        self.test_mode = test_mode
        self.shuffle_dataset()
    
    def __len__(self):
        """
        This function returns the number of batches per epoch.

        Returns
        -------
        number_of_batches : int
            The number of batches per epoch.

        """
        return self.number_of_batches
    
    def __getitem__(self, index):
        """
        This function returns the training batch.

        Parameters
        ----------
        index : integer
            The index describes which batch shall be returned.

        Returns
        -------
        current_images : numpy array
            This are the normalized training images of the current batch index.
        images_class_labels : numpy array
            This are the class labels of the images from the current batch.

        """
        if self.test_mode:
            current_index = self.sample_index[np.arange(index*self.batch_size, np.min(((index+1)*self.batch_size, len(self.sample_index))))]
        else:
            current_index = self.sample_index[np.arange(index*self.batch_size, (index+1)*self.batch_size) % len(self.sample_index)]
        current_images = load_images_image_temp(self.temp_filename_dir, self.images_line_index[current_index], self.resolution)
        current_images = aux.preprocess_samples(self.keras_net_application, np.array(current_images))
        if type(self.augmentation_list) != type(None):
            current_images = self.apply_image_augmentation(current_index, current_images)
        return current_images, self.images_class_labels[current_index]
    
    def shuffle_dataset(self):
        """
        This function shuffles the dataset, if self.shuffle_training_samples is true.

        Returns
        -------
        None.

        """
        if self.shuffel_training_samples:
            self.sample_index = np.arange(len(self.images_line_index))
            np.random.shuffle(self.sample_index)
    
    def on_epoch_end(self):
        """
        This function shuffles the dataset after each epoch, if self.shuffle_training_samples is true.

        Returns
        -------
        None.

        """
        self.shuffle_dataset()
    
    def calculate_number_of_batches(self):
        """
        This function calculates the number of batches for one epoch.

        Returns
        -------
        None.

        """
        if type(self.augmentation_list) != type(None) and any(["random" not in i.__name__ if type(i) != type(tuple()) else i[0].__name__ for i in self.augmentation_list]):
            self.non_random_augmentation = [i for i in self.augmentation_list if type(i) == type(tuple()) and "random" not in i[0].__name__ or "random" not in i.__name__]
            self.random_augmentation = [i for i in self.augmentation_list if type(i) == type(tuple()) and "random" not in i[0].__name__ or "random" in i.__name__]
            self.number_of_batches = (len(self.non_random_augmentation) + 1) * int(np.ceil(len(self.images_line_index) / self.batch_size))
            self.augmentation_task = np.ones((len(self.images_class_labels), (len(self.non_random_augmentation) + 1)), dtype = bool)
        else:
            if type(self.augmentation_list) != type(None):
                self.random_augmentation = self.augmentation_list
            self.number_of_batches = int(np.ceil(len(self.images_line_index) / self.batch_size))
    
    def apply_image_augmentation(self, index, images):
        """
        This function prepares the augmentation list for the current step.

        Parameters
        ----------
        index : numpy array
            The index describes which batch shall be returned.
        images : numpy array
            This are the normalized training images of the current batch index.

        Returns
        -------
        images : numpy array
            This are the normalized training images of the current batch index after applying image augmentation.

        """
        if type(self.augmentation_task) != type(None):
            at = []
            for i in range(len(index)):
                unused_index = np.where(self.augmentation_task[index[i]] == True)[0]
                if len(unused_index) != 0:
                    at.append(np.random.choice(unused_index))
            at = np.array(at)
            aux = np.where(at == 0)[0]
            self.augmentation_task[index[aux], 0] = False
            for i in range(1, np.shape(self.augmentation_task)[1]):
                aux = np.where(at == i)[0]
                if len(aux) != 0:
                    if type(self.non_random_augmentation[i - 1]) != type(tuple()):
                        images[aux] = tf.map_fn(self.non_random_augmentation[i - 1], images[aux])
                    else:
                        func = self.non_random_augmentation[i - 1][0]
                        params = self.non_random_augmentation[i - 1][1]
                        def aux_augmentation(x):
                            return func(x, params)
                        images[aux] = tf.map_fn(aux_augmentation, images[aux])
                        
                    self.augmentation_task[index[aux], i] = False
        
        if type(self.random_augmentation) != type(None):
            for i in range(len(self.random_augmentation)):
                if type(self.random_augmentation[i]) != type(tuple()):
                    images = tf.map_fn(self.random_augmentation[i], images)
                else:
                    func = self.random_augmentation[i][0]
                    params = self.random_augmentation[i][1]
                    def aux_augmentation(x):
                        return func(x, params)
                    images = tf.map_fn(aux_augmentation, images)
        
        return images

class TrainingValidationSetCuiGenerator:
    """
    This class defines a generator for training- and validation set similar to Cui et al. (except upsampling):
        Y. Cui, Y. Song, C. Sun, A. Howard, und S. Belongie, „Large Scale Fine-Grained
        Categorization and Domain-Specific Transfer Learning“, in 2018 IEEE/CVF
        Conference on Computer Vision and Pattern Recognition, Salt Lake City,
        UT, USA, Juni 2018, S. 4109–4118. doi: 10.1109/CVPR.2018.00432.
    """
    def __init__(self, temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, number_of_classes = None, validation_split = 0.1, resolution = None, shuffel_training_samples = False, seed = None, split_seed = None, **kwargs):
        """
        This function initializes the TrainingValidationSetCuiGenerator.

        Parameters
        ----------
        temp_filename_dir : string
            The path and filename to the temporary directory.
        images_line_index : numpy array
            The line index of the images in the temp file.
        images_class_labels : numpy array
            The labels of the image set.
        batch_size : int
            The batch size denotes how many training samples shall be used during an epoch.
        keras_net_application : module handle
            This module handle addresses a tf.keras.applications.MODULE module, such that keras_net_application.preprocess_input is valid.
        number_of_classes : int, optional
            The number of classes, if None, take the number of unique labels as number of classes. The default is None.
        validation_split : float, optional
            The validation split implies the ratio between the size of the trainings set and the validation set. The default is 0.1.
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
        **kwargs : dict
            The kwargs can be used to handle data augmentation with the statement "augmentation_list":
            The augmentation list contains the augmentation operations.
            Each entry is eather a function handle or a tuple containing a
            function handle and a dictionary with kwargs passed to the augmentation function.

        Returns
        -------
        None.

        """
        self.temp_filename_dir = temp_filename_dir
        self.images_line_index = images_line_index
        self.images_class_labels = images_class_labels
        self.keras_net_application = keras_net_application
        self.number_of_classes = number_of_classes
        self.validation_split = validation_split
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.split_seed = split_seed
        self.shuffel_training_samples = shuffel_training_samples
        self.seed = seed
        if seed != None:
            np.random.seed(seed)
        self.split_seed = split_seed
        self.images_line_index_training = None
        self.images_line_index_validation = None
        self.images_line_index_training_backup = None
        self.images_line_index_validation_backup = None
        self.images_class_labels_training = None
        self.images_class_labels_validation = None
        self.training_generator = None
        self.validation_generator = None
        self.images_line_index_backup = deepcopy(images_line_index)
        self.images_class_labels_backup = deepcopy(images_class_labels)
        self.train_kwargs = deepcopy(kwargs)
        if "train_augmentation_list" in kwargs.keys():
            del self.train_kwargs["train_augmentation_list"]
            self.train_kwargs["augmentation_list"] = kwargs["train_augmentation_list"]
        self.val_kwargs = deepcopy(kwargs)
        if "val_augmentation_list" in kwargs.keys():
            del self.val_kwargs["val_augmentation_list"]
            self.val_kwargs["augmentation_list"] = kwargs["val_augmentation_list"]
        self.keep_validation_set_constant = True
        if "keep_validation_set_constant" in kwargs.keys():
            self.keep_validation_set_constant = kwargs["keep_validation_set_constant"]
    
    def generate_train_val_split(self):
        """
        This function generates the split between training- and validation set.

        Returns
        -------
        None.

        """
        [self.images_line_index_training, self.images_line_index_validation, self.images_class_labels_training, self.images_class_labels_validation] = train_test_split(self.images_line_index, self.images_class_labels, train_size = (1 - self.validation_split), random_state = self.split_seed)
        if self.keep_validation_set_constant:
            self.images_line_index_training_backup = deepcopy(self.images_line_index_training)
            self.images_class_labels_training_backup = deepcopy(self.images_class_labels_training)
    
    def get_train_val_objects(self):
        """
        This function initializes the generator objects for training and validation.

        Returns
        -------
        ClassificationSequenceCui
            The training generator generates training batches for the learning process.
        ClassificationSequenceCui
            The validation generator generates validation batches for the learning process.

        """
        self.training_generator = ClassificationSequenceCui(self, self.temp_filename_dir, self.images_line_index_training, self.images_class_labels_training, self.batch_size, self.num_epochs, self.keras_net_application, self.number_of_classes, self.resolution, self.shuffel_training_samples, self.seed, False, **self.train_kwargs)
        self.validation_generator = ClassificationSequenceCui(self, self.temp_filename_dir, self.images_line_index_validation, self.images_class_labels_validation, self.batch_size, self.num_epochs, self.keras_net_application, self.number_of_classes, self.resolution, self.shuffel_training_samples, self.seed, True, **self.val_kwargs)
        return self.training_generator, self.validation_generator
    
    def set_trigger_equilibration(self, trigger_equilibration):
        """
        This function passes the trigger euqilibration flag to the generator classes.

        Parameters
        ----------
        trigger_equilibration : boolean
            The trigger equilibration flag states, whether this class is allowed to trigger the distribution equilibration.

        Returns
        -------
        None.

        """
        self.training_generator.set_trigger_equilibration(trigger_equilibration)
        self.validation_generator.set_trigger_equilibration(trigger_equilibration)
    
    def equilibrate_distribution(self):
        """
        This function equilibrates the distribution of the images and propagates the results to the training- and validation generator.

        Returns
        -------
        None.

        """
        if not self.keep_validation_set_constant:
            [bins, counts] = np.unique(self.images_class_labels, return_counts = True)
            min_counts = np.min(counts)
            new_index = []
            for i in range(len(bins)):
                [new_index.append(j) for j in np.random.choice(np.where(self.images_class_labels == bins[i])[0], size = min_counts, replace = False)]
            
            self.images_line_index = self.images_line_index[new_index]
            self.images_class_labels = self.images_class_labels[new_index]
            self.generate_train_val_split()
            self.training_generator.set_images_labels(self.images_line_index_training, self.images_class_labels_training)
            self.validation_generator.set_images_labels(self.images_line_index_validation, self.images_class_labels_validation)
        else:
            [bins, counts] = np.unique(self.images_class_labels_training, return_counts = True)
            min_counts = np.min(counts)
            new_index = []
            for i in range(len(bins)):
                [new_index.append(j) for j in np.random.choice(np.where(self.images_class_labels_training == bins[i])[0], size = min_counts, replace = False)]
            
            self.images_line_index_training = self.images_line_index_training[new_index]
            self.images_class_labels_training = self.images_class_labels_training[new_index]
            self.training_generator.set_images_labels(self.images_line_index_training, self.images_class_labels_training)
    
    def reset_distribution(self):
        """
        This function resets the distribution of the images and propagates the results to the training- and validation generator.

        Returns
        -------
        None.

        """
        if not self.keep_validation_set_constant:
            self.images_line_index = deepcopy(self.images_line_index_backup)
            self.images_class_labels = deepcopy(self.images_class_labels_backup)
            self.generate_train_val_split()
            self.training_generator.set_images_labels(self.images_line_index_training, self.images_class_labels_training)
            self.validation_generator.set_images_labels(self.images_line_index_validation, self.images_class_labels_validation)
        else:
            self.images_line_index_training = deepcopy(self.images_line_index_training_backup)
            self.images_class_labels_training = deepcopy(self.images_class_labels_training_backup)
            self.training_generator.set_images_labels(self.images_line_index_training, self.images_class_labels_training)

class ClassificationSequenceCui(ClassificationSequence):
    """
    This class implements a data generator for training a classifier in keras implementing ClassificationSequence.
    """
    def __init__(self, training_validation_set_generator, temp_filename_dir, images_line_index, images_class_labels, batch_size, num_epochs, keras_net_application, number_of_classes = None, resolution = None, shuffel_training_samples = False, seed = None, test_mode = False, **kwargs):
        """
        This function initializes the ClassificationSequenceCui.

        Parameters
        ----------
        temp_filename_dir : string
            The path and filename to the temporary directory.
        images_line_index : numpy array
            The line index of the images in the temp file.
        images_class_labels : numpy array
            The labels of the image set.
        batch_size : int
            The batch size denotes how many training samples shall be used during an epoch.
        num_epochs : int
            The number of epochs for training.
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
        test_mode : boolean, optional
            Indicates, whether the data generator is used for model prediction (True) or not (False). The default is False
        **kwargs : dict
            The kwargs can be used to handle data augmentation with the statement "augmentation_list":
            The augmentation list contains the augmentation operations.
            Each entry is eather a function handle or a tuple containing a
            function handle and a dictionary with kwargs passed to the augmentation function.
        Returns
        -------
        None.

        """
        self.num_epochs = num_epochs
        self.training_validation_set_generator = training_validation_set_generator
        self.current_epoch = 1
        self.trigger_equilibration = False
        super().__init__(temp_filename_dir, images_line_index, images_class_labels, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed, test_mode, **kwargs)
    
    def set_trigger_equilibration(self, trigger_equilibration):
        """
        This function sets the trigger euqilibration flag.

        Parameters
        ----------
        trigger_equilibration : boolean
            The trigger equilibration flag states, whether this class is allowed to trigger the distribution equilibration.

        Returns
        -------
        None.

        """
        self.trigger_equilibration = trigger_equilibration
    
    def set_images_labels(self, images_line_index, images_class_labels):
        """
        This function allows to manipulate the images and labels set.

        Parameters
        ----------
        images_line_index : numpy array
            The line index of the images in the temp file.
        images_class_labels : numpy array
            The labels of the image set.

        Returns
        -------
        None.

        """
        self.images_line_index = images_line_index
        self.images_class_labels = aux.preprocess_labels(np.array(images_class_labels), self.number_of_classes)
        self.sample_index = np.arange(len(self.images_line_index))
        super().calculate_number_of_batches()
        super().shuffle_dataset()
    
    def on_epoch_end(self):
        """
        This function triggers the equalibration and resetting of the training- and
        validation set distribution, if the object is responsible for training.
        The equalibration is triggered in case 50 % of the desired epochs have been trained.
        The resetting is triggered at the end of the training.
        Furthermore the shuffling is applied.
        

        Returns
        -------
        None.

        """
        if self.trigger_equilibration and not self.test_mode and self.current_epoch == np.ceil(self.num_epochs / 2):
            pass#self.training_validation_set_generator.equilibrate_distribution()
        else:
            super().shuffle_dataset()
        if self.current_epoch == self.num_epochs:
            self.current_epoch = 1
            if not self.test_mode:
                self.training_validation_set_generator.reset_distribution()
        else:
            self.current_epoch += 1
