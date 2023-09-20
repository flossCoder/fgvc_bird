#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# auxiliary_ml_functions.py
# Copyright (C) 2022 flossCoder
# 
# auxiliary_ml_functions is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# auxiliary_ml_functions is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Tue Feb 15 09:37:37 2022

@author: flossCoder
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from contextlib import redirect_stdout
import os

def preprocess_labels(labels, number_of_classes = None):
    """
    This function converts the given labels into a one-hot encoding.

    Parameters
    ----------
    labels : numpy-array
        The given labels.
    number_of_classes : int, optional
        The number of classes, if None, take the number of unique labels as number of classes. The default is None.

    Returns
    -------
    preprocessed_labels : numpy-array
        The one-hot encoded labels.

    """
    unique_labels = np.unique(labels)
    min_unique_labels = np.min(unique_labels)
    if type(number_of_classes) == type(None):
        number_of_classes = len(unique_labels)
    preprocessed_labels = tf.keras.utils.to_categorical(labels-min_unique_labels, number_of_classes)
    return preprocessed_labels

def preprocess_samples(keras_net_application, samples):
    """
    This function converts the given samples into the keras model format.

    Parameters
    ----------
    keras_net_application : module handle
        This module handle addresses a tf.keras.applications.MODULE module, such that keras_net_application.preprocess_input is valid.
    samples : numpy-array or list of numpy arrays or PIL objects
        The raw input samples.

    Returns
    -------
    preprocessed_samples : numpy-array
        The preprocessed input samples.

    """
    if type(samples[0]) != type(np.array(None)):
        samples = [np.array(i) for i in samples]
    if type(samples) != type(np.array(None)):
        samples = np.array(samples)
    preprocessed_samples = keras_net_application.preprocess_input(samples)
    return preprocessed_samples

def save_compiled_model(model, output_wd, output_name):
    """
    This function saves the model summary as txt, and the model as json.

    Parameters
    ----------
    model : keras model
        The keras / tensorflow model.
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.

    Returns
    -------
    None.

    """
    with open(os.path.join(output_wd, "%s_model_summary.txt"%output_name), "w") as f:
        with redirect_stdout(f):
            model.summary()
    with open(os.path.join(output_wd, "%s_model.json"%output_name), "w") as f:
        f.write(model.to_json())

def load_compiled_model(output_wd, output_name):
    """
    This function loads the model json file.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.

    Returns
    -------
    model : keras model
        The keras / tensorflow model.

    """
    with open(os.path.join(output_wd, "%s_model.json"%output_name), 'r') as f:
        model = f.read()
    
    custom_objects = {}
    if "KerasLayer" in model:
        custom_objects["KerasLayer"] = hub.KerasLayer
    
    if custom_objects != {}:
        model = tf.keras.models.model_from_json(model, custom_objects = custom_objects)
    else:
        model = tf.keras.models.model_from_json(model)
    return model

def join_history_objects(history1, history2):
    """
    This function joins the second history object to the first one.

    Parameters
    ----------
    history1 : keras callbacks history
        The first history object.
    history2 : keras callbacks history
        The second history object.

    Returns
    -------
    history1 : keras callbacks history
        The first history object after joining the second history object.

    """
    epochs1 = len(history1.epoch)
    epochs2 = len(history2.epoch)
    history1.epoch = history1.epoch + [i + max(history1.epoch) + 1 for i in history2.epoch]
    for key in history1.history.keys():
        if key in history2.history.keys():
            history1.history[key] = history1.history[key] + history2.history[key]
        else:
            history1.history[key] = history1.history[key] + [0 for i in range(epochs2)]
    for key in history2.history.keys():
        if key not in history2.history.keys():
            history1.history[key] = [0 for i in range(epochs1)] + history2.history[key]
    return history1

def prepare_kwargs_rebalance_fit(**kwargs):
    """
    This function preparates the keyword argument split for the two phase training.

    Parameters
    ----------
    **kwargs : dict
        The keyword arguments.

    Raises
    ------
    Exception
        The exception is raised in case the "epochs" denoting the number of epochs are missing.

    Returns
    -------
    kwargs_first_phase : dict
        The keyword arguments for the first training phase.
    kwargs_second_phase : dict
        The keyword arguments for the second training phase.
    learning_rate_factor : float
        The factor for decreasing the learning rate in the second phase (the default is 10).

    """
    stage_split = 2
    if "stage_split" in kwargs.keys():
        stage_split = kwargs["stage_split"]
    if "epochs" in kwargs.keys():
        epochs_first_phase = int(np.ceil(kwargs["epochs"] / stage_split))
        epochs_second_phase = kwargs["epochs"] - epochs_first_phase
    else:
        raise Exception("kwargs epochs missing")
    
    kwargs_first_phase = {"epochs": epochs_first_phase}
    kwargs_second_phase = {"epochs": epochs_second_phase}
    learning_rate_factor = 10.
    for key, value in kwargs.items():
        if key != "epochs" and key != "learning_rate_factor" and key != "stage_split":
            kwargs_first_phase[key] = value
            kwargs_second_phase[key] = value
        elif key == "learning_rate_factor":
            learning_rate_factor = value
    
    return kwargs_first_phase, kwargs_second_phase, learning_rate_factor

def rename_files(wd, old_names, new_names):
    """
    This function renames the given files.

    Parameters
    ----------
    wd : string
        The working directory.
    old_names : list of strings
        The list of old file names.
    new_names : list of strings
        The list of new file names.

    Raises
    ------
    Exception
        The exception is raised in case the old_names list and new_names list differ in length.

    Returns
    -------
    None.

    """
    if len(old_names) != len(new_names):
        raise Exception("the length of old_names %s and new_names %s differ"%(str(old_names), str(new_names)))
    
    for i in range(len(old_names)):
        os.rename(os.path.join(wd, old_names[i]), os.path.join(wd, new_names[i]))

def resolve_cui_dir():
    """
    This function resolves the cui weights for the inception v3 net
    eather from the global variable HUB_DIR, or the link to hub.

    Returns
    -------
    tensorflow_hub.keras_layer.KerasLayer
        The weights of the cui inception v3.

    """
    if "HUB_DIR" in locals():
        return HUB_DIR
    else:
        return "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"
