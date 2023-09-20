#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model_tester.py
# Copyright (C) 2022 flossCoder
# 
# model_tester is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# model_tester is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Mar  7 11:17:16 2022

@author: flossCoder
"""

import os
import tensorflow as tf
import numpy as np
from auxiliary_ml_functions import load_compiled_model, preprocess_labels
from data_generator import ClassificationSequence

def evaluate_model(output_wd, output_name, temp_filename_dir, test_image_line_index, labels_test, loss, k, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed = None, epoch = None, model = None):
    """
    This function loads a model for a certain epoch step and evaluates it.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    test_image_line_index : numpy array
        The index of the temp file for the test set.
    labels_test : numpy array
        The labels of the test set.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    k : int
        The k of top k accuracy.
    batch_size : int
        The number of samples per training batch.
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
    epoch : int, optional
        The epoch for loading the model. If None, the final model is loaded. The default is None.
    model : keras model, optional
        The keras / tensorflow model. If None is given, the model will be loaded from json. The default is None.

    Returns
    -------
    list
        [epoch, score, acc, tkc_acc]
        epoch : The current epoch.
        score : The loss on the test set.
        acc : The accuracy on the test set.
        tkc_acc : The top-k accuracy on the test set.

    """
    labels_test_cat = preprocess_labels(labels_test, number_of_classes)
    
    if type(model) == type(None):
        model = load_compiled_model(output_wd, output_name)
    model.load_weights(os.path.join(output_wd, "%s_%s.h5"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")))
    
    prediction = model.predict(x = ClassificationSequence(temp_filename_dir, test_image_line_index, labels_test, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed, True), batch_size = batch_size)
    prediction.astype(labels_test_cat.dtype)
    
    if type(loss) == type(""):
        loss = tf.keras.losses.get(loss)
    loss_values = loss(labels_test_cat, prediction)
    score = np.mean(loss_values)
    
    m_c_acc = tf.keras.metrics.CategoricalAccuracy()
    m_c_acc.update_state(labels_test_cat, prediction)
    acc = m_c_acc.result().numpy()
    
    m_tkc_acc = tf.keras.metrics.TopKCategoricalAccuracy(k = k)
    m_tkc_acc.update_state(labels_test_cat, prediction)
    tkc_acc = m_tkc_acc.result().numpy()
    
    np.savetxt(os.path.join(output_wd, "%s_prediction_%s.txt"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")), np.concatenate((labels_test.reshape((-1,1)), np.array(loss_values).reshape((-1,1)), prediction), axis = 1), header = "score: %f, Top-1-Acc: %f, Top-%i-Acc: %f"%(score, acc, k, tkc_acc))
    
    unique_labels, counts = np.unique(labels_test, return_counts = True)
    categorical_results = np.zeros((len(unique_labels), 5))
    for i in range(len(unique_labels)):
        aux_index = np.where(labels_test == unique_labels[i])[0]
        categorical_results[i, 0] = unique_labels[i]
        categorical_results[i, 1] = counts[i]
        categorical_results[i, 2] = np.mean(np.array(loss_values)[aux_index])
        m_c_acc.update_state(labels_test_cat[aux_index], prediction[aux_index])
        categorical_results[i, 3] = m_c_acc.result().numpy()
        m_tkc_acc.update_state(labels_test_cat[aux_index], prediction[aux_index])
        categorical_results[i, 4] = m_tkc_acc.result().numpy()
    
    np.savetxt(os.path.join(output_wd, "%s_categorical-prediction_%s.txt"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")), categorical_results, header = "score: %f, Top-1-Acc: %f, Top-%i-Acc: %f"%(score, acc, k, tkc_acc))
    
    return [("model" if type(epoch) == type(None) else epoch), score, acc, tkc_acc]

def evaluate_training(output_wd, output_name, temp_filename_dir, test_image_line_index, labels_test, loss, k, number_of_epochs, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed = None, model = None):
    """
    This function evaluates all training epochs.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    temp_filename_dir : string
        The path and filename to the temporary directory.
    test_image_line_index : numpy array
        The index of the temp file for the test set.
    labels_test : numpy array
        The labels of the test set.
    loss : string or function handle
        The loss is eather a string of a build in loss of keras or a function handle.
    k : int
        The k of top k accuracy.
    number_of_epochs : int
        The number of epochs during training.
    batch_size : int
        The number of samples per training batch.
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
    epoch : int, optional
        The epoch for loading the model. If None, the final model is loaded. The default is None.
    model : keras model, optional
        The keras / tensorflow model. If None is given, the model will be loaded from json. The default is None.

    Returns
    -------
    None.

    """
    result = []
    for epoch in range(number_of_epochs):
        result.append(evaluate_model(output_wd, output_name, temp_filename_dir, test_image_line_index, labels_test, loss, k, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed, epoch+1, model))
    result.append(evaluate_model(output_wd, output_name, temp_filename_dir, test_image_line_index, labels_test, loss, k, batch_size, keras_net_application, number_of_classes, resolution, shuffel_training_samples, seed, None, model))
    with open(os.path.join(output_wd, "%s_prediction.txt"%(output_name)), "w") as f:
        f.write("# epoch\tscore\tacc\ttkc_acc\n")
        for i in result:
            f.write("\t".join([str(j) for j in i]) + "\n")

def load_prediction(output_wd, output_name, epoch = None):
    """
    This function loads the results of the evaluation using the test set.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    epoch : int, optional
        The epoch for loading the model. If None, the final model is loaded. The default is None.

    Returns
    -------
    output : dict
        The dictionary contains the prediction matrix plus the metrics defined in the header of the output.

    """
    prediction = np.loadtxt(os.path.join(output_wd, "%s_prediction_%s.txt"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")))
    output = {"prediction": prediction}
    
    with open(os.path.join(output_wd, "%s_prediction_%s.txt"%(output_name, "model" if type(epoch) == type(None) else f"{epoch:02d}")), "r") as f:
        fl = f.readline()
    aux = fl.replace("\n", "").replace("# ", "").split(", ")
    
    for i in aux:
        [name, value] = i.split(": ")
        output[name] = float(value)
    
    return output

def aggregate_prediction_results(output_wd, output_name, num_epochs):
    """
    This function aggregates the results of the prediction.

    Parameters
    ----------
    output_wd : string
        The working directory for the output.
    output_name : string
        The output name of the experiment.
    num_epochs : int
        The number of epochs.

    Returns
    -------
    None.

    """
    result = {}
    for epoch in range(num_epochs):
        output = load_prediction(output_wd, output_name, epoch + 1)
        for key in output.keys():
            if key != "prediction":
                if key not in result.keys():
                    result[key] = [output[key]]
                else:
                    result[key].append(output[key])
    header = ["epoch"] + [i for i in result.keys() if "score" in i] + [i for i in result.keys() if "Top-1-Acc" == i] + [i for i in result.keys() if "Top-1-Acc" != i and "Top" in i] + [i for i in result.keys() if "score" not in i and "prediction" != i and "Top" not in i]
    np.savetxt(os.path.join(output_wd, "%s_prediction_model.txt"%(output_name)), np.transpose(np.array([result[i] if i != "epoch" else np.arange(1, num_epochs+1).tolist() for i in header])), header = ", ".join(header))
