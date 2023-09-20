#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# split_naturgucker.py
# Copyright (C) 2022 flossCoder
# 
# split_naturgucker is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# split_naturgucker is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Mon Jan 24 10:43:09 2022

@author: flossCoder
"""

from train_test_eval_sampler import generate_train_test_eval_split

def split_naturgucker(data, data_index, train_frac, test_frac, val_frac, seed = 42):
    """
    This function splits the csv file of the naturgucker dataset in a random fashion.

    Parameters
    ----------
    data : numpy array
        The input from naturgucker.de.
    data_index : integer
        The column used for splitting (0 = Ordnung, 1 = Familie, 2 = Gattung, 3 = Art).
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
        [train_set, test_set, val_set].
        train_set : The data subset for training.
        test_set : The data subset for testing.
        val_set : The data subset for validation.

    """
    [train_index, test_index, val_index] = generate_train_test_eval_split(data[:,data_index], train_frac, test_frac, val_frac, seed)
    
    return [data[train_index], data[test_index], data[val_index]]
