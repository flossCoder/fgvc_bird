#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model_test_executor.py
# Copyright (C) 2022 flossCoder
# 
# model_test_executor is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# model_test_executor is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on Thu Feb 17 11:24:51 2022

@author: flossCoder
"""

import sys
from model_executor import load_module, Model_Executor

def main(config_path, config_filename, epoch = None):
    """
    This function executes the model.

    Parameters
    ----------
    config_path : string
        The path to the configuration file.
    config_filename : string
        The name of the config file.
    epoch : int, optional
        The epoch for loading the model. If None, the results are aggregated. The default is None

    Returns
    -------
    None.

    """
    module = load_module(config_path, config_filename)
    module.main(Model_Executor(), config_path, True, epoch)

if __name__ == "__main__":
    """
    Usage:
        python model_test_executor.py PATH_TO_CONFIG CONFIG.py
    
    Assume:
        This module (module_test_executor.py) is in the same directory than MODEL.py (specified in CONFIG.py).
        MODEL.py describes the AI model under usage (like cnn-architecture, loading pre-trained weights, ...).
        In PATH_TO_CONFIG the CONFIG.py exists.
        CONFIG.py contains all parameter required for executing the AI model.
        CONFIG.py implements the main(model_executor) function used in this module.
        epoch (optional) the epoch that should be loaded.
    """
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
