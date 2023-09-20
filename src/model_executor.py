#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# model_executor.py
# Copyright (C) 2022 flossCoder
# 
# model_executor is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.for k
# 
# model_executor is distributed in the hope that it will be useful, but
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
import importlib.util
import os

def load_module(path, module_name):
    """
    This function loads a specific module.

    Parameters
    ----------
    path : string
        Path to the module.
    module_name : string
        Name of the module.

    Returns
    -------
    module :object
        The object of the module.

    """
    spec = importlib.util.spec_from_file_location("", os.path.join(path, module_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Model_Executor:
    """
    This class is responsible for loading the modules.
    """
    def __init__(self):
        """
        The init function saves the path of this python file plus the name of the file (in the same directory than this module) that should be executed.

        Returns
        -------
        None.

        """
        self.path = os.path.split(os.path.realpath(__file__))[0]
    
    def load_module(self, module_name):
        """
        This function loads the specified module, which must be saved in the same directory than this module.

        Parameters
        ----------
        module_name : string
        Name of the module.

        Returns
        -------
        module : object
            The object of the module, specified in self.filename.

        """
        module = load_module(self.path, module_name)
        return module

def main(config_path, config_filename):
    """
    This function executes the model.

    Parameters
    ----------
    config_path : string
        The path to the configuration file.
    config_filename : string
        The name of the config file.

    Returns
    -------
    None.

    """
    module = load_module(config_path, config_filename)
    module.main(Model_Executor(), config_path)

if __name__ == "__main__":
    """
    Usage:
        python model_executor.py PATH_TO_CONFIG CONFIG.py
    
    Assume:
        This module (module_executor.py) is in the same directory than MODEL.py (specified in CONFIG.py).
        MODEL.py describes the AI model under usage (like cnn-architecture, loading pre-trained weights, ...).
        In PATH_TO_CONFIG the CONFIG.py exists.
        CONFIG.py contains all parameter required for executing the AI model.
        CONFIG.py implements the main(model_executor) function used in this module.
    """
    main(sys.argv[1], sys.argv[2])
