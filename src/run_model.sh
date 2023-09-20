# run_model.sh
# Copyright (C) 2022 flossCoder
# 
# run_model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# run_model is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

# This skript executes the training and test of a model. Following arguments are required:
# argv[0]: PATH_TO_CONFIG, where the CONFIG.py exists.
# argv[1]: CONFIG.py contains all parameter required for executing the AI model.
# argv[2]: The number of epochs for training.
python model_executor.py $1 $2
bash run_model_test_executor.sh $1 $2 $3
