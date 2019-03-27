#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# settings file abstraction
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#******************************************************************************

"""
Settings
========
Module for all settings of the DPFN project

Edit the __settings dict to add new settings, delete or change default values. 
Editing the settings.json only will result in settings that are not available to users that do not have your settings.json. It is not recommended.

Example:
```python
import settings

settings.load("settings.json")
project_name = settings.project.name
...
```
"""
import argparse
import json
import os.path
import collections
from collections import namedtuple
import sys


__settings_module = sys.modules[__name__]
__settings = {
    "dataset": {
        "pressure_static_normalization_factor": 3.0,
        "pressure_dynamic_normalization_factor": 5.5,
        "pressure_normalization_factor": 3.8,
        "velocity_normalization_factor": 5.7,
        "pressure_encoded_normalization_factor": 3.3, # 2.65 for liquid128
        "pressure_encoded_total_normalization_factor": 2.45, # 2.9 for smoke; 2.1 for liquid128
        "pressure_encoded_vae_normalization_factor": 0.7,
        "pressure_encoded_dynamic_normalization_factor" : 2.9,
        "velocity_encoded_normalization_factor": 3.2,
        "pressure_encoded_total_gas_normalization_factor": 2.9,
        "pressure_density_encoded_gas_normalization_factor": 3.3,
    },
    "ae": {
        "batch_size":          8,
        "code_layer_size":     2048,
        "dropout":             0.0,
        "learning_rate":       0.00015,
        "learning_rate_decay": 1e-05,
        "l1_regularization":   0.0,
        "l2_regularization":   0.0,
    },
    "lstm": {
        "time_steps":           6,
        "out_time_steps":       1,
        "encoder_lstm_neurons": 700,
        "decoder_lstm_neurons": 1500,
        "attention_neurons":    400,
        "activation":           'tanh',
        "loss":                 'mae',
        "batch_size":           32,
        "stateful":             False,
        "use_bidirectional":    False,
        "use_attention":        False,
        "use_deep_encoder":           False,
        "use_time_conv_encoder":      False,
        "time_conv_encoder_kernel":   2,
        "time_conv_encoder_dilation": 1,
        "time_conv_encoder_filters":  2048,
        "time_conv_encoder_depth":    0,
        "use_time_conv_decoder":      True,
        "time_conv_decoder_filters":  4096,
        "time_conv_decoder_depth":    0,
        "use_noisy_training": False,
        "noise_probability": 0.3,
    },
    "project": {
        "name":                 "default",
        "path":                 "../projects/",
    },
}

#-------------------------------------------------------------------------------------
def __dict_to_namedtuple(typename, dictionary):
    """ Convert a dictionary to a namedtuple """
    return namedtuple(typename, dictionary.keys())(**dictionary)

#-------------------------------------------------------------------------------------
def __set_dict(settings):
    __update_dict_recursive(__settings, settings)
    for key, val in __settings.items():
        setattr(__settings_module, key, __dict_to_namedtuple(key, val))

#-------------------------------------------------------------------------------------
def __update_dict_recursive(d, update):
    for key, value in update.items():
        if isinstance(value, collections.Mapping): 
            d[key] = __update_dict_recursive(d.get(key, {}), value)
        else:
            d[key] = update[key]
    return d

#-------------------------------------------------------------------------------------
def load(path):
    if os.path.isfile(path):
        with open(path, 'r') as f:
            __set_dict(json.load(f))
    with open(path, 'w') as f:
        json.dump(__settings, f, indent=4)

#-------------------------------------------------------------------------------------
def save(path):
    with open(path, 'w') as f:
        json.dump(__settings, f, indent=4)