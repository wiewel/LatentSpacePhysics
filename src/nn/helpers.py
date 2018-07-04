#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# model helper functions
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

import keras
from keras.layers import Activation, LeakyReLU
from keras.models import Model

#--------------------------------------------
# Freeze weights in all layers of a network
def make_layers_trainable(net, val, recurse=False):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        if recurse and isinstance(l, Model):
            make_layers_trainable(l, val)

#--------------------------------------------
# Prints weight summary of all layers of a network
# inputs are all subclasses of Container, e.g. Model
def print_weight_summary(name, container):
    def print_weight_layer(tabs, container):
        for layer in container:
            print("{}{} Trainable Weights: {}".format(tabs, layer.name, len(layer.trainable_weights)))
            print("{}{} Non-Trainable Weights: {}".format(tabs, layer.name, len(layer.non_trainable_weights)))
            if isinstance(layer, Model):
                print_weight_layer(tabs+"\t", layer.layers)
    print("{} Trainable Weights: {}".format(name, len(container.trainable_weights)))
    print("{} Non-Trainable Weights: {}".format(name, len(container.non_trainable_weights)))
    print_weight_layer("\t", container.layers)