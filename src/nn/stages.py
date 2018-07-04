#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# stage class used in autoencoder to build up structure for pretraining
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
from keras.models import Model
from keras.layers import Input
import numpy as np

def convert_shape(shape):
    out_shape = []
    for i in shape:
        try:
            out_shape.append(int(i))
        except:
            out_shape.append(None)
    return out_shape

#=====================================================================================
class StagedModel:
    """ convenience class for building models with multiple stages """

    #---------------------------------------------------------------------------------
    def start(self, first_input):
        self.stages = []
        assert input is not None, ("input to the StagedModel must be provided")
        # get parameters for first stage
        self.current_input = first_input
        return self.current_input

    #---------------------------------------------------------------------------------
    def stage(self, output):
        """ 
        end previous stage and begin new one
        :param output: the output of the previous stage
        :return: the input of the following stage
        """
        # build an intermediate stage
        stage = Model(inputs=self.current_input, outputs=output)
        self.stages.append(stage)
        self.current_input = Input(shape=stage.output_shape[1:])
        return self.current_input

    #---------------------------------------------------------------------------------
    def end(self, output):
        """
        end the staged model description
        :param output: the output of the last stage
        """
        # build the last stage
        stage = Model(inputs=self.current_input, outputs=output)
        self.stages.append(stage)

    #---------------------------------------------------------------------------------
    @property
    def model(self):
        # build the model that contains all stages
        model_input = []
        for layer_input in self.stages[0].inputs:
            shape = convert_shape(layer_input.shape[1:]) #[int(i) for i in layer_input.shape[1:]]
            model_input.append(Input(shape=shape))
        if len(model_input) == 1:
            model_input = model_input[0]
        x = model_input
        for stage in self.stages:
            x = stage(x)
        model_output = x
        model = Model(inputs = model_input, outputs= model_output)
        return model

    #---------------------------------------------------------------------------------
    def __getitem__(self, key):
        if isinstance(key, slice):
            substages = self.stages[key]
            model_input = []
            for layer_input in substages[0].inputs:
                shape = convert_shape(layer_input.shape[1:]) #[int(i) for i in layer_input.shape[1:]]
                model_input.append(Input(shape=shape))
            if len(model_input) == 1:
                model_input = model_input[0]
            x = model_input
            for index, stage in enumerate(substages):
                x = stage(x)
            model_output = x
            model = Model(inputs = model_input, outputs = model_output)
            return model
        else:
            return self.stages[key]