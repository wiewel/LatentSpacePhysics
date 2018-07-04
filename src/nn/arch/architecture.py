#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# abstract network layout class
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

from abc import ABC, abstractmethod
#from ...util import requirements
import inspect

import numpy as np
import tensorflow as tf
from keras import backend as K

class Network(ABC):
    """ Superclass for all model architectures """
    #---------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        # Initialize Variables (Hyperparams etc)
        self.model = None
        self._init_vars(**kwargs)
        # Optimizer
        self._init_optimizer(1)
        # # Build & Compile -> Call this by yourself when model is loaded by weights only
        # self._build_model()
        # self._compile_model()

    #---------------------------------------------------------------------------------
    def train(self, epochs, **kwargs):
        """ Trains and returns the training history """
        # Reset all random number generators to given seeds
        #requirements.reset_rng()
        np.random.seed(4213)
        tf.set_random_seed(3742)

        # Destroys the current TF graph and creates a new one.
        # Useful to avoid clutter from old models / layers.
        if self.model is not None:
            del self.model
            self.model = None
        K.clear_session()

        # Recompile (in case of updated hyper parameters)
        self._init_optimizer(epochs)
        self._build_model()
        self._compile_model()
        # Model Summary
        self.model.summary()
        self.print_attributes()
        # Train and return History
        history = self._train(epochs, **kwargs)
        return history

    #---------------------------------------------------------------------------------
    # Interface
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _init_vars(self, **kwargs):
        """ Setup internal variables like hyper parameters """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _init_optimizer(self, epochs=1):
        """ Initializes and stores the optimizer """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _build_model(self):
        """ Build the network structure and store in self.model """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _compile_model(self):
        """ Calls compile on the model """
        pass
    #---------------------------------------------------------------------------------
    @abstractmethod
    def _train(self, epochs, **kwargs):
        """ Trains and returns the training history """
        return
    #---------------------------------------------------------------------------------
    @abstractmethod
    def predict(self, X, batch_size):
        """ Predict the results of X using the compiled model and return the result """
        return

    #---------------------------------------------------------------------------------
    # Functions
    #---------------------------------------------------------------------------------
    # e.g. params = {"learning_rate": 0.1, "use_bias": True, ...}
    def update_parameters(self, params):
        """ Update the member variables by passing in names and new values as a dict """
        for key, value in params.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                print("[WARNING] Key {} does not exist!".format(key))
    #---------------------------------------------------------------------------------
    def print_attributes(self):
        attribute_list = []
        max_length = 10
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):
                if not inspect.isclass(i[1]) and not inspect.ismethod(i[1]):
                    attribute_list.append(i)
                    max_length = max_length if len(i[0]) < max_length else len(i[0])
        for attr in attribute_list:
            print("{:{width}}: {}".format(attr[0], attr[1], width=max_length))