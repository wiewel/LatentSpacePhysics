#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# custom callbacks
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
from keras.callbacks import Callback
from util.plot import Plotter
import numpy as np

#---------------------------------------------------------------------------------
class PlotVectorsCallback(Callback):
    def __init__(self, model, x):
        self._model = model
        self._x = x[np.newaxis, ...]
        self._plotter = Plotter()

    def on_epoch_end(self, acc, loss):
        self._y = self._model.predict(self._x)
        self._plotter.plot_vector_field(self._x[0], self._y[0])
        self._plotter.show(False)

#---------------------------------------------------------------------------------
class PlotRealsCallback(Callback):
    def __init__(self, model, x, output_directory, filename):
        self._model = model
        self._x = x[np.newaxis, ...]
        self._filename = filename
        self._out_dir = output_directory
        self._plotter = Plotter()

    def on_epoch_end(self, acc, loss):
        self._y = self._model.predict(self._x)
        self._plotter.plot_heatmap(self._x[0], self._y[0])
        self._plotter.show(False)
        self._plotter.save_figures(self._out_dir, filename=self._filename, filetype="png")

#---------------------------------------------------------------------------------
class StatefulResetCallback(Callback):
    def __init__(self, model):
        self.model = model
        self.counter = 0
        
    def on_batch_end(self, batch, logs={}):
        self.counter = self.counter + 1
        if self.counter % 2 == 0:
            print("Resetting states")
            self.model.reset_states()

#---------------------------------------------------------------------------------
class LossHistory(Callback):
    def __init__(self, plot_callback):
        assert plot_callback is not None, "plot_callback can not be 'None'"
        self.plot_callback = plot_callback

    def on_train_begin(self, logs={}):
        self.train_losses = {}
        self.val_losses = {}

    def on_epoch_end(self, epoch, logs):
        self.train_losses[epoch] = logs.get('loss')
        self.val_losses[epoch] = logs.get('val_loss')
        self.plot_callback(self.train_losses, self.val_losses)
