#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# additional training metrics
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
import tensorflow as tf
import keras.backend as K
import numpy as np

#=====================================================================================
class DivergenceMetric(object):
    """ Divergence loss for velocity fields """
    #---------------------------------------------------------------------------------
    def __init__(self, shape=(64, 64)):
        self.__name__ = "div"
        div = np.zeros(shape=shape)
        for j in range(int(shape[1]) - 1):
            div[j, j+1] = -1.0
            div[j+1, j] = 1.0
        self._div_tensor = tf.constant(div, dtype=tf.float32)

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        result = tf.map_fn(lambda y: tf.matmul(y, self._div_tensor), y_pred[:,:,:,0])
        result += tf.map_fn(lambda y: tf.matmul(self._div_tensor, y), y_pred[:,:,:,1])
        result = tf.reduce_sum(result, axis=[1,2])
        result = tf.reduce_mean(result)
        return result

#=====================================================================================
class KLDivergence(object):
    """ kullback leibler divergence """
    #---------------------------------------------------------------------------------
    def __init__(self, z_mean, z_log_var):
        self.__name__ = "kl_div"
        self.z_mean = z_mean
        self.z_log_var = z_log_var

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        kl_loss = -0.5 * K.sum(1.0 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)
        return kl_loss