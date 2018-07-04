#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# KL Div regularizer
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
from keras.regularizers import Regularizer
import keras.backend as K
import tensorflow as tf

class KLDivergence(Regularizer):
    def __init__(self, z_mean, z_log_var, kl_beta = 1.0, ):
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self._kl_beta = kl_beta

    def __call__(self, x):
        kld = K.cast(self._kl_beta, dtype=K.floatx()) * -0.5 * K.sum(1.0 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=1)
        kld = tf.expand_dims(kld, 1)
        kld = tf.expand_dims(kld, 2)
        kld = tf.expand_dims(kld, 3)
        return kld

    def get_config(self):
        return {'kl_beta':self._kl_beta}
