#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# custom loss objectives
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

import tensorflow as tf
import keras.backend as K
from keras import objectives
from keras import callbacks
from keras import losses
from keras import regularizers
import math
import numpy as np
from enum import Enum

#=====================================================================================
class LossType(Enum):
    mae = 1
    mse = 2
    rmse = 3
    variance = 4
    weighted_mae_mse = 5
    weighted_mae_rmse = 6
    binary_crossentropy = 7
    weighted_tanhmse_mse = 8

class Loss(object):
    def __init__(self, loss_type=LossType.mae, **kwargs):
        self.__name__ = loss_type.name

        self.loss_ratio = kwargs.get("loss_ratio", 1.0)
        self.data_input_scale = kwargs.get("data_input_scale", 1.0)
        self.loss_type = loss_type

    def info_struct(self):
        info = [("Loss",
                    {
                        'Type': self.loss_type.name,
                        'Ratio': self.loss_ratio
                    }
                )]
        return info

    def __call__(self, y_true, y_pred):
        result = None
        if self.loss_type == LossType.mae:
            result = objectives.mean_absolute_error(y_true, y_pred)
        elif self.loss_type == LossType.mse:
            result = objectives.mean_squared_error(y_true, y_pred)
        elif self.loss_type == LossType.rmse:
            result = K.sqrt(objectives.mean_squared_error(y_true, y_pred))
        elif self.loss_type == LossType.variance:
            result = K.sqrt(objectives.mean_squared_error(y_true, y_pred)) - objectives.mean_absolute_error(y_true, y_pred)
        elif self.loss_type == LossType.weighted_mae_mse:
            loss1=objectives.mean_absolute_error(y_true, y_pred)
            loss2=objectives.mean_squared_error(y_true, y_pred)
            result = self.loss_ratio*loss1+(1.0- self.loss_ratio)*loss2
        elif self.loss_type == LossType.weighted_mae_rmse:
            loss1=objectives.mean_absolute_error(y_true, y_pred)
            loss2=K.sqrt(objectives.mean_squared_error(y_true, y_pred))
            result = self.loss_ratio*loss1+(1.0- self.loss_ratio)*loss2
        elif self.loss_type == LossType.binary_crossentropy:
            result = objectives.binary_crossentropy(y_true, y_pred)
        elif self.loss_type == LossType.weighted_tanhmse_mse:
            loss1 = losses.mean_squared_error(K.tanh(self.data_input_scale*y_true), K.tanh(self.data_input_scale*y_pred))
            loss2 = losses.mean_squared_error(y_true, y_pred)
            result = self.loss_ratio*loss1+(1.0- self.loss_ratio)*loss2
        else:
            assert False, ("Loss function not supported")

        return result

#=====================================================================================
class tanhMSE(object):
    """ tanhMSE loss -> highlights surfaces of smooth data fields """
    #---------------------------------------------------------------------------------
    def __init__(self, scale=1.0):
        self.scale = scale
        self.__name__ = "tanhMSE"
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return losses.mean_squared_error(K.tanh(self.scale*y_true), K.tanh(self.scale*y_pred))

#=====================================================================================
class BCE(object):
    """ binary crossentropy error loss """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "BCE"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return self._weight * objectives.binary_crossentropy(y_true, y_pred)

#=====================================================================================
class MAE(object):
    """ mean absolute error loss """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "MAE"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return self._weight * objectives.mean_absolute_error(y_true, y_pred)

#=====================================================================================
class MSE(object):
    """ mean squared error loss """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "MSE"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        loss = self._weight * objectives.mean_squared_error(y_true, y_pred)
        return loss

#=====================================================================================
class MAPE(object):
    """ mean absolute percentage error loss """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "MAPE"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return self._weight * objectives.mean_absolute_percentage_error(y_true, y_pred)

#=====================================================================================
class ExpRegularizer(object):
    """ regularize loss to weight smaller values stronger """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "Exp"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        loss = self._weight * tf.exp(- y_true)  
        return loss

#=====================================================================================
class MSLE(object):
    """ mean squared error loss """
    #---------------------------------------------------------------------------------
    def __init__(self, weight=1.0):
        self._weight = weight
        self.__name__ = "MSLE"

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return self._weight * objectives.mean_squared_logarithmic_error(y_true, y_pred)

#=====================================================================================
class L1L2Loss(object):
    """ loss = W1 * L1 + W2 * L2 """

    #---------------------------------------------------------------------------------
    def __init__(self, l1_factor=100.0, l2_factor=100.0, name="Combined L1 & L2 loss"):
        """
        :param l1_factor: weight of the L1 contribution
        :param l2_factor: weight of the L2 contribution
        """
        self.l1_factor=l1_factor
        self.l2_factor=l2_factor
        self.__name__ = name

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        l2_error = tf.losses.mean_squared_error(y_true, y_pred, weights=self.l2_factor)
        abs_error = tf.losses.absolute_difference(y_true, y_pred, weights=self.l1_factor)
        return tf.losses.compute_weighted_loss([l2_error, abs_error])

#=====================================================================================
class VAELoss(object):
    """ loss for a variational autoencoder """
    #---------------------------------------------------------------------------------
    class VAECallback(callbacks.Callback):
        def __init__(self, vae_loss):
            self.vae_loss = vae_loss

        def on_epoch_begin(self, acc, loss):
            self.vae_loss.increase_kl_beta(0.1)
            #print("KL contribution {:.1f}".format(self.vae_loss.kl_beta))

    #---------------------------------------------------------------------------------
    def __init__(self, z_mean, z_log_var, kl_beta, loss, name="VAE Loss"):
        """
        :param z_mean: the z_mean tensor of the VAE
        :param z_log_var: the z_log_var tensor of the VAE
        :param loss: Reconstruction loss of the VAE
        :param loss_weight: Factor by which to weight the reconstruction loss
        :param warm_up: slowly warm up the kl loss, to prevent zero code layers
        :param warm_up_speed
        """
        self.loss = loss
        self.loss_weight = 1.0
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        self.__name__ = name
        # slowly introduce kl divergence loss, to prevent zero code http://orbit.dtu.dk/files/121765928/1602.02282.pdf
        self._kl_beta = kl_beta
        self.vae_callback = self.VAECallback(self)

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        rec_loss = self.loss_weight * self.loss(y_true, y_pred)
        kl_loss = K.cast(self._kl_beta, dtype=K.floatx()) * -0.5 * K.sum(1.0 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=1)
        kl_loss = tf.expand_dims(kl_loss, 1)
        kl_loss = tf.expand_dims(kl_loss, 2)
        kl_loss = tf.expand_dims(kl_loss, 3)
        loss = tf.add(kl_loss, rec_loss)
        return loss

    #---------------------------------------------------------------------------------
    @property
    def kl_beta(self):
        return self._kl_beta

    #---------------------------------------------------------------------------------
    @kl_beta.setter
    def kl_beta(self, value):
        self._kl_beta = min(1.0, value)

    #---------------------------------------------------------------------------------
    def increase_kl_beta(self, amount=0.1):
        self._kl_beta = self._kl_beta + amount 

#=====================================================================================
class MultiLoss(object):
    """ combine multiple losses to a weighted multi loss """
    #---------------------------------------------------------------------------------
    def __init__(self, loss_list = []):
        self._add_loss = loss_list
        self._mul_loss = []
        self.__name__ = "MultiLoss"
    
    #---------------------------------------------------------------------------------
    def add_loss(self, loss, weight=1.0):
        self._add_loss.append((weight, loss))
    
    #---------------------------------------------------------------------------------
    def mul_loss(self, loss):
        self._mul_loss.append(loss)

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        result = 0.0
        for weight, loss in self._add_loss:
            result = tf.add(result, weight * loss(y_true, y_pred))
        for loss in self._mul_loss:
            result = tf.mul(result, loss)
        return result

#=====================================================================================
class Divergence(object):
    """ Divergence loss for velocity fields """
    #---------------------------------------------------------------------------------
    def __init__(self, shape=(64, 64)):
        self.__name__ = "Divergence"
        div = np.zeros(shape=shape)
        for j in range(int(shape[1]) - 1):
            div[j, j+1] = -1.0
            div[j+1, j] = 1.0
        self._div_tensor = tf.constant(div, dtype=tf.float32)

    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        result = tf.map_fn(lambda y: tf.matmul(y, self._div_tensor), y_pred[:,:,:,0])
        result += tf.map_fn(lambda y: tf.matmul(self._div_tensor, y), y_pred[:,:,:,1])
        result = tf.abs(tf.reduce_sum(result, axis=[1,2]))
        result = tf.expand_dims(result, 1)
        result = tf.expand_dims(result, 2)
        return result

#=====================================================================================
class PressureDivergence(object):
    """ Divergence loss for pressure fields with velocity field evaluation """
    #---------------------------------------------------------------------------------
    def __init__(self):
        self.__name__ = "Pressure Divergence"
    
    #---------------------------------------------------------------------------------
    def __call__(self, y_true, y_pred):
        return
    
