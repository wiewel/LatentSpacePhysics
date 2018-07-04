#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# prediction network class
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

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Merge, RepeatVector, Add, LSTM, GRU, TimeDistributed, Dense, Reshape, Activation, Dropout, merge, Lambda, Bidirectional, Conv1D, Flatten
from keras.optimizers import RMSprop
from keras.regularizers import l1_l2, l2
from keras.losses import mean_absolute_error, mean_squared_error, mean_squared_logarithmic_error

import argparse
import time
import numpy as np
import random

from ..helpers import print_weight_summary, make_layers_trainable
from ..callbacks import StatefulResetCallback
from ..losses import Loss, LossType

from ..lstm import error_classification
from util import array
from math import floor

from .architecture import Network

class Lstm(Network):
    """ Long Short Term Memory for prediction of future code layer values """
    #---------------------------------------------------------------------------------
    def _init_vars(self, **kwargs):
        settings = kwargs.get("settings")
        # Training Settings
        self.model_name = "LSTM"
        self.data_dimension = kwargs.get("data_dimension", settings.ae.code_layer_size)
        self.time_steps = settings.lstm.time_steps
        self.stateful = settings.lstm.stateful
        self.use_bidirectional = settings.lstm.use_bidirectional
        self.encoder_lstm_neurons = settings.lstm.encoder_lstm_neurons
        self.decoder_lstm_neurons = settings.lstm.decoder_lstm_neurons
        self.attention_neurons = settings.lstm.attention_neurons
        self.out_time_steps = settings.lstm.out_time_steps
        self.lstm_activation = settings.lstm.activation
        self.batch_size = settings.lstm.batch_size

        self.use_deep_encoder = settings.lstm.use_deep_encoder

        self.use_time_conv_encoder = settings.lstm.use_time_conv_encoder
        self.time_conv_encoder_kernel = settings.lstm.time_conv_encoder_kernel
        self.time_conv_encoder_dilation = settings.lstm.time_conv_encoder_dilation
        self.time_conv_encoder_filters = settings.lstm.time_conv_encoder_filters
        self.time_conv_encoder_depth = settings.lstm.time_conv_encoder_depth

        self.use_time_conv_decoder = settings.lstm.use_time_conv_decoder
        self.time_conv_decoder_filters = settings.lstm.time_conv_decoder_filters
        self.time_conv_decoder_depth = settings.lstm.time_conv_decoder_depth

        self.use_noisy_training = settings.lstm.use_noisy_training
        self.noise_probability = settings.lstm.noise_probability

        # Model
        self.model = None

        # Variables
        self.kernel_regularizer = None #5e-8
        self.recurrent_regularizer = None #5e-8
        self.bias_regularizer = None #5e-8
        self.activity_regularizer = None #5e-8

        self.dropout=0.0132 # hyperparameter checked
        self.recurrent_dropout=0.385 # hyperparameter checked

        self.learning_rate = 0.000126 # hyperparameter checked
        self.decay = 0.000334 # hyperparameter checked (for 100 scenes / 20 epochs)
        self.use_bias = True

        # Loss
        self.lstm_loss = settings.lstm.loss # Loss(LossType.mae)

    #---------------------------------------------------------------------------------
    def _init_optimizer(self, epochs=1):
        self.lstm_optimizer = RMSprop(  lr=self.learning_rate,
                                        rho=0.9,
                                        epsilon=1e-08,
                                        decay=self.decay)#,
                                        #clipnorm=0.5)
        return self.lstm_optimizer

    #---------------------------------------------------------------------------------
    def _compile_model(self):
        self.model.compile( loss=self.lstm_loss,
                            optimizer=self.lstm_optimizer,
                            metrics=['mean_squared_error', 'mean_absolute_error'])

    #---------------------------------------------------------------------------------
    def _build_model(self):
        """ build the model """
        inputs = Input(shape=(self.time_steps, self.data_dimension), dtype="float32", name='main_input')

        x = inputs

        if self.use_time_conv_encoder:
            # Transforms from shape (None, time_steps, dimension) to (None, 1, filters)
            enc_time_conv = Conv1D(filters=self.time_conv_encoder_filters, padding="causal", kernel_size=self.time_conv_encoder_kernel, dilation_rate=self.time_conv_encoder_dilation)
            time_conv_encoder_shape = enc_time_conv.compute_output_shape(input_shape=(None, self.time_steps, self.data_dimension))
            assert time_conv_encoder_shape[1] == self.time_steps, ("TimeConv shape transformation failed. Result is '{}'".format(time_conv_encoder_shape))
            x = enc_time_conv(x)

            # deep time convolutions (won't change the shape)
            for i in range(self.time_conv_encoder_depth):
                x = Conv1D(filters=self.time_conv_encoder_filters, kernel_size=1)(x)
                x = Activation('tanh')(x)

        if self.use_deep_encoder:
            x1 = self._add_RNN_layer_func(  previous_layer=x,
                                            output_dim=self.encoder_lstm_neurons,
                                            go_backwards=True,
                                            return_sequences=True,
                                            bidirectional=self.use_bidirectional,
                                            use_gru=False)
            x2 = self._add_RNN_layer_func(  previous_layer=x1,
                                            output_dim=self.encoder_lstm_neurons,
                                            go_backwards=False,
                                            return_sequences=True,
                                            bidirectional=False,
                                            use_gru=False)
            xtmp = Add()([x1,x2])
            x3 = self._add_RNN_layer_func(  previous_layer=xtmp,
                                            output_dim=self.encoder_lstm_neurons,
                                            go_backwards=False,
                                            return_sequences=True,
                                            bidirectional=False,
                                            use_gru=False)
            xtmp = Add()([xtmp,x3])
            x = self._add_RNN_layer_func(   previous_layer=xtmp,
                                            output_dim=self.encoder_lstm_neurons,
                                            go_backwards=False,
                                            return_sequences=False,
                                            bidirectional=False,
                                            use_gru=False)
        else:
            x = self._add_RNN_layer_func(   previous_layer=x,
                                            output_dim=self.encoder_lstm_neurons,
                                            go_backwards=True,
                                            return_sequences=False,
                                            bidirectional=self.use_bidirectional,
                                            use_gru=False)

        x = RepeatVector(self.out_time_steps)(x)
        x = self._add_RNN_layer_func(   previous_layer=x,
                                        output_dim=self.decoder_lstm_neurons,
                                        go_backwards=False,
                                        return_sequences=True,
                                        bidirectional=self.use_bidirectional,
                                        use_gru=False)
        
        if self.use_time_conv_decoder:
            for i in range(self.time_conv_decoder_depth):
                x = Conv1D(filters=self.time_conv_decoder_filters, kernel_size=1)(x)
                x = Activation('tanh')(x)
            x = Conv1D(filters=self.data_dimension, kernel_size=1)(x)
            if self.out_time_steps == 1:
                x = Flatten()(x)
        else:
            x = self._add_RNN_layer_func(   previous_layer=x,
                                            output_dim=self.data_dimension,
                                            go_backwards=False,
                                            return_sequences=self.out_time_steps > 1,
                                            bidirectional=False,
                                            use_gru=False)

        outputs = x

        self.model = Model(inputs=inputs, outputs=outputs)

    #---------------------------------------------------------------------------------
    def _inner_RNN_layer(self, use_gru, output_dim, go_backwards, return_sequences):
        activation=self.lstm_activation #def: tanh
        recurrent_activation='hard_sigmoid' #def: hard_sigmoid

        kernel_regularizer = l2(l=self.kernel_regularizer) if self.kernel_regularizer is not None else None
        recurrent_regularizer = l2(l=self.recurrent_regularizer) if self.recurrent_regularizer is not None else None
        bias_regularizer = l2(l=self.bias_regularizer) if self.bias_regularizer is not None else None
        activity_regularizer = l2(l=self.activity_regularizer) if self.activity_regularizer is not None else None

        if use_gru:
            return GRU( units=output_dim,
                        stateful=self.stateful,
                        go_backwards=go_backwards,
                        return_sequences=return_sequences,
                        activation=activation, #def: tanh
                        recurrent_activation=recurrent_activation, #def: hard_sigmoid
                        dropout=self.dropout, #def: 0.
                        recurrent_dropout=self.recurrent_dropout,  #def: 0.
                        )
        else:
            return LSTM(units=output_dim,
                        activation=activation, #def: tanh
                        recurrent_activation=recurrent_activation, #def: hard_sigmoid
                        use_bias=self.use_bias,
                        kernel_initializer='glorot_uniform',
                        recurrent_initializer='orthogonal',
                        bias_initializer='zeros',
                        unit_forget_bias=True,

                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,

                        kernel_constraint=None,
                        recurrent_constraint=None,
                        bias_constraint=None,

                        dropout=self.dropout, #def: 0.
                        recurrent_dropout=self.recurrent_dropout,  #def: 0.

                        return_sequences=return_sequences,
                        go_backwards=go_backwards,
                        stateful=self.stateful,
                        #return_state=True
                        )

    #---------------------------------------------------------------------------------
    def _add_RNN_layer_func(self, previous_layer, output_dim, go_backwards, return_sequences, bidirectional=False, use_gru=False):
        def _bidirectional_wrapper(use_bidirectional, inner_layer, merge_mode='concat'):
            if use_bidirectional:
                return Bidirectional(layer=inner_layer, merge_mode=merge_mode)
            else:
                return inner_layer

        x = _bidirectional_wrapper(
                use_bidirectional = bidirectional,
                merge_mode = 'sum',  #'concat',
                inner_layer = self._inner_RNN_layer(
                                    use_gru=use_gru,
                                    output_dim=output_dim,
                                    go_backwards=go_backwards,
                                    return_sequences=return_sequences))(previous_layer)
        return x

    #---------------------------------------------------------------------------------
    def load_model(self, model_path):
        """ Load the model from file """
        print("Loading model: " + model_path)
        try:
            self.model = load_model(
                filepath = model_path,
                custom_objects= {'MULTI_LOSS': self.lstm_loss })
        except Exception as e:
            print("EXCEPTION: {}".format(str(e)))

        # Model Summary
        self.model.summary()

    #---------------------------------------------------------------------------------
    def save_model(self, model_path):
        """ Save the model to file """
        # save model with weights and optimzier settings
        print("Saving model {}".format(model_path))
        self.model.save(filepath = model_path)

    #---------------------------------------------------------------------------------
    def _train(self, epochs = 5, **kwargs):
        # Arguments
        X = kwargs.get("X")
        Y = kwargs.get("Y")
        train_scenes = kwargs.get("train_scenes")
        validation_split = kwargs.get("validation_split")

        # Default values for optional parameters
        if validation_split == None:
            validation_split = 0.1

        # Train
        model = self.model
        history = None # not supported in stateful currently
        train_generator = None
        validation_generator = None
        train_gen_nb_samples = 0
        val_gen_nb_samples = 0

        if train_scenes is not None:
            # validation split
            validation_scenes = train_scenes[ floor(len(train_scenes) * (1.0 - validation_split)) : ]
            train_scenes = train_scenes[ : floor(len(train_scenes) * (1.0 - validation_split)) ]

            # use generator
            train_gen_nb_samples = self.__generator_nb_batch_samples(train_scenes)
            print ("Number of train batch samples per epoch: {}".format(train_gen_nb_samples))
            assert train_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            train_generator = self.__generator_scene_func(train_scenes, use_noisy_training=self.use_noisy_training)

            # validation samples
            val_gen_nb_samples = self.__generator_nb_batch_samples(validation_scenes)
            assert val_gen_nb_samples > 0, ("Batch size is too large for current scene samples/timestep settings. Training by generator not possible. Please adjust the batch size in the 'settings.json' file.")
            print ("Number of validation batch samples per epoch: {}".format(val_gen_nb_samples))
            validation_generator = self.__generator_scene_func(validation_scenes)

        try:
            trainingDuration = 0.0
            trainStartTime = time.time()
            if (self.stateful is True):
                if (train_generator is None):
                    assert X is not None and Y is not None, ("X or Y is None!")
                    for i in range(epochs):
                        model.fit(
                            X,
                            Y,
                            nb_epoch=1,
                            batch_size=self.batch_size,
                            shuffle=False)
                        model.reset_states()
                else:
                    reset_callback = StatefulResetCallback(model)
                    for i in range(epochs):
                        model.fit_generator(
                            generator=train_generator,
                            steps_per_epoch=train_gen_nb_samples, # how many batches to draw per epoch
                            epochs = 1,
                            verbose=1,
                            callbacks=[reset_callback],
                            validation_data=validation_generator,
                            validation_steps=val_gen_nb_samples,
                            class_weight=None,
                            workers=1)
                        model.reset_states()
            else:
                if (train_scenes is None):
                    assert X is not None and Y is not None, ("X or Y is None!")
                    history = model.fit(
                        X,
                        Y,
                        nb_epoch=epochs,
                        batch_size=self.batch_size,
                        shuffle=True)
                else:
                    history = model.fit_generator(
                        generator=train_generator,
                        steps_per_epoch=train_gen_nb_samples,
                        epochs = epochs,
                        verbose=1,
                        callbacks=None,
                        validation_data=validation_generator,
                        validation_steps=val_gen_nb_samples,
                        class_weight=None,
                        workers=1)
            trainingDuration = time.time() - trainStartTime
        except KeyboardInterrupt:
            print("Training duration (s): {}\nInterrupted by user!".format(trainingDuration))
        print("Training duration (s): {}".format(trainingDuration))
        
        return history

    #--------------------------------------------
    def predict(self, X, batch_size=1):
        prediction = None
        try:
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            prediction = self.model.predict(x=X, batch_size=batch_size)  # e.g. (1, 2048)
            lstm_prefix_shape = (prediction.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.out_time_steps, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            prediction = prediction.reshape( lstm_prefix_shape + data_shape ) #(prediction.shape[0], self.out_time_steps, *input_shape[2:]) )
        except Exception as e:
            print(str(e))

        return prediction

    #--------------------------------------------
    # Helper functions
    #--------------------------------------------
    def __get_in_scene_iteration_count(self, sample_count):
        return floor((sample_count + 1 - (self.time_steps+self.out_time_steps)) / self.batch_size)
    #--------------------------------------------
    def __generator_nb_batch_samples(self, enc_scenes):
        scene_count = len(enc_scenes) # e.g. 10 scenes
        sample_count = enc_scenes[0].shape[0] # with 250 encoded samples each
        return scene_count * self.__get_in_scene_iteration_count(sample_count) 
    #--------------------------------------------
    def __generator_scene_func(self, enc_scenes, use_noisy_training=False):
        shuffle = self.stateful is False
        scene_count = len(enc_scenes)
        sample_count = enc_scenes[0].shape[0]
        in_scene_iteration = self.__get_in_scene_iteration_count(sample_count)
        print("Scene Count: {}  Sample Count: {} In-Scene Iteration: {}".format(scene_count, sample_count, in_scene_iteration))

        # for attr in dir(enc_scenes[0]):
        #     print("{}".format(attr))

        while 1:
            for i in range(scene_count):
                scene = enc_scenes[i]
                #print("Scene Count: {} => Scene {}: {}".format(scene_count, i, scene.scene_length))
                for j in range(in_scene_iteration):
                    enc_data = scene

                    start = j * self.batch_size
                    end = sample_count
                    #print("Scene Count: {} => Scene {}: Start: {} End: {} Shape: {}".format(scene_count, i, start, end, enc_data.shape))
                    X, Y = error_classification.restructure_encoder_data(
                                        data = enc_data[start : end],
                                        time_steps = self.time_steps,
                                        out_time_steps = self.out_time_steps,
                                        max_sample_count = self.batch_size)

                    # convert to (#batch, #ts, element_size)
                    X = X.reshape(*X.shape[0:2], -1)
                    #print("Batch Size: {} -- X Shape: {} -> {}".format(self.batch_size, input_shape, X.shape))
                    Y = np.squeeze(Y.reshape(Y.shape[0], self.out_time_steps, -1))
                    #print("Batch Size: {} -- Y Shape: {} -> {}".format(self.batch_size, input_shape, Y.shape))

                    if use_noisy_training:
                        noise_sample_ts = random.randint(0, self.time_steps-1)
                        np_sample_ts = np.random.choice(np.array([True,False]), self.time_steps, p=[self.noise_probability,1.0-self.noise_probability]) # e.g. [True, False, False, False, ...]
                        np_batch_mask = np.random.choice(np.array([True,False]), self.batch_size) # e.g. [True, False, False, False, ...]
                        input_noise = np.random.uniform(low=-1.0, high=1.0, size=X[0, 0].shape) * np.std(X[random.randint(0, self.batch_size-1), noise_sample_ts]) * 0.5
                        X[np.ix_(np_batch_mask, np_sample_ts)] += input_noise

                    if shuffle:
                        array.shuffle_in_unison(X, Y)
                    yield X, Y

    #--------------------------------------------
    # Helps to find an average gradient norm. Use the result to set your optimizer clip value
    # -> https://github.com/fchollet/keras/issues/1370
    # returns (avg_norm, min_norm, max_max)
    def find_average_gradient_norm(self, train_scenes):
        # use generator
        train_gen_nb_samples = self.__generator_nb_batch_samples(train_scenes)
        print ("Number of train batch samples per epoch: {}".format(train_gen_nb_samples))
        train_generator = self.__generator_scene_func(train_scenes)

        # just checking if the model was already compiled
        if not hasattr(self.model, "train_function"):
            raise RuntimeError("You must compile your model before using it.")

        weights = self.model.trainable_weights  # weight tensors

        get_gradients = self.model.optimizer.get_gradients(self.model.total_loss, weights)  # gradient tensors

        input_tensors = [
            # input data
            self.model.inputs[0],
            # how much to weight each sample by
            self.model.sample_weights[0],
            # labels
            self.model.targets[0],
            # train or test mode
            K.learning_phase()
        ]

        grad_fct = K.function(inputs=input_tensors, outputs=get_gradients)

        steps = 0
        total_norm = 0
        s_w = None
        min_norm = float("inf")
        max_norm = -float("inf")
        while steps < train_gen_nb_samples:
            X, y = next(train_generator)
            # set sample weights to one
            # for every input
            if s_w is None:
                s_w = np.ones(X.shape[0])

            gradients = grad_fct([X, s_w, y, 0])

            current_norm = np.sqrt(np.sum([np.sum(np.square(g)) for g in gradients]))
            min_norm = min(min_norm, current_norm)
            max_norm = max(max_norm, current_norm)
            total_norm += current_norm
            steps += 1

        return (total_norm / float(steps), min_norm, max_norm)