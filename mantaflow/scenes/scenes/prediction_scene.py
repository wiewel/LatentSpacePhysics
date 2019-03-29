#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta prediction scene
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

from scenes.scene import FLIPScene
from scenes.smoke_scene import SmokeScene

from manta import *
import numpy as np
import keras
import json
import os

assert keras.__version__ == "2.1.6", ("Only Keras 2.1.6 is supported. Currently installed Keras version is {}.".format(keras.__version__))

#----------------------------------------------------------------------------------
# profiling tools
import tensorflow as tf
from tensorflow.python.client import timeline

profile_execution = False
profile_counter = 0
profile_step = 12
if profile_execution:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                output_partition_graphs=True)
    run_metadata_encoder = tf.RunMetadata()
    run_metadata_decoder = tf.RunMetadata()
    run_metadata_lstm = tf.RunMetadata()

#----------------------------------------------------------------------------------
class PredictionHistory(object):
    def __init__(self, in_ts, data_shape):
        self.lstm_input_shape = (1, in_ts) # (batch_size, in_ts)
        self.data_shape = data_shape #(1, 1, 1, 1024)
        self.simulation_history = np.zeros( self.lstm_input_shape + self.data_shape )
        self.last_prediction = None

    # append element of shape data_shape
    def _history_append_back(self, element):
        # overwrite first batch entry (only one is used)
        self.simulation_history[0] = np.insert( np.delete(self.simulation_history[0], 0, axis=0),
                                                self.lstm_input_shape[1] - 1,
                                                element,
                                                axis=0)

    # add simulation frame in AE code layer format -> e.g. (data_shape)
    def add_simulation(self, new_frame):
        assert new_frame.shape == self.data_shape, "Shape of simulation frame {} must match simulation history shape {}".format(new_frame.shape, self.data_shape)
        # invalidate last prediction
        self.last_prediction = None
        self._history_append_back(new_frame)

    # add predictions frame(s) in AE code layer format (without batch_size!) -> e.g. (out_ts, data_shape)
    def add_prediction(self, prediction):
        assert prediction[0].shape == self.data_shape, "Shape of prediction data {} must match simulation history shape {}".format(prediction[0].shape, self.data_shape)
        # add first prediction step to history
        self._history_append_back(prediction[0])
        # add the remaining predictions to last_prediction
        self.last_prediction = prediction[1:] if prediction.shape[0] > 1 else None

    # returns last predictions with shape: (remaining_steps, data_shape) -> [0] is the oldest prediction
    def get_last_prediction(self):
        return self.last_prediction

    def get(self):
        return self.simulation_history



#----------------------------------------------------------------------------------
def get_output_shape(model, input_shape):
    assert input_shape is not None, ("You must provide an input shape for autoencoders with variable input sizes")
    if isinstance(input_shape, list):
        dummy_input = [np.expand_dims(np.zeros(input_shape[0]), axis = 0), np.expand_dims(np.zeros(input_shape[1]), axis = 0)]
    else:
        dummy_input = np.expand_dims(np.zeros(input_shape), axis = 0)
    shape = model.predict(dummy_input, batch_size=1).shape[1:]
    return shape


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class SplitPredictionScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), smooth_prediction=True, store_divergence_history=True, predict_dynamic_only=False, name="SplitPredictionScene", show_gui=True):
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps
        self.predict_dynamic_only = predict_dynamic_only
        self.smooth_prediction = smooth_prediction

        # input shape: [(None, 64, 64, 64, 1), (None, 64, 64, 64, 1)]
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5", compile=not profile_execution)
        enc_shape = get_output_shape(self.encoder, [(resolution, resolution, resolution, 1), (resolution, resolution, resolution, 1)] )
        self.code_layer_shape = enc_shape #self.encoder.output_shape[1:] # e.g. (1, 1, 1, 1024)

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5", compile=not profile_execution)
        self.lstm = keras.models.load_model(project_directory + "/lstm.h5", compile=not profile_execution)
        if profile_execution:
            self.encoder.compile(loss='MSE', optimizer='Adam', options=run_options, run_metadata=run_metadata_encoder)
            self.decoder.compile(loss='MSE', optimizer='Adam', options=run_options, run_metadata=run_metadata_decoder)
            self.lstm.compile(loss='MAE', optimizer='RMSProp', options=run_options, run_metadata=run_metadata_lstm)
            
        self.lstm_in_ts = self.lstm.input_shape[1] # e.g. 6
        self.lstm_out_ts = self.lstm.output_shape[1] if len(self.lstm.output_shape) > 2 else 1 # (None, 1024) or (None, 3, 1024)

        def load_description(desc_name):
            desc = None
            if os.path.isfile(project_directory + "/" + desc_name):
                with open(project_directory + "/" + desc_name, 'r') as f:
                    desc = json.load(f)
            assert desc is not None, ("model description '" + desc_name + "' not found")
            return desc

        self.ae_description = load_description("description_autoencoder")
        self.ae_settings = load_description("arguments_autoencoder.json")
        self.lstm_description = load_description("description_lstm")

        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape) 

        super(SplitPredictionScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, store_divergence_history=store_divergence_history, merge_ghost_fluid=False, name=name, show_gui=show_gui)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(SplitPredictionScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")

        self.np_pressure_sta = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
        self.np_pressure_dyn = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(SplitPredictionScene,self)._reset()
        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape)

        if profile_execution:
            trace = timeline.Timeline(step_stats=run_metadata_encoder.step_stats)
            with open('timeline_split_encoder.ctf.json', 'w') as f:
                f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
            trace = timeline.Timeline(step_stats=run_metadata_decoder.step_stats)
            with open('timeline_split_decoder.ctf.json', 'w') as f:
                f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
            trace = timeline.Timeline(step_stats=run_metadata_lstm.step_stats)
            with open('timeline_split_lstm.ctf.json', 'w') as f:
                f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # predict frame
            self._predict_step()

            if self.predict_dynamic_only:
                # get static pressure from levelset
                computeStaticPressure(self.phi_fluid, self.flags, self.gravity, self.pressure_static, 6)
                # add dynamic pressure part evaluated by NN
                # dynamic pressure is stored in self.pressure in predict_step() -> since combinePressure kernel works on distinct idx no overlapping read/write
                combinePressure(self.pressure, self.pressure_static, self.pressure)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            if self.boundary>0:
                self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)
            if self.smooth_prediction:
                smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
                smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)
                smoothPressureCorner(flags=self.flags, pressure=self.pressure, boundary=self.boundary)

            if self.boundary>0:
                self.pressure.setBoundNeumann(self.boundary-1)
        else:
            # simulate frame
            super(SplitPredictionScene,self)._solve_pressure()
            # if only simulation should happen, skip the pressure split
            separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6)
            if self.prediction_interval > 1:
                self._encode_simulation_results()

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure is the new predicted pressure value in denormalized simulation format
    def _predict_step(self):
        predicted_pressure = self.prediction_history.get_last_prediction()
        if predicted_pressure is None:
            X = self.prediction_history.get()
            # predict new pressure field
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            predicted_pressure = self.lstm.predict(x=X, batch_size=1)
            lstm_prefix_shape = (predicted_pressure.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.lstm_out_ts, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            predicted_pressure = predicted_pressure.reshape( lstm_prefix_shape + data_shape )
            # remove batch dimension
            predicted_pressure = predicted_pressure[0]

        # add to history
        self.prediction_history.add_prediction(predicted_pressure)
        # take the current (first) prediction step
        predicted_pressure = np.take(a=predicted_pressure, indices=[0], axis=0)

        # denormalize (lstm)
        if self.ae_settings.get("ae_vae", False):
            predicted_pressure *= self.lstm_description["dataset"]["pressure_encoded_vae_normalization_factor"]
        else:
            predicted_pressure *= self.lstm_description["dataset"]["pressure_encoded_normalization_factor"]
        # decode (ae)
        [pressure_stat, pressure_dyn] = self.decoder.predict(x=predicted_pressure, batch_size=1)

        # denormalize (stat/dyn ae)
        pressure_stat *= self.ae_description["dataset"]["pressure_static_normalization_factor"]
        pressure_dyn  *= self.ae_description["dataset"]["pressure_dynamic_normalization_factor"]
        # calculate combined pressure field
        if self.predict_dynamic_only:
            np_pressure = pressure_dyn
            if self.show_gui:
                copyArrayToGridReal(pressure_dyn, self.pressure_dynamic)
        else:
            np_pressure = pressure_dyn + pressure_stat
            if self.show_gui:
                copyArrayToGridReal(pressure_dyn, self.pressure_dynamic)
                copyArrayToGridReal(pressure_stat, self.pressure_static)

        copyArrayToGridReal(np_pressure, self.pressure)

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure_stat and pressure_dyn are in denormalized simulation format
    def _encode_simulation_results(self):
        copyGridToArrayReal(self.pressure_static, self.np_pressure_sta)
        copyGridToArrayReal(self.pressure_dynamic, self.np_pressure_dyn)

        # normalize (stat/dyn ae)
        self.np_pressure_sta /= self.ae_description["dataset"]["pressure_static_normalization_factor"]
        self.np_pressure_dyn /= self.ae_description["dataset"]["pressure_dynamic_normalization_factor"]

        # encode (ae)
        encoded_pressure = self.encoder.predict(x=[self.np_pressure_sta, self.np_pressure_dyn], batch_size=1)
        # remove batch dimension
        encoded_pressure = encoded_pressure[0]
        # normalize (lstm)
        if self.ae_settings.get("ae_vae", False):
            encoded_pressure /= self.lstm_description["dataset"]["pressure_encoded_vae_normalization_factor"]
        else:
            encoded_pressure /= self.lstm_description["dataset"]["pressure_encoded_normalization_factor"]
        # add to history
        self.prediction_history.add_simulation(encoded_pressure)

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class TotalPredictionScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), smooth_prediction=True, store_divergence_history=True, name="TotalPredictionScene", show_gui=True):
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps
        self.smooth_prediction = smooth_prediction

        # input shape: (None, 64, 64, 64, 1)
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5", compile=not profile_execution)
        if len(self.encoder.inputs[0].shape) == 4: # e.g. (?, 64, 64, 1)
            # 2D
            enc_shape = get_output_shape(self.encoder, (resolution, resolution, 1))
            dimension = 2
        else: # e.g. (?, 64, 64, 64, 1)
            # 3D
            enc_shape = get_output_shape(self.encoder, (resolution, resolution, resolution, 1))
            dimension = 3

        self.code_layer_shape = enc_shape #self.encoder.output_shape[1:] # e.g. (1, 1, 1, 1024)

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5", compile=not profile_execution)
        self.lstm = keras.models.load_model(project_directory + "/lstm.h5", compile=not profile_execution)
        if profile_execution:
            self.encoder.compile(loss='MSE', optimizer='Adam', options=run_options, run_metadata=run_metadata_encoder)
            self.decoder.compile(loss='MSE', optimizer='Adam', options=run_options, run_metadata=run_metadata_decoder)
            self.lstm.compile(loss='MAE', optimizer='RMSProp', options=run_options, run_metadata=run_metadata_lstm)

        self.lstm_in_ts = self.lstm.input_shape[1] # e.g. 6
        self.lstm_out_ts = self.lstm.output_shape[1] if len(self.lstm.output_shape) > 2 else 1 # (None, 1024) or (None, 3, 1024)

        def load_description(desc_name):
            desc = None
            if os.path.isfile(project_directory + "/" + desc_name):
                with open(project_directory + "/" + desc_name, 'r') as f:
                    desc = json.load(f)
            assert desc is not None, ("model description '" + desc_name + "' not found")
            return desc

        self.ae_description = load_description("description_autoencoder")
        self.ae_settings = load_description("arguments_autoencoder.json")
        self.lstm_description = load_description("description_lstm")

        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape) 

        super(TotalPredictionScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(TotalPredictionScene,self)._init_grids()
        if self.dimension == 2:
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _profile_step(self):
        global profile_counter
        if profile_execution:
            timestep = self.solver.frame
            if (profile_counter % profile_step) == 0:
                trace = timeline.Timeline(step_stats=run_metadata_encoder.step_stats)
                with open('timeline_{}_encoder_{}.ctf.json'.format(self.name, timestep), 'w') as f:
                    f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                trace = timeline.Timeline(step_stats=run_metadata_decoder.step_stats)
                with open('timeline_{}_decoder_{}.ctf.json'.format(self.name, timestep), 'w') as f:
                    f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
                trace = timeline.Timeline(step_stats=run_metadata_lstm.step_stats)
                with open('timeline_{}_lstm_{}.ctf.json'.format(self.name, timestep), 'w') as f:
                    f.write(trace.generate_chrome_trace_format(show_dataflow=True, show_memory=True))
            profile_counter += 1

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(TotalPredictionScene,self)._reset()
        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape)

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # predict frame
            self._predict_step()

            # profile
            self._profile_step()

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            if self.boundary>0:
                self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)
            if self.smooth_prediction:
                smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
                smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)
                smoothPressureCorner(flags=self.flags, pressure=self.pressure, boundary=self.boundary)

            if self.boundary>0:
                self.pressure.setBoundNeumann(self.boundary-1)
        else:
            # simulate frame
            super(TotalPredictionScene,self)._solve_pressure()
            # if only simulation should happen, skip the pressure split
            if self.prediction_interval > 1:
                self._encode_simulation_results()

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure is the new predicted pressure value in denormalized simulation format
    def _predict_step(self):
        predicted_pressure = self.prediction_history.get_last_prediction()
        if predicted_pressure is None:
            X = self.prediction_history.get()
            # predict new pressure field
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            predicted_pressure = self.lstm.predict(x=X, batch_size=1)
            lstm_prefix_shape = (predicted_pressure.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.lstm_out_ts, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            predicted_pressure = predicted_pressure.reshape( lstm_prefix_shape + data_shape )
            # remove batch dimension
            predicted_pressure = predicted_pressure[0]

        # add to history
        self.prediction_history.add_prediction(predicted_pressure)
        # take the current (first) prediction step
        predicted_pressure = np.take(a=predicted_pressure, indices=[0], axis=0)

        # denormalize (lstm)
        predicted_pressure *= self.lstm_description["dataset"]["pressure_encoded_total_normalization_factor"]
        # decode (ae)
        self.np_pressure = self.decoder.predict(x=predicted_pressure, batch_size=1)

        # denormalize (ae)
        self.np_pressure *= self.ae_description["dataset"]["pressure_normalization_factor"]

        # store in grid
        copyArrayToGridReal(self.np_pressure, self.pressure)

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure_stat and pressure_dyn are in denormalized simulation format
    def _encode_simulation_results(self):
        copyGridToArrayReal(self.pressure, self.np_pressure)

        # normalize (ae)
        self.np_pressure /= self.ae_description["dataset"]["pressure_normalization_factor"]

        # encode (ae)
        encoded_pressure = self.encoder.predict(x=self.np_pressure, batch_size=1)
        # remove batch dimension
        encoded_pressure = encoded_pressure[0]
        # normalize (lstm)
        encoded_pressure /= self.lstm_description["dataset"]["pressure_encoded_total_normalization_factor"]
        # add to history
        self.prediction_history.add_simulation(encoded_pressure)



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class DynamicPredictionScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), smooth_prediction=True, store_divergence_history=True, name="DynamicPredictionScene", show_gui=True):
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps
        self.smooth_prediction = smooth_prediction

        # input shape: [(None, 64, 64, 64, 1), (None, 64, 64, 64, 1)]
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5")
        enc_shape = get_output_shape(self.encoder, [(resolution, resolution, resolution, 1), (resolution, resolution, resolution, 1)] )
        self.code_layer_shape = enc_shape #self.encoder.output_shape[1:] # e.g. (1, 1, 1, 1024)

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5")
        self.lstm = keras.models.load_model(project_directory + "/lstm.h5")
        self.lstm_in_ts = self.lstm.input_shape[1] # e.g. 6
        self.lstm_out_ts = self.lstm.output_shape[1] if len(self.lstm.output_shape) > 2 else 1 # (None, 1024) or (None, 3, 1024)

        def load_description(desc_name):
            desc = None
            if os.path.isfile(project_directory + "/" + desc_name):
                with open(project_directory + "/" + desc_name, 'r') as f:
                    desc = json.load(f)
            assert desc is not None, ("model description '" + desc_name + "' not found")
            return desc

        self.ae_description = load_description("description_autoencoder")
        self.ae_settings = load_description("arguments_autoencoder.json")
        self.lstm_description = load_description("description_lstm")

        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape) 

        super(DynamicPredictionScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(DynamicPredictionScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")

        self.np_pressure_sta = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
        self.np_pressure_dyn = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(DynamicPredictionScene,self)._reset()
        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape)

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # predict frame -> stores dyn pressure in self.pressure_dynamic grid
            self._predict_step()

            # get static pressure from levelset
            computeStaticPressure(self.phi_fluid, self.flags, self.gravity, self.pressure_static, 12)

            # add dynamic pressure part evaluated by NN
            combinePressure(self.pressure_dynamic, self.pressure_static, self.pressure)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            if self.boundary>0:
                self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)
            if self.smooth_prediction:
                smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
                smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)
                smoothPressureCorner(flags=self.flags, pressure=self.pressure, boundary=self.boundary)

            if self.boundary>0:
                self.pressure.setBoundNeumann(self.boundary-1)
        else:
            # simulate frame
            super(DynamicPredictionScene,self)._solve_pressure()
            # if only simulation should happen, skip the pressure split
            if self.prediction_interval > 1:
                separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6)
                self._encode_simulation_results()

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure is the new predicted pressure value in denormalized simulation format
    def _predict_step(self):
        predicted_pressure = self.prediction_history.get_last_prediction()
        if predicted_pressure is None:
            X = self.prediction_history.get()
            # predict new pressure field
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            predicted_pressure = self.lstm.predict(x=X, batch_size=1)
            lstm_prefix_shape = (predicted_pressure.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.lstm_out_ts, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            predicted_pressure = predicted_pressure.reshape( lstm_prefix_shape + data_shape )
            # remove batch dimension
            predicted_pressure = predicted_pressure[0]

        # add to history
        self.prediction_history.add_prediction(predicted_pressure)
        # take the current (first) prediction step
        predicted_pressure = np.take(a=predicted_pressure, indices=[0], axis=0)

        # denormalize (lstm)
        predicted_pressure *= self.lstm_description["dataset"]["pressure_encoded_dynamic_normalization_factor"]
        # decode (ae)
        pressure_dyn = self.decoder.predict(x=predicted_pressure, batch_size=1)

        # denormalize (dyn ae)
        pressure_dyn  *= self.ae_description["dataset"]["pressure_dynamic_normalization_factor"]

        # transfer to manta grids
        copyArrayToGridReal(pressure_dyn, self.pressure_dynamic)

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure_stat and pressure_dyn are in denormalized simulation format
    def _encode_simulation_results(self):
        copyGridToArrayReal(self.pressure_static, self.np_pressure_sta)
        copyGridToArrayReal(self.pressure_dynamic, self.np_pressure_dyn)

        # normalize (stat/dyn ae)
        self.np_pressure_sta /= self.ae_description["dataset"]["pressure_static_normalization_factor"]
        self.np_pressure_dyn /= self.ae_description["dataset"]["pressure_dynamic_normalization_factor"]

        # encode (ae)
        encoded_pressure = self.encoder.predict(x=[self.np_pressure_sta, self.np_pressure_dyn], batch_size=1)
        # remove batch dimension
        encoded_pressure = encoded_pressure[0]
        # normalize (lstm)
        encoded_pressure /= self.lstm_description["dataset"]["pressure_encoded_dynamic_normalization_factor"]
        # add to history
        self.prediction_history.add_simulation(encoded_pressure)





#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class VelocityPredictionScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), store_divergence_history=True, name="VelocityPredictionScene", show_gui=True):
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps

        # input shape: (None, 64, 64, 64, 1)
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5")
        enc_shape = get_output_shape(self.encoder, (resolution, resolution, resolution, 3))
        self.code_layer_shape = enc_shape #self.encoder.output_shape[1:] # e.g. (1, 1, 1, 1024)

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5")
        self.lstm = keras.models.load_model(project_directory + "/lstm.h5")
        self.lstm_in_ts = self.lstm.input_shape[1] # e.g. 6
        self.lstm_out_ts = self.lstm.output_shape[1] if len(self.lstm.output_shape) > 2 else 1 # (None, 1024) or (None, 3, 1024)

        def load_description(desc_name):
            desc = None
            if os.path.isfile(project_directory + "/" + desc_name):
                with open(project_directory + "/" + desc_name, 'r') as f:
                    desc = json.load(f)
            assert desc is not None, ("model description '" + desc_name + "' not found")
            return desc

        self.ae_description = load_description("description_autoencoder")
        self.ae_settings = load_description("arguments_autoencoder.json")
        self.lstm_description = load_description("description_lstm")

        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape) 

        super(VelocityPredictionScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(VelocityPredictionScene,self)._init_grids()
        self.np_vel = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 3], order='C')

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(VelocityPredictionScene,self)._reset()
        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape)

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # predict frame
            self._predict_step()
        else:
            # simulate frame
            super(VelocityPredictionScene,self)._solve_pressure()
            # if only simulation should happen, skip the encoding
            if self.prediction_interval > 1:
                self._encode_simulation_results()

    #----------------------------------------------------------------------------------
    def _update_velocities(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            if self.store_divergence_history:
                divergence = getDivergence(self.divergence, self.vel, self.flags)
                self.divergence_history.append(divergence)
        else:
            super(VelocityPredictionScene,self)._update_velocities()

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure is the new predicted pressure value in denormalized simulation format
    def _predict_step(self):
        predicted = self.prediction_history.get_last_prediction()
        if predicted is None:
            X = self.prediction_history.get()
            # predict new pressure field
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            predicted = self.lstm.predict(x=X, batch_size=1)
            lstm_prefix_shape = (predicted.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.lstm_out_ts, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            predicted = predicted.reshape( lstm_prefix_shape + data_shape )
            # remove batch dimension
            predicted = predicted[0]

        # add to history
        self.prediction_history.add_prediction(predicted)
        # take the current (first) prediction step
        predicted = np.take(a=predicted, indices=[0], axis=0)

        # denormalize (lstm)
        predicted *= self.lstm_description["dataset"]["velocity_encoded_normalization_factor"]
        # decode (ae)
        self.np_vel = self.decoder.predict(x=predicted, batch_size=1)

        # denormalize (ae)
        self.np_vel *= self.ae_description["dataset"]["velocity_normalization_factor"]

        copyArrayToGridMAC(self.np_vel, self.vel)

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure_stat and pressure_dyn are in denormalized simulation format
    def _encode_simulation_results(self):
        copyGridToArrayMAC(self.vel, self.np_vel)

        # normalize (ae)
        self.np_vel /= self.ae_description["dataset"]["velocity_normalization_factor"]
        # encode (ae)
        encoded = self.encoder.predict(x=self.np_vel, batch_size=1)
        # remove batch dimension
        encoded = encoded[0]
        # normalize (lstm)
        encoded /= self.lstm_description["dataset"]["velocity_encoded_normalization_factor"]
        # add to history
        self.prediction_history.add_simulation(encoded)




#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class SmokePredictionScene(SmokeScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), store_divergence_history=True, pressure_and_density=False, name="SmokePredictionScene", show_gui=True):
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps
        self.pressure_and_density = pressure_and_density

        # input shape: (None, 64, 64, 64, 1)
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5")
        if self.pressure_and_density:
            enc_shape = get_output_shape(self.encoder, [(resolution, resolution, resolution, 1), (resolution, resolution, resolution, 1)] )
        else:
            enc_shape = get_output_shape(self.encoder, (resolution, resolution, resolution, 1))
        self.code_layer_shape = enc_shape #self.encoder.output_shape[1:] # e.g. (1, 1, 1, 1024)

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5")
        self.lstm = keras.models.load_model(project_directory + "/lstm.h5")
        self.lstm_in_ts = self.lstm.input_shape[1] # e.g. 6
        self.lstm_out_ts = self.lstm.output_shape[1] if len(self.lstm.output_shape) > 2 else 1 # (None, 1024) or (None, 3, 1024)

        def load_description(desc_name):
            desc = None
            if os.path.isfile(project_directory + "/" + desc_name):
                with open(project_directory + "/" + desc_name, 'r') as f:
                    desc = json.load(f)
            assert desc is not None, ("model description '" + desc_name + "' not found")
            return desc

        self.ae_description = load_description("description_autoencoder")
        self.ae_settings = load_description("arguments_autoencoder.json")
        self.lstm_description = load_description("description_lstm")

        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape) 

        super(SmokePredictionScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(SmokePredictionScene,self)._init_grids()
        self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
        if self.pressure_and_density:
            self.np_density = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(SmokePredictionScene,self)._reset()
        self.prediction_history = PredictionHistory(in_ts=self.lstm_in_ts, 
                                                    data_shape=self.code_layer_shape)

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # predict frame
            self._predict_step()
            if self.boundary>0:
                self.pressure.setBoundNeumann(self.boundary-1)
            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
        else:
            # simulate frame
            super(SmokePredictionScene,self)._solve_pressure()
            # if only simulation should happen, skip the pressure split
            if self.prediction_interval > 1:
                self._encode_simulation_results()

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure is the new predicted pressure value in denormalized simulation format
    def _predict_step(self):
        predicted_pressure = self.prediction_history.get_last_prediction()
        if predicted_pressure is None:
            X = self.prediction_history.get()
            # predict new pressure field
            input_shape = X.shape  # e.g. (1, 16, 1, 1, 1, 2048)
            X = X.reshape(*X.shape[0:2], -1)  # e.g. (1, 16, 2048)
            predicted_pressure = self.lstm.predict(x=X, batch_size=1)
            lstm_prefix_shape = (predicted_pressure.shape[0],) # parameter batch_size should be prediction.shape[0]
            lstm_prefix_shape += (self.lstm_out_ts, ) # if self.out_time_steps > 1 else () # add the out time steps if there are any
            data_shape = (*input_shape[2:],) # finally add the shape of the data, e.g. (1, 3, 1, 1, 1, 2048) with 3 out_ts and (1, 1, 1, 1, 1, 2048) with 1 out_ts
            predicted_pressure = predicted_pressure.reshape( lstm_prefix_shape + data_shape )
            # remove batch dimension
            predicted_pressure = predicted_pressure[0]

        # add to history
        self.prediction_history.add_prediction(predicted_pressure)
        # take the current (first) prediction step
        predicted_pressure = np.take(a=predicted_pressure, indices=[0], axis=0)

        # denormalize (lstm)
        if self.pressure_and_density:
            predicted_pressure *= self.lstm_description["dataset"]["pressure_density_encoded_gas_normalization_factor"]
        else:
            predicted_pressure *= self.lstm_description["dataset"]["pressure_encoded_total_gas_normalization_factor"]
        # decode (ae)
        if self.pressure_and_density:
            [self.np_pressure, _] = self.decoder.predict(x=predicted_pressure, batch_size=1)
        else:
            self.np_pressure = self.decoder.predict(x=predicted_pressure, batch_size=1)

        # denormalize (ae)
        self.np_pressure *= self.ae_description["dataset"]["pressure_normalization_factor"]

        copyArrayToGridReal(self.np_pressure, self.pressure)

    #----------------------------------------------------------------------------------
    # prediction_history contains normalized AE code layers
    # pressure_stat and pressure_dyn are in denormalized simulation format
    def _encode_simulation_results(self):
        copyGridToArrayReal(self.pressure, self.np_pressure)

        # normalize (ae)
        self.np_pressure /= (self.ae_description["dataset"]["pressure_normalization_factor"])

        if self.pressure_and_density:
            copyGridToArrayReal(self.density, self.np_density) # no normalization needed
            to_encode = [self.np_pressure, self.np_density]
        else:
            to_encode = self.np_pressure

        # encode (ae)
        encoded_pressure = self.encoder.predict(x=to_encode, batch_size=1)
        # remove batch dimension
        encoded_pressure = encoded_pressure[0]
        # normalize (lstm)
        if self.pressure_and_density:
            encoded_pressure /= self.lstm_description["dataset"]["pressure_density_encoded_gas_normalization_factor"]
        else:
            encoded_pressure /= self.lstm_description["dataset"]["pressure_encoded_total_gas_normalization_factor"]
        # add to history
        self.prediction_history.add_simulation(encoded_pressure)