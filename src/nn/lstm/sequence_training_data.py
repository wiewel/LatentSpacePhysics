#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# prepare sequence training data for training
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

import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from dataset import datasets
from util import settings

import time

from functools import reduce
from enum import Enum

#---------------------------------------------------------------------------------
class TrainingDataType(Enum):
    SplitPressure = 0
    TotalPressure = 1
    SplitPressureVAE = 2
    SplitPressureDynamic = 3
    Velocity = 4
    TotalPressureGas = 5
    PressureDensityGas = 6

#---------------------------------------------------------------------------------
class TrainingData(object):
    #--------------------------------------------
    def __init__(self, test_split = 0.1):
        settings.load("settings.json")
        self.test_split = test_split

    #--------------------------------------------
    def printData(self, file=None):
        print("Dataset dimensions:", file=file)
        print("\tScene count: {}".format(self.encoded_scene_list.num_scenes), file=file)
        print("\tTrain count: {}".format(self.encoded_scene_list.train_length), file=file)
        print("\tTest count: {}".format(self.encoded_scene_list.test_length), file=file)

    #--------------------------------------------
    def data_shape(self, flat=False):
        """ data shape of the CAE encoded frames """
        data_shape = self.encoded_scene_list.data.shape[2:]
        if flat:
            data_shape = reduce(lambda x,y: x*y, data_shape)
        return data_shape

    #--------------------------------------------
    @property
    def normalization_factor(self):
        """ the factor used to normalize the data block """
        return 1.0 if self.encoded_scene_list is None else self.encoded_scene_list.norm_factor

    #--------------------------------------------
    def get_encoded_file_name(self, training_data_type):
        if training_data_type is TrainingDataType.SplitPressure:
            out_file_name="encoded_split_pressure"
        elif training_data_type is TrainingDataType.TotalPressure:
            out_file_name="encoded_total_pressure"
        elif training_data_type is TrainingDataType.SplitPressureVAE:
            out_file_name="encoded_split_pressure_vae"
        elif training_data_type is TrainingDataType.SplitPressureDynamic:
            out_file_name="encoded_split_pressure_dynamic"
        elif training_data_type is TrainingDataType.Velocity:
            out_file_name="encoded_velocity"
        elif training_data_type is TrainingDataType.TotalPressureGas:
            out_file_name="encoded_total_pressure_gas"
        elif training_data_type is TrainingDataType.PressureDensityGas:
            out_file_name="encoded_pressure_density_gas"
        else:
            assert False, ("training_data_type '{}' is unkown".format(training_data_type))
        return out_file_name


    #--------------------------------------------
    def normalize_scene_list(self, training_data_type):
        norm_factor = settings.dataset.pressure_encoded_normalization_factor
        if training_data_type is TrainingDataType.SplitPressure:
            norm_factor = settings.dataset.pressure_encoded_normalization_factor
        if training_data_type is TrainingDataType.TotalPressure:
            norm_factor = settings.dataset.pressure_encoded_total_normalization_factor
        if training_data_type is TrainingDataType.SplitPressureDynamic:
            norm_factor = settings.dataset.pressure_encoded_dynamic_normalization_factor
        if training_data_type is TrainingDataType.SplitPressureVAE:
            norm_factor = settings.dataset.pressure_encoded_vae_normalization_factor
        if training_data_type is TrainingDataType.Velocity:
            norm_factor = settings.dataset.velocity_encoded_normalization_factor
        if training_data_type is TrainingDataType.TotalPressureGas:
            norm_factor = settings.dataset.pressure_encoded_total_gas_normalization_factor
        if training_data_type is TrainingDataType.PressureDensityGas:
            norm_factor = settings.dataset.pressure_density_encoded_gas_normalization_factor
        #print("Percentile 99.5: {}".format(np.percentile(self.encoded_scene_list.data, 99.5)))
        #print("Percentile 99.9: {}".format(np.percentile(self.encoded_scene_list.data, 99.9)))
        #print("Min: {}".format(np.amin(self.encoded_scene_list.data)))
        #print("Max: {}".format(np.amax(self.encoded_scene_list.data)))
        #print("Mean: {}".format(np.mean(self.encoded_scene_list.data)))
        assert self.normalization_factor == 1.0, ("SceneList already normalized!")
        self.encoded_scene_list.normalize(factor=norm_factor)

    #--------------------------------------------
    def load_from_file(self, training_data_type, dataset_name="", dataset_path=""):
        histogram_path = dataset_path if dataset_path else "."

        load_start_time = time.time()
        self.encoded_scene_list = datasets.SceneList.load(file_name=self.get_encoded_file_name(training_data_type)+".npz", dataset_name=dataset_name, dataset_path=dataset_path, test_split=self.test_split)
        self.encoded_scene_list.save_histogram(histogram_path + "/enc_scene_list_{}.png".format(self.encoded_scene_list.name))
        self.normalize_scene_list(training_data_type)
        self.encoded_scene_list.save_histogram(histogram_path + "/enc_scene_list_{}_normalized.png".format(self.encoded_scene_list.name))
        load_duration = time.time() - load_start_time
        print("Load Duration: {}".format(load_duration))
        self.printData()

    #--------------------------------------------
    def load_from_folder(self, autoencoder, autoencoder_desc, training_data_type, chunk_size, dataset_path, mirror_data = True):
        load_start_time = time.time()

        # Prepare dataset to be loaded from disk
        dataset = self.load_dataset(autoencoder_desc, training_data_type, chunk_size, dataset_path, mirror_data, shuffle=False)
        encoded_data = None

        # Iterate over whole dataset and transform to latent space
        while dataset.train.next_chunk(stop_on_overflow=True):
            if training_data_type is TrainingDataType.SplitPressure or training_data_type is TrainingDataType.SplitPressureVAE or training_data_type is TrainingDataType.SplitPressureDynamic:
                scene_size = dataset.train.pressure_dynamic.scene_size
                version = dataset.train.pressure_dynamic.version
                seed = dataset.train.pressure_dynamic.seed
                temp_enc_data = self.prepare_encoder_data(dataset.train.pressure_static, dataset.train.pressure_dynamic, autoencoder)
            elif training_data_type is TrainingDataType.TotalPressure or training_data_type is TrainingDataType.TotalPressureGas:
                scene_size = dataset.train.pressure.scene_size
                version = dataset.train.pressure.version
                seed = dataset.train.pressure.seed
                temp_enc_data = self.prepare_encoder_data_total(dataset.train.pressure, autoencoder)
            elif training_data_type is TrainingDataType.Velocity:
                scene_size = dataset.train.vel.scene_size
                version = dataset.train.vel.version
                seed = dataset.train.vel.seed
                temp_enc_data = self.prepare_encoder_data_total(dataset.train.vel, autoencoder)
            elif training_data_type is TrainingDataType.PressureDensityGas:
                scene_size = dataset.train.pressure.scene_size
                version = dataset.train.pressure.version
                seed = dataset.train.pressure.seed
                temp_enc_data = self.prepare_encoder_data(dataset.train.pressure, dataset.train.density, autoencoder)
            encoded_data = temp_enc_data if encoded_data is None else np.append(encoded_data, temp_enc_data, axis=0)

        assert len(encoded_data) % scene_size == 0, ("The provided data is not perfectly divisible by scene size. Something is wrong with the dataset.")
        num_scenes = dataset.description["num_scenes"]
        if mirror_data:
            num_scenes *= 4 if dataset.description["dimension"] == 3 else 2
        encoded_data = np.array(np.split(encoded_data, num_scenes, axis=0))
        out_file_name = self.get_encoded_file_name(training_data_type)

        self.create_encoded_scene(training_data_type, encoded_data, scene_size, version, seed, out_file_name)
        self.serialize(dataset_path=dataset_path)

        load_duration = time.time() - load_start_time
        print("Load Duration: {}".format(load_duration))
        self.printData()

    #--------------------------------------------
    def load_dataset(self, autoencoder_desc, training_data_type, chunk_size, dataset_path, mirror_data = True, shuffle = False):
        assert chunk_size>0, ("Dataset loading only possible with chunk_size > 0")
        print("Loading dataset from '{}'".format(dataset_path))

        # Variable Setup
        if training_data_type is TrainingDataType.SplitPressure or training_data_type is TrainingDataType.SplitPressureVAE or training_data_type is TrainingDataType.SplitPressureDynamic:
            blocks = ["pressure_static", "pressure_dynamic"]
            norm_factors = {
                "pressure_static": autoencoder_desc["dataset"]["pressure_static_normalization_factor"], #settings.dataset.pressure_static_normalization_factor,
                "pressure_dynamic": autoencoder_desc["dataset"]["pressure_dynamic_normalization_factor"] #settings.dataset.pressure_dynamic_normalization_factor
            }
        elif training_data_type is TrainingDataType.TotalPressure:
            blocks = ["pressure"]
            norm_factors= {
                "pressure": autoencoder_desc["dataset"]["pressure_normalization_factor"] #settings.dataset.pressure_normalization_factor
            }
        elif training_data_type is TrainingDataType.Velocity:
            blocks = ["vel"]
            norm_factors= {
                "vel": autoencoder_desc["dataset"]["velocity_normalization_factor"] #settings.dataset.velocity_normalization_factor
            }
        elif training_data_type is TrainingDataType.TotalPressureGas:
            blocks = ["pressure"]
            norm_factors= {
                "pressure": autoencoder_desc["dataset"]["pressure_normalization_factor"] #settings.dataset.pressure_normalization_factor
            }
        elif training_data_type is TrainingDataType.PressureDensityGas:
            blocks = ["pressure", "density"]
            norm_factors= {
                "pressure": autoencoder_desc["dataset"]["pressure_normalization_factor"]
            }

        # Load Logic
        dataset = datasets.DataSet()
        dataset.load(
            path=dataset_path,
            blocks=blocks, 
            norm_factors=norm_factors,
            files_per_batch=chunk_size,
            validation_split=0.0,
            test_split=0.0,
            augment=mirror_data,
            shuffle=shuffle)

        return dataset

    #--------------------------------------------
    def serialize(self, dataset_name="", dataset_path=""):
        self.encoded_scene_list.serialize(dataset_name=dataset_name, dataset_path=dataset_path)
        histogram_path = dataset_path if dataset_path else "."
        self.encoded_scene_list.save_histogram(histogram_path + "/enc_scene_list_{}_normalized.png".format(self.encoded_scene_list.name))

    #--------------------------------------------
    def prepare_encoder_data(self, static_datablock, dynamic_datablock, autoencoder):
        data_to_encode = [static_datablock.data, dynamic_datablock.data] # [(batch_size, 64, 64, 64, 1), (batch_size, 64, 64, 64, 1)]

        if autoencoder == None:
            encoded_data = [static_datablock.data.reshape((static_datablock.data.shape[0],-1)), dynamic_datablock.data.reshape((dynamic_datablock.data.shape[0],-1))]
        else:
            encoded_data = autoencoder.encode(data_to_encode, batch_size=settings.ae.batch_size)

        print("Encoding of {} data blocks successful!".format(encoded_data.shape))
        return encoded_data

    #--------------------------------------------
    def prepare_encoder_data_total(self, datablock, autoencoder):
        data_to_encode = datablock.data

        if autoencoder == None:
            encoded_data = data_to_encode.reshape((data_to_encode.shape[0],-1))
        else:
            encoded_data = autoencoder.encode(data_to_encode, batch_size=settings.ae.batch_size)

        print("Encoding of {} data blocks successful!".format(encoded_data.shape))
        return encoded_data

    #--------------------------------------------
    # static_datablock.scene_size, static_datablock.version, static_datablock.seed
    def create_encoded_scene(self, training_data_type, encoded_data, scene_size, version, seed, name):
        import datetime
        self.encoded_scene_list = datasets.SceneList(encoded_data, scene_size, version, str(datetime.datetime.now()), name, seed, self.test_split)
        self.normalize_scene_list(training_data_type)
