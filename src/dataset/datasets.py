#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# dataset classes
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

from __future__ import print_function
import os
import re
import numpy as np
import scipy as sp
import scipy.misc
import random
# if __name__ != "__main__":
#     from . import uniio
from pathlib import Path
import glob
import math
from datetime import datetime
import json
from functools import partial
from util import filesystem
from util import console
from collections import namedtuple

def get_dataset_root(name="", path=""):
    assert name or path, ("Must provide either a path to the dataset or the name of the dataset, in which case load will try to find the dataset")
    if name:
        dataset_root = filesystem.find_directory("datasets") + "/" + name
    if path:
        dataset_root = path
    #print("Loading dataset from {}".format(dataset_root))
    assert os.path.isfile(dataset_root + "/description.json"), ("The dataset does not contain a description.json")
    return dataset_root

#====================================================================================================
class SceneList(object):
    #------------------------------------------------------------------------------------------------
    # add data in format (scene_count, scene_size, ...)
    def __init__(self, data, scene_size, version, date, name, seed, test_split=0.1):
        self.data = data
        self.scene_size = scene_size
        self.version = version
        self.date = date
        self.name = name
        self.seed = seed
        self.test_split = test_split
        self.num_scenes = len(self.data)
        self.norm_factor = 1.0
        self.norm_shift = 0.0

        self.train_range = (0, self.num_scenes - int(self.num_scenes * self.test_split))
        self.test_range = (self.train_range[1], self.num_scenes)

    #------------------------------------------------------------------------------------------------
    @property
    def train(self):
        return self.data[self.train_range[0]:self.train_range[1]]

    #------------------------------------------------------------------------------------------------
    @property
    def test(self):
        return self.data[self.test_range[0]:self.test_range[1]]

    #------------------------------------------------------------------------------------------------
    @property
    def train_length(self):
        return self.train_range[1] - self.train_range[0]

    #------------------------------------------------------------------------------------------------
    @property
    def test_length(self):
        return self.test_range[1] - self.test_range[0]

    #------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, file_name, dataset_name="", dataset_path="", test_split=0.1):
        dataset_root = get_dataset_root(dataset_name, dataset_path)
        loaded_npz = np.load(dataset_root + "/" + file_name)

        assert loaded_npz["data"] is not None, ("Can not load data!")
        data = loaded_npz["data"]

        header = None
        if "header" in loaded_npz.keys():
            header = loaded_npz["header"]
        if header is not None:
            header = header[()]

        scene_size = header.get("scene_size", 0)
        version = header.get("version", 0)
        date = header.get("date", None)
        name = header.get("name", None)
        seed = header.get("seed", 0)

        return SceneList(data, scene_size, version, date, name, seed, test_split)

    #------------------------------------------------------------------------------------------------
    def get_train_sample(self, index):
        return self.train[index]

    #------------------------------------------------------------------------------------------------
    def get_test_sample(self, index):
        return self.test[index]

    #------------------------------------------------------------------------------------------------
    def serialize(self, dataset_name="", dataset_path=""):
        dataset_root = get_dataset_root(name=dataset_name, path=dataset_path)
        file_path = dataset_root + "/" + self.name + ".npz"
        header = {
            "scene_size": self.scene_size,
            "version": self.version,
            "date": self.date,
            "name": self.name,
            "seed": self.seed
        }
        norm_factor = self.norm_factor
        norm_shift = self.norm_shift
        self.denormalize()
        np.savez_compressed(file_path, data=self.data, header=header)
        self.normalize(norm_factor, norm_shift)

        from pathlib import Path
        file_path = Path(file_path).resolve()
        print("Saved data block {} to file {}".format(self.data.shape, file_path))

    #------------------------------------------------------------------------------------------------
    def save_histogram(self, path="."):
        import matplotlib.pyplot as plt
        data = self.data.flatten()
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.hist(data, range=(-np.percentile(-data, 99.9), np.percentile(data, 99.9)), bins='auto')  # plt.hist passes it's arguments to np.histogram
        ax.set_title("Code Layer Domain")
        ax.grid()
        fig.savefig(path)   # save the figure to file
        plt.close(fig)

    #------------------------------------------------------------------------------------------------
    def normalize(self, factor, shift=0.0):
        print("Normalizing with factor {} and shift {}".format(factor, shift))
        self.norm_factor = factor
        self.norm_shift = shift
        self.data -= self.norm_shift
        self.data /= self.norm_factor

    #------------------------------------------------------------------------------------------------
    def denormalize(self):
        self.data *= self.norm_factor
        self.data += self.norm_shift
        self.norm_factor = 1.0
        self.norm_shift = 0.0

#------------------------------------------------------------------------------------------------

class DataSubSet(object):
    """ 
    should contain either train, validation or test data
    must be recreated after each epoch
    """
    def __init__(self, dataset_root, block_names, norm_factors, norm_shifts, files_per_chunk, file_indices, description, shuffle=True, augment=False):
        """
        A subset of the data. Eg. train, validation and test.
        * __files_per_chunk__: use a number __as large as possible__ to ensure iid in the train batches
        """
        assert len(file_indices) >= files_per_chunk, ("files_per_chunk ({}) larger than available files for the dataset ({})".format(files_per_chunk, len(file_indices)))
        # allow loading the dataset in chunks
        self.num_files = len(file_indices) # indices was already constructed to contain the file numbers to load
        self.num_files_per_chunk = files_per_chunk if self.num_files > files_per_chunk > 0 else self.num_files
        self.chunk_range = (0, 0)
        self.num_chunks = self.num_files // self.num_files_per_chunk
        self.num_frames = description["simulation_steps"] * self.num_files
        self.num_frames_per_chunk = self.num_frames // self.num_chunks
        self.shuffle = shuffle
        self.dataset_root = dataset_root
        self.blocks = block_names
        self.norm_factors = norm_factors
        self.norm_shifts = norm_shifts
        self.description = description
        self._file_indices = file_indices
        if self.shuffle:
            np.random.shuffle(self._file_indices)

        self.augment = augment
        if self.augment:
            if(self.description["dimension"] == 3):
                self.num_frames_per_chunk *= 4
                self.num_frames *= 4
            else:
                self.num_frames_per_chunk *= 2
                self.num_frames *= 2

        self._active_blocks = []
        self._active_block_length = -1
        self._disable_next_chunk = False

        if files_per_chunk <= 0:
            # just load all in one step and then disable next_chunk
            self.next_chunk(True, True)
            self._disable_next_chunk = True

    #------------------------------------------------------------------------------------------------
    def next_chunk(self, stop_on_overflow=False, verbose=True):
        if self.shuffle:
            indices_in_chunk = np.random.permutation(self.num_frames_per_chunk)

        if self._disable_next_chunk:
            if self.shuffle:
                for block_name in self.blocks:
                    getattr(self, block_name).permute(indices_in_chunk) 
            return False

        # delete old datablocks
        self.clear_data_blocks()

        # reset vars
        overflow = False

        # update file range
        lower_bound = self.chunk_range[1]
        upper_bound = lower_bound + self.num_files_per_chunk

        # check if upper_bound is below the maximal value, elsewise an overflow is happening
        if upper_bound > self.num_files:
            overflow = True
            self.chunk_range = (0, self.num_files_per_chunk)
            if self.shuffle:
                np.random.shuffle(self._file_indices)
            # if the user only needs one epoch, don't load start again
            if stop_on_overflow:
                return False
        else:
            self.chunk_range = (lower_bound, upper_bound)

        # load data
        for block_name in self.blocks:
            if verbose:
                print("Loading data block {}".format(block_name))
            self.add_data_block(block_name, DataBlock.from_directory(self.dataset_root + "/" + block_name, self.description, self._file_indices, self.chunk_range, verbose=verbose, augment=self.augment))
            norm_factor = self.norm_factors.get(block_name, 1.0)
            norm_shift = self.norm_shifts.get(block_name, 0.0)
            getattr(self, block_name).normalize(factor=norm_factor, shift=norm_shift, verbose=verbose)
            if self.shuffle:
                getattr(self, block_name).permute(indices_in_chunk)
            assert self.num_frames_per_chunk == getattr(self, block_name).length, ("Num frames per chunk {} does not match the block length {}".format(self.num_frames_per_chunk, getattr(self, block_name).length))
        
        return overflow == False

    #------------------------------------------------------------------------------------------------
    def add_data_block(self, name, data_block):
        """ afterwards, the data set contains a member with the specified name """
        assert self._active_block_length == data_block.length or self._active_block_length == -1, ("All blocks in a dataset must be of equal length. should be:{} is:{}".format(self._block_length, data_block.length))
        setattr(self, name, data_block)
        self._active_blocks.append(name)
        self._active_block_length = data_block.length

    #------------------------------------------------------------------------------------------------
    def clear_data_blocks(self):
        """ afterwards, the dataset clears all data blocks """
        for block_name in self._active_blocks:
            self.__delattr__(block_name)
        self._active_blocks = []
        self._active_block_length = -1

    #------------------------------------------------------------------------------------------------
    def steps_per_epoch(self, batch_size, augment=False):
        """ number of batches to train on. can be used in fit_generator """
        augmentation_multiplier = 2
        if(self.description["dimension"] == 3):
            augmentation_multiplier = 4

        augmentation_multiplier = augmentation_multiplier if augment else 1
        return int(self.num_frames / batch_size) * augmentation_multiplier

    #------------------------------------------------------------------------------------------------
    def generator(self, batch_size, inputs=None, outputs=None, augment=False, noise=False):
        """ generator for use with keras __fit_generator__ function. runs in its own thread """
        index_in_chunk = 0

        while True:
            if index_in_chunk >= self._active_block_length:
                self.next_chunk(False, False)
                index_in_chunk = 0
            if inputs == None:
                inputs = self._active_blocks
            if outputs == None:
                outputs = self._active_blocks

            # TODO: check why augmentation is also implemented in the generator....
            if augment:
                if(self.description["dimension"] == 3):
                    assert batch_size % 4 == 0, ("Batch size must be divisible by 4 when augmentation is active")
                    used_data_size = batch_size // 4
                else:
                    assert batch_size % 2 == 0, ("Batch size must be divisible by 2 when augmentation is active")
                    used_data_size = batch_size // 2

                # Input generation -> numpy ordering: zyx (!)
                X = []
                for block in inputs:
                    batch_original = self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)]
                    batch_flipped_x = np.flip(batch_original, axis=2)
                    if(self.description["dimension"] == 3):
                        batch_flipped_z = np.flip(batch_original, axis=0)
                        batch_flipped_xz = np.flip(batch_flipped_x, axis=0)
                        batch = np.concatenate([batch_original, batch_flipped_x, batch_flipped_z, batch_flipped_xz], axis=0)
                    else:
                        batch = np.concatenate([batch_original, batch_flipped_x], axis=0)

                    if noise:
                        noise_data = np.random.normal(loc=0.0, scale=0.01, size=batch.shape)
                        batch += noise_data
                        # noise_sample_ts = random.randint(0, self.time_steps-1)
                        # np_sample_ts = np.random.choice(np.array([True,False]), self.time_steps, p=[self.noise_probability,1.0-self.noise_probability]) # e.g. [True, False, False, False, ...]
                        # np_batch_mask = np.random.choice(np.array([True,False]), self.batch_size) # e.g. [True, False, False, False, ...]
                        # input_noise = np.random.uniform(low=-1.0, high=1.0, size=X[0, 0].shape) * np.std(X[random.randint(0, self.batch_size-1), noise_sample_ts]) * 0.5
                        # X[np.ix_(np_batch_mask, np_sample_ts)] += input_noise
                    X.append(batch)

                # Output generation
                Y = []
                for block in outputs:
                    batch_original = self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)]
                    batch_flipped_x = np.flip(batch_original, axis=2)
                    if(self.description["dimension"] == 3):
                        batch_flipped_z = np.flip(batch_original, axis=0)
                        batch_flipped_xz = np.flip(batch_flipped_x, axis=0)
                        batch = np.concatenate([batch_original, batch_flipped_x, batch_flipped_z, batch_flipped_xz], axis=0)
                    else:
                        batch = np.concatenate([batch_original, batch_flipped_x], axis=0)
                    Y.append(batch)
            else:
                used_data_size = batch_size
                X = [self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)] for block in inputs]
                Y = [self.__getattribute__(block)[index_in_chunk:(index_in_chunk + used_data_size)] for block in outputs]

            index_in_chunk += used_data_size

            yield X, Y
    
    #------------------------------------------------------------------------------------------------
    @property
    def length(self):
        return self.num_frames

#================================================
class DataSet(object):
    """ Data set can contain multple data blocks of different types. E.g. velocities, pressures and levelsets"""

    #------------------------------------------------------------------------------------------------
    def __init__(self):
        self.__train = None
        self.__val = None
        self.__test = None

    #------------------------------------------------------------------------------------------------
    def load(self, path="", name="", blocks=["pressure","phi_fluid", "pressure_static", "pressure_dynamic", "vel"], norm_factors={}, norm_shifts={}, files_per_batch=0, shuffle=False, validation_split=0.1, test_split=0.1, augment=False):
        """ 
        ## load 
        load a dataset from a directory containing a description.json and subdirectories with serialized grids.
        * __path__: specify the path to the dataset
        * __name__: specify the name of the dataset to load. will try to find in surrounding directories
        * __blocks__: exclude/ add grid types to load
        * __norm_factors__: a dict containing block names as keys and normalisation factors as values
        * __files_per_batch__: number of elements to load on next call
        """
        # get the directory of the dataset, and load the description
        self.dataset_root = self._get_dataset_root(path=path, name=name)
        with open(self.dataset_root + "/description.json", 'r') as f:
            description = json.load(f)
        self.description = description

        # initialize the batch loading behaviour
        self.block_names = [block for block in blocks if block in self.description["grids"]]
        self.num_files = len(glob.glob(self.dataset_root + "/" + self.description["grids"][0] + "/*.npz"))
        assert self.num_files != 0, ("No .npz were found in the dataset location")
        if self.num_files < self.description["num_scenes"]:
            print("Warning: the dataset is incomplete")
        self.files_per_batch = files_per_batch
        self.norm_factors = norm_factors
        self.norm_shifts = norm_shifts
        
        # build index list that enables shuffling of the dataset
        self.indices = np.arange(start=0, stop=self.num_files)
        self.shuffle = shuffle
        self.augment = augment
        
        # split the dataset into train, val and test
        self.train_range, self.val_range, self.test_range = self._get_ranges(self.num_files, validation_split, test_split)
        
        if self.train_range[0] == self.train_range[1]:
            print("[WARNING] Dataset was not split for training data")
        if self.val_range[0] == self.val_range[1]:
            print("[WARNING] Dataset was not split for validation data")
        if self.test_range[0] == self.test_range[1]:
            print("[WARNING] Dataset was not split for test data")

    @property
    def train(self):
        if self.train_range[0] == self.train_range[1]:
            raise RuntimeError("Dataset was not split for training data, but training data was requested")
            return None
        if self.__train is None:
            print("Loading training data")
            self.__train = DataSubSet(self.dataset_root, self.block_names, self.norm_factors, self.norm_shifts, self.files_per_batch, self.indices[self.train_range[0]:self.train_range[1]], self.description, self.shuffle, self.augment)
        return self.__train

    @property
    def val(self):
        if self.val_range[0] == self.val_range[1]:
            raise RuntimeError("Dataset was not split for validation data, but validation data was requested")
            return None
        if self.__val is None:
            print("Loading validation data")       
            self.__val = DataSubSet(self.dataset_root, self.block_names, self.norm_factors, self.norm_shifts, self.files_per_batch, self.indices[self.val_range[0]:self.val_range[1]], self.description, self.shuffle, self.augment)            
        return self.__val

    @property
    def test(self):
        if self.test_range[0] == self.test_range[1]:
            raise RuntimeError("Dataset was not split for test data, but test data was requested")
            return None
        if self.__test is None:
            print("Loading test data")       
            self.__test = DataSubSet(self.dataset_root, self.block_names, self.norm_factors, self.norm_shifts, self.files_per_batch, self.indices[self.test_range[0]:self.test_range[1]], self.description, self.shuffle, self.augment)            
        return self.__test

    #------------------------------------------------------------------------------------------------
    def _get_dataset_root(self, path="", name=""):
        assert name or path, ("Must provide either a path to the dataset or the name of the dataset, in which case load will try to find the dataset")
        if name:
            dataset_root = filesystem.find_directory("datasets") + "/" + name
        if path:
            dataset_root = path
        #print("Loading dataset from {}".format(dataset_root))
        assert os.path.isfile(dataset_root + "/description.json"), ("The dataset does not contain a description.json")
        return dataset_root

    #------------------------------------------------------------------------------------------------
    def _get_ranges(self, num_files, val_split, test_split):
        train_range = (0, num_files - int(num_files * (val_split + test_split)))
        val_range = (train_range[1], num_files - int(num_files * (test_split)))
        test_range = (val_range[1], num_files)
        return train_range, val_range, test_range

#================================================
class DataBlock(object):
    """ Holds all data of one type """

    #------------------------------------------------------------------------------------------------
    def __init__(self, data, scene_size, version, seed, normalization_factor=1.0, normalization_shift=0.0, creation_date=None):
        """ Use the 'from_data' or 'from_file' methods to create a data block instead of the default constructor """
        assert data is not None, ("Must provide data.")
        self.data = data
        self._normalization_factor = normalization_factor
        self._normalization_shift = normalization_shift
        self.test_split = 0.1
        self.scene_size = scene_size
        self.version = version
        self.creation_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if creation_date is None else creation_date
        self.seed = seed
        self.permutation = np.arange(start=0, stop=len(data))

    #------------------------------------------------------------------------------------------------
    def __getitem__(self, key):
        return self.data[self.permutation[key]]

    #------------------------------------------------------------------------------------------------
    def __call__(self):
        return self.data[self.permutation]

    #------------------------------------------------------------------------------------------------
    def permute(self, permutation):
        self.permutation = permutation

    #------------------------------------------------------------------------------------------------
    @classmethod
    def from_directory(cls, directory, description, indices, file_range=None, verbose=True, augment=False):
        """ load datablock from directory with scene files """
        # list all scenes in directory
        dirlist = glob.glob(directory + "/*.npz")

        # natural sort the filenames
        convert = lambda text: int(text) if text.isdigit() else text
        dirlist.sort(key=lambda k: [ convert(c) for c in re.split('([0-9]+)', k) ] )
        dirlist = list(np.array(dirlist)[indices]) # apply permutation
        
        # filter range to current batch
        if file_range:
            dirlist = [filename for i, filename in enumerate(dirlist) if file_range[0] <= i < file_range[1]]

        # load the data files and merge into one single numpy array
        data = []
        for filename in dirlist:
            if verbose:
                console.progress_bar(len(data), len(dirlist), length = 50)
            scene = np.load(filename)
            scene_data = scene["data"]     
            assert scene_data is not None, ("Scene with no content {}".format(filename))

            # reshape if 2D
            if(description["dimension"] == 2):
                scene_data = scene_data.reshape( scene_data.shape[:-2] + (scene_data.shape[-1],) )

            data.append([scene_data]) # unfortunately numpy.append does not append in place but creates a new array containing both
            if augment:
                flipped_x = np.flip(scene_data, axis=2)
                data.append([flipped_x])
                if(description["dimension"] == 3):
                    #print("Applying 3D data augmentation (X, Z, XZ)")
                    flipped_z = np.flip(scene_data, axis=0)
                    data.append([flipped_z])
                    data.append([np.flip(flipped_x, axis=0)])
                # else:
                #     print("Applying 2D data augmentation (X)")
                #     # already done before if/else -> flip on x
        if verbose:
            console.progress_bar(len(data), len(dirlist), length=50)
        data = np.concatenate(data, axis=1)[0, ...] # remove the axis created by the list

        # create the datablock
        return DataBlock(data=data, scene_size=description["simulation_steps"], version=description["version"], seed=description["seed"], normalization_factor=description["norm_factor"], normalization_shift=description.get("norm_shift", 0.0), creation_date=description["creation_date"])

    #------------------------------------------------------------------------------------------------
    @classmethod
    def from_file(cls, file_path):
        """ load data block from .npz file """
        file_path = str(Path(file_path).resolve())
        print('Reading data block from ' + file_path)
        # assumes the data block is not split into chunks
        data = None
        header = None
        if file_path.endswith(".npz"):
            data = np.load(file_path)
            assert data["data"] is not None, ("Can not load data!")
            if "header" in data.keys():
                header = data["header"]
            if header is not None:
                header = header[()]

        assert data is not None, ("Can't find file!")
        
        scene_size = header.get('scene_size', 1)
        version = header.get('version', 0)
        creation_date = header.get('creation_date', None)
        seed = header.get('seed', 0)
        norm_factor = header.get('norm_factor', 1.0)
        norm_shift = header.get('norm_shift', 0.0)

        return DataBlock(data = data["data"], scene_size=scene_size, version=version, creation_date=creation_date, seed=seed, normalization_factor=norm_factor, normalization_shift=norm_shift)

    #------------------------------------------------------------------------------------------------
    def serialize(self, file_path):
        """ write whole datablock to .npz file without normalization """
        path = Path(file_path)
        assert path.suffix == '.npz', ("Serialization must be to a .npz file")

        header = {
            "norm_factor": self._normalization_factor,
            "scene_size": self.scene_size,
            "version": self.version,
            "creation_date": self.creation_date,
            "seed": self.seed
        }
        np.savez_compressed(file_path, data=self.data, header=header)
        file_path = path.resolve()
        print("Saved data block {} to file {}".format(self.data.shape, file_path))

    #------------------------------------------------------------------------------------------------
    def normalize(self, shift = 0.0, factor = None, percentile = None, max = False, verbose=True):
        """ normalization of the data block. use a precomputed normalization factor instead of percentile argument to speed up normalization"""
        if max:
            if verbose:
                print("Normalizing with maximum value", end=' ')
            self._normalization_factor = np.max(self.data)
        elif factor is not None:
            if verbose:
                print("Normalizing with factor.", end=' ')
            self._normalization_factor = factor
        elif percentile is not None:
            assert False, "Percentile is still used. Report to Steffen."
            print("Normalizing with {:4.2f}th percentile.".format(percentile), end = ' ')
            self._normalization_factor = np.percentile(self.data, percentile)
        else:
            self._normalization_factor = 1.0

        if verbose:
            print(" Factor: {}".format(self._normalization_factor))

        self.data -= self._normalization_shift
        self.data /= self._normalization_factor
    
    #------------------------------------------------------------------------------------------------
    def denormalize(self):
        self.data *= self._normalization_factor
        self._normalization_factor = 1.0

    #------------------------------------------------------------------------------------------------
    @property
    def normalization_factor(self):
        """ the factor used to normalize the data block """
        return self._normalization_factor

    #------------------------------------------------------------------------------------------------
    @property
    def normalization_shift(self):
        """ the value used to shift the data block (x=x-shift) """
        return self._normalization_shift

    #------------------------------------------------------------------------------------------------
    @property
    def length(self):
        """ the number of fields in the data block """
        return self.data.shape[0]

    #------------------------------------------------------------------------------------------------
    @property
    def shape(self):
        """ shape of the elements in the data block """
        return self.data[0].shape

    #------------------------------------------------------------------------------------------------
    def mask_boundary(self, width):
        self.data = self.data[:,width:-width, width:-width, width:-width]

    #------------------------------------------------------------------------------------------------
    def print_data_properties(self):
        percentile_99th = np.percentile(self.data, 99.0)
        percentile_99_9th = np.percentile(self.data, 99.9)
        average = np.average(self.data)
        max_val = np.max(self.data)
        min_val = np.min(self.data)
        print("\tMin: {}".format(min_val))
        print("\tMax: {}".format(max_val))
        print("\tAvg: {}".format(average))
        print("\t99th Percentile: {}".format(percentile_99th))
        print("\t99.9th Percentile: {}".format(percentile_99_9th))

def read_scenes(scene_path, max_samples = -1, size = None, scene_length = 1000):
    return SceneList.from_data(scene_path = scene_path, max_samples = max_samples, size = size, scene_length = scene_length).scenes

if __name__ == "__main__":
    import argparse
    #import uniio
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--uni", default=".", help="Location of .uni files to convert")
    parser.add_argument("-o", "--npz", default=".", help="Path of the .npz which will contain the serialized data block")
    args = parser.parse_args()
    block = DataBlock.from_uni(path=args.uni)
    block.serialize(file_path=args.npz)

