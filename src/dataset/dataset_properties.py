#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# output dataset properties (min, max, percentiles, ...) of all fields
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
from enum import Enum
import numpy as np
import json
from sys import float_info as finfo
from . import datasets as ds

## Usage:
# python -m dataset.dataset_properties --dataset path/to/datasets/small_liquid64/ --chunk_size 5

#---------------------------------------------------------------------------------
# https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJSONEncoder, self).default(obj)


#---------------------------------------------------------------------------------
class DataSetProperty(Enum):
    Min             = 0
    Max             = 1
    Mean            = 2
    StdDev          = 3
    Percentile99    = 4
    Percentile99_5  = 5
    Percentile99_9  = 6
    Histogram       = 7

export_bin_count = 1000

#---------------------------------------------------------------------------------
# https://stackoverflow.com/questions/47085662/merge-histograms-with-different-ranges
def merge_hist(a, b):
    def extract_vals(hist):
        # Recover values based on assumption 1.
        values = [[y]*x for x, y in zip(hist[0], hist[1])]
        # Return flattened list.
        return [z for s in values for z in s]

    # def extract_bin_resolution(hist):
    #     return hist[1][1] - hist[1][0]

    # def generate_num_bins(minval, maxval, bin_resolution):
    #     # Generate number of bins necessary to satisfy assumption 2
    #     return int(np.ceil((maxval - minval) / bin_resolution))

    vals = extract_vals(a) + extract_vals(b)
    num_bins = export_bin_count
    # bin_resolution = min(map(extract_bin_resolution, [a, b]))
    # num_bins = generate_num_bins(min(vals), max(vals), bin_resolution)

    return np.histogram(vals, bins=num_bins)

#---------------------------------------------------------------------------------
def evaluate_property(data_property, data):
    result = None
    if data_property is DataSetProperty.Min:
        result = np.amin(data)
    elif data_property is DataSetProperty.Max:
        result = np.amax(data)
    elif data_property is DataSetProperty.Mean:
        result = np.mean(data)
    elif data_property is DataSetProperty.StdDev:
        result = np.std(data)
    elif data_property is DataSetProperty.Percentile99:
        result = np.percentile(data, 99.0)
    elif data_property is DataSetProperty.Percentile99_5:
        result = np.percentile(data, 99.5)
    elif data_property is DataSetProperty.Percentile99_9:
        result = np.percentile(data, 99.9)
    elif data_property is DataSetProperty.Histogram:
        result = np.histogram(data, bins=export_bin_count)
    return result

#---------------------------------------------------------------------------------
# takes list of values and returns average or selected entry of list
def merge_property(data_property, data_list):
    if data_property is DataSetProperty.Min:
        data_list = np.amin(data_list)
    elif data_property is DataSetProperty.Max:
        data_list = np.amax(data_list)
    elif data_property is DataSetProperty.Histogram:
        result_data = data_list[0]
        for i in range(1, len(data_list)):
            result_data = merge_hist(result_data, data_list[i])
        data_list = result_data # num_bins = data_list[1].size - 1
    else:
        data_list = np.mean(data_list)
    return data_list

#---------------------------------------------------------------------------------
# store in json
def store_properties(properties, path, ignore_property_chunks, ignore_property):
    for grid, entry in properties.items():
        # Delete entries
        for prop in ignore_property_chunks:
            entry[prop.name]["Chunks"] = None
        for prop in ignore_property:
            entry[prop.name]["Total"] = None
            entry[prop.name]["Chunks"] = None 

        # Store histogram
        if DataSetProperty.Histogram not in ignore_property:
            import matplotlib
            matplotlib.use('Agg') # support for headless server
            import matplotlib.pyplot as plt
            hist, bin_edges = entry[DataSetProperty.Histogram.name]["Total"]
            fig, ax = plt.subplots( nrows=1, ncols=1 )
            ax.bar(bin_edges[:-1], hist, width = 1)
            ax.set_xlim([min(bin_edges), max(bin_edges)])
            ax.set_title("{} Data Histogram".format(grid.title()))
            ax.grid()
            fig.savefig(path+"Histogram_{}.svg".format(grid.title())) # save the figure to file
            plt.close(fig)

    with open(path+"properties.json", 'w') as f:
        json.dump(properties, f, indent=4, cls=NumpyJSONEncoder)
        print("Properties written to '{}'".format(path+"properties.json"))

#---------------------------------------------------------------------------------
# this function approximates the data set properties defined in DataSetProperty enum
# dataset_path should end with "/"
def evaluate_dataset_properties(dataset_path, chunk_size=50, blocks=["pressure","phi_fluid", "pressure_static", "pressure_dynamic", "vel"], ignore_property_chunks=[], ignore_property=[]):
    dataset = ds.DataSet()
    dataset.load(path=dataset_path,
            files_per_batch=chunk_size,
            shuffle=True,
            validation_split=0.0,
            test_split=0.0,
            blocks=blocks,
            augment=False)

    # initialize empty arrays for block names (pressure, phi, ...) with all desired properties
    block_properties = {}
    for block in dataset.block_names:
        data_properties = {}
        for entry in DataSetProperty:
            data_properties[entry.name] = {"Total": 0, "Chunks": []}
        block_properties[block] = data_properties

    # evaluate individual properties
    while dataset.train.next_chunk(stop_on_overflow=True):
        for block in dataset.block_names:
            data = dataset.train.__getattribute__(block)[:].flatten()
            for prop in DataSetProperty:
                if prop not in ignore_property:
                    block_properties[block][prop.name]["Chunks"].append(evaluate_property(prop, data))

    # select or average the final results
    for block in dataset.block_names:
        for prop in DataSetProperty:
            if prop not in ignore_property:
                block_properties[block][prop.name]["Total"] = merge_property(prop, block_properties[block][prop.name]["Chunks"])

    # store in json file
    store_properties(block_properties, dataset_path, ignore_property_chunks, ignore_property)


#----------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser()
    # General
    general = parser.add_argument_group("General")
    general.add_argument("--dataset", required=True, help="The path of the dataset to load")
    general.add_argument("--chunk_size", type=int, default=1, help="The number of files loaded as chunk from the dataset")
    general.add_argument("--blocks", type=str, nargs='+', default=["pressure","phi_fluid", "pressure_static", "pressure_dynamic", "vel"])
    general.add_argument("--histogram", action="store_true", help="Plot histograms of data blocks. Needs high amount of RAM.")
    args = parser.parse_args()

    del general
    del parser

    return args

# Start application
#----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    args = parse_arguments()
    assert args.chunk_size > 0, "chunk_size must be > 0"
    if not args.dataset.endswith("/") and not args.dataset.endswith("\\"):
        args.dataset += "/"
    print("Dataset Path: {}".format(args.dataset))

    ignore_properties = []
    if not args.histogram:
        ignore_properties.append(DataSetProperty.Histogram)

    evaluate_dataset_properties(dataset_path=args.dataset, chunk_size=args.chunk_size, blocks=args.blocks, ignore_property_chunks=ignore_properties, ignore_property=ignore_properties)
