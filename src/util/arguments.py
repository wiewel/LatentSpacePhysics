#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# argument parser for network training scripts and parameter search
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

import argparse

def parse():
    parser = argparse.ArgumentParser()
    # General
    general = parser.add_argument_group("General")
    general.add_argument("--name", required=True, help="The name for the project. Should be unique to what you are doing, projects with the same name will be overwritten")
    general.add_argument("--datasets_root", type=str, help="Optional path for the dataset. Useful if dataset is on other disk")
    general.add_argument("--dataset", required=True, help="The name or path of the dataset to load")
    general.add_argument("--chunk_size", type=int, default=0, help="The number of files loaded as chunk from the dataset")
    general.add_argument("--norm_desc", type=str, help="Optional path to a normalization description, similar to an AE description.")
    # Autoencoder
    autoencoder = parser.add_argument_group("Autoencoder")
    autoencoder.add_argument("--ae_load", type=str, help="name of the project from which to load the autoencoder")
    autoencoder.add_argument("--ae_test", action="store_true", help="Autoencoder test mode: no training") # TODO: delete me
    autoencoder.add_argument("--ae_evaluate", action="store_true", help="Autoencoder evaluation of current model")
    autoencoder.add_argument("--ae_epochs", type=int, default=5, help="Number of epochs to train the autoencoder on")
    autoencoder.add_argument("--ae_pretrainepochs", type=int, default=1, help="Number of epochs to pre train individual layers of the autoencoder")
    autoencoder.add_argument("--ae_vae", action="store_true", help="Autoencoder is VAE")
    autoencoder.add_argument("--ae_loss", type=str, default="", help="loss type used for training")
    # LSTM
    lstm = parser.add_argument_group("LSTM")
    lstm.add_argument("--lstm_load", action="store_true", help="Flag wether to load the lstm from file")
    lstm.add_argument("--lstm_load_dataset", action="store_true", help="Flag wether to load or generate and save an encoded dataset")
    lstm.add_argument("--lstm_epochs", type=int, default=50, help="Number of epochs to train the LSTM for")
    lstm.add_argument("--lstm_evaluate", type=int, default=0, help="LSTM evaluation of current model. Controls how many images are plotted.")
    args = parser.parse_args()

    del lstm
    del autoencoder
    del general
    del parser

    return args

#---------------------------------------------------------------------------------
def parse_parameter_search():
    # Parse Functions
    def hyperparameter(value):
        """ e.g. --lstm_parameter [learning_rate,random,float,[0.1,0.5],3] """
        """ e.g. --lstm_parameter [learning_rate,linear,float,[0.1,0.5],3] """
        """ e.g. --lstm_parameter [learning_rate,list,int,[0,1,2,3]] """
        value = value.replace("[","").replace("]","")
        value = value.split(',')
        result = []
        try:
            # name
            result.append(value[0])
            # search type -> hyper_parameter -> SearchType
            search_type = 0 # random
            if value[1] == "random":
                search_type = 0
            elif value[1] == "linear":
                search_type = 1
            elif value[1] == "list":
                search_type = 2
            result.append(search_type)
            # value type -> hyper_parameter -> ValueType
            val_type = 1 # float
            if value[2] == "int":
                val_type = 0
            elif value[2] == "float":
                val_type = 1
            result.append(val_type)

            # parse sequence
            if search_type == 0 or search_type == 1:
                # value range
                val_range = []
                val_range.append(float(value[3]))
                val_range.append(float(value[4]))
                result.append(val_range)
                # iterations
                result.append(int(value[5]))
            # parse list
            else:
                values = []
                for val in value[3:]:
                    values.append(float(val))
                result.append(values)
        except:
            raise argparse.ArgumentTypeError("No valid hyperparameter form given")
        return result

    parser = argparse.ArgumentParser()
    # General
    general = parser.add_argument_group("General")
    general.add_argument("--name", required=True, help="The name for the project")
    general.add_argument("--datasets_root", type=str, help="Optional path for the dataset. Usefull if dataset is on other disk")
    general.add_argument("--dataset", required=True, help="The name of the dataset to load")
    general.add_argument("--chunk_size", type=int, default=0, help="The number of files loaded as chunk from the dataset")
    # Autoencoder
    autoencoder = parser.add_argument_group("Autoencoder")
    autoencoder.add_argument("--ae_load", action="store_true", help="Flag wether to load the autoencoder from file")
    #autoencoder.add_argument("--ae_evaluate", action="store_true", help="Autoencoder evaluation of current model")
    autoencoder.add_argument("--ae_epochs", type=int, default=0, help="Number of epochs to train the autoencoder on")
    autoencoder.add_argument("--ae_pretrainepochs", type=int, default=0, help="Number of epochs for the staged pretraining of the autoencoder")
    #autoencoder.add_argument("--ae_vae", action="store_true", help="Autoencoder is VAE")
    autoencoder.add_argument("--ae_parameter", nargs="+", help="The parameters to train [name,[value range],iterations,type] e.g. --ae_parameter [learning_rate,random,float,[0.00005,0.0005],3]. Do not use spaces!", type=hyperparameter)

    # LSTM
    lstm = parser.add_argument_group("LSTM")
    lstm.add_argument("--lstm_load_dataset", action="store_true", help="Flag wether to load or generate and save an encoded dataset")
    lstm.add_argument("--lstm_epochs", type=int, default=0, help="Number of epochs to train the LSTM for")
    lstm.add_argument("--lstm_parameter", nargs="+", help="The parameters to train [name,[value range],iterations,type] e.g. --lstm_parameter [learning_rate,random,float,[0.00005,0.0005],3]. Do not use spaces!", type=hyperparameter)
    #lstm.add_argument("--lstm_evaluate", action="store_true", help="LSTM evaluation of current model")
    args = parser.parse_args()

    del lstm
    del autoencoder
    del general
    del parser

    return args