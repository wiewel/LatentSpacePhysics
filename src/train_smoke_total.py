#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# train the total pressure gas networks
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

# check and install requirements
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import util.requirements
util.requirements.fulfill()
util.requirements.init_packages()

import logging, datetime, time, shutil, pathlib, logging, math, json
from pathlib import Path
import itertools
import numpy as np
import tensorflow as tf

from nn.arch.autoencoder import Autoencoder, SplitPressureAutoencoder
from nn.arch.lstm import Lstm
from nn.lstm.error_classification import restructure_encoder_data
from nn.lstm.sequence_training_data import TrainingData, TrainingDataType

from util import evaluation
import dataset.datasets as ds
import util.plot
from util import settings
from util.filesystem import Filesystem
from util import arguments
from util import filesystem

# Arguments
#----------------------------------------------------------------------------
args = arguments.parse()

# Settings & Files
#----------------------------------------------------------------------------
settings.load("settings.json")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
fs = Filesystem(settings.project.path + "history/" + timestamp + "/")
datasets_root = "."
if args.datasets_root:
    datasets_root = args.datasets_root
else:
    datasets_root = filesystem.find_directory("datasets")
data_fs = Filesystem(datasets_root + "/" + args.dataset + "/")

def main():
    # Dataset
    #----------------------------------------------------------------------------
    dataset = ds.DataSet()
    dataset.load(path=data_fs[""], 
            blocks=["pressure"], 
            norm_factors={"pressure": settings.dataset.pressure_normalization_factor},
            files_per_batch=args.chunk_size,
            shuffle=True,
            validation_split=0.1,
            test_split=0.1)

    dataset_res = dataset.description["resolution"]

    # Autoencoder
    #----------------------------------------------------------------------------
    autoencoder = Autoencoder(input_shape=(dataset_res, dataset_res, dataset_res, 1))
    autoencoder.adam_lr_decay = 0.005
    autoencoder.adam_learning_rate = 0.001
    autoencoder.adam_epsilon = 1e-8
    autoencoder.pretrain_epochs = args.ae_pretrainepochs
    autoencoder.variational_ae = False

    # Loading from checkpoint
    if args.ae_load:
        autoencoder.load_model(settings.project.path + args.ae_load + "/autoencoder.h5")
        with open(settings.project.path + args.ae_load + "/description_autoencoder", 'r') as f:
            autoencoder_desc = json.load(f)

    pretrain_hist = None
    train_hist = None
    # Training
    if not args.ae_test:
        # training
        train_hist = autoencoder.train(dataset=dataset, epochs=args.ae_epochs, batch_size=settings.ae.batch_size, augment=True)
        # save model only if training has happened
        if (args.ae_epochs != 0):
            autoencoder.save_model(fs[""])
            # Update this value, or remove it completely from the settings file
            # settings["ae"]["code_layer_size"] = autoencoder.code_layer_size
            settings.save(fs["description_autoencoder"])
            with open(fs["ae_history.json"], "w") as f:
                json.dump(train_hist.history, f, indent=4)
            with open(fs["arguments_autoencoder.json"], 'w') as f:
                json.dump(vars(args), f, indent=4)

    # Evaluation & Plot
    if args.ae_evaluate:
        evaluation.evaluate_autoencoder(autoencoder, dataset, fs, pretrain_hist, train_hist, evaluation.QuantityType.TotalPressure)

    # At this point the description must have been stored already
    if not autoencoder_desc:
        with open(fs["description_autoencoder"], 'r') as f:
            autoencoder_desc = json.load(f)

    # LSTM
    #----------------------------------------------------------------------------
    # Generate training data for the LSTM with the current autoencoder and create LSTM training data
    # # # Load datasets
    training_data = TrainingData(test_split=0.05)
    training_data_type = TrainingDataType.TotalPressureGas

    # Should be dependent on whether the autoencoder was changed or just loaded
    if args.lstm_load_dataset:
        training_data.load_from_file(training_data_type=training_data_type, dataset_path=data_fs[""])
    else:
        training_data.load_from_folder(
            training_data_type = training_data_type,
            autoencoder_desc=autoencoder_desc,
            autoencoder=autoencoder,
            dataset_path=data_fs[""],
            chunk_size = args.chunk_size,
            mirror_data = True
        )

    # loading of scene list finished
    encoded_scene_list = training_data.encoded_scene_list

    # Create LSTM model
    data_dim = training_data.data_shape(True)
    lstm = Lstm(settings=settings, data_dimension=data_dim)

    if args.lstm_load:
        lstm.load_model(settings.project.path + args.name + "/lstm.h5")

    if args.lstm_epochs != 0:
        # train the lstm model
        lstm_history = lstm.train(
            epochs=args.lstm_epochs,
            train_scenes=encoded_scene_list.train,
            validation_split=0.1
        )
        if lstm_history is not None:
            with open(fs["eval/lstm_hist.json"], 'w') as f:
                json.dump(lstm_history.history, f, indent=4)

        # save the lstm model
        lstm.save_model(fs["lstm.h5"])
        settings.save(fs["description_lstm"])
        with open(fs["arguments_lstm.json"], 'w') as f:
            json.dump(vars(args), f, indent=4)

        # plot the history
        if lstm_history:
            lstm_history_plotter = util.plot.Plotter()
            lstm_history_plotter.plot_history(lstm_history.history)
            lstm_history_plotter.save_figures(fs["eval/"], "LSTM_History_{}_{}_{}".format(lstm.learning_rate, lstm.decay, lstm.dropout), filetype="svg")

    if args.lstm_evaluate:
        print("\n\nLSTM Evaluation")

        #avg_grad_norm, min_grad_norm, max_grad_norm = lstm.find_average_gradient_norm(encoded_scene_list.test_scenes)
        #print("\tAverage Gradient Norm: {} Min Gradient Norm: {} Max Gradient Norm: {}".format(avg_grad_norm, min_grad_norm, max_grad_norm))

        # Evaluation & Plot
        plotter = util.plot.Plotter3D()

        scn_enc_data = encoded_scene_list.test[0]
        X, Y = restructure_encoder_data(
                    data = scn_enc_data,  # [0 : lstm.time_steps + lstm.out_time_steps],
                    time_steps = lstm.time_steps,
                    out_time_steps = lstm.out_time_steps)

        X = X[0:len(X):len(X)//4]
        Y = Y[0:len(Y):len(Y)//4]

        ae_code_pred = lstm.predict(X, batch_size=1)

        # denormalize
        ae_code_pred *= training_data.normalization_factor
        Y *= training_data.normalization_factor
        #print("LSTM code prediction:\n{}".format(ae_code_pred))
        #print("Autoencoder code true:\n{}".format(Y))

        print("\n\nCode Layer Eval:")
        code_layer_eval = 0
        for x, y in zip(Y,ae_code_pred):
            x = np.squeeze(x)
            y = np.squeeze(y)
            print("\tOutput Shapes: True {} Predicted {}".format(x.shape, y.shape))
            mse_metric = ((x - y) ** 2).mean(axis=None)
            print("\tMean Squared Error: {}".format(mse_metric))
            print("\tRoot Mean Squared Error: {}".format(math.sqrt(mse_metric)))
            
            abs_dif = np.abs(np.abs(x)-np.abs(y))
            print("\tAbsDif Shape: {}".format(abs_dif.shape))
            out_list = np.array([x, y, abs_dif])
            print("\tout_list Shape: {}".format(out_list.shape))
            
            np.savetxt(fs["eval/CodeLayer_True_Pred_AbsDif_{}.txt".format(code_layer_eval)], out_list, fmt='%+7.3f')
            code_layer_eval += 1
            print("\t----------------------------------------------------------------------------------")

        # decode
        pred_y = autoencoder.decode(z=ae_code_pred, batch_size=1)
        true_y = autoencoder.decode(z=Y, batch_size=1)

        pred_y *= autoencoder_desc["dataset"]["pressure_normalization_factor"] # settings.dataset.pressure_normalization_factor
        true_y *= autoencoder_desc["dataset"]["pressure_normalization_factor"] # settings.dataset.pressure_normalization_factor

        for true, pred in zip(true_y, pred_y):
            x = true
            y = pred
            x = np.squeeze(x)
            y = np.squeeze(y)
            plotter.plot_pressure_widgets(fields={'True':x, 'Pred':y}, title="AE true and LSTM pred")

        plotter.show(block=True)
        plotter.save_figures(fs["eval"]+"/", "LSTM_Prediction")

    fs.copy(settings.project.path + args.name + "/")

# Start application
#----------------------------------------------------------------------------
if __name__=="__main__":
    main()
