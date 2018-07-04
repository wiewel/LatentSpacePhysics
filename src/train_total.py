#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# train the total pressure networks
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

from nn.arch.autoencoder import Autoencoder, Autoencoder2D
from nn.arch.lstm import Lstm
from nn.lstm.error_classification import restructure_encoder_data
from nn.lstm.sequence_training_data import TrainingData, TrainingDataType
from nn.callbacks import PlotRealsCallback
from nn.losses import tanhMSE, Loss, LossType

from util import evaluation
import dataset.datasets as ds
import util.plot
from util import settings
from util.filesystem import Filesystem
from util import arguments
from util import filesystem

from enum import Enum

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
    autoencoder = None
    autoencoder_desc = None

    #----------------------------------------------------------------------------
    if args.ae_load:
        with open(settings.project.path + args.ae_load + "/description_autoencoder", 'r') as f:
            autoencoder_desc = json.load(f)

    # Dataset
    #----------------------------------------------------------------------------
    dataset = ds.DataSet()
    dataset.load(path=data_fs[""], 
            blocks=["pressure"], 
            norm_factors={"pressure": settings.dataset.pressure_normalization_factor if not args.ae_load else autoencoder_desc["dataset"]["pressure_normalization_factor"] },
            files_per_batch=args.chunk_size,
            shuffle=True,
            validation_split=0.1,
            test_split=0.1,
            augment=False)
    dataset_res = dataset.description["resolution"]

    # Init Test Data
    dataset.test.next_chunk()
    dataset.train.next_chunk()
    dataset.val.next_chunk()

    # Autoencoder
    #----------------------------------------------------------------------------
    if args.ae_load or args.ae_epochs > 0:
        # Create AE
        if dataset.description["dimension"] == 2:
            autoencoder = Autoencoder2D(input_shape=(dataset_res, dataset_res, 1))
        else:
            autoencoder = Autoencoder(input_shape=(dataset_res, dataset_res, dataset_res, 1))
        autoencoder.adam_lr_decay = settings.ae.learning_rate_decay #0.0005
        autoencoder.adam_learning_rate = settings.ae.learning_rate #0.00015 #0.0003775 #0.0005
        autoencoder.adam_epsilon = None # 1e-8
        autoencoder.pretrain_epochs = args.ae_pretrainepochs #1
        autoencoder.variational_ae = args.ae_vae
        autoencoder.dropout = settings.ae.dropout
        autoencoder.l1_reg = settings.ae.l1_regularization
        autoencoder.l2_reg = settings.ae.l2_regularization

        # Create Loss
        loss = "mse"
        if "tanhMSE" in args.ae_loss and "_MSE" in args.ae_loss:
            loss_ratio = 0.5
            data_scale = 1.0
            if len(args.ae_loss) != len("tanhMSE_MSE"):
                try:
                    loss_ratio = float(args.ae_loss[args.ae_loss.find("_MSE")+4:])
                except:
                    loss_ratio = 0.5
                try:
                    data_scale = float(args.ae_loss[7:args.ae_loss.find("_MSE")])
                except:
                    data_scale = 1.0
            print("Using loss 'tanhMSE_MSE' with ratio {} and tanh input scaling {}".format(loss_ratio, data_scale))
            loss = Loss(
                loss_type=LossType.weighted_tanhmse_mse,
                loss_ratio=loss_ratio,
                data_input_scale=data_scale)
                # l1_reg=settings.ae.l1_regularization,
                # l2_reg=settings.ae.l2_regularization,
                # autoencoder=autoencoder)
        elif "tanhMSE" in args.ae_loss:
            loss_scale = 1.0
            if len(args.ae_loss) != len("tanhMSE"):
                loss_scale = float(args.ae_loss[7:])
            print("Using loss 'tanhMSE' with scaling {}".format(loss_scale))
            loss = tanhMSE(scale=loss_scale)
        autoencoder.set_loss(loss)

    # Loading from checkpoint
    if args.ae_load:
        autoencoder.load_model(settings.project.path + args.ae_load + "/autoencoder.h5")
        with open(settings.project.path + args.ae_load + "/description_autoencoder", 'r') as f:
            autoencoder_desc = json.load(f)

    pretrain_hist = None
    train_hist = None
    # Training
    if args.ae_epochs > 0:
        # Plot Callback
        train_eval_data = dataset.val.pressure.data[0]
        
        plot_eval_callback = PlotRealsCallback(autoencoder, train_eval_data, fs["eval/"], "Pressure_Eval_Plot")
        # training
        train_hist = autoencoder.train(dataset=dataset, epochs=args.ae_epochs, batch_size=settings.ae.batch_size, augment=True, plot_evaluation_callback=plot_eval_callback)
        # save model only if training has happened
        if (args.ae_epochs != 0):
            autoencoder.save_model(fs[""])
            # store autoencoder desc after training
            settings.save(fs["description_autoencoder"])
            # store the training results
            with open(fs["ae_history.json"], "w") as f:
                json.dump(train_hist.history, f, indent=4)
            with open(fs["arguments_autoencoder.json"], 'w') as f:
                json.dump(vars(args), f, indent=4)

    # Evaluation & Plot
    if args.ae_evaluate:
        evaluation.evaluate_autoencoder(autoencoder, dataset, fs, pretrain_hist, train_hist, evaluation.QuantityType.TotalPressure)

    # At this point the description must have been stored already, except we do not use the AE at all
    if not autoencoder_desc and not args.norm_desc:
        with open(fs["description_autoencoder"], 'r') as f:
            autoencoder_desc = json.load(f)
    elif args.norm_desc:
        with open(args.norm_desc, 'r') as f:
            autoencoder_desc = json.load(f)
            autoencoder_desc.save(fs["description_autoencoder"])

    assert autoencoder_desc, "autoencoder_desc is needed at this point. Consider using the norm_desc path"

    # LSTM
    #----------------------------------------------------------------------------
    # Generate training data for the LSTM with the current autoencoder and create LSTM training data
    # # # Load datasets
    training_data = TrainingData(test_split=0.05)
    training_data_type = TrainingDataType.TotalPressure

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

    # store in folder
    fs.copy(settings.project.path + args.name + "/")

    if args.lstm_evaluate > 0:
        print("\n\n----------------------------------------------------------------------------------")
        print("LSTM Evaluation")

        print("\tCalculating metrics on test data")
        evaluation.evaluate_lstm(lstm, training_data, fs, display_denormalized=False)

        print("\n\n----------------------------------------------------------------------------------")
        print("LSTM Example Prediction Plots")

        print("\tSelecting {} Test Scenes".format(args.lstm_evaluate))
        if encoded_scene_list.test_length < args.lstm_evaluate:
            selected_test_scenes = encoded_scene_list.test
        else:
            selected_test_scenes = encoded_scene_list.test[0:encoded_scene_list.test_length:encoded_scene_list.test_length//args.lstm_evaluate]
        print("\tSelected Scenes Shape: {}".format(selected_test_scenes.shape))
        X = []
        Y = []

        for test_scene in selected_test_scenes:
            new_x, new_y = restructure_encoder_data(data = test_scene,
                                                    time_steps = lstm.time_steps,
                                                    out_time_steps = lstm.out_time_steps)
            X.append(new_x[len(new_x)//2])
            Y.append(new_y[len(new_y)//2])

        X = np.array(X)
        Y = np.array(Y)

        # predict
        Prediction = lstm.predict(X, batch_size=1)
        # remove out ts dimension (batch, out_ts, data) -> (batch, data) [of first out_ts]
        Prediction = np.delete(Prediction, np.s_[1:], 1)
        Prediction = Prediction.reshape( (Prediction.shape[0],) + Prediction.shape[2:] )
        Y = np.delete(Y, np.s_[1:], 1)
        Y = Y.reshape( (Y.shape[0],) + Y.shape[2:] )

        # denorm
        Prediction *= training_data.normalization_factor
        Y *= training_data.normalization_factor

        # decode
        pred_y = autoencoder.decode(z=Prediction, batch_size=1)
        true_y = autoencoder.decode(z=Y, batch_size=1)

        pred_y *= autoencoder_desc["dataset"]["pressure_normalization_factor"]
        true_y *= autoencoder_desc["dataset"]["pressure_normalization_factor"]

        if dataset.description["dimension"] == 2:
            plotter = util.plot.Plotter3D()
            for true, pred in zip(true_y, pred_y):
                x = true
                y = pred
                # x = np.squeeze(x)
                # y = np.squeeze(y)
                plotter.plot_heatmap(x, y, title="AE true and LSTM pred")
        else:
            plotter = util.plot.Plotter3D()
            for true, pred in zip(true_y, pred_y):
                x = true
                y = pred
                x = np.squeeze(x)
                y = np.squeeze(y)
                plotter.plot_pressure_widgets(fields={'True':x, 'Pred':y}, title="AE true and LSTM pred")

        #plotter.show(block=True)
        plotter.save_figures(fs["eval"]+"/", "LSTM_Prediction")

    fs.copy(settings.project.path + args.name + "/")

# Start application
#----------------------------------------------------------------------------
if __name__=="__main__":
    main()
