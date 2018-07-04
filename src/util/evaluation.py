#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# prediction evaluation of trained networks
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

from util import plot
from util import settings
import json
import numpy as np
from enum import Enum
import math

from nn.lstm.error_classification import restructure_encoder_data

# ------------------------------------------------------------------------
# Definitions
# ------------------------------------------------------------------------
class QuantityType(Enum):
    TotalPressure = 0
    SplitPressure = 1
    Velocity = 2

class Metric(Enum):
    MSE = 0
    MAE = 1
    RMSE = 2

def calculate_metric(metric, true, pred):
    if metric == Metric.MSE:
        return ((true-pred) ** 2).mean(axis=None)
    elif metric == Metric.MAE:
        return (np.absolute(true-pred)).mean(axis=None)
    elif metric == Metric.RMSE:
        return math.sqrt(calculate_metric(Metric.MSE,true,pred))
    else:
        assert False, "Metric not implemented"

def build_metric_string():
    metric_string = ""
    for met in Metric:
        metric_string += "{:^12}".format(str(met.name))
    return metric_string

# ------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------
def evaluate_lstm(lstm, training_data, fs, display_denormalized=False):
    #avg_grad_norm, min_grad_norm, max_grad_norm = lstm.find_average_gradient_norm(encoded_scene_list.test_scenes)
    #print("\tAverage Gradient Norm: {} Min Gradient Norm: {} Max Gradient Norm: {}".format(avg_grad_norm, min_grad_norm, max_grad_norm))

    print("\t\t{:21}".format(" ") + build_metric_string() )
    
    encoded_scene_list = training_data.encoded_scene_list
    lstm_scenes_metric = np.zeros( (encoded_scene_list.test_length, len(Metric)) )

    for i, enc_scene in enumerate(encoded_scene_list.test):
        X, Y = restructure_encoder_data(
                    data = enc_scene,
                    time_steps = lstm.time_steps,
                    out_time_steps = lstm.out_time_steps)

        # predict and denormalize
        Prediction = lstm.predict(X, batch_size=40)

        if display_denormalized:
            Prediction *= training_data.normalization_factor
            Y *= training_data.normalization_factor

        scene_metric = np.zeros( (len(Metric),) )
        for metric in Metric:
            scene_metric[metric.value] = calculate_metric(metric, Y, Prediction)
        # scene_metric[Metric.MSE.value] = ((Y-Prediction) ** 2).mean(axis=None)
        # scene_metric[Metric.MAE.value] = (np.absolute(Y-Prediction)).mean(axis=None)
        # scene_metric[Metric.RMSE.value] = math.sqrt(scene_metric[Metric.MSE.value])
        print("\t\tScene {:5} Metric: {}".format(i, scene_metric))

        lstm_scenes_metric[i] = scene_metric

    # Build metrics over all scenes
    lstm_test_metric = lstm_scenes_metric.mean(axis=0)
    assert lstm_test_metric.shape[0] == len(Metric), ("Used wrong axis")
    print("\t\tTotal:{:14}{}".format("",lstm_test_metric))

    np.savetxt(fs["eval/Per_Scene_Metric.txt"],    lstm_scenes_metric, fmt='%+7.3f', header="Metrics: " + " ".join([str(met.name) for met in Metric]) + "\t\tScenes: {}".format(encoded_scene_list.test_length) )
    np.savetxt(fs["eval/Test_Dataset_Metric.txt"], lstm_test_metric,   fmt='%+7.3f', header="Metrics: " + " ".join([str(met.name) for met in Metric]) + "\t\tScenes: {}".format(encoded_scene_list.test_length) )

# ------------------------------------------------------------------------
def evaluate_autoencoder(autoencoder, dataset, fs, pretrain_hist, train_hist, quantity_type):
    plotter = plot.Plotter3D()
    example_count = 5
    print("Starting Evaluation")
    # fill quantities dict in the following (gets stored in npz file)
    quantities = {}
    if quantity_type == QuantityType.SplitPressure:
        xs_static, xs_dynamic = (dataset.test.pressure_static.data, dataset.test.pressure_dynamic.data)
        ys_static, ys_dynamic = autoencoder.predict([xs_static, xs_dynamic], batch_size=settings.ae.batch_size)
        xs_static *= dataset.test.pressure_static.normalization_factor
        ys_static *= dataset.test.pressure_static.normalization_factor
        xs_dynamic *= dataset.test.pressure_dynamic.normalization_factor
        ys_dynamic *= dataset.test.pressure_dynamic.normalization_factor
        quantities["xs_static"] = xs_static
        quantities["ys_static"] = ys_static
        quantities["xs_dynamic"] = xs_dynamic
        quantities["ys_dynamic"] = ys_dynamic
    elif quantity_type == QuantityType.TotalPressure:
        # load data into dataset
        dataset.test.next_chunk()
        xs = dataset.test.pressure.data
        ys = autoencoder.predict(xs, batch_size=settings.ae.batch_size)
        xs *= dataset.test.pressure.normalization_factor
        ys *= dataset.test.pressure.normalization_factor
        quantities["xs"] = xs
        quantities["ys"] = ys
    elif quantity_type == QuantityType.Velocity:
        assert False, "TODO"
    else:
        assert False, "Quantity type {} not implemented for evaluation".format(quantity_type)

    with open(fs["eval/predictions.npz"], 'wb') as f:
        np.savez_compressed(f, **quantities)

    if quantity_type == QuantityType.SplitPressure:
        indices_random = np.random.permutation(xs_static.shape[0])[:example_count]
        for x_static, x_dynamic, y_static, y_dynamic in zip(xs_static[indices_random], xs_dynamic[indices_random], ys_static[indices_random], ys_dynamic[indices_random]):
            x = x_static + x_dynamic
            y = y_static + y_dynamic
            x = np.squeeze(x)
            y = np.squeeze(y)
            plotter.plot_pressure_widgets(fields={'True':x, 'Pred':y}, title="autoencoder only")
    elif quantity_type == QuantityType.TotalPressure:
        indices_random = np.random.permutation(xs.shape[0])[:example_count]
        xs_step = xs[indices_random]
        xs_step = xs_step if len(xs_step) <= example_count else xs_step[0:example_count]
        ys_step = ys[indices_random]
        ys_step = ys_step if len(ys_step) <= example_count else ys_step[0:example_count]
        for x, y in zip(xs_step, ys_step):
            if len(autoencoder.input_shape) == 3:
                plotter.plot_heatmap(x, y, title="autoencoder only")
            else:
                x = np.squeeze(x)
                y = np.squeeze(y)
                plotter.plot_pressure_widgets(fields={'True':x, 'Pred':y}, title="autoencoder only")
    else:
        assert False, "Quantity type {} not implemented for evaluation".format(quantity_type)

    if pretrain_hist:
        with open(fs["eval/ae_pretrain_hist.json"], 'w') as f:
            json.dump(pretrain_hist.history, f, indent=4)

    if train_hist:
        cur_len = len(plotter._figures)
        plotter.plot_history(train_hist.history)
        for i in range(cur_len, len(plotter._figures)):
            plotter._figures[i].savefig(fs["eval/AE_History_{}.svg".format(i)], bbox_inches='tight')
        with open(fs["eval/ae_train_hist.json"], 'w') as f:
            json.dump(train_hist.history, f, indent=4)

    plotter.save_figures(path=fs["eval/"], filename="AE_EncDec")
    #plotter.show(block=True)