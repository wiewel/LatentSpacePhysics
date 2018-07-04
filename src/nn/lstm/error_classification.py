#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# evaluate error metrics and restructure sequence data
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

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime
import os, sys, inspect
import time
import math
from enum import Enum
import numpy as np

class PlotStyle(Enum):
    plot2D = 1
    plot3D = 2
    plotHeight = 3
    plotQuiver = 4

#=================================================
# Helper Functions
#=================================================

#=====================================================================================
# def computeDivergence(vel, image_dimension):
#         L2Div = 0

#         width = image_dimension[0]
#         height = image_dimension[1]

#         def getVel(x,y):
#             return vel[x + y * width]

#         # exclude outer boundary
#         for x in range(1, image_dimension[0]-1):
#             for y in range(1, image_dimension[1]-1):
#                 Div = - getVel(x,y)[0] + getVel(x+1,y)[0] - getVel(x,y)[1] + getVel(x, y+1)[1] # - vel(i,j,k).z + vel(i,j,k+1).z
#                 L2Div += Div * Div

#         return L2Div

#=====================================================================================
def restructure_image_training_data(data, time_steps, max_image_count=-1):
    image_count = len(data)
    if max_image_count <= 0:
        max_image_count = image_count
    print("Restructure Data:")
    # [100, 32, 32, 3]
    print(data.shape)
    # restructure to [samples, timeStep, features]
    # [
    #   [t-3 => [z0], t-2 => [z0], t-1 => [z0]], // TimeSteps
    #   [t-3 => [z1], t-2 => [z1], t-1 => [z1]]  // TimeSteps
    # ] // nb_Samples
    x_max = data.shape[1]
    y_max = data.shape[2]
    X_data = np.zeros((max_image_count * x_max * y_max, time_steps, data.shape[3]))
    y_data = np.zeros((max_image_count * x_max * y_max, data.shape[3]))
    curImage = 0
    for i in range(time_steps, image_count + time_steps): ## + time_steps maybe wrong here..
        curTS = i - time_steps  # if i == 3 -> start at 0

        # Traverse X
        for curX in range(x_max):
            # Traverse Y
            for curY in range(y_max):
                position = curImage * x_max * y_max
                position += curX * x_max
                position += curY
                # Traverse Time
                TS = 0
                while curTS < i:
                    X_data[position, TS] = np.array(data[curTS, curX, curY])  # => t-3, t-2, t-1
                    curTS += 1
                    TS += 1
                # save in output
                y_data[position] = np.array(data[i, curX, curY])  # => t
                curTS = i - time_steps
        if(i >= time_steps + max_image_count - 1):
            break
        curImage += 1

    print("X_data: {}".format(X_data.shape))
    print("y_data: {}".format(y_data.shape))

    # new_shape = (max_image_count *
    #              x_max * y_max, self.time_steps, data.shape[3])
    # print(new_shape)
    # X_data = np.reshape(X_data, new_shape)

    return X_data, y_data

#=====================================================================================
'''Restructure Encoder Data function.
This function takes a sorted list of coherent data as input and restructures it to
a RNN compatible format like [samples, time_steps, features].
'''
def restructure_encoder_data(data, time_steps, out_time_steps, max_sample_count=-1):
    """
    time_steps = 16
    out_time_steps = [1,2,3]
    """
    assert isinstance(data, (list, np.ndarray)), ("Argument 'data' must be of type 'list', found '{}'!".format(data.__class__.__name__))
    if isinstance(data, (list)):
        final_sample_count = len(data) - time_steps - out_time_steps
    else:
        final_sample_count = data.shape[0] - time_steps - out_time_steps
    assert final_sample_count >= 0, ("Not enough data for restructuring provided! Sample Count: {}".format(final_sample_count))
    final_sample_count += 1

    if max_sample_count > 0:
        final_sample_count = min(max_sample_count, final_sample_count)

    assert isinstance(data[0], np.ndarray), ("Content of 'data' must be of type 'numpy.ndarray'!")
    content_shape = data[0].shape

    # restructure to [samples, time_steps, features]
    # [
    #   [t-3 => [z0], t-2 => [z0], t-1 => [z0]], // TimeSteps
    #   [t-3 => [z1], t-2 => [z1], t-1 => [z1]]  // TimeSteps
    # ] // nb_Samples

    X_data = np.zeros( (final_sample_count, time_steps) + content_shape )
    #if out_time_steps > 1:
    y_data = np.zeros( (final_sample_count, out_time_steps) + content_shape )
    #else:
    #    y_data = np.zeros( (final_sample_count,) + content_shape )

    data_view = data #[::-1]

    curTS = 0
    for i in range(time_steps, final_sample_count + time_steps):
        X_data[curTS] = np.array( data_view[curTS : i] )
        #X_data[curTS] = X_data[curTS][::-1]
        # save in output
        #if out_time_steps > 1:
        y_data[curTS] = np.array( data_view[i : i + out_time_steps] ) # => current t until t+out_steps
        #else:
        #    y_data[curTS] = np.array( data_view[i] )  # => current t
        curTS += 1

    #print("Test Samples: {} {}".format(X_data.shape, y_data.shape))
    return X_data, y_data

#=====================================================================================
# create new dirs and update paths
def setup_directories(path):
    folder = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = path + folder + "/"
    os.makedirs(path, exist_ok=True)
    return path, folder

# ============================================
# Error
# ============================================
def error_classification(correct, prediction, rescale=True, file=None):
    assert correct.shape[0] == prediction.shape[0], ("Shapes do not match! correct.shape: %s prediction.shape: %s" % (correct.shape[0], prediction.shape[0]))

    # Divergence classification
    # --------------------------------------------
    #print("Divergence Classification")
    #print("\tCorrect: {}".format(computeDivergence(correct)))
    #print("\tPrediction: {}".format(computeDivergence(prediction)))

    if rescale is True:
        cor_max = np.max(np.abs(correct))
        correct = correct * 1.0 / cor_max
        prediction = prediction * 1.0 / cor_max
    # Error classification
    # --------------------------------------------
    print("Error Classification", file=file)
    # Mean Absolute Error (MAE): measures average magnitude of error -> linear
    # score, all individual differences are weighted equally
    MAE = mean_absolute_error(correct, prediction)
    print("\tMean Absolute Error: {}".format(MAE), file=file)
    # Root Mean Squared Error (RMSE): measures average magnitude of error ->
    # quadratic scoring rule, high weight to large errors, useful when large
    # errors are undesirable
    RMSE = mean_squared_error(correct, prediction) ** 0.5
    print("\tRoot Mean Squared Error: {}".format(RMSE), file=file)
    # RMSE will always be larger or equal to MAE: the greater the difference is, the greater the variance in the individual error rates
    # if RMSE == MAE: all errors of same magnitude
    ErrorVariance = RMSE - MAE
    print("\tError Variance: {}".format(ErrorVariance), file=file)
    # MAPE accuracy prediction in percent
    # MAPE = mean_absolute_percentage_error(self.test_data_Y, self.test_prediction)
    # print("\tMean Absolute Percentage Error: {}".format(MAPE), file=file)



# ============================================
# Visualization
# ============================================
def display(correct, predicted, plotstyle, image_dimension, save_path=""):
    if plotstyle == PlotStyle.plotQuiver:
        try:
            X = np.arange(image_dimension[0])
            Y = np.arange(image_dimension[1])
            X, Y = np.meshgrid(X, Y)
            fig, [ax1, ax2, ax3] = plt.subplots(1, 3, sharex=True, sharey=True)
            scale = 0.8
            _ = ax1.quiver(X, Y, correct[:, 0], correct[:, 1], pivot='tail', color='k', scale=scale)
            _ = ax2.quiver(X, Y, predicted[:, 0], predicted[:, 1], pivot='tail', color='k', scale=scale)

            ax1.set_xlim(0, image_dimension[0])
            ax1.set_ylim(0, image_dimension[1])
            ax2.set_xlim(0, image_dimension[0])
            ax2.set_ylim(0, image_dimension[1])
            ax3.set_xlim(0, image_dimension[0])
            ax3.set_ylim(0, image_dimension[1])
            ax1.autoscale(False)
            ax2.autoscale(False)
            ax3.autoscale(False)
            ax1.set_title("Correct")
            ax2.set_title("Predicted")
            ax3.set_title('Error')
            #ax1.get_xaxis().set_visible(False)
            #ax1.get_yaxis().set_visible(False)
            #ax2.get_xaxis().set_visible(False)
            #ax2.get_yaxis().set_visible(False)

            cor_max = np.max(np.abs(correct))
            correct = correct * (1.0 / cor_max)
            predicted = predicted * (1.0 / cor_max)

            # Draw Error
            ZError = (np.abs(correct[:, 0] - predicted[:, 0]) +
                      np.abs(correct[:, 1] - predicted[:, 1])) ** 2
            ZError = (np.reshape(ZError, image_dimension))  # np.flipud

            im = ax3.imshow(ZError, cmap='hot', vmin=0.0, vmax=np.max(ZError))
            plt.colorbar(im, orientation='horizontal')

            plt.show()
            if(save_path):
                fig.savefig(save_path, dpi=400)
        except Exception as e:
            print(str(e))
    else:
        print("Plotstyle not supported!")