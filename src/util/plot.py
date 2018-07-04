#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# plotter class for outputs
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

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.widgets as wgt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import itertools
from functools import partial
from .math_func import round_tuple, round_sig
from matplotlib.legend import Legend
from math import ceil
from heapq import nsmallest

import threading

class Plotter(object):
    """ convenience class to allow easy plotting from fields """
    #---------------------------------------------------------------------------------
    def __init__(self, colors='viridis'):
        self.colors = colors
        self._figures = []
        self._last_saved_index = 0
        matplotlib.rcParams.update({'font.size': 16})

    def plot_scatter(self, pos, value, title=""):
        # prepare the data
        x, y = zip(*pos) # inverse zip, using the * operator
        xi = np.linspace(min(x), max(x), len(x))
        yi = np.linspace(min(y), max(y), len(y))
        z = value
        zi = matplotlib.mlab.griddata(x, y, z, xi, yi, interp='linear')

        # create the figure
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.contour(xi, yi, zi, 15, linewidths=0.5, colors="k")
        ax.contourf(xi, yi, zi, 15, vmax=zi.max(), vmin=zi.min())
        # ax.colorbar()
        ax.scatter(x, y, marker="o", s=5, zorder=10)
        ax.xlim = (min(x), max(x))
        ax.ylim = (min(y), max(y))
        # ax.title = title

        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_multi_loss_scatter(self, histories, param_team, highlight_best_count=0, title=""):
        # e.g. histories = { (param0, param1): {loss: ..., val_loss: ..., ...}, (param0, param1): {loss: ..., val_loss: ..., ...} }
        assert len(param_team) == 2, "Two hyperparameters expected as input."

        # returns value in range [0,1], min == 0
        def rescale(x, min, max):
            return (x-min) * (1.0 / (max-min))

        x = []
        y = []
        val_loss = []
        loss = []
        for parameter, hist in histories.items():
            x.append(parameter[0])
            y.append(parameter[1])
            for metric, value in hist.items():
                if "val_loss" == metric:
                    val_loss.append(value[-1])
                elif "loss" == metric:
                    loss.append(value[-1])

        fig, ax = plt.subplots(1,1,figsize=(8,8))

        min_val_loss = min(val_loss)
        max_val_loss = max(val_loss)
        size_scale = 1000.0
        size_offset = 0.05 # size of zero shouldn't happen
        sizes = list(map(lambda x: size_scale * (1.0 - rescale(x, min_val_loss, max_val_loss) + size_offset), val_loss))
        min_loss = min(loss)
        max_loss = max(loss)
        color_scale = 10.0
        color_offset = 0.05
        colors = list(map(lambda x: color_scale * (1.0 - rescale(x, min_loss, max_loss)+ color_offset), loss))
        
        # Scatter the points, using size and color but no label
        scatter = ax.scatter(x, y, label=None,
                    c=colors, cmap='viridis',
                    s=sizes, linewidth=0,
                    alpha=0.3 if highlight_best_count > 0 else 0.5)

        # higlight the best val_loss
        if highlight_best_count > 0:
            sorted_val_loss = nsmallest(len(val_loss), enumerate(val_loss),  key=lambda x: x[1])
            for i in range(0, highlight_best_count):
                ax.scatter( x[sorted_val_loss[i][0]], y[sorted_val_loss[i][0]], label=None,
                            c=colors[sorted_val_loss[i][0]], cmap='viridis',
                            s=sizes[sorted_val_loss[i][0]], linewidth=1.0,
                            alpha=0.9)

        # rescale limits, since relim() don't work
        lim_x = (min(x),max(x))
        axis_offset_x = abs(lim_x[1] - lim_x[0]) * 0.1
        ax.set_xlim(lim_x[0]-axis_offset_x, lim_x[1]+axis_offset_x)
        lim_y = (min(y),max(y))
        axis_offset_y = abs(lim_y[1] - lim_y[0]) * 0.1
        ax.set_ylim(lim_y[0]-axis_offset_y, lim_y[1]+axis_offset_y)

        ax.axis(aspect='equal')
        ax.set_xlabel(param_team[0].name)
        ax.set_ylabel(param_team[1].name)

        plt.ticklabel_format(style='sci', axis='both', scilimits=(-4,4))

        # colorbar
        cb = fig.colorbar(scatter, ticks=[  color_offset * color_scale, # start
                                            (1.0 + color_offset) * color_scale * 0.5, # center
                                            (1.0 + color_offset) * color_scale]) # end
        cb.set_label("Loss")
        cb.ax.set_yticklabels([
            str(round_sig(min_loss, 4)), # start
            str(round_sig(min_loss + (max_loss-min_loss) * 0.5, 4)), # center
            str(round_sig(max_loss, 4))]) # end

        # Here we create a legend:
        # we'll plot empty lists with the desired size and label
        for loss in [min_val_loss, min_val_loss + (max_val_loss-min_val_loss) * 0.5, max_val_loss]:
            ax.scatter([], [], c='k', alpha=0.3, s= size_scale * (1.0 - rescale(loss, min_val_loss, max_val_loss)),
            label=str(round_sig(loss, sig=4)))
        ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Validation Loss', loc='best')

        ax.set_axisbelow(True)
        ax.grid(linewidth='0.5')

        fig.suptitle(title, fontsize=20)
        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_multi_loss_history(self, histories, highlight_best_count=0, title="", lim_x=None, lim_y=None):
        # e.g. histories = { (param0, param1): {loss: ..., val_loss: ..., ...}, (param0, param1): {loss: ..., val_loss: ..., ...} }
        loss_histories = {}
        val_loss_histories = {}
        for parameter, hist in histories.items():
            for metric, value in hist.items():
                if "loss" == metric:
                    loss_histories[parameter] = value
                if "val_loss" == metric:
                    val_loss_histories[parameter] = value

        assert len(loss_histories) == len(val_loss_histories), ("loss and val_loss must be of same length")

        fig, ax = plt.subplots(1,1,figsize=(8,8))
        colormap = plt.cm.gist_ncar

        # style definition
        loss_style = dict(
            alpha=0.15 if highlight_best_count > 0 else 0.5,
            linestyle='-'
        )
        val_loss_style = dict(
            alpha=0.15 if highlight_best_count > 0 else 0.5,
            linestyle='--'
        )

        # draw loss
        for idx, (parameter, hist) in enumerate(loss_histories.items()):
            # Setup plot color
            color = colormap( idx / len(loss_histories) )
            # Plot histories
            ax.plot(hist, color=color, label= str(round_tuple(parameter, sig=3)), **loss_style)
        # draw val loss
        for idx, (parameter, hist) in enumerate(val_loss_histories.items()):
            # Setup plot color
            color = colormap( idx / len(val_loss_histories) )
            # Plot histories
            ax.plot(hist, color=color, **val_loss_style)

        # higlight the best val_loss
        if highlight_best_count > 0:
            val_loss = val_loss_histories.items()
            sorted_val_loss = nsmallest(len(val_loss), enumerate(val_loss), key=lambda x: x[1][1][-1])
            for i in range(0, highlight_best_count):
                idx = sorted_val_loss[i][0] 
                parameter = sorted_val_loss[i][1][0]
                color = colormap( idx / len(val_loss_histories) )
                highlight_style = loss_style
                highlight_style["alpha"] = 0.9
                highlight_style["linestyle"] = "-"
                ax.plot(loss_histories[parameter], color=color, label= str(round_tuple(parameter, sig=3)), **highlight_style)
                highlight_style = val_loss_style
                highlight_style["alpha"] = 0.9
                highlight_style["linestyle"] = "--"
                ax.plot(val_loss_histories[parameter], color=color, **highlight_style)

        if lim_x is not None:
            ax.set_xlim(lim_x[0], lim_x[1])
        if lim_y is not None:
            ax.set_ylim(lim_y[0], lim_y[1])

        ax.set_axisbelow(True)
        ax.grid(linewidth='0.5')

        ax.set_ylabel("Loss")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer values on epoch axis

        # Labels to display
        handles, labels = ax.get_legend_handles_labels()
        #Create custom artists
        cust_loss_style = loss_style
        cust_loss_style["alpha"] = 1.0
        training = plt.Line2D((0,1),(0,0), color='k', **cust_loss_style)
        cust_loss_style = val_loss_style
        cust_loss_style["alpha"] = 1.0
        validation = plt.Line2D((0,1),(0,0), color='k', **cust_loss_style)
        handles = [handle for i,handle in enumerate(handles)]+[training,validation]
        labels = [label for i,label in enumerate(labels)]+['Training', 'Validation']

        ## Shrink current axis by 20%
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        #Create legend from custom artist/label lists
        ncols = ceil(len(handles) / 20) # 20 entries per column
        ncols = max(1, ncols)
        ax.legend(handles, labels, ncol=ncols, loc='center left', bbox_to_anchor=(1, 0.5))

        fig.suptitle(title, fontsize=20)
        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_history(self, history, lim_x=None, lim_y=None):
        # e.g. history = dict_keys(['loss', 'val_loss', 'val_mean_squared_error', 'val_mean_absolute_error', 'mean_squared_error', 'mean_absolute_error'])
        keys = history.keys()
        key_pairs = list()
        for A, B in itertools.product(keys,keys):
            A_split = A.split("val_", 1)
            if len(A_split) > 1 and A_split[1] == B:
                # "val_loss" and "loss" are a matching pair -> save
                key_pairs.append((A,B))

        for A, B in key_pairs:
            fig, ax = plt.subplots(1,1,figsize=(8,8))
            ax.plot(history[B])
            ax.plot(history[A])
            ax.grid()
            if lim_x is not None:
                ax.set_xlim(lim_x[0], lim_x[1])
            if lim_y is not None:
                ax.set_ylim(lim_y[0], lim_y[1])
            #plt.title("Model Loss")
            ax.set_ylabel(B.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer values on epoch axis
            ax.legend(['train', 'validation'], loc="upper left")
            self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_heatmap(self, x, y, title="", vmin=None, vmax=None):
        if not vmin:
            vmin = np.amin(x)
        if not vmax:
            vmax = np.amax(x)

        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 8))
        fig.tight_layout(pad=0.1)
        im1 = ax1.imshow(x[:,:,0], vmin=vmin, vmax=vmax, cmap=self.colors, interpolation='nearest') # Vega20c
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad = 0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_title("Reference", fontsize=20)
        ax1.set_xlim(0, x.shape[1])
        ax1.set_ylim(0, x.shape[0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        im2 = ax2.imshow(y[:,:,0], vmin=vmin, vmax=vmax, cmap=self.colors, interpolation='nearest') # vmin=0, vmax=1,
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad = 0.1)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        ax2.set_title("Predicted", fontsize=20)
        ax2.set_xlim(0, x.shape[1])
        ax2.set_ylim(0, x.shape[0])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        fig.suptitle(title, fontsize=20)
        self._figures.append(fig)
    
    #---------------------------------------------------------------------------------
    def plot_single(self, x, title="", vmin=None, vmax=None, plot_colorbar=True):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        im1 = ax1.imshow(x[:,:,0], vmin=vmin, vmax=vmax, cmap=self.colors, interpolation='nearest') # Vega20c
        if plot_colorbar:
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='5%', pad = 0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
        ax1.set_xlim(0, x.shape[1])
        ax1.set_ylim(0, x.shape[0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        fig.suptitle(title, fontsize=20)
        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_scalar_fields(self, xs, color_map='inferno', vmin=None, vmax=None, title="", plot_colorbar=True):
        """
        plot multiple scalar fields
        :param xs: A dictionary containing tuples (label, field)
        :param color_map: The color map to use for drawing the image. Typical choices are 'inferno' or 'viridis'
        :param vmin: The lowest value in the plot
        :param vmax: The highest value in the plot
        """
        assert isinstance(xs, dict), "you must feed a dict with labels and fields"
        assert len(xs) != 0, "The provided dict was of zero length"
        fig, axs = plt.subplots(1, len(xs), figsize=(8 * len(xs), 8))
        #fig.tight_layout(pad=0.1)
        
        for label, x, ax in zip(xs.keys(), xs.values(), axs):
            im = ax.imshow(x, vmin=vmin, vmax=vmax, cmap=color_map, interpolation='nearest')
            ax.set_title(label, fontsize=30)
            if plot_colorbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad = 0.1)
                fig.colorbar(im, cax=cax, orientation='vertical')
            ax.set_xlim(0, x.shape[1])
            ax.set_ylim(0, x.shape[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        fig.suptitle(title)
        self._figures.append(fig)
        return fig

    #---------------------------------------------------------------------------------
    def plot_vector_field(self, x, y, title=""):
        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 8))
        #fig.tight_layout(pad=0.1)
        im1 = ax1.quiver(x[:, :, 0], x[:, :, 1], pivot='tail', color='k', scale=10)
        ax1.set_title("Reference", fontsize=20)
        ax1.set_xlim(0, x.shape[1])
        ax1.set_ylim(0, x.shape[0])
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)   
        
        im2 = ax2.quiver(y[:, :, 0], y[:, :, 1], pivot='tail', color='k', scale=10)
        ax2.set_title("Predicted", fontsize=20)
        ax2.set_xlim(0, x.shape[1])
        ax2.set_ylim(0, x.shape[0])
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        fig.suptitle(title)        
        self._figures.append(fig)
        
    #---------------------------------------------------------------------------------
    def add_figure(fig):
        self._figures.append(fig)
    
    #---------------------------------------------------------------------------------
    def plot_graph(self, values, step=1, x_label="x", y_label="y"):
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        x = np.arange(0, len(values), step)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        
        if isinstance(values, dict):
            for key in values.keys():
                ax.plot(values[key], label=key)
        else:
            ax.plot(x, values)

        ax.legend()
        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def show(self, block=False):
        plt.show(block=block)

    #---------------------------------------------------------------------------------
    # Images are removed from the plotter object after saving them to disk. Make sure to show them beforehand if needed.
    def save_figures(self, path, filename="figure", filetype="png"):
        if not isinstance(threading.current_thread(), threading._MainThread):
            print("[WARNING] Current thread is not the main thread. Aborting save_figures call...")
            return

        # if len(self._figures) == 1:
        #     self._figures[0].savefig(path + "{}.{}".format(filename, filetype), bbox_inches='tight')
        # else:
        for fig in self._figures:
            fig.savefig(path + "{}_{}.{}".format(filename, self._last_saved_index, filetype), bbox_inches='tight')
            self._last_saved_index += 1
            plt.close(fig)
        self._figures = []

#=====================================================================================
class Plotter3D(Plotter):
    """ Plotter for 3D fields """

    #---------------------------------------------------------------------------------
    def __init__(self, colors='viridis'):
        self.colors = colors
        self._figures = []
        self._widgets = []
        self._last_saved_index = 0
        matplotlib.rcParams.update({'font.size': 16})
    
    #---------------------------------------------------------------------------------
    def plot_pressure_widgets(self, fields, title="", vmin=None, vmax=None):
        """ Plot widgets for pressure fields given in the fields dict. """
        assert isinstance(fields, dict), "you must feed a dict with labels and fields"
        assert len(fields) != 0, "The provided dict was of zero length"

        fig = plt.figure()
        widget = PressureWidget(fig=fig, fields=fields, vmin=vmin, vmax=vmax)
        self._widgets.append(widget)
        self._figures.append(fig)

    #---------------------------------------------------------------------------------
    def plot_vector_field(self, fields, title=""):
        assert isinstance(fields, dict), "you must feed a dict with labels and fields"
        assert len(fields) != 0, "The provided dict was of zero length"

        fig = plt.figure()
        widget = VelocityWidget(fig=fig, fields=fields)
        self._widgets.append(widget)
        self._figures.append(fig)

    # Should be inherited from super class
    # #---------------------------------------------------------------------------------
    # def show(self, block=False):
    #     plt.show(block=block)
        
    # #---------------------------------------------------------------------------------
    # def save_figures(self, path, filename="figure"):
    #     if len(self._figures) == 1:
    #         self._figures[0].savefig(path + "{}.png".format(filename), bbox_inches='tight')
    #     else:
    #         for index, fig in enumerate(self._figures):
    #             fig.savefig(path + "{}_{}.png".format(filename, index), bbox_inches='tight')

#=====================================================================================
class VelocityWidget(object):
    """ Widgets for 3D plots """
    #---------------------------------------------------------------------------------
    def __init__(self, fig, fields, color_map='k'):
        self.fields = fields
        self.images = {}
        self.res = list(fields.values())[0].shape[0]
        self.pos = int(self.res/2)
        self.dim = 'z'

        for i, name, field in zip(range(len(fields)), fields.keys(), fields.values()):
            ax = fig.add_axes([i * 0.45, 0.3, 0.5, 0.5])
            im = ax.quiver(field[:, :, self.pos, 0], field[:, :, self.pos, 1], pivot='mid', color=color_map)
            ax.set_title(name, fontsize=20)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.images[name] = im

        # Add colorbar to last axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad = 0.1)
        #fig.colorbar(im, cax=cax, orientation='vertical')

        posax = fig.add_axes([0.2, 0.0125, 0.65, 0.03])
        self.slider_pos = wgt.Slider(posax, 'Position', 0, self.res - 1, valinit=int(self.res/2))
        def slider_update(val):
            pos = int(self.slider_pos.val)
            for name, im in self.images.items():
                if self.dim == 'x':
                    im.set_UVC(self.fields[name][pos,:,:,1], self.fields[name][pos,:,:,2])
                if self.dim == 'y':
                    im.set_UVC(self.fields[name][:,pos,:,0], self.fields[name][:,pos,:,2])
                if self.dim == 'z':
                    im.set_UVC(self.fields[name][:,:,pos,0], self.fields[name][:,:,pos,1])
            fig.canvas.draw_idle()
        self.slider_pos.on_changed(slider_update)

        radax = fig.add_axes([0.075, 0.075, 0.1, 0.15])
        self.radio = wgt.RadioButtons(radax, ('x','y','z'), active=2)
        def radio_update(label):
            self.dim = label
            slider_update(self.pos)
        self.radio.on_clicked(radio_update)

#=====================================================================================
class PressureWidget(object):
    """ Widgets for 3D plots """
    #---------------------------------------------------------------------------------
    def __init__(self, fig, fields, vmin=None, vmax=None, color_map='inferno'):
        self.fields = fields
        self.images = {}
        self.res = 64
        self.pos = int(self.res/2)
        self.dim = 'z'

        for idx, name, field in zip(range(len(self.fields)), self.fields.keys(), self.fields.values()):
            print(field.shape)
            ax = fig.add_axes([idx * 0.45, 0.3, 0.5, 0.5])
            im = ax.imshow(field[:,:,self.pos], vmin=vmin, vmax=vmax, cmap=color_map, interpolation='nearest')
            ax.set_title(name, fontsize=30)
            ax.set_xlim(0, self.res)
            ax.set_ylim(0, self.res)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            self.images[name] = im
        
        # Add colorbar to last axis
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad = 0.1)
        fig.colorbar(im, cax=cax, orientation='vertical')

        posax = fig.add_axes([0.2, 0.0125, 0.65, 0.03])
        self.slider_pos = wgt.Slider(posax, 'Position', 0, self.res - 1, valinit=int(self.res/2))
        def slider_update(val):
            pos = int(self.slider_pos.val)
            for name, im in self.images.items():
                if self.dim == 'x':
                    im.set_data(self.fields[name][pos,:,:])
                if self.dim == 'y':
                    im.set_data(self.fields[name][:,pos,:])
                if self.dim == 'z':
                    im.set_data(self.fields[name][:,:,pos])
            fig.canvas.draw_idle()
        self.slider_pos.on_changed(slider_update)

        radax = fig.add_axes([0.075, 0.075, 0.1, 0.15])
        self.radio = wgt.RadioButtons(radax, ('x','y','z'), active=2)
        def radio_update(label):
            self.dim = label
            slider_update(self.pos)
        self.radio.on_clicked(radio_update)





#=====================================================================================
class PlotterLatentSpace(Plotter):
    """ Plotter for decoded latent space fields """

    #---------------------------------------------------------------------------------
    def __init__(self, decoder, colors='viridis'):
        self.decoder = decoder
        self.colors = colors
        self._figures = []
        self._widgets = []
        self._last_saved_index = 0
        matplotlib.rcParams.update({'font.size': 16})
    
    #---------------------------------------------------------------------------------
    def plot(self, latent_space_a, latent_space_b, title="", vmin=None, vmax=None):
        """ Plot widgets for pressure fields given in the fields dict. """
        assert len(latent_space_a) != 0, "The provided latent space a was of zero length"
        assert len(latent_space_b) != 0, "The provided latent space b was of zero length"
        assert len(latent_space_a) == len(latent_space_b), "The provided latent space a was of zero length"
        fig = plt.figure(figsize=(10,3))
        widget = LatentSpaceWidget(fig=fig, decoder=self.decoder, latent_space_a=latent_space_a, latent_space_b=latent_space_b, vmin=vmin, vmax=vmax, color_map=self.colors)
        self._widgets.append(widget)
        self._figures.append(fig)

#=====================================================================================
class LatentSpaceWidget(object):
    """ Widgets for latent space plots """
    #---------------------------------------------------------------------------------
    def __init__(self, fig, decoder, latent_space_a, latent_space_b, vmin=None, vmax=None, color_map='viridis'):
        self.decoder = decoder
        self.fig = fig
        self.latent_space_a = latent_space_a
        self.latent_space_b = latent_space_b
        self.axes = []
        self.res = 64

        self.cur_LS = None
        self.current_selection = -1

        self.latent_space_min = min(np.amin(latent_space_a), np.amin(latent_space_b))
        self.latent_space_max = max(np.amax(latent_space_a), np.amax(latent_space_b))

        # contours cmap
        self.contour_cmap = plt.cm.get_cmap(color_map)
        colors = self.contour_cmap(np.arange(self.contour_cmap.N))
        for i in range(len(colors)):
            for j in range(4):
                colors[i][j] =  min(colors[i][j] * 1.4, 1.0)
        self.contour_cmap = LinearSegmentedColormap.from_list("custom_vir", colors, N=256)

        def display_data(ax, field, label, vmin, vmax, color_map):
            im = ax.imshow(field, vmin=vmin, vmax=vmax, cmap=color_map, interpolation='nearest')
            ax.set_title(label, fontsize=15)
            ax.set_xlim(0, field.shape[1])
            ax.set_ylim(0, field.shape[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            return im

        def display_latentspace(ax, data, vmin, vmax):
            y = data
            N = len(y)
            x = range(N)
            width = 0.2
            ax.bar(x, y, width, color="blue", align='center')
            ax.set_ylim(vmin, vmax)

        labels = ["LS A", "Interpolation", "LS B"]
        data = [self.latent_space_a, self.latent_space_a, self.latent_space_b]
        for i in range(len(data)):
            ax = self.fig.add_subplot(230 + (i+1), aspect='equal') # self.fig.add_axes([i * 0.25, 0.45, 0.4, 0.4])
            ax.set(adjustable='box', aspect='equal')
            self.axes.append(ax)
            field = self.decoder.predict(data[i])[0].squeeze()
            im = display_data(ax, field, labels[i], vmin, vmax, color_map)
            ax.contour(field, levels=np.arange(vmin+(vmax-vmin)*0.025,vmax,(vmax-vmin)/10.0), cmap=self.contour_cmap)

        # Add colorbar to last axis
        divider = make_axes_locatable(self.axes[-1])
        cax = divider.append_axes('right', size='5%', pad = 0.1)
        self.fig.colorbar(im, cax=cax, orientation='vertical')

        ax = self.fig.add_subplot(212) # self.fig.add_axes([i * 0.25, 0.45, 0.4, 0.4])
        self.axes.append(ax)
        display_latentspace(ax, self.latent_space_a.flatten(), self.latent_space_min, self.latent_space_max)

        # Interpolation Slider
        posax_slider_pos = self.fig.add_axes([0.2, 0.05, 0.65, 0.03])
        self.slider_pos = wgt.Slider(posax_slider_pos, 'Interpolation', 0.0, 1.0, valinit=0.0)
        def slider_update(val):
            interpolant = self.slider_pos.val
            if self.current_selection == -1:
                self.cur_LS = self.latent_space_b * interpolant + self.latent_space_a * (1.0-interpolant)
            else:
                cur_shape = self.cur_LS.shape
                self.cur_LS = self.cur_LS.squeeze()
                self.cur_LS[self.current_selection] = interpolant
                self.cur_LS = np.reshape(self.cur_LS, cur_shape)
            field = self.decoder.predict(self.cur_LS)[0].squeeze()
            self.axes[1].clear()
            display_data(self.axes[1], field, labels[1], vmin, vmax, color_map)
            #self.images[1].set_data(field)
            self.axes[1].contour(field, levels=np.arange(vmin+(vmax-vmin)*0.025,vmax,(vmax-vmin)/10.0), cmap=self.contour_cmap)
            self.axes[-1].clear()
            display_latentspace(self.axes[-1], self.cur_LS.flatten(), self.latent_space_min, self.latent_space_max)

            self.fig.canvas.draw_idle()
        self.slider_pos.on_changed(slider_update)

        # Selection Slider
        posax = self.fig.add_axes([0.2, 0.0, 0.65, 0.03])
        self.slider_selection = wgt.Slider(posax, 'Selection', -1, self.latent_space_a.shape[-1]-1, valinit=-1, valstep=1)
        def slider_selection_update(val):
            self.current_selection = int(val)
            posax_slider_pos.clear()
            if self.current_selection == -1:
                self.slider_pos = wgt.Slider(posax_slider_pos, 'Interpolation', 0.0, 1.0, valinit=0.0)
                self.slider_pos.on_changed(slider_update)
            else:
                self.slider_pos = wgt.Slider(posax_slider_pos, 'Value', self.latent_space_min - 1.0, self.latent_space_max + 1.0, valinit= self.cur_LS.squeeze()[self.current_selection])
                self.slider_pos.on_changed(slider_update)
            self.fig.canvas.draw_idle()
        self.slider_selection.on_changed(slider_selection_update)

        self.fig.tight_layout()

#=====================================================================================
class LossPlotter(object):
    """ Plot train and validation loss """
    def __init__(self):
        self.train_losses = {}
        self.val_losses = {}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        plt.ion()
        self.train_graph, = self.ax.plot([0], [0])
        self.val_graph, = self.ax.plot([0], [0])
        #plt.pause(0.05)

    def on_loss(self, train_losses, val_losses):
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_graph.set_xdata(list(self.train_losses.keys()))
        self.train_graph.set_ydata(list(self.train_losses.values()))
        self.val_graph.set_xdata(list(self.val_losses.keys()))
        self.val_graph.set_ydata(list(self.val_losses.values()))
        plt.draw()
        #plt.pause(0.0001)
        #plt.pause(0.05)
        #plt.show(block=False)



#---------------------------------------------------------------------------------
if __name__ == "__main__":
    #----------------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_model_path", type=str, required=True)
    parser.add_argument("--latent_space_file_a", type=str, required=True)
    parser.add_argument("--latent_space_file_b", type=str, required=True)

    # Arguments
    #----------------------------------------------------------------------------
    args = parser.parse_args()

    # plotter = Plotter3D()
    # fields = {"Pred":np.random.uniform(size=(64,64,64)), "True": np.random.uniform(size=(64,64,64))}
    # plotter.plot_pressure_widgets(fields=fields)
    # plotter.show(block=True)
    # plotter = Plotter3D()
    # fields = {"Pred":np.random.uniform(size=(64,64,64,3)), "True": np.random.uniform(size=(64,64,64,3))}
    # plotter.plot_vector_field(fields=fields)
    # plotter.show(block=True)
    import keras
    plotter = PlotterLatentSpace(decoder=keras.models.load_model(args.decoder_model_path))
    latent_space_a = np.load(args.latent_space_file_a)["data"] #np.zeros(256).reshape(1,1,1,256)
    latent_space_b = np.load(args.latent_space_file_b)["data"] #np.ones(256).reshape(1,1,1,256)
    plotter.plot(latent_space_a, latent_space_b, title="Test", vmin=0.0, vmax=1.0)
    plotter.show(block=True)

