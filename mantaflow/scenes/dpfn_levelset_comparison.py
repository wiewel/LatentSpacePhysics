#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# evaluate the levelset distance of the prediction/encdec output
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

import pathlib
import numpy as np
import argparse
from random import randint, seed
import subprocess
import time
import os
import glob
import json

from util.path import make_dir, find_dir, get_uniqe_path

# Example Call
# .\build\Release\manta scenes/dpfn_levelset_comparison.py --name liquid_network_comparison -r liquid_reference -c split_liquid_25 split_liquid_mixed_enc_timeconv_dec_25 split_liquid_timeconv_dec_25 split_liquid_timeconv_dec_25_MSE total_liquid_25 -cd interval_100/
# .\build\Release\manta scenes/dpfn_levelset_comparison.py --name liquid128_network_comparison/FullRange -cd interval_5/ -r liquid128_reference -c split_liquid128 split_liquid128_1000 split_liquid128_DeepDec split_liquid128_EncDec

#----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Name of the comparison run")
parser.add_argument("-o", "--output", default="", help="directory to which the output is written")
parser.add_argument("-r", "--reference_ls", type=str, help="Name of the reference levelset")
parser.add_argument("-rn", "--reference_file_name", type=str, default="levelset_{:04d}.uni", help="Reference levelset file name")
parser.add_argument("-c", '--comparison_ls', nargs='+', type=str)
parser.add_argument("-cn", "--comparison_file_name", type=str, default="levelset_{:04d}.uni", help="Comparison levelset file name")
parser.add_argument("-f", "--max_frames", type=int, default=-1, help="Count of frames to process")
parser.add_argument("-cd", '--comparison_dir', type=str, default="")
parser.add_argument("-s", "--suffix", type=str, default="_Bench{}", help="File Suffix")
parser.add_argument("--prefix", type=str, default="", help="The prefix of the comparison_ls directories. Used to remove it from the legend of the plots.")
parser.add_argument("-b", "--benchmarks", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help="Benchmarks to check")
parser.add_argument("--replot_dir", help="Input directory where error.json file resides")
parser.add_argument("--custom_names", nargs='+', help="Custom names to replace the project name with.")
parser.add_argument("-w", "--warmup_steps", type=int, default=0, help="The first frame of prediction. Warmup of generating scene")
parser.add_argument("--height", type=int, default=8, help="height in pyplot units")
parser.add_argument("--width", type=int, default=8, help="width in pyplot units")
parser.add_argument("--differences", action="store_true", help="Plot derivatives with graph")
parser.add_argument("--skip_autoformat", action="store_true", help="Skip the autoformatting of the legend")
parser.add_argument("--image_type", type=str, default="svg", choices=["svg", "png"], help="The type of the output plot image")
parser.add_argument("--property_file_path", type=str, required=True, default="", help="The path to the properties.json file corresponding to the dataset the model was trained with.")
parser.add_argument("--y_limits", nargs='+', help="Limits for the evaluated metric plots.")
parser.add_argument("--custom_colors", nargs='+', help="Custom colors to replace the default with.")
parser.add_argument("--custom_linestyles", nargs='+', help="Custom linestyles to replace the default with.")
parser.add_argument("--legend_colors", nargs='+', help="Custom legend colors to replace the default with.")

# Arguments
#----------------------------------------------------------------------------
args = parser.parse_args()
output_path = "."
if args.output:
    output_path = str(pathlib.Path(args.output).resolve())
else:
    output_path = find_dir("predictions", 2)
predictions_dir = output_path + "/"
output_path += "/" + args.comparison_dir + args.name + "/"
#output_path = get_uniqe_path(output_path) + "/"
output_path += "levelset/"
make_dir(output_path)
print("Output path: {}".format(output_path))

with open(output_path + "arguments.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

graph_limits = args.y_limits
if graph_limits == None or len(graph_limits) != 4:
    graph_limits = [None]*4
else:
    for i, l in enumerate(graph_limits):
        graph_limits[i] = float(l)

#----------------------------------------------------------------------------
#predictions_dir = find_dir("predictions", 2) + "/"
assert os.path.exists(predictions_dir), ("The specified predictions directory does not exist or could not be found")

def evaluate_scenes(bench):
    # Dataset Property File
    #----------------------------------------------------------------------------
    dataset_properties = None
    if args.property_file_path:
        dataset_properties = load_properties(args.property_file_path)

    # Handle input directories
    #----------------------------------------------------------------------------
    reference_dir = predictions_dir + args.reference_ls + args.suffix.format(bench)
    assert os.path.exists(reference_dir), ("The specified reference directory ({}) does not exist or could not be found".format(reference_dir))
    print(reference_dir)
    reference_desc = load_description(reference_dir)
    print(reference_desc)
    reference_dir += "/uni/"
    assert os.path.exists(reference_dir), ("The specified reference directory ({}) does not exist or could not be found".format(reference_dir))

    comparison_dirs = []
    comparison_desc = []
    for comp_ls in args.comparison_ls:
        comp_dir = predictions_dir + args.comparison_dir + comp_ls + args.suffix.format(bench)
        assert os.path.exists(comp_dir), ("The specified comparison directory ({}) does not exist or could not be found".format(comp_dir))
        comparison_desc.append(load_description(comp_dir))
        print(comparison_desc[-1])
        comp_dir += "/uni/"
        print(comp_dir)
        comparison_dirs.append(comp_dir)

    # Scene
    #----------------------------------------------------------------------------
    warmup_steps = reference_desc["args"]["warmup_steps"]
    first_desc_warmup_step = comparison_desc[0]["args"]["warmup_steps"]
    for desc in comparison_desc:
        warmup_steps = desc["args"]["warmup_steps"]
        if desc["args"]["warmup_steps"] != first_desc_warmup_step:
            if(input("Warmup steps do not match. Continue? y/n") == "n"):
                return

    scene = LevelsetComparisonScene(reference_desc["scene"]["resolution"], reference_desc["scene"]["dimension"])

    # Main Loop
    #----------------------------------------------------------------------------
    # metrics store sequentially data for each comparison
    hausdorff = [None] * len(comparison_dirs)
    #hausdorff_stddev = [None] * len(comparison_dirs)
    mae = [None] * len(comparison_dirs)
    mae_stddev = [None] * len(comparison_dirs)
    mse = [None] * len(comparison_dirs)
    psnr = [None] * len(comparison_dirs)

    # loop
    frame = warmup_steps
    endFrame = len(glob.glob( reference_dir + "*.uni"))
    endFrame_compdir = len(glob.glob( comp_dir + "*.uni"))
    endFrame = min(endFrame, endFrame_compdir)

    if args.max_frames > 0:
        endFrame = min(endFrame, frame+args.max_frames)
    while frame < endFrame:
        reference_path = reference_dir + args.reference_file_name.format(frame)
        if os.path.isfile( reference_path ):
            # Store in ref grid
            scene.reference_ls.load(reference_path)

            # comparison field
            for i, comp_dir in enumerate(comparison_dirs):
                comp_path = comp_dir + args.comparison_file_name.format(frame)
                if os.path.isfile( comp_path ):
                    # Store in comp grid
                    scene.comparison_ls.load(comp_path)
                    # Calculate metrics

                    # hausdorff
                    ls_dist_bidir = levelsetDistance(scene.reference_ls, scene.comparison_ls, True)
                    if hausdorff[i] == None:
                        hausdorff[i] = [ls_dist_bidir]
                    else:
                        hausdorff[i].append(ls_dist_bidir)

                    # mae
                    mean_abs = meanAbsoluteError(scene.reference_ls, scene.comparison_ls)
                    if mae[i] == None:
                        mae[i] = [mean_abs]
                    else:
                        mae[i].append(mean_abs)

                    # mae stddev
                    absoluteErrorToGrid(scene.temp_grid, scene.reference_ls, scene.comparison_ls)
                    mae_stddev_val = standardDeviation(scene.temp_grid, mean_abs)
                    if mae_stddev[i] == None:
                        mae_stddev[i] = [mae_stddev_val]
                    else:
                        mae_stddev[i].append(mae_stddev_val)

                    # mse
                    mean_squared = meanSquaredError(scene.reference_ls, scene.comparison_ls)
                    if mse[i] == None:
                        mse[i] = [mean_squared]
                    else:
                        mse[i].append(mean_squared)

                    # psnr
                    psnr_temp = peakSignalToNoiseRatio(scene.reference_ls, scene.comparison_ls, dataset_properties["pressure"]["Min"]["Total"], dataset_properties["pressure"]["Max"]["Total"])
                    if psnr[i] == None:
                        psnr[i] = [psnr_temp]
                    else:
                        psnr[i].append(psnr_temp)
                #else:
                #    assert False, ("Comparison file ({}) not found".format(comp_path))
        frame += 1
    return hausdorff, mae, mae_stddev, mse, psnr

# Main
#----------------------------------------------------------------------------------
def main():
    if args.replot_dir:
        with open(args.replot_dir+"arguments.json", 'r') as f:
            arguments = json.load(f)
            args.comparison_ls = arguments["comparison_ls"]
        with open(args.replot_dir+"error.json", 'r') as f:
            errors = json.load(f)

        hausdorff = errors["hausdorff"]
        for i in range(len(hausdorff)):
            hausdorff[i] = hausdorff[i][:args.max_frames]
        mae = errors["mae"]
        for i in range(len(mae)):
            mae[i] = mae[i][:args.max_frames]
        mae_stddev = errors["mae_stddev"]
        for i in range(len(mae_stddev)):
            mae_stddev[i] = mae_stddev[i][:args.max_frames]
        mse = errors["mse"]
        for i in range(len(mse)):
            mse[i] = mse[i][:args.max_frames]
        psnr = errors["psnr"]
        for i in range(len(psnr)):
            psnr[i] = psnr[i][:args.max_frames]
    else:
        assert args.comparison_ls, ("'comparison_ls' must contain values")
        assert args.reference_ls, ("'reference_ls' must contain values")
        
        np_hausdorff = None
        #np_hausdorff_stddev = None
        np_mae = None
        np_mae_stddev = None
        np_mse = None
        np_psnr = None

        for bench in args.benchmarks:
            hausdorff, mae, mae_stddev, mse, psnr = evaluate_scenes(bench)
            if np_hausdorff is None:
                np_hausdorff = np.array(hausdorff)
            else:
                np_hausdorff += np.array(hausdorff)

            if np_mae is None:
                np_mae = np.array(mae)
            else:
                np_mae += np.array(mae)

            if np_mae_stddev is None:
                np_mae_stddev = np.array(mae_stddev)
            else:
                np_mae_stddev += np.array(mae_stddev)

            if np_mse is None:
                np_mse = np.array(mse)
            else:
                np_mse += np.array(mse)

            if np_psnr is None:
                np_psnr = np.array(psnr)
            else:
                np_psnr += np.array(psnr)

        nb_benchmarks = len(args.benchmarks)
        np_hausdorff /= nb_benchmarks
        np_mae /= nb_benchmarks
        np_mae_stddev /= nb_benchmarks
        np_mse /= nb_benchmarks
        np_psnr /= nb_benchmarks

        hausdorff = np_hausdorff.tolist()
        mae = np_mae.tolist()
        mae_stddev = np_mae_stddev.tolist()
        mse = np_mse.tolist()
        psnr = np_psnr.tolist()

    # Plot Results
    #----------------------------------------------------------------------------
    legend = []
    legend_names = args.comparison_ls
    if args.custom_names:
        legend_names = args.custom_names
    if not args.skip_autoformat:
        for comp_ls in legend_names:
            legend.append(r"{}".format(comp_ls.replace(args.prefix, "").replace("_", " ")))
    else:
        for comp_ls in legend_names:
            legend.append(r"{}".format(comp_ls))

    # plot hausdorff
    plot(hausdorff, legend, r"Surface distance $e_h$", path=output_path+"hausdorff." + args.image_type, first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[0], graph_limits[1]),  differences=args.differences, differences_name=r"Surface distance derivative $\frac{d e_h}{d t}$", colors=args.custom_colors, linestyles=args.custom_linestyles)
    # plot psnr
    plot(psnr, legend, r"PSNR", path=output_path+"psnr." + args.image_type, first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[2], graph_limits[3]),  differences=args.differences, differences_name=r"PSNR derivative", colors=args.custom_colors, linestyles=args.custom_linestyles)

    # Dump Data
    #----------------------------------------------------------------------------
    if not args.replot_dir:
        with open(output_path + "error.json", 'w') as f:
            errors = {
                "hausdorff": hausdorff,
                "mae": mae,
                "mae_stddev": mae_stddev,
                "mse": mse,
                "psnr": psnr
            }
            json.dump(errors, f, indent=4)

#----------------------------------------------------------------------------
class LevelsetComparisonScene(object):
    def __init__(self, resolution, dimension, name="LevelsetComparisonScene"):
        grid_size = vec3(resolution,resolution,resolution)
        if (dimension == 2):
            grid_size.z = 1

        self.solver = Solver(name=name, gridSize=grid_size, dim=dimension)
        self.solver.timestep = 0.1

        self.reference_ls = self.solver.create(LevelsetGrid, name="Reference")
        self.comparison_ls = self.solver.create(LevelsetGrid, name="Comparison")
        self.temp_grid = self.solver.create(LevelsetGrid, name="Temp")

#----------------------------------------------------------------------------
def load_description(prediction_dir):
    desc = None
    if os.path.isfile(prediction_dir + "/" + "description.json"):
        with open(prediction_dir + "/" + "description.json", 'r') as f:
            desc = json.load(f)
    assert desc is not None, ("Prediction description '" + prediction_dir + "/description.json" + "' not found")
    return desc

#----------------------------------------------------------------------------
def load_properties(property_file_path):
    properties = None
    if os.path.isfile(property_file_path):
        with open(property_file_path, 'r') as f:
            properties = json.load(f)
    assert properties is not None, ("Property file '" + property_file_path + "' not found")
    return properties

#----------------------------------------------------------------------------
def plot(metric_list, legend_list, metric_name, title="", path="", first=0, height=8, width=8, limit=None, differences=False, differences_name="", colors=[], linestyles=[]):
    from matplotlib import pyplot as plt
    import matplotlib.widgets as wgt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1,1,figsize=(width, height))
    if limit and limit[0] != None and limit[1] != None:
        print("Setting Limit: {} {}".format(limit[0], limit[1]))
        ax.set_ylim(limit[0], limit[1])
    if differences:
        ax_diff = ax.twinx()

    if linestyles == None or len(linestyles) != len(metric_list):
        linestyles = []
        for metric in metric_list:
            linestyles.append("-")

    for i, metric in enumerate(metric_list):
        x = np.arange(first, first + len(metric))
        if colors and len(colors) == len(metric_list):
            ax.plot(x, metric, color=colors[i], linestyle=linestyles[i].replace(" ", ""))
        else:
            ax.plot(x, metric, linestyle=linestyles[i].replace(" ", ""))
        if differences:
            diff = np.gradient(metric)
            ax_diff.plot(x, diff, linestyle='dotted')

    ax.grid()

    plt.title(title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel(r"Time step $t$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer values on epoch axis
    ax.legend(legend_list, loc="upper left")

    leg = ax.get_legend()
    if args.legend_colors and len(args.legend_colors) == len(leg.legendHandles):
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i].set_color(args.legend_colors[i])

    if differences:
        ax_diff.set_ylabel(differences_name)
        diff_legend_list = [r"$\frac{d e_h}{d t}$   " + l for l in legend_list]
        ax_diff.legend(diff_legend_list, loc="lower right")
    if path:
        fig.savefig(path, bbox_inches='tight')

#----------------------------------------------------------------------------------
main()