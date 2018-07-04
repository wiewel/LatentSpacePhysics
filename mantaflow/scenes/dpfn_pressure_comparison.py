#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# evaluate the pressure similarity of the prediction/encdec output
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
import scipy.misc

from util.path import make_dir, find_dir, get_uniqe_path

# Example Call
# .\build\Release\manta scenes/dpfn_pressure_comparison.py --name liquid_network_comparison -r liquid_reference -c split_liquid_25 split_liquid_mixed_enc_timeconv_dec_25 split_liquid_timeconv_dec_25 split_liquid_timeconv_dec_25_MSE total_liquid_25 -cd interval_100/
# .\build\Release\manta scenes/dpfn_pressure_comparison.py --name liquid128_network_comparison/FullRange -r liquid128_reference -c split_liquid128 split_liquid128_1000 split_liquid128_DeepDec split_liquid128_EncDec -cd interval_5/ 

#----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Name of the comparison run")
parser.add_argument("-o", "--output", default="", help="directory to which the output is written")
parser.add_argument("-r", "--reference", type=str, required=True, help="Name of the reference field")
parser.add_argument("-rn", "--reference_file_name", type=str, default="pressure_{:04d}.uni", help="Reference file name")
parser.add_argument("-c", '--comparison', nargs='+', type=str, required=True)
parser.add_argument("-cn", "--comparison_file_name", type=str, default="pressure_{:04d}.uni", help="Comparison file name")
parser.add_argument("-f", "--max_frames", type=int, default=0, help="Count of frames to process")
parser.add_argument("-cd", '--comparison_dir', type=str, default="")
parser.add_argument("-s", "--suffix", type=str, default="_Bench{}", help="File Suffix")
parser.add_argument("--prefix", type=str, default="", help="The prefix of the comparison_ls directories. Used to remove it from the legend of the plots.")
parser.add_argument("-b", "--benchmarks", type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help="Benchmarks to check")
parser.add_argument("--custom_names", nargs='+', help="Custom names to replace the project name with.")
parser.add_argument("-w", "--warmup_steps", type=int, default=0, help="The first frame of prediction. Warmup of generating scene")
parser.add_argument("--height", type=int, default=8, help="height in pyplot units")
parser.add_argument("--width", type=int, default=8, help="width in pyplot units")
parser.add_argument("--property_file_path", type=str, required=True, default="", help="The path to the properties.json file corresponding to the dataset the model was trained with.")
parser.add_argument("--y_limits", nargs='+', help="Limits for the evaluated metric plots.")
parser.add_argument("--output_sequence", action="store_true")
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
output_path += "/" + args.comparison_dir + args.name + "/"
#output_path = get_uniqe_path(output_path) + "/"
output_path += "pressure/"
make_dir(output_path)
print("Output path: {}".format(output_path))

with open(output_path + "arguments.json", 'w') as f:
    json.dump(vars(args), f, indent=4)

graph_limits = args.y_limits
if graph_limits == None or len(graph_limits) != 8:
    graph_limits = [None]*8
else:
    for i, l in enumerate(graph_limits):
        graph_limits[i] = float(l)

#----------------------------------------------------------------------------
predictions_dir = find_dir("predictions", 2) + "/"
assert os.path.exists(predictions_dir), ("The specified predictions directory does not exist or could not be found")

def evaluate_scenes(bench):
    # Dataset Property File
    #----------------------------------------------------------------------------
    dataset_properties = None
    if args.property_file_path:
        dataset_properties = load_properties(args.property_file_path)

    # Handle input directories
    #----------------------------------------------------------------------------
    reference_dir = predictions_dir + args.reference + args.suffix.format(bench)
    assert os.path.exists(reference_dir), ("The specified reference directory ({}) does not exist or could not be found".format(reference_dir))
    print(reference_dir)
    reference_desc = load_description(reference_dir)
    print(reference_desc)
    reference_dir += "/uni/"
    assert os.path.exists(reference_dir), ("The specified reference directory ({}) does not exist or could not be found".format(reference_dir))

    comparison_dirs = []
    comparison_desc = []
    for comp in args.comparison:
        comp_dir = predictions_dir + args.comparison_dir + comp + args.suffix.format(bench)
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

    scene = PressureComparisonScene(reference_desc["scene"]["resolution"], reference_desc["scene"]["dimension"])

    np_comp = np.empty(shape=[1, reference_desc["scene"]["resolution"], reference_desc["scene"]["resolution"], reference_desc["scene"]["resolution"], 1], order='C')
    np_comp_grad = np.empty(shape=[1, reference_desc["scene"]["resolution"], reference_desc["scene"]["resolution"], reference_desc["scene"]["resolution"], 3], order='C')
    np_comp_grad_min = 9999999999999999
    np_comp_grad_max = -999999999999999

    # Main Loop
    #----------------------------------------------------------------------------
    # metrics store sequentially data for each comparison
    mae = [None] * len(comparison_dirs)
    mse = [None] * len(comparison_dirs)
    psnr_grad = [None] * len(comparison_dirs)
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
            scene.reference.load(reference_path)
            getGradientGrid(scene.reference, scene.pressure_gradient_ref)

            # comparison field
            for i, comp_dir in enumerate(comparison_dirs):
                comp_path = comp_dir + args.comparison_file_name.format(frame)
                if os.path.isfile( comp_path ):
                    # Store in comp grid
                    scene.comparison.load(comp_path)
                    getGradientGrid(scene.comparison, scene.pressure_gradient_comp)

                    # Store Example Images of Predictions
                    if args.output_sequence:
                        pressure_example_dir = output_path + args.comparison[i] + "/bench_{}/".format(bench)
                        if not os.path.exists(pressure_example_dir):
                            make_dir(pressure_example_dir)
                        # Pressure Storage
                        copyGridToArrayReal(scene.comparison, np_comp)
                        np_selected_slice = np.squeeze(np_comp[0,reference_desc["scene"]["resolution"] // 2])[::-1, ::-1]
                        #print("Comp Shape: {} -> {}".format(np_comp.shape, np_selected_slice.shape))
                        scipy.misc.toimage(np_selected_slice, cmin=-0.5, cmax=1.5).save(pressure_example_dir+"pressure_{}.png".format(frame))
                        # Gradient Storage
                        copyGridToArrayVec3_New(scene.pressure_gradient_comp, np_comp_grad)
                        np_selected_slice = np.squeeze(np_comp_grad[0,reference_desc["scene"]["resolution"] // 2])[::-1, ::-1]
                        np_comp_grad_min = min(np_comp_grad_min, np_selected_slice.min())
                        np_comp_grad_max = max(np_comp_grad_max, np_selected_slice.max())
                        print("Comp Shape: {} -> {}; {} {}".format(np_comp_grad.shape, np_selected_slice.shape, np_comp_grad_min, np_comp_grad_max))
                        scipy.misc.toimage(np_selected_slice, cmin=-0.2, cmax=0.15).save(pressure_example_dir+"pressure_grad_{}.png".format(frame))

                    # Calculate metrics

                    # mae
                    mean_abs = meanAbsoluteError(scene.reference, scene.comparison)
                    if mae[i] == None:
                        mae[i] = [mean_abs]
                    else:
                        mae[i].append(mean_abs)

                    # mse
                    mean_squared = meanSquaredError(scene.reference, scene.comparison)
                    if mse[i] == None:
                        mse[i] = [mean_squared]
                    else:
                        mse[i].append(mean_squared)

                    # psnr_grad
                    max_pressure_grad_ref = dataset_properties["pressure"]["Max"]["Total"] - dataset_properties["pressure"]["Min"]["Total"]
                    psnr_grad_temp = peakSignalToNoiseRatioVec3(scene.pressure_gradient_ref, scene.pressure_gradient_comp, 0.0, max_pressure_grad_ref)
                    if psnr_grad[i] == None:
                        psnr_grad[i] = [psnr_grad_temp]
                    else:
                        psnr_grad[i].append(psnr_grad_temp)

                    # psnr
                    psnr_temp = peakSignalToNoiseRatio(scene.reference, scene.comparison, dataset_properties["pressure"]["Min"]["Total"], dataset_properties["pressure"]["Max"]["Total"])
                    if psnr[i] == None:
                        psnr[i] = [psnr_temp]
                    else:
                        psnr[i].append(psnr_temp)
                #else:
                #    assert False, ("Comparison file ({}) not found".format(comp_path))
        frame += 1
    return mae, mse, psnr_grad, psnr

# Main
#----------------------------------------------------------------------------------
def main():
    np_mae = None
    np_mse = None
    np_psnr_grad = None
    np_psnr = None

    for bench in args.benchmarks:
        mae, mse, psnr_grad, psnr = evaluate_scenes(bench)

        if np_mae is None:
            np_mae = np.array(mae)
        else:
            np_mae += np.array(mae)

        if np_mse is None:
            np_mse = np.array(mse)
        else:
            np_mse += np.array(mse)

        if np_psnr_grad is None:
            np_psnr_grad = np.array(psnr_grad)
        else:
            np_psnr_grad += np.array(psnr_grad)

        if np_psnr is None:
            np_psnr = np.array(psnr)
        else:
            np_psnr += np.array(psnr)

        if args.output_sequence:
            for co in args.comparison:
                for img in ["pressure", "pressure_grad"]:
                    comp_bench_dir = output_path + co + "/bench_{}/".format(bench)
                    if os.path.exists(comp_bench_dir):
                        video_name = comp_bench_dir + img + ".mp4"
                        subprocess.call(['ffmpeg',
                            '-r', '30',
                            '-f', 'image2',
                            '-start_number', str(args.warmup_steps),
                            '-i', comp_bench_dir + img + "_%d.png",
                            '-vcodec', 'libx264',
                            '-crf', '18',
                            '-pix_fmt', 'yuv420p',
                            '-y',
                            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                            video_name])

    nb_benchmarks = len(args.benchmarks)
    np_mae /= nb_benchmarks
    np_mse /= nb_benchmarks
    np_psnr_grad /= nb_benchmarks
    np_psnr /= nb_benchmarks
    
    mae = np_mae.tolist()
    mse = np_mse.tolist()
    psnr_grad = np_psnr_grad.tolist()
    psnr = np_psnr.tolist()

    # Plot Results
    #----------------------------------------------------------------------------
    legend = []
    legend_names = args.comparison
    if args.custom_names:
        legend_names = args.custom_names
    for comp in legend_names:
        legend.append(comp.replace(args.prefix, "").replace("_", " "))

    colors = args.custom_colors
    if colors == None or len(colors) == 0:
        colors = []
        for comp_name in args.comparison:
            if comp_name.startswith("split_liquid"):
                legend_color = 'C0'
            elif comp_name.startswith("total_liquid"):
                legend_color = 'C1'
            elif comp_name.startswith("vel_liquid"):
                legend_color = 'C2'
            elif comp_name.startswith("vae_split_liquid"):
                legend_color = 'C3'
            else:
                colors = []
                break
            colors.append(legend_color)

    # for comp in comparison_desc:
    #     legend.append("Interval {}".format(comp["args"]["interval"]))
    # plot mae
    plot(mae, legend, "Mean Absolute Error", path=output_path+"mae.svg", first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[0], graph_limits[1]), colors=colors, linestyles=args.custom_linestyles)
    # plot mse
    plot(mse, legend, "Mean Squared Error", path=output_path+"mse.svg", first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[2], graph_limits[3]), colors=colors, linestyles=args.custom_linestyles)
    # plot psnr_grad
    plot(psnr_grad, legend, "PSNR (Gradient)", path=output_path+"psnr_grad.svg", first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[4], graph_limits[5]), colors=colors, linestyles=args.custom_linestyles)
    # plot psnr_grad
    plot(psnr, legend, "PSNR", path=output_path+"psnr.svg", first=args.warmup_steps, height=args.height, width=args.width, limit=(graph_limits[6], graph_limits[7]), colors=colors, linestyles=args.custom_linestyles)

    # Dump Data
    #----------------------------------------------------------------------------
    with open(output_path + "error.json", 'w') as f:
        errors = {
            "mae": mae,
            "mse": mse,
            "psnr_grad": psnr_grad,
            "psnr": psnr
        }
        json.dump(errors, f, indent=4)

#----------------------------------------------------------------------------
class PressureComparisonScene(object):
    def __init__(self, resolution, dimension, name="PressureComparisonScene"):
        grid_size = vec3(resolution,resolution,resolution)
        if (dimension == 2):
            grid_size.z = 1

        self.solver = Solver(name=name, gridSize=grid_size, dim=dimension)
        self.solver.timestep = 0.1

        self.reference = self.solver.create(RealGrid, name="Reference")
        self.comparison = self.solver.create(RealGrid, name="Comparison")
        self.pressure_gradient_ref = self.solver.create(VecGrid, name="GradientRef")
        self.pressure_gradient_comp = self.solver.create(VecGrid, name="GradientComp")

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
def plot(metric_list, legend_list, metric_name, title="", path="", first=0, height=8, width=8, limit=None, colors=[], linestyles=[]):
    from matplotlib import pyplot as plt
    import matplotlib.widgets as wgt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(1,1,figsize=(width, height))
    if limit and limit[0] != None and limit[1] != None:
        print("Setting Limit: {} {}".format(limit[0], limit[1]))
        ax.set_ylim(limit[0], limit[1])

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


    ax.grid()

    plt.title(title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Frame")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # force integer values on epoch axis
    ax.legend(legend_list, loc="upper left")

    leg = ax.get_legend()
    if args.legend_colors and len(args.legend_colors) == len(leg.legendHandles):
        for i in range(len(leg.legendHandles)):
            leg.legendHandles[i].set_color(args.legend_colors[i])

    if path:
        fig.savefig(path, bbox_inches='tight')

#----------------------------------------------------------------------------------
main()