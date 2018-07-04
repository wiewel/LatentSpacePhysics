#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# encode decode scene -> used to evaluate the autoencoder baseline performance
# the autoencoder is used to encode and directly decode the quanitity in question
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
from scenes.encode_decode_scene import *
import scenes.volumes as v

from dpfn_scene_setup import *

import pathlib
import numpy as np
import argparse
from random import randint, seed
import subprocess
import time
import os
import json
import scipy.misc

from util.path import make_dir, find_dir, get_uniqe_path

#----------------------------------------------------------------------------------
def fulfill_req():
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    gpu_fraction=0.8
    '''You want to allocate gpu_fraction percent of it'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if num_threads:
        num_threads = int(num_threads)
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    print("Found {} threads. Using {} of GPU memory per process.".format(num_threads, gpu_fraction))
    if num_threads:
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    KTF.set_session(session)
#----------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Name of the prediction run")
parser.add_argument("-s", "--simulation_steps", type=int, default=100, help="number of simulation steps for a scene. must be constant for all scenes")
parser.add_argument("--project", required=True, help="The name of the project to load. The script will search for the projects directory and will expect a subdirectory with the specified project name")
parser.add_argument("-o", "--output", default="", help="directory to which the output video is written")
parser.add_argument("-w", "--warmup_steps", type=int, default=50, help="number of steps to discard in the beginning of the scene")
parser.add_argument("-i", "--interval", type=int, default=2, help="interval at which the pressure solve should occur. Default: every 2nd step")
parser.add_argument("--seed", type=int, default=1337, help="Seed for the random number generator")
parser.add_argument("-p", "--profile", action="store_true")
parser.add_argument("--predict_dynamic_only", action="store_true")
parser.add_argument("--scene_type", default="liquid", choices=["liquid", "liquid_simple", "liquid_breaking_dam", "liquid_anvil", "liquid_meshes", "liquid_center_obstacle", "liquid_border_obstacle", "gas"], help='simulation type (default: %(default)s)')
parser.add_argument("-pt", "--prediction_type", default="split_pressure", choices=["split_pressure", "total_pressure", "splitdyn_pressure", "dynamic_pressure", "velocity", "none", "variational_split", "naive"], help="Type of the prediction")
parser.add_argument("-ord", "--output_render_data", action="store_true", help="Store the pp field and levelset each frame as uni file")
parser.add_argument("-obd", "--output_benchmark_data", action="store_true", help="Store the pressure field as uni file")
parser.add_argument("-os", "--output_sequence", action="store_true", help="Store 2D images of the predicted quantity.")
parser.add_argument("--obstacles", action="store_true", help="add obstacles to the scene")
parser.add_argument("-hr", "--highres", action="store_true", help="Use a high resolution simulation (128)")
parser.add_argument("--benchmark", type=int, default=-1, choices=[-1,0,1,2,3,4,5,6,7,8,9], help="Start benchmark scene with given id")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--no_gui", action="store_true")

# Arguments
#----------------------------------------------------------------------------
args = parser.parse_args()

if args.benchmark != -1:
    if args.name == "":
        args.name = args.project + "_Bench{}".format(args.benchmark)
    else:
        args.name += "_Bench{}".format(args.benchmark)

output_path = "."
if args.output:
    output_path = str(pathlib.Path(args.output).resolve())
else:
    output_path = find_dir("predictions", 2)
output_path += "/" + args.name
output_path = get_uniqe_path(output_path) + "/"
make_dir(output_path)
print("Output path: {}".format(output_path))

if args.output_render_data or args.output_benchmark_data:
    make_dir(output_path+"uni/")

if args.output_sequence:
    output_seq_path = output_path + "sequence/"
    make_dir(output_seq_path)

description = {"args": vars(args)}

#----------------------------------------------------------------------------
project_dir = find_dir("projects", 2) + "/" + args.project
assert os.path.exists(project_dir), ("The specified project directory does not exist or could not be found")

show_gui = not args.profile and not args.no_gui

is_gas_scene = args.prediction_type == "total_pressure_gas"

# Main
#----------------------------------------------------------------------------------
def main():
    fulfill_req()

    args.simulation_steps += args.warmup_steps

    np.random.seed(seed=args.seed)
    seed(args.seed)

    print("Profiling: {}".format(args.profile))

    # Scene
    #----------------------------------------------------------------------------
    if args.prediction_type == "none":
        scene = EncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "split_pressure":
        scene = SplitPressureEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "total_pressure":
        scene = TotalPressureEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "velocity":
        scene = VelocityEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "splitdyn_pressure":
        scene = SplitDynamicPressureEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "dynamic_pressure":
        scene = DynamicPressureEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    elif args.prediction_type == "naive":
        scene = NaiveEncodeDecodeScene(project_directory=project_dir,
                            warmup_steps=args.warmup_steps,
                            resolution=128 if args.highres else 64,
                            dimension=3,
                            timestep=0.1,
                            interval=args.interval,
                            show_gui=args.no_gui is False)
    else:
        assert False, ("Prediction type not supported yet")

    description["scene"] = scene.get_settings()

    # reset seed again so that scene loading is identical
    # same starting condition for scene creation -> reproducibility
    np.random.seed(seed=args.seed)
    seed(args.seed)
    scene_selection(args.scene_type, scene, args.obstacles, is_gas_scene, args.benchmark)

    if args.output_render_data:
        scene.pressure.save(output_path + "uni/ref_flipParts_0000.uni")

    # Simulate
    #----------------------------------------------------------------------------
    simulation_start_time = time.time()
    scene.simulate(num_steps=args.simulation_steps, on_simulation_step=on_sim_step)
    simulation_end_time = time.time()

    profile(simulation_start_time, simulation_end_time)

    # Meta data
    #----------------------------------------------------------------------------
    with open(output_path + "description.json", 'w') as f:
        json.dump(description, f, indent=4)

    # Video
    #----------------------------------------------------------------------------
    create_video()
    convert_sequence()

#----------------------------------------------------------------------------------
def on_sim_step(scene, t):
    if args.profile:
        return
    if scene.show_gui:
        scene._gui.screenshot(output_path + "prediction_{:06d}.png".format(t))
    if args.output_render_data:
        scene.pp.save(output_path + "uni/flipParts_{:04d}.uni".format(t))
        # don't clamp the levelset
        scene.phi_fluid.reinitMarching(flags=scene.flags, ignoreWalls=False, maxTime=999)
        scene.phi_fluid.save(output_path + "uni/levelset_{:04d}.uni".format(t))
    if args.output_benchmark_data:
        scene.pressure.save(output_path + "uni/pressure_jacobifix_{:04d}.uni".format(t))
        scene.pressure_raw.save(output_path + "uni/pressure_{:04d}.uni".format(t))

    # print("Timestep {}".format(t))
    # print("\tPressure: {}\t{}\t{}\t{}".format(scene.pressure.getMin(), scene.pressure.getMax(), scene.pressure.getL1(scene.boundary), scene.pressure.getL2(scene.boundary)))
    # print("\tPressure Raw: {}\t{}\t{}\t{}".format(scene.pressure_raw.getMin(), scene.pressure_raw.getMax(), scene.pressure_raw.getL1(scene.boundary), scene.pressure_raw.getL2(scene.boundary)))

    if args.output_sequence:
        output_sequence(scene, t)

#----------------------------------------------------------------------------------
def output_sequence(scene, t):
    if args.profile:
        return

    if scene.dimension == 2:
        np_real_temp = np.empty(shape=[1, scene.resolution, scene.resolution, 1], order='C')
        np_vec_temp = np.empty(shape=[1, scene.resolution, scene.resolution, 3], order='C')
    else:
        np_real_temp = np.empty(shape=[1, scene.resolution, scene.resolution, scene.resolution, 1], order='C')
        np_vec_temp = np.empty(shape=[1, scene.resolution, scene.resolution, scene.resolution, 3], order='C')

    def get_slice(scene, np_arr):
        if scene.dimension == 2:
            return np.squeeze(np_arr[0])#[::-1, ::-1]
        else:
            return np.squeeze(np_arr[0, scene.resolution // 2])[::-1, ::-1]

    if args.prediction_type == "split_pressure" or args.prediction_type == "dynamic_pressure" or args.prediction_type == "total_pressure" or args.prediction_type == "total_pressure_gas" or args.prediction_type == "naive_static_pressure" or args.prediction_type == "naive_dynamic_pressure" or args.prediction_type == "splitdyn_pressure":
        # Pressure Jacobi Fix Storage
        copyGridToArrayReal(scene.pressure, np_real_temp)
        np_selected_slice = get_slice(scene, np_real_temp)
        #scipy.misc.toimage(np_selected_slice, cmin=-0.5, cmax=1.5).save(output_seq_path+"pressure_jacobifix_{}.png".format(t))
        plot_field(np_selected_slice, output_seq_path+"pressure_jacobifix_{}.png".format(t), vmin=0.0, vmax=4.0, plot_colorbar=False)
        # Pressure Storage
        copyGridToArrayReal(scene.pressure_raw, np_real_temp)
        np_selected_slice = get_slice(scene, np_real_temp)
        #scipy.misc.toimage(np_selected_slice, cmin=-0.5, cmax=1.5).save(output_seq_path+"pressure_{}.png".format(t))
        plot_field(np_selected_slice, output_seq_path+"pressure_{}.png".format(t), vmin=0.0, vmax=4.0, plot_colorbar=False)
    elif args.prediction_type == "velocity" or args.prediction_type == "naive_velocity":
        # Velocity Storage
        copyGridToArrayVec3(scene.vel, np_vec_temp)
        np_selected_slice = np.squeeze(np_vec_temp[0, :, :, scene.resolution // 2])[::-1, ::-1]
        scipy.misc.toimage(np_selected_slice, cmin=-2.0, cmax=2.0).save(output_seq_path+"velocity_{}.png".format(t))
    elif args.prediction_type == "reference":
        # Pressure Storage
        copyGridToArrayReal(scene.pressure, np_real_temp)
        np_selected_slice = np.squeeze(np_real_temp[0, scene.resolution // 2])[::-1, ::-1]
        scipy.misc.toimage(np_selected_slice, cmin=-0.5, cmax=1.5).save(output_seq_path+"pressure_{}.png".format(t))
        # Velocity Storage
        copyGridToArrayVec3(scene.vel, np_vec_temp)
        np_selected_slice = np.squeeze(np_vec_temp[0, :, :, scene.resolution // 2])[::-1, ::-1]
        scipy.misc.toimage(np_selected_slice, cmin=-2.0, cmax=2.0).save(output_seq_path+"velocity_{}.png".format(t))
    else:
        assert False, ("Prediction type not supported yet")

#----------------------------------------------------------------------------------
def create_video():
    if args.profile:
        return
    if not show_gui:
        return
    if args.no_video:
        return

    video_name = output_path + "prediction_" + str(args.interval) + "_" + str(args.seed) + ".mp4"

    subprocess.call(['ffmpeg',
    '-r', '10', # timestep is 0.1
    '-f', 'image2',
    '-start_number', '0',
    '-i', output_path + 'prediction_%06d.png',
    '-vcodec', 'libx264',
    '-crf', '18',
    '-pix_fmt', 'yuv420p',
    video_name])

#----------------------------------------------------------------------------------
def convert_sequence():
    if args.profile:
        return
    if not args.output_sequence:
        return

    for q in ["pressure", "pressure_jacobifix", "velocity"]:
        input_quantity_path = output_seq_path + q + "_%d.png"
        print(input_quantity_path.format(0))
        output_seq_video_path = output_seq_path + q + ".mp4"
        print(output_seq_video_path)
        if os.path.isfile( input_quantity_path % (0) ):
            subprocess.call(['ffmpeg',
            '-r', '30',
            '-f', 'image2',
            '-start_number', str(args.warmup_steps),
            '-i', input_quantity_path,
            '-vcodec', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-y',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            output_seq_video_path])

#----------------------------------------------------------------------------------
def profile(simulation_start_time, simulation_end_time):
    if args.profile is False:
        return

    duration = simulation_end_time - simulation_start_time
    per_step_duration = duration / args.simulation_steps
    print("\nProfiler")
    print("\tSimulation Duration: {}s".format(duration))
    print("\tPer Step Duration: {}s".format(per_step_duration))
    print("\tInterval: {}".format(args.interval))

#----------------------------------------------------------------------------------
main()