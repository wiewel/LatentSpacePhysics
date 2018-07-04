#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# argument parse for dataset creation and display scene
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

def dpfn_dataset():
    parser = argparse.ArgumentParser()
    output = parser.add_argument_group("Output")
    output.add_argument("--no_output", action="store_true", help="do not generate output")
    output.add_argument("--name", type=str, required=True, help="A distinguishable name for the dataset")
    output.add_argument("--datasets_path", type=str, required=False, help="Optional path for the datasets directory. Especially usefull, if datasets should be written to second disk")
    output.add_argument("--grids", nargs="*", default=["pressure", "pressure_static", "pressure_dynamic", "vel", "phi_fluid", "density"], choices=["pressure", "pressure_static", "pressure_dynamic", "vel", "phi_fluid", "density"], help="Specify which grids should be written to uni files.")
    output.add_argument("--quantization", type=int, default=0, help="Decimal places for quantization of grids. 0 means off")
    #output.add_argument("--mesh_types", nargs="*", default=["duck.obj", "bunny.obj", "simpletorus.obj"], choices=["duck.obj", "bunny.obj", "simpletorus.obj"], help="Specify which mesh types should be used.")

    scene = parser.add_argument_group("Scene")
    scene.add_argument("-n", "--num_scenes", type=int, default=1, help="number of scenes used in data generation")
    parser.add_argument('-t', "--type", default="liquid", choices=["liquid", "liquid_simple", "liquid_breaking_dam", "liquid_anvil", "liquid_meshes", "liquid_center_obstacle", "liquid_border_obstacle", "gas"], help='simulation type (default: %(default)s)')
    scene.add_argument("-s", "--simulation_steps", type=int, default=100, help="number of simulation steps for a scene. must be constant for all scenes")
    scene.add_argument("-w", "--warmup", type=int, default=75, help="number of steps to discard in the beginning of the scene")
    scene.add_argument("--seed", type=int, default=1337, help="Seed for the random number generator")
    scene.add_argument("--skip_steps", type=int, default=1, help="Write out interval. Value of 1 writes every step")
    scene.add_argument("--obstacles", action="store_true", help="add obstacles to the scene")
    scene.add_argument("--meshes", action="store_true", help="add meshes as water surface scene")
    scene.add_argument("-r", "--resolution", type=int, default=64, help="simulation resolution (res^3)")
    scene.add_argument("-z", "--zoom", type=float, default=1.0, help="zoom factor to down-/upscale the simulation fields. The zoomed fields are stored in the directory 'zoom'.")
    scene.add_argument("-d", "--dimension", type=int, default=3, help="simulation dimension (usually 2D or 3D)")
    parser.add_argument("-ord", "--output_render_data", action="store_true", help="Store the pp field and levelset each frame as uni file")

    mantaflow = parser.add_argument_group("Mantaflow")
    mantaflow.add_argument("-g", "--gui", action="store_true")

    debug = parser.add_argument_group("Debug")
    debug.add_argument("--ignore_git_status", action="store_true", help="ignore uncommitted changes")
    debug.add_argument("-p", "--pause", action="store_true", help="Pause on start")

    args = parser.parse_args()
    del output
    del scene
    del mantaflow
    del debug

    return args

# ---------------------------------------------------------------------------------------------------------
def dpfn_display():
    parser = argparse.ArgumentParser()
    dataset = parser.add_argument_group("Data Set")
    dataset.add_argument("--dataset", type=str, required=True, help="Path to already existing data set")
    dataset.add_argument("--datasets_path", type=str, required=False, help="Optional path for the datasets directory. Especially useful, if datasets should be read from second disk")
    dataset.add_argument("--grids", nargs="*", default=["pressure", "pressure_static", "pressure_dynamic", "vel", "phi_fluid", "density"], choices=["pressure", "pressure_static", "pressure_dynamic", "vel", "phi_fluid", "density"], help="specify the displayed grids.")

    scene = parser.add_argument_group("Scene")
    scene.add_argument("-s", "--start_scene", type=int, default=0, help="scene number to start")
    scene.add_argument("-n", "--num_scenes", type=int, default=-1, help="number of scenes to display beginning from start_scene. Default of -1 displays all scenes")
    scene.add_argument("-t", "--time_to_wait", type=float, default=0.0, help="time to wait in each iteration (additionally to IO)")
    scene.add_argument("--simulation_steps", type=int, default=-1, help="number of simulation steps to display for a scene")
    scene.add_argument("--skip_steps", type=int, default=1, help="display interval. Value of 1 displays all stored time steps")

    general = parser.add_argument_group("General")
    general.add_argument("--video", action="store_true", help="create a video of the dataset content")

    args = parser.parse_args()
    del dataset
    del scene
    del general

    return args
