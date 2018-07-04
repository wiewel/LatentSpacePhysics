#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# dataset creation scene
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

import os
import numpy as np
import json
from random import randint, getrandbits, seed
import datetime
from math import pi

from dpfn_scene_setup import MantaGridType
from scenes.static_dynamic_scene import StaticDynamicScene
from scenes.smoke_scene import SmokeScene
import scenes.volumes as v
from util import uniio
from util.path import find_dir, make_dir, get_uniqe_path
from util import git
from util import arguments

from dpfn_scene_setup import *

from scipy import ndimage

# import matplotlib
# matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
# import matplotlib.widgets as wgt
# from matplotlib.ticker import MaxNLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Example Call
# ./build/manta scenes/dpfn_display.py --dataset small_liquid64 --skip_steps 2 --simulation_steps 50 -n 5


# Arguments
#--------------------------------
args = arguments.dpfn_dataset()

# Version control
#--------------------------------
# get git revision of mantaflow submodule
git_version = git.revision()
debug = False
if not git.is_clean():
    debug = True
    if not args.ignore_git_status:
        raise RuntimeError("Trying to generate dataset with uncommitted changes. Use the argument '--ignore_git_status' ignore this.")

# Scene initialization
#--------------------------------
np.random.seed(seed=args.seed)
seed(args.seed)
simulation_steps = args.simulation_steps
simulation_steps *= args.skip_steps
simulation_steps += args.warmup
scene = None
if simulation_steps < 2 or args.skip_steps != 1:
    print("[WARNING] It will not be possible to train the LSTM with this dataset")
if "liquid" in args.type:
    scene = StaticDynamicScene(resolution=args.resolution, boundary=1, dimension=args.dimension, timestep=0.1, merge_ghost_fluid=True, show_gui=args.gui, pause_on_start=args.pause)
elif "gas" in args.type:
    scene = SmokeScene(resolution=args.resolution, boundary=1, dimension=args.dimension, timestep=0.5, gravity=Vec3(0,-0.003,0), merge_ghost_fluid=True, show_gui=args.gui, pause_on_start=args.pause)
grids = []
for grid_name in args.grids:
    try:
        getattr(scene, grid_name)
    except AttributeError as err:
        print("Scene has no grid named {}".format(grid_name))
    else:
        grids.append(grid_name)
args.grids = grids
print("Selected grids: {}".format(grids))
if args.meshes:
    mesh_path = find_dir("meshes")
    assert os.path.exists(mesh_path), ("Mesh directory {} does not exist".format(mesh_path))

# zoom settings
zoom_resolution = None
if args.zoom != 1.0:
    zoom_resolution = int(float(args.resolution) * args.zoom)
    print("Using zoom resolution: {}".format(zoom_resolution))

# Output
#--------------------------------
if not args.no_output:
    # get the datasets directory, in all datasets should reside
    if not args.datasets_path:
        output_path = find_dir("datasets", 2)
    else:
        output_path = args.datasets_path
    assert os.path.exists(output_path), ("Datasets directory {} does not exist".format(output_path)) 
    # set output parent directory name
    output_path += "/" + args.name
    if debug:
        output_path += "_DEBUG"
    output_path = get_uniqe_path(output_path)
    # create the directories
    for grid_name in args.grids:
        make_dir(output_path + "/" + grid_name)
        if zoom_resolution:
            make_dir(output_path + "/" + grid_name + "_orig")
    if args.gui:
        make_dir(output_path + "/screenshots/")
    if args.output_render_data:
        make_dir(output_path + "/uni/")
    # keep track of stored scenes
    stored_scenes_num = 0
print("Output path: {}".format(output_path))
    
# Dataset description
#--------------------------------
description = {}
description["version"] = git_version
description["creation_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
description["norm_factor"] = 1.0
description["grids"] = grids
description.update(vars(args)) # insert args
description["resolution_orig"] = args.resolution
if zoom_resolution:
    description["resolution"] = zoom_resolution

if not args.no_output:
    with open(output_path + "/description.json", 'w') as f:
        json.dump(description, f, indent=4)

# Scene build
#--------------------------------
def new_scene():
    scene_selection(args.type, scene, obstacles=args.obstacles, meshes=args.meshes, benchmark=-1)

    # Zoom Setup
    if zoom_resolution:
        grid_size = vec3(zoom_resolution, zoom_resolution, zoom_resolution)
        if (args.dimension == 2):
            grid_size.z = 1
        scene.solver_zoom = Solver(name="Zoom", gridSize=grid_size, dim=args.dimension)

        for grid_name in grids:  #"pressure", "pressure_static", "pressure_dynamic", "vel", "phi_fluid", "density"
            zoom_grid_name = grid_name+"_zoom"
            grid_type = MantaGridType(getattr(scene, grid_name).getGridType())
            new_grid = scene.solver_zoom.create(get_manta_type(grid_type), name=zoom_grid_name)
            setattr(scene, zoom_grid_name, new_grid)

# Scene simulate
#--------------------------------
def on_sim_step(scene, t):
    if args.no_output:
        return
    # screenshot of the mantaflow gui
    if t == args.warmup and scene.show_gui:
        scene._gui.screenshot(output_path + "/screenshots/screen_{:06d}.jpg".format(scene.file_num))
    # save all steps for rendering
    if args.output_render_data:
        if t == 0:
            make_dir(output_path + "/scene_{:04d}/uni/".format(stored_scenes_num))
            scene.pressure.save(output_path + "/scene_{:04d}/uni/ref_flipParts_0000.uni".format(stored_scenes_num))
        scene.pp.save(output_path + "/scene_{:04d}/uni/flipParts_{:04d}.uni".format(stored_scenes_num, t))
        scene.phi_fluid.save(output_path + "/scene_{:04d}/uni/levelset_{:04d}.uni".format(stored_scenes_num, t))
    # some steps should not be written to file
    if t < args.warmup or t % args.skip_steps != 0:
        return

    # zoom
    if zoom_resolution:
        if args.dimension == 2:
            zoom_mask = [1.0, args.zoom, args.zoom, 1.0]
            np_real_temp = np.empty(shape=[1, args.resolution, args.resolution, 1], order='C')
            np_vec_temp = np.empty(shape=[1, args.resolution, args.resolution, 3], order='C')
        else:
            zoom_mask = [1.0, args.zoom, args.zoom, args.zoom, 1.0]
            np_real_temp = np.empty(shape=[1, args.resolution, args.resolution, args.resolution, 1], order='C')
            np_vec_temp = np.empty(shape=[1, args.resolution, args.resolution, args.resolution, 3], order='C')

    # write grids to a file
    scene.file_num = scene.file_num + 1
    output_name = "{}_{:07d}".format(scene.resolution, scene.file_num)

    for grid_name in grids:
        # it was already checked if the attribute is present in the scene
        grid = getattr(scene, grid_name)

        if zoom_resolution:
            grid_zoom = getattr(scene, grid_name+"_zoom")
            grid_type = MantaGridType(grid.getGridType())
            if grid_type == MantaGridType.TypeReal or grid_type == MantaGridType.TypeLevelset or grid_type == MantaGridType.TypeLevelsetReal:
                copyGridToArrayReal(grid, np_real_temp)

                # if grid_name == "pressure":
                #     fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
                #     fig.tight_layout(pad=0.1)
                #     im1 = ax1.imshow(np_real_temp[0,:,:,0], vmin=None, vmax=None, cmap='viridis', interpolation='nearest')
                #     divider = make_axes_locatable(ax1)
                #     cax = divider.append_axes('right', size='5%', pad = 0.05)
                #     fig.colorbar(im1, cax=cax, orientation='vertical')
                #     ax1.set_title("Reference", fontsize=20)
                #     ax1.set_xlim(0, np_real_temp.shape[2]-1)
                #     ax1.set_ylim(0, np_real_temp.shape[1]-1)
                #     ax1.get_xaxis().set_visible(True)
                #     ax1.get_yaxis().set_visible(True)
                #     plt.show(block=True)

                # force on area m^2 
                if "pressure" in grid_name:
                    scale_factor = args.zoom * args.zoom
                # mass on volume m^3
                elif "density" in grid_name:
                    scale_factor = args.zoom * args.zoom * args.zoom
                else:
                    scale_factor = args.zoom
                np_zoomed = ndimage.zoom(np_real_temp, zoom_mask) * scale_factor
                copyArrayToGridReal(np_zoomed, grid_zoom)
            elif grid_type == MantaGridType.TypeInt:
                assert False, "Not supported"
            elif grid_type == MantaGridType.TypeVec3 or grid_type == MantaGridType.TypeMAC or grid_type == MantaGridType.TypeMACVec3:
                copyGridToArrayVec3(grid, np_vec_temp)
                np_zoomed = ndimage.zoom(np_vec_temp, zoom_mask) * args.zoom
                copyArrayToGridVec3(np_zoomed, grid_zoom)

            # save the zoomed grid to a uni file (will later be converted to .npz)
            grid_zoom.save(output_path + "/" + grid_name + "/" + output_name + ".uni")
            grid.save(output_path + "/" + grid_name + "_orig/" + output_name + ".uni")
        else:
            # Save normally
            # save the grid to a uni file (will later be converted to .npz)
            grid.save(output_path + "/" + grid_name + "/" + output_name + ".uni")

# reset seed -> same starting condition for scene creation -> reproducibility
np.random.seed(seed=args.seed)
seed(args.seed)
for scene_num in range(args.num_scenes):
    print("<><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Scene {} ({})\n".format(scene_num, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    # increase scene count
    stored_scenes_num = scene_num
    # create new pseudo random scene
    new_scene()
    # restart the simulation with the new scene and write out the grids as .uni
    scene.simulate(num_steps=simulation_steps, on_simulation_step=on_sim_step)
    # convert .uni to .npz and group them by scene
    if args.no_output:
        continue
    for grid_name in grids:
        uniio.convert_to_npz(output_path + "/" + grid_name + "/", "{:06d}.npz".format(scene_num), description, args.quantization)
        if zoom_resolution:
            uniio.convert_to_npz(output_path + "/" + grid_name + "_orig/", "{:06d}.npz".format(scene_num), description, args.quantization)