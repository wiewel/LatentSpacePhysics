#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# general functions that are used in encdec, prediction or dataset creation
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
from random import randint, getrandbits
import datetime
from math import pi

import scenes.volumes as v
from util.path import find_dir, make_dir, get_uniqe_path
from util import git
from util import arguments
from enum import Enum
from manta import *


#--------------------------------
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

#--------------------------------
def plot_field(x, out_path, title="", vmin=None, vmax=None, plot_colorbar=True):
    #print("plot_field", x.shape)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.arange(cmap.N))

    max_color = 10
    min_color = [1,1,1,1]
    final_color = colors[max_color]
    for i in range(max_color):
        for j in range(4):
            interpolant = (max_color - i) / max_color
            colors[i][j] =  min(final_color[j] * (1.0 - interpolant) + min_color[j] * interpolant, 1.0)
    cm = LinearSegmentedColormap.from_list("custom_vir", colors, N=256)

    im1 = ax1.imshow(x[:,:], vmin=vmin, vmax=vmax, cmap=cm, interpolation='nearest')

    # contours
    cmap = plt.cm.get_cmap('viridis')
    colors = cmap(np.arange(cmap.N))
    for i in range(len(colors)):
        for j in range(4):
            colors[i][j] =  min(colors[i][j] * 1.4, 1.0)
    cm = LinearSegmentedColormap.from_list("custom_vir", colors, N=256)
    #cm = plt.get_cmap('autumn')

    cs = ax1.contour(x[:,:], levels=np.arange(vmin+(vmax-vmin)*0.025,vmax,(vmax-vmin)/10.0), cmap=cm)

    if plot_colorbar:
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad = 0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')
    ax1.set_xlim(0, x.shape[1])
    ax1.set_ylim(0, x.shape[0])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    if title:
        fig.suptitle(title, fontsize=20)
    if plot_colorbar:
        fig.savefig(out_path, bbox_inches='tight')
    else:
        fig.savefig(out_path, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close(fig)

# Enum
#--------------------------------
# mirror class of GridBase::GridType -> keep in sync
class MantaGridType(Enum):
    TypeNone = 0
    TypeReal = 1
    TypeInt = 2
    TypeVec3 = 4
    TypeMAC = 8
    TypeLevelset = 16
    TypeFlags = 32
    # Manually added permutations
    TypeMACVec3 = 12
    TypeLevelsetReal = 17
    TypeFlagsInt = 34

def get_manta_type(enum_type):
    if enum_type is MantaGridType.TypeReal:
        return RealGrid
    if enum_type is MantaGridType.TypeLevelset or enum_type is MantaGridType.TypeLevelsetReal:
        return LevelsetGrid
    if enum_type is MantaGridType.TypeVec3:
        return MACGrid
    if enum_type is MantaGridType.TypeMAC or enum_type is MantaGridType.TypeMACVec3:
        return MACGrid
    assert False, "Not supported"


# Meshes
#--------------------------------
mesh_types=["duck.obj", "bunny.obj", "simpletorus.obj", "anvil.obj"]

def vec_to_str(vec):
    return "{}, {}, {}".format(vec.x, vec.y, vec.z)

def spawn_mesh(mesh_dir, scene):
    # use random mesh from meshes directory with random position, rotation and scaling
    mesh = mesh_dir + "/" + mesh_types[randint( 0, len(mesh_types)-1 )]
    center = v.random_vec3(low=[0.05, 0.5, 0.05], high=[0.95, 0.95, 0.95])
    scale = np.random.uniform(low=0.4, high=0.9)
    rotation = v.random_vec3(low=[-pi, -pi, -pi], high=[pi, pi, pi])
    print("Spawning '{}': Pos {} Rot {} Scale {}".format(mesh, vec_to_str(center), vec_to_str(rotation), scale ))
    mesh = v.MeshVolume(
        mesh,
        center=center,
        scale=scale,
        rotation=rotation
    )
    scene.add_fluid(mesh)
    scene.set_velocity(mesh, v.random_velocity(vel_min=[-3.5, -3.5, -3.5], vel_max=[3.5, 0, 3.5]))


#----------------------------------------------------------------------------------
# Drop (Box)
def spawn_drop(scene):
    box = v.random_box(center_min=[0, 0.7, 0], center_max=[1, 0.9, 1], size_min=[0.005, 0.005, 0.005], size_max=[0.25, 0.25, 0.25])
    scene.add_fluid(box)
    drop_velo = v.random_velocity(vel_min=[-2.5, -2.5, -2.5], vel_max=[2.5, 0, 2.5])
    scene.set_velocity(box, drop_velo)
    # print("Drop {}:\n\t{},{},{}\n\t{},{},{}\n\t{},{},{}".format(i, box._center.x, box._center.y, box._center.z,
    #                                                         box._size.x, box._size.y, box._size.z,
    #                                                         drop_velo.x, drop_velo.y, drop_velo.z))

#----------------------------------------------------------------------------------
def initialize_gas_scene(scene):
    source_count = randint(4, 10)
    for i in range(source_count):
        box = v.random_box(center_min=[0.2, 0.1, 0.2], center_max=[0.8, 0.6, 0.8], size_min=[0.005, 0.005, 0.005], size_max=[0.2, 0.2, 0.2])
        scene.add_source( box )
        #scene.set_velocity(box, random_vec3(vel_min=[-2.5, -2.5, -2.5], vel_max=[2.5, 2.5, 2.5]))
        #scene.add_source( box.shape(scene.solver) )

#----------------------------------------------------------------------------------
def initialize_scene(scene, simple=False, obstacles=False, meshes=False):
    if simple:
        # low basin is the large volume of liquid at the bottom of the scene. in general adds most of the liquid in the scene
        low_basin = v.random_box(center_min=[0.5, 0.0, 0.5], center_max=[0.5, 0.0, 0.5], size_min=[1, 0.3, 1], size_max=[1, 0.5, 1])
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, v.random_velocity(vel_min=[-0.5, 0, -0.5], vel_max=[0.5, 0, 0.5]))
        # the high basin is a tall but narrow block of liquid, adding motion to the scene
        high_basin = v.random_box(center_min=[0.1, 0.3, 0.1], center_max=[0.9, 0.4, 0.9], size_min=[0.1, 0.2, 0.1], size_max=[0.3, 0.4, 0.3])
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, v.random_velocity(vel_min=[-2.0, -2.0, -2.0], vel_max=[2.0, 0, 2.0]))
    else:
        # low basin is the large volume of liquid at the bottom of the scene. in general adds most of the liquid in the scene
        low_basin = v.random_box(center_min=[0.5, 0.0, 0.5], center_max=[0.5, 0.0, 0.5], size_min=[1, 0.3, 1], size_max=[1, 0.6, 1])
        scene.add_fluid(low_basin)
        low_basin_velo = v.random_velocity(vel_min=[-1.5, 0, -1.5], vel_max=[1.5, 0, 1.5])
        scene.set_velocity(low_basin, low_basin_velo)
        # print("Low:\n\t{},{},{}\n\t{},{},{}\n\t{},{},{}".format(low_basin._center.x, low_basin._center.y, low_basin._center.z,
        #                                                         low_basin._size.x, low_basin._size.y, low_basin._size.z,
        #                                                         low_basin_velo.x, low_basin_velo.y, low_basin_velo.z))

        # the high basin is a tall but narrow block of liquid, adding motion to the scene
        high_basin = v.random_box(center_min=[0.0, 0.3, 0.0], center_max=[1.0, .3, 1.0], size_min=[0.1, 0.2, 0.1], size_max=[0.4, 0.5, 0.4])
        scene.add_fluid(high_basin)
        high_basin_velo =  v.random_velocity(vel_min=[-2.5, -2.5, -2.5], vel_max=[2.5, 0, 2.5])
        scene.set_velocity(high_basin, high_basin_velo)
        # print("High:\n\t{},{},{}\n\t{},{},{}\n\t{},{},{}".format(high_basin._center.x, high_basin._center.y, high_basin._center.z,
        #                                                         high_basin._size.x, high_basin._size.y, high_basin._size.z,
        #                                                         high_basin_velo.x, high_basin_velo.y, high_basin_velo.z))

        # drops: spawn at most 3 drops or meshes
        drop_count = randint(0, 3)
        for i in range(drop_count):
            if meshes and bool(getrandbits(1)):
                spawn_mesh(mesh_path, scene)
            else:
                spawn_drop(scene)

    # optional: add solid boxes to scene. more realistic, harder to learn
    if obstacles:
        obstacle = v.random_box(center_min=[0.1,0.1,0.1], center_max=[1.0,0.5,1.0], size_min=[0.2,0.2,0.2], size_max=[0.3,1.0,0.3])
        print("Added obstacle to the scene")
        scene.add_obstacle(obstacle)

#----------------------------------------------------------------------------------
def initialize_breaking_dam(scene):
    right_dam = v.BoxVolume(p0=vec3(0.8, 0.0, 0.0), p1=vec3(1.0, 0.35, 1.0))
    scene.add_fluid(right_dam)

#----------------------------------------------------------------------------------
def initialize_anvil(scene):
    low_basin = v.BoxVolume(center=vec3(0.5,0.15,0.5), size=vec3(1,0.3,1))
    scene.add_fluid(low_basin)
    low_basin_velo = vec3(0.1,0.0,-0.06)
    scene.set_velocity(low_basin, low_basin_velo)

    mesh_name = "meshes/anvil.obj"
    center = vec3(0.5, 0.7, 0.5)
    scale = 1.0
    rotation = vec3(0, pi/2.0, 0)
    mesh = v.MeshVolume(
        mesh_name,
        center=center,
        scale=scale,
        rotation=rotation
    )
    scene.add_fluid(mesh)
    scene.set_velocity(mesh, vec3(0.5, -1.5, -0.25))

#----------------------------------------------------------------------------------
def initialize_center_obstacle(scene):
    initialize_anvil(scene)

    small_box = v.BoxVolume(center=vec3(0.9,0.9,0.9), size=vec3(0.07,0.1,0.07))
    scene.add_fluid(small_box)
    small_box_velo = vec3(-2.0,-0.5,-2.0)
    scene.set_velocity(small_box, small_box_velo)

    obstacle = v.BoxVolume(center=vec3(0.5,0.5,0.5), size=vec3(0.15,1.0,0.15))
    print("Added obstacle to the scene")
    scene.add_obstacle(obstacle)

#----------------------------------------------------------------------------------
def initialize_border_obstacle(scene):
    initialize_anvil(scene)

    obstacle = v.BoxVolume(center=vec3(0.9,0.5,0.1), size=vec3(0.2,1.0,0.2))
    print("Added obstacle to the scene")
    scene.add_obstacle(obstacle)

#----------------------------------------------------------------------------------
def initialize_mesh_scene(scene):
    low_basin = v.BoxVolume(center=vec3(0.5,0.15,0.5), size=vec3(1,0.3,1))
    scene.add_fluid(low_basin)
    low_basin_velo = vec3(0.1,0.0,-0.06)
    scene.set_velocity(low_basin, low_basin_velo)

    for i in range(randint(4, 8)):
        spawn_mesh("meshes", scene)

#----------------------------------------------------------------------------------
def initialize_benchmark(scene, id):
    if id == 0:
        low_basin = v.BoxVolume(p0=vec3(0.0, 0.0, 0.0), p1=vec3(1.0, 0.35, 1.0))
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, vec3(-0.25, 0, 0.1))

        high_basin = v.BoxVolume(p0=vec3(0.25, 0.4, 0.25), p1=vec3(0.55, 0.6, 0.55))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(0.05, -0.1, -0.1))

        right_dam = v.BoxVolume(p0=vec3(0.8, 0.0, 0.0), p1=vec3(1.0, 0.7, 1.0))
        scene.add_fluid(right_dam)
    elif id == 1:
        low_basin = v.BoxVolume(p0=vec3(0.0, 0.0, 0.0), p1=vec3(1.0, 0.35, 1.0))
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, vec3(-0.25, 0, 0.1))

        drop = v.BoxVolume(p0=vec3(0.35, 0.35, 0.35), p1=vec3(0.65, 0.65, 0.65))
        scene.add_fluid(drop)
        scene.set_velocity(drop, vec3(0.0, -0.1, 0.0))
    elif id == 2:
        low_basin = v.BoxVolume(p0=vec3(0.0, 0.0, 0.0), p1=vec3(1.0, 0.4, 1.0))
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, vec3(0.15, 0, 0.2))

        high_basin = v.BoxVolume(p0=vec3(0.15, 0.4, 0.2), p1=vec3(0.50, 0.6, 0.6))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(-1.15, -0.08, -0.2))

        box = v.BoxVolume(p0=vec3(0.65, 0.65, 0.65), p1=vec3(0.8, 0.8, 0.8))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(-0.15, -0.1, -0.1))

        box = v.BoxVolume(p0=vec3(0.5, 0.6, 0.1), p1=vec3(0.7, 0.7, 0.25))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(0.2, -0.05, 1.1))

        box = v.BoxVolume(p0=vec3(0.1, 0.75, 0.6), p1=vec3(0.3, 0.85, 0.7))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(1.1, -0.8, -0.2))
    elif id == 3:
        low_basin = v.BoxVolume(center=vec3(
            0.5, 0.0, 0.5), size=vec3(1.0, 0.46817633509635925, 1.0))
        scene.add_fluid(low_basin)
        scene.set_velocity(
            low_basin, vec3(-0.871836245059967, 0.0, -1.373809576034546))

        high_basin = v.BoxVolume(center=vec3(0.051326680928468704, 0.30000001192092896, 0.3669705092906952), size=vec3(
            0.31358590722084045, 0.21195632219314575, 0.3853992223739624))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(
            2.3747153282165527, -1.0215086936950684, 0.3464154601097107))

        box = v.BoxVolume(center=vec3(0.824302077293396, 0.7290175557136536, 0.6262162923812866), size=vec3(
            0.22131894528865814, 0.18742601573467255, 0.1500759869813919))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(-1.9065660238265991, -1.1057237386703491, -2.0492587089538574))
    elif id == 4:
        low_basin = v.BoxVolume(center=vec3(
            0.5, 0.0, 0.5), size=vec3(1.0, 0.5484904050827026, 1.0))
        scene.add_fluid(low_basin)
        scene.set_velocity(
            low_basin, vec3(-1.0241304636001587, 0.0, -1.255605697631836))

        high_basin = v.BoxVolume(center=vec3(0.7269397377967834, 0.30000001192092896, 0.2768973410129547), size=vec3(
            0.23952017724514008, 0.4261487126350403, 0.20288439095020294))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(
            0.2083175778388977, -1.0724661350250244, -2.020841360092163))

        box = v.BoxVolume(center=vec3(0.08181139081716537, 0.8128467798233032, 0.9683207273483276), size=vec3(
            0.13034817576408386, 0.03811933472752571, 0.08568242937326431))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(1.5890541076660156, -0.4044989049434662, 1.6848649978637695))

        box = v.BoxVolume(center=vec3(0.7036044597625732, 0.8770926594734192, 0.7076045870780945), size=vec3(
            0.033998481929302216, 0.20056499540805817, 0.1741935759782791))
        scene.add_fluid(box)
        scene.set_velocity(box, vec3(-0.6734871864318848, -2.3520143032073975, 0.5668471455574036))

        box = v.BoxVolume(
            center=vec3(0.12232201546430588,
                        0.79803466796875, 0.2184189260005951),
            size=vec3(0.21774353086948395, 0.11465508490800858, 0.1732807606458664))
        velo = vec3(-1.868835687637329, -2.04526948928833, -0.9829463362693787)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    elif id == 5:
        low_basin = v.BoxVolume(
            center=vec3(0.5, 0.0, 0.5),
            size=vec3(1.0, 0.5187215805053711, 1.0))
        velo = vec3(-1.2313419580459595, 0.0, 0.7753108739852905)
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, velo)

        high_basin = v.BoxVolume(center=vec3(0.27698397636413574, 0.30000001192092896, 0.4216991066932678), size=vec3(
            0.34849274158477783, 0.2475636750459671, 0.18932077288627625))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(
            2.3336164951324463, -1.7821544408798218, -2.446242570877075))

        box = v.BoxVolume(
            center=vec3(0.656795859336853, 0.8744591474533081,
                        0.06478223204612732),
            size=vec3(0.12068390846252441, 0.20077311992645264, 0.0729944258928299))
        velo = vec3(-2.1367764472961426, -2.2948267459869385, 1.2233843803405762)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.29505306482315063,
                        0.7604954838752747, 0.1863897144794464),
            size=vec3(0.18107445538043976, 0.10665582865476608, 0.20773538947105408))
        velo = vec3(1.877858281135559, -0.2517528533935547, 2.180844306945801)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    elif id == 6:
        low_basin = v.BoxVolume(
            center=vec3(0.5, 0.0, 0.5),
            size=vec3(1.0, 0.34942448139190674, 1.0))
        velo = vec3(-0.6067330837249756, 0.0, -1.3822189569473267)
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, velo)

        high_basin = v.BoxVolume(center=vec3(0.032917942851781845, 0.30000001192092896, 0.05717157572507858), size=vec3(
            0.20059509575366974, 0.2977581322193146, 0.25685209035873413))
        scene.add_fluid(high_basin)
        scene.set_velocity(
            high_basin, vec3(-0.9779403805732727, -1.6666839122772217, -0.9703491926193237))

        box = v.BoxVolume(
            center=vec3(0.8432857990264893, 0.7169050574302673,
                        0.5592822432518005),
            size=vec3(0.22429507970809937, 0.24883019924163818, 0.17685097455978394))
        velo = vec3(1.968638300895691, -1.6833604574203491, -1.325093150138855)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.5895683169364929, 0.8469659686088562,
                        0.4585215449333191),
            size=vec3(0.07316119223833084, 0.005376133136451244, 0.058106303215026855))
        velo = vec3(-0.5324712991714478, -0.02559758350253105, -1.0035665035247803)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.8326166868209839, 0.8913788199424744,
                        0.09664052724838257),
            size=vec3(0.14127902686595917, 0.24372991919517517, 0.23091773688793182))
        velo = vec3(-0.4065474271774292, -2.158449649810791, -2.3530707359313965)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    elif id == 7:
        low_basin = v.BoxVolume(
            center=vec3(0.5, 0.0, 0.5),
            size=vec3(1.0, 0.3127627372741699, 1.0))
        velo = vec3(0.2764146029949188, 0.0, 0.9838880300521851)
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, velo)

        high_basin = v.BoxVolume(center=vec3(0.7256541848182678, 0.30000001192092896, 0.7587382197380066), size=vec3(
            0.10348546504974365, 0.4684094786643982, 0.2304287552833557))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(
            1.4959570169448853, -0.7558392882347107, 0.25932949781417847))

        box = v.BoxVolume(
            center=vec3(0.8079801201820374, 0.7260262370109558,
                        0.8996701240539551),
            size=vec3(0.1506732851266861, 0.19234861433506012, 0.1592368632555008))
        velo = vec3(-2.434375524520874, -1.0714447498321533, -1.639163613319397)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.5895683169364929, 0.8469659686088562,
                        0.4585215449333191),
            size=vec3(0.07316119223833084, 0.005376133136451244, 0.058106303215026855))
        velo = vec3(-0.5324712991714478, -0.02559758350253105, -1.0035665035247803)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    elif id == 8:
        low_basin = v.BoxVolume(
            center=vec3(0.5, 0.0, 0.5),
            size=vec3(1.0, 0.3899596929550171, 1.0))
        velo = vec3(0.9648236036300659, 0.0, -0.2232801616191864)
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, velo)

        high_basin = v.BoxVolume(center=vec3(0.6966288685798645, 0.30000001192092896, 0.008833738043904305), size=vec3(
            0.2660067677497864, 0.4856005012989044, 0.3151404559612274))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(
            0.3095413148403168, -2.106633186340332, 0.041372936218976974))

        box = v.BoxVolume(
            center=vec3(0.35934045910835266,
                        0.7295283675193787, 0.6136208772659302),
            size=vec3(0.11720198392868042, 0.038290075957775116, 0.11583004891872406))
        velo = vec3(0.4675876498222351, -2.288301944732666, 2.3466708660125732)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.9810615181922913, 0.7501826286315918,
                        0.40847015380859375),
            size=vec3(0.22980323433876038, 0.2125728279352188, 0.09988266229629517))
        velo = vec3(-1.6714630126953125, -0.3550235331058502, -1.2862858772277832)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.6909376978874207, 0.8093129992485046,
                        0.8749793171882629),
            size=vec3(0.17899350821971893, 0.14774323999881744, 0.08175291866064072))
        velo = vec3(-0.8297789692878723, -2.351024866104126, -0.21956086158752441)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    elif id == 9:
        low_basin = v.BoxVolume(
            center=vec3(0.5, 0.0, 0.5),
            size=vec3(1.0, 0.39758017659187317, 1.0))
        velo = vec3(1.1275626420974731, 0.0, -0.45895346999168396)
        scene.add_fluid(low_basin)
        scene.set_velocity(low_basin, velo)

        high_basin = v.BoxVolume(center=vec3(0.6513636708259583, 0.30000001192092896, 0.5265719294548035), size=vec3(
            0.3874672055244446, 0.29336538910865784, 0.3927138149738312))
        scene.add_fluid(high_basin)
        scene.set_velocity(high_basin, vec3(-1.8544111251831055, -2.2138426303863525, -0.018335027620196342))

        box = v.BoxVolume(
            center=vec3(0.4429016709327698, 0.8089315891265869,
                        0.9250127673149109),
            size=vec3(0.09113061428070068, 0.18194787204265594, 0.2480262964963913))
        velo = vec3(-2.118415117263794, -2.372103452682495, 2.1927385330200195)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.9305604100227356, 0.7914493680000305,
                        0.3373067378997803),
            size=vec3(0.11360050737857819, 0.15121181309223175, 0.028010813519358635))
        velo = vec3(-0.40278196334838867, -2.106764793395996, -0.3708806335926056)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)

        box = v.BoxVolume(
            center=vec3(0.6779478192329407, 0.7241361141204834,
                        0.23077721893787384),
            size=vec3(0.016878172755241394, 0.20226021111011505, 0.17966283857822418))
        velo = vec3(-0.6755647659301758, -1.0057002305984497, 0.3956376314163208)
        scene.add_fluid(box)
        scene.set_velocity(box, velo)
    else:
        assert False, ("Benchmark with id {} is not supported".format(id))


#----------------------------------------------------------------------------------
def scene_selection(scene_type, scene, obstacles=False, meshes=False, benchmark=-1):
    if benchmark == -1:
        print("Scene Type: {}".format(scene_type))
        if scene_type == "liquid":
            initialize_scene(scene, simple=False, obstacles=obstacles, meshes=meshes)
        elif scene_type == "liquid_simple":
            initialize_scene(scene, simple=True, obstacles=obstacles, meshes=meshes)
        elif scene_type == "liquid_breaking_dam":
            initialize_breaking_dam(scene)
        elif scene_type == "liquid_anvil":
            initialize_anvil(scene)
        elif scene_type == "liquid_meshes":
            initialize_mesh_scene(scene)
        elif scene_type == "liquid_center_obstacle":
            initialize_center_obstacle(scene)
        elif scene_type == "liquid_border_obstacle":
            initialize_border_obstacle(scene)
        elif scene_type == "gas":
            initialize_gas_scene(scene)
        else:
            assert False, "Not supported scene type"
    else:
        initialize_benchmark(scene, benchmark)