#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta display scene
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

from scenes.scene import FLIPScene
from manta import *
import numpy as np
import json
import os
import time

class DisplayScene(FLIPScene):
    #----------------------------------------------------------------------------------
    def __init__(self, resolution=64, dimension=2, time_to_wait=0.0, skip_steps=1, name="DisplayScene", show_gui=True):
        super(DisplayScene,self).__init__(resolution=resolution, dimension=dimension, timestep=0.1, name=name, show_gui=show_gui)
        self.display_grids = {}
        self.time_to_wait = time_to_wait
        self.skip_steps = skip_steps
        self.file_num = 0

    #----------------------------------------------------------------------------------
    def simulate(self, num_steps=500, on_grid_copy_step=None, on_solver_step=None):
        self._create_scene()

        while self.solver.frame < num_steps:
            print("\r{} Step {:3d}, Time {:3.3f}, dt {:0.3f}".format(self.name, self.solver.frame + 1, self.solver.timeTotal, self.solver.timestep), end='\r')

            # some steps should not be displayed
            if self.solver.frame % self.skip_steps != 0:
                print("Skipping {}".format(self.solver.frame))
                self.solver.step()
                continue

            # otherwise continue as normal
            if self.time_to_wait > 0.0:
                time.sleep(self.time_to_wait)

            # read grids from data set
            if callable(on_grid_copy_step):
                assert on_grid_copy_step.__code__.co_argcount == 2, "on_grid_copy_step must be a function with 2 arguments (scene and timestep)!"
                on_grid_copy_step(self, self.solver.frame)

            if self.show_gui and self.dimension > 2:
                #self.phi_fluid.setBound(0.5, 0) # optionally, close sides
                self.phi_fluid.createMesh(self.debugmesh)

            self.solver.step()

            # store screenshots
            if callable(on_solver_step):
                assert on_solver_step.__code__.co_argcount == 2, "on_solver_step must be a function with 2 arguments (scene and timestep)!"
                on_solver_step(self, self.solver.frame-1) # solver already advanced one frame
                self.file_num += 1

        self._reset()

        return [], []
