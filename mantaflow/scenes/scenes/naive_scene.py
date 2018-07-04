#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta naive comparison scenes
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

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class NaiveStaticPressureScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), predict_dynamic_only=False, store_divergence_history=True, name="NaiveStaticPressureScene", show_gui=True):
        super(NaiveStaticPressureScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            computeStaticPressure(self.phi_fluid, self.flags, self.gravity, self.pressure, 6)
        else:
            # simulate frame
            super(NaiveStaticPressureScene,self)._solve_pressure()



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class NaiveDynamicPressureScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), predict_dynamic_only=False, store_divergence_history=True, name="NaiveDynamicPressureScene", show_gui=True):
        super(NaiveDynamicPressureScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(NaiveDynamicPressureScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            computeStaticPressure(self.phi_fluid, self.flags, self.gravity, self.pressure_static, 6)
            combinePressure(self.pressure_static, self.pressure_dynamic, self.pressure)
        else:
            # simulate frame
            super(NaiveDynamicPressureScene,self)._solve_pressure()
            separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6)



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
class NaiveVelocityScene(FLIPScene):
    def __init__(self, project_directory, prediction_interval=2, warmup_steps=25, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), predict_dynamic_only=False, store_divergence_history=True, name="NaiveVelocityScene", show_gui=True):
        super(NaiveVelocityScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=False, store_divergence_history=store_divergence_history, name=name, show_gui=show_gui)
        self.prediction_interval = prediction_interval
        self.warmup_steps = warmup_steps

    #----------------------------------------------------------------------------------
    # After this call the pressure was solved for this frame and the velocities are corrected
    # Additionally the current pressure field is encoded and added to the prediction history
    def _solve_pressure(self):
        if self.solver.frame >= self.warmup_steps and self.prediction_interval != 0 and self.solver.frame % self.prediction_interval != 0:
            # keep old velocity
            pass
        else:
            # simulate frame
            super(NaiveVelocityScene,self)._solve_pressure()

