#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta encode/decode scene to evaluate autoencoder performance
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

from scenes.scene import FLIPScene, NaiveScene
from manta import *
import numpy as np
import keras
import json
import os
from scipy import ndimage


class EncodeDecodeScene(FLIPScene):
    #----------------------------------------------------------------------------------
    def __init__(self, project_directory, warmup_steps=25, resolution=64, timestep=1.0, boundary=1, interval=1, gravity=vec3(0, -0.01, 0), name="EncodeDecodeScene", show_gui=True):
        self.warmup_steps = warmup_steps
        self.interval = interval
        self.encoder = keras.models.load_model(project_directory + "/encoder.h5")
        self.encoder.summary()

        if len(self.encoder.inputs[0].shape) == 4: # e.g. (?, 64, 64, 1)
            # 2D
            dimension = 2
        else: # e.g. (?, 64, 64, 64, 1)
            # 3D
            dimension = 3

        self.decoder = keras.models.load_model(project_directory + "/decoder.h5")
        if os.path.isfile(project_directory + "/description_autoencoder"):
                with open(project_directory + "/description_autoencoder", 'r') as f:
                    self.description = json.load(f)

        super(EncodeDecodeScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, name=name, show_gui=show_gui)

class VelocityEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(VelocityEncodeDecodeScene,self)._init_grids()
        if self.dimension == 2:
            self.np_velocities = np.empty(shape=[1, self.resolution, self.resolution, 3], order='C')
        else:
            self.np_velocities = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 3], order='C')

    #----------------------------------------------------------------------------------
    def _update_velocities(self):
        super(VelocityEncodeDecodeScene, self)._update_velocities()
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:
            copyGridToArrayMAC(self.vel, self.np_velocities)

            self.np_velocities /= self.description["dataset"]["velocity_normalization_factor"]

            encoded = self.encoder.predict(x=self.np_velocities, batch_size=1)
            self.np_velocities = self.decoder.predict(x=encoded, batch_size=1)

            self.np_velocities *= self.description["dataset"]["velocity_normalization_factor"]

            copyArrayToGridMAC(self.np_velocities, self.vel)

class TotalPressureEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(TotalPressureEncodeDecodeScene, self)._init_grids()
        if self.dimension == 2:
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        super(TotalPressureEncodeDecodeScene, self)._solve_pressure(max_iter_fac, accuracy)
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:
            copyGridToArrayReal(self.pressure, self.np_pressure)

            self.np_pressure /= self.description["dataset"]["pressure_normalization_factor"]
            encoded = self.encoder.predict(x=self.np_pressure, batch_size=1)
            self.np_pressure = self.decoder.predict(x=encoded, batch_size=1)
            self.np_pressure *= self.description["dataset"]["pressure_normalization_factor"]

            copyArrayToGridReal(self.np_pressure, self.pressure)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            # if self.boundary>0:
            #     self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)

            # Jacobi Step
            smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
            smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)

class SplitPressureEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(SplitPressureEncodeDecodeScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")
        if self.dimension == 2:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')  
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')        
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        super(SplitPressureEncodeDecodeScene, self)._solve_pressure(max_iter_fac, accuracy)
        separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6.0 * self.resolution / 64.0)
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:

            copyGridToArrayReal(self.pressure_static, self.np_static_pressure)
            copyGridToArrayReal(self.pressure_dynamic, self.np_dynamic_pressure)
            
            self.np_static_pressure /= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure /= self.description["dataset"]["pressure_dynamic_normalization_factor"]
            
            encoded = self.encoder.predict(x=[self.np_static_pressure, self.np_dynamic_pressure], batch_size=1)
            [self.np_static_pressure, self.np_dynamic_pressure] = self.decoder.predict(x=encoded, batch_size=1)
            
            self.np_static_pressure *= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure *= self.description["dataset"]["pressure_dynamic_normalization_factor"]

            self.np_pressure = self.np_static_pressure + self.np_dynamic_pressure
            copyArrayToGridReal(self.np_pressure, self.pressure)
            copyArrayToGridReal(self.np_static_pressure, self.pressure_static)
            copyArrayToGridReal(self.np_dynamic_pressure, self.pressure_dynamic)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            if self.boundary>0:
                self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)

            # Jacobi Step
            smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
            smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)

class SplitDynamicPressureEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(SplitDynamicPressureEncodeDecodeScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")
        if self.dimension == 2:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')  
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')        
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        super(SplitDynamicPressureEncodeDecodeScene, self)._solve_pressure(max_iter_fac, accuracy)
        separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6.0 * self.resolution / 64.0)
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:

            copyGridToArrayReal(self.pressure_static, self.np_static_pressure)
            copyGridToArrayReal(self.pressure_dynamic, self.np_dynamic_pressure)
            
            self.np_static_pressure /= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure /= self.description["dataset"]["pressure_dynamic_normalization_factor"]
            
            encoded = self.encoder.predict(x=[self.np_static_pressure, self.np_dynamic_pressure], batch_size=1)
            _, self.np_dynamic_pressure = self.decoder.predict(x=encoded, batch_size=1)
            
            self.np_static_pressure *= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure *= self.description["dataset"]["pressure_dynamic_normalization_factor"]

            self.np_pressure = self.np_static_pressure + self.np_dynamic_pressure
            copyArrayToGridReal(self.np_pressure, self.pressure)
            copyArrayToGridReal(self.np_static_pressure, self.pressure_static)
            copyArrayToGridReal(self.np_dynamic_pressure, self.pressure_dynamic)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)

            # Jacobi Step
            smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
            smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)

class DynamicPressureEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(DynamicPressureEncodeDecodeScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")
        if self.dimension == 2:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')  
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')
            self.np_dynamic_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')        
            self.np_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        super(DynamicPressureEncodeDecodeScene, self)._solve_pressure(max_iter_fac, accuracy)
        separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6.0 * self.resolution / 64.0)
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:

            copyGridToArrayReal(self.pressure_static, self.np_static_pressure)
            copyGridToArrayReal(self.pressure_dynamic, self.np_dynamic_pressure)
            
            self.np_static_pressure /= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure /= self.description["dataset"]["pressure_dynamic_normalization_factor"]
            
            encoded = self.encoder.predict(x=[self.np_static_pressure, self.np_dynamic_pressure], batch_size=1)
            self.np_dynamic_pressure = self.decoder.predict(x=encoded, batch_size=1)
            
            self.np_static_pressure *= self.description["dataset"]["pressure_static_normalization_factor"]
            self.np_dynamic_pressure *= self.description["dataset"]["pressure_dynamic_normalization_factor"]

            self.np_pressure = self.np_static_pressure + self.np_dynamic_pressure
            copyArrayToGridReal(self.np_pressure, self.pressure)
            copyArrayToGridReal(self.np_static_pressure, self.pressure_static)
            copyArrayToGridReal(self.np_dynamic_pressure, self.pressure_dynamic)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)
            if self.boundary>0:
                self.pressure_raw.setBoundNeumann(self.boundary-1)

            # pressure boundary conditions
            clearNonfluid(self.pressure, self.flags)

            # Jacobi Step
            smooth_resolution_kernel = (self.resolution / 64.0) * 2.0
            smoothenSurface(self.pressure, self.phi_fluid, iterations=int(smooth_resolution_kernel), surface_width=smooth_resolution_kernel)

class NaiveEncodeDecodeScene(EncodeDecodeScene):
    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(NaiveEncodeDecodeScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")
        if self.dimension == 2:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, 1], order='C')
        else:
            self.np_static_pressure = np.empty(shape=[1, self.resolution, self.resolution, self.resolution, 1], order='C')

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        super(NaiveEncodeDecodeScene, self)._solve_pressure(max_iter_fac, accuracy)
        if self.solver.frame >= self.warmup_steps and self.interval != 0 and self.solver.frame % self.interval != 0:
            separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6.0 * self.resolution / 64.0)

            copyGridToArrayReal(self.pressure_static, self.np_static_pressure)

            copyArrayToGridReal(self.np_static_pressure, self.pressure)
            copyArrayToGridReal(self.np_static_pressure, self.pressure_static)

            # Store raw pressure without post processing
            self.pressure_raw.copyFrom(self.pressure)