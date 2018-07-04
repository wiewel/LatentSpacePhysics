#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta gas scene
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

from manta import *

import os
import pathlib
import numpy
import argparse
import datetime
import time
from random import randint, seed
from scenes.scene import Scene
import scenes.volumes 

def random_vec3(vmin=[-1,-1,-1], vmax=[1,1,1]):
    return vec3(
        numpy.random.uniform(low=vmin[0], high=vmax[0]),
        numpy.random.uniform(low=vmin[1], high=vmax[1]),
        numpy.random.uniform(low=vmin[2], high=vmax[2])
    )

def random_vec3s(vmin=-1, vmax=1): # scalar params
    return vec3(
        numpy.random.uniform(low=vmin, high=vmax),
        numpy.random.uniform(low=vmin, high=vmax),
        numpy.random.uniform(low=vmin, high=vmax)
    )


class SmokeScene(Scene):
    file_num = 0
    open_bound = True
    sources = []
    source_strengths = []

    #----------------------------------------------------------------------------------
    def set_velocity(self, volume, velocity):
        super(SmokeScene,self).set_velocity(volume, velocity)

    #----------------------------------------------------------------------------------
    def add_sink(self, volume):
        print("WARNING - sinks not yet supported for smoke scene")

    #----------------------------------------------------------------------------------
    # sources used as smoke inflow in the following
    def add_source(self, volume):
        shape = volume.shape(self.solver)
        self.sources.append(shape)

    #----------------------------------------------------------------------------------
    # this is kind of a hack, since for smoke sources are much more desirable than just adding a fluid once
    def add_fluid(self, volume):
        self.add_source(volume)

    #----------------------------------------------------------------------------------
    def _init_solver(self):
        super(SmokeScene,self)._init_solver()
        self.max_iter_fac = 2
        self.accuracy =5e-4

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(SmokeScene,self)._init_grids()

        self.density = self.solver.create(RealGrid, name="Density")
        self.pressure_static = self.solver.create(RealGrid, name="StatPressure_dummy_") # not used for smoke

        print("noise1")
        noise = self.solver.create(NoiseField, loadFromFile=True)
        noise.posScale = vec3(40) * numpy.random.uniform(low=0.25, high=1.)
        noise.posOffset = random_vec3s( vmin=0. ) * 100.
        noise.clamp = True
        noise.clampNeg = 0
        noise.clampPos = 1.
        noise.valOffset = 0.15
        noise.timeAnim  = 0.4 * numpy.random.uniform(low=0.2, high=1.)
        self.noise = noise

        self.source_strengths = []
        for i in range(100): # some more for safety
            self.source_strengths.append( numpy.random.uniform(low=0.5, high=1.) )

    #----------------------------------------------------------------------------------
    def _create_scene(self):
        super(SmokeScene, self)._create_scene()
        self.flags.initDomain(boundaryWidth=self.boundary) 
        self.flags.fillGrid()
        if self.open_bound:
            setOpenBound(self.flags, self.boundary, 'yY', 16 | 4) # FlagOutflow|FlagEmpty) 

        if self.show_gui:
            self._gui.setPlane( self.resolution // 2 )

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(SmokeScene, self)._reset()

        # dont reset! multiple sims written into single file with increasing index...
        #self.file_num = 0 

        self.sources = [] 
        self.density.setConst(0) 
        self.vel.setConst(vec3(0)) 

    #----------------------------------------------------------------------------------
    def _compute_simulation_step(self):
        # Add source
        # randomize noise offset , note - sources are turned off earlier, the more there are
        for i in range(len(self.sources)):
            if self.solver.frame<i*(100./len(self.sources)):
                src,sstr = self.sources[i], self.source_strengths[i]
                densityInflow(flags=self.flags, density=self.density, noise=self.noise, shape=src, scale=2.0*sstr, sigma=0.5)

        super(SmokeScene, self)._compute_simulation_step()

    #==================================================================================
    # SIMULATION
    #----------------------------------------------------------------------------------
    def _advect(self, extrapol_dist=3, ls_order=2):
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.density, order=2, clampMode=2)
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.vel    , order=2, clampMode=2, openBounds=self.open_bound, boundaryWidth=self.boundary)

        vorticityConfinement( vel=self.vel, flags=self.flags, strength=0.1 )
        addBuoyancy(density=self.density, vel=self.vel, gravity=0.2*self.gravity, flags=self.flags)

    #----------------------------------------------------------------------------------
    def _solve_pressure(self):
        solvePressureOnly(flags=self.flags, vel=self.vel, pressure=self.pressure, cgMaxIterFac=self.max_iter_fac, cgAccuracy=self.accuracy)
        if self.boundary>0:
            self.pressure.setBoundNeumann(self.boundary-1)

    #----------------------------------------------------------------------------------
    def _enforce_boundaries(self, distance):
        setWallBcs(flags=self.flags, vel=self.vel)

    #----------------------------------------------------------------------------------
    def _update_velocities(self):
        correctVelocities(vel=self.vel, pressure=self.pressure, flags=self.flags)

        if self.store_divergence_history:
            self.divergence_history.append(getDivergence(self.divergence, self.vel, self.flags))

        self.vel.setBoundNeumann(self.boundary)
