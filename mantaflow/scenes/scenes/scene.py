#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta scene superclass
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
from manta import *
import scenes.volumes
import time

class Scene(object):
    #----------------------------------------------------------------------------------
    def __init__(self, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), merge_ghost_fluid=False, store_divergence_history=True, name="Scene", show_gui=True, pause_on_start=False):
        self.resolution = resolution
        self.dimension = dimension
        self.timestep = timestep
        self.boundary = boundary
        self.gravity = gravity
        self.name = name
        self.show_gui = show_gui
        self.pause_on_start = pause_on_start
        self.merge_ghost_fluid = merge_ghost_fluid
        self.store_divergence_history = store_divergence_history
        self.divergence_history = []
        self.profile_history_timestep = []
        self.profile_history_solve = []
        self._init_solver()
        self._init_grids()

    #----------------------------------------------------------------------------------
    def get_settings(self):
        settings = {
            "name": self.name,
            "resolution": self.resolution,
            "dimension": self.dimension,
            "timestep": self.timestep,
            #"gravity": self.gravity,
            "boundary": self.boundary,
            #"grid_size": self._grid_size,
            "merge_ghost_fluid": self.merge_ghost_fluid,
            "store_divergence_history": self.store_divergence_history,
        }
        return settings

    #----------------------------------------------------------------------------------
    def add_obstacle(self, volume):
        vol_ls = volume.computeLevelset(self.solver)
        self.phi_obs.join(vol_ls)

    #----------------------------------------------------------------------------------
    def add_mesh(self, volume):
        print("Deprecation Warning: The use of add_mesh on a scene is obsolete. Use add_fluid instead")
        # meshes should act like other volumes, so that they can be added as fluid OR obstacle
        self.add_fluid(volume)

    #----------------------------------------------------------------------------------
    def add_fluid(self, volume):
        vol_ls = volume.computeLevelset(self.solver)
        self.phi_fluid.join(vol_ls)

    #----------------------------------------------------------------------------------
    def set_velocity(self, volume, velocity):
        if self.dimension == 2:
            velocity.z = 0.0
        volume.applyToGrid(solver=self.solver, grid=self.vel, value=velocity)

    #----------------------------------------------------------------------------------
    def add_sink(self, volume):
        vol_ls = volume.computeLevelset(self.solver)
        self.phi_sink.join(vol_ls)

    #----------------------------------------------------------------------------------
    def add_source(self, volume):
        vol_ls = volume.computeLevelset(self.solver)
        self.phi_source.join(vol_ls)

    #----------------------------------------------------------------------------------
    def simulate(self, num_steps=500, on_simulation_step=None):
        self.divergence_history = []
        self.profile_history_timestep = []
        self.profile_history_solve = []
        self._create_scene()

        last_frame = -1
        while self.solver.frame < num_steps:
            #maxVel = self.vel.getMax()
            #self.solver.adaptTimestep( maxVel )
            print("\r{} Step {:3d}, Time {:3.3f}, dt {:0.3f}".format(self.name, self.solver.frame + 1, self.solver.timeTotal, self.solver.timestep), end='\r')

            # Execute simulation step
            simulation_start_time = time.time()
            self._compute_simulation_step()
            simulation_end_time = time.time()
            self.profile_history_timestep.append(simulation_end_time - simulation_start_time)

            if self.show_gui and self.dimension > 2:
                #self.phi_fluid.setBound(0.5, 0) # optionally, close sides
                self.phi_fluid.createMesh(self.debugmesh)

            self.solver.step()
            if callable(on_simulation_step) and (last_frame != self.solver.frame):
                assert on_simulation_step.__code__.co_argcount == 2, "on_simulation_step must be a function with 2 arguments (scene and timestep)!"
                on_simulation_step(self, self.solver.frame-1) # solver already progressed one frame
            last_frame = self.solver.frame

        self._reset()

        return self.profile_history_timestep, self.profile_history_solve

    #----------------------------------------------------------------------------------
    def _compute_simulation_step(self):
        self._advect(ls_order=1)
        self._enforce_boundaries(distance=2)

        solve_start_time = time.time()
        self._solve_pressure()
        solve_end_time = time.time()
        self.profile_history_solve.append(solve_end_time - solve_start_time)

        self._update_velocities()

    #----------------------------------------------------------------------------------
    def _init_solver(self):
        self._grid_size = vec3(self.resolution, self.resolution, self.resolution)
        if (self.dimension == 2):
            self._grid_size.z = 1

        self.solver = Solver(name=self.name, gridSize=self._grid_size, dim=self.dimension)
        self.solver.timestep = self.timestep

        # adaptive timestepping settings
        self.solver.frameLength = self.timestep
        self.solver.timestepMin = self.timestep #0.25 *
        self.solver.timestepMax = self.timestep #4.0  *
        self.solver.cfl         = 2.5

        setDebugLevel(0)

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        self.flags = self.solver.create(FlagGrid, name="Flags")
        self.vel = self.solver.create(MACGrid, name="Velocity")
        self.pressure = self.solver.create(RealGrid, name="Pressure")
        self.pressure_raw = self.solver.create(RealGrid, name="Pressure Raw")
        self.phi_fluid = self.solver.create(LevelsetGrid, name="Fluid")
        self.phi_obs = self.solver.create(LevelsetGrid, name="Obstacles")
        self.phi_sink = self.solver.create(LevelsetGrid, name="Sinks")
        self.phi_source = self.solver.create(LevelsetGrid, name="Sources")
        self.fractions = self.solver.create(MACGrid, name="Fractions")
        if self.store_divergence_history:
            self.divergence = self.solver.create(RealGrid, name="Divergence")
        
        if self.show_gui and self.dimension > 2:
            self.debugmesh = self.solver.create(Mesh)
            #self.ppm = self.solver.create(LevelsetGrid, name="PPM")
        self.flags.initDomain(boundaryWidth=self.boundary, phiWalls=self.phi_obs)

    #----------------------------------------------------------------------------------
    def _create_scene(self):
        self.phi_fluid.subtract(self.phi_obs)
        updateFractions(flags=self.flags, phiObs=self.phi_obs, fractions=self.fractions, boundaryWidth=self.boundary)
        setObstacleFlags(flags=self.flags, phiObs=self.phi_obs, fractions=self.fractions)

        if self.show_gui and "_gui" not in self.__dict__:
            self._gui = Gui()
            self._gui.show(True)
            self._gui.setCamRot(40.0,0,0)
            self._gui.setCamPos(0,0,-1.5)
            self._gui.setPlane(2)
            if self.pause_on_start:
                self._gui.pause()
        
        extrapolateLsSimple(phi=self.phi_fluid, distance=5)
        self.flags.updateFromLevelset(self.phi_fluid)

    #----------------------------------------------------------------------------------
    def _reset(self):
        self.flags.setConst(0)
        self.vel.setConst(vec3(0,0,0))
        self.pressure.setConst(0)
        self.phi_fluid.setConst(0)
        self.phi_obs.setConst(0)
        self.phi_sink.setConst(0)
        self.phi_source.setConst(0)
        self.fractions.setConst(vec3(0,0,0))
        self.flags.initDomain(boundaryWidth=self.boundary, phiWalls=self.phi_obs)
        
        self.solver.frame = 0

#==================================================================================
class NaiveScene(Scene):
    #----------------------------------------------------------------------------------
    def _init_solver(self):
        super(NaiveScene,self)._init_solver()

    #==================================================================================
    # SIMULATION
    #----------------------------------------------------------------------------------
    def _advect(self, extrapol_dist=3, ls_order=2):
        # extrapolate the grids into empty cells
        self.phi_fluid.reinitMarching(flags=self.flags, velTransport=self.vel, maxTime=32.0)

        # advect the levelset
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.phi_fluid, order=ls_order)

        # source & sink
        self.phi_fluid.subtract(self.phi_sink)
        #self.phi_fluid.join(self.phi_source)

        # enforce boundaries
        self.phi_fluid.setBoundNeumann(self.boundary)
        self.flags.updateFromLevelset(self.phi_fluid)

        # velocity self-advection
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.vel, order=2, boundaryWidth=self.boundary)
        addGravity(flags=self.flags, vel=self.vel, gravity=self.gravity)

    #----------------------------------------------------------------------------------
    def _enforce_boundaries(self, distance):
        # enforce boundaries
        setWallBcs(flags=self.flags, vel=self.vel, fractions=self.fractions, phiObs=self.phi_obs)

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=10, accuracy=5e-5):
        solvePressureOnly(flags=self.flags, fractions=self.fractions, vel=self.vel, pressure=self.pressure, cgMaxIterFac=max_iter_fac, cgAccuracy=accuracy, phi=self.phi_fluid, mergeGhostFluid=self.merge_ghost_fluid)

        smoothPressureCorner(flags=self.flags, pressure=self.pressure, boundary=self.boundary)
        
        if self.boundary>0:
            self.pressure.setBoundNeumann(self.boundary-1)
        self.pressure_raw.copyFrom(self.pressure)

    #----------------------------------------------------------------------------------
    def _update_velocities(self):
        if self.merge_ghost_fluid:
            correctVelocities(vel=self.vel, pressure=self.pressure, flags=self.flags, phi=self.phi_fluid, mergeGhostFluid=self.merge_ghost_fluid)
        else:
            correctVelocities(vel=self.vel, pressure=self.pressure, flags=self.flags)
        if self.store_divergence_history:
            self.divergence_history.append(getDivergence(self.divergence, self.vel, self.flags))
        self._enforce_boundaries(distance=4)

#==================================================================================
class FLIPScene(Scene):
    #----------------------------------------------------------------------------------
    def set_velocity(self, volume, velocity):
        super(FLIPScene,self).set_velocity(volume, velocity)
        mapGridToPartsVec3(source=self.vel, parts=self.pp, target=self.pVel)

    #----------------------------------------------------------------------------------
    def add_sink(self, volume):
        print("WARNING - sinks not yet supported for FLIP scene")
        #vol_ls = volume.shape(self.solver).computeLevelset()
        #self.phi_sink.join(vol_ls)

    #----------------------------------------------------------------------------------
    def add_source(self, volume):
        print("WARNING - sources not yet supported for FLIP scene")
        vol_ls = volume.computeLevelset(self.solver)
        self.phi_source.join(vol_ls)

    #----------------------------------------------------------------------------------
    def _init_solver(self):
        super(FLIPScene,self)._init_solver()

    #----------------------------------------------------------------------------------
    def _init_grids(self):
        super(FLIPScene,self)._init_grids()

        # new flip
        self.vel_org = self.solver.create(MACGrid, name="VelOrg")
        self.vel_parts = self.solver.create(MACGrid, name="VelParts")
        self.tmp_vec3 = self.solver.create(MACGrid, name="temp_vec3")
        self.phi_parts = self.solver.create(LevelsetGrid, name="PhiParticles")
        self.narrow_band = 3 

        self.pp     = self.solver.create(BasicParticleSystem) 
        self.pVel   = self.pp.create(PdataVec3) 
        self.pindex = self.solver.create(ParticleIndexSystem) 
        self.gpi    = self.solver.create(IntGrid)

    #----------------------------------------------------------------------------------
    def _create_scene(self):
        super(FLIPScene, self)._create_scene()

        # extrapolate velocities from 1-cell inside towards empty region for particles
        self.phi_fluid.addConst( 1.)
        self.flags.updateFromLevelset(self.phi_fluid)
        self.phi_fluid.addConst(-1.)
        extrapolateMACSimple( flags=self.flags, vel=self.vel , distance=3, intoObs=True )
        
        sampleLevelsetWithParticles(phi=self.phi_fluid, flags=self.flags, parts=self.pp, discretization=2, randomness=0.05)
        mapGridToPartsVec3(source=self.vel, parts=self.pp, target=self.pVel )

    #----------------------------------------------------------------------------------
    def _reset(self):
        super(FLIPScene, self)._reset()

        self.vel_org.setConst(vec3(0,0,0))
        self.vel_parts.setConst(vec3(0,0,0))
        self.tmp_vec3.setConst(vec3(0,0,0))
        self.phi_parts.setConst(0) 

        self.pp.clear()
        self.pVel.setConst(vec3(0,0,0)) # not really necessary...
        self.pindex.clear()
        self.gpi.setConst(0)

    #==================================================================================
    # SIMULATION
    #----------------------------------------------------------------------------------
    def _advect(self, extrapol_dist=3, ls_order=2):
        # extrapolate the grids into empty cells
        #self.phi_fluid.reinitMarching(flags=self.flags, velTransport=self.vel, maxTime=4.0)

        # ? self.pp.advectInGrid(flags=self.flags, vel=self.vel, integrationMode=IntRK4, deleteInObstacle=False, stopInObstacle=False )  # check - IntRK4 lost from manta?
        self.pp.advectInGrid(flags=self.flags, vel=self.vel, integrationMode=2, deleteInObstacle=False, stopInObstacle=False )
        pushOutofObs( parts=self.pp, flags=self.flags, phiObs=self.phi_obs )

        # make sure nothings sticks to the top... (helper in test.cpp)
        deleteTopParts( parts=self.pp, phi=self.phi_fluid, maxHeight=self.resolution-1-(self.boundary+2) ) # needs 2 more to make sure it's out of the setBoundNeumann range 

        # advect the levelset
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.phi_fluid, order=ls_order) 

        # velocity self-advection
        advectSemiLagrange(flags=self.flags, vel=self.vel, grid=self.vel, order=2, boundaryWidth=self.boundary)

        # particle SDF
        gridParticleIndex( parts=self.pp , flags=self.flags, indexSys=self.pindex, index=self.gpi )
        unionParticleLevelset( self.pp, self.pindex, self.flags, self.gpi, self.phi_parts )

        # source & sink , not yet working...
        #self.phi_fluid.subtract(self.phi_sink) 
        self.phi_fluid.join(self.phi_source)

        # combine level set of particles with grid level set
        self.phi_fluid.addConst(1.) # shrink slightly
        self.phi_fluid.join( self.phi_parts )
        extrapolateLsSimple(phi=self.phi_fluid, distance=self.narrow_band+2, inside=True ) 
        extrapolateLsSimple(phi=self.phi_fluid, distance=3 )

        # enforce boundaries
        self.phi_fluid.setBoundNeumann(self.boundary-1 if self.boundary>0 else 0) # 1 cell less...
        self.flags.updateFromLevelset(self.phi_fluid)

        # combine particles velocities with advected grid velocities
        mapPartsToMAC(vel=self.vel_parts, flags=self.flags, velOld=self.vel_org, parts=self.pp, partVel=self.pVel, weight=self.tmp_vec3)
        extrapolateMACFromWeight( vel=self.vel_parts , distance=2, weight=self.tmp_vec3 )
        combineGridVel(vel=self.vel_parts, weight=self.tmp_vec3 , combineVel=self.vel, phi=self.phi_fluid, narrowBand=(self.narrow_band-1), thresh=0)
        self.vel_org.copyFrom(self.vel)

        addGravity(flags=self.flags, vel=self.vel, gravity=self.gravity)

    #----------------------------------------------------------------------------------
    def _enforce_boundaries(self, distance):
        extrapolateMACSimple( flags=self.flags, vel=self.vel , distance=distance, intoObs=True )
        setWallBcs(flags=self.flags, vel=self.vel, fractions=self.fractions, phiObs=self.phi_obs)

    #----------------------------------------------------------------------------------
    def _solve_pressure(self, max_iter_fac=2, accuracy=5e-4):
        solvePressureOnly(flags=self.flags, fractions=self.fractions, vel=self.vel, pressure=self.pressure, cgMaxIterFac=max_iter_fac, cgAccuracy=accuracy, phi=self.phi_fluid, mergeGhostFluid=self.merge_ghost_fluid)

        smoothPressureCorner(flags=self.flags, pressure=self.pressure, boundary=self.boundary)

        # remove pressure discontinuity at boundary - note: only outer boundary, does not influence the required pressure gradients in any way
        if self.boundary>0:
            self.pressure.setBoundNeumann(self.boundary-1)
        self.pressure_raw.copyFrom(self.pressure)

    #----------------------------------------------------------------------------------
    def _update_velocities(self):
        if self.merge_ghost_fluid:
            correctVelocities(vel=self.vel, pressure=self.pressure, flags=self.flags, phi=self.phi_fluid, mergeGhostFluid=self.merge_ghost_fluid)
        else:
            correctVelocities(vel=self.vel, pressure=self.pressure, flags=self.flags)

        if self.store_divergence_history:
            self.divergence_history.append(getDivergence(self.divergence, self.vel, self.flags))

        self._enforce_boundaries(4)

        minParticles  = pow(2,self.dimension)
        self.pVel.setSource( self.vel, isMAC=True )
        adjustNumber( parts=self.pp, vel=self.vel, flags=self.flags, minParticles=1*minParticles, maxParticles=2*minParticles, phi=self.phi_fluid, exclude=self.phi_obs, narrowBand=self.narrow_band ) 
        flipVelocityUpdate(vel=self.vel, velOld=self.vel_org, flags=self.flags, parts=self.pp, partVel=self.pVel, flipRatio=0.97 )

