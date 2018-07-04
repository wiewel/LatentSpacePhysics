#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta volume classes to be used in scene creation
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
import numpy

class Volume():
    def shape(self, solver):
        pass

    def applyToGrid(self, solver, grid, value):
        self.shape(solver).applyToGrid(grid=grid, value=value)

    def computeLevelset(self, solver):
        return self.shape(solver).computeLevelset()

class CylinderVolume(Volume):
    def __init__(self, center, radius, height):
        self._center = center
        self._radius = radius
        self._height = height
    
    def shape(self, solver):
        return Cylinder(parent=solver, center=solver.getGridSize() * self._center, radius=solver.getGridSize().x * self._radius, z=solver.getGridSize().y * self._height)

class SphereVolume(Volume):
    def __init__(self, center, radius):
        self._center = center
        self._radius = radius

    def shape(self, solver):
        return Sphere(parent=solver, center=solver.getGridSize() * self._center, radius=solver.getGridSize().x * self._radius)

class MeshVolume(Volume):
    def __init__(self, file_path, center, scale, rotation):
        self._center = center
        self._scale = scale
        self._rotation = rotation
        self._file_path = file_path
        self._mesh = None

    def shape(self, solver):
        if self._mesh is None:
            self._mesh = solver.create(Mesh)
            self._mesh.load(self._file_path)
            self._mesh.scale( (solver.getGridSize() / 3.0) * self._scale )
            self._mesh.rotate( self._rotation )
            self._mesh.offset( solver.getGridSize() * self._center )
        return self._mesh

    def applyToGrid(self, solver, grid, value):
        self.shape(solver).applyMeshToGrid(grid=grid, value=value)
    
    def computeLevelset(self, solver):
        return self.shape(solver).getLevelset(sigma=2.0)

class BoxVolume(Volume):
    def __init__(self, p0=None, p1=None, center=None, size=None):
        if p0 is not None and p1 is not None:
            self._p0 = p0
            self._p1 = p1
        elif center is not None and size is not None:
            self._center = center
            self._size = size
        else:
            raise ValueError("Arguments must either be p0 and p1 OR center and size")

    def shape(self, solver):
        if "_center" in self.__dict__ and "_size" in self.__dict__:
            shape = Box(parent=solver, center=solver.getGridSize() * self._center, size=solver.getGridSize() * self._size)
        else:
            shape = Box(parent=solver, p0=solver.getGridSize() * self._p0, p1=solver.getGridSize() * self._p1)
        return shape

def random_vec3(low=[0,0,0], high=[1,1,1]):
    return vec3(
        numpy.random.uniform(low=low[0], high=high[0]),
        numpy.random.uniform(low=low[1], high=high[1]),
        numpy.random.uniform(low=low[2], high=high[2])
    )

def random_box(center_min=[0,0,0], center_max=[1,1,1], size_min=[0,0,0], size_max=[1,1,1]):
    size = random_vec3(size_min, size_max)
    center = random_vec3(center_min, center_max)
    return BoxVolume(center=center, size=size)

def random_velocity(vel_min=[-1,-1,-1], vel_max=[1,0,1]):
    return random_vec3(vel_min, vel_max)
