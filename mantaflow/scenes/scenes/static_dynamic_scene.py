#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# manta separate static/dynamic pressure scene
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

class StaticDynamicScene(FLIPScene):
    file_num = 0
    def __init__(self, resolution=64, dimension=2, timestep=1.0, boundary=1, gravity=vec3(0, -0.01, 0), merge_ghost_fluid=False, name="StaticDynamicScene", show_gui=True, pause_on_start=False):
        super(StaticDynamicScene,self).__init__(resolution=resolution, dimension=dimension, timestep=timestep, boundary=boundary, gravity=gravity, merge_ghost_fluid=merge_ghost_fluid, name=name, show_gui=show_gui, pause_on_start=pause_on_start)

    def _init_grids(self):
        super(StaticDynamicScene,self)._init_grids()
        self.pressure_static = self.solver.create(RealGrid, name="Static Pressure")
        self.pressure_dynamic = self.solver.create(RealGrid, name="Dynamic Pressure")

    def _init_solver(self):
        super(StaticDynamicScene,self)._init_solver()
        self.max_iter_fac=10
        self.accuracy=5e-5

    def _solve_pressure(self):
        super(StaticDynamicScene,self)._solve_pressure()
        separatePressure(self.phi_fluid, self.pressure, self.flags, self.gravity, self.pressure_dynamic, self.pressure_static, 6.0 * self.resolution / 64.0)
