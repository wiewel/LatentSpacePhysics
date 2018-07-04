#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# array operations
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

import numpy as np

#=====================================================================================
# pass in multiple arrays with same length and shuffle all in unison
def shuffle_in_unison(*np_arrays):
    #print("Passed {} arguments".format(len(np_arrays)))
    rng = np.random.get_state()
    for array in np_arrays:
        np.random.set_state(rng)
        np.random.shuffle(array)

#=====================================================================================
def rotate_to_new(target, new_input, dimension):  #def step_forward(X, new_ts, time_steps):
    # skip the batch_size and simply use the first entry
    if new_input.ndim > 2:
        new_input = new_input[0]

    target[0] = np.insert(  np.delete(target[0], 0, axis=0),
                            dimension-1,
                            new_input[0] if new_input.ndim > 1 else new_input,
                            axis=0)

    return target

#=====================================================================================