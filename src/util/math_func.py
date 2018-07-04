#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# required math functions
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

from math import log10, floor

# https://stackoverflow.com/a/3413529
def round_sig(x, sig=2):
    if x == 0.0:
        return round(x, sig)
    return round(x, sig-int(floor(log10(abs(x))))-1)

# https://stackoverflow.com/a/3928583
def round_tuple(tup, sig=2):
    if isinstance(tup, tuple):
        return tuple(map(lambda x: isinstance(x, float) and round_sig(x, sig) or x, tup))
    else:
        return tup