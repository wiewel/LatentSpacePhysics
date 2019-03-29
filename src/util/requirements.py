#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# requirements check, installs required packages via pip if possible
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

"""
### Check and install requirements 
Add as first import:
```python
# check and install requirements
import util.requirements
util.requirements.fulfill()
```
"""
import pip

# Add requirements to this set
_REQUIREMENTS = set([
    "numpy",
    "tensorflow-gpu",
    "keras",
    "sklearn",
    "h5py",
    "matplotlib",
])

def fulfill():
    # pip10.0 does not support internal calls anymore
    try:
        installed_packages = set([package.project_name.lower() for package in  pip.get_installed_distributions()])
        missing_packages = [package for package in _REQUIREMENTS if package not in installed_packages]
        for package in missing_packages:
            pip.main(["install", package])
    except:
        pass
    import keras
    assert keras.__version__ == "2.1.6", ("Only Keras 2.1.6 is supported. Currently installed Keras version is {}.".format(keras.__version__))

def reset_rng():
    '''Setup seeds to be as deterministic as possible. cudNN is still not deterministic'''
    import numpy as np
    import tensorflow as tf
    np.random.seed(4213)
    tf.set_random_seed(3742)

def init_packages():
    import tensorflow as tf
    import os
    import keras.backend.tensorflow_backend as KTF

    gpu_fraction=0.8
    '''Setup seeds to be as deterministic as possible. cudNN is still not deterministic'''
    reset_rng()
    '''You want to allocate gpu_fraction percent of it'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    if num_threads:
        num_threads = int(num_threads)
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=gpu_fraction)
    print("Found {} threads. Using {} of GPU memory per process.".format(num_threads, gpu_fraction))
    if num_threads:
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    KTF.set_session(session)
