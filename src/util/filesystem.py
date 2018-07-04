#******************************************************************************
#
# Latent-space Physics: Towards Learning the Temporal Evolution of Fluid Flow
# Copyright 2018 Steffen Wiewel, Moritz Becher, Nils Thuerey
#
# filesystem class for folder creation and copy operations
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

import os
import datetime
import pathlib
import shutil
import stat

def find_directory(dirname):
    dirs_to_search = ["./", "../", "../../"]
    for start_dir in map(os.path.abspath, dirs_to_search):
        for dirpath, _, _ in os.walk(start_dir): 
            if dirpath.split(os.path.sep)[-1] == dirname:
                return dirpath
    raise RuntimeError("Could not find directory '{}'. Check if programm is executed in the right place.".format(dirname))

def find_dir(dirname, parent_levels=0):
    """ 
    find_dir searches for a directory in the proximity of the working directory with the specified name
    It will always search in the subdirectories of the working directory, but can be allowed to search in
    parent directories as well.
    * __dirname__: the name of the directory to search for
    * __parent_levels__: a value of 0 to disable search in parent directories, > 0 max levels of steps upwards in the directory tree
    """
    dirs_to_search = ["./"]
    for i in range(1, parent_levels + 1):
        dirs_to_search.append("../" * i)
    for start_dir in map(os.path.abspath, dirs_to_search):
        for dirpath, _, _ in os.walk(start_dir): 
            if dirpath.split(os.path.sep)[-1] == dirname:
                return dirpath
    raise RuntimeError("Could not find directory '{}'. Check if programm is executed in the right place.".format(dirname))

def make_dir(directory):
    """ 
    check if directory exists, otherwise makedir
    
    (workaround for python2 incompatibility)
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_uniqe_path(path):
    """
    create directory with unique name

    if directory already exists, count up until unique name
    """
    unique_path = path
    output_num = 1
    while os.path.exists(unique_path):
        unique_path = path + "_{}".format(output_num)
        output_num += 1
    return unique_path

#=========================================================================================
class Filesystem(object):
    #-------------------------------------------------------------------------------------
    def __init__(self, root):
        self.root = root
    
    #-------------------------------------------------------------------------------------
    def __getitem__(self, key):
        return self.get_path(key)

    #-------------------------------------------------------------------------------------
    def get_path(self, relative_path):
        path = self.root + relative_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    #-------------------------------------------------------------------------------------
    def copy(self, destination):
        # https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
        def copytree(src, dst, symlinks = False, ignore = None):
            if not os.path.exists(dst):
                os.makedirs(dst)
                shutil.copystat(src, dst)
            if not os.path.exists(src):
                return
            lst = os.listdir(src)
            if ignore:
                excl = ignore(src, lst)
                lst = [x for x in lst if x not in excl]
            for item in lst:
                s = os.path.join(src, item)
                d = os.path.join(dst, item)
                if symlinks and os.path.islink(s):
                    if os.path.lexists(d):
                        os.remove(d)
                    os.symlink(os.readlink(s), d)
                    try:
                        st = os.lstat(s)
                        mode = stat.S_IMODE(st.st_mode)
                        os.lchmod(d, mode)
                    except:
                        pass # lchmod not available
                elif os.path.isdir(s):
                    copytree(s, d, symlinks, ignore)
                else:
                    shutil.copy2(s, d)

        dst_path = destination
        #shutil.copytree(self.root, dst_path)
        copytree(self.root, dst_path)
        print("Filesystem: copy from {} to {} finished".format(self.root, dst_path))