#!/bin/bash

#%% code examples
# include in a bash script 
    # source __directories.sh
# example to call array
    # echo ${assar_dirs[dir_repo]}
# concatenate
    #  echo "${assar_dirs[dir_repo]}${assar_dirs[subdir]}"

# declare associative array
declare -A assar_dirs

# populate associative array with folder and filepaths; this should be the only thing that has to be changed
assar_dirs[repo]="/project/quinnlab/dcl3nd/norfolk/stormy/"

