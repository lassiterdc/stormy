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
assar_dirs[sst]=${assar_dirs[repo]}"stochastic_storm_transposition/"

# scripts
assar_dirs[hpc_b_py]=${assar_dirs[sst]}"RainyDay2/Source/RainyDay_Py3.py"

# RainyDay Inputs
## for testing
assar_dirs[hpc_b_SST_in_subset]=${assar_dirs[sst]}"norfolk/norfolk_mrms_subset_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_subset_short_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003_stageIV]=${assar_dirs[sst]}"norfolk/norfolk_stageIV_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003_stageIV_with_nas]=${assar_dirs[sst]}"norfolk/norfolk_stageIV_with_nas_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_2001_to_2011]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_2001-2011_rivanna.sst"



