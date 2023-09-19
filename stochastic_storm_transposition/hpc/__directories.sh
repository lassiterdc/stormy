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
assar_dirs[hpc_scripts]=${assar_dirs[sst]}"hpc/"
assar_dirs[rainyday]=${assar_dirs[sst]}"RainyDay2/Source/"


# scripts
assar_dirs[hpc_rainyday_py]=${assar_dirs[sst]}"RainyDay2/Source/RainyDay_Py3.py"
# assar_dirs[hpc_c_py]=${assar_dirs[hpc_scripts]}"_c_running_rainyday_mrms_hourly.py"
# assar_dirs[hpc_c1_py]=${assar_dirs[hpc_scripts]}"_c1_combining_storm_catalogs.py"
# assar_dirs[hpc_c3a_py]=${assar_dirs[hpc_scripts]}"_c3a_creating_reformatted_catalog_for_xarray.py"
# assar_dirs[hpc_c3b_py]=${assar_dirs[hpc_scripts]}"_c3b_creating_reformatted_realizations_for_xarray.py"
assar_dirs[hpc_d4_py]=${assar_dirs[hpc_scripts]}"_d4_creating_rainfall_timeseries.py"
assar_dirs[hpc_d4b_py]=${assar_dirs[hpc_scripts]}"_d4b_creating_waterlevel_timeseries.py"
assar_dirs[hpc_d4c1a_py]=${assar_dirs[hpc_scripts]}"_d4c1a_analyzing_weather_consolidation.py"
assar_dirs[hpc_d4c1b_py]=${assar_dirs[hpc_scripts]}"_d4c1b_analyzing_weather_consolidation.py"
assar_dirs[hpc_d4c2_py]=${assar_dirs[hpc_scripts]}"_d4c2_analyzing_weather.py"
assar_dirs[hpc_d5_py]=${assar_dirs[hpc_scripts]}"_d5_creating_inps.py"
assar_dirs[hpc_d6_py]=${assar_dirs[hpc_scripts]}"_d6_running_swmm.py"
assar_dirs[hpc_d6b_py]=${assar_dirs[hpc_scripts]}"_d6b_analyzing_swmm_runs.py"
assar_dirs[hpc_d7_py]=${assar_dirs[hpc_scripts]}"_d7_consolidating_swmm_outputs.py"
assar_dirs[hpc_d7b_py]=${assar_dirs[hpc_scripts]}"_d7b_consolidating_swmm_outputs.py"
assar_dirs[hpc_8_py]=${assar_dirs[hpc_scripts]}"_d8_bootstrapping_results.py"
assar_dirs[hpc_8b_py]=${assar_dirs[hpc_scripts]}"_d8b_consolidating_bs_samples.py"
assar_dirs[hpc_d_py]=${assar_dirs[hpc_scripts]}"_d_running_rainyday_mrms.py"

# RainyDay Inputs
## for actual runs
assar_dirs[hpc_b_SST_stageVI]=${assar_dirs[sst]}"norfolk/norfolk_stageIV_rivanna.sst"
# assar_dirs[hpc_c_SST_mrms_hourly]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly.sst"

# c1
assar_dirs[hpc_mrms_hourly]=${assar_dirs[sst]}"norfolk/sst_mrms_hourly/"
assar_dirs[hpc_mrms]=${assar_dirs[sst]}"norfolk/sst_mrms/"



assar_dirs[hpc_c1_SST_mrms_hourly_inp]=${assar_dirs[hpc_mrms_hourly]}"_inputs/mrms_houlry_combined.sst"
assar_dirs[hpc_c1_cmbnd_cat]=${assar_dirs[hpc_mrms_hourly]}"strmcat_mrms_hourly_combined.nc"
assar_dirs[hpc_c1_sst]=${assar_dirs[hpc_mrms_hourly]}"mrms_hourly_combined.sst"

assar_dirs[hpc_d2_sst]=${assar_dirs[hpc_mrms]}"mrms_combined.sst"
assar_dirs[hpc_d1_sst]=${assar_dirs[hpc_mrms]}"mrms.sst"

## for testing
assar_dirs[hpc_b_SST_in_subset]=${assar_dirs[sst]}"norfolk/norfolk_mrms_subset_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_subset_short_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003_stageIV]=${assar_dirs[sst]}"norfolk/norfolk_stageIV_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_subset2003_stageIV_with_nas]=${assar_dirs[sst]}"norfolk/norfolk_stageIV_with_nas_hourly_subset_short_2003_rivanna.sst"
assar_dirs[hpc_b_SST_in_hrly_2001_to_2011]=${assar_dirs[sst]}"norfolk/norfolk_mrms_hourly_2001-2011_rivanna.sst"
assar_dirs[hpc_c1_sst_test]=${assar_dirs[hpc_mrms_hourly]}"mrms_hourly_combined_test.sst"

# reprex for Rivanna Support
assar_dirs[hpc_sup_c4_py]=${assar_dirs[hpc_scripts]}"_support__c4_creating_rainfall_timeseries.py"


