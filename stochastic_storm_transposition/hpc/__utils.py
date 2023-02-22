#%% import libraries

#%% user options
dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/"
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"

# script specific files
## c
dir_mrms_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
f_sst_mrms_hrly = dir_mrms_hrly + "mrms_hourly_template.sst"
dir_for_sst_files = dir_mrms_hrly + "_inputs/"

#%% hard coding

#%% functions
def c_rainyday_in():
    return dir_mrms_hrly, f_sst_mrms_hrly, dir_for_sst_files









