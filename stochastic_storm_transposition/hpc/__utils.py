#%% import libraries
from glob import glob
#%% user options
dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/"
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"
dir_sst_nrflk_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
#%% hard coding


#%% functions
def c1_combine_hrly_cats():
    parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined.sst"
    f_out = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined.nc"
    fs = glob(dir_sst_nrflk_hrly + "*_20*.nc")
    fs.sort() # sort alphabetically 
    return dir_sst_nrflk_hrly, parameterfile, f_out, fs

def c_rainyday_in():
    dir_mrms_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
    f_sst_mrms_hrly = dir_mrms_hrly + "mrms_hourly_template.sst"
    dir_for_sst_files = dir_mrms_hrly + "_inputs/"
    return dir_mrms_hrly, f_sst_mrms_hrly, dir_for_sst_files

def d_rainyday_in():
    dir_mrms = dir_sst_nrflk + "sst_mrms/"
    f_sst_mrms = dir_mrms + "mrms_template.sst"
    dir_for_sst_files = dir_mrms + "_inputs/"
    return dir_mrms, f_sst_mrms, dir_for_sst_files

def da_rainyday_in():
    dir_mrms = dir_sst_nrflk + "sst_mrms_subdivided/"
    f_sst_mrms = dir_mrms + "mrms_template_{}.sst"
    dir_for_sst_files = dir_mrms + "_inputs/"
    return dir_mrms, f_sst_mrms, dir_for_sst_files









