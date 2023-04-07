#%% import libraries
from glob import glob
#%% filenames, paths, and directories
dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/"
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"
dir_sst_nrflk_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined.sst"
f_sst_nrflk_hrly_combined_catalog = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined.nc"
f_sst_nrflk_hrly_combined_catalog_reformatted = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined_reformatted_for_xarray.nc"
# swmm stuff
dir_swmm_model = dir_repo + "swmm/hague/"
f_shp_swmm_subs = dir_swmm_model + "swmm_model/exported_layers/subcatchments.shp"

# script c3
# dir_sst_mrms_hourly = 
dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined/Realizations/"
# c4
dir_swmm_sst_scenarios_hrly = dir_swmm_model + "swmm_scenarios_sst_hourly/"
dir_time_series_hrly = dir_swmm_sst_scenarios_hrly + "time_series/"
f_key_subnames_gridind = dir_time_series_hrly + "_key_subnames_and_grid-indices.csv"
seed_mrms_hourly = 22901


#%% hard coded variables
name_out_realizations = "_combined_realizations.nc"
mm_per_inch = 25.4
grid_spacing = 0.009999999776482582
start_date = "2020-09-01" # start date for each of the time series

# SST parameters
nyears = 10 # should be 1,000 for the final experiment
nperyear = 20
nrealizations = 5

#%% functions
def c4_creating_rainfall_tseries():
    freq = "H"
    f_out_realizations = dir_swmm_sst_scenarios_hrly + name_out_realizations
    return f_out_realizations, f_shp_swmm_subs, dir_time_series_hrly, mm_per_inch, grid_spacing, start_date, freq, f_key_subnames_gridind, dir_sst_realizations_hrly

def c3_reformat_hrly_cats():
    f_in = f_sst_nrflk_hrly_combined_catalog
    f_out = f_sst_nrflk_hrly_combined_catalog_reformatted
    f_out_realizations = dir_swmm_sst_scenarios_hrly + name_out_realizations
    return f_in, f_out, dir_swmm_sst_scenarios_hrly, f_out_realizations

def c1_combine_hrly_cats():
    parameterfile = f_sst_nrflk_hrly_parameterfile
    f_out = f_sst_nrflk_hrly_combined_catalog
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









