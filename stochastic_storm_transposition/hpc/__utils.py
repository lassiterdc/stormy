#%% import libraries
from glob import glob
#%% filenames, paths, and directories
dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/"
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"
dir_sst_nrflk_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
dir_home = "/home/dcl3nd/stormy/"
dir_home_sst = dir_home + "sst/"

# WORK
f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined.sst"
# f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined_test.sst"
# END WORK
f_sst_nrflk_hrly_combined_catalog = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined.nc"
f_sst_nrflk_hrly_combined_catalog_reformatted = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined_reformatted_for_xarray.nc"
# swmm stuff
dir_swmm_model = dir_repo + "swmm/hague/"
f_shp_swmm_subs = dir_swmm_model + "swmm_model/exported_layers/subcatchments.shp"
lst_template_keys = ["rainfall_1", "rainfall_2", "rainfall_4", "rainfall_5", "rainfall_7", "water_level"]
work_f_water_level_path = dir_swmm_model + "swmm_timeseries/a_water_levels_ft.dat"

# script c3
# WORK 
dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined/Realizations/"
# dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined_tst/Realizations/"
# END WORK
# c4
# BEGIN WORK
# dir_swmm_sst_scenarios_hrly = dir_swmm_model + "swmm_scenarios_sst_hourly/"
dir_swmm_sst_scenarios_hrly = dir_home_sst + "swmm_sst_hourly/"
dir_time_series_hrly = dir_swmm_sst_scenarios_hrly + "time_series/"
f_key_subnames_gridind = dir_time_series_hrly + "_key_subnames_and_grid-indices.csv"
seed_mrms_hourly = 22901
# END WORK
# c5
dir_swmm_sst_models_hrly = dir_swmm_sst_scenarios_hrly + "models/"
f_inp_base_hrly = dir_swmm_sst_scenarios_hrly + "hague_sst_model_template.inp"
# dir_time_series_hrly = dir_swmm_sst_scenarios_hrly + "time_series/"
f_swmm_scenarios_catalog = dir_swmm_sst_scenarios_hrly + "swmm_scenario_catalogs/" + "_swmm_scenarios_catalog_yr{}.csv"

# c6
max_runtime_min_hrly = 15 # maximum minutes of runtime allowable for each SWMM simulation
#%% hard coded variables
# name_out_realizations = "_combined_realizations.nc"
f_realizations_hourly = dir_swmm_model + "swmm_scenarios_sst_hourly/_combined_realizations.nc"
mm_per_inch = 25.4
grid_spacing = 0.009999999776482582
start_date = "2020-09-01" # start date for each of the time series

# SST parameters (should line up with the SST input file)
# WORK 
nyears = 1000 # should be 1,000 for the final experiment
nperyear = 20
nrealizations = 10
# nyears = 2
# nperyear = 20
# nrealizations = 2
# END WORK

#%% functions
def c6_running_swmm():
    return f_swmm_scenarios_catalog, dir_swmm_sst_models_hrly, max_runtime_min_hrly

def c5_creating_inps():
    f_out_realizations = f_realizations_hourly
    return nyears, nperyear, nrealizations, dir_swmm_sst_models_hrly, f_inp_base_hrly, f_out_realizations, seed_mrms_hourly, dir_time_series_hrly, f_key_subnames_gridind, lst_template_keys, work_f_water_level_path, f_swmm_scenarios_catalog

def c4_creating_rainfall_tseries():
    freq = "H"
    f_out_realizations = f_realizations_hourly
    return f_out_realizations, f_shp_swmm_subs, dir_time_series_hrly, mm_per_inch, grid_spacing, start_date, freq, f_key_subnames_gridind, dir_sst_realizations_hrly

def c3_reformat_hrly_cats():
    f_in = f_sst_nrflk_hrly_combined_catalog
    f_out = f_sst_nrflk_hrly_combined_catalog_reformatted
    f_out_realizations = f_realizations_hourly
    return f_in, f_out, dir_sst_realizations_hrly, f_out_realizations

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









