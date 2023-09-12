#%% import libraries
from glob import glob
# filenames, paths, and directories
dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/"
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"
dir_sst_nrflk_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
dir_home = "/home/dcl3nd/stormy/"
dir_home_sst = dir_home + "sst/"
dir_scratch_sst = "/scratch/dcl3nd/stormy/"

# WORK
f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined.sst"
# f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined_test.sst"
# END WORK
f_sst_nrflk_hrly_combined_catalog = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined.nc"
f_sst_nrflk_hrly_combined_catalog_reformatted = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined_reformatted_for_xarray.nc"
# swmm stuff
dir_swmm_model = dir_repo + "swmm/hague/"
f_shp_swmm_subs = dir_swmm_model + "swmm_model/exported_layers/subcatchments.shp"
lst_template_keys = ["START_DATE", "START_TIME", "REPORT_START_DATE", "REPORT_START_TIME", "END_DATE", "END_TIME", "rainfall_0", "rainfall_1", "rainfall_2", "rainfall_4", "rainfall_5", "water_level", "OF_TYPE", "STAGE_DATA"]
# work_f_water_level_path = dir_swmm_model + "swmm_timeseries/a_water_levels_ft.dat"

# script c3
# WORK 
dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined/Realizations/"
# dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined_tst/Realizations/"
# END WORK
# c4
# BEGIN WORK
dir_swmm_sst_scenarios_hrly_proj = dir_swmm_model + "swmm_scenarios_sst_hourly/"
dir_swmm_sst_scenarios_hrly_scratch = dir_scratch_sst + "swmm_sst_hourly/"
dir_swmm_sst_scenarios_hrly_home = dir_home_sst + "swmm_sst_hourly/"
dir_time_series_hrly = dir_swmm_sst_scenarios_hrly_home + "time_series/"
f_key_subnames_gridind = dir_time_series_hrly + "_key_subnames_and_grid-indices.csv"
seed_mrms_hourly = 22901
# END WORK

# c4b 
dir_ssr = dir_repo + "stochastic_storm_rescaling/"
dir_ssr_outputs = dir_ssr + "outputs/"
dir_mrms_events = dir_ssr_outputs + "c_mrms_events/"
f_mrms_event_summaries = dir_mrms_events + "mrms_event_summaries.csv"
f_mrms_event_timeseries = dir_mrms_events + "mrms_event_timeseries.csv"
dir_noaa_water_levels = dir_ssr_outputs + "a_NOAA_water_levels/"
f_water_level_storm_surge = dir_noaa_water_levels + "a_water-lev_tide_surge.csv"
sst_hrly_tstep_min = 60
time_buffer = 6 # hours; this is the amount of time before either the start of rain or the peak storm surge and AFTER the end of rain or peak storm surge; this will also determine simulation start and end dates

c4b_gen_plots = False
wlevel_threshold = 0.5 # i don't want simulated time series that are 50% above or below the min and max observed waterlevel since 2000

n_attempts = 200
n_clusters = 5
resampling_inteval = 10 # after a number of attempts equal to a multiple of this value, the copula will be used to resample storm surge and peak lag  

# c5
# dir_swmm_sst_models_hrly = dir_swmm_sst_scenarios_hrly_proj + "models/"
dir_swmm_sst_models_hrly = dir_swmm_sst_scenarios_hrly_scratch + "models/"

f_inp_base_hrly = dir_swmm_sst_scenarios_hrly_proj + "hague_sst_model_template.inp"

# dir_time_series_hrly = dir_swmm_sst_scenarios_hrly + "time_series/"
f_swmm_scenarios_catalog = dir_swmm_sst_scenarios_hrly_proj + "swmm_scenario_catalogs/" + "_swmm_scenarios_catalog_yr{}.csv"

norain_gage_name = "no_rain"

# c6
max_runtime_min_hrly = 60 # maximum minutes of runtime allowable for each SWMM simulation

# c6b
f_model_perf_summary_hrly = dir_swmm_sst_scenarios_hrly_proj + "model_performance_summary.csv"
f_events_summary = dir_swmm_sst_scenarios_hrly_proj + "event_summaries.csv"

# c7
f_model_outputs_consolidated_hrly = dir_swmm_sst_scenarios_hrly_proj + "model_outputs_consolidated.nc"
# dir_swmm_sst_models_hrly_home = dir_swmm_sst_scenarios_hrly_home + "models/"

# c8
f_bootstrapped_quant_estimates = dir_swmm_sst_scenarios_hrly_scratch + "models/boostrapping/"
sst_recurrence_intervals = [0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]
f_bootstrapped_consolidated_hrly = dir_swmm_sst_scenarios_hrly_proj + "bootstrapping_consolidated.nc"
f_bootstrapped_consolidated_hrly_raw = dir_swmm_sst_scenarios_hrly_proj + "bootstrapping_allsamples_consolidated.nc"
export_raw_bs_samps = False

#d4
fldr_realizations = "/project/quinnlab/dcl3nd/norfolk/stormy/stochastic_storm_transposition/norfolk/sst_mrms/mrms_combined/Realizations/"
dir_swmm_sst_scenarios = dir_swmm_model + "swmm_scenarios_sst/"
# dir_swmm_sst_scenarios_scratch = dir_scratch_sst + "swmm_sst_hourly/"
# dir_swmm_sst_scenarios_hrly_home = dir_home_sst + "swmm_sst_hourly/"
dir_time_series = dir_swmm_sst_scenarios + "time_series/"
f_key_subnames_gridind = dir_time_series + "_key_subnames_and_grid-indices.csv"
seed_mrms_hourly = 22901

# d4b
plot_weather_gen_stuff = False
plt_fldr_weather_gen = dir_time_series + "_plots/"

# d5
f_inp_base = dir_swmm_sst_scenarios + "hague_sst_model_template.inp"
#%% hard coded variables
# name_out_realizations = "_combined_realizations.nc"
f_realizations_hourly = dir_swmm_model + "swmm_scenarios_sst_hourly/_combined_realizations.nc"
mm_per_inch = 25.4
grid_spacing = 0.009999999776482582
start_date = "2020-09-01" # start date for each of the time series
meters_per_foot = 0.3048
square_meters_per_square_foot = meters_per_foot * meters_per_foot
cubic_meters_per_cubic_foot = meters_per_foot*meters_per_foot*meters_per_foot

# SST parameters (should line up with the SST input file)
# WORK 
nyears = 1000 # should be 1,000 for the final experiment
nperyear = 20
nrealizations = 1
# nyears = 2
# nperyear = 20
# nrealizations = 2
# END WORK

#%% script specific functions
# def c8b_bootstrapping():
#     return f_bootstrapped_consolidated_hrly, f_bootstrapped_consolidated_hrly_raw, dir_swmm_sst_models_hrly, f_bootstrapped_quant_estimates, sst_recurrence_intervals, export_raw_bs_samps

# def c8_bootstrapping():
#     return f_model_outputs_consolidated_hrly, dir_swmm_sst_models_hrly, f_bootstrapped_quant_estimates, sst_recurrence_intervals, export_raw_bs_samps

# def c7b_consolidating_outputs():
#     return f_model_outputs_consolidated_hrly, dir_swmm_sst_models_hrly

# def c7_consolidating_outputs():
#     return f_model_perf_summary_hrly, dir_swmm_sst_models_hrly, cubic_meters_per_cubic_foot, nperyear

# def c6b_analyzing_swmm_runs():
#     return dir_swmm_sst_models_hrly, f_model_perf_summary_hrly, dir_time_series_hrly, f_events_summary


# def c6_running_swmm():
#     return f_swmm_scenarios_catalog, dir_swmm_sst_models_hrly, max_runtime_min_hrly

# def c5_creating_inps():
#     f_out_realizations = f_realizations_hourly
#     return nyears, nperyear, nrealizations, dir_swmm_sst_models_hrly, f_inp_base_hrly, f_out_realizations, seed_mrms_hourly, dir_time_series_hrly, f_key_subnames_gridind, lst_template_keys, f_swmm_scenarios_catalog, norain_gage_name

# # def c4b_creating_wlevel_tseries():
# #     return f_mrms_event_summaries, f_mrms_event_timeseries, f_water_level_storm_surge, f_realizations_hourly, f_key_subnames_gridind, nrealizations, sst_hrly_tstep_min, start_date, time_buffer, dir_time_series_hrly, c4b_gen_plots, wlevel_threshold, n_attempts, n_clusters, resampling_inteval

# def c4_creating_rainfall_tseries():
#     freq = "H"
#     f_out_realizations = f_realizations_hourly
#     return nrealizations, f_out_realizations, f_shp_swmm_subs, dir_time_series_hrly, mm_per_inch, grid_spacing, start_date, freq, f_key_subnames_gridind, dir_sst_realizations_hrly

# def c3_reformat_hrly_cats():
#     f_in = f_sst_nrflk_hrly_combined_catalog
#     f_out = f_sst_nrflk_hrly_combined_catalog_reformatted
#     f_out_realizations = f_realizations_hourly
#     return f_in, f_out, dir_sst_realizations_hrly, f_out_realizations

# def c1_combine_hrly_cats():
#     parameterfile = f_sst_nrflk_hrly_parameterfile
#     f_out = f_sst_nrflk_hrly_combined_catalog
#     fs = glob(dir_sst_nrflk_hrly + "*_20*.nc")
#     fs.sort() # sort alphabetically 
#     return dir_sst_nrflk_hrly, parameterfile, f_out, fs

# def c_rainyday_in():
#     dir_mrms_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
#     f_sst_mrms_hrly = dir_mrms_hrly + "mrms_hourly_template.sst"
#     dir_for_sst_files = dir_mrms_hrly + "_inputs/"
#     return dir_mrms_hrly, f_sst_mrms_hrly, dir_for_sst_files

# def d_rainyday_in():
#     dir_mrms = dir_sst_nrflk + "sst_mrms/"
#     f_sst_mrms = dir_mrms + "mrms_template.sst"
#     dir_for_sst_files = dir_mrms + "_inputs/"
#     return dir_mrms, f_sst_mrms, dir_for_sst_files

# def da_rainyday_in():
#     dir_mrms = dir_sst_nrflk + "sst_mrms_subdivided/"
#     f_sst_mrms = dir_mrms + "mrms_template_{}.sst"
#     dir_for_sst_files = dir_mrms + "_inputs/"
#     return dir_mrms, f_sst_mrms, dir_for_sst_files

#%% functions
import pandas as pd
import sys
import numpy as np
import xarray as xr

def parse_inp(f_inp):
    # format "rz{}_yr{}_strm{}.inp"
    lst_name_comp = f_inp.split("/")[-1].split("_")
    rz = lst_name_comp[0].split("rz")[-1]
    yr = lst_name_comp[1].split("yr")[-1]
    storm_id = lst_name_comp[2].split(".")[0].split('strm')[-1]
    freebndry = False
    if len(lst_name_comp) == 4:
        freebndry = True
    return int(rz), int(yr), int(storm_id), str(freebndry)

# for loading RainyDay realizations
def define_dims(ds):
    fpath = ds.encoding["source"]
    lst_f = fpath.split("/")[-1].split("_")
    rz = int(lst_f[0].split("rz")[-1])
    year = int(lst_f[1].split("y")[-1])
    strm = int(lst_f[2].split("stm")[-1].split(".")[0])
    first_tstep = ds.time.values[0]
    tseries = pd.Series(ds.time.values)
    tsteps_unique = tseries.diff().dropna().unique()
    if len(tsteps_unique) > 1:
        sys.exit("variable time step encountered in file {}".format(fpath))
    tstep_min = tsteps_unique[0] / np.timedelta64(1, "m")
    tstep_ind = np.arange(len(tseries))
    ds["time"] = tstep_ind
    ds = ds.assign_attrs(timestep_min = tstep_min)
    ds = ds.assign_coords(dict(realization=rz, year = year, storm_id = strm, first_tstep = first_tstep))
    ds = ds.expand_dims(dim=dict(realization=1, year = 1, storm_id = 1))
    return ds

# for loading RainyDay realizations for a specific year only
def return_rzs_for_yr(fldr_realizations, yr):
    lst_f_all_ncs = glob(fldr_realizations+"*.nc")
    lst_f_ncs = []
    for f in lst_f_all_ncs:
        lst_f = f.split("/")[-1].split("_")
        # rz = int(lst_f[0].split("rz")[-1])
        year = int(lst_f[1].split("y")[-1])
        # strm = int(lst_f[2].split("stm")[-1].split(".")[0])
        if year == yr:
            lst_f_ncs.append(f)
    lst_f_ncs.sort()
    return lst_f_ncs







