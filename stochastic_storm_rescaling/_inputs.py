#%% import libraries
from pathlib import Path
import os

#%% User inputs
# constants 
mm_per_inch = 25.4

# script a
in_a_begin_year = 2001

# minimum record length to use an NCEI dataset
min_record_length = 30
#%% Directories
fldr_nrflk = str(Path(os.getcwd()).parents[1]) + "/"
fldr_ssr = fldr_nrflk + "stormy/stochastic_storm_rescaling/"
fldr_mrms_processing = fldr_nrflk + "/highres-radar-rainfall-processing/data/"
dir_ssr_outputs = fldr_ssr + "outputs/"

# SWMM stuff
fldr_swmm = fldr_nrflk + "stormy/swmm/"
fldr_swmm_features = fldr_swmm + "hague/swmm_model/exported_layers/"
fldr_swmm_tseries = fldr_nrflk + "stormy/swmm/hague/swmm_timeseries/"
f_swmm_model = fldr_swmm + "hague_V1_using_norfolk_data.inp"
f_shp_swmm_subs = fldr_swmm_features + "subcatchments.shp"

# NCEI Data
fldr_NCEI = fldr_nrflk + "stormy/data/climate/NCEI/"
f_daily_summaries = fldr_NCEI + "2023-1-4_NCEI_daily summaries_download.csv"
f_hourlyglobal_precip_all_stations = fldr_NCEI + "2023-1-4_NCEI_Hourly Global_precip.csv"
f_hourlyglobal_precip_subset_stations = fldr_NCEI + "2023-1-4_NCEI_Hourly Global_station_subset_precip.csv"
f_hourlyprecip = fldr_NCEI + "2023-1-4_NCEI_Hourly Precip_precip.csv"


dir_noaa_water_levels = dir_ssr_outputs + "a_NOAA_water_levels/"
f_out_a_meta = dir_noaa_water_levels + 'sewells_pt_water_level_metadatap.json'
f_water_level_storm_surge = dir_noaa_water_levels + "a_water-lev_tide_surge.csv"
f_out_a_shp = dir_noaa_water_levels + "sewells_pt.shp"
f_out_swmm_waterlevel = fldr_swmm_tseries + "a_water_levels_ft.dat"

# event selection
min_interevent_time = 12 # hour
max_event_length = 72 # hours
min_event_threshold = 0.5 # inches of total rainfall
dir_mrms_events = dir_ssr_outputs + "c_mrms_events/"
f_mrms_event_summaries = dir_mrms_events + "mrms_event_summaries.csv"
f_mrms_event_timeseries = dir_mrms_events + "mrms_event_timeseries.csv"

fld_out_b = dir_ssr_outputs + "b_precip_time_series_at_gages/"
f_in_b_nc = fldr_mrms_processing+"mrms_nc_preciprate_fullres_atgages.nc"
f_mrms_rainfall = fld_out_b + "mrms_rainfall.csv"
f_out_b_csv_subs_w_mrms_grid = fld_out_b + "b_sub_ids_and_mrms_rain_col.csv"
f_out_swmm_rainfall = fldr_swmm_tseries + "b_mrms_rainfall_in_per_hr_{}.dat"

# unique to script c
fld_out_c_plts = fldr_NCEI + "qaqc_plots/"
fld_out_c_processed_data = fldr_NCEI + "processed_data/"

def def_work():
    return f_water_level_storm_surge

def def_work_b():
    return f_swmm_model, f_mrms_rainfall, f_out_b_csv_subs_w_mrms_grid

def def_work_c():
    return f_daily_summaries, f_hourlyglobal_precip_all_stations, f_hourlyglobal_precip_subset_stations, f_hourlyprecip

def def_inputs_for_a():
    return in_a_begin_year, f_out_a_meta, f_water_level_storm_surge, f_out_a_shp, f_out_swmm_waterlevel

def def_inputs_for_b():
    return f_in_b_nc, f_shp_swmm_subs, f_mrms_rainfall, f_out_b_csv_subs_w_mrms_grid, f_out_swmm_rainfall, mm_per_inch

def def_inputs_for_b2():
    return f_mrms_rainfall, f_water_level_storm_surge, min_interevent_time, max_event_length, min_event_threshold, mm_per_inch, f_mrms_event_summaries, f_mrms_event_timeseries

def def_inputs_for_c():
    return f_daily_summaries, f_hourlyprecip, fld_out_c_plts, fld_out_c_processed_data, min_record_length

def def_inputs_for_d():
    return f_mrms_event_summaries, f_mrms_event_timeseries, f_water_level_storm_surge