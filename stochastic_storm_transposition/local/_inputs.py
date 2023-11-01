


#%% defining directories
fldr_main = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/"
fldr_rainyday_working_dir = fldr_main + "norfolk/"

#%% script specific directories
# non script specific filepaths and folders
f_shp_wshed = fldr_rainyday_working_dir + "watershed/norfolk_wshed_4326.shp"
f_shp_trans_dom = fldr_rainyday_working_dir + "transposition_domain/norfolk_trans_dom_4326.shp"

# work_a

scen_name = "mrms_combined"
fldr_rainyday_outputs = fldr_rainyday_working_dir + "sst_mrms/{}/".format(scen_name)
f_csv_freq_analysis = fldr_rainyday_outputs + "{}_FreqAnalysis.csv".format(scen_name)
fldr_realizations = fldr_rainyday_outputs + "Realizations/"
# f_nc_storm_cat = fldr_rainyday_working_dir + "norfolk_mrms_sst_subset_rivanna.nc"

# script d4b1
dir_repo = "D:/Dropbox/_GradSchool/_norfolk/stormy/"
dir_swmm_model = dir_repo + "swmm/hague/"
dir_swmm_sst_scenarios = dir_swmm_model + "swmm_scenarios_sst/"
dir_scenario_weather = dir_swmm_sst_scenarios + "weather/"
f_rain_realizations = dir_scenario_weather + "rainfall.nc"

f_key_subnames_gridind = dir_scenario_weather + "_key_subnames_and_grid-indices.csv"

dir_ssr = dir_repo + "stochastic_storm_rescaling/"
dir_ssr_outputs = dir_ssr + "outputs/"
dir_mrms_events = dir_ssr_outputs + "c_mrms_events/"
f_mrms_event_summaries = dir_mrms_events + "mrms_event_summaries.csv"
f_mrms_event_timeseries = dir_mrms_events + "mrms_event_timeseries.csv"
dir_noaa_water_levels = dir_ssr_outputs + "a_NOAA_water_levels/"
f_water_level_storm_surge = dir_noaa_water_levels + "a_water-lev_tide_surge.csv"
#%% defining functions for working scripts
# def work_a_inspctng_strm_cat():
#     return f_nc_storm_cat, f_csv_freq_analysis, fldr_realizations, f_shp_wshed, f_shp_trans_dom

