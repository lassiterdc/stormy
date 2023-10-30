#%% load libraries
from glob import glob

# define directories
dir_stormy = "D:/Dropbox/_GradSchool/_norfolk/stormy/"
dir_ssr = dir_stormy + "stochastic_storm_rescaling/"
dir_ssr_outputs = dir_ssr + "outputs/"
dir_swmm = dir_stormy + "swmm/hague/"
dir_swmm_sst_hrly = dir_swmm + "swmm_scenarios_sst_hourly/"
dir_outputprocessing = dir_swmm_sst_hrly + "_output_processing/"
dir_swmm_shps = dir_swmm + "swmm_model/exported_layers/"
dir_swmm_design_storms = dir_swmm_sst_hrly + "design_storm_scenarios/"

# other inputs
cubic_feet_per_cubic_meter = 35.3147
ks_alpha = 0.05 # for performing a two sided KS test for fitting a PDF to an annual maxima series of flood vols from SST
bootstrap_iterations = 1000
volume_units = 1e6 # million m^3
mm_per_inch = 25.4
m_per_feet = 1/3.28084
sst_conf_interval = 0.9 # 90% confidence interval reflects confidence interval of design storm rain depths in NOAA atlas 14
sst_recurrence_intervals = [0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100]
size_scaling_factor = 0.875 # the PowerPoint for the figure is this fraction the size of the final poster
target_text_size_in = 0.984252 # 2.5 cm
pts_per_inch = 72
# define filepaths
## 

## other shapefiles
fldr_repo_hres_rdar = "D:/Dropbox/_GradSchool/_norfolk/highres-radar-rainfall-processing/"
fldr_data_hres_rdar = fldr_repo_hres_rdar + "data/"
f_shp_coast = fldr_data_hres_rdar + "geospatial/composite_shoreline_clipped.shp"


## swmm shapefiles and other swmm
f_shp_subs = dir_swmm_shps + "subcatchments.shp"
f_shp_jxns = dir_swmm_shps + "junctions.shp"
f_shp_strg = dir_swmm_shps + "storages.shp"
f_shp_out = dir_swmm_shps + "outfalls.shp"
lst_outfall_ids = ["E143250", "E147007"] # to get max water levels in script b
sub_id_for_rain = "S181" # to get rainfall in script b

## hydrologic data
f_obs_water_lev = dir_ssr_outputs + "a_NOAA_water_levels/a_water-lev_tide_surge.csv"

## model results
f_sst_results_hrly = dir_swmm_sst_hrly + "model_outputs_consolidated.nc"
f_bootstrap_hrly = dir_swmm_sst_hrly + "bootstrapping_consolidated.nc"
f_bootstrap_raw_hrly = dir_swmm_sst_hrly + "bootstrapping_allsamples_consolidated.nc"
f_model_perf_summary = dir_swmm_sst_hrly + "model_performance_summary.csv"
f_sst_event_summaries = dir_swmm_sst_hrly + "event_summaries.csv"

## processed data
f_sst_annual_max_volumes = dir_outputprocessing + "a_sst_annual_max_volumes.csv"
f_design_strms = dir_outputprocessing + "b_design_storms.csv"
f_sst_recurrence_intervals = dir_outputprocessing + "c_sst_flood_recurrence_intervals.csv"
fldr_swmm_analysis = dir_stormy + "swmm/_python/analysis/"
fldr_swmm_analysis_plots = fldr_swmm_analysis + "plots/"
f_bootstrapping_analysis = fldr_swmm_analysis + "df_comparison_bootstrapped.csv"

## plots 
dir_plots = dir_outputprocessing + "_plots/"
# define functions for scripts
# def a_analyzing_sst_results():
#     return f_sst_results_hrly, f_model_perf_summary, f_sst_event_summaries, f_sst_annual_max_volumes, volume_units

# def b_processing_design_strms():
#     return dir_swmm_design_storms, f_design_strms, cubic_feet_per_cubic_meter, volume_units, mm_per_inch, lst_outfall_ids, sub_id_for_rain

# def c_ams_with_sst():
#     return f_sst_annual_max_volumes, ks_alpha, bootstrap_iterations, volume_units, f_sst_recurrence_intervals, sst_conf_interval, sst_recurrence_intervals

# def d_comparing_design_strms_and_sst():
#     return f_sst_annual_max_volumes, f_design_strms, f_sst_recurrence_intervals, dir_plots, m_per_feet, sst_recurrence_intervals, size_scaling_factor, target_text_size_in, pts_per_inch

# def e_isolating_surge_effects():
#     return f_sst_results_hrly, sst_conf_interval, sst_recurrence_intervals, f_bootstrap_hrly,f_bootstrap_raw_hrly, f_shp_jxns, f_shp_strg, f_shp_out, f_shp_coast, f_shp_subs
