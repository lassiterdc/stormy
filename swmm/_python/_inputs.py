#%% load libraries
from glob import glob
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# define directories
dir_stormy = "D:/Dropbox/_GradSchool/_norfolk/stormy/"
dir_ssr = dir_stormy + "stochastic_storm_rescaling/"
dir_ssr_outputs = dir_ssr + "outputs/"
dir_swmm = dir_stormy + "swmm/hague/"
dir_swmm_sst = dir_swmm + "swmm_scenarios_sst/"
dir_outputprocessing = dir_swmm_sst + "_output_processing/"
dir_swmm_shps = dir_swmm + "swmm_model/exported_layers/"
dir_swmm_design_storms = dir_swmm_sst + "design_storm_scenarios/"

dir_data = dir_stormy + "data/"
dir_geospatial_data = dir_data + "geospatial/"

f_imagery_fullres = dir_geospatial_data + "imagery.tif"

# simulated event statistics
dir_sst = dir_stormy + "stochastic_storm_transposition/"
dir_local_outputs = dir_sst + "local/outputs/"
f_simulated_cmpnd_event_summaries = dir_local_outputs + "c_simulated_compound_event_summary.csv"
f_simulated_cmpnd_event_cdfs = dir_local_outputs + "r_a_sim_wlevel_cdf.csv"
f_wlevel_cdf_sims_from_copula_with_multvar_empcdf_subset = dir_local_outputs + "r_a_sim_wlevel_cdf_with_multivariate_empcdf_subset.csv"
dir_scenario_weather = dir_swmm_sst + "weather/"
f_sims_summary = dir_scenario_weather + "compound_event_summaries.csv"


# other inputs
storm_id_variables = ["realization", "year", "storm_id"] # these represent the coordinates to find each unique event
cubic_feet_per_cubic_meter = 35.3147
ks_alpha = 0.05 # for performing a two sided KS test for fitting a PDF to an annual maxima series of flood vols from SST
bootstrap_iterations = 1000
volume_units = 1e6 # million m^3
mm_per_inch = 25.4
m_per_feet = 1/3.28084
sst_conf_interval = 0.9 # 90% confidence interval reflects confidence interval of design storm rain depths in NOAA atlas 14
sst_recurrence_intervals = [0.5, 1, 2, 5, 10, 25, 50, 100]
size_scaling_factor = 0.875 # the PowerPoint for the figure is this fraction the size of the final poster
target_text_size_in = 0.984252 # 2.5 cm
pts_per_inch = 72
cubic_meters_per_cubic_foot = 0.0283168
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
f_sst_results = dir_swmm_sst + "model_outputs_consolidated.nc"
f_bootstrap = dir_swmm_sst + "bootstrapping_consolidated.nc"
f_bootstrap_raw = dir_swmm_sst + "bootstrapping_allsamples_consolidated.nc"
f_model_perf_summary = dir_swmm_sst + "model_performance_summary.csv"
f_sst_event_summaries = dir_swmm_sst + "event_summaries.csv"

## processed data
f_design_strms = dir_outputprocessing + "b_design_storms.csv"
f_sst_recurrence_intervals = dir_outputprocessing + "c_sst_flood_recurrence_intervals.csv"
fldr_python = dir_stormy + "swmm/_python/"
fldr_scratch = fldr_python + "_scratch/"
fldr_scratch_plots = fldr_scratch + "plots/"
fldr_swmm_analysis = fldr_python + "analysis/"
fldr_swmm_analysis_plots = fldr_swmm_analysis + "plots/"
f_bootstrapping_analysis = fldr_swmm_analysis + "df_comparison_bootstrapped.csv"
f_return_pd_analysis = fldr_swmm_analysis + "df_bootstrapped_eventids_for_each_node_and_return_period.csv"

## plots 
dir_plots = dir_outputprocessing + "_plots/"

# watershed scale attribution
min_vol_cutoff_watershed_scale_attribution = 1 # cubic meters of watershed-wide flooding threshold (basically events with less than this are dropped)

## attribution analysis parameters
quant_top_var = 0.7 # quantile of most variable nodes to use for deep dive into compound flood locations

## random forest model fitting
sse_quantile = 0.9 # used to subset random forest parameter sets that performed at or better than this quantile
    # then the parameter set with the lowset standard deviation will be chosen for that node
num_fits_for_estimating_sse = 10
min_rows_for_fitting = 20 # if there are fewer than this number of events with flooding, skip
ar_trees = [2,5,10,20,50,100]
ar_depths = [1,2,5,10,20]
response = ["flood_attribution"]
predictors = ["max_sim_wlevel", "event_duration_hr", "depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "surge_peak_after_rain_peak_min"]