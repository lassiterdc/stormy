#%% load libraries
from glob import glob
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

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


## attribution analysis parameters
quant_top_var = 0.8 # quantile of most variable nodes to use for deep dive into compound flood locations

## random forest model fitting
sse_quantile = 0.9 # used to subset random forest parameter sets that performed at or better than this quantile
    # then the parameter set with the lowset standard deviation will be chosen for that node
num_fits_for_estimating_sse = 10
min_rows_for_fitting = 20 # if there are fewer than this number of events with flooding, skip
ar_trees = [2,5,10,20,50,100]
ar_depths = [1,2,5,10,20]
response = ["flood_attribution"]
predictors = ["max_sim_wlevel", "rainfall_duration_hr", "depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "surge_peak_after_rain_peak_min"]

def return_period_to_quantile(ds, return_pds):
    total_years = len(ds_events.realization.values) * len(ds_events.year.values)
    total_events = len(ds_events.realization.values) * len(ds_events.year.values) * len(ds_events.storm_id.values)
    quants = []
    for return_pd in return_pds:
        expected_num_of_storms = total_years / return_pd
        quant = 1 - expected_num_of_storms / total_events
        quants.append(quant)
    return quants

def compute_return_periods(ds):
    ds_quants = ds.quantile(quants, dim = ["storm_id", "realization", "year"], method="closest_observation")
    ds_quants = ds_quants.assign_coords(dict(quantile = sst_recurrence_intervals))
    ds_quants = ds_quants.rename((dict(quantile="return_period_yrs")))
    df_quants = ds_quants.to_dataframe()
    return ds_quants, df_quants

def return_attribution_data():
    ds_sst = xr.open_dataset(f_sst_results_hrly)
    df_sst_events = pd.read_csv(f_sst_event_summaries)
    proj = ccrs.PlateCarree()
    gdf_jxns = gpd.read_file(f_shp_jxns)
    gdf_strg = gpd.read_file(f_shp_strg)
    gdf_out = gpd.read_file(f_shp_out)
    gdf_nodes = pd.concat([gdf_jxns, gdf_strg, gdf_out]).loc[:, ["NAME", "geometry"]]
    gdf_nodes = gdf_nodes.to_crs(proj)
    df_comparison = pd.read_csv(f_bootstrapping_analysis)

    gdf_node_attribution = gdf_nodes.merge(df_comparison, how = 'inner', left_on = "NAME", right_on = "node").drop("NAME", axis=1)

    df_node_attribution = pd.DataFrame(gdf_node_attribution)
    node_ids_sorted = df_node_attribution[df_node_attribution.flood_return_yrs == 100].sort_values("lower_CI", ascending=True)["node"].values

    df_node_attribution = df_node_attribution.set_index(["flood_return_yrs", "node"])
    df_node_attribution.frac_wlevel_mean[df_node_attribution.frac_wlevel_mean > 1] = 1
    df_node_attribution.frac_wlevel_mean[df_node_attribution.frac_wlevel_mean < 0] = 0
    df_ge1y = df_node_attribution[(df_node_attribution["frac_wlevel_mean"].reset_index().flood_return_yrs>=1).values]

    lst_ranges = []
    lst_nodes = []
    for node, group in df_ge1y.reset_index().groupby("node"):
        frac_range = group.frac_wlevel_mean.max() - group.frac_wlevel_mean.min()
        lst_ranges.append(frac_range)
        lst_nodes.append(node)

    df_node_ranges = pd.DataFrame(dict(node = lst_nodes, range = lst_ranges))
    df_variable_nodes = df_node_ranges[df_node_ranges.range >= df_node_ranges.range.quantile(quant_top_var)]

    ds_sst_compound = ds_sst.sel(freeboundary="False")
    ds_sst_freebndry = ds_sst.sel(freeboundary="True")

    ds_flood_attribution = 1 - (ds_sst_freebndry + .0000000000001) / (ds_sst_compound + .0000000000001)
    ds_flood_attribution = ds_flood_attribution.rename(dict(node_flooding_cubic_meters = "flood_attribution"))

    ds_flood_attribution["flood_attribution"]  =xr.where(ds_flood_attribution.flood_attribution < 0, 0, ds_flood_attribution.flood_attribution)

    ds_events =df_sst_events.set_index(["realization", "year", "storm_id"]).to_xarray()

    # computing quantiles and attribution by flood return period
    ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap_hrly)

    # load and transform shapefiles
    gdf_subs = gpd.read_file(f_shp_subs)
    gdf_coast = gpd.read_file(f_shp_coast)
    gdf_subs = gdf_subs.to_crs(proj)
    gdf_coast = gdf_coast.to_crs(proj)

    # ds_fld_dif = ds_sst_compound - ds_sst_freebndry


    quants = return_period_to_quantile(ds_sst_compound, sst_recurrence_intervals)
    # compute flooding quantiles
    ds_quants_fld, df_quants_fld = compute_return_periods(ds_sst_compound)

    ds_quants_fld["node_trns_flooding_cubic_meters"] = np.log10(ds_quants_fld["node_flooding_cubic_meters"]+.01)
    df_quants_fld = ds_quants_fld.to_dataframe()
    df_quants_fld = df_quants_fld.reset_index()
    gdf_node_flding = gdf_nodes.merge(df_quants_fld, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

    # # compute event statistic quantiles
    # ## rain depth
    # da_rain_depth = ds_events.depth_mm
    # ds_quants, df_quants = compute_return_periods(da_rain_depth)
    # ## rain intensity
    # da_rain_int = ds_events.max_mm_per_hour
    # ## peak storm surge
    # da_wlevel = ds_events.max_sim_wlevel

    return df_variable_nodes, ds_flood_attribution, ds_sst_compound, ds_sst_freebndry, ds_events, gdf_node_flding, gdf_subs, gdf_nodes, df_comparison