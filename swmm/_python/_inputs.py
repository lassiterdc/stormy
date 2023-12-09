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
dir_scenario_weather = dir_swmm_sst + "weather/"
f_sims_summary = dir_scenario_weather + "compound_event_summaries.csv"


# other inputs
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
f_sst_annual_max_volumes = dir_outputprocessing + "a_sst_annual_max_volumes.csv"
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

def return_period_to_quantile(ds, return_pds):
    total_years = len(ds.realization.values) * len(ds.year.values)
    total_events = len(ds.realization.values) * len(ds.year.values) * len(ds.storm_id.values)
    quants = []
    for return_pd in return_pds:
        expected_num_of_storms = total_years / return_pd
        quant = 1 - expected_num_of_storms / total_events
        quants.append(quant)
    return quants

def compute_return_periods(ds, quants, recurrence_interval_yrs = sst_recurrence_intervals, method = "closest_observation"):
    ds_quants = ds.quantile(quants, dim = ["storm_id", "realization", "year"], method=method)
    ds_quants = ds_quants.assign_coords(dict(quantile = recurrence_interval_yrs))
    ds_quants = ds_quants.rename((dict(quantile="flood_return_yrs")))
    df_quants = ds_quants.to_dataframe()
    return ds_quants, df_quants

#%% data processing
ds_sst = xr.open_dataset(f_sst_results)
# df_sst_events = pd.read_csv(f_sst_event_summaries)
storm_id_variables = ["realization", "year", "storm_id"]
proj = ccrs.PlateCarree()
gdf_jxns = gpd.read_file(f_shp_jxns)
gdf_strg = gpd.read_file(f_shp_strg)
gdf_out = gpd.read_file(f_shp_out)
gdf_nodes = pd.concat([gdf_jxns, gdf_strg, gdf_out]).loc[:, ["NAME", "geometry"]]
gdf_nodes = gdf_nodes.to_crs(proj)
df_comparison = pd.read_csv(f_bootstrapping_analysis)
df_events = pd.read_csv(f_sims_summary).set_index(storm_id_variables)

# compute number of years
n_years_generated = len(df_events.reset_index().realization.unique()) * len(df_events.reset_index().year.unique())
n_storms_generated = n_years_generated * len(df_events.reset_index().storm_id.unique()) 

# add multivariate empirical CDF columns to df_events
df_events_with_emp_cdf = pd.read_csv(f_simulated_cmpnd_event_summaries).set_index(storm_id_variables)
# load cdf values
df_events_cdf_simulation = pd.read_csv(f_simulated_cmpnd_event_cdfs).join(df_events_with_emp_cdf.reset_index().loc[:, storm_id_variables]).set_index(storm_id_variables)
# add everything to the df_events dataframe
df_events = df_events.join(df_events_with_emp_cdf.loc[:, ["n_emp_multivar_cdf", "emp_multivar_cdf"]])
df_events = df_events.join(df_events_cdf_simulation, rsuffix="_cdf")
df_events = df_events.join(df_events_cdf_simulation*n_years_generated, rsuffix="_emp_return_pd_yr")
n_years_simulated = df_events.reset_index().realization.max() * df_events.reset_index().year.max()
df_events["empirical_event_return_pd_yr"] = df_events["emp_multivar_cdf"] * n_years_simulated
# this is a workaround because I left the format as a string
try:
    event_duration_hr = pd.to_timedelta(df_events["event_duration_hr"]) / np.timedelta64(1, 'h')
    df_events.loc[:, "event_duration_hr"] = event_duration_hr.values
except:
    pass
ds_events = df_events.to_xarray()

# node_ids_sorted = df_node_attribution[df_node_attribution.flood_return_yrs == 100].sort_values("lower_CI", ascending=True)["node_id"].values

df_comparison = df_comparison.set_index(["flood_return_yrs", "node_id"])
# workaround - replacing any attribution values greater than 1 with 1 and less than 1 with 0
def workaround_replace_vals(var_to_replace, df_comparison=df_comparison):
    idx_mean_greater_than_1 = df_comparison.index[df_comparison[var_to_replace] > 1].tolist()
    idx_mean_less_than_1 = df_comparison.index[df_comparison[var_to_replace] < 0].tolist()
    df_comparison.loc[idx_mean_greater_than_1, var_to_replace] = 1
    df_comparison.loc[idx_mean_less_than_1, var_to_replace] = 0
    return df_comparison

df_comparison = workaround_replace_vals("frac_wlevel_mean", df_comparison)
df_comparison = workaround_replace_vals("frac_wlevel_median", df_comparison)

#%% dcl work - figuring out better way of isolating the nodes that have variable attribution
# identifying variable nodes
vrblity_anly_var = 'frac_wlevel_mean'
# return the standard deviation of the target variability analysis variable; replace NA values with  0 (this means there was just 1 observation) and sort values
df_node_variability = df_comparison.reset_index().loc[:, ["node_id", vrblity_anly_var]].groupby("node_id").std().fillna(0).sort_values(vrblity_anly_var)
std_varname = "std_of_{}".format(vrblity_anly_var)
df_node_variability = df_node_variability.rename(columns = {vrblity_anly_var:std_varname})

# df_variable_nodes = 
df_variable_nodes = df_node_variability[df_node_variability[std_varname] >= df_node_variability[std_varname].quantile(quant_top_var)]
# df_node_attribution_subset = df_node_attribution.join(df_node_variability_subset, how = "right")

# df_comparison.loc[(slice(None), "UN69"),]

# old way - start
# df_ge1y = df_comparison[(df_comparison["frac_wlevel_mean"].reset_index().flood_return_yrs>=1).values]
# lst_ranges = []
# lst_nodes = []
# for node, group in df_ge1y.reset_index().groupby("node_id"):
#     frac_range = group.frac_wlevel_mean.max() - group.frac_wlevel_mean.min()
#     lst_ranges.append(frac_range)
#     lst_nodes.append(node)

# df_node_ranges = pd.DataFrame(dict(node = lst_nodes, range = lst_ranges))

# df_node_variability = df_node_ranges[df_node_ranges.range >= df_node_ranges.range.quantile(quant_top_var)]
# end old way
#%% end dcl work
gdf_node_attribution = gdf_nodes.merge(df_comparison.reset_index(), how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

df_node_attribution = pd.DataFrame(gdf_node_attribution)
df_node_attribution = df_node_attribution.set_index(["node_id", "flood_return_yrs"])

ds_sst_compound = ds_sst.sel(freeboundary="False", norain = "False")
ds_sst_nosurge = ds_sst.sel(freeboundary="True", norain = "False")
ds_sst_norain = ds_sst.sel(freeboundary="False", norain = "True")

# for now, using the no surge time series to compute flood attribution
ds_sst_freebndry = ds_sst_nosurge

ds_flood_attribution = 1 - (ds_sst_freebndry + .0000000000001) / (ds_sst_compound + .0000000000001)
ds_flood_attribution = ds_flood_attribution.rename(dict(node_flooding_cubic_meters = "flood_attribution"))

ds_flood_attribution["flood_attribution"]  =xr.where(ds_flood_attribution.flood_attribution < 0, 0, ds_flood_attribution.flood_attribution)

# computing quantiles and attribution by flood return period
ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap)

# load and transform shapefiles
gdf_subs = gpd.read_file(f_shp_subs)
gdf_coast = gpd.read_file(f_shp_coast)
gdf_subs = gdf_subs.to_crs(proj)
gdf_coast = gdf_coast.to_crs(proj)

# ds_fld_dif = ds_sst_compound - ds_sst_freebndry


quants = return_period_to_quantile(ds_sst_compound, sst_recurrence_intervals)
# compute flooding quantiles
ds_quants_fld, df_quants_fld = compute_return_periods(ds_sst_compound, quants)

ds_quants_fld["node_trns_flooding_cubic_meters"] = np.log10(ds_quants_fld["node_flooding_cubic_meters"]+.01)
df_quants_fld = ds_quants_fld.to_dataframe()
df_quants_fld = df_quants_fld.reset_index()
gdf_node_flding = gdf_nodes.merge(df_quants_fld, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

# joining flood attribution by quantile to weather quantiles
