#%% load libraries and directories
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from tqdm import tqdm

# plotting parameters
# width_to_height = 7.5 / 3
# height = 5
# width = height * width_to_height 

# outlier cutoff:
outlier_cutoff = 1e10 # cubic meters

from _inputs import *

scratch_folder = "_scratch/"
scratch_file = "_scratch/{}"

cmap = "gist_rainbow"

#%%
ds_sst = xr.open_dataset(f_sst_results_hrly)
ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap_hrly)
# ds_bootstrap_raw = xr.open_dataset(f_bootstrap_raw_hrly, chunks=dict(node_id=1))
# df_model_perf = pd.read_csv(f_model_perf_summary)
df_sst_events = pd.read_csv(f_sst_event_summaries)

# load and transform shapefile
gdf_jxns = gpd.read_file(f_shp_jxns)
gdf_strg = gpd.read_file(f_shp_strg)
gdf_out = gpd.read_file(f_shp_out)
gdf_nodes = pd.concat([gdf_jxns, gdf_strg, gdf_out]).loc[:, ["NAME", "geometry"]]
gdf_subs = gpd.read_file(f_shp_subs)
gdf_coast = gpd.read_file(f_shp_coast)

proj = ccrs.PlateCarree()
gdf_nodes = gdf_nodes.to_crs(proj)
gdf_subs = gdf_subs.to_crs(proj)
gdf_coast = gdf_coast.to_crs(proj)
 
# also load weather statistics


#%% preprocessing
ds_sst_compound = ds_sst.sel(freeboundary="False")
ds_sst_freebndry = ds_sst.sel(freeboundary="True")
ds_fld_dif = ds_sst_compound - ds_sst_freebndry

ds_flood_attribution = 1 - (ds_sst_freebndry + .0000000000001) / (ds_sst_compound + .0000000000001)
ds_flood_attribution = ds_flood_attribution.rename(dict(node_flooding_cubic_meters = "flood_attribution"))

ds_flood_attribution["flood_attribution"]  =xr.where(ds_flood_attribution.flood_attribution < 0, 0, ds_flood_attribution.flood_attribution)

ds_events =df_sst_events.set_index(["realization", "year", "storm_id"]).to_xarray()



#%% exploring questions
# what is the 1, 2, 5, 10, 25, 50, and 100 year flood at each node?
def return_period_to_quantile(ds, return_pds):
    storms_per_year = ds.storm_id.shape[0]
    total_years = ds.year.shape[0]
    total_events = storms_per_year * total_years
    quants = []
    for return_pd in return_pds:
        expected_num_of_storms = total_years / return_pd
        quant = 1 - expected_num_of_storms / total_events
        quants.append(quant)
    return quants

quants = return_period_to_quantile(ds_sst_compound, sst_recurrence_intervals)


#%% compute quantiles
ds_quants = ds_sst_compound.quantile(quants, dim = ["storm_id", "realization", "year"], method="closest_observation")
ds_quants = ds_quants.assign_coords(dict(quantile = sst_recurrence_intervals))
ds_quants = ds_quants.rename((dict(quantile="flood_return_yrs")))

ds_quants["node_trns_flooding_cubic_meters"] = np.log10(ds_quants["node_flooding_cubic_meters"]+.01)

#%% plotting return periods for each flood volume
df_quants = ds_quants.to_dataframe()
df_quants = df_quants.reset_index()
#%%
gdf_node_flding = gdf_nodes.merge(df_quants, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)
#%% computing bootstrap confidence intervals around flood volumes
lst_ds_bs_CIs = []
for node in tqdm(np.unique(ds_sst_compound.node_id.values)):
    count = -1
    ds_bs_node = ds_bootstrap_rtrn.sel(dict(node_id = node))
    if ds_bs_node.node_flooding_cubic_meters.sum().values > 0:
        for flood_return_yrs in ds_bootstrap_rtrn.flood_return_yrs.values:
            count += 1
            quant = quants[count]
            node_flooding = df_quants[(df_quants.node_id==node) & (df_quants.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values

            lower_quant =  (1-sst_conf_interval) / 2
            upper_quant = 1 - lower_quant
            ds_bs_subset = ds_bs_node.sel(dict(flood_return_yrs = flood_return_yrs))
            upper_bound = float(ds_bs_subset.quantile(q = upper_quant, dim = "bootstrap_sample").node_flooding_cubic_meters.values)
            lower_bound = float(ds_bs_subset.quantile(q = lower_quant, dim = "bootstrap_sample").node_flooding_cubic_meters.values)

            ds_bs_CIs = pd.DataFrame({"node_id":[node], "flood_return_yrs":[flood_return_yrs], "upper_CI":[upper_bound],
                            "lower_CI":[lower_bound]})

            ds_bs_CIs = ds_bs_CIs.replace(to_replace=-1*np.inf, value =np.nan)

            lst_ds_bs_CIs.append(ds_bs_CIs)
        else:
            continue

df_bs_CIs = pd.concat(lst_ds_bs_CIs)
df_bs_CIs.to_csv(scratch_file.format("df_bootstrapped_CIs.csv"), index=False)
#%% create dataframe of all events within bootstrap confidence interval
df_bs_CIs = pd.read_csv(scratch_file.format("df_bootstrapped_CIs.csv"))

df_bs_CIs.groupby("node")

lst_ds_comp = []
for node in tqdm(df_bs_CIs.node.unique()):
    count = -1
    for flood_return_yrs in ds_bootstrap_rtrn.flood_return_yrs.values:
        count += 1
        quant = quants[count]
        # node_flooding = df_quants[(df_quants.node_id==node) & (df_quants.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values
        df_sst_compound_subset = ds_sst_compound.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
        ds_sst_freebndry_subset = ds_sst_freebndry.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)

        # df_closest_events = df_sst_compound_subset.iloc[(df_sst_compound_subset['node_flooding_cubic_meters']-node_flooding).abs().argsort()[:min_storms_to_analyze]]
        # filter events

        df_bs_CIs_subset = df_bs_CIs[(df_bs_CIs.node == node) & (df_bs_CIs.flood_return_yrs==flood_return_yrs)]

        upper_bound = float(df_bs_CIs_subset["upper_CI"])
        if upper_bound <= 0: # if the upper bound is 0, skip the return period 
            continue
        upper_cond = df_sst_compound_subset.node_flooding_cubic_meters.values <= upper_bound
        lower_bound = float(df_bs_CIs_subset["lower_CI"])
        lower_cond = ((df_sst_compound_subset.node_flooding_cubic_meters.values >= lower_bound) & (df_sst_compound_subset.node_flooding_cubic_meters.values > 0)) # don't want to include events with no flooding

        df_closest_events_filtered = df_sst_compound_subset[(upper_cond) & (lower_cond)]

        df_compare = df_closest_events_filtered.merge(ds_sst_freebndry_subset, on = ["storm_id", "realization", "year"], suffixes = ("_cmpnd", "_free"))

        df_compare['frac_wlevel'] = 1 -df_compare["node_flooding_cubic_meters_free"] / df_compare["node_flooding_cubic_meters_cmpnd"]

        df_compare['node_id'] = node

        df_compare['flood_return_yrs'] = flood_return_yrs

        df_frac_wlevel_var = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.var().reset_index().rename(dict(frac_wlevel = "frac_wlevel_var"), axis = 1)
        df_frac_wlevel_mean = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.mean().reset_index().rename(dict(frac_wlevel = "frac_wlevel_mean"), axis = 1)
        df_frac_wlevel_min = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.min().reset_index().rename(dict(frac_wlevel = "frac_wlevel_min"), axis = 1)
        df_frac_wlevel_max = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.max().reset_index().rename(dict(frac_wlevel = "frac_wlevel_max"), axis = 1)
        df_frac_wlevel_median = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.median().reset_index().rename(dict(frac_wlevel = "frac_wlevel_median"), axis = 1)


        df_attribution_stats = pd.merge(df_frac_wlevel_var, [df_frac_wlevel_mean, df_frac_wlevel_min,df_frac_wlevel_max, df_frac_wlevel_median], on = ["node_id", "flood_return_yrs"])
        
        df_out = pd.merge(df_bs_CIs_subset, df_attribution_stats, left_on = ["node", "flood_return_yrs"], right_on = ["node_id", "flood_return_yrs"])
        
        df_out["num_strms"] = df_compare.shape[0]

        lst_ds_comp.append(df_out)

df_comparison = pd.concat(lst_ds_comp)
df_comparison.to_csv(f_bootstrapping_analysis, index=False)