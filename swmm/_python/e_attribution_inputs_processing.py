#%% load libraries and directories
# environment: flood_attribution_py2
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

# cmap = "gist_rainbow"

#%% computing bootstrap confidence intervals around flood volumes
lst_ds_bs_CIs = []
for node in tqdm(np.unique(ds_sst_compound.node_id.values)):
    count = -1
    ds_bs_node = ds_bootstrap_rtrn.sel(dict(node_id = node))
    if ds_bs_node.node_flooding_cubic_meters.sum().values > 0:
        for flood_return_yrs in ds_bootstrap_rtrn.flood_return_yrs.values:
            count += 1
            quant = quants[count]
            node_flooding = df_quants_fld[(df_quants_fld.node_id==node) & (df_quants_fld.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values

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

df_bs_CIs.groupby("node_id")

lst_ds_comp = []
lst_df_node_fld_rtrn_event_id = []
for node in tqdm(df_bs_CIs.node_id.unique()):
    count = -1
    # break
    for flood_return_yrs in ds_bootstrap_rtrn.flood_return_yrs.values:
        # break
        count += 1
        quant = quants[count]
        # node_flooding = df_quants[(df_quants.node_id==node) & (df_quants.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values
        df_sst_compound_subset = ds_sst_compound.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
        ds_sst_freebndry_subset = ds_sst_freebndry.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)

        # df_closest_events = df_sst_compound_subset.iloc[(df_sst_compound_subset['node_flooding_cubic_meters']-node_flooding).abs().argsort()[:min_storms_to_analyze]]
        # filter events

        df_bs_CIs_subset = df_bs_CIs[(df_bs_CIs.node_id == node) & (df_bs_CIs.flood_return_yrs==flood_return_yrs)].set_index(["node_id", "flood_return_yrs"])

        upper_bound = float(df_bs_CIs_subset["upper_CI"].values[0])
        if upper_bound <= 0: # if the upper bound is 0, skip the return period 
            continue
        upper_cond = df_sst_compound_subset.node_flooding_cubic_meters.values <= upper_bound
        lower_bound = float(df_bs_CIs_subset["lower_CI"].values[0])
        lower_cond = ((df_sst_compound_subset.node_flooding_cubic_meters.values >= lower_bound) & (df_sst_compound_subset.node_flooding_cubic_meters.values > 0)) # don't want to include events with no flooding

        df_closest_events_filtered = df_sst_compound_subset[(upper_cond) & (lower_cond)].copy()
        df_closest_events_filtered["node_id"] = node
        df_closest_events_filtered["flood_return_yrs"] = flood_return_yrs
        lst_df_node_fld_rtrn_event_id.append(df_closest_events_filtered)

        df_compare = df_closest_events_filtered.merge(ds_sst_freebndry_subset, on = ["storm_id", "realization", "year"], suffixes = ("_cmpnd", "_free"))

        df_compare['frac_wlevel'] = 1 -df_compare["node_flooding_cubic_meters_free"] / df_compare["node_flooding_cubic_meters_cmpnd"]

        df_compare['node_id'] = node

        df_compare['flood_return_yrs'] = flood_return_yrs

        df_frac_wlevel_var = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.var().reset_index().rename(dict(frac_wlevel = "frac_wlevel_var"), axis = 1).set_index(["node_id", "flood_return_yrs"])
        df_frac_wlevel_mean = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.mean().reset_index().rename(dict(frac_wlevel = "frac_wlevel_mean"), axis = 1).set_index(["node_id", "flood_return_yrs"])
        df_frac_wlevel_min = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.min().reset_index().rename(dict(frac_wlevel = "frac_wlevel_min"), axis = 1).set_index(["node_id", "flood_return_yrs"])
        df_frac_wlevel_max = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.max().reset_index().rename(dict(frac_wlevel = "frac_wlevel_max"), axis = 1).set_index(["node_id", "flood_return_yrs"])
        df_frac_wlevel_median = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.median().reset_index().rename(dict(frac_wlevel = "frac_wlevel_median"), axis = 1).set_index(["node_id", "flood_return_yrs"])

        df_attribution_stats = df_frac_wlevel_var.join([df_frac_wlevel_mean, df_frac_wlevel_min,df_frac_wlevel_max, df_frac_wlevel_median])
        
        df_out = df_attribution_stats.join(df_bs_CIs_subset)
        # break
        
        df_out["num_strms"] = df_compare.shape[0]

        lst_ds_comp.append(df_out)

df_comparison = pd.concat(lst_ds_comp)
df_comparison.to_csv(f_bootstrapping_analysis, index=True)


df_return_pd_analysis = pd.concat(lst_df_node_fld_rtrn_event_id, ignore_index = True)
df_return_pd_analysis.to_csv(f_return_pd_analysis, index=False)