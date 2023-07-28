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
width_to_height = 7.5 / 3
height = 5
width = height * width_to_height 

# outlier cutoff:
outlier_cutoff = 1e10 # cubic meters

from _inputs import e_isolating_surge_effects

f_sst_results, sst_conf_interval, sst_recurrence_intervals, f_bootstrap_hrly, f_bootstrap_raw_hrly, f_shp_jxns, f_shp_strg, f_shp_out, f_shp_coast, f_shp_subs  = e_isolating_surge_effects()

scratch_folder = "_scratch/"
scratch_file = "_scratch/{}"

cmap = "gist_rainbow"

#%%
ds_sst = xr.open_dataset(f_sst_results)
ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap_hrly)
# ds_bootstrap_raw = xr.open_dataset(f_bootstrap_raw_hrly, chunks=dict(node_id=1))
# df_model_perf = pd.read_csv(f_model_perf_summary)
# df_sst_events = pd.read_csv(f_sst_event_summaries)

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

#%% preprocessing
ds_sst_compound = ds_sst.sel(freeboundary="False")
ds_sst_freebndry = ds_sst.sel(freeboundary="True")
ds_fld_dif = ds_sst_compound - ds_sst_freebndry

#%% trying to figure out what's going on with flooded volumes being mostly 0
# ds_sum_floding = ds_sst_compound.sum(dim=["node_id"])
# event_idx_most_flooding = ds_sum_floding.node_flooding_cubic_meters.argmax(dim=["storm_id", "realization", "year"])

# for key in event_idx_most_flooding:
#     newval = event_idx_most_flooding[key].values
#     event_idx_most_flooding[key] = newval

# df_biggest_event = ds_sst_compound.isel(event_idx_most_flooding).to_dataframe().reset_index()


# df_all_flding = ds_sst_compound.to_dataframe()
# # remove outliers
# df_all_flding = df_all_flding[df_all_flding.node_flooding_cubic_meters < outlier_cutoff]
# df_all_flding = df_all_flding[df_all_flding.node_flooding_cubic_meters > 0.01].reset_index()
# df_all_flding['log_flooding'] = np.log10(df_all_flding.node_flooding_cubic_meters)

# df_all_flding.log_flooding.plot.hist()

# # isolate outliers
# df_all_flding = ds_sst_compound.to_dataframe().reset_index()
# df_all_flding['log_flooding'] = np.log10(df_all_flding.node_flooding_cubic_meters)
# df_all_flding = df_all_flding[df_all_flding.node_flooding_cubic_meters > outlier_cutoff]
# df_all_flding = df_all_flding.sort_values("log_flooding")

# ## find events with just 1 flooded node
# df_1_flded_node = df_all_flding.groupby(["storm_id", "realization", "year"]).count()
# df_1_flded_node = df_1_flded_node[df_1_flded_node.node_id == 1]
# df_1_flded_node = df_1_flded_node.reset_index().drop(["node_id", "node_flooding_cubic_meters", "freeboundary", "log_flooding"], axis=1)
# df_1_flded_node = df_1_flded_node.merge(df_all_flding, how = "inner", on = ["storm_id", "realization", "year"])
# #%% inspecting node S1
# df_s1 = ds_sst_compound.sel(dict(node_id = "S1")).to_dataframe()
# df_s1 = df_s1[df_s1.node_flooding_cubic_meters>0]
# df_s1['log_flooding'] = np.log10(df_s1.node_flooding_cubic_meters)
# df_s1 = df_s1.dropna().sort_values(by = "log_flooding").reset_index()

# #%% inspecting results for rz 1 year 12 storm 1
# df_r1y12s1 = ds_sst_compound.sel(dict(storm_id = 1, realization = 1, year = 12)).to_dataframe().reset_index()

# #%% doing quantile analysis on node E143254
# df_E143254 = ds_sst_compound.sel(dict(node_id = "E143254")).to_dataframe().reset_index()


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

# investigate the counts
df_quants[df_quants["node_flooding_cubic_meters"]> 0].value_counts("node_id").sort_values()

#%%
gdf_node_flding = gdf_nodes.merge(df_quants, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

#%% plotting
# fig, axes = plt.subplots(1, len(sst_recurrence_intervals), figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)
vmax = gdf_node_flding.node_trns_flooding_cubic_meters.max()
count = -1
title_return_pd = "{}_yr_flood_vol_recurrence_interval.png"
for rtrn in sst_recurrence_intervals:
    fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)

    count += 1

    gdf_subs.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    gdf_subset = gdf_node_flding[gdf_node_flding.flood_return_yrs == rtrn]

    gdf_subset.plot(ax=ax, column="node_trns_flooding_cubic_meters",
                                     vmin=0, vmax=vmax,
                                     cmap="Blues", edgecolor="black", legend=True,
                                     missing_kwds=dict(color="none", edgecolor="lightgrey", label = "missing values"))
    ax.set_title("\n \n {} year log(flood volume in m$^3$)".format(rtrn))
    plt.tight_layout()
    plt.savefig(scratch_file.format(title_return_pd.format(rtrn)), transparent=False, bbox_inches='tight')



#%% quick and imprecise variance analysis by return period
# node = "E143722"
# max_storms_to_analyze = 100
# max_margins = 0.1 # I don't want to look at storms that are 10% bigger or smaller than the return period
# lst_ds_comp = []
# for node in tqdm(np.unique(ds_sst_compound.node_id.values)):
#     for flood_return_yrs in sst_recurrence_intervals:
#         node_flooding = df_quants[(df_quants.node_id==node) & (df_quants.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values
#         df_sst_compound_subset = ds_sst_compound.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
#         ds_sst_freebndry_subset = ds_sst_freebndry.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
#         n = 0
#         nearby_storm_margin = 0.01

#         # df_closest_events = df_sst_compound_subset.iloc[(df_sst_compound_subset['node_flooding_cubic_meters']-node_flooding).abs().argsort()[:min_storms_to_analyze]]
#         # filter events

#         upper_bound = node_flooding * (1+max_margins)
#         upper_cond = df_sst_compound_subset.node_flooding_cubic_meters.values < upper_bound
#         lower_bound = node_flooding * (1-max_margins)
#         lower_cond = df_sst_compound_subset.node_flooding_cubic_meters.values > lower_bound

#         df_closest_events_filtered = df_sst_compound_subset[(upper_cond) & (lower_cond)]

#         df_compare = df_closest_events_filtered.merge(ds_sst_freebndry_subset, on = ["storm_id", "realization", "year"], suffixes = ("_cmpnd", "_free"))

#         df_compare['frac_wlevel'] = 1 -df_compare["node_flooding_cubic_meters_free"] / df_compare["node_flooding_cubic_meters_cmpnd"]

#         df_compare['node_id'] = node

#         df_compare['flood_return_yrs'] = flood_return_yrs
        
#         lst_ds_comp.append(df_compare)

# df_comparison = pd.concat(lst_ds_comp)
# df_comparison.to_csv(scratch_file.format("df_comparison_quantmethodclosestobs.csv"))

    

#%% computing bootstrap confidence intervals around flood volumes
lst_ds_bs_CIs = []
for node in tqdm(np.unique(ds_sst_compound.node_id.values)):
    count = -1
    ds_bs_node = ds_bootstrap_rtrn.sel(dict(node_id = node))
    if ds_bs_node.node_flooding_cubic_meters.sum().values > 0:
        # print(node)
        for flood_return_yrs in ds_bootstrap_rtrn.flood_return_yrs.values:
            count += 1
            quant = quants[count]
            node_flooding = df_quants[(df_quants.node_id==node) & (df_quants.flood_return_yrs == flood_return_yrs)].node_flooding_cubic_meters.values
            # df_sst_compound_subset = ds_sst_compound.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
            # ds_sst_freebndry_subset = ds_sst_freebndry.sel(dict(node_id = node)).to_dataframe().reset_index().drop(["freeboundary", "node_id"], axis = 1)
            # n = 0

            # df_closest_events = df_sst_compound_subset.iloc[(df_sst_compound_subset['node_flooding_cubic_meters']-node_flooding).abs().argsort()[:min_storms_to_analyze]]
            # filter events
            # work
            lower_quant =  (1-sst_conf_interval) / 2
            upper_quant = 1 - lower_quant
            ds_bs_subset = ds_bs_node.sel(dict(flood_return_yrs = flood_return_yrs))
            upper_bound = float(ds_bs_subset.quantile(q = upper_quant, dim = "bootstrap_sample").node_flooding_cubic_meters.values)
            # upper_cond = df_sst_compound_subset.node_flooding_cubic_meters.values <= upper_bound
            lower_bound = float(ds_bs_subset.quantile(q = lower_quant, dim = "bootstrap_sample").node_flooding_cubic_meters.values)
            # lower_cond = df_sst_compound_subset.node_flooding_cubic_meters.values >= lower_bound

            # end work

            # df_closest_events_filtered = df_sst_compound_subset[(upper_cond) & (lower_cond)]

            # df_compare = df_closest_events_filtered.merge(ds_sst_freebndry_subset, on = ["storm_id", "realization", "year"], suffixes = ("_cmpnd", "_free"))

            # df_compare['frac_wlevel'] = 1 -df_compare["node_flooding_cubic_meters_free"] / df_compare["node_flooding_cubic_meters_cmpnd"]

            # df_compare['node_id'] = node

            # df_compare['flood_return_yrs'] = flood_return_yrs

            # df_frac_wlevel_var = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.var().reset_index().rename(dict(frac_wlevel = "frac_wlevel_var"), axis = 1)
            # df_frac_wlevel_mean = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.mean().reset_index().rename(dict(frac_wlevel = "frac_wlevel_mean"), axis = 1)
            

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

        # DCL WORK
        df_frac_wlevel_var = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.var().reset_index().rename(dict(frac_wlevel = "frac_wlevel_var"), axis = 1)
        df_frac_wlevel_mean = df_compare.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.mean().reset_index().rename(dict(frac_wlevel = "frac_wlevel_mean"), axis = 1)

        df_attribution_stats = pd.merge(df_frac_wlevel_var, df_frac_wlevel_mean,  on = ["node_id", "flood_return_yrs"])
        
        df_out = pd.merge(df_bs_CIs_subset, df_attribution_stats, left_on = ["node", "flood_return_yrs"], right_on = ["node_id", "flood_return_yrs"])
        
        df_out["num_strms"] = df_compare.shape[0]
        # print(df_compare.shape[0])
        # END WORK
        lst_ds_comp.append(df_out)

df_comparison = pd.concat(lst_ds_comp)
df_comparison.to_csv(scratch_file.format("df_comparison_bootstrapped.csv"), index=False)


#%% analysis
# group by node and recurrence interval and compute statistics on frac_wlevel (e.g., mean, variance)
# planning on coloring by mean and sizing points based on variance
df_comparison = pd.read_csv(scratch_file.format("df_comparison_bootstrapped.csv"))

# df_comparison.value_counts(["flood_return_yrs", "node_id"])
# # df_comparison = df_comparison.dropna()
# df_frac_wlevel_var = df_comparison.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.var().reset_index().rename(dict(frac_wlevel = "frac_wlevel_var"), axis = 1)
# df_frac_wlevel_mean = df_comparison.groupby(["node_id", "flood_return_yrs"]).frac_wlevel.mean().reset_index().rename(dict(frac_wlevel = "frac_wlevel_mean"), axis = 1)

# df_attribution_stats = pd.merge(df_frac_wlevel_var, df_frac_wlevel_mean,  on = ["node_id", "flood_return_yrs"])
# merge with the df with the quantile estimate

# merge with the geodataframe for plotting
gdf_node_attribution = gdf_nodes.merge(df_comparison, how = 'inner', left_on = "NAME", right_on = "node").drop("NAME", axis=1)

gdf_nodes_w_flding = gpd.GeoDataFrame(geometry=gdf_node_attribution.geometry.unique())

# nodes = pd.Series(gdf_node_attribution.geometry.unique())

# gdf_nodes_w_flding = gdf_node_attribution.node == nodes

# gdf_node_attribution = gdf_node_attribution.merge(df_bs_CIs, how = 'inner', left_on = ["node_id", "flood_return_yrs"], right_on = ["node", "flood_return_yrs"]).drop("node", axis=1)
#%% plotting
# fig, axes = plt.subplots(1, len(sst_recurrence_intervals), figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)
# count = -1
width_to_height = 7.5 / 4
height = 5
width = height * width_to_height 

title_fld_attribution = "{}_yr_return_flding_quantclosest_flood_vol_attribution.png"

for rtrn in sst_recurrence_intervals:
    fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)

    # count += 1
    gdf_subset = gdf_node_attribution[gdf_node_attribution.flood_return_yrs == rtrn]
    gdf_subset["marker_size"] = ((gdf_subset["frac_wlevel_var"]+1)**8)
    s_ranks = gdf_subset["frac_wlevel_var"].rank()
    gdf_subset["marker_size"] = (s_ranks/s_ranks.max()+1)*10

    gdf_subs.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # gdf_coast.plot(ax=ax, color='black', zorder=1)
    gdf_nodes_w_flding.plot(ax = ax, color = "none", edgecolor = "black", zorder = 1, alpha = 0.7,
                   linewidths = 0.5)

    gdf_subset.plot(ax=ax, column="frac_wlevel_mean",
                                     vmin=0, vmax=1,
                                    #  alpha = 0.7,
                                    #  markersize = "marker_size",
                                     cmap="plasma", edgecolor="none", legend=True,
                                     missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))

    
    ax.set_title("Average flood volume fraction attributable to downstream water level \n of all events causing flood volumes within a {}% bootstrapped confidence interval \n of the {} year flood volume for each node".format(int(sst_conf_interval*100), rtrn))
    plt.tight_layout()
    plt.savefig(scratch_file.format(title_fld_attribution.format(rtrn)),
                transparent=False, bbox_inches='tight')

#%% making an animation
# https://towardsdatascience.com/how-to-create-a-gif-from-matplotlib-plots-in-python-6bec6c0c952c
import imageio
from glob import glob


def create_gif(files, gif_filepath):
    lst_returns = []
    for f in files:
        lst_returns.append(float(f.split("\\")[1].split("_")[0]))

    df_files = pd.DataFrame(dict(return_periods = lst_returns,
                                files = files))

    df_files = df_files.sort_values("return_periods")

    frames = []

    for i, row in df_files.iterrows():
        if row["return_periods"] in lst_exluded_returns:
            continue
        f = row["files"]
        image = imageio.v2.imread(f)
        frames.append(image)

    imageio.mimsave(gif_filepath, # output gif
                    frames,          # array of input frames
                    duration = 1000*1.8,
                    loop = 0)         # lop=op - 0 means indefinite (https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.pillow_legacy.html)
    

#%%
lst_exluded_returns = [0.1, 0.25]

files = glob(scratch_file.format(title_return_pd.format("*")))
gif_filepath = scratch_folder + "recurrence_intervals.gif"
create_gif(files, gif_filepath)

files = glob(scratch_file.format(title_fld_attribution.format("*")))
gif_filepath = scratch_folder + "flood_attribution.gif"
create_gif(files, gif_filepath)