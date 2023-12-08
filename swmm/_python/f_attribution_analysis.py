#%% load libraries and directories
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from tqdm import tqdm
import imageio
from glob import glob
from _inputs import *
import os

# plotting parameters
width_to_height = 7.5 / 4
height = 5
width = height * width_to_height 

# outlier cutoff:
outlier_cutoff = 1e10 # cubic meters

from _inputs import *

cmap = "gist_rainbow"

#%% inspecting water level range across nodes

fig, ax = plt.subplots(dpi=300)
g = sns.ecdfplot(data=df_node_variability, x="std_of_frac_wlevel_median", ax = ax)
plt.axhline(y = quant_top_var, color = 'r', linestyle = '-') 
ax.set_xlabel("Standard Deviation of Median Attribution Across Return Periods")
ax.set_ylabel("Empirical Cumulative Probability")
plt.savefig(fldr_swmm_analysis_plots + "f_empirical_cdf_of_variability.png",
            transparent=False, bbox_inches='tight')

# df_counts = pd.concat(lst_dfs)
# df_counts = df_counts.set_index(["return_period", "frac_wlevel_upper_range"])

str_percentile = str(int(round((1 - quant_top_var)*100,0)))+"%"
title = "Histogram of attribution standard deviation for top {} variable nodes".format(str_percentile)

# subset based on selected quant
df_node_variability_subset = df_node_variability[df_node_variability["std_of_frac_wlevel_median"] >= df_node_variability["std_of_frac_wlevel_median"].quantile(quant_top_var)]
df_node_attribution_subset = df_node_attribution.join(df_node_variability_subset, how = "right")

fig, ax = plt.subplots(dpi=300)
ax.set_title(title)
sns.histplot(data=df_node_variability_subset, x="std_of_frac_wlevel_median", ax=ax)
plt.savefig(fldr_swmm_analysis_plots + "f_histogram_of_variability.png",
            transparent=False, bbox_inches='tight')
#%% inspecting the top 20% variable nodes
gdf_nodes_w_flding = gpd.GeoDataFrame(geometry=gdf_node_attribution.geometry.unique())
gdf_variable_nodes = gdf_node_attribution.merge(df_node_variability_subset, on = "node_id")
# gdf_variable_nodes = gdf_variable_nodes[gdf_variable_nodes.flood_return_yrs == 100]


fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)


gdf_subs.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# gdf_coast.plot(ax=ax, color='black', zorder=1)
gdf_nodes_w_flding.plot(ax = ax, color = "none", edgecolor = "black", zorder = 1, alpha = 0.7,
                linewidths = 0.5)

gdf_variable_nodes.plot(ax=ax, column="std_of_frac_wlevel_median",
                                    vmin=0,# vmax=1,
                                #  alpha = 0.7,
                                #  markersize = "marker_size",
                                    cmap="plasma", edgecolor="none", legend=True,
                                    missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))

title = "Top {} variable nodes".format(str_percentile)
ax.set_title(title)
plt.tight_layout()
plt.savefig(fldr_swmm_analysis_plots + "f_top_variable_nodes.png",
            transparent=False, bbox_inches='tight')


#%% violin plot
import seaborn as sns
fig, ax = plt.subplots(figsize = [8, 6], dpi=300)
sns.violinplot(data = df_node_attribution_subset.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_median",cut = 0, ax=ax)
title = "Top {} variable nodes".format(str_percentile)
ax.set_title(title)
plt.tight_layout()
plt.savefig(fldr_swmm_analysis_plots + "f_violin_plt_attribution_variable_nodes.png",
            transparent=False, bbox_inches='tight')


fig, ax = plt.subplots(figsize = [8, 6], dpi=300)
sns.violinplot(data = df_node_attribution.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_median",cut = 0, ax=ax)
title = "All nodes"
ax.set_title(title)
plt.tight_layout()
plt.savefig(fldr_swmm_analysis_plots + "f_violin_plt_attribution_allnodes.png",
            transparent=False, bbox_inches='tight')


#%% creating gifs
# gdf_variable_nodes = gdf_node_attribution.merge(df_node_variability_subset, on = "node_id")
gdf_node_flding_variable_nodes = gdf_node_flding.merge(df_node_variability_subset, on = "node_id")

quantile_for_vmax = 0.95
count = -1
title_return_pd = "{}_yr_flood_vol_recurrence_interval.png"
# delete old files
fs_old_plots = glob(fldr_scratch_plots + title_return_pd.format("*"))
for f in fs_old_plots:
    os.remove(f)

for rtrn in sst_recurrence_intervals:
    fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)

    count += 1

    gdf_subs.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    gdf_subset = gdf_node_flding_variable_nodes[gdf_node_flding_variable_nodes.flood_return_yrs == rtrn]
    gdf_subset = gdf_subset[gdf_subset.node_flooding_cubic_meters > 0]

    # trying to set the edge colors as attribution
    # gdf_subset_attribution = gdf_variable_nodes[gdf_variable_nodes.flood_return_yrs == rtrn]

    idx = gdf_subset.set_index(["node_id", "flood_return_yrs"]).index    
    gdf_subset_attribution = gdf_variable_nodes.set_index(["node_id", "flood_return_yrs"]).loc[idx]

    c_edges = plt.cm.plasma(gdf_subset_attribution["frac_wlevel_mean"])

    gdf_subset["10^3_cubic_meters"] = gdf_subset.node_flooding_cubic_meters * 10**-3

    vmax = gdf_subset["10^3_cubic_meters"].quantile(quantile_for_vmax, interpolation = "lower")
    gdf_subset.plot(ax=ax, column="10^3_cubic_meters",
                                     s=50, alpha = .5,vmin=0, vmax=vmax,
                                     cmap="Blues", edgecolor=None, legend=True, linewidths = 0)
                                    #  missing_kwds=dict(color="none", edgecolor="lightgrey", label = "missing values"))

    ax.scatter(gdf_subset.geometry.x, gdf_subset.geometry.y, s=30,
               edgecolor = c_edges, linewidths = 1.5)

    
    # gdf_subset_attribution.plot(ax=ax, column="frac_wlevel_mean",
    #                                 #  vmin=0, vmax=1,
                                    #  alpha = 0.7,
                                    #  markersize = "marker_size",
                                    #  cmap="plasma", 
                                    #  edgecolor=c_edges, legend=True, facecolors = "none")
                                    #  missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))

    ax.set_title("\n \n {} year log(flood volume in m$^3$) for top {} variable nodes".format(rtrn, str_percentile))
    plt.tight_layout()
    plt.savefig(fldr_scratch_plots + title_return_pd.format(rtrn), transparent=False, bbox_inches='tight')
    plt.clf()



title_fld_attribution = "{}_yr_return_flding_quantclosest_flood_vol_attribution.png"
# delete old files
fs_old_plots = glob(fldr_scratch_plots + title_fld_attribution.format("*"))
for f in fs_old_plots:
    os.remove(f)

for rtrn in sst_recurrence_intervals:
    fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)

    # count += 1
    gdf_subset = gdf_variable_nodes[gdf_variable_nodes.flood_return_yrs == rtrn]
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

    
    ax.set_title("Average flood attribution of flood events within a {}% bootstrapped confidence interval \n of the {} year flood volume for each node in the top {} variability".format(int(sst_conf_interval*100), rtrn, str_percentile))
    plt.tight_layout()
    plt.savefig(fldr_scratch_plots + title_fld_attribution.format(rtrn),
                transparent=False, bbox_inches='tight')
    plt.clf()
# making an animation
# https://towardsdatascience.com/how-to-create-a-gif-from-matplotlib-plots-in-python-6bec6c0c952c

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
    
lst_exluded_returns = [0.1, 0.25]

# files = glob(fldr_scratch_plots + title_return_pd.format("*"))
# gif_filepath = fldr_swmm_analysis_plots + "recurrence_intervals.gif"
# create_gif(files, gif_filepath)

files = glob(fldr_scratch_plots + title_fld_attribution.format("*"))
gif_filepath = fldr_swmm_analysis_plots + "flood_attribution.gif"
create_gif(files, gif_filepath)

#%% trying to create plot that shoes attribution as color and volume as point size
from pathlib import Path
import rioxarray as rxr
use_aerial_imagery = True
image_agg_factor = 10

f_imagery =  f_imagery_fullres
if Path(f_imagery).is_file() and use_aerial_imagery:
    imag_agg_factor = int(image_agg_factor)
    pass
else: 
    f_imagery = None
    imag_agg_factor = None

if f_imagery is not None:
    ds_im = rxr.open_rasterio(f_imagery, chunks=dict(x="1000MB"))
    ds_im = ds_im.coarsen(dict(x = imag_agg_factor, y = imag_agg_factor)).mean()
    ds_im = ds_im.rio.reproject("EPSG:4326")
    ds_im = ds_im/255
#%%
title_plt = "{}_yr_fld_attribution_sized_by_volume.png"
# delete old files
fs_old_plots = glob(fldr_scratch_plots + title_plt.format("*"))
for f in fs_old_plots:
    os.remove(f)

for rtrn in sst_recurrence_intervals:
    # DCL WORK
    # if rtrn not in [50,100]:
    #     continue
    # DCL WORK
    fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)

    # count += 1
    gdf_subset = gdf_variable_nodes[gdf_variable_nodes.flood_return_yrs == rtrn]
    # gdf_subset["marker_size"] = ((gdf_subset["frac_wlevel_var"]+1)**8)
    # s_ranks = gdf_subset["frac_wlevel_var"].rank()
    
    gdf_subset = gdf_subset.set_index(["node_id", "flood_return_yrs"])
    idx = gdf_subset.index    
    gdf_subset_flooding = gdf_node_flding_variable_nodes.set_index(["node_id", "flood_return_yrs"]).loc[idx]
    # gdf_subset_flooding["10^3_cubic_meters"] = gdf_subset_flooding.node_flooding_cubic_meters * 10**-3
    # var_to_use_for_size = 

    gdf_subset = gdf_subset.join(gdf_subset_flooding["node_flooding_cubic_meters"])
    # c_edges = plt.cm.plasma(gdf_subset_attribution["frac_wlevel_mean"])

    # np.histogram(gdf_subset[var_to_use_for_size], bins = 5)

    gdf_subset_flooding["log_of_flooding_cubic_meters"] = np.log10(gdf_subset["node_flooding_cubic_meters"])
    log_fld_floor = 1
    log_fld_ceil = np.ceil(gdf_subset_flooding["log_of_flooding_cubic_meters"].max())
    single_int_bin_edges = np.arange(log_fld_floor, log_fld_ceil+1)
    # bin_edges_skip_negatives = [log_fld_floor]
    bin_edges_skip_negatives = []
    # bin_labels = ["$<10$ $m^3$"]
    bin_labels = []
    label_pattern = "$>10^{}$ $m^3$"
    for edge in single_int_bin_edges:
        bin_edges_skip_negatives.append(edge)
        bin_labels.append(label_pattern.format(int(edge)))

    # for ind, row in gdf_subset_flooding["log_of_flooding_cubic_meters"].iter():
    #     continue

    s_size_categories = pd.cut(gdf_subset_flooding["log_of_flooding_cubic_meters"], bins=bin_edges_skip_negatives, right = True,
           labels = np.arange(1, len(bin_edges_skip_negatives)))
    s_size_categories.name = "marker_size"
        
    # max_marker_size = gdf_subset[var_to_use_for_size].quantile(quantile_for_vmax, interpolation = "lower")

    gdf_subset["order_of_magnitude"] = s_size_categories.values
    # this will exclude any flood value that is less than 10**-1 (0.1 cubic meters)
    gdf_subset = gdf_subset.dropna()
    gdf_subset["marker_size"] = (gdf_subset["order_of_magnitude"].astype(int)+1)**3
    # gdf_subset["marker_size"][gdf_subset["marker_size"] > max_marker_size] = max_marker_size
    gdf_subs.dissolve().geometry.exterior.plot(ax=ax, color="black", edgecolor="none", alpha = .7, zorder = 6)
    ax.grid(zorder = 5, alpha = 0.7)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if f_imagery is not None:
        ds_im.plot.imshow(x = "x", y = "y", ax = ax, zorder = 4, alpha = 0.6)

    # gdf_coast.plot(ax=ax, color='black', zorder=1)
    # gdf_nodes_w_flding.plot(ax = ax, color = "none", edgecolor = "black", zorder = 1, alpha = 0.7,
    #                linewidths = 0.5)
    gdf_subset.plot(ax=ax, column="frac_wlevel_mean",
                                     vmin=0, vmax=1,
                                     alpha = 0.8,
                                     markersize = 0,
                                     edgecolor="none",
                                     facecolor = "none",
                                     cmap="plasma", legend=True,
                                     zorder = 7)
                                    #  missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))
    # cmap = plt.cm.get_cmap("plasma", 5)
    # c_edges = plt.cm.plasma(gdf_subset["frac_wlevel_mean"])
    # c_edges = cmap((gdf_subset["frac_wlevel_mean"]))
    scatter = ax.scatter(gdf_subset.geometry.x, gdf_subset.geometry.y, s=gdf_subset["marker_size"],
               edgecolor = c_edges, facecolor = "none", linewidths = 1.75,
               alpha = .9, zorder = 7)
    
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend2 = ax.legend(handles, bin_labels, loc="upper right", title="Flood Volume")
    ax.set_title("Average flood attribution of flood events within a {}% bootstrapped confidence interval \n of the {} year flood volume for each node in the top {} variability".format(int(sst_conf_interval*100), rtrn, str_percentile))
    plt.tight_layout()
    plt.savefig(fldr_swmm_analysis_plots + title_plt.format(rtrn),
                transparent=False, bbox_inches='tight')
    # plt.clf()
files = glob(fldr_swmm_analysis_plots + title_plt.format("*"))
gif_filepath = fldr_swmm_analysis_plots + "flood_attribution_and_volume.gif"
create_gif(files, gif_filepath)

#%% plotting flood return period vs. rainfall and storm surge return period
df_return_pd_analysis = pd.read_csv(f_return_pd_analysis).set_index(["realization", "year", "storm_id", "node_id"])
# find unique events
lst_strm_id_vals = ["realization", "year", "storm_id"]
lst_weather_stats = ["max_sim_wlevel", "depth_mm"]
unique_storms_with_flding = df_return_pd_analysis.loc[:,lst_strm_id_vals].drop_duplicates()
unique_storms_with_flding = unique_storms_with_flding.sort_values(lst_strm_id_vals).reset_index(drop=True)

df_events_subset = df_events.join(unique_storms_with_flding.set_index(lst_strm_id_vals), how = "right")
df_events_subset = df_events_subset.loc[:, lst_weather_stats]

return_periods_for_plotting = np.arange(0.2, 2000+0.2, step = 0.2)
quants_weather = return_period_to_quantile(ds_events, return_periods_for_plotting)
ds_quants_weather, df_quants_weather = compute_return_periods(ds_events, quants_weather, return_periods_for_plotting)
df_quants_weather = df_quants_weather.loc[:, lst_weather_stats]

# assign a return period to each weather stat
# stat = "max_sim_wlevel"
lst_s_assigned_return_pds = []

max_return_pd = df_return_pd_analysis.reset_index().year.max()

for stat in lst_weather_stats:
    df_quants_weather_subset = df_quants_weather[stat].drop_duplicates(keep = "first")
    s_assigned_rtrn_pds = pd.cut(df_events_subset[stat], bins = df_quants_weather_subset, labels = df_quants_weather_subset.index.values[:-1],
                                 include_lowest = True)
    s_assigned_rtrn_pds = s_assigned_rtrn_pds.astype(float)
    s_assigned_rtrn_pds.name = "{}_return_pd_yrs".format(stat)
    missing_vals = s_assigned_rtrn_pds[s_assigned_rtrn_pds.isna()]
    if len(missing_vals)>0:
        # figure out whether the storms are really small or really large
        df_missing = df_events_subset.join(missing_vals, how = "right")
        if len(missing_vals) == 1:
            # if the missing value is the max value in the dataset,
            # assign it a return period equal to the number of years in the dataset
            if (df_missing[stat] == df_events_subset[stat].max()).values[0]:
                s_assigned_rtrn_pds = s_assigned_rtrn_pds.replace(np.nan, max_return_pd)
    lst_s_assigned_return_pds.append(s_assigned_rtrn_pds)

df_event_stat_return_pds = pd.DataFrame(lst_s_assigned_return_pds).T


# df_return_pd_analysis.join(df_event_stat_return_pds.reset_index(), on = lst_strm_id_vals, how = "left")

df_return_pd_analysis = df_return_pd_analysis.join(df_event_stat_return_pds)

#%% plot it!!!!!!
import matplotlib as mpl


cmap = plt.cm.viridis  # define the colormap
cmaplist = [cmap(i) for i in range(cmap.N)]
# define the bins and normalize
bounds = np.sort(df_return_pd_analysis.flood_return_yrs.unique())
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(dpi=300)
df_return_pd_analysis.plot.scatter(ax=ax, 
                           x = "max_sim_wlevel_return_pd_yrs",
                           y = "depth_mm_return_pd_yrs",
                           c="flood_return_yrs",
                           logx = True,
                           logy = True,
                           cmap = cmap,
                           norm = norm,
                           alpha = 0.9,
                           zorder = 8)
ax.grid(zorder = 5, alpha = 0.7)
ax.set_title("Flood Return Period vs. Boundary Condition Return Period")

plt.savefig(fldr_swmm_analysis_plots + "f_flood_return_pd_vs_bndry_return_pd.png",
            transparent=False, bbox_inches='tight')
    