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