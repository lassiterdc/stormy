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
from _utils import *
import os

# plotting parameters
width_to_height = 7.5 / 4
height = 5
width = height * width_to_height 

# outlier cutoff:
outlier_cutoff = 1e10 # cubic meters

cmap = "gist_rainbow"


#%% inspecting water level range across nodes
size_scaler = 1.6
fig, ax = plt.subplots(dpi=300, figsize=(2.5,4.5))
g = sns.ecdfplot(data=df_node_variability, x=std_varname, ax = ax)
plt.axhline(y = quant_top_var, color = 'r', linestyle = '-') 
ax.set_xlabel("Standard Deviation \nof Mean Attribution")
ax.set_ylabel("Empirical Cumulative Probability")
ax.axhspan(quant_top_var, 1, facecolor='green', alpha=0.5, label='Shaded Area')
ax.axhspan(quant_top_var, 0, facecolor='grey', alpha=0.5, label='Shaded Area')
plt.savefig(fldr_swmm_analysis_plots + "f_empirical_cdf_of_variability.png",
            transparent=False, bbox_inches='tight')

# df_counts = pd.concat(lst_dfs)
# df_counts = df_counts.set_index(["return_period", "frac_wlevel_upper_range"])

str_percentile = str(int(round((1 - quant_top_var)*100,0)))+"%"
title = "Histogram of attribution standard deviation for top {} variable nodes".format(str_percentile)

# subset based on selected quant
df_node_variability_subset = df_node_variability[df_node_variability[std_varname] >= df_node_variability[std_varname].quantile(quant_top_var)]
df_node_attribution_subset = df_node_attribution.join(df_node_variability_subset, how = "right")
#%%
# fig, ax = plt.subplots(dpi=300)
# ax.set_title(title)
# sns.histplot(data=df_node_variability_subset, x=std_varname, ax=ax)
# plt.savefig(fldr_swmm_analysis_plots + "f_histogram_of_variability.png",
#             transparent=False, bbox_inches='tight')
# inspecting the top 20% variable nodes
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

gdf_variable_nodes.plot(ax=ax, column=std_varname,
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
df_node_attribution["variable_node"] = False
index_values_to_update = df_node_attribution_subset.index.values
df_node_attribution.loc[index_values_to_update, "variable_node"] = True

df_node_attribution_gt_1yr = df_node_attribution.loc[df_node_attribution.index.get_level_values('flood_return_yrs') >= 0.5]
plt_scale = .9

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = [10*plt_scale, 5*plt_scale], dpi=300)
sns.violinplot(data=df_node_attribution_gt_1yr, x="flood_return_yrs", y="frac_wlevel_mean", hue="variable_node",
               cut = 0,
               split=True, inner="quart", fill=True, alpha = 0.5,
               palette={True: "green", False: "grey"}, ax = ax)
space = "               "
ax.set_ylabel("(Rain Driven)" + space + "Flood Attribution" + space + "(Surge Driven)")
ax.set_xlabel("Flood Return Period (years)")
ax.yaxis.grid(True)
# legend = ax.legend(title="Top {} Variable Node".format(str_percentile), loc = "upper center", ncols = 2, bbox_to_anchor=(.5, 1.129), alignment = 'center')
# legend = ax.legend(title="", labels = ["All Nodes","Variable Nodes"], ncols = 2, bbox_to_anchor=(.5, 1.129), alignment = 'center')


handles, titles = ax.get_legend_handles_labels()
legend_labels = []

for title in titles:
    if title == "True":
        legend_labels.append("Variable Nodes")
    if title == "False":
        legend_labels.append("Nonvariable Nodes")

ax.legend(handles, legend_labels, title="", title_fontsize="15", ncols=2, bbox_to_anchor=(0.5, 1.1), loc='upper center',)

plt.savefig(fldr_swmm_analysis_plots + "f_violin_plt_attribution.png",
            transparent=False, bbox_inches='tight')

gdf_node_flding_variable_nodes = gdf_node_flding.merge(df_node_variability_subset, on = "node_id")
#%% creating gifs
# gdf_variable_nodes = gdf_node_attribution.merge(df_node_variability_subset, on = "node_id")


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

#%% preparing imagery for plotting
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
#%% creating plot of variable node locations colored by attribution and sized by magnitude

use_aerial_imagery = True
# vmin = 0.01 # m
# vmax = 0.5 #m
ncols = 2
nrows = 2
size_mltpl = 1.1

fig, axes = plt.subplots(2,2, dpi = 300, figsize = (8*size_mltpl,7*size_mltpl), layout = 'tight')
count = -1

target_recurrence_intervals = [1, 10, 50, 100]

for ax in axes.reshape(-1):
    count += 1
    rtrn = target_recurrence_intervals[count]

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

    if f_imagery is not None and use_aerial_imagery:
        ds_im.plot.imshow(x = "x", y = "y", ax = ax, zorder = 4, alpha = 0.6)

    # gdf_coast.plot(ax=ax, color='black', zorder=1)
    # gdf_nodes_w_flding.plot(ax = ax, color = "none", edgecolor = "black", zorder = 1, alpha = 0.7,
    #                linewidths = 0.5)
    # ax_with_cbar = gdf_subset.plot(ax=ax, column="frac_wlevel_mean",
    #                                  vmin=0, vmax=1,
    #                                  alpha = 0.8,
    #                                  markersize = 0,
    #                                  edgecolor="none",
    #                                  facecolor = "none",
    #                                  cmap="plasma", legend=False,
    #                                  zorder = 7)

                                    #  missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))
    # cmap = plt.cm.get_cmap("plasma", 5)
    # c_edges = plt.cm.plasma(gdf_subset["frac_wlevel_mean"])
    # c_edges = cmap((gdf_subset["frac_wlevel_mean"]))
    # cmap = plt.colormaps["plasma"]

    # cmap = plt.cm.get_cmap("plasma")
    # c_edges = cmap(gdf_subset["frac_wlevel_mean"])
# Create a ScalarMappable
    cmap = plt.cm.get_cmap("plasma")
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    scatter = ax.scatter(gdf_subset.geometry.x, gdf_subset.geometry.y, s=gdf_subset["marker_size"],
               edgecolor = cmap(gdf_subset["frac_wlevel_mean"]), facecolor = "none", linewidths = 1.75,
               alpha = .9, zorder = 7)
    
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    legend2 = ax.legend(handles, bin_labels, loc="upper right", title="Flood Volume",framealpha=1)
    legend2.set_zorder(10)

    # if show_xticks == False:
    ax.set_xticklabels([])
    ax.set_xlabel("")
    # if show_yticks == False:
    ax.set_yticklabels([])
    ax.set_ylabel("")
    # ax.set_title("Average flood attribution of flood events within a {}% bootstrapped confidence interval \n of the {} year flood volume for each node in the top {} variability".format(int(sst_conf_interval*100), rtrn, str_percentile))
    ax.set_title("{} Year".format(rtrn))


fig.colorbar(sm, ax = axes, shrink = 0.6, orientation = "horizontal", location = "bottom",
             anchor = (0.5,-.49), label = "Mean Flood Attribution", cmap = cmap)

text_x = 0.18  # Adjust the X-coordinate
text_y = -0.035  # Adjust the Y-coordinate
text_content = "Rain\nDriven"
fig.text(text_x, text_y, text_content, ha='center', va='bottom', fontsize=11, color='black')


text_x = 0.825  # Adjust the X-coordinate
# text_y = -0.038  # Adjust the Y-coordinate
text_content = "Surge\nDriven"
fig.text(text_x, text_y, text_content, ha='center', va='bottom', fontsize=11, color='black')


# plt.tight_layout()
title_plt = "fld_attribution_by_return_pd_sized_by_volume.png"
plt.savefig(fldr_swmm_analysis_plots + title_plt,
            transparent=False, bbox_inches='tight')
    # plt.clf()
# files = glob(fldr_swmm_analysis_plots + title_plt.format("*"))
# gif_filepath = fldr_swmm_analysis_plots + "flood_attribution_and_volume.gif"
# create_gif(files, gif_filepath)


