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
g = sns.ecdfplot(data=df_node_variability, x=std_varname, ax = ax)
plt.axhline(y = quant_top_var, color = 'r', linestyle = '-') 
ax.set_xlabel("Standard Deviation of Mean Attribution Across Return Periods")
ax.set_ylabel("Empirical Cumulative Probability")
plt.savefig(fldr_swmm_analysis_plots + "f_empirical_cdf_of_variability.png",
            transparent=False, bbox_inches='tight')

# df_counts = pd.concat(lst_dfs)
# df_counts = df_counts.set_index(["return_period", "frac_wlevel_upper_range"])

str_percentile = str(int(round((1 - quant_top_var)*100,0)))+"%"
title = "Histogram of attribution standard deviation for top {} variable nodes".format(str_percentile)

# subset based on selected quant
df_node_variability_subset = df_node_variability[df_node_variability[std_varname] >= df_node_variability[std_varname].quantile(quant_top_var)]
df_node_attribution_subset = df_node_attribution.join(df_node_variability_subset, how = "right")

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

# df_node_attribution["variable_node"] = False
# index_values_to_update = df_node_attribution_subset.index.values
# df_node_attribution.loc[index_values_to_update, "variable_node"] = True

# import seaborn as sns
# fig, ax = plt.subplots(figsize = [8, 6], dpi=300)
# sns.violinplot(data = df_node_attribution_subset.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_median",cut = 0, ax=ax)
# title = "Top {} variable nodes".format(str_percentile)
# ax.set_title(title)
# plt.tight_layout()
# plt.savefig(fldr_swmm_analysis_plots + "f_violin_plt_attribution_variable_nodes.png",
#             transparent=False, bbox_inches='tight')


# fig, ax = plt.subplots(figsize = [8, 6], dpi=300)
# sns.violinplot(data = df_node_attribution.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_median",cut = 0, ax=ax)
# title = "All nodes"
# ax.set_title(title)
# plt.tight_layout()
# plt.savefig(fldr_swmm_analysis_plots + "f_violin_plt_attribution_allnodes.png",
#             transparent=False, bbox_inches='tight')


#%%
df_node_attribution["variable_node"] = False
index_values_to_update = df_node_attribution_subset.index.values
df_node_attribution.loc[index_values_to_update, "variable_node"] = True

df_node_attribution_gt_1yr = df_node_attribution.loc[df_node_attribution.index.get_level_values('flood_return_yrs') >= 0.5]
plt_scale = .9

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = [10*plt_scale, 5*plt_scale], dpi=300)
sns.violinplot(data=df_node_attribution_gt_1yr, x="flood_return_yrs", y="frac_wlevel_mean", hue="variable_node",
               cut = 0,
               split=True, inner="quart", fill=True,
               palette={True: "g", False: ".35"}, ax = ax)
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
        legend_labels.append("All Nodes")

ax.legend(handles, legend_labels, title="", title_fontsize="15", ncols=2, bbox_to_anchor=(0.5, 1.08), loc='upper center',)

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
#%%

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


#%% attempt 2 at plotting flood return period vs. event return period
# df_return_pd_analysis = pd.read_csv(f_return_pd_analysis) # .set_index(["realization", "year", "storm_id", "node_id"])
# find unique events
# lst_strm_id_vals = storm_id_variables

# unique_storms_with_flding = df_return_pd_analysis.loc[:,lst_strm_id_vals].drop_duplicates()
# unique_storms_with_flding = unique_storms_with_flding.sort_values(lst_strm_id_vals).reset_index(drop=True)

# df_events_subset = df_events.join(unique_storms_with_flding.set_index(lst_strm_id_vals), how = "right")

# df_flood_and_event_returns = df_return_pd_analysis.set_index(["realization", "year", "storm_id", "node_id"]).join(df_events_subset)

# df_sst_compound = ds_sst_compound.to_dataframe()
# ranking using xarray
# df_sst_compound_ranked = ds_sst_compound.node_flooding_cubic_meters.rank("node_id").to_dataframe()

# ranking using pandas dataframe
# rank_using_df = df_sst_compound.reset_index().loc[:,["node_id", "node_flooding_cubic_meters"]].groupby("node_id").rank(ascending = False, method = "average")
# rank_using_df.columns = ["flooding_ranked"]


# rank_using_df_with_event_info = df_sst_compound.reset_index().join(rank_using_df)
# max_ranks = rank_using_df_with_event_info.loc[:,["node_id", "flooding_ranked"]].groupby("node_id").max()
# max_ranks.columns = ["max_rank"]

# df_flood_and_event_returns = rank_using_df_with_event_info.set_index(["realization", "year", "storm_id", "node_id"]).join(df_events)
# df_flood_and_event_returns = df_flood_and_event_returns.join(max_ranks)

# nyears_modeled_in_swmm = df_sst_compound.reset_index().realization.max() * df_sst_compound.reset_index().year.max()
# nsimulations = nyears_modeled_in_swmm * df_sst_compound.reset_index().storm_id.max()

# df_flood_and_event_returns["flood_empirical_quantile"] = (df_flood_and_event_returns.flooding_ranked / df_flood_and_event_returns.max_rank).values
# df_flood_and_event_returns["flood_empirical_return_yr"] = (df_flood_and_event_returns.flood_empirical_quantile)*nyears_modeled_in_swmm

# trying to use scipy
#%% plotting event returns vs. node flooding returns
from scipy.stats.mstats import plotting_positions
plotting_positions
df_sst_compound = ds_sst_compound.to_dataframe()
nyears_modeled_in_swmm = df_sst_compound.reset_index().realization.max() * df_sst_compound.reset_index().year.max()
# sort index values
df_sst_compound = df_sst_compound.sort_index()
# extract sorted values of simulated storms
# simulated_storms_all = df_sst_compound.reset_index().loc[:, storm_id_variables]
# simulated_storms_all = simulated_storms_all.drop_duplicates().reset_index(drop = True).sort_values(storm_id_variables)
# only keep the n_largest storms per year
def keep_largest_n_events(df_to_subset, n_largest_to_keep, str_rank_var, lst_group_vars,):
    variable_ranked_and_subset = df_to_subset.loc[:, str_rank_var].sort_values(ascending = False).groupby(level = lst_group_vars).head(n_largest_to_keep)
    variable_ranked_and_subset = variable_ranked_and_subset.sort_index()
    df_nlargest = df_to_subset.loc[variable_ranked_and_subset.index.values,:]
    # confirm tha tthe 2 largest flood values are indeed being returned
    # df_nlargest.loc[(1,1,slice(None),"UN90"),:]
    # df_to_subset.loc[(1,1,slice(None),"E1331931"),:]
    print("Keeping {}% of the data.".format((len(variable_ranked_and_subset) / len(df_to_subset))*100))
    return df_nlargest
# df_events_n_largest = keep_largest_n_events(df_events, n_largest_to_keep=2, str_rank_var="max_mm_per_hour",
#                                              lst_group_vars=["realization", "year"])

df_sst_compound_n_largest = keep_largest_n_events(df_sst_compound, n_largest_to_keep=2, str_rank_var="node_flooding_cubic_meters",
                                                  lst_group_vars=["realization", "year", "node_id"])
# simulated_storms_subset = df_sst_compound_n_largest.reset_index().loc[:, storm_id_variables]
# simulated_storms_subset = simulated_storms_subset.drop_duplicates().reset_index(drop = True)
# compute plotting positions for all nodes (weibull)
emp_cdf = df_sst_compound_n_largest.loc[:,"node_flooding_cubic_meters"].groupby(level = "node_id").apply(plotting_positions, alpha=0, beta=0)
# create a dataframe with the plotting positions
lst_dfs = []
for node, row in emp_cdf.items():
    storms = df_sst_compound_n_largest.loc[(slice(None),slice(None),slice(None),node)].reset_index().loc[:,storm_id_variables]
    df_emp_cdf_loop = pd.DataFrame(dict(flood_empirical_quantile = row))
    df_emp_cdf_loop["node_id"] = node
    df_emp_cdf_loop["flood_empirical_return_yr"] = row * nyears_modeled_in_swmm
    df_emp_cdf_loop = df_emp_cdf_loop.join(storms)
    # add storm id info to the dataframe for re-joining
    lst_dfs.append(df_emp_cdf_loop)

df_empicial_cdf_vals_by_node = pd.concat(lst_dfs, ignore_index = True)

# join df_events with the simulated cdf values

df_flood_and_event_returns = df_empicial_cdf_vals_by_node.set_index(["realization", "year", "storm_id", "node_id"]).join(df_events)
df_flood_and_event_returns = df_flood_and_event_returns.join(df_sst_compound)

print("The minimum empirical return period for flooding was {}".format(df_flood_and_event_returns.flood_empirical_return_yr.min()))
# find unique events
vars_of_interest = ["node_flooding_cubic_meters", "flood_empirical_return_yr", "empirical_event_return_pd_yr",
                    'max_mm_per_hour_emp_return_pd_yr', 'max_surge_ft_emp_return_pd_yr', 'depth_mm_emp_return_pd_yr']
df_flood_and_event_returns_no_zeros = df_flood_and_event_returns.loc[:,vars_of_interest].copy()
# remove all rows where flooding is 0
df_flood_and_event_returns_no_zeros = df_flood_and_event_returns_no_zeros[df_flood_and_event_returns_no_zeros.node_flooding_cubic_meters>0].copy()
df_flood_and_event_returns_no_zeros["node_flooding_order_of_magnitude"] = np.log10(df_flood_and_event_returns_no_zeros.node_flooding_cubic_meters)
# only include flooding of 1 cubic meter or more
# df_flood_and_event_returns = df_flood_and_event_returns[df_flood_and_event_returns["node_flooding_order_of_magnitude"]>=0]
print("After removing node events with 0 flooding, the minimum empirical return period for flooding was {}".format(df_flood_and_event_returns_no_zeros.flood_empirical_return_yr.min()))
#%% plotting
import matplotlib as mpl


def plot_flood_return_vs_other_return(df, other_var, ylabel, ax = None, show_cbar = False, 
                                      show_ylabel = False, show_xlabel = False, savefig = False,
                                      point_size_scaler = 1, show_xticks= True,
                                      show_yticks = True, vmax = None):
    marker_size = df.node_flooding_order_of_magnitude**2/3 * point_size_scaler
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    if vmax is None:
        vmax = df.node_flooding_order_of_magnitude.quantile(0.98)
    artist = df.plot.scatter(ax=ax, 
                            x = "flood_empirical_return_yr",
                            y = other_var,
                            c = "node_flooding_order_of_magnitude",
                            s = marker_size,
                            vmin = 0,
                            vmax = vmax,
                            logx = True,
                            logy = True,
                            cmap = "viridis",
                            colorbar = False,
                            #    norm = norm,
                            alpha = 0.9,
                            zorder = 8)
    ax.grid(zorder = 5, alpha = 0.7, which = 'both')
    ax.set_ylim(1, 500)
    ax.set_xlim(1, 500)
    
    if show_xlabel:
        ax.set_xlabel("Empirical Node Flooding Return (yr)")
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(ylabel +  " Return (yr)")
    else:
        ax.set_ylabel("")

    if show_xticks == False:
        ax.set_xticklabels([])
    if show_yticks == False:
        ax.set_yticklabels([])


    ax.axline(xy1=(0,0), xy2=(100,100), zorder=10, c = "red", alpha = 0.5)
    if show_cbar:
        cbar = ax.figure.colorbar(ax.collections[0], label = "log$_{10}$ of node flooding (m$^3$)",
                           shrink = 1, orientation = "horizontal", location = "top",
                           anchor = (0.5, 0), aspect = 35)
        # cbar = ax.figure.colorbar(ax.collections[0], label = "log$_{10}$ of node flooding (m$^3$)",
        #                    aspect = 35)
        tick_locator = ticker.MaxNLocator(nbins = 4)
        cbar.locator = tick_locator
        cbar.update_ticks()

    # fig.colorbar(artist, ax = ax, shrink = 0.6, label = "log$_10$(node flooding m$^3$)")
    if savefig:
        plt.savefig(fldr_swmm_analysis_plots + "f_empirical_flooding_vs_{}.png".format(other_var),
                    transparent=False, bbox_inches='tight')


# plot_flood_return_vs_other_return(other_var, ylabel)
#%%
other_vars = ["empirical_event_return_pd_yr",'max_mm_per_hour_emp_return_pd_yr', 'max_surge_ft_emp_return_pd_yr', 'depth_mm_emp_return_pd_yr']
ylabels = ["Compound Event", "Rain Intensity", "Storm Surge", "Rain Depth"]

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker

scaling = 0.8

# Create a 2x2 grid for the plots
fig = plt.figure(figsize=(8*scaling, 10*scaling), layout = 'constrained')
gs = GridSpec(2, 3, height_ratios = [1,.3], figure = fig)
# top
ax1 = plt.subplot(gs[0, :])
# bottom left
ax2 = plt.subplot(gs[1, 0])
# bottom middle
ax3 = plt.subplot(gs[1, 1])
# bottom right
ax4 = plt.subplot(gs[1, 2])

lst_axes = [ax1, ax2, ax3, ax4]

df_for_plotting = pd.DataFrame(dict(other_var = other_vars, ylabel = ylabels))

df_data = df_flood_and_event_returns_no_zeros#.sample(1000)

base_pt_size = 1.3
for ind, row in df_for_plotting.iterrows():
    show_cbar = False
    show_ylabel = True
    point_size_scaler = 0.33*base_pt_size
    show_xticks = True
    show_yticks = False
    if ind == 0:
        show_cbar = True
        point_size_scaler = 1*base_pt_size
        show_yticks = True
        show_xticks = True
    show_xlabel = False
    if ind == 1:
        show_yticks = True
        show_xticks = True
        # show_xlabel = True
    if ind == 2:
        show_xlabel = True
    plot_flood_return_vs_other_return(df_data, row.other_var, row.ylabel, ax = lst_axes[ind],
                                       show_cbar=show_cbar, show_ylabel=show_ylabel,
                                       show_xlabel=show_xlabel, point_size_scaler=point_size_scaler,
                                       show_xticks= show_xticks, show_yticks = show_yticks,
                                       vmax = 4)
plt.savefig(fldr_swmm_analysis_plots + "f_empirical_flooding_vs_all_return_pds.png",
                    transparent=False, bbox_inches='tight', dpi = 400)
#%% plotting flood return period vs. rainfall and storm surge return period
lst_weather_stats = ["max_sim_wlevel", "depth_mm"]
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
    