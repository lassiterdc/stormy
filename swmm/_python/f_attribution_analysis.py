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

# plotting parameters
width_to_height = 7.5 / 4
height = 5
width = height * width_to_height 

# outlier cutoff:
outlier_cutoff = 1e10 # cubic meters

from _inputs import *

scratch_folder = "_scratch/"
scratch_file = "_scratch/{}"

cmap = "gist_rainbow"

# load data
_, _, _, _, _, gdf_node_flding, gdf_subs, gdf_nodes, df_comparison = return_attribution_data()

#%% plotting return periods
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
# plotting attribution statistics
# merge with the geodataframe for plotting
gdf_node_attribution = gdf_nodes.merge(df_comparison, how = 'inner', left_on = "NAME", right_on = "node").drop("NAME", axis=1)

gdf_nodes_w_flding = gpd.GeoDataFrame(geometry=gdf_node_attribution.geometry.unique())

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

files = glob(scratch_file.format(title_return_pd.format("*")))
gif_filepath = fldr_swmm_analysis_plots + "recurrence_intervals.gif"
create_gif(files, gif_filepath)

files = glob(scratch_file.format(title_fld_attribution.format("*")))
gif_filepath = fldr_swmm_analysis_plots + "flood_attribution.gif"
create_gif(files, gif_filepath)

#%% creating box and whiskers of attribution
# create xarray dataset
df = pd.DataFrame(gdf_node_attribution)
node_ids_sorted = df[df.flood_return_yrs == 100].sort_values("lower_CI", ascending=True)["node"].values

df = df.set_index(["flood_return_yrs", "node"])
df.frac_wlevel_mean[df.frac_wlevel_mean > 1] = 1
df.frac_wlevel_mean[df.frac_wlevel_mean < 0] = 0

ds = df["frac_wlevel_mean"].to_xarray()
ds = ds.sel(node = node_ids_sorted)
# sort by max flooding in 100 year storm

og_node_names = ds.node.values
ds["node"] = np.arange(len(og_node_names))
og_returns = ds.flood_return_yrs.values
ds["flood_return_yrs"] = np.arange(len(og_returns))
# plotting
vmin = -.000000000001
vmax = 1.0000000001
levels = 6
fig, ax = plt.subplots()
ds.plot.pcolormesh(cmap = "coolwarm", x = "flood_return_yrs", y = "node", vmin = vmin, vmax = vmax, ax = ax,
                   levels = levels, extend = "neither")

#%% violin plot
import seaborn as sns
sns.violinplot(data = df.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_mean",cut = 0)

#%% boxplot
df_ge1y = df[(df["frac_wlevel_mean"].reset_index().flood_return_yrs>=1).values]

df_box = df_ge1y["frac_wlevel_mean"].reset_index().pivot(index = "node", columns = "flood_return_yrs", values = "frac_wlevel_mean")

boxplot = df_box.boxplot()
boxplot.set_xlabel("Return Period")
boxplot.set_ylabel("Frac Water Level")


sns.boxplot(data=df_ge1y.reset_index(), x = "flood_return_yrs", y = "frac_wlevel_mean")
#%% inspecting counts

classes = pd.cut(df.frac_wlevel_mean.values, np.linspace(vmin, vmax, levels),labels = np.linspace(vmin,vmax, levels)[1:])
df["classes"] = np.asarray(classes)

df["classes"] = df["classes"].round(1)

lst_dfs = []
lst_returns = []

for rtrn, group in df.reset_index().groupby("flood_return_yrs"):
    if rtrn < 1:
        continue
    df_counts = group["classes"].value_counts()
    df_counts = df_counts.reset_index().sort_values("index")
    df_counts.columns = ["frac_wlevel_upper_range", "count"]
    df_counts["density"] = df_counts["count"] / df_counts["count"].sum()
    # print("Return period: {}".format(rtrn))
    # print(df_counts)
    # print("#################")
    df_counts['return_period'] = rtrn
    lst_dfs.append(df_counts)
    lst_returns.append(rtrn)

df_counts = pd.concat(lst_dfs)
df_counts = df_counts.set_index(["return_period", "frac_wlevel_upper_range"])

# plot bar chart
sns.set_theme(style="whitegrid")

# penguins = sns.load_dataset("penguins")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df_counts.reset_index(), kind="bar",
    x="return_period", y="density", hue="frac_wlevel_upper_range",
    errorbar="sd", palette="magma", alpha=.8, height=6,
)
g.despine(left=True)
g.set_axis_labels("return period", "density")
g.legend.set_title("Upper Frac")
sns.move_legend(g, "upper left", bbox_to_anchor=(.75, .5))

#
g = sns.catplot(
    data=df_counts.reset_index(), kind="bar",
    x="frac_wlevel_upper_range", y="density", hue="return_period",
    errorbar="sd", palette="magma", alpha=.8, height=6,
)
g.despine(left=True)
g.set_axis_labels("frac_wlevel_upper_range", "density")
g.legend.set_title("return_period")
sns.move_legend(g, "upper left", bbox_to_anchor=(.85, .5))

#%% inspecting water level range across nodes
lst_ranges = []
lst_nodes = []
for node, group in df_ge1y.reset_index().groupby("node"):
    frac_range = group.frac_wlevel_mean.max() - group.frac_wlevel_mean.min()
    # df_counts = group["classes"].value_counts()
    # df_counts = df_counts.reset_index().sort_values("index")
    # df_counts.columns = ["frac_wlevel_upper_range", "count"]
    # df_counts["density"] = df_counts["count"] / df_counts["count"].sum()
    # print("Return period: {}".format(rtrn))
    # print(df_counts)
    # print("#################")
    # df_counts['return_period'] = rtrn
    # lst_dfs.append(df_counts)
    lst_ranges.append(frac_range)
    lst_nodes.append(node)

df_node_ranges = pd.DataFrame(dict(node = lst_nodes, range = lst_ranges))

df_node_ranges.hist(column = "range")

sns.histplot(data=df_node_ranges, x="range")

fig, ax = plt.subplots(dpi=300)
g = sns.ecdfplot(data=df_node_ranges, x="range", ax = ax)
plt.axhline(y = 0.8, color = 'r', linestyle = '-') 
ax.set_xlabel("Frac Water Level Range per Node")
ax.set_ylabel("Empirical Cumulative Probability")


# df_counts = pd.concat(lst_dfs)
# df_counts = df_counts.set_index(["return_period", "frac_wlevel_upper_range"])
#%% inspecting the top 20% variable nodes
df_variable_nodes = df_node_ranges[df_node_ranges.range >= df_node_ranges.range.quantile(quant_top_var)]


gdf_variable_nodes = gdf_node_attribution.merge(df_variable_nodes, on = "node")
gdf_variable_nodes = gdf_variable_nodes[gdf_variable_nodes.flood_return_yrs == 100]


fig, ax = plt.subplots(figsize = [width, height], dpi=300) # , subplot_kw=dict(projection=proj)


gdf_subs.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
# gdf_coast.plot(ax=ax, color='black', zorder=1)
gdf_nodes_w_flding.plot(ax = ax, color = "none", edgecolor = "black", zorder = 1, alpha = 0.7,
                linewidths = 0.5)

gdf_variable_nodes.plot(ax=ax, column="range",
                                    vmin=0, vmax=1,
                                #  alpha = 0.7,
                                #  markersize = "marker_size",
                                    cmap="plasma", edgecolor="none", legend=True,
                                    missing_kwds=dict(color="none", edgecolor="none", label = "missing values"))


ax.set_title("Nodes in top 20% of variability")
plt.tight_layout()
plt.savefig(scratch_file.format(title_fld_attribution.format(rtrn)),
            transparent=False, bbox_inches='tight')