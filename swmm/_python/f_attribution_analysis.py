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
# load and process data
ds_sst = xr.open_dataset(f_sst_results_hrly)
ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap_hrly)
df_comparison = pd.read_csv(f_bootstrapping_analysis)
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

ds_sst_compound = ds_sst.sel(freeboundary="False")
ds_sst_freebndry = ds_sst.sel(freeboundary="True")
ds_fld_dif = ds_sst_compound - ds_sst_freebndry

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


# compute quantiles
ds_quants = ds_sst_compound.quantile(quants, dim = ["storm_id", "realization", "year"], method="closest_observation")
ds_quants = ds_quants.assign_coords(dict(quantile = sst_recurrence_intervals))
ds_quants = ds_quants.rename((dict(quantile="flood_return_yrs")))

ds_quants["node_trns_flooding_cubic_meters"] = np.log10(ds_quants["node_flooding_cubic_meters"]+.01)
df_quants = ds_quants.to_dataframe()
df_quants = df_quants.reset_index()
gdf_node_flding = gdf_nodes.merge(df_quants, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

# plotting return periods
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