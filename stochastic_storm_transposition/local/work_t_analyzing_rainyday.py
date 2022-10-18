# environment: mrms_analysis

#%% libraries and directories
from operator import index
from turtle import color
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import itertools
import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})
import shutil
import cartopy.crs as ccrs
import geopandas as gpd
import scipy.stats as st
from tqdm import tqdm
f_out_plts = "D:/mrms_processing/plots/work_t_sst_plots/{}"

f_in_nc_storm_cat_bigthomp_rivanna = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/_reproducing_BigThompsonExample/StageIV_72hour_BigThompson_Testing_rivanna.nc"
f_in_nc_storm_cat_nrflk_stgIV_rivanna = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/norfolk_sst_stage_iv_rivanna.nc"


f_shp_bt_domain = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/_reproducing_BigThompsonExample/BigThompsonSSTdomain-polygon.shp"
f_shp_bt_basin = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/_reproducing_BigThompsonExample/Watershed/BigThompson.shp"
f_shp_nrflk_domain = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/norfolk_trans_dom_4326.shp"
f_shp_nrflk_basin = "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/Watershed/norfolk_wshed_4326.shp"

proj = ccrs.PlateCarree()

chnk_sz = "5000MB"

inches_per_mm = 1/25.4
#%% loading data
# storm catalogs
ds_bigthomp_riv = xr.open_dataset(f_in_nc_storm_cat_bigthomp_rivanna, drop_variables='time')
ds_nrflk_stageIV_riv = xr.open_dataset(f_in_nc_storm_cat_nrflk_stgIV_rivanna, drop_variables='time')

# shapefiles (transform to lat and long)
gds_shp_bt_domain = gpd.read_file(f_shp_bt_domain).to_crs(proj)
gds_shp_bt_basin = gpd.read_file(f_shp_bt_basin).to_crs(proj)
gds_shp_nrflk_domain = gpd.read_file(f_shp_nrflk_domain).to_crs(proj)
gds_shp_nrflk_basin = gpd.read_file(f_shp_nrflk_basin).to_crs(proj)
#%% plotting sample storm
ds_bigthomp_riv.rainrate.sel({"nstorms":1}).sum('time').plot()


#%% creating code to plot stuff
figwidth, figheight = (7.5, 6)

def convert_mm_per_hour_to_in_per_event(da, hourly = True):
    if hourly == True:
        hrs_in_event = len(da.time.values)
    else:
        tseries = da.time.values
        hrs_in_event = (max(tseries) - min(tseries)) / np.timedelta64(1, 'h')

    da = da * hrs_in_event * inches_per_mm # mm/hr * hrs in event * inches_per_mm
    return da

def plot_day_of_mrms_data(ds, title, fname, gds_domain, gds_basin):
    fig, ax = plt.subplots(figsize = [figwidth, figheight], subplot_kw=dict(projection=proj), dpi=300)

    da = ds.rainrate
    da = convert_mm_per_hour_to_in_per_event(da)
    da = da.mean("time")
    
    da = da.rename(dict(outlat="latitude", outlon = "longitude"))
    da['latitude'] = ds['latitude'].values
    da['longitude'] = ds['longitude'].values
    da['longitude'].attrs['units'] = ds['longitude'].attrs['units']
    da['latitude'].attrs['units'] = ds['latitude'].attrs['units']

    da.plot.pcolormesh(x='longitude', y='latitude', ax=ax, 
                        cbar_kwargs=dict(pad=0.02, shrink=0.5, label="Event Total Rainfall (inches)"),
                         vmin=0, cmap='gist_rainbow')
    

    gds_basin.plot(ax=ax, color="grey", edgecolor="none", alpha = 0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    gls = ax.gridlines(draw_labels=True)
    gls.top_labels=False   # suppress top labels
    gls.right_labels=False # suppress right labels
    gds_domain.plot(ax=ax, color='none', edgecolor="black", zorder=1)
    ax.set_title(title)
    plt.savefig(fname)

#%% testing plot functions
# ds = ds_bigthomp_riv.sel({"nstorms":1})
# da = ds_bigthomp_riv.rainrate.sel({"nstorms":1})
# tseries = da.time.values

# title = "Test plot of a storm event from the big thompson storm catalog"
# fname = f_out_plts.format("_test_plot_bt_catalog.png")

plt_subfldr = 'bt_rainyday/'

for i in tqdm(ds_bigthomp_riv.nstorms.values):
    ds = ds_bigthomp_riv.sel({"nstorms":i})
    fname = f_out_plts.format(plt_subfldr + "plot_bt_catalog_storm_{}.png".format(i))
    title = "Big Thompson Storm Catalog: Storm {}".format(i)
    plot_day_of_mrms_data(ds, title, fname, gds_shp_bt_domain, gds_shp_bt_basin)

#%% Norflk stage IV rivanna
plt_subfldr = 'nrflk_stageiv_riv/'

for i in ds_nrflk_stageIV_riv.nstorms.values:
    ds = ds_nrflk_stageIV_riv.sel({"nstorms":i})
    fname = f_out_plts.format(plt_subfldr + "plot_norfolk_catalog_storm_{}.png".format(i))
    title = "Norfolk Storm Catalog: Storm {}".format(i)
    plot_day_of_mrms_data(ds, title, fname, gds_shp_nrflk_domain, gds_shp_nrflk_basin)