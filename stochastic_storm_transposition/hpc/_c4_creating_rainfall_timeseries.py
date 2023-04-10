#%% import libraries and load files and directories
from pathlib import Path
import os
import xarray as xr
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import pandas as pd
import shutil
from glob import glob
import sys
from tqdm import tqdm

from __utils import c4_creating_rainfall_tseries

yr = int(sys.argv[1]) # a number between 1 and 1000

f_out_realizations, f_shp_swmm_subs, dir_time_series, mm_per_inch, grid_spacing, start_date, freq, f_key_subnames_gridind, dir_sst_realizations = c4_creating_rainfall_tseries()
#%% loading data
ds_rlztns = xr.open_dataset(f_out_realizations)
gdf_subs = gpd.read_file(f_shp_swmm_subs)

# shift gridcell to center (the coordinates represent the upper left of each gridcell)
shift = grid_spacing /2

# shift half a gridcell east
lon_shifted = ds_rlztns.longitude + shift
# shift half a gridcell south
lat_shifted = ds_rlztns.latitude - shift

ds_rlztns = ds_rlztns.assign_coords({"longitude":lon_shifted, "latitude":lat_shifted})

#%% loading storm realizations
# fs_rlz = glob(dir_sst_realizations+"*.nc")

# # WORK
# ds_tst = xr.open_dataset(fs_rlz[0])
# # END WORK

#%% associate each sub with the closest grid coord
gdf_sub_centroid = gpd.GeoDataFrame(geometry=gdf_subs.centroid)

x,y = np.meshgrid(ds_rlztns.longitude.values, ds_rlztns.latitude.values, indexing="ij")
grid_length = x.shape[0] * x.shape[1]
x = x.reshape(grid_length)
y = y.reshape(grid_length)

df_mrms_coords = pd.DataFrame({"x_lon":x, "y_lat":y})

# create geopandas dataframe for mrms grid
gdf_mrms = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=df_mrms_coords.x_lon, y=df_mrms_coords.y_lat), crs="EPSG:4326")


gdf_mrms_state_plane = gdf_mrms.to_crs("EPSG:2284")

#%% join subcatchment centroids with the closest MRMS point
gdf_matching_subs_and_mrms = gpd.sjoin_nearest(gdf_sub_centroid, gdf_mrms_state_plane, how='left')

idx = gdf_matching_subs_and_mrms.index_right.values

df_mrms_at_subs = df_mrms_coords.iloc[idx, :]

# unique gridcells
df_mrms_at_subs_unique = df_mrms_at_subs.drop_duplicates()

#%% create a swmm .date file for each of the events

# create the time series directory if it does not exist
# clear everything in the directory
# try:
#     shutil.rmtree(dir_time_series)
# except:
#     pass
Path(dir_time_series).mkdir(parents=True, exist_ok=True)

for rz in tqdm(ds_rlztns.realization_id.values):
    for storm_id in ds_rlztns.storm_id.values:
        for row in df_mrms_at_subs_unique.iterrows():
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization_id = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            dti = pd.date_range(start_date, periods = len(ds_rlztns_subset.timestep_index.values), freq = freq)
            rainrate_inperhr = ds_rlztns_subset.rainrate.values / mm_per_inch
            df = pd.DataFrame(dict(date=dti.strftime('%m/%d/%Y'),
                                time = dti.time,
                                rainrate_inperhr = rainrate_inperhr))
            f_out_swmm_rainfall = dir_time_series + "realization{}_year{}_storm-id{}_grid-ind{}.dat".format(rz, yr, storm_id, mrms_index)
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(";;sst_storm\n")
                file.write(";;Rainfall (in/hr)\n")
            df.to_csv(f_out_swmm_rainfall, sep = '\t', index = False, header = False, mode="a")

#%% generate a csv file for matching rain time series to subcatchments
if yr == 1:
    df_subnames_and_gridind = pd.DataFrame(dict(sub_names = gdf_subs.NAME,
                                                grid_index = gdf_matching_subs_and_mrms.index_right))

    df_subnames_and_gridind.to_csv(f_key_subnames_gridind, index = False)

    print("Created time series and exported time series key to the file {}".format(f_key_subnames_gridind))