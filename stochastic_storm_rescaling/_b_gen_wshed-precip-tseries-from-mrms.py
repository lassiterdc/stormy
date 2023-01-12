#%% load libraries
from pathlib import Path
import os
from _inputs import def_inputs_for_b
import xarray as xr
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import pandas as pd

f_in_b_nc, f_shp_swmm_subs, f_out_b_csv_rainfall, f_out_b_csv_subs_w_mrms_grid, f_out_swmm_rainfall, mm_per_inch = def_inputs_for_b()

#%% load data
ds_rain = xr.open_dataset(f_in_b_nc)
gdf_subs = gpd.read_file(f_shp_swmm_subs)

# shift gridcell to center
shift = ds_rain.grid_spacing /2

# shift half a gridcell east
lon_shifted = ds_rain.longitude + shift
# shift half a gridcell south
lat_shifted = ds_rain.latitude - shift

ds_rain = ds_rain.assign_coords({"longitude":lon_shifted, "latitude":lat_shifted})

#%% associate each sub with the closest grid coord
gdf_sub_centroid = gpd.GeoDataFrame(geometry=gdf_subs.centroid)

x,y = np.meshgrid(ds_rain.longitude.values, ds_rain.latitude.values, indexing="ij")
grid_length = x.shape[0] * x.shape[1]
x = x.reshape(grid_length)
y = y.reshape(grid_length)

df_mrms_coords = pd.DataFrame({"x_lon":x, "y_lat":y})

# create geopandas dataframe for mrms grid
gdf_mrms = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=df_mrms_coords.x_lon, y=df_mrms_coords.y_lat), crs="EPSG:4326")


gdf_mrms_state_plane = gdf_mrms.to_crs("EPSG:2284")

#%% check
# base = gdf_sub_centroid.plot()
# gdf_mrms_state_plane.plot(ax=base)

#%% join subcatchment centroids with the closest MRMS point
gdf_matching_subs_and_mrms = gpd.sjoin_nearest(gdf_sub_centroid, gdf_mrms_state_plane, how='left')

idx = gdf_matching_subs_and_mrms.index_right.values

df_mrms_at_subs = df_mrms_coords.iloc[idx, :]

# unique gridcells
df_mrms_at_subs_unique = df_mrms_at_subs.drop_duplicates()

#%%
# idx = dict(latitude = df_mrms_at_subs_unique.y_lat.values, longitude = df_mrms_at_subs_unique.x_lon.values)
# ds_rain_subset = ds_rain.sel(idx)
# ds_rain_subset_loaded = ds_rain_subset.load()
cols = []
tseries = []
data = dict(time=ds_rain.time.values)
for row in df_mrms_at_subs_unique.iterrows():
    id, coords = row
    cols.append(id)
    idx = dict(latitude = coords.y_lat, longitude = coords.x_lon)
    ds_rain_subset = ds_rain.sel(idx)
    tseries.append(ds_rain_subset.rainrate.values)
    
    data[id]=ds_rain_subset.rainrate.values

df_rain_tseries = pd.DataFrame(data).set_index('time')
df_rain_tseries['mrms_mean'] = df_rain_tseries.mean(axis=1, skipna=True)

#%% append subcatchment geodataframe with associated rain time series
raingage_id = df_mrms_at_subs.index.values
sub_name = gdf_subs.NAME.values

df_subs_raingages = pd.DataFrame({"subcatchment_id":sub_name, "raingage_id":raingage_id})

#%% export csv files
df_rain_tseries.to_csv(f_out_b_csv_rainfall)
df_subs_raingages.to_csv(f_out_b_csv_subs_w_mrms_grid)

print("finished creating .csv files")

#%% creating .dat files for swmmm
"""
From SWMM Manual Version 5.1:
"a standard user-prepared format where each line of the file contains
the station ID, year, month, day, hour, minute, and non-zero precipitation
reading, all separated by one or more spaces."
Also
"""

# resample to a 5-minute interval
gage_ids = df_rain_tseries.columns
df = df_rain_tseries.resample("1min").ffill().resample("5min").mean()
colnames = df.columns.values
df = df.reset_index()
df['date'] = df.time.dt.strftime('%m/%d/%Y')
df['time'] = df.time.dt.time

# station ID has a ';' which is the comment symbol in SWMM
df_long = pd.melt(df, id_vars = ["date", "time"], var_name="station_id", value_name="precip")

# df_long["year"] = df_long.time.dt.year
# df_long["month"] = df_long.time.dt.month
# df_long["day"] = df_long.time.dt.day
# df_long["hour"] = df_long.time.dt.hour
# df_long["minute"] = df_long.time.dt.minute
# df_long["time"] = df_long.time.dt.time
df_long["precip_in"] = df_long.precip / mm_per_inch
df_long = df_long[["station_id", "date", "time", "precip_in"]]

# remove NA values and non-zero values
df_long = df_long[df_long['precip_in'] > 0]
#%% export to file
for g_id in gage_ids:
    # initialize file with proper header
    with open(f_out_swmm_rainfall.format(g_id), "w+") as file:
        file.write(";;MRMS Precipitation Data\n")
        file.write(";;Rainfall (in/hr)\n")
    df_long_subset = df_long[df_long['station_id'] == g_id]
    df_long_subset = df_long_subset.drop(["station_id"], axis=1)
    df_long_subset.to_csv(f_out_swmm_rainfall.format(g_id), sep = '\t', index = False, header = False, mode="a")
    

