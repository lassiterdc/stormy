#%% load libraries
from pathlib import Path
import os
from _inputs import def_inputs_for_b
import xarray as xr
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import pandas as pd

f_in_b_nc, f_shp_swmm_subs, f_out_b_csv_rainfall, f_out_b_csv_subs_w_mrms_grid = def_inputs_for_b()

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

#%% append subcatchment geodataframe with associated rain time series
raingage_id = df_mrms_at_subs.index.values
sub_name = gdf_subs.NAME.values

df_subs_raingages = pd.DataFrame({"subcatchment_id":sub_name, "raingage_id":raingage_id})

#%% export csv files
df_rain_tseries.to_csv(f_out_b_csv_rainfall)
df_subs_raingages.to_csv(f_out_b_csv_subs_w_mrms_grid)

print("finished creating .csv files")

# %%
