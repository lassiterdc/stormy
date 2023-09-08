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
from datetime import datetime
# from tqdm import tqdm

from __utils import *

yr = int(sys.argv[1]) # a number between 1 and 1000

script_start_time = datetime.now()
#%% loading data
lst_f_ncs = return_rzs_for_yr(fldr_realizations, yr)
ds_rlztns = xr.open_mfdataset(lst_f_ncs, preprocess = define_dims)


gdf_subs = gpd.read_file(f_shp_swmm_subs)

# shift gridcell to center (the coordinates represent the upper left of each gridcell)
shift = grid_spacing /2

# shift half a gridcell east
lon_shifted = ds_rlztns.longitude + shift
# shift half a gridcell south
lat_shifted = ds_rlztns.latitude - shift

ds_rlztns = ds_rlztns.assign_coords({"longitude":lon_shifted, "latitude":lat_shifted})

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
idx_mrms = gdf_matching_subs_and_mrms.index_right.values


df_mrms_at_subs = df_mrms_coords.iloc[idx_mrms, :]

# unique gridcells
df_mrms_at_subs_unique = df_mrms_at_subs.drop_duplicates()


#%% create a swmm .date file for each of the events
num_files = len(ds_rlztns.realization.values) * len(ds_rlztns.storm_id.values) * len(df_mrms_at_subs_unique)

times_fwright_min = []
count = 0
for rz in ds_rlztns.realization.values:
    dir_yr = dir_time_series + "weather_realization{}/year{}/".format(rz, yr)
    # try:
    #     shutil.rmtree(dir_yr)
    # except:
    #     pass
    Path(dir_yr).mkdir(parents=True, exist_ok=True)
    for storm_id in ds_rlztns.storm_id.values:
        # figure out first and last timesteps with rain and create datetime index
        # create mean_rain_timeseries
        lst_rains = []
        lst_grid_ind = []
        for row in df_mrms_at_subs_unique.iterrows():
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            rain_inperhr = ds_rlztns_subset.rain.values / mm_per_inch
            # replace negative values with 0 if any are present
            rain_inperhr[rain_inperhr<0] = 0
            # append to list
            lst_rains.append(rain_inperhr)
            lst_grid_ind.append("grid_{}".format(mrms_index))
        # create series of mean rain
        df_allrain = pd.DataFrame(lst_rains)
        df_allrain = df_allrain.T
        df_allrain.columns = lst_grid_ind
        s_meanrain = df_allrain.mean(axis=1)
        # if there is no rain in the storm event, don't write a time series file
        if s_meanrain.sum() == 0:
            continue
        else: # find the indices of the first and last rainfall and create a datetime index
            non_zero_indices = s_meanrain[s_meanrain != 0].index
            first_tstep_with_rain = non_zero_indices[0]
            last_tstep_with_rain = non_zero_indices[-1]
            s_meanrain = s_meanrain.loc[first_tstep_with_rain:last_tstep_with_rain]
            dti = pd.date_range(start_date, periods = len(s_meanrain), freq = "{}min".format(int(ds_rlztns_subset.timestep_min)))
        # create a time series file for each grid that overlaps a subcatchment
        for row in df_mrms_at_subs_unique.iterrows():
            count += 1
            time_start_fwrite = datetime.now()
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            rain_inperhr = ds_rlztns_subset.rain.values / mm_per_inch
            # replace negative values with 0 if any are present
            rain_inperhr[rain_inperhr<0] = 0
            # slice rain array where rainfall is present in s_meanrain (need to add 1 when using integer indices)
            rain_inperhr_subset = rain_inperhr[first_tstep_with_rain:last_tstep_with_rain+1] 
            # print(rain_inperhr_subset)
            # create dataframe to write to .dat file
            df = pd.DataFrame(dict(date=dti.strftime('%m/%d/%Y'),
                                time = dti.time,
                                rain_inperhr = rain_inperhr_subset))
            f_out_swmm_rainfall = dir_yr + "rz{}_yr{}_strm{}_grid-ind{}.dat".format(rz, yr, storm_id, mrms_index)
            # write .dat file
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(";;sst_storm\n")
                file.write(";;Rainfall (in/hr)\n")
            df.to_csv(f_out_swmm_rainfall, sep = '\t', index = False, header = False, mode="a")
            time_end_fwrite = datetime.now()
            time_fwright_min = round((time_end_fwrite - time_start_fwrite).seconds / 60, 1)
            times_fwright_min.append(time_fwright_min)
            mean_times = round(np.mean(times_fwright_min), 1)
#%% generate a csv file for matching rain time series to subcatchments
if yr == 1:
    df_subnames_and_gridind = pd.DataFrame(dict(sub_names = gdf_subs.NAME,
                                                grid_index = idx_mrms))

    df_subnames_and_gridind.to_csv(f_key_subnames_gridind, index = False)

    print("Exported time series key to the file {}".format(f_key_subnames_gridind))

time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)

print("Wrote {} time series files for each subcatchment-overlapping-grids, storms, and realizations for year {}. Script runtime: {} (min)".format(num_files, yr, time_script_min))