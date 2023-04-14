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

from __utils import c4_creating_rainfall_tseries

yr = int(sys.argv[1]) # a number between 1 and 1000

nrealizations, f_out_realizations, f_shp_swmm_subs, dir_time_series, mm_per_inch, grid_spacing, start_date, freq, f_key_subnames_gridind, dir_sst_realizations = c4_creating_rainfall_tseries()

script_start_time = datetime.now()
#%% loading data
ds_rlztns = xr.open_dataset(f_out_realizations)

# if the number of realizations defined in __utils is less than in the combined catalog, us the smaller of the two
if nrealizations < len(ds_rlztns.realization_id.values):
    realization_ids = np.arange(1, nrealizations+1)
    print("Using just {} out of {} available realizations based on user inputs in __utils.py.".format(nrealizations, len(ds_rlztns.realization_id.values)))
else:
    realization_ids = ds_rlztns.realization_id.values

gdf_subs = gpd.read_file(f_shp_swmm_subs)

# shift gridcell to center (the coordinates represent the upper left of each gridcell)
shift = grid_spacing /2

# shift half a gridcell east
lon_shifted = ds_rlztns.longitude + shift
# shift half a gridcell south
lat_shifted = ds_rlztns.latitude - shift

ds_rlztns = ds_rlztns.assign_coords({"longitude":lon_shifted, "latitude":lat_shifted})

# BEGIN WORK
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
prnt_statement = "loaded subcatchment shapefile and realizations in {} minutes".format(time_script_min)
print(prnt_statement)
# sys.exit(prnt_statement)
# END WORK
#%% loading storm realizations
# fs_rlz = glob(dir_sst_realizations+"*.nc")

# # WORK
# ds_tst = xr.open_dataset(fs_rlz[0])
# # END WORK

#%% associate each sub with the closest grid coord
gdf_sub_centroid = gpd.GeoDataFrame(geometry=gdf_subs.centroid)
# BEGIN WORK
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
prnt_statement = "Converted sub gdf to centroids in {} minutes".format(time_script_min)
print(prnt_statement)
#sys.exit(prnt_statement)
# END WORK

x,y = np.meshgrid(ds_rlztns.longitude.values, ds_rlztns.latitude.values, indexing="ij")
grid_length = x.shape[0] * x.shape[1]
x = x.reshape(grid_length)
y = y.reshape(grid_length)

df_mrms_coords = pd.DataFrame({"x_lon":x, "y_lat":y})

# create geopandas dataframe for mrms grid
gdf_mrms = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=df_mrms_coords.x_lon, y=df_mrms_coords.y_lat), crs="EPSG:4326")

# BEGIN WORK
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
prnt_statement = "Created gdf from grid coordinates {} minutes".format(time_script_min)
print(prnt_statement)
# sys.exit(prnt_statement)
# END WORK

gdf_mrms_state_plane = gdf_mrms.to_crs("EPSG:2284")

#%% join subcatchment centroids with the closest MRMS point
# BEGIN WORK
# print("Joining geodataframes to storm cat indices...")
# END WORK

try:
    gdf_matching_subs_and_mrms = gpd.sjoin_nearest(gdf_sub_centroid, gdf_mrms_state_plane, how='left')
    idx_mrms = gdf_matching_subs_and_mrms.index_right.values
    # idx_subs = gdf_matching_subs_and_mrms.index.values
except:
    # BEGIN WORK
    time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
    prnt_statement = "Made it to first except statement in {} minutes".format(time_script_min)
    # print(prnt_statement)
    # sys.exit(prnt_statement)
    # END WORK
    try:
        # print(gdf_mrms_state_plane)
        # print(gdf_sub_centroid)
        indices = gdf_mrms_state_plane.sindex.nearest(gdf_sub_centroid.geometry)
        idx_mrms = indices[1,:]
        # idx_subs = indices[0,:]
    except:
        # BEGIN WORK
        time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
        prnt_statement = "Made it to second except statement in {} minutes".format(time_script_min)
        # print(prnt_statement)
        # sys.exit(prnt_statement)
        # END WORK
        lst_mrms_indices = []
        for pt in gdf_sub_centroid.geometry:
            distances = []
            for mrms_grid_center in gdf_mrms_state_plane.geometry:
                distances.append(pt.distance(mrms_grid_center))
            distances = pd.Series(distances)
            mrms_idx_min_dist = distances.idxmin()
            lst_mrms_indices.append(mrms_idx_min_dist)
        idx_mrms = np.array(lst_mrms_indices)
        # idx_subs = np.arange(len(gdf_sub_centroid.geometry))

# BEGIN WORK
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
prnt_statement = "Joined grid coords to subs in {} minutes".format(time_script_min)
# print(prnt_statement)
# sys.exit(prnt_statement)

# END WORK

df_mrms_at_subs = df_mrms_coords.iloc[idx_mrms, :]

# unique gridcells
df_mrms_at_subs_unique = df_mrms_at_subs.drop_duplicates()

# BEGIN WORK
# time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
# if time_script_min > 1:
#     sys.exit("SCRIPT TAKING UNEXPECTEDLY LONG 1")
# END WORK

# BEGIN WORK
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
prnt_statement = "Part 2 joined subs to grid coordinates in {} minutes".format(time_script_min)
# print(prnt_statement)
# sys.exit(prnt_statement)
# END WORK

#%% create a swmm .date file for each of the events

# create the time series directory if it does not exist
# clear everything in the directory
# try:
#     shutil.rmtree(dir_time_series)
# except:
#     pass

num_files = len(realization_ids) * len(ds_rlztns.storm_id.values) * len(df_mrms_at_subs_unique)

times_fwright_min = []

count = 0
for rz in realization_ids:
    dir_yr = dir_time_series + "weather_realization{}/year{}/".format(rz, yr)
    try:
        shutil.rmtree(dir_yr)
    except:
        pass
    Path(dir_yr).mkdir(parents=True, exist_ok=True)
    for storm_id in ds_rlztns.storm_id.values:
        for row in df_mrms_at_subs_unique.iterrows():
            count += 1
            time_start_fwrite = datetime.now()
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization_id = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            rainrate_inperhr = ds_rlztns_subset.rainrate.values / mm_per_inch
            # remove preceding and trailing zeros
            ### BE AWARE THAT THE THE START DATE CORRESOPNDS TO THE FIRST TIMESTEP OF PRECIPITATION
            df_rr = pd.DataFrame(dict(rainrate = rainrate_inperhr))
            non_zero_indices = df_rr[df_rr.rainrate != 0].index
            first_tstep_with_rain = non_zero_indices[0]
            last_tstep_with_rain = non_zero_indices[-1]
            rainrate_inperhr = rainrate_inperhr[first_tstep_with_rain:last_tstep_with_rain]
            dti = pd.date_range(start_date, periods = len(rainrate_inperhr), freq = freq)
            # create dataframe to write to .dat file
            df = pd.DataFrame(dict(date=dti.strftime('%m/%d/%Y'),
                                time = dti.time,
                                rainrate_inperhr = rainrate_inperhr))
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
            # BEGIN WORK
            # print("Wrote file {} out of {}. File write time (min): {}   Average write time (min): {}".format(count, num_files, time_fwright_min, mean_times))
            # if count == 5:
            #     sys.exit("STOPPED SCRIPT EARLY TO SEE SOME PRELIMINARY RESULTS.")
            # END WORK


#%% generate a csv file for matching rain time series to subcatchments
if yr == 1:
    df_subnames_and_gridind = pd.DataFrame(dict(sub_names = gdf_subs.NAME,
                                                grid_index = idx_mrms))

    df_subnames_and_gridind.to_csv(f_key_subnames_gridind, index = False)

    print("Exported time series key to the file {}".format(f_key_subnames_gridind))

time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)

print("Wrote {} time series files for each subcatchment-overlapping-grids, storms, and realizations for year {}. Script runtime: {} (min)".format(num_files, yr, time_script_min))