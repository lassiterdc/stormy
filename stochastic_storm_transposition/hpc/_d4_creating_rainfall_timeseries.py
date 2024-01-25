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
# dcl work
# seeing if there are missing values in any of the realizations
lst_na_count = []
for f in lst_f_ncs:
    ds = xr.load_dataset(f)
    lst_na_count.append(int(ds.rain.to_dataframe().isna().sum().values))

df_na_counts = pd.DataFrame(dict(file = lst_f_ncs, 
                                 na_count = lst_na_count))

files_with_nas = df_na_counts[df_na_counts["na_count"] > 0]
if len(files_with_nas) > 0:
    print("The following SST realizations have NA values. This may or may not be a problem.")
    for f in files_with_nas.file:
        print(f)
# end dcl work

ds_rlztns = xr.open_mfdataset(lst_f_ncs, preprocess = define_dims)
ds_rlztns.attrs["rain_units"] = "mm_per_hour"

# subset to the top 5 largest rain events per year
# compute mean rainfall in mm/hr for each event
ds_mean = ds_rlztns.mean(dim = ["latitude", "longitude", "time"], skipna = True) # ignore na values for identifying the largest storm

# convert from mm/hr to mm
event_duration_hr = (len(ds_rlztns.time)*sst_tstep_min)/60
ds_tot = ds_mean * event_duration_hr # mm / hr * hr # total rain depth

# subset the largest 5 storms per year
lst_ds = []
for year in ds_tot.year.values:
    ds_yr = ds_tot.sel(year = year)
    for rz in ds_yr.realization.values:
        ds_rz = ds_yr.sel(dict(realization = rz)) # total rain depth for each storm
        top_storm_indices = ds_rz.rain.to_dataframe()["rain"].nlargest(n=nstormsperyear).index.values # indices of the largest storms
        ds_rz_out = ds_rlztns.sel(dict(storm_id = top_storm_indices, year = year, realization = rz)) # subset the largest storms
        # reassign coordinates that were lost in the subsetting
        ds_rz_out = ds_rz_out.assign_coords(dict(realization=rz, year = year)) 
        ds_rz_out = ds_rz_out.expand_dims(dim=dict(realization=1, year = 1)) 
        # re-define storm_id so that 1 is the biggest storm and nstormsperyear is the smallest
        ds_rz_out['storm_id'] = np.arange(1, nstormsperyear+1)
        # append to list
        lst_ds.append(ds_rz_out)

ds_rlztns = xr.combine_by_coords(lst_ds)
ds_rlztns.attrs["gridcell_loc"] = "coordinates represent the center of each gridcell"

#%%
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


#%% create a swmm .dat file for each of the events
num_files = len(ds_rlztns.realization.values) * len(ds_rlztns.storm_id.values) * len(df_mrms_at_subs_unique)

times_fwright_min = []
count = 0
for rz in ds_rlztns.realization.values:
    # break
    dir_yr = dir_time_series + "weather_realization{}/year{}/".format(rz, yr)
    # try:
    #     shutil.rmtree(dir_yr)
    # except:
    #     pass
    Path(dir_yr).mkdir(parents=True, exist_ok=True)
    for storm_id in ds_rlztns.storm_id.values:
        # break
        # figure out first and last timesteps with rain and create datetime index
        # create mean_rain_timeseries
        lst_rains = []
        lst_grid_ind = []
        for row in df_mrms_at_subs_unique.iterrows():
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            # dcl work
            s_rain = ds_rlztns_subset.rain.to_dataframe().rain
            tot_missing_rain_vals = s_rain.isna().sum()
            if tot_missing_rain_vals > 0:
                # print("There are {} missing values for {}".format(tot_missing_rain_vals, idx))
                # print("Converting missing values to 0...")
                # replace missing values with zeros
                s_rain = s_rain.fillna(0)
            # end dcl work
            s_rain_inperhr = s_rain / mm_per_inch
            # replace negative values with 0 if any are present
            s_rain_inperhr[s_rain_inperhr<0] = 0
            # append to list
            lst_rains.append(s_rain_inperhr)
            lst_grid_ind.append("grid_{}".format(mrms_index))
        # create series of mean rain
        df_allrain = pd.DataFrame(lst_rains)
        df_allrain = df_allrain.T
        df_allrain.columns = lst_grid_ind
        # if there is no rain in the storm event, don't write a time series file
        s_meanrain = df_allrain.mean(axis=1)
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
            # break
            count += 1
            time_start_fwrite = datetime.now()
            mrms_index, coords = row
            # extract rainfall time series from the storm catalog
            idx = dict(realization = rz, year = yr, storm_id = storm_id, latitude = coords.y_lat, longitude = coords.x_lon)
            ds_rlztns_subset = ds_rlztns.sel(idx)
            # dcl work
            s_rain = ds_rlztns_subset.rain.to_dataframe().rain
            tot_missing_rain_vals = s_rain.isna().sum()
            if tot_missing_rain_vals > 0:
                print("There are {} missing values for {}".format(tot_missing_rain_vals, idx))
                print("Converting missing values to 0...")
                # replace missing values with zeros
                s_rain = s_rain.fillna(0)
            # end dcl work
            s_rain_inperhr = s_rain / mm_per_inch
            # replace negative values with 0 if any are present
            s_rain_inperhr[s_rain_inperhr<0] = 0
            # slice rain array where rainfall is present in s_meanrain (need to add 1 when using integer indices)
            s_rain_inperhr_subset = s_rain_inperhr[first_tstep_with_rain:last_tstep_with_rain+1]
            # print(rain_inperhr_subset)
            # create dataframe to write to .dat file
            df = pd.DataFrame(dict(date=dti.strftime('%m/%d/%Y'),
                                time = dti.time,
                                rain_inperhr = s_rain_inperhr_subset))
            f_out_swmm_rainfall = dir_yr + "rz{}_yr{}_strm{}_grid-ind{}.dat".format(rz, yr, storm_id, mrms_index)
            # write .dat file
            # break
            with open(f_out_swmm_rainfall, "w+") as file:
                file.write(";;sst_storm\n")
                file.write(";;Rainfall (in/hr)\n")
            # break
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
#%% write a yearly netcdf file
f_out_nc = dir_rain_weather_scratch_ncs + "sst_yr_{}.nc".format(yr)
Path(f_out_nc).parent.mkdir(parents=True, exist_ok=True)
ds_rlztns_loaded = ds_rlztns.load()
ds_rlztns_loaded.to_netcdf(f_out_nc, encoding= {"rain":{"zlib":True}})
time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Exported netcdf file with rainfall realizations for year {}. Script runtime: {} (min)".format(yr, time_script_min))