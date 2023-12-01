"""
This script consolidates all of the water level time series
"""
#%% import libraries
from pathlib import Path
import xarray as xr
import pandas as pd
from glob import glob
import time
import shutil
from __utils import *
import os
# import dask
# dask.config.set(scheduler='single-threaded')


start_time = time.time()
#%% water level summaries
f_summary = dir_event_summary_csv_scratch + "_event_summary_year{}.csv".format('*') # must match format in script d4b
lst_f_summaries = glob(f_summary)
bm_time = time.time()
lst_df = []

for filename in lst_f_summaries:
    df = pd.read_csv(filename, index_col=None, header=0)
    lst_df.append(df)

df_summaries = pd.concat(lst_df, axis=0, ignore_index=True)

df_summaries = df_summaries.sort_values(["realization", "year", "storm_id"])

df_summaries.to_csv(f_sims_summary)
print("Total time elapsed: {}; time load and combine to single csv event summaries: {}".format(time.time() - start_time, time.time() - bm_time))

#%% water level csvs
# df_wlevel_combined.to_csv(dir_waterlevel_ncs_scratch + "waterlevels_yr{}.csv".format(yr)) # code from scripts d4b1

lst_f_wlevel_tseries_csvs = glob(dir_waterlevel_scratch + "waterlevels_yr{}.csv".format("*"))
bm_time = time.time()
lst_df = []

for filename in lst_f_wlevel_tseries_csvs:
    # continue
    df = pd.read_csv(filename) #.set_index(["realization", "year","storm_id", "datetime"])
    df["datetime"] = pd.to_datetime(df.datetime)
    lst_df.append(df)

df_wlevel_tseries = pd.concat(lst_df, axis = 0, ignore_index = True).sort_values(["realization", "year", "storm_id", "datetime"]).set_index(["realization", "year","storm_id", "datetime"])

df_wlevel_tseries.to_csv(f_sims_wlevel_tseries)
print("Total time elapsed: {}; time load and combine to single csv water level time series: {}".format(time.time() - start_time, time.time() - bm_time))

#%% attempting to export as netcdf now
ds_w = df_wlevel_tseries.to_xarray()

#%% export to a zarr and then export to a netcdf
bm_time = time.time()

ds_w_loaded = ds_w.load()
ds_w_loaded.to_netcdf(f_w_level_sims, engine = "h5netcdf", encoding= {"water_level":{"zlib":True}})

print("Total time elapsed: {}; time to export combined water level realizations to netcdf {}".format(time.time() - start_time, time.time() - bm_time))

#%% remove scratch files
shutil.rmtree(dir_waterlevel_scratch)
shutil.rmtree(dir_event_summary_csv_scratch)
os.remove(f_sims_wlevel_tseries)