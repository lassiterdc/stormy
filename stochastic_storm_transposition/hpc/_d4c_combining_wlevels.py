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

start_time = time.time()
#%% water level summaries
f_summary = dir_time_series + "_event_summary_year{}.csv".format('*') # must match format in script d4b
lst_f_summaries = glob(f_summary)
bm_time = time.time()
lst_df = []

for filename in lst_f_summaries:
    df = pd.read_csv(filename, index_col=None, header=0)
    lst_df.append(df)

df_summaries = pd.concat(lst_df, axis=0, ignore_index=True)

df_summaries.to_csv(f_sims_summary)
print("Total time elapsed: {}; time load and combine to single csv event summaries: {}".format(time.time() - start_time, time.time() - bm_time))
#%% water levels
lst_f_netcdfs = glob(dir_waterlevel_ncs_scratch + "waterlevels_yr{}.nc".format("*"))

bm_time = time.time()
ds_w = xr.open_mfdataset(lst_f_netcdfs, combine = "nested")
print("Total time elapsed: {}; time to run open_mfdatset on water levels: {}".format(time.time() - start_time, time.time() - bm_time))

#%% export to a zarr and then export to a netcdf
bm_time = time.time()
fl_out_zar = dir_swmm_sst_scenarios_scratch+"waterlevels_combined.zarr"
ds_w = ds_w.chunk(chunks={"realization": 1, "year": 1, "storm_id": 1})
ds_w.to_zarr(fl_out_zar, mode="w")
print("Total time elapsed: {}; time to export combined water level realizations to zarr: {}".format(time.time() - start_time, time.time() - bm_time))

# Load zarr and export to netcdf file
bm_time = time.time()
ds_from_zarr = xr.open_zarr(store=fl_out_zar, chunks={'year':"5000MB"})
ds_from_zarr.to_netcdf(f_rain_realizations, encoding= {"water_level":{"zlib":True}})
# delete zarr file
shutil.rmtree(fl_out_zar)
print("Total time elapsed: {}; time to export combined water level realizations to netcdf and delete Zarr: {}".format(time.time() - start_time, time.time() - bm_time))

#%% delete scratch water levels
# shutil.rmtree(dir_waterlevel_ncs_scratch)


#%% export directly to a netcdf
# bm_time = time.time()
# ds_w_loaded = ds_w.load()
# print("Total time elapsed: {}; time to load dataset into memory: {}".format(time.time() - start_time, time.time() - bm_time))

# bm_time = time.time()
# ds_w_loaded.to_netcdf(f_w_level_sims, encoding= {"water_level":{"zlib":True}})
# print("Total time elapsed: {}; time to export combined water levels to netcdf: {}".format(time.time() - start_time, time.time() - bm_time))