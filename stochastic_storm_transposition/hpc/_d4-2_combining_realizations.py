"""
This script consolidates all of the rainfall time series generated by RainyDay
"""
#%% import libraries
from pathlib import Path
import xarray as xr
import pandas as pd
from glob import glob
import time
import shutil
import os
from __utils import *

start_time = time.time()
#%%
f_pattern = dir_rain_weather_scratch_ncs + "sst_yr_{}.nc".format("*")
lst_f_ncs = glob(f_pattern)
lst_f_ncs.sort()
bm_time = time.time()
ds_rlztns = xr.open_mfdataset(lst_f_ncs, engine='h5netcdf')
print("Total time elapsed: {}; time to run open_mfdataset on realizations: {}".format(time.time() - start_time, time.time() - bm_time))

#%% writing to zarr
bm_time = time.time()
fl_out_zar = dir_rain_weather_scratch+"weather_combined.zarr"
ds_rlztns = ds_rlztns.chunk(chunks={"realization": 1, "year": 1, "storm_id": 1, "time": 864, "latitude": 2, "longitude": 3})
ds_rlztns.to_zarr(fl_out_zar, mode="w")
print("Total time elapsed: {}; time to export combined rainfall realizations to zarr: {}".format(time.time() - start_time, time.time() - bm_time))

# Load zarr and export to netcdf file
bm_time = time.time()
ds_from_zarr = xr.open_zarr(store=fl_out_zar, chunks={'year':"5000MB"})
ds_from_zarr.to_netcdf(f_rain_realizations, encoding= {"rain":{"zlib":True}})
# delete zarr file
shutil.rmtree(fl_out_zar)
# delete temporary netcdf files
for f in lst_f_ncs:
    print(f)
    os.remove(f)

print("Total time elapsed: {}; time to export combined rainfall realizations to netcdf and delete Zarr: {}".format(time.time() - start_time, time.time() - bm_time))
# %%
