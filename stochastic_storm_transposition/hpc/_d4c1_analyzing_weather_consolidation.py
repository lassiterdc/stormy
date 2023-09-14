#%% import libraries
from pathlib import Path
import xarray as xr
import pandas as pd
from glob import glob
import time

from __utils import *

start_time = time.time()
#%%
lst_f_ncs = glob(fldr_realizations+"*.nc")
bm_time = time.time()
ds_rlztns = xr.open_mfdataset(lst_f_ncs, preprocess = define_dims)
print("Total time elapsed: {}; time to run open_mfdataset on realizations: {}".format(time.time() - start_time, time.time() - bm_time))
#%% write all rainfall realizations to a new netcdf file
ds_rlztns.attrs["rain_units"] = "mm per hour"
bm_time = time.time()
ds_rlztns_loaded = ds_rlztns.load()
print("Total time elapsed: {}; time to load dataset into memory: {}".format(time.time() - start_time, time.time() - bm_time))
# create path if non-existant
Path(f_rain_realizations).parent.mkdir(parents=True, exist_ok=True)
bm_time = time.time()
ds_rlztns_loaded.to_netcdf(f_rain_realizations, encoding= {"rain":{"zlib":True}})
print("Total time elapsed: {}; time to export combined realizations to netcdf: {}".format(time.time() - start_time, time.time() - bm_time))
#%% water levels
f_summary = dir_time_series + "_event_summary_year{}.csv".format('*') # must match format in script d4b
lst_f_summaries = glob(f_summary)

lst_f_netcdfs = glob(dir_waterlevel_ncs_scratch + "*.nc")

bm_time = time.time()
ds_w = xr.open_mfdataset(lst_f_netcdfs, combine = "nested")
print("Total time elapsed: {}; time to run open_mfdatset on water levels: {}".format(time.time() - start_time, time.time() - bm_time))

bm_time = time.time()
ds_w_loaded = ds_w.load()
print("Total time elapsed: {}; time to load dataset into memory: {}".format(time.time() - start_time, time.time() - bm_time))

bm_time = time.time()
ds_w_loaded.to_netcdf(f_w_level_sims, encoding= {"water_level":{"zlib":True}})
print("Total time elapsed: {}; time to export combined water levels to netcdf: {}".format(time.time() - start_time, time.time() - bm_time))
#%% water level summaries
bm_time = time.time()
lst_df = []

for filename in lst_f_summaries:
    df = pd.read_csv(filename, index_col=None, header=0)
    lst_df.append(df)

df_summaries = pd.concat(lst_df, axis=0, ignore_index=True)

df_summaries.to_csv(f_sims_summary)
print("Total time elapsed: {}; time load and combine to single csv event summaries: {}".format(time.time() - start_time, time.time() - bm_time))