#%% import libraries
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

#%%
lst_f_ncs = glob(fldr_realizations+"*.nc")
ds_rlztns = xr.open_mfdataset(lst_f_ncs, preprocess = define_dims)

#%% write all rainfall realizations to a new netcdf file
ds_rlztns.attrs["rain_units"] = "mm per hour"
ds_rlztns_loaded = ds_rlztns.load()
# create path if non-existant
Path(f_rain_realizations).parent.mkdir(parents=True, exist_ok=True)
ds_rlztns_loaded.to_netcdf(f_rain_realizations, encoding= {"rain":{"zlib":True}})

#%% water levels
f_summary = dir_time_series + "_event_summary_year{}.csv".format('*') # must match format in script d4b
lst_f_summaries = glob(f_summary)

lst_f_netcdfs = glob(dir_waterlevel_ncs_scratch + "*.nc")

ds_w = xr.open_mfdataset(lst_f_netcdfs, combine = "nested")
ds_w_loaded = ds_w.load()
ds_w_loaded.to_netcdf(f_w_level_sims, encoding= {"water_level":{"zlib":True}})

#%% water level summaries
lst_df = []

for filename in lst_f_summaries:
    df = pd.read_csv(filename, index_col=None, header=0)
    lst_df.append(df)

df_summaries = pd.concat(lst_df, axis=0, ignore_index=True)

df_summaries.to_csv(f_sims_summary)