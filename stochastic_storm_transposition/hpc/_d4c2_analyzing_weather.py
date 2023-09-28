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
ds_rlztns_loaded.to_netcdf(f_rain_realizations, encoding= {"rain":{"zlib":True}})
