#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import *


f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")

lst_f_modresults = glob(f_modelresults)

#%% export to single netcdf file
ds = xr.open_mfdataset(lst_f_modresults, engine="h5netcdf")
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

ds_loaded = ds.load() # this seems to speed up the writing of the netcdf file

ds_loaded.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

#%% also export