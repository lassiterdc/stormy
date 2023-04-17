#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob

from __utils import c7b_consolidating_outputs

f_model_outputs_consolidated, dir_swmm_sst_models = c7b_consolidating_outputs()

#%%
f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")

lst_f_modresults = glob(f_modelresults)

ds = xr.open_mfdataset(lst_f_modresults)
#%% export to single netcdf file
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

ds.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding":{"zlib":True}})