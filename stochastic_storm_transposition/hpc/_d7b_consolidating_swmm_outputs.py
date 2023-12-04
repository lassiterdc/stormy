#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import *


f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")
f_modelresults_rerun = dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format("*", "*")

lst_f_modresults = []
for f in glob(f_modelresults):
    if "failed_run" not in f:
        lst_f_modresults.append(f)
lst_f_modresults_rerun = glob(f_modelresults_rerun)
#%% export to single netcdf file
def open_dataset_and_assign_attrs(lst_files):
    ds = xr.open_mfdataset(lst_files, engine="h5netcdf")
    attrs = dict(date_created = str(datetime.now()))
    ds.attrs = attrs
    return ds

ds_results = open_dataset_and_assign_attrs(lst_f_modresults)
ds_results_loaded = ds_results.load() # this seems to speed up the writing of the netcdf file

# update ds_results where the model has been re-run
for f in lst_f_modresults_rerun:
    ds_results_rerun = open_dataset_and_assign_attrs(f)
    ds_results_rerun_loaded = ds_results_rerun.load()
    dict_rerun_idx = dict()
    for coord in ds_results_rerun.coords:
        dict_rerun_idx[coord] = ds_results_rerun[coord].values

# https://docs.xarray.dev/en/stable/user-guide/indexing.html#assigning-values-with-indexing
ds_results_loaded.loc[dict_rerun_idx] = ds_results_rerun_loaded.sel(dict_rerun_idx)


ds_results_loaded.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}})