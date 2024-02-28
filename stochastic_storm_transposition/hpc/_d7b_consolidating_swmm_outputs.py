#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import *

script_start_time = datetime.now()

f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")
f_modelresults_rerun = dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format("*", "*")

lst_f_modresults = []
for f in glob(f_modelresults):
    if "failed_run" not in f:
        lst_f_modresults.append(f)
lst_f_modresults_rerun = glob(f_modelresults_rerun)
#%% export to single netcdf file
# def open_dataset_and_assign_attrs(lst_files):
#     ds = xr.open_mfdataset(lst_files, engine="h5netcdf")
#     attrs = dict(date_created = str(datetime.now()))
#     ds.attrs = attrs
#     return ds

# DCL WORK
# ds_first = xr.open_dataset(lst_f_modresults[0])
# for f in lst_f_modresults[1:]:
#     ds_next = xr.open_dataset(f)
#     # attempt to combine datasets
#     try:
#         test_ds = xr.combine_by_coords([ds_first, ds_next])
#     except Exception as e:
#         print("Combining datasets failed due to error:")
#         print(e)
#         print("####")
#         print("Problem dataset: {}".format(f))
#         print(ds_next)
#         print("###################################################")


    # xr.combine_nested([ds_first, ds_next])
def open_dataset_and_assign_attrs(lst_files):
    lst_ds = []
    for f in lst_files:
        lst_ds.append(xr.open_dataset(f, engine="h5netcdf"))
    ds_combined = xr.combine_by_coords(lst_ds)
    attrs = dict(date_created = str(datetime.now()))
    ds_combined.attrs = attrs
    return ds_combined
# END DCL WORK

ds_results = open_dataset_and_assign_attrs(lst_f_modresults)


# update ds_results where the model has been re-run
if len(lst_f_modresults_rerun) > 0:
    for f in lst_f_modresults_rerun:
        ds_results_rerun = open_dataset_and_assign_attrs(f)
        # ds_results_rerun_loaded = ds_results_rerun.load()
        dict_rerun_idx = dict()
        for coord in ds_results_rerun.coords:
            dict_rerun_idx[coord] = ds_results_rerun[coord].values
    # https://docs.xarray.dev/en/stable/user-guide/indexing.html#assigning-values-with-indexing
    ds_results.loc[dict_rerun_idx] = ds_results_rerun.sel(dict_rerun_idx)

ds_results_loaded = ds_results.load() # this seems to speed up the writing of the netcdf file
ds_results_loaded.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Created file: {}".format(f_model_outputs_consolidated))
print("Total script runtime (hr): {}".format(tot_elapsed_time_min/60))