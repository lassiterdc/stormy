#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import *

script_start_time = datetime.now()

f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")
f_modelresults_failed_rerun = dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format("*", "*")
f_modelresults_high_error_rerun = dir_swmm_sst_models + "_model_outputs_year{}_high_error.nc".format("*")

df_perf = pd.read_csv(f_model_perf_summary)

lst_f_modresults = []
for f in glob(f_modelresults):
    if ("failed_run" not in f) and ("high_error" not in f):
        lst_f_modresults.append(f)
lst_f_modresults_failed_rerun = glob(f_modelresults_failed_rerun)
lst_f_modresults_high_error_rerun = glob(f_modelresults_high_error_rerun)

if len(lst_f_modresults_failed_rerun + lst_f_modresults_high_error_rerun):
    sys.exit("There are model re-runs that need to be incorporated in the results. Run d6_running_previous.sh script before proceeding.")

#%% define functions
def open_dataset_and_assign_attrs(lst_files):
    lst_ds = []
    for f in lst_files:
        lst_ds.append(xr.open_dataset(f, engine="h5netcdf"))
    ds_combined = xr.combine_by_coords(lst_ds)
    attrs = dict(date_created = str(datetime.now()))
    ds_combined.attrs = attrs
    return ds_combined

def update_results_dataset_with_reurns(lst_f_modresults_rerun, ds_results):
    """
    On the need for using data arrays to subset an xarray dataset at a set of specific coordinates
    rather than slicing along the dimensions:
    https://docs.xarray.dev/en/stable/user-guide/indexing.html#more-advanced-indexing
    """
    # do nothing if there are no files
    if len(lst_f_modresults_rerun) == 0:
        return
    # load dataset of reruns
    ds_results_rerun_error = open_dataset_and_assign_attrs(lst_f_modresults_rerun)
    # create series of re-run simulation end times
    s_endtimes_ds_rerun = ds_results_rerun_error.analysis_end_datetime.to_dataframe().dropna().analysis_end_datetime
    rerun_pd_indices = s_endtimes_ds_rerun.index
    # subset analysis endtimes from ds_results
    s_endtimes_ds_results = ds_results.analysis_end_datetime.to_dataframe().analysis_end_datetime[pd.IndexSlice[rerun_pd_indices]]
    # determine where rerun analysis endtime is more recent than those in ds_results
    ind_of_ds_results_to_update = s_endtimes_ds_rerun[s_endtimes_ds_rerun > s_endtimes_ds_results].index
    # only update ds_results where the rerun result has a more recent simulation endtime
    if len(ind_of_ds_results_to_update) > 0:
        df_rerun_indices = ind_of_ds_results_to_update.to_frame(index = False)
        da_rerun_indices = df_rerun_indices.to_xarray()
        # create a selection dictionary to subset the ds_results dataset to ONLY the sims that were re-run
        dict_rerun_idx = dict(realization = da_rerun_indices["realization"], storm_id = da_rerun_indices["storm_id"],
                        year = da_rerun_indices["year"], sim_type = da_rerun_indices["sim_type"])
        # update the ds_results dataframe with the re-runs
        ds_results.loc[dict_rerun_idx] = ds_results_rerun_error.sel(dict_rerun_idx)

#%% load results and update simulation results that were re-run more recently
bm_time = datetime.now()
ds_results = open_dataset_and_assign_attrs(lst_f_modresults)
tot_elapsed_time_min = round((datetime.now() - bm_time).seconds / 60, 1)
print("Minutes to open results dataset: {}".format(tot_elapsed_time_min))

# bm_time = datetime.now()
# update_results_dataset_with_reurns(lst_f_modresults_failed_rerun, ds_results)
# tot_elapsed_time_min = round((datetime.now() - bm_time).seconds / 60, 1)
# print("Minutes to update results with failed model re-runs: {}".format(tot_elapsed_time_min))

# bm_time = datetime.now()
# update_results_dataset_with_reurns(lst_f_modresults_high_error_rerun, ds_results)
# tot_elapsed_time_min = round((datetime.now() - bm_time).seconds / 60, 1)
# print("Minutes to update results with high-error model re-runs: {}".format(tot_elapsed_time_min))
#%% export results to a netcdf file
ds_results_loaded = ds_results.load() # this seems to speed up the writing of the netcdf file
ds_results_loaded.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Created file: {}".format(f_model_outputs_consolidated))
print("Total script runtime (hr): {}".format(tot_elapsed_time_min/60))