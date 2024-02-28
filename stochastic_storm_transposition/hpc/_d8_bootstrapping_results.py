#%% load libraries
import xarray as xr
import pandas as pd
# from pyswmm import Output
# from swmm.toolkit.shared_enum import NodeAttribute
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
# from tqdm import tqdm

from __utils import *

script_start_time = datetime.now()
#%%
bs_id = int(sys.argv[1]) # a number between 1 and 1000

f_out_bs_results = f_bootstrapped_quant_estimates + "return_pds_btstrp_{}.nc".format(bs_id)
f_out_bs_results_raw = f_bootstrapped_quant_estimates + "raw_btstrp_{}.nc".format(bs_id)

#%%
ds_sst_all_outputs = xr.open_dataset(f_model_outputs_consolidated, engine = "h5netcdf")


#%% preprocessing
ds_sst_compound = ds_sst_all_outputs.sel(sim_type="compound")

#%% compute quantiles
def return_period_to_quantile(ds, return_pds):
    storms_per_year = ds.storm_id.shape[0]
    total_years = ds.year.shape[0]
    total_events = storms_per_year * total_years
    quants = []
    for return_pd in return_pds:
        expected_num_events = total_years / return_pd
        quants.append(1 - expected_num_events / total_events)
    return quants

quants = return_period_to_quantile(ds_sst_compound, sst_recurrence_intervals)
#%% figuring out bootstrapping system
# lst_ds_bootstrapped = []
n_samples = len(ds_sst_compound.storm_id.values) * len(ds_sst_compound.realization.values) * len(ds_sst_compound.year.values)
lst_bs_quants = []
lst_ds = []
for j in np.arange(n_samples):
    strm = np.random.choice(ds_sst_compound.storm_id.values)
    rz = np.random.choice(ds_sst_compound.realization.values)
    yr = np.random.choice(ds_sst_compound.year.values)
    d_idx = dict(storm_id = strm, realization = rz, year = yr)
    da_bs_smpl = ds_sst_compound.node_flooding_cubic_meters.sel(d_idx)
    lst_ds.append(da_bs_smpl)

da_bs = xr.concat(lst_ds, dim = "resample_id")
da_bs_qaunts = da_bs.quantile(quants, dim = "resample_id")
# lst_bs_quants.append(ds_bs_qaunts)
#%% compute upper and lower bound 
# ds_bs_quants_all = xr.concat(lst_bs_quants, dim = "bootstrap_id")
da_bs_qaunts = da_bs_qaunts.assign_coords(dict(quantile = sst_recurrence_intervals))
da_bs_qaunts = da_bs_qaunts.rename((dict(quantile="flood_return_yrs")))

da_bs_qaunts_loaded = da_bs_qaunts.load()
Path(f_bootstrapped_quant_estimates).mkdir(parents=True, exist_ok=True)
da_bs_qaunts_loaded.to_netcdf(f_out_bs_results, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Exported file: {}".format(f_out_bs_results))
print("Total script runtime (hr): {}".format(tot_elapsed_time_min/60))

if export_raw_bs_samps == True:
    da_bs_loaded = da_bs.load()
    da_bs_loaded.to_netcdf(f_out_bs_results_raw, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")
    tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
    print("Exported file: {}".format(f_out_bs_results_raw))
