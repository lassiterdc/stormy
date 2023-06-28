#%% load libraries
import xarray as xr
import pandas as pd
# from pyswmm import Output
# from swmm.toolkit.shared_enum import NodeAttribute
from pathlib import Path
import numpy as np
# from datetime import datetime
import sys
# from tqdm import tqdm

from __utils import c8_bootstrapping

f_model_outputs_consolidated, dir_swmm_sst_models, f_bootstrapped_quant_estimates, sst_recurrence_intervals = c8_bootstrapping()

#%%
bs_id = int(sys.argv[1]) # a number between 1 and 1000

f_out_bs_results = f_bootstrapped_quant_estimates + "return_pds_btstrp_{}.nc".format(bs_id)
f_out_bs_results_raw = f_bootstrapped_quant_estimates + "raw_btstrp_{}.nc".format(bs_id)

#%%
ds_sst = xr.open_dataset(f_model_outputs_consolidated)


#%% preprocessing
ds_sst_compound = ds_sst.sel(freeboundary="False")

#%% compute quantiles
def return_period_to_quantile(ds, return_pds):
    storms_per_year = ds.storm_id.shape[0]
    total_years = ds.year.shape[0]
    total_events = storms_per_year * total_years
    quants = []
    for return_pd in return_pds:
        quants.append(return_pd * storms_per_year / total_events)
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
    ds_bs_smpl = ds_sst_compound.sel(d_idx)
    lst_ds.append(ds_bs_smpl)

ds_bs = xr.concat(lst_ds, dim = "resample_id")
ds_bs_qaunts = ds_bs.quantile(quants, dim = "resample_id")
# lst_bs_quants.append(ds_bs_qaunts)
#%% compute upper and lower bound 
# ds_bs_quants_all = xr.concat(lst_bs_quants, dim = "bootstrap_id")
ds_bs_qaunts = ds_bs_qaunts.assign_coords(dict(quantile = sst_recurrence_intervals))
ds_bs_qaunts = ds_bs_qaunts.rename((dict(quantile="flood_return_yrs")))


ds_bs_qaunts_loaded = ds_bs_qaunts.load()
Path(f_bootstrapped_quant_estimates).mkdir(parents=True, exist_ok=True)
ds_bs_qaunts_loaded.to_netcdf(f_out_bs_results, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

ds_bs_loaded = ds_bs.load()
ds_bs_loaded.to_netcdf(f_out_bs_results_raw, encoding= {"node_flooding_cubic_meters":{"zlib":True}})
