import xarray as xr
import pandas as pd
from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute
from pathlib import Path
import numpy as np
from datetime import datetime
import sys
from tqdm import tqdm 

from __utils import c7_consolidating_outputs, parse_inp

f_model_perf_summary, dir_swmm_sst_models, cubic_feet_per_cubic_meter, nperyear = c7_consolidating_outputs()

script_start_time = datetime.now()
#%% inputs from bash
sim_year = int(sys.argv[1]) # a number between 1 and 1000
#%% define function for creating dataset of all zeros
def create_all_nan_dataset(a_fld_reshaped, rz, yr, storm_number, freebndry, lst_keys):
    # create dataset with na values with same shape as the flood data
    a_zeros = np.empty(a_fld_reshaped.shape)
    a_zeros[:] = np.nan
    # create dataset with those na values
    ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'freeboundary', 'node_id'], a_zeros)),
                    coords = dict(realization = np.atleast_1d(rz),
                                    year = np.atleast_1d(yr),
                                    storm_id = np.atleast_1d(storm_number),
                                    freeboundary = np.atleast_1d(freebndry),
                                    node_id = lst_keys
                                    ))
    return ds

#%% load and subset model performance dataframe
f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format(sim_year)

df_perf = pd.read_csv(f_model_perf_summary)
df_perf_success = df_perf[df_perf.run_completed==True]
# select for just the sim year
df_perf_success = df_perf_success[df_perf_success.year==sim_year].reset_index()

count = -1
diffs = df_perf_success.storm_num.diff() # these are the differences in storm id; a value more than 1 means that a storm was skipped because it had 0 rain
max_storm_num = df_perf_success.storm_num.max()
lst_ds_node_fld = []
for f_inp in tqdm(df_perf_success.swmm_inp):
    count += 1
    diff = diffs.iloc[count]
    rz, yr, storm_id, freebndry = parse_inp(f_inp)
    f_swmm_out = f_inp.split('.inp')[0] + '.out'
    with Output(f_swmm_out) as out:
        lst_tot_node_flding = []
        lst_keys = []
        for key in out.nodes:
            d_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
            tstep_seconds = float(pd.Series(d_t_series.index).diff().mode().dt.seconds)
            # convert from cfs to cf per tstep then cubic meters per timestep
            d_t_series = d_t_series * tstep_seconds * cubic_feet_per_cubic_meter
            # sum all flooded volumes and append lists
            lst_tot_node_flding.append(d_t_series.sum())
            lst_keys.append(key)
        # create array of flooded values with the correct shape for placing in xarray dataset
        a_fld_reshaped = np.reshape(np.array(lst_tot_node_flding), (1,1,1,1,len(lst_tot_node_flding))) # rz, yr, storm, node_id, freeboundary
        # add datasets with na flooding as place holders to make concatenation easier in script c7b
        if diff > 1:
            last_storm_id = storm_id - diff
            for storm_number in np.arange(last_storm_id+1, last_storm_id + diff): # for each missing storm
                ds = create_all_nan_dataset(a_fld_reshaped, rz, yr, storm_number, freebndry, lst_keys)
                lst_ds_node_fld.append(ds)          
        # create dataset with the flood values 
        ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'freeboundary', 'node_id'], a_fld_reshaped)),
                        coords = dict(realization = np.atleast_1d(rz),
                                        year = np.atleast_1d(yr),
                                        storm_id = np.atleast_1d(storm_id),
                                        freeboundary = np.atleast_1d(freebndry),
                                        node_id = lst_keys
                                        ))
        lst_ds_node_fld.append(ds)
        # add datasets with na flooding as place holders to ensure the right number of storms per year
        if (storm_id==max_storm_num and max_storm_num < nperyear):
            for storm_number in np.arange(max_storm_num+1, nperyear+1):
                ds = create_all_nan_dataset(a_fld_reshaped, rz, yr, storm_number, freebndry, lst_keys)
                lst_ds_node_fld.append(ds)
        

#%% concatenate the dataset
ds_all_node_fld = xr.combine_by_coords(lst_ds_node_fld)
ds_all_node_fld_loaded = ds_all_node_fld.load()
ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)

print("Total script runtime (min): {}".format(tot_elapsed_time_min))