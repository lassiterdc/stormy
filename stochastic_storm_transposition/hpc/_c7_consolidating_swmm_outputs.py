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

f_model_perf_summary, dir_swmm_sst_models, cubic_feet_per_cubic_meter = c7_consolidating_outputs()

script_start_time = datetime.now()
# sim_year = 1
#%% inputs from bash
sim_year = int(sys.argv[1]) # a number between 1 and 1000


#%% load and subset model performance dataframe
f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format(sim_year)

df_perf = pd.read_csv(f_model_perf_summary)
df_perf_success = df_perf[df_perf.run_completed==True]
# select for just the sim year
df_perf_success = df_perf_success[df_perf_success.year==sim_year]

storm_number = 0
lst_ds_node_fld = []
for f_inp in tqdm(df_perf_success.swmm_inp):
    storm_number += 1
    rz, yr, storm_id = parse_inp(f_inp)
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
        a_fld_reshaped = np.reshape(np.array(lst_tot_node_flding), (1,1,1,len(lst_tot_node_flding))) # rz, yr, storm, node_id
        # if there is a gap in the models that were run, fill with NA's to make concatenation easier in script c7b
        # (it is essential that coordinates are monotonically increasing).
        while storm_id > storm_number:
            # create dataset with na values with same shape as the flood data
            a_zeros = np.empty(a_fld_reshaped.shape)
            # create dataset with those na values
            ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'node_id'], a_zeros)),
                            coords = dict(realization = np.atleast_1d(rz),
                                            year = np.atleast_1d(yr),
                                            storm_id = np.atleast_1d(storm_number),
                                            node_id = lst_keys
                                            ))
            # append the dataset to the list of datasets
            lst_ds_node_fld.append(ds)
            # increment storm number by 1
            storm_number += 1
        # create dataset with the flood values
        ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'node_id'], a_fld_reshaped)),
                        coords = dict(realization = np.atleast_1d(rz),
                                        year = np.atleast_1d(yr),
                                        storm_id = np.atleast_1d(storm_id),
                                        node_id = lst_keys
                                        ))
        # append the dataset to the list of datasets
        lst_ds_node_fld.append(ds)

#%% concatenate the dataset
ds_all_node_fld = xr.combine_by_coords(lst_ds_node_fld)
ds_all_node_fld_loaded = ds_all_node_fld.load()
ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)

print("Total script runtime (min): {}".format(tot_elapsed_time_min))