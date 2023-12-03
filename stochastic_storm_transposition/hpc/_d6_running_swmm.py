#%% loading libraries and importing directories
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from swmm.toolkit.shared_enum import NodeAttribute
from pyswmm import Simulation, Output
from datetime import datetime
import os
from __utils import *

sim_year = int(sys.argv[1])
# from __utils import c6_running_swmm, parse_inp

# f_swmm_scenarios_catalog, dir_swmm_sst_models, max_runtime_min = c6_running_swmm()

 # a number between 1 and 1000

f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}.csv".format(sim_year)

script_start_time = datetime.now()

#%% define functions
def create_all_nan_dataset(a_fld_reshaped, rz, yr, storm_id, freebndry, norain, lst_keys):
    # create dataset with na values with same shape as the flood data
    a_zeros = np.empty(a_fld_reshaped.shape)
    a_zeros[:] = np.nan
    # create dataset with those na values
    ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'freeboundary', 'norain', 'node_id'], a_zeros)),
                    coords = dict(realization = np.atleast_1d(rz),
                                    year = np.atleast_1d(yr),
                                    storm_id = np.atleast_1d(storm_id),
                                    freeboundary = np.atleast_1d(freebndry),
                                    norain = np.atleast_1d(norain),
                                    node_id = lst_keys
                                    ))
    return ds

#%% loading data
df_strms = pd.read_csv(f_swmm_scenarios_catalog.format(sim_year))

# DCL WORK - SUBSET TO USE ONLY 1 REALIZATION
df_strms = df_strms[df_strms["realization"]==1]
df_strms = df_strms[df_strms.storm_id.isin([1])]
# END DCL WORK

# if "storm_num" in df_strms.columns: # this should become irrelevant, this was just so I didn't have to re-run previous script with desired column names
#     df_strms = df_strms.rename(columns=dict(storm_num = "storm_id"))

df_strms = df_strms.sort_values(["realization", "year", "storm_id"])

df_strms.drop(columns = "simulation_index", inplace = True)

df_strms.reset_index(drop=True, inplace=True)

# s_tot_rz = int(df_strms.realization.max())
# s_strms_per_year = int(df_strms.storm_id.max())
s_tot_sims = len(df_strms)

#%% run simulations
# DCL WORK - incorporating processing of outputs into the script
lst_ds_node_fld = []
lst_f_outputs_converted_to_netcdf = [] # for removing ones that are processed
lst_outputs_converted_to_netcdf = [] # to track success
f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format(sim_year)
# END DCL WORK
runtimes = []
export_dataset_times_min = []
# export_dataset_times = []
successes = []
problems = []
count = -1
for idx, row in df_strms.iterrows():
    problem = "None"
    f_inp = row["swmm_inp"]
    rz = int(row["realization"])
    yr = int(row["year"])
    storm_id = int(row["storm_id"])
    count += 1
    print("Running simulation for realization {} year {} storm {}. {} out of {} simulations complete.".format(rz, yr, storm_id, count, s_tot_sims))
    success = True
    output_converted_to_netcdf = False
    loop_start_time = sim_time = datetime.now()
    sim_runtime_min = np.nan
    # break
    try:
        with Simulation(f_inp) as sim:
            sim_start_time = datetime.now()
            for step in sim:
                sim_time = datetime.now()
                sim_runtime_min = round((sim_time - sim_start_time).seconds / 60, 1)
                if sim_runtime_min > max_runtime_min:
                    problem = "User-defined maximum simulation time limit of {} minutes reached, so simulation was halted.".format(max_runtime_min)
                    print(problem)
                    success = False
                    break
                pass
    except Exception as e:
        print("Simulation failed due to error: {}".format(e))
        problem = e
        success = False
    problems.append(problem)
    successes.append(success)
    # benchmarking write netcdf
    start_create_dataset = datetime.now()
    create_dataset_time_min = np.nan
    # MONTIROING
    if success == True:
        # print("Simulation runtime (min): {}, Mean simulation runtime (min): {}, Total elapsed time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(sim_runtime_min, mean_sim_time, tot_elapsed_time_hr, expected_tot_runtime_hr, expected_remaining_time_hr)) 
        #%% dcl work  - incorporating processing of outputs into the script
        print("Exporting node flooding as netcdfs....")
        rz, yr, storm_id, freebndry, norain = parse_inp(f_inp)
        f_swmm_out = f_inp.split('.inp')[0] + '.out'
        with Output(f_swmm_out) as out:
            lst_tot_node_flding = []
            lst_keys = []
            for key in out.nodes:
                d_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
                tstep_seconds = float(pd.Series(d_t_series.index).diff().mode().dt.seconds)
                # convert from cfs to cf per tstep then cubic meters per timestep
                d_t_series = d_t_series * tstep_seconds * cubic_meters_per_cubic_foot
                # sum all flooded volumes and append lists
                lst_tot_node_flding.append(d_t_series.sum())
                lst_keys.append(key)
        # create array of flooded values with the correct shape for placing in xarray dataset
        a_fld_reshaped = np.reshape(np.array(lst_tot_node_flding), (1,1,1,1,1,len(lst_tot_node_flding))) # rz, yr, storm, node_id, freeboundary, norain
        # create dataset with the flood values 
        ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'freeboundary', 'norain', 'node_id'], a_fld_reshaped)),
                        coords = dict(realization = np.atleast_1d(rz),
                                        year = np.atleast_1d(yr),
                                        storm_id = np.atleast_1d(storm_id),
                                        freeboundary = np.atleast_1d(freebndry),
                                        norain = np.atleast_1d(norain),
                                        node_id = lst_keys
                                        ))
        lst_ds_node_fld.append(ds)
        lst_f_outputs_converted_to_netcdf.append(f_swmm_out)
        output_converted_to_netcdf = True
        print("created xarray dataset with total flooding for each node")
        end_create_dataset = datetime.now()
        create_dataset_time_min = round((end_create_dataset - start_create_dataset).seconds / 60, 1)
    # benchmarking export netcdf
    export_dataset_times_min.append(create_dataset_time_min)
    mean_export_ds_time_min = round(np.nanmean(export_dataset_times_min), 1)
    # benchmarking simulations
    runtimes.append(sim_runtime_min)
    mean_sim_time_min = round(np.nanmean(runtimes), 1)
    # benchmarking entire loop and script
    ## estimating time remaining assuming successes
    estimated_loop_time = (mean_sim_time_min+mean_export_ds_time_min)
    expected_tot_runtime_hr = round(estimated_loop_time*s_tot_sims/60, 1)

    tot_elapsed_time_hr = round((datetime.now() - script_start_time).seconds / 60 / 60, 1)
    # tot_loop_time_hr = round((datetime.now() - loop_start_time).seconds / 60 / 60, 1)
    
    expected_remaining_time_hr = round((expected_tot_runtime_hr - tot_elapsed_time_hr), 1)
    if success == True:
        print("Sim runtime (min): {}, Mean sim runtime (min): {}, Time to create dataset (min): {}, Total script time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(sim_runtime_min, mean_sim_time_min,
                                                                                                                                                                                                 create_dataset_time_min,
                                                                                                                                                                                      tot_elapsed_time_hr, expected_tot_runtime_hr,
                                                                                                                                                                                        expected_remaining_time_hr)) 
        #%% end dcl work
    else: 
        print("Simulation failed after {} minutes.".format(sim_runtime_min))
    lst_outputs_converted_to_netcdf.append(output_converted_to_netcdf) # document success in processing outputs

#%% export model runtimes to a file
df_strms["run_completed"] = successes
df_strms["problem"] = problems
df_strms["runtime_min"] = runtimes
df_strms["export_dataset_min"] = export_dataset_times_min
df_strms["lst_outputs_converted_to_netcdf"] = lst_outputs_converted_to_netcdf
df_strms.to_csv(f_out_runtimes, index=False)
print('Exported ' + f_out_runtimes)
#%% dcl work - incorporating processing of outputs into the script
ds_all_node_fld = xr.combine_by_coords(lst_ds_node_fld)

ds_all_node_fld_loaded = ds_all_node_fld.load()
ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)

print("exported " + f_out_modelresults)

# remove processed outputs
print("Removing output files.....")
for f_processed_output in lst_f_outputs_converted_to_netcdf:
    os.remove(f_processed_output)
    print("removed file {}".format(f_processed_output))


#%% end dcl work
print("Total script runtime (min): {}".format(tot_elapsed_time_min))


