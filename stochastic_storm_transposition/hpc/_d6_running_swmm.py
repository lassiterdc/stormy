#%% loading libraries and importing directories
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from swmm.toolkit.shared_enum import NodeAttribute
from pyswmm import Simulation, Output
from datetime import datetime
import os
from glob import glob
from __utils import *

sim_year = int(sys.argv[1])
which_models = str(sys.argv[2]) # either all or failed or an integer
realizations_to_use = str(sys.argv[3])
print("Running models: {} for year {} (if an integer, simulating a specific storm number)".format(which_models, sim_year))
delete_swmm_outputs = int(sys.argv[4])
if delete_swmm_outputs == 1:
    delete_swmm_outputs = True
else:
    delete_swmm_outputs = False
print("Deleting SWMM .out files is set to {}".format(delete_swmm_outputs))
f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}.csv".format(sim_year)
f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format(sim_year)
storm_id_to_run = None
if which_models == 'failed': # NOTE THIS SHOULD ONLY BE RUN AFTER SCRIPT D6B HAS BEEN RUN OTHERWISE YOU MIGHT END UP RE-RUNNING SUCCESSFUL SIMULATIONS
    print("Re-running failed simulations.")
    print('NOTE THIS SHOULD ONLY BE RUN AFTER SCRIPT D6B HAS BEEN RUN OTHERWISE YOU MIGHT END UP RE-RUNNING SUCCESSFUL SIMULATIONS')
    df_perf = pd.read_csv(f_model_perf_summary)
    df_perf = df_perf[df_perf.run_completed == False]
    df_perf.reset_index(inplace = True)
    max_runtime_min = 180 # allowing 3 hours per simulation
    # only use needed tasks (NOTE THE TOTAL NUMBER OF TASKS MUST EQUAL OR EXCEED THE NUMBER OF FAILED RUNS)
    row_index = (sim_year-1) # subtract 1 since python is 0-indexed and tasks are 1-indexed
    if row_index > df_perf.index.max():
        sys.exit("Task number not needed for running simulation because they are all covered by other tasks.")
    # Subset the row with the failed model
    row_with_failed_run = df_perf.loc[row_index,:]
    # reset sim year 
    sim_year = int(row_with_failed_run.year)
    failed_inp_to_rerun = row_with_failed_run.swmm_inp
    failed_inp_problem = row_with_failed_run.problem
    f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format(sim_year, row_index)
    f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format(sim_year, row_index)
# clear all re-run outputs
elif which_models == "all":
    fs_re_runs_csvs = glob(dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format("*", "*"))
    fs_re_runs_netcdfs = glob(dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format("*", "*"))
    for f in fs_re_runs_csvs:
        os.remove(f)
    for f in fs_re_runs_netcdfs:
        os.remove(f)
else:
    try:
        storm_id_to_run = int(which_models)
    except Exception as e:
        print("Attempted to run a single storm, but which_models could not be converted to an integer. Check the arguments to the python script.")
        sys.exit(e)

try:
    realization_to_run = int(realizations_to_use)
    print("Running realization {}".format(realization_to_run))
except:
    if realizations_to_use == "all":
        realization_to_run = None
        print("Running all realizations")
    else:
        sys.exit("Missing or invalid argument for \"realization_to_run\"")
# from __utils import c6_running_swmm, parse_inp

# f_swmm_scenarios_catalog, dir_swmm_sst_models, max_runtime_min = c6_running_swmm()

 # a number between 1 and 1000

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
# df_strms = df_strms[df_strms["realization"]==1]
# df_strms = df_strms[df_strms.storm_id.isin([1])]
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
# lst_f_outputs_converted_to_netcdf = [] # for removing ones that are processed
lst_outputs_converted_to_dataset = [] # to track success
# END DCL WORK
runtimes = []
export_dataset_times_min = []
lst_flow_errors = []
lst_runoff_errors = []
lst_routing_timestep_used = []
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
    # if a certain storm is being run, skip all simulations but those for that storm
    if storm_id_to_run is not None:
        if storm_id != storm_id_to_run:
            continue
    # if certain realizations are being run, skip sims but those for that realization
    if realization_to_run is not None:
        if rz != realization_to_run:
            continue
    # if only running a single failed simulation, set the loop to continue until the correct simulation is reached
    if which_models == "failed":
        if f_inp != failed_inp_to_rerun:
            continue
        print("Re-running failed simulation {} which failed due to problem {}".format(f_inp, failed_inp_problem))
        s_tot_sims = 1
    count += 1
    print("Running simulation for realization {} year {} storm {}. {} out of {} simulations complete.".format(rz, yr, storm_id, count, s_tot_sims))
    success = True
    output_converted_to_dataset = False
    loop_start_time = sim_time = datetime.now()
    sim_runtime_min = np.nan
    for routing_tstep in lst_alternative_routing_tsteps:
        # modify inp file with routing timestep
        with open(f_inp, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
        for i, line in enumerate(lines):
            if "ROUTING_STEP" in line:
                line_of_interest = line
                # Modify the text in that line
                # write the first part of the line (to make sure I'm getting the spacing right)
                first_part = ""
                for substring in line_of_interest.split(' ')[0:-1]:
                    # spaces are collapsed when splitting by spaces so
                    # I am adding a space to each substring
                    first_part += substring + " "
                # write the full line to replace the original with
                newline = first_part + str(routing_tstep) + "\n"
                lines[i] = line.replace(line_of_interest, newline)
        with open(f_inp, 'w') as file:
            file.writelines(lines)
        # run simulation
        runoff_error = 9999
        flow_routing_error = 9999
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
                sim._model.swmm_end()
                runoff_error = sim.runoff_error
                flow_routing_error = sim.flow_routing_error
                lst_routing_timestep_used.append(routing_tstep)
        except Exception as e:
            print("Simulation failed due to error: {}".format(e))
            problem = e
            success = False
        # if the run was succesful and the flow routing and runoff routing errors are below the prespecified threshold,
        # there is no need to run the simulation again with a smaller timestep
        if success:
            if (abs(flow_routing_error) <= continuity_error_thresh) and (abs(runoff_error) <= continuity_error_thresh):
                break
            else:
                print("The simulation was run with a routing timestep of {}. Runoff and flow continuity errors were {} and {}. Re-running simulation.".format(
                    routing_tstep, runoff_error, flow_routing_error))
    problems.append(problem)
    successes.append(success)
    # record flow and runoff errors
    lst_flow_errors.append(flow_routing_error)
    lst_runoff_errors.append(runoff_error)
    # benchmarking write netcdf
    start_create_dataset = datetime.now()
    create_dataset_time_min = np.nan
    # if the run was succesful, process the results
    if success == True:
        # print("Simulation runtime (min): {}, Mean simulation runtime (min): {}, Total elapsed time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(sim_runtime_min, mean_sim_time, tot_elapsed_time_hr, expected_tot_runtime_hr, expected_remaining_time_hr)) 
        print("Exporting node flooding as netcdfs....")
        __, __, __, freebndry, norain = parse_inp(f_inp) # this function also returns rz, yr, storm_id which are not needed since they were determined earlier
        f_swmm_out = f_inp.split('.inp')[0] + '.out'
        with Output(f_swmm_out) as out:
            lst_tot_node_flding = []
            lst_keys = []
            for key in out.nodes:
                d_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
                tstep_seconds = pd.Series(d_t_series.index).diff().mode().dt.seconds.values[0]
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
        # lst_f_outputs_converted_to_netcdf.append(f_swmm_out)
        output_converted_to_dataset = True
        print("created xarray dataset with total flooding for each node")
        end_create_dataset = datetime.now()
        create_dataset_time_min = round((end_create_dataset - start_create_dataset).seconds / 60, 1)
        # delete output file after it's been processed
        if delete_swmm_outputs:
            os.remove(f_swmm_out)
            print("Deleted file {}".format(f_swmm_out))
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
                                                                                                                                                                                      tot_elapsed_time_hr, expected_tot_runtime_hr, expected_remaining_time_hr)) 
    else: 
        print("Simulation failed after {} minutes.".format(sim_runtime_min))
    lst_outputs_converted_to_dataset.append(output_converted_to_dataset) # document success in processing outputs
    # if only running single simulation, stop the script here
    if which_models == "failed":
        break

#%% export model runtimes to a file
if which_models == "failed":
    # export netcdf
    ds_all_node_fld = ds # single output only
    ds_all_node_fld_loaded = ds_all_node_fld.load()
    ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}})
    tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
    print("exported " + f_out_modelresults)
    # export performance info
    df_out = row_with_failed_run.to_frame().T
    df_out["run_completed"] = success
    df_out["routing_timestep"] = lst_routing_timestep_used
    df_out["flow_continuity_error"] = lst_flow_errors
    df_out["runoff_continuity_error"] = lst_runoff_errors
    df_out["problem"] = problem
    df_out["runtime_min"] = sim_runtime_min
    df_out["export_dataset_min"] = create_dataset_time_min
    df_out["lst_outputs_converted_to_netcdf"] = output_converted_to_dataset
    df_out.to_csv(f_out_runtimes, index=False)
    print('Exported ' + f_out_runtimes)
else:
    # export netcdf
    ds_all_node_fld = xr.combine_by_coords(lst_ds_node_fld)
    ds_all_node_fld_loaded = ds_all_node_fld.load()
    ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")
    tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
    print("exported " + f_out_modelresults)
    # export performance info
    df_strms["run_completed"] = successes
    df_strms["routing_timestep"] = lst_routing_timestep_used
    df_strms["flow_continuity_error"] = lst_flow_errors
    df_strms["runoff_continuity_error"] = lst_runoff_errors
    df_strms["problem"] = problems
    df_strms["runtime_min"] = runtimes
    df_strms["export_dataset_min"] = export_dataset_times_min
    df_strms["lst_outputs_converted_to_netcdf"] = lst_outputs_converted_to_dataset
    df_strms.to_csv(f_out_runtimes, index=False)
    print('Exported ' + f_out_runtimes)

print("Total script runtime (min): {}".format(tot_elapsed_time_min))