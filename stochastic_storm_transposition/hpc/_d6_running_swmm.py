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
import shutil

use_hotstart_override = False
hotstart_override_val = False
#%% work
# sim_year = 251
# which_models = "high_error"
# realizations_to_use = 1
# delete_swmm_outputs = False
# realization_to_run = None
#%% end work

sim_year = int(sys.argv[1]) # this is used as the row index of the failed model to run if which_models = "failed" (so there's only 1 simulation per node)
print("Slurm Job Number: {}".format(sim_year))
which_models = str(sys.argv[2]) # either all or failed or an integer
realizations_to_use = str(sys.argv[3])
delete_swmm_outputs = int(sys.argv[4])
if delete_swmm_outputs == 1:
    delete_swmm_outputs = True
if which_models == "all":
    print("Running {} storms for year {}".format(which_models, sim_year))
    remove_previous_runs = True # if rerunning all, remove old simulations and results
elif which_models == "failed":
    print("Running a failed simulation for year {}".format(sim_year))
    remove_previous_runs = False
elif which_models == "previous":
    remove_previous_runs = False
    delete_swmm_outputs = False # no outputs to remove
    print("Generating output netcdfs from previously run SWMM models.")
elif which_models == "high_error":
    remove_previous_runs = False
    delete_swmm_outputs = False
if delete_swmm_outputs:
    print("Deleting SWMM .out files is set to {}".format(delete_swmm_outputs))
else:
    delete_swmm_outputs = False
f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}.csv".format(sim_year)
f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format(sim_year)
storm_id_to_run = None
if which_models == "failed": # NOTE THIS SHOULD ONLY BE RUN AFTER SCRIPT D6B HAS BEEN RUN OTHERWISE YOU MIGHT END UP RE-RUNNING SUCCESSFUL SIMULATIONS
    print("Re-running failed simulations.")
    print('NOTE THIS SHOULD ONLY BE RUN AFTER SCRIPT D6B HAS BEEN RUN OTHERWISE YOU MIGHT END UP RE-RUNNING SUCCESSFUL SIMULATIONS')
    df_perf = pd.read_csv(f_model_perf_summary)
    df_perf = df_perf[df_perf.run_completed == False]
    df_perf.reset_index(inplace = True)
    max_runtime_min = time_permitted_for_reruns_min # allowing 3 hours per simulation
    # only use needed tasks (NOTE THE TOTAL NUMBER OF TASKS MUST EQUAL OR EXCEED THE NUMBER OF FAILED RUNS)
    row_index = (sim_year-1) # subtract 1 since python is 0-indexed and tasks are 1-indexed
    if row_index > df_perf.index.max():
        sys.exit("Task number not needed for running simulation because they are all covered by other tasks.")
    # Subset the row with the failed model
    df_perf = df_perf.loc[row_index,:]
    if isinstance(df_perf, pd.Series):
        pass
    else:
        print("df_perf should be a pandas series but it's not:")
        print(df_perf)
        sys.exit("df_perf should be a pandas series but it's not. See the output file.")
    # reset sim year 
    sim_year = int(df_perf.year)
    failed_inp_to_rerun = df_perf.swmm_inp
    failed_inp_problem = df_perf.problem
    f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format(sim_year, row_index)
    f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format(sim_year, row_index)
# clear all re-run outputs
if which_models == "all":
    fs_re_runs_csvs = glob(dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format("*", "*"))
    fs_re_runs_netcdfs = glob(dir_swmm_sst_models + "_model_outputs_year{}_failed_run_id{}.nc".format("*", "*"))
    for f in fs_re_runs_csvs:
        os.remove(f)
    for f in fs_re_runs_netcdfs:
        os.remove(f)
if which_models == 'high_error':
    print("Re-running simulations with a routing error of greater than or equal to {} %".format(continuity_error_to_rerun))
    df_perf = pd.read_csv(f_model_perf_summary)
    df_perf = df_perf[df_perf.year == sim_year]
    df_perf = df_perf[df_perf.flow_continuity_error_rpt.abs() >=  continuity_error_to_rerun]
    df_perf.reset_index(inplace = True)
    max_runtime_min = time_permitted_for_reruns_min # allowing 3 hours per simulation
    # only use needed tasks (NOTE THE TOTAL NUMBER OF TASKS MUST EQUAL OR EXCEED THE NUMBER OF FAILED RUNS)
    row_index = (sim_year-1) # subtract 1 since python is 0-indexed and tasks are 1-indexed
    if len(df_perf) == 0:
        print("Task number not needed because all sims from this year met flow continuity error threshold.")
        sys.exit("Task number not needed because all sims from this year met flow continuity error threshold.")
    f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}_high_error.csv".format(sim_year)
    f_out_modelresults = dir_swmm_sst_models + "_model_outputs_year{}_high_error.nc".format(sim_year)
    lst_alternative_routing_tsteps = lst_tsteps_for_reruns # update to only use new higher timesteps
    continuity_error_thresh = continuity_error_to_rerun # making the threshold to re-run a little higher
# if which_models is an integer indicating to run a single storm id
try:
    storm_id_to_run = int(which_models)
    print("Running a single storm with ID {}".format(storm_id_to_run))
except Exception as e:
    pass

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
# def create_all_nan_dataset(a_fld_reshaped, rz, yr, storm_id, freebndry, norain, lst_keys):
#     # create dataset with na values with same shape as the flood data
#     a_zeros = np.empty(a_fld_reshaped.shape)
#     a_zeros[:] = np.nan
#     # create dataset with those na values
#     ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'sim_type', 'node_id'], a_zeros)),
#                     coords = dict(realization = np.atleast_1d(rz),
#                                     year = np.atleast_1d(yr),
#                                     storm_id = np.atleast_1d(storm_id),
#                                     freeboundary = np.atleast_1d(freebndry),
#                                     norain = np.atleast_1d(norain),
#                                     node_id = lst_keys
#                                     ))
#     return ds

#%% loading data
df_strms = pd.read_csv(f_swmm_scenarios_catalog.format(sim_year))
df_strms = df_strms.sort_values(["realization", "year", "storm_id"])
df_strms.drop(columns = "simulation_index", inplace = True)
df_strms.reset_index(drop=True, inplace=True)

# subset the df_strms table:
if realization_to_run is not None:
    df_strms = df_strms[df_strms.realization == realization_to_run]
if storm_id_to_run is not None:
    df_strms = df_strms[df_strms.storm_id == storm_id_to_run]
if which_models == "high_error":
    df_strms = df_strms[df_strms.swmm_inp.isin(df_perf.swmm_inp)]
if which_models == "failed":
    df_strms = df_strms[df_strms.swmm_inp == df_perf.swmm_inp]
#%% run simulations 
lst_ds_node_fld = []
lst_outputs_converted_to_dataset = [] # to track success
runtimes = []
export_dataset_times_min = []
lst_tot_loop_time_hr = []
# simulation stats to return
lst_flow_errors_frompyswmm = []
lst_runoff_errors_frompyswmm = []
lst_routing_timestep_used = []
## from rpt
lst_flow_errors_fromrpt = []
lst_runoff_errors_fromrpt = []
lst_total_flooding_system_rpt = []
lst_total_flooding_nodes_rpt = []
lst_frac_diff_node_minus_system_flood_rpt = []

lst_runoff_continuity_issues = []
lst_flow_continuity_issues = []

lst_inp_files_to_keep = []
lst_rpt_files_to_keep = []
lst_out_files_to_keep = [] # these are for resuming runs from existing hotstarts
# export_dataset_times = []
successes = []
problems = []
notes = []
count = -1

# return folder of SWMM models
f_inp_name = df_strms.swmm_inp.iloc[0].split("/")[-1]
swmm_fldr = df_strms.swmm_inp.iloc[0].split(f_inp_name)[0]
print("Running SWMM models in folder: {}".format(swmm_fldr))
# make sure directory is set up for rpt backups
rpt_copy_fldr = swmm_fldr + "rpt_backup/"
rpt_copy_directory = Path(rpt_copy_fldr)
rpt_copy_directory.mkdir(parents=True, exist_ok=True)

if remove_previous_runs == True: # this can only be toggled on if which_models = "all"
    old_rpt_files = glob(rpt_copy_fldr + "*")
    contents_in_swmm_folder = glob(swmm_fldr + "*")
    files_in_swmm_folder = [path for path in contents_in_swmm_folder if os.path.isfile(path)]
    files_to_remove = []
    for f_compare in files_in_swmm_folder:
        keeper = False
        for f_swmm_to_keep in df_strms.swmm_inp:
            # if the file is one of the core swmm files, keep it
            if os.path.samefile(f_swmm_to_keep, f_compare):
                keeper = True
        if keeper == False:
            files_to_remove.append(f_compare)
    for f_to_remove in (files_to_remove + old_rpt_files):
        os.remove(f_to_remove)

for idx, row in df_strms.iterrows():
    note = ""
    problem = "None"
    f_inp = row["swmm_inp"]
    rz = int(row["realization"])
    yr = int(row["year"])
    storm_id = int(row["storm_id"])
    count += 1
    print("################################################################################")
    # print("Running sims for {}. {} out of {} simulations complete.".format(f_inp_name, count, len(df_strms)))
    output_converted_to_dataset = False
    loop_start_time = sim_time = datetime.now()
    sim_runtime_min = np.nan
    # if using previous results:
    if which_models == "previous":
        sim_runtime_min = -9999
        routing_tstep_to_report = -9999
        lowest_error = -9999
        f_inp_name = f_inp.split("/")[-1]
        rpt_name_pattern = f_inp_name.split(".inp")[0] + "_rt*.rpt"
        rpt_pattern = rpt_copy_fldr + rpt_name_pattern
        lst_f_rpts = glob(rpt_pattern)
        # find the rpt output and routing timestep with the lowest error
        for rpt_path in lst_f_rpts:
            routing_tstep = rpt_path.split("_rt")[-1].split(".rpt")[0]
            s_node_flooding,total_flooding_system_rpt,runoff_error_rpt,\
                        flow_routing_error_rpt,frac_diff_node_minus_system_flood_rpt,flow_units = return_flood_losses_and_continuity_errors(rpt_path, f_inp)
            if abs(flow_routing_error_rpt) < abs(lowest_error):
                routing_tstep_to_report = routing_tstep
                lowest_error = lowest_error
            # evaluate errors
            if (abs(flow_routing_error_rpt) <= continuity_error_thresh):
                flow_continuity_issues = False
            else:
                flow_continuity_issues = True

            if (abs(runoff_error_rpt) <= continuity_error_thresh):
                runoff_continuity_issues = False
            else:
                runoff_continuity_issues = True
            # define values to append later
            flow_routing_error_pyswmm = np.nan
            runoff_error_pyswmm = np.nan
        success = True
    # if running simulations..
    else:
        for routing_tstep in lst_alternative_routing_tsteps:
            routing_tstep_to_report = routing_tstep
            use_hotstart = False
            if which_models == "failed":
                idx_routing_tstep = lst_alternative_routing_tsteps.index(routing_tstep)
                idx_of_routing_tstep_of_last_attempted_sim = np.nan
                if df_perf.routing_timestep in lst_alternative_routing_tsteps:
                    idx_of_routing_tstep_of_last_attempted_sim = lst_alternative_routing_tsteps.index(df_perf.routing_timestep)
                # if a simulation has already been completed previously and was rejected due to high continuity errors, skip it
                previous_sim_run = False
                if idx_of_routing_tstep_of_last_attempted_sim > idx_routing_tstep: # won't trigger if the routing timestep used is no longer in the list of alternative routing timesteps
                    previous_sim_run = True
                    previous_routing_tstep = routing_tstep
                    f_inp_prevrun = f_inp.split(".inp")[0] + "_rt" + str(previous_routing_tstep) + ".inp"
                    f_out_prevrun = f_inp.split(".inp")[0] + "_rt" + str(previous_routing_tstep) + ".out"
                    f_rpt_prevrun = rpt_copy_fldr + f_inp_prevrun.split("/")[-1].split(".inp")[0] + ".rpt"
                    prev_rpt_file_exists = os.path.exists(f_rpt_prevrun)
                    continue
                if previous_sim_run:
                    print("Previous simulation was run with a routing timestep of: {}".format(previous_routing_tstep))
                # if attempting the routing timestep that ran into the runtime limit is the one up for trial, use the hotstart trial
                if idx_routing_tstep == idx_of_routing_tstep_of_last_attempted_sim:
                    use_hotstart = True
                    first_sim_attempt = True
                else:
                    first_sim_attempt = False
                if use_hotstart_override:
                    use_hotstart = hotstart_override_val
                ## if first sim for failed model AND there has already been a succesful run at a previous timestep
                if first_sim_attempt and previous_sim_run and prev_rpt_file_exists:
                    __,__,previous_runoff_error_pyswmm,\
                        previous_flow_routing_error_pyswmm,__,__ = return_flood_losses_and_continuity_errors(f_rpt_prevrun, f_inp_prevrun)
                    first_sim_attempt = False
            else:
                first_sim_attempt = (routing_tstep == lst_alternative_routing_tsteps[0])
            # modify inp file with routing timestep
            ## define filepath to new inp file
            f_inp_torun = f_inp.split(".inp")[0] + "_rt" + str(routing_tstep) + ".inp"
            f_inp_name = f_inp_torun.split("/")[-1]
            print("Running {}".format(f_inp_name))
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
            with open(f_inp_torun, 'w') as file:
                file.writelines(lines)
            # run simulation
            runoff_continuity_issues = np.nan
            flow_continuity_issues = np.nan
            runoff_error_pyswmm = -9999
            flow_routing_error_pyswmm = -9999
            runoff_error_rpt = -9999
            flow_routing_error_rpt = -9999
            tot_flood_losses_rpt_system_m3 = -9999
            tot_flood_losses_rpt_nodes_m3 = -9999
            frac_diff_node_minus_system_flood_rpt = -9999
            success = True
            try:
                with Simulation(f_inp_torun) as sim:
                    sim_start_time = datetime.now()
                    f_hotpath = f_inp_torun+".hot"
                    # check to make sure output files exist before using hotstart
                    f_out = f_inp_torun.split('.inp')[0] + '.out'
                    f_out_exists = os.path.exists(f_out)
                    f_hotstart_exists = os.path.exists(f_hotpath)
                    hotstart_used = False
                    if use_hotstart and f_out_exists and f_hotstart_exists:
                        hotstart_used = True
                        sim.use_hotstart(f_hotpath)
                        print("Using hotstart file to save some time on a previously incomplete run.....")
                    for step in sim:
                        sim_time = datetime.now()
                        sim_runtime_min = round((sim_time - sim_start_time).seconds / 60, 1)
                        if sim_runtime_min > max_runtime_min:
                            problem = "User-defined maximum simulation time limit of {} minutes reached, so simulation was halted.".format(max_runtime_min)
                            print(problem)
                            success = False
                            sim.save_hotstart(f_hotpath) # saving hotstart file so we can resume it when running failed models
                            sim.terminate_simulation()
                    sim._model.swmm_end() # must call this to accurately report the continuity errors
                    runoff_error_pyswmm = sim.runoff_error
                    flow_routing_error_pyswmm = sim.flow_routing_error
                    # write report file
                    # sim.report()
                    # sim.close()
            except Exception as e:
                print("Simulation failed due to error: {}".format(e))
                # problem = e
                success = False
            if success: # check continuity error and re-run if it is worse than a threshold target
                # remove hotstart file if the simulation completes succesfully
                if hotstart_used:
                    os.remove(f_hotpath)
                # save the rpt file to another directory to see if i can figure out what's going on with the dumping thing
                rpt_path = f_inp_torun.split('.inp')[0] + ".rpt"
                # rpt_name = rpt_path.split("/")[-1]
                source_file_path = Path(rpt_path)
                shutil.copy(source_file_path, rpt_copy_directory)
                # record runoff error
                if (abs(runoff_error_pyswmm) <= continuity_error_thresh):
                    runoff_continuity_issues = False
                else:
                    runoff_continuity_issues = True
                if (abs(flow_routing_error_pyswmm) <= continuity_error_thresh): # and (abs(runoff_error_pyswmm) <= continuity_error_thresh):
                    # print("Simulation succesfully completed with continuity errors within prespecified threshold of {}% using a routing timestep of {}. Flow routing and runoff errors are {} and {}".format(
                    #     continuity_error_thresh, routing_tstep, flow_routing_error_pyswmm, runoff_error_pyswmm
                    # ))
                    flow_continuity_issues = False
                    break
                else:
                    flow_continuity_issues = True
                    # is this the first simulation attempt?
                    ## if normal simulation run for all models or specific storms
                    if first_sim_attempt:
                        previous_flow_routing_error_pyswmm = flow_routing_error_pyswmm
                        previous_runoff_error_pyswmm = runoff_error_pyswmm
                        print("The simulation was run with a routing timestep of {}. Runoff and flow continuity errors were {} and {}. Sim runtime was {}. Re-running simulation.".format(
                            routing_tstep, runoff_error_pyswmm, flow_routing_error_pyswmm, sim_runtime_min))
                        previous_routing_tstep = routing_tstep
                        routing_tstep_to_report = previous_routing_tstep
                    else: # not first sim attempt
                        # compute the improvement between the last two attempts
                        net_improvement = abs(previous_flow_routing_error_pyswmm) - abs(flow_routing_error_pyswmm)
                        frac_improvement = net_improvement / abs(previous_flow_routing_error_pyswmm)
                        if frac_improvement < min_improvement_to_warrant_another_sim:
                            if frac_improvement < 0:
                                print("The simulation was run with a routing timestep of {}. Flow continuity error was {} which was actually WORSE than the previous run so I won't be trying a smaller routing timestep. Sim runtime was {}.".format(
                                    routing_tstep, flow_routing_error_pyswmm, sim_runtime_min))
                                note = note + "This routing timestep actually did {}% worse than *{}*s which resulted in a flow continuity error of {}%;".format(
                                    round(frac_improvement*100,1), previous_routing_tstep, previous_flow_routing_error_pyswmm
                                )
                                routing_tstep_to_report = previous_routing_tstep
                                print("Routing timestep being reported: {}".format(routing_tstep_to_report))
                                flow_routing_error_pyswmm = previous_flow_routing_error_pyswmm
                                runoff_error_pyswmm = previous_runoff_error_pyswmm
                            else:
                                print("The simulation was run with a routing timestep of {}. Flow continuity error was {} which is less than {}% better than the previous run. This does not warrant another simulation. Sim runtime was {}.".format(
                                    routing_tstep, flow_routing_error_pyswmm, min_improvement_to_warrant_another_sim*100, sim_runtime_min))
                                print("Routing timestep being reported: {}".format(routing_tstep_to_report))
                            break
                        else:
                            previous_flow_routing_error_pyswmm = flow_routing_error_pyswmm
                            previous_runoff_error_pyswmm = runoff_error_pyswmm
                            previous_routing_tstep = routing_tstep
                            routing_tstep_to_report = previous_routing_tstep
            else: # if simulation didn't run because of an error or the time limit, don't re-run the sim
                break
    notes.append(note)
    problems.append(problem)
    successes.append(success)
    lst_runoff_continuity_issues.append(runoff_continuity_issues)
    lst_flow_continuity_issues.append(flow_continuity_issues)
    # record flow and runoff errors
    lst_flow_errors_frompyswmm.append(flow_routing_error_pyswmm)
    lst_runoff_errors_frompyswmm.append(runoff_error_pyswmm)
    lst_routing_timestep_used.append(routing_tstep_to_report)
    # benchmarking write netcdf
    start_create_dataset = datetime.now()
    create_dataset_time_min = np.nan
    # if the run was succesful, process the results
    f_inp_to_report = f_inp.split(".inp")[0] + "_rt" + str(routing_tstep_to_report) + ".inp"
    # use the rpt file that was copied to the backup folder
    f_swmm_out = f_inp_to_report.split('.inp')[0] + '.out'
    rpt_name = f_inp_to_report.split("/")[-1].split(".inp")[0] + ".rpt"
    rpt_path = rpt_copy_fldr + rpt_name
    if success == True:
        analysis_end_datetime = return_analysis_end_date(rpt_path)
        lst_inp_files_to_keep.append(f_inp_to_report)
        lst_rpt_files_to_keep.append(rpt_path)
        __, __, __, __, __, sim_type = parse_inp(f_inp_to_report) # this function also returns rz, yr, storm_id which are not needed since they were determined earlier
        s_node_flooding,total_flooding_system_rpt,runoff_error_rpt,\
            flow_routing_error_rpt,frac_diff_node_minus_system_flood_rpt,flow_units = return_flood_losses_and_continuity_errors(rpt_path, f_inp_to_report)
        
        if flow_units == "cfs":
            tot_flood_losses_rpt_system_m3 = total_flooding_system_rpt * 1e6 * cubic_meters_per_gallon # Million gallons * gallons per million gallons * cubic meters per gallon
            node_flooding_m3 = s_node_flooding * 1e6 * cubic_meters_per_gallon # default units are in millions of gallons
        else:
            sys.exit('UNITS NOT RECOGNIZED; NEED TO BE UPDATED FOR METRIC PROBABLY')

        tot_flood_losses_rpt_nodes_m3 = node_flooding_m3.sum()
        # create array of flooded values with the correct shape for placing in xarray dataset
        a_fld_reshaped = np.reshape(np.array(node_flooding_m3), (1,1,1,1,len(node_flooding_m3))) # rz, yr, storm, sim_type, node_id
        a_sys_flding = np.reshape(np.array(tot_flood_losses_rpt_system_m3), (1,1,1,1)) # rz, yr, storm, sim_type
        a_tot_node_flding = np.reshape(np.array(tot_flood_losses_rpt_nodes_m3), (1,1,1,1)) # rz, yr, storm, sim_type
        a_flow_routing_error = np.reshape(np.array(flow_routing_error_rpt), (1,1,1,1)) # rz, yr, storm, sim_type
        a_runoff_error = np.reshape(np.array(runoff_error_rpt), (1,1,1,1)) # rz, yr, storm, sim_type

        a_routing_timestep = np.reshape(np.array(float(routing_tstep_to_report)), (1,1,1,1)) # rz, yr, storm, sim_type

        a_inp_file = np.reshape(np.array(f_inp_to_report), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rpt_file = np.reshape(np.array(rpt_path), (1,1,1,1)) # rz, yr, storm, sim_type

        a_rain0_file = np.reshape(np.array(row["rainfall_0"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rain1_file = np.reshape(np.array(row["rainfall_1"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rain2_file = np.reshape(np.array(row["rainfall_2"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rain3_file = np.reshape(np.array(row["rainfall_3"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rain4_file = np.reshape(np.array(row["rainfall_4"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_rain5_file = np.reshape(np.array(row["rainfall_5"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_waterlevel_file = np.reshape(np.array(row["water_level"]), (1,1,1,1)) # rz, yr, storm, sim_type
        a_analysis_end_datetime = np.reshape(np.array(analysis_end_datetime), (1,1,1,1)) # rz, yr, storm, sim_type
        
        # create dataset with the flood values
        ds = xr.Dataset(data_vars=dict(node_flooding_cubic_meters = (['realization', 'year', 'storm_id', 'sim_type', 'node_id'], a_fld_reshaped),
                                       sys_flding_cubic_meters = (['realization', 'year', 'storm_id', 'sim_type'], a_sys_flding),
                                       tot_node_flding_cubic_meters = (['realization', 'year', 'storm_id', 'sim_type'], a_tot_node_flding),
                                       flow_routing_error_percent = (['realization', 'year', 'storm_id', 'sim_type'], a_flow_routing_error),
                                       runoff_error_percent = (['realization', 'year', 'storm_id', 'sim_type'], a_runoff_error),
                                       routing_timestep = (['realization', 'year', 'storm_id', 'sim_type'], a_routing_timestep),
                                       analysis_end_datetime = (['realization', 'year', 'storm_id', 'sim_type'], a_analysis_end_datetime),
                                       file_inp = (['realization', 'year', 'storm_id', 'sim_type'], a_inp_file),
                                       file_rpt = (['realization', 'year', 'storm_id', 'sim_type'], a_rpt_file),
                                       file_rain0 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain0_file),
                                       file_rain1 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain1_file),
                                       file_rain2 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain2_file),
                                       file_rain3 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain3_file),
                                       file_rain4 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain4_file),
                                       file_rain5 = (['realization', 'year', 'storm_id', 'sim_type'], a_rain5_file),
                                       file_waterlevel = (['realization', 'year', 'storm_id', 'sim_type'], a_waterlevel_file)),
                        coords = dict(realization = np.atleast_1d(rz),
                                        year = np.atleast_1d(yr),
                                        storm_id = np.atleast_1d(storm_id),
                                        sim_type = np.atleast_1d(sim_type),
                                        node_id = node_flooding_m3.index.values
                                        ))
        # sort
        ds = ds.sortby(['realization', 'year', 'storm_id', 'sim_type', 'node_id'])
        lst_ds_node_fld.append(ds)
        # lst_f_outputs_converted_to_netcdf.append(f_swmm_out)
        output_converted_to_dataset = True
        # print("created xarray dataset with total flooding for each node")
        end_create_dataset = datetime.now()
        create_dataset_time_min = round((end_create_dataset - start_create_dataset).seconds / 60, 1)
        # delete output file
        if delete_swmm_outputs:
            os.remove(f_swmm_out)
    else:
        # keep all simulation files for failed runs
        lst_rpt_files_to_keep.append(rpt_path)
        lst_out_files_to_keep.append(f_swmm_out)
        lst_inp_files_to_keep.append(f_inp_to_report)
    # recording stuff that would be gotten from rpt
    lst_flow_errors_fromrpt.append(flow_routing_error_rpt)
    lst_runoff_errors_fromrpt.append(runoff_error_rpt)
    lst_total_flooding_system_rpt.append(tot_flood_losses_rpt_system_m3)
    lst_total_flooding_nodes_rpt.append(tot_flood_losses_rpt_nodes_m3)
    lst_frac_diff_node_minus_system_flood_rpt.append(frac_diff_node_minus_system_flood_rpt)
    export_dataset_times_min.append(create_dataset_time_min)
    # benchmarking
    tot_loop_time_hr = round((datetime.now() - loop_start_time).seconds / 60 / 60, 1)
    lst_tot_loop_time_hr.append(tot_loop_time_hr)
    mean_loop_time_hr = np.nanmean(lst_tot_loop_time_hr)
    runtimes.append(sim_runtime_min)
    if success == True:
        # benchmarking entire loop and script
        ## benchmarking export time
        mean_export_ds_time_min = round(np.nanmean(export_dataset_times_min), 1)
        ## benchmarking simulations
        mean_sim_time_min = round(np.nanmean(runtimes), 1)
        ## estimating time remaining assuming successes
        estimated_loop_time = max([mean_sim_time_min+mean_export_ds_time_min, mean_loop_time_hr])
        expected_tot_runtime_hr = round(estimated_loop_time*len(df_strms)/60, 1)
        tot_elapsed_time_hr = round((datetime.now() - script_start_time).seconds / 60 / 60, 1)
        expected_remaining_time_hr = round((expected_tot_runtime_hr - tot_elapsed_time_hr), 1)
        print("Total loop time (hr): {}, Sim runtime (min): {}, Mean sim runtime (min): {}, Time to create dataset (min): {}, Total script time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(
            tot_loop_time_hr, sim_runtime_min, mean_sim_time_min,create_dataset_time_min,tot_elapsed_time_hr, expected_tot_runtime_hr, expected_remaining_time_hr)) 
    else: 
        print("Simulation failed. Attempt took {} hrs".format(tot_loop_time_hr))
    lst_outputs_converted_to_dataset.append(output_converted_to_dataset)
    # if only running single simulation, stop the script here
    if which_models == "failed":
        break


if which_models == "all": # only remove files if running all models
    # remove all files but swmm input files, rpt files, and .hot files
    ## define list of all files
    contents_in_swmm_folder = glob(swmm_fldr + "*")
    ### do not include directories in the list
    files_in_swmm_folder = [path for path in contents_in_swmm_folder if os.path.isfile(path)]
    files_in_rpt_backup_folder = glob(rpt_copy_fldr + "*")
    lst_all_files = files_in_swmm_folder + files_in_rpt_backup_folder
    ## define list of all files to keep
    lst_hotstarts = glob(swmm_fldr + "*.hot")
    original_swmm_files = list(df_strms.swmm_inp)
    lst_to_keep = lst_hotstarts + original_swmm_files + lst_rpt_files_to_keep + lst_inp_files_to_keep + lst_out_files_to_keep
    files_to_remove = []
    for f_compare in lst_all_files:
        keeper = False
        try:
            for f_keeper in lst_to_keep:
                # if the file is one of the core swmm files, keep it
                if os.path.samefile(f_keeper, f_compare):
                    keeper = True
        except:
            pass
        if keeper == False:
            files_to_remove.append(f_compare)
    for f_to_remove in files_to_remove:
        try:
            os.remove(f_to_remove)
        except:
            pass
#%% export results and model summaries
if len(lst_ds_node_fld) > 0:
    at_least_1_sim_was_succesfull = True
else:
    at_least_1_sim_was_succesfull = False
if which_models == "failed":
    ds_all_node_fld = ds # single output only
try:
    ds_all_node_fld = xr.combine_by_coords(lst_ds_node_fld)
    if at_least_1_sim_was_succesfull:
        ds_all_node_fld_loaded = ds_all_node_fld.load()
        ds_all_node_fld_loaded.to_netcdf(f_out_modelresults, encoding= {"node_flooding_cubic_meters":{"zlib":True}}, engine = "h5netcdf")
        print("exported " + f_out_modelresults)
except Exception as e:
    print("Failed to export node flooding datasets due to error: {}")
    print("Looping through datasets to see if the issue can be spotted: ")
    for ds in lst_ds_node_fld:
        print(ds)
# export performance info
df_strms["run_completed"] = successes
df_strms["routing_timestep"] = lst_routing_timestep_used
df_strms["flow_continuity_error_pyswmm"] = lst_flow_errors_frompyswmm
df_strms["runoff_continuity_error_pyswmm"] = lst_runoff_errors_frompyswmm
df_strms["flow_continuity_error_rpt"] = lst_flow_errors_fromrpt
df_strms["runoff_continuity_error_rpt"] = lst_runoff_errors_fromrpt
df_strms["runoff_continuity_error_exceeds_threshold"] = lst_runoff_continuity_issues
df_strms["flow_continuity_error_exceeds_threshold"] = lst_flow_continuity_issues
df_strms["total_system_flooding_rpt_m3"] = lst_total_flooding_system_rpt
df_strms["total_flooding_from_nodes_rpt_m3"] = lst_total_flooding_nodes_rpt
df_strms["frac_diff_node_minus_system_flood_rpt"] = lst_frac_diff_node_minus_system_flood_rpt
df_strms["problem"] = problems
df_strms["notes"] = notes
df_strms["runtime_min"] = runtimes
df_strms["export_dataset_min"] = export_dataset_times_min
df_strms["lst_outputs_converted_to_netcdf"] = lst_outputs_converted_to_dataset
df_strms.to_csv(f_out_runtimes, index=False)
print('Exported ' + f_out_runtimes)
tot_elapsed_time_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Total script runtime (hr): {}".format(tot_elapsed_time_min/60))