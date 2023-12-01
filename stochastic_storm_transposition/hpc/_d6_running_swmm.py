#%% loading libraries and importing directories
sim_year = int(sys.argv[1])

import numpy as np
from pathlib import Path
import pandas as pd
import sys
from pyswmm import Simulation
from datetime import datetime
from __utils import *

# from __utils import c6_running_swmm, parse_inp

# f_swmm_scenarios_catalog, dir_swmm_sst_models, max_runtime_min = c6_running_swmm()

 # a number between 1 and 1000

f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}.csv".format(sim_year)

script_start_time = datetime.now()
#%% loading data
df_strms = pd.read_csv(f_swmm_scenarios_catalog.format(sim_year))
if "storm_num" in df_strms.columns: # this should become irrelevant, this was just so I didn't have to re-run previous script with desired column names
    df_strms = df_strms.rename(columns=dict(storm_num = "storm_id"))

df_strms = df_strms.sort_values(["realization", "year", "storm_id"])

df_strms.drop(columns = "simulation_index", inplace = True)

df_strms.reset_index(drop=True, inplace=True)

s_tot_rz = int(df_strms.realization.max())
s_strms_per_year = int(df_strms.storm_id.max())
s_tot_sims = len(df_strms)

#%% run simulations
runtimes = []
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
    print("Running simulation for realization {} year {} storm {}. {} out of {} simulations complete.".format(rz, yr, storm_id, s_strms_per_year, count, s_tot_sims))
    success = True
    sim_time = datetime.now()
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
    tot_elapsed_time_hr = round((sim_time - script_start_time).seconds / 60 / 60, 1)
    runtimes.append(sim_runtime_min)
    mean_sim_time = round(np.mean(runtimes), 1)
    expected_tot_runtime_hr = round(mean_sim_time*s_tot_sims/60, 1)
    expected_remaining_time_hr = round((expected_tot_runtime_hr - tot_elapsed_time_hr), 1)
    # MONTIROING
    if success == True:
        print("Simulation runtime (min): {}, Mean simulation runtime (min): {}, Total elapsed time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(sim_runtime_min, mean_sim_time, tot_elapsed_time_hr, expected_tot_runtime_hr, expected_remaining_time_hr)) 
    else: 
        print("Simulation failed after {} minutes.".format(sim_runtime_min))

#%% export model runtimes to a file
df_strms["run_completed"] = successes
df_strms["problem"] = problems
df_strms["runtime_min"] = runtimes

df_strms.to_csv(f_out_runtimes, index=False)