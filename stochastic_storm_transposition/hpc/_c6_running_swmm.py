#%% loading libraries and importing directories
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from pyswmm import Simulation
from datetime import datetime

from __utils import c6_running_swmm

f_swmm_scenarios_catalog, dir_swmm_sst_models, max_runtime_min = c6_running_swmm()

sim_year = int(sys.argv[1]) # a number between 1 and 1000

f_out_runtimes = dir_swmm_sst_models + "_model_performance_year{}.csv".format(sim_year)

script_start_time = datetime.now()
#%% loading data
df_strms = pd.read_csv(f_swmm_scenarios_catalog.format(sim_year))
df_strms = df_strms.loc[df_strms.year==sim_year]

s_tot_rz = int(df_strms.realization.max())
s_tot_storms = int(df_strms.storm_num.max())
s_tot_sims = s_tot_rz * s_tot_storms
#%% define functions
def parse_inp(f_inp):
    # format "rz{}_yr{}_strm{}.inp"
    lst_name_comp = f_inp.split("/")[-1].split("_")
    rz = lst_name_comp[0].split("rz")[-1]
    yr = lst_name_comp[1].split("yr")[-1]
    storm_id = lst_name_comp[2].split(".")[0].split('strm')[-1]
    return int(rz), int(yr), int(storm_id)

#%% run simulations
runtimes = []
successes = []
count = -1
for f_inp in df_strms.swmm_inp:
    count += 1
    rz, yr, storm_id = parse_inp(f_inp)
    print("Running simulation for realization {}/{}, year {}, storm {}/{}. {} out of {} simulations complete.".format(rz, s_tot_rz, yr, storm_id, s_tot_storms, count, s_tot_sims))
    success = True
    with Simulation(f_inp) as sim:
        sim_start_time = datetime.now()
        for step in sim:
            sim_time = datetime.now()
            sim_runtime_min = round((sim_time - sim_start_time).seconds / 60, 1)
            if sim_runtime_min > max_runtime_min:
                print("triggered")
                success = False
                break
            pass
    successes.append(success)
    tot_elapsed_time_hr = round((sim_time - script_start_time).seconds / 60 / 60, 1)
    runtimes.append(sim_runtime_min)
    mean_sim_time = round(np.mean(runtimes), 1)
    expected_tot_runtime_hr = round(mean_sim_time*s_tot_sims/60, 1)
    expected_remaining_time_hr = round((expected_tot_runtime_hr - tot_elapsed_time_hr), 1)
    # MONTIROING
    if success == True:
        print("Simulation runtime (min): {}, Mean simulation runtime (min): {}, Total elapsed time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(sim_runtime_min, tot_elapsed_time_hr, mean_sim_time, expected_tot_runtime_hr, expected_remaining_time_hr)) 
    else: 
        print("Simulation cancelled after {} minutes due to user-defined runtime limits, Mean simulation runtime (min): {}, Total elapsed time (hr): {}, Expected total time (hr): {}, Estimated time remaining (hr): {}".format(max_runtime_min, tot_elapsed_time_hr, mean_sim_time, expected_tot_runtime_hr, expected_remaining_time_hr)) 

#%% export model runtimes to a file
df_strms['runtime_min'] = runtimes
df_strms['run_completed'] = successes

df_strms.to_csv(f_out_runtimes, index=False)