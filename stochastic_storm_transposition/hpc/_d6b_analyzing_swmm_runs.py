#%% import libraries
import pandas as pd
from __utils import *
from glob import glob
from pathlib import Path
from datetime import datetime

f_performance = dir_swmm_sst_models + "_model_performance_year{}.csv".format("*")
# f_summaries = dir_time_series + "_event_summary_year{}.csv".format("*")
f_summaries = f_swmm_scenarios_catalog.format("*")
f_performance_reruns = dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format("*", "*")
f_performance_high_error = dir_swmm_sst_models + "_model_performance_year{}_high_error.csv".format("*")
# define file pattern of rpt files


#%% load performance dataframe
lst_f_perf = glob(f_performance)
lst_f_perf_reruns = glob(f_performance_reruns)
lst_f_perf_high_error_reruns = glob(f_performance_high_error)
lst_dfs_perf = []

for f in lst_f_perf:
    desc = "all_sims"
    if "failed_run" in f:
        desc = "failed_rerun"
    if "high_error" in f:
        desc = "high_error_rerun"
    df = pd.read_csv(f)
    df["run_type"] = desc
    for ind, row in df.iterrows():
        swmm_inp = row.swmm_inp
        analysis_end_datetime = check_rpt_results(swmm_inp, row.routing_timestep) 
        df.loc[ind, "analysis_end_datetime"] = analysis_end_datetime
    lst_dfs_perf.append(df)
df_perf = pd.concat(lst_dfs_perf)


# sort by analysis end_date_time (most recent at top), then groupby swmm inp, then use head(1) to choose the most recent run of each swmm model
df_perf = df_perf.sort_values("analysis_end_datetime", ascending = False).groupby("swmm_inp").head(1)

# df_perf = df_perf.set_index(["swmm_inp"], drop=True)
df_perf = df_perf.sort_values(["realization", "year", "storm_id"]).reset_index(drop = True)

# identify all runs that weren't attempted
## load all swmm scenarios
lst_f_swmm_scenarios = glob(f_swmm_scenarios_catalog.format("*"))
lst_dfs_strms = []
for f in lst_f_swmm_scenarios:
    df = pd.read_csv(f)
    lst_dfs_strms.append(df)
df_strms = pd.concat(lst_dfs_strms)
df_strms = df_strms.sort_values(["realization", "year", "storm_id"])
df_strms.drop(columns = "simulation_index", inplace = True)
df_strms.reset_index(drop=True, inplace=True)
df_strms['dummy'] = 0
idx_vals = ["swmm_inp", 'realization', 'year', 'storm_id', 'rainfall_0', 'rainfall_1',
       'rainfall_2', 'rainfall_3', 'rainfall_4', 'rainfall_5', 'water_level']
df_strms.set_index(idx_vals, inplace = True)

# df_perf = df_perf.reset_index()
df_perf = df_perf.set_index(idx_vals, drop=True)
df_perf['run_attempted'] = 1
df_perf = df_perf.join(df_strms['dummy'], how = "right")
df_perf['run_attempted'] = df_perf['run_attempted'].fillna(0).astype(bool)
df_perf.drop(columns="dummy", inplace = True)
df_perf = df_perf.reset_index()
df_perf = df_perf[df_perf.run_attempted == True]

# export performance dataframe
df_perf.to_csv(f_model_perf_summary, index=False)
print("Exported file {}".format(f_model_perf_summary))