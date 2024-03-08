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
    if ("failed_run" in f) or ("high_error" in f): # skip csvs from failed runs and re-runs with high errors
        continue
    df = pd.read_csv(f)
    df["run_type"] = "initial"
    lst_dfs_perf.append(df)
df_perf = pd.concat(lst_dfs_perf)
df_perf = df_perf.set_index(["swmm_inp"], drop=True)

# update the consolidated performance summary with the re-runs
if len(lst_f_perf_reruns) > 0:
    lst_dfs_perf_reruns = []
    for f in lst_f_perf_reruns:
        df_reruns = pd.read_csv(f)
        df_reruns["run_type"] = "rerun_of_failed_sims"
        lst_dfs_perf_reruns.append(df_reruns)
    df_perf_reruns = pd.concat(lst_dfs_perf_reruns)
    df_perf_reruns = df_perf_reruns.set_index(["swmm_inp"], drop=True)
    for ind, row in df_perf_reruns.iterrows():
        df_perf.loc[ind, :] = row

# update the dataframe with high error re-runs
if len(lst_f_perf_high_error_reruns) > 0:
    lst_dfs_perf_high_error_reruns = []
    for f in lst_f_perf_high_error_reruns:
        df_high_error_reruns = pd.read_csv(f)
        df_high_error_reruns["run_type"] = "rerun_of_sims_with_high_routing_error"
        lst_dfs_perf_high_error_reruns.append(df_high_error_reruns)
    df_perf_high_error_reruns = pd.concat(lst_dfs_perf_high_error_reruns)
    df_perf_high_error_reruns = df_perf_high_error_reruns.set_index(["swmm_inp"], drop=True)
    # check and see if there is a failed re-run and use the most recent result
    for ind, row in df_perf_high_error_reruns.iterrows():
        # check if this high_error re-run is present in the failed runs too
        use_high_error_result = True
        if ind in df_perf_reruns.index:
            row_failed_rerun = df_perf_reruns.loc[ind,:]
            analysis_end_datetime_failed = return_analysis_end_date(ind, row_failed_rerun.routing_timestep)
            analysis_end_datetime_higherror_rerun = return_analysis_end_date(ind, row.routing_timestep)
            # if the failed re-run has an analysis end date LATER than the high error run, keep the failed re-run
            if analysis_end_datetime_failed > analysis_end_datetime_higherror_rerun:
                use_high_error_result = False
        if use_high_error_result:
            df_perf.loc[ind, :] = row

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

df_perf = df_perf.reset_index()
df_perf = df_perf.set_index(idx_vals, drop=True)
df_perf['run_attempted'] = 1
df_perf = df_perf.join(df_strms['dummy'], how = "right")
df_perf['run_attempted'] = df_perf['run_attempted'].fillna(0).astype(bool)
df_perf.drop(columns="dummy", inplace = True)
df_perf = df_perf.reset_index()


# export performance dataframe
df_perf.to_csv(f_model_perf_summary, index=False)
print("Exported file {}".format(f_model_perf_summary))

#%% load event summaries
# lst_f_events = glob(f_summaries)
# lst_dfs_events = []
# for f in lst_f_events:
#     df = pd.read_csv(f)
#     lst_dfs_events.append(df)

# df_events = pd.concat(lst_dfs_events)

# df_events.rename(columns=dict(realization_id = "realization"), inplace=True)

# df_events_and_perf = df_perf.merge(df_events, on = ["realization", "year", "storm_id"])

# export eventsormance dataframe
# df_events.to_csv(f_events_summary, index=False)
# print("Exported file {}".format(f_events_summary))