#%% import libraries
import pandas as pd
from __utils import *
from glob import glob
from pathlib import Path


f_performance = dir_swmm_sst_models + "_model_performance_year{}.csv".format("*")
# f_summaries = dir_time_series + "_event_summary_year{}.csv".format("*")
f_summaries = f_swmm_scenarios_catalog.format("*")
f_performance_reruns = dir_swmm_sst_models + "_model_performance_year{}_failed_run_id{}.csv".format("*", "*")

#%% load performance dataframe
lst_f_perf = glob(f_performance)
lst_f_perf_reruns = glob(f_performance_reruns)
lst_dfs_perf = []
for f in lst_f_perf:
    if "failed_run" in f:
        continue
    df = pd.read_csv(f)
    lst_dfs_perf.append(df)
df_perf = pd.concat(lst_dfs_perf)
df_perf = df_perf.set_index(["swmm_inp"], drop=True)

# update the consolidated performance summary with the re-runs
lst_dfs_perf_reruns = []
for f in lst_f_perf_reruns:
    df_reruns = pd.read_csv(f)
    lst_dfs_perf_reruns.append(df_reruns)
df_perf_reruns = pd.concat(lst_dfs_perf_reruns)
df_perf_reruns = df_perf_reruns.set_index(["swmm_inp"], drop=True)

for ind, row in df_perf_reruns.iterrows():
    df_perf.loc[ind, :] = row


# export performance dataframe
df_perf.to_csv(f_model_perf_summary, index=True)


#%% load event summaries
lst_f_events = glob(f_summaries)
lst_dfs_events = []
for f in lst_f_events:
    df = pd.read_csv(f)
    lst_dfs_events.append(df)

df_events = pd.concat(lst_dfs_events)

# df_events.rename(columns=dict(realization_id = "realization"), inplace=True)

# df_events_and_perf = df_perf.merge(df_events, on = ["realization", "year", "storm_id"])

# export eventsormance dataframe
df_events.to_csv(f_events_summary, index=False)