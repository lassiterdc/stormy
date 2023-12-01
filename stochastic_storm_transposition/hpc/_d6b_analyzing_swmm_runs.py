#%% import libraries
import pandas as pd
from __utils import *
from glob import glob
from pathlib import Path


f_performance = dir_swmm_sst_models + "_model_performance_year{}.csv".format("*")
f_summaries = dir_time_series + "_event_summary_year{}.csv".format("*")


#%% load performance dataframe
lst_f_perf = glob(f_performance)
lst_dfs_perf = []
for f in lst_f_perf:
    df = pd.read_csv(f)
    lst_dfs_perf.append(df)

df_perf = pd.concat(lst_dfs_perf)

# export performance dataframe
df_perf.to_csv(f_model_perf_summary, index=False)


#%% load event summaries
lst_f_events = glob(f_summaries)
lst_dfs_events = []
for f in lst_f_events:
    df = pd.read_csv(f)
    lst_dfs_events.append(df)

df_events = pd.concat(lst_dfs_events)

df_events.rename(columns=dict(realization_id = "realization"), inplace=True)

# df_events_and_perf = df_perf.merge(df_events, on = ["realization", "year", "storm_id"])

# export eventsormance dataframe
df_events.to_csv(f_events_summary, index=False)







