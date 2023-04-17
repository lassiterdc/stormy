#%% import libraries
import pandas as pd
from __utils import c6b_analyzing_swmm_runs, parse_inp
from glob import glob
from pathlib import Path


dir_swmm_sst_models, f_model_perf_summary, dir_time_series = c6b_analyzing_swmm_runs()
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



#%% load event summaries
lst_f_events = glob(f_summaries)
lst_dfs_events = []
for f in lst_f_events:
    df = pd.read_csv(f)
    lst_dfs_events.append(df)

df_events = pd.concat(lst_dfs_events)

# join model performance summary table with the event summary table


# export eventsormance dataframe
df_perf.to_csv(f_model_perf_summary)







