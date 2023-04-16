#%% import libraries
import pandas as pd
from __utils import c6b_analyzing_swmm_runs
from glob import glob
from pathlib import Path

dir_swmm_sst_models, f_model_perf_summary = c6b_analyzing_swmm_runs()
f_performance = dir_swmm_sst_models + "_model_performance_year{}.csv".format("*")



#%% load performance dataframe
lst_f_perf = glob(f_performance)
lst_dfs_perf = []
for f in lst_f_perf:
    df = pd.read_csv(f)
    lst_dfs_perf.append(df)

df_perf = pd.concat(lst_dfs_perf)

# export
df_perf.to_csv(f_model_perf_summary)
#%% analyze
df_perf_failed = df_perf[df_perf.run_completed==False]
df_perf_succeeded = df_perf[df_perf.run_completed==True]

#%% export consolidated summary