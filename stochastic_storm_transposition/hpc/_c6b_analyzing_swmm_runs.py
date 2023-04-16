#%% import libraries
import pandas as pd
from __utils import c6b_analyzing_swmm_runs
from glob import glob

dir_swmm_sst_models = c6b_analyzing_swmm_runs()
f_performance = dir_swmm_sst_models + "_model_performance_year{}.csv".format("*")

#%% load performance dataframe
lst_f_perf = glob(f_performance)
lst_dfs_perf = []
for f in lst_f_perf:
    df = pd.read_csv(f)