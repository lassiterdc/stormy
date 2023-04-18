#%% import libraries and load directories
import pandas as pd

from _inputs import c_comparing_sst_and_design_storms

f_sst_annual_max_volumes, f_design_strms = c_comparing_sst_and_design_storms()

#%% load data
df_sst = pd.read_csv(f_sst_annual_max_volumes)