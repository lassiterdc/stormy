#%% load libraries, directories, and user-defined inputs
from _inputs import def_work
import pandas as pd

f_in_water_level = def_work()

#%%
df_wlevel = pd.read_csv(f_in_water_level, index_col=0, parse_dates=True)
print(df_wlevel)