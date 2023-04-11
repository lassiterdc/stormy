#%% load libraries and directories
import pandas as pd

from _inputs import def_inputs_for_d

f_mrms_event_summaries, f_mrms_event_timeseries, f_water_level_storm_surge = def_inputs_for_d()

#%% load data
df_mrms_event_summaries = pd.read_csv(f_mrms_event_summaries, parse_dates=True)
df_mrms_event_tseries = pd.read_csv(f_mrms_event_timeseries, parse_dates=True)
df_water_levels = pd.read_csv(f_water_level_storm_surge, parse_dates=True)