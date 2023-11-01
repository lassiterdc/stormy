#%% import python libraries
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats.mstats import meppf
from scipy.stats import rankdata
#%% load inputs
from _inputs import *
from _inputs import return_attribution_data


df_variable_nodes, ds_flood_attribution, ds_sst_compound, ds_sst_freebndry, ds_events, gdf_node_flding, gdf_subs, gdf_nodes, df_comparison = return_attribution_data()

#%% compute return periods of storm surge and event rain depths; classify each event
ds_events

tot_yrs = len(ds_events.realization) * len(ds_events.year)

for interval in sst_recurrence_intervals:
    pass

# convert quantiles to return periods in years 
tot_events = len(ds_events.realization.values) * len(ds_events.year.values) * len(ds_events.storm_id.values)
tot_years = len(ds_events.realization.values) * len(ds_events.year.values)
events_per_year = tot_events / tot_years
#%% compute empircal probabilities of rain depths
def calcluate_recurrence_interval(da):
    varname = da.name
    df = da.to_dataframe()
    df_processing = df.copy()
    df_processing = df_processing.fillna(0) # fill in na with zeroes
    df_processing['ranks'] = rankdata(df_processing[varname], method = 'min')
    df_processing['emp_F_weibull'] = df_processing['ranks'] / (tot_events+1)
    df_processing["annual_per_year_occurence"] =  (1-df_processing["emp_F_weibull"]) * events_per_year
    df_processing["recurrence_interval"] = 1 / df_processing["annual_per_year_occurence"]
    df_processing = df_processing.sort_values(varname)
    varname_out = "recurrence_int_{}".format(varname)
    df[varname_out] = df_processing["recurrence_interval"]
    ds_out = df.to_xarray()
    return ds_out

ds_rain_depth_prob = calcluate_recurrence_interval(ds_events.depth_mm)
ds_surge_prob = calcluate_recurrence_interval(ds_events.max_sim_wlevel)
ds_node_flooding = ds_sst_compound.node_flooding_cubic_meters.groupby("node_id").map(calcluate_recurrence_interval)
lst_ds = [ds_rain_depth_prob, ds_surge_prob, ds_node_flooding]
df_recurrence_ints = xr.merge(lst_ds).to_dataframe()
#%% hexagonal binning plot for single node
import matplotlib.pyplot as plt
df_reset_idx = df_recurrence_ints.reset_index()
df_nozeros = df_reset_idx[df_reset_idx.node_flooding_cubic_meters > 0]
df = df_reset_idx[df_reset_idx.node_id == "UN68"]

x = df.recurrence_int_depth_mm
y = df.recurrence_int_max_sim_wlevel
xlim = x.min(), 101
ylim = y.min(), 101

fig, ax0 = plt.subplots(ncols=1, sharey=True, figsize=(9, 4))

hb = ax0.hexbin(x, y, gridsize=50, cmap='inferno')
ax0.set(xlim=xlim, ylim=ylim)
ax0.set_title("Hexagon binning")
cb = fig.colorbar(hb, ax=ax0, label='counts')

plt.show()