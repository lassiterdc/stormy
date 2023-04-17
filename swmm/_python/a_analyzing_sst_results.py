#%% import libraries and load directories
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from _inputs import a_analyzing_sst_results

f_sst_results, f_model_perf_summary, f_sst_event_summaries, f_sst_annual_max_volumes = a_analyzing_sst_results()

#%% load data
ds_sst = xr.open_dataset(f_sst_results)
df_model_perf = pd.read_csv(f_model_perf_summary)
df_sst_events = pd.read_csv(f_sst_event_summaries)

#%% find the 1 and 100 year quantiles for each node
# first find the annual maximum flooding at each node
# ds_sst.groupby("year")

#%% testing
years = ds_sst.year.values
# years = years[0:20]
#%% testing

lst_ds_annual_max_by_node = []
lst_ds_strm_with_most_tot_flding = []

lst_total_flood_in_largest_event = []
lst_rz = []
lst_yr = []
lst_strm_id = []


for rz in ds_sst.realization.values:
    for yr in tqdm(years):
        ds_sub = ds_sst.sel(dict(realization=rz,year=yr))
        # ds_max = ds_sub.max(dim="storm_id")
        # lst_ds_annual_max_by_node.append(ds_max)
        lst_tot_flooding = []
        for strm in ds_sub.storm_id.values:
            ds_strm = ds_sub.sel(dict(storm_id = strm))
            tot_fld_vol = float(ds_strm.node_flooding.sum().values)
            lst_tot_flooding.append(tot_fld_vol)
        idx_strm_w_most_flding = pd.Series(lst_tot_flooding).idxmax()
        lst_total_flood_in_largest_event.append(max(lst_tot_flooding))
        lst_rz.append(rz)
        lst_yr.append(yr)
        lst_strm_id.append(ds_sub.storm_id.values[idx_strm_w_most_flding])
        # ds_strm_most_flding = ds_sub.isel(dict(storm_id = idx_strm_w_most_flding))
        # lst_ds_strm_with_most_tot_flding.append(ds_strm_most_flding)

#%% analysis
# ds_annual_max_by_node = xr.concat(lst_ds_annual_max_by_node, dim="year")

# ds_biggest_strm_by_node = xr.concat(lst_ds_strm_with_most_tot_flding, dim="year")

# s_annual_tot_fld = pd.Series(lst_total_flood_in_largest_event, name="maximum_event_volume_per_year")

df_annual_tot_fld = pd.DataFrame(dict(realization = lst_rz,
                                      year = lst_yr,
                                      storm_id = lst_strm_id,
                                      total_event_flooding_MG = lst_total_flood_in_largest_event))


#%% compare with event summaries
df_biggest_annual_strms = df_annual_tot_fld.merge(df_sst_events, on=["realization", "year", "storm_id"])

df_biggest_annual_strms.to_csv(f_sst_annual_max_volumes, index=False)

#%%
print("yay it completed succesfully.")