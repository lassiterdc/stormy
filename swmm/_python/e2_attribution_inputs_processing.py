#%% load libraries and directories
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from tqdm import tqdm
import imageio
from glob import glob
from _inputs import *
from _utils import *
import os


#%% computing flood attribution
ds_sum_independent = ds_sst_rainonly + ds_sst_surgeonly
ds_frac_rain = ds_sst_rainonly/ds_sst_compound
ds_frac_surge = ds_sst_surgeonly/ds_sst_compound
ds_frac_interaction = (ds_sst_compound - ds_sum_independent)/ds_sst_compound


# convert to dataframes and join together
df_sst_rainonly = ds_sst_rainonly.to_dataframe().drop(columns = ["freeboundary", "norain"]).rename(columns = dict(node_flooding_cubic_meters = "node_flooding_cubic_meters_rainonly"))
df_sst_surgeonly = ds_sst_surgeonly.to_dataframe().drop(columns = ["freeboundary", "norain"]).rename(columns = dict(node_flooding_cubic_meters = "node_flooding_cubic_meters_surgeonly"))
df_sst_compound = ds_sst_compound.to_dataframe().drop(columns = ["freeboundary", "norain"]).rename(columns = dict(node_flooding_cubic_meters = "node_flooding_cubic_meters_compound"))

df_node_flooding = pd.concat([df_sst_rainonly,df_sst_surgeonly,df_sst_compound], axis = 1)
# subset rows where there is flooding in at least one location
min_flood_cutoff = 1
df_node_flooding = df_node_flooding[(df_node_flooding>min_flood_cutoff).any(axis=1)]


df_node_flooding["sum_rainonly_surgeonly"] = df_node_flooding.node_flooding_cubic_meters_rainonly + df_node_flooding.node_flooding_cubic_meters_surgeonly

df_node_flooding["frac_rain"] = df_node_flooding.node_flooding_cubic_meters_rainonly/df_node_flooding.node_flooding_cubic_meters_compound
df_node_flooding["frac_surge"] = df_node_flooding.node_flooding_cubic_meters_surgeonly/df_node_flooding.node_flooding_cubic_meters_compound
df_node_flooding["frac_interaction"] = (df_node_flooding.node_flooding_cubic_meters_compound - df_node_flooding.sum_rainonly_surgeonly)/df_node_flooding.node_flooding_cubic_meters_compound

#%% inspecting locations where there is no flooding in the compound simulation but there
   # is flooding in another (infinite negative interaction)
df_no_cmpnd_flding = df_node_flooding[df_node_flooding.node_flooding_cubic_meters_compound==0]

print("There are a total of {} node/event combos with flooding of at least {} cubic meter".format(
    len(df_node_flooding), min_flood_cutoff
))

# looking at locations that have rain only flooding AND surge only flooding, but no compound flooding
idx_no_cmpond_rain_and_surge_flding = (np.isinf(df_no_cmpnd_flding.frac_surge) & np.isinf(df_no_cmpnd_flding.frac_rain))
df_no_cmpond_rain_and_surge_flding = df_no_cmpnd_flding[idx_no_cmpond_rain_and_surge_flding]
n_no_cmpond_rain_and_surge_flding = len(df_no_cmpond_rain_and_surge_flding)
print("There are {} node/event combos that exhibit zero flooding in the compound event \n but have non-zero flooding in BOTH the surge-only and rain-only simulations".format(n_no_cmpond_rain_and_surge_flding))


# looking at locations that have rain only flooding but no compound flooding 
idx_no_cmpond_rain_flding = (np.isinf(df_no_cmpnd_flding.frac_rain) & (df_no_cmpnd_flding.node_flooding_cubic_meters_surgeonly == 0))
df_no_cmpond_rain_flding = df_no_cmpnd_flding[idx_no_cmpond_rain_flding]
n_no_cmpond_rain_flding = len(df_no_cmpond_rain_flding)
print("There are {} node/event combos that exhibit zero flooding in the compound event \n but have non-zero flooding in the rain-only simulations".format(n_no_cmpond_rain_flding))


# looking at locations that have surge only flooding but no compound flooding 
idx_no_cmpond_surge_flding = (np.isinf(df_no_cmpnd_flding.frac_surge) & (df_no_cmpnd_flding.node_flooding_cubic_meters_rainonly == 0))
df_no_cmpond_surge_flding = df_no_cmpnd_flding[idx_no_cmpond_surge_flding]
n_no_cmpond_surge_flding = len(df_no_cmpond_surge_flding)
print("There are {} node/event combos that exhibit zero flooding in the compound event \n but have non-zero flooding in the surge-only simulations".format(n_no_cmpond_surge_flding))

# inspecting locations where there is a finite negative interaction
idx_finite_negative_interaction = (np.isfinite(df_node_flooding.frac_interaction) & (df_node_flooding.frac_interaction<0))

df_finite_negative_interaction = df_node_flooding[idx_finite_negative_interaction]

print('There are {} node/event combos with a negative interaction in the compound event.'.format(len(df_finite_negative_interaction)))

print("This represents {}% of the observations with >{} cubic meters of flooding in at least one of the 3 simulations.".format(
    round(len(df_finite_negative_interaction)/len(df_node_flooding),2)*100, min_flood_cutoff
))

print("This does not include the additional {}% of observations that have zero flooding\n\
      in the compound event and non-zero flooding in the surge-only or rain-only events.".format(
          round(len(df_no_cmpnd_flding)/len(df_node_flooding),2)*100
      ))


#%% qaqc - are there negative interactions?

ds_neg_interactions = ds_sst_compound < ds_sum_independent



ds_sst_compound >= ds_sum_independent
#%%

# ds_flood_attribution = 1 - (ds_sst_freebndry / ds_sst_compound)
# ds_flood_attribution = ds_flood_attribution.rename(dict(node_flooding_cubic_meters = "flood_attribution"))

# ds_flood_attribution["flood_attribution"]  =xr.where(ds_flood_attribution.flood_attribution < 0, 0, ds_flood_attribution.flood_attribution)
#%% end dcl work in progress