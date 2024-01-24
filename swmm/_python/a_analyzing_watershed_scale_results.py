#%% import libraries and load directories
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from _inputs import *
from _utils import *

min_flood_cutoff = 1 # cubic meters; less flood vol is not something i'd care about

plt_fldr = fldr_swmm_analysis_plots + "wateshed_scale_analysis/a_"
#%% determine the simulation type of each model run (this can be helpful for further investigation of specific simulations)
# df_sst_events = pd.read_csv(f_sst_event_summaries)
# lst_simtype = []
# for f_inp in df_sst_events.swmm_inp:
#     determine_sim_type = f_inp.split("/")[-1].split("_")[-1].split(".inp")[0]
#     if determine_sim_type == 'freebndry':
#         sim_type = "freeboundary"
#     elif determine_sim_type == 'norain':
#         sim_type = "norain"
#     elif 'strm' in determine_sim_type:
#         sim_type = 'compound'
#     else:
#         print("WARNING: Simulation could not be classified because filename pattern was not as expected.")
#         sim_type = "UNKNOWN"
#     lst_simtype.append(sim_type)

# df_sst_events.insert(4, "sim_type", value=lst_simtype)

# df_sst_events_compound = df_sst_events[df_sst_events.sim_type == "compound"]
#%% investigate watershed scale flood attribution


#%% inspecting locations where there is no flooding in the compound simulation but there
   # is flooding in another (infinite negative interaction)
df_no_cmpnd_flding = df_watershed_flood_attribution[df_watershed_flood_attribution.flooding_cubic_meters_compound==0]

print("There are a total of {} events with flooding of at least {} cubic meter".format(
    len(df_watershed_flood_attribution), min_flood_cutoff
))

# looking at locations that have rain only flooding AND surge only flooding, but no compound flooding
idx_no_cmpond_rain_and_surge_flding = (np.isinf(df_no_cmpnd_flding.frac_surge) & np.isinf(df_no_cmpnd_flding.frac_rain))
df_no_cmpond_rain_and_surge_flding = df_no_cmpnd_flding[idx_no_cmpond_rain_and_surge_flding]
n_no_cmpond_rain_and_surge_flding = len(df_no_cmpond_rain_and_surge_flding)
print("There are {} events that exhibit zero flooding in the compound event \n but have non-zero flooding in BOTH the surge-only and rain-only simulations".format(n_no_cmpond_rain_and_surge_flding))

# looking at locations that have rain only flooding but no compound flooding 
idx_no_cmpond_rain_flding = (np.isinf(df_no_cmpnd_flding.frac_rain) & (df_no_cmpnd_flding.flooding_cubic_meters_surgeonly == 0))
df_no_cmpond_rain_flding = df_no_cmpnd_flding[idx_no_cmpond_rain_flding]
n_no_cmpond_rain_flding = len(df_no_cmpond_rain_flding)
print("There are {} events that exhibit zero flooding in the compound event \n but have non-zero flooding in the rain-only simulations".format(n_no_cmpond_rain_flding))

# looking at locations that have surge only flooding but no compound flooding 
idx_no_cmpond_surge_flding = (np.isinf(df_no_cmpnd_flding.frac_surge) & (df_no_cmpnd_flding.flooding_cubic_meters_rainonly == 0))
df_no_cmpond_surge_flding = df_no_cmpnd_flding[idx_no_cmpond_surge_flding]
n_no_cmpond_surge_flding = len(df_no_cmpond_surge_flding)
print("There are {} events that exhibit zero flooding in the compound event \n but have non-zero flooding in the surge-only simulations".format(n_no_cmpond_surge_flding))

# inspecting locations where there is a finite negative interaction
idx_finite_negative_interaction = (np.isfinite(df_watershed_flood_attribution.frac_interaction) & (df_watershed_flood_attribution.frac_interaction<0))

df_finite_negative_interaction = df_watershed_flood_attribution[idx_finite_negative_interaction]

print('There are {} events with a negative interaction in the compound event.'.format(len(df_finite_negative_interaction)))

print("This represents {}% of the observations with >{} cubic meters of flooding in at least one of the 3 simulations.".format(
    round(len(df_finite_negative_interaction)/len(df_watershed_flood_attribution),2)*100, min_flood_cutoff
))

print("This does not include the additional {}% of observations that have zero flooding\n\
      in the compound event and non-zero flooding in the surge-only or rain-only events.".format(
          round(len(df_no_cmpnd_flding)/len(df_watershed_flood_attribution),2)*100
      ))


# df_watershed_flood_attribution_neg_inter.loc[:,"flooding_cubic_meters_interaction"] = s_interaction.copy()
#%% investigating events with a non-negative interaction
df_watershed_flood_attribution_nonneg_inter = df_watershed_flood_attribution[df_watershed_flood_attribution.frac_interaction>=0]

size_multiplier = 1.5

# histograms of attribution
fig, axes = plt.subplots(nrows=1, ncols = 3, dpi = 300, figsize = (6*size_multiplier,2.5*size_multiplier),
                         sharex = True, sharey=True)

s_frac_rain = df_watershed_flood_attribution_nonneg_inter.frac_rain
s_frac_surge = df_watershed_flood_attribution_nonneg_inter.frac_surge
s_frac_inter = df_watershed_flood_attribution_nonneg_inter.frac_interaction

s_frac_rain.hist(ax = axes[0])
axes[0].set_xlabel("Fraction Rainfall")
s_frac_surge.hist(ax = axes[1])
axes[1].set_xlabel("Fraction Surge")
s_frac_inter.hist(ax = axes[2])
axes[2].set_xlabel("Fraction Interaction")

fig.suptitle("Events with non-negative interaction (n$_{non-negative}$ = " +
              str(len(df_watershed_flood_attribution_nonneg_inter)) +
                "; n$_{total}$ = " + str(len(df_watershed_flood_attribution)) +
                  ")")

plt.tight_layout()

#%% histograms of volume
surge_quantile_cutoff = 0.98
cutoff_outliers = False
uselog = True
scalar_shift = 0.00001 # for log calcs

nrows=2
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols = ncols, dpi = 300, figsize = (6*size_multiplier,4*size_multiplier),
                         sharex = True, sharey=True)

s_rain_vol = df_watershed_flood_attribution_nonneg_inter.flooding_cubic_meters_rainonly
s_surge_vol = df_watershed_flood_attribution_nonneg_inter.flooding_cubic_meters_surgeonly
s_compound_vol = df_watershed_flood_attribution_nonneg_inter.flooding_cubic_meters_compound
s_inter_vol = df_watershed_flood_attribution_nonneg_inter.flooding_cubic_meters_compound - (s_rain_vol + s_surge_vol)
s_inter_vol.name = "flooding_cubic_meters_interaction"

df_volumes = pd.concat([s_rain_vol,s_surge_vol,s_compound_vol,s_inter_vol], axis = 1)

if uselog:
    df_volumes = np.log10(df_volumes + scalar_shift)

if cutoff_outliers:
    # remove storm surge outliers
    s_surge_vol = s_surge_vol[s_surge_vol < s_surge_vol.quantile(surge_quantile_cutoff)]
    idx_no_outliers = s_surge_vol.index
    df_volumes = df_volumes.loc[idx_no_outliers,:]

# define bin edges
bin_edges = np.linspace(df_volumes.min().min(), df_volumes.max().max(),
                        num=20)
ax_rowid = 0
ax_colid = -1
for colname in df_volumes:
    ax_colid+=1
    df_volumes[colname].hist(ax = axes[ax_rowid,ax_colid], bins = bin_edges)
    axes[ax_rowid,ax_colid].set_xlabel("log_10({})".format(colname))
    if ax_colid+1 == ncols:
        ax_colid = -1
        ax_rowid += 1



fig.suptitle("Events with non-negative interaction (n$_{non-negative}$ = " +
              str(len(df_watershed_flood_attribution_nonneg_inter)) +
                "; n$_{total}$ = " + str(len(df_watershed_flood_attribution)) +
                  ")")


plt.tight_layout()

#%% investigating events with a negative interaction
df_watershed_flood_attribution_neg_inter = df_watershed_flood_attribution[df_watershed_flood_attribution.frac_interaction<0]
print("There are {} events with a negative interaction.".format(len(df_watershed_flood_attribution_neg_inter)))

s_frac_rain_finite = df_watershed_flood_attribution_neg_inter.frac_rain[np.isfinite(df_watershed_flood_attribution_neg_inter.frac_rain)]
s_frac_surge_finite = df_watershed_flood_attribution_neg_inter.frac_surge[np.isfinite(df_watershed_flood_attribution_neg_inter.frac_surge)]
s_frac_inter_finite = df_watershed_flood_attribution_neg_inter.frac_interaction[np.isfinite(df_watershed_flood_attribution_neg_inter.frac_interaction)]

size_multiplier = 1.5
#%% negative interactions, histogram of volumes 
surge_quantile_cutoff = 0.98
cutoff_outliers = False
uselog = True
scalar_shift = 0.00001 # for log calcs

nrows=2
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols = ncols, dpi = 300, figsize = (6*size_multiplier,4*size_multiplier),
                         sharex = True, sharey=True)

df_volumes = df_watershed_flood_attribution_neg_inter.loc[:,["flooding_cubic_meters_rainonly","flooding_cubic_meters_surgeonly", "flooding_cubic_meters_compound"]]

s_interaction = df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_compound-(df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_rainonly + df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_surgeonly)
s_interaction = s_interaction*-1
s_interaction.name = "flooding_negative_cubic_meters_interaction"

df_volumes = pd.concat([df_volumes,s_interaction], axis = 1)

if uselog:
    df_volumes = np.log10(df_volumes + scalar_shift)

if cutoff_outliers:
    # remove storm surge outliers
    s_surge_vol = s_surge_vol[s_surge_vol < s_surge_vol.quantile(surge_quantile_cutoff)]
    idx_no_outliers = s_surge_vol.index
    df_volumes = df_volumes.loc[idx_no_outliers,:]

# left off here, trying to figure out same bin edges
bin_edges = np.linspace(df_volumes.min().min(), df_volumes.max().max(),
                        num=20)
ax_rowid = 0
ax_colid = -1
for colname in df_volumes:
    ax_colid+=1
    df_volumes[colname].hist(ax = axes[ax_rowid,ax_colid], bins = bin_edges)
    axes[ax_rowid,ax_colid].set_xlabel("log_10({})".format(colname))
    if ax_colid+1 == ncols:
        ax_colid = -1
        ax_rowid += 1



fig.suptitle("Events with negative interaction (n$_{negative}$ = " +
              str(len(df_watershed_flood_attribution_neg_inter)) +
                "; n$_{total}$ = " + str(len(df_watershed_flood_attribution)) +
                  ")")


plt.tight_layout()