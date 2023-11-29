#%% load libraries and directories
import pandas as pd
import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt 
import xarray as xr
from pathlib import Path
from datetime import datetime
import warnings
from __utils import *

script_start_time = datetime.now()
#%% load data
df_obs_cmpnd_summary = pd.read_csv(f_observed_cmpnd_event_summaries)
df_sim_cmpnd_summary = pd.read_csv(f_simulated_cmpnd_event_summaries)

vars_k = ["depth_mm", "max_mm_per_hour", "max_surge_ft"]
df_vars_stormclass = df_compound_summary.loc[:, vars_k]

df_vars_stormclass_scaler = preprocessing.StandardScaler().fit(df_vars_stormclass)
df_vars_stormclass_scaled = df_vars_stormclass_scaler.transform(df_vars_stormclass)

if plot_weather_gen_stuff:
    inertias = []
    ks_to_try = 20
    for i in range(1,ks_to_try):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(df_vars_stormclass_scaled)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1,ks_to_try), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig(plt_fldr_weather_gen + "kmeans.png")
    plt.clf()
    # plt.show()


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df_vars_stormclass_scaled)
#%%
# plt.scatter(x = df_vars_all.max_surge_ft, y= df_vars_all.duration_hr, c=kmeans.labels_)
# plt.show()
if plot_weather_gen_stuff:
    for v_sim in vars_sim:
        fig, axes = plt.subplots(2, 2, figsize = (10,8))
        row = 0
        col = 0
        for v_cond in vars_cond:
            axes[row, col].scatter(df_vars_all[v_sim], df_vars_all[v_cond], c=kmeans.labels_)
            axes[row, col].set_xlabel(v_sim)
            axes[row, col].set_ylabel(v_cond)
            col += 1
            if col > 1:
                row += 1
                col = 0
        plt.savefig(plt_fldr_weather_gen + "{}_sims.png".format(v_sim))
        plt.clf()

#%% predict k label of synthetic data
# df_synth_hydro_cond_scaled = df_vars_stormclass_scaler.transform(df_synth_hydro_cond.loc[:, vars_k])

# pred_ks = kmeans.predict(df_synth_hydro_cond_scaled)
obs_ks = kmeans.labels_

# define function to randomly select an event from the same category
def get_storm_to_rescale(pred_k):
    ind_obs_same_class = np.where(obs_ks==pred_k)[0]
    obs_event_index = np.random.choice(ind_obs_same_class)
    obs_event_id = df_compound_summary.event_id[obs_event_index] # necessary because the event IDs are 1-indexed
    return obs_event_id
#%% create synthetic time series
# df_water_levels
# prelim calcs
wlevel_tdiff = (pd.Series(df_water_rain_tseries.index).diff().dropna().mode())[0]

wlevel_freq = pd.tseries.frequencies.to_offset(wlevel_tdiff).freqstr

max_allowable_duration = (pd.Timedelta(3, "days") + 2 * pd.Timedelta(time_buffer, "hours"))

# run loop
min_obs_wlevel = df_water_levels.water_level.min()
max_obs_wlevel = df_water_levels.water_level.max()

i = -1
lst_s_wlevel_tseries = []
lst_keys = []
# source_event_id = []
min_sim_wlevels = []
max_sim_wlevels = []
lst_event_starts = []
lst_event_ends = []
lst_event_durations = []
lst_peak_surge_tsteps = []
lst_event_ids = []
lst_successful_sim = []
lst_ds = []
count = 0
lag_reset = False
for i, cond in df_cond.iterrows():
    count += 1
    # i += 1
    attempts = 0
    # absurd_simulation = True
    reasonable_sample = False
    generate_new_sim = True
    while reasonable_sample == False:
        success = True
        if attempts >= n_attempts:
            success = False
            break
            sys.exit("SCRIPT FAILED FOR YEAR {}: FAILED AFTER {} ATTEMPTS TO GENERATE A SYNTHETIC WATER LEVEL TIME SERIES FOR {}".format(yr, attempts, s_sim_event_summary))
        attempts += 1
        try:           
            if generate_new_sim == True:
                df_new_sim = gen_conditioned_samples(cop_hydro, cond.to_frame().T.reset_index(drop=True), n_samples=1)
                s_sim_event_summary = df_new_sim.loc[0,:]
                generate_new_sim = False
            s_sim_event_summary_scaled = df_vars_stormclass_scaler.transform(pd.DataFrame(s_sim_event_summary.loc[vars_k]).T)
            pred_k = kmeans.predict(s_sim_event_summary_scaled)
            obs_event_id = get_storm_to_rescale(pred_k)
            # source_event_id.append(obs_event_id)
            df_obs_event_tseries = df_water_rain_tseries[df_water_rain_tseries.event_id == obs_event_id]
            df_obs_event_summary = df_compound_summary.loc[df_compound_summary.event_id == obs_event_id, vars_all]
            # print("obs_event_id")
            # print(obs_event_id)
            # print("df_compound_summary")
            # print(df_compound_summary)
            # print("df_obs_event_summary")
            # print(df_obs_event_summary)
            # compute timestep of peak storm surge
            sim_tstep_max_int = df_sst_storm_summaries.tstep_of_max_intensity[i]
            sim_tstep_max_surge = sim_tstep_max_int + pd.Timedelta(s_sim_event_summary["surge_peak_after_rain_peak_min"], "minutes")
            # round to the closest timestep
            sim_tstep_max_surge = sim_tstep_max_surge.round(wlevel_freq)
            # start time is the minimum of the start date or the timestep of max surge minus the time buffer
            event_starttime = min(pd.to_datetime(start_date), sim_tstep_max_surge)-pd.Timedelta(time_buffer, "hours")
            # end time is the max of the last rainfall or the peak surge tstep plus the time buffer
            sim_tstep_lastrain = df_sst_storm_summaries.last_timestep_w_rainfall[i]
            event_endtime = max(sim_tstep_lastrain, sim_tstep_max_surge)+pd.Timedelta(time_buffer, "hours")
            # duration is start minus end
            duration = event_endtime - event_starttime
            # if duration is greater than allowable, generate new sim
            if duration > max_allowable_duration:
                generate_new_sim = True
                continue
            sim_wlevel_times = pd.date_range(event_starttime, event_endtime, freq=wlevel_freq)
            time_to_peak_surge = sim_tstep_max_surge - min(sim_wlevel_times)
            # extract observed surge data
            obs_tstep_max_surge = df_obs_event_tseries.surge_ft.idxmax()
            obs_start_time = obs_tstep_max_surge - time_to_peak_surge
            obs_end_time = obs_start_time + duration
            obs_surge_times = pd.date_range(obs_start_time, obs_end_time, freq=wlevel_freq)
            obs_surges = df_water_levels.surge_ft.loc[obs_surge_times]
            # obs_peak_tstep = obs_surges.index[0]+time_to_peak_surge
            obs_peak = obs_surges[obs_tstep_max_surge]
            # add predicted water level with a random shift of plus or minus 12 hours
            tide_shift = pd.Timedelta(np.random.choice(np.arange(-12, 12+1)), "hr")
            s_tides_times = pd.date_range(obs_start_time+tide_shift, obs_end_time+tide_shift, freq=wlevel_freq)
            s_tides = df_water_levels.predicted_wl.loc[s_tides_times]
            # set index to align with timesteps of sim_wlevel_times to add later
            s_tides.index = sim_wlevel_times
            # s_obs_wlevel = 
            # rescaling
            ## compute multiplier
            obs_frac_of_max_tseries = obs_surges / obs_peak # unit surge as fraction of the maximum
            # rescaling
            s_sim_surge_tseries = (s_sim_event_summary["max_surge_ft"] * obs_frac_of_max_tseries).reset_index(drop=True)
            s_sim_surge_tseries.index = sim_wlevel_times
            # adding tide
            s_sim_wlevel = s_sim_surge_tseries + s_tides
            s_sim_wlevel.name = "water_level_ft"
            min_sim_wlevel = s_sim_wlevel.min()
            max_sim_wlevel = s_sim_wlevel.max()
            # if the the simulated water levels exceed user defined thresholds, generate new sim
            if (max_sim_wlevel >= (1+wlevel_threshold)*max_obs_wlevel) or (min_sim_wlevel <= (1+wlevel_threshold)*min_obs_wlevel):
                # if these are exceeded and there have been at least some number of attempts to select and rescale a historical event
                if ((attempts % resampling_inteval) == 0) and (attempts > 1):
                    generate_new_sim = True
                continue
            reasonable_sample = True
        except:
            # if there is an error, generate a new sim
            if ((attempts % resampling_inteval) == 0) and (attempts > 1):
                print("After {} unsuccesful attempts to generate a reasonable water level time series, the peak surge and time lag were resampled using the copula.".format(attempts))
                generate_new_sim = True
            continue
    if success: # if successful, append lists of simulated values
        lst_event_ids.append(obs_event_id)
        min_sim_wlevels.append(min_sim_wlevel)
        max_sim_wlevels.append(max_sim_wlevel)
        lst_event_starts.append(event_starttime)
        lst_event_ends.append(event_endtime)
        lst_event_durations.append(duration)
        lst_peak_surge_tsteps.append(sim_tstep_max_surge)
    else: # if not successful, append with NA's
        lst_event_ids.append(np.nan)
        min_sim_wlevels.append(np.nan)
        max_sim_wlevels.append(np.nan)
        lst_event_starts.append(np.nan)
        lst_event_ends.append(np.nan)
        lst_event_durations.append(np.nan)
        lst_peak_surge_tsteps.append(np.nan)
    lst_successful_sim.append(success)
    # lst_obs_peak.append(obs_peak)
    # lst_obs_min.append(obs_peak)
    # writing to a file
    ## id the realization, and storm
    rz, yr, strm = df_sst_storm_summaries.rz_yr_strm.loc[i].split("_")
    f_out = dir_time_series + "weather_realization{}/year{}/_waterlevel_rz{}_yr{}_strm{}.dat".format(rz, yr, rz, yr, strm)
    # create data frame with proper formatting to be read in SWMM
    df = pd.DataFrame(dict(date = s_sim_wlevel.index.strftime('%m/%d/%Y'),
                    time = s_sim_wlevel.index.time,
                    water_level = s_sim_wlevel.values))
    Path(f_out).parent.mkdir(parents=True, exist_ok=True)
    with open(f_out, "w+") as file:
        file.write(";;synthetic water level\n")
        file.write(";;Water Level (ft)\n")
    df.to_csv(f_out, sep = '\t', index = False, header = False, mode="a")
    # export to a netcdf
    df["realization"] = int(rz)
    df["year"] = int(yr)
    df["storm_id"] = int(strm)
    df["datetime"] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    # print(df)
    # print("######################################")
    df = df.drop(["date", "time"], axis = 1)
    # df = df.reset_index(names = "tstep")
    df = df.set_index(["realization", "year","storm_id", "datetime"])
    # print(df)
    ds = df.to_xarray()
    lst_ds.append(ds)
    # ds = ds.assign_coords(["realization", "year","storm_id", "datetime"])

# combine netcdfs into one and export
# ds_combined = xr.combine_nested(lst_ds)
ds_combined = xr.combine_by_coords(lst_ds)
ds_combined_loaded = ds.load()
Path(dir_waterlevel_ncs_scratch).mkdir(parents=True, exist_ok=True)
ds_combined_loaded.to_netcdf(dir_waterlevel_ncs_scratch + "wlevel_yr{}.nc".format(rz, yr, strm))
#%% export event summaries
df_idx = df_sst_storm_summaries.rz_yr_strm.str.split("_", expand=True)
df_idx.columns = ["realization", "year", "storm_id"]

df_simulated_event_summaries = pd.DataFrame(dict(success = lst_successful_sim, min_sim_wlevel = min_sim_wlevels,max_sim_wlevel = max_sim_wlevels, obs_event_id_for_rescaling = lst_event_ids,
                                                 event_start = lst_event_starts, event_end = lst_event_ends,
                                                 event_duration_hr = lst_event_durations, tstep_peak_surge = lst_peak_surge_tsteps))

df_sim_summary = df_idx.join(df_simulated_event_summaries, how="outer")
df_sim_summary = df_sim_summary.join(df_sst_storm_summaries, how="outer")

df_sim_summary = df_sim_summary.join(df_synth_hydro_cond.surge_peak_after_rain_peak_min, how = "outer")

df_sim_summary = df_sim_summary.rename(columns=dict(tstep_of_max_intensity = "tstep_max_rain_intensity",
                                                    duration_hr = "rainfall_duration_hr"))

df_sim_summary.drop(columns=["rz_yr_strm"], inplace=True)

f_summary = dir_time_series + "_event_summary_year{}.csv".format(yr)
df_sim_summary.to_csv(f_summary, index=False)

# print(f_summary)

#%% report run times

time_script_min = round((datetime.now() - script_start_time).seconds / 60, 1)
print("Wrote {} time series files for each storm realizations for year {}. Script runtime: {} (min)".format(count, yr, time_script_min))

# sys.exit("THE SCRIPT RAN THROUGH TO COMPLETION")