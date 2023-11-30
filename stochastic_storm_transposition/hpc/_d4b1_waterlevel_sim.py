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
# define start time for simulations
start_datetime = pd.to_datetime(start_date)
#%% load data
df_obs_cmpnd_summary = pd.read_csv(f_observed_cmpnd_event_summaries)
df_sim_cmpnd_summary = pd.read_csv(f_simulated_cmpnd_event_summaries)
df_sst_event_summaries = pd.read_csv(f_sst_event_summaries)
df_water_levels = pd.read_csv(f_waterlevels_same_tstep_as_sst, parse_dates=True, index_col="date_time")
df_sst_event_summaries = df_sst_event_summaries.set_index(["realization","year","storm_id"])

vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]

df_water_rain_tseries = pd.read_csv(f_observed_wlevel_rainfall_tseries, parse_dates=True, index_col="date_time")

# validate that event IDs align in timeseries and compound summary
event_sum_computed_from_tseries = df_water_rain_tseries.loc[:,['event_id', "mrms_mm_per_hour","surge_ft"]].groupby("event_id").max().sort_values("event_id")

event_sum = df_obs_cmpnd_summary.sort_values("event_id").loc[:,["event_id", "max_mm_per_hour", "max_surge_ft"]].set_index("event_id")
event_sum_computed_from_tseries.columns = event_sum.columns

problem = False
for variable, value in (event_sum_computed_from_tseries - event_sum).sum().items():
    if not np.isclose(value, 0):
        problem = True
        print("ERROR: The event IDs in the observed event summary and observed event time series do not match up.")
        print("Variable: {} | Sum of differences across all events: {}".format(variable, value))
if problem:
    print("(event_sum_computed_from_tseries - event_sum)")
    print((event_sum_computed_from_tseries - event_sum))
    print("###################################")
    sys.exit()

# define function to randomly select an event from the same category
def get_storm_to_rescale(sim_k, df_obs_cmpnd_summary = df_obs_cmpnd_summary):
    sampled_event = df_obs_cmpnd_summary[df_obs_cmpnd_summary["kmeans_class"]==sim_k].sample() # select events of same class and sample one of them
    obs_event_id = sampled_event.event_id.values
    return int(obs_event_id)
#%% create synthetic time series
# df_water_levels
# prelim calcs
wlevel_tdiff = (pd.Series(df_water_rain_tseries.index).diff().dropna().mode())[0]

wlevel_freq = pd.tseries.frequencies.to_offset(wlevel_tdiff).freqstr

max_allowable_duration = (pd.Timedelta(3, "days") + 4 * pd.Timedelta(time_buffer, "hours"))

# run loop
min_obs_wlevel = df_water_rain_tseries.water_level.min()
# max_obs_wlevel = df_water_levels.water_level.max()

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
lst_realizations = []
lst_years = []
lst_storm_ids = []
lst_successful_sim = []
lst_ds = []
count = 0
lag_reset = False
for i, sim in df_sim_cmpnd_summary.iterrows():
    count += 1
    # i += 1
    attempts = 0
    rz = int(sim.realization)
    yr = int(sim.year)
    strm = int(sim.storm_id)
    reasonable_timeseries = False
    while reasonable_timeseries == False:
        success = True
        if attempts >= n_attempts:
            success = False
            break
            sys.exit("SCRIPT FAILED FOR YEAR {}: FAILED AFTER {} ATTEMPTS TO GENERATE A SYNTHETIC WATER LEVEL TIME SERIES FOR {}".format(yr, attempts, s_sim_event_summary))
        attempts += 1
        # try:           
        # sample event id for storm surge time series rescaling
        obs_event_id = get_storm_to_rescale(sim.kmeans_class)
        # return observed event time series and event summary for the event
        df_obs_event_tseries = df_water_rain_tseries[df_water_rain_tseries.event_id == obs_event_id]
        df_obs_event_summary = df_obs_cmpnd_summary.loc[df_obs_cmpnd_summary.event_id == obs_event_id, vars_all]
        # pull the sst event summary associated with the simulation to calculate the time of peak surge
        sst_event_summary = df_sst_event_summaries.loc[(rz,yr,strm),:]
        # calculate the datetime of the max rain intensity and max storm surge
        sim_tstep_max_int = sst_event_summary.tstep_of_max_intensity
        sim_datetime_max_int = start_datetime + pd.Timedelta(sim_tstep_max_int*sst_tstep_min, "minutes")
        sim_datetime_max_surge = sim_datetime_max_int + pd.Timedelta(sim["surge_peak_after_rain_peak_min"], "minutes")
        # round storm surge peak time to the closest timestep
        sim_tstep_max_surge = sim_datetime_max_surge.round(wlevel_freq)
        # start time is the minimum of the start date or the timestep of max surge minus the time buffer
        event_starttime = min(pd.to_datetime(start_date), sim_tstep_max_surge)-pd.Timedelta(time_buffer, "hours")
        # end time is the max of the last rainfall or the peak surge tstep plus the time buffer
        sim_datetime_of_last_rainfall = start_datetime + pd.Timedelta(sst_event_summary.duration_n_tsteps*sst_tstep_min, "minutes")
        event_endtime = max(sim_datetime_of_last_rainfall, sim_tstep_max_surge)+pd.Timedelta(time_buffer, "hours")
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
        tide_shift = pd.Timedelta(np.random.choice(np.arange(-12*60, 12*60+sst_tstep_min, sst_tstep_min)), "min") # randomly shift by a 5-minute discretized value between -12 and + 12 hours 
        s_tides_times = pd.date_range(obs_start_time+tide_shift, obs_end_time+tide_shift, freq=wlevel_freq)
        s_tides = df_water_levels.predicted_wl.loc[s_tides_times]
        # set index to align with timesteps of sim_wlevel_times to add later
        s_tides.index = sim_wlevel_times
        # rescaling
        ## compute multiplier
        obs_frac_of_max_tseries = obs_surges / obs_peak # unit surge as fraction of the maximum
        # rescaling
        s_sim_surge_tseries = (sim["max_surge_ft"] * obs_frac_of_max_tseries).reset_index(drop=True)
        s_sim_surge_tseries.index = sim_wlevel_times
        # adding tide
        s_sim_wlevel = s_sim_surge_tseries + s_tides
        s_sim_wlevel.name = "water_level_ft"
        min_sim_wlevel = s_sim_wlevel.min()
        max_sim_wlevel = s_sim_wlevel.max()
        # if the the simulated water levels is below user defined thresholds, try again
        if (min_sim_wlevel <= (1+wlevel_threshold)*min_obs_wlevel):
            # if these are exceeded and there have been at least some number of attempts to select and rescale a historical event
            if ((attempts % resampling_inteval) == 0) and (attempts > 1):
                generate_new_sim = True
            continue
        reasonable_sample = True
        # except:
        #     # if there is an error, generate a new sim
        #     if ((attempts % resampling_inteval) == 0) and (attempts > 1):
        #         # print("After {} unsuccesful attempts to generate a reasonable water level time series, the peak surge and time lag were resampled using the copula.".format(attempts))
        #         generate_new_sim = True
        #     continue
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
    lst_realizations.append(rz)
    lst_years.append(yr)
    lst_storm_ids.append(strm)
    lst_successful_sim.append(success)
    # writing to a file
    ## id the realization, and storm
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

#%% export event summaries
# df_idx = df_sst_storm_summaries.rz_yr_strm.str.split("_", expand=True)
# df_idx.columns = ["realization", "year", "storm_id"]

df_simulated_event_summaries = pd.DataFrame(dict(realization = lst_realizations, year = lst_years, storm_id = lst_storm_ids, 
                                                 success = lst_successful_sim, min_sim_wlevel = min_sim_wlevels,max_sim_wlevel = max_sim_wlevels, obs_event_id_for_rescaling = lst_event_ids,
                                                 event_start = lst_event_starts, event_end = lst_event_ends,
                                                 event_duration_hr = lst_event_durations, tstep_peak_surge = lst_peak_surge_tsteps))

df_simulated_event_summaries = df_simulated_event_summaries.set_index(["realization", "year", "storm_id"])
df_sim_cmpnd_summary = df_sim_cmpnd_summary.set_index(["realization", "year", "storm_id"])

df_sim_summary = df_sim_cmpnd_summary.join(df_simulated_event_summaries, how="left")

df_sim_summary = df_sim_summary.rename(columns=dict(tstep_of_max_intensity = "tstep_max_rain_intensity",
                                                    duration_hr = "rainfall_duration_hr"))

f_summary = dir_time_series + "_event_summary_year.csv"
df_sim_summary.to_csv(f_summary)

#%% report run times
end_time = datetime.now()
time_script_min = round((end_time - script_start_time).seconds / 60, 1)
print("Wrote {} water level time series files for each simulated storm. Script runtime: {} (min)".format(count, time_script_min))


# combine netcdfs into one and export
# ds_combined = xr.combine_nested(lst_ds)
ds_combined = xr.combine_by_coords(lst_ds)
ds_combined_loaded = ds.load()
# Path(dir_waterlevel_ncs_scratch).mkdir(parents=True, exist_ok=True)
ds_combined_loaded.to_netcdf(f_w_level_sims)

time_script_min = round((datetime.now() - end_time).seconds / 60, 1)
print("Wrote {} netcdf file of water level time series in an additional {} (min)".format(time_script_min))