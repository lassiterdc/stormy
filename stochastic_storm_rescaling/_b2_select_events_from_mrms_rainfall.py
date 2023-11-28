#%% Import libraries and load directories
import pandas as pd
import sys
from _inputs import *
# from tqdm import tqdm

# f_mrms_rainfall, f_water_level_storm_surge, min_interevent_time, max_event_length, min_event_threshold, mm_per_inch, f_mrms_event_summaries, f_mrms_event_timeseries = def_inputs_for_b()
min_event_threshold = min_event_threshold * mm_per_inch

#%% load data
df_rainfall = pd.read_csv(f_mrms_rainfall, parse_dates=True, index_col = "time")

df_sea = pd.read_csv(f_water_level_storm_surge, parse_dates=True, index_col = "date_time")

#%% preprocess data
# replace missing values with zeros
s_mean_rainfall = df_rainfall.mrms_mean.fillna(0)

# convert data to the same -minute timestep
## first convert to 1 minute, using fill forward (assuming a FOLLOWING time interval)
s_mean_rainfall_1min = s_mean_rainfall.resample('1min').mean().ffill()
## convert to 5 min
s_mean_rainfall_5min = s_mean_rainfall_1min.resample('5min').mean()

# create a timeseries with depths in mm (converting from mm/hr to mm)
s_mean_rainfall_5min_mm = s_mean_rainfall_5min * (5/60) # mm per --hour-- * 5 --min-- per tstep / (60 --min-- per --hour--) = mm per tstep

# compute a rolling sum of rainfall depth at a time interval equal to the sst event duration
s_rain_rollingsum_mm = s_mean_rainfall_5min_mm.rolling('{}h'.format(sst_event_duration), min_periods = 1).sum()

# s_rain_rollingsum_eventlen_mm = s_mean_rainfall_5min_mm.rolling('{}h'.format(max_event_length), min_periods = 1).sum()

# identify possible event starts (all timesteps where min event threshold was reached in the rolling sum)
possible_starts = s_rain_rollingsum_mm[s_rain_rollingsum_mm > min_event_threshold]

# create series with the time difference of the possible starts
sr_times = pd.Series(possible_starts.index, index=possible_starts.index)
sr_times = sr_times.diff()

# convert the min. interevent time and max storm length to TimeDelta datatype
min_tdif = pd.Timedelta('{} hours'.format(min_interevent_time))
## rolling sum is PREceding so subtract min_tdif from the max storm event length
max_strm_len = pd.Timedelta('{} hours'.format(sst_event_duration)) - min_tdif 

#%% perform event selection
# define lists
event_starts = []
event_ends = []
event_duration = []
event_ids = []
mean_intensities = []
max_intensities = []
max_intensity_tsteps = []
total_precip = []
event_id = 1
lst_s_intensities = []

# set up the loop so it skips timesteps already included in an event
prev_event_endtime = min(possible_starts.index.values) - min_tdif

# perform event selection
for t in possible_starts.index.values:
    if t <= prev_event_endtime: # ensures separation between events
        continue
    # compute max event end time
    max_end_time = t + max_strm_len
    # max_storm_tseries = pd.date_range(t, end = max_end_time, freq = '5min')
    sr_max_storm = possible_starts[t:max_end_time]

    # determine whether there there are periods with 0 rain equal in length to min interevent time
    sr_interevent_time_reached = sr_max_storm[sr_max_storm == 0]
    if len(sr_interevent_time_reached) == 0: # if there aren't any gaps
        # compute initial esimate of event start and end times
        event_start = sr_max_storm.index.min()-min_tdif
        event_end = sr_max_storm.index.max()

        # update the previous event endtime variable
        prev_event_endtime = sr_max_storm.index.max()

        # compute total depth
        tot_depth = s_mean_rainfall_5min_mm[event_start:event_end].sum()

        # extract the intensities time seriies, removing preceding and trailing zeros)
        intensities = s_mean_rainfall_5min[event_start:event_end]
        intensity_cumsum = intensities.cumsum()
        first_tstep_with_rain = (intensity_cumsum > 0).idxmax()
        last_tstep_with_rain = intensity_cumsum.idxmax()
        intensities = intensities[first_tstep_with_rain:last_tstep_with_rain]

        # append the lists with relevant data
        event_starts.append(first_tstep_with_rain)
        event_ends.append(last_tstep_with_rain)
        event_duration.append(last_tstep_with_rain - first_tstep_with_rain)
        mean_intensities.append(intensities.mean())
        max_intensities.append(intensities.max())
        max_intensity_tsteps.append(intensities.idxmax())
        total_precip.append(tot_depth)
        event_ids.append(event_id)
        event_id += 1
        intensities.name = "mrms_mm_per_hour" # name becomes columnname of the concatenated dataframe
        lst_s_intensities.append(intensities)
    else: # if there are gaps (I haven't coded this yet because it wasn't necessary for my dataset)
        sys.exit('WARNING: EVENT HAS A GAP IN PRECIPITATION THAT EXCEEDS THE MINIMUM INTER-EVENT TIME.')

#%% combine into a files and export
df_event_summaries = pd.DataFrame(dict(event_id = event_ids, start = event_starts, end = event_ends, 
                              duration = event_duration, depth_mm = total_precip, 
                              mean_mm_per_hr = mean_intensities, max_mm_per_hour = max_intensities,
                              max_intensity_tstep = max_intensity_tsteps))

df_event_summaries["year"] = df_event_summaries.start.dt.year
df_event_summaries = df_event_summaries.sort_values("depth_mm", ascending = False)
df_event_summaries = df_event_summaries.groupby("year").head(sst_storms_per_year)
df_event_summaries = df_event_summaries.sort_values(["year", "depth_mm"], ascending = [True, False])
old_event_id = list(df_event_summaries.event_id.values)
df_event_summaries["old_event_id"] = old_event_id
df_event_summaries.reset_index(drop=True, inplace=True)
df_event_summaries.drop(["year", "event_id"], axis = 1, inplace=True)
df_event_summaries.index.rename("event_id", inplace = True)
new_event_id = list(df_event_summaries.index.values)

# process time series data
df_event_tseries = pd.concat(lst_s_intensities, keys=event_ids).reset_index()
df_event_tseries = df_event_tseries.rename(columns = {"level_0":"event_id"})
## update event id's in time series to match the event summaries
df_event_tseries = df_event_tseries[df_event_tseries.event_id.isin(old_event_id)]

df_event_id_mapping = pd.DataFrame(dict(new_event_id = new_event_id, event_id = old_event_id))
df_event_id_mapping = df_event_id_mapping.set_index("event_id")
df_event_tseries = df_event_tseries.set_index(["event_id", "time"])
df_event_tseries = df_event_tseries.join(df_event_id_mapping, how = "left", on = "event_id").reset_index()
df_event_tseries.drop("event_id", inplace=True, axis = 1)
df_event_tseries.rename(dict(new_event_id = "event_id"), axis = 1, inplace=True)
df_event_tseries.sort_values(["event_id", "time"], inplace=True)
df_event_tseries.set_index(["event_id", "time"], inplace=True)

# export to csv
df_event_summaries.to_csv(f_mrms_event_summaries, index=True)
df_event_tseries.to_csv(f_mrms_event_timeseries, index = True)
