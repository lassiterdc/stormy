#%% import libraries and load directories
import xarray as xr
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
import time
from glob import glob
# files
f_nc_rainfall = "D:/Dropbox/_GradSchool/_norfolk/stormy/swmm/hague/swmm_scenarios_sst/weather/rainfall.nc"

## mrms events
fldr_stormy = str(Path(os.getcwd()).parents[1]) + "/"
fldr_ssr = fldr_stormy + "stochastic_storm_rescaling/"
dir_ssr_outputs = fldr_ssr + "outputs/"
dir_mrms_events = dir_ssr_outputs + "c_mrms_events/"
f_mrms_event_summaries = dir_mrms_events + "mrms_event_summaries.csv"

f_ncei_airport = fldr_stormy + "data/climate/NCEI/2023-1-4_NCEI_daily summaries_download.csv"

tstep_min = 5

nstormsperyear = 5

## raw mrms data over watershed
fldr_highres_data = "D:/Dropbox/_GradSchool/_norfolk/highres-radar-rainfall-processing/data/"
f_nc_mrms_daily = fldr_highres_data + "mrms_nc_preciprate_24h_atgages.nc"

## stageiv data
lst_f_stageiv = glob(fldr_highres_data + "stage_iv_nc_preciprate_fullres_yearlyfiles_atgages/*.nc")
#%% load rain data
ds_rainyday = xr.open_dataset(f_nc_rainfall)

# compute mean rainfall in mm/hr for each event
ds_mean = ds_rainyday.mean(dim = ["latitude", "longitude", "time"])

# convert from mm/hr to mm
event_duration_hr = (len(ds_rainyday.time)*5)/60
ds_tot = ds_mean * event_duration_hr # mm / hr * hr

def find_largest_n_storms_per_year(ds_yr, nstormsperyear = nstormsperyear):
    lst_ds = []
    for rz in ds_yr.realization.values:
        ds_rz = ds_yr.sel(dict(realization = rz))
        top_storm_indices = ds_rz.rain.to_dataframe()["rain"].nlargest(n=nstormsperyear).index.values
        ds_rz_out = ds_rz.sel(dict(storm_id = top_storm_indices))
        ds_rz_out = ds_rz_out.assign_coords(dict(realization = rz))
        ds_rz_out = ds_rz_out.expand_dims("realization")
        # re-define storm_id 
        ds_rz_out['storm_id'] = np.arange(1, nstormsperyear+1)
        # append to list
        lst_ds.append(ds_rz_out)
    ds_out = xr.combine_by_coords(lst_ds)
    return ds_out


ds_tot = ds_tot.groupby("year").map(find_largest_n_storms_per_year)

#%% analyzing stochastically generated rainfall

#%% statistical testing

def report_ks_test(d1, d2, alpha):
    ks_result = stats.ks_2samp(d1, d2)
    reject = False
    if ks_result.pvalue < alpha:
        result = "reject"
        conclusion = "This supports the conclusion that \n\
these datasets DO NOT share an underlying distribution"
    else:
        result = "fail to reject"
        conclusion = "This supports the conclusion that \n\
these datasets share an underlying distribution"

    ks_string = "KS statistic = {} \n\
p-value = {} ({} at a {}% confidence level) \n\
".format(str(ks_result.statistic), str(round(ks_result.pvalue, 5)), result, str(int(100*(1-alpha))))
    ks_string = ks_string + conclusion
    # print(ks_string)
    return ks_result, ks_string

# report_ks_test(ds_tot.rain.values.flatten(), df_top_20, alpha=0.05)

#%% look at other statistics
unique_rain_depths = pd.DataFrame(np.unique(ds_tot.rain.values.flatten(), return_counts = True)).T
unique_rain_depths.columns = ["value", "count"]
try:
    n_zero_rain = int(unique_rain_depths[unique_rain_depths.value == 0]["count"])
except:
    n_zero_rain = 0
n_unique_rain_depths_exc_zeros = len(unique_rain_depths[unique_rain_depths.value != 0])
n_events_exc_zeros = len(ds_tot.rain.values.flatten()) - n_zero_rain
frac_unique_exc_zeros = n_unique_rain_depths_exc_zeros / n_events_exc_zeros

print("{} events had no rainfall.".format(n_zero_rain))
print("Exluding events with no rainfall, there are {} unique average rain depths \n\
      out of {} total events (fraction of unique events = {})".format(n_unique_rain_depths_exc_zeros, n_events_exc_zeros, frac_unique_exc_zeros))

#%% define event selection formula
def event_selection_for_sst_validation(df, vname_date, vname_precip, rain_units, fname_csv):
    """
    The variable df must be daily rainfall totals
    """
    start_time = time.time()
    lst_acceptable_units = ["mm", "in"]
    if rain_units not in lst_acceptable_units:
        print("Units not in list: {}".format(lst_acceptable_units))

    # subset only days with non-zero rainfall
    df = df[df[vname_precip] > 0]
    # be sure dates are in order
    df = df.sort_values(vname_date)
    # reset index so they are sequential (necessary for while loop below)
    df.reset_index(inplace=True, drop=True)

    # define variables
    df.insert(2, "start", df[vname_date])
    df.insert(3, "end", df.loc[:, "start"] + np.timedelta64(1, "D"))
    if rain_units == "in":
        multiplier = 25.4
    elif rain_units == "mm":
        multiplier = 1
    df.insert(4, "precip_mm", df[vname_precip] * multiplier)
    df.insert(4, "duration", np.timedelta64(1, "D"))

    # loop through the dataframe to select non-overlapping events with a minimum of a 3 day duration
    ind_to_drop = []
    ind = -1
    while ind < max(df.index):
        ind += 1
        row = df.loc[ind, :]
        date = row.start
        precip = row.precip_mm
        # check if there is rain in subsequent timesteps
        rain_in_next_tstep = True
        next_ind = ind
        tot_precip = precip
        while rain_in_next_tstep:
            next_ind += 1
            if next_ind > max(df.index):
                break
            next_row = df.loc[next_ind, :]
            next_date = next_row.start
            next_precip = next_row.precip_mm
            # if there is more than 2 days into the future OR the event is already 3 days long, finish the while loop
            if (next_date - date > np.timedelta64(2, "D")) or ((next_row.end - row.start) > np.timedelta64(3, "D")):
                rain_in_next_tstep = False
                continue
            # update precip value and total duration
            tot_precip += next_precip
            df.loc[ind, "precip_mm"] = tot_precip
            df.loc[ind, "end"] = next_row.start
            df.loc[ind, "duration"] = next_row.end - row.start
            ind_to_drop.append(next_ind)
            # skip to the next index
            ind = next_ind
    
    # drop all indices where the rainfall was included in a previous timestep
    df = df.drop(ind_to_drop)

    df.to_csv(fname_csv)
    print("Finished performing event selection in {} minutes.".format((time.time() - start_time)/60))
    return df

#%% processing stageIV data
rainyday_lats = ds_rainyday.latitude.values
rainyday_lons = ds_rainyday.longitude.values

# preprocessing
def preprocess_stage_iv(ds_s4):
    ds_s4["time"] = ds_s4.time.values - np.timedelta64(1, 'h')
    unique_tsteps = np.unique(pd.DataFrame(dict(time=ds_s4.time.values)).diff().dropna(), return_counts=True)
    df_unique_tsteps = pd.DataFrame(unique_tsteps).T
    df_unique_tsteps.columns = ['tstep', 'count']
    num_unique_tsteps = len(df_unique_tsteps)
    if num_unique_tsteps > 1:
        print('WARNING: dataset {} has {} different tsteps'.format(ds_s4.encoding["source"], num_unique_tsteps))
        print(df_unique_tsteps)

    # make sure each dataset only spans 1 year and correct if not:
    years = pd.Series(ds_s4.time.values).dt.year.unique()
    if len(years) > 1:
        first_year = min(years)
        ds_s4 = ds_s4.sel(time = ds_s4.time.dt.year.isin(first_year))

    # consolidating to mean daily intensities
    ds_s4 = ds_s4.resample(time = "1D").mean()
    # converting from mm/hr to mm
    ds_s4["rainrate"] = ds_s4.rainrate * 24
    return ds_s4

ds_stageiv = xr.open_mfdataset(lst_f_stageiv, preprocess=preprocess_stage_iv)

# convert from positive degrees west to negative degrees west
if max(ds_stageiv.longitude.values) > 180:
    ds_stageiv["longitude"] = ds_stageiv.longitude.values - 360

# subset only gridcells that overlap with the rainyday data
lst_stageiv_lats = []
lst_stageiv_lons = []
for lat in rainyday_lats:
    closest_lat = ds_stageiv.latitude.values[np.argmin(abs(ds_stageiv.latitude.values - lat))]
    if closest_lat not in lst_stageiv_lats:
        lst_stageiv_lats.append(closest_lat)
for lon in rainyday_lons:
    closest_lon = ds_stageiv.longitude.values[np.argmin(abs(ds_stageiv.longitude.values - lon))]
    if closest_lon not in lst_stageiv_lons:
        lst_stageiv_lons.append(closest_lon)

ds_stageiv = ds_stageiv.sel(dict(latitude = lst_stageiv_lats, longitude = lst_stageiv_lons))

ds_stageiv_mean = ds_stageiv.mean(dim = ["latitude", "longitude"])

df_stageiv_mean = ds_stageiv_mean.to_dataframe().dropna().reset_index()

fname_csv = "_scratch/stageiv_events.csv"

df_stageiv_events = event_selection_for_sst_validation(df = df_stageiv_mean, vname_date="time", vname_precip="rainrate", rain_units="mm", fname_csv = fname_csv)

# #%% troubleshooting
# tbs = "D:/Dropbox/_GradSchool/_norfolk/highres-radar-rainfall-processing/data/stage_iv_nc_preciprate_fullres_yearlyfiles_atgages/2011.nc"
# ds_s4 = xr.open_dataset(tbs)
#%% processing airport gage data
df_ncei = pd.read_csv(f_ncei_airport, parse_dates=["DATE"])
fname_csv = "_scratch/ncei_events.csv"

df_ncei_events = event_selection_for_sst_validation(df = df_ncei, vname_date="DATE", vname_precip="PRCP", rain_units="in", fname_csv = fname_csv)

#%% processing MRMS data
ds_mrms = xr.open_dataset(f_nc_mrms_daily)

# convert from positive degrees west to negative degrees west
if max(ds_mrms.longitude.values) > 180:
    ds_mrms["longitude"] = ds_mrms.longitude.values - 360

# subset mrms data to only those gridcells used in RainyDay
ds_mrms = ds_mrms.sel(dict(latitude = rainyday_lats, longitude = rainyday_lons))

ds_mrms_mean = ds_mrms.mean(dim = ["latitude", "longitude"])

# convert from mm/hr to mm
# ds_mrms_mean_tot = ds_mrms_mean * 24 # mm / hr * hr
ds_mrms_mean_tot = ds_mrms_mean
# prepare for event selection
df_mrms_mean_tot = ds_mrms_mean_tot.to_dataframe().dropna().reset_index()

fname_csv = "_scratch/mrms_events.csv"

df_mrms_events = event_selection_for_sst_validation(df = df_mrms_mean_tot, vname_date="time", vname_precip="rainrate", rain_units="mm", fname_csv = fname_csv)

#%% define function for comparing datasets with one another
def compare_event_totals(df1, df2, df1_name, df2_name, df1_color, df2_color, ks_alpha = 0.05):
    fig, ax = plt.subplots()

    df1_heights, df1_bins = np.histogram(df1, density = True)
    df2_heights, df2_bins = np.histogram(df2, bins=df1_bins, density = True)

    width = (df1_bins[1] - df1_bins[0])/3

    ax.bar(df1_bins[:-1], df1_heights, width=width, facecolor=df1_color, label = df1_name)
    ax.bar(df2_bins[:-1]+width, df2_heights, width=width, facecolor=df2_color, label = df2_name)

    ax.legend()
    ax.set_title("histogram of event totals")
    ax.set_xlabel("event depth (mm)")
    ax.set_ylabel("density")

    print("##### comparing {} with {} #####".format(df1_name, df2_name))
    ks_result, ks_string = report_ks_test(df1, df2, alpha=ks_alpha)
    # transform=ax.transAxes
    ax.text(0.15, 0.8, ks_string, size = 8, ha="left", color="black", transform=ax.transAxes)

#%% comparing SST results with observed time series
### comparing rainyday with stageiv
ar_sst_event_tots = ds_tot.rain.values.flatten()

df_stageiv_events["year"] = df_stageiv_events.start.dt.year
df_stageiv_events_top = df_stageiv_events.groupby("year")["precip_mm"].nlargest(n=nstormsperyear)
compare_event_totals(ar_sst_event_tots, df_stageiv_events_top, df1_name="sst", \
                     df2_name="stage iv", df1_color = 'cornflowerblue', df2_color = 'seagreen')

### comparing rainyday with MRMS
df_mrms_events["year"] = df_mrms_events.start.dt.year
df_mrms_events_top = df_mrms_events.groupby("year")["precip_mm"].nlargest(n=nstormsperyear)
compare_event_totals(ar_sst_event_tots, df_mrms_events_top, df1_name="sst", \
                     df2_name="mrms", df1_color = 'cornflowerblue', df2_color = 'seagreen')

### comparing rainyday with NCEI
df_ncei_events["year"] = df_ncei_events.start.dt.year
df_ncei_events_top = df_ncei_events.groupby("year")["precip_mm"].nlargest(n=nstormsperyear)
compare_event_totals(ar_sst_event_tots, df_ncei_events_top, df1_name="sst", \
                     df2_name="NCEI", df1_color = 'cornflowerblue', df2_color = 'seagreen')
#%% comparing observed weather time series with one another
### comparing MRMS with NCEI
compare_event_totals(df_mrms_events_top, df_ncei_events_top, df1_name="mrms", df2_name="NCEI",\
                      df1_color='blue', df2_color='seagreen')

### comparing MRMS with Stage IV
compare_event_totals(df_mrms_events_top, df_stageiv_events_top, df1_name="mrms", df2_name="stage iv",\
                      df1_color='blue', df2_color='seagreen')

### comparing NCEI with Stage IV
compare_event_totals(df_stageiv_events_top, df_ncei_events_top, df1_name="stage iv", df2_name="NCEI",\
                      df1_color='blue', df2_color='seagreen')
#%% sanity check 
print("############# SANITY CHECK ######################")

# report_ks_test(stats.uniform.rvs(size=100), stats.norm.rvs(size=110), alpha=0.05)
compare_event_totals(stats.uniform.rvs(size=100), stats.norm.rvs(size=110), df1_name="uniform", df2_name="normal",\
                      df1_color='blue', df2_color='seagreen')

# report_ks_test(stats.uniform.rvs(size=100), stats.uniform.rvs(size=100), alpha=0.05)
compare_event_totals(stats.uniform.rvs(size=100), stats.uniform.rvs(size=100), df1_name="uniform", df2_name="uniform",\
                      df1_color='blue', df2_color='seagreen')