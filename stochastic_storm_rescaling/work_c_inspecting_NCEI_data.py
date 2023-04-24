#%% import libraries and load directories
from _inputs import def_work_c
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
f_daily_summaries, f_hourlyglobal_precip_all_stations, f_hourlyglobal_precip_subset_stations, f_hourlyprecip = def_work_c()
min_record_length = 30 # years; for subsetting stations with sufficiently long record
fldr_plt_out = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_rescaling/outputs/work_c_NCEI_qaqc/"

#%% load data
df_daily = pd.read_csv(f_daily_summaries, parse_dates=['DATE'], low_memory=False)
df_hrlyglbl_all = pd.read_csv(f_hourlyglobal_precip_all_stations, parse_dates=['DATE'], low_memory=False)
df_hrlyglbl_sbst = pd.read_csv(f_hourlyglobal_precip_subset_stations, parse_dates=['DATE'], low_memory=False)
df_hrlyprcp = pd.read_csv(f_hourlyprecip, parse_dates=['DATE'], low_memory=False)
#%% functions
def find_stations_with_min_record_length(df, sta_colname, date_colname, cutoff):
    df = df[[sta_colname, date_colname]]
    df_starts = df.groupby(by=sta_colname).min()[date_colname]
    df_ends = df.groupby(by=sta_colname).max()[date_colname]
    df_dur = df_ends.dt.year - df_starts.dt.year
    df_stations_meeting_critiera = df_dur[df_dur > min_record_length]
    print("{} stations had a record length of at least {} years. {} stations were dropped.".format(len(df_stations_meeting_critiera), cutoff, len(df_ends)-len(df_stations_meeting_critiera)))
    return list(df_stations_meeting_critiera.index.values)

def filter_stations_with_min_record(df, sta_colname = "STATION", date_colname = "DATE", cutoff=min_record_length):
    sta_ids = find_stations_with_min_record_length(df, sta_colname, date_colname, cutoff)
    df_subset = df[df[sta_colname].isin(sta_ids)]
    return df_subset

def gen_qaqc_plots(df, resolution, cutoff_date=None, colname_date = 'DATE', colname_station = 'STATION', colname_precip = 'PRCP'):
    # df = df_daily.copy()
    """
    df: data frame of NCEI precipitation data
    resolution: perported time resolution of the data readable by the pandas df.resmple() method (e.g., '1h', '1d')
    cutoff_date: date prior to which is sliced off, readable by a pandas conditional statement (e.g. '12/31/1953')
    colname_date: column name with pandas datetime
    colname_station: column name with unique NCEI station name
    colname_precip: column name with precipitation data in inches
    """
    lst_res_test = ['1h', '1d']
    if resolution not in lst_res_test:
        print("Resolution needs to be in this list: {}".format(lst_res_test))
        return
    # pull unique station IDs for faceting the plots
    sta_ids = df[colname_station].unique()
    # df = df_daily.copy()
    # formatting
    df = df[df.PRCP.notna()] # remove na precip values
    df = df[[colname_date, colname_station, colname_precip]]
    df = df.sort_values([colname_station, colname_date])
    df = df.set_index(colname_date, drop=True)
    df = df.groupby(colname_station, as_index=False).resample(resolution).asfreq(fill_value=0).reset_index()
    tstep_hrs = int(df.DATE.diff().mode().astype('timedelta64[h]')[0]) # ensure even time step; fill with zeros
    df = df[df.PRCP>0] # remove observations with no rainfall
    df = df[df.PRCP!=999.99] # remove observations with the 999.99 flag
    # df = df.reset_index()
    df['tdiff_hr'] = df.groupby(colname_station).diff().DATE.astype('timedelta64[h]')-tstep_hrs # compute the hourly time difference between non-zero observations
    df = df.set_index("DATE", drop=False) # add date index back to dataset
    if cutoff_date is not None:
        df = df[df.DATE>cutoff_date]

    for id in sta_ids:
        fig, ax = plt.subplots(dpi=300)
        df_plt = df[df.STATION==id]
        s = df_plt['tdiff_hr']
        upperlim = s.quantile(q=0.99)
        s = s[s<upperlim]
        s.resample('1y').mean().plot(ax =ax)
        ax.set_ylabel("Annual Average Time Difference Between \n Non-Zero Rainfall Observations (hours)")
        ax.set_xlabel("Year")
        ax.set_title("Station: {} ({}hr timestep) \n Time Differences Between Non-Zero Observations \n (0.99 quantile and up excluded)".format(id, tstep_hrs))

        fig, ax = plt.subplots(dpi=300)
        df_plt['PRCP'].resample('1y').sum().plot(ax =ax)
        ax.set_ylabel("Total Annual Rainfall (in)")
        ax.set_xlabel("Year")
        ax.set_title("Station: {} ({}hr timestep) \n Annual Precipitation Totals".format(id, tstep_hrs))
    return
#%% data processing
df_daily = df_daily[['STATION', 'NAME', 'DATE', 'PRCP']]
df_daily = filter_stations_with_min_record(df_daily)

# looks like garbage
hrlyglbl_precip_col = ['STATION', 'DATE', 'AA1', 'AA2', 'AA3', 'AA4']
df_hrlyglbl_all = df_hrlyglbl_all[hrlyglbl_precip_col]
# df_hrlyglbl_all = df_hrlyglbl_all.rename(columns={hrlyglbl_precip_col:'PRCP'})
# df_hrlyglbl_all = filter_stations_with_min_record(df_hrlyglbl_all)

# # looks like garbage
df_hrlyglbl_sbst = df_hrlyglbl_sbst[hrlyglbl_precip_col]
# df_hrlyglbl_sbst = df_hrlyglbl_sbst.rename(columns={hrlyglbl_precip_col:'PRCP'})
# df_hrlyglbl_sbst = filter_stations_with_min_record(df_hrlyglbl_sbst)

# looks good
df_hrlyprcp = df_hrlyprcp[['STATION', 'DATE', 'HPCP']]
df_hrlyprcp = df_hrlyprcp.rename(columns={'HPCP':'PRCP'})
df_hrlyprcp = filter_stations_with_min_record(df_hrlyprcp)

#%% figuring out which PRCP column is rainfall
"""
These look like garbage; none of the values in the precipitation column look like rainfall
"""
# df = df_hrlyglbl_all[df_hrlyglbl_all["STATION"]==72307513769].copy()
sta_id = 74598013702
df = df_hrlyglbl_sbst[df_hrlyglbl_sbst["STATION"]==sta_id].copy()

for col in ['AA1','AA2','AA3','AA4']:
    df_tmp = df[col].str.split(',', expand=True)
    for i in np.arange(4):
        newname = col+"_"+str(i)
        df_tmp = df_tmp.rename(columns={i:newname})
    # df_tmp = df_tmp.rename(columns={0:new_names, 1:"val2", 2:'val3', 3:'val4'})
    df = df.drop(columns=col)
    df = pd.concat([df, df_tmp], axis = 1)
# df = df.rename(columns={0:"val1", 1:"val2", 2:'val3', 3:'val4'})
# df = df[["DATE", 'val1', 'val2', 'val3', 'val4']]
df = df.drop(columns="STATION")
df = df.set_index("DATE")

for colname, series in df.items():
    sr = series.astype(float)
    sr = sr.dropna()
    if len(sr) == 0:
        continue
    fig, ax = plt.subplots(dpi=300)
    sr.resample('1y').sum().plot(ax =ax)
    ax.set_ylabel("Total Annual Rainfall (in)")
    ax.set_xlabel("Year")
    ax.set_title("Station: {} ({}hr timestep) \n Annual Precipitation Totals".format(sta_id, str(1)))
    fig.savefig(fldr_plt_out + "ncei_qaqc_hourly global_station {}_series {}.png".format(sta_id, colname))

valid_ids = []
for row, col in df.items():
    try:
        col.astype(float)
        valid_ids.append(row)
    except:
        print("{} contains non numeric values.".format(row))
        continue
df = df[valid_ids]
df = df.astype(float)

#%% plotting
for row, col in df.items():
    plt.figure()
    # sr = col
    sr = col.dropna()
    # sr = sr[sr>0] # non zero values only
    sr = sr[sr<20]
    # ax = sr.plot.hist(bins=50)
    ax = sr.plot(figsize=(8, 10))
    ax.set_title(row)
    # plt.clf()

# df.plot(subplots=True, figsize=(20, 20))

#%% plotting the other hourly subest
"""
This one looks reasonably like rainfall
"""
# df= df_hrlyprcp.copy()
# df = df.set_index("DATE")
# df.PRCP[df.PRCP<20].plot()
print("Plotting hourly dataset...")
gen_qaqc_plots(df_hrlyprcp, resolution='1h', cutoff_date = '12/31/1953')

#%% checking out daily data
"""
These also look reasonable.
"""
# df = df_daily.copy()
# df = df[["STATION", "DATE", "PRCP"]]
# df = df.set_index("DATE")
# df.groupby("STATION").plot()
print("Plotting daily dataset...")
gen_qaqc_plots(df_daily, resolution='1d')