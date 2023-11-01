#%% import libraries and load directories
import numpy as np
import xarray as xr
import pandas as pd
from _inputs import *
import pyvinecopulib as pv
import seaborn as sns
from scipy import stats
#%% load data
ds_rlztns = xr.open_dataset(f_rain_realizations)

df_key = pd.read_csv(f_key_subnames_gridind)

df_mrms_event_summaries = pd.read_csv(f_mrms_event_summaries, parse_dates=["start", "end", "max_intensity_tstep"])
df_mrms_event_summaries["duration"] = pd.to_timedelta(df_mrms_event_summaries["duration"])
df_mrms_event_tseries = pd.read_csv(f_mrms_event_timeseries, parse_dates=True, index_col="time")
df_water_levels = pd.read_csv(f_water_level_storm_surge, parse_dates=True, index_col="date_time")

ktau_alpha = 0.01
#%% processing data
df_sst_storms = ds_rlztns.mean(dim = ["latitude", "longitude"]).to_dataframe().reset_index()
df_sst_storms = df_sst_storms.rename(dict(time = "tstep_ind", rain = "precip_mm_per_hour"), axis = 1)
df_sst_storms["rz_yr_strm"] = df_sst_storms["realization"].astype(str) + "_" + df_sst_storms["year"].astype(str) + "_" + df_sst_storms["storm_id"].astype(str)
#%% join water level and time series data
df_water_rain_tseries = df_water_levels.join(df_mrms_event_tseries, how="inner")


#%% compute summary statistics by event for the water levels
s_surge_event_group = df_water_rain_tseries.groupby("event_id").surge_ft
s_peak_surge_time = s_surge_event_group.idxmax().rename("max_surge_tstep")
s_peak_surge = s_surge_event_group.max().rename("max_surge_ft")

df_compound_summary = df_mrms_event_summaries.join(s_peak_surge, on = "event_id")
df_compound_summary = df_compound_summary.join(s_peak_surge_time, on = "event_id")

s_lag_min = (df_compound_summary.max_intensity_tstep - df_compound_summary.max_surge_tstep) / pd.Timedelta(minutes=1)
df_compound_summary["surge_peak_after_rain_peak_min"] = s_lag_min

df_compound_summary["duration_hr"] = df_compound_summary['duration'] / pd.Timedelta("1H")
#%% fit statistical models
# predictors: [duration_hr, depth_mm, mean_mm_per_hour, max_mm_per_hour]
# response: [max_surge_ft, surge_peak_after_rain_peak_min]
vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]
vars_cond = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour"]
vars_sim = ["max_surge_ft", "surge_peak_after_rain_peak_min"]

df_vars_all = df_compound_summary.loc[:, vars_all]

df_vars_all_uniform = pd.DataFrame(pv.to_pseudo_obs(df_vars_all))
df_vars_all_uniform.columns = df_vars_all.columns
axes = sns.pairplot(df_vars_all_uniform)
#%% compute kendal taus
lst_taus = []
lst_pairs_processed = []
for col1 in df_vars_all.columns:
    for col2 in df_vars_all.columns:
        if col1 == col2:
            continue
        lst_vars = [col1, col2]
        lst_vars.sort()
        if lst_vars not in lst_pairs_processed:
            lst_pairs_processed.append(lst_vars)
        else:
            continue
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html#scipy.stats.kendalltau
        ktau, ktau_p = stats.kendalltau(df_vars_all[col1], df_vars_all[col2],
                                alternative = "two-sided")
        test = "fail to reject"
        conclusion = "no correlation"
        if ktau_p < ktau_alpha:
            test = "reject"
            conclusion = "correlation"
        dic_results = dict(variable_1 = col1,
                           variable_2 = col2, 
                           kendall_tau = ktau,
                           kendall_tau_pval = ktau_p,
                           test_result = test,
                           conclusion = conclusion)
        lst_taus.append(dic_results)
df_kendall_tau_test_results = pd.DataFrame(lst_taus)
#%% building vine copulas
# relevant literature:
    # developers said to look at section 4.2 of https://www.degruyter.com/document/doi/10.1515/demo-2021-0103/html
# Copulas libarary
    # https://sdv.dev/Copulas/index.html
# pyvinecopulib library
    # relevant issue on creating conditional simulations
        # https://github.com/vinecopulib/pyvinecopulib/issues/86
    # examples
        # https://github.com/vinecopulib/pyvinecopulib/blob/main/examples/vine_copulas_fit_sample.py
        # https://github.com/vinecopulib/pyvinecopulib/blob/main/examples/vine_copulas.ipynb
    # api
        # https://vinecopulib.github.io/pyvinecopulib/_generate/pyvinecopulib.Vinecop.html
    # github
        # https://github.com/vinecopulib/pyvinecopulib/tree/f6b5318dc7144aa7417047b16186482183064598