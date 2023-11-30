#%% loading packages
# from __ref_ams_functions import fit_3_param_pdf
# from __ref_ams_functions import fit_2_param_pdf
from __ref_ams_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import scipy
from scipy import stats
import sys
from tqdm import tqdm
from scipy.stats import bootstrap
import xarray as xr
from _inputs import *
import shutil
from pathlib import Path
from tqdm import tqdm

alpha = 0.05
bootstrap_iterations = 100
sst_tstep = 5 # minutes
n_events_per_year_sst = 5

# f_selection = "outputs/b_pdf_selections.csv"
# f_cdfs_obs = "outputs/b2_F_of_obs_data-cdfvals.csv"
# f_cdfs_sst = "outputs/b3_F_of_sst_data-cdfvals.csv"
f_observed_wlevel_rainfall_tseries = "outputs/b_observed_compound_event_timeseries.csv"
f_observed_compound_event_summaries = "outputs/b_observed_compound_event_summaries.csv"
f_sst_event_summaries = "outputs/b_sst_event_summaries.csv"
# f_wlevel_cdf_sims_from_copula = "outputs/r_a_sim_wlevel_cdf.csv"
f_pdf_performance="outputs/b_pdf_performance_comparison.csv"

#%% load and process data data
ds_rlztns = xr.open_dataset(f_rain_realizations)

df_mrms_event_summaries = pd.read_csv(f_mrms_event_summaries, parse_dates=["start", "end", "max_intensity_tstep"])
df_mrms_event_summaries["duration"] = pd.to_timedelta(df_mrms_event_summaries["duration"])
df_mrms_event_tseries = pd.read_csv(f_mrms_event_timeseries, parse_dates=True, index_col="time")
df_water_levels = pd.read_csv(f_water_level_storm_surge, parse_dates=True, index_col="date_time")

# join water level and time series data
# resample water level time series to same timestep as rainfall
df_water_levels_1min = df_water_levels.loc[:, ["water_level", "predicted_wl", "surge_ft"]].resample('1min').mean().ffill()
df_water_levels_5min = df_water_levels_1min.loc[:, ["water_level", "predicted_wl", "surge_ft"]].resample('5min').mean()


df_water_rain_tseries = df_water_levels_5min.join(df_mrms_event_tseries, how="inner")
# verify all data was preserved
if (len(df_water_rain_tseries) - len(df_mrms_event_tseries)) != 0:
    sys.exit("ERROR: joining the water level to rainfall time series dropped some observations. Timesteps likely aren't lining up.")
# verify NA values were NOT introduced
if df_water_rain_tseries.isna().sum().sum() > df_mrms_event_tseries.isna().sum().sum():
    sys.exit("ERROR: NA Values introduced in joining observed water level time series to observed rain event time series")

# compute summary statistics by event for the water levels
s_surge_event_group = df_water_rain_tseries.groupby("event_id").surge_ft
s_peak_surge_time = s_surge_event_group.idxmax().rename("max_surge_tstep")
s_peak_surge = s_surge_event_group.max().rename("max_surge_ft")

df_compound_summary = df_mrms_event_summaries.join(s_peak_surge, on = "event_id")
df_compound_summary = df_compound_summary.join(s_peak_surge_time, on = "event_id")

s_lag_min = (df_compound_summary.max_intensity_tstep - df_compound_summary.max_surge_tstep) / pd.Timedelta(minutes=1)
df_compound_summary["surge_peak_after_rain_peak_min"] = s_lag_min

df_compound_summary["duration_hr"] = df_compound_summary['duration'] / pd.Timedelta("1H")
events_per_year = df_compound_summary.start.dt.year.value_counts()
if (events_per_year.max() != n_events_per_year_sst) and (events_per_year.min() != n_events_per_year_sst):
    sys.exit("ERROR: The number of events per year used for SST is different than that of observed event selection.")

df_compound_summary.to_csv(f_observed_compound_event_summaries)

# verify that the event summaries are the same
df_water_rain_tseries_sorted = df_water_rain_tseries.sort_values(["event_id", "date_time"])

event_sum_computed_from_tseries = df_water_rain_tseries_sorted.loc[:,['event_id', "mrms_mm_per_hour","surge_ft"]].groupby("event_id").max().sort_values("event_id")

event_sum = df_compound_summary.loc[:,["event_id", "max_mm_per_hour", "max_surge_ft"]].set_index("event_id")
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

df_water_rain_tseries_sorted.to_csv(f_observed_wlevel_rainfall_tseries)

# processing sst data
df_sst_storms = ds_rlztns.mean(dim = ["latitude", "longitude"]).to_dataframe().reset_index()
df_sst_storms = df_sst_storms.rename(dict(time = "tstep_ind", rain = "precip_mm_per_hour"), axis = 1)

## drop all rows with zeros
df_sst_storms = df_sst_storms[~(df_sst_storms.precip_mm_per_hour==0)].reset_index(drop=True)

## compute event statistics
df_sst_storms["precip_mm"] = df_sst_storms["precip_mm_per_hour"] * (sst_tstep/60) # mm per hour * 5 min per tstep / (60 minutes per hour) = mm per tstep

df_sums = df_sst_storms.groupby(["realization", "year", "storm_id"]).sum()
df_mins = df_sst_storms.groupby(["realization", "year", "storm_id"]).min()
df_maxes = df_sst_storms.groupby(["realization", "year", "storm_id"]).max()
df_idxmaxes = df_sst_storms.groupby(["realization", "year", "storm_id"]).idxmax()
df_max_precip_intensity_tstep = df_sst_storms.iloc[df_idxmaxes.precip_mm_per_hour.values, :].groupby(["realization", "year", "storm_id"]).sum()

# duration in number of timesteps
df_sst_duration_ntsteps = (df_maxes.tstep_ind - df_mins.tstep_ind)
df_sst_duration_ntsteps.name = "duration_n_tsteps"
# duration in hours
df_sst_duration_hr = df_sst_duration_ntsteps* sst_tstep / 60
df_sst_duration_hr.name = "duration_hr"
# depth
df_sst_depth_mm = df_sums.precip_mm
df_sst_depth_mm.name = "depth_mm"
# mean intensity
df_sst_mean_mm_per_hr = df_sst_depth_mm / df_sst_duration_hr
df_sst_mean_mm_per_hr.name = "mean_mm_per_hr"
# max intensity
df_sst_max_mm_per_hour = df_max_precip_intensity_tstep.precip_mm_per_hour
df_sst_max_mm_per_hour.name = "max_mm_per_hour"
# timestep of max intensity
df_sst_tstep_max_intensity = df_max_precip_intensity_tstep.tstep_ind
df_sst_tstep_max_intensity.name = "tstep_of_max_intensity"
# last timestep with rainfall
df_sst_last_timestep_w_rainfall = df_maxes.tstep_ind
df_sst_last_timestep_w_rainfall.name = "last_timestep_w_rainfall"

## create sst event summaries table
df_sst_event_summaries = pd.concat([df_sst_depth_mm, df_sst_mean_mm_per_hr, df_sst_max_mm_per_hour,
                                    df_sst_tstep_max_intensity, df_sst_last_timestep_w_rainfall,
                                    df_sst_duration_ntsteps], axis = 1)

df_sst_event_summaries.to_csv(f_sst_event_summaries)
# create list of functions and variables
fxs = [gev, loggev, weibull_min_dist, logwweibull_min_dist, weibull_max_dist, logwweibull_max_dist,
       gumbel_right, gumbel_left, loggumbel, norm_dist, lognormal, student_t, genpareto_dist, chi_dist, gamma_dist, p3, lp3, sep]

vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]
#%% fit all possible pdfs and transformations for each variable
def return_boxcox_lamda(x_for_fitting_shifted):
    x_for_fitting, lmbda = stats.boxcox(x_for_fitting_shifted)
    return lmbda

lst_df_perf = []
df_perf = pd.DataFrame()
lst_normalize = [False, True]
lst_boxcox = [False, True]
ind = -1
for v in tqdm(vars_all):
    s_var_notrns = df_compound_summary[v]
    for nrmlz in lst_normalize:
        s_var = s_var_notrns.copy()
        if nrmlz == True:
            # use median of bootstrapped mean and standard deviations for estimating these statistics
            bs_mean = bootstrap((s_var,), np.mean, confidence_level=0.9, n_resamples = 10000)
            nrmlz_mean = pd.Series(bs_mean.bootstrap_distribution).median() # s_var.mean()
            bs_std = bootstrap((s_var,), np.std, confidence_level=0.9, n_resamples = 10000)
            nrmlz_std = pd.Series(bs_std.bootstrap_distribution).median() # s_var.std()
            s_var = (s_var - nrmlz_mean)/nrmlz_std
        else:
            nrmlz_mean = np.nan
            nrmlz_std = np.nan
        for bxcx in lst_boxcox:
            shift = 0
            lmbda = np.nan
            x_for_fitting = s_var.copy()
            if bxcx == True:
                if x_for_fitting.min() <= 0:
                    shift = abs(2*x_for_fitting.min())
                x_for_fitting_shifted = x_for_fitting+shift
                # estimate boxcox lambda using bootstrapping
                bs_lambda = bootstrap((x_for_fitting_shifted,), return_boxcox_lamda, confidence_level=0.9, n_resamples = 1000)
                lmbda = pd.Series(bs_lambda.bootstrap_distribution).median()
                x_for_fitting = stats.boxcox(x_for_fitting_shifted, lmbda = lmbda)
                x_for_fitting = pd.Series(x_for_fitting)
            for f in fxs:
                fx_name = f["args"]["fx"].name
                try:
                    if f["args"]["log"] == True:
                        fx_name = 'log_' + fx_name
                except:
                    pass
                try:
                    out = f["function"](x_for_fitting, **f['args'], normalized = nrmlz, boxcox = bxcx, scalar_shift = shift, nbs = bootstrap_iterations)
                    params = out["params"]
                    if len(params) == 2:
                        loc, scale = params
                        shape = np.nan
                    if len(params) == 3:
                        shape, loc, scale = params
                    d = {"data":v, "normalized":nrmlz, "normalize_mean":nrmlz_mean,
                        #  "mean":np.average(x_for_fitting), "stdv":np.std(x_for_fitting),
                         "normalize_std":nrmlz_std, "boxcox":bxcx, "boxcox_lambda":lmbda, "scalar_shift":shift,
                         "fx":fx_name, "madi":out["madi"],"msdi":out["msdi"], "ks_pval":out["ks_pval"], "cvm_pval":out["cvm_pval"],
                        "n_params":out["n_params"], "aic":out["aic"], "shape":shape, "loc":loc, "scale":scale}
                    ind += 1
                    lst_df_perf.append(pd.DataFrame(index=[ind], data=d))
                except:
                    # print("failed to fit function {} to {}. Normalize set to {} and boxcox set to {}.".format(fx_name, v, nrmlz, bxcx))
                    pass

df_perf = pd.concat(lst_df_perf)
df_perf = df_perf.reset_index(drop = True)

df_perf.to_csv(f_pdf_performance, index = False)
#%% consolidated the best n_to_analyze fits for each variable based on AIC and cramer von mises p-value
n_to_analyze = 10

cvm_pval_rank = df_perf.loc[:,["data", "cvm_pval"]].groupby("data").rank(ascending=False)
aic_rank = df_perf.loc[:,["data", "aic"]].groupby("data").rank(ascending=True)
df_perf["cvm_rank"] = cvm_pval_rank
df_perf["aic_rank"] = aic_rank

df_best_cvm = df_perf.sort_values("cvm_pval", ascending=False).groupby("data").head(n_to_analyze)

df_best_aic = df_perf.sort_values("aic", ascending=True).groupby("data").head(n_to_analyze)

idx_best = pd.Series(list(df_best_cvm.index.values) + list(df_best_aic.index.values)).unique()

#%% plotting all the possible fits to support manual selection of PDFs
fldr_plt = "plots/b_fitting_marginals/"
shutil.rmtree(fldr_plt)
Path(fldr_plt).mkdir(parents=True, exist_ok=True)

df_best = df_perf.loc[idx_best, :]

df_best = df_best[df_best.cvm_pval > alpha]
for index, row in df_best.iterrows():
    v = row.data
    s_var_notrns = df_compound_summary[v]
    s_var = s_var_notrns.copy()
    nrmlz = row.normalized
    if nrmlz == True:
        s_var = (s_var - s_var.mean())/s_var.std()
    shift = row.scalar_shift
    lmbda = row.boxcox_lambda
    x_for_fitting = s_var.copy()
    bxcx = row.boxcox
    if bxcx == True:
        x_for_fitting = stats.boxcox(x_for_fitting+shift, lmbda=lmbda)
        x_for_fitting = pd.Series(x_for_fitting)
    for f in fxs:
        fx_name = f["args"]["fx"].name
        try:
            if f["args"]["log"] == True:
                fx_name = 'log_' + fx_name
        except:
            pass
        try:
            best_fit_fx = row.fx
            if fx_name != best_fit_fx:
                continue
            args = {"plot":True, "recurrence_intervals":None, "xlab":v, "normalized":nrmlz, "boxcox":bxcx, "scalar_shift":shift}
            out = f["function"](x_for_fitting, **f['args'], **args)
            if row.aic < 0:
                aic_desc = "n" + str(round(row.aic, 1)*-1)
            else:
                aic_desc = str(round(row.aic, 1))
            figname = "{}_aic-{}_cvmpval-{}_dist-{}.png".format(v, aic_desc, round(row.cvm_pval, 2), row.fx)
            plt.savefig(fldr_plt + figname, bbox_inches='tight')
        except:
            print("failed to fit function {} to {}. Normalize set to {} and boxcox set to {}.".format(fx_name, v, nrmlz, bxcx))
            pass
