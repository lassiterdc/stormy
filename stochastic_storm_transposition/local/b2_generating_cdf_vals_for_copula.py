#%% loading packages
from __ref_ams_functions import fit_3_param_pdf
from __ref_ams_functions import fit_2_param_pdf
from __ref_ams_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# import scipy
from scipy import stats
# import sys
# from tqdm import tqdm
# from scipy.stats import bootstrap
# import xarray as xr
from _inputs import *
import shutil
from pathlib import Path
# from tqdm import tqdm


f_selection = "outputs/b_pdf_selections.csv"
f_cdfs_obs = "outputs/b2_F_of_obs_data-cdfvals.csv"
f_cdfs_sst = "outputs/b3_F_of_sst_data-cdfvals.csv"
f_observed_compound_event_summaries = "outputs/b_observed_compound_event_summaries.csv"
f_sst_event_summaries = "outputs/b_sst_event_summaries.csv"
f_pdf_performance="outputs/b_pdf_performance_comparison.csv"

# create list of variables
vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]

# picking best distribution for each dataset and exporting selection to csv
df_perf = pd.read_csv(f_pdf_performance)
df_compound_summary = pd.read_csv(f_observed_compound_event_summaries)
df_sst_event_summaries = pd.read_csv(f_sst_event_summaries)


fldr_plt_selected = "plots/b_fitting_marginals_final_selection/"
shutil.rmtree(fldr_plt_selected)
Path(fldr_plt_selected).mkdir(parents=True, exist_ok=True)


# create dictionaries to plot the top 3 of each variable based on visual inspection
lst_d_top3_depth_mm = [
    {"variable":"depth_mm", "nrmlz":False, "bxcx":True,"dist":"pearson3"},
    {"variable":"depth_mm", "nrmlz":False, "bxcx":True,"dist":"weibull_min"},
    {"variable":"depth_mm", "nrmlz":False, "bxcx":True,"dist":"gamma"}
]

lst_d_top3_max_mm_per_hour = [
    {"variable":"max_mm_per_hour", "nrmlz":True, "bxcx":True,"dist":"t"},
    {"variable":"max_mm_per_hour", "nrmlz":False, "bxcx":True,"dist":"t"},
    {"variable":"max_mm_per_hour", "nrmlz":True, "bxcx":True,"dist":"norm"}
]

lst_d_top3_mean_mm_per_hr = [
    {"variable":"mean_mm_per_hr", "nrmlz":True, "bxcx":True,"dist":"exponpow"},
    {"variable":"mean_mm_per_hr", "nrmlz":False, "bxcx":True,"dist":"genextreme"},
    {"variable":"mean_mm_per_hr", "nrmlz":False, "bxcx":True,"dist":"weibull_min"}
]

lst_d_top3_max_surge_ft = [
    {"variable":"max_surge_ft", "nrmlz":False, "bxcx":False,"dist":"genextreme"},
    {"variable":"max_surge_ft", "nrmlz":False, "bxcx":True,"dist":"t"},
    {"variable":"max_surge_ft", "nrmlz":True, "bxcx":True,"dist":"t"}
]

lst_d_top3_surge_peak_after_rain_peak_min = [
    {"variable":"surge_peak_after_rain_peak_min", "nrmlz":True, "bxcx":True,"dist":"t"},
    {"variable":"surge_peak_after_rain_peak_min", "nrmlz":True, "bxcx":False,"dist":"t"},
    {"variable":"surge_peak_after_rain_peak_min", "nrmlz":True, "bxcx":False,"dist":"pearson3"}
]

# Final choice
lst_d_selection = [
    lst_d_top3_depth_mm[2], # [2 0 1] = 12l; [0 0 1] = 14; [0 1 1] = 13, [0 2 1] = 24; [2 1 1] = 8
    lst_d_top3_max_mm_per_hour[1],
    lst_d_top3_mean_mm_per_hr[1],
    lst_d_top3_max_surge_ft[0],
    lst_d_top3_surge_peak_after_rain_peak_min[2]
]

# Option 1
# lst_d_selection = [
#     lst_d_top3_depth_mm[0],
#     lst_d_top3_max_mm_per_hour[0],
#     lst_d_top3_mean_mm_per_hr[0],
#     lst_d_top3_max_surge_ft[0],
#     lst_d_top3_surge_peak_after_rain_peak_min[0]
# ]

# Option 2
# lst_d_selection = [
#     lst_d_top3_depth_mm[1],
#     lst_d_top3_max_mm_per_hour[1],
#     lst_d_top3_mean_mm_per_hr[1],
#     lst_d_top3_max_surge_ft[1],
#     lst_d_top3_surge_peak_after_rain_peak_min[1]
# ]

# Option 3
# lst_d_selection = [
#     lst_d_top3_depth_mm[2],
#     lst_d_top3_max_mm_per_hour[2],
#     lst_d_top3_mean_mm_per_hr[2],
#     lst_d_top3_max_surge_ft[2],
#     lst_d_top3_surge_peak_after_rain_peak_min[2]
# ]



def return_selection(dic):
    v = dic["variable"]
    normalized = dic["nrmlz"]
    boxcox = dic["bxcx"]
    dist = dic['dist']
    idx_selection = (df_perf.data == v) & (df_perf.normalized == normalized) & (df_perf.boxcox == boxcox) & (df_perf.fx == dist)
    return df_perf[idx_selection]

lst_selections = []
for d in lst_d_selection:
    lst_selections.append(return_selection(d))

df_selection = pd.concat(lst_selections)
df_selection.to_csv(f_selection, index = False)
# creating dataframe of CDF values of observed data and SST data for copula fitting and conditional simulation
# df_selection = pd.read_csv(f_selection, index_col=0)

df_observations = df_compound_summary.loc[:, vars_all]

def return_cdf_for_var(df, varname, df_selection = df_selection, plot = False):
        s_dist_info = df_selection[df_selection.data == varname].T.squeeze()
        s_var_notrns = df[varname].copy()

        dist = s_dist_info["fx"]
        # distribution parameters
        shape = s_dist_info["shape"]
        loc = s_dist_info["loc"]
        scale = s_dist_info["scale"]

        nrmlz = s_dist_info["normalized"]
        nrmlz_mean = s_dist_info["normalize_mean"]
        nrmlz_std = s_dist_info["normalize_std"]
        boxcox = s_dist_info["boxcox"]
        boxcox_lambda = s_dist_info["boxcox_lambda"]
        scalar_shift = s_dist_info["scalar_shift"]

        if "log_" in s_dist_info["fx"]:
            log = True
            dist = dist.split("log_")[-1]
        else:
            log = False

        for f in fxs:
            fx_name = f["args"]["fx"].name
            if fx_name != dist:
                continue
            else:
                fx = f["args"]["fx"]
                fx_fit = f["function"]

        # perform transformations
        s_var_fit = s_var_notrns.copy()

        if nrmlz == True:
            s_var_fit = (s_var_fit - nrmlz_mean)/nrmlz_std

        if boxcox == True:
            s_var_fit = stats.boxcox(s_var_fit+scalar_shift, lmbda=boxcox_lambda)
            s_var_fit = pd.Series(s_var_fit)

        if log==True:
            s_var_fit=np.log(s_var_fit)

        # x_fit = pd.Series(fx.ppf(df_emp, shape, loc, scale), name = "x_fit")
        if np.isnan(shape): # 2 parameter
            s_var_fit_cdf = pd.Series(fx.cdf(s_var_fit, loc, scale), name = varname)
        else: # 3 parameter
            s_var_fit_cdf = pd.Series(fx.cdf(s_var_fit, shape, loc, scale), name = varname)
        if plot:
            args = {"plot":True, "recurrence_intervals":None, "xlab":varname, "normalized":nrmlz, "boxcox":boxcox, "scalar_shift":scalar_shift}
            out = fx_fit(s_var_fit, fx=fx, **args)
            aic = s_dist_info["aic"]
            if aic < 0:
                aic_desc = "n" + str(round(aic, 1)*-1)
            else:
                aic_desc = str(round(aic, 1))
            figname = "{}_aic-{}_cvmpval-{}_dist-{}.png".format(v, aic_desc, round(s_dist_info["cvm_pval"], 2), s_dist_info["fx"])
            plt.savefig(fldr_plt_selected + figname, bbox_inches='tight')
        return s_var_fit_cdf

# create cdf values of observed data for copula fitting
lst_cdf_obs_vals = []
for v in vars_all:
    s_var_fit_cdf = return_cdf_for_var(df=df_compound_summary, varname=v, df_selection = df_selection, plot = True)
    lst_cdf_obs_vals.append(s_var_fit_cdf)
df_obs_cdfs = pd.concat(lst_cdf_obs_vals, axis = 1)
df_obs_cdfs.to_csv(f_cdfs_obs)

# create cdf values of SST rainfall data for conditional simulatoin from copula
lst_cdf_sst_vals = []
for v in ['depth_mm', 'mean_mm_per_hr', 'max_mm_per_hour']:
    s_var_fit_cdf = return_cdf_for_var(df=df_sst_event_summaries, varname=v, df_selection = df_selection)
    lst_cdf_sst_vals.append(s_var_fit_cdf)
df_sst_cdfs = pd.concat(lst_cdf_sst_vals, axis = 1)
df_sst_cdfs.to_csv(f_cdfs_sst)

# plot comparisons of the histogram of the cdf values (hopefully they are roughly uniform)
for v in ['depth_mm', 'mean_mm_per_hr', 'max_mm_per_hour']:
    fig, ax = plt.subplots(1, 2, dpi=300, figsize=[8, 4])
    # plot sst rainfall
    df_sst_cdfs[v].hist(ax = ax[0])
    ax[0].set_ylabel("count")
    ax[0].set_xlabel("cdf value")
    ax[0].set_title("SST Rainfall")
    # plot observed rainfall
    df_obs_cdfs[v].hist(ax = ax[1])
    ax[1].set_xlabel("cdf value")
    ax[1].set_title("Observed Rainfall")
    fig.text(.5, 1, v, ha='center')
    plt.savefig("plots/b_comparing_sst_to_obs_cdf_for_{}.png".format(v))

