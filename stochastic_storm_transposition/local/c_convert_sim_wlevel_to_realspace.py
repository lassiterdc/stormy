
"""
If any changes were made to the univariate PDF selection or 
parameterization, the R script must be re-run in its entirety before running this script.
"""
#%% loading packages
from __ref_ams_functions import fit_3_param_pdf
from __ref_ams_functions import fit_2_param_pdf
from __ref_ams_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
from scipy.special import inv_boxcox
from scipy import stats
import sys
from tqdm import tqdm
from scipy.stats import bootstrap
import xarray as xr
from _inputs import *

alpha = 0.05
bootstrap_iterations = 100
sst_tstep = 5 # minutes
n_events_per_year_sst = 5

f_selection = "outputs/b_pdf_selections.csv"
# f_cdfs_obs = "outputs/b2_F_of_obs_data-cdfvals.csv"
f_cdfs_sst = "outputs/b3_F_of_sst_data-cdfvals.csv"
f_sst_event_summaries = "outputs/b_sst_event_summaries.csv"
f_observed_compound_event_summaries = "outputs/b_observed_compound_event_summaries.csv"
f_wlevel_cdf_sims_from_copula = "outputs/r_a_sim_wlevel_cdf.csv"
f_simulated_compound_event_summary = "outputs/c_simulated_compound_event_summary.csv"


df_sst_event_summaries = pd.read_csv(f_sst_event_summaries)
df_compound_summary = pd.read_csv(f_observed_compound_event_summaries)
df_selection = pd.read_csv(f_selection)
df_cond_sim = pd.read_csv(f_wlevel_cdf_sims_from_copula)


# create list of functions and variables

vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]

# creating dataframe of real-space water level values from conditional copula simulation
def return_realspace_for_var(df, varname, df_selection = df_selection):
        s_dist_info = df_selection[df_selection.data == varname].T.squeeze()

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

        s_cdf = df[varname].copy()
        if np.isnan(shape): # 2 parameter
            s_var_trns = pd.Series(fx.ppf(s_cdf, loc, scale), name = varname)
        else: # 3 parameters
            s_var_trns = pd.Series(fx.ppf(s_cdf, shape, loc, scale), name = varname)
        s_var_untrns = s_var_trns.copy()

        # perform reverse transformations
        ## these are originally done in the order normalize, then boxcox, then log
        ## so to reverse, I will do inverse log, inverse boxcox, and 
        ## unnormalize
        na_accounted_for = False # DCL WORK
        if log==True:
            # take inverse log
            s_var_untrns=np.exp(s_var_untrns)
            # DCL WORK
            n_na = len(s_var_untrns[s_var_untrns.isna()])
            if n_na > 0 and na_accounted_for == False:
                print("{} NA's have been introduced by inverse log for {}".format(n_na, v))
                na_accounted_for = True
            # END DCL WORK

        if boxcox == True:
            # reverse boxcox and subtract scalar shift which was originally added to remove negatives
            s_var_untrns = inv_boxcox(s_var_untrns, boxcox_lambda) - scalar_shift
            # s_var_untrns = (s_var_untrns * boxcox_lambda + 1) ** (1/boxcox_lambda) - scalar_shift
            # DCL WORK
            n_na = len(s_var_untrns[s_var_untrns.isna()])
            if n_na > 0 and na_accounted_for == False:
                print("{} NA's have been introduced by inverse boxcox for {}".format(n_na, v))
                print("Lambda = {}".format(boxcox_lambda))
                print("scalar_shift = {}".format(scalar_shift))
                # print("Transformed values that are NA on untransform:")
                # print(s_var_trns[s_var_untrns.isna()])
                # print("########################")
                # print("CDF values that are NA on untransform:")
                # print(s_cdf[s_var_untrns.isna()])
                # print("########################")
                na_accounted_for = True
            # END DCL WORK

        if nrmlz == True:
            # multiply by standard deviation and add the mean
            s_var_untrns = s_var_untrns * nrmlz_std + nrmlz_mean
            # DCL WORK
            n_na = len(s_var_untrns[s_var_untrns.isna()])
            if n_na > 0 and na_accounted_for == False:
                print("{} NA's have been introduced by inverse normalize for {}".format(n_na, v))
                na_accounted_for = True
            # END DCL WORK

        return s_var_untrns

# create dataframe of realspace simulations
lst_realspace_copula_sims = []
for v in vars_all:
    s_var_fit_cdf = return_realspace_for_var(df=df_cond_sim, varname=v, df_selection = df_selection)
    lst_realspace_copula_sims.append(s_var_fit_cdf)
df_realspace_copula_sims = pd.concat(lst_realspace_copula_sims, axis = 1)


# figuring out what's going on with some weird values
vars_surge = ["max_surge_ft", "surge_peak_after_rain_peak_min"]
df_simulated_compound_event_summaries = df_sst_event_summaries.join(df_realspace_copula_sims.loc[:, vars_surge])

df_inspection = df_simulated_compound_event_summaries.join(df_cond_sim, rsuffix="_cdf")

# looking at NA values for max_surge_ft
na_max_surge_ft = df_inspection[df_inspection["max_surge_ft"].isna()]
print("There are {} missing max_surge_ft values".format(len(na_max_surge_ft)))

# looking at NA values for surge_peak_after_rain_peak_min
na_surge_lag = df_inspection[df_inspection["surge_peak_after_rain_peak_min"].isna()]
print("There are {} missing surge_peak_after_rain_peak_min values".format(len(na_surge_lag)))

# looking at really big max_surge_ft
yr100_wlevel_ft = 2.26 * 3.28084 # meter * feet per meter
large_max_surge_ft = df_inspection[df_inspection["max_surge_ft"]>=yr100_wlevel_ft]
ax = large_max_surge_ft.max_surge_ft.hist()
ax.set_title("max_surge_ft values greater than 100 yr surge value of {} ft\nn={}".format(round(yr100_wlevel_ft,1), len(large_max_surge_ft)))
plt.savefig("plots/b_final_investigating_large_surges.png".format(v))
#%% visually inspect results
# verify that the rainfall data was properly reverse transformed and that storm order was preserved
vars_rain = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour"]

total_error = (df_sst_event_summaries.loc[:, vars_rain] - df_realspace_copula_sims.loc[:, vars_rain]).abs().sum()

print("The total absolute error in the real-space transformed copula simulated rainfall data (the variable used for conditioning is:")
print(total_error)
print("We want these to be close to 0 to idnicate that the rainfall data was preserved, the reverse transformation was succesfull, and that storm order was preserved.")

#%% fix storm surge and lag statistics (LEFT OFF HERE)
# make the max surge be like 10' or something

# make the surge lags something reasonable
#%% plot
for v in vars_all:
    fig, ax = plt.subplots(1, 2, dpi=300, figsize=[8, 4], sharex=True)
    # plot simulated
    bins=np.histogram(df_simulated_compound_event_summaries[v], bins=20)[1] #get the bin edges
    df_simulated_compound_event_summaries[v].hist(ax = ax[0], bins=bins)
    ax[0].set_ylabel("count")
    ax[0].set_xlabel("value")
    ax[0].set_title("Simulated")
    # plot observed data
    df_compound_summary[v].hist(ax = ax[1], bins=bins)
    ax[1].set_xlabel("value")
    ax[1].set_title("Observed")
    fig.text(.5, 1, v, ha='center')
    plt.savefig("plots/b_final_comparing_sst_to_obs_cdf_for_{}.png".format(v))



#%% concatenate with sst event summary table and export
df_simulated_compound_event_summaries.to_csv(f_simulated_compound_event_summary, index = False)

