
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
from sklearn.cluster import KMeans

alpha = 0.05
bootstrap_iterations = 100
sst_tstep = 5 # minutes
n_events_per_year_sst = 5
k_selection = 5
estimate_k = False # if true, run code for generating elbow method plot
sst_recurrence_intervals = [0.5, 1, 2, 5, 10, 25, 50, 100]

f_selection = "outputs/b_pdf_selections.csv"
f_cdfs_obs = "outputs/b2_F_of_obs_data-cdfvals.csv"
# f_cdfs_sst = "outputs/b3_F_of_sst_data-cdfvals.csv"
f_sst_event_summaries = "outputs/b_sst_event_summaries.csv"
f_observed_compound_event_summaries = "outputs/b_observed_compound_event_summaries.csv"
f_wlevel_cdf_sims_from_copula = "outputs/r_a_sim_wlevel_cdf.csv"
f_wlevel_cdf_sims_from_copula_with_multvar_empcdf = "outputs/r_a_sim_wlevel_cdf_with_multivariate_empcdf.csv"
f_simulated_compound_event_summary = "outputs/c_simulated_compound_event_summary.csv"
f_observed_compound_event_summaries_with_k = "outputs/c_observed_compound_event_summaries_with_k.csv"

# load cdf values for k-means clustering
df_obs_cdf_vals = pd.read_csv(f_cdfs_obs, index_col=0)
# df_compound_cdf_vals = pd.read_csv(f_cdfs_sst, index_col=0)
# load event summaries
df_sst_event_summaries = pd.read_csv(f_sst_event_summaries)
df_compound_summary = pd.read_csv(f_observed_compound_event_summaries, index_col=0)
# load selected marginal distribution information
df_selection = pd.read_csv(f_selection)
# load conditional simulations from R script
# df_cond_sim = pd.read_csv(f_wlevel_cdf_sims_from_copula)
df_cond_sim = pd.read_csv(f_wlevel_cdf_sims_from_copula_with_multvar_empcdf)

# create list of functions and variables

vars_all = ["depth_mm", "mean_mm_per_hr", "max_mm_per_hour", "max_surge_ft", "surge_peak_after_rain_peak_min"]

def return_period_to_quantile(n_events_per_year_sst, return_periods):
    total_years = 5000 # this number doesn't matter
    total_events = total_years*n_events_per_year_sst
    quants = []
    for return_pd in return_periods:
        expected_num_of_storms = total_years / return_pd
        quant = 1 - expected_num_of_storms / total_events
        quants.append(quant)
    return quants

return_period_quantiles = return_period_to_quantile(n_events_per_year_sst,sst_recurrence_intervals)



# creating dataframe of real-space water level values from conditional copula simulation
def return_realspace_for_var(df, varname, df_selection = df_selection, calc_return_pds = False, return_period_quantiles = None):
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
        if calc_return_pds:
            s_cdf = return_period_quantiles
        else:
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
# lst_realspace_return_periods = []
for v in vars_all:
    s_var_fit_cdf = return_realspace_for_var(df=df_cond_sim, varname=v, df_selection = df_selection)
    lst_realspace_copula_sims.append(s_var_fit_cdf)

    # s_var_fit_cdf_returns = return_realspace_for_var(df=df_cond_sim, varname=v, df_selection = df_selection, calc_return_pds = True, return_period_quantiles = return_period_quantiles)
    # lst_realspace_return_periods.append(s_var_fit_cdf_returns)
df_realspace_copula_sims = pd.concat(lst_realspace_copula_sims, axis = 1)
# add columns for empirical multivariate_cdf
df_realspace_copula_sims["n_emp_multivar_cdf"] = df_cond_sim["N.cum"]
df_realspace_copula_sims["emp_multivar_cdf"] = df_cond_sim["CDF"]
# figuring out what's going on with some weird values
vars_surge = ["max_surge_ft", "surge_peak_after_rain_peak_min"]
df_simulated_compound_event_summaries = df_sst_event_summaries.join(df_realspace_copula_sims.loc[:, vars_surge+["n_emp_multivar_cdf","emp_multivar_cdf"]])

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

#%% lk-means: determining K
vars_k = list(df_obs_cdf_vals.columns)
df_vars_stormclass = df_obs_cdf_vals.loc[:, vars_k]

def compute_msd_kmeans(df_data, df_pred, kmeans_centers, validate = False, kmeans_computed_ssd = None):
    df_data_w_pred = df_data.copy()
    df_data_w_pred['k'] = df_pred
    df_cluster_centers = pd.DataFrame(kmeans_centers)
    df_cluster_centers.columns = df_data.columns
    lst_ssd = []
    lst_msd = []
    for cluster_id, row in df_cluster_centers.iterrows():
        df_data_cluster = df_data[df_data_w_pred.k == cluster_id]
        # compute sum of square differences
        ssd = (df_data_cluster.subtract(row, axis = "columns")**2).sum(axis = 'index').sum()
        msd = (df_data_cluster.subtract(row, axis = "columns")**2).mean(axis = 'index').mean()
        lst_ssd.append(ssd)
        lst_msd.append(msd)
    total_ssd = sum(lst_ssd)
    total_msd = np.mean(lst_msd)
    if validate:
        if np.isclose(kmeans_computed_ssd - total_ssd, 0) == False:
            sys.exit("WARNING: Problem in the way SSD is calculated for K-means clustering")
    return total_msd

if estimate_k:
    nbs = 200
    # vars_k = ["depth_mm", "max_mm_per_hour", "max_surge_ft"]
    # df_vars_stormclass_scaler = preprocessing.StandardScaler().fit(df_vars_stormclass)
    # df_vars_stormclass_scaled = df_vars_stormclass_scaler.transform(df_vars_stormclass)

    lst_msd_fit = [] # sum of squared distances for the fitted values
    lst_msd_test = [] # sum of squared distances for the test
    ks = []
    bs_ids = []
    ks_to_try = 20
    for bs_id in tqdm(np.arange(nbs)):
        df_resampled_fit = df_vars_stormclass.sample(frac = 1, replace = True)
        ids_in_resample = df_resampled_fit.index.unique()
        locs_not_in_resample = ~df_vars_stormclass.index.isin(ids_in_resample)
        df_test = df_vars_stormclass[locs_not_in_resample]
        for k in range(1,ks_to_try):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(df_resampled_fit)
            kmeans_centers = kmeans.cluster_centers_
            pred_fit = kmeans.predict(df_resampled_fit)
            pred_test = kmeans.predict(df_test)

            msd_fit = compute_msd_kmeans(df_resampled_fit, pred_fit, kmeans_centers, validate = True, kmeans_computed_ssd = kmeans.inertia_)
            msd_test = compute_msd_kmeans(df_test, pred_test, kmeans_centers)
            if np.isnan(msd_test):
                print('problem! msd_test is nan')
                break

            lst_msd_fit.append(msd_fit)
            lst_msd_test.append(msd_test)
            ks.append(k)
            bs_ids.append(bs_id)

    df_kmeans_comparison = pd.DataFrame(
        dict(
            k = ks, 
            # bs_id = bs_ids,
            # msd_fit = lst_msd_fit,
            msd_test = lst_msd_test
        )
    )

    ci = 0.9
    ci_lower = (1 - ci)/2
    ci_upper = 1 - (1 - ci)/2

    ci_upper_bnds = df_kmeans_comparison.groupby('k').quantile(ci_upper)
    # ci_upper_bnds.columns = ["90%_CI"]
    ci_lower_bnds = df_kmeans_comparison.groupby('k').quantile(ci_lower)
    # ci_lower_bnds.columns = ["90%_CI"]
    msd_estimate = df_kmeans_comparison.groupby('k').mean()
    # msd_estimate.columns = ["Mean Squared Distance (test)"]

    fig, ax = plt.subplots(dpi = 300)

    ax.plot(ci_upper_bnds, linestyle = '--', color = 'black', label = "90% CI")
    ax.plot(ci_lower_bnds, linestyle = '--', color = 'black')
    ax.plot(msd_estimate, color = 'red', label = "MSD Out-of-bag")
    plt.legend()

    # plt.plot(range(1,ks_to_try), inertias, marker='o')
    plt.title('Selecting K Using Elbow Method (bootstrap n = {})'.format(nbs))
    plt.xlabel('Number of clusters')
    plt.ylabel('Mean Squared Distance to Centroid')

    ax.axvline(x = k_selection)
    text_y_loc = ci_upper_bnds.quantile(0.9).values
    ax.text(x = 5.3, y = text_y_loc, s = "Choosing k = {}".format(k_selection))

    plt.savefig("plots/c_kmeans_k_selection.png")


#%% classifying simulated events
kmeans = KMeans(n_clusters=k_selection)
kmeans.fit(df_vars_stormclass)

obs_classes = kmeans.predict(df_obs_cdf_vals)
simulated_classes = kmeans.predict(df_cond_sim.loc[:, df_obs_cdf_vals.columns])
#%% Explort simulated and observed compound event summaries with K-means class
df_compound_summary["kmeans_class"] = obs_classes
df_simulated_compound_event_summaries["kmeans_class"] = simulated_classes

df_compound_summary.to_csv(f_observed_compound_event_summaries_with_k, index = False)
df_simulated_compound_event_summaries.to_csv(f_simulated_compound_event_summary, index = False)

