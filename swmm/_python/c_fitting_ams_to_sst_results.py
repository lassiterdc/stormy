#%% loading packages
from __ref_ams_functions import fit_3_param_pdf
from __ref_ams_functions import fit_2_param_pdf
from scipy.stats import genextreme
from scipy.stats import weibull_min
from scipy.stats import gumbel_r
from scipy.stats import lognorm
from scipy.stats import pearson3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy
from scipy import stats
import sys
from tqdm import tqdm
from scipy.stats import bootstrap
from _inputs import c_ams_with_sst

f_sst_annual_max_volumes, ks_alpha, bootstrap_iterations, volume_units, f_sst_recurrence_intervals, sst_ams_conf_interval, recurrence_intervals = c_ams_with_sst()


df_annual_maxima = pd.read_csv(f_sst_annual_max_volumes)["total_flooding_1e+06m3"]

df_annual_maxima = pd.DataFrame(dict(year = np.arange(0, 1000))).join(df_annual_maxima)

df_annual_maxima.fillna(0.001, inplace=True)

df_annual_maxima.drop(columns="year", inplace=True)

df_annual_maxima = df_annual_maxima["total_flooding_1e+06m3"]

#%% test
# def comp_msdi(empirical, fitted):
#     return sum(((empirical - fitted)/empirical)**2)/len(empirical)

# def comp_madi(empirical, fitted):
#     return sum(abs((empirical - fitted)/empirical))/len(empirical)

# df = df_annual_maxima
# fx = pearson3
# recurrence_intervals = [1.001,2,10,100]
# plot = False
# log = True
# xlab = "cubic meters"
# data_lab = "data_lab"
# ax = None
# c = None

# # def fit_3_param_pdf(df, fx, recurrence_intervals,
# #                     plot = True, ax = None,
# #                     fig=None, data_lab = None, c = None,
# #                     log=False, xlab=None):
# #     # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
# #     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
# df = df.sort_values()

# cap = fx.name
# if log==True:
#     df=np.log(df).sort_values().reset_index(drop=True)
#     cap = 'log({})'.format(fx.name)

# shape, loc, scale = fx.fit(df)

# df_emp = pd.Series(scipy.stats.mstats.plotting_positions(df, 0.44,0.44)) # gringerton plotting position

# x_fit = fx.ppf(df_emp, shape, loc, scale)

# if log == True: # compute performance using untransformed fit
#     df_untrns = np.array(np.exp(df))
#     x_fit_untrns = np.array(np.exp(x_fit))
#     msdi = comp_msdi(df_untrns, x_fit_untrns)
#     madi = comp_madi(df_untrns, x_fit_untrns)

# else:
#     msdi = comp_msdi(df, x_fit)
#     madi = comp_madi(df, x_fit)

# if recurrence_intervals is not None:
#     quantiles = []
#     for t in recurrence_intervals:
#         q = 1-1/t # area to the left of the recurrence interval
#         x = fx.ppf(q, shape, loc, scale)
#         if log == True:
#             quantiles.append(np.exp(x))
#         else:
#             quantiles.append(x)

# out = {"msdi":msdi, "madi":madi, "fitted_quantiles":quantiles}

# y_pdf = fx.pdf(x_fit, shape, loc, scale)
# y_cdf = fx.cdf(x_fit, shape, loc, scale)


# # ks_result = stats.kstest(df, lambda x: fx.cdf(x, shape,loc,scale))
# # print("ks result with callable")
# # print(ks_result)

# stat, p_val = stats.ks_2samp(df, x_fit)
# print("ks result with callable")
# print(ks_result)

# if plot == True:
#     # x = np.linspace(fx.ppf(0.001, shape, loc, scale),
#     #                 fx.ppf(0.999, shape, loc, scale), 100)

    
#     fig, ax = plt.subplots(2, 1, dpi=300, figsize=[6, 8])

#     if log == True:
#         xlab_pdf = "log ({})".format(xlab)
#     else:
#         xlab_pdf = xlab
    
#     ax[0].plot(x_fit, y_pdf, label = "fitted", color=c)
#     ax[0].hist(df, density=True, histtype="stepfilled",
#             alpha=0.2, label=" empirical", color = c, bins=20)
#     ax[0].legend()
#     ax[0].set_ylabel("Probability Density")
#     ax[0].set_xlabel(xlab_pdf)
#     ax[0].set_title("Probability Density Function")
    
#     if log == True: # backtransform
#         # xlab = "log ({})".format(xlab)
#         df = np.exp(df)
#         x_fit = np.exp(x_fit)
        
#     # df_emp = scipy.stats.mstats.plotting_positions(df, 0.44,0.44) # gringerton plotting position
    
#     ax[1].plot(df.sort_values().reset_index(drop=True), df_emp.sort_values().reset_index(drop=True), label = "empirical", color = c)
#     ax[1].plot(x_fit, y_cdf, label = "fitted", color=c, ls="--")
#     ax[1].set_ylabel("Cumulative Probability")
#     ax[1].set_title("Cumulative Density Function")
#     ax[1].set_xlabel(xlab)
    
    
#     fig.set_tight_layout(True)
#     fig.text(.45, 1, cap, ha='left')
    
#     out = {"figure":fig, "axis":ax,
#             "msdi":msdi, "madi":madi,
#             "fitted_quantiles":quantiles}
        
#         return out





#%% comparing fits
# dfs = []
# create dataframe for each year
# for i in np.arange(0,len(df_annual_maxima)):
#     dfs.append(df_annual_maxima.iloc[i, :].sort_values())

# create dictionaries with the arguments to the fitting functions
common_args = {"plot":False, "recurrence_intervals":recurrence_intervals,
               "xlab":"Flood Volume ($m^3$)"}

gev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme}}

loggev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme, "log":True}}

weibull = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_min}}

logweibull = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_min, "log":True}}

gumbel = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_r}}

loggumbel = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_r, "log":True}}

lognormal = {"function":fit_3_param_pdf, 
             "args":{"fx":lognorm}}

p3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3}}

lp3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3, "log":True}}

fxs = [gev, loggev, weibull, logweibull, gumbel, loggumbel, lognormal, p3, lp3]

# fxs = [weibull, p3]

# fxs = [p3]

# fxs = [lp3]
lst_df_perf = []
df_perf = pd.DataFrame()
# fit each pdf to the data and store the performance metrics in a .csv
s_tot_flding = df_annual_maxima
ind = -1
for f in fxs:
    ind += 1
    out = f["function"](s_tot_flding, **f['args'], **common_args)
    fx_name = f["args"]["fx"].name
    try:
        if f["args"]["log"] == True:
            fx_name = 'log_' + fx_name
    except:
        pass
    # create dictionary with recurrence intervals
    qs = {}
    quants = out["fitted_quantiles"]
    count = -1
    for t in recurrence_intervals:
        count += 1
        qs[t] = quants[count]
    # create dictionary for performance metrics
    d = {"fx":fx_name, "madi":out["madi"],"msdi":out["msdi"], "ks_pval":out["ks_pval"]}
    
    d.update(qs)
    
    lst_df_perf.append(pd.DataFrame(index=[ind], data=d))

#%% evaluating performance
df_perf = pd.concat(lst_df_perf)


# seeing which fit has lowest madi and lowest msdi for each year
df_perf_pass_ks = df_perf[df_perf.ks_pval > ks_alpha]

idx_msdi = df_perf_pass_ks.loc[:, ["msdi"]].idxmin()["msdi"]
idx_madi = df_perf_pass_ks.loc[:, ["madi"]].idxmin()["madi"]
idx_ks_pval = df_perf_pass_ks.loc[:, ["ks_pval"]].idxmax()["ks_pval"]

df_perf_pass_ks_msdi = df_perf.iloc[idx_msdi, :]
df_perf_pass_ks_madi = df_perf.iloc[idx_madi, :]
df_perf_pass_ks_ks = df_perf.iloc[idx_ks_pval, :]


#%% plotting two best 
# df_annual_maxima_scaled = df_annual_maxima / volume_units

out_w = fit_3_param_pdf(df_annual_maxima, weibull_min, recurrence_intervals,
                 xlab = "Flood Volume ($10^6m^3$)")

out_p3 = fit_3_param_pdf(df_annual_maxima, pearson3, recurrence_intervals,
                 xlab = "Flood Volume ($10^6m^3$)")


shape, loc, scale = out_p3['params']


#%% set up for bootstrapping
def conv_return_pd_to_quant(return_pd):
    return 1 - 1/return_pd

def conv_quant_to_return_pd(quant):
    return 1/(1-quant)


fx = pearson3

s_return_pds = pd.Series(np.linspace(1.01, 500, num=len(df_annual_maxima)), name = 'return_periods_yrs')
s_quants = s_return_pds.apply(conv_return_pd_to_quant)
s_quants.name = "quantiles"

base_fit_flds = pd.Series(fx.ppf(s_quants, shape, loc, scale), name = "base_fit")

# base_cdf = pd.Series(pearson3.cdf(s_flds, shape, loc, scale), name = 'base_estimate')

units="{:.0e}".format(int(volume_units))
base_estimate = pd.Series(out_p3['fitted_quantiles'], index = recurrence_intervals, name = "estimate_{}".format(df_annual_maxima.name))


# going with log pearson because it had the best madi and msdi
#%% bootstrapping
df_quants = pd.DataFrame(dict(recurrence_intervals=recurrence_intervals))

# these are flood values to get the fitted cdf value for; I will be getting the upper and lower bounds at each value
# for the plot

lst_bs_samps = []
lst_rec_ests = []

for bs in tqdm(np.arange(bootstrap_iterations)):
    df_resampled = df_annual_maxima.sample(n=len(df_annual_maxima), replace=True).reset_index(drop=True)
    out_p3 = fit_3_param_pdf(df_resampled, fx, recurrence_intervals, plot=False)
    # df_cdf = out_p3['df']
    shape, loc, scale = out_p3['params']
    new_fit_flds = pd.Series(fx.ppf(s_quants, shape, loc, scale), name = "fit{}".format(bs))
    # new_cdf = pd.Series(fx.cdf(s_flds, shape, loc, scale), name = 'fit{}'.format(bs))
    if bs == 0:
        df_flds = new_fit_flds
        # s_max_cdf = new_cdf
    else:
        df_flds = pd.concat([df_flds, new_fit_flds], axis=1)
        # s_min_cdf = pd.concat([new_cdf, s_min_cdf], axis=1).min(axis = 1)

    lst_rec_ests.append(out_p3['fitted_quantiles'])
    lst_bs_samps.append(bs)
#%%
lower_quant = (1-sst_ams_conf_interval)/2
upper_quant = sst_ams_conf_interval + lower_quant

s_lower_quant = df_flds.quantile(q = lower_quant, axis = 1)
s_lower_quant.name = "quantile_{}".format(round(lower_quant,2))

# lower_quant.name = "{}perc_CI_lowerbound".format(int(sst_ams_conf_interval*100))
s_upper_quant = df_flds.quantile(q = upper_quant, axis = 1)
s_upper_quant.name = "quantile_{}".format(round(upper_quant,2))

# df_return_periods = pd.concat([base_estimate, upper_quant, lower_quant], axis=1)
# df_return_periods = df_return_periods.reset_index().rename(columns = dict(index="return_interval"))

df_fld_return_pds_upper_and_lower = pd.concat([s_return_pds, s_quants, base_fit_flds, s_lower_quant, s_upper_quant], axis=1)

# df_quants.set_index("recurrence_intervals", inplace=True)
#%% process
# df_quants = pd.DataFrame(lst_rec_ests, columns = recurrence_intervals)
# lower_quant = (1-sst_ams_conf_interval)/2
# upper_quant = sst_ams_conf_interval + lower_quant

# lower_quant = df_quants.quantile(q = lower_quant, axis = 0)
# lower_quant.name = "{}perc_CI_lowerbound".format(int(sst_ams_conf_interval*100))
# higher_quant = df_quants.quantile(q = upper_quant, axis = 0)
# higher_quant.name = "{}perc_CI_upperbound".format(int(sst_ams_conf_interval*100))

# df_return_periods = pd.concat([base_estimate, higher_quant, lower_quant], axis=1)
# df_return_periods = df_return_periods.reset_index().rename(columns = dict(index="return_interval"))

df_fld_return_pds_upper_and_lower.to_csv(f_sst_recurrence_intervals, index = False)




















