#%% loading packages
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import genextreme
from scipy.stats import weibull_min
from scipy.stats import weibull_max
from scipy.stats import gumbel_r
from scipy.stats import gumbel_l
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import pearson3
from scipy.stats import t
from scipy.stats import chi
from scipy.stats import genpareto
from scipy.stats import gamma
from scipy.stats import exponpow
import numpy as np
import matplotlib.pyplot as plt


#%% functions
def fit_3_param_pdf(df, fx, recurrence_intervals = None,
                    plot = False, data_lab = None, c = None,
                    log=False, xlab=None, normalized = False, boxcox = False, scalar_shift = 0,
                    nbs = 1):
    """df must be a dataframe with only one columns or a series"""
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
    if xlab is None:
        xlab = "observations"
    df_original = df.sort_values().reset_index(drop=True)
    df_original.name = "x_obs"

    dist = fx.name
    out = {}
    
    if log==True:
        df_original=np.log(df_original).sort_values().reset_index(drop=True)
        dist_for_labels = 'log({})'.format(fx.name)
    else:
        dist_for_labels = dist
    
    lst_shapes = []
    lst_locs = []
    lst_scales = []
    lst_cvm_pval = []
    lst_aic = []
    lst_msdi = []
    lst_madi = []
    lst_ks_pval = []

    for bs_id in np.arange(nbs):
        if nbs == 1: # this means that we aren't actually doing any bootstrapping
            df = df_original.copy()
        else:
            df = df_original.sample(n = len(df_original), replace = True, ignore_index = True)

        shape, loc, scale = fx.fit(df)   

        df_emp = pd.Series(scipy.stats.mstats.plotting_positions(df, 0.44,0.44), name = 'cumprob_emp') # gringerton plotting position
        x_fit = pd.Series(fx.ppf(df_emp, shape, loc, scale), name = "x_fit")
        y_cdf = pd.Series(fx.cdf(df, shape, loc, scale), name = 'cumprob_fit')

        if log == True: # backtransform
            df_untrns = np.exp(df)
            x_fit_untrns = np.exp(x_fit)
        else:
            df_untrns = df
            x_fit_untrns = x_fit

        # compute log likelihood and aic
        n_params = 3
        if normalized == True:
            n_params += 2 # estiamted mean and standard deviation
            xlab = "normalized({})".format(xlab)
        if scalar_shift != 0:
            xlab = xlab + " + {}".format(str(round(scalar_shift, 2)))
        if boxcox == True:
            n_params += 1 # estimated lambda
            xlab = "boxcox({})".format(xlab)

        log_likelihood = np.log(np.prod(fx.pdf(df,shape, loc, scale)))
        aic = 2 * n_params - 2 * log_likelihood
        

        df_cdf = pd.concat([df_untrns, x_fit_untrns, y_cdf, df_emp], axis = 1, ignore_index=True)
        df_cdf.columns = [df.name, x_fit.name, y_cdf.name, df_emp.name]
        df_cdf.sort_values(df_emp.name, inplace=True)
        df_cdf.reset_index(drop=True, inplace=True)

        msdi = comp_msdi(df_untrns, x_fit_untrns)
        madi = comp_madi(df_untrns, x_fit_untrns)
        
        stat, ks_pval = stats.ks_2samp(df, x_fit)
        cvm_output = stats.cramervonmises(df, fx.name, args = (shape, loc, scale))
        cvm_pval = cvm_output.pvalue

        lst_shapes.append(shape)
        lst_locs.append(loc)
        lst_scales.append(scale)
        lst_aic.append(aic)
        lst_msdi.append(msdi)
        lst_madi.append(madi)
        lst_ks_pval.append(ks_pval)
        lst_cvm_pval.append(cvm_pval)
    # use the median of each statistic in the output
    shape = pd.Series(lst_shapes).median()
    loc = pd.Series(lst_locs).median()
    scale = pd.Series(lst_scales).median()
    aic = pd.Series(lst_aic).median()
    msdi = pd.Series(lst_msdi).median()
    madi = pd.Series(lst_madi).median()
    ks_pval = pd.Series(lst_ks_pval).median()
    cvm_pval = pd.Series(lst_cvm_pval).median()

    
    out["msdi"] = msdi
    out["madi"] = madi
    out["ks_pval"] = ks_pval
    out["cvm_pval"] = cvm_pval
    out["n_params"] = n_params
    # out["log_likelihood"] = log_likelihood
    out["aic"] = aic
    out["params"] = (shape,loc,scale)
    # out["df"] = df_cdf

    if nbs == 1: # if this is not 1 (therefore I'm doing bootstrapping) I don't want to calculate these things
        if recurrence_intervals is not None:
            quantiles = []
            for t in recurrence_intervals:
                q = 1-1/t # area to the left of the recurrence interval
                x = fx.ppf(q, shape, loc, scale)
                if log == True:
                    quantiles.append(np.exp(x))
                else:
                    quantiles.append(x)
            out["fitted_quantiles"]  = quantiles


        if plot == True:
            y_pdf = fx.pdf(x_fit, shape, loc, scale)

            fig, ax = plt.subplots(1, 3, dpi=300, figsize=[8, 4])

            if log == True:
                # xlab_pdf = "log ({})".format(xlab)
                xlab_pdf = "log ({})".format("value")
            else:
                # xlab_pdf = xlab
                xlab_pdf = "value"

            ax[0].plot(x_fit, y_pdf, label = "fitted", color=c)
            ax[0].hist(df, density=True, histtype="stepfilled",
                    alpha=0.2, label=" empirical", color = c, bins=20)
            # ax[0].legend()
            ax[0].set_ylabel("Probability Density")
            ax[0].set_xlabel(xlab_pdf)
            ax[0].set_title("Probability Density Function")
                
            ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_emp"], label = "empirical", color = c)
            ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_fit"], label = "fitted", color=c, ls="--")
            ax[1].legend()
            ax[1].set_ylabel("Cumulative Probability")
            ax[1].set_title("Cumulative Density Function")
            # ax[1].set_xlabel(xlab)
            ax[1].set_xlabel("value")
            
            stats.probplot(df, sparams = (shape, loc, scale), dist = dist, plot = ax[2])

            perf_summary = ": AIC = {} | CVM P = {} | KS P = {}".format(round(aic,1), round(cvm_pval,2), round(ks_pval,2))
            fig.text(.5, 1, dist_for_labels+perf_summary, ha='center')
            fig.text(.5, -0.02, "value = {}".format(xlab), ha='center')

            fig.set_tight_layout(True)
            
            out["figure"] = fig
            out["axis"] = ax
    
    return out
    
def fit_2_param_pdf(df, fx, recurrence_intervals=None,
                    plot = False, data_lab = None, c = None,
                    log=False, xlab=None, normalized = False, boxcox = False, scalar_shift = 0,
                    nbs = 1):
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
    # xlab = "log ({})".format(xlab)
    if xlab is None:
        xlab = "observations"
    df_original = df.sort_values().reset_index(drop=True)
    df_original.name = "x_obs"
    
    dist = fx.name
    out = {}

    if log==True:
        df_original=np.log(df_original).sort_values().reset_index(drop=True)
        dist_for_labels = 'log({})'.format(fx.name)
    else:
        dist_for_labels = dist
    
    lst_shapes = []
    lst_locs = []
    lst_scales = []
    lst_cvm_pval = []
    lst_aic = []
    lst_msdi = []
    lst_madi = []
    lst_ks_pval = []

    for bs_id in np.arange(nbs):
        if nbs == 1: # this means that we aren't actually doing any bootstrapping
            df = df_original.copy()
        else:
            df = df_original.sample(n = len(df_original), replace = True, ignore_index = True)

        loc, scale = fx.fit(df) # defaults to MLE
        
        df_emp = pd.Series(scipy.stats.mstats.plotting_positions(df, 0.44,0.44), name = 'cumprob_emp') # gringerton plotting position
        x_fit = pd.Series(fx.ppf(df_emp, loc, scale), name = "x_fit")
        y_cdf = pd.Series(fx.cdf(df, loc, scale), name = 'cumprob_fit')

        if log == True: # backtransform
            df_untrns = np.exp(df)
            x_fit_untrns = np.exp(x_fit)
        else:
            df_untrns = df
            x_fit_untrns = x_fit

        # compute log likelihood and aic
        n_params = 2
        if normalized == True:
            n_params += 2 # estiamted mean and standard deviation
            xlab = "normalized({})".format(xlab)
        if scalar_shift != 0:
            xlab = xlab + " + {}".format(str(round(scalar_shift, 2)))
        if boxcox == True:
            n_params += 1 # estimated lambda
            xlab = "boxcox({})".format(xlab)

        log_likelihood = np.log(np.prod(fx.pdf(df,loc, scale)))
        aic = 2 * n_params - 2 * log_likelihood

        # create dataframe with info for creating cdf
        df_cdf = pd.concat([df_untrns, x_fit_untrns, y_cdf, df_emp], axis = 1, ignore_index=True)
        df_cdf.columns = [df.name, x_fit.name, y_cdf.name, df_emp.name]
        df_cdf.sort_values(df_emp.name, inplace=True)
        df_cdf.reset_index(drop=True, inplace=True)

        msdi = comp_msdi(df_untrns, x_fit_untrns)
        madi = comp_madi(df_untrns, x_fit_untrns)
        
        stat, ks_pval = stats.ks_2samp(df, x_fit)
        cvm_output = stats.cramervonmises(df, fx.name, args = (loc, scale))
        cvm_pval = cvm_output.pvalue

        lst_locs.append(loc)
        lst_scales.append(scale)
        lst_aic.append(aic)
        lst_msdi.append(msdi)
        lst_madi.append(madi)
        lst_ks_pval.append(ks_pval)
        lst_cvm_pval.append(cvm_pval)

    loc = pd.Series(lst_locs).median()
    scale = pd.Series(lst_scales).median()
    aic = pd.Series(lst_aic).median()
    msdi = pd.Series(lst_msdi).median()
    madi = pd.Series(lst_madi).median()
    ks_pval = pd.Series(lst_ks_pval).median()
    cvm_pval = pd.Series(lst_cvm_pval).median()

    out["msdi"] = msdi
    out["madi"] = madi
    out["ks_pval"] = ks_pval
    out["cvm_pval"] = cvm_pval
    out["n_params"] = n_params
    # out["log_likelihood"] = log_likelihood
    out["aic"] = aic
    out["params"] = (loc,scale)
    # out["df"] = df_cdf

    if nbs == 1: # if this is not 1 (therefore I'm doing bootstrapping) I don't want to calculate these things
        if recurrence_intervals is not None:
            quantiles = []
            for t in recurrence_intervals:
                q = 1-1/t # area to the left of the recurrence interval
                x = fx.ppf(q, loc, scale)
                if log == True:
                    quantiles.append(np.exp(x))
                else:
                    quantiles.append(x) 
                out["fitted_quantiles"] = quantiles
        else:
            # print('Skipping quantile calculations because recurrence intervals were not provided....')
            pass

        
        if plot == True:      
            y_pdf = fx.pdf(x_fit, loc, scale)
            
            fig, ax = plt.subplots(1, 3, dpi=300, figsize=[8, 4])

            if log == True:
                # xlab_pdf = "log ({})".format(xlab)
                xlab_pdf = "log ({})".format("value")
            else:
                # xlab_pdf = xlab
                xlab_pdf = "value"
        
            ax[0].plot(x_fit, y_pdf, label = "fitted", color=c)
            ax[0].hist(df, density=True, histtype="stepfilled",
                    alpha=0.2, label=" empirical", color = c, bins=20)
            # ax[0].legend()
            ax[0].set_ylabel("Probability Density")
            ax[0].set_xlabel(xlab_pdf)
            ax[0].set_title("Probability Density Function")
                
            ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_emp"], label = "empirical", color = c)
            ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_fit"], label = "fitted", color=c, ls="--")
            ax[1].legend()
            ax[1].set_ylabel("Cumulative Probability")
            ax[1].set_title("Cumulative Density Function")
            # ax[1].set_xlabel(xlab)
            ax[1].set_xlabel("value")

            stats.probplot(df, sparams = (loc, scale), dist = dist, plot = ax[2])

            perf_summary = ": AIC = {} | CVM P = {} | KS P = {}".format(round(aic,1), round(cvm_pval,2), round(ks_pval,2))
            fig.text(.5, 1, dist_for_labels+perf_summary, ha='center')
            fig.text(.5, -0.02, "value = {}".format(xlab), ha='center')

            fig.set_tight_layout(True)
            # fig.text(0.50, -0.02, perf_summary,
            #          horizontalalignment='center', wrap=True )
            
            out["figure"] = fig
            out["axis"] = ax

    return out
            
def comp_msdi(empirical, fitted):
    return sum(((empirical - fitted)/empirical)**2)/len(empirical)

def comp_madi(empirical, fitted):
    return sum(abs((empirical - fitted)/empirical))/len(empirical)


# create dictionaries with the arguments to the fitting functions
gev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme}}

loggev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme, "log":True}}

weibull_min_dist = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_min}}

logwweibull_min_dist = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_min, "log":True}}

weibull_max_dist = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_max}}

logwweibull_max_dist = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_max, "log":True}}

gumbel_right = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_r}}

gumbel_left = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_l}}

loggumbel = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_r, "log":True}}

norm_dist = {"function":fit_2_param_pdf, 
             "args":{"fx":norm}}

lognormal = {"function":fit_3_param_pdf, 
             "args":{"fx":lognorm}}

student_t = {"function":fit_3_param_pdf, 
             "args":{"fx":t}}

genpareto_dist = {"function":fit_3_param_pdf, 
             "args":{"fx":genpareto}}

chi_dist = {"function":fit_3_param_pdf, 
             "args":{"fx":chi}}

gamma_dist = {"function":fit_3_param_pdf, 
             "args":{"fx":gamma}}

p3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3}}

lp3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3, "log":True}}

sep = {"function":fit_3_param_pdf, 
      "args":{"fx":exponpow}}

fxs = [gev, loggev, weibull_min_dist, logwweibull_min_dist, weibull_max_dist, logwweibull_max_dist,
       gumbel_right, gumbel_left, loggumbel, norm_dist, lognormal, student_t, genpareto_dist, chi_dist, gamma_dist, p3, lp3, sep]