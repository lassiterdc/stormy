#%% loading packages
import pandas as pd
import scipy
from scipy import stats
# from scipy.stats import genextreme
# from scipy.stats import weibull_min
# from scipy.stats import gumbel_r
# from scipy.stats import lognorm
# from scipy.stats import pearson3 
import numpy as np
import matplotlib.pyplot as plt


#%% functions
def fit_3_param_pdf(df, fx, recurrence_intervals,
                    plot = True, data_lab = None, c = None,
                    log=False, xlab=None):
    """df must be a dataframe with only one columns or a series"""
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
    df = df.sort_values().reset_index(drop=True)

    cap = fx.name
    if log==True:
        df=np.log(df).sort_values().reset_index(drop=True)
        cap = 'log({})'.format(fx.name)
    
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

    df_cdf = pd.concat([df_untrns, x_fit_untrns, y_cdf, df_emp], axis = 1, ignore_index=True)
    df_cdf.columns = [df.name, x_fit.name, y_cdf.name, df_emp.name]
    df_cdf.sort_values(df_emp.name, inplace=True)
    df_cdf.reset_index(drop=True, inplace=True)


    msdi = comp_msdi(df_untrns, x_fit_untrns)
    madi = comp_madi(df_untrns, x_fit_untrns)
    
    if recurrence_intervals is not None:
        quantiles = []
        for t in recurrence_intervals:
            q = 1-1/t # area to the left of the recurrence interval
            x = fx.ppf(q, shape, loc, scale)
            if log == True:
                quantiles.append(np.exp(x))
            else:
                quantiles.append(x)
    
    stat, p_val = stats.ks_2samp(df, x_fit)

    out = {"msdi":msdi, "madi":madi, "ks_pval":p_val ,"fitted_quantiles":quantiles,
           "params":(shape,loc,scale), "df":df_cdf}
    
    if plot == True:
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=[6, 8])

        y_pdf = fx.pdf(x_fit, shape, loc, scale)

        if log == True:
            xlab_pdf = "log ({})".format(xlab)
        else:
            xlab_pdf = xlab

        ax[0].plot(x_fit, y_pdf, label = "fitted", color=c)
        ax[0].hist(df, density=True, histtype="stepfilled",
                alpha=0.2, label=" empirical", color = c, bins=20)
        ax[0].legend()
        ax[0].set_ylabel("Probability Density")
        ax[0].set_xlabel(xlab_pdf)
        ax[0].set_title("Probability Density Function")
        
        # df_emp = scipy.stats.mstats.plotting_positions(df, 0.44,0.44) # gringerton plotting position
        
        ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_emp"], label = "empirical", color = c)
        ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_fit"], label = "fitted", color=c, ls="--")
        ax[1].set_ylabel("Cumulative Probability")
        ax[1].set_title("Cumulative Density Function")
        ax[1].set_xlabel(xlab)
        
        
        fig.set_tight_layout(True)
        fig.text(.45, 1, cap, ha='left')
        
        out["figure"] = fig
        out["axis"] = ax,
    
    return out
    
def fit_2_param_pdf(df, fx, recurrence_intervals=None,
                    plot = True, data_lab = None, c = None,
                    log=False, xlab=None, caption=False):
    # https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.genextreme.html#scipy.stats.genextreme
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
    # xlab = "log ({})".format(xlab)
    df = df.sort_values()
    
    cap = fx.name
    out = {}

    if log==True:
        df=np.log(df).sort_values().reset_index(drop=True)
        cap = 'log({})'.format(fx.name)
    
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

    df_cdf = pd.concat([df_untrns, x_fit_untrns, y_cdf, df_emp], axis = 1, ignore_index=True)
    df_cdf.columns = [df.name, x_fit.name, y_cdf.name, df_emp.name]
    df_cdf.sort_values(df_emp.name, inplace=True)
    df_cdf.reset_index(drop=True, inplace=True)

    msdi = comp_msdi(df_untrns, x_fit_untrns)
    madi = comp_madi(df_untrns, x_fit_untrns)
    
    stat, p_val = stats.ks_2samp(df, x_fit)
    out["msdi"] = msdi
    out["madi"] = madi
    out["ks_pval"] = p_val
    out["params"] = (loc,scale)

    
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

    
    if plot == True:
        # x = np.linspace(fx.ppf(0.001, loc, scale),
        #                 fx.ppf(0.999, loc, scale), 100)
        
        y_pdf = fx.pdf(x_fit, loc, scale)
        
        fig, ax = plt.subplots(2, 1, dpi=300, figsize=[6, 8])

        if log == True:
            xlab_pdf = "log ({})".format(xlab)
        else:
            xlab_pdf = xlab
    
        ax[0].plot(x_fit, y_pdf, label = "fitted", color=c)
        ax[0].hist(df, density=True, histtype="stepfilled",
                alpha=0.2, label=" empirical", color = c, bins=20)
        ax[0].legend()
        ax[0].set_ylabel("Probability Density")
        ax[0].set_xlabel(xlab_pdf)
        ax[0].set_title("Probability Density Function")
        
        # if log == True: # backtransform
        #     # xlab = "log ({})".format(xlab)
        #     df = np.exp(df)
        #     x_fit = np.exp(x_fit)
            
        # df_emp = scipy.stats.mstats.plotting_positions(df, 0.44,0.44) # gringerton plotting position
        
        ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_emp"], label = "empirical", color = c)
        ax[1].plot(df_cdf.iloc[:,0], df_cdf["cumprob_fit"], label = "fitted", color=c, ls="--")
        ax[1].set_ylabel("Cumulative Probability")
        ax[1].set_title("Cumulative Density Function")
        ax[1].set_xlabel(xlab)
        
        fig.set_tight_layout(True)
        fig.text(.45, 1, cap, ha='left')
        
        out["figure"] = fig
        out["axis"] = ax
        out["df"] = df_cdf
    
    return out
            
def comp_msdi(empirical, fitted):
    return sum(((empirical - fitted)/empirical)**2)/len(empirical)

def comp_madi(empirical, fitted):
    return sum(abs((empirical - fitted)/empirical))/len(empirical)


#%% testing functions (in progress)
def test():
    df_annual_maxima = pd.read_csv("2015_to_2018_annual_maxima.csv").T
    df_annual_maxima = df_annual_maxima.iloc[1:,:]
    
    df_2015 = df_annual_maxima.iloc[1, :]
    df_2085 = df_annual_maxima.iloc[-1, :]
    
    from scipy.stats import genextreme
    from scipy.stats import weibull_min
    from scipy.stats import gumbel_r
    from scipy.stats import lognorm
    from scipy.stats import pearson3 
    #%%% GEV
    # out = fit_3_param_pdf(df_2015, genextreme, data_lab = "2015", c="b",
    #                           xlab="mm")
    # fit_3_param_pdf(df_2085, genextreme, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm")
    
    # #%%% test Weibull
    # out = fit_3_param_pdf(df_2015, weibull_min, data_lab = "2015", c="b",
    #                           xlab="mm")
    # fit_3_param_pdf(df_2085, weibull_min, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm")
    
    # #%%% test lognormal 3 parameter
    # out = fit_3_param_pdf(df_2015, lognorm, data_lab = "2015", c="b",
    #                           xlab="mm")
    # fit_3_param_pdf(df_2085, lognorm, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm")
    
    # #%%% gumbel
    # out = fit_2_param_pdf(df_2015, gumbel_r, data_lab = "2015", c="b",
    #                           xlab="mm")
    # fit_2_param_pdf(df_2085, gumbel_r, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm")
    # #%%% pearson type III
    # out = fit_3_param_pdf(df_2015, pearson3, data_lab = "2015", c="b",
    #                           xlab="mm")
    # fit_3_param_pdf(df_2085, pearson3, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm")
    # #%%% log pearson type III
    # out = fit_3_param_pdf(df_2015, pearson3, data_lab = "2015", c="b",
    #                           xlab="mm", log=True)
    # fit_3_param_pdf(df_2085, pearson3, ax = out['axis'], fig = fig, data_lab = "2085", c="y",
    #                 xlab="mm", log=True)

