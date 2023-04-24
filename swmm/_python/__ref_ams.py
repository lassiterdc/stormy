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

#%% load data
df_annual_maxima = pd.read_csv("2015_to_2018_annual_maxima.csv").T
df_annual_maxima = df_annual_maxima.iloc[1:,:]

#%% comparing fits
dfs = []
# create dataframe for each year
for i in np.arange(0,len(df_annual_maxima)):
    dfs.append(df_annual_maxima.iloc[i, :].sort_values())

# create dictionaries with the arguments to the fitting functions
recurrence_intervals = [1.001,2,5,10,25,50,100]
common_args = {"plot":False, "recurrence_intervals":recurrence_intervals}

gev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme}}

loggev = {"function":fit_3_param_pdf, 
       "args":{"fx":genextreme}, "log":True}

weibull = {"function":fit_3_param_pdf, 
           "args":{"fx":weibull_min}}

gumbel = {"function":fit_2_param_pdf, 
          "args":{"fx":gumbel_r}}

lognormal = {"function":fit_3_param_pdf, 
             "args":{"fx":lognorm}}

p3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3}}

lp3 = {"function":fit_3_param_pdf, 
      "args":{"fx":pearson3, "log":True}}

fxs = [loggev, gev, weibull, gumbel, lognormal, p3, lp3]

df_perf = pd.DataFrame()
# fit each pdf to the data and store the performance metrics in a .csv
ind = -1
for f in fxs:
    for df in dfs:
        ind += 1
        out = f["function"](df, **f['args'], **common_args)
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
        d = {"date":df.name, "fx":fx_name,
             "madi":out["madi"],"msdi":out["msdi"]}
        
        d.update(qs)
        
        df_perf = df_perf.append(pd.DataFrame(index=[ind], data=d))

df_perf.to_csv("model_performance_comparison.csv")

#%% computing return periods for chunks of time
dfs = []
# create dataframe for each year
bys = [5, 10, 20]
df_gum_mving_wndw = pd.DataFrame()
ind = -1
for by in bys:
    for i in np.arange(by,len(df_annual_maxima), by):
        ind += 1
        df = df_annual_maxima.iloc[(i-by):i, :]
        years = pd.to_datetime(df.index).year.values
        df = df.melt(value_name="precip").precip
        name="{} to {}".format(min(years), max(years))
        out = fit_2_param_pdf(df = df, fx=gumbel_r, plot=False,
                              recurrence_intervals=recurrence_intervals,
                              )
        qs = {}
        quants = out["fitted_quantiles"]
        count = -1
        for t in recurrence_intervals:
            count += 1
            qs[t] = quants[count]
        
        d = {"time_frame":name, "min_year":min(years), "max_year":max(years),
             "by":by, "fx":"gumbel", "madi":out["madi"], "msdi":out["msdi"]}
        d.update(qs)
        df_gum_mving_wndw = df_gum_mving_wndw.append(pd.DataFrame(index=[ind], data=d))
    

    

#%% evaluating performance
# seeing which fit has lowest madi and lowest msdi for each year
idx_msdi = df_perf.loc[:, ["date", "msdi"]].groupby("date").idxmax()["msdi"].values
idx_madi = df_perf.loc[:, ["date", "madi"]].groupby("date").idxmax()["madi"].values

df_perf_msdi = df_perf.iloc[idx_msdi, :]
df_perf_madi = df_perf.iloc[idx_madi, :]

# determining which functions have the best msdi and madi 
msdi_count = df_perf_msdi.groupby("fx").count()["msdi"]
madi_count = df_perf_madi.groupby("fx").count()["madi"]
# gumbel, log pearson, and weibull are the best
msdi_summary = df_perf_msdi.groupby("fx").agg(
    count_best = pd.NamedAgg(column="msdi", aggfunc= pd.Series.count),
    avg_msdi = pd.NamedAgg(column = "msdi", aggfunc = pd.Series.mean))

msdi_summary.sort_values(by = ["avg_msdi"])

madi_summary = df_perf_madi.groupby("fx").agg(
    count_best = pd.NamedAgg(column="madi", aggfunc= pd.Series.count),
    avg_madi = pd.NamedAgg(column = "madi", aggfunc = pd.Series.mean))

madi_summary.sort_values(by = ["avg_madi"])

print(msdi_summary)
print(madi_summary)
# it looks like gumbel and weibull are the best


#%% plotting
# cs = ["tab:orange", "tab:green"] 
# ts = ["10", "100"]

fxs = ["gumbel_r", "weibull_min", "log_pearson3"]
# styles = ["dotted", "dashed", "dashdot"]
cs = ['#e41a1c','#377eb8','#4daf4a']

legend_elements = {}
legend_elements_fx = {}

fix, ax = plt.subplots(1, 1, dpi = 300, figsize = [6, 4])

# i_t = -1
# for t in ts:
#     i_t += 1
#     lab = "{} year".format(t)

#     legend_elements[lab] = Line2D([0], [0], color=cs[i_t], label=lab)
t = "10"
i_fx = -1
for fx in fxs:    
    i_fx += 1
    
    df = df_perf[df_perf.fx == fx]
    lab = fx
    
    years = pd.to_datetime(df.date).dt.year
    ax.plot(years,df[t], color=cs[i_fx],
            linewidth = 1, alpha = 0.9, label = lab)
    
    # legend_elements_fx[fx] = Line2D([0], [0], color=cs[i_fx],
    #                                 label=lab) 

emp_quants = []
q = 1-1/int(t)
for index, row in df_annual_maxima.iterrows():
     df_emp = scipy.stats.mstats.plotting_positions(row, 0.44,0.44)
     ind = np.argmin(abs(df_emp-q))
     emp_quants.append(row[ind])

ax.plot(years,emp_quants, color="black", linewidth = 1, alpha = 0.5,
        label = "empirical", linestyle = "dotted")

# legend_elements.update(legend_elements_fx)

ax.legend(frameon=False, fontsize=8, ncol=2, handletextpad=0.4)

ax.set_ylabel("mm")
ax.set_xlabel("date")
ax.set_title("Distribution comparison for the {} year return period".format(t))
plt.savefig('figures/distribution_comparison.png')
#%%

# df_gumb = df_perf[df_perf.fx == "gumbel_r"]
# years = pd.to_datetime(df_gumb.date).dt.year
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6, 4])
# ax.plot(years,df_gumb['2'], label = "2 year")
# ax.plot(years,df_gumb['10'], label = "10 year")
# ax.plot(years,df_gumb['100'], label = "100 year")
# ax.set_ylabel("mm")
# ax.set_xlabel("date")
# ax.legend(frameon=False, fontsize=8)
# ax.set_title("Gumbel Recurrence Interval Rain Depths")

# # Weibull
# df_weib = df_perf[df_perf.fx == "weibull_min"]
# years = pd.to_datetime(df_weib.date).dt.year
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6, 4])
# ax.plot(years,df_weib['2'], label = "2 year")
# ax.plot(years,df_weib['10'], label = "10 year")
# ax.plot(years,df_weib['100'], label = "100 year")
# ax.set_ylabel("mm")
# ax.set_xlabel("date")
# ax.legend(frameon=False, fontsize=8)
# ax.set_title("weibel Recurrence Interval Rain Depths")

# # Log pearson Type III
# df_lp3 = df_perf[df_perf.fx == "log_pearson3"]
# years = pd.to_datetime(df_lp3.date).dt.year
# fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6, 4])
# ax.plot(years,df_lp3['2'], label = "2 year")
# ax.plot(years,df_lp3['10'], label = "10 year")
# ax.plot(years,df_lp3['100'], label = "100 year")
# ax.set_ylabel("mm")
# ax.set_xlabel("date")
# ax.legend(frameon=False, fontsize=8)
# ax.set_title("Log pearson Type III Recurrence Interval Rain Depths")

# Gumbel looks most reasonable
#%% fitting trend to gumbel fitted to each year
df_perf = pd.read_csv("model_performance_comparison.csv")
df_gumb = df_perf[df_perf.fx == "gumbel_r"]
years = pd.to_datetime(df_gumb.date).dt.year
markers = ["o", "^", "s"]
ts = ['2', '10', '100'] # recurrence intervals
cs = ["tab:blue", "tab:orange", "tab:green"] # plotting colors
x = np.arange(len(df_gumb))
fits = []
df_fits = pd.DataFrame()
fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6, 4])
i = -1
equations = [] 
perc_increases = []
for t in ts:
    i += 1
    fit = np.polyfit(x, df_gumb[t], 1)
    fit_function = np.poly1d(fit)
    fits.append(fit)
    lab = "{} year".format(t)
    equation = r"$y = {}x + {}$".format(round(fit[0], 2), round(fit[1], 2))
    equations.append(equation)
    perc_increases.append(round(fit[0]/fit[1]*100, 2))
    
    # plotting
    ax.plot(years,df_gumb[t], label = lab, color = cs[i])
    ax.plot(years, fit_function(x), color = cs[i], ls="--")


x = 2062
ax.text(x, 45, equations[0])
ax.text(x, 80, equations[1])
ax.text(x, 120, equations[2])
    
ax.legend(frameon=False, fontsize=8)

ax.set_ylabel("mm")
ax.set_xlabel("date")
ax.set_title("Gumbel 24-hour rain depths")

plt.savefig('figures/gumbel_2_10_100.png')

print("Percent increases per year by quantile:")
print(ts)
print(perc_increases)


#%% fitting to moving window
years = pd.to_datetime(df_gumb.date).dt.year

fig, ax = plt.subplots(1, 1, dpi=300, figsize=[6, 4])

markers = [".", "o", "^"]

legend_elements = {}
legend_elements_ws = {}
i = -1
for t in ts:
    i += 1
    lab = "{} year".format(t)
    
    legend_elements[lab] = Line2D([0], [0], color=cs[i], label=lab)
    
    ax.plot(years,df_gumb[t], label = lab, color = cs[i], linewidth=1, 
            ls = '-', alpha = 0.3)
    
    i2 = -1
    for w in df_gum_mving_wndw.by.unique():
        i2 += 1
        df = df_gum_mving_wndw[df_gum_mving_wndw.by==w]
        df = df[['max_year', int(t)]]
        lab = "{} year window".format(w)
        
        ax.scatter(df['max_year'], df[int(t)], facecolor ="none",
                   edgecolor=cs[i], marker=markers[i2], label = lab)
        
        legend_elements_ws[lab] = Line2D([0], [0], marker=markers[i2], color='black', label=lab,
               markerfacecolor='none', markersize=8, linewidth=0)   
        


legend_elements.update(legend_elements_ws)

ax.legend(handles = legend_elements.values(), frameon=False,
          fontsize=8, ncol=2, handletextpad=0.4)

ax.set_ylabel("mm")
ax.set_xlabel("date")
ax.set_title("Gumbel 24-hour rain depths fit to moving window")

plt.savefig('figures/gumbel_2_10_100_window.png')


#%% demonstrating fitting process
# option 1
df_2015_to_2020 = df_annual_maxima.iloc[0:5, :].melt(value_name = "precip").precip
df_2095_to_2100 = df_annual_maxima.iloc[-6:-1, :].melt(value_name = "precip").precip
out1 = fit_2_param_pdf(df_2015_to_2020, gumbel_r, data_lab = "2015 to 2020", c="b",
                          xlab="mm")#, recurrence_intervals=recurrence_intervals)
fit_2_param_pdf(df_2095_to_2100, gumbel_r, ax = out1['axis'], fig = fig, data_lab = "2095 to 2100", c="y",
                xlab="mm")#, recurrence_intervals = recurrence_intervals)

plt.savefig('figures/gumbel_fitted_option1.png')

# option 2
df_2015 = df_annual_maxima.iloc[0, :]
df_2099 = df_annual_maxima.iloc[-1, :]
out2 = fit_2_param_pdf(df_2015, gumbel_r, data_lab = "2015", c="b",
                          xlab="mm")#, recurrence_intervals=recurrence_intervals)
fit_2_param_pdf(df_2099, gumbel_r, ax = out2['axis'], fig = fig, data_lab = "2099", c="y",
                xlab="mm")#, recurrence_intervals = recurrence_intervals)

plt.savefig('figures/gumbel_fitted_option2.png')


# figs, axs = plt.subplots(nrows=1, ncols=2)
# figs[0] = out1['figure']
# figs[1] = out2['figure']



































