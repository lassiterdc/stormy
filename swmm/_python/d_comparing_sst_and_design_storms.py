#%% import libraries and load directories
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from scipy.stats import pearson3 
import numpy as np
import scipy
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib
# set desired font
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Franklin Gothic Book']})
rc('text', usetex=True)

from __ref_ams_functions import fit_3_param_pdf
from _inputs import d_comparing_design_strms_and_sst



f_sst_annual_max_volumes, f_design_strms, f_sst_recurrence_intervals, dir_plots, m_per_feet, recurrence_intervals, size_scaling_factor, target_text_size_in, pts_per_inch = d_comparing_design_strms_and_sst()

length_per_height = 5.7  / 7.5 
real_fig_height = 14
scaled_fig_height = real_fig_height * size_scaling_factor

default_fig_size = [scaled_fig_height, scaled_fig_height * length_per_height]


fontsize = round(target_text_size_in * pts_per_inch * size_scaling_factor * 0.5, 0)
# set font sizes
plt.rcParams['font.size'] = fontsize #fig text size is a frac of body text

# load data
df_sst = pd.read_csv(f_sst_annual_max_volumes)
df_sst_return_pds = pd.read_csv(f_sst_recurrence_intervals)
df_dsgn_strms = pd.read_csv(f_design_strms)

df_sst.name = "df_sst"
df_sst_return_pds.name = "df_sst_return_pds"
df_dsgn_strms.name = "df_dsgn_strms"

# make sure all the units are correct (rainfall in mm, surge in m, flood vols in 1e6m^3)
df_sst["max_sim_wlevel_m"] = df_sst["max_sim_wlevel"] * m_per_feet
df_dsgn_strms["max_sim_wlevel_m"] = df_dsgn_strms["max_sim_wlevel"] * m_per_feet
#%%
# define functions
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
def plotting_swmm_runs(df_plt, xvar, yvar, cvar, svar, cmap, alpha, exploration, dir_plots):
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=default_fig_size)

    if svar is not None:
        sz = [n/2 for n in df_plt[svar].rank()]
        im = ax.scatter(df_plt[xvar], df_plt[yvar], c=df_plt[cvar], cmap = cmap,
                s = sz, alpha = alpha, edgecolors = "none")
    else:
        im = ax.scatter(df_plt[xvar], df_plt[yvar], c=df_plt[cvar], cmap = cmap,
                alpha = alpha, edgecolors = "none")

    # pcm = ax.pcolormesh(df_plt[cvar], cmap=cmap)
    fig.colorbar(im, ax=ax, label = cvar)

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

    if exploration == True:
        fname = dir_plots + "_exploration/d_plot_{}_y-[{}]/x-[{}]_c-[{}]_s-[{}].svg".format(df_plt.name, yvar, xvar, cvar, svar)
    else:
        fname = dir_plots + "/d_plot_{}_y-[{}]_x-[{}]_c-[{}]_s-[{}].svg".format(df_plt.name, yvar, xvar, cvar, svar)

    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(fname)


def plotting_swmm_runs_sst_vs_dsgn(df_sst, df_dsgn_strms, xvar, yvar, cvar, cmap, alpha, dir_plots):
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=default_fig_size, linewidth=2, edgecolor = "#04253a")

    im = ax.scatter(df_sst[xvar], df_sst[yvar], c=df_sst[cvar], cmap = cmap, s = 100,
            alpha = alpha, edgecolors = "none", label = "Stochastic Hydrology")
    

    df_cvar_comb = pd.concat([df_sst[cvar], df_dsgn_strms[cvar]])

    im.set_clim(df_cvar_comb.min(), df_cvar_comb.max())

    lst_shps = ['X', '^', 's']
    lst_labs = []
    ind = -1
    for rec in recurrence_intervals:
        ind += 1
        med_rainfall = df_dsgn_strms[df_dsgn_strms.rain_return_period == rec].depth_mm.median()
        df_subset = df_dsgn_strms[df_dsgn_strms.rain_return_period == rec]
        df_subset = df_subset[df_dsgn_strms.depth_mm == med_rainfall]
        lab = "Design Storm, {} Year Rainfall".format(int(rec))
        im = ax.scatter(df_subset[xvar], df_subset[yvar], c=df_subset[cvar], cmap = cmap,
            alpha = alpha, edgecolors = "black", linewidth = 3, marker=lst_shps[ind], s = 600,
            label = lab)
        im.set_clim(df_cvar_comb.min(), df_cvar_comb.max())
        lst_labs.append(lab)

    # pcm = ax.pcolormesh(df_plt[cvar], cmap=cmap)
    fig.colorbar(im, ax=ax, label = "Peak Water Level (m)")

    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)

    mrkr_sz = 20
    mrkr_col = "black"
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Stochastically Generated',
                            markerfacecolor=mrkr_col, markeredgecolor=None ,markersize=mrkr_sz),
                       Line2D([0], [0], marker=lst_shps[0], color='w', label=lst_labs[0],
                            markerfacecolor=mrkr_col, markeredgecolor=None ,markersize=mrkr_sz),
                       Line2D([0], [0], marker=lst_shps[1], color='w', label=lst_labs[1],
                            markerfacecolor=mrkr_col, markeredgecolor=None ,markersize=mrkr_sz),
                       Line2D([0], [0], marker=lst_shps[2], color='w', label=lst_labs[2],
                            markerfacecolor=mrkr_col, markeredgecolor=None ,markersize=mrkr_sz)]
    
    ax.legend(handles=legend_elements, loc='best', handlelength = 0.5, handleheight=1.2,
               labelspacing=0.2, fancybox=True, framealpha=0.6, fontsize = fontsize*.8)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, 26000)

    # ax.legend()

    fname = dir_plots + "/d_plot_sst_vs_dsgn_y-[{}]_x-[{}]_c-[{}].svg".format(yvar, xvar, cvar)

    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    # ax.grid(which = 'minor', alpha = 0.4)
    ax.grid(which = 'major', alpha = 0.3)

    ax.set_ylabel("Total Flooding ($10^6$m$^3$)")

    ax.set_xlabel("Total Rain Depth (mm)")

    fig.tight_layout()

    fig.savefig(fname, edgecolor=fig.get_edgecolor())


# plotting swmm and sst runs
plotting_swmm_runs_sst_vs_dsgn(df_sst, df_dsgn_strms, xvar = "depth_mm", yvar = "total_flooding_1e+06m3",
             cvar = "max_sim_wlevel_m", cmap = 'plasma_r',
             alpha = 0.5, dir_plots = dir_plots)

#%%

#plotting swmm floods against cdf from sst
# prep data
def conv_return_pd_to_quant(return_pd):
    return 1 - 1/return_pd

def conv_quant_to_return_pd(quant):
    return 1/(1-quant)

df_sst_flds = df_sst["total_flooding_1e+06m3"]
df_sst_flds = pd.DataFrame(dict(year = np.arange(0, 1000))).join(df_sst_flds)
df_sst_flds.fillna(0.001, inplace=True)
df_sst_flds.drop(columns="year", inplace=True)
s_sst_flds = df_sst_flds["total_flooding_1e+06m3"]

out_p3 = fit_3_param_pdf(s_sst_flds, pearson3, recurrence_intervals,
                 xlab = "Flood Volume ($10^6m^3$)", plot = False)

shape, loc, scale = out_p3['params']



fig, ax = plt.subplots(1, 1, dpi=300, figsize=default_fig_size, linewidth=2, edgecolor = "#04253a")

return_pd_limit_emp = 100
return_pd_limit_svg = 500

out_p3 = fit_3_param_pdf(s_sst_flds, pearson3, recurrence_intervals,
                 xlab = "Flood Volume ($10^6m^3$)", plot = False)

df_cdf = out_p3['df']
s_return_pds = df_cdf.cumprob_emp.apply(conv_quant_to_return_pd)
s_return_pds.name = "return_periods_yrs"
df_emp_return_pds = pd.concat([df_cdf.iloc[:,0], s_return_pds], axis = 1)

df_emp_return_pds_plot = df_emp_return_pds[df_emp_return_pds["return_periods_yrs"]<=return_pd_limit_emp]
df_emp_return_pds_plot = df_emp_return_pds_plot.rename(columns={"total_flooding_1e+06m3":"empirical"})
# ax = df_sst_return_pds.plot("return_periods_yrs", "base_fit", logx=True)
df_sst_return_pds_subset = df_sst_return_pds[df_sst_return_pds.return_periods_yrs <= return_pd_limit_svg]

df_emp_return_pds_plot.plot("return_periods_yrs", "empirical",
                             label = 'Emprical Flood Probabilities\nfrom Stochastic Approach',
                                 logx=True, ax=ax,
                                 linewidth = 2.5, color = 'black', alpha = 0.8)

ax.fill_between(df_sst_return_pds_subset["return_periods_yrs"],
                 df_sst_return_pds_subset['quantile_0.05'],
                   df_sst_return_pds_subset['quantile_0.95'],
                   alpha = 0.15, color = 'grey',
                   label = "Pearson Type III 90\%\nConfidence Interval")

# df_sst_return_pds_subset.plot("return_periods_yrs", "quantile_0.05", logx=True, ax=ax)
# df_sst_return_pds_subset.plot("return_periods_yrs", "quantile_0.95", logx=True, ax=ax)

c_dict = dict(mean_high_water_level = "#ef8a62",
              same_as_rainfall = "#67a9cf")

lab_dict = dict(mean_high_water_level = "Design Storm, Independent\nSurge and Rainfall",
              same_as_rainfall = "Design Storm, Dependent\nSurge and Rainfall")

for surge_return in df_dsgn_strms.surge_return_period.unique():
    df_subset = df_dsgn_strms[df_dsgn_strms.surge_return_period == surge_return]
    col = c_dict[surge_return]
    df_subset.plot.scatter("rain_return_period", "total_flooding_1e+06m3",
                           c = col, ax = ax, label = lab_dict[surge_return],
                           s = 150, zorder=10, edgecolor = 'black')

ax.set_ylabel("Total Flooding ($10^6$m$^3$)")

ax.set_xlabel("Return Period (years)")

ax.legend(loc='best', handleheight=1.3, labelspacing=0.4, fontsize = fontsize*.8)

# add border
# ax.patch.set_edgecolor('black')  

# ax.patch.set_linewidth(1)  

fname = dir_plots + "/d_plot_{}.svg".format("flood_return_pds")

ax.grid(which = 'minor', alpha = 0.4)
ax.grid(which = 'major')

fig.tight_layout()

fig.savefig(fname, edgecolor=fig.get_edgecolor())
#%%



# lst_cols = ["#1b9e77","#d95f02", "#7570b3"]


# ind = -1
# rect_width = 1
# for rec in recurrence_intervals:
#     ind += 1
#     # design storms
#     # exceedance_prob = (1-(1/rec))
#     s_subset_dsgn = df_dsgn_strms[df_dsgn_strms.rain_return_period == rec]["total_flooding_1e+06m3"]
#     ymin, ymax = s_subset_dsgn.min(), s_subset_dsgn.max()
#     rect_height = ymax - ymin
#     # rect_width = xmax-xmin
#     rect_anchor = (rec-1/2*rect_width), ymin
#     ax.vlines(x=rec, ymin = ymin, ymax = ymax, color = lst_cols[ind], zorder=10, alpha = 0.8)
#     # ax.hlines(ymin, xmin = rec-rect_width/2, xmax =  rec+rect_width/2,
#     #            color = lst_cols[ind], zorder=10, alpha = 0.9)
#     # ax.hlines(ymax, xmin = rec-rect_width/2, xmax =  rec+rect_width/2,
#     #            color = lst_cols[ind], zorder=10, alpha = 0.9)


#%% exploration
vars = ["total_flooding_1e+06m3", "max_sim_wlevel_m", "depth_mm"]
cmap = 'jet'
alpha = 0.4
exploration = True

for cols in list(itertools.permutations(vars,3)):
    if "total_flooding_1e+06m3" not in cols:
        continue
    xvar,yvar,cvar = cols
    if xvar == yvar:
        continue
    if (cvar == xvar) or (cvar ==  yvar):
        continue
    svar = cvar
    plotting_swmm_runs(df_sst, xvar, yvar, cvar, svar, cmap, alpha, exploration, dir_plots)
    svar = None
    plotting_swmm_runs(df_sst, xvar, yvar, cvar, svar, cmap, alpha, exploration, dir_plots)
#%% keepers
plotting_swmm_runs(df_plt = df_sst, xvar = "depth_mm", yvar = "total_flooding_1e+06m3",
             cvar = "max_sim_wlevel_m", svar = None, cmap = 'jet',
             alpha = 0.4, exploration = False, dir_plots = dir_plots)

#%% plotting swmm results
plotting_swmm_runs(df_plt = df_dsgn_strms, xvar = "depth_mm", yvar = "total_flooding_1e+06m3",
             cvar = "max_sim_wlevel_m", svar = None, cmap = 'jet',
             alpha = 0.4, exploration = False, dir_plots = dir_plots)