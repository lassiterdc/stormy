#%% load libraries and directories
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
from tqdm import tqdm
import imageio
from glob import glob
from _inputs import *
import os
from scipy.stats.mstats import plotting_positions

# hard coded parameters
n_largest_to_keep = 2 # only using a subset of events

#%% functions
def keep_largest_n_events(df_to_subset, n_largest_to_keep, str_rank_var, lst_group_vars,):
    variable_ranked_and_subset = df_to_subset.loc[:, str_rank_var].sort_values(ascending = False).groupby(level = lst_group_vars).head(n_largest_to_keep)
    variable_ranked_and_subset = variable_ranked_and_subset.sort_index()
    df_nlargest = df_to_subset.loc[variable_ranked_and_subset.index.values,:]
    # confirm tha tthe 2 largest flood values are indeed being returned
    # df_nlargest.loc[(1,1,slice(None),"UN90"),:]
    # df_to_subset.loc[(1,1,slice(None),"E1331931"),:]
    print("Keeping {}% of the data.".format((len(variable_ranked_and_subset) / len(df_to_subset))*100))
    return df_nlargest

def plot_flood_return_vs_other_return(df, other_var, ylabel, ax = None, show_cbar = False, 
                                      show_ylabel = False, show_xlabel = False, savefig = False,
                                      point_size_scaler = 1, show_xticks= True,
                                      show_yticks = True, vmax = None, ylim = None,
                                      xlim = None):
    marker_size = df.node_flooding_order_of_magnitude**2/3 * point_size_scaler
    if ax is None:
        fig, ax = plt.subplots(dpi=300)
    if vmax is None:
        vmax = df.node_flooding_order_of_magnitude.quantile(0.98)
    artist = df.plot.scatter(ax=ax, 
                            x = "flood_empirical_return_yr",
                            y = other_var,
                            c = "node_flooding_order_of_magnitude",
                            s = marker_size,
                            vmin = 0,
                            vmax = vmax,
                            logx = True,
                            logy = True,
                            cmap = "viridis",
                            colorbar = False,
                            #    norm = norm,
                            alpha = 0.9,
                            zorder = 8)
    ax.grid(zorder = 5, alpha = 0.7, which = 'both')
    if ylim is not None:
        ax.set_ylim(1, ylim)
    if xlim is not None:
        ax.set_xlim(1, xlim)
    
    if show_xlabel:
        ax.set_xlabel("Node Flooding Return Period (yr)")
    else:
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")

    if show_xticks == False:
        ax.set_xticklabels([])
    if show_yticks == False:
        ax.set_yticklabels([])


    ax.axline(xy1=(0,0), xy2=(100,100), zorder=10, c = "red", alpha = 0.5)
    if show_cbar:
        cbar = ax.figure.colorbar(ax.collections[0], label = "log$_{10}$ of node flooding (m$^3$)",
                           shrink = 1, orientation = "horizontal", location = "top",
                           anchor = (0.5, 0), aspect = 35)
        # cbar = ax.figure.colorbar(ax.collections[0], label = "log$_{10}$ of node flooding (m$^3$)",
        #                    aspect = 35)
        tick_locator = ticker.MaxNLocator(nbins = 4)
        cbar.locator = tick_locator
        cbar.update_ticks()

    # fig.colorbar(artist, ax = ax, shrink = 0.6, label = "log$_10$(node flooding m$^3$)")
    if savefig:
        plt.savefig(fldr_swmm_analysis_plots + "f_empirical_flooding_vs_{}.png".format(other_var),
                    transparent=False, bbox_inches='tight')
#%% data processing to plot event returns vs. node flooding returns
df_sst_compound = ds_sst_compound.to_dataframe()
# inspecting data
n_na = sum(df_sst_compound.node_flooding_cubic_meters.isna())
print("There are {} na node flood values in the dataset. This represents {}% of the data.".format(
    n_na, round((n_na / len(df_sst_compound)), 2)*100
))


#%%
nyears_modeled_in_swmm = df_sst_compound.reset_index().realization.max() * df_sst_compound.reset_index().year.max()
# sort index values
df_sst_compound = df_sst_compound.sort_index()


# subset to keep the n largest events per year
df_sst_compound_n_largest = keep_largest_n_events(df_sst_compound, n_largest_to_keep=n_largest_to_keep, str_rank_var="node_flooding_cubic_meters",
                                                  lst_group_vars=["realization", "year", "node_id"])
# simulated_storms_subset = df_sst_compound_n_largest.reset_index().loc[:, storm_id_variables]
# simulated_storms_subset = simulated_storms_subset.drop_duplicates().reset_index(drop = True)
# compute plotting positions for all nodes (weibull)
emp_cdf = df_sst_compound_n_largest.loc[:,"node_flooding_cubic_meters"].groupby(level = "node_id").apply(plotting_positions, alpha=0, beta=0)
# create a dataframe with the plotting positions
lst_dfs = []
for node, row in emp_cdf.items():
    # extract the storms that resulted in each event at the node
    storms = df_sst_compound_n_largest.loc[(slice(None),slice(None),slice(None),node)].reset_index().loc[:,storm_id_variables]
    # convert the quantile values for the node to a dataframe
    df_emp_cdf_loop = pd.DataFrame(dict(flood_empirical_quantile = row))
    # assign a column with the node id for joining later
    df_emp_cdf_loop["node_id"] = node
    # compute the empirical return period
    df_emp_cdf_loop["flood_empirical_return_yr"] = comp_annual_return_period(n_largest_to_keep, df_emp_cdf_loop.flood_empirical_quantile)
    df_emp_cdf_loop = df_emp_cdf_loop.join(storms)
    # add storm id info to the dataframe for re-joining
    lst_dfs.append(df_emp_cdf_loop)

df_empicial_cdf_vals_by_node = pd.concat(lst_dfs, ignore_index = True)

# join df_events with the simulated cdf values

df_flood_and_event_returns = df_empicial_cdf_vals_by_node.set_index(["realization", "year", "storm_id", "node_id"]).join(df_events)
df_flood_and_event_returns = df_flood_and_event_returns.join(df_sst_compound)
# NEED TO FIGURE OUT WHY THERE ARE MISSING FLOOD VOLUME VALUES
#%% work
idx_1kyr_events = df_flood_and_event_returns.flood_empirical_return_yr > 1000
df_1kyr_events = df_flood_and_event_returns[idx_1kyr_events]
# these are all na for some reason
print(len(df_1kyr_events.dropna()))

# inspecting the events that have na 
idx_na_events = df_flood_and_event_returns.node_flooding_cubic_meters.isna()
df_na_events = df_flood_and_event_returns[idx_na_events]
print(df_na_events.flood_empirical_return_yr.unique())
print("FOR SOME REASON THESE ARE THE RETURN PERIODS THAT ARE ASSOCIATED WITH NA FLOOD VOLUMES")

# inspecting the events that are not na
idx_notna_events = ~df_flood_and_event_returns.node_flooding_cubic_meters.isna()
df_notna_events = df_flood_and_event_returns[idx_notna_events]

idx_100yr_events = df_notna_events.flood_empirical_return_yr > 100
df_100yr_events = df_notna_events[idx_100yr_events]
df_100yr_events.node_flooding_cubic_meters.hist(log=True)
print("Clearly there are non-zero flood nodes greater than 100 year events")
#%% end work
print("The minimum empirical return period for flooding was {}".format(df_flood_and_event_returns.flood_empirical_return_yr.min()))
# find unique events
vars_of_interest = ["node_flooding_cubic_meters", "flood_empirical_return_yr", "empirical_event_return_pd_yr",
                    'max_mm_per_hour_emp_return_pd_yr', 'max_surge_ft_emp_return_pd_yr', 'depth_mm_emp_return_pd_yr']
df_flood_and_event_returns_no_zeros = df_flood_and_event_returns.loc[:,vars_of_interest].copy()
# remove all rows where flooding is 0
df_flood_and_event_returns_no_zeros = df_flood_and_event_returns_no_zeros[df_flood_and_event_returns_no_zeros.node_flooding_cubic_meters>0].copy()
df_flood_and_event_returns_no_zeros["node_flooding_order_of_magnitude"] = np.log10(df_flood_and_event_returns_no_zeros.node_flooding_cubic_meters)
# only include flooding of 1 cubic meter or more
# df_flood_and_event_returns = df_flood_and_event_returns[df_flood_and_event_returns["node_flooding_order_of_magnitude"]>=0]
print("After removing node events with 0 flooding, the minimum empirical return period for flooding was {}".format(df_flood_and_event_returns_no_zeros.flood_empirical_return_yr.min()))

print("The maximum empirical return period for flooding was {}".format(df_flood_and_event_returns_no_zeros.flood_empirical_return_yr.max()))
#%% plotting


# plot_flood_return_vs_other_return(other_var, ylabel)
#%% create a multi-part plot comparing univariate event return periods with multivariate event return periods
other_vars = ["empirical_event_return_pd_yr",'max_mm_per_hour_emp_return_pd_yr', 'max_surge_ft_emp_return_pd_yr', 'depth_mm_emp_return_pd_yr']
ylabels = ["Return Period (yr) of Event (all 3 statistics exceeded)", "Rain Intensity Return (yr)", "Storm Surge Return (yr)", "Rain Depth Return (yr)"]

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker

scaling = 0.8

# Create a 2x2 grid for the plots
fig = plt.figure(figsize=(8*scaling, 10*scaling), layout = 'constrained')
gs = GridSpec(2, 3, height_ratios = [1,.3], figure = fig)
# top
ax1 = plt.subplot(gs[0, :])
# bottom left
ax2 = plt.subplot(gs[1, 0])
# bottom middle
ax3 = plt.subplot(gs[1, 1])
# bottom right
ax4 = plt.subplot(gs[1, 2])

lst_axes = [ax1, ax2, ax3, ax4]

df_for_plotting = pd.DataFrame(dict(other_var = other_vars, ylabel = ylabels))

df_data = df_flood_and_event_returns_no_zeros#.sample(1000)

base_pt_size = 1.3
for ind, row in df_for_plotting.iterrows():
    xlim = ylim = 250
    if row.other_var == "empirical_event_return_pd_yr":
        print('Variables used for computing multivariate return period:')
        print(vars_used_For_multivar_empirical_return_pd)
    show_cbar = False
    show_ylabel = True
    point_size_scaler = 0.33*base_pt_size
    show_xticks = True
    show_yticks = False
    if ind == 0:
        show_cbar = True
        point_size_scaler = 1*base_pt_size
        show_yticks = True
        show_xticks = True
    show_xlabel = False
    if ind == 1:
        show_yticks = True
        show_xticks = True
        # show_xlabel = True
    if ind == 2:
        show_xlabel = True
    plot_flood_return_vs_other_return(df_data, row.other_var, row.ylabel, ax = lst_axes[ind],
                                       show_cbar=show_cbar, show_ylabel=show_ylabel,
                                       show_xlabel=show_xlabel, point_size_scaler=point_size_scaler,
                                       show_xticks= show_xticks, show_yticks = show_yticks,
                                       vmax = 4, xlim=xlim, ylim=ylim) 
plt.savefig(fldr_swmm_analysis_plots + "f_empirical_flooding_vs_all_return_pds.png",
                    transparent=False, bbox_inches='tight', dpi = 400)