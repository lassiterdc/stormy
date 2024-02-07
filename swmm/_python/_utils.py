#%% load libraries
from glob import glob
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from _inputs import *

def return_period_to_quantile(n_storms_per_year, return_pds):
  # https://colab.research.google.com/drive/1Iaji34xdpuY1ZmxIkYhCe_zQcLXZNhJ8?usp=sharing#scrollTo=rUFCKYT7zWIS
  quants = []
  for return_period in return_pds:
    event_period = 1/n_storms_per_year
    exceedance_prob = event_period/return_period
    cdf_val = 1 - exceedance_prob
    quants.append(cdf_val)
  return quants

def compute_specific_return_periods(ds, n_storms_per_year, recurrence_interval_yrs, method = "hazen"):
    quants = return_period_to_quantile(n_storms_per_year, recurrence_interval_yrs)
    # find the quantiles associated with each return period
    ds_quants = ds.quantile(quants, dim = ["storm_id", "realization", "year"], method=method)
    # rename the quantile variable to the return period value
    ds_quants = ds_quants.assign_coords(dict(quantile = recurrence_interval_yrs))
    ds_quants = ds_quants.rename((dict(quantile="flood_return_yrs")))
    return ds_quants

def quantile_to_return_period(n_storms_per_year, cdf_vals):
    # https://colab.research.google.com/drive/1Iaji34xdpuY1ZmxIkYhCe_zQcLXZNhJ8#scrollTo=rUFCKYT7zWIS
    # that is a google colab notebook with a visual example of this function to make sure it's behaving as expected
    event_period = 1/n_storms_per_year
    return_periods = []
    for cdf_val in cdf_vals:
        exceedance_prob = 1 - cdf_val
        return_period = 1/exceedance_prob * event_period
        return_periods.append(return_period)
        # if the time series was annual maxima (period of 1 year per event), an event with a 0.1 exceedance probability 
        # would have a return period of 1/0.1 * 1 = 10 year return period
        # if it time series had a period of 1/5 years per event, the same exceedance probability would yield
        # a return period of 1/0.1 * (1/5) = 2 year return period (which makse sense)
    return return_periods

#%% data processing
# load geospatial data
proj = ccrs.PlateCarree()
gdf_jxns = gpd.read_file(f_shp_jxns)
gdf_strg = gpd.read_file(f_shp_strg)
gdf_out = gpd.read_file(f_shp_out)
gdf_nodes = pd.concat([gdf_jxns, gdf_strg, gdf_out]).loc[:, ["NAME", "geometry"]]
gdf_nodes = gdf_nodes.to_crs(proj)


df_comparison = pd.read_csv(f_bootstrapping_analysis)



df_events = pd.read_csv(f_sims_summary).set_index(storm_id_variables)

# compute number of years
n_years_generated = len(df_events.reset_index().realization.unique()) * len(df_events.reset_index().year.unique())
n_storms_generated = n_years_generated * len(df_events.reset_index().storm_id.unique()) 

# add multivariate empirical CDF columns to df_events
df_events_with_emp_cdf = pd.read_csv(f_simulated_cmpnd_event_summaries).set_index(storm_id_variables)

# load cdf values
df_events_cdf_simulation = pd.read_csv(f_simulated_cmpnd_event_cdfs).join(df_events_with_emp_cdf.reset_index().loc[:, storm_id_variables]).set_index(storm_id_variables)

# add everything to the df_events dataframe
df_events = df_events.join(df_events_with_emp_cdf.loc[:, ["n_emp_3var_cdf", "emp_3var_cdf"]])
df_events = df_events.join(df_events_cdf_simulation, rsuffix="_cdf")

# compute return periods for each 
n_storms_per_year = n_storms_generated / n_years_generated

# compute empirical event return periods
lst_series = []
for colname, series in df_events_cdf_simulation.items():
    ar_return_pds = quantile_to_return_period(n_storms_per_year, series)
    series_return_pds = pd.Series(ar_return_pds)
    series_return_pds.name = colname + "_emp_return_pd_yr"
    series_return_pds.index = series.index
    df_events = df_events.join(series_return_pds)
    
df_events["empirical_event_return_pd_yr"] = quantile_to_return_period(n_storms_per_year, df_events["emp_3var_cdf"])
#%% qaqcing quantiles and return periods
def plot_histogram_of_quantiles_and_return_pds(series_quants, series_return_pds):
    figsize_multiplier = 1.5
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6*figsize_multiplier,3*figsize_multiplier), dpi = 300)
    series_quants.hist(ax=axes[0])
    axes[0].set_xlabel(series_quants.name)
    axes[0].set_ylabel("Count")

    # fig, ax = plt.subplots()
    series_return_pds.hist(ax=axes[1], log = True)
    axes[1].set_xlabel(series_return_pds.name)
    axes[1].set_ylabel("")

# emp_3var_cdf
plot_histogram_of_quantiles_and_return_pds(df_events["emp_3var_cdf"], df_events["empirical_event_return_pd_yr"])
# depth_mm
plot_histogram_of_quantiles_and_return_pds(df_events["depth_mm_cdf"], df_events["depth_mm_emp_return_pd_yr"])
# max_mm_per_hour
plot_histogram_of_quantiles_and_return_pds(df_events["max_mm_per_hour_cdf"], df_events["max_mm_per_hour_emp_return_pd_yr"])
# max_sim_wlevel
plot_histogram_of_quantiles_and_return_pds(df_events["max_surge_ft_cdf"], df_events["max_surge_ft_emp_return_pd_yr"])
# mean_mm_per_hr
plot_histogram_of_quantiles_and_return_pds(df_events["mean_mm_per_hr_cdf"], df_events["mean_mm_per_hr_emp_return_pd_yr"])


# seeing if subsetting the rainfall statistics makes them more uniform
def keep_largest_n_events(df_to_subset, n_largest_to_keep, str_rank_var, lst_group_vars):
    variable_ranked_and_subset = df_to_subset.loc[:, str_rank_var].sort_values(ascending = False).groupby(level = lst_group_vars).head(n_largest_to_keep)
    variable_ranked_and_subset = variable_ranked_and_subset.sort_index()
    df_nlargest = df_to_subset.loc[variable_ranked_and_subset.index.values,:]
    # confirm tha tthe 2 largest flood values are indeed being returned
    # df_nlargest.loc[(1,1,slice(None),"UN90"),:]
    # df_to_subset.loc[(1,1,slice(None),"E1331931"),:]
    print("Keeping {}% of the data.".format((len(variable_ranked_and_subset) / len(df_to_subset))*100))
    return df_nlargest

df_events_subset_depth_mm = keep_largest_n_events(df_events, 2, "depth_mm", lst_group_vars=["realization", "year"])
# depth_mm
plot_histogram_of_quantiles_and_return_pds(df_events_subset_depth_mm["depth_mm_cdf"], df_events_subset_depth_mm["depth_mm_emp_return_pd_yr"])
# max_mm_per_hour
plot_histogram_of_quantiles_and_return_pds(df_events_subset_depth_mm["max_mm_per_hour_cdf"], df_events_subset_depth_mm["max_mm_per_hour_emp_return_pd_yr"])
# mean_mm_per_hr
plot_histogram_of_quantiles_and_return_pds(df_events_subset_depth_mm["mean_mm_per_hr_cdf"], df_events_subset_depth_mm["mean_mm_per_hr_emp_return_pd_yr"])




#%% end qaqc

# return events used for computing empirical return period
multivar_emp_cdf_from_r = pd.read_csv(f_wlevel_cdf_sims_from_copula_with_multvar_empcdf_subset)
vars_used_For_multivar_empirical_return_pd = multivar_emp_cdf_from_r.drop(columns = ["N.cum", "CDF"]).columns.tolist()
# this is a workaround because I left the format as a string
try:
    event_duration_hr = pd.to_timedelta(df_events["event_duration_hr"]) / np.timedelta64(1, 'h')
    df_events.loc[:, "event_duration_hr"] = event_duration_hr.values
except:
    pass
ds_events = df_events.to_xarray()

# node_ids_sorted = df_node_attribution[df_node_attribution.flood_return_yrs == 100].sort_values("lower_CI", ascending=True)["node_id"].values

df_comparison = df_comparison.set_index(["flood_return_yrs", "node_id"])
# workaround - replacing any attribution values greater than 1 with 1 and less than 1 with 0
# def workaround_replace_vals(var_to_replace, df_comparison=df_comparison):
#     idx_mean_greater_than_1 = df_comparison.index[df_comparison[var_to_replace] > 1].tolist()
#     idx_mean_less_than_1 = df_comparison.index[df_comparison[var_to_replace] < 0].tolist()
#     df_comparison.loc[idx_mean_greater_than_1, var_to_replace] = 1
#     df_comparison.loc[idx_mean_less_than_1, var_to_replace] = 0
#     return df_comparison

# df_comparison = workaround_replace_vals("frac_wlevel_mean", df_comparison)
# df_comparison = workaround_replace_vals("frac_wlevel_median", df_comparison)

#%% dcl work - figuring out better way of isolating the nodes that have variable attribution
# identifying variable nodes
vrblity_anly_var = 'frac_wlevel_mean'
# return the standard deviation of the target variability analysis variable; replace NA values with  0 (this means there was just 1 observation) and sort values
df_node_variability = df_comparison.reset_index().loc[:, ["node_id", vrblity_anly_var]].groupby("node_id").std().fillna(0).sort_values(vrblity_anly_var)
std_varname = "std_of_{}".format(vrblity_anly_var)
df_node_variability = df_node_variability.rename(columns = {vrblity_anly_var:std_varname})

# df_variable_nodes = 
df_variable_nodes = df_node_variability[df_node_variability[std_varname] >= df_node_variability[std_varname].quantile(quant_top_var)]
# df_node_attribution_subset = df_node_attribution.join(df_node_variability_subset, how = "right")

# df_comparison.loc[(slice(None), "UN69"),]

# old way - start
# df_ge1y = df_comparison[(df_comparison["frac_wlevel_mean"].reset_index().flood_return_yrs>=1).values]
# lst_ranges = []
# lst_nodes = []
# for node, group in df_ge1y.reset_index().groupby("node_id"):
#     frac_range = group.frac_wlevel_mean.max() - group.frac_wlevel_mean.min()
#     lst_ranges.append(frac_range)
#     lst_nodes.append(node)

# df_node_ranges = pd.DataFrame(dict(node = lst_nodes, range = lst_ranges))

# df_node_variability = df_node_ranges[df_node_ranges.range >= df_node_ranges.range.quantile(quant_top_var)]
# end old way
#%% end dcl work
gdf_node_attribution = gdf_nodes.merge(df_comparison.reset_index(), how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)

df_node_attribution = pd.DataFrame(gdf_node_attribution)
df_node_attribution = df_node_attribution.set_index(["node_id", "flood_return_yrs"])


#%% processing SWMM flooding
#%% define relevant functions for watershed scale attribution
def classify_sims(df_total_flooding):
    # classify the model runs
    lst_sim_type = []
    for ind, row in df_total_flooding.iterrows():
        if str(row.freeboundary) == "True" and str(row.norain) == "False":
            lst_sim_type.append("freeboundary")
        elif str(row.freeboundary) == "False" and str(row.norain) == "True":
            lst_sim_type.append("norain")
        elif str(row.freeboundary) == "False" and str(row.norain) == "False":
            lst_sim_type.append("compound")
        elif str(row.freeboundary) == "True" and str(row.norain) == "True":
            lst_sim_type.append("not_simulated")
        else:
            lst_sim_type.append("no_conditional_statements_triggered")

    df_total_flooding["sim_type"] = lst_sim_type
    # remove rows that weren't simulated (no rain and free boundary condition, i.e., no hydrology)
    df_total_flooding = df_total_flooding[~(df_total_flooding.sim_type == "not_simulated")]

    return df_total_flooding

def return_flood_losses_and_continuity_errors(swmm_rpt):
    from pyswmm import Simulation, Nodes
    with open(swmm_rpt, 'r', encoding='latin-1') as file:
        # Read all lines from the file
        lines = file.readlines()
    line_num = -1
    lst_node_fld_summary = []
    encountered_header_of_node_flooding_summary = False
    encountered_end_of_node_flooding_summary = False
    # encountered_runoff_quantity_continuity = False
    encountered_flow_routing_continuity = False
    for line in lines:
        line_num += 1
        # if "Runoff Quantity Continuity" in line:
        #     encountered_runoff_quantity_continuity = True
        if "Flow Routing Continuity" in line:
            encountered_flow_routing_continuity = True
        if "Continuity Error (%) ....." in line:
            # runoff routing is reported BEFORE flow routing
            if encountered_flow_routing_continuity == False:
                runoff_continuity_error_line = line
            else:
                flow_continuity_error_line = line
        # return system flood statistic
        if "Flooding Loss" in line:
            system_flood_loss_line = line
        # return node flooding summaries
        if "Node Flooding Summary" in line:
            node_fld_sum_1st_line = line_num
            encountered_header_of_node_flooding_summary = True
        if encountered_header_of_node_flooding_summary == False:
            continue
        if line_num < node_fld_sum_1st_line + 5: # skip the header line and the next 4 lines
            continue
        if "******" in line:
            encountered_end_of_node_flooding_summary = True
        if encountered_end_of_node_flooding_summary == False:
            lst_node_fld_summary.append(line)

    # the rpt file only has nodes with nonzero flooding but I need to account for all nodes
    # create pandas series of node flood summaries
    # return ids of all nodes
    lst_nodes = []
    with Simulation(swmm_rpt.split(".rpt")[0] + ".inp") as sim:
       for node in Nodes(sim):
           lst_nodes.append(node.nodeid)

    df_allnodes = pd.DataFrame(dict(
        node_id = lst_nodes,
        # dummy = np.zeros(len(lst_nodes))
    ))
    df_allnodes = df_allnodes.set_index("node_id")

    n_header_rows = 5
    node_ids = []
    flood_volumes = []
    for i in np.arange(n_header_rows, len(lst_node_fld_summary)):
        line = lst_node_fld_summary[i]
        lst_values = []
        if len(line.split("  ")) == 2:
            continue
        for item in line.split("  "):
            if item == "":
                continue
            lst_values.append(item)
        node_id = lst_values[0]
        flooding = float(lst_values[5])
        node_ids.append(node_id)
        flood_volumes.append(flooding)

    df_node_flooding_subset = pd.DataFrame(dict(
        node_id = node_ids,
        flood_volume = flood_volumes
    ))
    df_node_flooding_subset.set_index("node_id", inplace = True)
    df_node_flooding = df_allnodes.join(df_node_flooding_subset)
    # for the nodes not in the rpt, assign them a flood volume of 0
    df_node_flooding = df_node_flooding.fillna(0)
    s_node_flooding = df_node_flooding.flood_volume

    # return runoff and flow continuity
    runoff_continuity_error_perc = float(runoff_continuity_error_line.split(" ")[-1].split("\n")[0])
    flow_continuity_error_perc = float(flow_continuity_error_line.split(" ")[-1].split("\n")[0])

    # return system flood losses
    system_flooding = float(system_flood_loss_line.split(" ")[-1].split("\n")[0])
    frac_diff_node_minus_system_flood = (s_node_flooding.sum() - system_flooding)/system_flooding

    return s_node_flooding,system_flooding,runoff_continuity_error_perc,flow_continuity_error_perc,frac_diff_node_minus_system_flood

############################## WATERSHED SCALE ATTRIBUTION ##############################
# compute watershed scale flood attribution
def compute_wshed_scale_attribution(ds_sst):
    # sum up flooding in each event by node
    df_total_flooding = ds_sst.sum(dim = "node_id").to_dataframe()
    df_total_flooding = df_total_flooding.reset_index()       
    df_total_flooding = classify_sims(df_total_flooding)

    # add colume with total flooding in a more readable unit (e.g., 10^6 cubic meters; defined in _inputs.py)
    a_tot_fld = df_total_flooding.node_flooding_cubic_meters / volume_units
    units="{:.0e}".format(int(volume_units))
    df_total_flooding.insert(len(df_total_flooding.columns)-1, "total_flooding_{}m3".format(units), a_tot_fld)

    # compute attribution statistics
    df_rain_only_flding = df_total_flooding[df_total_flooding.sim_type == "freeboundary"]
    df_surge_only_flding = df_total_flooding[df_total_flooding.sim_type == "norain"]
    df_compound_flding = df_total_flooding[df_total_flooding.sim_type == "compound"] 

    # isolate series for for each simulation type
    s_flooding_cubic_meters_rainonly = df_total_flooding[df_total_flooding.sim_type == "freeboundary"].set_index(["realization", "year", "storm_id"]).node_flooding_cubic_meters
    s_flooding_cubic_meters_surgeonly = df_total_flooding[df_total_flooding.sim_type == "norain"].set_index(["realization", "year", "storm_id"]).node_flooding_cubic_meters
    s_flooding_cubic_meters_compound = df_total_flooding[df_total_flooding.sim_type == "compound"].set_index(["realization", "year", "storm_id"]).node_flooding_cubic_meters
    # rename series so names become distinguishable columns in a dataframe
    s_flooding_cubic_meters_rainonly.name = "flooding_cubic_meters_rainonly"
    s_flooding_cubic_meters_surgeonly.name = "flooding_cubic_meters_surgeonly"
    s_flooding_cubic_meters_compound.name = "flooding_cubic_meters_compound"
    # combine series
    df_watershed_flood_attribution = pd.concat([s_flooding_cubic_meters_rainonly, s_flooding_cubic_meters_surgeonly, s_flooding_cubic_meters_compound], axis = 1)
    # remove rows where at least one simulation type has a total flooded volume greater than the cutoff
    df_watershed_flood_attribution = df_watershed_flood_attribution[(df_watershed_flood_attribution>min_vol_cutoff_watershed_scale_attribution).any(axis=1)]
    # compute attribution to rainfall, storm surge, and their interaction
    df_watershed_flood_attribution["sum_rainonly_surgeonly"] = df_watershed_flood_attribution.flooding_cubic_meters_rainonly + df_watershed_flood_attribution.flooding_cubic_meters_surgeonly
    df_watershed_flood_attribution["frac_rain"] = df_watershed_flood_attribution.flooding_cubic_meters_rainonly/df_watershed_flood_attribution.flooding_cubic_meters_compound
    df_watershed_flood_attribution["frac_surge"] = df_watershed_flood_attribution.flooding_cubic_meters_surgeonly/df_watershed_flood_attribution.flooding_cubic_meters_compound
    df_watershed_flood_attribution["frac_interaction"] = (df_watershed_flood_attribution.flooding_cubic_meters_compound - df_watershed_flood_attribution.sum_rainonly_surgeonly)/df_watershed_flood_attribution.flooding_cubic_meters_compound

    # where the frac rain and frac surge is na AND the flood volume is 0, replace the frac with 0
    # (it's a safe bet that if there is 0 volume in the rain or surge only sim that it's attribution must be 0)
    # NA values happen when there is zero volume in the compound sim and non-zero in another sim
    for ind, row in df_watershed_flood_attribution.iterrows():
        # frac_rain
        if np.isnan(row.frac_rain):
            if row.flooding_cubic_meters_rainonly == 0:
                df_watershed_flood_attribution.loc[ind, "frac_rain"] = 0
        # frac_surge
        if np.isnan(row.frac_surge):
            if row.flooding_cubic_meters_surgeonly == 0:
                df_watershed_flood_attribution.loc[ind, "frac_surge"] = 0


    # confirm there are no missing values
    if df_watershed_flood_attribution.isna().sum().sum() > 0:
        print("WARNING: There are some missing values that need to be inspected")
        print("Number of missing values per column:")
        print(df_watershed_flood_attribution.isna().sum())

    return df_watershed_flood_attribution, df_total_flooding

#%% load swmm results
ds_sst = xr.open_dataset(f_sst_results)
df_sst = ds_sst.to_dataframe().reset_index()
df_watershed_flood_attribution, df_total_flooding = compute_wshed_scale_attribution(ds_sst)

# qaqc - verifying flood volumes are present
# how many missing values are there?


# compute number of missing values
n_na = df_sst.isna().sum().values[0]

if n_na > 0:
    print("There are {} node event combos with missing flood volumes. This represents {}% of the data.".format(
            n_na, round(n_na/len(df_sst), 2)*100))

else:
    print("All the expected node flood volumes appear to be valid, yay!")


#%% computing quantiles and attribution by flood return period
ds_bootstrap_rtrn = xr.open_dataset(f_bootstrap)

# isolate the compound, rainonly, and surgeonly results
ds_sst_compound = ds_sst.sel(freeboundary="False", norain = "False")
ds_sst_rainonly = ds_sst.sel(freeboundary="True", norain = "False")
ds_sst_surgeonly = ds_sst.sel(freeboundary="False", norain = "True")


# load and transform shapefiles
gdf_subs = gpd.read_file(f_shp_subs)
gdf_coast = gpd.read_file(f_shp_coast)
gdf_subs = gdf_subs.to_crs(proj)
gdf_coast = gdf_coast.to_crs(proj)

# ds_fld_dif = ds_sst_compound - ds_sst_freebndry

# compute flooding quantiles
ds_quants_fld = compute_specific_return_periods(ds_sst_compound, n_storms_per_year=len(ds_sst_compound.storm_id.values),
                                                                recurrence_interval_yrs = sst_recurrence_intervals)

ds_quants_fld["node_trns_flooding_cubic_meters"] = np.log10(ds_quants_fld["node_flooding_cubic_meters"]+.01)
df_quants_fld = ds_quants_fld.to_dataframe()
df_quants_fld = df_quants_fld.reset_index()
gdf_node_flding = gdf_nodes.merge(df_quants_fld, how = 'inner', left_on = "NAME", right_on = "node_id").drop("NAME", axis=1)
#%%
