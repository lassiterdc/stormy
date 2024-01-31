#%% import libraries
from glob import glob
# filenames, paths, and directories
# dir_repo = "/project/quinnlab/dcl3nd/norfolk/stormy/" 
dir_repo = "/scratch/dcl3nd/stormy/" # MODIFICATION TO RUN ON SCRATCH
dir_sst = dir_repo + "stochastic_storm_transposition/"
dir_sst_nrflk = dir_sst + "norfolk/"
# dir_sst_nrflk_hrly = dir_sst_nrflk + "sst_mrms_hourly/"
dir_mrms = dir_sst_nrflk + "sst_mrms/"
# dir_home = "/home/dcl3nd/stormy/"
# dir_home_sst = dir_home + "sst/"
# dir_scratch_sst = "/scratch/dcl3nd/stormy/"
dir_highres_repo = "/scratch/dcl3nd/highres-radar-rainfall-processing/"



# WORK
# f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined.sst"
# f_sst_nrflk_hrly_parameterfile = dir_sst_nrflk_hrly + "mrms_hourly_combined_test.sst"
# END WORK
# f_sst_nrflk_hrly_combined_catalog = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined.nc"
# f_sst_nrflk_hrly_combined_catalog_reformatted = dir_sst_nrflk_hrly + "strmcat_mrms_hourly_combined_reformatted_for_xarray.nc"
# swmm stuff
dir_swmm_model = dir_repo + "swmm/hague/"
f_shp_swmm_subs = dir_swmm_model + "swmm_model/exported_layers/subcatchments.shp"
lst_template_keys = ["START_DATE", "START_TIME", "REPORT_START_DATE", "REPORT_START_TIME", "END_DATE", "END_TIME", "rainfall_0", "rainfall_1", "rainfall_2","rainfall_3", "rainfall_4", "rainfall_5", "water_level", "OF_TYPE", "STAGE_DATA"]
# work_f_water_level_path = dir_swmm_model + "swmm_timeseries/a_water_levels_ft.dat"

# script c3
# WORK 
# dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined/Realizations/"
# dir_sst_realizations_hrly = dir_sst_nrflk_hrly + "mrms_hourly_combined_tst/Realizations/"
# END WORK
# c4
# BEGIN WORK
# dir_swmm_sst_scenarios_proj = dir_swmm_model + "swmm_scenarios_sst/"
# dir_swmm_sst_scenarios_scratch = dir_scratch_sst + "swmm_sst/"
# dir_swmm_sst_scenarios_home = dir_home_sst + "swmm_sst/"
# dir_time_series = dir_swmm_sst_scenarios_home + "time_series/"
# f_key_subnames_gridind = dir_time_series + "_key_subnames_and_grid-indices.csv"
# seed_mrms_hourly = 22901
# END WORK

# c4b 
dir_ssr = dir_repo + "stochastic_storm_rescaling/"
dir_ssr_outputs = dir_ssr + "outputs/"
dir_mrms_events = dir_ssr_outputs + "c_mrms_events/"
f_mrms_event_summaries = dir_mrms_events + "mrms_event_summaries.csv"
f_mrms_event_timeseries = dir_mrms_events + "mrms_event_timeseries.csv"
dir_noaa_water_levels = dir_ssr_outputs + "a_NOAA_water_levels/"
f_water_level_storm_surge = dir_noaa_water_levels + "a_water-lev_tide_surge.csv"
sst_hrly_tstep_min = 60
time_buffer = 6 # hours; this is the amount of time before either the start of rain or the peak storm surge and AFTER the end of rain or peak storm surge; this will also determine simulation start and end dates

c4b_gen_plots = False
wlevel_threshold = 0.5 # i don't want simulated time series that are 50% above or below the min and max observed waterlevel since 2000

n_attempts = 500
n_clusters = 5
resampling_inteval = 10 # after a number of attempts equal to a multiple of this value, the copula will be used to resample storm surge and peak lag  



#d
# dir_mrms = dir_sst_nrflk + "sst_mrms/"
dir_mrms_coarse = dir_mrms + "hourly/"
dir_mrms_fullres = dir_mrms + "mrms_combined/StormCatalog/" # this should be the same as the RainyDay scenario name
f_sst_mrms_coarse =  dir_mrms_coarse + "mrms_hourly_template.sst"
dir_for_sst_files = dir_mrms_coarse + "_inputs/"
dir_fullres_rain = dir_highres_repo + "data/mrms_nc_preciprate_fullres_dailyfiles_constant_tstep/"
# f_sst_mrms = dir_mrms + "mrms_template.sst"

#d4
fldr_realizations = dir_mrms + "mrms_combined/Realizations/"
dir_swmm_sst_scenarios = dir_swmm_model + "swmm_scenarios_sst/"
# dir_swmm_sst_scenarios_scratch = dir_scratch_sst + "swmm_sst_hourly/"
# dir_swmm_sst_scenarios_hrly_home = dir_home_sst + "swmm_sst_hourly/"
dir_time_series = "/home/dcl3nd/stormy/sst/" + "swmm_time_series/"
f_key_subnames_gridind = dir_time_series + "_key_subnames_and_grid-indices.csv"
seed_mrms_hourly = 22901

# d4b
dir_local_outputs = dir_sst + "local/outputs/"
f_sst_event_summaries = dir_local_outputs + "b_sst_event_summaries.csv"
f_waterlevels_same_tstep_as_sst = dir_local_outputs + "b_observed_waterlevels_sst_timestep.csv"
f_observed_wlevel_rainfall_tseries = dir_local_outputs + "b_observed_compound_event_timeseries.csv"
f_simulated_cmpnd_event_summaries = dir_local_outputs + "c_simulated_compound_event_summary.csv"
f_observed_cmpnd_event_summaries = dir_local_outputs + "c_observed_compound_event_summaries_with_k.csv"
dir_swmm_sst_scenarios_scratch = dir_sst + "_scratch/"
dir_event_summary_csv_scratch = dir_swmm_sst_scenarios_scratch + "event_summaries/"
dir_waterlevel_scratch = dir_swmm_sst_scenarios_scratch + "water_levels/"
dir_rain_weather_scratch = dir_swmm_sst_scenarios_scratch + "weather/"
dir_rain_weather_scratch_ncs = dir_rain_weather_scratch + "netcdfs/"
plot_weather_gen_stuff = False
plt_fldr_weather_gen = dir_time_series + "_plots/"

# d4c
dir_scenario_weather = dir_swmm_sst_scenarios + "weather/"
f_rain_realizations = dir_scenario_weather + "rainfall.nc"
f_w_level_sims = dir_scenario_weather + "water_levels.nc"
f_sims_summary = dir_scenario_weather + "compound_event_summaries.csv"
f_sims_wlevel_tseries = dir_scenario_weather + "simulated_wlevel_timeseries.csv"

# d5
f_inp_base = dir_swmm_sst_scenarios + "hague_sst_model_template.inp"
dir_swmm_sst_models = dir_sst + "swmm_scenarios/"

# c5
# # dir_swmm_sst_models_hrly = dir_swmm_sst_scenarios_hrly_proj + "models/"
# dir_swmm_sst_models_hrly = dir_swmm_sst_scenarios_hrly_scratch + "models/"

# f_inp_base_hrly = dir_swmm_sst_scenarios_hrly_proj + "hague_sst_model_template.inp"

# # dir_time_series_hrly = dir_swmm_sst_scenarios_hrly + "time_series/"
f_swmm_scenarios_catalog = dir_swmm_sst_scenarios + "swmm_scenario_catalogs/" + "_swmm_scenarios_catalog_yr{}.csv"

norain_gage_name = "no_rain"

# # c6
max_runtime_min = 20 # maximum minutes of runtime allowable for each SWMM simulation
lst_alternative_routing_tsteps = [2,1,0.5] # these are the routing timesteps to try
continuity_error_thresh = 1 # (%) i want the flow routing continuity error to be less than this 
min_improvement_to_warrant_another_sim = 0.1 # if the improvement to flow continuity error is less than this times the previous flow continuity error, don't bother running another simulation 
cubic_meters_per_gallon = 0.00378541

# # c6b
f_model_perf_summary = dir_swmm_sst_scenarios + "model_performance_summary.csv"
f_events_summary = dir_swmm_sst_scenarios + "event_summaries.csv"

# # c7
f_model_outputs_consolidated = dir_swmm_sst_scenarios + "model_outputs_consolidated.nc"
# # dir_swmm_sst_models_hrly_home = dir_swmm_sst_scenarios_hrly_home + "models/"

# # c8
f_bootstrapped_quant_estimates = dir_swmm_sst_scenarios_scratch + "models/boostrapping/"
sst_recurrence_intervals = [0.5, 1, 2, 5, 10, 25, 50, 100]
f_bootstrapped_consolidated = dir_swmm_sst_scenarios + "bootstrapping_consolidated.nc"
f_bootstrapped_consolidated_raw = dir_swmm_sst_scenarios + "bootstrapping_allsamples_consolidated.nc"
export_raw_bs_samps = False
#%% hard coded variables
# name_out_realizations = "_combined_realizations.nc"
f_realizations_hourly = dir_swmm_model + "swmm_scenarios_sst_hourly/_combined_realizations.nc"
mm_per_inch = 25.4
grid_spacing = 0.009999999776482582
start_date = "2020-09-01" # start date for each of the time series
meters_per_foot = 0.3048
square_meters_per_square_foot = meters_per_foot * meters_per_foot
cubic_meters_per_cubic_foot = meters_per_foot*meters_per_foot*meters_per_foot

# SST parameters (should line up with the SST input file)
# WORK 
# nyears = 1000 # should be 1,000 for the final experiment
# nperyear = 20
# nrealizations = 1

sst_tstep_min = 5
nstormsperyear = 5
# nyears = 2
# nperyear = 20
# nrealizations = 2
# END WORK

#%% functions
import pandas as pd
import sys
import numpy as np
import xarray as xr

def parse_inp(f_inp):
    # format "rz{}_yr{}_strm{}.inp"
    lst_name_comp = f_inp.split("/")[-1].split("_")
    rz = lst_name_comp[0].split("rz")[-1]
    yr = lst_name_comp[1].split("yr")[-1]
    storm_id = lst_name_comp[2].split(".")[0].split('strm')[-1]
    freebndry = False
    norain = False
    if "freebndry" in lst_name_comp[-1]:
        freebndry = True
    if "norain" in lst_name_comp[-1]:
        norain = True
    return int(rz), int(yr), int(storm_id), str(freebndry), str(norain)

# for loading RainyDay realizations
def define_dims(ds):
    fpath = ds.encoding["source"]
    lst_f = fpath.split("/")[-1].split("_")
    rz = int(lst_f[0].split("rz")[-1])
    year = int(lst_f[1].split("y")[-1])
    strm = int(lst_f[2].split("stm")[-1].split(".")[0])
    first_tstep = ds.time.values[0]
    tseries = pd.Series(ds.time.values)
    tsteps_unique = tseries.diff().dropna().unique()
    if len(tsteps_unique) > 1:
        sys.exit("variable time step encountered in file {}".format(fpath))
    tstep_min = tsteps_unique[0] / np.timedelta64(1, "m")
    tstep_ind = np.arange(len(tseries))
    ds["time"] = tstep_ind
    ds = ds.assign_attrs(timestep_min = tstep_min)
    ds = ds.assign_coords(dict(realization=rz, year = year, storm_id = strm))
    ds = ds.expand_dims(dim=dict(realization=1, year = 1, storm_id = 1))
    # drop unnecessary variables
    try:
        ds = ds.drop_vars(["xlocation","ylocation", "scalar_dim"]) # these were causing problems when trying to combine all weather realizations into a single netcdf
    except:
        pass
    return ds

# for loading RainyDay realizations for a specific year only
def return_rzs_for_yr(fldr_realizations, yr):
    lst_f_all_ncs = glob(fldr_realizations+"*.nc")
    lst_f_ncs = []
    for f in lst_f_all_ncs:
        lst_f = f.split("/")[-1].split("_")
        # rz = int(lst_f[0].split("rz")[-1])
        year = int(lst_f[1].split("y")[-1])
        # strm = int(lst_f[2].split("stm")[-1].split(".")[0])
        if year == yr:
            lst_f_ncs.append(f)
    lst_f_ncs.sort()
    return lst_f_ncs

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
    if encountered_flow_routing_continuity == False:
        print("WARNING: RPT file does not contain necessary fields")
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

    if len(flood_volumes) > 0: # if there is flooding in the model
        df_node_flooding_subset = pd.DataFrame(dict(
            node_id = node_ids,
            flood_volume = flood_volumes
        ))
        df_node_flooding_subset.set_index("node_id", inplace = True)
        df_node_flooding = df_allnodes.join(df_node_flooding_subset)
        # for the nodes not in the rpt, assign them a flood volume of 0
        df_node_flooding = df_node_flooding.fillna(0)
    else: # if there is no flooding in the model
        df_node_flooding = pd.DataFrame(dict(
            node_id = lst_nodes,
            flood_volume = np.zeros(len(lst_nodes))
        ))

    s_node_flooding = df_node_flooding.flood_volume

    # return runoff and flow continuity
    runoff_continuity_error_perc = float(runoff_continuity_error_line.split(" ")[-1].split("\n")[0])
    flow_continuity_error_perc = float(flow_continuity_error_line.split(" ")[-1].split("\n")[0])

    # return system flood losses
    system_flooding = float(system_flood_loss_line.split(" ")[-1].split("\n")[0])
    if (s_node_flooding.sum() > 0) and (system_flooding > 0):
        frac_diff_node_minus_system_flood = (s_node_flooding.sum() - system_flooding)/system_flooding
    else:
        frac_diff_node_minus_system_flood = np.nan

    return s_node_flooding,system_flooding,runoff_continuity_error_perc,flow_continuity_error_perc,frac_diff_node_minus_system_flood









