#%% import libraries
from glob import glob
from datetime import datetime
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
max_runtime_min = 40 # maximum minutes of runtime allowable for each SWMM simulation
lst_alternative_routing_tsteps = [1,0.1] # these are the routing timesteps to try
continuity_error_thresh = 2 # (%) i want the flow routing continuity error to be less than this 
continuity_error_to_rerun = 2 # (%) this is used when I'm re-running models with smaller timestep
lst_tsteps_for_reruns = [0.05] # these should be smaller than the smallest in the lst_alternative_routing_tsteps if it is found this improves continuity errors
time_permitted_for_reruns_min = 360 # minutes for either re-running failed models or re-running those with high contuity error
min_improvement_to_warrant_another_sim = 0.1 # only relevant for 3 or more alternative tsteps; if the improvement to flow continuity error is less than this times the previous flow continuity error, don't bother running another simulation 
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
    if "freebndry" in lst_name_comp:
        freebndry = True
    if "norain" in lst_name_comp:
        norain = True
    if freebndry:
        sim_type = "rain_only"
    elif norain:
        sim_type = "surge_only"
    else:
        sim_type = "compound"
    return int(rz), int(yr), int(storm_id), str(freebndry), str(norain), sim_type

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

def return_analysis_end_date(rpt_path):
    with open(rpt_path, 'r', encoding='latin-1') as file:
        # Read all lines from the file
        lines = file.readlines()
    for line in lines:
        if "Analysis ended on" in line:
            end_line = line
    # parse analysis end time
    lst_end_line = end_line.split("on:")[-1].split(' ')
    lst_info_in_line = []
    for substring in lst_end_line:
        if len(substring) > 0:
            lst_info_in_line.append(substring)
    # day_of_week = lst_info_in_line[0]
    month = lst_info_in_line[1]
    day = lst_info_in_line[2]
    assumed_year = datetime.today().year
    time = lst_info_in_line[3]
    datetime_string = "{}-{}-{} {}".format(month, day, assumed_year, time)
    analysis_end_datetime = pd.to_datetime(datetime_string, format='%b-%d-%Y %H:%M:%S')
    return analysis_end_datetime

def check_rpt_results(f_inp, routing_tstep): # this must match the file naming pattern in _d6_running_swmm.py
    # f_inp_to_report = f_inp.split(".inp")[0] + "_rt" + str(routing_tstep) + ".inp"
    # use the rpt file that was copied to the backup folder
    import os
    if routing_tstep % 1 == 0:
        routing_tstep = int(routing_tstep)
    inp_name = f_inp.split("/")[-1]
    swmm_fldr = f_inp.split(inp_name)[0]
    rpt_name = inp_name.split(".inp")[0]  + "_rt" + str(routing_tstep) + ".rpt"
    rpt_copy_fldr = swmm_fldr + "rpt_backup/"
    rpt_path_success = rpt_copy_fldr + rpt_name
    rpt_path_incomplete = swmm_fldr + rpt_name
    if os.path.exists(rpt_path_success):
        analysis_end_datetime = return_analysis_end_date(rpt_path_success)
    elif os.path.exists(rpt_path_incomplete):
        analysis_end_datetime = return_analysis_end_date(rpt_path_incomplete)
    else:
        analysis_end_datetime = np.nan
    return analysis_end_datetime

def return_flood_losses_and_continuity_errors(swmm_rpt, f_inp):
    # from pyswmm import Nodes
    import swmmio
    with open(swmm_rpt, 'r', encoding='latin-1') as file:
        # Read all lines from the file
        lines = file.readlines()
        # file.close()
    # verify validity of rpt file
    valid = False
    for line in lines:
        if "Element Count" in line:
            valid = True
            break
    if valid == False:
        sys.exit("The RPT file seems to not contain any information: {}".format(swmm_rpt))
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
        if "Flow Units" in line:
            line_flw_untits = line
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
    flow_units = line_flw_untits.split(".")[-1].split("\n")[0].split(" ")[-1].lower()
    # the rpt file only has nodes with nonzero flooding but I need to account for all nodes
    # create pandas series of node flood summaries
    ## return ids of all nodes
    model = swmmio.Model(f_inp)
    ar_nodes = model.nodes.dataframe.index.values
    df_allnodes = pd.DataFrame(dict(
        node_id = ar_nodes,
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
            node_id = ar_nodes,
            flood_volume = np.zeros(len(ar_nodes))
        ))
        df_node_flooding = df_node_flooding.set_index("node_id")
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
    return s_node_flooding,system_flooding,runoff_continuity_error_perc,flow_continuity_error_perc,frac_diff_node_minus_system_flood,flow_units










# %%
