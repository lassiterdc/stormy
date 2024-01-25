#%% import libraries
import pandas as pd
from swmm.toolkit.shared_enum import NodeAttribute
from pyswmm import Simulation, Output
from _inputs import *
from _utils import *


# define files
# define folders
fldr_models = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/_scratch/models/"
fldr_models_neg = fldr_models + "most_negative_interaction/"
fldr_models_neg_rivanna_jan2024 = fldr_models_neg + "models_exactly_as_they_are_on_rivanna/"
fldr_models_neg_rivanna_dec2023 = fldr_models_neg_rivanna_jan2024 + "original_results/"
fldr_models_neg_local_pyswmm = fldr_models_neg + "running_using_pyswmm/"
fldr_models_neg_local_PCSWMM = fldr_models_neg + "running_locally/"
nc_models_neg_rerun_1srouting_jan23_2024 = fldr_models_neg + "re-run_on_rivanna_smaller_routing_tstep/model_outputs_consolidated.nc"
nc_year764_jan24_2024 = fldr_models_neg + "re-run_on_rivanna_reporting_from_rpt/_model_outputs_year764.nc"


# isolate events with negative interaction
df_watershed_flood_attribution_neg_inter = df_watershed_flood_attribution[df_watershed_flood_attribution.frac_interaction<0]
df_volumes = df_watershed_flood_attribution_neg_inter.loc[:,["flooding_cubic_meters_rainonly","flooding_cubic_meters_surgeonly", "flooding_cubic_meters_compound"]]

s_interaction = df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_compound-(df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_rainonly + df_watershed_flood_attribution_neg_inter.flooding_cubic_meters_surgeonly)
s_interaction = s_interaction*-1
s_interaction.name = "flooding_negative_cubic_meters_interaction"

df_volumes = pd.concat([df_volumes,s_interaction], axis = 1)

# manually investigating the event with the largest negative interaction
# find the index of the most negative interaction simulated
idx_largest_neg_inter = df_volumes.idxmax().flooding_negative_cubic_meters_interaction

df_largest_neg_interaction = df_watershed_flood_attribution.loc[idx_largest_neg_inter,:]

print("(realization, year, storm_id) = {}".format(idx_largest_neg_inter))

df_swmm_summaries = pd.read_csv(f_sst_event_summaries).set_index(["realization", "year", "storm_id"])

df_swmm_model_most_neg = df_swmm_summaries.loc[idx_largest_neg_inter,:]

def return_lst_outs(fldr):
    lst_outs = []
    for f in glob(fldr+"*"):
        if ".out" in f:
            lst_outs.append(f)
    return lst_outs

def return_lst_inps_and_outs(fldr):
    lst_inps = []
    lst_outs = []
    for f in glob(fldr+"*"):
        if ".inp" in f:
            lst_inps.append(f)
            f_out = f.split(".")[0]+".out"
            lst_outs.append(f_out)
    return lst_inps, lst_outs

# create lists of input and output files
lst_pcswmm_out_path = return_lst_outs(fldr_models_neg_local_PCSWMM)
lst_pyswmm_inp_path, lst_pyswmm_out_path = return_lst_inps_and_outs(fldr_models_neg_local_pyswmm)
lst_rivanna_ins, lst_rivanna_outs = return_lst_inps_and_outs(fldr_models_neg_rivanna_jan2024)

#%% inspecting reurn from 1/23 with 1 second routing timestep for all models with a negative interaction
ds_sst_rerun = xr.open_dataset(nc_models_neg_rerun_1srouting_jan23_2024)
df_sst_rerun = ds_sst_rerun.to_dataframe().reset_index()

df_watershed_flood_attribution_1srouting, df_total_flooding_1srouting = compute_wshed_scale_attribution(ds_sst_rerun)

# negative interactions from original model run
df_watershed_flood_attribution_neg_inter = df_watershed_flood_attribution[df_watershed_flood_attribution.frac_interaction<0]

# negative interactions from rerun
df_watershed_flood_attribution_neg_inter_rerun = df_watershed_flood_attribution_1srouting[df_watershed_flood_attribution_1srouting.frac_interaction<0]

df_inter_compare = df_watershed_flood_attribution_neg_inter.join(df_watershed_flood_attribution_1srouting, rsuffix="_rerun")
df_inter_compare.to_csv("_scratch/attribution_comparison.csv")
#%% comparing with re-run on 1/24 that used the rpt to calculate flooding and had instability resolved
ds_sst_rerun_1_24_2024 = xr.open_dataset(nc_year764_jan24_2024)
df_sst_rerun_1_24_2024 = ds_sst_rerun_1_24_2024.to_dataframe().reset_index()

df_watershed_flood_attribution_1_24_2024, df_total_floodin_1_24_2024 = compute_wshed_scale_attribution(ds_sst_rerun_1_24_2024)


#%% work - trying to figure out how to compute errors
# when run on PCSWMM, this has a routing continuity error of 85.6%
f_model_with_instability = fldr_models_neg_local_PCSWMM + "reproducing_instability/onpyswmm/rz1_yr764_strm5.inp"


with Simulation(f_model_with_instability) as sim:
    for step in sim:
        pass
    # this is required because of a pyswmm issue see  https://github.com/pyswmm/pyswmm/issues/236
    sim._model.swmm_end()
    runoff_error = sim.runoff_error
    flow_routing_error = sim.flow_routing_error
        # lst_flow_errors.append(flow_routing_error)
        # lst_runoff_errors.append(runoff_error)

print("Flow routing error: {}".format(str(flow_routing_error)))
print("Runoff error: {}".format(str(runoff_error)))

#%% working on creating a loop to decrease routing timestep until either it's acceptable or 
# until the last option is reached
import time
continuity_error_thresh = 1.5

alt_routing_steps = [10, 5, 1, 0.5]
lst_flow_errors = []
lst_runoff_errors = []
lst_sim_runtimes_s = []
lst_routing_tsteps_tried = []
for routing_tstep in alt_routing_steps:
    with open(f_model_with_instability, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()
    for i, line in enumerate(lines):
        if "ROUTING_STEP" in line:
            line_of_interest = line
            # Modify the text in that line
            # write the first part of the line (to make sure I'm getting the spacing right)
            first_part = ""
            for substring in line_of_interest.split(' ')[0:-1]:
                # spaces are collapsed when splitting by spaces,
                # so when a blank string is the substring that means it was
                # a space in the original string
                # if substring == "":
                #     substring = " "
                first_part += substring + " "
            # write the full line to replace the original with
            newline = first_part + str(routing_tstep) + "\n"
            lines[i] = line.replace(line_of_interest, newline)

    with open(f_model_with_instability, 'w') as file:
        file.writelines(lines)
    # run the simulation
    ## time it
    start_time = time.time()
    with Simulation(f_model_with_instability) as sim:
        for step in sim:
            pass
        # this is required because of a pyswmm issue see  https://github.com/pyswmm/pyswmm/issues/236
        sim._model.swmm_end()
        runoff_error = sim.runoff_error
        flow_routing_error = sim.flow_routing_error
        elapsed_time = time.time() - start_time
        rounded_elapsed_time = round(elapsed_time , 2)
        lst_flow_errors.append(flow_routing_error)
        lst_runoff_errors.append(runoff_error)
        lst_sim_runtimes_s.append(rounded_elapsed_time)
        lst_routing_tsteps_tried.append(routing_tstep)
    if (abs(flow_routing_error) <= continuity_error_thresh) and (abs(runoff_error) <= continuity_error_thresh):
        print("A routing timestep of {} yielded routing and runoff continuity errors less than {}% ".format(
            routing_tstep, continuity_error_thresh))
        break
    else:
        print("The simulation was run with a routing timestep of {}".format(routing_tstep))
        print("The runoff continuity error was {}% and the flow error was {}%".format(
            runoff_error, flow_routing_error
        ))
        print("Re-running simulation with a smaller routing timestep")
#%%
pd.DataFrame(dict(
    routing_tstep = lst_routing_tsteps_tried,
    routing_error = lst_flow_errors,
    runoff_error = lst_runoff_errors,
    sim_runtime = lst_sim_runtimes_s
))
#%% figuring out alternative ways to compute flooding


# compute flooding from node time series


#%% work troubleshooting function
f_swmm_out = f_model_with_instability.split(".inp")[0] + ".out"

df_total_flooding_instability_testing = compute_total_node_flooding([f_swmm_out])

#%% (takes a minute or two) processing the local simulations
df_total_flooding_pcswmm = compute_total_node_flooding(lst_pcswmm_out_path)

#%% (running sims takes about 7-10 minutes each, plus another few minutes to process outputs) 
# run simulations using pyswmm
from datetime import datetime
max_runtime_min = 15
run_sims = False
run_compound_only = True
lst_runoff_error = []
lst_flow_error = []

if run_sims:
    for f_inp in lst_pyswmm_inp_path:
        if run_compound_only:
            if "freebndry" in f_inp:
                continue
            if "norain" in f_inp:
                continue
        print("Running simulation: {}".format(f_inp.split("/")[-1].split("\\")[-1]))
        with Simulation(f_inp) as sim:
            sim_start_time = datetime.now()
            for step in sim:
                sim_time = datetime.now()
                sim_runtime_min = round((sim_time - sim_start_time).seconds / 60, 1)
                runoff_error = sim.runoff_error
                flow_routing_error = sim.flow_routing_error
                lst_runoff_error.append(runoff_error)
                lst_flow_error.append(flow_routing_error)
                if sim_runtime_min > max_runtime_min:
                    problem = "User-defined maximum simulation time limit of {} minutes reached, so simulation was halted.".format(max_runtime_min)
                    print(problem)
                    # success = False
                    # break
                pass

df_total_flooding_pyswmm = compute_total_node_flooding(lst_pyswmm_out_path)

#%% inspecting pcswmm results
print("Flood volumes in local PCSWMM simulation:")
df_total_flooding_pcswmm
#%% inspecting pyswmm results
print("Flood volumes in local pyswmm simulation:")
df_total_flooding_pyswmm

#%% inspecting output netcdf generated by the running_swmm script on Rivanna
print("Investigating the original runs on the Rivanna from back in December 2023")

f_ds_out =  fldr_models_neg_rivanna_dec2023 + "_model_outputs_year764.nc"

ds_out_from_rivanna = xr.open_dataset(f_ds_out)

df_total_flooding_neg_inter_rivanna = ds_out_from_rivanna.sum(dim = "node_id").to_dataframe()

df_total_flooding_neg_inter_rivanna = df_total_flooding_neg_inter_rivanna.reset_index()       

df_total_flooding_neg_inter_rivanna = classify_sims(df_total_flooding_neg_inter_rivanna)

df_total_flooding_neg_inter_rivanna.set_index(["realization", "year", "storm_id"]).sort_index().loc[(1,764,5),:]

#%% I re-ran the simulations on the Rivanna and re-generated the netcdf file and maintained the swmm .out files in Jan 2024
print("Inspecting model results re-run on the Rivanna on 1/17/2024")
f_ds_out = fldr_models_neg_rivanna_jan2024 + "_model_outputs_year764.nc"

ds_out_from_rivanna = xr.open_dataset(f_ds_out, engine = "netcdf4")

df_total_flooding_neg_inter_rivanna = ds_out_from_rivanna.sum(dim = "node_id").to_dataframe()

df_total_flooding_neg_inter_rivanna = df_total_flooding_neg_inter_rivanna.reset_index()       

df_total_flooding_neg_inter_rivanna = classify_sims(df_total_flooding_neg_inter_rivanna)

# print("Flood volumes in local simulation look fine:")
df_total_flooding_neg_inter_rivanna.set_index(["realization", "year", "storm_id"]).sort_index().loc[(1,764,5),:]

#%% (takes a few minutes) now loading the .out files and inspecting them directly
df_total_flooding_neg_inter = compute_total_node_flooding(lst_rivanna_outs)
print("Flood volumes in 1/17/24 simulation directly from the .out files:")
df_total_flooding_neg_inter

#%% comparing attributes of the .out files from the different runs
lst_pcswmm_out_path
lst_pyswmm_out_path
lst_rivanna_outs

# WORK - for testing
f_out1 = lst_pyswmm_out_path[0]
f_out2 = lst_rivanna_outs[0]
f1_desc = "local pyswmm compound sim"
f2_desc = "rivanna compound sim 1/17/24"
diff_thresh = 0.005
# END WORK
from swmm.toolkit.shared_enum import SystemAttribute
import swmmio
def compare_outputs(f_out1, f_out2, f1_desc, f2_desc, diff_thresh):
    with Output(f_out1) as out1:
        with Output(f_out2) as out2:
            model1 = swmmio.Model(f_out1.split(".out")[0] + ".inp")
            model2 = swmmio.Model(f_out2.split(".out")[0] + ".inp")
            lst_no_difference = []
            # swmm version
            if out1.version != out2.version:
                print("WARNING: SWMM ENGINES ARE DIFFERENT")
            else:
                lst_no_difference.append("SWMM engines")
            # units
            if out1.units != out2.units:
                print("WARNING: UNITS ARE DIFFERENT")
            else:
                lst_no_difference.append("Units")
            # time series
            out1_tseries = pd.Series(out1.times)
            out2_tseries = pd.Series(out2.times)
            ## starting tstep
            if out1_tseries.iloc[0] != out2_tseries.iloc[0]:
                print("WARNING: STARTING TIME STEP IS DIFFERENT")
            else:
                lst_no_difference.append("Starting time step")
            ## ending tstep
            if out1_tseries.iloc[-1] != out2_tseries.iloc[-1]:
                print("WARNING: ENDING TIME STEP IS DIFFERENT")
            else:
                lst_no_difference.append("Ending time step")
            ## number of tsteps
            if len(out1_tseries) != len(out2_tseries):
                print("WARNING: NUMBER OF TIME STEPS IS DIFFERENT")
            else:
                lst_no_difference.append("Number of time steps")
            first_tstep = out1.times[0]
            last_tstep = out1.times[-1]
            system_series_of_interest = [SystemAttribute.RAINFALL, SystemAttribute.DIRECT_INFLOW, SystemAttribute.RUNOFF_FLOW,
                                         SystemAttribute.TOTAL_LATERAL_INFLOW, SystemAttribute.FLOOD_LOSSES,
                                         SystemAttribute.OUTFALL_FLOWS, SystemAttribute.VOLUME_STORED]
            lst_difference = []
            lst_val1 = []
            lst_val2 = []
            lst_perc_change = []
            for att in system_series_of_interest:
                out1_series_sum = pd.Series(out1.system_series(att, first_tstep, last_tstep)).sum()
                out2_series_sum = pd.Series(out2.system_series(att, first_tstep, last_tstep)).sum()
                # if the absolute value of the difference is less than the diff_threshold variable, 
                # we are assuming the difference is insignificant
                if abs(out1_series_sum - out2_series_sum) > np.mean([out1_series_sum,out2_series_sum])*diff_thresh:
                    # print("Comparing sum of {}".format(str(att)))
                    # print("{}: {}".format(f1_desc, out1_series_sum))
                    # print("{}: {}".format(f2_desc, out2_series_sum))
                    # print("##############################################")
                    lst_difference.append("sum({})".format(str(att)))
                    # lst_diff_sim1.append(f1_desc)
                    # lst_diff_sim2.append(f2_desc)
                    lst_val1.append(out1_series_sum)
                    lst_val2.append(out2_series_sum)
                    lst_perc_change.append((out2_series_sum - out1_series_sum )/out1_series_sum*100)
                else:
                    lst_no_difference.append(str(att))

            df_system_differences = pd.DataFrame({
                "attribute":lst_difference, f1_desc:lst_val1,
                f2_desc:lst_val2, "percent_change":lst_perc_change
            })
            # inspecting nodes
            outfalls1 = model1.inp.outfalls
            outfalls2 = model2.inp.outfalls
            if (outfalls1 != outfalls2).sum().sum() > 0:
                print("WARNING: There are differences in outfall parameters")
            ## inspecting outfall flows
            lst_outfall_series_to_inspect = [NodeAttribute.HYDRAULIC_HEAD,
                                             NodeAttribute.LATERAL_INFLOW,
                                             NodeAttribute.TOTAL_INFLOW,
                                             NodeAttribute.FLOODING_LOSSES]
            lst_outfall = []
            lst_difference = []
            lst_val1 = []
            lst_val2 = []
            lst_perc_change = []
            lst_out1_head_tseries = []
            lst_out2_head_tseries = []
            lst_out1_flow_tseries = []
            lst_out2_flow_tseries = []
            for node_id, row in outfalls2.iterrows():
                for outfall_attr in lst_outfall_series_to_inspect:
                    out1_series_sum = pd.Series(out1.node_series(node_id, outfall_attr, first_tstep, last_tstep)).sum()
                    out2_series_sum = pd.Series(out2.node_series(node_id, outfall_attr, first_tstep, last_tstep)).sum()
                    # if the absolute value of the difference is less than the diff_threshold variable, 
                    # we are assuming the difference is insignificant
                    if abs(out1_series_sum - out2_series_sum) > np.mean([out1_series_sum,out2_series_sum])*diff_thresh:
                        lst_difference.append("sum({})".format(str(outfall_attr)))
                        lst_outfall.append(node_id)
                        lst_val1.append(out1_series_sum)
                        lst_val2.append(out2_series_sum)
                        lst_perc_change.append((out2_series_sum - out1_series_sum )/out1_series_sum*100)
                    else:
                        lst_no_difference.append(str(outfall_attr))
                    if "HYDRAULIC_HEAD" in str(outfall_attr):
                        out1_series = pd.Series(out1.node_series(node_id, outfall_attr, first_tstep, last_tstep))
                        out1_series.name = node_id
                        out2_series = pd.Series(out2.node_series(node_id, outfall_attr, first_tstep, last_tstep))
                        out2_series.name = node_id
                        lst_out1_head_tseries.append(out1_series)
                        lst_out2_head_tseries.append(out2_series)
                    if "TOTAL_INFLOW" in str(outfall_attr):
                        out1_series = pd.Series(out1.node_series(node_id, outfall_attr, first_tstep, last_tstep))
                        out1_series.name = node_id
                        out2_series = pd.Series(out2.node_series(node_id, outfall_attr, first_tstep, last_tstep))
                        out2_series.name = node_id
                        lst_out1_flow_tseries.append(out1_series)
                        lst_out2_flow_tseries.append(out2_series)
            df_outfall_differences = pd.DataFrame({
                "outfall":lst_outfall,
                "attribute":lst_difference, f1_desc:lst_val1,
                f2_desc:lst_val2, "percent_change":lst_perc_change
            })

            df_head_tseries_1 = pd.concat(lst_out1_head_tseries, axis = 1)
            df_head_tseries_2 = pd.concat(lst_out2_head_tseries, axis = 1)
            df_flow_tseries_1 = pd.concat(lst_out1_flow_tseries, axis = 1)
            df_flow_tseries_2 = pd.concat(lst_out2_flow_tseries, axis = 1)

    return lst_no_difference, df_system_differences, df_outfall_differences, df_head_tseries_1, df_head_tseries_2, df_flow_tseries_1, df_flow_tseries_2
#%%
lst_no_difference, df_system_differences, df_outfall_differences,\
df_head_tseries_1, df_head_tseries_2,df_flow_tseries_1,df_flow_tseries_2  = compare_outputs(f_out1=lst_pyswmm_out_path[0], f_out2=lst_rivanna_outs[0],
                 f1_desc = "local pyswmm compound sim",
                 f2_desc = "rivanna compound sim 1/17/24", diff_thresh=0.005)

# look at the difference in the hydraulic head in the outfall
node_ids = df_head_tseries_1.columns
fig, axes = plt.subplots(ncols=2, figsize = (8, 4), sharex = True, sharey = True)
col = node_ids[0]
df_head_tseries_1[col].plot(ax = axes[0], label = "local sim", linestyle = (0, (1, 2)))
df_head_tseries_2[col].plot(ax = axes[0], label = "rivanna", linestyle = "dotted")
(df_head_tseries_1[col]-df_head_tseries_2[col]).plot(ax = axes[0], label = "difference")
axes[0].set_title(col)
axes[0].set_xlabel("Hydraulic Head (ft)")
# plt.legend()

col = node_ids[1]
df_head_tseries_1[col].plot(ax = axes[1], label = "local sim", linestyle = (0, (1, 2)))
df_head_tseries_2[col].plot(ax = axes[1], label = "rivanna", linestyle = "dotted")
(df_head_tseries_1[col]-df_head_tseries_2[col]).plot(ax = axes[1], label = "difference")
axes[1].set_title(col)
plt.legend()
plt.tight_layout()


# look at the difference in the flow in the outfall
node_ids = df_flow_tseries_1.columns
fig, axes = plt.subplots(ncols=2,nrows=2, figsize = (10, 8), sharex = True)
col = node_ids[0]
df_flow_tseries_1[col].plot(ax = axes[0,0], label = "local sim", linestyle = (0, (1, 2)),c = "blue")
df_flow_tseries_2[col].plot(ax = axes[1,0], label = "rivanna", linestyle = "dotted",c = "red")
# (df_flow_tseries_1[col]-df_flow_tseries_2[col]).plot(ax = axes[0,0], label = "difference",c = "green")
axes[0,0].set_title(col)
# axes[0,0].set_xlabel("Total Inflow (cfs)")


col = node_ids[1]
df_flow_tseries_1[col].plot(ax = axes[0,1], label = "local sim", linestyle = (0, (1, 2)),c = "blue")
df_flow_tseries_2[col].plot(ax = axes[1,1], label = "rivanna", linestyle = "dotted",c = "red")
# (df_flow_tseries_1[col]-df_flow_tseries_2[col]).plot(ax = axes[1,0], label = "difference",c = "green")
# axes[1,0].set_title(col)
# plt.legend()
axes[0,1].set_title(col)

axes[0,0].legend()
axes[1,0].legend()
axes[0,1].legend()
axes[1,1].legend()
plt.tight_layout()

