#%% import libraries and load directories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyswmm import Output, Node, Nodes
from swmm.toolkit.shared_enum import NodeAttribute
from glob import glob
import sys
from tqdm import tqdm
from _inputs import b_processing_design_strms


dir_swmm_design_storms, f_design_strms, cubic_feet_per_cubic_meter = b_processing_design_strms()

#%% load output data
fs_outputs = glob(dir_swmm_design_storms+"*.out")

#%% functions
def parse_outputs_f(f):
    if "nosurge" in f:
        surge_return_period = "mean_high_water_level"
        rain_return_period = int(f.split("_")[-2].split(".out")[0].split("yr")[0])
    else:
        surge_return_period = "same_as_rainfall"
        rain_return_period = int(f.split("_")[-1].split(".out")[0].split("yr")[0])
    return rain_return_period, surge_return_period

#%% compile outputs into a pandas dataframe
lst_rain_return_pd = []
lst_surge_return_pd = []
lst_tot_flding = []
for f in tqdm(fs_outputs):
    rain_rtrn_pd, surge_rtrn_pd = parse_outputs_f(f)
    with Output(f) as out:
        lst_tot_node_flding = []
        # lst_keys= []
        # lst_tseries = []
        for key in out.nodes:
            # lst_keys.append(key)
            d_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
            tstep_seconds = float(pd.Series(d_t_series.index).diff().mode().dt.seconds)
            # convert from cfs to cf per tstep then cubic meters per timestep
            d_t_series = d_t_series * tstep_seconds * cubic_feet_per_cubic_meter
            # sum all flooded volumes
            lst_tot_node_flding.append(d_t_series.sum())
        lst_rain_return_pd.append(rain_rtrn_pd)
        lst_surge_return_pd.append(surge_rtrn_pd)
        lst_tot_flding.append(sum(lst_tot_node_flding))
        
df = pd.DataFrame(dict(rain_return_period = lst_rain_return_pd,
                       surge_return_period = lst_surge_return_pd,
                        total_event_flooding_MG = lst_tot_flding))

#%% write to csv
df.to_csv(f_design_strms, index=False)