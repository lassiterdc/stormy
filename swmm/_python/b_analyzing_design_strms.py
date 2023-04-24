#%% import libraries and load directories
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyswmm import Output, Node, Nodes
from swmm.toolkit.shared_enum import NodeAttribute
from swmm.toolkit.shared_enum import SubcatchAttribute
from glob import glob
import sys
from tqdm import tqdm
from _inputs import b_processing_design_strms


dir_swmm_design_storms, f_design_strms, cubic_feet_per_cubic_meter, volume_units, mm_per_inch, lst_outfall_ids, sub_id_for_rain = b_processing_design_strms()

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
lst_rain_tot = []
lst_max_wlevel = []
# variables I want ["total_flooding_1e+06m3", "max_sim_wlevel", "depth_mm"]

for f in tqdm(fs_outputs):
    rain_rtrn_pd, surge_rtrn_pd = parse_outputs_f(f)
    with Output(f) as out:
        lst_tot_node_flding = []
        # lst_keys= []
        # lst_tseries = []
        rain_t_series = pd.Series(out.subcatch_series(sub_id_for_rain, SubcatchAttribute.RAINFALL)) # in/hr  assumes all subs have the same rainfall inputs
        tstep_hr = float(pd.Series(rain_t_series.index).diff().mode().dt.seconds)  / 60 / 60 # min per sec * hr per min
        tot_rain_mm = (rain_t_series * tstep_hr).sum()*mm_per_inch # (in/hr * hr/tstep).sum() = total inches * mm/in = mm
        lst_rain_tot.append(tot_rain_mm)
        lst_tmp_wlevel_max = []
        for out_id in lst_outfall_ids:
            wlevel_tseries = pd.Series(out.node_series(out_id, NodeAttribute.HYDRAULIC_HEAD))
            max_wlevel = wlevel_tseries.max() # feet * 
            lst_tmp_wlevel_max.append(max_wlevel)
        close = np.isclose(lst_tmp_wlevel_max[0], lst_tmp_wlevel_max[1])
        if close == True:
            lst_max_wlevel.append(round(max_wlevel, 3))
        else:
            sys.exit("There seem to be two different water level time series at the two outfalls.")
        for key in out.nodes:
            # lst_keys.append(key)
            fld_t_series = pd.Series(out.node_series(key, NodeAttribute.FLOODING_LOSSES)) #cfs
            tstep_seconds = float(pd.Series(fld_t_series.index).diff().mode().dt.seconds)
            # convert from cfs to cf per tstep then cubic meters per timestep
            fld_t_series = fld_t_series * tstep_seconds * cubic_feet_per_cubic_meter
            # sum all flooded volumes
            lst_tot_node_flding.append(fld_t_series.sum())
        lst_rain_return_pd.append(rain_rtrn_pd)
        lst_surge_return_pd.append(surge_rtrn_pd)
        lst_tot_flding.append(sum(lst_tot_node_flding)/volume_units)

#%%
units="{:.0e}".format(int(volume_units))

df = pd.DataFrame({"rain_return_period":lst_rain_return_pd,
                       "surge_return_period":lst_surge_return_pd,
                        "total_flooding_{}m3".format(units):lst_tot_flding,
                        "max_sim_wlevel":lst_max_wlevel,
                        "depth_mm":lst_rain_tot})

#%% write to csv
df.to_csv(f_design_strms, index=False)