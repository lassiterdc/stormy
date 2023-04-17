import xarray as xr
import pandas as pd
from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute

from __utils import c7_consolidating_outputs, parse_inp
f_model_perf_summary = c7_consolidating_outputs()

sim_year = 1
#%% inputs from bash
sim_year = int(sys.argv[1]) # a number between 1 and 1000


#%% load and subset model performance dataframe
df_perf = pd.read_csv(f_model_perf_summary)
df_perf_success = df_perf[df_perf.run_completed==True]

for f_inp in df_perf_success.swmm_inp:
    rz, yr, storm_id = parse_inp(f_inp)
    f_swmm_out = f_inp.split('.inp')[0] + '.out'


#%% testing
with Output(f_swmm_out) as out:
    tstep_final = out.times[-1]
    d_node_fld = out.node_attribute(NodeAttribute.FLOODING_LOSSES, tstep_final)
    lst_keys = []
    lst_vals = []
    for key in d_node_fld:
        lst_keys.append(key)
        lst_vals.append(d_node_fld[key])
    df_node_fld = pd.DataFrame(dict(node_id = lst_keys, flood_vol_cf = lst_vals))