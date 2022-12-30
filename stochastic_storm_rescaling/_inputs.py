#%% import libraries
from pathlib import Path
import os

#%% User inputs
# script a
in_a_begin_year = 2001

#%% Directories
fldr_nrflk = str(Path(os.getcwd()).parents[1]) + "/"
fldr_ssr = fldr_nrflk + "stormy/stochastic_storm_rescaling/"

fld_out = fldr_ssr + "outputs"
fldr_swmm = fldr_nrflk + "stormy/swmm/hague/swmm_model/"
fldr_swmm_features = fldr_swmm + "exported_layers/"
fldr_nrflk = str(Path(os.getcwd()).parents[1])
fldr_mrms_processing = fldr_nrflk + "/highres-radar-rainfall-processing/data/"

f_swmm_model = fldr_swmm + "hague_V1_using_norfolk_data.inp"
f_shp_swmm_subs = fldr_swmm_features + "subcatchments.shp"

# unique to script a
fld_out_a = fld_out + "/a_NOAA_water_levels/"
f_out_a_meta = fld_out_a + 'sewells_pt_water_level_metadatap.json'
f_out_a_all = fld_out_a + "a_water-lev_tide_surge.csv"
f_out_a_shp = fld_out_a + "sewells_pt.shp"

# unique to script b
fld_out_b = fld_out + "/b_precip_time_series_at_gages/"
f_in_b_nc = fldr_mrms_processing+"mrms_nc_preciprate_fullres_atgages.nc"
f_out_b_csv_rainfall = fld_out_b + "mrms_rainfall.csv"
f_out_b_csv_subs_w_mrms_grid = fld_out_b + "sub_ids_and_mrms_rain_col.csv"

def def_work():
    return f_out_a_all

def def_inputs_for_a():
    return in_a_begin_year, f_out_a_meta, f_out_a_all, f_out_a_shp

def def_inputs_for_b():
    return f_in_b_nc, f_shp_swmm_subs, f_out_b_csv_rainfall, f_out_b_csv_subs_w_mrms_grid