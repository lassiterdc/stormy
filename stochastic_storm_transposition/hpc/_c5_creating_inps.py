#%% importing libraries and loading directories
import numpy as np
from pathlib import Path
import shutil
import xarray as xr
import pandas as pd
from glob import glob
import pandas as pd
from string import Template
import sys
from tqdm import tqdm
from __utils import c5_creating_inps

nyears, nperyear, nrealizations, dir_swmm_sst_models_hrly, f_inp_base, f_out_realizations, seed_mrms_hourly, dir_time_series, f_key_subnames_gridind, lst_template_keys, work_f_water_level_path, f_swmm_scenarios_catalog = c5_creating_inps()
#%% define functions
def get_rainfiles(rz, yr, storm_id, df_key):
    # format "realization{}_year{}_storm-id{}_grid-ind{}_.dat"
    lst_f_rain_dats = glob(dir_time_series  + "realization{}_year{}_storm-id{}_*.dat".format(rz, yr, storm_id))
    lst_rain_dats = []
    lst_grid_ind = []
    for f in lst_f_rain_dats:
        rain_dat = f.split("\\")[-1]
        grid_ind = rain_dat.split("grid-ind")[-1].split(".")[0]
        lst_rain_dats.append(rain_dat)
        lst_grid_ind.append(grid_ind)
    df = pd.DataFrame(dict(rain_dats = lst_rain_dats,
                           grid_ind = lst_grid_ind,
                           rain_dats_fullpath = lst_f_rain_dats))
    return df

def generate_water_level_series():
        return fpath_waterlevel

#%% load data
ds_rlztns = xr.open_dataset(f_out_realizations)
num_storms = len(ds_rlztns.realization_id.values) * len(ds_rlztns.year.values) * len(ds_rlztns.storm_id.values)
df_strms = pd.DataFrame(dict(storm_number = np.arange(1, num_storms+1)))

df_key = pd.read_csv(f_key_subnames_gridind)

#%% clear folder of SWMM scenarios
try:
    shutil.rmtree(dir_swmm_sst_models_hrly)
except:
    pass

#%% create the folders and duplicate the model
with open(f_inp_base, 'r') as T:
    # loading template
    template = Template(T.read())
    count = -1

    for rz in tqdm(ds_rlztns.realization_id.values):
        dir_r = dir_swmm_sst_models_hrly + "weather_realization{}/".format(rz)
        for yr in ds_rlztns.year.values:
            dir_y = dir_r + "year{}/".format(yr)
            for storm_id in ds_rlztns.storm_id.values:
                count += 1
                df_strms.loc[count, "realization"] = rz
                df_strms.loc[count, "year"] = yr
                df_strms.loc[count, "storm_num"] = storm_id

                # append new row to pandas dataframe
                # dic_scen = dict(realization = rz, year = yr, storm = storm_id)
                
                # create copy of input file
                dir_strm = dir_y + "rz{}_yr{}_strm{}/".format(rz, yr, storm_id)
                f_inp_scen = dir_strm + "r{}_y{}_s{}.inp".format(rz, yr, storm_id)
                df_strms.loc[count, "swmm_inp"] = f_inp_scen
                ## shutil.copy(f_inp_base, f_inp_scen)
                df_rain_paths = get_rainfiles(rz, yr, storm_id, df_key)
                # fill in template stuff
                d_fields = {}
                for key in lst_template_keys:
                    fpath = None
                    if key == "water_level":
                        fpath = work_f_water_level_path

                    key_grid = key.split("_")[-1]
                    for grid_ind in df_rain_paths.grid_ind:
                        if grid_ind == key_grid:
                            fpath = df_rain_paths.rain_dats_fullpath[df_rain_paths.grid_ind == grid_ind].values[0]

                    if fpath is not None: # meaning, if a filepath associated with the grid index was found
                        d_fields[key] = fpath
                    else:
                        sys.exit("Filepath to fill out SWMM tepmlate not found.")

                    # format the filepath
                    fpath = fpath.replace("/", "\\")
                    df_strms.loc[count, key] = fpath

                new_in = template.safe_substitute(d_fields)
                new_in = template.substitute(d_fields)
                # new_file = f_inp_scen
                new_file_path = Path(f_inp_scen)
                # create _inputs folder if it doesn't already exist
                new_file_path.parent.mkdir(parents=True, exist_ok=True)
                # new_file_path.touch()
                with open (f_inp_scen, "w+") as f1:
                    f1.write(new_in)
#%% export swmm catalog to csv file
dtypes = dict(realization = int, year = int, storm_num = int)
df_strms = df_strms.astype(dtypes)

df_strms.to_csv(f_swmm_scenarios_catalog, index = False)