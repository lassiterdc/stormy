#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import c7b_consolidating_outputs

f_model_outputs_consolidated, dir_swmm_sst_models = c7b_consolidating_outputs()

#%%
f_modelresults = dir_swmm_sst_models + "_model_outputs_year{}.nc".format("*")

lst_f_modresults = glob(f_modelresults)

#%% work
# lst_f_modresults = lst_f_modresults[0:450]

#%% end work
lst_ds = []
count = -1
for f in tqdm(lst_f_modresults):
    ds_tmp = xr.open_dataset(f)
    lst_ds.append(ds_tmp)
    count += 1
    if count > 438:
        ds = xr.combine_by_coords(lst_ds)
#%% export to single netcdf file
ds = xr.open_mfdataset(lst_f_modresults)
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

ds.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding":{"zlib":True}})