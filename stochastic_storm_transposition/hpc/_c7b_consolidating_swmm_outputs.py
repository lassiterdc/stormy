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

#%% export to single netcdf file
ds = xr.open_mfdataset(lst_f_modresults, chunks=dict(node_id = 1133), engine="h5netcdf")
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

#%% testing
# import numpy as np
# lst_f_modresults = glob(f_modelresults)
# # lst_f_modresults = lst_f_modresults[0:200]

# count = -1
# for i in np.arange(len(lst_f_modresults)):
#     count += 1
#     if (count > 130) and (count>1):
#         print(count)
#         ds = xr.open_mfdataset(lst_f_modresults[0:i], chunks=dict(node_id = 1133))
#         attrs = dict(date_created = str(datetime.now())
#                     )
#         ds.attrs = attrs
#         ds_loaded = ds.load()


# min_ind_prob = 0
# ind_prob = 129 # 129 causes the message



# ds = xr.open_mfdataset([lst_f_modresults[min_ind_prob], lst_f_modresults[ind_prob]], chunks=dict(node_id = 1133))
# attrs = dict(date_created = str(datetime.now())
#             )
# ds.attrs = attrs
# ds_loaded = ds.load()

# ds_tmp = xr.open_dataset(lst_f_modresults[ind_prob])
# ds_tmp_loaded = ds_tmp.load()
#%% end testing

ds.to_netcdf(f_model_outputs_consolidated, encoding= {"node_flooding":{"zlib":True}})