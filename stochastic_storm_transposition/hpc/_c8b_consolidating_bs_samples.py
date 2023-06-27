#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm

from __utils import c8b_bootstrapping

f_bootstrapped_consolidated, dir_swmm_sst_models_hrly, f_bootstrapped_quant_estimates, sst_recurrence_intervals = c8b_bootstrapping()


f_out_bs_results = f_bootstrapped_quant_estimates + "return_pds_btstrp_{}.nc".format("*")
lst_f_modresults = glob(f_out_bs_results)

#%% export to single netcdf file
ds = xr.open_mfdataset(lst_f_modresults, engine="h5netcdf")
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

ds_loaded = ds.load() # this seems to speed up the writing of the netcdf file

ds_loaded.to_netcdf(f_bootstrapped_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}})