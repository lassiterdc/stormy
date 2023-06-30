#%% import libraries and load directories
import xarray as xr
from datetime import datetime
from glob import glob
from tqdm import tqdm
import shutil
from __utils import c8b_bootstrapping

f_bootstrapped_consolidated, f_bootstrapped_consolidated_hrly_raw, dir_swmm_sst_models_hrly, f_bootstrapped_quant_estimates, sst_recurrence_intervals = c8b_bootstrapping()


f_out_bs_results = f_bootstrapped_quant_estimates + "return_pds_btstrp_{}.nc".format("*")
lst_f_bsresults = glob(f_out_bs_results)

f_out_bs_results_raw = f_bootstrapped_quant_estimates + "raw_btstrp_{}.nc".format("*")
lst_f_bsresults_raw = glob(f_out_bs_results_raw)

#%% export to single netcdf file
ds = xr.open_mfdataset(lst_f_bsresults, engine="h5netcdf", concat_dim = "bootstrap_sample", combine="nested")
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

ds_loaded = ds.load() # this seems to speed up the writing of the netcdf file

ds_loaded.to_netcdf(f_bootstrapped_consolidated, encoding= {"node_flooding_cubic_meters":{"zlib":True}})

#%% do the same for the raw bootstrapped samples
ds = xr.open_mfdataset(lst_f_bsresults_raw, engine="h5netcdf", concat_dim = "bootstrap_sample", combine="nested")
attrs = dict(date_created = str(datetime.now())
             )
ds.attrs = attrs

# WORK
fl_out_zar = f_bootstrapped_consolidated_hrly_raw+".zarr"
# verify chunking
# ds = ds.chunk(chunks={"longitude":chnk_lon, "latitude":chnk_lat, "time":1})
ds.to_zarr(fl_out_zar, mode="w")
# print("Created zarr: {}".format(time.time() - bm_time))

# Load zarr and export to netcdf file
# bm_time = time.time()
ds_from_zarr = xr.open_zarr(store=fl_out_zar, chunks={'resample_id':"10000MB"})
ds_from_zarr.to_netcdf(f_bootstrapped_consolidated_hrly_raw, encoding= {"rainrate":{"zlib":True}})
# print("Created netcdf: {}".format(time.time() - bm_time))

# delete zarr file
# bm_time = time.time()
shutil.rmtree(fl_out_zar)
# END WORK

# ds_loaded = ds.load()

# ds_loaded.to_netcdf(f_bootstrapped_consolidated_hrly_raw, encoding= {"node_flooding_cubic_meters":{"zlib":True}})