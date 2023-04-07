#%% import libraries and load directories
from  netCDF4 import Dataset
import numpy as np
from datetime import datetime
import xarray as xr
from glob import glob
import sys
import pandas as pd
from __utils import c3_reformat_hrly_cats

f_in, f_out, dir_sst_realizations, f_out_realizations = c3_reformat_hrly_cats()
#%% functions from RainyDay_functions.py
def readcatalog(rfile):
    infile=Dataset(rfile,'r')

    outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
    outlatitude=np.array(infile.variables['latitude'][:])[::-1]
    outmask=np.array(infile.variables['gridmask'][:])[::-1,:]
    domainmask=np.array(infile.variables['domainmask'][:])[::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])
    timeresolution=int(infile.timeresolution)

    

    d_precrate_attributes = infile.variables['precrate'].__dict__
    lst_precrate_dims = list(infile.variables['precrate'].dimensions)
    lst_precrate_dimshape = list(infile.variables['precrate'].shape)

    d_time_attributes = infile.variables['time'].__dict__
    lst_time_dims = list(infile.variables['time'].dimensions)
    lst_time_dimshape = list(infile.variables['time'].shape)

    d_nc_attributes = infile.__dict__

    infile.close()

    return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution, d_precrate_attributes, lst_precrate_dims, lst_precrate_dimshape, d_time_attributes, lst_time_dims, lst_time_dimshape, d_nc_attributes
#%% read realizations adapted from readcatalog
def readrealization(rfile):
    infile=Dataset(rfile,'r')

    outrain=np.array(infile.variables['precrate'][:])[:,:,:,:]
    outlatitude=np.array(infile.variables['latitude'][:])
    outlongitude=np.array(infile.variables['longitude'][:])
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')

    
    d_precrate_attributes = infile.variables['precrate'].__dict__
    lst_precrate_dims = list(infile.variables['precrate'].dimensions)
    lst_precrate_dimshape = list(infile.variables['precrate'].shape)

    d_time_attributes = infile.variables['time'].__dict__
    lst_time_dims = list(infile.variables['time'].dimensions)
    lst_time_dimshape = list(infile.variables['time'].shape)

    d_nc_attributes = infile.__dict__

    infile.close()
 
    return outrain,outtime,outlatitude,outlongitude,d_precrate_attributes, lst_precrate_dims, lst_precrate_dimshape, d_nc_attributes, d_time_attributes, lst_time_dims, lst_time_dimshape

#%% combining, reformatting, and exporting the combined storm catalog
fs_rlz = glob(dir_sst_realizations + "*SST*.nc")
fs_rlz.sort()
# WORK
print("##########################")
print("fs_rlz")
print(fs_rlz)
print("##########################")
print("len(fs_rlz)")
print(len(fs_rlz))
print("##########################")
print("fs_rlz[0]")
print(fs_rlz[0])
print("##########################")
print("fs_rlz[-1]")
print(fs_rlz[-1])
# END WORK

lst_ds = []
count = 0
for f in fs_rlz:
    outrain,outtime,outlatitude,outlongitude,d_precrate_attributes, lst_precrate_dims, lst_precrate_dimshape, d_nc_attributes, d_time_attributes, lst_time_dims, lst_time_dimshape = readrealization(f)

    d_nc_attributes["note"] = "This dataset is a simple reformatting of the storm catalog output by RainyDay to allow it to be loaded into xarray."
    d_nc_attributes["reformatted_realizations_directory"] = dir_sst_realizations
    d_nc_attributes["date_reformatted"] = str(datetime.now())

    try:
        ds = xr.Dataset(data_vars=dict(
                rainrate = (['year', 'storm_id', 'timestep_index', 'latitude', 'longitude'], outrain)),
                coords=dict(
                    year = np.arange(1, lst_precrate_dimshape[0]+1),
                    storm_id = np.arange(1, lst_precrate_dimshape[1]+1),
                    timestep_index = np.arange(1, lst_precrate_dimshape[2]+1),
                    time = (["year", "storm_id", "datetime"], outtime), 
                    latitude = outlatitude, 
                    longitude = outlongitude),
                attrs=d_nc_attributes
                )
    except:
        # WORK
        print("##########################")
        print("count")
        print(count)
        print("##########################")
        print("f")
        print(f)
        print("##########################")
        print("d_time_attributes")
        print(d_time_attributes)
        print("##########################")
        print("lst_time_dims")
        print(lst_time_dims)
        print("##########################")
        print("lst_time_dimshape")
        print(lst_time_dimshape)
        print("##########################")
        print("outrain")
        print(outrain)
        print("##########################")
        print("outrain.shape")
        print(outrain.shape)
        # END WORK
        sys.exit("script failed.")

    ds.rainrate.attrs = d_precrate_attributes
    lst_ds.append(ds)

ds_all_realizations = xr.concat(lst_ds, pd.Index(np.arange(1, len(fs_rlz)+1),name = "realization_id"))

ds_all_realizations.to_netcdf(f_out_realizations, encoding= {"rainrate":{"zlib":True}})
#%% load storm catalog
outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution, d_precrate_attributes, lst_precrate_dims, lst_precrate_dimshape, d_time_attributes, lst_time_dims, lst_time_dimshape, d_nc_attributes = readcatalog(f_in)

#%% creating xarray dataset of combined storm catalog and exporting to netcdf
d_nc_attributes["note"] = "This dataset is a simple reformatting of the storm catalog output by RainyDay to allow it to be loaded into xarray."
d_nc_attributes["reformatted_catalog_from_RainyDay"] = f_in
d_nc_attributes["date_reformatted"] = str(datetime.now())
d_nc_attributes["time_resolution_min"] = timeresolution

ds = xr.Dataset(data_vars=dict(
        rainrate = (["storm_id", "timestep_index", "latitude", "longitude"], outrain)),
        coords=dict(
            storm_id = np.arange(1, len(outmax)+1), 
            timestep_index = np.arange(1, outrain.shape[1]+1),
            time = (["storm_id", "datetime"], outtime), 
            latitude = outlatitude, 
            longitude = outlongitude),
        attrs=d_nc_attributes)

ds.rainrate.attrs = d_precrate_attributes

ds.to_netcdf(f_out, encoding= {"rainrate":{"zlib":True}})