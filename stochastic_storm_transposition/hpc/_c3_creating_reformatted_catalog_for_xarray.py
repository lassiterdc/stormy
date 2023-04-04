#%% import libraries and load directories
from  netCDF4 import Dataset
import numpy as np
from datetime import datetime
import xarray as xr
from __utils import c3_reformat_hrly_cats

f_in, f_out = c3_reformat_hrly_cats()
#%% functions from RainyDay_functions.py
def readcatalog(rfile):
    infile=Dataset(rfile,'r')
    if 'rainrate' in infile.variables.keys():
        oldfile=True
        print("reading an older-style storm catalog")
    else:
        oldfile=False
    if oldfile:
        outrain=np.array(infile.variables['rainrate'][:])
        outlatitude=np.array(infile.variables['latitude'][:])
        outmask=np.array(infile.variables['gridmask'][:])
        domainmask=np.array(infile.variables['domainmask'][:])
    else:
        outrain=np.array(infile.variables['precrate'][:])[:,:,::-1,:]
        outlatitude=np.array(infile.variables['latitude'][:])[::-1]
        outmask=np.array(infile.variables['gridmask'][:])[::-1,:]
        domainmask=np.array(infile.variables['domainmask'][:])[::-1,:]
    outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
    outlongitude=np.array(infile.variables['longitude'][:])
    outlocx=np.array(infile.variables['xlocation'][:])
    outlocy=np.array(infile.variables['ylocation'][:])
    outmax=np.array(infile.variables['basinrainfall'][:])

    try:
        timeresolution=np.int(infile.timeresolution)
        resexists=True
    except:
        resexists=False
    infile.close()
    
    # print(resexists)  # DCL MOD
    if resexists:
        return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution
    else:
        return outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask
    

#%% load storm catalog
outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution = readcatalog(f_in)

#%% creating dataset
ds = xr.Dataset(data_vars=dict(
        rainrate = (["storm_id", "timestep_index", "latitude", "longitude"], outrain)),
        coords=dict(
            storm_id = np.arange(1, len(outmax)+1), 
            timestep_index = np.arange(1, outrain.shape[1]+1),
            time = (["storm_id", "datetime"], outtime), 
            latitude = outlatitude, 
            longitude = outlongitude),
        attrs=dict(description="This dataset is a simple reformatting of the storm catalog output by RainyDay to allow it to be loaded into xarray.",
                   reformatted_catalog_from_RainyDay=f_in,
                   date_processed=str(datetime.now()),
                   time_resolution_min = timeresolution)
        )
#%% exporting dataset
ds.to_netcdf(f_out, encoding= {"rainrate":{"zlib":True}})