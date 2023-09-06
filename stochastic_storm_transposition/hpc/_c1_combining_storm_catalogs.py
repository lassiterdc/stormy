#%% import libraries and load directories
from  netCDF4 import Dataset
import numpy as np
from datetime import datetime
import sys
from glob import glob
# from __utils import *

# load files and directories
# dir_sst_nrflk_hrly, parameterfile, f_out, fs = c1_combine_hrly_cats()


dir_sst = str(sys.argv[1])
f_sst_combined_cat = dir_sst + "strmcat_combined.nc"
fs = glob(dir_sst + "*_20*.nc")
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

def writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,gridmask,parameterfile,dmask,timeresolution=False):
    dataset=Dataset(catalogname, 'w', format='NETCDF4')
    
    # create dimensions
    lat_dim=dataset.createDimension('latitude',len(latrange))
    lon_dim=dataset.createDimension('longitude',len(lonrange))
    time_dim=dataset.createDimension('time',cattime.shape[1])
    nstorms=dataset.createDimension('nstorms',nstorms)

    # create variables
    times=dataset.createVariable('time',np.float64, ('nstorms','time',))
    latitudes=dataset.createVariable('latitude',np.float64, ('latitude',))
    longitudes=dataset.createVariable('longitude',np.float64, ('longitude',))
    rainrate=dataset.createVariable('precrate',np.float32,('nstorms','time','latitude','longitude',),zlib=True,complevel=4,least_significant_digit=1) 
    basinrainfall=dataset.createVariable('basinrainfall',np.float32,('nstorms')) 
    xlocation=dataset.createVariable('xlocation',np.int16,('nstorms')) 
    ylocation=dataset.createVariable('ylocation',np.int16,('nstorms')) 
    gmask=dataset.createVariable('gridmask',np.float32,('latitude','longitude',)) 
    domainmask=dataset.createVariable('domainmask',np.float32,('latitude','longitude',)) 
    
    times.long_name='time'
    latitudes.long_name='latitude'
    longitudes.long_name='longitude'
    rainrate.long_name='precipitation rate'
    basinrainfall.long_name='storm total basin averaged precipitation'
    xlocation.long_name='x index of storm'
    ylocation.long_name='y index of storm'
    gmask.long_name='mask for Aw (control volume)'
    domainmask.long_name='mask for transposition domain'
    
    # Variable Attributes (time since 1970-01-01 00:00:00.0 in numpys)
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    rainrate.units = 'mm/hr'
    basinrainfall.units='mm'
    times.units = 'minutes since 1970-01-01 00:00.0'
    times.calendar = 'gregorian'
    gmask.units="dimensionless"
    domainmask.units="dimensionless"
    xlocation.units='dimensionless'
    ylocation.units='dimensionless'
    
    # Global Attributes
    dataset.Conventions ='CF1.8'
    with open(parameterfile, "r") as myfile:
        params=myfile.read()
    myfile.close
    dataset.description=params
    if timeresolution!=False:
        dataset.timeresolution=timeresolution

    dataset.history = 'Created ' + str(datetime.now())
    dataset.source = 'RainyDay Storm Catalog for scenario '+scenarioname+'. See description for SST file contents.'
    # dataset.missing='-9999.' # original
    dataset.missing='missing values encoded with a zero (DCL MOD)' # DCL mod
    
    # fill the netcdf file
    latitudes[:]=latrange[::-1]
    longitudes[:]=lonrange
    # catrain[np.isnan(catrain)]=-9999. # original 
    catrain[np.isnan(catrain)]=0 # DCL mod
    rainrate[:]=catrain[:,:,::-1,:] 
    # catmax[np.isnan(catmax)]=-9999. # original
    catmax[np.isnan(catmax)]=0# DCL mod
    basinrainfall[:]=catmax 
    times[:]=cattime
    xlocation[:]=catx
    ylocation[:]=caty
    gmask[:]=gridmask[::-1,:] 
    domainmask[:]=dmask[::-1,:] 
    dataset.close()

#%% load the individual storm catalogs
lst_outrain = []
lst_outtime = []
lst_outlatitude = []
lst_outlongitude = []
lst_outlocx = []
lst_outlocy = []
lst_outmax = []
lst_outmask = []
lst_domainmask = []
lst_timeresolution = []

for f in fs:
    outrain,outtime,outlatitude,outlongitude,outlocx,outlocy,outmax,outmask,domainmask,timeresolution = readcatalog(f)
    lst_outrain.append(outrain)
    lst_outtime.append(outtime)
    lst_outlatitude.append(outlatitude)
    lst_outlongitude.append(outlongitude)
    lst_outlocx.append(outlocx)
    lst_outlocy.append(outlocy)
    lst_outmax.append(outmax)
    lst_outmask.append(outmask)
    lst_domainmask.append(domainmask)
    lst_timeresolution.append(timeresolution)

#%% combine the storm catalog attributes
# combine the outrain arrays
outrain_comb = np.concatenate(lst_outrain, axis = 0)

# combine the outtime arrays
outtime_comb = np.concatenate(lst_outtime, axis=0)

# combine outlocx,outlocy, outmax
outlocx_comb = np.concatenate(lst_outlocx, axis=0)
outlocy_comb = np.concatenate(lst_outlocy, axis=0)
outmax_comb = np.concatenate(lst_outmax, axis=0)

#%% write combined storm catalog to a new file
# assign variables
print("creating combining storm catalogs....")
scenarioname = "mrms hourly combined"
catrain = outrain_comb
catmax = outmax_comb
catx = outlocx_comb
caty = outlocy_comb
cattime = outtime_comb
latrange = outlatitude
lonrange = outlongitude
catalogname = f_sst_combined_cat
nstorms = len(outmax_comb)
gridmask = outmask
dmask = domainmask
timeresolution = timeresolution
writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,gridmask,parameterfile,dmask,timeresolution)