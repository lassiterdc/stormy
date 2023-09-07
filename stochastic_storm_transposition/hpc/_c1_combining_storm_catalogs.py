#%% import libraries and load directories
"""
This script is in progress and might not actually be necessary
"""
from  netCDF4 import Dataset
import numpy as np
from datetime import datetime
import sys
from glob import glob
sys.path.append(str(sys.argv[1]))
import RainyDay_utilities_Py3.RainyDay_functions as RainyDay
# from __utils import *

# load files and directories
# dir_sst_nrflk_hrly, parameterfile, f_out, fs = c1_combine_hrly_cats()


dir_sst = str(sys.argv[2])
f_sst_combined_cat = dir_sst + "strmcat_combined.nc"
parameterfile = dir_sst + "combined.sst"
fs = glob(dir_sst + "/mrms*/StormCatalog/*.nc")

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
lst_cattime = []
lst_timeresolution = []

for f in fs:
    catrain,stormtime,latrange,lonrange,catx,caty,catmax,catmask,domainmask,cattime,timeres = RainyDay.readcatalog(f)
    lst_outrain.append(catrain)
    lst_outtime.append(stormtime)
    lst_outlatitude.append(latrange)
    lst_outlongitude.append(lonrange)
    lst_outlocx.append(catx)
    lst_outlocy.append(caty)
    lst_outmax.append(catmax)
    lst_outmask.append(catmask)
    lst_domainmask.append(domainmask)
    lst_cattime.append(cattime)
    lst_timeresolution.append(timeres)

#%% combine the storm catalog attributes
# combine the outrain arrays
outrain_comb = np.concatenate(lst_outrain, axis = 0)

# combine the outtime arrays
outtime_comb = np.concatenate(lst_outtime, axis=0)

# combine outlocx,outlocy, outmax
outlocx_comb = np.concatenate(lst_outlocx, axis=0)
outlocf = y_comb = np.concatenate(lst_outlocy, axis=0)
outmax_comb = np.concatenate(lst_outmax, axis=0)

#%% write combined storm catalog to a new file
# assign variables
print("creating combining storm catalogs....")
scenarioname = "combined_storm_catalog"
# catrain = outrain_comb
# catmax = outmax_comb
# catx = outlocx_comb
# caty = outlocy_comb
# cattime = outtime_comb
# latrange = outlatitude
# lonrange = outlongitude
# catalogname = f_sst_combined_cat
# nstorms = len(outmax_comb)
# gridmask = outmask
# dmask = domainmask
# timeresolution = timeresolution
# RainyDay.writecatalog(scenarioname,catrain,catmax,catx,caty,cattime,latrange,lonrange,catalogname,nstorms,gridmask,parameterfile,dmask,timeresolution)

# storm_name = 
# catduration=np.nan

# RainyDay.writecatalog(scenarioname,catrain,\
#                         catmax,\
#                             catx,caty,\
#                                 cattime,latrange,lonrange,\
#                                     storm_name,catmask,parameterfile,domainmask,\
#                                         nstorms,catduration,storm_num=int(i),timeresolution=timeres)