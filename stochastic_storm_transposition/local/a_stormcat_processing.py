"""
The primary function of this script is to load storm catalogs and replace any instances of -9999 with 'nan'.
"""

#%% import libraries and load directories
import xarray as xr
from tqdm import tqdm
from _inputs import a_stormat_processing
from glob import glob
from netCDF4 import Dataset
fldr_rainyday_working_dir = a_stormat_processing()

lst_f_strmcats = glob(fldr_rainyday_working_dir + "*.nc")


#%%

for f_strmcat in lst_f_strmcats:
    with xr.open_dataset(f_strmcat) as ds:
        continue
#%% load with netcdf4 library
f_strmcat_stageVI = lst_f_strmcats[3]
nds = Dataset(f_strmcat_stageVI)
nds.dimensions["time"]
nds.variables['precrate']

#%%
import numpy as np
f_strmcat_mrms = lst_f_strmcats[1]
infile = Dataset(f_strmcat_mrms)
infile.dimensions["time"]
infile.variables['precrate']

outtime=np.array(infile.variables['time'][:],dtype='datetime64[m]')
#%%
# catmax_subdur[i]=dur_max
# temptime[i,:]=cattime[i,dur_j:dur_j+int(duration*60./rainprop.timeres)]
# cattime_subdur=temptime
# sind=np.argsort(catmax_subdur)
# cattime=cattime_subdur[sind,:]

nstorms = nds.dimensions["nstorms"].size
cattime=np.empty((nstorms,int(catduration*60/rainprop.timeres)),dtype='datetime64[m]')
cattime[:]=np.datetime64(datetime(1900,1,1,0,0,0))
str(cattime[i,-1])