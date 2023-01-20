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

#%% load storm catalogs and replace -9999 with NA
lst_f_strmcats = glob(fldr_rainyday_working_dir + "*.nc")

for f_strmcat in lst_f_strmcats:
    with xr.open_dataset(f_strmcat) as ds:

#%% load with netcdf4 library
nds = Dataset(f_strmcat)
nds.dimensions["time"]
nds.variables['precrate']
