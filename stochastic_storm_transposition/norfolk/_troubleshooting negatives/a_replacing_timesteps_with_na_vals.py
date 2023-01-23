"""
10/20/2003 contains the beginning of the largest storm that appears in norfolk's storm catalog. 
This script replaces a few timesteps with nan values to see what happens.
THe storm starts at 08:00.
"""

#%% import libraries and load directories
import xarray as xr
import numpy as np
fpath = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/norfolk/_troubleshooting negatives/StageIV_FilledCorr.20031030.03degree_original.nc"
fout = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/norfolk/_troubleshooting negatives/StageIV_FilledCorr.20031030.03degree.nc"

#%% load dataset
ds = xr.load_dataset(fpath)

#%% make all timesteps from 18:00 to 20:00 nan
ts = ['2003-10-30T18:00:00.000000000','2003-10-30T19:00:00.000000000', '2003-10-30T20:00:00.000000000']

ds.rainrate.loc[{"time":ts}] = np.nan

#%% write netcdf
ds.to_netcdf(fout)

#%% test
ds_tst = xr.load_dataset(fout)
ds_tst.rainrate.loc[{"time":ts}]