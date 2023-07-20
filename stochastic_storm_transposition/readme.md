# Setting up rainyday environment in python:
module load anaconda
conda create -n rainyday
source activate rainyday
conda install -c conda-forge cartopy geopandas xarray dask netCDF4 bottleneck rasterio numba

