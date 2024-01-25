# Setting up rainyday environment in python:
module load anaconda
conda create -n rainyday
source activate rainyday
conda install -c conda-forge cartopy geopandas xarray dask netCDF4 bottleneck rasterio numba

# setting up wlevel_sim environment in python
# https://www.rc.virginia.edu/userinfo/rivanna/software/anaconda/
module load anaconda
conda create -n cop_sim
source activate cop_sim
pip install pyvinecopulib
conda install -c conda-forge copulas xarray pandas xarray dask netCDF4 matplotlib scikit-learn seaborn 
