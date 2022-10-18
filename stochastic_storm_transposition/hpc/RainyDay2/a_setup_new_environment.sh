module load anaconda

# installing packages (this seems to include all the dependencies)
# source activate rainyday
 
 # environment on laptop - RainyDay
# conda install geopandas cartopy matplotlib netCDF4 rasterio numba xarray

# alternatively, this SHOULD be the code for replicating Daniel Wright's environment but it's not yet working
cd /project/quinnlab/dcl3nd/norfolk/RainyDay2/Source/
module load anaconda
conda env create -f RainyDayPy38_mod.yml #the modified script basically moved a bunch of packages to pip (https://stackoverflow.com/questions/49154899/resolvepackagenotfound-create-env-using-conda-and-yml-file-on-macos)