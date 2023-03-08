#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 168:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=20011,20012,20013,20014,20015,20016,20021,20022,20023,20024,20025,20026,20031,20032,20033,20034,20035,20036,20041,20042,20043,20044,20045,20046,20051,20052,20053,20054,20055,20056,20061,20062,20063,20064,20065,20066,20071,20072,20073,20074,20075,20076,20081,20082,20083,20084,20085,20086,20091,20092,20093,20094,20095,20096,20101,20102,20103,20104,20105,20106,20111,20112,20113,20114,20115,20116,20151,20152,20153,20154,20155,20156,20161,20162,20163,20164,20165,20166,20171,20172,20173,20174,20175,20176,20181,20182,20183,20184,20185,20186,20191,20192,20193,20194,20195,20196,20201,20202,20203,20204,20205,20206,20211,20212,20213,20214,20215,20216,20221,20222,20223,20224,20225,20226,
#SBATCH --mem-per-cpu=300000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# array job numbers are in \stormy\stochastic_storm_transposition\local\da_job_array_numbers.xlsx

module purge
module load anaconda
source activate rainyday

source __directories.sh

# year=${SLURM_ARRAY_TASK_ID} | cut -c1-4
year=${SLURM_ARRAY_TASK_ID:0:4}
template_num=${SLURM_ARRAY_TASK_ID:4:4}

# generate SST script for the year
sst_in=$(python ${assar_dirs[hpc_d_py]} ${SLURM_ARRAY_TASK_ID})

echo "Running sst for year $year using file $sst_in"

# running RainyDay
python ${assar_dirs[hpc_rainyday_py]} ${sst_in}