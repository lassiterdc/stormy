#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 6:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1       # Array of jobs to loop through 22 years (2001-2011, 15-22)
#SBATCH --mem-per-cpu=36000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

module purge
module load anaconda
source activate rainyday

source __directories.sh

cat ${assar_dirs[hpc_mrms_hourly]}*_20*.nc > ${assar_dirs[hpc_c1_cmbnd_cat]}

# generate SST script for the year
sst_in=$(python ${assar_dirs[hpc_c_py]} ${year})

echo "Running sst for year $year using file $sst_in"

# running RainyDay
python ${assar_dirs[hpc_rainyday_py]} ${sst_in}