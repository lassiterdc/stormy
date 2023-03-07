#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 168:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=20011-20016,20021-20026,20031-20036,20041-20046,20051-20056,20061-20066,20071-20076,20081-20086,20091-20096,20101-20106,20111-20116,20151-20156,20161-20166,20171-20176,20181-20186,20191-20196,20201-20206,20211-20216,20221-20226
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