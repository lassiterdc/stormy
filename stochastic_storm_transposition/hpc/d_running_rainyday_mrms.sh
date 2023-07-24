#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 72:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1-11,15-22	        # Array of jobs to loop through 22 years (2001-2011, 15-22)
#SBATCH --mem-per-cpu=200000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

module purge
module load anaconda
source activate rainyday

source __directories.sh

# assign year variable
if [ ${SLURM_ARRAY_TASK_ID} -lt 10 ]
then
	year=200${SLURM_ARRAY_TASK_ID}
else
	year=20${SLURM_ARRAY_TASK_ID}
fi

# generate SST script for the year
sst_in=$(python ${assar_dirs[hpc_d_py]} ${year})

echo "Running sst for year $year using file $sst_in"

# running RainyDay
python ${assar_dirs[hpc_rainyday_py]} ${sst_in}