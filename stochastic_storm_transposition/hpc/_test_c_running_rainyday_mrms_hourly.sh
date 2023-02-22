#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p dev				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 1:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1-22	        # Array of jobs to loop through 22 years (2001-2022)
#SBATCH --mem-per-cpu=100000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# to use dev:
# drop out mem-per-cpu line
# change partition from standard to dev
# change time to 1 hour
# can use the command qlimits to see table of computational limits
# apparently the number of core hours are calculated based on the 
# sacct 
# a node is like a computer
# use -c instead of ntasks if I want to run multi-processing python scripts 
# ntasks is used for MPI

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

echo "Year $year"

sst_in=$(python ${assar_dirs[hpc_c_py]} ${year})

echo "Running file $sst_in"

module purge
module load anaconda
source activate rainyday

#%% running on Rivanna
# using a subset of mrms data
python ${assar_dirs[hpc_b_py]} ${sst_in}