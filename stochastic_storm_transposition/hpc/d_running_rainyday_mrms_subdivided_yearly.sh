#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 84:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1-11,15-22	        # Array of jobs to loop through 22 years (1-11,15-22)
#SBATCH --mem-per-cpu=64000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# ijob -c 1 -A quinnlab_paid -p standard --time=0-06:00:00 --mem-per-cpu=64000

# testing with stageiv and mrms
# sst_in="/scratch/dcl3nd/stormy/stochastic_storm_transposition/norfolk/_sst_hpc_testing_stageiv/stageiv_testing_rivanna.sst"
# sst_in="/scratch/dcl3nd/stormy/stochastic_storm_transposition/norfolk/_sst_hpc_testing_mrms_hourly/mrms_hourly_testing_rivanna.sst"

source __directories.sh
module purge
module load anaconda
DIR=~/.conda/envs/rainyday
source activate rainyday
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH

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