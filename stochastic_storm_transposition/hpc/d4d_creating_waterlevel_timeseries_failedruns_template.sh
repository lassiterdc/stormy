#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial) [standard]
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 12:00:00				# Run time per serial job (hh:mm:ss) [24:00:00]
#SBATCH --array=${LIST_FAILED_TASKS}  # these are the jobs with errors; normally run 1-1000 unless re-running failed scripts        # Array of jobs, 1 for each of 1000 years  
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   

# ijob -c 1 -A quinnlab_paid -p standard --time=0-06:00:00 --mem-per-cpu=64000

source __directories.sh
module purge
module load gcc openmpi eccodes anaconda # the stuff other than anaconda was to ensure eccodes loaded correctly
DIR=~/.conda/envs/rainyday
source activate rainyday
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH

# echo "Creating rainfall time series for year ${SLURM_ARRAY_TASK_ID}..."

# running script
python ${assar_dirs[hpc_d4b_py]} ${SLURM_ARRAY_TASK_ID}