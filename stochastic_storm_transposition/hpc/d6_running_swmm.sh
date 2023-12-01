#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name # can use dev for testing
#SBATCH -t 24:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1-1000	# Array of jobs, 1 for each of 1000 years (1-1000)
# SBATCH --mem-per-cpu=36000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# ijob -c 1 -A quinnlab_paid -p standard --time=0-09:00:00 --mem-per-cpu=32000

source __directories.sh
module purge
module load gcc openmpi eccodes anaconda # the stuff other than anaconda was to ensure eccodes loaded correctly
DIR=~/.conda/envs/running_swmm
source activate running_swmm
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH

# echo "Running SWMM for year ${SLURM_ARRAY_TASK_ID}..."

# running swmm
python ${assar_dirs[hpc_d6_py]} ${SLURM_ARRAY_TASK_ID}