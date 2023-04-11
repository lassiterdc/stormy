#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial) [standard]
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 03:00:00				# Run time per serial job (hh:mm:ss) [24:00:00]
#SBATCH --array=1-1000	        # Array of jobs, 1 for each of 1000 years    
# SBATCH --mem-per-cpu=200000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-19,udc-ba27-17,udc-ba26-32c1,udc-ba26-14,udc-ba26-12,udc-ba26-10,udc-ba27-14,udc-ba26-9,udc-ba26-35c1,udc-ba27-16,udc-aw34-16c1,udc-aw34-21c0,udc-ba27-32c1,udc-aw34-7c1,udc-ba27-9,udc-ba27-20,udc-ba26-33c0,udc-aw34-17c1,udc-aw34-4c1,udc-aw34-21c1,udc-ba26-28c1,udc-ba26-29c1,udc-aw34-3c0,udc-aw34-14c1,udc-aw34-15c0,udc-ba25-12c1,udc-aw29-24b,udc-aw29-20b,udc-ba25-13c0,udc-aw29-23a,udc-ba25-13c1,udc-aw34-13c0,udc-aw34-12c1,udc-aw34-10c0

module purge
module load gcc openmpi eccodes anaconda # the stuff other than anaconda was to ensure eccodes loaded correctly
source activate geopandas # created an environment for this script specifically

source __directories.sh

echo "Creating rainfall time series for year ${SLURM_ARRAY_TASK_ID}..."

# running script
python ${assar_dirs[hpc_c4_py]} ${SLURM_ARRAY_TASK_ID}