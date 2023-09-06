#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 6:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1       # Array of jobs to loop through 22 years (2001-2011, 15-22)
#SBATCH --mem-per-cpu=275000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# ijob -c 1 -A quinnlab_paid -p standard --time=0-06:00:00 --mem-per-cpu=64000

module purge
module load anaconda
DIR=~/.conda/envs/rainyday
source activate rainyday
export PATH=$DIR/bin:$PATH
export LD_LIBRARY_PATH=$DIR/lib:$PATH
export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH

source __directories.sh

# run script for combining storm catalogs
echo "Combining storm catalogs...."
python ${assar_dirs[hpc_c1_py]} ${assar_dirs[hpc_mrms]}


echo "Finished creating storm catalog for: ${assar_dirs[hpc_mrms]}"