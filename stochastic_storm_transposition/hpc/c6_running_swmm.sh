#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name # can use dev for testing
#SBATCH -t 24:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=329,141,404,832,313,314,326,221,671,178,880,285,249,919,98,678,561,698,499,190,392,861,422,291,797,435,348,539,531,953,62,960,272,274,222,99,767,323,49,193,976,764,110,256,570,760,891,532,373,938,583,845,784,997,70,903,202,878,924,768,955,619,108,618,896,118,112,945,727,94,719,838,479,659,640,949,662,782,802,34,651,19,534,33,730,150,748,816,288,242,431,657,253,887,105,483,823,576,200,208,75,579,156,239,607,7,513,881,113,175,948,844,886,988,820,496,104,744,737,998,688,384,235,734,183,122,876,535,728,866,280,44,396,931,155,694,290,828,480,892,703,91,922,778,452,442,811,859,934,124,454,386,80,585,48,944,363,637,990,268,836,436,240,342,366,807,293,913,147,210,646,234,420,214,81,103,985,520,417,709,64,306,893,252,765,594,854,212,257,275,402,219,21,428,788,515,1,806,933,473,750,830,389,518,928,83,494,497,231,865,815,46,937,691,835,575,464,352,209,303,936,630,18	# Array of jobs, 1 for each of 1000 years (1-1000)
# SBATCH --mem-per-cpu=36000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

module purge
module load anaconda
source activate running_swmm

source __directories.sh

# echo "Running SWMM for year ${SLURM_ARRAY_TASK_ID}..."

# running swmm
python ${assar_dirs[hpc_c6_py]} ${SLURM_ARRAY_TASK_ID}