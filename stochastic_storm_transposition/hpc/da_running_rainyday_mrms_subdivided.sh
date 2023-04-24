#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 168:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=11-16,21-26,31-36,41-46,51-56,61-66,71-76,81-86,91-96,101-106,111-116,151-156,161-166,171-176,181-186,191-196,201-206,211-216,221-226
#SBATCH --mem-per-cpu=300000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# array job numbers are in \stormy\stochastic_storm_transposition\local\da_job_array_numbers.xlsx

module purge
module load anaconda
source activate rainyday

source __directories.sh

# SLURM_ARRAY_TASK_ID=152 # for testing
len=$(expr length $SLURM_ARRAY_TASK_ID)
if [ $len == 2 ]
then
    year_2dig=${SLURM_ARRAY_TASK_ID:0:1}
    template_num=${SLURM_ARRAY_TASK_ID:1:1}
else
    year_2dig=${SLURM_ARRAY_TASK_ID:0:2}
    template_num=${SLURM_ARRAY_TASK_ID:2:2}
fi


if [ $year_2dig -lt 10 ]
then
	year=200$year_2dig
else
	year=20$year_2dig
fi
# echo $template_num # for testing
# echo $year # for testing

year_tem=$year$template_num

# generate SST script for the year
sst_in=$(python ${assar_dirs[hpc_d_py]} ${year_tem})

echo "Running sst for year $year using file $sst_in"

# running RainyDay
python ${assar_dirs[hpc_rainyday_py]} ${sst_in}