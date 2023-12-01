#!/bin/bash
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name
#SBATCH -t 1:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1 

ID2=$(sbatch --parsable d6_running_swmm.sh)
# ID2=$(sbatch --parsable --dependency=afterok:$ID1 d6_running_swmm.sh)
ID3=$(sbatch --parsable --dependency=afterok:$ID2 d6b_analyzing_swmm_runs.sh)
ID4=$(sbatch --parsable --dependency=afterok:$ID3 d7_consolidating_swmm_outputs.sh)
ID5=$(sbatch --parsable --dependency=afterok:$ID4 d7b_consolidating_swmm_outputs.sh)
ID6=$(sbatch --parsable --dependency=afterok:$ID5 d8_bootstrapping_results.sh)
ID7=$(sbatch --parsable --dependency=afterok:$ID6 d8b_consolidating_bs_samples.sh)