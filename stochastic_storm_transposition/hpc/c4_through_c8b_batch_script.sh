#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name # can use dev for testing
#SBATCH -t 24:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=1	        # Array of jobs, 1 for each of 1000 years
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# to run, enter the following in the command line:
# chmod +x c4_through_c8b_batch_script.sh
# ./c4_through_c8b_batch_script.sh

# ID1=$(sbatch --parsable c5_creating_inps.sh)
# ID2=$(sbatch --parsable --dependency=afterok:$ID1 c6_running_swmm.sh)
# ID3=$(sbatch --parsable --dependency=afterok:$ID2 c6b_analyzing_swmm_runs.sh)
ID3=$(sbatch --parsable c6b_analyzing_swmm_runs.sh) # DELETE AFTER TROUBLESHOOTING
ID4=$(sbatch --parsable --dependency=afterok:$ID3 c7_consolidating_swmm_outputs.sh)
# ID5=$(sbatch --parsable --dependency=afterok:$ID4 c7b_consolidating_swmm_outputs.sh)
# ID6=$(sbatch --parsable --dependency=afterok:$ID5 c8_bootstrapping_results.sh)
# ID7=$(sbatch --parsable --dependency=afterok:$ID6 c8b_consolidating_bs_samples.sh)