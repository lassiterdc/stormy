# to run, enter the following in the command line:
# chmod +x c5__through_c8b_batch_script.sh
# ./c5__through_c8b_batch_script.sh

ID1=$(sbatch --parsable c5_creating_inps.sh)
ID2=$(sbatch --parsable --dependency=afterok:$ID1 c6_running_swmm.sh)
ID3=$(sbatch --parsable --dependency=afterok:$ID2 c6b_analyzing_swmm_runs.sh)
ID4=$(sbatch --parsable --dependency=afterok:$ID3 c7_consolidating_swmm_outputs.sh)
ID5=$(sbatch --parsable --dependency=afterok:$ID4 c7b_consolidating_swmm_outputs.sh)
ID6=$(sbatch --parsable --dependency=afterok:$ID5 c8_bootstrapping_results.sh)
ID7=$(sbatch --parsable --dependency=afterok:$ID6 c8b_consolidating_bs_samples.sh)