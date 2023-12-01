# to run, enter the following in the command line:
# chmod +x c5__through_c8b_batch_script.sh
# ./c5__through_c8b_batch_script.sh

ID1=$(sbatch --parsable d5_creating_inps.sh)
ID2=$(sbatch --parsable --dependency=afterok:$ID1 d6_running_swmm.sh)
ID3=$(sbatch --parsable --dependency=afterok:$ID2 d6b_analyzing_swmm_runs.sh)
ID4=$(sbatch --parsable --dependency=afterok:$ID3 d7_consolidating_swmm_outputs.sh)
ID5=$(sbatch --parsable --dependency=afterok:$ID4 d7b_consolidating_swmm_outputs.sh)
ID6=$(sbatch --parsable --dependency=afterok:$ID5 d8_bootstrapping_results.sh)
ID7=$(sbatch --parsable --dependency=afterok:$ID6 d8b_consolidating_bs_samples.sh)