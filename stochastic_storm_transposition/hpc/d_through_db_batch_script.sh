# to run, enter the following in the command line:
# chmod +x c__through_c4b_batch_script.sh
# ./c__through_c4b_batch_script.sh

ID1=$(sbatch --parsable d_running_rainyday_mrms_subdivided_yearly.sh)
ID2=$(sbatch --parsable --dependency=afterok:$ID1 d2_resampling_storms.sh)
ID3=$(sbatch --parsable --dependency=afterok:$ID2 d4_creating_rainfall_timeseries.sh)
# ID4=$(sbatch --parsable --dependency=afterok:$ID3 c3a_creating_reformatted_catalog_for_xarray.sh)
# ID5=$(sbatch --parsable --dependency=afterok:$ID4 c3b_creating_reformatted_realizations_for_xarray.sh)
# ID6=$(sbatch --parsable --dependency=afterok:$ID5 c4_creating_rainfall_timeseries.sh)
# ID7=$(sbatch --parsable --dependency=afterok:$ID6 c4b_creating_waterlevel_timeseries.sh)