ijob -c 1 -A quinnlab_paid -p standard --time=1-00:00:00 --mem=175000
module load anaconda
source activate rainyday
cd /project/quinnlab/dcl3nd/norfolk/stormy/stochastic_storm_transposition
#%% running on Rivanna
# using a subset of mrms data
python "RainyDay2/Source/RainyDay_Py3.py" "norfolk/norfolk_mrms_subset_rivanna.sst"




#%% running on Rivanna (old)
# using subset of mrms data
# python "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/norfolk_mrms_subset_rivanna.sst"
# # using all mrms data
# python "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/norfolk_mrms_rivanna.sst"
# # using stage IV data
# python "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/norfolk_stage_iv_rivanna.sst"
# # reproducing BigThompson exacmple
# python "/project/quinnlab/dcl3nd/norfolk/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "/project/quinnlab/dcl3nd/norfolk/RainyDay2/_reproducing_BigThompsonExample/BigThompson72hr_example_rivanna.sst"

# #%% Running on local computer (old)
# #laptop environment: RainyDay
# # running using StaveIV data
# python "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/norfolk_stage_iv_local.sst"
# # running using subset of mrms data that is also consolidated to an hourly timestep
# python "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/norfolk_mrms_subset_hourly_local.sst"
# # reproducing Big Thompson Example
# python "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/norfolk_mrms/Source/RainyDay_Py3.py" "D:/mrms_processing/scripts_rivanna_revised/RainyDay2/_reproducing_BigThompsonExample/BigThompson72hr_example_local.sst"