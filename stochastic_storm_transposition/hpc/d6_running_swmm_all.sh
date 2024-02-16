#!/bin/bash
#SBATCH -o _script_outputs/%x/%A_%a_%N.out
#SBATCH -e _script_errors/%x/%A_%a_%N.out
#SBATCH --ntasks=1				# Number of tasks per serial job (must be 1)
#SBATCH -p standard				# Queue name "standard" (serial)
#SBATCH -A quinnlab_paid				# allocation name # can use dev for testing
#SBATCH -t 15:00:00				# Run time per serial job (hh:mm:ss)
#SBATCH --array=2,8,10,11,19,22,24,25,26,28,29,37,40,42,43,45,51,52,58,62,63,65,67,68,69,70,71,72,73,75,80,81,82,84,85,86,87,88,89,91,92,93,95,96,97,98,100,102,103,104,106,108,110,113,114,115,119,120,121,122,123,125,126,127,128,129,130,131,132,139,142,145,146,148,152,153,154,155,157,158,160,163,164,165,167,168,170,171,174,175,176,179,180,181,182,183,185,186,190,193,194,195,198,199,200,201,204,205,206,210,212,213,214,215,216,221,222,223,227,230,236,238,239,240,241,244,245,246,247,248,250,253,256,257,260,261,263,266,268,269,270,271,274,275,278,284,288,290,294,295,297,298,299,301,303,305,306,307,311,312,314,317,319,320,323,325,327,329,333,334,335,336,337,338,340,342,343,345,347,348,349,358,359,360,361,362,368,374,375,376,377,380,383,386,387,388,389,390,391,392,394,398,399,400,405,408,409,410,412,413,414,416,419,423,424,426,428,429,430,431,432,433,434,435,436,437,438,439,442,443,444,446,448,451,452,454,456,458,459,461,462,463,464,465,468,470,475,478,481,485,486,489,492,493,494,495,501,503,505,513,518,521,525,526,527,531,532,533,535,544,546,549,550,554,559,564,566,569,571,572,573,576,577,579,582,583,587,588,590,592,595,599,600,602,604,607,608,609,611,612,613,617,619,621,622,623,624,626,628,629,630,635,636,637,640,643,645,646,654,656,657,659,661,663,665,666,670,671,672,674,678,679,684,686,687,689,691,694,696,698,700,701,704,705,706,707,713,714,715,716,718,721,722,723,725,726,727,728,729,735,736,737,738,740,741,743,745,747,748,752,755,756,759,760,761,763,764,766,767,769,772,774,775,776,777,779,780,781,784,788,793,794,796,797,798,799,800,802,803,807,808,811,813,815,816,818,820,823,824,826,828,829,831,832,833,835,838,839,843,844,845,846,852,854,858,860,862,864,865,867,870,871,872,873,874,875,876,885,887,889,893,894,896,898,902,906,907,909,910,911,912,913,914,915,916,917,918,919,920,922,925,928,930,931,933,934,943,948,950,951,952,959,964,966,968,970,971,976,978,980,982,983,984,987,988,989,994,997	# Array of jobs, 1 for each of 1000 years (1-1000)
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-user=dcl3nd@virginia.edu          # address for email notification
#SBATCH --mail-type=ALL   
# SBATCH --exclude=udc-ba26-18,udc-ba27-14,udc-ba26-16,udc-ba26-17

# ijob -c 1 -A quinnlab_paid -p standard --time=0-09:00:00 --mem-per-cpu=16000

source __directories.sh
module purge
module load gcc openmpi eccodes anaconda # the stuff other than anaconda was to ensure eccodes loaded correctly
# DIR=~/.conda/envs/running_swmm
source activate running_swmm
# export PATH=$DIR/bin:$PATH
# export LD_LIBRARY_PATH=$DIR/lib:$PATH
# export PYTHONPATH=$DIR/lib/python3.11/site-packages:$PATH

# echo "Running SWMM for year ${SLURM_ARRAY_TASK_ID}..."

# running swmm
python ${assar_dirs[hpc_d6_py]} ${SLURM_ARRAY_TASK_ID} "all" 1 1 # arguments: year, which models to run (failed, all, or specific storm number), which realizations to run, and whether to delete swmm .out files
# python ${assar_dirs[hpc_d6_py]} 764 "all" 2 0
# python ${assar_dirs[hpc_d6_py]} 99999 "all" 1 0 # this is a duplicate of year 251 that has been causing problems
# python ${assar_dirs[hpc_d6_py]} 252 "all" 1 0
# python ${assar_dirs[hpc_d6_py]} 268 2 1 0
