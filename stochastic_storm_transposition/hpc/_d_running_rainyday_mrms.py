#%% notes
"""
Source: https://waterprogramming.wordpress.com/2016/06/03/pythons-template-class/
"""

#%% local testing
# dir_mrms = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/norfolk/sst_mrms_subdivided/"
# f_sst_mrms = dir_mrms + "mrms_template_{}.sst"
# dir_for_sst_files = dir_mrms + "_inputs/"
# slurm_input = "20011" # this will ultimately be a user input file
#%% import libraries and load directories
from __utils import *
from string import Template
import pathlib
import sys
# inputs
year = str(sys.argv[1]) #YYYY
# if len(slurm_input) == 4:
#     year = slurm_input
#     dir_mrms, f_sst_mrms, dir_for_sst_files = d_rainyday_in()
# else:
#     year = slurm_input[:4]
#     template_id = slurm_input[4:5]

#     dir_mrms, f_sst_mrms, dir_for_sst_files = da_rainyday_in()
#     f_sst_mrms = f_sst_mrms.format(template_id)
#%% creating a new sst for the year

with open(f_sst_mrms, 'r') as T:
    template = Template(T.read())
    d = {"YEAR":year}
    new_in = template.safe_substitute(d)
    new_file = '{}{}.sst'.format(dir_for_sst_files, slurm_input)
    new_file_path = pathlib.Path(new_file)
    # create _inputs folder if it doesn't already exist
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    # new_file_path.touch()
    with open (new_file, "w+") as f1:
        f1.write(new_in)

#%% generate output for bash script
print(new_file)