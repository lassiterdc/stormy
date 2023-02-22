#%% notes
"""
Source: https://waterprogramming.wordpress.com/2016/06/03/pythons-template-class/
"""

#%% local testing
# dir_mrms_hrly = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/norfolk/sst_mrms_hourly/"
# f_sst_mrms_hrly = dir_mrms_hrly + "mrms_hourly_template.sst"
# dir_for_sst_files = dir_mrms_hrly + "_inputs/"
# year = "2001" # this will ultimately be a user input file
#%% import libraries and load directories
from __utils import c_rainyday_in
from string import Template
import pathlib
import sys
dir_mrms_hrly, f_sst_mrms_hrly, dir_for_sst_files = c_rainyday_in()

# inputs
year = str(sys.argv[1]) #YYYY
#%% creating a new sst for the year

with open(f_sst_mrms_hrly, 'r') as T:
    template = Template(T.read())
    d = {"YEAR":year}
    new_in = template.safe_substitute(d)
    new_file = '{}{}.sst'.format(dir_for_sst_files, year)
    new_file_path = pathlib.Path(new_file)
    # create _inputs folder if it doesn't already exist
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    # new_file_path.touch()
    with open (new_file, "w+") as f1:
        f1.write(new_in)

#%% generate output for bash script
print(new_file)