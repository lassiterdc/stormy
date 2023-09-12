


#%% defining directories
fldr_main = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/"
fldr_rainyday_working_dir = fldr_main + "norfolk/"

#%% script specific directories
# non script specific filepaths and folders
f_shp_wshed = fldr_rainyday_working_dir + "watershed/norfolk_wshed_4326.shp"
f_shp_trans_dom = fldr_rainyday_working_dir + "transposition_domain/norfolk_trans_dom_4326.shp"

# work_a

scen_name = "mrms_combined"
fldr_rainyday_outputs = fldr_rainyday_working_dir + "sst_mrms/{}/".format(scen_name)
f_csv_freq_analysis = fldr_rainyday_outputs + "{}_FreqAnalysis.csv".format(scen_name)
fldr_realizations = fldr_rainyday_outputs + "Realizations/"
# f_nc_storm_cat = fldr_rainyday_working_dir + "norfolk_mrms_sst_subset_rivanna.nc"



#%% defining functions for working scripts
# def work_a_inspctng_strm_cat():
#     return f_nc_storm_cat, f_csv_freq_analysis, fldr_realizations, f_shp_wshed, f_shp_trans_dom

