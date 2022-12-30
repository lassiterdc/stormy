"""
This script downloads tide gage data and tide prediction data. It also
computes storm surge (the difference
between the two) and saves the data to a .csv. This script also 
downloads the station metadata and saves it to a JSON and it creates a shapefile
at the location of the gage.

Last significant edits: 12/16/22
"""
#%% import libraries and define parameters
from _inputs import def_inputs_for_a
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json
import geopandas as gp
import noaa_coops as nc
import numpy as np
from datetime import date


begin_year, f_out_a_meta, f_out_a_all, f_out_a_shp = def_inputs_for_a()

# parameters for downloading data
b_md = "0101" # start date of each year downloaded
e_md = "1231" # end date of each year downloaded
record_start = "19270701" # beginning of record for station
sta_id = 8638610 # sewells point gage id
#%% download tide gage and tide prediction data
tday = date.today()
end_year = tday.year
end_md = "{}{}".format(tday.month, tday.day)

years = np.arange(begin_year, end_year+1, 1) 

prods = ["water_level", 'predictions', "wind", "air_pressure", "one_minute_water_level", "datums"]
datum = "NAVD"
units = "english"
time_zone = "lst" # local standard time
sta = nc.Station(sta_id)

lat, lon = sta.lat_lon

dfs_wl = []
dfs_tide_pred = []

for y in years:
    yr = str(y)
    # e_yr = str(y)
    
    if y == 1927:
        b_date = record_start
    else:
        b_date = yr + b_md

    if y == end_year:
        e_date = yr + end_md
    else:
        e_date = yr + e_md
    
    print("Begin: {}, End: {}".format(b_date, e_date))


    data_wl = sta.get_data(begin_date=b_date,
                        end_date=e_date,
                        product=prods[0],
                        datum=datum,
                        units=units,
                        time_zone=time_zone
        )
    
    data_tide_pred = sta.get_data(begin_date=b_date,
                        end_date=e_date,
                        product=prods[1],
                        datum=datum,
                        units=units,
                        time_zone=time_zone
        )
    
    dfs_wl.append(data_wl)
    dfs_tide_pred.append(data_tide_pred)
    
metadata = sta.metadata

df_wl = pd.concat(dfs_wl)
df_wl = df_wl[~df_wl.index.duplicated(keep='first')]

df_tide_pred = pd.concat(dfs_tide_pred)
df_tide_pred = df_tide_pred[~df_tide_pred.index.duplicated(keep='first')]

df_comb = df_wl.join(df_tide_pred, on="date_time", how='left')
df_comb['surge_ft']=df_comb.water_level - df_comb.predicted_wl
#%% saving tide data and metadata
with open(f_out_a_meta, 'w', encoding='utf-8') as outfile:
    json.dump(metadata,outfile,ensure_ascii=False, indent=4)

# yrs = "_{}_to_{}".format(str(min(years-1)), str(max(years)))

# df_wl.to_csv(fld_out_a+"a_water_level{}.csv".format(yrs))
# df_tide_pred.to_csv(fld_out_a+"a_tide_preds{}.csv".format(yrs))
df_comb.to_csv(f_out_a_all)

keys=['name', 'lat', 'lng']
geo_data={key:metadata[key] for key in keys}

df = pd.DataFrame(geo_data, index=[0])

gdf = gp.GeoDataFrame(df, geometry=gp.points_from_xy(df.lat, df.lng))

gdf.to_file(f_out_a_shp)
