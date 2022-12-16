"""
This script downloads tide gage data and tide prediction data (the difference
between the two is storm surge) and save them to a .csv. This script also 
downloads the station metadata and saves it to a JSON and it creates a shapefile
at the location of the gage.

Last significant edits: 5/27/2022
"""
#%% import libraries and define parameters
from _directories import def_fpaths_for_a
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import json
import geopandas as gp
import noaa_coops as nc
import numpy as np

begin_year = 2001
end_year = 2023

fld_out_a = def_fpaths_for_a()

#%% download tide gage and tide prediction data
years = np.arange(begin_year+1, end_year+1, 1) 
md = "0101"
record_start = "19270701"
sta_id = 8638610 # sewells point
prods = ["water_level", 'predictions', "wind", "air_pressure", "one_minute_water_level", "datums"]
datum = "NAVD"
units = "english"
time_zone = "lst" # local standard time
sta = nc.Station(sta_id)

lat, lon = sta.lat_lon

dfs_wl = []
dfs_tide_pred = []

for y in years:
    b_yr = str(y-1)
    e_yr = str(y)
    
    if y != 1927+1:
        b_date = b_yr + md
    else:
        b_date = record_start

    e_date = e_yr + md
    
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
#%% saving tide data and metadata
with open(fldr_out + 'sewells_pt_water_level_metadatap.json', 'w', encoding='utf-8') as outfile:
    json.dump(metadata,outfile,ensure_ascii=False, indent=4)

yrs = "_{}_to_{}".format(str(min(years-1)), str(max(years)))

df_wl.to_csv(fldr_out+"a_water_level{}.csv".format(yrs))
df_tide_pred.to_csv(fldr_out+"a_tide_preds{}.csv".format(yrs))

keys=['name', 'lat', 'lng']
geo_data={key:metadata[key] for key in keys}

df = pd.DataFrame(geo_data, index=[0])

gdf = gp.GeoDataFrame(df,
                             geometry=gp.points_from_xy(df.lat, df.lng))

gdf.to_file(fldr_out+"sewells_pt.shp")
