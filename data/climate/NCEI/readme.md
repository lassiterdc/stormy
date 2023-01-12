# Data Sources:
- [Mapper used to download daily data from stations around Norfolk](https://www.ncei.noaa.gov/maps/daily/)
  - I downloaded data in a rectangle that included the station in Newport News, Suffolk, and Virginia Beach
- [Mapper used to download sub-daily data from stations around Norfolk](https://www.ncei.noaa.gov/maps/hourly/)
  - for the same range, I downloaded 'Hourly Global' and 'Hourly Precip. (1950s-2014)'
  - I downloaded the 'Hourly Global' stations list and found that only 4 began data collection before 2000 and are still collecting data today:
    - (STATIONS) LANGLEY AFB AIRPORT, FELKER ARMY AIRFIELD, NORFOLK INTERNATIONAL AIRPORT, NORFOLK NAS
    - (STATION IDS) 74598013702, 72308793735, 72308013737, 72308513750

# Data Descriptions
- 2023-1-4_NCEI_daily summaries_download.csv
  - This contains every available daily dataset for 8 stations: NORFOLK NAS, VA US
NORFOLK NAS, VA US
NORFOLK INTERNATIONAL AIRPORT, VA US
OCEANA NAS, VA US
NEWPORT NEWS INTERNATIONAL AIRPORT, VA US
SUFFOLK LAKE KILBY, VA US
FENTRESS NAVAL AUXILIARY FIELD, VA US
HAMPTON ROADS EXECUTIVE AIRPORT, VA US
NORFOLK SOUTH, VA US
  - The key variable is **PRCP**
  - Inspecting the date ranges of available data, the only active stations with data prior to 2000 are:
    - FENTRESS NAVAL AUXILIARY FIELD, VA US
    - NORFOLK INTERNATIONAL AIRPORT, VA US
    - NORFOLK NAS, VA US
    - OCEANA NAS, VA US
    - SUFFOLK LAKE KILBY, VA US
  - Honorable mention
    -  NEWPORT NEWS INTERNATIONAL AIRPORT, VA US datest back to 2000
    - NORFOLK SOUTH, VA US dates back to 2003
- ~~2023-1-4_NCEI_Hourly Global_precip.csv~~
  - Hourly data for a bunch of stations
- 2023-1-4_NCEI_Hourly Global_station info.csv
  - Used to select stations from the Hourly Global dataset that have a sufficiently long record
  - only 4 began data collection before 2000 and are still collecting data today:
    - (STATIONS) LANGLEY AFB AIRPORT, FELKER ARMY AIRFIELD, NORFOLK INTERNATIONAL AIRPORT, NORFOLK NAS
    - (STATION IDS) 74598013702, 72308793735, 72308013737, 72308513750
- 2023-1-4_NCEI_Hourly Precip_station_subset_precip.csv
  - Data for the four stations identified in 2023-1-4_NCEI_Hourly Global_station info.csv
  - 
- 2023-1-4_NCEI_Hourly Precip_precip.csv
  - Downloaded using the 'Hourly Precip. (1950s-2014)' selection from the mapper
  - Contains precipitation data for CAPE HENRY VA US, NORFOLK INTERNATIONAL AIRPORT VA US, and NORFOLK WEATHER BUREAU CITY VA US.
  - **The only relevant one is NORFOLK INTERNATIONAL AIRPORT VA US which spans 8/5/1948 to 12/29/2013**
  - The other two just span two short periods, CAPE HENRY from 8/3/1948 to 8/1/1953, and NORFOLK WEATHER BUREAU CITY VA US from 8/4/1948 to 7/29/1953.


