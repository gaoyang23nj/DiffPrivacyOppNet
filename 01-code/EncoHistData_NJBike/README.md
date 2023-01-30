This folder contains the bicycle trip data and the station information.
And these data have been pre-processed.

# Clean Bicycle Trip Data Files (Useful)
data.csv
1. utf-8, Store the bicycle trips beween stations of both 'Pukou' and 'Qiaobei' town in 2017.
2. one item (line) in this file is a bicycle trip.
3. Table Header / Data Format: 
( Globe ID of the check-in station, checking-in time, Inner ID of the check-in station,
Globe ID of the check-out station, checking-out time, Inner ID of the check-out station,
Name of the check-out station, Name of the check-in station)

data_pukou.csv
1. utf-8, Store the bicycle trips beween stations of 'Pukou' town in 2017
2. Same trip data format as the above

data_qiaobei.csv
1. utf-8, Store the bicycle trips beween stations of 'Qiaobei' town in 2017
2. Same trip data format as the above

# Dirty Bicycle Trip Data Files (Useless)
data_bad.csv
1. utf-8, Store the bicycle trips relevant to other towns (besides 'Pukou' and 'Qiaobei' town), e.g., blank station name ''
2. same trip data format as the above

data_bad_pukou.csv
1. utf-8, Store the bicycle trips beween besides 'Pukou' town in 2017
2. Same trip data format as the above

data_bad_qiaobei.csv
1. utf-8, Store the bicycle trips beween besides 'Qiaobei' town in 2017
2. Same trip data format as the above

# Station info (Useful)
station_info.csv
1. utf-8, store 'station name' and 'station id'.
2. One item (line) in this file is a staion.
3. Table Header / Data Format: (Globe ID of the station, INNER ID of the station, Station's Name)

station_info_pukou.csv
1. utf-8, Store 'station name' and 'station id', 117 stations (0~116)
2. One item (line) in this file is a staion.
3. Same trip data format as the above

station_info_qiaobei.csv
1. utf-8, store 'station name' and 'station id', 99 stations (0~98)
2. One item (line) in this file is a staion.
3. Same trip data format as the above