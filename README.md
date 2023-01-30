This repository provides the simulations about the routing algorithm in OppNet.
It also supports the auxiliary information privacy-preserving method
based the differential privacy.

The simulation is implemented in Python.
It depends on the python packages, e.g., numpy, sklearn, pandas and cvxopt.
The figuring code is implemented in Matlab.

# Dataset 
[The dataset about the bicycle trips are published in this folder.](
https://github.com/gaoyang23nj/DiffPrivacyOppNet/tree/main/01-code/EncoHistData_NJBike
)

The bike stations are distributed in Pukou area ([117 stations](https://github.com/gaoyang23nj/DiffPrivacyOppNet/blob/main/01-code/EncoHistData_NJBike/station_info_pukou.csv))
and Qiaobei area ([99 stations](https://github.com/gaoyang23nj/DiffPrivacyOppNet/blob/main/01-code/EncoHistData_NJBike/station_info_qiaobei.csv)), Nanjing, China.
The published dataset has been pre-processed.
In each trip, the check-in station and the check-out station can be identified.

The number of trips between Pukou area and Qiaobei area is few (about 4,000 trips).
However, the number of trips among the stations in Qiaobei area
is more than 580,000 according to the file
[data_qiaobei.csv](
https://github.com/gaoyang23nj/DiffPrivacyOppNet/blob/main/01-code/EncoHistData_NJBike/data_qiaobei.csv
).
In the other word, the trips in Pukou area and Qiaobei area are relatively independent.

# Simulation

[The main programs in this folder](
https://github.com/gaoyang23nj/DiffPrivacyOppNet/tree/main/01-code/Main
) includes 'MainSimulator_XXX.py' in 'DiffPrivacyOppNet/01-code/Main/'.
For example, conducting
['MainSimulator_varyttl_OpDP.py'](https://github.com/gaoyang23nj/DiffPrivacyOppNet/blob/main/01-code/Main/MainSimulator_varyttl_OpDP.py)
will present the varying routing performace with the increasing TTL
(the message time lifespan or the message time-to-live).

# Figuring
[The figuring programs in this folder](
https://github.com/gaoyang23nj/DiffPrivacyOppNet/tree/main/01-code/RevisedVersion_EXP
) is the '*.m' file, which should be conducted by Matlab.