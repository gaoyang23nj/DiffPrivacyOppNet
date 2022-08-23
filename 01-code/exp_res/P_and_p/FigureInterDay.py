# 只按照周末来分, 没有分出是否节假日
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd

WeatherInfo = './Pukou_Weather.xlsx'
# 学校-地铁站
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/86_6.csv'
# 住宅小区-地铁口
TARGET_ENCOHIST1 = './210_183.csv'
# 地铁口-住宅小区
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/183_210.csv'

TARGET_ENCOHIST2 = './35_4.csv'

def readWeather():
    list_weather = []
    weather = pd.read_excel(WeatherInfo, engine='openpyxl')
    v1 = weather.values
    for i in range(0, len(weather.values)):
        tm = time.strptime(v1[i][0].split(' ')[0], "%Y-%m-%d")
        tmp_high = int(v1[i][1].replace('°', ''))
        tmp_low = int(v1[i][2].replace('°', ''))
        if 1.0 == v1[i][6]:
            is_holiday = True
        elif 0.0 == v1[i][6]:
            is_holiday = False
        elif (tm.tm_wday == 5) or (tm.tm_wday == 6):
            is_holiday = True
        else:
            is_holiday = False
        list_weather.append((tm, tmp_high, tmp_low, is_holiday))
    return list_weather

def getValuePandC(target_file):
    num_of_nodes = 1
    list_num_trip = np.ones((NUM_DAYS_INYEAR), dtype='int') * 0
    rho_star = np.ones((NUM_DAYS_INYEAR)) * -1
    # list 成员是 P值, 每天增加新的值
    P_holiday = []
    P_workday = []
    list_enco_hist = []

    file_object = open(target_file, 'r', encoding="utf-8")
    tmp_all_lines = file_object.readlines()
    for index in range(len(tmp_all_lines)):
        (i_station_id, a_time, i_node, j_station_id, b_time, j_node, i_name, j_name) \
            = tmp_all_lines[index].strip().split(',')
        tm = datetime.datetime.strptime(a_time, "%Y/%m/%d %H:%M:%S")
        i_node = int(i_node)
        j_node = int(j_node)
        list_enco_hist.append((tm, i_node, j_node))
    file_object.close()

    for ele in list_enco_hist:
        tm = ele[0]
        runningtime = time.strptime(tm.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        # 一年中的 第 runningtime.tm_yday 天
        list_num_trip[runningtime.tm_yday-1] = list_num_trip[runningtime.tm_yday-1] + 1
    rho_star = 1-np.exp(-list_num_trip)
    print(list_num_trip)
    print(rho_star)

    P_plot = np.zeros((NUM_DAYS_INYEAR), dtype='float')
    print(begin_time, end_time)
    for dayth in range(begin_time, end_time + 1):
        print(dayth)
        if list_weather[dayth][3]:
            if len(P_holiday) == 0:
                P_holiday.append((dayth, rho_star[dayth]))
            else:
                new_prob = alpha * P_holiday[-1][1] + (1 - alpha) * rho_star[dayth]
                P_holiday.append((dayth, new_prob))
            P_plot[dayth] = P_holiday[-1][1]
        else:
            if len(P_workday) == 0:
                P_workday.append((dayth, rho_star[dayth]))
            else:
                new_prob = alpha * P_workday[-1][1] + (1 - alpha) * rho_star[dayth]
                P_workday.append((dayth, new_prob))
            P_plot[dayth] = P_workday[-1][1]

    return P_plot, list_num_trip

alpha = 0.7
NUM_DAYS_INYEAR = 365
begin_time = time.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
end_time = time.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
list_weather = []
list_weather = readWeather()

P_plot1, C_plot1 = getValuePandC(TARGET_ENCOHIST1)
P_plot2, C_plot2 = getValuePandC(TARGET_ENCOHIST2)

x_h = np.linspace(begin_time, end_time, end_time-begin_time+1)

# 写入文件intra_day
file_object = open('inter_day.csv', 'w+', encoding="utf-8")

for i in range(len(x_h)):
    file_object.write('{}'.format(x_h[i]))
    if i != len(x_h)-1:
        file_object.write(',')
file_object.write('\r')

# C_plot1
C1 = C_plot1[begin_time:end_time+1]
for i in range(len(C1)):
    file_object.write('{}'.format(C1[i]))
    if i != len(C1)-1:
        file_object.write(',')
file_object.write('\r')

# C_plot2
C2 = C_plot2[begin_time:end_time+1]
for i in range(len(C2)):
    file_object.write('{}'.format(C2[i]))
    if i != len(C2)-1:
        file_object.write(',')
file_object.write('\r')

P1 = P_plot1[begin_time:end_time+1]
for i in range(len(P1)):
    file_object.write('{}'.format(P1[i]))
    if i != len(P1)-1:
        file_object.write(',')
file_object.write('\r')

P2 = P_plot2[begin_time:end_time+1]
for i in range(len(P2)):
    file_object.write('{}'.format(P2[i]))
    if i != len(P2)-1:
        file_object.write(',')
file_object.write('\r')

file_object.close()

# 起始值 结束值 多少个
plt.rc('font',family='Times New Roman')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
A1 = ax1.plot(x_h, C_plot1[begin_time:end_time+1], color='blue', ls="-", markersize=5, label='$C_{210,183}^k$')
A2 = ax1.plot(x_h, C_plot2[begin_time:end_time+1], color='blue', ls="-.", markersize=5, label='$C_{35,4}^k$')
B1 = ax2.plot(x_h, P_plot1[begin_time:end_time+1], color='red', ls="-", markersize=5, label='$P_{210,183}^{k}$')
B2 = ax2.plot(x_h, P_plot2[begin_time:end_time+1], color='red', ls="-.", markersize=5, label='$P_{35,4}^{k}$')

# legend
lns = A1+A2+B1+B2
# lns = A1+B1
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0, fontsize=10)

ax1.set_xlabel('$k$th day in the year', fontsize=15)
ax1.set_ylabel('No. of trips from $i$ to $j$ ($C_{i,j}^{k}$)', color='b', fontsize=15)
ax2.set_ylabel('Prob. that the contact occurs ($P_{i,j}^{k}$)', color='r', fontsize=15)

plt.tight_layout()
plt.show()
print(1)