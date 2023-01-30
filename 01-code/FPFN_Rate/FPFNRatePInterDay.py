# 只按照周末来分, 没有分出是否节假日
import os
import time
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd

# # 学校-地铁站
# # TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/86_6.csv'
# # 住宅小区-地铁口
# TARGET_ENCOHIST1 = './210_183.csv'
# # 地铁口-住宅小区
# # TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/183_210.csv'
#
# TARGET_ENCOHIST2 = './35_4.csv'

WeatherInfo = '../EncoHistData_NJBike/Pukou_Weather.xlsx'

FILEPATH = '../EncoHistData_NJBike/SDPair_NJBike_Data_qiaobei'

OUTPUTFILENAME_PC = 'matrix_Pinter.npz'
OUTPUTFILENAME_ThFPFN_Rate = 'FP_FN_rate.npz'

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

def getValuePandC(target_file, begin_datetime, end_datetime, list_weather):
    #
    # begin_datetime = datetime.datetime.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S")
    # end_datetime = datetime.datetime.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S")

    str_begin = begin_datetime.strftime("%Y/%m/%d %H:%M:%S")
    str_end = end_datetime.strftime("%Y/%m/%d %H:%M:%S")
    begin_time = time.strptime(str_begin, "%Y/%m/%d %H:%M:%S")
    end_time = time.strptime(str_end, "%Y/%m/%d %H:%M:%S")

    begin_time_dayinyear = begin_time.tm_yday-1
    end_time_dayinyear = end_time.tm_yday-1

    num_of_nodes = 1
    list_num_trip = np.ones((NUM_DAYS_INYEAR), dtype='int') * 0
    rho_star = np.ones((NUM_DAYS_INYEAR)) * -1
    # list 成员是 P值, 每天增加新的值
    P_holiday = []
    P_workday = []
    list_enco_hist = []

    # C is calculated in all the days of this year
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
    # print(list_num_trip)
    # print(rho_star)

    # P is calcualted only in [June, July]
    P_plot = np.ones((NUM_DAYS_INYEAR), dtype='float')*-1
    month = np.ones((NUM_DAYS_INYEAR), dtype='float')*-1
    dayinmonth = np.ones((NUM_DAYS_INYEAR), dtype='float')*-1
    print(begin_time_dayinyear, end_time_dayinyear)
    tmp_datetime = begin_datetime
    for dayth in range(begin_time_dayinyear, end_time_dayinyear + 1):
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


        str_tmp = tmp_datetime.strftime("%Y/%m/%d %H:%M:%S")
        tmp_time = time.strptime(str_tmp, "%Y/%m/%d %H:%M:%S")
        month[dayth] = tmp_time.tm_mon
        dayinmonth[dayth] = tmp_time.tm_mday

        # print(dayth, list_weather[dayth][3], tmp_time.tm_mon, tmp_time.tm_mday, tmp_datetime)
        tmp_datetime = tmp_datetime + datetime.timedelta(days=1)
    return P_plot, list_num_trip, month, dayinmonth


def get_FP_FN_matrix():
    files = os.listdir(FILEPATH)
    # print(files)
    print('number of files:{}'.format(len(files)))

    rows_once = end_time_dayinyear-begin_time_dayinyear+1
    rows = len(files)*rows_once
    all_matrix = np.zeros((rows, 7))
    for onefile_id in range(len(files)):
        onefile = files[onefile_id]
        tmp2tunple = onefile.split('.')[0]
        a_node, b_node = tmp2tunple.split('_')
        absfilepath = os.path.join(FILEPATH, onefile)
        print(a_node, b_node, onefile, absfilepath)


        # get P and C for all the days
        P_plot, C_plot, month, dayinmonth = getValuePandC(absfilepath, begin_datetime, end_datetime, list_weather)
        x_h = np.linspace(begin_time_dayinyear, end_time_dayinyear, end_time_dayinyear - begin_time_dayinyear + 1)
        # P, C, a_node, b_node, number of days(time), month, dayinmonth
        matrix = np.zeros((len(x_h), 7))
        matrix[:, 0] = P_plot[begin_time_dayinyear:end_time_dayinyear + 1]
        matrix[:, 1] = C_plot[begin_time_dayinyear:end_time_dayinyear + 1]
        matrix[:, 2] = a_node
        matrix[:, 3] = b_node
        matrix[:, 4] = x_h
        matrix[:, 5] = month[begin_time_dayinyear:end_time_dayinyear + 1]
        matrix[:, 6] = dayinmonth[begin_time_dayinyear:end_time_dayinyear + 1]

        all_matrix[rows_once * onefile_id : rows_once * (onefile_id + 1), :] = matrix

        print('hi')

    np.savez(OUTPUTFILENAME_PC, all_matrix = all_matrix)
    return all_matrix


alpha = 0.7
NUM_DAYS_INYEAR = 365

begin_datetime = datetime.datetime.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S")
end_datetime = datetime.datetime.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S")
str_begin = begin_datetime.strftime("%Y/%m/%d %H:%M:%S")
str_end = end_datetime.strftime("%Y/%m/%d %H:%M:%S")
begin_time = time.strptime(str_begin, "%Y/%m/%d %H:%M:%S")
end_time = time.strptime(str_end, "%Y/%m/%d %H:%M:%S")

begin_time_dayinyear = begin_time.tm_yday - 1
end_time_dayinyear = end_time.tm_yday - 1

list_weather = []
list_weather = readWeather()
threshold = 0.5


if __name__ == "__main__":
    if os.path.exists(OUTPUTFILENAME_PC):
        all_matrix = np.load(OUTPUTFILENAME_PC)['all_matrix']
    else:
        all_matrix = get_FP_FN_matrix()

    # granularity 0.01
    list_th = np.arange(0, 1, 0.01)
    list_FP_rate = np.ones((len(list_th)))*-1
    list_FN_rate = np.ones((len(list_th)))*-1
    for threshold_i in range(len(list_th)):
        threshold = list_th[threshold_i]
        count_FP = 0
        count_FN = 0
        count_TP = 0
        count_TN = 0

        for rows in range(all_matrix.shape[0]):
            label_believe = False
            if all_matrix[rows, 0] > threshold:
                label_believe = True
            label_truth = False
            if all_matrix[rows, 1] > 0:
                label_truth = True

            if label_truth != label_believe:
                if label_believe == True:
                    count_FP = count_FP + 1
                else:
                    count_FN = count_FN + 1
            else:
                if label_believe == True:
                    count_TP = count_TP + 1
                else:
                    count_TN = count_TN + 1
        num_sum = count_FP + count_FN + count_TP + count_TN
        print('Threshold:{} count FP:{}, FN:{}, TP:{}, TN:{} sum:{}'.format(threshold, count_FP, count_FN, count_TP, count_TN, num_sum))
        FP_rate = count_FP/(count_FP+count_TN)
        FN_rate = count_FN/(count_FN+count_TP)
        print('FP rate:{};  FN rate:{}'.format(FP_rate, FN_rate))
        list_FP_rate[threshold_i] = FP_rate
        list_FN_rate[threshold_i] = FN_rate
        print('')

    np.savez(OUTPUTFILENAME_ThFPFN_Rate, list_th = list_th, list_FP_rate = list_FP_rate, list_FN_rate = list_FN_rate)

    outputfilenamecsv = 'FPFNrate.csv'
    f = open(outputfilenamecsv, 'w+')
    for i in range(len(list_th)):
        f.write('{},{},{}\n'.format(list_th[i], list_FP_rate[i], list_FN_rate[i]))
    f.close()
    plt.figure(0)
    A1 = plt.plot(list_th, list_FP_rate, color='blue', ls="-", markersize=5, label='FP')
    A2 = plt.plot(list_th, list_FN_rate, color='red', ls="-", markersize=5, label='FN')
    plt.show()