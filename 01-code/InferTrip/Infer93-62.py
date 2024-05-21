# 只按照周末来分, 没有分出是否节假日
import time
import datetime
import os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from cvxopt import matrix, solvers
matplotlib.interactive(False)

WeatherInfo = './Pukou_Weather.xlsx'
# 学校-地铁站
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/86_6.csv'
# 住宅小区-地铁口
INFER_TARGET = '../EncoHistData_NJBike/SDPair_NJBike_Data_qiaobei'
TARGET_ENCOHIST1 = './210_183.csv'
# 地铁口-住宅小区
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/183_210.csv'

# TARGET_ENCOHIST2 = './35_4.csv'

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

def getRealValuePandC(target_file):
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
    for dayth in range(begin_time-1, end_time):
        print(dayth, list_weather[dayth], list_num_trip[dayth])
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
NUM_of_NODES = 99
begin_time = time.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
end_time = time.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
list_weather = []
list_weather = readWeather()
Weather_real = list_weather[begin_time-1:end_time]

i_node = 93
P_real = []
C_real = []
for j_node in [62]:
    print('process get P, node {}\n'.format(j_node))
    if j_node == i_node:
        C_real.append([])
        P_real.append([])
        continue
    filename = '93_{}.csv'.format(j_node)
    filename = os.path.join(INFER_TARGET, filename)
    Pij_real_year, Cij_real_year = getRealValuePandC(filename)
    Cij_real = Cij_real_year[begin_time-1:end_time]
    Pij_real = Pij_real_year[begin_time-1:end_time]
    C_real.append(Cij_real)
    P_real.append(Pij_real)

x_h = np.linspace(begin_time, end_time, end_time-begin_time+1)

# inference attack without DP
print('#'*50)
print('inference attack for P')
print(begin_time, end_time)

MSE = []
for j_node in [62]:
    print('process infer C, node {}\n'.format(j_node))
    if j_node == i_node:
        continue
    P_holiday = []
    P_workday = []
    C_est = []
    idx = -1
    for dayth in range(begin_time-1, end_time):
        idx = idx + 1
        # print(dayth, list_weather[dayth], C_real[j_node][idx], P_real[j_node][idx])
        # received_Pijk = P_real[j_node][idx]
        print(dayth, list_weather[dayth], C_real[0][idx], P_real[0][idx])
        received_Pijk = P_real[0][idx]
        # print(dayth, list_weather[dayth], P_plot1[dayth])
        if list_weather[dayth][3]:
            if len(P_holiday) == 0:
                C_est_tmp = -np.log(1 - received_Pijk)
                P_holiday.append((dayth, received_Pijk))
            else:
                # new_prob = alpha * P_holiday[-1][1] + (1 - alpha) * rho_star[dayth]
                tmp_term1 = (received_Pijk - alpha * P_holiday[-1][1])/(1-alpha)
                C_est_tmp = -np.log(1-tmp_term1)
                P_holiday.append((dayth, received_Pijk))
            # P_plot[dayth] = P_holiday[-1][1]
        else:
            if len(P_workday) == 0:
                C_est_tmp = -np.log(1 - received_Pijk)
                P_workday.append((dayth, received_Pijk))
            else:
                # new_prob = alpha * P_workday[-1][1] + (1 - alpha) * rho_star[dayth]
                tmp_term1 = (received_Pijk - alpha * P_workday[-1][1])/(1-alpha)
                C_est_tmp = -np.log(1-tmp_term1)
                P_workday.append((dayth, received_Pijk))
        print(C_est_tmp)
        C_est.append(C_est_tmp)

    # around C [from begin_time to end_time]
    C_final_est = np.zeros(len(C_est))
    for idx in range(len(C_est)):
        C_final_est[idx] = int(np.around(C_est[idx]))
    print(C_final_est)
    # print(C_real[j_node])
    print(C_real[0])

    # MSE
    # MSE_j = np.sum((C_final_est - C_real[j_node])*(C_final_est - C_real[j_node]))
    MSE_j = np.sum((C_final_est - C_real[0])*(C_final_est - C_real[0]))

    print(MSE_j)
    MSE.append(MSE_j)

    if j_node == 66:
        plt.figure(1)
        plt.plot(x_h, C_final_est, color='red', ls="-", marker='o', markersize=5, label='estimation')
        # plt.plot(x_h, C_real[j_node], color='blue', ls="-", marker='*', markersize=5, label='real')
        plt.plot(x_h, C_real[0], color='blue', ls="-", marker='*', markersize=5, label='real')
        plt.show()
print(sum(MSE))

#
# # DODP
# # 对self.P_holiday[-1][1] self.P_workday[-1][1]进行差分隐私扰动
# def DODP_param(LapNoiseScale_Pinter):
#     epsP = 1. / LapNoiseScale_Pinter
#     W = np.exp(epsP)
#     print('set s.t. ...')
#     # 由a c 导出 tranP转成行矩阵对应的位置 c*11+a
#     num_st1 = 0
#     list_st1 = []
#     # s.t.1 隐私保护带来的限制条件
#     for c in range(length_P):
#         # 选取a,b
#         for a in range(length_P):
#             for b in range(a + 1, length_P):
#                 num_st1 = num_st1 + 1
#                 list_st1.append((c, a, b))
#     num_st1 = num_st1 * 2
#     # print(num_st1)
#     num_st2 = 0
#     list_st2 = []
#     # s.t.2 prob的天然条件 [0,1]
#     for a in range(length_P):
#         # 选取a,b
#         for c in range(length_P):
#             num_st2 = num_st2 + 1
#             list_st2.append((c, a))
#     num_st2 = num_st2 * 2
#     # print(num_st2)
#     num_neq_st = num_st1 + num_st2
#     list_st3 = []
#     num_st3 = 0
#     for a in range(length_P):
#         # 选取a,b
#         num_st3 = num_st3 + 1
#         list_st3.append(a)
#     # print(num_st3)
#     # 不等式条件，矩阵
#     label = 0
#     Gnp = np.zeros((num_neq_st, length_P * length_P))
#     hnp = np.zeros((num_neq_st, 1))
#     for ele in list_st1:
#         (c, a, b) = ele
#         # p_{a,c} - W p_{b,c} <=0
#         Gnp[label, a * length_P + c] = 1.
#         Gnp[label, b * length_P + c] = -W
#         label = label + 1
#         # -W p_{a,c} + p_{a,c} <=0
#         Gnp[label, a * length_P + c] = -W
#         Gnp[label, b * length_P + c] = 1.
#         label = label + 1
#     for ele in list_st2:
#         (c, a) = ele
#         # -p_{a,c} <= 0
#         Gnp[label, a * length_P + c] = -1.
#         label = label + 1
#         # p_{a, c} <= 1
#         Gnp[label, a * length_P + c] = 1.
#         hnp[label, 0] = 1.
#         label = label + 1
#     # print(label)
#     # 等式条件，矩阵
#     Anp = np.zeros((num_st3, length_P * length_P))
#     bnp = np.ones((num_st3, 1))
#     label = 0
#     for ele in list_st3:
#         (a) = ele
#         Anp[label, a * length_P:(a + 1) * length_P] = 1.
#         label = label + 1
#     # print(label)
#     Gcx = matrix(Gnp)
#     hcx = matrix(hnp)
#     Acx = matrix(Anp)
#     bcx = matrix(bnp)
#     return Gcx,hcx,Acx,bcx
#
# def optProb(PrioProb):
#     LapNoiseScale_Pinter = 0.2
#     Gcx, hcx, Acx, bcx = DODP_param(LapNoiseScale_Pinter)
#     # Pr(A) * p_{a,c} * (从a变成c偏差)
#     cnp = np.zeros((length_P * length_P, 1))
#     label = 0
#     def getCost(c, a):
#         return (delta_P * (c+0.5) - delta_P * (a+0.5)) ** 2
#     for a in range(length_P):
#         for c in range(length_P):
#             cnp[label, 0] = PrioProb[a] * getCost(c, a)
#             label = label + 1
#
#     ccx = matrix(cnp)
#     print('cal ... opt Prob')
#     sol = solvers.lp(ccx, Gcx, hcx, Acx, bcx)
#     xSolution = np.array(sol['x'])
#     tranP = xSolution.reshape((length_P, length_P))
#     return tranP
#
# # 扰动后的结果放在self.all_P_holiday[self.node_id] = tmp 和 self.all_P_workday[self.node_id]
# # 1.整理获取先验概率
# delta_P = 0.05
#
# length_P = np.math.floor(1. / delta_P)
# ProbValueBucket = np.zeros(length_P)
# for dayth in range(begin_time - 1, end_time):
#
#     print(dayth, list_weather[dayth], C_plot1[dayth], P_plot1[dayth])
#     for i in range(num_of_nodes):
#         loc = np.math.floor(P_plot1[dayth] / delta_P)
#         if loc == length_P:
#             loc = loc - 1
#         ProbValueBucket[loc] = ProbValueBucket[loc] + 1
#     PrioProb = ProbValueBucket / np.sum(ProbValueBucket)
#
#     tranP = optProb(PrioProb)
#     OutputProb = np.zeros(num_of_nodes)
#     once = np.random.random(num_of_nodes)
#     for i in range(num_of_nodes):
#         loc = np.math.floor(P_plot1[dayth] / delta_P)
#         tmp = 0.
#         if loc == length_P:
#             loc = loc - 1
#         for target in range(tranP.shape[1]):
#             if once[i] > tmp and once[i] < tmp + tranP[loc, target]:
#                 break
#             tmp = tmp + tranP[loc, target]
#         OutputProb[i] = (target + 0.5) * delta_P
#         # err = err + (OutputProb[i] - P_plot1[dayth]) ** 2
#
#     if isholiday:
#         self.all_P_holiday[self.node_id] = OutputProb.copy()
#     else:
#         self.all_P_workday[self.node_id] = OutputProb.copy()
#
