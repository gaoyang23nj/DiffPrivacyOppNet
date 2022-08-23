# 只按照周末来分, 没有分出是否节假日
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd

WeatherInfo = './Pukou_Weather.xlsx'

# 学校-地铁站
TARGET_ENCOHIST3 = './86_6.csv'
# 住宅小区-地铁口
TARGET_ENCOHIST1 = './210_183.csv'
# 地铁口-住宅小区
TARGET_ENCOHIST2 = './183_210.csv'

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

def plot_by_hour(target_file):
    list_hist = []
    dataset_inweekday = []
    dataset_inweekend = []
    list_hour_inweekday = np.zeros(24, dtype='int')
    list_hour_inweekend = np.zeros(24, dtype='int')
    onedayfile_obj = open(target_file, 'r', encoding='utf-8')
    while True:
        line = onedayfile_obj.readline()
        if not line:
            break
        else:
            units = line.split(',')
            tm = time.strptime(units[1], "%Y/%m/%d %H:%M:%S")
            list_hist.append((tm, units[2], units[5]))
    onedayfile_obj.close()
    list_hist.sort()
    print('num: {}'.format(len(list_hist)))
    for tunple in list_hist:
        tm = tunple[0]
        hour = tm.tm_hour
        min = tm.tm_min
        if list_weather[tm.tm_yday-1][3]:
            list_hour_inweekend[hour] = list_hour_inweekend[hour] + 1
            dataset_inweekday.append(hour+min/60.0)
        else:
            list_hour_inweekday[hour] = list_hour_inweekday[hour] + 1
            dataset_inweekend.append(hour+min/60.0)
    print(list_hour_inweekday)
    print(list_hour_inweekend)
    dataset_inweekday = np.array(dataset_inweekday).reshape(-1,1)
    dataset_inweekend = np.array(dataset_inweekend).reshape(-1,1)
    return dataset_inweekday, dataset_inweekend, list_hour_inweekday, list_hour_inweekend

def precdit_GMM(dateset):
    clf = mixture.GaussianMixture(n_components=3, covariance_type='diag')
    clf.fit(dateset)
    XX = np.linspace(0, 24 - 1, 24).reshape(-1, 1)
    Z = clf.score_samples(XX)
    probs = np.exp(Z)
    sum_probs = probs.sum()
    score = clf.score(XX)
    # print(Z)
    # print(probs)
    # print(sum_probs)
    return probs, sum_probs, score, (clf.weights_, clf.means_, clf.covariances_)

# GMM算法 正向计算的实质
def comp_GMM(x, weights_, means_, vars_):
    w_len = len(weights_)
    r = 0.
    for i in range(w_len):
        print(weights_[i], means_[i], vars_[i])
        f_i = (math.exp(-((x-means_[i])**2)/(2*vars_[i])) / math.sqrt(2*math.pi*vars_[i]))*weights_[i]
        r = r + f_i
    return r

list_weather = readWeather()
# GMM计算
# 读取文件 按day分割
dateset_inweekday, dataset_inweekend, list_hour_inweekday, list_hour_inweekend = plot_by_hour(TARGET_ENCOHIST1)
probs_weekday, sum_probs_weekday, score_weekday, params_weekday = precdit_GMM(dateset_inweekday)
probs_weekend, sum_probs_weekend, score_weekend, params_weekend = precdit_GMM(dataset_inweekend)
print(score_weekday, score_weekend)

dateset_inweekday2, dataset_inweekend2, list_hour_inweekday2, list_hour_inweekend2 = plot_by_hour(TARGET_ENCOHIST2)
probs_weekday2, sum_probs_weekday2, score_weekday2, params_weekday2 = precdit_GMM(dateset_inweekday2)
probs_weekend2, sum_probs_weekend2, score_weekend2, params_weekend2 = precdit_GMM(dataset_inweekend2)
print(score_weekday2, score_weekend2)

dateset_inweekday3, dataset_inweekend3, list_hour_inweekday3, list_hour_inweekend3 = plot_by_hour(TARGET_ENCOHIST3)
probs_weekday3, sum_probs_weekday3, score_weekday3, params_weekday3 = precdit_GMM(dateset_inweekday)
probs_weekend3, sum_probs_weekend3, score_weekend3, params_weekend3 = precdit_GMM(dataset_inweekend)
print(score_weekday3, score_weekend3)

x_h = np.linspace(0, 24 - 1, 24)
tmp_res = np.zeros((24))
weights_, means_, vars_ = params_weekday
print('weight', weights_)
print('mean', means_)
print('var', vars_)
for i in range(len(x_h)):
    tmp_res[i] = comp_GMM(x_h[i], weights_, means_, vars_)
print(tmp_res)
print(probs_weekday)

# fig = plt.figure(figsize=[10, 6])
# plt.subplot(121)
#
# _ = plt.plot(x_h, list_hour_inweekday, label="list_hour_inweekday", color='blue', marker='o', ls="-", markersize=5)
# _ = plt.plot(x_h, list_hour_inweekend, label="list_hour_inweekend", color='green', marker='*', ls="-", markersize=5)
# plt.title("count")
# plt.subplot(122)
# x_h = np.linspace(0, 24 - 1, 24)
# _ = plt.plot(x_h, probs_weekday, label="probs_weekday", color='blue', marker='^', markersize=5, ls="--")
# _ = plt.plot(x_h, probs_weekend, label="probs_weekend", color='green', marker='s', markersize=5, ls="--")
# plt.title("prob")


x_h = np.linspace(0, 24-1, 24)

# 写入文件intra_day
file_object = open('intra_day.csv', 'w+', encoding="utf-8")

for i in range(len(x_h)):
    file_object.write('{}'.format(x_h[i]))
    if i != len(x_h)-1:
        file_object.write(',')
file_object.write('\r')

# C_plot1
C1 = list_hour_inweekday
for i in range(len(C1)):
    file_object.write('{}'.format(C1[i]))
    if i != len(C1)-1:
        file_object.write(',')
file_object.write('\r')

# C_plot2
C2 = list_hour_inweekday2
for i in range(len(C2)):
    file_object.write('{}'.format(C2[i]))
    if i != len(C2)-1:
        file_object.write(',')
file_object.write('\r')

P1 = probs_weekday
for i in range(len(P1)):
    file_object.write('{}'.format(P1[i]))
    if i != len(P1)-1:
        file_object.write(',')
file_object.write('\r')

P2 = probs_weekday2
for i in range(len(P2)):
    file_object.write('{}'.format(P2[i]))
    if i != len(P2)-1:
        file_object.write(',')
file_object.write('\r')

file_object.close()

num_data = len(dateset_inweekday)
x = np.zeros((num_data))
plt.rc('font',family='Times New Roman')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
A1 = ax1.plot(x_h, list_hour_inweekday, color='blue', marker='o', ls="-", markersize=5, label="no. # trip in $210-183$")
A2 = ax1.plot(x_h, list_hour_inweekday2, color='blue', marker='^', ls="--", markersize=5, label="no. # trips in $183-210$")
# A1 = ax1.scatter(dateset_inweekday, x, marker='.',color='green',label='time of contact')
B1 = ax2.plot(x_h, probs_weekday, color='red', marker='o', ls="-", markersize=5, label="$\{p_{i,j}\}$ in $210-183$")
B2 = ax2.plot(x_h, probs_weekday2, color='red', marker='^', ls="--", markersize=5, label="$\{p_{i,j}\}$ in $183-210$")
lns = A1+A2+B1+B2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0, fontsize=10)
ax1.set_xlabel('hour', fontsize=15)
ax1.set_ylabel('No. of trips in an hour', color='b', fontsize=15)
ax2.set_ylabel('Prob. density', color='r', fontsize=15)
plt.tight_layout()
plt.show()

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# _ = ax1.plot(x_h, list_hour_inweekend, label="list_hour_inweekend", color='blue', marker='*', ls="-", markersize=5)
# _ = ax2.plot(x_h, probs_weekend, label="probs_weekend", color='red', marker='s', markersize=5, ls="--")
# ax1.set_xlabel('hour')
# ax1.set_ylabel('Count', color='b')
# ax2.set_ylabel('Prob', color='r')
# plt.tight_layout()
# plt.title("weekend")
# plt.show()

