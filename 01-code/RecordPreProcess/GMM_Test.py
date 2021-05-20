# 只按照周末来分, 没有分出是否节假日
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

# 学校-地铁站
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/86_6.csv'
# 住宅小区-地铁口
TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/210_183.csv'
# 地铁口-住宅小区
# TARGET_ENCOHIST = '../EncoHistData_NJBike/SDPair_NJBike_Data/183_210.csv'


def plot_by_hour():
    list_hist = []
    dataset_inweekday = []
    dataset_inweekend = []
    list_hour_inweekday = np.zeros(24, dtype='int')
    list_hour_inweekend = np.zeros(24, dtype='int')
    onedayfile_obj = open(TARGET_ENCOHIST, 'r', encoding='utf-8')
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
        hour = tunple[0].tm_hour
        weekday = tunple[0].tm_wday
        if weekday == 5 or weekday == 6:
            list_hour_inweekend[hour] = list_hour_inweekend[hour] + 1
            dataset_inweekday.append(hour+tunple[0].tm_min/60.0)
        else:
            list_hour_inweekday[hour] = list_hour_inweekday[hour] + 1
            dataset_inweekend.append(hour+tunple[0].tm_min/60.0)
    print(list_hour_inweekday)
    print(list_hour_inweekend)
    dataset_inweekday = np.array(dataset_inweekday).reshape(-1,1)
    dataset_inweekend = np.array(dataset_inweekend).reshape(-1,1)
    return dataset_inweekday, dataset_inweekend, list_hour_inweekday, list_hour_inweekend

def plot_by_min():
    list_hist = []
    dataset_inweekday = []
    dataset_inweekend = []
    list_hour_inweekday = np.zeros(24 * 60, dtype='int')
    list_hour_inweekend = np.zeros(24 * 60, dtype='int')
    onedayfile_obj = open(TARGET_ENCOHIST, 'r', encoding='utf-8')
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
        hour = tunple[0].tm_hour
        min = tunple[0].tm_min
        total_min = 60 * hour + min
        weekday = tunple[0].tm_wday
        if weekday == 5 or weekday == 6:
            list_hour_inweekend[total_min] = list_hour_inweekend[total_min] + 1
            dataset_inweekend.append(total_min)
        else:
            list_hour_inweekday[total_min] = list_hour_inweekday[total_min] + 1
            dataset_inweekday.append(total_min)
    print(list_hour_inweekday)
    print(list_hour_inweekend)
    x_h = range(24 * 60)
    _ = plt.plot(x_h, list_hour_inweekend, label="R", color='green', marker='o', markersize=0.5)
    _ = plt.plot(x_h, list_hour_inweekday, label="I", color='blue', marker='o', markersize=0.5)
    plt.show()
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

if __name__=='__main__':
    # GMM计算
    # 读取文件 按day分割
    dateset_inweekday, dataset_inweekend, list_hour_inweekday, list_hour_inweekend = plot_by_hour()

    # dateset_inweekday, dataset_inweekend, list_hour_inweekday, list_hour_inweekend = plot_by_min()
    # weekday处理
    probs_weekday, sum_probs_weekday, score_weekday, params_weekday = precdit_GMM(dateset_inweekday)
    probs_weekend, sum_probs_weekend, score_weekend, params_weekend = precdit_GMM(dataset_inweekend)
    print(score_weekday, score_weekend)

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

    fig = plt.figure(figsize=[10, 6])
    plt.subplot(121)

    _ = plt.plot(x_h, list_hour_inweekday, label="list_hour_inweekday", color='blue', marker='o', ls="-", markersize=5)
    _ = plt.plot(x_h, list_hour_inweekend, label="list_hour_inweekend", color='green', marker='*', ls="-", markersize=5)
    plt.title("count")
    plt.subplot(122)
    x_h = np.linspace(0, 24 - 1, 24)
    _ = plt.plot(x_h, probs_weekday, label="probs_weekday", color='blue', marker='^', markersize=5, ls="--")
    _ = plt.plot(x_h, probs_weekend, label="probs_weekend", color='green', marker='s', markersize=5, ls="--")
    plt.title("prob")

    x_h = np.linspace(0, 24-1, 24)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    _ = ax1.plot(x_h, list_hour_inweekday, label="list_hour_inweekday", color='blue', marker='o', ls="-", markersize=5)
    _ = ax2.plot(x_h, probs_weekday, label="probs_weekday", color='red', marker='^', markersize=5, ls="--")
    ax1.set_xlabel('hour')
    ax1.set_ylabel('Count', color='b')
    ax2.set_ylabel('Prob', color='r')
    plt.tight_layout()
    plt.title("weekday")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    _ = ax1.plot(x_h, list_hour_inweekend, label="list_hour_inweekend", color='blue', marker='*', ls="-", markersize=5)
    _ = ax2.plot(x_h, probs_weekend, label="probs_weekend", color='red', marker='s', markersize=5, ls="--")
    ax1.set_xlabel('hour')
    ax1.set_ylabel('Count', color='b')
    ax2.set_ylabel('Prob', color='r')
    plt.tight_layout()
    plt.title("weekend")
    plt.show()

    # _ = plt.plot(x_h, Z, label="R", color='black', marker='o', markersize=0.5)
    # plt.show()