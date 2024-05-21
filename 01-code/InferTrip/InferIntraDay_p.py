# 只按照周末来分, 没有分出是否节假日
import time
import datetime
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import pandas as pd
from cvxopt import matrix, solvers
import matplotlib

matplotlib.interactive(False)

WeatherInfo = './Pukou_Weather.xlsx'
INFER_TARGET = '../EncoHistData_NJBike/SDPair_NJBike_Data_qiaobei'
# TARGET_ENCOHIST1 = './210_183.csv' is qiaobei '93-66.csv'

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

def __precdit_GMM(dateset, timelen):
    clf = mixture.GaussianMixture(n_components=3, covariance_type='diag')
    clf.fit(dateset)
    tmp_width = 24/timelen
    # XX = np.linspace(0, timelen - 1, timelen).reshape(-1, 1)
    XX = np.linspace(tmp_width/2, 24-(tmp_width/2), timelen).reshape(-1, 1)
    Z = clf.score_samples(XX)
    probs = np.exp(Z)
    sum_probs = probs.sum()
    score = clf.score(XX)
    if np.sum(probs) > np.power(0.1, 10):
        # print('sum probs:{}'.format(np.sum(probs)))
        nm_probs = probs / np.sum(probs)
    else:
        # print('Interal Err! GMM')
        nm_probs = np.zeros(timelen)
    return nm_probs, (clf.weights_, clf.means_, clf.covariances_)

def getRealpdfandDataset(j_node, target_file):
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

    # prepare valization dataset
    valid_dataset_k = []
    for dayth in range(begin_yday - 1, end_yday):
        if list_weather[dayth][3]:
            valid_dataset_k.append([dayth, True, []])
        else:
            valid_dataset_k.append([dayth, False, []])
    for ele in list_enco_hist:
        tm = ele[0]
        runningtime = time.strptime(tm.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        thm = runningtime.tm_hour + (runningtime.tm_min / 60.0)
        if runningtime >= begin_time and runningtime < endend_time:
            idx = runningtime.tm_yday - begin_time.tm_yday
            valid_dataset_k[idx][2].append(thm)

    # prepare GMM dataset
    dataset_k = []
    dataset_holiday_tmp = []
    dataset_workday_tmp = []
    for dayth in range(begin_yday - 1, end_yday):
        if list_weather[dayth][3]:
            dataset_k.append([dayth, True, []])
        else:
            dataset_k.append([dayth, False, []])
    for idx in range(len(dataset_k)):
        if dataset_k[idx][1]:
            dataset_holiday_tmp = dataset_holiday_tmp + valid_dataset_k[idx][2].copy()
            dataset_k[idx][2] = dataset_holiday_tmp.copy()
        else:
            dataset_workday_tmp = dataset_workday_tmp + valid_dataset_k[idx][2].copy()
            dataset_k[idx][2] = dataset_workday_tmp.copy()

    pdf_k = []
    for ele in dataset_k:
        if len(ele[2]) >= MIN_SAMPLES_for_GMM:
            data_np_tmp = np.array(ele[2]).reshape(-1, 1)
            probs_h, tunple_b_h = __precdit_GMM(data_np_tmp, NUM_of_SEGs)
        else:
            # print('Internal Err! MIN_SAMPLES_for_GMM')
            probs_h = np.zeros(NUM_of_SEGs)
        # print(ele[0], ele[1], probs_h)
        pdf_k.append(probs_h)
    return pdf_k, dataset_k, valid_dataset_k

NUM_DAYS_INYEAR = 365
NUM_of_NODES = 99
NUM_of_SEGs = 6
begin_time = time.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S")
begin_yday = time.strptime('2017/6/1 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
end_time = time.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S")
end_yday = time.strptime('2017/7/31 0:0:00', "%Y/%m/%d %H:%M:%S").tm_yday
endend_time = time.strptime('2017/8/1 0:0:00', "%Y/%m/%d %H:%M:%S")
MIN_SAMPLES_for_GMM = 8
list_weather = readWeather()
Weather_real = list_weather[begin_yday-1 : end_yday]

# calculate real {p} from dataset
print('#'*50)
print('Calculate Real {p} from dataset D')
print(begin_time, end_time)
i_node = 93
all_pdf = []
all_Dataset = []
all_Dataset_valid = []
for j_node in range(NUM_of_NODES):
# for j_node in [66]:
    print('process get {{p}}, node {}'.format(j_node))
    if j_node == i_node:
        all_pdf.append([])
        all_Dataset.append([])
        all_Dataset_valid.append([])
        continue
    filename = '93_{}.csv'.format(j_node)
    filename = os.path.join(INFER_TARGET, filename)
    Pdf_k, Dataset_k, valid_Dataset_k = getRealpdfandDataset(j_node, filename)
    all_pdf.append(Pdf_k)
    all_Dataset.append(Dataset_k)
    all_Dataset_valid.append(valid_Dataset_k)
    y_h = np.zeros(len(Pdf_k))
    i = 0
    for ele in Pdf_k:
        y_h[i] = sum(ele)
        i = i+1
    print(y_h)
x_h = np.linspace(begin_yday, end_yday, end_yday-begin_yday+1)

NUM_of_GUESS = 50

def guesstrip1stday(received_pijk, num_trip, valid_dataset):
    if num_trip < MIN_SAMPLES_for_GMM:
        # print('Internal Err! guess trip time, MIN_SAMPLES_for_GMM')
        once_seqtime = np.random.random(num_trip) * 24
        res_time = once_seqtime.reshape(-1).tolist()
        res_time.sort()
        mse = np.sum((np.array(valid_dataset) - np.array(res_time)) ** 2)
        return res_time, 0., mse
    res_err = np.Inf
    res_time = np.ones(num_trip) * -1
    for onces in range(NUM_of_GUESS):
        once_seqtime = np.random.random(num_trip).reshape(-1,1)*24
        probs_h, tunple_b_h = __precdit_GMM(once_seqtime, NUM_of_SEGs)
        err = np.sum((probs_h - received_pijk) ** 2)
        # print('tmp mse: {}'.format(mse))
        if err < res_err:
            res_err = err
            res_time = once_seqtime.reshape(-1).tolist()
    res_time.sort()
    mse = np.sum((np.array(valid_dataset) - np.array(res_time)) ** 2)
    return res_time, res_err, mse

def guesstrip(old_timeseq, received_pijk, num_trip, valid_dataset):
    if len(old_timeseq) + num_trip < MIN_SAMPLES_for_GMM:
        # print('Internal Err! guess trip time, MIN_SAMPLES_for_GMM')
        once_seqtime = np.random.random(num_trip) * 24
        new_time = once_seqtime.reshape(-1).tolist()
        new_time.sort()
        res_time = old_timeseq.copy() + new_time
        mse = np.sum((np.array(valid_dataset) - np.array(new_time)) ** 2)
        return res_time, 0., new_time, mse
    res_err = np.Inf
    res_time = np.ones(num_trip) * -1
    new_time = np.ones(num_trip) * -1
    for onces in range(NUM_of_GUESS):
        once_seqtime = np.random.random(num_trip).reshape(-1,1) * 24
        data_np_tmp = np.array(old_timeseq).reshape(-1, 1)
        combine_data = np.vstack((data_np_tmp, once_seqtime))
        probs_h, tunple_b_h = __precdit_GMM(combine_data, NUM_of_SEGs)
        err = np.sum((probs_h - received_pijk) ** 2)
        # print('tmp mse: {}'.format(mse))
        if err < res_err:
            res_err = err
            new_time = once_seqtime.reshape(-1).tolist()
    new_time.sort()
    res_time = old_timeseq.copy() + new_time
    mse = np.sum((np.array(valid_dataset)-np.array(new_time))**2)
    return res_time, res_err, new_time, mse


# inference attack without DP
print('#'*50)
print('Inference Attack to Real {{p}}')
print(begin_time, end_time)

total_num_trips = np.zeros((NUM_of_NODES, 2))
total_MSE_real = np.zeros(NUM_of_NODES)
for j_node in range(NUM_of_NODES):
# for j_node in [66]:
    print('process infer trip time, node {}'.format(j_node))
    if j_node == i_node:
        continue

    total_MSE_real_idx = []
    time_holiday_k = []
    time_workday_k = []
    time_est = []
    last_num_trip_holiday = -1
    last_num_trip_workday = -1
    # for dayth in range(begin_time-1, end_time):
    for idx in range(len(all_pdf[j_node])):
    # for idx in range(1):
        dayth = all_Dataset[j_node][idx][0]
        isholiday = all_Dataset[j_node][idx][1]
        assert isholiday == list_weather[dayth][3]
        once_data_source = all_Dataset[j_node][idx][2]
        once_data_source_valid = all_Dataset_valid[j_node][idx][2]
        received_pijk = all_pdf[j_node][idx]
        print('to infer:', idx, isholiday, once_data_source_valid, received_pijk)
        # print(dayth, list_weather[dayth], P_plot1[dayth])
        if isholiday:
            if len(time_holiday_k) == 0:
                # i.e., Cijk
                tmp_num_trip = len(once_data_source)
                last_num_trip_holiday = tmp_num_trip
                time_holiday_k, err, mse = guesstrip1stday(received_pijk, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, True, mse, time_holiday_k.copy()))
            else:
                tmp_num_trip = len(once_data_source) - last_num_trip_holiday
                last_num_trip_holiday = len(once_data_source)
                time_holiday_k, err, new_time, mse = guesstrip(time_holiday_k, received_pijk, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, True, mse, new_time.copy()))
            # P_plot[dayth] = P_holiday[-1][1]
        else:
            if len(time_workday_k) == 0:
                tmp_num_trip = len(once_data_source)
                last_num_trip_workday = tmp_num_trip
                time_workday_k, err, mse = guesstrip1stday(received_pijk, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, False, mse, time_workday_k.copy()))
            else:
                tmp_num_trip = len(once_data_source) - last_num_trip_workday
                last_num_trip_workday = len(once_data_source)
                time_workday_k, err, new_time, mse = guesstrip(time_workday_k, received_pijk, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, False, mse, new_time.copy()))
        total_MSE_real_idx.append(mse)
        print('guess', idx, time_est[-1])
    total_MSE_real[j_node] = sum(total_MSE_real_idx)
    total_num_trips[j_node, 0] = last_num_trip_holiday
    total_num_trips[j_node, 1] = last_num_trip_workday
print('SUM MSE from inference attacks to real {{p}}: ', sum(total_MSE_real))
print('total count of trips: ', np.sum(total_num_trips))

# DODP
print('#'*50)
print('DODP for {{}}')
print(begin_time, end_time)

def getone(a, i, choose):
    if i == a.size:
        if sum(a) == 1.:
            test_list_pdf.append(a.copy())
        return
    for j in range(choose.size):
        a[i:] = 0
        a[i] = choose[j]
        if sum(a) > 1.:
            continue
        getone(a, i + 1, choose)
    return test_list_pdf

oneset = np.zeros(NUM_of_SEGs)
# 每段的数值可能性 0 0.2 0.4 0.6 0.8 1.//0.1,0.3,0.5,0.7,0.9
choose = np.arange(0.1, 1.1, 0.2)
# 所有可能的pdf组合
test_list_pdf = []
getone(oneset, 0, choose)
NUM_PDF_COMB = len(test_list_pdf)

# 为pdf的opt+DP 做出s.t.
def dodp_param_for_pdf(LapNoiseScale_pintra):
    epsp = 2. / LapNoiseScale_pintra
    w = np.exp(epsp)
    # print('set s.t. ...')
    # 由a c 导出 tranp转成行矩阵对应的位置 c*length+a
    num_st1 = 0
    list_st1 = []
    # s.t.1 隐私保护带来的限制条件
    for c in range(NUM_PDF_COMB):
        # 选取a,b
        for a in range(NUM_PDF_COMB):
            for b in range(a + 1, NUM_PDF_COMB):
                num_st1 = num_st1 + 1
                list_st1.append((c, a, b))
    num_st1 = num_st1 * 2
    # print(num_st1)
    num_st2 = 0
    list_st2 = []
    # s.t.2 prob的天然条件 [0,1]
    for a in range(NUM_PDF_COMB):
        # 选取a,b
        for c in range(NUM_PDF_COMB):
            num_st2 = num_st2 + 1
            list_st2.append((c, a))
    num_st2 = num_st2 * 2
    # print(num_st2)
    num_neq_st = num_st1 + num_st2
    list_st3 = []
    num_st3 = 0
    for a in range(NUM_PDF_COMB):
        # 选取a,b
        num_st3 = num_st3 + 1
        list_st3.append(a)
    # print(num_st3)
    # 不等式条件，矩阵
    label = 0
    Gnp = np.zeros((num_neq_st, NUM_PDF_COMB * NUM_PDF_COMB))
    hnp = np.zeros((num_neq_st, 1))
    for ele in list_st1:
        (c, a, b) = ele
        # p_{a,c} - W p_{b,c} <=0
        Gnp[label, a * NUM_PDF_COMB + c] = 1.
        Gnp[label, b * NUM_PDF_COMB + c] = -w
        label = label + 1
        # -W p_{a,c} + p_{a,c} <=0
        Gnp[label, a * NUM_PDF_COMB + c] = -w
        Gnp[label, b * NUM_PDF_COMB + c] = 1.
        label = label + 1
    for ele in list_st2:
        (c, a) = ele
        # -p_{a,c} <= 0
        Gnp[label, a * NUM_PDF_COMB + c] = -1.
        label = label + 1
        # p_{a, c} <= 1
        Gnp[label, a * NUM_PDF_COMB + c] = 1.
        hnp[label, 0] = 1.
        label = label + 1
    # print(label)
    # 等式条件，矩阵
    Anp = np.zeros((num_st3, NUM_PDF_COMB * NUM_PDF_COMB))
    bnp = np.ones((num_st3, 1))
    label = 0
    for ele in list_st3:
        (a) = ele
        Anp[label, a * NUM_PDF_COMB:(a + 1) * NUM_PDF_COMB] = 1.
        label = label + 1
    # print(label)
    Gcx_pdf = matrix(Gnp)
    hcx_pdf = matrix(hnp)
    Acx_pdf = matrix(Anp)
    bcx_pdf = matrix(bnp)
    return Gcx_pdf, hcx_pdf, Acx_pdf, bcx_pdf

LapNoiseScale_Pinter = 1.2

def opt_pdf(PrioP_pdf):
    Gcx_pdf, hcx_pdf, Acx_pdf, bcx_pdf = dodp_param_for_pdf(LapNoiseScale_Pinter)
    cnp = np.zeros((NUM_PDF_COMB * NUM_PDF_COMB, 1))
    label = 0
    for a in range(NUM_PDF_COMB):
        for c in range(NUM_PDF_COMB):
            l1 = np.array(test_list_pdf[c])
            l2 = np.array(test_list_pdf[a])
            tmp = ((l1 - l2) * (l1 - l2)).sum()
            cnp[label, 0] = PrioP_pdf[a] * tmp
            label = label + 1
    ccx = matrix(cnp)
    # print('cal ... opt pdf')
    solvers.options['show_progress'] = False
    sol = solvers.lp(ccx, Gcx_pdf, hcx_pdf, Acx_pdf, bcx_pdf)
    xSolution = np.array(sol['x'])
    tran = xSolution.reshape((NUM_PDF_COMB, NUM_PDF_COMB))
    return tran

pdfValueBucket = np.zeros(NUM_PDF_COMB)
assert len(all_pdf[0]) == end_yday-begin_yday+1
# pdf_dodp = np.ones((end_yday-begin_yday+1, NUM_of_NODES, NUM_of_SEGs))*-1
pdf_dodp = []
for idx in range(len(all_pdf[0])):
    print('dodp process idx (day):{}'.format(idx))
    isholiday = all_Dataset[0][idx][1]
    now_list = []
    for j_node in range(NUM_of_NODES):
        if j_node == i_node:
            continue
        tmp_pdf = all_pdf[j_node][idx]
        # 1. map to list pdf
        if tmp_pdf.sum() < np.power(0.1, 10):
            nm_probs = np.zeros(NUM_of_SEGs)
        else:
            # impossible pdf err
            tmp_err = NUM_of_SEGs + 0.
            loc = -1
            loc_probs = np.ones(NUM_of_SEGs)
            for once_idx in range(len(test_list_pdf)):
                tmp_once_pdf = np.array(test_list_pdf[once_idx])
                tmp_try_once = ((tmp_once_pdf - tmp_pdf) * (tmp_once_pdf - tmp_pdf)).sum()
                if tmp_try_once < tmp_err:
                    tmp_err = tmp_try_once
                    loc = once_idx
                    loc_probs = np.array(test_list_pdf[once_idx])
            now_list.append((j_node, loc, loc_probs, tmp_pdf))
            pdfValueBucket[loc] = pdfValueBucket[loc] + 1

    # 2. opt: (1)get the prior distribution (2) opt Tran_Prob_Matrix
    if np.sum(pdfValueBucket) < np.power(0.1, 10):
        # if no trip (prior distribution), skip
        continue
    PrioP_pdf = pdfValueBucket / np.sum(pdfValueBucket)
    Tran = opt_pdf(PrioP_pdf)
    # 3.get disturbed pdf
    tmp_lap_probs = np.zeros((NUM_of_NODES, NUM_of_SEGs))
    for ele in now_list:
        (j_id, loc, loc_probs, tmp_pdf) = ele
        once = np.random.random()
        tmp = 0.
        target = -1
        for target in range(Tran.shape[1]):
            if once > tmp and once < tmp + Tran[loc, target]:
                break
            tmp = tmp + Tran[loc, target]
        outprob = np.array(test_list_pdf[target])
        # err_opt = ((outprob - tmp_pdf) * (outprob - tmp_pdf)).sum()
        # print('err_opt:{},{}->{}'.format(err_opt, i, j))
        tmp_lap_probs[j_id, :] = outprob
        # store results
    pdf_dodp.append(tmp_lap_probs.copy())

len(pdf_dodp)

# inference attack with DODP
print('#'*50)
print('Inference Attack to dodp P')
print(begin_time, end_time)

total_num_trips_dodp = np.zeros((NUM_of_NODES, 2))
total_MSE_dodp = np.zeros(NUM_of_NODES)
for j_node in range(NUM_of_NODES):
# for j_node in [66]:
    print('process infer trip time from dodp {{p}}, node {}'.format(j_node))
    if j_node == i_node:
        continue

    time_holiday_k = []
    time_workday_k = []
    last_num_trip_holiday = -1
    last_num_trip_workday = -1
    time_est = []
    total_MSE_dodp_idx = []

    for idx in range(len(pdf_dodp)):
        received_pdf = pdf_dodp[idx][j_node, :]
        once_data_source = all_Dataset[j_node][idx][2]
        once_data_source_valid = all_Dataset_valid[j_node][idx][2]
        dayth = all_Dataset[j_node][idx][0]
        isholiday = all_Dataset[j_node][idx][1]

        print('to infer:', idx, dayth, isholiday, once_data_source_valid, received_pdf)
        # print(dayth, list_weather[dayth], P_plot1[dayth])
        if isholiday:
            if len(time_holiday_k) == 0:
                # i.e., Cijk
                tmp_num_trip = len(once_data_source)
                last_num_trip_holiday = tmp_num_trip
                time_holiday_k, err, mse = guesstrip1stday(received_pdf, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, True, mse, time_holiday_k.copy()))
            else:
                tmp_num_trip = len(once_data_source) - last_num_trip_holiday
                last_num_trip_holiday = len(once_data_source)
                time_holiday_k, err, new_time, mse = guesstrip(time_holiday_k, received_pdf, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, True, mse, new_time.copy()))
            # P_plot[dayth] = P_holiday[-1][1]
        else:
            if len(time_workday_k) == 0:
                tmp_num_trip = len(once_data_source)
                last_num_trip_workday = tmp_num_trip
                time_workday_k, err, mse = guesstrip1stday(received_pdf, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, False, mse, time_workday_k.copy()))
            else:
                tmp_num_trip = len(once_data_source) - last_num_trip_workday
                last_num_trip_workday = len(once_data_source)
                time_workday_k, err, new_time, mse = guesstrip(time_workday_k, received_pdf, tmp_num_trip, once_data_source_valid)
                time_est.append((dayth, False, mse, new_time.copy()))
        total_MSE_dodp_idx.append(mse)
        print('guess', idx, time_est[-1])
    total_MSE_dodp[j_node] = sum(total_MSE_dodp_idx)
    total_num_trips_dodp[j_node, 0] = last_num_trip_holiday
    total_num_trips_dodp[j_node, 1] = last_num_trip_workday
print('SUM MSE from inference attacks to dodp {{p}}:', sum(total_MSE_dodp))
print('total count of trips: ', np.sum(total_num_trips_dodp))
# print(total_MSE_dodp)

print('SUM MSE from inference attacks to real {{p}}: ', sum(total_MSE_real))
print('total count of trips: ', np.sum(total_num_trips))

filename = 'infer_intra_p.csv'
fobj = open(filename, 'a+')
fobj.write('{},{},{},{}\n'.format(LapNoiseScale_Pinter, sum(total_MSE_real), np.sum(total_MSE_dodp), np.sum(total_num_trips)))
fobj.flush()
fobj.close()