import configparser
import datetime
import socket
import sys
import os
os.environ["OMP_NUM_THREADS"] = '1'
import threading
import numpy as np
import math
# import keyboard
import time
import schedule
from cvxopt import matrix, solvers
from sklearn import mixture

NODE_ID = -1
NODE_IP = '0.0.0.0'
NUM_OF_NODES = 1
NODE_P2P_IP = '0.0.0.0'
MAX_TTL_DAY = 7
Fd_MessageList = []
PpStr = b''

Param_num_pdf_seg = 6
Param_for_log = 0.5
Param_alpha = 0.7
Param_GMM_Components = 3
Param_MIN_SAMPLES = 8

CountTripInDay = np.zeros(NUM_OF_NODES)
Dataset_holiday = []
Dataset_workday = []

IsHoliday = False
IsSetHoliday = False
IsSetWorkday = False

P_holiday = np.zeros(NUM_OF_NODES)
P_workday = np.zeros(NUM_OF_NODES)
Pdf_holiday = np.zeros((NUM_OF_NODES, Param_num_pdf_seg))
Pdf_workday = np.zeros((NUM_OF_NODES, Param_num_pdf_seg))

# received P and {p}
# received from different node, pdf_holiday [num_nodes, Segs]*num_nodes 和 pdf_workday [num_nodes, Segs]*num_nodes
all_pdf_holiday = [np.zeros((NUM_OF_NODES, Param_num_pdf_seg))] * NUM_OF_NODES
all_pdf_workday = [np.zeros((NUM_OF_NODES, Param_num_pdf_seg))] * NUM_OF_NODES
# received from different node, P_holiday [num_nodes]*num_nodes 和 P_workday [num_nodes]*num_nodes
all_P_holiday = [np.zeros(NUM_OF_NODES)] * NUM_OF_NODES
all_P_workday = [np.zeros(NUM_OF_NODES)] * NUM_OF_NODES

MIN_TIME = datetime.datetime.now()
all_UpdatingTime = [MIN_TIME] * NUM_OF_NODES

# prepare seq
all_res_cal = [np.zeros((NUM_OF_NODES, (MAX_TTL_DAY + 1) * Param_num_pdf_seg))] * NUM_OF_NODES

# for DODP
# DODP P
Param_delta_P = 0.05
Param_length_P = np.math.floor(1. / Param_delta_P)
LapNoiseScale_Pinter = 0.2
PValueBucket = np.zeros(Param_length_P)
# DODP {p}
oneset = np.zeros(Param_num_pdf_seg)
# 每段的数值可能性 0 0.2 0.4 0.6 0.8 1.//0.1,0.3,0.5,0.7,0.9
choose = np.arange(0.1, 1.1, 0.2)
# 所有可能的pdf组合
def getone(a, i, choose):
    if i == a.size:
        if sum(a) == 1.:
            Param_list_pdf.append(a.copy())
        return
    for j in range(choose.size):
        a[i:] = 0
        a[i] = choose[j]
        if sum(a) > 1.:
            continue
        getone(a, i + 1, choose)
    return Param_list_pdf
Param_list_pdf = []
getone(oneset, 0, choose)
Param_num_pdf_comb = len(Param_list_pdf)
LapNoiseScale_Pintra = 0.2
pdfValueBucket = np.zeros(Param_num_pdf_comb)

def isholiday(current_time):
    if not isinstance(current_time, datetime.datetime):
        print('Internal Err! Error Type of time')
        return
    res = False
    # Saterday or Sunday will not be workday
    if current_time.isoweekday()==6 or current_time.isoweekday()==7:
        res = True
    yday_today = time.strptime(current_time.strftime('%Y/%m/%d'), '%Y/%m/%d').tm_yday
    list_specialday = []
    with open('holidaylist.txt') as fp:
        lines = fp.readlines()
        for line in lines:
            tunple = line.split(',')
            dayth = time.strptime(tunple[0].strip(), "%Y/%m/%d").tm_yday
            label = True
            if tunple[1].strip() == 'False':
                label = False
            list_specialday.append((dayth, label))
            if dayth == yday_today:
                res = label
                break
    return res

def warmup_by_trips():
    filename = 'trips_warm_node{}.csv'.format(NODE_ID)
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                tunple_list = line.split(',')
                i = int(tunple_list[0].strip())
                j = int(tunple_list[1].strip())
                triptime = time.strptime(tunple_list[2].strip(), "%Y/%m/%d %H:%M:%S")
                assert i >= 0 and i < NUM_OF_NODES and j >= 0 and j < NUM_OF_NODES
                CountTripInDay[j] = CountTripInDay[j] + 1
                timevalue = triptime.tm_hour + (triptime.tm_min/60.0)
                if IsHoliday:
                    Dataset_holiday[j].append(timevalue)
                else:
                    Dataset_workday[j].append(timevalue)

def init():
    global NODE_ID, NODE_IP, NODE_P2P_IP, NUM_OF_NODES, MAX_TTL_DAY
    cf = configparser.ConfigParser()
    cf.read("node_config.ini")
    NODE_ID = int(cf.get('node', 'node_id'))
    NODE_IP = cf.get('node', 'node_ip')
    NUM_OF_NODES = int(cf.get('node', 'num_of_nodes'))

    MAX_TTL_DAY = int(cf.get('node', 'max_ttl'))

    NODE_P2P_IP = cf.get('bike2node_comm', 'node_p2p_ip')
    BIKE_P2P_IP = cf.get('bike2node_comm', 'bike_p2p_ip')

    host = socket.gethostname()
    ip_list = socket.gethostbyname_ex(host)[2]
    assert NODE_IP == ip_list[-2]
    assert NODE_P2P_IP == ip_list[-1]

    # pdf划分的间隔数
    # self.num_seg = theconfig.num_pdf_seg
    # Param_num_seg = 24
    global Param_num_pdf_seg, Param_for_log, Param_alpha, Param_GMM_Components, Param_MIN_SAMPLES
    Param_num_pdf_seg = 6
    # log的底数
    Param_for_log = 0.5
    Param_alpha = 0.7
    Param_GMM_Components = 3
    Param_MIN_SAMPLES = 8

    global IsHoliday, CountTripInDay, Dataset_holiday, Dataset_workday, P_holiday, P_workday, Pdf_holiday, Pdf_workday, IsSetHoliday, IsSetWorkday
    IsSetHoliday = False
    IsSetWorkday = False
    current_time = datetime.datetime.now()
    IsHoliday = isholiday(current_time)

    CountTripInDay = np.zeros(NUM_OF_NODES)
    for i in range(NUM_OF_NODES):
        Dataset_holiday.append([])
        Dataset_workday.append([])
    # Real P and {p}, (1) update each day & disturb (2) use in contact, calculate G
    P_holiday = np.ones(NUM_OF_NODES) * 0
    P_workday = np.ones(NUM_OF_NODES) * 0

    Pdf_holiday = np.zeros((NUM_OF_NODES, Param_num_pdf_seg))
    Pdf_workday = np.zeros((NUM_OF_NODES, Param_num_pdf_seg))

    global all_pdf_holiday, all_pdf_workday, all_P_holiday, all_P_workday
    # received P and {p}
    # received from different node, pdf_holiday [num_nodes, Segs]*num_nodes 和 pdf_workday [num_nodes, Segs]*num_nodes
    all_pdf_holiday = [np.zeros((NUM_OF_NODES, Param_num_pdf_seg))] * NUM_OF_NODES
    all_pdf_workday = [np.zeros((NUM_OF_NODES, Param_num_pdf_seg))] * NUM_OF_NODES
    # received from different node, P_holiday [num_nodes]*num_nodes 和 P_workday [num_nodes]*num_nodes
    all_P_holiday = [np.zeros(NUM_OF_NODES)] * NUM_OF_NODES
    all_P_workday = [np.zeros(NUM_OF_NODES)] * NUM_OF_NODES

    global all_res_cal
    all_res_cal = [np.zeros((NUM_OF_NODES, (MAX_TTL_DAY + 1) * Param_num_pdf_seg))] * NUM_OF_NODES

    check_messages_in_node()

    # for warmup P and p[should del in the applications]
    warmup_by_trips()
    # updating time
    global all_UpdatingTime
    MIN_TIME = current_time
    all_UpdatingTime = [MIN_TIME] * NUM_OF_NODES
    notify_new_day()

    print('Init success')

def __cal_delivery_metric_G(trip_time_s, deadline_s, pktdst_id):
    trip_time = datetime.datetime.strptime(trip_time_s, '%Y/%m/%d %H:%M:%S')
    deadline = datetime.datetime.strptime(deadline_s, '%Y/%m/%d %H:%M:%S')
    # tmp_end = gentime + self.max_ttl
    # # in simulation, endtime
    # if self.MAX_TIME < tmp_end:
    #     tmp_end = self.MAX_TIME
    delta_time = deadline - trip_time
    # each seg in 'seq' means nr_hour hours (4 hours).
    n_hour = 24./Param_num_pdf_seg
    nr_hours = math.ceil(delta_time.total_seconds()/(3600*n_hour))
    curr_tmhour = math.floor((trip_time.hour+0.)/n_hour)

    # 1.init weights in Graph (N*N)
    matrix = np.ones((NUM_OF_NODES, NUM_OF_NODES)) * np.inf
    for o_id in range(NUM_OF_NODES):
        for d_id in range(NUM_OF_NODES):
            if o_id == d_id:
                matrix[o_id, d_id] = 0
                continue
            tmp = np.sum(all_res_cal[o_id][d_id, curr_tmhour:curr_tmhour+nr_hours])
            if tmp > 1.:
                # print('\033[1;35;46m DJK Ensure weight will be positive \033[0m')
                tmp = 1.
            # 0 < Param_for_log < 1,
            # if tmp is samll (probability is small), matrix will be big (distance will be big)
            matrix[o_id, d_id] = np.log(tmp)/np.log(Param_for_log)
    # 2.Djk Alg., min distance (max prob.) in Graph (N*N)
    # record distance
    dis = np.ones((NUM_OF_NODES)) * np.inf
    # record previous node
    pre = np.ones((NUM_OF_NODES), dtype='int') * -1
    # record whether a node has been visited
    vis = np.ones((NUM_OF_NODES), dtype='int') * 0
    # 2.1 init
    for i in range(NUM_OF_NODES):
        dis[i] = matrix[NODE_ID, i]
        pre[i] = NODE_ID
    vis[NODE_ID] = 1
    count = 1
    while count != NUM_OF_NODES:
        # find the node has the min distance, 'min' record distance, 'tmp_idx' record node's id
        tmp_idx = -1
        min = np.inf
        for i in range(NUM_OF_NODES):
            # choose a node, which has not been visited and has the possible distance.
            if vis[i] != 1 and dis[i] < min:
                min = dis[i]
                tmp_idx = i
        vis[tmp_idx] = 1
        count = count + 1
        for i in range(NUM_OF_NODES):
            if vis[i] != 1 and matrix[tmp_idx,i] != np.inf and dis[tmp_idx] + matrix[tmp_idx][i] < dis[i]:
                dis[i] = dis[tmp_idx] + matrix[tmp_idx][i]
                pre[i] = tmp_idx
    # calculate G value
    dis_value = dis[pktdst_id]
    # np.power(Param_for_log, dis_value)
    G_value = np.exp(dis_value * np.log(Param_for_log))
    # seek the (reverse) route related to G_value
    path = [pktdst_id]
    cur = pktdst_id
    while cur != NODE_ID:
        # path is not connected
        if pre[cur] == -1:
            path = []
            break
        path.insert(0, pre[cur])
        cur = pre[cur]
    return G_value, path

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


# DODP
# 对self.P_holiday[-1][1] self.P_workday[-1][1]进行差分隐私扰动
def DODP_param_for_P(LapNoiseScale_Pinter, length_P):
    epsP = 1. / LapNoiseScale_Pinter
    W = np.exp(epsP)
    # print('set s.t. ...')
    # 由a c 导出 tranP转成行矩阵对应的位置 c*11+a
    num_st1 = 0
    list_st1 = []
    # s.t.1 隐私保护带来的限制条件
    for c in range(length_P):
        # 选取a,b
        for a in range(length_P):
            for b in range(a + 1, length_P):
                num_st1 = num_st1 + 1
                list_st1.append((c, a, b))
    num_st1 = num_st1 * 2
    # print(num_st1)
    num_st2 = 0
    list_st2 = []
    # s.t.2 prob的天然条件 [0,1]
    for a in range(length_P):
        # 选取a,b
        for c in range(length_P):
            num_st2 = num_st2 + 1
            list_st2.append((c, a))
    num_st2 = num_st2 * 2
    # print(num_st2)
    num_neq_st = num_st1 + num_st2
    list_st3 = []
    num_st3 = 0
    for a in range(length_P):
        # 选取a,b
        num_st3 = num_st3 + 1
        list_st3.append(a)
    # print(num_st3)
    # 不等式条件，矩阵
    label = 0
    Gnp = np.zeros((num_neq_st, length_P * length_P))
    hnp = np.zeros((num_neq_st, 1))
    for ele in list_st1:
        (c, a, b) = ele
        # p_{a,c} - W p_{b,c} <=0
        Gnp[label, a * length_P + c] = 1.
        Gnp[label, b * length_P + c] = -W
        label = label + 1
        # -W p_{a,c} + p_{a,c} <=0
        Gnp[label, a * length_P + c] = -W
        Gnp[label, b * length_P + c] = 1.
        label = label + 1
    for ele in list_st2:
        (c, a) = ele
        # -p_{a,c} <= 0
        Gnp[label, a * length_P + c] = -1.
        label = label + 1
        # p_{a, c} <= 1
        Gnp[label, a * length_P + c] = 1.
        hnp[label, 0] = 1.
        label = label + 1
    # print(label)
    # 等式条件，矩阵
    Anp = np.zeros((num_st3, length_P * length_P))
    bnp = np.ones((num_st3, 1))
    label = 0
    for ele in list_st3:
        (a) = ele
        Anp[label, a * length_P:(a + 1) * length_P] = 1.
        label = label + 1
    # print(label)
    Gcx = matrix(Gnp)
    hcx = matrix(hnp)
    Acx = matrix(Anp)
    bcx = matrix(bnp)
    return Gcx,hcx,Acx,bcx

def opt_P(PrioProb, LapNoiseScale_Pinter, length_P, delta_P):
    Gcx, hcx, Acx, bcx = DODP_param_for_P(LapNoiseScale_Pinter, length_P)
    # Pr(A) * p_{a,c} * (从a变成c偏差)
    cnp = np.zeros((length_P * length_P, 1))
    label = 0
    def getCost(c, a):
        return (delta_P * (c+0.5) - delta_P * (a+0.5)) ** 2
    for a in range(length_P):
        for c in range(length_P):
            cnp[label, 0] = PrioProb[a] * getCost(c, a)
            label = label + 1

    ccx = matrix(cnp)
    # print('cal ... opt Prob')
    solvers.options['show_progress'] = False
    sol = solvers.lp(ccx, Gcx, hcx, Acx, bcx)
    xSolution = np.array(sol['x'])
    tranP = xSolution.reshape((length_P, length_P))
    return tranP



# pdf opt+DP, build s.t.
def DODP_param_for_pdf(LapNoiseScale_pintra, NUM_PDF_COMB):
    epsp = 2. / LapNoiseScale_pintra
    w = np.exp(epsp)
    # print('set s.t. ...')
    num_st1 = 0
    list_st1 = []
    # s.t.1 preserving privacy s.t.
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
    # s.t.2 prob s.t. [0,1]
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
    # inequality matrix
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
    # equality matrix
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


def opt_pdf(PrioP_pdf, LapNoiseScale_Pinter, test_list_pdf, NUM_PDF_COMB):
    Gcx_pdf, hcx_pdf, Acx_pdf, bcx_pdf = DODP_param_for_pdf(LapNoiseScale_Pinter, NUM_PDF_COMB)
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

def __cal_cond_prob(res_list_betwday):
    # number_of_days* number_of_nodes
    res = np.ones((res_list_betwday.shape[0], NUM_OF_NODES))
    for index in range(0, res_list_betwday.shape[0]):
        # 0~index-1 row；first day (0) ~ this day (index)
        for j in range(0, index):
            res[index, :] = res[index, :] * (1. - res_list_betwday[j, :])
        res[index, :] = res[index, :] * res_list_betwday[index, :]
    return res

def notify_new_day():
    global IsHoliday, IsSetHoliday, IsSetWorkday, CountTripInDay, P_holiday, P_workday
    global Dataset_holiday, Dataset_workday, Pdf_holiday, Pdf_workday
    global all_pdf_holiday, all_pdf_workday, all_P_holiday, all_P_workday

    current_time = datetime.datetime.now()
    IsHoliday = isholiday(current_time)
    print('update TTPM', current_time.strftime("%Y-%m-%d %H:%M:%S"))

    # skip some days
    # 1. update P and disturb
    # 1.1 update P
    rho_star = 1. - np.exp(-1*CountTripInDay)
    if IsHoliday:
        if not IsSetHoliday:
            P_holiday = rho_star
        else:
            P_holiday = P_holiday * Param_alpha + (1-Param_alpha)*rho_star
        tmpP = P_holiday.copy()
    else:
        if not IsSetWorkday:
            P_workday = rho_star
        else:
            P_workday = P_workday * Param_alpha + (1-Param_alpha)*rho_star
        tmpP = P_workday.copy()
    if IsHoliday:
        IsSetHoliday = True
    else:
        IsSetWorkday = True
    # 1.2 disturb P
    global LapNoiseScale_Pinter, PValueBucket, all_P_holiday, all_P_workday
    # global Param_length_P, Param_delta_P
    for i in range(NUM_OF_NODES):
        if i == NODE_ID:
            continue
        loc = np.math.floor(tmpP[i] / Param_delta_P)
        if loc == Param_length_P:
            loc = loc - 1
        PValueBucket[loc] = PValueBucket[loc] + 1
    prioProb = PValueBucket / np.sum(PValueBucket)
    tranP = opt_P(prioProb, LapNoiseScale_Pinter, Param_length_P, Param_delta_P)
    disturbed_P = np.zeros(NUM_OF_NODES)
    once = np.random.random(NUM_OF_NODES)
    for i in range(NUM_OF_NODES):
        if i == NODE_ID:
            continue
        loc = np.math.floor(tmpP[i] / Param_delta_P)
        tmp = 0.
        if loc == Param_delta_P:
            loc = loc - 1
        target = 0
        for target in range(tranP.shape[1]):
            if once[i] > tmp and once[i] < tmp + tranP[loc, target]:
                break
            tmp = tmp + tranP[loc, target]
        disturbed_P[i] = (target + 0.5) * Param_delta_P
    if IsHoliday:
        all_P_holiday[NODE_ID] = disturbed_P.copy()
    else:
        all_P_workday[NODE_ID] = disturbed_P.copy()

    # 2. update {p} and disturb {p}
    global LapNoiseScale_Pintra, pdfValueBucket, all_pdf_holiday, all_pdf_workday
    now_list = []
    for tmp_node in range(NUM_OF_NODES):
        if tmp_node == NODE_ID:
            continue
        if IsHoliday:
            probs_h = np.zeros(Param_num_pdf_seg)
            if len(Dataset_holiday[tmp_node]) >= Param_MIN_SAMPLES:
                dataset1 = np.array(Dataset_holiday[tmp_node]).reshape(-1, 1)
                probs_h, tunple_b_h = __precdit_GMM(dataset1, Param_num_pdf_seg)
            Pdf_holiday[tmp_node, :] = probs_h
            tmp_pdf = probs_h
        else:
            probs_w = np.zeros(Param_num_pdf_seg)
            if len(Dataset_workday[tmp_node]) >= Param_MIN_SAMPLES:
                dataset2 = np.array(Dataset_workday[tmp_node]).reshape(-1, 1)
                probs_w, tunple_b_w = __precdit_GMM(dataset2, Param_num_pdf_seg)
            Pdf_workday[tmp_node, :] = probs_w
            tmp_pdf = probs_w

        # all 0 pdf has no combination
        if tmp_pdf.sum() < np.power(0.1, 10):
            nm_probs = np.zeros(Param_num_pdf_seg)
        else:
            tmp_err = Param_num_pdf_seg + 0.
            loc = -1
            loc_probs = np.ones(Param_num_pdf_seg)
            for once_idx in range(len(Param_list_pdf)):
                tmp_once_pdf = np.array(Param_list_pdf[once_idx])
                tmp_try_once = ((tmp_once_pdf - tmp_pdf) * (tmp_once_pdf - tmp_pdf)).sum()
                if tmp_try_once < tmp_err:
                    tmp_err = tmp_try_once
                    loc = once_idx
                    loc_probs = np.array(Param_list_pdf[once_idx])
            now_list.append((tmp_node, loc, loc_probs, tmp_pdf))
            pdfValueBucket[loc] = pdfValueBucket[loc] + 1
    if np.sum(pdfValueBucket) >= np.power(0.1, 10):
        PrioP_pdf = pdfValueBucket / np.sum(pdfValueBucket)
        Tran = opt_pdf(PrioP_pdf, LapNoiseScale_Pintra, Param_list_pdf, Param_num_pdf_comb)
        tmp_disturbed_probs = np.zeros((NUM_OF_NODES, Param_num_pdf_seg))
        for ele in now_list:
            (b_id, loc, loc_probs, tmp_pdf) = ele
            once = np.random.random()
            tmp = 0.
            target = -1
            for target in range(Tran.shape[1]):
                if once > tmp and once < tmp + Tran[loc, target]:
                    break
                tmp = tmp + Tran[loc, target]
            outprob = np.array(Param_list_pdf[target])
            # err_opt = ((outprob - tmp_pdf) * (outprob - tmp_pdf)).sum()
            # print('err_opt:{},{}->{}'.format(err_opt, i, j))
            tmp_disturbed_probs[b_id, :] = outprob
        # keep the disturbed pdf, {p}
        if isholiday:
            all_pdf_holiday[NODE_ID] = tmp_disturbed_probs.copy()
        else:
            all_pdf_workday[NODE_ID] = tmp_disturbed_probs.copy()

    # 3. prepare pdf from t to ttl
    global all_res_cal
    # print('update at {}-{}-{}, in node {}, number_of_nodes {}'.format(
    #     runningtime.tm_year, runningtime.tm_mon, runningtime.tm_mday,
    #     self.node_id, self.num_of_nodes))
    tmp_res_cal = np.zeros((NUM_OF_NODES, (MAX_TTL_DAY + 1) * Param_num_pdf_seg))
    for target_node in range(NUM_OF_NODES):
        # 1. process conditional prob, considering isHoliday, real (local) or disturbed (received)
        res_list_betwday = []
        for i in range(MAX_TTL_DAY + 1):
            new_time = current_time + datetime.timedelta(days=i)
            newday_isholiday = isholiday(new_time)
            if newday_isholiday:
                # [np.array(each node), np.array, np.array, ... ] 一共 ttl个
                if target_node == NODE_ID:
                    res_list_betwday.append(P_holiday.copy())
                else:
                    res_list_betwday.append(all_P_holiday[target_node].copy())
                # 1. itself use real P and {p}, 2. other node use the received disturbed P and {p}
                if target_node == NODE_ID:
                    tmp_res_cal[:, i * Param_num_pdf_seg:(i + 1) * Param_num_pdf_seg] = Pdf_holiday.copy()
                else:
                    tmp_res_cal[:, i * Param_num_pdf_seg:(i + 1) * Param_num_pdf_seg] = all_pdf_holiday[target_node][:, :].copy()
            else:
                if target_node == NODE_ID:
                    res_list_betwday.append(P_workday.copy())
                else:
                    res_list_betwday.append(all_P_workday[target_node].copy())
                # 1. itself use real P and {p}, 2. other node use the received disturbed P and {p}
                if target_node == NODE_ID:
                    tmp_res_cal[:, i * Param_num_pdf_seg:(i + 1) * Param_num_pdf_seg] = Pdf_workday.copy()
                else:
                    tmp_res_cal[:, i * Param_num_pdf_seg:(i + 1) * Param_num_pdf_seg] = all_pdf_workday[target_node][:, :].copy()
        res_list_betwday = np.array(res_list_betwday)
        # ttl.days * num_nodes, which day to which node
        cond_P = __cal_cond_prob(res_list_betwday).transpose()
        tmp = np.repeat(cond_P, Param_num_pdf_seg, axis=1)
        tmp_res_cal = np.multiply(tmp_res_cal, tmp)
        all_res_cal[target_node] = tmp_res_cal.copy()

    # 4. clear C
    CountTripInDay = np.zeros(NUM_OF_NODES)

    # 5. updating time
    all_UpdatingTime[NODE_ID] = current_time

def preparingPpBuf(for_list):
    cols = 2 + Param_num_pdf_seg * 2
    Res_Matrix = np.zeros((NUM_OF_NODES, cols*len(for_list)))
    for idx in range(len(for_list)):
        for_node = for_list[idx]
        # prepare string (auxilary info.) to send to other nodes
        PpStrTransmit = np.zeros((NUM_OF_NODES, 2 + Param_num_pdf_seg * 2))
        PpStrTransmit[:, 0] = all_P_holiday[for_node]
        PpStrTransmit[:, 1] = all_P_workday[for_node]
        PpStrTransmit[:, 2:2 + Param_num_pdf_seg] = all_pdf_holiday[for_node]
        PpStrTransmit[:, 2 + Param_num_pdf_seg:] = all_pdf_workday[for_node]
        Res_Matrix[:, idx*cols:(idx+1)*cols] = PpStrTransmit

    # np.frombuffer(b, dtype=float)
    data = Res_Matrix.tobytes()
    length = len(data)
    buf = length.to_bytes(2, 'big', signed=False) + data
    return buf

def processPpbuf(buf, recv_time_list):
    update_list = []
    length = int.from_bytes(buf[0:2], 'big', signed=False)
    data = np.frombuffer(buf[2:]).reshape(NUM_OF_NODES, -1)
    cols = 2 + Param_num_pdf_seg * 2
    assert cols*NUM_OF_NODES == data.shape[1]
    for i_node in range(NUM_OF_NODES):
        if i_node == NODE_ID:
            continue
        # if the received info is more new, update
        if all_UpdatingTime[i_node] < recv_time_list[i_node][1]:
            node_id = recv_time_list[i_node][0]
            PpStrTransmit = data[:, i_node * cols:(i_node + 1) * cols]
            all_P_holiday[node_id] = PpStrTransmit[:, 0]
            all_P_workday[node_id] = PpStrTransmit[:, 1]
            all_pdf_holiday[node_id] = PpStrTransmit[:, 2:2+Param_num_pdf_seg]
            all_pdf_workday[node_id] = PpStrTransmit[:, 2 + Param_num_pdf_seg:]
            update_list.append(i_node)
    return update_list

# N2N connect as node j (client), trip: i->j
def Node2Node_connect(target_id, target_ip, message_list, trip_time, notify_delete_list, notify_forward_list):
    try:
        #create an AF_INET, STREAM socket (TCP)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()
    print('N2N connect Socket Created')
    # hostname = socket.gethostname()
    # hostip = socket.gethostbyname(hostname)
    port = 12345
    s.connect((target_ip, port))
    # 0.Node2Node response
    s.send('{};{}'.format(NODE_ID, trip_time).encode())
    # socket construct
    tmp_str = s.recv(1024).decode()
    print(tmp_str)
    node_from_id = int(tmp_str.split(' ')[4])
    assert int(tmp_str.split(' ')[7]) == NODE_ID
    # 1. record trip, only in node i, [target_id->NODE_ID, trip_time] []
    # for i->j, i should record trip, j do not need
    # 2. exchange P and {p} ! [socket 172.16.2.2---172.16.2.4]
    # send disturbed P and {p}
    # 2.1 [updating time] * NUM_OF_NODES
    all_UpdatingTime[NODE_ID] = datetime.datetime.now()
    tmp_str = ''
    tmp_l = 20
    for t in all_UpdatingTime:
        tmp_str += t.strftime('%Y/%m/%d %H:%M:%S') + ';'
        tmp_l = len(t.strftime('%Y/%m/%d %H:%M:%S') + ';')
    tmp_str = tmp_str[:-1]
    s.send(tmp_str.encode())
    # 20*NUM_OF_NODES
    tmp_recv_str = s.recv((tmp_l+5)*NUM_OF_NODES+5).decode()
    time_list = tmp_recv_str.split(';')
    assert len(time_list) == NUM_OF_NODES
    # 2.2 find required Pp's idx list
    recv_time_list = []
    for idx in range(len(time_list)):
        once_updatetime = time_list[idx]
        once_updatetime = once_updatetime.strip()
        t = datetime.datetime.strptime(once_updatetime, '%Y/%m/%d %H:%M:%S')
        recv_time_list.append((idx, t))
    # 2.3 Prepare String (Auxilary information)
    send_list = list(range(NUM_OF_NODES))
    buf = preparingPpBuf(send_list)
    print('send auxiliary info.:', send_list)
    # send -> recv
    s.sendall(buf)
    recv_buf = b''
    while True:
        seg_buf = s.recv(1024)
        recv_buf += seg_buf
        if len(seg_buf) < 1024:
            break
    # 2.4 process received P and {p} --- recv_buf
    update_list = processPpbuf(recv_buf, recv_time_list)
    print('receive auxiliary info.:', update_list)

    # 3.possible message list [src, dst, ttl]
    message_str = ''
    for tuple in message_list:
        # dst, src, message_id, deadline
        message_str += '{},{},{},{};'.format(tuple[0], tuple[1], tuple[2], tuple[3])
    num_mg = len(message_list)
    message_str = '{};'.format(num_mg) + message_str
    message_str = message_str[:-1]
    s.send(message_str.encode())
    print('send possible message_list:', message_str)
    tmp_str = s.recv(1024)
    print(tmp_str.decode())

    # 4. 'G of each message carried here', send to node i
    G_send_str = np.ones((len(message_list), NUM_OF_NODES + 2)) * -1
    for idx in range(len(message_list)):
        pktdstid = message_list[idx][0]
        deadline = message_list[idx][3]
        G_value, G_path = __cal_delivery_metric_G(trip_time, deadline, pktdstid)
        G_send_str[idx, 0] = G_value
        for idx_item in range(len(G_path)):
            G_send_str[idx, 1 + idx_item] = G_path[idx_item]
        print('metric from listen side:', G_value, G_path)
    buf = num_mg.to_bytes(4, 'big', signed=False)
    s.sendall(buf + G_send_str.tobytes())

    # 5.notify fwd_G messages ! [socket 192.168.1.8 (Raspberry)---]
    tmp_str = s.recv(2048).decode()
    notify_list = []
    notify_list_str = tmp_str.split(';')
    num_notify_list = int(notify_list_str[0])
    notify_list_str.pop(0)
    assert num_notify_list == len(notify_list_str)
    for line in notify_list_str:
        # if line.strip() == '':
        #     continue
        tunple = line.split(',')
        notify_list.append((int(tunple[0].strip()), int(tunple[1].strip()), tunple[2].strip()))
    print('i notify j list:', notify_list)

    # 6.
    notify_delete_list.extend(notify_list.copy())
    tmp_str = ''
    for tunple in notify_delete_list:
        tmp_str += '{},{},{};'.format(tunple[0], tunple[1], tunple[2])
    num_delete_list = len(notify_delete_list)
    if num_delete_list > 0:
        tmp_str = '{};'.format(num_delete_list) + tmp_str[:-1]
    else:
        tmp_str = '{}'.format(num_delete_list)
    s.send(tmp_str.encode())
    print('j notify i list:', notify_delete_list)

    notify_forward_list.extend(notify_list.copy())
    s.close()
    print('N2N socket close (connect side)')
    return notify_forward_list

# delete messages (node j notify node i)
def processDelFile(notify_delete_list):
    messageList = []
    with open('node_message_list.csv', 'r+') as fp:
        lines = fp.readlines()
        for idx in range(1, len(lines), 1):
            tlist = lines[idx].split(',')
            messageList.append((tlist[0].strip(), tlist[1].strip(), tlist[2].strip(), tlist[3].strip(),
                                tlist[4].strip(), tlist[5].strip(), tlist[6].strip(), idx))
    record_del_loc = []
    lines_label = np.zeros(len(lines), dtype=int)
    for item in notify_delete_list:
        # notify message list (dst, deadline)
        findLable = False
        for message in messageList:
            if item[0] == message[0] and item[1] == message[1] and item[2] == message[2]:
                # loc, filename
                record_del_loc.append((message[-1], message[-2]))
                lines_label[message[-1]] = 1
                findLable = True
                break
        if findLable == False:
            print('Internal Err! notify delete {}-{}-{} not found!'.format(item[0], item[1], item[2]))
    record_del_loc.sort(reverse=True)
    with open('node_message_list.csv', 'w+') as fp:
        for idx in range(len(lines)):
            if lines_label[idx] != 1:
                fp.write(lines[idx])
    for del_loc in record_del_loc:
        os.remove(del_loc[1])

# N2N connect as node i (server), i->j
def Node2Node_listen(node_id, node_ip):
    try:
        # create an AF_INET, STREAM socket (TCP)
        s = socket.socket()
    except socket.error:
        print('Failed to create socket (listen side)')
        sys.exit()
    print('N2N socket created (listen side)')
    # hostname = socket.gethostname()
    # hostip = socket.gethostbyname(hostname)
    port = 12345
    s.bind((node_ip, port))
    # wait a client node (node j) connect with N2N
    s.listen(5)
    while True:
        c, addr = s.accept()
        print('N2N socket accept:{} (listen side)', addr)
        # 0.node id
        tmp_str = c.recv(1024).decode().split(';')
        node_to_id = int(tmp_str[0])
        trip_time = tmp_str[1]
        node_to_ip = addr[0]
        c.send('Bicycle Trip from ({}) {} to ({}) {} at {}'.format(node_ip, node_id, node_to_ip, node_to_id, trip_time).encode())
        # 1.record this trip [node_id->node_to_id, ]
        CountTripInDay[node_to_id] = CountTripInDay[node_to_id] + 1
        # trip_time_s = time.strptime(trip_time,'%Y/%m/%d %H:%M:%S')
        trip_time_d = datetime.datetime.strptime(trip_time, '%Y/%m/%d %H:%M:%S')
        # trip_time_s.tm_hour + trip_time_s.tm_min/60.
        value = trip_time_d.hour + trip_time_d.minute
        if isholiday(trip_time_d):
            Dataset_holiday.append(value)
        else:
            Dataset_workday.append(value)
        # 2.exchange P and {p} ! [socket 172.16.2.2---172.16.2.4]
        # 2.1 updating time
        all_UpdatingTime[NODE_ID] = datetime.datetime.now()
        tmp_str = ''
        tmp_l = 20
        for t in all_UpdatingTime:
            tmp_str += t.strftime('%Y/%m/%d %H:%M:%S') + ';'
            tmp_l = len(t.strftime('%Y/%m/%d %H:%M:%S') + ';')
        tmp_str = tmp_str[:-1]
        # 20*NUM_OF_NODES
        tmp_recv_str = c.recv((tmp_l + 5) * NUM_OF_NODES+5).decode()
        c.send(tmp_str.encode())
        time_list = tmp_recv_str.split(';')
        assert len(time_list) == NUM_OF_NODES
        # 2.2 find required Pp
        recv_time_list = []
        for idx in range(len(time_list)):
            once_updatetime = time_list[idx]
            once_updatetime = once_updatetime.strip()
            t = datetime.datetime.strptime(once_updatetime, '%Y/%m/%d %H:%M:%S')
            recv_time_list.append((idx, t))
        # 2.3 Prepare String (Auxilary information)
        send_list = list(range(NUM_OF_NODES))
        buf = preparingPpBuf(send_list)
        print('send auxiliary info.:', send_list)
        # recv -> send
        recv_buf = b''
        while True:
            seg_buf = c.recv(1024)
            recv_buf += seg_buf
            if len(seg_buf) < 1024:
                break
        c.sendall(buf)
        # 2.4 process received P and {p} --- recv_buf
        update_list = processPpbuf(recv_buf, recv_time_list)
        print('receive auxiliary info.:', update_list)

        # 3.possible message list [src, dst, ttl]
        tmp_str = c.recv(4096).decode()
        MessageList = []
        message_list_str = tmp_str.split(';')
        num_mg = int(message_list_str[0])
        message_list_str.pop(0)
        assert num_mg == len(message_list_str)
        for line in message_list_str:
            # if line.strip() == '':
            #     continue
            tunple_list = line.split(',')
            dst = int(tunple_list[0])
            src = int(tunple_list[1])
            message_id = tunple_list[2]
            deadline = tunple_list[3]
            MessageList.append((dst, src, message_id, deadline))
        print('ask message metric:', MessageList)
        c.send('ack message list:{}'.format(len(MessageList)).encode())

        # 4. node j receive, 'G of each message carried here'
        # 4.1 local (j's) G
        G_local = []
        for tuple in MessageList:
            pktdstid = tuple[0]
            deadline = tuple[3]
            G_value, G_path = __cal_delivery_metric_G(trip_time, deadline, pktdstid)
            G_local.append((G_value, G_path.copy()))
        # 4.2 i's G
        tmp_buf = c.recv(NUM_OF_NODES * (NUM_OF_NODES + 3) * 8 + 5)
        num_mg_recv = int.from_bytes(tmp_buf[0:4], 'big', signed=False)
        assert num_mg_recv == num_mg
        G_from_j = []
        if num_mg_recv > 0:
            G_recv_str = np.frombuffer(tmp_buf[4:]).reshape(len(MessageList), -1)
            for idx in range(len(MessageList)):
                pktdstid = MessageList[idx][0]
                G_value = G_recv_str[idx, 0]
                path = []
                for idx_item in range(1, G_recv_str.shape[1]):
                    relay_node = int(G_recv_str[idx, idx_item])
                    if relay_node != -1:
                        path.append(relay_node)
                    else:
                        break
                G_from_j.append((G_value, path.copy()))
        assert len(G_local) == len(G_from_j)

        notify_list = []
        for idx in range(num_mg):
            # if j is a better relay for message_list[idx]
            # local is i, remote is j, i->j
            if G_local[idx][0] < G_from_j[idx][0]:
                # dst_id, src_id, message_id
                notify_list.append((MessageList[idx][0], MessageList[idx][1], MessageList[idx][2]))
        print('metric in i (i->j):', G_local)
        print('metric in j (i->j):', G_from_j)

        # 5.notify node i replicate messages ! [socket 192.168.1.8 (Raspberry)---]
        # s.send('Notify Delete Message List! from {} to {}'.format(NODE_ID, target_id).encode())
        tmp_str = ''
        for tunple in notify_list:
            tmp_str += '{},{},{};'.format(tunple[0], tunple[1], tunple[2])
        num_notify_list = len(notify_list)
        if num_notify_list > 0:
            tmp_str = '{};'.format(num_notify_list) + tmp_str[:-1]
        else:
            tmp_str = '{}'.format(num_notify_list)
        c.send(tmp_str.encode())
        print('i notify j list:', notify_list)

        # 6. delete files
        tmp_str = c.recv(2048).decode()
        notify_delete_list = []
        notify_delete_list_str = tmp_str.split(';')
        num_delete_list = int(notify_delete_list_str[0])
        notify_delete_list_str.pop(0)
        assert num_delete_list == len(notify_delete_list_str)
        for line in notify_delete_list_str:
            # if line.strip() == '':
            #     continue
            tunple = line.split(',')
            notify_delete_list.append((tunple[0].strip(), tunple[1].strip(), tunple[2].strip()))
        print('j notify i delete_list:', notify_delete_list)
        processDelFile(notify_delete_list)
        print('delete forwarded messages')

        c.close()
        print('N2N socket close (listen side)')

# copy node_message_list.csv messages
def init_bikeInNode(socket_c):
    # 1. update bike_config
    # 2. update message list in node
    print('Init Bike buffer to be sent to other nodes!')
    # for i, j, k in os.walk('./'):
    #     print(i, j, k)
    to_send_files = ['node_message_list.csv']
    trans_n2b_filelist = []
    trans_n2b_messagelist = 'dst_id, src_id, message_id, deadline, gentime, size(bytes), filename'

    with open('node_message_list.csv', 'r+') as fp:
        lines = fp.readlines()
        for idx in range(1, len(lines), 1):
            if lines[idx].strip() == '':
                continue
            tlist = lines[idx].split(',')
            filename = tlist[-1].strip()
            to_send_files.append(filename)
            # # dst, src, message_id, deadline
            # trans_n2b_filelist.append(filename)
            # trans_n2b_messagelist += '\n'+lines[idx]
    # send header (filenames)
    tmp_str = ''
    for item in to_send_files:
        tmp_str += '{};'.format(item)
    buf = tmp_str[:-1].encode()
    length = len(buf)
    buf = length.to_bytes(4, 'big', signed=False) + buf
    socket_c.sendall(buf)

    for filepath in to_send_files:
        # check filename
        filename = socket_c.recv(1024).decode()
        assert filename == filepath
        print('begin:', filename)
        data = b''
        with open(filename, 'rb') as fp:
            while True:
                tmp_data = fp.read(1024)
                data += tmp_data
                if len(tmp_data) < 1024:
                    break
        length = len(data)
        buf = length.to_bytes(5, 'big', signed=False) + data
        print(filename, length, length.to_bytes(5, 'big', signed=False))
        socket_c.sendall(buf)

    tmp_str = socket_c.recv(1024).decode()
    print(tmp_str)
    if tmp_str != 'end transmission':
        print('Internal Err!')
    tmp_str = '{};{}'.format(NODE_ID, NODE_IP)
    print(tmp_str)
    socket_c.send(tmp_str.encode())

def check_messages_in_node():
    message_list = []
    with open('node_message_list.csv', 'r') as fp:
        lines = fp.readlines()
        # label which row should be delete
        delete_idx_label = np.zeros(len(lines), dtype='int')
        for idx in range(len(lines)):
            if idx == 0:
                # if lines[0] != 'dst_id, src_id, message_id, deadline, gentime, size(bytes), filename':
                #     print('Internal Err! check header of node_message_list.csv')
                continue
            line = lines[idx]
            tunple_list = line.strip().split(',')
            filename = tunple_list[-1].strip()
            if not os.path.exists(filename):
                print('Internal Err! check_messages_in_node, file {} miss!'.format(filename))
                delete_idx_label[idx] = 1
                continue
            if not os.path.isfile(filename):
                print('Internal Err! check_messages_in_node, {} is not a file!'.format(filename))
                delete_idx_label[idx] = 1
                continue
            message_list.append((tunple_list[0].strip(), tunple_list[1].strip(), tunple_list[2].strip(), idx))
        # remove duplicate messages
        for idx_m in range(len(message_list)):
            (dst_id, src_id, m_id, idx) = message_list[idx_m]
            for other_idx_m in range(idx_m+1, len(message_list)):
                (oth_dst_id, oth_src_id, oth_m_id, oth_idx) = message_list[other_idx_m]
                if oth_dst_id == dst_id and oth_src_id == src_id and oth_m_id == m_id:
                    assert idx != oth_idx
                    print('Internal Err! duplicate message {}-{}-{}'.format(oth_dst_id, oth_src_id, oth_m_id))
                    delete_idx_label[oth_idx] = 1
    last_valid_idx = -1
    for i in range(len(delete_idx_label)):
        if delete_idx_label[i] == 0 and last_valid_idx < i:
            last_valid_idx = i
    with open('node_message_list.csv', 'w+') as fp:
        for i in range(len(delete_idx_label)):
            if delete_idx_label[i] == 0:
                fp.write(lines[i].strip())
                if i != last_valid_idx:
                    fp.write('\n')

# connect and listen as node j (client)
def Bike2Node_listen(node_id, node_p2p_ip):
    try:
        #create an AF_INET, STREAM socket (TCP)
        s = socket.socket()
    except socket.error:
        print('Failed to create socket (listen side)')
        sys.exit()
    print('B2N socket created (listen side)')
    # hostname = socket.gethostname()
    # hostip = socket.gethostbyname(hostname)
    port = 12349  # 设置端口
    s.bind((node_p2p_ip, port))

    s.listen(5)
    while True:
        c, addr = s.accept()
        print('B2N accept：', addr)
        # 1.wait bike2node to connect
        # tmp_str = c.recv(1024).decode()
        # print(tmp_str)
        TripTime = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        # 2.read (node_from_id, node_from_ip) & message list in bike
        tmp_str = c.recv(1024).decode()
        print(tmp_str)
        tunple_list = tmp_str.split(';')
        Node_from_ID = int(tunple_list[0])
        Node_from_IP = tunple_list[1]
        c.send('Bike2Node connect success!;{}'.format(NODE_ID).encode())

        tmp_str = c.recv(2048).decode()
        print(tmp_str)
        message_list_str = tmp_str.split(';')
        num_mg = int(message_list_str[0])
        message_list_str.pop(0)
        assert num_mg == len(message_list_str)
        MessageList = []
        for line in message_list_str:
            if line.strip() == '':
                continue
            tunple_list = line.split(',')
            dst = int(tunple_list[0])
            src = int(tunple_list[1])
            message_id = tunple_list[2].strip()
            deadline = tunple_list[3].strip()
            MessageList.append((dst, src, message_id, deadline))
        print(MessageList)
        # some message was forwarded
        # tell bike, node j (itself) need some messages
        notify_forward_list = []
        # tell i, node i should delete some forwarded messages
        notify_delete_list = []
        # adjust MessageList to Query G
        tmp_del_idx_list = []
        for idx in range(len(MessageList)):
            (dst, src, message_id, deadline) = MessageList[idx]
            if dst == NODE_ID:
                notify_forward_list.append((dst, src, message_id))
                notify_delete_list.append((dst, src, message_id))
                tmp_del_idx_list.append(idx)
                continue
            for fd_idx in range(len(Fd_MessageList)):
                (fd_dst, fd_src, fd_message_id) = Fd_MessageList[fd_idx]
                if fd_message_id == message_id and fd_dst == dst and fd_src == src:
                    notify_delete_list.append((dst, src, message_id))
                    tmp_del_idx_list.append(idx)
                    break
        for del_idx in range(len(tmp_del_idx_list)-1, -1, -1):
            MessageList.pop(tmp_del_idx_list[del_idx])

        # 3.connect to [node2node]
        notify_forward_list = Node2Node_connect(Node_from_ID, Node_from_IP, MessageList, TripTime, notify_delete_list, notify_forward_list)
        # c.send('Notify Bike to Forward Messages! from {}'.format(NODE_ID).encode())
        print('Notify Bike to Forward Messages! from {}'.format(NODE_ID))
        tmp_str = ''
        for tunple in notify_forward_list:
            tmp_str += '{},{},{};'.format(tunple[0], tunple[1], tunple[2])
        num_fwd_file = len(notify_forward_list)
        if num_fwd_file > 0:
            tmp_str = '{};'.format(num_fwd_file) + tmp_str[:-1]
        else:
            tmp_str = '{}'.format(num_fwd_file)
        c.sendall(tmp_str.encode())
        print('notify_forward_list:', tmp_str)

        # 4.foward message ! [socket 192.168.1.8 (Raspberry)---]
        header_str = ''
        for tunple in notify_forward_list:
            header = c.recv(1024).decode()
            print(header)
            list1 = header.split(';')
            assert int(list1[1]) == int(tunple[0])
            assert int(list1[2]) == int(tunple[1])
            assert int(list1[3]) == int(tunple[2])
            filename = list1[7].strip()
            c.send('ack begin transmission'.encode())
            header_str += '\n{},{},{},{},{},{},{}'.format(list1[1], list1[2], list1[3], list1[4], list1[5], list1[6], list1[7])
            # receive messages
            label = True
            data = b''
            p_str = ''
            reserve_len = int.from_bytes(b'\xff\xff\xff\xff\xff', 'big', signed=False)
            while True:
                buf = c.recv(1024)
                buf_len = len(buf)
                if label:
                    length = int.from_bytes(buf[0:5], 'big', signed=False)
                    print(length, buf[0:5], len(buf))
                    buf = buf[5:buf_len]
                    buf_len = buf_len-5
                    label = False
                    reserve_len = length
                    p_str = 'received {} bytes'.format(reserve_len)
                # print(buf_len)
                data += buf
                reserve_len = reserve_len - buf_len
                if reserve_len <= 0:
                    print(p_str)
                    break
            with open(filename, 'wb') as fp:
                fp.write(data)
            c.send('end transmission'.encode())
            print('end transmission')
        tmp_str = c.recv(1024).decode()
        print(tmp_str)
        assert tmp_str == 'file transmission B2N closed'
        with open('node_message_list.csv', 'a+') as fp:
            fp.write(header_str)

        # check 'duplicate file' and 'miss file'
        check_messages_in_node()
        # modify the bike that arrived that this node
        init_bikeInNode(c)
        c.close()
        print('B2N socket closed (listen side)')

def ThdUpdatePandpEachDay():
    # 1 day
    # schedule.every().day.at("00:00").do(notify_new_day)
    # 10s
    schedule.every(360).seconds.do(notify_new_day)
    while True:
        schedule.run_pending()
        time.sleep(5)
        # print('\nsleep ThdUpdatePandpEachDay()')


init()
print('Node ID:', NODE_ID)
print('N2N IP:', NODE_IP)
print('B2N IP:', NODE_P2P_IP)

# node (as node i) listen node j connect
thd_n2n = threading.Thread(target=Node2Node_listen, args=(NODE_ID, NODE_IP))
thd_n2n.setDaemon(True)
thd_n2n.start()

# node (as node i) listen to bike connect
thd_b2n = threading.Thread(target=Bike2Node_listen, args=(NODE_ID, NODE_P2P_IP))
thd_b2n.setDaemon(True)
thd_b2n.start()

thd_updateP = threading.Thread(target=ThdUpdatePandpEachDay)
thd_updateP.setDaemon(True)
thd_updateP.start()

while True:
    str = input('ESC(e)? NEW DAY NOTIFY (n)? Add a trip (a) ?')
    if str == 'e':
        # statistics
        break
    elif str == 'n':
        notify_new_day()
    elif str == 'a':
        # tmp_str = input('from i to j at XX.XX (0~24)')
        # print(tmp_str)
        current_time = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        Node2Node_connect(1, '172.16.2.2', [], current_time, [], [])
