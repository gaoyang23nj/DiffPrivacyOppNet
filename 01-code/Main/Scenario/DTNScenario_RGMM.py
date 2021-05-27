from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt

import copy
import numpy as np
import pandas as pd
import math
import datetime
import time
from sklearn import mixture
import os
import matplotlib.pyplot as plt

# 用GMM描述 day内
# 用access方法描述 day间
EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data'
StationInfoPath = '../EncoHistData_NJBike/station_info.csv'
WeatherInfo = '../NanjingBikeDataset/Pukou_Weather.xlsx'

NUM_DAYS_INYEAR = 365

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_RGMM(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, num_of_nodes, buffer_size, min_time):
        self.scenarioname = scenarioname
        # (tm,tmp_high,tmp_low,is_holiday)*365 每天的天气/休假状况
        self.list_weather = []
        self.readWeather()
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        self.listRouter = []
        for node_id in range(num_of_nodes):
            tmpRouter = RoutingRGMM(node_id, num_of_nodes, min_time, self.list_weather)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size)
            self.listNodeBuffer.append(tmpBuffer)
        self.num_comm = 0
        return

    def readWeather(self):
        weather = pd.read_excel(WeatherInfo, engine='openpyxl')
        v1 = weather.values
        for i in range(0, len(weather.values)):
            tm = time.strptime(v1[i][0].split(' ')[0], "%Y-%m-%d")
            tmp_high = int(v1[i][1].replace('°',''))
            tmp_low = int(v1[i][2].replace('°',''))
            if 1.0 == v1[i][6]:
                is_holiday = True
            elif 0.0 == v1[i][6]:
                is_holiday = False
            elif (tm.tm_wday==5) or (tm.tm_wday == 6):
                is_holiday = True
            else:
                is_holiday = False
            self.list_weather.append((tm,tmp_high,tmp_low,is_holiday))

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(newpkt)
        return

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # 控制信息 进行更新操作
        self.listRouter[a_id].notifylinkup(runningtime, b_id)
        # ================== 控制信息 交换==========================
        info_switch_a = self.listRouter[a_id].getInfo(runningtime)
        info_switch_b = self.listRouter[b_id].getInfo(runningtime)
        self.listRouter[a_id].updateInfo(info_switch_b)
        self.listRouter[b_id].updateInfo(info_switch_a)
        # ================== 报文 交换==========================
        self.sendpkt(runningtime, a_id, b_id)

    # 报文发送 a_id -> b_id
    def sendpkt(self, runningtime, a_id, b_id):
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        # b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        b_listpkt_hist = []
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        # hist列表 和 当前内存里都没有 来自a的pkt   a才有必要传输
        for a_pkt in a_listpkt:
            isDuplicateExist = False
            for bpktid_hist in b_listpkt_hist:
                if a_pkt.pkt_id == bpktid_hist:
                    isDuplicateExist = True
                    break
            if not isDuplicateExist:
                for bpkt in b_listpkt:
                    if a_pkt.pkt_id == bpkt.pkt_id:
                        isDuplicateExist = True
                        break
            if not isDuplicateExist:
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1
                continue
            P_b_dst = self.listRouter[b_id].get_values_before_up(runningtime, tmp_pkt.dst_id)
            P_a_dst = self.listRouter[a_id].get_values_before_up(runningtime, tmp_pkt.dst_id)
            if P_a_dst < P_b_dst:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1

    def print_res(self, listgenpkt):
        output_str = '{}\n'.format(self.scenarioname)
        total_delay = datetime.timedelta(seconds=0)
        total_succnum = 0
        total_pkt_hold = 0
        for i_id in range(len(self.listNodeBuffer)):
            list_succ = self.listNodeBuffer[i_id].getlistpkt_succ()
            for i_pkt in list_succ:
                tmp_delay = i_pkt.succ_time - i_pkt.gentime
                total_delay = total_delay + tmp_delay
            tmp_succnum = len(list_succ)
            total_succnum = total_succnum + tmp_succnum

            list_pkt = self.listNodeBuffer[i_id].getlistpkt()
            total_pkt_hold = total_pkt_hold + len(list_pkt)
        succ_ratio = total_succnum/len(listgenpkt)
        if total_succnum != 0:
            avg_delay = total_delay/total_succnum
            output_str += 'succ_ratio:{} avg_delay:{} '.format(succ_ratio, avg_delay)
        else:
            avg_delay = ()
            output_str += 'succ_ratio:{} avg_delay:null '.format(succ_ratio)
        output_str += 'num_comm:{}\n'.format(self.num_comm)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, len(listgenpkt), total_succnum)
        print(output_str)
        res = {'succ_ratio': succ_ratio, 'avg_delay': avg_delay, 'num_comm': self.num_comm}
        config = {'ratio_bk_nodes': 0, 'drop_prob': 1}
        return output_str, res, config


class RoutingRGMM(object):
    def __init__(self, node_id, num_of_nodes, min_time, input_list_weather):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.MIN_TIME = time.strptime(min_time.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        # 从datetime结构 转化为time结构
        self.lastAgeUpdate = self.MIN_TIME
        self.list_weather = input_list_weather

        self.Threshold_P = 0.2
        self.alpha = 0.7
        self.GMM_Components=3
        self.MIN_SAMPLES = 15
        self.MAX_HOPE = 10
        self.input_data_dir = EncoHistDir_SDPair
        # 记录不满足一天的记录
        self.tmp_list_record = []

        # 1.用于day间计算
        self.list_num_trip = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes), dtype='int') * -1
        self.p_star = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes)) * -1
        # np.ones( self.num_of_nodes)形成的列表
        self.P_holiday = []
        self.P_workday = []
        # 按照时间顺序 合并 P_holiday和P_workday
        self.P_day_merge = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes)) * -1

        # 2.用于day内计算
        # 对各个目标节点b_id 都收集数据;
        # self.dataset_holiday[b_id] 表示针对节点b_id的相遇时间记录
        self.dataset_holiday = []
        self.dataset_workday = []
        # (weights[n] means[n] vars[n]) * 节点个数; b_id; self.GMM_Components即n个高斯
        # self.list_paramsGMM_holiday[b_id] 表示针对节点b_id的(weights[n] means[n] vars[n])
        self.list_paramsGMM_holiday = [-1] * self.num_of_nodes
        self.list_paramsGMM_workday = [-1] * self.num_of_nodes
        # 比较粗糙的粒度 每个小时一个数值
        self.probs_holiday = np.zeros((self.num_of_nodes, 24))
        self.probs_workday = np.zeros((self.num_of_nodes, 24))
        for i in range(self.num_of_nodes):
            self.dataset_holiday.append([])
            self.dataset_workday.append([])

        # 3.用于各个节点之间的信息交换
        # 来自于不同节点的 self.probs_workday 和 self.probs_holiday
        self.all_probs_holiday = [np.zeros((self.num_of_nodes, 24))] * self.num_of_nodes
        self.all_probs_workday = [np.zeros((self.num_of_nodes, 24))] * self.num_of_nodes
        # 来自不同节点的 P_holiday [num_nodes]*num_nodes 和 P_workday
        self.all_P_holiday = [np.zeros(self.num_of_nodes)] * self.num_of_nodes
        self.all_P_workday = [np.zeros(self.num_of_nodes)] * self.num_of_nodes
        # 更新时间
        self.all_UpdatingTime = [self.MIN_TIME] * self.num_of_nodes

    # 1.day间, 执行以前的方案; 新的时间已经是新的一天；更新缓冲区self.tmp_list_record
    def process_record_betwday(self, update_y_day):
        # 1.日间prob: (老方法)
        # 1.1 list_num_trip (365,num_nodes) 记录一年内 trip: src-dst 每天发生的次数;
        # list_num_trip[i] 记录本年(2017年)内第i天 trip:src-dst 发生的次数;
        # 1.2 p_star (365,num_nodes) 记录某一天trip次数对prob的影响
        # p_star[i] = 1-e^list_num_trip[i]
        # 1.3 分成holiday (P_holiday)和 workday (P_weekday)来分别处理
        # P_holiday [(yday, num_nodes)]
        # P_holiday[0] = p_star[0];  P_holiday[i] = alpha*P_holiday[i-1] + (1-alpha)*p_star[i-1];
        # P_weekday的处理 与 P_holiday相同
        # P_weekday 和 P_holiday 按照时间顺序合并到一起, P_merge
        tmplist_numtrip_oneday = np.zeros(self.num_of_nodes)
        for tunple in self.tmp_list_record:
            assert tunple[1] == self.node_id
            assert tunple[0].tm_yday == update_y_day
            tmplist_numtrip_oneday[tunple[2]] = tmplist_numtrip_oneday[tunple[2]] + 1
        # 为了防止跨几天的情况出现(下一次更新在好几天之后), 强制更新 list_num_trip和p_star
        self.list_num_trip[update_y_day-1, :] = tmplist_numtrip_oneday
        # 确保跨过几天也不会影响 矩阵全部更新
        self.p_star = 1-np.exp(-self.list_num_trip)
        # holiday or not
        if self.list_weather[update_y_day-1][3]:
            if len(self.P_holiday) == 0:
                self.P_holiday.append((update_y_day-1, self.p_star[update_y_day-1,:]))
            else:
                new_prob = self.alpha * self.P_holiday[-1][1] \
                                   + (1-self.alpha) * self.p_star[update_y_day-1, :]
                self.P_holiday.append((update_y_day-1, new_prob))
        else:
            if len(self.P_workday) == 0:
                self.P_workday.append((update_y_day-1, self.p_star[update_y_day-1,:]))
            else:
                new_prob = self.alpha * self.P_workday[-1][1] \
                                   + (1-self.alpha) * self.p_star[update_y_day-1, :]
                self.P_workday.append((update_y_day-1, new_prob))
        # 更新self.P_day_merge; 此处可以优化一下流程先建立好 映射表(索引表)
        for i in range(NUM_DAYS_INYEAR):
            for tmp in self.P_holiday:
                if tmp[0] == i:
                    self.P_day_merge[i,:] = tmp[1]
                    break
            for tmp in self.P_workday:
                if tmp[0] == i:
                    self.P_day_merge[i,:] = tmp[1]
                    break

    # 2.day间, 执行GMM; 日内prob:GMM参数/概率串
    def process_dataset_withGMM(self):
        for b_id in range(self.num_of_nodes):
            if b_id == self.node_id:
                continue
            dataset1 = np.array(self.dataset_holiday[b_id]).reshape(-1,1)
            if dataset1.shape[0] >= self.MIN_SAMPLES:
                probs_h, tunple_b_h = self.__precdit_GMM(dataset1)
            else:
                probs_h = np.zeros(24)
                tunple_b_h = (np.zeros(3),np.zeros(3),np.zeros(3))
            dataset2 = np.array(self.dataset_workday[b_id]).reshape(-1,1)
            if dataset2.shape[0] >= self.MIN_SAMPLES:
                probs_w, tunple_b_w = self.__precdit_GMM(dataset2)
            else:
                probs_w = np.zeros(24)
                tunple_b_w = (np.zeros(3),np.zeros(3),np.zeros(3))
            self.list_paramsGMM_holiday[b_id] = tunple_b_h
            self.list_paramsGMM_workday[b_id] = tunple_b_w
            self.probs_holiday[b_id, :] = probs_h
            self.probs_workday[b_id, :] = probs_w

    # 按照24小时计算概率
    def __precdit_GMM(self, dateset):
        clf = mixture.GaussianMixture(n_components=self.GMM_Components, covariance_type='diag')
        clf.fit(dateset)
        XX = np.linspace(0, 24 - 1, 24).reshape(-1, 1)
        Z = clf.score_samples(XX)
        probs = np.exp(Z)
        sum_probs = probs.sum()
        score = clf.score(XX)
        # print(Z)
        # print(probs)
        # print(sum_probs)
        return probs, (clf.weights_, clf.means_, clf.covariances_)

    def __comp_GMM(self, x, weights_, means_, vars_):
        w_len = len(weights_)
        r = 0.
        # 遍历各个高斯分量
        for i in range(w_len):
            print(weights_[i], means_[i], vars_[i])
            f_i = (math.exp(-((x-means_[i])**2)/(2*vars_[i])) / math.sqrt(2*math.pi*vars_[i]))*weights_[i]
            r = r + f_i
        return r

    # 生成矩阵 ttl.days * num_nodes 每个位置表示 条件概率 a-b事件在这天发生
    def __cal_cond_prob(self, res_list_betwday):
        res = np.ones((res_list_betwday.shape[0], self.num_of_nodes))
        for index in range(0, res_list_betwday.shape[0]):
            # 从0到index-1 行
            for j in range(0, index):
                res[index, :] = res[index, :] * (1. - res_list_betwday[j, :])
            res[index, :] = res[index, :] * res_list_betwday[index, :]
        return res

    def getprobdensity(self, runningtime, ttl):
        today_yday = runningtime.tm_yday
        # 收集day间数据
        res_list_betwday = []
        # 1.处理cond_prob
        for i in range(ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # [np.array(各个节点), np.array, np.array, ... ] 一共 ttl个
                if len(self.P_holiday) > 0:
                    res_list_betwday.append(self.P_holiday[-1][1])
                else:
                    res_list_betwday.append(np.zeros(self.num_of_nodes))
            else:
                if len(self.P_workday) > 0:
                    res_list_betwday.append(self.P_workday[-1][1])
                else:
                    res_list_betwday.append(np.zeros(self.num_of_nodes))
        res_list_betwday = np.array(res_list_betwday)
        # ttl.days * num_nodes
        cond_P = self.__cal_cond_prob(res_list_betwday)
        # 2.计算final prob streaming
        # 结果应该是 节点个数(216)*day内时间粒度(24hour)*day间(ttl.days+1)
        res_cal = np.zeros((self.num_of_nodes, (ttl.days+1)*24))
        for i in range(ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # 216*24, 13*216
                for j in range(i*24, (i+1)*24):
                    res_cal[:, j] = self.probs_holiday[:, j - i * 24] * cond_P[i, :]
            else:
                for j in range(i*24, (i+1)*24):
                    res_cal[:, j] = self.probs_workday[:, j - i * 24] * cond_P[i, :]
        return res_cal

    def __find_next_node(self, fwlist, pktdst_id, remain_hop):
        if remain_hop == 0:
            if fwlist[-1] == pktdst_id:
                return fwlist
            else:
                return
        candidatelist_set = []
        for tonode in range(self.num_of_nodes):
            # 不走回头路
            if tonode in fwlist:
                continue
            # holiday or not; 小于阈值 概率过低，link不成立
            if (self.all_P_workday[fwlist[-1]][tonode] < self.Threshold_P) \
                    or (self.all_P_holiday[fwlist[-1]][tonode] < self.Threshold_P):
                continue
            if tonode == pktdst_id:
                tmplist = fwlist.copy()
                tmplist.append(tonode)
                candidatelist_set.append(tmplist)
            else:
                tmplist = fwlist.copy()
                tmplist.append(tonode)
                candpath = self.__find_next_node(tmplist, pktdst_id, remain_hop-1)
                if candpath != None:
                    candidatelist_set.extend(candpath)
        return candidatelist_set

    def __getpath_fromgraph(self, pktdst_id):
        fwlist_set = [self.node_id]
        path_set = self.__find_next_node(fwlist_set, pktdst_id, self.MAX_HOPE)
        return path_set

    def __cal_convolprob(self, runningtime, onepath, res_cal):
        t = runningtime.tm_hour
        targetmatrix = res_cal[onepath, t:]
        matrix_size = targetmatrix.shape
        assert len(onepath) == matrix_size[0]
        # 计算概率 的 sum次数
        num_sum = matrix_size[1]-len(onepath)+1
        sum_prob = 0.
        for index in range(num_sum):
            # 获取新的矩阵 从而上下相乘即可
            oneprob = np.ones(matrix_size[1]-index-len(onepath)+1)
            tmp_res = np.zeros((matrix_size[0], matrix_size[1]-index-len(onepath)+1))
            for i in range(matrix_size[0]):
                tmp_res[i,:] = targetmatrix[i,i:i+tmp_res.shape[1]]
                oneprob = oneprob * tmp_res[i,:]
            sum_prob =  sum_prob + oneprob.sum()
        return sum_prob

    # ========================= 提供给上层的功能 ======================================
    # 更新后, 提供 本node 的 delivery prob Matrix 给对端
    def get_values_before_up(self, runningtime, pktdst_id):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        ttl = datetime.timedelta(days=12)
        # 获得概率密度 ttl以内 每个hour一个probdensity数值
        res_cal = self.getprobdensity(runningtime, ttl)
        # 在ttl时间内 从self.node_id到b_id的成功概率
        # 1.精简graph, 从graph中提取path
        path_set = self.__getpath_fromgraph(pktdst_id)
        # 2.按照节点顺序 卷积过去;  # *关键点获得？
        # 计算连续数天的概率密度, 从runningtime 到 runningtime+ttl
        max_value = 0.
        max_index = -1
        for index in range(len(path_set)):
            value = self.__cal_convolprob(runningtime, path_set[index], res_cal)
            if max_value < value:
                max_index = index
                max_value = value
        # return max_value, path_set[max_index]
        return max_value

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id, *args):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # b到任何节点的值
        # P_b_any = args[0]
        a_id = self.node_id
        # 1.day间处理；如果新的一天已经开始, 更新在每天的概率
        if self.lastAgeUpdate.tm_yday < runningtime.tm_yday:
            # 1. day间 更新处理;
            self.process_record_betwday(self.lastAgeUpdate.tm_yday)
            # 2. day内 更新处理
            self.process_dataset_withGMM()
            # 已经满一天 清空buffer
            self.tmp_list_record.clear()
        self.tmp_list_record.append((runningtime, a_id, b_id))
        # 2.day内处理; 整理GMM所需的数据集
        thm = runningtime.tm_hour + (runningtime.tm_min/60.0)
        # holiday or not
        if self.list_weather[runningtime.tm_yday-1][3]:
            self.dataset_holiday[b_id].append(thm)
        else:
            self.dataset_workday[b_id].append(thm)
        # 更新处理时间
        self.lastAgeUpdate = runningtime

    # 控制信息发布
    def getInfo(self, runningtime):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # 交换信息更新
        if len(self.P_holiday) > 0 and len(self.P_workday) > 0 :
            self.all_P_holiday[self.node_id] = self.P_holiday[-1][1]
            self.all_P_workday[self.node_id] = self.P_workday[-1][1]
        self.all_probs_holiday[self.node_id] = self.probs_holiday
        self.all_probs_workday[self.node_id] = self.probs_workday
        self.all_UpdatingTime[self.node_id] = runningtime
        return self.all_UpdatingTime, self.all_P_holiday, self.all_P_workday, \
               self.all_probs_holiday, self.all_probs_workday

    def updateInfo(self, info_othernode):
        other_UpdatingTime, other_P_holiday, other_P_workday, \
        other_probs_holiday, other_probs_workday = info_othernode
        for update_node in range(self.num_of_nodes):
            # 本节点不更新
            if update_node == self.node_id:
                continue
            # 更早的 节点不更新
            if other_UpdatingTime[update_node] <= self.all_UpdatingTime[update_node]:
                continue
            # 满足条件开始更新
            self.all_P_holiday[update_node] = other_P_holiday[update_node]
            self.all_P_workday[update_node] = other_P_workday[update_node]
            self.all_probs_holiday[update_node] = other_probs_holiday[update_node]
            self.all_probs_workday[update_node] = other_probs_workday[update_node]
            self.all_UpdatingTime[update_node] = other_UpdatingTime[update_node]