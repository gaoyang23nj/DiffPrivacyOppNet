# 进行路由metric比较时候，记下来当前的数值;
# 路径挖掘的时候, 先得到metric值,只有高于之前才继续进行
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
    def __init__(self, scenarioname, num_of_nodes, buffer_size, min_time, max_ttl):
        print('memo')
        self.scenarioname = scenarioname
        # 最大max_ttl
        self.max_ttl = max_ttl
        # 节点个数
        self.num_of_nodes = num_of_nodes
        # (tm,tmp_high,tmp_low,is_holiday)*365 每天的天气/休假状况
        self.list_weather = []
        self.readWeather()
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        self.listRouter = []
        for node_id in range(num_of_nodes):
            tmpRouter = RoutingRGMM(node_id, num_of_nodes, min_time, self.list_weather, self.max_ttl)
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

    # 通知每个节点更新自己的状态
    def notify_new_day(self, runningtime):
        for each_node_id in range(self.num_of_nodes):
            self.listRouter[each_node_id].notify_new_day(runningtime)

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
        # print('{}->{}'.format(a_id, b_id))
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        # a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        a_listpkt_hist = []
        # 1) b_id 告诉 a_id: b_id有哪些pkt
        b_listpkt = self.listNodeBuffer[b_id].getlistpkt()
        a_listpkt = self.listNodeBuffer[a_id].getlistpkt()
        isNeedtoRoutingDcs = False
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
                else:
                    totran_pktlist.append(cppkt)
                    isNeedtoRoutingDcs = True
        # 在ttl时间(从当前这天(包括当前这天)开始, 到ttl结束的当天(当天))内 从self.node_id到b_id的成功概率
        # 获得概率密度 ttl以内 每个hour一个probdensity数值
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                print('pkt_{} ({}->{}): {}->{} received!.'.format(tmp_pkt.pkt_id, tmp_pkt.src_id,
                                                                tmp_pkt.dst_id, a_id, b_id))
                self.num_comm = self.num_comm + 1
                continue
            assert isNeedtoRoutingDcs
            isNeedtoFwd = False
            # 判断是否有必要从a转发给b
            P_a_dst, path_a = self.listRouter[a_id].get_values_before_up(runningtime, tmp_pkt.dst_id)
            # 若之前已经计算过 有path存在, 并且最优path经过b节点, 不必计算 直接转发即可；
            # 否则 a 和 b 分别计算路径 并且评价出最优的那个;
            # 如果刚好遇上最佳path上的节点直接转发，
            if len(path_a) > 0 and path_a[1] == b_id:
                isNeedtoFwd = True
                print('pkt_{} ({}->{}):{}({}),{}(...) find as wish'.format(tmp_pkt.pkt_id, tmp_pkt.src_id,
                                                                          tmp_pkt.dst_id, a_id, P_a_dst,
                                                                           b_id))
            else:
                P_b_dst, path_b = self.listRouter[b_id].get_values_before_up(runningtime, tmp_pkt.dst_id)
                print('pkt_{} ({}->{}):{}({}),{}({})'.format(tmp_pkt.pkt_id, tmp_pkt.src_id,
                                                             tmp_pkt.dst_id, a_id, P_a_dst, b_id, P_b_dst))
                print('path_a {}'.format(path_a))
                print('path_b {}'.format(path_b))
                if P_a_dst < P_b_dst:
                    isNeedtoFwd = True
            if isNeedtoFwd:
                print('fwd pkt_{} from {} to {}'.format(tmp_pkt.pkt_id, a_id, b_id))
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
    def __init__(self, node_id, num_of_nodes, min_time, input_list_weather, max_ttl):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.MIN_TIME = time.strptime(min_time.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        # 从datetime结构 转化为time结构
        self.lastAgeUpdate = self.MIN_TIME
        self.list_weather = input_list_weather
        self.max_ttl = max_ttl

        self.Threshold_P = 0.2
        self.alpha = 0.7
        self.GMM_Components=3
        self.MIN_SAMPLES = 8
        self.MAX_HOPE = 5
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
        # self.P_day_merge = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes)) * -1

        # 2.用于day内计算
        # 对各个目标节点b_id 都收集数据;
        # self.dataset_holiday[b_id] 表示针对节点b_id的相遇时间记录
        self.dataset_holiday = []
        self.dataset_workday = []
        # (weights[n] means[n] vars[n]) * 节点个数; b_id; self.GMM_Components即n个高斯
        # self.list_paramsGMM_holiday[b_id] 表示针对节点b_id的(weights[n] means[n] vars[n])
        # self.list_paramsGMM_holiday = [-1] * self.num_of_nodes
        # self.list_paramsGMM_workday = [-1] * self.num_of_nodes
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

        # 用来计算概率
        # 结果应该是 节点个数(216)*day内时间粒度(24hour)*day间(ttl.days+1)
        self.all_res_cal = [np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))] * self.num_of_nodes
        # 保存每天临时出现的metric (pktdst_id, value, pathset[index])
        self.list_metric_memo = []

    # 1.day间, 执行以前的方案; 新的时间已经是新的一天；更新缓冲区self.tmp_list_record
    def __process_record_betwday(self, update_y_day):
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
        self.p_star = 1.-np.exp(-self.list_num_trip)
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
        # 考虑第一天刚好是holiday; 如果更新了 就放到all_P_holiday; 否则还是用默认值
        if len(self.P_holiday)>0:
            self.all_P_holiday[self.node_id] = self.P_holiday[-1][1]
        if len(self.P_workday)>0:
            self.all_P_workday[self.node_id] = self.P_workday[-1][1]
        # 更新self.P_day_merge; 此处可以优化一下流程先建立好 映射表(索引表)
        # for i in range(NUM_DAYS_INYEAR):
        #     for tmp in self.P_holiday:
        #         if tmp[0] == i:
        #             self.P_day_merge[i,:] = tmp[1]
        #             break
        #     for tmp in self.P_workday:
        #         if tmp[0] == i:
        #             self.P_day_merge[i,:] = tmp[1]
        #             break

    # 2.day内, 执行GMM; 日内prob:GMM参数/概率串
    def __process_dataset_withGMM(self):
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
            # self.list_paramsGMM_holiday[b_id] = tunple_b_h
            # self.list_paramsGMM_workday[b_id] = tunple_b_w
            self.probs_holiday[b_id, :] = probs_h
            self.probs_workday[b_id, :] = probs_w
        self.all_probs_holiday[self.node_id] = self.probs_holiday
        self.all_probs_workday[self.node_id] = self.probs_workday

    # 刷新从self.node_id的视角来看 target_id到各个节点的评价
    def __update_probdensity(self, runningtime):
        tmp_res_cal = np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))
        today_yday = runningtime.tm_yday
        # 收集day间数据
        res_list_betwday = []
        # 1.处理cond_prob
        for i in range(self.max_ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # [np.array(各个节点), np.array, np.array, ... ] 一共 ttl个
                res_list_betwday.append(self.all_P_holiday[self.node_id])
            else:
                res_list_betwday.append(self.all_P_workday[self.node_id])
        res_list_betwday = np.array(res_list_betwday)
        # ttl.days * num_nodes 第几天/发往哪个节点
        cond_P = self.__cal_cond_prob(res_list_betwday)
        # 2.计算final prob streaming
        for i in range(self.max_ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # 216*24, 13*216 第j个小时
                for j in range(i*24, (i+1)*24):
                    tmp_res_cal[:, j] = self.all_probs_holiday[self.node_id][:, j - i * 24] * cond_P[i, :]
            else:
                for j in range(i*24, (i+1)*24):
                    tmp_res_cal[:, j] = self.all_probs_workday[self.node_id][:, j - i * 24] * cond_P[i, :]
        self.all_res_cal[self.node_id] = tmp_res_cal
        if self.node_id == 3:
            tmp = self.all_res_cal[self.node_id][86,:].sum()
            print('sum_3_res_cal:{}'.format(tmp))

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
        # 多少天 * 多少个对端节点
        res = np.ones((res_list_betwday.shape[0], self.num_of_nodes))
        for index in range(0, res_list_betwday.shape[0]):
            # 从0到index-1 行；从最开始的一天(0) 到 这一天(index)
            for j in range(0, index):
                res[index, :] = res[index, :] * (1. - res_list_betwday[j, :])
            res[index, :] = res[index, :] * res_list_betwday[index, :]
        return res

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
            # 不借助当前评价中的的对手节点
            # 每条路径不能太弱。holiday or not; 小于阈值 概率过低， link不成立; 从[fwlist[-1]]到[tonode]
            if (self.all_P_workday[fwlist[-1]][tonode] < self.Threshold_P) \
                    and (self.all_P_holiday[fwlist[-1]][tonode] < self.Threshold_P):
                continue
            # 宏观link强 但是微观link还不足以构建也不行
            if (self.all_probs_workday[fwlist[-1]][tonode, :].sum() < self.Threshold_P) \
                    and (self.all_probs_holiday[fwlist[-1]][tonode].sum() < self.Threshold_P):
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

    #
    def __cal_convolprob(self, runningtime, onepath):
        tmp_targetmatrix = np.zeros((len(onepath)-1, (self.max_ttl.days+1)*24))
        for i in range(1, len(onepath)):
            # onepath[i-1] -> onepath[i]
            tmp_targetmatrix[i-1, :] = self.all_res_cal[onepath[i-1]][onepath[i], :]
        t = runningtime.tm_hour
        targetmatrix = tmp_targetmatrix[:, t:]
        matrix_size = targetmatrix.shape
        assert len(onepath)-1 == matrix_size[0]
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
    def notify_new_day(self, runningtime):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # 1.day间处理；如果新的一天已经开始, 更新在每天的概率
        assert self.lastAgeUpdate.tm_yday + 1 == runningtime.tm_yday

        # 1. day间 更新处理;
        self.__process_record_betwday(self.lastAgeUpdate.tm_yday)
        # 2. day内 更新处理
        self.__process_dataset_withGMM()
        # 3. 更新矩阵 self.all_res_cal [num_station * (24*ttl)] num_nodes
        self.__update_probdensity(runningtime)
        # 已经满一天 清空buffer
        self.tmp_list_record.clear()
        # 清空前一天的metric缓存
        self.list_metric_memo.clear()
        # 更新处理时间
        self.lastAgeUpdate = runningtime

    # 从本节点出发 不经过nby_nodeid 从而到达pktdst_id的 各种路径所提供概率的最大值
    # 为了memo 不做nby_nodeid
    def get_values_before_up(self, runningtime, pktdst_id):
        # 查询一下 今天是否已经计算过了
        for tunple in self.list_metric_memo:
            if tunple[0] == pktdst_id:
                return tunple[1], tunple[2]
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        cur_list = []
        cur_id = self.node_id
        for i in range(self.num_of_nodes):
            # 到哪里
            tmplist = [i, 0., [cur_id]]
            cur_list.append(tmplist)
        # 逐hop推进
        for cur_hop in range(self.MAX_HOPE):





        if max_index != -1:
            res_path = path_set[max_index]
            # 加速计算
            self.list_metric_memo.append((pktdst_id, max_value, res_path))
        else:
            self.list_metric_memo.append((pktdst_id, 0., []))
        # print('end cal_convolprob')
        # print(datetime.datetime.now())
        # pkt_dst_id a-b a(-b)
        return max_value, res_path

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id, *args):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # b到任何节点的值
        # P_b_any = args[0]
        a_id = self.node_id

        self.tmp_list_record.append((runningtime, a_id, b_id))
        # 2.day内处理; 整理GMM所需的数据集
        thm = runningtime.tm_hour + (runningtime.tm_min/60.0)
        # holiday or not
        if self.list_weather[runningtime.tm_yday-1][3]:
            self.dataset_holiday[b_id].append(thm)
        else:
            self.dataset_workday[b_id].append(thm)

    # 控制信息发布
    def getInfo(self, runningtime):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # 交换信息更新
        # 已经实现每日更新
        self.all_UpdatingTime[self.node_id] = runningtime
        return self.all_UpdatingTime, self.all_P_holiday, self.all_P_workday, \
               self.all_probs_holiday, self.all_probs_workday, self.all_res_cal

    def updateInfo(self, info_othernode):
        other_UpdatingTime, other_P_holiday, other_P_workday, \
        other_probs_holiday, other_probs_workday, other_res_cal = info_othernode
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
            self.all_res_cal[update_node] = other_res_cal[update_node]
            self.all_UpdatingTime[update_node] = other_UpdatingTime[update_node]