# 用GMM描述 day内
# 用access方法描述 day间
# 结合了RGMM和memo的方案
import time
import numpy as np
from sklearn import mixture
import math
from scipy.stats import laplace

EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data'
StationInfoPath = '../EncoHistData_NJBike/station_info.csv'
WeatherInfo = '../NanjingBikeDataset/Pukou_Weather.xlsx'

NUM_DAYS_INYEAR = 365

class RoutingRTPMLap(object):
    def __init__(self, node_id, num_of_nodes, min_time, input_list_weather, max_ttl, lap_noise_scale):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.MIN_TIME = time.strptime(min_time.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        # 从datetime结构 转化为time结构
        self.lastAgeUpdate = self.MIN_TIME
        self.list_weather = input_list_weather
        self.max_ttl = max_ttl

        # laplace noise的scale参数, 应该作为输入参数
        self.LapNoiseScale = lap_noise_scale
        self.Threshold_P = 0.2
        self.alpha = 0.7
        self.GMM_Components=3
        self.MIN_SAMPLES = 8
        self.MAX_HOPE = 5
        self.input_data_dir = EncoHistDir_SDPair

        # list 记录当天的相遇事件; 例如 a_id->b_id 添加 (runningtime, a_id, b_id)
        self.contact_record_today = []

        # 1.用于day间计算
        # (365, num_nodes) 记录每天发生的contact次数
        self.list_num_trip = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes), dtype='int') * -1
        # (365, num_nodes) 记录每天发生的rho_star值
        self.rho_star = np.ones((NUM_DAYS_INYEAR, self.num_of_nodes)) * -1
        # list 成员是 P值, 每天增加新的值
        self.P_holiday = []
        self.P_workday = []

        # 2.用于day内计算
        # 两重list,对各个目标节点b_id 都收集 相遇时刻hour+min/60.的数据;
        # 其中 self.dataset_holiday[b_id] 表示self.node_id对b_id的相遇时刻记录
        self.dataset_holiday = []
        self.dataset_workday = []
        for i in range(self.num_of_nodes):
            self.dataset_holiday.append([])
            self.dataset_workday.append([])
        # (weights[n] means[n] vars[n]) * 节点个数; b_id; self.GMM_Components即n个高斯
        # self.list_paramsGMM_holiday[b_id] 表示针对节点b_id的(weights[n] means[n] vars[n])
        # self.list_paramsGMM_holiday = [-1] * self.num_of_nodes
        # self.list_paramsGMM_workday = [-1] * self.num_of_nodes
        # 比较粗糙的粒度 每个小时一个数值
        self.probs_holiday = np.zeros((self.num_of_nodes, 24))
        self.probs_workday = np.zeros((self.num_of_nodes, 24))


        # 3.用于各个节点之间的信息交换
        # 来自于不同节点的 self.probs_workday 和 self.probs_holiday
        self.all_pdf_holiday = [np.zeros((self.num_of_nodes, 24))] * self.num_of_nodes
        self.all_pdf_workday = [np.zeros((self.num_of_nodes, 24))] * self.num_of_nodes
        # 来自不同节点的 P_holiday [num_nodes]*num_nodes 和 P_workday
        self.all_P_holiday = [np.zeros(self.num_of_nodes)] * self.num_of_nodes
        self.all_P_workday = [np.zeros(self.num_of_nodes)] * self.num_of_nodes
        # 更新时间
        self.all_UpdatingTime = [self.MIN_TIME] * self.num_of_nodes

        # 4.用来计算概率
        # 结果应该是 节点个数(216)*day内时间粒度(24hour)*day间(ttl.days+1)
        self.all_res_cal = [np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))] * self.num_of_nodes
        # 保存每天临时出现的metric (pktdst_id, value, pathset[index])
        self.list_metric_memo = []

    # =======================  用于每日更新notify_new_day的内部函数     ====================
    # 1)day间, rho_star 通过每天的contact次数转换得到;
    # 2)EWEA方法 平滑 条件概率P
    def __process_Prob_interday(self, update_y_day):
        numtrip_oneday = np.zeros(self.num_of_nodes)
        for tunple in self.contact_record_today:
            assert tunple[1] == self.node_id
            assert tunple[0].tm_yday == update_y_day
            numtrip_oneday[tunple[2]] = numtrip_oneday[tunple[2]] + 1
        # 为了防止跨几天的情况出现(下一次更新在好几天之后), 强制更新 list_num_trip和p_star
        self.list_num_trip[update_y_day-1, :] = numtrip_oneday
        # 确保跨过几天也不会影响 矩阵得到全体更新
        self.rho_star = 1.-np.exp(-self.list_num_trip)
        # holiday or not
        if self.list_weather[update_y_day-1][3]:
            if len(self.P_holiday) == 0:
                self.P_holiday.append((update_y_day-1, self.rho_star[update_y_day-1,:]))
            else:
                new_prob = self.alpha * self.P_holiday[-1][1] \
                           + (1-self.alpha) * self.rho_star[update_y_day-1, :]
                self.P_holiday.append((update_y_day-1, new_prob))
        else:
            if len(self.P_workday) == 0:
                self.P_workday.append((update_y_day-1, self.rho_star[update_y_day-1,:]))
            else:
                new_prob = self.alpha * self.P_workday[-1][1] \
                           + (1-self.alpha) * self.rho_star[update_y_day-1, :]
                self.P_workday.append((update_y_day-1, new_prob))
        # 如果是第一天的holiday 或者 第一天的workday;
        # 就把新的值放到all_P_holiday (更新对应于本节点的 向量);
        # 如果第一天还没出现，仍然保持使用默认值
        if len(self.P_holiday) > 0:
            self.all_P_holiday[self.node_id] = self.P_holiday[-1][1]
        if len(self.P_workday) > 0:
            self.all_P_workday[self.node_id] = self.P_workday[-1][1]

    # day内, 执行GMM; intra-day probability density:GMM参数/24小时概率串
    def __process_pdf_intraday_withGMM(self):
        # 从self.node_id 到 每个b_id 都做一次基于GMM的day内预测(intra-day probability)
        for b_id in range(self.num_of_nodes):
            if b_id == self.node_id:
                continue
            dataset1 = np.array(self.dataset_holiday[b_id]).reshape(-1,1)
            # dataset小于一定个数 GMM无法正常启动
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
            # 把24小时概率串放到 对应的b_id里
            self.probs_holiday[b_id, :] = probs_h
            self.probs_workday[b_id, :] = probs_w
        # 更新 self.all_pdf_holiday 和 self.all_pdf_workday
        self.all_pdf_holiday[self.node_id] = self.probs_holiday
        self.all_pdf_workday[self.node_id] = self.probs_workday

    # 更新all_res_cal矩阵, 评价从self.node_id到各个节点的直接评价
    # 生成今天的ij概率序列（q_{1}^{k}...q_{24}^{k},q_{1}^{k+1}...q_{24}^{k+1}）
    def __update_probdensity(self, runningtime):
        tmp_res_cal = np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))
        # 1.按照每天的类别(是否holiday) 处理cond_prob
        today_yday = runningtime.tm_yday
        res_list_betwday = []
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
        # 2.计算final prob streaming 第i天第j个小时的概率序列
        for i in range(self.max_ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # 216*24, 13*216 第j个小时
                for j in range(i*24, (i+1)*24):
                    tmp_res_cal[:, j] = self.all_pdf_holiday[self.node_id][:, j - i * 24] * cond_P[i, :]
            else:
                for j in range(i*24, (i+1)*24):
                    tmp_res_cal[:, j] = self.all_pdf_workday[self.node_id][:, j - i * 24] * cond_P[i, :]
        # 3.更新到all里面
        self.all_res_cal[self.node_id] = tmp_res_cal
        # 观察窗口
        if self.node_id == 3:
            tmp = self.all_res_cal[self.node_id][86,:].sum()
            print('sum_3_res_cal:{}'.format(tmp))

    #=========================  函数组件：GMM拟合, 计算条件概率 ===========================
    # 基于GMM按照24小时计算概率
    def __precdit_GMM(self, dateset):
        clf = mixture.GaussianMixture(n_components=self.GMM_Components, covariance_type='diag')
        clf.fit(dateset)
        XX = np.linspace(0, 24 - 1, 24).reshape(-1, 1)
        Z = clf.score_samples(XX)
        probs = np.exp(Z)
        sum_probs = probs.sum()
        score = clf.score(XX)
        lap_noise = laplace.rvs(loc=0., scale=self.LapNoiseScale, size=24)
        for i in range(24):
            if lap_noise[i] < 0:
                lap_noise[i] = 0.
            if lap_noise[i] > 1:
                lap_noise[i] = 1.
        probs = probs + lap_noise
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

    #=========================  函数组件：图的搜索 ===========================
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
            if (self.all_pdf_workday[fwlist[-1]][tonode, :].sum() < self.Threshold_P) \
                    and (self.all_pdf_holiday[fwlist[-1]][tonode].sum() < self.Threshold_P):
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

    #===================== 离散方法 计算累计概率  ============================
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
        assert self.lastAgeUpdate.tm_yday + 1 == runningtime.tm_yday
        # 1. day间处理; 更新每天的概率P
        self.__process_Prob_interday(self.lastAgeUpdate.tm_yday)
        # 2. day内处理; 更新
        self.__process_pdf_intraday_withGMM()
        # 3. 更新矩阵 self.all_res_cal [num_station * (24*ttl)] num_nodes
        self.__update_probdensity(runningtime)
        # 4. 新的一天开始, 之前的记录已经处理完毕, 清空临时保存的相遇记录
        self.contact_record_today.clear()
        # 5. metric_memo机制 加速routing metric, 计算清空前一天的metric缓存
        self.list_metric_memo.clear()
        # 更新处理时间
        self.lastAgeUpdate = runningtime

    # 从本节点出发 不经过nby_nodeid 从而到达pktdst_id的 各种路径所提供概率的最大值
    def get_values_before_up(self, runningtime, pktdst_id):
        # 1. 查询一下 今天是否已经计算过了
        for tunple in self.list_metric_memo:
            if tunple[0] == pktdst_id:
                return tunple[1], tunple[2]
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # print('begin get path set')
        # print(datetime.datetime.now())
        # 2.1 精简graph, 从graph中提取path
        path_set = self.__getpath_fromgraph(pktdst_id)
        print('node{} num_path_set:{}'.format(self.node_id, len(path_set)))
        # print('end get path set ')
        # print(datetime.datetime.now())
        # 2.2 按照节点顺序 积分计算连续数天的概率密度和, 从runningtime 到 runningtime+ttl
        max_value = 0.
        max_index = -1
        # print('begin cal_convolprob')
        # print(datetime.datetime.now())
        for index in range(len(path_set)):
            value = self.__cal_convolprob(runningtime, path_set[index])
            if self.node_id == 3 and value > 0.1:
                print('node[{}], path[{}]:{}'.format(self.node_id, path_set[index], value))
            if max_value < value:
                max_index = index
                max_value = value
        res_path = []
        # 3. 更新memo
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

    # 当a->b(linkup时候)
    # 1.增加记录到 self.contact_record_today
    # 2.按照 holiday or not, 更新 day内数据集
    def notifylinkup(self, runningtime, b_id, *args):
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        # b到任何节点的值 P_b_any = args[0]
        a_id = self.node_id
        # 1.把今天的事件添加到记录list  self.contact_record_today
        self.contact_record_today.append((runningtime, a_id, b_id))
        # 2.day内处理; 整理pdf_GMM所需的数据集
        thm = runningtime.tm_hour + (runningtime.tm_min/60.0)
        # 2.1 按照 holiday or not, 分别存放
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
               self.all_pdf_holiday, self.all_pdf_workday, self.all_res_cal

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
            self.all_pdf_holiday[update_node] = other_probs_holiday[update_node]
            self.all_pdf_workday[update_node] = other_probs_workday[update_node]
            self.all_res_cal[update_node] = other_res_cal[update_node]
            self.all_UpdatingTime[update_node] = other_UpdatingTime[update_node]
