# 用GMM描述 day内
# 用access方法描述 day间
# 结合了RGMM和memo的方案
import time
import datetime
import numpy as np
from sklearn import mixture
import math
import sys
from scipy.stats import laplace
from cvxopt import matrix, solvers

EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data'
StationInfoPath = '../EncoHistData_NJBike/station_info.csv'
WeatherInfo = '../NanjingBikeDataset/Pukou_Weather.xlsx'

NUM_DAYS_INYEAR = 365

class RoutingRTPMSpdUp_Theory_Djk_OpDP(object):
    def __init__(self, node_id, num_of_nodes, min_time, max_time, input_list_weather, max_ttl, lap_noise_scale):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.MIN_TIME = time.strptime(min_time.strftime('%Y/%m/%d %H:%M:%S'), "%Y/%m/%d %H:%M:%S")
        # 从datetime结构 转化为time结构
        self.lastAgeUpdate = self.MIN_TIME
        self.list_weather = input_list_weather
        self.max_ttl = max_ttl

        # laplace noise的scale参数, 应该作为输入参数
        # 天内概率
        self.LapNoiseScale = lap_noise_scale[0]
        # 天间概率
        self.LapNoiseScale_Pinter = lap_noise_scale[1]
        # log的底数
        self.for_log = 0.5
        # hard TTL
        self.MAX_TIME = max_time
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
        # 保存真实 intra-p值
        self.probs_holiday = np.zeros((self.num_of_nodes, 24))
        self.probs_workday = np.zeros((self.num_of_nodes, 24))
        # 保存真实 inter-P值
        self.Ptrue_holiday = np.zeros(self.num_of_nodes)
        self.Ptrue_workday = np.zeros(self.num_of_nodes)

        # 3.用于各个节点之间的信息交换
        # 对于self.all_pdf_holiday[self.node_id], 保存带lap noise的intra-p值(p'=p+lap); 其他self.all_pdf_holiday[*]来自于复制更新
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
        # self.list_metric_memo = []

        # 5.Opt+Dp
        self.delta = 0.05
        self.length = np.math.floor(1./self.delta)
        self.ProbValueBucket = np.zeros(self.length)
        self.pdfValueBucket = np.zeros(500)
        self.set_st()

    # 做成全局的变量
    def set_st(self):
        epsP = 1./self.LapNoiseScale_Pinter
        W = np.exp(epsP)
        print('set s.t. ...')
        # 由a c 导出 tranP转成行矩阵对应的位置 c*11+a
        num_st1 = 0
        list_st1 = []
        # s.t.1 隐私保护带来的限制条件
        for c in range(self.length):
            # 选取a,b
            for a in range(self.length):
                for b in range(a + 1, self.length):
                    num_st1 = num_st1 + 1
                    list_st1.append((c, a, b))
        num_st1 = num_st1 * 2
        # print(num_st1)
        num_st2 = 0
        list_st2 = []
        # s.t.2 prob的天然条件 [0,1]
        for a in range(self.length):
            # 选取a,b
            for c in range(self.length):
                num_st2 = num_st2 + 1
                list_st2.append((c, a))
        num_st2 = num_st2 * 2
        # print(num_st2)
        num_neq_st = num_st1 + num_st2
        list_st3 = []
        num_st3 = 0
        for a in range(self.length):
            # 选取a,b
            num_st3 = num_st3 + 1
            list_st3.append(a)
        # print(num_st3)
        # 不等式条件，矩阵
        label = 0
        Gnp = np.zeros((num_neq_st, self.length * self.length))
        hnp = np.zeros((num_neq_st, 1))
        for ele in list_st1:
            (c, a, b) = ele
            # p_{a,c} - W p_{b,c} <=0
            Gnp[label, a * self.length + c] = 1.
            Gnp[label, b * self.length + c] = -W
            label = label + 1
            # -W p_{a,c} + p_{a,c} <=0
            Gnp[label, a * self.length + c] = -W
            Gnp[label, b * self.length + c] = 1.
            label = label + 1
        for ele in list_st2:
            (c, a) = ele
            # -p_{a,c} <= 0
            Gnp[label, a * self.length + c] = -1.
            label = label + 1
            # p_{a, c} <= 1
            Gnp[label, a * self.length + c] = 1.
            hnp[label, 0] = 1.
            label = label + 1
        # print(label)
        # 等式条件，矩阵
        Anp = np.zeros((num_st3, self.length * self.length))
        bnp = np.ones((num_st3, 1))
        label = 0
        for ele in list_st3:
            (a) = ele
            Anp[label, a * self.length:(a + 1) * self.length] = 1.
            label = label + 1
        # print(label)
        self.Gcx = matrix(Gnp)
        self.hcx = matrix(hnp)
        self.Acx = matrix(Anp)
        self.bcx = matrix(bnp)
        return

    # Optimization DP; 根据先验概率 和 差分隐私的要求, 计算最优的概率转移矩阵
    def optProb(self, PrioProb):
        print('set obj ...')
        # Pr(A) * p_{a,c} * (从a变成c偏差)
        cnp = np.zeros((self.length * self.length, 1))
        label = 0
        def getCost(c, a):
            return (self.delta * (c+0.5) - self.delta * (a+0.5)) ** 2
        for a in range(self.length):
            for c in range(self.length):
                cnp[label, 0] = PrioProb[a] * getCost(c, a)
                label = label + 1

        ccx = matrix(cnp)
        print('cal ...')
        sol = solvers.lp(ccx, self.Gcx, self.hcx, self.Acx, self.bcx)
        xSolution = np.array(sol['x'])
        tranP = xSolution.reshape((self.length, self.length))
        return tranP

    # =======================  用于每日更新notify_new_day的内部函数     ====================
    # 1)day间, rho_star 通过每天的contact次数转换得到;
    # 2)EWEA方法 平滑 条件概率P
    def __process_Prob_interday(self, update_y_day):
        err = 0.
        isholiday = self.list_weather[update_y_day-1][3]
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

        # 对self.P_holiday[-1][1] self.P_workday[-1][1]进行差分隐私扰动
        # 扰动后的结果放在self.all_P_holiday[self.node_id] = tmp 和 self.all_P_workday[self.node_id]
        # 1.整理获取先验概率
        if isholiday:
            self.Ptrue_holiday = self.P_holiday[-1][1]
            tmpP = self.P_holiday[-1][1].copy()
        else:
            self.Ptrue_workday = self.P_workday[-1][1]
            tmpP = self.P_workday[-1][1].copy()
        for i in range(self.num_of_nodes):
            loc = np.math.floor(tmpP[i]/self.delta)
            if loc == self.length:
                loc = self.length - 1
            self.ProbValueBucket[loc] = self.ProbValueBucket[loc] + 1
        PrioProb = self.ProbValueBucket / np.sum(self.ProbValueBucket)

        tranP = self.optProb(PrioProb)
        OutputProb = np.zeros(self.num_of_nodes)
        once = np.random.random(self.num_of_nodes)
        for i in range(self.num_of_nodes):
            loc = np.math.floor(tmpP[i] / self.delta)
            tmp = 0.
            if loc == self.length:
                loc = self.length-1
            for target in range(tranP.shape[1]):
                if once[i] > tmp and once[i] < tmp + tranP[loc, target]:
                    break
                tmp = tmp + tranP[loc, target]
            OutputProb[i] = (target + 0.5) * self.delta
            err = err + (OutputProb[i] - tmpP[i])**2
        if isholiday:
            self.all_P_holiday[self.node_id] = OutputProb.copy()
        else:
            self.all_P_workday[self.node_id] = OutputProb.copy()
        print('OpDP node_{}: err:{}'.format(self.node_id, err))

    # day内, 执行GMM; intra-day probability density:GMM参数/24小时概率串
    def __process_pdf_intraday_withGMM(self):
        tmp_lap_probs_holiday = np.zeros((self.num_of_nodes, 24))
        tmp_lap_probs_workday = np.zeros((self.num_of_nodes, 24))
        # 从self.node_id 到 每个b_id 都做一次基于GMM的day内预测(intra-day probability)
        for b_id in range(self.num_of_nodes):
            if b_id == self.node_id:
                continue
            dataset1 = np.array(self.dataset_holiday[b_id]).reshape(-1,1)
            # dataset小于一定个数 GMM无法正常启动
            if dataset1.shape[0] >= self.MIN_SAMPLES:
                probs_h, tunple_b_h, probs_h_lap = self.__precdit_GMM(dataset1)
            else:
                probs_h = np.zeros(24)
                probs_h_lap = np.zeros(24)
                tunple_b_h = (np.zeros(3),np.zeros(3),np.zeros(3))
            dataset2 = np.array(self.dataset_workday[b_id]).reshape(-1,1)
            if dataset2.shape[0] >= self.MIN_SAMPLES:
                probs_w, tunple_b_w, probs_w_lap = self.__precdit_GMM(dataset2)
            else:
                probs_w = np.zeros(24)
                probs_w_lap = np.zeros(24)
                tunple_b_w = (np.zeros(3),np.zeros(3),np.zeros(3))
            # self.list_paramsGMM_holiday[b_id] = tunple_b_h
            # self.list_paramsGMM_workday[b_id] = tunple_b_w
            # 把24小时概率串放到 对应的b_id里
            # if np.sum(probs_h) > 0.1:
            #     probs_h = probs_h / np.sum(probs_h)
            # if np.sum(probs_w) > 0.1:
            #     probs_w = probs_w / np.sum(probs_w)
            self.probs_holiday[b_id, :] = probs_h
            self.probs_workday[b_id, :] = probs_w
            tmp_lap_probs_holiday[b_id, :] = probs_h_lap
            tmp_lap_probs_workday[b_id, :] = probs_w_lap
        # 更新 self.all_pdf_holiday 和 self.all_pdf_workday
        self.all_pdf_holiday[self.node_id] = tmp_lap_probs_holiday
        self.all_pdf_workday[self.node_id] = tmp_lap_probs_workday

    # 更新all_res_cal矩阵, 评价从self.node_id到各个节点的直接评价
    # 生成今天的ij概率序列（q_{1}^{k}...q_{24}^{k},q_{1}^{k+1}...q_{24}^{k+1}）
    def __update_probdensity(self, runningtime):
        # print('update at {}-{}-{}, in node {}, number_of_nodes {}'.format(
        #     runningtime.tm_year, runningtime.tm_mon, runningtime.tm_mday,
        #     self.node_id, self.num_of_nodes))
        today_yday = runningtime.tm_yday
        tmp_res_cal = np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))
        for target_node in range(self.num_of_nodes):
            # 1.按照每天的类别(是否holiday) 处理cond_prob
            res_list_betwday = []
            for i in range(self.max_ttl.days + 1):
                index = today_yday + i - 1
                if self.list_weather[index][3]:
                    # [np.array(各个节点), np.array, np.array, ... ] 一共 ttl个
                    if target_node == self.node_id:
                        res_list_betwday.append(self.Ptrue_holiday.copy())
                    else:
                        res_list_betwday.append(self.all_P_holiday[target_node])
                    # 本地节点使用真实值 其他节点使用交换的值
                    if target_node == self.node_id:
                        tmp_res_cal[:,i*24:(i+1)*24] = self.probs_holiday.copy()
                    else:
                        tmp_res_cal[:,i*24:(i+1)*24] = self.all_pdf_holiday[target_node][:,:]
                else:
                    if target_node == self.node_id:
                        res_list_betwday.append(self.Ptrue_workday.copy())
                    else:
                        res_list_betwday.append(self.all_P_workday[target_node])
                    # 本地节点使用真实值 其他节点使用交换的值
                    if target_node == self.node_id:
                        tmp_res_cal[:,i*24:(i+1)*24]=self.probs_workday.copy()
                    else:
                        tmp_res_cal[:,i*24:(i+1)*24]=self.all_pdf_workday[target_node][:,:]
            res_list_betwday = np.array(res_list_betwday)

            # ttl.days * num_nodes 第几天/发往哪个节点
            cond_P = self.__cal_cond_prob(res_list_betwday)
            # 各个对端节点 * 14days
            cond_P = cond_P.transpose()
            tmp = np.repeat(cond_P, 24, axis=1)
            tmp_res_cal = np.multiply(tmp_res_cal, tmp)
            # 3.更新到all里面
            self.all_res_cal[target_node] = tmp_res_cal.copy()
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
        # add Laplace Noise
        lap_noise = laplace.rvs(loc=0., scale=self.LapNoiseScale, size=24)
        probs_lap = probs + lap_noise
        for i in range(24):
            if probs_lap[i] < 0:
                probs_lap[i] = 0.
        # normalize
        default = np.power(0.1, 10)
        if np.sum(probs)>default:
            # print('sum probs:{} probs_lap:{}'.format(np.sum(probs), np.sum(probs_lap)))
            nm_probs = probs / np.sum(probs)
        else:
            nm_probs = np.zeros(24)

        if np.sum(probs_lap)>default:
            nm_probs_lap = probs_lap / np.sum(probs_lap)
        else:
            nm_probs_lap = np.zeros(24)

        # print(Z)
        # print(probs)
        # print(sum_probs)
        return nm_probs, (clf.weights_, clf.means_, clf.covariances_), nm_probs_lap

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
        # self.list_metric_memo.clear()
        # 更新处理时间
        self.lastAgeUpdate = runningtime

    # 从本节点出发 不经过nby_nodeid 从而到达pktdst_id的 各种路径所提供概率的最大值
    def get_values_before_up(self, runningtime, gentime, pktdst_id):
        tmp_end = gentime + self.max_ttl
        # 如果实验已经快要结束了 取结束时间为终止条件
        if self.MAX_TIME < tmp_end:
            tmp_end = self.MAX_TIME
        delta_time = tmp_end - runningtime
        nr_hours = math.ceil(delta_time.total_seconds()/3600)
        runningtime = time.strptime(runningtime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "%Y/%m/%d %H:%M:%S")
        curr_tmhour = runningtime.tm_hour
        # 1.计算各跳权重（从seq计算得到）
        matrix = np.ones((self.num_of_nodes, self.num_of_nodes)) * sys.float_info.max
        for o_id in range(self.num_of_nodes):
            for d_id in range(self.num_of_nodes):
                if o_id == d_id:
                    matrix[o_id, d_id] = 0
                    continue
                tmp = np.sum(self.all_res_cal[o_id][d_id, curr_tmhour:curr_tmhour+nr_hours])
                if tmp>1.:
                    # print('\033[1;35;46m DJK 必须 保证权重为正数 \033[0m')
                    tmp = 1.
                # 如果tmp的值太小（概率很小 接近0）？
                matrix[o_id, d_id] = np.log(tmp)/np.log(self.for_log)
        # 2.Djk算法,推定最短距离(最大概率)
        # 记录距离
        dis = np.ones((self.num_of_nodes))*sys.float_info.max
        # 记录上一个节点
        pre = np.ones((self.num_of_nodes),dtype='int')*-1
        # 记录是否访问过 1表示访问过
        vis = np.ones((self.num_of_nodes),dtype='int')*0
        # 2.1 init
        for i in range(self.num_of_nodes):
            dis[i] = matrix[self.node_id, i]
            pre[i] = self.node_id
        vis[self.node_id] = 1
        count = 1
        while count != self.num_of_nodes:
            # 选取dis最小的下标
            tmp_idx = -1
            min = sys.float_info.max
            for i in range(self.num_of_nodes):
                if vis[i]!=1 and dis[i]<min:
                    min = dis[i]
                    tmp_idx = i
            vis[tmp_idx] = 1
            count = count + 1
            for i in range(self.num_of_nodes):
                if vis[i]!=1 and matrix[tmp_idx,i]!=sys.float_info.max and dis[tmp_idx]+matrix[tmp_idx][i]<dis[i]:
                    dis[i]=dis[tmp_idx]+matrix[tmp_idx][i]
                    pre[i]=tmp_idx
        # 值 tmp = log(G)/log(0.5), 求出G
        tmp = dis[pktdst_id]
        max_value = np.exp(tmp*np.log(self.for_log))
        # 取出路径
        path = [pktdst_id]
        cur = pktdst_id
        while cur != self.node_id:
            # 中断掉了 续不上
            if pre[cur] == -1:
                path = []
                break
            path.insert(0, pre[cur])
            cur = pre[cur]
        return max_value, path

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
        # return self.all_UpdatingTime, self.all_P_holiday, self.all_P_workday, \
        #        self.all_pdf_holiday, self.all_pdf_workday, self.all_res_cal
        return self.all_UpdatingTime, self.all_P_holiday, self.all_P_workday, \
               self.all_pdf_holiday, self.all_pdf_workday

    def updateInfo(self, info_othernode):
        # other_UpdatingTime, other_P_holiday, other_P_workday, \
        # other_probs_holiday, other_probs_workday, other_res_cal = info_othernode
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
            self.all_pdf_holiday[update_node] = other_probs_holiday[update_node]
            self.all_pdf_workday[update_node] = other_probs_workday[update_node]
            # self.all_res_cal[update_node] = other_res_cal[update_node]
            self.all_UpdatingTime[update_node] = other_UpdatingTime[update_node]
