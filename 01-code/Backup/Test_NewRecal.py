import numpy as np
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

def __update_probdensity(self, runningtime):
    today_yday = runningtime.tm_yday
    for target_node in range(self.num_of_nodes):
        tmp_res_cal = np.zeros((self.num_of_nodes, (self.max_ttl.days+1)*24))
        # 1.按照每天的类别(是否holiday) 处理cond_prob
        res_list_betwday = []
        for i in range(self.max_ttl.days + 1):
            index = today_yday + i - 1
            if self.list_weather[index][3]:
                # [np.array(各个节点), np.array, np.array, ... ] 一共 ttl个
                res_list_betwday.append(self.all_P_holiday[target_node])
                tmp_res_cal[:,i*24:(i+1)*24]=self.all_pdf_holiday[target_node][:,:]
            else:
                res_list_betwday.append(self.all_P_workday[target_node])
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