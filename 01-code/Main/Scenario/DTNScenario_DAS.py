# 存在bug 更新方式
from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt, DTNPktSandW

import copy
import numpy as np
import math
import datetime

#2022 Wireless Netw; Destination-aware metric based social routing for mobile opportunistic networks

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_DAS(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, num_of_nodes, buffer_size, min_time, max_ttl, init_token):
        self.scenarioname = scenarioname
        self.inittoken = init_token
        self.max_ttl = max_ttl
        self.num_of_nodes = num_of_nodes

        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        self.listRouter = []
        for node_id in range(num_of_nodes):
            tmpRouter = RoutingDAS(node_id, num_of_nodes, min_time)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size, max_ttl)
            self.listNodeBuffer.append(tmpBuffer)
        self.num_comm = 0


        # record social map
        self.global_contacts = np.zeros((num_of_nodes, num_of_nodes))
        # 0 disconnect; >=1 connect
        self.global_isconnect = np.zeros((num_of_nodes, num_of_nodes), dtype='int')

        self.matrix_DBC_i_d = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype='double')

        return

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPktSandW(pkt_id, src_id, dst_id, gentime, pkt_size, self.inittoken)
        self.listNodeBuffer[src_id].gennewpkt(gentime, newpkt)
        return

    # 通知每个节点更新自己的状态
    def notify_new_day(self, runningtime):

        label_change = False
        for a_id in range(self.num_of_nodes):
            for b_id in range(self.num_of_nodes):
                if a_id != b_id:
                    # there is a change from disconnect to connect
                    # if from disconnect to connnect or and conctacs >= 2?, update DBC
                    if self.global_contacts[a_id, b_id] > 10 and self.global_isconnect[a_id, b_id] == 0:
                        self.global_isconnect[a_id, b_id] = 1
                        label_change = True
        if label_change:
            print('begin, notify new day, update DBC at {}'.format(datetime.datetime.now()))
            self.__updateDBC()
            print('end, notify new day, update DBC at {}'.format(datetime.datetime.now()))

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # ================== 控制信息 交换==========================
        # 单向操作!!!
        # 获取 b_node Router 向各节点的值(带有老化计算)
        # P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        # 根据 b_node Router 保存的值, a_node更新向各其他node传递值 (带有a-b响应本次相遇的更新)
        self.listRouter[a_id].notifylinkup(runningtime, b_id)
        self.global_contacts[a_id, b_id] = self.global_contacts[a_id, b_id] + 1

        # # there is a change from disconnect to connect
        # # if from disconnect to connnect or and conctacs >= 2?, update DBC
        # if self.global_contacts[a_id, b_id]>0 and self.global_isconnect[a_id, b_id] == 0:
        #     self.global_isconnect[a_id, b_id] = 1
        #     self.__updateDBC()
        # ================== 报文 交换==========================
        self.sendpkt(runningtime, a_id, b_id)

    # 内部函数
    def __findshortestpath(self, s_id, d_id):
        # num_of_nodes = self.num_of_nodes
        # connect_matrix = self.global_isconnect
        list_pop_node = []
        num_Between = np.zeros(self.num_of_nodes)
        tmp_queue = [(s_id,0,[])]
        min_depth_for_sdpair = -1
        is_get = False
        count_path = 0
        while len(tmp_queue) > 0 and is_get == False:
            (old_node, old_depth, old_list) = tmp_queue.pop(0)
            list_pop_node.append(old_node)
            for new_id in range(self.num_of_nodes):
                if self.global_isconnect[old_node, new_id] == 1:
                    # duplicate in old_list, or duplicate to s_id,
                    if new_id == s_id or new_id in old_list or new_id in list_pop_node:
                        continue
                    elif new_id == d_id:
                        min_depth_for_sdpair = old_depth + 1
                        # first shortest path
                        for n in old_list:
                            # s_id, old_list, d_id
                            num_Between[n] = num_Between[n] + 1
                        # print(s_id, old_list, d_id)
                        count_path = count_path + 1
                        # the other shortest path
                        for tunple in tmp_queue:
                            (old_last_node, old_last_depth, old_last_list) = tunple
                            if old_last_depth == old_depth and self.global_isconnect[old_last_node, d_id] == 1:
                                for n in old_last_list:
                                    # s_id, old_list, d_id
                                    num_Between[n] = num_Between[n] + 1
                                # print(s_id, old_last_list, d_id)
                                count_path = count_path + 1
                        is_get = True
                        break
                    else:
                        new_list = old_list.copy()
                        new_list.append(new_id)
                        tmp_queue.append((new_id, old_depth + 1, new_list))
        # print(count_path)
        # print(num_Between)
        return num_Between, count_path

    def __updateDBC(self):
        # only update when change
        # only if new connection, DBC update
        # from s_id to d_id, i_id inter
        self.matrix_DBC_i_d = np.zeros((self.num_of_nodes, self.num_of_nodes), dtype='double')
        for s_id in range(self.num_of_nodes):
            # print('update DBC for s_id:{}'.format(s_id))
            # find d_id
            candidate_target_d = []
            for d_id in range(self.num_of_nodes):
                # d_id must not be directly connected
                if self.global_isconnect[s_id, d_id] == 0 and d_id != s_id:
                    # direct connect
                    candidate_target_d.append(d_id)
            # find i_id, from s_id to d_id
            for d_id in candidate_target_d:
                # print('update DBC for s_id:{} d_id:{}'.format(s_id, d_id))
                num_Between_s_d, count_path_s_d = self.__findshortestpath(s_id, d_id)
                if count_path_s_d > 0:
                    DBC_s_i_d = np.divide(num_Between_s_d, count_path_s_d)
                else:
                    DBC_s_i_d = np.zeros(self.num_of_nodes)
                self.matrix_DBC_i_d[:,d_id] = self.matrix_DBC_i_d[:,d_id] + DBC_s_i_d.transpose()
        # print(self.matrix_DBC_i_d)

    # 报文发送 a_id -> b_id
    def sendpkt(self, runningtime, a_id, b_id):
        # 准备从a到b传输的pkt 组成的list<这里保存的是deepcopy>
        totran_pktlist = []
        b_listpkt_hist = self.listNodeBuffer[b_id].getlistpkt_hist()
        a_listpkt_hist = self.listNodeBuffer[a_id].getlistpkt_hist()
        # b_listpkt_hist = []
        # a_listpkt_hist = []
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
                # update token due to expected delay
                # we do not know the expected delay, so we use 0.75*ttl
                if (runningtime - a_pkt.gentime) >= 0.75 * self.max_ttl:
                    a_pkt.token = 1
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                else:
                    totran_pktlist.append(cppkt)
                break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            changed_token = math.floor(tmp_pkt.token/2)
            tmp_pkt.token = changed_token
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1
            else:
                # print('from {} to {}, pkt_id:{}'.format(a_id, b_id, tmp_pkt.pkt_id))
                if self.matrix_DBC_i_d[b_id, tmp_pkt.dst_id] > self.matrix_DBC_i_d[a_id, tmp_pkt.dst_id] and changed_token > 0:
                    self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                    self.listNodeBuffer[a_id].decrease_token(runningtime, tmp_pkt.pkt_id, changed_token)
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
            avg_delay = avg_delay.total_seconds()
            output_str += 'succ_ratio:{} avg_delay:{} '.format(succ_ratio, avg_delay)
        else:
            avg_delay = ()
            output_str += 'succ_ratio:{} avg_delay:null '.format(succ_ratio)
        output_str += 'num_comm:{}\n'.format(self.num_comm)
        output_str += 'total_hold:{} total_gen:{}, total_succ:{}\n'.format(total_pkt_hold, len(listgenpkt), total_succnum)
        print(output_str)
        res = {'succ_ratio': succ_ratio, 'avg_delay': avg_delay, 'num_comm': self.num_comm,
               'num_gen':len(listgenpkt), 'num_succ':total_succnum, 'gen_freq': 0}
        config = {}
        return output_str, res, config

class RoutingDAS(object):
    def __init__(self, node_id, num_of_nodes, min_time):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.min_time = min_time

        # number of contacts list
        # self.numContactsList = np.zeros(self.num_of_nodes, dtype='double')


    # ===============================================  SMART内部逻辑  ================================

    # ========================= 提供给上层的功能 ======================================
    # 更新后, 提供 本node 的 信息(social map) 给对端
    def get_values_before_up(self, runningtime, contactNum):
        # get DBC; betweenness centrality (DBC),
        # if np.sum(contactNum - self.globalcontacts)>0:
        #     self.globalcontacts = contactNum
        # for i in range(self.num_of_nodes):
        #     for j in range(self.num_of_nodes):
        #         if i == j:
        #             continue
        #         # there is a change from disconnect to connect
        #         # if from disconnect to connnect or and conctacs >= 2?, update DBC
        #         if self.globalcontacts[i,j]>0 and self.global_isconnect[i,j] == 0:
        #             self.global_isconnect[i, j] = 1
        #             self.__updateDBC()
        return

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id):
        # # add new contact
        # # self.numContactsList[b_id] = self.numContactsList[b_id] + 1
        # self.globalcontacts[self.node_id, b_id] = self.globalcontacts[self.node_id, b_id] + 1
        return

