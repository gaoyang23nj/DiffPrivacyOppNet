# 存在bug 更新方式
from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt

import copy
import numpy as np
import math
import datetime

#

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_SMART(object):
    # node_id的list routingname的list
    def __init__(self, scenarioname, num_of_nodes, buffer_size, min_time, max_ttl):
        self.scenarioname = scenarioname
        # 为各个node建立虚拟空间 <内存+router>
        self.listNodeBuffer = []
        self.listRouter = []
        for node_id in range(num_of_nodes):
            tmpRouter = RoutingSMART(node_id, num_of_nodes, min_time)
            self.listRouter.append(tmpRouter)
            tmpBuffer = DTNNodeBuffer(self, node_id, buffer_size, max_ttl)
            self.listNodeBuffer.append(tmpBuffer)
        self.num_comm = 0
        return

    def gennewpkt(self, pkt_id, src_id, dst_id, gentime, pkt_size):
        # print('senario:{} time:{} pkt_id:{} src:{} dst:{}'.format(self.scenarioname, gentime, pkt_id, src_id, dst_id))
        newpkt = DTNPkt(pkt_id, src_id, dst_id, gentime, pkt_size)
        self.listNodeBuffer[src_id].gennewpkt(gentime, newpkt)
        return

    # 通知每个节点更新自己的状态
    def notify_new_day(self, runningtime):
        pass

    # routing接到指令aid和bid相遇，开始进行消息交换a_id -> b_id
    def swappkt(self, runningtime, a_id, b_id):
        # ================== 控制信息 交换==========================
        # 单向操作!!!
        # 获取 b_node Router 向各节点的值(带有老化计算)
        # P_b_any = self.listRouter[b_id].get_values_before_up(runningtime)
        # 根据 b_node Router 保存的值, a_node更新向各其他node传递值 (带有a-b响应本次相遇的更新)
        self.listRouter[a_id].notifylinkup(runningtime, b_id)
        # ================== 报文 交换==========================
        self.sendpkt(runningtime, a_id, b_id)

    # 报文发送 a_id -> b_id
    def sendpkt(self, runningtime, a_id, b_id):
        friend_list_a = self.listRouter[a_id].getfriendlist()
        friend_list_b = self.listRouter[b_id].getfriendlist()

        W_a, degree_a = self.listRouter[a_id].get_values_before_up(runningtime, b_id, friend_list_b)
        W_b, degree_b = self.listRouter[b_id].get_values_before_up(runningtime, a_id, friend_list_a)
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
                cppkt = copy.deepcopy(a_pkt)
                if a_pkt.dst_id == b_id:
                    totran_pktlist.insert(0, cppkt)
                else:
                    totran_pktlist.append(cppkt)
                # break
        for tmp_pkt in totran_pktlist:
            # <是目的节点 OR P值更大> 才进行传输; 单播 只要传输就要删除原来的副本
            if tmp_pkt.dst_id == b_id:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1
                continue
            # np.inf == np.inf
            if W_b[tmp_pkt.dst_id] < W_a[tmp_pkt.dst_id]:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1
                continue
            if W_b[tmp_pkt.dst_id] == np.inf and W_a[tmp_pkt.dst_id] == np.inf and degree_b > degree_a:
                self.listNodeBuffer[b_id].receivepkt(runningtime, tmp_pkt)
                self.listNodeBuffer[a_id].deletepktbyid(runningtime, tmp_pkt.pkt_id)
                self.num_comm = self.num_comm + 1
                continue

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

class RoutingSMART(object):
    def __init__(self, node_id, num_of_nodes, min_time):
        self.node_id = node_id
        self.num_of_nodes = num_of_nodes
        self.min_time = min_time

        # the unit of calculating freq; e.g., 1 hour
        self.daysInTimeUnit = datetime.timedelta(days=1)

        # number of contacts list
        self.numContactsList = np.zeros(self.num_of_nodes, dtype='double')
        # frequency list
        self.freqList = np.zeros(self.num_of_nodes, dtype='double')
        # record friends
        self.friendList = []

        # record social map
        self.globalsocialmapList = []
        for i in range(self.num_of_nodes):
            self.globalsocialmapList.append([i, []])

        # self.socialmaplist = []
        self.W = np.ones(self.num_of_nodes, dtype='double') * np.inf


    # ===============================================  SMART内部逻辑  ================================
    # get rank with frequency
    def __convertorank(self, item):
        rank = 7
        if item > 2:
            rank = 1
        elif item > 1:
            rank = 2
        elif item > 0.5:
            rank = 3
        elif item > 0.25:
            rank = 4
        elif item > 0.125:
            rank = 5
        elif item > 0.0625:
            rank = 6
        else:
            rank = 7
        return rank

    # ========================= 提供给上层的功能 ======================================
    # 更新后, 提供 本node 的 信息(social map) 给对端
    def get_values_before_up(self, runningtime, other_id, other_friendlist):
        # update globalsocialmapList
        assert other_id == self.globalsocialmapList[other_id][0]
        self.globalsocialmapList[other_id][1] = other_friendlist
        # calculate W_{a,b}
        # merge map; calculate w_{a,b}; calculate W_{a,b}
        self.W = np.ones(self.num_of_nodes, dtype='double') * np.inf
        for target_node in range(self.num_of_nodes):
            w = 0.
            # find 1-hop path (or rank) from selfnode to target_node
            for tunple in self.friendList:
                (i_node, i_rank) = tunple
                if i_node == target_node:
                    w = 1 / i_rank
            # find 2-hop
            for tunple in self.friendList:
                (i_node, i_rank) = tunple
                assert i_node == self.globalsocialmapList[i_node][0]
                i_friendlist = self.globalsocialmapList[i_node][1]
                for item in i_friendlist:
                    (j_node, j_rank) = item
                    if j_node == target_node:
                        w = w + 1 / (i_rank + j_rank)
            self.W[target_node] = np.divide(1, w)
        # get degree
        degree = len(self.friendList)
        return self.W.copy(), degree

    # 当a->b 相遇(linkup时候) 更新a->b相应的值
    def notifylinkup(self, runningtime, b_id):
        # add new contact
        self.numContactsList[b_id] = self.numContactsList[b_id] + 1
        # update freq; how many times in 1 day ?
        delta_time = (runningtime - self.min_time) / self.daysInTimeUnit
        self.freqList = self.numContactsList / delta_time
        # update friend list
        self.friendList = []
        max_freq = np.max(self.freqList)
        for i_node in range(self.num_of_nodes):
            if self.freqList[i_node] > 0.5 * max_freq:
                # segment freq to get rank; !!!! now use freq directly
                i_rank = self.__convertorank(self.freqList[i_node])
                self.friendList.append((i_node, i_rank))
        # sort i_node with increasing rate
        self.friendList.sort()
        return

    # get friend list
    def getfriendlist(self):
        return self.friendList.copy()

