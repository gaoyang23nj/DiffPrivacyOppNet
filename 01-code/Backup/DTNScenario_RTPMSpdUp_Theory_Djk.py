# 进行路由metric比较时候，记下来当前的数值;
# 路径挖掘的时候, 先得到metric值,只有高于之前才继续进行
from Main.DTNNodeBuffer import DTNNodeBuffer
from Main.DTNPkt import DTNPkt
from Backup.RoutingRTPMSpdUp_Theory_Djk import RoutingRTPMSpdUp_Theory_Djk

import copy
import pandas as pd
import datetime
import time

# 用GMM描述 day内
# 用access方法描述 day间
EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data'
StationInfoPath = '../EncoHistData_NJBike/station_info.csv'
WeatherInfo = '../NanjingBikeDataset/Pukou_Weather.xlsx'

NUM_DAYS_INYEAR = 365

# Scenario 要响应 genpkt swappkt事件 和 最后的结果查询事件
class DTNScenario_RTPMSpdUp_Theory_Djk(object):
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
            tmpRouter = RoutingRTPMSpdUp_Theory_Djk(node_id, num_of_nodes, min_time, self.list_weather, self.max_ttl)
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

