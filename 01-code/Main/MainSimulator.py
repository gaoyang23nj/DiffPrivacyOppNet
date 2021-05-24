# 执行仿真的主程序
import numpy as np
import datetime
import sys
import winsound
import os

from Main.Scenario.DTNScenario_EP import DTNScenario_EP
from Main.Scenario.DTNScenario_SandW import DTNScenario_SandW
from Main.Scenario.DTNScenario_Prophet import DTNScenario_Prophet

EncoHistDir = '../EncoHistData_NJBike/data.csv'
StationInfoPath = '../EncoHistData_NJBike/station_info.csv'
EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data'

class Simulator(object):
    def __init__(self, num_station, enco_file, pktgen_freq, result_file_path):
        # 相遇记录文件
        self.ENCO_HIST_FILE = enco_file
        # 汇总 实验结果
        self.result_file_path = result_file_path

        # 节点个数默认216个, id 0~215
        self.MAX_NODE_NUM = num_station
        # 最大运行时间 执行时间 36000*24个间隔, 即24hour; 应该根据 enco_hist 进行更新
        self.MIN_RUNNING_TIMES = datetime.datetime.strptime('2017/01/01 0:0:0', "%Y/%m/%d %H:%M:%S")
        self.MAX_RUNNING_TIMES = datetime.datetime.strptime('2017/01/01 0:0:0', "%Y/%m/%d %H:%M:%S")
        # 每个间隔的时间长度 0.1s
        # self.sim_TimeStep = 0.1
        # 仿真环境 现在的时刻
        self.sim_TimeNow = self.MIN_RUNNING_TIMES
        # 报文生成的间隔,即每60*20个时间间隔(60*20*1s 即20分钟)生成一个报文
        # self.THR_PKT_GEN_CNT = pktgen_freq
        self.GENPKT_DELTA_TIME = datetime.timedelta(seconds=pktgen_freq)
        # # node所组成的list
        # self.list_nodes = []
        # 生成报文的时间计数器 & 生成报文计算器的触发值
        # self.cnt_genpkt = self.THR_PKT_GEN_CNT
        # self.thr_genpkt = self.THR_PKT_GEN_CNT
        # 下一个pkt的id
        self.pktid_nextgen = 0
        # 全部生成报文的list
        self.list_genpkt = []
        # 读取文件保存所有的相遇记录; self.mt_enco_hist.shape[0] 表示记录个数
        # self.mt_enco_hist = np.empty((0, 0), dtype='int')
        self.list_enco_hist = []
        self.list_gen_eve = []
        # 读取相遇记录
        self.read_enco_hist_file()
        print('read enco file end!')
        print(datetime.datetime.now())
        # winsound.Beep(200, 500)
        self.build_gen_event()
        # 初始化各个场景 spamming节点的比例
        self.init_scenario()
        # 根据相遇记录执行 各场景分别执行路由
        short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = 'result_' + short_time + '.tmp'
        self.run()
        self.print_res(filename, ctstring='a+')

    def read_enco_hist_file(self):
        file_object = open(self.ENCO_HIST_FILE, 'r', encoding="utf-8")
        tmp_all_lines = file_object.readlines()
        tm1 = tmp_all_lines[0].split(',')[1]
        self.MIN_RUNNING_TIMES = datetime.datetime.strptime(tm1, "%Y/%m/%d %H:%M:%S")
        tm2 = tmp_all_lines[-1].split(',')[1]
        self.MAX_RUNNING_TIMES = datetime.datetime.strptime(tm2, "%Y/%m/%d %H:%M:%S")
        print(self.MIN_RUNNING_TIMES)
        print(self.MAX_RUNNING_TIMES)
        for index in range(len(tmp_all_lines)):
            # 读取相遇记录
            (i_station_id, a_time, i_node, j_station_id, b_time, j_node, i_name, j_name) \
                = tmp_all_lines[index].strip().split(',')
            tm = datetime.datetime.strptime(a_time, "%Y/%m/%d %H:%M:%S")
            i_node = int(i_node)
            j_node = int(j_node)
            self.list_enco_hist.append((tm, i_node, j_node))
        file_object.close()

    def build_gen_event(self):
        tmp_time = self.MIN_RUNNING_TIMES
        while True:
            gen_time = self.MIN_RUNNING_TIMES + self.GENPKT_DELTA_TIME
            if gen_time > self.MAX_RUNNING_TIMES:
                break
            (src_index, dst_index) = self.__gen_pair_randint(self.MAX_NODE_NUM)
            self.list_gen_eve.append((gen_time, self.pktid_nextgen, src_index, dst_index))
            self.pktid_nextgen = self.pktid_nextgen + 1

    def run(self):
        while self.sim_TimeNow <= self.MAX_RUNNING_TIMES:
            if len(self.list_gen_eve) == 0 and len(self.list_enco_hist) == 0:
                break
            gen_time = sys.maxsize
            enco_time = sys.maxsize
            if len(self.list_gen_eve)>0:
                gen_time = self.list_gen_eve[0][0]
            if len(self.list_enco_hist)>0:
                enco_time = self.list_enco_hist[0][0]
            if gen_time <= enco_time:
                self.sim_TimeNow = gen_time
                # 执行报文生成
                # controller记录这个pkt
                self.list_genpkt.append((self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3]))
                # 各scenario生成pkt, pkt大小为100k
                print('time:{} pkt_id:{} src:{} dst:{}'.format(self.sim_TimeNow, self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3]))
                for key, value in self.scenaDict.items():
                    value.gennewpkt(self.list_gen_eve[0][1], self.list_gen_eve[0][2], self.list_gen_eve[0][3], self.sim_TimeNow, 500)
                # 删除这个生成事件 以便继续进行
                self.list_gen_eve.pop(0)
            if gen_time >= enco_time:
                self.sim_TimeNow = enco_time
                # 执行相遇事件list 同一时刻可能有多个相遇事件
                tmp_enc = self.list_enco_hist[0][1:]
                for enc_eve in tmp_enc:
                    for key, value in self.scenaDict.items():
                        value.swappkt(self.sim_TimeNow, enc_eve[2], enc_eve[3])
                        # value.swappkt(self.sim_TimeNow, enc_eve[3], enc_eve[2])
                self.list_enco_hist.pop(0)
        assert(len(self.list_gen_eve)==0 and len(self.list_enco_hist)==0)

    def __gen_pair_randint(self, int_range):
        src_index = np.random.randint(int_range)
        dst_index = np.random.randint(int_range-1)
        if dst_index >= src_index:
            dst_index = dst_index + 1
        return (src_index, dst_index)

    def init_scenario(self):
        self.scenaDict = {}
        # list_scena = self.init_scenario_testEric()
        list_scena = self.init_scenario_testProphet()
        # list_scena = self.init_scenario_testOur()
        return list_scena

    def init_scenario_testv3(self):
        index = -1
        # ===============================场景1 Prophet ===================================
        index += 1
        tmp_senario_name = 'scenario' + str(index) + '_Prophet'
        tmpscenario = DTNScenario_EP(tmp_senario_name, self.MAX_NODE_NUM, 20000)
        self.scenaDict.update({tmp_senario_name: tmpscenario})

        # ===============================场景单个单个的实验吧===================================
        list_scena = list(self.scenaDict.keys())
        return list_scena

    # 打印出结果
    def print_res(self, filename, ctstring):
        # 防止numpy转化时候换行
        np.set_printoptions(linewidth=200)

        res_file_object = open(self.result_file_path, ctstring, encoding="utf-8")
        res_file_object.write('gen_freq, delivery ratio, avg delivery delay, graynodes ratio\n')

        file_object = open(filename, ctstring, encoding="utf-8")
        gen_total_num = len(self.list_genpkt)
        file_object.write('genfreq:{} RunningTime_Min_Max:{};{} '
                          'gen_num:{} nr_nodes:{}\n '.format(
            self.GENPKT_DELTA_TIME, self.MIN_RUNNING_TIMES, self.MAX_RUNNING_TIMES,
            gen_total_num, self.MAX_NODE_NUM))

        for key, value in self.scenaDict.items():
            outstr, res, config = value.print_res(self.list_genpkt)
            file_object.write(outstr+'\n')

            res_file_object.write(str(self.GENPKT_DELTA_TIME)+',')
            # 3个res 是 res = {'succ_ratio':succ_ratio, 'avg_delay':avg_delay, 'num_comm':num_comm}
            # 5个res res = {'succ_ratio': succ_ratio, 'avg_delay': avg_delay, 'num_comm': num_comm,
            #                'DetectResult':self.DetectResult, 'tmp_DetectResult':self.tmp_DetectResult}
            # config = {'ratio_bk_nodes': ratio_bk_nodes, 'drop_prob': 1}
            assert((len(res) == 3) or (len(res)==5))
            res_file_object.write(str(res['succ_ratio'])+','+str(res['avg_delay'])+','+str(res['num_comm'])+',')
            res_file_object.write(str(config['ratio_bk_nodes']) + ',' +str(config['drop_prob']))
            if len(res) == 5:
                res_file_object.write('\n' + str(res['DetectResult']) + '\n' + str(res['tmp_DetectResult']) + ',')
            res_file_object.write('\n')

        file_object.close()
        res_file_object.write('\n')
        res_file_object.close()

if __name__ == "__main__":
    t1 = datetime.datetime.now()
    print(datetime.datetime.now())

    simdurationlist = []

    # result file path
    short_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    result_file_path = "res_blackhole_" + short_time + ".csv"

    # genpkt_freqlist = [10 * 30, 10 * 60, 10 * 90, 10 * 120, 10 * 150]
    # 6个hours
    genpkt_freqlist = [60*60*6]
    for i in range(5):
        for genpkt_freq in genpkt_freqlist:
            print(EncoHistDir, genpkt_freq)
            t_start = datetime.datetime.now()
            theSimulator = Simulator(216, EncoHistDir, genpkt_freq, result_file_path)
            t_end = datetime.datetime.now()
            print('running time:{}'.format(t_end - t_start))
            simdurationlist.append(t_end - t_start)
    t2 = datetime.datetime.now()
    print(datetime.datetime.now())

    winsound.Beep(500, 2000)
    print(t1)
    print(t2)
    print('running time:{}'.format(t2 - t1))
    print(simdurationlist)