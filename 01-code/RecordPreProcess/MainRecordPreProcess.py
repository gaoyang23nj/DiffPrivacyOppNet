import math
import os
import datetime
import numpy as np


nanjingbike_input_data = '../NanjingBikeDataset/2017RentReturnData'
# nanjingbike_interpolation = 'E:\\00-Trace\\01-shanghai_taxi\\taxi_100_interpolation'
EncoHistDir = '../EncoHistData_NJBike/data.csv'
EncoHistBadDir = '../EncoHistData_NJBike/data_bad.csv'

class RecordPreProcess(object):
    def __init__(self):
        # input_file_path
        # self.num_nodes = 100
        self.input_dir = nanjingbike_input_data
        self.output_dir = EncoHistDir
        self.outputbad_dir = EncoHistBadDir
        # 每10s保留一个位置
        self.sim_TimeStep = 1
        # id 文件 真实id对应关系
        self.list_id = []
        # 各个节点的时空位置
        self.list_space_time_loc = []
        # 也是node_id
        self.num_nodes = 0
        self.__preprocess()
        # 每10s一个 ？
        self.MAX_RUNNING_TIMES = 100
        output_obj = open(self.output_dir, 'w+', encoding="utf-8")
        badoutput_obj = open(self.outputbad_dir, 'w+', encoding="utf-8")
        for tunple in self.list_id:
            self.extractdata(output_obj, badoutput_obj, tunple)
        badoutput_obj.close()
        output_obj.close()

    def extractdata(self, output_obj, badoutput_obj, tunple):
        filepath = tunple[0]
        num_att = tunple[2]
        oneinput_obj = open(filepath, 'r', encoding="gbk")
        # 跳过表头
        oneinput_obj.readline()
        # 读文件
        while True:
            line = oneinput_obj.readline()
            if not line:
                break
            else:
                units = line.replace('"','').split(',')
                (biswelldefined, output_str) = self.parse_items(units, num_att)
                if biswelldefined:
                    output_obj.write(output_str)
                else:
                    badoutput_obj.write(output_str)

    def parse_items(self, units, num_att):
        if num_att == 46:
            itemlist = (units[6], units[7], units[14], units[15], units[41], units[42])
        elif num_att == 10:
            itemlist = (units[3], units[4], units[6], units[7], units[5], units[8])
        # check
        biswelldefined = True
        for item in itemlist:
            # 空字符串
            if len(item) == 0:
                biswelldefined = False
            # 未知站点
        output_str = '{},{},{},{},{},{}\n'.format(
            itemlist[0], itemlist[1],itemlist[2],itemlist[3],itemlist[4],itemlist[5])
        return (biswelldefined, output_str)

    def __preprocess(self):
        # 每个月一个文件
        file_list = os.listdir(self.input_dir)
        for file_name in file_list:
            month_id = file_name.split('.')[0]
            filepath = os.path.join(self.input_dir, file_name)
            oneinput_obj = open(filepath, 'r', encoding="gbk")
            tmpline = oneinput_obj.readline().split(',')
            num_att = len(tmpline)
            oneinput_obj.close()
            self.list_id.append((filepath, month_id, num_att))
        # for item in self.list_id:
        #     print(item)
        print(self.list_id)


if __name__=='__main__':
    RecordPreProcess()
    print('OK')