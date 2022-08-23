# 预处理Nanjing Bike Station 数据
# 1.过滤未知站点 生成'../EncoHistData_NJBike/data.csv'
# 2.按照Src-Dst Pair 生成 '../EncoHistData_NJBike/SDPair_NJBike_Data'
# 3.每个文件按照时间顺序排列

# 生成 只包含 Pukou的数据集合！！！恢复最早以前的方案
import os
import pandas as pd
import time
import datetime

NanjingBike_InputData = '../NanjingBikeDataset/2017RentReturnData'
Station1Info = '../NanjingBikeDataset/Pukou.xlsx'
# Station2Info = '../NanjingBikeDataset/Qiaobei.xlsx'
EncoHistDir = '../EncoHistData_NJBike/data_pukou.csv'
EncoHistBadDir = '../EncoHistData_NJBike/data_bad_pukou.csv'
StationInfoPath = '../EncoHistData_NJBike/station_info_pukou.csv'

EncoHistDir_SDPair = '../EncoHistData_NJBike/SDPair_NJBike_Data_pukou'

# 该class解析文件
class RecordPreProcess(object):
    def __init__(self):
        # input_file_path
        # self.num_nodes = 100
        self.input_dir = NanjingBike_InputData
        self.output_dir = EncoHistDir
        self.outputbad_dir = EncoHistBadDir
        self.station_info_path = StationInfoPath

        # 每10s保留一个位置
        self.sim_TimeStep = 1
        # 每10s一个 ？
        self.MAX_RUNNING_TIMES = 100

        # station id list
        self.list_station_id = []
        # station id, inner id, name, list
        self.list_stationinfo_id = []
        self.count = 0
        self.read_station_info(Station1Info)
        # self.read_station_info(Station2Info)

        self.list_instation_id = []
        self.list_outstation_id = []
        # id 文件 真实id对应关系
        self.list_file_id = []
        self.__preprocess()
        output_obj = open(self.output_dir, 'w+', encoding="utf-8")
        badoutput_obj = open(self.outputbad_dir, 'w+', encoding="utf-8")
        # 提取每月的数据
        for tunple in self.list_file_id:
            self.extractdata(output_obj, badoutput_obj, tunple)
        badoutput_obj.close()
        output_obj.close()

        # 写入station_info文件
        station_info_obj = open(self.station_info_path, 'w+', encoding="utf-8")
        for tmp_station_info in self.list_stationinfo_id:
            station_info_obj.write('{},{},{}\n'.format(tmp_station_info[0],tmp_station_info[1],tmp_station_info[2]))
        station_info_obj.close()
        print(len(self.list_station_id))
        print(self.list_station_id)
        print(len(self.list_instation_id))
        print(len(self.list_outstation_id))
        print(self.list_instation_id)
        print(self.list_outstation_id)

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
                (biswelldefined, output_str) = self.__parse_items(units, num_att)
                if biswelldefined:
                    output_obj.write(output_str)
                else:
                    badoutput_obj.write(output_str)

    # 顺序 (起始站点id, 起始时间, 到达站点id, 到达时间, 起始站点名字, 到达站点名字, )
    def __parse_items(self, units, num_att):
        if num_att == 46:
            itemlist = [units[6], units[7], units[14], units[15], units[41], units[42]]
        elif num_att == 10:
            itemlist = [units[3], units[4], units[6], units[7], units[5], units[8]]

        output_str = '{},{},{},{},{},{}\n'.format(
            itemlist[0], itemlist[1], itemlist[2], itemlist[3], itemlist[4], itemlist[5])
        # check 空字符串
        biswelldefined = True
        for item in itemlist:
            if len(item) == 0:
                biswelldefined = False
                return (biswelldefined, output_str)
        # check 未知站点 src dest
        itemlist[0] = int(itemlist[0])
        itemlist[2] = int(itemlist[2])
        if itemlist[0] not in self.list_station_id:
            biswelldefined = False
            if itemlist[0] not in self.list_instation_id:
                self.list_instation_id.append(itemlist[0])
        elif itemlist[2] not in self.list_station_id:
            biswelldefined = False
            if itemlist[2] not in self.list_outstation_id:
                self.list_outstation_id.append(itemlist[2])
            # print(itemlist)
        # check 同站进出
        if itemlist[0] == itemlist[2]:
            biswelldefined = False
        # check成功
        if biswelldefined:
            output_str = '{},{},{},{},{},{},{},{}\n'.format(
                itemlist[0], itemlist[1], self.list_station_id.index(itemlist[0]),
                itemlist[2], itemlist[3], self.list_station_id.index(itemlist[2]), itemlist[4], itemlist[5])
            return (biswelldefined, output_str)
        else:
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
            self.list_file_id.append((filepath, month_id, num_att))

    def read_station_info(self, station_file_path):
        station = pd.read_excel(station_file_path, engine='openpyxl')
        v1 = station.values
        for i in range(1, len(station.values)):
            if v1[i][1] in self.list_station_id:
                print('Internal Err! duplicate station id--- in  def read_station_info(self, station_file_path)')
                break
            self.list_station_id.append(v1[i][1])
            self.list_stationinfo_id.append((v1[i][1], self.count, v1[i][2]))
            self.count = self.count + 1

    def get_station(self):
        return self.list_stationinfo_id


# 该class按照(src,dest)-pair收集数据
class ResolveSDpair(object):
    def __init__(self, num_station):
        self.input_data_dir = EncoHistDir
        self.output_data_dir = EncoHistDir_SDPair
        # 清理 文件夹
        self.clear_sd_dir()
        # 保存src-dst id
        self.list_sdpair_index = []
        # 保存src-dst record
        self.list_sdpair_record = []
        for s_id in range(num_station):
            for d_id in range(num_station):
                if s_id == d_id:
                    continue
                self.list_sdpair_index.append((s_id, d_id))
                self.list_sdpair_record.append([])
        self.divide_all_record_data()
        for sd_index in range(len(self.list_sdpair_index)):
            # 写文件
            s_id, d_id = self.list_sdpair_index[sd_index]
            new_sdpair_filepath = os.path.join(self.output_data_dir, '{}_{}.csv'.format(s_id, d_id))
            new_sdpair_file = open(new_sdpair_filepath, 'w+', encoding="utf-8")
            # print(len(self.list_sdpair_record[sd_index]))
            for line in self.list_sdpair_record[sd_index]:
                new_sdpair_file.write(line)
            new_sdpair_file.close()

    # 清除SD文件夹
    def clear_sd_dir(self):
        for i in os.listdir(self.output_data_dir):
            if i.split('.')[1] == 'csv':
                filepath = os.path.join(self.output_data_dir, i)
                os.remove(filepath)

    # 按照src-dest pair进行文件划分
    def divide_all_record_data(self):
        input_obj = open(self.input_data_dir, 'r', encoding="utf-8")
        line = input_obj.readline()
        while True:
            line = input_obj.readline()
            if not line:
                break
            else:
                units = line.split(',')
                t = (int(units[2]), int(units[5]))
                sd_index = self.list_sdpair_index.index(t)
                self.list_sdpair_record[sd_index].append(line)
        input_obj.close()


# 两个文件按照时间顺序排列各个记录
def seq_datacsv():
    list_seq_record = []
    output_obj = open(EncoHistDir, 'r', encoding="utf-8")
    while True:
        line = output_obj.readline()
        if not line:
            break
        else:
            uints = line.split(',')
            tm = time.strptime(uints[1], "%Y/%m/%d %H:%M:%S")
            list_seq_record.append((tm, line))
    output_obj.close()
    list_seq_record.sort()

    output_obj = open(EncoHistDir, 'w+', encoding="utf-8")
    for i in range(len(list_seq_record)):
        output_obj.write(list_seq_record[i][1])
    output_obj.close()

def seq_sdpaircsv(num_station):
    output_data_dir = EncoHistDir_SDPair
    list_sdpair_index = []
    for s_id in range(num_station):
        for d_id in range(num_station):
            if s_id == d_id:
                continue
            list_sdpair_index.append((s_id, d_id))
    for sd_index in range(len(list_sdpair_index)):
        # 写文件
        s_id, d_id = list_sdpair_index[sd_index]
        new_sdpair_filepath = os.path.join(output_data_dir, '{}_{}.csv'.format(s_id, d_id))
        new_sdpair_file = open(new_sdpair_filepath, 'r', encoding="utf-8")
        list_seq_record = []
        while True:
            line = new_sdpair_file.readline()
            if not line:
                break
            else:
                uints = line.split(',')
                tm = time.strptime(uints[1], "%Y/%m/%d %H:%M:%S")
                list_seq_record.append((tm, line))
        new_sdpair_file.close()
        list_seq_record.sort()
        output_obj = open(new_sdpair_filepath, 'w+', encoding="utf-8")
        for i in range(len(list_seq_record)):
            output_obj.write(list_seq_record[i][1])
        output_obj.close()

if __name__=='__main__':
    print(datetime.datetime.now())
    # 1.删除无效数据
    RPP = RecordPreProcess()
    list_station = RPP.get_station()
    # 2.按照src-dst划分
    RSD = ResolveSDpair(len(list_station))
    # 3.按照时间排序
    seq_datacsv()
    seq_sdpaircsv(len(list_station))
    # RSD = ResolveSDpair(216)
    print(datetime.datetime.now())
    print('OK')