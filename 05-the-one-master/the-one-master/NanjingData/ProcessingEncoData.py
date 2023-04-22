import datetime
filename_stations = 'station_info_qiaobei.csv'
filename_data = 'data_qiaobei.csv'

list_stations = []
f = open(filename_stations, encoding='utf-8')
lines = f.readlines()
for line in lines:
    terms = line.split(',')
    tmp_id = int(terms[1])
    tmp_globalid = int(terms[0])
    list_stations.append((tmp_id, tmp_globalid))
f.close()
print(len(list_stations))
print(list_stations)

BEGIN_RUNNING_TIMES = datetime.datetime.strptime('2017/7/1 0:0:00', "%Y/%m/%d %H:%M:%S")
MAX_RUNNING_TIMES = datetime.datetime.strptime('2017/7/14 23:59:59', "%Y/%m/%d %H:%M:%S")

list_enco_hist = []
# rewrite external event
f = open(filename_data, encoding='utf-8')
lines = f.readlines()
for line in lines:
    (i_station_id, a_time, i_node, j_station_id, b_time, j_node, i_name, j_name) \
        = line.strip().split(',')
    tm = datetime.datetime.strptime(a_time, "%Y/%m/%d %H:%M:%S")
    # i_node = int(i_node)
    # j_node = int(j_node)
    if (tm >= BEGIN_RUNNING_TIMES) and (tm <= MAX_RUNNING_TIMES):
        total_sec = int((tm - BEGIN_RUNNING_TIMES).total_seconds())
        list_enco_hist.append([total_sec, i_node, j_node, tm])
f.close()
print(len(list_enco_hist))
# for i in range(20):
#     print(list_enco_hist[i])

# 计数
list_enco_hist.sort()
# for i in range(len(list_enco_hist)):
#     print(list_enco_hist[i])

# first round
res = 0
index = 0
while index < len(list_enco_hist):
    tmp = 1
    while (index+tmp < len(list_enco_hist)) and (list_enco_hist[index][0] == list_enco_hist[index+tmp][0]):
        tmp = tmp + 1
    if res < tmp:
        res = tmp
    if tmp > 1:
        print('{} times:{}'.format(list_enco_hist[index][0], tmp))
        i = 1
        while i < tmp and (index+i) < len(list_enco_hist) :
            list_enco_hist[index+i][0] = list_enco_hist[index][0] + 1
            i = i + 1
        index = index + i
    else:
        index = index + 1
print(res)

# second round
print(100*'#')
res = 0
index = 0
while index < len(list_enco_hist):
    tmp = 1
    while (index+tmp < len(list_enco_hist)) and (list_enco_hist[index][0] == list_enco_hist[index+tmp][0]):
        tmp = tmp + 1
    if tmp > 1:
        print('{} times:{}'.format(list_enco_hist[index][0], tmp))
        i = 1
        while i < tmp and (index+i) < len(list_enco_hist):
            list_enco_hist[index+i][0] = list_enco_hist[index][0] + 1
            i = i + 1
    if res < tmp:
        res = tmp
    index = index + 1
print(res)

# check
print(100*'#')
res = 0
index = 0
while index < len(list_enco_hist):
    tmp = 1
    while (index+tmp < len(list_enco_hist)) and (list_enco_hist[index][0] == list_enco_hist[index+tmp][0]):
        tmp = tmp + 1
    if tmp > 1:
        print('{} times:{}'.format(list_enco_hist[index][0], tmp))
        # i = 1
        # while i < tmp and (index+i) < len(list_enco_hist):
        #     list_enco_hist[index+i][0] = list_enco_hist[index][0] + 1
        #     i = i + 1
    if res < tmp:
        res = tmp
    index = index + 1
print(res)

#
new_list_enco_hist = []
connection_time = 0.5
for i in range(len(list_enco_hist)):
    new_list_enco_hist.append((list_enco_hist[i][0], 'ONEWAY_CONN', list_enco_hist[i][1], list_enco_hist[i][2], 'up'))
    # tmp_outputstr = str(list_enco_hist[i][0]) + ' ' + 'ONEWAY_CONN' + ' ' + list_enco_hist[i][1] + \
    #                 ' ' + list_enco_hist[i][2] + ' ' + 'up'
    new_list_enco_hist.append((list_enco_hist[i][0]+ connection_time, 'ONEWAY_CONN', list_enco_hist[i][1], list_enco_hist[i][2], 'down'))
    # tmp_outputstr = str(list_enco_hist[i][0] + connection_time) + ' ' + 'ONEWAY_CONN' + ' ' + list_enco_hist[i][1] + \
    #                 ' ' + list_enco_hist[i][2] + ' ' + 'down'

# sort according to time increasing
new_list_enco_hist.sort()

filename_output = 'data_for_ONE_new.txt'
f = open(filename_output, 'w+')
for i in range(len(new_list_enco_hist)-1):
    tmp_outputstr = str(new_list_enco_hist[i][0]) + ' ' + new_list_enco_hist[i][1] + ' ' + new_list_enco_hist[i][2] + \
                    ' ' + new_list_enco_hist[i][3] + ' ' + new_list_enco_hist[i][4] + '\n'
    f.write(tmp_outputstr)

i = -1
tmp_outputstr = str(new_list_enco_hist[i][0]) + ' ' + new_list_enco_hist[i][1] + ' ' + new_list_enco_hist[i][2] + \
                ' ' + new_list_enco_hist[i][3] + ' ' + new_list_enco_hist[i][4]
f.write(tmp_outputstr)
f.close()
