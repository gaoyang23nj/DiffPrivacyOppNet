import socket
import sys
import configparser
import os
# import platform
import shutil

NODE_FROM_ID = -1
NODE_FROM_IP = '0.0.0.0'
BIKE_P2P_IP = '0.0.0.0'
NODE_P2P_IP = '0.0.0.0'

FILENAME = 'bike_message_list.csv'

cf = configparser.ConfigParser()
cf.read("bike_config.ini")
cf.set('node_from', 'node_from_id', '1')

cf.set('node_from', 'node_from_ip', '172.16.2.4')
cf.set('bike2node_comm', 'node_p2p_ip', '192.168.1.5')

# cf.set('node_from', 'node_from_ip', '172.16.2.2')
# cf.set('bike2node_comm', 'node_p2p_ip', '192.168.1.4')

cf.set('bike2node_comm', 'bike_p2p_ip', '192.168.1.7')
with open('bike_config.ini', 'w') as f:
    cf.write(f)

files = []
with open('./Backup/node_message_list.csv', 'r') as fp1:
    with open('bike_message_list.csv', 'w+') as fp2:
        lines = fp1.readlines()
        for idx in range(len(lines)):
            fp2.write(lines[idx])
            if idx != 0:
                files.append(lines[idx].split(',')[-1].strip())
# i, j, k for os.walk('./')
# i, j, k for  os.walk('./BikeCarriedMessage')
fileandirs = os.listdir('./BikeCarriedMessage')
for del_filepath in fileandirs:
    del_filepath = os.path.join('./BikeCarriedMessage', del_filepath)
    if os.path.exists(del_filepath):
        if not os.path.isfile(del_filepath):
            print('Internal Err! {} is not a file.'.format(del_filepath))
        os.remove(del_filepath)

for filename in files:
    backuppath = os.path.join('./Backup', filename)
    if os.path.exists(backuppath) and os.path.isfile(backuppath):
        targetfilepath = os.path.join('./BikeCarriedMessage', filename)
        shutil.copyfile(backuppath, targetfilepath)


