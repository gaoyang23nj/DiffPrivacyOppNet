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

files = []
with open('./Backup/node_message_list.csv', 'r') as fp1:
    with open('node_message_list.csv', 'w+') as fp2:
        lines = fp1.readlines()
        for idx in range(len(lines)):
            fp2.write(lines[idx])
            if idx != 0:
                files.append(lines[idx].split(',')[-1].strip())

for filename in files:
    backuppath = os.path.join('./Backup', filename)
    if os.path.exists(backuppath) and os.path.isfile(backuppath):
        targetfilepath = os.path.join('./', filename)
        shutil.copyfile(backuppath, targetfilepath)


