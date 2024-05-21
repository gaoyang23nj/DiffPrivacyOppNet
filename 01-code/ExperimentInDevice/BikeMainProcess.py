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

def init():
    if os.path.exists('./BikeCarriedMessage'):
        if os.path.isfile('./BikeCarriedMessage'):
            os.remove('./BikeCarriedMessage')

    if not os.path.exists('./BikeCarriedMessage'):
        os.mkdir('./BikeCarriedMessage')
        print('create dir to store messages (files)')

    global NODE_FROM_ID, NODE_FROM_IP, BIKE_P2P_IP, NODE_P2P_IP
    cf = configparser.ConfigParser()
    cf.read("bike_config.ini")
    NODE_FROM_ID = int(cf.get('node_from', 'node_from_id'))
    NODE_FROM_IP = cf.get('node_from', 'node_from_ip')

    host = socket.gethostname()
    hostip = socket.gethostbyname(host)
    print('HOST IP:', host, hostip)
    BIKE_P2P_IP = cf.get('bike2node_comm', 'bike_p2p_ip')
    NODE_P2P_IP = cf.get('bike2node_comm', 'node_p2p_ip')
    print('BIKE_P2P_IP:', BIKE_P2P_IP)
    print('NODE_P2P_IP:', NODE_P2P_IP)

# bike connect 2 node j (client), trip i->j
def Bike2Node_connect(target_ip, node_from_id, node_from_ip):
    try:
        #create an AF_INET, STREAM socket (TCP)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        print('Failed to create socket')
        sys.exit()
    print('Socket Created')
    # hostname = socket.gethostname()  # 获取本地主机名
    # hostip = socket.gethostbyname(hostname)  # 获取本地ip
    port = 12349  # 设置端口号
    s.connect((target_ip, port))
    #
    # buf = s.recv(1024)
    # str = buf.decode()
    # notify bike2node
    # s.send('Bike2Node connect! bike_ip:{}'.format(BIKE_P2P_IP).encode())
    # notify node2node_from_id node2node_from_ip
    s.send('{};{}'.format(NODE_FROM_ID, NODE_FROM_IP).encode())
    tmp_str = s.recv(1024).decode().split(';')
    print(tmp_str[0])
    assert tmp_str[0] == 'Bike2Node connect success!'
    node_to_id = int(tmp_str[1])

    # notify message list (dst, deadline)
    MessageList = []
    tmp_str = ''
    with open(FILENAME, 'r+') as fobj:
        lines = fobj.readlines()
        for idx in range(1, len(lines), 1):
            tlist = lines[idx].split(',')
            # dst, src, message_id, deadline
            tmp_str += '{},{},{},{};'.format(tlist[0].strip(), tlist[1].strip(), tlist[2].strip(), tlist[3].strip())
            MessageList.append((tlist[0].strip(), tlist[1].strip(), tlist[2].strip(), tlist[3].strip(), tlist[4].strip(), tlist[5].strip(), tlist[6].strip()))
    num_mg = len(MessageList)
    tmp_str = '{};'.format(num_mg) + tmp_str
    tmp_str = tmp_str[:-1]
    print('Carried Messages:', tmp_str)
    s.sendall(tmp_str.encode())

    # 3. accept auth 'Notify Bike to Forward Messages! from {}'.format(NODE_ID).encode()
    # tmp_str = s.recv(1024).decode()
    # print(tmp_str)
    # accept forward notify
    tmp_str = s.recv(2048).decode()
    print(tmp_str)
    notify_forward_list_str = tmp_str.split(';')
    num_fwd_file = int(notify_forward_list_str[0])
    notify_forward_list_str.pop(0)
    assert num_fwd_file == len(notify_forward_list_str)
    notify_forward_list = []
    for line in notify_forward_list_str:
        if line.strip() == '':
            continue
        tunple = line.split(',')
        # dst_id,src_id,message_id
        notify_forward_list.append((tunple[0], tunple[1], tunple[2]))
    print('notify_forward_list:', notify_forward_list)

    # send messages
    for tunple in notify_forward_list:
        filepath = ''
        tmp_header_str = ''
        for message in MessageList:
            if message[0]==tunple[0] and message[1]==tunple[1] and message[2]==tunple[2]:
                filepath = message[-1]
                tmp_header_str = 'begin transmission;{};{};{};{};{};{};{}'.format(
                    message[0], message[1], message[2], message[3], message[4], message[5], message[6])
                break
        filepath = os.path.join('./BikeCarriedMessage', filepath)
        if filepath == '' or not os.path.exists(filepath):
            print('Internal Err! Message Miss! {}'.format(filepath))
            continue
        # ack begin transmission
        s.send(tmp_header_str.encode())
        print(s.recv(1024).decode())
        data = b''
        with open(filepath, 'rb') as fp:
            while True:
                tmp_data = fp.read(1024)
                data += tmp_data
                if len(tmp_data) < 1024:
                    break
        length = len(data)
        buf = length.to_bytes(5, 'big', signed=False) + data
        print(length, length.to_bytes(5, 'big', signed=False))
        s.sendall(buf)
        tmp_str = s.recv(1024).decode()
        print(tmp_str)
        if tmp_str != 'end transmission':
            print('Internal Err!')
            break
    s.send('file transmission B2N closed'.encode())
    # clear files (messages) & rewrite bike_message_list.csv
    print('flush files (messages) & message list in Bike')
    # clear files (message)
    fileandirs = os.listdir('./BikeCarriedMessage')
    for del_filepath in fileandirs:
        del_filepath = os.path.join('./BikeCarriedMessage', del_filepath)
        if os.path.exists(del_filepath):
            if not os.path.isfile(del_filepath):
                print('Internal Err! {} is not a file.'.format(del_filepath))
            os.remove(del_filepath)
    # receive header, filename
    buf = b''
    label = True
    while True:
        tmp_buf = s.recv(1024)
        buf_len = len(tmp_buf)
        if label:
            length = int.from_bytes(tmp_buf[0:4], 'big', signed=False)
            print(length, tmp_buf[0:4], len(tmp_buf))
            tmp_buf = tmp_buf[4:buf_len]
            buf_len = buf_len-4
            remain_length = length
            label = False
        buf += tmp_buf
        remain_length = remain_length-buf_len
        if remain_length <= 0:
            break
    tmp_str = buf.decode()
    filenames = tmp_str.split(';')

    for filename in filenames:
        # check filename
        s.send(filename.encode())
        print('begin:', filename)
        buf = b''
        label = True
        while True:
            tmp_buf = s.recv(1024)
            buf_len = len(tmp_buf)
            if label:
                length = int.from_bytes(tmp_buf[0:5], 'big', signed=False)
                print(length, tmp_buf[0:5], len(tmp_buf))
                tmp_buf = tmp_buf[5:buf_len]
                buf_len = buf_len - 5
                remain_length = length
                label = False
            buf += tmp_buf
            remain_length = remain_length - buf_len
            if remain_length <= 0:
                break
        if filename != 'node_message_list.csv':
            filepath = os.path.join('./BikeCarriedMessage', filename)
        else:
            filepath = 'bike_message_list.csv'
        # receive messages
        with open(filepath, 'wb') as fp:
            fp.write(buf)

    print('end transmission')
    s.send('end transmission'.encode())
    tmp_str = s.recv(1024).decode()
    print(tmp_str)
    # assert tmp_str == 'Refresh Bike with node_{} success!'.format(node_to_id)
    s.close()
    print('B2N socket closed')

    # modify config
    node_to_id = tmp_str.split(';')[0].strip()
    node_to_ip = tmp_str.split(';')[1].strip()
    cf = configparser.ConfigParser()
    cf.read("bike_config.ini")
    cf.set('node_from', 'node_from_id', node_to_id)
    cf.set('node_from', 'node_from_ip', node_to_ip)
    with open('bike_config.ini', 'w') as f:
        cf.write(f)


init()
# 连接
Bike2Node_connect(NODE_P2P_IP, NODE_FROM_ID, NODE_FROM_IP)