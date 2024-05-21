import configparser


cf = configparser.ConfigParser()
cf.read("bike_config.ini")
cf.read("bike_config.ini")
NODE_FROM_ID = int(cf.get('node_from', 'node_from_id'))
NODE_FROM_IP = cf.get('node_from', 'node_from_ip')
print(NODE_FROM_ID, NODE_FROM_IP)


cf.set('node_from', 'node_from_id', '1')
cf.set('node_from', 'node_from_ip', '172.16.2.2')
with open('bike_config.ini', 'w') as f:
    cf.write(f)