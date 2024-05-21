import os

for i,j,k in os.walk('./'):
    print(i,j,k)

t1 = os.path.exists('node_message_list.csv')
print(t1)

t1 = os.path.isfile('node_message_list.csv')
print(t1)

t1 = os.path.exists('NoName')
print(t1)

t1 = os.path.isfile('NoName')
print(t1)

with open('file_test_write.txt', 'a+') as fp:
    fp.write('a')
    fp.write('awwww')