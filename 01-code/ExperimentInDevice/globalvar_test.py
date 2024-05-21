import threading
import time
import numpy as np


NUM_OF_NODES = 3
Pdf_holiday = np.zeros((NUM_OF_NODES, 4))
Pdf_workday = np.zeros((NUM_OF_NODES, 4))

def init():
    global NUM_OF_NODES, Pdf_holiday, Pdf_workday, Z
    NUM_OF_NODES = 5
    Pdf_holiday = np.zeros((NUM_OF_NODES, 5))
    Pdf_workday = np.zeros((NUM_OF_NODES, 5))
    Z = 12345

def values():
    Pdf_workday[2,3] = 1
    Pdf_holiday[2,3] = np.random.random()
    print('values()', Z)



init()
print(NUM_OF_NODES)
print(Pdf_holiday)
print(Pdf_workday)
print(Z)
thd_updateP = threading.Thread(target=values)
thd_updateP.setDaemon(True)  # 设置守护线程
thd_updateP.start()
time.sleep(10)

print(NUM_OF_NODES)
print(Pdf_holiday)
print(Pdf_workday)
print(Z)