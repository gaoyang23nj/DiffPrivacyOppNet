import schedule
import time
import datetime

def job():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


print('hi')
schedule.every(10).seconds.do(job)
print('hello')
i = 1
while True:
    schedule.run_pending()
    time.sleep(5)
    print(i)
    i = i +1