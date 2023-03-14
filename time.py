import time


currenttime = time.time()

time.sleep(5)
elapsed_time = time.time() - currenttime

print('%.0f' % (elapsed_time))
