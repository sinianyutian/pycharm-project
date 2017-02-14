import pickle

import  matplotlib.pyplot as plt
import numpy as np
plt.plot([1,2,3,4],[4,5,6,7])
plt.figure()
file=open("data.pkl")
time_statics=pickle.load(file)
plt.plot(time_statics[0,:])
plt.plot(time_statics[1,:])
plt.plot(time_statics[2,:])
plt.plot(time_statics[3,:])
plt.show()
exit()
print(time_statics)

#####################################################
##   progressbar test
#####################################################
# from progressbar import ProgressBar
# import time
# pbar = ProgressBar(maxval=10)
# pbar.start()
# for i in range(1, 11):
#     pbar.update(i)
#     time.sleep(1)
# pbar.finish()

# import progressbar
# import time
#
# bar = progressbar.ProgressBar(widgets=[
#     ' [', progressbar.Timer(), '] ',
#     progressbar.Bar(),
#     ' (', progressbar.ETA(), ') ',
# ])
#
# for i in bar(range(20)):
#     time.sleep(0.1)
