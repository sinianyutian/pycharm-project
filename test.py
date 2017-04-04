# -*- coding:utf-8 -*-
import fire

def identity(arg=None,other=None):
  return arg, type(arg),other,type(other)
# class Widget(object):
#
#   def whack(self, n=1):
#     """Prints "whack!" n times."""
#     return ' '.join('whack!' for _ in xrange(n))
#
#   def bang(self, noise='bang'):
#     """Makes a loud noise."""
#     return '{noise} bang!'.format(noise=noise)


def main():
  # fire.Fire(Widget(), name='test')
  fire.Fire(identity, name='test')

if __name__ == '__main__':
  main()

# import difflib
#
# import fire
#
#
# def main():
#   fire.Fire(difflib, name='difffull')
#
# if __name__ == '__main__':
#   main()

#####################################################
##   cv2 test
#####################################################
# import cv2
#
# img = cv2.imread("./images/didi.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img.shape)
# img = cv2.resize(img, (168, 168), interpolation=cv2.INTER_CUBIC)
# cv2.namedWindow("Image")
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#####################################################
##   pandas and pandasql test
#####################################################
# import pandas as pd
# df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
# df1 = pd.DataFrame({'AAA' : [4,5,6,7], 'BBA' : [10,20,30,40],'CCD' : [100,50,-30,-50]})
# # print(df)
# df.ix[df.AAA >= 5,'BBB'] = -1
# # print(df)
#
# from pandasql import sqldf
# # 查找内存中的pandas数据框
# pysqldf = lambda q: sqldf(q, globals())
# e=pysqldf("select * from df join df1 where df1.AAA=df1.AAA")
# print(e)

#####################################################
##   progressbar test
##   put /r ahead in order to show on pychamr
#####################################################
# import sys
# import time

# sys.stdout.write("abc")
# sys.stdout.write("\r def")
# exit()
# def show_progress(progress, step):
#     progress += step
#     # sys.stdout.write("\r Processing progress: %d%%	" % (int(progress)))
#     sys.stdout.write("\r Processing: " + "*" * int(progress))
#     sys.stdout.flush()
#     return progress
#
#
# vidfiles = 100
# progress = 0
# progress_step = 100. / vidfiles
# progress = show_progress(progress, 0)
# for vf in range(vidfiles):
#     time.sleep(1)
#     progress = show_progress(progress, progress_step)
# progress = show_progress(progress, progress_step)

# import sys
# class progressbar(object):
#     def __init__(self, finalcount, block_char='.'):
#         self.finalcount = finalcount
#         self.blockcount = 0
#         self.block = block_char
#         self.f = sys.stdout
#         if not self.finalcount: return
#         self.f.write('\n------------------ % Progress -------------------1\n')
#         self.f.write('    1    2    3    4    5    6    7    8    9    0\n')
#         self.f.write('----0----0----0----0----0----0----0----0----0----0\n')
#     def progress(self, count):
#         count = min(count, self.finalcount)
#         if self.finalcount:
#             percentcomplete = int(round(100.0*count/self.finalcount))
#             if percentcomplete < 1: percentcomplete = 1
#         else:
#             percentcomplete=100
#         blockcount = int(percentcomplete//2)
#         if blockcount <= self.blockcount:
#             return
#         for i in range(self.blockcount, blockcount):
#             self.f.write(self.block)
#         self.f.flush()
#         self.blockcount = blockcount
#         if percentcomplete == 100:
#             self.f.write("\n")
#
# if __name__ == "__main__":
#     from time import sleep
#     pb = progressbar(8, "*")
#     for count in range(1, 9):
#         pb.progress(count)
#         sleep(1)
#     pb = progressbar(100)
#     pb.progress(20)
#     sleep(0.3)
#     pb.progress(47)
#     sleep(0.3)
#     pb.progress(90)
#     sleep(0.3)
#     pb.progress(100)
#     print "testing 1:"
#     pb = progressbar(1)
#     pb.progress(1)

#####################################################
##   pickle and  matplotlib test
#####################################################
# import pickle
# import  matplotlib.pyplot as plt
# import numpy as np
#
# plt.plot([1,2,3,4],[4,5,6,7])
# plt.figure()
# file=open("data.pkl")
# time_statics=pickle.load(file)
# plt.plot(time_statics[0,:])
# plt.plot(time_statics[1,:])
# plt.plot(time_statics[2,:])
# plt.plot(time_statics[3,:])
# plt.show()
# exit()
# print(time_statics)

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
