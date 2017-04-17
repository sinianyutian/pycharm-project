#!/usr/bin/python
# -*- coding:utf-8 -*-  
# Filename:na-sa.py
# Author:Huang Weihang
# Email:huangweihang14@mails.ucas.ac.cn
# Data:2016-12-28

import os
from pyspark import SparkConf

os.environ["SPARK_HOME"] = "/usr/local/spark"
import time
from numpy import *
from pyspark import SparkContext
import progressbar
from matplotlib.pyplot import *
import numpy as np

# Initialize SparkContext
# sc = SparkContext("yarn", "spark na-sa")
conf = SparkConf().setMaster("local[8]").setAppName("spark na-sa")
sc = SparkContext(conf=conf)

n_itr = 20  # 迭代次数
ns = 100  # ns参数
nr = 20  # nr参数 可设为自适应参数
n_par = 2  # 参数个数
lowb = asarray([-10, -10])  # 下边界
upperb = asarray([10, 10])  # 上边界
parallel_num = 8  # 设置并行度

# ##---------------------------------------------------------------
# 目标函数设置
##---------------------------------------------------------------
f = lambda x: (x[0] - 3.5) ** 2 + (x[1] + 1) ** 2  # 目标函数
# 测试函数 参照MATLAB优化算法安全分析与应用 Page155
# f = lambda x: 0.5 - (sin(sqrt(x[0] ** 2 + x[1] ** 2)) ** 2 - 0.5) / \
#                     (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2  # Schaffer函数
# f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2  # Rosenbrock函数
# f = lambda x: 1 / 4000 * sum(x ** 2) - prod(cos(x / \
#                         (sqrt(arange(1, len(x) + 1))))) + 1  # Griewank函数
# f=lambda x:sum(asarray(x)**2-10*cos(2*pi*asarray(x))+10) #Rastrigin函数
# f=lambda x:-20*exp(-0.2*sqrt(1/len(x)*sum(asarray(x)**2)))-exp(1/len(x)* \
#     sum(cos(2*pi*asarray(x))))+exp(1)+20  #Ackley函数
##---------------------------------------------------------------


# 确保lowb、upperb与n_par一致
if len(lowb) != n_par:
    print('lowb is not the same length as n_par')
    exit()
if len(upperb) != n_par:
    print('upperb is not the same length as n_par')
    exit()

# 确保上边界大于下边界
if sum(lowb >= upperb) > 0:
    print('lowb is larger than upperb')
    exit()


##---------------------------------------------------------------
# NA算法
##---------------------------------------------------------------


def init_point(index):
    data = lowb + (upperb - lowb) * random.rand(n_par)
    misfit = f(data)
    return [data, misfit]


def genarate_point(index):
    orpt = chindex[index]
    vdata = data[orpt, :].copy()
    tmpdata = vdata.copy()
    oridata = []
    misfit = []
    for kk in xrange(int(ns / nr)):
        for i in xrange(n_par):  # 进行n_par个坐标循环
            range1 = [lowb[i], upperb[i]]
            tmp2data = tmpdata.copy()
            tmp2data[i] = vdata[i]
            vji = vdata[i]
            dj2 = sum((vdata - tmp2data) ** 2)
            for k in xrange(size(datasum, 0)):
                if k == orpt:
                    continue
                tmp2data[i] = datasum[k, i]
                vki = datasum[k, i]
                dk2 = sum((datasum[k, :] - tmp2data) ** 2)
                if abs(vki - vji) < 1e-10:
                    continue
                # 计算xj的第i个分量
                xji = 1 / 2. * (vki + vji + (dk2 - dj2) / (vki - vji))
                if xji > range1[0] and xji <= tmpdata[i]:
                    range1[0] = xji
                elif xji < range1[1] and xji >= tmpdata[i]:
                    range1[1] = xji
            # 均分分布生成新的坐标
            newx_loc = range1[0] + (range1[1] - range1[0]) * random.rand()
            tmpdata[i] = newx_loc
        oridata += [list(tmpdata)]  # 该次迭代生成的ns个样本
        misfit += [f(tmpdata)]
    return (oridata, misfit)


start = time.time()
par_index = sc.parallelize(np.arange(0, ns), parallel_num)
outvalue = par_index.map(init_point).collect()
data = map(lambda x: x[0], outvalue)
data = asarray(data)
misfit = map(lambda x: x[1], outvalue)
misfit = asarray(misfit)

progress = progressbar.ProgressBar(n_itr)
bar = progressbar.ProgressBar(widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
])

# 选择nr个样本点，生成ns个样本
datasum = data.copy()
for itr in bar(range(1, n_itr + 1)):
    # print('iteration number is ' + str(itr))
    if itr != 1:
        data = oridata.copy()
    else:
        orimisfit = misfit

    # 在前一次迭代生成的ns个样本中选择misfit最小的nr个
    chdata = sort(orimisfit)
    chindex = argsort(orimisfit)
    oridata = []

    # print('data: ', str(data[chindex[0], :]))
    # print('misfit is ', str(chdata[0]))

    chdata = sc.broadcast(chdata).value
    chindex = sc.broadcast(chindex).value
    datasum = sc.broadcast(datasum).value

    par_index = sc.parallelize(np.arange(0, nr), parallel_num)
    outvalue = par_index.map(lambda x: genarate_point(x)).collect()
    oridata = map(lambda x: x[0], outvalue)
    oridata = asarray(oridata).reshape((-1, 2))
    orimisfit = map(lambda x: x[1], outvalue)
    orimisfit = asarray(orimisfit).reshape((-1,))

    # 生成的总模型数与总模型对应的误差函数
    datasum = asarray(list(datasum) + list(oridata))
    misfit = asarray(list(misfit) + list(orimisfit))

end = time.time()
costTime = end - start

print "The job is finished!"
print "Cost time: {:.2f}s".format(costTime)
