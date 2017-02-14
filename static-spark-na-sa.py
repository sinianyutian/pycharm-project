#!/usr/bin/python
# -*- coding:utf-8 -*-  
# Filename:na-sa.py
# Function:
# Author:Huang Weihang
# Email:huangweihang14@mails.ucas.ac.cn
# Data:2016-12-28

import os

os.environ["SPARK_HOME"] = "/usr/local/spark"

import time

from numpy import *
from pyspark import SparkContext
import progressbar

# Initialize SparkContext
sc = SparkContext("spark://ubuntu:7077", "spark na-sa")
# words = sc.parallelize(["scala", "java", "hadoop", "spark", "akka"])
# print words.count()

n_itr = 20  # 迭代次数
ns = 10  # ns参数
nr = 2  # nr参数 可设为自适应参数
n_par = 2  # 参数个数
lowb = asarray([-10, -10])  # 下边界
upperb = asarray([10, 10])  # 上边界
x_idx = 0  # 显示i分量为x轴
y_idx = 1  # 显示j分量为y轴
debug = 0  # 是否开启调试模式,0关闭，1开启（n_par~=2时自动关闭）
method = 'NA'  # 'MC' 蒙特卡洛算法,'NA' 邻近算法
parallel_nums = [1, 2, 3, 4]

# ##---------------------------------------------------------------
# 目标函数设置
##---------------------------------------------------------------
# f = lambda x: (x[0] - 3.5) ** 2 + (x[1] + 1) ** 2  # 目标函数
# 测试函数 参照Page153 MATLAB优化算法安全分析与应用
# f = lambda x: 0.5 - (sin(sqrt(x[0] ** 2 + x[1] ** 2)) ** 2 - 0.5) / \
#                     (1 + 0.001 * (x[0] ** 2 + x[1] ** 2)) ** 2  # Schaffer函数
f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2  # Rosenbrock函数
# f=lambda x:1/4000*sum(x.**2)-prod(cos(x./(sqrt(1:length(x)))))+1 #Griewank函数
# f=lambda x:sum(asarray(x)**2-10*cos(2*pi*asarray(x))+10) #Rastrigin函数
# f=lambda x:-20*exp(-0.2*sqrt(1/len(x)*sum(asarray(x)**2)))-exp(1/len(x)* \
#     sum(cos(2*pi*asarray(x))))+exp(1)+20  #Ackley函数
##---------------------------------------------------------------

# def f(x):
#     y = (x[0] - 3.5) ** 2 + (x[1] + 1) ** 2
#     return y
from matplotlib.pyplot import *
import numpy as np

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

# 参数个数大于2，关闭debug模式
if n_par != 2:
    print('Can''t open debug mode!')
    debug = 0

# 生成图形，并指定坐标轴范围；
figure(figsize=(4, 4))
axis([lowb[x_idx], upperb[x_idx], lowb[y_idx], upperb[y_idx]])
grid(True)
box(True)
hold(True)
xlabel(str(x_idx) + ' component')
ylabel(str(y_idx) + ' component')

# 参数为2且不为debug的情况下作出目标函数f(x)的等值图
if n_par == 2 and debug != 1:
    x = linspace(lowb[0], upperb[0], 100)
    y = linspace(lowb[1], upperb[1], 100)
    X, Y = meshgrid(x, y)
    z = zeros_like(X)
    for i in xrange(0, size(X, 0)):
        for j in xrange(0, size(X, 1)):
            z[i, j] = f([X[i, j], Y[i, j]])

    contourf(X, Y, z)
    # cmap = matplotlib.cm.jet
    # # 设定每个图的colormap和colorbar所表示范围是一样的，即归一化
    # norm = matplotlib.colors.Normalize(vmin=160, vmax=300)
    # # 显示图形，此处没有使用contourf #>>>ctf=plt.contourf(grid_x,grid_y,grid_z)
    # imshow(z.T, origin='lower', cmap=cmap, norm=norm)
    # colorbar()
    # mycmap = cm.get_cmap('jet')
    # mycmap.set_under('w')
    # colorbar(mycmap)
    colorbar()
    # cbar = colorbar()
    # cbar.set_ticks([0,200,400,600,800,1000])

# Monte Carlo方法
if method == 'MC':
    data = zeros([ns * nr + nr, n_par])
    misfit = zeros(ns * nr + nr)
    for i in xrange(0, (ns * nr + nr)):
        data[i, :] = lowb + (upperb - lowb) * random.rand(1, n_par)
        misfit[i] = f(data[i, :])
    title('Monte Carlo with point nums=' + str(ns * nr + nr))
    scatter(data[:, x_idx], data[:, y_idx])
    chdata = amin(misfit)
    chindex = argmin(misfit)
    print('data and misfist is:')
    print(data[chindex, :], chdata)
    show()
    exit()


##---------------------------------------------------------------
# NA算法
##---------------------------------------------------------------
# data = zeros([ns, n_par])
# misfit = zeros(ns)


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
        # print "kk",kk
        for i in xrange(n_par):  # 进行n_par个坐标循环
            # print('  i='+str(i))
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
                # print 'dk2',dk2
                if abs(vki - vji) < 1e-10:
                    continue
                xji = 1 / 2. * (vki + vji + (dk2 - dj2) / (vki - vji))  # 计算xj的第i个分量
                # print xji
                if xji > range1[0] and xji <= tmpdata[i]:
                    range1[0] = xji
                elif xji < range1[1] and xji >= tmpdata[i]:
                    range1[1] = xji
                    # print "range1",range1
            # exit()
            # print range1
            newx_loc = range1[0] + (range1[1] - range1[0]) * random.rand()  # 均分分布生成新的坐标
            tmpdata[i] = newx_loc
        oridata += [list(tmpdata)]  # 该次迭代生成的ns个样本
        misfit += [f(tmpdata)]
    return (oridata, misfit)

time_statics=np.zeros((len(parallel_nums),n_itr))

parallel_num_index = 0
for parallel_num in parallel_nums:
    print("parallel_num: {} ,n_iter: {}".format(parallel_num,n_itr))
    start = time.time()
    par_index = sc.parallelize(np.arange(0, ns), numSlices=parallel_num)
    outvalue = par_index.map(init_point).collect()
    data = map(lambda x: x[0], outvalue)
    data = asarray(data)
    misfit = map(lambda x: x[1], outvalue)
    misfit = asarray(misfit)

    # print(data.shape)
    # print(misfit.shape)

    # # 随机生成ns个样本，并绘图
    # for i in xrange(ns):
    #     data[i, :] = lowb + (upperb - lowb) * random.rand(1, n_par)
    #     misfit[i] = f(data[i, :])
    # #

    # title('neighbourhood algoritthm with ns=' + str(ns) + ' nr=' + str(nr))
    # scatter(data[:, x_idx], data[:, y_idx])
    # show()
    # exit()

    # progress = progressbar.ProgressBar(n_itr)
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
        # for i in xrange(size(data, 0)):
        #     misfit[i] = f(data[i, :])
        # 在前一次迭代生成的ns个样本中选择misfit最小的nr个

        chdata = sort(orimisfit)
        chindex = argsort(orimisfit)
        oridata = []
        # print('data: ', str(data[chindex[0], :]))
        # print('misfit is ', str(chdata[0]))

        chdata = sc.broadcast(chdata).value
        chindex = sc.broadcast(chindex).value
        datasum = sc.broadcast(datasum).value

        par_index = sc.parallelize(np.arange(0, nr), numSlices=parallel_num)
        outvalue = par_index.map(lambda x: genarate_point(x)).collect()
        oridata = map(lambda x: x[0], outvalue)
        oridata = asarray(oridata).reshape((-1, 2))
        orimisfit = map(lambda x: x[1], outvalue)
        orimisfit = asarray(orimisfit).reshape((-1,))

        datasum = asarray(list(datasum) + list(oridata))  # 生成的总样本数
        misfit = asarray(list(misfit) + list(orimisfit))

        end = time.time()
        costTime = end - start
        time_statics[parallel_num_index,itr-1] = costTime

    parallel_num_index += 1
    print "The job is finished!"
    print "Cost time: {:.2f}s".format(costTime)
    # scatter(datasum[:, x_idx], datasum[:, y_idx])
    # show()

import pickle
output = open('data.pkl', 'wb')
# Pickle dictionary using protocol 0.
pickle.dump(time_statics, output)
output.close()

file=open("data.pkl")
time_statics=pickle.load(file)
figure()
plot(time_statics[0,:])
hold(True)
plot(time_statics[1,:])
plot(time_statics[2,:])
plot(time_statics[3,:])
show()
exit()

