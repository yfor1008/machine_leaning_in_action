#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : kmeans.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/12 11:19:11
# @Docs   : kmeans算法
'''

import numpy as np
from matplotlib import pyplot as plt

def loadDataSet(fileName):
    '''
    ### Docs: 载入数据
    ### Args:
        - fileName: str, 文件名, 数据以tab键分隔
    ### Returns:
        - dataSet: m*n, array, m行数据, n列特征
    '''
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = [float(ft) for ft in curLine] 
            dataSet.append(fltLine)
    dataSet = np.array(dataSet)
    return dataSet

def distEclud(dataA, dataB):
    '''
    ### Docs: 计算欧氏距离, 这里没有开方
    ### Args:
        - dataA: vector/array, 特征A, 若为vector则为行向量; 若为array则为矩阵, m*n, m行数据n列特征
        - dataB: vector/array, 特征B
    ### Returns:
        - dist: float, 距离, shape与dataA/dataB一致
    '''

    if len(dataA.shape) == 1 and len(dataB.shape) == 1:
        # vector/vector
        # dist = np.sqrt(np.sum(np.power(dataA - dataB, 2)))
        dist = np.sum(np.power(dataA - dataB, 2))
    else:
        # vector/array or array/array
        # dist = np.sqrt(np.sum(np.power(dataA - dataB, 2), axis=1))
        dist = np.sum(np.power(dataA - dataB, 2), axis=1)
    return dist

def randCent(dataSet, k):
    '''
    ### Docs: 随机生成k个质心(聚类中心)
        - 随机质心必须在这个数据集的边界之内
        - 可以通过找到数据集每一个特征的最小最大值来确定
    ### Args:
        - dataSet: m*n, array, m行数据, n列特征
        - k: int, 聚类个数
    ### Returns:
        - centroids: k*n, array, 生成的k个质心
    '''

    m, n = dataSet.shape
    minJ = np.min(dataSet, axis=0) # 计算每列最小值, 即每一维度特征的最小值
    maxJ = np.max(dataSet, axis=0) # 计算每列最大值, 即每一维度特征的最大值
    rangeJ = maxJ - minJ # 每列数据的范围, 即每一维度特征的范围

    minJ = np.tile(minJ, (k, 1)) #　扩展成k*n, 方便矩阵计算
    rangeJ = np.tile(rangeJ, (k, 1))
    centroids = minJ + rangeJ * np.random.rand(k, n) # np.random.rand(k, n)返回k*n个[0, 1]的数
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    ### Docs: kmeans聚类
    ### Args:
        - dataSet: m*n, array, m行数据, n列特征
        - k: int, 聚类个数
        - distMeas: obj, 距离度量方法
        - createCent: obj, 生成初始质心方法
    ### Returns:
        - centroids: k*n, array, 聚类中心
        - clusterAssment: m*2, 分类结果: 第0列记录分类索引值; 第1列存储误差SSE
    '''

    m, n = dataSet.shape

    clusterAssment = np.zeros((m, 2)) # 记录分类结果: 第0列记录分类索引值; 第1列存储误差SSE
    centroids = createCent(dataSet, k) # 初始化质心
    clusterChanged = True # 聚类结果是否改变标志, 用于判断是否结束迭代

    while clusterChanged:
        clusterChanged = False

        minDist = np.inf # 正无穷
        minIndex = -1
        for i in range(m):
            # 计算每个样本点到所有质心的距离
            dist = distMeas(dataSet[i, :], centroids) # 当前质心到所有数据的距离, 行向量与矩阵计算, k*1
            minIndex = np.argmin(dist) # 查找最小距离
            minDist = dist[minIndex]

            if clusterAssment[i, 0] != minIndex:
                # 如果有任何点的分类结果发生改变, 则继续迭代
                clusterChanged = True

            # 更新分类结果
            clusterAssment[i, :] = minIndex, minDist

        # 更新质心
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment[:, 0] == cent] # 得到分类为cent所有样本点, x*n
            centroids[cent, :] = np.mean(ptsInClust, axis=0) # 计算每列的均值, 即样本点的中心位置, 1*n

    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    '''
    ### Docs: 二分kmeans聚类
        - 首先将所有点作为一个聚类, 然后将该聚类一分为二
        - 选择其中一个聚类继续进行划分, 选择哪一个聚类进行划分取决于对其划分时候可以最大程度降低SSE(平方和误差)的值
        - 不断重复, 直到达到停止条件
    ### Args:
        - dataSet: m*n, array, m行数据, n列特征
        - k: int, 聚类个数
        - distMeas: obj, 距离度量方法
    ### Returns:
        - centroids: k*n, array, 聚类中心
        - clusterAssment: m*2, 分类结果: 第0列记录分类索引值; 第1列存储误差SSE
    '''

    m, n = dataSet.shape

    clusterAssment = np.zeros((m, 2)) # 记录分类结果: 第0列记录分类索引值; 第1列存储误差SSE
    centroid0 = np.mean(dataSet, axis=0) # 整个数据的质心作为初始质心
    centList = [centroid0] # [array]

    clusterAssment[:, 1] = distMeas(centroid0, dataSet) # 所有样本点到初始质心的距离
    
    # 不停进行划分, 直到得到聚类数目k为止
    while len(centList) < k:
        lowestSSE = np.inf # 正无穷

        # 遍历所有的聚类来决定最佳的聚类进行划分
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[clusterAssment[:, 0] == i, :] # 当前聚类类别中的所有样本点, x*n
            centroid, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # 使用kmeans对当前样本进行2分类
            # 将误差值与剩余数据集的误差之和作为本次划分的误差
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[clusterAssment[:, 0] != i, 1])
            # print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            # 如果本次划分的SSE值最小, 则本次划分被保存
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroid
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
    
        # k=2时, kmeans会得到分别为0和1的两个类别
        bestClustAss[bestClustAss[:, 0] == 1, 0] = len(centList) # kmeans返回1类的类别标签, 设置成总类别数, 保证类别不会重复
        bestClustAss[bestClustAss[:, 0] == 0, 0] = bestCentToSplit # kmeans返回0类的类别标签, 设置当前类别
        # print('the bestCentToSplit is: ', bestCentToSplit)
        # print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表
        centList[bestCentToSplit] = bestNewCents[0, :] # kmeans返回0类的质心, 更新为当前类别的质心
        centList.append(bestNewCents[1, :]) # kmeans返回1类的质心, 更新为最后类别的质心
        # 更新分类结果
        clusterAssment[clusterAssment[:, 0] == bestCentToSplit, :] = bestClustAss

    return np.array(centList), clusterAssment

def kmeansShow(dataSet, centroids, clusterAssment):
    '''
    ### Docs: 显示聚类结果
    ### Args:
        - dataSet: m*n, array, m行数据, n列特征
        - centroids: k*n, array, 聚类中心
        - clusterAssment: m*2, 分类结果: 第0列记录分类索引值; 第1列存储误差SSE
    '''

    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=clusterAssment[:, 0])
    plt.scatter(centroids[:, 0], centroids[:, 1], c="r")
    plt.show()


if __name__ == "__main__":

    dataSet = loadDataSet('testSet2.txt')

    # randCent(dataSet, 3)
    # dist = distEclud(dataSet[2, :], dataSet[4:6, :])
    # print(dist)
    # centroids, clusterAssment = kMeans(dataSet, 3, distMeas=distEclud, createCent=randCent)
    centroids, clusterAssment = biKmeans(dataSet, 3, distMeas=distEclud)
    # print(centroids)
    # print(clusterAssment)
    kmeansShow(dataSet, centroids, clusterAssment)

