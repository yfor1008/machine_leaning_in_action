#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : pca.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/13 16:50:23
# @Docs   : PCA主成分分析
'''

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
    '''
    ### Docs: 加载数据
    ### Args:
        - fileName: str, 文件名
        - delim: str, 数据分隔方式
    ### Returns:
        - dataSet: m*n, array, m行数据, n列特征
    '''
    dataSet = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split(delim)
            fltLine = [float(ft) for ft in curLine] 
            dataSet.append(fltLine)
    dataSet = np.array(dataSet)
    return dataSet

'''pca算法
    cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)]/(n-1)
    Args:
        dataSet   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDdataSet  降维后数据集
        reconMat     新的数据集空间
'''
def pca(dataSet, t=1.0):
    '''
    ### Docs: pca算法
        - 降维: lowDdataSet_m*k = (dataSet_m*n - meanVals_1*n) * redEigVects_n*k
        - 重建: dataSet_m*n = lowDdataSet_m*k * redEigVects_n*k.T + meanVals_1*n
    ### Args:
        - dataSet: m*n, array, m行数据, n列特征
        - t: float, 保留比例, [0, 1.0]
    ### Returns:
        - lowDdataSet: m*k, array, 降维后的数据
        - reconArr: m*n, array, 重建后的数据, 若k=n, 则reconArr=dataSet
        - meanVals: 1*n, array, 特征最小值
        - redEigVects: n*k, array, 特征向量构成的矩阵W
    '''

    meanVals = np.mean(dataSet, axis=0) # 1*n, 计算每个特征的均值, readme中的公式1
    meanRemoved = dataSet - meanVals # m*n, 去中心化, readme中的公式2
    covArr = np.cov(meanRemoved, rowvar=0) # n*n, 协方差矩阵, readme中的公式3, rowvar=0表示每行一个样本
    eigVals, eigVects = np.linalg.eig(covArr) # 每列为一个特征向量, eigVects[:, i]对应eigVals[i], readme中的步骤3
    eigValInd = np.argsort(eigVals) # 对特征值进行排序, 从小到大, readme中的步骤4
    k = percentageK(eigVals, t) # readme中的公式6
    eigValInd = eigValInd[:-(k+1):-1]
    redEigVects = eigVects[:, eigValInd] # n*k, 根据特征值顺序调整特征向量
    lowDdataSet = np.dot(meanRemoved, redEigVects) # m*k, 降维后的数据, readme中的步骤5
    reconArr = np.dot(lowDdataSet, redEigVects.T) + meanVals # m*n, 重建后的数据

    return lowDdataSet, reconArr, meanVals, redEigVects

def percentageK(eigVals, t):
    '''
    ### Docs: 计算k的值
    ### Args:
        - eigVals: n*1, array, 特征值
        - t: float, 保留比例, [0, 1.0]
    ### Returns:
        - k: int, k值
    '''
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1] # 降序
    arraySum = np.sum(sortArray)
    tmpSum = 0
    k = 0

    for i, val in enumerate(sortArray):
        tmpSum += val
        k = i + 1
        if tmpSum >= arraySum * t:
            # readme中的公式6
            break
    return k

def replaceNanWithMean():
    '''
    ### Docs: 半导体数据
    ### Returns:
        - dataSet: m*n, array, m行数据, n列特征
    '''
    dataSet = loadDataSet('./secom.data', ' ')
    m, n = dataSet.shape
    for i in range(n):
        meanVal = np.mean(dataSet[~np.isnan(dataSet[:, i]), i]) # 求非nan数据的均值
        dataSet[np.isnan(dataSet[:, i]), i] = meanVal # 用均值替代nan数据
    return dataSet

def analyse_data(dataSet):
    '''
    ### Docs: 数据分析
    '''
    meanVals = np.mean(dataSet, axis=0)
    meanRemoved = dataSet - meanVals
    covSet = np.cov(meanRemoved, rowvar=0)
    eigvals, eigVects = np.linalg.eig(covSet)
    eigValInd = np.argsort(eigvals)

    topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(np.sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))

def show_picture(dataSet, reconArr):
    '''
    ### Docs: 结果可视化
    ### Args:
        - dataSet: m*n, array, 原始数据
        - reconArr: m*n, array, 重建数据
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1], marker='^', s=90,c='green')
    ax.scatter(reconArr[:, 0], reconArr[:, 1], marker='o', s=50, c='red')
    plt.show()

if __name__ == "__main__":

    # dataSet = loadDataSet('testSet.txt')
    # # print(dataSet)
    # lowDdataSet, reconArr, meanVals, redEigVects = pca(dataSet, 0.5)
    # show_picture(dataSet, reconArr)

    dataSet = replaceNanWithMean()
    analyse_data(dataSet)
    lowDdataSet, reconArr, meanVals, redEigVects = pca(dataSet, 0.99)
    show_picture(dataSet, reconArr)
