#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : kNN.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/02 17:52:21
# @Docs   : 最近邻kNN分类实现
'''
import numpy as np
import operator
from os import listdir

def classify_kNN(in_data, data_set, labels, k):
    '''
    ### Docs: K近邻分类
    ### Args:
        - in_data: 1*n, array, 输入特征数据, n为特征维度
        - data_set: m*n, array, 数据集, m为数据个数, n为特征维度
        - labels: m*1, array, 数据标签, m为数据个数
        - k: int, 最近邻参数
    ### Returns:
        - classed: 分类类别
    '''

    data_num = data_set.shape[0]
    in_data_extended = np.tile(in_data, (data_num, 1)) # 将输入特征扩展到与数据集相同长度, 方便计算

    distances = in_data_extended - data_set
    distances = distances ** 2 # 这里没有开方, 加速运算
    distances = np.sum(distances, axis=1) # 计算数据到数据集中每个样本的距离, m*1

    distances_index = np.argsort(distances) # 从小到大排列, 返回对应index
    
    # 统计k近邻的分类
    class_count = {}
    for i in range(k):
        class_vote = labels[distances_index[i]]
        class_count[class_vote] = class_count.get(class_vote, 0) + 1 # 分类计数, 如果字典有key的value, get()返回value, 否则get()返回设定的值

    # 对分类统计进行排序, 得到分类
    class_count_sorted = sorted(class_count.items(), key=lambda x:x[1], reverse=True) # 字典排序: x[0]时, 对key进行排序; x[1]时, 对value进行排序; 返回[(key, value)]列表
    classed = class_count_sorted[0][0]

    return classed

def file2matrix(filename):
    '''
    ### Docs: 
        - 从文件中读取数据, 转换成matrix
        - 文件存储数据格式为: feature1, feature2, feature3, ..., label
        - 数据没有标题, 以table隔开
    ### Args:
        - filename: str, 文件名
    ### Returns:
        - feature_mat: m*n, array, m行数据, n列特征
        - label_mat: list, m个数据
    ### Examples:
    '''

    # 预先获取数据大小
    data_len = 0
    feature_len = 0
    with open(filename) as fr:
        lines = fr.readlines()
        data_len = len(lines)
        line = lines[0]
        line = line.strip()
        feature_len = len(line.split('\t'))
    feature_mat = np.zeros((data_len, feature_len - 1))
    # label_mat = np.zeros(data_len) # np.zeros(data_len)与np.zeros((data_len,1))不同!!!!
    label_mat = [0] * data_len

    # 读取数据
    with open(filename) as fr:
        for index, line in enumerate(fr.readlines()):
            line = line.strip()
            data_line = line.split('\t')
            feature_mat[index, :] = [float(data) for data in data_line[:-1]]
            # label_mat[index] = float(data_line[-1])
            label_mat[index] = data_line[-1]

    return feature_mat, label_mat

def data_norm(data_set):
    '''
    ### Docs:
        - 数据归一化到[0,1]
        - norm = (data - min) / (max - min)
    ### Args:
        - data_set: m*n, array, m行数据, n列特征
    ### Returns:
        - data_normed: m*n, array, m行数据, n列特征
        - 
    '''

    data_min = np.min(data_set, axis=0) # 计算每个特征的最小值, 1*n
    data_max = np.max(data_set, axis=0)
    data_range = data_max - data_min

    data_num = data_set.shape[0] # 数据个数
    data_min_extended = np.tile(data_min, (data_num, 1)) # 将数据扩展到与数据长度一致, m*n, 方便计算
    data_range_extended = np.tile(data_range, (data_num, 1))
    data_normed = (data_set - data_min_extended) / data_range_extended
    return data_normed, data_range, data_min

def datingClassTest():
    hoRatio = 0.50
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    normMat, ranges, minVals = data_norm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify_kNN(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    
def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    trainingFileList = listdir('data/trainingDigits')
    m = len(trainingFileList)
    hwLabels = [0] * m
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels[i] = (classNumStr)
        trainingMat[i, :] = img2vector('data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector('data/testDigits/%s' % fileNameStr)
        classifierResult = classify_kNN(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))

# datingClassTest()
handwritingClassTest()
