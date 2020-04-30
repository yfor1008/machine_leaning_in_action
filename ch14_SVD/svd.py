#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : svd.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/15 19:45:06
# @Docs   : SVD奇异值分解
'''

import numpy as np
import numpy.linalg as la

def ecludSim(inA, inB):
    '''
    ### Docs: 欧几里得距离相似性
        - 相似度=1/(1+距离), (0, 1]
    ### Args:
        - inA, vector, 数据A
        - inB, vector, 数据B
    ### Returns:
        - sim: float, shape与inA/inB一致
    '''

    sim = 1.0/(1.0 + la.norm(inA - inB))
    return sim

def pearsSim(inA, inB):
    '''
    ### Docs: 相关系数(皮尔逊相关系数)
        - 相关性=sum[(xi-mux)(yi-muy)]/[sqrt(sum(xi-mux)^2)*sqrt(sum(yi-muy)^2)], [-1, 1]
        - 归一化到[0, 1]
    ### Args:
        - inA, vector, 数据A
        - inB, vector, 数据B
    ### Returns:
        - sim: float
    '''

    if len(inA) < 3:
        sim = 1.0
    else:
        sim = 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0, 1]
    return sim

def cosSim(inA, inB):
    '''
    ### Docs: 余弦相似度
        - 相似度=A*B/(|A|*|B|), 这里为向量相乘, [-1, 1]
        - 归一化到[0, 1]
    ### Args:
        - inA, vector, 数据A
        - inB, vector, 数据B
    ### Returns:
        - sim: float
    '''

    num = np.sum(inA * inB)
    denom = la.norm(inA)*la.norm(inB)
    sim = 0.5 + 0.5*(num/denom)
    return sim

def loadExData3():
    '''
    ### Docs: 每行代表一个人, 列代表菜名, 值为人对菜肴的评分, 0表示为评分
    '''
    dataSet = [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]
    return np.array(dataSet)


def loadExData2():
    # 书上代码给的示例矩阵
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def loadExData():
    """
    # 推荐引擎示例矩阵
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]
    """
    # # 原矩阵
    # return[[1, 1, 1, 0, 0],
    #        [2, 2, 2, 0, 0],
    #        [1, 1, 1, 0, 0],
    #        [5, 5, 5, 0, 0],
    #        [1, 1, 0, 2, 2],
    #        [0, 0, 0, 3, 3],
    #        [0, 0, 0, 1, 1]]

    # 原矩阵
    return[[0, -1.6, 0.6],
           [0, 1.2, 0.8],
           [0, 0, 0],
           [0, 0, 0]]

# 基于物品相似度的推荐引擎
def standEst(dataSet, user, simMeas, item):
    '''
    ### Docs: 基于物品相似度的推荐引擎, 协同滤波(Collaborative Filtering, CF)
    ### Args:
        - dataSet: m*n, array, m行用户, n列物品
        - user: int, 用户index
        - simMeas: object, 相似度计算方法
        - item: int, 物品index
    ### Returns:
        - 
    '''
    """standEst(计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分)
    Collaborative Filtering （CF）
    Args:
        dataSet         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 得到数据集中的物品数目
    # n = np.shape(dataSet)[1]
    m, n = dataSet.shape
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataSet[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        # 变量 overLap 给出的是两个物品当中已经被评分的那个元素的索引ID
        # logical_and 计算x1和x2元素的真值。
        overLap = np.nonzero(np.logical_and(dataSet[:, item] > 0, dataSet[:, j] > 0))[0]
        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = simMeas(dataSet[overLap, item], dataSet[overLap, j])
        # print('the %d and %d similarity is : %f'(iten,j,similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal


# 基于SVD的评分估计
# 在recommend() 中，这个函数用于替换对standEst()的调用，该函数对给定用户给定物品构建了一个评分估计值
def svdEst(dataSet, user, simMeas, item):
    """svdEst( )
    Args:
        dataSet         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        item            未评分的物品编号
    Returns:
        ratSimTotal/simTotal     评分（0～5之间的值）
    """
    # 物品数目
    n = np.shape(dataSet)[1]
    # 对数据集进行SVD分解
    simTotal = 0.0
    ratSimTotal = 0.0
    # 奇异值分解
    # 在SVD分解之后，我们只利用包含了90%能量值的奇异值，这些奇异值会以NumPy数组的形式得以保存
    U, Sigma, VT = la.svd(dataSet)

    # # 分析 Sigma 的长度取值
    # analyse_data(Sigma, 20)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = np.mat(eye(4) * Sigma[: 4])

    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataSet.T * U[:, :4] * Sig4.I
    print('dataSet', shape(dataSet))
    print('U[:, :4]', shape(U[:, :4]))
    print('Sig4.I', shape(Sig4.I))
    print('VT[:4, :]', shape(VT[:4, :]))
    print('xformedItems', shape(xformedItems))

    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
    # 这和standEst()函数中的for循环的目的一样，只不过这里的相似度计算时在低维空间下进行的。
    for j in range(n):
        userRating = dataSet[user, j]
        if userRating == 0 or j == item:
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度不断累加求和
        simTotal += similarity
        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal/simTotal


# recommend()函数，就是推荐引擎，它默认调用standEst()函数，产生了最高的N个推荐结果。
# 如果不指定N的大小，则默认值为3。该函数另外的参数还包括相似度计算方法和估计方法
def recommend(dataSet, user, N=3, simMeas=cosSim, estMethod=standEst):
    """svdEst( )
    Args:
        dataSet         训练数据集
        user            用户编号
        simMeas         相似度计算方法
        estMethod       使用的推荐算法
    Returns:
        返回最终 N 个推荐结果
    """
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    # unratedItems = np.nonzero(dataSet[user, :].A == 0)[1]
    unratedItems = np.nonzero(dataSet[user, :] == 0)[0]

    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'
    # 物品的编号和评分值
    itemScores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        # 获取 item 该物品的评分
        estimatedScore = estMethod(dataSet, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]


def analyse_data(Sigma, loopNum=20):
    """analyse_data(分析 Sigma 的长度取值)
    Args:
        Sigma         Sigma的值
        loopNum       循环次数
    """
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2
    SigmaSum = np.sum(Sig2)
    for i in range(loopNum):
        SigmaI = np.sum(Sig2[:i+1])
        '''
        根据自己的业务情况，就行处理，设置对应的 Singma 次数
        通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
        '''
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '4.2f')))



if __name__ == "__main__":

    dataSet = loadExData3()

    # print(ecludSim(dataSet[1, :], dataSet[2, :]))
    # print(pearsSim(dataSet[1, :], dataSet[2, :]))
    # print(cosSim(dataSet[1, :], dataSet[2, :]))

    print(recommend(dataSet, 2))
