#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : bayes.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/09 20:56:51
# @Docs   : 贝叶斯分类
'''

import numpy as np
from functools import reduce

def loadDataSet():
    '''
    ### Docs: 生成数据集
    ### Returns:
        - postingList: m*x, list, 切分后的数据, m为词条个数, x为每个词条中的单词
        - classVec: m*1, list, 词条对应的分类标签: 1-侮辱类; 0-非侮辱类
    '''
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    '''
    ### Docs: 生成词汇表, 所有词条中不重复的单词
    ### Args:
        - dataSet: m*x, list, 切分后的数据, m为词条个数, x为每个词条中的单词
    ### Returns:
        - vocabList: list, 所有词汇构成的list
    '''
    vocabSet = set([])
    for document in dataSet:                
        vocabSet = vocabSet | set(document) # 使用集合来去除重复

    vocabList = list(vocabSet)
    return sorted(vocabList)

def setOfWords2Vec(vocabList, inputSet):
    '''
    ### Docs:
        - 利用词汇表, 将输入词条向量化
        - 向量长度为词汇表的长度, 与词汇表一一对应
        - 对于输入词条中的每个单词, 若其存在词汇表中, 则词汇表中单词对应位置为1, 否则为0
    ### Args:
        - vocabList: list, 所有词汇构成的list
        - inputSet: list, 切分成单词后的词条list
    ### Returns:
        - returnVec: list, 向量化的词条, 长度与vocabList一致
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1 # 如果词条存在于词汇表中，则对应位置为1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    '''
    ### Docs:
        - 利用词汇表, 将输入词条向量化
        - 向量长度为词汇表的长度, 与词汇表一一对应
        - 对于输入词条中的每个单词, 若其存在词汇表中, 则词汇表中单词对应位置+1, 否则为0
    ### Args:
        - vocabList: list, 所有词汇构成的list
        - inputSet: list, 切分成单词后的词条list
    ### Returns:
        - returnVec: list, 向量化的词条, 长度与vocabList一致
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 # 如果词条存在于词汇表中，则对应位置+1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB(trainMatrix, trainCategory):
    '''
    ### Docs: 朴素贝叶斯分类器训练函数
    ### Args:
        - trainMatrix: m*n, array, 词条向量化后的数据, m个数据, n个特征属性
        - trainCategory: m*1, list 词条对应的分类标签
    ### Returns:
        - prob_c: y*1, vector, 每个类别对应的概率, y为类别个数
        - prob_xi_c: y*n, array, 类别c中每个属性xi的条件概率
        - prob_tag: y*1, list, 概率对应的分类标签
    '''
    numTrainDocs = len(trainCategory) # 词条数量, 即公式中的D
    numWords = len(trainMatrix[0])  # 词汇表的长度, 即公式中的d
    prob_tag = list(set(trainCategory))
    numClasses = len(prob_tag) # 分类标签个数
    
    # 计算每个类别的概率P(c)=|Dc|/|D|, readme中的公式1
    prob_c = np.zeros(numClasses)
    for class_tag in trainCategory:
        class_idx = prob_tag.index(class_tag)
        prob_c[class_idx] += 1 # 统计Dc
    prob_c /= float(numTrainDocs) # 计算P(c)
    prob_c = np.log(prob_c) # 取对数, 对应公式3-1

    # 计算c类别中每个单词的条件概率P(xi|c)=|Dc,xi|/|Dc,x|, readme中的公式2
    d_c_xi = np.ones((numClasses, numWords)) # 初始化为1, 避免为0
    d_c_x = np.ones((numClasses, 1)) * 2 # 初始化为2
    for class_tag in prob_tag:
        class_idx = prob_tag.index(class_tag)
        for index_doc in range(numTrainDocs):
            if trainCategory[index_doc] == class_tag:
                d_c_xi[class_idx] += trainMatrix[index_doc] # 公式中的 |Dc,xi|
                d_c_x[class_idx] += sum(trainMatrix[index_doc]) # 公式中的 |Dc,x|
    prob_xi_c = d_c_xi / np.tile(d_c_x, (1, numWords))
    prob_xi_c = np.log(prob_xi_c) # 取对数, 对应公式3-1
    return prob_c, prob_xi_c, prob_tag


def classifyNB(vec2Classify, prob_c, prob_xi_c, prob_tag):
    '''
    ### Docs: 朴素贝叶斯分类器
    ### Args:
        - vec2Classify: 1*n, array, 向量化后待分类的词条, n为特征属性个数
        - prob_c: y*1, vector, 每个类别对应的概率, y为类别个数
        - prob_xi_c: y*n, array, 类别c中每个属性xi的条件概率
        - prob_tag: y*1, list, 概率对应的分类标签
    ### Returns:
        - class_label: 分类标签
    '''

    numWords = len(vec2Classify)  # 词汇表的长度

    # 计算属于类别c的概率P(c|x), readme中的公式3-1
    prob_c_x = np.sum(prob_xi_c * vec2Classify, axis=1) + prob_c

    # 查找最大类别, readme中的公式4
    prob_max_index = np.argmax(prob_c_x)
    class_label = prob_tag[prob_max_index]

    return class_label

def testingNB():
    '''
    ### Docs: 测试贝叶斯分类器
    '''
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    prob_c, prob_xi_c, prob_tag = trainNB(np.array(trainMat), listClasses)

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    if classifyNB(np.array(thisDoc), prob_c, prob_xi_c, prob_tag):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(np.array(thisDoc), prob_c, prob_xi_c, prob_tag):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList) # 词汇表
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        # 随机选取10个样本作为测试集
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        # 训练样本
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    prob_c, prob_xi_c, prob_tag = trainNB(np.array(trainMat), trainClasses) # 模型训练
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), prob_c, prob_xi_c, prob_tag) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))


if __name__ == '__main__':

    # testingNB()

    spamTest()

