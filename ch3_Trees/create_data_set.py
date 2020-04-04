#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : create_data_set.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/03 09:50:04
# @Docs   : 生成测试数据
'''

def creat_data_set():
    '''
    ### Docs: 贷款申请数据集
        - 年龄: 0代表青年, 1代表中年, 2代表老年
        - 有工作: 0代表否, 1代表是
        - 有自己的房子: 0代表否, 1代表是
        - 信贷情况: 0代表一般, 1代表好, 2代表非常好
        - 类别(是否给贷款): no代表否, yes代表是
    ### Returns:
        - data_set: m*n, list, m行数据, n列特征
        - data_tag: 1*n, list, 特征属性说明
        - labels: m*1, list, 数据分类标签
    '''
    # 数据集
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]

    data_set, labels = [data[:-1] for data in data_set], [data[-1] for data in data_set] # list并列操作
    # 数据属性
    data_tag = ['年龄', '有工作', '有自己的房子', '信贷情况', '是否放款']
    return data_set, data_tag, labels

if __name__=='__main__':
    data_set, data_tag, labels = creat_data_set()
    print(data_set)
    print(labels)
    print(data_tag)