#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : calc_entropy.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/03 10:58:28
# @Docs   : 计算熵
'''

from math import log

def calc_entropy(datas):
    '''
    ### Docs: 计算数据的熵
    ### Args:
        - datas: list, 数据
    ### Returns:
        - enrtopy: float, 熵
    '''

    data_len = len(datas)
    data_cnt = {}
    for data in datas:
        data_cnt[data] = data_cnt.get(data, 0) + 1 # 如果字典有key的value, get()返回value, 否则get()返回设定的值
    
    # 计算熵
    probs = [float(data_cnt[key])/data_len for key in data_cnt]
    probs = [-prob * log(prob, 2) for prob in probs]
    entropy = sum(probs)

    return entropy

if __name__=='__main__':

    from create_data_set import creat_data_set
    data_set, data_tag, labels = creat_data_set()

    print(calc_entropy(labels))
