#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : tress.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/03 13:34:52
# @Docs   : 决策树相关
'''

from calc_entropy import calc_entropy

def best_feature(data_set, labels):
    '''
    ### Docs: 使用信息增益选取最优的特征
        - 信息增益是相对于特征而言的
        - 特征A对训练数据集D的信息增益g(D,A), 定义为集合D的熵H(D)与特征A给定条件下D的条件熵H(D|A)之差, 即
        - g(D,A) = H(D) − H(D|A)
    ### Args:
        - data_set: m*n, list, 数据集, m行数据, n列特征
        - labels: m*1, list, 数据分类标签
    ### Returns:
        - best_feature_index: int, 最好分类的特征index
    '''

    data_num = len(data_set)
    feature_num = len(data_set[0])
    entropy_base = calc_entropy(labels)

    best_info_gain = 0.0
    best_feature_index = -1

    for feature_idx in range(feature_num):
        feature_col = [data[feature_idx] for data in data_set] # 当前列特征数据
        feature_unique = set(feature_col) # 这里特征数据是离散的, 直接按值统计分类

        entropy_condition = 0.0
        for feature_value in feature_unique:
            _, labels_sub = split_data_set(data_set, labels, feature_idx, feature_value)
            prob = len(labels_sub) / float(data_num) # 自数据集站总数据集的比例
            entropy_condition += prob * calc_entropy(labels_sub) # 条件熵

        info_gain = entropy_base - entropy_condition # 信息增益
        print("\t第%d个特征的增益为%.3f" % (feature_idx, info_gain))
        if info_gain > best_info_gain:
            # 最高信息增益
            best_info_gain = info_gain
            best_feature_index = feature_idx

    return best_feature_index

def split_data_set(data_set, labels, feature_idx, feature_value):
    '''
    ### Docs: 取第feature_idx维特征中所有值为feature_value的子集
    ### Args:
        - data_set: m*n, list, 数据集, m行数据, n列特征
        - labels: m*1, list, 数据分类标签
        - feature_idx: int, 特征数据序号, 第几个特征
        - feature_value: 需要挑选出来的特征
    ### Returns:
        - data_set_sub: x*(n-1), list, 挑选出来的数据子集
        - labels_sub: x*1, list, 挑选出来的数据子集对应的标签
    '''

    data_set_sub = []
    labels_sub = []
    for index, feature_vector in enumerate(data_set):
        if feature_vector[feature_idx] == feature_value:
            # 去除当前列, 并且仅挑选特征值为feature_value的数据
            feature_vector_new = feature_vector[:feature_idx]
            feature_vector_new.extend(feature_vector[feature_idx+1:])
            data_set_sub.append(feature_vector_new)
            labels_sub.append(labels[index])

    return data_set_sub, labels_sub

def create_tree(data_set, labels, data_tag, tree_node):
    '''
    ### Docs: 使用ID3创建决策树
        - ID3算法的核心是在决策树各个结点上对应信息增益准则选择特征，递归地构建决策树, 具体方法是:
        - 从根节点开始, 对结点计算所有特征的信息增益, 选择最大信息增益的特征作为该结点特征
        - 由该特征的不同取值建立子节点, 然后递归调用上述方法, 构建决策树
        - 直到所有特征的信息增益均很小或者没有特征可以选择为止
    ### Args:
        - data_set: m*n, list, 数据集, m行数据, n列特征
        - data_tag: 1*n, list, 特征属性说明
        - labels: m*1, list, 数据分类标签
        - tree_node: list, 返回结点对应的特征tag
    ### Returns:
        - my_tree: 嵌套字典, 生成的决策树
    '''

    if labels.count(labels[0]) == len(labels):
        # 如果类别完全相同, 则停止继续划分
        return labels[0]
    
    feature_num = len(data_set[0])
    if feature_num == 1:
        # 如果没有待分类特征时, 停止划分
        return major_cnt(labels)

    print(data_set)
    best_feature_index = best_feature(data_set, labels) # 最优特征的index
    print('最优特征为: %d' % best_feature_index, data_tag)
    best_feature_tag = data_tag[best_feature_index] # 最优特征对应的tag

    tree_node.append(best_feature_tag) # 返回结点对应的特征tag

    # 生成最优特征的标签树
    my_tree = {best_feature_tag: {}}
    data_tag_copy = data_tag[:]
    del(data_tag_copy[best_feature_index])
    feature_col = [data[best_feature_index] for data in data_set] # 最优特征所有列的所有数据
    feature_unique = set(feature_col)
    for feature_value in feature_unique:
        data_set_sub, labels_sub = split_data_set(data_set, labels, best_feature_index, feature_value)
        my_tree[best_feature_tag][feature_value] = create_tree(data_set_sub, labels_sub, data_tag_copy, tree_node)

    return my_tree

def major_cnt(datas):
    '''
    ### Docs: 统计数据中出现最多的数
    ### Args:
        - datas: list, 待统计的数据
    ### Returns:
        - data_cnt_max: 出现次数最多数据
    '''

    data_cnt = {}
    for data in datas:
        data_cnt[data] = data_cnt.get(data, 0) + 1 # 如果字典有key的value, get()返回value, 否则get()返回设定的值
    
    # 字典排序: x[0]时, 对key进行排序; x[1]时, 对value进行排序; 返回[(key, value)]列表
    data_cnt_sorted = sorted(data_cnt.items(), key=lambda x:x[1], reverse=True)
    data_cnt_max = data_cnt_sorted[0][0]
    return data_cnt_max

def get_tree_leafs(my_tree):
    '''
    ### Docs: 获取决策树的叶子结点
    ### Args:
        - my_tree: 嵌套字典, 决策树
    ### Returns:
        - leaf_nums: int, 叶子节点的数目
    '''

    leaf_nums = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 不是叶子结点, 继续递归查找
            leaf_nums += get_tree_leafs(second_dict[key])
        else:
            leaf_nums += 1
    return leaf_nums

def get_tree_depth(my_tree):
    '''
    ### Docs: 获取决策树的深度
    ### Args:
        - my_tree: 嵌套字典, 决策树
    ### Returns:
        - max_depth: int, 叶子节点的数目
    '''

    max_depth = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            # 不是叶子结点, 继续递归查找
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth

def classify_tree(test_vector, my_tree, best_feature_tags):
    '''
    ### Docs: 使用决策树分类
    ### Args:
        - test_vector: list, 测试数据, 顺序对应最优特征
        - my_tree: 生成的决策树
        - best_feature_tags: 最优特征标签
    ### Returns:
        - classed_label: 分类结果
    '''

    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    feature_idx = best_feature_tags.index(first_str)

    for key in second_dict.keys():
        if test_vector[feature_idx] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classed_label = classify_tree(test_vector, second_dict[key], best_feature_tags)
            else:
                classed_label = second_dict[key]
    return classed_label

if __name__ == "__main__":

    from create_data_set import creat_data_set
    data_set, data_tag, labels = creat_data_set()
    # print("最优索引值："+str(best_feature(data_set, labels)))

    tree_node = []
    my_tree = create_tree(data_set, labels,  data_tag, tree_node)
    print(my_tree)
    print(get_tree_leafs(my_tree))
    print(get_tree_depth(my_tree))

    test_vector = [0, 1]
    result=classify_tree(test_vector, my_tree, tree_node)
    if result=='yes':
        print('放贷')
    if result=='no':
        print('不放贷')

