#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : plot_tree.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2020/04/04 13:54:48
# @Docs   : 绘制决策树
'''

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from trees import create_tree, get_tree_leafs, get_tree_depth

def plot_node(note_txt, center_pt, parent_pt, node_type):
    '''
    ### Docs: 绘制结点
    ### Args:
        - note_txt: str, 结点名
        - center_pt: (x,y), 绘制位置
        - parent_pt: (x,y), 父节点位置
        - node_type: 
    '''
    arrow_args = dict(arrowstyle='<-')
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    create_plot.ax1.annotate(note_txt, 
                            xy=parent_pt, xycoords='axes fraction', 
                            xytext=center_pt, textcoords='axes fraction',
                            va='center', ha='center',
                            bbox=node_type, arrowprops=arrow_args, FontProperties=font)

def plot_mid_text(center_pt, parent_pt, node_txt):
    '''
    ### Docs: 标注有向边的属性
    ### Args:
        - center_pt: (x,y), 绘制位置
        - parent_pt: (x,y), 父节点位置
        - note_txt: str, 结点名
    '''

    x_mid = (parent_pt[0] - center_pt[0]) / 2.0 + center_pt[0]
    y_mid = (parent_pt[1] - center_pt[1]) / 2.0 + center_pt[1]
    create_plot.ax1.text(x_mid, y_mid, node_txt, va='center', ha='center', rotation=30)

def plot_tree(my_tree, parent_pt, node_txt):
    '''
    ### Docs: 绘制决策树
    ### Args:
        - my_tree: 嵌套字典, 决策树
        - parent_pt:
        - node_txt: 节点名
    '''
    decision_node = dict(boxstyle='sawtooth', fc='0.8') # 结点格式
    leaf_node = dict(boxstyle='round4', fc='0.8') # 叶子格式
    leaf_nums = get_tree_leafs(my_tree)
    depth = get_tree_depth(my_tree)

    first_str = next(iter(my_tree))
    center_pt = (plot_tree.xOff + (1.0 + float(leaf_nums)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(center_pt, parent_pt, node_txt) # 绘制有向边
    plot_node(first_str, center_pt, parent_pt, decision_node) # 绘制结点
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.totalD

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], center_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), center_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD

def create_plot(my_tree):
    '''
    ### Docs: 创建绘制面板
    ### Args:
        - my_tree: 嵌套字典, 决策树
    '''
    fig = plt.figure(1, facecolor='white')
    fig = fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_tree_leafs(my_tree))
    plot_tree.totalD = float(get_tree_depth(my_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW; plot_tree.yOff = 1.0
    plot_tree(my_tree, (0.5, 1.0), '')
    plt.show()

if __name__ == "__main__":

    # from create_data_set import creat_data_set
    # data_set, data_tag, labels = creat_data_set()
    # tree_node = []
    # my_tree = create_tree(data_set, labels,  data_tag, tree_node)
    # print(my_tree)
    # create_plot(my_tree)

    # 测试隐形眼镜数据
    with open('lenses.txt', 'r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    
    lense_set, labels = [lense[:-1] for lense in lenses], [lense[-1] for lense in lenses]
    lense_tag = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree_node = []
    lenses_tree = create_tree(lense_set, labels, lense_tag, tree_node)
    create_plot(lenses_tree)
