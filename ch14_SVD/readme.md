# SVD

奇异值分解(Singular Value Decomposition, 以下简称SVD)是在机器学习领域广泛应用的算法, 它不光可以用于降维算法中的特征分解, 还可以用于推荐系统, 以及自然语言处理等领域, 是很多机器学习算法的基石.

## 定义

对于 $m*n$ 的矩阵A, 可以分解为以下形式:
$$
A = U \Sigma V^T \tag{公式1}
$$
其中:

- $U$ 和 $V$ 均为单位正交矩阵, 满足 $U U^T = I$ , $V V^T = I$ 
- $U$ 为左奇异向量构成的矩阵, 大小为 $m*m$ 
- $V$ 为右奇异向量构成的矩阵, 大小为 $n*n$ 
- $\Sigma$ 为奇异值构成的对角矩阵, 大小为 $m*n$ 

求上述矩阵的过程为SVD.

## 求解过程

- 步骤1: 计算 $A A^T$ (大小为 $m*m$ )和 $A^T A$ (大小为 $n*n$ )
- 步骤2: 分别计算 $A A^T$ 和  $A^TA$ 的特征向量及其特征值
- 步骤3:  $A A^T$ 的特征向量构成矩阵  $U$ ,   $A^TA$ 的特征向量构成矩阵 $V$ 
- 步骤4: 对 $A A^T$ 和  $A^TA$ 的非0特征值取平方根, 对应上述特征向量的位置, 填入 $\Sigma$ 对角线

注:

- $A A^T$ 和  $A^TA$ 的特征值相同
- $A A^T$ 和  $A^TA$ 都为方阵, 方便计算特征值与特征向量
- $\Sigma$ 为特征值的开方, 一般从大到小排列

##  数据降维

数据降维可以用如下公式表示:
$$
A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V_{n \times n}^T \approx U_{m \times k} \Sigma_{k \times k} V_{n \times k}^T \tag{公式2}
$$
其中, $U_{m \times k}$ , $\Sigma_{k \times k}$ ,  $V_{n \times k}$ , 为降维后的数据.

## 优缺点

- 优点: 简化数据, 去除噪声, 优化算法的结果
- 缺点: 数据的转换可能难以理解
- 适用数据类型: 数值型数据





参考

1. https://www.cnblogs.com/pinard/p/6251584.html
2. https://www.cnblogs.com/baiboy/p/pybnc11.html
3. https://www.zhihu.com/question/22237507?sort=created
4. https://www.cnblogs.com/endlesscoding/p/10033527.html
5. https://ww2.mathworks.cn/help/matlab/ref/double.svd.html

