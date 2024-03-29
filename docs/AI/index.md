# AI入门
1. 第一暑假以及大二上学期通过CS50的学习入门AI。详情请跳转[CS50]()
2. Pytorch中文[手册](https://handbook.pytorch.wiki/chapter1/1_tensor_tutorial.ipynb)
3. 基本操作、3b1b或者tensorflow playground可以让你有些直观理解



## 简单的一些概念

>  建议入门可以看些3b1b的直观视频，再去学习cs50等课程内容

1. 以最简单的手写数字为例，对于简单神经网络的数学而言，每个神经元可以看作一个数字具有激活值，我们可以把输入（例如把二维灰度值转换成一维）看成一个列向量，权重值看为矩阵，bias为附加列向量，以及最后附加到整体的激活函数。最后通过代价函数loss来计算结果上的偏差，这便是一次网络训练的过程。

   > 梯度下降和反向传播:根据一次训练我们即可指导初始设定的weight和bias的好坏，但如何该改进是个问题？

2. 梯度下降：找到局部最小值，学习率即为步长。如何找到下降的方向便是反向传播的。反向传播会在很多地方求平均来进行微调。是来求单个训练样本想怎样修改权重和偏置。
