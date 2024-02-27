## 花书《深度学习》

机器学习的本质上属于应用统计学，更多地会关注于如何用计算机估计的统计复杂函数。因此我们会探讨两种统计学的主要方法：频率派估计和贝叶斯推断。机器学习算法可以大体分为监督/无监督/强化学习。大部分深度学习都是基于随机梯度下降的算法求解的。

### 学习算法

#### 任务T

1. 学习的过程不能算任务，学习是指我们所谓的获取完成任务的能力。我们一般将任务定义为机器学习系统应该如何处理样本。样本是指我们从某些对象或者事件当中收集到的已经量化的特征的集合。
2. 机器学习的任务主要有：分类、输入缺失分类（描述联合概率分布的函数）、回归（与分类相比更偏向于一种预测）、转录（如语音识别等）、机器翻译（例如英语转换成法语等）、结构化输出（类似于语法分析|图像分割等）、异常检测、合成与采样、缺失值填补、去噪、密度估计或概率质量估计

#### 性能度量P

我们通过度量模型的准确率，使用测试集来评估系统性能

#### 经验E

1. 大部分算法可以理解为在整个数据集上获取经验。
2. 由于概率的链式法则，联合分布可以拆解成n个监督学习问题。

#### 示例：线性回归

这里有数学推导，建议可以看看书。