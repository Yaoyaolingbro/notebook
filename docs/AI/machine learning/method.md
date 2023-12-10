# 机器学习分类与回归
<!-- prettier-ignore-start -->
!!! abstract "目录"
    [TOC]

    === "资源"
    [YouTube]()
    [code](https://github.com/Suji04/ML_from_Scratch)
<!-- prettier-ignore-end -->
## 1. 决策树和随记森林
### 1.1.1 decision tree classification
机器学习中的决策树就好似搜索树一样（请与fds中学到的决策树区分），能够对非线性分割的目标进行逐步拆分。如图：
![](graph/decision_tree.png)

而当我们探讨，$\text{which split is better?}$ 这个问题的时候, 我们需要`information gain`和 `entropy`两个概念

$$
Entropy = \sum{-p_i log(p_i)} \\
IG = E(parent) - \sum w_i E(child)
$$

![IG](graph/IG.png)

由此我们便可以计算出哪种分类方式更好。最后，我们确保每个叶节点都只要单一类型的元素。

代码建议阅读[源码](https://github.com/Suji04/ML_from_Scratch)看看！


