# 第14讲 | 并行

有两种并行算法模型：PRAM（Parallel Random Access Machine）和 WD（Work-Depth）模型。

解决访问冲突的算法共有三种：
- CRCW（Concurrent Read Concurrent Write）不能同时读写一个位置
- CREW（Concurrent Read Exclusive Write）可以同时读，但不能同时写
- EREW（Exclusive Read Exclusive Write）可以同时读写

其中，WD模型更容易考察。

<!-- prettier-ignore-start -->
!!! note "测量性能"
    ![20240601171733.png](graph/20240601171733.png)
<!-- prettier-ignore-end -->

在WD模型中，可以通过任意P(n)个处理器在O(W(n)/P(n) + T(n))的时间内实现算法，使用与WD演示中相同的并发写约定。
> - Work Load 工作总量/操作总数量 W(N)，单核去跑需要的时间
> - 最坏的运行时间 T(N)/把T(N)理解为深度


下面让我来看三个案例分析


## Prefix Sum
> 从下到上计算B值，从上到下计算C值


## 数组归并
> 两个数组归并。不容易并行化，因为需要保证顺序。

### Partition
$Rank(i,\ B) = rank\ of A[i] \ in\ B$

```cpp
for i in 1 to n do
    j = Rank(i, B)
    k = Rank(i, A)
    C[i+j] = A[i]
    D[i+k] = B[i]
```

> 这个算法的$T(n) = O(logn)$，$W(n) = O(nlogn)$
> 传统serial算法的$T(n) = W(n) = O(n + m)$

### Parallel Rank
对拆分下去的子算法进行并行化，分成n/logn个子问题.
> $T(n) = O(logn)$，$W(n) = O(n)$

## 找最大值
### Base
> 把求和算法+改成max
> $T(n) = O(logn)$，$W(n) = O(n)$

### Parallel Rank
当然基于暴力对比的算法也是可以的，$T(n) = 1, W(n) = O(n^2)$
因此我们再想到多分机组，引出双对数级别的算法。$T(n) = O(loglogn)$，$W(n) = O(nloglogn)$

### Randomized
随机算法，高概率使得$T(n) = O(1)$，$W(n) = O(n)$

> 【Theorem】The algorithm finds the maximum among n elements. With very high probability it runs in O(1) time and O(n) work. The probability of not finishing within this time and work complexity is $O(1/n^c)$ for some positive constant c.