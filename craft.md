## External Sorting
> num of pass = $1+log_{k}^{N/M}$
>
> 普通放置需要2k tape
> 斐波纳切数列安排run k+1 tape
>
> 可以利用replace section的方式，获得longer的run length
>
> k-way merge： 2k input buffers, 2 output buffers


seek 和 transfer的时间主要是花在哪里？


算法本身的时间复杂度并未脱离mergesort


![20240618214224.png](graph/20240618214224.png)


> 逆天的坑人题目
![20240618224352.png](graph/20240618224352.png)