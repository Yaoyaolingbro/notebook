# Instruction-Level Parallelism
## Pipelining
这里几乎都是机组的复习
![20241028103230.png](graph/20241028103230.png)

### What about floating point?
> Except the fp division, all the other fp operations are pipelined.
![20241220122622.png](graph/20241220122622.png)

之后会讲pipeline中遇到的不同hazard。

### How to detect fp hazard?
提供了一种解决方案： MIPS 4000
![20241220151054.png](graph/20241220151054.png)

## ILP exploit
