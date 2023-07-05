# 超算短学期笔记



## 7月3号 —— 一些些介绍

* 大家的自我介绍以及一点点入门
* 一点点计网的知识



## 7月4日 —— 计算机体系结构和高性能计算基础



#### Overview

* ISA and x86 Instructions
* Processor Architecture
* Memory Hierarchy
* Concurrency Basic

#### Prerequisite Checklist

* Digital representation of values
* Memory & Address
* C code

#### ISA & x86 instruction

1. ISA: instruction ser architecture.
1. Assembly Language

#### Processor Architecture

1. Microarchitecture is implementation.(could make faster)
2. thread（线程）
3. fetch -> Decode -> Execute -> Commit 
4. Data hazards: You haven't already written but the next process need to call the data.
5. control hazard: parallel processing could make a fault.
6. structure hazard : like fetch and execute could make conflict.
7. SRAM & DRAM：SRAM = cache（高缓冲）.

#### Memory Layout

1. Stack: Runtime stack (8MB limit) local variables.
2. Heap: Dynamically allocated as needed.
3. Data: statically allocated data.
4. virtual memory: OS utilizes virtual memory to isolate address space of different processes and provide each process the same linear address space(线性映射，类似于哈希)
5. Translation Lookaside buffer（TLB） -> could accelerate the virtual memory and physical memory interactive.
6. NVM: between SSD and SRAM.
7. Cache Organization(valid bit; dirty bit ; Tag: check if matching)
8. Cache usage : Read hit ; Read miss ; Write hit(write-back); Write miss(write-allocate)
9. Multiple Cache Hierarchy。（L1 L2 L3)
10. Processes vs. Threads.

> differences: Threads in one process share memory but processes not.

```
//当两个线程一块进行这个代码时，最小的计算结果是2，思考为什么？
for(int i=1; i < 10000; i++)
{
	counter++;
}
```

#### x86 microarchitecture



## 7月5日 —— 高性能计算方法论（HPC）

### Overview

* Basic Theories for HPC
* Performance Analysis and Optimization Methodology
* Practical Optimization Strategies
* HPC Skill Tree
* How to Learn HPC/CS

### Basic Theories for HPC

Factor affecting performance:

1. Algorithms
2. Models 
3. Software
4. Hardware
5. Physics

> Example: Large Matrix Multiplication(详情可看AIPP中的MPI优化和BLAS矩阵计算，GPU速度会比CPU跑得快)



### Performance Analysis and Optimization Methodology

1. 斐波那契数列计算，编译器会有优化（O2、O3）。可以通过IDA反编译看看实际运算的代码。
2. Maximize performance： Speed、Throughout、Latency（延迟）or Resource is limited（quota配额）
3. black box  Dominant component

> Roofline Performance Mode:
>
> Arithmetic Intensity(AI) = FLOP's/Bytes (this could judge the performance of program)
>
> 屋顶线可以判断 CPU 和缓存的使用情况。我们是的最终目的是为了让它达到拐点！
>
> 而2020后有深度学习模型来训练黑箱测整体性能

4. Amadal’s law（水桶效应，补全最短的）
5. Methods ： Analysis in math； Hardware simulator； Profile: sampling some usage of a resource; Trace: collecting highly detailed data about the execution of a system.
6. General Optimization Pipeline



### Practical Optimization Strategies

1. Algorithm Optimization - Prefetch & Prediction
2. Caching ：stores results from previous executions ; Limited cache size.
3. Lock - Free: Use atomic primitives(CAS Atomic_add)

> Negative example: GIL in Python

4. Load Balancing(make or cores to work)
5. Reduce Precision(精度)
6. Reduce Branching(skip list or like binary tree of branch)
7. Vectorization(High-level: vectorized computation graph ; Instruction-level: SIMD instructions)

> See in your lab2

8. Optimize Memory Access Locality

> * GEMM
>   * Blocking
>   * Loop Permutation(排列)
>   * Array Packing
>
> See in your lab3

9. Instruction / Data Alignment

> eg: compiler could auto optimize.（例如结构体会内存自动对齐）



### Discussion

* Domain Specific Language 
* Manual Optimization is indispensable
* Core Affinity（亲和力）（[NUMA](https://zh.wikipedia.org/zh-cn/%E9%9D%9E%E5%9D%87%E5%8C%80%E8%AE%BF%E5%AD%98%E6%A8%A1%E5%9E%8B) **non-uniform memory access**）
* Adapts general code to local machine
* Auto - learning eg. black-box method : TVM
* You can learn something about TPU and DPU and FPGA.



### HPC skill tree

* Linux:  操作系统相关知识、Linux基本结构、 Shell使用
* 集群运维和网络管理（分布式）：NFS；
* 协作开发与版本控制
* 脚本自动化（Linux shell 或者 Python）
* 带依赖程序的手动编译链接
* 并行程序设计、测试和优化
* 功耗控制与调参



### 如何学习

