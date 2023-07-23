<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:10%;">
        </br>
        <img src="https://raw.githubusercontent.com/Keldos-Li/pictures/main/typora-latex-theme/ZJU-name.svg" alt="校名" style="width:100%;"/>
    </div>
    </br></br></br></br></br>
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:40%;">
        <img src="https://raw.githubusercontent.com/Keldos-Li/pictures/main/typora-latex-theme/ZJU-logo.svg" alt="校徽" style="width:100%;"/>
	</div>
    </br></br></br></br></br></br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">《基于CUDA对GEMM加速》</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto"lab3</p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 基于CUDA对GEMM的加速</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 23年暑假</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">ZJUSCT </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">姓　　名</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 杜宗泽</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">学　　号</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">3220105581 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">组　　别</td>
    		<td style="width:%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 个人</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">7月15日</td>     </tr>
    </tbody>              
    </table>
</div>




<!-- 注释语句：导出PDF时会在这里分页 -->

#　基于CUDA对GEMM的优化

[TOC]



## Lab Description

---

具体描述请见{实验手册}(https://zjusct.pages.zjusct.io/summer-course-2023/HPC101-Labs-2023/Lab3-Cuda/)

通用矩阵乘法（[General matrix multiply](https://en.wikipedia.org/wiki/General_matrix_multiply), GEMM）是 BLAS 中经典的子程序之一。[2] 作为当今科学计算最常见的计算任务之一，GEMM 需要实现一个非常高效的矩阵乘法。优化 GEMM 也是 HPC 界非常基础的任务。

本次实验需要你使用 CUDA 完成一个高性能 GEMM 实现。

**Bonus：**另外本次实验提供的 GPU 上，包含上述提及的 Tensor Core 模块。合理的使用它能够进一步加速卷积的计算。在 Cuda 9.0 之后，你可以使用内嵌 `PTX` 汇编或者 CUDA 的 C++ 扩展 `nvcuda::wmma` 的方式来显式地调用 Tensor Core 来进行计算。



## Introduction Knowledge(可以跳过不看)

---

1. CUDA使用：建议上官网。至于lab中提到的不同API的区别，可见[博客](https://blog.csdn.net/weixin_44966641/article/details/124500258)

2. 关于高性能计算矩阵乘法（GEMM）的[说明](https://www.cs.utexas.edu/users/pingali/CS378/2008sp/papers/gotoPaper.pdf). [CS217](https://zhuanlan.zhihu.com/p/280771849#:~:text=gotoBLAS中的GEMM实现就使用了分块算法。 从图中我们可以看到三种处理方法。,第一种是将A和B矩阵分块，第二种方法是将C和B矩阵分块，第三种方法是将C和A矩阵分块。 GEMM的子任务是GEPP或GEMP；最小粒度的任务是GEBP或GEPB或点乘。)

3. Introduction to shared memory.([link](https://zhuanlan.zhihu.com/p/597529982)) 

4. 一个Github上不同优化方法的[对比](https://github.com/mrzhuzhe/riven/tree/main/cuda_test)

5. CUDA自己对shared memory 的使用的示例[3.2.4 GEMM](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

6. 有些知识感觉问gpt获取的速度会更快，但是具体的细节还是在官网上查阅更好（[官网](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)的知识介绍非常全）\

   

## Lab Design & Test Result

---

```c
/// \brief Let A to be A + B.
__global__ void AdderCudaKernel(double *__restrict__ a,
                                  const double *__restrict__ b)
{
    // const int i = blockIdx.x * block_size + threadIdx.x;
    // const int j = blockIdx.y * block_size + threadIdx.y;
    // if (i < size && j < size)
    // {
    //   a(i, j) += b(i, j);
    // }
    __shared__ double shared_a[block_size][block_size];
    __shared__ double shared_b[block_size][block_size];

    int i = blockIdx.x * block_size + threadIdx.x;
    int j = blockIdx.y * block_size + threadIdx.y;

    if (i < size && j < size) {
        shared_a[threadIdx.x][threadIdx.y] = a[i * size + j];
        shared_b[threadIdx.x][threadIdx.y] = b[i * size + j];
        __syncthreads();

        shared_a[threadIdx.x][threadIdx.y] += shared_b[threadIdx.x][threadIdx.y];
        __syncthreads();

        a(i,j) = shared_a[threadIdx.x][threadIdx.y];
    }
}

/// \brief Do Matrix Multiplication on GPU.
__global__ void MultipleCudaKernel(const double *__restrict__ a, 
                                   const double *__restrict__ b, 
                                   double *__restrict__ result) 
{     
    // Get the index of the current thread
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int block_num = (size + block_size - 1) / block_size;

    // Define shared memory
    alignas(32) __shared__ double shared_a[block_size][block_size];
    alignas(32) __shared__ double shared_b[block_size][block_size];

    // Perform matrix multiplication operation
    double sum = 0.0f;
    for (int i = 0; i < block_num; i++) {
        // Load data from A and B into shared memory
        int idx_a = row * size + i * block_size + threadIdx.x;
        int idx_b = (i * block_size + threadIdx.y) * size + col;
        shared_a[threadIdx.y][threadIdx.x] = (row < size && (i * block_size + threadIdx.x) < size) ? a[idx_a] : 0.0f;
        shared_b[threadIdx.y][threadIdx.x] = ((i * block_size + threadIdx.y) < size && col < size) ? b[idx_b] : 0.0f;
         // Synchronize to make sure the matrices are loaded before starting the computation
        __syncthreads();

        #pragma unroll
        for (int j = 0;  j < block_size; j ++) {
            // sum = fma(shared_a[threadIdx.y][j], shared_b[j][threadIdx.x], sum);
            // sum = fma(shared_a[threadIdx.y][j + 1], shared_b[j + 1][threadIdx.x], sum);
            // sum = fma(shared_a[threadIdx.y][j + 2], shared_b[j + 2][threadIdx.x], sum);
            // sum = fma(shared_a[threadIdx.y][j + 3], shared_b[j + 3][threadIdx.x], sum);
            sum += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
        }
        // Synchronize to make sure the computation is done before loading the next sub-matrix
        __syncthreads();
    }

    if(row < size && col < size) {
        // Write the result back to result_kernel
        result(row, col) = sum;
    }
}
```

**Baseline分析：**对于baseline而言，其速度慢的原因一方面是只能串行线性进行计算；另一方面在于每次循环都要对主存进行数据的读写。故我们要针对以上的内容进行优化。

**优化策略：**共享内存，内存对齐，循环展开（编译器会自动实现）、fma等

### AdderCudaKernel

![](F:\Note of computer\docs\ZJU_CS\超算\homework\graph\Snipaste_2023-07-23_21-06-03.png)

第一个是常规的**AdderCudaKernel**的测试，第二个是使用共享内存的测试时间，由于每次测试具有随机以及不稳定性，针对这种情况我分析可能是`__syncthreads()；`导致停顿的时间。

### Shared Memory

共享内存是测试中加速的主要原因，我们其中使用内存对齐的方式能够有些许加速。

在测试中我们发现共享内存的大小是`0xc000`,因此我们将blocksize的大小设置为16比较合适。

### 循环展开

这是一个比较常见的优化手段，但是因为我们使用的是O3优化，以及`#pragma unroll`,提示编译器，故在我的测试中发现不用自己手写循环展开的优化。

其中`fma` 值得我们学习，这是CUDA本身自带的一种加速指令。

### 测试结果

![](F:\Note of computer\docs\ZJU_CS\超算\homework\graph\Snipaste_2023-07-23_21-08-27.png)

![](F:\Note of computer\docs\ZJU_CS\超算\homework\graph\Snipaste_2023-07-23_21-58-40.png)

故加速比为**0.94818**

## Discussion

---

本次lab教会了我如何使用CUDA进行GPU编程，在整个编程过程中，基本阅读了官方文档，并且了解了thread和warp编程思想，以及结合课上的知识，深入理解了内存的内部构造以及NVIDA显卡的自身结构特性，在此基础上进行了优化。
