# 并行计算设计导论

## MPI

MPI is designed to work in a heterogeneous（异构） environment.

fairly heavily（相当）

### 第一个类似于Hello world 的程序

```c
#include <stdio.h>
#include <string.h>
#include <mpi.h>

const int MAX_STRING = 100;

int main (void){
    char greeting[MAX_STRING];
    int comm_sz;
    int my_rank;

    MPI_Init( NULL , NULL);
    MPI_Comm_size( MPI_COMM_WORLD , &comm_sz);
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank);

    if (my_rank != 0) {
        sprintf(greeting, "Greetings from process %d of %d!\n", my_rank, comm_sz);
        MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("Greetings from process %d of %d!\n", my_rank, comm_sz);
        for(int i = 1; i < comm_sz; i++)
        {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s", greeting);
        }
    }

    MPI_Finalize();
    return 0;
}

```

用`mpicc -g -Wall -o hello hello.c`来编译

用`mpiexec -n 1 ./hello`来运行

### 一点点MPI程序的代码要点

1. 每个程序是由MPI_Init和MPI_Finalize进行必要的初始化和结束。与MPI有关的函数都要在这里面进行（其中，MPI_Init传入的是*argc和*argv的指针。
2. 通信子（communicator）、MPI_Comm_size 和 MPI_Comm_rank （一般来说，comm_sz表示进程的数量，my_rank表示进程号）

### SPMD编程

> Single Program and Multiple Data：我们称为单程序多数据。理论上是每个进程编译不同的程序。但是我们常用的手段是编写一个程序，根据不同的进程号来分配进程。

### 通信

在上面的示例程序中，我们将信息通过`printf & sprintf`来传递。通过MPI_Send来发送MPI_Recv来接受。

![](graph\Snipaste_2023-07-04_21-09-26.png)

![](graph\Snipaste_2023-07-04_21-48-39.png)

值得注意的是size要加上'\0'。**Data type could make the program portable.**

（还有两个函数是MPI_Isend和MPI_IRecv）

![](graph\Snipaste_2023-07-04_21-12-32.png)

注意：

1. 消息的大小不要超过我们所分配的缓冲区。

2. ```c
   recv_comm = send_comm
   recv_tag = send_tag
   dest = r
   src = q
   且前三个信息兼容
   ```

​	这意味着q号进程的信息可以被r号进程收到

3. MPI使用的是“推”通信机制
4. MPI消息不可超越。

### 潜在的一些陷阱

* 我们需要每条接受语句有相应的发送语句匹配，防止进程悬挂。

![](graph\Snipaste_2023-07-04_21-46-21.png)

![](graph\Snipaste_2023-07-04_21-49-20.png)

* also have a function `MPI_Probe`is like a mpi receive.

![](graph\Snipaste_2023-07-04_22-08-22.png)

* unblocking IO 会使编程变复杂。

![](graph\Snipaste_2023-07-04_22-24-28.png)