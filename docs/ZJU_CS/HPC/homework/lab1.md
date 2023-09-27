
# Lab-1

---

[TOC]

## Lab Description

---

> ps：这个实验是浙江大学计算机科学与计算专业大一（2023）短学期——超算课程的实验，详情请点击[此处](https://zjusct.pages.zjusct.io/summer-course-2023/HPC101-Labs-2023/)

本次实验要求使用四台虚拟机搭建一个简易的集群，并对该集群进行性能测试，最后提交测试结果和实验报告。

集群搭建的任务包括创建虚拟机、安装 Linux 发行版、配置网络和 ssh 通信。

性能测试通过使用 OpenMPI 将 HPL 测试程序分配到四个虚拟机节点上执行。因此，需要下载并编译 OpenMPI、BLAS 和 HPL 的源代码，其中 OpenMPI、BLAS是 HPL 的依赖项。



## Introduction Knowledge(可以跳过不看)

---

(ps:若不想看可跳过，后面实验过程中看不懂了可以再来学习)

### [集群](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E9%9B%86%E7%BE%A4)

**计算机集群**（英语：computer cluster）是一组松散或紧密连接在一起工作的[计算机](https://zh.wikipedia.org/wiki/電子計算機)。由于这些计算机协同工作（并行计算），在许多方面它们可以被视为单个系统。与[网格计算机](https://zh.wikipedia.org/wiki/网格计算)不同，计算机集群将每个[节点](https://zh.wikipedia.org/w/index.php?title=节点_(计算机科学)&action=edit&redlink=1)设置为执行相同的任务，由软件控制和调度。



### Linux命令

这个感觉网上都会有很多教程，可以自己熟悉一下就能学会。 （Tips：man很多时候是个好东西）

```unix
cd ls clear mkdir rmdir ……
```



### HPL

HPL（The High-Performance Linpack Benchmark）是测试高性能计算集群系统浮点性能的基准程序。HPL通过对高性能计算集群采用高斯消元法求解一元N次稠密线性代数方程组的测试，评价高性能计算集群的浮点计算能力。



### UFW

**Uncomplicated Firewall**[[1\]](https://zh.wikipedia.org/wiki/Uncomplicated_Firewall#cite_note-1)，简称**UFW**，是[Ubuntu](https://zh.wikipedia.org/wiki/Ubuntu)系统上默认的[防火墙](https://zh.wikipedia.org/wiki/防火墙)组件。UFW是为轻量化配置[iptables](https://zh.wikipedia.org/wiki/Iptables)而开发的一款工具。UFW 提供一个非常友好的界面用于创建基于[IPV4](https://zh.wikipedia.org/wiki/IPV4)，[IPV6](https://zh.wikipedia.org/wiki/IPV6)的防火墙规则。UFW 在 Ubuntu 8.04 LTS 后的所有发行版中默认可用[[2\]](https://zh.wikipedia.org/wiki/Uncomplicated_Firewall#cite_note-2)。UFW 的[图形用户界面](https://zh.wikipedia.org/wiki/圖形使用者界面)叫**Gufw**



### BLAS

**BLAS**（英语：**Basic Linear Algebra Subprograms**，基础线性代数程序集）是一个[应用程序接口](https://zh.wikipedia.org/wiki/应用程序接口)（API）标准，用以规范发布基础线性代数操作的数值库（如矢量或矩阵乘法）。该程序集最初发布于1979年，并用于创建更大的数值程序包（如[LAPACK](https://zh.wikipedia.org/wiki/LAPACK)）。在高性能计算领域，BLAS被广泛使用。例如，[LINPACK](https://zh.wikipedia.org/wiki/LINPACK)的运算成绩则很大程度上取决于BLAS中子程序[DGEMM](https://zh.wikipedia.org/w/index.php?title=DGEMM&action=edit&redlink=1)的表现。



### MPI

**消息传递接口**（英语：Message Passing Interface，缩写MPI）是一个[并行计算](https://zh.wikipedia.org/wiki/平行計算)的[应用程序接口](https://zh.wikipedia.org/wiki/应用程序接口)（API），常在[超级电脑](https://zh.wikipedia.org/wiki/超級電腦)、[电脑集群](https://zh.wikipedia.org/wiki/電腦叢集)等非共享内存环境程序设计。

而本次实验的OpenMPI是MPI的一种，共享内存的并行编程的一个API。OpenMPI 将 HPL 测试程序分配到四个虚拟机节点上执行。MPI的核心主要是：端到端通信，数据类型，和集合通信（Collective Communication）。具体内容可以见[MPI参考手册](https://mpitutorial.com/tutorials/)



### SSH

**安全外壳协议**（Secure Shell Protocol，简称**SSH**）是一种加密的[网络传输协议](https://zh.wikipedia.org/wiki/网络传输协议)，可在不安全的网络中为网络服务提供安全的传输环境[[1\]](https://zh.wikipedia.org/wiki/Secure_Shell#cite_note-rfc4251-1)。SSH通过在网络中创建[安全隧道](https://zh.wikipedia.org/w/index.php?title=安全隧道&action=edit&redlink=1)来实现SSH客户端与服务器之间的连接[[2\]](https://zh.wikipedia.org/wiki/Secure_Shell#cite_note-rfc4252-2)。SSH最常见的用途是远程登录系统，人们通常利用SSH来传输[命令行界面](https://zh.wikipedia.org/wiki/命令行界面)和远程执行命令。（我们可以远程访问桌面或者服务器）

此外，我们常用的Github也是比较建议采用SSH通信，这样访问比较稳定。



### Makefile

我们常说的Makefile就是GNU [make](https://www.gnu.org/software/make/manual/make.html).它能够帮助我们快速完成编译。上过C大的同学应该已经有所了解了。我也不在此过多说明。但是有一个深入浅出的[手册](https://seisman.github.io/how-to-write-makefile/overview.html)放在这里。



### 内网 IP

本次实验中，我们采取的是内网IP互通。由于网关的存在，不同局域网内的内网IP是可以重复的。

[内网](https://baike.baidu.com/item/内网/427841?fromModule=lemma_inlink)[IP地址](https://baike.baidu.com/item/IP地址/150859?fromModule=lemma_inlink)，也就是局域网[网络地址](https://baike.baidu.com/item/网络地址/9765459?fromModule=lemma_inlink)，内网的计算机以NAT（[网络地址转换](https://baike.baidu.com/item/网络地址转换/2985755?fromModule=lemma_inlink)）协议，通过一个公共的网关访问Internet。内网的计算机可向Internet上的其他计算机发送连接请求，但Internet上其他的计算机无法向内网的计算机发送连接请求。

NAT（Network Address Translator）是[网络地址转换](https://baike.baidu.com/item/网络地址转换/2985755?fromModule=lemma_inlink)，它实现内网的IP地址与公网的地址之间的相互转换，将大量的内网IP[地址转换](https://baike.baidu.com/item/地址转换/9739652?fromModule=lemma_inlink)为一个或少量的[公网IP](https://baike.baidu.com/item/公网IP?fromModule=lemma_inlink)地址，减少对公网IP地址的占用。NAT的最典型应用是：在一个局域网内，只需要一台计算机连接上Internet，就可以利用NAT共享Internet连接，使局域网内其他计算机也可以上网。

而本次实验的集群及采取这样的模式，来建立集群间的网络连接。





## Lab Design & Test Result

---



### 虚拟机

1. 由于我之前寒假想上手下Linux系统，故已经在自己电脑上安装过`VMware`，此前使用的是`redhat`。现在只需要下载Linux的镜像文件即可。由于自己已经使用了半年多的`WSL2`，故选择下载的是Ubuntu系统，可以直接访问[ZJU-mirror](https://mirrors.zju.edu.cn/docs/ubuntu/)即可。（我建议使用Desktop版本，你会快乐很多，后面也会方便很多）![image](graph\Snipaste_2023-05-27_19-11-05.png)



2. 建议使用虚拟机的话可以单独出来磁盘便于管理以及安全。

> ![image](graph\Snipaste_2023-05-27_16-56-32.png)



1. VMware虚拟机的建立，网上也有很多教程，故不在此说明。(ps:基本上选择你刚下载好的iso镜像，选择好配置后一路确定即可)

* 虚拟机创建成功，如图：![image](graph\Snipaste_2023-05-27_18-26-36.png)

  

### HPL环境搭建

HPL是`Linpack`测试的一种，需要依赖`OpenMPI`&`OpenBLAS`来实现。



#### 虚拟机环境配制

> 目标：安装相关编译器以及make工具和网络工具

1. `sudo apt install gcc g++ python3 gfortran -y`

检查的话有`… --version`即可，或者`apt list --installed `(但这个太蠢了)

2. `sudo apt install make net-tools -y`

3. `sudo apt install openssh-server -y`

用`sudo systemctl status ssh`和`ifconfig`验证网络工具和SSH Server

4. UFW打开SSH端口

   `sudo ufw allow ssh`



#### Openmpi安装

1. 下载并解压OpenMPI

   网址为：[Open MPI: Version 4.1 (open-mpi.org)](https://www.open-mpi.org/software/ompi/v4.1/)

   用火狐浏览器打开点击下载后，从终端打开当前文件夹，使用： `tar xvf openmpi-4.1.1.tar.gz`(学会使用`tab`补全)

   （ps：用桌面版的好处来啦！）

   非桌面版的用户就正常用`wget`和`curl`下载就行。

   ![image](graph\Snipaste_2023-05-27_19-19-35.png)

   

2. 安装OpenMPI

   ```
   cd openmpi-4.1.4
   ./configure
   sudo make all install
   ```

​	 关于`sudo make all`&`sudo make install`的原理，有一篇博主写的蛮不错的。[link](https://zhuanlan.zhihu.com/p/77813702)



3. 添加路径

   `sudo vim /etc/profile`(这里需要一点点vim的知识)

   在profile文件末尾添加

   ![image](graph\Snipaste_2023-05-27_22-43-18.png)

   执行以下语句使配制生效

   `source /etc/profile`

   (ps:可能用户名会变成白色，但是不影响环境更新)

   

   4.测试环境

   ```
   cd Downloads/openmpi-4.1.4/examples
   make && mpirun -np 2 hello_c.c
   ```

   * 测试结果

     ![image](graph\Snipaste_2023-05-27_21-44-01.png)



#### OpenBLAS安装

* 用`sudo apt-get install libopenblas-dev`命令安装

* 安装路径在`/usr/lib/x86_64-linux-gnu/openblas-pthread`

  想要测试自己是否安装成功可以在本地写个程序测试，代码见[link](https://blog.csdn.net/qq_42694450/article/details/111058653)

  ![image](graph\Snipaste_2023-05-28_01-43-52.png)



### HPL的安装以及环境配制

* [下载](https://netlib.org/benchmark/hpl/software.html)并解压`hpl-2.3.tar.gz`

  `tar xvf hpl-2.3.tar.gz`

* 复制并粘贴Make文件（当然也可以图形化操作喽）

  ```
  cp Downloads/hpl-2.3/setup/Make.Linux_PII_CBLAS Downloads/hpl-2.3/Make.test
  mv hpl-2.3 hpl(重命名)
  ```

* 修改makefile

  这里不得不吐槽一下，应该是我目前来说费时最长的地方，基本上自己把整个Makefile全部重新读和写了一遍，模板基本上不能怎么用。然后自己看着make编译出来的报错一点点改。基本上就对应自己找了一遍。如果是完全按照我之前步骤来的，那么可以直接照抄我的Makefile了

```
  SHELL        = /bin/sh
  CD           = cd
  CP           = cp
  LN_S         = ln -s
  MKDIR        = mkdir
  RM           = /bin/rm -f
  TOUCH        = touch
  ARCH = test
  
  TOPdir       = $(HOME)/Downloads/hpl
  INCdir       = $(TOPdir)/include
  BINdir       = $(TOPdir)/bin/$(ARCH)
  LIBdir       = $(TOPdir)/lib/$(ARCH)
  HPLlib       = $(LIBdir)/libhpl.a
  
  MPdir = /usr/local/lib/openmpi
  MPinc =
  MPlib = /usr/local/lib/libmpi.so
  
  LAdir        = /usr/lib/x86_64-linux-gnu/openblas-pthread
  LAinc        =
  LAlib        = $(LAdir)/libblas.a $(LAdir)/libblas.so
  
  F2CDEFS      =
  
  HPL_INCLUDES = -I$(INCdir) -I$(INCdir)/$(ARCH) $(LAinc) $(MPinc)
  HPL_LIBS     = $(HPLlib) $(LAlib) $(MPlib)
  
  HPL_OPTS     =
  
  HPL_DEFS     = $(F2CDEFS) $(HPL_OPTS) $(HPL_INCLUDES)
  
  CC           = /usr/local/bin/mpicc
  CCNOOPT      = $(HPL_DEFS)
  CCFLAGS      = $(HPL_DEFS) -fomit-frame-pointer -O3 -funroll-loops -w -Wall -pthread
  
  LINKER       = /usr/local/bin/mpif77
  LINKFLAGS    = $(CCFLAGS)
  
  ARCHIVER     = ar
  ARFLAGS      = r
  RANLIB       = echo
  
  arch = test
```

* 编译并检查是否成功

  ```
  make arch=test(一定不要因为变成的习惯在等号两边多打空格)
  cd hpl/bin/test
  dir
  HPL.dat xhpl(证明安装成功!如图)
  ```

  ![image](graph\Snipaste_2023-05-28_01-45-30.png)



### 克隆节点

在VMware中克隆已经配置好的节点，成为集群的其他三个节点.

可以在克隆前在主机上先安装好`SSH`.后面能节约一点点时间。

（Tips：一般先会在左上角的编辑进入 -->虚拟网络编辑器，将子网IP更改为不是192.168.64（可以是65…）。然后再右键克隆，记得提前格出来近100G磁盘，并在里面建四个小文件夹有助于管理和虚拟机的稳定）

![image](graph\Snipaste_2023-05-28_02-13-02.png)



### 测试集群

#### ping通

* 获得每台虚拟机的IP地址

  `ifconfig`

  ![image](graph\Snipaste_2023-05-28_02-24-17.png)

* 获得的虚拟机地址如下：

  ```
  192.168.253.129
  192.168.253.130
  192.168.253.131
  192.168.253.132
  ```

*  确认能相互ping通

  `ping 192.168.253.130 ……`  

  ![image](graph\Snipaste_2023-05-28_02-30-07.png)



#### 配制SSH

1. 生成公钥私钥对

​	`ssh-kegen -t rsa`

​	这里跟git前期步骤一样，详情可以见[git笔记](https://yaoyaolingbro.github.io/notebook/home/Missing%20semester/git/)

2. 将主机上的公钥拷贝到另外三台虚拟机的目录下

   ![image](graph\Snipaste_2023-05-28_02-46-29.png)

3. 远程免密访问测试

   `ssh username@192.168.***.***`

   复制完成后，注意检查 `authorized_keys` (600) 和 `.ssh/` (700) 目录的权限，否则无法顺利 `ssh`。

   > ssh passphrase —— 如果自己的密钥有 passphrase，那么请使用 `ssh-agent` 确保能暂时不用输入 passphrase，以免之后影响 `mpirun` 正确运行。

   注：我们还是用`ifconfig`判断自己当前所在位置；用`logout`退出SSH访问



#### mpirun尝试

1. 创建主机中的hostfile文件（不用在意格式）

   `vim myhostfile`

   （slots是核心数的意思，这跟你在创建虚拟的时候设定有关，记得一定加s）

   ```
   192.168.253.129 slots=2
   192.168.253.130 slots=2
   192.168.253.131 slots=2
   192.168.253.132 slots=2
   ```

2. 查看每个节点上线时间

   `mpirun --hostfile myhostfile uptime`

   ![image](graph\Snipaste_2023-05-30_23-39-21.png)

3. 运行HPL

   `mpirun --hostfile myhostfile ./xhpl`

   运行结果：

   ![image](graph\Snipaste_2023-05-28_03-27-36.png)



## Discussion

---

首先值得一提的是，在本次实验中我收获了很多，一方面了解了超级计算机在并行运算方面的一些简单原理；此外，我学习到了许多网络与通信相关的知识，并且很大程度上在做lab的过程当中，我发现我自己之前所学习到的知识都能整合起来并运用。

此外，在本次实验中，在OpenMPI安装并测试程序（由于忽略了`&&`连接命令符，导致耽搁了很多时间）以及在HPL的makefile编写过程（最终自己相当于找目录重新手写了一遍makefile），略有磕绊，其他情况都算顺利。本次实验还有一个心得体会就是我们一定要学会看终端反馈回来的报错，不能盲目在网上查找教程（国内教程拉的一匹），这样能够有效并且快速的帮助我们找出自己的问题所在。

最后测试的实验结果如上图，虽然结果有反馈，但仍有困惑的点就是不知道如何去对比体现并行计算的优越性。这个lab给我的五月画上了一个还算圆满的句号。希望能够加入`ZJUSCT`，在段学期内收获.

