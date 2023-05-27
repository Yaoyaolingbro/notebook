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
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">《HPC101-Lab1-2023》</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto">实验报告 </p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 简单集群的搭建</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 2023.7</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">ZJU-SCT </td>     </tr>
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
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 个人完成</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">2023.5.28</td>     </tr>
    </tbody>              
    </table>
</div>

<!-- 注释语句：导出PDF时会在这里分页 -->

# Content

---

[TOC]

## Lab Introduction

---

> ps：这个实验是浙江大学计算机科学与计算专业大一（2023）短学期——超算课程的实验，详情请点击[此处](https://zjusct.pages.zjusct.io/summer-course-2023/HPC101-Labs-2023/)

本次实验要求使用四台虚拟机搭建一个简易的集群，并对该集群进行性能测试，最后提交测试结果和实验报告。

集群搭建的任务包括创建虚拟机、安装 Linux 发行版、配置网络和 ssh 通信。

性能测试通过使用 OpenMPI 将 HPL 测试程序分配到四个虚拟机节点上执行。因此，需要下载并编译 OpenMPI、BLAS 和 HPL 的源代码，其中 OpenMPI、BLAS是 HPL 的依赖项。

## Needed Knowledge

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





## The Whole Procedure of Lab

### 虚拟机

1. 由于我之前寒假想上手下Linux系统，故已经在自己电脑上安装过`VMware`，此前使用的是`redhat`。现在只需要下载Linux的镜像文件即可。由于自己已经使用了半年多的`WSL2`，故选择下载的是Ubuntu系统，可以直接访问[ZJU-mirror](https://mirrors.zju.edu.cn/docs/ubuntu/)即可。（我建议使用Desktop版本，你会快乐很多，后面也会方便很多）![image](F:\Note of computer\docs\home\超算\graph\Snipaste_2023-05-27_19-11-05.png)



2. 建议使用虚拟机的话可以单独出来磁盘便于管理以及安全。

> ![image](F:\Note of computer\docs\home\超算\graph\Snipaste_2023-05-27_16-56-32.png)



3. VMware虚拟机的建立网上也有很多教程，故不在此说明。(ps:基本上选择你刚下载好的iso镜像，选择好配置后一路确定即可)

* 虚拟机创建成功，如图：![image](F:\Note of computer\docs\home\超算\graph\Snipaste_2023-05-27_18-26-36.png)

  

### HPL环境搭建

HPL是`Linpack`测试的一种，需要依赖`OpenMPI`&`OpenBLAS`来实现。



#### 虚拟机环境配制

> 目标：安装相关编译器以及make工具和网络工具

* `sudo apt install gcc g++ python3 gfortran -y`

  检查的话有`… --version`即可，或者`apt list --installed `(但这个太蠢了)

* `sudo apt install make net-tools -y`

* `sudo apt install openssh-server -y`

  用`sudo systemctl status ssh`和`ifconfig`验证网络工具和SSH Server

* UFW打开SSH端口

  

#### Openmpi安装

1. 下载并解压OpenMPI

   网址为：[Open MPI: Version 4.1 (open-mpi.org)](https://www.open-mpi.org/software/ompi/v4.1/)

   用火狐浏览器打开点击下载后，从终端打开当前文件夹，使用： `tar xvf openmpi-4.1.1.tar.gz`(学会使用`tab`补全)

   （ps：用桌面版的好处来啦！）

   非桌面版的用户就正常用`wget`和`curl`下载就行。

   ![image](F:\Note of computer\docs\home\超算\graph\Snipaste_2023-05-27_19-19-35.png)

2. 

















