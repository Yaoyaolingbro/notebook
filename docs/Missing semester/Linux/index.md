# 一些简单的Linux使用

## 环境

1. 你可以选择虚拟机来使用(建议使用VMware Station；可以使用Ubuntu； 记得最好更换下下载源，可以是[zuj_mirrors](https://mirrors.zju.edu.cn/)or[清华源](https://mirrors.tuna.tsinghua.edu.cn/))
2. 或者使用[WSL](https://learn.microsoft.com/en-us/windows/wsl/install)来操作

## 学习建议
关于Linux的学习，首先关于书籍，我推荐一本[Linux程序设计](https://item.jd.com/10077374760063.html)（暂未找到电子书）。这本比较概括性的讲到了你所需要学习的大多数linux的知识。有示例，比较入门，大概两三天就可以有个初步的学习，但是十分浅尝辄止。

进阶一点的书籍的话推荐[linux就该这样学](https://www.linuxprobe.com/docs/LinuxProbe.pdf)和[鸟哥的Linux私房菜](https://tiramisutes.github.io/images/PDF/vbird-linux-basic-4e.pdf)

## 一些些基础知识

* `man + 带查询内容`一个非常好用的命令，可以帮助你查询手册。（不光是一些命令，还有一些c语言的东西）
* `info`在线文档系统。（基础操作： 用shift+h唤醒帮助。space 和 ctrl+h可以上下翻页等等）
* 文件系统是在`/`目录下的，进入系统会默认进入`/home/user`。
* `apt 或 brew`是Ubuntu和MacOS下的包管理器，用来下载和安装
* 有一些用户上的操作我们之后补充。
* `alias`可以自定义文件操作的别称。
* 下载软件（tar压缩包版）（snap或者应用商店）(`cur`l 或者 `wget`)
* su用户会是'#';普通用户是'$'

![](graph\Snipaste_2023-07-04_09-02-43.png)

## 文件操作

* 涉及到目录的操作，我们会用`ls`来显示当前文件夹下的所有内容。（-a显示所有；-l 为长格式显示，可叠加使用，如：`ls -hla`）
* 用`mkdir`创建文件夹（用 -m 设置参数，例如：`mkdir -m 777 tsk`,777表示所有的用户斗鱼rwx的权限）
* `chmod`可以用来更改权限。（r代表read，w代表write，x代表可执行）`chgrp`用来更改文件或目录的所属组。
* `rmdir`是删除文件夹。或者`rm`（用-r可以删除非空文件夹）
* `cd`切换目录（`..`是上一级目录，或者直接从`/`目录开始的绝对路径去操作）
* `gedit`或者`vim`是两种文本编辑器。第一种可以可视化，第二个会有许多快捷键操作。vim的操作会在另一个文件记录。
* `mv` 代表为文件摸着目录改名。或者将文件移动位置。（类似ctrl+x）（`*`可以代表着文件夹下所有东西，或者`*.c`意味着当前文件夹下所有.c文件
* `cp`是将文件或目录复制到其他目录下面  
* `grep anon *.conf`代表在当前文件夹下，所有.conf文件中搜索anon字符串的东西。
* `find / -name hello.c`在所有目录下寻找名字是hello.c的文件。
* `head  tail`可以用来查看文件的头和尾部的内容
* `wc`用来统计文件行数单词数等（如-m字符数，-l文件行数）
* `gzip`压缩（-d是解压缩）（注意！：gzip压缩文件后源文件会自动删除）
* `tar`也是一个非常常用的命令，用来打包文件，几乎可以工作于任何环境。

（简单的命令`tar -cvf home.tar ./home`把根目录下的home文件夹打包成home.tar；`-xvf`就是解压）

* `mount 和umount`一般来说是配合u盘使用。你也可以在`wsl`下看到我们windows本机的东西都挂载到/mnt文件夹下）

  
## 结语
Linux可以说是无论你身处何地，很多人都会默认你会但是学校又不会怎么教你去使用它。
相信你有以上这些知识你就可以进行一些基础的使用了，快去开启你的Linux探索之旅吧！  
我始终坚信实践才是最快的学习方法，在我学习的过程当中，很多命令都是边查并且不断使用你就自然而然就学会了。
一些更高级的用法请看本章其他内容。
  

  

  