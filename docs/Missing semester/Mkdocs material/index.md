# Mkdocs material使用教程
注：mkdocs material是markdown的一个插件，可以基于github-pages布置，本教程使用的方法是最简单的基于git本地部署配制



## 环境配制(windows 版)
1. 你需要安装[python](https://www.python.org/downloads/windows/)，连接点进去下载即可，安装3.10即以下（最新版可能会有bug）
1. 其次你需要安装pip，详情请看[教程](https://zhuanlan.zhihu.com/p/38603105)
1. 此外你需要安装[git](https://git-scm.com/download/win)(选择64-bits的standalone版本，直接一路next即可)

![](graph\Snipaste_2023-08-10_23-20-31.png)

4. 你需要一个[Github](https://github.com/)账号
5. 最好下载一个Vscode[教程](https://blog.csdn.net/qq_30640671/article/details/109704940)
6. 安装mkdocs，在电脑终端使用`   pip install mkdocs`&`pip install mkdocs-material`
7. markdown建议使用Typora编辑器，可以从b站上搜索安装教程（选做）



##  使用

1. 首先你需要建立一个Github上的远程仓库，可以具体看我另外一个快速上手远程仓库[教程](https://yaoyaolingbro.github.io/notebook/Missing%20semester/Git/fast_git/#githubrepository)
2. 此外，你需要学习markdown的语法，详情请看我的[记录](https://yaoyaolingbro.github.io/notebook/Missing%20semester/Markdown/)（如果想要插入图片,请在文件夹下建立一个graph文件夹，并在markdown文件中使用相对路径）![](graph\Snipaste_2023-08-10_23-29-01.png)



3. 请在命令行上使用`mkdocs new`, 会自动生成一个docs文件夹，site文件夹以及一个mkdocs.yml![](graph\Snipaste_2023-08-10_23-31-59.png)



4. 重要的配制东西都在mkdocs.yml中，大家去我仓库看吧，最好是能理解你写的每行代码的含义。



4. 如果想要像我一样偷懒的话，可以手写个update.sh即可，每次在git bash中操作就好。

```sh
 
#给出一个默认的项目路径
path="F:\Note of computer"

#先进入项目当中
 
cd $path
 
echo "####### 进入自己的项目 #######"
 
ls
 
echo "开始执行命令"
 
git add .
 
git status
 
#写个sleep 1s 是为了解决并发导致卡壳
 
sleep 1s
 
echo "####### 添加文件 #######"

ls_date=`date +%Y%m%d`

git commit -m "${ls_date}"
 
echo "####### commit #######"
 
sleep 1s
 
echo "####### 开始推送 #######"

git push

mkdocs gh-deploy
 
echo "####### 推送并页面部署成功 #######"

```



**恭喜你，如果能按步骤完成你基本上就能使用了**

## 使用技巧
### 提示块插件
```python
markdown_extensions:
    - admonition  # 提示块
    - pymdownx.details  # 提示块可折叠
```

你可以写vscode的snippet帮助你编写，同时他支持的提示块如下：
- note, seealso
- abstract, summary, tldr
- info, todo
- tip, hint, important
- success, check, done
- question, help, faq
- warning, caution, attention
- failure, fail, missing
- danger, error
- bug
- example
- quote, cite
ps：这中间所需要的小技巧有点多，我也在不断摸索，如果想要配置的同学可以按照我这个教程来写就好。