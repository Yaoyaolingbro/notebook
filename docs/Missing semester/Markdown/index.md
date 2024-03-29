## Markdown相关记录

## 引言

关于markdown的介绍呢，感觉确实没什么可说的。就是一种比较通用的语言（一些网页上也通用），然后能够便于我们进行一个快速的排版。至于美观与否，感觉还是因人而异的。好处就是轻量级的语法，你甚至可以在1h内速通，但与此同时便是他的确没有`latex`那么好看，（但latex的学习成本对于一个freshman来说是有点点高的）。所以任何事情都是利弊兼具的，但我还是推荐大家不妨简单学习一下。

至于学习路线的话按照目前的个人经验来说，应该是：`基础语法 --> 进阶（html和css语法积累和mermaid画简单流程图）`，如果有数学要求，可以选学latex的数学公式。这样大抵是够用的。我也会根据这个结构在这篇文档里记录相关信息。

这里是[官网](https://markdown.com.cn/intro.html#markdown-%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F)
学习的时候直接照个官网每个自己敲一遍就学会了。



## 基础语法

不管甚么语言，首先说的是：请不要保养你的space和enter键！！！

### 标题

标题总而言之就是根据你的`#`个数决定层级的。你可以读这篇文档的源代码就是看到标题的写法。

除此之外还有一种写标题的方式是：
1. 大标题在你文字的下一行加"------"(不知道多少个)
2. 小标题在文字下一行加"==="（显然比上面多）

### 分割线

三个'-'或者'='，上下不要有多余的文字即可。

### 不同字型

```
*斜体*  
_斜体文本_  
**粗体文本**  
__粗体文本__  
<strong>粗体</strong>（不知道为什么，有时候我的粗体不能用）
***粗斜体文本***  
___粗斜体文本___  
~~对整个文字画横线~~ 
<u>带下划线文本</u> 
```

### 列表

无序列表使用星号(*)、加号(+)或是减号(-)作为列表标记，这些标记后面要添加一个空格，然后再填写内容：

```
* 第一项
* 第二项
* 第三项

+ 第一项
+ 第二项
+ 第三项


- 第一项
- 第二项
- 第三项
```



有序列表使用数字并加上 . 号来表示，如：

```
1. 第一项
2. 第二项
3. 第三项
```



列表嵌套只需在子列表中的选项前面添加两个或四个空格即可：

1. 第一项：
  - 第一项嵌套的第一个元素
    - 第一项嵌套的第二个元素
2. 第二项：
  - 第二项嵌套的第一个元素
    - 第二项嵌套的第二个元素

### 脚注

创建脚注格式类似这样 

```

[^RUNNOB]。  
[^RUNOOB]: 菜鸟教程 -- 学的不仅是技术，更是梦想！！！（这个网站有利有弊，新手入门相对还是比较友好的）


```

### 代码

```c
    #include <stdio.h>
    int main (void)
    {
        return 0;
    }
```

### 区块

Markdown 区块引用是在段落开头使用 > 符号 ，然后后面紧跟一个空格符号：

> 区块引用  
> 菜鸟教程  
> 学的不仅是技术更是梦想

或者可以这样操作
* 第一项
    > 菜鸟教程  
    > 学的不仅是技术更是梦想
* 第二项
  

### 链接

这是一个链接 [菜鸟教程](https://www.runoob.com)  
或者<http://www.runoob.com>的形式表示链接。

这个链接用 1 作为网址变量 [Google][1]  
这个链接用 runoob 作为网址变量 [Runoob][runoob]  
然后在文档的结尾为变量赋值（网址）  

[1]: http://www.google.com/
[runoob]: http://www.runoob.com/

文档内跳转的示例：[示例]()

### 图片

图片是类似网址的：  
```markdown
![alt 属性文本](图片地址)  
![alt 属性文本](图片地址 "可选标题")  
```


例如：  
![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png)  
![RUNOOB 图标](http://static.runoob.com/images/runoob-logo.png "RUNOOB")

### 表格

| 左对齐 | 右对齐 | 居中对齐 |  
| :-----| ----: | :----: |  
| 单元格 | 单元格 | 单元格 |  
| 单元格 | 单元格 | 单元格 |  



### 数学公式

[一个不错的博主的汇总](https://zinglix.xyz/2017/08/23/latex-maths-cheatsheet/)
$$
\begin{Bmatrix}
   a & b \\
   c & d
\end{Bmatrix}
$$

$$
\begin{CD}
   A @>a>> B \\
@VbVV @AAcA \\
   C @= D
\end{CD}
$$

$f(x) = sin(x)+12$

$$
f(x) =
\begin{cases}
x,  & x\ge0 \\
-x, & x<0
\end{cases}
$$

$$ 
\sum_{n=1}^{100}x!
$$



## 进阶内容

### Mermaid

[官方文档](https://mermaid.js.org/intro/)

mermaid 图形：



```mermaid
   sequenceDiagram
       Alice->>Bob: Hello Bob, how are you?
       Bob-->>Alice: I'm good, thanks!
```

```mermaid
   gantt
       title 项目计划
       section 项目A
       任务1 :a1, 2023-10-21, 3d
       任务2 :after a1, 2d
       section 项目B
       任务3 :2023-10-25, 2d
       任务4 : 2d

```

```mermaid
   classDiagram
       class Animal {
           +name: string
           +eat(): void
       }
       class Dog {
           +bark(): void
       }
       Animal <|-- Dog

```



### HTML

除了之前提到的常用HTML语法，Markdown还支持其他一些扩展的HTML语法，这些语法可以帮助你更灵活地定制和美化你的文档。以下是一些常见的扩展HTML语法在Markdown中的使用示例：

1. `<div>`和`<span>`标签：可以用于自定义样式和布局。
   ```markdown
   <div style="background-color: #f1f1f1; padding: 10px;">
       这是一个带背景颜色的区块
   </div>
   
   <span style="color: red;">这是红色文本</span>
   ```





2. `<iframe>`标签：可以嵌入其他网页或多媒体内容。
   ```markdown
   <iframe src="https://www.example.com" width="500" height="300"></iframe>
   ```

3. `<audio>`和`<video>`标签：可以插入音频和视频文件。
   ```markdown
   <audio controls>
       <source src="audio.mp3" type="audio/mpeg">
   </audio>
   
   <video controls width="500" height="300">
       <source src="video.mp4" type="video/mp4">
   </video>
   ```

4. `<mark>`标签：用于突出显示文本。
   
   ```markdown
   <mark>这段文字将被突出显示</mark>
   ```
   

5. `<blockquote>`标签：用于引用文本块。



```markdown
H<sub>2</sub>O 是水的化学式。
E = mc<sup>2</sup> 是相对论中的著名公式。
```

6. `<del>`和`<ins>`标签：用于表示删除和插入的文本。

```markdown
<del>这段文字被删除了</del>
<ins>这段文字被插入了</ins>
```

7. `<pre>`标签：用于显示预格式化的文本，保留空格和换行符。

```markdown
<pre>
这是预格式化的文本。
    保留空格和换行符。
</pre>
```

8. `<details>`和`<summary>`标签：用于创建可折叠的内容。

```markdown
<details>
    <summary>点击展开</summary>
    这是可折叠的内容。
</details>
```

这些是更多的HTML语法示例，你可以根据需要使用它们来扩展你的Markdown文档。请记住，在使用这些扩展语法时，要确保它们在Markdown解析器中得到正确支持！！！
