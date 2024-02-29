# python使用技巧

每个语言都有着其自己的特性，当然python也不例外。我将分成下面几个大类进行记录。

<!-- prettier-ignore-start -->
!!! note "catalogue"
    - [ ] 基本技巧
    - [ ] OS使用
    - [ ] matplotlib相关
    - [ ] 杂项
<!-- prettier-ignore-end -->


## Basic


## 正则表达式
1. `re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)`
   上述代码是用来匹配html中的链接的，其中`r`表示原始字符串，`[^>]*?`表示匹配任意字符，`*?`表示非贪婪匹配，`[^\"]*`表示匹配任意字符，`()`表示分组，`[]`表示匹配其中的任意一个字符。




## OS使用

1. ```python
   for filename in os.listdir(directory):
           if not filename.endswith(".html"):
               continue
           with open(os.path.join(directory, filename)) as f:
               contents = f.read()
               links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
               pages[filename] = set(links) - {filename}
   ```

2. 


## 杂项

1. python 中的三元运算符：`return X if X_count == O_count else O`
2. 返回最大值的索引：`max_score_index = max(enumerate(scores), key=lambda x: x[1])[0]`
3. `with open(os.path.join(directory, filename)) as f:`使用with处理[异常情况](https://www.runoob.com/python3/python-with.html)