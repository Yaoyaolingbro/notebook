# 笔记

一个简明的从c学习python语法的[视频](https://cs50.harvard.edu/x/2023/shorts/python/)

## Python基础语法



### 判断语句

![](graph\Snipaste_2023-08-07_11-06-24.png)

and 或者 or，后面用冒号，tab分层，`elif`

```python
if course_num == 50:
	# code block1
elif not course_num == 51:
    # code block2
```

> we dont have to use the exclamation point symbol！, the vertical bar|, ampersand&



### bool变量（二元判断符）

```python
letter = True if input().isalpha() else False
```



### Loop循环

```python
counter = 0
while counter < 100:
	print(counter)
	counter += 1
	
for x in range(100):
    print(x)
for x in range(0, 100, 2)
	print(x)
# not include 100
```



#### Loop(redux)

```python
pizza = {
    "cheese": 9,
    "pepperoni": 10,
    "vegetable": 11,
    "chicken": 12
}

for pie in pizza:
    print(pie)
    
for pie,price in pizza.items():
    print(price)
    print("A whole {} pizza costs ${}".format(pie,price))
    print("A whole"+ pie + "pizza costs $" + str(price))
    print("A whole %s pizza costs $%2d" % (pie,price)) # deprecated in python3
```



### List列表

```python
nums = [x for x in range (10)]
nums.append(11)
nums.insert(10,11)
nums[len(nums):] = [11]
```

### Tuple元组

![](graph\Snipaste_2023-08-07_19-59-14.png)

```python
for prez,year in presidents:
    print("In {1}, {0} took office".format(prez,year))
```



### Dictionary

![](graph\Snipaste_2023-08-07_20-07-58.png)



### Function

![](graph\Snipaste_2023-08-07_20-22-32.png)

```python
def square(x):
    return x * x || return x**2
```



### Object

![](graph\Snipaste_2023-08-07_20-28-44.png)

It's similar to c++.

```python
class Student():
    
    def __init__(self, name, id):
        self.name = name
        self.id = id
        
    def change_ID(self, id):
        self.id = id
        
    def print(self):
        print("{} - {}".format(self.name, self.id))
        
jane = Student("Jane", 10)
jane.print()
jane.change_ID(11)
jane.print()
```









## Tips

1. python 中的三元运算符：`return X if X_count == O_count else O`
2. 返回最大值的索引：`max_score_index = max(enumerate(scores), key=lambda x: x[1])[0]`