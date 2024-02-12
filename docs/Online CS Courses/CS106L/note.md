# Assignment Note
这里是对每次作业的一些要点和好的编程习惯的记录

## Assignment 1
1. 重载运算符
```cpp
bool operator==(const Course &other) const {
    return title == other.title && number_of_units == other.number_of_units &&
           quarter == other.quarter;
  }
```

2. auto 关键字使用
```cpp
void delete_elem_from_vector(std::vector<Course> &v, Course &elem) {
  auto it = std::find(v.begin(), v.end(), elem);
  v.erase(it);
}
```

3. C中嵌套python进程的方法
```cpp
FILE *pipe = popen("python3 utils/autograder.py", "r");
  if (!pipe)
    return -1;

  char buffer[128];
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != NULL)
      std::cout << buffer;
  }
  pclose(pipe);
```

4. vector 的三种遍历方式
```cpp
// 下标
std::vector<int> vec = {1, 2, 3, 4, 5};
for (int i = 0; i < vec.size(); ++i) {
    std::cout << vec[i] << " "; // 输出每个元素的值
}

// 迭代器
for (std::vector<int>::iterator it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " "; // 输出每个元素的值
}

// 范围for循环
for (int i : vec) {
    std::cout << i << " "; // 输出每个元素的值
}
```

## Assignment 2 Marriage pact
