# STL
## String
1. npos是一个常量，它的值是-1，用于表示一个不存在的位置。
```cpp
std::string s = "Hello, world!";
std::cout << s.find("world") << std::endl; // 7
char c = 'c';

bool find = s.find(c) != std::string::npos;
```

## vector 
1. 获取当前索引两种方式：
```cpp
for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << "Index: " << std::distance(vec.begin(), it) << ", Value: " << *it << std::endl;
}
```
```cpp
size_t index = 0;
for (const auto& element : vec) {
    std::cout << "Index: " << index << ", Value: " << element << std::endl;
    index++;
}
```
