# Advanced C Language

## C 中的一些工具（gcc、gdb、makefile）
我放在了[这里](https://yaoyaolingbro.github.io/notebook/Missing%20semester/Linux/C/)

## 跟汇编相关

1. [内联汇编（如何扩充函数的栈的大小）](https://blog.csdn.net/qq_38600065/article/details/110321320)

## 左右值



## ++的前缀和后缀

```
C:
counter++;

assembly:
 mov [addr], &eax
 add $0x1, &eax
 mov &eax, [addr]
```

