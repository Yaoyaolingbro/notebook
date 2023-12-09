# Advanced C Language

## C 中的一些工具（gcc、gdb、makefile）
我放在了[这里](https://yaoyaolingbro.github.io/notebook/Missing%20semester/Linux/C/)

## 跟汇编相关

1. [内联汇编（如何扩充函数的栈的大小）](https://blog.csdn.net/qq_38600065/article/details/110321320)

## 左右值

## <stdlib.h>中的sort函数
```C
#include <stdio.h>
#include <stdlib.h>

// 比较函数
int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int main() {
    int arr[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    int n = sizeof(arr) / sizeof(arr[0]);

    qsort(arr, n, sizeof(int), compare);

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }

    return 0;
}
```

## ++的前缀和后缀

```
C:
counter++;

assembly:
 mov [addr], &eax
 add $0x1, &eax
 mov &eax, [addr]
```

