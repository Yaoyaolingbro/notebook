# 作业要点

## HW1
1. 用递归的方法求斐波那契数列的时间复杂度
<details>
在递归树中，每个节点表示一个递归调用，而树的深度表示递归的层数。对于斐波那契数列，每个节点会生成两个子节点，因为每个数都依赖于前两个数的和。

假设我们要计算第n个斐波那契数，递归树的深度将是n。每个节点的计算时间是常数时间，因为它只涉及到简单的加法操作。

因此，递归方法计算斐波那契数列的时间复杂度可以表示为O(2^n)。这是因为递归树的节点数是指数级增长的。
</details>

2. $$ P_1:T(1) = 1, T(N) = T(N/3)+1\\
P_2:T(1) = 1, T(N) = 3T(N/3) $$

求 $P_1, P_2$ 的复杂度

<details>
O(logN) for P1, O(N) for P2
</details>

3. 要温习``Mergesort`中merge的思想。


## HW2
1. Linear List（线性表）的初始定义是数组。
2. insertNode 函数可以背下来
```C
void insertNode(struct Node* head, int Element) {
    // 创建新节点
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->Element = Element;
    
    newNode->Next = (head)->Next;
    (head)->Next = newNode;
}
```


## HW3
1. stack pop `ooops`有多少种方式？   
<details>
5
</details>
2. stack的一种集成方法

```C
//初始化时将top赋值为-1
typedef struct {
    int stack[MAX_SIZE];
    int top;
} Stack;

void push(Stack *s, int num) {
    s->stack[++s->top] = num;
}

int pop(Stack *s) {
    return s->stack[s->top--];
}
```

## HW4
1. There exists a binary tree with 2016 nodes in total, and with 16 nodes having only one child.
<details>
F 本题是个脑经急转弯题目
</details>

2. Given a tree of degree 3. Suppose that there are 3 nodes of degree 2 and 2 nodes of degree 3. Then the number of leaf nodes must be ____.

<details>
8
</details>

3. If a general tree T is converted into a binary tree BT, then which of the following BT traversals gives the same sequence as that of the post-order traversal of T?

<details>
[普通树转二叉树](https://blog.csdn.net/best_LY/article/details/121346561)
</details>

<!-- prettier-ignore-start -->

??? info "Tips"
    [普通树转二叉树](https://blog.csdn.net/best_LY/article/details/121346561)

<!-- prettier-ignore-end -->

!!! Note
    T的preorder = BT的preorder
    T的postorder = BT的inorder

4. Threaded Binary Trees(一种对二叉树的优化，老师不讲但要掌握) 
!!! Note 
    [线索二叉树](tree.md)


## HW5
1. In a binary search tree which contains several integer keys including 4, 5, and 6, if 4 and 6 are on the same level, then 5 must be their parent.
<details>
F 5 could be their grandparents
</details>
2. 2-3?
3. 什么是decision tree?



2. A binary search tree if ood nodes, 如果我们选`i/2`，之后每次都选`i/2`；选`i/2+1`则都选`i/2+1`。


## HW6
1. heap两种插入方式，具体可见hello算法书
2. 编程题有序数字串建完全二叉树可以利用中序历遍的思想来建树
3.  红黑树？
