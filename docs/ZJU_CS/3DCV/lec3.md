# Lec 3 : Image Processing

!!! note

    该部分笔记不是很完整。

## Image processing basics

### 一些基本的处理

- 增加对比度(使用"S curve")
- 模糊，锐化，边缘提取

### convolution

$$
(f*g)(x)=\int_{-\infty}^{\infty}f(y)g(x-y)dy
$$

#### Discrete 2D convolution


$$
(f*g)(x,y)=\sum_{i,j=-\infty}^{\infty}f(i,j)I(x-i,y-j)
$$
![](image/3.1.png)

#### Padding

如果没有padding的话，那么我们经过卷积操作之后得到的图像size会变小，为了得到大小相同的图像，我们选择在图像周围进行padding。

![](image/3.2.png)



### Guassian blur

- Obtain filter coefficients by sampling 2D Gaussian function $f(i,j)=\frac{1}{2\pi\sigma^2}e^{-\frac{i^2+j^2}{2\sigma^2}}$

![](image/3.3.png)

### Sharpening

锐化操作就是往图像里添加高频信息

- Let $I$ be the original image
- High frequencies in image $I=I-blur(I)$
- Sharpened image = $I+(I-blur(I))$



### Bilateral filter

**removing noise while preserving image edges**

![](image/3.4.png)

![](image/3.5.png)



## Image sampling

采样就是给定一个连续的函数，在不同的点求它的值，也可以认为，采样是把一个连续的函数离散化的过程。

![20240810123341.png](graph/20240810123341.png)

Image resizing: change image size/resolution.

Reducing image size -> down-sampling

但是我们在采样时有可能发生**反走样/锯齿**现象：

> alias
> ![20240810123003.png](graph/20240810123003.png)

原因可能是：

- 像素本身有一定大小
- 采样的速度跟不上信号变化的速度（高频信号采样不足）

### Fourier Transform

傅里叶变换可以将信号分解为频率，并且能将满足一定条件的某个函数表示成三角函数（正弦和/或余弦函数）或者它们的积分的线性组合。

![20240810122628.png](graph/20240810122628.png)
![20240810122701.png](graph/20240810122701.png)

> We can also use Fourier Transform to analyze the frequency content of an image and fix the aliasing problem.
> ![20240810124833.png](graph/20240810124833.png)
> 
> ![20240810123102.png](graph/20240810123102.png)
>
> ![20240810125139.png](graph/20240810125139.png)


## Image magnification

### Interpolation

![](image/3.6.png)

#### Nearest-neighbor interpolation

![](image/3.7.png)

- Not continuous
- Not smooth(光滑函数：几阶导连续就称该函数几阶光滑)

#### Linear interpolation

![](image/3.8.png)

#### Cubic interpolation

![](image/3.9.png)

**For each interval**: 并不是对整体进行拟合，而是每两点首先拟合线性函数，非线性项的引入是为了使曲线光滑



#### Bilinear Interpolation

