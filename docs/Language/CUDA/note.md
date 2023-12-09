# CUDA入门基础
<!-- prettier-ignore-start -->
!!! note "导读"
    本站将记录CUDA语法的基础用法，以及一定的举例。具体问题的研究可以见其他板块。
    [TOC]
<!-- prettier-ignore-end -->
## 环境搭建
<!-- prettier-ignore-start -->
!!! warning "待搭建"
    === "linux"
    ```
    sudo apt install nvidia-driver-515 # 版本需要查询
    sudo apt install nvidia-cuda-toolkit
    sudo reboot
    ```

    === "windows"
<!-- prettier-ignore-end -->

其中编译时我们采用`nvcc`编译，用法类似gcc/g++。
```shell
nvcc main.cu -o outpot_filename
```

如果只是编译多个源文件，即CUDA代码写到其他源文件里然后include到main文件时，需要添加编译选项
```shell
nvcc main.cu -o output_filname -x cu
```

**Attention**: 有时候可能因为nvcc默认使用g++/gcc编译器不兼容，可以利用`--compiler-bindir`设定

