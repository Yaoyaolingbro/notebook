# 计算机科学导论
<strong>注：这里是计算机科学导论这本书以及[计算机科学速成课](https://www.bilibili.com/video/BV1EW411u7th/?spm_id_from=333.337.search-card.all.click&vd_source=3e9ace1dd8c2c6302f6a14e69cb4cdab)记录的一些计算机科学的基本内容。二者一起食用会更美味。相信大家学习完这个课程之后会对“计科”有一个整体的了解！</strong>

## 数字逻辑

## 计算机组成

1. ALU 

- 算术单元

- 半加器
- 全加器

2. 逻辑单元

* 寄存器和内存
* 锁存器和寄存器

3. RAM + 寄存器 + ALU 组成一个CPU

   * 取指令-》解释-》执行（fetch -> decode -> execute) 

   * 时钟
4. instruction and programs

   - operation code = opcode
   - opcode could instruct the machine to do something on the provided address。
   - HALT instruction(make CPU stop)
   - jump
5. Advanced CPU Designs
   * cache（缓存）（synced up 同步）
   * more advanced instruction
   * parallelize(并行处理)（throughput 吞吐量）