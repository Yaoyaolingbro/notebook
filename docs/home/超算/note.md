# 超算短学期笔记

## 7月3号 —— 一些些介绍

* 大家的自我介绍以及一点点入门
* 一点点计网的知识

## 7月4日 —— 计算机体系结构和高性能计算基础

### 计算机系统

#### Overview

* ISA and x86 Instructions
* Processor Architecture
* Memory Hierarchy
* Concurrency Basic

#### Prerequisite Checklist

* Digital representation of values
* Memory & Address
* C code

#### ISA & x86 instruction

1. ISA: instruction ser architecture.
1. Assembly Language

#### Processor Architecture

1. Microarchitecture is implementation.(could make faster)
2. thread（线程）
3. fetch -> Decode -> Execute -> Commit 
4. Data hazards: You haven't already written but the next process need to call the data.
5. control hazard: parallel processing could make a fault.
6. structure hazard : like fetch and execute could make conflict.
7. SRAM & DRAM：SRAM = cache（高缓冲）.

#### Memory Layout

1. Stack: Runtime stack (8MB limit) local variables.
2. Heap: Dynamically allocated as needed.
3. Data: statically allocated data.
4. virtual memory: OS utilizes virtual memory to isolate address space of different processes and provide each process the same linear address space(线性映射，类似于哈希)
5. Translation Lookaside buffer（TLB） -> could accelerate the virtual memory and physical memory interactive.
