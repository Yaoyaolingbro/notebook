# Fundamentals of Computer Design

## History and background
![20240914103716.png](graph/20240914103716.png)
> 当多核出现时，每个核都有自己的pipeline，这时的指令冲突更为严峻。

memory也会按照分层结构进行设计，以减少访问时间。
![20241023234631.png](graph/20241023234631.png)

接下来讲解了cpu发展史，Moore law & Dennard scaling，以及cpu的性能指标。
<!-- prettier-ignore-start -->
??? info "Dennard scaling"
    ![20241023235105.png](graph/20241023235105.png)
<!-- prettier-ignore-end -->

<!-- prettier-ignore-start -->
??? info "Amdahl's Law"
    ![20240914110521.png](graph/20240914110521.png)
<!-- prettier-ignore-end -->

## What type of Computer?

- PMD: Personal Mobile Device
  > real-time performance: a maximum execution time for each application segment (牺牲性能但提高反应时间和predict的能力)
  
- Desktop
- Server
- Cluster/WSC (Warehouse-Scale Computer)
- Embedded/IoT computer

![20241024000812.png](graph/20241024000812.png)

![20241023235708.png](graph/20241023235708.png)

## What make the computer fast?
### Parrallelism
- application-level parallelism
  - DLP: Data-level parallelism > ![20241023235930.png](graph/20241023235930.png)
  - TLP: Task-level parallelism > ![20241024000012.png](graph/20241024000012.png)
- hardware-level parallelism (four ways) > ![20241024000035.png](graph/20241024000035.png)

### Class of parrallel architecture
![20241024000251.png](graph/20241024000251.png)
    

## What's computer architecture
### ISA (Instruction Set Architecture)
![20241024000329.png](graph/20241024000329.png)

<!-- prettier-ignore-start -->
???+ note "why we need memory alignment"
    ![20241024000518.png](graph/20241024000518.png)
<!-- prettier-ignore-end -->

> Different Address Model and meaning
> ![20241024000634.png](graph/20241024000634.png)

---

## Trends in Technology
![20241024154639.png](graph/20241024154639.png)

## Power and Energy
- 1 watt = 1 joule per second
- energy to execute a workload = avg power x execution time

### Power Consumption
![20241024155916.png](graph/20241024155916.png)

<!-- prettier-ignore-start -->
??? info "Example"
    ![20240923110258.png](graph/20240923110258.png)
<!-- prettier-ignore-end -->

### How to economize energy?
- do nothing!!!
- **DVFS **(Dynamic Voltage and Frequency Scaling)
- design for typical case
- Proportional(相应，对应) to the number of devices



## Cost
- time
- volume
- commoditization

> ![20241024160348.png](graph/20241024160348.png)



### How to measure dependability
![20240930102627.png](graph/20240930102627.png)
$$
\text{Module availability} = \frac{\text{MTTF}}{\text{MTTF} + \text{MTTR}}
$$

### RAID
RAID: redundant array of inexpensive/independent disks（磁盘冗余阵列）
![20240930102803.png](graph/20240930102803.png)
- RAID 0: just a bunch of disks(JBOD or stripe)
- RAID 1: mirroring or shadowing
- RAID 2: http://www.acnc.com/raidedu/2 
- RAID 3: byte-level striping with dedicated parity http://www.acnc.com/raidedu/2 


<!-- prettier-ignore-start -->
??? info "Further reading"
    https://www.youtube.com/watch?v=jgO09opx56o 
<!-- prettier-ignore-end -->


### Measure Performance
![20240930103036.png](graph/20240930103036.png)
<!-- prettier-ignore-start -->
???+ note "SPEC"
    [SPEC CPU2017](https://www.spec.org/cpu2017/)
    20 SPECspeed benchmarks                  
    23 SPECrate benchmarks
<!-- prettier-ignore-end -->

例子：
![20241024161837.png](graph/20241024161837.png)

### Quantitative Principles of Computer Design
- Amdahl's Law: performance improvement to be gained from using some faster mode of execution is limited by the fraction of the time the faster mode can be used
- spatial locality
- temporal locality

![20241024162003.png](graph/20241024162003.png)

![20240930103339.png](graph/20240930103339.png)