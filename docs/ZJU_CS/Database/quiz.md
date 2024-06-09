# Quiz
> 本节为sjl老师数据库课后随堂习题的汇总

## week9
1. which physical storage media is non-volatile?
    > flash memory | magnetic disk | SSD | magnetic tape | optical disk

2. which physical storage medias belong to secondary storage?
    > flash memory | magnetic disk | SSD 

3. which term represents the time that the disk controller takes to reposition the disk arm over the correct track?(考察`Performance Measures of Disks`)
    > seek time

4. What is the right approach to optimizing data access on a disk?
    > - Buffering 
    > - Read-ahead
    > - defragment the file system
    > - Non-volatile write buffer
    > - Log disk

5/6. What's MTTF & IOPS?
    > - MTTF: Mean Time To Failure
    > - IOPS: Input/Output Operations Per Second

7. What is contained in the header of slotted page?
   ![20240609152124.png](graph/20240609152124.png)

8. Judge different file organization!!!

9. Which statement is incorrect? (`D`)
    > - A. For heap file organization, records can be placed anywhere in the file where there is space
    > - B. Database system seeks to minimize the num of block transfers between the disk and memory
    > - C. If the needed block is not in the buffer, the buffer manager will replace some block in the buffer.
    > - D. LRUs are the most efficient replacement policy

10. LRU Quiz
