# 深究git

一个篇错的[思路汇总](https://missing.csail.mit.edu/2020/version-control/)，视频资源还是比较多的，我会把一些重要的命令或者工作流程记录在这里

## How to do team work.
学会使用git pull是十分重要的，以及分支管理，创建属于自己的feature-name分支。

<!-- prettier-ignore-start -->
!!! note "摘要"
    1. ZJU-git 中的group一定是你校内合作非常便捷的一种方式了。（当然Github上也有对印的，但个人觉得不如前者方便）
    2. 每次在更改前请先使用`git pull --all`来拉取完整仓库更新。
<!-- prettier-ignore-end -->


## Git常用命令指南
1. git commit
2. git branch <new branch name>
3. git checkout <new branch name>
  > 2&3可以合并为：git checkout -b <new branch name>
4. git merge(如果我们现在在main分支想要将checkout分支的合并过来，我们使用 git merge checkout)
5. git rebase(与merge相比也有它的好处),会使提交记录线性化更简介，~~清晰？~~
6. git -> main -> c1 or git checkout c1
7. git checkout HEAD^ | bugFix^(向上移动一次)
8. git branch -f main HEAD~3