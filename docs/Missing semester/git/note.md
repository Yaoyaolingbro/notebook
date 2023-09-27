# 深究git

一个篇错的[思路汇总](https://missing.csail.mit.edu/2020/version-control/)，视频资源还是比较多的，我会把我大致的学习经过记录在这里

## Missing Semester Note


## Git图形化教程
1. git commit
2. git branch <new branch name>
3. git checkout <new branch name>
4. 2&3可以合并为：git checkout -b <new branch name>
5. git merge(如果我们现在在main分支想要将checkout分支的合并过来，我们使用 git merge checkout)
6. git rebase(与merge相比也有它的好处)
7. git -> main -> c1 or git checkout c1
8. git checkout HEAD^ | bugFix^(向上移动一次)
9. git branch -f main HEAD~3
  patch(修补)



## 团队合作

学会使用git pull是十分重要的，以及分支管理，创建属于自己的feature-name分支。

可以图形化教程中有远程仓库教学。