## CS50 AI



## Lecture 0    search

* Knowledge: draw inference from information.

* Uncertain/Probably

* Optimization

* Learning
* Neural networks: computer analog to that sort of idea.
* Language.



### Search Problem :

1. Result(s,a): state & action
2. State space: the set of all states reachable from initial state.
3. graphic associate all states and we need to know the  goal test.
4. path cost: we hope it could be low.(minimize)



###　General Search

a data structure that keeps track of

![](graph\Snipaste_2023-08-04_11-13-22.png)

to avoid the mistakes, we need add the node to explored set.

pseudocode:伪代码

> Depth-First Search <--> Breadth-First Search

Heuristic function? leads **Greedy Best-First Search.**



However, the Greedy Best-First Search is not always best for finding the shortest way. The A* Search generates.

![](graph\Snipaste_2023-08-04_16-48-54.png)





###　Adversarial Search （tic-tac-toe 井字棋）

![](graph\Snipaste_2023-08-05_11-12-30.png)

 

### Alpha-Beta Pruning

### Depth-Limited Minimax

Evaluation function that estimate the expected utility of the game from a given state.





## Lecture 1 Knowledge

Knowledge-based agents that reason by operating on internal representation of knowledge.

assertion断言

### Proposition(命题) Knowledge

five logical connectives:

![](graph\Snipaste_2023-08-06_15-27-01.png) 

> implication: only P is true and Q is false ,the result is false.
>
> biconditional: both P & Q are true or false leads true .



* entailment: A is true , B is true.

If we wonder whether a logic is right, we can check it in all model.

### Knowledge Engineering

Game Clue

![](graph\Snipaste_2023-08-07_10-42-52.png)

![](graph\Snipaste_2023-08-07_10-43-28.png)