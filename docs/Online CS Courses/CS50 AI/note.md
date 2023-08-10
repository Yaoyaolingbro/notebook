## CS50 AI



## Lecture0    search

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





## Lecture1 Knowledge

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



## Lecture2  Probability

Probably ultimately boils down to (归结为) the idea(like roll a die)

$0 \leq P(\omega) \leq 1$  &  $\sum_{\omega\in\Omega}^{} P(\omega) = 1$

Negative: $P(\neg a) = 1 - P(a)$

Marginalization:$P(a) = P(a,b) + P(a,\neg b)$

![](graph\Snipaste_2023-08-07_21-24-32.png)

![](graph\Snipaste_2023-08-07_21-24-32.png)

calculate solution：$$P(a|b) =\frac{P(a \land b)}{P(b)}$$

Independence is crucial. When a & b is independent, $P(a \land b)=P(a) \times P(b)$



### Bayer's Rule

$P(b|a) = \frac{P(b) \times P(a|b)}{P(a)}$

### Joint Probability

$P(C|rain) = \alpha \times P(C, rain)$

### Marginalization

$P(X = x_i) = \sum_{j}P(X = x_i, Y = y_j) = \sum_{j}P(X = x_i| Y = y_j)P(Y = y_j)$

### Condition

$ P(a) = P(a|b)P(b) + P(a|\neg b)P(\neg b)$

### Bayesian Network

data structure that represents the dependencies among random variable 



![](graph\Snipaste_2023-08-08_15-34-32.png)

### Markov

![](graph\Snipaste_2023-08-08_16-19-44.png)