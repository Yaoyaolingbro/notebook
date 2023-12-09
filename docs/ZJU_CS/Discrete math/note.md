# 摘要

## Chapter 1
|英文|中文|数学符号|
|:--:|:--:|----|
|disjunction|并|$\cup$|
| conjunction | 交 | $\cap$ |
| lemma | 引理 |  |









## Chapter 7 Relations

### Relations and their properties

1. We write  $aRb$  for $(a,b) \in R$

> **Definition:** A binary relation R between A and B is a subset of Cartesian product $A \times B$ : $R \subseteq A \times B$
>
> When $A=B$, $R$ is called **a relation on** set $A$.

2. Distinguish **domain** and **range**. (just like function)

3. n-ary realtions: n is called its **degree**.

4. The composite of R and S is the realtion: $S \circ R = \{(a,c)|a \in A, c \in C \quad \exists b \in B\}$ such that  $(a,b)\in R$ and $(b,c)\in S$

5. Power: $R^{n+1} = R^n \circ R$

6. Inverse:$R^{-1} = \{(y,x)|(x,y) \in R\}$

7. Reflexive / Irreflexive: R is **reflexive** $\Leftrightarrow \forall x \in A, (x,x) \in R$ 

8. symmetric / antisymmetric: 

   R is **symmetric** $\Leftrightarrow \forall x, y \in A, (x,y)\in R \Rightarrow (y,x)\in R $   $\Leftrightarrow R^{-1} = R$

   R is **antisymmetric**: $\Leftrightarrow \forall x, y \in A, (x,y)\in R \quad and \quad (y,x)\in R \Rightarrow x = y $  $\Leftrightarrow R \cap R^{-1} \subseteq R_=$

   > Non-symmetric $\not\Leftrightarrow$ antisymmetric (eg. $R_=$)

9. R is **transitive** $\Leftrightarrow \forall x, y, z \in A((x,y)\in R \wedge(y,z) \in R) \Rightarrow (x,z)\in R$

   > **Theorem**:R on a set A is transitive if and only if $R^n \subseteq R$ for $n = 2, 3, \dots$
   >
   > **Inductive step:** $R^{n+1}$ is also a subset of $R$



### Representing Relations

1. Matrices representation.

   > - reflexive $\Leftrightarrow$ All terms $m_{ii}$ in the main diagonal of $M_R$ are 1
   > - symmetric $\Leftrightarrow m_{ij} = m_{ji}$ for all $i,j$.
   > - anti-symmetric $\Leftrightarrow$ if $m_{ij} = 1 $ and $i\not= j$ then $m_{ij} = 0$
   > - Transitive $\Leftrightarrow$  whenever $c_{ij}$ in $C=M_R^2$ is nonzero then entry $m_{ij}$ in $M_R$ is also nonzero

2. Digraphs representation.

   > - A edge $(a,b)$, a isiInitial vertx and b is terminal vertex
   > - A edge of form $(a,a)$, called **loop**
   > - reflexible $\Leftrightarrow$ There are loops at every vertex of digraph.
   > - symmetric $\Leftrightarrow$ Every edge between distinct vertices is accompanied by a edge in the opposite direction.



### Closures of Relations

1. **Definition:** $R \& S$ are relation,while S satisfy:

   > - S with property **P** and $R \subseteq S$
   > - $\forall S'$ with property **P** and $R \subseteq S'$ , then $S \subseteq S'$

2. **Theorem:** R be a relation on set A.

   > - The **reflexive closure** of relation R: $$r(R) =R \cup \Delta$$, where $\Delta = \{(a,a)|a \in A\}$
   > - The **symmmetric closure** of relation R: $$S(R) = R \cup R^{-1}$$

3. **Definition:** Path is a sequence of one or more edges in graph G.

   > **Theorem:**  Let R be a relation on set $A$. There is a path of length n from a to b $\Leftrightarrow (a,b) \in R^n$

4. **Definition**: The **connectivity relation** $R^* = \{(a,b)|\text{there is a path from a to b}\}$.$$R^* = \cup^{\infin}_{n=1} R^n$$

   > **Theorem**: The transitive closure of R :$$t(R) = R^*$$

5. WARSHALL'S algorithm!



### Equivalence Relations

1. **Definition**: Relation $R$~ : $A \leftrightarrow A$ is an **equivalence relation**, if it reflexive, symmetric and transitive.
2. **Definition**: Let $R: A \leftrightarrow A$ is an equivalence relation. For any $a \in A$, the **equivalence class** of a is the set of the elements related to a. $$ [a]_R = \{x\in A|(x,a) \in R\}$$ . If $b\in [a]_R$. b is called a representative of this equivalence class.















## Chapter 8

