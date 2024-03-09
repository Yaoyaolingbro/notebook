## Chap2: Relational Model

### 0. Outline

- 键值
- 关系代数

### 1. key

- superkey
- candidate key：minimal superkey
- primary key：selected candidate key
- foreign key：referencing->referenced(primary key)
- referential integrity

### 2. relational algebra

<!-- prettier-ignore-start -->
!!! definition "attributes"
    - The set of allowed values for each attribute is called the domain（域）of the attribute
    - Attribute values are (normally) required to be atomic（原子的）; that is, indivisible
    - The special valuenull （空）值is a member of every domain
<!-- prettier-ignore-end -->

- "Pure” languages:
  - Relational algebra：非图灵机等价
  - Tuple/Domain relational calculus
- 6 basic
  - select：条件称为**selection predicate**
  - project
  - union
  - set difference
    - union, set differenct都要求same arity，对应属性的type要compatible
  - Cartesian product
  - rename
- Additional
  - set intersection
  - natural join、theta join
  - assignment
  - outer join
  - semijoin
  - division operation
- extended
  - **Generalized Projection**：arithmetic functions of attributes
  - **Aggregation function**：avg、min、max、sum、count
- multiset中duplicates树
  - selection: 满足条件，保持
  - projection: 保持
  - cross product: **m** of *t1* in *r*, **n** of *t2* in *s*, **mn** in *r*  x *s*
  - union: m + n
  - intersection: min(m, n)
  - difference: min(0, m - n)