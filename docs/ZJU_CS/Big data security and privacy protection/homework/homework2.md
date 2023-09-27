<div class="cover" style="page-break-after:always;font-family:方正公文仿宋;width:100%;height:100%;border:none;margin: 0 auto;text-align:center;">
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:10%;">
        </br>
        <img src="https://raw.githubusercontent.com/Keldos-Li/pictures/main/typora-latex-theme/ZJU-name.svg" alt="校名" style="width:100%;"/>
    </div>
    </br></br></br></br></br>
    <div style="width:60%;margin: 0 auto;height:0;padding-bottom:40%;">
        <img src="https://raw.githubusercontent.com/Keldos-Li/pictures/main/typora-latex-theme/ZJU-logo.svg" alt="校徽" style="width:100%;"/>
	</div>
    </br></br></br></br></br></br></br></br>
    <span style="font-family:华文黑体Bold;text-align:center;font-size:20pt;margin: 10pt auto;line-height:30pt;">《Big Data Security and Privacy Protection》</span>
    <p style="text-align:center;font-size:14pt;margin: 0 auto">Set 2 </p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> Set 2</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> August 27th</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">授课教师</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> Rongxing Lu </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">姓　　名</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 杜宗泽</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">学　　号</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">3220105581 </td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">组　　别</td>
    		<td style="width:%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> 个人</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">日　　期</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">8月27日</td>     </tr>
    </tbody>              
    </table>
</div>



<!-- 注释语句：导出PDF时会在这里分页 -->

#　Set 2

## Question 4

![](graph\Snipaste_2023-08-28_09-42-50.png)

![](graph\Snipaste_2023-08-28_09-43-07.png)

**My answer:**

To balance patients' privacy and the utility of the dataset when releasing a dataset about Hospital Covid19 Cases, the hospital can follow these design guidelines:

1. Anonymize Personally Identifiable Information (PII): Remove or generalize any attributes that directly identify individuals, such as names, addresses, social security numbers, or specific dates of birth. This helps protect patient privacy.(Just like the k-anonymity in question 5)
2. Aggregate Data: Instead of releasing individual-level data, aggregate the data to a higher level, such as by grouping patients based on age ranges, geographic regions, or other relevant categories. Aggregating data helps protect individual privacy while still providing useful insights.
3. Implement Access Controls: Limit access to the released dataset to authorized individuals or organizations. This helps prevent unauthorized use or disclosure of sensitive information.
4. Obtain Informed Consent: If possible, obtain informed consent from patients before releasing their data. This ensures that patients are aware of how their data will be used and allows them to make an informed decision.
5. Data Sharing Agreements: Establish data sharing agreements with the recipients of the dataset. These agreements should outline the purpose of data usage, restrictions on data sharing, and measures to protect patient privacy.
7. Regular Data Audits: Conduct regular audits to ensure compliance with privacy regulations and data protection measures. This helps identify any potential privacy risks and allows for timely corrective actions.

By following these design guidelines, the hospital can release a version of the dataset that balances patients' privacy and the utility of the dataset for scientific analytics.



## Question 5

![](graph\Snipaste_2023-08-28_09-43-16.png)



**My answer:**

The Generalized-table:

|         |    QI    |      |
| :-----: | :------: | :--: |
| Zipcode |   Age    | Sex  |
|  476**  |    2*    |  *   |
|  476**  |    2*    |  *   |
|  476**  |    2*    |  *   |
|  4790*  | [43, 52] |  *   |
|  4790*  | [43, 52] |  *   |
|  4790*  | [43, 52] |  *   |

In the generalized table, the value of k is 2 for that each combination of quasi-identifiers (Z1, A1, S1) appears at least twice in the table.

To determine the value of l for distinct l-diversity, we need to check each QI group in the generalized table. In this case, we have one QI group, which is (Z1, A1, S1). Since each combination appears at least twice, the value of l for distinct l-diversity is 2.

To determine the value of l for entropy l-diversity, we calculate the entropy for each QI group. In this case, we have one QI group, which is (Z1, A1, S1). The entropy of this group is calculated as:

entropy(g) = log2(number of distinct values in g)

For (Z1, A1, S1), the distinct values are: Z1: 476**, 4790* A1: 2*, [43, 52] S1: *

The number of distinct values in (Z1, A1, S1) is 4. Therefore, the value of l for entropy l-diversity is 4.

Summarization:

- The value of k in the generalized table is 2.
- The value of l for distinct l-diversity is 2.
- The value of l for entropy l-diversity is 4.



## Question 6

![](graph\Snipaste_2023-08-28_09-43-44.png)

![](graph\Snipaste_2023-08-28_09-44-00.png)

**My answer:**

To generate a k-anonymous version of the table while balancing privacy and utility, we need to ensure that each combination of quasi-identifiers (attributes that can potentially identify individuals) appears at least k times in the released table. In this case, the quasi-identifiers are Gender, Married, Age, and Sports Car.

To determine the value of k, we need to consider the sensitivity of the quasi-identifiers and the desired level of privacy. A higher value of k provides stronger privacy protection but may result in a loss of utility. Generally, a common approach is to start with a value of k=2 and then increase it if necessary to achieve the desired privacy level.

Regarding l-diversity, it refers to the requirement that each group of records with the same quasi-identifiers must have at least l distinct values for sensitive attributes. The value of l depends on the sensitivity of the sensitive attribute and the desired level of diversity. 

Above all, we set the k=3 & l=2 for this model. The concrete decision tree label is:

| Gender | Married | Age  | Sports Car | Loan Risk |      |
| ------ | ------- | ---- | ---------- | --------- | ---- |
| Male   | *       | Y    | *          | GOOD      |      |
| Male   | *       | Y    | *          | GOOD      |      |
| Male   | *       | Y    | *          | GOOD      |      |
| Male   | *       | O    | *          | GOOD      |      |
| Male   | *       | O    | *          | BAD       |      |
| Male   | *       | O    | *          | BAD       |      |
| Female | Yes     | *    | *          | GOOD      |      |
| Female | Yes     | *    | *          | GOOD      |      |
| Female | Yes     | *    | *          | BAD       |      |
| Female | No      | *    | *          | BAD       |      |
| Female | No      | *    | *          | BAD       |      |
| Female | No      | *    | *          | BAD       |      |



## Question 7

![](graph\Snipaste_2023-08-28_09-44-12.png)

![](graph\Snipaste_2023-08-28_09-44-22.png)

**My answer:**

(a) To determine the sensitivity of the function S(F) of Sum() in this table D, we need to consider the maximum possible change in the output of the function when a single record in the table is modified. In this case, the function Sum(D) calculates the total electricity consumption in the residential area for a given day. The sensitivity of this function can be determined by finding the maximum difference in the output when a single record is changed.

Let's consider two scenarios:
1. The maximum electricity use in the table is 5, and we change the electricity use of a user from 5 to 0. In this case, the maximum change in the output would be 5 (from subtracting 5 from the sum).
2. The minimum electricity use in the table is 0, and we change the electricity use of a user from 0 to 5. In this case, the maximum change in the output would be 5 (from adding 5 to the sum).

Therefore, the sensitivity of the function S(F) of Sum() in this table D is 5.



(b) By reading some relevant book, my answer are as follows.

Consider achieving ε-differential privacy in the statistical function query "Sum(D)" by adding random noise from a Laplacian distribution.

To achieve ε-differential privacy, we need to add random noise from a Laplacian distribution with a scale parameter of $\Delta/\epsilon$, where $\Delta$ is the sensitivity of the function. In this case, the sensitivity $\Delta$ is 5, and let's assume we want to set the privacy level $\epsilon$.

To randomly choose a random noise from a Laplacian distribution, we can use the following steps:
1. Generate a random number r from a uniform distribution between 0 and 1.
2. Calculate the noise value $N$ as $N = -\Delta/\epsilon \times sign(r-0.5)\times \ln^{1-2|r-0.5|}$.

By adding this noise N to the result of the function Sum(D), we can achieve ε-differential privacy in the statistical function query "Sum(D)".

Pr[A(D + I) ∈ T] ≤ e^ε * Pr[A(D - I) ∈ T], is a general result for ε-differential privacy, and it holds for the Laplacian noise mechanism as well.

