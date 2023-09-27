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
    <p style="text-align:center;font-size:14pt;margin: 0 auto">Set 1 </p>
    </br>
    </br>
    <table style="border:none;text-align:center;width:72%;font-family:仿宋;font-size:14px; margin: 0 auto;">
    <tbody style="font-family:方正公文仿宋;font-size:12pt;">
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">题　　目</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> set 1</td>     </tr>
    	<tr style="font-weight:normal;"> 
    		<td style="width:20%;text-align:right;">上课时间</td>
    		<td style="width:2%">：</td> 
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋"> August 24th</td>     </tr>
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
    		<td style="width:40%;font-weight:normal;border-bottom: 1px solid;text-align:center;font-family:华文仿宋">8月24日</td>     </tr>
    </tbody>              
    </table>
</div>

## Question 1

![](graph\Snipaste_2023-08-24_17-12-45.png)



**My answer:**

Confidentiality, integrity, and availability are three key aspects of information security. In the context of an automated cash deposit machine, here are examples of requirements for each aspect:

1. Confidentiality:
   - Requirement: User account information should be kept confidential and protected from unauthorized access.
   - Importance: High. Maintaining the confidentiality of user account information is crucial to prevent identity theft, fraud, and unauthorized transactions.
2. Integrity:
   - Requirement: Cash deposits should be accurately recorded and credited to the correct user account without any tampering or alteration.
   - Importance: High. Ensuring the integrity of cash deposits is crucial to maintain trust in the system and prevent financial discrepancies or disputes.
3. Availability:
   - Requirement: The cash deposit machine should be available and operational for users to deposit cash at any time.
   - Importance: High. Availability is critical for users who rely on the machine to deposit cash conveniently. Downtime or unavailability may inconvenience users and impact their trust in the system.

It's important to note that the importance of these requirements may vary depending on the specific context and the organization's risk assessment. These examples provide a general understanding of the confidentiality, integrity, and availability requirements associated with an automated cash deposit machine.



## Question 2

![](graph\Snipaste_2023-08-24_20-51-48.png)

![](graph\Snipaste_2023-08-24_20-52-10.png)

![](graph\Snipaste_2023-08-24_20-56-57.png)

**My answer:**

(a) The encryption algorithm used in this case is a simple substitution cipher. In a simple substitution cipher, each letter in the plaintext is replaced with a corresponding letter from the ciphertext according to a fixed substitution rule.

To decrypt the given ciphertext, we need to find the corresponding plaintext letters based on the substitution rule. In this case, the substitution rule is based on the first sentence of the book "The Other Side of Silence" by using the snowflakes as a key.

The plaintext can be obtained by replacing each ciphertext letter with the corresponding letter from the substitution rule. Here is the plaintext:

`SIDKHKDM AF HCRKIABIE SHIMC KD LFEAILA`
becomes
`BASILISK TO LEVIATHAN BLAKE IS CONTACT`

The reason is that the substitution rule is that the letter in turn in the given sentence is corresponding to the alphabetical.(eg: $s \rightarrow a$\ $i \rightarrow b$\ $d \rightarrow s$  )



(b) The simple substitution cipher is not considered secure. It is vulnerable to frequency analysis attacks, where an attacker can analyze the frequency of letters in the ciphertext and compare it to the expected frequency of letters in the language being used (in this case, English). By identifying the most frequently occurring letters in the ciphertext, an attacker can make educated guesses about the corresponding plaintext letters.

Additionally, the simple substitution cipher does not provide any form of key management or key distribution, making it susceptible to brute-force attacks. An attacker can try all possible substitution rules until the correct one is found.

Overall, the simple substitution cipher is a relatively weak encryption algorithm and is not recommended for secure communication.



## Question 3

![](graph\Snipaste_2023-08-24_20-57-10.png)



**My answer:**

In this question, we can use the python to help us solve it easily.

(a) how you would decrypt the cipher text (given values for *m* and *n*).

1. Partition the ciphertext into blocks of $m,n$ letters.
1. use python to rearrange the block.(just transpose the matrix and print in a line)

```python
string = "CTAROPYGHPRY"

matrix = [string[i:i+3] for i in range(0, len(string), 3)]

transposed_matrix = ["".join(row) for row in zip(*matrix)]

output = "".join(matrix)

print(output)
```



(b) in this question, I have to acknowledge that I use the google because I don't be conscious of the ciphertext matrix could be partitioned in blocks before transposed. Google shows me that this encryption is called "Rail Fence Cipher".

**The decrypt Python code:**

```python
import re

print("*****Rail Fence Cipher*****")

string = "MYAMRARUYIQTENCTORAHROYWDSOYEOUARRGDERNOGW"
m_ = []

for f in range(len(string)):
    if len(string) % (f + 1) == 0 and f > 0:
        m_.append(f + 1)

# print(m_)

for p in m_:
    print("\nGrouping into ", p, " characters per group, total ", int(len(string) / p), " groups", sep='')
    part = re.findall(r'.{%s}' % p, string)  # Grouping every p characters
    for q in range(p):
        if p % (q + 1) == 0:
            print("Each group ", q + 1, "x", int(p / (q + 1)), sep='')
            for each_part in part:
                for i in range(q + 1):
                    str_part = each_part[i::q + 1]  # [start:end:step] step defaults to 1
                    print(str_part, end='')
                # The next line controls the space, whether to include it or not
                print(" ", end='')
            print()

```

The operation results are as follows:

```shell
yaoyaoling@localhost ~/c/test> python3 test.py
*****Rail Fence Cipher*****

Grouping into 2 characters per group, total 21 groups
Each group 1x2
MY AM RA RU YI QT EN CT OR AH RO YW DS OY EO UA RR GD ER NO GW
Each group 2x1
MY AM RA RU YI QT EN CT OR AH RO YW DS OY EO UA RR GD ER NO GW

Grouping into 3 characters per group, total 14 groups
Each group 1x3
MYA MRA RUY IQT ENC TOR AHR OYW DSO YEO UAR RGD ERN OGW
Each group 3x1
MYA MRA RUY IQT ENC TOR AHR OYW DSO YEO UAR RGD ERN OGW

Grouping into 6 characters per group, total 7 groups
Each group 1x6
MYAMRA RUYIQT ENCTOR AHROYW DSOYEO UARRGD ERNOGW
Each group 2x3
MARYMA RYQUIT ECONTR ARYHOW DOESYO URGARD ENGROW
Each group 3x2
MMYRAA RIUQYT ETNOCR AOHYRW DYSEOO URAGRD EORGNW
Each group 6x1
MYAMRA RUYIQT ENCTOR AHROYW DSOYEO UARRGD ERNOGW

Grouping into 7 characters per group, total 6 groups
Each group 1x7
MYAMRAR UYIQTEN CTORAHR OYWDSOY EOUARRG DERNOGW
Each group 7x1
MYAMRAR UYIQTEN CTORAHR OYWDSOY EOUARRG DERNOGW

Grouping into 14 characters per group, total 3 groups
Each group 1x14
MYAMRARUYIQTEN CTORAHROYWDSOY EOUARRGDERNOGW
Each group 2x7
MARRYQEYMAUITN COARYDOTRHOWSY EURGENGOARDROW
Each group 7x2
MUYYAIMQRTAERN COTYOWRDASHORY EDOEURANRORGGW
Each group 14x1
MYAMRARUYIQTEN CTORAHROYWDSOY EOUARRGDERNOGW

Grouping into 21 characters per group, total 2 groups
Each group 1x21
MYAMRARUYIQTENCTORAHR OYWDSOYEOUARRGDERNOGW
Each group 3x7
MMRIETAYRUQNOHAAYTCRR ODYUREOYSEAGRGWOORDNW
Each group 7x3
MUCYYTAIOMQRRTAAEHRNR OEDYOEWURDANSROORGYGW
Each group 21x1
MYAMRARUYIQTENCTORAHR OYWDSOYEOUARRGDERNOGW

Grouping into 42 characters per group, total 1 groups
Each group 1x42
MYAMRARUYIQTENCTORAHROYWDSOYEOUARRGDERNOGW
Each group 2x21
MARRYQECOARYDOEURGENGYMAUITNTRHOWSYOARDROW
Each group 3x14
MMRIETAODYUREOYRUQNOHYSEAGRGAAYTCRRWOORDNW
Each group 6x7
MREADUEYUNHSARAYCRORNMITOYRORQOYEGGATRWODW
Each group 7x6
MUCOEDYYTYOEAIOWURMQRDANRTASROAEHORGRNRYGW
Each group 14x3
MCEYTOAOUMRARARAHRRRGUODYYEIWRQDNTSOEOGNYW
Each group 21x2
MOYYAWMDRSAORYUEYOIUQATRERNGCDTEORRNAOHGRW
Each group 42x1
MYAMRARUYIQTENCTORAHROYWDSOYEOUARRGDERNOGW
```

**Conclusion: ** The meaningful result is the situation where each group is divided into 2x3 fences for every 6 groups of 1, a total of 7 groups. The result is `MARY MARY QUITE CONTRARY HOW DOES YOUR GARDEN GROW`