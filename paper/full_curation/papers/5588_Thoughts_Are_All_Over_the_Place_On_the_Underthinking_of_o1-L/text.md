## Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs

Yue Wang∗,1,2 , Qiuzhi Liu∗,1 , Jiahao Xu∗,1 , Tian Liang∗,1 , Xingyu Chen∗,1,3 , Zhiwei He∗,1,3 , Linfeng Song1 , Dian Yu1 , Juntao Li2 , Zhuosheng Zhang3 , Rui Wang3 ,

Zhaopeng Tu†1 , Haitao Mi1 , and Dong Yu1 1Tencent AI Lab 2Soochow University 3Shanghai Jiao Tong University

# arXiv:2501.18585v2[cs.CL]18Feb2025

Correct Response Incorrect Response

59.3

18000

60

|26.4<br><br>43.1<br><br>7.0<br><br>9.6<br><br>12.6|
|---|

|16,108<br><br>10,335<br><br>13,206<br><br>1,284 957<br><br>5,549 3,909<br><br>5,224<br><br>943 977|
|---|

NumberofThoughts

GeneratedTokens

12000

40

6000

20

0

0

Qwen-Math-72BLlama3.3-70BQwQ-32B-PreviewDeepSeek-R1-PreviewDeepSeek-R1-671B

QwQ-32B-PreviewDeepSeek-R1-PreviewDeepSeek-R1-671B

(a) Generated Tokens (b) Number of Thoughts

Figure 1: Illustration of the underthinking issue on the challenging AIME2024 testset: In o1-like models (e.g., QwQ-32B-Preview and DeepSeek-R1-671B), incorrect answers often switch reasoning strategies more frequently than correct ones (Figure b), leading to longer responses without improved accuracy (Figure a). In contrast, conventional LLMs (e.g., Qwen-Math-72B and Llama3.3-70B) show no significant difference in response length between incorrect and correct answers.

#### Abstract

Large language models (LLMs) such as OpenAI’s o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting humanlike deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty (TIP) that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities.

∗Equal Contribution. The work was done when Yue, Xingyu and Zhiwei were interning at Tencent AI Lab. †Correspondence to: Zhaopeng Tu <zptu@tencent.com>.

#### 1 Introduction

Large Language Models (LLMs), such as OpenAI’s o1 (OpenAI, 2024), have revolutionized artificial intelligence by enabling models to tackle increasingly complex tasks. The o1 model and its replicas (Qwen, 2024; DeepSeek, 2025; Kimi, 2025), known for their deep reasoning capabilities, exemplify the potential of LLMs to exhibit human-like deep thinking by scaling test-time computation during problem-solving. These models aim to explore diverse reasoning strategies, reflect on their decisions, and iteratively refine solutions, closely mimicking human cognitive processes.

Despite their successes, a critical yet underexplored question remains: Are o1-like LLMs thinking deeply enough? This study provides an initial exploration of this problem. In this work, we investigate a phenomenon we term underthinking, which refers to the tendency of o1-like LLMs to prematurely abandon promising lines of reasoning, leading to inadequate depth of thought. To systematically analyze underthinking, we conduct experiments on three challenging test sets (e.g., MATH500, GPQA Diamond, and AIME2024) and two open-source o1-like models with visible long chains of thought (e.g., QwQ-32B-Preview and DeepSeek-R1-671B). Through extensive analyses, we found that underthinking manifests in the following patterns: (1) it occurs more frequently on harder problems, (2) it leads to frequent switching between different thoughts without reaching a conclusion in each, and (3) it correlates with incorrect responses due to insufficient exploration of reasoning paths. For example, Figure 1 compares the token usage and number of thoughts of correct and incorrect responses. On average, o1-like LLMs consume 225% more tokens in incorrect responses than in correct ones due to 418% more frequent thought-switching behaviors.

To quantify this phenomenon, we introduce a novel underthinking metric that measures token efficiency in incorrect responses by evaluating the proportion of the response that contributes to reaching correct thoughts. Combining the widely-used accuracy metric with the proposed underthinking metric provides a more comprehensive assessment of o1-like models: accuracy measures how often the model can produce correct responses, while the underthinking metric evaluates the token efficiency within incorrect responses that contributes to reaching correct thoughts.

In response to these findings, we propose a decoding strategy with thought switching penalty (TIP) that discourages premature transitions between thoughts during the generation process. By adjusting decoding penalties for tokens associated with thought switching, the model is encouraged to thoroughly develop each line of reasoning before considering alternatives. Experimental results show that employing TIP improves accuracy across challenging test sets without requiring additional model fine-tuning.

Our study makes the following contributions:

- 1. We formally define and characterize the underthinking issue in o1-like LLMs, where models frequently abandon promising reasoning paths prematurely, leading to inadequate depth of reasoning on challenging problems.
- 2. We introduce a novel metric to evaluate underthinking by measuring token efficiency in incorrect responses, providing a quantitative framework to assess reasoning inefficiencies.
- 3. We propose a decoding approach with thought switching penalty (TIP) that encourages models to deeply explore each reasoning thought before switching, improving accuracy without additional model fine-tuning.

#### 2 Observing Underthinking Issues

In this section, we present a comprehensive analysis of outputs from o1-like models on challenging math problems. We begin by illustrating the frequent thinking switch phenomenon observed in responses to these problems, as shown in Figure 2, highlighting how this behavior differs significantly between correct and incorrect answers (Section 2.1). We then show that this phenomenon leads to an inadequate depth of reasoning, causing models to abandon promising reasoning paths prematurely (Section 2.2). Based on this observation, we propose a metric to empirically assess the underthinking

Let a, b, x, and y be real numbers with a>4 and b>1 such that \frac{x^2}{a^2}+\frac{y^2}{a^2-16}

###### Q:

=\frac{(x-20)^2}{b^2-1}+\frac{(y-11)^2}{b^2}=1. Find the least possible value of a+b.

So I've got this problem here that involves some equations with variables a, b, x, and y, and I need to find the least possible value of a

- Thought1 411 tokens

Alternatively, maybe there's a geometric interpretation. Since both equations represent ellipses, perhaps the condition is that these ellipses are tangent to each other. ... So, unless there's some transformation or something, they don't seem to be the same ellipse.

- Thought2 588 tokens

Alternatively, perhaps the problem is that for some (x,y), both equations hold simultaneously, and I need to find a and b such that this is possible. ... solve the first equation for y in terms of x and substitute into the second equation. But that seems messy.

- Thought3 109 tokens

Alternatively, perhaps I can consider this as an optimization problem where I need to minimize a + b subject to the condition that the equations have a solution. ... But I'm not sure how to set that up algebraically.

- Thought4 280 tokens

Alternatively, perhaps I can consider the distance between the centers of the two ellipses and relate it to their axes. ... But (y - 11)^2 is y^2 - 22y + 121. However, I only have y^2, not y. So maybe this isn't the best way to substitute.

- Thought5 613 tokens

+ b given some conditions. ... Perhaps I can think of this as a system of equations and try to solve for x and y in terms of a and b.

Solution1

20 More Thoughts Averaged 284 tokens

……

Given that, perhaps I need to accept that this problem is beyond my current ability and seek help or look for a different problem.

**Final Answer**: \[ \boxed{15} \]

- Figure 2: An example of underthinking issue for QwQ-32B-Preview model’s output response that consists of 25 reasoning thoughts within a single solution.

issues and present empirical results in Section 2.3. We conclude that o1-like LLMs often underthink when they fail to tackle challenging math problems.

2.1 Frequent Thinking Switch of o1-Like LLMs We conduct experiments on three testsets:

- • MATH500 (Hendrycks et al., 2021): a challenging dataset consisting of problems from high school math competitions across seven subjects (e.g., Prealgebra, Algebra, Number Theory) and difficulty levels based on AoPS (ranging from 1 to 5). Problems in these competitions range from level 1, the easiest, often found in AMC 8 exams, to level 5, like those in AIME.
- • GPQA (Rein et al., 2023): a graduate-level dataset consisting of multiple-choice questions in subdomains of physics, chemistry, and biology. For our experiment, we select the highest quality subset, known as GPQA Diamond (composed of 198 questions).
- • AIME (MAA Committees): a dataset from the American Invitational Mathematics Examination, which tests math problem solving across multiple areas (e.g. algebra, counting, geometry, number theory, and probability). Because AIME 2024 contains only 30 examples, we also considered 60 more examples from AIME 2022 and 2023.

We mainly investigate two widely recognized open-source o1-like models featuring visible long CoT: QwQ-32B-Preview and DeepSeek-R1-671B. We also include DeepSeek-R1-Preview to show the development of R1 series models. Given DeepSeek-R1-Preview’s daily message limit of 50 via web interface, we evaluated this model solely on the MATH500 and AIME test sets.

Definition of Reasoning Thoughts In this paper, we define thoughts as the intermediate cognitive steps within a reasoning solution produced by the model. O1-like LLMs often switch reasoning thoughts using terms like “alternatively”. For instance, as shown in Figure 2, the problem-solving process involves multiple reasoning thoughts, shifting from algebraic manipulation to geometric interpretation and optimization strategies. The ability to switch between different reasoning strategies allows for a broader exploration of potential solutions and demonstrates the flexibility of the model

QwQ-32B-Preview

QwQ-32B-Preview

QwQ-32B-Preview

DS-R1-Preview

DS-R1-671B

11.2

12

12

12

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

12

12

12

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

12

12

12

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

- 0
- 1
- 2
- 3
- 4
- 5

| | | | |h o<br><br>1<br><br>T T|oug ken<br><br>.5<br><br>hou oken<br><br>1.5<br><br>Tho Toke<br><br>1.5<br><br>4.2|h s<br><br>ug|ht s g n|shtsts s<br><br>1.9<br><br>4.8| | |2.3<br><br>6.2| | |4.0| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |1.1<br><br>3.5| | | | | | | | | | | | | | |

| | | | |h o|oug ken hou oken ho oke<br><br>1.2<br><br>2.7|h s<br><br>ug|t<br><br>1<br><br>h s g n|s<br><br>.5<br><br>ts<br><br>1.5<br><br>hts s<br><br>1.5<br><br>2.9| |2<br><br>|2.1.12.1<br><br>5.6| |3<br><br>|3.3.33.3<br><br>9.4| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |1.0<br><br>3.1| | | | | | | | | | | | | | |

| |1<br><br>| | |h o<br><br>1<br><br>T T|oug ken<br><br>.4<br><br>hou oken<br><br>1.4<br><br>Tho Tok<br><br>1.4<br><br>2.0|h s<br><br>ug e|t h s g n<br><br>2<br>3<br><br><br>|s<br><br>.1<br><br>.3<br><br>ts<br><br>2.1<br><br>3.3<br><br>hts s<br><br>2.1<br><br>3.3| | |2.8<br><br>4.6| | |4.4<br><br>6.0| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |1.4.41.4<br><br>2.6<br>| | | | | | | | | | | | | | |

10

10

10

10

10

10

10

10

10

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

#Tokens(K)

8

8

8

8

8

8

8

8

8

#Thoughts

#Thoughts

#Thoughts

#Thoughts

#Thoughts

#Thoughts

#Thoughts

#Thoughts

#Thoughts

6

6

6

6

6

6

6

6

6

4

4

4

4

4

4

4

4

4

2

2

2

2

2

2

2

2

2

0

0

0

0

0

0

0

0

0

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

(a) QwQ-32B-Preview

(b) DeepSeek-R1-Preview

(c) DeepSeek-R1-671B

- Figure 3: Average number of thoughts (“Thoughts”) and tokens (“Tokens”) in generated responses across different difficulty levels of the MATH500 test set.

50%

100%

46.7%

DeepSeek-R1-671B

DeepSeek-R1-671B

DeepSeek-R1-671B

DeepSeek-R1-671B

DeepSeek-R1-671B

DeepSeek-R1-671B

48.6%

48.6%

48.6%

84.2%

DeepSeek-R1-Preview

DeepSeek-R1-Preview

DeepSeek-R1-Preview

DeepSeek-R1-Preview

DeepSeek-R1-Preview

DeepSeek-R1-Preview

38.9%

38.9%

38.9%

QwQ-32B-Preview

QwQ-32B-Preview

QwQ-32B-Preview

QwQ-32B-Preview

QwQ-32B-Preview

QwQ-32B-Preview

44.1%

40%

40%

40%

80%

###### Math500-Level5 GPQA

71.1%

34.3% 28.6%

CorrectnessRatio

33.3% 32.4% 30.6% 30.6% 30.6%

33.3% 32.4% 30.6% 30.6% 30.6%

33.3% 30.6% 30.6% 30.6%

39.4%

32.4%

78.4%

32.0% 28.1%

63.2%

37.1% 31.4%

72.2%

Distribution

30%

60%

67.6%

33.3% 31.4%

33.3% 31.4%

33.3%

31.4% 31.4%

31.4% 31.4%

31.4%

31.4%

25.0% 25.0%

25.0% 25.0%

25.0% 25.0%

Correct Response Incorrect Response

Correct Response Incorrect Response

Correct Response Incorrect Response

Correct Response Incorrect Response

47.4%

59.5%

55.6%

20.0% 20.0%

20.0% 20.0%

20.0% 20.0%

20.0% 13.3%

20.0% 13.3%

20.0% 13.3%

25.5%

19.0% 13.3% 13.3%

36.8%

24.2%

48.6%

48.6%

48.6%

14000

14000

40

40

8000

8000

15

15

20%

20%

20%

40%

40%

40%

7,271

7,271

36.3

36.3

38.9%

38.9%

38.9%

13.3% 13.3%

13.3% 13.3%

12.9

12.9

12000

12000

NumberofThoughts

NumberofThoughts

NumberofThoughts

NumberofThoughts

21.1%

32.4%

32.4%

32.4%

GeneratedTokens

GeneratedTokens

GeneratedTokens

GeneratedTokens

10%

20%

20%

20%

13.2% 10.5%

30

30

6000

6000

24.3%

10.6

10.6

22.2%

9,769

9,769

10000

10000

9.9

9.9

5.3%

10

10

25.2 23.8

25.2 23.8

16.2%

0.0% 0.0% 0.0%

0.0% 0.0% 0.0%

0.0%

11.1%

10.8% 10.8%

10.8% 10.8%

10.8% 10.8%

4,441

4,441

8000

8000

0%

0%

0%

0%

0%

0%

5.6% 5.6% 5.6%

5.6% 5.6% 5.6%

5.6% 5.6% 5.6%

7,271 6,843

7,271 6,843

>0 ≥0.1 ≥0.2 ≥0.3 ≥0.4 ≥0.5 ≥0.6 ≥0.7 ≥0.8 ≥0.92.7%

20

20

4000

4000

0.0% 0.0%

1 2 3 4 5 6 7 8 9 ≥10

1 2 3 4 5 6 7 8 9 ≥10

1 2 3 4 5 6 7 8 9 ≥10

≥ ≥ ≥ ≥ ≥ ≥ ≥ ≥ ≥

≥ ≥ ≥ ≥ ≥ ≥ ≥ ≥ ≥

3,439

3,439

6000

6000

3,086

3,086

5.1

5.1

5

5

Index of Thought

Ratio of Correct Thoughts

3,706 2,883 2,624

3,706 2,883 2,624

4000

4000

10

10

2000

2000

6.6 6.6

6.6 6.6

2000

2000

4.5

4.5

0

0

0

0

0

0

0

0

QwQ-Prev

QwQ-Prev

QwQ-Prev

QwQ-Prev

QwQ-Prev

QwQ-Prev

QwQ-Prev

QwQ-Prev

R1-671B

R1-671B

R1-671B

R1-671B

R1-671B

R1-671B

R1-671B

R1-671B

R1-Prev

R1-Prev

R1-Prev

R1-Prev

(a) Math500-Hard

(b) GPQA Diamond

- Figure 4: O1-like LLMs switch thinking more frequently on incorrect responses, thus expend more tokens without contributing to accuracy.

in tackling complex problems. In this study, we provide a comprehensive analysis of the side effects associated with this ability to switch reasoning thoughts.

We utilize the Llama-3.3-70B model to automatically segment a response into reasoning thoughts due to its superior capabilities in both instruction following and mathematical reasoning. Initially, we manually analyzed responses from the QwQ-32B-Preview model to gather expressions indicative of shifts in thought. We then tasked the Llama-3.3-70B model with scanning the entire response to identify all occurrences of such expressions. Furthermore, we asked the model to determine whether these expressions truly signify a change in thought or merely reflect a stylistic pattern in the response. Only the expressions indicating a genuine thought shift were used as separators for reasoning processes.

- o1-Like LLMs Switch Thinking More Frequently on Harder Problems Figure 3 shows the averaged thoughts and tokens in generated responses across various difficulty levels in the MATH500 test set. Clearly, all models generate more reasoning thoughts with the increase of difficulty level, which is consistent with the growth of generated tokens. This observation suggests that as the complexity of the problems increases, the models tend to switch thoughts more frequently. This behavior implies that o1-like LLMs are able to dynamically adjust their reasoning processes to tackle more challenging problems. The following experiments focus on Level 5 in the MATH500 test set (MATH500-Hard).

- Increased Thought Switching in o1-Like LLMs during Incorrect Responses When examining the behavior of o1-like LLMs, we observe a distinct pattern in how they handle incorrect responses. As depicted in Figures 1 and 4, these models exhibit a significant increase in the frequency of thought switching while generating incorrect answers across all test sets. This trend suggests that although the models are designed to dynamically adjust their cognitive processes to solve problems, more frequent thought switching does not necessarily lead to higher accuracy. Essentially, the models may be expending additional computational resources – evidenced by an increase in generated tokens – without achieving more accurate solutions. These insights are crucial because they highlight the need not only to explore additional cognitive pathways when faced with challenges but also to
- operate in a more targeted and efficient manner, thereby improving accuracy even when complex reasoning is required. In the following sections, we empirically validate the inefficiencies associated with frequent thought switching in incorrect responses.

##### 2.2 Existence of Underthinking

The behavior of frequent thinking switch in incorrect responses could stem either from (1) genuine underthinking, where the model succeeds in finding promising strategies but fails to stick with them, or from (2) a lack of understanding, prompting it to explore diverse but ineffective approaches. To disentangle these possibilities, we propose an assessment framework that evaluates whether an abandoned reasoning path is actually sufficient to derive a correct answer. By focusing on whether the model can persistently follow and deepen a single, promising line of thought, we can identify instances of underthinking.

Assessing Thought Correctness In the example presented in Figure 2, we observe that some early thoughts may lead to the correct answer. For instance, Thought 1 initiates a correct interpretation by recognizing that the given equations resemble those of ellipses centered at (0,0) and (20,11). Setting the two expressions equal is a valid approach to finding common points (x, y) that satisfy both equations. Instead of concentrating on thoroughly exploring the plausible thought with further algebraic manipulation and optimization techniques, the model frequently shifts its focus and uses approximately 7,270 additional tokens without arriving at a correct answer. Ultimately, it concludes with a guessed answer that lacks support from the extended COT process.

We leverage LLMs to assess whether each thought leads to a correct answer using the following prompt:

|Problem P = {problem} Solution Draft S = {split solutions} Correct Answer A = {expected answer}<br><br>1. Please analyze the relevance between the solution S and the problem P, and conduct some verifications to check the correctness of the solution itself. Please think step by step to give an explanation **EXPLANATION**.<br>2. If you think the solution draft S can lead to the correct answer A of the problem P, please stick to the line of thinking without deviation and carry it through to completion. If you think it cannot yield the correct answer or you’re not sure, don’t force yourself to give an answer and generate **None**.<br>3. Please tell me honestly how confident you are that you can solve the problem P correctly based on the the solution draft S. Out of 2, please generate your confidence score **CONFIDENT SCORE**.<br><br><br>Please output **EXPLANATION** and **CONFIDENT SCORE** according to the following format: EXPLANATION: \boxed{} CONFIDENT SCORE: \boxed{}<br><br>|
|---|

QwQ-32B-Preview

DS-R1-Preview

DS-R1-671B

11.2

12

- 0
- 1
- 2
- 3
- 4
- 5

12

- 0
- 1
- 2
- 3
- 4
- 5

12

- 0
- 1
- 2
- 3
- 4
- 5

|1.1<br><br>3.5|Thoug Token<br><br>1.5<br><br>4.2|hts s<br><br>1.9<br><br>4.8|2.3<br><br>6.2|4.0|
|---|---|---|---|---|

|1.0<br><br>3.1|houg oken<br><br>1.2<br><br>2.7|hts s<br><br>1.5<br><br>2.9|2.1<br><br>5.6|3.3<br><br>9.4|
|---|---|---|---|---|

|1.4<br><br>2.6|Thoug Token<br><br>1.4<br><br>2.0|hts s<br><br>2.1<br><br>3.3|2.8<br><br>4.6|4.4<br><br>6.0|
|---|---|---|---|---|

10

10

10

#Tokens(K)

#Tokens(K)

#Tokens(K)

8

8

8

#Thoughts

#Thoughts

#Thoughts

6

6

6

4

4

4

2

2

2

0

0

0

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

[Figure 6]

On the Underthinking of o1-Like LLMs

50%

100%

46.7%

|10.8% 10.8%<br><br>16.2%<br><br>24.3%<br><br>32.4%<br><br>48.6%<br><br>59.5%<br><br>67.6%<br><br>78.4%<br><br>0.0%<br><br>5.3%<br><br>13.2% 10.5%<br><br>21.1%<br><br>36.8%<br><br>47.4%<br><br>63.2%<br><br>71.1%<br><br>84.2%<br><br>5.6% 5.6% 5.6%<br><br>11.1%<br><br>22.2%<br><br>38.9%<br><br>55.6%<br><br>72.2%<br><br>DeepSeek-R1-671B<br><br>DeepSeek-R1-Preview<br><br>QwQ-32B-Preview|
|---|

DeepSeek-R1-671B

48.6%

84

DeepSeek-R1-Preview

38.9%

QwQ-32B-Preview

44.1%

40%

80%

34.3% 28.6%

CorrectnessRatio

33.3% 30.6% 30.6% 30.6%

39.4%

32.4%

78

32.0% 28.1%

37.1% 31.4%

72

Distribution

30%

60%

33.3%

31.4%

31.4%

25.0% 25.0%

20.0% 20.0%

20.0% 13.3%

25.5%

19.0% 13.3% 13.3%

24.2%

20%

40%

10%

20%

0%

0%

>0 ≥0.1 ≥0.2 ≥0.3 ≥0.4 ≥0.5 ≥0.6 ≥0.7 ≥0.8 ≥0.92.7%

0.0% 0.0%

1 2 3 4 5 6 7 8 9 ≥10

Index of Thought

Ratio of Correct Thoughts

- Figure 5: The ratio of correct reasoning thoughts at each index in incorrect responses. A notable portion of early-stage thoughts (e.g., the first few thoughts) are correct but abandoned without being fully explored.

Specifically, we use two models distilled from DeepSeek-R1-671B based on Llama and Qwen – DeepSeek-R1-Distill-Llama-70B and DeepSeek-R1-Distill-Qwen-32B, which achieve new state-of-the-art results for dense models across various reasoning benchmarks. If at least one model generates a confidence score of 2 for a thought, we regard it as a correct thought.

We evaluate the accuracy of our assessment approach using responses generated by Qwen-32BPreview for 90 instances from the AIME 2022, 2023, and 2024 test sets. We utilize the final thought in each response as the test example and its correctness as the ground-truth label. To ensure a fair comparison, we randomly streamline correct thoughts to match the average length of incorrect thoughts. Ultimately, we have 35 correct thoughts with an average length of 278.1 tokens and 55 incorrect thoughts with an average length of 278.3 tokens. Our assessment approach achieves accuracies of 82.9% for correct examples and 81.8% for incorrect examples, demonstrating its effectiveness.

Early-Stage Thoughts Are Correct but Abandoned in Incorrect Responses Figure 5 depicts the ratio of correct thoughts at each index in incorrect responses on the three challenging test sets. The analysis highlights a critical insight into the phenomenon of underthinking. Specifically, a notable proportion of initial thoughts across various models were correct but were not pursued to completion. This tendency to abruptly shift away from these promising thoughts indicates an inadequate depth of reasoning, where potentially correct solutions are prematurely abandoned before being thoroughly explored. This observation suggests a need for enhancing the models’ ability to persistently explore a specific line of reasoning deeply and accurately before opting to switch to alternative thought processes.

Most Incorrect Responses Contain Correct Thoughts Figure 6 illustrates the distribution of thought correctness ratios in incorrect responses from various models. We observe that over 70% of incorrect responses contain at least one correct thought. Furthermore, in more than 50% of these responses, over 10% of the thoughts are correct. Combined with observations from Figure 5, this suggests that while o1-like models can initiate correct reasoning pathways, they may struggle to continue these pathways to reach the correct conclusion. This highlights the importance of encouraging models to maintain and expand their initial correct thoughts to synthesize them into accurate final answers. These insights lead us to propose an underthinking metric based on the presence of the first correct thought in the subsequent section.

QwQ-32B-Preview

DS-R1-Preview

DS-R1-671B

11.2

12

- 0
- 1
- 2
- 3
- 4
- 5

12

- 0
- 1
- 2
- 3
- 4
- 5

12

- 0
- 1
- 2
- 3
- 4
- 5

|1.1<br><br>3.7|Thoug Token<br><br>1.5<br><br>4.0|hts s<br><br>1.9<br><br>4.6|2.3<br><br>6.6|4.0|
|---|---|---|---|---|

|1.1<br><br>3.1|houg oken<br><br>1.2<br><br>3.0|hts s<br><br>1.6<br><br>2.9|2.3<br><br>5.6|3.5<br><br>9.4|
|---|---|---|---|---|

|1.5<br><br>2.5|Thoug Token<br><br>1.5<br><br>2.0|hts s<br><br>2.3<br><br>3.3|3.0<br><br>4.6|4.7<br><br>6.1|
|---|---|---|---|---|

10

10

10

#Tokens(K)

#Tokens(K)

#Tokens(K)

8

8

8

#Thoughts

#Thoughts

#Thoughts

6

6

6

4

4

4

2

2

2

0

0

0

1 2 3 4 5

1 2 3 4 5

1 2 3 4 5

Level (Difficulty↑)

Level (Difficulty↑)

Level (Difficulty↑)

[Figure 7]

On the Underthinking of o1-Like LLMs

50%

100%

47.1%

|25.9% 25.9% 24.9% 21.4%<br><br>17.9%<br><br>36.7%<br><br>26.7%<br><br>30.0% 30.0% 26.7%<br><br>13.6%<br><br>19.4%<br><br>22.6% 18.8%<br><br>12.5%<br><br>25.0% 18.8% 18.8% 18.8%<br><br>21.9%<br><br>17.6% 18.3%<br><br>23.5%<br><br>29.4%<br><br>17.6%<br><br>23.5% 17.6%<br><br>36.8% 31.6%<br><br>DeepSeek-R1-671B<br><br>DeepSeek-R1-Preview<br><br>QwQ-32B-Preview|
|---|

|10.8% 10.8%<br><br>16.2%<br><br>24.3%<br><br>32.4%<br><br>48.6%<br><br>59.5%<br><br>67.6%<br><br>78.4%<br><br>0.0%<br><br>5.3%<br><br>13.2% 10.5%<br><br>21.1%<br><br>36.8%<br><br>47.4%<br><br>63.2%<br><br>71.1%<br><br>84.2%<br><br>5.6% 5.6% 5.6%<br><br>11.1%<br><br>22.2%<br><br>38.9%<br><br>55.6%<br><br>72.2%<br><br>DeepSeek-R1-671B<br><br>DeepSeek-R1-Preview<br><br>QwQ-32B-Preview|
|---|

84

40%

80%

CorrectnessRatio

78

31

72

Distribution

30%

60%

26

%

%

20%

40%

21

10%

20%

%

0%

0%

>0 ≥0.1 ≥0.2 ≥0.3 ≥0.4 ≥0.5 ≥0.6 ≥0.7 ≥0.8 ≥0.92.7%

0.0% 0.0%

1 2 3 4 5 6 7 8 9 ≥10

Index of Thought

Ratio of Correct Thoughts

Figure 6: The distribution of thought correctness ratio in incorrect responses. More advanced models contain fewer correct thoughts.

##### 2.3 Empirical Underthinking Results

In this section, we propose a metric for empirically assessing underthinking issues based on token efficiency, complementing the widely used accuracy metric.

Underthinking Metric Intuitively, if a model generates a correct thought at an early stage and then switches to other thoughts without reaching a correct answer, the tokens generated thereafter do not contribute to reaching a correct solution and are considered inefficient due to underthinking. The underthinking score, denoted as ξUT, is defined as:

Tˆi Ti

N

1 N

### ∑

1 −

(1)

ξUT =

i=1

Here, N represents the number of instances in a given test set where the evaluated model generates incorrect responses. Ti is the total number of tokens in the i-th incorrect response, and Tˆi is the number of tokens from the beginning of that response up to and including the first correct thought. If there is no correct thought in the i-th response, Tˆi = Ti, indicating that the model lacks an understanding of this problem, leading it to explore diverse but ineffective approaches. Therefore, it cannot be considered underthinking. Consider Figure 2 as an example: the first reasoning thought can reach a correct answer if fully explored, with Tˆ = 411. Consequently, ξUT = 1 − 7681411 = 0.946, which can be considered extremely inefficient, reflecting a high underthinking score.

The metric ξUT quantifies the extent of underthinking by measuring the token efficiency in generating effective content within an incorrect response. Specifically:

- • A lower value of ξUT indicates higher token efficiency, meaning that a greater proportion of tokens in incorrect responses contribute towards reaching a correct thought before switching to another thought. This suggests that the model is more efficient in its token utilization even when it fails to provide a correct answer.
- • Conversely, a higher value of ξUT signifies lower token efficiency, indicating that a larger proportion of tokens do not contribute effectively towards generating a correct thought. This reflects greater underthinking, where the model may generate redundant or irrelevant tokens by frequently switching thoughts.

Empirical Results Table 1 provides insights into model performance across challenging test sets, evaluating both accuracy and underthinking (UT) scores. Clearly, all o1-like LLMs suffer from significant underthinking issues, although there are considerable differences across models and test sets. The results reveals that the relationship between model accuracy and underthinking varies

Table 1: Underthinking scores on challenging testsets. Models Accuracy (↑) UT Score (↓) MATH500-Hard (Level 5) QwQ-32B-Preview 84.3 58.2 DeepSeek-R1-Preview 83.6 61.5 DeepSeek-R1-671B 92.5 65.4 GPQA Diamond QwQ-32B-Preview 59.6 48.3

- DeepSeek-R1-671B 73.2 58.8

AIME24 QwQ-32B-Preview 46.7 65.0 DeepSeek-R1-Preview 46.7 75.7

- DeepSeek-R1-671B 73.3 37.0

across different datasets. On the MATH500-Hard and GPQA Diamond datasets, higher accuracy achieved by the superior DeepSeek-R1-671B model is accompanied by higher UT Scores, indicating more underthinking in incorrect responses. This suggests that while the model is more capable overall, it may produce longer but less effective reasoning when uncertain, possibly due to exploring multiple incorrect reasoning paths without efficiently converging on the correct solution. Conversely, on the AIME2024 test set, the DeepSeek-R1-671B model not only attains higher accuracy but also exhibits a lower UT score, reflecting less underthinking and greater token efficiency. This implies that the model’s reasoning remains focused and effective even when it does not arrive at the correct answer, perhaps due to better alignment with the problem types and reasoning processes required by the AIME2024 task.

These findings illustrate that underthinking behavior is sensitive to the nature of the dataset and the tasks involved. The larger model’s superior capabilities do not uniformly translate to less underthinking across all tasks. In some cases, increased model capacity leads to more elaborate but inefficient reasoning in incorrect responses, while in others, it enhances both accuracy and reasoning efficiency. Understanding the underthinking phenomenon is crucial for developing models that not only provide correct answers but also exhibit effective reasoning processes.

#### 3 Mitigating Underthinking Issues

In this section, we propose a lightweight mechanism that mitigates underthinking issues without requiring any model fine-tuning. Our experimental results using the QwQ-32B-Preview model demonstrate the effectiveness of this approach across all challenging test sets.

##### 3.1 Decoding with Thought Switching Penalty

Aforementioned findings show that o1-like LLMs prioritize exploring many solutions over deeply investigating one. Inspired by the success of the coverage penalty in neural machine translation (Tu et al., 2016; Wu et al., 2016), we propose a novel decoding algorithm with a thought switching penalty to encourage the model to explore potential thoughts more thoroughly before moving on to new ones.

Standard Decoding In standard decoding, the probability of each token v at position t is computed using the softmax function over the logits zt ∈ R|V| (where |V| is the vocabulary size) in the output layer:

exp (zt,v)

P(xt = v|x<t) =

∑v′∈V exp zt,v′

Table 2: Accuracy on AIME2022-23 with respect to different values of α and β.

|Pass@1 Accuracy| |α 3 5 10 20<br><br>|
|---|---|---|
| | | |
|β|300 400 500 600 700|35.2 37.0 39.0 39.4 39.3 37.1 37.1 38.4<br><br>38.5 38.7 39.1 39.2<br>39.8 39.4 38.0 38.0 37.1 39.4 39.0 38.3<br>|

where zt,v ∈ zt is the logit (unnormalized score) for token v. By repeating this step for each position in the sequence, the model generates sequences of tokens, computing probabilities for each possible continuation.

Thought Switching Penalty (TIP) To encourage the model to delve deeper into current thoughts before switching, we introduce a penalty on tokens that are associated with thought transitions. Let V ⊂ V be the set of tokens associated with thought switching (e.g., “alternatively”). We modify the logits as follows:

zt,v − α, if v ∈ V and t < Ψ + β zt,v, otherwise

zˆt,v =

(2)

where

- • α ≥ 0 (Penalty Strength) is a parameter controlling the strength of the penalty applied to thoughtswitching tokens. A larger α results in a greater reduction of the logits for these tokens, making them less likely to be chosen.
- • β ≥ 0 (Penalty Duration) specifies the number of positions from the start of a thought at Ψ, during which the penalty is active. A larger β extends the penalty over more positions, further discouraging early thought switching.

When α = 0 or β = 0 , the penalty is effectively disabled, and the decoding process reduces to the standard decoding algorithm. The adjusted logits zˆt,v reduce the probability of generating thought-switching tokens within a specified window, encouraging the model to continue expanding on the current thought before moving on. The new probability distribution becomes

exp (zˆt,v)

Pˆ(xt = v | x<t) =

∑v′∈V exp z ˆt,v′

##### 3.2 Experimental Results

To ensure robust conclusions, we report Pass@1 results computed from 32 samples per instance. We calculate the weighted underthinking score for each instance over its 32 samples:

1 32

ξwUT =

32

### ∑

ξUT(si) (3)

i=1

where si is the i-th sample of the instance, and ξUT(si) = 0 when si is correct.

By adjusting α and β, we can control the model’s behavior to achieve the desired level of thought exploration. We performed a grid search with α values in [3,5,10,20,30] and β values in [300,400,500,600,700] using a development set that included the AIME 2022 and 2023 test sets.

- Table 2 lists the impact of varying the penalty strength α and penalty duration β on the model’s accuracy. We observe that increasing the penalty strength α generally leads to an improvement in

- Table 3: Pass@k performance of the proposed TIP method. For each problem, we generated 32 responses with a temperature of 0.7 and a top p value of 0.95. Since it is infeasible to calculate the Pass@k Underthinking Score, we instead report the average score and standard deviation from the 32 generated samples. We also report the average number of thought-switching tokens (Vˆ in Equation 2) and the average interval between them in the generated samples.

Accuracy (↑) Switching Tokens Weighted Pass@1 Pass@4 Pass@8 Pass@16 Number Interval UT Score (↓)

Models

MATH500-Hard (Level 5) QwQ-32B-Preview 83.1 92.4 94.4 95.8 12.6 445.6 11.7±20.5

+ TIP 83.7 93.2 95.3 96.4 5.7 517.6 11.0±19.5

GPQA Diamond QwQ-32B-Preview 57.6 78.5 85.3 90.3 21.1 356.8 25.1±23.9

+ TIP 59.1 78.9 85.8 91.2 7.3 432.5 23.2±23.2

AIME2024 QwQ-32B-Preview 38.3 53.7 58.5 62.7 16.1 459.7 40.6±28.4

+ TIP 44.1 61.6 68.3 74.0 13.9 515.7 35.8±27.8 R1-Distill-Qwen-32B 61.4 75.9 79.1 81.7 8.2 819.5 19.6±20.6

+ TIP 64.1 79.0 81.7 83.0 4.5 1018.0 17.7±20.6

DeepSeek-R1 73.8 86.2 88.8 89.8 13.8 580.1 14.6±19.1 + PROMPT 72.4 84.9 88.2 89.8 12.0 520.1 14.2±18.4 + TIP 74.8 86.4 88.8 89.8 5.7 941.6 13.0±18.0

accuracy up to a certain threshold, after which the benefits plateau or even diminish. Adjusting the penalty duration β also significantly affects performance: At a lower penalty strength (α = 3), increasing β from 300 to 600 results in accuracy gains from 35.2% to 39.8%, the highest observed accuracy in our experiment. Conversely, at higher penalty strengths (α = 20), extending β beyond 300 leads to a decrease in accuracy, indicating that too long a penalty duration can hinder performance when combined with a strong penalty. We selected α = 3 and β = 600 for our subsequent experiments.

Standard Decoding Table 3 lists the results of our approach in the three challenging test sets. Clearly, our approach consistently improves accuracy over the vanilla QwQ-32B-Preview in all cases by mitigating the underthinking issues. These consistent improvements across diverse and challenging datasets validate the effectiveness of the TIP approach in mitigating the underthinking issue identified in o1-like LLMs. By penalizing thought switches during decoding, TIP encourages the model to elaborate more thoroughly on each reasoning thought before considering alternative ones. This mechanism aligns with the human problem-solving process, where a focused and indepth exploration of a particular approach often leads to correct solutions, especially in complex mathematical problem-solving contexts.

To understand the impact of the TIP method on the models’ reasoning processes, we analyzed the average number of thought-switching tokens and the intervals between them. Across all test sets, the TIP method reduces the number of thought-switching tokens and increases the average interval between them. This indicates that the models are committing more deeply to individual lines of reasoning before considering alternatives, aligning with our goal of mitigating underthinking. For example, on the AIME2024 testset, the number of thought-switching tokens for DeepSeekR1 dramatically decreases from 13.8 to 5.7 when using the TIP method, and the average interval increases from 580.1 to 941.6 tokens. This shift suggests that the model is exploring each thought more thoroughly, reducing premature transitions that could lead to underthinking issues.

Prompting Some researchers hypothesize that prompt engineering can foster “thought persistence”

by directing models to maintain a consistent line of reasoning. To investigate this hypothesis, we use a prompt that encourages the model to fully develop each idea without abandoning it prematurely:

|<context><br><br>You are an expert math-solving assistant who prioritizes clear, concise solutions. You solve problems in a single thought process, ensuring accuracy and efficiency. You seek clarification when needed and respect user preferences even if they are unconventional.<br><br></context> <solving rules><br><br>- Try to complete every idea you think of and don’t give up halfway<br>- Don’t skip steps<br>- Display solution process clearly<br>- Ask for clarification on ambiguity </solving rules> <format rules><br><br>- Use equations and explanations for clarity<br>- Keep responses brief but complete<br>- Provide step-by-step reasoning if needed </format rules> PROBLEM: {problem}<br><br><br>OUTPUT: Following above rules to get the correct answer for PROBLEM. Focus on clear, concise solutions while maintaining a helpful, accurate style.|
|---|

Although prompt engineering (DeepSeek-R1 + PROMPT) provides certain guidance, Table 3 shows only modest changes in switching tokens and overall accuracy compared with TIP. This finding indicates that inherent generation patterns can still lead to premature reasoning transitions, underscoring the need for a dedicated mechanism such as TIP. The results also point to substantial possibilities for more sophisticated prompt engineering that can better guide DeepSeek-R1 in following instructions. Moreover, combining advanced prompt engineering with decoding approaches (e.g., TIP) could further enhance “thought persistence”. In this approach, prompts offer high-level guidance, whereas decoding penalties reinforce consistent reasoning at the token level. We aim to explore how these methods can work together to deepen reasoning in LLMs.

Best-of-N Sampling To further evaluate the effectiveness of our TIP approach, we applied it in conjunction with best-of-N sampling methods, specifically Self-Consistency (Wang et al., 2023) and Laconic Decoding 1:

- • Self-Consistency: This algorithm first samples N reasoning paths and then selects the most consistent answer by marginalizing over the sampled reasoning paths.
- • Laconic Decoding: Raoof and Dimakis independently observed that incorrect answers tend to be longer while correct answers are shorter for long reasoning models. Based on this observation, they propose a simple idea called Laconic Decoding: run the model N times (in parallel) and select the answer with the fewest number of tokens.

- Table 4 presents the results of these methods applied to various models, including QwQ-32B-Preview, R1-Distill-Qwen-32B, and DeepSeek-R1, both with and without the TIP enhancement. For each setting, we conducted 10,000 trials by randomly sampling from the 32 generated responses (as detailed in Table 3). We report both the average accuracy and the weighted underthinking score. The “Averaged” rows represent the average performance over K samples without any selection

###### 1https://x.com/AlexGDimakis/status/1885447830120362099

Table 4: Results of the best-of-N sampling methods applied to different models enhanced with our TIP approach on AIME2024. For each setting, we conducted 10,000 trials by randomly sampling from the 32 samples in Table 3 and reported the average results. “Averaged” denotes the average performance over N samples.

4 Samples 8 Samples 16 Samples Acc.(↑) UT (↓) Acc.(↑) UT (↓) Acc.(↑) UT (↓)

Models

QwQ (Averaged) 38.4 40.5 38.3 40.6 38.3 40.6

+ TIP (Averaged) 44.1 35.8 44.0 35.9 44.0 35.9 QwQ + Self-Consistency 43.7 35.4 44.3 34.0 44.6 31.9

+ TIP 51.4 26.6 53.4 24.3 53.9 24.1 QwQ + Laconic Decoding 47.0 28.2 47.0 25.5 45.1 24.0 + TIP 50.3 26.7 51.6 23.3 50.9 20.8

R1-Distill-Qwen (Averaged) 61.4 19.2 61.3 19.2 61.3 19.1

+ TIP (Averaged) 64.1 17.8 64.0 17.7 64.1 17.7 R1-Distill-Qwen + Self-Consistency 67.0 13.4 67.8 11.4 68.9 8.9

+ TIP 69.9 12.5 71.4 11.0 72.3 9.1 R1-Distill-Qwen + Laconic Decoding 71.1 11.3 74.4 8.7 77.5 7.4 + TIP 75.4 9.8 78.0 7.3 77.9 6.5

R1 (Averaged) 73.9 14.5 73.7 14.6 73.8 14.5

+ TIP (Averaged) 74.8 13.0 74.8 12.9 74.8 13.0 R1 + Self-Consistency 79.3 10.1 79.8 9.8 79.7 9.5

+ TIP 81.3 7.5 82.2 6.4 82.1 5.8 R1 + Laconic Decoding 81.4 8.1 82.6 6.2 83.2 5.1 + TIP 83.1 7.4 83.8 6.6 83.3 6.7

strategy, while the “Oracle” rows represent the best possible outcome, assuming that whenever a correct answer exists among the sampled outputs, it is always selected.

Our findings indicate that incorporating the TIP approach consistently improves performance across all models and sampling methods. Specifically, when combined with Self-Consistency, the TIP method enhances the Pass@4 accuracy of QwQ-32B-Preview from 43.7% to 51.4% on the AIME2024 dataset, representing a significant gain. Similarly, the Underthinking Score decreases, indicating that the models are engaging in more thorough reasoning processes.

Notably, with Laconic Decoding, the combination with TIP yields substantial improvements. For instance, the Pass@4 accuracy of R1-Distill-Qwen-32B increases from 74.4% to 78.0%, while the Underthinking Score decreases from 8.7 to 7.3. This suggests that the TIP approach complements existing decoding strategies by further encouraging comprehensive reasoning.

Experimental Conclusion Our experiments demonstrate that the TIP approach effectively mitigates the underthinking issue in O1-like LLMs by penalizing unnecessary thought switches during decoding. Although the hyperparameters are tuned on the AIME 2022 and 2023 test sets using the QwQ-32B-Preview model, the consistent improvements observed across various test sets and models with the same hyperparameters validate the generalizability of the method.

Importantly, the TIP method enhances performance without requiring additional training or modifications to the model architecture. Operating at the decoding level, it serves as a practical solution that can be readily applied to existing models. This ease of integration, combined with the observed performance gains, highlights the potential of the TIP approach for enhancing the reasoning capabilities of large language models in complex problem-solving tasks.

Our findings also suggest that the TIP method synergizes well with best-of-N sampling strategies. When combined with Self-Consistency and Laconic Decoding, the TIP approach leads to further

improvements in accuracy and reductions in underthinking scores. This indicates that encouraging more thorough exploration of individual reasoning paths complements the diversity introduced by sampling methods.

Overall, the TIP approach represents a significant step toward addressing the underthinking problem in long reasoning models. By encouraging deeper reasoning and reducing hasty thought-switching, it enhances the models’ ability to solve complex tasks that require sustained cognitive effort.

#### 4 Related Work

##### 4.1 Scaling Test-Time Compute

The advent of deep reasoning models, epitomized by OpenAI’s o1, has sparked significant interest in scaling test-time compute to enhance models’ abilities to solve complex problems. Scaling test-time compute often involves two major strategies. The first is expanding the search space, which aims to broaden the scope of candidate solutions explored during decoding to ensure better final outcomes. Techniques in this category include self-consistency (Wang et al., 2023), where multiple answers are generated with a majority voting mechanism to select the final answer. Other methods include best-of-n decoding and minimum Bayes risk decoding (Lightman et al., 2024; Li et al., 2023; Khanov et al., 2024; Heineman et al., 2024; Wu et al., 2024).

The second direction, and arguably more transformative, focuses on human-like deep thinking. Starting with Chain-of-Thought Wei et al. (2022), people realized that models can mimic the human thought process for reasoning Cesista (2024); Pfau et al. (2024). Recent efforts such as QwQ (Qwen, 2024), DeepSeek-R1 (DeepSeek, 2025) and Kimi-1.5 (Kimi, 2025), which aim to replicate OpenAI’s o1, leverage reinforcement learning (RL) to endow models with advanced reasoning capabilities. Under large-scale RL training, these models exhibit emergent human-like thinking abilities characterized by deep, extended, and strategic reasoning. This allows them to explore diverse strategies, reflect on their decisions, revisit previous steps, and verify their conclusions. Such human-like thinking markedly improves accuracy, especially on complex reasoning tasks.

Efficient Thinking Given that o1-like models aim to mimic human thought processes, the efficiency of their reasoning is critical to their performance on challenging problems. Just as human thinking can occasionally be inefficient, models may face similar issues. For instance, Chen et al. (2024) study the problem of overthinking in o1-like LLMs, where models waste substantial computational resources revisiting trivial or self-evident paths, leading to inefficiency in simple problems. Conversely, our focus lies on the underexplored problem of underthinking, which occurs when a model fails to deeply explore promising paths, instead frequently switching strategies prematurely, resulting in computational waste. This inefficiency becomes especially pronounced when tackling challenging problems. We assert that truly intelligent systems must learn to adaptively allocate their computational resources, concentrating on paths that are both promising and challenging. During evaluations on the NPR Sunday Puzzle Challenge, Anderson et al. (2025) note that o1-like LLMs often produce “I give up” mid-reasoning, prematurely ending their thought process and leading to incorrect outcomes. We view this as a form of underthinking.

To promote efficient reasoning, our subsequent work has also explored methods that limit unnecessary computation. Building on the observation that incorrect reasoning often leads to longer responses, Raoof and Dimakis propose Laconic decoding,2 which employs a shortest-of-n strategy to boost accuracy. Muennighoff et al. (2025) control test-time compute by cutting off the model’s reasoning earlier or by repeatedly adding “Wait” to extend the reasoning when the model tries to end prematurely. Arora & Zanette (2025) propose a modified reinforcement learning objective encouraging models to produce correct answers with relatively short CoT, thereby minimizing inference costs while preserving accuracy. Similarly, Wu et al. (2025) introduce Length-filtered Vote,

2https://x.com/AlexGDimakis/status/1885447830120362099

which adaptively identifies the best CoT length for majority voting, excluding CoTs that are either overly short or unnecessarily long.

##### 4.2 Manipulating Decoding Penalties

The role of penalty mechanisms in Natural Language Processing decoding has garnered significant attention. Traditional decoding methods, such as greedy search and beam search, focus primarily on maximizing the likelihood of generated sequences without considering the broader implications of the outputs. However, researchers have identified various shortcomings in these approaches, leading to the exploration of penalty mechanisms to enhance the quality of generated text.

Length normalization is a widely used strategy to adjust decoding penalties. Jean et al. (2015); Koehn & Knowles (2017); Tu et al. (2017); Murray & Chiang (2018) highlighted that length normalization and length penalties can prevent models from generating overly verbose or excessively brief translations, leading to improved fluency and adequacy. In addition, Tu et al. (2016) introduced coverage penalties in neural machine translation to mitigate the problems of “over-translation” and “under-translation” by integrating a coverage metric that penalizes repeated attention to tokens. Along this direction, Wu et al. (2016) proposed a coverage penalty in decoding to encourage the generation of an output that is most likely to cover all the words in the source sentence. See et al. (2017) incorporated the concept of coverage into the summarization task by modeling the coverage content in summarization outputs.

In this paper, we adjust decoding penalties to address the problem of underthinking. Our approach encourages the model to maintain its original line of reasoning and engage in deeper thought processes, avoiding frequent shifts in strategy and superficial reasoning patterns. To the best of our knowledge, we are the first to investigate the effectiveness of decoding penalties in mitigating the underthinking issue.

#### 5 Conclusion

In this work, we investigated underthinking in o1-like LLMs, identifying it as a significant factor limiting their performance on challenging reasoning tasks. Through comprehensive analysis, we observed that these models frequently abandon promising reasoning paths prematurely, leading to inefficient problem-solving and lower accuracy. We introduced a novel metric to quantify underthinking by assessing token efficiency in incorrect responses. To mitigate this issue, we proposed a decoding strategy with a thought switching penalty (TIP), which encourages models to thoroughly explore each reasoning thought before considering alternatives. Our empirical results demonstrate that TIP effectively reduces underthinking and enhances accuracy across difficult mathematical and scientific problem sets without necessitating additional model training.

This work contributes to a deeper understanding of reasoning processes in o1-like LLMs and provides a practical approach to align their problem-solving capabilities. By addressing underthinking, we aim to bring models closer to human-like deep thinking, efficiently utilizing computational resources to achieve higher accuracy on complex tasks. Future directions include exploring adaptive mechanisms within models to self-regulate thought transitions and further improving reasoning efficiency in o1-like LLMs.

#### References

Carolyn Jane Anderson, Joydeep Biswas, Aleksander Boruch-Gruszecki, Federico Cassano, Molly Q Feldman, Arjun Guha, Francesca Lucchetti, and Zixuan Wu. Phd knowledge not required: A reasoning challenge for large language models. arXiv preprint arXiv:2502.01584, 2025.

Daman Arora and Andrea Zanette. Training language models to reason efficiently. arXiv preprint arXiv:2502.04463, 2025.

Franz Louis Cesista. Multimodal structured generation: Cvpr’s 2nd mmfm challenge technical report. arXiv preprint arXiv:2406.11403, 2024.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do not think that much for 2+3=? on the overthinking of o1-like llms, 2024. URL https://arxiv.org/ abs/2412.21187.

DeepSeek. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. 2025. URL https://api.semanticscholar.org/CorpusID:275789950.

David Heineman, Yao Dou, and Wei Xu. Improving minimum bayes risk decoding with multiprompt. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 22525–22545, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS, 2021.

S´ebastien Jean, Orhan Firat, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. Montreal neural machine translation systems for wmt’15. In Proceedings of the tenth workshop on statistical machine translation, pp. 134–140, 2015.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. Args: Alignment as reward-guided search. In

The Twelfth International Conference on Learning Representations, 2024. Kimi. Kimi k1.5: Scaling reinforcement learning with llms. 2025. Philipp Koehn and Rebecca Knowles. Six challenges for neural machine translation. In Proceedings

of the First Workshop on Neural Machine Translation, pp. 28–39, 2017.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315–5333, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.291. URL https://aclanthology.

###### org/2023.acl-long.291.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=v8L0pN6EOi.

MAA Committees. Aime problems and solutions. https://artofproblemsolving.com/wiki/

###### index.php/AIME_Problems_and_Solutions.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Cand`es, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Kenton Murray and David Chiang. Correcting length bias in neural machine translation. In Proceedings of the Third Conference on Machine Translation: Research Papers, pp. 212–223, 2018.

OpenAI. Learning to reason with llms. https://openai.com/index/

###### learning-to-reason-with-llms, 2024.

Jacob Pfau, William Merrill, and Samuel R Bowman. Let’s think dot by dot: Hidden computation in transformer language models. arXiv preprint arXiv:2404.15758, 2024.

Qwen. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https:

###### //qwenlm.github.io/blog/qwq-32b-preview/.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

Abigail See, Peter J Liu, and Christopher D Manning. Get to the point: Summarization with pointergenerator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1073–1083, 2017.

Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. Modeling coverage for neural machine translation. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 76–85, 2016.

Zhaopeng Tu, Yang Liu, Lifeng Shang, Xiaohua Liu, and Hang Li. Neural machine translation with reconstruction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Ian Wu, Patrick Fernandes, Amanda Bertsch, Seungone Kim, Sina Pakazad, and Graham Neubig. Better instruction-following through minimum bayes risk. arXiv preprint arXiv:2410.02902, 2024.

Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, and Jeffrey Dean. Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation. arXiv, 2016.

Yuyang Wu, Yifei Wang, Tianqi Du, Stefanie Jegelka, and Yisen Wang. When more is less: Understanding chain-of-thought length in llms. arXiv preprint arXiv:2502.07266, 2025.

