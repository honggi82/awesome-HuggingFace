# arXiv:2507.06138v1[cs.CL]8Jul2025

## Coding Triangle: How Does Large Language Model Understand Code?

Taolin Zhang1,2,∗, Zihan Ma 1,3,∗, Maosong Cao1, Junnan Liu1, Songyang Zhang1,†,‡, Kai Chen1,† 1Shanghai AI Laboratory 2Tsinghua University 3Xi’an Jiaotong University

#### Abstract

Large language models (LLMs) have achieved remarkable progress in code generation, yet their true programming competence remains underexplored. We introduce the Code Triangle framework, which systematically evaluates LLMs across three fundamental dimensions: editorial analysis, code implementation, and test case generation. Through extensive experiments on competitive programming benchmarks, we reveal that while LLMs can form a self-consistent system across these dimensions, their solutions often lack the diversity and robustness of human programmers. We identify a significant distribution shift between model cognition and human expertise, with model errors tending to cluster due to training data biases and limited reasoning transfer. Our study demonstrates that incorporating human-generated editorials, solutions, and diverse test cases, as well as leveraging model mixtures, can substantially enhance both the performance and robustness of LLMs. Furthermore, we reveal both the consistency and inconsistency in the cognition of LLMs that may facilitate self-reflection and self-improvement, providing a potential direction for developing more powerful coding models. 1

#### 1 Introduction

Recent advances in large language models (LLMs) with rapid development in model design and data scaling [6, 1, 25, 29, 10, 16, 5, 31, 27, 2, 23] have driven remarkable progress on code generation benchmarks [9, 4, 7, 15, 20, 18, 13, 30]. For example, DeepSeek-V3 [19] achieves a score of 65.2 on HumanEval [12, 9], Qwen3 attains 65.7 on LiveCodeBench [27, 15], and o3 reaches a Codeforces Elo rating of 2724 [25, 11], demonstrating impressive coding abilities of modern foundation models.

With rapid advancements in coding capabilities of large language models, a growing concern is that current coding benchmarks fail to accurately and comprehensively evaluate the coding ability of LLMs. In this work, we address a foundational question: How should the coding capability of LLM be defined? When human developers solve coding problems, they typically follow a structured pipeline involving problem analysis, designing a preliminary solution strategy, code implementation, manual testing, and iterative refinement until the solution passes all test cases.

Motivated by this, we investigate the coding capabilities of LLMs through three interconnected dimensions: Editorial, Code, and Cases. We develop such a three-dimensional analysis framework called Coding Triangle to study the coding behaviors of LLMs. Our goal is to understand how LLMs fundamentally interpret coding problems within each dimension and to explore the interactions across them. For example: (a) Does performance across dimensions exhibit consistency?(b) Does a model’s code generation benefit from its natural language problem breakdowns?(c) Does the generated code reliably pass self-generated test cases?(d) To what extent do these test cases adequately reflect the reasoning outlined in the editorial?

1This work is done when Taolin Zhang and Zihan Ma are on internship at Shanghai AI Laboratory, * means equal contribution, † means corresponding author, ‡ means project lead.

Preprint. Under review.

###### Problem Statement Code

Human Generated Content LLM Generated Content

#include <iostream> using namespace std;

There are N people labeled 1 to N, who have played several one-onone games without draws. ……………………………….. Determine the final score of person N if the final score of person (1≤i≤N−1) is Ai.

[Figure 1]

int main() { int N; ……. return 0;

}

Code

###### Editorial Cases

[Figure 2]

[Figure 3]

Input

Key Observations Conservation of Points : ……………………………….. Therefore: A_N = -S

6 10 20 30 40 50

Ouput

###### Time Complexity

•Sum Calculation : The sum of N1 elements takes O(N) time.

Editorial Cases

-150

Figure 1: The framework of Coding Triangle. Editorial, code, and cases form the three fundamental vertices of the triangle, with each vertex can be sampled from either human solutions or model predictions. These vertices are interconnected, influencing one another, and their relationships form the six directed edges of the triangle, representing the mutual interactions between Editorial, Code, and Cases.

Based on our Coding Triangle framework, we conduct extensive experiments on 200 problems collected from AtCoder and evaluate various LLMs including general models, coding model and reasoning model. By analyzing their capabilities and interactions among different dimensions, we gain deeper insights into how LLMs truly comprehend and tackle coding tasks with several surprising and interesting findings.

Our results demonstrate prevalent self-consistency across three dimensions in various LLMs. This self-consistency often confines LLM reasoning, leading it to converge on narrow patterns and repeat similar errors, particularly with corner cases or implementation details. Consequently, a significant distribution shift emerges between LLM predictions and human submissions. Notably, ensembling multiple models can effectively mitigate these cognitive biases, enhancing performance diversity and robustness. Conversely, we observe self-inconsistency across these dimensions, indicating that their respective capabilities are not always fully aligned. An LLM might, for example, accurately analyze its own failed solutions to pinpoint root causes and effectively use self-generated test cases to differentiate correct from incorrect solutions; yet, these individual strengths may not be cohesively integrated, exemplifying the aforementioned misalignment. These findings suggest potential for self-reflection and self-improvement by iteratively aligning these dimensions.

To summarize, this paper makes the following contributions.

- • We propose a framework called Coding Triangle to systematically examine the internal knowledge of LLMs in programming, enabling a comprehensive evaluation of their coding abilities.
- • We investigate the distribution shift between the LLM predictions and actual solutions from human, and find that incorporating human information can substantially improve performance.
- • By analyzing the self-consistency and self-inconsistency inside LLMs, we observe that their strengths and weaknesses vary across the three dimensions in Coding Triangle, demonstrating the advantages of model mixtures and the potential for self-reflection and self-improvement.

#### 2 Overview

In this section, we first introduce the three fundamental vertices of Coding Triangle and define how to evaluate the ability of LLM across these three dimensions. We will proceed with the analysis and explore the distribution shift compared to the distribution of human coding in Section 3, and discuss the interaction among these three dimensions in Section 4. Our experiments cover general models, coder models and reasoning models2 and we utilize AtCoder3 as the evaluation problem sets.

- 2We denote the DeepSeek-V3 [19], Qwen2.5-72B-Instruct[32], Qwen2.5-Coder-32B-Instruct [14] and QwQ-32B [28], as DS-V3, 72B, Coder and QWQ for short, respectively. All the results are obtained with Nvidia-A800.
- 3Problems A-F from AtCoder Beginner Contest (ABC) 175-374.

##### 2.1 Coding Triangle

In this study, we systematically examine the comprehensive understanding of LLMs when addressing competitive programming challenges. Motivated by how human solve the problem steps by steps through analyzing, coding and manually testing, we formally decompose coding ability into three interconnected perspectives, and establish the overall framework of Coding Triangle:

- • Editorial captures how LLM interprets and analyzes the the problem in natural language, providing the most accessible explanation for human readers.
- • Code reflects the programming logic and ability of algorithm implementation, serving as the machine-executable counterpart to the human-readable editorial.
- • Cases indicate the depth of understanding in terms of validation criteria, including edge scenarios and boundary conditions.

Intuitively, these three dimensions create a comprehensive system that captures all aspects of a coding problem, from interpretation to execution and validation. We then introduce evaluation metrics for each dimension, allowing us to quantify its strengths and weaknesses in a structured way.

##### 2.2 Evaluation Metric

Editorial. We adopt an LLM-as-a-judge approach to evaluate the quality of model-generated editorials. With the model-generated editorial Emodel and the ground-truth editorial Egt, we require o3-mini [26] to judge them and predict a correctness score. Let N be the total number of problems, the overall editorial score Sedi is defined as

1 N

Sedi =

N

LLM(Emodel(i) ,Egt(i)). (1)

i=1

Code. During the evaluation, the model is prompted to generate a solution for each problem, which is then validated against all public ground-truth test cases. For each problem i, let Ti denote the set of public ground-truth test cases, J as the judge function and si the solution from model. With N as the total number of problems, we count the number of problems where the solution passes all test cases: Ncode = |{i | ∀t ∈ Ti, J (si,t) = Accepted}| . The code score is defined as Pass@1 accuracy:

Ncode N

. (2)

Scode =

Cases. It is observed that official cases for some problems are not sufficiently comprehensive and may fail to cover all edge scenarios, potentially leading to incorrect solutions being accepted. Therefore, we focus on those errors that can be identified by the official cases and evaluate modelgenerated cases with only incorrect solutions. Furthermore, directly generating input-output pairs [8, 21] is often insufficient, as most generated cases are incorrect and will be filtered out. Instead, we prompt the model to generate inputs and use the official solution to produce the outputs.

For each problem i, let Hiwrong denote the set of all human submissions that are incorrect, and J (h,Ci) as the judge results of submission h on the model-generated cases Ci. We consider the cases set to be correct if the judge results are consistent with the official evaluation and all wrong solutions are identified as incorrect. Let Ncase to be the number of problems for which the model-generated cases can effectively distinguish all the edge cases, we have Ncase = |{i | ∀h ∈ Hiwrong, J (h,Ci) = J (h,Ti)}| , where Ti is the set of ground-truth test cases for problem i. The case score is then defined as

Ncase N

. (3)

Scase =

#### 3 Ability Analysis and Distribution Shift

With the evaluation metrics established above, we analyze the capabilities of LLMs across the three dimensions in this section. Furthermore, we explore the distribution shift between model cognition and human submissions, and demonstrate the robustness introduced by model mixture.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

Case Code Editorial

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

Case Code Editorial

(b) Qwen2.5-Coder-32B

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

Case Code Editorial

(c) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

Case Code Editorial

(d) QwQ-32B

Figure 2: Ability analysis across different dimensions.

##### 3.1 Ability Analysis

We present the performance across the three dimensions in Figure 2 to illustrate their relationships. It can be observed that models with strong coding abilities tend to perform well across the editorial dimension and the code dimension, indicating a consistency between these two dimensions. As problem difficulty increases, both the editorial analysis and code generation abilities decline. Notably, the editorial score remains consistently higher than the code score, resulting in a performance gap that typically ranges from 0% to 20% when the model translates its problem analysis into executable code. This trend is observed across all models, including reasoning-oriented ones such as QwQ-32B, where correct reasoning does not always lead to correct solutions. Moreover, this gap is most pronounced on medium-difficulty problems, suggesting the conflict between reasoning and implementation.

In contrast, the case score exhibits a different trend and does not decrease monotonically as problem difficulty increases. And we surprisingly find that the code score can even surpass editorial and code score on the most difficult problems. Interestingly, performance drops on medium-difficulty problems, as seen in problems C and D, which we attribute to the increased complexity and abundance of edge cases at this level. For harder problems such as E and F, the main challenge shifts to algorithmic selection and advanced techniques rather than numerous edge conditions, resulting in a rebound in case accuracy. These findings suggest that the ability to generate test cases does not fully align with editorial and coding abilities, indicating an inconsistency among the three dimensions.

##### 3.2 Distribution Shift

In practice, LLMs are capable of performing problem analysis, code generation, and test case generation to form a self-consistent system. However, this system is actually limited to its own cognition, and has a significant distribution shift from human solutions. In the following part, we explore how the self-cognition causes this distribution shift. Since editorial evaluation is based on subjective assessment using LLM-as-Judge, we only investigate the distribution shift results in objective evaluations of code and test cases.

Distribution Shift on Code. To analyze the distribution shift between model and human solutions, we first construct a performance matrix Pcode ∈ Rm×n for each problem, where m is the total number of solutions (including those generated by different models and those submitted by humans), and n is the number of test cases. Each entry Pij is assigned a value of 1 if solution i passes test case j, and −1 otherwise. For error analysis, we exclude solutions that pass all test cases. Each row of Pcode represents the performance vector of a specific solution. We normalize these vectors and compute the cosine similarity between every pair of solutions:

Pi · Pk |Pi||Pk|

sim(i,k) =

, ∀i,k ∈ 1,...,m (4)

The resulting similarity matrix is visualized in Figure 3a. Our results reveal that model-generated solutions are highly similar to each other, with most pairs exhibiting a similarity score above 0.8. This suggests that model tends to make similar mistakes and generating multiple roll-outs does not prevent the inherent reasoning patterns. Notably, solutions produced by the same model show even higher internal cognition, while human-submitted solutions are much more diverse, displaying lower similarity scores and a wider variety of errors.

To further quantify the diversity of solutions, we construct a set of unique performance vectors for each problem and then compute the average size of the set, as illustrated in Figure 3c. Our findings

1.0

[Figure 4]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | |[Figure 5]| | |

Coder

0.8

72B

CodeSimilarityScore

0.6

DS-V3

0.4

QwQ

0.2

Human

0.0

Coder 72B DS-V3 QwQ Human

(a) Code Similarity Score.

1.0

[Figure 6]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | |[Figure 7]| | |

Coder

15

0.8

UniqueSetSize

72B

CaseSimilarityScore

10

0.6

DS-V3

5

0.4

QwQ

0

A B C D E F

0.2

Human

DS-V3

72B

Coder+72B

QwQ

Coder

Human Submission

0.0

Coder 72B DS-V3 QwQ Human

(b) Case Similarity Score.

(c) Code Unique Set Size.

20

UniqueSetSize

15

10

5

0

A B C D E F

DS-V3

72B

Coder+72B

QwQ

Coder

Ground Truth Cases

(d) Cases Unique Set Size.

Figure 3: Similarity analysis and unique set size for error analysis of codes and cases.

reveal that human submissions display significantly greater diversity compared to model-generated solutions, indicating a wider variety of errors in human attempts. As problem difficulty increases, the diversity of errors in both model and human solutions grows, reflecting the heightened complexity of the tasks. Notably, combining solutions from multiple models yields a broader range of unique behaviors than relying on any single model, suggesting that different models produce distinct types of errors. By integrating solutions from various models, we can identify more boundary conditions, and a comprehensive analysis of the error encountered by these models may further enhance overall robustness and performance.

Distribution Shift on Cases. We also conduct a similarity analysis on test cases and construct a performance matrix Pcase for each problem, where the rows correspond to human submissions and the columns correspond to test cases. We then use the column vectors of the matrix to represent the performance vector of each test case and calculate the cosine similarity between every pair of test cases, as shown in Figure 3b. The results show that, unlike solutions, model-generated cases and ground-truth cases do not exhibit extremely high similarity. However, we still observe that cases generated by the same model, as well as ground-truth cases, tend to be more similar within their own groups. The reason for this is that both model-generated cases and official cases are often constructed following certain fixed patterns or templates, leading to internal redundancy.

We use the unique set as an analytical tool to evaluate the diversity of generated cases as well. Notably, for easier problems, the unique set size from the models exceeds that of the ground truth cases. Upon reviewing the official cases in these situations, we find that they tend to be relatively easy and short due to the simplicity of the problems. As a result, the model-generated cases are able to capture a wider range of potential errors. For harder problems, the unique set size decrease and the test cases from models fail to recognize different error of the solutions. Surprisingly, we observe that the combination of cases set generated from different models also shows larger unique set size, indicating mixture of different models provides a better generalization of cases.

#### 4 Bridging the Edges in Coding Triangle

With the analysis framework established above, we now explore and quantify how the three dimensions of the coding triangle are interconnected, with a particular focus in the context of self-cognition inside LLMs. We also investigate how the introduction of external human knowledge, such as ground truth editorials, solutions, and test cases, affects these relationships. To this end, we systematically examine the six possible directed edges of the triangle, analyzing how providing information about one aspect (A), either from the model itself or from human sources, influences another aspect (X).

- 4.1 From Editorial to X

From Editorial to Code: Does LLM benefit from the editorial generated by itself?

To have a better understanding of the gap between the ability to analyze problems and to implement code in LLM, we inspect the performance when feeding its own editorial to generate code, and study how its own problem understanding affects code implementation. For comparison, we also include the original pass rate as well as the results when feeding ground truth editorials, as shown in Figure 4.

We observe that providing self-generated editorials does not significantly enhance coding performance. This suggests that the stage of problem analysis and code implementation are largely self-consistent. Even for reasoning-oriented models like QwQ, which perform reflection on both

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Self-Edi. w/ GT-Edi.

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Self-Edi. w/ GT-Edi.

(b) Coder-32B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Self-Edi. w/ GT-Edi.

(c) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Self-Edi. w/ GT-Edi.

(d) QwQ-32B

Figure 4: Pass@1 score with self-generated and ground truth editorials.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Edi. w/ GT-Edi.

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Edi. w/ GT-Edi.

(b) Coder-32B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Edi. w/ GT-Edi.

(c) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Edi. w/ GT-Edi.

(d) QwQ-32B

Figure 5: Case score w.r.t. ground truth editorials.

the editorial and the generated code, there is no notable performance boost. In some instances, prompting the model to write an editorial before coding can encourage CoT reasoning and slightly improve the pass rate. However, if the editorial contains flawed analysis, referencing it may actually reduce performance. In contrast, providing ground truth editorials leads to a much more significant improvement in coding performance. However, even with access to ground truth editorials, model still fail to pass difficult problems, and high success rates are not always achieved, indicating the upper bound of its ability to utilize correct analysis for code generation. Notably, we surprisingly find that DS-V3 and QwQ display very similar patterns. For easy problems, neither model gains much from ground truth editorials, suggesting that correct analysis alone is insufficient in that their internal reasoning and implementation details remain the primary bottlenecks. For harder problems, both models show comparable performance when prompted with ground truth editorials, implying that the main factor influencing the original pass rate is the their reasoning and understanding of the problem, rather than their code generation capability.

##### From Editorial to Cases: Does case generation benefit from the ground truth editorials?

We further investigate the impact of editorials on test case generation. As shown in Figure 5, even when the model is provided with detailed human-written editorials, its ability to generate high-quality test cases does not significantly improve. This suggests that the skills required for creating diverse and comprehensive test cases are distinct from those needed for code generation, and such skills are not effectively transferred through editorial. In particular, test case creation is more closely linked to the specific implementation details of code, while editorials provide only high-level abstraction and are less effective in guiding the model to cover all edge scenarios. Therefore, simply providing high quality editorials is not sufficient to improve the capability in generating robust test cases, highlighting the necessity for other strategies to address this challenge.

- 4.2 From Code to X

From Code to Editorial: Can the LLM Recognize Its Own Mistakes?

Within our framework, the editorial dimension serves as a comprehensive reflection of how understand and analyze the problem. To demonstrate the ability to self-evaluate in LLM, we provide DS-V3 with a candidate code solution and ask it to determine whether the solution is correct. This setup allows us to probe the capacity for error detection in self-generated solutions, thereby shedding light on its potential for self-reflection and self-improvement. We present the results in Figure 6.

When evaluating human-generated solutions, we observe that the judge accuracy consistently decreases as problem difficulty rises. This is expected: as tasks become more complex, the model finds it harder to fully understand the requirements and to follow the logic in human-written solution, much of

###### A B C D E F G

A B C D E F G

False Positive

True Negative

False Positive

True Negative

False Negative

True Positive

False Negative

True Positive

(a) DeepSeek-V3 over human solutions.

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |83.4%<br><br>73.8%| | | | | | |
| |68.0% 65.7%<br><br>60.1% 60.2%| | | | | | |
| |32.0% 34.3%<br><br>39.9% 39.8%<br><br>26.2%| | | | | | |
| |16.6%| | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

(b) QwQ-32B over human solutions.

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |85.4%| | | | | | |
| |64.4% 64.7%<br><br>60.3% 57.3%<br><br>63.4%| | | | | | |
| |35.6% 35.3%<br><br>39.7% 42.7%<br><br>36.6%| | | | | | |
| |14.6%| | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

(c) DeepSeek-V3 over self-generated solutions.

(d) QwQ-32B over self-generated solutions.

Figure 6: Judge distribution over solutions.

which may differ from its training data. However, when the model evaluates its own solutions, we find that its judge accuracy actually increases for the hardest problems. This indicates that, it is still able to recognize its own mistakes although the model often produces incorrect code on challenging tasks. This trend is similar to human behavior in that confidence are high for easy problems and very hard ones where failure is expected, and uncertainty peaks for problems of intermediate difficulty whose errors are hard to detect. Interestingly, this kind of self-awareness is also observed in reasoning models like QwQ. The rebound of accuracy in QwQ is less significantly than in DS-V3, highlighting the influence of self-reflection in the reasoning stage.

##### From Code to Cases: Can LLM generate more comprehensive cases with the code?

Given that the cases is highly related with code, a natural approach is to provide the model with a reference solution. By examining the structure of this solution, the model can better understand the underlying logic of the inputs and outputs to generate accurate test cases. In our experiments, we prompt the model with correct human-written solutions and present the results in Figure 8. Our results show that providing code solutions significantly improves the accuracy of generated test cases, particularly for challenging problems like E and F. With access to validated human solutions that handle various edge cases, the model can produce more comprehensive test cases, covering a wide range of scenarios by leveraging this prior knowledge. Inspired by these findings, it is potential to generate accurate test cases for problems lacking official cases as offline validation. This approach enables richer training data and more robust validation results for code training.

- 4.3 From Cases to X

From Cases to Editorial: Is LLM also capable to judge cases?

Following Section 4.2, we prompt the model to judge the correctness of the test case. The ground truth cases are official and guaranteed to be correct, while the model-generated cases contains both correct and incorrect samples. We exclude test cases whose total string length is larger than 200 and present the results in Figure 9. We observe that the model is able to identify some incorrect cases. Notably, for ground truth test cases, the model achieves a judging accuracy of up to 90% for QwQ-32B, and this high accuracy is consistent across different problem difficulties. In contrast, when evaluating its own generated cases, most misjudgments occur when the ground truth case fails but the model incorrectly deems it correct. This suggests that the self-consistency still limits its ability to judge cases, which aligns with the trends observed in Figure 6.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

GT Self-Random Self-Direct

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

GT Self-Random Self-Direct

(b) Coder-32B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

GT Self-Random Self-Direct

(c) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

GT Self-Random Self-Direct

(d) QwQ-32B

Figure 7: Pass@1 score over Self-Generated cases.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Code w/ GT-Code

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Code w/ GT-Code

(b) Coder-32B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Code w/ GT-Code

(c) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

w/o Code w/ GT-Code

(d) QwQ-32B

Figure 8: Case Score w.r.t. GT Solutions.

##### From Cases to Code: Can self-generated solution pass all the self-generated cases?

In this section, we explore another aspect of self-consistency: whether self-generated solutions can fully pass test cases produced by the model itself. We create two types of test cases: (1) Self-Direct: the model directly generates both inputs and outputs; (2) Self-Random: the model generates random inputs, and the outputs are derived from ground truth solutions. As shown in Figure 7, the pass rate on self-generated test cases is significantly higher than on the ground truth test cases. This is because self-generated cases often lack comprehensive coverage, particularly of edge conditions, and tend to remain within the bounds of the model’s own understanding. Consequently, the accuracy on Self-Random and Self-Direct test cases can exceed that on ground truth cases by up to 5% and 40%, respectively. Interestingly, for some problems that the model fails to solve entirely (e.g., Problem F), it can still achieve an "Accepted" result when evaluated on its own test cases. Nevertheless, the pass rate on self-generated test cases is not 100% across all problems, indicating that these test cases can still uncover errors in the self-generated solutions to some extent. Thus, leveraging self-generated test cases to verify the correctness of solutions remains viable and offers opportunities for self-reflection and iterative self-improvement.

#### 5 Related Works

Coding Models. Code generation has been fully influenced by LLMs, starting with Codex [8], which powers GitHub Copilot and excels in converting natural language prompts into functional code. AlphaCode [17] leverage an encoder-decoder architecture to solve complex algorithmic problems on platforms like Codeforces. Among open-source models, StarCoder [22] stands out for its community accessibility and robust performance. Qwen2.5-Coder [14] achieves impressive results, rivaling GPT-4o, while DeepSeek-R1 [12] employs reinforcement learning to enhance reasoning and coding capabilities. General-purpose models like GPT-4 [1] and o1 [24] demonstrate exceptional code generation, highlighting the versatility and rapid evolution of LLMs in this domain.

Coding Evaluation Benchmark. Evaluating code generation models relies on robust benchmarks that primarily judge functional correctness. HumanEval [8] serves as a foundational benchmark with 164 Python problems measure the ability of generating correct solutions. MBPP [3] complements HumanEval by providing approximately 1,000 crowd-sourced Python tasks aimed at entry-level programmers. EvalPlus [21] further enhances evaluation rigor by expanding HumanEval and MBPP with extensive test cases. LiveCodeBench [15] introduces a dynamic, contamination-free evaluation with over 880 problems from platforms such as LeetCode and Codeforces, covering diverse tasks including code repair and test output prediction. While these benchmarks collectively offer a comprehensive evaluation of code generation capabilities, they mainly focus on evaluating coding solutions and do not consider the model’s abilities in editorial analysis or test case generation.

###### A B C D E F

A B C D E F

False Positive

True Negative

False Positive

True Negative

False Negative

True Positive

False Negative

True Positive

(a) DeepSeek-V3 over ground truth cases.

JudgeDistribution(%)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| |47.4%<br><br>56.2%<br><br>60.5%<br><br>55.8%<br><br>59.7% 62.8%<br><br>66.5%<br><br>52.6%<br><br>43.8% 44.2%| | | | | | | |
| |33.5%<br><br>39.5% 40.3% 37.2%| | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

100

75

50

25

0

A B C D E F G

False Positive

True Negative

False Negative

True Positive

(b) QwQ-32B over ground truth cases.

JudgeDistribution(%)

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| |82.2%<br><br>73.3%| | | | | | | |
| |45.2% 48.5%<br><br>64.7%<br><br>58.1% 57.0% 54.8% 51.5%| | | | | | | |
| |26.7%<br><br>35.3%<br><br>41.9% 43.0%| | | | | | | |
| |17.8%| | | | | | | |
| | | | | | | | | |

100

75

50

25

0

A B C D E F G

False Positive

True Negative

False Negative

True Positive

(c) DeepSeek-V3 over self-generated cases.

(d) QwQ-32B over self-generated cases.

Figure 9: Judge accuracy over cases.

#### 6 Discussion and Conclusion

Model cognition of LLMs significantly differs from human distribution. Our large-scale experiments reveal that the distribution of solutions and cases they produce significantly differs from that of real-world data from human. This phenomenon indicates that the internal cognitive framework shaped by training data imposes constraints, limiting the ability to develop creative reasoning pathways and invent novel approaches when solving the code challenge.

Self-consistency exists within model cognition. Analyzing the prediction across different dimensions, we observe that self-consistency exists within the model cognition. For example, leveraging the self-generated editorial does not lead to significant improvements in solutions. Besides, the roll-out solutions can easily pass the self-generated test cases, resulting in a kind of reward hacking with respect to test cases. These observations indicate that, LLM exhibits self-consistency within its cognitive stage due to limitations of its own training data.

Inconsistency across various dimensions may facilitate self-improvement. LLM demonstrates different performance across the dimensions of the Coding Triangle and these differences can be further harnessed for self-improvement, such as using its own judgment to distinguish correct solutions, or leveraging a correct code to improve edge-case generation. This approach indicates that the development of self-improvement can be realized through iteratively align these dimensions, gradually reducing error correlations and develop LLMs with more powerful coding ability.

Model mixture enhance diversity and robustness. Recognizing the existence of self-consistency, we find that the combination of different models can significantly reduce bias while improving both robustness and diversity. For example, solutions generated by different models can help identify a wider range of boundary cases, and the combined test cases from multiple models are effective at detecting potential errors. These results demonstrate that leveraging model mixtures mitigates the biases and diverse outputs from different models contribute to better robustness.

Limitation and Conclusion. We systematically investigate the coding ability of LLMs through the lens of the coding triangle, i.e., editorial, code, and cases. There are still some limitations as we does not fully explore all possible interactions between these three dimensions. Overall, our study reveal the self-consistency and self-inconsistency inside LLMs, point out the importance of bridging the gap between model cognition and human expertise, and provide a potential direction of aligning and mutually reinforcing the three dimensions to achieve more reliable and generalizable coding models.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

- [2] Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet,

2024. 2024.06.21.

- [3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [4] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

- [5] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

- [6] Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020.

- [7] Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, et al. Multiple: A scalable and extensible approach to benchmarking neural code generation. arXiv preprint arXiv:2208.08227, 2022.

- [8] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

- [9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

- [10] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

- [11] Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, et al. Competitive programming with large reasoning models. arXiv preprint arXiv:2502.06807, 2025.

- [12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

- [13] Jiawei Guo, Ziming Li, Xueling Liu, Kaijing Ma, Tianyu Zheng, Zhouliang Yu, Ding Pan, Yizhi Li, Ruibo Liu, Yue Wang, et al. Codeeditorbench: Evaluating code editing capability of large language models. arXiv preprint arXiv:2404.03543, 2024.

- [14] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

- [15] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

- [16] AQ Jiang, A Sablayrolles, A Mensch, C Bamford, DS Chaplot, D de las Casas, F Bressand, G Lengyel, G Lample, L Saulnier, et al. Mistral 7b (2023). arXiv preprint arXiv:2310.06825, 2023.

- [17] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

- [18] Ziming Li, Qianbo Zang, David Ma, Jiawei Guo, Tianyu Zheng, Xinyao Niu, Xiang Yue, Yue Wang, Jian Yang, Jiaheng Liu, et al. Autokaggle: A multi-agent framework for autonomous data science competitions. arXiv preprint arXiv:2410.20424, 2024.

- [19] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

- [20] Jiaheng Liu, Ken Deng, Congnan Liu, Jian Yang, Shukai Liu, He Zhu, Peng Zhao, Linzheng Chai, Yanan Wu, Ke Jin, et al. M2rc-eval: Massively multilingual repository-level code completion evaluation. arXiv preprint arXiv:2410.21157, 2024.

- [21] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems, 36:21558–21572, 2023.

- [22] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.

- [23] OpenAI. Gpt-4o. https://openai.com/index/hello-gpt-4o, 2024. 2024.05.13.
- [24] OpenAI. Openai o1 system card. https://openai.com/index/ openai-o1-system-card/, 2025.
- [25] OpenAI. Openai o3 system card. https://openai.com/index/ o3-o4-mini-system-card/, 2025.
- [26] OpenAI. Openai o3-mini system card. https://openai.com/index/openai-o3-mini/, 2025.
- [27] Qwen Team. Qwen3: Think deeper, act faster. https://qwenlm.github.io/blog/qwen3/, 2025.
- [28] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.
- [29] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

- [30] Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xinrun Du, Di Liang, Daixin Shu, Xianfu Cheng, Tianzhen Sun, et al. Tablebench: A comprehensive and complex benchmark for table question answering. arXiv preprint arXiv:2408.09174, 2024.

- [31] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

- [32] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

### Appendix

#### A Dataset and License

The evaluating problem dataset is collected from AtCoder (https://atcoder.jp). We only collect publicly available content, including visible editorials, sample codes, and test cases published on the website. All collected materials are strictly used for research and evaluation purposes only, specifically to assess the performance of candidate models. No part of the collected dataset is used for training any model and we fully respect the content ownership and terms of use of AtCoder. We follow LiveCodeBench and abide by Fair Use §107: "the fair use of a copyrighted work, including such use by ... scholarship, or research, is not an infringement of copyright", where fair use is determined by "the purpose and character of the use, including whether such use is of a commercial nature or is for nonprofit educational purposes" and "the effect of the use upon the potential market for or value of the copyrighted work."

#### B More Experiments

In this section, we provide more experiments as additional results. We mainly focus on model mixture and the interaction across different models.

##### B.1 Does LLM benefit from the editorial from other models?

We further investigate whether editorials generated by other models can benefit code generation. To this end, we evaluate the performance of Coder-32B-Instruct when provided with editorials from Qwen2.5-72B-Instruct, DeepSeek-V3, and QwQ-32B, as shown in Figure 10. The results indicate that using editorials as prompts can serve as a form of knowledge distillation, enhancing the performance of the student model. With access to relatively accurate editorials from reasoning-oriented models, Coder-32B-Instruct, which excels at code implementation, can achieve performance comparable to that obtained with ground truth editorials, demonstrating the practicality of leveraging logical knowledge embedded in reasoning models to improve the code generation capabilities of coding models.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Target-Edi. w/ GT Edi.

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Target-Edi. w/ GT Edi.

(b) DeepSeek-V3

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Pass@1Score

75

50

25

0

A B C D E F

w/o Edi. w/ Target-Edi. w/ GT Edi.

(c) QwQ-32B

Figure 10: Pass@1 score of Coder-32B-Instruct with editorials from other models.

##### B.2 Can LLM recognize mistakes from other models?

We further conduct experiments to investigate whether LLMs can identify mistakes made by other models, aiming to explore the judging capability of LLMs. Specifically, we utilize DeepSeek-V3 to determine whether the solutions generated by other models contain errors. As shown in Figure 11, the model is also capable of identifying mistakes in the solution, and the accuracy all exhibits a decreasing-then-increasing trend.

For relatively weaker models (such as coder-32B and Qwen-72B), which lack the ability to solve difficult problems, DeepSeek-V3 can easily identify the errors in their solutions to such challenging problems, which is reflected by True Negatives constituting the majority of accurate judgments. Moreover, in the cases where DeepSeek-V3 judges its own solutions and those of QWQ-32B, we observe that DeepSeek-V3 shows a stronger tendency to judge its own solutions as correct, revealing a form of self-consistency in the model’s self-perception.

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |80.0% 78.0%| | | | | | |
| |45.1% 45.7%<br><br>61.7%<br><br>54.9% 54.3%<br><br>66.7%| | | | | | |
| |20.0%<br><br>38.3%<br><br>33.3%<br><br>22.0%| | | | | | |
| | | | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |78.9%<br><br>82.6%| | | | | | |
| |43.4%<br><br>47.1%<br><br>62.9%<br><br>56.6%<br><br>52.9%<br><br>67.3%| | | | | | |
| |21.1%<br><br>37.1%<br><br>32.7%| | | | | | |
| |17.4%| | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

(a) Qwen2.5-72B-Instruct

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |83.4%<br><br>73.8%| | | | | | |
| |68.0% 65.7%<br><br>60.1% 60.2%| | | | | | |
| |32.0% 34.3%<br><br>39.9% 39.8%<br><br>26.2%| | | | | | |
| |16.6%| | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

(b) Coder-32B

JudgeDistribution(%)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| |80.6%<br><br>69.4%| | | | | | |
| |48.8%<br><br>63.2%<br><br>51.2%<br><br>57.3%<br><br>66.9%| | | | | | |
| |19.4%<br><br>30.6%<br><br>36.8%<br><br>42.7%<br><br>33.1%| | | | | | |
| | | | | | | | |
| | | | | | | | |

100

75

50

25

0

A B C D E F

False Positive

True Negative

False Negative

True Positive

(c) DeepSeek-V3

(d) QwQ-32B

Figure 11: Judge distribution of DeepSeek-V3 over solutions from different models.

##### B.3 How does case generation benefit from model mixture?

To further validate the effectiveness of model mixture, we merge the cases generated by different models and calculate the accuracy of these case sets. The results are presented in Figure 12. We observe that the model mixture brings significant improvements in case score performance, as different models exhibit different biases and the diversity introduced by these models in the case generation task directly contributes to the improvement in scores.

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

DS-V3 Qwen-72B DS-V3 + Qwen-72B

(a) Qwen2.5-72B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

DS-V3 Coder-32B DS-V3 + Coder-32B

(b) Coder-32B-Instruct

100

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

75

Score

50

25

0

A B C D E F

DS-V3 QwQ-32B DS-V3 + QwQ-32B

(c) QwQ-32B

Figure 12: Case score of model mixture between DeepSeek-V3 and other models.

###### Editorial Generation Prompt

[[ Instruction ]] You are a code expert, and your goal is to develop a editorial analysis for a given coding problem. You will analyze the problem, explain the approach, including any necessary constraints and mathematical formulas, to help the reader understand how to solve it. Your explanation should not include the final code but should guide the reader to implement it themselves. Please indicate the time complexity at the end of the analysis. [[ Constraints ]] Your editorial analysis must satisfy the following:

- - Clearly explain the approach to solve the problem.
- - Include any necessary constraints from the problem statement.
- - Use mathematical formulas where applicable to clarify the solution.
- - Do not provide the final code, only the editorial.
- - Indicate the time complexity of the solution at the end.
- - Ensure that the editorial addresses all aspects of the problem as stated. [[ Example ]] [Editorial] The leaves of T are leaves of initial stars, and those vertices distant by at most two from it belong to the same star. Thus, the following algorithm is possible.
- - Choose a leaf of T.
- - Count the number of vertices distant by at most two from it (including itself). If there are x of them, they form a level-(x − 1) star.
- - Remove the counted vertices and adjacent edges from T. After you repeat this until no vertices remain in T, you obtain the answer. With an appropriate implementation, it works in a total of O(N) time, but the implementation is a bit complicated. We describe a simpler implementation with some observations. The following lemma holds:
- - In an original star, let us call the non-leaf vertex the center. In T, the distance between centers is always a multiple of 3, and that between a center and a leaf is always a non-multiple of 3. This can be shown by induction. With this lemma, we can come up with the following algorithm:
- - Choose a vertex from T that was the center of an original star. You can do so by choosing the vertex adjacent to a leaf.
- - Find the shortest distance of each vertex from the chosen vertex.
- - For each vertex whose shortest distance is a multiple of 3, add the degree of that vertex to L. [Time Complexity] O(N) [[ Output Format ]] Your response should be formatted as follows and should not include any additional information: [Think] Your thinking about the reasoning process in the mind. [Editorial] Your final editoral of the problem. [Time Complexity] The time complexity of your solution. [[ Problem Begin ]] {problem} [[ Problem End ]]

###### Solution Generation Prompt

[[ Instruction ]] You are an expert C++ programmer. Your goal is to generate a complete, correct C++ program for a given coding problem. The program should handle all edge cases, follow best practices, and be efficient where necessary. Enclose your program within C++ code delimiters as shown below. [[ Constraints ]]

- - Generate a fully functional C++ solution that compiles and passes all tests.
- - The code should be standalone and not rely on external libraries beyond what’s standard in C++, unless specified in the problem.
- - Adhere strictly to the problem’s input and output formats.
- - Ensure the code is clean, well-indented, and includes comments to explain complex logic.
- - Include all necessary headers and use the standard namespace.
- - Wrap your code in triple backticks with C++ annotation. [[ Example ]] #include <bits/stdc++.h> using namespace std; int main() { // sample solution return 0; } [[ Output Format ]] Your response should be formatted as follows and should not include any additional information: [Analysis] Your analysis of the problem. [Code] Your C++ code in a code block. [[Problem begin]] {problem} [[Problem end]]

###### Case Generation Prompt

[[ Instruction ]] You are an expert Python competitive programmer and your goal is to construct input-generators for testing programming contest problems. You will write relevant generators and finally implement a ‘construct_inputs‘ function that returns a list of 50 diverse inputs sampled from those generators. Remember to strictly follow the instructions and constraints present in the problem statement. [[ Constraints ]] Your input-generators and ‘construct_inputs‘ must satisfy all of the following:

- - **Deterministic framework**: the code may call randomness internally, but the overall scheme and parameter ranges must be hard-coded (no external configuration or user prompts).
- - **Coverage**: include edge-case ranges (smallest/largest sizes, boundary weight values), typical scenarios, and stress scenarios near the problem’s limits.
- - **Validity**: generated inputs must always respect the problem’s stated input format and numeric bounds (e.g. 1 ≤ N ≤ Nmax, weightmin ≤ weighti ≤ weightmax, etc.).
- - **Reproducibility**: allow for seeding if needed (e.g. accept a seed parameter), but default behavior needs no external input.
- - **Diversity**: return a list containing at least three tiers of size/scale (e.g. small, medium, large) and within each tier cover multiple parameter combinations.
- - **Clarity**: each testcase’s ‘input‘ string must be parseable by the contestant’s code. [[ Example ]]

| |
|---|

- 1 import numpy as np
- 2 def random_input_generator(weight_min , weight_max , size_min , size_max , seed=None):
- 3 if seed is not None:
- 4 np.random.seed(seed)
- 5 n = np.random.randint(size_min , size_max +1)
- 6 weights = np.random.randint(weight_min , weight_max+1, size=n).tolist()
- 7 k = np.random.randint(1, n+1)
- 8 return { ’input’: ’␣’.join(map(str , weights)) + ’␣’ + str(k) + ’\\n’ }
- 9
- 10 def edge_case_generator(case_id):
- 11 cases = [
- 12 # 0: smallest size , smallest weight
- 13 { ’input’: ’1␣1\\n’ },
- 14 # 1: smallest size , largest weight
- 15 { ’input’: ’1␣1000000000\\n’ },
- 16 ...
- 17 # 9: mixed boundary in medium
- 18 { ’input’: ’1000␣’ + ’␣’.join([’1’]*499 + [’1000000’]*500) + ’␣250000\\n’ },
- 19 ]
- 20 return cases[case_id]
- 21
- 22 def construct_inputs ():
- 23 inputs_list = []
- 24 # 10 edge cases
- 25 for i in range (10):
- 26 inputs_list.append(edge_case_generator(i))
- 27 # small tier
- 28 for i in range(10, 20):
- 29 inputs_list.append(random_input_generator(1, 10**3, 1, 10, seed=i))
- 30 # medium tier
- 31 for i in range(20, 35):
- 32 inputs_list.append(random_input_generator(1, 10**6, 1, 10**3, seed=i))
- 33 # large tier
- 34 for i in range(35, 50):
- 35 inputs_list.append(random_input_generator(1, 10**9, 1, 10**5, seed=i+100))
- 36
- 37 return inputs_list

[[ Output Format ]] Your response should be formatted as follows and should not include any additional information: [Analysis] Your analysis of the problem. [Code] Your Python scripts in a code block. [[ Problem Begin ]] {problem} [[ Problem End ]]

###### Editorial Judge Prompt

[[ Instruction ]] You are a code expert and judge. Your goal is to evaluate a candidate’s editorial for a given coding problem, using the official editorial as a reference. First, extract the time complexities from both the official editorial and the candidate’s editorial. If the candidate’s time complexity is asymptotically worse than the official one, assign a score of 0. Otherwise, analyze whether the candidate’s solution is logically correct and solves the problem as required. Note that the candidate’s approach may differ from the official one, but it should still be a valid solution to the problem. Finally, assign a binary score: 1 if the solution is correct and has an acceptable time complexity, otherwise 0. [[ Constraints ]] Your judgment must satisfy the following:

- - Read and understand the problem statement in full.
- - Use the official editorial as a reference to verify the correctness of the candidate’s approach, but allow for different valid solutions.
- - Assign a score of 1 if the candidate’s solution is logically correct and has an acceptable time complexity, otherwise assign 0. [[ Output Format ]] Your response must follow exactly this format without any extra information: [Analysis] Your analysis of the problem, the two editorials, and the correctness of the candidate’s solution. [Score] Your final judge score. [[ Problem Begin ]] {problem} [[ Problem End ]] [[ Official Editorial Begin ]] {gt editorial} [[ Official Editorial End ]] [[ Candidate Editorial Begin ]] {editorial} [[ Candidate Editorial End ]]

Solution Judge Prompt

[[ Instruction ]] You are a programming competition judge. Your task is to analyze a submitted solution for a specified problem and determine its correctness. You should focus on logical correctness, coverage of all edge cases, and any implementation flaws that would cause test failures. [[ Constraints ]]

- - Provide a detailed analysis of potential logical errors or omissions.
- - Indicate whether the solution passes all test cases or fails some.
- - Do not execute code; base your judgment on static reasoning.
- - Assign a score of 1 if the candidate’s solution is logically correct and has an acceptable time complexity, otherwise assign 0. [[ Example ]] [Analysis] The solution attempts binary search but has an off-by-one error in the termination condition (line 8). This causes incorrect results when the target is at array boundaries. [Score] 0 [[ Output Format ]] Your response should be formatted as follows and should not include any additional information: [Analysis] Your analysis of the problem and the solution. [Score] Your 0/1 score of the solution. [[ Problem begin ]] {problem} [[ Problem end ]] [[ Solution begin ]] {solution} [[ Solution end ]]

###### Case Judge Prompt

[[ Instruction ]] You are a programming competition judge. Your task is to determine whether a given test case’s input and output match according to the problem’s specification. Focus solely on whether the provided output is the correct result for the provided input under the problem logic. [[ Constraints ]]

- * Check that the “Case Output” is exactly what the problem would produce for the given “Case Input.”
- * Identify any discrepancies, incorrect results, or mismatches.
- * Do not assess solution code—only compare input versus output.
- * Do not execute code; base your judgment on logical reasoning and the problem statement.
- * Assign a score of 1 if the candidate’s case is logically correct, otherwise assign 0. [[ Example ]] [Analysis] For input ‘3 5 2‘, the problem asks for the sum of the first two numbers. The expected result is ‘8‘, but the provided output is ‘15‘, so they do not match. [Score] 0 [[ Output Format ]] Your response must follow this exact format, with no additional text: [Analysis] <your detailed analysis of input-output matching> [Score] <0 or 1> [[ Problem begin ]] {problem} [[ Problem end ]] [[ Case Input Begin]] {case input} [[ Case Input end ]] [[ Case Output Begin ]] {case output} [[ Case Output end ]]

