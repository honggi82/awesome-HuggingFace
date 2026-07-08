arXiv:2506.09820v2[cs.CL]12Jun2025

# CoRT: Code-integrated Reasoning within Thinking

Chengpeng Li∗1,2, Zhengyang Tang∗2,3, Ziniu Li∗3,4, Mingfeng Xue2, Keqin Bao1,2, Tian Ding4, Ruoyu Sun3,4, Benyou Wang3, Xiang Wang1, Junyang Lin2, and Dayiheng Liu†2

1University of Science and Technology of China 2Qwen Team, Alibaba Inc. 3The Chinese University of Hong Kong, Shenzhen 4Shenzhen Research Institute of Big Data

## Abstract

Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model’s internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4% and 8% absolute improvements on DeepSeekR1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30% fewer tokens for the 32B model and 50% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at: https://github.com/ChengpengLi1003/CoRT.

## 1 Introduction

Benefiting from advancements in reinforcement learning (RL) techniques [1–4], Large Reasoning Models (LRMs) such as OpenAI-o1 [5], QwQ [6], DeepSeek-R1 [7] and Kimi-1.5 [8] have achieved breakthrough progress in complex reasoning tasks. These models exhibit numerous human-like cognitive strategies with long Chain of Thought (CoT) [9, 10] reasoning, including self-refinement, self-reflection, and multi-strategy exploration. However, LRMs still demonstrate limitations in accuracy and efficiency when handling complex mathematical operations, such as precise computation and complex equation solving [11, 12], which are better suited for code interpreters (CIs). Leveraging CIs, LRMs like o3 and o4-mini [13] have substantially enhanced their mathematical reasoning capabilities.

∗: Equal contribution. Email: chengpengli@mail.ustc.edu.cn, zhengyangtang@link.cuhk.edu. cn, ziniuli@link.cuhk.edu.cn †: Corresponding authors.

Preprint.

[Figure 1]

Figure 1: Performance vs. token efficiency on AIME24. The x-axis represents average token usage while the y-axis shows Pass@1 accuracy. Hint-Engineering-RFT-32B achieves comparable accuracy to other frontier models while using significantly fewer tokens.

A key open challenge is teaching LRMs when and how to effectively and efficiently use CIs to generate structured reasoning. This is a scientifically new problem, because unlike pure natural language reasoning, CIs introduce external deterministic knowledge that exists beyond the model’s internal representations. This raises critical questions: (1) How can we synthesize high-quality training data when models like o3 and o4-mini do not expose their detailed reasoning traces? (2) How to effectively coordinate between CI’s computational precision and CoT’s abstract reasoning capabilities? (3) How can the self-reflection mechanisms inherent to LRMs be reconciled with the exact external knowledge provided by CIs? These challenges are particularly acute given that effective CI integration requires teaching LRMs not only when to utilize external tools but also how to structure their reasoning.

This paper explores to answer the above questions. We begin by tackling the data synthesis challenge, which forms the foundation for post-training via supervised fine-tuning (SFT) [14], rejection finetuning (RFT) [15] and RL [7]. Based on the open-source LRM like QwQ and DeepSeek-R1, we investigate direct prompting methods like [16] for CI integration. Our key discovery is that inserting a simple hint—"Okay, let’s try to solve this problem step by step using multiple python code calls"—immediately after the model’s thinking token <think> improves code triggering rates from 50% to 90% (on 100 problems from [16]). We term this approach the prompt-hint method. This confirms that LRMs possess the latent capability to leverage CIs for reasoning despite being primarily trained on natural language. However, we also find that they struggle with efficient tool utilization.

It highlights a fundamental challenge: LRMs do not yet understand how to incorporate external knowledge into their reasoning processes. The two most prominent inefficiencies are delayed code computation (preferring text reasoning before utilizing CI) and code result distrust (unnecessarily verifying CI outputs manually), as shown in Figure 3. To address these limitations, we design another approach, which refers to hint-engineering. The key idea is to strategically inserting different hints at appropriate positions throughout the reasoning process. These hints are specifically designed to teach the LLM understand the outputs of CIs, mitigating meaningless reflection behaviors.

Following the principle that data quality outweighs quantity (less is more) [17–19], we manually generate 30 high-quality samples with human verification. Using these samples, we post-train models of varying sizes based on available computational resources. For large 32B parameter models, we conduct SFT and RFT, while RL remains computationally infeasible within our infrastructure. However, we successfully implement the complete SFT-RFT-RL pipeline for smaller models. Moreover, we carefully design outcome rewards to encourage writes the codes correctly.

Our experiments confirm the effectiveness of the above approaches. Results across five challenging mathematical reasoning datasets demonstrate that Hint-Engineering models achieve significant improvements: 4% absolute accuracy gain for DeepSeek-R1-Distill-Qwen-32B and 8% for DeepSeek-

[Figure 2]

Figure 2: The training framework of CoRT.

R1-Distill-Qwen-1.5B. Moreover, on the most challenging AIME benchmarks, our approach reduces token consumption by 30% for the 32B model and 50% for the 1.5B model.

To summarize, our key contributions include:

- • A new data synthesis framework specifically engineered for code-integrated reasoning that effectively addresses the critical data scarcity challenge in this emerging domain.
- • An efficient and scalable training pipeline that enables LLMs to acquire sophisticated codeintegrated reasoning capabilities through targeted post-training procedures.
- • Comprehensive empirical evaluations demonstrating significant performance and token efficiency improvements across 5 challenging mathematical benchmarks.

In the main text, we will present our main methodology and defer the related work to Appendix A.

## 2 Methodology

In this section, we introduce the CoRT framework as illustrated in Figure 2.1, encompassing the modeling of code-integrated reasoning processes, the training pipeline for 32B models (including Cold Start, SFT, and RFT), and the development of 1.5B models through strong-to-weak distillation and RL.

- 2.1 Task Formulation

By leveraging executable programs, LRMs can now perform precise calculations and complex logical operations. The framework comprises three essential components: a problem input P, a language model π, and an executor environment E. During the reasoning process, the system constructs a sequence τt at time step t, which can be represented as:

τt = {(n1,p1,o1),...,(nt,pt,ot)} (1)

Here, ni represents the textual reasoning step, pi denotes the program snippet generated by the model, oi indicates the execution output, with i indexing the sequential interactions between the language model and the execution environment. The sequential reasoning process follows these steps:

(tt,pt) = π(P ⊕ τt−1),ot = E(pt) τt = τt−1 ⊕ nt ⊕ pt ⊕ ot

(2)

This iterative mechanism establishes a dynamic feedback loop, where each reasoning step is informed by previous computational results. The process continues until the model reaches a definitive answer.

- 2.2 Cold Start Methods

- 2.2.1 Prompt-hint

To initiate our data generation process, we carefully crafted a prompt in Appendix E designed to instruct R1 [7] to leverage both natural language reasoning and interactive Python code execution during inference. We integrated a code interpreter that enables R1 to perform real-time interactive reasoning, as outlined in section 2.1.

Our initial observations revealed that models exhibited a relatively low probability of generating reasoning trajectories that incorporate code. Inspired by [20], we enhanced the generation process by

Each face of two noncongruent parallelepipeds is a rhombus whose diagonals have lengths 21and 31. The ratio of the volume of the larger of the two polyhedra to the volume of the smaller is 𝑚/𝑛, where $m$ and $n$ are relatively prime positive integers. Find 𝑚 + 𝑛.

<think> Okay, let's try to solve this problem step by step using multiple python code calls.

<think> Okay, let's try to solve this problem step by step using multiple python code calls. [skip 3261 tokens] Let‘s compute V1 and V2 with Python for accuracy.

... [skip 8524 tokens] … Thus, the answer is 125. Now, to confirm using Python.

###### Without Hint-1

Let's compute V1 and V2 for

these two cases. [skip 5057 tokens] V1= 63sqrt(2)/2,V2= 31sqrt(2)

Hint-1: Complex calculations

import sympy as sp …

[Figure 3]

>>> Volume 1: 63*sqrt(2)/2

Volume 2: 31*sqrt(2)

import numpy as np …

Without Hint-2

We don't need to doubt the accuracy of python calculations. [skip 216 tokens] </think>… The answer is \boxed{125}.

Wait, but let me check These calculation… [ skip 2215 tokens] </think>… The answer is \boxed{125}.

[Figure 4]

>>> 125

Hint-2:

Confirm python calculation

So m=63, n=62, m+n=125. </think> … The sum m + n = 63 + 62 = \boxed{125}.

(a) Prompt-hint (b) Hint-engineering

- Figure 3: Comparison between prompt-hint and hint-engineering approaches using Problem 13 from AIME23 I as a case study (prompt prefix omitted for brevity).Both methods begin with a general hint (in green) after <think> to encourage code usage. While prompt-hint (a) allows natural interaction between R1 and the Code Interpreter (CI), leading to inefficient token usage, hint-engineering (b) introduces strategic hints at key decision points. Hint-1 is inserted when the model begins manual calculation of complex volumes (V1 and V2), redirecting to Python computation. Hint-2 is added to prevent unnecessary verification of Python calculations. Through these targeted interventions, hint-engineering achieves approximately 5000 token reduction while maintaining solution accuracy. More examples are provided in Appendix F.

introducing a strategic hint - "Okay, let’s try to solve this problem step by step using multiple python code calls" - following the model’s thinking beginning token <think>. This intervention significantly increased the likelihood of models producing reasoning processes that integrate codes. This approach is referred to as prompt-hint, with an example shown in Fig. 3 (a).

Leveraging the publicly available STILL3 [16] dataset comprising 820 math problems and R1, we employed our prompt-hint annotation method to generate 800 training instances, denoted as Dprompt−hint. We then performed SFT on DeepSeek-R1-Distill-Qwen-32B using this dataset, resulting in our Prompt-Hint-SFT-32B model.

### 2.2.2 Hint-engineering

Despite the effectiveness of the prompt-hint approach, where LRMs autonomously decide when and how to utilize the Code Interpreter (CI), we identified several inefficiencies and instances of overthinking. These limitations can be categorized into two main issues:

- • Delayed code computation: When handling complex mathematical operations, models tend to first engage in text-based reasoning before writing code and using CI for verification. This pattern often results in redundant computational steps.
- • Code result distrust: Upon receiving CI execution results, models frequently display a lack of trust in the output, leading to unnecessary manual verification and redundant calculations.

These behavioral patterns significantly impact the model’s reasoning efficiency, particularly in terms of the number of tokens required for problem-solving.

To address these inefficiencies, we implement a targeted approach named hint-engineering. When delayed computation deferral is detected, specifically at the point where the model begins manual calculation of complex mathematical operations, we insert a strategic hint like "It looks tedious, and we can use python code to simplify the reasoning.‘‘‘python". Similarly, when computational result distrust behavior emerges, we introduce the hint like "We don’t need to doubt the accuracy of python calculations." This prompt redirects the model’s focus back to the core problem rather than engaging in unnecessary verification of computational accuracy. It’s important to note that while we discourage

the verification of Python’s numerical calculations, we maintain the model’s behaviour to verify the logical correctness of the code structure. Figure 3 (b) illustrates a concrete example.

A critical challenge in our approach was identifying suitable positions for hint insertion. While we initially attempted to automate this process using LLMs like Qwen2.5-72B-instruct [21],DeepSeek-v3 [22] and R1 [7], we found the results to be insufficiently precise. Hence, we opted for manual hint insertion with 30 probelms from AIME problems before 2024 to obtain DHint−engineering−SFT and Hint-Engineering-SFT-32B.

To further enhance model performance, we conducted rejection fine-tuning(RFT) using the HintEngineering-SFT-32B model on the 820 problems from STILL3. Specifically, we performed multiple sampling iterations on each problem and implemented a filtering process to eliminate trajectories with incorrect final answers, as well as those exhibiting delayed code computation or code result distrust behaviors. We combined the filtered trajectories with DHint−engineering−SFT to create DHint−engineering−RFT, a dataset of 830 examples. This curated dataset was then used to fine-tune DeepSeek-R1-Distill-Qwen-32B, resulting in our Hint-Engineering-RFT-32B model.

### 2.3 Strong-to-weak Distillation

Given the computational constraints of our infrastructure in performing reinforcement learning on 32B-parameter code-integrated LRMs, we distilled both Prompt-Hint-SFT-32B and Hint-EngineeringRFT-32B models into DeepSeek-R1-Distill-Qwen-1.5B for RL experimentation. This process yielded two smaller models: Prompt-Hint-1.5B-SFT and Hint-Engineering-SFT-1.5B, which served as the foundation for our reinforcement learning exploration. We selected and preprocessed 10k examples from publicly available datasets for distillation, with detailed procedures provided in Appendix G.

### 2.4 Code-integrated Reinforcement Learning

We conduct reinforcement learning on Prompt-Hint-1.5B-SFT and Hint-Engineering-SFT-1.5B models with GRPO [3] algorithm. We mainly want to answer the following two questions through some series of reinforcement learning experiments.

- • Whether models can enhance their reasoning capabilities through RL after acquiring code-integrated reasoning skills via SFT?
- • How the models’ interaction patterns with the Code Interpreter evolve during the RL process?

In applying the GRPO algorithm to our models, we introduced several modifications to the standard text-based GRPO framework:

- • Rollout with Code Interpreter: We enable multiple model-CI interactions during the RL rollout process, as described in Section 2.1. To manage computational overhead during rollouts, we implement a maximum tool usage limit T. Once this limit is reached, we append a hint informing the model to proceed without further Python usage.
- • Persistent Execution Environment: Unlike traditional TIR environments that execute each Python block independently, we construct a Jupyter-like environment where variables, environments, and functions persist across code blocks, enhancing code efficiency and reducing errors.
- • Output Masking: To ensure training stability, we implement execution result masking, significantly reducing model collapse probability during training. This crucial modification prevents potential training failures that would otherwise occur without such masking.
- • Reward Design: We implement a dual reward system comprising accuracy reward and code execution reward as defined in Equations 3. For accuracy assessment, we require models to present final answers in a specified format (e.g., within boxed{}), enabling reliable rule-based verification against ground truth answers. To prevent infinite loops resulting from repeated code failures, we implement a code execution penalty for responses where all code execution attempts fail. The total reward R is computed as a weighted sum of these two components, where ω controls the contribution of the code execution penalty.

1 if answers match 0 otherwise

Rc = −1 if all codes fail 0 otherwise

R = Ra + ωRc (3)

Ra =

## 3 Experiments

In this section, we evaluate the effectiveness of our proposed CoRT framework through comprehensive experiments on five challenging mathematical reasoning benchmarks: (1) AIME24, (2) AIME25,(3) AMC23, (4) MATH500, and (5) OlympiadBench. Comprehensive descriptions of the evaluation datasets are provided in Appendix C. For space saving, our implementation details of SFT, RFT, RL and inference are listed in Appendix B. Due to space constraints, we present only representative experimental results in the main text, with comprehensive results available in Appendix D. Moreover, the baseline models are described in Appendix H.

### 3.1 Main Results

Table 1: Performance comparison of different math reasoning models across benchmarks. For each section, best results are shown in bold and second-best results are underlined. During inference, we set temperature 0.6 and topp 0.95. Results for AIME24, AIME25, and AMC23 are averaged over 16 samples, while MATH500 and Olympiad results are averaged over 4 samples. All experiments use a maximum sequence length of 32,768 tokens and limit tool usage to 15 calls.

Model Tool-Use AIME24 AIME25 AMC23 MATH500 Olympiad Avg SOTA Models

o1 ✗ 74.3 79.2 - 96.4 - DeepSeek-R1 ✗ 79.8 70.0 - 97.3 - QwQ-32B ✗ 79.5 65.3 94.3 92.3 79.7 82.2

Frontier Models (32B)

DeepSeek-R1-32B ✗ 72.9 59.0 88.8 94.3 72.5 77.5 START-32B ✓ 66.7 47.1 95.0 94.4 - STILL-3-TOOL-32B ✓ 76.7 64.4 91.3 96.6 75.9 81.0 ReTool-R1-32B ✓ 72.5 54.3 92.9 94.3 69.2 76.6 Prompt-Hint-SFT-32B ✓ 77.3 65.0 95.0 96.6 75.1 81.8 Hint-Engineering-SFT-32B ✓ 72.1 60.2 91.3 94.4 71.2 77.8 Hint-Engineering-RFT-32B ✓ 76.7 67.1 94.4 95.1 73.4 81.3

Lightweight Models (1.5B)

DeepSeek-R1-1.5B ✗ 28.8 21.8 62.9 83.9 43.3 48.1 DeepScaleR-1.5B-Preview ✗ 40.0 30.0 73.6 87.8 50.0 56.3 ToRL-1.5B ✓ 26.7 26.7 67.5 77.8 44.0 48.5 Prompt-Hint-1.5B-SFT ✓ 30.6 25.0 63.1 83.3 50.4 50.5 Prompt-Hint-1.5B-RL ✓ 43.1 30.2 73.8 87.3 57.1 58.3 Hint-Engineering-1.5B-SFT ✓ 34.0 23.5 64.6 84.2 49.8 51.2 Hint-Engineering-1.5B-RL ✓ 41.0 29.4 70.0 85.8 55.6 56.4

Table 1 presents our main results, comparing our models with state-of-the-art baselines across multiple mathematical reasoning benchmarks. We organize the results into three sections: SOTA Models, Frontier Models (32B), and Lightweight Models (1.5B).

For 32B models, we observe that after SFT, our models achieve performance comparable to existing tool-integrated models, with Prompt-Hint-SFT-32B slightly outperforming others with an average accuracy of 81.8% across benchmarks. Notably, Hint-Engineering-RFT-32B, despite being trained on just 30 manually annotated examples initially, achieves competitive performance with an average accuracy of 81.3%. This highlights the effectiveness of our rejection fine-tuning approach and the importance of high-quality data over quantity.

For 1.5B models, the reinforcement learning stage brings substantial improvements. Prompt-Hint1.5B-RL achieves state-of-the-art performance among lightweight models with an average accuracy of 58.3%, outperforming the non-tool-using DeepScaleR-1.5B-Preview. Similarly, Hint-Engineering1.5B-RL shows strong performance at 56.4%. The dramatic improvement from SFT to RL stages (approximately 8% absolute gain) demonstrates the effectiveness of our reinforcement learning approach for tool-integrated reasoning.

Avg Tokens for Correct Responses

80

50%

20000

40%

70

11364

50%

9438

10000

8391

7594

5363

5233

60

4896

AverageTokens

Pass@1(%)

72.9%

79.5%

76.7%

72.5%

77.3%

72.1%

77.1%

0

50

10000

40

12257

12713

14538

16542

17861

17986

30

20000

R1-distill-32B

Prompt-Hint-SFT-32B

Avg Tokens for Incorrect Responses

Hint-Engineering-RFT-32B

24712

20

4 6 8 10 12 16 20 24 28 32

R1-distill-32B QwQ-32BSTILL-3-TOOL ReTool-R1Prompt-HintHint-Eng-SFTHint-Eng-RFT

Inference Token Budget (k)

(a) Performance on AIME24 with different token budget

(b) Average token usage on AIME24

- Figure 4: Token efficiency analysis on AIME24. (a): Token efficiency comparison showing HintEngineering-RFT-32B achieves comparable accuracy with significantly fewer tokens (40-50% token saving) compared to Prompt-Hint-SFT-32B. (b): Average token usage for correct and incorrect responses across different models, with Hint-Engineering models maintaining lower token consumption while achieving competitive performance.

### 3.2 Token Efficiency Analysis

Beyond raw performance, we analyze the token efficiency of our models. Token efficiency can be roughly estimated by dividing the model’s accuracy by its average token consumption. Figures 1 and 4 illustrate this analysis, revealing several key insights:

- • Superior Efficiency of Hint-Engineering: At equivalent performance levels, Hint-Engineering series models demonstrate the highest token efficiency. For example, as shown in Figure 1, HintEngineering-RFT-32B achieves the same performance as QwQ-32B while using 50% fewer tokens (7K vs 14K). Comparing Hint-Engineering-SFT-32B with R1-distill-32B, with just 30 training examples for fine-tuning, the model reduces token consumption by approximately 30% while maintaining comparable performance. As observed from the inference token budget analysis in

- Figure 4 (a), Hint-Engineering achieves superior performance compared to Prompt-Hint under limited token budgets, while Prompt-Hint shows no significant advantage over CoT in these constrained conditions.

- • Low Token Usage for both Correct and Incorrect Responses: As shown in Figure 4 (b), compared to CoT and Prompt-Hint approaches, Hint-Engineering reduces token consumption in both correct and incorrect responses, indicating that it not only solves problems efficiently but also minimizes token waste during unsuccessful attempts. This improvement stems from Hint-Engineering’s fundamental design: increasing the utilization of code for computations and enhancing confidence in code execution results.

As shown in Figure 1, when plotting performance against token usage, Hint-Engineering-RFT-32B sits in the optimal region with high performance and low token consumption, demonstrating the best performance-to-efficiency ratio among all compared models.

### 3.3 Code Behavior Analysis between Prompt-Hint and Hint-Engineering

We first establish a taxonomy for Python code usage based on two dimensions. From the perspective of reasoning relationship, we categorize code usage into Calculation (computing results not present in the current reasoning chain) and Verification (validating results derived from chain-of-thought reasoning). In terms of specific functionality, we classify code into categories including Solving Equations, Numerical Approximation, Pattern Recognition, Combinatorial Enumeration and so on.

We conduct a comprehensive analysis of Python code usage patterns across all test sets, comparing Prompt-Hint-SFT-32B and Hint-Engineering-RFT-32B, two models with comparable overall performance. We employ DeepSeek-V3 to classify Python code functionality, with the corresponding classification prompts detailed in Appendix E.

###### Prompt-Hint

Hint-Engineering

100

| |1.3%<br><br>1.0%<br><br>1.0%| | | | | |
|---|---|---|---|---|---|---|
| |4.5%<br><br>2.0%<br><br>1.5%| | | | | |
| |6.5|%| | | | |
| |8|.7%| | | | |
| | |14.0%| | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| |2.8|%| | | | | |
|---|---|---|---|---|---|---|---|
| |3.|5%| | | | | |
| | |5.2%| | | | | |
| | |7.6|%| | | | |
| | |8.1|%| | | | |
| | | |11.7%| | | | |
| | | |13|.3%| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Prompt-Hint

Hint-Engineering

1.6%

Statistics

Data Generation and Simulation

82.4%

1.9%

Probability

Geometric and Computational Geometry Calculations

80

Prime Number Testing and Factorization

Prime Number Testing and Factorization

Combinatorial Enumeration and Game Theory

Pattern Recognition and Analysis

###### Percentage(%)

60

Linear Algebra: Matrix and Vector Operations

Combinatorial Enumeration and Game Theory

51.1%

48.9%

Modular Arithmetic and Number Theory Operations

Modular Arithmetic and Number Theory Operations

40

Symbolic Mathematics and Manipulation

Symbolic Mathematics and Manipulation

Numerical Approximation Methods

Numerical Approximation Methods

20

17.6%

Solving Equations and Systems of Equations

Solving Equations and Systems of Equations

51.0%

29.6%

Property Verification and Theorem Checking

Property Verification and Theorem Checking

- 0

0 10 20 30 40 50

0 5 10 15 20 25 30

Calculation Verification

Percentage (%)

Percentage (%)

(a) (b) (c)

- Figure 5: Analysis of Python code usage patterns. (a) Distribution of code usage types: HintEngineering shows a preference for calculation while Prompt-Hint favors verification tasks. (b) and (c) Function-specific distribution: Hint-Engineering demonstrates more balanced usage across different Python functions compared to Prompt-Hint.

- Figure 5 provides a qualitative analysis of the code behavior patterns in our different approaches. The most striking difference is in how the models utilize code:

- • Prompt-Hint Approach: Code is predominantly used for verification purposes (68.2%), with only 31.8% dedicated to actual computational tasks. This indicates an inefficient utilization pattern where the model performs calculations in natural language and then uses code primarily to verify these calculations.
- • Hint-Engineering Approach: Shows a much more balanced usage, with 51.1% of code dedicated to direct calculation and 48.9% for verification. This more optimal distribution reflects the model’s understanding of when to leverage computational tools versus when to rely on reasoning.

Additionally, the Hint-Engineering approach shows greater diversity in the types of computational operations performed, including symbolic mathematics, equation solving, and combinatorial enumeration. This suggests that the model has developed a more sophisticated understanding of the appropriate use cases for different types of code operations.

As shown in Figure 5, Prompt-Hint and Hint-Engineering exhibit distinct code-integrated Reasoning patterns. Prompt-Hint demonstrates a strong preference for verification (82.4%), while Hint-Engineering maintains a relatively balanced distribution between calculation and verification (approximately 50% each). This balanced distribution emerges from the interplay between our Hint-Engineering design, which encourages computational efficiency, and the model’s inherent tendency toward verification in long chain-of-thought reasoning. Regarding specific mathematical functionalities, we observe that Property Verification and Theorem Checking dominates PromptHint’s code usage (51%), whereas Hint-Engineering exhibits a more uniform distribution across different functions. Interestingly, both approaches share the same top-5 most frequently used Python functions, suggesting the influence of the test sets’ mathematical domains. Moreover, we present representative examples in Appendix F.

- 3.4 Impact of Code Reward in RL

- Figure 6 presents an ablation study on the effect of incorporating code execution reward into the RL process. We set the code reward ratio ω = 0.1 here. The results demonstrate that incorporating this code reward consistently improves performance for both approaches:

- • For Prompt-Hint: Models trained with code reward achieve up to 5% higher accuracy than those without, reaching a peak of 43.1% versus 37.9%.
- • For Hint-Engineering: A similar pattern emerges with approximately 3% performance improvement, reaching 41.0% versus 38.3%.

Notably, we found that the magnitude of this reward is crucial: a modest code reward ratio ω = 0.1 provides optimal results, while stronger penalties (e.g., 0.5) degraded performance. This suggests that while encouraging code correctness is valuable, overly penalizing experimental code attempts

[Figure 5]

- Figure 6: Ablation study on the impact of code execution reward during RL training on AIME24. Left: Performance of Prompt-Hint-1.5B-RL with and without code reward. Right: Performance of Hint-Engineering-1.5B-RL with and without code reward. Both approaches show consistent performance improvements when trained with the additional code execution reward.

[Figure 6]

- Figure 7: Pass@k performance on AIME24 (top) and MATH500-Level5 (bottom) for both PromptHint (left) and Hint-Engineering (right). The analysis compares the base model DeepSeek-R1-1.5B with SFT and RL variants. While SFT does not significantly improve the Pass@k upper bound, RL substantially elevates performance across all k values, particularly at lower sampling budgets.

can inhibit the model’s exploration and learning. Additional experimental results can be found in Appendix D.

### 3.5 Pass@K Analysis

- Figure 7 illustrates the performance of our models as a function of sample size (k) on both AIME24 and MATH500-Level5 datasets. Several important patterns emerge:

- • SFT Impact on Reasoning Ceiling: Supervised fine-tuning alone does not significantly raise the Pass@k upper bound for either approach. This suggests that while SFT can teach the model format and basic tool usage, it doesn’t fundamentally enhance the model’s reasoning capabilities for 1.5B size model with the selected 10k probelms.

- • RL Significantly Raises Performance Ceiling: Both Prompt-Hint-1.5B-RL and Hint-Engineering1.5B-RL show substantially higher Pass@k curves than their SFT counterparts, particularly at lower k values. This indicates that reinforcement learning successfully improves not just the average performance but the model’s ability to consistently arrive at correct solutions with fewer attempts.

These observations confirm that RL effectively amplifying the benefits of the more optimal code usage patterns established during the Hint-Engineering training.

Additional interesting findings, such as the impact of problem difficulty on RL performance and the evolution of code behavior during RL training, are documented in Appendix D.

## 4 Conclusion

Our experiments reveal that properly integrated code tools enhance mathematical reasoning across model scales. High-quality data with optimal code behavior patterns can match or exceed the performance of larger datasets, while reinforcement learning significantly improves performance beyond SFT, particularly for smaller models. The Hint-Engineering approach achieves remarkable efficiency, reducing token usage by 30-50% while maintaining competitive performance. Moreover, RL shapes code usage behavior toward either efficiency or increased integration. These findings demonstrate that combining high-quality data curation, targeted fine-tuning, and reinforcement learning with carefully designed rewards effectively enhances mathematical reasoning capabilities through tool integration.

## References

- [1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [2] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. In Forty-first International Conference on Machine Learning, 2024.
- [3] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [4] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\" ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.
- [5] OpenAI. Learning to reason with llms. https://openai.com/index/learnin g-to-reason-with-llms/, 2024.
- [6] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.
- [7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025.
- [8] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.
- [9] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [10] Siwei Wu, Zhongyuan Peng, Xinrun Du, Tuney Zheng, Minghao Liu, Jialong Wu, Jiachen Ma, Yizhi Li, Jian Yang, Wangchunshu Zhou, et al. A comparative study on reasoning patterns of openai’s o1 model. arXiv preprint arXiv:2410.13639, 2024.

- [11] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.
- [12] Yancheng He, Shilong Li, Jiaheng Liu, Weixun Wang, Xingyuan Bu, Ge Zhang, Zhongyuan Peng, Zhaoxiang Zhang, Zhicheng Zheng, Wenbo Su, et al. Can large language models detect errors in long chain-of-thought reasoning? arXiv preprint arXiv:2502.19361, 2025.
- [13] OpenAI. Introducing openai o3 and o4-mini, 2025. URL https://openai.com/index/int roducing-o3-and-o4-mini/.
- [14] Ziniu Li, Congliang Chen, Tian Xu, Zeyu Qin, Jiancong Xiao, Zhi-Quan Luo, and Ruoyu Sun. Preserving diversity in supervised fine-tuning of large language models. In The Thirteenth International Conference on Learning Representations, 2025.
- [15] Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023. URL https://arxiv.org/abs/2308.01825.
- [16] Zhipeng Chen, Yingqian Min, Beichen Zhang, Jie Chen, Jinhao Jiang, Daixuan Cheng, Wayne Xin Zhao, Zheng Liu, Xu Miao, Yang Lu, et al. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548, 2025.
- [17] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023.
- [18] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.
- [19] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv:2502.03387, 2025.
- [20] Chengpeng Li, Mingfeng Xue, Zhenru Zhang, Jiaxi Yang, Beichen Zhang, Xiang Wang, Bowen Yu, Binyuan Hui, Junyang Lin, and Dayiheng Liu. Start: Self-taught reasoner with tools. arXiv preprint arXiv:2503.04625, 2025.
- [21] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.
- [22] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang,

- X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei,
- Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng

Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou,

- Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report,

#### 2025. URL https://arxiv.org/abs/2412.19437.

- [23] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.
- [24] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.
- [25] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.
- [26] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403, 2022.
- [27] Aske Plaat, Annie Wong, Suzan Verberne, Joost Broekens, Niki van Stein, and Thomas Back. Reasoning with large language models, a survey. arXiv preprint arXiv:2407.11511, 2024.
- [28] Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, et al. Towards large reasoning models: A survey of reinforced reasoning with large language models. arXiv preprint arXiv:2501.09686, 2025.
- [29] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [30] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: a theoretical perspective. Advances in Neural Information Processing Systems, 36:70757–70798, 2023.
- [31] Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.
- [32] Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.
- [33] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2023.
- [34] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.
- [35] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

- [36] Richard Yuanzhe Pang, Weizhe Yuan, He He, Kyunghyun Cho, Sainbayar Sukhbaatar, and Jason Weston. Iterative reasoning preference optimization. Advances in Neural Information Processing Systems, 37:116617–116637, 2024.
- [37] Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023.
- [38] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [39] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/abs/2501.19393.
- [40] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.
- [41] Huggingface. Open r1, 2025. URL https://github.com/huggingface/open-r1.
- [42] Yuxiang Zhang, Shangxi Wu, Yuqi Yang, Jiangming Shu, Jinlin Xiao, Chao Kong, and Jitao Sang. o1-coder: an o1 replication for coding, 2024. URL https://arxiv.org/abs/2412.0 0154.
- [43] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https: //arxiv.org/abs/2503.14476.
- [44] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.
- [45] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.
- [46] Ruida Wang, Jipeng Zhang, Yizhen Jia, Rui Pan, Shizhe Diao, Renjie Pi, and Tong Zhang. Theoremllama: Transforming general-purpose llms into lean4 experts. arXiv preprint arXiv:2407.03203, 2024.
- [47] Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan Liang. Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data. arXiv preprint arXiv:2405.14333, 2024.
- [48] Kefan Dong and Tengyu Ma. Beyond limited data: Self-play llm theorem provers with iterative conjecturing and proving. arXiv preprint arXiv:2502.00212, 2025.
- [49] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.
- [50] Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh RN, et al. Apigen: Automated pipeline for generating verifiable and diverse function-calling datasets. Advances in Neural Information Processing Systems, 37: 54463–54482, 2024.
- [51] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. Frontiers of Computer Science, 19(8):198343, 2025.

- [52] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2503.05592.
- [53] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models, 2025. URL https://arxiv.org/abs/2501.05366.
- [54] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL https://arxiv.org/abs/2302.04761.
- [55] Yusen Zhang, Ruoxi Sun, Yanfei Chen, Tomas Pfister, Rui Zhang, and Sercan Arik. Chain of agents: Large language models collaborating on long-context tasks. Advances in Neural Information Processing Systems, 37:132208–132237, 2024.
- [56] Yao Zhang, Hongxiao Zhang, Jiacheng Zhang, Jingcheng Zhao, Rui Yan, Xiaoqing Liu, Jiahuan Wang, Min Zhang, Houfeng Wang, and Zhengguang Guo. rStar-Math: Small LLMs can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2403.01707, 2024.
- [57] Haozhe Wang, Long Li, Chao Qu, Fengming Zhu, Weidi Xu, Wei Chu, and Fangzhen Lin. Learning autonomous code integration for math language models, 2025. URL https://arxi v.org/abs/2502.00691.
- [58] Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, and Wenqiang Zhang. Agent rl scaling law: Agent rl with spontaneous code execution for mathematical problem solving,

2025. URL https://arxiv.org/abs/2505.07773.

- [59] Kezhou Wang, Ruijie Wu, Qinlin Zeng, Huao Lu, Hanye Wu, Qingfeng Cui, Haichao Lin, Yujia Liu, Xiaoyan Huang, Qingpeng Guo, Songtao Jian, Kaiyuan Lu, Shiyu Li, Hao Tian, Yongqin Sun, Xue Yang, Libin Song, Zejun Ou, and Guoqing Wang. ToRL: Scaling tool-integrated RL for LLMs. arXiv preprint arXiv:2312.10372, 2023.
- [60] Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Otc: Optimal tool calls via reinforcement learning. arXiv preprint arXiv:2504.14870, 2025.
- [61] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025. URL https://arxiv.org/abs/2504.11536.
- [62] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. RAFT: reward ranked finetuning for generative foundation model alignment. Trans. Mach. Learn. Res., 2023.
- [63] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [64] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface. co/AI-MO/NuminaMath-1.5](https://github.com/project-numina/aimo-progres s-prize/blob/main/report/numina_dataset.pdf), 2024.
- [65] Hynek Kydlíˇcek. Math-verify: Math verification library, 2024. URL https://github.com/h uggingface/math-verify.
- [66] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/D eepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681 902c1468005bed8ca303013a4e2, 2025. Notion Blog.

- [67] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- [68] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks, 2021.
- [69] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://github.com/pro ject-numina/aimo-progress-prize](https://github.com/project-numina/aimo

#### -progress-prize/blob/main/report/numina_dataset.pdf), 2024.

## A Related Work

- A.1 Reasoning in LLMs

The evolution of reasoning capabilities in LLMs has progressed rapidly in recent years [23–25, 3, 5, 7, 13]. For comprehensive coverage of this field, we direct readers to recent surveys [26–28].

- A pivotal technique in this field is Chain-of-Thought reasoning [9], which, when combined with Transformer architectures [29], enables models to perform complex computational tasks [30]. This field has benefited substantially from scaling up synthetic data [31–33] and has evolved toward training directly from RL using pre-trained models [7, 34, 35]. These advances leverage various approaches, including preference optimization [36], Monte Carlo Tree Search [37], and advanced RL techniques [1–3]. Notably, following OpenAI’s o1 [38], there has been a significant trend towards long-form Chain-of-Thought reasoning, incorporating human-like cognitive patterns such as multiple reflections, task decomposition, and strategic exploration and these LLMs are often called large reasoning models(LRMs). This integration of human-inspired reasoning approaches has led to substantial improvements in complex reasoning tasks, particularly in coding and mathematics, where models have demonstrated unprecedented capabilities in systematic problem-solving [39–43]. While natural language remains the primary medium for reasoning in this domain, there is growing interest in integrating external tools such as code interpreters [44, 45, 11] and automatic verification systems [46–48] to overcome the inherent limitations of natural language reasoning, such as accurate computation.

A.2 Code-Integrated Reasoning for LLMs

Recent research has explored integrating external tools with language models to enhance their reasoning capabilities [49–54]. ToRA [11] pioneered code-integrated reasoning specifically for mathematical problem-solving, demonstrating that offloading complex calculations to specialized systems significantly improves performance. Since then, COA [55] trains LLMs to decode reasoning chains with abstract placeholders, and then call domain tools to reify each reasoning chain by filling in specific knowledge. rStar-Math [56] introduced a code-augmented Chain of Thought data synthesis method through Monte Carlo Tree Search. Recent works [57–60] have initiated investigations into frameworks that enable base models to autonomously develop code-integrated reasoning capabilities for mathematical problem-solving. With the advancement of Large Reasoning Models (LRMs), the organic integration of code interpreters within long-form Chain-of-Thought reasoning has emerged as a crucial research challenge. Several concurrent works have explored this direction. START [20], which shares similarities with our approach, introduces hints to guide code generation within large reasoning modes; however, their random hint insertion may lead to suboptimal utilization of code interpreters. STILL3 [16] employs prompting techniques to construct code-integrated data, similar to our proposed Prompt-Hint method, but does not address reasoning efficiency. Retool [61] attempts to bootstrap training data by rewriting long Chain-of-Thought reasoning, yet shows limited performance improvements when based on DeepSeek-R1-Distill-Qwen-32B. While OTC [60] considers efficiency from the perspective of tool call frequency, it does not explore methods for enhancement building upon existing LRMs. CoRT proposed a highly sample-efficient approach that achieved both performance breakthroughs and significant improvements in reasoning efficiency. Through human-in-the-loop annotation of 30 high-quality samples, combined with techniques such as RFT [15, 62] and RL, they demonstrated that substantial improvements in both reasoning capabilities and efficiency can be achieved with minimal high-quality training data.

- B Experiment Implementation

- B.1 Training Implementation

For our experiments, we implemented several model variants with different training stages and architectures: 32B Models:

- • Prompt-Hint-SFT-32B: Starting from the DeepSeek-R1-32B base model, we fine-tuned using 800 data samples with a learning rate of 1 × 10−5, running for 17 epochs with a batch size of 96.

- • Hint-Engineering-SFT-32B: Based on DeepSeek-R1-32B, we fine-tuned using only 30 high-quality, human-annotated data samples with a batch size of 96, learning rate of 1×10−5, and 40 epochs.
- • Hint-Engineering-RFT-32B: Building upon Hint-Engineering-SFT-32B, we further finetuned using 800 filtered data samples with a learning rate of 1 × 10−5, 17 epochs, and batch size of 96.

### 1.5B Models:

- • We distilled both Prompt-Hint-SFT-32B and Hint-Engineering-RFT-32B down to the DeepSeek-R1-1.5B architecture using 10k data samples with a learning rate of 7 × 10−6, 6 epochs, and batch size of 128.
- • For reinforcement learning, we adapted the veRL framework [63] to implement our specialized design outlined in Section 2.4. We further trained these 1.5B models with a learning rate of 1 × 10−6, maximum response length of 16,000 tokens, 8 rollouts per problem, and maximum function calls limited to 15 per response, with each function call having a maximum length of 16,000 tokens.
- • The RL training data was carefully selected by computing the average accuracy over 8 samples (avg@8) on 20k randomly selected problems from the NuminaMath-1.5 [64] dataset, then selecting only 1k challenging problems where avg@8 = 1/8 for focused training. This selective approach is motivated by our data ablation studies (Appendix D.2), which demonstrate that training on hard queries, while requiring longer convergence time, ultimately yields superior performance compared to easy or uniformly distributed queries.

### B.2 Evaluation Methodology

To ensure comprehensive and fair comparisons across different approaches, we implemented the following evaluation protocol:

- • Fair Comparison: For publicly available models, we re-evaluated them on our local infrastructure using their original evaluation scripts to ensure consistent comparison conditions across all models.
- • Evaluation Protocol: For all datasets, we extract the final answer from each model response and compare it directly to the ground truth using Math-Verify [65], considering a problem correctly solved only when Math-Verify returns True.
- • Evaluation Metrics: We primarily used pass@1 as our base metric. Concretely, we employed avg@16 (average accuracy over 16 samples) as pass@1 for AIME24, AIME25, and AMC23 datasets. For MATH500 and OlympiadBench datasets, we used avg@4 as pass@1 due to their significantly larger test sizes.
- • Inference Setting: Across all evaluations, we standardized inference parameters with maximum sequence length of 32,768 tokens, maximum function calls limited to 15, maximum tokens per function call set to 32,768, temperature of 0.6, and top-p sampling parameter of 0.95.

### B.3 Function Call Limiting Strategy

During both evaluation and reinforcement learning rollout sampling with Python usage, we implemented a mechanism to handle scenarios where models reached the maximum allowed function calls. When this limit was reached, we appended the following system message to guide further reasoning:

[SYSTEM] You have exceeded the allowed number of code executions. You can no longer write or run code. Please continue solving the problem using your reasoning and analytical skills.

This approach ensures that models can still complete their reasoning process without unlimited computational resources, better reflecting real-world usage constraints.

- B.4 Computational Hardware Our experiments utilized the following hardware:

- • Training: All training procedures, including supervised fine-tuning (SFT), rejection finetuning (RFT), and reinforcement learning (RL), were conducted on 4 servers, each equipped with 8 NVIDIA A100 GPUs.
- • Evaluation: All model evaluations were performed on single servers, each equipped with 8 NVIDIA A100 GPUs, ensuring consistent measurement conditions across all compared approaches.

## C Evaluation Dataset Description

To comprehensively assess the mathematical reasoning capabilities of our models, we utilize several challenging benchmarks that span diverse difficulty levels and mathematical domains:

AIME24 and AIME25. The American Invitational Mathematics Examination (AIME) represents a significant advancement beyond standard high school mathematics competitions, featuring problems that demand sophisticated reasoning techniques. We employ AIME24 and AIME25 as our primary benchmarks for evaluating advanced mathematical reasoning capabilities in our models.

AMC23. Following DeepScaleR [66], we utilize their American Mathematics Competition (AMC) test set, which presents problems of moderate yet substantial difficulty, requiring considerable mathematical insight to solve. This dataset enables us to evaluate our models’ proficiency in addressing a wider spectrum of mathematical challenges typically encountered in standard high school competitions.

MATH500. Curated from the test split of OpenAI’s PRM800K dataset [25], MATH500 encompasses 500 carefully selected problems that represent a diverse range of mathematical challenges. We utilize this dataset to evaluate our models’ ability to generalize across varied mathematical topics and problem structures.

OlympiadBench. Following DeepScaleR [66], we incorporate their OlympiadBench [67] into our evaluation framework. This benchmark comprises 675 Olympiad-level problems sourced from elite mathematics competitions, providing an exceptionally rigorous test of advanced mathematical reasoning.

## D Extra Experiments

### D.1 Code Behavior Evolution During RL

[Figure 7]

Figure 8: Evolution of code behavior metrics during RL training on AIME24.

- Figure 8 tracks six key metrics of code behavior throughout the RL training process:

- • Code Usage Rate: The percentage of responses containing Python code out of all responses. Both approaches show increasing code usage rates during training, with Hint-Engineering consistently maintaining higher rates, starting at 86% and quickly rising above 95%.
- • Code Success Rate: The percentage of code blocks that execute without errors. The success rate shows different patterns between the two approaches. Prompt-Hint maintains relatively stable success rates around 78%, while Hint-Engineering shows more variability but achieves higher peaks.
- • Average Code Blocks: The average number of Python code blocks per response. Interestingly, Hint-Engineering shows a steady decrease in the average number of code blocks from 4.4 to around 3.5 at peak performance, while Prompt-Hint increases from 1.9 to around 2.7. This divergence reveals a fundamental difference in evolution: Hint-Engineering evolves toward more efficient code usage (fewer but more effective blocks), while Prompt-Hint develops more code integration capabilities from its lower starting point.

These patterns reveal that reinforcement learning not only improves raw performance but actively shapes code usage behavior, with Hint-Engineering evolving toward an efficiency-optimized pattern (less code but more effective) while Prompt-Hint evolves toward increased code integration (more code with improving effectiveness).

### D.2 RL Training Data Ablation

To understand the impact of training data characteristics on RL performance, we conduct three ablation studies examining data volume, query difficulty distribution, and topic distribution effects.

[Figure 8]

- Figure 9: RL training data ablation studies on AIME24. Left: Data scaling ablation with 1K, 5K, and 20K training samples under uniform difficulty and topic distributions. Middle: Query difficulty ablation comparing easy (avg@8=7/8), uniform, and hard (avg@8=1/8) difficulty distributions with 1K samples. Right: Topic distribution ablation comparing uniform and hard topic distributions with 1K samples under uniform difficulty.

Data Scaling Ablation (Figure 9, left): We investigate whether increasing training data volume directly improves optimal performance by comparing 1K, 5K, and 20K training samples while maintaining uniform difficulty and topic distributions. Surprisingly, we find that simply scaling up RL training data does not lead to better optimal performance. This finding suggests that the "less is more" principle still holds in large reasoning model (LRM) training, and data quality may be more important than quantity for RL fine-tuning.

Query Difficulty Ablation (Figure 9, middle): We examine how query difficulty distribution affects learning dynamics by comparing three settings with 1K samples: easy queries (avg@8=7/8), uniform difficulty distribution, and hard queries (avg@8=1/8). Our results reveal distinct learning patterns: easy queries achieve optimal performance earliest but with lower peak accuracy, uniform distribution shows intermediate behavior, while hard queries take longer to reach optimal performance but ultimately achieve the best results. This suggests that training on challenging examples, though slower to converge, leads to superior final performance in mathematical reasoning tasks.

Topic Distribution Ablation (Figure 9, right): We investigate whether aligning training topic distribution with hard query topics improves performance by comparing uniform topic distribution

against a distribution matching hard queries (avg@8=1/8). The results show minimal differences between the two distributions, indicating that topic distribution changes do not significantly impact optimal performance. This suggests that the model’s reasoning capabilities generalize well across different mathematical topics, and the difficulty level is more crucial than specific topic coverage.

### D.3 Code Reward Ablation

To investigate the effectiveness of our code reward mechanism and determine the optimal penalty strength, we conduct ablation studies comparing different penalty values in the code reward function.

[Figure 9]

Figure 10: Code reward penalty ablation study on AIME24. Left: Comparison between training with and without code reward using penalty=0.1. Right: Comparison between training with and without code reward using penalty=0.5 on a different SFT model.

Code Reward Effectiveness: Our results demonstrate that incorporating code reward significantly improves model performance compared to training without it. In the penalty=0.1 setting (Figure 10, left), the model with code reward achieves a peak accuracy of 43.1% at step 100, substantially outperforming the baseline without code reward (37.9% peak). This improvement persists throughout most of the training process, indicating that the code reward provides consistent learning signals that guide the model toward better reasoning strategies.

Penalty Strength Analysis: Comparing different penalty values reveals that penalty=0.1 yields superior and more stable performance than penalty=0.5. The penalty=0.1 configuration shows smoother convergence and maintains higher accuracy levels across training steps. In contrast, penalty=0.5 (Figure 10, right) exhibits more erratic behavior, with a notable spike at step 60 (41.4%) followed by a sharp decline. This suggests that excessive penalty strength may introduce instability in the training process, potentially causing the model to over-correct its behavior when generating incorrect code.

### D.4 Token Efficiency Analysis for Lightweight Models

We conduct a detailed analysis of token efficiency in 1.5B parameter lightweight models, examining how different approaches impact computational resource utilization while maintaining mathematical reasoning capabilities.

Our token efficiency analysis reveals two key insights about the computational efficiency of lightweight mathematical reasoning models:

Superior Performance with Limited Token Budgets: As shown in Figure 11 (left), HintEngineering-SFT consistently outperforms both the base model (DeepSeek-R1-1.5B) and PromptHint-SFT across all token budget constraints. Most notably, with just a 4k token budget, HintEngineering-SFT achieves 22.7% accuracy, nearly double the performance of DeepSeek-R1-1.5B (12.1%) and Prompt-Hint-SFT (11.0%). This indicates that Hint-Engineering’s structured approach to problem-solving enables more efficient reasoning within constrained computational environments.

[Figure 10]

- Figure 11: Token efficiency analysis for 1.5B parameter models on AIME24. Left: Performance comparison across different token budgets. Right: Detailed token usage breakdown for each model, showing average token consumption for both correct (above axis) and incorrect (below axis) responses, with Pass@1 accuracy displayed at the center.

Substantial Token Savings Across All Response Types: Figure 11 (right) demonstrates that Hint-Engineering models maintain significantly lower token usage compared to alternatives. For correct responses, Hint-Engineering-SFT uses 47% fewer tokens than Prompt-Hint-SFT (4,711 vs.

- 8,862), while for incorrect responses, the savings are even more dramatic, with Hint-Engineering-RL consuming 31% fewer tokens than Prompt-Hint-RL (8,485 vs. 12,292). Overall, Hint-EngineeringRL achieves a 32% reduction in total token consumption compared to Prompt-Hint-RL (6,684 vs.
- 9,891) while maintaining comparable accuracy (41.0% vs. 43.1%). Furthermore, Hint-Engineering models use about 50% fewer tokens for the 1.5B model compared with the natural language models.

- D.5 Pass@K Analysis for Frontier Models

To understand how our approaches scale with sampling budget and model size, we conduct a comprehensive Pass@k analysis on 32B parameter frontier models.

- Figure 12 illustrates the performance of our 32B parameter models as a function of sample size (k) on both AIME24 and MATH500-Level5 datasets. Our analysis reveals two key patterns in the scaling behavior of these models.

SFT Provides Modest Gains at Lower Sampling Budgets: Both Prompt-Hint-32B-SFT and HintEngineering-32B-SFT show improvements over the baseline DeepSeek-R1-32B, particularly at lower k values. At k=1 on AIME24, Prompt-Hint-32B-SFT achieves 77.3% accuracy compared to the baseline’s 72.9%. However, all models eventually converge to similar maximum performance levels (93.3% for AIME24 and 99.2% for MATH500-Level5) as k increases, suggesting that SFT primarily enhances the model’s efficiency rather than raising its reasoning ceiling.

RFT Significantly Enhances Sample Efficiency: Hint-Engineering-32B-RFT demonstrates remarkable efficiency, reaching 93.3% accuracy with just k=8 samples on AIME24, while other approaches require 2-4 times more samples to achieve comparable results. This indicates that reinforcement fine-tuning successfully optimizes the model’s ability to consistently arrive at correct solutions with fewer attempts, making it particularly valuable in scenarios where computational efficiency is critical.

[Figure 11]

Figure 12: Pass@k analysis for 32B parameter models. Top row: Performance on AIME24 comparing Prompt-Hint (left) and Hint-Engineering (right) approaches against the DeepSeek-R1-32B baseline. Bottom row: Performance on MATH500-Level5 showing similar patterns.

## E Prompts

Hint-Prompt and Hint-Engineering Prompt

Given a mathematical problem, follow the instructions below to solve it. Instructions: When solving mathematical problems, you should leverage both natural language reasoning and interactive Python code execution. Your goal is to provide clear, detailed explanations while utilizing Python to perform complex calculations, symbolic manipulations, data analysis, or any other tasks that can aid in problem-solving. Follow these guidelines to ensure a coherent and effective response:

### 1. Natural Language Reasoning:

- - Provide comprehensive, step-by-step explanations of your thought process.
- - Ensure that each step logically follows from the previous one, maintaining clarity and coherence.
- - Use appropriate mathematical terminology and notation where necessary.
- - Planning, Modeling, and Analysis:

- • Use natural language to outline the overall approach to the problem.
- • Develop mathematical models or representations as needed.
- • Analyze the problem to determine the best strategies for finding a solution.

### 2. Inserting Python Code Blocks:

- - When a Python code snippet can aid in analysis, computation, or symbolic manipulation, insert a Python code block.
- - Use triple backticks with ‘python‘ to denote the start of a Python code block and triple backticks to close it.
- - Example: ”’python

”’

### 3. Displaying Code Output:

- - Immediately after a Python code block, present the output generated by the code.
- - Use triple backticks with ‘output‘ to denote the start of the output block and triple backticks to close it.
- - Example: ”’output ”’ 4. Encouraging Multiple Python Calls and Diverse Functionality:
- - Utilize Python multiple times throughout your solution to handle different aspects of the problem.
- - Take advantage of various Python libraries and functionalities such as:
- - ‘numpy‘ for numerical computations
- - ‘scipy‘ for scientific computing and advanced mathematical functions
- - ‘sympy‘ for symbolic mathematics
- - ‘pandas‘ for data manipulation and analysis
- - ‘math‘ for fundamental mathematical operations
- - ‘statistics‘ for statistical computations
- - ‘fractions‘ for rational number calculations
- - Ensure that each Python snippet is purposeful and enhances the understanding or resolution of the problem.
- - Specific Calculations and Complex Operations:
- - Use Python to perform detailed calculations that would be cumbersome by hand.
- - Implement complex algorithms or data processing tasks that facilitate the solution.
- - Handle any intricate operations that support the overall analysis and modeling of the problem Problem:

Code Behavior Analysis Prompt

You are an expert in analyzing code and understanding its purpose, especially within the context of mathematical problem-solving. Your task is to analyze Python code snippets within a solution to a mathematical problem and classify each snippet based on its purpose.

You will be given a problem/solution pair. The solution may contain multiple Python code snippets. For each Python code snippet, you must determine:

- 1. Is it Verification or Calculation?

- - Verification: The Python code verifies a result or conclusion that was already reached through reasoning in the solution. The Python code confirms a pre-existing answer or property.
- - Calculation: The Python code calculates a result that was NOT explicitly present in the solution’s reasoning up to that point. The Python code derives a new, previously unknown answer or intermediate value.

- 2. What is the specific function of the Python code snippet? Choose one or more from the following list of functions (be as specific as possible). If none of these functions are appropriate, provide a brief (one sentence) description of the function of the code.

### 1. Solving Equations and Systems of Equations

- Finding numerical or symbolic solutions to algebraic, differential, and other types of equations.

### 2. Symbolic Mathematics and Manipulation

- Performing algebraic operations such as differentiation, integration, simplification, and

expansion using symbolic math libraries like SymPy.

### 3. Numerical Approximation Methods

- Approximating solutions for problems that lack analytical solutions, including numerical integration, root finding, and solving differential equations.

### 4. Data Visualization and Plotting

- Creating graphs, charts, and other visual representations using libraries like Matplotlib and Seaborn to illustrate mathematical concepts and data patterns.

### 5. Pattern Recognition and Analysis

- Identifying and analyzing patterns or relationships in data using statistical and machine learning techniques.

### 6. Optimization and Solution Searching

- Implementing algorithms to find optimal solutions to problems, including linear programming, integer programming, and heuristic methods.

### 7. Property Verification and Theorem Checking

- Verifying mathematical properties and theorems for given inputs using computational methods.

### 8. Modular Arithmetic and Number Theory Operations

- Performing calculations involving modular arithmetic, such as finding inverses, solving congruences, and applying the Chinese Remainder Theorem.

### 9. Prime Number Testing and Factorization

- Determining the primality of numbers and performing prime factorization using efficient algorithms.

### 10. Geometric and Computational Geometry Calculations

- Calculating areas, volumes, distances, angles, convex hulls, intersections, and performing geometric transformations.

### 11. Probability, Statistics, and Simulations

- Computing probabilities, expected values, variances, and running Monte Carlo simulations to model random processes.

### 12. Linear Algebra: Matrix and Vector Operations

- Performing matrix multiplication, inversion, eigenvalue decomposition, and other linear algebra operations using libraries like NumPy and SciPy.

### 13. Data Generation and Simulation

- Creating synthetic data sets and simulating mathematical models to explore and analyze behaviors.

### 14. Combinatorial Enumeration and Game Theory

- Counting permutations, combinations, and analyzing combinatorial games to determine winning strategies.

### 15. Graph Theory Algorithms

- Implementing algorithms for graph traversal (DFS, BFS), shortest paths (Dijkstra’s, Floyd-Warshall), and finding minimum spanning trees (Kruskal’s, Prim’s).

### 16. Dynamic Programming and Recurrence Relations

- Designing dynamic programming solutions and solving linear and non-linear recurrence

relations to find closed-form expressions.

### 17. Fast Fourier Transforms (FFT) and Signal Processing

- Utilizing FFT for problems involving polynomial multiplication, number-theoretic transforms, and analyzing frequency components.

### 18. Boolean Algebra and Logic Operations

- Manipulating and simplifying logical expressions, constructing truth tables, and solving Boolean equations.

### 19. Big Integer and Arbitrary-Precision Arithmetic

- Handling calculations with very large integers beyond standard data type limits using Python’s arbitrary-precision capabilities.

### 20. Symbolic Integration, Differentiation, and Proof Verification

- Performing advanced calculus operations and assisting in verifying mathematical proofs using symbolic computation libraries.

### 21. Linear Programming and Optimization Techniques

- Formulating and solving linear optimization problems using libraries like PuLP and SciPy.

### 22. Algorithm Optimization and Numerical Stability

- Enhancing algorithm performance by improving time and space complexity and ensuring numerical stability for accurate results.

### 23. Automated Theorem Proving and Symbolic Logic

- Utilizing tools and libraries to automatically prove mathematical theorems and manipulate symbolic logic statements.

### 24. Data Structures Implementation and Management

- Creating and utilizing advanced data structures such as trees, graphs, and heaps to efficiently solve competition problems.

### 25. Fractal and Recursive Pattern Generation

- Creating and analyzing fractals and other recursive patterns relevant to geometry and combinatorics problems.

You will be given a "Problem" and a "Solution". The solution may contain one or more "Python code" snippets. For each Python code snippet, provide the following output:

Python Code idx: [The idx of Python code](for example 1,2,3) Classification: [Verification or Calculation] Function: [one or more specific functions from the list above OR some one-sentence descriptions]

Problem: Solution:

## F Case Study

#### F.1 5 Samples of DHint−engineering−SFT

- Sample 1

Problem: Find the least positive integer n for which 2n + 5n − n is a multiple of 1000. Solution: Let’s solve this step by step. The problem is to find the least positive integer n such that 2n + 5n − n ≡ 0 (mod 1000). Since 1000 = 8 × 125, by the Chinese Remainder Theorem, this is equivalent to solving:

2n + 5n − n ≡ 0 (mod 8) 2n + 5n − n ≡ 0 (mod 125)

For modulo 8: - When n ≥ 3: 2n ≡ 0 (mod 8) - 5n (mod 8) cycles every 2: 5,1,5,1,... After analysis, we find n ≡ 5 (mod 8) For modulo 125, let’s first find the period of 2n mod 125:

Python Code 1 def find_period_mod125():

seen_values = {} current_value = 1 position = 0 sequence = []

while True:

if current_value in seen_values: period_length = position - seen_values[current_value] period_sequence = sequence[seen_values[current_value]... return period_length, period_sequence

seen_values[current_value] = position sequence.append(current_value) current_value = (current_value * 2) % 125 position += 1

period_length, period_sequence = find_period_mod125() print(f"Period length: {period_length}")

print("\nExample cases:") test_numbers = [0, 1, 2, period_length-1] for n in test_numbers:

result = pow(2, n, 125) print(f"2^{n} mod 125 = {result}")

Output 1 Period length: 100 Example cases:

- 2^0 mod 125 = 1
- 2^1 mod 125 = 2
- 2^2 mod 125 = 4 2^99 mod 125 = 63

Similarly, let’s check the period of 6k mod 125: Python Code 2 def find_period_mod125():

seen_values = {} current_value = 1 position = 0 sequence = []

while True:

if current_value in seen_values: period_length = position - seen_values[current_value] period_sequence = sequence[seen_values[current_value]... return period_length, period_sequence

seen_values[current_value] = position sequence.append(current_value) current_value = (current_value * 6) % 125 position += 1

period_length, period_sequence = find_period_mod125() print(f"Period length: {period_length}")

print("\nExample cases:") test_numbers = [0, 1, 2, period_length-1] for n in test_numbers:

result = pow(6, n, 125) print(f"6^{n} mod 125 = {result}")

Output 2 Period length: 25 Example cases:

- 6^0 mod 125 = 1
- 6^1 mod 125 = 6
- 6^2 mod 125 = 36 6^24 mod 125 = 21

[...detailed analysis omitted...] After analysis, we need to solve:

#### 32 · 6k ≡ 8k + 5 (mod 125) Let’s find the solution:

Python Code 3 def find_k():

for k in range(0, 150): mod_6k = pow(6, k, 125) lhs = (32 * mod_6k) % 125 rhs = (8*k + 5) % 125 if lhs == rhs:

print(f"Found k={k}, n=8*{k}+5={8*k+5}") return 8*k+5

return None

result = find_k() print(result)

Output 3

Found k=99, n=8*99+5=797 797

Therefore, k = 99 gives us n = 8 × 99 + 5 = 797. Verification: - 5797 ≡ 125 (mod 1000) (since 5n cycles 125, 625 for odd/even n ≥ 3) 2797 ≡ 672 (mod 1000) - Therefore, 2797 + 5797 − 797 ≡ 0 (mod 1000) The answer is

|797|
|---|

.

- Sample 2

### Problem:

Find the largest prime number p < 1000 for which there exists a complex number z satisfying: (1) the real and imaginary part of z are both integers; (2) |z| = √p, and (3) there exists a triangle whose three side lengths are p, the real part of z3, and the imaginary part of z3.

### Solution:

Let’s try to solve this problem step by step. The goal is to find the largest prime number p < 1000 such that there exists a complex number z with integer real and imaginary parts, |z| = √p, and there’s a triangle with side lengths p, Re(z3), and Im(z3).

First, let’s understand the problem. We need to find a prime p < 1000. The complex number z must have integer real and imaginary parts, so z = a + bi where a and b are integers. The magnitude |z| is √a2 + b2 = √p, which implies that a2 + b2 = p. Therefore, p must be a prime that can be expressed as the sum of two squares. That’s a key point.

[...detailed analysis omitted...] So the algorithm we will implement is: For each prime p (starting from largest to smallest):

- • Find (a,b) using find_a_b(p)
- • Generate all variations: [(a,b),(−a,b),(a,−b),(−a,−b),(b,a),(−b,a),(b,−a),(−b,−a)]
- • For each variation (x,y):

- – Compute Re = x3 − 3xy2
- – Compute Im = 3x2y − y3
- – If Re > 0 and Im > 0:

- * Check triangle inequalities: · p + Re > Im · p + Im > Re · Re + Im > p

- * If all three are true, then return p as a valid solution

Let’s implement this in Python: Python Code import math def sieve(n):

sieve = [True] * (n + 1) sieve[0] = sieve[1] = False for i in range(2, int(math.sqrt(n)) + 1):

if sieve[i]:

sieve[i * i : n + 1 : i] = ... primes = [i for i, is_prime in enumerate(sieve) if is_prime] return primes

def find_a_b(p):

max_a = int(math.isqrt(p)) for a in range(max_a + 1):

b_sq = p - a * a if b_sq < 0:

continue b = int(math.isqrt(b_sq)) if b * b == b_sq:

return (a, b) return None

def is_triangle(p, re, im): return (re + im > p) and (p + re > im) and (p + im > re)

def check_primes_for_triangle(): eligible_primes = ... eligible_primes.sort(reverse=True)

for p in eligible_primes: ab = find_a_b(p) if ab is None:

continue a, b = ab variations = [

- (a, b), (-a, b), (a, -b), (-a, -b),
- (b, a), (-b, a), (b, -a), (-b, -a)

] for x, y in variations:

re = x ** 3 - 3 * x * y ** 2 im = 3 * x ** 2 * y - y ** 3 if re > 0 and im > 0:

if is_triangle(p, re, im): print(f"Found valid prime ...") print(f"Checking triangle inequalities: ...") print(f"Answer: {p}") return

check_primes_for_triangle()

Found valid prime 349 with (x,y)=(-5,-18), re=4735, im=4482 Checking triangle inequalities: True, True, True Answer: 349

|349|
|---|

Therefore, the largest prime that satisfies all conditions is

.

- Sample 3 Problem:

Each√21 andface√of31two. Thenoncongruentratio of theparallelepipedsvolume of the islargera rhombusof the whosetwo polyhedradiagonalstohavethe volumelengths of the smaller is mn , where m and n are relatively prime positive integers. Find m + n. A parallelepiped is a solid with six parallelogram faces such as the one shown below.

### Solution:

Let’s try to solve this problem step by step. The problem is about two noncongruent parallelepipeds where each face is a rhombus with diagonals of lengths √21 and √31. We need to find the ratio of their volumes (larger to smaller) and then compute m + n where the ratio is

- m

- n in simplest terms. [...detailed analysis omitted...] Therefore, the two possible rhombohedrons would have all angles equal to arccos(5/26) or arccos(−5/26). Thus, these are two distinct rhombohedrons with the same edge length but different angles, hence noncongruent. Let’s compute the volumes using Python:

Python Code import sympy as sp # Define the square root of 13 symbolically s = sp.sqrt(13) # Define angles symbolically using arccos theta = sp.acos(sp.Rational(5, 26)) theta_prime = sp.acos(sp.Rational(-5, 26)) # Define cos(theta) and cos(theta_prime) as symbolic fractions cos_theta = sp.Rational(5, 26) cos_theta_prime = sp.Rational(-5, 26) # Compute volume for theta using the symbolic expression vol1_expr = s**3 * sp.sqrt(1 - 3*cos_theta**2 + 2*cos_theta**3) vol1 = sp.simplify(vol1_expr) # Compute volume for theta_prime using the symbolic expression vol2_expr = s**3 * sp.sqrt(1 - 3*cos_theta_prime**2 + ... vol2 = sp.simplify(vol2_expr) # Print the symbolic results

- print("Volume 1:", vol1)
- print("Volume 2:", vol2)

- Volume 1: 63*sqrt(2)/2
- Volume 2: 31*sqrt(2)

√2 2 and 31√2. Since 632 = 31.5 > 31, 63

√2 2 is larger than 31√2.

The two volumes are 63

Therefore, mn = 6362. Since 63 = 7 × 9 and 62 = 2 × 31, they are coprime. So m = 63, n = 62, and m + n = 125.

|125|
|---|

Therefore, the answer is

.

- Sample 4

### Problem:

Let △ABC be an equilateral triangle with side length 55. Points D, E, and F lie on BC, CA, and AB, respectively, with BD = 7, CE = 30, and AF = 40. Point P inside △ABC has the property that

#### ∠AEP = ∠BFP = ∠CDP.

Find tan2(∠AEP). Solution: Let’s try to solve this problem step by step. We have an equilateral triangle ABC with side length 55. Points D, E, F are on BC, CA, AB respectively, with BD=7, CE=30, AF=40. There’s a point P inside the triangle such that the angles ∠AEP, ∠BFP, and ∠CDP are equal. We need to find the square of the tangent of this angle. First, let’s place the triangle ABC in a coordinate system with A(0,0), B(55,0), and

√3 2 ). Then we can calculate the coordinates of points D, E, and F:

C(552 , 55

√3 2 ) - Point E: E(252 , 25

√3 2 ) - Point F: F(40,0)

- Point D: D(1032 , 7

[...detailed analysis omitted...] For each angle θ, we can write equations involving the slopes of the lines:

- 1. For ∠AEP:

tan(θ) =

y− 25

√3 2

x−252 −

√3 1 + √3 · y−

25√3 2

x− 252

- 2. For ∠BFP:

tan(θ) =

y x − 40

- 3. For ∠CDP:

√3 2

x−1032 + √3 1 −

y− 7

tan(θ) =

7√3 2

√3 · y−

x− 1032

Let’s solve this system using Python:

Python Code import sympy as sp # Define variables x, y = sp.symbols(’x y’) # Coordinates of points

- E_x = sp.Rational(25, 2)
- E_y = sp.Rational(25, 2) * sp.sqrt(3)

- F_x = sp.Rational(40, 1)
- F_y = sp.Rational(0, 1)

- D_x = sp.Rational(103, 2)
- D_y = sp.Rational(7, 2) * sp.sqrt(3)

# Slopes and tan(theta) expressions m_EA = sp.sqrt(3) m_EP = (y - E_y) / (x - E_x) tan_theta_E = ...

# Solve the system of equations solution = sp.solve([eq1, eq2], (x, y)) print("Solutions for x and y:") print(solution)

Output

Solutions for x and y: [(1525/38, 25*sqrt(3)/38)]

With these coordinates, we can compute tan(θ): tan(θ) =

25√3 38 1525

38 −40 = 5√3 Therefore: tan2(θ) = (5√3)2 = 75 Therefore, the answer is

|75|
|---|

.

- Sample 5

### Problem:

In △ABC with side lengths AB = 13, BC = 14, and CA = 15, let M be the midpoint of BC. Let P be the point on the circumcircle of △ABC such that M is on AP. There exists a unique point Q on segment AM such that ∠PBQ = ∠PCQ. Then AQ can be written as √mn, where m and n are relatively prime positive integers. Find m + n.

### Solution:

Let’s solve this step by step. First, let’s establish a coordinate system: - Place B at (0,0) Place C at (14,0) - Then M is at (7,0) - Using distances AB=13 and AC=15, we can find A at (5,12)

[...detailed analysis omitted...] First, let’s find the circumcircle of triangle ABC:

import sympy as sp # Define symbols for the circle’s center (h, k) and radius r h, k, r = sp.symbols(’h k r’) # Points A(5,12), B(0,0), C(14,0) # Equations for the circle passing through each point

- eq1 = (0 - h)**2 + (0 - k)**2 - r**2 # B(0,0)
- eq2 = (5 - h)**2 + (12 - k)**2 - r**2 # A(5,12)
- eq3 = (14 - h)**2 + (0 - k)**2 - r**2 # C(14,0)

# Subtract equations to eliminate r^2 eq21 = sp.simplify(eq2 - eq1) eq31 = sp.simplify(eq3 - eq1)

# Solve the system of equations solution = sp.solve([eq21, eq31], (h, k)) h_value, k_value = solution[h], solution[k] print("Center (h, k):", (h_value, k_value))

radius_eq = eq1.subs({h: h_value, k: k_value}) radius_solution = sp.solve(radius_eq, r) r_value = [r_val for r_val in radius_solution if r_val > 0] print("Radius r:", r_value)

Output 1

Center (h, k): (7, 33/8) Radius r: [65/8]

[...detailed analysis omitted...] Next, let’s find point P by finding where line AM intersects the circumcircle:

from sympy import symbols, solve, Rational # Define the variable t = symbols(’t’) # Define the parametric equations for x and y x = 5 + 2*t y = 12 - 12*t # Define the equation of the circle circle_eq = (x - 7)**2 + ... # Solve for t expanded_eq = circle_eq.expand() solution_t = solve(expanded_eq, t) # Calculate the coordinates for each solution t points = [] for t_solution in solution_t:

- x_value = x.subs(t, t_solution)
- y_value = y.subs(t, t_solution) points.append((x_value, y_value))

print("Solutions for t:", solution_t) for idx, point in enumerate(points):

print(f"Coordinates of point for t={solution_t[idx]}...")

Output 2

Solutions for t: [0, 197/148] Coordinates of point for t=0: (x, y) = (5, 12) Coordinates of point for t=197/148: (x, y) = (567/74, -147/37)

[...detailed analysis omitted...] Now, let’s find point Q using the condition that ∠PBQ = ∠PCQ:

from sympy import symbols, Eq, solve, sqrt, simplify, fraction from sympy import Rational

t = symbols(’t’) # Coordinates of Q Qx = 5 + 2*t Qy = 12 - 12*t # Coordinates of P

- Px = Rational(567, 74)
- Py = Rational(-147, 37) # Vectors BP, BQ, CP, CQ

- BPx = Px - 0
- BPy = Py - 0

- BQx = Qx - 0
- BQy = Qy - 0

CPx = Px - 14 CPy = Py - 0

- CQx = Qx - 14
- CQy = Qy - 0

# Compute dot products and magnitudes dot_BP_BQ = BPx * BQx + BPy * BQy dot_CP_CQ = CPx * CQx + CPy * CQy

- magnitude_BP = sqrt(BPx**2 + BPy**2)
- magnitude_BQ = sqrt(BQx**2 + BQy**2) magnitude_CP = sqrt(CPx**2 + CPy**2) magnitude_CQ = sqrt(CQx**2 + CQy**2)

# Set up and solve the equation equation = Eq(dot_BP_BQ/(magnitude_BP * magnitude_BQ),

dot_CP_CQ/(magnitude_CP * magnitude_CQ)) solution = solve(equation, t) print("Solutions for t:", solution)

Output 3 Solutions for t: [99/148, 197/148]

[...detailed analysis omitted...] Finally, let’s compute AQ and verify our answer:

from math import gcd # First compute AQ distance

- Qx_val = Qx.subs(t, Rational(99,148))
- Qy_val = Qy.subs(t, Rational(99,148)) AQ_distance = sqrt((Qx_val - 5)**2 + (Qy_val - 12)**2) simplified_AQ = AQ_distance.simplify() print("AQ =", simplified_AQ)

# Verify m and n are coprime m = 99 n = 148 gcd_val = gcd(m, n) print("GCD of 99 and 148:", gcd_val)

Output 4

AQ = 99*sqrt(37)/74 GCD of 99 and 148: 1

After manipulation, we find AQ = √99148 where m = 99 and n = 148 are coprime positive integers.

|247|
|---|

Therefore, m + n = 99 + 148 =

.

### F.2 Test case of Prompt-Hint-SFT-32B

A case from AIME24 Problem:

Let B be the set of rectangular boxes with surface area 54 and volume 23. Let r be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of B.

The value of r2 can be written as pq, where p and q are relatively prime positive integers. Find

- p + q. Solution: The problem is about finding the smallest sphere that can contain any rectangular box from the set B, where each box has a surface area of 54 and volume 23. The radius of this minimal sphere would be half of the maximum space diagonal among all possible boxes. For a box with side lengths a, b, c, we have: 1. Space diagonal d = √a2 + b2 + c2 2. Surface area: 2(ab + bc + ac) = 54 3. Volume: abc = 23

2+b2+c2

Therefore, r2 = a

4 , and we need to maximize a2 + b2 + c2 subject to the constraints. [...detailed analysis omitted...] Let’s approach this using Lagrange multipliers. Our Lagrangian is:

L = a2 + b2 + c2 − λ(2(ab + bc + ac) − 54) − µ(abc − 23) Taking partial derivatives:

∂L

- ∂a

= 2a − λ(2b + 2c) − µbc = 0 ∂L

- ∂b

= 2b − λ(2a + 2c) − µac = 0 ∂L

- ∂c

= 2c − λ(2a + 2b) − µab = 0

[...detailed analysis omitted...]

After analysis, we find that the maximum occurs when: - a = b = 2 - c = 234 Let’s verify this solution:

Python Code

from sympy import symbols, Eq, solve, sqrt from sympy import Rational

# Define variables

- a = 2
- b = 2
- c = Rational(23, 4)

# Check surface area surface_area = 2 * (a*b + b*c + a*c) print(f"Surface area = {surface_area}")

# Check volume volume = a * b * c print(f"Volume = {volume}")

# Calculate space diagonal squared diagonal_squared = a**2 + b**2 + c**2 print(f"Space diagonal squared = {diagonal_squared}")

# Calculate r squared r_squared = diagonal_squared / 4 print(f"r squared = {r_squared}")

Output

Surface area = 54 Volume = 23 Space diagonal squared = 657/16 r squared = 657/64

Therefore: - Space diagonal squared = 65716 - r2 = 65764 Since 657 = 3 × 3 × 73 and 64 = 26, these numbers are coprime. Therefore, p = 657 and

- q = 64.

|721|
|---|

The answer is p + q = 657 + 64 =

.

A Case from AIME25 Problem:

Six points A,B,C,D,E and F lie in a straight line in that order. Suppose that G is a point not on the line and that AC = 26, BD = 22, CE = 31, DF = 33, AF = 73, CG = 40, and DG = 30. Find the area of △BGE.

Solution: Let’s solve this step by step:

- 1) First, let’s assign coordinates to the points on the line, placing A at origin: - A: 0 (origin) AF = 73, so F: 73 - AC = 26, so C: 26 - CE = 31, so E: 26 + 31 = 57 - DF = 33, so D: 73 - 33

= 40 - BD = 22, so B: 40 - 22 = 18 [...detailed analysis omitted...]

- 2) Next, let’s find coordinates of G using distances CG = 40 and DG = 30: - From C(26,0): (xG − 26)2 + yG2 = 1600 ...(1) - From D(40,0): (xG − 40)2 + yG2 = 900 ...(2)

Let’s verify this using Python: Python Code from sympy import symbols, Eq, solve x_G, y_G = symbols(’x_G y_G’) # Equations based on CG = 40 and DG = 30

- eq1 = Eq((x_G - 26)**2 + y_G**2, 1600)
- eq2 = Eq((x_G - 40)**2 + y_G**2, 900)

# Subtract equations to find x_G subtracted_eq = Eq((x_G - 26)**2 - (x_G - 40)**2, 700)

- x_G_val = solve(subtracted_eq, x_G)[0] # Substitute x_G into eq2 to find y_G
- y_G_val = solve(eq2.subs(x_G, x_G_val), y_G)[0] print(f"G coordinates: ({x_G_val}, {y_G_val})")

Output G coordinates: (58, 24)

Therefore, the coordinates of the triangle vertices are: - B(18,0) - G(58,24) - E(57,0)

- 3) Calculate the area using the shoelace formula: Area = 12|x1(y2 − y3) + x2(y3 − y1) + x3(y1 − y2)| Where: - (x1,y1) = (18,0) - (x2,y2) = (58,24) - (x3,y3) = (57,0) Substituting:

- 1

- 2|18(24 − 0) + 58(0 − 0) + 57(0 − 24)|

Area = = =

- 1

- 2|432 + 0 − 1368|

1 2|−936|

= 468

|468|
|---|

Therefore, the area of triangle BGE is

.

### F.3 Test case of Hint-Engineering-RFT-32B

A Case form AIME24 Problem:

Let B be the set of rectangular boxes with surface area 54 and volume 23. Let r be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of B.

The value of r2 can be written as pq, where p and q are relatively prime positive integers. Find p + q.

Solution: The radius of the smallest sphere that can contain any rectangular box must be half of the maximum space diagonal. For a box with dimensions a, b, c, we have:

1. Space diagonal = √a2 + b2 + c2 2. Surface area: 2(ab + bc + ac) = 54 3. Volume: abc = 23 Our goal is to maximize a2 + b2 + c2 subject to these constraints, then r2 = max(a

2+b2+c2)

4 .

Let’s try assuming a = b to simplify the problem: - Surface area becomes: 2(a2 + 2ac) = 54 → a2 + 2ac = 27 - Volume becomes: a2c = 23 → c = a232 [...detailed analysis omitted...] Let’s solve the resulting cubic equation:

Python Code 1 from sympy import symbols, solve, Eq a = symbols(’a’) equation = Eq(a**3 - 27*a + 46, 0) solutions = solve(equation, a) print("Solutions:") for sol in solutions:

print(sol.simplify())

Output 1

Solutions: 2

- -1 + 2*sqrt(6)
- -2*sqrt(6) - 1

The only valid solution is a = 2, giving us: - a = b = 2 - c = 234 = 5.75 Let’s verify this is indeed the maximum using numerical optimization:

Python Code 2 from scipy.optimize import minimize def objective(x):

a, b, c = x return -(a**2 + b**2 + c**2)

- def constraint1(x): a, b, c = x return 2*(a*b + b*c + a*c) - 54
- def constraint2(x): a, b, c = x return a*b*c - 23

x0 = [2, 2, 5.75] cons = [{’type’: ’eq’, ’fun’: constraint1},

{’type’: ’eq’, ’fun’: constraint2}] bounds = [(0.0001, None), (0.0001, None), (0.0001, None)] result = minimize(objective, x0, method=’SLSQP’,

bounds=bounds, constraints=cons)

print("Optimization result:", result.success) print("Optimal a, b, c:", result.x) print("Maximum a² + b² + c²:", -result.fun)

Output 2

Optimization result: True Optimal a, b, c: [2. 2. 5.75] Maximum a² + b² + c²: 41.0625

Therefore: 1. Maximum space diagonal squared = 65716 2. r2 = 65716 /4 = 65764 3. 657 = 3 × 3 × 73 and 64 = 26 are coprime 4. Thus, p = 657 and q = 64

|721|
|---|

.

The answer is p + q = 657 + 64 =

A Case from AIME25 Problem:

Six points A,B,C,D,E and F lie in a straight line in that order. Suppose that G is a point not on the line and that AC = 26, BD = 22, CE = 31, DF = 33, AF = 73, CG = 40, and DG = 30. Find the area of △BGE.

Solution: First, let’s assign coordinates to all points on the line, placing A at origin: Given distances: - AC = 26 - BD = 22 - CE = 31 - DF = 33 - AF = 73 This allows us to determine coordinates: - A: (0,0) - B: (18,0) - C: (26,0) - D: (40,0) - E: (57,0) - F: (73,0) [...detailed analysis omitted...] To find G’s coordinates, we use distances CG = 40 and DG = 30:

#### 1) From C (26,0): (xG−26)2+yG2 = 1600 ...(1) 2) From D (40,0): (xG−40)2+yG2 = 900

...(2) Let’s solve this system using Python:

Python Code 1 from sympy import symbols, Eq, solve x_G, y_G = symbols(’x_G y_G’) # Equations based on CG = 40 and DG = 30

- eq1 = Eq((x_G - 26)**2 + y_G**2, 1600)
- eq2 = Eq((x_G - 40)**2 + y_G**2, 900)

# Solve for x_G and y_G subtracted_eq = Eq((x_G - 26)**2 - (x_G - 40)**2, 700)

- x_G_val = solve(subtracted_eq, x_G)[0]
- y_G_val = solve(eq2.subs(x_G, x_G_val), y_G)[0] print(f"G coordinates: ({x_G_val}, {y_G_val})")

Output 1 G coordinates: (58, 24)

Therefore, we have all coordinates: - B: (18,0) - G: (58,24) - E: (57,0) Let’s calculate the area using the shoelace formula:

Python Code 2 # Calculate area using shoelace formula B_coord = (18, 0) G_coord = (58, 24) E_coord = (57, 0) area = 0.5 * abs(

B_coord[0] * (G_coord[1] - E_coord[1]) + G_coord[0] * (E_coord[1] - B_coord[1]) + E_coord[0] * (B_coord[1] - G_coord[1])

) print(f"Area of triangle BGE: {area}")

Output 2 Area of triangle BGE: 468.0

Verification using Base × Height method: - Base (BE) = 57 − 18 = 39 - Height = 24 - Area

= 39×24

2 = 468

|468|
|---|

Therefore, the area of triangle BGE is

.

## G Stron-to-Weak Distillation Data

For the strong-to-weak distillation process, we constructed a comprehensive mathematical problem dataset of approximately 10,000 problems from three primary sources. The dataset comprises 800 problems from AIME (American Invitational Mathematics Examination) competitions prior to 2024, 2,280 problems from the MATH [68] dataset’s training set (Level 5), and 7,000 problems sampled from the Numina-math [69] collection. For the Numina-math sampling, we implemented several filtering criteria: problems containing figures were excluded, as were proof-based problems, multiple-choice questions, and problems where answers appeared within the problem statements. After applying these filtering criteria, we randomly sampled 7,000 problems to ensure a clean and standardized dataset suitable for our distillation training process.

## H Baselines

We compare our approaches against a range of state-of-the-art mathematical reasoning models across different parameter scales:

SOTA Models. We benchmark against the most advanced large reasoning models currently available: OpenAI’s o1 [5], QwQ-32B [6], and DeepSeek-R1 [7]. These models represent the frontier of mathematical reasoning capabilities and serve as an upper bound for performance comparison.

Frontier Models (32B). For the 32B parameter scale, we use DeepSeek-R1-32B [7] as our foundation model since it serves as the starting point for many open-source models in this size range. We also compare against contemporary tool-integrated reasoning (TIR) models, including START-32B [20], STILL-3-TOOL-32B [16], and ReTool-R1-32B [61], the latter being a concurrent work focusing on reinforcement learning for tool use.

Lightweight Models (1.5B). In the lightweight category, we use DeepSeek-R1-1.5B [7] as our base model for the same reason as its 32B counterpart. We benchmark against DeepScaleR-1.5BPreview [66], the current state-of-the-art reinforcement learning model at this scale, and ToRL1.5B [59], a concurrent work on tool-integrated reasoning with reinforcement learning.

Our selection of baselines spans different model sizes, training methodologies (SFT, RL, and RFT), and approaches to mathematical reasoning (with and without explicit tool use). This comprehensive comparison allows us to evaluate the effectiveness of our Prompt-Hint and Hint-Engineering approaches relative to both established benchmarks and recent innovations in the field.

## I Broader Impacts

Our work on enhancing code-integrated reasoning capabilities in LRMs offers several potential benefits to society. The improved efficiency and accuracy in mathematical reasoning could enhance educational applications and academic research, while reduced token usage contributes to environmental sustainability through decreased computational resource consumption. The integration of precise computational tools with natural language reasoning may also improve the reliability of AI systems in applications requiring accurate calculations.

However, as with any advancement in AI capabilities, potential risks should be considered. While our models are trained on public mathematical datasets and intended for educational and research purposes, the enhanced reasoning capabilities could potentially be misused if not properly secured. We recommend implementing appropriate access controls and maintaining transparency about the system’s limitations and intended use cases.

We encourage future work to continue exploring responsible deployment strategies that maximize beneficial impacts while minimizing potential risks.

## J Limitation

Our work presents a novel framework for enhancing code-integrated reasoning in LRMs through hintengineering and efficient post-training strategies. While we demonstrate significant improvements in both performance and efficiency, particularly in mathematical reasoning tasks, several directions remain for future exploration. Due to our use of DeepSeek-R1-Distill-Qwen models (32B and 1.5B), the computational requirements for training and inference may affect broader accessibility. While we utilize publicly available datasets and models, ensuring our work maintains transparency and reproducibility, the rapid evolution of LRM capabilities suggests opportunities for continued enhancement as new architectures emerge. Furthermore, while our current work emphasizes the integration of code interpreters within long-form reasoning, there remain opportunities to investigate additional synergies between external tools and language models’ inherent reasoning capabilities. As the field of large reasoning models continues to advance rapidly, we anticipate further developments in optimizing these interactions.

