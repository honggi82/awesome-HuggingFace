### M-A-P

## CodeCriticBench: A Holistic Code Critique Benchmark for Large Language Models

Alexander Zhang2,∗, Marcus Dong2,∗, Jiaheng Liu1,2,∗,†, Wei Zhang4, Yejie Wang6, Jian Yang4, Ge Zhang2, Tianyu Liu2, Zhongyuan Peng5, Yingshui Tan3, Yuanxing Zhang7, Zhexu Wang6, Weixun Wang3, Yancheng He3, Ken Deng3, Wangchunshu Zhou2,8, Wenhao Huang2, Zhaoxiang Zhang5

1NJU, 2M-A-P, 3Alibaba, 4BUAA, 5CASIA, 6BUPT, 7Kuaishou, 8OPPO

https://github.com/multimodal-art-projection/CodeCriticBench

# arXiv:2502.16614v1[cs.CL]23Feb2025

#### Abstract

The critique capacity of Large Language Models (LLMs) is essential for reasoning abilities, which can provide necessary suggestions (e.g., detailed analysis and constructive feedback). Therefore, how to evaluate the critique capacity of LLMs has drawn great attention and several critique benchmarks have been proposed. However, existing critique benchmarks usually have the following limitations: (1). Focusing on diverse reasoning tasks in general domains and insufficient evaluation on code tasks (e.g., only covering code generation task), where the difficulty of queries is relatively easy (e.g., the code queries of CriticBench are from Humaneval and MBPP). (2). Lacking comprehensive evaluation from different dimensions. To address these limitations, we introduce a holistic code critique benchmark for LLMs called CodeCriticBench. Specifically, our CodeCriticBench includes two mainstream code tasks (i.e., code generation and code QA) with different difficulties. Besides, the evaluation protocols include basic critique evaluation and advanced critique evaluation for different characteristics, where fine-grained evaluation checklists are well-designed for advanced settings. Finally, we conduct extensive experimental results of existing LLMs, which show the effectiveness of CodeCriticBench.

Basic Critique Evaluation

/Code Generation/LiveCodeBench

You are a senior programming expert. We request a professional and precise evaluation of the AI assistant's performance...... Evaluation Process:

You are given a 0-indexed binary string s of length n on which you can apply two types of operations: 1. Choose an index i and invert all characters from index 0 to index i (both inclusive), with a cost of i + 1. 2. Choose an index i and invert all characters ...... Constraints: 1 <= s.length == n <= 10^5......

{"reasons": "Let's evaluate the solution in detail: 1. Algorithm Logic: The solution is correct and clever. ......, Flipping from i to end (cost = n-i). 2. Implementation Correctness:

1. Carefully review the user’s question and the assistant’s answer. ..... Evaluation Requirements:

LLMs

[Figure 1]

......

- - Provide detailed, point-by-point feedback on the answer. ...... Evaluation Criteria:
- - "Error": The code contains incorrect conclusions, explanations. ....
- - "Correct": The code is entirely accurate and meets all outlined requirements. Output Format: Provide the evaluation in JSON format as follows: ......

###### Question

point. 3. Time and Space Complexity:...... ",

class Solution:

"is_assistant_correct": "Correct"}

def minimumCost(self, s: str) -> int: n = len(s) cost = 0 # Calculate the cost to make all characters '0' for i in range(1, n):

Predict: Correct

if s[i] != s[i - 1]: cost += min(i, n - i)

- 1. Input Handling: The code.....Score: 8
- 2. Time Complexity: ...... Score: 10......

returncost Answer

You are a professional code evaluation expert with the following qualifications: ...... Evaluation Process:

###### LLMs

10. Code Structure: The solution is wellstructured ......Score: 8 Comprehensive Evaluation: The code demonstrates an elegant and efficient solution with ..... Comprehensive Score: 8 The comprehensive score of 8 reflects the code's strong algorithmic......

Correctness Verification: Can the code correctly handle all possible input scenarios, ...... Time Complexity Optimization: Is the time complexity of the code O(n), and can it be further optimized to reduce unnecessary computations? Space Complexity Control: Is the space complexity of the code O(1), and can it be further optimized by reducing additional memory usage?

[Figure 2]

1. Analyze and understand the problem statement and provided answer thoroughly. ...... Output Specifications:

- - Scoring Range: - Each evaluation metric is scored on an integer scale from 1-10. ...... Scoring Guidelines:
- - 1-2 points: Critical flaws; fails to meet requirements. ......
- - 9-10 points: High-quality, near-perfect implementation. Output Format:......

Fine-Grained Evaluation Checklists

Predict: 8

Advanced Critique Evaluation

Figure 1. Illustration of the Basic Critique Evaluation and Advanced Critique Evaluation.

* Equal Contribution. † Corresponding Author.

##### Contents

- 1 Introduction 3
- 2 Related Works 4

- 2.1 Code LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 2.2 Critic Models for LLMs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3 CodeCriticBench 4

- 3.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.3 Dataset Construction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 3.4 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4 Experiments 7

- 4.1 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.2 Main Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.3 Further Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5 Conclusion 13

- A More Experimental Results 17

- A.1 Anonymous Data Link . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.2 Relations Between Basic Correctness and Advanced Evaluations Scores . . . . . . 17
- A.3 Case Visualization of CodeCriticBench . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.4 Scaling Law . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.5 Different Application Scenes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.6 Bug Identification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- A.7 Critique Evaluation on “Code Gen” and “Code QA” Subset . . . . . . . . . . . . 19
- A.8 Critique Evaluation on Multiple Fine-Grained Dimensions . . . . . . . . . . . . . 19

- B Prompt 20
- C Model Lists 21

##### 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains [Team, 2024, Touvron et al., 2023, Rozière et al., 2023], including natural language processing, code generation and complex reasoning tasks [Wang et al., 2025, Liu et al., 2024a,b, Bai et al., 2024]. As these models continue to evolve, their ability to critique and refine their outputs has emerged as a crucial area of research. This critique capacity is fundamental to enhancing the reasoning abilities of LLMs, which enables them to provide detailed analysis and constructive feedbacks and improves the quality and reliability of their outputs [Madaan et al.,

- 2023, Song et al., 2025b]. Recently, several critique benchmarks have been proposed [Lan et al., 2024b, Lin et al.,
- 2024, Tan et al., 2024, Song et al., 2025a, Zheng et al., 2024a, Tang et al., 2025]. For example, CriticBench [Zheng et al., 2024b] is proposed to assess LLMs’ abilities to critique across a variety of tasks including math, commonsense, symbolic, code and algorithmic, where the code subset is from the code generation datasets (i.e., Humaneval [Chen et al., 2021] and MBPP [Austin et al., 2021]). However, these benchmarks have predominantly focused on diverse reasoning tasks in general domains, leaving a significant gap in the evaluation of code-related tasks. Moreover, these benchmarks often lack comprehensive evaluation across different dimensions of critique capacity. Besides, in software development, the ability of LLMs to generate, understand and critique code is important. As LLMs are increasingly employed in coding assistance tools and automated code review systems, there is a pressing need for a robust framework to evaluate their code critique capabilities. This evaluation should encompass not only the accuracy of code generation but also the model’s ability to provide insightful feedback, identify errors and suggest improvements in the code QA scenario.

To address these limitations, as shown in Figure 1, we introduce CodeCriticBench, a holistic code critique benchmark for LLMs. Specifically, first, our benchmark covers two mainstream code tasks: code generation and code QA. These tasks are presented with varying levels of difficulty, allowing for a nuanced assessment across different coding challenges. Second, CodeCriticBench incorporates both basic and advanced critique evaluations for assessing different characteristics. For the basic setting, we prompt the judge model to provide the “correct/error” response and the corresponding reasoning process. For the advanced setting, we have developed fine-grained evaluation checklists for each problem, enabling a more detailed and precise assessment of the models’ critique capabilities, which allows for a more thorough evaluation of existing LLMs. In summary, our contributions are as follows:

- • To evaluate the critique abilities of LLMs on the code domain, we introduce the first holistic code critique benchmark CodeCriticBench, which includes the critique on both code generation and code QA tasks.
- • Based on our proposed CodeCriticBench, we design both the basic and advanced critique evaluations, which provide a comprehensive analysis of the critique ability.
- • We systematically evaluate the critique abilities of 38 LLMs (including general and codespecific models) on CodeCriticBench and provide detailed analysis regarding the critique capabilities of LLMs.

##### 2. Related Works

- 2.1. Code LLMs

Large Language Models (LLMs) have rapidly developed and are significantly impacting automated software development. These foundational models can produce fluent human language and understand semantics to perform complex tasks, bringing new possibilities to automated code generation, including Santacoder Allal et al. [2023], CodeGPT Lu et al. [2021], etc. More and more open-source and proprietary Code LLMs have emerged and demonstrated competitive performance, including Starcoder Li et al. [2023], CodeLlama Roziere et al. [2023], Wizardcoder Luo et al. [2023], DeepSeek-Coder Guo et al. [2024], Qwen-Coder Hui et al. [2024], Claude-3.5-Sonnet Anthropic [2024] and GPT-4o OpenAI [2024] etc.

- 2.2. Critic Models for LLMs

Reinforcement learning from human feedback has proven effective Achiam et al. [2023], though it can be very labor-intensive. A promising approach involves using LLMs to themselves to assist in the evaluation, which can further improve model outputs Saunders et al. [2022], McAleese et al. [2024]. Accurate and informative critique feedback allows LLMs to refine outputs, moving towards more advanced intelligence. However, despite their problem-solving strengths, LLMs currently demonstrate weak performance in critique tasks [Zheng et al., 2024b, Yang et al., 2024]. Improving LLM critique abilities relies on supervision from human annotations Saunders et al. [2022], McAleese et al. [2024] and stronger LLMs acting as human proxies Lan et al. [2024a], Zhang et al. [2024], Ke et al. [2024], Ankner et al. [2024], Zheng et al. [2024b], Yang et al. [2024], Sharma et al. [2024]. Recently, some critic benchmarks [Zheng et al., 2024b, Tang et al., 2025] have also been proposed.

- 3. CodeCriticBench

###### 3.1. Overview Table 1. Dataset statistics of CodeCriticBench.

CodeCriticBench is a holistic evaluation benchmark for Code Critic tasks, encompassing code evaluation in both Code Generation and Code QA tasks. Table 1 presents an overview of CodeCriticBench, which comprises 4,300 samples. Each sample includes a question, an answer, a set of fine-grained evaluation checklists across multiple dimensions and associated labels—namely, correctness labels, perdimension evaluation scores, a final score and a difficulty level. Furthermore, we employ the LLaMA3 tokenizer Dubey et al. [2024] to determine the token counts for questions, answers and evaluation problems in Table 1. For finegrained evaluation checklists, token counts are summed across all evaluation problems. As reported in Table 1, the average token length is 451.06 for questions, 322.22 for answers and 287.49 for the aggregated evaluation checklists.

Statistics Number #Problems 4,300 Difficulty Level

- Easy/Medium/Hard 1,517/1,084/1,699 Length Question

- - maximum length 32,063 tokens
- - minimum length 8 tokens
- - average length 451.06 tokens

Answer

- - maximum length 32175 tokens
- - minimum length 9 tokens
- - average length 322.22 tokens

Fine-Grained Evaluation Checklists

- - maximum length 676 tokens
- - minimum length 97 tokens
- - average length 287.49 tokens

- Table 2. Comparisons between CodeCriticBench and other benchmarks. “Code Gen” denotes “Code Generation”. “-” represents that the corresponding task is not included in the dataset.

Benchmark Data Size Code Gen. Code QA Basic Advanced

CriticBench 3,825 464 - ✓ × CriticEval 3,608 1,340 - ✓ × JudgeBench 350 42 - ✓ × RealCritic 2,093 - - ✓ ×

CodeCriticBench 4,300 3,200 1,100 ✓ ✓

Most existing Code Critic Benchmarks primarily focus on evaluating basic properties, such as code correctness. In contrast, CodeCriticBench offers both a fundamental assessment of code correctness and a comprehensive evaluation of associated question–answer pairs, alongside a fine-grained scoring system that spans multiple dimensions. Notably, every data sample in CodeCriticBench is paired with uniquely tailored, fine-grained evaluation checklists, ensuring that each instance is assessed in a context-specific and comprehensive manner. Table 2 presents a comparison between CodeCriticBench and other Code Critic Benchmarks.

###### 3.2. Data Collection

Code Generation. In the code generation task, instructions are mainly algorithmic problems. We collect many such data (restricted to test sets) from CodeForces, MBPP Austin et al. [2021] and LiveCodeBench Jain et al. [2024]. Given the limited test sets in MBPP and LiveCodeBench, we employ DeepSeek-v3 to rewrite problem while preserving their semantics, thereby enhancing diversity and test case reusability. To assess the model’s ability to detect specific programming errors, we create a Debug subset. We first compile a list of bugs through iterative discussions with experts and LLMs, then prompt the LLM to insert specified error types into sandbox-verified correct code. The modified samples undergo two filtering rounds: sandbox execution to confirm errortriggering and manual review to ensure the errors matched their intended categories. Details on the error types and bug insertion prompts are provided in Appendix B.

[Figure 3]

[Figure 4]

[Figure 5]

Human Check Filter

Insert Bug

[Figure 6]

Buggy

Error Lists Code

CodeForces LCB MBPP Debug

|Algorithm Problems<br><br>[Figure 7]<br><br>[Figure 8]<br><br>LiveCodeBench<br><br>(LCB)<br><br>MBPP|
|---|

Extraction & Filtering Code Generation

||[Figure 9]<br><br>Technical Forum| |
|---|---|
| | |
<br><br>Rule-Based Filter<br><br>[Figure 10]<br><br>Q-A Pairs<br><br>Generation<br><br>Raw Q-A Pairs<br><br>[Figure 11]<br><br>[Figure 12]<br><br>Quality<br><br>Filtering<br><br>[Figure 13]<br><br>|Code QA<br><br>|
|---|
|
|---|

Figure 2. Illustration of data collection process.

Code QA. Building on previous work Baars and Meester [2019], we first apply a rule-based filtering method to clean the extracted content, removing site information, ads and HTML tags. To ensure our benchmark reflects real-world scenarios, we collect authentic code requirements and manually craft responses from StackOverflow. Unlike Yue et al. [2024], which directly extract question–answer pairs, we use Qwen2.5-72B to generate new questions. As shown in Figure 2, Qwen2.5-Coder samples the same document multiple times to generate diverse responses, which are then filters through manual and LLM-assisted reviews, forming a high-quality subset.

###### 3.3. Dataset Construction

Difficulty. To assess sample difficulty, we employ twelve state-of-the-art LLMs1 and classify each sample based on the proportion of models that produce correct predictions. Specifically, a sample is labeled “Easy” if at least 80% of the models reason correctly, “Medium” if the success rate is between 60% and 80% and “Hard” if fewer than 60% succeed. Note that because random scoring yields an expected success rate of approximately 50%, a success rate below 60% suggests that the model’s performance is close to random guessing. This process yields 1,517 Easy, 1,084 Medium and 1,699 Hard samples, ensuring a balanced distribution of difficulty levels in CodeCriticBench.

Category. To determine the specific application scenarios of the “Code QA” subset, we prompt DeepSeek-v3 to classify the data based on a predefined category list adopted from Liu et al. [2024d]. Detailed prompts used for this classification are provided in Appendix B.

Correctness. For the “Code Gen” subset, each sample is accompanied by test cases that facilitate automated correctness evaluation within a sandbox. A sample is marked as “Correct” only if all test cases pass; otherwise, it is labeled as “Error.” In contrast, for the “Code QA” subset, we engage 20 volunteers with coding experience to assess correctness. Each question is independently evaluated by three raters and the final correctness label is determined by majority vote.

Fine-Grained Evaluation Checklists and Corresponding Scores. In contrast to previous Code Critic Benchmarks, we introduce a more detailed and customized scoring framework. Initially, we define 10 evaluation dimensions for both code generation and code QA tasks through iterative discussions with human experts and LLMs. Next, we employ prompts for DeepSeekV3 to generate tailored evaluation questions for each data instance. To ensure the generated questions met the desired criteria, we perform random sampling and manual inspections, repeating the process until the pass rate exceeds 95%. Inspired by previous work Que et al. [2024], we manually annotate 20% of the dataset to establish baseline scores. Subsequently, three advanced models (Claude3.5-Sonnet, GPT-4o and Gemini2.0), evaluate the entire dataset, with each evaluation dimension’s final score determined by majority vote. Finally, we calibrate the LLM-generated scores by applying linear regression, using them as independent variables (x) and the human ratings as the dependent variable (y), to produce the final scores for all samples. Detailed prompts and multiple evaluation names are provided in Appendix B and A.8.

###### 3.4. Evaluation Metrics We employ multiple evaluation metrics to rigorously assess our model’s performance.

Accuracy (ACC). ACC measures binary classification performance by determining whether the model’s predictions match the ground truth labels for basic critique evaluation. Let 𝑁 denote the total number of instances, 𝑦ˆ𝑖 the predicted label for instance 𝑖 and 𝑦𝑖 the true label. Accuracy is computed as follows:

###### ∑︁𝑁

1

ACC =

𝐼(𝑦ˆ𝑖 = 𝑦𝑖) (1)

𝑁

𝑖=1

where 𝐼(𝑦ˆ𝑖 = 𝑦𝑖) is an indicator function that returns 1 if the prediction is correct and 0 otherwise. Mean Squared Error (MSE). MSE quantifies the discrepancy between the model’s predicted and

1Claude3.5-Sonnet, GPT 4o, DeepSeek-v2.5, DeepSeek-v3, Doubao-Coder-Preview, Llama3.3-70B-Instruct, Qwen2.5-72B-Instruct, DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1, GLM-4-Plus, Qwen2.5-Max and OpenAI o1-Preview.

true scores across multiple dimensions for advanced critique evaluation, which measures how closely the model’s predictions approximate the actual values. MSE is calculated as follows:

###### ∑︁𝑁

1

(𝑦ˆ𝑖 − 𝑦𝑖)2 (2)

MSE =

𝑁

𝑖=1

where 𝑦ˆ𝑖 represents the predicted score and 𝑦𝑖 the true score for instance 𝑖.

Pass@1 Accuracy. In code error detection, each code snippet contains at least one error (𝑛𝑖 ≥ 1). The model’s objective is to detect at least one error per snippet. Let 𝐸ˆ𝑖 denotes the set of errors predicted by the model for instance 𝑖 and 𝐸𝑖 the set of actual errors. A prediction is considered successful if the intersection between 𝐸ˆ𝑖 and 𝐸𝑖 is non-empty. Pass@1 accuracy is defined as:

###### ∑︁𝑁

1

𝐼(𝐸ˆ𝑖 ∩ 𝐸𝑖 ≠ ∅) (3)

ACCPass@1 =

𝑁

𝑖=1

##### 4. Experiments

###### 4.1. Baselines

We base our model selection on benchmarks from the code domain, including those outlined in Liu et al. [2024c] and Liu et al. [2024d], ultimately selecting 38 models for evaluation. These models encompass both open-source and closed-source options, with sizes ranging from 0.5 billion to over 100 billion parameters. A detailed list of the models tested, along with their respective links, is provided in Tables 7 and 8 in Appendix C.

###### 4.2. Main Results

Table 4. The accuracy of different models in identifying programming error types.

Critique Evaluation. Table 3 shows the basic evaluation (ACC metric) and advanced evaluation (MSE metric) for all models in our study. The results are grouped by model size, with separated blocks for closed-source and o1-like models. Within each size block, models are sorted by ACC. As observed, performance generally improves with more parameters, with o1-like models achieving milestone results and all exceed 70%. In the advanced evaluation, DeepSeek-R1 performs best on the “Code Gen” subset with an MSE of 3.92, while Claude3.5-Sonnet leads in “Code QA” with an MSE of 1.02. This difference likely reflects the distinct nature of the two tasks: “Code QA” prioritizes concise, high-quality answers, while “Code Gen” emphasizes solving algorithmic problems that require stronger reasoning abilities, whereas “Code QA” may benefit more from dialogue optimization.

Model ACC

GLM-4-Plus 47.75 Claude3.5-Sonnet 54.00 Qwen2.5-Max 59.50 GPT 4o 61.00 Doubao-Coder-Preview 67.00

Close-Sourced

- Qwen2.5-Coder-0.5B-Instruct 21.50

- Qwen2.5-Coder-1.5B-Instruct 32.75

Qwen2.5-Coder-3B-Instruct 48.00 Qwen2.5-Coder-7B-Instruct 61.00 Qwen2.5-Chat-7B-Instruct 41.00 Qwen2.5-Coder-14B-Instruct 62.25 Qwen2.5-Chat-14B-Instruct 47.50 Qwen2.5-72B-Instruct 61.25

Open-Sourced

Bug Identification. We evaluate thirteen models to assess their accuracy in identifying code error types. As shown in Table 4, larger models generally perform better in distinguishing code errors, which aligns well with our expectations.

- Table 3. Results of different models. “gen” and “qa” denote code generation and code qa tasks respectively. “ACC” and “MSE” metrics are used for basic and advanced critique evaluations.

Model ACCAll ACCgen ACCqa MSEAll MSEgen MSEqa

0.5B+ Instruction Tuned Model

- Qwen2.5-Coder-0.5B-Instruct 51.79 51.28 53.27 24.24 23.96 25.06 1B+ Instruction Tuned Model

Yi-Coder-1.5B-Chat 50.98 51.56 49.27 29.26 28.27 32.13 DeepSeek-Coder-1.3B-Instruct 51.44 52.31 48.91 22.67 24.48 17.39

OpenCoder-1.5B-Instruct 53.16 53.06 53.45 27.60 28.07 26.23

- Qwen2.5-Coder-1.5B-Instruct 54.47 53.87 56.18 16.56 17.13 14.93 Qwen2.5-Coder-3B-Instruct 55.21 55.22 55.18 12.43 13.01 10.76

6B+ Instruction Tuned Model

CodeLlama-7B-Instruct 54.47 53.94 56.00 19.12 20.89 13.96 Qwen2.5-Chat-7B-Instruct 54.63 51.81 62.82 8.07 8.73 6.15

CodeQwen1.5-7B-Chat 55.63 55.41 56.27 18.22 19.06 15.78 OpenCoder-8B-Instruct 56.37 57.66 52.64 19.33 19.45 18.99

Qwen2.5-Coder-7B-Instruct 57.95 56.78 61.36 5.64 6.17 4.10 Yi-Coder-9B-Chat 61.67 63.66 55.91 14.21 14.28 14.02

13B+ Instruction Tuned Model

CodeLlama-13B-Instruct 52.17 52.16 52.18 13.94 15.55 9.26 StarCoder2-15B-Instruct 53.43 53.59 52.97 21.42 20.92 22.88

Qwen2.5-Coder-14B-Instruct 59.00 56.38 66.64 5.08 5.57 3.65

DeepSeek-v2-Lite-Chat 59.20 58.78 60.42 5.57 6.35 3.29 DeepSeekCoder-v2-Lite-Instruct 59.81 59.34 61.18 5.67 6.46 3.35

Qwen2.5-Chat-14B-Instruct 59.98 58.59 64.00 4.38 5.02 2.51 20B+ Instruction Tuned Model

CodeLlama-34B-Instruct 54.79 54.06 56.91 13.45 15.06 8.76 Qwen2.5-Coder-32B-Instruct 61.67 59.00 69.45 4.60 5.19 2.89

Qwen2.5-Chat-32B-Instruct 63.98 62.38 68.64 4.32 5.09 2.09 70B+ Instruction Tuned Model DeepSeek-v2.5 60.35 58.46 65.85 3.97 4.78 1.63

DeepSeekCoder-v2-Instruct 64.42 62.42 70.23 4.19 5.14 1.46 Llama3.3-70B-Instruct 65.91 65.16 68.09 4.78 5.65 2.24 Qwen2.5-72B-Instruct 68.35 68.44 68.09 3.99 4.61 2.20

Close-Sourced API Model GPT 4o-mini 60.56 58.31 67.09 3.92 4.82 1.30

Doubao-Coder-Preview 61.42 60.06 65.36 6.51 7.07 4.90 GLM-4-Plus 61.55 60.94 63.35 3.60 4.25 1.69 DeepSeek-v3 62.00 61.44 63.64 3.64 4.49 1.18

Qwen2.5-Max 63.36 62.74 65.17 4.09 5.04 1.33

GPT 4o 68.06 67.56 69.53 4.15 5.04 1.55 Claude3.5-Sonnet 68.79 66.06 76.73 3.78 4.73 1.02

o1-like Models

QwQ-32B-Preview 57.35 56.59 58.82 7.20 8.07 4.67 Gemini2.0-Flash-Thinking 64.53 64.88 63.55 3.88 4.80 1.19

OpenAI o1-mini 71.77 76.06 59.27 4.92 6.08 1.54 DeepSeek-R1-Distill-Qwen-32B 72.49 75.38 64.09 4.34 5.25 1.71

DeepSeek-R1 72.76 79.09 54.36 4.20 3.92 5.02 OpenAI o1-Preview 75.30 80.53 59.89 4.81 5.68 2.26

- 4.3. Further Analysis

Scaling Law. To assess the effectiveness of the scaling law, we visualize the performance across nearly all models in our study. Figure 3 presents the results of the basic evaluation ACC, with

corresponding MSE results provided in Appendix A.4. The data clearly demonstrate that as the number of parameters increases, the ACC increases, which further validate the robustness of our CodeCriticBench dataset.

*

*

*

*

*

*

*

*

*

Figure 3. Scaling law on basic critique evaluation (ACC) across models. “*” indicates an estimated parameter size.

Different Application Scenes. Our “Code QA” data subset includes 11 scenarios, such as “Fundamental Programming”, “Software Engineering” and ‘Mathematics”. To evaluate model performance across these domains, we test five models of varying sizes. We calculate their ACC on basic evaluation and MSE on advanced evaluation. As shown in Figures 4 and 17, larger models generally show improved ACC and reduced MSE, supporting our expectations and reinforcing their validity.

###### Comparison Across Different Models on "Code QA" (Basic Critique Evaluation)

90

Qwen2.5-Coder-7B Qwen2.5-Coder-14B

Qwen2.5-Coder-32B Claude3.5-Sonnet

87.0

87.0

86.0

| |
|---|

| |
|---|

82.0

80.0

80

78.0

78.0

78.0

###### Score(%)

75.0

75.0

73.0

73.0

73.0

73.0

72.0

70.0 70.0

70.0

70.0

70.0 69.0

70.0

70.0

70.0

69.0

70

67.0

66.0

65.0 66.0

65.0

64.0

63.0

62.0 62.0 62.0

62.0

61.0

61.0

60.0

59.0

60

56.0

54.0

52.0

51.0

50

Fundamental Programming

Advanced Programming

Software Engineering

Data Analysis

Mathematics Desktop and Web Development

Machine Learning

Scientific Computing

Databases Multimedia Operating Systems

- Figure 4. Comparison across different models on “Code QA” (Basic Critique Evaluation).

Different Difficulty’s Performance. As previously discussed, the difficulty of data is quantified by the proportion of models that accurately predict its correctness. To further analyze performance across different difficulty levels, we evaluate twelve advanced LLMs. Figure 5 shows that all models perform well on Easy data, with accuracies above 90%. For Medium data, the top performers are Qwen2.5-72B-Instruct, GPT 4o, Claude3.5-Sonnet and DeepSeekR1-Distill-Qwen-32B, each achieving around 82%. This performance is consistent with their rankings on other established leaderboards. In contrast, DeepSeek-R1 and OpenAI o1-Preview underperform on Easy to Medium data, likely due to overthinking. On more challenging data, most models show a significant drop in performance, with accuracies around 30%. However, DeepSeek-R1 and OpenAI o1-Preview maintain relatively strong performance, with accuracies

of 51.97% and 55.75%, respectively. We hypothesize their strong performance is partly due to o1-like architecture, which help them handle complex data.

###### Model Performance (ACC) on Different Difficulty Levels

98.75 97.96 98.29 95.98

93.40 94.26 96.44 94.53 96.50 96.04

100

92.21 94.39

81.83 81.81 81.18 82.64

78.84 79.22

77.63 77.12

80

73.99 73.34

Score(%)

70.18 70.97

60

55.75

51.97

44.88

Easy

40

31.84 32.61 32.59 34.55

Medium

27.59

25.72 24.69

Hard

23.26 22.13

20

GLM-4 Plus

DeepSeek-v2.5 Doubao-Coder Preview

DeepSeek-v3 Qwen2.5 Max

Llama3.3 70B-Instruct

Qwen2.5 72B-Instruct

GPT 4o Claude3.5 Sonnet

DeepSeek-R1 Distill-Qwen-32B

DeepSeek-R1 OpenAI o1-Preview

- Figure 5. Model performance (ACC) on different difficulty levels (Basic Critique Evaluation) .

Different Bug Types. To gain a more intuitive understanding of how models with different parameter sizes detect code errors, we select five common error types and evaluate six models of varying sizes to assess their ability to correctly identify the corresponding error types. As shown in Figure 6, the accuracy of error type identification generally improves with the increase in model size. Among the five error types, Qwen2.5-72B, Claude3.5-Sonnet and GPT-4o emerge as the top-performing models, each securing first place in the identification of 2, 2 and 1 error categories, respectively.

Figure 6. Comparison of the accuracy of different models in identifying five common programming error types.

Relations Between Basic Correctness and Advanced Evaluations Scores. To explore the relationship between data correctness and the final multidimensional score, we examine the distribution of correct and error data across different scores. As shown in Figure 7, almost all “Error” instances have scores below 5, while “Correct” instances consistently score 5 or higher. Correct scores are mainly around 7, 8 and 9, while error scores are typically between 2 and 3. Additional statistical details for the “Code Gen” and “Code QA” subsets can be found in Appendix A.2. Overall, the consistency between data correctness and final scores supports the robustness of our dataset.

Figure 7. Rating distribution of the CodeCriticBench.

Error Studies. We evaluate seven distinct LLMs, both open-source and closed-source, of varying sizes, to assess their ability to identify three common error types: “Reference Error”,

“Performance Issue” and “Security Vulnerability”, out of a total of twenty-three categories. The results for these error types are shown in Figure 8, with additional details in Appendix A.6. As shown in Figure 8, models with higher capabilities perform better at distinguishing errors. However, most models, except Doubao-Coder-Preview and Claude3.5-Sonnet, struggle with identifying “Performance Issue”. We suggest this is due to insufficient code-related optimization and models’ smaller sizes.

Doubao-CoderPreview

47.27%

48.75%

27.59%

Claude3.5Sonnet

47.27%

41.25%

29.89%

45.45%

52.50%

13.79%

GPT 4o

52.73%

38.75%

12.64%

GLM-4-Plus

Qwen2.5-Coder14B-Instruct

41.82%

40.00%

16.09%

Qwen2.5-Coder7B-Instruct

Security Vulnerability

27.27%

41.25%

12.64%

Reference Error

Qwen2.5-Coder3B-Instruct

23.64%

8.75%

14.94%

Performance Issue

0 20 40 60 80 100 120 140

Figure 8. Experimental accuracy results of different models across various error types.

Why Advanced Critique Evaluation Is Important. To further validate the consistency between basic critique, advanced critique and human evaluations, we randomly sample 400 instances (named CodeCritic_400) and test 8 models of different sizes, including both open-source and closed-source models. We then rank the models based on human, basic critique and advanced critique evaluations. As shown in Figure 9, rankings of the advanced setting closely match human evaluation, while results of the basic setting differ significantly. This highlights the effectiveness of our advanced evaluation approach, which provides more accurate and consistent results with human evaluations.

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

Advanced

Basic

Human

Rank(1=highest)

Qwen2.5-Coder 7B-Instruct

Qwen2.5-Coder 14B-Instruct

Qwen2.5-Coder 32B-Instruct

Qwen2.5 72B-Instruct

Claude3.5 Sonnet

GPT 4o OpenAI o1-mini

OpenAI o1-Preview

Figure 9. Comparison of ranking of model responses by three methods: basic critique, advanced critique and human evaluations.

Case Visualization of CodeCriticBench. To get an intuitive understanding of data in CodeCriticBench, we present a visualization for the “Code QA” correct example in Figure 10, with additional visualization results provided in Appendix A.3. As observed, each sample consists of

a question, an answer, multi-dimensional evaluation checklists and associated labels, which include correctness, multi-dimensional scores and final scores. Furthermore, each sample includes supplementary attributes such as partition identifiers, subset names and difficulty levels.

[Figure 14]

[Figure 15]

[Figure 16]

Question

[Figure 17]

[Figure 18]

###### Fine-Grained Evaluation Checklists

[Figure 19]

C# Algorithmic Question Problem Statement: You are tasked with creating a method in C# that filters a collection of objects based on a specified key and value. The method should be generic and type -safe, ensuring that the value provided matches the type of the property specified by the key. The method should return an IEnumerable<T> containing the filtered results. Function Signature: ```csharp public static IEnumerable<T> FilterByKey<T, TValue>(IEnumerable<T> data, Func<T, TValue> keySelector, TValue value) ``` ··· Task: Implement the FilterByKey method according to the specifications above. Ensure that your implementation is type-safe and handles all specified edge cases.

Correctness: Does the `FilterByKey` method correctly handle the case where the ` keySelector` returns `null` for some items in the collection, and the `value` is also `null`? … Reliability: Is the implementation of the ` FilterByKey` method based on reliable and verifiable evidence, such as the use of the `Equals` method for type -safe comparison?

[Figure 20]

[Figure 21]

###### Ground Label

[Figure 22]

###### Correcrness: Correct Multiple Evaluation Problems Rating : [10, 10, 10, 5, 9,

[Figure 23]

[Figure 24]

Answer

10, 9, 10, 9, 10] Final Rating: 9

[Figure 25]

```csharp … public class FilterByKeyExample {

public static void Main(string[] args) { …

[Figure 26]

[Figure 27]

###### Other Keys

var filteredById = FilterByKey(items, item => item.Id, "item-1");

[Figure 28]

###### Difficulty: Easy

Partition: Real Subset: Fundamental Programming

… Explanation: The FilterByKey method …

Figure 10. Example of correct case of code qa. Table 5. The effect of CoT on model evaluation.

Effect of CoT Evaluation. During the basic evaluation, we observe that using a single prompt to assess answer’s correctness often led to incorrect prediction. However, using fine-grained evaluation checklists, where the model scores each question individually before aggregating the results, produced more accurate outcomes. This improvement is due to the model having access to more detailed context. Based on this, we randomly select 400 instances (CodeCritic_400) and test four models of varying sizes. The CoT prompt used in this experiment is in Appendix B. As shown in Table 5, the use of the CoT leads to improvements in both ACC and MSE. Compared to the evaluation without CoT, Qwen2.5-Coder3B-Instruct shows gains of 7.5% in ACC and 1.31 in MSE, while Qwen2.5-Coder-14B-Instruct demonstrates improves of 15.5% in ACC and 3.13 in MSE. These results clearly indicate that providing more useful information enhances model evaluation accuracy, further validating the necessity of incorporating fine-grained evaluation metrics.

Model CoT ACC MSE Qwen2.5-Coder-1.5B-Instruct

###### √ 55.00 14.57 × 47.00 11.89

###### √ 54.75 8.33 × 47.25 9.64

Qwen2.5-Coder-3B-Instruct

###### √ 56.50 4.06

Qwen2.5-Coder-7B-Instruct

- × 49.75 4.31

Qwen2.5-Coder-14B-Instruct

√ 66.00 1.99

- × 50.50 5.12

Effect of Critique. To evaluate the effect of model-generated critiques in improving answers, we use four Qwen2.5-Coder models (1.5B, 3B, 7B, 14B) as critics, with GPT-4o as the evaluator and CodeCritic_400 as data. First, the critic model generates critiques for (𝑄, 𝐴) pairs based on fine-grained evaluation checklists. The evaluator then scores original pairs. Next, the polishing model refines answers using critiques and the evaluator scores the updated answers. The critic model varies, while the evaluator and the polishing model remain the same. As shown in Figure 11, applying critiques improves scores and critique quality increases with model size, aligning with the model’s capabilities.

7.0

6.88 6.91

6.76 6.79

6.8

Fine-GrainedScore

6.6

6.4

6.2

6.0

5.8

5.72

5.6

Original Qwen2.5-Coder 1.5B-Instruct

Qwen2.5-Coder 3B-Instruct

Qwen2.5-Coder 7B-Instruct

Qwen2.5-Coder 14B-Instruct

Figure 11. Scoring results of QA pairs before and after applying critiques to refine the answers.

##### 5. Conclusion

In this work, we introduce CodeCriticBench, a comprehensive benchmark designed to evaluate the code critique capabilities of large language models (LLMs) through two core tasks: code generation and code-based question answering (QA), each incorporating multiple levels of difficulty. Our benchmark comprises both basic and advanced evaluation metrics that target distinct aspects of LLM performance. In the basic evaluations, we assess whether the model can accurately judge the correctness of a given question-answer pair. For the advanced evaluations, we employ detailed, fine-grained checklists to facilitate a thorough assessment of the model’s critique abilities. Extensive testing of current LLMs with CodeCriticBench demonstrates its efficacy in measuring and comparing code critique performance across different models, thereby providing valuable insights for the continued refinement of foundational models.

##### Limitations

While CodeCriticBench comprehensively incorporates both basic and advanced evaluations for code-related tasks—including code generation and code question answering—it is not without limitations. For example, our evaluation is confined to single-file scenarios. In future work, we plan to extend it to encompass repository-level critiques. Furthermore, as CodeCriticBench is presently focused solely on code, we intend to broaden its scope to include additional domains, thereby enabling the assessment of critique capabilities across a wider range of tasks and application scenarios.

##### References

J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,

- S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

L. B. Allal, R. Li, D. Kocetkov, C. Mou, C. Akiki, C. M. Ferrandis, N. Muennighoff, M. Mishra, A. Gu, M. Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023.

Z. Ankner, M. Paul, B. Cui, J. D. Chang, and P. Ammanabrolu. Critique-out-loud reward models.

arXiv preprint arXiv:2408.11791, 2024. Anthropic. Introducing claude. https://www.anthropic.com/claude, 2024. J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry,

- Q. Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732,

###### 2021. URL https://arxiv.org/abs/2108.07732.

- S. Baars and S. Meester. Codearena: Inspecting and improving code quality metrics using minecraft. In 2019 IEEE/ACM International Conference on Technical Debt (TechDebt), pages 68–70. IEEE, 2019.

- G. Bai, J. Liu, X. Bu, Y. He, J. Liu, Z. Zhou, Z. Lin, W. Su, T. Ge, B. Zheng, et al. Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. arXiv preprint arXiv:2402.14762, 2024.

- M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan, H. Edwards, Y. Burda,
- N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji, S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. Knight, M. Brundage,

- M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and W. Zaremba. Evaluating large language models trained on code, 2021. URL https: //arxiv.org/abs/2107.03374.

- A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten,

- A. Yang, A. Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li, et al. Deepseekcoder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

- B. Hui, J. Yang, Z. Cui, J. Yang, D. Liu, L. Zhang, T. Liu, J. Zhang, B. Yu, K. Dang, et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024.

- N. Jain, K. Han, A. Gu, W.-D. Li, F. Yan, T. Zhang, S. I. Wang, A. Solar-Lezama, K. Sen, and

- I. Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. ArXiv, abs/2403.07974, 2024. URL https://api.semanticscholar.org/Corp usID:268379413.

P. Ke, B. Wen, A. Feng, X. Liu, X. Lei, J. Cheng, S. Wang, A. Zeng, Y. Dong, H. Wang, et al. Critiquellm: Towards an informative critique generation model for evaluation of large language model generation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13034–13054, 2024.

- T. Lan, W. Zhang, C. Lyu, S. Li, C. Xu, H. Huang, D. Lin, X.-L. Mao, and K. Chen. Training language models to critique with multi-agent feedback. arXiv preprint arXiv:2410.15287, 2024a.

- T. Lan, W. Zhang, C. Xu, H. Huang, D. Lin, K. Chen, and X.-L. Mao. Criticeval: Evaluating largescale language model as critic. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b.

- R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone, C. Akiki, J. Li, J. Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.

Z. Lin, Z. Gou, T. Liang, R. Luo, H. Liu, and Y. Yang. Criticbench: Benchmarking llms for critique-correct reasoning, 2024.

- J. Liu, Z. Ni, H. Que, T. Sun, N. Wang, J. Yang, JiakaiWang, H. Guo, Z. Peng, G. Zhang, J. Tian, X. Bu, K. Xu, W. Rong, J. Peng, and Z. Zhang. Roleagent: Building, interacting, and benchmarking high-quality role-playing agents from scripts. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https: //openreview.net/forum?id=hORTHzt2cE.

J. Liu, Z. ZhiqiBai, Y. Zhang, C. Zhang, Y. YuangZh, G. Zhang, J. JiakaiWang, H. Que, Y. Chen,

- W. Su, T. Ge, J. Fu, W. Chen, and B. Zheng. E2-LLM: Efficient and extreme length extension of large language models. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 4243–4253, Bangkok, Thailand, Aug. 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.252. URL https://aclanthology.org/2024.findings-acl.252/.

- S. Liu, L. Chai, J. Yang, J. Shi, H. Zhu, L. Wang, K. Jin, W. Zhang, H. Zhu, S. Guo, et al. Mdeval: Massively multilingual code debugging. arXiv preprint arXiv:2411.02310, 2024c.

S. Liu, H. Zhu, J. Liu, S. Xin, A. Li, R. Long, L. Chen, J. Yang, J. Xia, Z. Peng, et al. Fullstack bench: Evaluating llms as full stack coder. arXiv preprint arXiv:2412.00535, 2024d.

S. Lu, D. Guo, S. Ren, J. Huang, A. Svyatkovskiy, A. Blanco, C. Clement, D. Drain, D. Jiang, D. Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664, 2021.

Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and D. Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.

- A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder, K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= S37hOerQLB.

N. McAleese, R. M. Pokorny, J. F. C. Uribe, E. Nitishinskaya, M. Trebacz, and J. Leike. Llm critics help catch llm bugs. arXiv preprint arXiv:2407.00215, 2024.

###### OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o, 2024.

- H. Que, F. Duan, L. He, Y. Mou, W. Zhou, J. Liu, W. Rong, Z. M. Wang, J. Yang, G. Zhang, et al. Hellobench: Evaluating long text generation capabilities of large language models. arXiv preprint arXiv:2409.16191, 2024.

- B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

- B. Rozière, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez, J. Rapin, A. Kozhevnikov, I. Evtimov, J. Bitton, M. Bhatt, C. C. Ferrer, A. Grattafiori, W. Xiong, A. Défossez, J. Copet, F. Azhar, H. Touvron, L. Martin, N. Usunier, T. Scialom, and G. Synnaeve. Code llama: Open foundation models for code. arXiv preprint arXiv: 2308.12950, 2023.

- W. Saunders, C. Yeh, J. Wu, S. Bills, L. Ouyang, J. Ward, and J. Leike. Self-critiquing models for assisting human evaluators. arXiv preprint arXiv:2206.05802, 2022.

A. Sharma, S. Keh, E. Mitchell, C. Finn, K. Arora, and T. Kollar. A critical evaluation of ai feedback for aligning large language models. arXiv preprint arXiv:2402.12366, 2024.

M. Song, Z. yu Su, X. Qu, J. Zhou, and Y. Cheng. Prmbench: A fine-grained and challenging benchmark for process-level reward models. 2025a.

- X. Song, Y. Wu, W. Wang, J. Liu, W. Su, and B. Zheng. Progco: Program helps self-correction of large language models. ArXiv, abs/2501.01264, 2025b. URL https://api.semanticscho lar.org/CorpusID:275212297.

S. Tan, S. Zhuang, K. Montgomery, W. Y. Tang, A. Cuadron, C. Wang, R. A. Popa, and I. Stoica. Judgebench: A benchmark for evaluating llm-based judges. ArXiv, abs/2410.12784, 2024.

Z. Tang, Z. Li, Z. Xiao, T. Ding, R. Sun, B. Wang, D. Liu, F. Huang, T. Liu, B. Yu, et al. Realcritic: Towards effectiveness-driven evaluation of language model critiques. arXiv preprint arXiv:2501.14492, 2025.

L. Team. The llama 3 herd of models. arXiv preprint arXiv: 2407.21783, 2024. H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra,

P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

P. Wang, Y. Wu, N. Wang, J. Liu, X. Song, Z. Peng, K. Deng, C. Zhang, JiakaiWang, J. Peng, G. Zhang, H. Guo, Z. Zhang, W. Su, and B. Zheng. MTU-bench: A multi-granularity tool-use benchmark for large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=6guG2OlXsr.

L. Yang, Z. Yu, T. Zhang, M. Xu, J. E. Gonzalez, B. Cui, and S. Yan. Supercorrect: Supervising and correcting language models with error-driven insights. arXiv preprint arXiv:2410.09008, 2024.

- X. Yue, T. Zheng, G. Zhang, and W. Chen. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548, 2024.

L. Zhang, A. Hosseini, H. Bansal, M. Kazemi, A. Kumar, and R. Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024.

- C. Zheng, Z. Zhang, B. Zhang, R. Lin, K. Lu, B. Yu, D. Liu, J. Zhou, and J. Lin. Processbench: Identifying process errors in mathematical reasoning. ArXiv, abs/2412.06559, 2024a. URL https://api.semanticscholar.org/CorpusID:274598010.

X. Zheng, J. Lou, B. Cao, X. Wen, Y. Ji, H. Lin, Y. Lu, X. Han, D. Zhang, and L. Sun. Critic-cot: Boosting the reasoning abilities of large language model via chain-of-thoughts critic. arXiv preprint arXiv:2408.16326, 2024b.

##### A. More Experimental Results

###### A.1. Anonymous Data Link

We have released our data on an anonymous website (https://anonymous.4open.scienc e/r/CodeCriticBench-D657/).

###### A.2. Relations Between Basic Correctness and Advanced Evaluations Scores

We present the fine-grained score distributions for both the “Code Gen” and “Code QA” subsets. As illustrated in Figures 12a and 12b, these distributions exhibit favorable characteristics, thereby affirming the validity of our dataset.

(a) The Code Gen subset. (b) The Code QA subset.

Figure 12. Rating distribution.

###### A.3. Case Visualization of CodeCriticBench

We present remaining case visualization. Specifically, Figures 10- 15 illustrate a correct example from the “Code Gen” subset, an error example from the “Code Gen” subset and an error example from the “Code QA” subset, respectively.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Question

[Figure 33]

###### Fine-Grained Evaluation Checklists

[Figure 34]

You are given a 0-indexed binary string s of length n on which you can apply two types of operations:

Correctness Verification : Can the code correctly handle all possible input scenarios, including edge cases (such as strings of length 1)?

- 1. Choose an index i and invert all characters from index 0 to index i (both inclusive), with a cost of i + 1
- 2. Choose an index i and invert all characters from index i to index n - 1 (both inclusive), with a cost of n - i Return the minimum cost to make all characters of the string equal. Invert a character means if its value is '0' it becomes '1' and vice-versa. Example 1: Input: s = \"0011\" Output: 2 Explanation: Apply the second operation with i = 2 to obtain s = \"0000\" for a cost of 2. It can be shown that 2 is the minimum cost to make all characters equal. … Constraints: 1 <= s.length == n <= 10^5, s[i] is either '0' or '1'

Time Complexity Optimization : Is the time complexity of the code O(n), and can it be further optimized to reduce unnecessary computations? … Maintainability: Is the code structure clear and conducive to subsequent optimization and expansion?

[Figure 35]

[Figure 36]

###### Ground Label

[Figure 37]

Correcrness: Correct Multiple Evaluation Problems Rating : [6, 7, 9, 7, 7, 6, 6, 8, 8, 8] Final Rating: 9

[Figure 38]

[Figure 39]

Answer

[Figure 40]

###### class Solution:

def minimumCost(self, s: str) -> int: n = len(s) cost = 0 # Calculate the cost to make all characters '0' for i in range(1, n):

[Figure 41]

[Figure 42]

###### Other Keys

[Figure 43]

if s[i] != s[i - 1]:

Partition: Algo Subset: Live Code Bench

###### Difficulty: Hard

cost += min(i, n - i) return cost

Figure 13. Example of correct case of code generation.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Question

[Figure 48]

###### Fine-Grained Evaluation Checklists

[Figure 49]

You are given two numeric strings num1 and num2 and two integers max_sum and min_sum. We denote an integer x to be good if num1 <= x <= num2, min_sum <= digit_sum(x) <= max_sum. Return the number of good integers. Since the answer may be large, return it modulo 10^9 + 7. Note that digit_sum(x) denotes the sum of the digits of x. Example 1: Input: num1 = "1", num2 = "12", min_sum = 1, max_sum = 8, Output: 11 Explanation: There are 11 integers whose sum of digits lies between 1 and 8 are 1,2,3,4,5,6,7,8,10,11, and 12. Thus, we return 11. … Constraints: 1 <= num1 <= num2 <= 10^22, 1 <= min_sum <= max_sum <= 400

Time Complexity Optimization : The code has extremely high time complexity when handling large -range inputs (e.g., `num1 = "1"`, `num2 = "10^22"`), … Algorithm Optimization : The code does not utilize dynamic programming or memoization techniques to optimize the computation process. How can these techniques be introduced to reduce redundant calculations?

[Figure 50]

[Figure 51]

###### Ground Label

[Figure 52]

Correcrness: Error Multiple Evaluation Problems Rating : [3, 3, 5, 4, 4, 4, 6, 5, 4, 3] Final Rating: 2

[Figure 53]

[Figure 54]

Answer

[Figure 55]

class Solution: def count(self, num1: str, num2: str, min_sum: int,

max_sum: int) -> int: MOD = 10**9 + 7 def digit_sum(x): return sum(int(d) for d in str(x))

[Figure 56]

[Figure 57]

###### Other Keys

… count = 0 for i in range(num1_int, num2_int + 1):

[Figure 58]

###### Difficulty: Medium

Partition: Algo Subset: Live Code Bench

if min_sum <= digit_sum(i) <= max_sum: count += 1 return count % MOD

Figure 14. Example of error case of code generation.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Question

[Figure 63]

###### Fine-Grained Evaluation Checklists

[Figure 64]

### Bash Algorithmic Question

Correctness: The script uses `grep -oP` with a lookbehind and lookahead to extract CSS rules. How does this approach handle cases where the `<style>` tag contains nested tags or comments … … Reliability: The script does not provide any error handling

**Problem Statement:** You are tasked with writing a Bash script that processes an SVG file and extracts the CSS rules from the `<style>` element within the SVG. … ···

**Edge Cases:**

- 1. SVG string with no `<style>` element.
- 2. SVG string with multiple `<style>` elements.
- 3. SVG string with nested elements and attributes.
- 4. SVG string with comments and whitespace.

or logging. How would you enhance the script to log errors, such as invalid input or failed regex matches, to improve its reliability and debuggability?

**Note:**

- - You may use tools like `grep`, `sed`, `awk`, or ` perl` for string processing.
- - Ensure your script is robust and handles various edge cases gracefully.

[Figure 65]

[Figure 66]

###### Ground Label

[Figure 67]

Correcrness: Error Multiple Evaluation Problems Rating : [4, 4, 4, 4, 6, 5, 4, 4, 5, 4] Final Rating: 4

[Figure 68]

[Figure 69]

Answer

[Figure 70]

To solve the problem of extracting CSS rules from an SVG string using a Bash script, we can leverage tools like grep and sed for string processing. The key steps involve searching for the <style> element and extracting its content, which contains the CSS rules. … This script should cover the basic requirements and handle typical edge cases effectively. Adjustments may be needed for more complex SVG structures or additional requirements.

[Figure 71]

[Figure 72]

###### Other Keys

[Figure 73]

Partition: Real Subset: Fundamental Programming

###### Difficulty: Medium

Figure 15. Example of error case of code qa.

###### A.4. Scaling Law

We present the scaling law results for the MSE in advanced critique evaluation. As depicted in Figure 16, an increase in model parameters is accompanied by a steady decrease in MSE, which aligns well with our expectations and further validates the rationality of our dataset.

###### A.5. Different Application Scenes

We present a comparative analysis of the advanced evaluation MSE across various models and application scenarios within the “Code QA” subset. As depicted in Figure 17, an increase in model parameters is generally accompanied by a steady decrease in MSE. Notably, Claude3.5Sonnet consistently achieves lower MSE across nearly all scenarios, which may be attributed to its specialized optimizations for code and dialogue contexts.

###### A.6. Bug Identification

Table 6 presents the performance metrics of evaluated models across the 23 distinct error categories comprising our Debug evaluation subset. These categories span critical software integrity domains, including but not limited to “Input Validation and Data Processing Error”,

*

* *

*

* *

*

*

- Figure 16. Scaling law on advanced critique evaluation (MSE) across models. “*” indicates an estimated parameter size.

Fundamental Programming

Advanced Programming

Software Engineering

Data Analysis

Mathematics Desktop and Web Development

Machine Learning

Scientific Computing

Databases Multimedia Operating Systems

- 1

- 2

- 3

- 4

- 5

- 6

Score(%)

3.8

2.7

4.7

4.9

3.6

3.2

5.7

4.8

3.7

3.2

4.8

2.4

4.4

6.1

2.6

5.9

2.6

3.8

2.4

3.9

2.3

3.7 3.1

2.5

3.9

2.9 2.9

2.1

4.3

3.2

1.7

3.0

2.3 1.7

0.8

1.1

0.8

1.1 1.0 1.2

0.9 0.8 0.8

1.0

Comparison Across Different Models on "Code QA" (Advanced Critique Evaluation)

Qwen2.5-Coder-7B-Instruct

Qwen2.5-Coder-14B-Instruct

| |
|---|

Qwen2.5-Coder-32B-Instruct

Claude3.5-Sonnet

- Figure 17. Comparison across different models on “Code QA” (Advanced Critique Evaluation).

“Security Vulnerabilities” and “Reference Error”, providing a comprehensive assessment of model prediction’s robustness in practical debugging scenarios.

###### A.7. Critique Evaluation on “Code Gen” and “Code QA” Subset

We present more fine-grained scores for all the models used in our experiments across different scenarios. Tables 9 and 10 display the ACC and MSE of basic evaluation for each subset under the “Code Gen” subset. Tables 11 and 12 show the ACC and MSE of basic evaluation for each application scenario within the “Code QA” subset.

Note that we have adopted the following abbreviations in our tables to denote various application scenarios: Fundamental Programming (FP), Advanced Programming (AP), Software Engineering (SE), Data Analysis (DA), Mathematics (MA), Desktop and Web Development (DW), Machine Learning (ML), Scientific Computing (SC), Databases (DS), Multimedia (MM) and Operating Systems (OS).

###### A.8. Critique Evaluation on Multiple Fine-Grained Dimensions

Our Code Gen subset incorporates 10 fine-grained evaluation dimensions, including [“Correctness Verification”, “Time Complexity Optimization”, “Space Complexity”, “Code Readability”,

Table 6. The accuracy of different models in identifying programming error types.

Open-Sourced Close-Sourced Qwen2.5-Coder 0.5B-Instruct

Error Category

Qwen2.5-Coder 1.5B-Instruct

Qwen2.5-Coder 3B-Instruct

Qwen2.5-Coder 7B-Instruct

Qwen2.5-Chat 7B-Instruct

Qwen2.5-Coder 14B-Instruct

Qwen2.5-Chat 14B-Instruct

Qwen2.5 72B-Instruct

Claude3.5 Sonnet

Doubao-Coder Preview

GLM-4 Plus

Qwen2.5 Max

GPT 4o

All 21.50 32.75 48.00 61.00 41.00 62.25 47.50 61.25 61.00 54.00 67.00 47.75 59.50 Configuration Management Error 50.57 3.45 1.15 2.30 3.45 3.45 1.15 0.00 1.15 0.00 3.45 0.00 3.45 Data Management Error 40.00 5.45 10.91 7.27 3.64 3.64 7.27 10.91 0.00 3.64 1.82 12.73 7.27 Input Validation and Data Processing Error 7.41 31.48 42.59 27.78 37.04 46.30 20.37 50.00 37.04 33.33 22.22 9.26 37.04 Monitoring and Logging Management Error 1.85 0.00 0.00 1.85 0.00 0.00 0.00 0.00 0.00 5.56 5.56 1.85 1.85 Environment Variable Error 0.00 0.00 20.00 20.00 40.00 30.00 30.00 0.00 40.00 10.00 10.00 10.00 20.00 Dependency Management Error 0.00 0.00 25.00 29.17 4.17 33.33 33.33 16.67 25.00 16.67 25.00 20.83 25.00 Syntax Error 19.64 67.86 37.50 25.00 28.57 51.79 30.36 39.29 21.43 14.29 41.07 23.21 26.79 Design Flaw 2.63 0.00 13.16 10.53 2.63 2.63 18.42 7.89 0.00 18.42 7.89 10.53 5.26 Security Vulnerability 12.73 25.45 23.64 27.27 21.82 41.82 27.27 38.18 45.45 47.27 47.27 52.73 40.00 Log Security Issue 0.00 0.00 0.00 0.00 0.00 0.00 11.11 0.00 0.00 0.00 11.11 11.11 0.00 Reference Error 3.75 2.50 8.75 41.25 20.00 40.00 38.75 40.00 52.50 41.25 48.75 38.75 45.00 Session Management Error 0.00 14.29 14.29 33.33 14.29 23.81 14.29 14.29 4.76 9.52 14.29 9.52 19.05 Code Quality and Maintenance Error 2.04 0.00 0.00 4.08 0.00 0.00 4.08 8.16 8.16 12.24 14.29 2.04 6.12 Logic Error 1.35 24.32 29.73 31.08 40.54 27.03 25.68 31.08 36.49 22.97 36.49 18.92 31.08 Testing and Verification Error 3.51 0.00 0.00 5.26 0.00 3.51 0.00 3.51 5.26 5.26 15.79 1.75 3.51 Network and Communication Error 0.00 14.75 9.84 14.75 4.92 9.84 0.00 3.28 6.56 3.28 8.20 4.92 6.56 Exception Handling Error 2.38 4.76 4.76 4.76 0.00 0.00 7.14 9.52 2.38 4.76 4.76 2.38 4.76 User Permission and Authentication Error 0.00 6.25 4.69 15.62 7.81 14.06 12.50 15.62 17.19 15.62 21.88 14.06 15.62 File and I/O Error 0.00 10.96 24.66 31.51 17.81 31.51 20.55 20.55 32.88 1.37 26.03 24.66 32.88 Type Error 0.00 21.43 33.33 38.10 7.14 23.81 33.33 45.24 38.10 26.19 26.19 30.95 30.95 Internationalization and Localization Error 1.47 1.47 5.88 14.71 2.94 8.82 7.35 4.41 7.35 7.35 10.29 7.35 7.35 Performance Issue 0.00 2.30 14.94 12.64 1.15 16.09 11.49 19.54 13.79 29.89 27.59 12.64 16.09 Concurrency and Multithreading Error 0.00 23.21 57.14 85.71 60.71 76.79 41.07 80.36 69.64 75.00 66.07 55.36 64.29

“Robustness Validation”, “Algorithm Optimization”, “Comprehensive Testing”, “Output Format”, “Code Style Consistency”, “Maintainability”], while the Code QA subset is assessed across 10 distinct dimensions, including [“Correctness”, “Completeness”, “Performance”, “Maintainability”, “Clarity”, “Depth”, “Practicality”, “Logic Coherence”, “Innovation”, “Reliability”].

Given that our advanced evaluation includes multiple evaluation dimensions, we further measure the MSE results across these dimensions for both the “Code Gen” subset and “Code QA” subset of the dataset. Table 13 presents the results for all evaluation dimensions within the “Code Gen” subset. Tables 14-17 show the results for each specific evaluation dimension across different subsets. Table 18 displays the results for all evaluation dimensions within the “Code QA” subset, while Tables 19-29 provide the results for each evaluation dimension within various specific application scenarios.

##### B. Prompt

We present all the evaluation prompts utilized in our experiment. Specifically, Figure 18 displays the prompt for basic evaluation ACC, Figure 19 presents the prompt for advanced evaluation MSE and Figure 22 shows the prompt for assessing the model’s ability to identify the corresponding code error type. Figures 26 and 27 illustrate the prompts for basic and advanced evaluations using the CoT method, respectively. Additionally, Figure 28 provides the prompt used for polishing and modifying the answer based on feedback, which is employed to verify the effectiveness of the critique.

To enhance the transparency and credibility of our data construction process, we have made public the specific prompts used to create the Debug dataset subset, as well as the prompts employed for generating customized, fine-grained evaluation questions for each questionanswer pair (𝑄, 𝐴) in the advanced evaluation phase. In particular, Figure 20 illustrates the types of programming errors we identified and categorized, encompassing 23 major categories and over 110 subcategories. Figure 21 presents the corresponding prompts used to insert these programming errors. The detailed classification within our Code QA scenario was also generated through prompts utilizing large language models (LLMs), as shown in Figure 23. Lastly, Figures 24 and 25 display the prompts for generating customized, fine-grained evaluation questions for each question-answer pair (𝑄, 𝐴) in the Code Generation and Code QA scenarios.

##### C. Model Lists

Our experiments utilizes a diverse set of inference models, spanning both closed-source and open-source architectures and covering a broad spectrum of sizes. Specifically, Table 7 details the open-source models employed in our study, including those from the Qwen series (Qwen2.5Coder/Chat, QwQ, CodeQwen), the LLaMA series (CodeLlama, Llama), the DeepSeek series (DeepSeek-Coder/Chat), as well as the OpenCoder, Yi-Coder series and StarCoder2. These models range in size from 0.5B to over 200B, ensuring extensive coverage across different scales. Table 8, on the other hand, presents the closed-source models used, which encompass some of the most powerful models available to date, such as Claude3.5-Sonnet, GPT 4o-mini, GPT 4o, Gemini2.0-Flash-Thinking, DeepSeek-v3, GLM-4-Plus, Qwen-Max and the current popular o1-like models, including OpenAI o1-Preview, OpenAI o1-mini and DeepSeek-R1.

Table 7. Open-sourced models adopted in our experiments.

Open-Sourced Model Model Link CodeQwen1.5-7B-Instruct https://hf.co/Qwen/CodeQwen1.5-7B-Instruct

- Qwen2.5-Coder-0.5B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-0.5B-Instruct

- Qwen2.5-Coder-1.5B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen2.5-Coder-3B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-3B-Instruct Qwen2.5-Coder-7B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-7B-Instruct Qwen2.5-Coder-14B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-14B-Instruct Qwen2.5-Coder-32B-Instruct https://hf.co/Qwen/Qwen2.5-Coder-32B-Instruct Qwen2.5-Chat-7B-Instruct https://hf.co/Qwen/Qwen2.5-7B-Instruct Qwen2.5-Chat-14B-Instruct https://hf.co/Qwen/Qwen2.5-14B-Instruct Qwen2.5-Chat-32B-Instruct https://hf.co/Qwen/Qwen2.5-32B-Instruct Qwen2.5-72B-Instruct https://hf.co/Qwen/Qwen2.5-72B-Instruct QwQ-32B-Preview https://hf.co/Qwen/QwQ-32B-Preview

CodeLlama-7B-Instruct https://hf.co/meta-llama/CodeLlama-7b-Instruct-hf CodeLlama-13B-Instruct https://hf.co/meta-llama/CodeLlama-13b-Instruct-hf CodeLlama-34B-Instruct https://hf.co/meta-llama/CodeLlama-34b-Instruct-hf Llama3.3-70B-Instruct https://hf.co/meta-llama/Llama-3.3-70B-Instruct

DeepSeek-Coder-1.3B-Instruct https://hf.co/deepseek-ai/deepseek-coder-1.3b-instruct DeepSeek-Coder-V2-Lite-Instruct https://hf.co/deepseek-ai/DeepSeek-Coder-V2-Lite-instruct DeepSeek-Coder-V2-Instruct https://hf.co/deepseek-ai/DeepSeek-Coder-V2-Instruct DeepSeek-v2-Lite-Chat https://hf.co/deepseek-ai/DeepSeek-V2-Lite-Chat DeepSeek-v2-Chat https://hf.co/deepseek-ai/DeepSeek-V2-Chat DeepSeek-v2.5 https://hf.co/deepseek-ai/DeepSeek-V2.5 DeepSeek-R1-Distill-Qwen-32B https://hf.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

OpenCoder-1.5B-Instruct https://hf.co/infly/OpenCoder-1.5B-Instruct OpenCoder-8B-Instruct https://hf.co/infly/OpenCoder-8B-Instruct Yi-Coder-1.5B-Chat https://hf.co/01-ai/Yi-Coder-1.5B-Chat Yi-Coder-9B-Chat https://hf.co/01-ai/Yi-Coder-9B-Chat StarCoder2-15B-Instruct-v0.1 https://hf.co/bigcode/starcoder2-15b-instruct-v0.1

Table 8. Close-sourced models (APIs) adopted in our experiments.

Close-Sourced Model API Entry

Claude-3.5-Sonnet https://www.anthropic.com/news/claude-3-5-sonnet OpenAI o1-Preview https://platform.openai.com/docs/models#o1 OpenAI o1-mini https://platform.openai.com/docs/models#o1 GPT 4o-mini https://platform.openai.com/docs/models#gpt-4o-mini GPT 4o https://platform.openai.com/docs/models#gpt-4o Gemini2.0-Flash-Thinking https://ai.google.dev/gemini-api/docs/thinking Doubao-Coder-Preview https://www.volcengine.com/product/doubao DeepSeek-v3 https://www.deepseek.com DeepSeek-R1 https://www.deepseek.com GLM-4-Plus https://open.bigmodel.cn/dev/api/normal-model/glm-4 Qwen-Max https://www.aliyun.com/product/bailian

- Table 9. Results of different models on basic critique evaluations ACC in the Code Gen Subset Dataset.

Model All MBPP CodeForce LiveCodeBench Debug

0.5B+ Instruction Tuned Model Qwen2.5-Coder-0.5B-Instruct 51.28 50.88 50.12 52.88 51.25 1B+ Instruction Tuned Model

Yi-Coder-1.5B-Chat 51.56 61.75 51.50 42.75 50.25 DeepSeek-Coder-1.3B-Instruct 52.31 54.00 52.75 51.38 51.12

OpenCoder-1.5B-Instruct 53.06 57.38 51.62 53.87 49.38 Qwen2.5-Coder-1.5B-Instruct 53.87 56.25 50.62 50.50 58.13

Qwen2.5-Coder-3B-Instruct 55.22 56.25 56.12 52.38 56.12 6B+ Instruction Tuned Model

Qwen2.5-Chat-7B-Instruct 51.81 49.88 52.00 54.00 51.38 CodeLlama-7B-Instruct 52.16 51.25 52.50 50.25 54.62 CodeQwen1.5-7B-Chat 55.41 58.63 52.75 53.25 57.00

Qwen2.5-Coder-7B-Instruct 56.78 60.38 56.38 59.13 51.25 OpenCoder-8B-Instruct 57.66 58.75 56.12 52.38 63.38 Yi-Coder-9B-Chat 63.66 61.12 64.50 66.62 62.38

13B+ Instruction Tuned Model

CodeLlama-13B-Instruct 52.16 50.75 52.50 50.31 55.16 StarCoder2-15B-Instruct 53.59 54.25 53.50 50.50 56.12

Qwen2.5-Coder-14B-Instruct 56.38 57.75 57.25 59.62 50.88 Qwen2.5-Chat-14B-Instruct 58.59 56.62 62.62 58.25 56.88

DeepSeek-v2-Lite-Chat 58.78 61.75 58.63 62.14 51.83 DeepSeekCoder-v2-Lite-Instruct 59.34 63.25 57.50 60.25 56.38

20B+ Instruction Tuned Model

CodeLlama-34B-Instruct 54.06 56.00 52.12 47.15 61.08 Qwen2.5-Coder-32B-Instruct 59.00 58.75 57.88 66.38 53.00

Qwen2.5-Chat-32B-Instruct 62.38 60.62 63.12 65.50 60.25 70B+ Instruction Tuned Model DeepSeek-v2.5 58.46 55.00 60.25 64.62 53.94

DeepSeekCoder-v2-Instruct 62.42 60.62 63.33 66.38 59.29 Llama3.3-70B-Instruct 65.16 61.00 67.88 55.62 76.12 Qwen2.5-72B-Instruct 68.44 66.88 70.12 67.88 68.88

Close-Sourced API Model GPT 4o-mini 58.31 53.25 57.25 66.00 56.75

Doubao-Coder-Preview 60.06 61.75 60.88 58.00 59.62 GLM-4-Plus 60.94 53.50 59.62 72.25 58.38 DeepSeek-v3 61.43 60.88 64.88 61.62 58.32

Qwen2.5-Max 62.74 59.70 61.12 71.50 58.63 Claude3.5-Sonnet 66.06 58.00 70.88 68.88 66.50

GPT 4o 67.56 63.88 73.75 64.62 68.00 o1-like Model

QwQ-32B-Preview 56.59 59.00 54.87 56.05 56.46 Gemini2.0-Flash-Thinking 64.88 70.25 57.00 63.50 68.75

DeepSeek-R1-Distill-Qwen-32B 75.38 67.75 82.75 87.33 63.52 OpenAI o1-mini 76.06 64.00 82.88 87.75 69.62

DeepSeek-R1 79.09 65.75 86.12 89.25 75.06 OpenAI o1-Preview 80.53 65.75 84.75 91.38 80.25

- Table 10. Results of different models on advanced critique evaluations MSE in the Code Gen Subset Dataset.

Model All MBPP CodeForce LiveCodeBench Debug

0.5B+ Instruction Tuned Model Qwen2.5-Coder-0.5B-Instruct 23.96 20.75 24.77 27.18 23.13 1B+ Instruction Tuned Model

Yi-Coder-1.5B-Chat 28.27 27.28 26.58 31.29 27.94 OpenCoder-1.5B-Instruct 28.07 30.73 29.95 25.25 26.35

DeepSeek-Coder-1.3B-Instruct 24.48 21.12 23.93 28.95 23.94 Qwen2.5-Coder-1.5B-Instruct 17.13 14.52 18.26 18.58 17.14 Qwen2.5-Coder-3B-Instruct 13.01 9.88 15.00 14.47 12.67

6B+ Instruction Tuned Model

CodeLlama-7B-Instruct 20.89 19.80 23.16 20.48 20.08 OpenCoder-8B-Instruct 19.45 13.56 24.09 19.12 21.04 CodeQwen1.5-7B-Chat 19.06 16.47 20.99 21.19 17.59

Yi-Coder-9B-Chat 14.28 11.95 16.73 14.54 13.88

Qwen2.5-Chat-7B-Instruct 8.73 6.95 8.76 8.66 10.55 Qwen2.5-Coder-7B-Instruct 6.17 4.72 6.63 5.65 7.69

13B+ Instruction Tuned Model

StarCoder2-15B-Instruct 20.92 17.12 22.38 24.50 19.65 CodeLlama-13B-Instruct 15.55 14.64 18.24 14.80 14.57

DeepSeekCoder-v2-Lite-Instruct 6.46 5.82 7.89 5.53 6.62

DeepSeek-v2-Lite-Chat 6.35 5.75 7.40 5.59 6.81 Qwen2.5-Coder-14B-Instruct 5.57 4.41 6.00 4.61 7.27

Qwen2.5-Chat-14B-Instruct 5.02 4.28 5.09 5.00 5.70 20B+ Instruction Tuned Model

CodeLlama-34B-Instruct 15.06 17.23 16.98 14.24 11.80 Qwen2.5-Coder-32B-Instruct 5.19 4.45 6.23 3.28 6.78

Qwen2.5-Chat-32B-Instruct 5.09 4.03 5.10 5.14 6.08 70B+ Instruction Tuned Model

Llama3.3-70B-Instruct 5.65 4.45 5.25 7.17 5.74 DeepSeekCoder-v2-Instruct 5.14 4.59 5.23 5.50 5.22

DeepSeek-v2.5 4.78 4.11 4.63 5.27 5.08 Qwen2.5-72B-Instruct 4.61 4.42 4.46 4.67 4.88

Close-Sourced API Model

Doubao-Coder-Preview 7.07 5.68 7.75 5.67 9.16 Qwen2.5-Max 5.04 4.54 5.26 4.79 5.58

GPT 4o 5.04 4.51 4.87 5.37 5.40 GPT 4o-mini 4.82 4.61 5.04 5.10 4.52

Claude3.5-Sonnet 4.73 3.66 5.26 5.29 4.70 DeepSeek-v3 4.49 3.49 6.05 4.12 4.31 GLM-4-Plus 4.25 3.94 4.79 4.22 4.04

o1-like Model QwQ-32B-Preview 8.07 9.21 6.95 6.87 9.26

OpenAI o1-mini 6.08 5.87 6.27 6.54 5.67 OpenAI o1-Preview 5.68 6.21 5.73 4.44 6.33

DeepSeek-R1-Distill-Qwen-32B 5.25 4.91 5.82 4.12 6.13 Gemini2.0-Flash-Thinking 4.80 4.29 4.55 5.57 4.81 DeepSeek-R1 3.92 3.79 3.81 2.31 5.84

- Table 11. Results of different models on basic critique evaluations ACC in the Code QA Subset Dataset.

Size Model All FP AP SE DA MA DW ML SC DB MM OS 0.5B+ Qwen2.5-Coder-0.5B-Instruct 53.27 53.00 46.00 52.00 52.00 53.00 49.00 69.00 59.00 50.00 55.00 48.00

DeepSeek-Coder-1.3B-Instruct 48.91 50.00 47.00 46.00 51.00 46.00 47.00 53.00 48.00 51.00 52.00 47.00 Yi-Coder-1.5B-Chat 49.27 48.00 55.00 43.00 61.00 44.00 47.00 44.00 44.00 44.00 61.00 51.00 OpenCoder-1.5B-Instruct 53.45 62.00 57.00 55.00 58.00 47.00 42.00 58.00 55.00 49.00 52.00 53.00 Qwen2.5-Coder-3B-Instruct 55.18 47.00 52.00 57.00 58.00 53.00 50.00 75.00 57.00 49.00 59.00 50.00 Qwen2.5-Coder-1.5B-Instruct 56.18 53.00 56.00 62.00 62.00 51.00 47.00 68.00 60.00 43.00 62.00 54.00

1B+

OpenCoder-8B-Instruct 52.64 48.00 48.00 50.00 54.00 46.00 48.00 75.00 51.00 52.00 57.00 50.00

Yi-Coder-9B-Chat 55.91 51.00 59.00 58.00 54.00 50.00 49.00 72.00 56.00 52.00 61.00 53.00 CodeLlama-7B-Instruct 56.00 40.00 65.00 56.00 56.00 53.00 48.00 76.00 55.00 49.00 58.00 60.00 CodeQwen1.5-7B-Chat 56.27 46.00 65.00 58.00 61.00 50.00 54.00 64.00 60.00 51.00 61.00 49.00

6B+

Qwen2.5-Coder-7B-Instruct 61.36 66.00 69.00 62.00 62.00 62.00 56.00 65.00 63.00 52.00 67.00 51.00 Qwen2.5-Chat-7B-Instruct 62.82 57.00 72.00 70.00 61.00 65.00 52.00 67.00 69.00 57.00 67.00 54.00

CodeLlama-13B-Instruct 52.18 52.00 51.00 53.00 49.00 46.00 53.00 65.00 54.00 47.00 57.00 47.00 StarCoder2-15B-Instruct 52.97 45.83 56.00 61.00 54.00 53.00 54.00 54.55 49.00 48.00 54.00 53.00

DeepSeek-v2-Lite-Chat 60.42 53.00 67.00 61.00 61.00 64.00 52.00 77.00 67.00 52.00 57.58 53.00 DeepSeekCoder-v2-Lite-Instruct 61.18 54.00 55.00 64.00 64.00 62.00 55.00 71.00 70.00 60.00 63.00 55.00 Qwen2.5-Chat-14B-Instruct 64.00 61.00 62.00 69.00 54.00 72.00 64.00 70.00 62.00 58.00 67.00 65.00 Qwen2.5-Coder-14B-Instruct 66.64 72.00 70.00 70.00 62.00 73.00 59.00 70.00 69.00 54.00 73.00 61.00

13B+

CodeLlama-34B-Instruct 57.27 55.00 55.00 56.00 60.00 48.00 55.00 73.00 65.00 48.00 64.00 51.00 Qwen2.5-Chat-32B-Instruct 68.64 67.00 79.00 70.00 59.00 74.00 65.00 71.00 67.00 62.00 72.00 69.00 Qwen2.5-Coder-32B-Instruct 69.45 75.00 70.00 82.00 64.00 73.00 61.00 78.00 65.00 66.00 70.00 60.00

20B+

DeepSeek-v2.5 65.85 68.69 65.00 69.00 65.00 71.00 61.00 59.00 63.00 67.00 78.00 57.58 Llama3.3-70B-Instruct 68.09 67.00 73.00 68.00 69.00 73.00 67.00 78.00 65.00 66.00 66.00 57.00 Qwen2.5-72B-Instruct 68.09 68.00 65.00 68.00 68.00 76.00 61.00 78.00 70.00 57.00 77.00 61.00

70B+

DeepSeekCoder-v2-Instruct 70.23 66.67 71.00 74.00 67.00 70.00 73.00 70.71 71.00 66.00 79.00 64.00

GLM-4-Plus 63.35 66.00 64.00 65.00 57.00 67.00 65.96 58.00 62.00 59.00 72.00 61.00 DeepSeek-v3 63.64 71.00 59.00 66.00 59.00 66.00 65.00 59.00 61.00 62.00 67.00 65.00

Qwen2.5-Max 65.17 68.42 64.00 61.00 58.00 73.00 66.00 68.69 65.00 66.00 67.00 60.00 Doubao-Coder-Preview 65.36 67.00 72.00 63.00 62.00 69.00 65.00 62.00 58.00 66.00 69.00 66.00 GPT 4o-mini 67.09 66.00 61.00 64.00 67.00 74.00 68.00 68.00 70.00 63.00 73.00 64.00 GPT 4o 69.53 72.92 65.00 75.00 71.00 70.00 70.00 72.00 61.00 61.00 78.00 69.00 Claude3.5-Sonnet 76.73 73.00 87.00 78.00 75.00 87.00 70.00 80.00 78.00 70.00 86.00 60.00

Close-Sourced

DeepSeek-R1 54.36 62.86 58.00 53.00 51.00 59.00 60.00 37.00 56.00 60.00 53.61 49.43 QwQ-32B-Preview 58.82 68.00 63.00 56.00 58.00 56.00 60.00 57.00 59.00 53.00 61.00 56.00 OpenAI o1-mini 59.27 65.00 57.00 63.00 55.00 59.00 63.00 54.00 54.00 66.00 60.00 56.00 OpenAI o1-Preview 59.89 67.71 62.00 69.00 63.00 51.00 64.00 59.00 55.00 58.00 60.00 49.45 Gemini2.0-Flash-Thinking 63.55 65.00 57.00 68.00 65.00 61.00 64.00 63.00 57.00 62.00 74.00 63.00 DeepSeek-R1-Distill-Qwen-32B 64.09 69.00 68.00 66.00 62.00 65.00 61.00 66.00 56.00 62.00 71.00 59.00

o1-like

- Table 12. Results of different models on advanced critique evaluations MSE in the Code QA Subset Dataset.

Size Model All FP AP SE DA MA DW ML SC DB MM OS 0.5B+ Qwen2.5-Coder-0.5B-Instruct 25.06 25.92 24.92 24.91 29.49 25.20 25.52 18.71 28.84 24.07 23.42 24.64

Yi-Coder-1.5B-Chat 32.13 31.45 28.35 32.74 37.67 33.47 34.50 25.28 34.15 33.39 31.86 30.59 OpenCoder-1.5B-Instruct 26.23 26.21 24.05 27.31 29.19 28.31 24.83 20.11 34.02 22.59 25.08 26.87 DeepSeek-Coder-1.3B-Instruct 17.39 15.41 17.22 19.72 16.55 16.79 18.57 15.61 26.35 14.74 15.45 14.85

1B+

Qwen2.5-Coder-1.5B-Instruct 14.93 9.96 15.42 17.74 16.09 15.15 13.98 16.11 18.31 16.66 13.00 11.76 Qwen2.5-Coder-3B-Instruct 10.76 6.94 11.35 11.52 11.79 9.25 9.39 12.83 13.83 12.48 10.18 8.78

OpenCoder-8B-Instruct 18.99 17.38 15.87 24.59 18.98 20.28 20.20 15.64 21.13 16.81 18.02 19.99 CodeQwen1.5-7B-Chat 15.78 14.17 16.00 19.26 17.81 17.60 15.15 14.69 18.99 15.34 11.57 12.90

Yi-Coder-9B-Chat 14.02 13.05 14.55 18.27 13.07 12.81 11.24 16.02 16.71 11.19 14.10 13.25 CodeLlama-7B-Instruct 13.96 15.77 8.69 12.61 9.32 13.68 11.31 14.85 26.80 13.23 13.01 14.48

6B+

Qwen2.5-Coder-7B-Instruct 4.10 3.75 2.73 4.68 4.89 3.60 3.19 5.73 4.78 3.72 3.25 4.79 Qwen2.5-Chat-7B-Instruct 6.15 8.76 6.15 4.48 4.58 6.22 5.55 4.18 7.32 5.59 6.05 8.82

StarCoder2-15B-Instruct 22.88 21.39 24.64 23.79 28.60 22.30 25.07 19.56 25.62 21.86 20.58 18.22

CodeLlama-13B-Instruct 9.26 9.40 8.72 8.23 9.01 11.07 10.20 11.33 9.48 6.69 10.21 7.11 Qwen2.5-Coder-14B-Instruct 3.65 2.35 4.44 6.06 2.65 5.91 2.61 3.78 2.43 3.92 2.32 3.70 DeepSeekCoder-v2-Lite-Instruct 3.35 4.16 3.46 3.58 2.70 3.25 3.15 3.65 3.20 3.28 3.58 2.81 DeepSeek-v2-Lite-Chat 3.29 5.85 3.80 3.75 2.47 3.83 3.18 2.47 2.57 3.45 2.43 3.06 Qwen2.5-Chat-14B-Instruct 2.51 1.97 2.23 2.54 1.66 2.12 3.08 4.64 2.20 1.87 1.95 3.35

13B+

CodeLlama-34B-Instruct 8.76 7.20 6.84 8.04 9.40 7.23 14.64 8.31 9.01 6.21 12.33 7.13 Qwen2.5-Coder-32B-Instruct 2.89 3.09 2.46 3.92 2.86 2.91 2.13 4.27 3.23 1.66 2.97 2.27 Qwen2.5-Chat-32B-Instruct 2.09 2.50 1.94 1.24 2.78 1.74 1.64 2.77 3.14 1.45 1.41 2.36

20B+

Llama3.3-70B-Instruct 2.24 2.80 2.52 2.25 2.09 1.93 2.21 1.86 2.12 2.20 2.32 2.31 Qwen2.5-72B-Instruct 2.20 2.51 1.68 2.48 2.65 2.35 1.83 1.45 2.60 1.78 2.17 2.70

70B+

DeepSeek-v2.5 1.63 2.26 1.70 1.72 1.02 2.44 1.10 1.66 1.25 1.89 0.91 2.00 DeepSeekCoder-v2-Instruct 1.46 1.70 1.19 1.44 0.88 1.40 1.52 2.19 1.39 1.46 0.84 2.05

Doubao-Coder-Preview 4.90 5.56 4.24 4.61 6.19 4.79 4.08 5.89 4.69 4.97 4.50 4.37 GLM-4-Plus 1.69 2.34 1.33 1.82 2.10 1.49 1.85 1.73 1.42 1.55 1.16 1.82 GPT 4o 1.55 3.04 0.81 1.50 1.37 0.81 1.45 1.36 1.73 1.01 2.30 1.78

Close-Sourced

Qwen2.5-Max 1.33 1.48 1.07 1.88 1.72 1.17 1.40 1.01 1.11 1.07 1.24 1.43 GPT 4o-mini 1.30 2.26 1.37 1.10 1.18 1.17 1.49 1.74 0.91 1.19 0.84 1.01 DeepSeek-v3 1.18 1.56 0.54 1.00 0.78 0.77 0.73 2.51 1.13 1.17 1.37 1.41

Claude3.5-Sonnet 1.02 1.73 0.76 1.07 0.81 1.06 1.04 1.19 0.87 0.85 0.82 1.02

DeepSeek-R1 5.02 7.20 4.50 4.91 8.61 6.86 3.71 4.11 6.01 2.75 3.36 3.65 QwQ-32B-Preview 4.67 5.78 3.65 3.96 6.10 4.88 3.77 2.76 5.84 4.25 5.33 5.21 OpenAI o1-Preview 2.26 2.53 2.08 1.67 2.61 4.35 1.70 1.62 2.45 1.88 1.50 2.45 DeepSeek-R1-Distill-Qwen-32B 1.71 2.59 1.40 1.55 1.61 2.47 1.30 1.16 1.46 1.31 1.33 2.71 OpenAI o1-mini 1.54 1.84 1.62 1.30 1.26 1.19 1.12 1.73 1.19 0.92 1.01 3.71 Gemini2.0-Flash-Thinking 1.19 1.26 0.76 1.37 0.94 1.66 1.03 1.52 1.10 0.85 1.03 1.62

o1-like Model

- Table 13. Results of different models on advanced critique evaluations MSE in the Code Gen Subset Dataset across all fine-grained evaluation dimensions.

Size Model

Correctness Verification

Code Readability

Robustness Validation

Comprehensive Testing

Space Complexity

Code Style Consistency

Output Format

Maintain-ability

Time Complexity

Algorithm Optimization 0.5B+ Qwen2.5-Coder-0.5B-Instruct 35.17 30.70 26.72 24.09 50.49 35.20 61.39 36.78 42.97 42.84

1B+

Yi-Coder-1.5B-Chat 40.34 34.04 28.72 25.81 55.82 39.10 66.84 39.57 48.42 43.84 DeepSeek-Coder-1.3B-Instruct 37.59 30.75 26.10 25.41 50.82 35.65 61.07 37.07 43.62 40.34 OpenCoder-1.5B-Instruct 28.53 33.76 28.70 25.36 55.37 39.30 66.55 39.45 24.38 43.70 Qwen2.5-Coder-1.5B-Instruct 18.78 12.78 13.84 14.83 19.94 14.96 23.68 15.23 18.45 16.10 Qwen2.5-Coder-3B-Instruct 16.67 12.16 13.68 14.20 17.78 15.62 20.70 13.91 16.76 15.78

6B+

OpenCoder-8B-Instruct 27.22 24.39 22.05 23.03 36.40 27.75 44.76 29.13 32.16 30.65 CodeQwen1.5-7B-Chat 27.07 22.94 20.66 21.26 33.80 25.17 41.28 27.42 30.22 28.81 CodeLlama-7B-Instruct 19.55 17.24 16.93 17.62 26.16 21.55 33.11 22.13 21.25 22.62

Qwen2.5-Chat-7B-Instruct 17.46 7.28 10.81 8.32 10.14 10.35 15.32 9.14 10.70 11.56

Qwen2.5-Coder-7B-Instruct 12.95 3.90 8.05 5.09 6.97 5.91 11.01 5.28 7.83 8.44 Yi-Coder-9B-Chat 11.53 10.29 11.16 10.93 8.29 13.23 10.57 11.63 9.08 9.87

13B+

StarCoder2-15B-Instruct 29.73 26.86 24.43 22.81 42.03 32.07 52.33 32.50 36.79 34.92 CodeLlama-13B-Instruct 15.26 11.76 13.91 13.81 14.14 14.66 17.48 13.99 13.76 13.24

DeepSeekCoder-v2-Lite-Instruct 9.10 5.06 7.54 6.04 6.76 5.70 8.79 5.34 7.09 6.96 DeepSeek-v2-Lite-Chat 8.26 5.39 7.55 4.99 6.84 6.12 8.84 5.23 7.21 7.26 Qwen2.5-Coder-14B-Instruct 7.34 4.96 7.25 5.28 6.34 7.63 11.01 5.91 8.11 8.47 Qwen2.5-Chat-14B-Instruct 6.96 2.77 6.02 3.86 4.17 4.01 5.75 3.63 5.07 5.51

20B+

CodeLlama-34B-Instruct 13.41 9.45 12.11 12.18 11.52 11.78 13.92 10.97 10.77 11.12

Qwen2.5-Coder-32B-Instruct 7.40 3.67 5.83 4.32 4.75 5.71 8.97 5.13 6.49 7.33 Qwen2.5-Chat-32B-Instruct 4.68 2.58 4.59 3.34 3.52 4.66 5.14 3.19 4.24 4.40

70B+

DeepSeek-v2.5 5.74 2.21 4.45 2.56 2.84 3.11 3.88 3.03 3.69 3.59 Llama3.3-70B-Instruct 5.52 2.61 5.23 4.15 3.03 3.41 4.55 3.24 3.87 4.57 DeepSeekCoder-v2-Instruct 5.42 3.05 4.68 2.95 3.38 3.77 4.44 3.45 4.12 4.21 Qwen2.5-72B-Instruct 3.59 1.65 3.33 2.08 2.07 2.10 3.42 1.70 2.66 2.88

Close-Sourced

Doubao-Coder-Preview 9.43 5.82 9.23 10.41 7.29 9.18 11.63 9.15 8.44 9.66 DeepSeek-v3 6.63 3.12 4.31 3.76 5.28 4.51 6.79 4.39 5.34 5.70 GPT 4o-mini 5.02 1.47 4.04 3.45 3.58 2.76 4.15 2.46 3.79 4.72

GPT 4o 4.13 2.03 4.11 2.48 2.94 2.52 4.06 2.24 3.03 3.32 Qwen2.5-Max 4.00 1.55 3.55 2.79 2.34 2.20 3.04 1.74 2.59 2.92 GLM-4-Plus 3.92 1.81 4.08 2.16 2.73 2.15 3.55 1.90 3.24 3.41 Claude3.5-Sonnet 3.47 1.95 3.54 2.46 2.25 3.44 3.91 2.89 2.66 3.48

o1-like Model

DeepSeek-R1 9.07 6.01 10.09 7.19 7.21 9.47 10.33 5.31 7.86 7.90 QwQ-32B-Preview 8.97 6.26 9.44 7.55 9.97 7.81 13.18 8.37 9.13 9.56 OpenAI o1-mini 6.28 3.76 5.67 4.10 5.56 3.95 6.64 4.05 4.85 5.22 OpenAI o1-Preview 6.06 3.52 6.23 3.55 4.18 3.13 5.45 2.80 4.12 4.31 DeepSeek-R1-Distill-Qwen-32B 5.80 2.54 6.35 3.94 3.34 3.50 4.50 3.00 3.59 4.48 Gemini2.0-Flash-Thinking 5.71 2.82 6.47 3.66 3.74 3.10 4.74 3.33 3.94 4.83

- Table 14. Results of different models on advanced critique evaluations MSE in the Code Gen’s MBPP subset Dataset across all fine-grained evaluation dimensions.

Algorithm Optimization 0.5B+ Qwen2.5-Coder-0.5B-Instruct 26.43 23.94 17.57 15.84 48.93 35.47 47.13 31.68 41.82 37.50

Correctness Verification

Code Readability

Robustness Validation

Comprehensive Testing

Space Complexity

Code Style Consistency

Output Format

Maintain-ability

Time Complexity

Size Model

Yi-Coder-1.5B-Chat 32.02 29.14 20.26 18.26 57.32 42.15 53.41 37.14 50.27 44.75 DeepSeek-Coder-1.3B-Instruct 27.81 24.03 17.17 18.85 48.77 35.64 46.03 32.95 41.76 38.65 OpenCoder-1.5B-Instruct 22.53 29.68 21.54 17.15 58.31 43.84 55.42 38.25 24.05 45.71 Qwen2.5-Coder-1.5B-Instruct 17.75 10.64 11.09 14.23 19.44 14.05 16.93 12.07 18.76 14.27 Qwen2.5-Coder-3B-Instruct 13.90 9.70 9.81 12.17 16.41 13.00 14.65 11.06 15.31 13.29

1B+

CodeQwen1.5-7B-Chat 20.23 18.18 15.72 19.31 28.67 22.33 28.19 22.62 26.22 25.67 OpenCoder-8B-Instruct 19.54 18.33 14.71 18.19 29.52 24.13 28.31 21.40 26.28 23.99

Qwen2.5-Chat-7B-Instruct 16.76 6.12 7.49 5.36 5.81 8.54 9.14 5.60 7.37 7.29

6B+

CodeLlama-7B-Instruct 14.65 14.61 13.38 15.92 24.01 19.61 24.78 17.76 17.31 19.82

Yi-Coder-9B-Chat 9.65 10.15 9.00 11.81 8.80 11.74 11.17 9.49 10.08 9.44 Qwen2.5-Coder-7B-Instruct 9.54 3.38 5.67 3.77 5.64 5.33 6.40 4.37 6.30 6.51

StarCoder2-15B-Instruct 22.82 22.11 18.80 17.21 40.65 32.56 39.51 28.27 36.23 33.50 CodeLlama-13B-Instruct 14.22 10.87 13.06 14.70 13.34 12.93 13.38 12.61 12.67 11.42

DeepSeekCoder-v2-Lite-Instruct 7.53 4.56 6.42 3.74 5.62 5.20 6.52 3.59 5.99 5.55 Qwen2.5-Chat-14B-Instruct 7.17 2.51 5.05 3.35 3.00 4.07 5.39 3.61 4.47 5.12 DeepSeek-v2-Lite-Chat 6.92 5.27 6.02 3.85 5.83 5.57 5.77 3.96 6.09 5.76 Qwen2.5-Coder-14B-Instruct 5.85 3.99 4.74 3.80 3.69 6.70 5.68 4.14 4.92 5.66

13B+

CodeLlama-34B-Instruct 13.39 10.93 13.00 15.18 13.83 12.62 14.58 11.26 11.43 12.70

20B+

Qwen2.5-Coder-32B-Instruct 6.24 2.90 4.30 3.46 3.52 4.39 6.47 3.77 5.07 5.69 Qwen2.5-Chat-32B-Instruct 4.22 1.66 3.58 3.08 1.98 3.96 4.25 2.32 2.72 3.76

Llama3.3-70B-Instruct 5.99 2.38 4.23 3.92 2.20 3.07 4.61 2.23 3.16 3.98 DeepSeekCoder-v2-Instruct 5.06 3.70 3.86 2.89 2.97 3.75 4.31 3.64 3.84 4.13 DeepSeek-v2.5 4.82 2.19 3.90 2.01 2.06 2.60 3.37 2.96 3.11 3.06 Qwen2.5-72B-Instruct 3.72 1.90 2.93 2.10 1.56 1.68 3.53 1.60 2.26 3.05

70B+

Doubao-Coder-Preview 8.97 4.78 5.90 7.02 5.20 8.41 10.21 7.88 6.49 7.87 DeepSeek-v3 5.12 1.86 2.89 2.50 3.04 3.44 4.71 3.48 3.64 4.56 GPT 4o-mini 5.01 1.37 2.76 2.95 3.59 3.08 5.04 2.40 4.10 5.09 GLM-4-Plus 3.95 1.70 3.24 2.19 1.89 1.59 2.93 1.55 2.83 3.62

Close-Sourced

Qwen2.5-Max 3.88 1.43 2.84 2.68 1.67 2.29 3.30 1.47 2.05 2.85 GPT 4o 3.69 1.99 3.23 2.43 1.98 2.29 3.58 1.87 2.41 2.81 Claude3.5-Sonnet 3.07 1.28 3.00 2.31 1.23 2.48 3.28 2.25 1.81 2.80

DeepSeek-R1 7.76 4.54 7.13 5.66 6.77 6.52 9.42 4.74 7.36 7.92 QwQ-32B-Preview 7.13 4.81 7.72 6.46 7.81 6.99 10.09 6.43 7.21 8.10 OpenAI o1-Preview 5.27 4.15 5.18 2.87 2.49 2.65 4.84 2.41 2.99 3.32 DeepSeek-R1-Distill-Qwen-32B 5.10 2.26 4.45 2.92 2.47 3.46 4.11 2.66 2.79 3.09 OpenAI o1-mini 5.00 3.35 4.60 3.75 4.29 3.53 4.62 3.20 3.92 3.87 Gemini2.0-Flash-Thinking 4.82 2.88 6.11 3.44 2.17 2.35 3.73 2.64 2.76 4.37

o1-like Model

- Table 15. Results of different models on advanced critique evaluations MSE in the Code Gen’s CodeForce subset Dataset across all fine-grained evaluation dimensions.

Size Model

Correctness Verification

Code Readability

Robustness Validation

Comprehensive Testing

Space Complexity

Code Style Consistency

Output Format

Maintain-ability

Time Complexity

Algorithm Optimization 0.5B+ Qwen2.5-Coder-0.5B-Instruct 34.15 27.62 26.64 23.15 51.75 35.96 58.01 31.43 46.22 50.37

1B+

Yi-Coder-1.5B-Chat 38.90 30.59 27.77 24.72 57.34 39.40 64.40 33.70 51.93 42.65 DeepSeek-Coder-1.3B-Instruct 37.69 29.16 26.22 24.49 53.43 37.18 60.31 32.40 48.45 40.60 OpenCoder-1.5B-Instruct 25.00 30.12 26.82 24.28 56.40 38.77 63.49 33.38 20.84 42.03 Qwen2.5-Coder-3B-Instruct 18.29 14.20 16.34 15.18 19.11 16.31 23.05 15.23 18.38 16.74 Qwen2.5-Coder-1.5B-Instruct 17.96 11.96 14.28 14.77 19.65 14.69 22.52 14.13 17.39 14.92

6B+

OpenCoder-8B-Instruct 29.19 24.30 24.07 23.59 40.27 30.71 46.84 28.38 36.51 33.33 CodeQwen1.5-7B-Chat 28.03 22.64 20.85 21.29 36.82 27.90 42.01 25.63 33.73 30.33 CodeLlama-7B-Instruct 21.40 18.21 18.33 18.53 29.30 23.66 33.51 22.19 24.09 24.56

Qwen2.5-Chat-7B-Instruct 16.45 6.83 10.92 7.75 10.81 10.74 16.22 9.59 12.06 11.77

Qwen2.5-Coder-7B-Instruct 13.26 3.95 9.11 5.10 7.41 6.17 11.91 6.10 9.06 8.46

Yi-Coder-9B-Chat 12.69 12.50 11.88 11.30 7.90 13.15 10.29 14.22 8.64 10.46

13B+

StarCoder2-15B-Instruct 30.09 24.96 24.25 22.72 42.45 32.61 51.13 28.87 38.93 34.46 CodeLlama-13B-Instruct 17.06 13.98 16.09 15.45 16.08 15.80 20.77 16.28 15.39 14.93 DeepSeek-v2-Lite-Chat 9.50 5.83 7.65 6.29 7.04 5.72 9.21 6.30 8.23 7.80

Qwen2.5-Coder-14B-Instruct 7.02 4.61 7.77 5.28 6.46 7.44 12.22 5.66 8.45 7.88 Qwen2.5-Chat-14B-Instruct 6.02 2.68 5.85 3.29 4.46 3.94 6.06 3.28 5.33 4.97 DeepSeekCoder-v2-Lite-Instruct 5.42 2.63 5.11 2.63 3.26 3.43 4.94 3.36 3.96 4.18

20B+

CodeLlama-34B-Instruct 15.57 10.31 13.31 12.51 11.74 11.45 14.30 11.69 10.86 10.64

Qwen2.5-Coder-32B-Instruct 7.85 4.31 6.78 4.28 5.82 6.79 11.28 5.66 7.83 7.59 Qwen2.5-Chat-32B-Instruct 4.10 2.61 4.80 3.26 3.54 5.15 5.08 3.13 4.56 3.92

70B+

DeepSeekCoder-v2-Instruct 8.82 5.99 7.46 4.90 6.80 5.73 10.40 6.13 7.91 7.53 Llama3.3-70B-Instruct 6.00 2.49 5.47 3.81 3.23 3.66 5.08 3.17 4.45 4.35 DeepSeek-v2.5 5.03 2.26 5.04 2.21 3.16 2.99 4.69 2.88 3.52 3.17 Qwen2.5-72B-Instruct 3.25 1.54 3.44 1.96 2.40 2.29 4.18 1.81 2.93 2.82

Close-Sourced

Doubao-Coder-Preview 8.73 7.42 10.25 9.80 7.57 9.16 13.04 10.09 9.68 10.23

DeepSeek-v3 8.44 4.40 5.63 4.97 8.01 6.13 9.72 5.56 8.11 7.56 GPT 4o-mini 5.03 1.46 4.42 3.33 4.07 2.73 4.36 2.35 4.04 4.84 GLM-4-Plus 3.93 1.53 4.59 2.07 2.99 2.13 3.72 1.96 3.30 2.97

Qwen2.5-Max 3.92 1.52 4.19 2.66 2.97 2.24 3.56 1.92 2.95 2.89 GPT 4o 3.85 2.01 4.66 2.28 3.33 2.69 4.87 2.23 3.36 3.14 Claude3.5-Sonnet 3.36 2.65 4.23 2.35 2.64 3.75 4.59 3.39 3.28 3.55

o1-like Model

DeepSeek-R1 9.10 6.55 11.55 7.28 7.16 6.79 10.55 4.88 8.23 7.08 QwQ-32B-Preview 9.02 4.89 9.52 6.40 9.83 6.85 11.55 6.14 9.08 8.16 OpenAI o1-Preview 6.71 3.07 6.86 3.53 5.16 3.28 6.01 2.91 4.78 4.56 OpenAI o1-mini 6.57 3.64 5.64 3.62 6.39 4.18 7.52 4.00 5.31 5.31 DeepSeek-R1-Distill-Qwen-32B 5.75 2.66 7.89 4.03 3.74 3.53 5.14 3.21 3.97 4.39 Gemini2.0-Flash-Thinking 5.33 2.99 6.56 3.63 4.59 3.36 5.54 3.69 4.64 4.72

- Table 16. Results of different models on advanced critique evaluations MSE in the Code Gen’s LiveCodeBench subset Dataset across all fine-grained evaluation dimensions.

Algorithm Optimization 0.5B+ Qwen2.5-Coder-0.5B-Instruct 45.79 49.24 40.35 29.94 47.71 - 75.26 48.04 35.84 36.23

Correctness Verification

Code Readability

Robustness Validation

Comprehensive Testing

Space Complexity

Code Style Consistency

Output Format

Maintain-ability

Time Complexity

Size Model

Yi-Coder-1.5B-Chat 51.04 52.77 42.89 31.57 50.98 - 79.92 50.79 38.91 38.13 DeepSeek-Coder-1.3B-Instruct 48.18 48.61 39.79 30.54 47.19 - 72.97 47.45 36.29 35.78 OpenCoder-1.5B-Instruct 35.07 51.96 41.85 31.65 51.09 - 79.21 50.33 16.39 38.30 Qwen2.5-Coder-1.5B-Instruct 21.61 16.57 18.42 16.07 17.96 - 27.35 17.89 15.98 15.79 Qwen2.5-Coder-3B-Instruct 16.58 11.20 15.68 14.37 11.94 - 15.83 11.92 11.02 11.96

1B+

CodeQwen1.5-7B-Chat 34.25 32.96 28.22 23.22 32.62 - 50.29 34.24 26.06 26.25 OpenCoder-8B-Instruct 33.91 35.04 30.15 26.34 34.22 - 52.35 35.80 28.32 28.16

Yi-Coder-9B-Chat 33.91 35.04 30.15 26.34 34.22 - 52.35 35.80 28.32 28.16 CodeLlama-7B-Instruct 23.81 21.98 21.45 18.14 23.34 - 36.81 25.05 18.43 20.51

6B+

Qwen2.5-Chat-7B-Instruct 17.80 8.66 15.02 9.36 8.86 - 14.02 8.76 7.28 8.96 Qwen2.5-Coder-7B-Instruct 12.91 4.15 10.25 4.84 5.40 - 8.32 4.42 4.90 5.67

StarCoder2-15B-Instruct 36.90 40.33 34.07 26.89 39.94 - 62.76 40.95 31.07 31.53 CodeLlama-13B-Instruct 14.85 9.48 13.67 12.34 11.73 - 15.63 11.99 12.02 11.43

DeepSeekCoder-v2-Lite-Instruct 10.10 4.10 9.10 6.49 6.07 - 8.38 5.03 5.28 5.80 DeepSeek-v2-Lite-Chat 8.12 3.79 9.36 4.83 5.22 - 6.28 4.27 4.65 6.19 Qwen2.5-Chat-14B-Instruct 7.33 3.24 8.22 4.43 4.59 - 5.53 3.97 4.70 5.27 Qwen2.5-Coder-14B-Instruct 7.00 5.92 9.09 5.28 5.17 - 9.57 6.16 6.51 6.52

13B+

CodeLlama-34B-Instruct 14.71 7.43 12.44 11.49 10.12 - 13.19 10.33 10.20 10.84

20B+

Qwen2.5-Coder-32B-Instruct 6.33 3.66 6.16 3.59 4.20 - 6.79 4.77 4.77 4.88 Qwen2.5-Chat-32B-Instruct 5.31 3.23 5.90 3.15 4.15 - 4.98 3.14 4.11 3.68

DeepSeek-v2.5 6.47 1.55 4.84 2.44 2.63 - 2.37 2.23 3.02 3.25

DeepSeekCoder-v2-Instruct 6.03 2.33 5.78 2.86 3.15 - 3.14 2.76 2.92 3.95 Llama3.3-70B-Instruct 5.66 2.55 6.82 5.03 3.81 - 4.02 3.44 4.18 5.60 Qwen2.5-72B-Instruct 4.06 1.38 3.73 2.00 2.06 - 2.86 1.55 2.83 2.66

70B+

Doubao-Coder-Preview 7.80 3.43 11.79 13.24 7.02 - 8.52 7.42 6.33 7.44 DeepSeek-v3 7.19 3.70 5.53 3.70 4.97 - 6.18 4.31 4.44 4.69 GPT 4o-mini 5.38 1.85 6.15 4.18 3.25 - 3.73 2.91 3.05 3.76

Close-Sourced

GPT 4o 5.07 2.04 5.27 2.66 3.10 - 3.65 2.62 2.95 3.36 Claude3.5-Sonnet 4.13 1.52 3.65 2.51 2.93 - 3.41 1.87 2.87 3.35 GLM-4-Plus 4.06 1.59 4.95 2.10 2.45 - 2.79 1.66 2.79 2.66 Qwen2.5-Max 3.80 1.43 3.75 2.36 2.18 - 1.81 1.50 2.15 2.10

DeepSeek-R1 9.51 6.04 11.44 6.89 7.62 - 11.51 5.88 6.98 7.78 QwQ-32B-Preview 8.61 8.59 11.06 7.98 9.82 - 13.92 10.62 7.42 8.26 OpenAI o1-mini 7.10 4.83 7.55 4.92 6.19 - 6.80 4.86 4.72 5.31 Gemini2.0-Flash-Thinking 7.04 2.72 7.21 4.10 4.65 - 4.31 3.27 4.55 5.29 OpenAI o1-Preview 6.02 2.99 6.69 3.90 4.78 - 4.51 3.01 3.92 4.03 DeepSeek-R1-Distill-Qwen-32B 6.01 2.50 7.17 4.27 4.10 - 3.87 2.90 3.53 4.41

o1-like Model

- Table 17. Results of different models on advanced critique evaluations MSE in the Code Gen’s Debug subset Dataset across all fine-grained evaluation dimensions.

Algorithm Optimization 0.5B+ Qwen2.5-Coder-0.5B-Instruct 35.30 25.15 26.83 27.25 53.51 34.16 63.47 34.21 48.51 47.20

Correctness Verification

Code Readability

Robustness Validation

Comprehensive Testing

Space Complexity

Code Style Consistency

Output Format

Maintain-ability

Time Complexity

Size Model

Yi-Coder-1.5B-Chat 40.35 26.66 28.28 28.52 57.99 35.75 68.02 35.32 53.35 50.52 DeepSeek-Coder-1.3B-Instruct 37.69 24.41 25.70 27.59 53.81 34.12 63.02 33.84 48.61 46.54 OpenCoder-1.5B-Instruct 32.33 26.16 28.32 28.15 56.14 35.30 66.79 34.73 36.22 49.57 Qwen2.5-Coder-3B-Instruct 18.10 13.77 14.51 15.02 23.75 17.55 28.24 17.32 22.61 21.13 Qwen2.5-Coder-1.5B-Instruct 17.96 12.59 13.10 14.16 22.72 16.14 27.05 16.20 21.68 19.17

1B+

OpenCoder-8B-Instruct 26.91 22.03 22.67 23.80 40.94 28.41 49.26 28.76 37.80 36.40 CodeQwen1.5-7B-Chat 26.41 19.92 20.34 21.12 36.59 25.29 42.87 25.55 35.12 32.81

Qwen2.5-Chat-7B-Instruct 18.93 7.75 11.34 10.88 14.68 11.76 20.94 12.27 16.19 17.81 CodeLlama-7B-Instruct 18.76 15.25 16.39 17.79 27.90 21.38 36.20 22.34 25.35 25.40 Qwen2.5-Coder-7B-Instruct 16.41 4.19 8.26 6.70 9.34 6.24 16.69 6.15 11.19 13.10 Yi-Coder-9B-Chat 12.07 12.20 11.81 11.15 9.61 14.81 13.10 14.35 11.35 11.24

6B+

StarCoder2-15B-Instruct 29.81 22.39 23.57 24.23 45.01 31.09 54.27 30.45 41.31 40.27 CodeLlama-13B-Instruct 14.93 12.65 13.29 12.66 15.44 15.24 19.39 14.76 15.12 15.11

Qwen2.5-Coder-14B-Instruct 9.65 5.51 8.38 6.83 9.80 8.74 15.76 7.55 12.52 13.57

13B+

DeepSeekCoder-v2-Lite-Instruct 9.41 5.71 7.55 7.66 8.23 6.19 10.69 6.08 8.98 8.61 DeepSeek-v2-Lite-Chat 9.30 6.38 8.07 6.48 9.56 7.05 12.52 6.44 10.49 9.53 Qwen2.5-Chat-14B-Instruct 7.37 2.74 5.53 4.39 4.48 4.02 5.95 3.66 5.79 6.65

CodeLlama-34B-Instruct 9.86 8.83 9.72 9.46 10.71 11.28 13.68 10.59 10.64 10.57

20B+

Qwen2.5-Coder-32B-Instruct 9.25 3.95 6.61 6.04 5.34 5.95 10.86 6.13 8.39 11.16 Qwen2.5-Chat-32B-Instruct 5.17 3.01 4.56 3.93 4.21 4.88 6.14 4.12 5.53 6.20

DeepSeek-v2.5 6.76 2.73 4.30 3.64 3.43 3.74 4.97 4.28 5.07 4.83

DeepSeekCoder-v2-Instruct 5.18 3.34 4.43 3.48 4.09 4.13 5.34 4.29 5.83 4.59 Llama3.3-70B-Instruct 4.35 3.02 4.96 3.79 2.75 3.51 4.47 3.99 3.69 4.14 Qwen2.5-72B-Instruct 3.35 1.70 3.41 2.26 2.22 2.33 3.10 1.84 2.62 3.02

70B+

Doubao-Coder-Preview 12.29 7.50 10.34 11.49 9.15 9.96 14.43 11.25 11.37 13.06

DeepSeek-v3 5.82 2.84 3.91 3.79 4.88 3.96 6.12 3.82 5.38 5.93 GPT 4o-mini 4.67 1.25 3.55 3.29 3.44 2.38 3.57 2.09 4.02 5.33

Close-Sourced

Qwen2.5-Max 4.42 1.81 3.70 3.52 2.50 2.08 3.50 2.08 3.22 3.91 GPT 4o 3.96 2.10 3.78 2.57 3.25 2.57 4.01 2.10 3.39 3.89 GLM-4-Plus 3.73 2.36 3.96 2.29 3.50 2.73 4.65 2.45 4.05 4.50 Claude3.5-Sonnet 3.36 2.39 3.53 2.69 2.09 4.10 4.23 4.13 2.68 4.14

QwQ-32B-Preview 11.32 7.04 10.14 9.52 12.21 9.58 16.82 10.20 12.81 13.66

DeepSeek-R1 10.04 7.08 11.36 8.98 7.24 15.11 9.69 5.65 8.95 8.85 OpenAI o1-mini 6.56 3.42 5.46 4.12 5.23 4.14 7.28 3.87 5.45 6.17 DeepSeek-R1-Distill-Qwen-32B 6.40 2.75 6.69 4.51 2.93 3.50 4.78 3.15 4.07 5.83 OpenAI o1-Preview 6.28 3.68 6.59 3.90 4.09 3.47 6.31 2.75 4.78 5.24 Gemini2.0-Flash-Thinking 5.75 2.70 6.21 3.45 3.35 3.60 5.21 3.57 3.78 4.84

o1-like Model

- Table 18. Results of different models on advanced critique evaluations MSE in the Code QA Subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 40.75 53.88 33.81 39.04 52.26 39.14 40.37 41.45 39.28 40.21

Size Model Depth

Yi-Coder-1.5B-Chat 43.30 57.61 35.68 41.71 55.84 41.77 43.19 44.28 41.76 42.69 OpenCoder-1.5B-Instruct 42.51 56.39 35.24 40.89 54.77 41.32 40.97 43.45 39.77 41.90 DeepSeek-Coder-1.3B-Instruct 29.37 36.60 24.85 26.27 34.98 27.78 28.25 27.93 27.48 27.74 Qwen2.5-Coder-1.5B-Instruct 21.08 25.80 19.24 18.97 23.34 20.82 19.55 19.46 19.90 19.70 Qwen2.5-Coder-3B-Instruct 17.38 20.88 15.68 15.49 19.03 17.88 16.76 16.35 14.87 15.69

1B+

OpenCoder-8B-Instruct 39.72 51.43 33.14 37.68 49.73 37.79 38.42 39.84 34.09 39.17 CodeQwen1.5-7B-Chat 29.31 36.57 24.71 27.02 34.76 28.67 28.75 28.54 27.39 28.62

Yi-Coder-9B-Chat 21.15 24.56 17.54 18.89 23.69 21.59 20.21 20.23 20.96 19.54 CodeLlama-7B-Instruct 18.03 21.22 16.38 15.92 18.31 18.63 16.25 15.76 14.78 15.92 Qwen2.5-Chat-7B-Instruct 14.91 21.12 15.01 14.19 17.61 18.06 11.82 13.63 10.02 12.60

6B+

Qwen2.5-Coder-7B-Instruct 8.92 13.94 10.18 9.31 9.94 12.35 8.46 7.95 7.90 8.06

StarCoder2-15B-Instruct 34.81 45.40 30.17 33.39 43.36 34.86 34.05 34.99 34.01 34.23 CodeLlama-13B-Instruct 12.12 12.87 11.01 11.39 12.00 13.14 12.07 11.72 12.33 11.68

Qwen2.5-Coder-14B-Instruct 7.99 12.61 9.34 8.57 8.49 11.48 7.38 7.13 6.14 6.97 Qwen2.5-Chat-14B-Instruct 6.50 12.32 8.71 7.48 5.96 10.62 5.68 5.62 4.69 4.81 DeepSeekCoder-v2-Lite-Instruct 5.43 5.25 4.05 5.16 4.92 6.69 6.94 5.28 7.09 4.89 DeepSeek-v2-Lite-Chat 5.35 4.51 4.04 4.97 4.11 6.38 6.77 5.09 6.16 5.23

13B+

CodeLlama-34B-Instruct 10.32 12.17 9.64 9.85 10.10 11.83 10.37 9.39 10.40 9.61 Qwen2.5-Chat-32B-Instruct 6.65 11.94 8.30 6.96 5.90 10.15 5.28 4.86 4.85 4.71 Qwen2.5-Coder-32B-Instruct 6.55 11.69 8.97 7.42 6.20 10.39 5.43 5.46 4.98 4.94

20B+

DeepSeek-v2.5 4.31 4.67 3.72 4.21 4.72 4.23 5.01 4.62 4.88 4.73

DeepSeekCoder-v2-Instruct 3.38 3.51 2.76 3.46 3.48 3.84 4.22 3.51 4.21 3.29 Llama3.3-70B-Instruct 3.19 2.61 3.59 3.28 2.82 4.30 3.53 3.16 3.51 2.96 Qwen2.5-72B-Instruct 2.93 2.48 2.84 3.06 2.67 3.38 3.57 3.10 3.92 3.13

70B+

Doubao-Coder-Preview 6.72 4.87 8.49 6.73 4.09 8.13 4.86 7.20 5.39 6.16

GPT 4o 3.13 3.51 2.45 3.29 3.49 3.30 3.63 3.38 3.63 2.95 DeepSeek-v3 2.86 3.55 2.29 2.85 3.25 2.94 3.48 2.74 3.25 2.63 GLM-4-Plus 2.84 2.42 2.95 3.37 2.73 3.38 3.31 3.31 3.25 3.18 GPT 4o-mini 2.12 2.31 1.75 2.34 2.20 2.95 1.97 1.96 2.03 1.84

Close-Sourced

Qwen2.5-Max 1.95 1.88 1.53 1.73 1.86 2.07 2.38 1.61 2.16 1.72 Claude3.5-Sonnet 1.67 1.74 1.11 1.55 1.97 2.05 1.85 1.81 1.65 1.53

QwQ-32B-Preview 9.73 11.34 8.99 9.21 10.64 10.25 9.70 9.46 8.88 9.27 DeepSeek-R1 7.30 9.12 4.86 6.60 6.15 7.14 7.74 5.06 7.91 4.87 OpenAI o1-Preview 4.53 3.90 2.95 3.92 3.84 4.27 4.72 3.68 5.36 2.94 Gemini2.0-Flash-Thinking 3.38 2.35 2.28 2.80 2.14 3.30 3.79 2.05 3.62 2.00 DeepSeek-R1-Distill-Qwen-32B 3.20 2.84 2.39 2.69 2.52 2.83 3.54 2.59 3.59 2.56 OpenAI o1-mini 3.16 3.06 2.36 3.00 2.79 3.69 3.50 3.15 3.68 2.82

o1-like Model

- Table 19. Results of different models on advanced critique evaluations MSE in the Code QA’s Fundamental Programming (FP) subset Dataset across all fine-grained evaluation dimensions.

Size Model Depth

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 44.06 60.14 35.97 43.10 53.45 43.56 47.37 43.64 44.09 44.82

1B+

Yi-Coder-1.5B-Chat 46.06 62.58 36.59 44.81 55.22 45.37 49.76 45.23 46.58 45.80 OpenCoder-1.5B-Instruct 43.14 57.25 34.64 41.50 51.09 43.96 43.37 41.92 41.03 43.76 DeepSeek-Coder-1.3B-Instruct 28.22 35.67 23.48 25.73 30.89 29.53 30.92 27.49 30.15 26.68 Qwen2.5-Coder-1.5B-Instruct 12.76 12.22 12.60 9.95 11.02 12.80 12.07 9.54 12.54 12.35 Qwen2.5-Coder-3B-Instruct 9.94 9.56 9.32 7.93 9.80 11.33 9.93 8.41 8.50 10.61

6B+

OpenCoder-8B-Instruct 41.60 52.93 32.34 38.69 46.25 39.27 42.98 38.88 32.81 41.62 CodeQwen1.5-7B-Chat 29.28 33.64 23.77 23.46 29.25 27.01 29.91 25.05 25.52 25.64

Yi-Coder-9B-Chat 17.42 16.53 12.10 12.77 16.43 17.03 15.46 14.92 16.98 14.74 CodeLlama-7B-Instruct 17.22 22.56 15.52 16.56 19.12 19.04 16.46 19.53 15.22 17.92 Qwen2.5-Chat-7B-Instruct 16.99 22.46 14.32 16.68 19.11 18.81 14.87 15.51 13.63 17.14

Qwen2.5-Coder-7B-Instruct 6.81 5.88 5.13 6.43 5.15 8.18 7.03 4.34 7.60 7.79

13B+

StarCoder2-15B-Instruct 33.16 42.19 28.54 30.14 37.59 31.42 35.19 30.62 33.12 33.72

CodeLlama-13B-Instruct 10.25 9.47 9.13 9.30 9.38 10.44 8.48 8.95 10.17 9.20 DeepSeek-v2-Lite-Chat 7.10 4.83 6.69 6.67 6.78 7.67 8.88 7.24 9.45 8.88 DeepSeekCoder-v2-Lite-Instruct 5.76 4.80 4.48 5.59 5.64 7.51 8.42 5.06 8.68 5.09 Qwen2.5-Chat-14B-Instruct 5.67 6.67 5.09 3.78 4.27 5.44 5.10 4.52 3.23 3.94 Qwen2.5-Coder-14B-Instruct 5.07 5.42 3.03 4.76 4.08 5.16 4.55 4.10 5.40 5.08

20B+

CodeLlama-34B-Instruct 12.86 14.59 11.65 10.70 9.28 14.09 9.32 8.75 11.82 9.91 Qwen2.5-Chat-32B-Instruct 6.64 6.85 4.96 5.27 5.13 5.26 5.49 4.77 4.25 4.81 Qwen2.5-Coder-32B-Instruct 4.58 5.15 4.43 4.01 4.05 5.30 4.58 4.30 4.18 4.46

70B+

DeepSeek-v2.5 4.57 4.29 3.11 5.02 4.66 4.31 6.25 4.47 5.39 4.31

DeepSeekCoder-v2-Instruct 3.49 3.90 2.87 2.94 3.59 4.46 4.60 2.87 5.42 2.81 Qwen2.5-72B-Instruct 3.29 3.11 3.54 3.47 2.99 3.58 4.04 3.21 4.03 3.23 Llama3.3-70B-Instruct 3.16 2.70 3.97 3.48 3.15 4.84 3.53 3.08 4.20 3.52

Close-Sourced

Doubao-Coder-Preview 7.76 6.57 11.05 7.21 4.49 9.10 6.19 7.76 4.52 7.93

GPT 4o 5.44 6.51 4.11 5.51 6.09 5.59 6.53 4.87 5.77 4.72 GPT 4o-mini 3.78 3.62 2.80 3.77 2.93 3.73 2.92 2.20 2.87 3.25 GLM-4-Plus 3.43 2.96 3.02 3.99 3.35 4.30 4.02 3.18 4.18 3.55

Qwen2.5-Max 2.67 2.75 2.09 2.31 2.39 2.54 3.10 1.78 2.22 2.15 DeepSeek-v3 2.21 1.85 1.40 2.29 1.94 2.33 3.03 1.83 2.39 2.49 Claude3.5-Sonnet 2.11 2.36 1.46 2.50 2.55 3.00 2.74 2.04 1.90 2.21

o1-like

QwQ-32B-Preview 10.09 11.84 10.89 11.13 10.70 12.51 10.93 10.97 8.94 9.66 DeepSeek-R1 8.11 13.56 6.12 6.94 6.49 10.29 8.47 5.69 8.67 5.83 OpenAI o1-Preview 5.23 4.23 4.06 5.77 5.57 4.27 5.28 4.13 5.95 3.67 DeepSeek-R1-Distill-Qwen-32B 4.32 4.13 2.80 4.30 2.93 3.19 4.15 2.22 4.66 4.08 OpenAI o1-mini 3.16 2.57 2.37 3.96 3.40 4.24 2.75 2.88 2.84 3.55 Gemini2.0-Flash-Thinking 2.93 3.05 2.10 2.67 1.97 2.99 3.08 1.60 3.12 2.24

- Table 20. Results of different models on advanced critique evaluations MSE in the Code QA’s Advanced Programming (AP) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 45.51 55.76 41.08 37.07 51.21 39.29 41.18 38.59 38.69 37.36

Size Model Depth

OpenCoder-1.5B-Instruct 47.26 57.97 42.39 38.99 52.06 41.32 40.33 39.88 40.10 38.19 Yi-Coder-1.5B-Chat 46.46 57.16 42.39 38.11 52.85 40.50 42.84 39.88 40.11 38.53 DeepSeek-Coder-1.3B-Instruct 29.62 37.65 26.04 22.34 31.06 28.40 24.99 23.25 25.14 22.16 Qwen2.5-Coder-1.5B-Instruct 25.54 28.91 21.78 17.47 22.52 23.57 20.11 16.92 21.39 16.79 Qwen2.5-Coder-3B-Instruct 18.83 22.93 16.52 12.92 17.87 18.54 16.49 12.38 14.15 12.93

1B+

OpenCoder-8B-Instruct 45.56 54.22 40.70 36.34 48.87 38.99 41.24 37.71 33.51 37.85 CodeQwen1.5-7B-Chat 29.25 34.74 25.77 22.61 31.05 29.08 26.85 22.99 25.57 24.62

Yi-Coder-9B-Chat 25.04 29.08 22.07 20.18 25.56 24.04 22.07 20.29 21.01 22.16 CodeLlama-7B-Instruct 16.67 19.73 17.17 13.63 16.80 15.58 15.70 13.92 10.02 13.19 Qwen2.5-Chat-7B-Instruct 13.32 19.65 17.16 12.14 15.33 20.43 9.30 10.33 10.32 10.20

6B+

Qwen2.5-Coder-7B-Instruct 10.63 14.78 10.88 7.73 11.71 12.73 5.43 6.50 6.20 6.01

StarCoder2-15B-Instruct 42.52 52.45 37.76 34.50 44.28 36.54 35.67 34.80 34.92 34.48 CodeLlama-13B-Instruct 11.94 11.09 10.46 10.05 10.63 15.27 10.95 9.37 11.36 10.50

Qwen2.5-Chat-14B-Instruct 7.88 10.95 8.16 5.64 5.62 11.55 5.14 4.03 5.17 3.41 Qwen2.5-Coder-14B-Instruct 7.36 11.84 9.98 7.53 6.34 12.22 5.50 6.10 4.35 5.63 DeepSeekCoder-v2-Lite-Instruct 7.16 3.54 3.23 4.81 3.20 7.26 6.95 4.33 5.75 5.07 DeepSeek-v2-Lite-Chat 7.04 4.51 3.62 5.16 4.04 9.18 8.54 4.76 5.66 6.13

13B+

CodeLlama-34B-Instruct 11.17 12.89 13.11 9.88 9.17 14.97 10.15 9.80 10.17 10.22

20B+

Qwen2.5-Chat-32B-Instruct 9.85 14.46 9.48 7.26 10.29 11.64 7.05 6.88 5.89 5.76 Qwen2.5-Coder-32B-Instruct 7.47 11.07 9.76 5.16 5.00 10.67 4.75 4.84 3.60 4.18

DeepSeek-v2.5 6.73 7.23 6.02 6.08 7.62 5.49 6.58 7.10 6.93 7.76 Llama3.3-70B-Instruct 2.88 2.91 2.35 3.12 2.79 6.43 4.36 3.61 4.15 2.64 Qwen2.5-72B-Instruct 2.68 2.09 1.77 2.31 2.24 3.23 3.38 2.05 3.94 2.25

70B+

DeepSeekCoder-v2-Instruct 2.47 2.46 1.76 1.87 2.55 2.51 2.85 2.20 2.38 1.57

Doubao-Coder-Preview 7.57 3.86 8.80 5.69 2.61 6.79 4.95 6.59 4.50 7.01 Qwen2.5-Max 3.06 2.78 1.90 2.39 3.11 2.86 3.64 2.35 1.19 1.88

GPT 4o 2.83 2.62 2.76 2.20 2.89 2.68 3.83 2.05 2.52 1.77 DeepSeek-v3 2.73 1.95 1.73 1.25 1.70 1.03 2.57 1.77 2.08 1.12 GPT 4o-mini 2.69 2.55 1.95 1.68 1.99 2.98 2.06 1.86 1.86 1.72 GLM-4-Plus 2.26 1.76 2.72 4.09 2.66 2.40 4.07 4.28 2.43 5.89

Close-Sourced

Claude3.5-Sonnet 1.26 1.84 1.17 1.17 1.57 1.60 1.36 1.43 1.60 1.02

QwQ-32B-Preview 8.14 10.11 9.23 8.62 9.83 8.18 10.53 8.20 8.24 7.82 DeepSeek-R1 8.01 10.50 5.07 6.44 5.18 7.20 7.46 4.55 7.46 5.97 OpenAI o1-Preview 4.49 4.41 2.37 3.27 4.79 4.62 5.76 3.31 4.66 2.66 Gemini2.0-Flash-Thinking 3.83 2.20 1.33 2.48 2.37 2.51 4.88 1.75 4.13 1.55 OpenAI o1-mini 3.15 4.55 2.41 3.12 3.51 3.55 4.19 2.66 3.61 2.90 DeepSeek-R1-Distill-Qwen-32B 2.86 3.39 2.52 2.81 2.80 3.28 2.88 2.33 2.85 2.25

o1-like Model

- Table 21. Results of different models on advanced critique evaluations MSE in the Code QA’s Software Engineering (SE) subset Dataset across all fine-grained evaluation dimensions.

Size Model Depth

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 42.40 55.81 36.02 39.60 57.80 41.68 42.49 41.45 39.90 39.56

1B+

Yi-Coder-1.5B-Chat 42.78 57.08 36.60 40.32 59.63 42.54 43.69 43.42 40.96 40.63 OpenCoder-1.5B-Instruct 42.00 55.89 36.60 39.51 59.29 41.58 41.00 42.23 38.40 39.62 DeepSeek-Coder-1.3B-Instruct 32.07 39.88 25.91 27.22 40.49 30.48 31.29 28.67 28.82 27.88 Qwen2.5-Coder-1.5B-Instruct 24.80 30.91 22.52 22.00 30.58 25.67 23.78 21.41 22.62 22.27 Qwen2.5-Coder-3B-Instruct 20.79 25.54 19.43 18.70 23.39 22.44 19.52 18.45 17.07 18.10

6B+

OpenCoder-8B-Instruct 41.53 53.09 35.66 38.07 53.75 40.88 41.22 39.73 36.33 38.95 CodeQwen1.5-7B-Chat 33.92 39.91 26.86 28.96 40.25 33.55 32.24 29.83 31.49 29.97

Yi-Coder-9B-Chat 27.23 30.52 20.39 22.94 31.47 25.34 25.88 23.30 24.46 23.85 Qwen2.5-Chat-7B-Instruct 16.14 22.56 16.30 14.58 19.58 20.02 11.97 13.37 8.96 12.74 CodeLlama-7B-Instruct 13.87 16.57 13.15 13.54 17.92 16.10 14.19 13.21 14.71 17.13

Qwen2.5-Coder-7B-Instruct 10.97 18.91 14.18 12.00 10.63 17.33 9.42 8.46 9.10 9.68

13B+

StarCoder2-15B-Instruct 36.58 50.06 33.47 36.23 50.42 36.76 37.31 37.49 35.34 35.01 CodeLlama-13B-Instruct 11.51 11.28 10.84 11.36 10.49 12.17 11.07 10.39 12.09 10.14

Qwen2.5-Coder-14B-Instruct 11.19 16.52 12.97 10.43 14.82 17.29 11.44 9.57 8.71 9.74 Qwen2.5-Chat-14B-Instruct 7.93 17.26 12.55 8.62 7.38 14.98 7.32 6.52 4.21 6.42 DeepSeekCoder-v2-Lite-Instruct 5.92 6.44 5.06 5.04 5.84 7.61 7.93 5.79 8.39 6.57 DeepSeek-v2-Lite-Chat 4.54 4.44 3.98 5.45 3.35 6.21 6.67 5.81 7.06 7.03

20B+

CodeLlama-34B-Instruct 10.93 11.85 9.46 9.39 10.22 11.06 9.05 8.50 11.00 9.02 Qwen2.5-Coder-32B-Instruct 9.57 18.18 12.48 9.93 10.81 15.15 6.05 7.11 6.35 6.26 Qwen2.5-Chat-32B-Instruct 5.99 17.93 11.86 7.24 4.68 14.98 3.20 2.66 3.74 2.81

70B+

DeepSeek-v2.5 5.12 5.42 5.26 3.73 6.08 5.12 6.76 4.34 5.85 4.68

DeepSeekCoder-v2-Instruct 4.78 6.84 3.78 4.79 6.41 5.69 6.60 5.53 5.39 4.32 Qwen2.5-72B-Instruct 3.34 2.70 2.94 3.25 2.53 2.93 3.86 3.36 5.07 3.62 Llama3.3-70B-Instruct 3.24 2.17 3.43 2.62 2.54 3.60 4.26 3.19 4.18 3.68

Close-Sourced

Doubao-Coder-Preview 6.17 5.30 8.63 6.69 3.10 7.59 5.63 6.54 6.88 5.37 GPT 4o 4.70 5.81 4.23 5.84 6.33 5.33 5.11 5.28 5.88 5.52 DeepSeek-v3 4.06 4.69 2.63 2.99 4.10 4.18 4.54 2.83 4.29 2.43 Qwen2.5-Max 2.70 2.34 2.06 2.55 2.15 3.00 3.01 2.53 3.87 3.40 GLM-4-Plus 2.47 2.12 2.71 3.10 2.68 3.61 2.88 3.50 3.65 3.15 Claude3.5-Sonnet 1.88 1.46 1.31 1.64 2.16 1.95 1.78 1.71 2.03 1.58 GPT 4o-mini 1.45 1.90 1.28 2.05 1.80 2.61 1.89 1.89 2.24 1.74

o1-like Model

QwQ-32B-Preview 12.17 12.55 8.77 10.12 13.28 10.97 10.40 9.24 10.65 10.12

DeepSeek-R1 7.13 8.56 4.30 6.20 6.30 6.90 6.47 5.44 10.61 3.89 OpenAI o1-mini 3.41 3.95 3.15 2.88 3.10 5.06 3.28 3.52 4.98 3.22 Gemini2.0-Flash-Thinking 3.09 2.21 2.60 2.87 1.67 3.54 3.12 2.37 3.09 2.23 OpenAI o1-Preview 2.94 4.30 1.87 2.89 2.80 3.54 2.99 3.70 6.63 2.90 DeepSeek-R1-Distill-Qwen-32B 2.85 3.10 2.36 1.95 2.54 3.09 3.32 2.21 3.48 2.37

- Table 22. Results of different models on advanced critique evaluations MSE in the Code QA’s Data Analysis (DA) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 53.91 64.58 39.54 48.70 59.13 44.81 46.18 47.06 44.33 44.24

Size Model Depth

OpenCoder-1.5B-Instruct 56.32 67.10 41.17 50.86 62.70 46.76 45.45 48.57 44.34 45.50 Yi-Coder-1.5B-Chat 55.52 65.71 40.44 50.03 60.74 45.97 46.89 48.08 45.21 45.27 DeepSeek-Coder-1.3B-Instruct 29.48 35.79 24.35 27.24 33.61 26.33 29.00 27.71 27.03 27.78 Qwen2.5-Coder-1.5B-Instruct 22.73 27.50 19.45 20.50 23.24 23.12 20.73 20.23 21.34 20.80 Qwen2.5-Coder-3B-Instruct 17.51 20.07 16.84 16.93 16.79 18.84 15.71 15.48 16.84 16.01

1B+

OpenCoder-8B-Instruct 50.71 60.43 38.12 46.03 56.41 42.14 43.60 44.37 36.61 42.88 CodeQwen1.5-7B-Chat 39.19 45.51 29.16 36.29 41.33 34.66 36.30 35.60 33.82 34.79

Yi-Coder-9B-Chat 21.13 25.14 17.35 20.45 22.57 22.16 20.72 20.52 21.72 21.52 Qwen2.5-Chat-7B-Instruct 17.55 23.04 16.94 15.64 18.61 23.39 11.47 15.28 8.77 13.94 CodeLlama-7B-Instruct 16.75 19.93 14.32 14.58 15.18 18.44 16.47 15.59 17.45 16.42 Qwen2.5-Coder-7B-Instruct 15.04 20.33 13.22 13.86 13.93 16.77 12.79 11.19 10.45 12.34

6B+

StarCoder2-15B-Instruct 49.43 55.21 35.67 42.86 51.17 41.52 42.67 42.32 40.25 39.31 CodeLlama-13B-Instruct 12.52 13.56 11.82 10.92 11.87 13.43 13.81 12.80 12.79 12.18

Qwen2.5-Coder-14B-Instruct 12.33 17.79 12.15 12.33 9.76 14.91 8.62 9.05 6.97 9.09 Qwen2.5-Chat-14B-Instruct 8.84 18.42 12.24 11.98 6.87 15.40 5.66 7.19 5.84 4.45 DeepSeek-v2-Lite-Chat 4.13 3.61 3.65 4.70 4.05 6.82 6.62 4.64 6.23 5.45 DeepSeekCoder-v2-Lite-Instruct 3.59 2.93 1.78 2.30 2.49 4.64 4.85 2.93 4.17 2.86

13B+

CodeLlama-34B-Instruct 12.36 17.83 11.02 12.49 13.32 14.09 11.98 11.71 10.47 11.98

20B+

Qwen2.5-Chat-32B-Instruct 9.48 16.75 9.95 9.79 7.20 13.77 6.15 6.56 5.00 5.03 Qwen2.5-Coder-32B-Instruct 8.01 13.88 10.59 10.10 5.59 14.48 5.97 5.47 6.94 4.55

DeepSeekCoder-v2-Instruct 3.44 1.76 2.27 2.50 2.10 2.73 3.03 2.77 3.53 3.07 Llama3.3-70B-Instruct 2.88 2.46 3.29 2.82 2.01 4.12 3.29 2.53 4.12 2.80 Qwen2.5-72B-Instruct 2.56 2.31 3.41 3.65 3.22 4.12 4.49 3.81 5.01 4.31

70B+

DeepSeek-v2.5 2.33 1.88 3.05 3.16 2.63 2.79 3.02 3.19 3.67 3.45

Doubao-Coder-Preview 7.19 3.25 9.27 7.47 4.17 10.03 4.94 6.83 7.72 6.55 DeepSeek-v3 5.84 6.96 3.91 5.30 6.29 5.57 4.81 4.26 5.18 3.72 GLM-4-Plus 3.25 2.49 3.40 3.75 2.96 4.01 3.85 2.90 3.39 2.98

Close-Sourced

GPT 4o 2.69 3.22 2.26 3.93 3.71 4.13 4.68 4.14 5.93 4.18 Qwen2.5-Max 1.87 1.56 1.46 1.74 2.09 2.25 2.57 1.25 3.18 1.49 GPT 4o-mini 1.85 1.69 1.37 1.84 1.66 2.83 1.64 1.28 2.00 1.53 Claude3.5-Sonnet 1.59 1.14 0.79 1.33 1.68 2.22 1.25 1.65 1.36 1.18

DeepSeek-R1 14.20 15.74 5.80 10.98 9.54 10.24 12.10 6.73 9.85 7.42

QwQ-32B-Preview 11.15 11.61 9.88 10.51 11.21 11.84 10.86 10.29 9.72 10.16

OpenAI o1-Preview 6.52 3.78 3.06 5.29 3.93 5.38 7.74 3.02 7.31 3.27 Gemini2.0-Flash-Thinking 3.39 1.93 2.00 2.42 1.23 2.12 3.79 1.14 2.77 2.04 DeepSeek-R1-Distill-Qwen-32B 2.36 1.39 1.43 1.96 1.48 2.30 3.39 2.08 3.07 2.16 OpenAI o1-mini 2.21 2.17 1.82 2.44 1.56 3.47 3.34 2.15 3.61 2.15

o1-like Model

Mathematics (MA) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 39.44 50.08 31.14 38.55 52.32 45.26 38.25 42.16 43.15 46.26

Size Model Depth

Yi-Coder-1.5B-Chat 41.41 52.87 33.03 41.97 55.99 47.62 40.33 45.31 46.09 50.73 OpenCoder-1.5B-Instruct 39.39 50.19 32.16 39.58 53.12 45.45 38.79 43.04 42.16 48.33 DeepSeek-Coder-1.3B-Instruct 28.37 33.01 24.05 27.39 34.94 30.55 26.75 27.89 31.29 33.01 Qwen2.5-Coder-1.5B-Instruct 18.35 22.74 18.58 19.60 20.08 21.63 17.84 18.89 20.95 21.10 Qwen2.5-Coder-3B-Instruct 15.08 18.64 14.03 15.36 17.96 17.31 15.19 17.55 13.29 16.26

1B+

OpenCoder-8B-Instruct 34.28 42.44 28.84 35.28 46.06 39.15 32.56 37.18 35.57 42.30 CodeQwen1.5-7B-Chat 24.65 30.99 22.53 24.86 31.43 27.67 23.92 26.98 27.83 31.18 CodeLlama-7B-Instruct 18.87 24.14 17.40 18.78 22.18 21.96 18.99 18.04 18.56 19.71

6B+

Yi-Coder-9B-Chat 16.42 19.04 13.46 17.48 16.77 20.25 15.62 16.89 19.98 16.30 Qwen2.5-Chat-7B-Instruct 12.47 13.41 10.41 11.82 13.75 13.60 9.62 11.18 11.22 12.50

Qwen2.5-Coder-7B-Instruct 8.25 11.55 7.78 8.79 10.16 10.92 7.27 8.42 7.30 8.07

StarCoder2-15B-Instruct 29.92 40.04 29.15 30.35 41.77 40.90 28.42 32.49 35.15 35.80 CodeLlama-13B-Instruct 15.03 19.68 13.26 15.26 15.09 19.77 16.02 16.93 17.31 16.67

Qwen2.5-Coder-14B-Instruct 6.93 8.24 6.94 6.94 9.87 8.36 5.98 7.11 6.45 6.71 DeepSeek-v2-Lite-Chat 6.11 6.48 4.34 5.53 5.14 7.03 7.87 6.31 6.56 5.98 Qwen2.5-Chat-14B-Instruct 5.01 6.92 5.27 6.44 5.38 6.88 3.82 4.43 2.85 4.39 DeepSeekCoder-v2-Lite-Instruct 4.17 4.33 2.92 4.79 3.27 6.42 5.70 3.46 6.10 4.41

13B+

CodeLlama-34B-Instruct 8.29 9.59 7.60 8.88 6.96 10.76 9.53 7.74 10.72 9.65 Qwen2.5-Chat-32B-Instruct 5.02 7.07 5.79 5.34 5.03 7.18 3.69 3.82 4.01 3.48 Qwen2.5-Coder-32B-Instruct 4.90 8.19 6.89 6.41 6.82 8.26 4.59 6.00 4.94 5.84

20B+

DeepSeek-v2.5 4.60 5.49 3.73 5.12 4.73 5.05 6.11 5.27 5.70 6.97 Llama3.3-70B-Instruct 2.53 1.98 2.95 2.41 2.91 2.93 3.22 2.46 2.55 2.58 Qwen2.5-72B-Instruct 2.36 2.75 2.47 2.38 3.03 3.17 3.08 2.75 3.14 3.19

70B+

DeepSeekCoder-v2-Instruct 1.94 2.24 1.59 2.64 1.45 2.93 2.47 2.99 2.82 2.33

Doubao-Coder-Preview 6.23 5.04 9.71 7.64 4.99 7.16 4.33 9.56 4.52 5.31 GLM-4-Plus 2.37 1.96 2.70 2.91 2.53 2.74 2.14 2.93 2.25 2.73 GPT 4o-mini 2.36 2.90 1.92 2.68 2.55 3.67 2.49 2.81 2.33 2.72

Close-Sourced

GPT 4o 2.36 2.88 1.94 2.54 2.94 2.35 2.17 2.85 2.56 2.10 Claude3.5-Sonnet 2.03 1.56 1.17 1.68 1.62 2.34 2.06 1.74 1.69 2.14 Qwen2.5-Max 1.60 1.26 1.15 1.49 1.88 1.64 1.77 1.59 1.70 2.01 DeepSeek-v3 1.46 1.41 1.58 1.10 1.40 1.29 1.39 1.62 1.34 2.12

QwQ-32B-Preview 10.63 12.57 10.36 9.65 12.32 11.60 10.56 12.06 10.81 12.53

DeepSeek-R1 8.89 9.68 5.92 8.30 8.37 9.70 9.36 6.23 10.92 5.50 OpenAI o1-Preview 6.17 5.82 4.15 5.87 5.80 6.12 6.11 4.92 6.85 4.36 Gemini2.0-Flash-Thinking 4.57 2.64 2.96 3.28 2.62 3.07 3.21 2.34 4.56 2.90 DeepSeek-R1-Distill-Qwen-32B 4.54 2.77 2.44 3.28 2.75 3.32 3.80 3.20 4.89 3.48 OpenAI o1-mini 2.49 2.04 1.44 2.27 1.54 2.09 2.58 2.38 3.30 2.18

o1-like Model

Desktop and Web Development (DW) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 42.60 56.97 36.01 40.85 56.98 39.75 38.04 40.97 38.68 45.10

Size Model Depth

Yi-Coder-1.5B-Chat 47.11 62.39 38.85 44.77 61.72 43.45 41.81 44.59 41.62 48.50 OpenCoder-1.5B-Instruct 46.00 60.24 38.01 43.41 59.68 43.35 39.63 43.46 39.89 47.42 DeepSeek-Coder-1.3B-Instruct 31.50 37.38 27.30 27.93 37.93 26.98 25.75 27.05 25.01 31.89 Qwen2.5-Coder-1.5B-Instruct 20.89 25.86 18.81 19.54 24.28 21.08 18.83 19.33 18.62 20.70 Qwen2.5-Coder-3B-Instruct 15.06 18.75 14.58 14.04 16.75 15.90 15.61 15.51 14.50 15.58

1B+

OpenCoder-8B-Instruct 44.90 58.01 37.53 42.20 57.47 40.85 38.37 41.82 36.62 45.83 CodeQwen1.5-7B-Chat 29.37 38.00 27.00 28.50 36.38 27.36 27.23 27.30 27.61 31.05

Yi-Coder-9B-Chat 22.31 25.42 19.54 19.53 23.69 19.57 18.89 18.46 20.49 20.36 CodeLlama-7B-Instruct 17.47 22.49 20.45 16.71 19.73 19.78 16.80 15.43 16.31 17.71

6B+

Qwen2.5-Chat-7B-Instruct 12.33 16.99 11.37 10.96 14.15 11.89 8.68 11.17 7.88 9.94 Qwen2.5-Coder-7B-Instruct 5.10 9.22 7.56 5.12 5.99 9.18 5.74 4.28 5.70 5.17

StarCoder2-15B-Instruct 36.76 49.44 32.99 35.85 47.77 37.39 31.17 34.52 39.05 38.86 CodeLlama-13B-Instruct 10.27 9.56 9.42 11.54 10.62 10.25 9.43 11.06 11.90 11.78

Qwen2.5-Coder-14B-Instruct 7.90 11.41 9.44 7.98 6.37 9.16 6.67 6.27 5.06 6.26 Qwen2.5-Chat-14B-Instruct 6.80 12.38 8.43 7.80 5.07 8.58 5.35 4.95 5.13 4.21 DeepSeekCoder-v2-Lite-Instruct 4.99 4.42 4.42 4.48 4.11 6.11 5.84 4.77 8.17 4.56 DeepSeek-v2-Lite-Chat 4.97 5.20 4.12 5.47 4.32 6.38 5.99 5.31 7.22 4.48

13B+

CodeLlama-34B-Instruct 10.63 12.82 9.34 10.31 11.30 10.19 10.49 10.25 8.83 10.52

20B+

Qwen2.5-Coder-32B-Instruct 6.01 10.05 8.53 6.77 6.56 8.17 5.11 5.13 5.26 5.25 Qwen2.5-Chat-32B-Instruct 5.24 9.23 6.52 6.94 6.01 7.55 4.21 4.07 3.54 3.85

DeepSeek-v2.5 4.60 6.95 4.22 4.76 6.74 4.77 4.58 4.75 4.08 5.49

DeepSeekCoder-v2-Instruct 4.21 3.54 3.37 3.84 3.77 3.51 4.58 3.66 5.20 4.88 Llama3.3-70B-Instruct 3.94 3.09 3.91 4.10 2.94 3.81 4.05 3.37 4.03 3.31 Qwen2.5-72B-Instruct 2.39 1.89 1.96 2.50 1.67 2.59 2.90 2.80 3.85 2.65

70B+

Doubao-Coder-Preview 5.42 4.53 7.65 5.02 3.73 8.27 4.21 6.59 4.42 5.84 GLM-4-Plus 3.19 2.57 3.27 3.27 2.39 3.23 3.12 3.25 3.43 2.87 GPT 4o-mini 2.07 1.90 1.67 2.31 2.31 2.67 1.33 1.93 1.62 1.85

Close-Sourced

GPT 4o 2.02 2.13 1.57 2.49 1.91 2.34 2.50 3.51 3.47 2.65 Qwen2.5-Max 1.84 1.49 1.76 1.61 1.12 1.79 1.86 1.40 2.87 1.44 DeepSeek-v3 1.62 1.52 1.26 1.47 1.22 1.11 1.26 1.15 1.35 0.95 Claude3.5-Sonnet 1.49 1.63 0.92 1.50 1.85 1.88 1.58 1.92 1.64 1.36

QwQ-32B-Preview 8.98 9.29 8.27 7.81 9.10 7.00 7.76 7.00 7.52 8.18 DeepSeek-R1 5.51 6.81 4.63 6.19 4.07 6.14 6.85 5.14 7.01 4.29 OpenAI o1-Preview 5.18 3.35 3.55 4.08 3.80 3.71 4.51 3.85 4.11 3.62 DeepSeek-R1-Distill-Qwen-32B 2.94 1.91 2.08 2.56 1.59 1.82 2.48 2.64 2.68 1.72 Gemini2.0-Flash-Thinking 2.76 2.14 1.90 2.74 1.83 2.07 4.02 2.01 3.27 1.77 OpenAI o1-mini 1.96 0.94 1.94 1.73 1.12 2.86 1.56 2.38 2.30 1.58

o1-like Model

- Table 25. Results of different models on advanced critique evaluations MSE in the Code QA’s Machine Learning (ML) subset Dataset across all fine-grained evaluation dimensions.

Size Model Depth

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 29.82 43.62 26.61 33.38 49.07 25.95 33.75 39.83 29.71 29.50

1B+

Yi-Coder-1.5B-Chat 31.58 46.40 28.07 35.97 51.14 28.35 36.87 43.52 31.58 31.65 OpenCoder-1.5B-Instruct 30.77 45.54 27.71 35.61 50.15 27.65 35.35 42.62 30.90 31.34 DeepSeek-Coder-1.3B-Instruct 25.91 35.68 24.42 29.28 42.38 23.51 28.56 31.58 25.36 24.71 Qwen2.5-Coder-1.5B-Instruct 21.61 29.59 22.04 23.89 30.93 19.78 22.75 26.89 18.88 19.86 Qwen2.5-Coder-3B-Instruct 18.82 25.57 18.97 20.60 28.27 17.56 21.74 23.98 16.83 18.25

6B+

OpenCoder-8B-Instruct 29.08 41.59 26.87 33.34 45.99 26.01 33.70 39.13 28.16 29.15 CodeQwen1.5-7B-Chat 23.87 31.52 21.82 24.87 37.19 21.80 26.22 30.98 23.26 24.36

Yi-Coder-9B-Chat 19.83 28.28 21.89 21.98 33.52 24.67 23.50 28.09 22.05 19.35 CodeLlama-7B-Instruct 13.85 17.48 15.26 15.14 15.37 16.52 15.28 14.27 9.26 11.45 Qwen2.5-Chat-7B-Instruct 11.68 17.61 15.33 13.10 15.99 14.62 13.12 14.74 8.48 10.33

Qwen2.5-Coder-7B-Instruct 9.37 17.97 14.63 13.51 13.27 14.01 10.86 10.99 9.14 7.56

13B+

StarCoder2-15B-Instruct 26.24 38.13 24.56 30.88 41.17 24.89 32.70 36.40 27.10 28.28 CodeLlama-13B-Instruct 11.28 12.08 11.31 11.85 13.84 12.16 13.27 12.44 11.80 12.40

Qwen2.5-Coder-14B-Instruct 8.31 17.80 12.81 10.91 11.95 13.30 9.89 10.99 6.24 8.20 Qwen2.5-Chat-14B-Instruct 7.52 18.32 13.33 10.14 7.15 13.03 7.25 7.61 6.95 6.85 DeepSeekCoder-v2-Lite-Instruct 6.07 6.52 4.48 6.57 5.62 5.73 7.26 8.17 6.05 3.45 DeepSeek-v2-Lite-Chat 4.56 4.44 3.90 5.22 2.72 4.38 5.45 5.19 4.86 3.22

20B+

Qwen2.5-Coder-32B-Instruct 8.13 17.51 13.70 10.68 8.51 13.20 9.10 9.44 6.89 6.23 CodeLlama-34B-Instruct 7.42 10.18 9.39 8.66 9.25 10.09 8.43 7.40 7.20 6.68 Qwen2.5-Chat-32B-Instruct 6.49 14.92 11.06 8.98 5.49 12.24 5.92 6.40 6.56 5.25

70B+

DeepSeekCoder-v2-Instruct 3.84 5.44 3.02 4.84 4.80 4.18 5.27 4.33 5.22 3.49 Llama3.3-70B-Instruct 2.88 2.74 3.77 4.31 2.25 4.43 2.89 3.20 3.49 2.28 DeepSeek-v2.5 2.52 3.10 1.79 2.91 2.33 2.66 3.79 2.43 3.70 1.65 Qwen2.5-72B-Instruct 2.48 2.38 2.01 2.28 1.38 2.21 2.68 2.04 2.70 1.68

Close-Sourced

Doubao-Coder-Preview 7.98 6.77 7.32 7.69 4.16 8.23 5.80 7.67 6.81 5.75 DeepSeek-v3 3.32 4.84 2.92 4.69 4.53 3.43 4.63 3.58 3.88 3.54

GPT 4o 2.93 3.19 1.82 2.71 1.65 2.46 2.62 2.20 1.89 1.74 GLM-4-Plus 2.85 2.66 2.74 3.09 2.42 2.68 3.35 3.52 4.33 2.39 GPT 4o-mini 2.24 2.87 1.72 2.84 2.75 2.33 2.46 2.83 3.06 1.40

Claude3.5-Sonnet 2.04 2.19 1.18 1.99 2.21 1.95 2.21 2.12 1.40 1.43 Qwen2.5-Max 1.71 1.79 1.34 1.51 1.45 1.47 1.99 1.27 2.19 1.09

o1-like Model

QwQ-32B-Preview 5.34 6.50 6.22 5.90 5.69 5.92 6.45 5.83 6.76 4.97 DeepSeek-R1 4.37 6.78 4.42 4.46 5.04 4.19 6.15 4.40 5.86 3.80 Gemini2.0-Flash-Thinking 4.13 2.77 3.07 3.08 3.60 7.16 4.16 2.03 7.30 1.23 OpenAI o1-mini 3.24 4.46 2.13 3.23 2.60 3.89 3.38 3.63 3.95 1.99 OpenAI o1-Preview 3.01 3.41 2.12 2.40 2.72 3.69 2.51 3.08 4.76 1.41 DeepSeek-R1-Distill-Qwen-32B 2.69 3.08 2.68 2.53 2.44 2.66 3.81 2.26 3.50 2.01

- Table 26. Results of different models on advanced critique evaluations MSE in the Code QA’s Scientific Computing (SC) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 40.92 50.51 30.96 37.29 49.62 36.30 37.76 40.67 38.71 37.56

Size Model Depth

OpenCoder-1.5B-Instruct 46.36 57.60 33.92 42.06 55.02 41.98 41.68 46.33 41.49 42.02 Yi-Coder-1.5B-Chat 46.15 57.60 33.31 42.06 55.67 41.38 41.88 45.75 42.17 41.66 DeepSeek-Coder-1.3B-Instruct 39.93 45.04 29.06 32.73 40.89 35.29 35.46 34.46 36.02 32.66 Qwen2.5-Coder-3B-Instruct 30.53 35.70 21.45 23.44 31.06 27.60 25.32 24.74 20.67 21.75 Qwen2.5-Coder-1.5B-Instruct 29.26 33.70 22.28 24.02 29.82 25.49 22.84 23.99 24.63 22.91

1B+

OpenCoder-8B-Instruct 42.00 51.16 30.91 37.28 49.80 37.75 35.17 40.96 36.43 37.89 CodeQwen1.5-7B-Chat 37.02 47.17 26.96 32.58 43.38 35.30 34.24 35.13 32.26 31.55 CodeLlama-7B-Instruct 32.31 24.30 17.67 17.86 21.60 22.33 17.15 15.69 16.00 14.46

6B+

Yi-Coder-9B-Chat 30.43 34.09 20.31 23.78 31.09 26.73 25.52 25.95 28.13 24.17 Qwen2.5-Chat-7B-Instruct 19.34 25.70 17.93 16.35 22.49 22.02 15.23 17.63 10.03 14.96

Qwen2.5-Coder-7B-Instruct 13.03 20.71 13.63 12.71 13.17 15.05 10.85 10.44 8.70 9.89

StarCoder2-15B-Instruct 36.51 43.41 27.42 30.69 41.25 32.81 30.24 34.38 32.45 30.17 CodeLlama-13B-Instruct 14.79 17.06 13.72 13.05 15.03 15.33 14.63 13.99 15.78 13.00

Qwen2.5-Coder-14B-Instruct 9.81 16.88 11.29 10.39 8.98 13.31 7.66 6.92 6.21 6.62 Qwen2.5-Chat-14B-Instruct 9.04 18.09 10.78 9.51 8.98 13.85 6.64 7.30 3.68 6.06 DeepSeekCoder-v2-Lite-Instruct 4.26 4.58 4.04 5.08 3.43 6.13 7.86 5.51 10.05 4.86 DeepSeek-v2-Lite-Chat 4.21 4.26 3.97 4.11 4.44 4.82 6.44 4.72 5.10 4.49

13B+

Qwen2.5-Chat-32B-Instruct 11.02 17.85 11.56 9.94 7.77 14.38 7.31 6.22 8.35 7.33 Qwen2.5-Coder-32B-Instruct 10.54 16.68 11.18 9.47 8.01 14.28 6.89 6.19 5.34 6.32 CodeLlama-34B-Instruct 9.97 11.21 8.86 9.30 8.95 12.52 11.56 9.30 12.76 9.96

20B+

Qwen2.5-72B-Instruct 4.03 3.36 4.06 3.81 3.55 5.04 4.57 4.88 4.25 4.49 DeepSeek-v2.5 3.64 4.00 3.35 3.19 4.66 2.96 4.30 4.94 4.57 4.82 DeepSeekCoder-v2-Instruct 3.40 3.41 2.77 3.77 4.35 4.65 4.34 3.99 4.85 3.54 Llama3.3-70B-Instruct 3.13 2.58 3.82 3.73 2.22 4.87 3.55 3.12 3.09 3.07

70B+

Doubao-Coder-Preview 7.48 4.71 7.43 7.74 5.12 7.41 4.71 7.87 4.48 6.12

GPT 4o 4.32 4.01 2.73 3.43 3.83 4.28 3.88 4.33 3.66 3.65 GLM-4-Plus 3.27 2.55 3.33 2.86 2.67 3.10 2.87 2.99 2.71 2.54 DeepSeek-v3 2.10 3.31 2.36 2.40 3.44 2.75 3.18 2.88 2.48 2.51 GPT 4o-mini 2.03 2.38 1.55 2.26 1.85 2.68 1.43 1.65 1.69 1.26

Close-Sourced

Qwen2.5-Max 1.78 1.90 1.31 1.16 1.56 1.89 1.95 1.04 1.48 1.45 Claude3.5-Sonnet 1.29 1.76 1.16 1.57 1.73 1.86 2.14 1.93 1.65 1.78

QwQ-32B-Preview 12.58 15.10 9.30 11.81 12.87 12.30 13.23 12.45 9.53 12.30

DeepSeek-R1 8.61 9.65 5.07 7.39 7.11 8.05 8.97 6.25 9.63 4.85 OpenAI o1-Preview 4.48 4.83 2.64 4.24 3.39 4.94 5.67 3.81 6.27 3.36 OpenAI o1-mini 3.77 3.84 2.59 3.78 3.62 4.15 5.01 3.92 4.57 3.37 Gemini2.0-Flash-Thinking 3.10 2.27 2.33 2.54 2.10 3.81 4.70 2.11 3.31 2.05 DeepSeek-R1-Distill-Qwen-32B 3.05 2.48 2.29 2.37 2.14 2.72 4.29 2.43 4.77 2.23

o1-like

- Table 27. Results of different models on advanced critique evaluations MSE in the Code QA’s Databases (DB) subset Dataset across all fine-grained evaluation dimensions.

Size Model Depth

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 38.65 50.80 33.44 38.38 45.36 38.22 38.33 39.77 40.57 40.81

1B+

Yi-Coder-1.5B-Chat 40.85 56.44 36.37 42.39 50.84 41.84 42.52 43.71 46.36 44.75 OpenCoder-1.5B-Instruct 40.15 55.56 35.60 41.68 50.43 41.08 38.78 43.57 44.09 43.83 DeepSeek-Coder-1.3B-Instruct 27.13 34.74 25.18 24.42 31.10 26.93 27.42 27.53 29.42 28.47 Qwen2.5-Coder-1.5B-Instruct 21.59 27.05 19.98 20.00 22.60 22.17 20.72 21.00 23.43 22.81 Qwen2.5-Coder-3B-Instruct 16.91 20.18 15.56 16.61 19.44 17.75 17.58 18.19 17.79 18.34

6B+

OpenCoder-8B-Instruct 36.05 50.97 33.33 37.15 45.27 37.50 37.29 39.31 38.22 39.70 CodeQwen1.5-7B-Chat 28.70 35.98 25.07 28.15 32.11 29.90 28.70 29.25 31.34 30.42

Yi-Coder-9B-Chat 19.27 22.12 18.07 17.60 21.81 21.36 19.95 20.41 20.62 19.70 CodeLlama-7B-Instruct 15.20 21.60 15.14 15.87 14.87 18.32 15.60 14.97 16.16 15.85 Qwen2.5-Chat-7B-Instruct 12.67 25.12 15.99 16.31 16.03 19.08 10.68 12.45 10.06 12.53

Qwen2.5-Coder-7B-Instruct 7.53 11.58 8.44 7.27 7.81 10.43 7.06 7.03 7.28 6.31

13B+

StarCoder2-15B-Instruct 33.90 47.22 30.69 35.93 42.27 36.29 34.54 35.68 35.46 37.44 CodeLlama-13B-Instruct 9.28 11.06 9.82 9.86 10.72 10.61 10.71 8.97 9.52 10.21

Qwen2.5-Coder-14B-Instruct 8.46 11.87 8.88 8.61 9.40 11.99 7.07 7.54 6.96 7.02 DeepSeekCoder-v2-Lite-Instruct 7.61 7.85 6.69 5.84 7.18 7.38 7.11 6.53 5.21 5.95 DeepSeek-v2-Lite-Chat 6.59 5.02 4.27 4.53 4.93 6.56 6.49 5.00 5.14 4.74 Qwen2.5-Chat-14B-Instruct 5.12 9.19 7.08 6.09 5.18 9.75 5.03 5.09 4.18 4.37

20B+

CodeLlama-34B-Instruct 10.13 10.91 7.55 8.66 11.04 9.86 10.50 9.51 10.37 8.38 Qwen2.5-Chat-32B-Instruct 4.43 8.23 6.92 4.79 3.31 9.09 3.94 3.56 3.96 3.96 Qwen2.5-Coder-32B-Instruct 3.81 7.81 6.22 6.21 3.07 8.44 2.88 3.02 2.94 2.81

70B+

DeepSeek-v2.5 4.63 3.40 3.38 3.97 3.54 4.52 4.06 5.38 5.71 5.14 Llama3.3-70B-Instruct 3.91 3.23 3.66 3.13 3.21 4.21 2.72 3.42 2.29 2.82 DeepSeekCoder-v2-Instruct 3.11 3.00 3.19 3.30 3.80 3.66 4.38 3.54 3.37 3.63 Qwen2.5-72B-Instruct 2.96 2.02 3.13 2.52 2.19 3.09 3.09 2.39 2.90 2.66

Close-Sourced

Doubao-Coder-Preview 6.39 5.30 7.76 7.39 4.70 9.24 4.04 5.50 4.06 5.43 DeepSeek-v3 4.70 6.49 3.29 5.93 6.40 5.61 6.08 5.10 6.56 4.94 GLM-4-Plus 2.91 2.13 3.00 3.47 2.79 3.72 3.63 3.05 3.04 2.35 GPT 4o-mini 1.87 1.93 1.57 2.57 2.37 3.13 1.79 1.75 1.42 1.80

Claude3.5-Sonnet 1.68 1.68 1.15 1.41 1.82 1.97 1.33 1.74 1.54 1.07 GPT 4o 1.60 1.68 1.13 1.54 1.53 1.72 1.45 1.38 1.80 1.17 Qwen2.5-Max 1.52 1.49 1.34 1.13 1.84 1.73 1.82 1.51 1.45 0.99

o1-like Model

QwQ-32B-Preview 7.15 11.46 9.29 7.64 8.83 10.34 7.36 8.23 7.57 8.72 DeepSeek-R1 5.19 5.90 3.18 4.48 4.28 5.45 5.67 3.68 6.74 4.19 OpenAI o1-Preview 3.19 3.79 2.39 3.56 3.22 3.72 3.80 3.08 4.01 2.07 Gemini2.0-Flash-Thinking 3.07 2.36 2.25 2.09 1.88 2.61 2.85 1.65 2.35 1.62 DeepSeek-R1-Distill-Qwen-32B 3.06 2.81 2.53 2.26 3.30 2.83 2.92 2.44 3.15 2.17 OpenAI o1-mini 2.80 1.90 1.66 2.22 1.77 3.27 2.32 2.57 2.88 2.12

- Table 28. Results of different models on advanced critique evaluations MSE in the Code QA’s Multimedia (MM) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 37.28 55.85 30.54 36.18 53.12 37.42 39.91 42.45 36.76 38.01

Size Model Depth

Yi-Coder-1.5B-Chat 39.84 58.96 32.30 37.82 55.74 39.69 43.07 44.62 37.86 40.32 OpenCoder-1.5B-Instruct 39.39 58.91 31.84 37.82 56.57 39.29 42.63 44.98 35.65 40.09 DeepSeek-Coder-1.3B-Instruct 26.69 35.51 21.31 22.86 32.08 23.82 25.93 27.99 24.07 26.92 Qwen2.5-Coder-1.5B-Instruct 18.84 22.73 16.66 15.18 22.08 16.13 17.56 19.59 18.06 19.32 Qwen2.5-Coder-3B-Instruct 15.93 17.98 13.72 12.74 15.66 15.74 14.80 13.81 13.71 12.72

1B+

OpenCoder-8B-Instruct 35.80 51.98 28.56 33.64 50.21 34.92 35.63 40.08 28.35 35.70 CodeQwen1.5-7B-Chat 23.02 33.77 20.79 22.77 30.84 21.63 22.33 24.90 21.01 22.86

Yi-Coder-9B-Chat 18.47 24.07 15.97 16.22 22.44 17.69 17.82 18.01 18.77 17.04 CodeLlama-7B-Instruct 17.34 22.91 16.37 16.57 20.30 18.20 15.55 17.43 14.77 15.04 Qwen2.5-Chat-7B-Instruct 16.08 24.86 13.77 13.77 18.80 16.82 9.43 13.49 9.37 11.30

6B+

Qwen2.5-Coder-7B-Instruct 5.10 12.00 7.31 7.89 7.18 9.83 6.91 6.82 6.77 6.62

StarCoder2-15B-Instruct 30.42 43.21 25.15 29.29 41.75 32.33 32.45 34.08 29.31 30.39 CodeLlama-13B-Instruct 14.18 14.43 10.80 11.44 13.10 11.49 12.36 13.05 11.98 12.08

DeepSeekCoder-v2-Lite-Instruct 6.22 6.85 4.28 6.29 7.19 8.18 7.57 6.11 8.21 6.78 DeepSeek-v2-Lite-Chat 4.97 2.55 2.79 2.94 2.92 4.99 5.23 3.24 5.76 3.32 Qwen2.5-Coder-14B-Instruct 3.76 9.68 6.69 7.29 5.08 8.43 4.94 3.84 4.27 4.14 Qwen2.5-Chat-14B-Instruct 3.33 8.15 5.57 5.84 3.48 7.45 4.47 3.89 3.67 3.13

13B+

CodeLlama-34B-Instruct 11.22 11.94 9.66 10.03 10.81 10.09 11.56 9.60 11.07 9.15 Qwen2.5-Coder-32B-Instruct 5.36 10.97 7.05 7.17 6.16 8.03 4.94 4.56 3.95 4.36 Qwen2.5-Chat-32B-Instruct 3.67 7.71 4.93 4.80 3.40 6.11 3.37 2.63 3.87 3.35

20B+

DeepSeekCoder-v2-Instruct 3.30 3.14 2.69 2.92 2.58 3.08 3.27 2.96 2.89 3.17 Llama3.3-70B-Instruct 3.23 2.79 4.78 3.78 3.62 4.10 3.09 3.31 3.00 3.39 DeepSeek-v2.5 3.10 3.79 2.72 2.75 3.14 2.92 3.30 2.84 2.79 2.61 Qwen2.5-72B-Instruct 2.55 2.31 2.93 3.12 3.19 3.32 2.64 3.03 3.82 3.44

70B+

Doubao-Coder-Preview 6.26 3.38 8.05 5.64 3.15 7.95 3.94 8.04 4.81 6.80

GPT 4o 3.41 3.94 2.86 2.91 4.08 3.37 3.31 3.80 3.24 2.65 GLM-4-Plus 1.91 2.19 2.38 2.37 2.19 2.74 2.29 2.43 2.14 2.15 DeepSeek-v3 1.89 3.94 2.22 2.47 3.32 2.82 3.99 3.28 3.37 2.69

Close-Sourced

Qwen2.5-Max 1.40 1.88 1.17 1.48 1.35 1.59 1.53 1.29 1.41 1.62 GPT 4o-mini 1.28 1.60 1.46 1.81 1.54 3.11 1.82 1.79 1.58 1.81 Claude3.5-Sonnet 1.25 1.61 0.74 1.09 2.07 1.63 1.55 2.04 1.19 1.58

QwQ-32B-Preview 12.22 13.09 9.34 10.01 12.18 12.14 9.13 11.29 8.33 9.62 DeepSeek-R1 6.40 7.35 4.28 5.48 6.21 6.57 6.36 5.09 5.19 3.57 OpenAI o1-Preview 4.24 2.05 3.40 2.44 2.85 3.21 2.92 4.23 3.63 2.39 OpenAI o1-mini 3.08 2.86 2.14 2.84 2.65 3.10 3.57 3.74 2.55 3.28 Gemini2.0-Flash-Thinking 2.57 1.48 1.59 2.75 1.79 2.75 3.33 3.34 2.47 2.01 DeepSeek-R1-Distill-Qwen-32B 2.47 2.44 2.30 2.84 2.41 2.29 3.33 3.70 2.81 2.41

o1-like Model

- Table 29. Results of different models on advanced critique evaluations MSE in the Code QA’s Operating Systems (OS) subset Dataset across all fine-grained evaluation dimensions.

Logical Coherence

Innovation Practicality Clarity Reliability Completeness Maintainability Correctness Performance 0.5B+ Qwen2.5-Coder-0.5B-Instruct 37.23 52.04 32.67 36.58 48.52 38.73 39.77 39.46 38.16 39.88

Size Model Depth

Yi-Coder-1.5B-Chat 41.51 59.50 36.44 41.02 55.47 43.11 44.83 43.20 42.22 43.62 OpenCoder-1.5B-Instruct 40.25 57.55 35.67 39.37 53.62 42.63 43.07 41.66 40.35 42.08 DeepSeek-Coder-1.3B-Instruct 23.79 33.21 21.92 22.70 30.87 24.01 25.23 24.17 21.66 24.68 Qwen2.5-Coder-1.5B-Instruct 15.99 22.99 16.69 17.20 21.40 18.32 18.09 17.31 16.62 18.07 Qwen2.5-Coder-3B-Instruct 11.24 14.93 11.58 12.06 13.43 14.15 13.40 12.40 9.88 11.92

1B+

OpenCoder-8B-Instruct 38.89 52.56 33.80 36.67 48.56 38.94 39.70 39.08 33.09 39.31 CodeQwen1.5-7B-Chat 26.00 33.41 22.72 25.06 30.73 27.94 27.82 26.39 22.00 27.94 CodeLlama-7B-Instruct 17.26 21.52 17.58 16.23 17.53 18.50 16.49 15.44 14.46 17.34

6B+

Qwen2.5-Chat-7B-Instruct 15.88 21.87 15.90 14.87 19.28 18.26 16.49 14.86 12.37 13.29 Yi-Coder-9B-Chat 15.62 17.23 11.80 15.22 17.75 19.07 17.22 16.28 16.31 13.83

Qwen2.5-Coder-7B-Instruct 7.26 11.33 9.17 7.73 10.77 11.63 9.75 9.46 8.65 8.36

StarCoder2-15B-Instruct 31.39 40.87 28.16 30.52 39.59 33.08 33.06 32.25 31.41 32.10 CodeLlama-13B-Instruct 11.71 11.04 9.69 10.53 10.31 13.87 11.85 11.09 10.75 10.06

Qwen2.5-Coder-14B-Instruct 7.39 11.90 8.66 7.55 7.70 12.53 8.93 7.39 7.30 7.61 DeepSeek-v2-Lite-Chat 5.24 2.87 3.56 4.23 2.28 6.60 6.15 2.74 4.81 2.90 Qwen2.5-Chat-14B-Instruct 4.89 10.21 7.40 7.02 6.34 10.24 6.93 6.44 6.34 5.86 DeepSeekCoder-v2-Lite-Instruct 4.05 4.77 2.88 5.88 5.95 6.65 7.32 5.60 7.31 4.68

13B+

CodeLlama-34B-Instruct 9.52 11.46 9.12 10.10 11.14 12.80 11.42 10.53 10.26 9.76 Qwen2.5-Chat-32B-Instruct 5.88 11.80 8.22 6.62 6.87 9.89 7.91 6.02 4.28 5.79 Qwen2.5-Coder-32B-Instruct 4.05 9.67 7.79 6.17 4.42 8.57 4.89 4.44 4.06 4.14

20B+

DeepSeek-v2.5 5.62 6.06 4.70 5.26 5.94 6.21 6.49 5.72 5.98 5.01 Qwen2.5-72B-Instruct 3.48 2.38 2.94 4.35 3.11 3.88 4.55 3.63 4.21 2.80 DeepSeekCoder-v2-Instruct 3.33 2.65 3.01 4.50 3.30 4.72 5.33 3.74 5.41 3.84 Llama3.3-70B-Instruct 3.28 2.07 3.49 2.74 3.24 4.12 3.76 3.44 3.27 2.57

70B+

Doubao-Coder-Preview 5.52 4.33 7.97 6.03 4.38 7.45 4.72 6.47 6.18 5.32 GLM-4-Plus 3.32 3.27 3.19 4.05 3.28 4.55 4.12 4.28 4.22 3.72 GPT 4o 2.25 3.01 1.75 2.95 3.49 2.09 3.79 2.66 2.93 2.16 DeepSeek-v3 2.16 1.99 1.88 1.86 1.74 1.94 2.94 1.95 2.80 2.42 Claude3.5-Sonnet 1.67 1.80 1.16 1.19 2.36 2.13 2.49 1.67 2.23 1.72 GPT 4o-mini 1.66 1.89 1.92 1.98 2.41 2.68 1.88 1.73 1.77 1.38 Qwen2.5-Max 1.59 1.70 1.36 1.47 1.61 2.11 2.85 1.62 2.03 1.32

Close-Sourced

DeepSeek-R1 5.19 8.54 5.10 6.00 4.96 4.44 6.80 2.51 5.46 3.41 QwQ-32B-Preview 8.83 10.97 7.63 8.23 10.58 10.00 9.64 8.27 10.06 8.18 OpenAI o1-mini 5.29 4.51 4.40 4.46 5.47 4.92 6.85 4.76 6.08 4.88 OpenAI o1-Preview 4.61 2.75 2.83 3.43 3.20 3.86 4.15 3.37 5.06 2.81 DeepSeek-R1-Distill-Qwen-32B 3.77 3.57 2.74 2.76 3.11 3.61 4.76 3.02 4.09 3.61 Gemini2.0-Flash-Thinking 3.67 2.66 2.79 3.80 2.40 3.48 4.58 2.24 3.61 2.52

o1-like Model

###### Basic Critique Evaluation Judge Prompt

You are a senior programming expert. We request a professional and precise evaluation of the AI assistant’s performance based on the user’s question.

###### Evaluation Process:

- 1. Carefully review the user’s question and the assistant’s answer.
- 2. Verify whether the answer (may include code) fully meet the user’s requirements.
- 3. Pay close attention to specific aspects of the answer, including:

. - Appropriateness of the answer’s perspective.

. - Accuracy of technical details. . - Completeness of the solution.

###### Evaluation Requirements:

- - Provide detailed, point-by-point feedback on the answer.
- - Each critique should be specific and self-contained.
- - Clearly identify any issues, avoiding vague or ambiguous descriptions.
- - Offer constructive suggestions for improvement. Evaluation Criteria:
- - "Error": The code contains incorrect conclusions, explanations, or fails to meet the specified requirements.
- - "Correct": The code is entirely accurate and meets all outlined requirements.

Output Format: Provide the evaluation in JSON format as follows: ‘‘‘json {

. "reasons": "Detailed explanation evaluating the response, addressing specific points",

. "is_assistant_correct": "Based on the evaluation, indicate whether the assistant’s response is [’Correct’, ’Error’]" }

‘‘‘

Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

Figure 18. Basic critique evaluation judge prompt.

###### Advanced Critique Evaluation Judge Prompt

You are a professional code evaluation expert with the following qualifications:

- - Expertise in programming languages and algorithms.
- - Skilled in accurately assessing answer (may include code) quality.
- - Proficient in defining and applying precise, professional scoring criteria. Evaluation Process:

- 1. Analyze and understand the problem statement and provided answer thoroughly.
- 2. Evaluate systematically against defined criteria.
- 3. Assign professional scores for each evaluation problem.
- 4. Aggregate individual scores to determine a final, comprehensive score. Output Specifications:

- - Scoring Range:

. - Each evaluation metric is scored on an integer scale from 1-10.

. - The comprehensive score is also on a 1-10 scale.

- - Justification: All scores must be supported by clear, specific reasoning.
- - Structure: The final output must be well-organized, concise and professional.
- - Objectivity: Scoring must be neutral and unbiased. Scoring Guidelines:
- - 1-2 points: Critical flaws; fails to meet requirements.
- - 3-4 points: Significant deficiencies; largely unusable.
- - 5-6 points: Functional but requires substantial improvement.
- - 7-8 points: Well-implemented with minor issues.
- - 9-10 points: High-quality, near-perfect implementation.

Output Format: ‘‘‘md

- 1. <Reason>, Score: xx
- 2. <Reason>, Score: xx

... Comprehensive evaluation: <Reason>, Comprehensive Score: xx

‘‘‘

Important Notes:

- The final comprehensive score should reflect a weighted assessment of all sub-scores.

###### Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

— Start of Fine-Grained Evaluation Checklists $Checklists

— End of Fine-Grained Evaluation Checklists —

Figure 19. Advanced critique evaluation judge prompt.

###### Full Error Typelists

- 1. Syntax Error: [“Spelling mistakes (e.g., incorrect variable names)”, “Missing semicolons or mismatched brackets”, “Incorrect use of keywords”, “Indentation errors (for indentation-sensitive languages)”, “Incorrect comment formatting”]
- 2. Reference Error: [“Undefined variables”, “Undefined functions”, “Null pointer references”, “Array out of bounds”, “Dangling pointers”]
- 3. Type Error: [“Type conversion errors”, “Undefined type errors”, “Type checking errors”]
- 4. Logic Error: [“Incorrect conditional statements”, “Incorrect loop conditions”, “Algorithm errors”, “Variable scope errors”, “Variable name conflicts”, “Boundary condition errors”]
- 5. Performance Issue: [“Memory leaks”, “Performance bottlenecks”, “Memory fragmentation”, “High time complexity”, “Resource contention”]
- 6. Design Flaw: [“Resource leaks”, “Lack of memory management”, “Absence of modular design”]
- 7. Security Vulnerability: [“Unsafe string operations”, “Failure to prevent SQL injection”, “Use of insecure random number generators”, “Failure to handle external service unavailability”, “Privilege escalation vulnerabilities”]
- 8. Configuration Management Error: [“Not using version control systems”, “Failure to back up code”, “Absence of continuous integration and continuous delivery for code”, “Configuration file errors”, “Improper branch management”, “Lack of tag management”, “No tracking of historical versions”]
- 9. Data Management Error: [“Database errors”, “Data format errors”, “Data consistency errors”, “Data integrity errors”]
- 10. Concurrency and Multithreading Error: [“Race conditions”, “Deadlocks”, “Thread safety issues”]
- 11. Input Validation and Data Processing Error: [“Insufficient input validation”, “Failure to handle boundary conditions”, “Failure to handle user input formats”, “Failure to filter user input”, “Failure to validate user input”]
- 12. Exception Handling Error: [“Improper exception handling”, “Failure to handle exceptions”, “Failure to use appropriate exception types”, “Failure to use appropriate error codes or exception messages”]
- 13. Internationalization and Localization Error: [“Failure to consider internationalization and localization”, “Failure to handle timezone issues”, “Failure to handle string operations in different locales”, “Currency format errors”, “Date format errors”]
- 14. Monitoring and Logging Management Error: [“Insufficient logging”, “Failure to use logging frameworks”, “Lack of performance monitoring for code”, “Failure to use appropriate log levels”, “Failure to use appropriate log rotation mechanisms”]
- 15. Code Quality and Maintenance Error: [“Comment errors”, “Failure to follow coding standards”, “Failure to conduct code reviews”, “Failure to refactor code”, “Failure to perform static code analysis”]
- 16. User Permission and Authentication Error: [“Failure to handle user permission control”, “Failure to handle user permission management”, “Failure to handle user authentication and authorization”, “Insufficient identity verification”, “Permission validation errors”]
- 17. Testing and Verification Error: [“Failure to use unit tests”, “Insufficient modular testing of code”, “Lack of automated testing for code”, “Insufficient integration testing”, “Insufficient regression testing”]
- 18. Network and Communication Error: [“Failure to handle network request exceptions”, “Failure to handle user request timeouts”, “Network protocol errors”, “Data transmission errors”, “Connection timeouts”]
- 19. File and I/O Error: [“File permission errors”, “File format errors”, “File path errors”, “File locking issues”, “Failure to handle file operation exceptions”]
- 20. Dependency Management Error: [“Dependency management issues”]
- 21. Session Management Error: [“Failure to handle user session management”]
- 22. Log Security Issue: [“Log security issues”]
- 23. Environment Variable Error: [“Environment variable errors”]

Figure 20. Full Error typelists.

Insert Bug Prompt You are an experienced developer tasked with purposefully injecting errors into codes based on specified error types. Task Objectives:

- - Analyze the original code’s structure and functionality.
- - Modify the code according to the provided error type. - Ensure that the injected error adheres to the specified category.
- - Preserve the overall functional integrity of the code.
- - Provide clear instructions for the error injection process. Error Injection Requirements:

- 1. Code Modification Principles:

- - Clarity and precision: Modifications must be straightforward and targeted.
- - Category compliance: The injected error must belong to the specified error category.
- - Avoid unsanctioned errors: Only introduce errors of the specified type.
- - Preserve code integrity: Ensure that the other parts of the code remain unaffected.

- 2. Error Type Selection:

- - Select from the specified error categories: Choose the error type from the predefined set of categories.
- - Subcategory compliance: Ensure that the subcategory chosen aligns with the main error category.
- - Full coverage: All specified error types must be incorporated.
- - Reasonable error placement: The error injection should be placed logically within the code, reflecting real-world scenarios.

- 3. Modification Description Requirements:

- - Specify modification locations: Clearly define where each change takes place.
- - Detail the modification process: Provide a thorough explanation of how the modification is carried out. - Explain the error type: Clarify which error type corresponds to each modification.
- - Repair difficulty evaluation: Assess the ease or difficulty of repairing the error.

Output Format: ‘‘‘json { “Changed_code”: <Complete modified code>, “Selected error type category”: [<Error category 1>, ...], “Selected error subcategory”: [<Error subcategory for category 1>, ...], “Number of error types included”: <Integer>, “Specific modified location and processing”: [{

. “Location”: <Description of the code location>,

. “Corresponding error type”: <Error type for this modification>,

. “Modification description”: <Detailed explanation of the modification>

. }, ...], “Difficulty of repair”: <“Easy” | “Medium” | “Difficult”> }

‘‘‘

Error Modification Evaluation Criteria:

- 1. Difficulty Level Definitions:

- Easy: The error is clear and the repair method is straightforward. - Medium: The error requires some analysis to identify and repair. - Difficult: The error demands a deep understanding of the code structure and the repair is complex.

- 2. Modification Quality Requirements:

- - Natural error injection: Ensure the error injection appears seamless and not forced.
- - Readability: Maintain the code’s readability, ensuring the injected errors don’t obscure the logic.
- - Error type accuracy: Ensure each error type is accurately identified and implemented. Notes:
- - Provide a complete solution with all fields populated.
- - Ensure the instructions are clear and specific, with an emphasis on easy understanding.
- - The repair difficulty should be assessed objectively and realistically.
- - Maintain accuracy and completeness regarding the error types.

###### Input Data:

— Start of Code $Code

— End of Code —

— Start of Error Typelists $Error_Typelist

— End of Error Typelists —

Figure 21. Insert bug prompt.

###### Identifying Programming Error Types Prompt

You are an experienced programming expert responsible for professionally categorizing code error types. Your task is to analyze the code and determine which predefined error category it belongs to.

###### Available Error Categories:

1. Configuration Management Error\n 2. Data Management Error\n 3. Input Validation and Data Processing Error\n 4. Monitoring and Logging Management Error\n 5. Environment Variable Error\n 6. Dependency Management Error\n 7. Syntax Error\n 8. Design Flaw\n 9. Security Vulnerability\n 10. Log Security Issue\n 11. Reference Error\n 12. Session Management Error\n 13. Code Quality and Maintenance Error\n 14. Logic Error\n 15. Testing and Verification Error\n 16. Network and Communication Error\n 17. Exception Handling Error\n 18. User Permission and Authentication Error\n 19. File and I/O Error\n 20. Type Error\n 21. Internationalization and Localization Error\n 22. Performance Issue\n 23. Concurrency and Multithreading Error

###### Category Definitions:

- - Configuration Management Error: Issues related to system configuration settings
- - Data Management Error: Problems with data handling, storage, or retrieval
- - Input Validation Error: Failures in validating or processing input data
- - Monitoring/Logging Error: Issues with system monitoring or logging functions
- - Environment Variable Error: Problems with environment variables
- - Dependency Management Error: Issues with external dependencies or libraries
- - Syntax Error: Basic programming language syntax violations
- - Design Flaw: Fundamental problems in code architecture or design
- - Security Vulnerability: Security-related weaknesses
- - Log Security Issue: Security problems in logging mechanisms
- - Reference Error: Invalid references to variables or objects
- - Session Management Error: Problems with user session handling
- - Code Quality Error: Issues affecting maintainability or readability
- - Logic Error: Flaws in business logic or algorithms
- - Testing/Verification Error: Problems with testing processes
- - Network/Communication Error: Issues in network interactions
- - Exception Handling Error: Problems with error handling
- - User Permission Error: Authentication or authorization issues
- - File/I/O Error: Problems with file operations
- - Type Error: Issues with data types or type conversions
- - Internationalization Error: Problems with language/locale support
- - Performance Issue: Efficiency or resource usage problems
- - Concurrency Error: Issues with parallel processing Categorization Requirements:

- 1. Primary Analysis

. - Carefully examine the code structure and syntax

. - Identify all existing errors

. - Match errors against the predefined categories

- 2. Classification Criteria:

. - Exact Match: The error perfectly fits one of the predefined categories

. - Best Fit: If multiple issues exist, select the most significant one . - No Match: If the error doesn’t match any predefined category

- 3. Analysis Steps:

. - Locate the specific error in the code

. - Compare it with the predefined categories

. - Determine the most appropriate classification

. - Provide reasoning for the classification Output Format: ‘‘‘json {

. "Category": "<Selected error category from the predefined list>",

. "Confidence": "<High/Medium/Low>" }

‘‘‘

Classification Notes:

- - Only select one primary category even if multiple errors exist
- - If no predefined category matches, select the most appropriate error category
- - Include specific code references in the explanation
- - Indicate confidence level in the classification
- - Provide clear reasoning for the chosen category Additional Guidelines:
- - Focus on syntactic and structural analysis
- - Consider the context of the entire code
- - Be specific about error location
- - Explain any ambiguous cases
- - Maintain objectivity in classification

40

###### Input Data:

— Start of Code $Code

— End of Code —

Figure 22. Identifying programming error types prompt.

###### Prompt for Classify Application Scenarios in “Code QA” Subset

As a professional expert in technical field classification, you are tasked with accurately categorizing programming problems and code snippets into the appropriate technical domains.

###### Required Expertise:

- 1. A deep understanding of programming concepts and the distinctive characteristics of various technical fields.
- 2. The ability to accurately identify the core technical features of code and problems.
- 3. Proficiency in discerning the boundaries and relationships between different technical domains.
- 4. Capacity to provide well-reasoned and persuasive justification for each classification. Technical Field Labels:

- 1. Basic Programming: Foundational concepts such as syntax, control flow and data structures.
- 2. Advanced Programming: In-depth topics like design patterns, concurrency and performance optimization.
- 3. Software Engineering: Practices including project management, code quality, testing and deployment.
- 4. Data Analysis: Data processing, statistical analysis and data visualization.
- 5. Mathematics: Algorithms, computation methods and mathematical modeling.
- 6. Desktop and Web Development: User interface design, front-end and back-end development and user interactions.
- 7. Machine Learning: Model training, deep learning and artificial intelligence.
- 8. Scientific Computing: Numerical computations, simulations and scientific modeling.
- 9. Database: Data storage, query optimization and database design.
- 10. Multimedia: Image, audio and video processing, as well as multimedia applications.
- 11. Operating Systems: System programming, memory management and process scheduling.
- 12. Other: Fields that do not fall into the categories above. Evaluation Process:

- 1. Carefully Review the given problem and code.
- 2. Identify the core technical features and key concepts.
- 3. Match these features with the relevant technical fields.
- 4. Select the most appropriate label from the predefined list.
- 5. Justify your selection with detailed reasoning.

Output Format: ‘‘‘md Label classification: <Chosen label from the predefined list> Label selection rationale: <Provide a detailed explanation, addressing the following aspects:>

. - Analysis of the core technical features

. - Alignment with the selected technical field

. - Reasons for not selecting other relevant fields

. - Confidence level in the selection

‘‘‘

Guidelines:

- - Only one predefined label should be selected.
- - Avoid ambiguity in your decision-making process.
- - Provide clear, specific reasoning for your classification.
- - Focus on the core technical aspects of the problem rather than superficial characteristics. Make sure to select the label that best represents the core technical features of the problem.

###### Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

Figure 23. Prompt for classify application scenarios in “Code QA” subset.

###### Fine-Grained Checklists Generation Prompt for “Code Gen” Subset

You are a senior code reviewer with extensive technical expertise. Your role is to formulate insightful, high-quality review questions for provided code and problems, ensuring they are accurately categorized by technical dimensions.

###### Criteria for evaluation questions:

- - Conduct a thorough analysis of the design intent and implementation strategy of the solution.
- - Go beyond surface-level checks and explore deeper potential optimizations or hidden value in the solution.
- - Combine the context of the problem and the provided answer to generate innovative, thought-provoking questions.
- - The questions should inspire developers to critically reflect on the quality of their solution and consider avenues for improvement.
- - Ensure that each question addresses a unique dimension, avoiding overlap or redundancy.

For each of the following dimensions, generate one question. Each question should focus on the corresponding evaluation aspect. Evaluation Dimensions:

- 1. Correctness Verification: Assess whether the code is capable of solving the problem accurately and passing all test cases.
- 2. Time Complexity Optimization: Focus on optimizing the efficiency and performance of the algorithm.
- 3. Space Complexity: Evaluate how efficiently the code uses memory and manages resources.
- 4. Code Readability: Focus on improving the clarity, understandability and standardization of the code.
- 5. Robustness Validation: Evaluate the code’s ability to handle edge cases and exceptions.
- 6. Algorithm Optimization: Assess whether the algorithm’s design and implementation are optimal.
- 7. Comprehensive Testing: Evaluate the scope and strategy behind the test coverage.
- 8. Output Format: Verify that the code’s output strictly adheres to the required format.
- 9. Code Style Consistency: Ensure that the code follows consistent coding standards and best practices.
- 10. Maintainability: Assess whether the code structure is modular, easily optimized and expandable in the future.

For each evaluation dimension, provide a question following this format. Output Format: ‘‘‘md

1. Question: <Specific judgment question>

. Dimension: <Select the most relevant evaluation . dimension>

. Classification reason: <Provide a detailed explanation of why this dimension was selected>

. Classification confidence: <Integer value between 1-10>

...

‘‘‘

Ensure that all 10 dimensions are covered with their respective questions, with each question addressing a distinct aspect of the code’s quality.

###### Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

Figure 24. Fine-Grained checklists generation prompt for “Code Gen” subset.

Fine-Grained Checklists Generation Prompt for “Code QA” Subset You are a senior code review expert with deep technical expertise. Your task is to formulate insightful and high-quality review questions for the provided questions and answers, ensuring the accurate classification of these questions across various technical dimensions. Criteria for Evaluating Questions:

- - Conduct an in-depth analysis of the design intent and implementation strategy presented in the answer.
- - Move beyond basic technical assessments, exploring the deeper value and potential for optimization within the solution.
- - Consider the context of both the question and the specific answer to propose unique and valuable technical questions that demonstrate innovative thinking.
- - Ensure that the questions inspire developers to critically evaluate the quality of the answer and its potential for improvement.
- - Questions addressing different technical dimensions must be independent, avoiding overlap or redundancy.

For each of the following dimensions, generate one question. Each question should focus on the corresponding evaluation aspect. Evaluation Dimensions:

- 1. Correctness Ensure the solution is accurate and addresses the core needs of the problem.
- 2. Completeness: Ensure the answer comprehensively covers all aspects of the problem without leaving out any critical information.
- 3. Performance: Assess whether the solution optimizes time and space complexity effectively.
- 4. Maintainability: Evaluate the clarity and structure of the solution to ensure it is easy to optimize and extend in the future.
- 5. Clarity: Ensure that the language used is concise and clear, facilitating understanding and quick access to essential information.
- 6. Depth: Evaluate whether the answer provides an in-depth analysis of the problem, offering a thorough understanding rather than a superficial response.
- 7. Practicality: Ensure that the solution is feasible and can be implemented effectively in real-world scenarios.
- 8. Logic Coherence: Assess whether the reasoning behind the solution is clear, coherent and convincing.
- 9. Innovation: Determine if the solution offers a unique perspective or introduces an innovative approach.
- 10. Reliability: Ensure that the opinions and suggestions are based on reliable, verifiable evidence.

For each evaluation dimension, provide a question following this format. Output Format: ‘‘‘md

1. Question: <Specific judgment question>

. Dimension: <Select the most relevant evaluation . dimension>

. Classification reason: <Provide a detailed explanation of why this dimension was selected>

. Classification confidence: <Integer value between 1-10>

...

‘‘‘

Ensure that all 10 dimensions are covered with their respective questions, with each question addressing a distinct aspect of the answer’s quality.

###### Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

Figure 25. Fine-Grained checklists generation prompt for “Code QA” subset.

Basic Critique Evaluation Prompt (with CoT) You are an AI Senior Programming Expert with advanced analytical reasoning capabilities. Your task is to conduct a comprehensive, step-by-step evaluation of an AI assistant’s solution using a detailed Chain of Thought (CoT) approach. Evaluation Framework:

###### Stage 1: Comprehensive Problem Understanding

- - Carefully analyze the original problem statement
- - Identify explicit and implicit requirements
- - Assess technical complexity and contextual constraints
- - Determine key evaluation dimensions Stage 2: Systematic Solution Decomposition
- - Break down the proposed solution into discrete components
- - Evaluate each component’s technical correctness
- - Assess code quality, efficiency and alignment with best practices
- - Trace the logical flow and problem-solving approach Stage 3: Detailed Performance Assessment
- - Verify functional accuracy
- - Analyze performance optimization potential
- - Evaluate scalability and maintainability
- - Check comprehensive error handling strategies Evaluation Methodology:

- 1. Maintain complete objectivity
- 2. Provide specific, actionable feedback
- 3. Explain reasoning transparently at each stage
- 4. Highlight both solution strengths and improvement opportunities Output Requirements:

- - Provide a clear, structured Chain of Thought explanation
- - Render a definitive assessment: [Correct/Error]
- - Include detailed reasoning for each evaluation stage
- - Offer constructive, implementable improvement suggestions Evaluation Criteria:
- - Technical Precision
- - Code Quality
- - Problem-Solving Approach
- - Efficiency and Optimization
- - Adherence to Best Practices

Output Format: ‘‘‘json {

. "is_assistant_correct": "Correct/Error",

. "detailed_reasoning": "Explicit step-by-step logical deduction" }

‘‘‘

Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

Figure 26. Basic critique evaluation prompt (with CoT).

###### Advanced Critique Evaluation Prompt (with CoT)

You are an Advanced Code Evaluation Specialist with Comprehensive Analytical Capabilities

Evaluation Methodology: Systematic Chain of Thought (CoT) Code Assessment Core Evaluation Framework:

###### Stage 1: Holistic Problem Understanding

- - Conduct an in-depth analysis of the problem domain
- - Identify multi-dimensional evaluation criteria
- - Map out complex technical expectations
- - Establish a comprehensive assessment perspective Stage 2: Detailed Code Decomposition
- - Systematically break down code into core components
- - Analyze each segment through multiple lenses:

- . 1. Technical Correctness
- . 2. Algorithmic Efficiency
- . 3. Code Readability
- . 4. Scalability
- . 5. Best Practice Adherence

###### Stage 3: Precision Scoring Methodology Evaluation Dimensions (Each Scored 1-10):

- 1. Algorithmic Complexity and Efficiency
- 2. Code Structure and Readability
- 3. Error Handling and Robustness
- 4. Performance Optimization
- 5. Solution Creativity
- 6. Technical Precision
- 7. Memory Management
- 8. Scalability Potential
- 9. Maintainability
- 10. Overall Problem-Solving Approach Scoring Nuanced Guidelines:

- - 1-2 points: Critical, fundamental failures
- - 3-4 points: Significant technical deficiencies
- - 5-6 points: Functional but requires major improvements
- - 7-8 points: Solid implementation with minor refinement needs
- - 9-10 points: Exceptional, near-perfect solution Comprehensive Evaluation Process:
- - Conduct granular, multi-dimensional assessment
- - Provide explicit reasoning for each score
- - Generate weighted comprehensive evaluation
- - Offer constructive, actionable improvement suggestions

Reasoning Chain Output Format: ‘‘‘json {

. "comprehensive_score": "Weighted comprehensive score",

. "detailed_reasoning": "Explicit step-by-step logical deduction" }

‘‘‘

Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

— Start of Fine-Grained Evaluation Checklists $Checklists

— End of Fine-Grained Evaluation Checklists —

Figure 27. Advanced critique evaluation prompt (with CoT).

Prompt for Refining Origin Answer Based on Model’s Critiques You are a Professional Self-Correction AI Specialist Objective: Systematically Refine and Improve Your Original Response Evaluation Process:

- 1. Carefully review the detailed feedback provided
- 2. Critically analyze your original response
- 3. Identify specific areas requiring improvement
- 4. Develop a comprehensive, precise correction strategy Key Refinement Focus Areas:

- - Accuracy of information
- - Completeness of solution
- - Clarity of explanation
- - Technical depth
- - Problem-solving approach Correction Guidelines:
- - Address each piece of feedback explicitly
- - Provide clear rationale for modifications
- - Enhance the original response’s quality
- - Maintain the core intent of the original answer

###### Output Requirements:

- 1. Detailed explanation of identified limitations
- 2. Comprehensive corrected response
- 3. Specific improvements made
- 4. Reasoning behind each modification Output Fromat: ‘‘‘json {

. "corrected_response": "Fully updated and refined answer", }

‘‘‘

Critical Evaluation Principles:

- - Be objective and critical
- - Focus on substantive improvements
- - Demonstrate intellectual rigor
- - Enhance overall response quality Mandatory Considerations:
- - Fully address all feedback points
- - Provide clear, precise modifications
- - Maintain professional and technical accuracy

###### Input Data:

— Start of Question $Question

— End of Question —

— Start of Answer $Answer

— End of Answer —

— Start of Feedback $Feedback

— End of Feedback —

Figure 28. Prompt for refining origin answer based on model’s critiques.

