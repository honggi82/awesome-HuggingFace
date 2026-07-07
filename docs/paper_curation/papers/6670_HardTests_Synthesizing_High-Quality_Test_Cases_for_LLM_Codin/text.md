arXiv:2505.24098v1[cs.CL]30May2025

# HARDTESTS: Synthesizing High-Quality Test Cases for LLM Coding

Zhongmou He1∗ Yee Man Choi1∗ Kexun Zhang1∗† Jiabao Ji2 Junting Zhou1 Dejia Xu3 Ivan Bercovich2 Aidan Zhang1 Lei Li1

1Carnegie Mellon University 2UC Santa Barbara 3UT Austin

## Abstract

Verifiers play a crucial role in large language model (LLM) reasoning, needed by post-training techniques such as reinforcement learning. However, reliable verifiers are hard to get for difficult coding problems, because a well-disguised wrong solution may only be detected by carefully human-written edge cases that are difficult to synthesize. To address this issue, we propose HARDTESTGEN, a pipeline for high-quality test synthesis using LLMs. With this pipeline, we curate a comprehensive competitive programming dataset HARDTESTS with 47k problems and synthetic high-quality tests. Compared with existing tests, HARDTESTGEN tests demonstrate precision that is 11.3 percentage points higher and recall that is 17.5 percentage points higher when evaluating LLM-generated code. For harder problems, the improvement in precision can be as large as 40 points. HARDTESTS also proves to be more effective for model training, measured by downstream code generation performance. We will open-source our dataset and synthesis pipeline at https://leililab.github.io/HardTests/.

## 1 Introduction

Post-training large language models (LLMs) with outcome verifiers2 (Guo et al., 2025; Kimi Team et al., 2025) can greatly improve their reasoning ability. LLMs trained with these techniques are approaching the level of the best humans on challenging problems in math and programming olympiads (OpenAI et al., 2025). To properly assign outcome rewards in post-training, reliable verifiers are needed for both reinforcement learning and (self-) distillation.

Verification is a non-trivial process. How good are current verifiers? How to get better verifiers? How much does verifier quality matter in LLM post-training? Verification loops become increasingly less tractable as the notion of correctness increases in complexity. For math, it is relatively straightforward to determine correctness by looking at the answer, whereas verifying programs needs execution. An effective approach to verify programs is through test cases (Le et al., 2022; Singh et al., 2023). However, most datasets of coding problems and associated test cases are less than comprehensive. 60% of the programs that pass test cases in APPS (Hendrycks et al., 2021) are in fact, wrong. 46% of the programs that pass test cases in CodeContests (Li et al., 2022) are semantically correct, but too inefficient to pass human-written tests. More importantly, scraping human-written tests is unfeasible

— according to our study, for 80% of the problems, human-written test cases are proprietary and impossible to scrape, demanding synthesized tests. Previous test synthesis attempts, such as TACO (Li et al., 2023), have limited reliability, with the false positive rate being more than 90% for difficult problems in our experiments.

∗Equal contribution. †Project lead. Correspondence to {zhongmou,yeemanc,kexunz}@andrew.cmu.edu 2In this paper, the term “verifier” refers to rule-based systems that attempt to check the correctness of problem

solutions. It is used to differentiate from model-based rewards, such as those in RLHF. “Verifiers” are not necessarily formal and do not necessarily guarantee correctness.

Preprint. Under review.

Precision

Recall

100

| |TACO<br><br>| |
|---|
<br><br>CodeContests<br><br>| |
|---|
<br><br>HardTests| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

80

60

40

20

0

AtCoder Atcoder Hard

Codeforces Codeforces Hard

AtCoder Atcoder Hard

Codeforces Codeforces Hard

- Figure 1: HARDTESTS test cases are significantly better than the baselines. The large improvement in precision indicates that our tests greatly reduce false positives and are indeed harder.

The low quality of synthetic tests is due to the challenging nature of coding problems. Coding competitions often require efficient solutions with advanced data structures and algorithms. A bad choice of algorithm can lead to a well-disguised wrong solution, which may easily pass most random tests but still break on human-written special cases. For example, on a random rooted tree with n nodes and depth of d, an algorithm with the time complexity of Θ(nd) can be very efficient, as E[d] = Θ(log n) for randomly generated trees (Devroye et al., 2012). For such an algorithm to time out, the test case needs to be a valid tree that is large enough (so that n is large) and special enough (so that d is large). A chain (each non-leaf node has exactly one child), whose depth d = n can cause the algorithm to be as slow as Θ(n2). We need valid, comprehensive tests that cover edge cases.

Generating valid and comprehensive tests is hard. Existing test synthesis methods, such as CodeT (Chen et al., 2023) and TACO (Li et al., 2023) rely on LLMs to directly write test inputs. While this works when the test inputs are small, it can barely keep the test inputs valid at a larger scale, let alone make them special. To alleviate these issues, we propose HARDTESTGEN, an LLM-based test synthesis pipeline. Our main insights are 1) Test case validity is better preserved when generated from LLM-produced programs rather than directly from the LLMs themselves, and 2) Each test generator has different hypotheses about the programs under test and creates tests from a different distribution. With these insights, HARDTESTGEN prompts an LLM with different aspects to consider for test cases, extracts LLM-generated test generator programs, and filters the test cases using human-written oracle programs, which widely exist for all problems in online coding competitions.

With HARDTESTGEN, we curate HARDTESTS, a comprehensive dataset for coding competitions with 47,136 problems and high-quality test cases. As shown in Figure 1, compared to existing test synthesizers, HARDTESTS tests are more reliable in terms of precision and recall when evaluating programs. The gap in precision can be as large as 40 percentage points for harder problems. Higher reliability of verification makes HARDTESTS the ideal playground for post-training research in the coding domain.

To further demonstrate the benefits of high-quality tests, we conduct post-training experiments with HARDTESTS and baseline tests. Our experiments in 3 different scenarios show that test quality matters significantly for self-distillation and reinforcement learning. Higher-quality tests can lead to improvements in downstream performance. However, our results also indicate that test quality is less important for teacher distillation.

In summary, this work provides:

- • HARDTESTGEN, an LLM-based test synthesis pipeline that generates high-quality test cases for coding problems, improving precision by 11.3 points and recall by 17.5 points on average.
- • HARDTESTS, a comprehensive problem set for competition-level code generation, with 47,136 problems, among which 32.5k have high-quality test cases generated by HARDTESTGEN.
- • Empirical analyses on how test quality affects LLM post-training. We show that test quality is of great importance for reinforcement learning and self-distillation.

## 2 Related work

RLVR. Reinforcement learning has shown great potential in improving LLM reasoning abilities in various domains, such as math (Guo et al., 2025; Zeng et al., 2025b; Ren et al., 2025) and coding (OpenAI, 2025; Liu & Zhang, 2025; Luo et al.). The resulting long-reasoning LLMs, such as OpenAI-o3 (OpenAI, 2024) and DeepSeek-R1 (Guo et al., 2025), largely outperform shortreasoning LLMs through simple RL training to improve outcome-based reward, i.e., whether the model-generated code solution passes all test cases. Although some previous works have explored heuristic rules for selecting training data to improve RL performance (Ye et al., 2025; Wang et al., 2025b; Li et al., 2025) or reward design (Hou et al., 2025; Kimi Team et al., 2025; Costello et al., 2025), the impact of test case quality on coding LLMs during RL training remains underexplored. In this work, we show that high-quality test cases, those better at detecting subtle bugs in code, can largely improve coding LLM performance after RL training.

LLM-based test synthesis. Test cases are crucial in evaluating the functional correctness and performance of LLM-generated code. Benchmarks such as HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and APPS (Hendrycks et al., 2021) provide hand-written test cases that serve as a proxy for code correctness. However, such human-authored test cases are often only publicly available for a limited set of problems. Early approaches such as EvoSuite (Fraser & Arcuri, 2011) and Pynguin (Lukasczyk & Fraser, 2022) employ search-based heuristics methods. More recently, CodeContests

- (Li et al., 2022) generates additional test cases by mutating existing crawled inputs. Several efforts leverage LLMs to synthesize test inputs and (pseudo)-oracle programs for test outputs. CodeT (Chen

- et al., 2023) and ALGO (Zhang et al., 2023) rely on LLMs to generate both tests and reference programs for existing coding problems. EvalPlus (Liu et al., 2023) extends HumanEval with more tests by providing the reference implementation to LLMs to synthesize seed input.Similarly, TACO

- (Li et al., 2023) also generates test inputs with LLMs and outputs with reference implementation. STGen (Peng et al., 2025) generates stressful test cases for evaluating the time efficiency of code. KodCode (Xu et al., 2025) and AceCoder (Zeng et al., 2025a) push synthetic data even more by generating coding questions, reference solutions, and tests all with LLMs. Although existing LLM test synthesis methods prove to be useful in many scenarios, their quality is far from perfect. We present a more thorough discussion on the quality issues in LLM synthetic tests and their implications in Appendix A.1. Concurrently with our work, rStar-Coder (Liu et al., 2025) and HF-Codeforces (Penedo et al., 2025) also study more reliable test synthesis in the competition context. Comparing to them, our work highlights a thorough analysis of test quality and a unique set of post-training experiments that demonstrate the downstream effects of high-quality tests.

Datasets for competition code generation. Existing datasets for competition code generation focus on scaling the number of problems and CoTs. Luo et al. filters a high-quality 24k problemset of TACO, LiveCodeBench, and other contest programming problems. CodeForces-CoTs, the dataset of 10k Codeforces problems created by Penedo et al. (2025), contains 100k reasoning traces and solutions generated by DeepSeek R1. OpenCodeReasoning (Ahmad et al., 2025) also compiles a dataset of 28k problems, generates 735k reasoning traces, and filters them for syntactic correctness. While these efforts have shown that better models can be trained with more data and more trajectories from teacher models, they are facing a “code verifiability crisis”, as described by Open-R1 (Face, 2025), and programs that pass test cases in these problem sets are not necessarily correct. In our paper, we curate HARDTESTS, the competitive coding problem set with the most number of problems (47k). More importantly, we push the scaling of training data towards higher quality of test cases and evaluate how test quality affects model training.

## 3 HARDTESTGEN: Synthesizing High-Quality Test Cases

### 3.1 Problem Setting

Coding problems. We study test generation for coding problems with natural language specifications. We denote the space of problem specifications as X, the space of candidate programs as Y, and the space of test suites as V. A test suite V is a set of test cases {t1,t2 ··· ,t|V |}. A test case is a pair (a,c), where a is an input to a program, and c is a checker for the corresponding output 3. A candidate

3In most cases, the output checker is simply a comparison between golden outputs and program outputs. Others might be equivalence checkers that do not directly compare strings.

###### Baselines HardTestGen

Problem

Problem

Specification

Specification

Input Validator

LLM

Type3 (Hacking)

Type1 (Direct)

Type2 (Regular)

Filtering

？

？

？

？

？

LLM Rule-based Perturbation Inputs

Input Generator

Inputs

Inputs

Inputs

Input Generator

Inputs

- Figure 2: Comparison of the input generation process between previous test synthesizers (left) and HARDTESTGEN (right).

program y ∈ Y takes an input and generates an output y(a), which is then sent to the output checker c for a boolean verdict c(y(a)) ∈ {⊤,⊥}. When y exceeds a pre-defined runtime limit, its verdict is also ⊥.

Oracle tests and correctness. For every coding problem x ∈ X, we assume the existence of an oracle test suite V ∗ ∈ V, which definitively tells the correctness of a program y ∈ Y , i.e.

Correctness(y,V ∗) :=

ci(y(ai)). (1)

(ai,ci)∈V ∗

In practice, the oracle tests are usually carefully written and proprietary by problem authors. Only very few of them are available for downloading, which makes them infeasible for model training.

Oracle programs. Compared to rarely available oracle tests, oracle programs (y∗ such that Correctness(y∗,V ∗) = ⊤) are available for almost all coding problems in online competitions. Therefore, we assume the existence of oracle programs y∗ in our setting.

Test synthesis. Given a problem x, and an oracle program y∗, the task of test synthesis is to create a test suite V that agrees with V ∗, i.e., we want Correctness(y,V ) = Correctness(y,V ∗) for as many ys as possible. In HARDTESTGEN, we create a set of inputs {a1,a2,··· ,a|V |} and utilize the oracle program to get the outputs, i.e., ci = y∗(ai).

### 3.2 Generating Inputs of Test Cases

We synthesize three types of test inputs. One is directly generated by an LLM, while the other two are generated by LLM-generated programs. Before generating inputs, we first prompt an LLM to generate an input validator in Python that checks whether a given input satisfies all the constraints in the problem specification. We subsequently prompt the LLM to generate the inputs. In the prompt, we include the input validator and an oracle program, as we find that doing so increases the likelihood of synthesizing valid inputs. Figure 2 illustrates the differences between the input generation processes of previous test synthesiziers and HARDTESTGEN.

- Type 1. Directly Generated Inputs. We prompt an LLM to directly generate nD = 10 inputs by imitating the sample test cases provided in the problem specification. This type of input is typically small in scale, making it easy to generate and understand, and allowing for quick testing of the candidate program’s functional correctness.
- Type 2. Regular Inputs. Regular inputs are generated randomly according to the constraints specified in the problem specifications. For most problems, we prompt an LLM to generate a Python function gR with no parameters that returns a random input on each call. We call this function nR = 20 times to get nR random inputs. For some problems, there are some unusual categories of outputs that are rarely triggered by random inputs. For example, when a problem’s expected output is either “Yes” or “No”, the correct output for almost all random inputs might be “Yes”. In such cases, random

inputs can potentially lead to false positives. For these problems, we prompt an LLM to generate mR functions, each corresponding to one output category (e.g., “Yes” and “No”). We call each function

nR = 10 times to obtain a total of mR × nR inputs with their outputs specified.

- Type 3. Hacking Inputs. Some well-disguised false positives cannot be easily detected with random inputs. For example, some programs may be functionally correct but inefficient in worst-case scenarios, or some programs may fail to handle certain edge cases that require special treatment. Therefore, we first prompt an LLM to list several candidate programs for the problem in natural language. Then, we prompt it to generate mH input generation functions, each attempting to cause one candidate program to fail. Each function is called nH = 10 times, generating mH × nH inputs.

After generating the inputs, we filter out all inputs that fail to pass the examination of the input validator.

### 3.3 Generating Outputs of Test Cases

We use human-written oracle programs that exist for all online competitions to test outputs. For each problem, we use at most noracle = 8 oracle programs, prioritizing those from more reliable sources. Each oracle program generates outputs for all synthesized inputs. If the outputs generated by two oracle programs match for more than 90% of the cases, we consider the outputs to be acceptable and adopt the matching portion as the final outputs.

For the majority of problems, a simple string comparison between two outputs is sufficient to determine whether they match. However, some problems require a special judge. For example, a problem might require returning a set (where element order does not matter) or a sequence of operations that achieves a certain effect. In that case, we prompt an LLM to implement a special judge function. This function takes the input and two outputs as parameters, and returns a Boolean value indicating whether the two outputs are equivalent. In our dataset, 25.4% of the problems require a special judge function. In subsequent training and testing processes, this function will continue to be used to determine whether the candidate output and the reference output match.

In our dataset, we use GPT-4o to generate all of the above content. For all functions that need to be generated, we include two to three carefully crafted examples in the prompts. The implementation details of HARDTESTGEN (e.g., prompts), the number of generated test cases, the failure rate and reasons for failure, as well as a concrete example, are provided in Appendix A.2.

### 3.4 HARDTESTS: 47k Problems with High-Quality Test Cases

The HARDTESTS dataset comprises 47,136 competitive programming problems with high-quality test cases, aggregated from 13 major online judges (OJs) for competitive programming. The dataset is constructed from five direct data sources: Codeforces, AtCoder, Luogu, CodeContests (Li et al.,

- 2022), and TACO (Li et al., 2023). We apply HARDTESTGEN to synthesize test cases for 32.5k problems among them. The detailed constitution and description of the data sources are described in Appendix A.3.

Cleaning, deduplication, and decontamination. For problems with only non-English descriptions, we translated them into English using GPT-4o. To handle overlapping content among the five direct data sources, we filtered out duplicated problems using problem IDs and n-gram overlaps in description, prioritizing versions from the original platforms rather than mirror sites. For correct programs, we retained all available versions and annotated them with their respective sources. We conduct decontamination by removing the problems that are in LiveCodeBench (Jain et al., 2025b) from our dataset. Since most of its problems are from Codeforces and AtCoder, we directly compare the URLs to the problems.

Labelling difficulty. We retained the difficulty labels assigned by all five data sources in our dataset. In the experiments presented in Section 4, we used the difficulty labels from Luogu, as it provides consistent and fine-grained labels for problems from both AtCoder and Codeforces. Luogu’s difficulty labels are divided into seven levels, with the first level representing beginner-level problems and the seventh level corresponding to problems at the level of national competitions.

## 4 Direct Evaluation of Test Case Quality

### 4.1 Evaluation Criteria

We regard the testing of candidate programs as a binary classification process: a program is classified as positive if it passes all test cases, and negative otherwise. To directly assess the quality of test cases, we evaluate how good they are as binary classifiers. Given a problem x, an oracle test suite V ∗, a synthesized test suite V , and a set of candidate programs {y1 ···yn}, we categorize the programs with their correctness according to V and V ∗. When V and V ∗ both find a candidate program correct, it’s a true positive (TP). When V finds a program correct while V ∗ finds it wrong, it’s a false positive (FP). Similarly, we can define true negatives and false negatives. With these categories defined, we use precision and recall to measure test quality:

#### TP TP + FP

TP TP + FN

Precision =

, Recall =

.

### 4.2 Baselines

CodeContests. CodeContests (Li et al., 2022) primarily consists of problems from Codeforces. Codeforces only provides test cases within certain length constraints. CodeContests collects these and refers to them as “private test cases.” Additionally, it generates new test cases by introducing random perturbations to the private test cases; these are referred to as "generated test cases." This gives CodeContests an unfair advantage as it has access to the distribution of oracle tests. In our experiments, we only use generated test cases, which reduces the unfairness but does not eliminate it.

TACO. TACO (Li et al., 2023) integrates several existing datasets, such as APPS (Hendrycks et al., 2021) and CodeContests (Li et al., 2022), while retaining their test cases. In addition to this, TACO generates several additional test cases by using GPT-4o to directly generate the inputs and using oracle programs for outputs. Furthermore, we observed that for some problems from AtCoder and Codeforces, the TACO test cases included official test cases. To ensure fair comparisons, we removed these official test cases.

Ablative Baselines. We also evaluate HARDTESTGEN with only Type 1 or Type 2 inputs to demonstrate the necessity of all 3 types. Notably, the scenario with only Type 1, LLM directly generated inputs, very much resembles many existing test synthesis methods such as KodCoder (Xu et al., 2025), except that they synthesize not only the inputs but also the oracle programs.

### 4.3 Evaluation Pipeline

To evaluate the accuracy of rewards that our test cases can give to model training, we evaluate the precision and recall over candidate programs generated by LLMs and written by humans on subsets of problems in HARDTESTS. Details about the evaluation protocol can be found in Appendix A.4.

Generating candidate problems. To compare our tests with other synthesizers, we choose the problems that exist in both HARDTESTS and the baseline datasets. For problems from AtCoder, we select 653 problems that exist in both HARDTESTS and TACO. For problems from Codeforces, we select 600 problems that exist in HARDTESTS, CodeContests, and TACO.

Generating candidate programs. We compare our tests with baseline tests on candidate programs generated by 3 LLMs and also by human programmers. Specifically, we use three LLMs: Qwen2.5Coder-7B-Instruct (Yang et al., 2024), Qwen2.5-Coder-14B-Instruct, and GPT-4o. For each problem, we sample 10 candidate programs from each LLM using a temperature of 0.7 and a top-p of 0.95. We also randomly select 10 real-world human submissions for each problem.

Generating gold labels. We need gold labels to compute precision and recall. For AtCoder, we run candidate programs on official tests that have been previously made available. For Codeforces, we submit candidate programs to the website to obtain ground-truth verdicts. The human-written candidate programs are sampled from MatrixStudio/Codeforces-Python-Submissions, which provides official verdicts. We then use synthetic test cases to classify the correctness of these programs and compare the results against the ground-truth labels, thereby evaluating test case quality.

### 4.4 Results

We evaluate the correctness of programs written by three LLMs and human programmers for problems from AtCoder and Codeforces using test cases from TACO, CodeContests, and HARDTESTS. The results are in Table 1 and 2. We present qualitative analyses of the synthetic tests in Appendix A.5.

We find that HARDTESTS significantly outperforms TACO and CodeContests in terms of both precision and recall under most evaluation settings. Moreover, this advantage becomes more pronounced as problem difficulty increases. For example, for the Qwen2.5-Coder-7B-Instruct model on AtCoder problems with difficulty level 4+, TACO achieves a precision of 21.67 and a recall of 68.42, whereas HARDTESTS achieves a precision of 60.00 and a recall of 94.74. This implies that using HARDTESTS during RL training would yield more true positive rewards and much fewer false positive rewards.

Furthermore, we observe that as the source of programs becomes less “intelligent” (ranging from human-written to 7B LLM-generated), the precision advantage of HARDTESTS becomes more pronounced. We attribute this to the fact that less skilled programmers are more likely to produce functionally correct but inefficient programs. For instance, among incorrect human-written programs, 14.9% are due to TLE (Time Limit Exceeded), whereas among the incorrect programs written by

- Table 1: Precision and recall of the test cases of TACO, HARDTESTS, and ablative baseline on AtCoder. HT-TYPE1 refers to the results using only the test cases of Type 1 from HARDTESTS. while TH-TYPE1+2 refers to the results using only the test cases of Type 1 and Type 2 from HARDTESTS.

difficulty 1 difficulty 2 difficulty 3 difficulty 4+ average

prec. recall prec. recall prec. recall prec. recall prec. recall Qwen2.5-Coder-7B-Instruct

TACO 99.48 77.09 89.66 62.90 69.07 81.71 21.67 68.42 69.97 72.53 HT-TYPE1 94.63 99.84 74.70 100.0 42.20 89.02 10.40 94.74 55.48 95.90 HT-TYPE1+2 97.85 99.35 97.58 100.0 74.23 87.80 58.06 94.74 81.93 95.47

- HARDTESTS 98.15 98.95 97.64 97.58 86.75 87.80 60.00 94.74 85.64 94.77 Qwen2.5-Coder-14B-Instruct

- TACO 99.82 78.00 93.24 69.00 80.23 73.40 39.33 72.92 78.16 73.33 HT-TYPE1 96.21 99.72 77.22 100.0 58.90 96.81 18.50 97.92 62.71 98.61 HT-TYPE1+2 97.31 99.02 94.79 100.0 87.50 96.81 65.71 95.83 86.33 97.92 HARDTESTS 97.99 99.02 96.95 95.50 93.33 96.81 67.16 93.75 88.86 96.27

GPT-4o

- TACO 100.0 73.06 99.75 67.29 92.74 74.08 62.07 71.05 88.64 71.37 HT-TYPE1 99.42 99.47 94.31 99.32 86.39 99.42 45.56 99.67 81.42 99.47 HT-TYPE1+2 99.53 99.18 99.82 97.60 96.04 98.45 79.00 99.01 93.60 98.56

- HARDTESTS 99.53 99.18 100.0 97.43 96.04 98.45 84.18 98.03 94.94 98.27

- Table 2: Precision and recall of the test cases of TACO, CodeContests, and HARDTESTS evaluated using LLM-generated programs for problems on Codeforces.

difficulty 1 difficulty 2 difficulty 3 difficulty 4 average

prec. recall prec. recall prec. recall prec. recall prec. recall Qwen2.5-Coder-7B-Instruct

TACO 89.64 86.13 71.07 92.91 31.06 39.47 9.82 100.0 50.40 79.63 CodeContests 85.74 89.24 63.73 97.64 23.80 47.54 6.67 100.0 44.99 83.61 HARDTESTS 87.61 95.45 93.30 98.82 48.38 55.61 50.00 100.0 69.82 87.47

Qwen2.5-Coder-14B-Instruct

TACO 80.67 87.45 83.88 81.13 53.87 73.88 25.76 100.0 61.05 85.62 CodeContests 79.70 95.59 79.29 86.16 46.49 91.84 18.68 100.0 56.04 93.40 HARDTESTS 83.19 98.64 88.44 100.0 67.47 80.41 46.58 90.80 71.42 92.46

GPT-4o

TACO 99.58 80.02 95.76 81.72 89.64 74.83 62.64 78.17 86.91 78.69 CodeContests 99.47 94.80 95.25 89.89 86.83 87.08 58.28 94.31 84.96 91.52 HARDTESTS 98.80 98.20 95.66 98.71 92.73 88.50 79.82 94.31 92.00 94.93

Human Submission

TACO 96.28 88.89 91.48 81.59 75.90 78.84 62.23 73.77 81.47 64.62 CodeContests 94.15 90.06 87.47 89.99 73.11 85.10 56.80 79.88 77.88 69.01 HARDTESTS 93.29 94.13 85.15 95.05 73.71 93.59 64.16 89.35 79.08 74.42

the three LLMs, 30.0% are due to TLE. Consequently, the larger and more diverse test cases in HARDTESTS are more likely to catch inefficient programs than the small-scale test cases in TACO and CodeContests.

Compared with the ablative baselines in Table 1, HARDTESTS that includes Type2 (Regular) and Type3 (Hacking) test cases consistently leads to a precision improvement ranging from 2% to 48%, while the decrease in recall is always within 2.5%. This demonstrates the necessity for having different types of tests.

## 5 Downstream Effects of Test Case Quality in LLM Post-Training

In this section, we aim to answer two questions with HARDTESTS: when does verifier/test quality matter, and how much does it matter in post-training? We run experiments in 3 different post-training scenarios: teacher-distillation, self-distillation, and reinforcement learning. We examine how much verifier quality affects the training results in code generation, if any.

### 5.1 Experiment Setup

Teacher-distillation. Various papers, such as DeepSeek-R1 (Guo et al., 2025) suggest that fine-tuning a smaller student model with reasoning trajectories from a stronger reasoning model can greatly improve the student’s performance. In this scenario, verifiers can be used to filter out the incorrect trajectories. We sample one reasoning trajectory with a C++ solution program from DeepSeek-R1 for each question in HARDTESTGEN, obtaining 46.6k trajectories in total after deduplication and decontamination against all LiveCodeBench questions. We fine-tune two models from Qwen2.5Coder-Instruct-7B: one with all 46.6k trajectories, the other with 13k trajectories that are correct according to HARDTESTS. As a baseline, we also evaluate OlympicCoder-7B (Face, 2025), another Qwen2.5-Coder derivation fine-tuned with ∼100k trajectories of ∼10k Codeforces problems.

Self-distillation. Fine-tuning a model with its own reasoning trajectories can also improve its reasoning ability (Zelikman et al., 2022). Hence, determining which trajectories to use is a critical issue. To examine the effects of test quality, we sampled 5 traces of Qwen3-4B and used the tests generated by HARDTESTGEN for filtering. We selected 4989 questions where there is at least one Qwen3-4B generated program that passes the tests and at least one that fails the tests. We create 3 datasets for self-fine-tuning, each containing one trajectory per question. The bad 5k randomly samples one incorrect trajectory for each question. The good 5k randomly samples one correct trajectory. The random 5k randomly samples one trajectory, regardless of its correctness, for each question. We further fine-tune Qwen3-4B with these 3 datasets and compare the performance of the resulting models. All our fine-tuning experiments were done with Llama-factory (Zheng et al., 2024).

Reinforcement learning. Verifier feedback is an option for distillation, but it is a must for reinforcement learning. To investigate how verifier quality affects RL, we train Qwen3-4B with RL using the same problem set, the identical training setup, and different test cases. We select a problem set with ∼5k problems that exist in both HARDTESTS and TACO for training. We use a modified version of veRL (Sheng et al., 2024) inspired by Code-R1 (Liu & Zhang, 2025) for training with GRPO (Shao et al., 2024). When a program passes all tests, it gets a reward of 1, otherwise, it gets a reward of 0. We compare different verifiers by looking at the final performance and the validation curve.

Evaluation protocol. We use LiveCodeBench (Jain et al., 2025b) version 5 to evaluate the model performance. Since all the programs we use for tuning are in C++, we build an evaluation pipeline for evaluating C++ programs for LiveCodeBench and select a 105-problem subset where all problems have test cases of “stdin” type. We name this subset of problems we use “LiveCodeBench-105”. Details about our training and evaluation procedure can be found in Appendix A.6, including the problems and hyperparameters we use for training and the sampling parameters we use for evaluation.

### 5.2 Results

Teacher-distillation benefits more from question scaling than test quality or sample scaling. We evaluate models fine-tuned from Qwen2.5-Coder-7B using different training sets on LiveCodeBench105 and report the results in Table 3. Note that the difficulty labels are obtained from LiveCodeBench. The model trained with HARDTESTS with all 46.6k examples outperforms OlympicCoder-7B (trained with 100k trajectories of 10k questions), suggesting that the quality and diversity of training questions matter more than the number of training samples. Interestingly, the model trained on smaller but more curated subsets (13k filtered trajectories) does not match the performance of using larger, unfiltered data, suggesting that data scaling dominates trajectory correctness in the teacher-distillation setting. This observation aligns with the concurrent findings from OpenCodeReasoning (Ahmad et al., 2025).

Self-distillation performance is highly dependent on sample quality and needs a good verifier. We evaluated variants of Qwen3-4B models self-distilled with different 5k subsets on LiveCodeBench105 and present the results in Table 4. Model self-distilled from incorrect samples identified by HARDTESTGEN’s tests drops more significantly in pass@k. Self-distillation with randomly selected

Table 3: pass@k (%) of teacher-distilled LLMs based on Qwen2.5-Coder-7B on LiveCodeBench-105.

Easy Medium Hard All pass@1 pass@1 pass@1 pass@1 pass@10

QC2.5-7B-Ins 58.75 9.58 2.46 16.95 27.62 OlympicCoder-7B (100k trajectories) 65.83 41.25 2.46 25.81 46.67 QC2.5-7B-Ins + HARDTESTS (13k, filtered) 77.08 29.17 1.75 25.24 39.05 QC2.5-7B-Ins + HARDTESTS (46.6k, full) 83.65 44.58 6.49 32.86 53.33

Table 4: pass@k (%) self-distilled LLMs based on Qwen3-4B on LiveCodeBench-105.

Easy Medium Hard All pass@1 pass@1 pass@1 pass@1 pass@5 pass@10

Qwen3-4B 88.75 53.33 11.05 38.48 52.04 56.19 Qwen3-4B (with bad 5k) 84.17 45.42 8.07 34.00 48.42 54.92 Qwen3-4B (with random 5k) 84.58 36.25 9.12 32.75 50.85 57.14 Qwen3-4B (with good 5k) 85.42 47.08 10.53 36.00 55.15 60.00

RL with HardTests (ours)

0.20

RL with TACO

0.15

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.10

| |
|---|

0.05

| |
|---|

| |
|---|

0.00

0 20 40 60 80 100 Step

- Figure 3: RL Validation Rewards Over Time. Reward from HARDTESTS makes the training better. data could harm pass@1 even more, despite the slight improvements in pass@10. In contrast, using a

- 5k subset verified by HARDTESTGEN’s test cases results in a smaller drop in pass@1 and a notable gain in pass@5 and pass@10, suggesting that verifiers are important to self-distillation.

Test quality matters significantly for reinforcement learning. As shown in Figure 3, the validation reward curve for HARDTESTS during RL training is generally higher than that for TACO. This indicates that for the same problems, HARDTESTS is giving better rewards. To evaluate on LiveCodeBench-105, we run the best checkpoints (according to valid reward) of both training jobs within 100 steps. As reported in Table 5, TACO tests hurt the model’s overall performance, while HARDTESTS improves the model’s overall performance.

Table 5: pass@k (%) for LLMs RL-trained from Qwen3-4B on LiveCodeBench-105.

pass@1 pass@5 pass@10

Qwen3-4B 38.48 52.04 56.19 Qwen3-4B (RL with TACO) 36.95 51.01 57.14 Qwen3-4B (RL with HARDTESTS) 39.42 57.89 64.76

- 6 Conclusion

We present HARDTESTGEN, an LLM-based test synthesis pipeline, which is used to create HARDTESTS, a competitive coding dataset with 47k problems and significantly higher-quality tests. We examine when and how much test quality matters in LLM post-training, showing that harder tests generated by HARDTESTGEN can indeed help LLM post-training in many scenarios.

## Limitation

Although HARDTESTS has higher-quality tests than the baselines, they are still not as good as humanwritten ones. Moreover, we assume the existence of oracle solutions to utilize HARDTESTGEN, which may not be true for some coding domains. To address this issue, we briefly discuss an initial idea for synthesizing tests without oracles in Appendix A.7. Another limitation of the HARDTESTGEN is that the code being tested is constrained to a single file that uses Standard I/O for input and output. However, many real-world coding problems are more complicated, e.g. coding problems in SWEbench that may involve file I/O or web I/O, and we leave the exploration of applying HARDTESTGEN to these scenarios as future work.

## Acknowledgments and Disclosure of Funding

The OpenAI API credits used in this paper were partially supported by the OpenAI Research Access Program. The training compute used was partially supported by National Center for Supercomputing Applications and ScOp Venture Capital. KZ was partially supported by ChipAgents.ai.

## References

Wasi Uddin Ahmad, Sean Narenthiran, Somshubra Majumdar, Aleksander Ficek, Siddhartha Jain, Jocelyn Huang, Vahid Noroozi, and Boris Ginsburg. Opencodereasoning: Advancing data distillation for competitive coding, 2025. URL https://arxiv.org/abs/2504.01943.

Toufique Ahmed, Martin Hirzel, Rangeet Pan, Avraham Shinnar, and Saurabh Sinha. Tdd-bench verified: Can llms generate tests for issues before they get resolved?, 2024. URL https://arxiv. org/abs/2412.02883.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.07732.

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. In ICLR, 2023.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Caia Costello, Simon Guo, Anna Goldie, and Azalia Mirhoseini. Think, prune, train, improve: Scaling reasoning without scaling models. arXiv preprint arXiv: 2504.18116, 2025.

Luc Devroye, Omar Fawzi, and Nicolas Fraiman. Depth properties of scaled attachment random recursive trees. Random Structures & Algorithms, 41(1):66–98, 2012.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https: //github.com/huggingface/open-r1.

Gordon Fraser and Andrea Arcuri. Evosuite: automatic test suite generation for object-oriented software. In Proceedings of the 19th ACM SIGSOFT Symposium and the 13th European Conference on Foundations of Software Engineering, ESEC/FSE ’11, pp. 416–419, New York, NY, USA, 2011. Association for Computing Machinery. ISBN 9781450304436. doi: 10.1145/2025113.2025179. URL https://doi.org/10.1145/2025113.2025179.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv: 2504.01296, 2025.

Kush Jain, Gabriel Synnaeve, and Baptiste Rozière. Testgeneval: A real world unit test generation and test completion benchmark, 2025a. URL https://arxiv.org/abs/2410.00752.

Naman Jain, Manish Shetty, Tianjun Zhang, King Han, Koushik Sen, and Ion Stoica. R2E: Turning any github repository into a programming agent environment. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 21196–21224. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/jain24c.html.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=chfJJYC3iL.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms, 2025. URL https://arxiv.org/abs/2501.12599.

Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, and Steven Chu Hong Hoi. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35:21314–21328, 2022.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling. arXiv preprint arXiv: 2502.11886, 2025.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. arXiv preprint arXiv:2203.07814, 2022.

Jonathan Light, Yue Wu, Yiyou Sun, Wenchao Yu, Yanchi liu, Xujiang Zhao, Ziniu Hu, Haifeng Chen, and Wei Cheng. Scattered forest search: Smarter code space exploration with llms, 2025. URL https://arxiv.org/abs/2411.05010.

Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. 2025. Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by

chatgpt really correct? rigorous evaluation of large language models for code generation, 2023. URL https://arxiv.org/abs/2305.01210.

Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, and Mao Yang. rstar-coder: Scaling competitive code reasoning with a large-scale verified dataset, 2025. URL https://arxiv.org/abs/2505.21297.

Stephan Lukasczyk and Gordon Fraser. Pynguin: automated unit test generation for python. In Proceedings of the ACM/IEEE 44th International Conference on Software Engineering: Companion Proceedings, ICSE ’22. ACM, May 2022. doi: 10.1145/3510454.3516829. URL http://dx.doi.org/10.1145/3510454.3516829.

Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level.

Niels Mündler, Mark Niklas Müller, Jingxuan He, and Martin Vechev. Swt-bench: Testing and validating real-world bug-fixes with code agents, 2025. URL https://arxiv.org/abs/2406. 12952.

OpenAI. Openai o1 system card. arXiv preprint arXiv: 2412.16720, 2024. OpenAI. Competitive programming with large reasoning models. arXiv preprint arXiv: 2502.06807,

2025.

OpenAI, :, Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaiev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, Jerry Tworek, Lorenz Kuhn, Lukasz Kaiser, Mark Chen, Max Schwarzer, Mostafa Rohaninejad, Nat McAleese, o3 contributors, Oleg Mürk, Rhythm Garg, Rui Shu, Szymon Sidor, Vineet Kosaraju, and Wenda Zhou. Competitive programming with large reasoning models, 2025. URL https://arxiv.org/ abs/2502.06807.

Guilherme Penedo, Anton Lozhkov, Hynek Kydlíˇcek, Loubna Ben Allal, Edward Beeching, Agustín Piqueres Lajarín, Quentin Gallouédec, Nathan Habib, Lewis Tunstall, and Leandro von Werra. Codeforces. https://huggingface.co/datasets/open-r1/codeforces, 2025.

Yun Peng, Jun Wan, Yichen Li, and Xiaoxue Ren. Coffe: A code efficiency benchmark for code generation, 2025. URL https://arxiv.org/abs/2502.02827.

Z. Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. Deepseek-prover-v2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv: 2504.21801, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, et al. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585, 2023.

Wenhan Wang, Chenyuan Yang, Zhijie Wang, Yuheng Huang, Zhaoyang Chu, Da Song, Lingming Zhang, An Ran Chen, and Lei Ma. Testeval: Benchmarking large language models for test case generation, 2025a. URL https://arxiv.org/abs/2406.04531.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, et al. Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025b.

Yuxiang Wei, Federico Cassano, Jiawei Liu, Yifeng Ding, Naman Jain, Zachary Mueller, Harm de Vries, Leandro Von Werra, Arjun Guha, and Lingming Zhang. Selfcodealign: Self-alignment for code generation. arXiv preprint arXiv:2410.24198, 2024.

Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding, 2025. URL https://arxiv.org/abs/ 2503.02951.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115,

- 2024.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. arXiv preprint arXiv: 2502.03387, 2025.

Zhiqiang Yuan, Yiling Lou, Mingwei Liu, Shiji Ding, Kaixin Wang, Yixuan Chen, and Xin Peng. No more manual tests? evaluating and improving chatgpt for unit test generation, 2024. URL https://arxiv.org/abs/2305.04207.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, and Wenhu Chen. Acecoder: Acing coder rl via automated test-case synthesis, 2025a. URL https://arxiv.org/abs/2502. 01718.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv: 2503.18892, 2025b.

Kexun Zhang, Danqing Wang, Jingtao Xia, William Yang Wang, and Lei Li. Algo: Synthesizing algorithmic programs with llm-generated oracle verifiers, 2023. URL https://arxiv.org/abs/ 2305.14591.

Quanjun Zhang, Ye Shang, Chunrong Fang, Siqi Gu, Jianyi Zhou, and Zhenyu Chen. Testbench: Evaluating class-level test case generation capability of large language models, 2024. URL https://arxiv.org/abs/2409.17561.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403.13372.

## A Appendix

### A.1 More Related Work on Synthetic Test Quality and its Implications

Although existing LLM test synthesis methods prove to be useful in many scenarios, such as improving the quality of synthetic data (Wei et al., 2024) and software engineering(Mündler et al., 2025; Jain et al., 2024), their quality is far from perfect (Yuan et al., 2024) and are bounded in complexity, because direct generations of complicated data structures often result in inconsistency (Zhang et al., 2023). Weak verifiers can harm downstream code generation and search performance (Light et al., 2025). The quality of those synthetic tests and their implications are less discussed. Existing benchmarks for LLM test case generation abilities focus on code coverage and/or mutation scores (Wang et al., 2025a; Zhang et al., 2024; Jain et al., 2025a, 2024), the success rate for reproducing issues (Mündler et al., 2025), and the code change coverage for generated code patches (Ahmed et al., 2024; Mündler et al., 2025).

### A.2 Details of the Test Cases Generation Pipeline HARDTESTGEN

As we mentioned in Section 3.2, HARDTESTGEN constructs both the input generator functions and the validator functions for verifying input correctness. In this section, we first introduce the detailed HARDTESTGEN implementation, including the coding problem filtering process, and detailed prompts for input generator/validator synthesis (Section A.2.1), followed by detailed dataset statistics for the final HARDTESTS dataset (Section A.2.2) and some examples in HARDTESTS (Section A.2.3).

### A.2.1 HARDTESTGEN Implementation

Coding problem filtering. Before generating test cases, we first filter out questions not suitable for our test case generation. For example, those without oracle code solutions, and the questions that do not use standard I/O for input and output. More specifically, our question filtering process is as follows: We first remove problems that do not have any oracle programs. Next, we exclude all problems where the starter_code field is non-empty, as they are so-called “core logic” problems, rather than “input-output” style problems, and typically originate from online judges like LeetCode and GeeksforGeeks. In such problems, the programmer is not responsible for handling input and output logic, but only for implementing the core function based on a given function signature. Since the inputs and outputs in these problems are often not strings, they are difficult to use for test case generation. After the filtering, we are left with 32.5k unique coding problems.

Input validator prompt. We use the following LLM prompt to generate an input validator function, and a special judge function when necessary. This prompt includes the problem specification and the oracle program to help the LLM have a better understanding.

- 1 I have a competitive programming problem. To test the correctness of candidate programs,

→ I need to create many test cases.

- 2
- 3 Each test case is an input-output pair. The input part will be fully provided as stdin to the candidate program, and then the candidate output will be collected from stdout. In most cases, we determine the correctness of the program by comparing the candidate output with the output part of the test case (i.e., the reference output), while sometimes, we need to use a custom function to judge the correctness of the candidate output, instead.

→ → → → →

- 4
- 5 Note: Sometimes, a problem may require a single test case to contain multiple sub-tasks. For example: the first line of the input contains an integer $t\ (1 \leq t \leq 1000)$, followed by inputs of $t$ independent sub-tasks. The problem statement may sometimes refer to a sub-task as a "test case", but this is merely a difference in terminology.

→ → → →

- 6
- 7 # Input Validator
- 8

- 9 Suppose I have already written some input generator functions, and used them to generate many test case inputs. However, since they are randomly generated, they may not fully adhere to the constraints specified in the problem. In order to filter out invalid test cases, I need you to write a function `validate_input(input_str: str) -> bool`. Its input is the complete input string of a test case, and it returns a boolean indicating whether the input is valid.

→ → → → →

- 10
- 11 You should first clearly list all the constraints given in the problem statement, and then generate the function. Your function should check as many of the given constraints as possible, including the complex ones. However, if a constraint cannot be verified within a reasonable time complexity (e.g., $O(n)$ for $n \leq 10^6$, or $O(n^2)$ for $n \leq 10^3$), or if it makes the code too complex, then it can be skipped.

→ → → → →

- 12
- 13 **Pay close attention**: If the problem says "It's guaranteed that...", then what follows is precisely something that must be verified. This is because the so-called "guarantee" in the problem is typically enforced through the Input Validator, so you must validate it in `validate_input`. Of course, only if it can be done in reasonable time complexity.

→ → → →

- 14
- 15 **Example 1**: Cicasso has $n$ sticks of lengths $l = (l_0, l_1, \dots, l_{n-1})$. But these $n$ sticks cannot form a convex polygon with non-zero area. You need to add one stick so that the resulting $n+1$ sticks can form such a polygon. The input consists of two lines: the first line is an integer $n$ ($3 \leq n \leq 10^5$). The second line has $n$ integers $l_i$ ($1 \leq l_i \leq 10^9$).

→ → → →

- 16
- 17 The `validate_input` function should not only check that $n$ and $l_i$ are within the correct range and that there are exactly $n$ numbers in the second line, but also check that the $n$ sticks cannot form a convex polygon with non-zero area, i.e., that the longest stick is greater than or equal to the sum of the rest.

→ → →

- 18
- 19 **Example 2**: Suppose there is a permutation $p = (p_0, p_1, \dots, p_{n-1})$ of numbers from 1 to $n$ ($1 \leq n \leq 2 \times 10^5$). But you do not know the permutation $p$. Instead, you are given an array $s = (s_0, s_1, \dots, s_{n-1})$, where $s_i$ is the sum of all $p_j < p_i$ for $j < i$. Your task is to recover $p_i$.

→ → →

- 20
- 21 In theory, we should verify whether the $s_i$ values correspond to a valid permutation $p_i$, but that requires solving for $p_i$, which is too complex. Moreover, when generating inputs, it's quite easy to ensure that the $s_i$ comes from a valid permutation, so mistakes are unlikely. (Note: If verifying a constraint isn't too complex, you should still check it.) Therefore, we only need to check that $n$ is within the range and that $s$ has exactly $n$ elements.

→ → → → →

- 22
- 23 # Output Judging Function
- 24
- 25 In most cases, we can determine whether the candidate program has passed the test case by comparing the `candidate_output` and `reference_output` as strings. The specific function is shown below.

→ →

- 26
- 27 ```python
- 28 def output_judging_function(input_str: str, candidate_output: str, reference_output:

→ str) -> bool:

- 29 normalized_candidate_output = '\n'.join(line.rstrip() for line in

→ candidate_output.rstrip().splitlines())

- 30 normalized_reference_output = '\n'.join(line.rstrip() for line in

→ reference_output.rstrip().splitlines())

- 31 return normalized_candidate_output == normalized_reference_output
- 32 ```
- 33
- 34 However, for a few problems, the above `output_judging_function` does not work.
- 35
- 36 **Example 1**: The problem asks to output a list (`List[int]`), but the order of elements

→ in the list does not matter.

- 37

- 38 In this case, we should convert both `candidate_output: str` and `reference_output: str`

→ into `List[int]`, sort them, and then compare them.

- 39
- 40 **Example 2**: Given a graph with both directed and undirected edges, you must make all undirected edges directed so that the resulting graph has no cycles. If it is possible, output "YES" and the resulting graph (list of directed edges), otherwise output "NO".

→ → →

- 41
- 42 Here, in `output_judging_function`, we should first determine from `reference_output` whether a solution is possible. If both `candidate_output` and `reference_output` say "YES", then we should also validate whether the graph provided in `candidate_output` is valid: check whether all edges exist in the input and whether the graph is acyclic (e.g., via DFS).

→ → → →

- 43
- 44 **Example 3**: There are a total of $T$ sub-tasks. Each sub-task gives a pair of integers $l, r$ ($1 \leq l \leq r \leq 998244353$), and the goal is to find a pair of integers $x, y$ such that $l \leq x, y \leq r$, $x \ne y$, and $y$ is divisible by $x$. It is guaranteed that every sub-task has a valid solution.

→ → →

- 45
- 46 For each pair $x, y$ provided in the `candidate_output`, simply check whether they satisfy all the conditions mentioned in the problem statement. The `output_judging_function` for this problem does not need to use the `reference_output`; it only requires the `input_str`.

→ → →

- 47
- 48 You need to first analyze whether this particular problem requires a custom `output_judging_function` (different from the one given above). If yes, generate a custom `output_judging_function`. If not, don't output it. Sometimes only `input_str` is needed and `reference_output` is not required; other times only `reference_output` is needed and `input_str` is not required; and in some cases, both are needed. However, regardless of which ones are actually used, the function signature must always be: `output_judging_function(input_str: str, candidate_output: str, reference_output: str) -> bool`.

→ → → → → → →

- 49
- 50 Generally speaking, if a problem states "there are multiple possible answers, any one is acceptable," this implies that the problem requires a custom Output Judging Function. However, even if this is not explicitly mentioned, the problem may still actually require a custom Output Judging Function. You need to determine this yourself.

→ → →

- 51
- 52 ---
- 53
- 54 Also, when generating the above two functions, some known tricks or conclusions may be helpful, and you should derive them yourself if needed. I will give you the correct solution to the problem, and you can use it to derive certain conclusions or tricks.

→ →

- 55
- 56 Your output format must strictly follow:
- 57
- 58 # Analysis
- 59
- 60 ... (Analyze the problem, constraints, how to generate the Input Validator and Output

→ Judging Function, etc.)

- 61
- 62 # Result
- 63
- 64 ```json
- 65 {
- 66 "input_validator": "A block of Python code containing the `validate_input`

→ function. No other content.",

- 67 "needs_custom_output_judging_function": true

|or|
|---|

false,

- 68 "output_judging_function": "A block of Python code containing the `output_judging_function` function. No other content."

|or|
|---|

→ null

- 69 }
- 70 ```
- 71
- 72 ---
- 73

- 74
- 75 Note:
- 76 * All your code should be in Python 3.
- 77 * Do not wrap the Python code in ```python```, just provide it plainly.
- 78 * The Python code block under each field should be independent. In other words, they should not call or reference each other. If one block imports a library, other blocks must re-import it as needed.

→ →

- 79 * In a Python block, you should first import the necessary libraries, and then start

→ defining functions. Important: Do not place import statements inside the functions.

- 80 * Only Python's built-in libraries are permitted for import.
- 81
- 82 For example, a block of Python code for Input Validator should look like this:
- 83
- 84 import ... (some modules)
- 85
- 86 def input_validator(input_str: str) -> bool:
- 87 ... (some code)
- 88
- 89 A block of Python code for Output Judging Function (if needed) should look like this:
- 90
- 91 import ... (some modules)
- 92
- 93 def output_judging_function(input_str: str, candidate_output: str, reference_output:

→ str) -> bool:

- 94 ... (some code)
- 95
- 96 ---
- 97
- 98 # Problem Statement
- 99
- 100 {{ problem_specification }}
- 101
- 102 ---
- 103
- 104 # Correct Program
- 105
- 106 {{ oracle_program }}
- 107

Input generator prompt. We use the following prompt to have the LLM generate inputs directly (Type 1), a regular input generator (Type 2), and a hacking input generator (Type 3). This prompt makes use of the problem specification, oracle program, and input validator to help the LLM better understand the problem requirements.

- 1 I have a competitive programming problem. To test candidate programs' correctness, I need

→ to create many test cases.

- 2
- 3 Each test case is an input-output pair. The input part will be fully provided as stdin to the candidate program, and then the candidate output will be collected from stdout. In most cases, we determine the correctness of the program by comparing the candidate output with the output part of the test case (i.e., the reference output), while sometimes, we need to use a custom function to judge the correctness of the candidate output, instead.

→ → → → →

- 4
- 5 Note: Sometimes, a problem may require a single test case to contain multiple sub-tasks. For example: the first line of the input contains an integer $t\ (1 \leq t \leq 1000)$, followed by inputs of $t$ independent sub-tasks. The problem statement may sometimes refer to a sub-task as a "test case", but this is merely a difference in terminology.

→ → → →

- 6
- 7 Since the output part can be obtained by running correct programs, I only need you to

→ help me generate the input part.

- 8

- 9 The input should comply with the constraints given in the problem statement. I will give you an Input Validator that checks whether the input meets all the constraints specified in the problem statement. However, some constraints may not be checked by the Input Validator due to the difficulty of verification. Nevertheless, the input you generate should still comply with all of these constraints.

→ → → →

- 10
- 11 # Directly Generated Input
- 12
- 13 Directly generated input (DGI) refers to inputs of small size and scale that can be directly generated, such as the input parts of the sample test cases given in the problem statement. You can get more DGIs by making minor modifications to these inputs. I need you to directly generate {{ num_DGI }} DGIs. Note: each DGI's length should be similar to the sample test cases' input, comply with the constraints given in the problem, and must not exceed 300 characters under any circumstances. If it is not possible to generate DGIs under this length limit, give up on generating them.

→ → → → → →

- 14
- 15 # Regular Input
- 16
- 17 ## Regular Problems
- 18
- 19 Regular Input (RI) refers to ordinary inputs, where all data is fully random and satisfies the constraints specified by the problem statement. You need to write a function `gen_regular_input() -> str`, and each time it is called, it should generate one random RI. You should ensure the generated input satisfies the constraints as much as possible, and may even sacrifice some degree of randomness to do so. But if trying to enforce a constraint leads to a function that cannot run within finite and reasonable time complexity (e.g., $O(n)$ for $n \leq 10^6$, or $O(n^2)$ for $n \leq 10^3$), then you may ignore that constraint.

→ → → → → → →

- 20
- 21 **Pay close attention**: do not use `while` loops, especially ones that "keep generating until a constraint is satisfied." That can cause unlimited running time and make input generation fail.

→ →

- 22
- 23 Some problems may require certain test cases to satisfy specific constraints (for example, 10% of test cases satisfy $n \leq 100$, 10% of the test cases satisfy $n \leq 1000$, etc.). Ignore this requirement. All test cases should be generated according to the most general constraints.

→ → →

- 24
- 25 Sometimes, generating input that satisfies the constraints requires some trick. You need to deduce it yourself (e.g., the example below about when $n$ sticks cannot form a convex polygon). I will give you the correct solution for the problem, and you can analyze it to discover some tricks or conclusions.

→ → →

- 26
- 27 **Example 1**: Cicasso has $n$ sticks ($3 \leq n \leq 10^5$) of lengths $l_i$ ($1 \leq l_i \leq 10^9$, for $i=0,1,\dots,n-1$). But these $n$ sticks cannot form a convex polygon of non-zero area. You need to add one more stick, so that the $n+1$ sticks can form a convex polygon of non-zero area. Output the minimum length of the additional stick.

→ → →

- 28
- 29 We can randomly generate $n \in [3, 10^5]$, but cannot randomly generate $l_i$, because such $l_i$ will likely not satisfy the constraint that the $n$ sticks cannot form a convex polygon of non-zero area. (It's not feasible to randomly generate and then filter, since it's too time-consuming.) We know that this constraint actually requires "the maximum $l_i$ is greater than or equal to the sum of all the other $l_i$." So we can first randomly sample a $l_0$ in $[n-1, 10^9]$ as the maximum $l_i$, then sample an integer $s \in [n-1, l_0]$ as the total sum of the other $l_1, \dots, l_{n-1}$, and finally use a partitioning trick to sample $l_1, \dots, l_{n-1}$ such that each element is at least 1 and the total sum is $s$. After that, we can shuffle the $l_i$ list.

→ → → → → → → → →

- 30
- 31 **Example 2**: There is a permutation $p = (p_0, p_1, ..., p_{n-1})$ of numbers from 1 to $n$ ($1 \leq n \leq 2\cdot 10^5$). You do not know this permutation, but you are given an array $s = (s_0,\dots,s_{n-1})$, where $s_i$ is the sum of all $p_j < p_i$ with $j < i$. Find $p_i$.

→ → →

- 32

- 33 We can first randomly generate $n \in [1, 2 \times 10^5]$. But we cannot directly generate an array $s_i$ randomly, because it is very unlikely to satisfy the constraints. Instead, we should reverse the process: first generate a random permutation $p_i$, and then compute the corresponding $s_i$.

→ → →

- 34
- 35 **Example 3**: This problem has $t \in [1, 1000]$ groups of independent sub-tasks. Each sub-task has an integer $n \in [1, 10^5]$ and an array $a$ of length $n$, where $a_i \in [1,10^5]$. The problem guarantees that the total sum of all $n$ across all $t$ sub-tasks does not exceed $2 \times 10^5$.

→ → →

- 36
- 37 We can first randomly generate $t \in [1, 1000]$. But at this point we cannot directly sample $t$ values of $n$ from $[1, 10^5]$, because their sum is likely to exceed $2 \times 10^5$. So instead, we randomly sample $s \in [t, 2\times 10^5]$, and then partition $s$ into $n_0, n_1, \dots, n_{t-1}$ such that each value is at least 1 and their sum is $s$.

→ → → →

- 38
- 39 The following Python function demonstrates how, given positive integers $m$ and $s$, with $m \leq s$, one can randomly select $m$ positive integers such that their sum equals $s$. This is just for your reference.

→ →

- 40
- 41 import random
- 42
- 43 assert m <= s
- 44 if m >= 2:
- 45 breaks = random.sample(range(1, s), m - 1)
- 46 breaks.sort()
- 47 results = [breaks[0]] + [breaks[i] - breaks[i - 1] for i in range(1,

→ len(breaks))] + [s - breaks[-1]]

- 48 else:
- 49 results = [s]
- 50
- 51 ## Multi-Category Output Problems
- 52
- 53 For most problems, there is only one type of output. But there are some problems where outputs fall into multiple categories. These are called Multi-Category Output Problems. For example, some problems require the output to be "Yes" or "No", while others ask you to output the solution if it exists, otherwise output -1. In such cases, if we treat it as a regular problem and only write a single `gen_regular_input` function to generate inputs randomly, the resulting outputs will be very imbalanced. For example, the "Yes" outputs may require special construction, so nearly all generated inputs produce "No" as the answer. Thus, even a candidate program that always prints "No" would pass all test cases.

→ → → → → → → →

- 54
- 55 For such problems, instead of one `gen_regular_input`, you need to design a series of functions `gen_regular_input_suffix`, where each function is responsible for generating inputs corresponding to one category of output. Each time a function is called, it should be able to generate--within reasonable time complexity--one random input that satisfies the constraints and whose corresponding output belongs to the corresponding category. If it is difficult to write a function that randomly generates some category, you can:

→ → → → → →

- 56
- 57 1. Sacrifice randomness and perform special construction, even returning a fixed value

→ each time

- 58 or 2. Construct completely random data, similar to `gen_regular_input`
- 59
- 60 Sometimes, a problem may require a single test case to contain multiple independent sub-tasks. In this case, each sub-task in each input generated by `gen_regular_input_suffix` should have the corresponding output category, e.g., all corresponding outputs should be "No".

→ → →

- 61
- 62 **Example 1**: Given two $n \times m$ binary matrices $A, B$. You can take the following operation: select a rectangle in matrix $A$ with height and width both at least 2, and flip the values at the four corner positions. You are to answer whether it's possible to make $A$ equal to $B$ using this operation. If possible, output "Yes" and the resulting matrix; otherwise, output "No".

→ → → →

- 63
- 64 There are two outputs here: "Yes" and "No", corresponding to two categories of inputs. For the first category, we create `gen_regular_input_yes`, such that $A$ can be transformed into $B$. We can randomly construct matrix $A$, then perform $t$ operations (you can decide $t$ yourself, but it should not be too small or too large to avoid long generation time), where each operation selects a rectangle and flips the corners. Then the result becomes matrix $B$. For the second category, we write `gen_regular_input_no`, where $A$ cannot be transformed into $B$. One way is to randomly flip a position in matrix $B$ from the previous construction, which makes it impossible. This sacrifices randomness, but is simple and acceptable.

→ → → → → → → →

- 65
- 66 **Example 2**: Given two numbers $n, m$ ($1\leq n \leq m\leq 5\times10^8$), you are to determine whether it is possible to transform $n$ into $m$ by multiplying by 2 and 3, and if so, output the minimum number of operations. Otherwise, output -1.

→ →

- 67
- 68 There are two outputs: the minimum operation count, and -1. Correspondingly, we have two input generators. For the first case, where $n$ can be transformed into $m$, we can randomly generate $n\in [1, 5\times 10^8]$, then perform $t$ operations (multiply by 2 or 3) until $t$ steps are complete or further multiplication would exceed $5\times10^8$. The result becomes $m$. For the second case, where $n$ cannot be transformed into $m$, we can firstly randomly generate $m > n$, and then if $n$ can be transformed into $m$, simply set $m = m-1$.

→ → → → → →

- 69
- 70 **Example 3**: Player A and B are playing tic-tac-toe. Player A goes first. You are given a $3 \times 3$ board, where each cell is ".", "X", or "0". Output the current state, one of: "first" (next move is A), "second" (next is B), "illegal" (not possible in a legal game), "the first player won", "the second player won", or "draw".

→ → →

- 71
- 72 There are 6 output categories, corresponding to 6 input categories. For the first output category, we need to create `gen_regular_input_first` where the next move is A's. We can randomly select $t\in[0, 4]$, then randomly place $t$ X's and $t$ 0's. This may lead to a win or illegal state, but we should NOT filter those during generation, because doing so would make the code too complex and slow. We only need most of the generated inputs to match this category. For the second category, place $t+1$ X's and $t$ 0's ($t\in [1,3]$). For the third category, it must be illegal, e.g. X and 0 count difference is too large, or both players have already won. We can create `gen_regular_input_illegal_mark_num` and `gen_regular_input_illegal_both_win`, etc. Do the same for the remaining categories.

→ → → → → → → → →

- 73
- 74 # Hacking Input
- 75
- 76 Although Regular Input can guarantee large data size (because most of the time, the magnitude of random data is close to the maximum magnitude), for some problems, large-scale random data is not enough. We also need Hacking Input (HI). HI refers to inputs that are very tricky for candidate programs. Specifically, I need you to generate a series of functions `gen_hacking_input_suffix() -> str`. Each function is responsible for one type of HI. Every time each function is called, it should return one HI (can be random or fixed).

→ → → → → →

- 77
- 78 Most types of HI are designed to cause brute-force candidate programs with insufficient optimization to run into time limits. Specifically, you should first list what kinds of straightforward or brute-force algorithms candidate programs might use, then construct inputs that would cause them to time out. Note: for most problems, RI data is enough to cause brute-force solutions to TLE, so you don't need to generate more. But for some problems, even though the brute-force algorithm's worst-case complexity is $O(n^2)$, due to rare worst-case inputs, the actual runtime is closer to $O(n)$. In these cases, you need to specially construct the data to repeatedly trigger the worst-case scenario for those brute-force algorithms.

→ → → → → → → →

- 79
- 80 For some problems, we also need some types of HI to expose bugs caused by failure to handle edge cases. So you should think about whether there are any special edge cases (e.g., input $n=0$, or tree root is None, etc.). Note that the randomness of the input data itself at this time is not important. The key point is to expose the errors of the candidate programs.

→ → → →

- 81

- 82 Of course, if the problem doesn't require any HI, then do not generate them. Especially if an HI is simply large-scale data, then you shouldn't bother. HI must be specially constructed--random RI should almost never produce them.

→ →

- 83
- 84 **Example 1**: Given two numbers $n$ and $m$ ($1 \leq n \leq m \leq 5 \times 10^8$), the task is to determine whether it is possible to transform $n$ into $m$ by repeatedly multiplying $n$ by 2 or by 3. If possible, output the minimum number of operations required; otherwise, output -1.

→ → →

- 85
- 86 A brute-force approach that a candidate program might take is to use DFS, recursively trying to multiply $n$ by 2 or 3 until it becomes greater than or equal to $m$. If we randomly choose $n$ and $m$, the ratio between them is usually small, so this approach might still pass. One kind of effective HI is to set $n \in [1, 5]$ and $m \in [4 \times 10^8, 5 \times 10^8]$. This creates a large gap between $n$ and $m$, making the brute-force DFS approach inefficient. We can name the corresponding function `gen_hacking_input_small_n_big_m`. You should consider other types of HIs yourself.

→ → → → → → →

- 87
- 88 **Example 2**: Given a string $S$ of length $n \in [1, 10^5]$, we repeatedly perform the following operation: find two identical adjacent characters and delete them. This continues until there are no more identical adjacent characters in $S$.

→ →

- 89
- 90 This problem should be solved using a stack to achieve an $O(n)$ time complexity. However, some candidate programs might use a brute-force simulation approach -repeatedly scanning the string and removing adjacent equal characters -- which can result in a worst-case time complexity of $O(n^2)$. If we generate $S$ completely at random, it's likely that there will only be a few pairs of identical adjacent characters. One kind of HI is to construct a string $S$ of a long even length (e.g., in $[5 \times 10^4, 10^5]$) and set `S[2*k] == S[2*k+1]`, thereby introducing a large number of adjacent equal character pairs. However, if the candidate program deletes all adjacent equal pairs in each round, the time complexity remains $O(n)$. Another HI is to construct a string $S$ of a long even length (e.g., in $[5 \times 10^4, 10^5]$) such that `S[:n//2] == S[n//2:][::-1]`, which forces the program to go through $n$ rounds to completely remove all characters, resulting in the true worst-case time complexity of $O(n^2)`. These two functions can be named `gen_hacking_input_pairwise_equal` and `gen_hacking_input_mirrored_halves`, respectively.

→ → → → → → → → → → → → → →

- 91
- 92 **Example 3**: Given integer $w\in[1, 100]$, determine whether it can be written as the

→ sum of two positive even integers.

- 93
- 94 Candidate programs may output "Yes" when $w$ is even, and "No" when $w$ is odd. But a special case is $w=2$, which should be "No". So we can create `gen_hacking_input_two`, which always returns the string `"2"`.

→ →

- 95
- 96 Important: if a type of HI is just setting data to their largest scale, then it is

→ unnecessary.

- 97
- 98 ---
- 99
- 100 Your output format must strictly be
- 101
- 102 # Analysis
- 103
- 104 ...
- 105 (generally, you should first analyze the problem and data constraints, and then analyze how to generate Directly Generated Input, how to generate Regular Input, and whether the problem is a Multi-Category Output Problem (In that case, generate regular input generation functions for each output category. Make sure you mentioned the corresponding function names in the Analysis part). Then you should list some naive candidate programs and analyze how to generate Hacking Input.)

→ → → → →

- 106
- 107 # Result
- 108
- 109 ```json

- 110 {
- 111 "directly_generated_inputs": ["DGI1", "DGI2",

|...|
|---|

],

- 112 "is_multi_category_output_problem": true

|or|
|---|

false,

- 113 "regular_input_generator": "a block of Python code containing a function gen_regular_input (for Regular Problem), or multiple functions gen_regular_input_suffix (for Multi-Category Output Problem)",

→ →

- 114 "hacking_input_generator": "a block of Python code containing multiple gen_hacking_input_suffix functions"

|or|
|---|

null

|(i|
|---|

f no

|Hacki|
|---|

ng In

|pu|
|---|

t

|is|
|---|

nee

|ded)|
|---|

→

- 115 }
- 116 ```
- 117
- 118 ---
- 119
- 120 Note:
- 121 * All your code should be in Python 3.
- 122 * Do not wrap the Python code in ```python```, just provide it plainly.
- 123 * The Python code block under each field should be independent. In other words, they should not call or reference each other. If one block imports a library, other blocks must re-import it as needed.

→ →

- 124 * In a Python block, you should first import the necessary libraries, and then start

→ defining functions. Important: Do not place import statements inside the functions.

- 125 * Only Python's built-in libraries are permitted for import.
- 126
- 127 For example, a block of Python code for RI of Regular Problems should look like this:
- 128
- 129 import ... (some modules)
- 130
- 131 def gen_regular_input(input_str: str) -> bool:
- 132 ... (some code)
- 133
- 134 A block of Python code for RI of Multi-Category Output Problem may look like this:
- 135
- 136 import ... (some modules)
- 137
- 138 def gen_regular_input_some_suffix(input_str: str) -> bool:
- 139 ... (some code)
- 140
- 141 def gen_regular_input_some_suffix(input_str: str) -> bool:
- 142 ... (some code)
- 143
- 144 ...
- 145
- 146 And the Hacking Input block is similar.
- 147
- 148 ---
- 149
- 150 # Problem Statement
- 151
- 152 {{ problem_specification }}
- 153
- 154 ---
- 155
- 156 # Correct Program
- 157
- 158 {{ oracle_program }}
- 159
- 160 ---
- 161
- 162 # Input Validator
- 163
- 164 {{ input_validator }}
- 165
- 166

Type 1 (Directly Generated)

25000

20000

15000

Frequency

Frequency

10000

5000

0

0 2 4 6 8 10

Number of Test Cases

Type 3 (Hacking)

14000

12000

10000

Frequency

Frequency

8000

6000

4000

2000

0

0 5 10 15 20 25 30 35 40

Number of Test Cases

Type 2 (Regular)

20000

17500

15000

12500

10000

7500

5000

2500

0

0 5 10 15 20 25 30

Number of Test Cases

Total

6000

5000

4000

3000

2000

1000

0

0 10 20 30 40 50 60

Number of Test Cases

Figure 4: The distribution of the number of Type1, Type2, and Type3 test cases, as well as the total number of test cases in HARDTESTS.

Note that in the prompts above, we provide two to three carefully crafted examples for each function that we ask the LLM to generate, enabling in-context learning. Additionally, we prompt the LLM to perform chain-of-thought reasoning. These two requirements help the LLM understand the task better and improve the data synthesis.

- A.2.2 HARDTESTS Statistics

We generated test cases for all 32.5k valid questions in the HARDTESTS. The status distribution of test case generation is shown in Figure 5. While we carefully designed the test-case generation prompt, we didn’t attain 100% coverage. We successfully generated test cases for 81.9% of the questions. The main failure reasons include: no valid oracle programs (i.e., compiles and runs without errors) (6.62%), all output verification failed (5.85%), and input generation failed (3.72%). The distribution of the number of Type1, Type2, and Type3 test cases, as well as the total number of test cases, is shown in Figure 4.

- A.2.3 HARDTESTS Examples

Example 1

This example demonstrates the input validator, Type 1 (Directly Generated) and Type 2 (Regular) test cases, as well as a custom judging function. Here’s the problem description:

Codeforces 1096A: There are a total of T (1 ≤ T ≤ 1000) sub-tasks. Each sub-task gives a pair of integers l,r (1 ≤ l ≤ r ≤ 998244353), and the goal is to find a pair of integers x,y such that l ≤ x,y ≤ r, x ̸= y, and y is divisible by x. It is guaranteed that every sub-task has a valid solution.

Note: It can be mathematically proven that a sub-task has a solution if and only if 2l < r.

[Figure 1]

Figure 5: The result status distribution of our test case generation pipeline HARDTESTGEN.

The input validator is as follows. It checks whether input_str conforms to the required format specified in the problem specification, whether all data falls within the required ranges, and whether other constraints are satisfied (e.g., whether each sub-task has a solution).

- 1 import sys
- 2
- 3 def input_validator(input_str: str) -> bool:
- 4 lines = input_str.strip().split('\n')
- 5 if not lines:
- 6 return False
- 7
- 8 try:
- 9 T = int(lines[0])
- 10 except:
- 11 return False
- 12
- 13 if not (1 <= T <= 1000):
- 14 return False
- 15
- 16 if len(lines) != T + 1:
- 17 return False
- 18
- 19 for i in range(1, T + 1):
- 20 parts = lines[i].strip().split()
- 21 if len(parts) != 2:
- 22 return False
- 23 try:
- 24 l, r = map(int, parts)
- 25 except:
- 26 return False
- 27
- 28 if not (1 <= l <= r <= 998244353):
- 29 return False
- 30
- 31 if 2 * l > r:
- 32 return False # No valid pair possible
- 33
- 34 return True

Since this problem allows multiple correct solutions, simple string comparison is not sufficient. We need a special, customized output judging function. The output judging function is as follows.

- 1 def output_judging_function(input_str: str, candidate_output: str, reference_output:

→ str) -> bool:

- 2 try:
- 3 input_lines = input_str.strip().split('\n')
- 4 T = int(input_lines[0])
- 5 queries = [tuple(map(int, line.strip().split())) for line in input_lines[1:T+1]]
- 6
- 7 output_lines = candidate_output.strip().split('\n')
- 8 if len(output_lines) != T:
- 9 return False
- 10
- 11 for (l, r), line in zip(queries, output_lines):
- 12 parts = line.strip().split()
- 13 if len(parts) != 2:
- 14 return False
- 15 x, y = map(int, parts)
- 16 if not (l <= x <= r and l <= y <= r):
- 17 return False
- 18 if x == y:
- 19 return False
- 20 if y % x != 0:
- 21 return False
- 22
- 23 return True
- 24 except:
- 25 return False The Type1 (Directly Generated) inputs are as follows.

- 1 [
- 2 "3\n1 10\n2 8\n3 10",
- 3 "2\n5 20\n10 25",
- 4 "3\n7 30\n1 5\n2 6",
- 5 "1\n100 300",
- 6 "2\n999 2000\n1000 3000"
- 7 ]

The Type 2 input (Regular) generator is as follows. To ensure a solution always exists, the LLM sets r ≥ 2l.

- 1 import random
- 2
- 3 def gen_regular_input() -> str:
- 4 T = random.randint(1, 1000)
- 5 queries = []
- 6 for _ in range(T):
- 7 l = random.randint(1, 499122176)
- 8 r = random.randint(2 * l, 998244353)
- 9 queries.append(f"{l} {r}")
- 10 return f"{T}\n" + "\n".join(queries)

The LLM believes that there is no need to generate a Type 3 (Hacking) input generator for this problem.

### Example 2

This example demonstrates the input validator, as well as the Type 1 (Directly Generated), Type 2 (Regular), and Type 3 (Hacking) test cases. Here’s the problem description:

Codeforces 1141 A: Given two numbers n,m (1 ≤ n ≤ m ≤ 5 × 108), you are to determine whether it is possible to transform n into m by multiplying by 2 and 3, and if so, output the minimum number of operations. Otherwise, output -1.

The input validator is as follows. It checks whether input_str conforms to various format requirements and constraints.

- 1 import re
- 2
- 3 def validate_input(input_str: str) -> bool:
- 4 try:
- 5 # Split the input string into two parts
- 6 parts = input_str.strip().split()
- 7 if len(parts) != 2:
- 8 return False
- 9 # Convert parts to integers
- 10 n, m = map(int, parts)
- 11 # Check the constraints
- 12 if not (1 <= n <= m <= 5 * 10**8):
- 13 return False
- 14 return True
- 15 except:
- 16 return False The Type1 (Directly Generated) inputs are as follows.

- 1 [
- 2 "120 51840",
- 3 "42 42",
- 4 "48 72",
- 5 "1 1",
- 6 "2 6",
- 7 "3 9",
- 8 "5 10",
- 9 "7 21",
- 10 "8 24",
- 11 "10 30"
- 12 ]

- The Type 2 input (Regular) generator is as follows. The output of this problem has two categories (i.e., possible and impossible), so the LLM generates two regular input generating functions, corresponding to these two categories respectively.

- 1 import random
- 2
- 3 def gen_regular_input_possible() -> str:
- 4 n = random.randint(1, 10**8)
- 5 m = n
- 6 for _ in range(random.randint(1, 20)):
- 7 if random.choice([True, False]):
- 8 m *= 2
- 9 else:
- 10 m *= 3
- 11 if m > 5 * 10**8:
- 12 break
- 13 return f"{n} {m}"
- 14
- 15 def gen_regular_input_impossible() -> str:
- 16 n = random.randint(1, 10**8)
- 17 m = random.randint(n + 1, 5 * 10**8)
- 18 while m % n == 0:
- 19 m += 1
- 20 return f"{n} {m}"

- The Type 3 input (Hacking) generator is as follows. The LLM generates two hacking input generating functions. The first function sets a small n and a large m. This is because a brute-force approach that a candidate program might take is to use DFS, recursively trying to multiply n by 2 or 3 until it becomes greater than or equal to m. If we randomly choose n and m, the ratio between them is usually small, so this approach might still pass. Setting n to be small and m to be big creates a large gap between n and m, making the brute-force DFS approach inefficient. The second function sets m = n, which serves as an edge case.

- 1 import random
- 2
- 3 def gen_hacking_input_small_n_big_m() -> str:
- 4 n = random.randint(1, 5)
- 5 m = random.randint(4 * 10**8, 5 * 10**8)
- 6 return f"{n} {m}"
- 7
- 8 def gen_hacking_input_edge_case() -> str:
- 9 n = random.randint(1, 5 * 10**8)
- 10 return f"{n} {n}"

For this problem, the LLM believes that a string comparison function would be enough for output judging.

### A.3 Details of the Collection of Problem Specifications and Oracle Programs in HARDTESTS

HARDTESTS consists of 47,136 coding problems collected from 13 OJs. In practice, the dataset obtains problem specifications and oracle programs from five direct data sources: AtCoder, Codeforces, Luogu, CodeContests, and TACO.

Data sources. Codeforces (https://codeforces.com/) is one of the largest English OJs. We collected all publicly available problem specifications up to September 2024 from Codeforces. AtCoder. (https://atcoder.jp/) is a large OJ offering problems in both Japanese and English. We scraped all problem specifications available up to September 2024, along with three correct user-submitted C++ programs for each problem. We used those directly for problems with official English versions. Luogu (https://www.luogu.com.cn/) is a large Chinese OJ consisting of a main section (Luogu-Main) and four mirror sections. The main section hosts original problems authored by users and administrators, as well as problems sourced from real-world contests (e.g. USACO). The mirror sections contain problems from other OJs, including AtCoder, SPOJ, Codeforces, and UVa. We collected all available problem specifications and community-authored tutorials, which often include both correct C++ programs and corresponding natural language explanations, from Luogu. CodeContests (Li et al., 2022) is a dataset comprising 13,493 problems collected from five OJs. Each entry includes a problem specification and several correct programs in C++, Python 2, Python 3, and Java. Only Codeforces problems in CodeContests were used in our dataset, as only their problem IDs were explicitly provided. TACO (Li et al., 2023) is a large-scale English dataset containing 25.4k problems sourced from ten OJs. Each entry includes a problem specification and multiple correct Python programs. We collect all problems from TACO.

The distribution of problem counts across each OJ is shown in Figure 6. The URLs of each OJ, along with the direct data sources of their problem specifications and oracle programs, are listed in Table 6.

Note that since some problems have multiple oracle program sources, we prioritize programs from more reliable sources when generating test cases. The reliability, supported languages, and notes regarding each direct source of oracle programs are presented in Table 7. The distribution of the number of oracle programs per problem in HARDTESTS is shown in Figure 7.

### A.4 Direct Evaluation Details

Evaluation details for LLM-generated programs on AtCoder. AtCoder previously made its official test cases publicly available. Although this is no longer the case, we obtained a partial archive from the Github repository conlacda/atcoder-testcases. On AtCoder, we use the test cases in TACO as the baselines. We selected problems that have at least one test case in each dataset, resulting in a total of 653 problems.

Evaluation details for LLM-generated programs on Codeforces. Codeforces does not make its test cases publicly available. Therefore, we manually submit LLM-generated candidate programs to the Codeforces platform to obtain ground-truth verdicts. We use TACO and CodeContests as baselines. For problems where the results of all three datasets agree, we randomly select 5% of them for submission. For problems where the datasets produce conflicting results, we submit 50% of the candidate programs. We compute precision and recall based on the combined submission outcomes. For each difficulty level from 1 to 4, we randomly select 150 problems with at least one test case in each dataset, yielding a total of 600 problems.

[Figure 2]

[Figure 3]

Figure 6: Number of problems from each OJs. Figure 7: Distribution of the number of oracle programs in HARDTESTS.

Table 6: Problem specification sources and oracle solution sources of each OJ.

Problem Specification Sources

Oracle Program Sources

OJ URL

TACO, CodeContests, Luogu

Codeforces https://codeforces.com/ Codeforces

AtCoder, TACO, Luogu Luogu https://www.luogu.com.cn/ Luogu Luogu UVa https://onlinejudge.org/ Luogu Luogu SPOJ https://www.spoj.com/ Luogu Luogu Aizu https://onlinejudge.u-aizu.ac.jp/ TACO TACO GeeksforGeeks https://www.geeksforgeeks.org/ TACO TACO Codewars https://www.codewars.com/ TACO TACO Kattis https://open.kattis.com/ TACO TACO CodeChef https://www.codechef.com/ TACO TACO HackerEarth https://www.hackerearth.com/ TACO TACO LeetCode https://leetcode.com/ TACO TACO HackerRank https://www.hackerrank.com/ TACO TACO

AtCoder https://atcoder.jp/contests/ AtCoder

Table 7: Oracle program sources with reliability, languages, and notes

Oracle Program Source Reliability Languages Notes

User-submitted and accepted programs from AtCoder

High Python, C++ Some code (either Python or C++) may use AtCoder’s custom library.

Code solutions from CodeContests

High Python 2/3, C++, Java

—

Community-authored editorials from Luogu

Medium C++ Some editorials may lack complete, directly executable code. But if the code has no compilation or runtime errors, it is very likely to be completely correct.

Verified programs from TACO, i.e., programs that can pass all TACO’s own test cases

Medium Python There’s some false positives in TACO’s test cases.

Other programs from TACO Low Python Reliability is not zero due to some false negatives in TACO’s test cases.

Evaluation details for human-written programs on Codeforces. A dataset at Huggingface titled MatrixStudio/Codeforces-Python-Submissions collects 690k human-submitted programs on Codeforces along with their official verdicts. We use the verdicts as the ground-truth labels. All other settings are the same as those of evaluation using LLM-generated programs.

- A.5 Qualitative Analysis of Generated Tests A.5.1 Example 1: False Positive of TACO and HARDTESTS Type 1

In this example we show how TACO and HARDTESTS Type 1 tests cannot break a wrong program and result in a false positive, while HARDTESTS Type 2 tests succeeds in making the program fail. Here’s the problem description:

AtCoder ABC117C: Given an integer N (2 ≤ N ≤ 2 × 105) and an integer array A of length N (0 ≤ Ai ≤ 109), compute the value of i N=1−1 Nj=i+1 AiAj modulo 109 + 7. Since 2 ≤ N ≤ 2 × 105, the solution to the problem needs to be relatively efficient. The correct solution employs mathematical techniques to simplify the equation into: 12 Ni=1 Ai

2

− Ni=1 A2i ,

which yields an O(N) algorithm. However, a candidate program generated by Qwen2.5-Coder-7B-Instruct uses a brute-force algorithm with a time complexity of O(N2). The candidate program is as follows:

- 1 #include <iostream>
- 2 #include <vector>
- 3
- 4 int main() {
- 5 std::ios::sync_with_stdio(false);
- 6 std::cin.tie(nullptr);
- 7
- 8 int N;
- 9 std::cin >> N;
- 10
- 11 std::vector<long long> A(N);
- 12 for (int i = 0; i < N; ++i) {
- 13 std::cin >> A[i];
- 14 }
- 15
- 16 const int MOD = 1000000007;
- 17 long long sum = 0;
- 18
- 19 for (int i = 0; i < N; ++i) {
- 20 for (int j = i + 1; j < N; ++j) {
- 21 sum = (sum + A[i] * A[j]) % MOD;
- 22 }
- 23 }
- 24
- 25 std::cout << sum << std::endl;
- 26
- 27 return 0;
- 28 }

Due to its inefficiency, this candidate program failed to pass the official test cases. Nevertheless, because the test cases in TACO and HARDTESTS Type 1 (Directly Generated) were relatively small (with small N), the candidate program successfully passed these cases.

Furthermore, the HARDTESTS Type 2 (Regular) input for this problem is generated using the following Python function:

- 1 import random
- 2
- 3 def gen_regular_input() -> str:
- 4 n = random.randint(2, 200000)

- 5 a = [random.randint(0, 1000000000) for _ in range(n)]
- 6 return f"{n}\n" + " ".join(map(str, a))

Due to the larger scale of HARDTESTS Type 2 (Regular) inputs, the candidate program failed to pass these test cases and we have a true negative.

### A.5.2 Example 2: False Positive of TACO and HARDTESTS Type 1 + 2

In this example we show how TACO and HARDTESTS Type 1 + 2 tests cannot break a wrong program and result in a false positive, while HARDTESTS Type 3 tests succeeds in making the program fail. Here’s the problem description:

AtCoder ABC139C: There are N (1 ≤ N ≤ 105) squares arranged from left to right, with the height of the i-th square from the left being Hi (1 ≤ Hi ≤ 109). Starting from any square, you can move one step to the right as long as the next square’s height is not greater than the current one. Find the maximum number of moves possible.

Given that 1 ≤ N ≤ 105, the solution needs to be relatively efficient. The correct solution uses an O(N) greedy algorithm. Specifically, it traverses the array from left to right, counting how many consecutive heights satisfy Hi ≥ Hi+1. Each time a condition fails, the counter is reset, and the maximum value is maintained throughout the traversal as the answer.

A candidate program generated by Qwen-Coder-14B-Instruct uses a brute-force approach, iterating over each starting position and simulating the moves. Although functionally correct, its complexity is O(N2) and too inefficient to work. The code is as follows:

- 1 #include <iostream>
- 2 #include <vector>
- 3 using namespace std;
- 4
- 5 int main() {
- 6 int N;
- 7 cin >> N;
- 8 vector<int> H(N);
- 9 for(int i = 0; i < N; i++) {
- 10 cin >> H[i];
- 11 }
- 12
- 13 int max_moves = 0;
- 14 for(int i = 0; i < N; i++) {
- 15 int current_moves = 0;
- 16 for(int j = i + 1; j < N; j++) {
- 17 if(H[j] <= H[j - 1]) {
- 18 current_moves++;
- 19 } else {
- 20 break;
- 21 }
- 22 }
- 23 if(current_moves > max_moves) {
- 24 max_moves = current_moves;
- 25 }
- 26 }
- 27
- 28 cout << max_moves << endl;
- 29 return 0;
- 30 }

Because of its inefficiency, this candidate program failed the official test cases. Nevertheless, due to the relatively small scale of the test cases in TACO and HARDTESTS Type 1 (Directly Generated), the candidate program passed these tests.

Additionally, the HARDTESTS Type 2 (Regular) input for this problem is generated using the following Python function:

- 1 import random
- 2
- 3 def gen_regular_input() -> str:
- 4 N = random.randint(1, 100000)
- 5 heights = [random.randint(1, 1000000000) for _ in range(N)]
- 6 return f'{N}\n' + ' '.join(map(str, heights))

We observe that since the Hi sequence is randomly generated, it fluctuates significantly, reducing the complexity of the “simulate moving from a certain square” procedure from O(N) to approximately O(1). Thus, the tests generated do not lead to the worst case complexity of the inefficient program and its overall time complexity effectively becomes O(N), enabling the candidate program to pass HARDTESTS Type 2 (Regular) test cases.

The HARDTESTS Type 3 (Hacking) inputs for this problem are generated using the following Python functions:

- 1 import random
- 2
- 3 # Monotonically decreasing sequence
- 4 def gen_hacking_input_monotonic_decreasing() -> str:
- 5 N = 100000
- 6 heights = list(range(1000000000, 1000000000 - N, -1))
- 7 return f'{N}\n' + ' '.join(map(str, heights))
- 8
- 9 # Monotonically increasing sequence
- 10 def gen_hacking_input_monotonic_increasing() -> str:
- 11 N = 100000
- 12 heights = list(range(1, N + 1))
- 13 return f'{N}\n' + ' '.join(map(str, heights))
- 14
- 15 # Alternating heights
- 16 def gen_hacking_input_alternating() -> str:
- 17 N = 100000
- 18 heights = [1000000000 if i % 2 == 0 else 1 for i in range(N)]
- 19 return f'{N}\n' + ' '.join(map(str, heights))

There are three hacking input generation functions: monotonically decreasing, monotonically increasing, and alternating sequences. The first generated input (monotonically decreasing) successfully increased the actual runtime complexity of the candidate program to O(N2), causing a timeout and consequently a failure on this test case.

### A.5.3 Example 3: False Negative of TACO

In this example, we show an example of false negative caused by the lack of special judge function in TACO tests. We also show how HARDTESTS can correctly evaluate the candidate program. Here’s the problem description:

AtCoder ABC117A: Given an integer T and an integer X (1 ≤ T ≤ 100, 1 ≤ X ≤ 100). Compute the value of T/X with an error tolerance within 10−3.

A candidate program generated by Qwen2.5-Coder-14B-Instruct is:

- 1 #include <iostream>
- 2 #include <iomanip>
- 3
- 4 int main() {
- 5 int T, X;
- 6 std::cin >> T >> X;
- 7
- 8 double timeInWorldA = static_cast<double>(T) / X;
- 9
- 10 std::cout << std::fixed << std::setprecision(10) << timeInWorldA << std::endl;
- 11
- 12 return 0;
- 13 }

This is clearly correct and passes all official test cases. It also passes all test cases from HARDTESTS, but it fails on TACO’s test cases. This is because using a simple string comparison function is insufficient due to potential differences in precision between the candidate output and the reference output. TACO does not provide a special output judging function for problems, which leads to false negatives. HARDTESTS provides a special output judging function, shown below:

- 1 def output_judging_function(input_str: str, candidate_output: str, reference_output:

→ str) -> bool:

- 2 # Parse the input
- 3 T, X = map(int, input_str.split())
- 4
- 5 # Calculate the expected output
- 6 expected_output = T / X
- 7
- 8 # Parse the candidate output
- 9 try:
- 10 candidate_value = float(candidate_output.strip())
- 11 except ValueError:
- 12 return False
- 13
- 14 # Check the absolute and relative error
- 15 absolute_error = abs(candidate_value - expected_output)
- 16 relative_error = absolute_error / abs(expected_output) if expected_output != 0 else

→ float('inf')

- 17
- 18 # The output is correct if either error is within the tolerance
- 19 return absolute_error <= 1e-3 or relative_error <= 1e-3

### A.6 Downstream Training and Evaluation Details

Teacher-distillation training and evaluation details. In the teacher-distillation experiments, our model is trained with the same training parameters used to train OlympicCoder-7B (epochs=10, learning_rate=4e-5, batch_size=128, cosine learning rate schedule with a decay to 10% of the peak learning rate and 32,768 max length). The evaluations are sampled with temperature=0.7, top_p=0.95, max_new_tokens=16384.

Self-distillation training and evaluation details. In the self-distillation experiments, our model is trained with the following training parameters (epochs=20, learning_rate=4e-5, batch_size=128, cosine learning rate schedule with a decay to 10% of the peak learning rate and 32,768 max length). The evaluations are sampled with temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_new_tokens=32768 as recommended by Qwen.

RL training and evaluation details. We use verl for RL training and firejail for sandboxing code execution. The rollouts are generated with temperature=1, top_p=0.95, top_k=20, min_p=0, response_length=24000, initial learning rate 5e-7. We use a global batch size of 32 and generate 32 samples per rollout. All our experiments are run on 8 NVIDIA H100 GPUs. We do not use KL divergence in our RL loss.

### A.7 Test Case Generation Without an Oracle Model

In the case that an oracle program y∗, or an oracle test suite V ∗ does not exist for a problem x, such as when problems are synthetically generated, we propose a method, based on ALGO (Zhang et al., 2023) that synthesizes both the oracle and tests. To start, we prompt an LLM, such as Anthropic Claude 3.5 Sonnet, to generate a brute-force solution ybf to the problem. Specifically, we encourage it to use inefficient methods such as exhaustive search and enumeration of the possible output space. This is founded on the observation that it is relatively easy to generate a solution that exhaustively searches the correct output, but more difficult to optimize it within a time complexity bound.

Then, an LLM is prompted to create a validator program and 10 edge test input generators, which are used to generate one test input each, {a1,...,a10}. To prevent the ybf from timing out when computing their respective outputs, we explicitly prompt the LLM to keep input values small. Once these test inputs are verified for correctness using the validator, the brute-force solution is used to generate the corresponding outputs ci = ybf(ai) for each input, resulting in a total of 10 input-output

pairs as test cases. Finally, the LLM is prompted to create one maximum-length test case amax with inputs at the upper bounds of the problem’s constraints, designed to catch solutions that are functionally correct but inefficient. This test case is considered to be passsed as long as the program produces an output before timing out. Crucially, all 11 of the generated test cases {a1,...,a10,amax} are designed to cause seemingly correct programs to fail, and none are generated using random inputs.

We compare this method to the baseline method outlined in AceCoder (Zeng et al., 2025a), which uses a direct prompt to generate 20 full test cases (inputs and corresponding outputs), also using Claude 3.5. Then, after prompting a stronger model such as Qwen2.5- Coder-32B-Instruct to generate a solution, the test cases that cause the solution to fail are considered hallucinated and are filtered out. Problems with fewer than 5 test cases after filtering are discarded.

To evaluate the accuracy of rewards that our test cases can give to model training, we evaluate the precision and recall over candidate programs generated by LLMs and written by humans on subsets of problems in HARDTESTS.

The quality of the test cases are verified using 165 Atcoder problems, each with 50 sample solutions. It is clear from these experiments (shown in Table 8) that our method can also work much better than the baseline even when oracle programs are not available. The false positive rate of HARDTESTGEN is only half as high as AceCoder, showing that deliberately crafting high-quality, hard test cases is crucial for effective program verifiers.

We will show some examples of the test generation process in the following sections.

Table 8: Performance comparison of oracle-free test generation algorithms based on error rates.

### False Positive Rate (FPR) False Negative Rate (FNR)

AceCoder 32.49 2.59 HARDTESTGEN 17.67 2.19

### A.7.1 Example 1: Brute-force oracle solution generated by ALGO

In this example, we show a brute-force oracle solution ybf that ALGO generates. The problem is challenging, and while the LLM is unable to correctly solve it directly, its brute-force solution works. Here’s the problem description:

AtCoder ABC301C: A single-player card game is popular in AtCoder Inc. Each card in the game has a lowercase English letter or the symbol written on it. There is plenty number of cards for each kind. The game goes as follows.

- - Arrange the same number of cards in two rows.
- - Replace each card with with one of the following cards: a, t, c, o, d, e, r.
- - If the two rows of cards coincide, you win. Otherwise, you lose. To win this game, you will do the following cheat.
- - Freely rearrange the cards within a row whenever you want after step 1.

You are given two strings S and T (1 ≤ |S|,|T| ≤ 2 × 105), representing the two rows you have after step 1. Determine whether it is possible to win with cheating allowed. If it is possible to win with cheating allowed, print Yes; otherwise, print No.

The solution generated by Claude 3.5 Sonnet is:

- 1 from itertools import product
- 2
- 3 def solve():
- 4 S = input().strip()
- 5 T = input().strip()
- 6
- 7 # Find positions of @ in both strings

- 8 s_at_pos = [i for i, c in enumerate(S) if c == '@']
- 9 t_at_pos = [i for i, c in enumerate(T) if c == '@']
- 10
- 11 # Possible characters to replace @
- 12 possible = ['a', 't', 'c', 'o', 'd', 'e', 'r']
- 13
- 14 # Try all possible combinations
- 15 for s_comb in product(possible, repeat=len(s_at_pos)):
- 16 # Create new S string with replacements
- 17 s_new = list(S)
- 18 for pos, char in zip(s_at_pos, s_comb):
- 19 s_new[pos] = char
- 20 s_new = ''.join(s_new)
- 21
- 22 for t_comb in product(possible, repeat=len(t_at_pos)):
- 23 # Create new T string with replacements
- 24 t_new = list(T)
- 25 for pos, char in zip(t_at_pos, t_comb):
- 26 t_new[pos] = char
- 27 t_new = ''.join(t_new)
- 28
- 29 # Sort both strings (since we can rearrange)
- 30 if sorted(s_new) == sorted(t_new):
- 31 print("Yes")
- 32 return
- 33
- 34 print("No")
- 35
- 36 solve()

This solution is evidently inefficient. A full analysis yields runtime in the bound of O(nlog n7n), where n = |S| = |T|, but the exponential complexity is clear from the Cartesian product function in the implementation. For inputs n > 10, this algorithm quickly becomes intractable. However, for inputs n ≤ 10 it is able to generate valid test outputs, allowing it to correctly evaluate the validity of submitted solutions. The test outputs it generates achieve a 100% accuracy, compared to actual execution results from the online judge.

### A.7.2 Example 2: Test cases generated by ALGO

In this example we show a contest coding problem for which ALGO effectively generates a testing suite. Here’s the problem description:

AtCoder cafeteria sells meals consisting of a main dish and a side dish. There are N types of main dishes, called main dish 1, main dish 2, ..., main dish N. Main dish i costs ai yen. There are M types of side dishes, called side dish 1, side dish 2, ..., side dish M. Side dish i costs bi yen.

A set meal is composed by choosing one main dish and one side dish. The price of a set meal is the sum of the prices of the chosen main dish and side dish.

However, for L distinct pairs (c1,d1), ..., (cL,dL), the set meal consisting of main dish ci and side dish di is not offered because they do not go well together. That is, NM − L set meals are offered. (The constraints guarantee that at least one set meal is offered.)

Find the price of the most expensive set meal offered. The input is given from Standard Input in the following format: N M L

- a1 a2 ... aN
- b1 b2 ... bM
- c1 d1 c2 d2

.

cL dL Constraints:

- - 1 ≤ N,M ≤ 105
- - 0 ≤ L ≤ min(105,NM − 1)
- - 1 ≤ ai,bi ≤ 109

The first 3 edge test input generators created by ALGO are shown below, corresponding to the following test inputs. Note that the values are at the boundaries of the input bounds and follow clearly defined structures.

- 1 ["1 1 0\n1000000000\n1000000000",
- 2 "10 10 1\n1000 2000 3000 4000 5000 6000 7000 8000 9000 10000\n1000 2000 3000 4000 5000

→ 6000 7000 8000 9000 10000\n1 1",

- 3 "50 50 100\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000\n1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000 1000000000\n33 36\n5 1\n18 44\n43 12\n5 37\n50 36\n15 14\n10 27\n34 3\n16 40\n47 18\n28 14\n9 10\n20 40\n41 8\n4 41\n10 20\n7 21\n37 1\n49 39\n9 3\n14 26\n42 18\n41 28\n26 9\n13 21\n39 3\n16 17\n24 30\n17 46\n33 42\n9 42\n34 7\n36 13\n6 36\n33 8\n26 2\n43 18\n4 45\n35 23\n39 5\n33 35\n8 45\n34 18\n43 48\n14 12\n17 11\n26 41\n4 47\n44 28\n9 9\n50 19\n2 15\n20 5\n9 39\n20 14\n24 39\n5 13\n4 15\n10 39\n48 23\n43 36\n21 33\n4 35\n11 13\n46 10\n11 31\n8 44\n2 37\n46 37\n16 39\n39 43\n34 38\n3 50\n7 20\n34 13\n34 31\n28 33\n31 4\n22 1\n38 19\n12 9\n6 23\n27 18\n9 4\n11 10\n28 26\n2 7\n45 42\n31 43\n39 1\n15 1\n11 19\n12 48\n15 10\n4 7\n45 26\n2 18\n35 15\n33 27"]

→ → → → → → → → → → → → → → → → → → → → →

- 1 def gen_edge_case_input_1():
- 2 # Minimum case: N=M=1, L=0
- 3 return "1 1 0\n1000000000\n1000000000"
- 4
- 5 def gen_edge_case_input_2():
- 6 # Maximum values case
- 7 N = M = 100 # Using smaller values for practicality
- 8 L = N*M - 1
- 9 a = [10**9] * N
- 10 b = [10**9] * M
- 11 pairs = [(i, j) for i in range(1, N+1) for j in range(1, M+1)]
- 12 pairs.pop() # Leave one combination possible
- 13
- 14 result = f"{N} {M} {L}\n"
- 15 result += " ".join(map(str, a)) + "\n"
- 16 result += " ".join(map(str, b)) + "\n"
- 17 for c, d in pairs:
- 18 result += f"{c} {d}\n"
- 19 return result.strip()
- 20
- 21 def gen_edge_case_input_3():
- 22 # Only one forbidden pair
- 23 N = M = 10
- 24 a = [i*1000 for i in range(1, N+1)]
- 25 b = [i*1000 for i in range(1, M+1)]
- 26 result = f"{N} {M} 1\n"
- 27 result += " ".join(map(str, a)) + "\n"
- 28 result += " ".join(map(str, b)) + "\n"

- 29 result += "1 1"
- 30 return result

Also, the generator for the maximum-length test input amax is shown here. It produces a test input where N = M = 105, which is the upper bound of the problem.

- 1 import random
- 2
- 3 def gen_maximum_edge_case_input():
- 4 N = 100000
- 5 M = 100000
- 6 L = 100000
- 7
- 8 # Generate main dish prices close to max value
- 9 main_prices = [random.randint(999999000, 1000000000) for _ in range(N)]
- 10
- 11 # Generate side dish prices close to max value
- 12 side_prices = [random.randint(999999000, 1000000000) for _ in range(M)]
- 13
- 14 # Generate L unique forbidden pairs
- 15 used_pairs = set()
- 16 forbidden_pairs = []
- 17
- 18 # Start with some specific high-value combinations
- 19 for i in range(L):
- 20 while True:
- 21 c = random.randint(1, N)
- 22 d = random.randint(1, M)
- 23 if (c, d) not in used_pairs:
- 24 used_pairs.add((c, d))
- 25 forbidden_pairs.append((c, d))
- 26 break
- 27
- 28 # Build the input string
- 29 result = []
- 30 result.append(f"{N} {M} {L}")
- 31 result.append(" ".join(map(str, main_prices)))
- 32 result.append(" ".join(map(str, side_prices)))
- 33
- 34 for c, d in forbidden_pairs:
- 35 result.append(f"{c} {d}")
- 36
- 37 return "\n".join(result)

This test suite effectively achieves 100% accuracy on evaluating submissions, demonstrating that precise test inputs are crucial for oracle-free verifiers.

### A.7.3 Example 3: Test cases generated by AceCoder

For the same Atcoder problem as Example A.7.2, AceCoder generates the following 16 test cases with inputs and outputs after filtering. While the LLM implicitly knows to generate edge test cases, shown in the maximal values of ci,di, all of the test cases have relatively similar and low values of M and N.

- 1 [{"input": "2 3 3\n2 1\n10 30 20\n1 2\n2 1\n2 3", "output": "31"},
- 2 {"input": "2 1 0\n1000000000 1\n1000000000", "output": "2000000000"},
- 3 {"input": "1 1 0\n5\n7", "output": "12"},
- 4 {"input": "3 3 4\n10 20 30\n5 15 25\n1 1\n2 2\n3 1\n1 3", "output": "55"},
- 5 {"input": "5 3 7\n100 200 300 400 500\n100 200 300\n1 1\n1 2\n1 3\n2 1\n2 2\n3 1\n4 1",

→ "output": "800"},

- 6 {"input": "2 2 1\n999999999 999999998\n999999997 999999996\n1 1", "output":

→ "1999999995"},

- 7 {"input": "3 2 2\n5 4 3\n2 1\n1 1\n2 2", "output": "6"},
- 8 {"input": "4 3 5\n10 9 8 7\n6 5 4\n1 1\n2 2\n3 3\n4 1\n4 2", "output": "15"},

- 9 {"input": "2 4 3\n100 200\n300 400 500 600\n1 1\n1 2\n2 3", "output": "800"},
- 10 {"input": "3 3 0\n1 2 3\n4 5 6", "output": "9"},
- 11 {"input": "4 2 3\n10 20 30 40\n50 60\n1 1\n2 2\n3 1", "output": "100"},
- 12 {"input": "5 2 4\n1 2 3 4 5\n6 7\n1 1\n2 1\n3 1\n4 1", "output": "12"},
- 13 {"input": "3 4 6\n100 200 300\n400 500 600 700\n1 1\n1 2\n1 3\n2 1\n2 2\n3 3", "output":

→ "1000"},

- 14 {"input": "2 2 0\n1000000000 999999999\n1000000000 999999999", "output": "2000000000"},
- 15 {"input": "3 3 3\n100 200 300\n100 200 300\n1 1\n2 2\n3 3", "output": "500"},
- 16 {"input": "5 5 12\n1 2 3 4 5\n1 2 3 4 5\n1 1\n1 2\n1 3\n2 1\n2 2\n2 3\n3 1\n3 2\n3 3\n4

→ 1\n4 2\n5 1", "output": "10"}]

These test cases fail to correctly categorize solutions that exceed the problem’s time limit. One such example is shown below, which AceCoder falsely categorizes as a positive solution. Compared to Example A.7.2, in which ALGO generated test inputs as large as N = M = 105, the test cases from AceCoder are no larger than N = M = 5, making them unable to break inefficient programs. Without a brute-force reference oracle, and constrained by the requirement of generating input-output pairs simultaneously, the LLM used by AceCoder sticks to simple test cases that it can be confident are correct. Moreover, longer test cases are likelier to contain hallucinations, and get removed by their filtering process. As a result, their test cases are relatively weaker and result in less effective verifiers.

- 1 import sys
- 2
- 3 def main():
- 4 input = sys.stdin.readline
- 5 N, M, L = map(int, input().split())
- 6 a = list(map(int, input().split()))
- 7 b = list(map(int, input().split()))
- 8 incompatible_pairs = set()
- 9 for _ in range(L):
- 10 c, d = map(int, input().split())
- 11 incompatible_pairs.add((c - 1, d - 1)) # Adjusting indices to be zero-based
- 12
- 13 max_price = 0
- 14 for i in range(N):
- 15 for j in range(M):
- 16 if (i, j) not in incompatible_pairs:
- 17 max_price = max(max_price, a[i] + b[j])
- 18
- 19 print(max_price)
- 20
- 21 if __name__ == "__main__":
- 22 main()

