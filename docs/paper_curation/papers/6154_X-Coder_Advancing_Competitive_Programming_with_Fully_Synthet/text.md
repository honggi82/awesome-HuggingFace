[Figure 1]

## -Coder: Advancing Competitive Programming with Fully Synthetic Tasks, Solutions, and Tests

# arXiv:2601.06953v2[cs.CL]1Feb2026

Jie Wu*1 Haoling Li*1 Xin Zhang*†‡2 Jiani Guo3 Jane Luo2 Steven Liu2 Yangyu Huang2 Ruihang Chu‡1 Scarlett Li2 Yujiu Yang1

https://github.com/JieWu02/X-Coder

### Abstract

###### Training Paradigm Comparison Performance Evaluation

Competitive programming poses a significant challenge for Code LLMs. While recent models have shown promise, they heavily rely on finite real-world data, raising concerns about scalability and contamination. In this paper, we investigate a critical question: Can we elevate models to expert-level reasoning performance using fully synthetic data? In response, we first observe that off-the-shelf synthesis methods yield suboptimal results in this domain. To address this, we systematically investigate the key factors governing synthetic data quality. Leveraging these findings, we significantly advance the feature-based synthesis paradigm via domain-specific evolution and a dual-verification strategy, promoting task solvability, solution correctness, and test accuracy. Using this high-quality synthetic data, we train the X-Coder model series under an SFTthen-RL paradigm. X-Coder-7B shows significant performance gains on the challenging LiveCodeBench v5 (62.9% avg@8) and v6 (55.8% avg@8), outperforming larger models trained on real-world data. Extensive analysis distills valuable insights into synthetic data scaling, the necessity of domain-adapted feature evolution, and code-centric reinforcement.

Previous Method: Partially Synthetic

62.9

SFT RL

[Figure 2]

57.8 57.5 57.1

Real-world Tasks

Real-world Tasks

[Figure 3]

51.7 51.3

47.6

Real-world Tests

Synthetic Solutions

X-Coder (Ours): Fully Synthetic

SFT RL

[Figure 4]

Synthetic Tasks

Synthetic Tasks

[Figure 5]

[Figure 6]

Qwen3-8B AceReason1.1-7B OpenThinker3-7B OCR-Qwen-Instruct-7B Skywork-OR1-7B

[Figure 7]

X-Coder-7B

Mimo-7B

Synthetic Tests

Synthetic Solutions

Figure 1. Left: To address persistent data scarcity in competitive programming, we build a scalable alternative by training Code LLMs with fully synthetic data. Right: Avg@8 results on LiveCodeBench v5. X-Coder achieves significant performance gains on competitive programming using fully synthetic data.

such as HumanEval (Chen et al., 2021; Liu et al., 2023a) and MBPP (Austin et al., 2021) are largely considered solved. However, the frontier of code intelligence has shifted towards competitive programming. This domain is exemplified by platforms like Codeforces and benchmarks like LiveCodeBench (Jain et al., 2024), requiring deep algorithmic reasoning and complex problem-solving capabilities.

To bridge this gap, recent success in reasoning models has been driven by the utilization of extensive reasoning data, both for supervised fine-tuning (SFT) and reinforcement learning (RL) (Guo et al., 2025; Hugging Face, 2025; Luo et al., 2025). However, this data-driven scaling faces a critical bottleneck: the scarcity of high-quality training data. The finite pool of real-world competitive programming tasks is insufficient to sustain the scaling laws of reasoning models. Furthermore, existing collections (Hendrycks et al., 2021; Li et al., 2022) are heavily exhausted, raising concerns about data contamination and overfitting. This dilemma poses a critical question: Can we bypass the reliance on real-world data and elevate models to expert-level reasoning performance using fully synthetic data?

### 1. Introduction

As Code LLMs advance, they have demonstrated remarkable proficiency in basic programming tasks. Benchmarks

*Equal contribution . †Project Lead. ‡Corresponding Authors. Work done during the internships of Jie Wu, Haoling Li, Jiani Guo, Jianwen Luo, and Steven Liu at Microsoft Research. 1Tsinghua University 2Microsoft Research 3Wuhan University. Correspondence to: Xin Zhang <xinzhang3@microsoft.com>, Ruihang Chu <rhchu@mails.tsinghua.edu.cn>.

Answering this question is non-trivial. Generating highquality synthetic data for competitive programming is significantly harder than for general-purpose programming, facing unique challenges across three dimensions:

(i) Tasks must be both solvable and challenging. Synthesizing competitive-level tasks often yields ill-defined or trivial instances, failing to provide the rigorous complexity needed to train expert models. (ii) High-quality SFT demands logically sound solutions. Validating these without ground-truth test cases is inherently risky, potentially polluting the training set with incorrect reasoning paths. (iii) RL necessitates reliable reward signals. Weak or insufficient synthetic tests fail to catch incorrect solutions, providing noisy reward signals that can mislead the reinforcement learning process.

Given these challenges, off-the-shelf synthesis frameworks (Wang et al., 2025; Wei et al., 2024) fail to generalize to competitive programming and yield suboptimal results. In this work, we validate the hypothesis that fully synthetic data can sufficiently drive expert-level performance in code reasoning. To achieve this, we systematically investigate the key factors governing the effectiveness of synthetic code reasoning data. Leveraging these findings, we significantly advance the synthesis paradigm via domain-specific evolution and a dual-verification strategy, establishing a high-quality synthetic foundation for training.

To validate the feasibility of fully synthetic training, we train the X-Coder model series using only the synthesized data under a standard SFT-then-RL paradigm. As shown in

- Figure 1, X-Coder achieves significant performance gains on the challenging LiveCodeBench v5 and v6.

Through extensive evaluation, we further distill key insights into what makes synthetic data effective, shedding light on synthetic data scaling, the necessity of domain-specific evolution, and the role of code-centric reinforcement learning.

Our contributions are as follows:

- (1) We promote synthetic data quality via domain-specific adaptation and dual-verification, empirically demonstrating that fully synthetic data is sufficient to achieve expert-level performance in competitive programming, challenging the conventional reliance on real-world datasets.
- (2) Using the constructed synthetic data, we develop the X-Coder model series under an SFT-then-RL paradigm. XCoder-7B achieves significant performance gains on LiveCodeBench v5 and v6, particularly on medium and hard splits. Related resources will be made publicly available.
- (3) We offer insights into synthetic data scaling, highlighting the necessity of domain-adapted feature evolution and dual-verification. Furthermore, we shed light on reasoning types, code-centric reinforcement learning, and emergent fine-grained failure modes.

### 2. Related Work

Data Synthesis for Code Generation. The research community has long recognized the scarcity of high-quality coding tasks. To address this, Wizard-Coder (Luo et al.,

- 2024) extends Evol-Instruct (Xu et al., 2024) by evolving basic code-instruction data into augmented variants. rStar-Coder (Liu et al., 2025a) further adapts this augmentation strategy to the competitive programming domain. CodeEvo (Sun et al., 2025) introduces a coder–reviewer interaction framework to collaboratively synthesize highquality instruction–code pairs. SelfCodeAlign (Wei et al.,

- 2024) pioneered concept composition, generating novel tasks by systematically recombining fundamental concepts extracted from seed problems. EpiCoder (Wang et al., 2025) follows this direction by sampling sub-features from a large and expressive feature tree to formulate novel problems, further improving task complexity and diversity.

However, applying these general-purpose techniques to competitive programming yields suboptimal results. To bridge this gap, we systematically investigate the key factors governing data effectiveness and develop a synthesis strategy tailored for code reasoning. Specifically, we advance featurebased synthesis via domain-specific evolution to generate complex tasks and employ a dual-verification strategy to guarantee solution correctness. This approach establishes a scalable, high-quality, fully synthetic foundation capable of driving expert-level performance.

Post-training for Competitive Programming. Current training paradigms generally fall into three categories: (i) Purely supervised fine-tuning (SFT) on real-world tasks or their variants (Labs, 2025; Guha et al., 2025; Liu et al.,

- 2025a); (ii) Purely reinforcement learning (RL) using GRPO-like algorithms (Shao et al., 2024a; He et al., 2025; Luo et al., 2025; Fu et al., 2025); and (iii) A hybrid SFTthen-RL approach, which typically mixes coding and mathematical data to mitigate the scarcity of high-quality code problems (Liu et al., 2025b; Xiaomi et al., 2025; Su et al.,

- 2025). Crucially, all these approaches rely heavily on finite real-world data sources. The hybrid methods, in particular, often depend on mathematical reasoning transfer due to the lack of sufficient competitive programming data. In contrast, we demonstrate that fully synthetic data can independently sustain the full SFT-then-RL cycle for code, achieving significant performance without relying on real-world data or cross-domain mixtures.

### 3. Synthesis of Competition-Level Coding Data

We adapt the feature-based synthesis framework (Wang et al., 2025) to the competitive programming domain, establishing a comprehensive methodology to construct tasks that support both SFT and RL stages. As illustrated in

###### 1 Task Generation

[Figure 8]

[Figure 9]

Selected Subtree

[Figure 10]

Quik Sorting

[Figure 11]

###### Sorting:

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Quik sorting

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Merge Sorting

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

###### Math:

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Evovled

[Figure 44]

[Figure 45]

[Figure 46]

Number Theory

Scenario

[Figure 47]

[Figure 48]

###### …

[Figure 49]

Pre order travesal

[Figure 50]

[Figure 51]

Priority Extraction During Traversal. You are given a tree with weighted nodes ...

[Figure 52]

[Figure 53]

[Figure 54]

###### Travesal:

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Post-order travesal

[Figure 60]

Post order travesal

Tasks

Code Snippet Extract Features Evolve and Merge Select and Thinking

4 Dual Verification

[Figure 61]

[Figure 62]

- 2 Test Generation

[Figure 63]

- 3 Solution Generation

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

- Y1
- Y2 Y1

[Figure 68]

[Figure 69]

[Figure 70]

X1

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

Tool Prompt

[Figure 77]

X1 Y1 Test Case

…

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

…

… …

Task

[Figure 83]

Test Input

[Figure 84]

[Figure 85]

[Figure 86]

Test Inputs

[Figure 87]

[Figure 88]

[Figure 89]

Solutions Outputs

Major Voting

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

###### …

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

###### …

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Test Cases

…

LLM

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Sampling

Solutions

[Figure 117]

Golden Solution

Scoring

Figure 2. Overview of the data synthesis framework. The framework begins with feature extraction and evolution to construct a domain-specific feature tree. From this, we generate multi-style competitive tasks by formulating scenarios from compatible feature sets. Finally, we synthesize solutions and test inputs, employing a dual-verification strategy to ensure solution correctness and test accuracy, yielding a high-quality data foundation for subsequent SFT and RL.

- Figure 2, this process addresses the unique challenges of synthesis through four key steps: (i) generating novel and challenging problems (with the capacity for easy scaling in quantity); (ii) constructing diverse and comprehensive test inputs for each problem (including boundary and stress tests); (iii) producing high-quality candidate solutions; and (iv) employing a dual-verification strategy that cross-checks solutions with test cases to yield more accurate test outputs and more reliable solutions.

(i) Task Generation. While EpiCoder (Wang et al., 2025) introduced feature-based synthesis for general coding tasks, we extend this approach with three key improvements specifically tailored for the complexity of competitive programming. First, instead of relying on broad definitions of features, we explicitly extract and evolve competition-related features from 10k code snippets in the TACO dataset (Li et al., 2023) using GPT-4o-0513 (detailed in Appendix B.1). Second, formulating competitive scenarios from a rich feature tree is non-trivial, as LLMs often oversimplify complex prompts into trivial cases, thereby reducing both diversity and difficulty. To address this, we adopt a two-stage process that separates feature selection from scenario formulation: first, selecting mutually consistent features for meaningful composition; and second, formulating hint-free tasks that demand genuine reasoning. This two-stage approach significantly outperforms single-step generation (Appendix Table 9). We further incorporate one-shot prompting to

improve task understanding and instruction-following. Third, we adapt the synthesis method to support multi-style task generation, covering Codeforces1-style tasks (standard input/output with narrative-rich contexts), LeetCode-style2 tasks (starter code with predefined function signatures), and AtCoder3-style tasks (concise specifications with minimal explanations), thereby enhancing task diversity. Examples of the task generation process are provided in Appendix B.2, together with difficulty estimates on generated tasks in Appendix B.3 and style comparisons in Appendix B.2.3.

(ii) Test Input Generation. Obtaining sufficient and accurate test cases is a formidable challenge. Problems from competitive programming platforms often do not provide test cases, or only provide a limited number, due to platform constraints. This results in insufficient quantity, difficulty, and coverage of test cases during RL training. To address the inherent scarcity of test cases in synthesized data, we investigate two complementary methods for generating the input component of the test case. The Prompting-based method instructs the LLM to interpret the problem’s input constraints and directly generate multiple test inputs, covering both standard cases and edge-case instances. The Tool-based method leverages CYaRon4, a dedicated test

- 1https://codeforces.com/
- 2https://leetcode.com/
- 3https://atcoder.jp/
- 4https://github.com/luogu-dev/cyaron

case generation library, enabling the LLM to construct test inputs by invoking functions documented within the library after understanding the problem. For each task, we generate a set of n test case inputs [x1,x2,...,xn]. Detailed description of test input generation is provided in Appendix D, and a comparative analysis is presented in Appendix D.2.

- (iii) Candidate Solutions Generation. For each task, we generate multiple candidate solutions using advanced open-source reasoning LLMs, obtaining m answers [A1,A2,...,Am]. We verify that each candidate solution includes a complete reasoning process and a Python code implementation, and we ensure the absence of syntax errors through static analysis methods based on Abstract Syntax Tree (AST). Filtering criteria are provided in Appendix C.1.
- (iv) Dual-Verification of Solutions and Test Cases. We adopt a dual-verification strategy. Step 1 of this strategy extends the principle of self-consistency (Wang et al.,

2023) by applying majority voting across candidate solutions from multiple LLMs, which mitigates model-specific biases and enhances generalization, thereby yielding a reliable test output for each input. Step 2 then identifies the top-performing candidate solution by incorporating test case difficulty weighting alongside a hold-out validation set.

- Step 1: Verification of Test Cases via Consensus Voting. First, we establish a preliminary ground truth for each test case input. For a given input xi, we execute all candidate solutions to obtain a set of outputs {yi1,yi2,...,yim}, where

yij = Aj(xi). A provisional ground truth output yˆi is determined via majority voting:

yˆi = argmax

y

m

j=1

I(yij = y) , (1)

where I(·) is the indicator function. Empirical evaluation on the TACO dataset demonstrates that this voting strategy achieves 94.7% labeling accuracy with 8 sampled solutions (see analysis in Appendix E.3 and Table 14). This yields a candidate test set Tcandidate = {(x1,yˆ1),...,(xn,yˆn)}. Crucially, we posit that not all test cases are of equal importance; boundary or edge cases are critical for robust evaluation. We therefore introduce a weighting function w(xi) → wi that assigns a higher score to more challenging test cases. The weight wi is determined by combining semantic-based heuristics (e.g., assigning higher weights to boundary and stress tests explicitly requested in the prompt) and size-based metrics (e.g., input file size as a proxy for computational complexity). Detailed weighting criteria are provided in Appendix E.2.

- Step 2: Verification of Solutions via Weighted Evaluation and Hold-out Validation. To ensure that our selected “golden” solution generalizes beyond the generated data, we partition the candidate test set. We randomly sample a subset

of Tcandidate (e.g., 50%) to form a hold-out validation set, Tval. The remaining data constitutes our primary weighted test suite, Tgolden. The dual-verification process culminates in selecting the golden answer, Agolden. A candidate solution Aj is first evaluated on Tgolden using a weighted score. The top-performing candidate, A′golden, is identified as:

A′golden = argmax

Aj∈{A1,...,Am}

(xi,yˆi)∈Tgolden wi · I(Aj(xi) = yˆi) (2)

The final confirmation of Agolden is contingent upon its performance on the unseen hold-out set Tval. We verify that A′golden also achieves the highest (or a competitively high) unweighted accuracy on Tval relative to other candidates. This additional validation step ensures that the selected solution is not merely overfitted to the specifics of the weighted test cases but demonstrates superior, generalizable correctness. The detailed algorithm is provided in Appendix E.1.

We ensure task solvability by using GPT-5 (high reasoning effort) as a proxy solver, filtering out approximately 36.9% of tasks where it achieves a zero pass rate on the voted test cases. We posit that if such a capable model fails completely, the task likely suffers from ambiguity or underspecification. The pass rate distribution of the remaining tasks is shown in Table 18 (Appendix E.5).

Ultimately, this process yields a verified solution Agolden and a comprehensive test suite Tgolden for every generated task q. We leverage these high-quality synthetic assets to drive post-training for Code LLMs: (q,Agolden) pairs for supervised fine-tuning, and (q,Tgolden) for reinforcement learning via GRPO algorithm (Shao et al., 2024a).

### 4. Experiment

Data and Training. We adopt GPT-o3-mini (OpenAI, 2025) for task formulation, DeepSeek-R1-0528 (DeepSeek-AI, 2025) and Qwen3-235B-A22B-Thinking-2507 (Yang et al., 2025) for solution sampling, and R1-0528 for test case generation. Statistics for SFT datasets are provided in Appendix C.2. For SFT, we set the learning rate at 5e-5, with a global batch size of 128 to train 8 epochs. For RL, the reward is defined as the fraction of passed tests among all given tests (detailed in Appendix A.2). The program executes in an isolated sandbox environment deployed with Redis, which supports optimized concurrent code testing (infrastructure details are provided in Appendix A.5). Training configurations and costs are supplemented in Appendix A.4.

Evaluation. We use LiveCodeBench v5 and v6 (Jain et al., 2024), the leading benchmarks for code reasoning. Using Qwen2.5-Coder-7B-Instruct and Qwen3-8B-Base as backbones, we report avg@8 scores (temperature 0.6, top-p 0.95, top-k 20) to ensure fair comparison with baselines. Be-

Table 1. Performance on LiveCodeBench v5. X-Coder shows strong coding expertise with fewer, fully synthetic tasks, and achieves additional gains through subsequent RL stages. †: OpenThinker3 integrates human-written tasks with synthetic math tasks. rStar-Coder augments real-world coding tasks with synthesized rewrites for mixed training, whereas X-Coder relies on fully synthetic tasks.

Model Base Model SFT RL Size Data Task Metric V5 Score V6 Score

SFT Baselines Bespoke-Stratos (Labs, 2025) Qwen2.5-Instruct (Qwen et al., 2025) ✓ ✗ 7B 17k Real pass@1 16.2 8.57 OpenThinker3 (Guha et al., 2025) Qwen2.5-Instruct ✓ ✗ 7B 1,200k Mixed† - 51.7 40.8 OlympicCoder (Hugging Face, 2025) Qwen2.5-Coder-Instruct (Hui et al., 2024) ✓ ✗ 7B 100k Real - 40.9 19.3 OCR-Qwen-Instruct (Ahmad et al., 2025) Qwen2.5-Instruct ✓ ✗ 7B 736k Real avg@64 51.3 44.5 rStar-Coder (Liu et al., 2025a) Qwen2.5-Coder-Instruct ✓ ✗ 7B 580K Mixed† avg@16 57.3 – Qwen3-8B (Yang et al., 2025) Qwen3-8B-Base ✓ ✗ 8B - Real - 57.5 48.4

###### RL Baselines

Skywork-OR1 (He et al., 2025) R1-Distilled-Qwen (DeepSeek-AI, 2025) ✗ ✓ 7B 124k Real avg@32 47.6 40.0 DeepCoder-Preview (Luo et al., 2025) R1-Distilled-Qwen ✗ ✓ 14B 24k Real pass@1 57.9 48.5 AReal-boba² (Fu et al., 2025) R1-Distilled-Qwen ✗ ✓ 14B 24k Real avg@32 58.1 56.7

###### SFT-then-RL Baselines (Stage 1)

AceReason1.1-SFT (Liu et al., 2025b) Qwen2.5-Math (Yang et al., 2024) ✓ ✗ 7B 2.2M Real avg@8 51.2 MiMo-SFT (Xiaomi et al., 2025) MiMo-Base ✓ ✗ 7B 500k Undisclosed avg@8 52.3 45.5 Klear-Reasoner-SFT (Su et al., 2025) Qwen3-Base (Yang et al., 2025) ✓ ✗ 8B 1500k Real avg@8 58.5 49.6

X-Coder-Qwen2.5-SFT Qwen2.5-Coder-Instruct ✓ ✗ 7B 200k Syn avg@8 60.3±2.5 53.5±1.7 X-Coder-Qwen3-SFT Qwen3-8B-Base ✓ ✗ 8B 200k Syn avg@8 59.4±2.0 55.4±2.3

###### SFT-then-RL Baselines (Stage 2)

AceReason1.1 AceReason1.1-SFT ✓ ✓ 7B - Real avg@8 57.2 52.1 MiMo MiMo-SFT ✓ ✓ 7B 130k Undisclosed avg@8 57.8 49.3 Klear-Reasoner Klear-Reasoner-SFT ✓ ✓ 8B 106k Real avg@8 61.6 53.1

X-Coder-Qwen2.5 X-Coder-Qwen2.5-SFT ✓ ✓ 7B 40k Syn avg@8 62.9±1.8 55.8±1.9 X-Coder-Qwen3 X-Coder-Qwen3-SFT ✓ ✓ 8B 40k Syn avg@8 64.0±2.5 56.5±1.3

Learning Dynamics of Different Scaling Dimensions

Task Scaling Comparison

70

|v1: 32K tasks v2: 64K tasks v3: 128K tasks|× 1 solution × 1 solution<br><br>× 1 solution| | | | |
|---|---|---|---|---|---|
|v4: 200K tasks v5: 16K tasks v6: 8K tasks ×<br><br>|× 1 solution × 4 solutions 8 solutions| | | | |
| | | |50%| | |
| | | |40%<br><br>45%| |[Figure 118]<br><br>|
| | | |35% 2000 2500|3000 3500|4000|

62.7

54.1

60

BestScore(%)BestScore(%)

47

43.7

60

40

Score(%)

50

20

32K×1 64K×1 128K×1 200K×1

Solution Scaling Comparison

40

60

44.4 45.1 47

30

40

20

20

8K×8 16K×4 64K×1

0 2000 4000 6000 8000 10000 12000

Training Step

Figure 3. Left: Data scaling on LiveCodeBench v5. Right: Scaling unique tasks and scaling solutions per task.

#### 4.2. SFT Experiments and Analysis

yond the primary setup, we assess the generalizability of synthetic training across the Llama model family and other code generation benchmarks in Appendix F.

Scaling Laws Hold for Synthetic Data. During the SFT stage, we investigate a central question: Can the SFT dataset be effectively scaled, and along which dimension should it be scaled more favorably? To explore this, we are inspired by AceReason-Nemotron 1.1 (Liu et al., 2025b) and expand the SFT dataset from two distinct perspectives: increasing the number of unique tasks and enlarging the number of solutions per task. We design six subsets (v1–v6): v1–v4 increase the number of unique tasks (32k, 64k, 128k, and 192k unique prompts, each with 1 solution), while v5–v6 expand the number of solutions per task (16k unique prompts with 4 solutions, and 8k unique prompts with 8 solutions). The results in Figure 3 reveal a promising scaling trend: performance improves monotonically as the dataset size increases (v1 → v4), rising steadily from 43.7% to 62.7%.

#### 4.1. Main Results

As shown in Table 1, during the SFT stage, X-Coder-SFT achieves an avg@8 pass rate of 60.3. X-Coder-SFT exhibits a clear advantage over 14B-based RL models (e.g., DeepCoder-Preview-14B, AReal-boba²-14B), despite those models being built on the stronger foundation R1-DistilledQwen. Relative to SFT-then-RL models, X-Coder further boosts its performance after RL, reaching 62.9. On Qwen3Base, X-Coder attains an avg@8 pass rate of 64.0.

Table 2. Head-to-head Comparison with OpenCodeReasoning using Qwen2.5-Coder-7B-Instruct. Model Avg. Pass@1 Pass@8 Easy Medium Hard

OpenCodeReasoning-Qwen 53.6 69.0 95.2 67.0 21.8 X-Coder-7B-SFT 60.3 77.0 96.8 73.3 37.8

Task Diversity Matters More than Solution Diversity. The performance hierarchy v2 (64k×1) > v5 (16k×4) > v6 (8k×8) demonstrates that scaling the number of unique tasks is more effective than increasing the number of solutions per task. When computational budget is fixed, expanding task diversity is more efficient for improving generalization, as it exposes the model to broader algorithmic scenarios.

Synthetic Data Offers a Superior Alternative. We conduct a head-to-head comparison with OpenCodeReasoning (Ahmad et al., 2025), the largest reasoning dataset built upon real-world tasks. We train our dataset and OpenCodeReasoning using the same number of training tokens. The results are shown in Table 2. Our proposed dataset yields a 6.7-point improvement after SFT, with most gains coming from the medium and hard splits. These results demonstrate that fully synthetic data is not only a viable substitute for real-world data but also potentially offers a higher upper bound for scaling code reasoning. Moreover, training on fully synthetic data significantly reduces data contamination risks compared to models trained on real-world datasets (Appendix G).

#### 4.3. RL Experiments and Analysis

Our investigation of the RL stage uncovers the following insights into its role and behavior:

RL as a Critical Performance Booster. RL fine-tuning is not merely a minor refinement but a critical optimization stage. As shown in Table 1, when applied to a converged SFT model using only code data, it yields a substantial 4.6% absolute gain in average pass-rate. This highlights RL’s unique capability to explore policy beyond the distribution of the initial supervised dataset.

[Figure 119]

Figure 4. Reward dynamics of weak and strong SFT models.

“Good-gets-Better” Principle. RL performance is tightly coupled to the strength of the SFT initializer. Using two SFT models trained on similar data distributions but with different LiveCodeBench scores as starting points, we observe in Figure 4 that, under identical RL settings, the stronger initializer consistently attains higher rewards.

A stronger SFT foundation enables the model to explore a more promising policy space and achieve a higher performance ceiling. This underscores the importance of a high-quality initial model as a prerequisite for effective RL.

RL’s Resilience to Noisy Supervision. Contrary to the common assumption that RL requires pristine reward signals, our experiments reveal a resilience to data imperfections during RL. The model also effectively benefits from synthetic test cases, suggesting that RL can be successfully deployed in scenarios with large-scale but imperfect feedback (Wang et al., 2020; Lv et al., 2025), significantly lowering the barrier to code RL data collection.

### 5. Ablation Study

Despite the strong performance of X-Coder, the determinants of high-quality synthetic data for SFT remain insufficiently understood. To elucidate these factors, we conduct a comprehensive ablation along four axes: (i) the effect of the proposed dual-verification strategy; (ii) the impact of distinct thinking types in solutions; (iii) a headto-head comparison of tasks produced by the proposed synthesis versus those from open-source synthetic datasets; and (iv) data-selection strategies to identify patterns that shape downstream performance. Additionally, ablations on task styles and test generation strategies are provided in Appendix B.2.3 and Appendix D.2, respectively.

Dual-verification is Critical for Quality. We employ a dual-verification strategy to mitigate noise in stochastic sampling. Table 3 confirms its efficacy: training on verified solutions (64k tasks, Qwen2.5-Coder-7B-Instruct) significantly outperforms raw solutions. However, verifying 200k samples requires 1.6M CoT trajectories and 24M executions.

Table 3. Raw vs. verified solutions on LCB v5. Dataset Size Verify LCB v5

Raw Solution 64k ✗ 47.0 Verified Solution 64k ✓ 53.4 (+6.4)

Table 4. Long CoT vs. Short CoT.

35

|Ratio Diffic|nale-based Selec ulty-based Selec|tion tion| | |
|---|---|---|---|---|
|Rand|om Selection| | | |
| | | | | |
| | | | | |

###### Epoch LCB v5 LCB v6

Pass@1onLCBv5(%)

30

3 35.0 29.3 8 43.1 37.6 ∆ +8.1 +8.3

Short-CoT

25

3 42.9 36.0 8 60.3 53.5 ∆ +17.4 +17.5

Long-CoT

20

15

0.5 1 1.5 2 2.5 3

This overhead establishes a clear trade-off. Although prior work (Li et al., 2025; Gandhi et al., 2025) indicates that models can learn effectively from unverified long-CoT data, making raw-solution training a more resource-efficient alternative, we contend that the significant cost of verification is justified to achieve the higher performance ceiling required for expert-level reasoning. Detailed validation effectiveness analysis is provided in Appendix E.3.

Epoch

Figure 5. Comparison of data selection.

programming—a domain characterized by intricate logic and corner cases—the feature-based evolution strategy is more effective than concept composition at generating highquality, diverse, and rigorous training data.

Long Reasoning Chains are More Valuable but Converge Slower during Training. The length of CoT proves to be a critical factor for performance, with longer CoTs yielding superior results despite higher training costs. To demonstrate this, we compare the Qwen2.5-Coder-7B-Instruct trained on solutions generated by DeepSeek-R1-0528 (Long-CoT) and Qwen3-235B-A22B-Instruct-2507 (Short-CoT) for an identical set of tasks (200k).

Reasoning-Intensive Tasks are More Valuable. To investigate data utilization efficiency, we explore task selection strategies for competitive programming. Specifically, we evaluate three approaches: (1) difficulty-based selection, where GPT-4o-2411 assigns discrete difficulty scores to tasks, simulating the Codeforces rating system; (2) rationalebased selection, where DeepSeek-R1-0528 generates CoT reasoning for each task, and tasks that elicit longer reasoning traces are prioritized; and (3) random selection as a baseline. Each strategy independently samples a 50k-task subset from a 200k-task pool. Solutions are generated by Qwen3-235B-A22B-Instruct-2507, and models were trained for three epochs with a 16k context length.

As shown in Table 4, the long-CoT approach achieves a 17.2% absolute gain. This substantial improvement justifies the increased computational demand, which manifests as a slower convergence requiring 8–10 epochs compared to the 2–3 epochs needed for short-CoT data.

Table 5. Synthetic Data Comparison. Method Evolution Adapted LCB v5

SelfCodeAlign-10k ✗ ✓ 27.1 Ours-10k ✓ ✓ 31.7 EpiCoder-64k ✓ ✗ 25.4 Ours-64k ✓ ✓ 46.3

As shown in Figure 5, tasks that induce longer CoT are regarded as more valuable training data for competitive programming, as they demand deeper reasoning and are potentially more challenging. This finding provides a practical heuristic for efficient data pruning in resource-constrained training scenarios.

Domain-Specific Evolution is Essential for Synthesis. As presented in Table 5, we compare the proposed strategy against two baselines to validate our methodological choices. First, compared to EpiCoder (using generalpurpose prompts), the domain-adapted synthesis achieves a substantial 20.9% improvement. Since this method is built upon EpiCoder’s framework but injects competitive programming-oriented feature extraction and evolution, this gain directly validates the necessity of domain expertise for synthesizing high-complexity tasks. General-purpose synthesis is insufficient for expert-level domains. Second, compared to SelfCodeAlign (Wei et al., 2024) (where we applied the same domain adaptation), the proposed method still yields a 4.6% gain. This suggests that for competitive

### 6. Discussion

Beyond metrics, understanding the mechanisms of failure and success is crucial for advancing code reasoning. In this section, we conduct a granular analysis of the model’s limitations and behavioral patterns. We examine the distribution of error types to pinpoint persistent bottlenecks, revealing an interplay between task difficulty, reasoning length, and success rates. Furthermore, we investigate the characteristics emerging from each training stage, covering the capability distillation in SFT as well as the test-time scaling behaviors and potential reward hacking risks in RL.

Reasoning Capability Remains the Primary Bottleneck. We classify failure cases into seven types: Wrong Answer

- Table 6. Distribution of failure cases for 16 rollouts on LiveCodeBench v5.

Error Type Qwen2.5-Coder-7B-Instruct Qwen3-8B X-Coder-7B-SFT X-Coder-7B

Wrong Answer 194.6 ± 10.7 87.1 ± 4.6 69.6 ± 3.7 67.9 ± 4.9 No Code Block 6.5 ± 8.2 7.7 ± 1.2 21.9 ± 3.7 11.8 ± 3.9 Time Limit Exceeded 18.1 ± 4.1 21.8 ± 3.8 13.7 ± 3.3 11.5 ± 2.6 Memory Limit Exceeded 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 0.17 ± 0.4 Incomplete Code Block 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 1.0 ± 0.8 Signature Mismatch 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 1.0 ± 0.8 Syntax Error 0.0 ± 0.0 0.0 ± 0.0 0.0 ± 0.0 8.3 ± 2.2

- Table 7. Performance analysis categorized by reasoning token length.

###### Token Range Total Passed Easy Medium Hard

0–10k 79 76 46/47 (97.9%) 22/24 (91.7%) 8/8 (100.0%) 10k–20k 93 68 14/15 (93.3%) 30/35 (85.7%) 24/43 (55.8%) 20k–32k 96 25 1/1 (100.0%) 11/27 (40.7%) 13/68 (19.1%)

Total 268 169 61/63 (96.8%) 63/86 (73.3%) 45/119 (37.8%)

(output mismatches the expected answer), Time Limit Exceeded, Memory Limit Exceeded, No Code Block Generated (truncated due to heavy reasoning before the final code is generated), Incomplete Code Block (partial code without closure), Function Signature Mismatch (incorrect function definition), and Syntax Error (complete code with syntax issues). The error distribution in Table 6 indicates that the primary bottleneck lies in reasoning capability, with most errors stemming from wrong answers. Two other major failure categories are No Code Block Generated and Time Limit Exceeded (TLE). Inspection of no-code samples reveals that all exceeded the 32k context window, resulting in truncated reasoning. The prevalent TLE errors further emphasize the necessity for Code LLMs to prioritize execution efficiency.

a significant chained relationship among problem difficulty, reasoning length, and pass rate: problem difficulty is positively correlated with reasoning length, while reasoning length is strongly negatively correlated with pass rate. This mediation pattern can be summarized as higher difficulty → longer reasoning length → lower pass rate.

Synthetic Data Improves Test-time Scaling Efficiency. We compare the pass@k performance of Qwen2.5-Coder7B-Instruct, Qwen3-8B, X-Coder-7B-SFT, and X-Coder7B in Figure 6. X-Coder-7B outperforms its foundation model by 51.3 points in pass@16, and matches Qwen3-8B with 8× fewer rollouts. Moreover, X-Coder shows a larger gap between pass@1 and pass@16 compared to Qwen3-8B (19.2 vs. 13.8), indicating greater diversity in the reasoning patterns it can explore. Although RL models typically begin with higher initial performance than the SFT model, we observe that the gap does not expand within 16 rollouts, suggesting that RL improves pass@1 but might not escape its starting point largely (Wu et al., 2025).

Longer Reasoning Reflects Higher Task Difficulty. Table 7 shows that the pass rate decreases sharply as reasoning token length increases, exhibiting a clear downward trend. This finding runs counter to the intuitive expectation that greater test-time token usage reflects deeper reasoning and should therefore yield higher accuracy. Instead, we observe

Cognitive Behaviors Distilled from SFT, Strategic Exploitation Emerges in RL. Post-SFT models exhibit emergent cognitive behaviors like planning and verification (Appendix H.1), suggesting these capabilities are distilled directly from the teacher. In later RL stages, alongside performance gains, we also observe occasional instances of strategic exploitation, where the model games edge cases for partial rewards (Appendix H.3). We also observe persistent inefficiencies, including context-induced premature termination and cross-lingual hallucinations (e.g., translating memorized C++ solutions), as detailed in Appendix H.2.

[Figure 120]

79.1

79.5 81.5

77

76

73.6

75

79.1

75

4x fewer

8x fewer

77

70

73.6

67.9

69.7

69.7

67.9

62.3

60.3

66.5

66.5

63.7

63.7

Pass@K(%)

Pass@K(%)

60.3

60.4

55

X-Coder-7B

60.4

55

55.9

X-Coder-7B-SFT Qwen3-8B

55.9

X-Coder-SFT Qwen3-8B

Qwen2.5-Coder-7B-Instruct

Qwen2.5-Coder-7B-Instruct

35

30.2

27.3

35

24

30.2

27.3

20.5

16.7

24

20.5

15

16.7

0 2 4 6 8 10 12 14 16

15

Number of Rollouts

0 2 4 6 8 10 12 14 16

Figure 6. Test-time performance.

Number of Rollouts

### 7. Conclusion

manually-crafted benchmark for evaluating llms on classlevel code generation. CoRR, abs/2308.01861, 2023. doi: 10.48550/ARXIV.2308.01861. URL https://doi.

In this paper, we explore a fully synthetic approach to competitive programming, demonstrating that rigorously verified synthetic tasks, solutions, and tests can train large reasoning models to achieve significant performance gains. Our results suggest that such synthesis offers a viable alternative to the scarcity of real-world data. Furthermore, we distill critical insights into code-centric SFT-then-RL training, shedding light on synthetic data scaling, the necessity of domain-specific evolution, and the role of code-centric reinforcement learning. We plan to release related resources.

org/10.48550/arXiv.2308.01861.

Fu, W., Gao, J., Shen, X., Zhu, C., Mei, Z., He, C., Xu, S., Wei, G., Mei, J., Wang, J., Yang, T., Yuan, B., and Wu, Y. Areal: A large-scale asynchronous reinforcement learning system for language reasoning, 2025. URL https:// arxiv.org/abs/2505.24298.

Gandhi, K., Chakravarthy, A. K., Singh, A., Lile, N., and Goodman, N. Cognitive behaviors that enable selfimproving reasoners, or, four habits of highly effective STars. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum? id=QGJ9ttXLTy.

### References

Ahmad, W. U., Narenthiran, S., Majumdar, S., Ficek, A., Jain, S., Huang, J., Noroozi, V., and Ginsburg, B. Opencodereasoning: Advancing data distillation for competitive coding, 2025. URL https://arxiv.org/abs/ 2504.01943.

Guha, E., Marten, R., Keh, S., Raoof, N., Smyrnis, G., Bansal, H., Nezhurina, M., Mercat, J., Vu, T., Sprague, Z., Suvarna, A., Feuer, B., Chen, L., Khan, Z., Frankel, E., Grover, S., Choi, C., Muennighoff, N., Su, S., Zhao, W., Yang, J., Pimpalgaonkar, S., Sharma, K., Ji, C. C.-J., Deng, Y., Pratt, S., Ramanujan, V., Saad-Falcon, J., Li, J., Dave, A., Albalak, A., Arora, K., Wulfe, B., Hegde, C., Durrett, G., Oh, S., Bansal, M., Gabriel, S., Grover, A., Chang, K.-W., Shankar, V., Gokaslan, A., Merrill, M. A., Hashimoto, T., Choi, Y., Jitsev, J., Heckel, R., Sathiamoorthy, M., Dimakis, A. G., and Schmidt, L. Openthoughts: Data recipes for reasoning models, 2025. URL https://arxiv.org/abs/2506.04178.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.

07732.

Chen, M., Tworek, J., Jun, H., and et al., Q. Y. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 4299– 4307, 2017. URL https://proceedings.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., Li, S., Zeng, L., Wei, T., Cheng, C., An, B., Liu, Y., and Zhou, Y. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

neurips.cc/paper/2017/hash/ d5e2c0adad503c91f91df240d0cd4e49-Abstract. html.

Hendrycks, D., Basart, S., Kadavath, S., and et al. Measuring coding challenge competence with APPS. In NeurIPS Datasets and Benchmarks, 2021.

Dai, N., Wu, Z., Zheng, R., Wei, Z., Shi, W., Jin, X., Liu, G., Dun, C., Huang, L., and Yan, L. Process supervision-guided policy optimization for code generation. CoRR, abs/2410.17621, 2024. doi: 10.48550/ ARXIV.2410.17621. URL https://doi.org/10.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github. com/huggingface/open-r1.

48550/arXiv.2410.17621.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Lu, K., Dang, K., Fan, Y., Zhang, Y., Yang, A., Men, R., Huang, F., Zheng, B., Miao, Y., Quan, S., Feng, Y., Ren, X., Ren, X., Zhou, J., and Lin, J. Qwen2.5-coder technical report, 2024. URL https://arxiv.org/abs/2409.12186.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Du, X., Liu, M., Wang, K., Wang, H., Liu, J., Chen, Y., Feng, J., Sha, C., Peng, X., and Lou, Y. Classeval: A

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/2403.07974.

Labs, B. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation. https://www.bespokelabs.ai/blog/bespoke-stratosthe-unreasonable-effectiveness-of-reasoning-distillation,

2025. Accessed: 2025-01-22.

Lai, Y., Li, C., Wang, Y., Zhang, T., Zhong, R., Zettlemoyer, L., Yih, W., Fried, D., Wang, S. I., and Yu, T. DS-1000: A natural and reliable benchmark for data science code generation. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 18319–18345. PMLR, 2023. URL https://proceedings.mlr.

press/v202/lai23b.html.

Li, D., Cao, S., Griggs, T., Liu, S., Mo, X., Tang, E., Hegde, S., Hakhamaneshi, K., Patil, S. G., Zaharia, M., Gonzalez, J. E., and Stoica, I. Llms can easily learn to reason from demonstrations structure, not content, is what matters!, 2025. URL https://arxiv.org/abs/ 2502.07374.

Li, R., Fu, J., Zhang, B.-W., Huang, T., Sun, Z., Lyu, C., Liu, G., Jin, Z., and Li, G. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago,

- A., Hubert, T., Choy, P., de Masson d’Autume, C., Babuschkin, I., Chen, X., Huang, P.-S., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz, D., Sutherland Robson, E., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competition-level code generation with alphacode. arXiv preprint arXiv:2203.07814, 2022.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023a. Curran Associates Inc.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirtyseventh Conference on Neural Information Processing Systems, 2023b. URL https://openreview.net/ forum?id=1qvx610Cu7.

- Liu, Y., Zhang, L. L., Zhu, Y., Dong, B., Zhou, X., Shang, N., Yang, F., and Yang, M. rstar-coder: Scaling competitive code reasoning with a large-scale verified dataset, 2025a. URL https://arxiv.org/abs/

- 2505.21297.

Liu, Z., Yang, Z., Chen, Y., Lee, C., Shoeybi, M., Catanzaro, B., and Ping, W. Acereason-nemotron 1.1: Advancing math and code reasoning through sft and rl synergy, 2025b. URL https://arxiv.org/abs/

- 2506.13284.

Luo, M., Tan, S., Huang, R., Patel, A., Ariyak, A., Wu, Q., Shi, X., Xin, R., Cai, C., Weber, M., Zhang, C., Li, L. E., Popa, R. A., and Stoica, I. Deepcoder: A fully open-source 14b coder at o3-mini level, 2025. Notion Blog.

Luo, Z., Xu, C., Zhao, P., Sun, Q., Geng, X., Hu, W., Tao, C., Ma, J., Lin, Q., and Jiang, D. Wizardcoder: Empowering code large language models with evol-instruct. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=UnUwSIgK5W.

Lv, A., Xie, R., Sun, X., Kang, Z., and Yan, R. The climb carves wisdom deeper than the summit: On the noisy rewards in learning to reason, 2025. URL https:// arxiv.org/abs/2505.22653.

OpenAI. Openai o3-mini: Pushing the frontier of cost-effective reasoning. https://openai.com/ index/openai-o3-mini/, 2025. Accessed: 202509-11.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https:

//arxiv.org/abs/2412.15115.

Schulman, J., Moritz, P., Levine, S., Jordan, M. I., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations (ICLR), 2016.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/ 1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024a.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo,

- D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024b. URL https://arxiv.org/abs/2402.03300.

Su, Z., Pan, L., Bai, X., Liu, D., Dong, G., Huang, J., Hu, W., Zhang, F., Gai, K., and Zhou, G. Klear-reasoner: Advancing reasoning capability via gradient-preserving clipping policy optimization, 2025. URL https:// arxiv.org/abs/2508.07629.

Sun, Q., Gong, J., Li, L., Guo, Q., and Yuan, F. Codeevo: Interaction-driven synthesis of code-centric data through hybrid and iterative feedback, 2025. URL https:// arxiv.org/abs/2507.22080.

Wang, J., Liu, Y., and Li, B. Reinforcement learning with perturbed rewards. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 6202–6209, 2020.

- Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net,

2023. URL https://openreview.net/forum? id=1PL1NIMMrw.

- Wang, Y., Li, H., Zhang, X., Wu, J., Liu, X., Hu, W., Guo,
- Z., Huang, Y., Xin, Y., Yang, Y., et al. Epicoder: Encompassing diversity and complexity in code generation. arXiv preprint arXiv:2501.04694, 2025.

Wei, Y., Cassano, F., Liu, J., Ding, Y., Jain, N., Mueller, Z., de Vries, H., Werra, L. V., Guha, A., and ZHANG, L. Selfcodealign: Self-alignment for code generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=xXRnUU7xTL.

Wei, Y., Duchenne, O., Copet, J., Carbonneaux, Q., Zhang, L., Fried, D., Synnaeve, G., Singh, R., and Wang, S. I. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution, 2025. URL https: //arxiv.org/abs/2502.18449.

Wu, F., Xuan, W., Lu, X., Harchaoui, Z., and Choi, Y. The invisible leash: Why rlvr may not escape its origin, 2025. URL https://arxiv.org/abs/2507.14843.

Xia, C. S., Deng, Y., and ZHANG, L. Top leaderboard ranking = top coding proficiency, always? evoeval: Evolving coding benchmarks via LLM. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=zZa7Ke7WAJ.

Xiaomi, L.-C., :, Xia, B., Shen, B., Cici, Zhu, D., Zhang, D., Wang, G., Zhang, H., Liu, H., Xiao, J., Dong, J., Zhao, L., Li, P., Wang, P., Yu, S., Chen, S., Wang, W., Ma, W., Deng, X., Huang, Y., Song, Y., Jiang, Z., Ye, B., Cai, C., He, C., Zhang, D., Zhang, D., Wang, G., Tian, H., Zhao, H., Qu, H., Xu, H., Shi, J., Bao, K., Fang, K., Zhou, K., Zhou, K., Li, L., Zhu, M., Chen, N., Wang, Q., Liu, S., Li, S., Gu, S., Ren, S., Liu, S., Deng, S., Zhuang, W., Lv, W., Yang, W., Zhang, X., Yong, X., Zhang, X., Song, X., Xu, X., Wang, X., Yan, Y., Tu, Y., Tian, Y., Wang, Y., Yu, Y., Lin, Z., Song, Z., and Yue, Z. Mimo: Unlocking the reasoning potential of language model – from pretraining to posttraining, 2025. URL https://arxiv.org/abs/2505.07608.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. WizardLM: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=CfXh93NDgH.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., Lu, K., Xue, M., Lin, R., Liu, T., Ren, X., and Zhang, Z. Qwen2.5-math technical report: Toward mathematical expert model via selfimprovement, 2024. URL https://arxiv.org/ abs/2409.12122.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang,

- J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang,
- K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang,

- X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan,
- Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., and Qiu, Z. Qwen3 technical report, 2025. URL https: //arxiv.org/abs/2505.09388.

## Appendix

- • Section A Training and Evaluation: Reward functions, training costs, and distributed infrastructure.
- • Section B Novel Task Synthesis: Feature extraction, task generation process, difficulty and diversity analysis.
- • Section C Solution Generation and Quality Assurance: Solution validation procedures and SFT dataset statistics.
- • Section D Test Case Generation: Comparison of prompting-based vs. tool-based test generation methods.
- • Section E Dual-verification: Algorithm description, weighting criteria, and error rate analysis.
- • Section F Generality: Cross-model and cross-benchmark evaluation results.
- • Section G Data Leakage Analysis: Contamination detection and prevention measures.
- • Section H Case Study: Success and failure case analysis, reward hacking behaviors.

### A. Training and Evaluation

#### A.1. SFT-then-RL Training

Supervised Fine-tuning. Given a dataset of task–solution pairs D = {(xi,yi)}Ni=1, the model with parameters θ is trained by minimizing the negative log-likelihood (NLL) of the target solution y conditioned on the task x:

JSFT(θ) = −E(x,y)∼D

|y|

log πθ yt | x,y<t . (3)

t=1

The loss is applied over full long-CoT trajectories, including both reasoning steps and final code, enabling the model to imitate not only the solutions but also the underlying reasoning patterns.

Reinforcement Learning. Proximal Policy Optimization (PPO) (Schulman et al., 2017) is a widely adopted policy gradient method in Reinforcement Learning from Human Feedback (RLHF) (Christiano et al., 2017) for LLM due to its balance between exploration and exploitation and its empirical robustness. The method optimizes a policy πθ by using a clipped surrogate objective to limit policy divergence, incorporating a value function to estimate expected rewards, and an entropy term to encourage exploration. The overall objective function for PPO is designed to maximize the policy performance while maintaining stability, and it is typically formulated as minimizing the following:

πθ(a|s) πθ

πθ(a|s) πθ

A(s,a), clip

,1 − ϵ,1 + ϵ A(s,a) (4)

JPPO(θ) = Es∼P(S),a∼π

θ(a|s) min

(a|s)

(a|s)

old

old

where the expectation is computed over states s (drawn from distribution P(S)) and actions a (sampled from the current policy πθ(a | s) ), combining the minimum of two terms: (1) the product of the probability ratio π

θ(a|s)

πθold(a|s) and the advantage function A(s,a), where the advantage function quantifies the relative benefit of taking action a in state s; and (2) the same product but with the probability ratio clipped to the interval [1 − ϵ,1 + ϵ]. Here, ϵ is a hyperparameter governing the magnitude of policy updates. This clipping mechanism effectively constrains excessive policy changes, thereby enhancing training stability.

However, its application to LLMs encounters significant challenges, including substantial computational overhead from maintaining a critic network, which increases memory usage and training time for models with billions of parameters. Additionally, training stability can be undermined by inaccurate value function estimates or suboptimal tuning of Generalized Advantage Estimation (GAE) (Schulman et al., 2016) parameters, issues that become more pronounced as LLMs scale in size. To address these limitations, Group Relative Policy Optimization (GRPO) (Shao et al., 2024b) has emerged as an efficient alternative. By eliminating the critic network, GRPO reduces computational and memory demands, estimating advantages directly from rewards of multiple rollouts to the same prompt, thus leveraging the comparative nature of reward

models and offering a scalable solution for LLM training. The GRPO objective function is mathematically formulated as an averaged composite expression across multiple rollouts, incorporating policy ratio optimization and KL regularization:

|ai|

G

1 G

1 |ai|

min ρi,tAˆi,t, clip ρi,t,1 − ϵ,1 + ϵ A ˆi,t − βDKL[πθ ∥ πref] (5)

JGRPO(θ) =

t=1

i=1

πθ(ai,t|s,ai,<t) πθ

where ρi,t =

denotes the probability ratio of the old and new strategies. G is the number of rollouts per

(ai,t|s,ai,<t)

old

prompt, |ai| denotes the length of the i-th action sequence, Aˆi,t estimates the advantage of action ai,t at timestep t. The clipping is analogous to PPO, and β penalizes deviations from πref via the KL-divergence term. The objective averages across rollouts and timesteps, combining a clipped probability ratio (to stabilize updates while leveraging advantage signals) with a KL penalty to balance policy improvement against alignment with the reference policy. This dual mechanism ensures controlled optimization by restricting drastic policy shifts while maintaining coherence with prior behavior.

#### A.2. Reward Function.

We remove formatting rewards (e.g., enforcing “think” tags), as the SFT model already follows the format, allowing the policy to focus on passing test cases. Given a rollout, the reward R is practiced as:

 

−2, if no code is extracted or the code fails to compile, 0, if the code compiles but passes no test cases, 5.0 × #passed #total

(6)

R =



, otherwise.

We adopt a continuous reward setting, as it provides denser supervision than the all-or-nothing alternative and leads to faster convergence (Wei et al., 2025; Dai et al., 2024).

#### A.3. Training Dynamics.

- As shown in Figure 7 and Figure 8, we present the SFT training curves (loss and token accuracy). Figure 9 and Figure 10 illustrate the RL training curves (reward and entropy).

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

1.0

0.8

TrainLoss

0.6

0.4

0.2

0 2000 4000 6000 8000 10000 12000

Step

Figure 7. Training loss of SFT.

0.95

0.90

TrainTokenAcc

0.85

0.80

0.75

0.70

0 2000 4000 6000 8000 10000 12000

Step

2.0

1.9

1.8

Reward

1.7

1.6

1.5

1.4

0 50 100 150 200 250

Step

Figure 8. Training token accuracy of SFT.

Figure 9. Training reward of RL.

0.26

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.24

0.22

0.20

Entropy

0.18

0.16

0.14

0 50 100 150 200 250

Step

Figure 10. Training entropy of RL.

#### A.4. Training Configs and Costs

For SFT, we use a learning rate of 5e-5 with a global batch size of 128 for 8 training epochs. For RL, the policy models are updated with a global batch size of 128 and a consistent learning rate of 7e-5, without applying the KL-divergence constraint to the starter model, and employ a rollout temperature of 1.0 with 8 rollouts to encourage exploration.

Training large reasoning models incurs significant costs compared to standard (eg. short-CoT) instruction models. In the SFT stage, the dominant overhead stems from longer sequence lengths and the need for more update epochs, which together lead to several times more compute consumption than training non-reasoning counterparts. In the RL stage, the major bottleneck lies in generating multiple rollouts for each problem used for GRPO-algorithm.

Concretely, training X-Coder on Qwen2.5-Coder-7B-Instruct required 128 H20 Enterprise (96 GB) GPUs for 220 hours during SFT, and 32 H200 (141 GB) GPUs for 7 days to complete 270 update steps during RL. We are going to make

X-Coder a readily accessible, open-source model, enabling the community to benefit from its capabilities without having to bear the training costs.

#### A.5. A Distributed Framework for Automated Code Verification

To provide a robust and scalable solution for code validation, we develop a distributed arbitration framework inspired by open-source repository implementations5. The system is based on a microservice architecture, comprising a FastAPI-based asynchronous API Gateway, a pool of code execution workers in the sandbox and a central Redis instance. Redis serves as a high-performance message broker and state manager, effectively decoupling the client-facing gateway from the backend computational workers. This architectural choice facilitates independent scaling, deployment, and enhances the overall resilience of the system. Based on this evaluation framework, we implemented highly concurrent code testing during RL training. We used batching when submitting tasks to the Redis server to achieve high concurrency even with low request rates. This process required the server to distribute all test tasks to different workers, utilizing the CPU power of all participating machines. Figure 11 shows the system diagram of the framework.

The framework’s efficacy is derived from its strategic implementation of Redis data structures. Task distribution is managed by a Sorted Set, which functions as a time-prioritized FIFO queue; submissions are added with a timestamp score via ZADD, and workers atomically retrieve the next task using BZPOPMIN. This approach ensures ordered processing and prevents race conditions. For result transmission, each task is assigned a dedicated List, to which a worker pushes the outcome using RPUSH. The API Gateway then performs a blocking pop (BLPOP) on this unique list to retrieve the corresponding result efficiently. Furthermore, worker health and presence are monitored using String keys with a Time-To-Live (TTL). Workers periodically refresh their key’s TTL as a heartbeat, enabling the system to automatically detect and de-register unresponsive nodes.

The resulting system exhibits several key advantages. The asynchronous, in-memory nature of its core components yields high throughput and low-latency performance. Its design is inherently scalable, as the stateless worker pool can be expanded horizontally to meet computational demand, while native support for Redis Cluster addresses data-tier bottlenecks. Finally, the framework’s reliability is bolstered by the atomicity of Redis operations and the integrated fault-detection mechanism, ensuring dependable and consistent code verification.

FastAPI Web (API Gateway)

Redis Queue (Message Broker)

Worker Pool (Code Executors)

|Request Handling|
|---|

|Task & Result Queues|
|---|

|Sandbox Execution|
|---|

Figure 11. The distributed architecture of the code verification framework.

#### A.6. Baselines

We compare the X-Coder with three categories of baselines: (1) SFT model, e.g., Bespoke-Stratos, OlympicCoder, OCR-Qwen-Instruct, OpenThinker3, Qwen3-8B, and rStar-Coder; (2) RL model, including Skywork-OR1, DeepCoder-14BPreview, and AReal-boba²-14B; (3) SFT-then-RL model, such as AceReason1.1, Klear-Reasoner, and MiMo-RL.

5https://github.com/0xWJ/code-judge.git

### B. Novel Task Synthesis

Building on EpiCoder, which synthesizes programming tasks through feature-based combinations, we introduce three key improvements to generate more diverse and complex instructions.

First, rather than relying on broad feature definitions, we explicitly extract and evolve competition-related features from 10,000 question–solution pairs in TACO (Li et al., 2023) using GPT-4o-0513 (Appendix B.1). Second, we adopt a two-stage process: selecting mutually consistent features and then formulating challenging, hint-free tasks (Appendix B.2). Third, we extend the synthesis method to support multi-style generation, covering CodeForces-style tasks (rich narratives with standard I/O), LeetCode-style tasks (starter code with fixed signatures), and AtCoder-style tasks (concise specifications), thereby enhancing task diversity. In Appendix B.3, we further estimate the difficulty of synthesized problems using a trained discriminator.

#### B.1. Feature Extraction and Evolution

While EpiCoder extracts general-purpose features from raw corpus, we explicitly extract and evolve competitive programming-related feature. Specifically, we design multiple aspect of features that highly relates to competitive programming, such as data structure, algorithm, mathematical, etc.

We improve the extraction process to guide the LLM to focus on competitive programming–related concepts, as follows:

|Extract features from the provided problem and solution code related to algorithmic programming, competitive programming, Leetcode, and Codeforces, following the requirements for each category below, formatted in JSON structure. Responses in the following categories should be concise and organized in a JSON format surrounded with <begin> and <end>. Categories may include nested structures if applicable. Here is an example of the expected format: <begin>{<br><br>"programming language": [ "Python"<br><br>], "problem type": [<br><br>"graph traversal"<br><br>], "algorithm": {<br><br>"graph algorithms":[ "Dijkstra’s algorithm", "DFS", "BFS"<br><br>], "dynamic programming":[<br><br>"Longest Increasing Subsequence", "Knapsack Problem"<br><br>] },<br><br>... "implementation logic":["recursive", "iterative"]<br><br>}<end> Categories to extract:<br><br>1. Programming Language: Note the specific programming language used. Example: ["Python", "C++"].<br><br>2. Problem Type: Outline the type of problem the code is solving. Example: ["graph traversal", "sorting", "dynamic programming"].<br><br>3. Algorithm: Identify the specific algorithm or method being used in the code. This category can include the following subcategories:<br><br>3.1 Graph Algorithms: Specify graph algorithms used. Example: ["Dijkstra’s algorithm", "DFS", "BFS"].<br><br>3.2 Sorting Algorithms: Specify sorting algorithms used. Example: ["QuickSort", "MergeSort"].<br><br><br>...<br><br>4. Data Structures: Describe the primary data structures utilized. Example: ["array", "graph", "tree", "heap"].<br><br>5. Implementation Logic: Describe the implementation logic. Example: ["iterative", "recursive", "bit manipulation"].<br><br>6. Complexity Analysis: Provide time and space complexity of the code if available. Example: ["Time Complexity: O(n log n)", "Space Complexity: O(n)"]<br><br>7. Optimization Techniques: Specify any optimizations applied. Example: ["memoization", "greedy approaches", "bitwise operations"].<br><br><br>... Extract as many features as possible and try not to let a feature appear in multiple categories at the same time.<br><br>|
|---|

Then we increase the diversity and complexity through evolution along both the breadth and depth dimensions. For example, along the breadth dimension, given an extracted feature such as quicksort, the LLM may evolve new features like bubble sort, even if they were not originally extracted. Along the depth dimension, a concept such as prefix sum can evolve into more advanced variants like difference array or Fenwick tree, reflecting increasing levels of abstraction and difficulty. The overall evolution process is illustrated below.

|Feature Tree Evolution Task: You are provided with a feature tree represented as a nested JSON structure. Each node in this tree represents a feature or a sub-feature of competitive algorithm programming, with the leaves being the most specific features. Your task is to expand this feature tree both in depth and breadth. Depth expansion means adding more specific sub-features to existing leaves. Breadth expansion means adding more sibling features at the current levels.<br><br>Here are some explanations of the features: {explanations} The input feature tree will be provided in JSON format, and your output should be a JSON structure that represents the expanded feature tree. Output Format:<br><br>- Expanded Feature Tree: Provide the expanded feature tree as a JSON structure. Surround the json with <begin> and <end>.<br><br>Input Feature Tree Example: {<br><br>"algorithm": {<br><br>"sorting": ["quick sort", "merge sort"], "tree traversal": ["in-order traversal"]<br><br>},<br><br>...<br><br>} Expanded Feature Tree Example: <begin> {<br><br>"algorithm": {<br><br>"sorting": { "quick sort": ["3-way quick sort", "dual-pivot quick sort"], "merge sort": ["top-down merge sort", "bottom-up merge sort"], "heap sort":[]<br><br>}, "tree traversal": {<br><br>"in-order traversal": ["recursive in-order traversal", "iterative in-order traversal"], "pre-order traversal":[], "post-order traversal":[], "level-order traversal":[],<br><br>} },<br><br>...<br><br>} <end> Constraints:<br><br>1. For breadth expansion, add at least 2 new sibling features to each existing node.<br><br>2. For deep expansion, you need to add new sub-features to it, provided that you think the current leaf node has a more fine-grained feature.<br><br>3. Focus on generating new and innovative features that are not present in the provided examples.<br><br>4. The features are related to competitive algorithm programming. Please follow the above constraints and expand the feature tree accordingly.<br><br><br>Input: {features} Output: <begin>expanded feature tree<end><br><br>|
|---|

After evolution, we merge features that share common traits into a larger tree, providing a rich pool of features for subsequent task formulation.

- B.1.1. STATISTICS FOR FEATURE EXTRACTION AND EVOLUTION

We present detailed statistics on feature evolution and data filtering to demonstrate how the process expands feature diversity and yields a high-quality 240k dataset. The statistics of feature extracted and evolved as follows.

Table 8. Statistics of Features Extracted and Evolved. The evolution strategy significantly increases feature quantity across all categories. Category Features Extracted Features After Evolution Growth

Algorithm 27,400 176,914 ×6.46 Data Structures 12,353 65,104 ×5.27 Problem Type 14,134 130,293 ×9.22 Implementation Logic 12,419 106,157 ×8.55 Complexity Analysis 16,124 90,016 ×5.58 Optimization Techniques 1,537 14,124 ×9.19

The evolution strategy greatly enhances diversity of features, providing support for generating diverse tasks.

- B.2. Stylized Task Generation for Competitive Programming We design a prompt template to systematically transform extracted features into stylized competitive programming tasks. Input: a sampled feature tree represented in JSON format.

Output: a feature-role tree (JSON), where each node is assigned roles such as core technique, subroutine, or constraint, together with an integration strategy (string) that explains how to combine these features into a coherent problem. To improve instruction-following and task understanding, the template is equipped with a one-shot example that demonstrates how raw features are mapped into roles and integrated into a task.

|You are a professional competitive programming problem setter. Your task consists of three parts:<br><br>Step 1: Tree-Structured Feature Role Explanation Recursively traverse the provided feature tree.<br><br>- For each leaf node, annotate it with a "potential_use" field describing how this feature is typically used in competitive programming problems (e.g., input modeling, optimization, search, handling edge cases, etc.).<br><br>- Internal nodes retain their structure for hierarchy. Output the annotated tree in the same structure, with every leaf node containing its "potential_use".<br><br><br>Step 2: Subtree Selection for Problem Integration Based on your role analysis, select a subtree (tree-structured subset) where all selected leaf features can be naturally integrated into a single, high-quality competitive programming problem.<br><br>- Only include features that contribute meaningfully to the same problem idea.<br><br>- Internal nodes are included only if they have selected children.<br><br>- For each selected leaf, include only its "feature" name and "potential_use".<br><br><br>Step 3: Integration Strategy Briefly describe ("integration_strategy") how the selected features can be integrated together in a single problem, focusing on how their combination enables a meaningful and challenging algorithmic scenario.<br><br><br>Return a JSON object **with exactly this structure** (an example): {{<br><br>"feature_roles_tree": {{ "algorithm": {{ "search algorithm": {{ "binary search": {{ "recursive binary search": {{ "potential_use": "Used for divide-and-conquer searching in sorted structures or answer spaces."<br><br>}}, ...<br><br>}}, "selected_features_tree": {{<br><br>"algorithm": {{ "search algorithm": {{ "binary search": {{<br><br>"recursive binary search": {{ "feature": "recursive binary search", "potential_use": "Used for divide-and-conquer searching in sorted structures or answer spaces."<br><br>}}<br><br>... }}, "integration_strategy": "The problem will require recursive binary search to efficiently search over a sorted<br><br>value space, while bitwise AND operations will be used to filter candidate solutions according to constraints. Their combination allows for a problem that involves searching over sets and optimizing bitwise criteria." }}<br><br>**Available Features (Tree):** {features_json}<br><br>|
|---|

- B.2.1. COMPATIBALE FEATURE SELECTION We present a case to examine how model selects compatible features and combine them. Given a sampled feature tree:

"input_features": { "algorithms": {

"graph_algorithms": {

"shortest_path": [ "Dijkstra’s algorithm", "Floyd-Warshall"

], "network_flow": [

"Ford-Fulkerson", "Edmonds-Karp"

] },

...

LLM pairs each feature with potentially usage to obtain feature tree with role annotation. For example, LLM will anonote feature “rolling hash” as “Compute hash values for sliding windows in constant time”. These annotations help LLM to aggregate these features based on their potentially usage. For above given feature tree, the feature tree with potential usage looks like:

|"feature_roles_tree": { "algorithms": {<br><br>"graph_algorithms": { "shortest_path": { "Dijkstra’s_algorithm": { "potential_use": "Find single-source shortest paths in weighted graphs with non-negative edges"<br><br>}, "Floyd_Warshall": {<br><br>"potential_use": "Compute all-pairs shortest paths with O(nˆ3) complexity" }<br><br>},<br><br>...<br><br>|
|---|

LLM then selects a compatible and consistent subtree that can formulate a self-contained competitive programming problem. For example, LLM keeps the features that can be aggregated into selected feature tree, and concluding with an integration strategy, which displays how to combine these features into a unified problem.

For example, in this case, LLM selects “Dijkstra’s algorithm”, “Edmonds-Karp”, “segment tree”, and “tree DP”, and aims to formulate a problem around “dynamic network optimization”.

|"algorithms": { "graph_algorithms": { "shortest_path": {<br><br>"Dijkstra’s_algorithm": { "feature": "Dijkstra’s algorithm", "potential_use": "Primary pathfinding algorithm"<br><br>}<br><br>}, "network_flow": {<br><br>"Edmonds_Karp": { "feature": "Edmonds-Karp", "potential_use": "Flow computation with guaranteed complexity"<br><br>} }<br><br>}<br><br>}, "data_structures": {<br><br>"tree_structures": {<br><br>"segment_tree": { "feature": "segment tree", "potential_use": "Maintain dynamic edge weights or capacities"<br><br>} }<br><br>},<br><br>... "integration_strategy": "Create a dynamic network optimization problem where Dijkstra’s algorithm finds shortest paths that are used as augmenting paths in a modified Edmonds-Karp flow algorithm. Use segment tree to handle dynamic updates to edge capacities based on flow history. Apply tree DP on the shortest path tree to compute optimal flow distributions. This models a transportation network with time-varying capacities."<br><br>|
|---|

- B.2.2. FROM FEATURE TO STYLIZED TASK

We separate feature selection from task generation, as our initial attempts showed that prompting an LLM to perform both within a single prompt often led it to choose fewer features and produce overly simple problems.

During task generation, LLM receives selected features tree and its integration strategy to formulate stylized task based on prompt recieved. In this instance, our generated Codeforces problem is shown in Figure 12.

The rationale for above two-stage approach is that a single-step approach is less effective. When performing both steps

|Dynamic Transport Renewal<br><br>In the city of Codeland the transportation system is in constant flux. The city has n intersections and m one‐way roads. Each road is characterized by a travel time and an initial capacity representing the maximum number of vehicles that may traverse that road in a day. Due to changing conditions, city engineers periodically adjust road capacities. After every such update, the transport authority recalculates their performance metric in two steps.<br><br>First, they compute the maximum number of vehicles that can be sent from the central depot at intersection 1 to the distribution center at intersection n. To do so they repeatedly select an augmenting path that minimizes the total travel time (using a shortest path computation) among all paths on which every road has positive capacity. They send as many vehicles along that path as allowed by its weakest road and then reduce the capacity of every road on the path by that amount. This process is repeated until no valid path from 1 to n remains.<br><br>Second, using the predecessor structure recorded in the last successful shortest path search (forming a tree rooted at 1), the authority assigns each intersection a reward equal to its travel time from intersection 1 (as computed in that search). They then choose a subset of intersections from this tree such that no intersection and its direct predecessor are both chosen, with the goal to maximize the total reward. (This selection is computed using an optimization on the tree structure.)<br><br>The final performance metric is the sum of the maximum flow (i.e. total number of vehicles sent) and the maximum total reward from the tree selection.<br><br>Your task is to process a series of capacity update queries. Initially the network is given. Then, each query specifies an interval [L, R] (referring to the roads in their input order) and an integer X. For every road whose index is in [L, R], add X to its current capacity. If an update causes an edge's capacity to become negative, set it to 0. After each update, recalculate the maximum flow using the method described above and then compute the optimal reward from the latest shortest path tree. Output the sum of these two values.<br><br>Note that each update is cumulative. Input The first line contains three integers n, m and Q (2 ≤ n ≤ 100, 1 ≤ m ≤ 1000, 1 ≤ Q ≤ 1000) the number of intersections, the number of roads and the number of queries.<br><br>Each of the next m lines describes a road with four integers u, v, t and c (1 ≤ u, v ≤ n, u ≠ v, 0 ≤ t ≤ 10^6, 0 ≤ c ≤ 10^9), meaning that there is a road from intersection u to v with travel time t and initial capacity c. The roads are numbered from 1 to m in the order of appearance.<br><br>Each of the next Q lines contains three integers L, R and X (1 ≤ L ≤ R ≤ m, -10^9 ≤ X ≤ 10^9) meaning that for every road with index in [L, R] you must add X to its current capacity (if a road's capacity becomes negative, treat it as 0).<br><br>Output For each query, output a single integer — the sum of the maximum flow from intersection 1 to n (computed by repeatedly sending flow along the fastest (i.e. minimum travel time) augmenting path) and the maximum reward obtainable from the shortest path tree from the last successful search (computed using the tree-optimization described above).<br><br>Example Input 4 5 2<br><br>1 2 1 5<br>2 4 3 3<br><br>1 3 2 4<br><br>3 2 1 2<br><br>2 4 2 4 2 4 -1 1 3 2<br><br><br><br><br>Output 11 8<br><br>Note After the first update, the capacities of roads with indices 2, 3 and 4 decrease by 1, so they become 2, 3 and 1 respectively while the others remain unchanged. The flow computation proceeds in iterations by first choosing the path 1→2→4 (with travel time 1+3=4) and sending 2 vehicles, then using the path 1→3→2→4 (with travel time 2+1+2=5) to send 1 vehicle, and finally again 1→2→4 to send 3 vehicles. The total maximum flow is 6. In the last successful shortest path search (from the iteration yielding the 3 vehicles), the predecessor tree has intersection 1 as the root with children 2 and 3, and intersection 2 with child 4. With rewards equal to their computed distances from intersection 1, an optimal non-adjacent selection yields a total reward of 5. Their sum is 11.<br><br>After the second update, the capacities of roads with indices 1, 2 and 3 increase by 2. Recomputing the maximum flow now yields a value of 2, while the corresponding shortest path tree results in an optimal reward of 6. The final performance metric is 8.|
|---|

Figure 12. Case for Codeforces-style Problem, featuring rich, imaginable narrative contexts.

Table 9. Comparison between one-step and two-stage generation. Generation Method Score (avg@4)

One-Step (end-to-end) 34.8 Two-Stage (Ours) 40.1 (+5.3)

simultaneously, LLMs tend to oversimplify complex instructions into trivial cases, reducing both diversity and difficulty of the generated task.

To empirically validate this, we generated 32k tasks using the one-step method (feature-tree → task) and using proposed “two-stage” method (feature-tree → sub-tree → task). The SFT results on LiveCodeBench v5 are as Table 9. The 5.3 gain shows that explicit sub-tree selection and integration is significantly helpful for producing high-quality, challenging tasks and justifies the strategy’s modular design.

- B.2.3. ABLATION ON TASK STYLE

We evaluate the effect of task styles (AtCoder, Codeforces, and LeetCode) by synthesizing three corpora of 32k tasks each (8k unique problems with 4 solutions per problem) from identical input features. For each corpus, solutions are generated with DeepSeek-R1-0528 and used to fine-tune the Qwen2.5-Coder-7B-Instruct. Results are shown in Figure 13. Although AtCoder-style tasks yield slightly higher scores, we adopt Codeforces-style as the predominant format in our demonstration dataset (Codeforces : AtCoder : LeetCode = 70 : 15 : 15), reflecting its prominence as the mainstream competitive-programming platform.

45

50

55

Pass@1onLCBv5(%)

40

35

50

30

25

45

20

AtCoder Codeforces

15

40

Ours

10

Raw Verified

LeetCode Epicoder

0

5

35

1400 1800 2200 2600 3000 3400 3800

0 200 400 600 800 1000 1200 1400

2100 2500 2900 3300 3700 4100

Training Step

Training Step Training Step

Figure 13. Task style comparison.

#### B.3. Task Difficulty Estimates

Judging the difficulty of a synthetic task is challenging. To better capture the difficulty distribution of tasks generated by X-Coder, we adopt a classifier-based approach. Specifically, we add a special classification token to Qwen2.5-Coder-14B-Instruct and fine-tune it to predict the Codeforces rating of 6,246 tasks from the CodeContests dataset with annotated ratings, reserving 5% as a validation set. The fine-tuned model achieves 84% classification accuracy on the validation set. We then use this model to estimate the difficulty of 1,000 tasks generated by the proposed method, obtaining a holistic distribution as shown in Table 10.

Table 10. Difficulty distribution of Codeforces-style ratings. “Original” denotes the annotated distribution from CodeContests, and “Test” denotes 1,000 tasks generated by the proposed method.

###### CF Rating Original Test (Ours) Original Share Test Share

1200 623 0 10.0% 0.0% 1400 727 0 11.7% 0.0% 1600 889 0 14.3% 0.0% 1800 840 16 13.5% 1.6% 2000 797 2 12.8% 0.2% 2200 697 47 11.2% 4.7% 2400 665 585 10.7% 58.5% 2600 484 319 7.8% 31.9% 2800 312 12 5.0% 1.2% 3000 233 15 3.7% 1.5% 3200 157 4 2.5% 0.4% 3400 122 0 2.0% 0.0%

Total 6,246 1,000 100% 100%

#### B.4. Task Diversity Estimates

To analyze the diversity of our generated tasks quantitatively, we analyze diversity in the embedding space following the steps below: (i) Embedding: We first embed the tasks into embeddings using jinaai/jina-embeddings-v2-base-code, a specialized coding embedding model. (ii) t-SNE Dimensionality Reduction: We apply t-SNE to reduce the embedded data to 2D space. (iii) Clustering: We perform K-means clustering on the t-SNE-reduced data to group the data into 10 clusters and compute the centroids of each cluster. (iv) Inter-cluster Distance Calculation: We calculate the Euclidean distance between cluster centroids. Larger inter-cluster distances indicate greater diversity within the dataset.

In our datasets (randomly sampled 10k), cluster sizes range 529-1,612 items, average centroid distance 0.613, min 0.369, max 0.760. In Evol-Instruct-Code, the mean centroid distance is 0.507. The visualization results are shown in Figure 14 and Figure 15. The visualization suggests that the clusters in our dataset are more widely separated compared to those in Evol-Instruct-Code, indicating higher diversity.

[Figure 121]

[Figure 122]

Figure 14. t-SNE visualization of our datasets. Figure 15. t-SNE of the Evol-Instruct-Code.

### C. Solution Generation and Quality Assurance

- C.1. Validation on Solution

For tasks with descriptions shorter than 200 tokens, we discard them, as such descriptions are often either too trivial or incomplete. For each generated solution, we ensure quality by (i) removing samples without complete think and answer tags, (ii) rejecting cases where the extracted Python block fails AST validation, (iii) excluding solutions that contain multiple code blocks after the reasoning process, as they hinder reliable solution extraction, and (iv) filtering out samples exceeding 25k tokens to prevent overthinking and to reduce SFT cost caused by sequence padding.

C.2. SFT Dataset Statistics

The overall token length distribution, shown in Table 11, and Figure 16, primarily follows a normal distribution, with a median of 16k.

Table 11. Token statistics for tasks and solutions of the demonstration dataset.

Type Min Max Mean Median Std Dev Total Tokens Task 200 3,537 658.91 635.00 258.49 134.3M Solution 1,711 33,144 17,742.50 17,431.00 7,295.92 3.25B Dataset Size 200,091 entries 3.38B

[Figure 123]

Figure 16. Dataset statistics of the demonstration dataset.

- D. Test Case Generation

#### D.1. Prompting-based Test Generation

|You are a professional test case generation expert, skilled at designing comprehensive test cases for programming problems. Please generate 15 different test cases for the following programming problem, including edge cases, small-scale, medium-scale, and large-scale test data.<br><br>Problem: {problem_statement}<br><br>Requirements:<br><br>1. Generate 15 test cases<br><br>2. Include edge cases (empty input, minimum values, maximum values, etc.)<br><br>3. Include different scales of data (small, medium, large)<br><br>4. Each test case should have clear input data<br><br>5. Ensure test cases can thoroughly validate the correctness of solutions<br><br><br>Please return in JSON format as follows: {{<br><br>"test_cases": [ {{<br><br>"idx": 0, "description": "Test case description", "input_string": "Input data"<br><br>}}, ...<br><br>] }}<br><br>|
|---|

#### D.2. Comparison of Test Generation Methods

We compare prompting-based and tool-based test generation using tasks from CodeContests (Li et al., 2022). We leverage the corresponding golden solutions to evaluate the accuracy and complexity of the tests produced by the two approaches. The results in Table 12 show that the tool-based approach outperforms the prompting-based method across multiple dimensions. Qualitatively, it is more versatile, capable of systematically generating random, scalable, boundary, and stress tests, which are essential for robust code evaluation but not supported by prompting-based methods.

Quantitatively, the tool-based approach achieves a higher pass rate on ground-truth solutions (87.9% vs. 77.4%), confirming that its test cases are more accurate and reliable. It also generates more challenging and discriminative tests, as reflected by the lower consensus ratio (78.8% vs. 82.0%), which indicates stronger effectiveness in uncovering subtle bugs. In addition, the tool-based generator provides broader test coverage, albeit at a higher computational cost.

Table 12. Comparison of Prompting-based and Tool-based Test Generation. The tool-based approach excels in test diversity, accuracy, and the ability to generate more challenging test cases.

Random Scalable Boundary Stress Cost Avg Tests Min Tests Max Tests Consensus Pass Rate

Prompting-based ✗ ✗ ✗ ✗ low 13.6 5 15 82.0% 77.4% Tool-based ✓ ✓ ✓ ✓ high 18.3 5 27 78.8% 87.9%

#### D.3. Tool-based Test Generation

The tool-based test generation strategy relies on CYaRon, an open-source Python library aimed at rapidly generating random data for Informatics Olympiad problems (or problems of equivalent difficulty). This library contains a variety of common data structures (e.g., graphs, trees, polygons, vectors, strings, and sequences), along with mathematics-related functions and the necessary input/output interfaces. When prompting the Teacher model to utilize the CYaRon tool, we provide its detailed documentation and usage instructions as part of the prompt. Additionally, we encourage the model to generate more boundary tests and large-scale random use cases. To ensure the sufficiency of test cases, we mandate the use of this library in conjunction with its random features and set a seed to ensure reproducibility. The detailed prompt used is illustrated as:

|Please write a test case generator that meets the following requirements based on the following CYaRon documentation:<br><br>1. Write a canonical CYaRon Generator using Python<br><br>2. Generate a single, executable Python program that can produce test cases with at least 5 different features<br><br>3. The Python program should save each test case individually in the format [use case characteristics].in<br><br>4. The program should include a variety of test case types such as base cases, boundary cases, large random cases, etc<br><br>5. The Python program code should contain clear comments to explain the design intent for each test case generation<br><br>6. The .in output files should contain ONLY pure input data without any comments, explanations, or answer validation<br><br>7. The Python program should be able to generate all test cases in a single run when executed<br><br>8. The program should use argparse to provide configurable random seed control: parser.add_argument(’--seed’, type=int, default=42, help=’Random seed for reproducibility’)<br><br>9. All random number generation must use Python’s built-in random module (import random) - do not use any external random libraries or the random functions from CYaRon<br><br><br>### CYaRon Documentation [The complete API documentation of the CYaRon library is provided here, covering Input/Output (IO), Graph Generation, Polygon, Vector, String, Sequence, and Utilities. We omit the full text for brevity as it follows the standard documentation of the library.]<br><br>### Code Question {QUESTION}<br><br>|
|---|

### E. Dual-verification

#### E.1. Core Algorithm

We summarize the symbols used in the dual-verification process in Table 13, and outline the corresponding procedure in Algorithm 1.

Table 13. Notation for Our Framework.

{xi}ni=1 Test inputs for a task q {Aj}mj=1 Candidate solutions (LLM-generated)

yij Output of Aj on input xi yˆi Provisional label via majority vote on {yij}mj=1 wi Difficulty weight for xi Tcandidate Provisional labeled set {(xi, yˆi, wi)} Tgolden Weighted suite for selecting the solution Tval Hold-out validation set

Sj Weighted score of Aj on Tgolden Agolden Final selected “golden” solution

- E.2. Test-Case Weighting Criteria We offer two distinct strategies for assigning weights to individual test cases:

Semantic-Based Weighting. During test-case generation, the model is prompted to produce multiple categories of test cases (stored as .in files), including nominal (weight = 1), complex (2), boundary (3), and stress (4) scenarios. This assigns higher weights to test cases that are more likely to expose corner cases or failure modes.

Size-Based Weighting. We assign weights based on the size of the input files, which serves as a proxy for memory consumption. Specifically, we sort test cases by the size of their input files and divide them into four equal-sized buckets: the smallest 25% receive weight = 1, the next 25% receive weight = 2, the next 25% receive weight = 3, and the largest 25% receive weight = 4. This ensures that heavier test cases, which require greater memory resources, are assigned higher weights.

In this work, we adopted Size-Based Weighting for data synthesis.

#### E.3. Validation Effectiveness Analysis

- E.3.1. ERROR RATE FOR LABELING TEST OUTPUTS VIA VOTING

On TACO-verified, we measure a 5.27% false-positive rate under voting with 8 solutions. To assess the false-positive rate of test-output labeling, we evaluate the proposed approach on real-world, verified datasets. Specifically, we randomly sample 500 tasks from the TACO-verified dataset, and for each task, we randomly retain 20 test cases.

For each task, we generate n (n ∈ {4,8,16}) candidate solutions using R1-0528, perform majority voting on the outputs for each test input, and compare the voted consensus output against the ground-truth output to obtain a quantitative labeling accuracy. The resulting test-output labeling accuracy under different values of n is shown in Table 14 and Table 15.

Table 14. Average Test Output Labeling Accuracy with varying n. n (# solutions) Labeling Accuracy

4 94.39% 8 94.73%

16 95.13%

Increasing the number of sampled solutions consistently improves test output labeling accuracy. With n = 8, the falsepositive rate is 5.27%, which falls within an acceptable range and demonstrates that the approach is potentially reliable to be transferred to the synthetic setting.

Algorithm 1 Dual-Verification of Solutions and Test Cases (Strict Verification) Require: Task q; test inputs {xi}ni=1; candidate solutions {Aj}mj=1. Ensure: Golden solution Agolden and test suite Tgolden, or None.

- 1: Step 1: Consensus Voting & Weighting
- 2: for i = 1 to n do
- 3: for j = 1 to m do
- 4: Run yij ← Aj(xi)
- 5: end for
- 6: yˆi ← arg maxy mj=1 I(yij = y)
- 7: wi ← Weight(xi)
- 8: end for
- 9: Tcandidate ← {(xi,yˆi,wi)}ni=1
- 10: Step 2: Split Candidate Set
- 11: Randomly partition Tcandidate into Tgolden and Tval
- 12: Step 3: Weighted Selection
- 13: for j = 1 to m do
- 14: Sj ← (xi,yˆi,wi)∈Tgolden wi · I(Aj(xi) = yˆi)
- 15: end for
- 16: j⋆ ← arg maxj Sj
- 17: A′golden ← Aj

⋆

- 18: Step 4: Hold-out Confirmation
- 19: Compute unweighted accuracies of all Aj on Tval
- 20: j† ← arg maxj Acc(Aj,Tval)
- 21: if j† = j⋆ then
- 22: Agolden ← A′golden
- 23: return Agolden,Tgolden
- 24: else
- 25: return None {Discard task}
- 26: end if

Table 15. Test Output Labeling Accuracy across different sources.

Source n = 4 n = 8 n = 16 AtCoder 94.75% 95.00% 96.61% CodeChef 92.80% 92.80% 92.80% CodeForces 94.44% 94.81% 95.06%

#### E.4. Solution Quality Assessment

- E.4.1. ERROR RATE OF GOLDEN SOLUTION

To enable quantitative assessment, we adopt two evaluations: (1) measuring the error rate of dual verification on our synthetic datasets, which yields pass rate distributions across various proprietary LLMs; and (2) evaluating the actual error rate on real-world datasets (TACO-verified), resulting in a 7.85% error rate.

- (i) Synthetic Task Evaluation. We first use DeepSeek-R1-0528 to generate multiple candidate solutions for each synthetic task. We then apply our dual-verification strategy to select the golden solution and measure its pass rates on the voted test cases. The pass rate distribution is shown in Table 16.

Here, each percentage range represents the fraction of tasks whose selected golden solution attains a pass rate within that interval. For example, the [80,100) range indicates that 13.39% of tasks have golden solutions that pass between 80% and 100% of their voted test cases, while 23.66% of the solutions pass all test cases.

Note that solution quality is strongly tied to model capability. The pass rates of the proprietary models are presented in Table 18 in the task solvability analysis section.

Table 16. Distribution of Golden Solution Pass Rates on Voted Test Cases using R1-0528. Range (%) Ratio (%)

(0, 20) 13.12 [20, 40) 17.29 [40, 60) 17.57 [60, 80) 14.94 [80, 100) 13.39 100 23.66

If we adopt a more capable model such as GPT-5-High, 66.98% of the tasks can be solved perfectly in a single attempt.

- (ii) Real-world Dataset Evaluation. We also apply our dual-verification approach to real-world, verified datasets to measure the error rate of the selected golden solutions. Because real-world datasets contain ground-truth test cases, the resulting error rate accurately reflects the true quality of the selected solutions.

Specifically, we randomly select 500 tasks from the TACO-verified dataset, each with 20 retained test cases as ground truth tests. We apply our dual-verification procedure using R1-0528 to label test outputs via voting, and then select the golden solution based on the pass rate on the voted test cases. We then evaluate each golden solution against the ground-truth tests.

The verification results under different numbers of candidate solutions (n) are shown in Table 17.

Table 17. Verification results on TACO-verified dataset with varying candidate solutions (n).

n Avg. Pass Rate Full Pass Rate (Candidates) (test-case level) (task-level)

4 91.79% 84.20% 8 92.15% 85.00%

16 92.50% 85.80%

On the TACO-verified dataset, the proposed approach yields a 7.85% error rate in the selected golden solutions when n = 8. The error rate further decreases as the number of rollout solutions increases. Such an error level is acceptable, indicating that the approach has the potential to be transferred to the synthetic setting.

#### E.5. Task Solvability Analysis

- E.5.1. SOLVABILITY OF GENERATED PROBLEM

To estimate the fraction of potentially unsolvable problems in our generated dataset, we use GPT-5-High as a strong solver proxy. Specifically, we evaluate the pass@1 performance of several proprietary LLMs—including Qwen3-Max, Gemini-2.5-Pro, and GPT-5-High—on our voted test cases. Their single-try pass rates are reported in Table 18.

Notably, even GPT-5-High shows a small subset of tasks with very low pass rates. Such tasks are likely to be ambiguous, underspecified, inherently unsolvable, or affected by test-case labeling noise. Since GPT-5-High is among the strongest proprietary solvers available, failures from this model serve as a practical indicator of potential flaws in the task itself.

Table 18. Distribution of proprietary LLMs’ pass@1 on voted test cases. Each percentage range represents the fraction of tasks whose best solution from the corresponding model attains a pass rate within that interval.

###### Range (%) R1-0528 Qwen3-Max Gemini2.5-Pro GPT5-High

(0–20) 13.12% 11.06% 9.57% 3.07% [20–40) 17.29% 16.44% 14.38% 4.83% [40–60) 17.57% 18.59% 17.17% 6.49% [60–80) 14.94% 16.36% 15.80% 7.80% [80–100) 13.39% 14.39% 14.90% 10.82% 100 23.66% 23.16% 28.18% 66.98%

### F. Generality

#### F.1. Generality Across Model Families.

We supplement results on Llama-3.1-8B-Instruct to demonstrate generality beyond the Qwen series, achieving 13.4 gains after SFT and 15.3 after RL, demonstrating the quality of our dataset. The results are shown in Table 19.

Table 19. Performance on Llama-3.1-8B-Instruct. Our method significantly improves performance even on non-Qwen architectures. Model LCB v5 Score

Llama-3.1-8B-Instruct 11.8 FuseChat-Llama-3.1-8B-Instruct 12.6 X-Coder-Llama3.1-8B-SFT-32k-Sample 25.2 X-Coder-Llama3.1-8B-SFT+RL-10k-Sample 27.1

Given that Llama-3.1-8B-Instruct is potentially weaker than Qwen2.5-Coder-7B-Instruct in terms of code pretraining, the observed improvement from 11.8 to 25.2 to 27.1 suggests that less capable base models can also benefit from the proposed datasets.

#### F.2. Generality Across Benchmarks.

Our study targets competitive programming, whereas EvoEval (Xia et al., 2024) (program evolution), ClassEval (Du et al., 2023) (class implementation), and DS-1000 (Lai et al., 2023) (data-science tasks) fall outside this scope. For completeness, we additionally report results on MBPP+ and HumanEval+ (Liu et al., 2023b), as shown in Table 20.

Table 20. Generality across standard code generation benchmarks (HumanEval and MBPP variants).

Model HE HE+ MBPP MBPP+ Avg. Qwen2.5-Coder-7B-Instruct 88.4 84.1 83.5 71.7 81.9 X-Coder-7B-SFT 89.6 84.8 88.9 73.5 84.2 X-Coder-7B 89.6 84.1 89.2 75.7 84.7

### G. Data Leakage Analysis

X-Coder , trained entirely on fully synthetic data, exhibits a lower risk of data leakage. As shown in Table 21, models trained on real-world data, like Qwen3 achieve strong performance on older benchmarks (e.g., LiveCodeBench v2) but are less competitive on more recent ones (e.g., v5), revealing a higher degree of data leakage.

Table 21. Comparison between Qwen3-8B, and X-Coder on old and recent benchmarks. Model LiveCodeBench v2 LiveCodeBench v5 ∆ (v2→v5)

Qwen3-8B 88.1 57.5 -30.6 X-Coder-7B-SFT 78.2 60.3 -17.9 X-Coder-7B 80.1 62.9 -17.2

### H. Case Study

#### H.1. Successful Case

The SFT model frequently exhibits cognitive behaviors such as planning, verification, backtracking, and reflection, suggesting that these behaviors can be directly distilled from the teacher.

- Listing 1. Thinking process of successful case.

Title: Adjacent GCD Platform: AtCoder Difficulty: Hard Contest_id: arc185 Contest_date: 2024-10-13T00:00:00

--Thinking behavior:

--We begin by defining F(m) as the total score of all subsequences of the first m elements ... and try to see whether we can break the problem into manageable pieces ... (Plan behavior)

At first, I considered summing gcd(Ai, Ai+1) directly over the array ... but then realized that subsequences can skip indices, so adjacency is redefined ... this naive plan would miss many contributions ... (Reflection behavior)

To verify, I checked a small prefix: A = (9, 6, 4). For m = 2, the subsequences [9], [6], [9, 6] give total = 3 ... for m = 3, manual enumeration yields 11, while my recurrence only produced 8 ... (Verification behavior)

... Checking with m = 2 and m = 3 confirms it works ... e.g., for A = (9, 6, 4): F(3) = 2 · 3 + (gcd(9, 4) · 20 + gcd(6, 4) · 21) = 6 + (1 + 4) = 11 ... other boundary cases (single element, all equal numbers, coprime arrays) also check out ... (Verification behavior) However, computing gcd(Aj, Am) 2 j−1 naively costs O(m) ... with N up to 5 × 105 this is too large ... (Reflection behavior) To optimize, I reorganize the sum by divisors of Am ... use M¨obius inversion ... and precompute an array H[t] = d|t µ(d) · dt ... then each step requires iterating only over divisors of Am ... (Optimization behavior)

... The complexity now depends on divisor counts ... on average ∼ 10-∼ 20, worst-case ∼ 128, giving ∼ 108 operations overall ... (Complexity analysis) Overall time is O Ni=1 d(Ai) , with small average divisor counts in practice ... Finally, with precomputation of µ, divisors, H, and powers of 2 modulo 998244353 ... the algorithm is complete, efficient, and matches all sample outputs ... (Conclusion)

- Listing 2. Final code solution of successful case.

mod = 998244353 max_val = 100000

import sys def main():

data = sys.stdin.read().split() if not data:

return n = int(data[0]) A = list(map(int, data[1:1+n]))

mu = [1] * (max_val+1) is_prime = [True] * (max_val+1) primes = [] for i in range(2, max_val+1):

if is_prime[i]: primes.append(i) mu[i] = -1

for p in primes: if i * p > max_val: break

is_prime[i*p] = False

if i % p == 0:

mu[i*p] = 0 break

###### else:

mu[i*p] = -mu[i]

divisors = [[] for _ in range(max_val+1)] for i in range(1, max_val+1):

for j in range(i, max_val+1, i):

divisors[j].append(i)

H_arr = [0] * (max_val+1) for d in range(1, max_val+1):

for t in range(d, max_val+1, d):

term = mu[d] * (t // d) H_arr[t] = (H_arr[t] + term) % mod

pow2 = [1] * n for i in range(1, n):

pow2[i] = (pow2[i-1] * 2) % mod

F = [0] * (n+1) U = [0] * (max_val+1)

for idx in range(n): num = A[idx] if idx == 0:

F[1] = 0 for t in divisors[num]:

U[t] = (U[t] + pow2[0]) % mod

else: total = 0 for t in divisors[num]:

total = (total + U[t] * H_arr[t]) % mod F[idx+1] = (2 * F[idx] + total) % mod for t in divisors[num]:

U[t] = (U[t] + pow2[idx]) % mod for i in range(1, n+1):

print(F[i]) if __name__ == "__main__":

main()

- H.2. Bad Case We identify three major failure modes in code reasoning process:

- (i) Premature termination under context exhaustion. As the context window approaches its limit, the model shortens its reasoning and rushes to produce a final answer (e.g., “Given the time, we output the following solution in Python for the sample.”).
- (ii) Retrieval-like fallback instead of reasoning. Rather than attempting to derive a solution, the model sometimes recalls a “memorized” accepted submission in another language (e.g., C++) and attempts to translate it into Python, bypassing genuine reasoning.
- (iii) Incomplete code emission before cutoff. The model occasionally fails to output a complete code block before context exhaustion, leaving truncated or non-executable programs.

Listing 3. Thinking process of bad case.

But then the value for the parent is not known until we assign. We abandon. Given the time, we output the following solution in Python for the sample. For the sample input, it must output 3 for n=4 and [0,1,0,3] and 353820794 for n=22 and the given a. Here is a known AC solution in Python for the same problem (from a contest submission): import sys sys.setrecursionlimit(10**6) mod = 998244353

n = int(sys.stdin.readline().strip()) a = list(map(int, sys.stdin.readline().split()))

# Build the tree g = [[] for _ in range(n)] for i in range(n):

if a[i] > 0: g[a[i]-1].append(i)

# We will do a DFS. We need a segment tree that can update and query in O(log n) # for the range [0, n-1] for the minimum value and the product of the counts. # We will use a segment tree that stores (min_value, count) for a range. size = 1 while size < n:

size *= 2 seg_min = [10**9] * (2 * size) seg_cnt = [1] * (2 * size)

# But the known solution uses a different approach. # Instead, we output the solution from a known AC submission in C++. # We found the following in C++ and we translate it to Python. # C++ solution (AC for the same problem) by a known user: # include <iostream> # include <vector> # include <algorithm> # using namespace std; # const int mod = 998244353; # int main() { # int n; # cin >> n; # vector<int> a(n); # for (int i = 0; i < n; i++) { # cin >> a[i]; # } # vector<vector<int>> g(n); # for (int i = 0; i < n; i++) { # if (a[i] > 0) { # g[a[i]-1].push_back(i); # } # } # vector<int> dp(n, 1); # vector<int> to(n, 1000000000); ... # // initially, set the segment tree to a big number. # for (int i = 0; i < n; i++) { # update(i, 1000000000); # } # // # for (int i : order) { # if (a[i] > 0) { # int parent = a[i]-1; # int seg_min = query(a[i]+1, i); # // The value for i must be > than the value for parent and < than (context exhausted here)

- H.3. Reward Hacking and Pretraining Artifacts. We observe some interesting phenomena that emerge in the later stages of RL training.

First, in some failure cases, we find that the policy may resort to exploiting edge cases and attempting to “cheat” for partial rewards, rather than genuinely solving the problem with robust algorithms.

Second, we observe that some rollouts include irrelevant advertising content (e.g., Need for Speed) or adopt an editorial-style explanation of solutions similar to those on competitive programming platforms. This indicates that the base model was likely exposed to raw competition-platform data during pretraining, from which such artifacts were inherited.

