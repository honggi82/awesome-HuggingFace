# arXiv:2512.17385v1[cs.CL]19Dec2025

## UCoder: Unsupervised Code Generation by Internal Probing of Large Language Models

Jiajun Wu1, Jian Yang1∗, Wei Zhang1, Lin Jing1, Yuqing Ma2, Ensheng Shi2, Yuchi Ma2, Zhoujun Li2, Xianglong Liu1 1Beihang University; 2Huawei; {wuyuverse,jiayang}@buaa.edu.cn

Abstract

Large language models (LLMs) have demonstrated remarkable capabilities in code generation tasks. However, their effectiveness heavily relies on supervised training with extensive labeled (e.g., question-answering pairs) or unlabeled datasets (e.g., code snippets), which are often expensive and difficult to obtain at scale. To address this limitation, this paper introduces a method IPC, an unsupervised framework that leverages Internal Probing of LLMs for Code generation without any external corpus, even unlabeled code snippets. We introduce the problem space probing, test understanding probing, solution space probing, and knowledge consolidation and reinforcement to probe the internal knowledge and confidence patterns existing in LLMs. Further, IPC identifies reliable code candidates through self-consistency mechanisms and representation-based quality estimation to train UCoder (coder with unsupervised learning). We validate the proposed approach across multiple code benchmarks, demonstrating that unsupervised methods can achieve competitive performance compared to supervised approaches while significantly reducing the dependency on labeled data and computational resources. Analytic experiments reveal that internal model states contain rich signals about code quality and correctness, and that properly harnessing these signals enables effective unsupervised learning for code generation tasks, opening new directions for training code LLMs in resource-constrained scenarios.

### 1 Introduction

Large language models (LLMs) have demonstrated strong capabilities in code generation, producing functional code from natural language descriptions. This progress has attracted substantial interest from both academia and industry due to its practical impact on software development. Closed-source

∗Corresponding Author.

Human Annotation

[Figure 1]

Time-consuming

[Figure 2]

[Figure 3]

Raw Data Training Data

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Data Distillation

Expensive

Base LLM

Code LLM

Supervised Code Generation

[Figure 8]

[Figure 9]

Unsupervised Code Generation

|[Figure 10]|Internal Knowledge| |
|---|---|---|
|[Figure 11]| |[Figure 12]|
|[Figure 13]| |[Figure 14]|

[Figure 15]

probe

[Figure 16]

[Figure 17]

self-training

Figure 1: Comparison between supervised and unsupervised paradigms for code generation.

LLMS such as GPT-5 (OpenAI, 2025) and Claude4.5 (Anthropic, 2025) can generate file-level code with high accuracy, while open-source alternatives, including StarCoder (Li et al., 2023; Lozhkov et al., 2024), DeepSeek-Coder (Guo et al., 2024), and QwenCoder (Hui et al., 2024) have emerged as competitive solutions for code intelligence.

Most existing approaches for improving code generation rely on supervised instruction tuning, where LLMs are fine-tuned on curated problemsolution pairs annotated by human experts or LLMbased annotated. However, creating high-quality instruction data requires substantial human effort in problem design, implementation, and verification, with costs increasing as model capabilities advance. The recent (Yue et al.; Ye et al., 2025; Chu et al., 2025) works emphasize that pre-training brings the knowledge, and post-training is weak at knowledge integration but focuses on knowledge utilization and alignment. These challenges motivate a fundamental question: Can LLMs au-

[Figure 18]

###### Stage1-3: Problem Space Probing

###### Stage5:Solution Space Probing

###### Stage6:Knowledge Consolidation and Rein-forcement

Stage1: Problem Generator

|[Figure 19]|
|---|

[Figure 20]

Stage2:Quality Oracle

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

execution success rate

language perplexity n = 128

consensus strength

[Figure 29]

basic algorithm

natural instruction

quality samples train dataset

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Trained LLM

[Figure 35]

[Figure 36]

consensus-based filtering

[Figure 37]

[Figure 38]

[Figure 39]

Multi-domain algorithmic tasks

tool usage

competitive programming

Base LLM clusting

[Figure 40]

[Figure 41]

[Figure 42]

Base LLM

|[Figure 43]|
|---|

|[Figure 44]|
|---|

dense sampling

solution space

test execution

Stage3:Interface Synthesizer

eval

n = 128

[Figure 45]

[Figure 46]

def function(param: Type) -> ReturnType: """ Brief description of function purpose. Args:

###### Stage4:Test Understanding Probing

[Figure 47]

[Figure 48]

|[Figure 49]|
|---|

[Figure 50]

[Figure 51]

implicit knowledge

100 Input validation cases

param: Description of the parameter. Returns:

test generation

boundary detection

Signatures & speciﬁcations

Base LLM

Problem Description

probe

[Figure 52]

[Figure 53]

Description of the return value. """

|[Figure 54]|
|---|

[Figure 55]

[Figure 56]

[Figure 57]

orchestrator

Figure 2: Overview of the proposed six-stage self-bootstrapping framework for unsupervised code generation.

tonomously improve their generation capabilities using post-training without any external corpus, relying only on pre-trained knowledge?

In this work, we introduce an unsupervised framework that performs Internal Probing of LLMs for Code generation, enabling post-training without external corpora or human-annotated instruction data. Our approach exploits latent programming knowledge in LLMs and uses execution feedback as a scalable, deterministic supervision signal grounded in program semantics. We implement a six-stage self-bootstrapping process that generates diverse programming tasks, synthesizes test suites, samples candidate solutions, and applies execution-driven consensus clustering to identify correct implementations. High-consensus solutions are iteratively consolidated as training data, forming a feedback loop that progressively improves model performance.

Despite using no external data, UCoder achieves comparable performance to the supervised baseline across multiple benchmarks. The primary contributions of this work are:

- • We successfully probe latent programming knowledge in LLMs by forcing models to generate programming problems and their solutions, then identify correct solutions by finding clusters of similar implementations. Then, the self-training method progressively improves the LLM by reinforcing solutions.
- • Based on the self-generated data from the unsupervised framework using internal probing of LLMs (IPC) without any external data, UCoder (7B, 14B, 32B) achieves performance competitive with supervised baselines.
- • We provide empirical analysis showing that self-generated data maintains rich lexical,

semantic, and structural diversity, while consensus-based selection improves solution quality and exhibits inverse scaling behavior.

### 2 Unsupervised Code Generation

#### 2.1 Task Definition

- 2.1.1 Supervised Code Generation Supervised code generation is formulated as a sequence-to-sequence learning task. Given a train-

ing dataset D = {(xi,yi)}Ni=1, the model parameters are optimized by maximizing log-likelihood:

θ∗ = arg max

θ

N

i=1

log pθ(yi|xi) (1)

where xi ∈ X denotes a natural language query, yi ∈ Y represents the corresponding reference implementation, and pθ(y|x) is the conditional distribution over code sequences parameterized by θ. We adopt Pass@k (Chen et al., 2021a) to evaluate code correctness, which measures the probability that at least one of k independently sampled solutions passes all test cases.

- 2.1.2 Unsupervised Code Generation Unsupervised code generation aims to improve LLM code generation capabilities without humanannotated supervision. Given an initial model

M0 : X → Y, the objective is to develop a selfimprovement algorithm A producing an enhanced model M∗ = A(M0) such that Pass@k(M∗) > Pass@k(M0) on held-out test sets, without access to paired training data (x,y) ∈ X × Y. This presents three fundamental challenges: (1) Problem Space Construction. Automatically generating diverse programming problems with appropriate difficulty distributions while maintaining semantic clarity; (2) Unsupervised Correctness Verification. Assessing functional correct-

ness without reference implementations; (3) SelfBootstrapping Signal Construction. Extracting reliable training signals from noisy candidates while ensuring iterative stability. We address these through an execution-driven consensus mechanism coupled with a self-bootstrapping framework, detailed in the following sections.

#### 2.2 Probing Internal Knowledge in LLMs

LLMs encode extensive programming knowledge through pre-training, yet this knowledge remains implicit and difficult to elicit. We propose a sixstage framework to surface and reinforce these latent capabilities, as shown in Figure 2. First, we probe the problem space (Stages 1–3) by prompting the model to generate algorithmic problems with complete specifications, revealing its understanding of programming paradigms and data structures. Representative examples of problem generation, difficulty assessment, and solution skeleton construction are illustrated in Figure 3.

We then assess semantic understanding (Stage 4) by generating approximately 100 test cases per problem to identify boundary conditions and edge cases. At the core (Stage 5), we probe the solution space via dense sampling, where executiondriven consensus clustering reveals that correct implementations form tight clusters while incorrect ones are dispersed. We quantify solution quality using execution success rate e(r), consensus strength s(r), and code fluency f(r). Finally (Stage 6), we consolidate high-consensus samples through supervised fine-tuning, reinforcing correct patterns. The process forms a positive feedback loop: at iteration t, the improved LLM Mt produces higherquality candidates, enabling more reliable selection and further strengthening Mt+1. Our experimental results in subsection 3.2 validate that pre-trained models already contain the knowledge required to solve target tasks in implicit form.

#### 2.3 Execution-Driven Consensus Clustering

Our approach exploits that correctness is singular while incorrectness is diverse: correct implementations produce identical outputs, but incorrect ones fail heterogeneously. This clustering structure allows the maximum-consensus cluster to indicate correctness without ground truth, formalized in Theorem 2.4 and validated in subsection 4.2.

2.3.1 Definitions Consensus clustering has three definitions.

[Figure 58]

[Figure 59]

###### Stage1:Problem

###### Stage3:Skeleton

<SKELETON_START> [imports for type annotations] def abstract_function

<PROBLEM_START> Title: [Descriptive Task Name] Description: [Clear explanation of the function's purpose] Function Signature:

(param: type)->return_type:

""" [1-3 sentences explaining the logic, approach,and key operations performed]. Args:

def specific_descriptive_function (param: type) -> return_type:

Example:

param (type): [Parameter description]. Returns:

Input: [sample input]

Output: [expected output] Hint: [One sentence suggesting an approach] <PROBLEM_END>

return_type: [The returned value]. """

<SKELETON_END>

[Figure 60]

###### Stage2:Rating

<RATING_START> Problem Quality Score: X Summary: [sentences explaining your rating based on clarity,completeness,and quality.] <RATING_END>

Figure 3: Problem space probing proceeds through three stages: problem generation with function signatures and input-output contracts, difficulty rating assessment and categorization, and solution skeleton generation with implementation structure.

- Definition 2.1 (Execution Signature). Given candidates R = {r1,...,rn} and tests T = {t1,...,tm}, define Exec : R × T → {0,1} as the pass indicator: Exec(ri,tj) = 1 if ri passes tj, and 0 otherwise. The execution signature of ri on T is

σ(ri; T) =

m

j=1

Exec(ri, tj), (2)

where denotes ordered concatenation. Thus σ(ri;T) ∈ {0,1}m, and σ(ri;T) = 1m indicates that ri passes all m tests.

- Definition 2.2 (Consensus Clusters). If σ(ri,T) = σ(rj,T), we can regard the ri and rj as equivalent solution. Given the value of σ(ri,T), we can partition R into clusters C = {C1,...,Cℓ} of behaviorally identical candidates.
- Definition 2.3 (Quality Metrics). Each candidate r ∈ R is scored by:

- e(r) = |{t ∈ T : Exec(r,t) ̸=⊥}| |T|

, s(r) = |{r′ ∈ R : σ(r′) = σ(r)}|,

- f(r) = exp

(3)

 −

 ,

|r|

1 |r|

log p(xi | x<i)

i=1

where e(r) measures execution success, s(r) consensus strength, and f(r) code fluency.

2.3.2 Hierarchical Selection We select the valid candidates using three criteria:

- (1) Reliability Filtering. Candidates with low execution success are removed (threshold ρ = 0.8): R′ = {r ∈ R : e(r) ≥ ρ}.
- (2) Consensus Selection. We select the largest nontrivial cluster: C∗ = arg maxC∈C′,|C|≥τ |C|

HumanEval MBPP BCB-Complete BCB-Instruct LiveCode- FullStack-

Model

HE HE+ MBPP MBPP+ Full Hard Full Hard Bench Bench

6B+ LLMs CodeLlama-7B-Instruct 40.9 33.5 39.9 33.6 25.7 4.1 21.9 3.4 7.1 25.40 DS-Coder-6.7B-Instruct 74.4 71.3 74.9 65.6 43.8 15.5 35.5 10.1 15.5 40.16 OpenCoder-8B-Instruct 83.5 78.7 79.1 69.0 50.9 18.9 43.2 18.2 23.2 41.08 Qwen2.5-Coder-7B 61.6 53.0 76.9 62.9 45.8 16.2 - - - Qwen2.5-Coder-7B-Instruct 88.4 84.1 83.5 71.7 48.8 20.3 41.0 18.2 18.2 47.95 UCoder-7B 83.5 76.8 85.2 72.2 52.0 22.3 41.1 15.5 22.9 51.27

13B+ LLMs CodeLlama-13B-Instruct 40.2 32.3 60.3 51.1 31.7 6.8 28.5 9.5 6.1 27.00 Starcoder2-15B-Instruct-v0.1 67.7 60.4 78.0 65.1 45.1 14.9 37.2 11.5 12.1 42.68 DS-Coder-V2-Lite-Instruct 81.1 75.6 82.8 70.4 47.6 18.2 36.8 16.2 24.3 Qwen2.5-Coder-14B 64.0 57.9 81.0 66.7 51.8 22.3 - - - Qwen2.5-Coder-14B-Instruct 89.6 87.2 86.2 72.8 56.7 29.7 48.4 22.2 23.4 55.28 UCoder-14B 87.8 81.1 86.5 74.3 53.9 24.3 40.9 16.2 20.6 52.52

32B+ LLMs CodeLlama-34B-Instruct 48.2 40.2 61.1 50.5 35.6 10.8 29.0 8.8 8.4 27.56 DS-Coder-33B-Instruct 81.1 75.0 80.4 70.1 51.1 20.9 42.0 17.6 21.3 48.19 DS-Coder-V2-Instruct 85.4 82.3 89.4 75.1 59.7 29.7 48.2 24.3 27.9 56.37 Qwen2.5-Coder-32B 65.9 60.4 83.0 68.2 53.6 26.4 - - - Qwen2.5-Coder-32B-Instruct 92.7 87.2 90.2 75.1 58.0 33.8 49.6 27.0 31.4 56.88 UCoder-32B 89.0 82.9 89.7 75.7 55.4 27.7 45.7 17.6 21.4 53.35

###### Closed-APIs

GPT-4o-2024-08-06 92.1 86.0 86.8 72.5 - 36.5 50.1 25.0 34.6 58.89 Claude-3.5-Sonnet-20241022 92.1 86.0 91.0 74.6 58.6 35.1 46.8 25.7 31.6 60.70

- Table 1: Performance comparison of Qwen2.5-Coder Base and Instruct models with our iterative SFT models across code generation benchmarks. All metrics represent Pass@1 execution rates (%). Complete split is reported for Base models and Instruct split for Instruct models. Bold indicates best performance within each size category. “-” denotes unavailable or inapplicable results.

(3) Intra-Cluster Selection. Within C∗, we choose r∗ = arg maxr∈C∗⟨e(r),−f(r)⟩.

- 2.3.3 Theoretical Guarantee Theorem 2.4 (Consensus Convergence). Let R =

{r1,...,rn} be n candidates sampled independently from a model, and let T denote a set of unit tests. Assume that at least k candidates in R are functionally correct with probability at least 1 − δ, and that any pair of incorrect implementations produces identical outputs on a single test with probability at most p < 1. If the test set size satisfies

log(n/k) −log p

|T| ≥

,

then the largest consensus cluster Cmax contains only correct implementations with probability at least

P(Cmax is correct) ≥ 1 − δ − n2p|T |.

#### 2.4 Iterative Self-Training

We formalize the iterative self-training procedure and explain why it yields consistent improvement.

Definition 2.5 (Iterative Update). At iteration t, we construct training set Dt = {(qi,ri∗)}, where each ri∗ is selected via consensus from n candidates sampled from Mt, and update:

log pθ(r∗ | q). (4)

θt+1 = arg max

θ

(q,r∗)∈Dt

Why Self-Training Improves Performance? Iterative self-training is effective because consensus selection acts as a quality filter. Let Q(r) ∈ [0,1] denote candidate quality. For n independent samples, random selection yields expected quality Er∼Mt[Q(r)], whereas consensus selection favors correct implementations that cluster by execution behavior, achieving (for some ∆ > 0):

E[Q(r∗)] = Er∼Mt[Q(r)] + ∆, (5)

Optimizing on Dt shifts the model toward higherquality samples. As Mt+1 increases pθ(r∗ | q) for above-average outputs:

Er∼Mt+1[Q(r)] ≥ Er∼Mt[Q(r)], (6)

where it induces a positive feedback loop where improved models generate higher-quality candidates and more reliable training signals.

HumanEval MBPP BCB-Complete BCB-Instruct LiveCode- FullStack-

Iter

HE HE+ MBPP MBPP+ Full Hard Full Hard Bench Bench

###### Ucoder-7B

- 0 77.4 67.1 72.0 63.0 44.4 15.5 34.6 14.9 13.0 40.2

- 1 81.7+4.3 74.4+7.3 72.00.0 63.00.0 51.3+6.9 23.0+7.5 42.2+7.6 20.9+6.0 15.3+2.3 48.2+7.9

- 2 84.1+6.7 77.4+10.3 79.1+7.1 66.4+3.4 44.7+0.3 14.9-0.6 35.8+1.2 12.2-2.7 14.5+1.5 40.2-0.1

- 3 84.1+6.7 76.8+9.7 81.2+9.2 69.0+6.0 44.40.0 15.50.0 34.5-0.1 13.5-1.4 21.4+8.4 40.20.0

- 4 84.1+6.7 77.4+10.3 79.1+7.1 66.4+3.4 44.7+0.3 14.9-0.6 35.8+1.2 12.2-2.7 14.5+1.5 40.2-0.1

- 5 81.7+4.3 75.0+7.9 83.9+11.9 71.2+8.2 52.2+7.8 19.6+4.1 40.7+6.1 14.2-0.7 20.6+7.6 50.0+9.7

- 6 83.5+6.1 76.8+9.7 85.2+13.2 72.2+9.2 52.0+7.6 22.3+6.8 41.1+6.5 15.5+0.6 22.9+9.9 51.3+11.0 Ucoder-14B

- 0 83.5 76.8 75.9 64.0 53.3 23.6 43.2 16.2 22.1 50.1

- 1 85.4+1.9 77.4+0.6 87.8+11.9 73.3+9.3 53.1-0.2 21.6-2.0 41.0-2.2 14.2-2.0 17.6-4.5 53.6+3.5

- 2 84.8+1.3 76.2-0.6 84.9+9.0 70.6+6.6 50.4-2.9 23.60.0 40.9-2.3 15.5-0.7 19.1-3.0 49.9-0.2

- 3 84.8+1.3 78.0+1.2 83.6+7.7 72.2+8.2 51.6-1.7 23.0-0.6 41.8-1.4 15.5-0.7 18.3-3.8 49.8-0.3

- 4 84.8+1.3 78.0+1.2 84.1+8.2 72.8+8.8 51.8-1.5 18.2-5.4 41.1-2.1 12.8-3.4 22.10.0 50.4+0.3

- 5 87.8+4.3 81.1+4.3 86.5+10.6 74.3+10.3 53.9+0.6 24.3+0.7 40.9-2.3 16.20.0 20.6-1.5 52.5+2.4

- 6 87.2+3.7 80.5+3.7 84.4+8.5 71.7+7.7 53.30.0 23.0-0.6 43.5+0.3 16.20.0 22.9+0.8 51.6+1.5 Ucoder-32B

- 0 86.0 78.0 86.2 72.2 54.8 28.4 44.6 18.9 22.1 53.0

- 1 87.8+1.8 82.3+4.3 89.9+3.7 75.9+3.7 55.4+0.6 23.6-4.8 44.5-0.1 17.6-1.3 17.6-4.5 54.3+1.2

- 2 86.6+0.6 81.1+3.1 88.4+2.2 74.6+2.4 56.2+1.4 23.0-5.4 44.8+0.2 18.90.0 16.0-6.1 52.4-0.6

- 3 87.8+1.8 81.1+3.1 88.9+2.7 74.6+2.4 54.3-0.5 22.3-6.1 43.8-0.8 16.9-2.0 19.8-2.3 54.7+1.7

- 4 89.0+3.0 82.9+4.9 89.7+3.5 75.7+3.5 55.4+0.6 27.7-0.7 45.7+1.1 17.6-1.3 21.4-0.7 53.4+0.3

- 5 88.4+2.4 81.7+3.7 87.8+1.6 73.3+1.1 54.5-0.3 24.3-4.1 43.8-0.8 18.90.0 22.9+0.8 53.3+0.2

- 6 88.4+2.4 82.3+4.3 89.2+3.0 74.1+1.9 54.0-0.8 23.6-4.8 44.1-0.5 16.9-2.0 22.9+0.8 53.8+0.7

- Table 2: Performance (Pass@1) across iterative SFT rounds at different model scales using Qwen2.5-Coder as the base model. Orange-highlighted rows show Iter 0 (initial model trained on seed data), while subsequent iterations use self-generated synthetic data. Blue-highlighted rows indicate the best-performing iteration for each scale. Bold denotes best performance, underline denotes second-best performance for each metric within each scale, and subscripts show differences from Iter 0 (green for improvement, red for decline).
- 3 Experiments

1.0

P10 P25 P50 P75 P90

1200

###### CumulativeProbability

KDE CDF

Statistics: n = 16,867 μ = 3.639 σ = 0.687

Frequency(Count)

0.8

1000

#### 3.1 Training and Evaluation Details

800

0.6

Model Configuration. We experiment with Qwen2.5-Coder(Hui et al., 2024) models at 7B, 14B, and 32B scales, starting from base checkpoints without prior instruction tuning and applying identical self-bootstrapping procedures across all scales for fair comparison.

600

0.4

400

Median 3.617

0.2

200

0

0.0

0 1 2 3 4 5 6

0 1 2 3 4 5 6

Lexical Entropy (bits)

Figure 4: Lexical entropy distribution of 16,867 generated problems. Histogram with KDE shows perproblem entropy; CDF (green) and boxplot show cumulative coverage.

Training Hyperparameters. We employ consistent training settings across experiments. Models are fine-tuned for 3 epochs per iteration using AdamW with a learning rate of 5e-6 and a cosine decay schedule. We use a batch size of 128 with gradient accumulation to fit memory constraints.

et al., 2024) covers diverse real-world scenarios. We report Pass@1 accuracy with execution-based validation; solutions must pass all test cases.

#### 3.2 Main Results

Evaluation Benchmarks. We evaluate on six benchmarks: HumanEval (Chen et al., 2021b) and MBPP/MBPP+ (Austin et al., 2021) assess classic Python programming; LiveCodeBench (Jain et al., 2024) provides contamination-free competitive programming problems; BigCodeBench (BCB) (Zhuo et al., 2024) evaluates function completion with broader context and API usage (both Complete and Instruct variants); and FullStackBench (Liu

Table 1 demonstrates that unsupervised selfbootstrapping achieves performance comparable to supervised instruction tuning across diverse code generation benchmarks. Compared with other instruction-tuned models of similar scale, UCoder exhibits competitive or superior performance, consistently matching or exceeding Qwen2.5-CoderInstruct on challenging benchmarks including

- 2

- 4

6

8

10

SemanticCoverageScore

[Figure 61]

r = 0.664 n = 16,867 Complexity: 3.55 ± 1.60 Semantic: 3.16 ± 1.32

Linear Trend

0.025

0.050

0.075

0.100

0.125

0.150

PointDensity

[Figure 62]

- Figure 5: Complexity versus semantic coverage distribution. Color encodes density; red line shows linear trend (r = 0.664).

1.00 1.02 1.04 1.06 1.08 1.10 1.12

Perplexity (PPL)

0

10

20

30

40

50

Density

High Quality (< Q1) Medium-High (Q1-Q2)

| |
|---|

Medium-Low (Q2-Q3) Low Quality (> Q3)

| |
|---|

KDE Fit

Mean: 1.03 N = 9,700 Std = 0.03

| | | | | | |
|---|---|---|---|---|---|
| | |Hi|gh Quality 8|0%| |
| | | | | | |
| | | |Threshold 5|0%| |
| | | | | | |
| | | | | | |
| | | | | | |

1.00 1.05 1.10 1.15 1.20

Perplexity (PPL)

0.0

0.2

0.4

0.6

0.8

1.0

SuccessRate

- Figure 6: Quality distribution characterization. Top: perplexity distribution across 9,700 samples with quartile stratification and KDE overlay (truncated at 1.12). Bottom: perplexity versus execution success rate, showing high-quality samples (80%+ success) concentrated below perplexity 1.05.

MBPP+, BigCodeBench-Complete, and FullStackBench at all model scales (7B, 14B, 32B). Although certain Instruct models retain advantages on HumanEval, our approach progressively narrows this gap as model scale increases. These results indicate that execution-driven self-training can effectively elicit the latent instruction-following capabilities embedded in pre-trained code models, achieving supervised-level performance without requiring any human annotations or curated instruction data.

- 3.3 Effects of Iterative Self-Improvement

0 2 4 6 8 10

Complexity Score

- Table 2 reports performance across six selfbootstrapping iterations at three model scales, demonstrating both framework effectiveness and scale-dependent dynamics.

Framework Effectiveness. Iterative selfbootstrapping consistently improves performance without external supervision. Across benchmarks, all model scales show substantial gains over seed-trained baselines (Iter 0), with improvements of +6.1 to +13.2 points at 7B, +4.3 to +10.6 at 14B, and +3.0 to +4.9 at 32B. Gains are most pronounced on benchmarks requiring diverse programming skills, such as MBPP, FullStackBench, and LiveCodeBench, rather than narrowly scoped tasks like HumanEval. This supports our hypothesis that self-generated problem diversity combined with execution-driven consensus expands capability coverage beyond seed data.

Inverse Scaling of Improvement. Performance gains exhibit an inverse scaling trend, with smaller models benefiting disproportionately. We attribute this effect to latent capability gaps, where pretrained knowledge is only partially accessible through standard prompting. Consensus-based selection mitigates this gap by reinforcing correct patterns that smaller models generate inconsistently. Notably, the self-improved 7B model reaches 85.2% on MBPP, approaching the 32B baseline (86.2%), highlighting self-bootstrapping as a compute-efficient alternative to model scaling.

Convergence Characteristics. The optimal number of iterations decreases with model scale—six for 7B, five for 14B, and four for 32B. Beyond these points, performance exhibits mild oscillation rather than degradation, reflecting a trade-off between specialization on synthetic data and generalization to held-out distributions. These observations motivate early stopping based on validation performance.

### 4 Analysis

#### 4.1 Diversity of Self-Generated Problems

Lexical Diversity. We quantify lexical diversity using Shannon entropy H = − i pi log2 pi, where pi denotes token probability. Figure 4 shows a near-Gaussian entropy distribution (mean µ = 3.64 bits, σ = 0.69, median 3.62), indicating natural variation rather than templated construction. The smooth CDF and moderate interquartile range further suggest balanced coverage from concise to elaborate specifications.

Semantic Coverage. As shown in Figure 9 (Appendix), the generated problems contain 229 domain-specific terms across seven categories, with

HumanEval MBPP BCB-Complete BCB-Instruct LiveCode- FullStack-

Method

HE HE+ MBPP MBPP+ Full Hard Full Hard Bench Bench

###### 7B Models

UCoder 77.4 67.1 72.0 63.0 44.4 15.5 34.6 14.9 13.0 40.25 Random 73.8 64.6 65.6 56.6 45.5 14.9 37.5 13.5 10.7 33.55 Cluster 73.8 66.5 68.8 58.5 41.3 12.2 36.5 12.2 10.7 38.35 Low PPL 77.4 70.7 70.1 59.3 45.6 17.6 37.3 12.2 12.2 37.11 Success Rate 75.0 67.7 66.9 57.1 41.4 11.5 33.0 12.2 12.2 40.01

###### 14B Models

UCoder 83.5 76.8 75.9 64.0 53.3 23.6 43.2 16.2 22.1 50.09 Random 81.7 76.2 71.7 60.1 53.8 20.9 43.3 17.6 16.8 47.95

- Cluster 82.3 76.8 73.0 64.0 50.4 18.2 42.4 15.5 17.6 48.55 Low PPL 82.9 77.4 73.3 63.5 50.7 20.3 42.2 23.0 19.1 49.32 Success Rate 82.3 75.6 74.1 63.2 50.6 20.9 41.3 14.9 17.6 47.36

32B Models

UCoder 86.0 78.0 86.2 72.2 54.8 28.4 44.6 18.9 22.1 53.05 Random 84.8 79.3 80.7 69.6 53.7 25.0 46.6 21.6 15.3 39.18

- Cluster 83.5 75.6 81.5 71.2 53.9 23.0 45.1 16.9 13.0 49.91 Low PPL 84.1 76.2 79.9 67.5 53.5 27.7 45.4 23.0 18.3 50.09 Success Rate 82.3 75.6 81.2 69.6 54.1 27.7 44.2 17.6 21.4 50.50

- Table 3: Ablation study comparing data selection strategies across model scales: UCoder (execution-driven consensus), Random (random sampling from successful solutions), Cluster (clustering-based), Low PPL (lowest perplexity), and Success Rate (weighted by execution success). All metrics show Pass@1 execution rates (%). Bold/Underline indicate best/second-best performance per size category.

Data Structures (18.3%), Algorithms (14.8%), and String Processing (11.4%) being most prominent. No category exceeds 20%, demonstrating broad semantic coverage, while concrete algorithmic terms (e.g., dijkstra, greedy, traversal) indicate non-generic, verifiable challenges.

Complexity Distribution. We assess problem difficulty and conceptual breadth using a Complexity Score (0–10, aggregating parameter count, description length, algorithmic keywords, and constraints) and a Semantic Coverage Score (weighted keyword matches across seven categories). Figure 5 shows a moderate positive correlation (r = 0.664), indicating that more complex problems tend to integrate multiple concepts. Problems span the full score ranges (complexity: 3.55 ± 1.60, semantic: 3.16 ± 1.32) with a continuous density profile, suggesting a natural difficulty continuum suitable for curriculum learning.

#### 4.2 Execution-Driven Consensus Effectiveness

Solution Space Diversity. We first examine whether dense sampling with n = 128 candidates yields a sufficiently diverse solution pool. As shown in Figure 8, the abstract syntax tree (AST) node distribution spans 15 syntactic constructs across 2.6M generated samples (212M total nodes), indicating diverse implementation structures beyond primitive elements. The joint distri-

bution of cyclomatic complexity and code length further shows broad dispersion (mean complexity 2.7 ± 2.3, mean length 22.4 ± 10.5 lines), confirming substantial heterogeneity in the solution space.

Quality Distribution Characterization. Given this diversity, we examine whether solution quality exhibits sufficient separation for reliable selection. Figure 6 shows the perplexity distribution over 9,700 sampled candidates and its relationship with execution success. The distribution is rightskewed (mean 1.03, std 0.03) with clear stratification: high-quality samples concentrate at low perplexity values around 1.01, while lower-quality samples progressively shift toward higher perplexity, extending beyond 1.10. Consistently, solutions with execution success rates above 80% cluster predominantly below perplexity 1.05, with rapid performance degradation beyond this range. This sharp transition indicates that high-quality solutions form a distinct and identifiable subset within the candidate pool.

Quality Improvement. Consensus-based filtering yields substantial improvements across all quality metrics. In Figure 7, the filtered dataset consistently outperforms the full dataset in success rate, error-free rate, and code uniqueness. Besides, these gains are achieved without sacrificing diversity, where structural and semantic entropy remain

###### Structure Entropy

|.1|4|
|---|---|
| | |
|.8|4|
| | |

100%

75%

###### Semantic Entropy

###### Error-Free Rate

50%

1.00

2.68

2.74

25%

0.43

0.46

0.63

0.98

1.00

Success Rate

Code Uniqueness

All Dataset Filtered Dataset

- Figure 7: Filtered dataset (green) improves quality over full dataset (gray) while maintaining diversity. comparable between the two datasets. This demonstrates that execution-driven consensus selection effectively resolves the quality–diversity trade-off.

#### 4.3 Ablation Study

To isolate the effect of data selection, we compare our consensus-based approach with four alternatives: Random uniform sampling, Cluster selection from dominant output-hash clusters, Low PPL selection based on minimal perplexity, and Success Rate filtering with a 0.5 execution threshold. Table 3 shows our approach consistently outperforms all baselines across three model scales, achieving the best results on most benchmarks. The performance gap widens with scale, on FullStackBench, the margin over Random increases from 6.7 points at 7B to 13.9 points at 32B, suggesting larger models produce more diverse candidate pools where principled selection matters most. While individual baselines excel on specific benchmarks, none matches the robustness of consensus-based selection. Notably, Success Rate filtering fails to consistently outperform Random, indicating binary pass/fail criteria inadequately capture quality differences, whereas consensus-based selection leverages behavioral consistency across test inputs as a richer quality signal.

### 5 Related Work

Unsupervised Learning for Code Post-training. Unsupervised learning has become increasingly prominent in code generation through pre-training on vast unlabeled code repositories, building on

early findings that source code exhibits statistical regularities similar to natural language (Rozière et al., 2023; Guo et al., 2024; Li et al., 2023; Lozhkov et al., 2024; Zhang et al., 2025a,b). Recent work on unsupervised code post-training has focused on leveraging unlabeled code snippets and use LLM to generate synthetic question-answering data. Magicoder (Wei et al., 2023) uses opensource code examples to teach LLMs how to create varied coding instructions and training data. Besides, WizardCoder (Luo et al., 2023) progressively evolves simple coding instructions into complex ones for training. Further, WaveCoder (Yang et al.,

- 2024) and CodeArena (Yang et al., 2024) generate more diverse and high-quality instruction data from the open source code dataset. Besides, there are some works (Pravilov et al., 2021; Ahmad et al.,

2023) that adopt supervised learning for code translation and code change tasks.

Code Instruction Tuning. Instruction tuning (Huang et al., 2025; Hui et al., 2024; Yang et al.,

- 2025) has emerged as an effective approach for improving LLMs by fine-tuning them on instructionbased datasets, enabling better generalization and more accurate instruction-following capabilities. To improve the capabilities of code LLM, researchers have enhanced multiple code tasks and benchmarks, such as code generation of multiple domains (e.g., BigCodeBench (Zhuo et al., 2024) and FullStackBench (Liu et al., 2024)), multilingual code generation (e.g., MultiPl-E (Cassano et al., 2023) and McEval (Chai et al., 2024)), and competitive programming problems (e.g., LiveCodeBench (Jain et al., 2024)). 6 Conclusion

In this work, we introduce an execution-driven self-bootstrapping framework that removes the need for human-annotated instruction data in code generation. UCoder exploits latent programming knowledge in pre-trained models and uses execution feedback as a scalable, deterministic supervision signal for autonomous improvement. Guided by execution-driven consensus clustering, iterative self-training identifies correct implementations via behavioral consistency and constructs high-quality training data. Our results demonstrate that code models can achieve performance competitive with supervised baselines through fully autonomous learning, highlighting a scalable and cost-effective path for advancing code intelligence.

### Limitations

Despite the promising results, our work has several limitations that warrant future investigation. First, the effectiveness of our consensus-based selection mechanism relies on the availability of executable test cases, which may not always be feasible for certain programming tasks or domains where formal specifications are difficult to construct. Second, while our method demonstrates strong performance on standard benchmarks, the computational cost of generating and evaluating 128 candidate solutions per problem remains substantial, potentially limiting its applicability in resource-constrained scenarios. Third, our approach primarily focuses on functional correctness through execution-based validation, which may not capture other important code quality attributes such as maintainability, documentation quality, or adherence to specific coding standards beyond those explicitly testable. Fourth, the iterative self-training process exhibits diminishing returns and potential overfitting to synthetic data distributions after a certain number of iterations, requiring careful validation-based early stopping. Finally, our analysis is primarily conducted on Python programming tasks, and the generalizability of our findings to other programming languages with different execution characteristics and paradigms remains to be thoroughly validated. These limitations highlight important directions for future work in unsupervised code generation.

### Ethics Statement

This work on unsupervised code generation acknowledges several ethical considerations. Automated code generation systems can be misused to produce malicious code or security vulnerabilities, requiring appropriate safeguards, including content filtering and usage monitoring. All experiments used publicly available benchmarks and open-source models, ensuring transparency without collecting proprietary data. We advocate for responsible development with clear documentation, transparent methodologies, and adherence to intellectual property rights.

### Acknowledgment

This work was supported by State Key Laboratory of Complex & Critical Software Environment (SKLCCSE) of Beihang University and supported by the Fundamental Research Funds for the Central

Universities. This work was supported in part by the National Natural Science Foundation of China (Grant Nos. 62276017, 62406033, U1636211, 61672081), and the State Key Laboratory of Complex & Critical Software Environment (Grant No. SKLCCSE-2024ZX-18).

### References

Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2023. Summarize and generate to back-translate: Unsupervised translation of programming languages. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1528–1542.

Anthropic. 2025. Introducing claude sonnet 4.5.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.

Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. 2023. Multipl-e: A scalable and polyglot approach to benchmarking neural code generation. IEEE Transactions on Software Engineering, 49(7):3675–3691.

Linzheng Chai, Shukai Liu, Jian Yang, Yuwei Yin, Ke Jin, Jiaheng Liu, Tao Sun, Ge Zhang, Changyu Ren, Hongcheng Guo, et al. 2024. Mceval: Massively multilingual code evaluation. arXiv preprint arXiv:2406.07436.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021a. Evaluating large language models trained on code.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph,

Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021b. Evaluating large language models trained on code. abs/2107.03374.

Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V Le, Sergey Levine, and Yi Ma. 2025. Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. 2024. Deepseek-coder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196.

Siming Huang, Tianhao Cheng, Jason Klein Liu, Weidi Xu, Jiaran Hao, Liuyihan Song, Yang Xu, Jian Yang, Jiaheng Liu, Chenchen Zhang, Linzheng Chai, Ruifeng Yuan, Xianzhen Luo, Qiufeng Wang, YuanTao Fan, Qingfu Zhu, Zhaoxiang Zhang, Yang Gao, Jie Fu, Qian Liu, Houyi Li, Ge Zhang, Yuan Qi, Yinghui Xu, Wei Chu, and Zili Wang. 2025. Opencoder: The open cookbook for top-tier code large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 33167– 33193. Association for Computational Linguistics.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. 2024. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando SolarLezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. CoRR, abs/2403.07974.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel

Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy V, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour MoustafaFahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you! CoRR, abs/2305.06161.

Siyao Liu, He Zhu, Jerry Liu, Shulin Xin, Aoyan Li, Rui Long, Li Chen, Jack Yang, Jinxiang Xia, ZY Peng, et al. 2024. Fullstack bench: Evaluating llms as full stack coder. arXiv preprint arXiv:2412.00535.

Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. 2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. 2023. Wizardcoder: Empowering code large language models with evolinstruct. CoRR, abs/2306.08568.

OpenAI. 2025. Introducing upgrades to codex: Gpt-5-codex. https://openai.com/index/ introducing-upgrades-to-codex/.

Mikhail Pravilov, Egor Bogomolov, Yaroslav Golubev, and Timofey Bryksin. 2021. Unsupervised learning of general-purpose embeddings for code changes. In Proceedings of the 5th international workshop on machine learning techniques for software quality evolution, pages 7–12.

Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton-Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. 2023. Code llama: Open foundation models for code. CoRR, abs/2308.12950.

Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. 2023. Magicoder: Source code is all you need. CoRR, abs/2312.02120.

Jian Yang, Xianglong Liu, Weifeng Lv, Ken Deng, Shawn Guo, Lin Jing, Yizhi Li, Shark Liu, Xianzhen Luo, Yuyu Luo, Changzai Pan, Ensheng Shi, Yingshui Tan, Renshuai Tao, Jiajun Wu, Xianjie Wu, Zhenhe Wu, Daoguang Zan, Chenchen

Zhang, Wei Zhang, He Zhu, Terry Yue Zhuo, Kerui Cao, Xianfu Cheng, Jun Dong, Shengjie Fang, Zhiwei Fei, Xiangyuan Guan, Qipeng Guo, Zhiguang Han, Joseph James, Tianqi Luo, Renyuan Li, Yuhang Li, Yiming Liang, Congnan Liu, Jiaheng Liu, Qian Liu, Ruitong Liu, Tyler Loakman, Xiangxin Meng, Chuang Peng, Tianhao Peng, Jiajun Shi, Mingjie Tang, Boyang Wang, Haowen Wang, Yunli Wang, Fanglin Xu, Zihan Xu, Fei Yuan, Ge Zhang, Jiayi Zhang, Xinhao Zhang, Wangchunshu Zhou, Hualei Zhu, King Zhu, Bryan Dai, Aishan Liu, Zhoujun Li, Chenghua Lin, Tianyu Liu, Chao Peng, Kai Shen, Libo Qin, Shuangyong Song, Zizheng Zhan, Jiajun Zhang, Jie Zhang, Zhaoxiang Zhang, and Bo Zheng. 2025. From code foundation models to agents and applications: A comprehensive survey and practical guide to code intelligence.

Jian Yang, Jiaxi Yang, Ke Jin, Yibo Miao, Lei Zhang, Liqun Yang, Zeyu Cui, Yichang Zhang, Binyuan Hui, and Junyang Lin. 2024. Evaluating and aligning codellms on human preference. arXiv preprint arXiv:2412.05210.

Junjie Ye, Yuming Yang, Yang Nan, Shuo Li, Qi Zhang, Tao Gui, Xuan-Jing Huang, Peng Wang, Zhongchao Shi, and Jianping Fan. 2025. Analyzing the effects of supervised fine-tuning on model knowledge from token and parameter levels. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 471–513.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?, 2025. URL https://arxiv. org/abs/2504.13837.

Wei Zhang, Jack Yang, Renshuai Tao, Lingzheng Chai, Shawn Guo, Jiajun Wu, Xiaoming Chen, Ganqu Cui, Ning Ding, Xander Xu, Hu Wei, and Bowen Zhou. 2025a. V-gamegym: Visual game generation for code large language models.

Wei Zhang, Jian Yang, Jiaxi Yang, Ya Wang, Zhoujun Li, Zeyu Cui, Binyuan Hui, and Junyang Lin. 2025b. Turning the tide: Repository-based code reflection.

Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, et al. 2024. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. arXiv preprint arXiv:2406.15877.

### A Additional Analysis Figures

45

3,427,900 Statistics Total AST Nodes:

BinOp

[Figure 63]

40

40

4,556,399

Expr

212,455,883

4,673,438

alias

Unique Types: 15 Code Samples: 2,647,388

35

35

4,715,098

arg

##### LinesofCode

###### LinesofCode

30

30

6,768,596

Attribute

7,789,009

Assign

25

25

8,906,025

Subscript

20

20

11,158,213

Call

13,282,571

Store

15

15

Diversity Metrics Complexity: 2.7 ± 2.3 Length: 22.4 ± 10.5 Samples: 2,647,388

13,421,709

Constant

10

59,034,190

Name

10

65,235,666

Load

5

0 1 2 3 4 5 6 7

2 4 6

Cyclomatic Complexity

Frequency (×107)

- Figure 8: Solution space diversity analysis. Left: AST node type distribution across 2.6M samples totaling 212M nodes, spanning 15 distinct syntactic constructs. Right: Cyclomatic complexity versus code length with density contours, showing broad dispersion (complexity: 2.7 ± 2.3, length: 22.4 ± 10.5 lines).

[Figure 64]

Top 229 most frequent words colored by semantic category

Semantic Categories

| |
|---|

Data Structures Algorithms

| |
|---|

| |
|---|

String Processing Math Operations

| |
|---|

| |
|---|

Control Flow Io Operations

| |
|---|

| |
|---|

Application Domains Others

| |
|---|

- Figure 9: Semantic distribution of 229 frequent terms in generated problems. Word size indicates frequency; colors denote categories.

