## AdaptMMBench: Benchmarking Adaptive Multimodal Reasoning for Mode Selection and Reasoning Process

# arXiv:2602.02676v3[cs.CV]8Apr2026

Xintong Zhang1,2,∗ Xiaowen Zhang3,2,∗ Jingrong Wu2,∗ Zhi Gao1,2,4,†,♣ Shilin Yan5,† Zhenxin Diao1,‡ Kunpeng Gao1,‡ Xuanyan Chen1,‡ Yuwei Wu1,4,♣ Yunde Jia4 Qing Li2,♣

1Beijing Key Laboratory of Intelligent Information Technology, School of Computer Science & Technology, Beijing Institute of Technology 2State Key Laboratory of General Artificial Intelligence, BIGAI 3Xidian University 4Guangdong Laboratory of Machine Perception and Intelligent Computing, Shenzhen MSU-BIT University 5Alibaba Group ∗ Core contribution, † Project supervisor, ‡ Equal contribution, ♣ Corresponding authors

Project Page: https://adaptmmbench.github.io/

[Figure 1]

### Abstract

### 1. Introduction

Adaptive multimodal reasoning has emerged as a promising frontier in Vision-Language Models (VLMs), aiming to dynamically modulate between tool-augmented visual reasoning and text reasoning to enhance both effectiveness and efficiency. However, existing evaluations rely on static difficulty labels and simplistic metrics, which fail to capture the dynamic nature of difficulty relative to varying model capacities. Consequently, they obscure the distinction between adaptive mode selection and general performance while neglecting fine-grained process analyses. In this paper, we propose AdaptMMBench, a comprehensive benchmark for adaptive multimodal reasoning across five domains: real-world, OCR, GUI, knowledge, and math, encompassing both direct perception and complex reasoning tasks. AdaptMMBench utilizes a Matthews Correlation Coefficient (MCC) metric to evaluate the selection rationality of different reasoning modes, isolating this meta-cognition ability by dynamically identifying task difficulties based on models’ capability boundaries. Moreover, AdaptMMBench facilitates multi-dimensional process evaluation across key step coverage, tool effectiveness, and computational efficiency. Our evaluation reveals that while adaptive mode selection scales with model capacity, it notably decouples from final accuracy. Conversely, key step coverage aligns with performance, though tool effectiveness remains highly inconsistent across model architectures.

C

o

r

r

e

c

t

n

e

s

s

M o

- n

c

y

)

A

c

c

u

r

a

c

y

T

o

k

e

n

E

f

f

i

c

i

e

n

c

y

[Figure 2]

M

C C

T

o

o

l

E

f

f

e

c

t

i

v

e

n

e

s

s

K

e

y

S

t

e

p

C

o

v

e

r

a

g

e

- R e

a

s

o

n

i

n

g

- S

t

e

p

s

T

- o

- d

- e

e

i

c

S e

i

f

f

E

l e c

n

(

o

s

t

i o n

i

t

a

l

o

c

- n

v

- o

I

)

y

- n

g

P

- r

o

c

e

- s

t

i

i

n

l

a

o

u

s

Q

a

e

(

R

s

s

e

c

o

r

P

g

n

i

n

o

s

a

e

R

Thyme Adaptvision Deepeyes v2

Gpt-5 Gemini-3-pro Qwen3-VL-235B

Figure 1. Comparative Analysis of Accuracy, Reasoning Mode Selection, and Reasoning Process. Closed-source models achieve stronger performance in accuracy and mode selection, while reasoning process quality is analyzed on open-source models due to limited access to closed-source reasoning traces.

Vision Language Models (VLMs) have evolved from passive observers of static visual inputs to proactive models capable of dynamic information seeking. This evolution makes a shift from direct perception and textual chain-ofthought (CoT) to the tool-augmented visual reasoning (i.e., thinking with images) (OpenAI, 2025), where models iteratively manipulate the visual content using visual tools, such as zoom-in, and enhancement(e.g., contrast and rotation) to acquire more visual information and resolve ambiguities. (Zheng et al., 2025; Hu et al., 2024). However, this capability introduces significant computational redundancy.

Correspondence to: Zhi Gao <gaozhibit@bit.edu.cn>, Yuwei Wu <wuyuwei@bit.edu.cn>, Qing Li <dylan.liqing@gmail.com>.

Preprint. April 9, 2026.

Lacking a mechanism to discern task necessity, models often fall into a ‘tool-invocation’ trap, applying intensive visual tools to tasks solvable by direct perception or text reasoning. Consequently, adaptive multimodal reasoning is a promising direction for VLMs, which balances the necessity of such tool-augmented visual reasoning against text reasoning (Lin et al., 2025b; Wang et al., 2025a).

Despite the emergence of adaptive multimodal reasoning formulations, evaluating adaptive multimodal reasoning remains an open problem. Most existing evaluations rely on token-level reduction, coarse tool-call statistics, or final accuracy as proxies for adaptive intelligence. While intuitive, these metrics primarily reflect observable outcomes rather than evaluating the internal reasoning process itself. In particular, they fail to disentangle adaptive reasoning mode selection from subsequent reasoning execution. The ability to select an appropriate reasoning mode is crucial, as it reflects difficulty-aware meta-cognition. From the data perspective, adaptive reasoning is commonly evaluated on domain-specific logic tasks (e.g., math and knowledge reasoning) or high-resolution perception benchmarks (Wu & Xie, 2024; Wang et al., 2025c; Zhang et al., 2025b). These benchmarks lack a hierarchy of difficulty, limiting their effectiveness in evaluating adaptive reasoning. Recent efforts such as Omni-AutoThink (Yang et al., 2025a) attempt to quantify adaptiveness through thinking rates under predefined difficulty levels, as shown in Fig. 2. While this encourages increased reasoning effort on harder tasks, predefined difficulty levels are not universally applicable across models, leading to evaluation bias. Moreover, existing evaluations largely overlook reasoning process quality, losing detailed analyzes to guide future multimodal reasoning research.

To bridge these gaps, we propose AdaptMMBench to quantify adaptive multimodal reasoning in VLMs. AdaptMMBench includes 1420 samples across five domains: realworld, OCR, GUI, knowledge, and math. Each domain contains both text-only solvable tasks and complex scenarios of varying difficulties requiring proactive visual tool invocation. AdaptMMBench enables separate evaluation of adaptive reasoning mode selection and the reasoning process. Specifically, it adopts the Matthews Correlation Coefficient (MCC) to evaluate mode selection by dynamically identifying task difficulties based on model performance boundaries. For reasoning process evaluation, we assess key step coverage, tool invocation effectiveness, and efficiency to measure reasoning coherence, tool correctness, and computational cost alongside accuracy.

We evaluate closed-source and open-source VLMs on AdaptMMBench, results shown in Fig. 1. Experiments reveal a relatively weak correlation between adaptive mode selection performance and final accuracy, whereas closed-source and larger models demonstrate stronger adaptive capability. By

contrast, key step coverage correlates more closely with accuracy, and tool execution effectiveness varies substantially across models.

Our contributions are summarized as follows.

- (1) We propose the AdaptMMBench to quantify the adaptive multimodal reasoning capabilities of VLMs, which contains 1420 samples across five domains with detailed reasoning annotations for comprehensive evaluations.
- (2) We establish a suite of metrics for adaptive multimodal reasoning, which disentangle the adaptive capability from other model capabilities and assess three aspects of the reasoning process, providing detailed and in-depth evaluations.
- (3) We analyze current VLMs from the perspective of adaptive reasoning, highlighting that the relationship between mode selection performance and final accuracy is relatively small, while closed-source and larger models exhibit stronger adaptive behavior. In contrast, key step coverage correlates more closely with accuracy, and tool execution effectiveness varies substantially across models.

### 2. Related Work

#### 2.1. Multimodal Reasoning in VLMs

Early VLMs predominantly rely on text-only reasoning over fixed visual encodings, imposing a “first-glance” bottleneck that limits access to fine-grained visual details (Lu et al., 2023; Huang et al., 2025; Zhang et al., 2023; Yang et al., 2025b). Recent advanced models, including GPT5 (Singh et al., 2025), Qwen3-VL (Bai et al., 2025), and InternVL (Zhu et al., 2025) have shifted multimodal reasoning from passive visual interpretation toward active, tool-augmented information seeking. Under this “thinking with images” paradigm, models acquire additional visual information through mechanisms such as multi-turn visual search (OpenAI, 2025; Zheng et al., 2025), region zoomin (Wang et al., 2025b; Lai et al., 2025), and self-generated visual cues (Li et al., 2025a; Chern et al., 2025). In parallel, adaptive multimodal reasoning models have emerged to selectively invoke tools, trading off between text-only and tool-based reasoning to improve inference efficiency (Lin et al., 2025b; Zhang et al., 2025a; Wang et al., 2025a; Li et al., 2025d;e). More advanced systems further incorporate agentic workflows and code generation to support precise execution (Hong et al., 2025; Zhang et al., 2025c). While these works emphasize improvements in precision and efficiency, they offer limited evaluation of whether models invoke tool-based reasoning until text-only reasoning is insufficient, avoiding unnecessary computational overhead.

Previous work

[Figure 3]

[Figure 4]

[Figure 5]

Dataset

Easy Tooto callhard,tools.I need

It is an easy problem, you should not call tools.

[Figure 6]

[Figure 7]

Hard

Category by human or specific models

###### The difficulty of the task varies for each model.

Fixed Difficulty Distribution

Weak model

Adaptive reasoning

AdaptMMBench

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Easy

Dataset

Tools are required only when text reasoning fails, balancing cost and capability.

[Figure 12]

Weak model Adaptive reasoning

Weak model Text reasoning

[Figure 13]

Hard

[Figure 14]

Tools:

[Figure 15]

[Figure 16]

Category by model itself

[Figure 17]

[Figure 18]

Easy

Zoom-in Rotation

[Figure 19]

Strong model Adaptive reasoning

Strong model Text reasoning

[Figure 20]

[Figure 21]

Hard

Dynamic Difficulty Distribution

Contrast

Calculate

- Figure 2. Illustration of our model-specific difficulty evaluation. Existing methods rely on static difficulty levels, while difficulty is inherently model-dependent.

#### 2.2. Benchmarks for VLMs

- 3.1. Data Formulation Formally, AdaptMMBench is constructed as a set of samples

- D = {di}Ni=1, where each data sample di is defined as: di = (I,Q,A,E,K). (1)

Here, I ∈ RH×W×3 denotes the input image, Q is the textual query, and A is the ground-truth answer. To support adaptive evaluation, we provide the visual tool annotation

- E that specifies how essential visual information can be obtained, including the coordinates of target regions as well as the required image transformations such as rotation and

contrast adjustment. K = {k1,...,km} is an ordered sequence of human-verified key reasoning steps describing the solution path from (I,Q) to A.

During inference, the model only observes the image I and the query Q. Acquiring the visual information specified by E requires invoking a visual tool t(I,τ) via code execution or function calls, where t ∈ T denotes a tool from the predefined toolset and τ its execution arguments.

- 3.2. Data Collection

Traditional VLM benchmarks mainly assess multimodal reasoning in structured domains with coarse visual content, such as chart understanding (Mathew et al., 2021; Masry et al., 2022), mathematical problem solving (Lu et al., 2023; Xiao et al., 2024), and other general-purpose VQA (Liu et al., 2024; Chen et al., 2024). MME-CoT (Jiang et al., 2025) further evaluates the correctness of the text reasoning process. As VLM capabilities improve, more recent benchmarks (Wu & Xie, 2024; Wang et al., 2025c) introduce higher-resolution images to better reflect complex conditions. Building on this trend, benchmarks such as VisualProbe (Lai et al., 2025), InSight-o3 (Li et al., 2025b), and TIR-Bench (Li et al., 2025c) emphasize fine-grained visual understanding and active visual reasoning through operations like region zoom-in and iterative exploration, implicitly requiring models to “thinking with images”. In parallel, generative benchmarks including VTBench (Lin et al., 2025a) and AuxSolidMath (Guo et al., 2025a) evaluate multimodal reasoning via self-produced auxiliary visual cues, extending visual reasoning beyond the information directly available in the input image. However, these visualgrounded benchmarks (Li et al., 2025b; Lin et al., 2025a) largely focus on task accuracy, overlooking the problem of redundant computation where models use visual tools for tasks already solvable through text-only reasoning. Relying solely on token reduction for efficiency evaluation fails to evaluate the adaptive decisions and the reasoning quality.

AdaptMMBench encompasses 1,420 samples spanning five domains: real-world, OCR, GUI, math, and knowledge, enabling a comprehensive evaluation of adaptive reasoning across diverse scenarios, as detailed in Fig. 3.

To ensure that AdaptMMBench contains both samples solvable via text-only reasoning and samples that require visual tool invocation under adaptive reasoning, we deliberately construct the dataset with diverse difficulty levels during data collection. One subset consists of samples solved by Qwen2.5-VL-7B under text-only reasoning. A second subset includes samples that Qwen2.5-VL-7B fails but can be

### 3. AdaptMMBench

AdaptMMBench focuses on two perspectives: adaptive reasoning mode selection and reasoning process.

[Figure 22]

[Figure 23]

[Figure 24]

GUI

###### Real World OCR

###### i. Zoom-in Tool

###### i. Zoom-in Tool

###### i. Zoom-in Tool

Bounding Box:

Bounding Box:

[Figure 25]

[Figure 26]

[Figure 27]

Bounding Box:

[930, 599, 1387, 887]

[160, 260, 780, 325]

[650, 350, 1340, 390]

Key step:

Key step:

Key step:

- 1. Identify the cat.
- 2. Locate the cat's eyes.
- 3. Determine the irises are yellow and the pupils are black.

- 1. Identify the line's values: ...
- 2. Compute the sum: ...
- 3. Calculate the average: ...

- 1. Identify the open 'Fields' dialog in the screen.
- 2. Identify the tab row at the top of this dialog.
- 3. Count the tabs total 6.

Q: How many tabs are visible along the top of the "Fields" dialog?

Q: Average the values of the line in this chart.

Q: What color are the cat's eyes?

A: 7.99

A: Yellow with black

A: Six

###### i. Enhance Tool

###### i. Enhance Tool

i. Enhance Tool

Brighter: 5 Brighter: 2.5

Rotation: 180°

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Key step:

Q: Does clicking on the third suggestion in the search panel open a webpage?

Q: What is the zip code? Key step:

Key step:

Q: What brand name appears under tree? A: 20418

- 1. Identify the tree.
- 2. Locate the word under ...
- 3. Read the word as Somersby.

- 1. Identify the address block ...
- 2. Locate the line with city...
- 3. Extract the zip code ...

- 1. Identify the 'Suggestions' ...
- 2. Identify the third ...

A: Somersby

A: no 3. Observe this item is not ...

[Figure 34]

[Figure 35]

Math

###### Knowledge

###### i. Zoom-in Tool

i. Generation Tool

###### i. Zoom-in Tool

Bounding Box:

Bounding Box:

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[838, 65, 1561, 663]

[50, 657, 1200, 1483]

Key step:

Key step:

- 1. Identify the question...
- 2. Compare Experiment 1 vs 2 ...
- 3. Calculate the average: ...
- 4. Calculate for [A]=0.02 ...

- 1. Identify the relation criterion ...
- 2. Recall the one-to-one ...
- 3. Observe that the line y ...
- 4. Conclude the relation fails ...

Q: Answer question 465.

Q: Answer question 401.

A: B:This is not a oneto-one function

A: (B) 4.8e-4

###### i. Enhance Tool

###### i. Enhance Tool

Rotation: 180° Rotation: 135°

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Key step:

Q: As shown in the figure, the perimeter of triangle ABC is 40cm. The distance from P to each side of the triangle is 3cm. What is the area of triangle ABC? Options:

- 1. From 'the distance from P to each side is 3 cm', identify ...
- 2. Set the inradius r ...
- 3. Compute the semiperimeter ...
- 4. Use the theorem: ...
- 5. Calculate A = 3 × 20 = 60
- 6. Select the options C.

Key step:

Key step:

- 1. Identify square = 5.
- 2. Identify circle = square - 2.
- 3. Substitute square = 5 ...
- 4. Identify triangle + circle ...
- 5. Substitute circle = 3 and ...

Q: What is the value represented by the triangle?

Q: Which pigment has the greatest Rf value?

- 1. Identify the pigment bands ...
- 2. Observe that carotene ...
- 3. Recall formula: Rf = ...
- 4. The farthest band ...

(A) 50 (B) 55 (C) 60 (D) 90

A: carotene

A: 2

A: (C) 60

- Figure 3. An Overview of AdaptMMBench. The benchmark contains data from five domains. Each domain includes samples requiring zoom-in and enhancement tools. We annotate zoom-in regions, enhancement arguments, and key reasoning steps.

[Figure 45]

is made by the model itself, as detailed in Sec. 4.

Building on prior adaptive reasoning methods (Chern et al., 2025; Zhang et al., 2025c; Zhao et al., 2025), AdaptMMBench evaluates diverse visual tools beyond zoom-in, including geometric transformations for orientation correction and photometric adjustments for visual enhancement. During data construction, these requirements are induced via controlled distortions such as changes in contrast, brightness, and orientation, with zoom-in and transformation samples with a ratio of 5:2. We further include 120 samples requiring auxiliary-line generation, suggesting that reasoning with self-generated images constitutes an important extension of the think-with-images paradigm.

Realworld 21.1%

Math 22.6%

Knowledge 14.1%

OCR 21.1%

GUI

21.1%

5.3%Web

3.3. Annotation and Quality Control Visual Tool & Key Step Annotation.

Figure 4. Domains and category of AdaptMMBench.

solved by Qwen3-VL-235B based on adaptive reasoning. A small portion remains unsolved even by Qwen3-VL-235B. The relative proportions of these three subsets are approximately 24%, 70%, and 6%. Notably, these subsets are introduced only to ensure difficulty diversity and do not determine the ground truth reasoning mode during evaluation. The reasoning mode selection label in adaptive mode

We collect initial data from existing benchmarks, with annotators providing bounding-box annotations for key regions, while visual enhancement annotations are generated through predefined transformations. Distortion parameters are constrained to maintain recoverability. GPT-5 is used to generate key reasoning steps K, which are manually verified. These components form annotated quintuples

[Figure 46]

[Figure 47]

Adaptive Evaluation Reasoning Process

Key Steps Coverage Tool Execution Efectiveness

Decision Label

Oracle-visual Reasoning

- Step 1: First, I need…

- Tool 1: Parameter/code

- Tool 1 execution: Success… <image>

Step 2: Now I can …

- Tool 2: Parameter/Code

Tool 2 execution: Success… <image> Step 3: Finally… Tool 3: Parameter/Code

- Tool 3 execution: Success… <image>

[Figure 48]

[Figure 49]

Text Reasoning

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- Step 1…
- Step 2…
- Step 3…
- Step 4…
- Step 5…

[Figure 54]

[Figure 55]

[Figure 56]

Consistent

[Figure 57]

[Figure 58]

Tool Redundant

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Tool Required

[Figure 66]

[Figure 67]

Consistent

Evaluation Metric

Annotations

[Figure 68]

Adaptive Reasoning Tool-use Textual

- Key Step 1: Count…
- Key Step 2: Confirm...
- Key Step 3: Locate…

[Figure 69]

Inconsistent

[Figure 70]

[Figure 71]

Tool Required

TP FN

Eficiency

MCC

[Figure 72]

[Figure 73]

Tool Redundant

FP TN

Tokens Numbers Reasoning Turns Tool Usage Frequency

Figure 5. Evaluation pipeline for mode selection and reasoning process.

#### 4.2. Adaptive Mode Selection Evaluation

##### (I,Q,A,E,K).

Quality Control. Benchmark quality is ensured through a multi-stage verification pipeline. First, three independent annotators cross-validate each QA pair to remove ambiguity and verify correctness. Annotated image transformations and generated key reasoning steps are then reviewed by additional annotators for precision. Inaccurate instances are iteratively refined or re-annotated. This process ensures high-fidelity ground truth with precise pixel-level annotations and reliable key reasoning steps for comprehensively evaluating adaptive reasoning. More statistical information of AdaptMMBench can be found in Appendix A.

Adaptive intelligence depends on a model’s ability to assess whether its current information is sufficient to solve a task. Consequently, the appropriateness of a reasoning mode should be evaluated independently of answer correctness.

Under this principle, the necessity of tool invocation is determined by the outcome of text-only reasoning. If a task can be solved using text reasoning alone, it is labeled as Tool-Redundant, indicating that visual tool invocation is unnecessary and may introduce noise. Conversely, tasks that cannot be solved via text-only reasoning are labeled as Tool-Required, indicating that visual tool invocation is necessary to obtain additional information. This categorization defines the mode selection labels used in our evaluation, as detailed in Fig. 5. Accordingly, tool invocation decisions are evaluated using a confusion matrix: TP denotes ToolRequired cases where the model invokes tools, FN denotes Tool-Required cases where the model does not invoke tools, TN denotes Tool-Redundant cases where the model selects text-only reasoning, and FP denotes Tool-Redundant cases where the model unnecessarily invokes tools.

### 4. Evaluation Strategy

#### 4.1. Evaluation Modes

Following the formulation defined in Sec. 3.1, we define three evaluation modes to systematically assess the model’s adaptive reasoning capabilities.

- • Text-Reasoning Mode: Given (I,Q), the model relies solely on text reasoning over the given image, without invoking active visual transformations, providing a baseline for assessing tool necessity.
- • Adaptive Reasoning Mode: Given (I,Q), the model adaptively selects between text-only reasoning and tool-augmented visual reasoning with tools. It generates a reasoning trajectory and records all tool invocation parameters, enabling evaluation of both its ability to decide when tool usage is required and the correctness of the reasoning process.
- • Oracle-Visual Mode: Given (I,Q,IE), where IE denotes gold-standard visual evidence from annotation E, the model performs text-only reasoning over the provided visual evidence, providing an upper-bound performance estimate under perfect visual acquisition.

Matthews Correlation Coefficient (MCC). In adaptive mode selection, the proportions of tool-redundant and toolrequired cases are model-dependent, leading to varying degrees of class imbalance in the resulting confusion matrix. To ensure a robust evaluation, we adopt the MCC,

TP · TN − FP · FN (TP + FP)(TP + FN)(TN + FP)(TN + FN) + ϵ

, (2)

MCC =

where ϵ is a small constant for numerical stability. MCC ranges from [−1,1], with 1 indicating perfect agreement with the optimal mode selection, 0 denoting the chance-level performance, and −1 indicating complete misalignment.

Adaptive Label Robustness. We analyze the effects of minor prompt variations on text and adaptive reasoning.

Only 0.02 of samples show inconsistent outcomes between text reasoning mode and text-only reasoning in adaptive mode. This indicates that the performance difference is stable under prompt variations, and adaptive reasoning rarely degrades text-solvable samples.

#### 4.3. Reasoning Process Evaluation

While MCC measures the quality of mode selection, it does not assess the validity of the reasoning process. Models may produce correct answers despite logical errors or improper tool usage. To address this limitation, we introduce three process-oriented metrics to evaluate reasoning coherence and tool execution fidelity.

A reasoning trajectory R is formalized as an interleaved sequence of reasoning steps and tool invocations:

R = {(s1,t1),(s2,t2),...,sn}, (3)

where si is the reasoning at step i and ti ∈ T represents the corresponding tool invocation. The trajectory terminates at the final reasoning step sn, and produces the answer.

- 4.3.1. KEY STEPS COVERAGE

Following the evaluation paradigm of (Jiang et al., 2025), we assess whether a model’s reasoning chain {si}ni=1 covers the essential human-annotated key steps K defined in Sec. 3.1. We employ GPT-5 as an evaluator to identify the presence of these key steps within the generated reasoning, and define the key step coverage as:

KCoverage =

1 |K|

max

j

j

i=1

I ki ∈

GPT-5

{s1,...,sn} . (4)

This metric measures how far the model’s reasoning progresses along the key steps. Rather than penalizing skipped or compressed steps, KCoverage captures the maximum extent to which the reasoning aligns with the solution structure, allowing different reasoning styles and reflecting how close the model comes to a correct solution.

- 4.3.2. TOOL EXECUTION EFFECTIVENESS

To assess the precision of tool usage, we evaluate whether each tool invocation is semantically appropriate for its corresponding reasoning step and free of execution errors. The tool effectiveness is defined as:

1 Ntool

Effecttool =

Ntool

validGPT-5(ti | si), (5)

i=1

where Ntool denotes the total number of tool invocations, ti ∈ T is the tool invoked at step i, and validGPT-5(·) ∈ {0,1} is a semantic validity judgment provided by GPT-5.

- 4.3.3. REASONING EFFICIENCY

Efficiency is evaluated in terms of token numbers, reasoning turns, and tool usage frequency, collectively capturing the conciseness of reasoning and the computational cost of adaptive execution.

- Table 1. Evaluation of mode selection performance across models. We report TP, FP, TN, FN, and the MCC to assess meta-cognitive calibration in adaptive reasoning mode. Best and second-best scores in each category are highlighted in blue and green .

Model TP FP TN FN MCC ↑ Open-Source Models

PixelReasoner 280 196 434 390 0.11 Deepeyes 662 638 0 0 0.00 Thyme 20 20 655 605 0.01 PyVision 540 405 231 124 0.20 Deepeyes v2 623 676 1 0 0.03 AdaptVision 385 279 375 261 0.17 Qwen3-vl-8B-Instruct 328 381 351 240 0.06 Qwen3-vl-32B-Instruct 348 646 245 61 0.14 Qwen3-vl-235B-Instruct 286 437 487 90 0.26

Closed-Source Models

GPT-5 482 392 376 50 0.41 Gemini-3-Pro 284 703 296 17 0.24

- Table 2. Comprehensive evaluation of reasoning process, including key step coverage (Key Step Cov.), tool effectiveness (Tool Effect.), and efficiency. This assesses the logical rigor of the reasoning paths alongside their computational efficiency.

Model

Key Step Cov. (%) ↑

Tool Effect. (%) ↑

Efficiency Steps ↓ Tools ↓ Tokens ↓

PixelReasoner 76.02 56.51 1.37 0.37 4229.00 Deepeyes 75.56 50.99 2.00 1.68 7601.45 Thyme 77.14 56.50 1.05 0.06 6708.47 PyVision 77.43 62.02 2.76 1.76 2481.00 Deepeyes v2 75.14 56.79 2.09 1.09 6918.90 AdaptVision 72.60 81.70 1.51 0.51 4175.96 Qwen3-vl-8B 78.40 91.62 1.76 1.20 8282.40 Qwen3-vl-32B 83.79 92.98 2.42 1.44 7725.99 Qwen3-vl-235B 84.83 89.64 2.04 1.04 7531.95

- 5. Experiments

We conduct a comprehensive quantitative evaluation of adaptive reasoning on AdaptMMBench, focusing on three complementary dimensions: (i) reasoning mode selection capability, (ii) quality and efficiency of reasoning process, and (iii) accuracy across reasoning modes.

#### 5.1. Experiment Setting

We evaluate a set of VLMs to establish baselines for AdaptMMBench. For closed-source models, we select GPT5 (Singh et al., 2025) and Gemini3 (Google DeepMind, 2025). For open-source models, we include the Qwen3-VL family (Bai et al., 2025) at multiple scales (8B, 32B, and 235B). In addition, we evaluate several specialized adaptive reasoning models, including DeepEyes (Zheng et al., 2025;

Table 3. Accuracy across different domains under three reasoning modes. Results are reported on 1,300 AdaptMMBench samples, with auxiliary-line tasks evaluated separately. * indicates that the model supports enhancement operations. “w/o enh.” denotes results without enhancement-based data transformations (e.g., rotation and contrast).

Real-world OCR GUI Knowledge Math (w/o aux) Overall Accuracy w/o enh. All w/o enh. All w/o enh. All w/o enh. All w/o enh. All w/o enh. All

Model Mode

Open-Source Models

Text 42.08 38.00 58.75 58.33 48.33 48.00 56.88 55.50 46.25 43.00 50.29 48.46 Adaptive 53.75 51.33 61.25 62.33 61.67 53.00 58.13 59.00 51.88 50.00 55.19 55.23 Oracle 70.83 67.67 72.92 75.00 66.67 58.67 65.62 67.50 62.50 64.00 65.96 66.69

PixelReasoner

Text 50.00 48.33 50.42 51.33 52.50 53.33 50.62 48.50 43.12 41.50 49.71 49.15 Adaptive 54.17 53.00 57.08 57.67 51.25 52.33 53.12 50.00 55.00 50.50 54.13 53.08 Oracle 67.08 66.33 62.92 66.67 61.67 64.00 66.88 69.00 61.88 62.00 64.04 65.69

Deepeyes

Text 55.00 51.67 58.33 57.67 50.00 49.67 55.00 51.00 51.88 48.00 54.13 51.92 Adaptive 60.83 58.00 61.67 60.00 55.83 53.67 58.75 51.00 51.88 50.00 58.17 55.15 Oracle 75.83 73.33 66.67 69.00 63.75 64.67 64.38 67.50 63.12 63.00 67.21 67.85

Thyme∗

Text 40.00 38.00 62.08 60.33 75.00 70.00 35.62 34.50 35.00 31.00 51.73 48.92 Adaptive 50.83 47.33 72.50 70.67 77.92 73.33 58.75 52.50 35.00 31.00 60.87 57.00 Oracle 94.58 84.67 79.58 80.67 90.00 86.00 60.00 62.50 58.13 53.50 79.13 75.85

PyVision∗

Text 58.75 55.00 58.33 58.33 54.58 54.67 48.12 47.50 41.88 39.00 53.46 52.08 Adaptive 61.25 56.33 59.58 57.67 57.08 55.67 59.38 53.50 50.00 49.00 57.88 54.92 Oracle 75.42 74.33 70.83 74.33 68.33 69.33 65.00 65.50 63.12 64.00 69.23 70.23

Deepeyes v2∗

Text 45.83 43.00 62.92 61.67 48.75 47.67 54.37 54.00 45.00 44.50 51.63 50.31 Adaptive 49.17 46.33 64.17 64.00 52.92 54.33 60.62 54.50 49.38 45.00 55.29 53.31 Oracle 74.17 70.67 71.25 76.00 62.92 65.67 71.88 73.00 68.75 69.00 69.71 70.85

AdaptVision

Text 56.25 50.00 64.17 62.33 57.50 54.00 72.50 66.00 55.62 50.50 60.77 56.31 Adaptive 57.50 52.33 68.33 65.67 65.83 59.67 78.75 71.50 69.38 62.00 67.02 61.54 Oracle 83.75 78.00 79.17 80.67 68.75 67.00 80.62 81.00 80.00 80.00 78.17 76.85

Qwen3-vl

-8B-Instruct

Text 55.83 51.00 82.92 78.00 76.67 71.33 83.75 77.00 76.25 68.00 74.33 68.54 Adaptive 63.33 57.33 85.42 82.67 77.00 70.00 84.38 79.00 81.88 73.50 77.79 71.92 Oracle 87.08 81.67 92.92 92.00 81.67 75.67 96.25 95.00 90.00 87.50 89.04 85.62

Qwen3-vl

-32B-Instruct

Text 59.58 52.33 81.25 78.00 84.58 76.67 83.12 78.50 83.12 73.00 77.60 71.08 Adaptive 64.17 56.33 82.08 80.00 87.08 80.00 93.75 86.50 85.62 76.50 81.44 75.00 Oracle 87.92 80.67 90.83 91.33 97.08 91.33 96.25 96.50 96.88 94.00 93.37 90.08

Qwen3-vl

-235B-Instruct

Closed-Source Moddels

Text 46.67 45.67 77.08 73.67 79.17 74.33 60.00 54.50 44.38 39.00 62.88 59.08 Adaptive 70.83 64.67 89.17 86.33 88.75 85.33 92.50 86.00 76.88 71.00 83.46 78.69 Oracle 97.92 88.00 90.42 90.00 91.67 88.00 96.88 94.50 90.62 83.00 93.46 88.69

GPT-5∗

Text 59.17 53.33 87.92 87.33 90.83 86.67 85.00 83.00 81.88 75.50 80.58 76.85 Adaptive 80.42 74.00 89.58 89.67 92.08 90.00 92.50 93.50 93.12 87.00 89.04 86.31 Oracle 87.50 80.33 92.59 93.00 94.17 92.67 92.50 94.00 93.75 91.00 91.92 89.85

Gemini-3-Pro∗

Hong et al., 2025), PixelReasoner (Wang et al., 2025b), Thyme (Zhang et al., 2025c), PyVision (Zhao et al., 2025), and AdaptVision (Lin et al., 2025b). For all evaluated models, we follow the implementation details provided in their official codebases. For evaluations under different reasoning modes, we apply a unified and minimal modification to the prompts, as detailed in the Appendix D.

#### 5.2. Adaptive Reasoning Mode Selection Capability

A closer analysis of mode selection capability reveals clear differences across models. As shown in Table 1 and Table 3, mode selection capability does not exhibit a strong correlation with final task accuracy. For example, AdaptVision achieves a relatively modest accuracy, yet demonstrates strong mode selection behavior with an MCC of 0.17, outperforming all other models trained on Qwen2.5-VL-7B

backbones. In contrast, GPT-5 attains the highest MCC of 0.41, demonstrating good mode selection capability.

Model scaling improves mode selection. Table 1 demonstrates a clear scaling trend within the Qwen3-VL family, where larger models exhibit more reliable mode selection. This pattern suggests that increased model capacity contributes to improved calibration when determining whether tool-based reasoning is necessary. Similarly, large-scale closed-source models outperform open-source models.

Imbalanced mode selection behavior is observed in some models. Several specialized adaptive models exhibit imbalanced mode selection behavior, either invoking tools excessively or rarely. For example, Deepeyes v2 invokes tools in all but one of the 1,300 samples in AdaptMMBench, whereas Thyme triggers tool usage in only about 3% of

Table 4. Experimental results on geometric auxiliary-line problems across different reasoning modes.

cases. Such imbalanced patterns are associated with lower mode selection performance, despite competitive accuracy.

Model Text Acc Adaptive Acc Oracle Acc Open-Source VLMs

#### 5.3. Quality and Efficiency of the Reasoning Process

Thyme 21.67 21.67 24.17 PyVision 15.83 29.17 32.50 Deepeyes v2 19.17 19.17 25.83 Qwen3-vl-8B 50.00 46.67 62.50 Qwen3-vl-32B 63.33 58.33 79.17 Qwen3-vl-235B 62.50 68.33 84.17

Since intermediate reasoning steps of closed-source models (e.g., GPT-5 and Gemini-3-Pro) are not accessible, we restrict process-level analysis to open-source models. Table 2 evaluates key step coverage, tool effectiveness, and efficiency. Consistent with Table 3, key step coverage shows a similar ranking, with Qwen3-VL-235B among the top models. Larger models also demonstrate stronger tool effectiveness, better aligning tool usage with reasoning intent.

Closed-Source Models

Gemini-3-Pro 85.00 78.33 94.17 GPT-5 75.00 86.67 89.17

#### 5.5. Error Analysis

Tool effectiveness varies with models. Qwen3 family shows strong performance, while some smaller models are less effective. This may stem from repeated or unnecessary tool calls, as well as code-based tool invocation in Deepeyes v2, Thyme, and PyVision, which introduces more complexity than the function-call interface used by Qwen models.

[Figure 74]

Errors without Tool Invocation 13.3%

Capability Ceiling 28.8%

Token usage is not positively correlated with steps or tool calls. Considering efficiency, token usage varies across models and does not correspond to the number of reasoning steps or tool calls. For example, Thyme uses the fewest steps and tool invocations, yet consumes more tokens than PyVision, which has the most steps. This shows that fewer steps or tool calls do not necessarily reduce token cost.

Errors with Tool Invocation 57.9%

TextEnough but Noisy Tool Invocation

t

c

e

r

r

- n

c

- o

I

8.3%

g

n

i

n

%

o

Visual Reasoning correct 7.3%

- 2

.

- 3

s

a

e

4

R

l

a

u

s

i

V

#### 5.4. Accuracy across Reasoning Modes

Figure 6. Error Analysis on GPT-5.

We analyze model performance across different reasoning modes, including text-only, adaptive, and oracle tool reasoning. The oracle tool reflects upper-bound performance. As shown in Table 3, adaptive reasoning consistently improves accuracy over text-only baselines for all evaluated models.

In this section, we analyze the causes of incorrect predictions made by GPT-5 under the adaptive mode to understand the gap between adaptive reasoning and oracle-visual mode. As shown in Fig. 6, most errors are related to tool usage. Specifically, 42.3% of the errors stem from visual reasoning failures, such as zoom-in into incorrect regions or applying wrong image transformations. Another 7.3% of errors occur even when visual reasoning is correct. Since these samples are solvable in the oracle-visual mode, this suggests that intermediate images in multi-step reasoning may introduce visual noise affecting the final prediction. In addition, 8.3% of errors are caused by incorrect mode selection, where text reasoning is sufficient but the model unnecessarily invokes tools, leading to degraded performance. For cases without tool usage, forcing tool invocation corrects 7.0% of the errors, while 6.3% remain incorrect. The remaining 28.8% of errors exceed the capability of the GPT-5 model.

Significant performance gap between adaptive and oracle reasoning. Although adaptive reasoning yields clear gains, oracle tool reasoning reveals substantial remaining headroom. For example, GPT-5 improves from 78.69% under adaptive reasoning to 88.69% in the oracle setting, with similar trends observed in open-source models. These results indicate that current performance is mainly limited by imperfect tool invocation rather than reasoning capability. Moreover, the high oracle-visual accuracy of 90.08% indicates the reliability and accuracy of our visual annotations.

Generation-Based Tools Are Beneficial for Certain Tasks. We conduct an exploratory analysis on self-generated auxiliary-line tasks as shown in Table 4. As current opensource models cannot generate visual representations, adaptive reasoning shows limited or negative gains over textonly reasoning, while oracle-visual inputs bring substantial improvements. This highlights the importance of visual generation for future adaptive reasoning models.

### 6. Conclusion

In this paper, we present AdaptMMBench, a benchmark for evaluating adaptive multimodal reasoning in VLMs. AdaptMMBench covers diverse domains and reasoning

scenarios, and enables model-dependent identification of tool-redundant and tool-required cases by comparing performance across reasoning modes. We further propose a set of metrics that assess mode selection quality, reasoning process quality, and efficiency. Through systematic evaluation of state-of-the-art models, we observe that high accuracy does not necessarily imply strong reasoning mode selection capability. The substantial performance gap between adaptive and oracle-visual reasoning further suggests that performance is often limited by suboptimal tool invocation. This highlights adaptive tool selection as a key challenge for future multimodal reasoning models.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems, 37: 27056–27087, 2024.

Chern, E., Hu, Z., Chern, S., Kou, S., Su, J., Ma, Y., Deng, Z., and Liu, P. Thinking with generated images. arXiv preprint arXiv:2505.22525, 2025.

Google DeepMind. A New Era of Intelligence with Gemini 3, 2025. URL https: //blog.google/products-and-platforms/ products/gemini/gemini-3/.

Guo, S., Pang, L., Wang, X., Wang, Y., Shen, H., and Zhang, J. Geovlmath: Enhancing geometry reasoning in visionlanguage models via cross-modal reward for auxiliary line creation. arXiv preprint arXiv:2510.11020, 2025a.

Guo, Z., Zhang, R., Chen, H., Gao, J., Jiang, D., Wang,

J., and Heng, P.-A. Sciverse: Unveiling the knowledge comprehension and visual reasoning of lmms on multimodal scientific problems. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 19683– 19704, 2025b.

Hong, J., Zhao, C., Zhu, C., Lu, W., Xu, G., and Yu, X. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.

Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.

Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., and Lin, S. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Jiang, D., Zhang, R., Guo, Z., Li, Y., Qi, Y., Chen, X., Wang, L., Jin, J., Guo, C., Yan, S., et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.

Lai, X., Li, J., Li, W., Liu, T., Li, T., and Zhao, H. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.

Li, C., Wu, W., Zhang, H., Xia, Y., Mao, S., Dong, L., Vuli´c, I., and Wei, F. Imagine while reasoning in space: Multimodal visualization-of-thought. arXiv preprint arXiv:2501.07542, 2025a.

Li, K., Yao, L., Wu, J., Yu, T., Chen, J., Bai, H., Hou, L., Hong, L., Zhang, W., and Zhang, N. L. Insight-o3: Empowering multimodal foundation models with generalized visual search. arXiv preprint arXiv:2512.18745, 2025b.

Li, M., Zhong, J., Zhao, S., Zhang, H., Lin, S., Lai, Y., Wei, C., Psounis, K., and Zhang, K. Tir-bench: A comprehensive benchmark for agentic thinking-with-images reasoning. arXiv preprint arXiv:2511.01833, 2025c.

Li, X., Li, X., Gao, J., Pi, R., Hu, S., and Zhang, W. Look less, reason more: Rollout-guided adaptive pixel-space reasoning. arXiv preprint arXiv:2510.01681, 2025d.

Li, Z., Zhao, Y., Zhang, J., Wang, S., Yao, Y., Zhao, R., Song, J., Zheng, B., and Wei, Z. Mixture-of-visualthoughts: Exploring context-adaptive reasoning mode selection for general visual reasoning. arXiv preprint arXiv:2509.22746, 2025e.

Lin, H., Geng, T., Xu, Z., and Zhao, W. Vtbench: Evaluating visual tokenizers for autoregressive image generation. arXiv preprint arXiv:2505.13439, 2025a.

Lin, Z., Liu, Y., Yang, Y., Tao, L., and Ye, D. Adaptvision: Efficient vision-language models via adaptive visual acquisition. arXiv preprint arXiv:2512.03794, 2025b.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024.

Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

Masry, A., Do, X. L., Tan, J. Q., Joty, S., and Hoque, E. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pp. 2263–2279, 2022.

Masry, A., Islam, M. S., Ahmed, M., Bajaj, A., Kabir, F., Kartha, A., Laskar, M. T. R., Rahman, M., Rahman, S., Shahmohammadi, M., et al. Chartqapro: A more diverse and challenging benchmark for chart question answering. arXiv preprint arXiv:2504.05506, 2025.

Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, 2025. URL https://cdn.openai.com/pdf/ 2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf.

Qiao, R., Tan, Q., Dong, G., MinhuiWu, M., Sun, C., Song, X., Wang, J., Gongque, Z., Lei, S., Zhang, Y., et al. Wemath: Does your large multimodal model achieve humanlike mathematical reasoning? In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 20023–20070, 2025.

Shi, C., Yu, Z., Gao, Z., Feng, R., Liu, E., Wu, Y., Jia, Y., Xiang, L., He, Z., and Li, Q. Gui knowledge bench: Revealing the knowledge gap behind vlm failures in gui tasks. arXiv preprint arXiv:2510.26098, 2025.

Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Wang, C., Feng, K., Chen, D., Wang, Z., Li, Z., Gao, S., Meng, M., Zhou, X., Zhang, M., Shang, Y., et al. Adatooler-v: Adaptive tool-use for images and videos. arXiv preprint arXiv:2512.16918, 2025a.

Wang, H., Su, A., Ren, W., Lin, F., and Chen, W. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025b.

Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y., Yu, W., and Tao, D. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025c.

Wu, J., Yin, W., Jiang, Y., Wang, Z., Xi, Z., Fang, R., Zhang, L., He, Y., Zhou, D., Xie, P., et al. Webwalker: Benchmarking llms in web traversal. arXiv preprint arXiv:2501.07572, 2025.

Wu, P. and Xie, S. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024.

Xiao, Y., Sun, E., Liu, T., and Wang, W. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Xu, W., Wang, J., Wang, W., Chen, Z., Zhou, W., Yang, A., Lu, L., Li, H., Wang, X., Zhu, X., et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

Yang, D., Liu, S., Wang, D., Wang, Y., Wan, G., and Meng, H. Omni-autothink: Adaptive multimodal reasoning via reinforcement learning. arXiv preprint arXiv:2512.03783, 2025a.

Yang, S., Han, C., Luo, S., and Hovy, E. Magic-vqa: Multimodal and grounded inference with commonsense knowledge for visual question answering. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 16967–16986, 2025b.

Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

Zhang, R., Jiang, D., Zhang, Y., Lin, H., Guo, Z., Qiu, P., Zhou, A., Lu, P., Chang, K.-W., Qiao, Y., et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision, pp. 169–186. Springer, 2024.

- Zhang, X., Gao, Z., Zhang, B., Li, P., Zhang, X., Liu, Y., Yuan, T., Wu, Y., Jia, Y., Zhu, S.-C., et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025a.
- Zhang, Y., Zhang, H., Tian, H., Fu, C., Zhang, S., Wu, J., Li, F., Wang, K., Wen, Q., Zhang, Z., Wang, L., and Jin, R. Mme-realworld: Could your multimodal LLM challenge high-resolution real-world scenarios that are difficult for humans? In The Thirteenth International Conference on Learning Representations (ICLR), 2025b.

- Zhang, Y.-F., Lu, X., Yin, S., Fu, C., Chen, W., Hu, X., Wen, B., Jiang, K., Liu, C., Zhang, T., et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025c.
- Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

Zhao, S., Zhang, H., Lin, S., Li, M., Wu, Q., Zhang, K., and Wei, C. Pyvision: Agentic vision with dynamic tooling. arXiv preprint arXiv:2507.07998, 2025.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

### A. More Data Details

#### A.1. Data Source Distribution

Real-World VQA. We target high-resolution natural scenes by leveraging VisualProbe (Lai et al., 2025) for small-object search and a custom SA-1B (Kirillov et al., 2023) subset for large-scale object reasoning. Queries are explicitly designed to evaluate attributes, spatial, counting, physical state and text-recognition across distinct scales. Moreover, statistics of bounding box sizes are presented in Fig. 7.

Text-Rich VQA. This domain covers diverse charts, tables, and documents. We aggregate standard samples from ChartQA (Masry et al., 2022) and DocVQA (Mathew et al., 2021) with high-resolution challenges from ChartQA-Pro (Masry et al., 2025), MM-RealWorld (Zhang et al., 2025b), and Insight-o3 (Li et al., 2025b) to demand precise visual inspection and deep reasoning.

Math VQA. To assess mathematical reasoning in visual contexts, we consolidate high-quality samples from a spectrum of established benchmarks including MathVista (Lu et al., 2023), MathVerse (Zhang et al., 2024), We-Math (Qiao et al., 2025), LogicVista (Xiao et al., 2024), Visulogic (Xu et al., 2025), AuxSolidMath, and VTBench (Lin et al., 2025a).

GUI VQA. We construct a cross-platform suite covering iOS, Android, Web, macOS, Windows, and Linux. This is achieved by integrating generic datasets like GUI-Knowledge-Bench (Shi et al., 2025) and MMBench-GUI (Liu et al., 2024) with domain-specific samples from WebWalker (Wu et al., 2025).

Knowledge VQA. This category is sourced from disciplinary benchmarks across Physics, Chemistry, and Biology. Specifically, we incorporate expert-level samples from MMMU (Yue et al., 2024) and SciVerse (Guo et al., 2025b) to evaluate the models’ ability to integrate specialized domain knowledge with visual reasoning.

Median Ratio: 0.0428

120

400

90

300

60

200

30

100

0

0

10 6 10 5 10 4 10 3 10 2 10 1 100

0.0 0.2 0.4 0.6 0.8

BBox Area / Image Area

BBox Area / Image Area (Log Scale)

Figure 7. Statistics of bounding box sizes in AdaptMMBench.

Data Selection Data Processing Quality Verification

[Figure 75]

Domain Filter

[Figure 76]

Bounding Box Key Steps

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Tool

No Tool

[Figure 81]

[Figure 82]

Image Data Filter Transformation Annotation Human Review

Data Pool

AdaptBench

Figure 8. Overview of the data curation and annotation process.

#### A.2. Data Construction Pipeline

The construction workflow of AdaptBench is depicted in Figure 8. Initially, raw data is partitioned based on reasoning complexity, specifically separating tasks that necessitate external tool intervention from those amenable to text-only inference.

To further challenge model adaptability, we augment the visual inputs with diverse transformations, thereby mandating finegrained perception. Finally, we implement a multi-stage verification pipeline involving expert annotation of transformation logic, key reasoning steps, and rigorous human review, ensuring a high-fidelity ground truth for the final benchmark.

### B. Transform Results

Tab. 5 serves as a supplementary detailed analysis to the main experimental results presented in Tab. 3. While Tab. 3 reports the model performance on the original dataset (denoted as “w/o enh.”) and the aggregated dataset (“All”), it does not explicitly isolate the performance on the transformed data. To provide a comprehensive view of model robustness against data variations, Tab. 5 exclusively presents the accuracy results for the Transformed subset across all five domains. The highlighting scheme follows the same convention as the main table to facilitate direct comparison of the autonomous decision-making capabilities in the Adaptive mode.

Table 5. Main experimental results on the Transform subset of AdaptBench. We report accuracy (%) across five specific domains and overall aggregates. Performance for Text and Oracle modes is displayed in deep gray to prioritize adaptive results. * indicates that the model supports enhancement operations.

Model Mode Real-world OCR GUI Knowledge Math Overall Accuracy Open-Source Models

Text 46.67 50.00 30.00 56.67 21.67 41.15 Adaptive 61.67 62.50 42.50 66.67 41.67 55.38 Oracle 66.67 75.00 70.00 83.33 55.00 69.62

PixelReasoner

Text 56.67 40.00 35.00 55.00 41.67 46.92 Adaptive 56.67 37.50 32.50 60.00 48.33 48.85 Oracle 73.33 77.50 65.00 81.67 63.33 72.31

Deepeyes

Text 48.33 35.00 32.50 55.00 38.33 43.08 Adaptive 45.00 20.00 42.50 53.33 46.67 43.08 Oracle 68.33 80.00 62.50 78.33 63.33 70.38

Thyme∗

Text 50.00 30.00 15.00 53.33 30.00 37.69 Adaptive 55.00 27.50 15.00 63.33 33.33 41.54 Oracle 70.00 72.50 35.00 85.00 45.00 62.69

PyVision∗

Text 55.00 45.00 27.50 58.33 40.00 46.54 Adaptive 50.00 30.00 45.00 50.00 36.67 43.08 Oracle 73.33 67.50 67.50 88.33 70.00 74.23

Deepeyes v2∗

Text 43.33 52.50 42.50 56.67 31.67 45.00 Adaptive 60.00 30.00 27.50 63.33 35.00 45.38 Oracle 76.67 77.50 70.00 95.00 56.67 75.38

AdaptVision

Text 40.00 40.00 30.00 55.00 25.00 38.46 Adaptive 35.00 42.50 32.50 55.00 31.67 39.62 Oracle 60.00 82.50 80.00 86.67 55.00 71.54

Qwen3-VL

-8B-Instruct

Text 50.00 50.00 35.00 58.33 31.67 45.38 Adaptive 40.00 57.50 40.00 71.67 33.33 48.46 Oracle 51.67 90.00 77.50 88.33 60.00 71.92

Qwen3-VL

-32B-Instruct

Text 45.00 60.00 32.50 65.00 23.33 45.00 Adaptive 51.67 57.50 40.00 71.67 25.00 49.23 Oracle 68.33 97.50 82.50 93.33 51.67 76.92

Qwen3-VL

-235B-Instruct

Closed-Source Models

Text 55.00 32.50 17.50 60.00 41.67 43.85 Adaptive 71.67 60.00 47.50 75.00 40.00 59.62 Oracle 73.33 85.00 52.50 88.33 48.33 69.62

GPT-5∗

Text 70.00 75.00 50.00 85.00 30.00 61.92 Adaptive 81.67 97.50 62.50 90.00 48.33 75.38 Oracle 86.67 100.00 80.00 95.00 51.67 81.54

Gemini-3-Pro∗

### C. Category Results

In this section, we provide a fine-grained analysis of model performance across all specific categories defined in our benchmark. To ensure legibility and accommodate the wide range of sub-domains, the detailed accuracy results are presented in two separate tables:

- • Tab. 6 reports the performance metrics for the GUI and Realworld domains.
- • Tab. 7 covers the Knowledge, Math, and OCR domains.

In both tables, the first row (labeled as N) denotes the number of test samples available for each corresponding category. All accuracy values are reported in decimal format.

Table 6. Detailed Accuracy (Part 1/2): GUI and Realworld domains. * indicates that the model supports enhancement operations.

GUI Realworld Model Mode And. Lin. Mac. Web Win. iOS Attr. Count Integ. Phys. Spat. N (Count) 83 22 39 75 51 30 118 16 121 10 35

Text 0.51 0.45 0.31 0.52 0.53 0.47 0.45 0.12 0.35 0.50 0.34 Adaptive 0.51 0.50 0.41 0.61 0.57 0.50 0.55 0.50 0.49 0.70 0.43 Oracle 0.59 0.68 0.36 0.63 0.67 0.57 0.68 0.75 0.69 0.50 0.63

PixelReasoner

Text 0.59 0.45 0.36 0.63 0.45 0.57 0.55 0.38 0.44 0.30 0.51 Adaptive 0.48 0.36 0.44 0.68 0.49 0.53 0.58 0.44 0.50 0.30 0.57 Oracle 0.67 0.50 0.46 0.75 0.63 0.63 0.65 0.69 0.69 0.40 0.69

Deepeyes

Text 0.53 0.45 0.36 0.60 0.41 0.50 0.57 0.62 0.51 0.40 0.34 Adaptive 0.54 0.55 0.41 0.64 0.45 0.57 0.64 0.75 0.54 0.40 0.51 Oracle 0.61 0.68 0.56 0.80 0.55 0.60 0.73 0.75 0.79 0.60 0.60

Thyme∗

Text 0.81 0.64 0.82 0.47 0.76 0.77 0.51 0.44 0.24 0.40 0.40 Adaptive 0.83 0.59 0.82 0.51 0.80 0.90 0.58 0.44 0.36 0.50 0.51 Oracle 0.92 0.77 0.92 0.79 0.88 0.83 0.87 0.88 0.80 0.70 0.94

PyVision∗

Text 0.46 0.45 0.46 0.68 0.59 0.57 0.60 0.50 0.52 0.70 0.46 Adaptive 0.49 0.45 0.44 0.71 0.53 0.63 0.58 0.69 0.55 0.50 0.49

Deepeyes v2∗

- Oracle 0.64 0.68 0.51 0.77 0.76 0.77 0.72 0.81 0.79 0.80 0.63

AdaptVision

Text 0.48 0.32 0.28 0.59 0.49 0.53 0.48 0.25 0.40 0.60 0.40 Adaptive 0.51 0.36 0.41 0.68 0.55 0.60 0.54 0.25 0.40 0.60 0.46

- Oracle 0.65 0.50 0.51 0.76 0.71 0.63 0.69 0.69 0.75 0.80 0.60

Text 0.52 0.45 0.49 0.64 0.51 0.53 0.55 0.44 0.47 0.70 0.40 Adaptive 0.61 0.64 0.46 0.63 0.57 0.67 0.58 0.50 0.44 0.70 0.57 Oracle 0.70 0.59 0.51 0.77 0.67 0.60 0.70 0.88 0.84 0.80 0.77

Qwen3-VL

-8B-Instruct

Text 0.81 0.77 0.56 0.75 0.57 0.77 0.56 0.56 0.50 0.50 0.34 Adaptive 0.76 0.77 0.51 0.68 0.67 0.83 0.64 0.50 0.54 0.80 0.46 Oracle 0.80 0.68 0.62 0.83 0.73 0.77 0.78 0.81 0.86 0.70 0.83

Qwen3-VL

-32B-Instruct

Text 0.80 0.77 0.79 0.65 0.88 0.73 0.61 0.50 0.50 0.40 0.34 Adaptive 0.83 0.77 0.87 0.76 0.78 0.77 0.60 0.50 0.57 0.50 0.46 Oracle 0.92 0.91 0.95 0.88 0.94 0.90 0.81 0.75 0.83 0.60 0.83

Qwen3-VL

-235B-Instruct

Text 0.86 0.82 0.87 0.47 0.75 0.90 0.62 0.44 0.31 0.40 0.46 Adaptive 0.89 0.86 0.90 0.75 0.88 0.90 0.77 0.62 0.55 0.60 0.57 Oracle 0.96 0.95 0.82 0.84 0.80 0.90 0.92 0.88 0.82 0.80 0.97

GPT-5∗

Text 0.90 0.86 0.92 0.79 0.86 0.90 0.58 0.38 0.50 0.50 0.54 Adaptive 0.93 0.86 0.95 0.84 0.90 0.93 0.82 0.69 0.69 0.90 0.60 Oracle 0.94 0.91 0.97 0.91 0.90 0.93 0.83 0.75 0.78 0.70 0.86

Gemini-3-Pro∗

Table 7. Detailed Accuracy (Part 2/2): Knowledge, Math, and OCR domains. * indicates that the model supports enhancement operations.

Knowledge Math OCR Model Mode Bio. Chem. Geo. Phys. Alg. Geo. Log. Stat. Chart Diag. Doc. Tab. N (Count) 57 58 10 75 64 75 13 48 171 40 55 34

Text 0.68 0.47 0.70 0.51 0.44 0.44 0.38 0.42 0.57 0.70 0.53 0.62 Adaptive 0.70 0.45 0.50 0.63 0.52 0.49 0.38 0.52 0.65 0.62 0.55 0.62 Oracle 0.82 0.62 0.60 0.61 0.69 0.69 0.46 0.54 0.72 0.88 0.73 0.79

PixelReasoner

Text 0.61 0.38 0.70 0.44 0.45 0.48 0.31 0.29 0.49 0.52 0.51 0.62 Adaptive 0.65 0.40 0.70 0.44 0.47 0.59 0.31 0.48 0.58 0.62 0.55 0.53 Oracle 0.79 0.66 1.00 0.60 0.66 0.64 0.54 0.58 0.63 0.78 0.64 0.76

Deepeyes

Text 0.65 0.40 0.70 0.47 0.53 0.52 0.31 0.40 0.56 0.55 0.55 0.76 Adaptive 0.67 0.41 0.60 0.45 0.52 0.53 0.54 0.42 0.61 0.57 0.56 0.65 Oracle 0.75 0.66 0.90 0.60 0.64 0.73 0.38 0.52 0.69 0.75 0.62 0.74

Thyme∗

Text 0.51 0.16 0.60 0.33 0.28 0.33 0.38 0.29 0.59 0.62 0.58 0.68 Adaptive 0.72 0.36 0.60 0.49 0.25 0.39 0.31 0.27 0.72 0.75 0.64 0.71 Oracle 0.56 0.74 0.70 0.57 0.48 0.52 0.46 0.65 0.74 0.85 0.87 0.97

PyVision∗

Text 0.60 0.43 0.30 0.44 0.48 0.40 0.15 0.31 0.58 0.68 0.51 0.62 Adaptive 0.65 0.48 0.50 0.49 0.50 0.57 0.31 0.40 0.61 0.55 0.51 0.53 Oracle 0.84 0.53 0.50 0.63 0.66 0.69 0.31 0.62 0.74 0.80 0.73 0.71

Deepeyes v2∗

Text 0.63 0.47 0.70 0.51 0.41 0.59 0.15 0.35 0.59 0.70 0.58 0.71 Adaptive 0.61 0.55 0.80 0.45 0.45 0.52 0.46 0.33 0.63 0.65 0.64 0.68 Oracle 0.84 0.67 0.90 0.67 0.70 0.72 0.46 0.69 0.70 0.98 0.78 0.76

AdaptVision

Text 0.75 0.62 0.40 0.65 0.55 0.47 0.31 0.56 0.65 0.75 0.36 0.76 Adaptive 0.75 0.71 0.70 0.69 0.69 0.56 0.38 0.69 0.68 0.70 0.53 0.71 Oracle 0.91 0.79 0.60 0.77 0.86 0.77 0.38 0.88 0.78 0.98 0.69 0.91

Qwen3-VL

-8B-Instruct

Text 0.84 0.78 1.00 0.68 0.67 0.68 0.31 0.79 0.81 0.82 0.62 0.82 Adaptive 0.82 0.76 0.90 0.77 0.67 0.77 0.38 0.85 0.83 0.90 0.75 0.85

Qwen3-VL

-32B-Instruct

- Oracle 0.95 0.95 0.90 0.96 0.95 0.84 0.38 0.96 0.91 1.00 0.91 0.91

Qwen3-VL

-235B-Instruct

Text 0.81 0.84 0.60 0.75 0.73 0.75 0.23 0.83 0.79 0.78 0.67 0.91 Adaptive 0.91 0.79 1.00 0.87 0.66 0.83 0.77 0.81 0.80 0.92 0.73 0.79

- Oracle 0.96 0.98 0.90 0.96 0.94 0.96 0.69 0.98 0.89 1.00 0.89 0.94

Text 0.63 0.45 0.60 0.55 0.38 0.43 0.15 0.42 0.75 0.72 0.67 0.76 Adaptive 0.88 0.83 0.80 0.88 0.67 0.64 0.23 1.00 0.87 0.90 0.78 0.94 Oracle 0.93 0.98 0.90 0.93 0.80 0.84 0.46 0.96 0.86 0.98 0.93 0.97

GPT-5∗

Text 0.86 0.86 0.90 0.77 0.72 0.76 0.46 0.88 0.86 0.95 0.82 0.94 Adaptive 0.93 0.95 1.00 0.92 0.86 0.91 0.31 0.98 0.86 0.98 0.89 1.00 Oracle 0.95 0.91 0.90 0.96 0.94 0.91 0.62 0.96 0.89 1.00 0.98 0.97

Gemini-3-Pro∗

- D. Reasoning Mode Prompt Here we provide the detailed prompts used in our experiments.

### Text-Reasoning Mode Prompts

[Multiple Choice] Question: {question} Options: {options} Please think step-by-step and give the final answer following the format: <think> reasoning process </think> <answer> the option’s letter </answer>

[Short Answer] Question: {question} Please think step-by-step and give the final answer following the format: <think> reasoning process </think> <answer> a single word or phrase </answer>

### Oracle-Visual Mode Prompts

[Zoom-in Setting] The first image is the global view, and the second image is the key region (zoomed-in) to help answer the question. Question: {question} Please think step-by-step and give the final answer following the format: <think> reasoning process </think> <answer> a single word or phrase </answer>

[Transformed Setting] The first image is the original input which might be distorted (rotated or dark), and the second image has been corrected and enhanced to show the true content. Question: {question} Please think step-by-step and give the final answer following the format: <think> reasoning process </think> <answer> a single word or phrase </answer>

[Auxiliary Line Setting] The first image is the original image, and the second image is the same image with auxiliary lines to help solve the problem. Question: {question} Please think step-by-step and give the final answer following the format: <think> reasoning process </think> <answer> a single word or phrase </answer>

### LLM Judge Prompt

I will give you a question related to the image, the ground truth answer, and the model predicted answer. Your task is to determine whether the model’s predicted answer and the ground truth answer are consistent, and output the Judgement. Note that [Model Predicted Answer] is consistent with [Ground Truth Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, ’pink’ and ’it is pink’. If they are consistent, the Judgement is 1; if they are different, the Judgement is 0. Output Format: Just output the Judgement and don’t output anything else. [Question]: {question} [Ground Truth Answer]: {ground truth} [Model Predicted Answer]: {prediction} [Judgement]:

### E. Error Analysis

In this section, we provide detailed visualizations of the failure modes discussed in the Error Analysis (Sec. 5.5 of the main paper. By examining the intermediate reasoning steps, we offer concrete examples across different categories of tool-related errors.

Visual Reasoning Failures. As noted in the main text, 42.3% of errors stem from the model’s inability to correctly manipulate or locate visual information. We present two representative scenarios:

- • Wrong Image Transformations: Figure 9 illustrates a case where the model repeatedly fails to correct the image orientation. This visual reasoning failure propagates to the OCR stage, causing the model to misread ”831K” as ”83K” and producing an incorrect prediction.
- • Incorrect Region Selection: Figure 10 demonstrates a spatial grounding failure in a dense document. The model zooms into an incorrect region (Question 235 instead of Question 238), leading to reasoning that is logically valid but based on irrelevant visual evidence.

Context Noise in Multi-step Reasoning. Figure 11 depicts the specific error type (accounting for 7.3% of cases) where visual perception is initially correct but overridden by context noise. In this example, the model successfully enhances the image and identifies the correct number of objects (”two”) in the intermediate step. However, distracted by the accumulated visual and textual context from the multi-step process, it becomes overly cautious and hallucinates a negation, resulting in a failure.

Correction via Forced Tool Invocation. Figure 12 illustrates a specific scenario (representative of the 7.0% of corrected errors) where forcing tool invocation rectifies an initial estimation failure. In this example, the model originally relies on imprecise visual intuition, incorrectly identifying ”Cerulean Blue” as the answer. However, when forced to invoke tools, it bypasses the typical spatial zooming approach and adopts a creative programmatic strategy: using Python to perform a pixel-level RGB count. By rigorously verifying consistency across multiple tolerance thresholds, the model successfully overrides its initial hallucination and derives the correct answer based on quantitative data.

Performance Degradation due to Incorrect Mode Selection. Figure 13 exemplifies the 8.3% of cases caused by incorrect mode selection, where the model unnecessarily invokes tools for tasks solvable by direct visual inspection. In this example, accurate icon counting is achievable via standard OCR or visual recognition (as seen in the Text-CoT mode). However, in the adaptive mode, the model complicates the task by adopting an unreliable engineering approach: using OpenCV edge detection to count squares. This strategy proves fragile, as the model struggles with parameter tuning—first detecting excessive noise and then over-filtering actual targets—ultimately leading to a hallucinated final count due to the confused tool outputs.

[Figure 83]

###### Figure 9. A failure case caused by incorrect image transformation (rotation) leading to OCR errors.

[Figure 84]

###### Figure 10. A failure case caused by zooming into an incorrect region (spatial misalignment).

[Figure 85]

###### Figure 11. A failure case where correct intermediate visual grounding is overridden by context noise.

[Figure 86]

###### Figure 12. Example of correcting visual estimation errors via code-based pixel analysis.

[Figure 87]

###### Figure 13. Example of performance degradation caused by unnecessary tool usage in a simple visual task.

### F. Process Evaluation Example

To better understand our evaluation protocol, we present detailed cases of process reasoning quality assessment in Figure 14 and Figure 15.

[Figure 88]

Question: Based on the playback interface, what is the current status of the audio content?

- A: The audio is currently playing
- B: The audio is paused
- C: The audio is stopped
- D: "The audio is buffering Answer: A

Qwen-3-VL-235B: Thinking: The playback interface shows a large pause

button (two vertical bars) in the center, which is the standard icon for

indicating that audio or video content is currently paused. If it were playing, it would typically show a triangle pointing to the right. The progress bar also

indicates a position at 05:41 out of 09:33, suggesting the content has been

started and is not at the beginning or stopped. Therefore, the correct status

is paused. Answer: B. The audio is paused"

[Figure 89]

Judgements:

- Key step1: Identify the playback interface on the screen. Judgment: Matched Reason: The solution explicitly identifies the playback interface.
- Key step2: Locate the central play/pause control button. Judgment: Matched Reason: It locates and describes the central play/pause control button.
- Key step3: Observe the button shows a pause icon. Judgment: Matched Reason: It observes and notes that the button shows a pause icon.
- Key step4: Apply UI convention: a pause icon indicates the media is currently playing. Judgment: Unmatched Reason: The solution incorrectly claims a pause icon means paused, contradicting the UI convention that it indicates currently playing.
- Key step5: Map this function to option A. Judgment: Unmatched Reason: The solution maps the status to option B, while the ground truth maps it to option A.

Rate: 3/5

Figure 14. Illustration of key step coverage evaluation.

Question: What's written on the Aunt left chest? Correct Answer: M·5

Qwen-3-VL-235B:

[Figure 90]

- Step1: Thinking: I need to get a closer look at the Aunt's left chest to read the text on her shirt. I will zoom in on that specific area for a clearer view. Tool1: { "name": "image_zoom_in_tool", "parameter": "{"label": "Aunt wearing a striped shirt and a straw hat", "bbox_2d": [730, 331, 1000, 854], "img_idx": 0}" }
- Step2: Thinking: Now I see the Aunt's shirt more clearly. There is a small, rectangular, brownish patch on her left chest. It appears to have some white text or a logo on it. The text looks like "M’5" or something similar, but it's still a bit blurry. I need to zoom in even closer on just that patch to read it accurately. Tool2: { "name": "image_zoom_in_tool", "parameter": "{"label": "close-up of the patch on the Aunt's left chest", "bbox_2d": [437, 350, 550, 386], "img_idx": 0}"}
- Step3: Thinking: I apologize, but it seems there was an error in the zooming process. The image I received is a close-up of a grey, textured fabric with some small specks, which does not match the description of the Aunt's striped shirt or the patch on her chest. This suggests the tool may have focused on the wrong area. I will now try to zoom in again on the correct region, specifically targeting the patch on the Aunt's left chest to get a clear view of the text. Tool3: { “name”: “image_zoom_in_tool”, “parameter”: “{”label“: ”close-up of the patch on the Aunt‘s left chest“, ”bbox_2d“: [437, 350, 550, 386], ”img_idx“: 0}"}

[Figure 91]

[Figure 92]

Step4: Thinking: I see that the zoom tool is consistently returning an image of a grey,

textured fabric, which is not the Aunt's shirt. This indicates a persistent error in the tool's

targeting. Since I cannot rely on the tool to get a clear view, I will have to make my best assessment from the original, wider image. In the original photo, the patch on her left chest

is visible. It is a small, rectangular, brownish patch with white lettering. The lettering appears

to be "M’5", with the 'M' being a capital letter and the 'S' following it. The 'S' might have a small apostrophe or design element before it, but "M’5" is the most legible interpretation.

Answer: M5

[Figure 93]

Judgements:

Correct Tools: [Tool1] Brief Analysis: Tool1 correctly zooms in on the woman on the right, matching the intent to focus on the aunt’s area. Tools 2 and 3 aimed to zoom the patch on

her left chest but instead targeted the man’s shirt area, returning a grey fabric region;

therefore, they did not satisfy the intended focus.

Rate: 1/3

Figure 15. Illustration of tool effectiveness evaluation.

### G. Process Evaluation Prompt

To ensure a reproducible and standardized assessment, we leverage LLM-based judges with specialized prompts for process auditing. Specifically, we employ the tool invocation effectiveness prompt (detailed in Figure 16) is used to audit the functional correctness and intent alignment of each tool call within the adaptive process. Subsequently, the key step coverage Prompt (detailed in Figure 17) to verify the logical completeness of the model’s reasoning trajectory against annotated ground truth.

### Tool Invocation Effectiveness Prompt

The above content contains the model’s multi-step reasoning process. You are a top-tier visual reasoning audit expert. You possess strong logical analysis skills and can accurately evaluate whether a model’s tool usage aligns with its stated reasoning intent. All Available Tools: {tools} Each step in the reasoning process includes:

- • [Step] A textual reasoning step.
- • [Tool] A tool invocation (All Available Tools above).
- • [Tool Execution Output] The execution output (image, text, or highlighted region).

Evaluation Criteria:

- 1. Tool Invocation Correctness: Check if the invocation is valid, properly formatted, and consistent with the tool’s definition and the corresponding Step ID.
- 2. Tool Execution Output Validation: Check if the output satisfies the intent of the corresponding Step ID and Tool ID. The output only needs to fulfill the specific step’s purpose, not the overall problem. Evaluate based on the global image.

Attention:

- • Only steps satisfying ALL criteria are listed as correct.
- • Do not analyze for errors unless the output explicitly indicates a failure.
- • Ignore image resolution/size info; assume the tool operates on the original source image.

Output Format (JSON-like): Correct Tools: [Tool1, Tool2, Tool3...] Brief Analysis: [Briefly explain why these tool invocations are correct and why others are incorrect]

Figure 16. Prompt used for evaluating tool invocation effectiveness.

### Key Step Coverage Prompt

You are an expert system for verifying solutions to image-based problems. Your task is to: 1. Segment the provided solution into logical reasoning steps. 2. Match each ground truth middle step with the solution steps. INPUT FORMAT: 1. Problem: The original question/task. 2. Solution: A continuous paragraph containing the model’s reasoning. 3. Ground Truth: Essential steps required for a correct answer. TASK:

- Step 1: Segment the Solution Divide the solution paragraph into distinct logical reasoning steps. Each step should represent a coherent reasoning unit.
- Step 2: Match Ground Truth Steps For each ground truth step, determine if it is matched in any of the solution steps based on the following:

- • Content Match: Must match specific values and details.
- • Tool Integrity: If tool execution has errors, the step is "Unmatched".
- • Implicit Logic: Reasonable logical skips (like identifying obvious objects or

using background knowledge) are permissible. OUTPUT FORMAT (JSON): {

"solution_steps": ["Step 1...", "Step 2..."], "judgments": [

{

"step_index": <integer>, "judgment": "Matched" | "Unmatched", "reason": "Brief explanation."

} ]

} ADDITIONAL RULES: 1. Only output the JSON object with no extra text. 2. Judge each ground truth step in order without omission. DATA:

- • [Problem] {question}
- • [Answer] {answer}
- • [Solution] {solution}
- • [Ground Truth Information] {gt annotation}

Figure 17. Prompt used for key step coverage evaluation.

