# arXiv:2604.24198v2[cs.CL]20Jun2026

## Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis

Zhisong Qiu

Zhejiang University Ant Group Zhejiang University - Ant Group Joint Laboratory of Knowledge Graph Hangzhou, China qiuzhisong@zju.edu.cn

Lun Du

Ant Group Hangzhou, China dulun.dl@antgroup.com

Shuofei Qiao Kewei Xu

Zhejiang University Hangzhou, China kewe1x@zju.edu.cn

Yuqi Zhu

Zhejiang University Hangzhou, China zhuyuqi@zju.edu.cn

Ningyu Zhang

School of Software Technology, Zhejiang University Hangzhou, China zhangningyu@zju.edu.cn

Huajun Chen∗

Zhejiang Key Laboratory of Intelligent Manufacturing for Functional Chemicals, ZJU-Hangzhou Global Scientific and Technological Innovation Center Zhejiang University Hangzhou, China huajunsir@zju.edu.cn

### Abstract

Process Reward Models (PRMs) have achieved remarkable success in augmenting the reasoning capabilities of Large Language Models (LLMs) within static domains such as mathematics. However, their potential in dynamic data analysis tasks remains underexplored. In this work, we first present a empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect silent errors, logical flaws that yield incorrect results without triggering interpreter exceptions, and erroneously penalize exploratory actions, mistaking necessary trialand-error exploration for grounding failures. To bridge this gap, we introduce DataPRM, a novel environment-aware generative process reward model that (1) can serve as an active verifier, autonomously interacting with the environment to probe intermediate execution states and uncover silent errors, and (2) employs a reflection-aware ternary reward strategy that distinguishes between correctable grounding errors and irrecoverable mistakes. We design a scalable pipeline to construct over 7K high-quality training instances for DataPRM via diversity-driven trajectory generation and knowledgeaugmented step-level annotation. Experimental results demonstrate that DataPRM improves downstream policy LLMs by 7.21% on ScienceAgentBench and 11.28% on DABStep using Best-of-N inference. Notably, with only 4B parameters, DataPRM outperforms strong

∗Corresponding author

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference acronym ’XX, Woodstock, NY © 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/https://doi.org/10.1145/3770855.3819049

baselines, and exhibits robust generalizability across diverse TestTime Scaling strategies. Furthermore, integrating DataPRM into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving 78.73% on DABench and 64.84% on TableBench, validating the effectiveness of process reward supervision1.

### CCS Concepts

• Computing methodologies → Natural language processing.

### Keywords

Process Reward Models, Data Analysis Agent, Large Language Models

### 1 Introduction

Automated data science, aiming to autonomously generate novel scientific knowledge or hypotheses from complex datasets, stands as a core objective in modern scientific discovery [58, 75]. Central to this pursuit is automated data analysis, the key step to derive evidence-based insights and supportive scientific conclusions to help human decision-making. As Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities [8, 44] on a wide spectrum of tasks such as mathematics [6, 18, 37, 47, 50] and science [5, 9, 29, 34, 48], researchers are now increasingly locating them as the backbone of data analysis agents to automate the scientific discovery pipeline [1, 19, 39, 45, 54, 62, 66, 73, 76]. However, prevailing approaches focus only on outcome supervision, overlooking the multi-step rigor of data analysis. In scientific research, where the process must be error-free, this outcome-centric paradigm risks propagating hallucinated logic, yielding seemingly plausible but invalid discoveries.

1Code: https://github.com/zjunlp/DataMind.

[Figure 1]

Data Analysis Agent

[Figure 2]

Bioinformatics

Output

Step t+1

###### Step 1

###### Step t

[Figure 3]

For cell samples in the heart cell atlas dataset, ﬁlter out lowly expressed genes and plot UMAP results using the top 30 PCA components … Save the UMAP visualization as 'pred_results/ hca_cell_type_pca.png'

<Analyze>I will plot the UMAP with cell …</Analyze> <Code>visualization</Code> <Execute>FileNotFoundError </Execute>

<Analyze>I need understand data structure …</Analyze> <Code>data view</Code> <Execute>Data Details … </Execute>

<Analyze>Let me modify the code …</Analyze> <Code>visualization</Code> <Execute>Successfully </Execute>

###### Answer

###### ……

Process Reward Model

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

The image meets requirements. Score: 1.0

The visualization code is accurate. Score: 1.0

[Figure 8]

Analysis is correct despite minor errors. Score: 0.5

Data info is accurate. The step is correct. Score: 1.0

Env

……

#### Figure 1: The Collaborative Pipeline Between Data Analysis Agent and Process Reward Model (PRM). The agent addresses data analysis tasks while the PRM supervises the agent’s procedural steps.

Conversely, Process Reward Models (PRMs) have exhibited remarkable success in domains such as mathematical reasoning [22, 35, 57, 79, 80, 88] and code generation [24, 68, 72]. By providing step-level supervision and fine-grained verification during both training and inference time, PRMs can significantly boost the models’ reasoning reliability and performance boundary [17, 30, 53, 82]. Despite their proven efficacy, the application of step-level supervision in the domain of data analysis remains largely unexplored. This leads to a key question: How can we effectively implement step-level supervision for automated data analysis tasks?

11.28% on DABStep. Notably, our model outperforms powerful baselines, such as self-rewarding strategies using Qwen3-235B-A22BInstruct, while achieving 58× parameter efficiency. In RL settings, models trained with our process supervision achieve 78.73% on DABench and 64.84% on TableBench, surpassing methods relying solely on outcome supervision. Our extensive analysis offers two valuable insights to the community: (1) Environment interaction is critical for process supervision in data analysis; (2) In scenarios with vast exploration spaces, the diversity of supervision steps may outweigh the strict specialization of annotations. DataPRM not only improves LLM-based data analysis reliability but also provides a scalable framework for fine-grained process supervision in scientific discovery.

To bridge this gap, we first analyze the cross-domain applicability of state-of-the-art general PRMs to data-analytic tasks. Our preliminary analysis reveals that existing PRMs fail to reliably verify two specific categories of errors inherent to this domain: (1) Silent Errors: General PRMs struggle to identify logical flaws that yield incorrect results without triggering interpreter exceptions. (2) Grounding Errors: they often mistake necessary trial-and-error exploration for irrecoverable failures, leading to early penalization. These findings indicate that off-the-shelf PRMs are insufficient for reliable process supervision in data analysis.

In summary, the main contributions of this work are as follows:

- • We propose DataPRM, a novel process reward model that utilizes environment interaction and ternary rewards to resolve the grounding and silent error challenges in automated data analysis.
- • We introduce a robust pipeline for generating fine-grained process supervision data, producing a dataset of over 7K annotated instances through diversity-driven trajectory generation and knowledge-augmented step-level annotation.
- • We empirically validate DataPRM in both TTS and RL settings, achieving significant performance gains on benchmarks such as ScienceAgentBench, while demonstrating 58× parameter efficiency over comparable large-scale baselines.

Driven by these insights, we introduce DataPRM, a novel Process Reward Model tailored specifically for data analysis agents. Unlike previous PRMs designed for static reasoning tasks, DataPRM can interact dynamically with the environment to validate steps based on real-world data contexts, thereby avoiding deception by mere code execution success. Furthermore, DataPRM employs a ternary reward strategy to distinguish between incorrect steps, correct steps, and neutral exploratory steps, preventing the suppression of necessary exploration. To construct DataPRM, we design a scalable data generation pipeline utilizing diversity-driven trajectory generation and knowledge-augmented expert annotation, yielding over 7K high-quality supervision instances. We apply DataPRM in both Test-Time Scaling (TTS) and Reinforcement Learning (RL) frameworks to further boost the performance boundary of current data analysis agents.

### 2 Preliminary 2.1 Data-Analytic Agents

We formalize the data analysis process as a Partially Observable Markov Decision Process, denoted by the tuple (U, S, A, T, O). Here, the state space S characterizes the environment, which typically comprises a code interpreter I and a set of files F. The process commences with a specific task 𝑢 ∈ U associated with an initial environmental state 𝑠0 ∈ S. Given the current state 𝑠, the agent performs an action 𝑎 ∈ A through code generation. The code interpreter I also functions as the state transition mechanism, 𝑇 (𝑠′|𝑠,𝑎) ∈ T, determining the subsequent state 𝑠′. Under the assumption of partial observability, the agent perceives the current state solely through an observation 𝑜 ∈ O from the interpreter.

We evaluate DataPRM across multiple data analysis benchmarks. In TTS settings, incorporating a 4B-parameter DataPRM improves downstream policy models by 7.21% on ScienceAgentBench and

#### Table 1: Representative cases for both error types. Key errors are highlighted in red alongside their descriptions and the PRM’s misjudged reward, illustrating the blind spots of current PRMs.

###### Category Error Explanation Wrong Agent Step Error Description & PRM output

Step 5: <Analyze>Now, we will create a visualization of proposed roads on the deforestation: ... 5.5 km risky buffer

The data analysis agent claims to have drawn the 5.5 km risky buffer (in light coral/pink) and saved the visualization image,

The code interpreter executes successfully. But the actual result obtained is incorrect.

(in light coral/pink) ... We’ll save the plot to ‘pred_results/predictedRiskyArea.png’. </Analyze> <Code>visualization code</Code> <Execute> Successfully saved </Execute>

Silent Error

but the buffer is not actually successfully drawn in the visualization .

PRM → {correct} ✗ Fail to detect the silent error

Step 1: <Analyze>The goal is to compare R2 results between single-task and multitask models across different datasets from the TDC ADMET benchmark. I’ll start by loading the data and examining its structure.</Analyze> <Code>data view code</Code> <Execute> KeyError: ‘dataset’ </Execute>

The data analysis agent encounters a KeyError error during the initial attempt to load the file. The actual key is ’Dataset’ . This is part of the agent’s environment exploration.

The model’s prior knowledge conflicts with actual data in the environment.

Grounding Error

PRM → {incorrect} ✗ Detect the error but not realize that the agent is in environment exploration.

48

50

[Figure 9]

Majority Voting

N=4

N=8

| |
|---|

GenPRM-32B

45

Qwen2.5-Math-PRM-72B

44

42.67

ThinkPRM-14B

Accuracy(%)

Accuracy(%)

41.33

41.33

40

40

Over-Penalization!

38.0

37.33

35

35.33

36

30

20 22 23 24

32

Number of Solutions (N)

CoT One Turn Code Multi Turn Code

(a) General PRMs’ Performance.

(b) Score Distribution for Grounding Errors.

(c) Ablation on Environment Interaction.

#### Figure 2: (a): General PRMs’ Best-of-N performance on a subset of DABStep. (b): General PRMs’ scores on steps with grounding errors despite correct final answers. (c): Ablation study on environment interaction based on prompted LLMs.

Then the historical interaction trajectory at time 𝑡 can be represented asℎ𝑡 = (𝑢,𝑎0,𝑜0,𝑎1,𝑜1, . . .,𝑎𝑡−1,𝑜𝑡−1). In scenarios adopting the ReAct [65] framework, where explicit reasoning 𝑧 guides action generation, the trajectory can be finally formulated as:

where A(·) represents an aggregation function, typically Sum or Mean [30]. By providing either step-level reward 𝑟𝑡 or trajectorylevel reward 𝑟𝑡𝑟𝑎𝑗, the verifier can not only enhance the policy model’s reasoning performance through search algorithms (e.g. Best-of-N or Beam Search), but also provide more fine-grained reward signals for reinforcement learning.

ℎ𝑡 = (𝑢,𝑧0,𝑎0,𝑜0,𝑧1,𝑎1,𝑜1, . . .,𝑧𝑡−1,𝑎𝑡−1,𝑜𝑡−1). (1)

In our problem setup, the components 𝑧𝑡,𝑎𝑡,𝑜𝑡 at time step 𝑡 are regarded as a unified step 𝜏𝑡 of data analytic agents.

### 3 General PRMs on Data Analysis Tasks

### 2.2 Reward Modeling for Data Analysis

We begin by assessing the efficacy of existing general-domain PRMs in supervising data analysis agents. Specifically, we conduct a pilot study to investigate two Research Questions (RQs):

As illustrated in Fig.1, given a data-analytic agent’s historical interaction trajectory ℎ𝑡 and the current step 𝜏𝑡, a standard Process Reward Model (PRM) parameterized by𝜃, utilizes a scoring function 𝑅𝜃 (·) to assign a step-level reward 𝑟𝑡. The overall trajectory-level reward𝑟𝑡𝑟𝑎𝑗 is then derived by aggregating these step-level rewards:

RQ1 - Can general-domain PRMs effectively distinguish valid reasoning steps in data analysis tasks compared to simple ensemble baselines (e.g., Majority Voting)?

𝑟𝑡 ∼ 𝑅𝜃 (·|ℎ𝑡,𝜏𝑡), with 𝑟𝑡𝑟𝑎𝑗 = A(𝑟1,𝑟2, . . .,𝑟𝑇) (2)

- (a) Step-Level Data Construction
- (b) DataPRM Process-Supervised Pipeline

[Figure 10]

Expert Annotation

Yes Step-Level Annotation

SFT

Policy Trajectories

Error Knowledge

Inconsistency?

DataPRM

Few shot

thought action observation

[Figure 11]

###### Step k

###### Step 1

import pandas as pd df = pd.read_csv(…)

[Figure 12]

Think

Think

Policy Model

[Figure 13]

[Figure 14]

Input

###### Reasoning Progress

Types

Step 1

Action

Action

… query_image(‘Describe

the details’, ‘city.png’)

Step 2

[Figure 15]

Observation

Observation

……

query_document(‘What’s the cost rule?’, ‘rule.md’)

###### Final Response

The step successfully completes its task as required

1.0

###### Current Step

[Figure 16]

Error

Grounding

The step is correct, but contains a minor error that can be easily ﬁxed

Score

[Figure 17]

0.5

[Figure 18]

Env

Previous Feedback

…

[Figure 19]

[Figure 20]

This step has a fatal error: it falsely claims the task was completed as required.

Rationale

Error

0.0

Silent

#### Figure 3: Overview of DataPRM Framework. (a): A diversity-driven trajectory generation strategy followed by knowledgeaugmented step-level annotation. (b): DataPRM employs multi-turn interaction, tool-augmented capability and reflection-aware reward strategy for scoring.

RQ2 - What are the specific failure modes of current PRMs when supervising data analysis agents, particularly regarding environment interaction?

Takeaway 1: Existing PRMs fail to distinguish between fatal errors and recoverable exploratory steps, often penalizing the latter harshly and impeding environment adaptation.

We utilize Qwen3-235B-A22B-Instruct [63] as the policy model and evaluate on a subset of the DABStep benchmark.

Inability to Detect Silent Errors. Conversely, “Silent Errors” occur when code executes without exceptions but produces incorrect results due to logical flaws. Since current PRMs rely primarily on static reasoning (reading the code text), they cannot verify the semantic correctness of the execution result. As shown in Fig.2c, we employ in-context learning to have Qwen3-30B-A3B-Instruct [63] function as the PRM and evaluate it on the same subset of DABStep. We observe that when the PRM is granted the ability to actively interact with the environment (via one-turn code or multi-turn code), it can more accurately select the correct steps. Moreover, the performance under the multi-turn setting surpasses that of the one-turn setting. This is likely because, in the multi-turn setting, the PRM can attempt more interaction to verify the correctness.

Performance Bottleneck of General PRMs. To address RQ1, we benchmark three state-of-the-art PRMs (Qwen2.5-Math-PRM-72B [79], GenPRM [80], and ThinkPRM [22]) against a Majority Voting baseline. As shown in Fig.2a, while PRM-guided search (Best-of-N) improves over single-path generation (e.g., ThinkPRM improves performance from 32.67% to 40.00% at N=16), it surprisingly fails to surpass the Majority Voting baseline. This suggests that generaldomain PRMs lack the specific discriminative capability required for data analysis, rendering them less cost-effective than simple sampling strategies.

Error Analysis: Grounding vs. Silent Errors. To answer RQ2, we perform a fine-grainederror analysis and identify two critical failure modes (Tab.1) that baffle current PRMs:

Takeaway 2: PRMs with environment interaction capability can better verify the correctness of data analysis steps.

Misjudgment of Exploratory Failures (Grounding Errors). Data analysis agents often encounter “Grounding Errors” —syntax or schema errors arising from a lack of prior knowledge about the data file (e.g., guessing a wrong column name). These are often recoverable and necessary steps for the agent to learn the environment through feedback. We collect steps that contain grounding errors but yield correct final answers, and have the existing PRMs score them. As Fig.2b shows, existing PRMs often treat these steps as fatal errors, assigning them low scores. This penalizes exploration and causes the search algorithm to prune trajectories that would have led to a correct solution after self-correction.

Motivation for DataPRM. Our analysis reveals that the core limitation of current methods is the lack of an environment-aware verifier. We need a PRM that can (1) forgive recoverable grounding errors to encourage exploration, and (2) actively interact with the data to catch silent errors. Motivated by these observations, we introduce a novel process reward model specifically tailored to enhance data-analytic agents.

- 4 Methodology

- 4.1 Environment-Aware Verifier Architecture

We introduce DataPRM, an environment-aware generative PRM. It adopts the ReAct paradigm and can interact with the environment.

- 4.1.1 Generative ReAct Paradigm for Verification. We argue that effective verification in data analysis requires as many contextual interaction capabilities as the solution generation itself. Consequently, our PRM is modeled using the same ReAct paradigm as the data analysis agent.

Given a trajectory ℎ𝑡 of policy model and its current step 𝜏𝑡 at time 𝑡, the input context for DataPRM is:

ℎ𝑡,𝑝𝑟𝑚0 = ℎ𝑡 ⊕ 𝜏𝑡 = ℎ𝑡 ⊕ (𝑧𝑡,𝑎𝑡,𝑜𝑡) (3) where ⊕ denotes sequence concatenation. This ensures the reward model judges the current step 𝑎𝑡 in light of the entire problemsolving trajectory ℎ𝑡 and its immediate outcome 𝑜𝑡. Then DataPRM engages in a multi-step reasoning and verification process. Let 𝑘 denote the internal time step. At each step𝑘, the DataPRM generates a verification tuple 𝜅𝑡,𝑘 = (𝑧ˆ𝑘,𝑎ˆ𝑘,𝑜ˆ𝑘). Then the internal context updates as follows:

ℎ𝑡,𝑘𝑝𝑟𝑚+1 = ℎ𝑡,𝑘𝑝𝑟𝑚 ⊕ 𝜅𝑡,𝑘 (4) This internal ReAct loop continues until the DataPRM decides to terminate at step 𝐾. The final action 𝑎ˆ𝐾 is no longer in code form, but rather a verification result composed of a score and a rationale. Let 𝜌𝜙 denote the DataPRM, the final output is:

(𝑧ˆ𝐾,𝑟𝑡,𝑐𝑡) ∼ 𝜌𝜙(·|ℎ𝑡,𝐾𝑝𝑟𝑚) (5) Here,𝑟𝑡 is the scalar quality score for the step𝜏𝑡 of the policy model, and 𝑐𝑡 is the explanatory rationale derived from the verification trajectory. And the feedback tuple (𝑟𝑡,𝑐𝑡) generated by the DataPRM is not discarded but is explicitly appended to the context for the next time step 𝑡 + 1 verification. Given the historical verification result 𝑓𝑡 = (𝑟0,𝑐0,𝑟1,𝑐1, . . .,𝑟𝑡−1,𝑐𝑡−1) of the verifier, we redefine the input of DataPRM in Formula 3 as follows:

ℎ𝑡,𝑝𝑟𝑚0 = ℎ𝑡 ⊕ 𝑓𝑡 ⊕ 𝜏𝑡 (6) This form ensures that DataPRM can access verification information from previous steps, thereby guaranteeing the consistency and continuity of the evaluation. Additionally, we provide a theoretical perspective in Appx.A.

- 4.1.2 Tool-Augmented Capability Integration. When interacting with a data analysis environment, PRMs may require multiple capabilities, such as multimodal understanding (reading images) or long-context comprehension (reading manual documents). Recent studies indicate that LLM agents can autonomously leverage tools to engage with external environments and progressively refine their reasoning ability [15, 43, 70, 87]. Inspired by their work, we decouple the verifier’s capabilities into intrinsic reasoning (acquired via training) and extrinsic perception (acquired via tools). We equip DataPRM with two tools, namely query_document and query_image. DataPRM can query related questions about documents or images through function calls in the code, and the tools will invoke the corresponding expert models to provide answers. By bridging internal code generation with external tool usage, DataPRM achieves

comprehensive verification coverage across data files, manual documents, and images.

- 4.1.3 Reflection-Aware Reward Strategy. As shown in §3, existing PRMs cannot distinguish between grounding errors and other types when assigning scores. To address this, we expand the step-level

reward space 𝑟𝑡 ∈ {0, 1} to a ternary set R = {0, 0.5, 1} to capture the nuance of agentic behaviors:

- • Strictly Correct (𝑟𝑡 = 1.0): The step is logically sound and it advances the solution directly.
- • Irrecoverable Error (𝑟𝑡 = 0.0): The step contains fundamental logic flaws or hallucinations that steer the trajectory to a dead end from which recovery is impossible.
- • Correctable Error (𝑟𝑡 = 0.5): The step contains a minor error (e.g. syntax error, incorrect file path) but effectively triggers an environment feedback loop that allows for potential correction.

4.2 Step-Level Data Construction

Existing public data analysis datasets rarely provide both source files andfine-grainedstep annotations, making off-the-shelf processsupervised training difficult. We therefore introduce a data generation pipeline with diversity-driven trajectory generation and knowledge-augmented step-level annotation.

- 4.2.1 Diversity-Driven Trajectory Generation. We primarily adapt the AutoSDT [25] methodology to crawl GitHub for files related to scientific data analysis. To increase the volume of usable data, human experts revise and extend a subset of these files. For query generation, we use DeepSeek-V3.2 [12] as an expert model to synthesize reasoning-focused queries, while directly adopting validated AutoSDT queries for visualization tasks. For each validated query 𝑥 from the collection phase, we employ Qwen3-235B-A22B-Instruct

[63] as the policy model 𝜋𝜃 to perform parallel sampling. We generate 𝐾 = 4 distinct trajectories and use a judge model M based on DeepSeek-V3.2 [12] to determine whether their final answers are mutually inconsistent. To maximize the information gain during PRM training, we retain the trajectory set {𝑦𝑖}𝑖𝐾=1 only when the final answers are not all identical. This strategy focuses the data on informative boundary cases.

4.2.2 Knowledge-Augmented Step-Level Annotation. We convert collectedtrajectoriesintodiscretesteps. Qwen3-235B-A22B-Instruct [63] conducts an initial pass for step annotation and error attribution. To systematically categorize failures, we apply the AutoManual [7] framework to merge similar error categories. Human experts manually verify the rationales of these merged categories and inject them as structured few-shot examples into the annotation prompt. DeepSeek-V3.2 then assigns final step-level rewards using the ternary reward strategy defined in Section 4.1.3, constructing the final dataset for process supervision. For quality control, we filter non-analytical errors, such as timeouts and broken files, and verify LLM annotations against human experts before scaling up the pipeline. Based on 100 manual spot checks, the model achieves 86.0% raw accuracy and a quadratic weighted Cohen’s 𝜅 of 0.83, confirming high reliability.

#### Table 2: Main results on ScienceAgentBench and DABStep. We compare DataPRM against various step verifiers using best-of-𝑁 sampling (𝑁 ∈ {4, 8, 16}) with Qwen3-235B-A22B-Instruct-2507 as the base policy. Best results are in bold. DataPRM achieves state-of-the-art TTS performance using substantially fewer parameters.

ScienceAgentBench DABStep

Verifier (Best-of-N) Params

SR Easy Hard Avg. 4 8 16 4 8 16 4 8 16 4 8 16

Majority Vote - 24.36 24.36 23.08 75.00 76.39 76.39 26.98 29.63 30.69 34.66 37.11 38.00 LLM-as-a-judge - 24.36 24.36 24.36 75.00 76.39 75.00 25.13 27.51 29.63 33.11 35.33 36.89 Self-Rewarding - 24.36 24.36 24.36 75.00 76.39 76.39 28.04 30.16 32.80 35.55 37.56 39.77 Math-Shepherd-PRM-7B 7B 19.23 21.79 20.51 75.00 75.00 75.00 23.28 23.28 19.31 31.56 31.56 28.22 Qwen2.5-Math-PRM-7B 7B 19.23 20.51 19.23 75.00 72.22 73.61 20.90 18.25 14.55 29.56 26.89 24.00 ReasonFlux-PRM-7B 7B 19.23 21.79 19.23 73.61 72.22 75.00 20.63 17.99 13.76 29.11 26.67 23.56 ThinkPRM 14B 19.23 21.79 17.95 75.00 75.00 72.22 25.13 24.34 26.72 33.11 32.45 34.00 GenPRM 32B 21.79 20.51 20.51 75.00 73.61 73.61 24.60 25.40 26.72 32.66 33.11 34.22 Qwen2.5-Math-PRM-72B 72B 23.08 23.08 20.51 73.61 76.39 75.00 21.96 22.75 20.37 30.22 31.33 29.11

DataPRM 4B 24.36 25.64 25.64 75.00 76.39 77.78 29.89 32.80 33.86 37.11 39.77 40.89

### 4.3 End-to-End RL Training with PRM

For end-to-end RL training, we employ the Group Relative Policy Optimization (GRPO) [51] algorithm with several effective strategies such as clip-higher and token-level loss to ensure stable opti-

mization [67]. Define 𝜚𝑖,𝑡 (𝜃) = 𝜋𝜋𝜃 (𝑜𝑡 |𝑞,𝑜<𝑡)

𝜃old(𝑜𝑡 |𝑞,𝑜<𝑡) , the loss is:

##### J(𝜃) = E(𝑞,𝑎)∼D,{𝑜𝑖}𝐺

𝑖=1∼𝜋𝜃old (·|𝑞)

##### ∑︁𝐺

##### ∑︁|𝑜𝑖|

1

min 𝜚𝑖,𝑡 (𝜃)𝐴ˆ𝑖,𝑡, clip(𝜚𝑖,𝑡 (𝜃), 1 − 𝜖𝑙, 1 + 𝜖ℎ)𝐴ˆ𝑖,𝑡

𝐺 𝑖=1 |𝑜𝑖|

𝑖=1

𝑡=1

(7) The total reward 𝑟total is formulated as a weighted combination

of the outcome reward 𝑟outcome and the PRM scores 𝑟prm:

∑︁𝑇

1 𝑇

𝑟total = (1 − 𝛽) · 𝑟outcome + 𝛽 · (

𝑟prm(𝜏𝑡)) (8)

𝑡=1

where 𝛽 controls the trade-off between outcome correctness and process validity and 𝜏𝑡 is the step of agent at time 𝑡. With a group size of 𝐺, we calculate the group-normalized advantage 𝐴ˆ𝑖,𝑡 for the 𝑖-th output as:

𝑟total,𝑖 − mean({𝑟total,𝑗}𝐺𝑗=1) std({𝑟total,𝑗}𝐺𝑗=1)

𝐴ˆ𝑖,𝑡 =

(9)

Moreover, we observe that discrepancies may arise between the ground truth outcome and the PRM’s final step estimation. To address this, we enforce a consistency check:

𝑟outcome if 𝑟prm(𝜏𝑇) ≠ 𝑟outcome 𝑟prm(𝜏𝑇) otherwise

(10)

𝑟prm(𝜏𝑇) ←

This alignment ensures that the model does not learn from conflicting signals at the termination of a trajectory.

DataPRM (Ours)

GenPRM-32B

ThinkPRM-14B

Self-Rewarding

Qwen2.5-Math-PRM-72B

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

40.0

Accuracy(%)

37.5

35.0

32.5

30.0

20 21 22 23 Number of Beams (N)

20 21 22 23 Number of Beams (N)

(a) Beam Search

(b) DVTS

Figure 4: Performance of DataPRM evaluated under two extended TTS strategies, namely: (a) Beam Search and (b) Diverse Verifier Tree Search (DVTS).

### 5 Experiments 5.1 Experiment Settings

We first empirically evaluate DataPRM on the Test-Time Scaling (TTS) experiment (Section 5.2), then conduct an in-depth analysis (Section 5.3), and finally, we apply DataPRM to Reinforcement Learning (RL) and perform experiments (Section 5.4).

5.1.1 Datasets and Metrics. For TTS, we evaluate DataPRM on two datasets: ScienceAgentBench [10] and DABStep [14]. For ScienceAgentBench, we filter out ML/DL tasks and retain 78 tasks related to data analysis to ensure a focus on automating data analysis tasks and avoid introducing confounding factors associated with model training processes. And we utilize its provided evaluation procedure to report the Success Rate (SR), in which visualization metrics are assessed by Qwen3-VL-235B-A22B-Instruct [3] as a judge. For DABStep, we utilize accuracy as the final evaluation metric. For RL, we evaluate our model on two other datasets:

- Table 3: Ablation study of different components. "Env", "Multi", and "Refl" denote the Code Environment, Multi-turn Interaction, and Reflection-aware/Ternary Reward Strategy, respectively. Best results are in bold.

Components Easy Hard Avg.

Variant

Env Multi Refl 4 8 16 4 8 16 4 8 16

CoT ✗ ✗ ✗ 75.00 75.00 75.00 25.93 29.37 32.01 33.78 36.67 38.89 Single-turn Code w/ Env ✓ ✗ ✗ 76.39 76.39 76.39 28.57 30.95 32.80 36.22 38.22 39.77 Multi-turn Code w/o Env ✗ ✓ ✗ 73.61 75.00 76.39 26.46 29.89 31.75 34.00 37.11 38.89 Multi-turn Code w/ Env ✓ ✓ ✗ 75.00 76.39 76.39 29.37 30.69 32.80 36.67 38.00 39.77 DataPRM ✓ ✓ ✓ 75.00 76.39 77.78 29.89 32.80 33.86 37.11 39.77 40.89

DABench [20] and TableBench [60]. We use judge model powered by Qwen3-30B-A3B-Instruct [63] to assess the accuracy of the answers, reporting both pass@1 and pass@3 scores.

- 5.1.2 Models and Baselines. For TTS, we compare DataPRM with various step-level verification baselines, including advanced PRMs, majority voting [30], LLM-as-a-judge [83] using DeepSeek-V3.2 [12], andself-rewarding[69,74]utilizingQwen3-235B-A22B-Instruct [63]. For PRM approaches, we include both discriminative (QwenPRM series [79], Math-Shepherd-PRM-7B [57], and ReasonFluxPRM-7B [88]) and generative (ThinkPRM [22] and GenPRM [80]). For the policy reasoning models, we evaluate the proposed method on Qwen3-235B-A22B-Instruct [63]. For RL, we use Qwen2.5-Coder7B-Instruct as the base model and compare with the SFT model and the model trained with outcome rewards.

- 5.1.3 Implementation Details. We use ms-swift [81] for DataPRM SFT training. For SFT, our learning rate is 1𝑒 − 5 with a warmup ratio of 0.05. We train 3 epochs and use liger kernel. Our global batch size is set to 32. For DataPRM inference, temperature is 0.7, the top-p is 0.9 and the top-k is 20. For applying DataPRM to RL, we use verl [52]. We use a learning rate of 1𝑒 − 6. The batch size is 32 with a mini batch size of 2. The balancing coefficient 𝛽 is set to 0.5. The rollout temperature is 0.7, the top-p is 1.0, and the group size 𝐺 is 4. We use AgentLoop and RewardLoop to carry out asynchronous rollout and rewarding, thereby accelerating the training process. All the experiments are conducted on 8 × H20 GPUs.

- 5.2 Main Results

- 5.2.1 DataPRM Surpasses Larger Baselines with Effective Scaling in Best-of-N. Tab.2 presents the performance of DataPRM and other baselines in the Best-of-N setting. Although parameterized at only 4B, DataPRM consistently achieves superior results compared to robust baselines like GenPRM-32B and Qwen2.5Math-PRM-72B. Furthermore, it surpasses both the DeepSeek-V3.2 LLM-as-a-judge framework and the Qwen3-235B-A22B-Instruct self-rewarding baseline. Moreover, as 𝑁 and the number of responses in the candidate pool increase, existing PRMs may discard originally correct responses and select incorrect ones. For example, as 𝑁 expands from 8 to 16, the performance of Qwen2.5-Math-PRM72B drops from 31.33% to 29.11%. This indicates that existing PRMs have not truly acquired the ability to distinguish between valid reasoning and hallucinations in data analysis tasks. In contrast, DataPRM achieves effective scaling, delivering consistent performance improvements as 𝑁 increases. This suggests that it can

#### Table 4: Ablation study on filtering strategies. Best results are marked in bold.

Filter Strategy 4 8 16

Unfiltered 37.11 39.77 40.89 Meta-Critic 36.67 36.45 40.00 Outcome-Consistency 36.22 38.22 39.77 Process-Consistency 38.00 38.22 39.34

#### Table 5: Inference cost analysis.

Verifier Total Tokens Turns Time(s) Tool Calls GenPRM 7061.25 1.00 14.86 Self-Rewarding 25282.51 3.32 194.95 0.63 DataPRM 21455.78 2.57 24.66 0.87 DataPRM (parallel) 21455.78 2.57 3.30 0.87

discern high-quality data analysis trajectories, thereby providing stronger reward supervision.

- 5.2.2 DataPRM Generalizes Across Search Strategies and Resists Reward Hacking. Beyond best-of-N search, we assess DataPRM under two extended TTS strategies: Beam Search and Diverse Verifier Tree Search (DVTS). These results are then benchmarked against the self-rewarding method and the most competitive PRM baselines. As shown in Fig.4, DataPRM consistently outperforms all baselines across both search strategies and all computation budgets. Moreover, we observe the instability of other baselines under Beam Search. For instance, Qwen2.5-Math-PRM-72B exhibits a performance degradation as the search budget increases (33.56% → 30.89% → 32.44%). This phenomenon is often attributed to “reward hacking” where the greedy nature of Beam Search exploits inaccuracies in the reward model, leading to high-scoring but incorrect paths. In contrast, DataPRM maintains a consistent improvement (35.33% → 38.00% → 38.89%), indicating the robustness against the exploitative tendencies of search policies. 5.3 In-Depth Analysis
- 5.3.1 Environment Interaction is Critical for Data Analysis Tasks. To assess the efficacy and necessity of individual modules, we conduct an ablation study on the DataPRM architecture. As shown in Tab.3, we compared the full method with four variants:

SFT

RL with Outcome Reward

RL with DataPRM

| |
|---|

| |
|---|

| |
|---|

90

0.9

100

| | | | |
|---|---|---|---|
| | | | |
|Outcom<br><br>Proces|e Reward s Reward| | |
| | | | |

Pass@1 Pass@3

Pass@1 Pass@3

Outcome Reward

Process Reward

TrainingReward

0.20

0.8

80

###### Accuracy(%)

77.5

76.7

89.5

90

74.5

Entropy

86.8 86.8

0.7

70

0.15

64.8

78.7

80

77.4

0.6

76.0

61.5

60.2

60

0.10

0.5

70

0 100 Steps200 300

0 100 Steps200 300

DABench

TableBench

(a) DABench and TableBench Results for RL.

(b) Training Reward Dynamics.

(c) Entropy Dynamics.

Figure 5: Experiment results on RL training and benchmarks. (a): The evaluation results on DABench and TableBench for models trained with different strategies. (b) and (c): The training reward dynamics and entropy dynamics in RL training for outcome reward and process reward.

(1) CoT (Chain-of-Thought baseline), (2) Single-turn Code w/ Env, (3) Multi-turn Code w/o Env, and (4) Multi-turn Code w/ Env. First, equipping the model with the ability of environment interaction (Single-turn Code w/ Env) yields a consistent improvement over the CoT baseline, verifying that executable feedback helps ground the reasoning process. Second, while multi-turn interaction alone provides marginal gains, its combination with the environment (Multi-turn Code w/ Env) significantly boosts performance, suggesting that iterative refinement is most effective when supported by execution results. After incorporating the reflection-aware strategy, DataPRM achieves optimal performance, demonstrating that assigning fine-grained scores to exploratory steps helps in selecting the correct trajectory. We also observe that our proposed components are most pronounced on the Hard subset. While the CoT baseline struggles with complex reasoning (32.01% at 𝑁 = 16), introducing the Code Environment and Interaction (Multi-turn Code w/ Env) improves this to 32.80%. DataPRM further raises this to 33.86%, demonstrating the effectiveness of our method in complex data analysis tasks.

5.3.3 DataPRM Provides Efficient Verification for Agentic Workflows. We further analyze the inference cost of different verifier designs in Tab. 5. Compared with GenPRM, DataPRM incurs higher token usage and latency, which is expected because it actively interacts with the execution environment to verify intermediate states. Specifically, DataPRM uses 2.57 turns and 0.87 tool calls on average, enabling the verifier to ground its judgment in executable feedback rather than relying on textual reasoning. This additional cost is therefore a necessary trade-off for detecting silent errors and evaluating exploratory data-analysis trajectories.

Despite this overhead, DataPRM is substantially more efficient than the Self-Rewarding baseline. It reduces total token consumption from 25.3K to 21.5K tokens, corresponding to a 15.1% reduction, and decreases the average number of turns from 3.32 to 2.57, corresponding to a 22.6% reduction. This suggests that DataPRM performs more focused verification by using targeted environment interaction instead of lengthy self-evaluation. Moreover, we build a parallel evaluation environment with isolated file systems and lightweight string-based context tracking to mitigate the latency overhead. With this optimization, the practical verification latency is reduced from 24.66s to 3.30s per sample, demonstrating that DataPRM can be deployed in large-scale agentic evaluation without becoming an inference bottleneck.

- 5.3.2 Data Diversity Outweighs Purity for Scalable Reward Modeling. Since we do not have any requirement for the correctness of policy model trajectory answers in our data generation process, we explore three types of reference-free trajectory filtering methods [46]: Meta Critic, Outcome Consistency, and Process Consistency. Performance comparison with respect to the inference sampling budget 𝑁 (Best-of-𝑁) is reported in Tab.4. Counterintuitively, aggressive filtering does not consistently yield better reward modeling performance. While Process Consistency achieves a marginal gain at a low sampling budget (𝑁 = 4, 38.00% vs. 37.11%), the unfiltered baseline demonstrates superior scalability, significantly outperforming all filtered variants at 𝑁 = 16 (40.89%). We attribute this phenomenon to the trade-off between data purity and diversity. While strict filtering strategies can enhance data purity, they may also discard other effective and diverse step-wise supervision samples, leading the PRM to become overly conservative. In contrast, the PRM trained on the full dataset is exposed to the complete trajectory distribution. By learning from a richer set of step-wise supervision samples, it can more effectively distinguish correct solutions from a larger candidate pool.

### 5.4 Applying DataPRM to Agentic RL

As shown in Fig.5a, the model trained with process-supervised rewards achieves accuracy rates of 78.73% on DABench and 64.84% on TableBench, outperforming both the SFT model and the model trained with outcome-only rewards. Furthermore, as shown in Fig.5b and Fig.5c, a noticeable entropy collapse occurred when training with outcome-only rewards. After 200 steps, the entropy decreases to approximately 0.12 and the reward ceases to increase. In contrast, training with the incorporation of process-supervised rewards avoids this phenomenon. The entropy remains around 0.18, and the reward continues to rise steadily. This indicates that more fine-grained rewards can enable the model to conduct more thorough exploration. Similarly, the model trained with processsupervised rewards also demonstrates an increase in pass@3, which is likely attributable to the consistently high entropy maintained

throughout training. In contrast, the model trained with outcome rewards shows no growth in the pass@3 metric.

### 6 Related Work

6.1 Process Reward Models

Process Reward Models (PRMs) [27, 36, 82] are capable of providing granular rewards and demonstrate significant potential for applications in Test Time Scaling [17, 30, 53] and Reinforcement Learning [11, 13, 33, 49, 59]. Current PRMs primarily focus on scenarios that do not require environmental interaction, such as mathematics [22, 35, 57, 79, 80, 88], code generation [24, 68, 72], tabular reasoning [55, 78, 87], and others [23, 28, 84]. Recently, there has been a growing trend of applying PRMs to agent scenarios. Web-Shepherd [4] can provide step-wise feedback and reward for web navigation tasks using structured subgoal checklists. AgentPRM [61] employs Temporal Difference-based estimation method combined with Generalized Advantage Estimation, demonstrating excellent performance across multiple agent tasks. SWE-PRM [16] validates that using proprietary models as PRMs can enhance the capabilities of agents in the field of software engineering. To the best of our knowledge, this work represents the first systematic investigation of Process Reward Models (PRMs) within the domain of data analysis, with the broader aim of providing insights for other complex, agent-driven fields.

6.2 Data-Analytic Agents

Data analysis agents are aimed at autonomously accomplishing end-to-end data analysis tasks [10, 14, 20, 21, 32, 41, 60, 85, 86]. To handle complex data analysis problems in real-world scenarios, early approaches primarily relied on prompt engineering and predefined workflows to leverage the reasoning and coding capabilities of closed-source models in addressing these challenges, including data visualization [64], insight and report generation [1, 31, 38, 62, 77], heterogeneous data analysis [39, 42, 54, 76], general data science [19, 66], etc. Recently, an increasing number of data analysis agents have demonstrated promising performance through agentic training based on open-source models. DataMind [45] employs fine-grained query generation, knowledge-based trajectory sampling, and combined agent training paradigm of SFT and RL. DeepAnalyze [73] constructs a data-grounded trajectory synthesis framework and employs a curriculum-based agentic training paradigm. Both consistently achieve outstanding performance across multiple data analysis tasks. Unlike methods that rely on predefined workflows or data-driven model training, we utilize PRMs to enhance agents’ data analysis capability, offering a novel perspective through the lens of Test Time Scaling.

### 7 Conclusion

In this work, we introduced DataPRM, an environment-aware process reward model designed to overcome the limitations of general PRMs in detecting silent and grounding errors within interactive data analysis. By leveraging active environment verification and a ternary reward strategy, DataPRM provides precise step-level supervision. To construct DataPRM, we designed a scalable data generation pipeline utilizing diversity-driven trajectory generation and knowledge-enhanced expert annotation. Empirical results

demonstrate that DataPRM significantly enhances both Test-Time Scaling and Reinforcement Learning performance.

### 8 Limitations and Ethical Considerations

Our current work has several limitations. First, we focus primarily on data analysis tasks involving reasoning and visualization, leaving complex engineering tasks, such as model training and prediction, for future exploration. Second, we train DataPRM solely via Supervised Fine-Tuning (SFT), a paradigm that relies heavily on the availability of high-quality trajectory data. To mitigate this data dependency and further enhance the capabilities of PRMs, our future work will explore methods that require less human-curated data, such as Reinforcement Learning [87] and Skill [2, 26, 40, 56, 71].

This work follows established ethical research practices, utilizing only synthesized or publicly available datasets. We have accurately cited all sources to ensure transparency and proper attribution.

### Acknowledgments

We would like to express sincere gratitude to the reviewers for their thoughtful and constructive feedback. This work was supported by the National Natural Science Foundation of China (No. 62576307, No. NSFCU23B2055, No. NSFCU19B2027), the Fundamental Research Funds for the Central Universities (226-2023-00138), Yongjiang Talent Introduction Programme (2021A-156-G), and Information Technology Center and State Key Lab of CAD&CG, Zhejiang University. This work was supported by Ant Group and Zhejiang University Ant Group Joint Laboratory of Knowledge Graph.

### References

- [1] Amirhossein Abaskohi, Amrutha Varshini Ramesh, Shailesh Nanisetty, Chirag Goel, David Vázquez, Christopher Pal, Spandana Gella, Giuseppe Carenini, and Issam H. Laradji. 2025. AgentAda: Skill-Adaptive Data Analytics for Tailored Insight Discovery. CoRR abs/2504.07421 (2025). arXiv:2504.07421 doi:10.48550/ ARXIV.2504.07421
- [2] Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Weiyuan Chen, and Tu Vu. 2026. EvoSkill: Automated Skill Discovery for Multi-Agent Systems. CoRR abs/2603.02766 (2026). arXiv:2603.02766 doi:10.48550/ARXIV.2603.02766
- [3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. 2025. Qwen3-VL Technical Report. CoRR abs/2511.21631 (2025). arXiv:2511.21631 doi:10.48550/ARXIV.2511.21631
- [4] Hyungjoo Chae, Sunghwan Kim, Junhee Cho, Seungone Kim, Seungjun Moon, Gyeom Hwangbo, Dongha Lim, Minjin Kim, Yeonjun Hwang, Minju Gwak, Dongwook Choi, Minseok Kang, Gwanhoon Im, ByeongUng Cho, Hyojun Kim, Jun Hee Han, Taeyoon Kwon, Minju Kim, Beong-woo Kwak, Dongjin Kang, and Jinyoung Yeo. 2025. Web-Shepherd: Advancing PRMs for Reinforcing Web Agents. CoRR abs/2505.15277 (2025). arXiv:2505.15277 doi:10.48550/ARXIV.2505.15277
- [5] Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Weinan E, Yuzhi Zhang, Linfeng Zhang, and Siheng Chen. 2025. SciMaster: Towards General-Purpose Scientific AI Agents, Part I. X-Master as Foundation: Can We Lead on Humanity’s Last Exam? CoRR abs/2507.05241 (2025). arXiv:2507.05241 doi:10.48550/ARXIV.2507.05241
- [6] Jiangjie Chen, Wenxiang Chen, Jiacheng Du, Jinyi Hu, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Wenlei Shi, Zhihong Wang, Mingxuan Wang, Chenrui Wei, Shufa Wei, Huajian Xin, Fan Yang, Weihao Gao, Zheng Yuan, Tianyang Zhan, Zeyu Zheng, Tianxi Zhou, and Thomas Hanwen Zhu. 2025. Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning

- from Experience. CoRR abs/2512.17260 (2025). arXiv:2512.17260 doi:10.48550/ ARXIV.2512.17260
- [7] Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu, Binbin Lin, and Xiaofei He. 2024. AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (Eds.). http://papers.nips.cc/paper_files/paper/2024/hash/ 0142921fad7ef9192bd87229cdafa9d4-Abstract-Conference.html
- [8] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. 2025. Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models. CoRR abs/2503.09567 (2025). arXiv:2503.09567 doi:10.48550/ARXIV.2503.09567
- [9] Qiguang Chen, Ming-Hsuan Yang, Libo Qin, Jinhao Liu, Zheng Yan, Jiannan Guan, Dengyun Peng, Yiyan Ji, Hanjing Li, Mengkang Hu, Yimeng Zhang, Yihao Liang, Yu Zhou, Jiaqi Wang, Zhi Chen, and Wanxiang Che. 2025. AI4Research: A Survey of Artificial Intelligence for Scientific Research. CoRR abs/2507.01903

(2025). arXiv:2507.01903 doi:10.48550/ARXIV.2507.01903

- [10] Ziru Chen, Shijie Chen, Yuting Ning, Qianheng Zhang, Boshi Wang, Botao Yu, Yifei Li, Zeyi Liao, Chen Wei, Zitong Lu, Vishal Dey, Mingyi Xue, Frazier N. Baker, Benjamin Burns, Daniel Adu-Ampratwum, Xuhui Huang, Xia Ning, Song Gao, Yu Su, and Huan Sun. 2025. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https://openreview.net/forum?id=6z4YKr0GK6
- [11] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, Jiarui Yuan, Huayu Chen, Kaiyan Zhang, Xingtai Lv, Shuo Wang, Yuan Yao, Xu Han, Hao Peng, Yu Cheng, Zhiyuan Liu, Maosong Sun, Bowen Zhou, and Ning Ding. 2025. Process Reinforcement through Implicit Rewards. CoRR abs/2502.01456 (2025). arXiv:2502.01456 doi:10. 48550/ARXIV.2502.01456
- [12] DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. CoRR abs/2512.02556 (2025). arXiv:2512.02556 doi:10.48550/ARXIV.2512. 02556
- [13] Yuyang Ding, Chi Zhang, Juntao Li, Haibin Lin, Xin Liu, and Min Zhang. 2025. FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning. CoRR abs/2510.22543 (2025). arXiv:2510.22543 doi:10.48550/ARXIV.2510.22543
- [14] Alex Egg, Martin Iglesias Goyanes, Friso Kingma, Andreu Mora, Leandro von Werra, and Thomas Wolf. 2025. DABstep: Data Agent Benchmark for Multi-step Reasoning. CoRR abs/2506.23719 (2025). arXiv:2506.23719 doi:10.48550/ARXIV. 2506.23719
- [15] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. 2025. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. CoRR abs/2504.11536 (2025). arXiv:2504.11536 doi:10.48550/ARXIV.2504.11536
- [16] Shubham Gandhi, Jason Tsay, Jatin Ganhotra, Kiran Kate, and Yara Rizk. 2025. When Agents go Astray: Course-Correcting SWE Agents with PRMs. CoRR abs/2509.02360 (2025). arXiv:2509.02360 doi:10.48550/ARXIV.2509.02360
- [17] Xinyan Guan, Yanjiang Liu, Xinyu Lu, Boxi Cao, Ben He, Xianpei Han, Le Sun, Jie Lou, Bowen Yu, Yaojie Lu, and Hongyu Lin. 2024. Search, Verify and Feedback: Towards Next Generation Post-training Paradigm of Foundation Models via Verifier Engineering. CoRR abs/2411.11504 (2024). arXiv:2411.11504 doi:10.48550/ ARXIV.2411.11504
- [18] Feijuan He, Han Lai, Jiaqi Liu, Bo Wang, Haoran Chen, Haohan Liu, and Chenxi Zhang. 2025. Solving Mathematical Problems using Large Language Models: A Survey. Data Intell. 7, 4 (2025), 907–946. doi:10.3724/2096-7004.DI.2025.0064
- [19] Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Robert Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Yongxin Ni, Zhibin Gou, Zongze Xu, Yuyu Luo, and Chenglin Wu. 2025. Data Interpreter: An LLM Agent for Data Science. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 19796–19821. https://aclanthology.org/2025.findingsacl.1016/
- [20] Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Qianli Ma, Guoyin Wang, Xuwu Wang, Jing Su, Jingjing Xu, Ming Zhu, Yao Cheng, Jianbo Yuan, Jiwei Li, Kun Kuang, Yang Yang, Hongxia Yang, and Fei Wu. 2024. InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https://openreview.net/forum?id=d5LURMSfTx
- [21] Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, Hongming Zhang, Xinya Du, and Dong Yu. 2025. DSBench: How Far Are Data Science Agents from Becoming Data Science Experts?. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https://openreview.net/forum?id=DSsSPr0RZJ

- [22] Muhammad Khalifa, Rishabh Agarwal, Lajanugen Logeswaran, Jaekyeom Kim, Hao Peng, Moontae Lee, Honglak Lee, and Lu Wang. 2025. Process Reward Models That Think. CoRR abs/2504.16828 (2025). arXiv:2504.16828 doi:10.48550/ ARXIV.2504.16828
- [23] Dawei Li, Yuguang Yao, Zhen Tan, Huan Liu, and Ruocheng Guo. 2026. ToolPRMBench: Evaluating and Advancing Process Reward Models for Tool-using Agents. arXiv preprint arXiv:2601.12294 (2026).
- [24] Qingyao Li, Xinyi Dai, Xiangyang Li, Weinan Zhang, Yasheng Wang, Ruiming Tang, and Yong Yu. 2025. CodePRM: Execution Feedback-enhanced Process Reward Model for Code Generation. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 8169–8182. https: //aclanthology.org/2025.findings-acl.428/
- [25] Yifei Li, Hanane Nour Moussa, Ziru Chen, Shijie Chen, Botao Yu, Mingyi Xue, Benjamin Burns, Tzu-Yao Chiu, Vishal Dey, Zitong Lu, Chen Wei, Qianheng Zhang, Tianyu Zhang, Song Gao, Xuhui Huang, Xia Ning, Nesreen K. Ahmed, Ali Payani, and Huan Sun. 2025. AutoSDT: Scaling Data-Driven Discovery Tasks Toward Open Co-Scientists. CoRR abs/2506.08140 (2025). arXiv:2506.08140 doi:10.48550/ARXIV.2506.08140
- [26] Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Xin Xu, Tongtong Wu, Kun Wang, Yang Liu, Zhen Bi, Jungang Lou, Yuchen Eleanor Jiang, Hangcheng Zhu, Gang Yu, Haiwen Hong, Longtao Huang, Hui Xue, Chenxi Wang, Yijun Wang, Zifei Shan, Xi Chen, Zhaopeng Tu, Feiyu Xiong, Xin Xie, Peng Zhang, Zhengke Gui, Lei Liang, Jun Zhou, Chiyu Wu, Jin Shang, Yu Gong, Junyu Lin, Changliang Xu, Hongjie Deng, Wen Zhang, Keyan Ding, Qiang Zhang, Fei Huang, Ningyu Zhang, Jeff Z. Pan, Guilin Qi, Haofen Wang, and Huajun Chen.

2026. SkillNet: Create, Evaluate, and Connect AI Skills. CoRR abs/2603.04448

(2026). arXiv:2603.04448 doi:10.48550/ARXIV.2603.04448

- [27] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s Verify Step by Step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. https://openreview.net/forum?id=v8L0pN6EOi
- [28] Jianghao Lin, Yuanyuan Shi, Xin Peng, Renjie Ding, Hairui Wang, Yuxuan Peng, Bizhe Bai, Weixi Song, Fengshuo Bai, Huacan Chai, Weinan Zhang, Fei Huang, and Ying Wen. 2025. ToolPRM: Fine-Grained Inference Scaling of Structured Outputs for Function Calling. CoRR abs/2510.14703 (2025). arXiv:2510.14703 doi:10.48550/ARXIV.2510.14703
- [29] Xin Lin, Yajiao Wang, Zhixiong Zhang, and Mengting Zhang. 2025. Scientific Claim Recognition via Staged Fine-Tuning with LoRA. Data Intell. 7, 2 (2025), 303–335. doi:10.3724/2096-7004.DI.2025.0009
- [30] Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, and Bowen Zhou. 2025. Can 1B LLM Surpass 405B LLM? Rethinking ComputeOptimal Test-Time Scaling. CoRR abs/2502.06703 (2025). arXiv:2502.06703 doi:10. 48550/ARXIV.2502.06703
- [31] Shicheng Liu, Yucheng Jiang, Sajid Farook, Camila Nicollier Sanchez, David Fernando Castro Pena, and Monica S. Lam. 2026. DataSTORM: Deep Research on Large-Scale Databases using Exploratory Data Analysis and Data Storytelling. https://api.semanticscholar.org/CorpusID:287248168
- [32] Wei Liu, Peijie Yu, Michele Orini, Yali Du, and Yulan He. 2026. Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models. arXiv preprint arXiv:2602.02039 (2026).
- [33] Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Junge Zhang, and Jianbin Jiao. 2025. Agentic Reinforcement Learning with Implicit Step Rewards. CoRR abs/2509.19199 (2025). arXiv:2509.19199 doi:10.48550/ARXIV.2509.19199
- [34] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob N. Foerster, Jeff Clune, and David Ha. 2024. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. CoRR abs/2408.06292 (2024). arXiv:2408.06292 doi:10.48550/ARXIV. 2408.06292
- [35] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, and Abhinav Rastogi. 2024. Improve Mathematical Reasoning in Language Models by Automated Process Supervision. CoRR abs/2406.06592 (2024). arXiv:2406.06592 doi:10.48550/ARXIV.2406.06592
- [36] Yitian Luo, Yu Liu, Lu Zhang, Feng Gao, and Jinguang Gu. 2025. A Survey on Quality Evaluation of Instruction Fine-tuning Datasets for Large Language Models. Data Intell. 7, 3 (2025), 527–566. doi:10.3724/2096-7004.DI.2025.0021
- [37] Thang Luong, Dawsen Hwang, Hoang H. Nguyen, Golnaz Ghiasi, Yuri Chervonyi, Insuk Seo, Junsu Kim, Garrett Bingham, Jonathan Lee, Swaroop Mishra, Alex Zhai, Clara Huiyi Hu, Henryk Michalewski, Jimin Kim, Jeonghyun Ahn, Junhwi Bae, Xingyou Song, Trieu H. Trinh, Quoc V. Le, and Junehyuk Jung. 2025. Towards Robust Mathematical Reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, EMNLP 2025, Suzhou, China, November 4-9, 2025, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, 35418–35442. doi:10.18653/V1/2025.EMNLP-MAIN.1794

- [38] Pingchuan Ma, Rui Ding, Shuai Wang, Shi Han, and Dongmei Zhang. 2023. InsightPilot: An LLM-Empowered Automated Data Exploration System. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, Yansong Feng and Els Lefever (Eds.). Association for Computational Linguistics, 346–352. doi:10.18653/V1/2023.EMNLP-DEMO.31
- [39] Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, and Tomas Pfister. 2025. DS-STAR: Data Science Agent via Iterative Planning and Verification. CoRR abs/2509.21825

- (2025). arXiv:2509.21825 doi:10.48550/ARXIV.2509.21825

[40] Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills. CoRR abs/2603.25158

- (2026). arXiv:2603.25158 doi:10.48550/ARXIV.2603.25158

- [41] Fan Nie, Junlin Wang, Harper Hua, Federico Bianchi, Yongchan Kwon, Zhenting Qi, Owen Queen, Shang Zhu, and James Zou. 2026. DSGym: A Holistic Framework for Evaluating and Training Data Science Agents. arXiv preprint arXiv:2601.16344

(2026).

- [42] Ruyi Qi, Zhou Liu, and Wentao Zhang. 2026. DataCross: A Unified Benchmark and Agent Framework for Cross-Modal Heterogeneous Data Analysis. ArXiv abs/2601.21403 (2026). https://api.semanticscholar.org/CorpusID:285140426
- [43] Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek HakkaniTür, Gokhan Tur, and Heng Ji. 2025. ToolRL: Reward is All Tool Learning Needs. CoRR abs/2504.13958 (2025). arXiv:2504.13958 doi:10.48550/ARXIV.2504.13958
- [44] Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei Huang, and Huajun Chen. 2023. Reasoning with Language Model Prompting: A Survey. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, 5368–5393. doi:10. 18653/V1/2023.ACL-LONG.294
- [45] Shuofei Qiao, Yanqiu Zhao, Zhisong Qiu, Xiaobin Wang, Jintian Zhang, Zhao Bin, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen.

2025. Scaling Generalist Data-Analytic Agents. CoRR abs/2509.25084 (2025). arXiv:2509.25084 doi:10.48550/ARXIV.2509.25084

- [46] Salman Rahman, Sruthi Gorantla, Arpit Gupta, Swastik Roy, Nanyun Peng, and Yang Liu. 2025. SPARK: Stepwise Process-Aware Rewards for Reference-Free Reinforcement Learning. CoRR abs/2512.03244 (2025). arXiv:2512.03244 doi:10. 48550/ARXIV.2512.03244
- [47] Z. Z. Ren, Zhihong Shao, Junxiao Song, Huajian Xin, Haocheng Wang, Wanjia Zhao, Liyue Zhang, Zhe Fu, Qihao Zhu, Dejian Yang, Z. F. Wu, Zhibin Gou, Shirong Ma, Hongxuan Tang, Yuxuan Liu, Wenjun Gao, Daya Guo, and Chong Ruan. 2025. DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition. CoRR abs/2504.21801

(2025). arXiv:2504.21801 doi:10.48550/ARXIV.2504.21801

- [48] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. 2025. Agent Laboratory: Using LLM Agents as Research Assistants. CoRR abs/2501.04227 (2025). arXiv:2501.04227 doi:10.48550/ARXIV.2501.04227
- [49] Amrith Setlur, Chirag Nagpal, Adam Fisch, Xinyang Geng, Jacob Eisenstein, Rishabh Agarwal, Alekh Agarwal, Jonathan Berant, and Aviral Kumar. 2025. Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net. https://openreview.net/forum?id= A6Y7AqlzLW
- [50] Zhihong Shao, Yuxiang Luo, Chengda Lu, Z. Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. 2025. DeepSeekMath-V2: Towards SelfVerifiable Mathematical Reasoning. CoRR abs/2511.22570 (2025). arXiv:2511.22570 doi:10.48550/ARXIV.2511.22570
- [51] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. CoRR abs/2402.03300

(2024). arXiv:2402.03300 doi:10.48550/ARXIV.2402.03300

- [52] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A Flexible and Efficient RLHF Framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 3 April 2025. ACM, 1279–1297. doi:10.1145/3689031.3696075
- [53] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling LLM TestTime Compute Optimally can be More Effective than Scaling Model Parameters. CoRR abs/2408.03314 (2024). arXiv:2408.03314 doi:10.48550/ARXIV.2408.03314
- [54] Ji Sun, Guoliang Li, Peiyao Zhou, Yihui Ma, Jingzhe Xu, and Yuan Li. 2025. AgenticData: An Agentic Data Analytics System for Heterogeneous Data. CoRR abs/2508.05002 (2025). arXiv:2508.05002 doi:10.48550/ARXIV.2508.05002
- [55] Lei Tang, Wei Zhou, and Mohsen Mesgar. 2025. Exploring Generative Process Reward Modeling for Semi-Structured Data: A Case Study of Table Question Answering. CoRR abs/2510.20304 (2025). arXiv:2510.20304 doi:10.48550/ARXIV. 2510.20304

- [56] Chenxi Wang, Zhuoyun Yu, Xinghong Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, and Shumin Deng. 2026. SkillX: Automatically Constructing Skill Knowledge Bases for Agents. https://api.semanticscholar.org/CorpusID:287204111
- [57] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024. Math-Shepherd: Verify and Reinforce LLMs Stepby-step without Human Annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 9426–9439. doi:10.18653/V1/2024.ACL-LONG.510
- [58] Peiran Wang, Yaoning Yu, Ke Chen, Xianyang Zhan, and Haohan Wang. 2025. Large Language Model-based Data Science Agent: A Survey. CoRR abs/2508.02744

(2025). arXiv:2508.02744 doi:10.48550/ARXIV.2508.02744

- [59] Tongyu Wen, Guanting Dong, and Zhicheng Dou. 2026. SmartSearch: Process Reward-Guided Query Refinement for Search Agents. arXiv preprint arXiv:2601.04888 (2026).
- [60] Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xeron Du, Di Liang, Daixin Shu, Xianfu Cheng, Tianzhen Sun, Tongliang Li, Zhoujun Li, and Guanglin Niu. 2025. TableBench: A Comprehensive and Complex Benchmark for Table Question Answering. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, Toby Walsh, Julie Shah, and Zico Kolter (Eds.). AAAI Press, 25497–25506. doi:10.1609/AAAI.V39I24.34739
- [61] Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, and Xuanjing Huang. 2025. AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress. CoRR abs/2511.08325 (2025). arXiv:2511.08325 doi:10.48550/ARXIV.2511.08325
- [62] Wenyi Xu, Yuren Mao, Xiaolu Zhang, Chao Zhang, Xuemei Dong, Mengfei Zhang, and Yunjun Gao. 2025. DAgent: A Relational Database-Driven Data Analysis Report Generation Agent. CoRR abs/2503.13269 (2025). arXiv:2503.13269 doi:10.48550/ARXIV.2503.13269
- [63] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report. CoRR abs/2505.09388 (2025). arXiv:2505.09388 doi:10.48550/ARXIV.2505.09388
- [64] Zhiyu Yang, Zihan Zhou, Shuo Wang, Xin Cong, Xu Han, Yukun Yan, Zhenghao Liu, Zhixing Tan, Pengyuan Liu, Dong Yu, Zhiyuan Liu, Xiaodong Shi, and Maosong Sun. 2024. MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, 11789–11804. doi:10.18653/V1/2024.FINDINGS-ACL.701
- [65] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview. net/forum?id=WE_vluYUL-X
- [66] Ziming You, Yumiao Zhang, Dexuan Xu, Yiwei Lou, Yandong Yan, Wei Wang, Huaming Zhang, and Yu Huang. 2025. DatawiseAgent: A Notebook-Centric LLM Agent Framework for Automated Data Science. CoRR abs/2503.07044 (2025). arXiv:2503.07044 doi:10.48550/ARXIV.2503.07044
- [67] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. 2025. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. CoRR abs/2503.14476 (2025). arXiv:2503.14476 doi:10.48550/ARXIV.2503.14476
- [68] Zhuohao Yu, Weizheng Gu, Yidong Wang, Zhengran Zeng, Jindong Wang, Wei Ye, and Shikun Zhang. 2024. Outcome-Refining Process Supervision for Code Generation. CoRR abs/2412.15118 (2024). arXiv:2412.15118 doi:10.48550/ARXIV. 2412.15118
- [69] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason Weston. 2024. Self-Rewarding Language Models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. https://openreview.net/forum?id= 0NphYCmgua

- [70] Daojian Zeng, Lin Zhou, Zhiheng Zhang, and Lincheng Jiang. 2025. AuToGen: Automated Tool Learning Data Generation with Domain-specific Structured Data. Data Intell. 7, 4 (2025), 1108–1128. doi:10.3724/2096-7004.DI.2024.0005
- [71] Hanrong Zhang, Shichen Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayuan Zhou, Chengze Li, Wei-Chieh Huang, Yifei Yao, Kening Zheng, Xue Liu, Xiaoxiao Li, and Philip S. Yu. 2026. CoEvoSkills: Self-Evolving Agent Skills via CoEvolutionary Verification. https://api.semanticscholar.org/CorpusID:287071917
- [72] Ruiyi Zhang, Peijia Qin, Qi Cao, Eric Xue, and Pengtao Xie. 2026. FunPRM: Function-as-Step Process Reward Model with Meta Reward Correction for Code Generation. arXiv preprint arXiv:2601.22249 (2026).
- [73] Shaolei Zhang, Ju Fan, Meihao Fan, Guoliang Li, and Xiaoyong Du. 2025. DeepAnalyze: Agentic Large Language Models for Autonomous Data Science. CoRR abs/2510.16872 (2025). arXiv:2510.16872 doi:10.48550/ARXIV.2510.16872
- [74] Shimao Zhang, Xiao Liu, Xin Zhang, Junxiao Liu, Zheheng Luo, Shujian Huang, and Yeyun Gong. 2025. Process-based Self-Rewarding Language Models. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 18097–18110. https://aclanthology.org/2025.findings-acl.930/
- [75] Wenlin Zhang, Xiaopeng Li, Yingyi Zhang, Pengyue Jia, Yichao Wang, Huifeng Guo, Yong Liu, and Xiangyu Zhao. 2025. Deep Research: A Survey of Autonomous Research Agents. CoRR abs/2508.12752 (2025). arXiv:2508.12752 doi:10.48550/ ARXIV.2508.12752
- [76] Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. 2023. DataCopilot: Bridging Billions of Data and Humans with Autonomous Workflow. CoRR abs/2306.07209 (2023). arXiv:2306.07209 doi:10.48550/ARXIV.2306.07209
- [77] Xilin Zhang, Zhixin Mao, Ziwen Chen, and Shen Gao. 2024. Effective Tool Augmented Multi-Agent Framework for Data Analysis. Data Intell. 6, 4 (2024), 923–945. doi:10.3724/2096-7004.DI.2024.0013
- [78] Yuxin Zhang, Meihao Fan, Ju Fan, Mingyang Yi, Yuyu Luo, Jian Tan, and Guoliang Li. 2025. Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and ProcessSupervised Rewards. CoRR abs/2505.04671 (2025). arXiv:2505.04671 doi:10.48550/ ARXIV.2505.04671
- [79] Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. The Lessons of Developing Process Reward Models in Mathematical Reasoning. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 10495–10516. https: //aclanthology.org/2025.findings-acl.547/
- [80] Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, and Bowen Zhou. 2025. GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning. CoRR abs/2504.00891 (2025). arXiv:2504.00891 doi:10.48550/ARXIV.2504.00891
- [81] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. 2025. SWIFT: A Scalable Lightweight Infrastructure for Fine-Tuning. In AAAI-25, Sponsored by the Association for the Advancement of Artificial Intelligence, February 25 - March 4, 2025, Philadelphia, PA, USA, Toby Walsh, Julie Shah, and Zico Kolter (Eds.). AAAI Press, 29733–29735. doi:10.1609/AAAI.V39I28.35383
- [82] Congming Zheng, Jiachen Zhu, Zhuoying Ou, Yuxiang Chen, Kangning Zhang, Rong Shan, Zeyu Zheng, Mengyue Yang, Jianghao Lin, Yong Yu, and Weinan Zhang. 2025. A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. CoRR abs/2510.08049 (2025). arXiv:2510.08049 doi:10.48550/ARXIV.2510.08049
- [83] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (Eds.). http://papers.nips.cc/paper_files/paper/2023/hash/ 91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html
- [84] Yuanchen Zhou, Shuo Jiang, Jie Zhu, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. 2025. Fin-PRM: A Domain-Specialized Process Reward Model for Financial Reasoning in Large Language Models. CoRR abs/2508.15202 (2025). arXiv:2508.15202 doi:10.48550/ARXIV.2508.15202
- [85] Yizhang Zhu, Liangwei Wang, Chenyu Yang, Xiaotian Lin, Boyan Li, Wei Zhou, Xinyu Liu, Zhangyang Peng, Tianqi Luo, Yu Li, Chengliang Chai, Chong Chen, Shimin Di, Ju Fan, Ji Sun, Nan Tang, Fugee Tsung, Jiannan Wang, Chenglin Wu, Yanwei Xu, Shaolei Zhang, Yong Zhang, Xuanhe Zhou, Guoliang Li, and Yuyu Luo.

2025. A Survey of Data Agents: Emerging Paradigm or Overstated Hype? ArXiv abs/2510.23587 (2025). https://api.semanticscholar.org/CorpusID:282389107

- [86] Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, and Ningyu Zhang. 2025. Why Do OpenSource LLMs Struggle with Data Analysis? A Systematic Empirical Study. CoRR abs/2506.19794 (2025). arXiv:2506.19794 doi:10.48550/ARXIV.2506.19794

- [87] Jiaru Zou, Soumya Roy, Vinay Kumar Verma, Ziyi Wang, David Wipf, Pan Lu, Sumit Negi, James Zou, and Jingrui He. 2025. TaTToo: Tool-Grounded Thinking PRM for Test-Time Scaling in Tabular Reasoning. CoRR abs/2510.06217 (2025). arXiv:2510.06217 doi:10.48550/ARXIV.2510.06217
- [88] Jiaru Zou, Ling Yang, Jingwen Gu, Jiahao Qiu, Ke Shen, Jingrui He, and Mengdi Wang. 2025. ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-ofThought Reasoning in LLMs. CoRR abs/2506.18896 (2025). arXiv:2506.18896 doi:10.48550/ARXIV.2506.18896

### A Theoretical Perspective for Environment-Aware Verifier

We formalize data analysis as a Partially Observable Markov Decision Process (POMDP) where the true environment state is a latent variable 𝜀. To evaluate an agent’s trajectory, traditional static PRMs must implicitly estimate this unknown environment by relying on an internal prior distribution 𝑃prior(𝜀|ℎ𝑡) learned during training. However, real-world scientific data is highly heterogeneous and frequently out-of-distribution (𝜀true ∉ 𝑃prior). This uncertainty causes the "Incorrect Rewarding for Silent Errors" (Tab.1), where the PRM hallucinates a compatible environment.

DataPRM mitigates this via explicit interaction, drawing groundtruth observations 𝑜𝑡 ∼ 𝑃(𝑂 | 𝜀,𝑎𝑡,ℎ𝑡) to update the uncertain prior into an accurate posterior via Bayes’ theorem:

𝑃post(𝜀 | 𝑜𝑡,𝑎𝑡,ℎ𝑡) ∝ 𝑃(𝑜𝑡 | 𝜀,𝑎𝑡) · 𝑃prior(𝜀 | ℎ𝑡)

Mechanistically, environmental interaction acts as a necessary Bayesian evidence-gathering step that grounds latent variables and reduces reward estimator variance.

Furthermore, this Bayesian perspective rigorously derives our ternary reward. In an exploratory POMDP, optimal steps require balancing exploitation (task progress) and exploration (uncertainty reduction). Therefore, the reward of an agent’s step 𝑅(𝑎𝑡) can be theoretically composed of two parts Progress towards the Final Goal 𝐺 and Information Gain about the Hidden Environment 𝐼. We formalize the reward as a balanced combination (𝜆=0.5):

𝑅(𝑎𝑡) = 𝜆 · 𝐺(𝑎𝑡) + (1 − 𝜆) · 𝐼(𝑎𝑡)

Here, 𝐼(𝑎𝑡) is the KL divergence 𝐷KL(𝑃post∥𝑃prior). Because reliably annotating continuous rewards for KL divergence is intractable in practice, we approximate the Information Gain using an indicator function, I[𝐼(𝑎𝑡) > 𝜖] for a small threshold 𝜖, signifying effective information gain. This maps exactly to our 3-value mechanism:

- • Strictly Correct (𝑅 = 1): The action makes progress on the task (𝐺 = 1) and confirms the validity of the current logic (I = 1).
- • Grounding / Correctable Error (𝑅 = 0.5): The action fails to make direct task progress (𝐺 = 0), but the resulting observation provides critical information about the environment (I = 1).
- • Irrecoverable Error (𝑅 = 0): The action makes no progress (𝐺 = 0) and yields no information or produces hallucinations (I = 0).

