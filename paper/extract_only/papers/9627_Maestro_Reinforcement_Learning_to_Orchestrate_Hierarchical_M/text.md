# arXiv:2605.22177v1[cs.LG]21May2026

## MAESTRO: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles

Jinyang Wu1⋆, Guocheng Zhai1⋆, Ruihan Jin1⋆, Yuhao Shen2, Zhengxi Lu2, Fan Zhang3, Haoran Luo4, Zheng Lian5†, Zhengqi Wen1†, Jianhua Tao1 1Tsinghua University 2Zhejiang University 3The Chinese University of Hong Kong 4Nanyang Technological University 5Tongji University

wu-jy23@mails.tsinghua.edu.cn

### Abstract

The proliferation of large language models (LLMs) and modular skills has endowed autonomous agents with increasingly powerful capabilities. Existing frameworks typically rely on monolithic LLMs and fixed logic to interface with these skills. This gives rise to a critical bottleneck: different LLMs offer distinct advantages across diverse domains, yet current frameworks fail to exploit the complementary strengths of models and skills, thereby limiting their performance on downstream tasks. In this paper, we present MAESTRO (Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration), a Reinforcement Learning (RL)-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. Rather than consolidating all knowledge into a single model, MAESTRO trains a lightweight policy to dynamically compose ensembles of frozen expert models and a two-tier skill library, deciding at each step whether to invoke an external expert, which model-skill pair to select, and when to terminate. The policy is optimized via outcome-based RL, requiring no step-level supervision. We evaluate MAESTRO across ten representative multimodal benchmarks spanning mathematical reasoning, chart understanding, high-resolution perception, and domain-specific analysis. With only a 4B orchestrator, MAESTRO achieves an average accuracy of 70.1%, surpassing both GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Crucially, the learned coordination policy generalizes to unseen models and skills without retraining: augmenting the registry with out-of-domain experts yields a 59.5% average on four challenging benchmarks, outperforming all closed-source baselines. MAESTRO further maintains high computational efficiency with low latency, offering a scalable and robust pathway for deploying collaborative agentic ecosystems. The source code is available at https://github.com/jinyangwu/Maestro.

### 1 Introduction

The evolution of Large Language Models (LLMs) from static knowledge bases to autonomous agents has been significantly propelled by the integration of modular skills and specialized expert models [17, 43, 62]. Early frameworks explored utilizing language models to dispatch tasks across diverse model repositories [46]. As the ecosystem scales to include tens of thousands of functional tools [49], subsequent research has introduced specialized retrieval techniques and hierarchical organizational strategies to manage massive API registries [38, 10]. These components are now treated as firstclass capabilities within extensive modern registries [2, 33, 34]. However, a critical coordination bottleneck emerges as the diversity of backbones and specialized skills scales: multimodal tasks are

Preprint.

###### (a) Traditional agent : Fixed Logic （b）Maestro: Model–Skill Orchestration ( Dynamic, Self-evolved )

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Environment

[Figure 5]

[Figure 6]

Skills Library

|| | | |
|---|---|---|
| | | |
| | | |
<br><br>…|
|---|

[Figure 7]

[Figure 8]

[Figure 9]

4B

Static retrieval / hand-written logic

[Figure 10]

No model–skill coordination

Suitable model

Multi-modal task

[Figure 11]

[Figure 12]

Orchestrator

[Figure 13]

[Figure 14]

[Figure 15]

Trajectory

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Task-specific skill Candidate LLM pool

Single-modal

skills library

Single-backbone

Trajectory

task

model

Hierarchical Skills

- Figure 1: Architectural comparison of agent paradigms. (Left) Traditional agents utilize a monolithic model with fixed logic to interface with skills. (Right) MAESTRO employs an RL-trained orchestrator to dynamically compose task-specific ensembles of expert models and hierarchical skills based on accumulated environmental feedback.

inherently heterogeneous, where solving a geometric proof, parsing a medical report, or counting objects in a high-resolution satellite image requires vastly different inductive biases and expertise.

Existing frameworks typically rely on static retrieval-based dispatching or a uniform approach centered around a single backbone model [66, 65]. While some methods attempt to enhance performance by constructing specialized tool sets [70, 30, 69], they generally operate under the implicit assumption that a single model can effectively utilize any retrieved skill regardless of the task domain or modality. This assumption often fails in realistic, large-scale deployments where the functional nuances of a skill require alignment with a specific model’s expertise to ensure success. Furthermore, established benchmarks [20, 40, 15] primarily evaluate downstream tool selection or single-model reasoning. This leaves a significant gap in understanding the synergistic interdependencies between heterogeneous LLMs and modular skills in complex, multi-step multimodal scenarios.

In this paper, we propose a paradigm shift in autonomous agent design: rather than consolidating all specialized knowledge into a monolithic model, we train a high-level orchestrator to strategically coordinate heterogeneous external capabilities. To this end, we introduce MAESTRO, a generalizable Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration. As shown in Figure 1, MAESTRO reframes multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. At each reasoning step, the orchestrator dynamically evaluates the state to determine: (i) the necessity of external delegation, (ii) the selection of the optimal expert model, (iii) the invocation of task-specific modular skills, and (iv) the satisfaction of termination criteria. The registry is organized into a two-tier hierarchy: coarse-grained Level-1 skills exposed to the orchestrator, and fine-grained Level-2 skills that support specialized reasoning through keyword-based activation or expert-model classification. Unlike prior frameworks restricted by static dispatching, MAESTRO optimizes its orchestration policy via outcome-based RL, enabling the discovery of latent synergies between reasoning backbones and fine-grained perception tools that often elude heuristic-based pipelines.

We evaluate MAESTRO on 10+ representative multimodal benchmarks spanning mathematical reasoning, chart understanding, medical analysis, high-resolution perception, embodied question answering, and other challenging scenarios. Our empirical results demonstrate that RL-based routing significantly improves task success rates over state-of-the-art baselines. We show that our policy effectively bridges the gap between general-purpose reasoning and domain-specific expertise, achieving these gains with remarkable token efficiency and low serving latency. Our contributions are as follows:

- • We introduce MAESTRO, a generalizable orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making problem over a hierarchical model-skill registry.
- • We formalize model-skill coordination as a finite-horizon POMDP and train the orchestration policy via outcome-based RL, requiring no step-level supervision of routing decisions.
- • We design a two-tier hierarchical skill library paired with a multi-expert model pool, enabling compositional and extensible orchestration across diverse task domains.

- • We demonstrate our 4B orchestrator’s strong performance (70.1%), exceeding frontier models (e.g., GPT-5), and plug-and-play generalization to unseen models and skills without retraining.

### 2 Related Works

LLM Agent and Skills. LLM-based agents have evolved from prompt-based interaction to modular systems capable of autonomous reasoning and tool invocation [35, 57, 37]. Early frameworks relied on fixed reasoning traces or predefined action spaces [67, 60], whereas recent work encapsulates taskspecific procedures as reusable skills to improve adaptability [78, 51, 29]. For example, SkillX [50] introduces hierarchical skill representations for structured knowledge distillation, and AutoSkill [66] supports lifelong experience accumulation through autonomous skill evolution. Other efforts scale skill management via retrieval and reranking pipelines [16, 64]. However, most agents remain tied to a single backbone model, limiting their robustness across domains. In contrast, our work introduces a multi-model orchestration layer that jointly optimizes skill selection and model assignment.

Reinforcement Learning for Agent Optimization. Reinforcement learning (RL) has become an effective paradigm for aligning LLM agents with complex task objectives and human preferences [36, 47, 11, 76, 61]. Compared with supervised fine-tuning, which depends on static demonstrations, RL enables agents to explore and discover effective behaviors through trial and error [44, 31, 71]. Recent studies further show the potential of recursive RL for co-evolving agent policies and skill banks [65], as well as for balancing task performance with computational constraints such as token efficiency in long-context or visual-heavy settings [22, 12]. We build upon these RL-based tuning strategies but shift the focus toward training a high-level policy model to navigate the combinatorial search space of model-skill combinations.

Multimodal LLM Collaboration. Extending LLM agents to multimodal environments requires the seamless integration of visual perception and linguistic reasoning [48]. Existing multimodal agents often rely on specialized VLMs or executable vision tools [24, 59]. Recent frameworks such as AppAgent V2 [21] and InternVideo2 [56] employ structured action spaces and modular tools for complex visual tasks, while optical self-compression [12] and hierarchical memory [68] address the challenge of high-density multimodal histories. Nevertheless, the synergy between visual tool affordances and the heterogeneous reasoning strengths of different LLMs remains under-explored. Our work addresses this gap through policy-driven routing, showing that aligning perception skills with suitable reasoning backbones is essential for complex multimodal orchestration.

### 3 Method

As illustrated in Figure 2, we present the MAESTRO framework, a non-invasive orchestration system that utilizes an RL-driven policy model to dynamically compose optimal ensembles of models and skills, enabling adaptive, multi-step reasoning in complex multimodal environments.

#### 3.1 Preliminaries

LLM Agent. We consider an agent interacting with a multimodal environment E = (S,A,P), where S denotes the set of observable states, A denotes the action space and P(· | s,a) denotes the transition dynamics. Let q and x denote a multimodal query and its associated context (e.g., images), respectively. The agent maintains a context ct = (q,x,a1,o1,...,at−1,ot−1) and at each time step t, the agent receives an observation ot ∼ E(· | at,x) and generates an action from a policy:

at ∼ π(· | ct), at ∈ A. (1)

The environment then transitions to a new state according to P(st+1 | st,at). The final trajectory is τ = (q,x,a1,o1,...,aT,oT).

Skill-Conditioned Execution. To reduce redundant exploration and improve task completion in complex domains, we equip the agent with a hierarchical skills library K = {k1,...,kn}. In traditional skill-augmented frameworks, a retrieval function ρ : Q × X → 2K provides a relevant subset of skills K′ = ρ(q,x) ⊆ K for a given task. The agent then generates a trajectory by conditioning on these retrieved skills:

τ′ ∼ π(· | K′,q,x). (2)

Candidate LLM pool

[Figure 22]

[Figure 23]

[Figure 24]

|Skills<br><br>[Figure 25]|
|---|

|Hierarchical Skills<br><br>[Figure 26]|
|---|

[Figure 27]

[Figure 28]

[Figure 29]

LLMs

Selected LLM

[Figure 30]

Hierarchical Skills Library

A. Compositional

B. Hierarchical

Selected Level 1-Skill

###### Chart Problem Solver

Level 1

[Figure 31]

Analyzes diverse data

Search

Skills Allocation

visualizations by routing tasks to three specialized solvers…

###### task query

###### Bar Chart Solver

Level 2

[Figure 32]

sample

[Figure 33]

[Figure 34]

[Figure 35]

4B

Utilizes OCR to parse titles,

Examine the distribution of

[Figure 36]

database

axes, and legends…

global smartphone operating system market…

###### Pie Chart Solver

Level 2

Selected Level 2-Skill

[Figure 37]

Specializes in proportional reasoning. Extracts sector labels…

Orchestrator

Skill-Conditioned Execution 𝜏′ ∼ 𝜋(⋅ ∣ 𝛫′,𝑞,𝑥)

…

Level 1 Geometry Problem Solver Dedicated to resolving complex Euclidean geometry tasks…

###### C. Multi-step Reasoning Loop

###### D. Policy Update

[Figure 38]

| |𝑅(𝜏) = 𝑟ans + 𝑟fmt| |
|---|---|---|
| | | |

Context Transition 𝑐𝑡+1 = Concat(𝑐𝑡,𝑎𝑡, 𝑜𝑡ctx)

…

Level 1 Science Problem Solver Tailored for tasks involving schematics and imagery…

…

|𝑨𝑮|
|---|

|𝑨𝟏| |𝑨𝟐| |𝑨𝟑|
|---|---|---|---|---|

parse

|Answer<br><br>[Figure 39]|
|---|

|Trajectory 𝜏<br><br>[Figure 40]|
|---|

[Figure 41]

Group Relative Advantage

…

- Figure 2: Overview of the MAESTRO framework. Unlike static agent systems, MAESTRO treats model selection and skill invocation as a unified compositional action space. The orchestrator (the policy model) dynamically determines which expert model should use which skill for the current reasoning step. This iterative reasoning process is optimized by a multi-dimensional reward function to guarantee logical consistency and precise grounding in multimodal environments.

The fundamental objective is to design the usage of K within π such that the expected success rate is significantly improved over direct reasoning:

Eq∈Q,τ′∼π(·|K′,q,x)[R(τ′,q)] > Eq∈Q,τ∼π(·|q,x)[R(τ,q)]. (3)

Heterogeneous Registries in MAESTRO. While previous works treat skills as standalone tools, MAESTRO introduces a dual-registry system. In addition to the skills library K, we maintain a candidate LLM pool M = {m1,...,ml}. Each m ∈ M represents a frozen expert LLM with distinct inductive biases (e.g., visual perception, mathematical reasoning, or code generation). Unlike static retrieval, our framework aims to learn a dynamic mapping that selects the optimal model-skill ensemble for each reasoning step. The agent maintains a time-varying context ct = (q,x,a1,o1,...,at−1,ot−1), where each action at is sampled from the orchestrator policy πθ(· | ct).

#### 3.2 Problem Formulation

We formalize the dynamic orchestration of models and skills as a finite-horizon Partially Observable Markov Decision Process (POMDP), defined by the tuple (S,A,O,P,R,γ,T). In this setting, the orchestrator acts as a high-level conductor, where the objective is to generate an optimal trajectory τ that maximizes task-specific utility through strategic resource allocation.

Compositional Action Space. The action space A is partitioned into three functional primitives: latent reasoning, external searching, and terminal answering. A distinguishing feature of MAESTRO is the compositional search action, which treats model selection and skill invocation as a unified decision. Formally, a search action at step t is defined as a triplet:

asearcht = (mt,st,zt) (4)

where mt ∈ M denotes the selected expert backbone, st ∈ K represents the functional skill, and zt is the semantic query string dispatched to the ensemble. In the deployment protocol, this is serialized as <search> Model@@Skill: Query </search>. This structured formulation explicitly forces the policy πθ to internalize the cross-modal compatibility between heterogeneous backbones and modular tools. Conversely, the termination action is defined as aanst = yt, where yt is the final resolution encapsulated within <answer> tags.

Context Transition. Upon the execution of asearcht , the environment E yields a raw observation ot (e.g., visual coordinates, scientific facts, or chart data). To maintain the structural integrity of the reasoning chain, we wrap this feedback into a standardized context-injection block:

octxt = <information>,ot,</information> (5)

The transition logic follows a recursive concatenation: ct+1 = Concat(ct,at,octxt ). This mechanism ensures that the orchestrator’s belief state is continuously refined by grounding its subsequent decisions in the evidence accumulated from prior expert invocations.

#### 3.3 RL-Driven Sequential Orchestration

MAESTRO resolves complex multimodal tasks through a "perceive-then-reason" iterative loop. The policy πθ is trained to interleave internal latent reasoning (within <think> tags) with the aforementioned dynamic external invocations.

Optimization Objective. We optimize the policy parameters θ to maximize the expected total reward over the trajectory distribution:

[R(τ)]. (6)

J(θ) = Eτ∼π

θ

To handle the sparse rewards inherent in long-horizon reasoning, we employ Group Relative Policy Optimization (GRPO). Specifically, for each query, we sample a group of G trajectories {τ1,...,τG}. The advantage Ai for trajectory τi is computed as Ai = (Ri − R¯)/(σR + ϵ), where R¯ and σR are the mean and standard deviation of rewards within the group. The orchestrator is optimized via the clipped surrogate objective:

G

1 G

min(ρi(θ)Ai,clip(ρi(θ),1 − ε,1 + ε)Ai) (7)

LGRPO(θ) = −

i=1

where ρi(θ) is the probability ratio between the current and previous policies.

Token-level Policy Gradient with Masking. In our framework, the context ct is a hybrid sequence consisting of both policy-generated tokens and environment-provided observation tokens. To prevent the policy from erroneously attempting to model the distribution of external environment feedback, we apply an indicator mask Action during training. The token-level policy loss is defined as:

N

wi∈Action log πθ(wi | w<i) (8)

Lpolicy = −

i=1

where wi represents the i-th token in the trajectory τ. By effectively zeroing out the loss contribution of observation tokens (i.e., tokens within <information> blocks), this objective concentrates the optimization effort solely on the orchestrator’s strategic reasoning and routing capabilities.

#### 3.4 Multi-Dimensional Reward Modeling

The reward function R(τ) is designed to balance task accuracy with structural rigor, consisting of two primary components:

R(τ) = rans + rfmt (9) The outcome reward rans provides a sparse task-dependent signal, where rans = 1 if the final output yT enclosed by <answer> tags is correct and 0 otherwise. To ensure reliable multi-agent communication, the format reward rfmt penalizes malformed trajectories with rfmt = −1 when any protocol constraint is violated: all XML-style tags must be balanced; each step must contain exactly one pair of <think> tags; the number of <search> calls must match the number of <information> blocks; the selected model mt and skill st must be valid identifiers in M and K; and the trajectory must terminate with exactly one <answer> block. This reward design guides the orchestrator to explore the combinatorial model-skill space while preserving the structural consistency required for multi-turn settings.

### 4 Experiments

#### 4.1 Experimental Setup

LLM Pool and Hierarchical Skills Library. In the main experiments, MAESTRO operates over five frozen expert models with complementary capabilities: GLM-4.6V-Flash (9B) [72], ChartR1 (8B) [8], Qwen3-VL-8B-Instruct [4], Intern-S1-mini (9B) [3], and MedGemma-1.5-4b-it [45].

- Table 1: Performance comparison on in-domain and out-of-domain benchmarks. We evaluate our proposed MAESTRO against proprietary closed-source models, open-source models, and recent “Think with Images” methods. “∆ vs. best” reports our method’s absolute performance gain compared to the strongest baseline in “Think with Images” methods.

In-Domain Out-of-Domain

Method

Avg. Geom ChartQA Slake MicroVQA MSE TallyQA VStar HRB-4K HRB-8K MathV

Closed-Source Models

- GPT-4o 34.1 81.4 58.5 47.8 42.0 77.8 66.0 59.0 55.0 30.4 55.2
- GPT-5 73.5 76.7 61.8 57.0 61.8 79.2 72.5 75.3 74.1 61.5 69.3 Gemini-2.5-Flash 67.4 79.6 56.0 58.6 54.0 80.6 72.3 79.4 73.7 39.8 66.1 Gemini-2.5-Pro 68.6 83.6 56.8 59.2 55.8 79.0 79.1 83.3 81.5 39.8 68.7

Open-Source & Baselines GLM-4.6V 60.4 85.0 63.1 51.0 54.2 82.2 81.2 76.6 73.0 39.1 66.6 Kimi-K2.5 68.7 79.4 59.6 51.6 55.0 78.4 72.8 68.4 65.1 53.3 65.2 Qwen3-VL-32B 68.9 77.8 57.6 52.6 51.0 78.6 78.0 75.0 69.5 45.4 65.4 Direct Answering 16.6 76.8 56.0 40.8 39.8 74.8 77.0 72.4 68.1 24.9 54.7 Untrained Model 38.9 74.8 54.3 38.8 36.8 74.4 41.4 70.3 68.5 29.0 52.7

Think with Images Methods DeepEyes 20.8 69.4 58.7 48.8 45.0 73.0 85.6 75.1 72.6 26.6 57.6 DeepEyes-v2 38.9 72.2 66.2 41.4 46.4 70.6 81.8 77.9 73.8 28.9 59.8 Thyme 17.5 86.1 62.6 48.8 42.2 73.2 82.2 77.0 72.0 27.6 58.9 VTOOL-R1 24.1 86.7 60.7 43.8 45.4 79.4 78.5 68.5 66.4 29.3 58.3 VTS-V 21.5 81.2 57.9 49.4 45.4 72.8 75.9 69.8 67.3 27.0 56.8 MathCoder-VL 26.5 78.8 54.0 44.0 43.8 73.4 77.5 73.8 70.6 26.0 56.8 Visual-ARFT 22.5 79.0 58.5 50.4 46.6 70.8 58.6 58.9 54.0 21.4 52.1 VisionReasoner 21.1 79.2 56.8 49.0 45.8 70.2 59.7 68.5 66.5 21.7 53.9 PixelReasoner 34.6 76.2 59.8 50.8 46.2 71.8 81.7 68.6 65.4 23.4 57.9 Chain-of-Focus 20.0 68.8 48.2 47.4 44.2 72.6 82.2 71.0 67.5 21.1 54.3

MAESTRO (Ours) 77.4 86.8 66.2 53.0 52.4 79.8 88.0 79.6 74.4 43.4 70.1 ∆ vs. best +38.5 +0.1 +0.0 +2.2 +5.8 +0.4 +2.4 +1.7 +0.6 +14.1 +10.3

The skill library K adopts a two-tier hierarchy. The orchestrator selects among five Level-1 skills: Geometric Problem Solver, Chart Problem Solver, Counting Problem Solver, Perception Problem Solver, and Science Problem Solver, which are further mapped to 8 fine-grained Level-2 skills through keyword matching or expert-model classification. This hierarchical routing effectively constrains the action space of the orchestrator while maintaining expert-level precision. Full details are provided in Appendix C.

For the extended out-of-domain (OOD) evaluation (§4.3), we augment the registry with two additional experts, Step3-VL-10B [14] and Qwen3.5-9B [41], together with four new Level-1 skills: Embodied Scene Problem Solver, OCR Problem Solver, Diagram Reasoning Skill, and Python Code Generator. The augmented registry contains 9 Level-1 and 24 Level-2 skills in total, and is used without retraining the orchestrator.

Training Data. The orchestrator is trained on 9,200 samples from seven multimodal datasets: ChartQA, Geometry3K, ZwZ-RL-VQA, TallyQA, Slake, MicroVQA, and MSEarthMCQ. The mixture covers the core domains targeted by the default model-skill registry, including chart understanding, geometric reasoning, high-resolution perception, object counting, medical VQA, and scientific reasoning. Detailed dataset statistics are reported in Appendix D.

Benchmarks and Metrics. We evaluate MAESTRO on ten representative multimodal benchmarks. The in-domain set includes chart parsing: ChartQA [32]; geometric reasoning: Geometry3K [28]; microscopic reasoning: MicroVQA [7]; earth-science reasoning: MSEarthMCQ; medical QA: Slake [23]; and object counting: TallyQA [1]. The out-of-domain set includes HRBench-4K/8K [54], VStar [9], and MathVision [52], which test high-resolution perception and advanced multimodal mathematical reasoning. We further evaluate extensibility on four specialized OOD benchmarks: ERQA [18], OCRBench [25], VlmsAreBlind [42], and Humaneval_V [73], which use the augmented registry described above. We also report latency and token consumption to assess efficiency.

Baselines. We evaluate three categories of baselines: Closed-Source Models, including GPT-4o, GPT-5, Gemini-2.5-Flash/Pro; Open-Source & Baselines, including GLM-4.6V, Kimi-K2.5, Qwen3VL-32B, direct answering, and the untrained workflow model; and Think with Images Methods,

- Table 2: Performance on specialized OOD benchmarks. MAESTRO uses the default pool (5 experts, 5 Level-1 skills). MAESTRO* augments the registry with 2 additional experts and 4 new Level-1 skills, without retraining.

Method ERQA OCRBench Vlms. Human. Avg. Closed-Source Models

- GPT-4o 44.8 63.1 48.9 23.3 45.0
- GPT-5 45.8 75.3 66.7 25.3 53.3 Gemini-2.5-Flash 44.3 82.5 68.4 23.7 54.7 Gemini-2.5-Pro 43.0 82.9 74.6 21.7 55.6

Open-Source & Baselines GLM-4.6V 44.0 82.0 69.6 27.7 55.8 Kimi-K2.5 46.0 84.5 77.8 28.5 59.2 Qwen3-VL-32B 49.3 81.6 71.1 5.5 51.9 Direct Answering 47.5 78.5 67.8 4.0 49.5 Untrained Model 41.5 74.8 48.4 2.7 41.9 Best Model 55.0 80.7 73.6 13.3 55.7

Think with Images Methods DeepEyes 35.5 78.5 48.1 4.0 41.5 DeepEyes-v2 42.3 81.0 55.1 1.6 45.0 Thyme 47.3 78.0 48.1 1.2 43.7 VTOOL-R1 36.8 78.3 48.4 4.0 41.9 VTS-V 37.5 77.2 42.1 2.4 39.8 MathCoder-VL 37.3 74.9 42.6 1.2 39.0 Visual-ARFT 37.0 76.3 44.6 2.8 40.2 VisionReasoner 39.3 78.5 48.9 4.0 42.7 PixelReasoner 33.5 79.9 50.9 1.6 41.5 Chain-of-Focus 41.8 80.9 39.4 1.6 40.9

MAESTRO 42.8 79.4 69.1 19.4 52.7 MAESTRO* 52.5 85.2 72.1 28.1 59.5

DeepEyes

Thyme

MAESTRO

VTOOL-R1

DeepEyes-v2

(a) Token Consumption

1400

1200

Tokens↓

1000

800

600

400

(b) Inference Latency

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

- 2
- 3
- 4
- 5
- 6

Latency(s)↓

(c) Accuracy

90

Accuracy(%)↑

80

70

60

50

VStar HRB-4K HRB-8K SlakeMicroVQA TallyQA Avg.

#### Figure 3: Average token consumption, inference latency, and accuracy per benchmark. MAESTRO achieves the best performance and efficiency.

including DeepEyes, DeepEyesV2, Thyme, VTOOL-R1, VTS-V, MathCoder-VL, Visual-ARFT, VisionReasoner, PixelReasoner, and Chain-of-Focus. More details are provided in Appendix D.3.

Implementation Details The orchestrator is initialized from Qwen3-VL-4B-Thinking [4] and optimized with GRPO to handle sparse, high-variance rewards in long-horizon reasoning. For each query, we sample G = 8 trajectories to compute group-relative advantages, and use an asynchronous rollout mechanism to decouple experience collection from gradient updates. The interaction horizon is limited to T = 4 turns per episode. To avoid context overflow, we truncate over-length policy actions and environment observations during rollout. All experiments are based on 4 A100 GPUs.

#### 4.2 Main Results

Table 1 presents a comprehensive performance comparison between MAESTRO and leading closedsource, open-source, and specialized multimodal reasoning models across ten benchmarks.

In-Domain Performance. With a lightweight 4B orchestrator, MAESTRO achieves a leading average accuracy of 70.1%, surpassing powerful closed-source frontiers including GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Performance gains are particularly pronounced in domain-specific tasks. For example, on Geometry3K, MAESTRO reaches 77.4% accuracy, far exceeding GPT-4o (34.1%) and GLM-4.6V (60.4%), demonstrating how the RL-trained policy effectively routes geometric problems to the specialized Geometric Problem Solver skill. On ChartQA, MAESTRO matches the best baseline (86.8%) while maintaining superior performance across all remaining tasks.

Out-of-Domain Generalization. The robustness of MAESTRO is further highlighted by its performance on Out-of-Domain (OOD) datasets. On high-resolution benchmarks, our method achieves 88.0% on VStar and 79.6% on HRBench-4K, outperforming specialized “Think with Images” methods such as DeepEyes (85.6% on VStar) and Thyme (77.0% on HRB-4K). This superiority on unseen distributions confirms that the orchestrator internalizes a generalizable coordination logic rather than memorizing task-specific mappings. By dynamically selecting the optimal model-skill ensembles (e.g., matching Chart-R1 with the Chart Problem Solver), MAESTRO effectively bridges the gap

100

- 1

- 2

- 3

- 4

N=2 N=4 N=5 N=8

3.55

3.50

3.41

3.38

88.0

88.0

85.9

90

3.04

2.97

Accuracy(%)

80.6

2.84

79.8

2.81

79.6

79.6

79.3

79.0

Latency(s)

2.69

2.67

2.65

2.61

75.4

80

74.4

74.4

73.3

71.8

70.5

69.5

2.14

2.14

2.12

66.2

65.9

65.7

70

1.94

1.56

57.9

1.46

1.43

1.42

60

1.38

1.35

1.35

53.0

52.8

52.8

1.10

49.4

50

VStar H-4K H-8K Slake M-QA T-QA

VStar H-4K H-8K Slake M-QA T-QA

Figure 4: Performance (Acc.) and latency (s) as a function of skill pool size N. The RL-based routing consistently leverages additional skills to improve accuracy with sub-linear latency growth.

- Table 3: Performance on realistic agentic benchmarks. BFCL-V4 average is weighted by subset size following the official evaluation protocol [39]. tau2-bench average is the unweighted mean across four domain scenarios.

BFCL-V4 tau2-bench Single-turn

Method

Multi-turn Avg. Retail Airline Telecom Banking Avg. Non-live Live

GPT-5.2 78.29 67.14 43.75 68.58 67.5 80.5 61.6 12.4 55.5 Gemini-2.5-Flash 84.96 74.39 36.25 72.88 50.9 76.0 56.1 9.3 48.1 Claude-Opus-4.5 89.65 76.02 16.12 72.14 79.6 84.0 92.3 24.7 70.2

MAESTRO (Ours) 87.33 82.38 44.62 78.09 85.1 84.0 95.6 26.9 72.9

between general-purpose reasoning and specialized tool invocation, even when encountering unseen data distributions like MathVision.

#### 4.3 Extensibility to Unseen Experts and Skills

To assess the plug-and-play flexibility of MAESTRO, we augment the registry with two additional expert models: Step3-VL-10B for vision-grounded code problems and Qwen3.5-9B for embodied scene reasoning, OCR, and diagram understanding. We also add four new Level-1 skills tailored to ERQA, OCRBench, VlmsAreBlind, and Humaneval_V, all without retraining the orchestrator. We denote this augmented configuration as MAESTRO*, retaining the default setup (5 expert models, 5 Level-1 skills) as the unaugmented baseline.

As shown in Table 2, while closed-source frontiers such as GPT-5 achieve competitive performance through general-purpose reasoning, they lack the fine-grained tool-use optimization inherent in our framework. Notably, the unaugmented MAESTRO already attains a competitive average accuracy of 52.7% using only the default registry of general skills, outperforming all “Think with Images” baselines and remaining comparable to strong closed-source models. This suggests that the default skills capture transferable multimodal reasoning primitives rather than being narrowly tailored to the extended OOD benchmarks. After augmenting the registry with newly introduced experts and skills, MAESTRO further improves from 52.7% to 59.5%, outperforming all baselines including Gemini-2.5-Pro (55.6%) and Kimi-k2.5 (59.2%). Since the orchestrator was never exposed to these experts or skills during training and requires no policy retraining, these results indicate that the learned policy can exploit semantically described new capabilities in a plug-and-play manner, supporting MAESTRO’s extensibility to evolving multimodal expert ecosystems.

#### 4.4 Efficiency and Scalability Analysis

Token Consumption and Latency. We evaluate MAESTRO’s computational efficiency by comparing token consumption and inference latency against representative “Think with Images” methods.

As shown in Figure 3, MAESTRO achieves the lowest average latency (2.88s) and token consumption (648.20 tokens). Unlike iterative “Think with Images” methods that rely on redundant image zooming and repetitive prompting, our hierarchical routing allows the orchestrator to immediately identify the most suitable skill-expert pair, avoiding unnecessary intermediate calls. While some specialized mathematical solvers (e.g., VTOOL-R1) show slightly lower token usage in Geometry3K, our

framework’s ability to balance speed and accuracy across all ten benchmarks proves its robustness for real-world deployment. Detailed results are provided in Table 6.

Scaling with Skill Pool Size. We investigate how performance and latency evolve as the skill pool grows from N=2 to N=8, across four configurations: N=2 (Chart, Geometric); N=4 (+Counting, Science); N=5 (+Perception); N=8 (+Embodied Scene, OCR, Python Code Generator). As shown in Figure 4, expanding from N=2 to N=8 raises average accuracy from 60.7% to 66.5% (+5.8%), with pronounced gains on domain-specific benchmarks: VStar improves by 7.4% (80.6%→88.0%) and Slake by 7.8% (57.9%→65.7%). Crucially, latency grows sub-linearly relative to accuracy, indicating that the RL-trained orchestrator learns to invoke richer expert combinations only when necessary. Detailed results are in Appendix E.2.

#### 4.5 Discussion on Realistic Agentic Benchmarks

To assess MAESTRO beyond static VQA, we evaluate on two realistic benchmarks: BFCL-V4 [39] and tau2-bench [6]. As shown in Table 3, on BFCL-V4, it reaches an average of 78.09, outperforming GPT-5.2 (68.58), Gemini-2.5-Flash (72.88), and Claude-Opus-4.5 (72.14). Gains are most pronounced on the Live split (82.38 vs. 76.02) and the Multi-turn split (44.62 vs. 43.75), which demand dynamic adaptation to evolving API schemas and stateful reasoning across turns. On tau2-bench, MAESTRO achieves an average of 72.9 across four domain-specific customer service scenarios, surpassing Claude-Opus-4.5 (70.2), GPT-5.2 (55.5), and Gemini-2.5-Flash (48.1). These results confirm that the orchestration policy learned on static VQA transfers effectively to dynamic, multi-turn, tool-use settings that closely mirror real-world agentic deployments.

#### 4.6 Ablation Study

Component Ablation. We compare MAESTRO against three variants: (1) w/o Skill Pool: full model pool without the hierarchical skill library; (2) w/o Model Pool: hierarchical skills paired with the base 4B model only; and (3) w/o Both: the base 4B model answering directly without any augmentation. Figure 5(a) reveals three key insights: First, removing the Skill Pool causes a −2.7% average drop, confirming that structured hierarchical prompting yields consistent gains even when expert models are present. Second, removing the Model Pool leads to a larger −12.1% decline, with particularly severe degradation on reasoning-intensive benchmarks (MathVision: 43.4%→27.6%; Geometry3K: 77.4%→22.3%), underscoring that the base 4B model alone cannot substitute for domain-specific expert capacity. Third, removing both components reduces average accuracy to 55.8%, yet the remaining gap above direct answering confirms that the skill library retains utility even without expert-model routing. Overall, the two components are complementary: specialized models supply the domain-specific “brain” for reasoning, while hierarchical skills serve as the “eyes” and “hands” for precision visual parsing and tool execution.

Reward Design Ablation. As shown in Figure 5(b), removing rfmt causes a -13.1% average drop: without structural constraints, the policy generates malformed action sequences that break the multi-turn communication protocol. Removing rans leads to a −8.8% decline, confirming that outcome supervision is the primary signal for routing quality. The two rewards thus play complementary roles: rfmt ensures communication reliability, while rans drives task performance.

Full w/o Skill Pool w/o Model Pool w/o Both w/o rfmt w/o rans

90

|88.0|
|---|
|77.0<br><br>79.6 79.8<br><br>|
|76.8<br><br>73.8 73.8<br><br>70.8<br><br>74.4<br><br>66.2|
|65.5<br><br>67.5<br><br>59.0<br><br>|
|61.3<br><br>57.9<br><br>49.8<br><br>53.0|
|47.2|

80

Performance

| |
|---|

70

60

50

| |
|---|

40

VStar H-4K H-8K Slake M-QA T-QA

VStar H-4K H-8K Slake M-QA T-QA

(a) Component Ablation

(b) Reward Ablation

Figure 5: Ablation study. (a) Component ablation: the model pool and skill library each contribute independently, and their combination is essential for peak performance. (b) Reward ablation: both the format reward rfmt and the outcome reward rans are necessary for stable multi-turn orchestration.

### 5 Conclusion

We present MAESTRO, an RL-driven framework that reframes heterogeneous model-skill orchestration as a sequential decision-making problem, decoupling coordination logic from underlying model parameters. Evaluated across ten multimodal benchmarks, MAESTRO outperforms leading closed-source models, uncovers non-trivial model-skill synergies, and generalizes its routing logic to out-of-domain settings, all while maintaining low inference latency. These results suggest that intelligent orchestration is a high-leverage alternative to scaling model size. Future work will explore self-evolving skill registries and online policy adaptation to broader, open-domain environments.

### References

- [1] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019.
- [2] Anthropic. Claude code overview. https://code.claude.com/docs, 2025.
- [3] Lei Bai, Zhongrui Cai, Yuhang Cao, Maosong Cao, Weihan Cao, Chiyu Chen, Haojiong Chen, Kai Chen, Pengcheng Chen, Ying Chen, et al. Intern-s1: A scientific multimodal foundation model. arXiv preprint arXiv:2508.15763, 2025.
- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.
- [5] Tianyi Bai, Zengjie Hu, Fupeng Sun, Jiantao Qiu, Yizhen Jiang, Guangxin He, Bohan Zeng, Conghui He, Binhang Yuan, and Wentao Zhang. Multi-step visual reasoning with visual tokens scaling and verification. arXiv preprint arXiv:2506.07235, 2025.
- [6] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.
- [7] James Burgess, Jeffrey J Nirschl, Laura Bravo-Sánchez, Alejandro Lozano, Sanket Rajan Gupte, Jesus G Galaz-Montoya, Yuhui Zhang, Yuchang Su, Disha Bhowmik, Zachary Coman, et al. Microvqa: A multimodal reasoning benchmark for microscopy-based scientific research. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19552–19564, 2025.
- [8] Lei Chen, Xuanle Zhao, Zhixiong Zeng, Jing Huang, Yufeng Zhong, and Lin Ma. Chart-r1: Chain-ofthought supervision and reinforcement for advanced chart reasoner. arXiv preprint arXiv:2507.15509, 2025.
- [9] Zixu Cheng, Jian Hu, Ziquan Liu, Chenyang Si, Wei Li, and Shaogang Gong. V-star: Benchmarking video-llms on video spatio-temporal reasoning. arXiv preprint arXiv:2503.11495, 2025.
- [10] Yu Du, Fangyun Wei, and Hongyang Zhang. Anytool: Self-reflective, hierarchical agents for large-scale api calls. arXiv preprint arXiv:2402.04253, 2024.
- [11] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.
- [12] Lang Feng, Fuchao Yang, Feng Chen, Xin Cheng, Haiyang Xu, Zhenglin Wan, Ming Yan, and Bo An. Agentocr: Reimagining agent history via optical self-compression. arXiv preprint arXiv:2601.04786, 2026.
- [13] Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.
- [14] Ailin Huang, Chengyuan Yao, Chunrui Han, Fanqi Wan, Hangyu Guo, Haoran Lv, Hongyu Zhou, Jia Wang, Jian Zhou, Jianjian Sun, et al. Step3-vl-10b technical report. arXiv preprint arXiv:2601.09668, 2026.
- [15] Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, et al. Metatool benchmark for large language models: Deciding whether to use tools and which to use. arXiv preprint arXiv:2310.03128, 2023.
- [16] Chengyou Jia, Minnan Luo, Zhuohang Dang, Qiushi Sun, Fangzhi Xu, Junlin Hu, Tianbao Xie, and Zhiyong Wu. Agentstore: Scalable integration of heterogeneous agents as specialized generalist computer assistant. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8908–8934,

- [17] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [18] Anastasia Kirillova., Eugene Lyapustin., Anastasia Antsiferova., and Dmitry Vatolin. Erqa: Edgerestoration quality assessment for video super-resolution. In Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP,, pages 315–322. INSTICC, SciTePress, 2022.
- [19] Samuel Kotz and Norman L. Johnson, editors. Breakthroughs in Statistics: Methodology and Distribution. Springer New York, New York, NY, 1992.
- [20] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.
- [21] Yanda Li, Chi Zhang, Wenjia Jiang, Wanqi Yang, Bin Fu, Pei Cheng, Xin Chen, Ling Chen, and Yunchao Wei. Appagent v2: Advanced agent for flexible mobile interactions. arXiv preprint arXiv:2408.11824, 2024.
- [22] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.
- [23] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. Slake: A semantically-labeled knowledge-enhanced dataset for medical visual question answering. In 2021 IEEE 18th international symposium on biomedical imaging (ISBI), pages 1650–1654. IEEE, 2021.
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [25] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12), December 2024. ISSN 1869-1919.
- [26] Yuqi Liu, Tianyuan Qu, Zhisheng Zhong, Bohao Peng, Shu Liu, Bei Yu, and Jiaya Jia. Visionreasoner: Unified visual perception and reasoning via reinforcement learning. arXiv e-prints, pages arXiv–2505, 2025.
- [27] Ziyu Liu, Yuhang Zang, Yushan Zou, Zijian Liang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual agentic reinforcement fine-tuning. arXiv preprint arXiv:2505.14246, 2025.
- [28] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. Intergps: Interpretable geometry problem solving with formal language and symbolic reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6774–6786, 2021.
- [29] Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Skill0: In-context agentic reinforcement learning for skill internalization. arXiv preprint arXiv:2604.02268, 2026.
- [30] Zhiyuan Ma, Zhenya Huang, Jiayu Liu, Minmao Wang, Hongke Zhao, and Xin Li. Automated creation of reusable and diverse toolsets for enhancing llm reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 24821–24830, 2025.
- [31] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems, 36:46534–46594, 2023.
- [32] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263–2279, 2022.
- [33] OpenAI. Codex — ai coding partner from openai. https://openai.com/codex/, 2025.
- [34] OpenClaw. Skills - openclaw. https://docs.openclaw.ai/tools/skills, 2026.
- [35] Yixin Ou, Yujie Luo, Jingsheng Zheng, Lanning Wei, Zhuoyun Yu, Shuofei Qiao, Jintian Zhang, Da Zheng, Yuren Mao, Yunjun Gao, et al. Automind: Adaptive knowledgeable agent for automated data science. arXiv preprint arXiv:2506.10974, 2025.

- [36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [37] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.
- [38] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information Processing Systems, 37:126544–126565, 2024.
- [39] Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025.
- [40] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789, 2023.
- [41] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.
- [42] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision, pages 18–34, 2024.
- [43] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551, 2023.
- [44] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [45] Andrew Sellergren, Chufan Gao, Fereshteh Mahvar, Timo Kohlberger, Fayaz Jamil, Madeleine Traverse, Alberto Tono, Bashir Sadjad, Lin Yang, Charles Lau, et al. Medgemma 1.5 technical report. arXiv preprint arXiv:2604.05081, 2026.
- [46] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems, 36:38154–38180, 2023.
- [47] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.
- [48] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11888– 11898, 2023.
- [49] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301, 2023.
- [50] Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, et al. Skillx: Automatically constructing skill knowledge bases for agents. arXiv preprint arXiv:2604.04804, 2026.
- [51] Jiayu Wang, Yifei Ming, Zixuan Ke, Shafiq Joty, Aws Albarghouthi, and Frederic Sala. Skillorchestra: Learning to route agents via skill transfer. arXiv preprint arXiv:2602.19672, 2026.
- [52] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. Advances in Neural Information Processing Systems, 37:95095–95169, 2024.
- [53] Ke Wang, Junting Pan, Linda Wei, Aojun Zhou, Weikang Shi, Zimu Lu, Han Xiao, Yunqiao Yang, Houxing Ren, Mingjie Zhan, et al. Mathcoder-vl: Bridging vision and code for enhanced multimodal mathematical reasoning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2505–2534,

- [54] Wenbin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 7907–7915, 2025.
- [55] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023.
- [56] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European conference on computer vision, pages 396–416. Springer, 2024.
- [57] Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, and Daniel Fried. Inducing programmatic skills for agentic tasks. arXiv preprint arXiv:2504.06821, 2025.
- [58] Lai Wei, Liangbo He, Jun Lan, Lingzhong Dong, Yutong Cai, Siyuan Li, Huijia Zhu, Weiqiang Wang, Linghe Kong, Yue Wang, et al. Zooming without zooming: Region-to-image distillation for fine-grained multimodal perception. arXiv preprint arXiv:2602.11858, 2026.
- [59] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.
- [60] Jinyang Wu, Mingkuan Feng, Shuai Zhang, Feihu Che, Zengqi Wen, Chonghua Liao, and Jianhua Tao. Beyond examples: High-level automated reasoning paradigm in in-context learning via mcts. arXiv preprint arXiv:2411.18478, 2024.
- [61] Jinyang Wu, Shuo Yang, Changpeng Yang, Yuhao Shen, Shuai Zhang, Zhengqi Wen, and Jianhua Tao. Spark: Strategic policy-aware exploration via dynamic branching for long-horizon agentic learning. arXiv preprint arXiv:2601.20209, 2026.
- [62] Jinyang Wu, Guocheng Zhai, Ruihan Jin, Jiahao Yuan, Yuhao Shen, Shuai Zhang, Zhengqi Wen, and Jianhua Tao. Atlas: Orchestrating heterogeneous models and tools for multi-domain complex reasoning. arXiv preprint arXiv:2601.03872, 2026.
- [63] Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, Chengxiang Zhai, and Klara Nahrstedt. Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255, 2025.
- [64] Menglin Xia, Xuchao Zhang, Shantanu Dixit, Paramaguru Harimurugan, Rujia Wang, Victor Ruhle, Robert Sim, Chetan Bansal, and Saravan Rajmohan. Memora: A harmonic memory representation balancing abstraction and specificity. arXiv preprint arXiv:2602.03315, 2026.
- [65] Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.
- [66] Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, et al. Autoskill: Experience-driven lifelong learning via skill self-evolution. arXiv preprint arXiv:2603.01145, 2026.
- [67] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.
- [68] Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. Worldmm: Dynamic multimodal memory agent for long video reasoning. arXiv preprint arXiv:2512.02425, 2025.
- [69] Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R Fung, Hao Peng, and Heng Ji. Craft: Customizing llms by creating and retrieving from specialized toolsets. arXiv preprint arXiv:2309.17428, 2023.
- [70] Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Kan Ren, Dongsheng Li, and Deqing Yang. Easytool: Enhancing llm-based agents with concise tool instruction. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 951–972, 2025.
- [71] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026.

- [72] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.
- [73] Fengji Zhang, Linquan Wu, Huiyu Bai, Guancheng Lin, Xiao Li, Xiao Yu, Yue Wang, Bei Chen, and Jacky Keung. Humaneval-v: benchmarking high-level visual reasoning with complex diagrams in coding tasks. arXiv preprint arXiv:2410.12381, 2024.
- [74] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, et al. Adaptive chain-of-focus reasoning via dynamic visual search and zooming for efficient vlms. arXiv preprint arXiv:2505.15436, 2025.
- [75] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.
- [76] Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. RLVMR: Reinforcement learning with verifiable meta-reasoning rewards for robust long-horizon agents. In The Fourteenth International Conference on Learning Representations, 2026.
- [77] Xiangyu Zhao, Wanghan Xu, Bo Liu, Yuhao Zhou, Fenghua Ling, Ben Fei, Xiaoyu Yue, Lei Bai, Wenlong Zhang, and Xiao-Ming Wu. Msearth: A multimodal scientific dataset and benchmark for phenomena uncovering in earth science. arXiv preprint arXiv:2505.20740, 2025.
- [78] YanZhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuai Zhu, Yong Wu, Tianze Xu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, et al. Skillrouter: Skill routing for llm agents at scale. arXiv e-prints, pages arXiv–2603, 2026.
- [79] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

### Contents

- A Theoretical Analysis 16

- A.1 Problem Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 Hierarchical Action-Space Compression . . . . . . . . . . . . . . . . . . . . . . . 16
- A.3 Model-Skill Compatibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.4 Extensibility . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.5 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- B Algorithmic Details 18
- C Detailed Hierarchical Skill Taxonomy 19

- C.1 Default Skill Configuration (S1–S5) . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Extended Skill Configuration (S6–S9) . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.3 Hierarchical Execution Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D Detailed Experimental Details 21

- D.1 Training Data Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.2 Evaluation Benchmark Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.3 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- D.4 Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- E More Results and Analysis 26

- E.1 Efficiency and Scalability Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.2 Scaling with Skill Pool Size . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.3 Test-Time Scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.4 Additional Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- E.5 Statistical Significance Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . 28

- F Case Study 28
- G Additional Discussion 29

- G.1 Limitation and Failure Case Analysis . . . . . . . . . . . . . . . . . . . . . . . . . 29
- G.2 Clarification on Model Scale and Computational Cost . . . . . . . . . . . . . . . . 29
- G.3 Skill Design Cost and Engineering Basis . . . . . . . . . . . . . . . . . . . . . . . 30
- G.4 Sensitivity to Skill Descriptions . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- G.5 Detailed Comparison with Concurrent Works . . . . . . . . . . . . . . . . . . . . 31
- G.6 Why Reinforcement Learning over Retrieval-Based Routing . . . . . . . . . . . . 32
- G.7 Emergent Behavior During Training . . . . . . . . . . . . . . . . . . . . . . . . . 32
- G.8 Future Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- H Broader Impact 33

### A Theoretical Analysis

This section provides an informal theoretical explanation for why MAESTRO’s RL-driven model-skill orchestration (Algorithm 1) over a hierarchical model-skill registry improves both performance and efficiency. The goal is not to give formal proofs, but to explain the main mechanisms behind the empirical results: action-space compression, model-skill compatibility, and plug-and-play extensibility.

#### A.1 Problem Setup

Consider a multimodal task with query-context pair (q,x). At step t, the orchestrator maintains the history

ct = (q,x,a1,o1,...,at−1,ot−1), and selects a compositional search action

asearcht = (mt,st,zt),

where mt ∈ M is a frozen expert model, st ∈ K is a skill, and zt is the dispatched query. The policy maximizes

J(θ) = Eτ∼π

[R(τ)], R(τ) = rans + rfmt. For each context c, define the utility of a model-skill pair as

θ

Uc(m,s) = E[R(τ) | c,asearch = (m,s,z)]. The ideal routing decision is

(m∗,s∗) = arg max

Uc(m,s).

m∈M,s∈K

Since this utility is unobserved and combinatorial, MAESTRO trains a lightweight policy model to infer useful model-skill pairs from context and feedback.

###### A.2 Hierarchical Action-Space Compression Flat routing selects directly from all model-skill pairs, giving

|Aflat| = |M| · |K|.

MAESTRO instead exposes only coarse Level-1 skills to the orchestrator and delegates fine Level-2 routing to skill-local logic:

|Ahier| = |M| · |K1|. Thus the direct search space is compressed by

|Aflat| |Ahier|

= |K| |K1|

.

In sparse-reward training, the number of samples required to identify good actions grows at least linearly with the effective action space. Under fixed target accuracy,

N(M,K) ∝ |A|. Therefore, flat routing requires a budget proportional to

Nflat(M,K) ∝ |M||K|, whereas hierarchical routing reduces it to

Nhier(M,K1) ∝ |M||K1|.

The benefit is not only a smaller action space: Level-1 skills also correspond to semantically meaningful task types, making routing easier to learn and reducing redundant tool calls.

Empirical Connection. This explains MAESTRO’s low latency and token consumption: by selecting a suitable model-skill pair early, the orchestrator avoids redundant visual zooming, repeated prompting, and trial-and-error tool calls. The skill-pool ablation further supports this view, since removing the hierarchical skills causes a consistent performance drop even when expert models remain available.

- A.3 Model-Skill Compatibility For a fixed context c, compare four utilities:

U0(c) = Uc(∅,∅), UM(c;m) = Uc(m,∅),

UK(c;s) = Uc(∅,s), UMK(c;m,s) = Uc(m,s). The model-only and skill-only gains are

∆M(c;m) = UM(c;m) − U0(c), ∆K(c;s) = UK(c;s) − U0(c).

If model and skill effects were independent, the joint gain would be their sum. MAESTRO instead assumes that useful model-skill pairs can have positive compatibility. Define

Cc(m,s) = [UMK(c;m,s) − U0(c)] − ∆M(c;m) − ∆K(c;s). Equivalently,

Cc(m,s) = UMK(c;m,s) − UM(c;m) − UK(c;s) + U0(c). Hence the joint gain decomposes as

UMK(c;m,s) − U0(c) = ∆M(c;m) + ∆K(c;s) + Cc(m,s).

When Cc(m,s) > 0, the model-skill pair provides value beyond choosing a strong model and a relevant skill independently. This explains why MAESTRO learns a policy over joint pairs (m,s) rather than performing separate model retrieval and skill retrieval.

Empirical Connection. The component ablation matches this interpretation: removing the skill pool hurts performance even with the expert model pool, while removing the model pool causes a larger drop, especially on reasoning-intensive tasks such as MathVision and Geometry3K. The full system works best because it optimizes joint assignments instead of treating model choice and skill choice as independent retrieval problems.

- A.4 Extensibility Suppose the registry expands from (M,K) to (M′,K′) without retraining, where

##### M′ = M ∪ Mnew, K′ = K ∪ Knew.

Mnew and Knew denote newly added expert models and skills. We do not assume that the trained policy can perfectly use these new entries. Instead, we first consider how registry expansion changes the oracle upper bound. For context c, define the oracle utility under registry (M,K) as

Uc∗(M,K) = max

Uc(m,s).

m∈M, s∈K

Since (M,K) is a subset of (M′,K′), the oracle utility after expansion cannot decrease:

Uc(m,s) = Uc∗(M,K).

Uc∗(M′,K′) = max

Uc(m,s) ≥ max

m∈M′, s∈K′

m∈M, s∈K

This only shows that newly added models and skills enlarge the candidate space and therefore improve or preserve the theoretical oracle upper bound. It does not guarantee that the learned orchestrator will select the new capabilities. To capture the gap between the learned policy and the oracle, let Regretθ(c;M,K) denote the routing regret of policy πθ under registry (M,K):

##### Regretθ(c;M,K) = Uc∗(M,K) − Uc(πθ;M,K),

where Uc(πθ;M,K) is the expected utility actually obtained by πθ under registry (M,K). The practical gain after expansion can then be decomposed as

Uc(πθ;M′,K′) − Uc(πθ;M,K)

= [Uc∗(M′,K′) − Uc∗(M,K)] − [Regretθ(c;M′,K′) − Regretθ(c;M,K)].

This decomposition shows that extensibility is not an unconditional guarantee. It depends on whether the oracle gain introduced by new experts and skills is larger than the additional routing regret caused by the expanded registry. The semantic action interface of MAESTRO remains important because model descriptions, skill names, and skill documents give the policy a basis for identifying new entries; nevertheless, the more conservative theoretical claim is that practical improvement depends on the balance between oracle gain and extra routing regret.

Algorithm 1 MAESTRO: RL-driven Model-Skill Orchestration Input: Multimodal query qj, visual context xj, orchestrator policy πθ, pool M, skill library K, max

steps Tmax, parameters G,ε;

Output: Response yj and trajectory τ;

- // Step 1: Initialization

- 1: c0 ← {qj,xj},τ ← ∅

// Step 2: Iterative Reasoning Loop

- 2: for t = 0 to Tmax − 1 do
- 3: at ∼ πθ(· | ct) ▷ Sample action at ∈ {think,search,answer}
- 4: if at is think then
- 5: ot ← πθ.Reasoning(ct) ▷ Internal logic generation
- 6: ct+1 ← Concat(ct,at,ot)
- 7: else if at is search then
- 8: Parse at as (mt,st,zt) where mt ∈ M,st ∈ K
- 9: ot ← Execute(mt,st,zt)
- 10: octxt ← <information> + ot + </information> ▷ Context transition
- 11: ct+1 ← Concat(ct,at,octxt )
- 12: else if at is answer then
- 13: yj ← ExtractAnswer(at)
- 14: τ ← τ ∪ {(ct,at)}
- 15: break
- 16: end if
- 17: τ ← τ ∪ {(ct,at,ot)}
- 18: end for

// Step 3: Policy Optimization (Training Mode)

- 19: if training_mode then
- 20: Sample a group of G trajectories {τ1,...,τG} for query qj
- 21: R(τ) ← rans + rfmt
- 22: Ai ← (Ri − R¯)/(σR + ϵ)
- 23: LGRPO(θ) ← −G1 Gi=1 min(ρi(θ)Ai,clip(ρi(θ),1 − ε,1 + ε)Ai)

- 24: Update θ via GRPO objective: ∇θLGRPO(θ)
- 25: end if
- 26: return (yj,τ)

Empirical Connection. The extended OOD evaluation can be interpreted through this decomposition: after adding new expert models and Level-1 skills to the registry, MAESTRO improves on specialized OOD benchmarks without retraining the orchestrator. This indicates that, in these tasks, the oracle gain from the new capabilities exceeds the additional routing regret introduced by the expanded registry. The result supports practical plug-and-play capability, without claiming that the trained policy must generalize to arbitrary new models or skills.

#### A.5 Summary

MAESTRO’s effectiveness can be understood through three mechanisms. First, the hierarchical skill registry compresses the action space from |M||K| to |M||K1|, explaining its efficiency gains. Second, joint model-skill routing captures compatibility gains that independent model or skill selection would miss, matching the component ablation results. Third, registry expansion improves or preserves the oracle performance upper bound, while practical extension gains depend on whether the oracle gain from new capabilities exceeds the additional routing regret; the OOD extension experiments indicate that this condition holds in the evaluated setting.

### B Algorithmic Details

We provide the pseudo-code of MAESTRO in Algorithm 1. The procedure summarizes both inferencetime orchestration and training-time policy optimization. At each step, the orchestrator samples an action conditioned on the current context, either performing internal reasoning, invoking a selected

model-skill pair, or terminating with a final answer. During training, multiple trajectories are sampled for each query to compute group-relative advantages, and the policy is updated with the GRPO objective.

### C Detailed Hierarchical Skill Taxonomy

MAESTRO employs a two-tier hierarchical skill library consisting of 9 Level-1 skills and 24 Level-2 skills in total. The first five Level-1 skills (S1–S5) form the default configuration used in the main experiments, while the remaining four (S6–S9) are introduced only in the extended out-of-domain evaluation (Section 4.3). This hierarchy minimizes the action space of the 4B orchestrator while ensuring expert-level precision through domain-specific sub-routines. Detailed prompts are provided in Figures 15–23.

#### C.1 Default Skill Configuration (S1–S5)

- S1: Geometric Problem Solver Dedicated to resolving complex Euclidean geometry tasks.

• S1.1: Structural Geometric Analysis. Extracts structured primitives (points, segments, angles, circles) and annotations from the image. It employs ImageCaption for global context and OCR for textual metadata, then cross-references model-internal reasoning with tool-derived outputs to form a consistent geometric representation before executing step-bystep deduction.

- S2: Chart Problem Solver Analyzes diverse data visualizations by routing to three Level-2 sub-solvers based on chart type.

- • S2.1: Bar Chart Solver. Uses OCR to parse titles, axes, and legends, then performs comparative operations (sorting, difference calculation, trend estimation) based on bar heights or lengths.
- • S2.2: Line Chart Solver. Distinguishes data series via line styles or colors, correlates X-axis positions with Y-axis scales, and identifies critical inflections and trend shifts.
- • S2.3: Pie Chart Solver. Extracts sector labels and percentage text to establish part-to-whole relationships, supporting total-sum conversions and relative size comparisons.

- S3: Counting Problem Solver Provides robust object enumeration in cluttered visual environments.

• S3.1: Precision Counter. Integrates a Detection tool for bounding box generation and DeepEyes-7B for localized attention. It catalogs targets with approximate spatial coordinates to prevent double-counting and reduce omissions of occluded or partially visible objects.

- S4: Perception Problem Solver Handles tasks requiring fine-grained visual discrimination via two sub-skills.

- • S4.1: Color Perception. Uses DeepEyes-7B to isolate regions of interest and magnify color-relevant areas. It distinguishes similar hues and neutralizes interference from shadows, reflections, or low-saturation conditions.
- • S4.2: Relative Position and General Perception. Magnifies micro-structures and critical spatial interfaces, evaluating topological relationships (e.g., above/below, front/back) by concurrently processing original and zoomed views.

- S5: Science Problem Solver Tailored for tasks involving experimental schematics and scientific imagery.

• S5.1: Scientific Reasoning. Combines ImageCaption, OCR, and DeepEyes-7B to parse experimental diagrams. It fuses visual and textual evidence to derive scientifically rigorous conclusions.

#### C.2 Extended Skill Configuration (S6–S9)

The following four Level-1 skills are introduced exclusively for the extended OOD evaluation. No retraining of the orchestrator is required; the skills are plugged into the existing registry directly.

- S6: Embodied Scene Problem Solver Addresses robotic manipulation and interactive visual reasoning through five Level-2 skills.

- • S6.1: Trajectory Outcome Skill. Analyzes motion cues (arrows, candidate paths) and crops focal interaction areas to reason about the terminal state of a specified trajectory, rather than describing intermediate steps.
- • S6.2: Action Adjustment Skill. Evaluates pose, height, or angle deviations between the current and goal states, selecting the minimal corrective action or rotation required for task success.
- • S6.3: Spatial Mechanics Skill. Establishes a reference frame to judge spatial relationships (e.g., left/right, inside/outside) and infers mechanism motion (rotation, translation, linkage) from structural contact constraints.
- • S6.4: Pointing and Part Localization Skill. Compares candidate points or arrows against semantic boundaries extracted via OCR and captioning to accurately identify the intended functional component.
- • S6.5: Multi-view Correspondence Skill. Resolves cross-view consistency and task-state progression via joint multi-view inputs and bounding-box alignment. It identifies stable anchors across perspectives or time steps to judge the agent’s progress toward the goal state.

- S7: OCR Problem Solver Designed for text-dense tasks in the style of OCRBench. It routes queries to five Level-2 sub-skills based on the OCR task type.

- • S7.1: Text Recognition. Treats the task as faithful transcription, focusing on exact character sequences while preserving case and disambiguating visually similar characters (e.g., O/0, I/1, S/5, B/8).
- • S7.2: Key Information Extraction. Identifies the target field type (e.g., total amount, date, company name) and uses OCR, detection, and local crops to separate field labels from values in structured documents such as receipts and invoices.
- • S7.3: Scene Text QA. Localizes the real-world object referenced in the question (e.g., signboard, label) and isolates its text from background distractors before answering.
- • S7.4: Document and Chart QA. Determines whether the input is a document, table, chart, or calendar, then applies appropriate parsing strategies (axis/header matching for charts; cell/title matching for documents).
- • S7.5: Formula Recognition. Recovers two-dimensional mathematical expressions by

attending to fraction bars, superscripts, radicals, and Greek letters, outputting valid LATEX rather than interpreting the formula’s meaning.

- S8: Diagram Reasoning Skill Handles synthetic diagram tasks in the style of VlmsAreBlind, routing via question keywords to five Level-2 sub-skills.

- • S8.1: Circle Contact and Overlap Judge. Determines whether two specified circles are separated, tangent, or overlapping by examining boundary contact and shared area, outputting Yes or No.
- • S8.2: Intersection and Route Counting. Distinguishes between intersection-counting tasks (true red-blue crossings only) and route-counting tasks (complete monochromatic paths from start to end).
- • S8.3: Grid Structure Parsing. Ignores cell content and counts rows and columns solely from external borders and internal dividers, cross-validating with the total visible cell count.
- • S8.4: Highlighted Character Recognition. Localizes the circled or ellipse-highlighted region and reads the single character at its center, strictly preserving case.
- • S8.5: Geometric Shape Counting. Counts fully closed instances of the target shape using a stable scan order to avoid double-counting overlapping contours or nested figures.

- S9: Python Code Generator Generates executable Python code from visual examples and function signatures in the style of Humaneval_V.

• S9.1: Code Problem Solver. Extracts the function signature from the prompt, derives a concrete test case as an assert statement from the visual example, and generates a complete implementation. If execution fails, the error message and failing case are fed back for iterative repair within a fixed number of rounds, producing a verified runnable solution.

#### C.3 Hierarchical Execution Protocol

The orchestration follows a non-invasive, two-stage routing protocol. First, the 4B policy model selects a coarse-grained Level-1 skill and the corresponding expert model. Second, the Level-2 sub-routine is invoked either through keyword-based activation or through classification by the expert model. This design keeps the orchestrator focused on strategic resource allocation while Level-2 skills provide the execution depth required for each domain.

### D Detailed Experimental Details

#### D.1 Training Data Statistics

Table 4 summarizes the composition of the training mixture used to optimize the MAESTRO orchestrator. The 9,200 samples span seven datasets covering the five task domains of the default skill configuration. No data from the extended OOD benchmarks (ERQA, OCRBench, VlmsAreBlind, Humaneval_V) is included during training, ensuring a clean separation between the training distribution and the out-of-domain evaluation. And the system prompt is shown in Figure 7.

We explicitly verify that there is no sample-level overlap between the training mixture and any evaluation benchmark used in this paper. Although several benchmark names appear in both the training and evaluation splits (e.g., ChartQA, Geometry3K, TallyQA, Slake, MicroVQA, MSEarthMCQ), training samples are drawn exclusively from the official training splits of each dataset, while evaluation is conducted on the corresponding held-out test splits. All out-of-domain benchmarks (HRBench-4K/8K, VStar, MathVision, ERQA, OCRBench, VlmsAreBlind, Humaneval_V) are entirely absent from the training mixture, ensuring a clean zero-shot evaluation on these splits. No data from the extended out-of-domain benchmarks is included during training.

- Table 4: Composition of the training mixture used to optimize the MAESTRO orchestrator. Task domain indicates the primary capability targeted by each dataset.

Dataset Task Domain Samples Split (%)

ChartQA [32] Chart Understanding 1,814 19.7 Geometry3K [28] Mathematical Reasoning 1,248 13.6 ZwZ-RL-VQA [58] High-Resolution Perception 2,000 21.7 TallyQA [1] Object Counting 1,664 18.1 Slake [23] Medical VQA 816 8.9 MicroVQA [7] Scientific Reasoning 316 3.4 MSEarthMCQ [77] Scientific Reasoning 1,342 14.6

Total 9,200 100.0

#### D.2 Evaluation Benchmark Statistics

Below we provide detailed descriptions of each benchmark used in our evaluation, grouped by task category. The full list of in-domain (ID) and out-of-domain (OOD) splits is summarized in Table 5.

#### Chart Understanding.

• ChartQA [32]: A benchmark of 2,500 test questions covering bar charts, line charts, and pie charts drawn from real-world sources. Questions require both visual data extraction and multi-step

numerical reasoning (e.g., trend comparison, ratio calculation), making it a comprehensive test of chart comprehension.

#### Mathematical and Geometric Reasoning.

- • Geometry3K [28]: A dataset of 3,002 plane geometry problems paired with formal diagrams and multi-choice answers. Each problem requires interpreting geometric figures, applying relevant theorems, and executing step-by-step deductive reasoning, posing substantial challenges for both visual perception and logical inference.
- • MathVision [52]: An OOD benchmark of 3,040 multi-modal math problems spanning 16 subjects and 5 difficulty levels, sourced from real mathematical competitions. It evaluates advanced visual-mathematical reasoning that goes well beyond standard arithmetic, serving as a rigorous test of generalization.

#### Scientific Reasoning.

- • MicroVQA [7]: A benchmark targeting scientific visual question answering in microscopy and biomedical imaging. Questions require domain-specific knowledge alongside fine-grained visual analysis of experimental imagery.
- • MSEarthMCQ: A multiple-choice benchmark focused on earth science and remote sensing, requiring models to integrate scientific knowledge with satellite or aerial imagery to answer domain-specific questions.

#### Medical Visual QA.

• Slake [23]: A bilingual (English and Chinese) medical VQA dataset containing 14,000 questionanswer pairs over radiology images. Questions span pathology identification, organ recognition, and clinical attribute reasoning, demanding specialized medical knowledge combined with visual understanding.

#### Object Counting.

• TallyQA [1]: A large-scale counting benchmark with over 287,000 question-answer pairs covering simple and complex counting scenarios. Complex questions involve relational reasoning (e.g., counting objects satisfying multiple spatial or attribute conditions), requiring robust object localization and enumeration under occlusion and clutter.

#### High-Resolution Visual Perception.

- • HRBench (4K / 8K) [54]: An OOD benchmark specifically designed for ultra-high-resolution image understanding at 4K and 8K resolutions. Tasks include fine-grained object recognition, attribute identification, and spatial reasoning over images that far exceed the resolution typically encountered in standard VQA benchmarks.
- • VStar [9]: An OOD benchmark probing visual search and focus capabilities in high-resolution scenes. It requires models to locate and reason about small, semantically critical regions within large images, testing the ability to ground attention on task-relevant details.

#### Embodied Scene Reasoning.

• ERQA [18]: An OOD benchmark for embodied reasoning question answering in robotic manipulation scenarios. Questions involve trajectory prediction, action adjustment, spatial relationship judgment, and multi-view correspondence, requiring integrated perception and physical reasoning over scene imagery.

#### OCR and Text-Rich Understanding.

• OCRBench [25]: An OOD comprehensive OCR evaluation suite covering text recognition, key information extraction, scene text QA, document and chart QA, and formula recognition. It assesses a model’s ability to read, localize, and reason over text-dense images across diverse real-world document types.

#### Synthetic Diagram Reasoning.

• VlmsAreBlind [42]: An OOD benchmark composed of synthetic visual puzzles designed to expose failures in low-level visual perception. Tasks include circle overlap judgment, line intersection counting, grid structure parsing, highlighted character recognition, and geometric shape counting, targeting capabilities that are often overlooked by standard VQA benchmarks.

#### Visual Code Generation.

• Humaneval_V [73]: An OOD benchmark that extends the HumanEval code generation task to the visual modality. Each problem presents a function signature alongside a visual example illustrating the intended input-output behavior, requiring models to infer the programming logic from images and produce correct, executable Python code.

#### Realistic Agentic Benchmarks.

- • BFCL-V4 [39]: The Berkeley Function-Calling Leaderboard (Version 4) evaluates an agent’s ability to invoke external functions accurately across single-turn and multi-turn settings, covering both live and non-live API scenarios. It tests real-world tool-use reliability under diverse and compositional function-calling requirements.
- • tau2-bench [6]: A realistic multi-turn agent benchmark simulating customer service interactions (e.g., airline ticketing). It evaluates an agent’s ability to follow complex policies, manage state across turns, and invoke tools correctly to resolve user requests, providing a practical assessment of deployment-ready agentic behavior.

- Table 5: Detailed information on the evaluation datasets used in MAESTRO. ID denotes indomain benchmarks included in the training distribution; OOD denotes out-of-domain benchmarks used for zero-shot generalization; OOD∗ denotes the extended out-of-domain benchmarks evaluated with the augmented registry.

#### Category Task Dataset #Test Samples

Chart Understanding ChartQA [32] 500 Geometric Reasoning Geometry3K [28] 601 Scientific Reasoning MicroVQA [7] 500 Scientific Reasoning MSEarthMCQ 500 Medical VQA Slake [23] 361 Object Counting TallyQA [1] 500

ID

High-Resolution Perception HRBench-4K [54] 800 High-Resolution Perception HRBench-8K [54] 800 High-Resolution Perception VStar [9] 191 Mathematical Reasoning MathVision [52] 304

OOD

Embodied Scene Reasoning ERQA [18] 400 OCR & Text-Rich Understanding OCRBench [25] 1,000 Synthetic Diagram Reasoning VlmsAreBlind [42] 401 Visual Code Generation Humaneval_V [73] 253 Realistic Agentic Evaluation BFCL-V4 [39] 4441 Realistic Agentic Evaluation tau2-bench [6] 370

OOD∗

#### D.3 Baselines

In our experiments, we compare the proposed methods against several baseline approaches. Below, we provide detailed descriptions of each baselines.

- • GPT-4o: GPT-4o is a proprietary multimodal foundation model developed by OpenAI. We use it as a strong general-purpose closed-source baseline to evaluate how well a frontier vision-language model can solve the tasks without task-specific training or access to our learned tool-use policy.

- • GPT-5: GPT-5 is a more recent proprietary model from OpenAI with enhanced multimodal reasoning capability. It serves as a stronger closed-source reference point for assessing the performance gap between our specialized training framework and frontier generalist models.
- • Gemini-2.5-Flash: Gemini-2.5-Flash is a lightweight and latency-oriented multimodal model from Google. We include it to compare against a cost-efficient proprietary model that is designed for fast inference while retaining competitive visual reasoning ability.
- • Gemini-2.5-Pro: Gemini-2.5-Pro is the stronger model in the Gemini-2.5 family and is intended for more complex reasoning scenarios. This baseline measures the performance of a highcapability proprietary VLM on our benchmark.
- • GLM-4.6V: GLM-4.6V is a proprietary vision-language model from Zhipu AI. We evaluate it as a representative Chinese-developed multimodal foundation model with strong general visual understanding and reasoning capabilities.
- • Kimi-K2.5: Kimi-K2.5 is a proprietary model from Moonshot AI. It is included as another competitive closed-source baseline, allowing us to compare our method with a recent large-scale model that emphasizes long-context understanding and general reasoning.
- • Qwen3-VL-32B-Instruct: Qwen3-VL-32B-Instruct is an open-source vision-language model from Alibaba’s Qwen series. We evaluate it as a strong Chinese-developed multimodal baseline, providing a competitive reference for instruction-following, visual understanding, and multimodal reasoning capabilities.
- • Direct Answering: This baseline uses the original, untrained Qwen3-VL-4B-Thinking model to answer each query directly. No external model consultation, skill routing, or learned workflow is used. It isolates the raw task-solving ability of the backbone model before any training.
- • Untrained Model: This baseline also starts from the original, untrained Qwen3-VL-4B-Thinking model, but allows it to follow our proposed workflow. Specifically, the model can attempt to call external models and use the available skills when producing an answer. This setting separates the benefit of the workflow interface itself from the benefit of our training procedure.
- • best_model: This baseline denotes the strongest checkpoint selected from our model pool according to validation performance. It provides an upper reference among individual trained models and helps quantify the additional gain brought by our full method beyond simply choosing the best single model.
- • DeepEyes [79]: DeepEyes is a recent visual reasoning method that trains a multimodal model to “think with images” through reinforcement learning, enabling active perception by grounding reasoning in visual information without relying on external specialized models or APIs. We compare against it to evaluate whether our approach can more effectively coordinate model selection and skill usage for complex visual reasoning tasks.
- • DeepEyesV2 [13]: DeepEyesV2 is an agentic multimodal model that learns to actively invoke external tools, including code execution environments and web search, and integrate these operations into multimodal reasoning. It uses a two-stage training pipeline with cold-start tool-use learning followed by reinforcement learning, making it a strong baseline for evaluating our method’s coordination of model selection and skill usage in complex visual reasoning tasks.
- • Thyme [75]: Thyme is a tool-enhanced multimodal reasoning framework that enables models to autonomously generate and execute code for diverse image processing and computational operations, such as cropping, rotation, contrast enhancement, and mathematical calculation. It activates this capability through SFT followed by reinforcement learning, making it a representative baseline for comparing against approaches that augment VLMs with executable operations for complex visual reasoning.
- • VTOOL-R1 [63]: VTOOL-R1 is a reinforcement-learning finetuning framework that trains VLMs to produce multimodal chains of thought by interleaving textual reasoning with intermediate visual reasoning steps. It integrates Python-based visual editing tools into training and uses outcome-based rewards to elicit strategic tool use, providing a relevant baseline for evaluating our method against tool-oriented visual reasoning systems.
- • VTS-V [5]: VTS-V is an inference-time visual token scaling framework that enables MLLMs to iteratively refine visual understanding through verifier-guided reasoning. We include it as a dynamic visual reasoning baseline with adaptive, context-aware perception during inference.

- • MathCoder-VL[53]: MathCoder-VL is a multimodal mathematical reasoning model trained with code-supervised cross-modal alignment. It first uses the large-scale image-code dataset ImgCode-8.6M to align mathematical figures with their underlying code representations, and is then fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. We include it as a strong open-source baseline specialized for mathematical figure understanding and geometry reasoning.
- • Visual-ARFT[27]: Visual-ARFT is a visual agentic reinforcement fine-tuning method that enables LVLMs to use external tools for multimodal reasoning, including browsing websites for realtime information and writing code to manipulate and analyze images through operations such as cropping and rotation. It provides a relevant baseline for evaluating our method against open-source multimodal agents with both search and coding abilities.
- • VisionReasoner[26]: VisionReasoner is a unified visual perception reasoning framework that enhances a vision-language model through a unified reward mechanism and multi-object cognitive learning strategies. It generates structured reasoning processes for diverse perception tasks, including detection, segmentation, and counting, making it a relevant baseline for evaluating unified visual reasoning and perception capabilities.
- • Chain-of-Focus[74]: Chain-of-Focus is an adaptive multimodal reasoning framework that teaches vision-language models to perform visual search and image zooming only when necessary. It first constructs multi-step reasoning trajectories with diverse resolutions and question complexities for supervised fine-tuning, and then applies reinforcement learning with an adaptive group-aware reward to learn when to focus on local visual details. We include it as a relevant baseline for evaluating efficient visual reasoning methods that balance fine-grained perception, global understanding, and computational cost.

#### D.4 Implementation Details

We train MAESTRO with GRPO using the verl / verl-tool stack. The policy is initialized from Qwen3-VL-4B-Thinking and trained on one node with 4 NVIDIA A100 GPUs, each with 80GB memory, under FSDP. Training takes 3 days and 11 hours. We use AdamW with learning rate

- 1 × 10−6, weight decay 0.01, betas (0.9,0.999), no warmup, a constant learning-rate schedule, and gradient clipping at norm 1.0. Training runs for 380 update steps. We sample n = 8 trajectories per prompt, giving a GRPO group size of 8. The training batch size is 32 prompts per update, and the PPO mini-batch size is also 32 at the prompt level. Thus, each update contains 32 × 8 = 256 rollout trajectories before FSDP sharding. The per-GPU PPO micro-batch size is 1, and the per-GPU log-probability micro-batch size is 8. Dynamic batch sizing is enabled. No critic/value model is used.

Rollouts are generated asynchronously with vLLM using tensor parallel size 1, bfloat16 rollout weights, max_num_seqs=512, and gpu_memory_utilization=0.6. The maximum prompt and response lengths are 12,288 and 4,096 tokens, respectively. Sampling uses temperature 1.0, top-p 1.0, top-k = −1, and repetition penalty 1.0; validation also uses 8 sampled rollouts per prompt. The agent can make up to 4 tool-interaction turns, with observations truncated to 1,024 tokens and actions capped at 8,192 tokens. Observation tokens are masked from the policy loss and KL computation.

We use the vanilla clipped policy objective with clip ratio 0.2, i.e., ratios are clipped to [0.8,1.2], and dual-clip constant 3.0. Entropy regularization is disabled. Rewards are computed by the rule-based torl reward manager. The scalar reward is assigned to the last valid response token. For GRPO, rewards are normalized within the 8-rollout group:

ri − µg σg + 10−6,

Ai =

and the resulting scalar advantage is broadcast to all unmasked response tokens. We do not apply additional global reward normalization or reward clipping. Expert calls use the format <search>Model@@Skill: query</search> and are served through local OpenAI-compatible vLLM endpoints.

### E More Results and Analysis

#### E.1 Efficiency and Scalability Analysis

We provide detailed results in Table 6. We emphasize that all reported numbers (token consumption and end-to-end latency) reflect the full system cost, including both the 4B orchestrator and every invocation of the 4–9B expert models together with the skills. The numbers are not restricted to the orchestrator alone.

A natural question is why MAESTRO achieves lower latency and fewer tokens than single-model “Think with Images” baselines such as VTOOL-R1, despite involving multiple experts in the loop. The explanation lies in how the workload is decomposed. The entire decomposition step (deciding what kind of help is needed, which expert to call, and which skill to attach), is performed by the lightweight 4B orchestrator alone. A heavier 4–9B expert is then triggered only when a specific capability is genuinely required, and each expert call is narrowly scoped to a single sub-problem expressed by the skill prompt. As a result, expert invocations are short, focused, and infrequent, in contrast to monolithic “Think with Images” approaches that repeatedly re-prompt the same large model with redundant image zooming and trial-and-error tool calls. The net effect is that the total token budget and wall-clock latency of the ensemble remain below those of single-model iterative approaches, even when the active expert is comparable in size.

#### E.2 Scaling with Skill Pool Size

We provide the detailed scaling skill results in Table 7. The configurations are: N=2 (Chart, Geometric); N=4 (+Counting, Science); N=5 (+Perception); N=8 (+Embodied Scene, OCR, Python Code Generator).

- As shown in Table 7, expanding the skill pool from N=2 to N=8 improves average accuracy from 60.7% to 66.5% (+5.8%). Gains are particularly notable on specialized benchmarks: ERQA improves from 43.0% to 52.3% (+9.3%) and OCRBench from 74.9% to 85.0% (+10.1%), reflecting the benefit of task-specific skills. While the average latency rises accordingly from 3.27s to 4.03s, the increase is sub-linear compared to the performance gains. This suggests that the RL-driven orchestrator learns an efficient dispatching logic, invoking higher-order expert combinations only when necessary, thereby maintaining a favorable balance between reasoning power and computational efficiency as the expert ecosystem expands.

#### E.3 Test-Time Scaling

We further explore test-time scaling using Self-Consistency (SC) [55], sampling multiple reasoning trajectories and selecting answers by majority vote. As shown in Table 8, increasing the number of sampled paths consistently improves performance across all benchmarks.

The results indicate that MAESTRO benefits significantly from additional computation at inference time. Moving from pass@1 to sc@16 yields a steady improvement in average accuracy, particularly in complex domains like MathVision (+2.7%) and TallyQA (+4.4%). This scalability demonstrates that the RL-trained orchestrator provides a high-quality distribution of reasoning paths, where the correct coordination of models and skills can be more reliably identified through majority voting or consistent sampling.

#### E.4 Additional Discussion

Analysis of RL Training Convergence. As shown in Figure 6, we validate the stability and effectiveness of the RL-driven orchestration policy by monitoring the evolution of mean rewards and policy entropy during the training phase. Figure 6a illustrates that the total reward R exhibits a steady upward trajectory, eventually reaching a stable plateau. This progression indicates that the orchestrator successfully learns to optimize the coordination of expert models and skills to maximize task success rates. Simultaneously, as shown in Figure 6b, the policy entropy undergoes a significant and smooth reduction, terminating at a lower value compared to the initial exploration phase. This trend signifies a successful transition from early-stage stochastic exploration to a more deterministic and high-confidence orchestration strategy. The convergence of these metrics confirms

- Table 6: Average token consumption, inference latency (seconds), and accuracy per benchmark. MAESTRO achieves the best overall performance and efficiency. The evaluation is systematically divided into in-domain and out-of-domain datasets. For each metric, the subsequent “(∆ vs. best)” row reports the absolute difference of our method compared to the strongest baseline, with improvements highlighted in red.

Method

In-Domain Out-of-Domain

Avg. Geom ChartQA Slake MicroVQA MSE TallyQA VStar HRB-4K HRB-8K MathV

Token Consumption (↓, fewer is better)

DeepEyes 849.3 589.3 426.4 1024.7 1056.6 485.3 452.3 509.3 503.9 681.6 657.9 DeepEyes-v2 931.5 593.6 452.1 1302.3 1198.5 473.3 598.2 524.7 537.9 946.1 755.8 Thyme 815.4 643.2 455.5 1264.8 880.0 467.5 586.0 679.2 673.0 931.6 739.6 VTOOL-R1 524.3 497.2 525.4 1023.1 759.9 506.7 418.6 785.6 798.9 756.3 659.6 MAESTRO (Ours) 864.6 514.3 375.6 988.0 1007.2 428.0 416.1 517.0 506.1 865.0 648.2 (∆ vs. best) +340.3 +17.1 -50.8 -35.1 +247.3 -39.5 -2.5 +7.7 +2.2 +183.4 -9.7

Inference Time (↓, seconds)

DeepEyes 4.89 2.16 1.81 3.67 3.69 1.58 2.53 4.97 5.86 3.04 3.42 DeepEyes-v2 5.27 2.59 1.94 4.28 3.57 1.59 2.89 5.65 5.97 3.96 3.77 Thyme 4.75 2.73 2.04 3.89 3.16 1.53 2.96 3.81 4.15 3.24 3.23 VTOOL-R1 4.59 1.96 2.11 3.87 2.81 1.59 2.56 3.17 4.05 3.18 2.99 MAESTRO (Ours) 5.71 1.89 1.46 3.50 3.36 1.38 2.12 2.69 3.04 3.62 2.88 (∆ vs. best) +1.12 -0.07 -0.35 -0.17 +0.55 -0.15 -0.41 -0.48 -1.01 +0.58 -0.11

Accuracy (↑, %)

DeepEyes 20.8 69.4 58.7 48.8 45.0 73.0 85.6 75.1 72.6 26.6 57.6 DeepEyes-v2 38.9 72.2 66.2 41.4 46.4 70.6 81.8 77.9 73.8 28.9 59.8 Thyme 17.5 86.1 62.6 48.8 42.2 73.2 82.2 77.0 72.0 27.6 58.9 VTOOL-R1 24.1 86.7 60.7 43.8 45.4 79.4 78.5 68.5 66.4 29.3 58.3 MAESTRO (Ours) 77.4 86.8 66.2 53.0 52.4 79.8 88.0 79.6 74.4 43.4 70.1 (∆ vs. best) +38.5 +0.1 +0.0 +4.2 +6.0 +0.4 +2.4 +1.7 +0.6 +14.1 +10.3

- Table 7: Performance (Acc.) and latency (s) as a function of skill pool size N. The RL-based routing consistently leverages additional skills to improve accuracy with sub-linear latency growth.

N VStar HRB-4K HRB-8K MathV Geom ChartQA Slake MicroVQA MSE TallyQA ERQA OCRBench Human. Avg. Accuracy 2 80.6 71.8 69.5 34.9 77.9 86.8 57.9 49.4 49.0 75.4 43.0 74.9 18.6 60.7

- 4 85.9 73.3 70.5 38.8 77.5 86.4 65.9 52.8 52.6 79.6 43.5 74.3 18.2 63.0
- 5 88.0 79.6 74.4 43.4 77.4 86.8 66.2 53.0 52.4 79.8 42.8 79.4 19.4 64.8

- 8 88.0 79.3 74.4 43.8 77.5 86.2 65.7 52.8 52.4 79.0 52.3 85.0 28.1 66.5

Latency (s) 2 1.94 2.61 2.81 3.54 5.37 1.60 1.35 3.38 3.21 1.10 1.31 1.38 12.91 3.27

- 4 2.14 2.67 2.84 3.65 5.67 1.90 1.43 3.41 3.33 1.35 1.38 1.43 12.70 3.38
- 5 2.12 2.69 3.04 3.62 5.71 1.89 1.46 3.50 3.36 1.38 1.53 1.47 16.84 3.74 8 2.14 2.65 2.97 3.83 5.62 1.98 1.56 3.55 3.31 1.42 1.56 1.40 20.39 4.03

that MAESTRO internalizes a robust and consistent routing logic, ensuring reliable performance across diverse multimodal reasoning tasks.

Analysis of Performance Upper Bound. We further investigate the impact of Reinforcement Learning (RL) on the orchestrator’s decision-making and evaluate the potential performance upper bound using pass@k metrics. As shown in Table 9, RL training yields a substantial +17.4% gain in pass@1 average accuracy (52.7%→70.1%), confirming that the learned routing policy is the primary driver of performance. The pass@16 results further reveal meaningful headroom: MAESTRO reaches 94.0% on Geometry3K and 92.7% on VStar, with an average of 84.9%. The gap between pass@1 (70.1%) and pass@16 (84.9%) indicates that correct model-skill coordination is attainable within the existing registry for most cases, motivating future refinement of the orchestration search strategy.

Detailed Ablation Results. We provide detailed ablation results in Table 10 and Table 11. A closer inspection reveals that the format reward is crucial for enabling effective tool use. Without it, the model often fails to follow the required action schema and may sequentially output both <search>...</search> and <answer>...</answer>, while only one of them should be selected at each step. These invalid outputs block proper tool execution, so the system largely degenerates into direct answering by the 4B backbone, causing a much larger accuracy drop. In contrast, removing the outcome reward preserves the basic ability to issue well-formatted calls, so the model can still invoke external models and skills. Although the selected calls may no longer be consistently optimal due to

Table 8: Test-time scaling results using Self-Consistency (sc@k). Sampling more trajectories consistently improves accuracy across benchmarks.

Metric VStar HRB-4K HRB-8K MathV Geom ChartQA Slake MicroVQA MSE TallyQA Avg. pass@1 88.0 79.6 74.4 43.4 77.4 86.8 66.2 53.0 52.4 79.8 70.1 sc@4 88.3 80.0 75.0 44.7 78.0 87.2 66.5 54.2 54.0 82.8 71.1 sc@8 89.0 80.8 75.8 45.7 78.5 88.0 67.3 54.6 54.8 83.0 71.8 sc@16 89.5 81.8 77.3 46.1 79.5 89.6 68.7 55.2 55.8 84.2 72.8

[Figure 42]

[Figure 43]

(a) Reward Convergence (b) Entropy Loss

- Figure 6: Training dynamics of MAESTRO. (a) Mean reward rises steadily and plateaus, with the format reward variant (blue) converging to a higher level. (b) Policy entropy declines smoothly, indicating a transition from exploration to confident orchestration.

the missing discriminative signal, even imperfect tool use usually provides stronger support than the

- 4B backbone alone. Therefore, the degradation from removing the outcome reward is notably smaller than that from removing the format reward.

#### E.5 Statistical Significance Analysis

We apply the Wilcoxon signed-rank test [19] to assess statistical significance, pairing MAESTRO against the strongest “Think with Images” baseline, VTOOL-R1, across all ten benchmarks in Table 1. This yields p = 9.7 × 10−4 (p < 0.05), rejecting the null hypothesis H0 of no significant difference. On out-of-domain benchmarks, the test yields p = 6.1 × 10−3 (p < 0.05), validating both the robustness and generalization capability of MAESTRO.

### F Case Study

We present representative examples in Figures 10–14 to illustrate how MAESTRO orchestrates expert models and hierarchical skills across diverse multimodal tasks in both in-domain and out-of-domain settings.

Task-Aware Model-Skill Orchestration. Figures 10 and 12 demonstrate how the orchestrator aligns task semantics with the appropriate model-skill combination. For example, in Figure 10 (VStar), given a fine-grained color perception question, MAESTRO coordinates GLM-4.6V-Flash and the Perception_Problem_Solver skill to zoom into the relevant image region, identify the scarf color as red, and return the correct answer (B). In Figure 12 (Slake), given a chest X-ray with the question “Which part of the body does this image belong to?”, the orchestrator identifies this as a medical perception task and selects MedGemma-1.5-4b-it paired with Perception_Problem_Solver. The expert first performs a global scan recognizing the chest cavity structure, then zooms into the cardiac region to resolve local ambiguity. Both views confirm the answer “chest”. This case illustrates that routing to a medically fine-tuned backbone improves robustness on clinical imagery, a gain that a general-purpose model cannot reliably provide, even on seemingly straightforward anatomy questions.

#### Table 9: Performance upper bound at pass@16 vs. pass@1, with and without RL training.

Setting VStar HRB-4K HRB-8K MathV Geom ChartQA Slake MicroVQA MSE TallyQA Avg. Pass@1 Results with/without MAESTRO RL Training w/o RL 41.4 70.3 68.5 29.0 38.9 74.8 54.3 38.8 36.8 74.4 52.7 w/ RL 88.0 79.6 74.4 43.4 77.4 86.8 66.2 53.0 52.4 79.8 70.1 △ +46.6 +9.3 +5.9 +14.4 +38.5 +12.0 +11.9 +14.2 +15.6 +5.4 +17.4 Pass@16 Results with/without MAESTRO RL Training w/o RL 65.4 83.3 80.8 48.0 50.6 80.2 70.9 70.3 67.6 87.8 70.5 w/ RL 92.7 90.1 89.5 62.2 94.0 89.8 80.1 78.2 81.2 91.6 84.9 △ +27.3 +6.8 +8.7 +14.2 +43.4 +9.6 +9.2 +7.9 +13.6 +3.8 +14.4

###### Table 10: Ablation study of MAESTRO components. The model pool and skill library each contribute independently, and their combination is essential for peak performance.

Configuration VStar HRB-4K HRB-8K MathV Geom ChartQA Slake MicroVQA MSE TallyQA Avg. MAESTRO (Full) 88.0 79.6 74.4 43.4 77.4 86.8 66.2 53.0 52.4 79.8 70.1

w/o Skill Pool 82.7 77.6 71.6 41.1 75.0 83.6 63.5 51.0 50.2 77.4 67.4 w/o Model Pool 81.5 76.9 71.5 27.6 22.3 78.2 59.3 43.4 42.0 76.8 58.0 w/o Both 78.5 73.5 68.6 25.3 19.8 77.8 57.9 41.0 40.4 75.0 55.8

Plug-and-Play Generalization to OOD Experts and Skills. Figure 13 illustrates MAESTRO’s zero-shot extensibility on the ERQA benchmark. Presented with an embodied scene question about robot arm actions, MAESTRO invokes the newly added Qwen3.5-9B model jointly with the Embodied_Scene_Problem_Solver skill, neither of which was present during training. Together, they analyze the spatial relationship between the gripper, the open jar, and the nearby lid, correctly concluding that the robot is closing the jar (C). This demonstrates that the learned orchestration policy generalizes to unseen model-skill combinations on demand, without any structural retraining.

### G Additional Discussion

#### G.1 Limitation and Failure Case Analysis

The current skill library relies on human-authored descriptions, which may require some manual effort as the registry scales. However, given the general design of MAESTRO’s orchestration framework, extending it toward automated skill generation is a natural and feasible direction, which we discuss in Appendix G.8.

Beyond scalability, we examine representative failure cases to identify where future performance gains are most attainable. The most common failure pattern occurs on tasks that lie at the boundary between two skill categories, such as a chart question that also requires domain-specific scientific knowledge. In these cases, the policy tends to commit to one Level-1 skill and does not reconsider within the allotted turns. A second pattern appears on Humaneval_V, where the challenge lies not in routing but in the inherent difficulty of inferring programming logic from visual examples alone. Importantly, the pass@16 results in Table 9 show that the correct answer is reachable within the existing model-skill space in the vast majority of cases, indicating that these failures reflect routing precision rather than fundamental coverage gaps, and motivate further refinement of the orchestration policy itself.

#### G.2 Clarification on Model Scale and Computational Cost

A natural question concerns the overall computational footprint of MAESTRO relative to the closedsource frontier models it is compared against. We clarify two points here.

All expert models are open-source. The orchestrator and every model in the candidate pool are fully open-source. Specifically, the default registry comprises five expert models: GLM-4.6VFlash [72], Chart-R1 [8], Qwen3-VL-8B-Instruct [4], Intern-S1-mini [3], and MedGemma-1.5-4bit [45]. These are relatively compact, widely-used vision-language models with publicly available weights. No proprietary or closed-source model is involved at any stage of training or inference.

- Table 11: Ablation of reward components. Removing either the format reward or the outcome reward leads to substantial performance degradation, confirming that both signals are necessary for stable multi-turn orchestration.

Configuration VStar HRB-4K HRB-8K MathV Geom ChartQA Slake MicroVQA MSE TallyQA Avg. Full MAESTRO 88.0 79.6 74.4 43.4 77.4 86.8 66.2 53.0 52.4 79.8 70.1

w/o rfmt 65.5 67.5 61.3 25.7 40.9 80.2 57.9 47.2 46.8 76.8 57.0 w/o rans 73.8 73.8 70.8 26.0 52.9 81.2 59.0 49.8 48.2 77.0 61.3

Total parameter count and FLOPs are substantially smaller than frontier closed-source models. Although MAESTRO coordinates an ensemble of models rather than relying on a single backbone, the combined parameter count of the entire system remains well below that of frontier closed-source models such as GPT-5 or Gemini-2.5-Pro. Table 12 summarizes the parameter counts of each component. The 4B orchestrator and the five expert models all have parameter counts between 4B and 9B, well below 10B. Critically, during any single inference episode, only one expert model is invoked per reasoning step (Algorithm 1), so the active parameter count at each step is bounded by the orchestrator plus one expert, which is far smaller than the estimated scale of GPT-5 or Gemini-2.5-Pro. The inference latency results in Figure 3 and Table 6 further confirm that MAESTRO achieves the lowest average latency (2.88s) among all compared methods, corroborating its practical efficiency.

- Table 12: Parameter counts of all models in the default MAESTRO registry. The orchestrator is always active; expert models are invoked selectively, one per reasoning step.

#### Component Role Parameters

Qwen3-VL-4B-Thinking Orchestrator (always active) 4B GLM-4.6V-Flash Expert model 9B Chart-R1 Expert model 8B Qwen3-VL-8B-Instruct Expert model 8B Intern-S1-mini Expert model 9B MedGemma-1.5-4b-it Expert model 4B

In summary, MAESTRO demonstrates that a carefully coordinated ensemble of open-source models, guided by a lightweight RL-trained orchestrator, can match or exceed the performance of proprietary frontier models while remaining accessible, reproducible, and computationally efficient.

#### G.3 Skill Design Cost and Engineering Basis

A legitimate concern is the human engineering effort required to construct the hierarchical skill library. We address this transparently from two angles: the actual design cost, and the principled basis on which each skill was built.

Design basis. Rather than being designed from scratch, each Level-1 skill and its associated Level-

- 2 sub-routines were systematically derived from existing benchmark-specific methodologies and open-source toolchains. Concretely:

- • Geometric Problem Solver (S1) draws its multi-step structured extraction protocol from the interpretable geometry solver InterGPS [28], and its verification checklist mirrors the self-consistency strategy of [55].
- • Chart Problem Solver (S2) is grounded in the chart-type routing and OCR-aided value recovery pipeline from ChartQA [32] and Chart-R1 [8], which already establish strong baselines for bar, line, and pie chart parsing.
- • Counting Problem Solver (S3) adopts the detection-assisted enumeration paradigm from VisionReasoner [26] and the zoom-based localization strategy from DeepEyes [79], reusing their public prompting strategies with minor task-specific adaptation.
- • Perception Problem Solver (S4) and Science Problem Solver (S5) leverage the hierarchical visual grounding workflow from VTOOL-R1 [63] and the image-caption plus OCR fusion strategy from Thyme [75].

- • Extended skills (S6–S9), introduced for OOD evaluation, were adapted directly from the task definitions and evaluation protocols of their respective benchmarks: ERQA [18] for embodied scene reasoning, OCRBench [25] for text-rich understanding, VlmsAreBlind [42] for synthetic diagram reasoning, and Humaneval_V [73] for visual code generation. Each benchmark paper provides explicit task decompositions that directly informed the corresponding Level-2 sub-skill routing logic.

Actual engineering effort. Given this strong grounding in prior work, the marginal design cost per skill was modest. For each Level-1 skill, the primary effort involved: (i) formalizing the benchmark’s recommended solving procedure into a structured multi-step prompt, and (ii) defining keywordbased routing rules for Level-2 sub-skills based on question type taxonomy already provided by the benchmark authors. We estimate the total prompt engineering effort at approximately 3-5 personhours for the default five skills (S1–S5), and an additional 1-2 person-hours for the four extended skills (S6–S9). We acknowledge that this cost, while moderate, is a real constraint on scalability, and we discuss automated skill generation as a future direction in Appendix G.8.

Coverage and generalization of the default skill set. A natural concern is whether the humanauthored skill library tightly overfits to the benchmarks at hand. The empirical evidence supports, that this is not the case. The default Level-1 skills (S1–S5), i.e., geometric reasoning, chart understanding, counting, fine-grained perception, and scientific reasoning, are not benchmarkspecific subroutines but rather generic visual capability primitives that recur across virtually every task in multimodal evaluation. As a result, even without introducing any of the benchmarkaligned extension skills (S6–S9), the default configuration of 5 experts and 5 Level-1 skills is sufficient to cover a wide range of unseen task families.

The OOD specialized evaluation in Table 2 provides a clean test of this claim. The unaugmented MAESTRO (default 5/5 setup, no skill tailored to ERQA, OCRBench, VlmsAreBlind, or Humaneval_V) reaches an average of 52.7%, significantly outperforming every “Think with Images” baseline (best: DeepEyes-v2 at 45.0%) and remaining competitive with frontier closed-source models such as Gemini-2.5-Pro (55.6%) and GPT-5 (53.3%). The further +6.8% gain to 59.5% obtained with the augmented S6–S9 set should therefore be read as an additional benefit of registry expansion, not as a precondition for cross-domain generalization. In other words, the marginal engineering effort of S6– S9 buys additional precision on tasks whose structure is already known, while the underlying default skill set already provides broad coverage of unseen domains – consistent with the plug-and-play view developed in Section 4.3.

#### G.4 Sensitivity to Skill Descriptions

Each Level-1 skill is presented to the orchestrator as a concise natural-language description. We find that the orchestrator is robust to minor paraphrasing but benefits from descriptions that are discriminative in scope. Specifically, describing each skill in terms of its input type and expected output format, rather than abstract capabilities, leads to more consistent orchestration decisions. This observation also suggests a promising direction: as the skill pool grows, automatic skill description generation or refinement could further improve routing precision without any retraining of the orchestrator.

#### G.5 Detailed Comparison with Concurrent Works

Several concurrent works also study skill-based agents, and we expand on the brief discussion in the main-paper Related Work to make our positioning explicit. We group them into three categories.

Skill representation and lifelong skill evolution. SkillX [50] introduces skill representations as a vehicle for structured knowledge distillation, while AutoSkill [66] and Skill0 [29] focus on autonomous skill evolution and in-context skill internalization, respectively. These works primarily ask how skills are represented, accumulated, or distilled into a single backbone. MAESTRO is orthogonal: we take a skill library as given (in our case, two-tier and human-authored, but in principle replaceable by any of the above) and ask how a lightweight policy can coordinate multiple frozen experts on top of such a library. In other words, prior skill-evolution methods can be viewed as upstream providers of K, whereas MAESTRO learns a distribution over (m,s) pairs.

Skill routing without a model pool. SkillRouter [78] and SkillOrchestra [51] explicitly study routing among skills, and SkillRL [65] co-evolves a single agent’s policy with its skill bank via recursive RL. These methods assume a single reasoning backbone that selects among skills, and therefore do not encounter the model-skill compatibility problem that motivates our compositional action asearcht = (mt,st,zt). Section A formalizes the compatibility term Cc(m,s) that is invisible to single-backbone routers, and the ablation in Table 10 confirms that removing the model pool causes a substantially larger drop (−12.1%) than removing the skill pool (−2.7%).

Skill management at scale. AgentStore [16] and Memora [64] address scalable skill management through retrieval and reranking pipelines. We compare against this design philosophy directly in Appendix G.6: retrieval treats model and skill selection as independent similarity-based lookups, whereas MAESTRO learns joint (m,s) assignments from outcome-based rewards and revises them over multiple turns. The +17.4% gap between the untrained workflow (52.7%) and the RL- trained orchestrator (70.1%) in Table 1 is entirely attributable to this difference, since both share the same M × K registry.

Multimodal collaboration. Beyond the skill-centric line of work, recent multimodal agents such as AppAgent V2 [21] and InternVideo2 [56] use structured action spaces and modular vision tools, and orthogonal efforts on optical self-compression [12] and hierarchical memory [68] target highdensity multimodal histories. These works improve a particular component of the agent stack (action interface, tool execution, or memory). MAESTRO can in principle be combined with any of them: our orchestrator does not modify expert weights, action interfaces, or memory representations, and treats

- them as black-box capabilities to be composed.

Summary. Across all these categories, the distinguishing feature of MAESTRO is that it learns a policy over the joint M × K space, rather than over K alone, and that it does so via outcome-based RL rather than retrieval. We view concurrent skill-evolution and skill-management methods as complementary, and integrating an automatically grown skill registry into our orchestration layer is a natural direction for future work (Appendix G.8).

#### G.6 Why Reinforcement Learning over Retrieval-Based Routing

A natural alternative to our approach is retrieval-based dispatching, where the most relevant skill is selected via embedding similarity between the query and skill descriptions. Compared to retrieval, our RL-based approach offers two key advantages. First, retrieval treats model and skill selection independently, whereas MAESTRO learns to select joint model-skill ensembles, capturing cross-modal synergies that static similarity scores cannot model. Second, retrieval is inherently single-step, while our multi-turn formulation allows the orchestrator to revise its strategy based on expert feedback from prior steps. The pass@1 comparison in Table 9 quantifies this directly: without RL optimization, the policy achieves only 52.7%, while the RL-trained orchestrator reaches 70.1%, a gap of +17.4% that is entirely attributable to learned routing quality.

#### G.7 Emergent Behavior During Training

Beyond the aggregate curves in Figure 6, we observe several qualitative behavioral changes throughout training that are worth highlighting. In early stages (steps 0–50), the orchestrator frequently invokes multiple search actions per episode and occasionally generates malformed action sequences, reflected in high entropy and volatile rewards. After approximately 50 steps, the policy learns to produce well-formed, single-call trajectories for straightforward tasks, and the format reward stabilizes. In later stages (steps 100+), an emergent selective multi-turn behavior develops: the orchestrator reserves follow-up calls for genuinely ambiguous cases, such as high-resolution images where a first observation is insufficient, while solving simpler tasks in a single step. This behavior is not explicitly supervised and arises purely from outcome-based reward optimization, illustrating the expressive power of GRPO in long-horizon agentic settings.

###### G.8 Future Work MAESTRO opens several promising directions for future research.

Self-Evolving Skill Registries. The current skill library is manually curated and fixed after deployment. A natural extension is to enable the system to automatically discover, compose, and refine skills from interaction history, allowing MAESTRO to self-evolve skill registries.

Online Policy Adaptation. The orchestrator is currently trained offline on a fixed dataset. Enabling online adaptation, where the policy continues to improve from deployment-time interactions, would allow MAESTRO to specialize to user- or domain-specific distributions over time.

Multi-Turn Self-Correction. Incorporating an explicit revision mechanism, where the orchestrator can re-invoke a different model-skill pair upon detecting a low-confidence or contradictory response, could further close the gap between pass@1 and pass@16 performance observed in Table 9.

Broader Modalities and Action Types. Extending MAESTRO to video, audio, and structured data modalities, as well as to richer action types such as code execution and web interaction, would position it as a general-purpose orchestration layer for heterogeneous agentic ecosystems.

Theoretical Foundations. While our empirical results strongly support the effectiveness of outcome-based RL for orchestration, formalizing the sample complexity of learning orchestration policies and characterizing the conditions under which model-skill synergies emerge would provide a principled basis for future system design.

### H Broader Impact

MAESTRO advances the paradigm of collaborative AI orchestration, where a lightweight policy coordinates heterogeneous expert models and modular skills rather than consolidating all capabilities into a single large model. This approach carries several positive societal implications. By decoupling task routing from model parameters, MAESTRO lowers the barrier to deploying specialized AI capabilities: domain experts in medicine, science, and engineering can integrate purpose-built models into a unified agentic pipeline without retraining or architectural changes. The framework’s computational efficiency, achieving the lowest average latency among compared methods, also suggests that high-quality multimodal reasoning need not require frontier-scale compute, potentially broadening access to capable AI systems.

- At the same time, more capable agentic systems that autonomously invoke external tools and expert models introduce new risks. A framework that coordinates multiple specialized models could be misused to construct automated pipelines for disinformation generation, targeted content manipulation, or privacy-violating information aggregation at scale. The non-interventional design of MAESTRO, which leaves all expert models frozen, means that any harmful behaviors present in the underlying models are inherited rather than amplified; however, the orchestration layer could make such behaviors easier to trigger systematically. We encourage future deployments to pair MAESTRO with content moderation filters and access controls on the underlying expert models, particularly in open-ended agentic settings. We also note that the training data and evaluation benchmarks used in this work are sourced from publicly available academic datasets and do not involve personal data collection.

###### System Prompt for RL Experiments

You are a router and planner. Plan how to answer the given question. If there is no information between <information> and </information>, first analyze which model and skill to use. If information already exists, verify whether more evidence is needed from candidate models and skills. If assistance is needed, write a request inside <search> ... </search>. If enough information is available, summarize the final answer briefly inside <answer> ... </answer>.

================================================================ SKILL SELECTION GUIDELINES

================================================================

- 1. Geometric_Problem_Solver: Use for geometry diagrams, figures, angles, lengths, areas...
- 2. Chart_Problem_Solver: Use for chart tasks such as bar, line, pie, and trend analysis...
- 3. Science_Problem_Solver: Use for science figures in biology, geography, physics...
- 4. Counting_Problem_Solver: Use for counting target objects or repeated instances...
- 5. Perception_Problem_Solver: Use for local visual inspection and region-level perception...

================================================================ AVAILABLE MODELS

================================================================

1. Intern-S1-mini: Best for science questions: biology, geography, physics, chemistry... 2. GLM-4.6V-Flash: Best for math, geometry, logical reasoning, general perception... 3. qwen3-vl-8b-instruct: Best for counting objects or instances in images...

- 4. Chart-R1: Best for line charts, bar charts, pie charts, and visual data...
- 5. medgemma-1.5-4b-it: Best for medical images and clinical assessment...

================================================================ !!! STRICT FORMAT RULES for <search> !!!

================================================================ You MUST use the following format EXACTLY:

<search> Model-Name@@Skill-Name:Your-Text-Input </search> Examples:

<search> GLM-4.6V-Flash@@Geometric_Problem_Solver: Find angle ABC. </search>

... Model-Name must be one of: [Intern-S1-mini, GLM-4.6V-Flash, qwen3-vl-8b-instruct, Chart-R1, medgemma-1.5-4b-it]. Skill-Name must be one of: [Geometric_Problem_Solver, Chart_Problem_Solver, Science_Problem_Solver, Counting_Problem_Solver, Perception_Problem_Solver].

Use a concrete input that directly helps answer the original question. Never output placeholders, copy skill descriptions, or leave Model-Name, Skill-Name, or Your-Text-Input unreplaced.

================================================================ REASONING REQUIREMENTS BEFORE SKILL CALL

================================================================ Before EACH <search> call, briefly think about:

+ the question category: mathematical, counting, scientific, medical, chart, or perception;

+ why external help or verification is needed;

+ which skill and model best match the question. Do NOT solve the question by yourself. Keep the thinking process as short as possible.

================================================================ SKILL RESPONSE HANDLING & FINAL ANSWER

================================================================ Skill responses are returned inside <information> ... </information>. You may call different skills or models multiple times, or re-call the same skill when useful. If enough information is available, provide only: <answer> actual-answer </answer>

- Figure 7: System prompt used in the RL experiments. The prompt defines the orchestrator’s action format, model-skill invocation protocol, and response constraints during reinforcement learning.

[Figure 44]

- Figure 8: An example on the ChartQA dataset. MAESTRO performs two rounds of skill invocation: it first coordinates GLM-4.6V-Flash with Perception Problem Solver to locate the 2010 bar group,

- then invokes Chart-R1 with Chart Problem Solver to align the category and extract the value for “Uninsured now”.

[Figure 45]

###### Figure 9: An example on the MSEarthMCQ dataset. MAESTRO coordinates Intern-S1-mini and Science Problem Solver to interpret geological features in a gravity gradient map.

[Figure 46]

###### Figure 10: An example on the VStar dataset. MAESTRO coordinates GLM-4.6V-Flash and Perception_Problem_Solver to resolve a fine-grained color perception question.

[Figure 47]

###### Figure 11: An example on the TallyQA dataset. MAESTRO engages Qwen3-VL-8B-Instruct with Counting Problem Solver to enumerate objects under occlusion.

[Figure 48]

###### Figure 12: An example on the Slake dataset. MAESTRO coordinates MedGemma-1.5-4b-it and Perception Problem Solver to identify the anatomical region in a chest X-ray.

[Figure 49]

###### Figure 13: An example on the ERQA dataset (OOD extension). MAESTRO coordinates the newly added Qwen3.5-9B and Embodied Scene Problem Solver without retraining to resolve a robot manipulation question.

[Figure 50]

- Figure 14: An example on the VlmsAreBlind dataset (OOD extension). MAESTRO performs two rounds of skill invocation: it first coordinates qwen3.5-9b with ocr problem solver to recognize the full word “Subdermatoglyphic”, then invokes qwen3.5-9b with Diagram Reasoning Skill to localize the red-oval highlight and identify the target character “e”.

###### Design of Geometric Problem Solver Skill

- [Step 1] Question: {question} Extract ALL visual and text information in this structured format:

- 1. POINTS:

- Labeled points with type and approximate position.

- 2. LINES & SEGMENTS:

- Endpoints, line type, labels, and special markings.

- 3. ANGLES:

- Vertex, arms, measurements, and right-angle symbols.

- 4. CIRCLES/ARCS:

- Center, radius/diameter labels, and arc endpoints.

- 5. ANNOTATIONS:

- Lengths, angles, ratios, tick marks, and detached text.

- 6. RELATIONSHIPS:

- Parallelism, perpendicularity, tangency, and collinearity. Output this as a structured list with clear headings.

- [Step 2] Question: {question} Your own visual extraction from Step 1: {response_of_step_1} Imagecaption evidence read from file: {caption_result} OCR evidence read from file: {ocr_result}

Compare your own visual understanding with the outputs from imagecaption and OCR, then determine the most accurate description. Requirements:

- - Treat Step 1 as primary evidence, imagecaption as global support, and OCR as evidence for labels, numbers, and symbols.
- - Mention conflicts or ambiguities, resolve them using the most visually and logically consistent interpretation, and output a concise consolidated description.

- [Step 3] Question: {question} Structured extraction from Step 1: {response_of_step_1} Imagecaption evidence from Step 2: {caption_result} OCR evidence from Step 3: {ocr_result} Consolidated description from Step 2: {response_of_step_2}

Begin step-by-step reasoning with self-explanation to solve the problem. Requirements:

- - Restate the target, list key known facts, and identify the relevant geometric relations or theorems.
- - Derive the result step by step; if multiple interpretations exist, use the best-supported one.
- - End with a provisional answer before final verification.

- [Step 4] Question: {question}

- Step 1 structured extraction: {response_of_step_1}
- Step 2 consolidated description: {response_of_step_2}
- Step 3 step-by-step solution: {response_of_step_3}

Review the entire reasoning process and check the final solution for consistency and correctness. Consistency checklist:

- 1. DIMENSIONAL ANALYSIS:

- Check units, angle measures, and ratios.

- 2. EXTREME CASE TESTING:

- Test special configurations and boundary cases.

- 3. CROSS-METHOD VERIFICATION:

- Compare with an alternative method, estimation, or symmetry.

- 4. PLAUSIBILITY CHECK:

- Is magnitude reasonable?

If the previous reasoning contains an error, correct it. If the previous reasoning is sound, confirm it briefly. The final line must be the final answer only, stated clearly in the format the question asks for.

- Figure 15: Workflow design for the geometric problem solver skill. The skill first extracts structured geometric information, then consolidates visual, caption, and OCR evidence before solving and verifying the result.

###### Design of Chart Problem Solver Skill

- [Step 1] Question: {question} OCR evidence read from file: {ocr_result}

Classify the chart type from the image. Use the image as primary evidence and OCR as supporting evidence. If OCR is noisy, infer the chart type from visual structure. Respond with exactly one of:

bar chart line chart pie chart

- [Step 2] Question: {question} Step 1 chart type decision: {chart_type} OCR evidence: {ocr_result} Apply general chart operation rules.

- - Match positions in the question with the actual chart layout.
- - Use numerical magnitude when the question depends on values.
- - Read values from axes, labels, annotations, or visual bar/line/sector size.
- - Treat predicted rows and columns as helpful alignment hints, not final evidence.

- [Step 3] Use chart-type-specific instructions. For bar charts, identify row labels, legend colors, bar direction, and endpoint values. For line charts, match x-axis positions to rows, line colors to columns, and points to values. For pie charts, match sector colors to labels and read values from sector annotations or legend links.
- [Step 4] Answer the question based on the image itself, OCR evidence, and the selected chart rules. If the question depends on positions, align those positions with the chart image. If it depends on values, infer them from axes, labels, annotations, or visual magnitudes. Give concise reasoning, then provide the final answer on the last line.

- Figure 16: Workflow design for the chart problem solver skill. The skill guides the model to parse chart elements, recover numerical evidence, and perform chart-grounded reasoning.

Design of Science Problem Solver Skill

- [Step 1] Question: {question} Imagecaption evidence read from file: {caption_result} OCR evidence read from file: {ocr_result}

Collect global and textual evidence. Use imagecaption for overall context. Use OCR for text, numbers, labels, units, and symbols. The original image remains the primary source of truth.

- [Step 2] Question: {question}

Ask DeepEyes whether a local region needs magnification. Think first. If a region is necessary for accurate science reasoning, call the zoom tool with a bounding box:

image_zoom_in_tool(bbox_2d=[x1,y1,x2,y2]) If no local inspection is needed, answer without a crop.

- [Step 3] DeepEyes output: {deepeyes_response} Zoomed crop: {crop_image_if_provided} If a tool call is present, crop the selected image region and provide it with the original image for final answering.
- [Step 4] Question: {question} Evidence: original image, optional zoomed crop, imagecaption result, and OCR result. Solve the science question.

- - Use the image itself as the primary source of truth.
- - Use the crop to inspect the most relevant local region.
- - If any source conflicts with the image, trust the image.
- - Give concise reasoning.
- - The final line must be the final answer only, in the format required by the question.

- Figure 17: Workflow design for the science problem solver skill. The skill focuses on extracting scientific visual evidence and applying domain knowledge for step-by-step reasoning.

###### Design of Counting Problem Solver Skill

- [Step 1] Question: {question} Detection result read from file: {detection_result} Detection-box image: {box_image}

Prepare visual counting evidence. For each image, provide the original image, the detection-box image when available, and detection text as a helpful reference.

- [Step 2] Question: {question}

Use DeepEyes to decide whether closer inspection is needed. Inspect counting failure points: crowded areas, tiny targets, ambiguous regions, borders, reflections, and partial occlusions. If zoom helps, call:

image_zoom_in_tool(bbox_2d=[x1,y1,x2,y2])

- [Step 3] Question: {question} Detection result: {detection_result} Zoomed crop: {crop_image_if_provided} List all requested objects and their positions.

- - Focus only on the categories requested in the question.
- - Use the original image as primary evidence.
- - Use detection boxes and zoomed crops as support, not ground truth.
- - Scan region by region and enumerate each matched object clearly.
- - Mention suspected missed or partially occluded objects.

- [Step 4] Object position list from Step 3: {response_of_step_3}

Verify the count before answering. Check whether any object was double-counted or missed. Reconcile conflicts between the object list and the image evidence. Give concise reasoning and then output the final answer.

- Figure 18: Workflow design for the counting problem solver skill. The skill asks the model to identify target objects, check for occlusion or missing instances, and produce a verified count.

Design of Perception Problem Solver Skill

- [Step 1] Question: {question} Original image: {image}

Ask DeepEyes whether the perception question needs local zoom. Think first. If a local region is needed for judging color, object identity, or spatial relation, call:

image_zoom_in_tool(bbox_2d=[x1,y1,x2,y2]) Otherwise answer with the global image only.

- [Step 2] DeepEyes output: {deepeyes_response} Zoomed crop: {crop_image_if_provided} If a zoom call is present, crop the selected region and pass both the global view and focused view to the main model.
- [Step 3] Question: {question} Visual inputs: {original_image} and optional {zoomed_crop}

Answer using observation-focused reasoning. System guidance:

- - Carefully analyze the image before judging colors.
- - Identify which part corresponds to the target object.
- - For relative-position questions, compare object locations directly.
- - Use the zoomed crop only to clarify local details.

- [Step 4] Return the main model answer with token usage accumulated across the DeepEyes decision and final model call.

- Figure 19: Workflow design for the perception problem solver skill. The skill emphasizes finegrained visual inspection and evidence-based perceptual judgment.

###### Design of Embodied Scene QA Skill

- [Step 1] Question: {question} Images: {images}

Normalize the embodied scene question. Lowercase the question and collapse whitespace. Count how many images are provided, because multi-image input is a routing signal.

- [Step 2] Score each ERQA route with keyword matches.

- - trajectory: path, arrow, colored trajectory, follow the line.
- - action: adjust, align, rotate, move left/right/up/down, next action.
- - spatial: left, right, front, behind, inside, nearest, handle, hinge.
- - pointing: point, dot, coordinate, label, marker, arrow tip, corner.
- - multi_view: image 1/2/3, same part, correspond, another view, task state.

- [Step 3] Apply routing boosts and priority. Multiple images boost multi-view routing. Same-point or corresponding-part wording boosts pointing. Path or arrow wording boosts trajectory. Move or align wording boosts action. If all scores are zero, default to spatial reasoning. Otherwise choose the highest-scoring route using priority:

trajectory, pointing, multi-view, action, spatial.

- [Step 4] Selected solver: {selected_erqa_solver} Build the selected specialist solver with the original question, images, model settings, and system prompt:

- - Trajectory Outcome Solver
- - Action Adjustment Solver
- - Spatial Mechanics Solver
- - Pointing Localization Solver
- - Multi View Task Solver

Return the selected solver’s generated answer.

- Figure 20: Workflow design for the embodied scene QA skill. The skill supports scene understanding, spatial reasoning, and action-aware question answering in embodied environments.

Design of OCR Problem Solver Skill

- [Step 1] Question: {question} Images: {images}

Normalize the OCRBench question. Collapse whitespace, lowercase the question, and prepare one score for each OCR route.

- [Step 2] Classify the OCR task type by keyword scoring.

- - text recognition: written text or number in the image.
- - key information extraction: receipt, invoice, date, company, amount, nutrition fields.
- - scene text QA: sign, store, shirt, boat, text on a real object.
- - document/chart QA: chart, graph, table, book, calendar, author, title, year.
- - formula recognition: LaTeX, formula, equation, expression.

- [Step 3] Apply routing boosts and priority. LaTeX or formula wording strongly boosts formula recognition. Written-text wording boosts text recognition. Amount, serving, company, or issued wording boosts key information extraction. Book, chart, table, or calendar wording boosts document/chart QA. Shirt, boat, store, sign, or written-on wording boosts scene text QA. If all scores are zero, default to scene text QA. Otherwise use priority:

formula, key information extraction, document/chart QA, scene text QA, text recognition.

- [Step 4] Selected subskill: {selected_ocr_subskill} Build the selected OCR specialist solver with the original question, images, model settings, and the OCRBench system prompt.

- - Text Recognition
- - Key Information Extraction
- - Scene Text QA
- - Document Chart QA
- - Formula Recognition

Return the selected subskill’s generated answer.

- Figure 21: Workflow design for the OCR problem solver skill. The skill combines visual inspection with text recognition evidence to answer questions involving labels, symbols, and written content.

###### Design of Diagram Reasoning Skill

- [Step 1] Question: {question} Images: {images}

Normalize the synthetic diagram question. Lowercase the question, collapse whitespace, and initialize scores for all diagram reasoning routes.

- [Step 2] Score each diagram route with keyword matches.

- - contact/overlap: overlapping, touching each other, two circles.
- - intersection/route counting: blue and red lines, intersection points, single-color paths, routes.
- - grid parsing: rows and columns, row/column, table.
- - highlighted character: which letter, which character, red oval, circled, highlighted.
- - geometric shape counting: how many circles, pentagons, squares, count shapes.

- [Step 3] Apply routing boosts and priority. Which-letter wording boosts highlighted-character recognition. Rows-and-columns wording boosts grid parsing. Overlap or touching wording boosts circle contact/overlap. Intersection, blue/red lines, or single-color wording boosts intersection/route counting. Shapecount wording boosts geometric shape counting. If all scores are zero, default to geometric shape counting. Otherwise use priority:

highlighted character, grid parsing, contact/overlap, intersection/route counting, geometric shape counting.

- [Step 4] Selected subskill: {selected_diagram_subskill} Build the selected VLMsAreBlind specialist solver with the original question, images, model settings, and routing system prompt.

- - Circle Contact and Overlap Judge
- - Intersection and Route Counting
- - Grid Structure Parsing
- - Highlighted Character Recognition
- - Geometric Shape Counting

Return the selected subskill’s generated answer.

- Figure 22: Workflow design for the diagram reasoning skill. The skill extracts diagram structure, aligns textual and visual evidence, and performs structured reasoning over schematic information.

Design of Code Problem Solver Skill

- [Step 1] Question: {question} Image example: {image}

Extract the Python function target. If the question already contains a function signature starting with def, use it. Otherwise build a fallback stub:

def solution(*args, **kwargs): pass

- [Step 2] Candidate implementation: {stub_code}

Extract one concrete test case from the image. Read the example values, coordinates, numbers, or structures shown in the image. Convert them into one valid Python assert statement using the exact function signature. Try up to two extraction rounds. If no valid assertion is found, continue without runtime verification.

- [Step 3] Question: {question} Extracted test: {assert_statement}

Generate a complete runnable Python solution. Return only one final Python code block. The code must implement the function exactly as defined and must not include tests.

- [Step 4] Generated code: {candidate_code} Extracted test: {assert_statement} Validate and repair in a bounded loop.

- - If no valid Python code block is returned, ask for regeneration.
- - If a test exists, run the code plus the assert in a temporary Python file.
- - If execution fails, send the failing test, previous code, and error message back for repair.
- - Repeat up to the configured maximum rounds.

- [Step 5] Return the first code block that passes the extracted test. If no test exists, return the valid generated code. If all rounds fail, return the latest valid code when available.

- Figure 23: Workflow design for the code problem solver skill. The skill guides the model to inspect code-related visual or textual evidence, reason about program behavior, and verify the final answer.

