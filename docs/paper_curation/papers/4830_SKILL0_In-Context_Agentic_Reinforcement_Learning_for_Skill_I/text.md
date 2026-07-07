# arXiv:2604.02268v2[cs.LG]15May2026

## SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization

#### Zhengxi Lu1,2⋆, Zhiyuan Yao2, Jinyang Wu3, Chengcheng Han2, Qi Gu2† Xunliang Cai2, Weiming Lu1, Jun Xiao1, Yueting Zhuang1, Yongliang Shen1† 1Zhejiang University 2Meituan 3Tsinghua University

{zhengxilu, syl}@zju.edu.cn guqi03@meituan.com

### Abstract

Agent skills, structured packages of procedural knowledge and executable resources that agents dynamically load at inference time, have become a reliable mechanism for augmenting LLM agents. Yet inference-time skill augmentation is fundamentally limited: retrieval noise introduces irrelevant guidance, injected skill content imposes substantial token overhead, and the model never truly acquires the knowledge it merely follows. We ask whether skills can instead be internalized into model parameters, enabling zero-shot autonomous behavior without any runtime skill retrieval. We introduce SKILL0, an in-context reinforcement learning framework designed for skill internalization. SKILL0 introduces a training-time curriculum that begins with full skill context and progressively withdraws it. Skills are grouped offline by category and rendered with interaction history into a compact visual context, teaching the model tool invocation and multi-turn task completion. A Dynamic Curriculum then evaluates each skill file’s on-policy helpfulness, retaining only those from which the current policy still benefits within a linearly decaying budget, until the agent operates in a fully zero-shot setting. Extensive agentic experiments demonstrate that SKILL0 achieves substantial improvements over the standard RL baseline (+9.7% for ALFWorld, +6.6% for Search-QA and +10.1% for WebShop), while maintaining a highly efficient context of fewer than 0.5k tokens per step. Our code is available at https://github.com/ZJU-REAL/SkillZero.

### 1 Introduction

“Skills at training, zero at inference.”

— SKILL0

Large Language Models (LLMs) [11, 42, 54, 6, 43] have demonstrated strong decision-making capabilities across complex real-world tasks, including code generation [19], GUI automation [60], gameplay [40], and embodied control [46]. With the emergence of agent scaffolds like Claude Code and OpenClaw, structured Agent Skills [53, 23, 13], which are defined as compact and reusable strategies that capture the principles, have become the standard mechanism for extending agent capabilities on specialized tasks.

(a) Skill Augmentation (e.g. SkillRL)

[Figure 1]

[Figure 2]

Skill Evolve

#Skills

[Figure 3]

[Figure 4]

[Figure 5]

Env Agent Policy

steps

[Figure 6]

Similarity

SkillBank

(b) Skill Internalization (Ours Skill0)

[Figure 7]

On-Policy Helpfulness

[Figure 8]

#Skills

[Figure 9]

[Figure 10]

[Figure 11]

Agent Policy

Env

steps

Zero-shot Inference

[Figure 12]

SkillBank

Figure 1: Comparison of (a) Skill Augmentation methods and (b) our Skill Internalization method.

∗Work done during intership at Meituan. †Corresponding author

The prevailing paradigm is inference-time skill augmentation: relevant skills are retrieved from a skill bank and injected into the model’s context as natural language guidance at each step [24, 26]. This approach has proven effective and is now well-established, with growing ecosystems of skill libraries, retrieval pipelines, and evolution mechanisms [53, 52]. Yet this practical success obscures a more fundamental limitation. First, retrieval noise introduces irrelevant or misleading guidance that corrupts the agent’s context [58, 49]. Second, injected skill content imposes token overhead that compounds across multi-turn interactions, limiting scalability [30, 16]. Third, and most critically, a model that follows skill descriptions in its prompt is executing skills, not learning them: competence resides in the context, not in the model [47, 12].

This observation suggests a different question: rather than asking how to better retrieve and inject skills, can skills be internalized into model parameters, rendering retrieval unnecessary at inference time? Skill acquisition in humans follows a familiar progression: an explicit instruction phase gives way to an internalized phase in which the same behavior is executed autonomously from memory [2, 62]. Inference-time skill augmentation permanently anchors agents in the first stage. Reinforcement learning offers a natural path to the second, driving the agent to consolidate effective strategies as intrinsic policy rather than reading them from context [10, 37]. Yet a naive application of RL fails in both directions: without skill context, the agent lacks the structured guidance necessary to learn complex multi-step behavior; with full skill context throughout, the model remains dependent on external knowledge it has never been required to internalize. What is needed is a training regime that starts with skills and ends without them, systematically transferring competence from context to parameters.

We propose SKILL0, the first RL framework that formulates skill internalization as an explicit training objective. SKILL0 realized this curriculum through In-Context Reinforcement Learning (ICRL): skills are provided as in-context guidance during training rollouts but removed entirely at inference, so that RL optimization directly drives the transition from context-dependent execution to autonomous behavior. Concretely, skills are grouped offline by category and rendered with interaction history into a compact visual context, teaching the model tool invocation and multi-turn task completion. Dynamic Curriculum evaluates each skill file’s on-policy helpfulness by comparing agent performance with and without it on a matched validation sub-task. Skills are retained only where the current policy still benefits, and discarded otherwise, until the budget reaches zero and the agent operates without any skill context. Extensive experiments demonstrate that SKILL0 achieves substantial improvements over strong baselines like AgentOCR ((+9.7% for ALFWorld, +6.6% for Search-QA and +10.1% for WebShop)), and competitive performance against skill-augmented methods like SkillRL. Notably, by eliminating skill reliance at inference time, SKILL0 maintains a highly efficient context of fewer than 0.5k tokens per step, significantly reducing inference overhead without sacrificing task performance.

- • We propose SKILL0, the first RL framework that formulates skill internalization as an explicit training objective, moving agents from inference-time skill dependence to fully autonomous zero-shot behavior.
- • We introduce in-context reinforcement learning, which provides structured skill guidance during training rollouts and removes it entirely at inference, directly optimizing the transition from context-dependent execution to intrinsic competence.
- • We propose Dynamic Curriculum, a helpfulness-driven annealing mechanism that withdraws each skill only when the current policy no longer benefits from it, replacing rigid schedules with adaptive internalization.

### 2 Related Work

#### 2.1 LLM Agents

Recent advancements in instruction-tuned LLMs have enabled autonomous agents to operate across a wide range of dynamic, open-world environments, including code generation [19, 45], GUI automation [60, 28], gameplay [40], and embodied control [46]. With the recent development of reinforcement learning for LLMs [61, 66, 59, 4], agentic RL has emerged as a crucial post-training recipe for equipping LLM agents with robust decision-making capabilities [32, 31, 8].

(a) Skill Grouping

(b) In-Context Reinforcement Learning

- r1
- r2 rG

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- o1
- o2
- oG

[Figure 17]

- S1
- S2

- (1) Render
- (2) Compress

(3) Rollout

...

Domains

x N turns ...

[Figure 18]

[Figure 19]

[Figure 20]

...

- A1
- A2 AG

Relevance

Agent Loop

[Figure 21]

ot

[Figure 22]

[Figure 23]

Sk SN

...

(4) Update

at Policy Agent

[Figure 24]

[Figure 25]

[Figure 26]

Environment

[Figure 27]

SKILL.md

(c) Curriculum Learning

Inference Time

Better Alignment

[Figure 28]

Dynamic Filter

Dynamic Filter

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

[Figure 39]

[Figure 40]

Skill Internalize

SelfCompress

SelfCompress

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Agent Loop

[Figure 51]

[Figure 52]

[Figure 53]

Agent Loop

[Figure 54]

[Figure 55]

[Figure 56]

Agent Loop

[Figure 57]

Skill Markdowns Observations Visual Tokens

[Figure 58]

- Figure 2: Overview of SKILL0. (a) Relevance-Driven Skill Grouping; (b) In-Context Reinforcement Learning with skill-enhanced agent loop; (c) Dynamic curriculum learning during training process.

- 2.2 Agent Skills

Early memory-based approaches store raw trajectories directly into external databases during the sampling process, serving as references for experience replay [64, 39]. However, such raw trajectories are often lengthy, redundant, and noisy, making direct injection into the context window inefficient [5]. To address this limitation, a growing line of work has explored skills—reusable, abstracted, and structured behavioral primitives distilled from historical trajectories [53, 23, 13]. Skills serve as a form of episodic memory that agents can consult at decision time [24, 27, 26], and have further been shown to provide efficient guidance within reinforcement learning frameworks [52, 47, 18]. Despite these advances, existing methods predominantly focus on skill extraction, organization, and retrieval, leaving the question of whether skills can be internalized into model parameters largely unexplored.

- 3 Method: SKILL0

- 3.1 Agent Loop

Task Definition. We formulate agent automation as a sequential decision-making problem. Given a task instruction I, the agent generates a sequence of actions {a1,a2,...,aT} to complete the task. At each step t, the agent operates within a structured environment E (e.g., an online simulator or retrieval engine), which provides a textual observation ot describing the current environmental state. The agent then samples an action at from policy πθ(at|I,ht), where θ denotes model parameters and ht represents history up to time t.

ht = {o1,o2,...,ot} (1)

The environment E transitions to the next state and returns the next observation ot+1 = E(ot,at). This rollout continues until the task is successfully completed or the max step threshold is reached.

Skill Management. Following Xia et al. [52], we organize reusable behavioral knowledge into a hierarchical skill library SkillBank, structured into two levels: (1) General skills capture universal strategic principles applicable across all task types, such as exploration strategies and goal-tracking heuristics. (2) Task-specific skills store specialized knowledge for task category k, including domainspecific action sequences and preconditions.

Skills are organized in a directory structure “skills/{task_name}/{skill_category}.md”, where each Markdown file Sk stores a group of related skills sharing the same task and skill category

(e.g., “skills/search/entity_attribute_lookup.md”). The complete library SkillBank = {Sk}Nk=1 thus contains N such files in total. During training, rather than retrieving individual skills via semantic similarity, we select a subset S ⊆ SkillBank of m skill files ranked by an on-policy helpfulness criterion, which estimates the learning utility of each Sk to the current policy πθ (detailed in Section 3.3). at the current training stage.

Context Rendering. When expanding to more domains, token costs become a key challenge with both the accumulated retrieved skills and interaction history. Inspired by Feng et al. [9] and Shi et al. [38], we introduce a context rendering mechanism that maps the textual interaction context (including history ht and retrieved skills S) to a compact RGB image. Given compression ratio ct, the rendered image is encoded and compressed by the vision encoder Enc into visual representations:

Vt = Enc(ht,S; ct) (2)

where Vt ∈ Rd serves as the compressed visual context embedding fed into the policy, significantly reducing token overhead while preserving the structural information necessary for decision-making.

Rather than treating the compression ratio ct ∈ (0,1] as a fixed hyperparameter, we allow the policy to self-generate ct at each step alongside the task action at:

(at, ct) ∼ πθ(at,ct | I,Vt) (3)

#### 3.2 In-Context Reinforcement Learning (ICRL)

SKILL0 introduces In-Context Reinforcement Learning (ICRL), which combines the sample efficiency and inductive bias of skill prompting with the exploration capability of reinforcement learning. Through a dynamic online curriculum (Section 3.3), skills are progressively internalized into the model’s parameters, eliminating the need for explicit skill retrieval at inference time.

To incentivize both efficient context compression and skill internalization within the agent loop, we introduce a composite reward following Feng et al. [9], which jointly optimizes task success and compression efficiency. Let Isucc(τ) ∈ {0,1} denote the binary success indicator for trajectory τ; the composite reward is defined as:

rtcomp =

ln(ct), if Isucc(τ) = 1, 0, otherwise,

r˜t = rt + λ · rtcomp (4)

where ct ∈ (0,1] is the compression ratio at step t, and the logarithmic formulation reflects the diminishing marginal returns of higher compression. rt evaluates whether the agent completes the task correctly with skill enhancements at step t, and λ ≥ 0 controls the trade-off between task performance and compression efficiency.

For each query q ∼ D, πθ

samples a group of G trajectories {τi}Gi=1. The training objective is:

old

|τi|

G

1

clip(ri,t(θ), Ai, ϵ) − β · DKL[πθ∥πref] (5)

LSKILL0(θ) = Eτ

i∼πθold(q), q∼D

G i=1 |τi|

t=1

i=1

where the advantage Ai is computed by normalizing the total rewards {r˜(τi)}Gi=1 within the sampled group, and ri,t(θ) = πθ(τi,t | q,τi,<t)/πθ

(τi,t | q,τi,<t) is the importance sampling ratio.

old

#### 3.3 Adaptive Curriculum Learning

As training progresses, the reliance on external skills undergoes a controlled annealing process to avoid abrupt distribution shifts in the context space. We formulate this curriculum as a linear decay of the skill budget M(s) at each stage s ∈ {1,...,NS}:

NS − s NS − 1

|S(s)| ≤ M(s) = N ·

(6)

This linear decay bounds the step-wise reduction of the skill context to M(s) − M(s+1) ≈ NN

S−1. By constraining changes to the active skill set S(s), we strictly limit the deviation in the rendered visual context Vt(s) = Enc(ht,S(s);ct). This ensures the distribution shift of the policy πθ(at,ct | I,Vt(s)) remains smooth and stable, safely transitioning the agent to a fully self-reliant state (S(N

##### S) = ∅).

Based on above design, our curriculum operates in two phases: (a) an offline RelevanceDriven Skill Grouping that associates each skill file Sk with a dedicated validation sub-task; and (b) an online Helpfulness-Driven Dynamic Curriculum that adaptively selects the active skill subset S based on the current policy’s learning state during training process.

- (a) Relevance-Driven Skill Grouping. We define the relevance between a validation sub-

task and a skill file Sk as whether the subtask’s domain and objective align with the skill

category encoded in Sk. Based on this relevance, we partition the validation set (subtracted from training dataset) into N sub-tasks {Tk}Nk=1 prior to training, where Tk groups all validation instances whose skill requirements correspond to Sk. This offline grouping ensures each Sk has a dedicated sub-task for evaluating its utility, forming the structural basis for the subsequent dynamic curriculum.

- (b) Helpfulness-Driven Dynamic Curricu-

lum. We split training process into NS progressive stages with a decreasing skill bud-

get M (|M| = NS), gradually reducing the agent’s reliance on external skill guidance until it operates without any retrieved skills. We quantify the helpfulness metric ∆k of each skill file Sk to the current policy πθ by evaluating Tk under two conditions: with Sk provided (w/ skill) and without it (w/o skill) per d training steps. For stage s, we Filter, Rank, and Select top-m (m ≤ M(s)) files from the active skill pool by ∆k (see Algorithm 1).

Algorithm 1 Curriculum Learning for SKILL0

Require: Initial policy πθ; reference model πref; training dataset D; total training steps Ttotal; skill library SkillBank = {Sk}Nk=1; validation sub-tasks {Tk}Nk=1; number of stages NS; validation interval d.

Ensure: Trained policy πθ

- 1: M ← N, ((NNS−2)

S−1)N , . . . , N 1

S−1N , 0

- 2: // Step 0: Initialize active skill subset
- 3: S ← SkillBank
- 4: for stage s = 1, . . . , NS do
- 5: for step t = 1, . . . , ⌊Ttotal/NS⌋ do
- 6: if t mod d = 0 and M(s) > 0 then
- 7: // Step 1: Helpfulness Evaluation for ∀k
- 8: Accw/k skill ← Validate(πθ, Tk, S)
- 9: Accw/ok skill ← Validate(πθ, Tk, ∅)
- 10: ∆k ← Accw/k skill − Accw/ok skill,
- 11: // Step 2: Filter & Rank
- 12: S ← {Sk | ∆k > 0}
- 13: Sort S by ∆k in descending order
- 14: // Step 3: Select top-M(s) skill files
- 15: S ← S[1 : M(s)]
- 16: else if M(s) = 0 then
- 17: S ← ∅
- 18: end if
- 19: // Step 4: Policy update via ICRL
- 20: for q ∼ Batched(D) do
- 21: Rollout trajectories {τi}Gi=1 via Eq. 3
- 22: for each τi do
- 23: Compute reward r˜(τi) via Eq. 4
- 24: Ai ← Normalize {r˜(τi)}Gi=1
- 25: end for
- 26: πθ ← πθ − ∇θ LSKILL0(θ)
- 27: end for
- 28: end for
- 29: end for

### 4 Experiment

#### 4.1 Experiment Setup

Benchmarks We evaluate our methods on ALFWorld [40], Search-based QA [20], and Webshop [56]. ALFWorld is a text-based game aligned with the ALFRED embodied AI benchmark, including 3,827 task instances across six categories of common household activities: Pick and Place (Pick), Look at Obj in Light (Look), Pick Clean then Place in Recep (Clean), Pick Heat then Place in Recep (Heat), Pick Cool then Place in Recep (Cool), and Pick Two Obj and Place (Pick2). Searchbased QA contains several widely-used search-augmented QA benchmarks, including single-hop QA datasets (NQ [22], TriviaQA [21], and PopQA [33]) and multi-hop QA datasets (HotpotQA [55], 2Wiki [14], MuSiQue [44], and Bamboogle [36]). WebShop is a complex, web-based interactive environment designed to test the LLM agents in realistic online shopping scenarios. Agents navigate a realistic web interface to find and purchase products matching user specifications. We select 128 fixed tasks in validation set, which aligns with Feng et al. [8].

Baselines. We first compare SKILL0 with in-context skills prompting methods (with text and OCRbased history) and RL-based methods (GRPO [37], AgentOCR [9], EvolveR [51], and SkillRL [52]) across both benchmarks in Table 1. For ALFWorld only (as shown in Table 6), we additionally report prompt-based agentic or memory-based methods, including ReAct [57] and Reflexion [39], as well as Mem0 [5], ExpeL [64], MemP [7], MemRL [63], and SimpleMem [29]. For search-augmented QA (as shown in Table 7), we include Search-o1 [25], Search-R1 [20], ZeroSearch [41], O2-Searcher [34],

- Table 1: Performance on ALFWorld and Search-QA tasks. We report the success rate (%) and the average context token cost (k) per step. † denotes models validated with skill augmentation; ⋆ denotes methods that encodes visual context with reduced token overhead. We simply reproduce results of SkillRL-3B without cold start and skill evolution. Best and second-best are highlighted.

ALFWorld Search-QA Method Pick Look Clean Heat Cool Pick2 Avg↑ Cost↓ NQ Triv Pop Hotp 2Wk MuS Bam Avg↑ Cost↓ Qwen2.5-(VL)-3B-Instruct

Zero-Shot 27.0 24.3 4.5 20.5 10.2 0.0 15.2 1.21 9.4 31.3 19.8 15.0 14.8 4.7 16.8 15.9 0.48 Few-Shot† 44.1 45.1 30.5 44.1 9.2 3.3 29.3 2.30 11.8 30.9 20.2 13.7 18.4 4.5 25.6 17.9 0.86 Zero-Shot⋆ 44.3 27.6 8.6 3.1 5.7 3.1 17.6 0.48 10.2 27.7 10.9 9.1 12.2 3.7 15.2 12.7 0.15 Few-Shot⋆† 57.1 25.3 4.5 5.5 10.2 9.4 23.8 0.88 11.1 26.2 14.2 15.4 13.4 3.0 19.2 14.6 0.36 GRPO 92.6 85.7 70.6 86.6 79.3 65.0 79.9 1.02 39.3 60.6 41.1 37.4 34.6 15.4 26.4 36.4 0.61 AgentOCR⋆ 91.9 81.8 76.0 73.3 76.1 70.0 78.2 0.38 38.6 56.5 41.7 33.6 30.7 14.6 24.0 34.2 0.26 EvolveR 77.3 24.5 47.9 41.7 24.6 22.5 44.1 1.89 43.4 58.4 43.4 37.3 38.1 13.7 32.8 38.2 –

SkillRL† 91.9 100 82.9 87.4 78.7 70.0 82.4 2.21 38.6 57.6 40.3 33.6 31.1 13.3 58.1 38.9 0.87 SKILL0⋆ 95.6 80.4 100 86.7 78.7 75.2 87.9 0.38 39.8 57.5 42.3 35.1 33.7 13.3 63.7 40.8 0.18

Qwen2.5-(VL)-7B-Instruct Zero-Shot 67.6 35.4 19.3 31.3 30.1 4.4 31.3 1.08 10.4 32.4 22.3 15.8 15.4 7.2 19.2 17.5 0.70 Few-Shot† 75.4 64.9 67.5 26.7 19.4 8.9 48.4 2.12 12.3 36.8 24.5 17.7 18.2 6.5 24.8 20.1 0.97 Zero-Shot⋆ 46.0 35.6 19.1 7.1 5.5 5.4 21.1 0.52 6.9 30.4 12.0 10.5 9.1 5.5 24.0 14.0 0.26 Few-Shot⋆† 44.3 55.4 52.9 0.0 11.2 5.4 28.9 1.79 10.5 31.9 18.7 14.2 14.4 6.9 24.8 17.3 0.41 GRPO 92.6 93.8 85.2 80.0 82.7 56.5 81.8 0.95 45.1 63.7 44.0 43.6 43.2 16.8 37.6 41.9 0.73 AgentOCR⋆ 95.6 96.2 78.1 73.2 72.4 72.0 81.2 0.43 43.1 61.0 45.4 40.8 38.3 15.7 36.8 40.1 0.36 EvolveR 64.9 33.3 46.4 13.3 33.3 33.3 43.8 – 43.5 63.4 45.9 38.2 42.0 15.6 54.4 43.1 – SkillRL† 97.9 71.4 90.0 90.0 95.5 87.5 89.9 – 45.9 63.3 45.9 43.2 40.3 20.2 73.8 47.1 – SKILL0⋆ 100 85.8 94.6 81.9 85.7 80.1 89.8 0.41 42.7 61.1 45.3 40.0 38.3 16.4 66.9 44.4 0.34

- Table 2: Performance on WebShop (128 tasks). We report the Score (%), Acc (%) and the average context token cost (k) per step. All methods encode visual context with reduced token overhead.

Inference Qwen2.5-VL-3B Qwen2.5-VL-7B Method w/ Skills Score↑ Accuracy↑ Tokens↓ Score↑ Accuracy↑ Tokens↓ Baselines Zero-Shot 23.4 6.3 0.46k 26.8 7.8 0.48k Few-Shot ✓ 45.8 [+22.4] 18.4 [+12.1] 0.78k [+0.52k] 51.2 [+24.4] 24.2 [+16.4] 0.91k [+0.43k] AgentOCR 75.2 [+51.8] 56.3 [+50.0] 0.42k [-0.04k] 78.6 [+51.8] 59.3 [+51.5] 0.40k [-0.08k] Ours: RL w/ Skills

SKILL0 ✓ 80.9 [+57.5] 64.1 [+57.8] 0.94k [+0.48k] 85.3 [+58.5] 71.9 [+64.1] 0.89k [+0.41k] SKILL0 78.6 [+55.2] 66.4 [+60.1] 0.49k [+0.03k] 85.1 [+58.3] 74.2 [+66.4] 0.46k [-0.02k]

ParallelSearch [65] and StepSearch [50]. Some closed-source models are also included, such as GPT-4o [17] and Gemini-2.5-Pro [6].

Implementation Details. We train the Qwen2.5-VL series using SKILL0 for at most 180 steps on

- 4 H800 GPUs. For ALFWorld, we adopt the training data split from GiGPO [8], with each batch

- sampling 16 tasks and 8 rollouts per prompt, and a maximum prompt length of 3,072 tokens. For Search-QA, we follow the experimental setup of Search-R1 [20], using E5 [48] as the retriever. The training data are drawn from NQ and HotpotQA, making these two benchmarks in-domain, while the remaining datasets serve as out-of-domain evaluation. Each batch samples 128 tasks with a maximum prompt length of 4,096 tokens. For Webshop, 1000 tasks are selected for training, with each batch
- sampling 16 tasks and 8 rollouts per prompt, and a maximum prompt length of 4,096 tokens. For the curriculum learning schedule, we set the validation subset size to 1,000, the number of curriculum stages to NS = 3, and initialize SkillBank from SkillRL [52] for both environments.

#### 4.2 Main Results

Method Performance. As shown in Table 1 and Table 2, SKILL0 demonstrates exceptional performance across ALFWorld, Search-QA, and WebShop. While introducing explicit skill prompts (Few-Shot) brings moderate improvements over Zero-Shot baselines, the gains are limited, indicating that LLMs struggle to fully leverage skill descriptions without sufficient exploration. In contrast, without external skill prompting during inference, SKILL0 (3B) achieves an average success rate of 87.9 on ALFWorld, 40.8 on Search-QA, and a score of 78.6 with 66.4% accuracy on WebShop,

8

AgentOCR

Skill0

6

4

Reward

2

0

-2

0 20 40 60 80 100

Steps

- Figure 3: Comparison of training dynamics with AgentOCR on Qwen2.5-VL-3B.

10

AgentOCR

Skill0

8

6

Reward

4

2

0

0 24 48 72 96 120

Steps

Figure 4: Comparison of training dynamics with AgentOCR on Qwen2.5-VL-7B.

###### (a) Skill0

###### (b) Inference w/o skill (OCR)

###### (c) Inference w/o skill (Omni)

1.0

1.0

1.0

w/ skill

Skill0

Skill0 GRPO SkillRL

w/o skill

AgentOCR

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.8

0.8

0.8

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|
| |

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Accuracy

Accuracy

Accuracy

0.6

0.6

0.6

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.4

0.4

0.4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

0.2

0.2

| |
|---|

0.0

0.0

0.0

0 30 60 90 120 150

0 30 60 90 120 150

0 30 60 90 120 150

Training Steps

Training Steps

Training Steps

Figure 5: Training Dynamics Comparison. (a) Validation performance of SKILL0 (OCR) with and without skill augmentation, evaluated every 10 training steps. (b) Performance comparison between SKILL0 and AgentOCR, both evaluated without skill augmentation. (c) Performance comparison of SKILL0 (OCR) against GRPO (Text) and SkillRL (Text), all evaluated without skill augmentation.

outperforming AgentOCR by +9.7, +6.6, and +10.1 (accuracy) respectively. Based on 7B models, it delivers scores of 89.8 on ALFWorld, 44.4 on Search-QA, and 85.1/74.2 (score/accuracy) on WebShop, substantially outperforming other RL-based methods such as EvolveR, AgentOCR, and GRPO. Notably, on WebShop the skill-free SKILL0 (7B) surpasses AgentOCR by +6.5 in score and +14.9 in accuracy, demonstrating strong internalization of complex e-commerce navigation behaviors. Furthermore, SKILL0 achieves competitive or even stronger performance against skill-augmented methods like SkillRL. These consistent gains over both zero-shot and skill-augmented baselines across three diverse benchmarks demonstrate that our approach successfully internalizes complex reasoning and tool-use behaviors into the model’s parameters.

Table 6 and Table 7 provide broader comparisons. On ALFWorld, SKILL0 (89.8) largely outperforms memory-augmented learning methods, including ExpeL (46.3), Mem0 (54.7), and MemRL (21.4). On Search-based QA, SKILL0 (44.4) likewise surpasses search-based methods such as Search-R1 (38.5), ZeroSearch (39.1), and EvolveR (43.1), further highlighting its generality.

Token Efficiency. Beyond strong task performance, SKILL0 achieves these results with a substantially lower context token cost. Due to visual context modeling and skill internalization, SKILL0 fundamentally maintains an ultra-low average token cost per step. For instance, using 3B models, it consumes only 0.38k tokens per step on ALFWorld, 0.18k on Search-QA, and 0.49k on WebShop. This is a massive reduction compared to text-based or skill-augmented methods like SkillRL, which costs 2.21k and 0.87k tokens per step on ALFWorld and Search-QA respectively (more than 5× higher). On WebShop, skill-augmented Few-Shot inference requires 0.78–0.91k tokens per step, while SKILL0 without skills achieves superior accuracy at only 0.46–0.49k tokens—a nearly 2× reduction in context cost while delivering over 3× the accuracy of the Few-Shot baseline.

#### 4.3 Training Dynamics

Reward. Throughout RL optimization, SKILL0 maintains consistently higher reward curves on both the 3B and 7B backbones compared to the AgentOCR baseline, as illustrated in Figure 3 and 4.

Look At Obj In Light

Pick And Place

Pick Clean Then Place In Recep

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.5

0.05

0.15

0.4

0.00

0.10

Helpfulness

Helpfulness

Helpfulness

0.3

- -0.20
- -0.15
- -0.10
- -0.05

0.05

0.2

0.00

0.1

0.0

-0.05

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

Training Steps

Pick Cool Then Place In Recep

Pick Heat Then Place In Recep

Success Rate

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

0.20

0.20

0.8

0.15

0.15

0.6

0.10

Helpfulness

Helpfulness

Helpfulness

0.10

0.05

0.4

0.00

0.05

0.2

- -0.15
- -0.10
- -0.05

0.00

0.0

-0.05

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

Training Steps

- Figure 6: Training Dynamics of Helpfulness, which are reported by ∆k for each sub-task k.

[Figure 59]

- Figure 7: Ablations of skill budget M.

###### (a) Inference w/ skill

###### (b) Inference w/o skill

1.0

1.0

Skill0

Skill0

[3,3,3]

Fixed Full

| |
|---|

0.8

0.8

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Accuracy

Accuracy

| |
|---|

| |
|---|

0.6

0.6

| |
|---|

| |
|---|

| |
|---|

0.4

0.4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

0.2

| |
|---|

| |
|---|

0.0

0.0

0 30 60 90 120 150

0 30 60 90 120 150

Training Steps

Training Steps

Figure 8: Ablations of skill budget during training process.

Method Comparison. We further monitor validation accuracy over the course of training in Figure 5. (a) demonstrates that when validated with skill augmentation, the model achieves faster early-stage performance improvement; while validation without skill prompts yields lower initial performance, it gradually catches up toward the end of optimization, revealing a clear trend of skill internalization. To further validate this observation, (b) evaluates models without skill prompts at inference time under a strictly fair comparison setting: SKILL0 still outperforms AgentOCR, confirming that the performance advantage stems from internalized knowledge rather than reliance on explicit skill descriptions. For a broader comparison, (c) contrasts SKILL0 against standard text-based RL baselines under the same skill-free inference protocol. Unlike GRPO and SkillRL, which plateau relatively early in training, SKILL0 continues to improve steadily throughout optimization, ultimately achieving the highest performance upper bound among all compared methods. We also provide subtask dynamics of SKILL0 in Appendix B to further support it.

Helpfulness. Figure 6 illustrates that the helpfulness of skills exhibits a consistent rise-then-fall pattern across all sub-tasks throughout training. In the early stages, helpfulness remains low, as the policy has not yet learned to leverage skill prompts via direct in-context prompting. As training progresses, the policy gradually learns to ground its actions in the provided skill context, leading to a steady increase in helpfulness. In the later stages, the dynamic curriculum progressively reduces the skill budget, compelling the policy to internalize skill knowledge into its parameters rather than relying on external prompts; consequently, ∆k converges back toward zero. This characteristic trajectory empirically validates the synergistic working mechanism of ICRL and curriculum learning, demonstrating that skills serve as effective yet transient scaffolding during policy optimization.

#### 4.4 Ablations

Skill Budget M. Given NS as 3, the Skill Budget M for ALFWorld is calculated as [6,3,0] (and [5,3,0] for Search-QA) according to Algorithm 1. This design enforces a hard upper bound that

- Table 3: Ablations of Dynamic Skill Curriculum on ALFWorld in different inference settings.

Table 4: Impact of Validation Interval d on ALFWorld and Search-QA (subset).

Method w/ S w/o S ∆ Filter & Rank & Select 86.3 87.9 ↑1.6

w/o Filter 81.6 78.9 ↓2.7 w/o Rank (Random Select) 76.6 62.9 ↓13.7

d ALFWorld Search-QA

10 87.9 48.9 5 87.5 49.6 20 78.1 42.3

gradually anneals the number of available skills across curriculum stages, compelling the model to actively and flexibly prune less helpful skills within the Budget, thereby progressively internalizing skill knowledge. To validate the effectiveness of this design, we compare our [6,3,0] against other budgets ([6,6,6],[3,3,3], and [0,0,0]) as well as a Fixed Full setting (without filter). Figure 7 highlights our superior skill internalization: while Fixed Full and [6,6,6] collapse by -12.3 and -13.3 when skill prompts are removed, our method even achieves a +1.6 gain. Training dynamics in Figure 8 (a) show that a static low budget ([3,3,3]) limits early exploration, leading to unstable learning and lower peaks. Conversely, Figure 8 (b) demonstrates that our curriculum strategy consistently outperforms Fixed Full in skill-free inference settings, likely due to the training-inference gap and the skill over-reliance induced by maintaining a constant full skill set throughout training.

Dynamic Curriculum. Table 3 validates the necessity of our three-step helpfulness-driven strategy (Filter & Rank & Select) (Algorithm 1, Step 1-3). It achieves the highest performance (87.9% w/o S) and is the only setting to show a positive transfer (∆ = +1.6%) when skill prompts are removed at inference. In contrast, simply using all skills up to the budget (“w/o Filter”) introduces context noise that drops performance by 2.7%. Worse, selecting skills randomly (“w/o Rank”) causes a severe collapse (∆ = −13.7%, dropping to 62.9%), proving that retaining strictly helpful skills is essential for stable policy learning and preventing superficial prompt dependency.

Validation Interval d. Table 4 explores the impact of the validation interval d used for helpfulness evaluation. While a smaller interval (d = 5) provides marginal gains on Search-QA, it incurs significantly higher computational overhead. We select d = 10 as the optimal trade-off, balancing high task performance with training efficiency.

We also provide more detailed ablation results in Table 5 to demonstrate our careful design.

### 5 Conclusion

In this work, we proposed SKILL0, an in-context reinforcement learning framework that internalizes agent skills directly into model parameters via a Dynamic Curriculum mechanism, eliminating external skill reliance at inference time. Extensive experiments across ALFWorld and Search-QA demonstrate substantial improvements over RL baselines (+9.7, +6.6, and +10.1 respectively) with fewer than 0.5k tokens per step, establishing skill internalization as a principled alternative to the inference-time skill augmentation paradigm. We believe SKILL0 establishes skill internalization as a new principled and scalable paradigm, paving the way from tool-augmented toward truly autonomous LLM agents and self-sufficient intelligence.

Limitations. SKILL0 relies on the quality of the initial SkillBank, and the offline relevance-driven skill grouping requires re-partitioning when applied to new task domains.

### References

- [1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, 2024.
- [2] John R Anderson. Acquisition of cognitive skill. Psychological review, 89(4):369, 1982.
- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.
- [4] Yuxin Chen, Yu Wang, Yi Zhang, Ziang Ye, Zhengzhou Cai, Yaorui Shi, Qi Gu, Hui Su, Xunliang Cai, Xiang Wang, An Zhang, and Tat-Seng Chua. Learning to self-verify makes language models better reasoners. arXiv preprint arXiv:2602.07594, 2026.
- [5] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.
- [6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [7] Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025.
- [8] Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.
- [9] Lang Feng, Fuchao Yang, Feng Chen, Xin Cheng, Haiyang Xu, Zhenglin Wan, Ming Yan, and Bo An. Agentocr: Reimagining agent history via optical self-compression. arXiv preprint arXiv:2601.04786, 2026.
- [10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [12] Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, Youcheng Sun, and Lijie Hu. Swe-skills-bench: Do agent skills actually help in real-world software engineering? arXiv preprint arXiv:2603.15401, 2026.
- [13] Chaoyue He, Xin Zhou, Di Wang, Hong Xu, Wei Liu, and Chunyan Miao. Openclaw as language infrastructure: A case-centered survey of a public agent ecosystem in the wild. 2026.
- [14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 6609–6625, 2020.
- [15] Thibaut Horel and Yaron Singer. Maximization of approximately submodular functions, 2024. URL https://arxiv.org/abs/2411.10949.
- [16] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.
- [17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

- [18] Zhengbo Jiao, Shaobo Wang, Zifan Zhang, Xuan Ren, Wei Wang, Bing Zhao, Hu Wei, and Linfeng Zhang. Agentic proposing: Enhancing large language model reasoning via compositional skill synthesis. arXiv preprint arXiv:2602.03279, 2026.
- [19] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.
- [20] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.
- [21] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, 2017.
- [22] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.
- [23] Hao Li, Chunjiang Mu, Jianhao Chen, Siyue Ren, Zhiyao Cui, Yiqun Zhang, Lei Bai, and Shuyue Hu. Organizing, orchestrating, and benchmarking agent skills at ecosystem scale. arXiv preprint arXiv:2603.02176, 2026.
- [24] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, et al. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670, 2026.
- [25] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 5420–5438, 2025.
- [26] Yuan Liang, Ruobin Zhong, Haoming Xu, Chen Jiang, Yi Zhong, Runnan Fang, Jia-Chen Gu, Shumin Deng, Yunzhi Yao, Mengru Wang, et al. Skillnet: Create, evaluate, and connect ai skills. arXiv preprint arXiv:2603.04448, 2026.
- [27] Chang Liu, Sibo Tian, Xiao Liang, and Minghui Zheng. Self-vla: A skill enhanced agentic vision-language-action framework for contact-rich disassembly. arXiv preprint arXiv:2603.11080, 2026.
- [28] Guangyi Liu, Pengxiang Zhao, Yaozhen Liang, Qinyi Luo, Shunye Tang, Yuxiang Chai, Weifeng Lin, Han Xiao, WenHao Wang, Siheng Chen, et al. Memgui-bench: Benchmarking memory of mobile gui agents in dynamic environments. arXiv preprint arXiv:2602.06075, 2026.
- [29] Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. Simplemem: Efficient lifelong memory for llm agents. arXiv preprint arXiv:2601.02553, 2026.
- [30] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.
- [31] Zhengxi Lu, Jiabo Ye, Fei Tang, Yongliang Shen, Haiyang Xu, Ziwei Zheng, Weiming Lu, Ming Yan, Fei Huang, Jun Xiao, et al. Ui-s1: Advancing gui automation via semi-online reinforcement learning. arXiv preprint arXiv:2509.11543, 2025.
- [32] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Pengxiang Zhao, Guangyi Liu, et al. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 17608–17616, 2026.

- [33] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 9802–9822, 2023.
- [34] Jianbiao Mei, Tao Hu, Daocheng Fu, Licheng Wen, Xuemeng Yang, Rong Wu, Pinlong Cai, Xinyu Cai, Xing Gao, Yu Yang, et al. O2-searcher: A searching-based agent model for open-domain open-ended question answering. arXiv preprint arXiv:2505.16582, 2025.
- [35] George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. An analysis of approximations for maximizing submodular set functions—i. Mathematical programming, 14(1):265–294, 1978.
- [36] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 5687–5711, 2023.
- [37] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [38] Yaorui Shi, Shugui Liu, Yu Yang, Wenyu Mao, Yuxin Chen, Qi Gu, Hui Su, Xunliang Cai, Xiang Wang, and An Zhang. Memocr: Layout-aware visual memory for efficient long-horizon reasoning. arXiv preprint arXiv:2601.21468, 2026.
- [39] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv. org/abs/2303.11366, 8, 2024.
- [40] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.
- [41] Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching. arXiv preprint arXiv:2505.04588, 2025.
- [42] Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.
- [43] Meituan LongCat Team, Anchun Gui, Bei Li, Bingyang Tao, Bole Zhou, Borun Chen, Chao Zhang, Chen Gao, Chen Zhang, Chengcheng Han, et al. Longcat-flash-thinking-2601 technical report. arXiv preprint arXiv:2601.16725, 2026.
- [44] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multi-hop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.
- [45] Aozhe Wang, Yuchen Yan, Nan Zhou, Zhengxi Lu, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Code-a1: Adversarial evolving of code llm and test llm via reinforcement learning. arXiv preprint arXiv:2603.15611, 2026.
- [46] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.
- [47] Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. Reinforcement learning for self-improving agent with skill library. arXiv preprint arXiv:2512.17102, 2025.
- [48] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022.

- [49] Ruipeng Wang, Yuxin Chen, Yukai Wang, Chang Wu, Junfeng Fang, Xiaodong Cai, Qi Gu, Hui Su, An Zhang, Xiang Wang, et al. Agentnoisebench: Benchmarking robustness of tool-using llm agents under noisy condition. arXiv preprint arXiv:2602.11348, 2026.
- [50] Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization. arXiv preprint arXiv:2505.15107, 2025.
- [51] Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, et al. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079, 2025.
- [52] Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.
- [53] Renjun Xu and Yang Yan. Agent skills for large language models: Architecture, acquisition, security, and the path forward. arXiv preprint arXiv:2602.12430, 2026.
- [54] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [55] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pp. 2369–2380, 2018.
- [56] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022.
- [57] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.
- [58] Zhiyuan Yao, Zishan Xu, Yifu Guo, Zhiguang Han, Cheng Yang, Shuo Zhang, Weinan Zhang, Xingshan Zeng, and Weiwen Liu. Toolace-mcp: Generalizing history-aware routing from mcp tools to the agent web. arXiv preprint arXiv:2601.08276, 2026.
- [59] Zhiyuan Yao, Yi-Kai Zhang, Yuxin Chen, Yueqing Sun, Zishan Xu, Yu Yang, Tianhao Hu, Qi Gu, Hui Su, and Xunliang Cai. Coba-rl: Capability-oriented budget allocation for reinforcement learning in llms. arXiv preprint arXiv:2602.03048, 2026.
- [60] Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Fundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.
- [61] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.
- [62] Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, and Hao Peng. From f(x) and g(x) to f(g(x)): Llms learn new skills in rl by composing old ones, 2025. URL https://arxiv.org/abs/2509.25123.
- [63] Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, et al. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192, 2026.
- [64] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024.

- [65] Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, and Rama Akkiraju. Parallelsearch: Train your llms to decompose query and search sub-queries in parallel with reinforcement learning. arXiv preprint arXiv:2508.09303, 2025.
- [66] Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

- A Theoretical Analysis Given Eq. 14, each stage s ∈ {1,...,NS} satisfies:

NS − s NS − 1

|S(s)| ≤ M(s) = N ·

- A.1 Stability Analysis The advantage for trajectory τi is estimated via group-level normalization:

Ai =

r˜(τi) − µG σG

, µG =

1 G

G

j=1

r˜(τj), σG =

1 G

G

j=1

(˜r(τj) − µG)2 (8)

An abrupt skill context shift between stages would degrade the signal-to-noise ratio of Ai. We now quantify how the linear budget schedule controls this effect.

Under the linear budget schedule, the number of skills removed at each stage transition is uniformly bounded:

n(∆s) := |S(s) \ S(s+1)| ≤ M(s) − M(s+1) ≤

N NS − 1

(9)

This follows from M(s) − M(s+1) = ⌈N(NS − s)/(NS − 1)⌉ − ⌈N(NS − s − 1)/(NS − 1)⌉ ≤ ⌈N/(NS − 1)⌉, using ⌈a⌉ − ⌈b⌉ ≤ ⌈a − b⌉ for a ≥ b ≥ 0.

Let R(s)(q) := r˜(τ) denote the reward under context Vt(s) for query q. Assuming each skill file Sk has bounded influence on the expected reward:

sup

q∈D

E[R(s+1)(q)] − E[R(s)(q)] ≤ n(∆s) · δr ≤

N NS − 1

δr (10)

Following Shao et al. [37], the per-step policy improvement satisfies:

J(θs+1) − J(θs) ≥

1 1 − γ

Ea∼π

θs+1

[Aπ

θs(a)] −

2ϵγ 1 − γ · DTVmax(πθ

s+1

,πθ

s

) (11)

where ϵ = maxa |Aπ

θs(a)|. A stage transition introduces non-stationarity of magnitude ⌈N/(NS − 1)⌉δr (Eq. 10), yielding advantage bias:

|Bias(A(is→s+1))| ≤

n(∆s) · δr σG(s)

(12)

The estimator remains reliable when (i) n(∆s) is small (controlled by NS), or (ii) σG(s) is large relative to the perturbation. By Guo et al. [11], GRPO retains monotonic improvement when:

⌈N/(NS − 1)⌉ · δr σG

< ϵclip (13) This yields a sufficient condition on the number of stages:

NS > 1 +

N · δr ϵclip · σG

(14)

- A.2 Skill Selection Optimality

(7)

For skill file Sk and active set S, the marginal helpfulness is defined as:

∆k(S) := Acc(πθ, Tk, S) − Acc(πθ, Tk, S \ {Sk}) (15) The selection objective at stage s is:

S∗(s) = argmaxS⊆SkillBank, |S|≤M(s) J(S; πθ) (16)

where J(S;πθ) = k:S

k∈S Acc(πθ,Tk,S).

Assuming J(·;πθ) satisfies α-approximate submodularity [15, 35], i.e., for all A ⊆ B ⊆ SkillBank and Sk ∈/ B:

J(A ∪ {Sk}) − J(A) ≥ α · (J(B ∪ {Sk}) − J(B)), α ∈ (0,1] (17) the greedy selection of skills with ∆k > 0 in decreasing order up to budget M(s) satisfies:

J(Sgreedy; πθ) ≥ 1 − e−α · J(S∗(s); πθ) (18) When α = 1, Eq. 18 recovers the classical (1 − 1/e)-approximation ratio. As the policy internalizes skills via GRPO, helpfulness scores evolve as:

∂ ∂θ

∂∆k ∂θ

[Acc(πθ,Tk,S) − Acc(πθ,Tk,∅)] (19)

=

Successful internalization of Sk drives Acc(πθ,Tk,∅) to Acc(πθ,Tk,S), hence ∆k to 0. Such skills are automatically excluded by the positivity filter in Eq. 16, yielding a self-paced curriculum where M(s) serves as an upper bound.

### B More Training Dynamics

Figure 9 and Figure 10 present the per-subtask training dynamics of SKILL0 (Qwen2.5VL-3B) with and without skill context. Across both ALFWorld and Search-QA, the w/ skill result consistently achieves faster early-stage performance improvement. The skill-free result yields lower initial performance and gradually catches up toward the end of optimization, mirroring the skill internalization trend observed in Figure 5. These fine-grained per-subtask dynamics further confirm that the progressive annealing of skills drives the model to internalize task-relevant knowledge into its parameters.

Look At Obj In Light

Pick And Place

Pick Clean Then Place In Recep

1.0

1.0

1.0

w/o skill

w/o skill

w/o skill

| |
|---|

| |
|---|

w/ skill

w/ skill

w/ skill

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.8

0.8

0.8

| |
|---|

| |
|---|

| |
|---|

| |
|---|

SuccessRate

SuccessRate

SuccessRate

| |
|---|

| |
|---|

0.6

0.6

0.6

| |
|---|

0.4

0.4

0.4

| |
|---|

| |
|---|

| |
|---|

0.2

0.2

0.2

| |
|---|

| |
|---|

| |
|---|

0.0

0.0

0.0

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

Training Steps

Pick Cool Then Place In Recep

Pick Heat Then Place In Recep

Pick Two Obj And Place

1.0

1.0

1.0

w/o skill

w/o skill

w/o skill

w/ skill

w/ skill

w/ skill

| |
|---|

0.8

0.8

0.8

| |
|---|

| |
|---|

| |
|---|

SuccessRate

SuccessRate

SuccessRate

| |
|---|

| |
|---|

0.6

0.6

0.6

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.4

0.4

0.4

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.2

0.2

0.2

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.0

0.0

0.0

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

Training Steps

- Figure 9: Training dynamics of SKILL0 on Qwen2.5VL-3B, with ALFWorld accuracy reported.

Compare

Direct Retrieval

0.56

w/o skill

w/o skill

0.54

| |
|---|

w/ skill

w/ skill

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.48

| |
|---|

0.45

| |
|---|

| |
|---|

| |
|---|

SuccessRate

SuccessRate

0.40

0.36

| |
|---|

0.32

0.27

0.24

0.18

0.09

0.16

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

Entity Attribute Lookup

Multi Hop Reasoning

0.48

w/o skill

w/o skill

0.30

| |
|---|

w/ skill

w/ skill

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0.40

| |
|---|

| |
|---|

0.25

| |
|---|

| |
|---|

SuccessRate

SuccessRate

| |
|---|

0.32

| |
|---|

0.20

| |
|---|

0.24

0.15

0.16

0.10

0.08

0.05

0 20 40 60 80 100 120 140

0 20 40 60 80 100 120 140

Training Steps

Training Steps

- Figure 10: Training dynamics of SKILL0 on Qwen2.5VL-3B, with SearchQA sub-tasks (split by skill categories) accuracy reported.

### C Ablation Details

We provide detailed ablation results of our curriculum design in Table 5, as a supplement for Figure 7 and Table 3.

Table 5: Detailed Results of Ablations.

Method Pick Look Clean Heat Cool Pick2 Avg vs. SKILL0

SKILL0 (w/ S) 98.5 80.5 97.2 77.8 82.8 71.3 86.3 SKILL0 (w/o S) 95.6 80.4 100 86.7 78.7 75.2 87.9

- ∆ ↓2.9 ↓0.1 ↑2.8 ↑8.9 ↓4.1 ↑3.9 ↑1.6

Ablations on Skill Budget [6,6,6] (w/ S) 90.4 69.4 97.0 95.2 74.1 81.3 85.9 ↓0.4 [6,6,6] (w/o S) 90.3 55.9 100.0 25.8 36.7 85.7 72.6 ↓15.3

- ∆ ↓0.1 ↓13.5 ↑3.0 ↓69.4 ↓37.4 ↑4.4 ↓13.3 [6,4,2,1,0] (w/ S) 81.9 65.3 88.5 88.6 83.9 30.4 70.3 ↓16.0 [6,4,2,1,0] (w/o S) 80.4 66.8 87.9 93.7 58.5 40.7 71.1 ↓16.8
- ∆ ↓1.5 ↑1.5 ↓0.6 ↑5.1 ↓25.4 ↑10.3 ↑0.8 [0,0,0] (w/o S) 95.7 67.8 80.1 63.6 83.2 61.1 78.9 ↓9.0

Ablations on Dynamic Curriculum (Filter & Rank & Select)

w/o Filter (w/ S) 91.7 65.9 97.5 73.0 73.0 74.1 81.6 ↓4.7 w/o Filter (w/o S) 91.0 45.0 98.3 65.5 71.9 71.5 78.9 ↓9.0 ∆ ↓0.7 ↓20.9 ↑0.8 ↓7.5 ↓1.1 ↓2.6 ↓2.7

w/o Rank (w/ S) 92.0 62.4 93.4 86.7 46.1 68.4 76.6 ↓9.7 w/o Rank (w/o S) 88.4 42.3 95.5 25.0 25.3 56.5 62.9 ↓25.0

- ∆ ↓3.6 ↓20.1 ↑2.1 ↓61.7 ↓20.8 ↓11.9 ↓13.7

### D More Comparisons

Table 6: Comparison on ALFWorld benchmark. ∗ denotes the results trained with GRPO.

Method Pick Look Clean Heat Cool Pick2 Avg Closed-source LLMs

GPT-4o [17] 75.3 60.8 31.2 56.7 21.6 49.8 48.0 Gemini-2.5-Pro [6] 92.8 63.3 62.1 69.0 26.6 58.7 60.3

Qwen2.5-(VL)-3B-Instruct

Vanilla [3] 27.0 24.3 4.5 20.5 10.2 0.0 15.2 GRPO [37] 92.6 85.7 70.6 86.6 79.3 65.0 79.9 EvolveR [51] 77.3 24.5 47.9 41.7 24.6 22.5 44.1 AgentOCR [9] 91.9 81.8 76.0 73.3 76.1 70.0 78.2 SKILL0 (Ours) 95.6 80.4 100 86.7 78.7 75.2 87.9

Qwen2.5-(VL)-7B-Instruct

Vanilla [3] 33.4 21.6 19.3 6.90 2.80 3.20 14.8 ReAct [57] 48.5 35.4 34.3 13.2 18.2 17.6 31.2 Reflexion [39] 62.0 41.6 44.9 30.9 36.3 23.8 42.7 Mem0 [5] 54.0 55.0 26.9 36.4 20.8 7.69 33.6 ExpeL [64] 21.0 67.0 55.0 52.0 71.0 6.00 46.3 MemP [7] 54.3 38.5 48.1 56.2 32.0 16.7 41.4 SimpleMem [29] 64.5 33.3 20.0 12.5 33.3 3.84 29.7 RLOO [1] 87.6 78.2 87.3 81.3 71.9 48.9 75.5 GRPO [37] 90.8 66.1 89.3 74.7 72.5 64.7 77.6 MemRL [63] 62.8 38.5 22.2 12.5 8.00 0.00 21.4 EvolveR [51] 64.9 33.3 46.4 13.3 33.3 33.3 43.8 Mem0∗ [5] 78.1 54.8 56.1 31.0 65.0 26.9 54.7 SimpleMem∗ [29] 89.5 36.3 60.0 50.0 64.9 26.3 62.5 AgentOCR [9] 95.6 96.2 78.1 73.2 72.4 72.0 81.2 SKILL0 (Ours) 100 85.8 94.6 81.9 85.7 80.1 89.8

Table 6 and Table 7 present extended comparisons against a broader set of baselines beyond those reported in Table 1. On ALFWorld, SKILL0 achieves average success rates of 87.9 (3B) and 89.8 (7B), substantially outperforming memory-augmented approaches such as ExpeL (46.3), SimpleMem (62.5), Mem0 (54.7), and MemRL (21.4), as well as closed-source models including GPT-4o (48.0) and Gemini-2.5-Pro (60.3). On Search-QA, SKILL0 attains average scores of 40.8 (3B) and 44.4 (7B), surpassing retrieval-augmented and search-based methods including RAG (27.0/30.4), Search-R1 (32.5/38.5), ZeroSearch (31.7/39.1), and EvolveR (38.2/43.1). Notably, SKILL0 achieves particularly strong performance on out-of-domain multi-hop datasets such as Bamboogle (63.7/66.9), highlighting its robust generalization to unseen reasoning tasks without any domain-specific adaptation.

### E Implementation Details

We follow the rendering configurations in Feng et al. [9] to construct the visual context in SKILL0 for each benchmark, with the full prompts shown in Figure 11 and Figure 12. Text is rendered in a monospace font with a line spacing of 1.2 across both environments, with a font size of 10pt and a maximum width of 392px for ALFWorld and WebShop, and 12pt with 560px for SearchQA. To enable visual disambiguation of different context components, we apply a semantic color coding scheme: task instructions and general context are rendered in black, while for ALFWorld, observations are highlighted in blue and actions in red; for Search-QA, the same convention is applied to <search> queries and <information> results, respectively, allowing the vision encoder to clearly distinguish between perceived states, executed actions, and retrieved content at a glance.

- Table 7: Results on Search-based QA. † and ⋆ denote in-domain and out-of-domain respectively.

Method

Single-Hop QA Multi-Hop QA

Avg. NQ† TriviaQA⋆ PopQA⋆ HotpotQA† 2Wiki⋆ MuSiQue⋆ Bamboogle⋆

Qwen2.5-(VL)-3B-Instruct

Vanilla 12.4 30.6 5.6 16.0 19.2 4.4 16.8 15.0 CoT 15.0 33.6 3.6 16.2 18.0 3.6 12.8 14.7

- RAG 34.8 54.4 38.7 25.5 22.6 4.7 8.0 27.0 RA-Agent 15.2 28.4 6.6 12.6 16.6 2.6 13.6 13.7 IRCoT 11.1 31.2 20.0 16.4 17.1 6.7 24.0 18.1 Search-o1 16.6 31.0 8.2 14.8 22.4 5.2 22.4 17.2 SFT 24.9 29.2 10.4 18.6 24.8 4.4 11.2 17.6 R1-Instruct 21.0 44.9 17.1 20.8 27.5 6.0 19.2 22.4 Reject Sampling 29.4 48.8 33.2 24.0 23.3 5.9 21.0 26.5 Search-R1 34.1 54.5 37.8 32.4 31.9 10.3 26.4 32.5 ZeroSearch 41.4 57.4 44.8 27.4 30.0 9.8 11.1 31.7 StepSearch - - - 34.5 32.0 17.4 34.4 –

- EvolveR 43.4 58.4 43.4 37.3 38.1 13.7 32.8 38.2 SKILL0 (Ours) 39.8 57.5 42.3 35.1 33.7 13.3 63.7 40.8

Qwen2.5-(VL)-7B-Instruct

Vanilla 11.6 35.6 1.2 16.4 22.2 4.8 14.4 15.2 CoT 12.8 35.6 3.8 16.2 22.6 6.6 24.0 17.4 RAG 34.9 58.5 39.2 29.9 23.5 5.8 20.8 30.4 RA-Agent 21.2 40.2 8.8 19.6 19.6 7.6 28.0 20.7 IRCoT 22.4 47.8 30.1 13.3 14.9 7.2 22.4 23.9 Search-o1 19.4 40.6 11.4 17.0 27.0 8.6 30.4 22.1 SFT 31.8 35.4 12.1 21.7 25.9 6.6 11.2 20.7 R1-Instruct 27.0 53.7 19.9 23.7 29.2 7.2 29.3 27.1 Reject Sampling 36.0 59.2 38.0 33.1 29.6 12.3 35.5 34.8 Search-R1 39.3 61.0 39.7 37.0 41.4 14.6 36.8 38.5 ZeroSearch 43.6 61.8 51.5 34.6 35.2 18.4 27.8 39.1 StepSearch – – – 38.6 36.6 22.6 40.0 –

- EvolveR 43.5 63.4 44.6 38.2 42.0 15.6 54.4 43.1 SKILL0 (Ours) 42.7 61.1 45.3 40.0 38.3 16.4 66.9 44.4

- Table 8 and Table 9 present representative examples of the skill files stored in SkillBank, illustrating the structured procedural knowledge provided to the agent across all three task categories.

#### Prompt of SKILL0 on ALFWorld

You are an expert agent operating in the ALFRED embodied Environment. {skill_context} Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s). The provided image shows the most recent {history_length} observations and the corresponding actions you took. You are now at step {current_step} and your current observation is: {current_observation}. Your admissible actions of the current situation are: [{admissible_actions}]. Now it’s your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. Once you’ve finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags. Additionally, select an image compression factor larger than 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. You must provide the next compression factor within <compression> </compression> tags (e.g., <compression>1.1</compression>).

Figure 11: Prompt template used by SKILL0 for the ALFWorld embodied task environment.

#### Prompt of SKILL0 on Search-based QA

You are an expert agent tasked with answering the given question step-by-step. {skill_context} Your question: {task_description}. Prior to this step, you have already taken {step_count} step(s). The image contains the full history:

- • Past queries are inside <search>...</search>
- • Past results are inside <information>...</information>

Now it’s your turn to respond for the current step. You should first conduct a reasoning process. After completing your reasoning, choose only one of the following actions (do not perform both):

- 1. If any required knowledge is missing or uncertain, you MUST call a search engine to get more external information using format: <search> your query </search>.
- 2. Only if you have sufficient information to answer the question with high confidence, provide your final answer within <answer> </answer> tags.

Additionally, select an image compression factor larger than 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. You must provide the next compression factor within <compression> </compression> tags (e.g., <compression>1.1</compression>).

#### Output format:

- 1. Reasoning: state what you found in the image.
- 2. <search>...</search> or <answer>...</answer>
- 3. <compression>...</compression>

Figure 12: Prompt template used by SKILL0 for the Search-based QA task environment.

#### Prompt of SKILL0 on WebShop

You are an expert autonomous agent operating in the WebShop e-commerce environment. {skill_context} Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history} You are now at step {current_step} and your current observation is: {current_observation}. Your admissible actions of the current situation are: [ {available_actions} ]. The image contains the full history:

- • Past observations are the page content after each action
- • Past actions are inside <action>...</action>

Now it’s your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you’ve finished your reasoning, you should choose an admissible action for the current step and present it within <action> </action> tags. Additionally, select an image compression factor larger than 1.0 for the next image. Higher compression lowers cost, but too much compression harms image quality. You must provide the next compression factor within <compression> </compression> tags (e.g., <compression>1.1</compression>). Output format:

- 1. <think>Reasoning: state what you found in the image and which action best advances the goal.</think>
- 2. <action>...</action>
- 3. <compression>...</compression>

Figure 13: Prompt template used by SKILL0 for the WebShop task environment.

#### Table 8: Representative Skills in SkillBank.

Skill Title Principle (Actionable Pattern) When to Apply skills/ALFWorld/general.md Systematic Exploration Search every plausible surface or container exactly

Anytime the goal count is not met and unexplored areas remain.

once before revisiting; prioritize unseen locations.

Immediate Acquisition As soon as a required object becomes visible and reachable, take it immediately before moving elsewhere.

Upon first visual confirmation of a goal-relevant object.

skills/ALFWorld/pick_and_place.md Grab When Seen Whenever a needed object is visible and reachable,

Upon first sight of an unheld object matching the goal specification.

immediately take it before moving elsewhere.

Place Before More Search When holding a goal object and the target location is known, navigate there and place it immediately.

While carrying a required object and the destination has been identified.

skills/ALFWorld/look_at_obj_in_light.md Switch Lamp On Issue the use desklamp command as soon as you

Upon arriving at a desklamp that is currently off.

reach it so the light condition is satisfied.

Grab Target First If the target is visible but the desklamp is not, take the target immediately to carry it to the lamp.

When the target is visible and not yet held, while desklamp location is unknown.

skills/ALFWorld/clean.md Phase-Ordered Plan Execute in fixed sequence: (1) locate & acquire, (2)

As soon as the goal specifies the object must be clean before placement.

clean at sink, (3) navigate, (4) place.

Sink First for Cleaning Upon holding the target, go straight to the nearest sink and issue the clean command.

Once the target is in hand and its required state is clean.

skills/ALFWorld/heat.md Secure Exact Target First Identify and pick up the exact object named in the goal

After spotting any candidate object, before opening or using appliances.

before interacting with the microwave.

Open Then Heat Upon reaching the microwave with the target in hand, open the door, place the object, then heat.

Immediately after navigating to the microwave with the target object held.

skills/ALFWorld/cool.md Prep Cooling Appliance Locate the fridge first and open it so it is ready before

As soon as the fridge comes into view or right after acquiring the target object.

or immediately after grabbing the target.

Enforce Cooling Before Placement

Do not place the target object in its final location until a cooling action has been successfully executed.

When holding the correct object and before any placement action is attempted.

skills/Search/general.md Decompose Then Search Break the question into minimal sub-questions and

Any complex or multi-hop question requiring multiple intermediate facts.

handle each with its own targeted query before synthesizing.

Exit When Evidence Is Solid

Stop issuing further queries once clear, corroborated evidence is found; avoid premature termination.

After each read step—answer only if confidence is justified, otherwise refine search.

skills/Search/direct_retrieval.md Isolate Core Query Strip the question to its key entity plus sought fact and

At the start of any direct-retrieval task.

search exactly that pair first.

Evidence-Bound Answer Only state an answer explicitly supported by retrieved text; continue searching rather than guess.

Before finalizing any factoid answer.

skills/Search/multi_hop_reasoning.md Targeted Sequential Searches

Issue separate, focused searches for each sub-question instead of one broad query.

After decomposition, when distinct pieces of information must be collected individually.

Collect-Then-Compare Retrieve concrete values for all items before performing any comparison or conclusion.

For comparative tasks involving dates, places, or quantitative attributes.

skills/Search/entity_attribute_lookup.md Direct Attribute Query Include both the full entity name and target attribute

Whenever the entity’s full, unambiguous name is provided in the question.

in the first search to surface authoritative results.

Two-Source Cross-Check Confirm the attribute in at least two independent, reputable sources to avoid hallucinations.

After the first plausible answer appears or when the attribute seems uncommon or uncertain.

skills/Search/compare.md Parallel Attribute Lookup Independently retrieve the identical attribute for each

After identifying entities and the comparison attribute.

entity via separate, focused searches.

Normalize Before Comparing

Convert retrieved values to a common comparable form before judging equality or ordering.

After gathering each entity’s attribute but before drawing any conclusion.

#### Table 9: Representative Skills in SkillBank for WebShop.

Skill Title Principle (Actionable Pattern) When to Apply skills/WebShop/general.md

Prioritize Core Keywords Include product type, 1–2 key functional attributes, and hard constraints (price, size, color) in the search query; omit secondary descriptors to avoid overconstraining.

Before issuing the first search or when refining an over-specific query that yields few results.

Scan Before You Click Read product titles, thumbnails, and prices in the results list to ensure the item plausibly meets core constraints before opening it.

On any search results page when deciding which product link to open next.

Set Mandatory Variants Always select required variant options (size, color, capacity) before evaluating price or purchasing.

After confirming the product type matches and before any purchase action.

Purchase Decisively Once all constraints are confirmed on a variant and price fits, execute Buy Now without unnecessary further navigation.

After validating every constraint on the current product variant.

skills/WebShop/apparel.md Focus Key Query Include only product type plus must-have attributes

Before issuing or refining any search query for apparel items.

(gender, garment, fabric, fit, price cap) in the search; drop minor terms to widen relevant results.

Check Variant Price After choosing size and color, verify the displayed price for that specific variant is within budget; abandon if it exceeds the limit.

Immediately after variant selection and before clicking Buy Now.

skills/WebShop/electronics.md Constraint-Rich Search Pack product type plus every mandatory attribute (fea-

When starting a new product hunt or refining after poor results.

tures, color, size, price cap) into the initial search string to surface only highly relevant electronics items.

Bail on Mismatch Fast If variant clicks do not update product details to the required spec, use Back and seek another item instead of retrying the same option.

When repeated option clicks leave title or specs unchanged or incompatible.

skills/WebShop/footwear.md Verify Key Features Open the product description or specs to explicitly con-

Immediately after opening a product page, before selecting variants or buying.

firm required functional attributes (e.g., slip-resistant, rubber sole) before purchasing.

Exit Non-Matches Fast If a product clearly violates any hard constraint, back out immediately instead of toggling options or repeating identical searches.

Upon noticing missing attributes, wrong category, unavailable size, or over-budget price.

skills/WebShop/beauty_health.md Feature-Led Click Open the first result whose title explicitly mentions the

After search results appear and at least one headline contains the core feature term.

key functional attribute to maximize match likelihood.

Minimal Path Purchase Once a product fully satisfies all constraints, proceed directly to Buy Now without extra browsing to reduce error risk.

When all user requirements are confirmed on the current product page.

skills/WebShop/home_decor.md Use Variant Selectors Systematically choose color, size, shape, and other

After landing on a suitable product that offers configurable attributes.

visible variants, confirming each selection updates the listing to match constraints and price.

Open Details For Hidden Specs

Expand Description or Features tabs to confirm lessvisible requirements (material, washability, printing method) not obvious from titles.

When any user constraint is not directly visible in the main listing or variant options.

skills/WebShop/accessories.md Explicit Variant Selection Always click and visibly confirm the exact color, size,

After opening a product that offers multiple variant options.

or pattern variant instead of assuming the default matches the request.

Post-Config Price Audit After selecting all variants, re-confirm the displayed price is within budget; if exceeded, backtrack and search for alternatives.

Just before initiating checkout or adding to cart.

