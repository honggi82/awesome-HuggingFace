# arXiv:2606.17628v1[cs.CL]16Jun2026

## OPD-Evolver: Cultivating Holistic Agent Evolver via On-Policy Distillation

Guibin Zhang† Xun Xu† Yanwei Yue Zikun Su Wangchunshu Zhou Xiaobin Hu‡ Shuicheng Yan‡

‡ Corresponding Authors † Equal Contribution LV-NUS Lab FDU PKU Bytedance Inc.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

GitHub Hugging Face

### Abstract

Memory has become a standard substrate for self-evolving agents, yet retaining experience is not the same as learning how to evolve through it. Existing memory agents can store trajectories, retrieve reflections, or accumulate skills, but often lack the holistic competence to select useful experience, act on it, write reusable knowledge, and maintain a growing repository. We introduce OPD-Evolver, a slow-fast co-evolution framework that cultivates such an agent evolver through on-policy self-distillation. In the fast loop, OPD-Evolver interacts with a four-level memory hierarchy to read, use, write, and maintain experience for rapid test-time evolution. In the slow loop, outcome-calibrated memory attribution and privileged hindsight distill these four abilities into the deployable policy. Across multidomain benchmarks, OPD-Evolver surpasses memory systems such as ReasoningBank by up to 11.5%, and training-based methods such as Skill0 by ∼ 5.8%. Further analysis shows that OPD-Evolver internalizes high-value experience and memory management, enabling OPD-Evolver-9B to challenge giant counterparts such as QWEN3.5-397B-A17B and STEP-3.5-FLASH, pointing beyond memoryaugmented agents toward genuinely qualified agent evolvers.

### 1 Introduction

What defines a self-evolving agent? A natural answer is memory, since preserving past trajectories, retrieving prior lessons, and reusing accumulated skills seem to provide the material basis for improvement over time (Hu et al., 2025; Fang et al., 2025a). This intuition has made memory an indispensable component of modern agentic foundation models (Team et al., 2025; MiniMax, 2026) and agent systems (Pan et al., 2026; Zhou et al., 2026), especially in interactive environments where failures reveal latent constraints, successful rollouts

expose reusable strategies, and rewards ground behavior in environmental feedback (Wu et al., 2024; Zheng et al., 2025b; He et al., 2026). Yet memory is only the substrate of self-evolution, not its definition. Despite the proliferation of memory systems (Zhang et al., 2025a; Ouyang et al., 2025), skill libraries (Zheng et al., 2025a; Lu et al., 2026; Xia et al., 2026), reflection mechanisms (Liang

- et al., 2024), and self-improvement pipelines (Wu

- et al., 2025b; Wei et al., 2025), the notion of “selfevolution” remains under-specified: many agents can retain experience or expose it to the prompt, while far fewer can convert it into sustained behavioral improvement. In this work, we define an agent evolver as an agent that systematically transforms interaction history and feedback into persistent improvements in future behavior.

Existing work has approached this objective from diverse perspectives. Memory-augmented agents store trajectories, reflections, tips, or lessons and inject them into later prompts (Shinn et al., 2023; Zhao et al., 2024; Chhikara et al., 2025). Skill-augmented agents, which we regard as a structured instantiation of memory, distill experience into reusable strategies, tools, or procedures (Xia

- et al., 2026; Zhang et al., 2026a; Lu et al., 2026; Shi et al., 2026). Other methods more directly parameterize experience by training on collected trajectories through supervised fine-tuning (SFT), reinforcement learning (RL), or on-policy distillation (Liu et al., 2026; Zhang et al., 2025b; Xi

- et al., 2025). Despite this progress, most methods optimize only one fragment of the evolution process, such as retrieving experience, using it in context, distilling it into parameters, or engineering the memory architecture. Two challenges remain central: task rewards provide relatively direct supervision for execution, but not for memory selection, writing, or long-term management; and how to train these coupled abilities within one policy without making them interfere remains underexplored.

➣ Objective Formulation. We define the qualified agent evolver as an agent with four coupled competencies: experience selection, experiencegrounded execution, experience writing, and experience management.

As a result, existing agents may improve within a particular setting, yet still fall short of becoming holistic agent evolvers.

We characterize a qualified agent evolver through four coupled capabilities: ❶ experience selection identifies useful memories from a growing and noisy repository; ❷ experience-grounded execution converts selected experience into effective multi-turn actions; ❸ experience writing extracts reusable knowledge from new trajectories and feedback; and ❹ experience management scores, consolidates, updates, and retires memories over time. These competencies cannot be safely separated: weak selection amplifies retrieval noise, weak execution leaves the agent dependent on prompt-time guidance, weak writing pollutes future memory, and weak management causes long-term degradation (Zhang et al., 2025a). This motivates our central research question:

➣ Practical Solution. We propose OPD-Evolver, a fast-slow on-policy self-distillation framework that converts task outcomes and privileged hindsight into supervision for selection, execution, writing, and memory management.

➣ Experimental Validation. Evaluations on four self-evolving benchmarks show that OPD-Evolver4B/9B surpass contemporary memory systems and remain competitive with giant counterparts such as QWEN3.5-397B and STEP-3.5-196B.

### 2 Related Work

Self-Evolving Agents. Recent self-evolving agents can be organized around the experience lifecycle defined in this work. For ❶ experience selection, prior systems use embedding retrieval (Tan

|How can we train an agent to acquire the holistic competence of evolving through experience, so as to become a qualified agent evolver?|
|---|

- et al., 2025), utility scoring, learned routers, or policy-based ranking (Li et al., 2026a; Xu et al.,

- 2025) to surface useful memories from growing repositories. For ❷ experience-grounded execution, memory- and skill-augmented agents condition policies on retrieved memories (Ferraz et al., 2026; Wang et al., 2026d), while SFT/RL-based methods internalize experience into policy parameters (Zhang et al., 2025c; Liu et al., 2026). For ❸ experience writing, existing methods distill trajectories into reflections (Wu et al., 2025a; Fang et al.,

- 2025b), reasoning memories, procedural tips (Yue

et al., 2026), executable tools (Zhao et al., 2026a), or reusable skills (Xia et al., 2026; Zhang et al.,

- 2026b). For ❹ experience management, recent systems study scoring, consolidation (Yue et al.,

- 2026), forgetting, and architecture-level adaptation (Zhang et al., 2025a). These works demonstrate the value of experience reuse, but most optimize one or two stages of the lifecycle, whereas our goal is to train a truly holistic agent evolver.

To this end, we introduce OPD-Evolver, a slowfast co-evolution framework where the fast loop lets the agent evolve through online interaction with environments and memory, while the slow loop distills these interaction-derived behaviors into intrinsic evolver capabilities via on-policy selfdistillation. In the ♣ fast evolution loop, the agent operates over a four-level memory substrate of trajectories, tips, skills, and tools: it selects taskrelevant memories before execution, acts with them, then writes and periodically maintains memories from trajectories, rewards, and feedback. Across this stream, OPD-Evolver estimates an outcomecalibrated memory attribution signal, converting delayed task outcomes into supervision for what should be selected, used, written, and preserved.

The ♠ slow evolution loop then turns this calibrated stream into on-policy self-distillation targets for the same policy. The privileged teacher observes attribution-enriched evidence unavailable at deployment, including candidate memory value, trajectory snippets, future utility of written memories, and repository-level diagnostics. Distilling the student’s own rollout states against this hindsight supervision jointly shapes selection, execution, writing, and management, enabling OPD-Evolver to acquire transferable lifecycle-level competence for self-improvement. Our contributions can be summarized as:

On-Policy Distillation. On-policy distillation (OPD) trains a student on its own visited states by querying a teacher for dense supervision, reducing the train-inference mismatch of off-policy distillation (Hinton et al., 2015; Agarwal et al., 2024; Li et al., 2026b; Song and Zheng, 2026). Recent studies have widely explored OPD for improving agent execution, including mathematical reasoning (Wu et al., 2026), knowledge QA (Ye et al., 2026), and

###### Selection Execution Update Manage

Episodic Manage

Updated Memory

[Figure 5]

Environment

[Figure 6]

Traj. Experience Tip Experience Skill Experience

[Figure 7]

Lookup("web")

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Web • OS • Embodied ...

[Figure 12]

Progressiveconsolidation

Traj 1,3

Tool 2,5,9

[Figure 13]

[Figure 14]

[Figure 15]

Apply update New memories

Agent Execution Loop

FastLoopSlowloop

Feedback

[Figure 16]

Merge(Tool 2/5)

[Figure 17]

Execute

Tool artifacts

[Figure 18]

[Figure 19]

[Figure 20]

... ...

[Figure 21]

Merged tool

Experience Selection

[Figure 22]

[Figure 23]

Trajectory: ~ ~ ... Tip: [No tips generated]

Lookup("numpy")

[Figure 24]

... ...

[Figure 25]

Tips 8,9,10

Tip 8 (selected)

[Figure 26]

Skill: Pandasdraw.ioforforreshufflevisual policy: Tool:

[Figure 27]

[Figure 28]

Agent Evolver

[Figure 29]

Avoid clicking ads...

Delete(Tip 8/9)

[Figure 30]

Reflect

Skill 19 (selected)

Equip

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

Finished

###### Holistic Evolution Capability Distillation

Teacher's Privileged Context

Selected high-quality memories

Demo: Management process

0.9 0.7 0.2

0.9 0.7 Demonstrate good

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

experience examplers Lookup ~ Del ~ Merge ...

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Experience Update

Experience Internalization

Experience Consolidation

Experience Selection

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

select filter

[Figure 57]

[Figure 58]

[Figure 59]

Student Internalization

[Figure 60]

+ rollout& execute

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

update

[Figure 73]

[Figure 74]

Figure 1: (Top) The fast loop lets the agent interact with environments and a four-level memory hierarchy; (Down) the slow loop converts outcome-calibrated hindsight into on-policy self-distillation signals for OPD-Evolver.

tool-use tasks (Zhong et al., 2026). In contrast, our work uses OPD not merely to strengthen execution, but to jointly cultivate the four capabilities required by a holistic agent evolver.

### 3 Methodology

#### 3.1 Problem Formulation

We consider a lifelong agent that faces a stream of interactive tasks rather than a single isolated query. At round t, an agent parameterized by θ receives a task xt ∼ D, interacts with an environment through observations ot,i and actions at,i, and obtains environmental feedback rt,i. The resulting episode is denoted by

τt = {(ot,i,at,i,rt,i)}Hi=1t , Rt =

Ht

rt,i (1)

i=1

where Ht is the episode horizon and Rt is the task return. The per-step signal rt,i may be sparse; when only terminal feedback is available, we place it at the final step. Unlike standard evaluation that discards τt, our agent maintains an evolving repository Mt of reusable experience, so future behavior depends on how history is selected, used, written, and maintained. We call such an agent an agent evolver if it realizes a closed transition from the

current task-memory state (xt,Mt) to both task behavior and an updated repository. Concretely, it first retrieves a noisy candidate set Ct and selects a compact context St:

Ct = Ret(xt,Mt), St = Selθ(xt,Ct) (2)

where Ret is a non-parametric retriever, Selθ is the agent’s selection behavior, and St ⊆ Ct. The same agent πθ then executes with St and, after observing the outcome, emits an experience update ∆t:

τt ∼ πθ(· | xt,St), ∆t ∼ πθ(· | xt,τt,Rt,St), Mt+1 = Mt ⊕θ ∆t

(3) where ∆t denotes the memories and retention decisions produced from the episode, and ⊕θ denotes the agent-controlled repository update. Thus, selfevolution depends on the full experience lifecycle, not merely a memory buffer: Ct must contain relevant evidence, St must be actionable, ∆t reusable, and Mt+1 useful as it grows. The streaming objective is to maximize performance along the task stream while preserving the future utility of the repository:

1 T

max

liminf

T→∞

θ

T

E[Rt + λU(Mt+1)], (4)

t=1

where U(Mt+1) denotes the downstream usefulness of the updated repository and λ controls how much the agent values future evolvability. The objective of OPD-Evolver is to train this closed loop so the agent attains high Rt while improving future repositories. The following sections instantiate this objective with a fast loop (▷ Section 3.2) for online experience-conditioned interaction and a slow loop (▷ Section 3.3) that distills privileged hindsight into the unified evolver.

- 3.2 Fast Evolution Loop The fast loop runs online for each task and performs test-time evolution without changing θ. Its

role is to turn Mt into a small, actionable context before execution, and to turn new episodes back into reusable experiences afterward.

Hierarchical experience memory. We organize OPD-Evolver’s Mt into four complementary tiers. Let L = {traj,tip,skill,tool} denote the tier set, where each Mℓt stores memories of type ℓ:

Mt = Mtrajt ∪ Mtipt ∪ Mskillt ∪ Mtoolt (5)

where Mtrajt preserves episodic evidence, Mtipt stores local warnings or heuristics, Mskillt abstracts reusable procedures, and Mtoolt stores executable command or code templates. The hierarchy is useful because agent experience has different reuse granularity: full trajectories are faithful but verbose, while skills and tools are compact but must be distilled from enough evidence.

Retrieval and selection. Given the task xt and optional environment metadata et, we form a query zt = [xt;et] and retrieve top-K candidates from each tier by embedding similarity:

Ctℓ = TopKm∈Mℓ

Ctℓ (6)

sim(ϕ(zt), ϕ(m)) , Ct =

t

ℓ∈L

where ϕ(·) is the embedding encoder, sim(·,·) is cosine similarity, and TopK returns the K memories with largest scores. Retrieval alone is intentionally high-recall: it surfaces potentially useful memories, but also brings stale, redundant, or taskmismatched items. The selector then compresses Ct into the memory context shown to the agent:

Stℓ = Selℓθ(xt, Ctℓ), St =

Stℓ, ct = Fmt(St) (7)

ℓ∈L

where Selℓθ selects memories Stℓ from the tier-ℓ memory pool Ctℓ, and Fmt renders selected memory items into the textual prompt context. The compact context ct is the operational interface between external memory and the acting policy.

Experience-grounded execution and writing. During rollout, the actor conditions on ct together with the task and interaction history:

at,i ∼ πθ(· | xt,ot,≤i,at,<i,ct) (8)

where ot,≤i and at,<i denote the partial interaction history before action at,i. After the episode ends, the same interaction becomes evidence for future tasks. The same agent decides which tiers should be updated and how many memories to write to each tier. Given xt, τt, Rt, and St, it produces ∆t = {∆ℓt}ℓ∈L, where each ∆ℓt may be empty or contain multiple new memories:

ℓ t

∆ℓt = {mℓt,j}n

j=1, Mℓt+1 = Mℓt ⊕θ ∆ℓt (9)

The count nℓt is chosen by the agent, so the fast loop can write to any subset of the four tiers rather than following a fixed memory schema. In addition to per-task writing, the agent periodically performs repository maintenance. Let Ht = {(τj,Rj,∆j)}j<t denote the logged interaction history. Every Q tasks, it enters a multi-turn interaction with HqQ and the current repository, using lookup plus merge(mi,mj) and delete(mi):

ηq ∼ πθ(· | MqQ,HqQ,T ), M+qQ = MqQ ⊕θ ηq

(10) where q indexes maintenance rounds, qQ denotes the task index at which the q-th maintenance is triggered, T = {lookup,merge,delete}, ηq is the maintenance tool-call trajectory, and M+qQ is the repository after maintenance. In parallel, the loop logs retrieved candidates, selected memories, outcomes, created memory ids, and maintenance actions. These logs are not needed for immediate execution, but they provide the hindsight evidence used by the slow loop to estimate memory value and distill the evolver.

#### 3.3 Slow Evolution Loop

The fast loop lets the agent accumulate experience, but not necessarily use it well. An arbitrary agent does not know which memories help, how to ground execution in them, what deserves to become memory, or when a repository update helps future tasks. The slow loop trains these coupled abilities by turning the agent’s own interaction stream into supervision. The only trustworthy external signal available after each task is the environment feedback Rt; the central question is how to propagate this scalar outcome back to selection, execution, writing, and maintenance decisions.

Outcome-calibrated attribution. For each memory m, we estimate its value only on tasks where m was actually retrieved, so the comparison is candidate-controlled rather than confounded by irrelevant tasks. Let g(t) denote the task group of round t, and define Ωg(m) = {t : g(t) = g, m ∈ Ct \ St}} and Ω+g (m) = {t : g(t) = g, m ∈ St}. The outcome-calibrated attribution of m is

A(m) =

g

ρg(m)(Et∈Ω+

g (m)[Rt] − Et∈Ωg(m)[Rt]) (11)

where ρg(m) = |Ω+g (m)|/(|Ω+g (m)| + |Ωg(m)|) downweights unreliable selections; empty groups are omitted. We then convert attribution into a memory score

V (m) = αℓ(m) γ(m) A(m), γ(m) = 1 −

1 1 + N+(m)

,

(12)

where ℓ(m) is the tier of m, αℓ(m) is a tier prior, N+(m) = |Ω+g (m)|, and γ(m) is a confidence factor. This calibration converts delayed environment feedback into dense hindsight labels: memories with high V (m) become evidence for what should have been selected, used, and preserved, while low-value memories expose noise that the evolver should learn to ignore.

Unified hindsight self-distillation. We use the calibrated stream to train the same agentic evolver under a single on-policy distillation principle. Let K = {sel,act,write,maint}. For each lifecycle decision k, the student sees only the public input zk, while the teacher additionally sees a privileged hindsight view hk:

(ztsel, hselt ) = (xt, Ct), {(m, V (m))}m∈Ct , (ztact, hactt ) = xt, (St+, τt+) ,

(ztwrite, hwritet ) = (xt, τt, Rt, St), {( ˜m, V ( ˜m))}m˜∈∆t , (zqmaint, hmaintq ) = (MqQ, HqQ, T ), Dqmem

(13)

where St+ = {m ∈ St : V (m) > 0} denotes valuable selected memories, τt+ is a successful trajectory from the same task group, and Dqmem = {(m,V (m),γ(m),ν(m))}m∈MqQ ∪ {κ(mi,mj)}i,j contains memory value, confidence, usage statistic ν(m), and redundancy score κ(mi,mj). Thus, ❶ selection sees the value of every retrieved candidate; ❷ execution internalizes useful selected memories by asking the student to solve without memory; ❸ writing sees which produced memories later become valuable; and ❹

maintenance sees calibrated diagnostics for producing merge/delete tool-call trajectories. For each (zk,hk), the student first samples an on-policy output yˆk under the deployment condition, and the teacher is evaluated on the same student prefixes:

yˆk ∼ pSθ (· | zk), pSθ,n = pSθ (· | zk,yˆ<nk ), pTθ,n¯ = pTθ¯(· | zk,hk,yˆ<nk )

(14)

where pSθ is the student policy, pTθ¯ is the same evolver under privileged conditioning with stop-

gradient parameters θ¯, and yˆ<nk is the studentgenerated prefix. Let δk,n denote the corresponding token-level discrepancy:

δk,n = Dtok sg[pTθ,n¯ ] ∥ pSθ,n (15)

where Dtok is full-vocabulary token-level KL divergence, and sg[·] blocks gradients through the teacher distribution. The unified slow-loop objective is the token-level distillation loss over studentvisited prefixes:

Lk

1 Lk

E(zk,hk)∼dkπ

Lslow(θ) =

δk,n (16)

θ yˆk∼pSθ (·|zk)

n=1

k∈K

where dkπ

is the on-policy distribution of decision inputs for lifecycle decision k. Selection, execution, writing, and maintenance are therefore supervised as four views of one experience lifecycle: the teacher explains what the agent should have selected, how it should have acted with useful experience, and what it should have written or retained after observing the outcome. After distillation, only the student-facing behavior is deployed back into the fast loop, so the agent can exercise these evolver abilities without privileged feedback at test time.

θ

### 4 Experiments

4.1 Experiment Setup Training setup. We train OPD-Evolver on heterogeneous agentic experience from Agent World Model (Wang et al., 2026c), nvidia/NemotronTerminal-Corpus (Pi et al., 2026), and EnvScaler (Song et al., 2026), which are disjoint from all evaluation benchmarks (see Section D). Unless specified otherwise, we use QWEN3-4B-INSTRUCT-2507 (Yang et al., 2025) and QWEN3.5-9B (Qwen, 2026) as backbones, QWEN3-EMBEDDING-0.6B for retrieval, retrieve 50 candidates, set Q = 30, keep at most 20 memories in privileged teacher contexts, filter supervision with minimum score 0.01. More parameter setups are detailed in Section D.

- Table 1: Main results across self-evolving agent benchmarks. Each cell reports the official task success metric for the corresponding benchmark subset; higher is better. For AMA-Bench, CI: Causal Inference, SU: State Updating, SA: State Updating. Best and second-best mark the top two results within each backbone group.

LifelongAgentBench MemoryArena AMA-Bench InterCode DB OS Math Physics CI SU SA Bash CTF SQL

LLM Method

QWEN3.5-397B-A17B 86.00 63.00 3.98 4.65 57.21 58.73 51.41 51.34 56.00 62.74 STEP-3.5-FLASH (196B) 81.00 58.00 1.98 2.33 49.50 46.99 48.88 44.20 48.00 59.87

No Memory 65.00 36.50 2.40 2.33 24.83 27.67 29.73 30.36 26.00 38.85 ExpeL 62.50 34.50 1.98 1.74 26.17 24.88 28.49 30.80 33.00 34.08 AWM 63.00 36.00 1.69 3.49 28.36 30.14 33.49 27.68 33.00 38.54 Cheatsheet 69.00 38.50 2.97 1.74 25.34 30.91 30.37 33.04 25.00 37.90 Memp 73.00 41.00 4.66 0.00 32.38 28.44 31.81 31.70 27.00 43.31 ReasoningBank 70.00 38.00 3.81 1.16 27.35 29.68 30.97 31.25 25.00 43.95 EvolveR 71.50 46.50 1.27 2.91 26.51 24.42 30.01 33.93 31.00 44.90 MemEvolve 72.00 44.00 1.84 2.33 29.53 31.53 32.73 33.93 29.00 44.59 OPD-Evolver-4B 74.00 49.50 5.51 4.07 32.89 34.93 34.90 36.16 34.00 45.86

QWEN3-4B

No Memory 75.50 52.50 5.51 5.23 40.10 47.14 45.63 41.52 44.00 55.73 ExpeL 74.00 46.50 8.19 8.72 41.11 43.74 44.67 40.18 50.00 52.23 AWM 72.50 44.00 7.34 11.05 43.96 48.38 48.00 39.73 49.00 51.59 Cheatsheet 79.50 52.50 6.64 5.23 40.77 49.30 46.19 41.96 47.00 56.05 Memp 82.00 56.00 4.80 6.40 45.81 47.14 46.59 44.20 50.00 58.28 ReasoningBank 80.50 55.00 4.94 6.98 42.28 48.38 47.04 43.75 45.00 57.96 EvolveR 82.50 59.50 6.07 7.56 41.78 42.81 46.11 44.64 52.00 60.51 MemEvolve 81.00 61.00 4.24 9.88 44.30 49.77 47.20 45.98 53.00 61.15 OPD-Evolver-9B 84.50 65.00 10.88 11.63 47.32 53.94 52.92 49.55 57.00 64.01

QWEN3.5-9B

- 2025a); and ■ training-based methods: SFT, GRPO (Shao et al., 2024), Skill0 (Lu et al., 2026), MemRL (Zhang et al., 2026c) and Complementary RL (Muhtar et al., 2026). For fairness, all memorybased methods, including OPD-Evolver, start evaluation with an empty experience repository and accumulate memory only from the evaluation stream (see details in Section C.1).

Evaluation benchmarks. We evaluate on five benchmarks: LifelongAgentBench (Zheng et al.,

- 2025b) with database (DB) and operating system (OS); MemoryArena (He et al., 2026) with Math and Physics subsets; AMA-Bench (Zhao et al., 2026b) with Causal Inference, State Updating, and State Abstraction; InterCode (Yang et al., 2023) with Bash, CTF, and SQL subsets; and embodied environment MiniHack (Samvelyan et al., 2021). Benchmark details are in Section C.

Algorithm 1: Workflow of OPD-Evolver.

Input: Task stream {xt}; memory repository M0;

evolver policy πθ; maintenance period Q Output: Deployable evolver πθS and evolved

repository M /* Fast loop: online experience use */

- 1 for each task xt do

- 2 Form query zt from task and environment
- 3 for ℓ ∈ L do

- 4 Retrieve Ctℓ from Mℓt and select Stℓ ⊆ Ctℓ
- 5 St ← ℓ∈L Stℓ, ct ← Fmt(St)
- 6 Roll out πθ with ct to obtain (τt, Rt)
- 7 for ℓ ∈ L do

- 8 Write ∆ℓt and update Mℓt+1 ← Mℓt ⊕θ ∆ℓt
- 9 if t mod Q = 0 then

- 10 q ← t/Q
- 11 Produce maintenance trajectory ηq with {lookup, merge, delete} and update M+qQ
- 12 Log (Ct, St, Rt, ∆t) and any ηq /* Slow loop: unified distill */
- 13 for each logged memory m do

- 14 Estimate V (m) from logged outcomes
- 15 for k ∈ {sel, act, write, maint} do

- 16 Construct (zk, hk), sample yˆk ∼ pSθ (· | zk), and minimize Lslow// OPD
- 17 return Qualified agent evolver πθS

#### 4.2 Main Results

Comparison with Memory Systems We first compare OPD-Evolver with seven mainstream selfevolving memory systems as well as several larger models in Table 1. OPD-Evolver achieves the best result on all 10 subsets among same-backbone memory baselines for both 4B and 9B, with clear gains over the strongest memory baseline on OS (65.00% vs. 61.00%), AMA-SA (52.92% vs. 48.00%), and InterCode-CTF (57.00% vs. 53.00%) for 9B. Notably, OPD-Evolver-9B challenges its

Baselines. We compare with ■ memoryaugmented agents: ExpeL (Zhao et al., 2024), AWM (Wang et al., 2024), Cheatsheet (Suzgun et al., 2025), MemP (Fang et al., 2025c), ReasoningBank (Ouyang et al., 2025), EvolveR (Wu et al., 2025a), and MemEvolve (Zhang et al.,

- Table 2: Comparison with training-based agent improvement methods. Each cell reports the official task success metric.

Method

MiniHack InterCode Room Maze KeyRoom Bash CTF SQL

Vanilla 80.39 19.61 0.00 55.73 44.00 41.25 SFT 82.35 17.65 0.00 59.87 49.00 44.64 GRPO 100.00 23.53 3.92 63.69 58.00 47.77 Skill0 94.12 25.49 3.92 62.10 54.00 47.32 MemRL 96.08 19.61 1.96 61.15 50.00 44.20 Complementary RL 96.88 20.85 5.16 63.20 55.00 48.10 OPD-Evolver 98.04 27.45 9.80 64.01 59.00 49.55

Table 3: Ablation study on InterCode (Bash/CTF/SQL) with OPD-Evolver-4B. “Writing Distill.” denotes the exclusion of self-distilling experience writing capability.

Variant Bash CTF SQL OPD-Evolver-4B 36.16 34.00 45.86

w/o Slow Evolution 32.14 28.00 39.17 w/o Memory Attribution 31.20 26.69 38.50 w/o Selection 35.27 32.00 42.04 w/o Writing Distill. 34.38 29.00 41.08 w/o Maintenance 35.30 30.10 43.51

much larger counterparts: OPD-Evolver-9B exceeds STEP-3.5-FLASH (196B) on 9/10 subsets and surpasses QWEN3.5-397B-A17B on 6/10 subsets, including AMA-SA (52.92% vs. 51.41%), InterCode-CTF (57.00% vs. 56.00%), and SQL

- (64.01% vs. 62.74%), showing that lifecycle-level evolution can make a compact agent competitive with far larger counterparts.

Comparison with Training-based Methods Table 2 shows that OPD-Evolver achieves the best average performance among training-based methods, winning on 5/6 subsets. Compared with GRPO, OPD-Evolver improves Maze (27.45% vs. 23.53%), KeyRoom (9.80% vs. 3.92%), Bash

- (64.01% vs. 63.69%), CTF (59.00% vs. 58.00%), and SQL (49.55% vs. 47.77%). Against Complementary RL, OPD-Evolver further improves the hardest MiniHack subsets, including Maze (27.45% vs. 20.85%) and KeyRoom (9.80% vs. 5.16%). It also consistently outperforms MemRL on both MiniHack and InterCode, with a 5.35point gain on SQL. These results suggest that OPD-Evolver learns beyond task-level reward fitting, yielding a more transferable mechanism for selecting, internalizing, and reusing experience.

4.3 Ablation Study We ablate five components of OPD-Evolver in

- Table 3: ❶ w/o Slow Evolution removes onpolicy distillation, ❷ w/o Memory Attribution replaces outcome-calibrated attribution with a simpler memory-frequency calibrated score, ❸ w/o Selection directly chooses the top-5 memories by similarity, ❹ w/o Writing Distillation removes supervision for reusable memory writing, and ❺ w/o Maintenance disables repository update training. All variants degrade InterCode performance, with the largest drop from removing memory attribution (average 38.67%→32.13%; Bash/CTF/SQL decrease by 4.96%, 7.31%, and 7.36%). Removing slow evolution is similarly harmful (average

1.0

0.8

Memoryscore

0.6

0.4

0.2

0.0

Base OPD

Base OPD CTF

Base OPD Bash

SQL

Figure 2: Distribution of calibrated memory scores for memories selected by the original QWEN3.5-9B and OPD-Evolver-9B after selection distillation.

- 33.10%), showing that privileged hindsight must be distilled into the deployable policy. Selection, writing, and maintenance also matter: similarity-only selection reduces SQL from 45.86% to 42.04%, and removing writing distillation lowers CTF from
- 34.00% to 29.00%. Together, these results show that calibrated attribution, learned selection, memory writing, and repository maintenance must be trained jointly for reliable self-evolution.

#### 4.4 Framework Analysis

Impact of Selection Distillation. We compare the memory scores of items selected by the original 9B backbone and the trained OPD-Evolver-9B selector. As shown in Figure 2, selection distillation consistently raises the median selected-memory score, from 0.66/0.69/0.66 to 0.79/0.76/0.76 on SQL/CTF/Bash. The lower quartile also moves upward from 0.50 to above 0.62 across the three subsets, suggesting that OPD-Evolver reduces lowutility retrieval noise rather than merely shifting a few high-scoring outliers.

Impact of Writing Distillation We next examine whether the trained evolver writes memories with higher future utility. Figure 3 compares the calibrated score distribution of memories produced by vanilla QWEN3.5-9B and by OPD-Evolver9B. Writing distillation increases the median score from 0.80/0.82/0.82 to 0.91/0.90/0.89 on

1.0

0.9

Memoryscore

0.8

0.7

0.6

0.5

Base OPD

Base OPD CTF

Base OPD Bash

SQL

- Figure 3: Distribution of calibrated memory scores for memories written by the original QWEN3.5-9B and OPD-Evolver-9B after writing distillation.

25 30 40 45 55 58

Task success

8

12

16

Steps

9B-SQL

9B-CTF

4B-SQL 9B-Bash

4B-CTF 4B-Bash

34 38 48

9 10.5

13.5 14.5

9B 4B

- Figure 4: Experience internalization effect. Each arrow starts from the vanilla backbone and ends at the corresponding OPD-Evolver policy; rightward movement indicates higher task success and upward movement in the inverted axis indicates fewer steps.

SQL/CTF/Bash, and lifts the lower quartile to 0.83 or higher. This tighter high-score distribution indicates that OPD-Evolver writes memories that are not only better formatted, but more reliably useful for downstream tasks.

Impact of Experience Internalization Finally, we test whether slow-loop distillation internalizes useful experience into the policy. Figure 4 compares each vanilla backbone with its trained OPD-Evolver counterpart (w/o external M; merely for task execution) in terms of task success and execution steps. Every arrow moves toward higher success and fewer steps: OPD-Evolver raises success by about 3–4 points for 4B and 3– 7 points for 9B on Bash/CTF/SQL, while reducing trajectories by up to 2.5 steps. This suggests that OPD-Evolver converts high-value memories into more direct and efficient behavior, rather than merely retrieving better context at inference time.

4.5 Case Study

- Figure 5 illustrates how slow-loop training makes both memory selection and memory writing more precise. For the LifelongAgentBench-OS task, the vanilla model selects broad directory and configuration memories, while OPD-Evolver keeps only the modification-log skill and permission tip that directly support creating files, setting modes, and

###### Experience selection case (LifelongAgent - OS)

[Figure 75]

Create a directory '/reports', generate ﬁve ﬁles named report1.txt to report5.txt with content 'Report X' (where X is 1-5), set their permissions to 640, and list

Task them sorted by modiﬁcation time in 'ﬁle_list.txt' with permissions 644. Retrieved candidate memories

Selected by vanilla model

[Figure 76]

[Figure 77]

- Skill 1: directory_permission_setup

[Figure 78]

- Skill 2: find_backup_modify_log Tip 1: about_directory_permissions

[Figure 79]

- Skill 1 Tool 2 Tip 1

[Figure 80]

[Figure 81]

[Figure 82]

- Skill 2 Tip 1

[Figure 83]

Tool 1 Selected

[Figure 84]

Selected by OPD-Evolver

[Figure 85]

[Figure 86]

- Tool 1: directory_setup_helper

[Figure 87]

- Tool 2: config_file_helper

[Figure 88]

Selected

###### Experience writing case (MiniHack - room)

[Figure 89]

Solve MiniHack environment MiniHack-Room-5x5-v0 with seed 16. Navigate the map, collect/use required items, handle doors or rivers, and reach the goal.

Written by vanilla model

###### Execution trajectory (failed)

Tip: Avoid no_action; verify goal; preview moves

[Figure 90]

Moved east to explore

Started in a small room

Continued moving east

| | |
|---|---|
| | |

[Figure 91]

Tool: Action validator

Turned south toward stairs

Adjusted eastward position

Written by OPD-Evolver

| | |
|---|---|
| | |

Tip: Exploring adjacent is insufficient!

[Figure 92]

Moved south to bottom row

Stood left of stairs

Submitted failed

Figure 5: Case study of experience selection and writing in OPD-Evolver. (Top) on LifelongAgentBench-OS, OPD-Evolver selects a compact task-matched subset from retrieved memories. (Bottom) on MiniHack-Room, OPD-Evolver writes a more causal and reusable memory from a failed trajectory.

producing the sorted file list. For the MiniHackRoom failure, the vanilla model writes generic advice such as verifying the goal and adding an action validator, but the trajectory shows a sharper error: the agent stopped after local east/south exploration without sufficiently expanding adjacent states around the stairs. OPD-Evolver therefore writes the reusable causal tip “Exploring adjacent is insufficient,” showing that the trained evolver learns not only to retrieve relevant experience, but also to turn failures into compact future-facing memory. More case studies are at Section F.

### 5 Conclusion

In this work, we present OPD-Evolver, a slowfast co-evolution framework that combines online memory-conditioned interaction with outcomecalibrated attribution and privileged on-policy selfdistillation. Across diverse evaluation benchmarks, OPD-Evolver consistently improves compact open models over memory-augmented and training-based baselines, showing that evolver abilities learned from heterogeneous streams can transfer beyond the training environments. Together, these results suggest a shift from building agents that merely store experience toward cultivating agents that can continuously transform experience into their own capacity for evolution.

### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang,

- Han Zhou, and Zaiqiao Meng. 2025a. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems. Preprint, arXiv:2508.07407.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2025b. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433.

Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2025c. Memp: Exploring agent procedural memory. Preprint, arXiv:2508.06433.

Thomas Palmeira Ferraz, Romain Deffayet, Vassilina Nikoulina, Hervé Déjean, and Stéphane Clinchant. 2026. Retrieval-augmented llm agents: Learning to learn from experience. Preprint, arXiv:2603.18272.

Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, TzuPing Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, and Alex Pentland. 2026. Memoryarena: Benchmarking agent memory in interdependent multi-session agentic tasks. Preprint, arXiv:2602.16313.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Preprint, arXiv:1503.02531.

Yuyang Hu, Shichun Liu, Yanwei Yue, Guibin Zhang, Boyang Liu, Fangyi Zhu, Jiahang Lin, Honglin Guo, Shihan Dou, Zhiheng Xi, Senjie Jin, Jiejun Tan, Yanbin Yin, Jiongnan Liu, Zeyu Zhang, Zhongxiang Sun, Yutao Zhu, Hao Sun, Boci Peng, and 28 others. 2025. Memory in the age of ai agents. Preprint, arXiv:2512.13564.

Chunyu Li, Jingyi Kang, Ding Chen, Mengyuan Zhang, Jiajun Shen, Bo Tang, Xuanhe Zhou, Feiyu Xiong, and Zhiyu Li. 2026a. Memreranker: Reasoningaware reranking for agent memory retrieval. Preprint, arXiv:2605.06132.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding.

2026b. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. Preprint, arXiv:2604.13016.

Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, and JingSong Yang. 2024. Self-evolving agents with reflective and memory-augmented abilities. arXiv preprint arXiv:2409.00872.

Weize Liu, Minghui Liu, Sy-Tuyen Ho, Souradip Chakraborty, Xiyao Wang, and Furong Huang. 2026. Agentic critical training. Preprint, arXiv:2603.08706.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Skill0: In-context agentic reinforcement learning for skill internalization. arXiv preprint arXiv:2604.02268.

MiniMax. 2026. MiniMax M2.7 - Model SelfImprovement, Driving Productivity Innovation Through Technological Breakthroughs — minimax.io. https://www.minimax.io/models/text/ m27.

Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, and 1 others. 2026. Complementary reinforcement learning. arXiv preprint arXiv:2603.17621.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2skill: Distill trajectory-local lessons into transferable agent skills. Preprint, arXiv:2603.25158.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2025. Reasoningbank: Scaling agent self-evolving with reasoning memory. Preprint, arXiv:2509.25140.

Linyue Pan, Lexiao Zou, Shuo Guo, Jingchen Ni, and Hai-Tao Zheng. 2026. Natural-language agent harnesses. Preprint, arXiv:2603.25723.

Renjie Pi, Grace Lam, Mohammad Shoeybi, Pooya Jannaty, Bryan Catanzaro, and Wei Ping. 2026. On data engineering for scaling llm terminal capabilities. Preprint, arXiv:2602.21193.

Qwen. 2026. Qwen Studio — qwen.ai. https://qwen. ai/blog?id=qwen3.5.

Mikayel Samvelyan, Robert Kirk, Vitaly Kurin, Jack Parker-Holder, Minqi Jiang, Eric Hambro, Fabio Petroni, Heinrich Küttler, Edward Grefenstette, and Tim Rocktäschel. 2021. Minihack the planet: A sandbox for open-ended reinforcement learning research. Preprint, arXiv:2109.13202.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Junhao Shen, Teng Zhang, Xiaoyan Zhao, and Hong Cheng. 2026. Dynamic skill lifecycle management for agentic reinforcement learning. Preprint, arXiv:2605.10923.

Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi GU, Xunliang Cai, Xiang Wang, and An Zhang. 2026. Skill1: Unified evolution of skill-augmented agents via reinforcement learning. Preprint, arXiv:2605.06130.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Mingyang Song and Mao Zheng. 2026. A survey of on-policy distillation for large language models. Preprint, arXiv:2604.00626.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2026. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis. Preprint, arXiv:2601.05808.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. 2025. Dynamic cheatsheet: Test-time learning with adaptive memory. Preprint, arXiv:2504.07952.

Zhen Tan, Jun Yan, I-Hung Hsu, Rujun Han, Zifeng Wang, Long T. Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, Anand Iyer, Tianlong Chen, Huan Liu, Chen-Yu Lee, and Tomas Pfister. 2025. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. Preprint, arXiv:2503.08026.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, and 150 others. 2025. Kimi k2: Open agentic intelligence. Preprint, arXiv:2507.20534.

Yash Vishe, Rohan Surana, Xunyi Jiang, Zihan Huang, Xintong Li, Nikki Lijing Kuang, Tong Yu, Ryan A. Rossi, Jingbo Shang, Julian McAuley, and Junda Wu. 2026. Skill-r1: Agent skill evolution via reinforcement learning. Preprint, arXiv:2605.09359.

- Hao Wang, Guozhi Wang, Han Xiao, Yufeng Zhou, Yue Pan, Jichao Wang, Ke Xu, Yafei Wen, Xiaohu Ruan, Xiaoxin Chen, and Honggang Qi. 2026a. Skill-sd: Skill-conditioned self-distillation for multi-turn llm agents. Preprint, arXiv:2604.10674.

Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. 2026b. Openclaw-rl: Train any agent simply by talking. Preprint, arXiv:2603.10165.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. 2026c. Agent world model: Infinity synthetic environments for agentic reinforcement learning. Preprint, arXiv:2602.10090.

Zhenting Wang, Huancheng Chen, Jiayun Wang, and Wei Wei. 2026d. Memex(rl): Scaling long-horizon llm agents via indexed experience memory. Preprint, arXiv:2603.04257.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2024. Agent workflow memory. Preprint, arXiv:2409.07429.

Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H Chi, and 1 others. 2025. Evo-memory: Benchmarking llm agent test-time learning with self-evolving memory. arXiv preprint arXiv:2511.20857.

Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, YunNung Chen, and Hung yi Lee. 2024. Streambench: Towards benchmarking continuous improvement of language agents. Preprint, arXiv:2406.08747.

Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, and Botian Shi. 2025a. Evolver: Self-evolving llm agents through an experience-driven lifecycle. Preprint, arXiv:2510.16079.

Rong Wu, Xiaoman Wang, Jianbiao Mei, Pinlong Cai, Daocheng Fu, Cheng Yang, Licheng Wen, Xuemeng Yang, Yufan Shen, Yuxin Wang, and 1 others. 2025b. Evolver: Self-evolving llm agents through an experience-driven lifecycle. arXiv preprint arXiv:2510.16079.

Yecheng Wu, Song Han, and Hai Cai. 2026. Lightning opd: Efficient post-training for large reasoning models with offline on-policy distillation. Preprint, arXiv:2604.13010.

Zhiheng Xi, Jixuan Huang, Chenyang Liao, Baodai Huang, Honglin Guo, Jiaqi Liu, Rui Zheng, Junjie Ye, Jiazheng Zhang, Wenxiang Chen, Wei He, Yiwen Ding, Guanyu Li, Zehui Chen, Zhengyin Du, Xuesong Yao, Yufei Xu, Jiecao Chen, Tao Gui, and 4 others. 2025. Agentgym-rl: Training llm agents for long-horizon decision making through multi-turn reinforcement learning. Preprint, arXiv:2509.08755.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, and 1 others. 2026. Skillrl: Evolving agents via recursive skillaugmented reinforcement learning. arXiv preprint arXiv:2602.08234.

Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, wenlin zhang, Yichao Wang, Huifeng Guo, Ruiming Tang, Xiangyu Zhao, Enhong Chen, and Tong Xu. 2025. From single to multi-granularity: Toward long-term memory association and selection of conversational agents. Preprint, arXiv:2505.19549.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

John Yang, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao. 2023. Intercode: Standardizing and benchmarking interactive coding with execution feedback. Preprint, arXiv:2306.14898.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026. On-policy context distillation for language models. Preprint, arXiv:2602.12275.

Yanwei Yue, Boci Peng, Xuanbo Fan, Jiaxin Guo, Qiankun Li, and Yan Zhang. 2026. Mem-t: Densifying rewards for long-horizon memory agents. Preprint, arXiv:2601.23014.

Guibin Zhang, Haotian Ren, Chong Zhan, Zhenhong Zhou, Junhao Wang, He Zhu, Wangchunshu Zhou, and Shuicheng Yan. 2025a. Memevolve: Metaevolution of agent memory systems. Preprint, arXiv:2512.18746.

Hanchen Zhang, Xiao Liu, Bowen Lv, Xueqiao Sun, Bohao Jing, Iat Long Iong, Zhenyu Hou, Zehan Qi, Hanyu Lai, Yifan Xu, Rui Lu, Hongning Wang, Jie Tang, and Yuxiao Dong. 2025b. Agentrl: Scaling agentic reinforcement learning with a multi-turn, multi-task framework. Preprint, arXiv:2510.04206.

Haozhen Zhang, Quanyu Long, Jianzhu Bao, Tao Feng, Weizhi Zhang, Haodong Yue, and Wenya Wang. 2026a. Memskill: Learning and evolving memory skills for self-evolving agents. arXiv preprint arXiv:2602.02474.

Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, Jian Xie, Yuxuan Sun, Boyu Gou, Qi Qi, Zihang Meng, Jianwei Yang, Ning Zhang, Xian Li, Ashish Shah, and 11 others. 2025c. Agent learning via early experience. Preprint, arXiv:2510.08558.

Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, Feiyu Xiong, Yutao Qi, Bo Tang, and Muning Wen. 2026b. Memrl: Selfevolving agents via runtime reinforcement learning on episodic memory. Preprint, arXiv:2601.03192.

Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, and 1 others. 2026c.

Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeng Zhang, and Chen Wei. 2026a. Pyvision-rl: Forging open agentic vision models via rl. Preprint, arXiv:2602.20739.

Yujie Zhao, Boqin Yuan, Junbo Huang, Haocheng Yuan, Zhongming Yu, Haozhou Xu, Lanxiang Hu, Abhilash Shankarampeta, Zimeng Huang, Wentao Ni, Yuandong Tian, and Jishen Zhao. 2026b. Ama-bench: Evaluating long-horizon memory for agentic applications. Preprint, arXiv:2602.22769.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. 2025a. Skillweaver: Web agents can self-improve by discovering and honing skills. Preprint, arXiv:2504.07079.

Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, and Qianli Ma. 2025b. Lifelongagentbench: Evaluating llm agents as lifelong learners. Preprint, arXiv:2505.11942.

Qiyong Zhong, Mao Zheng, Mingyang Song, Xin Lin, Jie Sun, Houcheng Jiang, Xiang Wang, and Junfeng Fang. 2026. Sod: Step-wise on-policy distillation for small language model agents. Preprint, arXiv:2605.07725.

Chenyu Zhou, Huacan Chai, Wenteng Chen, Zihan Guo, Rong Shan, Yuanyi Song, Tianyi Xu, Yingxuan Yang, Aofan Yu, Weiming Zhang, Congming Zheng, Jiachen Zhu, Zeyu Zheng, Zhuosheng Zhang, Xingyu Lou, Changwang Zhang, Zhihui Fu, Jun Wang, Weiwen Liu, and 2 others. 2026. Externalization in llm agents: A unified review of memory, skills, protocols and harness engineering. Preprint, arXiv:2604.08224.

### A Related Work Discussion

We further discuss several closely related contemporary works, many of which study skill-based experience internalization. SKILL0 (Lu et al., 2026) and OpenClaw-RL (Wang et al., 2026b) use RL or OPD to internalize agent execution behaviors, while our focus additionally includes how experience is selected, written, and maintained. SkillRL (Xia et al., 2026), Trace2Skill (Ni et al., 2026), and Skill-R1 (Vishe et al., 2026) mainly

improve skill or lesson generation, corresponding most closely to the experience-writing part of our lifecycle. Skill-SD (Wang et al., 2026a) uses skills as privileged context during distillation, emphasizing experience-guided execution. SLIM (Shen et al., 2026) and Skill1 (Shi et al., 2026) take broader views of skill lifecycle management. In comparison, OPD-Evolver studies a unified evolver policy over multiple experience forms, including trajectories, tips, skills, and tools, and evaluates its cross-domain generalization across heterogeneous agent benchmarks.

### B Potential Risks

We do not observe direct evidence that OPD-Evolver introduces qualitatively new social risks beyond those already associated with LLM-based agents. However, because the framework improves an agent’s ability to reuse experience and maintain memory, standard concerns around unsafe tool use, privacy leakage, biased behavior, and unintended task generalization still require careful attention. In practice, deployments should follow the same safeguards expected for agentic LLM systems, including controlled environments, memory auditing, access restrictions, and human oversight when actions may affect real users or external systems.

### C Benchmark Details

This section provides additional details on the benchmarks used for evaluation, including their subsets, test-set sizes, environment interfaces, and action spaces.

#### LifelongAgentBench. LifelongAgent-

Bench (Zheng et al., 2025b) is a multi-domain lifelong agent benchmark. We evaluate on its database (DB) and operating system (OS) subsets, with 100 test tasks for each subset. The DB subset consists of MySQL tasks, where the agent receives a natural-language instruction together with database context and acts through SQL execution followed by final submission. The OS subset consists of Ubuntu-container tasks, where the agent receives a natural-language instruction and interacts with the environment through bash commands followed by final submission. These two subsets evaluate whether the agent can reuse memory for long-horizon executable tasks that require precise environment operations.

MemoryArena. MemoryArena (He et al., 2026) evaluates memory over multi-session reasoning tasks. We evaluate on the Math and Physics subsets, which contain 354 and 86 test tasks, respectively. Each task provides shared background information and a sequence of question-answer sessions. The agent must answer the current question while preserving and reusing information from prior sessions. Unlike interactive control benchmarks, MemoryArena does not require live environment actions; instead, it tests whether the agent can maintain, retrieve, and apply useful memory across sessions.

AMA-Bench. AMA-Bench (Zhao et al., 2026b) evaluates long-horizon memory over completed agent trajectories. We use 208 episodes, containing 2,496 question-answer pairs in total. Following the benchmark taxonomy, the QA pairs are divided into four categories: Recall (839 pairs), Causal Inference (596 pairs), State Updating (647 pairs), and State Abstraction (414 pairs). In our evaluation, we focus on the memory-intensive Causal Inference, State Updating, and State Abstraction categories. Each example contains an action-observation trajectory and a question about the trajectory. Causal Inference requires the agent to reason about why events occurred, State Updating requires tracking changes across the trajectory, and State Abstraction requires summarizing or abstracting the relevant state. The agent does not control a live environment in AMA-Bench; it reads the trajectory and produces an answer, optionally using retrieved memory.

InterCode. InterCode (Yang et al., 2023) provides interactive coding benchmarks in isolated execution environments. We evaluate on the SQL, Bash, and CTF subsets, which contain 314, 224, and 100 test tasks, respectively. In SQL, the agent solves text-to-SQL tasks by issuing SQL queries against a database. In Bash, the agent solves natural-language shell tasks by executing bash commands. In CTF, the agent solves capture-the-flag challenges by executing commands and attempting flags. In all three subsets, the agent interacts through a JSON action interface and terminates with final submission. For Bash and SQL, grading is based on the output of the last executed command or query.

MiniHack. MiniHack (Samvelyan et al., 2021) is a suite of procedurally generated grid-world

navigation environments built on NetHack. We evaluate on Room, Maze, and KeyRoom, with 51 test tasks for each environment. Room requires the agent to reach the goal in a small open room; Maze requires systematic exploration in a larger maze; and KeyRoom requires the agent to locate a key, open a locked door, and then reach the goal. We render each environment as a text-based dungeon view using NetHack glyph conventions, such as @ for the agent, > for stairs or the goal, and . for floor tiles. At each step, the agent receives the ASCII map, the latest environment message, inventory text, and step/reward statistics. The agent acts through a unified JSON action interface with compass moves, object-interaction actions such as pickup, apply, open, and search, navigation actions such as climb_up and wait, and final submit.

#### C.1 Evaluation Protocol

For all memory-based methods, evaluation starts from an empty experience repository M0 = ∅. The agent then solves tasks sequentially in a streaming order, so its repository can grow only from trajectories, memories, and feedback observed during the current evaluation stream. After each task, the environment returns task-level feedback indicating whether the attempt succeeds or fails, which is used for memory update and attribution. The agent does not receive the ground-truth answer or an expert solution after the task; it only observes the same environment feedback available to all compared methods. For the training-based comparison in Table 2, we additionally train all compared methods on MiniHack using the same training data and evaluate them on the same test tasks to ensure a fair comparison.

### D Training Details

#### D.1 Training Data

We construct the training data from 7,000 interactive agent tasks: 3,000 from Agent World Model (AWM) (Wang et al., 2026c), 2,000 from nvidia/Nemotron-Terminal-Corpus (Pi et al., 2026), and 2,000 from EnvScaler (Song et al., 2026). These sources cover diverse executable and toolinteractive scenarios, exposing OPD-Evolver to heterogeneous trajectories for memory selection, memory writing, memory-conditioned execution, and repository maintenance.

Agent World Model (AWM). AWM provides synthetic multi-turn tool-use environments with executable, code-driven state transitions and databasebacked application scenarios. We use it to collect trajectories with reliable task-level outcomes and rich tool interactions.

#### nvidia/Nemotron-Terminal-Corpus.

Nemotron-Terminal-Corpus provides terminalstyle interaction data covering command execution, shell operations, and multi-step problem solving in textual environments. It complements AWM by exposing the agent to command-line workflows and environment feedback patterns.

EnvScaler. EnvScaler provides scalable toolinteractive environments for collecting diverse agent trajectories. We use it to broaden the training distribution beyond a single benchmark style and to expose OPD-Evolver to varied interaction structures.

Note that, for the training-based comparison in Table 2, we additionally train all compared methods on MiniHack using the same training data and evaluate them on the same test tasks to ensure a fair comparison.

#### D.2 Training Parameters

This section summarizes the training and rollout hyperparameters used in our experiments.

Shared Settings Unless otherwise specified, we use Qwen3.5-9B and Qwen3-4B backbones with bf16 precision. We set the maximum prompt length to 8192 tokens. For GRPO training, we use temperature 1.0 and top-p 1.0. For OPD-Evolver distillation data generation, we use temperature 1.0 and top-p 0.95. Across interactive environments, the maximum episode length is 40 steps unless otherwise specified.

SFT For the SFT baseline, we fine-tune the base model with LoRA. We use a learning rate of 1 × 10−5, per-device batch size 2, gradient accumulation 4, and train for 3 epochs. The LoRA rank is 32, the LoRA scaling factor is 64, and the LoRA dropout is 0.05.

GRPO For the GRPO baseline, we use the VERL framework and initialize the policy directly from the base model. We use group size 8 and rollout batch size 2. The learning rate is 1 × 10−6, and we train for one epoch. We apply actor-side KL regularization with coefficient 0.001, but do not include

the KL term in the reward. The reward is based on the environment cumulative reward. Invalid actions receive a penalty of −0.1, and premature submission receives a penalty of −0.2.

MemRL For MemRL, we follow its nonparametric runtime memory reinforcement learning setup, where learning is performed through episodic memory updates rather than model weight updates. We use Qwen3.5-9B as the LLM backend and Qwen3-Embedding-0.6B as the embedding model. We retrieve at most 3 memories per query. The memory construction, retrieval, and update strategies are proceduralization, query, and adjustment, respectively. For Q-value updates, we use a success reward of +1.0 and a failure reward of 0.0. Other MemRL reinforcementlearning hyperparameters are kept at the package defaults.

SkillZero For SkillZero, we use its skillconditioned GRPO training setup implemented with the VERL framework. Skill conditioning is enabled during training. We initialize SkillZero with a MiniHack-specific skill set and use a curriculum schedule that starts with all skills, switches to core skills after step 100, and removes skills after step 200. The rollout group size is 4, the learning rate is 1 × 10−6, and the batch size is 2. We train SkillZero for one GRPO epoch. Invalid actions receive a penalty of −0.1.

ComplementaryRL ComplementaryRL is implemented with the ROLL framework under MODE=complementaryrl. It uses a fully asynchronous actor-memory pipeline with three nonshared modules: a Qwen3.5-9B actor, a Qwen34B-Thinking-2507 memory model, and a Qwen34B-Thinking-2507 memory actor, with Qwen3Embedding-0.6B for retrieval. The actor is trained with GRPO, while the memory actor is trained with CISPO and REINFORCE-style advantage estimation. We use learning rate 1 × 10−6, rollout batch size 128, group size 1, gradient accumulation 64, and 512 training steps. Actor rollouts use temperature 0.99, top-p 0.99, and at most 4096 new tokens per step. The memory system stores trajectory memories, retrieves the top-1 memory by embedding similarity, uses FIFO updates, and caps the memory pool at 20,000 items. KL regularization is disabled, and episodes are limited to 80 actions per trajectory.

1.0

0.8

Memoryscore

0.6

0.4

0.2

0.0

Base OPD

Base OPD OS

DB

- Figure 6: Additional results for memory selection on LifelongAgentBench. Left: DB; right: OS.

Base OPD

0.0

0.2

0.4

0.6

0.8

1.0

Memoryscore

DB

Base OPD OS

- Figure 7: Additional results for memory writing on LifelongAgentBench. Left: DB; right: OS.

OPD-Evolver For OPD-Evolver, the memory pool contains four types of memories: skills, tips, tools, and trajectories. During teacher-side retrieval, we first retrieve 50 candidate memories and then select 20 memories for injection. For teacher memory filtering, we use a minimum score threshold of 0.01. For writer-side memory selection, we keep the top 30% of generated memories according to the memory score. We set the maintenance period to Q = 30 tasks. The evolver is optimized with the unified slow-loop objective described in Section 3.3.

The main experiments are conducted on a Linux server with 8 NVIDIA A800 (80G) GPUs.

### E More Results

Figures 6 and 7 report additional results on LifelongAgentBench. These experiments complement the main results by evaluating the selection and writing distillation effects on the lifelong DB and OS environments. OPD-Evolver consistently shifts the memory-score distributions upward compared with the vanilla model, indicating that slow-loop training improves both which memories are injected and what memories are written for future reuse.

Impact of Selection Distillation on LifelongAgentBench. Figure 6 compares the calibrated memory-score distributions of memories selected by the vanilla model and by OPD-Evolver on Life-

longAgentBench DB and OS tasks. On both environments, OPD-Evolver yields higher-score selected memories than the base model. The score distribution shifts upward, and the low-score region is reduced, showing that the trained selector is less likely to inject broad, noisy, or weakly related memories. This supports the claim that selection distillation improves not only the amount of retrieved memory, but also the task relevance of the injected memory context.

Impact of Writing Distillation on LifelongAgentBench. Figure 7 evaluates whether the trained writer produces memories with higher future utility on LifelongAgentBench. Compared with the vanilla writer, OPD-Evolver produces memories whose calibrated scores are higher and more concentrated on both DB and OS tasks. The reduction of low-score memories suggests that the trained writer is less prone to storing generic or misleading advice, and instead writes compact memories that better capture the causal error patterns and taskspecific constraints needed for future attempts. Together with the main results, these LifelongAgentBench experiments show that slow-loop training generalizes beyond the primary benchmark settings and improves both memory selection and memory writing in long-horizon agent environments.

### F Case Study

Figures 8 to 10 provide qualitative examples of how slow-loop training improves the three components of the memory evolution pipeline: execution, memory selection, and memory writing. Across InterCode-SQL, LifelongAgentBench-OS, and MiniHack, the vanilla models often either overgeneralize from previous experience or fail to localize the immediate cause of failure. In contrast, OPD-Evolver produces more task-specific behavior: it retrieves memories that match the current task, writes compact causal tips instead of generic advice, and executes actions that better satisfy environment-specific constraints.

- Figure 8 focuses on the EXECUTOR. The exam-

ples show that the vanilla executor can violate tasklevel constraints, stop before the true success condition is reached, or omit required side effects. In InterCode-SQL, it issues multiple statements in one action and continues schema exploration after noisy observations. In MiniHack, it submits while merely adjacent to the goal. In LifelongAgentBench-OS, it completes group membership but misses the re-

Task: sql_107 Names of math-course teachers

|Shared Prompt @<br><br>Divergence Steps Task + SQL patterns +<br><br>Retrieved Knowledge Constraint: ONE SQL per action| |
|---|---|
| | |

Step 1 First action?

Vanilla OPD

|execute: USE course_teach; SHOW TABLES; [X] multi-statement| |
|---|---|
| | |

|execute: USE course_teach; [OK] single statement| |
|---|---|
| | |

|Steps 2-9: schema noise Table Course doesn't exist command/obs mismatch| |
|---|---|
| | |

|Steps 2-9: clean exploration DESCRIBE teacher / course / course_arrange| |
|---|---|
| | |

Step 10 After error obs

Vanilla OPD

|execute: SELECT DISTINCT t.Name JOIN course_arrange + course WHERE c.Course = 'math'| |
|---|---|
| | |

|execute: DESCRIBE teacher; still exploring| |
|---|---|
| | |

Outcome: failure wrong JOIN at step 13

Outcome: success 4 teacher names -> submit

Task: MiniHack-Room-5x5v0 seed33 Reach goal >

|Shared Prompt @ Steps 8 & 12 Map @ agent near goal ><br><br>Strategy: submit only after success| |
|---|---|
| | |

Step 8 Map: ...@> Agent adjacent to goal

Vanilla OPD

|submit [X] premature - not on goal tile reward still 0| |
|---|---|
| | |

|move south [OK] continue pathfinding| |
|---|---|
| | |

Step 12 Map: ...@> Goal immediately east

Outcome: failure early submit

Vanilla OPD

|null action no recovery|
|---|

|move east [OK] step onto goal tile| |
|---|---|
| | |

Outcome: success path reach goal before submit

Task: os_0007 Group devteam + users + /devteam dir

|Shared Prompt @ Step 1 OS rules + action space No retrieved knowledge| |
|---|---|
| | |

Step 1 Single bash chain?

Vanilla OPD

|groupadd devteam useradd -M user1/2/3 gpasswd add/remove members [X] missing /devteam directory| |
|---|---|
| | |

|Same group/user/gpasswd chain + mkdir -p /devteam + chgrp devteam /devteam + chmod 770 /devteam| |
|---|---|
| | |

|Later: submit without group-only directory| |
|---|---|
| | |

|submit with complete state group dir 770 group-only| |
|---|---|
| | |

Outcome: incomplete missing /devteam artifact

Outcome: success all requirements met

- Figure 8: Case studies for the EXECUTOR. Top-left: InterCode-SQL; top-right: MiniHack-Room-5×5; bottom: LifelongAgentBench-OS.

quired group-owned directory. OPD-Evolver instead learns to respect the operational constraints of each environment: it separates SQL actions, continues navigation until the agent reaches the goal tile, and performs the missing directory creation and permission-setting operations.

- Figure 9 examines the SELECTOR. The vanilla

selector often injects memories that are superficially related but not causally useful for the current task, such as choosing a minimum-count SQL trajectory for a paragraph-counting query or a user/group setup workflow for a permission-editing OS task. In the MiniHack-Trap example, it even produces an empty injection after a JSON parse

Task: sql_27 Doc ids, names + paragraph count each

|Retrieve Candidates 3 Skills - 3 Tips - 3 Tools - 3 Trajectories DB: cre_Doc_Template_Mgt| |
|---|---|
| | |

|Candidate Pool<br><br>SKILL_01 schema<br><br>exploration sim 0.609<br><br>SKILL_02 min/max<br><br><br>COUNT aggregation sim 0.605 TRAJ_01 fewestparagraphs MIN query sim 0.752 TOOL_01 find doc with fewest paragraphs sim 0.643| |
|---|---|
| | |

Selector Vanilla vs OPD?

Vanilla OPD

|Select: SKILL_01 + SKILL_02 TIP_03 DESCRIBE first TOOL_02 schema inspector TRAJ_01 MIN-paragraph pattern [!] pulls MIN-task trajectory| |
|---|---|
| | |

|Select: SKILL_02 + SKILL_03 TIP_01 full-row extraction TIP_03 DESCRIBE verify Skip TOOL_01/02 MIN tools Skip all 3 trajectories [OK] list-all not MIN-one| |
|---|---|
| | |

|Inject to Executor MIN-pattern bias risk| |
|---|---|
| | |

|Inject to Executor GROUP BY all docs focus| |
|---|---|
| | |

OPD selection 4 items task-aligned no wrong sub-task traj

Vanilla selection 4 items incl. misleading traj

Task: os_0205 /reports - 5 files 640 file_list.txt 644

|Retrieve Candidates 3 Skills - 3 Tips - 3 Tools - 3 Trajectories All similarity 0.000| |
|---|---|
| | |

|Candidate Pool<br><br>SKILL_01 dir+group ownership<br>SKILL_02 sequential chmod workflow<br><br><br>TOOL_02 atomic mkdir+write+chmod TRAJ_01 /app greeting.txt appuser setup| |
|---|---|
| | |

Selector Vanilla vs OPD?

Vanilla OPD

|Select: SKILL_01 directory permissions TIP_03 ls -la verify TOOL_02 config_file_creator TRAJ_01 appuser/appgroup task [!] wrong task family| |
|---|---|
| | |

|Select: SKILL_02 batch chmod workflow TIP_03 verify with ls -la TOOL_02 adapt atomic chain Skip all trajectories [OK] batch file task not user/group| |
|---|---|
| | |

|Inject: user/group setup bias| |
|---|---|
| | |

|Inject: file batch + chmod focus| |
|---|---|
| | |

Vanilla: 4 items 1 misleading trajectory

OPD: 3 items no unrelated trajectories

Task: MiniHack-Room-Trap5x5-v0 seed22 Navigate trap room to goal

|Retrieve Candidates High similarity 0.72-0.95 3 Skills - 3 Tips - 3 Tools - 3 Trajectories| |
|---|---|
| | |

|Candidate Pool SKILL_01-03 navigation sim ~0.82<br><br>TIP_01 reward=1.0 goal signal sim 0.764<br>TIP_02 Random-5x5 step budget sim 0.754<br><br><br>TRAJ_01/02 Trap-5x5<br><br>seeds 5/14 sim 0.94+ TRAJ_03 Room-5x5 no Trap sim 0.935| |
|---|---|
| | |

Selector Vanilla vs OPD?

Vanilla OPD

|Select: SKILL_01 + SKILL_02 navigation TIP_01 reward goal signal only TOOL_01 movement strategy TRAJ_01 + TRAJ_02 Trap5x5 only Reject TRAJ_03 non-Trap Reject TIP_02 Random step budget| |
|---|---|
| | |

|Raw LLM output wrapped in markdown json fenced block [X] parse failure selected_skills: [] selected_tips: [] Empty injection| |
|---|---|
| | |

|Inject: nothing agent runs blind| |
|---|---|
| | |

|Inject: curated 6 items environment-matched| |
|---|---|
| | |

OPD: 2S + 1T + 1Tool + 2Traj Trap-specific

Vanilla: empty selection format error

- Figure 9: Case studies for the SELECTOR. Top-left: InterCode-SQL; top-right: LifelongAgentBench-OS; bottom: MiniHack-Trap.

failure. OPD-Evolver filters memory more aggressively and keeps only the skills, tips, tools, or trajectories that directly support the present problem. This indicates that slow-loop training improves not only whether memory is used, but also which memory is allowed to influence the next attempt.

Figure 10 analyzes the WRITER. The vanilla writer tends to convert failures into broad reusable

Task: sql_207 Year with most EUR gas use Outcome: failure

|Execution Trace USE debit_card_specializing Schema exploration attempts Wrong column Currency assumed DESCRIBE syntax error -> data rows| |
|---|---|
| | |

Writer Reflect on failure?

Vanilla OPD

|Extract broadly: 3 new_skills<br><br>- schema exploration pattern<br>- year aggregation<br>- database context switching 4 new_tips generic SQL pitfalls 2 new_tools<br>- aggregate_by_year_query<br>- discover_table_schema should_save_trajectory:<br><br><br>true| |
|---|---|
| | |

|Extract minimally: 0 new_skills 2 new_tips root-cause only<br><br>- verify column names before WHERE<br>- DESCRIBE returning rows = syntax error<br><br><br>0 new_tools trajectory_outcome: detailed failure analysis| |
|---|---|
| | |

|Memory Bank += verbose generic reusable patterns| |
|---|---|
| | |

|Memory Bank += precise failure-specific tips| |
|---|---|
| | |

Vanilla: 3S + 4T + 2Tools high volume generic

OPD: 0S + 2T + 0Tools targeted diagnosis

Task: os_0205 Same as selector case Outcome: failure reward 0

|Execution Trace Step 1: mkdir + for-loop create reports chmod 640 + ls -lt -> file_list.txt Step 2-3: verify + submit [X] file_list has absolute paths not bare names| |
|---|---|
| | |

Writer Reflect on failure?

Vanilla OPD

|Extract broadly: 2 new_skills<br><br>- batch file creation bash<br>- immediate verification 3 new_tips generic<br>- don't submit before verify<br>- default perms 644 not 640<br>- truncated logs hide state 1 new_tool<br><br><br>create_restricted_files should_save_trajectory: true| |
|---|---|
| | |

|Extract minimally: 0 new_skills 3 new_tips root-cause<br><br>- file_list bare filenames only<br>- cat file_list before submit<br>- verify Report N literal content<br><br><br>0 new_tools should_save_trajectory: true| |
|---|---|
| | |

|Memory: generic file-ops skills misses exact grading requirement| |
|---|---|
| | |

|Memory: exact file_list format rule actionable for retry| |
|---|---|
| | |

OPD: 0S + 3T + 0Tools grading-spec aligned

Vanilla: 2S + 3T + 1Tool

Task: MiniHack-Room-5x5v0 seed16 Navigate to goal > Outcome: failure reward 0

|Execution Trace move actions toward goal no_action error at step 3 Premature submit attempts Map shows @ adjacent to > not on it| |
|---|---|
| | |

Writer Reflect on failure?

Vanilla OPD

|Extract broadly: 2 new_skills - navigate with move_up/down/left/right [X] wrong action API<br><br>- map parsing @ position 3 new_tips<br><br>- no_action invalid<br><br>- verify goal before submit<br>- preview map after move 1 new_tool<br><br><br>check_valid_actions [X] lists wrong action names| |
|---|---|
| | |

|Extract minimally: 0 new_skills 2 new_tips root-cause<br><br>- reward 1.0 only ON goal tile<br><br>adjacent != success<br><br>- use move+direction N/S/E/W<br><br><br>never no_action 0 new_tools should_save_trajectory: true| |
|---|---|
| | |

|Memory: incorrect MiniHack API move_up vs move+direction| |
|---|---|
| | |

|Memory: goal-tile + valid action format environment-accurate| |
|---|---|
| | |

Vanilla: 2S + 3T + 1Tool API errors in memory

OPD: 0S + 2T + 0Tools correct env semantics

Figure 10: Case studies for the WRITER. Top-left: InterCode-SQL; top-right: LifelongAgentBench-OS; bottom: MiniHack-Room-5×5.

advice, and in some cases even records incorrect environment APIs. OPD-Evolver instead writes narrower future-facing memories that target the root cause of the failure. In InterCode-SQL, it records the need to verify column names and interpret malformed schema queries. In LifelongAgentBenchOS, it records the grading-specific requirement to output bare filenames and verify the generated file list before submission. In MiniHack, it records the correct action interface and the fact that reward is obtained only by stepping onto the goal tile.

### G Information About Use Of AI Assistants

AI assistance was used only for writing polishing and auxiliary coding support in this work.

