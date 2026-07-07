[Figure 1]

[Figure 2]

ROLL 2026-06-30

## Complementary RL: Towards Efficient Experience-Driven Agent Learning

Dilxat Muhtar1,†, Jiashun Liu1,2,†, Wei Gao1,2, Weixun Wang1,∗, Shaopan Xiong1, Ju Huang1,∗, Siran Yang1, Wenbo Su1, Jiamang Wang1, Ling Pan2, Bo Zheng1

1Alibaba Group 2HKUST

## Abstract

Reinforcement Learning (RL) has emerged as a powerful paradigm for training LLM-based agents, yet remains limited by low sample efficiency, stemming not only from sparse outcome feedback but also from the agent’s inability to leverage prior experience across episodes. While augmenting agents with historical experience offers a promising remedy, existing approaches suffer from a critical weakness: the experience distilled from history is either stored statically or fail to coevolve with the improving actor, causing a progressive misalignment between the experience and the actor’s evolving capability that diminishes its utility over the course of training. Inspired by complementary learning systems in neuroscience, we present Complementary RL to achieve seamless co-evolution of an experience extractor and a policy actor within the RL optimization loop. Specifically, the actor is optimized via sparse outcome-based rewards, while the experience extractor is optimized according to whether its distilled experiences demonstrably contribute to the actor’s success, thereby evolving its experience management strategy in lockstep with the actor’s growing capabilities. Empirically, Complementary RL outperforms outcome-based agentic RL baselines that do not learn from experience, achieving 10% performance improvement in single-task scenarios and exhibits robust scalability in multi-task settings. These results establish Complementary RL as a paradigm for efficient experience-driven agent learning. a

# arXiv:2603.17621v2[cs.LG]27Jun2026

aWe release our training framework and training demo at here

[Figure 3]

###### Co-Evolution

[Figure 4]

[Figure 5]

[Figure 6]

Training Step #" ## #$ #%

[Figure 7]

| | | | | |
|---|---|---|---|---|
| | | | | |

[Figure 8]

[Figure 9]

[Figure 10]

Policy Actor

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Mutual Adaptation

Synchronized Growth

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Experience Extractor

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

Experience Bank

Patterns Rules Strategies Expert Knowledge

Figure 1: Complementary RL performance (left) and co-evolution paradigm (right).

###### 1 Introduction

Recent research has demonstrated the effectiveness of Reinforcement Learning (RL) in enhancing the agentic capabilities of Large Language Models (LLMs) (Jin et al., 2025; Dong et al., 2025; Xue et al., 2025). Despite this progress, outcome-based RL for LLMs-based agents remains limited by sample inefficiency. Policy updates rely solely on sparse reward signals (Shao et al., 2024; Li et al., 2023; Yu et al., 2025), which, while effective at optimizing task outcomes, provide no explicit signal for why a trajectory succeeded or failed throughout the multi-turn interaction process (Wang & Ammanabrolu, 2026). Consequently, the rich procedural information embedded in collected rollouts, such as effective behaviors, recoverable failure patterns, and critical decision points, is largely unexploited. This underutilization of these procedural information renders the agent’s learning process sample-inefficient (Zhang et al., 2026b).

To mitigate this inefficiency, a growing line of work explores how to leverage historical experience to increase the utilization of already-collected rollout data, therefore allowing the actor to learn fast (Silver & Sutton). Here, we define experience as structured textual knowledge distilled from raw trajectories, encompassing successful strategies, failure patterns, and generalizable decision rules. A direct approach distills experience through self-generated reflections and incorporates it as in-context guidance during training (Zhan et al., 2025). However, when the base model is weak or tasks are complex, self-reflection

†Equal contribution. ∗Corresponding authors: weixun.wwx@taobao.com; huangju.hj@alibaba-inc.com.

becomes unreliable, frequently producing hallucinations that corrupt rather than enrich the learning signal (Lin et al., 2025). To improve the reliability of experience used to guide the actor, some works focus on enhancing the quality of collected experience, either by maintaining auto-optimizing experience bank via specialized data structures (Qian et al., 2025; Ouyang et al., 2025) or by employing a dedicated experience model to distill and dynamically refine structured experience from actor interactions (Zhai et al., 2025; Zhang et al., 2025a; Xia et al., 2026; Yan et al., 2025). Others instead focus on designing multi-stage retrieval heuristics to surface the most valuable experience from the accumulated experience bank (Zhou et al., 2025; Zhang et al., 2026a).

Action

###### Environment

Despite the efforts to enable agents to learn from experience, prior works treat experience as a static resource, either maintaining fixed experience banks or employing non-adaptive experience extractors that progressively lag behind the actor’s evolving capabilities, producing increasingly misaligned experience as training advances. Such stale experience limits learning efficiency as the actor grows stronger (Figure 1 and Figure 3a). To improve the quality and relevance of experience throughout training, we argue that an RL algorithm for experience-driven agent training must satisfy three core design requirements: ❶ Actor-Extractor CoEvolution: the actor and experience extractor must mutually adapt throughout training, each continuously shaping the other toward greater capability; ❷ Experience Consolidation: the experience bank must be automatically constructed and maintained from trajectories, distilling transferable experience while resolving conflicts and redundancies; and ❸ Training-Distillation Coordination: actor training and experience distillation must be efficiently coordinated at scale without introducing blocking latency to actor training. Motivated by these requirements, in this paper we aim to answer:

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

Tasks

[Figure 30]

###### Action

[Figure 31]

𝒔𝟎

[Figure 32]

Agentic RL

Goal

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

###### Observation

[Figure 37]

[Figure 38]

[Figure 39]

Efficient exploration & fast learning

Guided by distilled experience from extractor

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Experience Extractor

Actor Policy

Tasks

[Figure 47]

Action

co-evolution

Goal

[Figure 48]

[Figure 49]

𝒔𝟎

Obs.

[Figure 50]

Complementary RL

Improving experience utility via actor feedback

[Figure 51]

Environment

Figure 2: Overview of Complementary RL.

###### Can we design a RL framework in which the policy actor and its experience extractor form a closed co-evolutionary loop, each continuously shaping the other toward better?

Interestingly, the human brain has long solved an analogous problem. Complementary Learning Systems (CLS) in neuroscience (O’Reilly et al., 2011) enable the brain to rapidly acquire new knowledge while preserving long-term structured representations through two complementary systems: the neocortex forms slow, structured long-term knowledge (analogous to the actor’s policy), while the hippocampus manages fast, episode-specific memories (analogous to generated experiences), consolidating valuable episodes via cortical feedback and replaying them to strengthen decision-making.

Motivated by CLS, we propose Complementary Reinforcement Learning (Complementary RL), a RL algorithm built around two complementary models: an actor that interacts with the environment and optimizes guided by distilled experience, and an experience extractor responsible for distilling and maintaining a continuously evolving experience bank. Both models are optimized via RL: the actor is trained using outcome-based rewards, while the extractor is optimized based on the utility of its distilled experience in facilitating the actor’s success (Figure 2). Through this mutual optimization, Complementary RL jointly meets the three requirements above: ❶ the actor and extractor form a closed co-evolutionary loop, where the extractor continuously refines experience to match the actor’s growing capability and the actor benefits from increasingly relevant guidance; ❷ the extractor distills experience from trajectories through structured addition, refining, and merging operations that automatically resolve conflicts and redundancies; and ❸ We introduce a dedicated asynchronous training framework with a centralized experience manager that decouples actor interaction from experience distillation and dual-model optimization, ensuring training efficiency without introducing additional blocking latency. In summary, our main contributions are as follows:

- 1. We propose Complementary RL, a paradigm that enables the co-evolution of a policy actor and an experience extractor during RL training, where the experience extractor continuously extracts and manages experience while the actor internalizes it to enable efficient policy improvement. (§2)
- 2. We develop an efficient training framework tailored for Complementary RL, featuring a fully asynchronous design with a centralized MemoryManager that enables experience management at scale. (§3, §B)
- 3. Through extensive empirical evaluation, we demonstrate the effectiveness of Complementary RL, and share key insights and lessons learned throughout the process. (§4, §B.2)

###### 2 Methodology

###### 2.1 Problem Formulation

We consider an LLM-based actor πθ operating in an interactive environment E, formalized as a Markov Decision Process (MDP) ⟨S, A, T , R⟩ (Silver & Veness, 2010), where S, A are the state and action spaces, T : S × A → S is the transition function, and R : S × A → R is the reward function. At the beginning of each episode, the agent receives a task goal g. At each timestep t, it receives an observation st ∈ S, produces an internal reasoning trace by reflecting on the current observation and interaction history, and then decides an action at ∼ πθ(· | s≤t, g) (Yao et al., 2022). The environment then transitions to the next state st+1. An episode terminates upon task completion or upon reaching Tmax steps, yielding a outcome reward R ∈ {0,1}. The objective is to maximize the expected success rate across diverse tasks and environments:

J (θ) = EE,g,τ∼πθ [R(τ)] , (1) where τ = (s0, a0, s1, a1, . . . , sT) denotes the full interaction trajectory.

The formulation above treats each trajectory τ in isolation, optimizing πθ solely from binary outcome rewards, leaving the rich behavioral information embedded in each trajectory unexploited. A natural path toward greater learning efficiency is to distill structured experience m from past trajectories, store it in an experience bank M, and retrieve relevant entries to guide πθ in subsequent episodes (Silver & Sutton; Ouyang et al., 2025; Zhang et al., 2026a; Zhai et al., 2025). This augments the original objective (Equation 1) to:

J (θ) = EE,g,m∼M,τ∼πθ(·|g,m) [R(τ)] . (2)

###### 2.2 From Static to Co-Evolutionary Experience

Having formalized the learning-from-experience framework, we now turn to answering a practical question: how should the experience bank M be constructed and maintained to maximally benefit actor learning? We analyze three design choices through a pilot study on the MiniHack Room (Samvelyan et al., 2021) 1 : (1) Baseline: learning without experience; (2) Offline Exp.: M is pre-constructed from prior collected trajectories using an external extractor (Zhai et al., 2025) and remains static during RL training; (3) Static Online Exp.: M is dynamically maintained by a fixed experience extractor πϕ during actor learning. Figure 3a shows that while offline experience provides an initial performance boost, its benefit decays progressively over the course of training. Similarly, static online experience yields only marginal gains over the baseline, suggesting that simply collecting online experience without co-evolving the extractor is insufficient. We attribute this to a distributional misalignment: a static M cannot track the evolving state-action distribution of πθ, causing the guidance to become stale and counterproductive. This insight motivates us to the co-evolutionary paradigm where πϕ and πθ are jointly optimized. In this framework, improved policies generate higher-quality trajectories that refine M, thereby providing more effective guidance for subsequent policy optimization. We formalize this mutually reinforcing mechanism as Complementary RL.

###### 2.3 Complementary Reinforcement Learning

Algorithm Design for Experience Extractor In Complementary RL, the experience bank M is maintained by an experience extractor πϕ, which is jointly optimized with the actor πθ. At the end of each episode, the extractor distills an experience entry m ∼ πϕ(· | g, τ) conditioned on the task goal g and the full interaction trace τ. We track how m influences subsequent actor behavior by assigning a binary reward r(m) ∈ {−1, +1} based on the outcome of the trajectory it guided. These experience-reward pairs

1Room-Ultimate-5x5-v0: minihack-room

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 52]| | |
| | |Experienc|e-Free|
| | |Experienc|e-Guided|
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| |[Figure 53]| | |
| | | | |
| |Experienc Experienc|e-Free e-Guided| |
| | | | |

1.0

1.0

[Figure 54]

0.9

0.8

[Figure 55]

[Figure 56]

###### SuccessRate

###### SuccessRate

###### SuccessRate

###### SuccessRate

0.8

0.8

[Figure 57]

0.8

0.6

[Figure 58]

0.7

0.6

0.6

0.4

0.6

Baseline

0.5

Offline Exp.

0.4

0.2

Experience-Free

Static Online Exp.

0.4

0.4

Experience-Guided

Comp. RL (Ours)

0.3

0.2

0 20 40 60

0 50 100 150

0 50 100 150

0 50 100 150

Training Steps

Training Steps

Training Steps

Training Steps

(a) Exp. Comparison

(b) w/o Group Split

(c) Cross-group Adv.

(d) Subgroup Adv.

- Figure 3: (a) Co-evolving the actor and experience extractor consistently outperforms static alternatives. (b–d) Ablation study on advantage estimation designs for the actor. The Exp. denotes experience.

are accumulated into a training batch Bϕ = {(gi, τi, mi,r(mi))}Oi=1, upon which πϕ is optimized via the CISPO objective (Chen et al., 2025):





|mi|)

O

IS high

sg([ρi,t]1+ε

1−εISlow ) Aˆi log πϕ(mi,t | gi, τi, mi,<t)

### ∑

### ∑

t=1

i=1

, (3)

JCISPO(ϕ) = E

 

 

B

### ∑

|mi|

i=1

where ρi,t = ππϕ(mi,t|gi,τi,mi,<t)

ϕold(mi,t|gi,τi,mi,<t) is the token-level importance sampling (IS) ratio clipped to [1 − εISlow, 1 + εIShigh]. sg(·) denotes the stop-gradient operation, and Aˆi = r(mi) − r¯ is the batch-level advantage, where r¯ denotes the mean reward over batch Bϕ, and |mi| denotes the number of tokens generated by πϕ for experience entry mi. We adopt CISPO instead of REINFORCE (Sutton et al., 1999) to ensure stable co-evolution: the clipping mechanism constrains the IS ratio, preventing excessive policy updates that could cause the experience distribution to shift abruptly while ensuring that the gradients of all tokens are not wasted.

Algorithm Design for Actor In practice, the actor πθ is usually optimized via the GRPO (Shao et al., 2024) objective, which maximizes the expected reward through group-relative advantage estimation over

K sampled trajectories {τi}kK=1 per (g, m):

K

1 K

min ρAˆ, clip (ρ,1 − ε,1 + ε) Aˆ , (4)

### ∑

JGRPO(θ) = E

k=1

where ρ = ππθ(τ|g,m)

θold(τ|g,m) is the sequence level IS ratio, Aˆ = (r(τ) − r¯)/σ is the group-normalized advantage, and ε is the clipping threshold.

However, we observe that when all interactions are conditioned on retrieved experience, the actor converges prematurely and lags behind the experience-guided setting (Figure 3b), suggesting that the actor fails to internalize experience into its own capabilities and instead develops an over-reliance on external guidance. Inspired by Zhai et al. (2025), we therefore partition the K rollouts evenly into two subgroups: experience-guided and experience-free. However, a critical issue arises when computing advantages across the two subgroups: the reward scales and variances differ between subgroups, causing advantage estimates to become biased and training to collapse (Figure 3c). To preserve signal integrity, we propose computing advantages within each subgroup, ensuring that relative performance is evaluated under consistent conditioning:

 1

  , (5)

Kc

1 Kc

JGRPOsplit (θ) = E

Lclip ρc, Aˆc

### 2 ∑

### ∑

k=1

c∈{m,∅}

where c ∈ {m,∅} indexes the subgroup with experience-guided and experience-free interactions, and Aˆc = (r(τc) − r¯c)/σc is normalized within subgroup c using its own mean r¯c and standard deviation σc. In practice, the two subgroups are of equal size Kc = K/2, which ensures balanced gradient contributions from both two subgroups and prevents either condition from dominating the training signal. Lclip(ρ, A) = min (ρA,clip(ρ,1 − ε,1 + ε)A) is the clipped surrogate loss. This condition-wise advantage estimation preserves the distinct learning signals of each condition and stabilizes training, yielding consistent improvement across both subgroups (Figure 3d).

###### Primary Training Loop

Observation 𝑠 Action 𝑎

[Figure 59]

[Figure 60]

[Figure 61]

Search & Ask

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Search Request

[Figure 66]

Experience Manager

[Figure 67]

𝑞

[Figure 68]

[Figure 69]

Experience 𝑚

[Figure 70]

ℋ

Policy Actor

𝜋

Experience 𝑚

[Figure 71]

𝑠 𝑎

###### Trajectory 𝝉 Policy Update

𝜋

Env.

Submit Distillation Request

Experience Managerℋ

[Figure 72]

[Figure 73]

[Figure 74]

Distillation Request Distillation Queue 𝒬 Put Request

Searching with Read Lock

[Figure 75]

Embedding Model 𝒇𝝍

|𝜏|𝑔|𝑜|𝑚|
|---|---|---|---|

[Figure 76]

Query Batching & Caching

|𝑞|
|---|

|𝑞|
|---|

Pull

[Figure 77]

|𝑞|
|---|

|𝑞|
|---|

[Figure 78]

Training Buffer 𝑩𝝓

𝑚 𝑣 𝑚 𝑣 𝑚 𝑣

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Parallel Search Worker

Add Update Return

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Write Lock

Experience Extractor 𝝅𝝓

Batch Size Reach

|[Figure 90]|
|---|

|[Figure 91]|
|---|

Experience Bank ℳ

Extractor Update 𝝅𝝓

Periodic Merge

###### Background Track

- Figure 4: Overview of the Complementary RL training infrastructure, where the actor and experience extractor are trained asynchronously and coordinated through a centralized experience manager.

###### 3 Training Framework

- 3.1 Overview

Complementary RL jointly optimizes the policy actor πθ and the experience extractor πϕ, where the two models are mutually dependent: πθ requires retrieved experience before each interaction, while πϕ depends on completed actor trajectories for distillation and receives training signals reflecting whether the experience it produced was beneficial. A na¨ıve implementation would serialize these dependencies, where after each batch of rollouts, actor training would block while waiting for experience distillation and πϕ optimization to complete, introducing synchronization barriers that cause significant resource idleness and degrade overall training throughput.

To eliminate this bottleneck, Complementary RL deliberately decouples rollout collection from experience distillation via a fully asynchronous design comprising a primary training loop and a background track, as illustrated in Figure 4. In the primary training loop, the actor πθ continuously interacts with the environment to collect rollouts and is optimized via outcome-based rewards. Concurrently, in the background track, the experience extractor πϕ processes completed trajectories, distills experience, and issues structured operations to maintain the experience bank M.

Although the two tracks run asynchronously, they remain tightly coupled: at the beginning of each episode, relevant experience is retrieved from M to condition πθ, and upon episode completion, regardless of success or failure, the full trajectory is forwarded to πϕ for distillation. Coordinating these interactions at scale, where hundreds of environments execute in parallel while sharing a single globally consistent M, requires careful concurrency management. To this end, we introduce a centralized Ex-

perienceManager H, which serves two coordinating roles: (1) Experience Consolidation: H maintains an internal queue Q to receive and schedule distillation requests, and manages all writes to M under a writer lock to prevent state conflicts (§3.2.1); (2) Experience Retrieval: H aggregates concurrent retrieval queries into micro-batches to maximize throughput, and distributes semantic search across parallel workers under a reader lock to enable concurrent reads (§3.2.2). Through H, Complementary RL achieves efficient experience management at scale, keeping the additional latency introduced to the actor training loop minimal.

In the following, we detail our infrastructure design for experience consolidation, retrieval, and coevolution of πθ and πϕ, with additional stabilization tricks deferred to Appendix B.

- 3.2 Experience Consolidation and Retrieval

###### 3.2.1 Experience Consolidation

Producer-Consumer Distillation Upon completion of each episode, regardless of outcome, the full interaction trace τ, together with the initial task goal g, the final outcome o ∈ {success,failure}, and the experience entry m ∈ M retrieved to guide the episode, are submitted to H as a distillation request. H maintains an internal queue Q to receive distillation requests from all parallel environments. A background process continuously dequeues pending requests and forwards them to the experience extractor πϕ for distillation.

For each distillation request R = (τ, g, o, m), πϕ reasons over the full interaction trace, the episode outcome, and how the retrieved experience m influenced the actor’s behavior, before issuing the following structured operations: Add a newly synthesized experience entry into M, Update the previously retrieved entry m, or Return without action when the episode yields no extractable insight. Upon receiving the issued operations from πϕ, H applies them to M under a writer lock, which temporarily suspends concurrent reads to prevent state conflicts. For each newly added experience entry m, it is first passed through an embedding model fψ to obtain its dense vector vm = fψ(m). The entry m, its embedding vm, and the generation prompt-response pair produced by πϕ are then jointly persisted to M, enabling both semantic retrieval and future evolving of πϕ.

Periodic Merge The above consolidation process treats each episode independently. However, in groupbased RL, multiple instances of the same task typically run in parallel, which can lead to redundant or conflicting experience entries being added to M. Such redundancy degrades the quality of semantic retrieval and consequently impairs the actor’s learning (Figure 5a). To mitigate this, we periodically trigger a Merge operation every several actor updates. Experiences in M are processed in chunks, each passed to πϕ with a structured prompt that instructs the model to analyze the semantic relationships among entries and decide which to retain, which to merge, and which to discard. The merged output is then carried forward and concatenated with the next chunk, forming a chunk-wise sliding process over the full M. This design bounds the context length presented to πϕ while ensuring all entries are considered, yielding a compact experience bank that benefits actor learning.

###### 3.2.2 Experience Retrieval

Query Batching and Parallel Search At the beginning of each episode, the environment submits a Search request to H using the task description as a query q. Rather than processing queries individually, H accumulates incoming queries into a waiting buffer until either a predefined batch size B or a maximum waiting time tmax is reached. Each query is then checked against an embedding cache C before invoking fψ, which is particularly effective in group-based RL training where many parallel environments share identical task descriptions. Cache misses are forwarded to fψ for batched embedding computation, yielding vq = fψ(q). The resulting embeddings are distributed via round-robin to one of W parallel search workers, each performing semantic similarity search over M under a reader lock, allowing concurrent reads while blocking writes. Finally, the most relevant experience entry m is then returned to the requesting environment. Through batching, caching, and parallel search, this design maximizes retrieval throughput while minimizing latency introduced to the actor’s environment interaction.

Search-and-Ask Using the task description alone as query q tends to retrieve the same experience entry m repeatedly, since parallel environments in group-based RL training often share identical task descriptions or differ only in environment-specific details such as map layouts (e.g., MiniHack (Samvelyan et al., 2021)). This reduces the utilization of M and limits the diversity of training signal available for optimizing πϕ. To address this, we introduce the search and ask tool, which allows πθ to actively query M at any decision step during environment interaction. When the actor invokes this tool, it constructs a context-aware query q′ by summarizing its current state and the difficulties it faces, and submits q′ to H for retrieval. If a relevant entry m is found, the pair (q′, m) is forwarded to πϕ, which refines m according to the actor’s specific situation before returning the result. This mechanism increases M utilization, enriches the training signal for πϕ, and enables the actor to obtain more targeted guidance aligned with its current situation at critical decision points, further improving learning efficiency (Figure 5b).

###### Ablation Study: Search & Ask

###### Ablation Study: Periodic Merge

1.0

[Figure 92]

[Figure 93]

[Figure 94]

SuccessRate

0.8

SuccessRate

0.8

[Figure 95]

[Figure 96]

[Figure 97]

0.6

0.6

Baseline

0.4

Baseline

w/o Search & Ask

w/o Merge

0.4

0.2

Comp. RL (Ours)

Comp. RL (Ours)

0 50 100 150

0 50 100 150

Training Steps

Training Steps

(a) w/o Merge

(b) w/o search and ask

Figure 5: Ablation on Merge and search and ask in MiniHack Room.

###### ALFWorld

###### SWE-Bench

###### WebShop

###### Minihack Room

1.0

ValidationScore

[Figure 98]

0.8

0.8

1.0

SuccessRate

SuccessRate

SuccessRate

[Figure 99]

[Figure 100]

[Figure 101]

0.8

0.4

0.8

0.6

0.6

0.0

0.4

0.6

0.4

0.4

0.2

0.4

Baseline

Baseline

Baseline

Baseline

0.2

Comp. RL (Ours)

Comp. RL (Ours)

Comp. RL (Ours)

Comp. RL (Ours)

0.8

0.0

0.2

0.0

0 25 50 75 100 125

0 25 50 75 100 125

0 50 100 150 200 250

0 50 100 150

Training Steps

Training Steps

Training Steps

Training Steps

Figure 6: Single-task evaluation scores across four different environments.

###### Minihack Room

###### SWE-Bench

###### WebShop

###### ALFWorld

25.0

27.5

| | | | | | | |
|---|---|---|---|---|---|---|
| | | |Bas Com|eline p. RL|(Our|s)|
| | | | | |[Figure 102]| |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | |B|aselin|e| |
| | |C|omp.|RL (O|urs)|
| |[Figure 103]| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | |B|aselin|e| | |
|---|---|---|---|---|---|---|
| | |C|omp.|RL (O|urs)| |
| | | | | | | |
| | | |[Figure 104]| | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

25.0

17.5

Baseline

[Figure 105]

22.5

25.0

22.5

Comp. RL (Ours)

NumActions

NumActions

15.0

NumActions

NumActions

20.0

22.5

17.5

12.5

20.0

15.0

20.0

10.0

17.5

12.5

7.5

17.5

10.0

15.0

5.0

7.5

15.0

12.5

5.0

2.5

12.5

10.0

2.5

0 25 50 75 100 125

0 25 50 75 100 125

0 50 100 150 200 250

0 25 50 75 100 125

Training Steps

Training Steps

Training Steps

Training Steps

Figure 7: Average number of actions per task.

- 3.3 Co-Evolution Training

The actor πθ is evolved following the objective described in Equation 5. For the evolution of πϕ, after each rollout collection step that yields a batch of trajectories T = {τi}iN=1 for training πθ, we extract the experience entry m that guided each trajectory τi and assign it a binary reward r(m) ∈ {−1,1} based on whether the corresponding episode succeeded. The prompt-response pair generated by πϕ to produce m is then stored in a training buffer Bϕ. Since multiple trajectories in T may share the same retrieved entry m, we treat each unique m as a single training sample and accumulate its rewards across all associated trajectories, assigning the average reward r¯(m) = |T1

m| ∑τ∈Tm r(m, τ), where Tm ⊆ T denotes the subset of trajectories guided by m. As a result, the number of unique training samples for πϕ may be smaller than defined batch size for πϕ, and a single rollout collection step may not suffice to fill Bϕ. We therefore accumulate samples across multiple rollout collection steps, and only trigger the optimization of πϕ once |Bϕ| reaches the required training batch size, as described in Equation 3. Crucially, πϕ and πθ are optimized on fully independent schedules, ensuring neither blocks nor interferes with the other throughout co-evolution training.

- 4 Experiments

- 4.1 Experimental Settings

We evaluate Complementary RL on four open-ended environments: MiniHack (Samvelyan et al., 2021), WebShop (Yao et al., 2023), ALFWorld (Shridhar et al., 2021), and SWE-Bench (Jimenez et al., 2024). During training, we track success rate on MiniHack and WebShop, and reward on held-out evaluation sets for ALFWorld and SWE-Bench. For a fair comparison of final performance, all methods are evaluated on fixed evaluation tasks for all environments. Detailed environment descriptions are provided in the Appendix C.1.

Without other specification, we use Qwen2.5-7B-Instruct (Qwen et al., 2025) as actor ϕθ and use Qwen34B-Thinking-2507 (Yang et al., 2025) as the experience extractor ϕϕ. For all of the comparison methods, we use the same hyperparameters for fail comparison, which we defer to Appendix C.2 for detail introduction.

###### 4.2 Main Result

Single-Task Training We first evaluate Complementary RL separately on each of the four tasks and compare it against baselines that do not leverage experience. We use Qwen3-4B-Instruct-2507 as the actor πθ for SWE-Bench in this experiment, while all other tasks follow the default settings described earlier.

As shown in Figure 6, Complementary RL consistently outperforms the baseline across all four tasks. In tasks requiring strategic exploration and environmental understanding, such as MiniHack Room and ALFWorld, Complementary RL achieves a 1.3× performance margin with notably better training stability. Moreover, on the challenging software engineering benchmark SWE-Bench, Complementary RL demonstrates faster improvement and achieves a +3.0% gain over the baseline. Furthermore, Figure 7 reveals

All Tasks (Avg)

MiniHack Room

WebShop

ALFWorld

1.00

1.0

1.0

1.0

| | |aseline Static O|nline E|xp.| | |
|---|---|---|---|---|---|---|
| | |Exp. On Comp. R|ly L (Ours|)| | |
| | | | | |[Figure 106]<br><br>[Figure 107]| |
| | | | |[Figure 108]<br><br>[Figure 109]| | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | |Ba|seline| | |
| | | |St Ex<br><br>|atic On p. Only|line Exp|[Figure 110]<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>.|
| | | |Co|mp. RL|(Ours)| |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | |[Figure 114]<br><br>[Figure 115]<br><br>[Figure 116]<br><br>|Bas Sta|eline tic Onl|ine Exp|.|
| |[Figure 117]| |Exp Com|. Only p. RL|(Ours)| |
| | | | | | | |

ValidationScore

RewardScore

[Figure 118]

SuccessRate

SuccessRate

0.75

[Figure 119]

0.8

0.8

0.5

[Figure 120]

0.50

0.6

0.6

0.0

Baseline

0.25

0.4

0.4

Static Online Exp.

0.5

Exp. Only

[Figure 121]

0.00

0.2

0.2

Comp. RL (Ours)

0 25 50 75 100 125

0 25 50 75 100 125

0 25 50 75 100 125

0 25 50 75 100 125

Training Steps

Training Steps

Training Steps

Training Steps

Figure 8: Multi-task training curves on overall and per-task performance.

that Complementary RL not only achieves higher success rates but also completes tasks more efficiently, requiring 1.5× fewer actions on MiniHack Room and 2× fewer actions on ALFWorld, demonstrating that distilled experience guides the actor toward more effective decision-making. Although Complementary RL exhibits an increasing number of actions on SWE-Bench, we find that this is because the agent takes more actions to fully complete tasks, thereby achieving a higher success rate, rather than submitting prematurely before a task is finished.

Multi-Task Training Instead of training each task separately, we jointly train on MiniHack Room, ALFWorld, and WebShop to investigate whether Complementary RL can further benefit from cross-task experience distillation. We compare against three baselines that ablate the co-evolutionary design: (1) Baseline: actor training without any experience; (2) Static Online Exp.: πϕ dynamically maintains and constructs M during training but is not optimized, isolating the effect of extractor co-evolution; and (3) Exp. Only: πϕ is trained to maintain and refine M, but the actor πθ is held fixed, isolating the effect of actor co-evolution. Together, these baselines allow us to disentangle the mutual benefit of co-evolving both πθ and πϕ. Table 1 reports final evaluation performance, and Figure 8 shows the training curves for each method. We also provide the number of actions per task throughout training in Appendix A.1. For methods that leverage experience, we evaluate under two settings: with and without retrieving from M at test time.

The results reveal several key findings. First, integrating experience at test time consistently improves performance (e.g., +5% for both Static Online Exp. and Complementary RL), confirming the value of retrieved experience during inference. However, Static Online Exp. fails to surpass the baseline even with experience at test time (gap > 10%), and its training curves are dominated by the baseline across nearly all tasks. We attribute this to distributional misalignment: without parametric updates, the fixed extractor cannot adapt its experience maintenance strategy to the evolving actor, leading to noisy and inconsistent retrieval, particularly in the multi-task setting where cross-task experience contamination is observed. In contrast, Complementary RL consistently outperforms the baseline both with and without experience at test time (+7% and +2% on average, respectively), demonstrating that co-evolutionary training internalizes useful experience into the actor itself. Finally, optimizing only the experience extractor (Exp. Only) yields marginal actor improvement, suggesting that experience quality alone is insufficient when the actor’s base capability is limited (Ouyang et al., 2025).

Table 1: Multi-task evaluation performance. Methods with (w/ exp.) retrieve experience from M at test time, while (w/o exp.) evaluates the actor πθ alone.

Minihack Room WebShop ALFWorld Avg.

Baseline 0.68 0.81 0.72 0.75 Static Online Exp. (eval w/ exp.) 0.41 0.67 0.69 0.59 Static Online Exp. (eval w/o exp.) 0.39 0.59 0.64 0.54 Exp. Only 0.49 0.37 0.13 0.33 Comp. RL (eval w. exp.) 0.78 0.87 0.82 0.82 Comp. RL (eval w/o exp.) 0.75 0.84 0.74 0.78

###### 4.3 Analysis

Effect of Experience Extractor Capacity We investigate whether a stronger experience extractor πϕ can further amplify the benefits of Complementary RL. Specifically, we compare Qwen3-30B-A3B-Instruct-

2507 against the default Qwen3-4B-Thinking-2507 as the experience extractor in multi-task training. As shown in Figure 9a, a larger experience extractor yields consistent improvement across tasks (+5% on average), suggesting that greater extractor capacity enables the extraction of more generalizable and informative experience, which in turn further benefits actor learning. Per-task results are provided in Appendix A.2.

Complementary RL with Self-Distillation Inspired by self-distillation (Hubotter¨ et al., 2026), we explore integrating self-distillation into Complementary RL. For each trajectory in the experience-guided subgroup, we compare its score against the mean score of the experience-free subgroup; trajectories that exceed this threshold are collected into a self-distillation batch. For each sample in this batch, we strip all experience-related context, including the retrieved experience at the first turn and all search and ask

###### Task Scaling

All Tasks (Avg)

###### Minihack Room

###### Avg.RolloutTime(s)

Baseline 0.99×

0.9

1.00

1.0

600

Baseline

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | |[Figure 122]<br><br>[Figure 123]<br><br>[Figure 124]|
| | |Ba Co|seline mp. RL|(4B Ex|p.)| |
| | |Co|mp. RL|(30BA|3B Exp|.)|
| | | | | | | |

Comp. RL

| |
|---|

[Figure 125]

0.8

RewardScore

+6.6%

Comp. RL

RewardScore

0.75

SuccessRate

0.8

450

0.7

[Figure 126]

0.50

[Figure 127]

0.6

0.6

300

0.25

Baseline

0.4

0.5

1.00×

+8.1% +7.9%

Comp. RL + Distill.

150

0.00

0.98× 0.99×

0.4

Comp. RL

0.2

0

0.3

0 25 50 75 100 125

0 25 50 75 100 125 150

128 256 512 1024

3 6 8

Training Steps

Rollout Batch Size

# of Task

Training Steps

(a) Extractor Capacity

(b) Comp. RL w/ Distill.

(c) Rollout Time

(d) Task Scaling

Figure 9: Analysis of Complementary RL across different aspects of its design.

interactions, and supervise the actor πθ via next-token prediction loss jointly with the RL objective. This integration yields a dual benefit Complementary RL continues to optimize the actor through outcomebased rewards and evolving experience, while self-distillation additionally enables the actor to internalize successful experience-guided behaviors directly into its parameters, converting externally scaffolded reasoning into intrinsic capability.

However, results on MiniHack Room in Figure 9b show that, while this integration initially improves upon Complementary RL, it collapses in later training. We suspect this may stem from suboptimal hyperparameter choices, or alternatively, applying self-distillation at periodic intervals rather than every step may alleviate this issue. Due to resource constraints, we leave additional investigation to future work.

Rollout Latency We run a series of experiments to evaluate whether our framework introduces additional latency to rollout collection during training. We compare our framework against a baseline without experience integration and measure the average rollout collection time across different rollout batch sizes (i.e., varying numbers of parallel running environments). Across all settings, we fix the number of parallel search workers and the batch processing query size as described in Appendix C.2. As shown in Figure 9c, our framework introduces no appreciable latency to rollout collection across all settings, remaining consistently on par with the baseline. We also provide the detailed average search time per step during training in Figure 12.

Task Scaling We next investigate whether Complementary RL continues to deliver benefits over the RL baseline without experience integration as the number of tasks scales up, more closely reflecting real-world industrial post-training settings where a broad mixture of tasks is used for RL. To this end, in addition to the three-task mixture introduced in Section 4.2, we further construct a six-task mixture by incorporating more challenging tasks; detailed configurations of the mixture are provided in Appendix C.3. The results are presented in Figure 9d, which shows that Complementary RL consistently outperforms the baseline in both settings (+6.6% and +8.1% on the 3-task and 6-task mixtures, respectively), demonstrating that the performance gains of Complementary RL scale robustly with the number of tasks.

###### 5 Related Works

Leveraging accumulated experience to accelerate reinforcement learning has garnered significant attention for its potential to improve training efficiency (Silver & Sutton; Zhao et al., 2025; Zhai et al., 2025).

- A direct approach is to store historical trajectories or workflows and retrieve them at inference time to improve performance on similar situations (Moeini et al., 2025; Deng et al., 2025; Wang et al., 2024; Li et al., 2025). However, such approaches cannot guarantee the quality or relevance of retrieved experience, potentially introducing noise that hinders learning. To address this, one line of work introduces a dedicated experience extractor that dynamically constructs and maintains the experience bank in accordance with the agent’s learning progress (Xia et al., 2026; Zhai et al., 2025; Zhang et al., 2026a), while another line optimizes the retrieval process to ensure that high-quality and relevant experience is surfaced for agent improvement (Zhang et al., 2026a; Zhou et al., 2025). However, these works treat experience as a static resource, either maintaining fixed experience banks or employing non-adaptive extractors decoupled from the agent’s evolving capabilities, which limits the full potential of the learning-from-experience paradigm. In contrast, Complementary RL co-evolves the agent and experience extractor, enabling dynamic and mutually beneficial adaptation throughout training.

Another key question is how to effectively utilize experience during RL training. The most straightforward approach treats experience as context, including it when computing policy gradients during RL optimization (Li et al., 2025; Salama et al., 2025; Zhang et al., 2025b; Xia et al., 2026). However, this paradigm cannot guarantee improved performance when experience is absent at test time. One line of work addresses this by decoupling rollout collection and policy optimization: experience is provided

during rollout collection, while policy gradients are computed without experience in context, with the trust region adjusted accordingly (Zhai et al., 2025). Another line of work leverages experience to collect high-quality successful trajectories and optimizes the policy to reproduce them without experience in context (Hubotter¨ et al., 2026; Song et al., 2026). In contrast, Complementary RL not only orchestrates the co-evolutionary training of both models, but also introduces experience-guided and experience-free rollout groups with separate advantage estimation for joint optimization under both conditions.

###### 6 Conclusion

In this work, we present Complementary RL, a unified algorithm and infrastructure co-design framework that enables agents to effectively leverage and accumulate experience throughout the RL training process. Rather than treating experience construction and management as a static component with a fixed extractor, we propose jointly training the policy actor and the experience extractor within an asynchronous dualloop. This co-evolutionary design ensures that the actor’s growing capabilities continuously reshape what the extractor learns to distill, while the extractor’s improving outputs in turn accelerate the actor’s learning, each mutually and continuously shaping the other toward better performance.

###### 7 Acknowledgement We would like to thank Johan Obando-Ceron for the valuable discussions and feedback.

###### References

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Hexuan Deng, Wenxiang Jiao, Xuebo Liu, Jun Rao, and Min Zhang. Rea-rl: Reflection-aware online reinforcement learning for efficient large reasoning models. arXiv preprint arXiv:2505.19862, 2025.

Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, et al. Agentic entropy-balanced policy optimization. arXiv preprint arXiv:2510.14545, 2025.

Jonas Hubotter,¨ Frederike Lubeck,¨ Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. URL https://arxiv.org/abs/2310.06770.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Zhiyu Li, Shichao Song, Hanyu Wang, Simin Niu, Ding Chen, Jiawei Yang, Chenyang Xi, Huayi Lai, Jihao Zhao, Yezhaohui Wang, et al. Memos: An operating system for memory-augmented generation (mag) in large language models. arXiv preprint arXiv:2505.22101, 2025.

Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models. arXiv preprint arXiv:2310.10505, 2023.

Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Kun Wang, et al. Llm-based agents suffer from hallucinations: A survey of taxonomy, methods, and directions. arXiv preprint arXiv:2509.18970, 2025.

Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, and Min Lin. Gem: A gym for agentic llms, 2026. URL https://arxiv.org/abs/2510.01051.

Amir Moeini, Jiuqi Wang, Jacob Beck, Ethan Blaser, Shimon Whiteson, Rohan Chandra, and Shangtong Zhang. A survey of in-context reinforcement learning. arXiv preprint arXiv:2502.07978, 2025.

Randall C O’Reilly, Rajan Bhattacharyya, Michael D Howard, and Nicholas Ketz. Complementary learning systems. Cogn Sci, 38(6):1229–1248, December 2011.

Siru Ouyang, Jun Yan, I Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T Le, Samira Daruki, Xiangru Tang, et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.

Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, pp. 2366–2377, 2025.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, and Yassine Benajiba. Meminsight: Autonomous memory augmentation for llm agents, 2025. URL https://arxiv. org/abs/2503.21760, 2025.

Mikayel Samvelyan, Robert Kirk, Vitaly Kurin, Jack Parker-Holder, Minqi Jiang, Eric Hambro, Fabio Petroni, Heinrich Kuttler, Edward Grefenstette, and Tim Rockt¨aschel. Minihack the planet: A sandbox for open-ended reinforcement learning research. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021. URL https://openreview.net/forum?id= skFwlyefkWJ.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks, 2020. URL https://arxiv.org/abs/1912.01734.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cˆot´e, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/abs/2010.03768.

David Silver and Richard Sutton. Welcome to the era of experience. URL https://api.semanticscholar.

###### org/CorpusID:277919528.

David Silver and Joel Veness. Monte-carlo planning in large pomdps. In J. Lafferty, C. Williams, J. ShaweTaylor, R. Zemel, and A. Culotta (eds.), Advances in Neural Information Processing Systems, volume 23. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper files/paper/2010/ file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf.

Yuda Song, Lili Chen, Fahim Tajwar, Remi Munos, Deepak Pathak, J Andrew Bagnell, Aarti Singh, and Andrea Zanette. Expanding the capabilities of reinforcement learning via text feedback. arXiv preprint arXiv:2602.02482, 2026.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Ruiyi Wang and Prithviraj Ammanabrolu. A practitioner’s guide to multi-turn agentic reinforcement learning, 2026. URL https://openreview.net/forum?id=K6T0o875zF.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory, 2024. URL https://arxiv.org/abs/2409.07429.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.

Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Xiaosen Zheng, Zejun Ma, and Bo An. Simpletir: Endto-end reinforcement learning for multi-turn tool-integrated reasoning. arXiv preprint arXiv:2509.02479, 2025.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z Pan, et al. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents, 2023. URL https://arxiv.org/abs/2207.01206.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, et al. Agentevolver: Towards efficient self-evolving agent system. arXiv preprint arXiv:2511.10395, 2025.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.

Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, et al. Agent learning via early experience. arXiv preprint arXiv:2510.08558, 2025a.

Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, et al. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192, 2026a.

Yuheng Zhang, Wenlin Yao, Changlong Yu, Yao Liu, Qingyu Yin, Bing Yin, Hyokun Yun, and Lihong Li. Improving sampling efficiency in RLVR through adaptive rollout and response reuse, 2026b. URL https://openreview.net/forum?id=YVeTYwBZWD.

Zeyu Zhang, Quanyu Dai, Rui Li, Xiaohe Bo, Xu Chen, and Zhenhua Dong. Learn to memorize: Optimizing llm-based agents with adaptive memory framework. arXiv preprint arXiv:2508.16629, 2025b.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.

Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, et al. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025.

###### A Additional Result

- A.1 Action Efficiency Under Multi-Task Training

We further report the average number of actions per task during multi-task RL training in Figure 10. The results consistently show that Complementary RL achieves superior action efficiency alongside higher success rates, further demonstrating the benefit of co-evolutionary experience in the multi-task setting.

0 25 50 75 100 125

Training Steps

10

15

20

25

NumActions

[Figure 128]

[Figure 129]

[Figure 130]

MiniHack Room

Baseline

Static Online Exp.

Comp. RL (Ours)

0 25 50 75 100 125

Training Steps

5

10

15

20

25

NumActions

[Figure 131]

[Figure 132]

[Figure 133]

WebShop

Baseline

Static Online Exp.

Comp. RL (Ours)

0 25 50 75 100 125

Training Steps

5.0

7.5

10.0

12.5

15.0

17.5

NumActions

[Figure 134]

[Figure 135]

[Figure 136]

ALFWorld

Baseline

Static Online Exp.

Comp. RL (Ours)

- Figure 10: Average number of actions per task throughout multi-task training (corresponding to Figure 8).

A.2 Per-Task Performance with Stronger Experience Extractor

- Figure 11 presents per-task performance in multi-task training across two experience extractor sizes (4B and 30B-A3B). Results show that a larger experience extractor consistently yields greater benefit across all tasks.

- A.3 Search Time Throught Training

| | | | |1|28|
|---|---|---|---|---|---|
| | |[Figure 137]| |2|56|
| | | | |5 1<br><br>|12 024|
| |[Figure 138]<br><br>| | | | |
| | | |[Figure 139]<br><br>[Figure 140]<br><br>| | |
| | | | | | |

0 15 30 45 60

Training Steps

0.00

0.25

0.50

0.75

1.00

Time(s)

Search Time (Avg.)

Figure 12: Average search time per training step across different rollout batch sizes.

We further report the detailed average search time across all environments and training steps, corresponding to the experiment in Figure 9c. The results are presented in Figure 12, which shows that although search time increases with larger rollout batch sizes, the maximum observed search time remains around 1 second, which is negligible. We believe that by carefully tuning the query batch size B, the maximum waiting time Tmax, and the number of parallel search workers, the search time can be further reduced even in large rollout batch settings.

- A.4 Training Curves for Task Scaling Experiments

MiniHack Room

WebShop

ALFWorld

1.0

1.0

1.0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | |[Figure 141]| | | | |
| | |B<br><br>Co|[Figure 142]<br><br>[Figure 143]<br><br>aseline mp. RL|(4B Ex|p.)| |
| | |Co|mp. RL|(30BA|3B Exp|.)|
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | |[Figure 144]<br><br>[Figure 145]<br><br>[Figure 146]| | |
| | | | | | | |
| | |B<br><br>Co|aseline mp. RL|(4B Ex|p.)| |
| | |Co|mp. RL|(30BA|3B Exp|.)|
| | | | | | | |

ValidationScore

[Figure 147]

[Figure 148]

[Figure 149]

SuccessRate

SuccessRate

0.8

0.8

0.5

0.6

0.6

0.0

0.4

0.4

Baseline

0.5

Comp. RL (4B Exp.)

0.2

0.2

Comp. RL (30BA3B Exp.)

0 25 50 75 100 125

0 25 50 75 100 125

0 25 50 75 100 125

Training Steps

Training Steps

Training Steps

Figure 11: Per-task training dynamic (corresponding to Figure 9a).

We provide the training curves for the task scaling experiments introduced in Section 4.3 in Figure 13.

###### B Implementation Tricks

Training the experience extractor πϕ is highly unstable due to two compounding challenges. First, the training is severely off-policy: since retrieval timing is uncontrolled, a distilled experience m may be retrieved long after it was generated, introducing a large policy lag between the retrieving actor and the current πϕ. Second, when task descriptions exhibit low diversity, a single experience m tends to be retrieved repeatedly across different training buffer steps, causing data redundancy and πϕ may be updated multiple times on the same experience m, severely undermining training stability. To address these challenges, we introduce two stabilization techniques: Retrieval Diversification and TrainingCount-Aware Advantage Reweighting.

#### EnvScaling-3Tasks

#### EnvScaling-6Tasks

1.0

0.6

RewardScore

RewardScore

0.8

[Figure 150]

[Figure 151]

0.4

0.6

0.4

0.2

0.2

Baseline

Baseline

0.0

0.0

Comp. RL (Ours)

Comp. RL (Ours)

0.2

0 25 50 75 100 125

0 25 50 75 100 125

Training Steps

Training Steps

Figure 13: Training curves for Complementary RL and the baseline across different task mixture settings. Retrieval Diversification For each retrieval query q, instead of retrieving only the top-K most relevant experiences, we oversample by drawing N independent candidate sets, each of size K, yielding a total pool of N × K candidate experiences C(q) = {m1, m2, . . . , mN×K}. We then re-rank the oversampled candidate pool C(q) according to a diversity-aware scoring function that penalizes frequently retrieved experiences:

s(m) = srank(m) − λ · log(1 + c(m)) − 1[recent(m)], (6) where srank(m) is the base relevance rank score of experience m, c(m) denotes its historical retrieval count, λ is a penalty hyperparameter controlling retrieval diversity, and 1[recent(m)] is an indicator that penalize experiences retrieved within a predefined recency window. The final top-K experiences R(q) are selected as the highest-scoring entries under s(m). With this diversification strategy, πϕ is exposed to a broader and more varied set of experiences during training, mitigating the data redundancy issue and producing more diverse advantage signals for stable extractor optimization.

Training-Count-Aware Advantage Reweighting We observe that a single experience m may be retrieved across multiple training buffer steps and optimized repeatedly, leading to overfitting and training instability for πϕ. To mitigate this, we reweight the advantage of each experience in the training buffer

- Bϕ according to both its cumulative training count and its recency of optimization. Specifically, after computing the advantage for each sample in Bϕ according to Section 2.3, we apply a per-experience weight w(m) defined as:

w(m) =

0 if (t − tlast) < δ, (1 + ctrain(m))−α otherwise,

(7)

where t is the current global training step, tlast is the most recent step at which m was trained, δ is a cooldown window that suppresses gradient updates from experiences optimized too recently, ctrain(m) is the cumulative number of times m has been trained on, and α ≥ 0 is a decay exponent controlling how aggressively the advantage is discounted as m accumulates training counts. Together, the cooldown mechanism prevents repeated optimization within a short window, while the count-based decay progressively down-weights overused experiences, yielding more stable and balanced training of πϕ.

###### B.1 Actor Critic

During training of Complementary RL, we observe that retrieved experiences can sometimes confuse rather than benefit the actor, particularly in the early stages of training. Upon closer inspection, we identify two failure modes: (1) Experience Staleness: when the actor has already mastered a given task, the retrieved experience may be overly conservative or even incorrect relative to the actor’s current capability, thereby degrading performance rather than improving it; (2) Experience Imprecision: when the actor’s success rate is low, retrieved experiences are often directionally helpful but may require adaptation to the task at hand, as they are not always precisely aligned with the current context. To address the above failure modes, we propose Actor-Critic, which introduces explicit communication between the policy actor πθ and the experience extractor πϕ.

Specifically, prior to launching the main dual training loop, we run πθ for Twarm warm-up iterations on the current task to estimate its initial average success rate r¯θ. Once training begins, after each retrieval of experience m for a given task query q, we prompt the actor πθ to reflect on the retrieved experience in light of both the current task and its accumulated success rate r¯θ(q). Based on this reflection, the actor produces one of three critic actions:

##### Minihack Room

###### Search Time (Avg.)

1.0

1.5

SuccessRate

[Figure 152]

[Figure 153]

1.21

0.8

Time(s)

1.0

w. Actor-Critic

0.6

w/o Actor-Critic

0.5

0.4

w. Actor-Critic

w/o Actor-Critic

0.04

0.0

0.2

0 25 50 75 100 125 150

0 25 50 75 100 125 150

Training Steps

Training Steps

(a) Success Rate

(b) Retrieval Speed

Figure 14: Analysis of incorporating Actor-Critic into Complementary RL.

- • accept: the experience m is accepted as-is, receiving a critic score sc(m) = 1;
- • refine: the experience m is refined using the actor’s own knowledge, receiving a critic score sc(m) = 0.5;
- • reject: the experience m is discarded, receiving a critic score sc(m) = 0.

This mechanism allows the actor to selectively consume experience commensurate with its current capability. Furthermore, the critic score sc(m) is combined with the task completion reward r(m)—the outcome obtained when using experience m to solve the task—to form an enriched learning signal for the experience extractor πϕ:

r˜(m) = sc(m) + r(m). (8)

As shown in Figure 14a, Actor-Critic yields improved success rates, particularly in the early stages of training on MiniHack Room. However, since the actor must produce a critic decision before each environment interaction, rollout collection is blocked pending the critic result, incurring non-trivial latency overhead (Figure 14b). Therefore, we do not adopt Actor-Critic as a default component in our main experiments, but recommend it as a beneficial addition in scenarios where final performance is prioritized over training throughput.

###### B.2 Lessons Learned

Separate Model Parameters for Actor and Extractor In early attempts, we served a single set of parameters shared between the training and inference engines, using the same weights for both the policy actor πθ and the experience extractor πϕ. This design optimizes a single model under two distinct objectives simultaneously (Equation 5 and 3). However, since the two optimization objectives impose possible conflicting gradient directions, we were unable to guarantee stable training despite extensive tuning efforts. We ultimately resolved this by maintaining separate parameter suites for the actor πθ and the experience extractor πϕ, which decouples the two optimization objectives and yields stable training.

Direct Reward over Relative Reward for Experience In early attempts, rather than using the actor’s task completion reward as the direct reward signal r(m) for experience m, we explored a relative reward strategy. Specifically, we first computed the average reward of the experience-free subgroup as a baseline, and then assigned each sample in the experience-guided subgroup a reward proportional to its improvement over this baseline. However, empirical comparison revealed that this relative reward strategy consistently underperforms direct reward assignment, and we therefore adopt the latter in Complementary RL.

Auxiliary Perplexity Reduction Reward For challenging tasks, a retrieved experience may be instructive yet insufficient to directly yield task success. We therefore explored augmenting the reward signal for πϕ with a perplexity reduction bonus, motivated by the intuition that a genuinely helpful experience should increase the actor’s confidence, and thus reduce its entropy, when processing the task at hand.

Concretely, at the start of each task, we compute the actor’s entropy H(πθ | q) over the task query q without any retrieved experience, and then re-compute the entropy H(πθ | q, m) after injecting the retrieved experience m into the system message. The entropy reduction ∆H(m) = H(πθ | q) − H(πθ | q, m) is then used as an auxiliary reward bonus for πϕ. We evaluated five normalization strategies for computing this bonus:

- • Relative: scale-invariant percentage reduction, b = w · ∆H(m)/H(πθ | q), clipped to a predefined range;
- • Tanh: smooth non-linear scaling bounded to [−1,1], b = w · tanh(∆H(m));
- • Sigmoid: temperature-scaled sigmoid bounded to [0, 1] and re-centered, b = w·(2σ(∆H(m)/τ)− 1);
- • Asymmetric Clipping: asymmetrically clips negative and positive gains to encourage exploration without heavy penalty;
- • Log-Space: sign-preserving logarithmic compression, b = w · sgn(∆H(m)) · log(1 + |∆H(m)|), suitable when entropy magnitudes vary widely.

However, none of these strategies yielded a consistent improvement in practice, and we therefore exclude this auxiliary reward from Complementary RL.

###### C Implementation Details

###### C.1 Environment Description

During RL training, we implement each environment following the protocol of GEM (Liu et al., 2026), with the the SWE task is additionally executed using ROCK2. All environments adopt a binary reward scheme, assigning r = 1 upon task success and r = 0 upon failure, with the exception of ALFWorld, which assigns r = 1 upon success and r = −1 upon failure. In the following, we provide a brief introduction to each environment and our corresponding implementation details.

MiniHack MiniHack (Samvelyan et al., 2021) is a collection of game environments in which an agent explores a world under a fog-of-war observation model, meaning the agent can only observe its immediately surrounding grid cells. The goal of the agent is to reach a target destination by avoiding traps and obstacles, using tools to cross rivers or lava, or defeating monsters. We adapt MiniHack for LLM-based agents by representing each entity—including items, traps, monsters, and the agent itself—as a text symbol following the NetHack convention3 (e.g., @ represents the agent’s position, > represents the goal position). At each timestep, the agent is provided with the current observable grid layout along with a legend of symbol meanings, and is asked to decide the next action. The action space typically consists of directional movements, with additional task-specific actions such as pick up or apply in more complex environments.

In this work, we evaluate on the following MiniHack environments of increasing difficulty:

- • MiniHack Room4: The agent navigates a dark room, avoiding traps, obstacles, and monsters to reach the goal. We use MiniHack-Room-Ultimate-5x5-v0. The action space consists solely of directional actions (e.g., north, south, east, west).
- • MiniHack Maze5: A more challenging environment in which the agent must navigate a maze to reach the goal. We use MiniHack-MazeWalk-9x9-v0. The action space also consists of directional actions.
- • MiniHack KeyRoom6: The agent must first locate a key, find a locked door, open the door with the key, and finally reach the goal position. We use MiniHack-KeyRoom-Dark-S5-v0. This environment includes additional actions beyond directional movement, such as pick up and apply.

- • MiniHack River7: The agent must first push a boulder into the river, then cross it, and finally reach the goal. We use MiniHack-River-Narrow-v0.

WebShop WebShop (Yao et al., 2023) is a benchmark that simulates web-based shopping, in which agents navigate a realistic web interface to find and purchase products matching user specifications. At each timestep, the agent receives a product specification and must choose between two types of actions: issuing a text search query (e.g., search[red shoes]) or clicking a text button (e.g., choose[Size 9]).

- 2https://github.com/alibaba/ROCK
- 3https://nethackwiki.com/wiki/Main Page

- 4https://minihack.readthedocs.io/en/latest/envs/navigation/room.html
- 5https://minihack.readthedocs.io/en/latest/envs/navigation/mazewalk.html
- 6https://minihack.readthedocs.io/en/latest/envs/navigation/keyroom.html
- 7https://minihack.readthedocs.io/en/latest/envs/navigation/river.html

The environment returns an observation after each action, and the agent continues until the target product is successfully purchased or the episode terminates. In our implementation, we adopt the small variant configuration, restricting the searchable product catalog to 1,000 items, with goals sampled from the instruction pool via weighted sampling based on attribute frequency.

ALFWorld ALFWorld (Shridhar et al., 2021) is a text-based interactive environment aligned with the ALFRED embodied AI benchmark (Shridhar et al., 2020), in which agents complete household tasks by navigating rooms and interacting with objects through natural language commands. Each task presents the agent with a high-level goal (e.g., put a heated plate in the fridge), and at each timestep, the agent receives a textual observation describing the objects visible in the current room and must issue a natural language action (e.g., go to countertop 1, pick up plate). The episode terminates upon successful task completion or when the maximum number of steps is reached. In our implementation, we train on 1,466 task instances from ALFWorld and hold out 134 instances for evaluation.

SWE-Bench SWE-Bench (Jimenez et al., 2024) is a real-world software engineering benchmark in which an agent must resolve GitHub issues by modifying the relevant portions of a codebase such that all provided unit tests pass successfully. For each task instance, the agent receives a GitHub issue description and interacts with the codebase through three tools: execute bash for executing shell commands, str replace editor for viewing and editing source files, and submit for submitting the final patch. The environment returns the tool execution result as a textual observation after each action.

In our experiments, we utilize SWE-Bench-Verified for training. However, since many tasks in the full dataset are too challenging for smaller models, naively training Qwen3-4B-Instruct-2507 on the complete dataset yields unstable and ineffective learning. To address this, we perform a preliminary pass@16 evaluation using Qwen3-4B-Instruct-2507 and retain only tasks with a success rate in the range (0,80%), filtering out both trivially easy and prohibitively difficult instances. This yields a curated training set of 124 tasks, and we report the final success rate evaluated throughout training.

Sokoban Sokoban is a classic text-based puzzle game in which an agent must navigate a grid and push boxes onto designated target positions while avoiding walls. The task requires multi-step planning and spatial reasoning, as boxes can only be pushed and an incorrectly pushed box may render the puzzle unsolvable. We represent the walls, boxes, targets, agent, and empty positions using structured text symbols such as W, A, C, @, and ., respectively. We configure each episode as a 6 × 6 room with two boxes and two corresponding target positions, yielding a challenging combinatorial search space for the agent.

###### C.2 Training Configuration

We implement Complementary RL within the ROLL8 framework, using Megatron as the training engine and vLLM as the inference engine across all experiments. We do not apply KL regularization for either

πθ or πϕ, and adopt the AdamW optimizer with a constant learning rate of 1 × 10−6 throughout. Unless otherwise specified, we run 4 parallel search workers and 4 parallel embedding workers in our framework.

We use Qwen3-Embedding-0.6B9 as the embedding model, served via vLLM. The query batch size B is set to 16, and the maximum waiting time tmax is set to 0.001 seconds. The training buffer size |Bϕ| for πϕ is set to 64, and the periodic merge interval is set to 5 steps. The importance sampling clip thresholds

ϵlowIS and ϵhighIS in Equation 3 are both set to 0.1. In the following, we describe the specific implementation details for each experimental group.

Configuration of Experiments in Figure 3 We run each experiment with a total rollout batch size of 128, a group size of K = 8, and a clip ratio of ϵ = 0.2. Each experiment runs for 145 steps with a micro-batch size of 64 for πθ. We set the maximum number of interaction turns to 30, the maximum output tokens per step to 4,096, the maximum sequence length for πθ to 32,768 tokens, and the maximum sequence length for πϕ to 65,536 tokens.

• Offline Exp. We run Qwen2.5-7B-Instruct offline to interact with MiniHack Room for a

maximum of 30 interaction turns. The resulting trajectories are then routed to Qwen3-30BA3B-Instruct-2507 for experience distillation, followed by the same merging and semanticsimilarity-based deduplication pipeline used in our main experiments to construct a high-quality offline experience bank.

8https://github.com/alibaba/ROLL 9https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

• Static Online Exp. This variant follows the same setup as Complementary RL, except that the

experience extractor πϕ is not optimized during training. All other components remain active, including subgroup separation, query diversification, and search and ask.

Configuration of Single-Task Training Unless otherwise noted, all other settings follow the general configuration described above.

- • WebShop: We use a rollout batch size of 64, a group size of K = 8, and a micro-batch size of 16. The number of warmup steps for πθ is set to 10, the maximum number of training steps is 256, and the maximum sequence length for πθ is 16,384 tokens.
- • ALFWorld: We use a rollout batch size of 128, a group size of K = 8, and a micro-batch size of 32.

The maximum number of training steps is 128, the maximum sequence length for πθ is 16,384 tokens, the maximum number of interaction turns is 40, and the maximum output tokens per step is 2,048.

Configuration of Multi-Task Training We run all experiments with a total rollout batch size of 384, with each task contributing a batch size of 128, a group size of K = 8, and a micro-batch size of 96. Training runs for 128 steps, with a maximum sequence length of 32,768 tokens, a maximum interaction count of 30, and a maximum output tokens per step of 4,096. All other settings follow the general configuration described above.

- C.3 Task Mixture 3-Tasks: Minihack Room, Webshop, and ALFWorld. 6-Tasks: Minihack Room, Minihack Maze, Minihack KeyRoom, Sokoban, Webshop, and ALFWorld.

- D Illustration Here, we provide representative examples of distilled experience in our experiments.

Single-Task Experience We present representative distilled experience entries from single-task training for MiniHack (Table 2), WebShop (Table 3), ALFWorld (Table 4), and SWE-Bench (Table 5).

Multi-Task Experience We also find that the experience extractor is capable of distilling universal experience transferable across tasks, for which we show representative examples in Table 6.

Table 2: Minihack distilled experience.

When an agent observes a visible exit (e.g., staircase down symbol ’>’) in the immediate field of view and the direct path to it is unobstructed by hazards or threats (e.g., no monsters, traps, or dead ends), prioritize moving toward that exit immediately. This strategy works best when the exit is in a cardinal direction with clear visibility and no blocking obstacles.

Action Sequence: 1. Identify visible exits (e.g., ’>’, ’door’ symbols) within the current dungeon view. 2. For each visible exit, verify the path to it has no immediate threats or hazards (e.g., jackals, traps) in the adjacent cells. 3. Move in the direction of the nearest visible exit with an unobstructed path (e.g., north if the staircase is directly north).

Decision Logic: - If a visible exit path is unobstructed, move toward it. - If threats are present but do not block the exit path, do not deviate toward threats—focus on moving toward the exit first. - If the environment offers a temporary benefit (e.g., ”full moon” implying enhanced visibility or reduced aggression), prioritize the direct path to the exit over waiting for the benefit, unless the benefit explicitly reduces threat levels (e.g., ”reduced monster aggression”).

Failure Prevention: - Do NOT move toward adjacent threats (e.g., jackals) if they do not block the exit path, as this increases exposure to potential hazards. - Do NOT delay action due to environmental effects (e.g., full moon) when a clear path exists, as the benefit may be passive and not directly actionable (e.g., no explicit effect on movement or threat reduction). - Do NOT assume environmental effects (e.g., full moon) provide direct benefits unless the environment explicitly states their impact (e.g., ”reduced monster aggression”).

Generalization: This principle applies to any grid-based navigation task with visually distinguishable exits and localized threats (e.g., mazes, dungeons, code navigation). It ensures agents focus on immediate, actionable progress toward known goals without unnecessary delays caused by environmental conditions that lack explicit, actionable benefits.

Table 3: WebShop distilled experience.

When searching for products with specific color variants (e.g., ’type 3-camel’) as a critical constraint, **first** filter the search results by this color variant to isolate relevant products, then verify the remaining products meet all other requirements (price, features, installation specs) before purchase.

1. **Initial search**: Use broad keywords matching all non-color requirements (e.g., ”height adjustable high density easy install easy assemble home office chairs price < 130”). 2. **Color filter**: From the search results, explicitly set the color to the required variant (e.g., ‘click[type 3-camel]‘) to narrow results. 3. **Verification**: Check the filtered list for products that satisfy the non-color constraints (e.g., price < $130, features like ”easy install”). 4. **Proceed**: If a product matches, select and purchase; if not, refine the search or adjust filters.

**Decision Logic**: - *If* the initial search yields products with multiple color options and the task specifies

- a **particular** color variant, *then* apply the color filter to exclude irrelevant products. - *If* no products match after filtering, *then* re-run the search with adjusted keywords (e.g., broader color terms) or expand the search scope.

**Failure Prevention**: - Do **not** skip color filtering when the task explicitly requires a specific color variant. - Do **not** assume products with similar names or descriptions match the exact color requirement. - **Why**: Color variants are often mislabeled or not clearly specified in product listings; filtering first ensures compliance with the task’s color constraint.

**Generalization**: This strategy applies to any e-commerce environment where color variants are a critical requirement (e.g., furniture, apparel, electronics). It is invariant to the specific product category, price thresholds, or feature lists.

- Table 4: ALFWorld distilled experience.

- 1. **Situational Context**: When a task requires locating a specific object (e.g., food items, tools) in a household environment and initial checks fail.
- 2. **Action Sequence**: a. Check all countertops (in numerical order from 1 to N) for the object. Countertops are common storage areas for frequently used items like food. b. If the object isn’t found on countertops, check the fridge (after opening it if closed) for perishables or stored food. c. If still not found, check sinks (for washed items), cabinets (for stored items), and shelves (for items kept at eye level) in a logical order.
- 3. **Decision Logic**: - If the object is a food item (e.g., fruit, vegetables), prioritize countertops first as they are common for fresh items. - If the object is a tool or consumable that’s typically stored in containers (e.g., spices, packaged goods), check the fridge after countertops. - If the object is small and likely hidden (e.g., a key, battery), adjust the order to check high shelves, drawers, or enclosed spaces first.
- 4. **Failure Prevention**: - Do not assume the object is in a specific location without checking (e.g., the agent didn’t assume the apple was in the fridge despite checking it). - Do not skip checking countertops when searching for food items (e.g., the agent first checked countertop 1 before the fridge). - Avoid redundant searches by tracking which locations have been checked (e.g., noting ”countertop 1 checked” to prevent re-checking).
- 5. **Generalization**: This strategy applies universally to household environments where objects are stored in common locations. It works across variations in object types (e.g., food, tools, household items) and environments (e.g., different house layouts). It avoids coordinates (e.g., ”countertop 2”) by using relative terms like ”numerical order” and ”logical order” based on typical household practices.

- Table 5: SWE-Bench distilled experience.

When attempting to make precise code edits using string replacement tools (like the str replace editor function), follow this systematic approach to ensure success:

- 1. **Verify the actual code content**: - IF you plan to perform a string replacement: - EXECUTE a ‘view‘ command with an appropriate line range to get the exact code - Use the output to determine the precise string pattern matching the target code
- 2. **Extract the exact replacement pattern**: - IF the string you want to replace is multi-line: - Capture the exact output from the ‘view‘ command with appropriate line ranges - Include all whitespace (indentation, spaces, line breaks) in your replacement pattern - IF the target pattern has variable whitespace (e.g., different indents): - Use the most common or canonical whitespace pattern from your ‘view‘ output Consider adding a fallback for variations
- 3. **Handle failure cases**: - IF the tool reports no occurrences found: a. Re-examine the ‘view‘ output for subtle differences b. Check if whitespace differences exist (e.g., extra spaces, tabs, newlines) c. Try matching with a slightly different pattern that accounts for common whitespace variations - IF the code is in a version control system with line endings: a. Normalize line endings before attempting the replacement

b. Ensure the replacement pattern matches the line endings (e.g., Unix vs. Windows)

- 4. **Alternative approach**: - IF precise string replacement fails repeatedly: - Create a temporary file with the fix using the ‘str replace‘ tool - Save the fixed file to a temporary location - Then replace the original file using an atomic operation - This avoids multiple failed attempts

- 5. **Verification**: - AFTER making a replacement: a. Run ‘view‘ on the modified code to confirm the change b. Test the code functionality (if possible) to ensure the fix worked c. Check for potential side effects like unintended changes to other parts of the code This strategy ensures reliable code edits in environments with string-based editing tools by emphasizing precision in pattern matching and verification through explicit code validation. Special considerations for Python and type systems: - When working with type-related code (especially with TypeVar, generics, etc.): - Be mindful that TypeVar objects are not iterable - Always verify if a parameter is a tuple before iterating over it - Add clear comments to explain non-iterable behavior for type safety

Table 6: Universal distilled experience from multi-task training.

- 1. Situational Context: This protocol governs how an agent detects stagnation, breaks out of unproductive loops, and escalates to higher-level problem-solving when direct attempts repeatedly fail.
- 2. When to Escalate: Escalation is triggered when the same category of action has been attempted 3 or more consecutive times with no meaningful state change or improvement in outcome. This threshold applies regardless of domain, whether it is repeated failed movements, unsuccessful search queries, recurring test failures, or fruitless object searches. At this point, persisting with the same approach is counterproductive. The agent must halt, shift into diagnostic mode, and treat all prior feedback from the environment as structured evidence rather than mere failure.
- 3. How to Break the Loop: Upon detecting stagnation, the agent should first articulate, within its reasoning step, exactly what was tried, what the environment returned, and what gap in knowledge or strategy is causing the block. The agent should then consider whether the problem has been correctly framed, whether an environmental rule has been misunderstood, and whether any untried alternative exists. If external tools or a knowledge base are available (like search and ask), they should be consulted immediately with a precise, context-rich query rather than deferred. Action resumes only under a substantively different strategy.

- 4. What to Avoid: The agent must not treat repeated failure as a reason to keep trying the same action, nor mistake tool availability for optional assistance, tools exist to be used when direct attempts stall. Overconfidence in a single strategy, blindness to deadlock states, and circular revisiting of alreadyexhausted options are the primary failure modes this protocol is designed to prevent. Equally, the agent should not escalate prematurely, slow progress is not stagnation, and a single failure does not warrant a full strategy shift.
- 5. Underlying Principle: Every failed attempt carries diagnostic value — it narrows the solution space and reveals constraints. An intelligent agent does not persist blindly but adapts purposefully: recognizing when a strategy has run its course, identifying whether the obstacle stems from a knowledge gap or a wrong approach, and pivoting with intent. This capacity to self-monitor, diagnose, and escalate is what separates robust task completion from brittle, loop-prone behavior across any environment.

