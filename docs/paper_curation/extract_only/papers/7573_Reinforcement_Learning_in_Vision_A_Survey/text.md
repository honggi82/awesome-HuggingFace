# Reinforcement Learning for Large Model: A Survey

Weijia Wu1, Chen Gao1, Joya Chen1, Kevin Qinghong Lin1, Qingwei Meng2, Yiming Zhang3 , Yuke Qiu2, Hong Zhou2, Mike Zheng Shou1‡

1National University of Singapore 2Zhejiang University 3The Chinese University of Hong Kong

arXiv:2508.08189v3[cs.CV]23Dec2025

Abstract—Recent advances at the intersection of reinforcement learning (RL) and visual intelligence have enabled agents that not only perceive complex visual scenes but also reason, generate, and act within them. This survey offers a critical and up-to-date synthesis of the field. We first formalize visual RL problems and trace the evolution of policy-optimization strategies from RLHF to verifiable reward paradigms, and from Proximal Policy Optimization to Group Relative Policy Optimization. We then organize more than 250 representative works into four thematic pillars: multi-modal large language models, visual generation, unified model frameworks, and vision-language-action models. For each pillar we examine algorithmic design, reward engineering, benchmark progress, and we distill trends such as curriculum-driven training, preference-aligned diffusion, and unified reward modeling. Finally, we review evaluation protocols spanning set-level fidelity, sample-level preference, and state-level stability, and we identify open challenges that include sample efficiency, generalization, and safe deployment. Our goal is to provide researchers and practitioners with a coherent map of the rapidly expanding landscape of visual RL and to highlight promising directions for future inquiry. Resources are available at: Visual RL.

Index Terms—Reinforcement Learning, Multimodal Model, Survey, Visual Generation.

✦

CONTENTS

- 1 Introduction 2
- 2 Preliminary: Reinforcement Learning in LLM 2

- 2.1 Notation and Problem Formulation . . . 2
- 2.2 Alignment Paradigms . . . . . . . . . . 3

- 2.2.1 RL from Human Feedback . . 3
- 2.2.2 Direct Preference Optimization 4
- 2.2.3 Reinforcement Learning with Verifiable Rewards . . . . . . . 4

- 2.3 Policy-Optimization Algorithms . . . . . 5

- 2.3.1 Proximal Policy Optimization 5
- 2.3.2 Group Relative Policy Optimization . . . . . . . . . . . . 6

- 3 Reinforcement Learning in Vision 6

- 3.1 Multi-Modal Large Language Models . 6

- 3.1.1 Conventional RL-based MLLMs 6
- 3.1.2 Spatial and 3D Perception . . 7
- 3.1.3 Image Reasoning . . . . . . . 7
- 3.1.4 Video Reasoning . . . . . . . . 8

- 3.2 Visual Generation . . . . . . . . . . . . . 8

- 3.2.1 Image Generation . . . . . . . 8
- 3.2.2 Video Generation . . . . . . . 8
- 3.2.3 3D Generation . . . . . . . . . 9
- 3.2.4 Reward Design and Preference Modeling for Visual Generation . . . . . . . . . . . . . 9

- 3.3 Unified Model . . . . . . . . . . . . . . . 9

- 3.3.1 Unified RL . . . . . . . . . . . 9
- 3.3.2 Task specific RL . . . . . . . . 9

Corresponding author (‡) : Mike Zheng Shou (mikeshou@nus.edu.sg).

3.4 Vision Language Action Models . . . . . 9

- 3.4.1 GUI Automation . . . . . . . . 9
- 3.4.2 Visual Navigation . . . . . . . 10
- 3.4.3 Visual Manipulation . . . . . . 10

- 4 Metrics and Benchmarks 10

- 4.1 Formalizing Metric Granularity . . . . . 11
- 4.2 Evaluation of Multi-Modal Large Language Models . . . . . . . . . . . . . . . 12
- 4.3 Evaluation of Visual Generation Models 12
- 4.4 Evaluation of Unified Models . . . . . . 12
- 4.5 Evaluation of Vision Language Action Models . . . . . . . . . . . . . . . . . . . 13
- 4.6 Benchmarks . . . . . . . . . . . . . . . . 13

- 5 Challenges and Future Works 14

- 5.1 Effective Reasoning: Balancing Depth and Efficiency . . . . . . . . . . . . . . . 14
- 5.2 Long-Horizon RL in VLA . . . . . . . . 14
- 5.3 RL for Thinking with Vision . . . . . . . 14
- 5.4 Reward Model Design for Visual Generation . . . . . . . . . . . . . . . . . . . . 14

- 6 Conclusion 14 References 15

## 1 INTRODUCTION

Reinforcement Learning (RL) has achieved remarkable successes in the field of Large Language Models (LLMs) [1], [2], most notably through paradigms like Reinforcement Learning from Human Feedback (RLHF) [3] and innovative frameworks such as DeepSeek-R1 [4]. These methodologies have significantly enhanced the capabilities of LLMs, aligning generated outputs closely with human preferences and enabling nuanced, complex reasoning and interaction capabilities previously unattainable through supervised learning alone.

In recent years, inspired by these remarkable achievements from LLM, there has been an explosive interest in extending the RL methodologies that proved successful for LLMs to multimodal large models, including VisionLanguage Models (VLM) [5], [6], [7], Vision-Language-Action models (VLA) [8], [9], [10], [11], diffusion-based visual generation models [12], [13], [14], and unified multimodal frameworks [15], [16], [17], as shown in Figure 1. Multimodal models such as Gemini 2.5 [18] have leveraged RL to align visual-textual reasoning processes and produce outputs with higher semantic coherence and alignment with human judgments. Concurrently, VLA models integrating vision and language with action-oriented outputs have adopted RL to optimize complex sequential decision-making processes in interactive environments, significantly improving taskspecific performance in GUI automation [11], [19], robotic manipulation [8], and embodied navigation [20]. The rapid advancement of diffusion-based generative models has further spurred this RL-driven innovation wave. Works like ImageReward [21] have introduced reinforcement learning to enhance the semantic alignment and visual quality of generative outputs, refining diffusion-based generation through iterative feedback mechanisms derived from human preferences or automated reward critics. Moreover, unified models that blend multiple tasks, i.e., understanding, and generation, into single architectures [15], [22] have increasingly relied on RLdriven fine-tuning, achieving generalization and task transfer previously considered challenging. Despite the substantial progress in integrating reinforcement learning with multimodal large language models, several core challenges remain. These include stabilizing policy optimization under complex reward signals, managing high-dimensional and diverse visual inputs, and designing scalable reward functions that support long-horizon decision-making. Addressing these challenges necessitates methodological innovations in both algorithmic design and evaluation protocols.

In this survey, we present a comprehensive synthesis of recent advances in visual reinforcement learning within the context of multimodal large models, with a focus on the surge of research activity since 2024. We begin by revisiting foundational RL successes in language models, such as RLHF [3] and DeepSeek-R1 [4], which have laid the groundwork for multimodal adaptation. Subsequently, we discuss how these strategies have evolved in the visual domain, categorizing over 200 representative works into four key domains: (i)

multimodal large language models, (ii) visual generation, (iii) unified RL frameworks, and (iv) vision-language-action agents, as shown in Figure 1. Within each category, we analyze key developments in algorithmic formulations, reward modeling, and benchmarking methodologies. Finally, we identify open challenges and future directions, highlighting the need for more efficient multimodal reasoning, robust long-horizon learning strategies for VLA tasks, and scalable, high-fidelity reward signals tailored to visual generation. Through this comprehensive overview, we offer a structured overview of visual reinforcement learning to support future research and practical deployment in this rapidly evolving field.

Our key contributions are as follows:

- • We provide a systematic and up-to-date survey of over 200 visual reinforcement learning studies, encompassing MLLMs, visual generation, unified models, and visionlanguage-action agents.
- • We analyze advances in policy optimization, reward modeling, and benchmarking across subfields, revealing key challenges and future directions: such as reward design in visual generation and the lack of intermediate supervision in reasoning and VLA tasks.
- • We introduce a principled taxonomy of Visual RL methods based on metric granularity and reward supervision, including three reward paradigms for image generation. This framework clarifies the design trade-offs across domains and offers actionable insights for selecting and developing RL strategies.

## 2 PRELIMINARY: REINFORCEMENT LEARNING IN LLM

This section lays the foundation for the RL of multi-modal models. We first formalize the notation (§2.1), casting text and image generation as a Markov Decision Process. Next, we examine three alignment paradigms (§2.2): RL from Human Feedback (RLHF), Group-Relative Policy Optimization (GRPO), and Reinforcement Learning with Verifiable Rewards (RLVR), each aligning policies via human preferences or deterministic checks. Finally, §2.3 reviews the core policy gradient methods (PPO, GRPO) and their adaptations to visual reasoning and generation.

#### 2.1 Notation and Problem Formulation

We cast text- or image-generation as an episodic Markov decision process. We treat the user prompt p as the initial state s0 = p. At timestep t, the state is the prompt plus all previously generated actions:

### st = p, a1,...,at−1 . (1)

A continuation is the full action sequence {a1,...,aT}, where each token at ∈ A is sampled autoregressively from the policy:

### πθ at | st = πθ at | p,a1,...,at−1 . (2)

In words, the prompt anchors the state sequence, and each new action is chosen in the context of that prompt and the tokens already produced.

Skywork R1V2

[Figure 1]

[Figure 2]

VideoChat-R1

Multimodal LLM

[Figure 3]

InstructGPT

VRAG-RL

[Figure 4]

DeepEyes

Visual-RFT

MM-Eureka

[Figure 5]

[Figure 6]

R1-VL

[Figure 7]

VLM-R1

Perception-R1

WeThink

V-Triune

[Figure 8]

DDPO

Deepseek-R1

ViCrit

DPOK

Visual Gen.

[Figure 9]

[Figure 10]

[Figure 11]

TexForce

DanceGRPO

LOOP

RePrompt

ReasonGen-R1 FocusDiff

Visual RL

GoT-R1

[Figure 12]

[Figure 13]

Selftok

[Figure 14]

[Figure 15]

VARGPT-v1.1

Unified Model

[Figure 16]

UnifiedReward-Think

UnifiedReward

[Figure 17]

##### RLDG

[Figure 18]

RIPT-VLA

[Figure 19]

DreamerV3 DeepMind

HIL-SERL

ReinboT

VLA Model

[Figure 20]

GUI-R1

[Figure 21]

Publicly Available/Unavailable

iRe-VLA

[Figure 22]

Robot-R1 VLN-R1

[Figure 23]

##### OctoNav-R1

[Figure 24]

2025 May

2023 Jan

2024 Oct

2025 Oct

- Fig. 1: Timeline of Representative Visual Reinforcement Learning Models. The figure presents a chronological overview of key Visual RL models from 2023 to 2025, organized into four tracks: Multimodal LLM, Visual Generation, Unified Models, and VLA Models.

A fixed reference model (e.g., the supervised fine-tuned checkpoint) is denoted πref. Human preferences are distilled into a scalar reward model Rϕ(p,y), replacing the reward from the unknown environment. We write ρt(θ) = πθ(at | st) πθ

Bradley–Terry likelihood:

### LRM = −

y log σ Rϕ(p,yA) − Rϕ(p,yB)

(p,yA,yB)

+ (1 − y) log σ Rϕ(p,yB) − Rϕ(p,yA) , (3)

(at | st) for the importance ratio between new and behavior policies. Aˆt is used to denote the advantage estimate. PPO reduces the variance of Aˆt with a learned critic Vˆψ, whereas GRPO replaces the critic by a group-relative baseline computed from a set O = {ai}Gi=1 of continuations that share the same prompt. All two algorithms add a KL regulariser KL πθ(·|p)∥πref(·|p) weighted by β to keep the updated policy close to the reference. Unless stated otherwise, expectations E[·] are over prompts p∼D and continuations drawn from the specified policy.

old

where σ(·) is the logistic function. After convergence, Rϕ provides a dense, differentiable proxy for human preference.

Policy optimization. The policy πθ is finally fine-tuned by maximizing (i) the learned reward, (ii) a KL penalty that keeps the policy close to the supervised-fine-tuned baseline πSFT, and (iii) an optional log-likelihood regulariser on the original pre-training distribution, as introduced in InstructGPT [3]1:

E(p,y)∼πθ Rϕ(p,y)

### −β Ep KL πθ(·|p) ∥ πSFT(·|p)

max

θ

SFTanchoring

reward

. (4)

### + γ Ex∼D

### log πθ(x)

#### 2.2 Alignment Paradigms

pretrain

pre-training log-likelihood

- 2.2.1 RL from Human Feedback

In practice, the first two terms are optimised with KLregularised PPO over minibatches of sampled continuations, while the third term adds the pre-training gradients (“PPOptx” in [3]) to mitigate performance regressions on the original corpus.

RLHF [3] extends the underlying MDP with pairwise preference data curated from human annotators. Each preference example is a triple (p, yA, yB), where p is the prompt (or state sequence) and (yA,yB) are two candidate continuations (trajectories, images, etc.); the label y∈{0,1} records which continuation is preferred, as shown in Figure 2.

Three-stage recipe. Most modern RLHF pipelines follow the three-stage recipe, as shown in Figure 2 (a). Step 1: Collect

1. The coefficients β and γ respectively control the strength of the KL penalty and the pre-training log-likelihood term. Setting γ =0 recovers the standard PPO objective.

Reward-model learning. A scalar reward model Rϕ is trained to reproduce the pairwise ordering via a

TABLE 1: Glossary of Symbols for Visual Reinforcement Learning. It consolidates the notation that recurs across Sections §2.1–§2.3, with the rightmost column pointing to each appearance of the symbol.

Symbol Alias Meaning Appears in p prompt User prompt (initial state) §2.1, §2.2.1, §2.3.1, §2.3.2 at action Token / pixel patch / diffusion noise at step t §2.1, Eq. (8), §2.3.2 y traj Full continuation (a1, . . . , aT ) §2.1, §2.2.1, §2.2.3 yi continuation i-th continuation in a GRPO group §2.1, §2.3.2 st state Prompt plus previously generated actions §2.1, §2.3.1, §2.3.2 πθ policy Trainable model (current parameters) §2.1, §2.2.1, §2.2.2, §2.3.1, §2.3.2 πθold behaviour policy Frozen policy that produced current batch Eq. (10), §2.3.2 πSFT SFT baseline Supervised-fine-tuned checkpoint §2.2.1, §2.3.1 πref reference Policy used in KL regulariser §2.1, §2.3.1, §2.3.2 ρt ratio Importance weight πθ/πθold Eq. (10), §2.3.2 Vψ critic Value network predicting future return §2.3.1 Aˆt advantage GAE advantage (token-level) Eq. (9), §2.3.1 Aˆi,t group adv. Group-normalised advantage (GRPO) Eq. (12), §2.3.2

O = {at}G1 group Set of G continuations for one prompt at t timestep §2.1, §2.3.2 G group size Number of continuations per prompt §2.1, §2.3.2

rϕ(st, at) token reward Immediate reward from frozen preference model §2.3.1 ri token reward Reward of the i-th continuation in group §2.3.2 mean(·) mean Group reward mean in GRPO §2.3.2 std(·) std Group reward standard deviation in GRPO §2.3.2 Rϕ(p, y) RM Sequence-level reward model (RLHF) §2.2.1 ϵ clip PPO clipping threshold Eq. (10), §2.3.2 β KL weight Weight balancing KL vs. reward Eq. (8), §2.2.1, §2.3.2 KL(·∥·) KL Divergence between policy and reference §2.2.1, §2.3.1, §2.3.2 DKL(p) est. KL Token-average KL estimator in GRPO Eq. (13)

demonstration data, and train a supervised policy; Step 2: Collect comparison data, and train a reward model; Step 3: Optimize a policy πθ against the reward model using PPO. The paradigm was pioneered by Christiano et al., [23], who trained Atari and robotic agents from pairwise human preferences. Ouyang et al., [3] later scaled the recipe to large language models (InstructGPT) by coupling preference modeling with PPO. For vision, reward models such as ImageReward [21] and Human Preference Score (HPS) [24] supply dense aesthetic signals that guide text-to-image diffusion and related tasks.

- 2.2.2 Direct Preference Optimization

Direct Preference Optimisation (DPO) [2] takes exactly the same pairwise-preference data as RLHF but removes the intermediate reward-model and RL loop. Instead, it derives a closed-form, supervised objective that implicitly enforces a KL constraint to a frozen reference policy πref, as shown in Figure 2 (b).

Closed-form objective. For every prompt p annotators rank two continuations yA,yB and order them so that yA is the preferred continuation (“winner”) and yB the nonpreferred one (“loser”). Thus the dataset consists of triples (p, yA, yB) ∼ D. Let πref be a frozen reference policy (e.g., the SFT checkpoint) and let β > 0 be a temperature hyperparameter. DPO minimizes:

LDPO = −E(p,yA,yB)∼D log σ β ∆θ(p,yA,yB) , (5)

where the log-odds gap is:

πθ(yA | p) πref(yA | p) − log

πθ(yB | p) πref(yB | p)

∆θ(p,yA,yB) = log

= log πθ(yA | p) − log πθ(yB | p) − log πref(yA | p) − log πref(yB | p) . (6)

The logistic function σ(z) = 1/(1 + e−z) turns the gap into a binary-classification loss; training proceeds with standard maximum-likelihood gradients, no reward model, value network, or importance sampling is required.

2.2.3 Reinforcement Learning with Verifiable Rewards

Reinforcement Learning with Verifiable Rewards (RLVR) eliminates the subjectivity and data-collection cost of RLHF by replacing pairwise human preferences with deterministic, programmatically checkable reward signals v : (p,y)  → {0,1,...,K}. Typical examples include pass/fail unit tests for code synthesis, exact-match answers in mathematics, IoU/DICE thresholds for segmentation, or formal outputformat validators (e.g., LeetCode compiler). Because the reward is generated online by execution or metric evaluation, RLVR removes both (i) the reward-model training stage of RLHF and (ii) the contrastive surrogate loss of DPO, while still enabling substantial policy improvements beyond supervised learning [4], [25], [26], as shown in Figure 2(c).

Verifiable reward. For a prompt (state) p and a sampled continuation y, a verifier returns:

### r(p,y) = v(p,y) ∈ {0,1}, (7)

e.g., “ pass” if the generated program solves all hidden tests. The same idea applies to vision: a generated mask that

PPO

[Figure 25]

y > y

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

𝑝 𝑅∅(𝑝,𝑦)

[Figure 30]

[Figure 31]

Preference Data Reward Model 𝑅∅

Policy Model 𝜋 Reward Model 𝑅∅

Data Policy Model 𝜋

Step 2 : Train Reward Model

Step 1 : Train Supervised Policy

Step 3 : Optimize Policy with RL

(a) RL from Human Feedback (RLHF)

[Figure 32]

𝑝

y > y

[Figure 33]

[Figure 34]

𝜋𝜃(𝑦A ∣ 𝑝) 𝜋ref(𝑦A ∣ 𝑝)

𝜋𝜃(𝑦B ∣ 𝑝) 𝜋ref(𝑦B ∣ 𝑝)

[Figure 35]

[Figure 36]

Policy Model 𝜋

Δ𝜃(𝑝,𝑦A,𝑦B) = log

− log

[Figure 37]

[Figure 38]

𝑝

Preference Data

Data Policy Model 𝜋

without reward model

Frozen Reference Model 𝜋

Step 2 : Optimize Policy with DPO

Step 1 : Train Supervised Policy

(b) Direct Preference Optimization (DPO)

Rule-based (e.g., LeetCode compiler)

[Figure 39]

GRPO

[Figure 40]

[Figure 41]

𝑝

𝛼, if correct, 0, otherwise

𝜐 𝑝,𝑦 =

Data Policy Model 𝜋

Policy Model 𝜋

Verifiable Reward 𝜐 𝑝,𝑦

Step 2 : Optimize Policy against Verifiable Reward

Step 1 : Train Supervised Policy

(c) RL with Verifiable Rewards (RLVR)

- Fig. 2: Three Alignment Paradigms for Reinforcement Learning. (a) RLHF learns a reward model from human preference data and optimizes the policy via PPO. (b) DPO removes the reward model and directly optimizes a contrastive objective against a frozen reference. (c) RLVR replaces subjective preferences with deterministic verifiable signals and trains the policy using GRPO.

attains IoU ≥ 0.9 with ground truth, or an image whose CLIP similarity exceeds a hard threshold can be awarded r = 1. Current most RLVR systems (e.g., DeepSeekMath, Deepseek-R1) adopt GRPO [25] (see Equ. 12) and standard KL regularization to train the policy model.

Optimization Objective

𝐴

p prompt

KL penalty

RLVR therefore follows a concise two-stage pipeline. Step 1: Supervised policy pre-training on demonstrations {(p,y)}, producing the initial πSFT. Step 2: RL fine-tuning with GRPO/PPO against the on-the-fly verifiable reward v(p,y), optionally mixing in a small percentage of teacher-forced SFT updates to stabilise training.

- (a) PPO

[Figure 42]

[Figure 43]

[Figure 44]

Policy Model 𝜋

[Figure 45]

[Figure 46]

[Figure 47]

Reference Model 𝜋

[Figure 48]

[Figure 49]

[Figure 50]

Reward Model 𝑟∅ Value

[Figure 51]

[Figure 52]

[Figure 53]

Model 𝑉

𝑟

𝑉 (𝑠 )

GAE

[Figure 54]

[Figure 55]

[Figure 56]

Policy Model 𝜋

𝜋 𝑎 ∣ 𝑠 𝜋 𝑎 ∣ 𝑠

[Figure 57]

[Figure 58]

Frozen Model Trainable Model

[Figure 59]

[Figure 60]

log

𝜋 𝑎 ∣ 𝑠 𝜋 𝑎 ∣ 𝑠

{𝑎 ,𝑠 }

ℒ

ℒ

- (b) GRPO

[Figure 61]

[Figure 62]

[Figure 63]

Policy Model 𝜋

𝐺1 log 𝐴 , 

𝜋 𝑎 ,  ∣ 𝑠 ,  𝜋 𝑎 ,  ∣ 𝑠 , 

1 𝐺

𝜋 𝑎 ,  ∣ 𝑠 ,  𝜋 𝑎 ,  ∣ 𝑠 , 

p prompt

KL penalty

[Figure 64]

[Figure 65]

[Figure 66]

Reference Model 𝜋

###### ℒ

[Figure 67]

[Figure 68]

[Figure 69]

Policy Model 𝜋

#### 2.3 Policy-Optimization Algorithms

𝑎 , 

𝑟 ,  𝑟 , 

𝐴 ,  𝐴 , 

- 2.3.1 Proximal Policy Optimization Proximal Policy Optimization (PPO) [27] is a first order trust region method that updates the policy πθ while staying close

[Figure 70]

[Figure 71]

[Figure 72]

𝑟 ,  − mean(𝐫·,𝒕) std(𝐫·,𝒕) Group Computation

Reward Model 𝑟∅

𝑎 ,  𝑎 , 

…

…

…

𝑟 , 

𝐴 , 

at every step, as shown in Figure 3 (a). In text- or image-generation problems we treat a prompt p as the initial state s0 and the continuation {a1,...,aT} as the trajectory. Define the importance-sampling ratio ρt(θ) and immediate reward rϕ:

to the previous policy πθ

Fig. 3: Two Representative Policy Optimization Algorithms for LLM. PPO (a) uses a learned value model Vψ for advantage estimation and injects the KL penalty at each token. GRPO (b) removes the value model, computes groupnormalized advantages Aˆi,t across G continuations, and applies an explicit prompt-level KL penalty.

old

- • Importance–sampling ratio

ρt(θ) =

πθ(at | st) πθ

old

(at | st)

,

which re-weights the gradient estimate from the behavior policy to the updated policy.

- • Immediate reward rϕ(st,at), provided by a frozen reward model rϕ that has been pre-trained to approximate human preference.

• Value baseline Vψ(st), produced by a learned value network Vψ that regresses the expected discounted return from state st.

With the KL-regularised reward between the policy model and reference model, rtPPO can be defined:

πθ(at | st) πref(at | st)

rtPPO = rϕ(st,at) − β log

, (8)

where the KL term(latter item) keeps the updated policy πθ from drifting too far from the frozen reference πref. β balances exploration (via KL proximity to the frozen reference model πref) against exploitation of the reward model. A larger β implies stricter proximity and thus safer but potentially slower learning. Then the generalized advantage estimator (GAE) [28] produces Aˆt:

### Aˆt = GAE rtPPO, Vψ , (9)

where GAE computes advantage values by exponentially weighting multi-step reward estimates, providing a smooth trade-off between low-variance learning and high-variance Monte Carlo returns.

The surrogate objective maximized by PPO is then:

### LPPO = Et min ρt(θ)Aˆt,clip ρt(θ),1 − ϵ,1 + ϵ A ˆt , (10)

where the hyper-parameter ϵ ∈ (0,1) controls the width of the trust region. Accurate and low-variance Aˆt estimates are therefore critical, they direct each policy update and ultimately determine the stability and sample efficiency of PPO.

- 2.3.2 Group Relative Policy Optimization Group Relative Policy Optimisation (GRPO) [25] extends PPO by discarding the learned value (critic) network and replacing it with a group-relative baseline computed from multiple outputs sampled for the same prompt. This design markedly reduces memory consumption while aligning the advantage estimator with the comparison-based reward model, as shown in Figure 3 (b).

Group Relative Baseline. For each prompt p, we sample a group of G full continuations a1,...,aG ∼ πθ(· | p), where each continuation a·,t = (a1,t,...,aG,t) is a sequence of tokens indexed by timestep t. The frozen reward model rϕ(p,ai,t) then assigns a scalar score to each token ai,t conditioned on the prompt. These sequence level rewards are then normalized across the group to compute a grouprelative advantage signal:

ri,t − mean(r·,t) std(r·,t)

Aˆi,t = r˜i,t =

, (11)

where mean(·) and std(·) denote the mean and standard deviation functions used to compute the group relative advantage. The same Aˆi is reused for every token ai,t in the continuation, producing the clipped surrogate:

G

LGRPO = Ep∼D G 1

i=1

|ai|

min ρi,t Aˆi,t,

1 |ai|

t=1

clip ρi,t,1 − ϵ,1 + ϵ A ˆi,t

(12)

− β Ep DKL πθ(· | p) πref(· | p) ,

### where ρi,t = πθ(ai,t | si,t) πθ

(ai,t | si,t). The explicit KL penalty DKL(·) keeps πθ near the reference πref, while the

old

group-relative advantage Aˆi replaces the value baseline Vψ, roughly halving memory and compute yet retaining a lowvariance learning signal.

Prompt-level KL estimator. Instead of injecting a tokenwise penalty into the reward (as PPO does with β log π

πref ), GRPO adds a separate prompt-level regulariser. With the G sampled continuations we form an unbiased token-average estimate:

θ

|ai|

G

πθ(ai,t | si,t) πref(ai,t | si,t)

1 G

1 |ai|

, (13)

DKL(p) =

log

t=1

i=1

which measures how far the current policy drifts from the frozen reference πref over the whole continuation.

Relative to PPO in Equ. (10), GRPO introduces two key improvements: 1) Eliminates the value (critic) network. Variance reduction is achieved by a group-relative baseline, leading to lower memory footprint and fewer hyper-parameters. 2) Separates the KL loss channel. The KL divergence is optimized as an explicit regulariser rather than folded into the advantage, yielding a transparent trade-off between reward maximization and reference anchoring.

3 REINFORCEMENT LEARNING IN VISION

#### 3.1 Multi-Modal Large Language Models

We categorize the works into four coherent groups, each defined by shared RL-driven objectives and internal reasoning mechanisms.

3.1.1 Conventional RL-based MLLMs

We refer to conventional RL-based MLLMs as approaches that apply reinforcement learning primarily to align a vision–language backbone with verifiable, task-level rewards, without explicitly modeling multi-step chain-of-thought reasoning. Typical works RePIC [29], GoalLadder [30], DriveR1 [31] and VLM-R1 [32] replace preference models with deterministic validators (e.g., exact-match, IoU, BLEU) and optimize the policy by GRPO/PPO variants under a KL regulariser. This design yields stable value-free training, improves zero-shot robustness on captioning, grounding and autonomous-driving benchmarks, and substantially reduces the annotation cost typically incurred by supervised finetuning.

Recent extensions demonstrate the flexibility of this paradigm. GRPO-CARE [33] introduces consistency-aware group normalization to mitigate reward variance, while Q-Ponder [34] adds a pondering controller. From a data perspective, MoDoMoDo formulates a multi-domain mixture optimization that predicts reward distributions and selects optimal curricula [35]; V-Triune further unifies perception and reasoning tasks within a single triple-objective pipeline, empirically validating that rule-based RL scales to diverse visual signals [36]. Collectively, these studies indicate that (i) verifiable rewards can serve as a low-cost alternative to human feedback, (ii) group-relative objectives offer higher training stability than token-level PPO on heterogeneous visual tasks, and (iii) curriculum or data-mixture scheduling is emerging as a key ingredient for broad generalization.

e.g., RePIC [29], GoalLadder [30], Drive-R1 [31], VLM-R1 [32], GRPO-CARE [33], Q-Ponder [34], MoDoMoDo [35], V-Triune [36],Vision-R1 [37], ProxyThinker [38], Jigsaw-R1 [39], SRPO [40], R1-Onevision [41], Reason-RFT [42], Segagent [43], VisualQuality-R1 [44], Zhang et al. [45], SATORI-R1 [46], Skywork r1v2 [47], Yang et al. [48], THOR [49] Caprl [50], LENS [51], Docr1 [52], Bigcharts-r1 [53], etc.

Conventional RL-based Frameworks(§3.1.1)

| | |
|---|---|
| | |
| | |
| | |

Spatial & 3D Perception (§3.1.2)

e.g., Omni-R1 [54], DIP-R1 [55], Metaspatial [56], Bindgpt [57], Scene-R1 [58], Perception-R1 [59], VisRL [60], ViCrit [61], VisionReasoner [62], ViGoRL [63], Visual-RFT [64], etc.

e.g., SVQA-R1 [65], VL-GenRM [66], RACRO [67], EasyARC [68], STAR-R1 [69], Visionary-R1 [70], UniVG-R1 [71] , EchoInk-R1 [72], WeThink [73], G1 [74], GThinker [75], Observe-R1 [76], Mm-eureka [77], Perception-R1 [78], MiMo-VL [79], SearchExpert [80], etc.

Multimodal Large Language Models (§3.1)

Think about Image

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |
| | |

Image Reasoning (§3.1.3)

e.g., Visual Planning [81], GRIT [82], VILASR [83], BRPO [84], Ground-R1 [85], Pixel Reasoner [86], DeepEyes [6] , TACO [87], VRAG-RL [88], VisTA [89],TGI [90], Chain-of-Focus [7], Openthinkimg [91], Active-O3 [92] , RRVF [93], Visionthink [94], etc.

Think with Image

e.g., VQ-Insight [95], TW-GRPO [96], EgoVLM [97], VAU-R1 [98], DeepVideo-R [99], TimeMaster [100], VideoR1 [101], VideoChat-R1 [102], Temporal-RLT [103], Moss-chatv [104], Tempsamp-r1 [105], Video-mtr [106], Thinking with videos [107], MIRO [108], Chunk-GRPO [109], Onereward [110], Uso [111], Pref-grpo [112] , Tempflow-grpo [113], etc.

Video Reasoning(§3.1.4)

ReinforcementLearninginVision

e.g., ImageReward [21], ReasonGen-R1 [114], FocusDiff [115], Dpok [116], LOOP [117], Prefpaint [118], RLRF [119] , GoT-R1 [120], D-Fusion [121], Black et al., [13], DiffusionDPO [122], Diffusion-KTO [123],DiffusionRPO [124], Miao et al., [125], Parrot [126], DPG-T2I [127], RPO [128], RePrompt [129], RLCM [130], B2-DiffuRL [131], DiffPPO [132], Simplear [133], LLaVA-Reward [134], Flow-grpo [135], Gallici et al. [136], TexForce [137], etc.

Image Generation(§3.2.1)

| | |
|---|---|
| | |
| | |

Visual Generation(§3.2)

Video Generation(§3.2.2) e.g.,InstructvideoDanceGRPO[144],[138Videoscore], InfLVG[145[139],],GradeoPhys-AR[146[140], VideoRM], VideoReward[147], etc[141. ], TeViR [142], GAPO [143],

3D Generation(§3.2.3) e.g., DreamCS [148], Dreamreward [149], DreamDPO [14], Nabla-R2D3 [150], etc.

Unified RL(§3.3.1) e.g., UniRL [15], CoRL [22], SelfTok [17], Hermesflow [151], UnifiedReward [152], UnifiedReward-Think [153], etc.

| | |
|---|---|
| | |

Unified Model(§3.3)

Task-specific RL(§3.3.2) e.g., Vargpt-v1. 1 [154], Emu3 [16], X-Omni [155], etc.

GUI Interaction(§3.4.1) e.g.,, Mobile-R1GUI-R1[160[10],], SE-GUIGTA1 [161[11],],LPOUI-R1[162[156], ],Ui-tarsUIShift[163[157], RUIG], AgentCPM-GUI[164], Appvlm[158[165], MobileGUI-RL], DigiRL [166],[19ARPO], ProgRM[167], [etc159. ]

| | |
|---|---|
| | |

Vision Language Action Models(§3.4)

Visual Navigation(§3.4.2) e.g.,EmbCLIP-AgentOctoNav-R1[174[168],],etcMore. [169], RAPID [20], VLN-R1 [170], Flare [171], IRL-VLA [172], Embodied-R [173],

Visual Manipulation(§3.4.3) e.g.,ReinBotTGRPO[180],[175VIKI-R], RFTF[181[],176Robot-R1], RLVLA[182[177],],RLDGVLA-RL[183[],8],HIL-SERLConRFT [[178184],],iRe-VLAModem [[185179],], VPGRIPT-VLA[186], [etc9],.

- Fig. 4: Overall taxonomy of reinforcement-learning research in vision. The chart groups existing work by high-level domain (MLLMs, visual generation, unified models, and vision-language action agents) and then by finer-grained tasks, illustrating representative papers for each branch.

- 3.1.2 Spatial and 3D Perception

- 2D perception. Perception centric works applies RL to sharpen object detection, segmentation and grounding without engaging in lengthy chain–of–thought reasoning. OmniR1 [54] introduces a two-system (global–local) GRPO pipeline that verifies predictions via rule-based metrics, yielding notable improvements on region-level benchmarks and emotion-recognition tasks. DIP-R1 [55] further decomposes perception into step-wise “inspect→observe→act” cycles, where each stage receives deterministic IoU or countingbased rewards to boost fine-grained detection. PerceptionR1 [59] revisits the effect of GRPO on a spectrum of detection and OCR datasets. Complementing these, VisRL [60] frames intention-guided focus selection as an RL sub-policy, eliminating expensive region labels and consistently outperforming supervised strong baselines on visual grounding tasks.

3D perception. Beyond 2-D, several studies leverage RL to align multimodal models with physically consistent 3D layouts. MetaSpatial [56] employs rendered depth/IoU rewards to refine spatial reasoning for AR/VR scene generation, whereas Scene-R1 [58] couples video-grounded snippet selection with a two-stage grounding policy to learn 3-D scene structure without point-level supervision. At molecular scale, BindGPT [57] treats atom placement as sequential actions and uses binding-affinity estimators as verifiable rewards, demonstrating the scalability of perception-focused RL to 3-D biochemical design. Collec-

tively, these approaches underscore a common recipe: (i) formulate detection/segmentation/3-D alignment as Markov decision problems, (ii) craft deterministic spatial rewards (e.g., IoU, depth consistency, binding energy), and (iii) finetune pretrained VLM backbones via GRPO/PPO for stable perception enhancement—thereby differentiating themselves from reasoning-oriented RL variants.

3.1.3 Image Reasoning

Thinking about Images methods enhance multimodal reasoning by verbalising observations of a static picture before producing an answer, but the visual content itself is not modified during inference. By contrast, Thinking with Images elevates the picture to an active, external workspace: models iteratively generate, crop, highlight, sketch or insert explicit visual annotations as tokens in their chain-of-thought, thereby aligning linguistic logic with grounded visual evidence.

Think about Image. Early think about image works for spatial VQA employs view-consistent or transformationinvariant objectives, such as SVQA-R1 [65] and STARR1 [69]. VL-GenRM [66] and RACRO [67] refine preference data or caption rewards to curb hallucinations. Benchmarkoriented efforts such as EasyARC [68] offer procedurally generated, fully verifiable tasks that suit outcome-based reinforcement learning. To mitigate shortcut reliance and improve generality, Visionary-R1 [70] enforces image interpretation before reasoning, whereas UniVG-R1 [71] unifies referring, captioning, and detection by coupling a grounding

corpus with GRPO fine-tuning. Extensions such as EchoInkR1 [72] further enrich visual reasoning by integrating audio–visual synchrony under GRPO optimization. Meanwhile, curriculum-driven frameworks—WeThink, G1, GThinker, and Observe-R1 progressively increase task complexity or introduce re-thinking cues (e.g., difficulty ladders, multimodal format constraints) to cultivate deeper and more structured reasoning capabilities in MLLMs. These methods show that language only RL with well-designed visual correctness rewards significantly improves model accuracy, robustness, and out-of-distribution performance.

Think with Image. Early think with image works grounds reasoning via discrete region–level operations: GRIT [82] interleaves bounding-box tokens with language and trains under GRPO to maximize both answer correctness and box fidelity, while VILASR [83] generalizes this idea to multi-view and video settings, enforcing cross-view spatial consistency. Ground-R1 [85] and BRPO [84] adopt twostage pipelines that first highlight evidence regions (via IoU-based or reflection rewards) before verbal reasoning. A complementary thread explores pixel-space or sequencelevel manipulation. Visual Planning [81] replaces text chains with imagined image roll-outs rewarded by downstream task success; Pixel Reasoner [86] augments the action space with crop, erase and paint primitives and balances exploration through curiosity-driven rewards, whereas DeepEyes [6] shows that end-to-end RL can spontaneously induce such visual thinking behaviours. Finally, TACO [87] introduces a think–answer consistency objective that resamples long visual–verbal chains until their intermediate edits align with the final answer. Together, these systems demonstrate that explicitly generating or editing visual artefacts during reasoning optimized via GRPO or R1-style outcome RL—yields more faithful, interpretable and robust image understanding than language only approaches.

- 3.1.4 Video Reasoning

Video reasoning extends the capabilities of MLLMs to process temporal dynamics, requiring not only spatial perception but also sequential understanding and causal inference. Recent works in this domain have proposed diverse approaches to tackle complex reasoning over video inputs. For instance, VQ-Insight [95] introduces a hierarchical reward design and self-consistency voting mechanism tailored to the question–answering process over long videos. TW-GRPO [96] combines token wise credit assignment with GRPO-style optimization to improve fine-grained temporal alignment between textual reasoning and video evidence. Meanwhile, several R1-style frameworks have been developed to unlock video understanding in complex real-world or egocentric settings. EgoVLM [97] and VAU-R1 [98] focus on egocentric video reasoning with visual memory and utility-based rewards. DeepVideo-R [99] integrates dense video encodings and external reward functions to supervise long-horizon reasoning. TimeMaster [100] explicitly structures temporal abstraction and reasoning via curriculum learning, while VideoR1 [101] proposes a scalable RL framework for videobased QA tasks across multiple domains. Collectively, these works highlight the importance of aligning temporal representations with language trajectories through reinforcement

learning, paving the way for robust and generalizable video reasoning agents.

#### 3.2 Visual Generation

- 3.2.1 Image Generation RL for image generation models departs from the language counterpart in both action space continuous diffusion steps or prompt refinements and reward design, which must capture perceptual quality, text–image alignment, and subject fidelity. A first family learns an explicit visual reward model: ImageReward [21] supplies human-preference scores that drive policy-gradient fine-tuning of diffusion backbones in DiffPPO [132], Dpok [116], and FocusDiff [115]. A second line bypasses reward modelling by optimising pairwise or unary preferences: DDPO [13], DiffusionDPO [122], DiffusionKTO [123], and DiffusionRPO [124] treat denoising trajectories as MDPs and apply R1/GRPO updates to maximise comparative human feedback. Beyond alignment, works such as PrefPaint [118], Parrot [126], and RLRF [119] craft multi-objective or render-and-compare rewards to refine aesthetics, diversity, or vector graphics. RL has also been used to inject reasoning and prompt adaptation: ReasonGenR1 [114], GoT-R1 [120] and RePrompt [129] first generate textual plans or improved prompts, then reinforce the generator for coherent scene synthesis. Finally, personalisation methods—DPG-T2I [127], RPO [128], and B2-DiffuRL [131] optimize rewards that measure subject fidelity under scarce reference images. Collectively, these studies show that RL, armed with perceptual or preference-based rewards, can steer diffusion models toward higher realism, stronger prompt faithfulness, controllable layout, and user-specific appearance capabilities difficult to achieve with likelihood training alone.
- 3.2.2 Video Generation Applying RL to video generation introduces challenges absent in images: rewards must capture temporal coherence, motion naturalness, and text–video alignment across hundreds of frames. Early work such as InstructVideo [144] repurposed image scorers and applied PPO to refine short clips, whereas VideoRM [147] and VideoReward [141] learn dedicated preference models that grade entire sequences for smoothness, consistency and prompt faithfulness. Building on GRPO/R1, DanceGRPO [138] shows that group-normalized returns stabilize long-horizon optimization and boost aesthetic and alignment scores across diffusion and flow backbones.

Beyond preference alignment, specialized rewards tackle domain-specific goals. GAPO [143] exploits gap-aware ranking to fine-tune anime videos; Phys-AR [140] penalizes violations of physics to yield plausible trajectories; and InfLVG [139] trains an inference-time policy that retains only context tokens beneficial for multi-scene coherence. Auxiliary critics further facilitate training: VideoScore [145] and Gradeo [146] offer explainable, multi-factor scores, while TeViR [142] feeds imagined video roll-outs as dense rewards to downstream control agents. Collectively, these studies demonstrate that carefully crafted sequence level rewards and group-relative policy updates are key to producing temporally consistent, semantically aligned, and physically plausible videos capabilities difficult to obtain with likelihood training alone.

- 3.2.3 3D Generation RL for 3D generation differs from 2D and video tasks as rewards operate on volumetric structures or rendered views, often requiring expensive render-and-compare loops. DreamCS [148] pioneers this paradigm by framing text-tomesh synthesis as a Markov decision process: a diffusion backbone proposes coarse shapes, then a policy refines vertex positions under a reward that jointly measures silhouette IoU, CLIP text-mesh alignment, and mesh smoothness. On the reward side, DreamReward [149] introduces a large-scale human preference dataset of rendered 3-D assets and trains a geometry-aware critic that scores prompts, renders, and latent SDFs; the learned signal enables stable PPO fine-tuning of existing text-to-shape models.

A complementary line adopts direct preference optimization. DreamDPO [14] extends DPO to NeRF and mesh diffusion models by sampling paired 3D outputs and maximizing the margin dictated by human comparisons, achieving superior prompt fidelity without an explicit value network. Finally, Nabla-R2D3 [150] tackles interactive scene editing: the agent sequentially adds, deletes, or transforms objects in a

- 3D scene; reward is computed via real-time rasterized views and task-specific validators (e.g., occupancy, spatial relations). Group-relative policy optimization (R2D3) stabilizes training despite sparse feedback. Together, these studies demonstrate that RL, equipped with geometry-aware or render-based rewards, provides an effective mechanism for controlling structural integrity, text alignment, and interactive editability capabilities that standard likelihood or score-distillation objectives struggle to capture in the 3D domain.

- 3.2.4 Reward Design and Preference Modeling for Visual Generation As shown in Figure 5, the reward design in RL-based visual generation can broadly be divided into three paradigms: (a) Human-Centric Preference Optimization (using human feedback or learned reward models for video alignment and fidelity). Representative works include ImageReward [21], which pioneered large-scale text-to-image reward modeling from human comparisons and provided a critic for diffusion fine-tuning. Dpok [116] and DiffPPO [132] adapted PPO with KL regularization and human aesthetic scores to enhance alignment and fidelity. Extending these ideas to video, VideoRM [147] and VideoReward [141] trained reward models on large-scale human (and GPT-assisted) video comparisons, enabling RL fine-tuning of text-to-video generation. (b) Multimodal Reasoning-Based Evaluation (using a question-answering or reasoning module to score consistency between videos and textual descriptions). For example, LLaVA-Reward [134] leverages a large VLM to evaluate images across multiple dimensions including alignment, fidelity, and safety, providing a multi-faceted reward signal that better approximates human judgment. and (c) Metric-Driven Objective Optimization (directly optimizing quantitative metrics like FID or IoU for quality).

#### 3.3 Unified Model

Task specific RL maximizes a reward tied to a single objective, whereas Unified RL optimizes a shared policy and reward across multiple vision–language tasks (e.g., understanding and generation).

- 3.3.1 Unified RL

Unlike task specific pipelines that attach RL to a single downstream objective, Unified RL methods optimize a shared policy across heterogeneous multimodal tasks under a single reinforcement signal. The central idea is to merge understanding and generation trajectories into one training loop typically using Group-Relative or R1-style methods.

UniRL [15] exemplifies this paradigm: a visual autoregressive backbone is first instruction-tuned, then jointly finetuned on VQA, captioning and image generation with a blended reward measuring textual correctness, CLIP-based alignment, and aesthetic quality. CoRL [22] pushes the idea further by alternating “co-understanding” and “cogeneration” batches within the same GRPO step. To address inefficiency in dense token spaces, SelfTok [17] discretises multi-modal actions into a self-evolving token set and demonstrates that a single RL head can govern retrieval, grounding, and synthesis with minimal extra parameters. Finally, HermesFlow [151] couples an autoregressive text module with a rectified flow image decoder under one crosstask reward, illustrating that diffusion-style and languagestyle policies can be harmonized through unified reinforcement updates. Together, these works suggest that sharing a common RL objective across tasks not only reduces training cost but also encourages emergent cross modal generalization unavailable to isolated, task specific fine-tuning.

- 3.3.2 Task specific RL

In contrast to the unified approaches of §3.3.1, task-specific RL confines the reward signal to a single downstream objective, optimizing one functional head while leaving other capabilities untouched. VARGPT-v1.1 [154] exemplifies this strategy: although the underlying visual autoregressive model can handle both understanding and generation, its RL stage targets only visual generation with DPO. Similarly, Emu3 [16] introduces RL exclusively to polish its image generation branch, which leveraging pair wise human preferences. For the multimodal understanding abilities of model (e.g., captioning, VQA), the work just train this part by task specific fine-tuning alone.

3.4 Vision Language Action Models 3.4.1 GUI Automation

Modern GUI RL research frames screen understanding and action prediction as a vision–language decision process, then employs rule-based or preference rewards to close the perception–action loop. On desktop and web interfaces, GUI-R1 [10] introduces an R1-style rule set that maps click success, text entry, and scroll validity to dense rewards. UI-R1 [156] adds GRPO with a novel action-specific KL term to stabilize longhorizon plans, while SE-GUI [11] applies self-evolutionary filtering to distil high-fidelity trajectories. Focusing on trajectory reuse, UIShift [157] formulates an inverse dynamics objective that lets MLLM learn actions from unlabeled GUI pairs and then refines them via RL. Complementary preference-based frameworks include LPO [162] that rewards spatial proximity for precise clicks. ProgRM [159] injects program-level logical checks, and RUIG [164] leverages instruction grounding with reinforcement signals. Tool-specific

[Figure 73]

[Figure 74]

[Figure 75]

Generative Model (Diffusion/AR)

RL

- （a） Policy

(b) Multimodal Reasoning-Based Evaluation (c) Metric-Driven Objective Optimization

Score/Rank

Rating Rule (Alignment,Fidelity) Images User-Evaluated

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Images Score/Rank Visual Question Answering

[Q1,Q2,···]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Images Score/Rank

[Figure 84]

[Figure 85]

[Figure 86]

- • FID:
- • IoU:

··· Metric-Driven Evaluation

[Figure 87]

[Figure 88]

Distance

A ∩ B A ∪ B

Generative Model (Diffusion/AR)

[Figure 89]

Generative Model (Diffusion/AR)

[Figure 90]

RL Policy

RL Policy

Human-Centric Preference

- （b）Reasoning-BasedMultimodal
- （c）Metric-DrivenOptimization

[Figure 91]

- Fig. 5: Three reward paradigms for RL-based image generation. (a) Human-Centric Preference Optimization: aligns outputs with human aesthetic scores (HPS [24], ImageReward [21]); (b) Multimodal Reasoning-Based Evaluation: scores images via multimodal reasoning consistency (UnifiedReward [152], PARM [187]); (c) Metric-Driven Objective Optimization: minimizes task-specific quantitative metrics such as FID and IoU.

(a) Human-Centric Preference Optimization

baselines such as Ui-tars [163] offer larger action vocabularies yet still rely on rule-driven RL for robust execution.

integrates pose priors for faster convergence in unseen layouts. These works show that using temporal rewards, memory sampling, and environment priors with GRPO/PPO helps VLA agents navigate more reliably and efficiently.

Mobile scenarios introduce latency and on-device constraints. AgentCPM-GUI [158] compresses the action space and conducts GRPO fine-tuning. MobileGUI-RL [19] advances this line via online RL with task-level rewards to improve exploration under limited memory, and MobileR1 [160] extends interactive multi-turn RL to correct error cascades during long tasks. At inference, GTA1 [161] samples multiple action candidates and employs a judge model to pick the best, effectively trading compute for higher success rates. Additional, light-weight models such as Appvlm [165] demonstrate that modest-sized MLLM, after GRPO finetuning, can control smartphone apps with competitive precision. To adaptively reduce the thinking length, the TON [188] proposes a thought-dropout solution during supervised fine-tuning stage, then GRPO skill adaptivley skip unnecessary reasoning process for efficiently thinking.

3.4.3 Visual Manipulation

Visual manipulation tasks (e.g., object relocation, tool use, and multi-step rearrangement) require fine-grained perception and long-horizon planning. Recent works [175], [176] integrate reinforcement learning with vision–language models to enhance generalization, interactivity, and policy consistency. TGRPO [175] introduces a task-grounded reward formulation and group normalized updates to stabilize training for open-ended object manipulation. RFTF [176] applies rule-based rewards to support interactive table top tasks and emphasizes training with minimal human supervision. Meanwhile, RLVLA [177] and VLA-RL [8] explore curriculum-based or progressive reinforcement learning for VLM-based robot agents, achieving high success rates across diverse rearrangement environments.

Collectively, these studies show that GUI agents benefit from rule-verifiable rewards, group-normalzsed policy updates, and preference-guided localization, achieving rapid progress toward reliable, cross-platform automation.

Building on this, ConRFT [178] and iRe-VLA [179] introduce consistency-aware and instruction-refinement strategies respectively, using RL to align visual predictions with physical interaction outcomes. RIPT-VLA focuses on interactive prompting during manipulation, bridging LLM planning and low-level control through reinforced feedback [9]. Finally, ReinBot [180] leverages multimodal rollouts and preferencebased updates to improve real-world manipulation robustness [180]. Collectively, these studies highlight the role of vision-language reasoning, structured reward design, and RL-based refinement in advancing embodied manipulation under complex, language-conditioned settings.

- 3.4.2 Visual Navigation

RL-driven visual navigation research now couples large vision–language models with embodied control, employing group-normalized or time-decayed returns to maintain longhorizon stability. OctoNav-R1 [168] exploits a hybrid RL pipeline with a “think-before-action” ability for VLA model, then translate egocentric frames into low-level actions. Focusing on dataset efficiency, VLN-R1 [170] builds an end-to-end navigator and introduces a time-decayed reward to handle continuous trajectories. At the system level, Flare [171] demonstrates that fine-tuning a multi-task robotics policy with large-scale RL in simulation can generalize to realworld household tasks. Complementary advances include More [169], which augments omni-directional inputs with memory-guided policy distillation, and RAPID [20], which

## 4 METRICS AND BENCHMARKS

Evaluating visual reinforcement learning (RL) with large models requires both traditional RL metrics [234], [235] and new ones designed for complex, open-ended outputs. Metrics

TABLE 2: Overview of evaluation metrics in large-model visual RL. Each task family is broken down into Task Metrics (RL-free external benchmarks), Reward Metrics (how the learning signal is computed), and Model-state Metrics (diagnostics tracked during optimization).

Model-state Metric Mtstate (training diagnostics)

Task Family Task Metric Mset

Reward / Advantage Estimation Msamp (scoring & preference signals)

(RL-free benchmarks / scores)

- • Comprehensive Evaluation: e.g., MME [189], SEED-Bench [190], VQA v2 [191], MMBench [192].
- • OCR: e.g., TextVQA [193], OCR-VQA [194], OCRBench [195].
- • Mathematical: e.g., MathVista [196].
- • Documentation: e.g., ChartQA [196], DocVQA [197], InfoVQA [198].
- • Multilingual: e.g., CMMMU [199], CMMU [200].

- • Reward from Human Preference: e.g., InstructGPT [3], LLaVA-RLHF [201].
- • Verifiable Rewards: Deepseekmath [25], IoU [202], CodeRL [203].
- • Reward from Model Preference: e.g., CriticGPT [204].

- • Output length monitoring: e.g., DPO [2].
- • KL-divergence: e.g., InstructGPT [3].

Multimodal LLMs/ VLMs

###### . . .

. . .

. . .

- • Image Fidelity & Diversity: e.g., FID [205], Inception Score (IS) [206].
- • Pixel-level Reconstruction: e.g., PSNR [207], SSIM [207].
- • Semantic Alignment: e.g., Geneval [208], T2I-CompBench [209], CLIP Score [210], Frechet CLIP Distance [211].
- • Video Fidelity & Diversity: e.g., FVD [212], Video IS [213].

- • Reward from Human Preference: e.g., ImageReward [21], HPS [24], HPS V2 [24], Pick-a-pic [214], VideoReward [141], RichHF-18K [215].
- • Verifiable Rewards: e.g., MotionPrompt [216], DSPO [217], Instructrl4pix [218].
- • Reward from Model Preference: e.g., VideoPrefer [147], PARM [219].

- • Denoising Trajectory Diagnostics: e.g., VARD [220], Inversion-DPO [221].
- • KL-divergence: e.g., DDPO [13], VARD [220].

Visual Generation (Image / Video / 3D)

###### . . .

. . .

. . .

Unified Models • Generation–oriented Task Metrics: e.g., GenEval [208], DPG-Bench [222], ParaPrompts [223].

- • Unified Rewards: e.g., UniRL [15], CoRL [22].
- • Task-specific Rewards: e.g., Vargpt-v1. 1 [154], Emu3 [16].

- • Generation–Understanding Imbalance: e.g., UniRL [15].
- • KL-divergence: e.g., HermesFlow [151]

###### • Understanding–oriented Task Metrics: e.g., MME [189], POPE [224].

. . .

. . .

. . .

- • GUI Action Accuracy (offline): e.g., ScreenSpot [225], ScreenSpot-Pro [226], Ui-vision [227].
- • Task Success rate (online): e.g., Webarena [228], Osworld [229], Windows agent arena [230].
- • Model-based Evaluation: e.g., Agentrewardbench [231], Webworld [232], Digirl [233].
- • Reward from Human Preference: e.g., RFTF [176].
- • Navigation Success & SPL:

- • Rule-based Success (binary) (IoU, Action Accuracy): e.g., UI-R1 [156], ARPO [167], VLA-RL [8].
- • Model Preference Critic: e.g., ProgRM [159].
- • Dense Shaping (distance or coverage): e.g., LPO [162], Gui-r1 [10].

- • Action trajectory length: e.g., Osworld [229].
- • KL penalty for policy stability: e.g., GUI-R1 [10], UI-R1 [156].
- • Output length monitoring: e.g., UI-R1 [156]. . . .

Vision-Language Action Agents (GUI / Navigation / Manip.)

. . .

. . .

like cumulative return and task success rate are still useful especially in tasks involving agents and environments but they are increasingly combined with preference-based evaluations. In this section, we discuss metrics and benchmarks across four major application areas of large-model RL.

#### 4.1 Formalizing Metric Granularity

Let Ptest = {p1,...,pD} denote a fixed set of prompts (inputs) and let a generative policy πθ(y | p) produce an output y (text, image, video, etc.) conditioned on each prompt p ∈ Ptest. As shown in Figuer 6, we distinguish three granularities of evaluation:

Task Metric Mset (Set-level). As illustrated in Fig. 6 (a), set-level metrics evaluate the generative policy πθ over the full prompt set Ptest, by comparing the distribution of generated outputs Ygen = {πθ(· | pi)}Di=1 to a reference set of ground-truth outputs Ygt = {ygti }Di=1. When the evaluation function operates without access to ground-truth outputs such as CLIPScore [236], we define:

1 D

Mset =

D

Ey∼πθ(·|pi) Msamp(ygeni ,pi) , (14)

i=1

where Msamp(y,p) denotes a sample-level reward function applied to each generated output. By contrast, many classical vision metrics do rely on reference outputs such as IoU [202]. For these cases, set-level evaluation is defined as:

1 D

Mset =

D

Ey∼πθ(·|pi) Msamp(ygeni ,ygti ,pi) , (15)

i=1

where each ground-truth output ygti denotes i-th ground truth from the reference set.

Reward/Advantage metric Msamp (Sample-level). As illustrated in Fig. 6 (b), reward and advantage metrics Msamp operate at the granularity of individual input-output pairs, forming the backbone of reinforcement learning in generative settings. Given a prompt pi, the policy πθ generates a sample ygeni , which is then scored by Msamp(ygeni ,pi) to compute a scalar reward or advantage signal. This feedback is used to optimize the policy via reinforcement learning (e.g., PPO [27], DPO [2]). In preference-based learning, the sample-level metric is often learned from human or GPT-4 comparisons [3], [201], or automatically derived via scoring models like CLIPScore [236], or ImageReward [21]. When rewards are reference-dependent (e.g., using PSNR [207] or

suites such as MME [189], SEED-BENCH [190] and MMBENCH [192] measure factual QA, commonsense and multistep chain-of-thought across images. Domain-specific subsets probe OCR (TextVQA [193]), mathematics (MathVista [196]), documents (ChartQA [196]) and multilingual grounding (CMMMU [199]).

Set-level Score 𝑆

p p

𝑦 𝑦

𝑦 𝑦

ℳ (·)

…

…

…

𝜋 (·)

p

𝑦

𝑦

(a) Task Metric

Reward Metric. During training, each generated answer is scored with a sample-level reward Msamp. Three sources dominate current practice. (i) Human-preference rewards are learned from large RLHF corpora e.g., InstructGPT [3] and LLaVA-RLHF [201], and give dense feedback that closely matches user judgements. (ii) Verifiable rewards arise when a sub-task admits deterministic checks, such as unit-test pass rate in CodeRL [203] or symbolic exactness in DeepSeekMath [25]; they are noise-free but limited in scope. (iii) Modelpreference rewards replace humans with a stronger frozen critic, e.g., CriticGPT [204], delivering scalable but potentially biased supervision. The chosen reward is converted to advantages Aˆi,t and optimized via PPO, GRPO or DPO.

Sample-level Score ∑ 𝐴 , 

p p

𝑦 𝑦

𝑦 𝑦

Reward 𝑅∅(·)

𝜋 (·)

ℳ (·) ℳ (·) ℳ (·)

Reward 𝑅∅(·)

…

…

…

…

p

𝑦

𝑦

Reward 𝑅∅(·)

(b) Reward / Advantage Metric

State-level Score 𝐷(𝜋𝜃𝑡(⋅∣ 𝑝) ∥ 𝜋𝜃0(⋅∣ 𝑝))

… …

p p

p p

s ,  s , 

s ,  s , 

ℳ (·) ℳ (·) ℳ (·)

𝜋 (·)

𝜋 (·)

…

…

…

…

Model-State Metric. Beyond external scores, practitioners track light-weight diagnostics Mtstate throughout RL updates. Two lightweight diagnostics are widely adopted: (i) Output length drift, the deviation of answer length from the SFT baseline large drift foreshadows verbosity or repetition [2]; (ii) KL divergence between the current policy πθ

…

p

p

s , 

s , 

(c) Model-state Metric

- Fig. 6: Metric Granularity in Visual RL. (a) Set-level metric Mset: one score over the whole prompt set, used for final evaluation (e.g., FID). (b) Sample-level metric Msamp: peroutput rewards that train the policy (RLHF, DPO). (c)

t and frozen SFT reference πθ

0, as used in InstructGPT [3].

State-level metric Mtstate: training-time signals like KL or length drift, used to monitor stability. Notation: pi, ygeni , ygti denote the prompt, the generated output, and ground truth, respectively. πθ

#### 4.3 Evaluation of Visual Generation Models

Task Metric. As listed in Table 2, final quality is judged on standard, RL–free benchmarks that target complementary axes. Image fidelity & diversity is measured by FID and Inception Score, while pixel-level reconstruction tasks (superresolution, inpainting) use PSNR or SSIM. For prompt alignment, CLIP Score and Fr´echet CLIP Distance quantify semantic correctness; video models additionally report FVD or Video IS to capture temporal coherence.

t refer to the 0-th and t-th policy model. Rϕ(·) denotes the reward model.

0 and πθ

IoU [202]), Msamp compares the generated output ygeni to a ground-truth output ygti . Formally, the reward signal can be expressed as:

Reward Metric. During RL fine-tuning, each generated image or video receives a sample-level reward Msamp. Human-preference rewards, e.g., ImageReward [21] and HPS [24] supply dense signals that correlate well with aesthetic appeal. When a deterministic checker exists, authors turn to verifiable rewards: MotionPrompt [216] and DSPO [217] use optical-flow, object masks that can be evaluated without humans. A third route relies on model preference rewards, where a stronger frozen critic (e.g., VideoPrefer [147] or PARM [219]) scores samples, enabling scalable DPO/PPO training.

Msamp(ygeni ,pi) = Rϕ(ygeni ,pi) or Rϕ(ygeni ,ygti ,pi), (16) depending on whether the reward model Rϕ operates with or without access to ground-truth outputs. In practice, the per-sample scores are transformed into step-wise advantages Aˆi,t (where t indexes generation steps). These advantages directly drive policy updates, enabling reward shaping and exploration control at the level of individual outputs.

State-level Metric Mtstate. As depicted in Fig. 6 (c), state-level metrics monitor the training dynamics of the current policy πθ

t at iteration t. A common choice is the KL divergence to a frozen reference policy πθ

Model-State Metric. Two light diagnostics track training stability. (i) Denoising trajectory statistics: VARD [220] and Inversion DPO [221] record per-step noise predictions or DDIM traces; pathological spikes reveal early collapse. (ii) KL divergence between the current diffusion policy and its frozen base (πθ

0: Mtstate = Ep∼P

(· | p) . (17)

(· | p)∥πθ

D πθ

val

t

0

Other diagnostics include output-length drift for autoregressive models and DDIM step-trace variance for diffusion models. By tracking Mtstate during optimisation, practitioners detect reward hacking, mode collapse, or excessive policy shift before these issues degrade final performance.

0), popularized by DDPO [13] and reused in VARD.

t ∥πθ

#### 4.4 Evaluation of Unified Models

Task Metric. Two benchmark families are widely adopted. Generation-oriented suites such as GenEval [208], DPGBench [222], and ParaPrompts [223] focus on prompt-faithful generation, testing multi-object composition, style control,

#### 4.2 Evaluation of Multi-Modal Large Language Models

Task Metric. As summarized in Table 2, MLLM are first judged on external, RL-free benchmarks. General reasoning

and long-caption adherence. Conversely, understandingoriented benchmarks (MME [189], POPE [224]) measure grounding, reasoning and hallucination detection from the same backbone.

Reward Metric. Recent work explores two design philosophies for training signals. Unified rewards ( e.g., UniRL [15], CoRL [22]) blend multiple objectives textual correctness, CLIP alignment, aesthetic quality—into a single scalar that drives one shared policy across tasks. In contrast, taskspecific rewards keep the generator and understanding heads separate, applying RL only to the generation branch as in VARGPT-V1.1 [154] or Emu3 [16]. The former promotes crossmodal transfer, while the latter preserves the stability of perception modules.

Model-State Metric. Unified models additionally track fine-grained diagnostics during RL. UniRL [15] proposes a generation–understanding imbalance score the absolute gap between batch-level rewards on the two task families to prevent one modality from dominating the update. HermesFlow [151] monitors the KL divergence between the current shared policy πθ

t and its supervised baseline πθ

0

on both generation and understanding prompts, serving as an early-warning signal for policy collapse. These statelevel curves Mtstate allow practitioners to stop or re-weight training before external task scores degrade.

#### 4.5 Evaluation of Vision Language Action Models

Task Metric. In GUI Automation task, there are multiple benchmarks could be classified into online or offline scenarios. For offline setting, it mainly have grounding and navigation parts. For grounding, mainly check whether the click action fail into the target button; For navigation, it requires model to predict current action conditioned on oracle past history, this mainly dependent on whether the action class (click or type) are correctly predicted per step. For online setting, it is more challenging, because it requires the model to fully perform multi-step execution which is long procedural then check whether the final outcome meet the task requirement. Such long procedural setups will produce sparse signal in term of model evalation.

Reward Metric. For reward modeling, most offline RL methods borrow the metric from task metric like IoU, while come to the online environment, due to the sparsity of task success rate, which present significant challenges for end-toend multi-turn RL training [167] i.e., lack of training efficency, lack of informativeness, step-level reward is proposed to address this issue, such as developing a reward or critic models [232], [233].

Model-State Metric. To fully understand the model behavior beyond task success rate, trajectory length being an important metric as it can reflect how efficient model can address the task. A smart agent should be able to resolve the task with minimal steps. This pose challenges for agents with advanced planning ability.

#### 4.6 Benchmarks

A variety of new benchmarks explicitly support RL-based training and evaluation in the visual domain (see Table 3). For MLLM, recent datasets target complex multi-hop reasoning

and alignment with human preferences. For example, SEEDBench-R1 [237] introduces a hierarchical egocentric video question-answering benchmark with 50k training questions and a human-verified validation set. Long Video RL [242] scales up multi-step reasoning on long videos: it provides 52k QA pairs with detailed reasoning annotations. Another recent benchmark, Ego-R1 Bench [241], focuses on ultralong (week-long) egocentric videos; an RL-trained “chainof-tool-thought” agent must invoke perception tools in 7 sequential steps on average to answer each query, illustrating the use of step-wise reasoning accuracy as a core challenge. In the image domain, VisuLogic [239] contains 1,000 carefully crafted visual reasoning puzzles (e.g., spatial and logic problems) to evaluate pure vision-centric reasoning, and most models perform only slightly above random on this benchmark.

Benchmarks for visual generation tasks predominantly supply human preference data that serve as reward models for policy optimization. Datasets like ImageReward [21] and HPS v1 & v2 [24], [24] collect human-ranked pairs of text-toimage outputs, allowing one to train a scalar reward function that scores generations. Such reward models have been used to refine text-to-image diffusion models via RLHF, aligning outputs with human aesthetic preferences. Similarly, Picka-Pic [214] and VideoReward [141]extend this to broader user preferences (motion smoothness, text alignment). Some benchmarks also facilitate robust evaluation of generative RL agents on generalization. T2I-CompBench [209] is a textto-image compositionality test set that requires correctly binding novel combinations of attributes and object relations – a measure of compositional generalization often used to assess RL-trained generators. Likewise, domain-specific benchmarks define verifiable success criteria as rewards: StarVector [262] provides SVG code-generation tasks with a strict shape-matching reward, and AnimeReward [143] targets consistency in animated video generations with multi-dimensional human preference scores (image-video coherence, character consistency, etc.).

For vision–language action agents, numerous benchmarks provide expert trajectories and simulated environments with clear reward signals for policy training and robust evaluation. Many are centered on GUI and web interaction tasks, where success can be unambiguously measured. For instance, GUI-R1-3K [10] compiles 3,000+ GUI manipulation trajectories across Windows, Linux, macOS, Android, and web platforms. It introduces an “R1-style” dense reward scheme mapping each correct action (e.g., clicking the right button, entering correct text, a valid scroll) to positive feedback, providing step-wise reinforcement that guides an agent through multi-step UI tasks. Building on this, SE-GUI [11] curates 3k high-quality GUI examples with grounded instructions and bounding-box annotations, which are used to train agents with a self-imitation RL strategy. Evaluation-focused suites like UI-R1 [156] define a fixed set of unseen tasks (e.g., 136 mobile GUI tasks spanning click, scroll, swipe, text-input actions) to test generalization of learned policies. Meanwhile, web interaction benchmarks such as Mind2Web [263] offer 2,000 tasks on real websites with a binary success/failure reward for completing each task. Some datasets emphasize exact match and reproducibility: AITZ [264] (Android Interaction w/ CoAT

reasoning) logs 18k screen-action pairs with corresponding tool-assisted rationales, and uses an exact action match reward to ensure precise adherence to instructions. On the other hand, broader benchmarks like OmniAct [265] and GUICoURS [266] target generalist agent capabilities across diverse domains. OmniAct integrates nearly 10k scripted desktop and web tasks into a single environment, while GUICoURS combines multimodal resources (10M OCR observations, 67k navigation demonstrations) spanning GUI, web, and chat interfaces. The reward structures in these benchmarks are carefully crafted, from rule-based metrics to preference scores, to guide policy learning and reflect task goals. They enable visual RL agents to learn from meaningful feedback and be evaluated not just on task success, but also on alignment with human reasoning and performance on complex, long-horizon decisions.

5 CHALLENGES AND FUTURE WORKS

- 5.1 Effective Reasoning: Balancing Depth and Efficiency

A recurrent challenge in visual RL is reasoning calibration: excessively long chains of visual or verbal thoughts incur latency and compounding errors, whereas overly aggressive pruning discards salient cues. We foresee two research thrusts. (i) Adaptive horizon policies: train a termination critic that jointly optimizes answer quality and computational cost; curriculum-based reward shaping can gradually penalize redundant steps while preserving informative ones. (ii) Meta-reasoning and few-shot self-evaluation: incorporate a lightweight evaluator that critiques partial chains (e.g., via frozen vision–language models) and decides whether further thinking is worthwhile. Future benchmarks should therefore report both success rate and reasoning efficiency metrics (average steps, FLOPs, latency), encouraging algorithms that achieve high accuracy with just-enough deliberation rather than maximal cogitation.

#### 5.2 Long-Horizon RL in VLA

Long-horizon vision–language agents (VLA) must execute dozens of atomic actions (e.g., clicks, drags, text edits) before any end-task reward is observed. Existing works such as OS-World [273] and ARPO [167] therefore fall back on sparse reward for each click and a binary task success flag yet empirical results suggest that even GRPO yields limited gains under such supervision. Future research should (i) discover intrinsic sub-goals: segment trajectories via statechange detection or language-conditioned clustering, then assign dense rewards to sub-goal completions; (ii) learn affordance critics: train contrastive vision–language models to score how much an action reduces the distance to the verbal goal, providing shaped feedback without manual labels; (iii) hierarchical or option-based RL: couple a high-level language planner that proposes semantic sub-tasks with a low-level policy fine-tuned by off-policy RL or decision transformers;

#### 5.3 RL for Thinking with Vision

Recent works for visual planning, such as Chain-of-Focus [7] and Openthinkimg [91] all treat the picture as an external workspace: the agent may crop, sketch, highlight or insert

visual tokens before emitting the next language token. While early prototypes rely on supervised heuristics for these spatial actions, moving to reinforcement learning exposes four open problems. (i) Action-space design. Cropping or doodling is naturally continuous (x,y,w,h,...) yet RL libraries and GPU memories favor small discrete sets. Hybrid schemes that learn a differentiable proposal policy and then refine coordinates via policy gradient fine-tuning, as hinted by BRPO [84] and VRAG-RL [88], remain largely unexplored. (ii) Credit assignment. Most benchmarks only reward the final task success (e.g., answer correctness in VILASR [83]); the whole visual chain-of-thought shares a single sparse scalar. Future work should mine step-wise proxy rewards, e.g., CLIP similarity increase after a crop, or entropy drop in a learned belief state—to enable bootstrapped or hierarchical RL. (iii) Data efficiency. Sketching or inserting patches triggers extra forward passes through the vision encoder, making naive on-policy RL prohibitively expensive. Relabeling (DeepEyes [6]) and model-based imagination (Pixel Reasoner [86]) point to sample-efficient alternatives, but principled replay and uncertainty-aware planners for visual actions are still missing. Therefore, future directions include: learning structured visual skills (crop, zoom, draw) via skill-prior RL; devising cross-modal reward shaping that scores each edit by how much it simplifies the remaining reasoning; and curating benchmarks whose metrics expose not just final accuracy.

#### 5.4 Reward Model Design for Visual Generation

A central obstacle for reinforcement-learning–based visual generation is the lack of a scalable and faithful reward function. Widely used handcrafted metrics such as FID [205] offer a convenient numerical signal, yet correlate only weakly with human judgments of aesthetics, semantic fidelity, or temporal coherence, especially when the task extends beyond single-frame images. Recent learned critics, such as ImageReward [21] and HPS [24] for images, and VideoReward [141] for videos to bridge this gap by training on pairwise human-preference data, but each model targets a narrow modality and captures only a slice of perceptual quality (e.g., prompt alignment or visual appeal). As a result, policies optimized with PPO or GRPO often exploit loopholes in a single scalar signal, producing high-contrast artifacts, repetitive textures, or physically implausible motion that “game” the critic without improving real user satisfaction. The challenge, therefore, is to design reward models that (i) integrate complementary low-level signals (consistency, physics, geometry) with high-level human preferences, (ii) generalize across images, video and 3-D scenes, and (iii) remain robust against reward hacking while being cheap enough to update continually as user tastes evolve.

## 6 CONCLUSION

Visual reinforcement learning has transitioned from isolated proof-of-concepts to a vibrant research frontier that bridges vision, language, and action. Our review shows that modern progress is driven by three converging forces: (i) scalable reward supervision, moving from labour-intensive RLHF to group-relative and verifiable-signal pipelines; (ii) unified

architectures, where a single policy is jointly optimised for perception, reasoning, and generation; and (iii) ever-richer benchmarks, which measure not only task success but also alignment with human preference and policy stability.

Yet significant challenges remain. First, data and compute efficiency are pressing: current methods often require orders of magnitude more samples than supervised counterparts. Second, robust generalization across domains, viewpoints, and embodiment settings is still limited. Third, reward design for long-horizon, open-world tasks lacks principled guidance, risking reward hacking, and unsafe behaviors. Finally, evaluation standards must evolve to capture real-world utility, ethical alignment, and energy footprint. Addressing these issues will likely involve tighter integration of modelbased planning, self-supervised visual pre-training, adaptive curricula, and safety-aware optimization.

In summary, visual RL stands poised to transform how intelligent systems perceive and interact with their surroundings. By unifying methodological insights and charting unresolved questions, this survey aims to serve as both a reference and a catalyst for the next wave of research toward sample-efficient, reliable, and socially aligned visual decisionmaking agents.

## REFERENCES

- [1] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney et al., “Openai o1 system card,” arXiv preprint arXiv:2412.16720, 2024.
- [2] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in Neural Information Processing Systems, vol. 36, pp. 53728–53741, 2023.
- [3] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.
- [4] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi et al., “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” arXiv preprint arXiv:2501.12948, 2025.
- [5] G. Zhou, P. Qiu, C. Chen, J. Wang, Z. Yang, J. Xu, and M. Qiu, “Reinforced mllm: A survey on rl-based reasoning in multimodal large language models,” arXiv preprint arXiv:2504.21277, 2025.
- [6] Z. Zheng, M. Yang, J. Hong, C. Zhao, G. Xu, L. Yang, C. Shen, and X. Yu, “Deepeyes: Incentivizing” thinking with images” via reinforcement learning,” arXiv preprint arXiv:2505.14362, 2025.
- [7] X. Zhang, Z. Gao, B. Zhang, P. Li, X. Zhang, Y. Liu, T. Yuan, Y. Wu, Y. Jia, S.-C. Zhu et al., “Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl,” arXiv preprint arXiv:2505.15436, 2025.
- [8] G. Lu, W. Guo, C. Zhang, Y. Zhou, H. Jiang, Z. Gao, Y. Tang, and Z. Wang, “Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning,” arXiv preprint arXiv:2505.18719, 2025.
- [9] S. Tan, K. Dou, Y. Zhao, and P. Kr¨ahenbuhl,¨ “Interactive post-training for vision-language-action models,” arXiv preprint arXiv:2505.17016, 2025.
- [10] R. Luo, L. Wang, W. He, and X. Xia, “Gui-r1: A generalist r1style vision-language action model for gui agents,” arXiv preprint

- arXiv:2504.10458, 2025.

[11] X. Yuan, J. Zhang, K. Li, Z. Cai, L. Yao, J. Chen, E. Wang, Q. Hou, J. Chen, P.-T. Jiang et al., “Enhancing visual grounding for gui agents via self-evolutionary reinforcement learning,” arXiv preprint

- arXiv:2505.12370, 2025.

- [12] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, “Reinforcement learning for fine-tuning text-to-image diffusion models,” in Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS) 2023. Neural Information Processing Systems Foundation, 2023.

- [13] K. Black, M. Janner, Y. Du, I. Kostrikov, and S. Levine, “Training diffusion models with reinforcement learning,” 2023.
- [14] Z. Zhou, X. Xia, F. Ma, H. Fan, Y. Yang, and T.-S. Chua, “Dreamdpo: Aligning text-to-3d generation with human preferences via direct preference optimization,” arXiv preprint arXiv:2502.04370, 2025.
- [15] W. Mao, Z. Yang, and M. Z. Shou, “Unirl: Self-improving unified multimodal models via supervised and reinforcement learning,” arXiv preprint arXiv:2505.23380, 2025.
- [16] X. Wang, X. Zhang, Z. Luo, Q. Sun, Y. Cui, J. Wang, F. Zhang, Y. Wang, Z. Li, Q. Yu et al., “Emu3: Next-token prediction is all you need,” arXiv preprint arXiv:2409.18869, 2024.
- [17] B. Wang, Z. Yue, F. Zhang, S. Chen, L. Bi, J. Zhang, X. Song, K. Yanting Chan, J. Pan, W. Wu et al., “Discrete visual tokens of autoregression, by diffusion, and for reasoning,” arXiv e-prints, pp. arXiv–2505, 2025.
- [18] G. DeepMind, “Gemini 2.5,” https://deepmind.google/ technologies/gemini/, 2025, accessed: 2025-08-09.
- [19] Y. Shi, W. Yu, Z. Li, Y. Wang, H. Zhang, N. Liu, H. Mi, and D. Yu, “Mobilegui-rl: Advancing mobile gui agent through reinforcement learning in online environment,” arXiv preprint arXiv:2507.05720, 2025.
- [20] M. Kim, G. Bae, J. Lee, W. Shin, C. Kim, M.-Y. Choi, H. Shin, and H. Oh, “Rapid: Robust and agile planner using inverse reinforcement learning for vision-based drone navigation,” arXiv preprint arXiv:2502.02054, 2025.
- [21] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong, “Imagereward: Learning and evaluating human preferences for text-to-image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 15903–15935, 2023.
- [22] J. Jiang, C. Si, J. Luo, H. Zhang, and C. Ma, “Co-reinforcement learning for unified multimodal understanding and generation,” arXiv preprint arXiv:2505.17534, 2025.
- [23] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017.
- [24] X. Wu, K. Sun, F. Zhu, R. Zhao, and H. Li, “Human preference score: Better aligning text-to-image models with human preference,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 2096–2105.
- [25] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.
- [26] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, S. Lyu et al., “T\” ulu 3: Pushing frontiers in open language model post-training,” arXiv preprint arXiv:2411.15124, 2024.
- [27] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.
- [28] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel, “Highdimensional continuous control using generalized advantage estimation,” arXiv preprint arXiv:1506.02438, 2015.
- [29] Y. Oh, J. Mok, D. Chung, J. Shin, S. Park, J. Barthelemy, and S. Yoon, “Repic: Reinforced post-training for personalizing multi-modal language models,” arXiv preprint arXiv:2506.18369, 2025.
- [30] A. Zakharov and S. Whiteson, “Goalladder: Incremental goal discovery with vision-language models,” arXiv preprint arXiv:2506.16396, 2025.
- [31] Y. Li, M. Tian, D. Zhu, J. Zhu, Z. Lin, Z. Xiong, and X. Zhao, “Driver1: Bridging reasoning and planning in vlms for autonomous driving with reinforcement learning,” arXiv preprint arXiv:2506.18234, 2025.
- [32] H. Shen, P. Liu, J. Li, C. Fang, Y. Ma, J. Liao, Q. Shen, Z. Zhang, K. Zhao, Q. Zhang et al., “Vlm-r1: A stable and generalizable r1-style large vision-language model,” arXiv preprint arXiv:2504.07615, 2025.
- [33] Y. Chen, Y. Ge, R. Wang, Y. Ge, J. Cheng, Y. Shan, and X. Liu, “Grpocare: Consistency-aware reinforcement learning for multimodal reasoning,” arXiv preprint arXiv:2506.16141, 2025.
- [34] Z. Cai, J. Zhang, X. Yuan, P. Jiang, W. Chen, B. Tang, L. Yao, Q. Wang, J. Chen, and B. Li, “Q-ponder: A unified training pipeline for reasoning-based visual quality assessment,” arXiv preprint arXiv:2506.05384, 2025.
- [35] Y. Liang, J. Qiu, W. Ding, Z. Liu, J. Tompkin, M. Xu, M. Xia, Z. Tu, L. Shi, and J. Zhu, “Modomodo: Multi-domain data

- mixtures for multimodal llm reinforcement learning,” arXiv preprint arXiv:2505.24871, 2025.
- [36] Y. Ma, L. Du, X. Shen, S. Chen, P. Li, Q. Ren, L. Ma, Y. Dai, P. Liu, and J. Yan, “One rl to see them all: Visual triple unified reinforcement learning,” arXiv preprint arXiv:2505.18129, 2025.
- [37] Y. Zhan, Y. Zhu, S. Zheng, H. Zhao, F. Yang, M. Tang, and J. Wang, “Vision-r1: Evolving human-free alignment in large vision-language models via vision-guided reinforcement learning,” arXiv preprint arXiv:2503.18013, 2025.
- [38] Z. Xiao, J. Koo, S. Ouyang, J. Hernandez, Y. Meng, and V. Ordonez, “Proxythinker: Test-time guidance through small visual reasoners,” arXiv preprint arXiv:2505.24872, 2025.
- [39] Z. Wang, J. Zhu, B. Tang, Z. Li, F. Xiong, J. Yu, and M. B. Blaschko, “Jigsaw-r1: A study of rule-based visual reinforcement learning with jigsaw puzzles,” arXiv preprint arXiv:2505.23590, 2025.
- [40] Z. Wan, Z. Dou, C. Liu, Y. Zhang, D. Cui, Q. Zhao, H. Shen, J. Xiong, Y. Xin, Y. Jiang et al., “Srpo: Enhancing multimodal llm reasoning via reflection-aware reinforcement learning,” arXiv preprint arXiv:2506.01713, 2025.
- [41] Y. Yang, X. He, H. Pan, X. Jiang, Y. Deng, X. Yang, H. Lu, D. Yin, F. Rao, M. Zhu et al., “R1-onevision: Advancing generalized multimodal reasoning through cross-modal formalization,” arXiv preprint arXiv:2503.10615, 2025.
- [42] H. Tan, Y. Ji, X. Hao, M. Lin, P. Wang, Z. Wang, and S. Zhang, “Reason-rft: Reinforcement fine-tuning for visual reasoning,” arXiv preprint arXiv:2503.20752, 2025.
- [43] M. Zhu, Y. Tian, H. Chen, C. Zhou, Q. Guo, Y. Liu, M. Yang, and C. Shen, “Segagent: Exploring pixel understanding capabilities in mllms by imitating human annotator trajectories,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 3686–3696.
- [44] T. Wu, J. Zou, J. Liang, L. Zhang, and K. Ma, “Visualquality-r1: Reasoning-induced image quality assessment via reinforcement learning to rank,” arXiv preprint arXiv:2505.14460, 2025.
- [45] B. Zhang, H. Li, T. Zhang, C. Yan, J. Cai, X. Jiang, and Y. Hao, “Improving the reasoning of multi-image grounding in mllms via reinforcement learning,” arXiv preprint arXiv:2507.00748, 2025.
- [46] C. Shen, W. Wei, X. Qu, and Y. Cheng, “Satori-r1: Incentivizing multimodal reasoning with spatial grounding and verifiable rewards,” arXiv preprint arXiv:2505.19094, 2025.
- [47] P. Wang, Y. Wei, Y. Peng, X. Wang, W. Qiu, W. Shen, T. Xie, J. Pei, J. Zhang, Y. Hao et al., “Skywork r1v2: Multimodal hybrid reinforcement learning for reasoning,” arXiv preprint arXiv:2504.16656, 2025.
- [48] H. Yang, Y. Zhou, W. Han, and J. Shen, “Self-rewarding large vision-language models for optimizing prompts in text-to-image generation,” arXiv preprint arXiv:2505.16763, 2025.
- [49] Q. Chang, Z. Zhang, P. Hu, J. Du, J. Ma, Y. Pan, J. Zhang, Q. Liu, and J. Gao, “Thor: Tool-integrated hierarchical optimization via rl for mathematical reasoning,” arXiv preprint arXiv:2509.13761, 2025.
- [50] L. Xing, X. Dong, Y. Zang, Y. Cao, J. Liang, Q. Huang, J. Wang, F. Wu, and D. Lin, “Caprl: Stimulating dense image caption capabilities via reinforcement learning,” arXiv preprint arXiv:2509.22647, 2025.
- [51] L. Zhu, B. Ouyang, Y. Zhang, T. Cheng, R. Hu, H. Shen, L. Ran, X. Chen, L. Yu, W. Liu et al., “Lens: Learning to segment anything with unified reinforced reasoning,” arXiv preprint arXiv:2508.14153, 2025.
- [52] J. Xiong, Y. Wang, W. Zhao, C. Liu, B. Yin, W. Zhou, and H. Li, “Docr1: Evidence page-guided grpo for multi-page document understanding,” arXiv preprint arXiv:2508.07313, 2025.
- [53] A. Masry, A. Puri, M. Hashemi, J. A. Rodriguez, M. Thakkar, K. Mahajan, V. Yadav, S. T. Madhusudhan, A. Pich´e, D. Bahdanau et al., “Bigcharts-r1: Enhanced chart reasoning with visual reinforcement finetuning,” arXiv preprint arXiv:2508.09804, 2025.
- [54] H. Zhong, M. Zhu, Z. Du, Z. Huang, C. Zhao, M. Liu, W. Wang, H. Chen, and C. Shen, “Omni-r1: Reinforcement learning for omnimodal reasoning via two-system collaboration,” arXiv preprint arXiv:2505.20256, 2025.
- [55] S. Park, H. Kim, J. Kim, S. Kim, and Y. M. Ro, “Dip-r1: Deep inspection and perception with rl looking through and understanding complex scenes,” arXiv preprint arXiv:2505.23179, 2025.
- [56] Z. Pan and H. Liu, “Metaspatial: Reinforcing 3d spatial reasoning in vlms for the metaverse,” arXiv preprint arXiv:2503.18470, 2025.
- [57] A. Zholus, M. Kuznetsov, R. Schutski, R. Shayakhmetov, D. Polykovskiy, S. Chandar, and A. Zhavoronkov, “Bindgpt:

- A scalable framework for 3d molecular design via language modeling and reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 24, 2025, pp. 26083– 26091.
- [58] Z. Yuan, S. Jiang, C.-M. Feng, Y. Zhang, S. Cui, Z. Li, and N. Zhao, “Scene-r1: Video-grounded large language models for 3d scene reasoning without 3d annotations,” arXiv preprint arXiv:2506.17545, 2025.
- [59] E. Yu, K. Lin, L. Zhao, J. Yin, Y. Wei, Y. Peng, H. Wei, J. Sun, C. Han, Z. Ge et al., “Perception-r1: Pioneering perception policy with reinforcement learning,” arXiv preprint arXiv:2504.07954, 2025.
- [60] Z. Chen, X. Luo, and D. Li, “Visrl: Intention-driven visual perception via reinforced reasoning,” arXiv preprint arXiv:2503.07523, 2025.
- [61] X. Wang, Z. Yang, C. Feng, Y. Liang, Y. Zhou, X. Liu, Z. Zang, M. Li, C.-C. Lin, K. Lin et al., “Vicrit: A verifiable reinforcement learning proxy task for visual perception in vlms,” arXiv preprint arXiv:2506.10128, 2025.
- [62] Y. Liu, T. Qu, Z. Zhong, B. Peng, S. Liu, B. Yu, and J. Jia, “Visionreasoner: Unified visual perception and reasoning via reinforcement learning,” arXiv preprint arXiv:2505.12081, 2025.
- [63] G. Sarch, S. Saha, N. Khandelwal, A. Jain, M. J. Tarr, A. Kumar, and K. Fragkiadaki, “Grounded reinforcement learning for visual reasoning,” arXiv preprint arXiv:2505.23678, 2025.
- [64] Z. Liu, Z. Sun, Y. Zang, X. Dong, Y. Cao, H. Duan, D. Lin, and J. Wang, “Visual-rft: Visual reinforcement fine-tuning,” arXiv preprint arXiv:2503.01785, 2025.
- [65] P. Wang and H. Ling, “Svqa-r1: Reinforcing spatial reasoning in mllms via view-consistent reward optimization,” arXiv preprint arXiv:2506.01371, 2025.
- [66] J. Zhang, K. Miao, R. Pi, Z. Wang, R. Liu, R. Pan, and T. Zhang, “Vlgenrm: Enhancing vision-language verification via vision experts and iterative training,” arXiv preprint arXiv:2506.13888, 2025.
- [67] Y. Gou, K. Chen, Z. Liu, L. Hong, X. Jin, Z. Li, J. T. Kwok, and Y. Zhang, “Perceptual decoupling for scalable multi-modal reasoning via reward-optimized captioning,” arXiv preprint arXiv:2506.04559, 2025.
- [68] M. Unsal and A. Akkus, “Easyarc: Evaluating vision language models on true visual reasoning,” arXiv preprint arXiv:2506.11595, 2025.
- [69] Z. Li, Z. Ma, M. Li, S. Li, Y. Rong, T. Xu, Z. Zhang, D. Zhao, and W. Huang, “Star-r1: Spacial transformation reasoning by reinforcing multimodal llms,” arXiv preprint arXiv:2505.15804, 2025.
- [70] J. Xia, Y. Zang, P. Gao, Y. Li, and K. Zhou, “Visionary-r1: Mitigating shortcuts in visual reasoning with reinforcement learning,” arXiv preprint arXiv:2505.14677, 2025.
- [71] S. Bai, M. Li, Y. Liu, J. Tang, H. Zhang, L. Sun, X. Chu, and Y. Tang, “Univg-r1: Reasoning guided universal visual grounding with reinforcement learning,” arXiv preprint arXiv:2505.14231, 2025.
- [72] Z. Xing, X. Hu, C.-W. Fu, W. Wang, J. Dai, and P.-A. Heng, “Echoink-r1: Exploring audio-visual reasoning in multimodal llms via reinforcement learning,” arXiv preprint arXiv:2505.04623, 2025.
- [73] J. Yang, F. Ma, Z. Wang, D. Yin, K. Rong, F. Rao, and R. Zhang, “Wethink: Toward general-purpose vision-language reasoning via reinforcement learning,” arXiv preprint arXiv:2506.07905, 2025.
- [74] L. Chen, H. Gao, T. Liu, Z. Huang, F. Sung, X. Zhou, Y. Wu, and B. Chang, “G1: Bootstrapping perception and reasoning abilities of vision-language model via reinforcement learning,” arXiv preprint arXiv:2505.13426, 2025.
- [75] Y. Zhan, Z. Wu, Y. Zhu, R. Xue, R. Luo, Z. Chen, C. Zhang, Y. Li, Z. He, Z. Yang et al., “Gthinker: Towards general multimodal reasoning via cue-guided rethinking,” arXiv preprint arXiv:2506.01078, 2025.
- [76] Z. Guo, M. Hong, and T. Jin, “Observe-r1: Unlocking reasoning abilities of mllms with dynamic progressive reinforcement learning,” arXiv preprint arXiv:2505.12432, 2025.
- [77] F. Meng, L. Du, Z. Liu, Z. Zhou, Q. Lu, D. Fu, T. Han, B. Shi, W. Wang, J. He et al., “Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning,” arXiv preprint arXiv:2503.07365, 2025.
- [78] T. Xiao, X. Xu, Z. Huang, H. Gao, Q. Liu, Q. Liu, and E. Chen, “Advancing multimodal reasoning capabilities of multimodal large language models via visual perception reward,” arXiv preprint arXiv:2506.07218, 2025.
- [79] L.-C.-T. Xiaomi, “Mimo-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2506.03569

- [80] J. Li, S. Ju, Y. Su, H. Li, and Y. Shen, “Enhancing llms’ reasoningintensive multimedia search capabilities through fine-tuning and reinforcement learning,” arXiv preprint arXiv:2505.18831, 2025.
- [81] Y. Xu, C. Li, H. Zhou, X. Wan, C. Zhang, A. Korhonen, and I. Vuli´c, “Visual planning: Let’s think only with images,” arXiv preprint arXiv:2505.11409, 2025.
- [82] Y. Fan, X. He, D. Yang, K. Zheng, C.-C. Kuo, Y. Zheng, S. J. Narayanaraju, X. Guan, and X. E. Wang, “Grit: Teaching mllms to think with images,” arXiv preprint arXiv:2505.15879, 2025.
- [83] J. Wu, J. Guan, K. Feng, Q. Liu, S. Wu, L. Wang, W. Wu, and T. Tan, “Reinforcing spatial reasoning in vision-language models with interwoven thinking and visual drawing,” arXiv preprint arXiv:2506.09965, 2025.
- [84] X. Chu, X. Chen, G. Wang, Z. Tan, K. Huang, W. Lv, T. Mo, and W. Li, “Qwen look again: Guiding vision-language reasoning models to re-attention visual information,” arXiv preprint arXiv:2505.23558, 2025.
- [85] M. Cao, H. Zhao, C. Zhang, X. Chang, I. Reid, and X. Liang, “Ground-r1: Incentivizing grounded visual reasoning via reinforcement learning,” arXiv preprint arXiv:2505.20272, 2025.
- [86] A. Su, H. Wang, W. Ren, F. Lin, and W. Chen, “Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning,” arXiv preprint arXiv:2505.15966, 2025.
- [87] Z. Kan, Y. Liu, K. Yin, X. Jiang, X. Li, H. Cao, Y. Liu, D. Jiang, X. Sun, Q. Liao et al., “Taco: Think-answer consistency for optimized longchain reasoning and efficient data learning via reinforcement learning in lvlms,” arXiv preprint arXiv:2505.20777, 2025.
- [88] Q. Wang, R. Ding, Y. Zeng, Z. Chen, L. Chen, S. Wang, P. Xie, F. Huang, and F. Zhao, “Vrag-rl: Empower vision-perceptionbased rag for visually rich information understanding via iterative reasoning with reinforcement learning,” arXiv preprint arXiv:2505.22019, 2025.
- [89] Z. Huang, Y. Ji, A. S. Rajan, Z. Cai, W. Xiao, J. Hu, and Y. J. Lee, “Visualtoolagent (vista): A reinforcement learning framework for visual tool selection,” arXiv preprint arXiv:2505.20289, 2025.
- [90] E. Chern, Z. Hu, S. Chern, S. Kou, J. Su, Y. Ma, Z. Deng, and P. Liu, “Thinking with generated images,” arXiv preprint arXiv:2505.22525, 2025.
- [91] Z. Su, L. Li, M. Song, Y. Hao, Z. Yang, J. Zhang, G. Chen, J. Gu, J. Li, X. Qu et al., “Openthinkimg: Learning to think with images via visual tool reinforcement learning,” arXiv preprint arXiv:2505.08617, 2025.
- [92] M. Zhu, H. Zhong, C. Zhao, Z. Du, Z. Huang, M. Liu, H. Chen, C. Zou, J. Chen, M. Yang et al., “Active-o3: Empowering multimodal large language models with active perception via grpo,” arXiv preprint arXiv:2505.21457, 2025.
- [93] Y. Chen, Y. Shen, W. Huang, S. Zhou, Q. Lin, X. Cai, Z. Yu, J. Bu, B. Shi, and Y. Qiao, “Learning only with images: Visual reinforcement learning with reasoning, rendering, and visual feedback,” arXiv preprint arXiv:2507.20766, 2025.
- [94] S. Yang, J. Li, X. Lai, B. Yu, H. Zhao, and J. Jia, “Visionthink: Smart and efficient vision language model via reinforcement learning,” arXiv preprint arXiv:2507.13348, 2025.
- [95] X. Zhang, W. Li, S. Zhao, J. Li, L. Zhang, and J. Zhang, “Vq-insight: Teaching vlms for ai-generated video quality understanding via progressive visual reinforcement learning,” arXiv preprint arXiv:2506.18564, 2025.
- [96] J. Dang, J. Wu, T. Wang, X. Lin, N. Zhu, H. Chen, W.-S. Zheng, M. Wang, and T.-S. Chua, “Reinforcing video reasoning with focused thinking,” arXiv preprint arXiv:2505.24718, 2025.
- [97] A. Vinod, S. Pandit, A. Vavre, and L. Liu, “Egovlm: Policy optimization for egocentric video understanding,” arXiv preprint arXiv:2506.03097, 2025.
- [98] L. Zhu, Q. Chen, X. Shen, and X. Cun, “Vau-r1: Advancing video anomaly understanding via reinforcement fine-tuning,” arXiv preprint arXiv:2505.23504, 2025.
- [99] J. Park, J. Na, J. Kim, and H. J. Kim, “Deepvideo-r1: Video reinforcement fine-tuning via difficulty-aware regressive grpo,” arXiv preprint arXiv:2506.07464, 2025.
- [100] J. Zhang, L. Feng, X. Guo, Y. Wu, Y. Dong, and D. Xu, “Timemaster: Training time-series multimodal llms to reason via reinforcement learning,” arXiv preprint arXiv:2506.13705, 2025.
- [101] K. Feng, K. Gong, B. Li, Z. Guo, Y. Wang, T. Peng, B. Wang, and X. Yue, “Video-r1: Reinforcing video reasoning in mllms,” arXiv: 2503.21776, 2025.
- [102] X. Li, Z. Yan, D. Meng, L. Dong, X. Zeng, Y. He, Y. Wang, Y. Qiao, Y. Wang, and L. Wang, “Videochat-r1: Enhancing spatio-

- temporal perception via reinforcement fine-tuning,” arXiv preprint arXiv:2504.06958, 2025.
- [103] H. Li, S. Han, Y. Liao, J. Luo, J. Gao, S. Yan, and S. Liu, “Reinforcement learning tuning for videollms: Reward design and data efficiency,” arXiv preprint arXiv:2506.01908, 2025.
- [104] S. Tao, J. Li, Y. Yan, J. Zhang, Y. Gao, H. Li, S. Xun, Y. Fan, H. Chen, J. He et al., “Moss-chatv: Reinforcement learning with process reasoning reward for video temporal reasoning,” arXiv preprint arXiv:2509.21113, 2025.
- [105] Y. Li, J. Cheng, S. Jia, H. Kuang, S. Jiao, Q. Hou, and M.-M. Cheng, “Tempsamp-r1: Effective temporal sampling with reinforcement fine-tuning for video llms,” arXiv preprint arXiv:2509.18056, 2025.
- [106] Y. Xie, T. Chen, Z. Ge, and L. Ni, “Video-mtr: Reinforced multiturn reasoning for long video understanding,” arXiv preprint arXiv:2508.20478, 2025.
- [107] H. Zhang, X. Gu, J. Li, C. Ma, S. Bai, C. Zhang, B. Zhang, Z. Zhou, D. He, and Y. Tang, “Thinking with videos: Multimodal toolaugmented reinforcement learning for long video reasoning,” arXiv preprint arXiv:2508.04416, 2025.
- [108] N. Dufour, L. Degeorge, A. Ghosh, V. Kalogeiton, and D. Picard, “Miro: Multi-reward conditioned pretraining improves t2i quality and efficiency,” arXiv preprint arXiv:2510.25897, 2025.
- [109] Y. Luo, P. Du, B. Li, S. Du, T. Zhang, Y. Chang, K. Wu, K. Gai, and X. Wang, “Sample by step, optimize by chunk: Chunk-level grpo for text-to-image generation,” arXiv preprint arXiv:2510.21583, 2025.
- [110] Y. Gong, X. Wang, J. Wu, S. Wang, Y. Wang, and X. Wu, “Onereward: Unified mask-guided image generation via multitask human preference learning,” arXiv preprint arXiv:2508.21066, 2025.
- [111] S. Wu, M. Huang, Y. Cheng, W. Wu, J. Tian, Y. Luo, F. Ding, and Q. He, “Uso: Unified style and subject-driven generation via disentangled and reward learning,” arXiv preprint arXiv:2508.18966, 2025.
- [112] Y. Wang, Z. Li, Y. Zang, Y. Zhou, J. Bu, C. Wang, Q. Lu, C. Jin, and J. Wang, “Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning,” arXiv preprint arXiv:2508.20751, 2025.
- [113] X. He, S. Fu, Y. Zhao, W. Li, J. Yang, D. Yin, F. Rao, and B. Zhang, “Tempflow-grpo: When timing matters for grpo in flow models,” arXiv preprint arXiv:2508.04324, 2025.
- [114] Y. Zhang, Y. Li, Y. Yang, R. Wang, Y. Yang, D. Qi, J. Bao, D. Chen, C. Luo, and L. Qiu, “Reasongen-r1: Cot for autoregressive image generation models through sft and rl,” arXiv preprint arXiv:2505.24875, 2025.
- [115] K. Pan, W. Bu, Y. Wu, Y. Wu, K. Shen, Y. Li, H. Zhao, J. Li, S. Tang, and Y. Zhuang, “Focusdiff: Advancing fine-grained text-image alignment for autoregressive visual generation through rl,” arXiv preprint arXiv:2506.05501, 2025.
- [116] Y. Fan, O. Watkins, Y. Du, H. Liu, M. Ryu, C. Boutilier, P. Abbeel, M. Ghavamzadeh, K. Lee, and K. Lee, “Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models,” Advances in Neural Information Processing Systems, vol. 36, pp. 79858–79885, 2023.
- [117] S. Gupta, C. Ahuja, T.-Y. Lin, S. D. Roy, H. Oosterhuis, M. de Rijke, and S. N. Shukla, “A simple and effective reinforcement learning method for text-to-image diffusion fine-tuning,” arXiv preprint arXiv:2503.00897, 2025.
- [118] K. Liu, Z. Zhu, C. Li, H. Liu, H. Zeng, and J. Hou, “Prefpaint: Aligning image inpainting diffusion model with human preference,” Advances in Neural Information Processing Systems, vol. 37, pp. 30554–30589, 2024.
- [119] J. A. Rodriguez, H. Zhang, A. Puri, A. Feizi, R. Pramanik, P. Wichmann, A. Mondal, M. R. Samsami, R. Awal, P. Taslakian et al., “Rendering-aware reinforcement learning for vector graphics generation,” arXiv preprint arXiv:2505.20793, 2025.
- [120] C. Duan, R. Fang, Y. Wang, K. Wang, L. Huang, X. Zeng, H. Li, and X. Liu, “Got-r1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning,” arXiv preprint arXiv:2505.17022, 2025.
- [121] Z. Hu, F. Zhang, and K. Kuang, “D-fusion: Direct preference optimization for aligning diffusion models with visually consistent samples,” arXiv preprint arXiv:2505.22002, 2025.
- [122] B. Wallace, M. Dang, R. Rafailov, L. Zhou, A. Lou, S. Purushwalkam, S. Ermon, C. Xiong, S. Joty, and N. Naik, “Diffusion model alignment using direct preference optimization,” in Pro-

- ceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 8228–8238.
- [123] S. Li, K. Kallidromitis, A. Gokul, Y. Kato, and K. Kozuka, “Aligning diffusion models by optimizing human utility,” Advances in Neural Information Processing Systems, vol. 37, pp. 24897–24925, 2024.
- [124] Y. Gu, Z. Wang, Y. Yin, Y. Xie, and M. Zhou, “Diffusion-rpo: Aligning diffusion models through relative preference optimization,” arXiv preprint arXiv:2406.06382, 2024.
- [125] Z. Miao, J. Wang, Z. Wang, Z. Yang, L. Wang, Q. Qiu, and Z. Liu, “Training diffusion models towards diverse image generation with reinforcement learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 10844–10853.
- [126] S. H. Lee, Y. Li, J. Ke, I. Yoo, H. Zhang, J. Yu, Q. Wang, F. Deng, G. Entis, J. He et al., “Parrot: Pareto-optimal multi-reward reinforcement learning framework for text-to-image generation,” in European Conference on Computer Vision. Springer, 2024, pp. 462–478.
- [127] F. Wei, W. Zeng, Z. Li, D. Yin, L. Duan, and W. Li, “Powerful and flexible: Personalized text-to-image generation via reinforcement learning,” in European Conference on Computer Vision. Springer, 2024, pp. 394–410.
- [128] Y. Miao, W. Loh, S. Kothawade, P. Poupart, A. Rashwan, and Y. Li, “Subject-driven text-to-image generation via preferencebased reinforcement learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 123563–123591, 2024.
- [129] M. Wu, L. Wang, P. Zhao, F. Yang, J. Zhang, J. Liu, Y. Zhan, W. Han, H. Sun, J. Ji et al., “Reprompt: Reasoning-augmented reprompting for text-to-image generation via reinforcement learning,” arXiv preprint arXiv:2505.17540, 2025.
- [130] O. Oertell, J. D. Chang, Y. Zhang, K. Brantley, and W. Sun, “Rl for consistency models: Reward guided text-to-image generation with fast inference,” in Reinforcement Learning Conference, 2024.
- [131] Z. Hu, F. Zhang, L. Chen, K. Kuang, J. Li, K. Gao, J. Xiao, X. Wang, and W. Zhu, “Towards better alignment: Training diffusion models with reinforcement learning against sparse rewards,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 23604–23614.
- [132] Z. Xiao, P. Jiang, M. Zhou, J. Zhang, Z. Huang, and J. Yang, “Diffppo: Reinforcement learning fine-tuning of diffusion models for text-to-image generation,” in 2024 International Conference on Neuromorphic Computing (ICNC). IEEE, 2024, pp. 1–5.
- [133] J. Wang, Z. Tian, X. Wang, X. Zhang, W. Huang, Z. Wu, and Y.-G. Jiang, “Simplear: Pushing the frontier of autoregressive visual generation through pretraining, sft, and rl,” arXiv preprint arXiv:2504.11455, 2025.
- [134] S. Zhou, R. Zhang, H. Zhu, B. Kveton, Y. Zhou, J. Gu, J. Chen, and C. Chen, “Multimodal llms as customized reward models for text-to-image generation,” 2025. [Online]. Available: https://arxiv.org/abs/2507.21391
- [135] J. Liu, G. Liu, J. Liang, Y. Li, J. Liu, X. Wang, P. Wan, D. Zhang, and W. Ouyang, “Flow-grpo: Training flow matching models via online rl,” arXiv preprint arXiv:2505.05470, 2025.
- [136] M. Gallici and H. S. d. O. Borde, “Fine-tuning next-scale visual autoregressive models with group relative policy optimization,” arXiv preprint arXiv:2505.23331, 2025.
- [137] C. Chen, A. Wang, H. Wu, L. Liao, W. Sun, Q. Yan, and W. Lin, “Enhancing diffusion models with text-encoder reinforcement learning,” in European Conference on Computer Vision. Springer, 2024, pp. 182–198.
- [138] Z. Xue, J. Wu, Y. Gao, F. Kong, L. Zhu, M. Chen, Z. Liu, W. Liu, Q. Guo, W. Huang et al., “Dancegrpo: Unleashing grpo on visual generation,” arXiv preprint arXiv:2505.07818, 2025.
- [139] X. Fang, L. Ma, Z. Chen, M. Zhou, and G.-j. Qi, “Inflvg: Reinforce inference-time consistent long video generation with grpo,” arXiv preprint arXiv:2505.17574, 2025.
- [140] W. Lin, L. Jia, W. Hu, K. Pan, Z. Yue, W. Zhao, J. Chen, F. Wu, and H. Zhang, “Reasoning physical video generation with diffusion timestep tokens via reinforcement learning,” arXiv preprint arXiv:2504.15932, 2025.
- [141] J. Liu, G. Liu, J. Liang, Z. Yuan, X. Liu, M. Zheng, X. Wu, Q. Wang, W. Qin, M. Xia et al., “Improving video generation with human feedback,” arXiv preprint arXiv:2501.13918, 2025.
- [142] Y. Chen, H. Li, Z. Jiang, H. Wen, and D. Zhao, “Tevir: Text-to-video reward with diffusion models for efficient reinforcement learning,” arXiv preprint arXiv:2505.19769, 2025.

- [143] B. Zhu, Y. Jiang, B. Xu, S. Yang, M. Yin, Y. Wu, H. Sun, and Z. Wu, “Aligning anime video generation with human feedback,” arXiv preprint arXiv:2504.10044, 2025.
- [144] H. Yuan, S. Zhang, X. Wang, Y. Wei, T. Feng, Y. Pan, Y. Zhang, Z. Liu, S. Albanie, and D. Ni, “Instructvideo: Instructing video diffusion models with human feedback,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 6463–6474.
- [145] X. He, D. Jiang, G. Zhang, M. Ku, A. Soni, S. Siu, H. Chen, A. Chandra, Z. Jiang, A. Arulraj et al., “Videoscore: Building automatic metrics to simulate fine-grained human feedback for video generation,” arXiv preprint arXiv:2406.15252, 2024.
- [146] Z. Mou, B. Xia, Z. Huang, W. Yang, and J. Jia, “Gradeo: Towards human-like evaluation for text-to-video generation via multi-step reasoning,” arXiv preprint arXiv:2503.02341, 2025.
- [147] X. Wu, S. Huang, G. Wang, J. Xiong, and F. Wei, “Boosting text-tovideo generative model with mllms feedback,” Advances in Neural Information Processing Systems, vol. 37, pp. 139444–139469, 2024.
- [148] X. Zou, R. Xia, H. Wang, and P. Zhou, “Dreamcs: Geometry-aware text-to-3d generation with unpaired 3d reward supervision,” arXiv preprint arXiv:2506.09814, 2025.
- [149] J. Ye, F. Liu, Q. Li, Z. Wang, Y. Wang, X. Wang, Y. Duan, and J. Zhu, “Dreamreward: Text-to-3d generation with human preference,” in European Conference on Computer Vision. Springer, 2024, pp. 259– 276.
- [150] Q. Liu, Z. Liu, D. Zhang, and K. Jia, “Nabla-r2d3: Effective and efficient 3d diffusion alignment with 2d rewards,” arXiv preprint arXiv:2506.15684, 2025.
- [151] L. Yang, X. Zhang, Y. Tian, C. Shang, M. Xu, W. Zhang, and B. Cui, “Hermesflow: Seamlessly closing the gap in multimodal understanding and generation,” arXiv preprint arXiv:2502.12148, 2025.
- [152] Y. Wang, Y. Zang, H. Li, C. Jin, and J. Wang, “Unified reward model for multimodal understanding and generation,” arXiv preprint arXiv:2503.05236, 2025.
- [153] Y. Wang, Z. Li, Y. Zang, C. Wang, Q. Lu, C. Jin, and J. Wang, “Unified multimodal chain-of-thought reward model through reinforcement fine-tuning,” arXiv preprint arXiv:2505.03318, 2025.
- [154] X. Zhuang, Y. Xie, Y. Deng, D. Yang, L. Liang, J. Ru, Y. Yin, and Y. Zou, “Vargpt-v1. 1: Improve visual autoregressive large unified model via iterative instruction tuning and reinforcement learning,” arXiv preprint arXiv:2504.02949, 2025.
- [155] Z. Geng, Y. Wang, Y. Ma, C. Li, Y. Rao, S. Gu, Z. Zhong, Q. Lu, H. Hu, X. Zhang et al., “X-omni: Reinforcement learning makes discrete autoregressive image generative models great again,” arXiv preprint arXiv:2507.22058, 2025.
- [156] Z. Lu, Y. Chai, Y. Guo, X. Yin, L. Liu, H. Wang, H. Xiao, S. Ren, G. Xiong, and H. Li, “Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning,” arXiv preprint arXiv:2503.21620, 2025.
- [157] L. Gao, L. Zhang, and M. Xu, “Uishift: Enhancing vlm-based gui agents through self-supervised reinforcement learning,” arXiv preprint arXiv:2505.12493, 2025.
- [158] Z. Zhang, Y. Lu, Y. Fu, Y. Huo, S. Yang, Y. Wu, H. Si, X. Cong, H. Chen, Y. Lin et al., “Agentcpm-gui: Building mobile-use agents with reinforcement fine-tuning,” arXiv preprint arXiv:2506.01391, 2025.
- [159] D. Zhang, S. Zhang, Z. Yang, Z. Zhu, Z. Zhao, R. Cao, L. Chen, and K. Yu, “Progrm: Build better gui agents with progress rewards,” arXiv preprint arXiv:2505.18121, 2025.
- [160] J. Gu, Q. Ai, Y. Wang, P. Bu, J. Xing, Z. Zhu, W. Jiang, Z. Wang, Y. Zhao, M.-L. Zhang et al., “Mobile-r1: Towards interactive reinforcement learning for vlm-based mobile agent via task-level rewards,” arXiv preprint arXiv:2506.20332, 2025.
- [161] Y. Yang, D. Li, Y. Dai, Y. Yang, Z. Luo, Z. Zhao, Z. Hu, J. Huang, A. Saha, Z. Chen et al., “Gta1: Gui test-time scaling agent,” arXiv preprint arXiv:2507.05791, 2025.
- [162] J. Tang, Y. Xia, Y.-F. Wu, Y. Hu, Y. Chen, Q.-G. Chen, X. Xu, X. Wu, H. Lu, Y. Ma et al., “Lpo: Towards accurate gui agent interaction via location preference optimization,” arXiv preprint arXiv:2506.09373, 2025.
- [163] Y. Qin, Y. Ye, J. Fang, H. Wang, S. Liang, S. Tian, J. Zhang, J. Li, Y. Li, S. Huang et al., “Ui-tars: Pioneering automated gui interaction with native agents,” arXiv preprint arXiv:2501.12326, 2025.
- [164] Z. Zhang, W. Xie, X. Zhang, and Y. Lu, “Reinforced ui instruction grounding: Towards a generic ui task automation api,” arXiv preprint arXiv:2310.04716, 2023.

- [165] G. Papoudakis, T. Coste, Z. Wu, J. Hao, J. Wang, and K. Shao, “Appvlm: A lightweight vision language model for online app control,” arXiv preprint arXiv:2502.06395, 2025.
- [166] H. Bai, Y. Zhou, J. Pan, M. Cemri, A. Suhr, S. Levine, and A. Kumar, “Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 12461–12495, 2024.
- [167] F. Lu, Z. Zhong, S. Liu, C.-W. Fu, and J. Jia, “Arpo: End-to-end policy optimization for gui agents with experience replay,” arXiv

- preprint arXiv:2505.16282, 2025.

[168] C. Gao, L. Jin, X. Peng, J. Zhang, Y. Deng, A. Li, H. Wang, and S. Liu, “Octonav: Towards generalist embodied navigation,” arXiv

- preprint arXiv:2506.09839, 2025.

- [169] H. Zhao, W. Song, D. Wang, X. Tong, P. Ding, X. Cheng, and Z. Ge, “More: Unlocking scalability in reinforcement learning for quadruped vision-language-action models,” arXiv preprint arXiv:2503.08007, 2025.
- [170] Z. Qi, Z. Zhang, Y. Yu, J. Wang, and H. Zhao, “Vln-r1: Visionlanguage navigation via reinforcement fine-tuning,” arXiv preprint arXiv:2506.17221, 2025.
- [171] J. Hu, R. Hendrix, A. Farhadi, A. Kembhavi, R. Mart´ın-Mart´ın, P. Stone, K.-H. Zeng, and K. Ehsani, “Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning fine-tuning,” arXiv preprint arXiv:2409.16578, 2024.
- [172] A. Jiang, Y. Gao, Y. Wang, Z. Sun, S. Wang, Y. Heng, H. Sun, S. Tang, L. Zhu, J. Chai et al., “Irl-vla: Training an vision-language-action policy via reward world model,” arXiv preprint arXiv:2508.06571, 2025.
- [173] B. Zhao, Z. Wang, J. Fang, C. Gao, F. Man, J. Cui, X. Wang, X. Chen, Y. Li, and W. Zhu, “Embodied-r: Collaborative framework for activating embodied spatial reasoning in foundation models via reinforcement learning,” arXiv preprint arXiv:2504.12680, 2025.
- [174] A. Eftekhar, K.-H. Zeng, J. Duan, A. Farhadi, A. Kembhavi, and R. Krishna, “Selective visual representations improve convergence and generalization for embodied ai,” arXiv preprint arXiv:2311.04193, 2023.
- [175] Z. Chen, R. Niu, H. Kong, and Q. Wang, “Tgrpo: Fine-tuning vision-language-action model via trajectory-wise group relative policy optimization,” arXiv preprint arXiv:2506.08440, 2025.
- [176] J. Shu, Z. Lin, and Y. Wang, “Rftf: Reinforcement fine-tuning for embodied agents with temporal feedback,” arXiv preprint arXiv:2505.19767, 2025.
- [177] J. Liu, F. Gao, B. Wei, X. Chen, Q. Liao, Y. Wu, C. Yu, and Y. Wang, “What can rl bring to vla generalization? an empirical study,” arXiv preprint arXiv:2505.19789, 2025.
- [178] Y. Chen, S. Tian, S. Liu, Y. Zhou, H. Li, and D. Zhao, “Conrft: A reinforced fine-tuning method for vla models via consistency policy,” arXiv preprint arXiv:2502.05450, 2025.
- [179] Y. Guo, J. Zhang, X. Chen, X. Ji, Y.-J. Wang, Y. Hu, and J. Chen, “Improving vision-language-action model with online reinforcement learning,” arXiv preprint arXiv:2501.16664, 2025.
- [180] H. Zhang, Z. Zhuang, H. Zhao, P. Ding, H. Lu, and D. Wang, “Reinbot: Amplifying robot visual-language manipulation with reinforcement learning,” arXiv preprint arXiv:2505.07395, 2025.
- [181] L. Kang, X. Song, H. Zhou, Y. Qin, J. Yang, X. Liu, P. Torr, L. Bai, and Z. Yin, “Viki-r: Coordinating embodied multi-agent cooperation via reinforcement learning,” arXiv preprint arXiv:2506.09049, 2025.
- [182] D. Kim, S. Park, H. Jang, J. Shin, J. Kim, and Y. Seo, “Robotr1: Reinforcement learning for enhanced embodied reasoning in robotics,” arXiv preprint arXiv:2506.00070, 2025.
- [183] C. Xu, Q. Li, J. Luo, and S. Levine, “Rldg: Robotic generalist policy distillation via reinforcement learning,” arXiv preprint arXiv:2412.09858, 2024.
- [184] J. Luo, C. Xu, J. Wu, and S. Levine, “Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning,” arXiv preprint arXiv:2410.21845, 2024.
- [185] N. Hansen, Y. Lin, H. Su, X. Wang, V. Kumar, and A. Rajeswaran, “Modem: Accelerating visual model-based reinforcement learning with demonstrations,” arXiv preprint arXiv:2212.05698, 2022.
- [186] A. Zeng, S. Song, S. Welker, J. Lee, A. Rodriguez, and T. Funkhouser, “Learning synergies between pushing and grasping with self-supervised deep reinforcement learning,” in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2018, pp. 4238–4245.

- [187] Z. Guo, R. Zhang, C. Tong, Z. Zhao, P. Gao, H. Li, and P.-A. Heng, “Can we generate images with cot? let’s verify and reinforce image generation step by step,” arXiv preprint arXiv:2501.13926, 2025.
- [188] J. Wang, K. Q. Lin, J. Cheng, and M. Z. Shou, “Think or not? selective reasoning via reinforcement learning for vision-language models,” arXiv preprint arXiv:2505.16854, 2025.
- [189] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, J. Yang, X. Zheng, K. Li, X. Sun et al., “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2023.
- [190] B. Li, Y. Ge, Y. Ge, G. Wang, R. Wang, R. Zhang, and Y. Shan, “Seed-bench: Benchmarking multimodal large language models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 13299–13308.
- [191] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh, “Making the v in vqa matter: Elevating the role of image understanding in visual question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 6904–6913.
- [192] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu et al., “Mmbench: Is your multi-modal model an all-around player?” in European conference on computer vision. Springer, 2024, pp. 216–233.
- [193] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.
- [194] A. Mishra, S. Shekhar, A. K. Singh, and A. Chakraborty, “Ocr-vqa: Visual question answering by reading text in images,” in 2019 international conference on document analysis and recognition (ICDAR). IEEE, 2019, pp. 947–952.
- [195] Y. Liu, Z. Li, H. Li, W. Yu, M. Huang, D. Peng, M. Liu, M. Chen, C. Li, L. Jin et al., “On the hidden mystery of ocr in large multimodal models,” arXiv preprint arXiv:2305.07895, vol. 2, no. 5, p. 6, 2023.
- [196] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” arXiv preprint arXiv:2310.02255, 2023.
- [197] M. Mathew, D. Karatzas, and C. Jawahar, “Docvqa: A dataset for vqa on document images,” in Proceedings of the IEEE/CVF winter conference on applications of computer vision, 2021, pp. 2200–2209.
- [198] M. Mathew, V. Bagal, R. Tito, D. Karatzas, E. Valveny, and C. Jawahar, “Infographicvqa,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2022, pp. 1697–1706.
- [199] G. Zhang, X. Du, B. Chen, Y. Liang, T. Luo, T. Zheng, K. Zhu, Y. Cheng, C. Xu, S. Guo et al., “Cmmmu: A chinese massive multidiscipline multimodal understanding benchmark,” arXiv preprint arXiv:2401.11944, 2024.
- [200] Z. He, X. Wu, P. Zhou, R. Xuan, G. Liu, X. Yang, Q. Zhu, and H. Huang, “Cmmu: A benchmark for chinese multi-modal multi-type question understanding and reasoning,” arXiv preprint arXiv:2401.14011, 2024.
- [201] Z. Sun, S. Shen, S. Cao, H. Liu, C. Li, Y. Shen, C. Gan, L.-Y. Gui, Y.-X. Wang, Y. Yang et al., “Aligning large multimodal models with factually augmented rlhf,” arXiv preprint arXiv:2309.14525, 2023.
- [202] H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, and S. Savarese, “Generalized intersection over union: A metric and a loss for bounding box regression,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 658– 666.
- [203] H. Le, Y. Wang, A. D. Gotmare, S. Savarese, and S. C. H. Hoi, “Coderl: Mastering code generation through pretrained models and deep reinforcement learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 21314–21328, 2022.
- [204] N. McAleese, R. M. Pokorny, J. F. C. Uribe, E. Nitishinskaya, M. Trebacz, and J. Leike, “Llm critics help catch llm bugs,” arXiv preprint arXiv:2407.00215, 2024.
- [205] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “Gans trained by a two time-scale update rule converge to a local nash equilibrium,” Advances in neural information processing systems, vol. 30, 2017.
- [206] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, “Improved techniques for training gans,” Advances in neural information processing systems, vol. 29, 2016.

- [207] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: from error visibility to structural similarity,” IEEE transactions on image processing, vol. 13, no. 4, pp. 600–612, 2004.
- [208] D. Ghosh, H. Hajishirzi, and L. Schmidt, “Geneval: An objectfocused framework for evaluating text-to-image alignment,” Advances in Neural Information Processing Systems, vol. 36, pp. 52132– 52152, 2023.
- [209] K. Huang, K. Sun, E. Xie, Z. Li, and X. Liu, “T2i-compbench: A comprehensive benchmark for open-world compositional textto-image generation,” Advances in Neural Information Processing Systems, vol. 36, pp. 78723–78747, 2023.
- [210] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.
- [211] E. Betzalel, C. Penso, A. Navon, and E. Fetaya, “A study on the evaluation of generative models,” arXiv preprint arXiv:2206.10935, 2022.
- [212] T. Unterthiner, S. Van Steenkiste, K. Kurach, R. Marinier, M. Michalski, and S. Gelly, “Fvd: A new metric for video generation,” 2019.
- [213] M. Saito, S. Saito, M. Koyama, and S. Kobayashi, “Train sparsely, generate densely: Memory-efficient unsupervised training of highresolution temporal gan,” International Journal of Computer Vision, vol. 128, no. 10, pp. 2586–2606, 2020.
- [214] Y. Kirstain, A. Polyak, U. Singer, S. Matiana, J. Penna, and O. Levy, “Pick-a-pic: An open dataset of user preferences for text-to-image generation,” Advances in neural information processing systems, vol. 36, pp. 36652–36663, 2023.
- [215] Y. Liang, J. He, G. Li, P. Li, A. Klimovskiy, N. Carolan, J. Sun, J. Pont-Tuset, S. Young, F. Yang et al., “Rich human feedback for text-to-image generation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 19401–19411.
- [216] H. Nam, J. Kim, D. Lee, and J. C. Ye, “Optical-flow guided prompt optimization for coherent video generation,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 7837–7846.
- [217] M. Cai, S. Li, W. Li, X. Huang, H. Chen, J. Hu, and Y. Wang, “Dspo: Direct semantic preference optimization for real-world image super-resolution,” arXiv preprint arXiv:2504.15176, 2025.
- [218] T. Li, J. Liu, H. Chen, and Q. Liu, “Instructrl4pix: Training diffusion for image editing by reinforcement learning,” arXiv preprint arXiv:2406.09973, 2024.
- [219] R. Zhang, C. Tong, Z. Zhao, Z. Guo, H. Zhang, M. Zhang, J. Liu, P. Gao, and H. Li, “Let’s verify and reinforce image generation step by step,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 28662–28672.
- [220] F. Dai, Z. Zhuang, Y. Huang, S. Huang, B. Liao, D. Wang, and F. Yuan, “Vard: Efficient and dense fine-tuning for diffusion models with value-based rl,” arXiv preprint arXiv:2505.15791, 2025.
- [221] Z. Li, Y. Li, C. Meng, Z. Liu, Y. Ling, S. Zhang, G. Yang, C. Yang, Z. Yang, and L. Sun, “Inversion-dpo: Precise and efficient posttraining for diffusion models,” arXiv preprint arXiv:2507.11554, 2025.
- [222] X. Hu, R. Wang, Y. Fang, B. Fu, P. Cheng, and G. Yu, “Ella: Equip diffusion models with llm for enhanced semantic alignment,” arXiv preprint arXiv:2403.05135, 2024.
- [223] W. Wu, Z. Li, Y. He, M. Z. Shou, C. Shen, L. Cheng, Y. Li, T. Gao, and D. Zhang, “Paragraph-to-image generation with informationenriched diffusion model,” International Journal of Computer Vision, pp. 1–22, 2025.
- [224] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, “Evaluating object hallucination in large vision-language models,” arXiv preprint arXiv:2305.10355, 2023.
- [225] K. Cheng, Q. Sun, Y. Chu, F. Xu, Y. Li, J. Zhang, and Z. Wu, “Seeclick: Harnessing gui grounding for advanced visual gui agents,” arXiv preprint arXiv:2401.10935, 2024.
- [226] K. Li, Z. Meng, H. Lin, Z. Luo, Y. Tian, J. Ma, Z. Huang, and T.-S. Chua, “Screenspot-pro: Gui grounding for professional highresolution computer use,” arXiv preprint arXiv:2504.07981, 2025.
- [227] S. Nayak, X. Jian, K. Q. Lin, J. A. Rodriguez, M. Kalsi, R. Awal, N. Chapados, M. T. Ozsu,¨ A. Agrawal, D. Vazquez et al., “Uivision: A desktop-centric gui benchmark for visual perception and interaction,” arXiv preprint arXiv:2503.15661, 2025.

- [228] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried et al., “Webarena: A realistic web environment for building autonomous agents,” arXiv preprint arXiv:2307.13854, 2023.
- [229] T. Xie, D. Zhang, J. Chen, X. Li, S. Zhao, R. Cao, T. J. Hua, Z. Cheng, D. Shin, F. Lei et al., “Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments,” Advances in Neural Information Processing Systems, vol. 37, pp. 52040–52094, 2024.
- [230] R. Bonatti, D. Zhao, F. Bonacci, D. Dupont, S. Abdali, Y. Li, Y. Lu, J. Wagle, K. Koishida, A. Bucker et al., “Windows agent arena: Evaluating multi-modal os agents at scale,” arXiv preprint arXiv:2409.08264, 2024.
- [231] X. H. Lu,` A. Kazemnejad, N. Meade, A. Patel, D. Shin, A. Zambrano, K. Stanczak,´ P. Shaw, C. J. Pal, and S. Reddy, “Agentrewardbench: Evaluating automatic evaluations of web agent trajectories,” arXiv preprint arXiv:2504.08942, 2025.
- [232] H. Chae, N. Kim, K. T.-i. Ong, M. Gwak, G. Song, J. Kim, S. Kim, D. Lee, and J. Yeo, “Web agents with world models: Learning and leveraging environment dynamics in web navigation,” arXiv preprint arXiv:2410.13232, 2024.
- [233] H. Bai, Y. Zhou, J. Pan, M. Cemri, A. Suhr, S. Levine, and A. Kumar, “Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning,” Advances in Neural Information Processing Systems, vol. 37, pp. 12461–12495, 2024.
- [234] S. Jordan, Y. Chandak, D. Cohen, M. Zhang, and P. Thomas, “Evaluating the performance of reinforcement learning algorithms,” in International Conference on Machine Learning. PMLR, 2020, pp. 4962–4973.
- [235] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. Bellemare, “Deep reinforcement learning at the edge of the statistical precipice,” Advances in neural information processing systems, vol. 34, pp. 29304–29320, 2021.
- [236] J. Hessel, A. Holtzman, M. Forbes, R. L. Bras, and Y. Choi, “Clipscore: A reference-free evaluation metric for image captioning,” arXiv preprint arXiv:2104.08718, 2021.
- [237] Y. Chen, Y. Ge, R. Wang, Y. Ge, L. Qiu, Y. Shan, and X. Liu, “Exploring the effect of reinforcement learning on video understanding: Insights from seed-bench-r1,” arXiv preprint arXiv:2503.24376, 2025.
- [238] J. Cheng, Y. Ge, T. Wang, Y. Ge, J. Liao, and Y. Shan, “Video-holmes: Can mllm think like holmes for complex video reasoning?” arXiv preprint arXiv:2505.21374, 2025.
- [239] W. Xu, J. Wang, W. Wang, Z. Chen, W. Zhou, A. Yang, L. Lu, H. Li, X. Wang, X. Zhu, W. Wang, J. Dai, and J. Zhu, “Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models,” arXiv preprint arXiv:2504.15279, 2025. [Online]. Available: https://arxiv.org/abs/2504.15279
- [240] H. Yao, Q. Yin, J. Zhang, M. Yang, Y. Wang, W. Wu, F. Su, L. Shen, M. Qiu, D. Tao, and J. Huang, “R1-sharevl: Incentivizing reasoning capability of multimodal large language models via share-grpo,” 2025.
- [241] S. Tian, R. Wang, H. Guo, P. Wu, Y. Dong, X. Wang, J. Yang, H. Zhang, H. Zhu, and Z. Liu, “Ego-r1: Chain-of-tool-thought for ultra-long egocentric video reasoning,” 2025.
- [242] Y. Chen, W. Huang, B. Shi, Q. Hu, H. Ye, L. Zhu, Z. Liu, P. Molchanov, J. Kautz, X. Qi et al., “Scaling rl to long videos,” arXiv preprint arXiv:2507.07966, 2025.
- [243] H. Shao, S. Qian, H. Xiao, G. Song, Z. Zong, L. Wang, Y. Liu, and H. Li, “Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models,” arXiv preprint arXiv:2403.16999,

- 2024.

[244] Y. Wang, Z. Wang, B. Xu, Y. Du, K. Lin, Z. Xiao, Z. Yue, J. Ju, L. Zhang, D. Yang, X. Fang, Z. He, Z. Luo, W. Wang, J. Lin, J. Luan, and Q. Jin, “Time-r1: Post-training large vision language model for temporal video grounding,” arXiv preprint arXiv:2503.13377,

- 2025.

- [245] J. Yang, S. Yang, A. Gupta, R. Han, L. Fei-Fei, and S. Xie, “Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces,” arXiv preprint arXiv:2412.14171, 2024.
- [246] J. Yuan, T. Peng, Y. Jiang, Y. Lu, R. Zhang, K. Feng, C. Fu, T. Chen, L. Bai, B. Zhang et al., “Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms,” arXiv preprint arXiv:2505.21327, 2025.
- [247] K. Wang, J. Pan, L. Wei, A. Zhou, W. Shi, Z. Lu, H. Xiao, Y. Yang, H. Ren, M. Zhan, and H. Li, “Mathcoder-

- VL: Bridging vision and code for enhanced multimodal mathematical reasoning,” in The 63rd Annual Meeting of the Association for Computational Linguistics, 2025. [Online]. Available: https://openreview.net/forum?id=nuvtX1imAb
- [248] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao, “Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts,” in International Conference on Learning Representations (ICLR), 2024.
- [249] R. Zhang, D. Jiang, Y. Zhang, H. Lin, Z. Guo, P. Qiu, A. Zhou, P. Lu, K.-W. Chang, P. Gao et al., “Mathverse: Does your multimodal llm truly see the diagrams in visual math problems?” arXiv preprint arXiv:2403.14624, 2024.
- [250] Y. Hao, J. Gu, H. W. Wang, L. Li, Z. Yang, L. Wang, and Y. Cheng, “Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark,” arXiv preprint arXiv:2501.05444, 2025.
- [251] R. Qiao, Q. Tan, G. Dong, M. Wu, C. Sun, X. Song, Z. GongQue, S. Lei, Z. Wei, M. Zhang et al., “We-math: Does your large multimodal model achieve human-like mathematical reasoning?” arXiv preprint arXiv:2407.01284, 2024.
- [252] C. Zou, X. Guo, R. Yang, J. Zhang, B. Hu, and H. Zhang, “Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models,” 2024.
- [253] H. Cai, Y. Yang, and W. Hu, “Mm-iq: Benchmarking human-like abstraction and reasoning in multimodal models,” arXiv preprint arXiv:2502.00698, 2025.
- [254] C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, J. Liu, L. Qi, Z. Liu, and M. Sun, “Olympiadbench: A challenging benchmark for promoting agi with olympiadlevel bilingual multimodal scientific problems,” 2024.
- [255] J. Roberts, M. R. Taesiri, A. Sharma, A. Gupta, S. Roberts,

I. Croitoru, S.-V. Bogolin, J. Tang, F. Langer, V. Raina et al., “Zerobench: An impossible visual benchmark for contemporary large multimodal models,” arXiv preprint arXiv:2502.09696, 2025.

- [256] X. Yue, T. Zheng, Y. Ni, Y. Wang, K. Zhang, S. Tong, Y. Sun, B. Yu, G. Zhang, H. Sun, Y. Su, W. Chen, and G. Neubig, “Mmmupro: A more robust multi-discipline multimodal understanding benchmark,” arXiv preprint arXiv:2409.02813, 2024.
- [257] D. Jiang, R. Zhang, Z. Guo, Y. Li, Y. Qi, X. Chen, L. Wang, J. Jin, C. Guo, S. Yan et al., “Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency,” arXiv preprint arXiv:2502.09621, 2025.
- [258] Q. Yan, Y. Fan, H. Li, S. Jiang, Y. Zhao, X. Guan, C.-C. Kuo, and X. E. Wang, “Multimodal inconsistency reasoning (mmir): A new benchmark for multimodal reasoning models,” 2025. [Online]. Available: https://arxiv.org/abs/2502.16033
- [259] J. Wang, Y. Ming, Z. Shi, V. Vineet, X. Wang, Y. Li, and N. Joshi, “Is a picture worth a thousand words? delving into spatial reasoning for vision language models,” in The Thirty-Eighth Annual Conference on Neural Information Processing Systems, 2024.
- [260] Q. Yang, S. Yao, W. Chen, S. Fu, D. Bai, J. Zhao, B. Sun, B. Yin, X. Wei, and J. Zhou, “Humanomniv2: From understanding to omni-modal reasoning with context,” arXiv preprint arXiv:2506.21277, 2025.
- [261] X. Wu, Y. Hao, K. Sun, Y. Chen, F. Zhu, R. Zhao, and H. Li, “Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis,” arXiv preprint arXiv:2306.09341, 2023.
- [262] J. A. Rodriguez, A. Puri, S. Agarwal, I. H. Laradji, P. Rodriguez, S. Rajeswar, D. Vazquez, C. Pal, and M. Pedersoli, “Starvector: Generating scalable vector graphics code from images and text,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16175–16186.
- [263] X. Deng, Y. Gu, B. Zheng, S. Chen, S. Stevens, B. Wang, H. Sun, and Y. Su, “Mind2web: Towards a generalist agent for the web,” Advances in Neural Information Processing Systems, vol. 36, pp. 28091–28114, 2023.
- [264] J. Zhang, J. Wu, Y. Teng, M. Liao, N. Xu, X. Xiao, Z. Wei, and D. Tang, “Android in the zoo: Chain-of-action-thought for gui agents,” arXiv preprint arXiv:2403.02713, 2024.
- [265] R. Kapoor, Y. P. Butala, M. Russak, J. Y. Koh, K. Kamble, W. AlShikh, and R. Salakhutdinov, “Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web,” in European Conference on Computer Vision. Springer, 2024, pp. 161–178.
- [266] W. Chen, J. Cui, J. Hu, Y. Qin, J. Fang, Y. Zhao, C. Wang, J. Liu, G. Chen, Y. Huo et al., “Guicourse: From general vision language

- models to versatile gui agents,” arXiv preprint arXiv:2406.11317, 2024.
- [267] X. Puig, E. Undersander, A. Szot, M. D. Cote, T.-Y. Yang, R. Partsey, R. Desai, A. W. Clegg, M. Hlavac, S. Y. Min et al., “Habitat 3.0: A co-habitat for humans, avatars and robots,” arXiv preprint arXiv:2310.13724, 2023.
- [268] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the nav-graph: Vision-and-language navigation in continuous environments,” in European Conference on Computer Vision. Springer, 2020, pp. 104–120.
- [269] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “Rlbench: The robot learning benchmark & learning environment,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3019–3026, 2020.
- [270] S. Nasiriany, A. Maddukuri, L. Zhang, A. Parikh, A. Lo, A. Joshi, A. Mandlekar, and Y. Zhu, “Robocasa: Large-scale simulation of everyday tasks for generalist robots,” arXiv preprint arXiv:2406.02523, 2024.
- [271] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone, “Libero: Benchmarking knowledge transfer for lifelong robot learning,” Advances in Neural Information Processing Systems, vol. 36, pp. 44776–44791, 2023.
- [272] S. Zhang, Z. Xu, P. Liu, X. Yu, Y. Li, Q. Gao, Z. Fei, Z. Yin, Z. Wu, Y.-G. Jiang et al., “Vlabench: A large-scale benchmark for languageconditioned robotics manipulation with long-horizon reasoning tasks,” arXiv preprint arXiv:2412.18194, 2024.
- [273] R. Abhyankar, Q. Qi, and Y. Zhang, “Osworld-human: Benchmarking the efficiency of computer-use agents,” arXiv preprint arXiv:2506.16042, 2025.

- TABLE 3: Public benchmarks of MLLM most commonly used in visual RL. Only benchmarks relevant to visual reinforcement learning are included (RL-focused training & evaluation); task-specific benchmarks, such as MME [189], are excluded from consideration. ‘Tr’ and ‘Te’ refer to the ‘Train’ and ‘Test’, respectively.

Task family Benchmark Date Tr/Te Description (benchmark info. and RL reward signal)

SEED-Bench-R1 [237] Mar 2025 Tr&Te Video-QA pairs with human-preference reward model (used in [237]) Video-Holmes [238] May 2025 Te Human-ranked T2I pairs from diverse generation models (used in [238]) VisuLogic [239] Apr 2025 Tr&Te Visual-reasoning QA set; exact-match reward enables RL fine-tuning (used in [239]) R1-ShareVL [240] May 2025 Tr 52 k MM-Eureka subset for Share-GRPO RL training (used in [240]) Ego-R1 [241] Jun 2025 Tr&Te 25 k CoTT egocentric traces enabling RL training for ultra-long video reasoning (used in [241]) Long-RL [242] Jul 2025 Tr&Te 104 K long-video QA pairs (GRPO accuracy / format reward) (used in [242]) VisCOT [243] Mar 2024 Tr&Te 438 k visual chain-of-thought traces with step-wise rewards for RL (used in [243]) MMK12 [77] Mar 2025 Tr&Te 15.6 K multimodal math problems (rule-based accuracy / format rewards) (used in [77]) Time-R1 [244] Mar 2025 Tr&Te 2.5 K TimeRFT grounding spans (IoU reward) (used in [244]) VSI-Bench [245] Dec 2024 Te Spatial QA benchmark offering RL exact-match reward (used in [245]) MMEReasoning [246]

May 2025 Te Logic QA benchmark (used in [246])

K12-2M [247] May 2025 Tr 2 M multimodal math pairs for RL training (used in [247]) MathVista [248] Oct 2023 Te Visual-math QA benchmark; test-only RL exact-match reward (used in [248]) MathVerse [249] Mar 2024 Te Diagram-math QA benchmark; test-only RL exact-match reward (used in [249]) EMMA [250] Jan 2025 Te Robust multimodal reasoning benchmark challenging current MLLMs (used in [250]) WeMath [251] Jul 2024 Te Multimodal math reasoning benchmark with diagrams (used in [251]) DynaMATH [252] Oct 2024 Te Dynamic visual math reasoning robustness benchmark for VLMs (used in [252]) MM-IQ [253] Feb 2025 Te Human-like multimodal abstraction and reasoning benchmark (used in [253]) OlympiadBench [254] Feb 2024 Te Bilingual multimodal Olympiad-level scientific reasoning benchmark (used in [254]) ZeroBench [255] Feb 2025 Te near-impossible visual reasoning stress test for LMMs (used in [255]) MMMU-Pro [256] Sep 2024 Te robust multidisciplinary multimodal understanding benchmark (used in [256]) MME-CoT [257] Feb 2025 Te Multimodal chain-of-thought reasoning benchmark dataset (used in [257]) MMIR [258] Feb 2025 Te Multimodal inconsistency reasoning benchmark dataset (used in [258]) SpatialEval [259] Jun 2024 Te Synthetic spatial reasoning benchmark for VLMs (used in [259]) MMReason [259] Jun 2025 Te Synthetic spatial reasoning benchmark for VLMs (used in [259]) IntentBench [260] Jun 2025 Te Omnimodal evaluation demands unified audio-visual comprehension (used in [260])

Multimodal LLMs / VLMs

ImageReward [21] Apr 2023 Tr&Te Human-ranked pairs for T2I; reward model outputs scalar score (used in [129], [132], [134]) HPS [24] Mar 2023 Tr&Te Human-ranked T2I pairs from diverse generation models (used in [24]) HPS V2 [261] Jun 2023 Tr&Te Human-ranked T2I pairs across diverse prompts, preference-based reward (used in [120], [124]) Pick-a-Pic [214] May 2023 Tr&Te User preferences for pairwise rankings over T2I generations (used in [122], [123], [124], [126]) VideoReward [141] Jan 2025 Te Human-ranked video pairs across quality, motion, and text alignment (used in [141]) T2I-CompBench [209] Jul 2023 Tr&Te Compositional text-to-image dataset covering attributes and object relations. (used in [117],

Visual Generation (image/video/3D)

[120]) StarVector [262] Dec 2023 Tr&Te SVG Code Generation Data, Match Reward (used in [119]) AnimeReward [143] Apr 2025 Tr Multi-dimensional (e.g., character consisten) human preference anime videos (used in [143]) VideoPrefer [147] Dec 2024 Tr MLLM-annotated 135K video preference pairs (used in [147])

GUI-R1-3K [10] Apr 2025 Tr&Te GUI trajectories spanning Windows, Linux, macOS, Android, and Web platforms (used in [10]) SE-GUI-3k [11] May 2025 Tr 3,018 examples (desktop / web / mobile) with instruction and bounding box (used in [11]) UI-R1 [156] May 2025 Tr 136 mobile GUI tasks covering 5 action types (click, scroll, swipe, text-input) (used in [11]) CAGUI [158] Jun 2025 Te 55 K trajectories from 30 Chinese Android apps, 8 domains (used in [11]) Mobile-R1 [160] Jun 2025 Tr&Te More than 500 online task trajectories from 28 Chinese apps (used in [160]) Mind2web [263] Jun 2023 Tr&Te 2 k tasks on 137 real websites; success/fail reward for RL (used in [162], [263]) AITZ [264] Jun 2023 Tr&Te 18,643 Android screen–action pairs with CoAT reasoning (used in [264]) Omniac [265] Feb 2024 Tr&Te Desktop + web 9.8 k scripted tasks (used in [265]) GUICours [266] Jun 2024 Tr&Te GUIEnv/Act/Chat datasets (10 M OCR + 67 k navigation) (used in [266]) Habitat 3.0 [267] Oct 2023 Tr&Te Interactive embodied-AI scenes with humans and robots (used in [267]) VLN-CE [268] Apr 2020 Tr&Te Continuous embodied navigation dataset with language instructions (used in [268]) Rlbench [269] Apr 2020 Tr&Te Multi-task simulated robot manipulation benchmark dataset (used in [269]) RoboCasa [270] Jun 2024 Tr Large-scale kitchen-task simulation for generalist robotics (used in [270]) LIBERO [271] Jun 2023 Tr Lifelong robot learning benchmark with 100 manipulation tasks (used in [271]) VLABench [272] Dec 2024 Tr&Te Long-horizon language-conditioned manipulation benchmark for robots (used in [272])

Vision–Language Action Agents

- TABLE 4: Public benchmarks of Visual Generation (image/video/3D) most commonly used in visual RL. Only benchmarks relevant to visual reinforcement learning are included (RL-focused training & evaluation). ‘Tr’ and ‘Te’ refer to the ‘Train’ and ‘Test’, respectively.

Benchmark Date Tr/Te Description (benchmark info. and RL reward signal) ImageReward [21] Apr 2023 Tr&Te Human-ranked pairs for T2I; reward model outputs scalar score (used

in [129], [132], [134]) HPS [24] Mar 2023 Tr&Te Human-ranked T2I pairs from diverse generation models (used in [24]) HPS V2 [261] Jun 2023 Tr&Te Human-ranked T2I pairs across diverse prompts, preference-based reward

(used in [120], [124])

Pick-a-Pic [214] May 2023 Tr&Te User preferences for pairwise rankings over T2I generations (used in [122],

[123], [124], [126])

VideoReward [141] Jan 2025 Te Human-ranked video pairs across quality, motion, and text alignment (used

in [141])

T2I-CompBench [209] Jul 2023 Tr&Te Compositional text-to-image dataset covering attributes and object relations.

(used in [117], [120]) StarVector [262] Dec 2023 Tr&Te SVG Code Generation Data, Match Reward (used in [119]) AnimeReward [143] Apr 2025 Tr Multi-dimensional (e.g., character consisten) human preference anime videos

(used in [143])

VideoPrefer [147] Dec 2024 Tr MLLM-annotated 135K video preference pairs (used in [147])

- TABLE 5: Public benchmarks of VLAs most commonly used in visual RL. Only benchmarks relevant to visual reinforcement learning are included (RL-focused training & evaluation). ‘Tr’ and ‘Te’ refer to the ‘Train’ and ‘Test’, respectively.

Benchmark Date Tr/Te Description (benchmark info. and RL reward signal) GUI-R1-3K [10] Apr 2025 Tr&Te GUI trajectories spanning Windows, Linux, macOS, Android, and Web

platforms SE-GUI-3k [11] May 2025 Tr 3,018 examples (desktop / web / mobile) with instruction and bounding box UI-R1 [156] May 2025 Tr 136 mobile GUI tasks covering 5 action types (click, scroll, swipe, text-input) CAGUI [158] Jun 2025 Te 55 K trajectories from 30 Chinese Android apps, 8 domains Mobile-R1 [160] Jun 2025 Tr&Te More than 500 online task trajectories from 28 Chinese apps Mind2web [263] Jun 2023 Tr&Te 2 k tasks on 137 real websites; success/fail reward for RL AITZ [264] Jun 2023 Tr&Te 18,643 Android screen–action pairs with CoAT reasoning Omniac [265] Feb 2024 Tr&Te Desktop + web 9.8 k scripted tasks GUICours [266] Jun 2024 Tr&Te GUIEnv/Act/Chat datasets (10 M OCR + 67 k navigation) Habitat [267] Oct 2023 Tr&Te Interactive embodied-AI scenes with humans and robots VLN-CE [268] Apr 2020 Tr&Te Continuous embodied navigation dataset with language instructions RLBench [269] Apr 2020 Tr&Te Multi-task simulated robot manipulation benchmark dataset RoboCasa [270] Jun 2024 Tr Large-scale kitchen-task simulation for generalist robotics LIBERO [271] Jun 2023 Tr Lifelong robot learning benchmark with 100 manipulation tasks VLABench [272] Dec 2024 Tr&Te Long-horizon language-conditioned manipulation benchmark for robots

