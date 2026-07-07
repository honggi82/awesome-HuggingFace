## πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models

#### Kang Chen2,6,∗ Zhihao Liu3,6,∗ Tonghe Zhang4,∗,♯ Zhen Guo5 Si Xu5 Hao Lin5 Hongzhi Zang1 Xiang Li5 Bingwen Wei1 Jiakai Zhou1 Quanlu Zhang5 Zhaofei Yu2 Guoliang Fan3 Tiejun Huang2 Yu Wang1† Chao Yu1†

[Figure 1]

###### https://github.com/RLinf/RLinf https://huggingface.co/RLinf

###### Training Paradigm OOD Eval

Flow-based VLAs

[Figure 2]

🚩

[Figure 3]

[Figure 4]

Buffer Data

# arXiv:2510.25889v3[cs.LG]29Jan2026

PerformanceGain

I. Policy Rollout II. Actor Update

[Figure 5]

[Figure 6]

[Figure 7]

PPO GRPO

###### III. RL

[Figure 8]

👆

EmbodiedIntelligence

Online Interaction with Env RL with the Buffer

Model Weight

[Figure 9]

🚩

A Vision-Language-Action Flow Model for General Robot Control

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

II. SFT

👆

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Large-scale Datasets

[Figure 21]

[Figure 22]

I. Pretraining

[Figure 23]

🚩

Specific Task Datasets

[Figure 24]

👆

A Vision-Language-Action Model with Open-World Generalization

Data

In-Distribution RL Training Key Experiments

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

SFT + RL 94.0%

Real Sim

[Figure 29]

[Figure 30]

[Figure 31]

LIBERO-Long

[Figure 32]

[Figure 33]

[Figure 34]

One-shot SFT 43.9%

Data Efficiency Real2Sim2Real

Figure 1. πRL: An online RL framework for flow-based VLAs. Incorporating two solutions, Flow-Noise and Flow-SDE, πRL enhances the performance and generalization of SFT-aligned models across extensive ID benchmarks and OOD settings. Refined with RL, few-shot SFT policies achieve performance comparable to full dataset baselines. Additionally, we facilitate seamless zero-shot sim-to-real transfer by constructing a simulator with 3D Gaussian Splatting as the rendering engine to narrow the visual domain gap.

### Abstract

π0.5) remains challenging due to intractable action log-likelihoods raised from flow matching. We address this challenge with πRL, featuring two technical approaches: (1) Flow-Noise models the denoising process as a discrete-time MDP with a learnable noise network for exact log-likelihood computation. (2) Flow-SDE integrates denoising with agent-environment interaction, formulating a two-layer MDP that employs ODE-to-SDE conversion for efficient RL exploration. We evaluate πRL across various benchmarks, with experiments demonstrating that RL yields significant performance improvements in both in-distribution and out-of-distribution settings.

Vision-Language-Action (VLA) models enable robots to understand and perform complex tasks from multimodal input. Although recent work explores using reinforcement learning (RL) to automate the laborious data collection process in scaling supervised fine-tuning (SFT), applying RL to large-scale flow-based VLAs (e.g., π0,

1Tsinghua University 2Peking University 3Institute of Automation, Chinese Academy of Sciences 4Carnegie Mellon University 5Infinigence AI 6Zhongguancun Academy ♯ Work done at Tsinghua University ∗ Equal Contributions . Correspondence to: Yu Wang <yu-wang@tsinghua.edu.cn>, Chao Yu <zoeyuchao@gmail.com>.

Preprint. January 30, 2026.

### 1. Introduction

Vision-Language-Action (VLA) models (Din et al., 2025) have emerged as a leading solution for general-purpose robots, effectively bridging the gap between high-level multimodal reasoning and low-level physical control (Firoozi et al., 2025). Conditioned on sensor inputs and language commands, VLAs (Team et al., 2024; Kim et al., 2024; Black et al., 2024; Intelligence et al., 2025) can translate abstract instructions into executable robotic actions, thereby enabling intuitive and flexible human-robot interaction.

The training methodology for VLAs follows the standard pre-training and supervised fine-tuning (SFT) paradigm as shown in Fig. 1. Building on the pretrained VisionLanguage Model (VLM) (Touvron et al., 2023; Beyer et al.,

- 2024), VLAs are fine-tuned on large-scale, heterogeneous human demonstration datasets (O’Neill et al., 2024; Khazatsky et al., 2024), followed by SFT on the target task to align their capabilities with the specific embodiment and environment. However, reliance on SFT introduces a critical challenge: curating large-scale, high-quality expert trajectories is both laborious and costly (Din et al., 2025). Besides, models obtained via SFT tend to overfit to expert demonstrations (Fei et al., 2025), with their performance fundamentally constrained by the quality of expert demonstrations.

Recent efforts (Zang et al., 2025; Li et al., 2025a; Tan et al., 2025; Liu et al., 2025a) have explored expanding the VLA training process with reinforcement learning (RL), establishing a pre-training, SFT, and RL paradigm as shown in Fig. 1, allowing VLAs to improve their performance beyond expert demonstrations through environmental interaction and develop more generalizable policies.

However, these RL advances have been largely confined to autoregressive VLAs, featuring OpenVLA (Kim et al.,

- 2024) and OpenVLA-OFT (Kim et al., 2025), which employ discrete action decoders that generate output in an autoregressive or parallel fashion. This stands in stark contrast to flow-based VLAs, exemplified by the π series models, which generate actions through iterative refinement in flow matching (Lipman et al., 2022), offering the advantages of generating action chunks in high-frequency and performing highly dexterous tasks (Black et al., 2024). Consequently, previous VLA-RL algorithms are incompatible with flowbased VLAs, and the fundamental challenge lies in how to characterize a logarithmic likelihood (Hutchinson, 1989; Chen et al., 2018) for the executed actions.

In this paper, we introduce πRL, a framework designed for fine-tuning flow-based VLAs with online RL algorithms. To address the intractable log-likelihood estimation problem in flow matching, we propose two solutions. Flow-Noise integrates a learnable noise network into the denoising process and models this stage as a discrete-time Markov decision

process (MDP) for exact log-likelihood estimation. FlowSDE converts the ordinary differential equation (ODE) denoising process into a stochastic differential equation (SDE) while maintaining equivalent marginal distributions for exploration, and builds a two-layer MDP that couples the denoising process with policy-environment interaction. Given the formulated MDP and the exact log-likelihood computation, πRL undergoes further optimization via the proximal policy optimization (PPO) (Schulman et al., 2017).

We conduct extensive experiments on various benchmarks to evaluate the effectiveness of πRL on π0 (Black et al., 2024) and π0.5 (Intelligence et al., 2025) models. Across all benchmarks, the proposed framework consistently yields substantial performance gains over SFT baselines. Furthermore, out of distribution evaluations confirm that our model yields genuine policy enhancement rather than narrow overfitting on the target environment. To sum up, our contributions are:

- • RL for flow-based VLAs. We introduce πRL, an online RL fine-tuning framework with Flow-Noise and FlowSDE formulations for flow-based VLAs.
- • Superior Performance. We demonstrate significant performance improvements and enhanced generalization of πRL across various benchmarks.
- • Comprehensive Ablation. We conduct thorough ablation studies, offering empirical insights to guide future RL research on flow-based VLAs.
- • Open-source Code and Models. We release all codes to ensure reproducibility, hoping that our study helps to advance further research in this field.

### 2. Related Work

#### 2.1. Vision-Language-Action Models

VLA models have recently achieved remarkable progress in robotics by integrating multimodal inputs to enable unified perception, reasoning, and control. This development has led to a series of architectures, including Octo (Team et al., 2024), RT (Brohan et al., 2022), OpenVLA, OpenVLA-OFT, π0, π0.5, and GR00T (Bjorck et al., 2025).

#### 2.2. Online RL Fine-tuning for VLA Models

Recent research has increasingly focused on enhancing the performance and generalization of VLAs with online RL. For example, SimpleVLA-RL (Li et al., 2025a), building on the OpenVLA-OFT and GRPO, demonstrated that RL can improve long-horizon planning of VLA models under data scarcity. RL4VLA (Liu et al., 2025a) empirically evaluated PPO, GRPO, and direct preference optimization

(DPO) (Rafailov et al., 2023) with stage-based sparse rewards. RLinf-VLA (Yu et al., 2025; Zang et al., 2025) provides a unified and efficient framework for scalable RL training of VLA models. These works demonstrate the effectiveness of RL fine-tuning VLA models.

- 2.3. RL Fine-tuning for Flow Models

Integrating RL with flow models is a promising way to transcend the limitations of imitation learning. To this end, Flow-GRPO (Liu et al., 2025b) converts the deterministic ODE into an equivalent SDE to enable stochasticity exploration, a foundation upon which subsequent works like MixGRPO (Li et al., 2025b) and TempFlow-GRPO (He et al.,

- 2025) further accelerate training through hybrid ODE-SDE rollouts. ReinFlow (Zhang et al., 2025) injects learnable noise into the flow path and transforms it into a discretetime Markov process with a tractable likelihood for stable policy gradient updates. Flow policy optimization (FPO) (McAllister et al., 2025) reframes policy optimization as maximizing the advantage-weighted ratio of the conditional flow matching loss.

- 3. Preliminary

- 3.1. Problem Formulation

We formulate the task as an MDP, defined by a tuple M = (S,A,P0,PENV,RENV,γ). The state st ∈ S is defined as the robot observation ot and P0 denotes the initial state distribution. Given the state, the flow policy predicts an action at ∼ π(·|st) ∈ A, resulting in the state transition st+1 ∼ PENV(·|st,at) and a reward RENV(st,at). The objective is to learn a policy πθ that maximizes the expected γ-discounted return over a horizon of T + 1:

J (πθ) = Eπ

θ,P0

T

t=0

γtRENV(st,at) . (1)

With the policy gradient surrogate (Williams, 1992), the gradient of the return expectation can be approximated from sampled trajectories:

∇θJ (πθ) = Eπ

θ,P0

T

t=0

∇θ log πθ(at|st)A(st,at) .

(2) The advantage function, A(st,at) = Q(st,at) − V (st), measures the relative merit of the action value Q(st,at) over the state value V (st), providing a low-variance signal for the policy update.

- 3.2. Flow-based Vision-Language-Action Model

A flow-based VLA model πθ is designed to map the observation ot comprising RGB images, language tokens, and

robot proprioception to a sequence of H future actions At = [at,0,...,at,H−1], formulated as p(At|ot). Within the model, the VLM extracts features from the visual and language inputs, while the flow matching expert is tasked with generating the actions. Specifically, the model learns a conditional vector field vθ that transforms a standard Gaussian noise distribution into the target action At. This is achieved by minimizing the Conditional Flow Matching (CFM) loss, which aligns the predicted vector field vθ with the ground-truth vector field u:

t,ot),q(Aτt|At) ∥vθ(Aτt ,ot) − u(Aτt |At)∥22 .

LCFM = Eτ,p(A

(3) Here, the conditional probability path q(Aτt |At) generates a noisy action1 Aτt = τAt + (1 − τ)ϵ from an action At, random noise ϵ ∼ N(0,I), and a continuous time τ ∈ [0,1] in flow matching. For this specific path, the corresponding ground-truth vector field is defined as u(Aτt |At) = At − ϵ. During the inference, the action sequence is generated by first sampling a noise vector A0t ∼ N(0,I), which is further iteratively refined by integrating the learned vector field vθ over a fixed number of steps based on the forward Euler method: Aτt+δ = Aτt + vθ(Aτt ,ot) · δ.

### 4. Methodology

Existing VLA-RL approaches leverage base models such as OpenVLA for discrete actions and OpenVLA-OFT for continuous actions. To compute the action log-likelihood log πθ(at|st), discrete models (Liu et al., 2025a) apply softmax to the output logits, while continuous models (Li et al., 2025a) treat the action as a Gaussian distribution, employing a prediction head to estimate the variance. As for the flow-based VLAs, directly computing the exact likelihood (Hutchinson, 1989) is inaccurate with few denoising steps. Moreover, the deterministic nature of its ODE sampling process precludes exploration, making its implementation within RL non-trivial. To this end, we propose Flow-Noise and Flow-SDE, two technical approaches that make flowbased VLAs amenable to RL.

#### 4.1. Flow-Noise

Inspired by Reinflow (Zhang et al., 2025), we incorporate a learnable noise network into the flow matching denoising process and solve the problem within the standard one-layer MDP framework detailed in Sec. 3.1. By modeling the denoising stage as a discrete MDP, we can directly compute the log-likelihood of the denoised sequence, enabling equivalent policy optimization via RL.

1Aτt incorporates two temporal indices, t denotes the discrete time step for environment interaction and τ represents the continuous time variable in flow matching.

[Figure 35]

Gaussian Distribution

| | |
|---|---|
|Action Expert<br><br>[Figure 36]| |

|Vision Language Model<br><br>[Figure 37]| |
|---|---|
| |KV-cache|

Denoising Process

###### Observation

Action

One-Layer MDP

Flow-Noise

Images: Multi-view RGB camera inputs

###### Noise Injection

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

 Flow-Noise: Learnable Noise from the noise network.

Inner-MDP

Outer-MDP

Language: Goal-oriented command

“pick up the black bowl on the stove and place it on the plate”

 Flow-SDE: Fixed noise from the ODE-SDE conversion.

State: Proprioceptive feedback

Figure 2. Illustration for the noise injection in πRL.

Two-Layer MDP

Flow-SDE

- 4.1.1. STOCHASTICITY INJECTION

Figure 3. Illustration of the MDP formulations in πRL.

In Flow-Noise, we parameterize the noise schedule with a neural network, allowing the magnitude of the injected noise to be learned dynamically during training for greater flexibility, as shown in Fig. 2. We focus on the generation process within a single environment timestep t. For notational simplicity, we omit the time subscript t, e.g., writing Aτ, and denote the predicted velocity vθ(Aτ,o) as vτ.

entire denoising sequence A = (A0,...,A1) is depicted in Fig. 3 and formulated as:

K−1

log π(A|o) = log π(A0|o)

π(Aτ

#### |Aτ

,o) .

k+1

k

k=0

The step transition during the denoising process is modeled as a Gaussian distribution p(Aτ+δ|Aτ) ∼ N(µτ,Στ), where the mean is determined by the forward Euler update of the original ODE and the variance is controlled by the learnable noise network θ′:

(5)

Building on this, we can treat flow-based policy optimization within a standard MDP framework.

#### 4.2. Flow-SDE

µτ = Aτ + vτ · δ Στ = diag(σθ2′)

Inspired by Flow-GRPO (Liu et al., 2025b), we enhance stochastic exploration by converting the denoising process from ODE into an SDE formulation. We further construct a two-layer MDP to couple the denoising process with the policy-environment interaction following DPPO (Ren et al., 2024), while leveraging the hybrid ODE-SDE sampling technique to accelerate the training process.

. (4)

Here, σθ′(·) is the standard deviation learned from the noise injection network, conditioned on the action Aτ, and the observation o. The noise network is trained jointly with the velocity but discarded after fine-tuning, leaving a deterministic policy for inference.

4.2.1. STOCHASTICITY INJECTION

- 4.1.2. LOG-LIKELIHOOD ESTIMATION

In Flow-SDE, we convert the deterministic ODE into an equivalent SDE that preserves the marginal probability density of the generated actions, as shown in Fig. 2.

The primary challenge in applying policy gradient methods to flow-based VLAs stems from the intractable loglikelihood of the final executed action. In Flow-Noise, we address it by substituting the gradient of the joint loglikelihood of the entire denoising process into the policy optimization objective in Eq. (2), which is theoretically grounded in Reinflow (Zhang et al., 2025).

The deterministic ODE sampling trajectory of the flow matching, especially the Rectified Flow (Liu et al., 2022), is described by the forward Euler method:

##### dAτ = vτdτ. (6)

The inference process for action generation is discretized into K uniform steps, which defines a sequence of time points {τ0,τ1,...,τK}. With the step interval defined as δ = 1/K, the discrete timestep at the k-th point is τk = k·δ, starting from τ0 = 0 and culminating at τK = 1. Given the observation o, the exact and tractable log probability for the

Building on the connection between the probability flow ODE and SDE (Song et al., 2020), we can transform the deterministic ODE in Eq. (6) into an equivalent SDE, with a drift term that corrects the original velocity and a diffusion

term that introduces noise:

- 1

- 2

dAτ = vτ −

g2(τ)∇log qτ(Aτ) dτ

##### + g(τ)dw

##### ,

Diffusion Term

Drift Term

(7) where g(τ) is a scalar function controlling the noise schedule, ∇log qτ(Aτ) is the score function of the marginal distribution qτ and dw denotes a Wiener process.

As established in Flow-GRPO, the score function and the velocity field are critically linked by ∇log qτ(Aτ) = −A

τ

τ − 1−ττ vτ. By substituting the score function with the velocity field in Eq. (7) and setting the noise schedule g(τ) to στ = a 1−ττ with a controlling the noise level, we derive the final SDE formulation for the flow-matching sampler:

στ2 2τ

dAτ = vτ +

(Aτ + (1 − τ)vτ) dτ + στdwτ. (8)

Discretizing this SDE reveals that the transition probability p(Aτ+δ|Aτ) ∼ N(µτ,Στ) is an isotropic Gaussian distribution, with the mean and variance formulated as:

2 τ

µτ = Aτ + vτ + σ

2τ (Aτ + (1 − τ)vτ) · δ Στ = στ2δ · I

. (9)

- 4.2.2. MDP FORMULATION

We couple the denoising process of the flow matching with environmental interaction in Flow-SDE. Specifically, we embed the inner MDP defined during the denoising process into the high-level, outer-loop MDP with the environment MENV in Sec. 3.1, formulating a two-layer MDP as shown in Fig. 3, with components defined with respect to the environment time t and denoising time τ.

- • State s¯τt = (ot,Aτt ) is the tuple of the observation ot and the action state Aτt .
- • Action a¯τt is defined as the next sampled denoised action in the inner-loop and the executed action for the outer loop:

a¯τt =

Aτt+δ if τ < 1 A1t if τ = 1

, (10)

where Aτt+δ = µτ + στ

√

δ · ϵ, ϵ ∼ N(0,I) is the randomly sampled noise.

- • Transition P¯(¯sτ

′

t′ |s¯τt ,a¯τt ) defines how the state evolves, formulated as:

′

s¯τ

t′ =

(ot,a¯τt ) if τ < 1 (ot+1,A0t+1) if τ = 1

. (11)

For τ < 1, the inner loop transition PFLOW(·) occurs between different denoised action states, where the observation ot remains fixed and the next action state is set by a¯τt = Aτt+δ.

For τ = 1, the final action a¯τt = A1t interacts with the outer-loop environment, resulting in a new observation

ot+1 according to the environment dynamics PENV(·). Concurrently, the action state is reset from a standard

normal distribution A0t+1 ∼ N(0,I).

• Reward R¯(¯sτt ,a¯τt ) is granted only upon completion of the denoising process and interaction with the environment:

R¯(¯sτt ,a¯τt ) =

0 if τ < 1 RENV(ot,A1t) if τ = 1

. (12)

Within the two-layer MDP framework, the problem of estimating the action log-likelihood log π(at|st) is transformed into estimating log π(¯aτt |s¯τt ), which is straightforward to compute due to the Gaussian nature of the transitions.

- 4.2.3. HYBRID ODE-SDE SAMPLING

The two-layer MDP formulation significantly extends the horizon, increasing training difficulty and computational cost. To mitigate this, we adopt a mixed ODE-SDE rollout strategy (Li et al., 2025b; He et al., 2025). At each step t, we randomly sample a denoising time τt for the stochastic SDE exploration, while treating all remaining denoising steps as deterministic ODE updates. Specifically, the policy acts on the state s¯τ

t

t = (ot,Aτ

t

t ); subsequently, an environment wrapper executes the remaining ODE steps and the environment transition, ultimately yielding the next state s¯τt+1t+1 = (ot+1,Aτt+1t+1) at a newly sampled time τt+1. This formulation effectively shortens the MDP horizon while maintaining theoretical consistency with the original twolayer framework.

- 4.3. Policy Optimization 4.3.1. ALGORITHM

Given the formulated flow policy MDP, our objective is to learn the optimal parameters θ∗ for the policy πθ that maximizes the expected discounted return J (πθ). To this end, we apply the widely adopted policy gradient algorithm PPO to optimize the policy.

π-series models (Black et al., 2024; Intelligence et al., 2025) adopt a chunk-based approach for action generation. Specifically, the policy outputs an entire sequence of H future actions At = [at,0,...,at,H−1] in response to each observation. In this approach, we treat the entire sequence as a single macro-step and define its corresponding reward Rt = Hj=0−1 rt,j as the sum of the per-step rewards rt,j,

| | |
|---|---|
| | |

| | |
|---|---|
| | |

| | |
|---|---|
| | |
| | |

Action

Critic

Action

Action

Critic Action

| | |
|---|---|
| | |

| | |
|---|---|
| | |

Critic

Critic

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Action Expert

Action Expert

Action Expert

Action Expert

Vision Language Model

Vision Language Model

Vision Language Model

Vision Language Model

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

“clean room”

[Figure 64]

[Figure 65]

[Figure 66]

“fold shirt”

[Figure 67]

[Figure 68]

[Figure 69]

state

“clean room”

“fold shirt”

state

state noise

noise

state noise

noise

(a) Critic with the action expert, exemplified by π0.

(b) Critic with the VLM, exemplified by π0.5. Figure 4. Illustration of the two critic placement configurations.

referred to as the chunk-level formulation in RLinf-VLA (Zang et al., 2025).

Vvlm(ot) conditioned on the integrated image, language, and state inputs. Conversely, for the π0 variant, achieving the value prediction is non-trivial due to the coupled input structure, where the action expert requires both the noisy action Aτt and the state. To this end, we approximate Vexpert(ot) by averaging the value estimates across the entire denoising trajectory, formulated as:

To effectively guide policy updates, PPO employs Generalized Advantage Estimation (GAE) (Schulman et al., 2015) to compute a low-variance estimate of the advantage, estimated as:

T−t

Aˆt =

(γλ)kTt+k, (13)

Vexpert(ot) ≈ Eτ∼U[0,1][Vexpert(ot,Aτt )]. (16)

k=0

where the TD-error is Tt = Rt + γV (st+1) − V (st). Here, V (·) is the state-value function derived from the critic network, γ is the discount factor, and λ is the parameter that balances the trade-off between bias and variance in the advantage estimate.

### 5. Experimental Results

#### 5.1. Setup

Benchmarks. We perform experiments on four widelyadopted robot manipulation benchmarks: LIBERO (Liu et al., 2023), ManiSkill (Tao et al., 2024), MetaWorld (McLean et al., 2025) and CALVIN (Mees et al., 2022).

PPO constrains policy updates to a small trust region to prevent large, destabilizing updates, with the objective function:

Flow-based VLAs. We conduct our primary experiments based on π0 and π0.5 models. Additionally, we conduct experiments on GR00T in Appendix Sec. H, which validates that our algorithm can be applied to other flow-based VLAs.

J (πθ) = Et min ρt(θ)Aˆt, clip(ρt(θ),1 − ϵ,1 + ϵ)Aˆt ,

(14) where the clip function, governed by a hyperparameter ϵ, restricts the ratio ρt(θ) to the interval [1−ϵ,1+ϵ] to ensure training stability.

#### 5.2. Main Results

Here, the probability ratio ρt(θ) between the updated and old policies takes the form of either:

In this section, we assess the in-distribution (ID) performance of πRL across various benchmarks, followed by an analysis of its out-of-distribution (OOD) generalization.

##### (¯aτt |s¯τt ) πθ

##### (at|st) πθ

πθ

πθ

or ρt(θ) =

. (15)

new

new

ρt(θ) =

(¯aτt |s¯τt )

(at|st)

old

old

5.2.1. IN-DISTRIBUTION RL TRAINING

- 4.3.2. CRITIC DESIGN

As detailed in Tab. 1, πRL yields substantial performance gains over SFT baselines across all evaluated benchmarks. Specifically, the π0 model achieves a maximum average improvement of +29.2%, while the π0.5 variant demonstrates a +31.0% increase in average success rate.

Following VLA-PPO works (Zang et al., 2025; Liu et al.,

- 2025a), we employ a shared actor-critic architecture for memory-efficient value prediction as shown in Fig. 4. However, the two flow-based VLAs process the proprioceptive state differently: in π0, the state is fed into the action expert model, whereas in π0.5, it is merged with prompt embeddings within the VLM.

Specifically for LIBERO, we perform few-shot SFT on the π0.5 model followed by RL optimization to achieve a 98.3% success rate, outperforming the 96.9% success rate of the full-dataset SFT baseline. These performance gains extend to other challenging environments, including ManiSkill with

To this end, for the π0.5 variant, we attach the critic network directly to the VLM output, providing the value estimate

Table 1. Comprehensive ID performance comparison across four benchmarks.

Benchmarks LIBERO ManiSkill MetaWorld CALVIN Avg. ∆ Avg.

Model

SFT 57.6 38.4 50.8 57.5 51.1 Flow-SDE 96.1 78.8 78.1 61.7 78.7 +27.6

π0

- Flow-Noise 97.6 77.8 85.8 59.9 80.3 +29.2

π0.5

SFT 77.1 40.1 43.8 61.3 55.6 Flow-SDE 97.9 90.9 70.7 87.0 86.6 +31.0

- Flow-Noise 98.3 89.7 66.1 84.5 84.7 +29.1

Instruct

|0.5-SFT<br><br>0.5-RL<br><br>|
|---|

100

Pos

Vis-Img

1.0

0.8

+17.8%

IND Train

0.8

79.1%

OOD Eval

80

Pos-Chg

Tex-0.3

0.6

0.4

61.3%

SuccessRate

0.7

SuccessRate

0.2

60

M-Plate (Train)

Tex-0.5

40

0.6

M-Plate (Test)

Whole-0.3

20

M-Carrot (Train)

Whole-0.5 M-Carrot (Test)

0.5

0 100 200

0

Step

SFT RL

(c) MetaWorld Figure 5. Comprehensive OOD evaluation results on CALVIN ABC-D, ManiSkill OOD, and MetaWorld ML45 benchmarks.

(a) CALVIN

(b) ManiSkill

its 4,352 pick-and-place task combinations, MetaWorld featuring 50 distinct manipulation primitives, and CALVIN for long-horizon sequential tasks. See Appendix Sec. C for comprehensive experimental details.

- 5.2.2. OUT-OF-DISTRIBUTION RL EVALUATION

While previous experiments demonstrate that RL yields performance improvements, a critical question remains: does RL yield an enhanced policy, or simply overfit to the ID environment driven by provided rewards? In this section, we evaluate RL-finetuned policies in OOD scenarios, where the environment distribution or the task objective deviates from the ID training setup, to assess their generalization capabilities.

As illustrated in Fig. 5, the performance gains achieved in the ID setting effectively transfer to OOD scenarios in ManiSkill and CALVIN, where the domain shift primarily stems from environmental variations. Conversely, for the OOD setting in MetaWorld, which involves distinct manipulation tasks, performance fluctuates without showing significant improvement. This finding suggests that the benefits of RL are primarily localized to action-level refinement rather than broader augmentation of cross-task generalization capabilities. See Appendix Sec. D for more details.

#### 5.3. Ablation Study

Given that Flow-SDE achieves performance comparable to Flow-Noise while offering higher computational efficiency, we conduct our ablation studies with the Flow-SDE method. Specifically, we investigate the impact of critic designs, noise injection strategies, and MDP formulations, with additional results on RL algorithms and hyper-parameters provided in Appendix Sec. F.

5.3.1. CRITIC DESIGN

Placement. We compare two critic placement strategies, one positioned after the action expert (Vexpert) and the other after the VLM (Vvlm), with π0 model on the LIBERO-Long task suite. As illustrated in Fig. 6, we observe that Vvlm exhibits slightly superior performance, lower value loss, and higher explained variance, despite not receiving the proprioceptive state as input. This advantage can be attributed to a key difference in their input: Vvlm learns a direct mapping from observation to value, while Vexpert must contend with optimization challenges arising from coupled state and noisy action inputs.

Nevertheless, to align with the concept of the value function, we maintain the Vexpert architecture for the π0, ensuring that state information is incorporated to estimate the value.

0.8

0.08

0.9

One-Layer MLP Vexpert Four-Layer MLP Vexpert Four-Layer MLP Vvlm

0.7

0.07

ExplainedVariance

0.6

SuccessRate

0.06

ValueLoss

0.5

0.05

0.4

0.7

0.04

0.3

One-Layer MLP Vexpert Four-Layer MLP Vexpert Four-Layer MLP Vvlm

One-Layer MLP Vexpert Four-Layer MLP Vexpert Four-Layer MLP Vvlm

0.2

0.03

0.1

0.02

0.5

0.0

0 10 20 30 40 50

0 10 20 30 40 50

0 100 200 300 400

Step

Step

Step

(a) Eval

(b) Value Loss

(c) Explained Variance

Figure 6. Ablation on the critic design within Flow-SDE π0 on the LIBERO-Long, indicating that the critic Vvlm attached after the VLM exhibits superior performance. Furthermore, a four-layer MLP demonstrates stronger regression capability in Vexpert.

1.0

1.0

821.4

814.2

ExplainedVariance

800

0.8

0.9

UpdateTime(s)

SuccessRate

600

0.6

0.8

428.6

0.4

0.7

400

Two-Layer

Two-Layer

Hybrid Two-Layer

Hybrid Two-Layer

0.2

0.6

200

One-Layer

One-Layer

0.0

0

0 100 200

0 100 200

Two-Layer Hybrid Two-Layer

One-Layer

Step

Step

(a) Eval

(c) Update Time Figure 7. Ablation on the MDP formulation within Flow-SDE of π0 on the LIBERO-Goal.

(b) Explained Variance

0.9

0.9

SuccessRate

SuccessRate

0.8

0.8

0.7

0.7

0.6

Fixed Noise

Fixed Noise

0.6

Learnable Noise

Learnable Noise

0.5

0.5

0 100 200 300 400

0 100 200 300 400

Step

Step

(b) Eval Figure 8. Ablation on the injection strategy within Flow-SDE of π0 on the LIBERO-Long.

(a) Train

Structure. We investigate a four-layer MLP versus a onelayer MLP, which mirrors the action-projection structure in the action expert. Results in Fig. 6 indicate that the fourlayer MLP leads to a more accurate value approximation, resulting in enhanced performance and training stability.

- 5.3.2. FLOW POLICY MDP

With the same fixed noise injection strategy, we evaluate the one-layer MDP of Flow-Noise with the two-layer MDP of Flow-SDE on the LIBERO-Goal, as shown in Fig. 7.

We observe that the one-layer formulation converges most rapidly, but the final success rates remain consistent across all three formulations. In terms of computational efficiency, the hybrid two-layer paradigm achieves a 2× speedup over the standard approach due to its shorter effective MDP chain. Notably, the one-layer MDP yields no substantial wall-clock

time advantage over the standard two-layer model, stemming from the requirement to recalculate full denoising trajectories for log likelihood estimation.

5.3.3. STOCHASTICITY INJECTION

We compare fixed and learnable noise injection strategies using the Flow-SDE MDP formulation on the LIBEROLong suite. To ensure a controlled comparison, we set the entropy bonus for the learnable noise to zero, aligning it with the fixed noise approach.

Specifically, we set the fixed noise to 0.5, and lower and upper bounds for the learnable noise log-variance to 0.08 and 0.16. As depicted in Fig. 8, two noise strategies exhibit similar train performance at step 0, which indicates comparable noise magnitudes. Furthermore, the converged performance affirms the efficiency of both injection methods.

Table 2. Ablation study of hyperparameters for Flow-SDE on the LIBERO-Spatial. Train refers to policy performance during the stochastic rollout phase, whereas Eval refers to performance during the deterministic evaluation phase.

Hyperparameters Noise Level Denoise Step Action Chunk

Models Stage

0.2 0.5 0.8 1 2 4 8 5 10 20 SFT

Train 62.3 56.0 46.6 9.4 28.3 56.1 62.6 56.0 60.7 70.3 Eval 65.2 65.2 65.2 63.8 64.9 65.2 63.2 65.2 70.5 72.6

Train 59.5 93.5 95.3 73.8 90.8 93.5 84.3 93.5 93.3 87.5 Eval 73.1 94.5 98.1 88.5 97.0 94.5 86.7 94.5 95.5 89.2

RL

1.0

1.0

0.25

ClippedFraction

0.8

0.8

0.20

SuccessRate

SuccessRate

0.6

0.6

0.15

0.4

0.4

0.10

noise=0.2 noise=0.5 noise=0.8

noise=0.2 noise=0.5 noise=0.8

noise=0.2 noise=0.5 noise=0.8

0.2

0.2

0.05

0.0

0.0

0.00

0 100

0 100

0 100

Step

Step

Step

(a) Train

###### (c) Clipped Fraction Figure 9. Ablation on the noise level a, conducted with the Flow-SDE π0 on the LIBERO-Spatial.

(b) Eval

1.0

1.0

0.9

0.8

SuccessRate

SuccessRate

0.8

0.6

0.7

- Step 1

- Step 2

- Step 1

- Step 2

0.4

0.6

Step 4 Step 8

Step 4 Step 8

0.2

0.5

0.0

0.4

0 100

0 100

Step

Step

###### (b) Eval Figure 10. Ablation on the denoise step, conducted with the Flow-SDE π0 on the LIBERO-Spatial.

(a) Train

1.0

1.0

ExplainedVariance

0.8

0.9

SuccessRate

0.6

0.8

0.4

chunk=5

chunk=5

0.7

chunk=20 chunk=10

chunk=20 chunk=10

0.2

0.6

0.0

0 100

0 100

Step

Step

(b) Explained Variance Figure 11. Ablation on the chunk size, conducted with the Flow-SDE π0 on the LIBERO-Spatial.

(a) Eval

#### 5.4. Hyper-Parameters

Building on the Flow-SDE with π0 model, we investigate the influence of the noise level, denoise step, and action chunk on the LIBERO-Spatial benchmark. We denote the train stage as the phase where the policy generates stochas-

tic actions for exploration, whereas the evaluation stage involves generating deterministic actions. The train and eval success rates for the SFT baseline and the RL fine-tuned model after 100 training steps are presented in Tab. 2.

Noise Level. The noise level a in the Flow-SDE is defined in

Eq. (8), which governs the noise injection magnitude during the denoising process. As shown in Tab. 2, the evaluation performance of the SFT baseline remains identical across all noise levels due to its reliance on deterministic ODE sampling. Conversely, its training performance exhibits a clear degradation as noise increases. This is intuitive, as higher noise levels can distort the flow trajectory, leading to an inaccurate estimation of the marginal action distribution.

Extending this analysis to the RL fine-tuning stage highlights a critical trade-off: while lower noise levels mitigate exploration-induced performance degradation, they simultaneously constrain the capacity for RL refinement. This trade-off is empirically supported by Fig. 9, which shows that training with minimal noise (a = 0.2) leads to instability, characterized by a significantly higher clip fraction. We attribute this instability to the substantially larger gradient magnitudes associated with low-noise regimes.

Denoise Step. The denoise step K defines the number of discretization steps for action generation and is critical for controlling the fidelity of the ODE-to-SDE transition in Eq. (8). In Tab. 2, we observe that while all configurations start with similar eval performance, the train success rate plummets at K = 1, indicating a significant ODE-to-SDE discretization error.

However, a larger K is not unequivocally optimal. As illustrated in Fig. 10, increasing K presents a distinct trade-off: while it enhances rollout performance, it simultaneously increases training complexity and computational overhead due to the extended sequence of denoising steps.

Action chunk. The action chunk refers to the number of consecutive actions the policy executes within a single observation. We ablate the action chunk size across 5, 10, and 20, with results visualized in Fig. 11.

Although a larger chunk size yields marginal performance gains, it inherently reduces the frequency of policyenvironment interactions and obscures precise reward credit assignment. These constraints lead to less reliable advantage estimation, as evidenced by the diminished explained variance. Consequently, while an increased chunk size may offer a superior SFT baseline, it paradoxically limits the ceiling for subsequent RL-driven refinement.

### 6. Conclusion

We introduce πRL, a framework that enables flow-based VLAs, π0 and π0.5, to be fine-tuned with online RL algorithms. We tackle the fundamental challenge of intractable log-likelihoods in flow matching with Flow-Noise and FlowSDE solutions. Our extensive experiments on the challenging benchmarks demonstrated that πRL achieves significant performance improvements over SFT baselines.

Limitation. Due to the low sample efficiency of online RL, our framework currently relies on sim-to-real deployment. We aim to develop more efficient algorithms to enable realworld RL training in the future.

### References

Beyer, L., Steiner, A., Pinto, A. S., Kolesnikov, A., Wang, X., Salz, D., Neumann, M., Alabdulmohsin, I., Tschannen, M., Bugliarello, E., et al. Paligemma: A versatile 3b vlm for transfer. arXiv preprint arXiv:2407.07726, 2024.

Bjorck, J., Castañeda, F., Cherniadev, N., Da, X., Ding, R., Fan, L., Fang, Y., Fox, D., Hu, F., Huang, S., et al. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025.

Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.

Chen, G., Li, Z., Wang, S., Jiang, J., Liu, Y., Lu, L., Huang, D.-A., Byeon, W., Le, M., Rintamaki, T., et al. Eagle 2.5: Boosting long-context post-training for frontier visionlanguage models. arXiv preprint arXiv:2504.15271, 2025.

Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11): 1684–1704, 2025.

Din, M. U., Akram, W., Saoud, L. S., Rosell, J., and Hussain, I. Vision language action models in robotic manipulation: A systematic review. arXiv preprint arXiv:2507.10672, 2025.

Fan, H., Dai, H., Zhang, J., Li, J., Yan, Q., Zhao, Y., Gao, M., Wu, J., Tang, H., and Dong, H. Twinaligner: Visual-dynamic alignment empowers physics-aware real2sim2real for robotic manipulation. arXiv preprint arXiv:2512.19390, 2025.

Fei, S., Wang, S., Shi, J., Dai, Z., Cai, J., Qian, P., Ji, L., He, X., Zhang, S., Fei, Z., et al. Libero-plus: In-depth robust-

ness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

Firoozi, R., Tucker, J., Tian, S., Majumdar, A., Sun, J., Liu, W., Zhu, Y., Song, S., Kapoor, A., Hausman, K., et al. Foundation models in robotics: Applications, challenges, and the future. The International Journal of Robotics Research, 44(5):701–739, 2025.

Guo, R., Lin, X., Liu, M., Gu, J., and Su, H. Mplib: a lightweight motion planning library. https://github.com/haosulab/MPlib, 2025. URL https://motion-planning-lib.

readthedocs.io/latest/.

He, X., Fu, S., Zhao, Y., Li, W., Yang, J., Yin, D., Rao, F., and Zhang, B. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Hutchinson, M. F. A stochastic estimator of the trace of the influence matrix for laplacian smoothing splines. Communications in Statistics-Simulation and Computation, 18(3):1059–1076, 1989.

Intelligence, P., Black, K., Brown, N., Darpinian, J., Dhabalia, K., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., et al. π0.5: a vision-language-action model with openworld generalization. arXiv preprint arXiv:2504.16054, 2025.

Jiang, G., Chang, H., Qiu, R.-Z., Liang, Y., Ji, M., Zhu, J., Dong, Z., Zou, X., and Wang, X. Gsworld: Closed-loop photo-realistic simulation suite for robotic manipulation. arXiv preprint arXiv:2510.20813, 2025.

Kerbl, B., Kopanas, G., Leimkühler, T., and Drettakis, G. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., Karamcheti, S., Nasiriany, S., Srirama, M. K., Chen, L. Y., Ellis, K., et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

Kim, M. J., Finn, C., and Liang, P. Fine-tuning visionlanguage-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

Li, H., Zuo, Y., Yu, J., Zhang, Y., Yang, Z., Zhang, K., Zhu, X., Zhang, Y., Chen, T., Cui, G., et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025a.

Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., and Zhong, Z. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025b.

- Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R., Fu, C., Lunawat, I., Sieh, I., Kirmani, S., et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024.
- Li, Y., Wang, Y., Zhu, Y., Zhao, Z., Lu, M., She, Q., and Zhang, S. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025c.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, B., Zhu, Y., Gao, C., Feng, Y., Liu, Q., Zhu, Y., and Stone, P. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Liu, J., Gao, F., Wei, B., Chen, X., Liao, Q., Wu, Y., Yu, C., and Wang, Y. What can rl bring to vla generalization? an empirical study. arXiv preprint arXiv:2505.19789, 2025a.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

McAllister, D., Ge, S., Yi, B., Kim, C. M., Weber, E., Choi, H., Feng, H., and Kanazawa, A. Flow matching policy gradients. arXiv preprint arXiv:2507.21053, 2025.

McLean, R., Chatzaroulas, E., McCutcheon, L., Röder, F., Yu, T., He, Z., Zentner, K., Julian, R., Terry, J., Woungang, I., et al. Meta-world+: An improved, standardized, rl benchmark. arXiv preprint arXiv:2505.11289, 2025.

Mees, O., Hermann, L., Rosete-Beas, E., and Burgard, W. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 7(3):7327–7334, 2022.

O’Neill, A., Rehman, A., Maddukuri, A., Gupta, A., Padalkar, A., Lee, A., Pooley, A., Gupta, A., Mandlekar,

- A., Jain, A., et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.

Peebles, W. and Xie, S. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Pertsch, K., Stachowicz, K., Ichter, B., Driess, D., Nair, S., Vuong, Q., Mees, O., Finn, C., and Levine, S. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.

Ren, A. Z., Lidard, J., Ankile, L. L., Simeonov, A., Agrawal, P., Majumdar, A., Burchfiel, B., Dai, H., and Simchowitz, M. Diffusion policy policy optimization. arXiv preprint arXiv:2409.00588, 2024.

Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shukor, M., Aubakirova, D., Capuano, F., Kooijmans, P., Palma, S., Zouitine, A., Aractingi, M., Pascal, C., Russi, M., Marafioti, A., et al. Smolvla: A vision-languageaction model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844, 2025.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Tan, S., Dou, K., Zhao, Y., and Krähenbühl, P. Interactive post-training for vision-language-action models. arXiv preprint arXiv:2505.17016, 2025.

Tao, S., Xiang, F., Shukla, A., Qin, Y., Hinrichsen, X., Yuan, X., Bao, C., Lin, X., Liu, Y., Chan, T.-k., et al. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. arXiv preprint arXiv:2410.00425, 2024.

Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wang, F. and Yu, Z. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Wen, J., Zhu, Y., Li, J., Zhu, M., Tang, Z., Wu, K., Xu, Z., Liu, N., Cheng, R., Shen, C., et al. Tinyvla: Towards fast, data-efficient vision-language-action models for robotic manipulation. IEEE Robotics and Automation Letters, 2025.

Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Yu, C., Wang, Y., Guo, Z., Lin, H., Xu, S., Zang, H., Zhang, Q., Wu, Y., Zhu, C., Hu, J., et al. Rlinf: Flexible and efficient large-scale reinforcement learning via macro-to-micro flow transformation. arXiv preprint

- arXiv:2509.15965, 2025.

Zang, H., Wei, M., Xu, S., Wu, Y., Guo, Z., Wang, Y., Lin, H., Shi, L., Xie, Y., Xu, Z., et al. Rlinf-vla: A unified and efficient framework for vla+ rl training. arXiv preprint

- arXiv:2510.06710, 2025.

Zhang, T., Yu, C., Su, S., and Wang, Y. Reinflow: Finetuning flow matching policy with online reinforcement learning. arXiv preprint arXiv:2505.22094, 2025.

### A. Appendix

This appendix provides additional technical details and experimental results for πRL. The content is organized as:

- • Appendix B: Experimental Setup.
- • Appendix C: In-Distribution RL Training.
- • Appendix D: Out-of-Distribution RL Evaluation.
- • Appendix E: Case Studies: Single-Task RL Training.
- • Appendix F: Ablation Details.
- • Appendix G: Insights from Large-Scale RL Training.
- • Appendix H: RL for GR00T N1.5.
- • Appendix I: Limitations and Future Work.
- • Appendix J: Training Hyperparameters.

#### B.2. Implementation Details.

Given that pre-trained models often struggle to generalize to task-specific benchmarks, we initiate our process with SFT on expert demonstrations. For the SFT stage, we fine-tune the entire 3.3B model following the official setting. In the subsequent RL stage, we freeze the VLM parameters and exclusively fine-tune the 300M action expert model, driven by GPU memory efficiency and the findings from RL4VLA that RL contributes more significantly to action generalization. We build the whole framework upon the RLinf (Yu et al., 2025) codebase, where we adopt a shared, co-located GPU allocation strategy that places the environment, rollout model, and actor model on the same GPU and executes them serially.

For the model configurations, we adhere to the official setting provided by openpi (Black et al., 2024; Intelligence et al., 2025). In these settings, π0 utilizes image, language, and proprioceptive states as input, whereas π0.5 notably omits state information2. Our experiments are conducted on 8 NVIDIA H100 80GB GPUs, and detailed training hyperparameters are available in the Appendix Sec. J.

### B. Experimental Setup

#### B.1. Benchmarks.

To rigorously assess the performance and generalization of our framework, we conduct evaluations across four diverse benchmarks, with diverse emphases on robotic capability:

- • LIBERO focuses on compositional variations and knowledge transfer in tasks. By evaluating across four task suites: Spatial, Object, Goal, and Long, it probes the model’s ability to transfer base skills to variations in object arrangements and long-horizon tasks.
- • ManiSkill emphasizes perceptual and execution robustness under massive environmental diversity. We follow the setup of RL4VLA (Liu et al., 2025a) and built a benchmark with 4,352 unique pick-and-place combinations, which challenges the policy to maintain physical interactions across a vast distribution of objects and receptacles.
- • CALVIN evaluates long-horizon sequential reasoning and vision-language grounding. It evaluates the model’s capacity to execute chains of five random subtasks in a persistent environment, requiring accurate alignment with linguistic instructions.
- • MetaWorld measures skill breadth and multi-task versatility. It requires a single policy to master 50 semantically distinct manipulation primitives, ranging from simple reaching to complex tool usage.

### C. In-Distribution RL Training

In this section, we evaluate the effectiveness of our framework across various benchmarks to demonstrate its robustness and superior performance in ID settings.

#### C.1. LIBERO Benchmark

SFT Procedure. The LIBERO benchmark comprises four task suites, each consisting of 10 distinct subtasks. To facilitate few-shot SFT on LIBERO, a minimum of 40 expert demonstration trajectories is necessary to ensure a positive success rate for each subtask across four task suites, thereby guaranteeing a positive optimization signal for the subsequent RL phase.

We perform few-shot SFT following the official training configurations. For the π0 model, we fine-tune on a subset of 58 trajectories sampled from the 1,692 total demonstrations3, which serves as the initial checkpoint for subsequent RL training on the Spatial, Object, and Goal task suites. For the Long task suite, a larger pool of 208 trajectories is employed to address its more challenging, long-horizon nature. In contrast, for the π0.5 model, benefiting from a superior pre-trained checkpoint and training configurations, we leverage only 40 trajectories to provide a unified few-shot SFT checkpoint across all task suites.

RL Procedure. In RL, the VLA model receives a multimodal input state comprising: an agent-view and a wrist-

- 2https://github.com/Physical-Intelligence/openpi/issues/687
- 3https://huggingface.co/datasets/physical-intelligence/libero

Table 3. Evaluation results on the LIBERO benchmark, evaluated based on the success rate (%).

LIBERO

Model

Spatial Object Goal Long Avg. ∆ Avg. # Full Dataset SFT

Octo 78.9 85.7 84.6 51.1 75.1 OpenVLA 84.7 88.4 79.2 53.7 76.5 πfast 96.4 96.8 88.6 60.2 85.5 OpenVLA-OFT 91.6 95.3 90.6 86.5 91.0 π0 96.8 98.8 95.8 85.2 94.2 π0.5 98.8 98.2 98.0 92.4 96.9 # Few-shot SFT + RL

SFT 65.3 64.4 49.8 51.2 57.6 Flow-SDE 98.4 99.4 96.2 90.2 96.1 +38.5 Flow-Noise 99.0 99.2 98.2 93.8 97.6 +40.0

π0

# Few-shot SFT + RL

SFT 84.6 95.4 84.6 43.9 77.1 Flow-SDE 99.6 100 98.8 93.0 97.9 +20.8 Flow-Noise 99.6 100 99.6 94.0 98.3 +21.2

π0.5

Table 4. Evaluation results on the ManiSkill benchmark, with more specific OOD results depicted in Tab. 7.

IND OOD Avg. ∆ Vision Semantic Execution Avg. ∆

Model

SFT 38.4 — 32.6 8.4 13.2 18.1 Flow-SDE 78.8 +40.4 61.1 25.4 31.5 39.3 +21.2 Flow-Noise 77.8 +39.4 63.4 23.1 24.2 36.9 +18.8

π0

SFT 40.1 — 40.2 16.6 22.4 26.4 Flow-SDE 90.9 +50.8 68.0 34.5 45.4 49.3 +22.9 Flow-Noise 89.7 +49.6 69.9 35.5 54.9 53.4 +27.0

π0.5

view, natural language guidance, the robot end effector pose, and the gripper state. The model outputs an action to interact with the LIBERO environment, which provides a binary reward of 1 for successful task completion and 0 otherwise.

Experiments. We benchmark the performance of πRL, which fine-tunes the few-shot SFT π0 and π0.5 models with Flow-Noise and Flow-SDE, against several state-of-the-art VLAs trained on the entire LIBERO dataset, including Octo, OpenVLA, OpenVLA-OFT, πfast (Pertsch et al., 2025), π0, and π0.5. We conduct experiments on four LIBERO task suites and report performance as the success rate across all 500 initial states (10 sub-tasks × 50 states each).

Analysis. For the few-shot π0 model, the SFT baseline performs poorly, with an average success rate of only 57.6%, indicating that the model struggles with limited demonstration data. πRL substantially boosts performance, with Flow-SDE and Flow-Noise reaching 96.1% and 97.6%, and surpassing the full-dataset π0 SFT baseline of 94.2%.

While the π0.5 few-shot SFT baseline achieves a decent average performance of 77.1%, it struggles with the challenging Long task suite, scoring only 43.9%. Our proposed πRL rectifies this deficiency, boosting the Long task success rate from 43.9% to 94.0%, constituting a 50.1% improvement. Notably, despite using only a single trajectory for SFT, πRL reaches 98.3% final performance, surpassing the 96.9% full-dataset SFT model.

#### C.2. ManiSkill Benchmark

SFT Procedure. In the ManiSkill benchmark, the policy is required to pick from 16 object types and place them onto 17 receptacles across 16 unique table scenes, yielding 4,352 unique task combinations. Given the high complexity of this setting, the SFT dataset consists of 16,384 episodes synthesized using the MPLib motion planning suite (Guo et al., 2025). To reinforce the concept of motion completion, 15 additional frames are appended to the end of each trajectory.

Table 5. Evaluation results on the CALVIN benchmark (Scene D), reporting the average completed subtasks and success rates for task sequences of length 1 to 5.

CALVIN-D Len-1 Len-2 Len-3 Len-4 Len-5 Avg. ∆ Avg.

Methods

SFT 94.7 84.9 74.3 65.2 57.5 3.766 Flow-SDE 96.4 88.0 77.5 70.8 61.7 3.944 +0.178 Flow-Noise 96.9 88.8 78.0 68.3 59.9 3.919 +0.153

π0

SFT 92.7 84.3 76.7 68.8 61.3 3.838 Flow-SDE 99.7 98.2 95.8 91.0 87.0 4.717 +0.879 Flow-Noise 99.6 97.6 93.9 89.6 84.5 4.652 +0.814

π0.5

Table 6. Evaluation results on the MetaWorld MT50 benchmark.

MetaWorld-MT50 Easy Medium Hard Very Hard Avg. ∆ Avg.

Methods

Diffusion Policy 23.1 10.7 1.9 6.1 10.5 TinyVLA 77.6 21.5 11.4 15.8 31.6 SmolVLA 87.1 51.8 70.0 64.0 68.2 —

SFT 77.9 51.8 53.3 20.0 50.8 Flow-SDE 92.1 74.6 61.7 84.0 78.1 +27.3 Flow-Noise 91.1 81.8 78.3 92.0 85.8 +35.0

π0

SFT 68.2 37.3 41.7 28.0 43.8 Flow-SDE 86.4 55.5 75.0 66.0 70.7 +26.9 Flow-Noise 86.8 58.1 63.3 56.0 66.1 +22.3

π0.5

RL Procedure. In RL, the VLA model receives a thirdperson RGB image, a concise language instruction, and the current joint proprioception. The environment provides a structured reward signal: 1.0 for correct object placement and an auxiliary 0.1 reward for successful gripper-object attachment, intended to encourage stable manipulation and mitigate undesired behaviors such as impulsive throwing.

Experiments. Following the RL4VLA experimental protocol, we conduct RL training on a comprehensive set of 4,352 task combinations and record the performance as the aggregate success rate across these tasks.

Analysis. As detailed in Tab. 4, πRL significantly boosts performance in the training environment. Specifically, the success rate of π0 increases from 38.4% to 77.8%, while π0.5 improves from 40.1% to 90.9%. These gains underscore the efficacy of RL in complex settings.

#### C.3. CALVIN Benchmark

SFT Procedure. We conduct SFT on the CALVIN ABC dataset4, which comprises approximately 24 hours of un-

4https://huggingface.co/datasets/InternRobotics/InternDataCalvin_ABC

structured "play data" across three distinct environments (A, B, and C). This dataset includes over 20,000 languagelabeled trajectories covering 34 unique manipulation tasks.

RL Procedure. Each episode consists of a sequence of five randomly sampled subtasks to be completed in succession without environment resets. The reward signal is defined at the subtask level, where the model receives a sparse binary reward of 1.0 for each successfully executed subtask and 0.0 otherwise.

Experiments. We evaluate the performance of πRL in Scene D over 1,000 episodes. Following the standard CALVIN evaluation protocol, we report two key metrics: (1) the success rate for task sequences of increasing lengths, namely Len-1 to Len-5. (2) the average number of completed subtasks, denoted as Avg. per episode.

Analysis. As detailed in Tab. 5, πRL yields substantial performance gains in long-horizon sequential execution, particularly with the π0.5 variant. The SFT models inherently struggle with compounding errors across sequential tasks, with π0.5 only achieving a 61.3% success rate on Len-5 sequences. RL effectively mitigates this issue with the average completed sub-tasks of Flow-SDE increasing from 3.838 to 4.717, and its Len-5 success rate surges to 87.0%.

Table 7. Specific generalization evaluation results in the ManiSkill OOD setting.

π0-RL π0-RL

π0.5-RL π0.5-RL

Environment Variation-Version-Type π0-SFT

π0.5-SFT

Flow-SDE Flow-Noise Flow-SDE Flow-Noise In distribution Main-v3-train 38.4 78.8 77.8 40.1 90.9 89.7

Instruct-v1-test 30.1 64.6 66.5 46.6 77.0 85.7 VisionImage-v1-test 38.3 68.8 71.7 46.2 78.8 83.1 VisionTexture03-v1-test 35.1 66.0 66.8 36.7 69.6 75.0 VisionTexture05-v1-test 31.0 55.8 60.5 32.7 58.0 62.2 VisionWhole03-v1-test 35.4 62.4 69.0 40.1 69.6 71.6 VisionWhole05-v1-test 28.5 49.0 53.9 30.7 55.0 57.0

Visual-Language Variations

MultiCarrot-v1-test 7.8 28.2 23.0 16.7 36.8 38.2 MultiCarrot-v1-train 12.5 36.5 31.8 28.2 49.5 50.1 MultiPlate-v1-test 5.0 16.4 18.3 11.8 29.4 28.3 MultiPlate-v1-train 7.3 20.5 19.6 9.7 22.3 25.4

Semantic Reasoning (object/receptacle confounders)

PositionChangeTo-v1-test 9.6 17.4 10.9 13.5 36.2 54.7 Position-v1-test 16.9 45.6 37.5 31.2 54.5 55.0

Action Execution

Notably, the performance gap between SFT and RL widens significantly as the sequence length increases. For π0.5, while Flow-SDE shows a modest 7.0% improvement over SFT in Len-1 tasks, the gap expands to an impressive 25.7% in the most challenging Len-5 sequences.

#### C.4. MetaWorld Benchmark

SFT Procedure. We perform SFT on the π0 and π0.5 models using the official dataset5, which consists of 2500 trajectories across 50 different manipulation tasks.

RL Procedure. During the RL procedure, the VLA model processes a multi-modal input comprising a RGB agentview image, language guidance, the robot’s end-effector position, and its gripper state. Based on this input, the model outputs an action to interact with the environment, which in turn provides a sparse reward: 1 for successful task completion and 0 otherwise.

Experiments. We benchmark the performance of πRL against Diffusion Policy (Chi et al., 2025), TinyVLA (Wen et al., 2025), and SmolVLA (Shukor et al., 2025). For the performance evaluation, we follow the setup from SmolVLA, i.e., classifying 50 tasks into easy, medium, hard, and very hard four categories according to their difficulties.

Analysis. As detailed in Tab. 6, RL fine-tuning substantially boosts performance. The π0 and π0.5 models achieve average success rates of 85.8% and 70.7%, respectively. This marks a significant improvement over their SFT-only counterparts and surpasses the SmolVLA baseline of 68.2%, confirming that RL can effectively enhance model capabilities across a diverse range of manipulation task types.

5https://huggingface.co/datasets/lerobot/metaworld_mt50

### D. Out-of-Distribution RL Evaluation

While previous experiments demonstrate significant RLdriven improvements in the ID domain, this section evaluates OOD generalization. As LIBERO lacks a dedicated interface for OOD testing, we utilize the ManiSkill, CALVIN, and MetaWorld benchmarks to investigate whether the RLdriven improvements represent genuine skill acquisition that scales to novel settings, or merely reflect the exploitation of environment-specific biases.

#### D.1. ManiSkill

Setup. Following RL4VLA, we evaluate the model’s generalization across three challenging OOD scenarios: (1) Vision, challenging the model with novel backgrounds and textures; (2) Semantics, probing comprehension with unseen objects, varied instructions, and confounding elements like extra objects or receptacles; (3) Execution, assessing robustness against varied initial states, unseen robot poses, and dynamic disturbances.

Results. In the OOD scenarios detailed in Tabs. 4 and 7, we observe that the π0-SFT model demonstrates strong generalization for visual information. This can be attributed to the robust foundation of its VLM, which allows it to handle visual disturbances better.

However, the semantic performance of π0 drops dramatically. This degradation is less pronounced when switching to the π0.5 baseline, a benefit likely stemming from the knowledge generalization of the pre-trained π0.5 model. Regarding action execution, π0 exhibits a larger performance drop than π0.5. We hypothesize that this discrepancy arises from the inclusion of joint angle states as input in π0, leading to severe overfitting in the control task. In contrast, π0.5

Table 8. Evaluation results on the SIMPLER benchmark for π0 and π0.5 with Flow-Noise method.

SIMPLER Carrot Eggplant Spoon Cube Avg.

Model

SFT 82.7 87.5 61.7 37.1 67.2 +RL 95.7 96.7 91.6 63.0 86.7 ∆ +13.0 +9.2 +29.9 +25.9 +19.5

π0

SFT 70.6 91.9 43.5 31.0 59.2 +RL 82.0 98.2 82.8 53.3 79.1 ∆ +11.4 +6.3 +39.3 +22.3 +19.9

π0.5

[Figure 70]

Figure 12. Real-world deployment of an RL refined policy performing a pick and place task.

omits these inputs, thereby avoiding the same degree of performance degradation.

As for the RL training, although the performance improvements in OOD scenarios are lower than those in IND settings, the proportional improvements achieved are notably comparable. As indicated in Tab. 4, for the π0.5 model, Flow-SDE enhances the IND success rate by 126.7%, while the OOD similarly increases by 102.3%. This consistency in relative gains indicates that RL-driven optimization promotes the acquisition of generalized action representations rather than merely overfitting the training environment, thus preserving efficacy under distribution shifts.

Nevertheless, we observe a performance gap in relative improvement between the Vision OOD tasks and the IND domain. Specifically, for π0.5, the 73.9% gain in visual generalization trails the 126.7% increase observed in the training environment. This discrepancy likely stems from freezing the VLM backbone during the RL stage for computational efficiency, which restricts the model’s ability to adapt its visual grounding features to novel textures and backgrounds.

#### D.2. CALVIN

Setup. We evaluate environmental and visual OOD robustness based on the ABC → D protocol in CALVIN. Under this setting, the model is trained on ABC environments and evaluated in a zero-shot manner on Scene D. Scene D introduces significant distribution shifts in terms of visual textures, lighting conditions, and spatial layouts, effectively assessing the agent’s ability to transfer skills to an unfamiliar physical environment.

Results. Under identical D→ D training settings, the RL finetuned policy in the ABC environment reaches a 79.1% success rate in the OOD scene D, improving over the 61.3% SFT baseline as shown in Fig. 5. Aligned with the findings in ManiSkill, this suggests that ID gains can be transferred to OOD settings characterized by visual variations.

#### D.3. MetaWorld

Setup. We utilize the ML45 benchmark from MetaWorld to evaluate the task-level generalization. This setup consists of 50 distinct robotic manipulation tasks: the agent is trained on 45 base tasks and subsequently evaluated on 5 heldout, unseen tasks, which require the model to generalize its learned manipulation primitives to entirely novel task objectives and workspace configurations.

Results. As evidenced in the Fig. 5, while success rates show consistent gains within the ID domain, OOD performance is characterized by persistent oscillation throughout the training process. This instability indicates that RL, in its current form, struggles to foster stable cross-category generalization.

Nevertheless, the model retains the OOD skills learned during the SFT phase throughout the RL training process. This highlights a significant advantage over standard SFT, which often causes the model to overfit on expert demonstrations and lose its broader capabilities (Li et al., 2025a). Unlike standard SFT, RL enables ID performance gains while preserving the general knowledge, which indicates that RL provides a more balanced optimization that maintains OOD robustness without catastrophic forgetting.

Table 9. Comparison of the PPO and GRPO with Flow-SDE on the LIBERO.

LIBERO Spatial Object Goal Long Avg. ∆ Avg.

Model

SFT 65.3 64.4 49.8 51.2 57.6 —

+GRPO 97.8 97.8 83.2 81.4 90.0 +32.4 +PPO 98.4 99.4 96.2 90.2 96.0 +38.4

π0

SFT 84.6 95.4 84.6 43.9 77.1 —

+GRPO 97.4 99.8 91.2 77.6 91.5 +14.4 +PPO 99.6 100 98.8 93.0 97.9 +20.8

π0.5

#### D.4. Summary

In conclusion, our OOD evaluation demonstrates that RL enhances performance for similar tasks but fails to generalize effectively to novel task objectives.

Specifically, RL training effectively enhances robustness against low-level variations such as the visual and execution shifts observed in ManiSkill and CALVIN. This indicates that the model acquires generalized action representations rather than merely overfitting to the training environment.

Regarding high-level generalization on MetaWorld, the model successfully retains the OOD skills inherited from the SFT phase, demonstrating that RL avoids the catastrophic forgetting and overfitting typical of standard imitation learning. However, transferring its performance gains to entirely novel task objectives remains a significant challenge.

### E. Case Studies: Single-Task RL Training

While the preceding experiments focused on performance across multi-task benchmarks, this section investigates single-task scenarios where the VLA is trained to master a specific task within a relatively static environment. Specifically, we evaluate our approach on the SIMPLER (Li et al.,

- 2024) and a Real2Sim2Real environment.

#### E.1. SIMPLER

Setup. In SIMPLER, the experimental setup comprises an 8-DoF WidowX-250S arm evaluated on four standard tasks: (1) Spoon: placing a spoon on a cloth. (2) Carrot: placing a carrot on a plate. (3) Eggplant: placing an eggplant in a basket. (4) Cube: stacking a cube. For the SFT stage, we employ a curated dataset in which each task is trained with 144 demonstration episodes.

Analysis. As detailed in Tab. 8, πRL increases the average success rate of the π0 model from 67.2% to 86.7%, with three tasks (carrot, eggplant, and spoon) exceeding 90% success.

#### E.2. Real2Sim2Real

While the SIMPLER benchmark demonstrates predictive correlation between simulation and real-world performance, a pronounced visual domain shift remains to be solved. To this end, we leverage recent Real2Sim2Real methodologies (Fan et al., 2025; Jiang et al., 2025) to construct a highfidelity simulation environment with ManiSkill for rigid body dynamics and Gaussian Splatting (Kerbl et al., 2023) for photorealistic rendering.

Setup. Our hardware platform comprises a Franka Panda robotic arm and an Intel RealSense D435 camera serving as the primary visual sensor. We perform manual calibration by aligning simulated viewpoints with real-world camera perspectives to synchronize their extrinsic matrices. As illustrated in Fig. 1, the photorealistic textures and color profiles in our simulator closely mirror the physical environment, effectively minimizing the visual domain shift from simulation to reality.

Results. Following the experimental protocol established in the ManiSkill benchmarks, we initially collect 20 expert trajectories via a motion planner for few-shot SFT, which is subsequently optimized through RL over 100 training iterations. We deploy the RL fine-tuned policies in the real world in a zero-shot manner. Empirical results indicate that while the SFT baseline fails to complete the task, the RLtuned policy achieves a 40% success rate. A representative successful episode is visualized in Fig. 12.

### F. Ablation Details F.1. RL algorithms

Given the significant performance gains from PPO on the LIBERO benchmark, we also investigated the effectiveness of GRPO (Shao et al., 2024), another widely used policy gradient method applied in VLA+RL training (Li et al., 2025a). We compare the performance of PPO and GRPO on both the π0 and π0.5 models, with results denoted in Tab. 9. Conclusion. To sum up, our findings highlight a critical

0.9

VLM Frozen

0.20

- VLM LoRA-I

- VLM LoRA-II

0.8

KLDivergence

SuccessRate

0.15

0.7

0.10

0.6

VLM Frozen

- VLM LoRA-I

- VLM LoRA-II

0.05

0.5

0.00

0.4

0 50 100 150

0 50 100 150

Step

Step

###### (b) KL Divergence Figure 13. Ablation study on the effectiveness of VLM during RL.

(a) Eval

0.9

No Scheduler

0.03

Cosine Scheduler

KLDivergence

0.8

SuccessRate

0.02

0.7

0.6

0.01

No Scheduler

0.5

Cosine Scheduler

0.00

0.4

0 100 200 300

0 100 200 300

Step

Step

(a) Eval

(b) KL Divergence Figure 14. Ablation study on the learning rate scheduler.

trade-off: parameters tailored for rollout success may adversely impact training stability, ultimately constraining the performance ceiling of RL. Therefore, careful parameter tuning is required to achieve a synergy between high-quality rollouts and stable policy convergence.

#### F.2. VLM Fine-tuning Analysis

In our previous experiments, the VLM is frozen, and the optimization is confined exclusively to the action expert during RL. In this subsection, we aim to investigate the role of the VLM during RL. Specifically, we employ LowRank Adaptation (LoRA) (Hu et al., 2022) for the VLM, facilitating its joint optimization with the action expert. We set the LoRA rank to r = 32 and the scaling parameter to α = 32, while the action expert remains fully trainable.

We conduct experiments with the π0 model with Flow-SDE on the LIBERO-Long benchmark, comparing three distinct configurations:

- • VLM Frozen: 5e−6 learning rate, 4 updates/epoch.
- • VLM LoRA-I: 5e−6 learning rate, 4 updates/epoch.
- • VLM LoRA-II: 1e−6 learning rate, 2 updates/epoch.

As presented in Fig. 13, the VLM LoRA-II configuration achieves a learning trajectory comparable to the VLM frozen baseline. This empirical observation yields two critical inferences: (1) The benefit of fine-tuning the VLM on the

LIBERO benchmark is not evident. We conjecture the limited performance gain owing to the limited scene variability within LIBERO, for which the pretrained VLM representations are already sufficiently robust. (2) Fine-tuning VLM together with the action expert requires a more conservative optimization configuration for training stability.

### G. Insights from Large-Scale RL Training

In this section, we elaborate on some empirical insights we gained during RL training.

Hyperparameters. According to the hyperparameters ablation detailed in Sec. 5.4, the performance disparity between the train and eval performance of the initial SFT checkpoint warrants close attention. If this disparity is significant, we recommend either reducing the noise magnitude or increasing the number of denoising steps to mitigate the performance loss when shifting from deterministic to stochastic execution. Furthermore, as previously established, lower noise levels yield larger gradients, requiring a smaller learning rate to maintain training stability.

We also observed that when train performance improves steadily while eval performance oscillates, increasing the number of denoising steps can help alleviate this, benefiting from reduced divergence in the action distributions between the deterministic and stochastic action generation processes. Regarding the action chunk, we empirically found that longhorizon tasks benefit from larger chunk sizes. For instance, we set the chunk size to 10 for LIBERO-Long and 5 for the

0.8

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | |0<br><br>| |
| | | | |0.5<br><br>| |
| | | | | | |

0.9

0.7

ExplainedVariance

0.8

SuccessRate

0.6

0.7

0.5

0.6

0.4

0.5

0.3

0.4

0.2

0

0.3

0.1

0.5

0.2

0.0

0 10 20 30 40 50

0 10 20 30 40 50

Step

Step

(a) Eval

(b) Explained Variance Figure 15. Training curves in ManiSkill.

65

Flow-SDE

Flow-Noise

60

Expert (34.5±2.6)

55

EpisodeLength

50

45

40

35

0 20 40 60 80 100 120 140

Step

Figure 16. Episode length: π0.5 RL training in ManiSkill.

other sub-tasks.

Training. In our π0.5 experiments on the LIBERO-Long benchmark, we observed that the Kullback–Leibler (KL) divergence metric increased steadily throughout training, potentially leading to instability. We mitigated this issue by implementing a learning rate scheduler with cosine annealing. As demonstrated in Fig. 14, this scheduler effectively prevents the KL divergence from escalating, thereby stabilizing the training process.

Critic. In our ManiSkill experiments, we observe that policy evaluation performance exhibits an initial dip before improving for both π0 and π0.5 models, as shown in Fig. 15. We attribute this transient degradation to the critic providing inaccurate signals during its warm-up phase. The subsequent eval improvement correlates directly with the critic’s value estimations stabilizing, as evidenced by the rising explained variance.

Temporal Efficiency. We also study how the rollout of RL in a physical simulator helps shape the policy to achieve expert-level temporal efficiency. We analyze the expert motion planning data used for SFT and then tracked the average episode lengths during the RL training of the π0.5 model in ManiSkill. As shown in Fig. 16, the SFT-initialized policy exhibits significantly longer episodes due to execution errors. In contrast, π0.5 achieves episode lengths that converge to the expert range after RL training, demonstrating a substantial improvement in temporal efficiency.

We attribute this convergence to two factors: (1) RL en-

hances the policy’s error-correction capabilities, allowing it to recover from execution failures. (2) Our partial reset mechanism incentivizes temporal efficiency through discounted rewards, as faster task completion enables the agent to trigger more resets and accumulate higher total rewards within each update cycle.

### H. RL for GR00T N1.5

#### H.1. Setup

GR00T N1.5. We conduct additional experiments based on the GR00T N1.5 model (Bjorck et al., 2025), which is a foundation model tailored for generalist humanoid robot reasoning and manipulation. The architecture integrates an Eagle 2.5 VLM (Chen et al., 2025), optimized for spatial grounding and physical reasoning, with a Diffusion Transformer head (Peebles & Xie, 2023) for action denoising. It facilitates multi-embodiment compatibility through specialized heads, supporting configurations such as humanoids with dexterous hands or grippers, as well as single-arm manipulators.

Regarding the critic implementation, we estimate value functions across the entire denoising trajectory by integrating the critic network directly with the action head. The complete framework is illustrated in Fig. 17.

Benchmark. We evaluate the model performance of GR00T across four manipulation task suites in LIBERO: Spatial, Object, Goal, and Long.

[Figure 71]

Figure 17. Illustration for the architecture of GR00T-N1.5. Table 10. Results of Finetuning GR00T using PPO with Flow-SDE on the LIBERO.

LIBERO Spatial Object Goal Long Avg. ∆ Avg. GR00T

Model

SFT 41.4 58.6 48.2 61.9 52.5 —

+PPO 92.5 96.2 84.3 86.6 89.9 +37.4

Implementation Details. Similar to the π0 implementation, we initiate our process with SFT on expert demonstrations. For the SFT stage, we fine-tune the entire model following the official setting. In the subsequent RL stage, we exclusively fine-tune the action expert model while keeping the vision-language model parameters fixed.

A crucial methodological refinement in our RL pipeline is the replacement of dropout layers in the expert model with identity layers. Dropout is widely recognized to induce instability during online RL training. Specifically, it introduces non-deterministic perturbations to the effective policy, shifting the standard probability ratio update from:

##### (at|st) πθ

πθ

new

ρt(θ) =

(at|st)

old

to a highly unstable form:

(17)

##### (at|st) πθ

πα

, (18)

new

ρt(θ) =

(at|st)

old

where ρt(θ) denotes the probability ratio, and αnew represents the policy state post-update as modified by the stochastic dropout mask. This structural stochasticity, compounded with per-step policy updates, severely undermines training convergence. The training hyperparameters are identical to those used for π0, with Flow-SDE employed as the primary RL algorithm.

#### H.2. Results

Results are summarized in Tab. 10. For the few-shot model, the SFT baseline achieves only a 52.5% success rate, reflecting limited generalization from sparse demonstrations. Conversely, our RL-based Flow-SDE significantly improves performance to 89.9%. These results, obtained using default π0 configurations, underscore the broad applicability of our method across architectures. While task-specific tuning could further enhance performance, we leave such optimization for future work.

### I. Limitations and Future Work

Noise Injection. Our current noise injection strategy exhibits a performance drop during the ODE-to-SDE conversion. Flow-CPS (Wang & Yu, 2025) attributes this loss to numerical error and proposes an improved coefficientspreserving sampling method. In our experiments, we attempted this configuration. Consistent with our hyperparameter ablation, our experiments showed that while this configuration mitigated the ODE-SDE precision error, it yielded limited RL improvement. Nevertheless, we argue that improving the noise injection strategy holds significant potential, specifically converting the ODE formulation to an SDE formulation while preserving the action distribution undisturbed.

Training Acceleration. Our current implementation of the mixed ODE-SDE rollout is simplistic in Flow-SDE, i.e., it

randomly selects one denoising step as an SDE step, while all other steps remain ODE steps. We posit that future investigations into mixed ODE-SDE rollouts, leveraging advances in accelerating flow-based image generation (Li et al., 2025b; He et al., 2025; Liu et al., 2025b; Li et al.,

- 2025c), could further enhance Flow-SDE, leading to faster training and improved performance.

Generalization. Maniskill OOD tests indicate that the semantic generalization of SFT and RL models remains limited. To address this, future work will leverage RL to enhance robustness by training on more diverse task distributions and varied linguistic instructions, thereby fostering better cross-task adaptability.

### J. Training Hyperparameters.

We record the training hyperparameters used to train both π0 and π0.5 on each benchmark, and present them in Tabs. 11 to 13.

Table 11. Hyperparameters across LIBERO.

Algorithms and Tasks π0 π0.5 Spatial Object Goal Long Spatial Object Goal Long

Parameters

Train epochs 500 500 500 500 500 500 500 500 Global batch size 2048 2048 2048 2048 2048 2048 2048 2048 Update epochs 4 4 4 4 1 1 3 4 Actor lr 5e-6 5e-6 5e-6 5e-6 5e-6 5e-6 5e-6 5e-6 Critic lr 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4 Scheduler False False False False False False False True

Reward discount rate γ 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 GAE λ 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 Clip ratio ϵ 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 Interaction steps 240 240 320 480 240 320 320 480 Parallel environments 64 64 64 64 64 64 64 64 Rollout epochs 8 8 8 8 8 8 8 8

Action prediction horizon H 50 50 50 50 10 10 10 10 Action replan horizon H′ 5 5 5 10 5 5 5 10

- Denoise steps 4 4 4 4 3 5 5 5 Noise level σ (Flow-SDE) 0.5 0.5 0.5 0.5 0.5 0.3 0.3 0.5 Max log-var (Flow-Noise) 0.16 0.16 0.16 0.16 0.10 0.10 0.10 0.10 Min log-var (Flow-Noise) 0.08 0.08 0.08 0.08 0.04 0.04 0.04 0.04 Entropy bonus (Flow-Noise) 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005

Table 12. Hyperparameters across SIMPLER and ManiSkill.

Parameters

Algorithms and Tasks π0 π0.5 Eggplant Carrot Spoon Cube ManiSkill Eggplant Carrot Spoon Cube ManiSkill SFT train steps 1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 RL train steps 40 40 40 130 150 40 40 40 70 150 Global batch size 2560 2560 2560 2560 5120 2560 2560 2560 2560 5120 Update epochs 4 4 4 4 4 4 4 4 4 5 Actor lr 5.6e-6 5.6e-6 5.6e-6 5.6e-6 7.91e-6 5.6e-6 5.6e-6 5.6e-6 5.6e-6 7.91e-6 Critic lr 1.1e-4 1.1e-4 1.1e-4 1.1e-4 1.55e-4 1.1e-4 1.1e-4 1.1e-4 1.1e-4 1.55e-4 Scheduler False False False False False False False False False False Reward discount rate γ 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 GAE λ 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 0.95 Clip ratio ϵ 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 Interaction steps 48 48 48 48 48 48 48 48 48 48 Parallel environments 256 256 256 256 320 256 256 256 256 320 Rollout epochs 1 1 1 1 1 1 1 1 1 1 Action prediction horizon H 8 8 8 8 8 8 8 8 8 8 Action replan horizon H′ 5 5 5 5 5 5 5 5 5 5

- Denoise steps 4 4 4 4 4 4 4 4 4 4 Noise level σ (Flow-SDE) 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 Max log-var (Flow-Noise) 0.16 0.16 0.16 0.16 0.16 0.10 0.10 0.10 0.10 0.10 Min log-var (Flow-Noise) 0.08 0.08 0.08 0.08 0.08 0.04 0.04 0.04 0.04 0.04 Entropy bonus (Flow-Noise) 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005 0.005

Table 13. Hyperparameters across MetaWorld and CALVIN benchmarks.

Benchmarks and models MetaWorld CALVIN π0 π0.5 π0 π0.5

Parameters

Train epochs 450 450 100 100 Global batch size 2048 2048 2048 2048 Update epochs 4 4 4 4 Actor lr 1e-5 5e-6 5e-6 5e-6 Critic lr 1e-4 1e-4 1e-4 1e-4 Scheduler False True False False

Reward discount rate γ 0.99 0.99 0.99 0.99 GAE λ 0.95 0.95 0.95 0.95 Clip ratio ϵ 0.2 0.2 0.2 0.2 Interaction steps 100 100 480 480 Parallel environments 64 64 64 64 Rollout epochs 8 8 8 8 Action prediction horizon H 5 5 5 5 Action replan horizon H′ 5 5 5 5

- Denoise steps 5 5 5 5 Noise level σ (Flow-SDE) 0.5 0.5 0.5 0.5 Max log-var (Flow-Noise) 0.10 0.10 0.16 0.16 Min log-var (Flow-Noise) 0.04 0.04 0.08 0.08 Entropy bonus (Flow-Noise) 0.005 0.005 0.005 0.005

