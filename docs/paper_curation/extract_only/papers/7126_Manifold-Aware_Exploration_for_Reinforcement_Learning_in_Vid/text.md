### Manifold-Aware Exploration for Reinforcement Learning in Video Generation

## arXiv:2603.21872v1[cs.CV]23Mar2026

###### Mingzhe Zheng*12 Weijie Kong*2 Yue Wu‡2 Dengyang Jiang1 Yue Ma1 Xuanhua He1 Bin Lin2 Kaixiong Gong2 Zhao Zhong2 Liefeng Bo2 Qifeng Chen†1 Harry Yang†1

[Figure 1]

Figure 1. Illustration of SAGE-GRPO. (Left) (a.1) At a higher noise region, Euler-style discretization introduces a purple region of extra energy (discretization error) beyond the true integral; we focus on the true integral region below, not this extra energy. (a.2) Our precise SDE removes unnecessary noise energy in high-noise regions, enabling more precise exploration and a better-learned data manifold. (Right) (b) Our method with improved exploration yields more stable and better-aligned generations compared with DanceGRPO (Xue et al., 2025), FlowGRPO (Liu et al., 2025b), and CPS (Wang & Yu, 2025).

###### Abstract

Group Relative Policy Optimization (GRPO) methods for video generation like FlowGRPO remain far less reliable than their counterparts for language models and images. This gap arises because video generation has a complex solution space, and the ODE-to-SDE conversion used for exploration can inject excess noise, lowering rollout quality and making reward estimates less reliable, which destabilizes post-training alignment. To address this problem, we view the pretrained model as defining a valid video data manifold and formulate the core problem as constraining exploration within the vicinity of this manifold, ensuring that rollout quality is preserved and reward estimates remain reliable. We propose SAGE-GRPO (Stable Alignment via Exploration), which applies constraints at both micro and macro levels. At the micro level, we derive

Work done during internship at Tencent Hunyuan, *Equal contribution, †Corresponding Authors, ‡Project Leader 1HKUST 2Tencent Hunyuan.

Preprint. March 24, 2026.

a precise manifold-aware SDE with a logarithmic curvature correction and introduce a gradient norm equalizer to stabilize sampling and updates across timesteps. At the macro level, we use a dual trust region with a periodic moving anchor and stepwise constraints so that the trust region tracks checkpoints that are closer to the manifold and limits long-horizon drift. We evaluate SAGEGRPO on HunyuanVideo1.5 using the original VideoAlign as the reward model and observe consistent gains over previous methods in VQ, MQ, TA, and visual metrics (CLIPScore, PickScore), demonstrating superior performance in both reward maximization and overall video quality. The code and visual gallery are available at here.

###### 1. Introduction

Group Relative Policy Optimization (GRPO) is a direct way to align video generation models with reward signals (Ho et al., 2020; Song et al., 2020b;a; Ma et al., 2025; Kong et al., 2024; Wu et al., 2025; Wan et al., 2025; Gao et al., 2025), but it has not yet been as reliable for video as it is

[Figure 2]

High-D Noise Space

(Off-Manifold) Case A: Linear Noise

(Current Baselines)

t+1

Manifold Departure

··(Jitter/Artifacts) 一 M

Zt (Video Manifold)

σα√△σ Xo

(Overestimated) (Data

ZT Zt+1

(Noise) Manifold-Aligned

Exploration

(Stable)

· M Caseσα Log-CorrectionB: Manifold-Aware(Ours)Noise

(Video Manifold) (Curvature Adapted)

- Figure 2. Geometric interpretation of noise injection strategies. Conventional linear SDEs (red) inject exploration noise using firstorder approximations, ignoring signal decay curvature and causing off-manifold drift that results in temporal jitter and artifacts. Our Manifold-Aware SDE (blue) uses a logarithmic correction term so that exploration noise is concentrated closer to the flow trajectory and the video manifold, reducing off-manifold drift.

- Table 1. Comparison of SDE noise injection strategies used in video GRPO.

|Method<br><br>|Standard Deviation Σ1t/2|
|---|---|
|DanceGRPO FlowGRPO<br><br>Ours (Precise)<br><br>|η√σt − σt+1 η σ<br><br>t<br><br>1−σt (σt − σt+1) η −(σt − σt+1) + log 1−σ<br><br>t+1 1−σt<br><br>|

for language models and images (Guo et al., 2025; Shao

- et al., 2024; Achiam et al., 2023; Shen et al., 2025). In GRPO training for video generation, we must draw a group of rollouts by converting the deterministic ODE sampler into an SDE sampler so that the policy can explore through diverse samples (Li et al., 2025a). Video generation has a large, structured solution space, so this exploration is easily disturbed. Current video GRPO baselines such as DanceGRPO and FlowGRPO rely on an Euler-style discretization and first-order approximations when deriving the SDE noise standard deviation (as shown in Table 1) (Black et al., 2023; Liu et al., 2025b; Xue et al., 2025). The resulting first-order truncation error can inject excess noise energy during sampling (shown in Figure 1(a.1)), which lowers rollout quality in high-noise steps and makes reward evaluation less reliable. This raises the following question: how can we obtain an accurate sampling path that improves rollout quality and stabilizes GRPO for video generation?

Flow-matching video generators parameterized by θ induce trajectories that are constrained by a pre-trained video generation model (Liu et al., 2022; Lipman et al., 2022; Wang et al., 2024). We treat this model as defining a valid data manifold M ⊂ RD. Because the pre-trained parameters θ0 are not yet sufficient for the target reward, GRPO must update θ through exploration while keeping trajectories within

the vicinity of M so that rollouts remain valid. As shown in Figure 2, FlowGRPO-style SDE exploration can overestimate the noise variance (red), push zt away from M, and produce temporal jitter. We therefore define the core problem of GRPO for video generation as how to constrain exploration within the vicinity of the data manifold so that each update improves rollouts while keeping reward evaluation reliable.

We propose SAGE-GRPO (Stable Alignment via Exploration), which organizes exploration at both micro and macro levels around the manifold. At the micro level, we refine the discrete SDE and couple it with a gradient norm equalizer as part of micro-scale exploration. Concretely, instead of using an area-based first-order variance approximation, we compute the noise variance by integrating diffusion coefficients over each step and add a logarithmic correction log 1−σ

1−σt , which yields a more accurate variance for ODE-to-SDE exploration. As in Figure 1(a.1), this corresponds to integrating only the effective energy under the curve rather than the extra discretization area, and Figure 1(a.2) shows that the resulting precise SDE uses smaller variance while staying closer to the underlying video manifold. Even with this corrected SDE, the diffusion process still has an inherent signal-to-noise imbalance across timesteps: gradients vanish at high noise (t → 1) and explode at low noise (t → 0), which biases learning toward certain phases. The Gradient Norm Equalizer normalizes optimization pressure across timesteps so that updates remain comparable in magnitude, which makes micro-level exploration more precise and stable.

t+∆t

With precise micro-level exploration, the policy after N steps updates tends to move closer to the data manifold; periodically updating a reference model from this trajectory therefore creates a trust region centered at a more manifold-consistent policy. This reduces long-horizon drift and helps avoid off-manifold local optima, as suggested by the red region in Figure 2. Traditional Fixed KL constraints DKL(πθ||π0) anchor the policy to the initial model π0, but as training progresses the optimal policy π∗ may be far from π0, which causes underfitting. Step-wise KL constraints DKL(πθ||πk−1) limit the magnitude of parameter updates per step (velocity control), ensuring smooth local transitions, but they only constrain the instantaneous update direction ∇θ and do not bound the cumulative displacement ∥θk − θ0∥ from the initial parameters. This allows unbounded drift: even if each step is small, the policy can move slowly but consistently away from the manifold over many steps, eventually leading to degradation or reward hacking. To counteract drift while preserving plasticity, we introduce a Periodical Moving Anchor that updates the reference policy πref every N steps, creating a dynamic trust region that repeatedly recenters exploration near a manifold-

consistent policy. We combine the moving anchor with step-wise constraints into a Dual Trust Region objective that provides position control towards the manifold and velocity control between successive policies, forming a positionvelocity controller that enables sustained plasticity.

We evaluate SAGE-GRPO on HunyuanVideo1.5 (Wu et al., 2025) using the original VideoAlign evaluator (Liu et al., 2025c) (no reward-model fine-tuning) and observe consistent gains over baselines such as DanceGRPO (Xue et al., 2025), FlowGRPO (Liu et al., 2025b), and CPS (Wang & Yu, 2025) in both overall reward and temporal fidelity. Extensive ablations confirm that both the micro-level design (precise manifold-aware SDE with temporal gradient equalization) and the macro-level Dual Trust Region objective are necessary to reduce the stability–plasticity gap.

Our main contributions are as follows:

- • We formulate GRPO for video generation as a manifoldconstrained exploration problem and show that the ODEto-SDE conversions used in existing methods can inject excess noise in high-noise steps, which reduces rollout quality and makes reward-guided updates less reliable.
- • At the micro-level, we constrain exploration with a Precise Manifold-Aware SDE and a Gradient Norm Equalizer, so that sampling noise stays manifold-consistent and updates are balanced across timesteps.
- • At the macro-level, we constrain long-horizon exploration with a Dual Trust Region with moving anchors and stepwise constraints, so that the trust region tracks more manifold-consistent checkpoints and prevents drift.

###### 2. Related Work

Reinforcement Learning for Diffusion and Flow Matching Models. Reinforcement learning has been adapted to fine-tune diffusion and flow matching models (Liu et al., 2025b; Xue et al., 2025; Xu et al., 2023; Jiang et al., 2025; Wallace et al., 2024; Xu et al., 2025; Lan et al., 2025; Jin

- et al., 2025; Lin et al., 2025a;b;c; Zhang et al., 2026) for alignment with human preferences. Early approaches such

- as DDPO (Black et al., 2023) and DPOK (Fan et al., 2023) treated the denoising process as a Markov Decision Process to enable policy gradient estimation. Inspired by GRPO in language models (Shao et al., 2024; Guo et al., 2025), FlowGRPO (Liu et al., 2025b) and DanceGRPO (Xue et al.,

2025) adapted GRPO to visual generation via ODE-to-SDE conversion for stochastic exploration (Li et al., 2025a). However, existing methods rely on first-order noise approximations that can drive exploration off the data manifold and overlook the inherent gradient imbalance across timesteps.

Preference Alignment for Video Generation. Aligning video generation models with human preferences is an active research area (Zheng et al., 2024; Long et al., 2025;

Huang et al., 2024; Lu et al., 2025; He et al., 2025). Building on video diffusion models (Wan et al., 2025; Kong et al.,

- 2024; Gao et al., 2025), researchers have developed video reward models (Liu et al., 2025c; Xu et al., 2024; Mi et al.,
- 2025; Zhang et al., 2025) and alignment algorithms (Li

- et al., 2024; Gambashidze et al., 2024; Yu et al., 2024; Zhou
- et al., 2025; Jia et al., 2025). DanceGRPO (Xue et al.,

2025) extends image-based RL to video, while Self-paced GRPO (Li et al., 2025b) proposes curriculum learning that dynamically adjusts reward weights. However, current alignment frameworks face a stability-plasticity dilemma: strict constraints (e.g., fixed KL anchored to initialization) limit plasticity, while relaxed constraints trigger reward hacking or catastrophic forgetting (Liu et al., 2025a; Li et al., 2025c). Unlike existing approaches that rely on heuristic scheduling or static anchors, our method integrates manifold-aware dynamics with a dual trust region to resolve this tension.

###### 3. Methodology

We formulate the problem of video alignment as maximizing the expected reward J(θ) = Ex

0∼πθ[R(x0)] within a Group Relative Policy Optimization (GRPO) framework. However, a standard application of GRPO to video diffusion models faces specific challenges in maintaining stable and effective exploration on the video manifold. SAGE-GRPO addresses these challenges by designing a unified exploration strategy that operates from micro-level noise injection to macro-level policy constraints, so that every exploration step remains valid and balanced across the diffusion process.

3.1. Preliminaries: Flow Matching and Group Relative Policy Optimization

Flow Matching and Rectified Flow. Flow Matching models generation as transport along a probability path pt(x) via an ordinary differential equation (ODE):

dxt dt

= vθ(xt,t), (1)

where vθ is a neural velocity field. Rectified Flow uses the linear interpolation path:

xt = (1 − σt)x0 + σtz1, (2) which implies the velocity field:

dσt dt

1 1 − σt

dxt dt

= −

(x0 − z1) =

(xt − x0).

vθ(xt,t) =

(3) Group Relative Policy Optimization (GRPO). Given a prompt c, GRPO samples a group of G rollouts and optimizes the diffusion policy πθ using a group-normalized advantage:

1 G

LGRPO(θ) = −

G

i=1

Ai ·

T

log πθ(x(t−i)1|x(ti),c), (4)

t=1

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(a) DanceGRPO (b) FlowGRPO (c) CPS (d) Ours

- Figure 3. Temporal gradient balancing ablation across SDE formulations. Overall VideoAlign reward curves comparing runs with and without the Gradient Norm Equalizer. Without balancing, low-noise timesteps dominate optimization, leading to unstable or plateaued rewards. With balancing, reward curves become smoother with consistent improvement, and gradient scale variation is reduced from more than one order of magnitude to within a small constant factor.

[Figure 7]

- Figure 4. Empirical gradient norm imbalance across noise levels. Observed norms (blue) decrease rapidly as σ increases and match

capture. Taking the square root yields the noise standard deviation:

1 − σt+1 1 − σt

Σ1t/2 = η −(σt − σt+1) + log

. (6)

Applying Euler-Maruyama discretization with timestep ∆t = σt − σt+1:

|xt+∆t = xt + vθ(xt,t)∆t +<br><br>Σt 2<br><br>sθ(xt) + Σ1t/2ϵ,|
|---|

(7)

where ϵ ∼ N(0,I) injects stochasticity, sθ(xt) ≈ −(xt − xˆ0)/σt2 is the score function estimate. Since Σt the integrated variance is over [σt+1,σt], the stochastic term is used Σ1t/2 directly without an additional √

the predicted relationship (red) ∥∇ log π∥ ∝ 1/Σ1t/2, leading to vanishing gradients at high noise (σ → 1) and exploding gradients

∆t factor. The Itˆo correction term Σt

- at low noise (σ → 0).

2 sθ(xt) ensures consistency with Rectified Flow marginals; a detailed derivation is provided in Appendix A.1.

where T is the number of diffusion steps. We defer the reward composition, advantage normalization, and the stochastic rollout formulation to Appendix A and only keep the key equations in the corresponding modules.

As shown in Figure 2, our method creates a smaller, manifold-aligned exploration region (blue ellipsoid) that stays tangent to the flow trajectory, whereas conventional methods create larger, off-manifold exploration regions (red sphere) that cause state drift. This geometric insight ensures that every exploration step remains within the legal video space, preventing temporal artifacts. Even with correct noise injection, the diffusion process has an inherent signalto-noise imbalance across timesteps: gradient norms vary by orders of magnitude (Figure 4), following a variancegradient inverse relationship. For a Gaussian transition π(xt−1|xt) = N(µθ,ΣtI):

###### 3.2. SAGE-GRPO Framework

- 3.2.1. MICRO-LEVEL EXPLORATION: PRECISE SDE AND GRADIENT EQUALIZATION

To enable stochastic exploration for GRPO, we perturb Rectified Flow with a marginal-preserving SDE whose noise stays aligned with the video manifold M ⊂ RD (Figure 2). The key challenge is computing the correct noise

standard deviation Σ1t/2 during discrete SDE discretization. For a marginal-preserving SDE with diffusion coefficient

1 Σ1t/2

, (8)

∥∇µ log π∥ ∝

εt = η σt/(1 − σt), we integrate the variance over the interval [σt+1,σt]:

causing gradients to vanish at high noise (t → 1) and explode at low noise (t → 0), biasing learning toward certain phases. To counteract this imbalance, we estimate a per-timestep gradient scale Nt from the SDE parameters (Appendix A.5) and apply a robust normalization:

σt

1 − σt+1 1 − σt

ε2s ds = η2 −(σt − σt+1) + log

Σt =

,

σt+1

(5) where η is the exploration scaling factor. The logarithmic term accounts for the geometric contraction of the signal coefficient (1 − σt), which linear approximations fail to

|St =<br><br>Median({Nτ}Tτ=1) Nt + ϵ<br><br>,|
|---|

(9)

[Figure 8]

Flow Matching Update πe Constraint:一β?DKL(πel|πref_N)Moving Anchor (Long-term)

Video Policy(πe) πe

Text Prompt

Manifold-Aware GRPO Step-wise Constraint Moving Anchor

###### SDE Sampling Update (Short-term) (πref_N)

Balanced Damping: πθ

t=T

Noise Video SAGE Noise Injection Gradients -β?DKL(πel|πt-1)Previous(πt-1)Step

Xt -staying tangent

∑tαlog(1--+1)I Temporal Gradient Equalizer

##### t

#### gt=St·gt

Video Manifold(M) Stα1/||E[▽Jt]ll

# 个

Gradient Reward Model

Calculation R(x)

t=0 t→1 t→0

Gradient Imbalance

Generated Video

Clip(xo) Feedback Loop: Reward & Temporal Gradient Equalization

- Figure 5. The SAGE-GRPO Framework. Our method resolves the stability-plasticity dilemma with three coupled components: (Left) a manifold-aware SDE that keeps exploration noise tangent to the video manifold, (Middle) a Temporal Gradient Equalizer that balances optimization across timesteps, and (Right) a Dual Trust Region that combines moving anchors and step-wise KL constraints for long-term stable alignment.

where ϵ is a small constant. This equalization normalizes optimization pressure across timesteps so that structural and textural updates contribute equally; empirical validation is provided in Figure 3 and Appendix A.5.

distance metric in policy space, anchoring the current policy to the reference. The choice of reference determines the constraint nature: a fixed reference creates a hard constraint, while a moving reference enables adaptive exploration.

GRPO With Composite Reward and Group-Normalized Advantage. We score each rollout x0 by a composite reward R(x0) and compute the group-normalized advantage Ai:

Fixed KL: Hard Constraint Limiting Optimality. Traditional approaches use a fixed reference policy πref = π0 from the pretrained video generation model. The constraint DKL(πθ||π0) forces the policy to remain close to the initial distribution. However, as training progresses, the optimal policy π∗ may be far from π0, and forcing DKL(πθ||π0) to be small prevents reaching π∗, leading to underfitting, which is too restrictive for long-term optimization where the policy needs to explore regions far from initialization.

ri − µR σR + ϵ

, (10)

Ai =

where ri = R(x(0i)), µR = G1 Gj=1 rj, and σR2 =

G j=1(rj − µR)2. Full definitions and implementation-

1 G

aligned details are in Appendix A.4.

Step-wise KL: Velocity Constraint. Step-wise KL uses the previous step’s policy as reference: πref = πk−1, where k denotes the optimization step. This constraint DKL(πθ||πk−1) acts as a velocity limit, restricting the magnitude of parameter updates per step:

- 3.2.2. MACRO-LEVEL EXPLORATION: DUAL TRUST REGION OPTIMIZATION

With micro-level exploration stabilized, we aim to prevent the policy model from drifting away from the data manifold and getting stuck in off-manifold local optima (Figure 2). We frame KL divergence as a dynamic anchoring mechanism that constrains exploration towards the data manifold.

###### ∥∇θDKL(πθ||πk−1)∥ ∝ ∥µθ − µk−1∥/Σt, (12)

ensuring smooth local transitions. However, velocity control alone only limits the magnitude of ∇θ (the update direction) but does not bound the cumulative displacement ∥θk − θ0∥ from the initial parameters. This allows unbounded drift: the policy move slowly but consistently away from the manifold, eventually leading to degradation or reward hacking.

KL Divergence as Dynamic Anchor. For a Gaussian policy π(xt−1|xt) = N(µθ,ΣtI), the KL divergence between the current policy πθ and a reference policy πref is:

(µθ − µref)2 2Σ2t ≈

(µθ − µref)2 2Σ2t

DKL(πθ||πref) = Ex

,

Periodical Moving KL: Position Control via Dynamic Trust Region. To counteract drift while maintaining plasticity, we introduce Periodical Moving KL that uses a periodically updated reference policy πref = πk−N, where

t∼πθ

(11) where µθ and µref are the mean predictions of the current and reference policies, respectively. KL divergence acts as a

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Baseline

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Ours

Prompt: The scene opens on a medium, low-angle shot of a teenage boy on an empty, red-surfaced running track during sunset…

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Baseline

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Ours

Prompt: …Issac Newton…sitting at the base of a large, gnarled apple tree...a ripe red apple…lands with a distinct ‘’thump’ on the top of Newton’s head…

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Baseline

- 1 2 3 4
- 2 34 5

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Ours

1

Prompt: The scene opens with a stunning wide shot…Five children, a diverse group of boys and girls…are running exuberantly across a vast field.

- Figure 6. Qualitative comparison against baselines. Three prompts illustrate our core gains: (Top) Reduced temporal jitter while preserving accurate visual contents; (Middle) Enhanced alignment and photorealism under occlusion and lighting changes; (Bottom) Stronger semantic alignment with consistent prompt matching across frames.

N is the update interval. For every N optimization step, we update the reference model: πref ← πθ, creating a resetting anchor mechanism. This allows the model to perform local exploration within N steps, then establish the new position as a safe region:

(µθ − µref N)2 2Σ2t

, (13)

DKL(πθ||πref N) =

where µref N is the mean prediction from the reference model updated N steps ago. This creates a dynamic trust region that periodically resets the safe zone, similar to a multistage relaxed version of TRPO (Schulman et al., 2015), enabling the model to climb the reward landscape in stages (plasticity) while tethered to a valid distribution (stability).

Dual KL: Position-Velocity Controller. We combine these two mechanisms into a dual KL objective that provides both position and velocity control:

LKL = βpos · DKL(πθ||πref N) + βvel · DKL(πθ||πk−1),

(14) where βpos and βvel are weighting coefficients. The position term DKL(πθ||πref N) provides the primary directional anchor, preventing long-term drift by constraining the policy to remain within a reasonable distance from a recent valid

distribution. The velocity term DKL(πθ||πk−1) acts as a damping factor, smoothing instantaneous updates and preventing abrupt policy changes. In practice, we compute the step-wise KL using log-probability differences from the rollout phase:

DKL(πθ||πk−1) ≈ E[log πk−1(xt−1|xt)−log πθ(xt−1|xt)],

(15) where the expectation is taken over samples generated with the previous policy πk−1. The full SAGE-GRPO objective that combines the GRPO policy loss, temporal equalization, and Dual KL regularization is provided in Appendix A.6.

###### 4. Experiments

###### 4.1. Experimental Setup

Implementation Details. We conduct all experiments on HunyuanVideo 1.5 (Kong et al., 2024) with per-GPU batch size 2 and 4 gradient accumulation steps (effective batch size 8). Each video contains 81 frames, and we apply GRPO updates every 20 sampling steps along the diffusion trajectory. Following (Liu et al., 2025b), we use VideoAlign (Liu et al., 2025c) as the reward oracle, evaluating Visual Quality (VQ), Motion Quality (MQ), and Text Alignment (TA),

- Table 2. Main Comparison on Video Generation Benchmarks. Comparison of SAGE-GRPO with baselines under two reward settings. The first row reports the original HunyuanVideo 1.5 performance. For each method, we report results without KL regularization (w/o KL) and with their Fixed KL constraints (w/ Fixed KL). For SAGE-GRPO, we demonstrate the w/ Dual Moving KL mechanism. Bold, underline, and gray indicate the best, second best, and third best results, computed across both settings (A+B).

|Method<br><br>|Configuration<br><br>|VideoAlign Metrics Overall VQ MQ TA|Visual Metrics CLIPScore PickScore<br><br>|
|---|---|---|---|
|HunyuanVideo 1.5 (Original)|-|0.0654 -0.7539 -0.5870 1.4063|0.5409 0.7397|

###### Setting A: Averaged Rewards (wvq = 1.0,wmq = 1.0,wta = 1.0)

|DanceGRPO<br><br>|w/o KL w/ Fixed KL<br><br>|0.2768 -0.7589 -0.3852 1.4209 0.0979 -0.8077 -0.5091 1.4147<br><br>|0.5386 0.7378 0.5403 0.7355<br><br>|
|---|---|---|---|
|FlowGRPO<br><br>|w/o KL w/ Fixed KL<br><br>|0.2733 -0.7151 -0.5286 1.5170 0.1880 -0.6771 -0.5912 1.4563<br><br>|0.5443 0.7394 0.5431 0.7407|
|CPS<br><br>|w/o KL w/ Fixed KL|0.6343 -0.4855 -0.4021 1.5219 0.0928 -0.7156 -0.5825 1.3908<br><br>|0.5479 0.7412 0.5479 0.7369<br><br>|
|SAGE-GRPO<br><br>|w/o KL w/ Fixed KL w/ Dual Mov KL|0.4859 -0.6104 -0.4141 1.5104 0.2244 -0.7438 -0.5320 1.5001 0.2173 -0.7881 -0.4249 1.4303<br><br>|0.5423 0.7360 0.5446 0.7382 0.5430 0.7452<br><br>|

###### Setting B: Alignment-Focused (wvq = 0.5,wmq = 0.5,wta = 1.0)

|DanceGRPO<br><br>|w/o KL w/ Fixed KL|-0.2172 -0.8854 -0.6218 1.2901 0.1290 -0.7739 -0.5083 1.4112<br><br>|0.5439 0.7352 0.5452 0.7276|
|---|---|---|---|
|FlowGRPO|w/o KL<br><br>w/ Fixed KL<br><br>|0.4773 -0.5671 -0.4731 1.5175 0.2103 -0.6654 -0.5506 1.4263<br><br>|0.5403 0.7349 0.5427 0.7408|
|CPS<br><br>|w/o KL w/ Fixed KL|0.3694 -0.6650 -0.5325 1.5669 0.3705 -0.6121 -0.4787 1.4613<br><br>|0.5479 0.7311 0.5458 0.7364<br><br>|
|SAGE-GRPO<br><br>|w/o KL w/ Fixed KL w/ Dual Mov KL<br><br>|-0.1222 -0.8720 -0.6046 1.3544 0.2857 -0.7062 -0.4425 1.4344 0.8066 -0.4765 -0.2384 1.5216<br><br>|0.5404 0.7357 0.5414 0.7377 0.5484 0.7420|

with overall reward R = wvqSvq +wmqSmq +wtaSta. We compare SAGE-GRPO against DanceGRPO (Xue et al., 2025), FlowGRPO (Liu et al., 2025b), and CPS (Wang & Yu, 2025). The KL regularization weight is scheduled in λKL ∈ [10−7,10−5] according to Appendix A.6.

###### 4.2. Main Results

We consider two reward configurations (Table 2): averaged (wvq = 1.0,wmq = 1.0,wta = 1.0) and alignmentfocused (wvq = 0.5,wmq = 0.5wta = 1.0). All rewards use the original VideoAlign model as a frozen evaluator (no reward-model fine-tuning), which ensures consistent evaluation across methods. Since current video GRPO baselines are implemented with substantial differences in engineering optimizations, directly reusing them would confound algorithmic effects with infrastructure choices. To obtain a fair comparison, we implement a unified training framework on HunyuanVideo1.5 with shared infrastructure across all methods and vary only the GRPO algorithm itself.

Under the averaged-reward setting that matches LongcatVideo (Team et al., 2025), adding KL regularization typically improves visual performance but yields worse reward behavior, which we attribute to reward hacking in the reward model as discussed in previous work (Li et al., 2025b). We

compare previous methods and SAGE-GRPO under both averaged and alignment-focused rewards, and evaluate variants with and without KL regularization, as summarized in Table 2. We further study how placing more weight on semantic alignment can reduce reward hacking artifacts. In the alignment-focused setting (Setting B), SAGE-GRPO with Dual Moving KL achieves the best Overall, VQ, MQ, and CLIPScore while remaining close to the best TA, and overall Table 2 suggests that emphasizing alignment provides a more reliable optimization target and yields more stable gains in both reward and visual metrics.

###### 4.3. Qualitative Analysis

We provide qualitative examples that complement the quantitative trends. Figure 6 highlights the improvement in coherence, photorealism, and semantic alignment over baselines, especially for prompts that require precise object interactions and long-range motion. Additional visual comparisons demonstrating superior alignment with emotional descriptions in text prompts are presented in Appendix Figure 10.

###### 4.4. User Study

To corroborate our automatic metrics, we conducted a user preference study with 29 evaluators on 32 prompts, compar-

[Figure 45]

[Figure 46]

[Figure 47]

(a) VQ reward (b) MQ reward (c) TA reward

- Figure 7. KL weight ablation on VideoAlign rewards. Comparison of three KL weight schedules: fixed 10−5 (green), two-stage 10−7 → 10−5 (red), and two-stage 10−7 → 10−6 (yellow). The two-stage schedule 10−7 → 10−5 achieves the strongest and most consistent gains across VQ, MQ, and TA, supporting gradually increasing λKL to tighten the trust region (Appendix A.6).

ing SAGE-GRPO with baselines (all at iter 100, sampling step 40, Setting B) across Visual Quality, Motion Quality, and Semantic Alignment. Table 3 reports the pairwise win rates of SAGE-GRPO against each baseline.

- Table 3. User Preference Study. Win rates of SAGE-GRPO against baselines. Results indicate a strong human preference for our method, especially in Motion Quality, confirming that automatic metrics align with perceptual quality.

SAGE-GRPO vs. Visual Quality Motion Quality Semantic Alignment

DanceGRPO 85.9% 75.8% 79.2% FlowGRPO 83.8% 79.2% 71.9% CPS 80.2% 70.8% 67.9%

- 4.5. Ablation Studies

- 4.5.1. IMPACT OF TEMPORAL GRADIENT EQUALIZER

To evaluate the effectiveness of the Temporal Gradient Equalizer in Section 3.2.1, we compare training dynamics with and without per-timestep balancing across three SDE formulations and CPS. Figure 3 shows the overall VideoAlign reward curves for baselines and our method.

[Figure 48]

[Figure 49]

(a) Mean reward (b) Std (exploration)

Figure 8. KL strategy ablation. (a) Dual Moving KL achieves the highest and most stable reward, supporting the position-velocity control interpretation (Equation (14)). (b) Moving KL attains high exploration in early training steps but the exploration level drops in later stages. Dual Moving KL maintains a higher and more stable exploration level throughout training.

shows that the two-stage schedule yields higher rewards and smoother trajectories across VQ, MQ, and TA, consistent with gradually increasing λKL to tighten the trust region. Implementation details are in Appendix A.6.

- 4.5.2. KL STRATEGY ABLATION

We next study the effect of different KL strategies introduced in Section 3.2.2. Figure 8 reports both the mean reward and standard deviation for four KL strategies, with qualitative comparisons in Appendix Figures 11 and 12.

- Figure 8a shows that Dual Moving KL consistently outperforms other variants in both convergence speed and final reward while avoiding the collapse observed in aggressive step-wise updates. Figure 8b shows that Moving KL explores quickly initially but exploration falls off; Dual Moving KL maintains higher exploration stably, validating the position-velocity controller interpretation in Equation (14).

- 4.5.3. KL WEIGHT SENSITIVITY

We compare three KL weight schedules: fixed 10−5, twostage 10−7 → 10−5, and milder 10−7 → 10−6. Figure 7

###### 5. Conclusion

We presented SAGE-GRPO, a manifold-aware GRPO framework for stable reinforcement learning for video generation. The core challenge is to design exploration strategies that respect the manifold structure, where each exploration step stays within the vicinity of the manifold rather than drifting into high-noise regions. At the micro-level, we derive a Precise Manifold-Aware SDE that keeps exploration noise closer to the flow trajectory, and introduce a Gradient Norm Equalizer that normalizes optimization pressure across timesteps. At the macro-level, we propose a Dual Trust Region mechanism combining position and velocity control to reduce off-manifold local optima while enabling sustained plasticity. Experiments on HunyuanVideo1.5 with VideoAlign reward show consistent improvements over strong baselines and validate the contribution of each component through ablations.

###### Impact Statement

This paper presents a method for more stable reinforcement learning alignment of text-to-video generation models. By improving temporal consistency and text alignment under a fixed reward model, our work may strengthen creative tools, scientific communication, and educational content that rely on controllable video synthesis. At the same time, stronger video generation systems can exacerbate existing concerns about misinformation, deepfakes, biased or harmful content, and the computational cost of large-scale training and sampling. Our experiments are conducted in a research setting on an existing model and evaluator, and our user study involves 29 voluntary evaluators rating 32 prompts comparing SAGE-GRPO against baselines in terms of visual quality, motion quality, and semantic alignment; there is no collection of personal data, but any future deployment should include safeguards such as content moderation, dataset auditing, and human oversight to reduce these risks.

###### References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Black, K., Janner, M., Du, Y., Kostrikov, I., and Levine, S. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

Fan, Y., Watkins, O., Du, Y., Liu, H., Ryu, M., Boutilier, C., Abbeel, P., Ghavamzadeh, M., Lee, K., and Lee, K. Dpok: Reinforcement learning for fine-tuning text-toimage diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.

Gambashidze, A., Kulikov, A., Sosnin, Y., and Makarov, I. Aligning diffusion models with noise-conditioned perception. arXiv preprint arXiv:2406.17636, 2024.

Gao, Y., Guo, H., Hoang, T., Huang, W., Jiang, L., Kong, F., Li, H., Li, J., Li, L., Li, X., et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, D., Feng, G., Ge, X., Niu, Y., Zhang, Y., Ma, B., Song, G., Liu, Y., and Li, H. Neighbor grpo: Contrastive ode policy optimization aligns flow models. arXiv preprint arXiv:2511.16955, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Huang, T., Jiang, G., Ze, Y., and Xu, H. Diffusion reward: Learning rewards via conditional video diffusion. In European Conference on Computer Vision, pp. 478–495. Springer, 2024.

Jia, Z., Nan, Y., Zhao, H., and Liu, G. Reward finetuning two-step diffusion models via learning differentiable latent-space surrogate reward. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12912–12922, 2025.

Jiang, D., Liu, D., Wang, Z., Wu, Q., Li, L., Li, H., Jin, X., Liu, D., Li, Z., Zhang, B., et al. Distribution matching distillation meets reinforcement learning. arXiv preprint arXiv:2511.13649, 2025.

Jin, D., Xu, R., Zeng, J., Lan, R., Bai, Y., Sun, L., and Chu, X. Semantic context matters: Improving conditioning for autoregressive models. arXiv preprint arXiv:2511.14063, 2025.

Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Lan, R., Bai, Y., Duan, X., Li, M., Jin, D., Xu, R., Sun, L., and Chu, X. Flux-text: A simple and advanced diffusion transformer baseline for scene text editing. arXiv preprint arXiv:2505.03329, 2025.

Li, J., Feng, W., Chen, W., and Wang, W. Y. Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027, 2024.

Li, J., Cui, Y., Huang, T., Ma, Y., Fan, C., Yang, M., and Zhong, Z. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025a.

Li, R., Liang, Y., Ni, Z., Huang, H., Zhang, C., and Li, X. Growing with the generator: Self-paced grpo for video generation. arXiv preprint arXiv:2511.19356, 2025b.

Li, Y., Wang, Y., Zhu, Y., Zhao, Z., Lu, M., She, Q., and Zhang, S. Branchgrpo: Stable and efficient grpo with structured branching in diffusion models. arXiv preprint arXiv:2509.06040, 2025c.

Lin, Y., Lin, Z., Chen, H., Pan, P., Li, C., Chen, S., Wen, K., Jin, Y., Li, W., and Ding, X. Jarvisir: Elevating autonomous driving perception with intelligent image restoration. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22369–22380, 2025a.

Lin, Y., Lin, Z., Lin, K., Bai, J., Pan, P., Li, C., Chen, H., Wang, Z., Ding, X., Li, W., et al. Jarvisart: Liberating human artistic creativity via an intelligent photo retouching agent. arXiv preprint arXiv:2506.17612, 2025b.

Lin, Y., Wang, L., Lin, K., Lin, Z., Gong, K., Li, W., Lin, B., Li, Z., Zhang, S., Peng, Y., et al. Jarvisevo: Towards a self-evolving photo editing agent with synergistic editorevaluator optimization. arXiv preprint arXiv:2511.23002, 2025c.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Liu, H., Huang, H., Wang, J., Liu, C., Li, X., and Ji, X. Diversegrpo: Mitigating mode collapse in image generation via diversity-aware grpo. arXiv preprint arXiv:2512.21514, 2025a.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025b.

Liu, J., Liu, G., Liang, J., Yuan, Z., Liu, X., Zheng, M., Wu, X., Wang, Q., Xia, M., Wang, X., et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025c.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Long, Z., Zheng, M., Feng, K., Zhang, X., Liu, H., Yang, H., Zhang, L., Chen, Q., and Ma, Y. Follow-your-shape: Shape-aware image editing via trajectory-guided region control. arXiv preprint arXiv:2508.08134, 2025.

Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K. L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025.

Ma, Y., Feng, K., Hu, Z., Wang, X., Wang, Y., Zheng, M., Wang, B., Wang, Q., He, X., Wang, H., et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025.

Mi, X., Yu, W., Lian, J., Jie, S., Zhong, R., Liu, Z., Zhang, G., Zhou, Z., Xu, Z., Zhou, Y., et al. Video generation models are good latent reward models. arXiv preprint arXiv:2511.21541, 2025.

Nair, A., Gupta, A., Dalal, M., and Levine, S. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359, 2020.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. PMLR, 2015.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Shen, X., Li, Z., Yang, Z., Zhang, S., Zhang, Y., Li, D., Wang, C., Lu, Q., and Tang, Y. Directly aligning the full diffusion trajectory with fine-grained human preference. arXiv preprint arXiv:2509.06942, 2025.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. International Conference on Learning Representations, 2020a.

Song, Y., Sohl-Dickstein, J. N., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. International Conference On Learning Representations, 2020b.

Team, M. L., Cai, X., Huang, Q., Kang, Z., Li, H., Liang, S., Ma, L., Ren, S., Wei, X., Xie, R., et al. Longcat-video technical report. arXiv preprint arXiv:2510.22200, 2025.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, F. and Yu, Z. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Wang, F.-Y., Yang, L., Huang, Z., Wang, M., and Li, H. Rectified diffusion: Straightness is not your need in rectified flow. arXiv preprint arXiv:2410.07303, 2024.

Wu, B., Zou, C., Li, C., Huang, D., Yang, F., Tan, H., Peng, J., Wu, J., Xiong, J., Jiang, J., et al. Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870, 2025.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: learning and evaluating human preferences for text-to-image generation. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pp. 15903–15935, 2023.

Xu, J., Huang, Y., Cheng, J., Yang, Y., Xu, J., Wang, Y., Duan, W., Yang, S., Jin, Q., Li, S., et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.

Xu, R., Jin, D., Bai, Y., Lan, R., Duan, X., Sun, L., and Chu, X. Scalar: Scale-wise controllable visual autoregressive learning. arXiv preprint arXiv:2507.19946, 2025.

Xue, Z., Wu, J., Gao, Y., Kong, F., Zhu, L., Chen, M., Liu, Z., Liu, W., Guo, Q., Huang, W., et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

Yu, X., Bai, C., He, H., Wang, C., and Li, X. Regularized conditional diffusion model for multi-task preference alignment. Advances in Neural Information Processing Systems, 37:139968–139996, 2024.

- Zhang, S., Zhang, Z., Dai, C., and Duan, Y. E-grpo: High entropy steps drive effective reinforcement learning for flow models. arXiv preprint arXiv:2601.00423, 2026.
- Zhang, T., Da, C., Ding, K., Yang, H., Jin, K., Li, Y., Gao, T., Zhang, D., Xiang, S., and Pan, C. Diffusion model as a noise-aware latent reward model for step-level preference optimization. arXiv preprint arXiv:2502.01051, 2025.

Zheng, M., Xu, Y., Huang, H., Ma, X., Liu, Y., Shu, W., Pang, Y., Tang, F., Chen, Q., Yang, H., et al. Videogen-of-thought: Step-by-step generating multi-shot video with minimal manual intervention. arXiv preprint arXiv:2412.02259, 2024.

Zhou, Y., Ling, P., Bu, J., Wang, Y., Zang, Y., Wang, J., Niu, L., and Zhai, G. Fine-grained grpo for precise preference alignment in flow models. arXiv preprint arXiv:2510.01982, 2025.

###### A. Appendix

###### A.1. Derivation of Manifold-Aware SDE Variance

Derivation To enable stochastic exploration for GRPO, we need to convert the deterministic Rectified Flow ODE into a stochastic differential equation (SDE) that preserves the marginal probability distribution at each timestep. Recall the general form of a marginal-preserving SDE for flow matching:

dzt = vθ(xt,t) − 12ε2tsθ(xt) dt + εt dwt, (16)

where εt is the diffusion coefficient (a function of t), wt is a Brownian motion, and sθ(xt) ≈ −(xt − xˆ0)/σt2 is the score function estimate. The Itˆo correction term 12ε2tsθ(xt) ensures that the SDE preserves the same marginal distribution as the deterministic ODE.

To discretize this SDE, we assume that vθ(xt,t) and sθ(xt) remain approximately constant during the interval [σt+1,σt], where σt is the noise level at timestep t. The key challenge is to compute the integrated variance Σt for the stochastic term. We define:

σt

ε2s ds. (17)

Σt :=

σt+1

For Rectified Flow, we choose εt = η σ

1−σt to match the geometric structure of the flow trajectory, where η is the exploration scaling factor. Substituting this form and integrating:

t

σt

σs 1 − σs

η2

ds (18)

Σt =

σt+1

σt

1 1 − σs − 1 ds (19)

= η2

σt+1

= η2 [−log(1 − σs) − σs]σσt

(20)

t+1

1 − σt+1 1 − σt

= η2 −(σt − σt+1) + log

. (21)

Taking the square root, we obtain the noise standard deviation:

1 − σt+1 1 − σt

Σ1t/2 = η −(σt − σt+1) + log

. (22)

The logarithmic term log((1 − σt+1)/(1 − σt)) accounts for the geometric contraction of the signal coefficient (1 − σt), which linear approximations fail to capture.

Applying Euler-Maruyama discretization with timestep ∆t = σt − σt+1, the discretized SDE becomes:

Σt 2

sθ(xt) + Σ1t/2ϵ, (23)

xt+∆t = xt + vθ(xt,t)∆t +

where ϵ ∼ N(0,I) is the injected stochasticity. Note that Σt is already the integrated variance over the interval [σt+1,σt], so the stochastic term uses Σ1t/2 directly without an additional √

∆t factor.

Problem Formulation. Let the noise level at timestep t be σt. In a Rectified Flow setting, the trajectory connects pure noise (σ = 1) to data (σ = 0). We aim to find the precise variance Σt required for the stochastic step such that the marginal distribution is preserved up to the second order.

Let ∆σ = σt − σt+1 > 0. We analyze the terms inside the square root of our proposed Eq. 7. Let Vt denote the variance term:

1 − σt+1 1 − σt

(24)

Vt = −(σt − σt+1) + log

Taylor Expansion Analysis. First, we express the logarithmic term using ∆σ:

1 − σt+1 1 − σt

1 − (σt − ∆σ) 1 − σt

∆σ 1 − σt

log

= log

= log 1 +

2

Let x = 1∆−σσ

. Since step sizes are small, |x| < 1. We apply the Taylor expansion log(1 + x) ≈ x − x

2 + O(x3):

t

2

∆σ 1 − σt ≈

∆σ 1 − σt −

- 1

- 2

∆σ 1 − σt

log 1 +

Substituting this back into Vt:

(25)

(26)

∆σ2 (1 − σt)2

∆σ 1 − σt −

- 1

- 2

(27)

Vt ≈ −∆σ +

∆σ2 (1 − σt)2

1 1 − σt − 1 −

1 2

(28)

= ∆σ

σt

1 − σt − O(∆σ2) (29) The leading term ∆σ σ

= ∆σ

1−σt represents the ideal variance scaling for a geometric schedule, which linear approximations fail to capture.

t

###### A.2. Standard Deviation Comparison: Ours vs. FlowGRPO

Figure 9 compares the noise standard deviation per step between our precise SDE and FlowGRPO under three parameterization regimes to understand how different noise handling strategies affect exploration behavior.

- Regime (a): Both methods using FlowGRPO’s σ schedule. When both methods use FlowGRPO’s default σ schedule (where σt is set to σmax at early steps), our precise SDE exhibits near-zero standard deviation at the first step. This

occurs because our method computes noise variance via integration: Σt = σ σt

t+1

ε2s ds. When both endpoints are equal (σt = σt+1 = σmax), the integration interval collapses, and the logarithmic term log((1 − σt+1)/(1 − σt)) evaluates to zero, yielding Σt ≈ 0. This demonstrates that our integral-based formulation is sensitive to the σ schedule and requires proper boundary handling.

- Regime (b): Both methods using aggressive clamping at 1 − 3 × 10−3. When both methods apply the same clamping threshold (1 − σ) ≥ 3 × 10−3 (equivalently, σ ≤ 1 − 3 × 10−3), FlowGRPO exhibits explosive behavior at the first step, with standard deviation reaching values around 3.0. This instability arises because FlowGRPO’s noise computation involves a ratio σ/(1 − σ); when (1 − σ) is clamped to a small constant while σ remains large, the denominator becomes artificially small, causing the ratio to explode. In contrast, our precise SDE maintains stable and controlled standard deviation throughout, starting around 1.0 and decaying smoothly, demonstrating that our manifold-aware formulation inherently handles low-noise regimes more robustly.
- Regime (c): Each method using its default implementation. Under their respective default configurations, FlowGRPO uses its standard σ schedule, while our method applies clamping at (1 − σ) ≥ 3 × 10−3. Our method maintains a lower standard deviation than FlowGRPO across most of the diffusion trajectory, particularly in later steps. This demonstrates that our precise SDE effectively reduces injected noise magnitude, leading to more refined exploration along the data manifold.

Across all three regimes, our method consistently achieves smaller or more stable standard deviation than FlowGRPO. This supports the main-figure narrative (Figure 1): we remove unnecessary high-frequency noise energy in high-noise regions, enabling more precise exploration that stays closer to the data manifold. This behavior aligns with the micro-level exploration design of our SDE in Section 3.2.1.

###### A.3. Theoretical Gradient Norm Analysis

Here we derive the relationship between the gradient norm and the noise schedule. For a Gaussian policy π(xt−1|xt) = N(µθ,ΣtI), the gradient of the log-probability with respect to the drift parameter µθ is:

xsample − µθ

(30)

[Figure 50]

[Figure 51]

[Figure 52]

(a) Both using FlowGRPO’s σ schedule. (b) Both clamped at 1 − 3 × 10−3. (c) Each using default implementation.

- Figure 9. Step-wise std comparison: our precise SDE vs. FlowGRPO. (a) When both use FlowGRPO’s σ schedule, our integral-based formulation yields near-zero std at the first step due to equal endpoints. (b) When both are clamped at (1 − σ) ≥ 3 × 10−3, FlowGRPO explodes at step 1, while ours remains stable. (c) Under default implementations, ours maintains lower std across most steps. This supports that we remove ineffective high-frequency noise and explore more precisely along the manifold (Section 3.2.1).

Since xsample ∼ N(µθ,Σt), the expected norm is proportional to the standard deviation of the noise:

√Σt Σt

1 √Σt

E[∥∇µ log π∥] ∝

=

(31)

Given our derived Manifold-Aware variance Σt ≈ η2∆σ σ

1−σt, the gradient norm scales as:

t

∥∇∥ ∝

1 − σt σt∆σ

(32)

This confirms that as σt → 0 (low noise), the gradient norm explodes, necessitating our proposed Gradient Equalizer.

###### A.4. GRPO Reward and Advantage Details

Here we provide the implementation-aligned definitions of reward composition and the group-normalized advantage used in Equation (4). Following VideoAlign (Liu et al., 2025c), we construct a composite reward for a generated video x0:

R(x0) = wvqSvq(x0) + wmqSmq(x0) + wtaSta(x0), (33) where Svq, Smq, and Sta score visual quality, motion quality, and text alignment, and wvq,wmq,wta are fixed scalar weights. Given a prompt c, GRPO samples a group of G rollouts {x(0i)}Gi=1 and computes rewards ri = R(x(0i)). We use the group mean and standard deviation as a baseline:

µR =

G

1 G

rj, σR =

j=1

G

1 G

(rj − µR)2, (34)

j=1

and define the normalized advantage:

ri − µR σR + ϵ

, (35) where ϵ is a small constant for numerical stability.

Ai =

###### A.5. Temporal Gradient Equalizer: Derivation of Nt

We outline how to obtain a per-timestep gradient scale proxy Nt that is compatible with the SDE transition used in Section 3.2.1. Consider a Gaussian transition π(xt−1 | xt) = N(µθ,ΣtI) parameterized through the network output (e.g., velocity/denoiser prediction) and a noise variance Σt determined by the chosen SDE. The log-probability gradient with respect to the mean parameter satisfies:

xsample − µθ

. (36)

[Figure 53]

- Figure 10. Qualitative comparison highlighting emotional alignment. Two prompts illustrate SAGE-GRPO’s ability to better align with emotional descriptions: (Top) A teenage boy in a coffee shop, where SAGE-GRPO captures the ”calm, contemplative expression” and gentle motion of lowering the mug, while baselines show neutral expressions and abrupt movements. (Bottom) An older chef in a kitchen, where SAGE-GRPO consistently renders the ”lines of fatigue” and ”somber mood” through deep side-lighting shadows, while baselines fail to convey the intended emotional depth. Our manifold-aware exploration enables precise alignment with subtle emotional and action cues.

Since xsample − µθ ∼ N(0,ΣtI), its magnitude is O(Σ1t/2) in expectation, yielding the inverse relationship:

1 Σ1t/2

. (37)

E ∥∇µ log π∥ ∝

In practice, the network does not directly parameterize µθ; instead, µθ is obtained by composing the network prediction with the SDE/solver update rule, introducing an additional sensitivity factor. Let λt denote the scalar sensitivity from the solver mapping (details depend on the SDE type and discretization). We use the proxy:

λt Σ1t/2

, (38)

Nt =

and define the Temporal Gradient Equalizer (Equation (9)) as a robust normalization:

Median({Nτ}Tτ=1) Nt + ϵ

. (39)

St =

This produces approximately uniform gradient scales across timesteps, aligning with the empirical observation in Figure 4 and the training-curve improvement in Figure 3.

###### A.6. SAGE-GRPO Objective and Adaptive KL Weighting

We provide the complete objective used in SAGE-GRPO, combining GRPO, the Temporal Gradient Equalizer, and Dual KL regularization, together with a principled schedule for the overall KL coefficient. At each optimization step, we sample a group of G rollouts and compute advantages {Ai}Gi=1 as in Appendix A.4.

Dual KL regularizer. We use two reference policies to implement a position–velocity controller in policy space (Section 3.2.2). The regularizer is

LKL = βpos · DKL(πθ∥πref N) + βvel · DKL(πθ∥πk−1), (40)

where πk−1 is the previous policy and πref N is a periodically refreshed anchor. The term DKL(πθ∥πk−1) constrains the instantaneous update (velocity control), while DKL(πθ∥πref N) constrains the cumulative displacement from the anchor (position control). This separation is important for long-horizon training: velocity-only constraints can still accumulate drift, whereas a single fixed anchor can be overly restrictive.

Adaptive KL weighting. We interpret the overall KL coefficient λKL as a Lagrange multiplier associated with a trust-region constraint E[DKL(πθ∥πref)] ≤ δ. Instead of fixing λKL, we adapt it online so that the realized KL remains close to a target scale, analogous in spirit to adaptive behavior regularization in AWAC (Nair et al., 2020), where a temperature parameter is adapted from advantage statistics.

Warm-up (two-stage increase). Let λmin = 10−7 and λmax = 10−5 denote the minimum and maximum KL coefficients. During the first K = 100 optimization steps, we use a linear warm-up:

k K

, k ≤ K, (41)

λKL(k) = λmin + (λmax − λmin) ·

which corresponds to the two-stage schedules reported in Figure 7. This design keeps the trust region weak early to avoid underfitting, and gradually strengthens it as the policy improves.

Conservative feedback control. After warm-up, we apply a proportional feedback update based on the recent KL history, similar to the P-term of a PID controller. Let D¯KL be the mean of the last H = 10 observed KL values and let Dtarget be the desired KL scale. We define the KL error eKL = D¯KL − Dtarget and update

 

- 0.9λKL, D¯KL > (1 + 0.5)Dtarget,
- 1.1λKL, D¯KL < (1 − 0.5)Dtarget, λKL, otherwise,

λKL ∈ [λmin,λmax], (42)

λKL ←



where clipping enforces the same bounds as the warm-up stage. Intuitively, if the empirical KL is much larger than Dtarget, the controller reduces λKL to relax the constraint; if it is much smaller, the controller increases λKL to tighten the trust region. This combination of warm-up and feedback control stabilizes the effective trust-region radius and explains the smooth reward trajectories observed in Figure 7.

Full SAGE-GRPO loss. Combining GRPO, the Temporal Gradient Equalizer, and the adaptively weighted Dual KL regularizer yields:

|LSAGE-GRPO(θ) = −<br><br>1 G<br><br>G<br><br>i=1<br><br>Ai ·<br><br>T<br><br>t=1<br><br>St · log πθ(x(t−i)1 | x(ti),c) − λKL · LKL.|
|---|

(43)

###### A.7. Additional Qualitative Results

We include additional qualitative visualizations to complement the quantitative experiments in Section 4. Figure 10 demonstrates SAGE-GRPO’s superior ability to align with emotional descriptions in text prompts, capturing subtle facial expressions and mood cues that baselines often miss.

To further validate the effectiveness of different KL strategies discussed in Section 3.2.2, we provide qualitative comparisons across five variants: no KL regularization, Fixed KL (anchored to the initial model π0), Step-wise KL (velocity control only), Moving KL (position control only), and Dual Moving KL (combining both position and velocity control). Figures 11 and 12 show that Dual Moving KL consistently produces more realistic details, better temporal consistency, and stronger alignment with prompt descriptions compared to other KL strategies, which aligns with the quantitative findings in Section 4.2.

|No KL|
|---|

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Fixed KL

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Step-wise KL

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Moving KL

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

Dual Mov KL

Prompt: …The soldier's features are etched with fatigue and grime; their skin pale against the dark dirt and sweat, their eyes hollow and bloodshot, their chins darkened by the rough stubble of days past…take a slow, measured sip, their throats noticeably moving as they swallow, and a look of desperate relief crosses their faces before they close their eyes.

- Figure 11. KL strategy ablation: qualitative comparison (Case 1). Visual comparison across different KL strategies (no KL, Fixed KL, Step-wise KL, Moving KL, Dual Moving KL) on a prompt describing a fatigued soldier. Dual Moving KL produces more realistic facial details, better dirt and grime rendering, and maintains temporal consistency across frames compared to other variants.

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Dual Mov KL

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Moving KL

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Step-wise KL

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Fixed KL

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

|No KL|
|---|

Prompt: …A canning workshop is completely engulfed in a spreading fire, the entire scene rendered in the turbulent, emotional style of Vincent van Gogh. The flames, composed of thick, swirling, heavy brushstrokes of blazing yellow and fiery orange…Below, panicked workers are depicted not as individuals but as forms of pure movement and terror; their bodies are captured in a frantic, vibrant, chaotic jumble of colors.

- Figure 12. KL strategy ablation: qualitative comparison (Case 2). Additional visual comparison demonstrating how different KL strategies affect generation quality. Dual Moving KL consistently achieves better photorealism and alignment with prompt descriptions compared to alternatives, validating the position-velocity control mechanism discussed in Section 3.2.2.

