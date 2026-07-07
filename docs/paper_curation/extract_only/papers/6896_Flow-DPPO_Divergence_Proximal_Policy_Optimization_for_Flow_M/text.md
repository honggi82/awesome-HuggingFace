# arXiv:2606.11025v2[cs.LG]27Jun2026

[Figure 1]

## Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models

Bowen Ping1,2,∗ Xiangxin Zhou2,∗ ¶ Penghui Qi3 Minnan Luo1,‡ Liefeng Bo2 Tianyu Pang2,‡ 1Xi’an Jiaotong University 2Tencent Hunyuan 3National University of Singapore

∗Equal contribution ¶Project Lead ‡Corresponding author

Abstract. Recent work has demonstrated that online reinforcement learning (RL) can substantially improve the quality and alignment of flow matching models for image and video generation. Methods such as Flow-GRPO and CPS cast the denoising process as a Markov Decision Process and apply PPO-style ratio clipping to enforce a trust region. However, we argue that ratio clipping is structurally ill-suited for flow models: the probability ratio between new and old policies is a noisy, single-sample estimate of the true policy divergence, leading to over-constraining in some regions of the trajectory and under-constraining in others. We propose Flow-DPPO (Flow Divergence Proximal Policy Optimization), which replaces ratio clipping with a divergence proximal constraint. A key observation is that the per-step policy in flow models is Gaussian, enabling exact and cheap computation of the KL divergence between old and new policies. Flow-DPPO employs an asymmetric divergence mask that blocks gradient updates only when they simultaneously move away from the trusted region and violate the divergence threshold. Experiments show that Flow-DPPO achieves higher rewards with better KL-proximal efficiency, alleviates catastrophic forgetting, promotes balanced multi-objective optimization, and enables stable multi-epoch training where ratio clipping degrades.

Date: June 8, 2026 Code: https://github.com/Tencent-Hunyuan/UniRL/tree/main/FlowDPPO

### 1 Introduction

Reinforcement learning (RL) has emerged as a core paradigm for aligning models with downstream objectives. In language models, RL methods such as DPO (Rafailov et al., 2023) and GRPO (Shao et al., 2024) have substantially improved alignment (Ouyang et al., 2022) and reasoning capabilities (Guo et al., 2025). Recently, these advances have been extended to image and video generation Liu et al. (2025); Wallace et al. (2024); Wang and Yu (2025); Xue et al. (2025a); Zheng et al. (2026), where flow matching models (Lipman et al., 2023; Liu et al., 2023) represent the dominant generative framework. Among them, Flow-GRPO (Liu et al., 2025) and DanceGRPO (Xue et al., 2025b)

###### FLUX.1-dev Flow-GRPO Flow-CPS GRPO-Guard Flow-DPPO

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

seven green croissants

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

a blue dog on top of three white sheep behind seven white candles

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

a blue giraffe behind seven pink clocks to the right of an elephant

- Figure 1: Qualitative comparison on FLUX.1-dev (Black Forest Labs, 2024) with GenEval2 (Kamath et al., 2025) prompts. Flow-DPPO achieves competitive compositional accuracy with notably less image quality degradation compared to Flow-GRPO (Liu et al., 2025), Flow-CPS (Wang and Yu, 2025), and GRPO-Guard (Wang et al., 2025), reflecting their superior KL-proximal efficiency.

demonstrated strong performance by transforming deterministic ODE sampling into stochastic SDE trajectories and introducing PPO-style ratio clipping to enforce trust-region optimization.

The theoretical foundation of trust-region methods originates from Trust Region Policy Optimization (TRPO) (Schulman et al., 2015), which establishes a policy improvement bound: monotonic improvement is guaranteed when policy updates remain within a trust region defined by the divergence between the old and new policies. PPO (Schulman et al., 2017) later introduced ratio clipping as a computationally efficient first-order approximation to TRPO. However, as noted by Qi et al. (2026), each clipping decision is based on a single-sample Monte Carlo estimate of the true Total Variation (TV) divergence, rather than the divergence itself. In the continuous and high-dimensional latent space of flow models, this estimation noise becomes substantially amplified, leading to a systematic left shift in the ratio distribution, with its mean falling below one (Wang et al., 2025). We show that this bias is intrinsic to Gaussian policies: the standard PPO clipping range [1−ϵ,1+ϵ] therefore becomes effectively asymmetric, failing to adequately constrain over-optimization for positive-advantage samples while excessively clipping negative-advantage ones.

To mitigate this bias, GRPO-Guard (Wang et al., 2025) proposed normalizing the ratio distribution. While this re-centering alleviates the symptom, it does not address the root cause: the ratio remains a noisy, per-sample proxy for the true policy divergence. We observe that flow models offer a structural advantage that sidesteps this problem entirely. Because each per-step policy is Gaussian with a mean µθ determined by the velocity network and a fixed, schedule-dependent variance σ, the KL divergence between old and new policies reduces to ∥µθold −µθ∥2/(2σ2), which is an exact, deterministic quantity that can be computed from two forward passes already performed during training. Unlike the LLM setting, where DPPO (Qi et al., 2026) must resort to approximate divergence reductions over large vocabularies, flow models admit exact divergence computation at no additional cost. This motivates replacing ratio clipping with a direct KL-proximal trust region constraint.

Building on this insight, we propose Flow-DPPO (Flow Divergence Proximal Policy Optimization), which replaces ratio clipping with a divergence-based mask. The mask blocks gradient updates only when two conditions are jointly met: (1) the advantage and ratio indicate that the update is moving the policy away from the old policy, and (2) the exact KL divergence already exceeds a threshold. This design directly enforces the trust region while preserving the beneficial asymmetric structure of PPO: updates that move the policy towards the old policy are never blocked, accelerating recovery from overshooting. Extensive experiments on various base models demonstrate that Flow-DPPO achieves superior reward optimization, improved KL-proximal efficiency, stronger robustness to catastrophic forgetting, balanced multi-objective optimization that mitigates reward hacking, and stable multi-epoch training that enables higher sample efficiency. Figure 1 presents qualitative generation results demonstrating that Flow-DPPO achieves competitive compositional accuracy while preserving notably higher visual quality than existing methods.

### 2 Preliminaries

Flow matching (Lipman et al., 2023; Liu et al., 2023) learns a continuous-time velocity field that transports samples from a simple source distribution to the data distribution. Specifically, let x0 ∼ π0 = pdata, and define an interpolating path xt = αtx0 + σtϵ with ϵ ∼ N(0,I), where αt and σt determine the probability path between data and noise. This construction induces a conditional distribution πt|0(xt | x0) = N(αtx0,σt2I). The goal of flow matching is to train a time-dependent vector field vθ(xt,t) to match the target velocity, which is given by v = ddxtt = α˙tx0 + σ˙tϵ and the functinal f˙t := dft/dt. The model vθ(xt,t) is then trained by minimizing the regression objective

Et,x0∼π0,ϵ∼N(0,I) w(t)∥vθ(xt,t) − v∥22 , (1) where w(t) is a weighting function. After training, samples are generated by solving the ODE

dxt

dt = vθ(xt,t). In practice, simple numerical solvers such as Euler discretization are often sufficient for high-quality sampling (Karras et al., 2022; Lu et al., 2022; Song et al., 2021a). A notable special case is rectified flow (Liu et al., 2023), which uses the linear conditional path αt = 1 − t and σt = t. Under this choice, the target velocity reduces to v = ϵ−x0. We adopt this linear schedule throughout the paper.

#### 2.1 RL Fine-Tuning for Flow Matching Models

For text-conditional flow matching models, given a conditioning prompt c, generation starts from a Gaussian latent xT ∼ N(0,I) and progressively transforms it into a clean sample x0. At each timestep t, the flow model predicts a velocity field vθ(xt,t,c), which specifies a deterministic generation direction. Applying RL algorithms such as GRPO (Shao et al., 2024) to flow matching models requires a sampler-induced stochastic policy at each denoising step. Flow-GRPO (Liu et al., 2025) constructs such a policy via an ODE-to-SDE conversion, which transforms the probability-flow ODE into an equivalent SDE with the same marginals (Albergo and Vanden-Eijnden, 2023; Albergo et al., 2024; Song et al., 2021b): dxt = vθ(xt,t,c) + σ

2 t

2t xt + (1 − t)vθ(xt,t,c) dt + σt dw, where

dw denotes Wiener process increments, σt = a 1−t t, and a is a scalar hyperparameter controlling the noise level. Applying Euler–Maruyama discretization yields the Flow-SDE sampler:

σt2 2t

xt−∆t = xt + vθ(xt,t,c) +

xt + (1 − t)vθ(xt,t,c) ∆t + σt

√

∆tϵ, (2)

with ϵ ∼ N(0,I). An alternative is Coefficients-Preserving Sampling (CPS) (Wang and Yu, 2025), which reduces the excessive noise injection in Flow-SDE and better preserves the interpolation structure of the scheduler. Let xˆ0 = xt − tvˆθ(xt,t,c),xˆ1 = xt + (1 − t)vˆθ(xt,t,c) denote the predicted clean sample and noise component, respectively. CPS updates the latent as

ηπ 2

ηπ 2

ϵ, (3)

xt−∆t = (1 − (t − ∆t))xˆ0 + (t − ∆t)cos

x ˆ1 + (t − ∆t)sin

where ϵ ∼ N(0,I) and η ∈ [0,1] controls the stochasticity. Both Flow-SDE and CPS therefore induce Gaussian per-step policies written as

pθ(xt−∆t | xt,t,c) = N xt−∆t; µθ(xt,t,c), σ2(t)I , (4)

where the specific forms of µθ and σ(t) depend on the sampler. The above generative process can be formulated as a finite-horizon Markov Decision Process (MDP) (Black et al., 2024; Fan et al., 2023; Liu et al., 2025). To distinguish the discrete decision process from the underlying continuous-time flow, we use k ∈ {1,...,K} for the MDP state index and t ∈ [0,1] for the reverse-time variable of the flow. Let 0 = tK < tK−1 < ··· < t1 = 1 be a discretization of reverse time, so that state k corresponds to flow time tk. The state at step k is sk = (c,tk,xtk). Note that tk − tk+1 = ∆t. For k = 1,...,K − 1, the action is the next latent sample, ak = xtk+1, drawn from the sampler-induced policy πθ(ak | sk) = πθ(xtk+1 | xtk,tk,c). Given the sampled action, the transition is deterministic, with next state sk+1 = (c,tk+1,xtk+1). The rollout starts from c ∼ p(c) and xt1 ∼ N(0,I), and terminates at k = K where tK = 0.

After the full generative process, a scalar reward R(x0,c) is provided. RL fine-tuning maximizes the expected terminal reward with a KL regularization term that penalizes deviation from the pretrained

reference policy πref: maxθ Ec∼p(c), τ∼πθ R(x0,c) − β Kk=1−1 DKL πθ(· | sk)∥πref(· | sk) , where τ = (s1,a1,s2,a2,...,sK−1,aK−1,sK) denotes a trajectory induced by πθ and β ≥ 0 controls the regularization strength. This KL penalty discourages reward hacking and mitigates catastrophic forgetting of the pretrained model’s capabilities.

Flow-GRPO (Liu et al., 2025) applies GRPO to the above MDP. Given a prompt c, the current policy generates a group of G samples {xi0}Gi=1. Their rewards are normalized within the group to obtain relative advantages: Aˆi = R(xi0,c) − mean({R(xj0,c)}Gj=1) /std({R(xj0,c)}Gj=1). In practice, each policy optimization iteration begins by rolling out a batch of data, which is then split into several minibatches for multiple gradient steps. This procedure introduces policy staleness: after the first update, the optimizing policy has already diverged from the behavior policy that generated the data. To control this off-policy drift, a trust region mechanism is applied. Following PPO (Schulman et al., 2017), the policy is optimized using the clipped surrogate objective

LFlow-GRPO(θ) = E

G

K

1 G

1 K

i=1

k=1

min rki (θ)Aˆi, clip(rki (θ),1 − ϵ,1 + ϵ)Aˆi , (5)

where we omit the KL penalty term DKL(πθ ∥πref) for brevity, and the per-step importance ratio is defined as rki (θ) = pθ(x

i tk−∆t|xitk,c)

k−∆t|xitk,c). Since both Flow-SDE and CPS define Gaussian per-step policies as in Eq. (4), the log-ratio admits the same closed-form expression:

pθold(xit

log rki (θ) = ∥xit

k−∆t − µθold∥2 − ∥xit

k−∆t − µθ∥2 2σ2(tk)

. (6)

Therefore, both samplers can be optimized within the same GRPO framework, differing only in the parameterization of the induced stochastic policy.

### 3 Methodology

In this section, we first derive a policy improvement bound that justifies trust-region methods for flow models. Then, we show that ratio clipping is a noisy proxy for the true divergence constraint. Finally, we present Flow-DPPO, which leverages exact KL computation to enforce a deterministic divergence mask, yielding a tighter and variance-free trust-region constraint.

#### 3.1 Trust-Region Policy Optimization for Flow Matching Models

Inspired by Schulman et al. (2017); Qi et al. (2026), we adapt the trust region framework to the flow model fine-tuning setting defined in Section 2.1. This setting differs from the classical discounted RL paradigm in two important ways. First, the problem is an undiscounted episodic task with a finite horizon of K − 1 decision steps. Second, due to the terminal reward structure, advantages are estimated at the trajectory level rather than per step. These properties necessitate a tailored policy improvement guarantee. We follow the MDP defined in Section 2.1.

- Theorem 1 (Performance Difference Identity for Flow Models). In the finite-horizon flow model MDP with K −1 decision steps, let J(π) = Ec∼p(c), τ∼π[R(x0,c)] denote the expected reward. For any

two policies πθ and πθold, the performance difference decomposes as: J(πθ) − J(πθold) = L′θ

old

(πθ) − ∆(πθold,πθ), where the surrogate objective is

L′θ

old

(πθ) = Eτ∼πθ

old

R(x0,c)

K−1

k=1

πθ(ak | sk)

πθold(ak | sk) − 1 , (7) and the error term is

∆(πθold,πθ) = Eτ∼πθ

old

 R(x0,c)

K−1

k=1

πθ(ak | sk) πθold(ak | sk) − 1

 1 −

K−1

j=k+1

πθ(aj | sj) πθold(aj | sj)

 

 .

The surrogate L′θ

old

(πθ) represents a first-order approximation to the true improvement, while the error term ∆ captures higher-order interactions between per-step policy changes. To yield a practical optimization objective, we bound this error term.

- Theorem 2 (Policy Improvement Bound for Flow Models). In the finite-horizon flow model MDP with K − 1 decision steps, the policy improvement is lower-bounded by:

(πθ) − 2ξ(K−1)(K−2) · DTVmax(πθold∥πθ)2, (8)

J(πθ) − J(πθold) ≥ L′θ

old

where DTVmax(πθold∥πθ) = maxsk DTV πθold(· | sk)∥πθ(· | sk) is the maximum per-step Total Variation divergence, and ξ = maxx0,c |R(x0,c)| is the maximum absolute reward.

Please refer to Appendix B for the detailed derivation; a tighter bound linear in K is given in Appendix B.3. This bound is structurally analogous to the policy improvement bound for LLMs derived in Qi et al. (2026). It provides a rigorous justification for trust-region methods in flow model fine-tuning: constraining the per-step divergence controls the penalty term and guarantees monotonic improvement. Similar to TRPO (Schulman et al., 2015), we can solve the following constrained optimization problem to ensure stable learning:

max

πθ

L′θ

(πθ), s.t. DTVmax(πθold∥πθ) ≤ δ. (9)

old

- Remark 3 (Exact Divergence in the Gaussian Setting). For the Gaussian per-step policies in Eq. (4), the TV divergence is a monotone function of the mean displacement:

DTV πθold(· | sk)∥πθ(· | sk) = 2Φ ∥µθold − µθ∥

2σ(tk) − 1, (10) where Φ is the standard normal CDF. Constraining the TV divergence below a threshold is therefore equivalent to constraining ∥µθold − µθ∥2 ≤ δ′ for an appropriate δ′, which is precisely the divergence measure that Flow-DPPO employs. Moreover, the Pinsker inequality DTV(p∥q)2 ≤ 12DKL(p∥q) ensures that our KL-based constraint also upper-bounds the TV divergence: when the per-step DKL ≤ δ, we have DTVmax ≤ δ/2. In the Gaussian equal-covariance case, the converse also holds since KL and TV are both monotone functions of ∥µθold − µθ∥/σ. Thus, our method is theoretically justified from both the KL and TV perspectives. Unlike the LLM setting, where the discrete vocabulary requires approximate divergence computations (Qi et al., 2026), the Gaussian structure of flow models provides exact per-step divergence at zero additional cost.

#### 3.2 Pitfalls of Ratio Clipping in Flow-GRPO

Flow-GRPO adopts PPO-style ratio clipping to enforce a trust region. For consistency with the Flow-GRPO notation (Liu et al., 2025), in this and the following subsections we index denoising steps by the flow time t (equivalently, t = tk in the MDP indexing of Section 2.1). The clipping condition |rti − 1| ≤ ϵ is intended to prevent the new policy from deviating too far from the old one. However, the probability ratio is a fundamentally noisy proxy for the true policy divergence. By definition of the Total Variation divergence,

- 1

- 2

|rti − 1| , (11)

Ext−∆t∼πθ

DTV πθold(· | xt)∥πθ(· | xt) =

old

so each individual |rti − 1| is merely a single-sample Monte Carlo estimate of 2DTV. While the policy improvement bound (Theorem 2) calls for constraining DTVmax, ratio clipping constrains this noisy per-sample surrogate instead. This issue was identified by Qi et al. (2026) in the LLM setting; we now show that the resulting pathology is particularly severe in flow models due to the high-dimensional continuous action space.

Recall from Eq. (6) that the log-ratio is:

log rti(θ) = ∥xit−∆t − µθold∥2 − ∥xit−∆t − µθ∥2

. (12)

2σ2

Since xit−∆t is sampled from N(µθold,σ2I), we can write xit−∆t = µθold + σϵ where ϵ ∼ N(0,I). Substituting and letting d = µθ − µθold:

2σ ϵ⊤d − ∥d∥2 2σ2

#### ϵ⊤d

log rti(θ) = ∥σϵ∥2 − ∥σϵ − d∥2

∥d∥2 2σ2

. (13)

σ −

=

=

2σ2

The first term, ϵ⊤d/σ, is a zero-mean random variable with variance ∥d∥2/σ2. This reveals that the log-ratio is dominated by noise: the signal (the deterministic second term −∥d∥2/(2σ2)) is exactly the negative of the KL divergence, but it is corrupted by a noise term whose standard deviation ∥d∥/σ is of the same order as the signal itself. This analysis yields two key insights:

- 1. High variance. The ratio rti is inherently noisy due to the stochastic sample ϵ. Even when the true KL divergence ∥d∥2/(2σ2) is moderate, individual ratio samples can be extreme (either very large or very small), triggering spurious clipping.

- 2. Noise-dependent clipping. Whether an update is clipped depends heavily on the random noise ϵ drawn during sampling, rather than the true policy divergence. Two trajectories with identical policy parameters but different noise realizations may receive entirely different clipping decisions.

In contrast, the true KL divergence DKL(πθold∥πθ) = ∥d∥2/(2σ2) is a deterministic function of the policy parameters alone, unaffected by the sampling noise. This motivates our approach: replace the

noisy ratio-based trust region with a direct divergence constraint. A detailed variance analysis is provided in Appendix D.

- 3.3 Divergence Proximal Policy Optimization for Flow Models

We now derive the divergence between old and new policies in the flow model setting and present our Flow-DPPO algorithm.

Exact KL divergence. Since both πθold(· | xt) and πθ(· | xt) are Gaussians with the same variance σ2I but different means, the KL divergence admits the closed form (see Appendix C for derivation):

DKL πθold(· | xt)∥πθ(· | xt) = ∥µθold(xt,t) − µθ(xt,t)∥2

. (14)

2σ2

For Flow-SDE (corresponding to Eq. (2)), σ2 = σt2∆t, giving:

2

σt(1 − t) 2t

∆t 2

1 σt

DKLSDE(πθold∥πθ) =

∥vθ(xt,t) − vθold(xt,t)∥2. (15) For CPS (corresponding to Eq. (3)), with σCPS = (t − ∆t)sin(ηπ/2):

+

DKLCPS(πθold∥πθ) = ∥µCPSθ (xt,t) − µCPSθ

(xt,t)∥2 2(t − ∆t)2 sin2(ηπ/2)

. (16)

old

- Remark 4. In the LLM setting, DPPO (Qi et al., 2026) must approximate the true divergence via Binary or Top-K reductions of the vocabulary distribution, as computing exact TV or KL over |V| > 100K tokens is memory-prohibitive. In flow models, the Gaussian policy structure yields exact divergence at negligible cost, namely the squared difference between two forward passes of the velocity network. This makes divergence-based trust regions strictly more natural for flow models than for LLMs. The Flow-DPPO mask. We define the Flow-DPPO objective as:

LFlow-DPPO(θ) = E

T−1

G

1 G

1 T

t=0

i=1

Mti · rti(θ) · Aˆi − βDKL(πθ∥πref) , (17)

where the divergence-based mask is:

 

- 0, if A ˆi > 0 and rti > 1 and Dt > δ or A ˆi < 0 and rti < 1 and Dt > δ ,
- 1, otherwise,

Mti =

(18)



with Dt ≡ DKL πθold(· | xit)∥πθ(· | xit) and δ a divergence threshold. Asymmetric design. The mask in Eq. (18) preserves the asymmetric structure that makes PPO effective. It only blocks updates that are already moving away from the old policy:

- • When Aˆi > 0 and rti > 1: the gradient is pushing the policy further from θold (increasing an already-increased action probability). The mask blocks this if the divergence exceeds δ.
- • When Aˆi < 0 and rti < 1: the gradient is decreasing an already-decreased action probability, again moving away from the old policy. The mask blocks this if divergence exceeds δ.
- • In all other cases (Aˆi > 0,rti < 1 or Aˆi < 0,rti > 1): the gradient is moving the policy towards the old policy. These beneficial updates are never blocked, regardless of the divergence level.

This asymmetry ensures that the trust region constraint does not impede recovery: when the policy has drifted too far, corrective updates remain uninhibited. We provide a justification of this directional condition and discuss refined mask variants in Appendix E.

- 4 Experiments Models and Baselines. We employ Stable Diffusion 3.5 Medium (Esser et al., 2024; Stability AI,

- 2024) (SD3.5), FLUX2-klein-base-9B (Black Forest Labs, 2026) (FLUX2-9B) and FLUX.1-dev (Black Forest Labs, 2024) as base models to cover diverse architectures and scales. We compare our method against four competitive baselines: Flow-GRPO (Liu et al., 2025), Flow-CPS (Wang and Yu,
- 2025), GRPO-Guard (Wang et al., 2025) and Diffusion-NFT (Zheng et al., 2026). Specifically, we evaluate two variants of our approach: Flow-DPPO (using SDE sampling from Flow-GRPO) and Flow-DPPO+CPS (using CPS-scheduled SDE sampling). Detailed configurations are deferred to Appendix F.

Metrics and Datasets. GenEval2 (Kamath et al., 2025) and PickScore (Kirstain et al., 2023) are selected as in-domain and out-of-domain (OOD) datasets, respectively. For GenEval2, we follow the official template to generate 20k synthetic training prompts and evaluate on the 800 officially released prompts. To monitor catastrophic forgetting under distribution shifts, we track PickScore (Kirstain et al., 2023), CLIP (Radford et al., 2021) score, and HPSv2 (Wu et al., 2023) during training. We report results for both single-reward optimization (GenEval2 only) and multi-reward training, where GDPO (Liu et al., 2026) aggregates advantages with equal reward weights.

| | | | | |
|---|---|---|---|---|
| | | | | |
| |GenEval2| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| |CLIP| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| |HPSv2| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | |
|---|---|---|---|
| |PickScore| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

2.5

0. 3

0.9

0.3

0.8

0.32

2

0.28

0.7

0.31

0.26

- 20

- 20.5
- 21

- 21.5

0.6

0.3

0.24

0.5

0.29

0. 2

0.4

0.3

0.2

0.28

0.2

0.18

0 1, 0 2, 0

0 1, 0 2, 0

0 1, 0 2, 0

0 1, 0 2, 0

GPU Hours GPU Hours GPU Hours GPU Hours

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS

##### Figure 2: Training curves on FLUX2-9B for single-reward setting. Flow-DPPO variants achieve state-of-the-art performance and less catastrophic forgetting on out-of-domain rewards.

Table 1: Performance comparison on SD3.5 and FLUX2-9B. The training is applied on the In-Domain (GenEval2). The Out-of-Domain (PickScore) prompts are only used for evaluation. The corresponding training curves are in Figures 4 and 8. The full version including single-reward training is in Table 4.

In-Domain (GenEval2) Out-of-Domain (PickScore) Model GenEval2 CLIP PickScore HPSv2 CLIP PickScore HPSv2 Pretrained baselines (before RL)

SD3.5-medium 12.4 0.250 21.00 0.213 0.244 19.99 0.210 FLUX2-klein-base-9B 25.4 0.281 20.92 0.228 0.254 20.05 0.230 FLUX.1-dev 23.3 0.297 23.26 0.315 0.276 21.91 0.304

SD3.5-medium, multi-reward RL fine-tuning

Flow-GRPO 39.9 0.358 25.09 0.399 0.273 22.07 0.349 Flow-CPS 44.6 0.359 25.51 0.407 0.265 22.08 0.343 GRPO-Guard 47.8 0.353 25.64 0.409 0.272 22.32 0.354 Diffusion-NFT 42.5 0.334 25.30 0.394 0.269 22.52 0.355 Flow-DPPO 48.1 0.345 25.63 0.409 0.273 22.58 0.360 Flow-DPPO + CPS 51.6 0.369 25.72 0.415 0.279 22.51 0.361

FLUX2-klein-base-9B, multi-reward RL fine-tuning

Flow-GRPO 46.8 0.371 25.61 0.412 0.277 22.62 0.357 Flow-CPS 47.1 0.361 25.70 0.416 0.276 22.85 0.364 GRPO-Guard 49.0 0.375 25.27 0.411 0.269 21.99 0.349 Diffusion-NFT 47.3 0.336 24.87 0.389 0.274 22.47 0.351 Flow-DPPO 57.7 0.364 25.76 0.418 0.282 22.90 0.368 Flow-DPPO + CPS 55.2 0.386 26.15 0.427 0.287 22.97 0.370

#### 4.1 Main results

Performance and Generalization. As summarized in Table 1, Flow-DPPO variants consistently outperform all baselines across both base models and all evaluation metrics, with particularly substantial gains in the GenEval2 reward. In the single-reward setting (optimizing GenEval2 only),

- Figure 2 demonstrates that our proposed variants not only achieve superior performance on FLUX2-9B compared to baselines but also exhibit a more stable training trajectory. These empirical advantages persist across SD3.5 (Figure 7) and FLUX.1-dev (Figure 9).

We attribute this superiority to the precise divergence-based mask in Flow-DPPO. By mitigating the influence of samples falling outside the trust region, which are susceptible to reward hacking, Flow-DPPO maintains a more robust optimization gradient. This constraint prevents the model from excessively exploiting individual rewards at the expense of others, thereby achieving a superior balance across multiple optimization objectives and fostering stable convergence. This is further corroborated by the multi-reward training curves in Figure 4, where Flow-DPPO variants consistently outperform all baselines across most metrics on SD3.5, without sacrificing any individual objective.

Out-of-domain Behavior and Catastrophic Forgetting. To investigate catastrophic forgetting, we analyze OOD metrics (PickScore, CLIP, and HPSv2) and the KL divergence from the pretrained model. As illustrated in Figure 2, OOD metrics initially increase across all methods as RL optimization drives the model toward higher visual quality. However, as training progresses, these metrics decline, indicating that the model overfits the in-domain reward (GenEval2) at the expense of OOD knowledge. Notably, Flow-DPPO variants exhibit significantly less OOD degradation, suggesting that catastrophic forgetting is effectively mitigated. Qualitative results in Figure 6 further support this, demonstrating that our methods better preserve visual fidelity on OOD prompts. Consistently, Table 2 shows that Flow-DPPO variants maintain a lower KL divergence in most settings. This reduced distribution drift aligns with OOD metric trends, collectively indicating stronger resistance to reward hacking and forgetting. Ultimately, these results highlight that the

- Figure 3: Asymmetric masking ablation on SD3.5 with single-reward on GenEval2.

0 50 1, 0 1,50 2, 0 2,50

0

0.2

0.4

0.6

0.8

- 1e-6 1e-5

- 1e-7 w/o asy metry

Training Epoch

GenEval2

Table 2: KL divergence (×10−3) between the RL fine-tuned model and the pre-trained reference. Lower is better. Full curves in Figure 12.

FLUX2-9B SD3.5 Method Single Multi +CFG Single Multi Flow-SDE schedule

Flow-GRPO 0.77 0.79 1.36 2.34 3.81 GRPO-Guard 1.07 1.01 1.63 2.05 3.33 Flow-DPPO 0.17 0.49 0.51 1.16 2.49

CPS schedule

Flow-CPS 0.24 1.66 1.51 2.41 3.18 Flow-DPPO + CPS 0.68 0.70 0.83 1.60 2.52

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |GenEva| |l2| | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| |CLIP| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| |HPSv2| | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| |PickSco| |re| | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0 50 1, 0 1,50

- 0.1
- 0.2
- 0.3
- 0.4

0 50 1, 0 1,50

0.24

0.26

0.28

0.3

0.32

0.34

0.36

0.38

0 50 1, 0 1,50

0.2

0.25

0.3

0.35

0.4

0 50 1, 0 1,50

21

2

- 23
- 24
- 25
- 26

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS Difusion-NFT

GPU Hours GPU Hours GPU Hours GPU Hours

- Figure 4: Training curves on SD3.5 for multi-reward setting. Flow-DPPO variants consistently outperform all baselines across all metrics.

divergence-based mask acts as a safety boundary, allowing the model to learn from rewards without losing its original generative quality or falling into distribution collapse.

#### 4.2 Analysis

Asymmetric Masking and Divergence Threshold. We investigate the impact of the divergence threshold and asymmetric masking in Flow-DPPO using SD3.5 with CPS sampling (Figure 3). Without asymmetric masking, the training process collapses as the trust-region regularization becomes ineffective; specifically, samples falling outside the trust region are largely ignored, preventing optimization progress. Conversely, asymmetric masking constrains these samples back within the trust region, thereby stabilizing the trajectory. Regarding the divergence threshold, a looser threshold (10−5) results in diminished stability and suboptimal convergence. A tighter threshold (10−7) initially slows down learning but fosters superior stability and slightly better final performance due to more rigorous trust-region enforcement.

Multi-epoch Training and Sample Efficiency. Given the high computational cost of rollouts, we investigate how sample reuse frequency affects optimization efficiency on SD3.5. Specifically, we vary two factors: (i) the number of groups per rollout, and (ii) the number of training epochs per rollout (inner loops). The latter determines the reuse frequency of each sample. For instance, two inner loops imply that each rollout batch is utilized for two consecutive gradient steps.

| | | | | | |
|---|---|---|---|---|---|
|Flow-GR Flow-D|PO PO<br><br>| | | | |
|G64-I1<br><br>| | | | | |
|G32-I2 G64-I2<br><br>| | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.7

0.6

GenEval2

0.5

0.4

0.3

0.2

0.1

0 50 1, 0 1,50 2, 0 2,50

Training Epoch

- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
- 0.6
- 0.7
- 0.8 Flow-CPS Flow-DPO + CPS G64-I1 G32-I2 G64-I2

GenEval2

0

0 50 1, 0 1,50 2, 0 2,50

Training Epoch

- Figure 5: Multi-epoch training on SD3.5 (Left: Flow-SDE, Right: CPS). Flow-DPPO variants show consistent long-term gains under multi-epoch training (G64-I2 and G32-I2), while baselines plateau or even degrade.

FLUX2-9B Flow-GRPO Flow-CPS GRPO-Guard Flow-DPPO Flow-DPPO+CPS

In-Domain

a stone pig in background, two black cats in front of the pig, and six yellow horses in front

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

five colorful bicycles below, two stone guitars above them, and a penguin at the highest point

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Out-of-Domain

a sun elf with a bow, facing the camera, in a jungle

waterfall scene

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

people at a barbecue in Brazil, captured in HD, Canon EOS 5D Mark IV DSLR

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

- Figure 6: Qualitative comparison on FLUX2-9B with single-reward setting and controlled seeds for each prompt at the same training iteration. Flow-DPPO and Flow-DPPO + CPS retain competitive in-domain performance with less reward hacking while exhibiting notably less catastrophic forgetting on out-of-domain prompts.

While our main experiments use 64 groups with 1 inner loop (G64-I1), we further explore two efficiency-oriented settings: G32-I2 (half the rollout computation with samples reused twice) and G64-I2 (standard rollout computation with doubled training intensity). As shown in Figure 5, baseline methods (Flow-GRPO, Flow-CPS) struggle to achieve sustained gains under multi-epoch training, often leading to performance plateaus or degradation. In contrast, Flow-DPPO variants successfully reuse rollout samples across multiple updates, yielding consistent long-term performance

improvements. This advantage stems from the divergence-based mask, which constrains updates within the trust region, ensuring efficient sample utilization. This offers a promising direction for scenarios where rollouts are computationally expensive, such as long-video generation.

### 5 Conclusion

We show ratio clipping in flow models is a noisy, biased proxy for divergence. To address this, we propose a divergence-based mask using the exact KL at zero extra cost. Across multiple base models, sampling schedules, and reward objectives, Flow-DPPO consistently achieves superior performance than baselines in terms of reward optimization and catastrophic forgetting. Furthermore, Flow-DPPO enables stable multi-epoch training where ratio clipping degrades, offering a promising direction for scenarios with expensive rollouts, such as long-video generation.

### References

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023.

Michael Samuel Albergo, Mark Goldstein, Nicholas Matthew Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. In International Conference on Machine Learning, pages 921–937. PMLR, 2024.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024.

Black Forest Labs. Flux.1: Announcing black forest labs. https://blackforestlabs.ai/announc ing-black-forest-labs/, 2024.

Black Forest Labs. FLUX.2 [klein]: Towards interactive visual intelligence. https://bfl.ai /blog/flux2-klein-towards-interactive-visual-intelligence, 2026. Model weights: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the nineteenth international conference on machine learning, pages 267–274, 2002.

Amita Kamath, Kai-Wei Chang, Ranjay Krishna, Luke Zettlemoyer, Yushi Hu, and Marjan Ghazvininejad. Geneval 2: Addressing benchmark drift in text-to-image evaluation. arXiv preprint arXiv:2512.16853, 2025.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, 2022.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group reward-decoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, 2022.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Penghui Qi, Xiangxin Zhou, Zichen Liu, Tianyu Pang, Chao Du, Min Lin, and Wee Sun Lee. Rethinking the trust region in llm reinforcement learning. arXiv preprint arXiv:2602.04879, 2026.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pages 1889–1897. PMLR, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International

- Conference on Learning Representations, 2021a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International

- Conference on Learning Representations, 2021b.

Stability AI. Stable diffusion 3.5. https://stability.ai/news/introducing-stable-diffusion -3-5, 2024. Model weights: https://huggingface.co/stabilityai/stable-diffusion-3.5-m

edium.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024.

Feng Wang and Zihao Yu. Coefficients-preserving sampling for reinforcement learning with flow matching. arXiv preprint arXiv:2509.05952, 2025.

Jing Wang, Jiajun Liang, Jie Liu, Henglin Liu, Gongye Liu, Jun Zheng, Wanyuan Pang, Ao Ma, Zhenyu Xie, Xintao Wang, et al. Grpo-guard: Mitigating implicit over-optimization in flow matching via regulated clipping. arXiv preprint arXiv:2510.22319, 2025.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

Shuchen Xue, Chongjian Ge, Shilong Zhang, Yichen Li, and Zhi-Ming Ma. Advantage weighted matching: Aligning rl with pretraining in diffusion models. arXiv preprint arXiv:2509.25050, 2025a.

Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025b.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online diffusion reinforcement with forward process. In The Fourteenth International Conference on Learning Representations, 2026.

## Appendix of Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models

- A The Flow-DPPO Algorithm 16
- B Policy Improvement Bound for Flow Models 16

- B.1 Proof of Performance Difference Identity . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.2 Proof of Policy Improvement Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- B.3 A Tighter Policy Improvement Bound . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B.4 Connection to Gaussian Per-Step Divergence . . . . . . . . . . . . . . . . . . . . . . 20

- C KL Divergence Between Gaussian Policies 20

- C.1 General Gaussian KL Divergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C.2 Connection Between KL and TV in the Gaussian Setting . . . . . . . . . . . . . . . . 20
- C.3 Application to Flow-SDE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.4 Application to CPS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- D Ratio Variance Analysis 21
- E Towards a Predictive Divergence Mask 22

- E.1 Predicting Post-Update Divergence . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- E.2 The Predictive Mask . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- E.3 Discussion on Mask Variants . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- F Experimental Details 24

- F.1 Computational Resources. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- F.2 Hyperparameters. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24

- G Additional Experimental Results 25

- G.1 Additional Training Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- G.2 KL Divergence Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- G.3 Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27

- G.3.1 Classifier-Free Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- G.3.2 Reference KL Regularization Strength . . . . . . . . . . . . . . . . . . . . . . 28

- G.4 Quantitative Summary on GenEval2 . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

### Appendix A. The Flow-DPPO Algorithm

We summarize the complete Flow-DPPO training procedure. The algorithm adopts the CPS sampling framework (Wang and Yu, 2025) for trajectory generation, uses group-relative advantage estimation, and applies the divergence-based mask during policy optimization.

Algorithm 1 Flow-DPPO Training

- 1: Input: Flow model vθ, reference model vref, reward function R, prompts C
- 2: Hyperparameters: group size G, divergence threshold δ, KL coefficient β, stochasticity η
- 3: for each training iteration do
- 4: Sample prompts {cj} ∼ C
- 5: // Rollout phase (with θold)
- 6: for each prompt cj do
- 7: Generate G trajectories {(xiT,...,xi0)}Gi=1 via Flow-SDE(Eq. (2)) or CPS (Eq. (3)) using vθold
- 8: Record log-probabilities log pθold(xit−∆t | xit) and means µθold(xit,t)
- 9: Compute rewards R(xi0,cj) and advantages Aˆi
- 10: end for
- 11: // Policy optimization phase
- 12: for each gradient step do
- 13: Compute current means µθ(xit,t) via forward pass of vθ
- 14: Compute divergence Dt = ∥µθold(xit,t) − µθ(xit,t)∥2
- 15: Compute ratio rti(θ) from log-probabilities
- 16: Compute mask Mti (Eq. (18))
- 17: Update θ by maximizing LFlow-DPPO (Eq. (17))
- 18: end for
- 19: θold ← θ
- 20: end for

Computational overhead. The divergence computation requires one additional forward pass of the velocity network to obtain µθ(xit,t) at training time. However, this forward pass is already required for computing the log ratio, so the divergence Dt = ∥µθold − µθ∥2 comes at zero additional cost: it is simply the squared norm of a difference that is already computed.

### Appendix B. Policy Improvement Bound for Flow Models

We adapt the classical policy improvement theory (Kakade and Langford, 2002; Schulman et al., 2015) to the finite-horizon, undiscounted setting of flow model denoising, following the approach of Qi et al. (2026) for the LLM regime. We use the MDP notation introduced in Section 2.1: K − 1 decision steps indexed by k ∈ {1,...,K −1}, states sk = (c,tk,xtk), actions ak = xtk+1, and terminal reward R(x0,c).

#### B.1 Proof of Performance Difference Identity

- Proof [Proof of Theorem 1] We begin by expressing the performance difference via its definition. Since the reward is only a function of the terminal state x0 and the prompt c, we have:

J(πθ) − J(πθold) = Eτ∼πθ[R(x0,c)] − Eτ∼πθ

old

[R(x0,c)]

= πθ(τ | c) − πθold(τ | c) R(x0,c)dτ,

where the integral is over all trajectories τ = (a1,...,aK−1) (we omit the deterministic transition structure for notational clarity).

The core of the proof is the telescoping identity for the difference in trajectory probabilities. Since πθ(τ | c) = Kk=1−1 πθ(ak | sk), we apply the algebraic identity Nk=1 ak − Nk=1 bk =

k−1 j=1 bj (ak − bk) Nj=k+1 aj :

N k=1

 

  · πθ(ak | sk) − πθold(ak | sk)

 

 .

K−1

k−1

K−1

πθ(τ | c) − πθold(τ | c) =

πθ(aj | sj)

πθold(aj | sj)

j=1

k=1

j=k+1

Substituting into the performance difference and converting to an expectation under πθold:

 

 

 .

 R(x0,c)

K−1

K−1

πθ(aj | sj) πθold(aj | sj)

πθ(ak | sk) πθold(ak | sk) − 1

J(πθ) − J(πθold) = Eτ∼πθ

old

k=1

j=k+1

We decompose this expression by adding and subtracting the term where the future ratio product is set to 1:

K−1

πθ(ak | sk) πθold(ak | sk) − 1

J(πθ) − J(πθold) = Eτ∼πθ

R(x0,c)

old

k=1

L′θ

(πθ)

old

 1 −

 

 

 R(x0,c)

K−1

K−1

πθ(aj | sj) πθold(aj | sj)

πθ(ak | sk) πθold(ak | sk) − 1

− Eτ∼πθ

.

old

j=k+1

k=1

∆(πθold,πθ)

This completes the proof.

#### B.2 Proof of Policy Improvement Bound

Lemma 5 (Bound on Trajectory-Level TV Divergence). Let πθold and πθ be two policies for the flow model MDP. Let πθold,>k(· | sk+1) and πθ,>k(· | sk+1) denote the distributions over future sub-trajectories (ak+1,...,aK−1) starting from state sk+1. Then:

K−1

Esj∼πθ

DTV πθold,>k(· | sk+1)∥πθ,>k(· | sk+1) ≤

DTV πθold(· | sj)∥πθ(· | sj) ,

old

j=k+1

where the expectation is over states visited under πθold starting from sk+1.

Proof Let P(τ>k) = πθold,>k(τ>k | sk+1) and Q(τ>k) = πθ,>k(τ>k | sk+1), where τ>k = (ak+1,...,aK−1). We have:

2DTV(P∥Q) = |P(τ>k) − Q(τ>k)|dτ>k =

K−1

K−1

πθold(aj | sj) −

πθ(aj | sj) dτ>k.

j=k+1

j=k+1

j−1 i=1 ai |aj −bj| Ni=j+1 bi (which

Applying the telescoping identity |a1 ···aN −b1 ···bN| ≤ Nj=1

follows from the triangle inequality) and integrating:

 

 dτ>k.

j−1

K−1

K−1

2DTV(P∥Q) ≤

πθold(ai | si) |πθold(aj | sj) − πθ(aj | sj)|

πθ(ai | si)

i=j+1

i=k+1

j=k+1

For each term indexed by j, integrating out the future actions aj+1,...,aK−1 yields 1 (since πθ is normalized), leaving:

K−1

2DTV(P∥Q) ≤

j=k+1

j−1

πθold(ai | si) |πθold(aj | sj) − πθ(aj | sj)|daj dak+1 ···daj−1.

i=k+1

The inner integral is 2DTV(πθold(· | sj)∥πθ(· | sj)), and the outer integral defines an expectation over states sj under policy πθold. Thus:

K−1

Esj∼πθ

DTV(P∥Q) ≤

old

j=k+1

DTV πθold(· | sj)∥πθ(· | sj) .

- Proof [Proof of Theorem 2] From Theorem 1, we start with the exact performance difference identity:

J(πθ) − J(πθold) = L′θ

(πθ) − ∆(πθold,πθ).

old

Our goal is to upper-bound |∆(πθold,πθ)|. We begin by bounding the reward by its maximum absolute value ξ = maxx0,c |R(x0,c)|:

|∆(πθold,πθ)|

 

 

K−1

K−1

πθ(aj | sj) πθold(aj | sj)

πθ(ak | sk) πθold(ak | sk) − 1 · 1 −

≤ ξ · Eτ∼πθ

(19)

old

j=k+1

k=1

K−1

πθ,>k(τ>k | sk+1) πθold,>k(τ>k | sk+1)

##### πθ(ak | sk) πθold(ak | sk) − 1 · Eτ>k∼πθ

Es≤k∼πθ

= ξ ·

1 −

.

old

old

k=1

The inner expectation over future sub-trajectories is exactly twice the TV divergence between future trajectory distributions:

πθ,>k(τ>k | sk+1) πθold,>k(τ>k | sk+1)

Eτ>k∼πθ

1 −

= 2DTV πθold,>k(· | sk+1)∥πθ,>k(· | sk+1) .

old

Applying Theorem 5 and bounding each term by DTVmax: DTV πθold,>k(· | sk+1)∥πθ,>k(· | sk+1) ≤ (K − 1 − k)DTVmax(πθold∥πθ).

Substituting back into Eq. (19):

K−1

Esk∼πθ

|∆(πθold,πθ)| ≤ ξ ·

k=1

K−1

= 2ξ · DTVmax

k=1

K−1

≤ 2ξ · DTVmax

k=1

= 4ξ · DTVmax2

old

Eak∼πθ

old(·|sk)

πθ(ak | sk) πθold(ak | sk) − 1 · 2(K − 1 − k)DTVmax

(K − 1 − k) · Esk∼πθ

old

2DTV πθold(· | sk)∥πθ(· | sk)

(K − 1 − k) · 2DTVmax

K−1

(K − 1 − k).

k=1

Evaluating the sum: Kk=1−1(K − 1 − k) = Km=0−2 m = (K−1)(2K−2). Therefore:

(K − 1)(K − 2)

2 · DTVmax2 = 2ξ(K − 1)(K − 2) · DTVmax2. Substituting into the performance difference identity yields the desired bound:

|∆(πθold,πθ)| ≤ 4ξ ·

(πθ) − 2ξ(K − 1)(K − 2) · DTVmax(πθold∥πθ)2. This completes the proof.

J(πθ) − J(πθold) ≥ L′θ

old

#### B.3 A Tighter Policy Improvement Bound

The quadratic dependence on the horizon K2 in Theorem 2 can be overly pessimistic. By exploiting the fact that DTV ≤ 1, we derive a tighter bound that is linear in K.

Starting from the intermediate step in Eq. (19), the inner expectation is 2DTV(πθold,>k(· | sk+1)∥πθ,>k(· | sk+1)). Instead of applying Theorem 5, we directly use the universal bound DTV ≤ 1:

K−1

πθ(ak | sk) πθold(ak | sk) − 1 · 2

Esk∼πθ

Eak∼πθ

|∆(πθold,πθ)| ≤ ξ ·

old(·|sk)

old

k=1

K−1

= 4ξ · Eτ∼πθ

DTV πθold(· | sk)∥πθ(· | sk) .

old

k=1

Combining both bounds, the policy improvement satisfies the composite guarantee:

J(πθ) − J(πθold) ≥ L′θ

old

(πθ) − min 2ξ(K−1)(K−2) · DTVmax2, 4ξ · Eτ∼πθ

old

K−1

DTV,k ,

k=1

where DTV,k = DTV(πθold(· | sk)∥πθ(· | sk)). The quadratic bound is tighter for small policy changes, while the linear bound is tighter for larger updates or longer horizons.

#### B.4 Connection to Gaussian Per-Step Divergence

For the Gaussian policies in Eq. (4), πθold(· | sk) = N(µθold,σ2(tk)I) and πθ(· | sk) = N(µθ,σ2(tk)I), the TV divergence admits the closed form:

DTV πθold(· | sk)∥πθ(· | sk) = 2Φ ∥µθold − µθ∥

2σ(tk) − 1, where Φ is the standard normal CDF. Since Φ is strictly monotonically increasing, the TV constraint DTVmax ≤ δ is equivalent to:

2

- 1 + δ

- 2

∥µθold(xtk,tk,c) − µθ(xtk,tk,c)∥2 ≤ 4σ2(tk) Φ−1

=: δ′.

max

sk

This formally establishes that the Flow-DPPO mask, which blocks updates when ∥µθold − µθ∥2 > δ, implements a trust-region constraint equivalent (up to a monotone rescaling) to constraining the

per-step TV divergence. The policy improvement bound (Theorem 2) thus provides a rigorous theoretical guarantee for Flow-DPPO: by enforcing a per-step divergence threshold, the penalty term remains controlled, ensuring monotonic policy improvement.

### Appendix C. KL Divergence Between Gaussian Policies

In this section, we derive the KL divergence between old and new policies in flow models and establish its connection to the TV divergence used in the policy improvement bound.

#### C.1 General Gaussian KL Divergence

Let p = N(µ1,σ2I) and q = N(µ2,σ2I) be two isotropic Gaussians in Rd with the same covariance. The KL divergence is:

 2(µ1 − µ2)⊤ Ep[x − µ1]

  = ∥µ1 − µ2∥2

- 1

- 2σ2

+∥µ1 − µ2∥2

. (20)

DKL(p∥q) =

2σ2

=0

Note that this is symmetric in the means: DKL(p∥q) = DKL(q∥p) when the covariances are identical.

#### C.2 Connection Between KL and TV in the Gaussian Setting For the same pair of Gaussians, the TV divergence is:

DTV(p,q) = 2Φ ∥µ1 − µ2∥ 2σ − 1.

Since both KL and TV are monotone functions of the single quantity ∥µ1 − µ2∥/σ, thresholding one is equivalent to thresholding the other. Specifically, the constraint DTV ≤ δTV is equivalent to ∥µ1 − µ2∥2 ≤ 4σ2[Φ−1((1 + δTV)/2)]2, which in turn is equivalent to DKL ≤ 2[Φ−1((1 + δTV)/2)]2. This shows that the squared ℓ2 distance ∥µθold − µθ∥2 used in our mask is a unified divergence measure equivalent (up to monotone transformations) to both KL and TV divergences.

#### C.3 Application to Flow-SDE

For Flow-SDE (Eq. (2)), the per-step policy is πθ(xt−∆t | xt) = N(µθ,σt2∆t · I) where:

σt2 2t

xt + (1 − t)vθ(xt,t) ∆t. The difference in means is:

µθ(xt,t) = xt + vθ(xt,t) +

σt2(1 − t) 2t

∆t · (vθ − vθold). Substituting into Eq. (20) with σ2 = σt2∆t:

µθ − µθold = 1 +

2

σt(1 − t) 2t

∆t 2

1 σt

DKLSDE(πθold∥πθ) =

∥vθ(xt,t) − vθold(xt,t)∥2. (21)

+

#### C.4 Application to CPS

For CPS (Eq. (3)), the policy mean is µCPSθ = (1 − (t − ∆t))xˆ0 + (t − ∆t)cos(ηπ/2)xˆ1 and the variance is σCPS2 = (t − ∆t)2 sin2(ηπ/2). Using xˆ0 = xt − tvθ and xˆ1 = xt + (1 − t)vθ, the difference in means is:

µCPSθ − µCPSθ

= [−(1 − (t − ∆t))t + (t − ∆t)(1 − t)cos(ηπ/2)](vθ − vθold). Let c(t) = −(1 − (t − ∆t))t + (t − ∆t)(1 − t)cos(ηπ/2). Then:

old

c(t)2∥vθ − vθold∥2 2(t − ∆t)2 sin2(ηπ/2)

DKLCPS(πθold∥πθ) =

. (22)

In previous work (Wang and Yu, 2025), the 2σCPS2 normalization is dropped for numerical stability, reducing the divergence to D(πθold∥πθ) = ∥µCPSθ

− µCPSθ ∥2. We instead retain the full normalization

old

in Eq. (22): because σCPS2 ∝ (t − ∆t)2 shrinks at later denoising steps, the σCPS−2 factor amplifies the divergence where small velocity changes most affect the output, yielding a tighter constraint that

prevents distribution collapse.

- Appendix D. Ratio Variance Analysis We provide a detailed analysis of the variance of the log-ratio in flow models. From Eq. (13), log rti = ϵ⊤d/σ − ∥d∥2/(2σ2), where d = µθ − µθold and ϵ ∼ N(0,I). It follows that:

∥d∥2 2σ2

= −DKL(πθold∥πθ), Var[log rti] = ∥d∥2

E[log rti] = −

= 2DKL(πθold∥πθ).

σ2

Thus std[log rti] = √2DKL. When the KL is moderate (e.g., DKL = 0.5), the standard deviation of the log-ratio is 1.0, meaning that individual log-ratio samples fluctuate by ±1 around the mean of −0.5. In terms of the ratio itself, this corresponds to roughly a 3× multiplicative spread.

Implication for clipping. With a typical clip parameter ϵ = 0.2 (i.e., clip range [0.8,1.2]), the log-clip range is [log 0.8,log 1.2] ≈ [−0.22,0.18]. Comparing this narrow range with the log-ratio standard deviation of √2DKL, we see that even for modest KL values, a significant fraction of samples will be clipped purely due to noise, not because the true divergence is excessive. This provides rigorous justification for replacing ratio-based clipping with direct divergence measurement.

### Appendix E. Towards a Predictive Divergence Mask

We recall the asymmetric mask in Flow-DPPO (Eq. (18)). The mask blocks the gradient (i.e., Mti = 0) when two conditions hold simultaneously: (i) the divergence Dt > δ already exceeds the trust-region threshold, and (ii) a directional condition signals that the optimization would push the policy further away from πθold. Concretely, the directional condition triggers when Aˆi > 0 ∧ rti > 1 (the gradient would further increase an already-elevated ratio) or Aˆi < 0 ∧ rti < 1 (the gradient would further decrease an already-reduced ratio). These two cases can be compactly unified as:

Mti = 0 ⇐⇒ sgn A ˆi · (rti − 1) > 0 ∧ Dt > δ. (23)

While this design is effective in practice, the directional indicator sgn A ˆi(rti − 1) is a heuristic proxy for whether the upcoming gradient step will increase the divergence. In a ratio-based trust region

(e.g., PPO clipping), this sign test is well-motivated: rti − 1 directly reflects the deviation of the single-sample Monte Carlo estimate of the importance ratio, so the sign of Aˆi(rti − 1) faithfully indicates whether the surrogate objective would drive the ratio further from unity. However, in a divergence-based trust region where the constraint is on Dt = DKL(πθold∥πθ), the connection is less direct. The ratio rti is a stochastic quantity evaluated at a single sampled action, whereas Dt measures a distributional distance that integrates over all actions. A positive Aˆi(rti − 1) does not guarantee that the gradient step will increase Dt, nor does a negative value guarantee a decrease.

In this section we exploit the Gaussian structure of flow model policies to derive a more principled masking criterion. We first predict how a single gradient step changes Dt (§E.1), obtaining a closedform expression that decomposes into a first-order directional term and a second-order magnitude term. The sign of the first-order term yields an exact directional criterion sgn A ˆ·(log rt−Dt) , which recovers the current sign test in the small-divergence regime but reveals a correction when the policy has already drifted. The full expression further accounts for the step size and gradient magnitude, leading to a predictive mask (§E.2) that directly forecasts whether the post-update divergence will exceed δ.

#### E.1 Predicting Post-Update Divergence

Fix a denoising step with state xt and suppress the time index for brevity. Write µ ≡ µθ(xt,t), µold ≡ µθold(xt,t), d = µ − µold, and Dt = ∥d∥2/(2σ2). The sampled action is xt−∆t = µold + σϵ with ϵ ∼ N(0,I).

We derive how a single gradient step on the surrogate objective L = rt · Aˆ changes the divergence Dt. The policy gradient with respect to µ is:

Aˆ · rt σ2

∇µL = Aˆ · rt · ∇µ log rt =

(σϵ − d).

With effective learning rate η, the updated mean is µnew = µ + η · ∇µL. Let g = σϵ − d. The predicted post-update divergence is:

Dtnew = ∥µnew − µold∥2

=

2σ2

η Arˆ t σ4

g⊤d +

= Dt +

- 1

- 2σ2

η Arˆ t σ2

g

d +

2

η2Aˆ2rt2 2σ6 ∥g∥2. (24)

From the ratio decomposition (Eq. (13)), log rt = ϵ⊤d/σ−∥d∥2/(2σ2), which gives g⊤d = σ2(log rt− Dt). The first-order term thus simplifies to (η Arˆ t/σ2)(log rt − Dt). The three terms in Eq. (24) have clear interpretations: (1) the current divergence Dt; (2) a first-order term whose sign determines whether the gradient step increases or decreases the divergence; (3) a non-negative second-order term that grows with the step size η and gradient magnitude ∥g∥, always contributing positively to Dtnew. The first-order directional criterion. The direction of divergence change is mainly determined by the sign of the first-order term. Since rt > 0 and η > 0, this sign equals:

sgn A ˆ · log rt − Dt . (25)

When this is positive, the gradient step increases Dt; when negative, it decreases Dt. Equivalently, this is the sign of the inner product ⟨∇µL, ∇µDt⟩, confirming that the surrogate gradient projects onto the divergence-increasing direction.

Recovery of the current mask. In the small-divergence regime Dt ≪ 1 (which is the typical operating range when the trust region is effective), the correction Dt ≈ 0 and the criterion simplifies to sgn(Aˆ · log rt). Since sgn(log rt) = sgn(rt − 1), this is equivalent to sgn A ˆ · (rt − 1) , which is exactly the directional condition in Eq. (23). Thus, the current Flow-DPPO mask implements the correct first-order divergence-increasing criterion in this regime.

The correction term. When Dt is non-negligible (i.e., the policy has already drifted appreciably), the true divergence-change direction is sgn A ˆ · (log rt − Dt) rather than sgn A ˆ · (rt − 1) . The subtracted term Dt shifts the decision boundary: a sample must have log rt > Dt > 0 (rather than merely log rt > 0) before the positive-advantage gradient is classified as divergence-increasing. Intuitively, when the policy has already moved away from πθold, a moderately elevated ratio does not necessarily push it further; only sufficiently large ratios do. This yields a first natural refinement of the mask: replacing sgn A ˆ(rt − 1) with sgn A ˆ · (log rt − Dt) as the directional indicator, which we call the first-order predictive mask:

Mt(1) =

- 0, if sgn A ˆ · log rt − Dt > 0 ∧ Dt > δ,
- 1, otherwise.

(26)

This mask uses only quantities already computed during training (Aˆ, rt, Dt) and requires no additional hyperparameters beyond the existing threshold δ.

#### E.2 The Predictive Mask

Based on Eq. (24), we define the (full) predictive mask that blocks updates whenever the predicted post-update divergence would exceed δ:

Mtpred =

- 0, if Dtnew > δ,
- 1, otherwise.

(27)

Comparison with the first-order mask. The first-order mask (Eq. (26)) only considers the direction of divergence change and still relies on the separate threshold condition Dt > δ. The full predictive mask unifies both into a single inequality: whether the gradient increases or decreases divergence is automatically encoded in the predicted value Dtnew, and the threshold comparison is applied to the predicted (rather than current) divergence. This has two consequences. First, when

Dt ≪ δ, even a divergence-increasing step may be permitted if the predicted Dtnew remains below δ. Second, when Dt is close to δ, the second-order term η2∥g∥2 may push Dtnew above δ even when the first-order direction is “safe” (i.e., the first-order mask would not fire), correctly blocking large gradient steps near the trust-region boundary.

Recovery of the existing mask. In the limit η → 0, the second-order term vanishes and Dtnew > δ reduces to requiring that the first-order direction is positive and Dt > δ. Combined with the small-divergence approximation (Dt ≈ 0), this exactly recovers the current Flow-DPPO mask (Eq. (23)).

#### E.3 Discussion on Mask Variants

Hierarchy of masks. The three masks form a natural hierarchy of increasing fidelity: sgn A ˆ(rt−1) current (Eq. (23))

⊂ sgn A ˆ(log rt−Dt) first-order (Eq. (26))

⊂ Dtnew > δ

.

full predictive (Eq. (27))

The current mask is the cheapest (no additional computation) and suffices when the trust region keeps Dt small throughout training. The first-order mask refines the directional decision with zero additional hyperparameters. The full predictive mask additionally requires an effective learning rate estimate but provides quantitative divergence prediction.

Local approximation. The analysis treats µ as a free vector, whereas in practice it is the output of a neural network. The actual change in µ(xt,t) is coupled to changes at all other inputs through shared parameters. The predictive mask is thus a local approximation that is most accurate when the effective learning rate is small and the network Jacobian is approximately preserved across one step.

We leave empirical validation of the predictive masks to future work. The key contribution of this analysis is twofold: it provides a theoretical justification for the existing asymmetric condition (showing it is the correct first-order criterion in the small-divergence regime), and it charts a principled path toward more refined trust-region enforcement that exploits the Gaussian structure of flow model policies.

### Appendix F. Experimental Details

#### F.1 Computational Resources.

All experiments are conducted on NVIDIA H20 96GB GPUs. The main results in Table 1 require approximately 90K GPU hours in total (across SD3.5, FLUX2-klein-base-9B, and FLUX1-dev with all methods and reward configurations). Including all ablation studies, multi-epoch experiments, and auxiliary runs, the overall computational cost for all experiments reported in this paper is approximately 140K GPU hours.

#### F.2 Hyperparameters.

LoRA is used for all models. We use LoRA r = 32 and α = 64 for SD3.5, r = 64 and α = 128 for FLUX2-9B and FLUX.1-dev. The learning rate is set to 3 × 10−4 for all models aligning to previous works. We set the training resolution to 512 × 512, number of denoising steps to 10 for SD3.5 and 14 for FLUX2-9B.

2

0.8

0.32

0.26

- 20

- 20.5
- 21

- 21.5

0.3

0.24

0.6

0. 2

0.28

0.4

0.2

0.26

0.2

0.18

0.24

0.16

0

0 50 1, 0 1,50

0 50 1, 0 1,50

0 50 1, 0 1,50

0 50 1, 0 1,50

GPU Hours GPU Hours GPU Hours GPU Hours

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS Difusion-NFT

- Figure 7: Training curves on SD3.5 for single-reward setting, including Diffusion-NFT (Zheng et al.,

2026) as an additional baseline. Flow-DPPO variants achieve state-of-the-art performance and less catastrophic forgetting on out-of-domain rewards, consistent with the main results.

| | | | |
|---|---|---|---|
| |GenEval2| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

| | | | | |
|---|---|---|---|---|
| |CLIP| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| |HPSv2| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | |PickScore| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0 1, 0 2, 0

0

0.1

0.2

0.3

0.4

0.5

0 1, 0 2, 0

0.15

0.2

0.25

0.3

0.35

0.4

0 1, 0 2, 0

0.1

0.15

0.2

0.25

0.3

0.35

0.4

0.45

0 1, 0 2, 0

18

20

2

24

26

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS Difusion-NFT

GPU Hours GPU Hours GPU Hours GPU Hours

- Figure 8: Training curves on FLUX2-9B for multi-reward setting (GPU hours). Flow-DPPO variants consistently outperform the baselines across all metrics, with a notable improvement on the GenEval2 reward.

For GRPO setting, we use group size 16 and number of groups 64 per epoch for all methods. The PPO clip threshold is set to 1×10−4 for Flow-GRPO and Flow-CPS, and 4×10−6 for GRPO-Guard, following the official recommendation. The thresholds for KL-clipping are set to 1 × 10−7 for FlowDPPO and 1 × 10−6 for Flow-DPPO+CPS due to their different KL-scaling factors. We applied the stragegy proposed in MixGRPO (Li et al., 2025) on all baselines and proposed methods for faster convergence and better performance. Specifically, we mix ODE and SDE sampling and randomly select 3 steps out of first half of the denoising steps for SDE sampling. The noise level for SDE sampling (η in CPS sampling) is set to 0.8.

For Diffusion-NFT, we follow the official implementation for SD3.5 for the rest of the hyperparameters, such as EMA schedule.

### Appendix G. Additional Experimental Results

#### G.1 Additional Training Curves

We provide the training curves on SD3.5 for the single-reward setting in Figure 7 (the multi-reward setting is in Figure 4 in the main body). We also provide the FLUX2-9B multi-reward training curves in Figure 8 and FLUX.1-dev in Figure 9.

0.34

0.32

0.8

0. 3

0.31

23

0.3

0.6

0.32

0.29

2.5

0.4

0.31

0.28

0.27

0.3

0.2

2

0.26

0 1, 0 2, 0 3, 0

0 1, 0 2, 0 3, 0

0 1, 0 2, 0 3, 0

0 1, 0 2, 0 3, 0

GPU Hours GPU Hours GPU Hours GPU Hours

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS

##### Figure 9: Training curves on FLUX.1-dev for single-reward setting.

| | | | | |
|---|---|---|---|---|
| | | | | |
|G| |enEval2| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| |CLIP| | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| |HPSv2| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
|P| |ickScore| | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

0.9

0.3

0.8

2

0.3

0.7

0.25

0.6

- 19
- 20
- 21

0.5

0.2

0.25

0.4

0.15

0.3

0.2

0.2

0.1

0.1

0 2, 0 4, 0

0 2, 0 4, 0

0 2, 0 4, 0

0 2, 0 4, 0

GPU Hours GPU Hours GPU Hours GPU Hours

Flow-GRPO Flow-CPS GRPO-Guard Flow-DPO Flow-DPO + CPS

##### Figure 10: Training curves on FLUX2-9B with CFG scale 4.0. Flow-DPPO variants remain robust under CFG, achieving strong performance with less catastrophic forgetting.

We additionally provide training curves on FLUX.1-dev in Figure 9.

| | | | | | | |
|---|---|---|---|---|---|---|
| |G| |enEval|2| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | |CLIP| | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |H| |PSv2| | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |Pi| |ckSco|re| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

- 23
- 24
- 25
- 26

0.38

0.4

0.5

0.36

0.35

0.34

0.4

0.32

0.3

0.3

0.3

2

0.25

0.28

21

0.2

0 20 40 60 80 1, 0

0 20 40 60 80 1, 0

0 20 40 60 80 1, 0

0 20 40 60 80 1, 0

Training Epoch Training Epoch Training Epoch Training Epoch

β = 1e-3 β = 1e-2 β = 0 (no KL reg.)

##### Figure 11: Training reward curves under three DKL(πθ∥πref) regularization strengths (β) on FLUX2klein-base-9B (multi-reward GDPO, CPS schedule). A moderate β=10−3 suppresses early reward hacking on PickScore and HPSv2, balancing cross-reward gradients and boosting final GenEval2 performance without hurting end-of-training performance on any individual reward.

G.2 KL Divergence Curves

##### Figure 12 visualises the per-step KL divergence between the current and reference (pre-trained) model across all six training settings and two SDE schedules. The corresponding end-of-training values are reported in Table 2 of the main body.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | |SD| |3.5| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |FLUX| |2-9Bw/o|CFG| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |FLU| |X2-9Bw/|CFG| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

KL (Flow-SDE) [×10⁻³] KL (CPS) [×10⁻³]

1.2

- 0
- 1
- 2
- 3
- 4

1

0.8

0.6

0.4

0.2

0

- 0

- 0.5
- 1

1.5

2

- 2.5

0 50 1 0 150 2 0 250 3 0

0 20 40 60 80 1 0

0 20 40 60 80 1 0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 0

- 0.5
- 1

- 1.5
- 2

0.8

0.6

0.4

0.2

- 0

- 0.5
- 1

1.5

2

- 2.5

0

0 50 1 0 150 2 0 250 3 0

0 20 40 60 80 1 0

0 20 40 60 80 1 0

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| |F| |LUX1-dev| | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | |SD3.5| |multi| | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| |FLU| |X2-9Bm|ulti| | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

8

KL (Flow-SDE) [×10⁻³] KL (CPS) [×10⁻³]

1.4

- 0
- 1
- 2
- 3
- 4
- 5

1.2

6

- 0.8
- 1

4

0.6

0.4

2

0.2

0

0

0 20 40 60 80 1 0

0 50 1 0 150 2 0 250 3 0

0 20 40 60 80 1 0

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

10

- 0
- 1
- 2
- 3
- 4

- 0

- 0.5
- 1

- 1.5
- 2

80

60

40

20

0

0 20 40 60 80 1 0

0 50 1 0 150 2 0 250 3 0

0 20 40 60 80 1 0

Training Epoch Training Epoch Training Epoch

Flow-GRPO GRPO-Guard Flow-DPO Flow-CPS Flow-DPO + CPS

Figure 12: KL-divergence between the current and reference (pre-trained) model during training, across six training settings (columns: four single-reward — SD3.5, FLUX2-9B w/o CFG, FLUX2-9B w/ CFG, FLUX.1-dev; two multi-reward — SD3.5 multi, FLUX2-9B multi) and two SDE schedules (rows: Flow-SDE, CPS). For each schedule, Flow-DPPO variants maintain a lower KL divergence with the pre-trained model, indicating less catastrophic forgetting and reward hacking. The only exception is in the FLUX2-9B w/o CFG setting under the CPS schedule, where Flow-DPPO + CPS shows a higher KL divergence than the Flow-CPS baseline after about epoch 500. The Flow-CPS run on FLUX2-9B multi collapsed at epoch 480; we plot its full logged trajectory and report its end-of-training KL at the run’s last logged step in Table 2.

#### G.3 Ablation Studies

- G.3.1 Classifier-Free Guidance

Previous works found that CFG heavily affects the training convergence and performance (Zheng et al., 2026). Here, we study the effect of CFG on the training of Flow-DPPO on FLUX2-9B, as shown

| | | | | |
|---|---|---|---|---|
|β = 1e-3<br><br>| | | | |
|β = 1e-2 β = 0 (no<br><br>|KL reg.)| | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

- 0.8
- 1

KL Divergence [×10⁻³]

0.6

0.4

0.2

0

0 20 40 60 80

Training Epoch

Figure 13: DKL(πθ∥πref) during training for different β settings.

- Table 3: End-of-training Soft TIFAGM on GenEval2 (%) across six training configurations (columns) and five RL algorithms (rows). The six columns correspond, left-to-right, to Figs. 2, 8, 10, 7, 4, 9. Per-column bold and underline mark the top-1 and top-2 methods; blue rows highlight our two contributions.

FLUX2-9B SD3.5 FLUX.1-dev Method Single Multi +CFG Single Multi Single

Flow-GRPO 84.5 46.8 54.6 56.6 39.9 87.8 Flow-CPS 82.7 47.1 89.0 74.8 44.6 91.2 GRPO-Guard 82.8 49.0 78.8 85.8 47.8 87.6 Diffusion-NFT – 47.3 – 64.5 42.5 – Flow-DPPO 85.1 57.7 87.4 78.9 48.1 90.7 Flow-DPPO + CPS 92.6 55.2 91.0 84.1 51.6 91.6

in Figure 10, where the CFG scale is set to 4.0 following the official recommendation. With CFG, Flow-DPPO variants still achieve state-of-the-art performance on the training reward (GenEval2) and mitigate catastrophic forgetting on the out-of-domain prompts, consistent with the observations in previous discussions. This shows that the divergence-based mask is robust under CFG and continues to deliver strong performance.

- G.3.2 Reference KL Regularization Strength

We ablate the strength of the DKL(πθ∥πref) regularization term (controlled by β) on FLUX2-kleinbase-9B under the multi-reward GDPO setting with CPS scheduling. Figure 11 shows the training reward curves and Figure 13 shows the KL divergence from the pretrained model. A moderate regularization strength (β=10−3) further mitigates early-stage reward hacking on auxiliary objectives (PickScore, HPSv2, etc.), thereby balancing the gradients across rewards and yielding an additional improvement in final GenEval2 performance over the unregularized baseline, without degrading end-of-training performance on any individual reward.

#### G.4 Quantitative Summary on GenEval2

To complement the per-setting training-curve figures above, Tables 4 and 3 report the end-of-training Soft TIFAGM score on GenEval2 for each method. Table 4 additionally reports end-of-training ancillary CLIP, PickScore, and HPSv2 rewards on both the in-domain GenEval2 prompt set and the held-out out-of-domain PickScore validation prompts, contextualising both SD3.5-medium and FLUX2-klein-base-9B by stacking six blocks: published reference numbers for state-of-the-art textto-image systems, the corresponding pretrained-baseline scores (no RL), and the five RL fine-tuning algorithms applied to each base model under both the single-reward (GenEval2-only) and multi-reward (GenEval2 + CLIP + PickScore + HPSv2) configurations. Table 3 then expands the per-method Soft TIFAGM comparison to all five training settings reported in this paper.

- Table 4: GenEval2 [Soft TIFAGM, defined in (Kamath et al., 2025)] together with ancillary CLIP, PickScore, and HPSv2 rewards at the end of training. The four in-domain columns are evaluated on the GenEval2 prompt set (the official released evaluation set of 800 prompts); the three out-ofdomain columns are evaluated on the PickScore prompt set. Within each RL block, bold marks the per-column top-1 method and underline the per-column top-2 method. Blue rows highlight our two contributions.

In-Domain (GenEval2) Out-of-Domain (PickScore) Model GenEval2 CLIP PickScore HPSv2 CLIP PickScore HPSv2 State-of-the-Art T2I Models

SD3.5-large 22.8 – – – – – – Bagel + CoT 23.1 – – – – – – Qwen-Image 33.8 – – – – – – Gemini 2.5 Flash Image 44.6 – – – – – –

Pretrained baselines (before RL)

SD3.5-medium 12.4 0.250 21.00 0.213 0.244 19.99 0.210 FLUX2-klein-base-9B 25.4 0.281 20.92 0.228 0.254 20.05 0.230 FLUX.1-dev 23.3 0.297 23.26 0.315 0.276 21.91 0.304

SD3.5-medium, single-reward RL fine-tuning

Flow-GRPO 56.6 0.297 21.21 0.219 0.252 19.33 0.206 Flow-CPS 74.8 0.313 21.68 0.235 0.260 19.94 0.220 GRPO-Guard 85.8 0.328 22.03 0.252 0.265 19.94 0.214 Diffusion-NFT 64.5 0.307 21.69 0.251 0.262 20.24 0.239 Flow-DPPO 78.9 0.319 22.06 0.263 0.265 20.45 0.253 Flow-DPPO + CPS 84.1 0.316 21.99 0.262 0.272 20.50 0.246

SD3.5-medium, multi-reward RL fine-tuning

Flow-GRPO 39.9 0.358 25.09 0.399 0.273 22.07 0.349 Flow-CPS 44.6 0.359 25.51 0.407 0.265 22.08 0.343 GRPO-Guard 47.8 0.353 25.64 0.409 0.272 22.32 0.354 Diffusion-NFT 42.5 0.334 25.30 0.394 0.269 22.52 0.355 Flow-DPPO 48.1 0.345 25.63 0.409 0.273 22.58 0.360 Flow-DPPO + CPS 51.6 0.369 25.72 0.415 0.279 22.51 0.361

FLUX2-klein-base-9B, single-reward RL fine-tuning

Flow-GRPO 84.5 0.314 21.82 0.276 0.264 20.84 0.280 Flow-CPS 82.7 0.311 21.82 0.261 0.275 21.15 0.267 GRPO-Guard 82.8 0.312 20.52 0.210 0.230 18.45 0.167 Flow-DPPO 85.1 0.331 22.22 0.294 0.278 21.27 0.285 Flow-DPPO + CPS 92.6 0.315 21.97 0.279 0.265 20.79 0.272

FLUX2-klein-base-9B, multi-reward RL fine-tuning

Flow-GRPO 46.8 0.371 25.61 0.412 0.277 22.62 0.357 Flow-CPS 47.1 0.361 25.70 0.416 0.276 22.85 0.364 GRPO-Guard 49.0 0.375 25.27 0.411 0.269 21.99 0.349 Diffusion-NFT 47.3 0.336 24.87 0.389 0.274 22.47 0.351 Flow-DPPO 57.7 0.364 25.76 0.418 0.282 22.90 0.368 Flow-DPPO + CPS 55.2 0.386 26.15 0.427 0.287 22.97 0.370

FLUX.1-dev, single-reward RL fine-tuning

Flow-GRPO 87.8 0.331 23.03 0.311 0.291 21.85 0.311 Flow-CPS 91.2 0.328 23.20 0.317 0.288 21.98 0.307 GRPO-Guard 87.6 0.333 22.69 0.293 0.286 21.03 0.276 Flow-DPPO 90.7 0.331 23.15 0.323 0.290 21.60 0.300 Flow-DPPO + CPS 91.6 0.331 23.29 0.322 0.289 21.91 0.305

