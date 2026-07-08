## GradientStabilizer: Fix the Norm, Not the Gradient

Tianjin Huang 12 Zhangyang Wang3 Haotian Hu4 Zhenyu Zhang3 Gaojie Jin1 Xiang Li5 Li Shen6 Jiaxing Shang7 Tianlong Chen8 Ke Li1 Lu Liu1 Qingsong Wen9 Shiwei Liu101112

# arXiv:2502.17055v4[cs.LG]26May2026

### Abstract

Training instability in modern deep learning systems is frequently triggered by rare but extreme gradient-norm spikes, which can induce oversized parameter updates, corrupt optimizer state, and lead to slow recovery or divergence. Widely used safeguards such as gradient clipping mitigate these failures but require threshold tuning and indiscriminately truncate large updates. We propose GradientStabilizer, a lightweight, drop-in gradient transform that preserves the instantaneous gradient direction while replacing the update magnitude with a statistically stabilized estimate derived from running gradient-norm statistics. We prove that the resulting stabilized magnitude is uniformly bounded on spike steps, independent of the spike size, and show how this boundedness controls optimizer state evolution in adaptive methods. Across LLM pre-training (FP16), quantizationaware pre-training (FP4), ImageNet classification, reinforcement learning, and time-series forecasting, GradientStabilizer consistently improves training stability, widens stable learningrate regions, and reduces divergence relative to clipping-based baselines, even substantially reducing Adam’s sensitivity to weight-decay strength. The Code is available at https:

//github.com/TianjinYellow/

1Department of Computer Science, University of Exeter 2Department of Mathematics and Computer Science, Eindhoven University of Technology 3Department of Electrical and Computer Engineering, University of Texas at Austin 4School of the Gifted Young, University of Science and Technology of China 5Department of Computer Science, University of Reading 6School of Cyber Science and Technology, Sun Yat-sen University 7College of Computer Science, Chongqing University 8Department of Computer Science, The University of North Carolina at Chapel Hill 9Squirrel Ai Learning 10ELLIS Institute Tubingen 11Max Planck Institute for Intelligent Systems 12T¨ubingen AI Center, T¨ubingen, Germany. Correspondence to: Tianjin Huang <t.huang2@exeter.ac.uk>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

GradientStabilizer.git.

### 1. Introduction

The optimization of deep neural networks has advanced rapidly over the past decade, driven by stochastic gradient descent (SGD) and adaptive variants such as Adam and its extensions (Kingma, 2014; Shazeer & Stern, 2018; Gupta et al., 2018; Loshchilov & Hutter, 2017). Despite these successes, training instability remains a persistent challenge in modern large-scale regimes. Instabilities are frequently observed in large language model (LLM) pre-training (Chowdhery et al., 2023; Liu et al., 2025; Takase et al., 2023), reinforcement learning with verifiable rewards (Yu et al., 2025; Zeng et al., 2025; He et al., 2025), and quantizationaware training (Wortsman et al., 2023; Zhang, 2025). In these settings, rare but extreme gradient-norm spikes can induce oversized parameter updates, distort optimizer state, and occasionally trigger catastrophic divergence.

A common safeguard against such failures is gradient clipping, which caps the norm (or coordinates) of the update to prevent excessive parameter changes (Pascanu et al., 2013; Brock et al., 2021; Kumar et al., 2025). While effective in practice, clipping operates as an extrinsic post-processing rule that enforces instantaneous constraints via fixed thresholds. As a result, it requires careful tuning and may either intervene too late to prevent instability or unnecessarily suppress informative updates during otherwise stable phases of training. More adaptive variants, such as ZClip and adaptive gradient clipping (Kumar et al., 2025; Wang et al., 2025; Brock et al., 2021), partially address threshold sensitivity but remain fundamentally reactive mechanisms that truncate gradients once constraints are violated.

This work proposes GradientStabilizer, an intrinsic stabilization mechanism that addresses gradient spikes and explosions by structurally decoupling the update direction from its magnitude (As shown in Figure 1). The core principle is simple: while the direction of the gradient often provides reliable descent information, its instantaneous norm can be highly volatile and dominated by noise or rare outliers. GradientStabilizer preserves the instantaneous gradient direction while replacing its magnitude with

101234101010101010 RawGradientNorm(LogScale)

3.54.04.55.05.5 EvalLoss

0 2500 5000 7500 10000 12500 15000 17500 20000 Update Steps

0 2500 5000 7500 10000 12500 15000 17500 20000 Update Steps

AdamW+ZClip AdamW+NormClip AdamW AdamW+GradientStabilizer

- Figure 1. Raw gradient norm and evaluation loss across update steps. AdamW+GradientStabilizer suppresses gradient norm explosion while achieving the lowest evaluation loss compared with AdamW, AdamW+ZClip, and AdamW+NormClip. Experiments are conducted on LLaMA-130M with 2.2B Tokens using a learning rate of 3 × 10−3.

a statistically stabilized estimate computed from running averages of gradient norms. This yields smooth, threshold-free control of update magnitudes without truncating directions or other manual intervention.

We then provide a theoretical analysis of Gradient Stabilizer in both stationary and spike-driven regimes. In stationary settings, we characterize the population target of the stabilized magnitude as a mean-to-RMS ratio that decreases with the coefficient of variation of gradient norms, explaining its variance-dampening behavior. More importantly, under a simple spike event model, we show that the stabilized update magnitude is uniformly bounded on spike steps, independent of the raw spike size. This property ensures that arbitrarily large gradient spikes cannot produce arbitrarily large parameter updates once passed through GradientStabilizer. When combined with adaptive optimizers such as Adam, this bounded effective-gradient property suffices to control the internal moment states of Adam/AMSGrad and to bound each per-coordinate update, which are key technical conditions assumed by standard nonconvex convergence analyses.

Our contributions are summarized as follows:

⋆ Method. We introduce GradientStabilizer, a lightweight, drop-in gradient transform that preserves the update direction while adaptively stabilizing the magnitude using running statistics of gradient norms, providing a threshold-free alternative to gradient clipping.

⋆ Theoretical characterization. We analyze Gradient Stabilizer in both stationary and spike regimes, establishing a variance-dampening interpretation in stationary settings and a spike-dampening bound that guarantees uniformly bounded update magnitudes on arbitrarily large gradient spikes.

⋆ Optimization implications. We show that the intrinsic bounded stabilized gradient property induced by GradientStabilizer suffices to control the internal moment states of Adam/AMSGrad and to bound each per-coordinate update, independent of the magnitude of raw gradient spikes.

⋆ Empirical evaluation. Across a wide spectrum of tasks, GradientStabilizer improves training stability, broadens the stable learning-rate region, and reduces divergence compared to clipping-based baselines. Additionally, we observe that gradient clipping exacerbates Adam’s sensitivity to weight-decay strength, whereas GradientStabilizer substantially reduces this sensitivity across tasks.

### 2. GradientStabilizer

In this section, we propose GradientStabilizer, a lightweight and optimizer-agnostic gradient transformation for stabilizing training under gradient-norm spikes. GradientStabilizer replaces the instantaneous magnitude of the gradient with a statistically stabilized estimate derived from the history of the optimization trajectory. We define the update direction dt as the unit vector of the current gradient:

gt ∥gt∥2

dt =

To determine the step magnitude, we track the first and second moments of the gradient norm using Exponential Moving Averages (EMA). Let Rt = ∥gt∥2. We update the moment estimates mt and vt as follows:

mRt = γ1mRt−1 + (1 − γ1)Rt vtR = γ2vtR−1 + (1 − γ2)Rt2

Algorithm 1 GradientStabilizer

computed via an exponential moving average. The spike event at iteration t is defined as St := {Rt ≥ κm(t−R)1 }.

- 1: Input: initial parameters θ0; base optimizer A with state ϕ0; loss function ℓt ; learning rates {ηt}Tt=1; EMA decays γ1, γ2 ∈ [0, 1)
- 2: Output: final parameters θT
- 3: mR0 ← 0, v0R ← 0 ▷EMA states for gradient norms
- 4: for t = 1 to T do
- 5: gt ← ∇θℓt(θt−1) ▷stochastic gradient
- 6: Rt ← ∥Flatten(gt)∥2
- 7: mRt ← γ1mRt−1 + (1 − γ1)Rt
- 8: vtR ← γ2vtR−1 + (1 − γ2)Rt2 ▷EMA moments
- 9: dt ← gt/Rt ▷unit direction
- 10: ρt ← mRt / vtR ▷stabilized magnitude

- 11: g˜t ← ρt · dt
- 12: (θt, ϕt) ← UpdateA(θt−1, g˜t, ηt, ϕt−1)
- 13: end for

Lemma 3.2 (Variance Dampening). Let R ≥ 0 be a random variable denoting the (approximately stationary) gradient norm, with µ = E[R], σ2 = Var(R) and ν = E[R2]. Define the target population ratio

E[R] E[R2]

µ √ν

ρ⋆ =

=

.

Let c2v = σ2/µ2 be the squared coefficient of variation. Then

1 1 + c2v ≤ 1. (1)

ρ⋆ =

where γ1,γ2 ∈ [0,1) are decay rates controlling the effective memory length. Using these moments, we compute the stablized magnitude ρt:

mRt vtR

ρt =

consider a generic stochastic optimization algorithm A (e.g. Adam (Kingma, 2014),AdamW (Loshchilov & Hutter, 2017),SPAM (Huang et al., 2025) and etc), the final param-

eter update is constructed by scaling the unit direction dt with this stabilized mangnitude ρt and the learning rate ηt:

θt+1 = UpdateA(θt,g˜t,ηt,ϕt), g˜t = dt · ρt

where ϕt represents the internal state of the optimizer (e.g., momentum buffers). GradientStabilizer is a lightweight, drop-in gradient transformation and can be integrated into standard training pipelines with minimal overhead. Algorithm 1 provides the pseudocode.

### 3. Theoretical Justification

In this section, we analyze the stability properties of GradientStabilizer. We characterize its behavior in stationary and spike-driven regimes, and show that it induces uniformly bounded effective gradients and optimizer moment states. These results establish key stability prerequisites that are required by existing convergence analyses for nonconvex optimization with adaptive methods.

3.1. Stability Properties We first characterize the behavior of the stabilized magnitude ρt under statistical assumptions on the gradient norm. Definition 3.1 (Spike Event). Fix κ ≫ 1. Let Rt denote the raw gradient norm at iteration t, and let m(t−R)1 denote the first moment estimator of the historical norms {Rs}ts−=11,

- Proof. See Appendix A.1.

Remark 3.3. Lemma 3.2 provides a population-level characterization of the stability ratio: the ratio ρ⋆ = E[R]/ E[R2] decreases monotonically with the variability of R (via c2v), capturing an intrinsic variancedampening effect. In our method, the stabilized magnitude ρt = mRt / vtR is computed from fixed-decay EMA statistics mRt ≈ E[R] and vtR ≈ E[R2] over an effective window of length ≍ (1−γ2)−1. Thus, under approximate stationarity over this window, ρt tracks the target ratio ρ⋆ (up to finite-window estimation error), inheriting the same qualitative behavior.(I) High-variance regime: In the presence of noise or gradient spikes, ρt diminishes, naturally contracting the update step to preserve stability. (II) Low-variance regime (i.e., cv → 0): As the variance vanishes, ρt → 1. In this phase, the algorithm utilizes the full learning rate η, recovering the dynamics of Normalized Gradient Descent.

Lemma 3.4 (Uniform Spike-Step Upper Bound). Let {gt}t≥1 be any stochastic gradient sequence with norms Rt = ∥gt∥2 and Let m(t−R)1 be the norm for the first moment mt− at step t − 1. Assume that gradient spikes occur with satisfying the condition Rt ≥ κm(t−R)1 for a threshold κ ≫ 1. Fix decay rates γ1,γ2 ∈ [0,1). For any historical state (mt−1,vt−1) consistent with the EMA recursions, the stablized gradient norm ∥g˜t∥2 is bounded by:

∥g˜t∥2 ≤ ρt ≤

1 − γ1 √1 − γ2

+

γ1 κ√1 − γ2

(2)

where ∥g˜t∥2 = ρt whenever Rt > 0, and the inequality is strict only in the degenerate case Rt = 0.

- Proof. See Appendix A.2.

Remark 3.5. Lemma 3.4 provides a spike-invariant upper bound on the stabilized gradient norm: even if the raw

- 1 1
- 1 2

- 1 1
- 1 2

1 1 12/ 2

18

| | | | | |
|---|---|---|---|---|
| | | |[Figure 1]| |
| | | | | |
| | | | | |
| | | | | |

| | | | | |
|---|---|---|---|---|
| | | | |[Figure 2]|
| | | | | |
| | | | | |
| | | | | |

[Figure 3]

0.99

0.9

0.9

0.8

0.8

0.7

0.6

1

1

0.7

0.6

0.6

0.5

0.02

0.7 0.8 0.90.990.999

0.85 0.90.950.990.999

2

2

- Figure 2. Heatmaps of Coefficient-dependent bound factors in Lemma 3.4 and Lemma 3.6. Heatmaps of √11−−γγ1

2

(left) and √ 1

1−γ12/γ2

(right) over a (γ1, γ2) grid. The right panel is re-

stricted to the feasible region γ12 < γ2, under which the bound in Lemma 3.6 is well-defined.

gradient norm Rt becomes arbitrarily large on spike steps where κ can be very large and reach up to 1000× in practice (Huang et al., 2025), the stabilized gradient norm ρt cannot blow up, since when κ is large, the second term is negligible and the effective ceiling is well-approximated by ρt ≲ (1 − γ1)/√1 − γ2. The heatmap in Figure 2 (Left) empirically validates that this upper bound remains within a controlled range across a wide spectrum of γ1 and γ2 values.

Lemma 3.6 (Uniform Bound on Stabilized Gradient). Assume m(0R) = v0(R) = 0 and γ1,γ2 ∈ [0,1) satisfy γ12 < γ2. Then for all t ≥ 1,

∥g˜t∥2 ≤ ρt ≤ ρ¯ :=

1 − γ1 √1 − γ2 ·

1 1 − γ12/γ2

. (3)

where ∥g˜t∥2 = ρt whenever Rt > 0, and the inequality is strict only in the degenerate case Rt = 0.

Proof. See Appendix A.3.

Remark 3.7. Lemma 3.6 establishes a time-uniform bound on the stabilized gradient. Under the condition γ12 < γ2, the stabilized gradient norm ∥g˜t∥2 is bounded by a closedform constant ρ¯that depends only on the EMA decay rates (γ1,γ2). In particular, this bound holds for all t and does not depend on the instantaneous raw norm ∥gt∥2, ruling out the divergence of stabilized gradient even when the raw gradients are extremely noisy or exhibit rare spikes.

- Figure 2 (Right) visualizes ρ¯across a spectrum of (γ1,γ2), demonstrating that the upper bound remains moderate for typical choices of decay rates.

- 3.2. Implications for Optimizer Stability

Proposition 3.8 (Coordinatewise Bound from Lemma 3.6). Under the conditions of Lemma 3.6, for all t ≥ 1,

∥g˜t∥∞ ≤ ∥g˜t∥2 ≤ ρ,¯ and hence |g˜t,i| ≤ ρ¯ ∀i. (4)

- Proof. See Appendix A.4

| |
|---|

Theorem 3.9 (Bounded Adam/AMSGrad Moment States under GradientStabilizer). Let {g˜t}t≥1 be the stabilized gradients produced by GradientStabilizer. Under the conditions of Proposition 3.8 and β1,β2 ∈ [0,1) and initialize m0 = 0, v0 = 0, and vˆ0 = 0. Consider Adam (Kingma, 2014)/AMSGrad (Reddi et al., 2019) moment recursions

mt = β1mt−1 + (1 − β1)˜gt, (5) vt = β2vt−1 + (1 − β2)(˜gt)⊙2, (6) vˆt = max{vˆt−1, vt} (AMSGrad only), (7)

where (·)⊙2 and max(·,·) are applied elementwise. Then, for all t ≥ 1,

∥mt∥∞ ≤ ρ¯(1 − β1t);∥vt∥∞, ∥vˆt∥∞ ≤ ρ¯2(1 − β2t). (8)

- Proof. See Appendix A.5.

Remark 3.10. Theorem 3.9 indicates that if Adam/AMSGrad is driven by the stabilized gradients {g˜t} , the internal moment states cannot blow up: the first-moment estimate mt and the second-moment accumulators vt (and vˆt for AMSGrad) remain uniformly bounded for all t, independent of the magnitude of the raw gradient spikes that may have occurred prior to GradientStabilizer. These bounds preclude unbounded moment growth and make the update mapping well-defined, providing a basic stability prerequisite for Adam-type methods. Importantly, we do not claim convergence by itself; rather, we isolate and guarantee a stability condition that is typically assumed (but rarely verified) in convergence analyses of adaptive optimizers.

Corollary 3.11 (Bounded Per-step Parameter Change for SGD under GradientStabilizer). Consider SGD driven by the stabilized gradients produced by GradientStabilizer: θt+1 = θt − ηtg˜t. Under the conditions of Proposition 3.8, for all t ≥ 1,

∥θt+1 − θt∥∞ ≤ ηtρ.¯ (9)

Moreover, on spike event St, i.e., those satisfying Rt ≥ κm(t−R)1 for some κ ≫ 1,

1 − γ1 √1 − γ2

γ1 κ√1 − γ2

. (10)

∥θt+1 − θt∥∞ ≤ ηt

+

Proof. See Appendix A.7.

Remark 3.12. Corollary 3.11 shows that, under GradientStabilizer, the worst-case coordinate update is controlled by the intrinsic bound ρ¯on the stabilized

CLIP (Pascanu et al., 2013), we use thresholds 0.1 and 1.0, respectively. For AGC (Brock et al., 2021), we set the clipping factor to 0.01. For ZCLIP (Kumar et al., 2025), we follow the recommended defaults and set α = 0.97 and the z-score threshold to 2.5. For a fair comparison, we do not tune hyperparameters for either the baselines or our method; the same settings are used throughout the paper.

gradient, rather than by the (possibly unbounded) raw gradients. As a result, raw gradient explosions cannot induce a catastrophic single-step parameter jump. Moreover, on spike steps the update bound is further reduced by the factor 1/κ, providing additional damping.

- Table 1. Performance of GradientStabilizer on FP4 and FP16 LLM pre-training. Experiments are based on LLaMA130M/350M. Validation perplexity is reported. The best results are in bold, and the second-best are underlined.

4.1. Superior Performence of GradientStabilizer LLM Pre-training and Quantization-Aware Training.

FP16 Training FP4 Training 130M 350M 130M 350M

Method

We evaluate GradientStabilizer against VALUE CLIP, NORM CLIP, AGC, and ZCLIP in LLM pretraining under both FP16 and FP4 quantization-aware training. Experiments use LLaMA-130M and LLaMA350M trained on C4 for 2.2B and 6.6B tokens. We consider ADAM and ADAMW as the base optimizers, and report validation perplexity (PPL) in Table 1. We observe that ❶ GradientStabilizer significantly improves final performance across model scales, for both Adam and AdamW base optimizers, under FP16 pre-training and FP4 quantization-aware training. For instance, on FP4-trained LLaMA-350M, GradientStabilizer reduces validation perplexity by approximately 2.5 PPL. ❷ With comparison to clipping baselines across settings, GradientStabilizer achieves the best performance among all the baselines. Among them, ZClip (Kumar et al., 2025), a recently proposed approach, typically yields the second-best results. We further observe larger gains under FP4 quantization-aware training than under FP16 training. A plausible explanation is that low-bit training is more prone to optimization instability due to quantization error (Panferov et al., 2025; Wortsman et al., 2023), in which case stabilizing the step magnitude provides greater benefit.

ADAM 24.60 19.04 29.02 21.57 + VALUE CLIP (Pascanu et al., 2013) 24.48 19.01 28.95 20.48 + NORM CLIP (Pascanu et al., 2013) 24.17 18.84 28.31 22.29 + AGC (Brock et al., 2021) 24.32 18.76 28.52 19.24 + ZCLIP (Kumar et al., 2025) 24.16 18.62 28.34 19.23 + GradientStabilizer 23.32 17.83 26.82 18.89

ADAMW 24.31 19.21 28.72 21.35 + VALUE CLIP (Pascanu et al., 2013) 24.34 19.15 28.72 21.84 + NORM CLIP (Pascanu et al., 2013) 23.86 18.98 28.05 20.13 + AGC (Brock et al., 2021) 24.09 18.90 28.26 19.42 + ZCLIP (Kumar et al., 2025) 23.89 18.89 28.04 21.39 + GradientStabilizer 23.14 17.80 26.66 18.84

Training Tokens 2.2B 6.6B 2.2B 6.6B

### 4. Experiments

To demonstrate the effectiveness of the proposed method, we evaluate it on a diverse set of widely used tasks spanning LLM and quantization-aware pre-training, image classification, reinforcement learning, and time-series forecasting.

Baselines. We compare our method against several widely used gradient clipping approaches. (1) VALUE CLIP (Pascanu et al., 2013) clips each gradient element when its absolute value exceeds a predefined threshold. (2) NORM CLIP (Pascanu et al., 2013) rescales the entire gradient when its ℓ2 norm exceeds a threshold. (3) Adaptive Gradient Clipping (AGC) (Brock et al., 2021) clips gradients when the unit-wise ratio between the gradient norm and the parameter norm exceeds a threshold. (4) ZCLIP (Kumar et al., 2025) detects and clips abnormal gradients by computing z-score statistics of gradient norms.

ImageNet Classification. To evaluate the effect of GradientStabilizer on vision task, we conduct experiments on ImageNet-1K using three standard architectures spanning transformers and CNNs: ViT-B, ConvNeXtT, and ResNet-50. For ViT-B and ConvNeXt-T, we follow the official torchvision training recipes.12 For ResNet50, we use the same recipe as ConvNeXt-T. All methods are trained with the same setup for 120 epochs. Top-1 accuracy (%) are reported in Table 2. We observe that GradientStabilizer consistently improves over the AdamW/Adam base optimizers across ViT-B, ConvNeXt-T, and ResNet-50. It obtains the best or second-best Top-1 accuracy in nearly all cases, yielding stable gains across diverse architectures. We also note that while ZCLIP is competitive on language models, it does not consistently out-

Models and Datasets. For image classification, we evaluate on ImageNet-1K (Deng et al., 2009) using both transformerand convolution-based architectures, including ViT-B (Dosovitskiy, 2020), ConvNeXt-T (Liu et al., 2022), and ResNet50 (He et al., 2016). For language modeling, we train LLaMA-130M and LLaMA-350M (Touvron et al., 2023) on the C4 dataset. For reinforcement learning, we report results on the HalfCheetah-v4 environment. For time-series forecasting, we conduct experiments on the widely used Weather dataset using PatchTST (Nie, 2022).

- 1https://github.com/pytorch/vision/tree/

main/references/classification#vit_b_16

- 2https://github.com/pytorch/vision/tree/

Hyperparameter Settings. We use fixed hyperparameters across all experiments. For GradientStabilizer, we set γ1 = 0.6 and γ2 = 0.999. For VALUE CLIP and NORM

main/references/classification#convnext

- Table 2. Performence of GradientStabilizer on ImageNet-1K. Top-1 accuracy (%) on ImageNet-1K using ViT-B, ConvNeXt-T, and ResNet-50 with AdamW/Adam baselines. The best results are in bold, and the second-best are underlined.

Method ViT-B ConvNeXt-T ResNet-50

ADAMW 79.3 79.6 77.3 + VALUE CLIP (Pascanu et al., 2013) 79.4 80.0 77.2 + NORM CLIP (Pascanu et al., 2013) 79.5 80.0 77.5 + AGC (Brock et al., 2021) 38.5 77.2 77.7 + ZCLIP (Kumar et al., 2025) 79.4 80.1 77.5 + GradientStabilizer 79.6 80.1 77.6

ADAM 77.1 77.6 75.5 + VALUE CLIP (Pascanu et al., 2013) 77.0 78.1 75.6 + NORM CLIP (Pascanu et al., 2013) 77.1 78.2 75.7 + AGC (Brock et al., 2021) 71.2 76.9 75.8 + ZCLIP (Kumar et al., 2025) 76.9 78.1 75.7 + GradientStabilizer 77.3 78.3 75.8

Training Datasets ImageNet-1K ImageNet-1K ImageNet-1K

Adam

AdamW

10K

TestReward()

| |
|---|

| |
|---|

| |
|---|
| |

5K

| |
|---|

| |
|---|

| |
|---|

| |
|---|

0 50 100 Epoch

0 50 100 Epoch

Base Optimizer

+Norm Clip

+AGC +Ours

+Value Clip

+ZClip

- Figure 3. Reinforcement learning on HalfCheetah-v4. Mean episodic return ± standard deviation over 10× evaluation rollouts, plotted against training epochs.

perform NORM CLIP on vision models, suggesting that its gains may be configuration-dependent.

Reinforcement Learning. We evaluate our proposed method on a continuous-control benchmark in MuJoCo. Specifically, we train the policy network using PPO (Schulman et al., 2017) on HALFCHEETAH-V4, following the standard training recipe from the Tianshou implementation.3

- Figure 3 reports the mean and standard deviation of the evaluation return computed over 10 rollouts. We observe that ❶ GradientStabilizer consistently attains the highest returns among all clipping baselines when combined with either Adam or AdamW, demonstrating robust gains across base optimizers; ❷ while standard clipping baselines generally improve over the unclipped optimizers, their relative ranking varies across settings, and no single clipping variant dominates consistently. The results for ANT-V4, HOPPERV4 and WALKER-V4 are provided in Appendix C.4.

Time Series Forecasting. To evaluate the effectiveness of GradientStabilizer on time-series forecasting tasks, we conducted experiments on the Weather dataset by adopt-

3https://github.com/thu-ml/tianshou/tree/ master/examples/mujoco

ing PatchTST (Nie, 2022), a widely used Transformer-based architecture, as the backbone model. We train PatchTST with Adam and AdamW, and compare against standard gradient-clipping baselines, following the official training recipes from the public PatchTST codebase.4 Figure 5 reports the mean ± standard deviation over 10× independent runs. We observe that ❶ GradientStabilizer yields substantial gains over the base optimizers and achieves the best performance among all clipping baselines; ❷ among clipping baselines, AGC is the strongest competitor, matching the leading performance, whereas VALUE CLIP provides no consistent improvement over the base optimizers.

Summary: ① GradientStabilizer consistently delivers top-tier performance across diverse tasks and backbones, suggesting strong general applicability. ② In contrast, existing gradient clipping methods can be competitive in particular tasks or settings, but do not exhibit comparable consistency across diverse tasks.

#### 4.2. Stability Analysis

Training stability. We assess the effectiveness of GradientStabilizer by tracking the raw gradient norm (i.e., before applying GradientStabilizer) together with the training loss. Experiments are conducted for LLaMA-130M pre-training on C4 using ADAMW with learning rate 10−3. The results are shown in the two right subfigures of Figure 4. We observe that the base optimizer ADAMW suffer from a large number of gradient-norm spikes after several thousand steps where the norm exhibits frequent spikes occurring in bursts. These repeated spikes are accompanied by pronounced loss spikes, and the training dynamics eventually become unstable and diverge. In contrast, when combined with GradientStabilizer, the training loss curve remains well-behaved: the raw gradient norm does not exhibit the same spike bursts, and the loss continues to decrease smoothly without pronounced spike-induced instabilities. Refer to Appendix C.2 for comparisons with gradient clipping baselines.

Learning rate stability. We investigate the sensitivity of training stability to the learning rate when employing GradientStabilizer. Specifically, we perform a parameter sweep over learning rates in the interval [10−4, 3 × 10−3] and report the final validation loss. Experiments are conducted on LLaMA-130M pre-training using the C4 dataset, evaluating both Adam and AdamW optimizers with the FP4 training regime. The results, summarized in Figure 4 (right), demonstrate the stabilizing effect of

4https://github.com/yuqinie98/PatchTST/ tree/main/PatchTST_supervised/scripts/ PatchTST

|Gradient| |
|---|---|
| | |
| | |

Adam

AdamW

Norm

Training Loss

100

8

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

ValidationLoss()

- 3
- 4
- 5

- 3
- 4
- 5

6

50

4

0

10 4 10 3 3 × 10 3 LR

10 4 10 3 3 × 10 3 LR

0 5k 10k Step

0 5k 10k Step

Base Optimizer Base Optimizer+Ours

- Figure 4. Training and learning-rate stability. Left: validation loss under different learning rates, illustrating learning-rate stability. Right: raw gradient norm (before GradientStabilizer) and training loss curves, illustrating training stability. All experiments are conducted on LLaMA-130M FP4 training on C4.

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0 50 100 Epoch

0.155

0.160

0.165

TestMSE()

Adam

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0 50 100 Epoch

AdamW

Base Optimizer

+Value Clip

+Norm Clip

+ZClip

+AGC +Ours

- Figure 5. Forecasting performance on Weather dataset. Experiments are conducted on Weather using PatchTST with ADAM/ADAMW base optimizers. Results are reported as mean ± standard deviation over 10× independent runs.

Table 3. Weight-decay stability on ADAM. Top-1 accuracy (%) of ViT-B under different weight-decay strengths using Adam on ImageNet-1K. The best results are in bold.

Weight Decay 0 1e-4 5e-4

Method

ADAM 77.1 59.3 52.9 + VALUE CLIP (Pascanu et al., 2013) 77.0 60.2 29.2 + NORM CLIP (Pascanu et al., 2013) 77.1 53.4 20.6 + AGC (Brock et al., 2021) 71.2 0.1 0.1

+ ZCLIP (Kumar et al., 2025) 76.9 59.9 30.5 + GradientStabilizer 77.3 78.7 72.4

range from 0 to 5 × 10−4 and trained the models for 120 epochs. The results, summarized in Table 3, reveal that as weight decay strength increases, all baseline gradient clipping methods suffer substantial performance degradation compared to standard ADAM. This suggests that traditional gradient clipping exacerbates ADAM’s sensitivity to weightdecay strength. In contrast, GradientStabilizer exhibits only minor performance drops and consistently outperforms ADAM, demonstrating that our method substantially mitigates this sensitivity. The results for ResNet-50 and ConvNeXt-T are provided in Appendix C.1.

our method. As the learning rate increases from 10−3 to

- 3 × 10−3, the validation loss for GradientStabilizer combined with the base optimizer degrades more gracefully than that of the base optimizer alone, indicating superior robustness in high-learning-rate region. Conversely, in the low-learning-rate region, reducing the rate from 10−3 to 10−4 results in a more attenuated increase in loss when using GradientStabilizer. Overall, these findings suggest that GradientStabilizer effectively widens the effective operating range of learning rates. Refer to Appendix C.2 for comparisons with gradient clipping baselines.

#### 4.3. Additional Analysis

Performance under corrupted data. Prior works (Shah et al., 2025; Liu & Ma, 2025; Talak et al., 2024) have established that corrupted inputs exacerbate training instability. To evaluate the efficacy of GradientStabilizer in mitigating these effects, we conduct experiments on a time series forecasting task using the Weather dataset. We employ the PatchTST architecture (Nie, 2022) with the ADAMW optimizer. To simulate data corruption, we inject Gaussian noise into the input X with a probability of p = 5%. The noise is sampled from N 0, σ2 , where the standard deviation is defined as σ = Noise Level ·

Weight Decay Stability on ADAM. Prior work (Loshchilov & Hutter, 2017) has demonstrated that the Adam optimizer is highly sensitive to weight decay strength, as the effective decay is scaled by the second moment estimate and coupled with the learning rate. To investigate the impact of GradientStabilizer on this sensitivity, we conducted experiments on ImageNet-1K using the ViT-Base architecture. We swept the weight decay strength across a

Noise Level = 1

Noise Level = 2

Noise Level = 3

0.170.200.24

| | | |
|---|---|---|
| | | |
|=0.008| | |
| | | |
| | | |

| | | |
|---|---|---|
|=0.018<br><br>| | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
|=0.025<br><br>| | |
| | | |

0.160.180.20 TestMSE()

0.180.230.27

=0.025

=0.018

=0.008

0 50 100 Epoch

0 50 100 Epoch

0 50 100 Epoch

Base Optimizer +Ours

- Figure 6. Test MSE under input perturbations on Weather time-series data. We corrupt 5% of randomly selected input by adding zero-mean Gaussian noise. Specifically, the Gaussian noise is samples from N 0, σ2 , where the standard deviation is defined as σ = Noise Level · max(X). Experiments are conducted with PatchTST using ADAMW optimizer and Test MSE is reported.

Table 4. Combining GradientStabilizer with other optimizers. Results are reported for Adam-Mini and Lion on LLaMA130M/350M under FP4 training. The best results are in bold, and the second-best are underlined.

ADAM-MINI LION 130M 350M 130M 350M

Method

Base Optimizer 31.66 20.36 28.52 23.59 + VALUE CLIP (Pascanu et al., 2013) 31.52 20.99 28.32 23.20 + NORM CLIP (Pascanu et al., 2013) 30.54 21.03 28.23 22.98 + AGC (Brock et al., 2021) 30.41 21.84 28.33 23.20 + ZCLIP (Kumar et al., 2025) 30.50 21.14 28.16 23.13 + GradientStabilizer 27.75 18.64 26.72 22.75

Training Tokens 2.2B 6.6B 2.2B 6.6B

max(X) and Noise Level control the corruption magnitude. Results in Figure 6 demonstrate that ADAM + GradientStabilizer significantly reduces the final Test MSE across all tested noise levels. Notably, the benefits of our method scale with the severity of the corruption; as the noise level increases from 1 to 3, the performance gain attributed to GradientStabilizer rises from approximately 0.008 to 0.025. This trend underscores the method’s robustness in counteracting the adverse effects of input corruption. Comparisons with gradient clipping baselines are provided in Appendix C.3.

Combined with other optimizers. Our method is designed as an optimizer-agnostic gradient stabilizer. To further assess its generalizability, we evaluate it in conjunction with two recent optimizers: LION (Chen et al., 2023) and ADAMMINI (Zhang et al., 2024). Experiments are conducted on LLaMA-130M and 350M models using FP4 training on the C4 dataset. The results in Table 4 demonstrate that applying GradientStabilizer to LION and ADAM-MINI yields significant improvements over the base optimizers. Furthermore, compared to standard gradient clipping baselines, our method achieves the largest and most consistent gains, underscoring its superior compatibility.

Hyper-parameters Analysis. GradientStabilizer introduces two hyperparameters, γ1 and γ2, which control the decay rates for the first and second moments of the

3.31

| | | | |
|---|---|---|---|
| | | | |

ValidationLoss()

3.29

= 0.015

3.27

0.4 0.6 0.8

0.95 0.999 0.99999

1

2

Figure 7. Hyper-parameter analysis. Experiments are conducted with FP4 training on LLaMA-130M and C4 with 2.2B tokens.

gradient norm. To investigate the sensitivity of the method to these parameters, we analyze the final validation loss while varying γ1 ∈ [0.5,0.8] (fixing γ2 = 0.999) and γ2 ∈ [0.95,0.99999] (fixing γ1 = 0.6). The results in Figure 7 indicate that while GradientStabilizer is relatively more sensitive to γ2 than γ1, it remains highly robust overall. Specifically, the maximum variation in validation loss across these broad ranges is less than 0.02, demonstrating that the method is stable across a wide configuration space.

### 5. Related Work

Training Instability.The training dynamics of deep neural networks are constrained by non-convex loss landscapes and pathological curvature, often leading to severe optimization difficulties (Martens et al., 2010). Early analyses by Bengio et al. (1994) and Glorot & Bengio (2010) identified the fundamental impediments of vanishing and exploding gradients, revealing that gradient norms can deviate exponentially with depth. In the context of Recurrent Neural Networks, Pascanu et al. (2013) demonstrated that error surfaces frequently exhibit steep cliffs, causing gradients to explode and weights to oscillate or diverge. Analogous instabilities persist in Reinforcement Learning (RL), where non-stationary data distributions and bootstrapping can induce value function degradations and even divergence (Sutton et al., 1998; Dasagi et al., 2019; Park et al., 2025). In the era of Large Language Models (LLMs), optimization instability manifests as sudden loss spikes that can reduce performence or even derail training entirely (Huang et al., 2025; Takase et al., 2023). Chowdhery et al. (2023) and Liu et al. (2025) report that scaling models beyond 60B parameters introduces critical gradient irregularities, often necessitating heuristic restart strategies or aggressive gradient clipping. This instability is exacerbated in low-bit training regimes, where the reduced dynamic range of quantized formats is ill-equipped to handle the heavy-tailed activation distributions inherent to large-scale models, leading to quantizationinduced divergence (Wortsman et al., 2023; Panferov et al., 2025; Hao et al., 2025).To address these instabilities, researchers have developed various stabilization techniques. One prominent approach involves architectural and initial-

ization modifications, such as relocating LayerNorm (Xiong

- et al., 2020), inserting additional normalization layers after embeddings (Dettmers et al., 2021), and employing initialization schemes with reduced variance (Nguyen & Salazar, 2019). Other methods specifically target embedding dynamics by shrinking embedding gradients via reweighting (Ding et al., 2021; Zeng et al., 2022) or upscaling embeddings to stabilize LayerNorm gradients (Takase et al., 2023). A complementary line of work focuses on gradient constraints, utilizing clipping mechanisms that employ either soft scaling or hard truncation to suppress anomalous gradient magnitudes during backpropagation.

Gradient Clipping Techniques. To mitigate the risk of divergent optimization trajectories, clipping mechanisms constrain the magnitude of parameter updates. The prevailing paradigm is global gradient norm/value clip (Pascanu et al., 2013), which rescales the raw gradient vector whenever its L2 norm/value exceeds a fixed hyperparameter. To address the sensitivity of fixed thresholds, adaptive variants have emerged: Adaptive Gradient Clipping (AGC) (Brock

- et al., 2021) and Clippy (Tang et al., 2023) dynamically modulate clipping thresholds relative to the norm of the model weights. More recently, Wang et al. (2025) and Kumar et al.

(2025) proposed to identify and clip gradient anomalies by comparing the current gradient norm against a local moving average. In contrast to these anomaly-detection strategies, GradientStabilizer adaptively transforms the gradient norm without explicit outlier detection, enforcing intrinsic stability within the training dynamics.

### Acknowledgements

The authors acknowledge the use of computing resources provided by the NVIDIA Academic Grant Program, the Isambard-AI National AI Research Resource (AIRR) and the Dutch national e-infrastructure, supported by the SURF Cooperative (Project EINF-17091).

### Impact Statement

The goal of this research is to advance the fundamental stability of neural network training. Optimization difficulties often act as a barrier to entry in deep learning, requiring significant computational budgets to tune hyperparameters and manage instabilities. By introducing a threshold-free mechanism that stabilizes training across diverse domains including LLMs, time-series forecasting, and RL, our work simplifies the training pipeline and broadens the stable operating region for optimizers. This improved reliability can help democratize access to large-scale model training for researchers with constrained computational resources. We do not foresee any direct negative societal consequences specific to this method, though we acknowledge the general dual-use nature of advancing deep learning performance.

### 6. Conclusion

In this paper, we introduced GradientStabilizer, an optimizer-agnostic gradient transformation that preserves update direction while regulating step magnitude via running norm statistics, offering a threshold-free alternative to heuristic clipping. Our theoretical analysis characterizes the variance-dampening properties of this approach in stationary regimes and establishes spike-invariant bounds, guaranteeing that transient gradient anomalies do not result in unbounded effective updates. Furthermore, we show that these bounds inherently regularize the moment states of Adam-style optimizers, ensuring well-defined dynamics per coordinate. Empirically, GradientStabilizer demonstrates superior stability across diverse tasks including low-precision LLM pre-training (FP16/FP4), ImageNet classification, RL, and time-series forecasting. By widening stable learning-rate regions and mitigating sensitivity to weight decay, our method offers a robust, drop-in solution for scaling deep learning optimization.

### References

Bengio, Y., Simard, P., and Frasconi, P. Learning long-term dependencies with gradient descent is difficult. IEEE transactions on neural networks, 5(2):157–166, 1994.

Brock, A., De, S., Smith, S. L., and Simonyan, K. Highperformance large-scale image recognition without normalization. In International conference on machine learning, pp. 1059–1071. PMLR, 2021.

Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., et al. Symbolic discovery of optimization algorithms. Advances in neural information processing systems, 36:49205–49233, 2023.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Dasagi, V., Bruce, J., Peynot, T., and Leitner, J. Ctrl-z: Recovering from instability in reinforcement learning. arXiv preprint arXiv:1910.03732, 2019.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L. 8bit optimizers via block-wise quantization. arXiv preprint arXiv:2110.02861, 2021.

Ding, M., Yang, Z., Hong, W., Zheng, W., Zhou, C., Yin, D., Lin, J., Zou, X., Shao, Z., Yang, H., et al. Cogview: Mastering text-to-image generation via transformers. Advances in neural information processing systems, 34: 19822–19835, 2021.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Glorot, X. and Bengio, Y. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256. JMLR Workshop and Conference Proceedings, 2010.

Gupta, V., Koren, T., and Singer, Y. Shampoo: Preconditioned stochastic tensor optimization. In International Conference on Machine Learning, pp. 1842–1850. PMLR, 2018.

Hao, Z., Guo, J., Shen, L., Luo, Y., Hu, H., Wang, G., Yu, D., Wen, Y., and Tao, D. Low-precision training of large language models: Methods, challenges, and opportunities. arXiv preprint arXiv:2505.01043, 2025.

He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

He, Z., Luo, X., Zhang, Y., Yang, Y., and Qiu, L. Vl norm: Rethink loss aggregation in rlvr. arXiv preprint arXiv:2509.07558, 2025.

Huang, T., Zhu, Z., Jin, G., Liu, L., Wang, Z., and Liu, S. Spam: Spike-aware adam with momentum reset for stable llm training. ICLR, 2025.

Kingma, D. P. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Kumar, A., Owen, L., Chowdhury, N. R., and G¨ura, F. Zclip: Adaptive spike mitigation for llm pre-training. arXiv preprint arXiv:2504.02507, 2025.

Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.

Liu, Q. and Ma, W. Navigating data corruption in machine learning: Balancing quality, quantity, and imputation strategies. Future Internet, 17(6):241, 2025.

Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11976–11986, 2022.

Liu, Z., Tan, B., Wang, H., Neiswanger, W., Tao, T., Li, H., Koto, F., Wang, Y., Sun, S., Pangarkar, O., et al. Llm360 k2: Scaling up 360-open-source large language models. arXiv e-prints, pp. arXiv–2501, 2025.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Martens, J. et al. Deep learning via hessian-free optimization. In Icml, volume 27, pp. 735–742, 2010.

Nguyen, T. Q. and Salazar, J. Transformers without tears: Improving the normalization of self-attention. arXiv preprint arXiv:1910.05895, 2019.

Nie, Y. A time series is worth 64words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730, 2022.

Panferov, A., Chen, J., Tabesh, S., Castro, R. L., Nikdan, M., and Alistarh, D. Quest: Stable training of llms with 1-bit weights and activations. arXiv preprint arXiv:2502.05003, 2025.

Park, J., Lee, J., Kim, J., and Han, S. Overcoming intermittent instability in reinforcement learning via gradient norm preservation. Information Sciences, 709:122081, 2025.

Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In International conference on machine learning, pp. 1310–1318. Pmlr, 2013.

Reddi, S. J., Kale, S., and Kumar, S. On the convergence of adam and beyond. arXiv preprint arXiv:1904.09237, 2019.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Shah, M. B., Rahman, M. M., and Khomh, F. Towards understanding the impact of data bugs on deep learning models in software engineering. Empirical Software Engineering, 30(6):168, 2025.

Shazeer, N. and Stern, M. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning, pp. 4596–4604. PMLR, 2018.

Sutton, R. S., Barto, A. G., et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

Takase, S., Kiyono, S., Kobayashi, S., and Suzuki, J. Spike no more: Stabilizing the pre-training of large language models. arXiv preprint arXiv:2312.16903, 2023.

Talak, R., Georgiou, C., Shi, J., and Carlone, L. Outlierrobust training of machine learning models. arXiv preprint arXiv:2501.00265, 2024.

Tang, J., Drori, Y., Chang, D., Sathiamoorthy, M., Gilmer, J., Wei, L., Yi, X., Hong, L., and Chi, E. H. Improving training stability for multitask ranking models in recommender systems. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4882–4893, 2023.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi`ere, B., Goyal, N., Hambro, E., Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Wang, G., Li, S., Chen, C., Zeng, J., Yang, J., Sun, T., Ma, Y., Yu, D., and Shen, L. Adagc: Improving training

stability for large language model pretraining. arXiv preprint arXiv:2502.11034, 2025.

Wortsman, M., Dettmers, T., Zettlemoyer, L., Morcos, A., Farhadi, A., and Schmidt, L. Stable and low-precision training for large-scale vision-language models. Advances in Neural Information Processing Systems, 36: 10271–10298, 2023.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. On layer normalization in the transformer architecture. In International conference on machine learning, pp. 10524–10533. PMLR, 2020.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., et al. Glm-130b: an open bilingual pre-trained model. arxiv. arXiv preprint arXiv:2210.02414, 2022.

Zeng, G., Zhou, Z., Arora, D., and Zanette, A. Shrinking the variance: Shrinkage baselines for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2511.03710, 2025.

Zhang, J. Survey of quantization-aware training (qat) applications in deep learning quantization. In Proceedings of the 2025 International Symposium on Artificial Intelligence and Computational Social Sciences, pp. 431–442, 2025.

Zhang, Y., Chen, C., Li, Z., Ding, T., Wu, C., Kingma, D. P., Ye, Y., Luo, Z.-Q., and Sun, R. Adam-mini: Use fewer learning rates to gain more. arXiv preprint arXiv:2406.16793, 2024.

### A. Proofs

- A.1. Proof of Lemma 3.2 Proof. Since E[R2] = Var(R) + (E[R])2 = σ2 + µ2, we have

ρ⋆ =

µ µ2 + σ2

=

1 1 + σ2/µ2

=

1 1 + c2v ≤ 1.

| |
|---|

- A.2. Proof of Lemma 3.4

Proof. We analyze the behavior of the stabilized magnitude ρt at a specific time step t where a ”gradient spike” occurs. Let the first and second moment estimators of the gradient norm Rt follow the standard exponential moving average (EMA) updates with decay rates γ1 and γ2:

mRt = γ1mRt−1 + (1 − γ1)Rt, (11)

vtR = γ2vtR−1 + (1 − γ2)Rt2. (12) where mRt >,vtR > 0 since the gradient norm Rt > 0. Given the stabilized magnitude is defined by the ratio:

ρt =

mRt vtR

=

γ1mRt−1 + (1 − γ1)Rt γ2vtR−1 + (1 − γ2)Rt2

. (13)

Since vtR−1 ≥ 0 and γ2 ≥ 0, therefore, we can lower bound the term inside the square root by ignoring the historical second moment:

γ2vtR−1 + (1 − γ2)Rt2 ≥ (1 − γ2)Rt2 = Rt 1 − γ2. (14) Applying this lower bound to the denominator yields an upper bound for ρt:

ρt ≤

γ1mRt−1 + (1 − γ1)Rt Rt√1 − γ2

=

γ1 √1 − γ2

mRt−1 Rt

+

1 − γ1 √1 − γ2

.. (15)

The lemma assumes a spike condition Rt ≥ κmRt−1 for some threshold κ ≫ 1. We can rearrange this inequality to bound the ratio of the historical moment to the current norm:

mRt−1 Rt ≤

1 κ

. (16)

Substituting this inequality into the expression for ρt, we have:

ρt ≤

1 − γ1 √1 − γ2

+

γ1 κ√1 − γ2

. (17)

| |
|---|

- A.3. Proof of Lemma 3.6 Proof. With m(0R) = v0(R) = 0, unrolling the EMA recursions yields

t

m(tR) = (1 − γ1)

k=1

t

vt(R) = (1 − γ2)

k=1

γ1t−kRk, (18)

γ2t−kRk2. (19)

Let wk = γ2t−k > 0. By Cauchy–Schwarz inequality,

t

t

γ1t−k √wk

√wkRk ≤

γ1t−kRk =

k=1

k=1

t−1

j 12 t

γ12 γ2

γ2t−kRk2

=

j=0

k=1

t

1 − (γ12/γ2)t 1 − γ12/γ2

γ2t−kRk2

=

k=1

since γ12 < γ2 so 1 − (γ12/γ2)t ≤ 1, then we have:

t

1 − (γ12/γ2)t 1 − γ12/γ2

γ1t−kRk ≤

k=1

t

γ12(t−k) wk

k=1

- 1

- 2

- 1

- 2

t

wkRk2

k=1

- 1

- 2

- 1

- 2

, (20)

t

γ2t−kRk2

k=1

- 1

- 2

Combining (18)–(20) and using (19), we obtain

1 1 − γ12/γ2

m(tR) ≤ (1 − γ1) ·

Dividing both sides by vt(R) yields

t

γ2t−kRk2

k=1

- 1

- 2

1 − γ1 √1 − γ2 ·

1 1 − γ12/γ2

=

vt(R).

The proof is complete.

ρt =

m(tR) vt(R)

≤

1 − γ1 √1 − γ2 ·

1 1 − γ12/γ2

= ρ.¯

| |
|---|

- A.4. Proof of Proposition 3.8 Proof. Fix any t ≥ 1. By the conclusion of Lemma 3.6, we have ∥g˜t∥2 ≤ ρ¯. Moreover, for any vector x ∈ Rd it holds that ∥x∥∞ ≤ ∥x∥2 since for each coordinate i, |xi| ≤ dj=1 x2j = ∥x∥2. Applying this inequality to x = g˜t yields

∥g˜t∥∞ ≤ ∥g˜t∥2 ≤ ρ.¯

Finally, ∥g˜t∥∞ = maxi |g˜t,i| implies |g˜t,i| ≤ ρ¯ for all i, completing the proof.

| |
|---|

- A.5. Proof of Theorem 3.9 Proof. Fix any coordinate i ∈ [d]. By Proposition 3.8, we have |g˜t,i| ≤ ρ¯ for all t ≥ 1.

Bound on mt. The base case holds since m0,i = 0 ≤ ρ¯(1 − β10). For t ≥ 1, using (5) and the triangle inequality,

|mt,i| = |β1mt−1,i + (1 − β1)˜gt,i| ≤ β1|mt−1,i| + (1 − β1)|g˜t,i| ≤ β1|mt−1,i| + (1 − β1)¯ρ. Unrolling the inequality from m0,i = 0 yields

t−1

|mt,i| ≤ (1 − β1)¯ρ

k=0

Taking the maximum over i gives ∥mt∥∞ ≤ ρ¯(1 − β1t) ≤ ρ¯.

β1k = ρ¯(1 − β1t).

Bound on vt. Since v0,i = 0 and g˜t,i2 ≥ 0, the recursion (6) implies vt,i ≥ 0 for all t. Moreover, by (6) and |g˜t,i| ≤ ρ¯,

vt,i = β2vt−1,i + (1 − β2)˜gt,i2 ≤ β2vt−1,i + (1 − β2)¯ρ2. Unrolling from v0,i = 0 yields

t−1

vt,i ≤ (1 − β2)¯ρ2

β2k = ρ¯2(1 − β2t).

k=0

Taking the maximum over i gives ∥vt∥∞ ≤ ρ¯2(1 − β2t) ≤ ρ¯2.

Bound on vˆt (AMSGrad). For AMSGrad, vˆt is defined elementwise by (7), with vˆ0,i = 0. Thus vˆt,i = max{vˆt−1,i,vt,i} = max1≤s≤t vs,i. Using the bound on vs,i above and the fact that 1 − β2s is nondecreasing in s,

ρ¯2(1 − β2s) = ρ¯2(1 − β2t).

vˆt,i ≤ max

1≤s≤t

Taking the maximum over i yields ∥vˆt∥∞ ≤ ρ¯2(1 − β2t) ≤ ρ¯2. Combining the three bounds completes the proof.

| |
|---|

#### A.6. Bounded Per-step Parameter Change for Adam/AMSGrad

Corollary A.1 (Bounded Per-step Parameter Change under GradientStabilizer for Adam/AMSGrad ). Consider Adam/AMSGrad updates with an explicit ϵ > 0 in the denominator:

mt √vˆt + ϵ

, (21)

θt+1 = θt − αt

where division and square-root are elementwise and vˆt = vt for Adam. Under the conditions of Theorem 3.9, with β1,β2 ∈ [0,1), for all t ≥ 1 and all coordinates i ∈ [d],

ρ¯(1 − β1t) vˆt,i + ϵ

. (22)

|θt+1,i − θt,i| ≤ αt

Proof. From the update rule (21) in main text, all operations are coordinatewise, so for any i ∈ [d],

mt,i vˆt,i + ϵ

θt+1,i − θt,i = −αt

.

Taking absolute values gives

|θt+1,i − θt,i| = αt |mt,i| vˆt,i + ϵ

.

By Theorem 3.9, we have the tight first-moment bound ∥mt∥∞ ≤ ρ¯(1 − β1t), and hence |mt,i| ≤ ρ¯(1 − β1t) for every coordinate i. Substituting this inequality into the display above yields

ρ¯(1 − β1t) vˆt,i + ϵ

|θt+1,i − θt,i| ≤ αt

,

which proves the claim.

Remark A.2. Corollary A.1 shows that, under GradientStabilizer, each coordinate update is controlled by the intrinsic bound ρ¯ on the stabilized gradient magnitude, rather than by the (possibly unbounded) raw gradients. As a result, raw gradient explosions cannot induce a catastrophic single-step parameter jump, ensuring training stability even in the presence of arbitrarily large gradient spikes.

- A.7. Proof of Corollary 3.11 Proof. Recall the update rule from Corollary 3.11, we have

θt+1 − θt = −ηtg˜t, Under Proposition 3.8 and thus

∥θt+1 − θt∥∞ = ηt∥g˜t∥∞ ≤ ηt∥g˜t∥2, Under the conditions of Lemma 3.6, ∥g˜t∥2 ≤ ρ¯ for all t ≥ 1, which yields

∥θt+1 − θt∥∞ ≤ ηtρ.¯

Moreover, on spike steps satisfying Rt ≥ κm(t−R)1, Lemma 3.4 gives

and therefore implies

∥g˜t∥2 ≤

1 − γ1 √1 − γ2

+

γ1 κ√1 − γ2

,

∥θt+1 − θt∥∞ ≤ ηt

1 − γ1 √1 − γ2

γ1 κ√1 − γ2

+

.

| |
|---|

### B. Experimental Details for Reproducibility

- B.1. Codes for gradient clipping baselines We adopt the official code for the gradient clipping baselines.

- • AGC:https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/ agc.py
- • ZCLIP:https://github.com/bluorion-com/ZClip
- • NORM CLIP:https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/utils/clip_ grad.py#L185
- • VALUE CLIP: https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/utils/clip_ grad.py#L257

Table 5 summarizes the hyper-parameter settings for our method and all baselines; these values are fixed across all experiments in the main paper.

Table 5. Hyper-parameter settings for GradientStabilizer and clipping baselines (fixed across all experiments). Method Hyper-parameters GradientStabilizer γ1 = 0.6, γ2 = 0.999 VALUE CLIP (Pascanu et al., 2013) threshold = 0.1 NORM CLIP (Pascanu et al., 2013) threshold = 1.0 AGC (Brock et al., 2021) clipping factor = 0.01 ZCLIP (Kumar et al., 2025) α = 0.97, z-score threshold = 2.5

Table 6. Training hyper-parameters for LLaMA models under Adam and AdamW. Model Optimizer LR Batch-size Total batch-size Warmup Steps LLaMA-130M

Adam 1e-3 128 512 2000 20000 AdamW 1e-3 128 512 2000 20000

Adam 1e-3 128 512 2000 60000 AdamW 1e-3 128 512 2000 60000

LLaMA-350M

Weight decay: Adam uses 0; AdamW uses 0.01.

#### B.2. Training configurations for Experiments in main content

LLM-Pretraining and Quantization-Aware Trainig: We implement our training framework based on the GaLore codebase https://github.com/jiaweizzhao/GaLore. For Quantization-Aware Training (QAT), we employ FP4 bit (E1M2 format: 1-bit exponent, 2-bit mantissa). Specifically, we quantize all weights and activations to 4-bit floating point (FP4) precision. Table 6 shows the details of hyper-parameter settings for the training.

ImageNet-1K Classification. We train ConvNeXt-T, ResNet-50, ViT-B using the default training scripts provided by torchvision but with 120 epochs only due to the limiation of our computing resources.

- • ConvNeXt-T:https://github.com/pytorch/vision/tree/main/references/ classification#convnext.
- • ViT-B:https://github.com/pytorch/vision/tree/main/references/classification# vit_b_16
- • ResNet-50: we use the same training script as ConvNeXt-T. https://github.com/pytorch/vision/tree/ main/references/classification#convnext.

Reinforcement Learning. We are training the policy network for the MuJoCo environment using the training scripts provided by the Tianshou framework https://github.com/thu-ml/tianshou/tree/master/examples/mujoco. Table 7 shows the hyper-parameter settings for reinforcement learning experiments.

Table 7. Optimizer hyper-parameters for MuJoCo policy training (Tianshou). Optimizer Learning rate Weight decay

Adam 5 × 10−4 0 AdamW 5 × 10−4 1 × 10−2

Time Series Forescasting.We trains PatchTST based on their official public code and scripts https://github.com/ yuqinie98/PatchTST/blob/main/PatchTST_supervised/scripts/PatchTST/weather.sh. Table 8 shows the hyper-parameter settings for the TSF task.

Table 8. Optimizer hyper-parameters for Time Series Forescasting task. Optimizer Learning rate Weight decay

Adam 1 × 10−4 0 AdamW 1 × 10−4 1 × 10−2

### C. Additional Experiments

#### C.1. Weight Decay Stability on ADAM for ResNet-50 and ConvNeXt-T.

The results in Table 9 demonstrate that as weight decay strength varies from 0 to 10−4, ADAM and all gradient clipping baselines experience a substantial drop in performance. In contrast, GradientStabilizer not only avoids this degradation but yields improved performance, highlighting its effectiveness in mitigating ADAM’s sensitivity to weight decay.

Table 9. Weight-decay stability on Adam. Top-1 accuracy (%) of ConvNeXt-T and ResNet-50 under different weight-decay strengths using Adam on ImageNet-1K. The best results are in bold.

ConvNeXt-T ResNet-50 WD=0 WD=1e-4 WD=0 WD=1e-4

Method

Base Optimizer 77.6 35.2 75.5 69.8 + Value Clip (Pascanu et al., 2013) 78.1 15.7 75.6 69.8 + Norm Clip (Pascanu et al., 2013) 78.2 27.2 75.7 64.5 + AGC (Brock et al., 2021) 76.9 2.6 75.8 64.2 + Zclip (Kumar et al., 2025) 78.1 26.4 75.8 64.5 + GradientStabilizer 78.3 78.8 75.8 76.6

Training Dataset ImageNet-1K ImageNet-1K ImageNet-1K ImageNet-1K

#### C.2. Comparison with Gradient Clipping baselines on Learning rate and Training Stability.

Learning rate stability. We compare learning rate stability against gradient clipping baselines in Figure 8 (Left). The results demonstrate that GradientStabilizer exhibits superior stability compared to all baselines across both low and high learning rate regimes. Additionally, we observe that ZCLIP and NORM CLIP achieve comparable stability in high learning rate regions.

Training stability. Figure 8 (Right) compares the training stability of our method against gradient clipping baselines. The results demonstrate that GradientStabilizer maintains lower gradient norms and yields smoother training loss curves than the baselines, highlighting its effectiveness in stabilizing training dynamics.

Adam

AdamW

Gradient Norm

Training Loss

100

ValidationLoss

| | | | |
|---|---|---|---|
| | | | |
| | | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |

10

- 3
- 4
- 5

50

5

0

10 4 10 3 3 × 10 3 LR

10 4 10 3 3 × 10 3 LR

0 5k 10k Step

0 5k 10k Step

Base Optimizer +Ours +Value Clip +Norm Clip +ZClip +AGC

- Figure 8. Training and learning-rate stability. Left: validation loss under different learning rates, illustrating learning-rate stability. Right: raw gradient norm (before GradientStabilizer) and training loss curves, illustrating training stability. All experiments are conducted on LLaMA-130M FP4 training on C4.

C.3. Comparison with Gradient Clipping baselines on Corrupted Data.

- Figure 9 compares our method against gradient clipping baselines on corrupted data, using the same experimental setup as Figure 6. GradientStabilizer consistently outperforms or matches baselines across all noise levels. Notably, AGC and ZCLIP also demonstrate competitive performance.

#### C.4. Experiments on RL Tasks with Additional Environments

We further evaluate our method on the ANT-V4, HOPPER-V4, and WALKER2D-V4 environments using the AdamW optimizer. The results presented in Figure 10 demonstrate that GradientStabilizer consistently achieves the highest rewards among all gradient clipping baselines across these tasks. Notably, while we observe that standard gradient clipping baselines suffer from performance degradation compared to the base optimizer on WALKER2D-V4, GradientStabilizer effectively maintains performance parity with the unclipped baseline.

Noise Level = 1

Noise Level = 2

Noise Level = 3

0.170.200.24

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |
| | | |

0.160.180.20 TestMSE()

0.180.230.27

0 50 100 Epoch

0 50 100 Epoch

0 50 100 Epoch

Base Optimizer

+Norm Clip

+AGC +Ours

+Value Clip

+ZClip

- Figure 9. Test MSE under input perturbations on Weather time-series data. We corrupt 5% of randomly selected input by adding zero-mean Gaussian noise. Specifically, the Gaussian noise is samples from N 0, σ2 , where the standard deviation is defined as σ = Noise Level · max(X). Experiments are conducted with PatchTST using ADAMW optimizer and Test MSE is reported.

| | | | |
|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| | |
| | | | |
| | | | |
| | | | |

0 50 100 Epoch

0

2K

4K

TestReward()

Ant-V4

| | | | |
|---|---|---|---|
| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
| || |
|---|
<br><br>| | |
| | | | |
| | | | |

0 50 100 Epoch

Hopper-V4

0 50 100 Epoch

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

Walker2d-V4

Base Optimizer

+Value Clip

+Norm Clip

+ZClip

+AGC +Ours

- Figure 10. Reinforcement learning on Ant-V4, Hopper-V4 and Walker-V4. Mean episodic return ± standard deviation over 10× evaluation rollouts, plotted against training epochs.

