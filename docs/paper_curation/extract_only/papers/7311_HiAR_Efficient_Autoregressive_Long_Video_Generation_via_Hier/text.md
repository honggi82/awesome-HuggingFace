# arXiv:2603.08703v1[cs.CV]9Mar2026

## HiAR: Efficient Autoregressive Long Video Generation via Hierarchical Denoising

#### Kai Zou1,5 Dian Zheng2 Hongbo Liu3 Tiankai Hang4 Bin Liu1,5* Nenghai Yu1,5

1University of Science and Technology of China 2The Chinese University of Hong Kong 3Tongji University 4Tencent Hunyuan 5Anhui Province Key Laboratory of Digital Security, USTC kzou@mail.ustc.edu.cn, *Corresponding author. https://jacky-hate.github.io/HiAR/

### Abstract

Autoregressive (AR) diffusion offers a promising framework for generating videos of theoretically infinite length. However, a major challenge is maintaining temporal continuity while preventing the progressive quality degradation caused by error accumulation. To ensure continuity, existing methods typically condition on highly denoised contexts; yet, this practice propagates prediction errors with high certainty, thereby exacerbating degradation. In this paper, we argue that a highly clean context is unnecessary. Drawing inspiration from bidirectional diffusion models, which denoise frames at a shared noise level while maintaining coherence, we propose that conditioning on context at the same noise level as the current block provides sufficient signal for temporal consistency while effectively mitigating error propagation. Building on this insight, we propose HiAR, a hierarchical denoising framework that reverses the conventional generation order: instead of completing each block sequentially, it performs causal generation across all blocks at every denoising step, so that each block is always conditioned on context at the same noise level. This hierarchy naturally admits pipelined parallel inference, yielding a ∼1.8× wall-clock speedup in our 4-step setting. We further observe that selfrollout distillation under this paradigm amplifies a low-motion shortcut inherent to the mode-seeking reverse-KL objective. To counteract this, we introduce a forward-KL regulariser in bidirectional-attention mode, which preserves motion diversity for causal inference without interfering with the distillation loss. On VBench (20s generation), HiAR achieves the best overall score and the lowest temporal drift among all compared methods.

### 1 Introduction

Recent years have witnessed rapid progress in video generation, with Diffusion Transformer (DiT) Peebles and Xie [2023] backbones powering strong foundation models Ho et al. [2022], Blattmann et al. [2023], Yang et al. [2024], Polyak et al. [2024], Zheng et al. [2024], Team [2025], Brooks et al. [2024] and conditional paradigms—including image-to-video and video-to-videofurther broadening controllable generation. A remaining frontier is long-horizon, and ultimately open-ended, video generation, central to interactive agents and world models He et al. [2025], Ye et al. [2025], Mao et al. [2025], Sun et al. [2025], Hong et al. [2025], Tang et al. [2026]. To scale video duration, causal autoregressive (AR) generation Wu et al. [2025], Jin et al. [2024b], Teng et al. [2025a], Chen et al. [2025a] is increasingly attractive: it supports streaming output, indefinite extension, and real-time interaction.

However, a critical challenge in this pipeline is maintaining strict temporal continuity between consecutive video blocks while simultaneously preventing distribution drift (e.g., oversaturation,

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Wan2.1

[Figure 7]

Quality: Stable Length: Fixed

[Figure 8]

[Figure 9]

0s 1s 2s 3s 4s Continuity: Good

(a). Bidirectional Diffusion Model

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Self Forcing

[Figure 15]

[Figure 16]

Quality: Degraded Length: Scalable Continuity: Good

0s 5s 10s 15s 20s

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Self Forcing

[Figure 23]

0s

50s 100s 150s 200s

(b). Causal AR with high clean context

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

w/o training

[Figure 30]

Quality: Stable Length: Scalable Continuity: Bad

[Figure 31]

[Figure 32]

0s 5s 10s 15s 20s

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

w/ training

[Figure 38]

[Figure 39]

Quality: Stable Length: Scalable Continuity: Good

0s 5s 10s 15s 20s

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

w/ training

[Figure 46]

0s 50s 100s 150s 200s

(c). HiAR (Ours)

- Figure 1: Motivation. (a) Bidirectional diffusion (Wan2.1) proves that a shared noise level provides sufficient context for temporal coherence, though limited to a fixed horizon. (b) Standard AR (SelfForcing) scales length but suffers quality drift, as conditioning on fully clean context amplifies error propagation. (c) Applying our hierarchical denoising (matched-noise context) only at inference (w/o training) mitigates drift but breaks continuity due to train–test mismatch; HiAR retrains under the hierarchical pipeline (w/ training), achieving scalable long-video generation with stable quality and seamless continuity.

over-sharpening, motion repetition, and semantic drift) caused by error accumulation. To ensure temporal coherence, existing methods mainly denoise the previous frames into a highly clean context before generating the next. Consequently, every denoising step of the current block is conditioned on a context with noise level tc = 0 (maximal SNR). While this highly clean context anchors the temporal consistency, it inadvertently causes the model to propagate accumulated prediction errors forward with high confidence, thereby exacerbating the degradation, as illustrated in fig. 1(b).

In this work, we recognize that a highly clean context is not a prerequisite. Drawing inspiration from bidirectional diffusion models, which denoise all frames concurrently from a shared noise level, yet still yield temporally coherent videos, as shown in fig. 1(a), demonstrating that noisy context already provides sufficient signal for continuity while reducing error propagation. Based on this principle, we introduce HiAR, a Hierarchical Denoising paradigm that swaps the denoising order: instead of fully denoising previous blocks first, we perform causal generation across all blocks within each denoising step, then move to the next step. This simple yet fundamental change substantially reduces inter-block error transmission and improves long-horizon stability as shown in fig. 1(c, w/o training). Moreover, the hierarchical structure enables pipelined parallelism across denoising steps at inference time, improving wall-clock efficiency (×1.8).

To maintain train–test consistency, we retrain under the hierarchical denoising pipeline. However, we find that self-rollout distillation Anonymous [2025], Yin et al. [2024b] exhibits a low-motion shortcut that worsens over training—consistent with the mode-seeking tendency of DMD-style reverse-KL objectives Lu et al. [2025a]: the model gradually collapses into near-static outputs that minimise distillation loss but sacrifice dynamics. Hierarchical denoising amplifies this effect,

- as the increased learning difficulty of conditioning on multi-level noisy contexts requires more training steps. Empirically, we find that motion diversity under bidirectional-attention denoising is strongly correlated with that under causal AR inference. Motivated by this observation, we introduce

a distillation-based forward-KL regulariser computed in bidirectional-attention denoising mode, effectively preventing dynamics collapse for the causal inference path and enabling stable long-step training.

We conduct extensive evaluation on VBench Huang et al. [2024], Zheng et al. [2025] and a dedicated drift metric tailored to long-horizon rollouts, together with thorough ablations, demonstrating that HiAR yields more stable long video generation and validating the contribution of each component. The visual result is shown in fig. 1(c, w/ training).

We highlight the main contributions of this paper below:

- • We propose HiAR, a hierarchical denoising pipeline that performs causal generation across blocks within each denoising step, substantially reducing inter-block error transmission and enabling pipelined inference across hierarchy levels for ∼1.8× wall-clock speedup in our implementation.
- • We introduce a simple forward-KL regulariser via bidirectional-attention distillation to prevent low-motion shortcuts in self-rollout training, enabling stable scaling to long training schedules while preserving dynamics.
- • Extensive experiments on VBench and a dedicated drift metric, together with thorough ablations, demonstrate the long-horizon stability and the effectiveness of each component.

### 2 Background

#### 2.1 Diffusion Models and Flow Matching

Diffusion-based generative models Ho et al. [2020], Song et al. [2021] learn to reverse a forward noising process that gradually corrupts data into Gaussian noise. In this work we adopt the flow matching formulation Lipman et al. [2023], Liu et al. [2023], Albergo and Vanden-Eijnden [2023]. Let x0 ∼ pdata denote a clean data sample and ϵ ∼ N(0,I) standard Gaussian noise. The forward interpolation (corruption) at continuous time t ∈ [0,1] is defined as

s · t 1 + (s − 1) · t

, (1)

xt = (1 − σt)x0 + σt ϵ, σt =

where s > 0 is a shift parameter that controls the noise schedule curvature. At t = 0 we recover x0; at t = 1 we obtain (approximately) pure noise. A neural network vθ(xt,t) is trained to predict the velocity field

v∗(xt,t) = ϵ − x0, (2) so that clean data can be recovered by integrating the probability-flow ODE backward from t = 1 to t = 0. In practice, one discretises the trajectory into S steps 1 = t1 > t2 > ··· > tS > 0 and applies the Euler update

. (3)

j+1 − σt

xt

= xt

+ vθ(xt

,tj) σt

j+1

j

j

j

#### 2.2 Autoregressive Video Diffusion

Bidirectional-attention diffusion models OpenAI [2025], Wan et al. [2025], Kling [2025], Google [2025], Runway [2025] operate on a fixed temporal window and cannot easily scale to arbitrary durations. Causal AR generation Po et al. [2025], Liu et al. [2025], Lu et al. [2025b], Zhang et al. [2025], Yang et al. [2025], Lin et al. [2025] overcomes this limitation by generating frames in a streaming manner: it naturally supports indefinite extension, allows real-time intervention, and provides a principled interface for interactive control—making it a key building block toward world models He et al. [2025], Ye et al. [2025], Mao et al. [2025]. To generate videos beyond a fixed temporal window, recent work partitions the video latent sequence into N successive blocks {B1,...,BN}, each containing k frames, and generates them autoregressively: for n = 2,...,N, block Bn is denoised conditioned on the previously generated blocks B<n.

Concretely, let x(tn) denote the noisy latent of block n at timestep t. The denoiser is queried as

vθ x(tn), t c<n , (4)

where c<n is the context representation of blocks B1,...,Bn−1 injected through causal attention: the query tokens come from x(tn) while the key/value tokens include c<n. Under teacher forcing Williams and Zipser [1989], Gao et al. [2024], Hu et al. [2024], Jin et al.

- [2024a], Zhang et al. [2025], training conditions on ground-truth context (c<n = x(0<n)), whereas at

inference c<n consists of model predictions xˆ(0<n); this train–test mismatch causes per-step errors to accumulate along the autoregressive chain—exposure bias Bengio et al. [2015]—manifesting as

progressive over-saturation, motion repetition, and semantic drift, collectively termed distribution drift. Diffusion Forcing Chen et al. [2024], Yin et al. [2025b], Chen et al. [2025b], Gu et al. [2025], Teng et al. [2025b], Song et al. [2025], Po et al. [2025] mitigates this by training with independent per-token noise levels, so the model learns to denoise under heterogeneous noise conditions and gains robustness to partially noisy contexts at inference.

Self-Forcing Anonymous [2025], Yin et al. [2024a,c], Yi et al. [2025] further closes the train–test gap through self-rollout training: during each training iteration, a block is first rolled out with the

student model vθ to obtain xˆ(0n−1), which is then used as context for the next block’s denoising. The training objective is an asymmetric Distribution Matching Distillation (DMD) loss Yin et al.

- [2024b,d], formulated as a reverse KL divergence between the student’s one-step output distribution and the teacher’s multi-step output distribution:

DKL pθ(x0 | xt) ∥ pteacher(x0 | xt) , (5)

LDMD = Et, x

t

where pθ(x0 | xt) denotes the distribution induced by the student’s single Euler step from xt, and pteacher is the distribution obtained by multi-step ODE integration with the teacher model. This reverse KL encourages the student to mode-seek toward the teacher’s high-density regions. In practice the gradient is computed via a learned score difference between student and teacher distributions. While Self-Forcing achieves notable improvements at moderate horizons, it still employs tc = 0 for context (i.e., the predicted clean frame), propagating errors with maximal confidence.

### 3 Method

We now formalise the intuition developed in Sec. 1: the context noise level tc governs a bias– information trade-off, and the optimal choice is t∗c = tj+1—the output noise level of the current denoising step. We first derive this result analytically and then build upon it to design Hierarchical Denoising.

#### 3.1 Context Noise Level and Error Propagation

Error decomposition. Consider block Bn being denoised at step j (from noise level tj to tj+1). Let x(0n−1) denote the ground-truth clean latent of the preceding block and xˆ(0n−1) = x(0n−1) + δ(n−1) the model’s prediction, where δ(n−1) is the accumulated prediction error. In AR diffusion, the context for Bn is derived from xˆ(0n−1) and presented at some noise level tc ∈ [0,1]:

)xˆ(0n−1) + σt

c(nt−c)1 = (1 − σt

η, η ∼ N(0,I). (6)

c

c

Expanding xˆ(0n−1) decomposes the context into three terms:

)x(0n−1) true signal

c(nt−c)1 = (1−σt

)δ(n−1) propagated bias

. (7)

+ (1−σt

+ σt

η

c

c

c

stochastic perturbation

The true-signal and propagated-bias terms share the same coefficient (1−σt

), while the stochastic term carries the complementary coefficient σt

c

. The noise level tc thus controls a bias–information trade-off: raising tc attenuates the bias but simultaneously reduces the useful conditioning signal by the same factor. In particular, prior AR methods Anonymous [2025] use tc = 0, which reduces Eq. 7

c

to c(0)n−1 = x(0n−1) + δ(n−1) and propagates the full prediction error with no attenuation.

Temporal causality. To produce temporally coherent continuations, the context must carry at least as much information as the current block possesses after step j. Under Eq. 1, the signal-to-noise ratio

[Figure 47]

###### Existing AR Rollout (e.g., Self-Forcing) HiAR (Ours)

…

…

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

𝑋 ( )

𝑋 ( )

𝑋 ( )

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

𝑋 ( )

𝑋 ( )

𝑋 ( )

step 𝑖

step 𝑖

…

…

𝑋 ( )

𝑋 ( )

𝑋 ( )

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

𝑋 ( )

𝑋 ( )

𝑋 ( )

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

step 𝑖 + 1

step 𝑖 + 1

… …

… …

… …

… …

step 𝑆

step 𝑆

…

…

𝑋 ( ) 𝑋 ( ) 𝑋 ( )

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

𝑋 ( )

𝑋 ( )

𝑋 ( )

Causal Rollout

DMD

Reverse KL

Student Causal Rollout Reverse KL DMD Teacher

Student Teacher

Trajectory. Sampling

Bi-attn Denoising

Forward KL

- Figure 2: Overview of HiAR. Left: Existing block-first AR (e.g., Self-Forcing) fully denoises each block before generating the next, conditioning every step on predicted clean context and thus amplifying inter-block error propagation. Right: Our hierarchical denoising performs causal generation across all blocks within each denoising step, conditioning on context at the matched noise level to suppress error accumulation. Bottom: Training combines causal self-rollout with a reverseKL (DMD) loss for distillation, and a forward-KL regulariser computed in bidirectional-attention mode via teacher trajectory sampling to preserve motion diversity.

SNR(t) = (1−σt)2/σt2 increases monotonically as t decreases, so after step j the current block at tj+1 contains strictly more information than at tj. Temporal causality therefore requires

SNR(tc) ≥ SNR(tj+1) ⇐⇒ tc ≤ tj+1. (8)

Any tc satisfying this bound provides sufficient information for step j. Since the bias coefficient (1−σt

) decreases monotonically in tc, choosing tc < tj+1 only transmits more prediction error without additional benefit. The optimum is therefore the boundary of the constraint:

c

|t∗c = tj+1,|
|---|

(9)

the noisiest context level that still fulfills temporal causality— attenuating inter-block bias while retaining all information the denoiser needs at step j.

#### 3.2 Hierarchical Denoising

The analysis above motivates a simple but fundamental change to the autoregressive denoising pipeline: instead of fully denoising each block before moving to the next, we perform causal generation across all blocks at each denoising step. We call this Hierarchical Denoising (Fig. 2).

Inference procedure. The complete procedure is summarised in Alg. 1. At each step j, block Bn is denoised with blocks B<n at noise level tj+1 as context—the noisiest level that still preserves temporal causality (Sec. 3.1).

Pipelined parallelism. In Alg. 1, block Bn at step j depends only on B<n at step j and on Bn at step j−1, so blocks at different (n,j) positions that lie on the same anti-diagonal of the N×S grid are mutually independent. We exploit this by assigning each denoising step to a dedicated process and traversing the grid along its N+S−1 anti-diagonals, with inter-stage latents exchanged via asynchronous point-to-point communication. Within each stage, naïvely updating the KV cache for block Bn and denoising block Bn+1 are two separate forward passes, totalling 2N per stage. We observe that under causal attention the two operations can be fused into one forward call by concatenating [c(n), x(tn+1)

] along the frame dimension with per-frame timesteps [tj+1,...,tj+1, tj,...,tj]: the

j

Algorithm 1 Hierarchical Denoising Require: Schedule t1>t2>···>tS≈0; initial noise {x(tn)

}Nn=1 Ensure: Generated blocks {xˆ(0n)}Nn=1

1

- 1: for j = 1,...,S do ▷ denoising steps
- 2: for n = 1,...,N do ▷ causal block sweep with KV cache
- 3: x(tn)

j+1

← x(tn)

j

+ vθ x(tn)

j

,tj | x(t<n)

j+1

(σt

j+1 − σt

j

)

- 4: end for
- 5: Update KV cache with {x(tn)

j+1

}Nn=1

- 6: end for
- 7: return xˆ(0n) ← x(tn)

for n = 1,...,N

S

first segment writes Bn’s context into the KV cache while the second segment denoises Bn+1 attending to the freshly written keys and values. This fusion reduces the cost to N+2 passes per stage (one standalone denoise for the first block, N−1 fused passes, and one trailing cache write), yielding an overall ∼1.8× wall-clock speedup in our 4-step setting.

#### 3.3 Training with Forward-KL Regulation

Although hierarchical denoising already mitigates degradation at test time, a train–test gap remains when the model has been trained under the conventional block-first rollout. We therefore retrain with self-rollout under the hierarchical schedule, optimising the DMD reverse-KL objective (Eq. 5) following Self-Forcing Anonymous [2025]. The overall training pipeline is illustrated in fig. 2 (bottom).

The low-motion shortcut. As training progresses, temporal coherence improves yet motion diversity collapses: the model increasingly produces near-static videos. The root cause is the mode-seeking nature of the reverse-KL objective: DKL(pθ∥pteacher) is minimised when the student concentrates its mass on a single high-density mode, so it can reduce loss by generating low-motion outputs that are inherently easier to denoise and less prone to rollout errors. Hierarchical denoising amplifies this shortcut, because conditioning on contexts at varying noise levels—rather than only clean onesincreases learning difficulty and demands more training steps, giving the mode-seeking objective more iterations to collapse onto the low-motion mode.

Forward-KL regularisation via distillation. To counteract this shortcut, we introduce a complementary loss that penalises mode dropping. We first run the teacher for a large number of ODE steps to obtain a dense denoising trajectory, from which we extract checkpoints {xreft

,...,xreft

} aligned with the student’s S-step schedule. The student is then supervised to match each consecutive pair via a single Euler step:

1

S

ref ti+1−xrefti

, ti) − x

2 . (10)

LFKL = Ei vθ(xreft

σti+1−σti

i

Because the targets xreft are drawn from the teacher’s distribution, optimising Eq. 10 amounts to minimising a forward-KL-direction objective that encourages the student to cover the teacher’s output modes rather than mode-seek, thereby preserving motion diversity.

Decoupling from DMD. To prevent interference between LFKL and the DMD objective, we adopt two design choices:

- 1. Bidirectional-attention mode only. Motion dynamics under bidirectional and causal attention are strongly positively correlated (Sec. 4.4), so regularising the former effectively

constrains the latter. We therefore compute LFKL exclusively in bidirectional-attention mode, leaving the causal self-rollout DMD loss unmodified and minimising gradient interference.

- 2. Early-step restriction. Motion dynamics are governed by low-frequency structures estab-

lished during the earliest denoising steps. We thus apply LFKL only to the first K of S steps, leaving subsequent high-frequency refinement steps unconstrained.

The overall training objective is

L = LDMD + λLFKL, (11)

where λ > 0 balances the two terms. We ablate the choice of K and the attention-mode decoupling strategy in Sec. 4.4.

### 4 Experiments

#### 4.1 Setups

Table 1: Quantitative comparison on 20s generation. Throughput is in frames/s; Latency is in seconds; VBench scores (Total/Quality/Semantic/Dynamic) are on a 0–1 scale; Drift is our proposed drift metric. “–” indicates the model is non-autoregressive and drift is not applicable. Best distilled AR results are bolded.

Model Thru. Lat. Total↑ Quality↑ Semantic↑ Dynamic↑ Drift↓ Bidirectional video diffusion models

LTX-Video HaCohen et al. [2025] 8.98 13.5 0.766 0.789 0.685 0.458 – Wan2.1-1.3B Team [2025] 0.78 103 0.802 0.813 0.766 0.690 –

Autoregressive video diffusion models

NOVA Deng et al. [2025] 0.88 4.1 0.773 0.777 0.757 0.444 – Pyramid Flow Jin et al. [2024b] 6.70 2.5 0.775 0.804 0.670 0.161 – SkyReels-V2-1.3B Chen et al. [2025a] 0.49 112 0.788 0.808 0.707 0.333 – MAGI-1-4.5B Teng et al. [2025a] 0.19 282 0.757 0.785 0.647 0.486 –

Distilled autoregressive video models

CausVid Yin et al. [2025a] 17 0.69 0.764 0.771 0.740 0.621 0.842 Self-Forcing Anonymous [2025] 17 0.69 0.805 0.829 0.708 0.542 0.355 Causal Forcing Zhu et al. [2025] 17 0.69 0.810 0.837 0.701 0.672 0.615 HiAR (Ours) 30 0.30 0.821 0.846 0.723 0.686 0.257

Implementation details. We use the Wan2.1-1.3B backbone Team [2025] as our base model. Following Self-Forcing Anonymous [2025], we fine-tune the model with causal attention masking on 16k ODE solution pairs sampled from the base model. We adopt a 4-step denoising schedule (S = 4) and use Wan2.1-14B as the teacher model for the DMD critic. All methods are implemented in a chunk-wise manner, where each chunk contains 3 latent frames. For the forward-KL regulariser, we sample 20k denoising trajectories (50 ODE steps each) from the Wan2.1-1.3B base model, and restrict LFKL to the first denoising step only (K = 1), with a balancing weight λ = 0.1. The critic model and generator are updated at a 5:1 ratio. We train with a learning rate of 2×10−6 and a total batch size of 64 for 20 k steps on 5-second clips. At inference time, we employ a sliding-window KV cache with a constant attention window of 5s.

Evaluation metrics. We adopt the VBench Huang et al. [2024] protocol, which measures 16 dimensions grouped into a Quality score and a Semantic score, providing a comprehensive assessment of average generation quality. All models are sampled to 20s to evaluate long-video capability. To quantify temporal degradation beyond aggregate scores, we introduce a drift metric suite specifically designed for long-horizon evaluation. Each 20-second video is evenly divided into five temporal segments, and the following per-segment statistics are computed: perceptual quality via MUSIQ Ke et al. [2021] and CLIP-IQA Wang et al. [2023]; temporal coherence via DINOv2 Oquab et al. [2024] consecutive-frame cosine similarity and LPIPS Zhang et al. [2018] consecutive-frame distance; and low-level statistics including HSV saturation mean and Laplacian variance (sharpness). For each metric, we report the slope of a linear fit over the five segments as a measure of drift rate. All per-metric slopes are then normalised and aggregated via a weighted sum into a single Drift Score (lower is better) that summarises overall temporal stability.

Baselines. We compare against recent open-source video generation methods spanning three categories: (i) bidirectional diffusion models—LTX-Video HaCohen et al. [2025] (real-time Video-VAE + spatiotemporal transformer) and Wan2.1-1.3B Team [2025] (the foundation model shared by all distilled methods below); (ii) autoregressive diffusion models—NOVA Deng et al. [2025] (nonquantised temporal AR with spatial diffusion), Pyramid Flow Jin et al. [2024b] (pyramidal flow matching with temporal pyramid), SkyReels-V2 Chen et al. [2025a] (1.3B; diffusion forcing with non-decreasing noise schedules), and MAGI-1 Teng et al. [2025a] (4.5B; block-causal attention); (iii) distilled AR models, all distilling Wan2.1 into a 4-step causal generator—CausVid Yin et al. [2025a] (bidirectional-to-AR DMD distillation), Self-Forcing Anonymous [2025] (self-rollout DMD

training), and Causal Forcing Zhu et al. [2025] (use diffusion forcing as mid-training before selfrollout distillation). All baselines use official checkpoints and are evaluated under identical prompts and generation lengths (5s for bidirectional models).

#### 4.2 Quantitative Results

- Table 1 reports VBench scores, drift, and inference efficiency for all methods.

VBench results. HiAR achieves the highest Total score (0.821) among all methods, surpassing both bidirectional and autoregressive baselines. Notably, it attains the best Quality score (0.846) while maintaining a strong Semantic score (0.723), indicating that hierarchical denoising does not sacrifice semantic fidelity for visual quality. On the Dynamic dimension, HiAR scores 0.686, closely preserving the motion diversity of the bidirectional Wan2.1-1.3B teacher (0.690) and substantially outperforming all other AR methods—including Causal Forcing (0.672) and Self-Forcing (0.542)demonstrating the effectiveness of our forward-KL regulariser in preventing motion collapse.

Temporal stability. On our proposed Drift metric, HiAR achieves 0.257, the lowest among all distilled AR models, indicating minimal quality degradation over the 20s horizon. By contrast, CausVid exhibits the highest drift (0.842), consistent with its visible colour oversaturation at later segments; Self-Forcing (0.355) and Causal Forcing (0.615) show intermediate degradation. HiAR reduces drift by 27.6% relative to Self-Forcing (0.257 vs. 0.355), confirming that hierarchical denoising with matched context noise levels substantially mitigates the compounding inter-block error that drives long-horizon degradation.

Inference efficiency. Owing to pipelined parallelism across hierarchy levels, HiAR achieves 30fps throughput and 0.30s per-chunk latency—a ∼1.8× wall-clock speedup over other distilled AR models (17fps, 0.69s) that share the same Wan2.1-1.3B backbone and 4-step denoising schedule. This speedup comes at no cost to generation quality; in fact, HiAR simultaneously achieves the best VBench scores and the lowest drift.

#### 4.3 Qualitative Results

Fig. 3 presents visual comparisons among all distilled autoregressive models on 20 s generation across six diverse prompts spanning natural scenery (beach, mountain landscape), objects (umbrellas), and human subjects (rock climbing, woman reading, baby portrait).

CausVid exhibits the most severe degradation: frames progressively shift toward neon green and yellow tints, with scene content largely unrecognisable by 20s. Self-Forcing and Causal Forcing alleviate this to some extent, yet still develop visible colour oversaturation and hue drift over time. The degradation is particularly pronounced on human-centric content—facial regions suffer from unnatural colour casts and loss of fine detail (e.g., skin texture, facial features), which are perceptually salient and difficult to mask. By contrast, HiAR maintains stable colour fidelity, sharpness, and structural coherence from the first frame to the last across all content types, with no perceptible drift in either scenery or portrait prompts.

#### 4.4 Ablation Studies

We conduct ablations along two axes: the context noise level tc (Table 2) and the design choices of the forward-KL regulariser (Table 3). All variants are retrained under the same rollout mode used at inference to ensure train–test consistency, unless stated otherwise.

Context noise level. Table 2 compares three context noise configurations. We evaluate overall video quality (Quality, Semantic), temporal smoothness approximated by the VBench motion smoothness score, and long-horizon stability (Drift).

When tc = tj (the input noise level of the current step), the context carries the same noise level as the current block’s input, meaning that block Bn cannot observe the result of denoising step j on block Bn−1—effectively removing intra-step causality. While this yields the lowest drift (0.184), the lack of any one-step-ahead information substantially degrades generation quality (Quality 0.799 vs. 0.846) and produces noticeably unsmooth motion (Smooth. 0.978). At the other extreme, tc = 0 (the standard Self-Forcing setting) fully denoises the context, exposing the model to maximum error propagation and the highest drift (0.355). Our default tc = tj+1 (the output noise level)—where each

###### 0s 10s 20s 0s 10s 20s

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

CausVid

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Self Forcing

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Causal Forcing

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Ours

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

CausVid

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Self Forcing

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Causal Forcing

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Ours

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

CausVid

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

Self Forcing

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Causal Forcing

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Ours

- Figure 3: Qualitative comparison of distilled AR models at 20s. We show temporally sampled frames from six diverse prompts covering natural scenery, objects, and human subjects. HiAR maintains consistent colour and detail throughout, while baselines exhibit progressive degradation.

block conditions on the context that has been denoised through the current step—strikes the optimal balance: it preserves nearly the same temporal smoothness as Self-Forcing (0.988 vs. 0.991) while substantially reducing drift and improving overall quality.

Forward-KL regulation design. Table 3 ablates the attention mode, number of constrained denoising steps K, and the necessity of each component. We focus on motion dynamics (Dynamic), overall quality (Quality, Semantic), and drift.

Attention mode. Applying LFKL in causal mode (“causal + 1 step”) leads to lower dynamics (0.625 vs. 0.686) and reduced quality compared with the bidirectional-attention default. To empirically justify

our design of applying LFKL in bidirectional-attention mode, we track the dynamic scores under both attention modes across training checkpoints (without LFKL). As shown in Fig. 4, both modes exhibit a consistent decline in dynamics over training, and the two scores are strongly positively correlated (Pearson r = 0.968). This confirms that regularising the bidirectional mode serves as an effective and non-intrusive proxy for preserving causal-mode motion diversity. Fig. 5 visualises single-step

- Table 2: Ablation study on context noise level tc. Quality, Semantic, and Smooth are VBench sub-scores; Drift is our proposed drift metric.

Context noise Quality↑ Semantic↑ Smooth.↑ Drift↓ tc = tj (input level) 0.799 0.692 0.978 0.184 tc = tj+1 (output level; default) 0.846 0.723 0.988 0.257 tc = 0 (Self-Forcing) 0.829 0.708 0.991 0.355

- Table 3: Ablation on forward-KL regulariser design. “bi-attn”, “causal” denotes the attention mode used for LFKL; “Kstep” is the number of denoising steps.

Configuration Quality↑ Semantic↑ Dynamic↑ Drift↓

- bi-attn + 1 step (default) 0.846 0.723 0.686 0.257 causal + 1 step 0.828 0.701 0.625 0.271
- bi-attn + 2 steps 0.835 0.708 0.693 0.296 bi-attn + 4 steps 0.813 0.684 0.691 0.306

w/o LFKL 0.839 0.732 0.445 0.218 w/o re-training 0.767 0.559 0.512 0.309 w/o hier. denoising (Self-Forcing) 0.829 0.708 0.542 0.355

denoising outputs under both modes. Under bidirectional attention, all frames exhibit a uniform level of quality and blur, since the full-sequence attention treats every position symmetrically. In contrast, causal denoising produces frames that become progressively sharper along the temporal axis: as preceding frames fix the low-frequency structure, the conditional distribution of later frames concentrates, resulting in higher-frequency details. This asymmetry means that a distillation target derived from bidirectional denoising provides a spatiotemporally uniform supervision signal well suited to regularising global dynamics, whereas directly constraining causal outputs introduces mismatched targets that are tightly coupled with the model’s autoregressive generation pathway, degrading overall quality. Bidirectional-mode regularisation is therefore the preferred configuration.

Number of constrained steps. Increasing K from 1 to 2 or 4 brings marginal gains in dynamics (0.693, 0.691 vs. 0.686) but monotonically degrades both quality and drift. This confirms that motion diversity is primarily governed by the low-frequency structure laid down in the first denoising step; constraining subsequent high-frequency refinement steps provides diminishing returns while

20.0

|[Figure 150]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

Pearson r=0.968 p<10 6

0.65

17.5

CausalDynamicScore

0.60

15.0

TrainingSteps(k)

0.55

12.5

10.0

0.50

7.5

0.45

5.0

0.40

2.5

95% CI

0.35

0.0

0.40 0.45 0.50 0.55 0.60 0.65 0.70

Bidirectional Dynamic Score

- Figure 4: Correlation between bidirectional and causal dynamics during training (w/o LFKL). Each point represents one training checkpoint; colour encodes the training step. A strong positive correlation (Pearson r = 0.968) confirms that the low-motion shortcut affects both attention modes simultaneously and that regularising the bidirectional mode effectively constrains causal-mode dynamics.

0s 1s 2s 5s

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

Bidirectional

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Causal

- Figure 5: Comparison of single-step denoising under bidirectional vs. causal attention. Bidirectional attention produces frames of uniform quality and blur across all positions, while causal attention yields progressively sharper frames as preceding context reduces uncertainty for later positions.

interfering with the model’s denoising capacity. A single constrained step (K = 1) is therefore sufficient and optimal.

Component necessity. Removing LFKL entirely (“w/o LFKL”) yields competitive quality and the lowest drift, but dynamics collapse drastically (0.445), confirming that the model falls into the low-motion shortcut without forward-KL regulation. “w/o re-training” applies hierarchical denoising only at inference without corresponding training, which significantly reduces drift compared with Self-Forcing (0.309 vs. 0.355) yet at a substantial cost to visual quality (Quality 0.767), highlighting the importance of train–test alignment. Finally, removing hierarchical denoising altogether recovers the standard Self-Forcing baseline, which exhibits the highest drift (0.355) and lower dynamics (0.542), validating the contribution of hierarchical denoising to long-horizon stability.

### 5 Conclusion

We presented HiAR, a hierarchical denoising framework that addresses the distribution drift problem in autoregressive long video generation. Our key insight is that a fully clean context is unnecessary and, in fact, harmful: by conditioning each block on context at matched noise level rather than predicted clean frames, hierarchical denoising attenuates inter-block error propagation while preserving temporal causality. This simple reordering—from the conventional block-first pipeline to a step-first paradigm—also enables pipelined parallel inference, achieving ∼1.8× wall-clock speedup in our 4-step setting. To stabilise training, we introduced a forward-KL regulariser in bidirectional-attention mode that counteracts the low-motion shortcut inherent to reverse-KL distillation, preserving motion diversity without interfering with the DMD objective. Experiments on VBench and a dedicated drift metric confirm that HiAR achieves the best overall quality and the lowest temporal degradation among all compared methods on 20-second generation.

### References

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR, 2023.

Anonymous. Self-forcing: Bridging the train-test gap in autoregressive video generation. arXiv preprint, 2025.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. OpenAI Technical Report, 2024.

Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024.

Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. SkyReels-V2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025a.

Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025b.

Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. In ICLR, 2025.

Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, Jun Xiao, and Long Chen. Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375, 2024.

Google. Introducing veo 3, our video generation model with expanded creative controls – including native audio and extended videos. https://deepmind.google/models/veo/, 2025.

Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, and Ofir Bibi. LTX-Video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103, 2025.

Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Size Wu, Wei Li, Xuchen Song, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 2.0: An open-source real-time and streaming interactive world model, 2025. URL https://arxiv.org/abs/2508.

##### 13009.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in neural information processing systems, 35:8633–8646, 2022.

Yicong Hong, Yiqun Mei, Chongjian Ge, Yiran Xu, Yang Zhou, Sai Bi, Yannick Hold-Geoffroy, Mike Roberts, Matthew Fisher, Eli Shechtman, Kalyan Sunkavalli, Feng Liu, Zhengqi Li, and Hao Tan. Relic: Interactive video world model with long-horizon memory, 2025. URL https: //arxiv.org/abs/2512.04040.

Jinyi Hu, Shengding Hu, Yuxuan Song, Yufei Huang, Mingxuan Wang, Hao Zhou, Zhiyuan Liu, Wei-Ying Ma, and Maosong Sun. Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720, 2024.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024a.

Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024b.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: Multi-scale image quality transformer. In ICCV, 2021.

Kling. Kling video 2.6 – kling’s first “native audio” model official launched! https://app.

##### klingai.com/global/release-notes/c605hp1tzd, 2025.

Shanchuan Lin, Ceyuan Yang, Hao He, Jianwen Jiang, Yuxi Ren, Xin Xia, Yang Zhao, Xuefeng Xiao, and Lu Jiang. Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350, 2025.

Yaron Lipman, Ricky T Q Chen, Heli Ben-Hamu, and Maximilian Nickel. Flow matching for generative modeling. In ICLR, 2023.

Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161, 2025.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

Yanzuo Lu, Yuxi Ren, Xin Xia, Shanchuan Lin, Xing Wang, Xuefeng Xiao, Andy J Ma, Xiaohua Xie, and Jian-Huang Lai. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16818–16829, 2025a.

Yunhong Lu, Yanhong Zeng, Haobo Li, Hao Ouyang, Qiuyu Wang, Ka Leong Cheng, Jiapeng Zhu, Hengyuan Cao, Zhipeng Zhang, Xing Zhu, et al. Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678, 2025b.

Xiaofeng Mao, Zhen Li, Chuanhao Li, Xiaojie Xu, Kaining Ying, Tong He, Jiangmiao Pang, Yu Qiao, and Kaipeng Zhang. Yume-1.5: A text-controlled interactive world generation model, 2025. URL https://arxiv.org/abs/2512.22096.

OpenAI. Sora 2 is here. https://openai.com/index/sora-2/, 2025. Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,

Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. TMLR, 2024.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

Ryan Po, Eric Ryan Chan, Changan Chen, and Gordon Wetzstein. Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080, 2025.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Runway. Introducing runway gen-4.5: A new frontier for video generation. https://runwayml.

##### com/research/introducing-runway-gen-4.5, 2025.

Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.

Wenqiang Sun, Haiyu Zhang, Haoyuan Wang, Junta Wu, Zehan Wang, Zhenwei Wang, Yunhong Wang, Jun Zhang, Tengfei Wang, and Chunchao Guo. Worldplay: Towards long-term geometric consistency for real-time interactive world modeling, 2025. URL https://arxiv.org/abs/ 2512.14614.

Junshu Tang, Jiacheng Liu, Jiaqi Li, Longhuang Wu, Haoyu Yang, Penghao Zhao, Siruis Gong, Xiang Yuan, Shuai Shao, Linfeng Zhang, and Qinglin Lu. Hunyuan-gamecraft-2: Instruction-following interactive game world model, 2026. URL https://arxiv.org/abs/2511.23429.

Wan Team. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W Q Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, et al. MAGI-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025a.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025b.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Jianyi Wang, Kelvin C K Chan, and Chen Change Loy. Exploring CLIP for assessing the look and feel of images. In AAAI, 2023.

Ronald J Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989.

Xiaofei Wu, Guozhen Zhang, Zhiyong Xu, Yuan Zhou, Qinglin Lu, and Xuming He. Pack and force your memory: Long-form and consistent video generation, 2025. URL https://arxiv.org/ abs/2510.01784.

Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Deheng Ye, Fangyun Zhou, Jiacheng Lv, Jianqi Ma, Jun Zhang, Junyan Lv, Junyou Li, Minwen Deng, Mingyu Yang, Qiang Fu, Wei Yang, Wenkai Lv, Yangbin Yu, Yewen Wang, Yonghang Guan, Zhihao Hu, Zhongbin Fang, and Zhongqian Sun. Yan: Foundational interactive video generation, 2025. URL https://arxiv.org/abs/2508.08601.

Jung Yi, Wooseok Jang, Paul Hyunbin Cho, Jisu Nam, Heeji Yoon, and Seungryong Kim. Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081, 2025.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Frédo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv preprint arXiv:2405.14867, 2024b.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024c.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation, 2024d. URL https://arxiv.org/abs/2311.18828.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Frédo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025a.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22963–22974, 2025b.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025.

Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness, 2025. URL https://arxiv.org/abs/2503.21755.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024.

Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2025.

