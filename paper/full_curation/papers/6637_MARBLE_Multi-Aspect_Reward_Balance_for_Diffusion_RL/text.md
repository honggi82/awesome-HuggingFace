# arXiv:2605.06507v1[cs.CV]7May2026

## MARBLE: Multi-Aspect Reward Balance for Diffusion RL

Canyu Zhao1∗ Hao Chen1 Yunze Tong1 Yu Qiao2 Jiacheng Li2 Chunhua Shen1,3 1 Zhejiang University 2 HiThink 3 Zhejiang University of Technology

#### Abstract

Reinforcement learning (RL) fine-tuning has become the dominant approach for aligning diffusion models with human preferences. However, assessing images is intrinsically a multi-dimensional task, and multiple evaluation criteria need to be optimized simultaneously. Existing practice deal with multiple rewards by training one specialist model per reward, optimizing a weighted-sum reward R(x)= k wkRk(x), or sequentially fine-tuning with a hand-crafted stage schedule. These approaches either fail to produce a unified model that can be jointly trained on all rewards or necessitates heavy manually tuned sequential training. We find that the failure stems from using a naive weighted-sum reward aggregation. This approach suffers from a sample-level mismatch because most rollouts are specialist samples, highly informative for certain reward dimensions but can be irrelevant for others; consequently, weighted summation dilutes their supervision. To address this issue, we propose MARBLE (Multi-Aspect Reward BaLancE), a gradient-space optimization framework that maintains independent advantage estimators for each reward, computes per-reward policy gradients, and harmonizes them into a single update direction without manually-tuned reward weighting, by solving a Quadratic Programming problem. We further propose an amortized formulation that exploits the affine structure of the loss used in DiffusionNFT, to reduce the per-step cost from K+1 backward passes to near single-reward baseline cost, together with EMA smoothing on the balancing coefficients to stabilize updates against transient single-batch fluctuations. On SD3.5 Medium with five rewards, MARBLE improves all five reward dimensions simultaneously, turns the worst-aligned reward’s gradient cosine from negative under weighted summation in 80% of mini-batches to consistently positive, and runs at 0.97× the training speed of baseline training. Homepage and code repo: HERE.

#### 1 Introduction

Reinforcement learning (RL) fine-tuning has emerged as the dominant paradigm for aligning diffusion model outputs with human preferences, yielding notable improvements in aesthetic quality, text-image alignment, and compositional accuracy [21, 50, 39, 55]. In practice, however, generation quality is inherently multi-dimensional. A high-quality image should simultaneously exhibit aesthetic appeal, faithfulness to the text prompt, and fine-grained correctness such as accurate text rendering and coherent object placement. These aspects are difficult to optimize jointly. Existing methods typically optimize a separate model for each individual reward [21, 50, 39], or sequentially fine-tune a single model on different reward datasets [55]. However, the former does not yield a unified model, while the latter relies on substantial manual effort in designing the training schedule and hyperparameters. For example, DiffusionNFT [55] uses a hand-crafted sequence of stages: 800 iterations on reward 1, followed by 300 iterations on reward 2; 200 iterations on reward 1; 200 iterations on reward 2,

∗Work done during an internship at HiThink.

Preprint.

###### One Model per Reward

###### Ours: 1 Model on All Rewards

###### Sequential Multi-Reward Training

Trained Sequentially

Trained Simultaneously

Trained Separately

###### Stage1 Stage2 Stage3 ... StageT

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

PickScore / OCR / ...

One Model

Model A Model B Model C

###### Simultaneous Multi-Reward Training

###### One Model

[Figure 10]

[Figure 11]

[Figure 12]

- • Less Manual Effort
- • One Model for All Objectives

- • Extensive Hyperparameter Tuning for Each Stage
- • Requires Handcrafted Stage Schedule

- • Need to Maintain Multiple Models
- • Hard to Generalize across Rewards

- Figure 1: Comparison of multi-reward training paradigms. Left: Training one model per reward requires maintaining multiple models and cannot generalize across reward dimensions. Middle: Sequential multi-reward training produces a single model but demands extensive hyperparameter tuning and handcrafted stage schedules. Right: MARBLE trains a single model on all rewards simultaneously with minimal manual effort.

and finally 100 iterations on reward 3, which requires substantial manual tuning and suffer from forgetting previously acquired rewards.

Therefore, the central challenge lies in developing a principled approach to conveniently and effectively optimize a single model across multiple reward objectives while eliminating heuristic manual tuning. A natural approach to multi-reward optimization is to combine all reward signals into a single scalar objective, typically via a weighted sum R(x) = k wkRk(x). However, in practice, directly optimizing a diffusion model with this naively aggregated reward often results in performance degradation rather than improvement. We trace the failure of scalar aggregation to a sample-level mismatch that we call the specialist sample phenomenon (Figure 2). Many rollouts are informative for only a part of reward dimensions and uninformative or even inapplicable for the rest. For example, an image of a cat carries no signal for OCR-related rewards, and a generation with strong text rendering may be only average aesthetically. Under R(x)= k wkRk(x), the value of such a sample is diluted by the unrelated dimensions, and the resulting advantage no longer reflects the dimension on which the sample is genuinely useful. We further empirically confirm this dilution at the gradient level (Section 3.2): the weighted-sum update direction is anti-aligned with single reward gradient, meaning the update actively pushes against some reward most of the time.

To address this problem, we propose MARBLE, a gradient-space reward balance framework that preserves reward-specific supervision throughout optimization. Rather than collapsing rewards into a scalar, MARBLE maintains an independent advantage estimator per reward so that each sample is credited precisely on the dimensions for which it is informative, computes per-reward policy gradients, normalizes them to remove scale disparities, and harmonizes them into a single update direction. To ensure scalability during training, we develop an amortized formulation that leverages the affine structure of the DiffusionNFT loss, thereby reducing the per-step computational cost to nearly that of a single-reward baseline. Also, we apply EMA smoothing on the balancing coefficients so that certain rewards are not transiently silenced when a single mini-batch happens to carry weak signal for them. In summary, our contributions are:

- • We characterize the specialist sample problem in multi-reward diffusion RL. Across rollouts on SD3.5 Medium, weighted-sum aggregation produces an update direction that is anti-aligned with at least one reward’s gradient in 80% of mini-batches, formally quantifying why scalar reward aggregation fails when reward signals are sample-sparse.
- • We propose MARBLE, a gradient-space reward balancing framework. MARBLE combines (i) per-reward advantage decomposition with normalize-and-rescale gradient harmonization, (ii) an amortized variant that reduces multi-reward training cost to near a single-reward baseline by exploiting the affine structure of the DiffusionNFT loss, and (iii) EMA coefficient smoothing that stabilizes amortized balancing weights against transient single-batch fluctuations.
- • MARBLE simultaneously improves all rewards with a single model. To the best of our knowledge, we are the first to address reward balancing in multi-reward diffusion RL. We

ocr source

geneval source

| | | | | | | | | |[Figure 13]| |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

| |[Figure 14]| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

ocr

geneval

2

[Figure 15]

z-score

pickscore

pickscore

0

hpsv2

hpsv2

2

clipscore

clipscore

0 20 40 60 80 100

0 50 100 150 200

samples (n=119)

samples (n=238)

- Figure 2: Sample-level specialist structure. Each column denotes one sample, and each row shows

its per-reward z-score advantage Ak(x). High advantages are concentrated on source-specific rewards such as OCR or GenEval. Few samples achieve positive rewards across all dimensions.

believe MARBLE provides a useful foundation for future work on scalable multi-objective alignment of generative models.

#### 2 Related Work

##### 2.1 Reinforcement Learning for Diffusion Models

Diffusion models [11, 37, 36] have become the dominant paradigm for high-fidelity image generation. Latent diffusion [30, 27] moved the generation process into a compressed latent space, enabling efficient high-resolution synthesis, while subsequent scaling efforts [5] further improved generation quality by combining rectified flow formulations [23, 19] with transformer-based architectures [26]. Diffusion models have since been extended far beyond text-to-image generation to a wide range of generative tasks, including image customization [51, 38, 24, 48], image editing [2, 16, 43, 41], video editing [14, 52], image understanding [54, 7] and even long-form video and movie generation [53, 13, 18, 45].

Reinforcement Learning [32, 28] has emerged as a primary approach for aligning models with human preferences. In diffusion RL, a reward model evaluates each generated sample, and the diffusion policy is optimized to maximize expected reward while remaining close to a pre-trained reference model [1, 6, 40]. Early work mainly relied on policy-gradient-based methods [1, 6]. More recently, inspired by the success of GRPO [34] in large language models, a growing body of work has adapted similar ideas to diffusion models [21, 39, 47, 9, 50, 17], achieving stronger empirical performance. Recent work such as DiffusionNFT [55] has further improved training efficiency. Despite these advances, existing diffusion RL methods largely optimize a single scalar reward. When multiple reward signals are available, practitioners typically either train separate models for different rewards, fine-tune sequentially on different datasets, or combine several rewards through a weighted sum. None of these strategies provides a principled way to jointly optimize multiple quality dimensions within a single training run without manual reward weighting.

##### 2.2 Multi-Task Learning

Multi-task learning [3, 4, 33, 49, 20, 25, 22] trains a shared model over multiple objectives and faces a closely related challenge: inter-task gradient interference can cause a single update to improve some objectives while harming others. To address this issue, prior work has developed a range of gradient-level optimization strategies, including finding the minimum-norm point in the convex hull of per-task gradients [4, 33], projecting out destructive gradient components [49], maximizing worstcase per-task improvement [20], and formulating gradient balancing as a game-theoretic bargaining problem [25]. These methods share a common principle: resolving interactions among objectives in gradient space rather than loss space. They have proven effective in supervised multi-task settings, particularly for jointly learning multiple vision tasks.

RL [58, 56, 34] and Multi-reward alignment [57, 29, 35] has also received growing attention in Large Language Model. However, to the best of our knowledge, there has been little attempt to address the corresponding problem in diffusion RL. MARBLE bridges this gap by adapting gradient harmonization to the diffusion RL setting, with per-reward advantage decomposition and scale-aware gradient balancing tailored to the diffusion training objective.

2 Per-Reward Backpropogation

1 Compute Different Reward

###### Find a Common Descent Direction

[Figure 16]

[Figure 17]

[Figure 18]

3

2

2

Shared Model

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

- Reward 1

[Figure 28]

[Figure 29]

- Reward 2

[Figure 30]

- Reward 3

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Taking 3 Rewards as an Example

[Figure 37]

[Figure 38]

3

[Figure 39]

[Figure 40]

1

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Prompt Batch 1

3

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Update

[Figure 50]

Compute Gradients Separately for Each Reward

Shared Model

= 

Reward K

[Figure 51]

[Figure 52]

[Figure 53]

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

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

- Figure 3: Overview of MARBLE. Given a prompt batch, the shared model πθ generates images that are scored by K reward models independently. Per-reward policy gradients gk = ∇θLk are computed via separate backpropagation passes. The gradient harmonization finds a common descent direction d that balances all reward objectives and the shared model is updated accordingly.

#### 3 Method

##### 3.1 Preliminaries: DiffusionNFT

Let πθ denote a diffusion model parameterized by θ, and let πref denote the frozen pre-trained reference policy. Given a single reward function R : X → R, diffusion RL optimizes

[R(x)] − βKL · DKL(πθ∥πref), (1)

Ex∼π

max

θ

θ

where βKL controls the regularization strength. We build on DiffusionNFT [55], which implements Equation (1) through a noise-free training (NFT) loss. For a generated sample x with advantage A(x), the NFT loss interpolates between a positive term that moves the model toward better predictions and a negative term that pushes it away:

ℓ(θ;x,t) = r · L+(θ) + (1 − r) · L−(θ), (2) where r = clamp 12 + 2AA(x)

, 0, 1 maps the advantage to an interpolation coefficient. Here

max

L+(θ) = ∥vθ+ − v∥2 and L−(θ) = ∥vθ− − v∥2 are velocity prediction losses, with vθ+ = (1 − β)vold + βvθ and vθ− = (1 + β)vold − βvθ constructed from the current policy vθ and the reference policy vold, and v denotes the ground-truth velocity target. A key structural property is that L+ and L− depend only on θ and the current sample, and are therefore independent of the advantage value. The advantage affects the loss only through the affine mapping to r.

When multiple rewards {Rk}Kk=1 are available, the standard approach first aggregates them into a scalar reward R(x) = k wkRk(x), then derives a single advantage A(x) and applies Equation (2) with one interpolation coefficient r. As we show next, this scalarization obscures which reward dimensions each sample is actually informative for, leading to poorly aligned updates in multi-reward training.

##### 3.2 Why Scalar Reward Aggregation Fails

The Introduction identifies specialist samples as the sample-level reason that scalar reward aggregation is unreliable; at the gradient level, the weighted-sum update has negative worst-reward alignment in 80% of the measured mini-batches, whereas MARBLE keeps the worst-reward alignment positive in all measured mini-batches (Appendix C.2). This gradient-level diagnostic motivates the harmonization procedure introduced next.

##### 3.3 Multi-Reward Gradient Harmonization

Per-reward advantage decomposition. To preserve reward-specific supervision, MARBLE decomposes the training signal along reward dimensions. Following DiffusionNFT, for each reward Rk, we

maintain an independent advantage estimator that normalizes Rk within prompt groups:

Rk(x) − µk(prompt) σk(prompt) + ε

, (3)

Ak(x) =

where µk and σk are the running mean and standard deviation of Rk for the same text prompt. Each Ak yields a separate interpolation coefficient rk ∈ [0,1], which defines a reward-specific NFT loss ℓk through Equation (2). A backward pass through ℓk produces the corresponding policy gradient

1 NT

gk = ∇θ

N

T

ℓk(θ;xi,t). (4)

t=1

i=1

All K gradients are computed on the same sampled batch; only the advantage signal differs across rewards. This decomposition allows each sample to be credited precisely on the dimensions for which it is informative, instead of forcing all information through a single aggregated advantage.

Gradient normalization and harmonization. Different reward models can induce gradients at drastically different scales. To remove this scale disparity from the harmonization step, MARBLE first normalizes each gradient:

gˆk = gk/∥gk∥. (5)

Given the normalized gradients {gˆk}Kk=1, MARBLE computes a unified update direction by solving a convex quadratic program, as previously shown in multi-task learning [4, 33]:

α∗ = arg min

α∈∆K

K

αkgˆk

k=1

2

, (6)

where ∆K = {α ∈ RK≥0 : k αk = 1} is the probability simplex. The solution gives a descent direction that improves all rewards as shown in [4]. The resulting direction d∗ = Kk=1 αk∗gˆk is the minimum-norm point in the convex hull of the normalized gradients and provides a balanced compromise across reward dimensions. When rewards are already aligned, the solution concentrates on their shared direction; when rewards emphasize different aspects, the solver adaptively reweights them according to the current batch.

Rescaling and KL-decoupled update. Because d∗ is computed from unit-normalized gradients, its magnitude no longer matches the scale expected by the optimizer or the KL schedule. We therefore restore the natural update scale by multiplying d∗ by the mean norm of the original gradients:

1 K

dfinal = d∗ · n,¯ n¯ =

K

∥gk∥. (7)

k=1

This normalize-then-rescale procedure separates directional balancing from step-size calibration. The final parameter update combines the rescaled reward gradient with KL regularization as a separate term:

θ ← θ − η dfinal + βKL · ∇θDKL(πθ∥πref) . (8)

We treat KL regularization outside the harmonization solve because it plays a different role from reward optimization: reward gradients determine which aspects to improve, while the KL term controls how far the policy is allowed to deviate from the reference model.

##### 3.4 Amortized Gradient Harmonization

The full harmonization procedure requires K+1 backward passes per iteration (K reward-specific passes plus one KL pass), which becomes expensive as the number of rewards grows. Moreover, solving for α∗ at every step introduces additional variance, since the harmonization weights are estimated from a single mini-batch and may fluctuate considerably across iterations. We observe that this instability can lead to undesirable visual artifacts at later training stages, even when average reward scores continue to improve. This motivates an amortized variant that reduces both computational overhead and short-term weight fluctuation.

Scalarization equivalence. Recall from Equation (2) that the NFT loss depends on the advantage only through the affine mapping r = A/(2Amax) + 1/2, while L+(θ) and L−(θ) are independent of A. This yields the following exact equivalence.

Proposition 1. Let α ∈ ∆K and let A1,...,AK be per-reward advantages with |Ak| < Amax for all k and k αkAk < Amax. Define the combined advantage A¯ = Kk=1 αkAk. Then

∇θℓ(θ;A¯) =

K

αk∇θℓk(θ;Ak). (9)

k=1

Proof. Since each advantage Ak is a fixed scalar independent of θ,

k

∇θℓk = rk∇θL+ + (1 − rk)∇θL−,

αk∇θℓk =

k

αkrk ∇θL+ + 1 −

k

αkrk ∇θL−,

(10)

where we used k αk = 1. Because rk = A

2Amax + 12,

k

k

αkrk =

k

αk

- 1

- 2

Ak 2Amax

+

A¯ 2Amax

=

+

- 1

- 2

= r.¯ (11)

Substituting r¯ recovers ∇θℓ(θ;A¯). This shows that, when the clamp is inactive, the convex combination of the per-reward NFT gradients can be recovered exactly by a single backward pass using the combined advantage A¯. The equivalence relies on two properties: (i) the NFT loss depends on the advantage only through the affine map r = A/(2Amax) + 1/2, and (ii) the simplex constraint k αk = 1 preserves the constant offset under convex combination. The clamp in Equation 2 introduces a bounded deviation only when |Ak| ≥ Amax. Following DiffusionNFT [55], we set Amax = 5 during training, which serves as a loose safety bound. Empirically, we never observed the clamp being activated during training.

Amortized procedure. Proposition 1 enables an efficient application of fixed reward-balancing coefficients through a single NFT backward pass. We emphasize that gradient normalization is used only when estimating the coefficients α∗: solving Equation (6) with normalized gradients removes reward-dependent scale disparities and makes α∗ reflect the directional conflict among reward objectives. In contrast, the amortized update applies the cached coefficients in the advantage space, because the exact single-backward equivalence holds for convex combinations of the NFT losses, or equivalently of the unnormalized per-reward gradients. Recovering a normalized-gradient combination at every amortized step would require per-reward gradient norms and thus defeat the purpose of amortization.

Therefore, every N steps, we run the full harmonization procedure to refresh α∗ from normalized gradient. During the intervening N−1 steps, we form A¯ = k αk∗Ak using the cached coefficients and perform only one reward backward pass. This coefficient-amortized approximation retains the scale-invariant reward-balancing information estimated by full harmonization, while preserving the natural gradient scale of the current NFT loss and reducing the average per-step cost from (K+1)× to (K + N)/N times that of a single-reward baseline.

##### 3.5 Coefficient Smoothing for Stable Amortization

While amortized harmonization reduces the computational cost of training, it also makes the optimization more sensitive to short-term fluctuations in the estimated balancing coefficients. In particular, we observe that some reward dimensions may receive little or no useful signal from a rollout batch, especially during the early stages of training. This often happens for specialist rewards that require precise compositional or spatial correctness: when none of the generated samples satisfies the corresponding constraint, the estimated gradient can become uninformative, and the harmonization solver may assign a near-zero coefficient to that reward. Under amortization, such a transient zero coefficient is then reused for the following N−1 steps, effectively suppressing that reward throughout the entire amortization window. This can slow down training and reduce final performance.

- Table 1: Main results. Comparison of MARBLE with pre-trained diffusion models and RL finetuning methods. MARBLE jointly optimizes all in-domain rewards in a single run. †Sequential multi-stage training with hand-crafted curriculum. ‡Simultaneous five-reward training with weightedsum aggregation. Gray denotes in-domain reward used during training. Bold denotes best and underline denotes second best. Composite is the per-row mean of column-wise z-scores (each metric standardized to zero mean and unit variance across the rows of this table); higher is better. Evaluated with the DiffusionNFT official code.

Rule-Based Model-Based

Model

Composite ↑ GenEval OCR PickScore CLIPScore HPSv2.1 Aesthetic ImgRwd UniRwd

SD-XL 0.55 0.14 22.42 0.287 0.280 5.60 0.76 2.93 -0.455 SD3.5-L 0.71 0.68 22.91 0.289 0.288 5.50 0.96 3.25 +0.116 FLUX.1-Dev 0.66 0.59 22.84 0.295 0.274 5.71 0.96 3.27 +0.104

SD3.5-M (w/o CFG) 0.24 0.12 20.51 0.237 0.204 5.13 -0.58 2.02 -2.319 + CFG 0.63 0.59 22.34 0.285 0.279 5.36 0.85 3.03 -0.255

+ FlowGRPO 0.95 0.66 22.51 0.293 0.274 5.32 1.06 3.18 +0.120 0.66 0.92 22.41 0.290 0.280 5.32 0.95 3.15 +0.013 0.54 0.68 23.50 0.280 0.316 5.90 1.29 3.37 +0.362

+ DiffusionNFT† 0.94 0.91 23.80 0.293 0.331 6.01 1.49 3.49 +1.015 + DiffusionNFT‡ 0.92 0.91 21.53 0.267 0.300 6.15 1.16 3.04 +0.184 + MARBLE 0.94 0.96 22.83 0.286 0.355 6.59 1.53 3.52 +1.116

To improve the stability of amortized harmonization, we apply exponential moving average (EMA) smoothing to the balancing coefficients. Let αt∗ denote the coefficients obtained from the full harmonization step at iteration t. Instead of directly using αt∗ for the subsequent amortized updates, we maintain a smoothed coefficient vector α¯t:

α¯t = ρα¯t−1 + (1 − ρ)αt∗, (12)

where ρ is the EMA decay. Since both α¯t−1 and αt∗ lie on the probability simplex, their convex combination also remains a valid simplex vector. We then use α¯t to construct the combined advantage

during amortized updates: A¯ = Kk=1 α¯t,kAk. This smoothing mechanism prevents occasional rollout failures from completely removing a reward signal over an amortization window, while still

allowing the coefficients to adapt to the gradient geometry estimated by the full harmonization step. In all experiments, we set the EMA decay to ρ = 0.7. Empirically, coefficient smoothing improves both training efficiency and effectiveness.

#### 4 Experiments

##### 4.1 Experimental Setup

We build on Stable Diffusion 3.5 Medium [5] and fine-tune LoRA adapters [12] with rank 32 and alpha 64 using the NFT loss in Equation 2. Unless otherwise specified, we use AdamW with a constant learning rate of 3 × 10−4. Our training objective jointly optimizes five rewards: three general-purpose rewards, PickScore [15], HPSv2 [44], and CLIPScore [10], and two specialist rewards, OCR accuracy and GenEval [8]. To assess transfer beyond the optimized rewards, we additionally report Aesthetic Score [31], ImageReward [46], and UniReward [42], none of which are used during training. The model is trained with 16 NVIDIA H200 GPUs. We compare against single-reward FlowGRPO specialists [21], each optimized for one reward, and two multi-reward DiffusionNFT variants [55]: sequential†, which follows a manually scheduled multi-stage training procedure, and simultaneous‡, which directly scalarizes all rewards. All methods are evaluated under the same framework for fair comparison.

##### 4.2 Main Results

Performance. Table 1 reports the main quantitative results. Single-reward FlowGRPO specialists excel on their target objective but transfer poorly to others, requiring a separate model per reward and offering limited cross-objective coverage. In contrast, MARBLE improves all five training rewards within a single model. Qualitative comparisons in Figure 4 further show that MARBLE simultaneously satisfies diverse reward dimensions, while the weighted-sum baseline fails to do so.

Medieval castle gate with a banner reading "Royal Wedding Tomorrow"

A wedding ring engraved with "Always Forever"

A pink tv remote and a blue airplane

Gentleman frog in a top hat

Face profile of a man

A book above a laptop

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

MARBLE DiffusionNFT

Weighted-sum DiffusionNFT

Sequential

Figure 4: Visualizations of qualitative comparisons between MARBLE and Baselines.

- Table 2: Training efficiency comparison on 8×H200. We report relative training speed and GPU memory, both normalized by the weighted-sum baseline.

Method Relative speed GPU memory Weighted Sum (K=5, DiffusionNFT Baseline) 1.00× 59G (1.00×) MARBLE w/ amortization (K=5, N=10) 0.97× 67G (1.14×) MARBLE w/o amortization (K=5) 0.56× 67G (1.14×)

Among multi-reward baselines, DiffusionNFT† (sequential) reaches comparable quality, but at considerable practical cost: it requires a manually scheduled multi-stage curriculum where practitioners must decide the ordering of rewards, the number of steps per stage, and the dataset assigned to each stage, all of which are sensitive to a lot of hyperparameters. More importantly, after introducing a new reward, sequential training often needs to revisit previously seen rewards to mitigate forgetting, so the schedule grows with the number of rewards rather than scaling automatically. The alternative DiffusionNFT‡ (simultaneous) avoids this manual scheduling by directly scalarizing all rewards into a single objective, but as a result performs substantially worse on the specialist objectives. MARBLE matches or exceeds both baselines from a single joint training run, with no per-stage hyperparameters and no explicit replay schedule.

DiffusionNFT† does score moderately higher on PickScore and slightly higher on CLIPScore, but MARBLE matches or surpasses it on every other reward and ranks first on the four held-out quality metrics (HPSv2.1, Aesthetic, ImageReward, UniReward). To summarize across all eight metrics, we report a Composite score in the last column of Table 1, computed as the mean of column-wise zscores so each metric contributes equally regardless of scale. MARBLE attains the highest Composite, showing that the small concession on PickScore/CLIPScore is more than offset by gains elsewhere. Appendix C.7 further demonstrates that MARBLE’s results are preferred.

Training efficiency. We further show the training efficiency comparison in Table 2. All results are measured on 8×H200 and normalized by the weighted-sum baseline. Full per-reward harmonization introduces noticeable overhead, reducing the relative training speed to 0.56× due to the need for multiple reward-specific backward passes. In contrast, the amortized variant substantially reduces this overhead and achieves 0.97× relative speed, which is close to the weighted-sum baseline. Both MARBLE variants require only a modest increase in GPU memory, from 59G to 67G per GPU, corresponding to 1.14× relative memory. These results show that amortization makes gradient-space reward balancing practical at nearly the same training speed as scalarized multi-reward training.

- Table 3: Ablation study. Each row removes or replaces one component of MARBLE. All variants use the same 5-reward setup and training budget.

Rule-Based Model-Based

Variant

GenEval OCR PickScore CLIPScore HPSv2.1 Aesthetic ImgRwd UniRwd MARBLE (full, ρ = 0.7 ) 0.93 0.96 22.62 0.283 0.355 6.59 1.52 3.45 w/o gradient normalization FAIL Fixed α = 0.2 0.86 0.89 22.64 0.272 0.346 6.55 1.45 3.42 Solve α every step 0.92 0.92 21.32 0.267 0.301 5.89 1.17 3.04

##### 4.3 Ablation Studies and Analysis

Table 3 ablates the main design choices that determine the update direction in MARBLE: replacing adaptive coefficients with fixed uniform coefficients (αk = 0.2), removing gradient normalization before solving α, and solving α at every step instead of using the amortized update. Due to space constraints, we report the headline ablations in the main text and defer the supporting analyses to Appendix C, including training dynamics and coefficient adaptation (Appendix C.3), amortizationinterval sensitivity (Appendix C.4), EMA-decay sensitivity (Appendix C.5), and alternative heuristic balancing strategies (Appendix C.6).

Gradient normalization before solving α. The harmonization coefficients α are computed from the per-reward gradients by solving the QP in Equation 6. Without gradient normalization, this optimization becomes highly sensitive to the raw magnitudes of different reward gradients. The resulting coefficients tend to be dominated by scale differences rather than the directional relationships among rewards. In practice, we observe that this often produces degenerate or numerically unstable coefficients, leading to failed optimization.

Equal α weighting. We examine a simple variant that uses fixed uniform coefficients, i.e., αk = 0.2 for all five rewards. This setting leads to imbalanced convergence across different rewards. We observe that general-purpose rewards related to overall visual aesthetics improve relatively quickly, whereas more challenging specialist objectives, such as object attributes and spatial relations, remain under-optimized. In contrast, MARBLE dynamically adjusts the coefficients during training, allocating more optimization emphasis to tasks that are currently harder to improve. This adaptive allocation enables more balanced convergence across both general and specialist rewards. We further find that the best final performance is obtained by using dynamic coefficients during most of training, followed by a short uniform-coefficient stage near the end. Intuitively, the dynamic stage helps the model allocate capacity to difficult reward dimensions, while the final uniform stage encourages all rewards to be jointly consolidated under an equal weighting. This strategy achieves the strongest overall balance.

Coefficient amortization. We also evaluate a variant that solves for α at every training step without amortization. Although this provides a fresh estimate of the gradient geometry at each iteration, it substantially increases training cost due to repeated per-reward backward passes. Moreover, the coefficients estimated from a single mini-batch can fluctuate considerably across iterations, injecting high-frequency variation into the update direction and negatively affecting training stability.

#### 5 Conclusion

We propose MARBLE, the first multi-reward balancing method for diffusion model RL fine-tuning. MARBLE preserves reward-specific supervision through per-reward advantage decomposition and gradient harmonization, avoiding the specialist-sample dilution that limits weighted-sum aggregation, and its amortized formulation keeps training cost close to the baseline. One limitation of our current study is that we validate MARBLE primarily on image generation. Extending the framework to video diffusion and generative world models remains an important direction, as these settings involve richer and more heterogeneous quality dimensions, such as temporal consistency, motion realism, and physical plausibility, making reward balancing even more critical. Another promising direction is scaling MARBLE to larger reward sets, where both optimization and efficiency become tighter challenges. We believe that MARBLE provides an important step toward scalable multi-objective alignment for future generative models.

#### References

- [1] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.
- [2] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.
- [3] Kalyanmoy Deb. Multi-objective optimisation using evolutionary algorithms: an introduction. In Multiobjective evolutionary optimisation for product design and manufacturing, pages 3–34. Springer, 2011.
- [4] Jean-Antoine Désidéri. Multiple-gradient descent algorithm (mgda) for multiobjective optimization. Comptes Rendus. Mathématique, 350(5-6):313–318, 2012.
- [5] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [6] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Dpok: Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36:79858–79885, 2023.
- [7] Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, and Radu Soricut. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.
- [8] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [9] Xiaoxuan He, Siming Fu, Yuke Zhao, Wanli Li, Jian Yang, Dacheng Yin, Fengyun Rao, and Bo Zhang. Tempflow-grpo: When timing matters for grpo in flow models. arXiv preprint arXiv:2508.04324, 2025.
- [10] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [12] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.
- [13] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.
- [14] Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17191–17202, 2025.
- [15] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-apic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.
- [16] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.
- [17] Junzhe Li, Yutao Cui, Tao Huang, Yinping Ma, Chun Fan, Yiming Cheng, Miles Yang, Zhao Zhong, and Liefeng Bo. Mixgrpo: Unlocking flow-based grpo efficiency with mixed ode-sde. arXiv preprint arXiv:2507.21802, 2025.
- [18] Wuyang Li, Wentao Pan, Po-Chien Luan, Yang Gao, and Alexandre Alahi. Stable video infinity: Infinitelength video generation with error recycling. arXiv preprint arXiv:2510.09212, 2025.

- [19] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [20] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-averse gradient descent for multi-task learning. Advances in neural information processing systems, 34:18878–18890, 2021.
- [21] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.
- [22] Suyun Liu and Luis Nunes Vicente. The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning. Annals of Operations Research, 339(3):1119–1148, 2024.
- [23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [24] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2iadapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 4296–4304, 2024.
- [25] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task learning as a bargaining game. arXiv preprint arXiv:2202.01017, 2022.
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [27] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.
- [28] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.
- [29] Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. Advances in Neural Information Processing Systems, 36:71095–71134, 2023.
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [31] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.
- [32] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [33] Ozan Sener and Vladlen Koltun. Multi-task learning as multi-objective optimization. Advances in neural information processing systems, 31, 2018.
- [34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [35] Ruizhe Shi, Yifang Chen, Yushi Hu, Alisa Liu, Hannaneh Hajishirzi, Noah A Smith, and Simon S Du. Decoding-time language model alignment with multiple objectives. Advances in Neural Information Processing Systems, 37:48875–48920, 2024.
- [36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint

- arXiv:2010.02502, 2020.

[37] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint

- arXiv:2011.13456, 2020.

- [38] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14940–14950, 2025.
- [39] Yunze Tong, Mushui Liu, Canyu Zhao, Wanggui He, Shiyi Zhang, Hongwei Zhang, Peng Zhang, Jinlong Liu, Ju Huang, Jiamang Wang, et al. Alleviating sparse rewards by modeling step-wise and long-term sampling effects in flow-based grpo. arXiv preprint arXiv:2602.06422, 2026.
- [40] Yunze Tong, Didi Zhu, Zijing Hu, Jinluan Yang, and Ziyu Zhao. Noise projection: Closing the promptagnostic gap behind text-to-image misalignment in diffusion models. arXiv preprint arXiv:2510.14526, 2025.
- [41] Jiyuan Wang, Chunyu Lin, Lei Sun, Zhi Cao, Yuyang Yin, Lang Nie, Zhenlong Yuan, Xiangxiang Chu, Yunchao Wei, Kang Liao, et al. Geometry-guided reinforcement learning for multi-view consistent 3d scene editing. arXiv preprint arXiv:2603.03143, 2026.
- [42] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025.
- [43] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.
- [44] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.
- [45] Junfei Xiao, Ceyuan Yang, Lvmin Zhang, Shengqu Cai, Yang Zhao, Yuwei Guo, Gordon Wetzstein, Maneesh Agrawala, Alan Yuille, and Lu Jiang. Captain cinema: Towards short movie generation. In The Fourteenth International Conference on Learning Representations, 2025.
- [46] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [47] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.
- [48] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [49] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in neural information processing systems, 33:5824–5836, 2020.
- [50] Liyu Zhang, Kehan Li, Tingrui Han, Tao Zhao, Yuxuan Sheng, Shibo He, and Chao Li. Op-grpo: Efficient off-policy grpo for flow-matching models. arXiv preprint arXiv:2604.04142, 2026.
- [51] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.
- [52] Canyu Zhao, Xiaoman Li, Tianjian Feng, Zhiyue Zhao, Hao Chen, and Chunhua Shen. Tinker: Diffusion’s gift to 3d–multi-view consistent editing from sparse inputs without per-scene optimization. arXiv preprint arXiv:2508.14811, 2025.
- [53] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.
- [54] Canyu Zhao, Yanlong Sun, Mingyu Liu, Huanyi Zheng, Muzhi Zhu, Zhiyue Zhao, Hao Chen, Tong He, and Chunhua Shen. Diception: A generalist diffusion model for visual perceptual tasks. arXiv preprint arXiv:2502.17157, 2025.
- [55] Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.
- [56] Hao Zhong, Muzhi Zhu, Zongze Du, Zheng Huang, Canyu Zhao, Mingyu Liu, Wen Wang, Hao Chen, and Chunhua Shen. Omni-r1: Reinforcement learning for omnimodal reasoning via two-system collaboration. arXiv preprint arXiv:2505.20256, 2025.

- [57] Zhanhui Zhou, Jie Liu, Jing Shao, Xiangyu Yue, Chao Yang, Wanli Ouyang, and Yu Qiao. Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization. In Findings of the Association for Computational Linguistics: ACL 2024, pages 10586–10613, 2024.
- [58] Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, et al. Active-o3: Empowering multimodal large language models with active perception via grpo. arXiv preprint arXiv:2505.21457, 2025.

#### A Appendix Overview

This appendix provides supporting material for MARBLE, including qualitative examples, paper-level takeaways, extended ablations, implementation details, and future directions. The contents are organized as follows:

- • Appendix B: Additional Qualitative Comparisons. We provide additional qualitative examples illustrating how a single MARBLE model handles text rendering, attribute and spatial understanding, and counting while maintaining coherent visual quality.
- • Appendix C: Additional Ablations and Analyses. We collect the main takeaways and extended empirical analyses behind MARBLE:

- – Appendix C.1: Key Insights and Takeaways. A paper-level summary of the main MARBLE insights, including why scalar reward aggregation fails, how to interpret the learned coefficients, and which practical defaults are important when using MARBLE.
- – Appendix C.2: Update-Direction Harmony Diagnostics. We provide the per-batch harmony visualization and aggregate statistics comparing weighted-sum and harmonized update directions.
- – Appendix C.3: Training Dynamics and Coefficient Adaptation. We provide training curves, coefficient trajectories, and a closer analysis of how the learned coefficients relate to optimization difficulty.
- – Appendix C.4: Amortization Interval. We analyze different coefficient refresh intervals and explain why N=10 is used as the main setting.
- – Appendix C.5: EMA Decay for Coefficient Smoothing. We discuss how the EMA decay controls the stability-adaptivity trade-off in coefficient smoothing.
- – Appendix C.6: Alternative Heuristic Strategies. We compare MARBLE with heuristic reward-balancing strategies, including fixed uniform coefficients, reward grouping, and specialist reward up-weighting.
- – Appendix C.7: Human Preference Evaluation and Metric Correlations. We provide a human preference study and metric-correlation analysis to examine the benefit of improving multiple reward dimensions simultaneously. We show the importance of broad reward coverage: improving across multiple dimensions leads to more generally preferred outputs than optimizing a single metric in isolation.

- • Appendix D: Additional Implementation Details. We describe implementation details for reproducing MARBLE in distributed training, focusing on how to extract, synchronize, and harmonize per-reward gradients under DDP.
- • Appendix E: Future Work. We discuss future directions, including scaling MARBLE to larger reward sets and extending reward-balanced optimization to video generation and generative world models.

#### B Additional Qualitative Comparisons

We provide additional qualitative results in Figure S1 and additional comparisons in Figure S8, S9, S10 and S11 to further illustrate the effectiveness of MARBLE. The visualizations cover three representative specialist capabilities: text rendering, attribute and spatial composition, and counting. As shown in Figure S1, MARBLE produces legible and semantically consistent text across diverse contexts, preserves fine-grained attribute-object bindings and spatial layouts, and follows counting constraints while maintaining coherent visual quality.

The comparisons in Figure S8, S9, S10 and S11 further highlight the limitations of existing baselines. The weighted-sum baseline often fails to improve all aspects simultaneously: although it can sometimes satisfy specific requirements such as object counts or attributes, its overall visual quality is visibly degraded. Compared with DiffusionNFT, MARBLE generates sharper images with fewer blur and distortion artifacts. Moreover, DiffusionNFT relies on a heavily tuned training schedule to obtain a competitive unified model, whereas MARBLE achieves balanced improvements through gradient-space reward harmonization. Overall, MARBLE better balances different reward dimensions, producing visually sharper images while more reliably satisfying text rendering, attribute binding, spatial layout, and counting requirements. These qualitative results are consistent with the quantitative improvements and demonstrate the strengths of MARBLE.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Text Rendering

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Attributes and Positions

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Counting

- Figure S1: Additional qualitative results of MARBLE. Our one single model demonstrates simultaneous improvements in text rendering, attribute and position understanding, and counting. MARBLE generates legible text, preserves fine-grained attribute-object bindings and spatial layouts, and follows counting constraints while maintaining coherent visual quality.

#### C Additional Ablations and Analyses

##### C.1 Key Insights and Takeaways

This section summarizes the main takeaways of MARBLE as a whole: what problem it addresses, why its gradient-space design is effective, and how to use it in practice.

- • The central problem is scalar reward aggregation, not merely a poor choice of scalar weights. Table 1 shows that direct simultaneous optimization with a weighted-sum reward underperforms on several reward dimensions. The gradient-alignment analysis in Figure S2 provides further evidence that scalar aggregation weakens reward-specific optimization signals. This suggests that multi-reward diffusion RL should preserve reward-specific supervision rather than heuristically collapse all feedback into a single scalar reward.
- • The key contribution of MARBLE is gradient-space balancing with per-reward credit assignment. By maintaining separate advantages and harmonizing reward-specific gradients, MARBLE outperforms the simultaneous weighted-sum baseline on all five optimized rewards within a single model, while matching or slightly surpassing the sequentially trained baseline that requires extensive manual schedule tuning (Table 1). The ablations in Table 3 further show that both gradient normalization and adaptive harmonization are important: removing

normalization leads to optimization failure, while fixed uniform coefficients, i.e., αk = 0.2, weaken the balance between general image-quality rewards and harder specialist rewards.

- • α partially balances optimization across tasks with different difficulty levels. As shown in Figure S4, the smoothed coefficients do not simply track the corresponding raw reward curves. Instead, they appear to reflect, to some extent, the relative optimization difficulty of each reward. Easier image-quality rewards, such as HPSv2, tend to receive coefficients below the uniform baseline of 0.2, whereas harder specialist rewards, such as GenEval, can receive larger coefficients, around 0.3 during training. Therefore, a larger coefficient should not be interpreted as indicating that the reward value is higher; rather, it suggests that the current gradient geometry allocates more optimization emphasis to that reward.
- • Amortization and EMA smoothing are practical defaults, not just efficiency tricks. Full per-step harmonization is conceptually clean but expensive and sensitive to batch-level coefficient fluctuations. The amortized update keeps the training speed close to the weightedsum baseline (Table 2), while EMA smoothing reduces abrupt changes in α between full harmonization steps and prevents a transient weak batch from suppressing a reward for an entire amortization window. We therefore use N = 10 and ρ = 0.7 as the default setting

- in the main experiments: N = 10 provides similar performance to N = 5 but is slightly faster because it refreshes coefficients less often, while avoiding the degradation observed at N = 20 (Table S2); ρ = 0.7 is the default EMA setting reported in Table S3.
- • Multi-dimensional image-quality improvement matters. Image quality is multidimensional, which cannot be fully captured by any single reward model. Different metrics, such as PickScore, CLIPScore, HPSv2, Aesthetic Score, ImageReward, and UniReward, emphasize different aspects of generation quality, including human preference, text-image alignment, aesthetics, and perceptual realism. We observe that improving one metric alone does not necessarily translate to consistent gains on others, and can sometimes lead to weaker overall visual quality. In contrast, MARBLE consistently achieves broader improvements across multiple image-quality metrics, suggesting that multi-reward balancing leads to more general and robust perceptual enhancement. Please see more details in Appendix C.7.

##### C.2 Update-Direction Harmony Diagnostics

Update-Direction Harmony: Weighted Sum vs MARBLE (per-batch)

minkcos(d, gk) (Pareto-safety)

meankcos(d, gk) (overall alignment)

varkcos(d, gk) (lower = fairer)

0.6

0.30

0.6

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

0.4

0.25

0.5

0.2

0.20

0.4

0.0

0.15

| | |
|---|---|
| | |

0.3

0.2

0.10

0.2

0.4

0.05

0.1

0.6

0.0

0.00

Weighted Sum MARBLE

Weighted Sum MARBLE

Weighted Sum MARBLE

Figure S2: Update-direction harmony. Per-batch mink, meank, and vark of {cos(d,gk)} for the weighted-sum direction d = gws = K1 k gk and the harmonized direction d = gmarble (Section 3.3). The corresponding aggregate statistics are reported in Table S1.

- Table S1: Update-direction harmony statistics. Averages over n = 5 mini-batches. ↑ means larger is better; ↓ means smaller is better.

Statistic Weighted sum MARBLE ∆

Worst-reward alignment mink cos(d, gk) ↑ −0.1346 +0.3721 +0.5067 Average alignment meank cos(d, gk) ↑ +0.4559 +0.4014 −0.0545 Alignment imbalance vark cos(d, gk) ↓ +0.1605 +0.0058 −0.1548 Conflict rate P(mink cos(d, gk) < 0) ↓ 0.800 0.000 −0.800

For a mini-batch, let gk denote the policy gradient induced by reward Rk, and let d denote the update direction produced by a multi-reward training rule. If cos(d,gk) < 0 for some reward k, then the shared update is anti-aligned with that reward’s own gradient on the same batch. Table S1 shows that the weighted-sum direction has a negative worst-reward cosine on average and produces a negative worst-reward alignment in 80% of the measured mini-batches. In contrast, the harmonized direction raises the worst-reward cosine from −0.1346 to +0.3721 and eliminates negative-minimum batches in this measurement, while keeping the mean cosine similar. Its across-reward variance is also much smaller (0.0058 vs. 0.1605), indicating that the update direction is more evenly aligned with the five reward gradients.

##### C.3 Training Dynamics and Coefficient Adaptation

We complement the Equal-α ablation in Section 4.3 with a closer view of how the smoothed coefficients α¯t,k evolve during training.

Uniform α = 0.2 is competitive on easy rewards but loses on specialists. Figure S5 shows the per-reward learning curves under fixed αk = 0.2 and under MARBLE. On HPSv2, an easy

[Figure 98]

- Figure S3: Training curves of MARBLE across the five optimized rewards. The curves show that all rewards continue improving during training, including both general image-quality rewards and specialist rewards.

[Figure 99]

- Figure S4: Dynamics of the smoothed balancing coefficients α¯t,k during training. The uniform baseline for five rewards is 0.2. The learned coefficients do not directly track raw reward values, but show different allocation patterns across rewards with different optimization difficulty. The curve pareto_fallback_used counts how often the clamp condition discussed in Section 3.4 is triggered.

image-quality reward, fixed αk = 0.2 trains slightly faster than MARBLE in the early iterations, while MARBLE converges to a higher final value. On GenEval, a harder specialist reward, fixed αk = 0.2 trains more slowly and ends at a lower score than MARBLE, matching the Equal-α row of Table 3.

α¯t,k is related to optimization difficulty. As shown in Figure S3 and S4, we do not observe a direct relationship between α¯t,k and the corresponding raw reward value. Instead, α¯t,k appears to be related, to some extent, to how hard each reward is to optimize (Figure S4). On HPSv2, which the base model already largely satisfies, α¯t,k stays below the uniform 0.2 baseline for most of training. On GenEval, which demands precise compositional correctness, α¯t,k often rises to around 0.3. Figure S4 also reports pareto_fallback_used, the number of times the clamp condition in Section 3.4 is triggered; this value stays at zero throughout training, indicating that the clamp is not activated in the plotted run. Together, these observations suggest that the learned coefficients can shift optimization emphasis away from easier rewards and toward harder specialist rewards, helping the five rewards progress more evenly during training (Figure S3).

[Figure 100]

[Figure 101]

- Figure S5: Learning-curve comparison between fixed uniform coefficients αk = 0.2 and full MARBLE. Uniform coefficients make fast early progress on HPSv2, a broad image-quality reward, but converge more slowly and to a lower final score on GenEval, a harder specialist reward.

##### C.4 Amortization Interval

- Table S2: Sensitivity to the amortization interval. We compare different values of N, which controls how often the full harmonization procedure refreshes α∗. Smaller intervals refresh the balancing coefficients more frequently, while larger intervals reuse older coefficients for more update steps.

Rule-Based Model-Based GenEval OCR PickScore CLIPScore HPSv2.1 Aesthetic ImgRwd UniRwd

Interval N

N = 5 0.92 0.96 22.51 0.280 0.350 6.56 1.48 3.43 N = 10 0.94 0.96 22.83 0.286 0.355 6.59 1.53 3.52 N = 20 0.92 0.93 21.97 0.277 0.332 6.31 1.37 3.28

The amortization interval N controls how often the full harmonization procedure recomputes α∗ before the cached coefficients are reused for single-backward updates. A smaller N refreshes the balancing coefficients more frequently, making the update direction more responsive to the current gradient geometry but increasing the average training overhead. A larger N lowers this overhead, but it also reuses potentially stale coefficients for more steps. In our sensitivity runs, N=5 and N=10 give similar performance where N=10 performs only slightly better, suggesting that the EMA-smoothed coefficients are stable enough to be reused over a moderate window. Between these two settings, we choose N=10 because it refreshes the coefficients less frequently and is therefore slightly faster than N=5 while maintaining comparable performance. However, increasing the interval to N=20 leads to a performance drop, indicating that overly infrequent refreshes can make the cached coefficients lag behind the changing reward-gradient geometry. We therefore use N=10 in the main experiments as a practical middle ground between frequent coefficient refresh and lower computational overhead.

##### C.5 EMA Decay for Coefficient Smoothing

The EMA decay ρ controls the adaptivity and stability in coefficient smoothing. Smaller values make the coefficients more responsive to the current gradient geometry, but also more sensitive to mini-batch noise, leading to larger fluctuations in the reward curves. Larger values produce smoother coefficient trajectories, but may become overly inertial and adapt slowly when the relative difficulty of rewards changes during training. We evaluate ρ ∈ {0.1,0.3,0.5,0.7,0.9}, with the quantitative results reported in Table S3, the training curves shown in Fig. S6, and qualitative visualizations provided in Fig. S7. Across these evaluations, ρ = 0.7 provides the best overall performance: it maintains stable optimization, achieves strong performance, and produces visually more faithful and coherent samples. We therefore use ρ = 0.7 as the default setting in all main experiments.

##### C.6 Alternative Heuristic Strategies

We also examine several simple heuristic strategies for balancing multiple rewards in diffusion RL, as shown in Table S4. These include using fixed uniform coefficients and increasing the scalar

- Table S3: EMA decay for coefficient smoothing. The default setting ρ = 0.7 is used in the main experiments; the remaining rows list decay values considered around this default.

Rule-Based Model-Based GenEval OCR PickScore CLIPScore HPSv2.1 Aesthetic ImgRwd UniRwd

EMA decay

ρ = 0.1 0.86 0.80 21.52 0.261 0.292 5.84 1.22 2.98 ρ = 0.3 0.88 0.84 21.76 0.266 0.312 6.03 1.27 3.04 ρ = 0.5 0.93 0.95 22.02 0.276 0.340 6.14 1.48 3.43 ρ = 0.7 0.94 0.96 22.83 0.286 0.355 6.59 1.53 3.52 ρ = 0.9 0.90 0.89 22.14 0.272 0.342 6.26 1.47 3.40

###### =0.1 =0.3 =0.5 =0.7 =0.9

[Figure 102]

Figure S6: Sensitivity to EMA decay ρ. ρ = 0.7 achieves the best overall performance.

weights of more challenging specialist rewards, such as OCR and GenEval, in the weighted-sum objective. However, our experiments show that these heuristic strategies fail to achieve simultaneous improvements across all reward dimensions and can lead to performance degradation on several metrics. This suggests that reward balancing in multi-reward diffusion RL cannot be reliably addressed by manual reward reweighting or coarse reward-level heuristics, further demonstrating the effectiveness of MARBLE.

##### C.7 Human Preference Evaluation and Metric Correlations

Evaluating image generation quality requires considering multiple complementary criteria rather than relying on a single automatic proxy. As shown in Table 1, different metrics do not always rank methods consistently: DiffusionNFT† obtains higher PickScore and CLIPScore, whereas MARBLE achieves the best Composite score and performs better on several broader quality- and preferenceoriented metrics. This discrepancy reflects the fact that automatic metrics capture different aspects of generation quality.

To obtain a more comprehensive assessment of human preference, we conduct a blind rating-based user study. For each method, we randomly sample 30 generated images. All images are anonymized, randomly shuffled, and shown with their corresponding prompts. A total of 20 anonymous participants who are unrelated to the project independently score each image on a 1–5 scale along two axes: text-image alignment and image quality. Higher scores indicate better perceived performance. We report the average score over all ratings in Table S5.

MARBLE receives the highest average score on both text-image alignment and image quality. We do not claim statistical significance from this study; rather, we use it as a complementary human-centered evaluation to automatic metrics. The results suggest that the lower PickScore and CLIPScore of MARBLE do not correspond to lower human-rated quality in this setting. Instead, the user study is

### =0.1 =0.3 =0.5 =0.7 =0.9

[Figure 103]

a photo of four stop signs

[Figure 104]

a photo of a purple keyboard and a red chair

[Figure 105]

a photo of a kite above of a toothbrush

[Figure 106]

a photo of a white handbag left of a red giraffe

Figure S7: Qualitative comparison of different EMA decay values ρ.

- Table S4: Alternative reward-balancing strategies. We compare MARBLE with several heuristic strategies for multi-reward diffusion RL. All variants use the same 5-reward setup and training budget.

Rule-Based Model-Based GenEval OCR PickScore CLIPScore HPSv2.1 Aesthetic ImgRwd UniRwd

Method

MARBLE 0.94 0.96 22.83 0.286 0.355 6.59 1.53 3.52 Reward Dropout 0.80 0.61 21.35 0.267 0.270 5.89 0.73 2.88 Reward Weighting (1:1:1:2:2) 0.90 0.92 21.06 0.271 0.285 5.91 1.19 3.13

more consistent with the broader set of automatic metrics, including the Composite score, supporting the need to evaluate image generation quality with multiple complementary criteria.

Automatic metrics and human judgment. We further analyze how different automatic metrics align with human preference. Specifically, we compute the Pearson correlation between each imagequality-related metric and the two human-rated axes, namely image quality and text-image alignment, across all images in the user study. Table S6 reports the results.

The correlation analysis shows that holistic preference metrics, including HPSv2.1, Aesthetic Score, UniReward, and ImageReward, are positively correlated with both human-rated axes, with HPSv2.1 showing the strongest agreement. In contrast, PickScore and CLIPScore are weaker predictors

- Table S5: User study results. Participants score anonymized and randomly shuffled images on a 1–5 scale for text-image alignment and image quality. Higher scores are better.

Method Text-image alignment ↑ Image quality ↑

DiffusionNFT‡ 3.60 2.79 DiffusionNFT† 4.26 3.58 MARBLE 4.63 4.41

of human ratings. This result further indicates that no single automatic metric is sufficient to characterize human-perceived generation quality. Therefore, the PickScore/CLIPScore advantage of DiffusionNFT† should be interpreted as a metric-specific difference rather than a definitive indication of superior perceptual quality.

The qualitative visualizations in Figure S8–S11 provide further evidence for this conclusion. Across diverse prompts, MARBLE better preserves fine-grained requirements such as text rendering, attribute binding, spatial layout, and counting, while maintaining sharper and more coherent visual quality. In contrast, the weighted-sum baseline often fails to improve all aspects simultaneously, and DiffusionNFT† sometimes produces less sharp or less detailed images despite its strong proxy scores. Taken together, the quantitative results in Table 1, the qualitative comparisons in Figure S8–S11, and the user study consistently show that MARBLE achieves the best overall balance between automatic metrics, visual quality, and human preference.

- Table S6: Pearson correlation between automatic metrics and human ratings, computed on all images of the user study (sorted by correlation with image quality). Higher (more positive) values indicate stronger agreement with the human axis.

Metric r with image quality r with text-image alignment

HPSv2.1 +0.66 +0.56 Aesthetic +0.36 +0.33 UniReward +0.25 +0.39 ImageReward +0.19 +0.35 PickScore +0.17 +0.26 CLIPScore +0.00 +0.09

#### D Additional Implementation Details

We provide full implementation details for reproducing MARBLE on the DiffusionNFT codebase with distributed training.

In distributed data-parallel (DDP) training, each GPU processes a different data shard and gradients are averaged across ranks via AllReduce before the optimizer step. With MARBLE, the standard DDP gradient synchronization is insufficient because we need per-reward gradients before combining them via gradient harmonization.

The synchronization protocol proceeds as follows for each training iteration:

Why no_sync() is necessary. Without no_sync(), DDP would trigger an AllReduce on every backward call. Since we call backward K times (once per reward), this would: (1) average gradients prematurely before we can extract per-reward gradients, and (2) incur K unnecessary collective operations. By wrapping each backward in no_sync(), we defer synchronization to our explicit all_reduce calls, where we synchronize the already-extracted per-reward gradient vectors.

retain_graph handling. The first K−1 backward passes use retain_graph=True because all per-reward losses share the same forward computation graph. The last backward pass uses retain_graph=False to release the computation graph and free memory.

Algorithm 1 MARBLE DDP Synchronization Require: Model with DDP wrapper, K per-reward losses {ℓk}

- 1: Initialize gradient storage: G = [ ] // list of K flattened gradient vectors
- 2: for k = 1 to K do
- 3: with model.no_sync(): // suppress DDP AllReduce
- 4: ℓk.backward(retain_graph=(k < K))
- 5: gk ← flatten_grads(model) // extract and flatten trainable .grad
- 6: zero_grads(model)
- 7: G.append(gk)
- 8: end for
- 9: for k = 1 to K do
- 10: dist.all_reduce(G[k], op=AVG) // synchronize across ranks
- 11: end for
- 12: All ranks now have identical {gk}Kk=1
- 13: Solve harmonization locally → identical α∗ on every rank
- 14: d∗ ← k αk∗gˆk
- 15: unflatten_to_grad(d∗, model) // restore to parameter .grad
- 16: Optimizer step proceeds as normal

#### E Future Work

Scalability to More Rewards. This work studies reward balancing with five reward dimensions, while scaling to a larger and more diverse set of rewards remains an important future direction. In future work, we aim to further investigate the scalability of MARBLE and develop more effective balancing strategies for handling increasingly diverse and potentially conflicting reward signals.

Extension to Video and World Models. Extending MARBLE to video generation is another promising direction, especially in the context of generative world modeling. Video models require jointly optimizing a broader set of objectives, including visual fidelity, temporal consistency, motion realism, and physical plausibility. These heterogeneous objectives are central to world models, which require not only high-quality generation but also coherent dynamics and plausible long-horizon evolution. We believe that MARBLE can serve as a promising optimization strategy for addressing these tasks.

###### DiffusionNFT Weighted-Sum SD3.5

###### MARBLE

[Figure 107]

a photo of a yellow train

[Figure 108]

a photo of a refrigerator

[Figure 109]

a photo of a red stop sign

[Figure 110]

a photo of four chairs

[Figure 111]

a photo of three pizzas

[Figure 112]

a photo of a brown giraffe and a white stop sign

###### MARBLE DiffusionNFT Weighted-Sum SD3.5

[Figure 113]

a photo of a yellow parking meter and a pink refrigerator

[Figure 114]

a photo of a purple sheep and a pink banana

[Figure 115]

a photo of a cake above of a backpack

[Figure 116]

a photo of a chair left of a zebra

[Figure 117]

a photo of a skateboard

[Figure 118]

a photo of a green tennis racket and a black dog

###### MARBLE DiffusionNFT Weighted-Sum SD3.5

[Figure 119]

A close-up photograph of an engraved silver ring with the inscription "Forever Yours" delicately etched into its surface, set against a soft, blurred background of romantic, warm tones.

[Figure 120]

A superhero stands proudly, their cape billowing in the wind, with the emblem "Heroes United" prominently displayed on the chest. The setting sun casts a golden glow, highlighting the intricate details of the emblem and the hero's determined expression.

[Figure 121]

A worn medieval scroll unfurls against an ancient wooden desk, revealing the ominous phrase "Here Be Dragons" in elegant calligraphy, surrounded by intricate illustrations of mythical beasts and nautical maps.

[Figure 122]

A vintage clock tower stands tall in a bustling city square, its face altered to read "No Time Like Now" in elegant, antique lettering, surrounded by pigeons and the shadows of passing pedestrians.

[Figure 123]

A realistic photograph of a calendar page with a reminder note that reads "Dentist 2 PM", placed on a desk with a pen and a cup of coffee, under a soft morning light.

Figure S10: Additional qualitative comparisons.

###### MARBLE DiffusionNFT Weighted-Sum SD3.5

[Figure 124]

a Ferari car that is made out of wood

[Figure 125]

Selfie photo of Jesus and Pope.

[Figure 126]

Man with dreads on a beach during a purple thunderstorm.

[Figure 127]

Dark dim dramatic, an image of a girl in a magical world, galaxy sky, long brown hair, artistic, trending on artstation, unreal engine 5.

[Figure 128]

A photo of a person with the head of a cow, wearing a tuxedo and black bowtie. Beach wallpaper in the background.

[Figure 129]

A giraffe by escher, insanely detailed, photorealistic, 8k, ultra high resolution, volumetric lighting, taken with canon eos.

Figure S11: Additional qualitative comparisons.

