# arXiv:2606.16519v1[cs.CV]15Jun2026

## BadWorld: Adversarial Attacks on World Models

https://linghuiishen.github.io/BadWorld/

##### Linghui Shen Mingyue Cui Xingyi Yang∗

The Hong Kong Polytechnic University {ling-hui.shen, ming-yue.cui}@connect.polyu.hk, xingyi.yang@polyu.edu.hk

#### Abstract

Visual world models (VWMs) synthesize interactive, action-conditioned rollouts from a single context image. However, it remains an open question how robust these models are to adversarial perturbations. Standard adversarial attacks fail to assess this vulnerability because attackers lack ground-truth future videos and cannot predict subsequent user controls. We introduce BADWORLD, a label-free adversarial framework tailored for autoregressive VWMs that systematically overcomes both constraints. First, to bypass the need for future supervision, we propose a self-supervised velocity attack that directly disrupts the early denoising dynamics of the model. Second, to ensure the attack generalizes across unpredictable user actions, we formulate a trajectory-adaptive bi-level optimization that actively mines hard control sequences to forge control-agnostic perturbations. Evaluated on representative VWMs with continuous and discrete controls, BADWORLD exposes severe structural fragility. Visually indistinguishable adversarial images reliably trigger catastrophic degradation in future rollouts, leading to incomplete denoising, structural collapse, and control inconsistency. These findings reveal critical risks for deploying VWMs in safety-critical systems while highlighting a practical mechanism for privacy protection.

#### 1 Introduction

Visual world models are moving from passive video generators to interactive simulators. Given a single context image and a sequence of user-defined actions, they can synthesize action-conditioned future videos [20–22]. This ability makes them useful for interactive games [8, 24, 46, 61], robotics [39, 7, 4, 14], and autonomous driving [9, 16, 28, 29, 60, 66]. As these models generate increasingly coherent rollouts, a common belief has emerged: visual world models may have implicitly learned physical and geometric rules of the world [47, 48, 42, 34].

This progress raises a fundamental robustness question: are the learned dynamics of current world models stable under small input perturbations? This question is especially important for safetycritical applications, where fragile rollouts may undermine simulation, prediction, or planning. In this work, we use adversarial perturbations as a stress test for this form of temporal robustness.

However, existing adversarial attacks on generative models do not directly fit world models. Prior work mainly targets text-to-image personalization [64, 12, 43, 37, 40, 56, 51], image-to-image editing [50, 57, 52], and conventional image-to-video or video generation pipelines [59, 38, 65, 17, 54, 13, 44]. These attacks are usually designed for fixed generation conditions, reference outputs, or localized spatial-temporal degradation. In contrast, autoregressive world models are interactive, history-dependent, and control-conditioned [1, 69, 24, 8, 61, 46], which leads to two central challenges.

▷ C1: Missing future supervision. A world-model adversary only observes a single context image, without paired future videos, ground-truth trajectories, or predefined correct rollouts. Therefore, reference-based adversarial losses are not directly applicable.

∗Corresponding author.

Preprint.

▷ C2: Unknown future controls. World-model rollouts depend on future camera paths, navigation commands, or discrete user actions. An attack optimized for one fixed trajectory may fail when the control signal changes.

To address these challenges, we propose BadWorld, a label-free adversarial framework for autoregressive world models. Given a pretrained world model and a single clean context image, BadWorld learns an imperceptible perturbation that drives future rollouts into out-of-distribution behaviors, without requiring paired future videos or knowledge of the user’s future actions. Specifically, BadWorld consists of two technical components.

- ▷ S1: Self-supervised velocity attack. To address C1, we attack the model’s predicted velocity space instead of comparing outputs with unavailable ground truth. We use the model’s own denoising dynamics as supervision, together with an early-denoising approximation and a simple context-based history proxy. This yields a label-free attack that requires neither future videos nor action annotations.
- ▷ S2: Trajectory-adaptive optimization. To address C2, we formulate the attack as a bi-level optimization problem. The outer loop searches for hard trajectories under which the current perturbation is least effective, while the inner loop updates the perturbation against these trajectories. This produces a more control-agnostic adversarial image that is harder to bypass by simply changing the action sequence.

We evaluate BadWorld on representative autoregressive world models with both continuous camera control and discrete action control. Our results show that current world models are surprisingly fragile: adversarial images remain visually close to clean inputs, yet the generated rollouts can suffer from incomplete denoising, structural collapse, semantic drift, and loss of control consistency. Across models and metrics, Velocity-Min consistently produces strong degradation, while trajectory-adaptive optimization further improves robustness across difficult trajectories.

Our findings reveal a robustness risk for deploying world models in safety-critical systems. They also suggest a practical privacy application: imperceptible perturbations can help protect images from unauthorized interactive generation.

Our contributions are summarized as follows: (1) We formalize adversarial attacks on autoregressive world models. (2) We propose BadWorld, a self-supervised attack framework that manipulates the model’s predicted velocity and introduces four label-free objectives. (3) We introduce trajectoryadaptive bi-level optimization to produce perturbations that remain effective across diverse future controls. (4) We show that representative interactive world models are highly fragile; imperceptible perturbations can severely corrupt future rollouts, with implications for both safety and privacy.

#### 2 Related Work

##### 2.1 Controllable Visual World Models

Recent advancements in diffusion and flow-matching frameworks [26, 41, 30, 10, 68] have shifted video generation from passive synthesis toward interactive simulation [6, 11, 5, 55, 35, 49, 67, 63]. Unlike standard generators, visual world models (VWMs) aim to internalize learned physics and transition dynamics, evolving in response to agent actions and temporal history. A prominent trend involves adapting pretrained models into autoregressive frameworks for viewpoint-aware rollouts. Within this landscape, control is typically implemented through three pathways: explicit continuous camera trajectories [3, 69, 62], discrete action tokens [24, 15, 33, 53], or implicit text descriptions [46]. We focus on these autoregressive denoising VWMs, selecting models from both continuous and discrete paradigms to investigate their adversarial robustness. This assesses whether these interactive environments reliably maintain consistency and control adherence, which is essential for safety-critical deployment.

##### 2.2 Adversarial Attacks on Generative Models

Adversarial research in generative modeling typically focuses on two objectives: assessing structural vulnerabilities and enhancing privacy protection. Early work predominantly targeted image generation, employing imperceptible pixel-level perturbations to disrupt generation process. Studies on Text-to-Image (T2I) personalization employ training data poisoning to prevent unauthorized model fine-tuning [64, 12, 43, 37, 40, 56], while Image-to-Image (I2I) attacks perturb input context images to trigger abnormal outputs for privacy protection [50, 57, 52]. Building upon prior image-based

work, adversarial paradigms have recently extended to video generation, where temporal dynamics introduce new attack surfaces. Specifically, identified threats include backdoor triggers in text prompts [59], jailbreaking techniques [38], and adversarial trajectories in drag-based I2V systems [65]. Recent studies further investigate world-model attacks via physical-condition perturbation in driving scenes [18] or automated attack search for world agents [19]. However, they address condition-level or search-level attacks, while we study image-space attacks on video world models. Most relevant to our work are approaches that target the input images in I2V pipelines [17, 54, 13, 44]. While effective, these frameworks are primarily evaluated on standard diffusion-based backbones like SVD [6], CogVideoX [27], or Wan2.1 [55]. A critical gap remains in evaluating the adversarial robustness of autoregressive video world models, particularly those conditioned on interactive signals like camera trajectories or action sequences. To bridge this gap, we propose the first adversarial attack framework tailored to unique architectures of autoregressive video world models.

#### 3 Background

This section provides the necessary background for studying the robustness of video world models. We first review autoregressive video generation and its flow-matching formulation, which together establish the foundation for defining our adversarial objective.

##### 3.1 Preliminary: Autoregressive Video Generation for World Models

In the study, we focus on the autoregressive (AR) model for world generation. Unlike bidirectional generation, the AR approach decomposes a video sequence into K segments (chunks) {z1,...,zK}. Given context frame x ∈ RH×W×3, the model factorizes the joint distribution:

K

p(zi|z<i,x,τi) (1)

p(z1:K|x,τ1:K) =

i=1

The control τi typically encodes camera motions or actions. Each segment zi is generated conditioned on context x, history z<i (via sliding window), control τi, and an optional prompt p.

Most recent AR world models instantiate each chunk generator with diffusion or flow-matching [41] dynamics. For a target chunk zi, Flow Matching constructs an interpolated latent between the data and Gaussian noise ϵ ∼ N(0,I), as that zti = (1 − t)zi + tϵ. The velocity network vθ is trained to predict the transport direction v∗ = dz

i t

dt = ϵ − zi by minimizing L(θ) = Ei,t,ϵ vθ(zti,t | z<i,x,τi) − v∗ 22 . (2)

During inference, zi is generated via sequential self-rollout. While capturing temporal dependencies, this AR structure is sensitive to error accumulation, amplifying small perturbations over the rollout.

##### 3.2 Problem Setup and Challenges

Given a pretrained autoregressive world model Gθ, we aim to construct an adversarial context image

xadv = x + δ, ∥δ∥∞ ≤ η, (3) such that the generated rollout videos under the same control sequence becomes substantially different from the clean rollout. The ideal rollout-level objective is

δ∗ = arg max

D(Gθ(x + δ,τ1:K),Gθ(x,τ1:K)), (4)

∥δ∥∞≤ϵ

where Gθ(·,τ1:K) is the video generated under controls τ1:K, and D measures the difference between the adversarial and clean videos. Intuitively, the attack finds a small perturbation δ within budget η that maximizes this difference. A large difference indicates OOD behavior, such as visual artifacts, temporal inconsistency, motion collapse, or semantic drift. However, Eq. 4 presents two challenges.

- C1: Missing future supervision. The first challenge is that the attacker only observes the context image x and has no ground-truth future video. Since there is no predefined “correct” rollout, the adversarial objective in Eq. 4 cannot be directly implemented. The attack must therefore be label-free and self-supervised.
- C2: Unknown future controls. A second challenge comes from the interactive nature of world models. The generated rollout depends on future control signals, such as camera paths, navigation commands, or discrete user actions. However, these controls may be unknown at attack time and can vary across users. A perturbation optimized for one fixed trajectory may therefore fail when the control sequence changes. A robust attack should remain effective across diverse future controls.

Autoregressive Video World Model + Rollout

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

AR Video World Model

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### Adversarial Objectives

Context Image

𝑭 𝝉;𝜹 = 𝔼𝝃 𝓛𝒂𝒕𝒌 𝒙 + 𝜹;𝝉,𝝃

###### Outer Loop: Hard Trajectory Search

| | |
|---|---|
| | |

fitness Sample For each chunk :

Hard Pool 𝜏hard

Velocity-Max

Sample Candidates Trajectory

ℒ𝒱ℳ𝑎𝑥 = − 𝑣ො 22.

Pool 𝜏​

CMA-ES

As Fitness

Top K

Update

| | |
|---|---|
| | |

###### Velocity-Min

𝓝​ 𝒎 𝒈 , 𝝈 𝒈 𝟐𝑪 𝒈

𝓛𝒱ℳ𝒊𝒏 = 𝒗ෝ 𝟐𝟐.

Add to Pool

Drift-Max:

𝑣ො

Inner Loop：Perturbation Update

Sample 𝜉 = 𝜏,𝑝,𝑛,𝑡,𝜖 , 𝜏~𝜏hard

2

ℒ𝒟ℳ𝑎𝑥 = − 𝑣ො − 𝑣ref

𝑣ref

Attention Window

2

[Figure 12]

AR Video

World Model Predicted

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

𝑣ො

Drift-Min:

𝑣ො 𝑣ref

2

Context Image

ℒ𝒟ℳ𝑖𝑛 = 𝑣ො − 𝑣ref

[Figure 25]

2

+

Duplicates Target Chunk

∇𝛿ℒ𝑎𝑡𝑘

𝑣ref = 𝜖 − 𝑥ctx

𝛿𝑘 = Π𝜂 𝛿𝑘−1 − 𝛼 ⋅ sign ∇𝛿ℒ 𝑥 + 𝛿𝑘−1;𝜉

- Figure 1: BadWorld Pipeline. An autoregressive video world model generates rollouts conditioned on context and camera (top). The outer loop employs CMA-ES for hard trajectory mining (middle), while the inner loop updates the adversarial perturbation under label-free settings (bottom). Four velocity-based objectives are provided (right), with Velocity-Min (LVMin) serving as the default.

These two challenges guide the design of our method. To address C1, we replace the unavailable rollout-level supervision with a self-supervised velocity-level attack objective that perturbs the model’s denoising dynamics. To address C2, we introduce trajectory-adaptive optimization that actively searches for hard control trajectories and optimizes the perturbation against them.

#### 4 Methodology

Following the two challenges identified, BADWORLD consists of two main components. First, to address C1, we design a label-free velocity attack objective that uses the model’s own denoising dynamics as supervision. Second, to address C2, we introduce trajectory-adaptive rollout optimization, which mines hard control trajectories and improves perturbation effectiveness across control signals.

##### 4.1 Label-Free Velocity Attack Objective

To address the absence of future supervision (C1), we avoid directly optimizing the final rollout discrepancy in Eq. 4. Since ground-truth future videos are unavailable in our setting, Eq. 4 cannot be directly optimized. What’s worse, such a rollout-level objective would require backpropagating through the entire sequence, which is computationally prohibitive.

Instead, we formulate the attack directly at the velocity level. Since velocity determines the denoising direction, corrupted velocity predictions can disrupt local generation and further compound across autoregressive steps. We therefore optimize the perturbation δ to induce harmful velocity behavior.

Formally, we define a state tuple representing the generation context:

ξ = (τ,p,n,t,ϵ), (5) where τ is a control signal, p is an optional prompt, n ∈ {1,...,K} denotes the target chunk, t is a denoising timestep, and ϵ ∼ N(0,I) is Gaussian noise. Given the adversarial context x + δ, the model predicts a velocity

vˆδ = vθ(ϵ,t | hδn,τ,p), (6)

where hδn denotes the history condition used for the n-th autoregressive chunk. Using this notation, we define the general attack objective to find the optimal perturbation δ∗ that minimizes an adversarial loss:

δ∗ = arg min

Eξ [Latk(x + δ;ξ)]. (7)

∥δ∥∞≤η

Here, Latk(x + δ;ξ) specifies the harmful velocity property induced by the adversarial context. In practice, we estimate the expectation by random sampling and update δ using projected gradient

descent (PGD)[45]. Specifically, we instantiate Latk through two types of self-supervised objectives.

Velocity magnitude objectives. The first type of objective attack the magnitude of the predicted velocity:

LVMax(x + δ;ξ) = −∥vˆδ∥22, LVMin(x + δ;ξ) = ∥vˆδ∥22. (8) By maximizing this magnitude, Velocity-Max (LVMax) forces aggressive denoising, which injects unnatural contrast and severe distortions into the chunk. Conversely, Velocity-Min (LVMin) suppresses the update norm, causing the model to retain initial noise and yield incomplete structures. Because of the autoregressive nature of the model, these localized denoising failures compound, leading to severe degradation over long-horizon rollouts.

Velocity direction objectives. The second type of objective attacks the velocity direction. We define a context-anchored reference velocity as vref = ϵ − xctx, where xctx is the encoded original image. This reference represents completely static video that simply preserves the initial frame. Using this reference velocity, we define:

LDMax(x + δ;ξ) = −∥vˆδ − vref∥22, LDMin(x + δ;ξ) = ∥vˆδ − vref∥22. (9) Intuitively, Drift-Max (LDMax) pushes the predicted velocity away from this static baseline video, encouraging semantic drift and temporal inconsistency. Conversely, Drift-Min (LDMin) pulls the velocity toward the static reference, suppressing scene evolution and inducing motion collapse.

Label-free realization. Evaluating these objectives requires querying the predicted velocity vˆδ. In standard flow-matching, computing the interpolated latent ztn = (1−t)zn +tϵ and the target velocity v∗ = ϵ − zn both require the future ground-truth chunk zn. Since zn is unavailable in our label-free setting, we introduce two practical approximations.

Early-denoising approximation. When the timestep t approaches 1, the interpolated latent is overwhelmingly dominated by Gaussian noise:

ztn = (1 − t)zn + tϵ ≈ ϵ. (10) By restricting queries to early denoising timesteps, we can input ϵ directly, entirely bypassing zn. Attacking these early stages remains highly effective: early disruptions alter the global structure, seeding fundamental errors that drastically amplify throughout the autoregressive rollout.

Context-based history proxy. Autoregressive generation also demands a historical condition z<n. Because the true future rollout is unavailable, we construct a differentiable proxy by repeating the encoded adversarial context:

hδn = Repeat(E(x + δ),Ln), (11) where Ln is the required history length for the n-th chunk. This proxy provides a reliable pathway for the perturbation δ to influence the velocity prediction. Combined with the early-denoising approximation, this establishes the fully label-free velocity query (Eq. 6), allowing us to optimize the objectives (Eq. 8–Eq. 9) without ground-truth future videos or explicit annotations.

- 4.2 Trajectory-Adaptive Bi-Level Optimization Although the self-supervised objectives in Sec. 4.1 address C1, interactive world models remain sensitive to control signals τ. Perturbations optimized for a single action sequence may overfit to its motion pattern and fail under unpredictable controls (C2). To ensure generalization across control sequences, we propose trajectory-adaptive bi-level optimization. The core intuition is that an attack that remains effective against hard trajectories is more likely to generalize to diverse user controls.

Min-Max Formulation. Let ct be the camera pose at frame t, and τ = {c1,...,cT} denote a continuous camera trajectory over T target frames. We frame our trajectory-adaptive attack as a min-max optimization game:

F(τ;δ), F(τ;δ) = Eξ∼Ω(τ) [Latk(x + δ;ξ)], (12)

min

max

τ∈T

∥δ∥∞≤η

where T is the feasible trajectory space, and Ω(τ) denotes the distribution of label-free query states whose control component is fixed to the candidate trajectory τ. The inner minimization updates the image perturbation δ to minimize the velocity surrogate under the current hard trajectories, while the outer maximization identifies hard trajectories that currently resist the attack and yield the highest expected loss F(τ;δ).

To parameterize the search space T efficiently, we discretize the continuous control at each frame into a compact vector preserving the primary degrees of freedom:

###### ct = (ψt,ft,st), τ = [ψ1,f1,s1,...,ψT,fT,sT], (13)

where ψt (yaw), ft (forward translation), and st (lateral shift) define the viewpoint transformation. Further conversion details are provided in the Appendix.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Clean

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Velocity Max

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

###### Velocity Min

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Drift max

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Drift min

Context Frame Generated Video

- Figure 2: Qualitative comparison on Astra. Velocity-Max and Drift objectives induce color distortions. Velocity-Min causes severe geometric collapse.

Outer loop: hard trajectory mining. The outer maximization in Eq. 12 aims to discover “hard” trajectories that resist the perturbation. However, optimizing this sequence is non-trivial; the autoregressive generation process is largely non-differentiable with respect to the control signals.

To search without gradients, we employ the Covariance Matrix Adaptation Evolution Strategy (CMAES)[23]. Unlike independent random sampling, CMA-ES adapts a multivariate Gaussian distribution to capture coherent camera motion patterns that effectively challenge the current perturbation.

Specifically, at each generation g, we maintain a search distribution over the trajectory vector τ:

q(g)(τ) = N m(g),(σ(g))2C(g) , τk(g) ∼ q(g)(τ), k = 1,...,λ. (14)

where m(g), σ(g), and C(g) denote the mean, step-size, and covariance matrix. Each sampled trajectory is projected onto T and scored by F(τ;δ) (Eq. 12). CMA-ES then ranks the candidates and updates its distribution: The mean m update shifts the search toward harder trajectories, while the covariance C update increases the probability of sampling successful correlated directions. This is crucial for camera control, where hard cases often arise from coherent temporal motions rather than independent frame changes (details refer to Appendix).

After the final CMA generation, we collect evaluated candidates and retain the top-K trajectories:

Phard(n) = TopKτ∈S(n) F(τ;δ), (15)

where S(n) is the set of trajectories evaluated for chunk n. We maintain a separate pool Phard(n) for each chunk because trajectory difficulty varies across autoregressive steps.

Inner loop: perturbation update. Given the hard trajectories identified by the outer loop, the inner loop updates the adversarial perturbation. At each PGD step, for a selected chunk n, we sample a

trajectory τ from Phard(n) and construct the query state ξτ. The perturbation is updated by:

δk+1 = Πη (δk − α sign(∇δLatk(x + δk;ξτ))), (16) where Πη projects the perturbation back into the valid ℓ∞ bounds. We alternate between hard-trajectory mining and perturbation updates, producing an adversarial context image that induces unstable autoregressive rollouts across diverse controls.

#### 5 Experiments

- 5.1 Experimental Setup Models. We evaluate on two representative open-source autoregressive video world models:

• Astra [69]: Built on Wan2.1 [55], Astra is fine-tuned for causal denoising and continuous camera pose conditioning, making it ideal for studying camera-driven adversarial behaviors.

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

Clean

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Velocity Max

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

###### Velocity Min

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Drift max

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Drift min

Context Frame Generated Video

- Figure 3: Qualitative comparison on Matrix-Game-2.0. Velocity-Max and Drift-Max inflate visual contrast. Drift-Min produces near-static videos. Velocity-Min causes complete structural collapse.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

Clean

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

Velocity Min

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

+Bilevel attack

Figure 4: Bi-level Attack performance.

• Matrix-Game 2.0 [24]: A lightweight, causal few-step generative model distilled from SkyReelsV2-I2V-1.3B [11]. Utilizing discrete action signals, it provides a complementary platform to test robustness under categorical control.

Dataset Setup. Since no established benchmark exists for adversarial attacks on video world models, we construct two 100-image datasets tailored for each model. For Astra, we sample the first frame from natural landscape videos in the SpatialVID dataset [58] and resized to 832 × 480. For MatrixGame 2.0, we extract Grand Theft Auto (GTA) gameplay frames and resize to 640 × 352. To align with the training domain of Matrix-Game-2.0 and enhance clean generation quality, we apply Flux Kontext [36] to standardize primary object appearances. Specifically, vehicles are transformed into black sedans, and human subjects are edited to wear black clothing.

Baselines. To the best of our knowledge, no prior work has studied adversarial attacks on autoregressive video generation for world models. Following I2VGuard [17], we therefore adopt a noise baseline by adding random noise to the context frame, matching the mean and variance of the perturbations produced by our method.

Evaluation Metrics. To assess adversarial impact, we employ VBench[31], VBench++[32], CLIP [25] and MEt3R[2] to evaluate four dimensions. First, we measure contextual consistency by accessing i2v subject and i2v background, which quantifies how well the generated video preserves identity and environmental details from the source image. We exclude i2v-subject metrics for Astra, as the dataset lacks explicit foreground subjects. Second, we evaluate video quality by reporting background consistency, aesthetic quality, and imaging quality to capture visual degradation. Third, we evaluate semantic preservation by computing the CLIP-I [25] score between the first and last frames. Fourth, we measure geometry reliance using MEt3R [2], which evaluates geometric consistency by reconstructing dense 3D geometry for each frame pair and measuring feature mismatch after cross-view projection.

Implementation Details. We execute all attacks on an A800 80GB GPU, simulating realistic labelfree scenarios by restricting optimization to early denoising timesteps and duplicating context frames.

I2V Background ↓

Background Consistency ↓

Aesthetic Quality ↓

Imaging Quality ↓

Method

CLIP-I↓ MEt3R↑

Clean 0.978 0.915 0.501 0.690 0.813 0.151 Random Noise 0.973 0.912 0.490 0.680 0.810 0.150 Self-supervised Attack Objectives

Velocity-Max 0.956 0.891 0.493 0.657 0.779 0.154 Velocity-Min 0.930 0.845 0.405 0.513 0.714 0.264 Drift-Max 0.938 0.866 0.466 0.564 0.750 0.207 Drift-Min 0.968 0.908 0.475 0.601 0.800 0.166 Trajectory Adaptive Bi-level Attack

VMin+Bilevel 0.925 0.842 0.396 0.524 0.712 0.265

Table 1: Comparison of attack objectives against Astra.

I2V Subject ↓

I2V Background ↓

Background Consistency ↓

Aesthetic Quality ↓

Imaging Quality ↓

CLIP-I ↓ MEt3R ↑

Method

Clean 0.897 0.897 0.973 0.512 0.648 0.796 0.098 Random Noise 0.878 0.895 0.974 0.522 0.647 0.785 0.101 Self-supervised Attack Objectives

Velocity-Max 0.788 0.795 0.964 0.443 0.491 0.631 0.119 Velocity-Min 0.708 0.750 0.960 0.352 0.442 0.584 0.204 Drift-Max 0.725 0.743 0.965 0.340 0.364 0.537 0.159 Drift-Min 0.868 0.868 0.964 0.478 0.623 0.631 0.101

Table 2: Comparison of attack objectives against Matrix-Game-2.0 (GTA).

During inference, we autoregressively generate three diverse videos per input using default model configurations. Comprehensive implementation details regarding model-specific conditionings and training hyperparameters are deferred to the Appendix.

##### 5.2 Main Results on Different Attack Objectives

To compare the adversarial objectives introduced in Sec. 4.1, we attack 100 images per model under a noise budget of η = 0.05, evaluating each image across three distinct camera or action sequences.

Comparison on Astra. As shown in Tab. 1 (best results are bolded, and second-best results are underlined), Velocity-Min is the most effective objective, achieving the lowest scores across nearly all metrics. Compared to the clean baseline, it induces the sharpest decline in visual fidelity, reducing aesthetic quality from 0.501 to 0.405 and imaging quality from 0.690 to 0.513. This trend is further evidenced by MEt3R, where Velocity-Min reaches 0.264, representing a 74.8% increase over the clean baseline. Drift-Max follows as a strong runner-up, notably reducing imaging quality to 0.564. Conversely, Velocity-Max causes only moderate degradation, while Random Noise and Drift-Min have negligible impact, closely mirroring the clean baseline. For qualitative comparison, Fig.2 shows that Velocity-Min triggers incomplete denoising and grayish artifacts, whereas other objectives primarily alter color temperature.

Comparison on Matrix-Game-2.0. As detailed in Tab. 1, Matrix-Game-2.0 exhibits a broader vulnerability, where most objectives serve as effective attacks. Velocity-Min remains the most potent, significantly reducing background consistency to 0.845, imaging quality to 0.513, and achieving the highest MEt3R of 0.264. Other strategies like Drift-Max and Velocity-Max also demonstrate clear effectiveness by degrading imaging quality and CLIP-I. As visualized in Fig. 3, Velocity-Min induces severe structural disintegration and noise, while Drift-Max and Velocity-Max primarily generate frames with unnaturally high color saturation. Drift-Min tends to produce near-static videos that fail to capture the intended motion.

Summary. In summary, while Astra shows high selectivity with Velocity-Min being the only truly effective objective, Matrix-Game-2.0 is susceptible to a wider range of perturbations. Across both benchmarks, Velocity-Min consistently remains the strongest objective for degrading video quality.

We therefore adopt the strongest objective, Velocity-Min, for all subsequent experiments. Furthermore, since baseline attacks on Matrix-Game-2.0 often trigger a near-complete collapse that masks performance nuances, we utilize the more challenging Astra benchmark to conduct ablation studies (5.3) and validate the efficacy of our bi-level optimization framework (5.4).

###### Budget Step Size I2V Back. ↓ Back. Consis. ↓ Aesthetic ↓ Imaging Qual. ↓ CLIP-I ↓ MEt3R ↑

η = 0.03 0.003 0.945 0.860 0.429 0.563 0.710 0.272 η = 0.05* 0.005 0.930 0.845 0.405 0.513 0.714 0.264 η = 0.10 0.005 0.903 0.794 0.387 0.507 0.678 0.291

Table 3: Performance under different attack budgets. * indicates the default setting.

Method I2V Back. ↓ Back. Consis. ↓ Aesthetic ↓ Imaging Qual. ↓ CLIP-I ↓ MEt3R ↑

Velocity-Min 0.930 0.845 0.405 0.513 0.714 0.264 +Self-Rollout 0.932 0.843 0.410 0.514 0.701 0.271 -Timesteps Selection 0.976 0.913 0.514 0.715 0.801 0.167

Table 4: Ablation on label-free designs.

##### 5.3 Ablation Studies

Following the experimental setup described in Sec. 5.1, we conduct three ablation studies on Astra [69] to evaluate the impact of different budgets and the effectiveness of our proposed designs.

Ablation on attack budget. We first examine the impact of the perturbation budget η on attack performance. As shown in Tab. 3, increasing η from 0.03 to 0.10 leads to a consistent decline in video quality metrics and a corresponding rise in MEt3R scores, indicating more effective disruption. However, larger budgets also make the adversarial perturbations more visually perceptible. To achieve a balance between attack potency and imperceptibility, we select η = 0.05 as our default setting.

Ablation on diffusion timesteps. We assess timestep selection by comparing our early-denoising strategy (Sec.4.1) against uniform timestep sampling from 0 to 1000, denoted as “-Timesteps Selection” in Tab. 4. Without this early focus, attack efficacy collapses: Aesthetic Quality rises to 0.514 and MEt3R drops to 0.167. Targeting initial structural formation is thus essential to effectively neutralize generative priors.

Ablation on history simulation. To verify the effectiveness of our context-based history proxy (Sec.

- 4.1), we compare it against a self-rollout approach, denoted as “+ Self-Rollout” in Tab. 4. Since performing rollouts at every attack step is computationally prohibitive, we use an interval-based scheme: every 50 steps, we generate four videos from current adversarial context under distinct trajectories to update a history pool. During the subsequent interval, we sample from this pool as the history reference. This sophisticated approach yields negligible improvements, with metrics like MEt3R and CLIP-I showing slight improvement. Given the heavy rollout overhead, duplicating context frames is a more efficient and effective strategy.

Method I2V Back. ↓ Back. Consis. ↓ Aesthetic ↓ Imaging Qual. ↓ CLIP-I↓ MEt3R↑

Velocity-Min 0.935 0.845 0.440 0.556 0.720 0.249 + Bi-level Attack 0.931 0.835 0.425 0.556 0.711 0.256

Table 5: Hard Sample Performance Comparison.

- 5.4 Performance of Trajectory-Adaptive Bi-level Optimization

Following the baseline protocol, we evaluate the bi-level attack on 100 Astra images across three trajectories each. As shown in Tab. 1, trajectory-adaptive optimization further improves attack potency over the standard Velocity-Min objective, though the overall gains are moderate on the full benchmark. This is expected, as Velocity-Min is already highly effective for most samples, leaving limited room for further degradation.

To further assess robustness under more challenging cases, we target the 10 most resilient samples from the Velocity-Min baseline, evaluating each against seven distinct trajectories (Tab. 5). Our method consistently degrades all metrics, reducing Aesthetic Quality to 0.425. As illustrated in Fig. 4, bi-level optimization triggers significantly more pronounced noise and geometric distortion where the baseline struggles. These results confirm its superior generalizability in ensuring rollout collapse regardless of sample difficulty or camera path.

#### 6 Conclusion

In this work, we present BadWorld, an adversarial framework designed for autoregressive visual world models (VWMs). To bypass the need for ground-truth future videos, we propose a selfsupervised velocity attack that directly disrupts early denoising dynamics. To handle unpredictable future controls, we formulate a trajectory-adaptive bi-level optimization that mines hard control sequences to forge control-agnostic perturbations. These findings expose critical safety risks for VWM deployment, yet provide a practical mechanism for privacy protection.

#### References

- [1] Sand. ai, Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, W. Q. Zhang, Weifeng Luo, Xiaoyang Kang, Yuchen Sun, Yue Cao, Yunpeng Huang, Yutong Lin, Yuxin Fang, Zewei Tao, Zheng Zhang, Zhongshu Wang, Zixun Liu, Dai Shi, Guoli Su, Hanwen Sun, Hong Pan, Jie Wang, Jiexin Sheng, Min Cui, Min Hu, Ming Yan, Shucheng Yin, Siran Zhang, Tingting Liu, Xianping Yin, Xiaoyu Yang, Xin Song, Xuan Hu, Yankai Zhang, and Yuqiao Li. Magi-1: Autoregressive video generation at scale, 2025. URL https://arxiv.org/abs/2505.13211.
- [2] Mohammad Asim, Christopher Wewer, Thomas Wimmer, Bernt Schiele, and Jan Eric Lenssen. Met3r: Measuring multi-view consistency in generated images, 2026. URL https://arxiv. org/abs/2501.06336.
- [3] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, and Di Zhang. Recammaster: Camera-controlled generative rendering from a single video, 2025. URL https://arxiv.org/abs/2503.11647.
- [4] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models, 2025. URL https://arxiv.org/abs/2412.03572.
- [5] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation, 2024. URL https://arxiv.org/abs/2401.12945.
- [6] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. URL https://arxiv.org/abs/2311.15127.
- [7] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale, 2023. URL https://arxiv.org/abs/2212.06817.
- [8] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, Yusuf Aytar, Sarah Bechtle, Feryal Behbahani, Stephanie Chan, Nicolas Heess, Lucy Gonzalez, Simon Osindero, Sherjil Ozair, Scott Reed, Jingwei Zhang, Konrad Zolna, Jeff Clune, Nando de Freitas, Satinder Singh, and Tim Rocktäschel. Genie: Generative interactive environments, 2024. URL https: //arxiv.org/abs/2402.15391.
- [9] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving, 2020. URL https://arxiv.org/abs/1903.11027.

- [10] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion, 2024. URL https://arxiv.org/abs/2407.01392.
- [11] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, Weiming Xiong, Wei Wang, Nuo Pang, Kang Kang, Zhiheng Xu, Yuzhe Jin, Yupeng Liang, Yubing Song, Peng Zhao, Boyuan Xu, Di Qiu, Debang Li, Zhengcong Fei, Yang Li, and Yahui Zhou. Skyreels-v2: Infinite-length film generative model, 2025. URL https://arxiv.org/abs/2504.13074.
- [12] June Suk Choi, Kyungmin Lee, Jongheon Jeong, Saining Xie, Jinwoo Shin, and Kimin Lee. Diffusionguard: A robust defense against malicious diffusion-based image editing, 2025. URL https://arxiv.org/abs/2410.05694.
- [13] Rohit Chowdhury, Aniruddha Bala, Rohan Jaiswal, and Siddharth Roheda. Vid-freeze: Protecting images from malicious image-to-video generation via temporal freezing, 2026. URL https://arxiv.org/abs/2509.23279.
- [14] Embodiment Collaboration, Abby O’Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anchit Gupta, Andrew Wang, Andrey Kolobov, Anikait Singh, Animesh Garg, Aniruddha Kembhavi, Annie Xie, Anthony Brohan, Antonin Raffin, Archit Sharma, Arefeh Yavary, Arhan Jain, Ashwin Balakrishna, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Schölkopf, Blake Wulfe, Brian Ichter, Cewu Lu, Charles Xu, Charlotte Le, Chelsea Finn, Chen Wang, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Christopher Agia, Chuer Pan, Chuyuan Fu, Coline Devin, Danfei Xu, Daniel Morton, Danny Driess, Daphne Chen, Deepak Pathak, Dhruv Shah, Dieter Büchler, Dinesh Jayaraman, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Ethan Foster, Fangchen Liu, Federico Ceola, Fei Xia, Feiyu Zhao, Felipe Vieira Frujeri, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Gilbert Feng, Giulio Schiavi, Glen Berseth, Gregory Kahn, Guangwen Yang, Guanzhi Wang, Hao Su, Hao-Shu Fang, Haochen Shi, Henghui Bao, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homanga Bharadhwaj, Homer Walke, Hongjie Fang, Huy Ha, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jaimyn Drake, Jan Peters, Jan Schneider, Jasmine Hsu, Jay Vakil, Jeannette Bohg, Jeffrey Bingham, Jeffrey Wu, Jensen Gao, Jiaheng Hu, Jiajun Wu, Jialin Wu, Jiankai Sun, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jimmy Wu, Jingpei Lu, Jingyun Yang, Jitendra Malik, João Silvério, Joey Hejna, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Jordi Salvador, Joseph J. Lim, Junhyek Han, Kaiyuan Wang, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Black, Kevin Lin, Kevin Zhang, Kiana Ehsani, Kiran Lekkala, Kirsty Ellis, Krishan Rana, Krishnan Srinivasan, Kuan Fang, Kunal Pratap Singh, Kuo-Hao Zeng, Kyle Hatch, Kyle Hsu, Laurent Itti, Lawrence Yunliang Chen, Lerrel Pinto, Li Fei-Fei, Liam Tan, Linxi "Jim" Fan, Lionel Ott, Lisa Lee, Luca Weihs, Magnum Chen, Marion Lepert, Marius Memmel, Masayoshi Tomizuka, Masha Itkina, Mateo Guaman Castro, Max Spero, Maximilian Du, Michael Ahn, Michael C. Yip, Mingtong Zhang, Mingyu Ding, Minho Heo, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Muhammad Zubair Irshad, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Ning Liu, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Osbert Bastani, Pannag R Sanketi, Patrick "Tree" Miller, Patrick Yin, Paul Wohlhart, Peng Xu, Peter David Fagan, Peter Mitrano, Pierre Sermanet, Pieter Abbeel, Priya Sundaresan, Qiuyu Chen, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Martín-Martín, Rohan Baijal, Rosario Scalise, Rose Hendrix, Roy Lin, Runjia Qian, Ruohan Zhang, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Shan Lin, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shubham Tulsiani, Shuran Song, Sichun Xu, Siddhant Haldar, Siddharth Karamcheti, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Subramanian Ramamoorthy, Sudeep Dasari, Suneel Belkhale, Sungjae Park, Suraj Nair, Suvir Mirchandani, Takayuki Osa, Tanmay Gupta, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Thomas Kollar, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Trinity Chung, Vidhi Jain, Vikash Kumar, Vincent Vanhoucke,

- Vitor Guizilini, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiangyu Chen, Xiaolong Wang, Xinghao Zhu, Xinyang Geng, Xiyuan Liu, Xu Liangwei, Xuanlin Li, Yansong Pang, Yao Lu, Yecheng Jason Ma, Yejin Kim, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Yilin Wu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yongqiang Dou, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yue Cao, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunchu Zhang, Yunfan Jiang, Yunshuang Li, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zehan Ma, Zhuo Xu, Zichen Jeff Cui, Zichen Zhang, Zipeng Fu, and Zipeng Lin. Open x-embodiment: Robotic learning datasets and rt-x models, 2025. URL https://arxiv.org/abs/2310.08864.
- [15] Decart, Julian Quevedo, Quinn McIntyre, Spruce Campbell, Xinlei Chen, and Robert Wachen. Oasis: A universe in a transformer. 2024. URL https://oasis-model.github.io/.
- [16] Tuo Feng, Wenguan Wang, and Yi Yang. A survey of world models for autonomous driving,

2025. URL https://arxiv.org/abs/2501.11260.

- [17] Dongnan Gui, Xun Guo, Wengang Zhou, and Yan Lu. I2vguard: Safeguarding images against misuse in diffusion-based image-to-video models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12595–12604, 2025.
- [18] Zhixiang Guo, Siyuan Liang, Andras Balogh, Noah Lunberry, Rong-Cheng Tu, Mark Jelasity, and Dacheng Tao. When world models dream wrong: Physical-conditioned adversarial attacks against world models, 2026. URL https://arxiv.org/abs/2602.18739.
- [19] Zhixiang Guo, Siyuan Liang, Shi Fu, Cheng Guo, Andras Balogh, Mark Jelasity, and Dacheng Tao. Wmattack: Automated attack search for adversarial evaluation of world-model agents,

2026. URL https://arxiv.org/abs/2605.23220.

- [20] David Ha and Jürgen Schmidhuber. World models. 2018. doi: 10.5281/ZENODO.1207631. URL https://zenodo.org/record/1207631.
- [21] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels, 2019. URL https: //arxiv.org/abs/1811.04551.
- [22] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL https://arxiv.org/abs/2301.04104.
- [23] Nikolaus Hansen. The cma evolution strategy: A tutorial, 2023. URL https://arxiv.org/ abs/1604.00772.
- [24] Xianglong He, Chunli Peng, Zexiang Liu, Boyang Wang, Yifan Zhang, Qi Cui, Fei Kang, Biao Jiang, Mengyin An, Yangyang Ren, Baixin Xu, Hao-Xiang Guo, Kaixiong Gong, Size Wu, Wei Li, Xuchen Song, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 2.0: An open-source real-time and streaming interactive world model, 2026. URL https://arxiv. org/abs/2508.13009.
- [25] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. URL https://arxiv.org/abs/ 2104.08718.
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. URL https://arxiv.org/abs/2006.11239.
- [27] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers, 2022. URL https://arxiv.org/ abs/2205.15868.
- [28] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving,

- 2023. URL https://arxiv.org/abs/2309.17080.

[29] Xiaotao Hu, Wei Yin, Mingkai Jia, Junyuan Deng, Xiaoyang Guo, Qian Zhang, Xiaoxiao Long, and Ping Tan. Drivingworld: Constructing world model for autonomous driving via video gpt,

- 2024. URL https://arxiv.org/abs/2412.19505.

- [30] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion, 2025. URL https://arxiv. org/abs/2506.08009.
- [31] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023. URL https://arxiv.org/abs/2311.17982.
- [32] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models, 2024. URL https://arxiv.org/abs/2411.13503.
- [33] Team HunyuanWorld. Hy-world 1.5: A systematic framework for interactive world modeling with real-time latency and geometric consistency. arXiv preprint, 2025.
- [34] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective, 2025. URL https://arxiv.org/abs/2411.02385.
- [35] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models,

2025. URL https://arxiv.org/abs/2412.03603.

- [36] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. URL https://arxiv.org/abs/2506.15742.
- [37] Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran, and Anh Tran. Anti-dreambooth: Protecting users from personalized text-to-image synthesis, 2023. URL https://arxiv.org/abs/2303.15433.
- [38] Guanlin Li, Shuai Yang, Jie Zhang, and Tianwei Zhang. Prime: Protect your videos from malicious editing, 2024. URL https://arxiv.org/abs/2402.01239.
- [39] Xinqing Li, Xin He, Le Zhang, Min Wu, Xiaoli Li, and Yun Liu. A comprehensive survey on world models for embodied ai, 2025. URL https://arxiv.org/abs/2510.16732.
- [40] Chumeng Liang, Xiaoyu Wu, Yang Hua, Jiaru Zhang, Yiming Xue, Tao Song, Zhengui Xue, Ruhui Ma, and Haibing Guan. Adversarial example does good: Preventing painting imitation from diffusion models via adversarial examples, 2023. URL https://arxiv.org/abs/2302. 04578.
- [41] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.
- [42] Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physics-grounded image-to-video generation, 2024. URL https://arxiv.org/abs/2409. 18964.
- [43] Yisu Liu, Jinyang An, Wanqian Zhang, Dayan Wu, Jingzi Gu, Zheng Lin, and Weiping Wang. Disrupting diffusion: Token-level attention erasure attack against diffusion-based customization,

###### 2024. URL https://arxiv.org/abs/2405.20584.

- [44] Zeqian Long, Ozgur Kara, Haotian Xue, Yongxin Chen, and James M. Rehg. Immune2v: Image immunization against dual-stream image-to-video generation, 2026. URL https: //arxiv.org/abs/2604.10837.
- [45] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks, 2019. URL https://arxiv. org/abs/1706.06083.
- [46] Xiaofeng Mao, Shaoheng Lin, Zhen Li, Chuanhao Li, Wenshuo Peng, Tong He, Jiangmiao Pang, Mingmin Chi, Yu Qiao, and Kaipeng Zhang. Yume: An interactive world generation model, 2025. URL https://arxiv.org/abs/2507.17744.
- [47] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsensebased benchmark for video generation, 2024. URL https://arxiv.org/abs/2410.05363.
- [48] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles?, 2025. URL https://arxiv.org/abs/2501. 09038.
- [49] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models, 2025. URL https://arxiv.org/abs/2410.13720.
- [50] Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, and Aleksander Madry. Raising the cost of malicious ai-powered image editing, 2023. URL https://arxiv.org/abs/2302. 06588.
- [51] Shawn Shan, Jenna Cryan, Emily Wenger, Haitao Zheng, Rana Hanocka, and Ben Y. Zhao. Glaze: Protecting artists from style mimicry by text-to-image models, 2025. URL https: //arxiv.org/abs/2302.04222.
- [52] Linghui Shen, Mingyue Cui, and Xingyi Yang. Decontext as defense: Safe image editing in diffusion transformers, 2025. URL https://arxiv.org/abs/2512.16625.
- [53] Robbyant Team, Zelin Gao, Qiuyu Wang, Yanhong Zeng, Jiapeng Zhu, Ka Leong Cheng, Yixuan Li, Hanlin Wang, Yinghao Xu, Shuailei Ma, Yihang Chen, Jie Liu, Yansong Cheng, Yao Yao, Jiayi Zhu, Yihao Meng, Kecheng Zheng, Qingyan Bai, Jingye Chen, Zehong Shen, Yue Yu, Xing Zhu, Yujun Shen, and Hao Ouyang. Advancing open-source world models. arXiv preprint arXiv:2601.20540, 2026.
- [54] Duc Vu, Anh Nguyen, Chi Tran, and Anh Tran. Anti-i2v: Safeguarding your photos from malicious image-to-video generation, 2026. URL https://arxiv.org/abs/2603.24570.
- [55] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei

- Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL https://arxiv.org/abs/2503.20314.
- [56] Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, and Qidong Huang. Simac: A simple anticustomization method for protecting face privacy against text-to-image synthesis of diffusion models, 2024. URL https://arxiv.org/abs/2312.07865.
- [57] Hanhui Wang, Yihua Zhang, Ruizheng Bai, Yue Zhao, Sijia Liu, and Zhengzhong Tu. Edit away and my face will not stay: Personal biometric defense against malicious generative editing,

2025. URL https://arxiv.org/abs/2411.16832.

- [58] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, Xiao-Xiao Long, Hao Zhu, Zhaoxiang Zhang, Xun Cao, and Yao Yao. Spatialvid: A large-scale video dataset with spatial annotations, 2025. URL https://arxiv.org/abs/2509.09676.
- [59] Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, and Baoyuan Wu. Badvideo: Stealthy backdoor attack against text-to-video generation, 2025. URL https: //arxiv.org/abs/2504.16907.
- [60] Yuqi Wang, Jiawei He, Lue Fan, Hongxin Li, Yuntao Chen, and Zhaoxiang Zhang. Driving into the future: Multiview visual forecasting and planning with world model for autonomous driving, 2023. URL https://arxiv.org/abs/2311.17918.
- [61] Zile Wang, Zexiang Liu, Jiaxing Li, Kaichen Huang, Baixin Xu, Fei Kang, Mengyin An, Peiyu Wang, Biao Jiang, Yichen Wei, Yidan Xietian, Jiangbo Pei, Liang Hu, Boyi Jiang, Hua Xue, Zidong Wang, Haofeng Sun, Wei Li, Wanli Ouyang, Xianglong He, Yang Liu, Yangguang Li, and Yahui Zhou. Matrix-game 3.0: Real-time and streaming interactive world model with long-horizon memory, 2026. URL https://arxiv.org/abs/2604.08995.
- [62] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory, 2026. URL https: //arxiv.org/abs/2504.12369.
- [63] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models, 2024. URL https://arxiv.org/abs/2310. 10647.
- [64] Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, and Xiang Wei. Perturbing attention gives you more bang for the buck: Subtle imaging perturbations that efficiently fool customized diffusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), page 24534–24543. IEEE, 2024. doi: 10.1109/cvpr52733.2024.02316. URL http://dx.doi.org/10.1109/cvpr52733.2024.02316.
- [65] Shuhan Xu, Siyuan Liang, Hongling Zheng, Yong Luo, Han Hu, Lefei Zhang, and Dacheng Tao. Ctrlattack: A unified attack on world-model control in diffusion models, 2026. URL https://arxiv.org/abs/2603.13435.
- [66] Kaiwen Zhang, Zhenyu Tang, Xiaotao Hu, Xingang Pan, Xiaoyang Guo, Yuan Liu, Jingwei Huang, Li Yuan, Qian Zhang, Xiao-Xiao Long, Xun Cao, and Wei Yin. Epona: Autoregressive diffusion world model for autonomous driving, 2025. URL https://arxiv.org/abs/2506. 24113.
- [67] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all,

2024. URL https://arxiv.org/abs/2412.20404.

- [68] Hongzhou Zhu, Min Zhao, Guande He, Hang Su, Chongxuan Li, and Jun Zhu. Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214, 2026.

###### [69] Yixuan Zhu, Jiaqi Feng, Wenzhao Zheng, Yuan Gao, Xin Tao, Pengfei Wan, Jie Zhou, and Jiwen Lu. Astra: General interactive world model with autoregressive denoising, 2026. URL https://arxiv.org/abs/2512.08931.

## Supplementary Material

### BadWorld: Adversarial Attack on World Models

In this supplement, we provide further details on BadWorld, including objective explanations, robustness evaluations, implementation specifics, and additional experimental results. Specifically, Sec. A offers a deeper explanation of the velocity magnitude objective, Sec. B evaluates adversarial robustness against image preprocessing and black-box transferability across models, Sec. C details trajectory sampling algorithms and experimental hyperparameters, Sec. D presents extension experiments on Matrix-Game-2 variants, and Sec. E provides further qualitative visuals of our attack.

- A Explanation of the Velocity Magnitude Objective This section further explains the velocity magnitude objectives introduced in the main paper.

In flow-matching [41] frameworks, the denoising network predicts a velocity field vˆθ specifying the direction and rate of latent-state change at each step. To isolate the effect of velocity magnitude, we perform a velocity scaling experiment during inference. Specifically, we modulate the predicted velocity by a positive scalar s, yielding vˆθ′ = s · vˆθ. This operation preserves the prediction direction and changes only its magnitude. The latent state is then updated via zt−∆t = SchedulerStep(s · vˆθ,t,zt) using the scaled velocity.

Empirical results in Fig. 5 and 6 show that the magnitude of vˆθ directly determines visual quality. When velocity is under-scaled (s < 1), the update is too weak to drive the latent state toward the data manifold. This yields under-denoised outputs that appear gray, blurry, and structurally disordered. Conversely, over-scaling velocity (s > 1) pushes latents toward extreme values. While amplifying contrast, it often introduces overshooting artifacts such as extreme saturation and pixel distortion. These observations indicate that a precise velocity norm is vital for a stable denoising trajectory.

Building on these observations, we find that at equivalent scaling intensity, reducing velocity magnitude disrupts video coherence more severely than increasing it. For adversarial purposes, Velocity-Min proves superior to Velocity-Max in triggering rapid structural collapse. We hypothesize that this vulnerability stems from autoregressive world models: weakened updates prevent the model from reaching stable manifolds, causing subtle denoising errors to accumulate and compound through the temporal history window. This also motivates our choice of Velocity-Min as the final attack objective.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Original

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

Original

Scale 1.1

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Scale 1.1

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

Scale 0.9

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Scale 0.9

Figure 5: Scale the velocity magnitude for Astra.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

Original

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

Original

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

Scale 1.2

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

Scale 1.2

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Scale 0.8

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Scale 0.8

Figure 6: Scale the velocity magnitude for Matrix-Game-2.0.

#### B Robustness and Transferability

Following the setup in Sec. C.2, we evaluate the robustness and transferability of our method. We randomly select 10 context images and generate videos across three distinct camera trajectories.

##### Robustness to Image Preprocessing.

We examine the resilience of our adversarial perturbations against common image preprocessing techniques, including Gaussian noise, JPEG compression, and Total Variation (TV) denoising. As shown in Tab. 6, Velocity-Min remains highly effective under light Gaussian noise (σ = 0.5), maintaining a strong MEt3R score of 0.255 and significantly low imaging quality. While more aggressive preprocessings like JPEG and TV denoising partially mitigate the attack, our method still induces measurable degradation compared to the clean baseline. These results demonstrate that our perturbations are not easily neutralized by standard image-level filters, confirming their practical robustness.

I2V Background ↓

Background Consistency ↓

Aesthetic Quality ↓

Imaging Quality ↓

Method

CLIP-I↓ MEt3R↑

Clean 0.978 0.915 0.501 0.690 0.813 0.151 Velocity-Min 0.922 0.816 0.405 0.506 0.701 0.266 +gaussianσ0.5 0.924 0.823 0.407 0.514 0.687 0.255 +gaussian σ0.7 0.927 0.846 0.437 0.575 0.721 0.245 +JPEG 75 0.969 0.896 0.477 0.670 0.715 0.164 +TV λ0.04 0.960 0.882 0.454 0.587 0.749 0.191

Table 6: Robustness to Image Preprocessing.

##### Black-box Transfer across Models and Limitations.

To evaluate the black-box transferability of our method, we conduct cross-model evaluations between Matrix-Game-2.0 [24] and Astra [69]. Given their different default input resolutions, we resize the generated adversarial context images to match the target model’s configuration prior to inference. As shown in Tab. 7: (1) Trained on Matrix-Game, test on Astra (Left): Our method demonstrates clear transferability. The video quality metrics notably decline compared to the clean baseline. Specifically, Background Consistency drops from 0.878 to 0.846, and Imaging Quality decreases from 0.553 to 0.515. This confirms the attack’s effectiveness on an unseen architecture. (2) Trained on Astra, Test on Matrix-Game (Right): The adversarial transferability is severely limited, showing negligible performance degradation. This asymmetry highlights a key limitation: excessive resizing required to bridge resolution gaps likely disrupts the delicate pixel-level perturbations, thereby degrading transferability in black-box scenarios.

Matrix-Game → Astra Astra → Matrix-Game Method Back. Consis. Aesthetic Imaging Qual. Back. Consis Aesthetic Imaging Qual.

Clean 0.878 0.492 0.553 0.955 0.488 0.659 Velocity-Min 0.846 0.466 0.515 0.954 0.485 0.645

Table 7: Black-box Transfer across Models.

#### C Implementation Details

- C.1 Trajectory Sampling

- C.1.1 Camera Trajectory Formulation

To ensure the adversarial search space is both expressive and tractable, we represent the camera control sequence using a compact, low-dimensional parameterization. At any target frame t, the control signal is defined as a triplet ct = (ψt,ft,st), representing the yaw angle, forward displacement, and lateral shift, respectively. This triplet is uniquely mapped to a relative camera pose matrix Pt ∈ R3×4:

Pt =

cosψt 0 sinψt st 0 1 0 0

−sinψt 0 cosψt −ft

The first 3 × 3 block encodes the yaw rotation around the vertical axis, while the translation vector captures horizontal and forward motions. For an autoregressive chunk comprising T = 8 frames, the complete continuous trajectory is vectorized as τ = [ψ1,f1,s1,...,ψT,fT,sT]⊤ ∈ R3T.To ensure the optimized trajectories remain physically plausible and strictly within the model’s in-distribution control space, we enforce bound constraints B on both the absolute magnitudes and the temporal derivatives of the sequence. Specifically, the frame-wise controls are bounded by ψmax = 0.05, fmax = 0.05, and smax = 0.025. To prevent erratic, discontinuous camera jumps, the step-wise variations are strictly bounded by ∆ψ = 0.03, ∆f = 0.03, and ∆s = 0.015. All sampled or optimized trajectories are projected onto this feasible convex set.

- C.1.2 Stochastic Trajectory Sampling via Random Walk

For the baseline self-supervised objectives (Sec.4.1) where adaptive mining is not employed, we approximate the expectation over control signals through stochastic trajectory sampling. At each step of the Projected Gradient Descent (PGD) [45], a fresh trajectory is generated to prevent the adversarial perturbation from overfitting to a static camera motion. The initial frame control c1 is drawn uniformly from the frame-wise feasible range. Subsequent frames follow a Gaussian random walk to ensure temporal coherence:

ct = ct−1 + ηt, ηt ∼ N(0,Σrw)

where Σrw = diag(σψ2,σf2,σs2). The resulting trajectory is subsequently clipped to satisfy the variation and range constraints defined in Sec. C.1.1. This unbiased stochastic exploration ensures the generated perturbation maintains generalizability across a wide distribution of camera motions.

- C.1.3 Trajectory-Adaptive Bi-Level Execution

For the trajectory-adaptive bi-level attack (Sec. 4.2), during the outer optimization loop, we search for trajectories that maximize the empirical adversarial loss. To account for the stochasticity of the denoising process, the fitness of a candidate trajectory τ is evaluated via Monte Carlo approximation over M evaluation contexts (fixing the history length, timestep, and latent noise):

M

1 M

Latk(x + δ;τ,ξj)

F(τ;δ) =

j=1

For efficiency, we default to M = 1. We maintain separate hard trajectory pools Phard tailored for different autoregressive history lengths, recognizing that the sensitivity of the model to specific camera motions diverges as the rollout progresses. During the inner loop, the PGD update dynamically queries these pools. For standard short-horizon steps, the attack defaults to the random-walk sampler to maintain broad robustness. For extended horizons where error accumulation is critical, the attack samples directly from Phard, forcing the inner minimization to prioritize control sequences under which the current perturbation is least effective.

##### C.1.4 Covariance Matrix Adaptation Evolution Strategy (CMA-ES)[23]

[Figure 164]

Figure 7: CMA trajectory updates.

The outer maximization over τ ∈ R3T presents a challenging non-convex optimization problem over a continuous sequence. Importantly, computing exact gradients through the multi-step autoregressive generation process is computationally prohibitive and prone to gradient shattering. We thus employ CMA-ES[23], a derivative-free evolutionary algorithm, to efficiently navigate this space. At generation g, CMA-ES maintains a multivariate Gaussian search distribution parameterized by a mean trajectory m(g), a global step size σ(g), and a covariance matrix C(g):

q(g)(τ) = N m(g),(σ(g))2C(g)

A population of λ candidate trajectories is sampled, projected into the feasible region, and evaluated against the fitness function −F(τ;δ). Let {τi(:gλ)}µi=1 denote the top µ candidates sorted by fitness. The distribution mean is updated toward the weighted average of these elite candidates:

µ

wiτi(:gλ)

m(g+1) =

i=1

where wi > 0 are the recombination weights. A critical advantage of CMA-ES for sequence mining is the adaptation of the covariance matrix C(g), which captures the strong temporal correlations inherent in hard camera motions (e.g., a coordinated continuous pan and forward zoom). The covariance is updated by integrating both the current elite population and an accumulated evolution path

µ

⊤

wiyi(:gλ) yi(:gλ)

C(g+1) = (1 − ccov)C(g) + ccov

(17)

i=1

where yi(:gλ) = (τi(:gλ) − m(g))/σ(g) is the normalized step. As shown in Fig.7, the trajectories are successively updated to higher velocity norm. Upon termination of the CMA-ES generations, the

globally top-ranked candidates across the evaluation history are distilled into the hard trajectory pool Phard for the inner PGD optimization.

##### C.2 Additional Experimental Details

Training. As discussed in Sec.4.1, the context frame available to users does not come with paired ground-truth videos or viewpoints annotations. To simulate realistic attack scenarios, we introduce two key adjustments: (i) we restrict training to the early denoising phase (timesteps ranging from 950 to 1000 by default) where target frames can be approximated as pure noise, and (ii) we simulate historical context by duplicating the context frame.

For conditioning, Astra uses 30 prompt variants generated via GPT-4o-mini, each semantically aligned with the context frames; we randomly sample one prompt per attack and a unique camera pose per latent frame. Matrix-Game 2.0 requires no text prompts, so we randomly sample discrete actions per frame.

For each model, all four attack objectives are evaluated under identical training schedules (Tab. 8). We conduct training on an A800 80GB GPU.

###### Model VAE Ratio Frames per Chunk Attack Budget Step Size Steps

Astra 4 8 0.05 0.005 300 Matrix-Game 2.0 4 3 0.05 0.004 300

Table 8: Training hyperparameters for different models.

Inference. We follow the default inference configurations (Tab. 9) of the base models and generate videos autoregressively. For each context frame (input image), we produce 3 videos using different camera or action sequences to ensure result diversity.

Model Frames per Chunk Steps per Chunk Target Latent Frames Fps Attn. Window Astra 8 50 33 20 20 Matrix-Game 2.0 3 3 60 5 4

Table 9: Inference settings for both models.

Evaluations. While Astra is evaluated via standard VBench metrics, we adapt our approach for the longer-form Matrix-Game-2. Specifically, we employ VBench-Long for video quality evaluation and segment the original videos into shorter clips to precisely assess contextual consistency. This length-aware strategy ensures a more rigorous and reliable evaluation.

#### D Extension Experiments on Matrix-Game-2 Variants

For the Matrix-Game-2 universal variants, we train the adversarial perturbation with an attack budget of 0.05 and a step size of 0.002 for a total of 700 optimization steps. During inference, we follow the default Matrix-Game-2 configuration: the local attention window is set to 6, each chunk contains 3 latent frames, and generation is performed with 3 sampling steps. For each input image, we generate 27 latent frames in total.

As shown in Table 10 and Figure 8, BADWORLD effectively generalizes to the Matrix-Game-2 Universal variant. The Velocity-Min objective triggers substantial degradation across all quantitative metrics, notably reducing Imaging Quality from 0.652 to 0.515 and Aesthetic Quality from 0.513 to 0.418. Qualitative results further confirm that the adversarial perturbation successfully induces structural collapse in the rollouts, validating the robustness of our framework across different model configurations.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

Clean

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

Velocity Min

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

Clean

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

Velocity Min

Figure 8: Attack Performance on Matrix-Game-2.0 (Universal).

I2V Background ↓

Background Consistency ↓

Aesthetic Quality ↓

Imaging Quality ↓

CLIP-I↓ MEt3R↑

Method

Clean 0.961 0.932 0.513 0.652 0.755 0.162 Velocity-Min 0.874 0.875 0.418 0.515 0.703 0.223

Table 10: Attack on Matrix-Game-2 (Universal)

- E Additional Qualitative Results This section provides additional qualitative results on Astra and Matrix-Game-2.0.

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Clean

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Velocity Max

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

Velocity Min

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Drift Max

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Drift Min

Context Frame Generated Videos

Figure 9: Qualitative Results on Astra.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Clean

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Velocity Min

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

+ Bilevel attack

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

Clean

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Velocity Min

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

+ Bilevel attack

Context Frame Generated Videos

Figure 10: Performance of Bi-Level Attack.

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

Clean

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Adversarial

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Trajectory Generated Videos

Context Frame

Figure 11: Attack performance unfer different camera trajectories.

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

Clean

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

Velocity Max

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Velocity Min

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

Drift Max

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Drift Min

###### Figure 12: Qualitative Results on Matrix-Game-2.0.

