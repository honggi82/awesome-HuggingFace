# arXiv:2604.12322v1[cs.CV]14Apr2026

## SELF-ADVERSARIAL ONE STEP GENERATION VIA CONDITION SHIFTING

Deyuan Liu1∗ Peng Sun2,1∗ Yansen Han2,1 Zhenglin Cheng3,2,1 Chuyan Chen4,1 Tao Lin1,† 1Westlake University 2Zhejiang University 3Shanghai Innovation Institute 4Peking University

ABSTRACT

The push for efficient text to image synthesis has moved the field toward one step sampling, yet existing methods still face a three way tradeoff among fidelity, inference speed, and training efficiency. Approaches that rely on external discriminators can sharpen one step performance, but they often introduce training instability, high GPU memory overhead, and slow convergence, which complicates scaling and parameter efficient tuning. In contrast, regression based distillation and consistency objectives are easier to optimize, but they typically lose fine details when constrained to a single step. We present APEX, built on a key theoretical insight: adversarial correction signals can be extracted endogenously from a flow model through condition shifting. Using a transformation creates a shifted condition branch whose velocity field serves as an independent estimator of the model’s current generation distribution, yielding a gradient that is provably GAN aligned, replacing the sample dependent discriminator terms that cause gradient vanishing. This discriminator free design is architecture preserving, making APEX a plug and play framework compatible with both full parameter and LoRA based tuning. Empirically, our 0.6B model surpasses FLUX-Schnell 12B (20× more parameters) in one step quality. With LoRA tuning on Qwen-Image 20B, APEX reaches a GenEval score of 0.89 at NFE=1 in 6 hours, surpassing the original 50-step teacher (0.87) and providing a 15.33× inference speedup. Code is available here.

[Figure 1]

Figure 1: An overview of generated images.

∗Equal contribution †Corresponding author.

- 1 INTRODUCTION

Continuous generative models now achieve strong fidelity across domains, from photorealistic image synthesis (Dhariwal & Nichol, 2021; Karras et al., 2024) to video generation (Ho et al., 2022; Chen et al., 2025b). This progress is largely driven by diffusion models (Ho et al., 2020; Dhariwal & Nichol, 2021) and flow matching frameworks (Lipman et al., 2022; Ma et al., 2024), which sample by integrating a Probability Flow Ordinary Differential Equation (PF-ODE), from noise to data (Song et al., 2020). The same iterative paradigm also dominates inference cost: multi step integration often requires tens of function evaluations and can be prohibitively expensive (Karras et al., 2024; Nichol & Dhariwal, 2021), motivating sustained interest in one step synthesis (Song et al., 2023; Salimans & Ho, 2022; Yin et al., 2024a).

Achieving number of function evaluations (NFE) = 1 at high resolution exposes a persistent trilemma among generation quality, inference efficiency, and training efficiency (Song et al., 2023; Lu & Song, 2024; Yin et al., 2024a; Sauer et al., 2024a). External adversarial components like a discriminator or auxiliary critic can improve one step realism, but they often hurt scalability by introducing training instability and additional system overhead (Yin et al., 2024a; Kim et al., 2023; Zheng et al., 2025). This overhead becomes especially costly when scaling pretrained backbones or doing parameter efficient tuning. In contrast, regression based distillation (Yin et al., 2024b) and consistency style objectives (Song et al., 2023; Sun & Lin, 2025) are typically easier to optimize, yet they often struggle to match adversarial realism in one step, especially for high frequency textures and fine details (Song et al., 2023; Lu & Song, 2024; Geng et al., 2025; Sun et al., 2025). Complementary to these lines, a recent work, TwinFlow (Cheng et al., 2025), also explores self adversarial methods that build adversarial signals by model itself.

Q: How can we obtain GAN level one step fidelity at NFE=1 without an external discriminator, while remaining scalable to large pretrained backbones and parameter efficient tuning?

Our approach. We introduce APEX, built on a key theoretical insight: the adversarial correction signal that GANs derive from an external discriminator can be generated endogenously within a flow model by separating real and fake scores in condition space. Concretely, APEX constructs a shifted condition cfake = Ac + b via an affine transformation and trains the model under cfake to fit trajectories toward its current one step outputs. This shifted condition branch provides an independent estimator of the fake distribution’s velocity field, enabling the main branch under the true condition c to receive an adversarial correction signal.

We also show that APEX admits a GAN aligned gradient interpretation. Under the Optimal Transport path, the score velocity duality connects velocity regression to score matching, allowing us to express APEX’s update in the same canonical score difference form as GANs. Crucially, while GANs weight the score difference using sample dependent discriminator terms such as D∗ or 1 − D∗, APEX corresponds to a constant weight with a target score induced by condition shifting. This yields stable, discriminator free signals while preserving an adversarial force toward photorealism.

#### Our main contributions are:

- a. Theoretical Foundation — GAN Aligned Gradient with Constant Weight: We establish a formal gradient level equivalence between APEX and GAN dynamics via score velocity duality ( Section 3.3 ), proving that APEX’s training gradient takes the canonical score difference form (sθ − smix) · ∂xt/∂θ with constant weight w ≡ 1 and an implicit score interpolation target smix = (1−λ)sdata + λsfake, connecting APEX to Fisher divergence minimization and explaining why it avoids the gradient instability of sample dependent discriminator weights.

- b. Methodology — Self Adversarial Framework via Condition Shifting: We propose APEX,

a discriminator free framework using an affine condition shift cfake = Ac + b to generate an endogenous adversarial signal for one step, high resolution text to image synthesis. This design makes APEX a plug and play replacement fully compatible with LoRA and other parameter efficient fine tuning pipelines.

- c. SOTA Performance and Scalability: Our 0.6B model surpasses FLUX-Schnell 12B in one step quality at NFE=1. With LoRA tuning on Qwen-Image 20B, APEX reaches GenEval 0.89 in 6 hours, surpassing the original 50 step teacher model (0.87).

- 2 PRELIMINARIES

Continuous Generative Models. Diffusion generative models (Ho et al., 2020; Song et al., 2020) and flow matching models (Lipman et al., 2022) both describe a continuous time evolution that transports a simple prior p(z) = N(0,I) toward a complex data distribution pdata(x). While classical diffusion is formulated as a stochastic forward noising process and a reverse time SDE, it admits an equivalent deterministic sampler given by the Probability Flow ODE (PF-ODE) associated with the same score field (Song et al., 2020). We define a time dependent random variable xt, t ∈ [0,1], as a linear interpolant between noise z and data x:

xt = α(t)z + γ(t)x. (1) Typically, we adopt the Optimal Transport (OT) path with α(t) = t, γ(t) = 1 − t, which satisfies the boundary conditions x1 = z for pure noise and x0 = x for pure data. This interpolation path induces a velocity field v(xt,t), defining the PF-ODE for sample generation:

dxt dt

= v(xt,t). (2)

Given an estimate of vt, we can numerically integrate Eq. (2) from t = 1 to t = 0 using standard ODE solvers (e.g., Euler (Karras et al., 2022)) to generate samples. For conditional generation with

condition c, flow matching trains a neural network Fθ(xt,t,c) to approximate a target velocity field. Along the OT path, conditional velocity of a particular pair (x,z) is defined as the time derivative:

d dt

(tz + (1 − t)x) = z − x. (3)

This quantity is an unbiased regression target; minimizing a squared error loss recovers the population optimal conditional mean v∗(xt,t). The standard FM loss is:

t,z, t ∥Fθ(xt,t,c) − (z − x)∥2 , (4)

LFM(θ) = Ex

where the expectation is taken over the joint distribution of (t,x,z), ensuring that Fθ recovers the vector field v∗ as the conditional expectation of the per sample velocity targets z − x given xt.

Score Velocity Duality. Under the OT path, the score function of any marginal density pt and its population optimal velocity field are related by (proof in Appendix B.2 ):

xt + (1 − t)v∗(xt,t) t

. (5)

st(xt) = −

Here v∗(xt,t) denotes the OT induced conditional velocity field. This Score Velocity Duality provides a bidirectional bridge between score functions and the velocity field parameterized by Fθ. We will apply it in Section 3.2 to convert the KL divergence gradient into velocity space, and in Section 3.3 to express APEX’s gradient in score space and connect it to GAN dynamics.

Few Step Generation. To overcome the inference latency caused by ODE numerical integration requiring tens of steps (NFE=50~250), a series of few step generation techniques have emerged (Song et al., 2023; Lu & Song, 2024; Frans et al., 2024; Geng et al., 2025).

- (i) Endpoint consistency methods like Consistency Models (CM) (Song et al., 2023) attempt to

directly learn the mapping from ODE trajectory to origin. A consistency function fθ(xt,t) is trained to satisfy the self consistency property: for any two points t,t′ on the same trajectory,

fθ(xt,t) = fθ(xt′,t′) = x0. This uses a first order Taylor expansion to approximate the trajectory integral.

- (ii) Higher order methods generalize this approach. RCGM (Sun & Lin, 2025) shows that CM and MeanFlow (Geng et al., 2025) are first order special cases (N = 1) of a more general framework. RCGM introduces N-th order recursive integral approximation, using future multi step trajectory information to more accurately estimate the current velocity field.
- (iii) Self adversarial methods. TwinFlow (Cheng et al., 2025) introduces twin trajectories by extending the time domain to t ∈ [−1,1]: the positive half maps noise to real data, while the negative half maps noise to the model’s current fake data. First, it trains the model on fake trajectories via:

t,z, t ∥Fθ(xfaket ,t) − (z − xfake)∥2 . (6)

LTF = Ex

Then minimizes the velocity discrepancy between the real score +t and the fake score −t via a rectification loss, steering generation toward higher fidelity without an external discriminator:

LTF-rect = Ext,z,t ∥Fθ(xt, t) − sg(Fθ(xt, −t) + ∆v)∥2 , (7)

where ∆v accounts for the gap between real and fake velocity targets. The two branches are separated by the sign of the time input t vs. −t; APEX achieves the same structure via a simpler separation in condition space c vs. cfake, as developed in Section 3 .

GAN Dynamics and Score Difference Gradients. GAN generator updates take the form of a score difference signal (sθ(x) − sdata(x)) modulated by a sample dependent weight from the discriminator; we review this structure, as APEX’s gradient admits the same form; see Section 3.3 . Let pθ(x), pdata(x), D(x) be the generator, data, and discriminator distributions, with s(x) := ∇x log p(x). In the analysis below, x denotes clean samples; in Section 3 we generalize to time marginal scores st(xt). Under the optimal discriminator D∗(x) = pdata(x)/pdata(x)+pθ(x) (Mohamed & Lakshminarayanan, 2016; Goodfellow et al., 2014), both GAN variants yield a generator gradient of the unified form:

∂x ∂θ

, (8)

∇θLGAN ∝ Ex∼p

w(x) · (sθ(x) − sdata(x)) ·

θ

where w(x) = D∗(x) or 1−D∗(x) for the saturating and non saturating variants respectively. This sample dependent weight encodes discriminator confidence: it vanishes when samples are highly realistic, causing gradient vanishing, and varies unpredictably across training, introducing instability. In Section 3.3 we show that APEX’s gradient takes exactly this score difference form but with a constant weight w ≡ 1, achieving adversarial level correction without a discriminator.

### 3 APEX

APEX achieves discriminator free, architecture preserving, self adversarial training by separating the real and fake scores in condition space rather than time space: an affine transformation cfake = Ac+b creates the fake score entirely within t ∈ [0,1], requiring no modification to time embeddings or model architecture. We develop the method in three stages:

- (i) Building the fake reference: define cfake and the fake sample xfake; train the shifted condition via Lfake so that vfake serves as an independent estimator of pfake’s velocity field.
- (ii) KL descent and practical loss: show that the velocity discrepancy ∆vAPEX is the exact descent direction on DKL(pfake∥preal); convert it into the consistency loss Lmix via endpoint equivalence.
- (iii) GAN aligned gradient structure: analyze the gradient in score space and show it is a GAN style score difference update with weight w ≡ 1, connecting to Fisher divergence minimization.

- 3.1 BUILDING THE ADVERSARIAL REFERENCE VIA CONDITION SHIFTING

Condition Space as the Separation Dimension. The two branch self adversarial structure requires a signal that distinguishes the real score from the fake score. TwinFlow uses the sign of the time input t vs. −t for this purpose; APEX instead uses the condition input c vs. cfake. Both achieve the same structure, but the condition space choice means the time domain, positional encodings, and time scheduling of any pretrained backbone remain completely unchanged, making APEX a plug and play replacement that is fully compatible with LoRA and other parameter efficient fine tuning pipelines without any adaptation of time embedding.

Condition Space Shifting and the Fake Sample. In particular, we use the OT interpolant in Eq. (1) with α(t) = t and γ(t) = 1 − t, so that x1 = z and x0 = x. We denote the conditional velocity field by vθ(xt,t,c), parameterized by a neural network Fθ(xt,t,c). We denote sg(·) as the stop gradient operator. Unless otherwise specified, all flows share the same interpolant family α(t),γ(t) and time weighting ω(t). We introduce a fake condition cfake, obtained through Self Condition Shifting of the original condition c:

cfake = Ac + b, (9) where A and b can be learnable parameter matrices/vectors or preset transformations.

Why affine shifting? The self adversarial design requires two properties of cfake: (i) it must be sufficiently distinct from c so that the network’s internal representations under the two conditions

decouple, allowing vfake to serve as an independent estimator of pfake’s velocity; and (ii) it must remain within the pretrained condition embedding space so that the network can produce semantically

coherent outputs. An affine map cfake = Ac + b is the most general linear class of transformations satisfying both: it preserves the algebraic structure of the embedding space while enabling strong representational decoupling when A reverses or attenuates the condition’s semantic direction. In particular, negative scaling A = −aI, a > 0 approximately inverts the condition embedding, creating a maximally contrastive branch that is consistent with our ablation finding that a ∈ {−1.0,−0.5} yields the most robust performance in Table 7 .

Self Adversarial Objective. APEX’s first stage trains the shifted condition branch to become an independent velocity estimator of the model’s current generation distribution pfake. We require the model to reconstruct its currently generated outputs when receiving the shifted condition cfake. Under the OT path, we define an endpoint predictor that maps a velocity estimate at (xt,t) to its implied clean sample:

##### fx(F,xt,t) := xt − t · F . (10)

Given a noisy sample xt at time t along the OT path in Eq. (1), the model’s implied clean data estimate under the real condition c is:

xfake = fx(Fθ(xt,t,c), xt, t) = xt − t · Fθ(xt,t,c). (11) When the model is imperfect, xfake deviates from the true x, capturing the model’s current generation error. We train the network under the shifted condition cfake to fit the trajectory toward xfake. Construct fake trajectory: xfaket = α(t)z + γ(t)xfake. The fake flow loss is defined as:

Lfake(θ) = Ext,z,t ∥Fθ(xfaket , t, cfake) − (z − xfake)∥2 . (12)

Concretely, ∂xfake/∂θ = −t · ∂Fθ(xt,t,c)/∂θ, so Lfake simultaneously trains the cfake branch and injects a direct adversarial gradient into Fθ(·,·,c). The stop gradient in APEX is applied separately in Lcons, where vfake := sg(Fθ(xt,t,cfake)) serves as a correction reference. When Lfake is minimized, vfake(·,·,cfake) approximates the velocity field of the fake distribution pfake. By training vfake on fake sample trajectories xfaket , we obtain an estimator of pfake’s velocity. Second, we show how this independence is exploited to construct a KL descent signal.

- 3.2 FROM VELOCITY DISCREPANCY TO KL DESCENT AND PRACTICAL LOSS

KL Gradient in Velocity Space. Let pfake(x|c) := pθ(x|c) denote the model’s current generation distribution and preal(x|c) := pdata(x|c) the true data distribution. Our ultimate goal is to close the gap between pfake and preal by minimizing KL divergence minθ DKL(pfake∥preal). The gradient of the KL divergence between pfake and preal admits a score difference form:

∂xt ∂θ

∇θDKL=Ext,z,t (∇xt log pfake(xt)−∇xt log preal(xt))·

. (13)

log pt(xt) is the score function of the marginal density pt at time t. We use the shorthand vdata(xt) := (z − x) for the supervised FM target velocity, and distinguish the two velocity fields by their gradient status:

##### Here, st(xt) := ∇xt

##### vfake(xt,t,cfake) := sg Fθ(xt,t,cfake) , vθ(xt,t,c) := Fθ(xt,t,c). (14)

By invoking the Score Velocity Duality defined in Eq. (5), we can analytically map the aforementioned velocity fields into the score space. This transformation yields the following induced score for both the original and fake signal:

xt + (1 − t)vθ(xt,t,c) t

xt + (1 − t)vfake(xt,t,cfake) t

. (15) Substituting into Eq. (13) (see Appendix B.3 ), the KL gradient in velocity space is:

sθ(xt) := −

, sfake(xt) := −

1 ω(t)

∇θDKL = −

∂xt ∂θ

Ex

t,z, t vθ(xt,t,c) − vdata(xt) ·

, (16)

Flux.1 [Dev] Qwen-Image

Flux.1 [schnell] SD3.5 Large SD3.5 Turbo

Qwen-Image-Lightning APEX (Ours)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

- NFE=1
- NFE=2

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

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

NFE=4

a tiny astronaut hatching from an egg on the moon

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

- NFE=1
- NFE=2

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

NFE=4

A paper origami dragon riding a boat in waves.

Figure 2: Qualitative Analysis between APEX and existing methods under different NFEs.

where ω(t) = 1−t t > 0. The apparent equivalence dissolves once we recognize that the derivation treats vθ itself as a proxy for the score of pfake, its descent signal degenerates into self regression. We replace this proxy with vfake the independent estimator of pfake’s velocity field constructed in

- Section 3.1 . Because vfake was trained on fake sample trajectories, it carries information about where pfake currently lies, providing a correction signal that goes beyond pure regression. Substituting vfake for the fake score proxy in Eq. (16), we define the APEX velocity correction signal:

∆vAPEX(xt) := vfake(xt,t,cfake) − vθ(xt,t,c). (17)

This difference measures the velocity discrepancy between vθ under c and vfake under cfake, evaluated at the same (xt,t). Because vfake is trained to track pfake, ∆vAPEX encodes the current deviation of the model’s generation from the data. We next construct a practical loss that combines this correction signal with data supervision, where the supervised component drives vθ → vdata and the fake correction component drives vθ → vfake; together they form an objective that steers pθ toward preal.

From Velocity Correction to Mixed Consistency Loss. ∆vAPEX(xt) is the KL descent direction: driving ∆vAPEX → 0 minimizes DKL(pfake∥preal). vfake is trained on fake trajectories but queried at real trajectory points xt; this deliberate asymmetry encodes pfake’s current structure at real trajectory locations, providing a correction signal that breaks the self referential loop. We convert the velocity objective to endpoint space: one can verify in Appendix B.4 that velocity matching and endpoint matching are exactly interchangeable:

fx(Fθ, xt, t) − x 22 = t2 Fθ − vdata(xt) 22, (18)

fx(Fθ,xt,t) − fx(vfake,xt,t) 22 = t2 Fθ − vfake(xt,t,cfake) 22 . (19) Thus matching velocities or matching their induced endpoints are exactly interchangeable up to the scalar factor t2. We therefore define two endpoint space objectives corresponding to the supervised FM branch and the fake branch, respectively:

Lsup(θ)=Ext,z,t ω( 1t) fx(Fθ, xt, t) − x 22 , (20)

Lcons(θ)=Ext,z,t ω( 1t) fx(Fθ, xt, t)−fx(vfake, xt, t) 22 , (21)

Table 1: System level comparison of efficiency and quality. Speeds are on a single A100 (BF16). Throughput is samples/s (batch=10); latency is seconds (batch=1). GenEval is the primary quality metric; FID/CLIP are reported for completeness. The best and second best entries are highlighted. † indicates methods requiring distinct models per NFE. Notation: Blue=full tuning; Red=LoRA; X.B=trainable params (B); r=LoRA rank.

Throughput Latency Params (samples/s) (s) (B) FID ↓ CLIP ↑ GenEval ↑

Methods NFEs

SDXL-LCM Luo et al. (2023) 2 2.89 0.40 0.9 18.11 27.51 0.44 PixArt-LCM Chen et al. (2024c) 2 3.52 0.31 0.6 10.33 27.24 0.42 SD3.5-Turbo Esser et al. (2024) 2 1.61 0.68 8.0 51.47 25.59 0.53 PCM Wang et al. (2024a)† 2 2.62 0.56 0.9 14.70 27.66 0.55 SDXL-DMD2 Yin et al. (2024a)† 2 2.89 0.40 0.9 7.61 28.87 0.58 FLUX-schnell (Labs, 2024) 2 0.92 1.15 12.0 7.75 28.25 0.71 Sana-Sprint (Chen et al., 2025b) 2 6.46 0.25 0.6 6.54 28.40 0.76 Sana-Sprint (Chen et al., 2025b) 2 5.68 0.24 1.6 6.50 28.45 0.77 Qwen-Image-Lightning (ModelTC, 2025) 2 3.15 0.48 20 (r=64,0.4) 6.76 28.37 0.85 RCGM (Sun & Lin, 2025) 2 3.15 0.48 20 (r=64,0.4) 6.80 28.63 0.82 TwinFlow (Cheng et al., 2025) 2 3.15 0.48 20 (r=64,0.4) 6.73 28.57 0.87

FewStepDistillationModels

APEX 2 6.50 0.25 0.6 6.75 28.33 0.84 APEX 2 5.72 0.23 1.6 6.42 28.24 0.85

APEX 2 3.21 0.49 20 (r=32,0.2) 6.72 28.71 0.87 APEX 2 3.17 0.47 20 (r=64,0.4) 6.51 28.42 0.89

APEX 2 3.30 0.45 20 6.44 28.51 0.90 SDXL-LCM Luo et al. (2023) 1 3.36 0.32 0.9 50.51 24.45 0.28 PixArt-LCM Chen et al. (2024c) 1 4.26 0.25 0.6 73.35 23.99 0.41 PixArt-DMD Chen et al. (2024b)† 1 4.26 0.25 0.6 9.59 26.98 0.45 SD3.5-Turbo Esser et al. (2024) 1 2.48 0.45 8.0 52.40 25.40 0.51 PCM Wang et al. (2024a)† 1 3.16 0.40 0.9 30.11 26.47 0.42 SDXL-DMD2 Yin et al. (2024a)† 1 3.36 0.32 0.9 7.10 28.93 0.59 FLUX-schnell (Labs, 2024) 1 1.58 0.68 12.0 7.26 28.49 0.69 Sana-Sprint (Chen et al., 2025b) 1 7.22 0.21 0.6 7.04 28.04 0.72 Sana-Sprint (Chen et al., 2025b) 1 6.71 0.21 1.6 7.69 28.27 0.76 Qwen-Image-Lightning (ModelTC, 2025) 1 3.29 0.40 20 (r=64,0.4) 7.06 28.35 0.85 RCGM (Sun & Lin, 2025) 1 3.29 0.40 20 (r=64,0.4) 11.38 27.69 0.52 TwinFlow (Cheng et al., 2025) 1 3.29 0.40 20 (r=64,0.4) 7.32 28.29 0.86 APEX 1 7.30 0.20 0.6 6.99 28.36 0.84 APEX 1 6.84 0.20 1.6 6.78 28.12 0.84 APEX 1 3.29 0.39 20 (r=32,0.2) 7.22 28.62 0.88 APEX 1 3.27 0.39 20 (r=64,0.4) 7.14 28.45 0.89 APEX 1 3.50 0.34 20 6.87 28.66 0.89

and combine them into the alternative loss:

GAPEX(θ) = (1 − λ)Lsup(θ) + λLcons(θ), λ ∈ [0,1]. (22) Here λ ∈ [0,1] controls the balance between data supervision and self adversarial correction: λ=0 recovers the standard FM objective, λ=1 yields purely adversarial consistency training, and intermediate values blend both signals. For later convenience we introduce the mixed endpoint target

Tmix(xt,t) := (1 − λ)x + λfx(vfake,xt,t), (23) where vfake := vfake(xt,t,cfake). Its score space counterpart the score interpolation smix defined in

Section 3.3 will reveal that Tmix corresponds to an implicit training target. The corresponding mixed consistency loss is:

Lmix(θ)=Ext,z,t ω( 1t) fx(Fθ, xt, t)−Tmix(xt, t) 22 . (24)

A direct gradient calculation with detailed steps in Appendix B.5 shows that for any θ we have ∇θLmix(θ) = ∇θGAPEX(θ), so optimizing the mixed endpoint regression in Eq. (24) is exactly equivalent, in parameter space, to following the KL inspired alternative loss in Eq. (22).

- 3.3 COMPLETE OBJECTIVE AND GAN GRADIENT STRUCTURE

Complete Training Objective. The full APEX objective combines the fake flow fitting Lfake with the mixed consistency loss Lmix:

LAPEX(θ) = λp Lfake(θ) + λe Lmix(θ), λp,λe ≥ 0. (25) Lfake is a prerequisite: it trains the shifted condition branch as an independent estimator of pfake’s velocity field so that vfake can serve as a valid correction reference. The KL descent interpretation of

- Section 3.2 applies to Lmix, which uses vfake to form the mixed target. We now analyze the gradient of Lmix in score space to reveal its formal connection to GAN dynamics.

GAN Aligned Gradient Structure. Via Score Velocity Duality Eq. (5), velocity differences translate to score differences by the time dependent factor −1−t t. Applying this to GAPEX, we define:

smix(xt) := (1−λ)sdata(xt) + λsfake(xt), (26) where sdata(xt) = ∇xt

log pdata,t(xt) and sfake(xt) = ∇xt

log pfake,t(xt). This yields:

Proposition (GAN-Aligned Gradient). The gradient of GAPEX takes the GAN canonical score difference form:

   1

   , (27)

∂xt ∂θ

∇θGAPEX(θ) ∝ Ext∼pθ,t

· (sθ(xt) − smix(xt)) ·

w≡1

with constant weight w ≡ 1, corresponding to minimizing the Fisher divergence DF(pθ∥pmix).

The Fisher divergence is:

##### DF(pθ∥pmix) := sθ(xt) − smix(xt) 22 pθ(xt)dxt . (28)

Here smix is a convex combination of score functions and need not correspond to a proper probability distribution; we interpret pmix as an implicit training target, analogous to the implicit distribution induced by score interpolation in classifier free guidance (Ho & Salimans, 2022). Eq. (27) reveals that APEX follows a GAN-aligned gradient with a constant weight w≡1: the time factor − 2t

3

1−t is absorbed into ω(t) and is uniform across all samples at each t.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

- • Backbones and tuning. We consider three capacities: APEX 0.6B and APEX 1.6B (full parameter tuning), and APEX 20B using LoRA on Qwen-Image (Wu et al., 2025a).
- • Datasets. Our training data comprises both open source and newly synthesized datasets. We utilize ShareGPT-4o (Chen et al., 2025c) and BLIP-3o (Chen et al., 2025a) as our part of open source resources. Additionally, we construct two synthetic datasets using the Qwen-Image-20B model. Part of the data includes 600K samples generated from prompts in the Flux-Reasoning-6M dataset (Fang et al., 2025), and another 200K samples synthesized from poster prompts.
- • Training and hardware. Training uses BF16 precision. For LoRA, we vary the rank r ∈ {32,64} and keep all other settings identical across ranks. We use 16×NVIDIA H800 80GB, 8×A100 80GB GPUs for training and evaluation.
- • Evaluation metrics. Our primary metric is GenEval Overall (Ghosh et al., 2023). We also report FID and CLIP on MJHQ-30K (Li et al., 2024a), DPGBench (Hu et al., 2024) and WISE (Niu et al.,

2025) for completeness. Unless noted, results are with NFE=1.

- 4.2 EFFICIENCY AND PERFORMANCE COMPARISON

We profile APEX under NFE=1/2 and contrast it with the strongest prior distilled models at each setting, summarized in Table 1 . GenEval Overall is our headline metric, with throughput and latency reported to highlight practical applicability.

At NFE=1, APEX 0.6B sustains 7.3 samples/s at 0.20s latency while achieving 0.84 GenEval a ≈ 0.15 absolute improvement over FLUX-Schnell 12B (GenEval 0.69), a model with 20× more parameters. This result suggests that the endogenous adversarial signal from condition shifting is more parameter efficient than scaling model capacity under standard distillation. Scaling to APEX 1.6B keeps latency flat with similar throughput. Our LoRA-tuned APEX 20B further lifts GenEval to 0.89 (r=64) at only 0.39s latency state of the art at NFE=1. Notably, this quality level is reached after only 6 hours of LoRA training (2K steps, global batch size 64), while the original Qwen-Image

###### Table 2: Quantitative Evaluation results on GenEval.

Attribute Object Object Binding Overall↑

Single Two

Model

Counting Colors Position

Show-o (Xie et al., 2024b) 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Emu3-Gen (Wang et al., 2024b) 0.98 0.71 0.34 0.81 0.17 0.21 0.54 PixArt-α (Chen et al., 2024d) 0.98 0.50 0.44 0.80 0.08 0.07 0.48 SD3 Medium (Esser et al., 2024) 0.98 0.74 0.63 0.67 0.34 0.36 0.62 FLUX.1 [Dev] (BlackForest, 2024) 0.98 0.81 0.74 0.79 0.22 0.45 0.66 SD3.5 Large (Esser et al., 2024) 0.98 0.89 0.73 0.83 0.34 0.47 0.71 JanusFlow (Ma et al., 2025) 0.97 0.59 0.45 0.83 0.53 0.42 0.63 Lumina-Image 2.0 (Qin et al., 2025) - 0.87 0.67 - - 0.62 0.73 Janus-Pro-7B (Chen et al., 2025d) 0.99 0.89 0.59 0.90 0.79 0.66 0.80 HiDream-I1-Full (Cai et al., 2025) 1.00 0.98 0.79 0.91 0.60 0.72 0.83 GPT Image 1 [High] (OpenAI, 2025) 0.99 0.92 0.85 0.92 0.75 0.61 0.84 Seedream 3.0 (Gao et al., 2025) 0.99 0.96 0.91 0.93 0.47 0.80 0.84 BAGEL (Deng et al., 2025) 0.98 0.95 0.84 0.95 0.78 0.77 0.88 Qwen-Image (Wu et al., 2025a) 0.99 0.92 0.89 0.88 0.76 0.77 0.87 Hyper-BAGEL (Lu et al., 2025) 0.97 0.86 0.75 0.90 0.67 0.62 0.80 Qwen-Image-Lightning (ModelTC, 2025) 0.99 0.89 0.85 0.87 0.75 0.76 0.85 TwinFlow (Cheng et al., 2025) (1-NFE) 1.00 0.91 0.84 0.90 0.75 0.74 0.86

- APEX 0.6B (1-NFE) 0.99 0.91 0.75 0.93 0.76 0.69 0.84

- APEX 1.6B (1-NFE) 0.99 0.91 0.75 0.93 0.76 0.68 0.84

APEX 20B (LoRA&r=32) (1-NFE) 0.99 0.95 0.85 0.90 0.79 0.78 0.88 APEX 20B (LoRA&r=64) (1-NFE) 0.99 0.94 0.88 0.90 0.85 0.78 0.89

APEX 20B (SFT) (1-NFE) 0.99 0.92 0.83 0.91 0.86 0.81 0.89

###### Table 3: Quantitative evaluation results on DPGBench.

###### Model Global Entity Attribute Relation Other Overall↑

SD v1.5 (Rombach et al., 2022) 74.63 74.23 75.39 73.49 67.81 63.18 PixArt-α (Chen et al., 2024d) 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next (Zhuo et al., 2024) 82.82 88.65 86.44 80.53 81.82 74.63 SDXL (Podell et al., 2023) 83.27 82.43 80.91 86.76 80.41 74.65 Hunyuan-DiT (Li et al., 2024b) 84.59 80.59 88.01 74.36 86.41 78.87 Janus (Wu et al., 2025b) 82.33 87.38 87.70 85.46 86.41 79.68 PixArt-Σ (Chen et al., 2024a) 86.89 82.89 88.94 86.59 87.68 80.54 Emu3-Gen (Wang et al., 2024b) 85.21 86.68 86.84 90.22 83.15 80.60 Janus-Pro-1B (Chen et al., 2025d) 87.58 88.63 88.17 88.98 88.30 82.63 DALL-E 3 (OpenAI, 2023) 90.97 89.61 88.39 90.58 89.83 83.50 FLUX.1 [Dev] (BlackForest, 2024) 74.35 90.00 88.96 90.87 88.33 83.84 SD3.5-Medium Esser et al. (2024) 84.08 87.90 91.01 88.83 80.70 88.68 SD3.5-Turbo Sauer et al. (2024b) 79.03 80.12 86.13 84.73 91.86 78.29 SD3.5-Large Esser et al. (2024) 83.21 84.27 88.99 87.35 93.28 80.35 FLUX.1-schnell Labs (2024) 84.94 86.62 90.82 88.35 93.45 82.00 Janus-Pro-7B (Chen et al., 2025d) 86.90 88.90 89.40 89.32 89.48 84.19 HiDream-I1-Full (Cai et al., 2025) 76.44 90.22 89.48 93.74 91.83 85.89 Lumina-Image 2.0 (Qin et al., 2025) - 91.97 90.20 94.85 - 87.20 Seedream 3.0 (Gao et al., 2025) 94.31 92.65 91.36 92.78 88.24 88.27 GPT Image 1 [High] (OpenAI, 2025) 88.89 88.94 89.84 92.63 90.96 85.15 Qwen-Image (Wu et al., 2025a) 91.32 91.56 92.02 94.31 92.73 88.32 Playground v3 (Liu et al., 2024) 87.04 91.94 85.71 90.90 90.00 92.72 TwinFlow (Cheng et al., 2025) (1-NFE) 92.34 92.12 92.45 92.86 92.63 86.52

- APEX 0.6B (1-NFE) 90.58 90.36 90.44 90.77 90.73 82.66

- APEX 1.6B (1-NFE) 90.77 90.56 90.63 90.98 90.94 83.22

APEX 20B (LoRA&r=32) (1-NFE) 93.12 90.95 91.38 90.65 91.73 86.17 APEX 20B (LoRA&r=64) (1-NFE) 92.46 91.14 90.71 91.30 91.98 85.77

APEX 20B (SFT) (1-NFE) 93.25 89.76 90.65 91.17 90.75 84.59

20B requires 50 integration steps to achieve GenEval 0.87. APEX thus simultaneously improves quality and reduces both training and inference cost.

Moving to NFE=2, APEX 1.6B rises to 0.85 GenEval, an ∼ 8-point margin over the strongest two-step baseline (Sana-Sprint 1.6B at 0.77) while running more than twice as fast. The 20B LoRA variant sustains 0.89 GenEval with a modest latency bump to 0.47s. Taken together, these results demonstrate that APEX closes the quality gap to multi-step generators without sacrificing the latency advantage that makes distilled models practical in production pipelines.

###### Table 4: Quantitative evaluation results on WISE.

###### Model Cultural Time Space Biology Physics Chemistry Overall↑

SD v1.5 (Rombach et al., 2022) 0.34 0.35 0.32 0.28 0.29 0.21 0.32 SDXL (Podell et al., 2023) 0.43 0.48 0.47 0.44 0.45 0.27 0.43 SD3.5-Large Esser et al. (2024) 0.44 0.50 0.58 0.44 0.52 0.31 0.46 PixArt-α (Chen et al., 2024d) 0.45 0.50 0.48 0.49 0.56 0.34 0.47 Playground-v2.5 (Li et al., 2024a) 0.49 0.58 0.55 0.43 0.48 0.33 0.49 FLUX.1 [Dev] (BlackForest, 2024) 0.48 0.58 0.62 0.42 0.51 0.35 0.50 Janus (Wu et al., 2025b) 0.16 0.26 0.35 0.28 0.30 0.14 0.23 VILA-U (Wu et al., 2024) 0.51 0.51 0.51 0.49 0.51 0.49 0.50 Show-o (Xie et al., 2024b) 0.95 0.52 0.49 0.82 0.11 0.28 0.53 Janus-Pro-7B (Chen et al., 2025d) 0.30 0.37 0.49 0.36 0.42 0.26 0.35 Emu3-Gen (Wang et al., 2024b) 0.34 0.45 0.48 0.41 0.45 0.47 0.39 MetaQuery-XL (Pan et al., 2025) 0.56 0.55 0.62 0.49 0.63 0.41 0.55 BAGEL (Deng et al., 2025) 0.44 0.55 0.68 0.44 0.60 0.39 0.52 GPT-4o 0.81 0.71 0.89 0.83 0.79 0.74 0.80 Qwen-Image (Wu et al., 2025a) - - - - - - 0.62 Qwen-Image-Lightning (ModelTC, 2025) - - - - - - 0.51 TwinFlow (Cheng et al., 2025) 0.52 0.51 0.67 0.48 0.61 0.40 0.54

APEX 20B (SFT) (1-NFE) 0.53 0.54 0.66 0.48 0.61 0.41 0.54

- Table 5: Effect of training data and steps on GenEval Overall (NFE=1). We compare ShareGPT-4o and BLIP-3o across training steps for APEX 0.6B/1.6B, and LoRA tuned Qwen-Image 20B with ranks r=32/r=64. All runs use global batch size 64.

Model

ShareGPT-4o Blip-3o

2Ksteps 8Ksteps 10Ksteps 2Ksteps 8Ksteps 10Ksteps

- APEX 0.6B 0.37 0.67 0.73 0.71 0.77 0.81

- APEX 1.6B 0.36 0.70 0.73 0.27 0.78 0.83 0.4Ksteps 1Ksteps 2Ksteps 0.4Ksteps 1Ksteps 2Ksteps

APEX 20B (r=32) 0.19 0.33 0.62 0.83 0.84 0.83 APEX 20B (r=64) 0.21 0.35 0.61 0.73 0.85 0.84

- Table 6: Ablation on the weights of Lfake vs. Lmix. We report GenEval Overall (NFE=1) for different weighting ratios (λp:λe). The dataset is BLIP-3o. Training steps are 8K for 0.6B/1.6B models and 0.4K for the 20B (LoRA) model. Best per model in bold.

Weighting Ratio (λp : λe) APEX 0.6B APEX 1.6B APEX 20B (r=32)

- 1.0 : 0.0 (Lfake Only) 0.32 0.35 0.42

- 0.0 : 1.0 (Lmix Only) 0.63 0.66 0.69

- 1.0 : 0.5 0.72 0.71 0.81

- 1.0 : 1.0 (Ours) 0.77 0.76 0.83

- 1.0 : 2.0 0.74 0.75 0.82

- 4.3 ABLATIONS

We present controlled ablations to isolate the effects of key design choices in APEX. Unless otherwise stated, all results are reported with NFE=1 and the GenEval Overall metric, using identical prompts, seeds, and resolution.

Balancing Lfake and Lmix. We dissect the contribution of the fake flow fitting objective Lfake (Eq. (12)) and the mixed consistency objective Lmix (Eq. (24)) by ablating their outer relative weights λp:λe in LAPEX = λp Lfake + λe Lmix on three models: APEX 0.6B, 1.6B, and 20B (LoRA). Here λp,λe ≥ 0 are the outer loss weights (distinct from the inner mixing ratio λ ∈ [0,1] in Eq. (22)); the default setting Eq. (25) corresponds to λp=λe=1. As shown in Table 6 , either component alone underperforms the balanced settings. A mild endpoint emphasis (e.g., 1.0:0.5) or equal weighting (1.0:1.0) yields the highest GenEval, whereas excessive endpoint emphasis (1.0:2.0) slightly harms path integrability and overall score. This validates our design: the fake flow fitting Lfake is necessary to retain one step stability, whereas Lmix is critical to reach high fidelity endpoints.

- • Condition shifting hyperparameters a and b. To probe the self conditioned contrast, we vary the

scale a and bias b in cfake = Ac + b (setting A = aI and b = b1, i.e. scalar multiples of the identity and all ones vector) and report GenEval on a (a,b) grid in Table 7 . Results show a broad optimum around a ∈ {−1.0,−0.5} with small positive biases (b ∈ [0.1,1.0]), consistent with the principled justification in Section 3.1 : negative scaling inverts the condition embedding direction, creating maximal representational contrast between the real and shifted branches, which enables vfake

to function as a more independent estimator of pfake’s velocity. Positive scaling (a=0.5) is generally suboptimal unless paired with a larger bias (b=10.0) to compensate for the reduced decoupling.

- • Datasets vs. training steps. We first study data and compute scaling by varying one factor at a time. The dataset ablation Table 5 compares ShareGPT-4o and BLIP-3o across fixed steps, evaluated on APEX 0.6B and 1.6B, and extends to Qwen-Image 20B (LoRA) at shorter step budgets. BLIP-3o consistently yields higher GenEval at larger step counts for both 0.6B and 1.6B (e.g., 0.81/0.83 vs 0.73 at 10K). For the 20B LoRA model, BLIP-3o reaches 0.84–0.85 by 1–2K steps, whereas ShareGPT-4o improves steadily with more steps (0.19→0.62).

Table 7: Effect of condition-shifting hyperparameters on GenEval Overall (NFE=1). Moderate negative scaling (a ∈ {−1.0, −0.5}) yields the most robust gains.

a \ b 0.0 0.1 1.0 10.0 −1.0 0.76 0.73 0.74 0.74 −0.5 0.75 0.79 0.81 0.70

0.5 0.29 0.37 0.30 0.73

- 5 CONCLUSION

We presented APEX, a discriminator free one step generative framework built on self condition shifting. APEX introduces a fake condition cfake = Ac + b and uses the model itself to generate a fake signal under cfake, replacing the need for an external discriminator or a frozen teacher network. The fake flow fitting loss Lfake (Eq. (12)) trains the fake condition branch to track the model’s current generation so that vfake serves as an independent estimator of pfake’s velocity. The mixed consistency loss Lmix then uses vfake as a correction reference, with the supervised component driving vθ → vdata and the fake correction component providing an adaptive signal that evolves as pθ improves. We showed that the resulting gradient takes the same score difference form as GAN objectives but with a constant weight w ≡ 1, connecting APEX to Fisher divergence minimization without sample dependent discriminator terms. APEX attains state of the art one step quality with low latency. At NFE=1, the 0.6B/1.6B models reach 0.84 GenEval at 0.20s latency (7.3/6.84 samples/s), and the 20B LoRA variant achieves 0.89 GenEval at 0.39s latency. At NFE=2, the 20B LoRA model sustains 0.89 GenEval at 0.47s latency. These results confirm that endogenous adversarial training via condition shifting closes the quality gap to multi-step generators while preserving the throughput advantage of one step synthesis.

REFERENCES

BlackForest. Flux. https://github.com/black-forest-labs/flux, 2024. Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng

Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, et al. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset. arXiv preprint arXiv:2505.09568, 2025a.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-Σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024a.

Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. arXiv preprint arXiv:2403.04692, 2024b.

Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li. Pixart-{\delta}: Fast and controllable image generation with latent consistency models. arXiv preprint arXiv:2401.05252, 2024c.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Zhongdao Wang, James T Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024d.

Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, and Song Han. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025b.

Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation. arXiv preprint arXiv:2506.18095, 2025c.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.

Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows. arXiv preprint arXiv:2512.05150, 2025.

Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In Advances in Neural Information Processing Systems, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A million-scale textto-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.

Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36: 52132–52152, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, 2014.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022.

Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition, 2024.

Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, and Il-Chul Moon. Refining generative process with discriminator guidance in score-based diffusion models. In International Conference on Machine Learning, pp. 16567–16598. PMLR, 2023.

Black Forest Labs. Flux, 2024. URL https://blackforestlabs.ai/. Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.

5: Three insights towards enhancing aesthetic quality in text-to-image generation. arXiv preprint arXiv:2402.17245, 2024a.

Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, et al. Hunyuan-dit: A powerful multi-resolution diffusion transformer with fine-grained chinese understanding. arXiv preprint arXiv:2405.08748, 2024b.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024.

Deyuan Liu, Peng Sun, Xufeng Li, and Tao Lin. Efficient generative model training via embedded representation warmup. arXiv preprint arXiv:2504.10188, 2025.

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

Yanzuo Lu, Xin Xia, Manlin Zhang, Huafeng Kuang, Jianbin Zheng, Yuxi Ren, and Xuefeng Xiao. Hyper-bagel: A unified acceleration framework for multimodal understanding and generation, 2025.

Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023.

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pp. 23–40. Springer, 2024.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7739–7751, 2025.

ModelTC. Qwen-image-lightning. GitHub-ModelTC/Qwen-Image-Lightning: Qwen-Image-Lightning:SpeedupQwen-Imagemodelwithdistilla, 2025.

Shakir Mohamed and Balaji Lakshminarayanan. Learning in implicit generative models. arXiv preprint arXiv:1610.03483, 2016.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, 2021.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

OpenAI. Dalle-3, 2023. URL https://openai.com/dall-e-3. OpenAI. Gpt-image-1, 2025. URL https://openai.com/index/

introducing-4o-image-generation/.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Jiakang Yuan, Xinyue Li, Dongyang Liu, et al. Lumina-image 2.0: A unified and efficient image generative framework. arXiv preprint arXiv:2503.21758, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. arXiv preprint arXiv:2403.12015, 2024a.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–11, 2024b.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023. Peng Sun and Tao Lin. Any-step generation via n-th order recursive consistent velocity field

estimation, 2025. URL https://github.com/LINs-lab/RCGM. GitHub repository. Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025.

Fu-Yun Wang, Zhaoyang Huang, Alexander William Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency model. arXiv preprint arXiv:2405.18407, 2024a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint arXiv:2509.04394, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 12966–12977, 2025b.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629, 2024a.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024b.

Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. arXiv:2405.14867, 2024a.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024b.

Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

Kaiwen Zheng, Yongxin Chen, Huayu Chen, Guande He, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Direct discriminative optimization: Your likelihood-based visual generative model is secretly a gan discriminator. arXiv preprint arXiv:2503.01103, 2025.

Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Xiangyang Zhu, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37:131278–131315, 2024.

CONTENTS

- 1 Introduction 2
- 2 Preliminaries 3
- 3 APEX 4

- 3.1 Building the Adversarial Reference via Condition Shifting . . . . . . . . . . . . . 4
- 3.2 From Velocity Discrepancy to KL Descent and Practical Loss . . . . . . . . . . . . 5
- 3.3 Complete Objective and GAN Gradient Structure . . . . . . . . . . . . . . . . . . 7

- 4 Experiments 8

- 4.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.2 Efficiency and Performance Comparison . . . . . . . . . . . . . . . . . . . . . . . 8
- 4.3 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- 5 Conclusion 11

- A Related Work 16

- A.1 From Macro level to Local Control . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 From External Discriminators to Self Adversarial Conditioning . . . . . . . . . . . 17
- A.3 Scalable Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- B Theoretical Analysis and Proofs 18

- B.1 Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2 Score–Velocity Duality under OT Path . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.3 KL Gradient in Velocity Space . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- B.4 Endpoint–Velocity Equivalence . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- B.5 Gradient Equivalence of Alternative Loss . . . . . . . . . . . . . . . . . . . . . . 23
- B.6 Fisher Divergence Perspective . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25

- C Visualizations Part I 27
- D Visualizations Part II 28
- E Visualizations Part III 28

- A RELATED WORK

- A.1 FROM MACRO LEVEL TO LOCAL CONTROL

The foundational paradigm in continuous generative modeling, including diffusion (Ho et al., 2020; Song et al., 2020; Karras et al., 2022) and flow matching (Lipman et al., 2022; Liu et al., 2025), involves learning an instantaneous velocity field. While effective for multi step integration, this first order approach is brittle under coarse discretization, as high path curvature causes truncation errors that degrade few step generation quality (Karras et al., 2022). To address this, a significant body of work has shifted focus from instantaneous dynamics to supervising the model’s behavior over a time interval. These methods attempt to ensure path integrability at a macro level. For instance, Consistency Models (CMs) (Song et al., 2023; Lu & Song, 2024) enforce a relative constraint, requiring that endpoint predictions remain consistent across different points on the same trajectory. While effective, this does not directly address the geometric properties of the path that cause discretization errors. More recent approaches such as MeanFlow (Geng et al., 2025) and Transition Models (TiM) (Wang et al., 2025) go a step further by directly modeling the average velocity or state transition over an interval. They learn the result of a large step, but the constraint remains on the interval’s endpoints rather than its internal geometry. UCGM (Sun et al., 2025) unifies different paradigms by interpolating between their respective training objectives with a hyperparameter. APEX takes a different approach. Rather than enforcing consistency constraints between trajectory endpoints, the fake flow fitting loss Lfake (Eq. 12) trains the shifted condition branch to track the model’s current generation errors, providing an adaptive self adversarial signal without requiring an external network.

This internal adversarial signal, combined with data supervision in Lmix, drives pθ toward preal in a self contained, architecture preserving manner.

- A.2 FROM EXTERNAL DISCRIMINATORS TO SELF ADVERSARIAL CONDITIONING

Achieving high one step fidelity requires strong, absolute anchoring of the endpoint prediction to the data manifold, a property that relative consistency constraints alone do not guarantee. A primary approach involves incorporating external adversarial signals. Distillation methods like DMD/DMD2 (Yin et al., 2024a) and other GAN based refiners (Kim et al., 2023; Sauer et al., 2024a; Zheng et al., 2025) use an auxiliary discriminator to sharpen outputs, even allowing a student to surpass its teacher. However, this reliance is a double edged sword: it introduces training instability, computational overhead, and, critically, often depends on a costly precomputed dataset for regularization. For large scale models, generating this dataset of teacher student pairs can be prohibitively expensive, exceeding the cost of training itself (Yin et al., 2024a). A distinct line of work generates adversarial signals internally. Direct Discriminative Optimization (DDO) (Zheng et al., 2025) reparameterizes the GAN discriminator using the likelihood ratio between a target model and a fixed reference, operating in probability space. TwinFlow (Cheng et al., 2025) constructs a self adversarial signal by extending the time domain to t ∈ [−1,1], but requires modifying time embeddings and positional encodings, limiting compatibility with pretrained backbones and parameter efficient tuning. APEX advances this line by replacing external discriminators with an endogenous adversarial signal derived from condition shifting. The shifted condition branch vfake is trained on fake sample trajectories using the same network weights — requiring no modification to time embeddings or model architecture — eliminating both discriminator overhead and precomputed teacher datasets while retaining the adversarial correction signal that drives pθ toward preal. We further prove that this yields a gradient identical in structure to the GAN update but with constant weight w ≡ 1, corresponding to Fisher divergence minimization (see main paper, Section 3.3).

- A.3 SCALABLE TRAINING

The practical implementation of generative models, including APEX, hinges on scalable system design. A key challenge is the need to compute time derivatives to enforce interval consistency. Methods like MeanFlow (Geng et al., 2025) relied on Jacobian-Vector Products (JVP), creating a significant scalability bottleneck. JVP is computationally intensive and, more importantly, incompatible with critical training optimizations like FlashAttention (Dao, 2023) and FSDP based distributed training (Zhao et al., 2023), limiting its use in billion parameter models. To overcome this, the field has converged on finite difference estimators, often termed Differential Derivation Equations (DDE), as a scalable alternative (Lu & Song, 2024; Wang et al., 2025). These estimators rely only on forward passes and are natively compatible with modern training infrastructure. APEX’s path integrability objective fully embraces this scalable approach. This design choice, combined with our efficient endogenous adversarial mechanism and established best practices for large scale training—ensures that APEX maintains 1-NFE fidelity and any-step scaling on large backbones like SDXL, SANA, and Qwen-Image (Podell et al., 2023; Xie et al., 2024a; Wu et al., 2025a), while remaining fully compatible with parameter efficient tuning.

- B THEORETICAL ANALYSIS AND PROOFS

We first establish notation and basic assumptions, then prove the Score–Velocity Duality under the Optimal Transport path, the exact equivalence between endpoint space and velocity space objectives, the gradient equivalence between the mixed consistency loss and the alternative loss, and finally interpret APEX’s alternative loss through the lens of Fisher divergence.

- B.1 SETUP

We use bold lowercase letters for vectors like x,z,v and bold uppercase letters for matrices and operators like F. The identity matrix is denoted by I, and 0 represents the zero vector. Let pdata(x) denote the data distribution over x ∈ Rd, and let p(z) = N(0,I) be the standard Gaussian prior over z ∈ Rd. For conditional generation, we write pdata(x|c) where c is a conditioning variable like text prompt. Throughout this appendix, we work with the Optimal Transport (OT) interpolation path defined by:

xt = α(t)z + γ(t)x, t ∈ [0,1], (29)

where α(t) = t and γ(t) = 1 − t. This satisfies the boundary conditions: x0 = x (pure data) and x1 = z (pure noise). Given a time dependent random variable xt following Eq. (29), we define the conditional mean velocity. Throughout the theory section, v(xt,t) refers to the conditional mean velocity induced by the OT noising construction, i.e.,

v(xt,t) := Ez−x|x

t

[z − x]. (30) The score function is st(xt) := ∇xt

log pt(xt), where pt(xt) is the marginal density of xt at time t. The target velocity under the OT path is vdata(xt) = z − x. We parameterize a velocity field estimator by a neural network Fθ : Rd × [0,1] × C → Rd, where θ denotes the model parameters and C is the conditioning space. We use the shorthand Fθ(xt,t,c) ≡ Fθ when the arguments are clear from context. The operator sg(·) denotes stop gradient, meaning gradients do not flow through the argument. The fake velocity vfake is evaluated by querying the same online network Fθ under the shifted condition cfake with stop gradient applied (in Lcons), so no separate teacher parameters are maintained. We define the endpoint predictor that maps a velocity estimate to its implied clean sample:

fx(F,xt,t) := xt − t · F. (31) This is motivated by the OT path: if xt = tz + (1 − t)x and F ≈ z − x, then fx(F,xt,t) ≈ x.

- B.2 SCORE–VELOCITY DUALITY UNDER OT PATH

We establish the fundamental relationship between the score function and the optimal velocity field under the OT path.

|Proposition 1 (Score–Velocity Duality) . Let xt = tz + (1 − t)x for t ∈ (0,1), where z ∼ N(0,I) and x ∼ pdata(x). Denote by pt(xt) the marginal density of xt, and define the OT induced conditional mean (least squares optimal) velocity field<br><br>v∗(xt,t) := Ez−x|x<br><br>t<br><br>[z − x]. (32) Then the score function st(xt) := ∇xt<br><br>log pt(xt) satisfies<br><br>st(xt) = −<br><br>xt + (1 − t)v∗(xt,t) t<br><br>. (33)|
|---|

Proof. Step 1: Rewrite as an additive Gaussian observation model. Define x′ := (1 − t)x. Then the OT path can be written as

xt = x′ + tz, z ∼ N(0,I).

Conditioned on x′, the likelihood is xt | x′ ∼ N(x′,t2I), since xt − x′ = tz and z ∼ N(0,I) implies tz ∼ N(0,t2I).

#### Step 2: Apply Tweedie’s formula to recover the posterior mean. For an additive Gaussian model

xt = x′ + tz where z ∼ N(0,I), Tweedie’s formula states that the posterior mean can be recovered from the score function:

Ex′|xt [x′] = xt + t2 ∇xt

log pt(xt) = xt + t2 st(xt). (34)

Justification of Tweedie’s formula: For a Gaussian perturbation model y = x′ +σϵ with ϵ ∼ N(0,I), we have

Ex′|y [x′] = y + σ2∇y log p(y). In our case, y = xt, x′ = (1 − t)x, and σ = t, so Eq. (34) follows directly. Since x′ = (1 − t)x, we can recover the conditional expectation of x:

1 1 − t

Ex′|xt [x′]

Ex|x

[x] =

t

1 1 − t

xt + t2 st(xt)

=

xt + t2 st(xt) 1 − t

. (35)

=

#### Step 3: Express the conditional mean of z. From the OT path xt = tz + (1 − t)x, we can solve for z:

xt − (1 − t)x t

. Taking conditional expectations on both sides given xt:

z =

xt − (1 − t)x t

[z] = E

Ez|x

xt

t

1 t

xt − (1 − t)Ex|x

[x]

=

t

xt + t2 st(xt) 1 − t

1 t

(substituting Eq. (35))

xt − (1 − t) ·

=

1 t

xt − xt + t2 st(xt) (simplifying the fraction)

=

1 t −t2 st(xt)

=

= −tst(xt). (36)

###### Step 4: Form the optimal velocity and rearrange. By definition, the (least squares) optimal velocity field along the OT path is the conditional expectation of the target velocity z − x:

v∗(xt,t) := Ez−x|x

[z − x] = Ez|x

[z] − Ex|x

[x]. Substituting Eq. (35) and Eq. (36):

t

t

t

xt + t2 st(xt) 1 − t

v∗(xt,t) = −tst(xt) −

= −t(1 − t)st(xt) − (xt + t2 st(xt)) 1 − t

(common denominator)

= −tst(xt) + t2 st(xt) − xt − t2 st(xt) 1 − t

= −tst(xt) − xt 1 − t

xt + tst(xt) 1 − t

. (37)

= −

#### Step 5: Rearrange to obtain the score-velocity duality. Multiplying both sides of Eq. (37) by (1 − t):

(1 − t)v∗(xt,t) = −xt − tst(xt).

Rearranging:

xt + (1 − t)v∗(xt,t) = −tst(xt), which, upon dividing both sides by −t, gives exactly Eq. (33):

xt + (1 − t)v∗(xt,t) t

st(xt) = −

.

|Corollary 1 (Velocity Difference as Score Difference) . For any two OT noising constructions that induce marginals p1,t,p2,t and corresponding conditional mean velocities vi(xt,t) := Ez−x|x<br><br>t<br><br>[z − x] (i ∈ {1,2}) at the same (xt,t), their velocity difference and score difference satisfy<br><br>v1(xt,t) − v2(xt,t) = −<br><br>t 1 − t<br><br>[s1(xt) − s2(xt)]. (38)|
|---|

Proof. Applying Proposition 1 to both velocity fields:

- s1(xt) = −

xt + (1 − t)v1(xt,t) t

, (39)

- s2(xt) = −

xt + (1 − t)v2(xt,t) t

. (40) Subtracting Eq. (40) from Eq. (39):

1 t

s1(xt) − s2(xt) = −

[(xt + (1 − t)v1) − (xt + (1 − t)v2)]

1 − t t

[v1(xt,t) − v2(xt,t)]. (41) Rearranging yields Eq. (38).

= −

| |
|---|

- B.3 KL GRADIENT IN VELOCITY SPACE

We now show how the KL divergence gradient between two flow-induced distributions can be expressed purely in terms of their velocity fields. This result is fundamental to understanding how APEX’s training objective connects to distribution matching.

|Lemma 1 (Gradient of KL Divergence via Reparameterization) . Let pθ(x) be a probability density parameterized by θ, defined by the push-forward of a fixed base distribution p(z) through a<br><br>differentiable mapping x = Tθ(z) (the reparameterization trick). Let q(x) be a target distribution independent of θ. The gradient of the KL divergence DKL(pθ∥q) with respect to θ satisfies:<br><br>∇θDKL(pθ∥q) = Ez∼p(z)(sθ(Tθ(z)) − sq(Tθ(z))) · ∇θTθ(z). (42)<br><br>where sθ(x) = ∇x log pθ(x) and sq(x) = ∇x log q(x) are the score functions of the model and target distributions, respectively.|
|---|

Proof. Consider the KL divergence defined as an expectation over the reparameterized variable z:

L(θ) = DKL(pθ∥q) = Ez ∼ p(z)log pθ(Tθ(z)) − log q(Tθ(z)). (43)

Since the base distribution p(z) does not depend on θ, we can move the gradient operator ∇θ inside the expectation. Applying the total derivative (chain rule) to the terms inside the expectation yields:

∇θL(θ) = Ez ∇θ log pθ(x) x=T

θ(z) − ∇θ log q(x) x=T

θ(z)

∂x ∂θ − ∇x log q(x) ·

∂x ∂θ

. (44)

= Ez ∇θ log pθ(x)

+ ∇x log pθ(x) ·

fixed x

Note that the first term corresponds to the standard score function estimator identity, which vanishes under expectation:

Ex∼p

θ∇θ log pθ(x)

##### = ∇θpθ(x)dx = ∇θ pθ(x)dx = ∇θ(1) = 0. (45)

fixed x

Substituting the definitions of the score functions sθ = ∇x log pθ and sq = ∇x log q into Eq. (44), and removing the zero-mean term, we obtain:

∂x ∂θ − sq(x) ·

∂x ∂θ

∇θL(θ) = Ezsθ(x) ·

= Ex∼p

(sθ(x) − sq(x)) ·

θ

∂x ∂θ

. (46)

|Proposition 2 (KL Gradient via Velocity Difference) . Let pfake(x|c) be the distribution induced by a flow with velocity field vθ(xt,t,c), and preal(x|c) the data distribution with velocity vdata(xt) = z − x under the OT path. Then the gradient of the KL divergence with respect to model parameters θ satisfies<br><br>∇θDKL(pfake∥preal) = −<br><br>1 ω(t)<br><br>Ex<br><br>t,z,t vθ(xt,t,c) − vdata(xt) ·<br><br>∂xt ∂θ<br><br>, (47)<br><br>where ω(t) = 1−t t > 0 is a positive time dependent weight. Since ω(t) > 0, this gradient drives vθ → vdata under gradient descent, confirming that minimizing DKL is equivalent to regressing vθ toward the real data velocity.<br><br>|
|---|

Proof. We derive the gradient by directly applying Lem. 1. Let the model distribution be pfake (parameterized by θ) and the target distribution be preal. By identifying the reparameterization mapping as the flow trajectory xt, Lem. 1 implies that the gradient of the KL divergence is the expectation of the dot product between the score difference and the path gradient:

∂xt ∂θ

∇θDKL(pfake∥preal) = Ex

t∼pfake (sfake(xt) − sreal(xt)) ·

, (48)

where sfake(xt) = ∇xt

log pfake(xt) and sreal(xt) = ∇xt

log preal(xt).

Next, we invoke the duality between score and velocity fields for Optimal Transport paths (Cor. 1). The difference between the model score and the target score is proportional to the difference between their respective velocity fields:

1 − t t

vθ(xt,t,c) − vdata(xt) . (49) Substituting Eq. (49) into Eq. (48), we obtain:

sfake(xt) − sreal(xt) = −

1 − t t

∇θDKL(pfake∥preal) = Ex

t,z,t −

∂xt ∂θ

vθ(xt,t,c) − vdata(xt) ·

. (50)

Defining ω(t) := 1−t t > 0, we identify −1−t t = −ω1(t), giving exactly:

1 ω(t)

∂xt ∂θ

, (51) which establishes Eq. (47).

Ex

∇θDKL(pfake∥preal) = −

t,z,t vθ(xt,t,c) − vdata(xt) ·

| |
|---|

- B.4 ENDPOINT–VELOCITY EQUIVALENCE

We prove that the endpoint space MSE and velocity space MSE are exactly equivalent up to a scalar factor t2. This result establishes that training objectives formulated in either space are mathematically interchangeable.

Proposition 3 (Endpoint–Velocity Equivalence for Supervised FM) . Let fx(F,xt,t) := xt − tF be the endpoint predictor defined in Eq. 10, and let vdata(xt) = z − x be the target velocity under the OT path. Then for any velocity estimate Fθ, we have

fx(Fθ,xt,t) − x 22 = t2 Fθ − vdata(xt) 22. (52)

Proof. Step 1: Expand the endpoint predictor. By definition of the endpoint predictor in Eq. (31), we have

fx(Fθ,xt,t) = xt − tFθ. (53)

- Step 2: Compute the squared error. The LHS of Eq. (52) is fx(Fθ,xt,t) − x 22 = (xt − tFθ) − x 22

= (xt − x) − tFθ 22. (54)

- Step 3: Use the OT path identity. Under the OT path xt = tz+(1−t)x from Eq. (29), we compute the difference xt − x step by step:

xt − x = tz + (1 − t)x − x (substituting the OT path)

= tz + (1 − t)x − x

= tz + (1 − t)x − 1 · x (writing x = 1 · x)

= tz + x − tx − x (expanding (1 − t)x)

= tz − tx (canceling x)

= t(z − x). (factoring out t) (55)

Recall that under the OT path, the target velocity is defined as vdata(xt) := z − x, which is the instantaneous rate of change from data x to noise z. Therefore, we obtain the key identity:

xt − x = tvdata(xt). (56)

This identity says that the displacement from the clean data x to the noised sample xt is exactly t times the target velocity, which makes intuitive sense since we’ve traveled for "time" t along the trajectory.

- Step 4: Substitute and simplify. Substituting Eq. (56) into Eq. (54): (xt − x) − tFθ 22 = tvdata(xt) − tFθ 22 (using Eq. (56))

= t(vdata(xt) − Fθ) 22 (factoring out t)

= t2 vdata(xt) − Fθ 22, (using ∥cv∥22 = c2∥v∥22) (57) which proves Eq. (52). The final step uses the homogeneity property of the squared ℓ2 norm.

Geometric interpretation: This result shows that predicting the clean endpoint x is equivalent to predicting the velocity z − x, scaled by the time factor t. When t is small (near clean data), the endpoint prediction is very sensitive to velocity errors. When t is large (near pure noise), the endpoint prediction is less sensitive, which motivates using time dependent weighting ω(t) in the loss.

|Proposition 4 (Endpoint–Velocity Equivalence for Fake Alignment) . For the fake alignment term, let vfake(xt,t,cfake) := sg(Fθ(xt,t,cfake)) be the fake velocity field obtained by querying the same online network Fθ under the shifted condition cfake with stop gradient applied. Then<br><br>fx(Fθ,xt,t) − fx(vfake,xt,t) 22 = t2 Fθ − vfake(xt,t,cfake) 22. (58)|
|---|

#### Proof. Step 1: Expand both endpoint predictors. By definition,

fx(Fθ,xt,t) = xt − tFθ, (59) fx(vfake,xt,t) = xt − tvfake(xt,t,cfake). (60)

- Step 2: Compute the difference. fx(Fθ,xt,t) − fx(vfake,xt,t) = xt − tFθ − xt − tvfake

= xt − tFθ − xt + tvfake

= tvfake − tFθ

= t(vfake − Fθ). (61)

#### Step 3: Square the norm.

fx(Fθ,xt,t) − fx(vfake,xt,t) 22 = t(vfake − Fθ) 22

= t2 vfake − Fθ 22

= t2 Fθ − vfake(xt,t,cfake) 22, (62) which proves Eq. (58).

| |
|---|

- B.5 GRADIENT EQUIVALENCE OF ALTERNATIVE LOSS

We now prove the key theoretical result: the gradient of the mixed consistency loss Lmix is exactly equal to the gradient of the alternative loss GAPEX. This establishes that these two seemingly different objectives induce identical training dynamics in parameter space.

|Theorem 1 (Gradient Equivalence) . Let Lmix(θ) and GAPEX(θ) be defined as in Eq. 24 and Eq. 22, respectively. Then for any parameter θ,<br><br>∇θ Lmix(θ) = ∇θ GAPEX(θ). (63)|
|---|

Proof. For notational simplicity, we focus on a single sample and omit the expectation Ex

t,z,t [·] and the weighting ω1(t) (these are linear operations that commute with gradients). We use the shorthand

- Fθ ≡ Fθ(xt,t,c) and vfake ≡ vfake(xt,t,cfake).

- Part A: Gradient of the mixed consistency loss. Step A1: Write the mixed consistency loss. From Eq. 24, the mixed consistency loss is

Lmix(θ) = fx(Fθ,xt,t) − Tmix(xt,t) 22, (64) where the mixed target is defined in Eq. 23 as

Tmix(xt,t) = (1 − λ)x + λfx(vfake,xt,t). (65)

- Step A2: Expand the endpoint predictors. Using the definition fx(F,xt,t) = xt − tF from Eq. (31):

fx(Fθ,xt,t) = xt − tFθ, (66) fx(vfake,xt,t) = xt − tvfake. (67)

- Step A3: Substitute into the mixed target. Substituting Eq. (67) into Eq. (65):

Tmix(xt,t) = (1 − λ)x + λ(xt − tvfake)

= (1 − λ)x + λxt − λtvfake. (68)

- Step A4: Compute the error term ∆. Define the error as

∆ := fx(Fθ,xt,t) − Tmix(xt,t). (69) Substituting Eq. (66) and Eq. (68):

∆ = (xt − tFθ) − (1 − λ)x + λxt − λtvfake = xt − tFθ − (1 − λ)x − λxt + λtvfake = xt(1 − λ) − (1 − λ)x − tFθ + λtvfake = (1 − λ)(xt − x) − tFθ + λtvfake. (70)

- Step A5: Apply the OT path identity. From Eq. (56) (proven in Section B.4), we have

xt − x = tvdata, where vdata = z − x. (71) Substituting into Eq. (70):

∆ = (1 − λ)tvdata − tFθ + λtvfake

= t (1 − λ)vdata + λvfake − Fθ . (72)

#### Step A6: Compute the gradient using the chain rule. The gradient of the squared norm Lmix = ∥∆∥22 with respect to θ is

∇θ Lmix(θ) = 2⟨∆,∇θ∆⟩, (73) where ⟨·,·⟩ denotes the inner product. This follows from the chain rule for the squared norm:

∇θ∥∆(θ)∥22 = ∇θ⟨∆,∆⟩ = 2⟨∆,∇θ∆⟩.

Since ∆ depends on θ only through Fθ (note that vdata = z − x does not depend on θ, and vfake = sg(Fθ(xt,t,cfake)) has stop gradient applied, so gradients do not flow through vfake), we have

∇θ∆ = ∇θ t[(1 − λ)vdata + λvfake − Fθ] = −t∇θFθ. (74)

###### Step A7: Substitute and simplify. Substituting Eq. (72) and Eq. (74) into Eq. (73): ∇θ Lmix(θ) = 2 t (1 − λ)vdata + λvfake − Fθ ,−t∇θFθ

= −2t2 ⟨(1 − λ)vdata + λvfake − Fθ,∇θFθ⟩

= 2t2 ⟨Fθ − (1 − λ)vdata − λvfake,∇θFθ⟩. (75)

#### Step A8: Distribute the inner product. Using the bilinearity of the inner product, we expand: ∇θ Lmix(θ) = 2t2 ⟨Fθ,∇θFθ⟩ − (1 − λ)⟨vdata,∇θFθ⟩ − λ⟨vfake,∇θFθ⟩

Now we regroup the terms by factoring out (1 − λ) and λ. Note that:

⟨Fθ,∇θFθ⟩ = (1 − λ)⟨Fθ,∇θFθ⟩ + λ⟨Fθ,∇θFθ⟩ Substituting back:

∇θ Lmix(θ) = 2t2 (1 − λ)⟨Fθ,∇θFθ⟩ − (1 − λ)⟨vdata,∇θFθ⟩

+ λ⟨Fθ,∇θFθ⟩ − λ⟨vfake,∇θFθ⟩

= 2t2 (1 − λ)⟨Fθ − vdata,∇θFθ⟩ + λ⟨Fθ − vfake,∇θFθ⟩ . (76)

- Part B: Gradient of the alternative loss. Step B1: Write the alternative loss. From Eq. 22, the alternative loss is

GAPEX(θ) = (1 − λ)Lsup(θ) + λLcons(θ), (77) where Lsup and Lcons are defined in Eq. 20 and Eq. 21.

#### Step B2: Apply the endpoint-velocity equivalence. By Proposition 3, we have

Lsup(θ) = fx(Fθ,xt,t) − x 22 = t2 Fθ − vdata 22. (78) By Proposition 4, we have

Lcons(θ) = fx(Fθ,xt,t) − fx(vfake,xt,t) 22 = t2 Fθ − vfake 22. (79)

###### Step B3: Compute the gradients of Lsup and Lcons. Using the gradient of a squared norm (Lemma from UCGM appendix):

∇θ Lsup(θ) = ∇θ t2 Fθ − vdata 22

= t2 ∇θ Fθ − vdata 22

= t2 · 2⟨Fθ − vdata,∇θFθ⟩

= 2t2 ⟨Fθ − vdata,∇θFθ⟩. (80) Similarly,

∇θ Lcons(θ) = 2t2 ⟨Fθ − vfake,∇θFθ⟩. (81)

###### Step B4: Combine the gradients. Substituting Eq. (80) and Eq. (81) into the gradient of Eq. (77): ∇θ GAPEX(θ) = (1 − λ)∇θ Lsup(θ) + λ∇θ Lcons(θ)

= (1 − λ) · 2t2 ⟨Fθ − vdata,∇θFθ⟩ + λ · 2t2 ⟨Fθ − vfake,∇θFθ⟩

= 2t2 (1 − λ)⟨Fθ − vdata,∇θFθ⟩ + λ⟨Fθ − vfake,∇θFθ⟩ . (82)

- Part C: Conclusion. Comparing Eq. (76) and Eq. (82), we see they are identical: ∇θ Lmix(θ) = ∇θ GAPEX(θ). (83)

This completes the proof of Theorem 1.

| |
|---|

- B.6 FISHER DIVERGENCE PERSPECTIVE

We provide an interpretation of APEX’s alternative loss through the lens of Fisher divergence. This analysis reveals that APEX minimizes a score-space distance with uniform weighting, contrasting with GAN based objectives that use sample dependent weights.

|Proposition 5 (APEX as Fisher Divergence Minimization) . The alternative loss GAPEX(θ) can be interpreted as minimizing a weighted Fisher divergence to a mixed distribution. Specifically, define the mixed score function<br><br>smix(xt) := (1 − λ)sdata(xt) + λsfake(xt), (84) where sdata(xt) = ∇xt<br><br>log pdata,t(xt) and sfake(xt) = ∇xt<br><br>log pfake,t(xt) are the score functions corresponding to the data distribution and fake distribution at time t, respectively. Then, up to time dependent weighting ω(t),<br><br>∇θ GAPEX(θ) ∝ Ex<br><br>t∼pθ,t (sθ(xt) − smix(xt)) ·<br><br>∂xt ∂θ<br><br>, (85) which corresponds to minimizing the Fisher divergence<br><br>DF(pθ∥pmix) := sθ(xt) − smix(xt) 22 pθ(xt)dxt. (86)|
|---|

Proof. Step 1: Relate velocity differences to score differences. By Corollary 1 (Eq. (38)), the velocity-score relationship gives

t 1 − t

(sθ(xt) − sdata(xt)), (87)

Fθ − vdata = −

t 1 − t

(sθ(xt) − sfake(xt)). (88)

Fθ − vfake = −

Derivation reminder: These equations follow from applying the score-velocity duality

xt + (1 − t)v(xt,t) t

st(xt) = −

to each pair of velocity fields. For instance, for Eq. (87):

xt + (1 − t)Fθ t

xt + (1 − t)vdata t

sθ(xt) − sdata(xt) = −

+

(1 − t)(vdata − Fθ) t

=

1 − t t

(Fθ − vdata).

= −

Rearranging gives Eq. (87).

- Step 2: Form the linear combination. From the proof of Theorem 1 (Eq. (82)), the gradient of GAPEX involves the weighted sum

##### (1 − λ)(Fθ − vdata) + λ(Fθ − vfake). (89)

Now we substitute the velocity-score relationships from Step 1. Substituting Eq. (87) and Eq. (88):

(1 − λ)(Fθ − vdata) + λ(Fθ − vfake)

t 1 − t

t 1 − t

= (1 − λ) −

(sθ − sdata) + λ −

(sθ − sfake)

t 1 − t

t 1 − t

(1 − λ)(sθ − sdata) + λ(sθ − sfake) (factor out −

)

= −

t 1 − t

(1 − λ)sθ − (1 − λ)sdata + λsθ − λsfake (expand)

= −

t 1 − t

[(1 − λ) + λ]sθ − (1 − λ)sdata − λsfake

= −

t 1 − t

sθ − (1 − λ)sdata + λsfake (since (1 − λ) + λ = 1)

= −

t 1 − t

sθ(xt) − smix(xt) , (90) where in the last line we used the definition of the mixed score function from Eq. (84):

= −

smix(xt) := (1 − λ)sdata(xt) + λsfake(xt).

#### Step 3: Write the gradient in score-space form. From Eq. (82), the gradient of GAPEX is

∇θ GAPEX(θ) = 2t2 Ex

t,z,t [⟨(1 − λ)(Fθ − vdata) + λ(Fθ − vfake),∇θFθ⟩]. (91) Substituting Eq. (90):

t 1 − t

∇θ GAPEX(θ) = 2t2 Ex

(sθ − smix),∇θFθ

t,z,t −

2t3 1 − t

t,z,t [⟨(sθ(xt) − smix(xt)),∇θFθ⟩]. (92)

Ex

= −

###### Step 4: Relate to Fisher divergence. The Fisher divergence between the model distribution pθ and a target distribution pmix is defined as

DF(pθ∥pmix) = sθ(xt) − smix(xt) 22 pθ(xt)dxt. (93)

Taking the gradient with respect to θ using the score identity ∇x log pθ = sθ and the path-wise gradient estimator:

∂xt ∂θ

. (94)

∇θ DF ∝ Ex

t∼pθ (sθ(xt) − smix(xt)) ·

3

#### Step 5: Absorb time dependent factors. The coefficient − 2t

1−t in Eq. (92) depends only on time t, not on the spatial position xt or the sample. This factor can be absorbed into the time weighting ω(t) used in the expectation. Thus, up to a time dependent proportionality constant,

∂xt ∂θ

, (95)

∇θ GAPEX(θ) ∝ Ex

t∼pθ,t (sθ(xt) − smix(xt)) ·

which matches the form of the Fisher divergence gradient in Eq. (94). Contrast with GAN objectives. For reference, we note that classical GAN objectives involve sample dependent weights. The non saturating GAN gradient takes the form

| |
|---|

∂xt ∂θ

, (96)

∇θ LNS-GAN ∝ Ex

t∼pθ wNS(xt)(sθ(xt) − sdata(xt)) ·

θ(xt)

where the weight wNS(xt) = 1 − D∗(xt) = p

pdata(xt)+pθ(xt) depends on the optimal discriminator D∗(xt). This sample dependent weight can become very small (when D∗ ≈ 1, i.e., generated samples are perfect) or very large (when D∗ ≈ 0, i.e., generated samples are easily distinguished), leading to gradient instability. In contrast, APEX’s gradient in Eq. (85) has a uniform weight across samples (the time dependent factor ω(t) is constant for all xt at a given t). This structural property ensures stable training signals throughout the learning process, independent of the current quality of generated samples.

- C VISUALIZATIONS PART I

This section provides additional qualitative results to complement the quantitative analysis in the main paper.

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

[Figure 58]

[Figure 59]

- Figure 3: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

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

[Figure 74]

- Figure 4: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- Figure 5: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 6: Qualitative Comparison of 512x512 in APEX 0.6B LoRA for NFE=1.

D VISUALIZATIONS PART II

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

- Figure 7: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

- E VISUALIZATIONS PART III

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

###### Figure 8: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

###### Figure 9: Qualitative Comparison of 512x512 in APEX 20B LoRA for NFE=1.

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

###### Figure 10: Qualitative Comparison of 512x512 in APEX 20B Full Parameter Tuning for NFE=1.

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

###### Figure 11: Qualitative Comparison of 512x512 in APEX 20B Full Parameter Tuning for NFE=1.

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

###### Figure 12: Qualitative Comparison of 512x512 in APEX 20B Full Parameter Tuning for NFE=1.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

###### Figure 13: Qualitative Comparison of 512x512 in Qwen-Image Lightning LoRA for NFE=1.

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

###### Figure 14: Qualitative Comparison of 512x512 in Qwen-Image Lightning LoRA for NFE=1.

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

###### Figure 15: Qualitative Comparison of 512x512 in Qwen-Image Lightning LoRA for NFE=1.

[Figure 143]

###### Figure 16: Qualitative Comparison of 512x512 in 20B Full Parameter Tuning of APEX methods and Synthetic dataset from NFE=1 to NFE=20.

[Figure 144]

###### Figure 17: Qualitative Comparison of 512x512 in 20B Full Parameter Tuning of APEX methods and BLIP-3o dataset from NFE=1 to NFE=20.

[Figure 145]

###### Figure 18: Qualitative Comparison of 512x512 in 20B Full Parameter Tuning of sCM methods and BLIP-3o dataset from NFE=1 to NFE=20.

[Figure 146]

###### Figure 19: Qualitative Comparison of 512x512 in 20B Full Parameter Tuning of CTM methods and BLIP-3o dataset from NFE=1 to NFE=20.

[Figure 147]

###### Figure 20: Qualitative Comparison of 512x512 in 20B Full Parameter Tuning of MeanFlow methods and BLIP-3o dataset from NFE=1 to NFE=20.

