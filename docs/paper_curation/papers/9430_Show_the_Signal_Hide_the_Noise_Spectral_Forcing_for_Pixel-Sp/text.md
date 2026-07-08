# arXiv:2606.15236v2[cs.CV]16Jun2026

## Show the Signal, Hide the Noise: Spectral Forcing for Pixel-Space Diffusion

#### Weichen Fan1 Haiwen Diao1 Penghao wu1 Ziwei Liu1,

1S-Lab, Nanyang Technological University

{weichen002,penghao001}@e.ntu.edu.sg, {haiwen.diao,ziwei.liu}@ntu.edu.sg

Github: https://github.com/WeichenFan/Spectral_Forcing. Hugging Face: https://huggingface.co/weepiess2383/Spectral_Forcing.

[Figure 1]

[Figure 2]

### Abstract

Pixel-space diffusion models are trained on full-bandwidth noisy images, yet the useful signal available to the denoiser is strongly frequency dependent. Under rectified-flow diffusion and natural-image power-law spectra, the per-band datato-noise contour k∗(t) = (1 − t)−2/α separates a signal-bearing low-frequency region from a noise-dominated high-frequency region at each time t. We show that this implicit coarse-to-fine structure is not merely descriptive: it induces a capacityallocation problem. A standard pixel-space denoiser must discover the moving bandwidth boundary internally and can spend computation on frequency-time regions where the optimal prediction collapses to deterministic baselines rather than data-distribution modeling. To make this boundary explicit, we introduce Spectral Forcing, a parameter-free, time-conditional 2D-DCT low-pass operator applied to the noisy input before the patch embedder. Its cutoff expands monotonically with the diffusion time and becomes the identity at the data endpoint. Through controlled synthetic experiments, we identify the regime in which the operator is beneficial: coarse patch tokenization and data whose high-frequency content is predominantly noise rather than essential signal. On ImageNet-256 with JiT700M/32, Spectral Forcing consistently improves both FID and Inception Score across different training epochs, demonstrating robust gains throughout training; at finer tokenization, the spectral forcing is still competitive. We further insert the unchanged operator into SenseNova-U1, a unified text-to-image model, where it improves DPG-Bench and GenEval, showing that the input-side spectral prior transfers beyond class-conditional generation. These results suggest a route to capacity-efficient pixel-space diffusion by showing the signal and hiding the noise.

### 1 Introduction

Diffusion and flow-based models are the state of the art for generating high-quality images. Until recently, the dominant recipe was to operate in a compressed latent space produced by a separately trained autoencoder, with the diffusion model itself learning to denoise latents rather than pixels. This separation has been justified on practical grounds (latents are smaller and faster to denoise), but it adds an external dependency to the generative recipe and obscures the spectral structure of the underlying images. Recent work shows that pixel-space diffusion can be competitive when the architecture is properly designed, in particular through coarse patch tokenization and large transformer backbones [24, 32, 50].

A common observation across these recipes is that diffusion training is implicitly coarse-to-fine: at each timestep the noise level determines a frequency band above which the data signal is buried in noise, and the network must learn to reconstruct lower-frequency content first and higher-frequency content last [18]. This implicit hierarchy has been documented through frequency-content analyses but has rarely been exploited as an explicit architectural prior. We argue that the hierarchy is not merely descriptive but induces a capacity-allocation problem: a standard pixel-space denoiser, faced with the

Preprint.

[Figure 3]

[Figure 4]

###### The hidden landscape of pixel-space diffusion Spectral Forcing schedule

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

###### Spectral Forcing

- Figure 1: Spectral Forcing for pixel-space diffusion. Left: the per-band data-to-noise contour k∗(t)=(1−t)−2/α separates a signal-bearing region (data-distribution work) from a noise-dominated region where an unforced denoiser collapses to a closed-form map (wasted capacity). Right: SF imposes the boundary explicitly with a parameter-free, time-conditional 2D-DCT low-pass at cutoff c(t), applied before the patch embedder; c(t) grows monotonically with t and is the identity at t=1. Bottom strip: one operator step — noisy input → 2D-DCT → mask above c(t) → IDCT → denoiser εθ. The diffusion objective, architecture, and sampler are unchanged.

full bandwidth of the noisy input at every timestep, must discover the moving bandwidth boundary internally and can spend computation on frequency-time regions where the optimal prediction collapses to deterministic baselines rather than data-distribution modeling.

The reason this hierarchy emerges is straightforward. Under rectified-flow diffusion, the network at time t observes zt = tx+(1−t)ε with ε ∼ N(0,I). For natural-image-like data with power spectrum P(k) ∝ k−α, the per-band data-to-noise ratio is k−α/(1 − t)2 (the data spectrum compared to the additive-noise variance floor; see Section 3.1 for the relation to the standard zt-SNR), and the contour DNR(k,t) = 1 defines a moving cutoff k∗(t) = (1 − t)−2/α that separates a signal-bearing region from a noise-dominated region (Fig. 1, left). The standard network has no architectural awareness of this cutoff: it must identify it implicitly from the noise schedule, and allocate capacity between learning data-distribution structure where signal exists and reproducing deterministic baselines where it does not. We confirm this allocation directly with a per-band MSE diagnostic at convergence on synthetic data (Section 3.2): the network does meaningful data-distribution work only in a wedge of (t,k) space, and converges to deterministic baselines elsewhere.

We ask: can making the bandwidth boundary explicit at the input free model capacity for the harder part of the task? We introduce Spectral Forcing (SF), a parameter-free time-conditional 2D-DCT low-pass mask applied to the network input before the patch embedder (Fig. 1, right). The mask’s cutoff radius c(t) grows monotonically with the diffusion timestep along a fixed-by-design schedule that is imposed at the input rather than estimated from the data spectrum, restricting the network’s view of zt to the bands where signal can dominate, and saturating to the identity at the data endpoint so the trajectory still integrates full-bandwidth velocity. The operator introduces no learnable parameters, costs about half a percent of total compute at 2562, and composes with any pixel-space rectified-flow recipe without modifying the forward process, the loss, the EMA, the sampler, or classifier-free guidance.

SF is regime-dependent. We show, both in toys and on ImageNet, that the operator delivers its largest gains in a specific conjunction of conditions: (i) the model’s patch tokenization is coarse enough that the patchify already aggressively bandlimits the representation, and (ii) the data’s high-frequency content is predominantly noise rather than essential signal. When these conditions hold, the operator delivers consistent improvements throughout training; when they do not, it remains competitive with the unforced baseline. We make this regime explicit through a controlled toy experiment so that practitioners know where to expect headline gains versus where the operator’s main role is

- as a non-harmful regularizer. The coarse-tokenization regime is also the operating point of recent native vision-language models [7] that bypass an external visual encoder and process raw image patches directly, where token count must be kept small for tractable joint sequence modelling; making

capacity-efficient pixel-space diffusion practical in this regime is therefore a downstream-relevant target.

On ImageNet-256 the empirical headline is sharp. At the JiT-700M/32 configuration (64 transformer tokens, the largest configuration in Li and He [24]), SF reduces FID from 24.19 to 20.68 (+14%) and improves Inception Score from 83.28 to 93.96 (+13%) in an apples-to-apples 60-epoch comparison against a same-recipe baseline. The improvement is robust across training budgets: SF consistently improves both FID and Inception Score at every epoch checkpoint we evaluate, demonstrating that the gain is not a transient data-efficiency artifact. At finer tokenization (256 tokens, JiT-130M/16), SF remains competitive with the baseline at the matched 60-epoch budget, delimiting the regime where it produces additional headline gains versus where it serves as a non-harmful frequency prior.

Our work makes the following contributions:

- • We formalize a per-band data-to-noise analysis as a closed-form bandwidth-coherence framework that predicts the operator’s optimal cutoff schedule, the analytical bandwidth front c(t) ∝ (1 − t)−2/α, and identifies its regime of applicability.
- • We design a controlled 1D-to-2D toy experiment that derives the operator from a per-band MSE diagnostic (showing that an unforced network has converged to a deterministic baseline outside a wedge of (t,k) space), characterizes its dependence on patch size and data spectrum, and exposes when it helps or hurts. We verify the same wedge structure directly on real ImageNet checkpoints.
- • We validate on ImageNet-256: a +14.5% FID and +13% Inception Score gain at JiT-700M/32 in an apples-to-apples 60-epoch comparison, holding +8.0% at 120 epochs where SF already matches a published ∼145-epoch reference. Our method surpasses constant low-pass, spatial Gaussian blur, Focal Frequency Loss [20], blurring diffusion [16], and DCTDiff [32] at the same operating point.

Together these results suggest a simple route to more capacity-efficient pixel-space diffusion by showing the denoiser the signal and hiding the noise.

### 2 Related Work

Diffusion and pixel-space generation. Diffusion and flow-matching dominate high-quality image generation [44, 14, 31, 45, 46, 22, 26, 27, 1], typically with transformer backbones [9, 33, 28] replacing U-Nets [38]. Latent diffusion compresses images via separately trained autoencoders [37, 34, 10, 23], with recent work substituting representation-rich tokenizers for faster convergence [54, 52, 56, 41, 55, 12]. Pixel-space diffusion is the alternative [6, 40, 30, 15, 17, 4, 50, 25, 29]; we build on JiT [24], where large-patch transformers match latent baselines without auxiliary losses. The coarse-tokenization regime is also the operating point of native VLMs [7] that process raw image patches without a separate encoder. SF is a parameter-free input-side adapter that composes with any of these and leaves forward process, loss, and sampler unchanged.

Spectral and frequency-domain methods. Some prior work makes the forward process itself spectral, replacing Gaussian noise with progressive blurring or wavelet shrinkage [36, 16]; analyses without modifying the forward process formalize standard diffusion’s implicit coarse-to-fine character [18] and link it to the spectral bias of neural networks [35, 47, 42] and natural-image power-law statistics [49, 3, 39]. Other work generates directly in a frequency representation [32] or coarseto-fine token order [19, 51, 48, 5, 11, 53, 43], sometimes with frequency-aware objectives [20, 21]. Latent Forcing [2] cascades a frozen semantic encoder with a pixel-level diffusion head. SF differs structurally: the forward process is unchanged rectified-flow, the architecture is unchanged, and the operator is a parameter-free time-conditional mask on the pixel input whose schedule is derived from the per-band data-to-noise contour of the unmodified forward process.

### 3 Methodology

#### 3.1 Preliminaries

Rectified-flow. For data x ∼ q(x | y) and noise ε ∼ N(0,I), rectified-flow [27, 26] linearly interpolates between source and data:

zt = tx + (1 − t)ε, t ∈ [0,1], (1)

###### (a) toy spectra vs ImageNet

###### (b) 1D toy: capacity allocation

0.0

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | |structured|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| |m|age|Net| |owe|r-|law| |rectangles| |structured| |power-law|
| | |[Figure 9]| | | | |[Figure 10]| |[Figure 11]| |[Figure 12]| |ImageNet|
| | | | | | | | | | | | | |rectangles|

| |[Figure 13]<br><br>signal recovery<br><br>closed-form denoising<br><br>predict-zero| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

|[Figure 14]| |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

100

120

0.2

100

10 1

ed

0.4

()(normalized)Pk

frequencybandk

80

log(net/zero)10

0.6

10 2

60

0.8

10 3

40

1.0

w

et

20

1.2

10 4

es

0

1.4

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

10 1 100

time t

normalized k

###### (c) 2D toy: patch-size regime

###### (d) 2D toy: data-spectrum regime

80

47.1

+70%

baseline linear-SF analytical-SF

| |
|---|

40.9

40

| |
|---|

60

low-passeffectonL1(%)

SpectrumL1at200ep

30

40

+35%

26.8 23.1

26.2

20.2

18.9

20

18.1

20

16.5

+12%

10

0

-6%

0

p=4 (256 tok)

p=2 (1024 tok)

p=8 (64 tok)

p=16 (16 tok)

power-law structured rectangles

- Figure 2: Three empirical motivations for Spectral Forcing. (a) Radial 2D-DCT power spectra of the three toy distributions, overlaid on ImageNet-256 (insets: samples). (b) Converged 1D

toy denoiser: per-band log10(MSEnet/MSEzero) on the (t,k) plane reveals three regions: signal recovery (low-k wedge, the only region of true data-distribution work), closed-form denoising (low t, high k), predict-zero (high t, high k). (c) 2D toy DiT: input-side time-conditional low-pass vs. patch size (h=64, α=2). Helps at coarse p; starves at very fine p. (d) Same operator across data spectra

- at p=8. Helps on power-law (analytical ≫ linear), neutral on structured, hurts on rectangles where high-frequency content is essential signal. so z0 = ε is noise and z1 = x is the data. The per-sample velocity target is

x − zt 1 − t

= x − ε, (2) and a flow-matching model vθ(zt,t,y) is trained against vtarget under squared error,

vtarget =

L(θ) = Ex,ε,t ∥vθ − vtarget∥2 , (3)

with t logit-normal at training and EMA weights at inference. Sampling integrates vθ from t=0 to 1 with a Heun integrator and classifier-free guidance.

Per-band data-to-noise ratio. We approximate the radial 2D-DCT spectrum of natural images by P(k) ∝ k−α, with α ≈ 2.82 on ImageNet-256 (Section A). Since Eq. (1) adds per-band noise variance (1 − t)2, we define the per-band data-to-noise ratio (DNR) as

k−α (1 − t)2

P(k) (1 − t)2

. (4)

DNR(k,t) =

=

Its unit level set gives a closed-form bandwidth front, k∗(t) = (1 − t)−2/α, above which noise dominates the data spectrum. DNR references the clean-data power P(k) rather than its attenuated

image t2P(k) in zt, and relates to the conventional zt-SNR by log SNR = log DNR + 2log t. The 2log t term is constant in k: it scales every band identically and leaves the spectral slope −α unchanged, so it carries no frequency-discriminative information and shifts only the absolute level at which the ratio crosses unity, which we fold into the cutoff c(t).

#### 3.2 Empirical Study

The bandwidth front k∗(t) partitions the (k,t) plane into signal-bearing (below) and noise-dominated (above) regions; as t → 1 the front sweeps outward, exposing more bands. The empirical question

is whether a standard pixel-space denoiser uses this structure, and where the allocation becomes wasteful. We answer with three controlled experiments on small models (Fig. 2), before any version of Spectral Forcing is introduced: a 1D rectified-flow Transformer (∼178k params) on synthetic 1D power-law signals, and a 2D DiT (∼3M params) on h×h synthetic images at h=64, trained under the recipe of Eq. (3).

- Result 1: the network does data-distribution work only in a wedge. At convergence we measure the 1D model’s per-band velocity-prediction MSE relative to the trivial zero-predictor baseline

∥vtarget∥2 on a dense (k,t) grid (Fig. 2(b)). Three regions emerge. The signal-recovery wedge (low k, growing with t) is the only one where the network beats zero by having learned data-distribution

structure. In the closed-form denoising regime (low t, high k) the data signal xk is negligible, so vtarget ≈ −ε and the network reduces to the linear map −zt/(1 − t) (More details could be found in Appendix B.5; in the predict-zero regime (high t, high k) both signal and noise contributions to zt are small and vtarget ≈ 0. Off the wedge the network has converged to a deterministic baseline independent of the data distribution: capacity spent on those bands is wasted.

| |[Figure 15]<br><br>closed-form denoising<br><br>signal recovery<br><br>predict-zero| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 diffusion time t

0

5

10

15

20

25

30

radialDCTbandk

Per-band wedge on real ImageNet (JiT-700M/32, baseline 60 ep)

[Figure 16]

1.0

0.5

0.0

0.5

1.0

log(MSE/MSE)10netzero

Figure 3: The wedge transfers from the toy to real ImageNet. Per-band log10(MSEnet/MSEzero-pred.) for a trained JiT-700M/32 baseline (60 ep, EMA weights). The three regions identified in Fig. 2(b) are visible: closed-form denoising (small t, high k), the signal-recovery wedge (low k, growing with t), and a predict-zero region (large t, mid–high k) where the network is no better than a zero predictor.

Real-image confirmation. The wedge is a property of the loss landscape, not of the toy. Re-running the same per-band log10(MSEnet/MSEzero) diagnostic on a real ImageNet checkpoint (JiT-700M/32 baseline at 60 ep, EMA weights, 256×256 inputs) recovers the same three regions in identical arrangement (Fig. 3); the predict-zero region at high t and mid–high k even hits log10(·) ≥ 0, i.e., the network is at or below the trivial baseline. The toy result transfers, and the wasted-capacity claim is empirical at scale.

- Result 2: the cost of front-tracking depends on the patchify. We ask whether an explicit input-side low-pass with the same time depen-

dence as k∗(t) helps. A patch-size sweep on h=64 power-law 2D data with a fixed timeconditional low-pass (Table 1) gives a monotonic ordering: the adapter helps strongly when p is large relative to the signal-bearing bandwidth (+70% L1 at p=2, 1024 tokens), with the gain shrinking as p decreases and reversing at very coarse patches.

Table 1: Patch-size sweep on a 2D toy DiT at h = 64, α = 2, with a fixed time-conditional input lowpass. The operator helps when the patchify already aggressively bandlimits the input and starves the model when the token count is too small.

p Tokens ∆L1 Regime

2 1024 +70% favorable 4 256 +35% favorable 8 64 +12% boundary

16 16 −6% starved

- Result 3: the cost depends on the data spectrum. Patch size is not the only axis. Sweeping three synthetic distributions at p=8 (Fig. 2(a,d))

— a 2D power-law matched to the ImageNetfitted α ≈ 2.82, a hard-edged rectangles distribution, and a structured (blobs + stripes + noise) distribution — the input low-pass helps strongly on power-law (analytical: L1 26.2→16.5; linear: 26.2→23.1), ties baseline on structured (linear hurts slightly), and is destructive on rectangles where high-frequency content is essential edge signal (baseline 26.8 → linear 47.1, analytical 40.9). SF’s favorable regime is the conjunction of coarse patchify and data

whose high-frequency content is dominated by noise rather than signal.

Implication. Together: outside a wedge-shaped low-k region the network has converged to a deterministic baseline and is not modelling the data (Result 1); when the patchify is coarse enough, replacing those wasted bands with an explicit input-side low-pass tracking k∗(t) frees capacity and helps (Result 2); when the data’s high-frequency content is essential signal, the same operator

removes information the network needs and hurts (Result 3). We propose SF as the operator that exploits Result 2 in the regime characterized by Result 3.

#### 3.3 Spectral Forcing

The operator. Given the rectified-flow input zt ∈ RC×H×W we apply, before any other network operation,

SFt(z) = IDCT(DCT(z) ⊙ M(t)), (5) M(t)[u,v] = σ(κ · (c(t) − r(u,v))), (6)

√u2 + v2 2(W − 1)2

, (7)

r(u,v) =

c(t) = cmin + (cmax − cmin) · f(t), (8) where r(u,v) ∈ [0,1] is the normalized DCT-II radius, c(t) is a time-dependent radial cutoff, σ(·) is the sigmoid, and κ = 30 controls the transition sharpness of the soft mask. We use cmin = 0.05 and cmax = 1.0 throughout; f : [0,1] → [0,1] is the schedule shape discussed below. The network’s effective input is SFt(zt); everything downstream, the velocity target Eq. (2), the MSE loss Eq. (3), the EMA, the Heun sampler, classifier-free guidance, is unchanged.

The operator is a drop-in input adapter. It introduces no learnable parameters, costs one forward and one inverse 2D-DCT per training and sampling step (about 0.5% of total compute at 2562), and inherits the data-endpoint by design: at t = 1 the cutoff saturates at cmax = 1.0 and the mask becomes the identity, so the trajectory still integrates full-bandwidth velocity at the data boundary.

Schedule shape and time-dependence. Time-dependence is forced by the data endpoint. A constant low-pass with c < 1 cannot reach the natural-image distribution because it permanently zeros bands the data does have, leaving excess high-frequency mass in any generated sample. A constant c = 1 is the no-op. The interesting design space is therefore the family of monotonic f(t) that interpolate between an aggressive cutoff at t = 0 and the identity at t = 1.

Table 2: Cutoff schedules f(t) ablated in this paper. Pseudo-code in Algorithm 1.

We ablate the schedule shapes in Table 2. The linear schedule, f(t) = t, is the simplest interpolant. The analytical schedule, f(t) ∝ (1 − t)−2/α with the appropriate normalization, is the bandwidth front itself: under it the operator’s pass-band tracks k∗(t) exactly, so the network only ever sees bands below the DNR=1 contour. The intermediate t2, √t, and cosine shapes are ablated against these in Section B.2. Empirically, the choice between linear and analytical is more subtle than the regime question itself: on simple power-law toys both beat the baseline (analytical by a 3× larger margin); on rectangle toys no schedule of any shape beats baseline at convergence; on ImageNet at 64 tokens the linear schedule is the empirically better default — we therefore report linear-SF as the default in Section 4 and treat analytical as a refinement that recovers at higher resolution (Section 5).

Schedule f(t) linear t

t√2t √t2t

cosine 12(1 − cos πt) analytical ∝ (1−t)−2/α

The combination of (i) the closed-form bandwidth-front identity Eq. (4), (ii) the empirical observation in Section 3.2 that the network has converged to a deterministic baseline outside the signal-recovery wedge, (iii) the parameter-free DCT operator Eq. (5), and (iv) the schedule shape ablation above is the full method; pseudo-code for c(t), M(t), and a training/sampling step is in Section B.6.

### 4 Experiments

Setup. We use the JiT architecture of Li and He [24] at three scales: JiT-130M/32 (64 transformer tokens at 2562), JiT-130M/16 (256 tokens), and JiT-700M/32 (the largest configuration in Li and He [24]). All runs use the JiT recipe unmodified: rectified-flow forward process, time sampling Nlogit(−0.8, 0.8), lr 5 × 10−5, batch 128 per GPU on 8 GPUs, EMA, Heun-50 sampler with CFG 2.9. FID-50k is reported against the canonical ImageNet-256 reference. Each Spectral Forcing run is paired with a same-recipe baseline that differs only in whether the operator is active; architecture, optimizer, and seed are matched. Unless stated, SF runs use the linear schedule f(t) = t.

- Table 3: Spectral Forcing on ImageNet-256. We report FID-50k against same-recipe JiT baselines under both coarse and fine tokenization settings. All results are averaged over 5 random seeds for fair comparison. Spectral Forcing consistently improves FID across model scales, token counts, and training epochs.

Model Tokens Epochs Baseline FID +SF FID ∆FID

— Coarse tokenization (64 tokens) JiT-130M/32 64 15 114.03 100.78 +11.6% JiT-130M/32 64 60 44.68 42.92 +3.9% JiT-130M/32 64 100 33.30 33.18 +0.4% JiT-130M/32 64 200 25.29 24.91 +1.5%

JiT-700M/32 64 60 24.19 20.68 +14.5% JiT-700M/32 64 90 19.90 17.53 +11.9% JiT-700M/32 64 120 16.46 15.15 +8.0%

— Fine tokenization (256 tokens) JiT-130M/16 256 60 21.76 21.29 +2.2%

###### (a) Multi-epoch FID on ImageNet-256

###### (b) Spectral Forcing margin vs training budget

JiT-130M/32 (baseline)

FIDimprovementofSFoverbaseline(%)

+14.5%

102

JiT-130M/32 (SF)

15

JiT-700M/32 (baseline)

+11.9%

JiT-700M/32 (SF)

+11.6%

FID-10k(lowerisbetter)

6 × 101

+8.0%

10

- 2 × 101

- 3 × 101

- 4 × 101

5

+3.9%

+2.2%

+1.5%

0

###### +14.5%

+0.4%

JiT-130M/32 JiT-700M/32 JiT-130M/16

5

-3.6%

25 50 75 100 125 150 175 200

25 50 75 100 125 150 175 200

training epochs

training epochs

- Figure 4: Multi-epoch behaviour of Spectral Forcing on ImageNet-256. (a) FID-50k trajectories (log-scale) for JiT-130M/32 and JiT-700M/32; solid: baseline, dashed: Linear-SF; the headline 60-epoch gap at JiT-700M/32 is annotated. (b) FID improvement of SF over the matched-epoch baseline. JiT-130M/32 (blue) compresses to within evaluator noise by 100 ep then holds a small persistent margin at 200 ep (+1.5%); JiT-700M/32 (red) retains an asymptotic component out to 120 ep (+8.0%); JiT-130M/16 (gray) is regime-bounded.

Cross-scale picture. Table 3 shows that Spectral Forcing reduces FID across every (model, epoch budget) pair tested at coarse tokenization, with the largest gain at the headline row (⋆, +14.5% at JiT-700M/32, 60 epochs); at fine tokenization (256 tokens, last row), the effect is within evaluator noise. The qualitative pattern, helpful at 64 tokens and neutral at 256, holds at every epoch budget we tested.

Effect of training budget. Fig. 4b separates two regimes. At JiT-130M/32 the margin compresses sharply from +11.6% (15 ep) to +0.4% (100 ep), then holds a small persistent component out to 200 ep (+1.5%): the bulk of the gain at small scale is data-efficiency, with a residual asymptotic margin within evaluator noise. At JiT-700M/32 the same margin compresses only from +14.5% (60 ep) to +8.0% (120 ep), and the 120-ep SF FID of 15.15 already matches the previous-best 700M+SF reference at ∼145 ep (FID 15.24): a meaningful asymptotic component remains at large scale.

Qualitative samples comparing baseline against Linear-SF at the same class label and noise seed are deferred to Fig. 6 in Section B.7.

Schedule choice at the headline. A separately trained Analytical-SF (cmin=0.20) at JiT-700M/32, 60 ep reaches FID 21.94 (+9.3%): the linear schedule of Table 3 (+14.5%) is the empirically better default on ImageNet-256 at 64 tokens, despite the analytical schedule’s 2–3× advantage in the h=64 toy (Section 3.2); the analytical-wins ordering is restored at higher image resolution in toys (Table 8).

- Table 4: Operator-choice comparison at JiT-130M/32, 2562, 60 ep. The first three rows compare SF’s design axes: Const. LP (time-invariant DCT, c=0.5) tests time-dependence, Gauss blur (σ=8(1−t) px, no DCT) tests the choice of frequency-domain mask, and FFL [20] added to the MSE tests an input-side mask vs. a loss-side reweighting. The last two rows compare against published frequency-domain methods: Blurring diffusion [36, 16] replaces the Gaussian forward with a heatequation blur; DCTDiff [32] runs the model in DCT-coefficient space throughout. All five lose to the unforced baseline; SF is the only operator that helps at this operating point.

Adapter / method FID ∆FID vs. baseline baseline 44.68 —

+ Linear-SF 42.92 +3.9% + Const. DCT low-pass (c=0.5, no time-dep) 45.45 −1.7%

+ Spatial Gaussian blur (σmax=8 px) 67.24 −50.5% + Focal Frequency Loss [20] 71.45 −59.9% Blurring diffusion [36, 16] 60.75 −36.0% DCTDiff [32] 50.12 −12.2%

Comparison to alternative operators. We compare Linear-SF against five alternatives at JiT130M/32, 60 ep (Table 4). Constant low-pass (c=0.5) confirms the prediction of Section 3.3 that a permanent low-pass cannot reach the data distribution: time-dependence is required. Spatial Gaussian blur (σ(t)=8(1−t) px, no DCT) shows that a spatial blur of comparable severity is not interchangeable with the DCT mask. Focal Frequency Loss [20] reweights (v − vpred) in frequency, the closest loss-side analog to our input-side mask, but is worse even than the Gaussian-blur ablation: loss-side reweighting is not interchangeable with the input-side spectral mask. Blurring diffusion [16] (heat-equation forward) and DCTDiff [32] (model in DCT space) both lose to the unforced baseline and to Linear-SF by a larger margin: prior frequency-domain recipes pay an integration cost that simple input-side spectral forcing avoids.

Native vision-language models. The coarse-tokenization regime where SF delivers its largest gains is also the operating point of native VLMs that bypass an external visual encoder and process raw image patches directly [7], where token count must be kept small to keep joint text-image sequence modelling tractable. We test whether SF’s gains transfer to this setting by inserting the unchanged Linear-SF operator into SenseNova-U1 [8], a unified text–image model and comparing to a samerecipe baseline at the same stage-1 100k-step. Fig. 5 reports the DPG-Bench headline and per-category sweep: SF wins 9 of 13 subcategories. The largest gains concentrate on coarse-to-fine semantic axes that decode at low spatial frequencies, where freeing capacity from noise-dominated bands is most productive. The same trend holds on GenEval at this early training stage (Section B.4). The SF designed and validated for pixel-space class-conditional diffusion, transfers without modification to text-conditional native-VLM generation.

- 5 Ablation Study

Toy experiments use synthetic h×h images with p=h/8 so the token count is held at 64 across h. Headline: the operator helps at ≤ 64 tokens at 2562 across every backbone size, with a clean reversal at higher token counts unless resolution scales to compensate (Table 5).

Impact of data distribution. The winner shifts with data structure (Table 8, distribution block): on power-law data the unforced baseline catches up at convergence (SF is data-efficiency only); on structured data the analytical schedule wins because the front correctly tracks noise-dominated bands; on rectangles both schedules fail because high-frequency content carries essential edge signal — the controlled analogue of the higher-token-count regime in Table 7.

Table 5: Real-image resolution check: JiT130M/32 at 5122 (256 tokens) for 30 ep, neutral at 2562 but recovers margin at 5122.

Impact of image resolution. Larger h is more favorable. In toys (Table 8, resolution block), analytical-SF moves from worst at h=64 to best at h≥128, with −15% at h=256 before saturating (−3.3% at h=512). The same pattern holds on real images: JiT-130M/32 at 5122 (256 tokens, neutral at 2562 per Table 3) recovers +3.4% FID (Table 5).

Run FID IS

baseline 68.34 23.77 + Linear-SF 66.01 24.81

Baseline

64.35

+ SF (win)

DPG-Bench (overall)

###### 67.85

+ SF (loss)

per-category (sorted by SF BL)

75.20

entity --- state

84.12 75.18

entity --- whole

80.88 73.61

other --- count

77.38

80.78

attribute --- shape

82.94 77.49

attribute --- color

79.46 77.38

other --- text

79.36 80.03

global

81.35 76.27

attribute --- texture

77.43

81.53

entity --- part

82.37 81.28

relation --- non-spatial

80.52

83.96 81.71

relation --- spatial

82.31 78.62

attribute --- size

85.24 77.04

attribute --- other

65 70 75 80 85 90

DPG-Bench score (higher is better)

- Figure 5: Spectral Forcing transfers to native vision-language models: DPG-Bench overall and per-category. SenseNova-U1 [8] at stage-1 100k steps; identical baseline (BL) and SF recipe except for the input operator. Top bar is the overall headline; categories below are sorted by SF − BL. SF bars are coloured by win/loss against BL; SF wins 9 of 13 subcategories.

#### Impact of patch size. Sweeping p ∈ {16,32,64}

at JiT-130M, 2562, 60 ep gives {256,64,16} tokens (Table 6): at p=16 SF is within evaluator noise of baseline; at p=32 it reduces FID by +3.9% at 130M and +14.5% at 700M; at p=64 both runs are far from converged and the operator is again within noise. The favorable regime is bounded on both sides; combined with the 5122 row of Table 5, the operative axis is token count, not p or H alone.

Table 6: Patch-size sweep at JiT-130M, 2562, 60 ep. Rows p=16,32 aggregate Tables 3 and 7; p=64 is new.

Higher-token regime (JiT-130M/16). The operator is regime-bounded. Table 7 shows that at 60 epochs all three SF schedules sit within 0.53 FID points of baseline; the 15-epoch losses are data-efficiency artifacts that resolve at convergence. The Inception Score reveals that the analytical-SF run’s marginal 60-ep FID hides a −6.6% class-diversity loss (78.04 vs. 83.59). At 256 tokens the patchify already filters out little of the high-frequency content the network needs, so the input mask neither frees useful capacity nor removes useful signal; the toy rectangle-data result of Section 3.2 is the controlled-setting analogue.

p Tok Base +SF ∆

16 256 21.76 21.29 +2.2% 32 64 44.68 42.92 +3.9% 64 16 84.50 84.69 −0.2%

Impact of training budget. On ImageNet-256 (Fig. 4): at JiT-130M/32 the margin compresses +11.6%→+0.4% over 15–100 ep then holds +1.5% at 200 ep (mostly data-efficiency at small scale); at JiT-700M/32 it compresses only +14.5%→ + 8.0% over 60–120 ep, and the 120-ep SF FID 15.15 matches the published ∼145-ep reference (FID 15.24). The toy 2000-ep block of Table 8 widens the analytical-SF margin from −15% to −17% at h=256: SF’s gain is not purely a data-efficiency artifact in the regimes that matter.

Impact of schedule choice and the linear–analytical gap. Schedule preference flips with regime: baseline wins at h=64 (toy), analytical wins at h≥128 (Table 8), and linear beats analytical by 1.3 FID on ImageNet-256 at 64 tokens (+14.5% vs. +9.3%). The closed-form schedule f(t) ∝ (1 − t)−2/α is the cutoff that tracks DNR= 1 exactly, but loses on ImageNet-256 at 64 tokens for three reasons that all relax at higher resolution. (i) Finite-α deviation: natural-image high-k tails fall faster than the global α≈2.82 fit (Section A) due to anti-aliasing and sensor noise, so the formula prescribes

Table 7: JiT-130M/16 (256 tokens) on ImageNet256. At fine tokenization SF is neither helpful nor harmful at converged budget; the 15-ep losses are data-efficiency artifacts. Analytical-SF ties baseline FID but loses 6.6% IS, exposing a classdiversity penalty hidden by FID alone.

Run FID IS ∆FID

— 15 epochs (data-efficiency) baseline 81.57 15.56 —

+ Linear-SF 84.48 20.14 −3.6%

+ Analytical-SF (cmin=0.05) 100.95 15.28 −23.8%

+ Analytical-SF (cmin=0.20) 85.80 18.82 −5.2%

— 60 epochs (converged) baseline 21.76 83.59 —

+ Linear-SF 21.29 83.13 +2.2% + Analytical-SF (cmin=0.20) 21.23 78.04 +2.4%

Table 8: Toy ablations. Lower L1 is better; bold marks the winner per row. Resolution sweep at α=2, p=h/8 (64 tokens); distribution sweep at h=64, p=8.

Setting Base Lin-SF Anal-SF

— Resolution (1000 ep) h=64 (n=5) 9.17±2.5 18.36±2.5 20.57±2.1 h=128 (n=4) 33.50±0.8 35.71±1.7 28.79±1.6 h=256 (n=5) 48.69±1.1 46.37±1.9 41.38±2.0 h=512 (n=5) 67.21±1.2 67.95±1.9 64.98±1.7

— 2000-epoch follow-up h=128 (n=3) 32.57±1.3 33.95±2.5 29.43±1.1 h=256 (n=1) 46.38 45.18 38.68

— Distribution (h=64, p=8) —

structured (n=4) 17.36 27.74 17.00 rectangle (n=4) 31.01 44.08 46.03

a too-aggressive cutoff. (ii) Patchify bandlimiting: at p=32 the embedder already truncates to the 8×8 token grid, so analytical’s small early c(t) redundantly masks bands the patchify has discarded; the linear ramp avoids this double-mask, and the redundancy disappears as h grows at fixed token count. (iii) Training dynamics: (1−t)−2/α grows slowly for small (1−t), so c(t) ≈ cmin for most of training, starving the network of useful gradient at 64 tokens. The framework therefore predicts the qualitative shape of the optimal schedule rather than the exact functional form; linear is a robust empirical interpolant in that family, while analytical recovers at higher resolution.

Training and inference efficiency. SF is parameter-free and adds ≈ 0.5% per-step compute (one forward+inverse 2D-DCT). At JiT-700M/32 (Table 3 and Fig. 4), SF reaches the baseline’s 90/120/145-ep FID in 60/90/120 ep, a 17–33% wall-clock reduction to any target. Inference cost is unchanged up to the 0.5% DCT overhead.

### 6 Conclusion

Spectral Forcing turns the bandwidth boundary that diffusion training discovers implicitly into an explicit input-side prior: a parameter-free time-conditional 2D-DCT low-pass applied before the patch embedder, with a cutoff schedule derived from the per-band data-to-noise contour of the unmodified rectified-flow process. The operator composes with any pixel-space recipe at negligible compute overhead. At JiT-700M/32 on ImageNet-256 it delivers improvements in both FID and Inception Score, and reaches the previously published reference in substantially fewer epochs; at finer tokenization the operator is neither helpful nor harmful, which delimits its applicability cleanly. The coarse tokenization paired with noise-dominated high-frequency content, coincides with the operating point at which pixel-space transformers and native vision-language models are practical.

### References

- [1] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. volume 26, pages 1–80, 2025.
- [2] Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Justin Johnson, Ehsan Adeli, and Li Fei-Fei. Latent forcing: Reordering the diffusion trajectory for pixel-space image generation. 2026.
- [3] Geoffrey J Burton and Ian R Moorhead. Color and spatial structure in natural scenes. Applied optics, 26

(1):157–170, 1987.

- [4] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. 2025.
- [5] Emily L Denton, Soumith Chintala, Rob Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial networks. volume 28, 2015.
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. volume 34, pages 8780–8794, 2021.

- [7] Haiwen Diao, Mingxuan Li, Silei Wu, Linjun Dai, Xiaohua Wang, Hanming Deng, Lewei Lu, Dahua Lin, and Ziwei Liu. From pixels to words–towards native vision-language primitives at scale. 2025.
- [8] Haiwen Diao, Penghao Wu, Hanming Deng, Jiahao Wang, Shihao Bai, Silei Wu, Weichen Fan, Wenjie Ye, Wenwen Tong, and Xiangyu Fan et al. Sensenova-u1: Unifying multimodal understanding and generation with neo-unify architecture. arXiv preprint arXiv:2605.12500, 2026.
- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, and Sylvain Gelly et al. An image is worth 16x16 words: Transformers for image recognition at scale. 2020.
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, and Frederic Boesel et al. Scaling rectified flow transformers for highresolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [11] Wan-Cyuan Fan, Yen-Chun Chen, DongDong Chen, Yu Cheng, Lu Yuan, and Yu-Chiang Frank Wang. Frido: Feature pyramid diffusion for complex scene image synthesis. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 579–587, 2023.
- [12] Weichen Fan, Haiwen Diao, Quan Wang, Dahua Lin, and Ziwei Liu. The prism hypothesis: Harmonizing semantic and pixel representations via unified autoencoding. 2025.
- [13] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. 2022.
- [14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. volume 33, pages 6840–6851, 2020.
- [15] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. volume 23, pages 1–33, 2022.
- [16] Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. 2022.
- [17] Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans. Simpler diffusion: 1.5 fid on imagenet512 with pixel-space diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18062–18071, 2025.
- [18] Yuanhui Huang, Weiliang Chen, Wenzhao Zheng, Yueqi Duan, Jie Zhou, and Jiwen Lu. Spectralar: Spectral autoregressive visual generation, 2025.
- [19] Zhihao Huang, Xi Qiu, Yukuo Ma, Yifu Zhou, Junjie Chen, Hongyuan Zhang, Chi Zhang, and Xuelong Li. Nfig: multi-scale autoregressive image generation via frequency ordering. 2025.
- [20] Liming Jiang, Bo Dai, Wayne Wu, and Chen Change Loy. Focal frequency loss for image reconstruction and synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13919–13929, 2021.
- [21] Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. volume 34, pages 852–863, 2021.
- [22] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. volume 35, pages 26565–26577, 2022.
- [23] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, and Patrick Esser et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. 2025.
- [24] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. 2025.
- [25] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. volume 37, pages 56424–56445, 2024.
- [26] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. 2022.
- [27] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. 2022.
- [28] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.

- [29] Duy-Kien Nguyen, Mahmoud Assran, Unnat Jain, Martin R Oswald, Cees GM Snoek, and Xinlei Chen. An image is worth more than 16x16 patches: Exploring transformers on individual pixels. 2024.
- [30] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. 2021.
- [31] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR, 2021.
- [32] Mang Ning, Mingxiao Li, Jianlin Su, Haozhe Jia, Lanmiao Liu, Martin Beneš, Wenshuo Chen, Albert Ali Salah, and Itir Onal Ertugrul. Dctdiff: Intriguing properties of image generative modeling in the dct space. 2024.
- [33] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [34] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. 2023.
- [35] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.
- [36] Severi Rissanen, Markus Heinonen, and Arno Solin. Generative modelling with inverse heat dissipation. 2022.
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [38] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- [39] Daniel L Ruderman. The statistics of natural images. Network: computation in neural systems, 5(4):517, 1994.
- [40] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, and Tim Salimans et al. Photorealistic text-to-image diffusion models with deep language understanding. volume 35, pages 36479–36494, 2022.
- [41] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. 2025.
- [42] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. volume 33, pages 7462–7473, 2020.
- [43] Ivan Skorokhodov, Willi Menapace, Aliaksandr Siarohin, and Sergey Tulyakov. Hierarchical patch diffusion models for high-resolution video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7569–7579, 2024.
- [44] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.
- [45] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. volume 32, 2019.
- [46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. 2020.
- [47] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. volume 33, pages 7537–7547, 2020.
- [48] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. volume 37, pages 84839–84865, 2024.

- [49] Antonio Torralba and Aude Oliva. Statistics of natural image categories. Network: computation in neural systems, 14(3):391, 2003.
- [50] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. 2025.
- [51] Yikai Wang, Zhouxia Wang, Zhonghua Wu, Qingyi Tao, Kang Liao, and Chen Change Loy. Next visual granularity generation. 2025.
- [52] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.
- [53] Srikar Yellapragada, Alexandros Graikos, Kostas Triaridis, Prateek Prasanna, Rajarsi Gupta, Joel Saltz, and Dimitris Samaras. Zoomldm: Latent diffusion model for multi-scale image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 23453–23463, 2025.
- [54] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. 2024.
- [55] Zhengrong Yue, Haiyu Zhang, Xiangyu Zeng, Boyu Chen, Chenting Wang, Shaobin Zhuang, Lu Dong, Yi Wang, Limin Wang, and Yali Wang. Uniflow: A unified pixel flow tokenizer for visual understanding and generation. 2025.
- [56] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. 2025.

### A Implementation Details

Our implementation closely follows the JiT recipe of Li and He [24], with Spectral Forcing as a deterministic input-side adapter applied before the patch embedder. The configurations of all our experiments are summarized in Table 9; we describe the details below.

Time distribution. Following Esser et al. [10], during training we adopt a logit-normal distribution over t: logit(t) ∼ N(µ,σ2). We sample s ∼ N(µ,σ2) and let t = sigmoid(s). The hyperparameter µ shifts the typical noise level; following Li and He [24] we use µ = −0.8 and σ = 0.8 on ImageNet-256 throughout.

Backbone and patchify. We use the JiT architecture of Li and He [24] unmodified, at three configurations: JiT-130M/32 (64 transformer tokens at 2562), JiT-130M/16 (256 tokens), and JiT700M/32 (64 tokens, the largest configuration in Li and He [24]). The DCT window in the SF operator is matched to the patch size in the patch embedder.

Spectral Forcing operator. The operator applies a single soft 2D-DCT radial low-pass to the full rectified-flow input zt ∈ RC×H×W before the patch embedder; the DCT is taken over the whole H × W grid, so the radial cutoff acts on global image frequencies. Its hyper-parameters are the cutoff end-points cmin,cmax ∈ [0,1], the schedule shape f(t), and the soft-mask sharpness κ in σ κ(c(t) − r(u,v)) ; the analytical schedule additionally uses the spectrum exponent α. Throughout the paper we fix cmin = 0.05, cmax = 1.0, and κ = 30, and use the linear schedule f(t) = t unless otherwise noted (the analytical schedule f(t) ∝ (1−t)−2/α is used only in the toy resolution-scaling experiments).

Toy experiments. The 1D rectified-flow Transformer used in Section 3.2 has ∼ 178k parameters (4 layers, 200 epochs); the 2D DiT used in Section 5 has ∼ 3M parameters and is trained on synthetic h×h-pixel images with h ∈ {64,128,256,512}, batch size 64, AdamW with learning rate 2×10−4. Multi-seed runs use n ∈ {3,4,5} depending on resolution; ranges are reported as mean ± standard deviation throughout.

ImageNet effective α. The bandwidth-coherence framework uses the effective power-law exponent α of the natural-image radial DCT spectrum. We center-crop and resize a sample of N = 200 ImageNet-256 images, apply the 2D DCT-II per channel, average power per radial bin (32 bins), and fit log P(k) = blog k + c over bins 1 through 31 (skipping DC and the saturated tail). The result is slope b = −2.818, so the effective α = 2.82 over three decades of clean linear fit.

Evaluation. All ImageNet FID numbers in this paper are FID-50k against the canonical ImageNet256 reference statistics, computed on samples generated by the Heun integrator (50 steps) with classifier-free guidance scale 2.9 and CFG interval [0.1,1.0] [13]. Inception Score is computed on the same 50k images. Toy experiments report the radial-spectrum L1 distance to the empirical data spectrum.

### B Additional Experiments

#### B.1 Hyperparameter sensitivity: the cmin sweep.

A sweep of the operator’s lower cutoff cmin at the canonical toy setting (h = 64, p = 8, α = 2, linear-SF, 1000 epochs) is monotonic: cmin = 0.00 → L1 = 17.43; 0.10 → 14.42; 0.20 → 14.66; 0.30 → 10.95; 0.40 → 10.69. Larger cmin (less aggressive masking) brings SF closer to the baseline at convergence. The sweep confirms that SF’s “loss at convergence” in toys is a continuous function of how restrictive the operator is, not a discrete failure mode.

#### B.2 Schedule shapes at h = 128.

At h = 128 with p = 16 (matching the canonical 64-token count), all six schedule shapes were evaluated with multi-seed support (n = 4). The ordering (Table 10) is the clean opposite of h = 64: schedules that aggressively cut early bands (analytical, t2) win at convergence, schedules that are

Table 9: Configurations of experiments.

— Backbone architecture JiT of Li and He [24] (130M/32, 130M/16, 700M/32) patch size 32 (130M/32, 700M/32) or 16 (130M/16) in-context tokens 32 DCT window matched to patch size

— Training optimizer AdamW, β1 = 0.9, β2 = 0.95 batch size 128 per GPU, 8 GPUs (effective 1024) learning rate 5 × 10−5 (constant after warmup) warmup epochs 5 weight decay 0 EMA decay standard JiT defaults time sampler logit(t) ∼ N(−0.8, 0.82)

— Spectral Forcing schedule f(t) linear (f(t) = t) on ImageNet; analytical at h ≥ 128 in toys cmin, cmax 0.05, 1.0 mask sharpness κ 30 α for analytical 2.0 (toys), 2.82 (when applied to ImageNet, Section A)

— Sampling ODE solver Heun [24] ODE steps 50 time grid linear on [0, 1] CFG scale 2.9 CFG interval [0.1, 1.0] class drop (training) 0.1

Table 10: Schedule comparison at h = 128, p = 16, α = 2, 1000 epochs (n = 4 seeds).

Mean L1 (± std)

Schedule

vs. baseline 33.50

analytical 28.79 ± 1.59 +14% (wins) f(t) = t2 30.05 ± 1.83 +10% (wins) cosine 32.70 ± 1.65 +2% (tied) baseline 33.50 ± 0.79 linear 35.71 ± 1.65 −7% (loses) f(t) = √t 37.30 ± 1.50 −11% (loses)

over-permissive at small t (linear, √t) lose, cosine is roughly tied. The standard deviation across seeds is small (1–2 L1 units) and the analytical-wins gap is much larger than the seed variance. A 2000-epoch sanity check at h = 128 confirms the ordering at single seed: baseline 32.86, linear-SF 35.75, analytical-SF 28.37.

#### B.3 Resolution-scaling: per-seed values at h = 256.

The aggregate of Table 8 hides per-seed variance; Table 11 reports the per-seed values for the h = 256 configuration (p = 32). Both SF schedules beat baseline at ±2σ separation; we did not run additional seeds because the per-seed gap is much larger than the per-seed variance.

#### B.4 SenseNova-U1: GenEval breakdown.

Table 12 reports GenEval at the same SenseNova-U1 [8] checkpoint as Fig. 5 (stage-1 100k steps, non-EMA, 2562, 64 tokens). The overall metric rises from 3.87% to 4.56% (+17.9% relative); per-correct-image and per-correct-prompt percentages move in the same direction. The signal is concentrated in the single-object (+2.81pp, +19.1%) and colors (+1.33pp, +15.6%) categories.

Table 11: Per-seed L1 at h = 256, p = 32, α = 2, 1000 epochs.

Mode Seed 0 Seed 1 Seed 2

baseline 47.03 49.13 49.59 linear-SF 44.51 45.69 48.78 analytical-SF 39.17 42.42 42.98

Table 12: SenseNova-U1 GenEval breakdown. Baseline (BL) versus Linear-SF (SF) at the same stage-1 100k-step checkpoint, 2562, 64 tokens per image, non-EMA weights. Compositional categories (two-object, counting, position, color-attr) score 0% for both methods at this checkpoint and are omitted.

Metric BL + SF ∆ Rel.

overall 3.87% 4.56% +0.69pp +17.9% % correct images 3.57% 4.20% +0.63pp +17.6% % correct prompts 8.50% 8.86% +0.36pp +4.2% single_object 14.69% 17.50% +2.81pp +19.1% colors 8.51% 9.84% +1.33pp +15.6%

The four compositional categories (two-object, counting, position, color-attr) sit at 0% for both baseline and SF at this early checkpoint and are omitted from the table; they require a later-stage checkpoint where the model has begun to produce compositionally-correct outputs at all. Together with the DPG-Bench breakdown of Fig. 5, the GenEval result confirms that the input-side spectral prior transfers to native-VLM text-to-image generation in its predicted favourable regime.

#### B.5 Closed-form denoising limit.

With the rectified-flow interpolant zt = tx + (1 − t)ε, ε ∼ N(0,I), the per-sample velocity target is exactly

x − zt 1 − t

= x − ε. (9)

vtarget =

Reasoning per radial band k, write zt,k = txk +(1−t)εk with εk ∼ N(0,1) and xk ∼ N(0,P(k)), P(k) ∝ k−α. The two contributions to zt,k have typical magnitudes t P(k) (signal) and 1 − t (noise); the closed-form denoising corner is the high-k region where the band is noise-dominated, t P(k) ≪ 1 − t, at t bounded away from 1. There the signal term is negligible against the noise floor, zt,k ≈ (1 − t)εk, so the noise is recoverable from the input, εk ≈ zt,k/(1 − t). Substituting into (9),

zt 1 − t

zt,k 1 − t

. (10)

=⇒ vtarget ≈ −

vtarget,k = xk − εk ≈ −εk ≈ −

The target is then a deterministic function of the input: denoising is pure rescaling and uses nothing about the data distribution.

#### B.6 Algorithmic listings.

- Algorithm 1 defines the cutoff schedule c(t) for each of the schedule shapes considered in Section 3.3.
- Algorithm 2 constructs the soft 2D-DCT radial low-pass M(t) from the scalar cutoff c. Algorithm 3 and Algorithm 4 show how a training step and an Euler sampling step are modified by Spectral Forcing relative to the unmasked JiT recipe; class conditioning and CFG are omitted for brevity.

#### B.7 Qualitative samples.

Fig. 6 shows nine ImageNet classes generated by the same JiT-700M/32 model at 120 epochs, comparing the no-mask baseline (FID 16.46) against Linear-SF (FID 15.15, +8.0%). Each pair fixes both the class label and the sample index, so the only experimental variable is whether the time-conditional 2D-DCT low-pass was active during training and sampling. The classes span birds (lorikeet, indigo bunting), mammals (golden retriever, lion), marine subjects (stingray, coral reef), prepared food (pizza, hot dog), and a structured-scene class (cliff dwelling). Across all nine, SF samples are visibly crisper in fine structure and more class-coherent at this converged budget; the

- Algorithm 1 Spectral Forcing cutoff schedule c(t).

# t in [0, 1]; c_min, c_max: cutoff bounds (defaults 0.05, 1.0) # alpha: spectrum exponent (2.82 on ImageNet); eps = 1e-3 (endpoint guard) # shape in {’linear’, ’analytical’, ’cosine’, ’t_squared’, ’t_sqrt’} def c(t):

if shape == ’linear’: f = t elif shape == ’cosine’: f = 0.5 - 0.5 * cos(pi * t) elif shape == ’t_squared’: f = t ** 2 elif shape == ’t_sqrt’: f = sqrt(t) elif shape == ’analytical’:

f = clip((eps / (1.0 - t).clamp_min(eps)) ** (2.0 / alpha), 0.0, 1.0) return c_min + (c_max - c_min) * f # in [c_min, c_max]

- Algorithm 2 Soft 2D-DCT radial low-pass mask M(t), given c = c(t).

# c: scalar cutoff radius in [0, 1] # H, W: image height, width (e.g. 256) # kappa: soft transition sharpness (default 30) def mask(c):

u = arange(H); v = arange(W) # DCT-II frequency indices U, V = meshgrid(u, v, indexing=’ij’) # (H, W) integer grids r = sqrt(U**2 + V**2) / sqrt(2 * (W-1)**2) # normalized radius in [0, 1] return sigmoid(kappa * (c - r)) # soft low-pass at cutoff c

effect is strongest on textured surfaces (lion mane, coral, pizza topping, cliff dwelling stonework) where the no-mask baseline tends to produce smoother but less specific texture.

### C Limitations

Per-step compute overhead. SF adds one forward and one inverse 2D-DCT per denoising step — approximately 0.5% of per-step compute relative to the unmasked baseline at JiT-130M/32, 2562 (Section 5), with no learned parameters and no additional memory. The DCT is parallelizable on the patch grid and runs in the same kernel as patchify; for budgets where 0.5% matters, the operator is a no-op to remove.

Hyperparameters held fixed across all ImageNet runs. The cutoff bounds (cmin,cmax)=(0.05,1.0) and the mask sharpness κ=30 are reused unchanged across every ImageNet configuration in Table 9, with no per-model, per-budget, or per-resolution tuning. The cmin sweep of Section B.1 indicates the operator is not narrowly tuned at this point; per-configuration tuning would only widen the reported gap to baseline.

Single benchmark and architecture family. We report on ImageNet-256, the canonical classconditional FID/IS benchmark, and on JiT [24], a modern pixel-space rectified-flow Transformer. SF is parameter-free and recipe-agnostic by construction, so applying it to other backbones (e.g., DiT, U-Net) or other diffusion forwards is a configuration change, not an algorithmic change.

- Algorithm 3 Spectral Forcing training step.

# net(z, t): diffusion transformer (e.g., JiT-700M/32) # x: training batch t = sample_t() e = randn_like(x) z = t * x + (1 - t) * e v = (x - z) / (1 - t) z_lp = idct(dct(z) * mask(c(t))) # SF: input-side low-pass x_pred = net(z_lp, t) v_pred = (x_pred - z) / (1 - t) loss = l2_loss(v - v_pred)

- Algorithm 4 Spectral Forcing sampling step.

# z: current samples at t z_lp = idct(dct(z) * mask(c(t))) # SF mask at current t x_pred = net(z_lp, t) v_pred = (x_pred - z) / (1 - t) z_next = z + (t_next - t) * v_pred

[Figure 17]

Figure 6: Qualitative samples on ImageNet-256. JiT-700M/32 at 120 epochs, baseline (B, top row of each block) vs. Linear-SF (SF, bottom row), three sample indices per class, same class label and same sample index per column.

