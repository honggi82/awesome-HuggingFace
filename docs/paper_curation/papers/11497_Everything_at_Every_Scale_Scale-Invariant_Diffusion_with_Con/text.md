# arXiv:2605.26032v1[cs.CV]25May2026

## Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution

Zixin Jessie Chen1∗ † Zhuo Chen134∗ Archer Wang2 Jeff Gore1 William T. Freeman2 Congyue Deng2 Marin Soljaˇci´c13

- 1Department of Physics, Massachusetts Institute of Technology
- 2Department of EECS, Massachusetts Institute of Technology

3NSF AI Institute for Artificial Intelligence and Fundamental Interactions 4Institute for Data, Systems and Society, Massachusetts Institute of Technology {jzxchen,chenzhuo,archerdw,gore,congyued,billf,soljacic}@mit.edu

Signal: decreasing effective resolution

32×32 px

80×80 px

|[Figure 1]|
|---|

|[Figure 2]|
|---|

[Figure 3]

|[Figure 4]|
|---|

|[Figure 5]|
|---|

- (a)

[Figure 6]

Noise: increasing correlation length

- (b)

[Figure 7]

Diffusion: scale-invariant process

- (c)

|[Figure 8]|
|---|

|[Figure 9]|
|---|

(d)

(e)

log log plot

log log plot

RadialSpectrum

RadialSpectrum

Noise

Noise

Data Variance

Data Variance

Signal

Signal

lost resolution

lost resolution

Radial Frequency

Radial Frequency

(g)

(f)

Figure 1: Conceptual illustration of our SKILD on a self-similar fractal image. During the forward process: (a) effective signal resolution decreases; (b) the pixel-space correlation length of the injected noise increases; and (c) for a self-similar field, the process respects the same frequency-space power spectrum across stages. (d-e) A smaller early-time patch is statistically similar to a larger late-time patch. (f-g) Radial power spectrum at corresponding early and later stages. Gray slashed regions indicate modes below the signal-to-noise ratio (SNR) threshold, where resolution is effectively lost.

### Abstract

Creating images from noise is image generation; reconstructing fine details from coarse inputs is super-resolution. Despite their practical differences, both can be understood as reversing information loss across scales. We introduce SKILD, a Scale-invariant K-Space Image Learning Diffusion model that unifies generation and continuous super-resolution within a single unconditional framework. Both natural images and critical physical systems exhibit scale invariance, and we leverage it to design a forward process that attenuates image content from fine to coarse scales while injecting spectrum-matched Gaussian noise, making scale an explicit coordinate of the diffusion dynamics. The same trained reverse process performs generation and continuous super-resolution by varying only the starting timestep: no task-specific architecture, no conditioning branch, no classifier-free guidance, no retraining per scale factor. Empirically, SKILD reaches FID 2.65 and Inception Score 9.63 on unconditional CIFAR-10, performs 2×–8× super-resolution on ImageNet from a single unconditional checkpoint while outperforming conditional models across perceptual metrics, and reconstructs critical Ising models whose connected four-point correlations closely track the ground truth.

∗Equal contribution †Corresponding author

Preprint.

### 1 Introduction

Scales in images have long been a subject of study in computer vision. Across different scales, images share recurring structure. A zoomed natural image still looks natural, and some natural objects are themselves self-similar, with textures, edges, and structures recurring at different scales. Statistically, this regularity is reflected in natural-image power spectra, which follow approximate power laws over wide frequency ranges [1–4], a signature of approximate scale invariance. The same concept has been studied in parallel in physics, where critical systems display similar scale-invariant behavior, made formal by the renormalization group [5, 6]. This physics perspective also points to a natural way of organizing the transformation between an image and pure noise. Can we take advantage of this scale invariance in diffusion? Rather than corrupting all scales at once, one can erase them in order, one scale at a time. Diffusion, framed in this way, becomes a denoising process respecting self-similarity across scales.

Such a denoising process across scales is, by construction, a progressive super-resolution. At each backward step, finer scales are added back, and running the full reverse process from pure noise produces an image scale by scale. This unifies generation and super-resolution into a single framework. Generation from noise is the extreme case of super-resolution in which the input contains no signal at all; super-resolution is the same reverse process initialized from an intermediate state in which coarser scales have survived. Both are reverse coarse-graining problems, distinguished only by where the reverse process begins.

We realize this idea with SKILD (Scale-invariant K-Space Image Learning Diffusion), a diffusion model whose forward process corrupts images one scale at a time, from finest to coarsest. Two design choices make this concrete. First, the forward process attenuates high-frequency content before low-frequency content. Second, the noise added at each step carries the spectrum of the dataset itself rather than being white noise, so the model learns to remove noise that statistically resembles the data it learns to generate. Together, these two choices make every intermediate state a coarse-grained, noisy version of the original image in a self-similar manner.

Our contributions are as follows.

- • We propose SKILD, a scale-invariant diffusion framework that unifies unconditional generation and continuous super-resolution within a single reverse process. A single, unconditional architecture handles both tasks, replacing what would otherwise be a stack of task-specific architectures, conditioning branches, classifier-free guidance, and per-scale retraining.
- • On unconditional CIFAR-10 [7], SKILD is competitive with state-of-the-art diffusion models and achieves the strongest sample quality among frequency-informed diffusion models.
- • One trained SKILD checkpoint performs continuous super-resolution at any factor, which we test on ImageNet [8] between 2× and 8×. At 4× super-resolution on ImageNet-256, the same model outperforms strong diffusion-based conditional super-resolution baselines on multiple perceptual quality metrics.
- • Evaluations on a scientific dataset generated using a critical Ising model show that SKILD reproduces explicit self-similar statistics while a strong diffusion-based conditional super-resolution baseline fails.

### 2 Related Works

Scale invariance and self-similarity. Scale-space theory analyzes images through continuous smoothing and identifies Gaussian convolution as the canonical linear scale-space operator [9– 14]. Natural-image statistics show approximate power-law spectra across scales [1–4, 15], while renormalization group theory describes how distributions transform under coarse-graining and rescaling [5]. These ideas motivate our forward process: attenuation from fine to coarse scales in frequency space, with noise covariance matching the dataset distribution.

Diffusion models across scale and frequency. Diffusion models learn to reverse a noising process [16–18], with various samplers and schedules [19–22]. Several lines of work connect diffusion to multi-scale structure: cascaded and relay models compose resolution-specific conditional stages [23, 24]; other work connects diffusion to renormalization-group flows, optimal transport, or inverse heat dissipation [25–28]. A separate line uses Fourier or wavelet structures to improve controllability,

[Figure 10]

100

|10 1<br><br>| | |Image Image<br><br>|Net-256 Net-128|
|---|---|---|---|---|
|10 7<br><br>10 6<br><br>10 5<br><br>10 4<br><br>10 3<br><br>10 2<br><br>RadialPowerSpectrum| | |CIFAR-|10 (32)|
| | | | | |
| | | | | |

RadialPowerSpectrum

101 102 103

Radial Frequency

(a) Natural-image variances (b) ImageNet-256 variances

Figure 2: Variance power spectra of natural-image datasets. (a) Spectra of CIFAR-10, ImageNet-128, and ImageNet-256 exhibit similar power-law decay over their shared frequency range, indicating approximate scale invariance. (b) Variance of ImageNet-256 computed independently for each color channel (RGB), with a power-law fit recovering the k−2 frequency decay.

efficiency, or inductive bias [29–36]. Recent works have also explored image generation as progressive super-resolution in pixel space, replacing additive noise with structured degradations or multi-scale reconstruction processes [37, 38]. Unlike these approaches, SKILD explicitly utilizes self-similarity in frequency space, where the forward process continuously attenuates image statistics from fine to coarse modes. As a result, a single reverse process supports both unconditional generation and continuous super-resolution without conditioning or guidance.

Super-resolution. Beyond classical priors and feed-forward neural methods [39–45], diffusion-based methods often rely on additional conditioning from low-resolution images [46–52]. SKILD requires no extra conditioning: the low-resolution input is an intermediate state of the model’s own forward process, and the same reverse process completes the missing fine scales.

### 3 Preliminaries and Motivations

Standard diffusion. Diffusion models [17, 18] generate samples by reversing a fixed forward noising process. The forward process gradually transforms a data sample x0 into isotropic Gaussian noise via

xt = √α¯t x0 + √1 − α¯t ϵ, ϵ ∼ N(0,I), (1)

where the schedule α¯t decreases monotonically from 1 at t = 0 to nearly 0 at the end of diffusion. Because the marginals are jointly Gaussian, the reverse-time conditional q(xt−1 | xt,x0) is itself Gaussian and analytically tractable. A neural network ϵθ(xt,t) is trained to predict ϵ given xt by minimizing E ∥ϵ − ϵθ(xt,t)∥2 , and substituting this prediction into the reverse posterior gives a tractable sampling step. Iterative denoising starting from pure noise produces samples from the data distribution.

Scale invariance in physics. Critical physical systems, exemplified by the two-dimensional Ising model at its critical temperature, exhibit scale invariance explicitly. Such systems have no characteristic length scale, so configurations look statistically the same after coordinates are coarse-grained and rescaled by any factor r → br. As a consequence, statistical observables follow power laws of the form O(k) ∝ k−α with universal exponents α [53, 5, 6], since power laws are the only functions invariant under rescaling up to a multiplicative constant. The renormalization group formalizes this picture. Coarse-graining out fine-scale degrees of freedom and rescaling the result acts as a transformation on probability distributions, and the distribution of a critical system is a fixed point of that transformation.

Power-law spectra of natural images. Natural-image distributions show approximate scale invariance. Their radially averaged power spectra, equivalently the variance per Fourier mode of the dataset

S0(k) = E |X0(k)|2 − E X0(k) 2, closely follow k−2 over a wide frequency range [1–3, 54], on average across a dataset. We confirm this on the datasets used in our experiments. Figure 2 shows the radially averaged variance power spectra for CIFAR-10, ImageNet-128, and ImageNet-256 computed in the discrete cosine transform (DCT) space [55], with the exact transform given in Appendix C. The spectra agree over their shared frequency range and differ mainly near finite-resolution cutoffs. We fit the radial variance with

S0(k) = C(k2 + k20)−a, (2)

where k0 regularizes the k → 0 limit. The fits recover the k−2 scaling, and Table C.3 lists the fitted parameters.

Toward scale-invariant diffusion. The observations above raise a natural design question. Given that natural images and critical physical systems share a hierarchy of structure across scales, what would a diffusion forward process look like if it were designed to respect this hierarchy rather than treating all scales on the same footing? We propose such a scale-invariant process in the frequency space in the next section.

### 4 Scale-Invariant Diffusion in Frequency Space

#### 4.1 Formulation

Forward process. Let X0(k) denote the DCT coefficients of an image and let S0(k) be the empirical variance spectrum estimated in Section 3. We define the continuous forward marginal

2λ(t)/2 ⊙ X0(k)

X(k,t) = e−k

, ϵt ∼ N(0,I), (3)

+ 1 − e−k2λ(t) ⊙ S0(k)ϵt

signal

noise

where ⊙ denotes Hadamard product. The schedule λ(t) is a scalar function that monotonically increases in t. As t grows, e−k

2λ(t)/2 narrows in frequency space, so high-frequency modes are attenuated before low-frequency ones. The noise prefactor is chosen so that the forward marginal preserves the per-mode covariance S0(k) in expectation at every t, and converges to N(0,S0(k)) as the signal term vanishes. In pixel space, the same process convolves the signal with a Gaussian kernel and adds spatially correlated noise whose correlation length grows with t, reflecting the progressive removal of scale structures.

Discretization. All experiments use a DDPM [17] discretization of Eq. (3). For 0 = t0 < ··· < tN = 1, let α¯n(k) = e−k

2λn, αn = α¯n/α¯n−1, and βn = 1 − αn. Then

Xn(k) = √α¯n ⊙ X0(k) + √1 − α¯n ⊙ S0(k)ϵn, ϵn ∼ N(0,I). (4) The one-step transition has the same form with α¯n replaced by αn. Since all covariances are diagonal in frequency space, the reverse posterior is Gaussian: q(Xn−1 | Xn,X0) = N(µq,S0β˜n), with

βn√S0 √1 − α¯n

βn(1 − α¯n−1) 1 − α¯n

1 √αn

β˜n =

ϵn . (5) Ancestral sampling proceeds by

Xn −

, µq(n,Xn) =

Xn−1 = µq(n,Xn) + S0β˜n ⊙ ϵn. (6) The full DDPM and stochastic differential equation (SDE) derivation appear in Appendices A and B.

#### 4.2 Training target and numerical cutoffs We train an ϵ-prediction network with the loss

0,ϵ ∥ϵ − ϵθ(n,Xn)∥22 . (7)

L = En,X

Two implementation details make the finite-resolution process stable. First, very small αn(k) values can cause large reverse updates for high-frequency modes, so we floor them at 10−6 in the ancestral sampler. Second, the zero mode would otherwise have no attenuation or noise because the Gaussian signal filter leaves it untouched. Therefore, we introduce a low-frequency cutoff kc, and use max(∥k∥,kc) as the schedule for modes with ∥k∥ ≤ kc. This preserves the algebra above while properly handles the low-frequency limit, where scale-invariance is affected by finite size effect.

#### 4.3 Schedules and effective resolution

The schedule λ(t) controls how frequencies are attenuated with time. We evaluate two schedules, named by how the damping cutoff in k moves with time. The log-linear schedule λ(t) = t · 10λ

i+(λf−λi)t moves it roughly uniformly on a log scale, and the linear schedule λ(t) = θ t/(λf(1−

t) + λi)2 moves it roughly uniformly on a linear scale. The multiplicative t ensures λ(0) = 0. Among the parameters, λi primarily sets the high-frequency, early-time behavior, while λf, kc, and θ primarily set the low-frequency, late-time behavior; all four jointly shape the full schedule.

The notion of time-evolving cutoff in k gives super-resolution a direct interpretation. For a chosen SNR threshold,

α¯n(k) 1 − α¯n(k)

, (8)

SNRn(k) =

the modes above the threshold define an effective resolution. Starting the reverse process from a timestep where the effective resolution is zero gives image generation; starting from a timestep whose surviving signals correspond to a lower-resolution input gives super-resolution. Because λ(t) is continuous before discretization and can be densely sampled in implementation, the effective resolution varies continuously along the schedule, yielding a continuum of super-resolution factors from a single trained model (Figure E.12).

#### 4.4 Connection to scale-space theory and renormalization group

Equation (3) extends the vanilla scale-space operation [10, 11] to frequency space with noise. In pixel space, multiplying DCT modes by exp[−k2λ(t)/2] amounts to Gaussian smoothing at scale λ(t), the same operator that appears in linear scale-space theory. Crucially, our additional noise term turns the deterministic smoothing into a stochastic coarse-graining process whose final covariance matches the dataset variance spectrum.

The same equation also conceptually resembles a renormalization group (RG) coarse-graining step, where short-distance degrees of freedom are discarded before long-distance structures. We do not claim that our method is an exact RG transformation; rather, we use the RG as analogy: if a dataset exhibits approximate scale invariance, a reverse model trained on the scale-ordered forward process should learn how fine scales are distributed conditioned on coarse scales. The critical-Ising experiment in Section 5.5 tests this idea in a setting with known scale-invariant structure.

### 5 Experiments

We evaluate whether the same frequency-space diffusion process can serve as an unconditional image generator and a continuous super-resolution model. We test SKILD in three settings: unconditional image generation on CIFAR-10, 2×–8× continuous super-resolution on ImageNet-128 and 256, and a scientific benchmark on the critical two-dimensional Ising model, probing scale invariance directly through four-point correlations. SKILD is competitive with or outperforms strong baselines in each setting.

#### 5.1 Setup

Architecture. All reported models use a score U-Net backbone from the NCSN++ family [18]. Exact channel counts, depths, and attention configurations are in Appendices D and E.

Data. We use CIFAR-10 [7] and ImageNet [8] as released. Critical-Ising configurations are sampled on a 128 × 128 square lattice using the Wolff cluster algorithm [56]; data-generation details are in Appendix F.

Noise schedule. We test both the log-linear and linear schedules on CIFAR-10 and the linear schedule on ImageNet and Ising experiments. All experiments use N = 1000 timesteps; exact schedule parameters are in Appendices D, E, and F.

Training. We train with AdamW and use an exponential moving average of weights at sampling time. Full hyperparameters and compute details are in Appendices D, E, and F.

Evaluation. For CIFAR-10 we report FID [57] and Inception Score (IS) [58] on 50K generated samples. For ImageNet super-resolution we report PSNR, SSIM [59], LPIPS [60], MUSIQ [61], and CLIPIQA [62] from the last checkpoint, on a random 3K-image subset of the ImageNet validation set, following the protocols of [48, 49]. The Ising super-resolution experiment is evaluated on 1K samples from the last checkpoint by a connected four-point correlation, a statistical-physics observable that probes how accurately the model reproduces Ising model structures across scales (Section 5.5).

#### 5.2 Effective resolution protocol

For super-resolution, we choose the reverse starting timestep by the SNR defined in Section 4. We use threshold 0.1 in all ImageNet and Ising super-resolution experiments. This corresponds to applying an effective low-pass filter in the frequency domain, where modes below the effective-resolution cutoff retain signal while higher-frequency modes are attenuated and dominated by noise. At sampling time, the model starts from the exact forward marginal of the paired high-resolution (HR) image at the chosen timestep, then reverses to t0. This protocol turns super-resolution into a partial reverse diffusion problem rather than a conditional generation problem.

We validate the effective-resolution interpretation by comparing the surviving signal at the chosen timestep with standard bicubic down-up sampling. The MSE and PSNR values in Table E.5 show that SNR of 0.1 produces low-resolution (LR) inputs close to conventional 4× or 8× degradations while preserving consistency with the forward diffusion process.

#### 5.3 Unconditional CIFAR-10 generation

- Table 1: Unconditional CIFAR-10 sample quality. Best results within each model group are in bold.

Frequency-agnostic model Frequency/scale-informed model DDPM [17]

Model

Cold diff.* (deblur) [37]

Cold diff.* (sr) [37]

DDPM++ [18]

NCSN++ [18]

EDM [21]

EqualSNR [32]

WaveDiff [29]

IHDM [26]

DCTdiff [31]

SKILD (ours)

FID ↓ 3.17 2.78 2.20 1.97 80.08 152.76 13.63 4.01 18.96 5.02 2.65 IS ↑ – 9.46 9.64 9.89 – – – – – 7.70 9.63*

* The [37] results are generated from highly-degraded images which still contains some signal, instead from scratch.

[Figure 11]

Figure 3: Uncurated samples of generated images on CIFAR-10; more in Appendix G (Figure G.17).

- Table 1 compares unconditional CIFAR-10 generation. Figure 3 shows uncurated samples that SKILD generates on CIFAR-10. SKILD is competitive with the state-of-the-art models and achieves the best FID and IS among the frequency or scale-informed models listed, using the linear schedule shown in

- Figure 4(a).

0.0 0.2 0.4 0.6 0.8 1.0

t

0.0

0.2

0.4

0.6

0.8

1.0

α

(a)

DDPM (Linear)

DDPM (Cosine)

100 150 200 250 300 400 Training steps [×103]

2.8

3.0

3.2

3.4

3.6

3.8

FID

Best FID 2.65

FID (b)

IS

[Figure 12]

0

20

40

60

80

100

120

DiagonalFrequencyk

9.2

9.4

9.6

9.8

10.0

10.2

IS

Best IS 10.03

Figure 4: (a) Linear schedule with best FID and IS on CIFAR-10 generation. Lower-frequency modes are attenuated later than higher-frequency ones. (b) Mode collapse with training steps. While IS continues to improve with training, FID converges early on and worsens at later times.

Ablation studies. We conduct several sets of ablation studies and discuss two of them briefly here.

- Figure 4(b) shows the model over training. A mode collapse appears with training course: FID reaches its best value before the final checkpoint, while IS continues to improve. We interpret this as evidence that low-frequency reconstruction remains the bottleneck for generation on object-centric datasets like CIFAR-10. The limitation section discusses this point directly.

In Table D.4, we verify the robustness of SKILD on image generation against a broad range of schedules in the log-linear and linear families. Most schedules reach FID below or near 5 and all reach IS near or above 9 within 400K training steps. This indicates that SKILD is not tuned to a single fragile schedule, albeit the convergence speed is schedule-dependent.

More details of ablations can be found in Appendix D.III, including FID and IS convergence over training compared to common pixel-space diffusion schedules (Figure D.11), different network predictions, the effectiveness of second-moment sampler, potential of reducing number of diffusion steps, and different numerical cutoffs.

#### 5.4 ImageNet super-resolution

- Table 2: 4× super-resolution quality on the ImageNet-Test [48, 49]. Best and second-best results among quality metrics are highlighted in bold and underline.

Model PSNR↑ SSIM↑ LPIPS↓ CLIPIQA↑ MUSIQ↑ # Param. (M) GAN-based

BSRGAN [43] 24.42 0.659 0.259 0.581 54.697 16.70 SwinIR [44] 23.99 0.667 0.238 0.564 53.790 16.70 RealESRGAN [45] 24.04 0.665 0.254 0.523 52.538 28.01

Conditional diffusion-based

LDM-30 [63] 24.49 0.651 0.248 0.572 50.895 113.60 LDM-15 [63] 24.89 0.670 0.269 0.512 46.419 113.60 ResShift [48] 25.01 0.677 0.231 0.592 53.660 118.59 SinSR [49] 24.56 0.657 0.221 0.611 53.357 118.59 IRSDE [51] 24.48 0.602 0.304 0.513 45.382 137.20 DDRM [50] 25.56 0.674 0.471 0.372 24.746 552.80 I2SB [52] 26.76 0.730 0.206 0.489 53.936 552.80

Unconditional diffusion SKILD (Ours) 24.10 0.683 0.186 0.612 59.226 121.12

LR Bicubic SKILD (Ours) Ground Truth

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

- Figure 5: 4× super-resolution samples on ImageNet-256. The model is initialized from a 64 × 64 low-resolution forward state and reconstructs high-frequency details through the reverse process.

- Table 2 reports 4× super-resolution quality on ImageNet. All conditional baselines receive the low-resolution input through an explicit conditioning path, most of which additionally use class labels or classifier-free guidance. SKILD uses no conditioning of any kind: it starts from the corresponding forward marginal and runs the same reverse process used for unconditional generation. Despite this simplicity in design, the 256-resolution model achieves the best LPIPS, CLIPIQA, and MUSIQ among the compared methods and the second-best SSIM. PSNR favors methods with higher raw pixel accuracy, while metrics that emphasize human perception favor our method. Two super-resolution samples are shown in Figure 5, with more in Appendix G.

8× SR 16×16

7× SR 18×18

6× SR 21×21

5× SR 26×26

4× SR 32×32

3× SR 42×42

2× SR 64×64

Ground Truth 128×128

|[Figure 21]|
|---|

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

- (a)
- (b)

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- Figure 6: Continuous super-resolution on ImageNet-128. (a) Low-resolution inputs at effective resolutions 16 × 16 through 64 × 64, corresponding to 8× through 2× super-resolution factors. (b) High-resolution reconstructions produced by the same checkpoint from each effective-resolution starting state. A continuum of super-resolution factors is accessible from a single trained model.

A single trained ImageNet model accommodates a continuum of super-resolution factors by varying only the starting timestep. Figure 6 shows reconstructions at factors from 2× to 8× produced by the same checkpoint, and Figure E.12 plots how the effective resolution decreases continuously with t.

5.5 Scientific benchmark

|[Figure 37]|
|---|

SKILD (ours)

|[Figure 38]|
|---|

Ground truth

[Figure 39]

- Figure 7: (a-b) Benchmark of four-point correlator accuracy. Our reconstruction closely tracks the ground truth, while SR3 shows a clear deviation. (c) SKILD-reconstructed critical Ising field sample compared to the ground truth.

Natural images are approximately scale-invariant only after averaging over many scenes. Critical physical systems let us ask a stricter question: can a model reconstruct missing fine scales while preserving observables that define the scale-invariant law? We test this on the prototypical twodimensional Ising model at criticality. Despite its simplicity–placing a spin si ∈ {−1,+1} on each lattice site, with nearest neighbors preferring to align–the Ising model serves a foundational role in areas across statistical mechanics, combinatorics, and computational complexity theory. At its critical temperature, the correlation length diverges, the distribution becomes statistically self-similar under RG coarse-graining, and the continuum limit is described by a conformal field theory [53, 5, 6]. Previous works also have applied neural networks to Ising super-resolution and inverse RG [64, 65].

This setting provides a more precise benchmark of scale-invariance than perceptual realism. A visually plausible spin configuration can still have the wrong connected correlations, the wrong response functions, or the wrong universality-class signatures.

Evaluation: connected four-point correlator. We evaluate a connected four-point correlator, equivalently a fourth-order joint cumulant, over the four corner spins of square patches at multiple side lengths. The join cumulant subtracts all pairwise contributions, isolating non-Gaussian dependence that cannot be inferred from the mean, variance, or two-point correlation alone. Higher-order correlations are central observables in critical systems, so matching them across scales is a stronger test than matching visual texture or pixel-level distortion. Data generation, paired evaluation, and the correlator estimation are detailed in Appendix F.

Results. We super-resolve from a 32 × 32 effective-resolution starting state to a 128 × 128 criticalIsing field. Figure 7 shows that SKILD’s reconstructed four-point correlator closely tracks the ground

truth at every patch size, while SR3 [46], a strong diffusion-based conditional super-resolution model, deviates significantly from the ground truth.

### 6 Conclusions

We introduce SKILD, a scale-invariant frequency-space diffusion model whose forward process is a stochastic coarse-graining operator: fine modes are damped before coarse ones, and the injected noise carries the dataset spectrum. Aligning diffusion with the scale structure of data makes scale an explicit coordinate of the generative process. Unconditional generation and super-resolution then become different starting points of the same reverse trajectory, eliminating task-specific conditioning and guidance. Empirically, SKILD reaches FID 2.65 and IS 9.63 on unconditional CIFAR-10, supports continuous 2×–8× super-resolution from a single ImageNet checkpoint, and reconstructs critical-Ising fields whose connected four-point correlations closes tracks the ground truth.

Discussion. Our method shifts the modeling burden from conditional mappings to the design of a scale-invariant diffusion process. Rather than learning a separate low-to-high-resolution mapping, the model learns a single reverse trajectory over scales, with low-resolution inputs as intermediate forward states. This removes task-specific conditioning or guidance. In the mean time, low-frequency generation becomes a central bottleneck: errors in coarse structure early in the reverse chain propagate and constrain later high-frequency generation. Our results reflect this trade-off: super-resolution benefits from accurate coarse initialization, while unconditional generation remains sensitive to lowfrequency modeling. The framework also suggests an evaluation criterion for self-similarity beyond perceptual quality, namely whether fine-scale details remain statistically similar with coarse-scale structure, which is particularly relevant for scientific applications. We provide such an instance through super-resolution experiments on critical-Ising fields.

Limitations and future work. The current model opens a vast range of directions for future work. (i) Currently, sampling requires 1000 ancestral steps; faster samplers or tailored solvers for mode-dependent schedules are a natural next step. (ii) Unconditional generation remains sensitive to low-frequency structure generation, and improvements there should stabilize global structure without sacrificing fine detail. (iii) Our super-resolution protocol uses exact forward marginals as low-resolution inputs; extending to real-world degradations such as unknown camera and compression pipelines is an important direction. (iv) The Ising experiment establishes scale-invariant diffusion on one critical system; extending to additional physical systems and higher-order observables would broaden the scientific benchmark. (v) Our model uses off-the-shelf neural network. New architecture designs tailored to our model could significantly improve its performance while furthering understanding of network designs for diffusion models.

Broader impacts. Unifying generation and super-resolution into a single reverse process can simplify deployment pipelines and reduce reliance on task-specific models. The same capability has dual-use risks: super-resolution can hallucinate plausible but incorrect high-frequency content, with consequences in forensics, medical imaging, and other sensitive settings where outputs require validation. The scientific-data setting illustrates a complementary benefit: scale-invariant models can be evaluated against known physical laws, a style of evaluation we expect to be useful wherever multi-scale structure carries scientific meaning.

Code and reproducibility. The code for reproducing the main results and data of this paper is available at https://github.com/JazzyCH/SKILD.

### Acknowledgment

The authors acknowledge support from the National Science Foundation under Cooperative Agreement PHY-2019786 (The NSF AI Institute for Artificial Intelligence and Fundamental Interactions) and the MIT Generative AI Impact Consortium. This work is supported by the Toyota Research Institute University 3.0 Program, and the Department of the Air Force Artificial Intelligence Accelerator under Cooperative Agreement No. FA8750-19-2-1000. ZJC is in part supported by the Kurt Forrest Foundation Fellowship and the Henry Kendall Fellowship. ZC is in part supported by the MathWorks Fellowship and the Henry Kendall Fellowship. AW is in part supported by the National Science Foundation Graduate Research Fellowship. CD is in part supported by the Tayebati Postdoctoral Fellowship. The authors also acknowledge the National Artificial Intelligence Research Resource

(NAIRR) Pilot Program, the DeltaAI advanced computing and data resource at the National Center for Supercomputing Applications (supported by NSF Award OAC-2320345 and the State of Illinois), and Lambda Inc. for providing compute resources.

### References

- [1] Field, D. J. Relations between the statistics of natural images and the response properties of cortical cells. J. Opt. Soc. Am. A 4, 2379–2394 (1987). URL https://opg.optica.org/ josaa/abstract.cfm?URI=josaa-4-12-2379.
- [2] Ruderman, D. L. The statistics of natural images. Network: computation in neural systems 5, 517 (1994).
- [3] van der Schaaf, A. & van Hateren, J. Modelling the power spectra of natural images: Statistics and information. Vision Research 36, 2759–2770 (1996). URL https://www. sciencedirect.com/science/article/pii/0042698996000028.
- [4] Simoncelli, E. P. & Olshausen, B. A. Natural image statistics and neural representation. Annual review of neuroscience 24, 1193–1216 (2001).
- [5] Wilson, K. G. & Kogut, J. The renormalization group and the epsilon expansion. Physics Reports 12, 75–199 (1974).
- [6] Belavin, A. A., Polyakov, A. M. & Zamolodchikov, A. B. Infinite conformal symmetry in two-dimensional quantum field theory. Nuclear Physics B 241, 333–380 (1984).
- [7] Krizhevsky, A. & Hinton, G. Learning multiple layers of features from tiny images. Tech. Rep., University of Toronto (2009).
- [8] Deng, J. et al. ImageNet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, 248–255 (2009).
- [9] Witkin, A. Scale-space filtering: A new approach to multi-scale description. In ICASSP ’84. IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 9, 150–153

(1984).

- [10] Koenderink, J. J. The structure of images. Biological Cybernetics 50, 363–370 (1984). URL https://doi.org/10.1007/BF00336961.
- [11] Lindeberg, T. Scale-space theory: A basic tool for analyzing structures at different scales. Journal of applied statistics 21, 225–270 (1994).
- [12] Lindeberg, T. Feature detection with automatic scale selection. International journal of computer vision 30, 79–116 (1998).
- [13] Lowe, D. G. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision 60, 91–110 (2004). URL https://doi.org/10.1023/B: VISI.0000029664.99615.94.
- [14] Burt, P. J. & Adelson, E. H. The laplacian pyramid as a compact image code. In Readings in computer vision, 671–679 (Elsevier, 1987).
- [15] Olshausen, B. A. & Field, D. J. Natural image statistics and efficient coding. Network: Computation in Neural Systems 7, 333 (1996). URL https://doi.org/10.1088/0954-898X/7/ 2/014.
- [16] Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N. & Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, 2256–2265 (pmlr, 2015).
- [17] Ho, J., Jain, A. & Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020).

- [18] Song, Y. et al. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020).
- [19] Nichol, A. Q. & Dhariwal, P. Improved denoising diffusion probabilistic models. In International conference on machine learning, 8162–8171 (PMLR, 2021).
- [20] Dhariwal, P. & Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems 34, 8780–8794 (2021).
- [21] Karras, T., Aittala, M., Aila, T. & Laine, S. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems 35, 26565–26577 (2022).
- [22] Chen, T. On the importance of noise scheduling for diffusion models. arXiv preprint arXiv:2301.10972 (2023).
- [23] Ho, J. et al. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research 23, 1–33 (2022).
- [24] Teng, J. et al. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350 (2023).
- [25] Cotler, J. & Rezchikov, S. Renormalizing diffusion models. arXiv preprint arXiv:2308.12355

(2023).

- [26] Rissanen, S., Heinonen, M. & Solin, A. Generative modelling with inverse heat dissipation. arXiv preprint arXiv:2206.13397 (2022).
- [27] Masuki, K. & Ashida, Y. Generative diffusion model with inverse renormalization group flows. arXiv preprint arXiv:2501.09064 (2025).
- [28] Sheshmani, A., You, Y.-Z., Buyukates, B., Ziashahabi, A. & Avestimehr, S. Renormalization group flow, optimal transport, and diffusion-based generative model. Physical Review E 111, 015304 (2025).
- [29] Guth, F., Coste, S., De Bortoli, V. & Mallat, S. Wavelet score-based generative modeling. Advances in neural information processing systems 35, 478–491 (2022).
- [30] Phung, H., Dao, Q. & Tran, A. Wavelet diffusion models are fast and scalable image generators. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10199–10208 (2023).
- [31] Ning, M. et al. Dctdiff: Intriguing properties of image generative modeling in the dct space. arXiv preprint arXiv:2412.15032 (2024).
- [32] Falck, F. et al. A fourier space perspective on diffusion models. arXiv preprint arXiv:2505.11278

(2025).

- [33] Yu, H., Luo, H., Yuan, H., Rong, Y. & Zhao, F. Frequency autoregressive image generation with continuous tokens. arXiv preprint arXiv:2503.05305 (2025).
- [34] Moser, B. B., Frolov, S., Raue, F., Palacio, S. & Dengel, A. Waving goodbye to low-res: A diffusion-wavelet approach for image super-resolution. In 2024 International Joint Conference on Neural Networks (IJCNN), 1–8 (IEEE, 2024).
- [35] Gao, X., Xu, Z., Zhao, J. & Liu, J. Frequency-controlled diffusion model for versatile text-guided image-to-image translation. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, 1824–1832 (2024).
- [36] Friedrich, P., Wolleb, J., Bieder, F., Durrer, A. & Cattin, P. C. Wdm: 3d wavelet diffusion models for high-resolution medical image synthesis. In MICCAI workshop on deep generative models, 11–21 (Springer, 2024).
- [37] Bansal, A. et al. Cold diffusion: Inverting arbitrary image transforms without noise. Advances in Neural Information Processing Systems 36, 41259–41282 (2023).

- [38] Tian, K., Jiang, Y., Yuan, Z., Peng, B. & Wang, L. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024).
- [39] Freeman, W. & Pasztor, E. Learning low-level vision. In Proceedings of the Seventh IEEE International Conference on Computer Vision, vol. 2, 1182–1189 vol.2 (1999).
- [40] Dong, C., Loy, C. C., He, K. & Tang, X. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence 38, 295–307 (2015).
- [41] Lim, B., Son, S., Kim, H., Nah, S. & Mu Lee, K. Enhanced deep residual networks for single image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, 136–144 (2017).
- [42] Wang, X. et al. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, 0–0 (2018).
- [43] Zhang, K., Liang, J., Van Gool, L. & Timofte, R. Designing a practical degradation model for deep blind image super-resolution. In Proceedings of the IEEE/CVF international conference on computer vision, 4791–4800 (2021).
- [44] Liang, J. et al. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, 1833–1844 (2021).
- [45] Wang, X., Xie, L., Dong, C. & Shan, Y. Real-esrgan: Training real-world blind super-resolution with pure synthetic data. In Proceedings of the IEEE/CVF international conference on computer vision, 1905–1914 (2021).
- [46] Saharia, C. et al. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence 45, 4713–4726 (2022).
- [47] Li, H. et al. Srdiff: Single image super-resolution with diffusion probabilistic models. Neurocomputing 479, 47–59 (2022).
- [48] Yue, Z., Wang, J. & Loy, C. C. Resshift: Efficient diffusion model for image super-resolution by residual shifting. Advances in neural information processing systems 36, 13294–13307 (2023).
- [49] Wang, Y. et al. Sinsr: diffusion-based image super-resolution in a single step. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 25796–25805 (2024).
- [50] Kawar, B., Elad, M., Ermon, S. & Song, J. Denoising diffusion restoration models. Advances in neural information processing systems 35, 23593–23606 (2022).
- [51] Luo, Z., Gustafsson, F. K., Zhao, Z., Sjölund, J. & Schön, T. B. Image restoration with mean-reverting stochastic differential equations. arXiv preprint arXiv:2301.11699 (2023).
- [52] Liu, G.-H. et al. I2SB: Image-to-image schrödinger bridge. arXiv preprint arXiv:2302.05872

(2023).

- [53] Onsager, L. Crystal statistics. i. a two-dimensional model with an order-disorder transition. Physical Review 65, 117–149 (1944).
- [54] Hyvrinen, A., Hurri, J. & Hoyer, P. O. Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. (Springer Publishing Company, Incorporated, 2009), 1st edn.
- [55] Ahmed, N., Natarajan, T. & Rao, K. R. Discrete cosine transform. IEEE Transactions on Computers C-23, 90–93 (1974).
- [56] Wolff, U. Collective monte carlo updating for spin systems. Physical Review Letters 62, 361–364 (1989).
- [57] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. & Hochreiter, S. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems, vol. 30 (2017).

- [58] Salimans, T. et al. Improved techniques for training GANs. In Advances in Neural Information Processing Systems, vol. 29 (2016).
- [59] Wang, Z., Bovik, A. C., Sheikh, H. R. & Simoncelli, E. P. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing 13, 600–612 (2004).
- [60] Zhang, R., Isola, P., Efros, A. A., Shechtman, E. & Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 586–595 (2018).
- [61] Ke, J., Wang, Q., Wang, Y., Milanfar, P. & Yang, F. MUSIQ: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5148–5157 (2021).
- [62] Wang, J., Chan, K. C. K. & Loy, C. C. Exploring CLIP for assessing the look and feel of images. In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, 2555–2563 (2023).
- [63] Rombach, R., Blattmann, A., Lorenz, D., Esser, P. & Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 10684–10695 (2022).
- [64] Efthymiou, S., Beach, M. J. S. & Melko, R. G. Super-resolving the ising model with convolutional neural networks. Physical Review B 99, 075113 (2019).
- [65] Shiina, K., Mori, H., Tomita, Y., Lee, H. K. & Okabe, Y. Inverse renormalization group based on image super-resolution using deep convolutional networks. Scientific Reports 11, 9617 (2021).

### A DDPM formulation

This appendix derives the DDPM posterior used in the main text and lists the training targets supported by our implementation. Figure A.8 shows a forward trajectory on a natural image, illustrating how high-frequency content is removed before low-frequency content under the scale-invariant schedule.

[Figure 40]

Figure A.8: Forward trajectory of SKILD on a natural image. (a), (b), and (c) show the signal, noise, and full forward process, respectively. (d) shows the radial power spectrum evolution measured from the trajectory. High frequencies are suppressed first, while low-frequency content persists until late timesteps. Gray regions indicate modes below the SNR threshold, where resolution is effectively lost.

- A.I Posterior derivation The forward marginal in main text Eq. (4) can also be written as a one-step Markov transition,

Xn = √αn ⊙ Xn−1 + βn ⊙ S0 ϵn, ϵn ∼ N(0,I), (A.9) with α¯n = e−k

2λn, αn = α¯n/α¯n−1, βn = 1 − αn, and α¯0 = 1. Note that α¯n = ni=1 αi. When X0 is known, the reverse posterior q(Xn−1 | Xn,X0) is exact. By Bayes’ theorem,

q(Xn−1 | Xn,X0) ∝ q(Xn | Xn−1)q(Xn−1 | X0), (A.10)

where we used the Markov property and dropped the Xn−1-independent factor q(Xn | X0). Eq. (A.9) gives

q(Xn | Xn−1) = N(Xn; √αn ⊙ Xn−1, S0βn) ∝ N(Xn−1; Xn/√αn, S0βn/αn), (A.11) where the second form completes the square in Xn−1. The forward marginal at n − 1 is

q(Xn−1 | X0) = N Xn−1; α¯n−1 ⊙ X0, S0(1 − α¯n−1) . (A.12)

Multiplying these two Gaussians and using the standard product formula v˜ = (v1−1 + v2−1)−1, m˜ = v˜(m1/v1 + m2/v2) gives

q(Xn−1 | Xn,X0) = N µq, S0β˜n , (A.13a)

βn(1 − α¯n−1) 1 − α¯n

β˜n =

, (A.13b)

√α¯n−1 βn 1 − α¯n

√αn (1 − α¯n−1) 1 − α¯n

Xn. (A.13c) Reverse sampling with the exact posterior reads

µq =

X0 +

Xn−1 = µq(n,Xn,X0) + S0β˜n ⊙ ϵn. (A.14)

#### A.II Training targets

At sampling time, X0 is unknown and is replaced by a network estimate X0,θ inserted into Eq. (A.13). Our implementation supports four prediction targets: noise ϵ, data X0, an unwhitened frequencyspace velocity w, and a whitened velocity v. The main results use ϵ-prediction; the other targets follow the same posterior algebra and are reported as diagnostic ablations.

ϵ-prediction. Minimizing the MSE against the noise variable, L = En,X

0,ϵ ∥ϵ − ϵθ(Xn,n)∥22 , (A.15) inverts main text Eq. (4) to obtain

√S0 ϵn,θ √α¯n

√1 − α¯n ⊙

Xn −

, (A.16) and substituting into Eq. (A.13) gives the ϵ-form of the posterior mean used at sampling,

X0,θ =

βn√S0 √1 − α¯n

1 √αn

ϵn,θ . (A.17)

Xn −

µθ(n,Xn) =

In finite precision we floor very small αn(k) values during sampling, as described in the main text, to avoid unstable divisions in high-frequency modes.

X0-prediction. Minimizing the MSE against the data,

0,ϵ ∥X0 − X0,θ(Xn,n)∥22 , (A.18) substitutes X0,θ directly into the posterior mean,

L = En,X

√α¯n−1 βn 1 − α¯n

√αn (1 − α¯n−1) 1 − α¯n

Xn. (A.19)

µθ(n,Xn) =

X0,θ +

w-prediction. Let a = √α¯n and b = √1 − α¯n. The forward marginal and the velocity target are

Xn = a ⊙ X0 + b ⊙ S0 ϵn, (A.20a) wn = a ⊙ S0 ϵn − b ⊙ X0. (A.20b)

Given a prediction wn,θ, the data and colored-noise estimates follow without division (using a2+b2 = 1):

X0,θ = a ⊙ Xn − b ⊙ wn,θ, (A.21a)

zn,θ = b ⊙ Xn + a ⊙ wn,θ, (A.21b) where zn,θ estimates √S0 ⊙ ϵn. The corresponding loss is

0,ϵ ∥wn − wn,θ(Xn,n)∥22 . (A.22) v-prediction. The whitened velocity divides w by √S0,

Lw = En,X

vn = a ⊙ ϵn − b ⊙ X0/ S0. (A.23)

For w and v, the loss can optionally be weighted by the local frequency shell α¯n−1 − α¯n. In our ablations, direct prediction of the colored noise √1 − α¯n ⊙

√S0 ϵn matches the CIFAR-10 FID and IS of ϵ-prediction but converges more slowly, while X0- and w-prediction converge more slowly still and were not used for the reported metrics.

### B SDE formulation

#### B.I Forward and reverse SDEs

Following the score-matching formulation of [18], we write a forward stochastic differential equation (SDE) and its time-reverse counterpart as

forward: dX = f(X,t)dt + g(t) ⊙ dw, (B.24a) reverse: dX = f(X,t) − g2(t) ⊙ ∇X log pt(X) dt + g(t) ⊙ dw¯ . (B.24b)

The whole process is computed in frequency space. We use the main text notation: X for DCT coefficients, S0 for the empirical spectrum, and ϵ ∼ N(0,I) for normalized Gaussian noise. We suppress explicit k arguments where no ambiguity arises.

The continuous-time SDE can be obtained either by differentiating Eq. (3) or, more compactly, by starting from a linear SDE ansatz, integrating it exactly, and matching the resulting mean and variance to the desired forward marginal. We follow the second route, which determines f(X,t) and g(t) mode by mode.

Since X is evolved under a Gaussian filter in Eq. (3), the SDE ansatz should be linear in X. For each mode k, write

dx = −a(t)xdt + b(t)dw. (B.25)

Define the integrating factor m(t) ≡ exp 0 t a(u)du , so that dm = a(t)m(t)dt. The Itô product rule gives

d(mx) = mdx + xdm + d[m,x] = mb(t)dw, (B.26) since m has finite variation and so d[m,x] = 0. Integrating and dividing by m(t), with m(0) = 1,

t

t

t

xt = e−

e−

0 a(u) du x0

s a(u) du b(s)dws

. (B.27)

+

0

signal

noise

Matching E[xt] in this expression to the forward marginal Eq. (3) gives

t

0 a(u) du E[x0] =! e−k

2λt/2 E[x0] =⇒ a(t) = 12 k2 λ˙(t). (B.28) Matching Var[xt] requires Var[xt] = S0 at every t when Var[x0] = S0, the dataset spectrum:

e−

t

t

t

s adu b2(s)ds =! S0. (B.29) Differentiating with respect to t and substituting,

e−2

e−2

0 adu S0 +

0

2λt) + b2(t) = S0 k2 λ˙t e−k

2λt =⇒ b2(t) = k2 λ˙t S0. (B.30) Restoring all k modes, we obtain

−k2λ˙t S0 (1 − e−k

- f(X,t) = −12 k2 λ˙(t)Xt, (B.31a)

- g(t) = k2 λ˙(t)S0, (B.31b)

and the forward and reverse SDEs read forward: dXt = −21 k2 λ˙(t)Xt dt + k2 λ˙(t)S0 ⊙ dw, (B.32a) reverse: dXt = −21 k2 λ˙(t)Xt − k2 λ˙(t)S0 ⊙ ∇X log pt(X) dt + k2 λ˙(t)S0 ⊙ dw¯ .

(B.32b)

#### B.II Score function and training objectives

To determine the training target for a score network, we compute the marginal score. Let At = e−k

2λ(t)/2 and Dt = S0(1 − e−k2λ(t)) denote the signal and noise prefactors. The forward conditional is N(X; At ⊙ X0, D2t), and the marginal is

pt(X) = p0(X0)N(X; At ⊙ X0, D2t)dX0. (B.33) Differentiating under the integral and dividing by pt(X) gives the score

∇X log pt(X) = E D−t 2 ⊙ (At ⊙ X0 − X) X = D−t 2 ⊙ (At ⊙ E[X0 | X] − X), (B.34) which gives the Tweedie identity

E[X0 | X] = A−t 1 ⊙ X + D2t ⊙ ∇X log pt(X) . (B.35)

The standard score-matching objective is

0,ϵ ω(t) ⊙ ∥sθ(X,t) + D−t 1 ⊙ ϵt∥22 ,

0,ϵ ω(t) ⊙ ∥sθ(X,t) − ∇X log pt(X)∥22 = Et,X

L = Et,X

(B.36) where sθ is the score network and ω(t) is a per-mode weighting function.3 The regression target −D−t 1 ⊙ ϵt is the anisotropic analogue of the variance-preserving (VP) diffusion target. Without a low-frequency cutoff, the DC mode k = 0 would be unchanged because neither drift nor noise act on it; in practice we replace ∥k∥ by max(∥k∥,kc) for small modes, so the same equations apply with a nonzero low-frequency cutoff.

The choice of ω(t) determines the prediction target. Three natural choices recover the targets discussed in Appendix A.

2λ(t)) and writing sθ = −D−t 1 ⊙ ϵθ, the loss reduces to

ϵ-net weight. Setting ω(t) ∝ D2t = S0(1 − e−k

0,ϵ ∥ϵθ(X,t) − ϵt∥22 . (B.37)

L = Et,X

X0-net weight. Setting ω(t) ∝ A−t 2⊙D4t and writing sθ = D−t 2⊙(At⊙X0,θ −X) via Tweedie, the loss reduces to

0,ϵ ∥X0,θ(X,t) − X0∥22 . (B.38)

L = Et,X

KL-minimizing weight. Setting ω(t) ∝ g2(t) minimizes the Kullback-Leibler (KL) divergence between the true and learned reverse dynamics over the whole path: a score error e(k) produces drift er-

ror g(t)⊙e(k), and for SDEs sharing the same diffusion the path-KL scales as 12 g2(t)⊙E[e(k)2]dt. Modes and times where the reverse drift amplifies score errors then receive proportionally more

weight during training.

#### B.III Samplers

The standard score-matching samplers [18] include the Euler-Maruyama (EM) sampler, the Ordinary Differential Equation (ODE) sampler from probability flows, and the predictor-corrector (PC) sampler. Our experiments use ancestral DDPM sampling (Eq. (A.14)); the SDE samplers below are reference forms.

EM sampler. On a decreasing time grid 1 = tN > tN−1 > ··· > t0 = 0 with ∆tn = tn−1 −tn < 0, discretizing Eq. (B.32b) gives

Xn−1 = Xn+ −12 k2 λ˙n Xn − k2 λ˙n S0 ⊙ sθ(Xn,tn) ∆tn+ k2 λ˙n S0⊙ −∆tn ϵn. (B.39) ODE sampler. For any SDE of the form Eq. (B.24a), the probability flow ODE

dX = f(X,t) − g2(t) ⊙ ∇X log pt(X) dt (B.40)

matches the SDE marginals pt(X). Solving in reverse time yields the same target distribution as the reverse SDE, with the Brownian term dropped:

Xn−1 = Xn + −12 k2 λ˙n Xn − k2 λ˙n S0 ⊙ sθ(Xn,tn) ∆tn. (B.41)

PC sampler. The PC sampler refines an EM prediction Xpn−1 via a few iterations of a score-based MCMC corrector at the same time level. A preconditioned Langevin update that leaves pt(X) invariant is

dXτ = B(t) ⊙ ∇X log pt(X)dτ + 2B(t) ⊙ dwτ, (B.42) which classically uses B(t) = I but admits an anisotropic, time-dependent choice given the modedependent diffusion. Discretizing with step ηn gives the corrector update

Xi+1 = Xi + B(tn) ⊙ sθ(Xn,tn)ηn + 2B(tn)ηn ⊙ ϵn, (B.43) and one full PC step combines the EM predictor with K Langevin corrections initialized at Xi=0 = Xpn−1. The corrector reduces the discretization error of the predictor.

3Strictly, the loss should reference ∇X log pt(X | X0). The two objectives differ by an sθ-independent constant via the Fisher identity E[∇X log pt(X | X0) | X] = ∇X log pt(X), so they share the same minimizer. This is the basis of denoising score matching.

### C Scale invariance and power laws in nature and physics

- C.I Dictionary between physics and natural images

We provide a brief analogy between field theory and natural images that motivates treating them in the same framework.

A Monte Carlo sample of a scalar field theory can be encoded directly as an image: each pixel at lattice site r stores the value of the field variable at that site. The full distribution over field configurations is then represented by the dataset of all such images. Conversely, a natural image can be viewed as a single sample from a three-channel scalar field theory, with each pixel storing a field value, the resolution playing the role of the inverse lattice spacing, and the image size playing the role of the physical system size. The infinite-resolution limit corresponds to the continuum limit of the field, and the infinite-size limit corresponds to the thermodynamic limit. The basic dictionary is

- • pixel value ↔ field variable;
- • image resolution ↔ inverse lattice spacing (continuum limit at infinite resolution);
- • image size ↔ system size (thermodynamic limit at infinite size).

A critical field theory is scale-invariant under renormalization-group coarse-graining and rescaling. The corresponding question for images is whether a sufficiently large and high-resolution naturalimage dataset is approximately scale-invariant under the analogous operation of low-pass filtering followed by zooming in. The variance spectra in Section 3 indicate that, in expectation across a dataset, the answer is approximately yes.

- C.II Power-law benchmark in frequency space

We specify the DCT, IDCT, and frequency normalization used throughout the paper, then summarize the per-dataset variance fits.

DCT and IDCT conventions. The forward transform is the type-II DCT and the inverse is the type-III IDCT. For an image x ∈ RH×W,

W−1

H−1

4 HW

xi,j cos H π i + 12 u cos W π j + 12 v , (C.44) with inverse

Xu,v =

j=0

i=0

γuγvXu,v cos H π i + 12 u cos W π j + 12 v , (C.45) where γ0 = 1/2 and γk>0 = 1.

xi,j =

u,v

Frequency normalization. We define the frequency vector by k = (πu,πv), so that mode indices are spaced uniformly by π and the magnitude ∥k∥ ranges in [0,√2π(H − 1)] for an H × H image. Two properties motivate this convention. First, the 4/(HW) amplitude factor in the DCT keeps the maximum of the radial power spectrum at the same order of magnitude across resolutions, as seen in Figure C.9 and main text Figure 2. Second, the spacing-fixed convention means that an image of higher resolution extends the spectrum to higher k rather than rescaling existing modes; in the infinite-resolution limit, the discrete grid fills out the continuum.

Variance fits. Figure C.9 complements the ImageNet-256 fit shown in main text Figure 2(b) with the corresponding fits for CIFAR-10 and ImageNet-128. All three recover an approximate k−2 decay across the shared frequency range; deviations at the highest frequencies reflect the finite-resolution cutoff and shrink as resolution grows. Table C.3 summarizes the fitted parameters.

### D Supplementary materials for CIFAR-10 experiments

#### D.I Architecture and training

CIFAR-10 generation uses a score U-Net backbone from the NCSN++ family [18] with discrete DDPM positional embeddings and 8 residual blocks. We scan the base channel count from 128 to

[Figure 41]

[Figure 42]

(a) CIFAR-10 variance (b) ImageNet-128 variance

Figure C.9: Variance spectra for (a) CIFAR-10 and (b) ImageNet-128, computed independently for each color channel (RGB). Both power-law fits recover the k−2 frequency decay reported in previous work. The deviations at the highest frequencies in (a) reflect finite-resolution limitations and shrink in (b) as resolution increases.

Table C.3: Power-law fits of variance power spectra for the natural-image datasets. Dataset C k20 a

CIFAR-10 0.9100 ± 0.0182 1.9406 ± 0.0481 1.0513 ± 0.0021 ImageNet-128 0.9281 ± 0.0016 1.5708 ± 0.0030 1.0590 ± 0.0002 ImageNet-256 0.9322 ± 0.0004 1.5708 ± 0.0008 1.0598 ± 0.0000

256 and report the best configuration. Models are trained with AdamW for 400K steps with batch size 128, learning rate 2 × 10−4, weight decay 0, and EMA rate 0.999. Checkpoints are saved every 20K steps. Each CIFAR-10 model is trained on a single H100 or GH200 GPU.

#### D.II Noise schedules

We describe the two schedule families tested on CIFAR-10 and motivate their forms. The schedule λ(t) controls how the damping cutoff in k moves with time: a frequency mode is effectively

suppressed when k2λ(t) ≈ θ for some threshold θ, so the damping front is kd(t) = θ/λ(t).

Log-linear schedule. In the continuum limit (∆k → 0), a damping that is uniform on a logarithmic frequency scale respects scale invariance: equal time intervals remove equal logarithmic ranges of modes. The corresponding schedule is

i+(λf−λi)t, (D.46) which gives kd(t) ∝ 10−(λ

λ(t) = 10λ

f−λi)t/2, log-linear in t.

Linear schedule. Real datasets are at finite resolution and finite size, neither at the continuum limit nor the thermodynamic limit, so it is also reasonable to damp k linearly with time:

λ(t) =

under which kd(t) moves linearly with t.

θ (λf(1 − t) + λi)2

, (D.47)

Boundary fix. Both schedules above have λ(0) > 0 and λ˙(0) > 0, so high-frequency modes are damped abruptly at t = 0. This discontinuity removes fine details before the model can learn them. We mitigate it by multiplying both schedules by t, ensuring λ(0) = 0 and a smooth onset. The schedules used in all experiments are

i+(λf−λi)t, (D.48) linear: λ(t) =

log-linear: λ(t) = t · 10λ

θ t (λf(1 − t) + λi)2

. (D.49)

Best schedules on CIFAR-10. Figure D.10 shows the best-performing schedule from each family. The exact parameters of the best linear schedule, used to report the FID in the main text, are θ = 5.0, λi = 137.7294, λf = 1.57, kc = 3, and N = 1000. The best log-linear schedule uses λi = −3.75, λf = −2.0, kc = 31.2, and N = 1000.

[Figure 43]

###### (a)

###### (b)

1.0

1.0

120

DDPM (Linear)

DiagonalFrequencyk

0.8

0.8

DDPM (Cosine)

100

0.6

0.6

80

α

α

60

0.4

0.4

40

0.2

0.2

20

0.0

0.0

0

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

t

t

Figure D.10: Best (a) log-linear and (b) linear schedules for CIFAR-10 generation. Different diagonal frequency modes are attenuated independently over time, with higher-frequency modes decaying before lower-frequency ones. The linear and cosine DDPM schedules are shown for reference; they are scalar in frequency space and attenuate all modes uniformly.

- D.III Ablation studies

- D.III.1 Mode collapse

In every CIFAR-10 run we observe a characteristic mode collapse: FID reaches an early minimum and then rises while IS continues to improve. For the linear schedule of Figure D.10(b), FID converges to 2.65 at step 160K with IS 9.63, then degrades to 2.82 at 200K (IS 9.77), 2.98 at 250K (IS 9.79), 3.26 at 300K (IS 9.98), and 3.87 at 400K (IS 10.03). Main text Figure 4(b) plots these trajectories.

We attribute this to instability in the low-frequency portion of the diffusion. Late in the forward process, small changes in the surviving low-frequency modes produce large overall color shifts in the noised image, visible in Figures 1 and A.8. This makes it harder for the model to fix the global color tone and class-level structure of an image, and the consequence is most pronounced on object-centric datasets such as CIFAR-10. The pattern is consistent with image fidelity continuing to improve (IS rising) while the marginal class distribution drifts away from the dataset (FID degrading). The effect is strongest at large kc, where many low-frequency modes are bundled together at late times.

The low-frequency limit is not strictly scale-invariant, so we expect this regime to need a different treatment. Plausible remedies include chaining the SKILD reverse process with a small VAE or with a standard pixel-space diffusion model that handles the very-low-frequency modes. The effect should also weaken on higher-resolution and less object-centric natural-image datasets.

- D.III.2 Schedule robustness

Diffusion models are sensitive to noise schedules [22, 19]. Table D.4 reports a sweep over λi, λf, kc, and θ in both the log-linear and linear families. Most schedules reach FID near or below 5 within 400K training steps, and all reach IS at or above 9. Convergence speed depends on the schedule, but the final converged scores are not strongly schedule-dependent, indicating that SKILD does not require a finely tuned schedule for stable training.

- D.III.3 Convergence against pixel-space schedules

- Figure D.11 compares one of our linear schedules against standard pixel-space DDPM and DDIM schedules. To isolate the effect of the schedule from the mode-collapse instability discussed above,

we used a low-frequency-stable variant with θ = 2.0, λi = 137.7294, and λf = 22.3, and used the forward marginal of the ground truth as the sampling input for diagnostics. The frequency-space schedule converges stably over 400K steps, with FID below the linear DDPM and the two DDIM baselines and IS above all pixel-space schedules.

- D.III.4 Additional ablations

Timestep sampler. The second-moment timestep sampler of [20] reduced the training loss faster but did not improve the final FID or IS, consistent with low-frequency recovery being the limiting factor.

Number of diffusion steps. Reducing the number of diffusion steps from 1000 to 500 left FID and IS nearly unchanged. A naive 100-step reduction degraded both metrics. Faster sampling is therefore feasible, but requires solver or distillation work tailored to the mode-dependent schedule.

- Table D.4: Schedule robustness on CIFAR-10. Each row reports one frequency-space diffusion schedule. Metrics are shown for 200K and 400K training steps. Across the completed 400K evaluations, most schedules reach FID below or near 5, and all reach Inception Score near or above 9, indicating that SKILD is not tuned to a single fragile schedule. Convergence speed varies by schedule. A dash denotes an unused parameter or an unavailable run.

Schedule parameters FID ↓ IS ↑ kc λi λf θ 200K 400K 200K 400K Log-linear schedule

3.0 −4.25 0.0 – 6.77 5.53 8.72 8.98 3.0 −4.25 0.4 – 6.80 5.63 8.63 8.84 3.0 −4.25 0.8 – 7.24 5.66 8.56 8.85 3.0 −3.75 0.0 – 5.65 4.47 8.77 9.07 3.0 −3.75 0.4 – 6.10 4.83 8.74 9.04 3.0 −3.75 0.8 – 6.56 5.17 8.73 8.99 3.0 −3.25 0.0 – 5.72 4.84 8.78 9.07 3.0 −3.25 0.4 – 6.02 4.77 8.77 9.06 3.0 −3.25 0.8 – 6.30 4.77 8.66 8.98

13.4 −4.25 −1.4 – 5.71 4.62 8.81 9.15 13.4 −4.25 −1.0 – 5.97 4.88 8.80 9.06 13.4 −4.25 −0.6 – 6.40 5.35 8.75 8.95 13.4 −3.75 −1.4 – 4.90 4.40 8.95 9.22 13.4 −3.75 −1.0 – 5.08 4.55 8.83 9.10 13.4 −3.75 −0.6 – 5.57 4.62 8.76 8.99 13.4 −3.25 −1.4 – 4.62 5.29 8.97 9.16 13.4 −3.25 −1.0 – 5.21 4.84 8.87 9.13 13.4 −3.25 −0.6 – 5.45 4.71 8.73 9.05 31.2 −4.25 −2.2 – 5.40 4.63 8.86 9.14 31.2 −4.25 −2.0 – 5.76 4.74 8.82 9.05 31.2 −4.25 −1.6 – 6.00 4.93 8.78 8.97 31.2 −3.75 −2.2 – 4.53 4.74 9.15 9.30 31.2 −3.75 −1.6 – 5.39 4.64 8.91 9.05 31.2 −3.25 −2.2 – 4.58 5.74 9.07 9.17 31.2 −3.25 −2.0 – 4.76 5.41 8.97 9.15 31.2 −3.25 −1.6 – 5.30 4.65 8.81 9.08

Linear schedule

3.0 137.7294 1.57 3.0 5.42 4.41 9.06 9.39 3.0 137.7294 1.57 5.0 4.89 4.27 9.12 9.29 3.0 137.7294 1.57 7.0 6.75 5.33 9.39 9.55

13.4 137.7294 12.0 3.0 5.61 5.14 9.23 9.29 13.4 137.7294 12.0 5.0 4.61 4.99 9.05 9.52 13.4 137.7294 12.0 7.0 4.45 – 9.07 – 31.2 137.7294 29.5 3.0 7.63 7.44 9.18 9.41 31.2 137.7294 29.5 5.0 4.43 5.01 9.12 9.51 31.2 137.7294 29.5 7.0 4.76 5.96 9.02 9.25

Numerical cutoffs. We tested numerical cutoffs of αn in {5 × 10−1,10−4,10−8}. The smaller cutoffs gave similar convergence with 10−6; 5 × 10−1 degraded sample quality. This is consistent with the low-frequency modes needing room to vary without being over-constrained.

Prediction target. Among the four prediction targets defined in Appendix A, ϵ-prediction (used for the reported metrics) converged fastest. Prediction of the full colored noise √1 − α¯n ⊙

√S0 ϵn matched the same FID and IS but required more training steps, while X0- and w-prediction converged slower still.

### E ImageNet super-resolution experiment supplements

Architecture and training. ImageNet super-resolution uses a score U-Net backbone from the NCSN++ family [18] with 6 residual blocks per resolution, attention at resolutions 32, 16, and 8,

6.5

###### (a)

###### (b)

9.6

6.0

5.5

9.4

5.0

9.2

FID

IS

4.5

9.0

DDIM-50 (Cosine)

4.0

DDIM-100 (Cosine)

8.8

DDPM (Linear)

3.5

DDPM (Cosine)

Ours (IR-fixed)

8.6

3.0

100 150 200 250 300 350 400 Training steps [×103]

100 150 200 250 300 350 400 Training steps [×103]

Figure D.11: FID and IS convergence comparison across schedules. (a) The FID of our low-frequency-fixed schedule converges stably over 400K training steps, outperforming the linear DDPM and two DDIM schedules. (b) The IS of our schedule leads the DDPM and DDIM schedules from 100K to 400K training steps.

- Table E.5: Agreement between forward-diffusion effective-resolution signals and bicubic down-up images across SNR thresholds. MSE is reported in units of 10−4 and PSNR in dB. At SNR = 0.1, the MSE for all degradation pipelines is within O(10−4) and the PSNR exceeds 30.

SNR=1 SNR=0.5 SNR=0.1 SNR=0.05 SNR=0.01 SNR=0.005

Dataset MSE PSNR MSE PSNR MSE PSNR MSE PSNR MSE PSNR MSE PSNR 4× ImageNet-256 15.9 27.99 10.5 29.79 4.48 33.49 3.84 34.15 4.53 33.44 5.33 32.73 4× ImageNet-128 21.5 26.67 14.0 28.53 4.87 33.12 3.55 34.50 3.51 34.55 4.23 33.74 8× ImageNet-128 29.8 25.26 19.7 27.05 7.09 31.49 5.15 32.88 4.87 33.12 5.78 32.38

and channel multipliers (1,1,2,2,2,2). We train with AdamW for up to 500K steps with batch size 256, learning rate 10−4, weight decay 10−5, and EMA rate 0.9999, and report metrics from the last checkpoint. The ImageNet-256 model is trained on 8 H100 GPUs; the ImageNet-128 models are trained on a single H100 or GH200 GPU.

Effective-resolution validation. To compare effective-resolution forward states with conventional image resizing, we generate a low-resolution reference by bicubic-downsampling the denormalized image and then upsampling it to the original resolution with bicubic interpolation (antialias=true). This bicubic reference validates the SNR-defined effective-resolution interpretation used for superresolution experiments.

Table E.5 reports the per-dataset MSE and PSNR between the surviving signal at each SNR threshold and the matching bicubic-degraded image. At the threshold SNR = 0.1 used in all SR experiments, the MSE is within O(10−4) for every degradation pipeline and the PSNR exceeds 30 dB, indicating that the SNR-defined low-resolution input closely matches a conventional 4× or 8× degradation while using the exact forward-process marginals.

Super-resolution schedules. The super-resolution experiments use linear schedules with N = 1000 steps:

4 × ImageNet-256: θ = 9.0, λi = 1132.9352, λf = 550.8723, kc = 0; 4 × ImageNet-128: θ = 9.0, λi = 564.2461, λf = 275.4361, kc = 0; 8 × ImageNet-128: θ = 5.0, λi = 564.2461, λf = 102.6489, kc = 0.

Figure E.12 shows the corresponding effective-resolution paths.

F Critical Ising super-resolution details

This appendix details the experiment in Section 5.5. We generate critical Ising configurations and evaluate connected four-point correlations across a range of scales rather than image-quality metrics.

- F.I Data We use the ferromagnetic two-dimensional Ising model on a square lattice,

 β

 , si ∈ {−1,+1}, (F.50)

P(s) ∝ exp

sisj

i,j being neighbors

EffectiveResolution(OneSide)

128

| | |
|---|---|
| | |
| | |

8× SR 4× SR Figure E.12: Effective resolution against

time for the ImageNet-128 super-resolution experiments. The 8× experiment ends at an effective resolution of 16, and the 4× experiment ends at 32. For both, the maximum resolution change in a single step is ∆R = 1, so each model supports continuous super-resolution up to its respective final effective resolution.

96

64

32

16

0

0.0 0.2 0.4 0.6 0.8 1.0

t

at the exact critical inverse temperature

√

1 Tc

- 1

- 2

2). (F.51)

βc =

=

log(1 +

Configurations are sampled with the Wolff cluster algorithm [56], a Markov chain Monte Carlo method for spin systems that avoids the critical slowing down of local spin-flip dynamics. We use 128 × 128 binary spin fields with values in {−1,+1} and periodic boundary conditions.

One transition of the Wolff Markov chain constructs and flips a same-spin cluster. For our ferromagnetic model with coupling J = 1, the bond probability at criticality is

p = 1 − exp(−2βc). A transition consists of the following steps:

- 1. Choose a seed site uniformly at random and let s⋆ be its spin. Initialize the cluster and the active frontier to contain only this seed site.
- 2. Given the current frontier, inspect nearest-neighbor sites with spin s⋆ that are not already in the cluster. If a candidate site touches k ∈ {1,2,3,4} frontier sites, add it to the cluster with probability 1 − (1 − p)k. This is equivalent to independently activating each bond from a frontier site with probability p.
- 3. Set the newly added sites as the next frontier and repeat the previous step until the frontier is empty.
- 4. Flip every spin in the completed cluster: si ← −si.

We run several independent chains in parallel, each initialized with independent random spins in {−1,+1}128×128. Every chain is thermalized for 2,000 Wolff transitions. After thermalization, we save the current configuration from each chain whenever all of them have accumulated at least 2L2 flipped spins since the last save. This is equivalent to two lattice sweeps of cluster updates between consecutive saves.

We use 90,000 samples for training and 1,000 held-out samples for testing. We do not use a separate validation set for model selection, since no hyperparameters are tuned on the Ising benchmark. The held-out samples are shared between SKILD and SR3, so all reported correlation differences reflect the models and not the test inputs.

#### F.II Ground-truth forward initialization

We initialize the reverse process from the exact forward marginal of a high-resolution spin field. Given σ0(r), we take its DCT to obtain σ0(k). For a chosen timestep n0, the initialization is

(k) ⊙ S0(k)ϵ(k), ϵ(k) ∼ N(0,I), (F.52)

(k) ⊙ σ0(k) + 1 − α¯n

σn

(k) = α¯n

0

0

0

which matches the forward marginal of the frequency-space DDPM used during training. The schedule is chosen so that at n0 = 1000 the low-frequency modes corresponding to roughly a 32×32 effective resolution remain largely intact, while higher modes are dominated by the noise term.

Reverse sampling from this initialization produces a reconstruction conditioned on the low-frequency content of the paired held-out sample.

The diffusion uses N = 1000 steps with a linear schedule, θ = 9.0, and kc = 0. The variance spectrum S0 is fit by a power law,

S0(k) = C(k2 + k02)−a, a = 0.811056, C = 0.26641, k02 = 3.0, (F.53) with a, C, and k02 fit on the Ising training set.

#### F.III Training hyperparameters

SKILD uses the same NCSN++ backbone as in the ImageNet SR experiments, with one input and output channel, base width 128, channel multipliers (1,1,2,2,2,2), six residual blocks per resolution, and attention at resolutions 32, 16, and 8. We train with ϵ-prediction using AdamW with learning rate 10−4, β1 = 0.9, weight decay 10−5, 1,000 warmup steps, gradient clipping at 1.0, batch size 256, microbatch size 128, and EMA rate 0.9999. Training uses mixed precision and a uniform timestep sampler. We train for 100,000 optimization steps and use the EMA checkpoint at this final step for all reported results. The Ising model is trained on a single H100 or GH200 GPU.

#### F.IV Connected four-point correlation

For a fixed side length, each correlation estimate is the empirical mean over all square patches in each image, including all translations and symmetry-equivalent orientations, and is then averaged across images. Let s00,s01,s10,s11 denote the four corner spins of a patch. The full four-corner statistic is G4 = E[s00 s01 s10 s11 ], the edge two-point correlation is Ca = E[s00 s01 ], and the diagonal two-point correlation is Cb = E[s00 s11 ]. The connected four-point correlation reported in the main text is

κ4 = G4 − 2Ca2 − Cb2, (F.54) the fourth-order joint cumulant of the four corner spins. It removes the contribution that pairwise correlations alone explain and tests whether the model reproduces the non-Gaussian critical structure of the field. We compute κ4 at side lengths {1,2,4,8,16,32,64}.

#### F.V Paired evaluation

Each generated sample is paired with the held-out spin field that produced its low-frequency initialization. The left panel of Figure 7 shows paired ground-truth and reconstruction examples; the right panel reports the paired correlation comparison. The baseline is SR3 [46], a conditional diffusion model trained to upsample 32 × 32 low-resolution Ising fields to their 128 × 128 ground truth.

For uncertainty estimates we use a paired bootstrap over the 1,000 held-out samples: a single bootstrap index matrix is sampled and applied to all compared methods at every side length, preserving the sample-wise dependence induced by conditioning on the same low-frequency input. We use 1,000 bootstrap resamples and report 99% percentile confidence intervals for the curves in Figure 7.

### G Additional image samples

This appendix collects additional super-resolution samples on ImageNet-128 and ImageNet-256 across the super-resolution factors reported in the main text, along with uncurated CIFAR-10 generation samples.

[Figure 44]

##### Figure G.13: Additional 4× super-resolution sample comparisons on ImageNet-256.

[Figure 45]

##### Figure G.14: Additional 4× super-resolution samples on ImageNet-256.

[Figure 46]

##### Figure G.15: Additional 4× super-resolution samples on ImageNet-128.

[Figure 47]

##### Figure G.16: Additional 8× super-resolution samples on ImageNet-128.

[Figure 48]

##### Figure G.17: Uncurated samples of generated images on CIFAR-10.

