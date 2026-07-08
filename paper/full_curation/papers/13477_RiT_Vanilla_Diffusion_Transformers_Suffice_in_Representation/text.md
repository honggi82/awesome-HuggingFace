# arXiv:2605.21981v1[cs.CV]21May2026

## RiT: Vanilla Diffusion Transformers Suffice in Representation Space

Le Zhang1∗ Ning Mang2 Aishwarya Agrawal1,3 1Mila – Québec AI Institute, UdeM 2Utrecht University 3Canada CIFAR AI Chair

### Abstract

Flow matching with x-prediction—regressing the clean data point rather than the ambient velocity—is known to exploit low-dimensional manifold structure effectively in pixel space [18]. We ask whether a pretrained representation space, while containing a low-dimensional data manifold of comparable intrinsic dimensionality, offers a distribution more favorable for flow-matching learning. Comparing pixel, SD-VAE, and DINOv2 features along four geometric axes, we find that pixel and DINOv2 share nearly identical intrinsic dimensionalities (both dˆ≈33) yet DINOv2 exhibits 7.3× higher effective rank, 35× better covariance conditioning, 11.5× lower excess kurtosis, and 1.7× lower on-manifold interpolation error; SD-VAE latents are consistently intermediate, indicating that the advantage stems from representation-learning objectives rather than mere compression. These statistical properties render the flow-matching regression well-conditioned and remove the need for the specialized prediction heads or Riemannian transport used by prior DINOv2 diffusion methods. We propose the Representation Image Transformer (RiT): a vanilla Diffusion Transformer trained by x-prediction on frozen DINOv2 features, augmented only by a dimension-aware noise schedule and joint [CLS]patch modeling. On ImageNet 256×256, RiT attains FID 1.45 without guidance and 1.14 with classifier-free guidance, outperforming DiTDH-XL with 19% fewer parameters (676M vs. 839M). The resulting ODE is efficiently solvable at coarse discretizations: with classifier-free guidance, 5 Heun steps already reach FID 2.0 and 10 steps reach 1.25, without distillation or consistency training. Code at https://github.com/lezhang7/RiT.

### 1 Introduction

Flow matching [19, 7] learns a velocity field that transports Gaussian noise to data along linear paths. When data concentrates near a low-dimensional manifold, x-prediction—parameterizing the network to output the clean data point zˆ0 rather than the ambient-space velocity—places the regression target on that manifold, as demonstrated by JiT [18] in pixel space. A natural question is whether a pretrained representation space, while containing a data manifold of comparable intrinsic dimensionality, offers a distribution more favorable for learning the flow-matching velocity field.

Comparing pixel, SD-VAE [23], and DINOv2 [21] features along four geometric axes, we find that pixel and DINOv2 share nearly identical intrinsic dimensionalities (both dˆ≈ 33) yet embed this manifold differently relative to N(0,I). The pixel manifold is anisotropic, has strongly non-Gaussian per-coordinate marginals, and admits linear chords that traverse low-density regions. DINOv2 features exhibit near-isotropic variance, near-Gaussian per-coordinate marginals [38], and approximately on-manifold linear interpolants. These are marginal properties, not joint ones: DINOv2 features still concentrate on a dˆ≈ 33-dimensional manifold, but each coordinate’s transport toward N(0,I) is short and well-conditioned.

∗Corresponding author: le.zhang@mila.quebec

Preprint.

Section 2 quantifies these gaps: DINOv2 attains 7.3× higher effective rank, 35× better covariance conditioning, 11.5× lower excess kurtosis, and 1.7× lower on-manifold interpolation error than pixels. SD-VAE latents fall consistently between the two, indicating that the advantage arises from representation-learning objectives rather than compression alone. These distributional advantages coexist with a DINOv2-specific pathology at off-manifold intermediate states: per-token LayerNorm pins ∥z∥≈

√

d, so linear flow-matching paths zt = tz0+(1−t)ϵ traverse ambient regions the encoder never outputs, and the v-target at such zt acquires a large radial component.

The prevailing response to this radial ambiguity has been architectural. RAE [44] handles it with a specialized wide prediction head (DDT [37]) atop v-prediction, alongside a ViT decoder that maps DINOv2 features back to pixels. Concurrent work [16] calls this phenomenon geometric interference and replaces the Euclidean transport with Riemannian Flow Matching on the norm-concentration sphere. Both modifications add complexity to either the architecture or the transport path.

We take a target-side alternative: x-prediction. Under this parameterization, the network regresses zˆ0, which lies on the data manifold by construction, so the radial ambiguity is resolved at the network’s

output (which targets z0 on the manifold) rather than at its input (where zt remains off-manifold). The reparameterization itself is not new [18]; its effectiveness here stems from the combination with DINOv2’s isotropic per-coordinate variance and near-Gaussian marginals, which render the denoising regression zt → z0 well-conditioned enough for a vanilla DiT. We instantiate this combination as the Representation Image Transformer (RiT) (Section 3): a vanilla Diffusion Transformer trained by x-prediction flow matching in representation space, augmented by a dimension-aware noise schedule and joint [CLS]-patch modeling. As DiT operates on the SD-VAE latent space, RiT operates on a representation space provided by a frozen encoder–decoder; we use RAE’s [44] frozen DINOv2 encoder and ViT decoder. RiT thus models the high-dimensional DINOv2 feature distribution directly, without adapting the encoder for generation. On ImageNet 2562, RiT attains FID 1.45 without guidance and 1.14 with classifier-free guidance, outperforming DiTDH-XL with 19% fewer parameters. The resulting ODE converges in few Heun steps, yielding 5-step FID 2.0 and 10-step FID 1.25 (guided) without distillation or consistency training (Section 4.3).

### 2 The Geometry of Representation Spaces for Flow Matching

Cumul.var.(%)

75

50

Pixel

25

SD-VAE DINOv2

0

| | | | | | |SD-VA DINOv|E 2| |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

Pixel

10 1

Varianceratio

10 2

10 3

10 4

0 100 200 300 400 500

Principal components

(a) PCA spectrum

Pixel

SD-VAE DINOv2

- 100

- 101

- 102

- 103

()t

0.0 0.2 0.4 0.6 0.8 1.0

t

(b) Optimization conditioning

0.016

0.014

Recon.MSE

0.012

Pixel interp

DINOv2 interp

0.010

0.008

0.006

0.0 0.2 0.4 0.6 0.8 1.0

Interpolation t

(c) On-manifold interpolation

- Figure 1: Manifold analysis across Pixel, SD-VAE, and DINOv2. (a) PCA spectrum: cumulative variance (top) and per-component variance on log scale (bottom); flatter decay indicates more uniform

spread. (b) Condition number κ(Σt) along the transport path; DINOv2 stays 35× better conditioned than Pixel at t=0.9. (c) Interpolation reconstruction MSE; Pixel stays off-manifold while DINOv2 remains close throughout.

The manifold hypothesis—that data concentrates on a low-dimensional surface—holds regardless of representation. What differs across representations is how favorably this manifold is positioned relative to N(0,I), and thus whether transport paths are short and the ODE is efficiently solvable in few steps. We characterize each representation along four complementary axes—intrinsic dimensionality (the manifold’s true degrees of freedom), effective rank (how uniformly variance is spread across directions), marginal Gaussianity (per-coordinate similarity to N(0,1)), and onmanifold linear interpolation (whether linear chords stay near data)—each predicting a distinct mechanism that makes flow matching easier or harder. We quantify these on three spaces over 10,000 ImageNet images: (i) raw pixels (3×256×256, D=196,608), (ii) DINOv2-Base features (768×16×16, D=196,608), and (iii) SD-VAE latents (4×32×32, D=4,096)—the pretrained VAE

Metric Pixel SD-VAE DINOv2

Mean |κ| 0.923 0.467 0.114 Median |κ| 0.958 0.228 0.083

- |κ| < 0.5 0.0% 74.2% 98.7%
- |κ| < 1.0 70.6% 86.7% 99.7%

Table 1: Marginal Gaussianity. Excess kurtosis across three representation spaces.

| | | | | |
|---|---|---|---|---|
| | | |Pixel (med | |=0.96) SD-VAE (med | |=0.23) DINOv2 (med | |=0.08)| |
| | | | | |

4

Density

2

0

3 2 1 0 1 2 3

Excess kurtosis

- Figure 2: Kurtosis distribution. DINOv2 marginals concentrate tightly around κ=0 (Gaussian); SD-VAE is intermediate; pixels deviate strongly.

[Figure 1]

PixelDINOv2PixelDINOv2

[Figure 2]

[Figure 3]

[Figure 4]

Figure 3: Cross-class interpolation. Top row of each pair: pixel-space blending xt=(1−t)xa+txb (ghosting artifacts). Bottom row: interpolation in DINOv2 representation space zt=(1−t)za+tzb, then decoded back to pixels via the RAE decoder (smooth semantic transitions).

used as the latent space of latent diffusion models [23]. Pixels and DINOv2 share the same ambient dimensionality, enabling direct geometric comparison; the inclusion of SD-VAE isolates the effect of representation-learning training (exemplified by DINOv2’s SSL) from generic compression.

Intrinsic dimensionality is the manifold’s true degrees of freedom—the number of independent directions needed to describe the data after stripping away ambient redundancy. Two spaces with comparable intrinsic dimensionality face manifolds of comparable underlying complexity; any difference in flow-matching learning difficulty must therefore arise from how the manifold is positioned rather than from its size. We use the TwoNN estimator [8], which recovers d by maximum likelihood from the ratio of second- to first-nearest-neighbor distances under local uniformity (Appendix C). Bootstrapping over 10 independent subsamples of 5,000 points gives dˆ= 33.6 ± 1.3 for pixels and dˆ= 32.6 ± 0.8 for DINOv2—nearly identical, with the 1-dimension gap well within the combined estimator standard deviation. Both spaces therefore share essentially the same underlying manifold dimensionality; DINOv2’s advantage, by elimination, lies in how that manifold is embedded relative to the noise.

Effective rank quantifies how uniformly variance is distributed across principal directions. It equals 1 when all variance concentrates in a single direction (a thin needle in RD) and equals the ambient dimension when variance spreads perfectly evenly (an isotropic ball). Since the flow-matching source N(0,I) is itself isotropic, higher effective rank on the data side translates to shorter, more uniform

λˆi log λˆi with λˆi = λi/ j λj the normalized PCA eigenvalues [24]. Figure 1(a) plots per-component variance (log scale) and its cumulative: pixel’s first 50 components capture ∼60% of total variance versus ∼25% for DINOv2. The effective ranks are 45, 98, and 327 for pixels, SD-VAE, and DINOv2 respectively—a 7.3× gap between pixels and DINOv2. DINOv2’s per-token LayerNorm further fixes ∥z∥2 = d by construction, so this high effective rank concentrates features near an approximately isotropic shell of radius √

transport paths from noise to data. Concretely, erank = exp − i

d containing the dˆ≈33-dim data manifold [38, 16].

Optimization conditioning reflects whether different variance directions can be learned in parallel during training: a well-conditioned regression converges along all directions at comparable rates, while a poorly-conditioned one over-fits high-variance directions while starving low-variance ones. Flow matching at time t is implicitly such a regression, with effective covariance interpolating between I at t=0 (pure noise) and the data covariance H at t=1 (clean data); ill-conditioned H therefore propagates into late-schedule training. Concretely, under a local Gaussian approximation p(z0)≈N(µ,H), Ahamed et al. [1] show the regression covariance is Σt = (1−t)2I + t2H; we use the standard condition number κ(Σt) = λmax/λmin as the diagnostic. Figure 1(b) plots κ(Σt) across t ∈ [0,1]: both spaces start at κ=1 near t=0 (where Σt → I) and grow monotonically toward κ(H)

- as t→1. At t=0.9 (representative of late-schedule fine-grained data-fitting), pixel space reaches κ ≈ 2,000 while DINOv2 stays at κ ≈ 56—a 35× gap, enabling all variance components to be

learned at comparable rates. The same distributional proximity to N(0,I) also tightens the posterior p(z0 | zt), shrinking the irreducible variance E∥v − E[v | zt]∥2 of the per-pair velocity target—a distinct mechanism contributing to the faster convergence in §4.

Marginal Gaussianity measures how close each individual coordinate’s 1D distribution is to a Gaussian. The source N(0,I) is Gaussian along every axis, so closer-to-Gaussian per-coordinate marginals on the data side keep each dimension’s transport from noise to data short and well-behaved. We use the per-dimension excess kurtosis κj = E[(zj − µj)4]/σj4 − 3, which is 0 for a Gaussian, positive for heavier-than-Gaussian (outlier-prone) tails, and negative for lighter tails. As shown in Table 1 and Figure 2, DINOv2 dimensions are markedly more Gaussian: 98.7% satisfy |κj| < 0.5 (vs. 74.2% for SD-VAE and 0% for pixels), with median |κj| 11.5× lower than pixels and 2.7× lower than SD-VAE. This captures marginal behavior only; the interpolation experiment below probes the joint geometry.

On-manifold linear interpolation. The previous three axes summarize variance per-direction; the final axis probes the joint geometry. Flow matching transports samples along straight paths zt = tz0+(1−t)ϵ, so if linear chords between data points themselves wander off the manifold, intermediate zt states will too, leaving the velocity target poorly defined. Cross-class image interpolation makes this concrete: pixel interpolation produces ghosting artifacts characteristic of paths crossing lowdensity voids, while DINOv2 interpolation yields smooth semantic transitions (Figure 3). We quantify this via a round-trip reconstruction error (full procedure in Appendix C): each intermediate framewhether obtained by pixel blending or by linear interpolation in DINOv2 space followed by RAE decoding—is passed through the same DINOv2 encoder–RAE-decoder pipeline [44], and the MSE versus the input measures off-manifold distance. Because both conditions traverse the identical pipeline, the encoder–decoder reconstruction bias is shared; the remaining gap isolates whether the frame lies on or off the image manifold. Pixel frames incur 1.7× higher error than DINOv2 frames (0.0136 vs. 0.0080); Figure 1(c) shows DINOv2 remains close throughout while pixel stays uniformly off-manifold.

Summary. Pixel and DINOv2 share nearly identical intrinsic dimensionalities (both dˆ≈ 33) yet DINOv2 is far better suited to flow-matching learning: 7.3× higher effective rank, 35× better covariance conditioning, 11.5× lower excess kurtosis, and 1.7× lower on-manifold interpolation error; SD-VAE is consistently intermediate, indicating the advantage arises from representation-learning objectives rather than compression alone. These properties predict that DDT heads, Riemannian transports, and wider backbones are not required for competitive performance—a prediction we validate in Sections 3–4 with a vanilla DiT and x-prediction.

### 3 RiT: A Vanilla DiT for Representation-Space Diffusion

Guided by the geometry of Section 2, we instantiate the Representation Image Transformer (RiT) (Figure 4): a vanilla DiT backbone (SwiGLU [26], RMSNorm [43], 2D-RoPE [32], QK-normalization [11], plus in-context class tokens following JiT [18]), trained with a recipe tailored to DINOv2 features. We reuse RAE’s [44] frozen DINOv2-with-Registers encoder [21] and ViT decoder to move between pixels and features. The encoder yields patch tokens z ∈ Rd×h×w and a [CLS] token zcls ∈ Rd; both are projected, concatenated, and jointly attended, with separate linear heads predicting zˆ0 and zˆcls,0. Full details in Appendix D.

(1+N ) x D

Linear Linear

Transformer Block

𝒙 − 𝒑𝒓𝒆𝒅 xN

Conv2d PatchEmbed

Linear

Add Noise

Figure 4: RiT Arch. Frozen RAE encoder/decoder (gray) bracket a vanilla DiT trained by x-prediction; [CLS] and patch tokens share selfattention, with separate heads for zˆ0 and zˆcls,0.

Flow matching preliminaries. Flow matching [19, 7] learns a velocity field that transports noise ϵ ∼ N(0,I) to data z0 along straight paths zt = tz0 + (1−t)ϵ, with t ∈ [0,1], t=0 pure noise and t=1 clean data; the path’s time derivative is v = z0 −ϵ. The standard v-prediction objective trains a network vθ(zt,t,c) conditioned on timestep

t and class c to regress this velocity: Lv = Et,z

0,ϵ∥vθ(zt,t,c) − v∥2. (1) Generation integrates the learned ODE z˙t = vθ from t=0 to t=1 via an Euler or Heun solver.

#### 3.1 x-Prediction on Standardized Features

Element-wise standardization. DINOv2’s per-token LayerNorm pins ∥z∥2 = d within each token, but leaves the cross-dataset per-channel variance heterogeneous (≳102 spread across channels, §4). Before diffusion, we therefore standardize both patch tokens and the [CLS] token to zero mean and

unit variance per element: z˜c,h,w = (zc,h,w − µc,h,w)/ σc,h,w2 + ϵ (analogously for zcls), using statistics precomputed on the training set. This diagonal preconditioner [1] reduces the condition number κ(H) of the data covariance and relaxes the near-constant-norm constraint that LayerNorm imposes on raw DINOv2 features [16]. The inverse transform is applied before decoding. Henceforth we use z to denote the standardized feature. We find this step is a prerequisite rather than an optimization: training on raw DINOv2 features diverges entirely (Table 3).

x-Prediction. As established in §2, DINOv2’s norm concentration means linear flow-matching paths zt = tz0 + (1−t)ϵ traverse ambient regions the encoder never outputs; at such off-manifold zt, the v-target acquires a radial component orthogonal to the data manifold—the geometric interference phenomenon diagnosed by Kumar and Patel [16], who resolve it with Riemannian Flow Matching [2] using SLERP paths. Under v-prediction, the network must fit this radial component and therefore spends capacity on the norm direction rather than the tangential (along-manifold) direction. We resolve the same problem more simply, by changing the output parameterization.

Setting z0 = fenc(x) to the standardized DINOv2 feature, x-prediction [18] instead outputs zˆ0 = fθ(zt,t,c) directly, with predicted velocity vˆθ = (zˆ0−zt)/(1−t); plugging this into the v-prediction loss (1) yields

0,ϵ∥vˆθ − v∥2, (2)

Lfm = Et,z

Table 2: x-pred vs v-pred. FID50K, ImageNet 2562, Heun 50 steps, w/o guidance.

which is equivalent to the x-prediction loss ∥zˆ0 − z0∥2 up to a (1−t)−2 reweighting (Appendix B). The v- and x-forms thus coincide as loss functionals, but impose different learning problems on the network, because the output parameterization determines which function is actually fit. Under v-prediction, the network must fit (z0−zt)/(1−t): a target that depends on the off-manifold zt, diverges as (1−t)−1 near t=1, and spans the full ambient space. Under x-prediction, the network fits z0: a target that lies on the low-dimensional data manifold by construction and does not depend on t explicitly. The chord through off-manifold zt persists at the network input; at the output, the target is confined to the data manifold rather than spanning the full ambient space. This manifold-targeting property is not itself DINOv2-specific [18]; what is unique here is the combination with DINOv2’s isotropic per-coordinate variance and near-Gaussian marginals, which together bound the target and smooth its dependence on zt (§2), so a vanilla DiT suffices. Table 2 confirms this empirically: with the same architecture, encoder, and noise schedule, x-prediction consistently outperforms v-prediction.

80 ep. 200 ep. 400 ep.

v-pred 3.17 2.11 1.86 x-pred 2.63 1.89 1.70

#### 3.2 Joint CLS–Patch Modeling

A unique advantage of operating in representation space is direct access to the [CLS] token—a global semantic summary that encodes category, layout, and appearance complementary to local patch content. In standard latent diffusion on VAE features such a global token is not part of the latent representation itself; in representation space, it is intrinsic. We model [CLS] jointly with patches in the same diffusion process: zcls,t = tzcls + (1 − t)ϵcls is projected, prepended to the patch sequence, and participates in bidirectional self-attention, aggregating spatial evidence into a global context and broadcasting refined guidance back to local tokens. A separate linear head produces the [CLS] prediction zˆcls,0, yielding an auxiliary x-prediction loss Lcls = E∥vˆcls,θ − vcls∥2 (written in velocity form for symmetry with Eq. 1; equivalent to ∥zˆcls,0 − zcls∥2 up to the same (1−t)−2 reweighting of Appendix B) and total objective L = Lfm + λLcls. During training, [CLS] noise is sampled independently from patch noise to avoid a 162=256× variance collapse ([CLS] is a single vector while patch noise is d×16×16). At inference, we advance [CLS] and patches jointly with Heun + classifier-free guidance under (optionally separate) guidance scales; we use the same scale for both in all reported experiments, but the mechanism permits decoupling. Only patch tokens are decoded while [CLS] is discarded. We also couple the two noise streams at initialization via ϵcls = meanh,w(ϵ) (coupled noise)—a minor but consistent improvement at convergence (§4.3).

#### 3.3 Dimension-Aware Noise Schedule

The SNR of zt = tz0 + (1−t)ϵ is SNR(t) = t2/(1−t)2, but the effective per-token SNR scales with the per-token dimension d

[13]:√ for a d-dimensional token, the ℓ2 noise magnitude grows as

d while the signal stays at unit scale, so higher-d tokens need lower t (more noise) to reach the same relative corruption. A DINOv2-Small token has d=384, 128× the per-pixel dimension of 3, so a pixel-space schedule undertrains on noisy states. Following RAE [44] and SD3 [7], we apply the dimension-dependent time shift X′ = Xs/(1+(s−1)X) with s = hwd/4096 ≈ 4.9 to X ∼ logit-N(0,1) and set t = 1−X′. This pushes the median t from ≈0.31 to ≈0.17 (5× lower median SNR, Figure 5); §4 shows this closes a 2× FID gap over the pixel-space logit-normal baseline (3.17 → 1.44 at 800 epochs).

(a) Time sampling distribution p(t)

| |JiT (pixel)| |
|---|---|---|
| |Logit-normal baseline RiT (DINOv2-S)| |
| | | |

- 0
- 1
- 2
- 3

Density

0.0 0.2 0.4 0.6 0.8 1.0

(b) Signal-to-Noise Ratio

log10 SNR(t)

SNR=0.20 SNR=0.042

2.5

logSNR10

JiT median

Baseline median

0.0

RiT median

2.5

0.0 0.2 0.4 0.6 0.8 1.0

Figure 5: Time sampling p(t) (top) and per-token SNR (bottom).

### 4 Experiments

Setup. RiT-XL has 28 layers, hidden dimension 1152, 16 attention heads, and FFN expansion ratio 4, totaling 676M parameters. We train with a frozen DINOv2-Small encoder (d=384) and a pretrained RAE decoder [44] on ImageNet, using 8 H200 GPUs (∼12min per epoch). We evaluate with FID50K using class-balanced sampling (50 images per class) following RAE [44]. Full hyperparameters are in Appendix D.

Figure 6: Convergence comparison on ImageNet 2562. FID-50K vs training epochs.

12.00

REPA-XL

RAE-XL (DINOv2-B)

REG-XL

RJF (DINOv2-B)

8.00

RAE-XL (DINOv2-S)

FID(w/oguidance)

RiT-XL (DINOv2-S)

5.90

4.28

7× faster 4× faster

2.39

1.80 1.44

80 100 200 400 800

Training Epochs

Table 3: Ablation study. Default: x-prediction (Table 2), element-wise standardization, time-shift schedule, λ=0.2, DINOv2-S. Each row replaces one factor. “†” indicates training diverges.

200 ep 400 ep 800 ep Default (ours) 1.83 1.67 1.44 Preconditioning

w/o standardization† 387.4 362.1 344.8 Noise schedule

Logit-normal 4.45 4.11 3.17 CLS loss weight λ

w/o CLS 1.89 1.70 1.63 λ=0.4 1.86 1.64 1.50

Encoder DINOv2-Base 2.20 1.78 1.56

#### 4.1 Convergence and Efficiency

Figure 6 compares RiT-XL against baselines. Against RAE-XL (DINOv2-S) [44]—a v-prediction DiT-XL with the same encoder, decoder, and parameter count (676M) as RiT-XL, isolating the §3 design choices—RiT-XL leads at every epoch and reaches FID 1.45 at 800 ep (23% better than

- 1.87). At 100 ep, RiT-XL already matches the larger RAE-XL (DINOv2-B) baseline at 720 ep (7× speedup); at 200 ep it matches RAE-XL (DINOv2-S) at 800 ep (4× speedup). RiT also surpasses the 800-ep FID of representation-alignment methods REPA [41] and REG [39] within 20–200 epochs. Concurrent RJF [16] tackles the same DINOv2 radial ambiguity via Riemannian Flow Matching on the norm-concentration sphere; at the matched 80-ep budget shown in Figure 6, RiT-XL reaches FID
- 2.48 (DINOv2-S) versus RJF’s 3.62 (DINOv2-B).

#### 4.2 Every Recipe Choice Is Necessary

Ablations use the full recipe unless noted, varying one factor at a time; we report ImageNet 256×256 FID-50K without guidance at Heun 50 steps.

Element-wise standardization. Raw DINOv2 features have heterogeneous per-channel variances (≳102 range across channels). Training on raw features diverges: the loss oscillates and FID stays at random-init level (> 300) throughout training.

Noise schedule. The time-shift closes a 2× FID gap over the original JiT logit-normal schedule (3.17 → 1.44 at 800 ep), confirming that reallocating training density toward higher noise is critical when per-token dimensionality grows (here d=384 vs. pixel’s d=3).

CLS token. Without CLS modeling (λ=0), FID plateaus at 1.63; λ=0.2 reaches 1.44. Attention visualization (Appendix I) shows [CLS] aggregates coarse scene cues in early layers, integrates object–context relations in middle layers, and broadcasts refined guidance back in late layers.

Encoder size. Despite half the feature dimensionality, DINOv2-Small consistently beats DINOv2Base (1.44 vs. 1.56 at 800 ep). Model capacity is not the bottleneck here: DINOv2-B has twice the ambient dimensionality at the same dˆ≈ 33, so the denoiser must regress over a higher-dimensional target without a corresponding gain in underlying structure. The Section 2 analysis on DINOv2-Base is thus a conservative characterization—the Small manifold used in main experiments is at least as favorable and the regression task is easier.

#### 4.3 Efficient ODE Convergence Enables Few-Step Generation

The four geometric properties established in Section 2—high effective rank, well-conditioned covariance, near-Gaussian marginals, and on-manifold linear interpolants—jointly predict that the noise-to-data ODE should be efficiently solvable in few Heun steps on DINOv2 features. We verify this directly by measuring Heun-solver truncation error in pixel space and show that it translates into order-of-magnitude gains under tight sampling budgets—without any distillation or consistency training.

- Table 4: Sampling schedule ablation. FID-50K across six ODE time-discretization schedules and Heun step counts, with and without classifier-free guidance. Each cell reports independent/coupled noise FID. RiT-XL on DINOv2-S, 800 epochs.

###### Figure 7: Few-step FID at matched NFE.

50

10 NFE 20 NFE

| |
|---|

26.2

w/o guidance (CFG = 1.0, Heun steps) w/ guidance (CFG = 3.7, Heun steps) Schedule 5 10 25 50 5 10 25 50

20

16.4

FID-50K(log)

10

Uniform 12.8/12.7 6.88/6.76 2.34/2.29 1.61/1.58 10.8/10.7 6.19/6.12 1.93/1.90 1.30/1.28 EDM 2.37/2.34 1.61/1.58 1.49/1.47 1.47/1.46 2.01/1.99 1.33/1.32 1.17/1.15 1.16/1.14 Cosine 5.96/5.86 2.16/2.12 1.57/1.56 1.48/1.45 5.69/5.63 1.91/1.88 1.29/1.28 1.19/1.18 Power-2 2.41/2.39 1.74/1.72 1.50/1.48 1.47/1.44 1.99/1.98 1.48/1.46 1.18/1.16 1.16/1.15 Log-SNR 4.56/4.51 1.96/1.95 1.51/1.50 1.46/1.44 3.79/3.78 1.73/1.74 1.23/1.22 1.18/1.17 Time-shift 2.44/2.38 1.59/1.58 1.47/1.45 1.46/1.44 1.99/1.99 1.27/1.25 1.15/1.14 1.15/1.14

5

3.29

2.38

1.87

- 1

- 2

1.58

DiTDH-XL (DINOv2-B)

JiT-H (pixel)

RiT-XL (DINOv2-S)

Pixel-space truncation error measurement. For each model we generate 128 trajectories per space from matched (ϵ,y) pairs, run Heun sampling at K ∈ {2,5,10,25,50} plus a reference K=125, decode every endpoint to pixel space (RiT through the frozen RAE decoder), and measure the Frobenius distance ∥x(K)−x(125)∥F. Because each model is compared against its own 125-step reference, the metric isolates ODE convergence speed from the absolute quality of either endpoint: a model that happens to converge to a poor fixed point is not artifactually rewarded. JiT uses the schedule reported in its own paper.

K 2 (Heun 2nd-order)

()(ref)truncationerrorKxxF

RiT-XL (DINOv2-S)

- 101

- 102

JiT-H/16 (pixel)

3.6 × decay

2.7 ×

12.9 × decay

2 5 10 25 50 sampling steps K (Heun, ref Kref = 125)

RiT’s truncation error decays 3.6× steeper than JiT’s. Figure 8 shows the averages. RiT’s truncation drops 12.9× from K=2 to K=50 (113.4→8.8), while JiT’s drops only 3.6× (87.1→23.9). At K=25 (RiT’s default), RiT is within 22.2 Frobenius units of its 125step endpoint—consistent with Table 4’s finding that RiT’s K=25 FID of 1.47 already matches the K=50 converged 1.46. JiT at the same K remains at 44.8 Frobenius units and its FID is still far from converged (Figure 7: JiT-H’s 10-NFE FID is 26.2). Under the 2nd-order Heun error bound err(K)∝

Figure 8: DINOv2 ODE converges in few Heun steps. Pixel-space truncation error ∥x(K)−x(ref)∥F vs step count K (mean ±1σ over 128 trajectories; each model vs its own Kref=125). RiT decays 12.9× (K=2→50) vs JiT’s 3.6×; late-K slope −1.33 vs −0.91, with the dashed line showing the Heun ∝ K−2 asymptote.

κeff(1/K)2, the 3.6× gap in decay rate translates into a correspondingly smaller effective curvature for DINOv2’s marginal flow. This is the empirical counterpart of the four geometric properties quantified in Section 2: higher effective rank shortens the transport, Gaussianity matches the source distribution, tighter posterior lowers velocity-target variance, and on-manifold interpolants keep zt in well-defined regions—each independently predicting a smoother, easier-to-integrate velocity field.

Few-step generation. Figure 7 isolates the role of representation space under matched NFE. At 10 NFE, pixel-space JiT-H yields FID 26.2 and DINOv2-space DiTDH-XL yields 3.29, while RiT-XL reaches 2.38—an order-of-magnitude improvement over pixel space and a clear gain over the DDTequipped DINOv2 baseline; the same ordering holds at 20 NFE (26.2→16.4 for JiT-H, 3.29→1.87 for DiTDH-XL, 2.38→1.58 for RiT-XL). Within RiT itself (Table 4), 5 Heun steps with a time-shift schedule reach FID 2.44 without guidance and 1.99 with guidance, 10 steps reach 1.59 and 1.25, and 25 steps already match full convergence—surpassing the majority of prior VAE-latent baselines in Table 5, which typically require ∼250 sampling steps.

Sampling schedule ablation. Table 4 compares six ODE time-discretization schedules (formal definitions in Appendix G). At ≥50 steps, all non-uniform schedules converge to similar FID (1.43– 1.45 w/o guidance, 1.14–1.15 w/ guidance), confirming the ODE is well-approximated. At 5 steps, the three concentrated schedules (EDM, power-2, time-shift: FID ≈2.4) outperform uniform spacing (12.7) by 5×, as they allocate more evaluations to the high-noise region where the velocity field varies most rapidly. Coupled noise (§3.2, independent/coupled cells in Table 4) shifts FID by < 0.1 uniformly across schedules and step counts; we adopt it in our main results for its small consistent gain but do not regard it as an essential component.

From geometry to empirics. The three mechanisms isolated in Section 2 map directly onto RiT’s gains. (i) The 35× better covariance conditioning (κ=56 vs. 2,000) lets all variance directions train

- at comparable rates. (ii) The tighter posterior p(z0 | zt) shrinks the irreducible target variance and contributes to the 7× convergence speedup (Figure 6). (iii) High effective rank (shorter transport), near-Gaussian marginals (smoother source→data interpolation), and on-manifold interpolants (no void-crossing) yield low effective curvature, verified by the 3.6× faster truncation-error decay (Figure 8) and the few-step regime (Figure 7). Pixel latents fail on all three and SD-VAE on at least two, so these mechanisms are empirically independent; their joint materialization on DINOv2 explains why a vanilla 676M DiT surpasses 839M DDT-equipped baselines both at convergence and under tight sampling budgets.

[Figure 5]

- Figure 9: Curated RiT-XL samples on ImageNet 2562 selected to span ImageNet categories.

#### 4.4 Comparison with Prior Methods

- Table 5 reports the full comparison. RiT’s unguided FID of 1.45 already surpasses the guided FID of every method without a representation encoder (PixelDiT-XL 1.61, JiT-G 1.82, MDTv2-XL 1.58, DiT-XL 2.27, SiT-XL 2.06)—the well-structured semantic space captures the distribution so faithfully that CFG becomes far less necessary. Among representation-based methods, RiT achieves the best unguided FID (1.45 vs. DiTDH-XL 1.51, FAE-DINOv2-G 1.48, RAE-XL 1.87) and the best guided FID (1.14 vs. DiTDH-XL 1.28, FAE 1.29, REPA-XL 1.29). The margin over FAE is narrow at the unguided level (1.45 vs. 1.48), but obtained under a substantially simpler setup: FAE uses the largest DINOv2 variant (DINOv2-G, d=1536) and jointly fine-tunes its encoder for generation, whereas RiT uses the smallest variant (DINOv2-S, d=384) with the encoder entirely frozen—indicating that the geometric advantages of Section 2 are already accessible off-the-shelf, without encoder co-adaptation. More fundamentally, FAE and RiT address different problems: FAE eases generation

- Table 5: Class-conditional image generation on ImageNet 256×256. With a vanilla DiT-XL backbone (676M parameters) and only 25 Heun steps, RiT achieves the best FID. Notably, RiT uses the smallest DINOv2 variant (DINOv2-S, d=384) among representation-based methods; DiTDH-XL uses DINOv2-B, and FAE [10] fine-tunes a DINOv2-G encoder, compressing its d=1536 features to a d=32 latent for generation.

w/o guidance w/ guidance FID↓ IS↑ Prec.↑ Rec.↑ FID↓ IS↑ Prec.↑ Rec.↑ Pixel Diffusion

Method AE Epochs #Params

ADM [6] – 400 554M 10.94 101.0 0.69 0.63 3.94 215.8 0.83 0.53 PixelFlow-XL [3] – 320 677M – – – – 1.98 282.1 0.81 0.60 PixNerd-XL [36] – 320 700M – – – – 1.93 298.0 0.80 0.60 PixelDiT-XL [42] – 320 797M – – – – 1.61 292.7 0.78 0.64 JiT-G [18] – 600 2B – – – – 1.82 292.6 0.79 0.62

Latent Diffusion

DiT-XL [22] SD-VAE 1400 675M 9.62 121.5 0.67 0.67 2.27 278.2 0.83 0.57 SiT-XL [20] SD-VAE 1400 675M 8.61 131.7 0.68 0.67 2.06 270.3 0.82 0.59 MaskDiT [45] SD-VAE 1600 675M 5.69 177.9 0.74 0.60 2.28 276.6 0.80 0.61 MDTv2-XL [9] SD-VAE 1080 675M – – – – 1.58 314.7 0.79 0.65 REPA-XL [41] SD-VAE 800 675M 5.78 158.3 0.70 0.67 1.29 306.3 0.79 0.64 LightningDiT [40] VA-VAE 800 675M 2.17 205.6 0.77 0.65 1.35 295.3 0.79 0.65 DDT-XL [37] SD-VAE 400 675M 6.27 154.7 0.68 0.69 1.26 310.6 0.79 0.65 SVG-XL [27] SVGTok 1400 675M 3.36 181.2 – – 1.92 264.9 – – REG-XL [39] SD-VAE 800 675M 1.80 - - - 1.36 299.4 0.77 0.66 RAE-XL [44] RAE-DINOv2-S 800 676M 1.87 209.7 0.80 0.63 1.41 309.4 0.80 0.63 RAEDH-XL[44] RAE-DINOv2-B 800 839M 1.51 242.9 0.79 0.63 1.16 261.0 0.77 0.67 FAE-XL [10] FAE-DINOv2-G 800 675M 1.48 239.8 0.81 0.63 1.29 268.0 0.80 0.64

RiT-XL (ours) RAE-DINOv2-S 800 676M 1.45 231.6 0.81 0.62 1.14 299.7 0.80 0.63

by adapting the encoder to produce a more diffusion-friendly latent space (compressing DINOv2-G’s d=1536 features to a d=32 generation latent), whereas RiT directly tackles modeling the existing high-dimensional representation distribution without altering the encoder. The two contributions are therefore largely orthogonal—encoder-side adaptation (FAE) and denoiser-side recipes (RiT) could in principle compose.

RiT uses uniformly smaller components. The denoiser is 676M parameters (19% smaller than DiTDH-XL’s 839M, no DDT head); the encoder is DINOv2-S (d=384, the smallest DINOv2 variant), versus DiTDH-XL’s DINOv2-B (d=768) and FAE’s DINOv2-G (d=1536). RiT also attains the highest unguided Precision (0.81) at competitive Recall (0.62), confirming that an off-the-shelf DINOv2 encoder and a vanilla backbone are sufficient when the representation distribution is favorable for flow matching.

### 5 Related Work

Diffusion and flow matching for images. Diffusion and flow matching [12, 30, 19, 7] underpin modern image generation. Latent diffusion [23] compresses images via a VAE; DiT/SiT [22, 20] replace the U-Net with transformers. A critical design choice is the prediction target—ϵ, v, or x. JiT [18] showed that x-prediction substantially outperforms the alternatives in pixel space by placing the target on the low-dimensional data manifold. Our work extends this insight by characterizing how a pretrained representation space embeds that manifold differently relative to N(0,I), and shows that DINOv2 is especially well-suited to x-prediction with a vanilla backbone.

Leveraging representations for generation. VA-VAE, EQ-VAE, and Diffusability shape autoencoder training for diffusion-friendly latents [40, 15, 28]. REPA [41] adds a DINOv2 alignment loss that is active only during training. REG [39] addresses this train–inference gap by entangling a DINOv2 [CLS] into the SD-VAE trajectory; RiT differs in operating natively in DINOv2 space where [CLS] is intrinsic. Neither REPA nor REG analyzes the manifold geometry, and both retain v-prediction on SD-VAE latents. RAE [44] replaces the VAE with a DINOv2 encoder but adopts v-prediction and needs a DDT head for the ill-conditioned velocity field; concurrent RJF [16] instead uses Riemannian Flow Matching with SLERP paths on the norm-concentration sphere. We show

x-prediction with element-wise standardization suffices to model DINOv2 features with a vanilla DiT—no architectural modification, no Riemannian reformulation.

Few-step generation and distillation. Progressive distillation [25], consistency models [31, 29], and rectified flow [19] reduce sampling cost by training a dedicated few-step student or straightening the teacher’s trajectories. These are orthogonal to RiT’s contribution: RiT shows that the base model itself already reaches competitive few-step FID in a geometry-friendly representation space, without any distillation or consistency loss, and remains a natural teacher for such methods.

Toward unified understanding–generation. Unified vision models [5, 33, 4, 34, 35] typically maintain separate encoders for perception (CLIP/DINOv2) and synthesis (SD-VAE), with task-specific architectural components bolted onto the generative side. RiT’s ability to generate competitively in DINOv2 space suggests a cleaner alternative: a single semantic representation and a single vanilla Transformer backbone—no DDT head, no Riemannian reformulation, no representation-alignment loss—can serve both tasks. The 7× training speedup at matched encoder (§4) and competitive few-step sampling (FID 2.0 at 5 Heun steps, 1.25 at 10 steps) further reduce the practical cost of attaching a generative head to an existing perception stack, making DINOv2-space RiT a promising base for unified pipelines in which the same features drive classification, retrieval, and synthesis.

### 6 Conclusion

We presented RiT, a vanilla DiT trained with x-prediction on frozen DINOv2 features that achieves FID 1.45 without guidance and 1.14 with classifier-free guidance on ImageNet 2562 using 19% fewer denoiser parameters than DiTDH-XL (676M vs. 839M) and the smallest DINOv2 variant (DINOv2-S, d=384); it further supports few-step generation (guided FID 2.0 at 5 Heun steps, 1.25 at 10 steps) without distillation or consistency training. The Section 2 analysis indicates that representation-space diffusion becomes architecturally simpler whenever the feature distribution satisfies the four geometric axes we identify—high effective rank, well-conditioned covariance, near-Gaussian marginals, and on-manifold linear interpolants. Our results argue for a target-side reformulation (x-prediction) over architecture-side (DDT heads) or transport-side (Riemannian flow matching) solutions whenever the representation’s distributional geometry is already favorable.

### Acknowledgments

We thank the Mila IDT team and their technical support for maintaining the Mila compute cluster. We also acknowledge the material support of NVIDIA in the form of computational resources. Throughout this project, Aishwarya Agrawal received support from the Canada CIFAR AI Chair award.

### References

- [1] Shadab Ahamed, Eshed Gal, Simon Ghyselincks, Md Shahriar Rahim Siddiqui, Moshe Eliasof, and Eldad Haber. Preconditioned score and flow matching. arXiv preprint arXiv:2603.02337, 2026.
- [2] Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. arXiv preprint arXiv:2302.03660, 2023.
- [3] Shoufa Chen, Chongjian Ge, Shilong Zhang, Peize Sun, and Ping Luo. Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963, 2025.
- [4] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.
- [5] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.
- [6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. In NeurIPS, 2021.

- [7] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [8] Elena Facco, Maria d’Errico, Alex Rodriguez, and Alessandro Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. Scientific reports, 7(1):12140, 2017.
- [9] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023.
- [10] Yuan Gao, Chen Chen, Tianrong Chen, and Jiatao Gu. One layer is enough: Adapting pretrained visual encoders for image generation. arXiv preprint arXiv:2512.07829, 2025.
- [11] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4246–4253, 2020.
- [12] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [13] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pages 13213–

13232. PMLR, 2023.

- [14] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35: 26565–26577, 2022.
- [15] Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Eq-vae: Equivariance regularized latent space for improved generative image modeling. arXiv preprint arXiv:2502.09509, 2025.
- [16] Amandeep Kumar and Vishal M Patel. Learning on the manifold: Unlocking standard diffusion transformers with representation encoders. arXiv preprint arXiv:2602.10099, 2026.
- [17] Elizaveta Levina and Peter Bickel. Maximum likelihood estimation of intrinsic dimension. Advances in neural information processing systems, 17, 2004.
- [18] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.
- [19] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [20] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.
- [21] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.
- [22] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https://arxiv.org/abs/2212.09748.
- [23] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [24] Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pages 606–610. IEEE, 2007.

- [25] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.
- [26] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.
- [27] Minglei Shi, Haolin Wang, Wenzhao Zheng, Ziyang Yuan, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Jie Zhou, and Jiwen Lu. Latent diffusion model without variational autoencoder. arXiv preprint arXiv:2510.15301, 2025.
- [28] Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. arXiv preprint arXiv:2502.14831, 2025.
- [29] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189, 2023.
- [30] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [31] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. 2023.
- [32] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.
- [33] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [34] Shengbang Tong, David Fan, Jiachen Li, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17001–17012, 2025.
- [35] Shengbang Tong, Boyang Zheng, Ziteng Wang, Bingda Tang, Nanye Ma, Ellis Brown, Jihan Yang, Rob Fergus, Yann LeCun, and Saining Xie. Scaling text-to-image diffusion transformers with representation autoencoders. arXiv preprint arXiv:2601.16208, 2026.
- [36] Shuai Wang, Ziteng Gao, Chenhui Zhu, Weilin Huang, and Limin Wang. Pixnerd: Pixel neural field diffusion. arXiv preprint arXiv:2507.23268, 2025.
- [37] Shuai Wang, Zhi Tian, Weilin Huang, and Limin Wang. Ddt: Decoupled diffusion transformer, 2025.
- [38] Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pages 9929–9939. PMLR, 2020.
- [39] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025.
- [40] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025.
- [41] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.
- [42] Yongsheng Yu, Wei Xiong, Weili Nie, Yichen Sheng, Shiqiu Liu, Anima Anandkumar, and Arash Vahdat. Pixeldit: Pixel diffusion transformers for image generation. arXiv preprint arXiv:2511.20645, 2025.

- [43] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.
- [44] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.
- [45] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. TMLR, 2023.

### A Limitations

DINOv2 encoder bias. RiT inherits the inductive biases of the frozen DINOv2 encoder. DINOv2’s SSL objective emphasizes semantic content over photometric detail, and prior work has observed weaker feature resolution on fine textures, thin structures, and small objects. These biases propagate directly into what RiT can generate, since the RAE decoder operates on the same features. Joint encoder fine-tuning (as in FAE [10]) could mitigate this, at the cost of the simpler frozen-encoder setup we advocate.

Class conditioning and resolution. All RiT results are class-conditional on ImageNet at 256×256. We have not evaluated text-to-image generation, higher resolutions (e.g., 512 or 1024), or non-image modalities. The Section 2 geometric analysis was measured on ImageNet images at DINOv2-Base and DINOv2-Small scales; whether the same four axes persist at larger model/data scales or under text conditioning is left to future work.

Local-Gaussian assumption in the analysis. The covariance-conditioning diagnostic (Section 2) rests on a local Gaussian approximation p(z0) ≈ N(µ,H); the other three geometric axes are assumption-light but still report aggregate scalars that cannot rule out adversarial pockets of the manifold where the favorable properties fail. The flow-matching results empirically corroborate the geometric claims without establishing each axis as individually necessary for the observed efficiency gains.

### B Equivalence of Velocity Loss and Reweighted x-Prediction Loss

We show that the velocity MSE loss with x-prediction parameterization (Eq. 2) is equivalent to a reweighted x-prediction loss. Given the forward process zt = tz0 + (1 − t)ϵ, the target velocity is:

z0 − zt 1 − t

, (3)

v = z0 − ϵ =

where the second equality follows from substituting ϵ = (zt − tz0)/(1 − t). Under x-prediction, the network outputs zˆ0 = fθ(zt,t,c), and the predicted velocity is vˆθ = (zˆ0 − zt)/(1 − t). Substituting both into the velocity loss:

2

z ˆ0 − zt

z0 − zt 1 − t

1 (1 − t)2 ∥zˆ0 − z0∥2 . (4)

0,ϵ ∥vˆθ − v∥2 = Et,z

= Et,z

Lfm = Et,z

1 − t −

0,ϵ

0,ϵ

Thus the velocity loss equals the x-prediction loss ∥zˆ0 − z0∥2 reweighted by (1 − t)−2, which upweights the loss at high t (near clean data). The two losses are therefore equivalent as functionals; what differs is the network’s parameterization, which determines the function actually fit (§3.1): x-prediction makes zˆ0 the direct network output, so its regression target always lies on the data manifold, whereas v-prediction asks the network to produce the ambient velocity (z0 − zt)/(1 − t), which depends on the off-manifold zt and diverges as t → 1.

### C Manifold Analysis Details

This appendix provides formal definitions and implementation details for the manifold analysis metrics used in Section 2. All experiments are conducted on 10,000 randomly sampled ImageNet training images.

PCA spectrum and effective rank. We fit PCA with 512 components on the flattened feature vectors of each space. Let λ1 ≥ λ2 ≥ ··· ≥ λk denote the eigenvalues of the sample covariance matrix and λˆi = λi/ j λj the normalized eigenvalues. The effective rank [24] is defined as

k

λˆi log λˆi . (5)

erank = exp −

i=1

It equals 1 when all variance concentrates in a single direction (maximally anisotropic) and equals k when variance is perfectly uniform (maximally isotropic). For flow matching, higher effective rank means the data distribution is closer to the isotropic Gaussian source, requiring a less complex velocity field.

Intrinsic dimensionality (TwoNN). The TwoNN estimator [8] estimates intrinsic dimensionality from the ratio of first- and second-nearest-neighbor distances. For each point xi, let r1(i) and r2(i) be the distances to its nearest and second-nearest neighbor, and µi = r2(i)/r1(i). Under the assumption that data is locally uniform on a d-dimensional manifold, the MLE estimator is

N

dˆ=

, (6)

N i=1 log µi

where N is the number of valid samples with µi > 1. We subsample 5,000 points and compute pairwise Euclidean distances in chunks to control memory.

Robustness of dˆ at D∼105. At such high ambient dimensionality, nearest-neighbor distances concentrate and any single-run intrinsic-dimension estimate can be noisy. We bootstrap TwoNN over 10 independent subsamples of 5,000 points, reporting the sample mean and standard deviation:

Space dˆ(mean ± std, 10 bootstraps)

Pixel 33.6 ± 1.3 DINOv2 32.6 ± 0.8

The√1.3pixel–DINOv22 + 0.82 ≈ 1.5gap(z ≈of01.7.0, dimensionp ≫ 0.05),issosubstantiallythe two estimatesbelowarethestatisticallycombined standardindistinguishable.deviation Larger-k variants of the MLE estimator [17] are known to suffer from upward bias at high ambient dimension [8] and do not share this convergence property; we therefore rely on TwoNN as the primary estimator. DINOv2’s advantage is not in manifold dimensionality but in the global geometry characterized by the other three axes of Section 2.

#### Marginal Gaussianity (excess kurtosis). For each dimension j, the excess kurtosis is

E[(xj − µj)4] σj4 − 3, (7)

κj =

where µj and σj are the per-dimension mean and standard deviation. A Gaussian distribution has κ = 0; positive values indicate heavier tails, negative values indicate lighter tails. We report the

median of |κj| across all dimensions as a scalar summary.

On-manifold interpolation score. For each intermediate frame along a linear interpolation path, we measure how well it stays on the natural image manifold via reconstruction error under a unified pipeline: image → encode(DINOv2) → decode → MSE versus the input image. This pipeline is applied identically to both pixel-space and DINOv2-space interpolation frames:

- • Pixel interpolation: The intermediate frame xt = (1−t)xa + txb is passed through encode→decode. Ghosting artifacts (off-manifold) produce high MSE because the encoder projects them to the nearest valid representation.
- • DINOv2 interpolation: The intermediate representation zt = (1−t)za + tzb is first decoded to a pixel-space frame, which is then passed through the same encode→decode pipeline.

By measuring both through the identical pipeline, the comparison is unbiased: the only difference is whether the frame was produced by pixel blending or DINOv2 latent interpolation. We average over 100 same-class pairs with 11 interpolation steps each.

Sanity check: encoder round-trip on DINOv2 interpolants. To rule out a trivial explanation that the DINOv2 interpolation pipeline enjoys near-zero reconstruction error by construction, we additionally measure whether zt and fenc(fdec(zt)) are close in feature space. If the encoder merely re-projected arbitrary inputs to their nearest valid representation, the pixel-versus-DINOv2 MSE gap could be artifactually large. We report the average cosine similarity between zt and its re-encoded counterpart, which remains high throughout the interpolation path; the gap in Figure 1(c) therefore reflects the off-manifold position of pixel blends rather than a baseline asymmetry in how the encoder treats each input.

### D Architecture and Hyperparameters

#### D.1 Model Architecture

RiT uses a modernized DiT backbone (following JiT [18]/LightningDiT [40]) operating on the 16×16 spatial grid of DINOv2 features. Our main experiments use DINOv2-Small (d=384); we also report DINOv2-Base (d=768) results in encoder ablations. Table 6 summarizes the model variants.

Table 6: RiT model variants. All models use patch size 1, SwiGLU FFN (MLP ratio 4×), QKnormalization, and VisionRoPE. Input dimension depends on the encoder: 384 × 16 × 16 for DINOv2-Small (main experiments) or 768 × 16 × 16 for DINOv2-Base.

Model Layers Hidden dim Heads FFN dim Params

RiT-L 24 1024 16 4096 458M RiT-XL 28 1152 16 4608 676M

Each DiT block consists of:

- 1. adaLN modulation: timestep and class embeddings are summed (c = Emb(t) + Emb(y)) and projected to per-layer scale/shift parameters via a shared SiLU–Linear layer.
- 2. Multi-head self-attention with QK-normalization (RMSNorm on Q and K before attention) and VisionRoPE for 2D spatial position encoding. [CLS] and register tokens are excluded from RoPE.
- 3. SwiGLU FFN: FFN(x) = (SiLU(xW1) ⊙ xW3)W2.

The final layer uses adaLN-modulated RMSNorm followed by a linear projection to d output channels (384 for DINOv2-Small, 768 for DINOv2-Base). A separate linear head predicts the [CLS] token. The code supports attention and projection dropout applied only in the middle 50% of layers; our main RiT-XL training sets both rates to 0.

Following JiT [18], we inject 32 learnable in-context tokens at an intermediate layer (layer 8 for RiTL). These tokens are initialized from the class embedding with added learnable positional embeddings, participate in self-attention for all subsequent layers, and are discarded before the final projection. They provide additional capacity for class-conditional generation without modifying the core DiT block.

#### D.2 Training Hyperparameters

Table 7: Training hyperparameters for RiT-XL on ImageNet 2562.

Hyperparameter Value Hardware 8 × NVIDIA H200 GPUs Throughput ∼12 minutes per epoch Optimizer AdamW (β1=0.9, β2=0.999) Base learning rate 5 × 10−5 (scaled by batch/256) LR schedule Constant (after warmup) Warmup epochs 5 Weight decay 0.0 Gradient clipping 1.0 (max ℓ2 norm) Total epochs 800 Batch size 1536 (8 × 192 per GPU) EMA decay 0.9999 / 0.9996 (dual tracking) Label dropout 0.1 Attention / projection dropout 0.0 (main configuration) Noise schedule Truncated logit-normal (µ=0, σ=1) Time shift s 16 × 16 × d/4096 (≈ 4.9 for d=384) Noise scale σϵ 1.0 Epsilon clamp ϵt 0.05 CLS loss weight λ 0.2

#### D.3 Sampling Hyperparameters

Table 8: Sampling hyperparameters.

Hyperparameter Value ODE solver Heun (2nd order) Number of steps 25 CFG scale (patches) 3.7 CFG scale (CLS) 3.7 CFG interval [0.1, 0.98] Epsilon clamp ϵsample 0.05 Generation precision FP32 EMA model 0.9999

#### D.4 Pseudocode

Training

|# z0: data [B,C,H,W], z_cls: [B,C] # dit: DiT model, y: class labels t = sample_logit_normal(B, shift=s) eps = randn_like(z0) * sigma zt = t * z0 + (1 - t) * eps # CLS noise: independent during training eps_cls = randn_like(z_cls) * sigma zt_cls = t * z_cls + (1 - t) * eps_cls # x-prediction: predict clean data z0_hat, cls_hat = dit(zt, t, y, zt_cls) # convert to velocity for loss v = (z0 - zt) / (1 - t).clamp_min(eps_t) v_hat = (z0_hat - zt) / (1 - t).clamp_min(eps_t) v_cls = (z_cls - zt_cls) / (1 - t).clamp_min(eps_t<br><br>) v_cls_hat = (cls_hat - zt_cls) / (1 - t).clamp_min<br><br>(eps_t) loss = mse(v_hat, v) + lam * mse(v_cls_hat, v_cls) loss.backward()<br><br>|
|---|

Sampling (Euler)

|# y: class labels, K: number of steps z = randn(B, C, H, W) * sigma z_cls = randn(B, C) * sigma dt = 1.0 / K for i in range(K):<br><br>t = i / K # x-prediction -> velocity z0_hat, cls_hat = dit(z, t, y, z_cls) v = (z0_hat - z) / (1 - t).clamp_min(eps_t) v_cls = (cls_hat - z_cls) / (1 - t).clamp_min(<br><br>eps_t) # classifier-free guidance # (omitted for clarity) z = z + dt * v z_cls = z_cls + dt * v_cls<br><br># decode to image x = rae_decoder(z)<br><br>|
|---|

- Figure 10: PyTorch-style pseudocode for RiT training (left) and sampling (right). The sampling code shows Euler for clarity; in all main experiments we use a 2nd-order Heun solver that additionally averages the velocity at t and the predicted next step. Key differences from standard flow matching: x-prediction and joint CLS modeling.

- E Encoder Size Ablation

See Section 2 (main text) for the encoder size ablation and Table 3 for the full ablation results. DINOv2-Small (d=384) consistently outperforms DINOv2-Base (d=768) despite having half the feature dimensionality, reaching FID 1.44 vs. 1.56 at 800 epochs. The lower-dimensional latent space (384 × 16 × 16 vs. 768 × 16 × 16) is easier for the denoiser to model, while DINOv2-Small still retains sufficient semantic information (TwoNN intrinsic dimensionality is comparable across encoder sizes).

- F Uncurated Sample Grid

- Figure 11 shows uncurated samples generated by RiT-XL (DINOv2-Small encoder, 760 training epochs) using the Heun sampler with 100 steps and classifier-free guidance scale 3.7. The samples span diverse ImageNet categories including animals, food, landscapes, vehicles, and plants, demonstrating RiT’s ability to produce high-fidelity, diverse images across a wide range of semantic categories.

[Figure 6]

- Figure 11: Uncurated RiT-XL samples on ImageNet 2562. 28 samples across diverse categories: macaw, jellyfish, flamingo, king penguin, golden retriever, Siberian husky, arctic fox, lion, monarch butterfly, red panda, giant panda, balloon, space shuttle, ice cream, cheeseburger, pizza, cliff, coral reef, volcano, and daisy. Generated with Heun 100 steps, CFG scale 3.7.

G Sampling Schedule Analysis

- Table 9 lists the six ODE time-discretization schedules evaluated in this work. Each schedule maps a

normalized step index i/K (i = 0,...,K) to a timestep ti ∈ [0,1], where t=0 is pure noise and t=1 is clean data.

Table 9: Sampling schedule definitions. K is the number of Heun steps and i = 0,...,K.

Schedule Formula Uniform ti = i/K Cosine ti = 12 1 − cos(π i/K) Log-SNR uniform ti = σ logit(ϵ) + Ki [logit(1−ϵ) − logit(ϵ)] , ϵ=10−3 EDM [14] σi = σmax1/ρ + Ki (σmin1/ρ − σmax1/ρ ) ρ, ti = 1/(1+σi)

σmin=0.002, σmax=80, ρ=7 Power-2 ti = (i/K)2 Time-shift ui = i/K, ti = 1 − ui s / (1 + (s−1) ui), s=4.9

- Figure 12 visualizes these schedule functions and their effect on generation quality. Panel (a) shows the schedule functions t(i): uniform distributes steps evenly, while EDM, power-2, and time-shift concentrate steps near t=0 (the high-noise end); cosine and log-SNR are denser near both endpoints. Panels (b) and (c) show the corresponding FID as a function of Heun step count. Schedules that allocate more evaluations to the high-noise regime—where the velocity field varies most rapidlyachieve dramatically better FID at low step counts (≤10), while all non-uniform schedules converge at ≥50 steps.

H Random Samples

- Figure 13 shows 192 randomly generated samples (8 per class, 24 classes) from RiT-XL without any curation or cherry-picking. Each row corresponds to a single ImageNet class. The model produces consistently high-quality and diverse samples across all categories.

###### (a) Schedule functions

###### (b) w/o guidance (CFG = 1.0)

(c) w/ guidance (CFG = 3.7)

1.0

| |Uniform<br><br>EDM<br><br>Cosine<br><br>Power-2<br><br>Log-SNR<br><br>Time-shift| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |
|---|

| |
|---|

0.8

- 100

- 101

| |
|---|

101

0.6

FID-50K

FID-50K

| |
|---|

| |
|---|

t

0.4

0.2

| |
|---|

| |
|---|

0.0

0.0 0.2 0.4 0.6 0.8 1.0 Normalized step index i/K

2 5 10 25 50 125

2 5 10 25 50 125

Heun steps

Heun steps

- Figure 12: Sampling schedule comparison. (a) Schedule functions t(i/K) for K=50 steps. (b, c) FID-50K vs. Heun step count without and with guidance. Coupled noise, RiT-XL on DINOv2-S, 800 epochs.

- Table 10: Full sampling schedule ablation (including 2-step). FID-50K for ODE time-discretization schedules and Heun step counts. Each cell: independent/coupled noise. RiT-XL on DINOv2-S, 800 epochs.

w/o guidance (CFG = 1.0, Heun steps) w/ guidance (CFG = 3.7, Heun steps)

Schedule 2 5 10 25 50 125 2 5 10 25 50 125

Uniform 23.67/23.75 12.84/12.72 6.88/6.76 2.34/2.29 1.61/1.58 1.47/1.45 20.67/20.63 10.80/10.68 6.19/6.12 1.93/1.90 1.30/1.28 1.16/1.15 EDM 11.78/11.70 2.37/2.34 1.61/1.58 1.49/1.47 1.47/1.46 1.47/1.45 10.43/10.34 2.01/1.99 1.33/1.32 1.17/1.15 1.16/1.14 1.14/1.13 Cosine 23.67/23.75 5.96/5.86 2.16/2.12 1.57/1.56 1.48/1.45 1.45/1.44 20.67/20.63 5.69/5.63 1.91/1.88 1.29/1.28 1.19/1.18 1.15/1.14 Power-2 11.16/11.09 2.41/2.39 1.74/1.72 1.50/1.48 1.47/1.44 1.45/1.43 9.44/9.37 1.99/1.98 1.48/1.46 1.18/1.16 1.16/1.15 1.15/1.14 Log-SNR 23.67/23.75 4.56/4.51 1.96/1.95 1.51/1.50 1.46/1.44 1.45/1.44 20.67/20.63 3.79/3.78 1.73/1.74 1.23/1.22 1.18/1.17 1.15/1.14 Time-shift 14.05/14.03 2.44/2.38 1.59/1.58 1.47/1.45 1.46/1.44 1.45/1.43 8.59/8.64 1.99/1.99 1.27/1.25 1.15/1.14 1.15/1.14 1.15/1.14

### I CLS–Patch Attention Analysis

- Figure 14 visualizes the bidirectional attention between [CLS] and patch tokens across layers and timesteps. We observe a clear stage-wise communication pattern:

CLS→Patch (left). In early layers, [CLS] attends broadly to salient foreground regions, aggregating coarse object cues. In middle layers, its attention expands to contextual/background regions, forming a global scene summary. In late layers, [CLS] re-focuses on semantically critical details (e.g., head and eyes), which strongly influence structural consistency and perceptual realism.

Patch→CLS (right). Patch tokens increasingly query [CLS] at deeper layers, with the strongest reliance in semantically important regions, while low-information background patches rely less on it. These observations suggest that [CLS] acts as a global message hub: it collects distributed evidence, integrates object–context relations, and broadcasts refined global guidance back to patch tokens, improving object–background disentanglement and final generation quality.

[Figure 7]

20

- Figure 13: Random (non-curated) RiT-XL samples on ImageNet 2562. Each row shows 8 independently generated samples for a single class. 24 classes shown: macaw, golden retriever, Siberian husky, arctic fox, lion, monarch butterfly, giant panda, balloon, ice cream, cheeseburger, pizza, cliff, coral reef, volcano, daisy, flamingo, king penguin, jellyfish, otter, red panda, cheetah, space shuttle, fountain, and loggerhead turtle. Generated with Heun 100 steps, CFG scale 3.7.

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

###### CLS-to-Patch Patch-to-CLS

- Figure 14: Layer-wise CLS–patch communication in RiT. Left: [CLS]-to-patch attention transitions from coarse scene aggregation to semantically salient regions. Right: patch-to-[CLS] attention shows global information exchange followed by focused refinement.

