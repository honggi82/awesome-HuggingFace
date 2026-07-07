## DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion

Noam Issachar* Guy Yariv* Sagie Benaim Yossi Adi Dani Lischinski Raanan Fattal The Hebrew University of Jerusalem

# arXiv:2510.20766v2[cs.CV]29Jan2026

### Abstract

Diffusion Transformer models can generate images with remarkable fidelity and detail, yet training them at ultra-high resolutions remains extremely costly due to the self-attention mechanism’s quadratic scaling with the number of image tokens. In this paper, we introduce Dynamic Position Extrapolation (DYPE), a novel, training-free method that enables pre-trained diffusion transformers to synthesize images at resolutions far beyond their training data, with no additional sampling cost. DYPE takes advantage of the spectral progression inherent to the diffusion process, where low-frequency structures converge early, while high-frequencies take more steps to resolve. Specifically, DYPE dynamically adjusts the model’s positional encoding at each diffusion step, matching their frequency spectrum with the current stage of the generative process. This approach allows us to generate images at resolutions that exceed the training resolution dramatically, e.g., 16 million pixels using FLUX. On multiple benchmarks, DYPE consistently improves performance and achieves state-of-the-art fidelity in ultra-high-resolution image generation, with gains becoming even more pronounced at higher resolutions. Project page is available at https:

//noamissachar.github.io/DyPE/.

### 1. Introduction

Diffusion Transformers (DiTs) (Peebles & Xie, 2022) have recently emerged as a powerful class of generative models, combining the stable training dynamics of diffusion (Ho et al., 2020; Song et al., 2020) with the expressiveness and scalability of transformers (Vaswani et al., 2017; Kaplan

*Equal contribution . Correspondence to: Noam Issachar <noam.issachar@mail.huji.ac.il>, Guy Yariv <guy.yariv@mail.huji.ac.il>.

Preprint.

“A mysterious woman in ornate dark armor holds a staﬀ before smoke, a red sky, and distant gothic buildings.”

[Figure 1]

[Figure 2]

[Figure 3]

FLUX YaRN Dy-YaRN

Figure 1. DYPE enables pre-trained diffusion transformers to generate ultra-high-resolution images (16M+ pixels) without retraining and without inference overhead, solely by coordinating the positional encoding with the diffusion’s progression. We compare the baseline FLUX, YaRN, and DYPE, specifically the DYYaRN variant, both applied on top of FLUX, at 4096 × 4096 resolution.

et al., 2020; Hoffmann et al., 2022). While this architecture fueled progress across large-scale vision (Dosovitskiy et al., 2021), training these models to ultra-high resolutions (e.g., 40962 and beyond) remains a formidable challenge: the quadratic complexity of self-attention in the number of image tokens drives up memory and compute costs, making direct training infeasible.

This limitation is analogous to the long-context challenge in large language models (LLMs), where transformers are trained with a fixed context horizon, but are expected to perform on much longer sequences during inference. The positional encoding (PE) mechanism is central to this generalization, as it dictates how transformers align and extrapolate positional relations across unseen ranges. Rotary Positional Embeddings (RoPE) (Su et al., 2021) are widely adopted but degrade when extrapolated beyond the training range. This has motivated inference-time adaptations such as position interpolation (PI) (Chen et al., 2023b), NTKaware rescaling (Peng et al., 2023), and YaRN (Peng et al.,

- 2023), which adjust the frequency spectrum to better preserve long-range dependencies.

In image generation, these LLM-derived schemes were adapted to accommodate aspect-ratio changes and moderate increases in resolution (Lu et al., 2024; Wang et al.,

- 2024). However, these static approaches do not account for the distinctive spectral progression of the diffusion process, where low-frequency structures are generated in the first

sampling steps, while high-frequency details are resolved later (Rissanen et al., 2023; Hoogeboom et al., 2023). As shown in Zhuo et al. (2024), aligning with these dynamics can facilitate better resolution extrapolation. These observations naturally lead to our guiding question: How should positional embeddings be dynamically adjusted to reflect the spectral progression of the diffusion process?

In this work, we analyze the spectral dynamics of the inverse diffusion process. Specifically, we assess the synthesis timeline at which each frequency component of the generated sample evolves as a function of the sampling step. This analysis shows that low-frequency Fourier components converge to their final values much earlier while high-frequency components evolve throughout the denoising. This finegrained observation allows us to design our Dynamic Position Extrapolation (DYPE), which exploits this progression: as sampling continues, the PE shifts more emphasis from the already-solidified low frequencies to the evolving highfrequency bands. By dynamically tailoring the PE’s spectral allocation, DYPE better serves the instantaneous needs of the diffusion operator throughout its sampling course.

This training-free strategy greatly improves generalization, allowing a pre-trained FLUX model (Labs, 2024) to generate images at ultra-high resolutions (exceeding 16M pixels), as shown in Fig. 1. We evaluate DYPE using quantitative metrics for image quality and prompt fidelity, alongside qualitative and human evaluations. The results show that DYPE achieves consistent improvements in ultra-high-resolution synthesis across multiple benchmarks and resolutions, all without retraining or additional sampling costs.

### 2. Preliminaries

#### 2.1. Diffusion Models

Diffusion models progressively evolve samples from a latent pure-noise, Gaussian distribution N(0,I), towards a target distribution q(x) via a sequence of intermediate mixture distributions. The process is governed by a time parameter t ∈ [0,1] that defines the mixture variables xt, by:

##### xt = αtx + σtϵ, x ∼ q(x), ϵ ∼ N(0,I), (1)

where the schedule coefficients αt and σt are chosen to achieve the endpoints x0 = x (pure data) and x1 = ϵ (pure Gaussian noise). We denote these mixture distributions by qt.

Different schedules αt and σt correspond to different formulations, e.g., Variance Preserving (Ho et al., 2020; Song et al., 2020) and Flow Matching (Lipman et al., 2022; Liu

- et al., 2022). The latter using the linear schedule αt = 1 − t and σt = t, which we adopt in our derivation.

2.2. Rotary Positional Embeddings and Position Extrapolation

Positional Embedding (PE). The transformer block, which is the basis of DiT, is permutation equivariant. Thus, a positional encoding mechanism is necessary to properly model the strong spatial dependencies in natural images (LeCun & Bengio, 1998). Early solutions use fixed sinusoidal positional embedding (Vaswani et al., 2017; Dosovitskiy et al., 2021), learned absolute embeddings (Devlin et al., 2019; Radford et al., 2019), or relative positional embeddings (Press et al., 2021). More recently, the Rotary Positional Embeddings (RoPE) (Su et al., 2021) emerged as a more effective alternative which provides the relative positions in the query–key interactions.

More specifically, RoPE represents a position coordinate m as a set of 2D rotations at different frequencies. The number of frequencies is determined and limited by D = dmodel/2, where dmodel is the hidden model dimension. The frequencies θd are typically obtained from a geometric series,

d

D−1, d = 0,...,D − 1, (2)

θd = θbase

with corresponding wavelength λd = 2π/θd, where θbase is a model hyper-parameter. We note that in case of 2D images RoPE is applied axially: half of the hidden vector is rotated horizontally, and the other half vertically. Thus this axial decomposition enables RoPE to encode relative offsets along each axis independently, considering the spatial structure of images (Heo et al., 2024).

As discussed above, training DiT models at high resolutions incurs substantial memory and compute cost. Applying a model at higher resolutions than it was trained on, suffers from degraded performance as illustrated in Fig. 1. This shortcoming spurred the development of inference-time positional encoding adaptations for a better generalization. Before we survey these approaches, let us establish useful notations from Peng et al. (2023).

Assuming the training context length, is L, and L′ is the extended context, we define the scaling factor s by:

##### s = L′/L. (3)

Moreover, the different extrapolation methods can be characterized by their action over the spatial coordinate m and frequencies θd that they represent, namely:

m  → g(m), θd  → h(θd), (4) where g and h are method-specific transformations.

Position Interpolation (PI) is an early approach (Chen et al., 2023b), that rescales uniformly the position m to the new context length L′ by:

##### g(m) = m/s, h(θd) = θd. (5)

This mapping resamples the waves cos(mθd),sin(mθd) at a finer rate in the larger context grid L′, and while it correctly reproduces the lower end of the spectrum, it fails to reach the new grid’s higher frequency band. While large scale content is properly synthesized in this approach, the missing highfrequencies manifest as blurriness and lack of fine detail, as discussed in Appendix A.

NTK-Aware Interpolation. To address this problem, the Neural Tangent Kernel (NTK-aware) interpolation (Peng

- et al., 2023) applies different scaling to the low and high frequencies, by:

θd

s2d/(D−2). (6) Thus, the low frequencies (large λd) remain nearly unchanged in the new grid as in PI, by trading off the representation of the high frequencies (small λd) due to the compression resulting from accommodating the higher band of the larger context L′.

g(m) = m, h(θd) =

YaRN. Yet another RoPE extensioN, or YaRN (Peng et al., 2023) extends the latter in two ways. The first is the NTKby-parts interpolation, which splits the spectrum to three bands, where different mappings are applied, namely:

g(m) = m, h(θd) = (1 − γ(r(d))) θ

s + γ(r(d))θd,

d

(7) where r(d) = L/λd. The ramp γ(r) provides a smooth transition from PI stretching to no scaling:

 

- 0, r < α, r−α β−α, α ≤ r ≤ β,

- 1, r > β,

(8)

γ(r) =



where α,β set the bands’ boundaries. Also here the bands are scaled non-uniformly, with more flexibility to control the allocation trade-offs made by NTK-aware interpolation.

The second extension is the attention scaling, where attention logits are modified by a factor τ(s) = 0.1ln(s) + 1. The resulting attention mechanism is defined as

qi⊤kj √dmodel

AttnYaRN(qi,kj) = softmax τ(s) ·

. (9)

This allows to counterbalance (reduce) the increase in entropy of the attention weights due to the introduction of additional keys in the larger context L′.

### 3. Method

We now present DYPE. We first analyze the spectral dynamics of the diffusion process, showing how different frequency modes evolve over time (Sec. 3.1). Based on this analysis, we derive DYPE, which dynamically adjusts positional encoding to match these dynamics (Sec. 3.2).

3.1. Evolution of Frequency Modes in the Diffusion Process

The simple mixture formulation in Eq. 1 allows us to derive a complementary perspective in Fourier space, as given by:

xˆt = (1 − t)ˆx + tϵ,ˆ (10)

where (ˆ·) denotes the Fourier transformed signals. The i.i.d noise vectors ϵ have a white (constant) Power Spectrum Density (PSD), and the data of natural images, xt, is known to have a well-characterized PSD with a power-law decay of ∝ 1/fω where ω ≈ 2 (van der Schaaf & van Hateren, 1996; Hyvrinen et al., 2009), as function of frequency f. These terms allow us to explicitly describe the time-dependent mean PSD in Eq. 10, given by

∥xˆt∥2f = (1 − t)2 C/fω + t2, (11)

which results from computing the mean PSD of xt, denoted by (·), according to Eq. 10, and noting that the covariance ⟨x,ˆ ϵˆ⟩ = 0 due to independence. The constant C is a characteristic PSD scale of the particular data distribution. Fig. 2a depicts the empirical evaluation of the averaged PSD computed over samples generated by a denoiser trained on ImageNet (Russakovsky et al., 2015). The function reveals the smooth transition between the two spectra and reflects the growth of low-frequency image structures and the decay of noise alongside the emergence of high-frequency fine-details, as predicted by Eq. 11.

The question we would like to address here is whether this evolution is fully “active” during the entire sampling process and at all the frequencies, or whether it shows some regularities which we can exploit for a better allocation of the represented spectrum.

To assess the rate at which these modes evolve, we consider a progression map relating each frequency component f to a progress index, 0 ≤ γ(f,t) ≤ 1, that indicates the relation of its log-PSD value at time t, i.e., log ∥xˆt∥2f , in relation to its endpoints. By utilizing the fact that the transition described by Eq. 11 is monotonic, this index is easily obtainable by

s(t)f − s(0)f s(1)f − s(0)f

(12)

γ(f,t) =

where s(t)f = log ∥xˆt∥2f .

Fig. 2b shows this progression map where a clear observation can be made. While the higher frequency components show a fairly constant evolution throughout the sampling process, the lower frequencies appear to evolve faster, and more importantly, cease to evolve fairly early in sampling. Assuming that the evolving modes depend more on their corresponding frequency representation in the PE than the

- 101
- 102
- 103
- 104
- 105
- 106
- 107
- 108

| | | | |
|---|---|---|---|
| | |Image (t=0)<br><br>Noise (t=1)| |
| | |1/f<br><br>| |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

PSD

10 2 10 1 100

Frequency

(a)

1.0

1.0

| |[Figure 4]| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

|[Figure 5]| |
|---|---|
| | |
| | |
| | |
| | |

0.8

0.8

Frequencyf

0.6

0.6

(,)ft

0.4

0.4

0.2

0.2

0.0

0.0

1.0 0.8 0.6 0.4 0.2 0.0

Diffusion time t (0 = image, 1 = noise)

(b)

Figure 2. Spectral Evolution of Samples in the Diffusion Process. (a) shows the average PSD of images produced by a diffusion model, power-law appears as the process ends (t = 0). The combinations of these spectra, corresponding to the mixture distributions qt, are seen at the intermediate steps (gray plots). The progression map γ(f, t) from Eq. 12 is shown in (b), and measures in relative terms how each Fourier component evolves from pure noise (t = 1) to its clean image value (t = 0). As seen in the top rows of this map (t ≈ 1), the high-frequency modes evolve gradually and nearly linearly across the entire reverse process. By contrast, the low-frequency modes converge much faster and cease to change early on, as indicated by the map’s saturation (yellow) in the lower rows (t ≈ 0).

- as a function of time t. The flat spectrum at t = 1 corresponds to the initial Gaussian samples, and the characteristic natural images

converged ones, the following frequency allocation strategy can be derived: at the beginning of the process, all modes evolve and hence all modes in the finer grid should be accommodated in the PE, e.g., using an existing extrapolation encoding strategies such as YaRN. As the sampling progresses, more and more modes in the lower end of the spectrum convergence and the PE emphasis should be allocated in favor of representing the yet-unresolved higher frequencies.

We further note that frequency extrapolation formulae allocate more low frequency components at the cost of removing higher ones, e.g., in NTK-aware and YaRN. Thus, switching off the extrapolation, as we suggest, has two benefits: (i) more high-frequency modes are represented in the PE, and (ii) the pretrained denoiser will operate in the conditions, namely the PE, it was trained with. These observations serve as a basis for the design of our new approach, DYPE, which we describe next.

This topic is briefly touched by (Zhuo et al., 2024) as part of deriving a new DiT architecture. However, the opposite conclusions are drawn. We discuss this strategy in Appendix B.

#### 3.2. Dynamic Position Extrapolation (DYPE)

Our new approach, DYPE, is motivated by two complementary insights. First, as discussed above, the reverse diffusion trajectory exhibits a clear spectral ordering: low-frequency, large-scale structures converge early, while high frequency bands are resolved throughout the sampling process. Second, while the existing positional extrapolation strategies, NTK-aware, and YaRN, are capable of representing the spectrum of the larger context using the limited number of

available modes, D, they involve representation trade-offs due to the compression they must employ. Thus, rather than pinpointing both ends of the spectrum at all times and accommodating these trade-offs, our method, DYPE, accounts for the spectral progression and gradually lowers their use to minimize the compression they involve.

We implement this strategy by introducing explicit time dependence into the formulae of PI, NTK-aware, and YaRN. A unifying observation is that all three methods effectively “shut-down” when the scaling factor s = 1, i.e., no change in context length. Specifically, in PI, we get g(m) = m/s = m; in NTK-aware, h(θd) = θds2d/(D−2) = θd; and YaRN, which combines the components of both PI and NTK-aware, likewise collapses to no scaling.

Consequently, we define the following family of timeparameterized scalings,

κ(t) = λs · tλ

, (13)

t

with tunable hyperparameters λs and λt. Early in sampling (t≈1), this formula yields near-maximal scaling κ(1)=λs; late in sampling (t≈0), it approaches no scaling κ(0)=1.

The exponent λt controls how scaling attenuates over time, allowing us to align the evolution of frequency emphasis with diffusion’s progression. The multiplier λs sets the maximal scaling that DYPE attains; in principle it reflects the ratio between the desired and the training context lengths.

Finally, let us now go through the resulting extrapolation strategies from plugging κ(t) into these methods, either by replacing the fixed scaling parameters s, or controlling the thresholds in YaRN.

[Figure 6]

DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion

"Surreal painting of a valley in Nara, ﬂoating cherry blossoms, golden rivers, single giant crane in pastel sky."

[Figure 7]

DY-PI. PI in Eq. 5 uses uniform position scaling. We make it step-aware by exponentiating the scale factor by κ(t):

[Figure 8]

| |
|---|

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

m sκ(t)

| | |
|---|---|

, h(θd,t) = θd. (14)

g(m,t) =

Early sampling steps (t ≈ 1) apply stronger compression to stabilize structure, while later steps (t ≈ 0) resolve finer detail.

| | |
|---|---|

YaRN Dy-YaRN

DY-NTK. NTK-aware interpolation in Eq. 6 rescales frequencies non-uniformly. Our time-aware variant generalizes this by multiplying the exponent with κ(t):

Figure 3. Zoom-in comparison at 40962 of DY-YaRN vs. YaRN. Additional example can be found in Fig. 17 in the Appendix.

θd

et al., 2024) (Sec. 4.2). We also include zoom-in studies to highlight improvements in preserving high-frequency details (Fig. 3). Furthermore, in Appendix D, we present an ablation study examining design choices, focusing on (i) scheduler variants for DY-NTK-aware and (ii) timestep incorporation strategies for DY-YaRN. Finally, additional results are provided in Appendix E, covering additional DiT-based architectures (Qwen-Image (Wu et al., 2025)), high-resolution video generation (Wan et al., 2025), and high-resolution image editing tasks, panorama generation, and more visual examples. Implementation details are provided in Appendix C.

sκ(t)·2d/(D−2). (15) In this scheme. the low frequencies are well-represented

g(m,t) = m, h(θd,t) =

- at the initial steps, at the cost of compressing the highfrequency band. As the sampling progresses, the lowfrequency modes converge, and the higher frequency band representation expands. An illustration of this approach is provided in Appendix A.

DY-YaRN. YaRN in Sec. 2.2 combines NTK-by-parts frequency scaling (Eq. 7) with global attention scaling (Eq. 9). Unlike the two methods above, here we introduce timedependence via κ(t) which dynamically adjusts the fixed ramp thresholds α and β in Eq. 8, resulting in

#### 4.1. Ultra-High-Resolution Text-to-Image Generation

 

In Sec. 4.1.1 we evaluate DYPE against PositionExtapolation-based approaches and in Sec. 4.1.2 we evaluate DYPE against more general baselines as custom in previous works (Bu et al., 2025).

- 0, r < α · κ(t), r−α·κ(t)

β·κ(t)−α·κ(t), α · κ(t) ≤ r ≤ β · κ(t),

- 1, r > β · κ(t),

(16)

γ(r,t) =



4.1.1. COMPARISON WITH POSITION-EXTRAPOLATION BASELINES

and since κ(t) is already multiplied by α and β, we set λs = 1, and hence κ(t) in this case reduces to

We evaluate DYPE on top of the pre-trained FLUX (Labs, 2024), specifically the FLUX.1-Krea-dev version, whose effective generation resolution is 1024 × 1024. As primary baselines, we use FLUX itself and, in test time only, apply on top of FLUX the positional–embedding extrapolation methods NTK-aware and YaRN, adapted to vision by applying them independently on the x and y axes. We also compare with Time-Aware Scaled RoPE (TASR) (Zhuo et al., 2024), which interpolates from PI to NTK-aware scaling as denoising advances (discussed in Appendix B). On top of these, we evaluate our DYPE, including both DY-NTK-aware and DY-YaRN.

κ(t) = tλ

. (17)

t

Being a monotonic increasing function, the scheduler κ(t) dynamically shifts the ramp boundaries towards 1, i.e., no scaling, as function of the sampling step t, which meets our design goal.

- 4. Experiments We evaluate the effectiveness of DYPE across multiple aspects of high-resolution image generation, covering both global structure (low-frequency aspects such as text–image alignment) and fine detail (high-frequency aspects such as texture fidelity).

Benchmarks. As for benchmarks, we first consider DrawBench (Saharia et al., 2022), a set of 200 text prompts for evaluating text-to-image models across multiple criteria. Following Ma et al. (2025); Chachy et al. (2025), we measure: (i) text-image alignment using CLIP-Score (Hessel et al., 2022), a similarity metric between image and text embeddings based on CLIP (Radford et al., 2021), (ii) human preference alignment using ImageReward (Xu et al., 2023),

We first apply DYPE on top of FLUX (Labs, 2024), with evaluations on two established benchmarks, DrawBench (Saharia et al., 2022) and Aesthetic-4K (Zhang et al., 2025a), including automatic metrics, human evaluation, and resolution-scaling analysis (Sec. 4.1). We then extend evaluation to class-conditional image synthesis on FiTv2 (Wang

Table 1. High-resolution image generation on DrawBench and Aesthetic-4K on multiple resolutions. Each row reports CLIPScore (CLIP), ImageReward (IR), Aesthetics (Aesth) for DrawBench, and CLIP, IR, Aesth, and FID for Aesthetic-4K. All methods are built on FLUX.

2048 × 3072 3072 × 2048

Method DrawBench Aesthetic-4K DrawBench Aesthetic-4K

CLIP↑ IR↑ Aesth↑ CLIP↑ IR↑ Aesth↑ FID↓ CLIP↑ IR↑ Aesth↑ CLIP↑ IR↑ Aesth↑ FID↓

FLUX 26.64 -0.28 5.14 28.64 0.32 6.11 186.31 26.56 0.16 5.33 28.74 0.97 6.17 148.29 NTK 27.68 0.21 5.31 29.13 0.99 6.49 180.87 27.28 0.51 5.39 28.97 1.17 6.25 146.74 TASR 27.86 0.30 5.15 29.13 0.97 6.12 201.40 27.40 0.55 5.22 29.05 1.15 5.95 186.21 DY-NTK 27.91 0.48 5.54 29.14 1.10 6.56 176.13 27.44 0.60 5.55 29.11 1.21 6.53 146.40 YaRN 28.27 0.52 5.63 29.28 1.01 6.59 179.54 27.79 0.62 5.48 29.12 1.24 6.49 147.12 DY-YaRN 28.43 0.71 5.69 29.44 1.17 6.61 179.51 28.17 0.81 5.68 29.20 1.28 6.51 146.84

3072 × 3072 4096 × 4096

Method DrawBench Aesthetic-4K DrawBench Aesthetic-4K

CLIP↑ IR↑ Aesth↑ CLIP↑ IR↑ Aesth↑ FID↓ CLIP↑ IR↑ Aesth↑ CLIP↑ IR↑ Aesth↑ FID↓

FLUX 25.11 -0.53 5.01 28.62 0.46 6.16 187.96 16.43 -1.97 3.29 25.50 -0.73 5.42 195.68 NTK 26.07 -0.14 5.05 28.68 0.96 6.45 182.38 17.49 -1.88 3.57 24.88 -0.54 5.50 203.85 TASR 26.87 0.18 5.01 28.79 1.00 6.01 194.23 21.21 -1.69 3.56 25.09 -0.09 5.96 221.39 DY-NTK 27.02 0.30 5.36 28.83 1.10 6.57 179.98 21.51 -1.22 4.25 28.06 0.79 6.42 183.72 YaRN 27.92 0.41 5.37 29.26 1.14 6.67 184.16 25.71 -0.34 4.85 28.57 0.85 6.47 192.19 DY-YaRN 28.12 0.66 5.55 29.75 1.24 6.70 179.82 26.94 0.15 5.17 29.28 1.09 6.67 186.00

“A woman with short hair and a black dress stands in a forest, holding an owl with large, outstretched wings…”

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

“A decorative vase with ﬂoral branches and white blossoms sits on a light cloth, accompanied by a shiny red apple.”

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

NTK-aware Dy-NTK-aware YaRN Dy-YaRN

Figure 4. Qualitative results at 40962 resolution using two representative prompts from Aesthetic-4K.

a reward model trained on large-scale human feedback for generated images, and (iii) image aesthetics using AestheticScore-Predictor (Schuhmann et al., 2022), a model trained to predict human aesthetic judgments. Additionally, to specifically assess fine-grained, ultra-high-resolution fidelity, we evaluate on Aesthetic-4K (Zhang et al., 2025a). We use its

4K subset (Aesthetic-Eval@4096), which comprises 195 curated image–prompt pairs, and downsample them to match the target test resolutions for fair comparison. Following the official protocol, we report (i) CLIPScore, (ii) ImageReward, (iii) Aesthetics score, and (iv) FID (Heusel et al., 2017), which assesses the fidelity and diversity of generated

images based on the distributional distance between real and generated features.

Results. Quantitative results across different resolutions and aspect ratios are presented in Tab. 1, with Fig. 4 showing side-by-side comparisons on Aesthetic-4K. Additional qualitative results are provided in the Appendix for DrawBench (Fig. 15) and Aesthetic-4K (Fig. 16). As can be seen in Fig. 1, FLUX exhibits repeating artifacts at ultra-high resolutions, revealing the periodicity of the sinusoidal positional encoding when extrapolated to larger spatial contexts,

- as further illustrated in Appendix A. We also observe that FLUX performs relatively better on landscape resolutions than portrait, likely reflecting a training-set bias. Notably, once DYPE is applied, this gap widens in favor of our approach on portrait settings as well, indicating that DYPE helps mitigate this limitation. Importantly, the advantage of DYPE becomes increasingly pronounced as the generation resolution grows (e.g., up to 30722 and 40962), underscoring the effectiveness of our method for ultra-high-resolution synthesis in diffusion transformers. Further visual results are presented in the Appendix.

Perceptual Evaluation. To complement the automatic metrics, we conduct a human study on a curated subset of 20 prompts from Aesthetic-4K, obtained by sampling every fourth entry to ensure uniform coverage. We consider 50 raters and present them with pairwise comparisons at 40962 resolution, generated on FLUX. Each prompt yields three comparisons: (i) NTK-aware vs. DY-NTK-aware, (ii) Time-Aware Scaled RoPE (TASR) vs. DY-NTK-aware, and (iii) YaRN vs. DY-YaRN. For each pair, participants answer the following three questions: (i) Which image is more aligned with the given text prompt? (ii) Which image has better overall geometry and structure (coherent shapes, correct proportions, fewer distortions) and (iii) Which image has more aesthetic and realistic textures and fine details? Results, summarized in Tab. 2 shows that DYPE consistently achieves superior quality, with preference rates ranging from about 70.5% to 94.2%.

Table 2. Human evaluation on Aesthetic4K. Each cell reports the percentage of pairwise comparisons in which DYPE was preferred.

Comp. Txt↑ Str↑ Det↑ NTK vs. DY-NTK 88.5 88.7 88.3 TASR vs. DY-NTK 70.5 80.6 94.2 YaRN vs. DY-YaRN 90.1 87.3 88.1

Resolution Scaling Analysis. We next investigate the resolution limit beyond which methods fail. Using 20 Aesthetic-4K prompts sampled at intervals of 10, we evaluate FLUX,

1.5

1.0

0.5

ImageReward

0.0

0.5

1.0

Dy-YaRN

1.5

YaRN FLUX

2.0

10242 20482 30722 40962 51202 61442

Resolution

Figure 5. Scaling analysis.

YaRN, and our DY-YaRN across six square resolutions from 10242 to 61442, reporting ImageReward (Fig. 5). The trend shows FLUX degrades sharply at 30722 and YaRN at 40962, while our method remains stable across scales until experiencing degradation at 61442.

- 4.1.2. COMPARISON WITH OTHER APPROACHES GENERAL BASELINES

In addition to PE approaches, we also compare with a broad set of non-PE methods spanning three families of highresolution diffusion pipelines:

- (i) DiT-based methods - UltraPixel (Ren et al., 2024) (which relies on SD3 (Esser et al., 2024a)), Diffusion4K (Zhang et al., 2025a) (which requires model finetuning), HiFlow (Bu et al., 2025), and I-Max (Du et al., 2024b) (both of which rely on multi-stage or progressive upscaling).
- (ii) U-Net–based methods - DemoFusion (Du et al., 2024a), FreCaS (Zhang et al., 2025b), DiffuseHigh (Kim et al., 2025), and FreeScale (Qiu et al., 2025).
- (iii) Diffusion + Super-Resolution - FLUX combined with BSRGAN (Zhang et al., 2021).

Benchmarks. Consistent with previous works (Bu et al., 2025), our evaluation focuses on patch-based fidelity and detail preservation. Specifically, we report: (i) FID (Heusel et al., 2017), (ii) patch-FID, (iii) Inception Score (IS) (Salimans et al., 2016), and (iv) patch-Inception Score (patchIS). These complementary metrics isolate local structure and texture quality at ultra-high resolutions, allowing us to more precisely assess how well each method maintains fine-grained detail.

Results. We evaluate at resolutions 20482 and 40962 to specifically assess local detail preservation at extreme scales. As can be seen in Tab. 3, at 20482 our DY-YaRN variant achieves the best performance across all four metrics among all compared methods. At 40962, DY-YaRN attains the best patch-FID and patch-IS, while the DY-YaRN+HiFlow combination yields the best global FID and IS. Notably, for every metric at both resolutions, at least one DYPE-based variant (either DY-YaRN or DY-YaRN+HiFlow) outperforms the baselines, highlighting the effectiveness of our approach. Additionally, in the Appendix, Fig. 14, we include qualitative comparison of our DY-YaRN variant with representative baselines.

- 4.2. Higher-Resolution Class-to-Image Generation After validating our method on text-to-image generation, we next test whether its consistency gains transfer to the core task of class-conditional generation on ImageNet (Russakovsky et al., 2015). We apply DYPE on FiTv2 (Wang

- Table 3. Evaluation at 20482 and 40962 resolutions on non-PE approaches. Baselines include U-Net-based methods (*) , tuning-based approaches (§), progressive refinement pipelines (‡), and diffusion combined with super-resolution (†).

2048 × 2048 4096 × 4096

Method FID↓ patch-FID↓ IS↑ patch-IS↑ FID↓ patch-FID↓ IS↑ patch-IS↑

DemoFusion* 205.59 199.00 10.03 10.49 205.86 195.69 10.93 7.92 FreCaS* 201.07 195.73 11.45 10.52 200.95 202.41 11.14 8.02 DiffuseHigh* 178.08 117.43 14.47 10.68 186.25 96.99 12.62 7.56 FreeScale* 199.87 126.45 9.89 10.55 259.24 191.23 10.66 7.74 I-Max‡ 174.35 107.81 13.91 9.77 187.29 87.71 13.44 5.78 FLUX+BSRGAN† 175.26 106.88 13.84 10.51 201.12 98.51 10.11 8.34 UltraPixel‡ 181.06 114.69 14.21 11.08 186.75 88.99 13.83 8.54 Diffusion-4K§ 178.25 98.35 13.41 10.37 198.16 94.82 13.82 4.72 HiFlow‡ 173.00 106.65 13.36 10.32 174.39 78.38 13.38 6.67

DY-YaRN (Ours) + HiFlow‡ 166.71 103.06 13.60 10.74 169.46 79.64 14.18 7.06 DY-YaRN (Ours) 142.74 96.34 14.76 11.95 186.00 78.33 13.96 10.13

- Table 4. ImageNet results on FiTv2-XL/2 comparing PI, NTK, TASR, YaRN and our DYPE variants. We report FID↓, sFID↓, Inception Score (IS)↑, Precision↑, and Recall↑ at 3202 and 3842.

FID↓ sFID↓ IS↑ Precision↑ Recall↑

3202 3842 3202 3842 3202 3842 3202 3842 3202 3842 FiTv2 5.79 38.90 13.7 49.51 233.03 99.28 0.75 0.39 0.55 0.57 PI 11.47 118.60 21.13 85.98 197.04 23.10 0.67 0.16 0.51 0.38 DY-PI 7.16 39.56 17.40 51.90 231.70 99.97 0.67 0.36 0.53 0.49 TASR 10.47 74.87 15.67 66.12 222.40 101.10 0.69 0.21 0.51 0.39 NTK 6.04 36.75 14.35 47.82 232.91 104.73 0.75 0.40 0.55 0.56 DY-NTK 5.22 36.04 14.29 47.46 233.11 106.45 0.75 0.42 0.57 0.56 YaRN 5.87 22.63 15.38 36.09 250.66 156.34 0.77 0.48 0.52 0.50 DY-YaRN 5.03 21.75 14.48 33.92 251.73 158.02 0.77 0.49 0.53 0.52

et al., 2024), a flexible DiT trained on multiple resolutions. Specifically, we use the FiTv2-XL/2 variant (675M parameters), which was trained at a maximum resolution of 256 × 256, and test it on resolutions 320 × 320 and 384 × 384. We compare the standard extrapolation methods (PI, NTK-aware, YaRN) and TASR against our DYPE variants (DY-PI, DY-NTK, DY-YaRN). All models are evaluated on the ImageNet validation set (50,000 images). We report FID (Heusel et al., 2017), sFID (Nash et al., 2021), Inception Score (IS) (Salimans et al., 2016), Precision, and Recall (Kynk¨a¨anniemi et al., 2019). Quantitative results are reported in Tab. 4, show that, as with FLUX, DYPE consistently improves over all vanilla baselines, with DYYaRN achieving the best overall performance. Notably, PI severely underperforms relative to base FiTv2, highlighting its ineffectiveness for image generation due to the loss of high-frequency details.

- 5. Related Work

2023; Ramesh et al., 2022), DiTs instead adopt transformerbased backbones that naturally capture global context and scale effectively with model and data size, enabling increasingly capable text-to-image models such as FLUX (Labs, 2024), Stable-Diffusion-3 (Esser et al., 2024b) and subsequent advances (Gao et al., 2024; Liu et al., 2024a; Chen et al., 2023a). Yet, training these architectures on ultra-high resolutions (e.g., 4K and beyond) remains an open challenge due to the quadratic cost of self-attention, which quickly becomes prohibitive in both memory and computation at such resolutions.

Ultra-High Resolution Image Synthesis. Despite this limitation, many works explored fine-tuning diffusion models on higher-resolution (Liu et al., 2025; Cheng et al., 2025; Hoogeboom et al., 2023; Liu et al., 2024b; Ren et al., 2024; Teng et al., 2023; Zheng et al., 2024; Zhang et al., 2025a; Huang et al., 2024), yet these remain limited in their ability to scale to ultra-high resolutions due to the expensive tuning phase. Alternatively, patch-based methods (Bar-Tal et al., 2023; Du et al., 2024a; He et al., 2023) aim to reduce costs by stitching generated regions, yet often suffer from duplication and local repetition. Input-level techniques suppress undesired semantics (Lin et al., 2024b; Liu et al., 2024c), but are limited to small artifacts and risk information leakage. Complementary strategies improve high-resolution fidelity within U-Net architectures by modifying internal feature processing, such as FreeU (Si et al., 2024), which enhances skip-connection feature mixing, and FAM-Diffusion (Yang et al., 2025), which introduces frequency modulation for sharper high-resolution outputs. More recently, tuning-free methods that synthesize full images without retraining (Qiu et al., 2025; Cao et al., 2025; Haji-Ali et al., 2024; Hwang et al., 2024; Jin et al., 2023; Kim et al., 2025; Lee et al., 2023; Lin et al., 2024a; Zhang et al., 2024) offer a practical alternative, but since all such approaches rely on U-Net backbones, adapting them to DiTs is non-trivial, leaving a critical gap for transformer-based methods capable of true

Diffusion Transformers. DiT (Peebles & Xie, 2022) have recently emerged as the leading architecture for diffusionbased text-to-image generation (Ho et al., 2020; Song et al.,

- 2020). While U-Nets (Ronneberger et al., 2015) underpinned earlier advances (Rombach et al., 2022; Podell et al.,

end-to-end ultra-high-resolution generation.

Position Extrapolation Schemes. The challenge of ultra–high-resolution generation in DiTs closely mirrors that of long-context generation in language models, often tackled through advances in positional encoding. RoPE (Su et al.,

- 2021) dominates this space, with extrapolation framed as frequency scaling: PI (Chen et al., 2023b) compresses positions to limit phase drift, while NTK-aware and YaRN (Peng

- et al., 2023) rescale frequencies to stabilize low modes and suppress unstable high ones. Inspired by these advances, vision models have begun to adopt these techniques. FiT (Lu
- et al., 2024) and FiT-v2 (Wang et al., 2024) introduce VisionPI, Vision-NTK, and Vision-YaRN within DiTs by applying these frequency-scaling techniques independently to the horizontal and vertical axes. While this approach allows for flexible aspect-ratio generation and modest resolution gains, it remains a generic solution that overlooks the low-to-high frequency progression inherent to diffusion. RIFLEx (Zhao
- et al., 2025) demonstrates that frequency-aware extrapolation can also be effective in DiTs for video, enabling substantial temporal length extension. However, RIFLEx focuses exclusively on the temporal axis and does not address spatial resolution scaling. Lumina-Next (Zhuo et al.,

2024) incorporates timestep dynamics by interpolating from PI to NTK-aware scaling as denoising advances. Yet, its heavy reliance on interpolation throughout the denoising process suppresses high frequencies, yielding blurry outputs. Our work, instead, directly analyzes the diffusion process frequency progression, leading to a principled approach that preserves fine-grained detail without compromising structural fidelity.

### 6. Conclusion

We presented DYPE, a training-free approach enabling diffusion transformers to synthesize ultra-high-resolution images without retraining or additional sampling overhead. Our method stems from a Fourier-space analysis of the samples’ spectrum evolution during the diffusion sampling process, revealing that low-frequency content converges faster than the higher frequency bands. This regularity allows DYPE to better represent the evolving frequencies in the PE dynamically as well as enable the denoiser to operate more effectively within its training conditions.

As demonstrated on a pre-trained FLUX model, this strategy enables generation at unprecedented resolutions. Extensive qualitative and quantitative evaluations consistently confirm that DYPE offers superior generalization over existing static extrapolation techniques, with its advantage growing

- at higher resolutions.

As future work, we aim to pursue even more ambitious resolutions, not only through inference-time scaling but also

by incorporating time-dependent positional extrapolation into a light tuning phase.

### Impact Statement

This paper presents work whose goal is to advance the efficiency and scalability of diffusion transformers for ultrahigh-resolution image synthesis. Our research does not involve human subjects, personal data, or deployment in user-facing systems. All experiments were conducted on widely adopted, publicly available benchmarks.

We acknowledge that advancements in high-fidelity image generation carry potential societal consequences, particularly regarding the risk of misuse for generating deceptive or harmful content. However, our contributions are strictly focused on methodological improvements and evaluation on standardized datasets. We believe that improving the efficiency of these models does not inherently increase these risks beyond those already well-established in the field of generative modeling. We are committed to the principles of responsible AI research and transparency in methodology.

### References

Bar-Tal, O., Yariv, L., Lipman, Y., and Dekel, T. Multidiffusion: Fusing diffusion paths for controlled image generation. 2023.

Bu, J., Ling, P., Zhou, Y., Zhang, P., Wu, T., Dong, X., Zang, Y., Cao, Y., Lin, D., and Wang, J. Hiflow: Trainingfree high-resolution image generation with flow-aligned guidance. arXiv preprint arXiv:2504.06232, 2025.

Cao, B., Ye, J., Wei, Y., and Shan, H. Repldm: Reprogramming pretrained latent diffusion models for high-quality, high-efficiency, high-resolution image generation, 2025. URL https://arxiv.org/abs/2410.06055.

Chachy, I., Yariv, G., and Benaim, S. Rewardsds: Aligning score distillation via reward-weighted sampling, 2025. URL https://arxiv.org/abs/2503.09601.

Chen, J., Yu, J., Ge, C., Yao, L., Xie, E., Wu, Y., Wang, Z., Kwok, J., Luo, P., Lu, H., and Li, Z. Pixart-α: Fast training of diffusion transformer for photorealistic text-toimage synthesis, 2023a. URL https://arxiv.org/ abs/2310.00426.

Chen, S., Lin, Z., Chen, Z., Ren, S., He, J., Chen, Z., Ma, S., Chen, W., Tang, J., and Sun, M. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023b.

Cheng, J., Xie, P., Xia, X., Li, J., Wu, J., Ren, Y., Li, H., Xiao, X., Wen, S., and Fu, L. Resadapter: Domain consistent resolution adapter for diffusion models. In Proceed-

ings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 2438–2446, 2025.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. URL https://arxiv. org/abs/1810.04805.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer,

- M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
- N. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021.

Du, R., Chang, D., Hospedales, T., Song, Y.-Z., and Ma, Z. Demofusion: Democratising high-resolution image generation with no $$$. In CVPR, 2024a.

Du, R., Liu, D., Zhuo, L., Qi, Q., Li, H., Ma, Z., and Gao, P. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow, 2024b. URL https://arxiv.org/abs/2410.07536.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R. Scaling rectified flow transformers for high-resolution image synthesis, 2024a. URL https://arxiv.org/abs/2403.

03206.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R. Scaling rectified flow transformers for high-resolution image synthesis, 2024b. URL https://arxiv.org/abs/2403.

03206.

Gao, P., Zhuo, L., Liu, D., Du, R., Luo, X., Qiu, L., Zhang, Y., Lin, C., Huang, R., Geng, S., Zhang, R., Xi, J., Shao, W., Jiang, Z., Yang, T., Ye, W., Tong, H., He, J., Qiao, Y., and Li, H. Lumina-t2x: Transforming text into any modality, resolution, and duration via flow-based large diffusion transformers, 2024. URL https://arxiv.

org/abs/2405.05945.

Haji-Ali, M., Balakrishnan, G., and Ordonez, V. Elasticdiffusion: Training-free arbitrary size image generation through global-local content separation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6603–6612, 2024.

He, Y., Yang, S., Chen, H., Cun, X., Xia, M., Zhang, Y., Wang, X., He, R., Chen, Q., and Shan, Y. Scalecrafter:

Tuning-free higher-resolution visual generation with diffusion models. In The Twelfth International Conference on Learning Representations, 2023.

Heo, B., Park, S., Han, D., and Yun, S. Rotary position embedding for vision transformer. In European Conference on Computer Vision, pp. 289–305. Springer, 2024.

Hessel, J., Holtzman, A., Forbes, M., Bras, R. L., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning, 2022. URL https://arxiv.org/ abs/2104.08718.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Hoogeboom, E., Heek, J., and Salimans, T. simple diffusion: End-to-end diffusion for high resolution images. In International Conference on Machine Learning, pp. 13213–13232. PMLR, 2023.

Huang, L., Fang, R., Zhang, A., Song, G., Liu, S., Liu, Y., and Li, H. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In European conference on computer vision, pp. 196–212. Springer, 2024.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. Vbench: Comprehensive benchmark suite for video generative models, 2023. URL https://arxiv.org/abs/2311.

17982.

Hwang, J., Park, Y.-H., and Jo, J. Upsample guidance: Scale up diffusion models without training. arXiv preprint arXiv:2404.01709, 2024.

Hyvrinen, A., Hurri, J., and Hoyer, P. O. Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. Springer Publishing Company, Incorporated, 1st edition, 2009. ISBN 1848824904.

Jin, Z., Shen, X., Li, B., and Xue, X. Training-free diffusion model adaptation for variable-sized text-to-image synthesis. Advances in Neural Information Processing Systems, 36:70847–70860, 2023.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

Kim, Y., Hwang, G., Zhang, J., and Park, E. Diffusehigh: Training-free progressive high-resolution image synthesis through structure guidance. In Proceedings of the AAAI conference on artificial intelligence, volume 39, pp. 4338– 4346, 2025.

Kynk¨a¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., and Aila, T. Improved precision and recall metric for assessing generative models, 2019. URL https://arxiv. org/abs/1904.06991.

Labs, B. F. Flux. https://github.com/ black-forest-labs/flux, 2024.

LeCun, Y. and Bengio, Y. Convolutional networks for images, speech, and time series. The handbook of brain theory and neural networks, 1998.

Lee, Y., Kim, K., Kim, H., and Sung, M. Syncdiffusion: Coherent montage via synchronized joint diffusions. Advances in Neural Information Processing Systems, 36: 50648–50660, 2023.

Lin, M., Lin, Z., Zhan, W., Cao, L., and Ji, R. Cutdiffusion: A simple, fast, cheap, and strong diffusion extrapolation method. arXiv preprint arXiv:2404.15141, 2024a.

Lin, Z., Lin, M., Zhao, M., and Ji, R. Accdiffusion: An accurate method for higher-resolution image generation. In European Conference on Computer Vision, pp. 38–53. Springer, 2024b.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 [cs.LG], 2022. https:// arxiv.org/abs/2210.02747.

- Liu, B., Akhgari, E., Visheratin, A., Kamko, A., Xu, L., Shrirao, S., Lambert, C., Souza, J., Doshi, S., and Li, D. Playground v3: Improving text-to-image alignment with deep-fusion large language models, 2024a. URL https://arxiv.org/abs/2409.10695.
- Liu, C., Hou, L., Zheng, M., Tao, X., Wan, P., Zhang, D., and Gai, K. Boosting resolution generalization of diffusion transformers with randomized positional encodings, 2025. URL https://arxiv.org/abs/2503. 18719.

Liu, S., Yu, W., Tan, Z., and Wang, X. Linfusion: 1 gpu, 1 minute, 16k image. arXiv preprint arXiv:2409.02097, 2024b.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Liu, X., He, Y., Guo, L., Li, X., Jin, B., Li, P., Li, Y., Chan, C.-M., Chen, Q., Xue, W., et al. Hiprompt: Tuningfree higher-resolution generation with hierarchical mllm prompts. arXiv preprint arXiv:2409.02919, 2024c.

Lu, Z., Wang, Z., Huang, D., Wu, C., Liu, X., Ouyang, W., and Bai, L. Fit: Flexible vision transformer for diffusion model, 2024. URL https://arxiv.org/abs/ 2402.12376.

Ma, N., Tong, S., Jia, H., Hu, H., Su, Y.-C., Zhang, M., Yang, X., Li, Y., Jaakkola, T., Jia, X., and Xie, S. Inferencetime scaling for diffusion models beyond scaling denoising steps, 2025. URL https://arxiv.org/abs/ 2501.09732.

Nash, C., Menick, J., Dieleman, S., and Battaglia, P. W. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.

Peebles, W. and Xie, J.-Y. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models, 2023. URL https://arxiv.org/abs/2309. 00071.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Press, O., Smith, N. A., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

Qiu, H., Zhang, S., Wei, Y., Chu, R., Yuan, H., Wang, X., Zhang, Y., and Liu, Z. Freescale: Unleashing the resolution of diffusion models via tuning-free scale fusion, 2025. URL https://arxiv.org/abs/2412.09626.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with

clip latents, 2022. URL https://arxiv.org/abs/ 2204.06125.

Ren, J., Li, W., Chen, H., Pei, R., Shao, B., Guo, Y., Peng, L., Song, F., and Zhu, L. Ultrapixel: Advancing ultra high-resolution image synthesis to new peaks. Advances in Neural Information Processing Systems, 37:111131– 111171, 2024.

Rissanen, S., Heinonen, M., and Solin, A. Generative modelling with inverse heat dissipation, 2023. URL https://arxiv.org/abs/2206.13397.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Ronneberger, O., Fischer, P., and Brox, T. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. Imagenet large scale visual recognition challenge, 2015. URL https:// arxiv.org/abs/1409.0575.

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S., Lopes, R. G., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image diffusion models with deep language understanding, 2022. URL https://arxiv.org/abs/2205.11487.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training gans, 2016. URL https://arxiv.org/abs/ 1606.03498.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35: 25278–25294, 2022.

Si, C., Huang, Z., Jiang, Y., and Liu, Z. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4733–4743, 2024.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Su, J., Lu, Y., Pan, S., Wei, B., and Zhu, Y. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Teng, J., Zheng, W., Ding, M., Hong, W., Wangni, J., Yang, Z., and Tang, J. Relay diffusion: Unifying diffusion process across resolutions for image synthesis. arXiv preprint arXiv:2309.03350, 2023.

van der Schaaf, A. and van Hateren, J. Modelling the power spectra of natural images: Statistics and information. Vision Research, 36(17): 2759–2770, 1996. ISSN 0042-6989. doi: https://doi.org/10.1016/0042-6989(96)00002-8. URL https://www.sciencedirect.com/ science/article/pii/0042698996000028.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models, 2025. URL https:

//arxiv.org/abs/2503.20314.

Wang, Z., Lu, Z., Huang, D., Zhou, C., Ouyang, W., , and Bai, L. Fitv2: Scalable and improved flexible vision transformer for diffusion model, 2024. URL https: //arxiv.org/abs/2410.13925.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., ming Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., and Liu, Z. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 15903–15935, 2023.

Yang, H., Bulat, A., Hadji, I., Pham, H. X., Zhu, X., Tzimiropoulos, G., and Martinez, B. Fam diffusion: Frequency and attention modulation for high-resolution image generation with stable diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2459–2468, 2025.

- Zhang, J., Huang, Q., Liu, J., Guo, X., and Huang, D. Diffusion-4k: Ultra-high-resolution image synthesis with latent diffusion models, 2025a. URL https://arxiv. org/abs/2503.18352.
- Zhang, K., Liang, J., Van Gool, L., and Timofte, R. Designing a practical degradation model for deep blind image super-resolution. In IEEE International Conference on Computer Vision, pp. 4791–4800, 2021.

Zhang, S., Chen, Z., Zhao, Z., Chen, Y., Tang, Y., and Liang, J. Hidiffusion: Unlocking higher-resolution creativity and efficiency in pretrained diffusion models. In European Conference on Computer Vision, pp. 145–161. Springer, 2024.

Zhang, Z., Li, R., and Zhang, L. Frecas: Efficient higherresolution image generation via frequency-aware cascaded sampling, 2025b. URL https://arxiv.org/ abs/2410.18410.

Zhao, M., He, G., Chen, Y., Zhu, H., Li, C., and Zhu, J. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

Zheng, Q., Guo, Y., Deng, J., Han, J., Li, Y., Xu, S., and Xu, H. Any-size-diffusion: Toward efficient text-driven synthesis for any-size hd images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 7571–7578, 2024.

Zhuo, L., Du, R., Xiao, H., Li, Y., Liu, D., Huang, R., Liu, W., Zhu, X., Wang, F.-Y., Ma, Z., et al. Luminanext: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems, 37: 131278–131315, 2024.

### A. Illustration of DYPE

- Fig. 6 illustrates the behavior of RoPE frequencies under different scaling strategies, highlighting how our approach compares with position extrapolation methods.

B. Comparison Between DYPE and Lumina-Next

The frequency allocation strategy behind DYPE is based on two complementary observations made in Sec. 3.1. The first related to the fact that low-frequency modes converge early in the sampling process, whereas the high frequency bands are resolved throughout the process. The second, is related to the trade-off exiting extrapolation method must take when trying to capture the entire spectrum of the larger resolution using the the fixed number of representable modes in the mode, D. Thus, rather than pinpointing both ends of the spectrum at all times and accommodating these trade-offs, DYPE, exploits the fact that low-frequencies are resolved earlier to better represent the higher bands and reduce the extrapolation compression.

The possibility of time-aware position extrapolation was briefly discussed in Zhuo et al. (2024) as part of introducing a new DiT architecture. However, the opposite conclusions were drawn by the authors. Specifically, their scheme starts by representing only the low-frequency band via PI (discarding high frequencies), and then switching to NTK-aware extrapolation that trades-off high frequency representation, in favor of low frequencies, which according to our analysis

- in Sec. 3.1 have already converged. We also note that in this scheme, the denoiser is not operating under the PE it was trained with unlike the case of DYPE.

- Fig. 7 illustrates the complementary strategies of DYPE specifically DY-NTK-aware, and Time-Aware Scaled RoPE (Zhuo et al., 2024) in terms of the wavelengths they cover throughout the sampling.

Quantitative and Qualitative Comparison with TimeAware Scaled RoPE. We conducted an experiment by applying DY-NTK-aware and Lumina-Next Time-Aware Scaled RoPE on top of the same pre-trained model, FLUX. Both methods are evaluated on the Aesthetic-4K benchmark using CLIPScore, ImageReward, Aesthetic-Score, and FID. For a better context, we also report the NTK-aware results.

The results in Tab. 5 show that DY-NTK-aware achieves the best performance across all metrics. Additionally, a qualitative comparison is provided in Fig. 8

### C. Implementation Details

Unless otherwise stated, all experiments are conducted on a single L40S GPU. We set α = 1, β = 32, and use an

effective resolution of L = 1024. Diffusion inference is performed with 28 sampling steps. For our method, we apply λs = λt = 2. Code will be released upon acceptance.

### D. Ablation Study

We perform an ablation study to better understand the role of specific design choices in DYPE, specifically, we consider alternative weighting schedulers for (i) DY-NTK-aware and (ii) DY-YaRN.

Scheduler designs for DY-NTK-aware. A central motivation for this ablation is to test how best to incorporate the low-to-high nature of diffusion into NTK-aware extrapolation. Recall from Sec. 2.2 that NTK-aware interpolation rescales each RoPE frequency θd as

θd s2d/(D−2), (18)

h(θd) =

compressing low frequencies more while preserving higher ones. However, this scaling is fixed across all denoising steps and thus agnostic to the diffusion dynamics.

In DY-NTK-aware, we introduce a timestep-dependent scheduler κ(t) to allow the effective frequency scaling to evolve with the diffusion timestep t. Here, we consider two ways the scheduler can interact with the NTK-aware rescaling factor s from Eq. 6: (i) Multiplicative scaling, where the scheduler linearly modulates the compression,

θd (s · κ(t))

, (19)

h(θd,t) =

2d D−2

and (ii) Exponential scaling, where the scheduler exponentiates the compression,

θd sκ(t)·

. (20)

h(θd,t) =

2d D−2

In both cases, the scheduler is defined by the following family of time-parameterized scalings from Eq. 13:

κ(t) = λs · tλ

, (21)

t

with λs and λt controlling the magnitude and progression of the scheduler.

We ablate along two axes. First, we fix λt = 1 and vary λs ∈ {1,1.5,2,2.5} to identify the best magnitude scaling. Then, we fix λs = 2 (the winner) and vary λt ∈ {0.5,1,2}, corresponding to sublinear, linear, and exponential progression. We also compare against an NTK-aware variant with λs = 2 for completeness.

Results are summarized in Tab. 6, showing that increasing the initial scaling (toward position interpolation) improves

RoPE without scaling

1.0

Pre-trained range

Low freq (period=4096)

Unseen range

High freq (period=512)

| |
|---|

| |
|---|

0.5

RoPE

0.0

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

| |
|---|

0.5

| |
|---|

| |
|---|

1.0

0 1000 2000 3000 4000 5000 6000 7000 8000

Position (token index)

(a)

Position Interpolation (PI)

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| | | | | | |
| | | | | | | | | | | |
| | | || |
|---|
<br><br>| |
|---|
|| |
|---|
<br><br>| |
|---|
| | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

1.0

0.5

RoPE

0.0

0.5

1.0

0 1000 2000 3000 4000 5000 6000 7000 8000

Position (token index)

(b)

NTK-Aware Interpolation

1.0

| |
|---|

| |
|---|

0.5

| |
|---|

RoPE

0.0

| |
|---|

| |
|---|

| |
|---|

0.5

| |
|---|

| |
|---|

| |
|---|

| |
|---|

1.0

| |
|---|

0 1000 2000 3000 4000 5000 6000 7000 8000

Position (token index)

(c)

Dy-NTK-Aware t = 0.2

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | || |
|---|
| || |
|---|
<br><br>| | |
| | | | | | || |
|---|
|| |
|---|
|| |
|---|
|| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
| |
| | | | | | | | | | | |
| | | | | | | | | || |
|---|
| |
| | | | | | | | | | | |
| | | | | | | | | | | |

1.0

0.5

RoPE

0.0

0.5

1.0

0 1000 2000 3000 4000 5000 6000 7000 8000

Position (token index)

(d)

Dy-NTK-Aware t = 0.8

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | || |
|---|
|| |
|---|
|| |
|---|
| |
<br><br>| |
|---|
| |
| | | | | | | | | | | |
| | | | | | || |
|---|
|| |
|---|
<br><br>|| |
|---|
<br><br>| |
|---|
<br><br>| | |
| | | | | | | | | | | |
| | | | | | | | | | | |

1.0

0.5

RoPE

0.0

0.5

1.0

0 1000 2000 3000 4000 5000 6000 7000 8000

Position (token index)

(e)

- Figure 6. Frequency Behavior Across Scaling Strategies. (a) RoPE without scaling. (b) Position Interpolation (PI) where the sinusoidal curves are unchanged but the positions are normalized. (c) NTK-Aware Interpolation (frequency-dependent normalization; low frequency normalized more than high). (d–e) Dy-NTK-Aware (ours): our method dynamically interpolates between RoPE and NTK-aware by blending their effective periods as a function of the diffusion timestep t (shown here for t=0.2—close to image—and t=0.8—close to noise). Across panels, low frequency is shown in blue and high frequency in orange; training-context markers use filled circles, and test-context markers use hollow squares. Shaded backgrounds indicate pretrained (left) and unseen (right) position ranges.

Table 5. Comparison of NTK-aware, Time-Aware Scaled RoPE, and DY-NTK-aware on the Aesthetic-4K benchmark

Method CLIPScore ↑ ImageReward ↑ Aesthetic-Score ↑ FID ↓ NTK-aware 24.88 -0.54 5.50 203.85 Time-Aware Scaled RoPE 25.09 -0.09 5.96 221.39 DY-NTK-aware 28.06 0.79 6.42 183.72

|| |
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
|---|
<br><br>| |
|---|
<br><br>Extrapolation<br><br>Interpolation<br><br>NTK-aware<br><br>Time-aware<br><br>Dy-NTK-aware| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

4 6 8 10 12 14 16 18 20 22 24 Dimension Index

10

20

30

40

50

60

Wavelength

- Figure 7. Wavelengths of the RoPE embeddings under different strategies. Solid curves show the baseline methods: Extrapolation (no scaling), PI, and NTK-aware. Dashed curves depict dynamic variants: Time-aware interpolates NTK-aware with PI, while DyNTK-aware interpolates NTK-aware with Extrapolation.

structural fidelity (CLIP), while faster attenuation with t (toward complete position extrapolation) yields more aesthetic outputs. The exponential scheduler with λs = 2 and λt = 2 achieves the best balance between these objectives.

Scheduler designs for DY-YaRN. Building on the ablation study of DY-NTK-aware, we explore how to incorporate timestep dynamics into YaRN’s frequency-dependent interpolation. Recall from Sec. 2.2 that YaRN introduces a weight γ(r). YaRN smoothly interpolates between PI and no scaling. Specifically, YaRN rescales each RoPE frequency θd as:

##### h(θd) = (1 − γ(r(d))) θ

##### s + γ(r(d))θd, (22)

d

where r(d) = L/λd. The ramp function γ(r) smoothly transitions between PI-like stretching and no scaling:

 

- 0, r < α, r−α β−α, α ≤ r ≤ β,

- 1, r > β,

(23)

γ(r) =



where α,β are hyperparameters setting the bands’ boundaries.

“A serene lake reﬂecting mountains and forested hills under a dramatic sky, with sunlight illuminating the trees on the opposite shore and rocky banks lining the water's edge.”

[Figure 22]

[Figure 23]

"A woman in a vintage, elegantly tailored gown with intricate embroidery gazes thoughtfully over a balcony, with a scenic river and historic buildings in the background."

[Figure 24]

[Figure 25]

Time-Aware Scaled RoPE Dy-NTK-aware

- Figure 8. Qualitative comparison between DYPE and Time-Aware Scaled RoPE (Lumina-Next) on the Aesthetic-4K benchmark.

- Table 6. Comparison of scheduler designs for DY-NTK-aware on FLUX at 30722 resolution. Evaluated on 50 DrawBench prompts (sampled every 4th index). Baselines (FLUX, NTK-Aware) are included. Metrics: CLIP-Score (CLIP↑), ImageReward (IR↑), and Aesthetics-Score (Aesth↑).

Variant λs λt CLIP↑ IR↑ Aesth↑

FLUX - - 25.33 -0.52 5.12 NTK-Aware - - 25.83 -0.13 4.99

- Multiplicative 1.0 1.0 25.67 0.11 5.08

- Multiplicative 1.5 1.0 25.75 0.10 5.18

- Multiplicative 2.0 1.0 26.09 0.16 5.31

- Multiplicative 2.5 1.0 26.38 0.21 5.34

Multiplicative 2.0 0.5 26.28 0.21 5.34 Multiplicative 2.0 2.0 26.12 0.17 5.40

- Exponential 1.0 1.0 25.81 -0.13 5.03

- Exponential 1.5 1.0 26.02 0.10 5.26

- Exponential 2.0 1.0 26.52 0.29 5.39

- Exponential 2.5 1.0 26.21 0.10 5.35

Exponential 2.0 0.5 26.69 0.24 5.34 Exponential 2.0 2.0 26.51 0.30 5.41

This can be viewed as partitioning frequencies into bands: low frequencies (small d) receive PI-like uniform scaling

- (γ(r) = 0), while high frequencies undergo no scaling
- (γ(r) = 1), while mid bands frequencies smoothly interpolate between the two by performing NTK-aware rescaling.

To leverage the low-to-high dynamics of diffusion, we introduce timestep dependence in three fashions: (i) Apply scheduler κ(t) to the mid-level NTK-aware components, similarly to DY-NTK-aware in Eq. 15. (ii) Weight modulation: Apply scheduler κ(t) to the ramp γ parameters α,β, effectively shifting the frequency bands assigned to each scaling regime as denoising progresses. (iii) Combined: Apply both κ(t) to the threshold and NTK components simultaneously.

Following Sec. D, we use the best scheduler configuration (exponential with λs = 2,λt = 2). For the thresholds scheduler, we found that the best performing scheduler is, κ(t) = t2. Intuitively, (i) controls how aggressively mid bands frequencies are compressed at each timestep, while (ii) controls which frequencies are considered “high”, “mid” and “low” as a function of t.

Results in Tab. 7 show that considering only κ(t) performs best. Further exhibiting our key idea—the fact that the diffusion process unfolds in a low-to-high manner, where early timesteps benefit from broader coverage of low frequencies, while later ones require sharper high-frequency detail. By modulating the ramp parameters α,β through κ(t), the model adaptively reassigns frequencies between low, mid, and high bands in synchrony with the denoising trajectory. This dynamic partitioning allows YaRN to better capture large-scale structure early on while still allocating capacity to finer details as synthesis progresses, thereby yielding more coherent and visually appealing generations.

### E. Additional Results

Evaluation on Additional DiT-Based Models. To demonstrate the generalization capabilities of our approach beyond FLUX, we implemented DYPE on QwenImage (Wu et al., 2025). We compared the vanilla model, the static YaRN baseline, and our dynamic variant, DY-YaRN. All methods were evaluated on the Aesthetic-4K benchmark using the metrics established in our main experiments. As shown in Tab. 8, DY-YaRN achieves the best performance across all metrics (CLIPScore, ImageReward, AestheticScore, and FID), validating the robustness and effectiveness of our method across different model architectures. Qualitative comparisons, illustrated in Fig. 9, further confirm that DY-YaRN produces superior visual quality with fewer artifacts compared to the baselines.

High-Resolution Video Generation. We extended our evaluation to the video domain by applying DYPE to the Wan2.1 1.3B model (Wan et al., 2025). While the original model has an effective resolution of 832×480, we assessed generation capabilities at a higher resolution of 1280 × 720. Due to GPU memory constraints, we fixed the sequence length to 33 frames. We compared our approach, specifically DY-YaRN, against the vanilla Wan2.1 model and YaRN using the VBench benchmark (Huang et al., 2023).

The quantitative results are summarized in Tab. 9. DYYaRN outperforms the vanilla model across all categories. Notably, while YaRN suffers a degradation in the Imaging Quality score compared to the vanilla baseline, our dynamic approach improves it. Figure 10 provides a qualitative comparison.

- Table 7. Comparison of scheduler application strategies for DY-YaRN on FLUX at 30722 resolution. Evaluated on 50 DrawBench prompts (sampled every 4th index), with baselines (FLUX, YaRN) included. Metrics: CLIP-Score (CLIP↑), ImageReward (IR↑), and Aesthetics-Score (Aesth↑). All experiments use the best scheduler configuration from Sec. D (exponential with λs = 2, λt = 2).

Variant NTK term κ(t) By-parts κ(t) CLIP↑ IR↑ Aesth↑

FLUX - - 25.33 -0.52 5.12 YaRN - - 27.32 0.36 5.47

κ(t) on NTK only ✓ - 27.35 0.37 5.50 κ(t) on by-parts only - ✓ 27.78 0.58 5.56 κ(t) on NTK & κ(t) on by-parts ✓ ✓ 27.76 0.36 5.41

Table 8. Comparison of Qwen-Image, YaRN, and DY-YaRN on the Aesthetic-4K benchmark.

Method CLIPScore ↑ ImageReward ↑ Aesthetic-Score ↑ FID ↓ Qwen-Image (Vanilla) 27.98 -0.26 5.52 201.15 Qwen-Image + YaRN 28.71 0.60 6.17 199.24 Qwen-Image + DY-YaRN 28.85 0.74 6.20 197.38

Table 9. Comparison of Wan2.1, YaRN, and DY-YaRN on high-resolution video generation (1280 × 720) using VBench.

Method Subj. Const. ↑ Bg. Const. ↑ Mot. Smooth. ↑ Dyn. Deg. ↑ Aesth. Qual. ↑ Img. Qual. ↑

Wan 2.1 (Vanilla) 0.9170 0.9518 0.9791 0.6532 0.4035 0.5107 Wan 2.1 + YaRN 0.9262 0.9513 0.9868 0.7350 0.4979 0.4364 Wan 2.1 + DY-YaRN 0.9303 0.9536 0.9899 0.8023 0.5095 0.6127

High-Resolution Image Editing. To demonstrate DyPE’s versatility, we further integrated it with the image editing model Qwen-Image-Edit-2509 (Wu et al., 2025). We evaluated performance on multi-concept composition, a challenging task requiring the seamless integration of distinct reference objects into a unified high-resolution scene. Experiments were conducted at an ultra-high resolution of 2656 × 2656. As shown in Fig. 11, the baseline (vanilla Qwen-Image-Edit-2509) tends to suffer from object duplication, such as the cats and baskets. In contrast, DyPE effectively mitigates these repetition artifacts, ensuring more precise object integration.

“A Moai statue gazes upward against a starry night sky, with a colorful sunset illuminating the horizon.”

[Figure 26]

[Figure 27]

[Figure 28]

“A woman’s proﬁle emerges among swirling clouds, with a deep gaze and delicate earrings, set against a gradient of blues and hints of gold.”

[Figure 29]

[Figure 30]

[Figure 31]

Panoramic Image Generation. We investigate DYPE’s ability to handle extreme aspect ratios, focusing on panoramic images (3 : 1, 4096 × 1365). Such generation poses challenges for position encoding, as large horizontal spans can intensify aliasing and spatial inconsistencies. We evaluate on 20 prompts from Aesthetic-4K (every 10th entry), comparing DY-YaRN with YaRN and FLUX using CLIP-Score, ImageReward, and Aesthetics-Score. As shown in Table 10, DY-YaRN consistently outperforms YaRN, suggesting strong suitability for extreme spatial layouts. Figure 12 shows that YaRN fails to maintain correct aspect ratio proportion, leading to distorted object placement, while DY-YaRN preserves coherent spatial structure across the panorama.

Qwen-Image YaRN Dy-YaRN

- Figure 9. Qualitative comparison of Qwen-Image, YaRN, and DY-YaRN (both adapted on top of Qwen-Image) at a resolution of 4096 × 4096.

"A teddy bear is swimming in the ocean."

Wan 2.1

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

"A squirrel eating a burger."

YaRN

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

DyYaRN

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

- Figure 10. Qualitative comparison between Wan 2.1, YaRN, and DY-YaRN, both on top of Wan 2.1, applied to video generation at 1280 × 720.

Additional Qualitative Results. We present a collage of multi- and high-resolution outputs (see Fig. 13), all gener-

### F. Attention Entropy Analysis

Input images

"The golden retriever and the tabby cat interacting playfully together in a lush green backyard garden."

[Figure 50]

Recent work (Jin et al., 2023) suggests that a primary reason trained attention mechanisms fail to generalize to higher resolutions is the shift in attention entropy relative to the training distribution. To investigate this, we analyze the Normalized Attention Entropy (scaled by the logarithm of the sequence length) as a function of the diffusion timestep, averaged across all layers and heads. We conducted this analysis using 20 random prompts from the Aesthetic-4K dataset, comparing the baseline FLUX model at its native resolution (1024 × 1024) against FLUX, YaRN, and DYYaRN at a resolution of 4096 × 4096.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

###### "The wicker basket overﬂowing with exotic fruits placed centrally on the wooden table inside the sunlit cottage kitchen."

[Figure 55]

[Figure 56]

As illustrated in Fig. 18, DY-YaRN best preserves the attention structure of the original distribution. Quantitatively, in terms of deviation from the baseline profile (measured by Mean Absolute Error), DY-YaRN achieves the lowest deviation (0.0455), outperforming both YaRN (0.0476) and the vanilla model (0.0529). This confirms that DYPE effectively mitigates the entropy shift typically observed during resolution extrapolation.

[Figure 57]

Qwen-Image-Edit-2509 Dy-YaRN

- Figure 11. Qualitative comparison of high-resolution multiconcept composition at 2656 × 2656 between DY-YaRN and the vanilla Qwen-Image-Edit-2509.

Table 10. Panoramic image generation at 4096 × 1365 resolution. Method CLIP-Score↑ ImageReward↑ Aesthetics-Score↑

YaRN 28.92 0.86 5.71 DY-YaRN 29.45 1.29 5.75

ated by DYPE.

Qualitative Comparison with General Baselines. Building on the comparisons presented in Sec. 4.1.2, we further provide qualitative results that highlight the differences between our approach and existing baselines (see Fig. 14). e

Qualitative results on the DrawBench benchmark. Building upon the comparisons presented in Sec. 4.1, we provide further qualitative results comparing our approach to existing baselines.

Additional Qualitative Results on the Aesthetic-4K Benchmark. Expanding upon the comparisons discussed

- in Sec. 4.1, we present additional qualitative examples that highlight the performance of our method relative to existing baselines.

Additional Zoom-in comparison. Expanding upon the comparisons discussed in Fig. 17, we present additional qualitative examples that illustrate the differences if DYYaRN with YaRN in fine details.

“A silhouetted pagoda stands against a large red moon, surrounded by dark mountains and trees, with some stylized birds ﬂying in the sky. Stars twinkle in the night backdrop.”

[Figure 58]

YaRN

[Figure 59]

###### Dy-YaRN

Figure 12. Qualitative comparison of panoramic generation at 4096 × 1365 resolution.

[Figure 60]

[Figure 61]

[Figure 62]

3072x2048

[Figure 63]

3072x3072 3072x3072

[Figure 64]

[Figure 65]

3072x3072

[Figure 66]

2048x3072

[Figure 67]

4096x4096

3072x2048

[Figure 68]

[Figure 69]

3072x2048

2048x3072

[Figure 70]

[Figure 71]

4096x4096

[Figure 72]

3072x3072

2048x3072

[Figure 73]

[Figure 74]

4096x4096

3072x3072

2048x3072

- Figure 13. Collage of multi- and high-resolution results generated by DYPE. Prompts were taken from the Aesthetic-4K test set. Zoom-in for details.

A vibrant, detailed landscape featuring a small town with red-brick buildings, green trees, and a rural backdrop. Prominently displayed in the background is a grand temple-like structure and a circular building, with a railroad track featuring a vintage streetcar running through the scene. Workers are seen in a ﬁeld, and livestock grazes nearby, under a blue sky with ﬂuﬀy clouds.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Majestic mountains bathed in pink and purple hues under a starry night sky, with a glowing tower overlooking a serene waterfall and tranquil blue pool, surrounded by dark trees and rocky terrain.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

DemoFusion Diffusion-4K HiFlow HiFlow + Dy-YaRN Dy-YaRN

- Figure 14. Qualitative results at 40962 resolution using two representative prompts from Aesthetic-4K. We compare DemoFusion, Diffusion-4K, HiFlow, DY-YaRN+HiFlow and DY-YaRN.

[Figure 85]

###### DyPE: Dynamic Position Extrapolation for Ultra High Resolution Diffusion

"A black colored dog."

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

"A bird scaring a scarecrow."

“Snow-covered landscape featuring a group of sheep in the foreground, with a quaint, snow-dusted church and rustic houses in the background surrounded by trees.”

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

| | |
|---|---|
| | |

[Figure 99]

[Figure 100]

“A realistic photo of a Pomeranian dressed up like a 1980s professional wrestler with neon green and neon orange face paint and bright green wrestling tights with bright orange boots.”

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

| | |
|---|---|

| | |
|---|---|
| | |

YaRN Dy-YaRN

"Rainbow coloured penguin."

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

- Figure 17. Zoom-in comparison at 40962 resolution showing DYYaRN vs. YaRN. Three magnified regions per image compare differences in fine details.

[Figure 109]

- Figure 18. Deviation of Normalized Attention Entropy from the baseline (1024 × 1024) profile across diffusion timesteps. Lower values indicate better preservation of the original attention structure.

NTK-aware Dy-NTK-aware YaRN Dy-YaRN

- Figure 15. Qualitative results for high-resolution text-to-image generation on the DrawBench benchmark.

NTK-aware Dy-NTK-aware YaRN Dy-YaRN

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

“A lone ﬁgure wearing a dark cloak and a horned hat stands on a rocky outcrop, gazing out over a vast, misty landscape of mountains and valleys, with autumn-hued foliage and dramatic, cloud-ﬁlled skies.”

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

“A female astronaut in a sleek, high-tech suit stands against a backdrop of a turbulent cosmic scene featuring asteroids and a distant, ﬁre-ridden planet, with spacecraft ﬂying in formation.”

“A man in a patterned vest and tie stands conﬁdently next to a woman wearing a sleek, cream-colored coat, both posing near an elegant train entrance.”

“A middle-aged man with long, gray hair and a short beard smiles gently, his warm, expressive eyes capturing attention against a dark background.”

- Figure 16. High-resolution text-to-image generation results on the Aesthetic-4K benchmark.

