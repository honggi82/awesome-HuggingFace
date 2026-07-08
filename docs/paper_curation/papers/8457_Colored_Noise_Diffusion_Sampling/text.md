# arXiv:2605.30332v1[cs.CV]28May2026

## Colored Noise Diffusion Sampling

Hadar Davidson Noam Issachar Sagie Benaim The Hebrew University of Jerusalem

### Abstract

Diffusion models achieve state-of-the-art image synthesis, with their generative trajectories fundamentally exhibiting a spectral bias, resolving low-frequency global structures early and high-frequency fine details later. Conventional stochastic differential equation (SDE) solvers fail to account for this dynamic, naively injecting uniform white noise throughout the entire process and misusing the finite energy budget. In this work, we establish a mathematical framework that reconsiders SDE inference as a targeted, frequency-decoupled energy transfer. Leveraging this framework, we introduce Colored Noise Sampling (CNS), a novel, training-free stochastic solver. Rather than injecting uniform white noise, CNS utilizes a dynamic, timestep- and frequency-dependent schedule that more efficiently allocates injected energy toward structurally unresolved frequency bands. By actively exploiting the model’s inherent spectral bias, CNS systematically steers the generated distribution toward the true data manifold. Extensive experiments demonstrate that CNS significantly outperforms standard ODE and SDE baselines as a strictly plug-and-play, inference-time sampler substitution across diverse architectures (SiT, JiT, FLUX). Compared to standard sampling on ImageNet-256, CNS achieves substantial unguided FID reductions, improving from 8.26 to 6.27 on SiT-XL/2, 32.39 to 26.69 on JiT-B/16, and 11.88 to 8.31 on JiT-H/16, while yielding consistent relative FID improvements with Classifier-Free Guidance. Project page is available at https://hadardavidson.github.io/CNS/.

### 1 Introduction

Diffusion models have established a new standard in photorealistic image synthesis, defining the stateof-the-art in high-fidelity generation [16, 58, 26]. Crucially, the sampling trajectory of these models exhibits a spectral bias [61, 20]. This inductive property dictates that diffusion models inherently resolve low-frequency global structures during early sampling steps, while filling in high-frequency fine details in later steps.

Current sampling algorithms [57, 58, 56] fail to account for this phenomenon. Standard stochastic methods based on Stochastic Differential Equations (SDEs) naively inject uniform white noise, completely disregarding how the frequency spectrum of the generated image dynamically evolves over time. To address this inefficiency at its root, we introduce a new class of stochastic solvers that actively leverages this spectral bias. By tailoring the injected noise to the specific denoising timestep, we improve generation fidelity without requiring any additional training.

Recognizing the importance of spectral bias, several recent works have attempted to exploit it. One line of research alters the training framework, interpolating from spectrally non-uniform or temporally evolving noise distributions [8, 54, 17]. Other approaches operate at inference time, introducing ad-hoc modifications such as frequency-decoupled operations [44, 67], internal activation reweighting [55], or step-size schedule adjustments [28]. While these methods yield measurable improvements, they remain fundamentally constrained by their underlying use of spectrally uniform solvers. This

Preprint.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

CNS(Ours)SDEODE

[Figure 7]

[Figure 8]

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

Figure 1: Colored Noise Sampling (CNS). Samples from SiT-XL/2 on ImageNet-256 (with CFG) for different sampling strategies. While standard SDEs inject uniform white noise, our Colored Noise Sampling (CNS) dynamically reallocates injected stochastic energy to unresolved frequency bands. This actively leverages the network’s spectral bias to systematically steer the output toward the true data manifold, outperforming standard ODE and SDE solvers.

naturally leads to our guiding question: How can we actively exploit the spectral bias of diffusion models to design a fundamentally new, general-purpose sampler that improves generation fidelity?

To answer this, we first establish a mathematical framework to control the generated distribution via frequency-aware noise injection. Geometrically, sampling trajectories resemble non-orthogonal rotations toward the data manifold [61]. This implies that diffusion models do not arbitrarily discard initial noise; rather, a significant structural component of this signal is preserved and mapped into final image features [64, 36].

A key observation of our work is that this signal-preserving transfer also applies to the continuous noise injected by SDE solvers throughout the trajectory. Furthermore, this process is frequencydecoupled: injected noise in a specific frequency band maps directly to spatial features in that same band. By ensuring our frequency-aware adjustments remain strictly variance-preserving, requiring only that the total injected energy per step remains normalized, we demonstrate that the classic Langevin requirement for uniform white noise [40] can be safely relaxed without pushing intermediate states out of distribution.

Building on this framework, we construct a timestep- and frequency-dependent noise schedule. By analyzing the progression rates of different frequency bands during generation, we propose that the network’s ability to convert injected noise into coherent image features significantly depends on how structurally “resolved” that specific band is at a given timestep. This insight allows us to reconsider SDE sampling as a targeted energy injection process. Rather than uniformly distributing the finite injected noise budget, our approach utilizes a dynamic schedule based on the expected evolution of the trajectory, allocating energy to the frequency bands where it is most needed. This principled allocation steers the output toward the true data manifold, yielding strictly higher-fidelity generation.

To validate our approach, we conduct extensive experiments across diverse architectures and modalities, including latent-space generation (SiT [35]), pixel-space generation (JiT [29]), and state-ofthe-art text-to-image synthesis (FLUX [25, 26]). Evaluated primarily via the Fréchet Inception Distance (FID) [14], empirical results demonstrate that our method significantly outperforms standard ODE and SDE baselines. On ImageNet-256 [50], we achieve substantial FID reductions under both unguided and Classifier-Free Guidance (CFG) [15] settings, while maintaining robust stability across varying discretization steps. We visually highlight the superiority of our approach over standard baselines in Fig. 1. Furthermore, our sampler proves effective when integrated into text-to-image pipelines like FLUX, improving automatic human-preference scores.

To summarize, our main contributions are:

- • We establish a mathematical framework that reframes SDE noise injection as a targeted energy transfer, and demonstrate that the standard Langevin requirement for spectrally uniform white noise can be safely relaxed to resolve spectral gaps.
- • We introduce Colored Noise Sampling (CNS), a novel, training-free stochastic solver that actively leverages spectral bias by dynamically allocating injected noise energy toward structurally unresolved frequency bands.
- • We validate CNS as a robust, general-purpose sampler across diverse architectures (SiT, JiT, FLUX). On ImageNet-256, CNS achieves substantial unguided FID reductions (e.g., 8.26 to 6.27 for SiT-XL/2, 32.39 to 26.69 for JiT-B/16) and relative CFG improvements ranging from ∼8% to ∼50% over standard ODE and SDE baselines.

### 2 Related Work

Samplers for Diffusion Models. Sampling in diffusion models is a highly researched domain primarily focused on numerically mitigating discretization errors. Prominent advancements include higher-order solvers [62, 65, 34] that maintain fidelity at low step counts, dynamic solver alternation [32, 72], and state reparameterizations that smooth integration pathways [70, 69, 5]. While these methods successfully reduce truncation errors and accelerate generation, they remain agnostic to the evolving spatial structure of the state. Our approach is fundamentally orthogonal: rather than strictly optimizing numerical precision, we optimize the allocation of stochastic energy by explicitly exploiting the model’s spectral bias.

Leveraging Spectral Bias in Diffusion Models. A prominent line of research exploits the spectral bias of diffusion models during training by altering noise distributions. These methods rely on empirical heuristics to modify initial [54] or temporally-evolving noise distributions [17], or introduce formally grounded frequency-dependent processes like EqualSNR [8]. However, fundamentally altering the learning objective demands costly model retraining. In sharp contrast, our approach overcomes this barrier by introducing a purely plug-and-play sampler that harnesses spectral bias exclusively at inference time. To circumvent retraining costs, a separate line of work leverages spectral bias via inference-only modifications. These methods introduce ad-hoc adjustments to the generation pipeline, such as applying frequency-decoupled operations to the predicted state [44, 67], dynamically reweighting internal network activations [55], adjusting step-size schedules [28], or coupling spectral bias with positional encodings [20]. While effective, these techniques treat the underlying stochastic solver as a static black box. Our work targets this unexplored component: rather than modifying the network or its outputs post-hoc, we directly embed the spectral bias into the core sampling mechanism itself.

#### 3 Method We now introduce our approach. Sec. 3.1 outlines standard diffusion background. Sec. 3.2 and

- 3.3 formalize the specific generative phenomena, inference-time spectral bias and noise energy preservation, that we leverage to build our framework. Building on these principles, Sec. 3.4 analyzes the spectral gap induced by standard SDEs. Finally, Sec. 3.5 details how CNS dynamically colors injected noise to actively steer the generated spectrum toward the true data manifold.

##### 3.1 Background: Diffusion Models and Sampling Dynamics

Diffusion Models [16, 58] and Flow Matching [31, 33] can be unified under the continuous-time framework of Stochastic Interpolants [3]. Given a target data distribution x0 ∼ pdata and a tractable noise prior ϵ ∼ N(0,I), these models construct a probability path via the time-dependent state:

xt = αtx0 + σtϵ, t ∈ [0,1] (1)

Boundary conditions are established such that x0 strictly represents the clean data (α0 = 1,σ0 = 0) and x1 approximates the pure noise prior (α1 ≈ 0,σ1 ≈ 1). Whether utilizing trigonometric schedules like Variance-Preserving (VP) diffusion or linear paths like Flow Matching (αt = 1 − t,σt = t), the objective remains learning to reverse this continuous-time probability flow.

To learn this reverse flow, these models approximate the conditional interpolant velocity:

vt = α˙tx0 + σ˙tϵ (2)

Because the intermediate state xt is a simple affine combination of data and noise, predicting the velocity vθ is algebraically equivalent to predicting the clean data xθ ≈ x0, the noise ϵθ ≈ ϵ, or the marginal score sθ ≈ ∇xt

log pt(xt). Omitting explicit (xt,t) dependencies for brevity, these parameterizations are deterministically linked by the following relations:

ϵθ σt

(3)

vθ = α˙txθ + σ˙tϵθ, xt = αtxθ + σtϵθ, sθ = −

During inference, novel samples are generated from the prior by substituting these learned predictions into reverse-time differential equations, which are then integrated using either deterministic or stochastic solvers.

Sampling Dynamics. Deterministic sampling formulates the trajectory as a Probability Flow ODE (PF-ODE), directly integrating the predicted velocity:

- 0
- 1

,ti)∆ti (4)

dxt ≈ x1 +

vθ(xt

dxt = vθ(xt,t)dt, x0 = x1 +

i

i

While computationally efficient and approximately invertible, this strict determinism lacks an inherent corrective mechanism. Consequently, discrete numerical approximations and network errors inevitably accumulate, causing intermediate states to gradually drift off the true data manifold and degrade final image fidelity [56].

Stochastic solvers address this drift by simulating the generative process as a reverse-time SDE. Introducing a time-dependent diffusion coefficient g(t) > 0 and a reverse-time Wiener process w¯, the dynamics expand to:

- 1

- 2

g(t)2sθ(xt,t) dt + g(t)dw¯ (5)

dxt = vθ(xt,t) −

This process fundamentally alters the trajectory by continuously counterbalancing white Gaussian noise injection with a restorative gradient step along the predicted score. The injected noise explores the local latent neighborhood, while the score-based denoising actively pulls the state back toward high-density regions. By natively correcting accumulated discretization errors at every step, SDE solvers keep the trajectory firmly anchored to the true data distribution, yielding superior visual quality [57, 58].

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

PowerSpectralDensityS(f)

LowenergyHighenergy

Frequency f

Low freq. High freq.

Power Spectral Density and Noise Colors. The frequency composition of the injected noise ϵ is characterized by its Power Spectral Density (PSD). Letting ϵˆ(f) = F(ϵ) denote its Fourier transform, the PSD S(f) evaluates the expected energy at frequency f:

Figure 2: PSD of different colored noises. The spectra transition smoothly from highfrequency dominant blue noise, through uniform white noise (center black line), to lowfrequency dominant red noise.

S(f) = E[|ϵˆ(f)|2] (6)

The shape of S(f) defines the noise “color” [41] as illustrated in Fig. 2. Standard Gaussian noise ϵ ∼ N(0,I) possesses a constant S(f), injecting equal energy across all frequencies (white noise). Conversely, non-uniform spectra produce colored noise, such as high-frequency dominant blue noise. Due to Fourier’s orthogonality, Parseval’s theorem ensures that integrating the PSD yields the total spatial energy: E[∥ϵ∥2] = S(f)df. Consequently, standard SDEs fundamentally operate by blindly injecting a fixed, frequency-agnostic white noise energy budget at every generative step.

##### 3.2 Spectral Bias of Diffusion Models

Spectral bias is a well-documented inductive property that extends beyond training optimization to fundamentally govern the inference dynamics of diffusion models [45, 46, 61]. Rather than resolving the image uniformly, generation follows a staggered frequency evolution.

To formalize this band-wise progression, we evaluate the model’s clean data prediction at each intermediate timestep t. Under a linear schedule, this prediction is given by:

xpredt = xθ(xt,t) = xt − tvθ(xt,t) (7)

Let X0(f) = F(x0)(f) and Xpred(f,t) = F(xpredt )(f) denote the spectral components at frequency band f for the final generated latent and the intermediate prediction, respectively. Following [20], we measure the resolved energy of this intermediate prediction relative to the final outcome to define the bounded progress index γ(f,t) ∈ [0,1] for every frequency band f:

1.00

|X0(f) − Xpred(f,t)|2 |X0(f)|2

[Figure 19]

1.00

[Figure 20]

(8)

γ(f,t) = 1 −

0.75

0.75

This index isolates exactly how much of a specific frequency band’s final structure has been resolved by the network at any given timestep t (see Alg. 2 and App. C.1 for further details). Visualizing this γ-matrix (Fig. 3) directly exposes these generation dynamics: low-frequency structures resolve early in the generation process. In contrast, high-frequency details evolve at a gradual rate, only fully materializing at the very end of the sampling trajectory. Ultimately, this provides a precise temporal map dictating exactly when specific frequency bands are actively being “built” by the network.

Frequencyf

γ(f,t)

0.50

0.50

0.25

0.25

0.00

0.00

1.0 0.8 0.6 0.4 0.2 0.0

Diffusion timestep t

Figure 3: Temporal progression of frequency bands during sampling.

##### 3.3 Structural Preservation and Energy Transfer in Diffusion Models

The mapping from the prior N(0,I) to the data distribution pdata is not an arbitrary coupling between the two spaces. Empirical evidence demonstrates that the inference process preserves significant information from the initial noise realization [66, 64, 59], naturally following minimal-distance trajectories [19].

To explain this geometrically, Wang and Vastola [61] demonstrate that sampling trajectories are surprisingly low-dimensional. Rather than taking an unconstrained walk across the latent space, these trajectories effectively resemble 2D rotations of θ ≈ 1 radian from the initial noise state toward the target data manifold. In high-dimensional spaces, where independent random vectors are nearly orthogonal, this rotational angle yields a remarkably high expected cosine similarity (cos(θ) ≫ 0). This mathematically confirms that the diffusion process does not generate novel structures from scratch, but rather preserves a substantial portion of the initial structural signal—a phenomenon we empirically visualize across spatial frequencies in Fig. 5a.

This rotational perspective has profound implications for the sampling dynamics. Since rotations preserve the L2 norm (i.e., the vector’s energy), inference acts as a signal transfer mechanism that retains a significant portion of the initial noise’s energy. The model deterministically maps this preserved noise onto the structured spatial features of the final image, a property implicitly leveraged by recent works optimizing initial noise selection [71, 2, 36]. This non-destructive property forms the foundational premise of our approach: it implies that by strategically controlling the energy injected into the process via noise, we directly control the structural features of the final image.

##### 3.4 The Generated Image Distribution Spectrum

As established in recent works [8, 1, 68], a distinct discrepancy exists between the PSD of generated images and the true data manifold—known as the spectral gap. Let Sdata(f) and S0(f) denote the PSDs of the real and generated distributions, respectively. As illustrated in Fig. 4, neither deterministic (S0ODE) nor stochastic (S0SDE) sampling perfectly recovers the ground truth Sdata(f). Crucially, resolving this gap requires more than simple post-hoc spectral matching; to achieve true distributional matching, the restored energy must align with the coherent spatial structures of the target manifold.

Fortunately, as established in Sec. 3.3, diffusion inference fundamentally operates as a partial energypreserving signal transfer. Beyond preserving the initial noise realization, we found that the diffusion process also maps the stochastic increments injected by SDE solvers directly into corresponding

0.1

- 102

- 103

- 104

- 105

- 106

- 107

low f mid f high f

0.0

Avg.PowerSpectrum

logRatiotoGT10

0.1

0.2

0.3

0.4

ImageNet GT

ImageNet GT

ODE: FID 14.40

ODE MAE=0.155 SDE MAE=0.065 CNS MAE=0.046

SDE: FID 8.73 CNS: FID 6.27

0.5

0.0 0.2 0.4 0.6 0.8 1.0

0.0 0.2 0.4 0.6 0.8 1.0

Frequency f

Frequency f

- Figure 4: The Spectral Gap Across Sampling Methods. (Left) PSDs of the generated distributions versus the PSD of the ground truth ImageNet. Standard ODE sampling over-generates low-frequency structures and under-generates high-frequency details, while standard SDE sampling exhibits an

energy deficit across the entire spectrum. (Right) The signed log10 error relative to the ground truth (black dashed line). By dynamically reallocating the injected noise budget, CNS better aligns the generated spectrum with the true data manifold, mitigating the spectral gap and achieving the lowest log-space Mean Absolute Error (MAE) across frequencies.

spatial frequencies of the final generated structure. We formalize this non-destructive, frequencycoupled mapping by isolating spatial frequency bands via a Fourier-space band-pass projection operator, Pb[·]. This allows us to quantify the structural alignment between the accumulated injected noise, ϵcumul = Ti=0−1 g(ti)dw¯t

, and the final generated image x0. By calculating their expected cosine similarity in Fig. 5b, we observe a significant positive correlation:

i

###### E cos Pb[ϵcumul],Pb[x0] ≫ 0 (9)

This strong alignment reveals a powerful theoretical pathway: strategically shaping the spectral profile of the injected noise provides a direct mechanism to steer the PSD of the generated distribution toward the target data manifold.

The Impact of Stochasticity. To restructure these dynamics, we quantify spectral deviation using the signed log error: ε(f) = log10(S0(f)/Sdata(f)). As shown in Fig. 4, comparing deterministic and stochastic solvers reveals that continuous noise injection fundamentally alters the final energy distribution. We suggest that this spectral divergence arises from inherent imperfections in the learned score function. During standard Langevin dynamics, the injected noise is not perfectly counterbalanced by the denoising drift. Consequently, score approximation errors cause unintended energy accumulations or deficits over the trajectory (App. B.1).

Crucially, the total stochastic energy injected over the generative trajectory is strictly bounded and mathematically independent of the time discretization (App. A.1). Because we cannot simply scale up the global noise injection to offset deficits without violating the underlying SDE (App. A.2), the stochastic noise acts as a strictly fixed injected energy budget:

###### E = g2(t)dt < ∞ (10)

Standard SDEs distribute this budget naively: uniform white noise allocates energy equally across the entire frequency spectrum (S(f) = 1). By transitioning to targeted colored noise, we treat this as a zero-sum game: we dynamically decrease energy allocation for structurally resolved frequency bands, freeing up the budget to inject energy into lagging frequencies. This principled reallocation steers the generated output toward the true data manifold without pushing the intermediate latents out-of-distribution (App. A.3).

1.0

1.0

cos(,)CosineSimilarityx0

cos(,)CosineSimilarityxcumul0

0.8

0.8

0.6

0.6

0.4

0.4

ODE (Total: 0.79) SDE (Total: 0.29) CNS (Total: 0.28)

0.2

0.2

SDE CNS

0.0

0.0

0.2 0.4 0.6 0.8 1.0

0.2 0.4 0.6 0.8 1.0

Frequency f

Frequency f

- Figure 5: Noise Signal Preservation and Transfer. (Left) Initial Noise Persistence. Cosine similarity between initial noise and the final generated image. ODEs strongly preserve structural information across the spectrum; stochastic methods (SDE, CNS) still retain a significant, though reduced, amount of this initial signal. (Right) Cumulative Injection Transfer. Cosine similarity between total injected

noise (ϵcumul) and the final generated image. This shows injected noise structure actively shapes the final features rather than serving as a temporary perturbation. Notably, CNS selectively routes this signal into higher frequency bands.

##### 3.5 Colored Noise Sampling (CNS)

CNS actively mitigates the spectral gap by repurposing the SDE’s stochastic energy leak to steer the generated profile. As derived in App. B.1, the effective energy a generated sample absorbs from noise injection is highly state-dependent. Specifically, the band-wise energy absorption rate depends strictly on the correlation between the current spectral state and the local score error. Because this absorption efficiency varies, uniform white-noise injection is highly suboptimal—it allocates the finite energy budget on frequency modes that are already sufficiently resolved. An optimal strategy must therefore dynamically adapt the noise spectrum to the timestep t and frequency band f.

To formalize this active reallocation, we introduce a frequency-dependent scaling weight βf(t) to the standard SDE noise increment. This colored-noise modification scales the stochastic Itô energy term

for a given frequency f from 12g2(t) to 12g2(t)βf2(t) (App. B.2). To maintain the overall stability of the generative process, we enforce a strict global variance-conservation constraint, ensuring the

average injected energy across all dimensions remains constant: D1 Df=1 βf2(t) = 1. In App. B.2, we demonstrate that a frequency band’s capacity to absorb injected stochastic noise into a permanent structure is strictly governed by its progression ratio, tracked by the γ(f,t)-matrix (Sec. 3.2). As a band approaches a fully resolved state (γ(f,t) → 1), the score error correlation decays. The network treats excess injected variance primarily as transient energy to be dissipated, severely diminishing the rate of permanent energy conversion.

To maximize the utility of the finite energy budget, CNS dynamically routes energy away from these resolved bands and actively channels it into lagging frequencies with a high structural deficit. We formulate the CNS allocation schedule such that the variance multiplier is strictly proportional to this structural deficit (further details in App. B.2.3). To satisfy the global energy constraint, we normalize this profile by its Root Mean Square (RMS) across frequencies:

Algorithm 1 CNS Sampling

# gamma: [T, F] completion matrix # freq_bins: (H,W) radial bin map x = torch.randn(x_shape)

for i, t in enumerate(torch. linspace(0, 1, T)):

- w = torch.randn_like(x) scale = torch.sqrt(1-gamma[i])

# colored PSD W = torch.fft2(w)*scale[

freq_bins] w_c = torch.real(ifft2(W)) w_c = w_c / torch.std(w_c) v = model(x, t) s = (t*v-x)/(1-t) # score D = diffusion(x, t)

- x += (v+D*s)*dt +torch.sqrt(2*D)

1 − γ(f,t)

(11)

β(f,t) =

1 D f′ (1 − γ(f′,t))

This dynamic coloring systematically increases the efficiency of the injected energy, yielding a generated distribution spectrally closer to the true data manifold (Fig. 4). The exact integration of this schedule into standard SDE solvers is detailed in Alg. 1.

*w_c*torch.sqrt(dt)

- Table 1: Evaluation of Unguided Image Generation. ImageNet-256 evaluation metrics without Classifier-Free Guidance across different sampling methods.

Model Sampler FID ↓ sFID ↓ IS ↑ Prec. ↑ Rec. ↑

SiT-XL/2

ODE 14.39 10.54 99.32 0.59 0.67 SDE 8.26 6.32 131.65 0.68 0.67 CNS (Ours) 6.27 4.73 147.33 0.71 0.65

JiT-H/16

ODE 12.41 9.35 43.61 0.63 0.62 SDE 11.88 8.64 44.30 0.64 0.63 CNS (Ours) 8.31 7.48 45.97 0.66 0.65

JiT-B/16

ODE 32.39 11.81 26.60 0.47 0.61 SDE 36.24 14.32 25.86 0.46 0.63 CNS (Ours) 26.69 9.67 27.95 0.51 0.63

- Table 2: Evaluation of SiT-XL/2 by Solver Order. ImageNet-256 evaluation metrics comparing different solvers and sampling methods. Solvers are categorized by their weak convergence order for SDEs (or deterministic order for ODEs).

Solver Method FID ↓ sFID ↓ IS ↑ Precision ↑ Recall ↑

ODE 14.39 10.54 99.32 0.59 0.67 SDE 8.26 6.32 131.65 0.68 0.67 CNS (Ours) 6.27 4.73 147.33 0.71 0.65

Euler-Maruyama [37] (1st-Order Weak)

ODE 9.35 6.38 126.06 0.67 0.68 SDE 8.00 5.49 132.72 0.69 0.67 CNS (Ours) 5.99 4.78 149.78 0.71 0.65

Heun [13, 22] (2nd-Order Weak)

SDE (SRK2) 8.14 5.69 132.53 0.69 0.67 SDE (SRK2S) 8.77 6.36 129.68 0.68 0.67 CNS (Ours, SRK2) 5.91 4.77 149.41 0.71 0.65 CNS (Ours, SRK2S) 5.97 4.73 148.55 0.70 0.66

Stochastic RK [48, 49] (2nd-Order Weak)

Deterministic RK [10] (5th-Order)

ODE (dopri5) 9.04 6.04 126.49 0.67 0.67

### 4 Experiments

We evaluate the performance and robustness of CNS across three key areas: (1) Class-Conditional Generation (Sec. 4.1), demonstrating superiority over ODE/SDE baselines across pixel and latent spaces, varying solver orders, and CFG settings; (2) Text-to-Image Generation (Sec. 4.2), integrating CNS into state-of-the-art flow-matching architectures to enhance visual fidelity while preserving strict semantic alignment; and (3) Ablations and Orthogonality, validating our schedule design choices and proving CNS provides orthogonal benefits to models trained with alternative noise distributions.

##### 4.1 Class-to-Image Generation

We evaluate CNS on class-conditional image generation (ImageNet-256 [50]), comparing it to standard ODE and SDE baselines [58, 56]. We assess both pixel-space x-prediction (JiT-H/16, JiT-B/16 [29]) and latent-space v-prediction (SiT-XL/2 [35]) models using their official pre-trained weights (Tab. 1). We measure generation quality, diversity, and manifold coverage using established metrics: Fréchet Inception Distance (FID) [14], spatial FID (sFID) [38] for high-frequency structural assessment, Inception Score (IS) [52], and Precision and Recall [24]. For fair comparison against reported baselines, we utilize originally published metric values wherever available. To maintain identical evaluation conditions across reproduced methods, we fix the initial seed across all methods. We match the sampling steps of the original works: 250 for SiT-XL/2 and 50 for JiT models. All quantitative metrics are computed using 50,000 generated samples against the standard 10,000 reference images. Qualitative visual comparisons of CNS against SDE and ODE baselines are provided in App. D.6.

##### Table 3: Evaluation of Guided Sampling (CFG) across SiT and JiT architectures on ImageNet-256. Model Sampler CFG Scale FID ↓ sFID ↓ IS ↑ Prec. ↑ Rec. ↑

ODE 1.5 2.15 4.60 258.09 0.81 0.60 SDE 1.5 2.06 4.49 277.50 0.83 0.59

SiT-XL/2

- CNS (Ours) 1.5 1.98 4.46 257.68 0.81 0.60

JiT-H/16

ODE 2.2 3.92 7.72 62.86 0.76 0.59 SDE 2.2 2.08 7.59 65.88 0.79 0.61

- CNS (Ours) 2.2 2.03 6.98 65.89 0.79 0.62

JiT-B/16

ODE 3.0 5.83 8.71 61.47 0.81 0.43 SDE 3.0 4.54 11.30 63.02 0.82 0.45

- CNS (Ours) 3.0 4.39 9.48 63.22 0.83 0.45 CNS (Ours) 2.6 4.19 9.01 60.27 0.81 0.47

###### Table 4: FID-30K (↓) on 642 Datasets. Comparison of baseline models and BNDM variants. Prior Methods BNDM Framework

###### Dataset IHDM DDPM DDIM IADB ODE SDE CNS (Ours)

AFHQ Cat 11.02 9.75 9.82 9.19 7.95 18.80 7.49 LSUN Church 17.76 13.07 16.46 13.12 10.16 66.71 8.70

Furthermore, Fig. 6 evaluates our sampler across varying time discretizations. Once a sufficient number of steps for proper stochastic simulation is reached, CNS consistently outperforms standard sampling methods. Further details are provided in App. D.3.

Solver Order. While deterministic ODEs rely on standard Taylor expansions, SDE solvers require Itô-Taylor expansions and are categorized by strong (pathwise) and weak (distributional) convergence orders. Because generative metrics evaluate distributional alignment, weak convergence is our primary focus. As shown in Tab. 2, we evaluate CNS across solvers with varying weak orders (1st-order Euler-Maruyama; 2nd-order Heun, SRK2, SRK2S). CNS outperforms all tested solvers on SiT-XL/2. Mathematical overviews of these solvers are in App. D.2.

Classifier-Free Guidance (CFG). CFG [15] significantly improves sample fidelity by extrapolating the conditional prediction away from the unconditional baseline. In Tab. 3, we demonstrate that CNS consistently outperforms standard ODE and SDE samplers under CFG across SiT-XL/2, JiT-H/16, and JiT-B/16 (using Euler, 250 steps for SiT, 50 for JiT). When optimal CFG scales diverge between methods, we report the best-performing scale for both the baseline and CNS. Qualitative comparisons of CFG-enhanced generation are in Fig. 1.

ODE (best FID=11.09)

100

SDE (best FID=7.83)

Ours (CNS) (best FID=5.97)

80

60

###### FID

40

20

10.2 8.7 7.8 7.2 6.3 5.97

0

10 20 35 50 100 250 1000

Orthogonality to Alternative Noise Training. Recent methods [54, 8, 17] alter the training framework to explicitly exploit spectral bias. To confirm CNS is not rendered redundant by such modifications, we integrate our sampler into Blue Noise for Diffusion Models (BNDM) [17], which trains a U-Net [47] with a time-evolving white-to-blue noise distribution. As shown in Tab. 4, applying CNS to BNDM’s pre-trained Iterative α-(de)Blending (IADB) [11] models still significantly improves generation quality. Evaluated across two 642 datasets (30,000 samples), this confirms CNS provides orthogonal inference-time benefits even to models inherently designed to leverage spectral bias. Further details are in App. D.5.

Sampling Steps

Figure 6: FID-50K vs sampling steps for different samplers.

##### 4.2 Text-to-Image Generation

Moving beyond class-conditional synthesis, we demonstrate the broad applicability of our method by seamlessly integrating it into complex downstream generation tasks. Specifically, we apply CNS as a plug-and-play sampler substitution within state-of-the-art Text-to-Image (T2I) flow-matching architectures: FLUX.1-dev [25] and FLUX.2-klein [26].

We evaluate this integration by comparing CNS against both standard white-noise SDEs and the deterministic ODEs that serve as the default solvers for these models. We evaluate generation quality and text-alignment across two comprehensive prompt benchmarks. DrawBench [51] probes specific T2I failure modes like complex text rendering (Tab. 5), while GenEval [9] tests precise compositional attributes such as object counts and spatial positioning (Tab. 9). Performance is measured using ImageReward [63] for human preference, CLIPScore [12] for semantic consistency, and Aesthetic Score [53] for visual appeal. Crucially, these evaluations confirm that the dynamic stochastic modifications introduced by CNS enhance overall visual fidelity without degrading the underlying model’s text comprehension or corrupting complex compositional instructions.

Table 5: Quantitative evaluation on DrawBench. Evaluated at 50 steps, CFG scales w = 3.5, 4.0 for FLUX1, FLUX2, respectively.

Sampler ImageReward ↑ CLIPScore ↑ Aesthetic ↑

FLUX.1-dev

ODE 0.965 0.681 5.787 SDE 0.990 0.689 5.804 CNS (Ours) 1.012 0.693 5.812

FLUX2.klein

ODE 0.984 0.735 5.233 SDE 0.924 0.733 5.291 CNS (Ours) 1.005 0.735 5.295

##### 4.3 Ablation Study

We conduct an ablation study on SiT-XL/2 (Euler, 250 steps), summarized in Tab. 6. While several alternative configurations outperform the standard whitenoise SDE, our derived CNS formulation consistently achieves the highest overall fidelity. Specifically, we demonstrate that scaling the injected variance without adhering to our normalization constraint significantly degrades quality, empirically validating the necessity of a strictly finite energy budget. Furthermore, perturbing the CNS schedule, via partial white-noise corruption, static colored spectra, or temporal shuffling, consistently yields inferior results, confirming the optimality of our dynamic, state-dependent allocation. Finally, we establish that replacing our schedule with Multifractional Brownian Motion (mBm) [42], a process generating time-varying colored noise via a shifting Hurst parameter H(t), also falls short of CNS performance. Further mathematical details for all ablations are provided in App. D.4.

Table 6: Key Ablation Studies (FID-10K). Full ablation results are in Appendix D.4.

Scenario FID ↓ sFID ↓ IS ↑ CNS (Ours) 9.61 18.17 143.20 White-noise SDE 11.82 19.15 107.75 Deterministic ODE 17.05 23.57 98.43

50% White Noise 10.64 19.08 136.36 Shuffled Schedule 10.46 19.03 138.02 Constant Spectrum 10.53 19.14 137.63

- Scale 0.90 Energy 16.17 42.03 111.29
- Scale 1.05 Energy 20.46 29.73 96.17 mBm 11.88 19.62 130.22

### 5 Conclusion

In this work, we address a fundamental inefficiency in standard diffusion SDE solvers: the uniform injection of white noise, which ignores the model’s inherent spectral bias and squanders the finite generative energy budget. By reconceptualizing SDE inference as a targeted energy transfer, we introduce Colored Noise Sampling (CNS), a novel stochastic sampler. CNS actively exploits spectral bias by dynamically reallocating injected noise toward structurally unresolved frequency bands. As a strictly plug-and-play sampler substitution at inference time, CNS systematically steers the generated distribution toward the true data manifold. Extensive experiments validate that CNS dramatically outperforms standard baselines across diverse architectures, yielding massive FID reductions and enhanced visual fidelity without requiring any model retraining.

Limitations and Future Work The primary limitation of CNS is its reliance on an SDE framework, rendering it incompatible with deterministic ODE solvers. Because stochastic sampling intrinsically requires a high step budget to prevent discretization error accumulation, standard ODEs remain preferable for ultra-fast inference. Future work will explore extending frequency-dependent energy routing into deterministic paradigms for low-step sampling, and applying CNS to video generation to leverage the temporal frequency dimension.

### References

- [1] K. Adamkiewicz, B. Moser, S. Frolov, T. C. Nauen, F. Raue, and A. Dengel. When pretty isn’t useful: Investigating why modern text-to-image models fail as reliable training data generators. arXiv preprint arXiv:2602.19946, 2026.
- [2] D. Ahn, J. Kang, S. Lee, J. Min, M. Kim, W. Jang, H. Cho, S. Paul, S. Kim, E. Cha, et al. A noise is worth diffusion guidance. arXiv preprint arXiv:2412.03895, 2024.
- [3] M. S. Albergo and E. Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.
- [4] B. D. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.
- [5] T. Chen, H. Zheng, D. Berthelot, J. Gu, J. Susskind, and S. Zhai. Tada: Improved diffusion sampling with training-free augmented dynamics. arXiv preprint arXiv:2506.21757, 2025.
- [6] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky, and J. C. Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.
- [7] B. Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.
- [8] F. Falck, T. Pandeva, K. Zahirnia, R. Lawrence, R. Turner, E. Meeds, J. Zazo, and S. Karmalkar. A fourier space perspective on diffusion models. arXiv preprint arXiv:2505.11278, 2025.
- [9] D. Ghosh, H. Hajishirzi, and L. Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023.
- [10] E. Hairer, S. Nørsett, and G. Wanner. Solving Ordinary Differential Equations I: Nonstiff Problems. Springer Series in Computational Mathematics. Springer Berlin Heidelberg, 2008. ISBN 9783540566700. URL https://books.google.co.il/books?id=F93u7VcSRyYC.
- [11] E. Heitz, L. Belcour, and T. Chambon. Iterative α-(de) blending: A minimalist deterministic diffusion model. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–8, 2023.
- [12] J. Hessel, A. Holtzman, M. Forbes, R. Le Bras, and Y. Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.
- [13] K. Heun et al. Neue methoden zur approximativen integration der differentialgleichungen einer unabhängigen veränderlichen. Z. Math. Phys, 45(23-38):7, 1900.
- [14] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.
- [15] J. Ho and T. Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [16] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [17] X. Huang, C. Salaun, C. Vasconcelos, C. Theobalt, C. Oztireli, and G. Singh. Blue noise for diffusion models. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024.
- [18] A. Hyvärinen, J. Hurri, and P. Hoyer. Natural Image Statistics: A Probabilistic Approach to Early Computational Vision. Computational Imaging and Vision. Springer London, 2009. ISBN

9781848824911. URL https://books.google.co.il/books?id=pq_Fr1eYr7cC.

- [19] N. Issachar, M. Salama, R. Fattal, and S. Benaim. Designing a conditional prior distribution for flow-based generative models. arXiv preprint arXiv:2502.09611, 2025.

- [20] N. Issachar, G. Yariv, S. Benaim, Y. Adi, D. Lischinski, and R. Fattal. Dype: Dynamic position extrapolation for ultra high resolution diffusion. arXiv preprint arXiv:2510.20766, 2025.
- [21] M. Kamien and N. Schwartz. Dynamic Optimization, Second Edition: The Calculus of Variations and Optimal Control in Economics and Management. Dover Books on Mathematics. Dover Publications, 2013. ISBN 9780486310282. URL https://books.google.co.il/books? id=liLCAgAAQBAJ.
- [22] T. Karras, M. Aittala, T. Aila, and S. Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.
- [23] P. Kloeden and E. Platen. Numerical Solution of Stochastic Differential Equations. Stochastic Modelling and Applied Probability. Springer Berlin Heidelberg, 2011. ISBN 9783540540625. URL https://books.google.co.il/books?id=BCvtssom1CMC.
- [24] T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.
- [25] B. F. Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [26] B. F. Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [27] A. B. L. Larsen, S. K. Sønderby, H. Larochelle, and O. Winther. Autoencoding beyond pixels using a learned similarity metric. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1558–1566, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr.press/v48/larsen16.html.
- [28] H. Lee, H. Lee, S. Gye, and J. Kim. Beta sampling is all you need: Efficient image generation strategy for diffusion models using stepwise spectral analysis. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 4215–4224. IEEE, 2025.
- [29] T. Li and K. He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.
- [30] D. Liberzon. Calculus of Variations and Optimal Control Theory: A Concise Introduction. Princeton University Press, 2011. ISBN 9781400842643. URL https://books.google.co. il/books?id=xQHEjXy8rlUC.
- [31] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [32] E. Liu, X. Ning, H. Yang, and Y. Wang. A unified sampling framework for solver searching of diffusion probabilistic models. In The Twelfth International Conference on Learning Representations, 2024.
- [33] X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
- [34] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems, 35:5775–5787, 2022.
- [35] N. Ma, M. Goldstein, M. S. Albergo, N. M. Boffi, E. Vanden-Eijnden, and S. Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.
- [36] J. Mao, X. Wang, and K. Aizawa. Guided image synthesis via initial image editing in diffusion model. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5321–5329, 2023.
- [37] G. Maruyama. Continuous markov processes and stochastic equations. Rendiconti del Circolo Matematico di Palermo, 4(1):48–90, 1955.

- [38] C. Nash, J. Menick, S. Dieleman, and P. W. Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.
- [39] M. Ning, M. Li, L. Zhang, L. Liu, M. B. Blaschko, A. A. Salah, and I. O. Ertugrul. Spectrum matching: a unified perspective for superior diffusability in latent diffusion. arXiv preprint arXiv:2603.14645, 2026.
- [40] B. Øksendal. Stochastic differential equations. In Stochastic differential equations: an introduction with applications, pages 38–50. Springer, 2003.
- [41] A. Oppenheim, R. Schafer, and J. Buck. Discrete-time Signal Processing. Prentice Hall International Editions Series. Prentice Hall, 1999. ISBN 9780130834430. URL https: //books.google.co.il/books?id=cR3CQgAACAAJ.
- [42] R.-F. Peltier and J. L. Véhel. Multifractional Brownian motion: definition and preliminary results. PhD thesis, INRIA, 1995.
- [43] M. Plancherel and M. Leffler. Contribution à l’étude de la représentation d’une fonction arbitraire par des intégrales définies. Rendiconti del Circolo Matematico di Palermo (1884-1940), 30(1): 289–335, 1910.
- [44] Y. Qian, Q. Cai, Y. Pan, Y. Li, T. Yao, Q. Sun, and T. Mei. Boosting diffusion models with moving average sampling in frequency domain. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8911–8920, 2024.
- [45] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.
- [46] B. Ronen, D. Jacobs, Y. Kasten, and S. Kritchman. The convergence rate of neural networks for learned functions of different frequencies. Advances in Neural Information Processing Systems, 32, 2019.
- [47] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015.
- [48] A. Rößler. Second order runge–kutta methods for itô stochastic differential equations. SIAM Journal on Numerical Analysis, 47(3):1713–1738, 2009.
- [49] A. Rößler. Runge–kutta methods for the strong approximation of solutions of stochastic differential equations. SIAM Journal on Numerical Analysis, 48(3):922–952, 2010.
- [50] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115(3):211–252, 2015.
- [51] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35: 36479–36494, 2022.
- [52] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.
- [53] C. Schuhmann. Improved aesthetic predictor. URL https://github. com/christophschuhmann/improved-aesthetic-predictor, 2022.
- [54] L. Scimeca, T. Jiralerspong, B. Earnshaw, J. Hartford, and Y. Bengio. Learning what matters: Steering diffusion via spectrally anisotropic forward noise. arXiv preprint arXiv:2510.09660, 2025.
- [55] C. Si, Z. Huang, Y. Jiang, and Z. Liu. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4733–4743, 2024.

- [56] J. Song, C. Meng, and S. Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.
- [57] Y. Song and S. Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.
- [58] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.
- [59] Ł. Staniszewski, Ł. Kuci´nski, and K. Deja. There and back again: On the relation between noise and image inversions in diffusion models. arXiv preprint arXiv:2410.23530, 2024.
- [60] A. van der Schaaf and J. van Hateren. Modelling the power spectra of natural images: Statistics and information. Vision Research, 36(17):2759–2770, 1996. ISSN 0042-6989. doi: https://doi. org/10.1016/0042-6989(96)00002-8. URL https://www.sciencedirect.com/science/ article/pii/0042698996000028.
- [61] B. Wang and J. J. Vastola. Diffusion models generate images like painters: an analytical theory of outline first, details later. arXiv preprint arXiv:2303.02490, 2023.
- [62] Y. Wu, Y. Chen, and Y. Wei. Stochastic runge-kutta methods: Provable acceleration of diffusion models. arXiv preprint arXiv:2410.04760, 2024.
- [63] J. Xu, X. Liu, Y. Wu, Y. Tong, Q. Li, M. Ding, J. Tang, and Y. Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.
- [64] K. Xu, L. Zhang, and J. Shi. Good seed makes a good crop: Discovering secret seeds in text-to-image diffusion models. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 3024–3034. IEEE, 2025.
- [65] S. Xue, M. Yi, W. Luo, S. Zhang, J. Sun, Z. Li, and Z.-M. Ma. Sa-solver: Stochastic adams solver for fast sampling of diffusion models. Advances in Neural Information Processing Systems, 36:77632–77674, 2023.
- [66] S. Yan, M. Li, B. Xinliang, J. Yang, Y. Zhang, G. Xiong, Y. Lan, T. Zhang, W. Zhai, and Z.-J. Zha. Beyond randomness: Understand the order of the noise in diffusion. arXiv preprint arXiv:2511.07756, 2025.
- [67] M. Yu, L. Sun, J. Zeng, X. Chu, and K. Zhan. Elucidating the snr-t bias of diffusion probabilistic models. arXiv preprint arXiv:2604.16044, 2026.
- [68] D. Zhang, T. Zhang, S. Ge, and S. Süsstrunk. Enhancing frequency forgery clues for diffusiongenerated image detection. arXiv preprint arXiv:2511.00429, 2025.
- [69] Q. Zhang and Y. Chen. Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902, 2022.
- [70] K. Zheng, C. Lu, J. Chen, and J. Zhu. Dpm-solver-v3: Improved diffusion ode solver with empirical model statistics. Advances in Neural Information Processing Systems, 36:55502– 55542, 2023.
- [71] Z. Zhou, S. Shao, L. Bai, S. Zhang, Z. Xu, B. Han, and Z. Xie. Golden noise for diffusion models: A learning framework. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17688–17697, 2025.
- [72] D. Zou, E. Liu, X. Ning, H. Yang, and Y. Wang. Usf++: A unified sampling framework for solver searching of diffusion probabilistic models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

### A Theoretical Constraints on Stochastic Energy Injection

In the main text, we establish that the stochastic noise injected during the generative process acts as a strictly fixed injected energy budget. This appendix provides the formal derivations underpinning this core premise. First, in App. A.1, we demonstrate that the total injected stochastic energy is mathematically finite and strictly invariant to the time discretization of the solver. Building upon this, in App. A.2, we show that this finite energy budget cannot be globally scaled to offset spectral deficits without violating the underlying SDE and breaking the deterministic convergence to the target data manifold. Together, these theoretical constraints necessitate our zero-sum, frequency-band redistribution approach.

##### A.1 Invariance of Total Injected Energy to Timestep Discretization

In this section, we demonstrate that for a given continuous diffusion schedule g(t) over a generative interval [t0,t1], the total stochastic energy injected by the noise process is finite and strictly invariant to the number of discretization steps used to integrate the reverse SDE. This formalizes our conceptualization of the SDE’s stochasticity as a strictly fixed, global injected energy budget.

Consider the continuous reverse-time SDE [4, 58]:

dxt = µ(xt,t)dt + g(t)dw¯ (12) where the deterministic drift is defined as µ(xt,t) = v(xt,t) − 12g2(t)∇xt

log pt(xt) , dw¯ is the standard reverse-time Wiener process, and g : [t0,t1] → R≥0 is a continuous and bounded diffusion coefficient.

Let our discrete sampler be evaluated over N uniform integration steps using standard EulerMaruyama numerical integration [23]:

xk+1 = xk + µ(xk,tk)∆t + g(tk)∆w¯k, ∆w¯k ∼ N(0,∆tI) (13) Isolating a single spatial dimension for clarity, we define the injected stochastic increment at step k as:

ηk := g(tk)∆w¯k (14) and the total expected injected variance per dimension over the entire trajectory as:

N−1

E[ηk2] (15)

EN :=

k=0

Proof of Convergence. Because the diffusion coefficient g(tk) is deterministic at step k, the expected variance of the increment is:

E[ηk2] = E (g(tk)∆w¯k)2 = g2(tk)E[∆w¯k2] = g2(tk)∆t (16) Therefore, the total expected variance is:

N−1

g2(tk)∆t (17)

EN =

k=0

This formulation is exactly the left Riemann sum of g2(t) over the interval [t0,t1]. By the continuity of g(t), we can take the continuous-time limit:

t1

g2(t)dt =: E (18)

EN =

lim

N→∞

t0

Because standard diffusion schedules g(t) are strictly bounded on a finite interval [58], the integral E evaluates to a finite constant.

Finite-N Discretization Error. Let Err(N) := |EN − E| denote the variance discrepancy for a discrete solver. If g2(·) ∈ C1([t0,t1]), the bounded error of the left Riemann sum guarantees:

(t1 − t0)2 2N

d dt

g2(t) (19)

Err(N) ≤

max

t∈[t0,t1]

This demonstrates that the convergence is O(1/N). Consequently, up to a minor, strictly bounded numerical integration error, any chosen number of steps N draws from the exact same finite pool of total injected energy.

##### A.2 Theoretical Demand for Injected Noise Variance Conservation

In this section, we demonstrate that a Langevin dynamics formulation in which the injected global noise variance is misaligned with the true score function—even under the assumption of an exact oracle score—yields a sampling process that fundamentally fails to converge to the target data distribution, pdata.

The Modified Reverse SDE and its Equivalent Flow. In continuous-time diffusion frameworks, the reverse generative process is governed by the stochastic differential equation (SDE) [58]:

- 1

- 2

g2(t)∇xt

log pt(xt) dt + g(t)dw¯ (20)

dxt = v(xt,t) −

where v(xt,t) is the deterministic drift, g(t) is the diffusion coefficient, and dw¯ ∼ N(0,dtI) is the reverse-time Wiener process.

Let us introduce a constant global variance multiplier β to the stochastic increment. The modified SDE becomes:

- 1

- 2

g2(t)∇xt

log pt(xt) dt + βg(t)dw¯ (21)

dxt = v(xt,t) −

To analyze the exact marginal distributions generated by this modified SDE, denoted ρt(xt), we rely on the continuous-time equivalence between SDEs and ODEs. For any reverse-time SDE of the

form dxt = µ(xt,t)dt + σ(t)dw¯, there exists a deterministic Probability Flow ODE that shares the identical marginal density ρt(xt) at all times [4]:

- 1

- 2

σ2(t)∇xt

log ρt(xt) dt (22)

dxt = µ(xt,t) +

Substituting our modified drift and diffusion terms into this relation yields the equivalent PF-ODE for our perturbed system:

- 1

- 2

- 1

- 2

g2(t)∇xt

β2g2(t)∇xt

log ρt(xt) dt (23)

dxt = v(xt,t) −

log pt(xt) +

If we enforce the requirement that this modified process successfully reaches the target data manifold exactly along the true marginal trajectory, we must have ρt(xt) = pt(xt) for all t. Under this condition, the equivalent flow simplifies to:

- 1

- 2

g2(t) 1 − β2 ∇xt

log pt(xt) dt (24) However, the true target PF-ODE that corresponds to the unperturbed forward process is known to be:

dxt = v(xt,t) −

dxt = v(xt,t)dt (25) This identity is recovered directly from the SDE-to-PF-ODE relation applied to the original reverse SDE (β = 1): the explicit score subtraction −12g2∇xt

log pt is exactly cancelled by the implicit Fokker–Planck correction +12g2∇xt

log pt contributed by the diffusion term, leaving only v(xt,t). Equating the drift of our modified process with this target requires:

- 1

- 2g2(t) 1 − β2 ∇xt

log pt(xt) = 0 (26) Since g(t) > 0 and the score ∇xt

log pt is generically nonzero along the generative trajectory, this constraint is satisfied strictly if and only if:

β2 = 1 (27) This mathematically confirms that any global rescaling of the injected stochastic noise disrupts the deterministic flow toward pdata.

The Rigid Coupling of Transport and Geometry. To rigorously justify why a mismatch (ρt ̸= pt) causes the process to fail, we examine the continuity equation governing the mass transport of

the system. Let v∗(xt,t) denote the exact PF-ODE vector field. The temporal evolution of the mismatched density ρt driven by this static vector field is:

∂ρt ∂t

= −∇xt · v∗(xt,t)ρt(xt)

(28)

= −v∗(xt,t) · ∇xt

ρt(xt) − ρt(xt)∇xt · v∗(xt,t)

This expansion exposes a fundamental structural incompatibility:

##### Advection (v∗ · ∇xt

ρt): The translation of mass acts upon the spatial gradient of the mismatched distribution ρt.

Compression/Expansion (ρt∇xt ·v∗): The local expansion and compression of probability mass is controlled by the score field, which is specifically tailored to the geometry of the true data distribution

pt.

Because ρt possesses a different spatial gradient than pt, the static local flow routes mass incorrectly. An area in the latent space might receive an influx of probability mass dictated by v∗ expecting the

geometry of pt, but instead encounters the density of ρt, inevitably creating artificial bottlenecks or overshoots. The score field does not act as a global attractor; rather, it serves as a rigidly coupled transport mechanism that remains valid only when the state distribution adheres strictly to its corresponding marginal trajectory.

Practical Implications of β2 ̸= 1. While β2 = 1 is required for perfect distributional matching, the practical failure modes diverge depending on the direction of the scaling error:

- • Over-injection (β2 > 1): The excessive stochastic exploration outpaces the restorative capacity of the score-based drift. The state rapidly diffuses off the true manifold, resulting in severe degradation of sample fidelity.
- • Under-injection (β2 < 1): The restorative gradient steps overpower the diminished stochastic exploration. While this theoretically violates perfect marginal matching, the samples may not immediately drift off-manifold. Instead, the generative distribution aggressively contracts around local high-density modes. This preserves the visual coherence of individual samples but results in mode concentration and a measurable loss of generative diversity.

In App. A.3 and App. B.2 we demonstrate that in diffusion models, anchoring the density ρt to the true probability path requires relaxing the strict per-dimension variance constraint. Instead, the model enforces an average variance across all dimensions, satisfying the following condition:

D

1 D

βf2(t) = 1 (29)

f=1

##### A.3 Model Robustness to Out-of-Distribution Spectral States

- A fundamental assumption in our Colored Noise Sampling (CNS) framework is that actively modu-

lating the per-frequency Signal-to-Noise Ratio (SNR) via the scaling vector βf(t) does not push the intermediate generative states so far Out-Of-Distribution (OOD) that the network fails. In this section, we demonstrate that this robustness is an inherent property of diffusion models. Specifically, because of the network’s inductive spectral bias, standard inference trajectories intrinsically operate on highly OOD spectral states, yet the model reliably resolves these states into the true data distribution.

The Theoretical Training Paradigm. During standard continuous-time training (e.g., under a linear Flow Matching schedule, for simplicity), the model is optimized over exact marginals pt(xt). Given a clean data sample x0 ∼ pdata and a noise sample ϵ ∼ N(0,I), the intermediate state flows from pure noise at t = 1 to clean data at t = 0 via:

xt = (1 − t)x0 + tϵ (30) Applying the Fourier transform yields the spectral composition of the training marginals:

xt(f) = (1 − t) x0(f) + t ϵ(f) (31) Later established in App. B.1.2, the expected clean data spectrum exhibits a power-law distribution | x0(f)| ≈ C/fω

amp, while the noise prior is spectrally flat | ϵ(f)| ≈ 1.

If the generative process were to perfectly trace this theoretical training path, the proportion of the resolved image signal at any frequency band f would evolve strictly linearly across time. By isolating the data term and normalizing, the expected target progression ratio γtarget(f,t) is defined as:

E[| xt(f) − t ϵ(f)|] E[| x0(f)|]

= 1 − t (32)

γtarget(f,t) =

Under this idealized paradigm, every frequency band accumulates image information uniformly, maintaining the exact SNR balance dictated by the training marginals.

Inference-Time Spectral Divergence. However, during standard inference, the model does not generate the image uniformly. As quantified by the γ-matrix (Sec. 3.2), the neural network exhibits a profound spectral bias.

Both when initiated from perfect pure noise and integrated with high precision, and when given some intermediate state of the interpolated distribution, the empirical progression of the frequency bands, γactual(f,t), drastically deviates from the linear target γtarget(f,t) = 1 − t. Low-frequency bands converge much faster than the training paradigm dictates, resolving their structural energy early in the generation process (t ≫ 0), while high-frequency bands remain noise-dominated until the very end of the trajectory.

Implications for Colored Noise Sampling. This empirical divergence reveals a critical operational reality: during standard inference, the intermediate latents xt are already severely out-of-distribution with respect to the training marginal’s PSD and SNR [8].

At an intermediate timestep t, the theoretical training marginal pt(x) expects the low frequencies to be only partially built (∼ 1 − t). In reality, the generated state xgent possesses heavily saturated low frequencies (γactual ≈ 1). Despite this massive spectral mismatch—where the local SNR of specific bands completely violates the theoretical training expectation—the network’s learned score function sθ(xt,t) does not collapse. It continues to stably process this OOD spectrum, utilizing the combined velocity/score dynamics to reliably push the trajectory toward the true image distribution rather than reverting to noise.

Algorithm 2 γ(f,t) Matrix Computation

# ODE traj x: [T, B, C, H, W], # x[0]=noise, x[T-1]=image gamma_sum = torch.zeros(T, F) for x in ODE_batches:

t = torch.linspace(0, 1, T) v = (x[1:] -x[:-1]) / dt # [T-1,B,C,H,W] xp = x[:-1]+(1-t[:-1])*v # clean pred [T-

1] xp = torch.cat([xp, x[-1:]]) # append

final [T] X = torch.fft2(xp) Xf = torch.fft2(x[-1]) # final spectrum g = 1 -torch.abs(X-Xf)**2 / torch.abs(Xf)

**2 g = torch.clamp(g, 0, 1).mean(dim=C) # ch

. avg -> [T,B,H,W]

This inherent spectral robustness directly validates the core mechanical assumption of Colored Noise Sampling. Because the model natively handles and corrects intermediate states with perturbed frequency-wise SNR distributions, intentionally shaping the injected stochastic energy via our variance scaling operator βf(t) operates safely within the model’s established envelope of robustness. By ensuring that the global variance budget is strictly conserved (D1 βf2 = 1), CNS leverages the network’s inherent capacity to convert specifically routed frequency-band energy into coherent image structures without inducing catastrophic OOD failure.

gamma_sum += bin_radially(g).mean(dim=B) gamma = gamma_sum / N

### B Theoretical Analysis of Spectral Dynamics and Colored Noise

In this appendix, we present a unified theoretical analysis of the spectral dynamics governing generative diffusion models. First, in App. B.1, we investigate the mechanistic origins of the observed spectral gap between ODE and SDE samplers, demonstrating how pathwise energy trajectories diverge due to imperfect score approximation. Building upon this foundation, in App. B.2, we formalize how Colored Noise SDEs could strategically reshape this continuous energy evolution on a strictly band-wise basis. We show that while ideal score conditions enforce a strict zero-sum energy redistribution, the time-dependent decay of state-error correlation allows CNS to fundamentally circumvent this constraint. This enables CNS to permanently and constructively alter the generated spectrum by dynamically routing variance to frequencies where structural conversion efficiency is maximized.

##### B.1 Theoretical Origins of the Generated Distributions Spectral Difference

To identify the mechanism behind the observed generated distribution PSDs differences (see Fig. 4), we theoretically analyze the temporal evolution of frequency-band energy during sampling for both ODE and SDE trajectories. The goal is to characterize where (which frequency bands and timesteps)

and how (rate and direction of energy transfer) the two samplers diverge. This comparison provides a mechanistic explanation for why their final generated spectra differ and, in turn, clarifies the source of their quality gap.

Although the Probability Flow ODE and reverse-time SDE are constructed to share identical marginals pt(x) at each time t, their pathwise energy dynamics differ substantially due to the network’s imperfect approximation of the score. Let the state energy at time t be defined as:

- 1

- 2||xt||22 (33)

Et :=

The factor of 12 is a standard normalization that ensures ∇x(21∥x∥22) = x, which cleanly aligns the energy drift expressions.

##### B.1.1 Pathwise Energy Dynamics in Continuous-Time Sampling

Let v(xt,t) denote the deterministic drift of the PF-ODE. The ODE trajectory is dxt = v(xt,t)dt. Applying standard differentiation, the expected energy progression is governed entirely by the alignment between the state and the velocity:

d dt

EtODE = x⊤t v(xt,t) (34)

We recall that the reverse-time SDE drift relates to the ODE drift via the score function. The SDE trajectory is:

- 1

- 2

g2(t)sθ(xt,t) dt + g(t)dw¯ (35)

dxt = v(xt,t) −

To evaluate the energy differential of this stochastic process, we apply Itô’s lemma. Because the process is integrated backwards in time from t = T to t = 0, we define the backward time variable

τ = T − t, such that dτ = −dt > 0. The forward-in-τ SDE is dxτ = − v − 12g2sθ dτ + g dwτ. Applying Itô’s lemma for f(x) = 12∥x∥22 yields:

- 1

- 2

1 2

D 2

Tr g2(t)I∇2xf(xτ) dτ = x⊤τ −v +

g2dτ + x⊤τ g dw¯τ (36)

dEτ = x⊤τ dxτ +

g2sθ dτ +

where D is the data dimension. Reverting to the forward-time derivative dtd = −dτd and taking the expectation to zero out the Wiener noise, the expected energy drift of the SDE is:

d dt

- 1

- 2

D 2

E[EtSDE] = E[x⊤t v(xt,t)] −

g2(t)E[x⊤t sθ(xt,t)] −

g2(t) (37)

The Ideal Score and Heat Cancellation. To understand this dynamic, assume an ideal oracle score sθ = s∗ := ∇xt

log pt(xt). By integration by parts, the expected alignment of the data with its true score is strictly E[x⊤t s∗] = −D. Substituting this into the SDE drift:

d dt

- 1

- 2

D 2

d dt

E[EtSDE] = E[x⊤t v(xt,t)] −

g2(t) = E[x⊤t v(xt,t)] =

EtODE (38) This mathematically formalizes the ideal generative balance: the stochastic heat explicitly injected by the Itô noise term (−D2 g2) is perfectly and exactly canceled by the restorative radial contraction of the true score (D2 g2).

g2(t)(−D) −

The Imperfect Score. In practice, the learned score contains approximation errors: sθ(xt,t) = s∗(xt,t) + ϵ(xt,t). Substituting this imperfect score breaks the perfect heat cancellation:

d dt

- 1

- 2

d dt

g2(t)E[x⊤t ϵ(xt,t)] (39)

E[EtSDE] =

EtODE −

Integrating this drift difference from t = T (where ETSDE = ETODE) down to t = 0, we isolate the exact energy gap between the generated distributions:

0

T

- 1

- 2

- 1

- 2

g2(t)E[x⊤t ϵ(xt,t)]dt =

g2(t)E[x⊤t ϵ(xt,t)]dt (40)

E[E0SDE] − E0ODE =

−

T

0

Thus, the macroscopic divergence between SDE and ODE generation is controlled strictly by the state-error correlation term E[x⊤t ϵ(xt,t)]. If nonzero, the “heat vs. contraction” balance is broken, and pathwise energy drifts away from the ideal trajectory.

##### B.1.2 State-Error Correlation and the Power Spectral Density Gap

By Parseval–Plancherel identity [43], this global correlation can be decomposed into independent frequency contributions:

x⊤t ϵ(xt,t) =

Re(ˆxt(f)∗ ϵˆt(f)) (41)

f

We formulate this explicitly for a single frequency f. Defining the correlation functional as Γf(t) := E[Re(ˆxt(f)∗ ϵˆt(f))], the band-wise energy gap at t = 0 obeys:

T

g2(t)Γf(t)dt (42)

∆f := E ||xˆSDE0 (f)||2 − E ||xˆODE0 (f)||2 =

0

Because g2(t) ≥ 0, the sign of ∆f is determined exclusively by the accumulated sign of Γf(t). To determine whether the network error ϵ induces positive or negative state-error correlation, we must account for neural network learning dynamics. Networks trained via Mean Squared Error (MSE) exhibit regression to the mean; when confronted with uncertainty, they output a conservative, smoothed estimate [27]. Consequently, sθ systematically underestimates the magnitude of the true score s∗. We model this approximation error as being anti-aligned with the true score: ϵt ≈ −αfs∗t

- where 0 < αf < 1.

To determine the direction of the true score s∗, we invoke Tweedie’s formula [7, 6], which states that the score is proportional to the difference between the expected clean data and the current noisy state: sˆ∗(f,t) ∝ (1 − t)E[ˆx0(f) | xt] − xˆt(f). Let Nf represent the energy of the initial noise prior, and Rf represent the target energy of the real data distribution. To ground this physically, natural images exhibit a well-documented 1/fω power-law spectrum [60, 18], a structural property that fundamentally persists even within the compressed latent spaces of modern autoencoders [39]. Thus, we can approximate the real-data spectral target as a normalized Rf ≈ C/fω

pow (with ωpow ≈ 2), while the standard Gaussian prior maintains a flat white-noise spectrum Nf = 1. Comparing these frequency-dependent magnitudes defines three distinct generative regimes:

- 1. The Attenuation Regime (Rf < Nf): At frequencies where the target data energy is lower than the initial noise (typically high frequencies), the required evolution is attenuation. Because the expected clean signal magnitude is smaller than the current noisy state, the vector difference points inward. Thus, the true score acts to destroy noise: sˆ∗ ∝ −c1xˆt. Because the network underestimates this inward pull, the resulting error points outward (ϵˆt ∝ +αfc1xˆt). The error is positively aligned with the state, meaning their inner product is positive:

Γf(t) > 0 =⇒ ∆f > 0 (43) Conclusion: The SDE over-allocates energy relative to the ODE. The continuous noise injection is not fully dissipated because the learned score is too weak to adequately attenuate the state.

- 2. The Amplification Regime (Rf > Nf): At frequencies where the target structural magnitude is larger than the initial noise (typically low frequencies), the required evolution is amplification. The expected clean data vector has a significantly larger magnitude (∥E[ˆx0(f)]∥ > ∥xˆt(f)∥). Thus, the vector difference points outward: sˆ∗ ∝ c2xˆt. The underestimation error therefore points inward (ϵˆt ∝ −αfc2xˆt). The error is anti-aligned with the state, yielding a negative inner product:

Γf(t) < 0 =⇒ ∆f < 0 (44) Conclusion: The SDE under-allocates energy relative to the ODE. The SDE’s constant stochastic disruption, coupled with a weakened score that cannot fully drive the necessary amplification, causes the trajectory to fall short of the ideal energy level.

- 3. The Crossover Point (Rf = Nf): The regime transition occurs exactly at the frequency where the inherent energy of the initial noise matches the target energy of the real data. Here, the expected magnitude of the clean data equals the magnitude of the current noisy state. The score provides a purely tangential (phase-rotational) pull, exerting zero radial pull. Consequently, the inner product of the state and the score (and therefore the error) is zero:

Γf(t) = 0 =⇒ ∆f = 0 (45) Conclusion: At the exact crossover frequency, the radial approximation error vanishes, and the ODE and SDE energy trajectories perfectly match.

##### B.2 Spectral Impact and Energy Dynamics of Colored Noise SDEs

As proven in App. A.1, the total energy injected by the stochastic process is finite and strictly bounded by the diffusion schedule g(t). We now seek to mathematically formalize how a Colored Noise SDEs reshapes the continuous evolution of this energy on a strictly band-wise basis.

##### B.2.1 Energy Dynamics under an Ideal Score Function

Uniform Energy Allocation in Standard SDEs. In a standard white-noise SDE, the injected stochastic power is distributed uniformly across all frequency bands at every timestep. Equivalently, each integration step injects a flat Power Spectral Density (PSD), ensuring all spatial frequencies receive an equal share of the finite injected energy budget.

The Colored Noise Variance Operator. We introduce a colored-noise process d wt by scaling the standard noise differently across frequency bands. In the Fourier domain, let:

B(t) = diag(β1(t),β2(t),...,βD(t)) (46)

where each βf(t) > 0 is a real, time-dependent scaling weight for frequency index f, and D is the data dimensionality. The modified noise increment in the frequency domain is defined as:

d wt = B(t)d w¯t, d wt(f) = βf(t)d w¯t(f) (47)

Mapping back to the spatial domain via the inverse Fourier transform F−1 yields the linear operator form:

d wt = F−1B(t)F dw¯t (48)

The Global Variance-Conservation Constraint. To preserve the global stability of the generative process (shown necessity in App. A.2), we require the total injected energy of CNS to precisely match the standard white-noise SDE. For our modified process, the expected variance is:

E ||g(t)d wt||2 = g2(t)

D

βf2(t)E ||d w¯t(f)||2 = g2(t)

f=1

D

βf2(t)dt (49)

f=1

Because the standard SDE injects a total variance of g2(t)·D ·dt, matching this global energy budget imposes a strict variance-conservation constraint (i.e., the weights must have a Root Mean Square of 1):

D

1 D

βf2(t) = 1 (50)

f=1

Band-Wise Energy Drift of Colored Noise SDE. Substituting this scaled noise process into the reverse-time SDE dynamics, the Fourier-domain state evolution for a specific frequency band f becomes:

d xt(f) = µ(xt,t)f dt + g(t)βf(t)d w¯t(f) (51)

Applying Itô’s lemma to the band energy Et(f) := 12|| xt(f)||2, as derived in App. B.1, the expected band-wise energy drift under the CNS takes the form:

d dt

- 1

- 2

E Et(f),CNS = E[Re( xt(f)∗ µ(xt,t)f)] −

g2(t)βf2(t) (52)

For comparison, under the standard white-noise SDE (βf(t) ≡ 1), the corresponding band-wise drift is:

1 2

d dt

E Et(f),SDE = E[Re( xt(f)∗ µ(xt,t)f)] −

g2(t) (53) Hence, the colored-noise modification dynamically alters only the injected Itô heat term from

- 1

- 2g2(t) to 12g2(t)βf2(t), enabling targeted frequency-dependent amplification (βf2 > 1) or attenuation (βf2 < 1) while safely preserving the global energy budget.

Interpretation and Required Assumptions. For theoretical tractability (while noting this is an approximation due to the network’s spectral bias), assume the ideal oracle score applies a per-band

restoring drift that perfectly cancels the standard Itô energy injection. See App. A.3 for further details. Under these assumptions, the perfect score cancels each band at the standard unit scale. Consequently, scaling the injected noise in band f by βf(t) creates an intentional mismatch: βf2 > 1 yields a net energy addition in that band, while βf2 < 1 yields a net energy dissipation. Assuming the model reliably maps this altered stochastic energy into coherent spatial structures (detailed in Sec. 3.3), higher injected variance reliably yields higher generated structural energy in that frequency band.

##### B.2.2 Zero-Sum Spectral Constraints under an Ideal Score

We isolate the explicit effect of noise shaping by subtracting the standard SDE energy drift from the CNS drift, yielding the instantaneous excess energy injected into band f:

- 1

- 2

d dt

E Et(f),SDE − E Et(f),CNS =

g2(t) βf2(t) − 1 (54)

Integrating this over the sampling trajectory t ∈ [T,0] formalizes the terminal excess-energy gap for frequency f:

T

- 1

- 2

g2(t)

βf2(t) − 1 dt (55)

Excessf :=

0

f

Conclusion (Zero-Sum Redistribution). If we scale a specific frequency band such that βf > 1, the generated energy in that band increases. However, due to the global conservation constraint

(D1 βf2 = 1), deviations from the unit baseline must sum strictly to zero:

D

βf2(t) − 1 = 0 (56)

f=1

Therefore, summing the excess energies over all bands confirms global conservation:

  dt = 0 (57)

 

D

D

T

1 2

(βf2(t) − 1)

g2(t)

Excessf =

0

f=1

f=1

Under an ideal score, any spectral adjustment is fundamentally a zero-sum game: specific bands may achieve higher energy under CNS, but exclusively at the expense of others.

##### B.2.3 Breaking the Zero-Sum Constraint via Score Approximation Error

The zero-sum paradigm relies on a critical assumption: that the conversion rate of raw injected noise into permanent generated structural energy is constant across all frequency bands. However,

- as derived in App. B.1, the actual amount of injected raw noise that successfully converts into

permanent structural divergence is strictly dictated by the state–error correlation term Γf(t). We now demonstrate that this correlation is highly non-stationary and decays rapidly as the structural content of the image resolves. To quantify structural formation, we use the normalized bounded metric representing how “built” a specific frequency band f is at any time t derived in Sec. 3.2 and visually shown in Fig. 3.

Proposed Mechanisms for State-Error Correlation Decay To theoretically explain the decay in state-error correlation (limγ

f(t)→1 Γf(t) = 0), we postulate two complementary mechanisms: (1) the leading-order radial component of the true score vanishes as a band resolves, and (2) a possible transition of the network’s error on unresolved high-frequency details toward phase-random scatter across consecutive late-phase steps.

We note that the framework presented here is intended as a plausibility argument rather than a rigorous derivation. In particular, we do not require the absolute magnitude of the score error ∥ϵ∥ to vanish; we merely suggest that the changing geometric and temporal properties of the terminal generative state could plausibly drive the accumulated correlation toward zero even while ∥ϵ∥ ≫ 0.

- 1. Tangential Dominance of the True Score. By Tweedie’s formula [7], the true score in the frequency domain is proportional to the displacement from the current state toward its conditional clean estimate:

sˆ∗(f,t) ∝ E[ˆx0(f) | xt] − xˆt(f) (58)

We suppress the schedule-dependent prefactor here because only the direction of sˆ∗ relative to xˆt enters the radial inner product we are about to compute. When a frequency band approaches a fully

“built” state (γf(t) → 1), the network has correctly resolved the band’s macroscopic content, so the conditional clean estimate is closely aligned with the current state in both magnitude and approximate direction. We characterize the residual mismatch as a small phase rotation:

E[ˆx0(f) | xt] ≈ ei δϕ

f(t) xˆt(f), δϕf(t) → 0 as γf(t) → 1. (59)

This is strictly stronger than merely ∥E[ˆx0(f) | xt]∥ ≈ ∥xˆt(f)∥: equal magnitudes alone leave room for an arbitrary chord-like displacement with a substantial radial component, whereas a small-angle phase rotation is tangential to leading order. Expanding:

sˆ∗(f,t) ∝ eiδϕ

− 1 x ˆt(f) = iδϕf xˆt(f) + O(δϕ2f). (60)

f

The leading-order displacement iδϕf xˆt is geometrically orthogonal to xˆt in the complex plane (multiplication by i rotates by 90◦), so the radial inner product of state and score is purely a quadratic remainder:

Re(ˆxt(f)∗ sˆ∗(f,t)) = O(δϕ2f). (61)

Carrying the early-phase underestimation model ϵˆt ≈ −αfsˆ∗ from App. B.1 forward, the same quadratic suppression propagates to the state-error correlation:

Re(ˆxt(f)∗ ϵˆt(f)) −−−−−−→γf(t)→1 0. (62)

##### 2. Transition to Phase-Random Error on Unresolved Details. During the early phases of band

formation (γf(t) ≪ 1), MSE training induces a temporally coherent radial bias—specifically, a systematic underestimation of the score’s amplitude along the state direction. However, this coherence

breaks once the band’s macroscopic magnitude is established (γf(t) → 1). At this terminal stage, the network’s residual task shifts to resolving fine sub-structural details, such as sharp edges and exact phase alignments. Because these high-frequency features are difficult to infer from a condensed latent representation, the network’s errors abandon any consistent directional bias, transitioning instead into erratic, phase-random fluctuations across consecutive late-phase integration steps.

To see how such a transition could matter, it is useful to write the band-wise inner product in polar form:

Re(ˆxt(f)∗ ϵˆt(f)) = |xˆt(f)||ϵˆt(f)| cos(θϵ(t) − θx(t)). (63) The contribution of the late steps to the accumulated energy gap is governed by the time average of cos(θϵ − θx) along their trajectory. In the early phase this relative phase is presumably concentrated near 0 or π for many consecutive steps (the radial directions associated with the underestimation model), so per-step contributions would add coherently with a fixed sign. If, in the terminal regime, the relative phase instead varied erratically across the final taken steps and behaved roughly as if uniform on [0,2π), the signed per-step contributions would tend to cancel under temporal accumulation, and the late-phase contribution to Γf could plausibly collapse toward zero — even while each individual step still satisfied ∥ϵˆt(f)∥ ≫ 0. We offer this as a speculative mechanism that, together with the geometric argument above, may help account for the proposed decay of Γf.

State-Dependent Efficiency and the Breakdown of Zero-Sum Redistribution. Together, these proposed mechanisms provide a theoretical framework explaining why the state-error correlation Γf(t) = E[Re(ˆxt(f)∗ϵˆt(f))] collapses once a frequency band is structurally resolved. Under this framework, when a band is fully built (Γf(t) ≈ 0), the SDE manages to properly balance stochastic noise injection and score-driven contraction. Here, the same Γf that governs the band-wise SDE-vsODE gap (Eq. 42) acts as a conversion gain. It quantifies the rate at which raw injected stochastic variance is translated into coherent radial growth, rather than dissipating as incoherent fluctuation. As Γf → 0, this conversion gain vanishes. Consequently, any surplus variance (βf2 > 1) enters the instantaneous variance budget but fails to materialize as coherent image content; instead, subsequent denoising steps treat it as ambient noise and remove it. As a result, the global conservation of injected variance ( (βf2 − 1) = 0) does not enforce a zero-sum conservation of generated energy, effectively dismantling the zero-sum redistribution constraint.

Ultimately, the true efficacy of colored noise—its energy absorbance efficiency—is dictated by the temporal intersection of the variance scaling βf(t) and the decaying structural correlation Γf(t).

Permanently altering the generated spectrum requires injecting variance while a band remains in structural deficit (γf(t) ≪ 1). This highlights a fundamental property of stochastic generation: macroscopic spectral divergence is not determined by the total volume of injected energy, but by its timing.

Derivation of the Proposed CNS Allocation Schedule. Having established that energy conversion efficiency decays as γf(t) → 1, we formulate the optimal time-dependent injection strategy. To maximize the utility of the finite variance budget, the system must avoid injecting excess energy into “built” bands where it will be wastefully dissipated. Instead, it must dynamically route variance into “unbuilt” lagging frequencies where the correlation Γf(t) remains high, maximizing the structural retention of the injected heat. While a naive maximization strategy might suggest a “bang-bang” controller [30, 21]—routing the entire variance budget exclusively to the single least-built frequency

- at any given moment—we hypothesize that energy conversion efficiency is fundamentally bounded by the local magnitude of the injected noise. Overwhelming a single frequency band with excessive stochastic variance surpasses the local restorative capacity of the score network, degrading the model’s ability to cohesively integrate or dissipate that noise. Consequently, the optimal schedule

must avoid greedy allocation. The variance scaling factor βf2(t) should instead be smoothly and directly proportional to the structural deficit of the frequency band, (1 − γf(t)). To satisfy the global energy conservation constraint (D1 βf2(t) = 1) at every timestep, we normalize this deficit across the frequency spectrum by its Root Mean Square (RMS):

1 − γf(t)

βf(t) =

D f′=1 (1 − γf′(t))

1 D

This schedule exhibits several highly desirable theoretical properties:

(64)

- • Initialization as White Noise: At t = T, all bands are entirely unbuilt (γf(T) = 0). The allocation evaluates to βf(T) = 1 for all frequencies. The generative process naturally begins as a standard uniform white-noise SDE.
- • Dynamic Routing: As generation progresses, bands accumulate structure at staggered rates.

The schedule autonomously drains the variance budget from bands approaching γf ≈ 1 and actively routes it into lagging frequencies.

- • Avoidance of Local Saturation: By scaling proportionally rather than utilizing a binary “bang-bang” injection, the schedule prevents any single frequency band from being overwhelmed by stochastic variance, ensuring the noise remains within the operable restorative capacity of the score network.
- • Optimal Retention: By shifting the mathematical weight exclusively to regions where the state–error correlation is strongest, the schedule mathematically circumvents the limitations of the zero-sum energy game, ensuring that every unit of injected variance is optimally converted into permanent macroscopic structure.

### C Methodological and Experimental Details

This appendix provides the comprehensive methodological and experimental details required to reproduce the Colored Noise Sampling (CNS) framework. While the main text establishes the theoretical premise of spectral energy control, this section translates these concepts into a robust and readily deployable inference algorithm. First, in App. C.1, we formally define the multi-dimensional Fourier projection operators standardly used to isolate and evaluate isotropic radial frequency bands. Next, in App. C.2, we detail our hardware configuration alongside practical schedule relaxationssuch as progression scaling and dynamic spectral tilting—that seamlessly integrate CNS into standard sampling pipelines, further enhance algorithmic robustness, and consistently improve synthesis results.

##### C.1 Isolation of Radial Spatial Frequency Bands

Let x ∈ RC×H×W denote a continuous, real-valued multi-channel spatial tensor, such as a generated noise sample or an image. To accurately evaluate the frequency-coupled dynamics of the diffusion

process, we must systematically isolate the signal components corresponding to specific spatial scales. This is achieved using the multi-dimensional discrete Fourier transform.

The 2D Discrete Fourier Transform and Shifted Grid. We begin by applying the 2D Discrete Fourier Transform (DFT), denoted as F, independently across each channel c:

H−1

W−1

H +vwW ) (65)

xc(h,w)e−i2π(

uh

Xc(u,v) = F[xc](u,v) =

w=0

h=0

To analyze frequencies based on their spatial scale rather than their directional orientation, we shift the 2D frequency indices to center the DC (zero-frequency) component at the origin. Let the shifted frequency coordinates be denoted as (fy,fx), defined over the discrete domain:

fy ∈ [−⌊H/2⌋,...,⌈H/2⌉ − 1], fx ∈ [−⌊W/2⌋,...,⌈W/2⌉ − 1] (66)

Isotropic Radial Frequencies. Because natural images and standard diffusion noise priors are generally isotropic (rotationally invariant in expectation), we collapse the 2D frequency grid into a 1D measure of spatial scale. We define the radial frequency ρ(fy,fx) as the Euclidean distance from the DC component:

ρ(fy,fx) = fy2 + fx2 (67) The maximum theoretical radial frequency, corresponding to the corners of the 2D Nyquist limit, is bounded by ρmax = (H/2)2 + (W/2)2.

Discrete Band Partitioning. To perform statistical analysis across the spectrum, we partition the continuous range [0,ρmax] into Nb discrete radial frequency bands. Each discrete coordinate (fy,fx) is mapped to an integer band index b ∈ {0,...,Nb − 1} via nearest-integer scaling:

ρ(fy,fx) ρmax

(Nb − 1) (68)

b(fy,fx) =

This mapping defines a set of mutually exclusive 2D frequency masks, Bb, where each set contains all coordinate pairs belonging to band b:

Bb = (fy,fx) ∈ Z2 : b(fy,fx) = b (69)

The Projection Operator and Hermitian Symmetry. Using these discrete sets, we define the band-pass projection operator Pb[·], which isolates the spatial signal residing exclusively in band b. This is achieved by taking the inverse 2D DFT of the masked spectrum:

Pb[x] = F−1 1(f

y,fx)∈Bb ⊙ F[x] (70)

- where 1 is the indicator function and ⊙ denotes element-wise multiplication.

A critical geometric property of the radial distance function ρ is its symmetry across the origin; if (fy,fx) ∈ Bb, then (−fy,−fx) ∈ Bb. Because the input spatial tensor x is real-valued, its Fourier transform inherently exhibits Hermitian conjugate symmetry (X(−fy,−fx) = X∗(fy,fx)). The isotropic symmetry of the mask Bb perfectly preserves this Hermitian property during the masking operation. Consequently, the inverse DFT guarantees that the projected tensor remains strictly real-valued:

Pb[x] ∈ RC×H×W (71)

This allows us to treat Pb[x] not as a complex Fourier series, but as a standard real vector in a lowerdimensional subspace Vb ⊂ RCHW. Because the projection operator returns real values, we can seamlessly compute standard spatial metrics—such as the L2 norm and cosine similarity—directly on the projected tensors.

##### C.2 Hardware Configuration and Empirical Relaxations

Hardware and Environment. All generative inference and evaluation experiments were conducted using PyTorch with Distributed Data Parallel (DDP) across a compute node equipped with four NVIDIA L40S (48GB VRAM) GPUs.

Empirical Schedule Relaxations. While App. B.2.3 derives the theoretically optimal β(f,t) variance allocation schedule, we empirically found that introducing minor algorithmic relaxations helps maintain optimal generative stability. In practice, we implement three parameterized adjustments to our inference framework:

- • Progression Scaling: To prevent premature variance routing, we soften the strictly computed structural progression matrix by introducing a constant scaling factor c > 1, such that the effective progression becomes γ˜(f,t) = γ(f,t)/c.
- • Dynamic Spectral Tilting: For specific experiments, we apply a temporally evolving exponential tilt to the injected colored noise spectrum. This smooths the strict band-wise allocation, providing a more stable transition across adjacent frequency bands as generation progresses.
- • Energy Equilibrium Tuning: Although App. A.2 derives a strict β2 = 1 constraint, empirical score networks systematically underestimate the restorative gradient (App. B.2.3), skewing the reverse-time SDE’s “heat vs. contraction” balance. We compensate with a micro-scaling of the total injected energy budget; the values used (0.98–0.999, Tab. 7) lie well inside the unit-energy stability basin characterized in Tab. 10.

Exact configuration values and hyperparameter selections across all evaluated architectures (SiT, JiT, FLUX) are comprehensively reported in Tab. 7.

- Table 7: Sampling hyperparameters for the baseline architectures evaluated in our experimental setup. Standard experiments.

SiT-XL/2 JiT-B/16 JiT-H/16 Parameter w/o CFG w/ CFG w/o CFG w/ CFG w/ CFG w/o CFG w/ CFG Architecture & Data Space Latent (VAE) Pixel Pixel Dataset ImageNet-256 ImageNet-256 ImageNet-256 Prediction v-pred x-pred x-pred Frequency Bands 32 32 32 Sampling Steps 250 50 50 CNS Sampling Settings Guidance Scale – 1.5 – 2.6 3.0 – 2.2 Solver Euler/Heun Euler Euler Euler Euler Euler Euler γ(t, f) Power 0.75 0.5 0.5 0.5 0.5 0.5 0.5 γ(f, t) Divider 1.73 25.0 5.0 10.0 14.0 7.5 25.0

Alpha Tilting Interpolation (0.15, −0.5) (−0.1, 0.03) - - - - Alpha Tilting Interpolation Type Exponential (0.75) Linear - - - - Noise Energy Scale 0.98 0.998 0.995 - - 0.98 0.999

### D Additional Results and Extended Evaluations

This section provides extended empirical evaluations that supplement the findings presented in the main text. We first present additional quantitative benchmarks across varying architectures and sampling settings (App. D.1). Next, we detail the numerical integration schemes utilized for our stochastic differential equations high-order solvers (App. D.2) and analyze the robustness of CNS across varying numbers of discretization steps (App. D.3). To rigorously validate our theoretical constraints and design choices, we then provide a comprehensive ablation study evaluating the effects of global energy scaling, spectral perturbation, and temporal schedule misalignment (App. D.4). Finally, we elaborate on experiments demonstrating the orthogonality and generalization of CNS to models trained with alternative noise distributions (App. D.5). Finally, we conclude with extended qualitative visual comparisons (without guidance) that highlight the superiority of CNS over standard baselines (App. D.6).

##### D.1 Extended Generative Benchmarks

To demonstrate consistency across model architectures and evaluation frameworks, we provide supplementary performance tables. Tab. 8 reports the unguided JiT-H/16 results utilizing 100

sampling steps. Under these conditions, CNS strictly dominates both the ODE and standard SDE baselines across all tracked metrics. Furthermore, Tab. 9 (referenced in the main text) details the extensive GenEval [9] benchmarking for the FLUX model, demonstrating CNS’s capacity to enhance complex text-to-image synthesis.

- Table 8: Evaluation of Unguided Image Generation Performing 100 Sampling Steps. ImageNet256 JiT-H/16 model evaluation metrics without Classifier-Free Guidance across different sampling methods.

Sampler FID ↓ sFID ↓ IS ↑ Prec. ↑ Rec. ↑

ODE 11.46 10.49 44.71 0.63 0.64 SDE 8.95 7.98 46.38 0.65 0.65 CNS (Ours) 8.57 7.16 46.72 0.66 0.65

- Table 9: GenEval quantitative compositional evaluation on FLUX.1-dev. All results are reported as accuracy scores ∈ [0,1] (higher is better).

GenEval Task Breakdown Single Obj. Two Obj. Counting Colors Color Attr. Position

Model & Sampler Overall

FLUX.1-dev

ODE 0.643 0.988 0.784 0.699 0.826 0.413 0.152 SDE 0.635 0.984 0.789 0.685 0.768 0.433 0.154 CNS (Ours) 0.647 0.988 0.794 0.714 0.803 0.420 0.164

##### D.2 Numerical Integration Schemes for SDEs

When discretizing the reverse-time Stochastic Differential Equation (SDE) of a generative model, dx = f(x,t)dt + g(x,t)dw [58], the choice of numerical solver dictates the approximation error. Unlike Ordinary Differential Equations (ODEs) which rely on standard Taylor series approximations, SDEs require Itô-Taylor expansions [23]. Consequently, SDE solvers are characterized by two distinct types of convergence orders:

- • Strong Order (Pathwise Accuracy): A solver has strong order p if the expected error of a

single trajectory scales as O(hp) as the step size h → 0, defined as E[∥xN −x(T)∥] ≤ Chp. This measures how accurately the solver tracks a specific noise realization.

- • Weak Order (Distributional Accuracy): A solver has weak order q if the error in the expec-

tation of smooth test functions ϕ scales as O(hq), defined as |E[ϕ(xN)] − E[ϕ(x(T))]| ≤ Chq.

In the context of generative modeling, our primary objective is to sample from the correct target distribution rather than accurately track a specific microscopic noise path. Because standard evaluation metrics like Fréchet Inception Distance (FID) [14] measure distributional distances, the weak order of the solver is the dominant quantity of interest. In our experiments (Sec. 4.1), we evaluate solvers of varying weak orders:

- 1. Euler-Maruyama [37]: The foundational 1st-order weak (and 1/2-order strong) SDE solver, requiring 1 function evaluation per step.
- 2. Stochastic Heun: A 2nd-order weak predictor-corrector method requiring 2 function evaluations per step [13, 22].
- 3. Stochastic Runge-Kutta (SRK): We utilize two high-order schemes derived by Rößler. SRK2 [48] achieves weak order 2 (and strong order 1 for additive noise) using 2 evaluations per step. SRK2S [49] achieves strong order 1 even for general diagonal multiplicative noise while maintaining weak order 2, requiring 3 evaluations per step.

##### D.3 Robustness to Discretization Steps

To evaluate the numerical robustness of our approach, we conducted an experiment analyzing performance across varying discretization steps. As the number of sampling steps increases, CNS exhibits a monotonic decrease in FID, consistently outperforming the standard SDE baseline. Notably, CNS matches the peak FID of the ODE sampler using less than half the number of steps required by the standard SDE.

Conversely, due to inherent limitations in the pre-trained model’s underlying framework, the deterministic ODE sampler fails to maintain a monotonic improvement at high step counts; thus, we omit its results beyond the standard 250 steps. While CNS significantly accelerates stochastic convergence, it still shares the fundamental limitation of SDE solvers, requiring a larger minimum number of discretizations than ODEs to properly integrate the underlying differential equations.

##### D.4 Comprehensive Ablation Study

To empirically validate the theoretical constraints and design choices of the Colored Noise Sampling (CNS) framework, we conduct an extensive ablation study. All ablations are performed on the unguided SiT-XL/2 architecture using 250 Euler sampling steps and evaluated on ImageNet-256 (FID-10K, sFID, and Inception Score). The full quantitative results are detailed in Tab. 10. Below, we formalize the methodology and theoretical implications for each ablation category.

##### D.4.1 Global Energy Scaling (Validating the Variance Constraint)

In App. A.2, we mathematically established the necessity of the global variance-

Table 10: Comprehensive Ablation Studies. Evaluated on FID-10K (↓), sFID (↓), and IS (↑). We isolate the specific effects of the CNS formulation against alternative baselines, partial noise corruption, temporal schedule permutations, global energy scaling, and multifractional Brownian Motion (mBm) schedules.

conservation constraint (D1 βf2 = 1). To empirically prove this, we uniformly scaled the total injected energy budget by factors ranging from 0.50 to 2.00. The results are stark: any deviation from a tight neighborhood of the unit-energy budget drastically degrades generation fidelity. Scaling the energy down (e.g., 0.90) starves the SDE of the necessary stochastic exploration required to correct numerical drift, pulling performance toward the deterministic ODE baseline. Conversely, scaling the energy up by even 5% (1.05) destabilizes the process. The excessive injected Itô heat easily overpowers the restorative gradient of the score network, rapidly pushing the latent states out-of-distribution and destroying semantic coherence.

Sampling Scenario FID-10K ↓ sFID ↓ IS ↑ CNS (Ours) 9.61 18.17 143.20 Baselines Deterministic ODE 17.05 23.57 98.43 Standard white-noise SDE 11.82 19.15 107.75 Temporal Partial Corruption

25% White Noise 10.47 18.95 137.91 50% White Noise 10.64 19.08 136.36 50% Random Unit-Energy Spectrum 11.28 19.21 134.81 100% Random Unit-Energy Spectrum 12.26 19.80 127.09

Temporal Schedule Permutations

Constant Spectrum (Temporal Mean) 10.53 19.14 137.63 Shuffled Schedule 10.46 19.03 138.02 Inverted Schedule (Reverse Time) 10.50 18.91 137.46

Global Energy Scaling (Variance Constraint Violation)

- Scale 0.50 (Half Energy) 106.82 278.69 9.83

- Scale 0.75 53.29 161.09 34.11

- Scale 0.90 16.17 42.03 111.29
- Scale 1.01 11.37 20.21 133.84

- Scale 1.05 20.46 29.73 96.17

- Scale 1.10 50.63 49.70 46.96 Scale 1.25 171.12 91.28 5.70

##### D.4.2 Spectral Perturbation and Partial Corruption

To validate the precision of our bandwise allocation, we randomly sampled a predefined ratio (25%, 50%, 100%) of timesteps to inject alternative noise distributions rather than the optimal CNS schedule. These injected distributions were either uniform white noise or random unit-energy spectra (where per-band weights are randomized but strictly maintain the global energy budget). As demonstrated in Tab. 10, any corruption of the optimal β(f,t) mapping monotonically degrades performance. This confirms that merely ensuring a unit-energy budget is insufficient; the energy must be explicitly routed to the specific frequency bands experiencing the highest structural deficit.

- Scale 1.50 256.56 134.67 2.12
- Scale 2.00 (Double Energy) 327.45 198.37 1.55

Multifractional Brownian Motion (mBm) Variants

White → Blue (H : 0.5 → 0.1) 13.46 25.16 122.49 White → Blue (H : 0.5 → 0.25) 11.88 19.62 130.22 Red → Blue (H : 0.9 → 0.1) 266.61 224.77 2.97

##### D.4.3 Temporal Schedule Permutations

As theorized in App. B.2.3, the state-error correlation decays as structures resolve, meaning the efficiency of energy injection is highly non-stationary. To prove that when energy is injected is just as critical as where, we dismantled the temporal alignment of the CNS schedule:

- • Constant Spectrum: We averaged the dynamic CNS matrix across all timesteps to create a single, static colored noise profile.
- • Shuffled Schedule: We randomly permuted the timesteps of the allocation matrix.
- • Inverted Schedule: We applied the schedule backwards relative to the generative time.

While all three variants maintain the exact same total frequency-wise energy injection as the optimal CNS over the full trajectory, they completely fail to match its FID. By injecting energy at incorrect timesteps—often into bands that are already structurally resolved—they squander the finite variance budget on transient rotational noise, validating the necessity of a dynamic, state-aware schedule.

##### D.4.4 Alternative Noise Formulations (mBm)

Finally, we explored an alternative mathematical formulation for generating non-stationary colored noise: multifractional Brownian Motion (mBm) [42]. Unlike standard Wiener processes, mBm introduces a time-varying Hurst parameter, H(t), allowing the stochastic increments to shift continuously between distinct noise colors (e.g., from H = 0.5 White noise to H < 0.5 Blue noise). Formally, this frequency-dependent shift can be understood through the harmonizable representation of mBm:

eitω − 1 |ω|H(t)+1/2

1 C(H(t)) R

dW˜ (ω) (72)

BH(t)(t) =

where ω denotes frequency, dW˜ (ω) is the complex Wiener measure, and C(H(t)) is a normalization constant. In this form, the exponent H(t) + 1/2 directly governs the spectral density, mathematically enforcing how the noise color evolves over time. While theoretically elegant, mBm dictates a rigid, parameterized shift across the entire spectrum. It lacks the highly granular, per-band structural awareness provided by the γ-matrix. Consequently, even meticulously tuned mBm schedules (e.g., White → Blue, H : 0.5 → 0.25) fall short of performance of the proposed CNS framework, although optimized mBm configurations can still offer marginal improvements over the standard SDE.

##### D.5 Generalization to Alternative Noise Training

To demonstrate that our framework is orthogonal to alternative noise training paradigms, we evaluated CNS on the official pre-trained IADB models provided by the authors of BNDM [17]. Because these BNDM experiments rely on a unique, temporally evolving noise distribution, we first derived and implemented a custom SDE sampler strictly tailored to their specific forward process. Despite the alternative training objective, we observed that these models still exhibit a pronounced spectral bias. We empirically tracked this bias to compute the corresponding structural progression γ(f,t)-matrix and integrated our CNS method on top of the custom SDE solver. By dynamically shaping the injected noise spectrum according to this matrix, CNS achieved significant generation improvements across two evaluated 64 × 64 datasets using the standard provided test batches.

##### D.6 Additional Visual Comparisons

In this section, we provide extended qualitative results demonstrating the efficacy of our proposed Colored Noise Sampling (CNS) framework compared to standard deterministic (ODE) and stochastic (SDE) baselines. Fig. 7 and Fig. 8 present generation samples across a diverse set of ImageNet classes. To ensure a fair and isolated evaluation of the sampling dynamics, all images within a given row are generated using the exact same noise realizations and class prompt.

As theoretically established earlier, standard uniform white-noise SDEs frequently struggle to resolve high-frequency spatial structures, often yielding blurry or structurally degraded textures. Conversely, while deterministic ODEs preserve structure, they suffer from accumulation errors that lead to oversmoothed, artificial appearances. By dynamically routing stochastic energy to structurally unresolved frequency bands, CNS consistently bridges this gap, yielding sharper fine details (e.g., fur, feathers, and foliage) and a more globally coherent output that better aligns with the true data manifold.

Class: Llama Class: Macaw

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

Class: Bald Eagle Class: Toucan

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Class: Jaguar Class: Lion

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Class: Tiger Class: Castle

- Figure 7: Visual comparison of samples generated using ODE, standard SDE, and our proposed CNS framework (without CFG). All images in a given triplet were generated using the same seed.

Class: Chow Class: Blenheim Spaniel

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Class: Bison Class: Sports Car

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Class: Lesser Panda Class: Great Grey Owl

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

Class: Peacock Class: Wolf Spider

- Figure 8: Visual comparison of samples generated using ODE, standard SDE, and our proposed CNS framework (without CFG). All images in a given triplet were generated using the same seed.

