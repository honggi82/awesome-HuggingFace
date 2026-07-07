## TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling

Hyunmin Cho1* Donghoon Ahn2* Susung Hong3* Jee Eun Kim1 Seungryong Kim4† Kyong Hwan Jin1†

# arXiv:2510.04533v2[cs.CV]26May2026

### Abstract

Diffusion models achieve state-of-the-art image generation but often produce semantic inconsistencies, or hallucinations. Existing inferencetime guidance methods rely on external signals or architectural modifications, adding computational overhead. We propose Tangential Amplifying Guidance (TAG), a training-free, architectureagnostic, plug-and-play guidance method that operates purely on trajectory signals. TAG uses an intermediate sample as a projection basis and amplifies the tangential components of the estimated score to correct the sampling trajectory. A firstorder Taylor analysis shows that this steers the state toward higher-probability regions of the data manifold, reducing inconsistencies and improving fidelity while adding negligible overhead to existing samplers. Code is available at our Project Page .

### 1. Introduction

Hallucination in diffusion models refers to the phenomenon of generating samples that violate the data distribution or contradict the conditioning, thereby failing to produce meaningful outputs. For example, it often manifests as mixed-up objects (Okawa et al., 2023; Oriyad et al., 2025) or anatomically implausible structures (e.g., extra-fingered hands). Recent evidence suggests that the primary source of such errors lies in a failure mode known as mode interpolation. During sampling, trajectories may traverse low-density valleys between distinct modes of the data distribution, causing attribute mismatches and structural inconsistencies (Aithal et al., 2024).

A widely adopted remedy involves inference-time guidance

*Equal contribution 1Korea University 2University of California, Berkeley 3University of Washington 4KAIST AI . Correspondence to: Seungryong Kim <seungryong.kim@kaist.ac.kr>, Kyong Hwan Jin <kyong jin@korea.ac.kr>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

[Figure 1]

[Figure 2]

[Figure 3]

(a) No Guidance (b) TAG Update

Figure 1. Conceptual visualization of Tangential Amplifying Guidance (TAG) from a mode-interpolation perspective (Aithal et al., 2024). Unlike (a) no guidance case, (b) TAG decomposes the base increment ∆k+1 on the latent sphere into parallel PMk+1∆k+1 and orthogonal (i.e., tangential) PM⊥k+1∆k+1 components (Equation (7)). By preserving the parallel component while adding a scaled tangential component, TAG isolates the data-relevant part of the update (Section 3) and can more effectively navigate the data manifold, leading to samples that contain more semantic structure. We make this precise by proving that amplifying the tangential component has the effect of guiding the trajectories toward regions of higher model density while mitigating off-manifold drift (Section 4 and Equation (18)).

strategies, such as classifier-free guidance (CFG) (Ho & Salimans, 2021) and their variants (Hong et al., 2023; Ahn et al., 2024; Hong, 2024; Rajabi et al., 2025; Kwon et al., 2025; Sadat et al., 2025; Dinh et al., 2025; Kim & Sim, 2025). Under the assumption that deviating from low-probability regions enhances sample quality, most of these methods employ residual scaling, using the difference between the conditional and unconditional branches to guide the generation process away from the unconditional model’s outputs. While effective, such guidance is largely geometry-unaware: it applies a single scalar magnification to the cond–uncond residual, without accounting for the local directional structure of the data distribution at each noise level, which can inadvertently distort the denoising trajectory.

Motivated by this, we instead adopt a score-level view of guidance grounded in Tweedie’s identity (Tweedie, 1984), which relates the score to the posterior mean under Gaussian corruption. This link motivates a decomposition of the model update based on its intrinsic geometry: a drift component that advances the radius along the prescribed noise schedule (i.e., noise level), and a tangential component that moves along the data-manifold, approximately preserving the overall radius while refining the sample’s structure and semantics. We observe that the tangential component carries rich structural information (Figure 2), and amplifying it

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

⊥∆P∆TAG∆P∆MM

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

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

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

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

t = 981 881 781 681 581 481 381 281 181 t = 981 t = 501 Gen. sample

- Figure 2. Amplifying the tangential component enhances semantic content by isolating it from noise. This figure illustrates the decomposition of the update step ∆ into normal and tangential components. Subtracting the unstructured, noisy normal component

PM∆ from the original update acts as a denoising operation, revealing the tangential component PM⊥ ∆, which preserves the principal semantic structure. Images decoded from intermediate timesteps (t = 981, 501) indicate that semantic information is most salient in the tangential component. Motivated by this observation, our method ∆TAG amplifies this semantically rich component, yielding a clearer and more coherent final sample (far right) than that obtained from the unmodified ∆.

reduces out-of-distribution samples (Figure 3).

Drawing upon the principle of amplifying the tangential component during inference, we derive Tangential Amplifying Guidance (TAG), a plug-and-play method that emphasizes the tangential component of the score update. TAG steers the sampling trajectory to closely follow the underlying data manifold. TAG integrates seamlessly with standard diffusion backbones—whether conditioned or not—without requiring additional denoising evaluations or retraining.

We can summarize our contributions as follows:

- • We establish a concrete link between the score’s intrinsic geometry and sample quality, proving that amplifying the tangential components steers sampling trajectories toward the in-distribution manifold.
- • We introduce TAG, a theoretically grounded, computationally lightweight, and architecture-agnostic algorithm that realizes this geometric principle in practice.

### 2. Preliminaries

Score-based Diffusion Model. Score-based generative models learn a time-indexed score function that approximates the gradient of the log-density of noise-perturbed data,

sθ(x,tk) ≈ ∇x log p(x | tk),

tk ∈ {tK > ··· > t0} denotes the k-th discretized timestep,

to reverse a gradual noising process for sample generation. This approach provides a continuous-time framework that unifies earlier discrete-time Denoising Diffusion Probabilistic Models (DDPMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020) through the lens of stochastic differential equations (SDEs) (Song et al., 2021b). The core idea involves a forward-time SDE that transforms complex data into a

simple prior distribution, given by

dx = f(x,t)dt + g(t)dW. Generation is then performed by the corresponding reversetime SDE, which becomes tractable by substituting the unknown true score with the learned model sθ (Anderson, 1982). To solve this numerically, we discretize the time horizon over timesteps tk ∈ {tK > ··· > t0}. This score network, typically a noise-conditional U-Net, is trained efficiently via denoising score matching across various noise levels (Vincent, 2011; Song & Ermon, 2019). For sampling, one can use numerical methods such as predictorcorrector schemes to simulate the stochastic reverse SDE, or solve an associated deterministic ordinary differential equation (ODE), known as the probability-flow ODE. This continuous-time framework not only provides a theoretical basis for widely used deterministic samplers like DDIM (Song et al., 2021a), but also has inspired modern refinements, such as preconditioning and parameterization in EDM (Karras et al., 2024b), which further enhance the trade-off between sample quality and efficiency.

Inference-Time Guidance. Numerous methods modify the update field during sampling to improve fidelity without retraining. Early CFG-style guidance (Ho & Salimans, 2021) steers samples by rescaling residual signals, and complementary approaches replace external cues with model-internal signals for guidance (Hong et al., 2023; Ahn et al., 2024; Hong, 2024). However, prior analyses show that na¨ıve, geometry-agnostic scaling can reduce diversity or perturb solver dynamics (Dhariwal & Nichol, 2021; Kynk¨a¨anniemi et al., 2024). These limitations motivate geometry-aware guidance that asks not only how much to scale, but which directions to emphasize. Representative methods along this line use projections to dampen undesired components (e.g., high-scale saturation or cond–uncond mismatch) (Armandpour et al., 2023; Sadat et al., 2025; Kwon et al., 2025), or

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

(a) No Guidance (b) Naive Truncation (c) CFG (d) TAG (Ours) (e) Ground Truth

- Figure 3. Sampling on a 2D branching distribution (Karras et al., 2024a) under different guidance methods. (a) No guidance: probability mass drifts off the data manifold, yielding fragmented branches and OOD (Out of Distribution) points. (b) Naive truncation: suppresses some OOD but oversimplifies the geometry, dropping fine branches. (c) CFG: reduces boundary violations but also reduces diversity and can still leave OOD strays in our run. (d) TAG (Ours): trajectories are steered toward high-density regions along the branches, suppressing off-manifold outliers while retaining detail. (e) Ground truth. Overall, TAG achieves the highest similarity to the GT distribution without additional #NFEs, concentrating mass on the correct branches while reducing OOD outliers.

are developed for more specific problem settings such as loss-guided inverse problems or unpaired I2I (Sun et al., 2023; He et al., 2024) (see Appendix E for more detailed discussion). Overall, these strategies integrate cleanly with modern solvers and effectively suppress off-manifold drift, but are often closely tied to particular guidance algebra or task-specific assumptions.

### 3. Motivation and Intuition

Under Gaussian corruption, Tweedie’s formula (Tweedie, 1984) links the posterior mean of the clean signal to the noisy observation via the score (i.e., the gradient of the log marginal density):

≜ Tweedie Increment ∆Twk

1 √α¯k

σk2∇x log p(x | tk) x=x

E[x0 | xk] =

xk +

,

k

(1)

where xk denotes the noisy observation and ∆Twk denotes the data-driven correction term. Geometrically, the score field

points in the direction of steepest increase of the marginal density. Tweedie’s formula therefore adjusts xk in this ascent direction, nudging the state toward higher-probability regions. Therefore, the aim of modeling is to accurately capture these data-driven directions.

∇x log p(x|tk) x=x

k

To get better intuition for these data-driven directions, we appeal to the Gaussian annulus theorem (Blum et al., 2020), which states that an isotropic Gaussian in Rd concentrates most of its mass near a thin spherical shell. Since each corruption step adds such noise to the data, the corrupted distribution p(x|tk) likewise places most of its mass near a thin shell. Consequently, we can regard the high-probability region of p(x | tk) as a noisy data manifold Mk embedded near this thin shell. Our goal is therefore to move along the high-probability shell of the corrupted distribution, rather than pushing the sample inward or outward in radius, which is already dictated by the diffusion noise schedule.

[Figure 60]

Figure 4. Illustration of Tangential Amplification. The curves Mk denote the noisy data manifolds at successive diffusion steps. TAG decomposes ∆k+1 into normal and tangential components with respect to Mk+1, and amplifies the tangential component within the local tangent space Txk+1Mk+1 by a factor η.

To separate the radial (normal-to-shell) component from the within-shell (surface-direction) component, we define:

= xk x⊤k , PM⊥

##### = I − PM

##### PM

(2)

k

k

k

where xk := xk/∥xk∥2,

where PM

projects onto the radial direction (normal to the shell) and PM⊥

k

onto the within-shell directions (Figure 4). Guided by this separation, we form the amplified state x+

k

∆Twk + η PM⊥

∆Twk , with η ≥ 1.

x+ = xk + PM

k

k

(3) By doing so, we preserve the radial first-order term (Equation (22)) while allocating additional gain to the within-shell component of the Tweedie increment, steering the sampling step toward higher-density regions along Mk. In the following section (Section 4.1), we formalize this idea as a constrained MLE update that allocates first-order gain to the tangential subspace.

### 4. TAG: Tangential Amplifying Guidance

We introduce Tangential Amplifying Guidance (TAG), which reweights base increments along normal/tangential directions on the latent space.

Definitions & Algorithm. We work per sample on RC×H×W ∼= Rd with Euclidean inner product ⟨·,·⟩ and norm ∥ · ∥2. Let {tk}0k=K be descending timesteps with tK > ··· > t0, and let ϵθ denote the denoiser. Given xk+1 at time tk+1, the denoiser predicts

εk+1 = ϵθ(xk+1,tk+1). A base solver (e.g., DDIM) then produces a provisional state (Karras et al., 2024b)

x˜k = ak+1 xk+1 + bk+1 εk+1, (4)

where ak+1,bk+1 are base solver coefficients. Corresponding base increment at xk+1 is defined as

∆k+1 := x˜k − xk+1. (5) For any x ∈ Rd, we define the unit vector and orthogonal projectors

PM(x) = x x⊤, PM⊥ (x) = I − PM(x), where x = ∥xx∥

.

2

(6)

- Figure 4 illustrates this tangential–normal decomposition along the sampling trajectory. Given positive scales η ≥ 1, TAG reweights the base increment at xk+1:

∆k+1 + η PM⊥

xk ← xk+1 + PM

∆k+1 where PM

k+1

k+1

= PM(xk+1),PM⊥

= PM⊥ (xk+1).

k+1

k+1

(7)

#### 4.1. Why does TAG improve Image Quality?

Log-likelihood maximization. A foundational goal of training generative models is to maximize the log-likelihood of the data, as formalized by the Maximum Likelihood Estimation (MLE) principle:

log pθ(xi). (8)

max

θ

i

This principle suggests that high-quality samples should concentrate in regions of high probability. To connect this idea to an update rule, we relate likelihood increase to movement along the score via a local linearization:

log pθ(x) = log pθ(x0)

(9)

+ (x − x0)⊤∇x log pθ(x) x=x

+ O(∥·∥2).

0

Diffusion models (Ho et al., 2020; Song et al., 2021b) are designed to predict a score function, ∇x log p(x | tk) x=x

≈ −ϵθ(xk,tk)/σk, which operates on noisy versions of the data. Because diffusion models learn this score field, optimizing the global likelihood (Equation (8)) for a sample x0 during inference is not directly tractable. Therefore, we propose to apply the spirit of MLE at each local step of the sampling trajectory.

k

log p(xk|tk+1) ≈ log p(xk+1|tk+1) + (xk−xk+1)⊤ ∇x log p(x | tk+1) x=x

##### + O(∥·∥2).

k+1

(10)

Algorithm 1 Tangential Amplifying Guidance (TAG)

- 1: Input: Denoiser ϵθ(·), timesteps {tk}0k=K, base solver coefficients ak+1,bk+1, TAG scale η ≥ 1
- 2: Sample xK ∼ N(0,I)
- 3: for k = K − 1 down to 0 do
- 4: εk+1 ← ϵθ(xk+1,tk+1)
- 5: x˜k ← ak+1xk+1 + bk+1 εk+1
- 6: ∆k+1 ← x˜k − xk+1
- 7: xk+1 ← xk+1/∥xk+1∥2
- 8: PM

k+1 ← xk+1 x⊤k+1

- 9: PM⊥

k+1

← I − PM

k+1

- 10: xk ← xk+1 + PM

k+1

∆k+1 + η (PM⊥

k+1

∆k+1)

- 11: end for
- 12: Output: x0

The idea of enhancing a pre-trained score function with inference-time guidance has proven effective. For instance, when the score function is well-trained on given training sets, and this leads to a well-trained maximum log-likelihood, we observe that the pre-trained score function can be improved by CFG (Ho & Salimans, 2021), which linearly steers the score toward the conditional target. Inspired by this, our approach provides inference-time guidance on the score function by maximizing the following local log-likelihood term over admissible one-step updates (with a step budget δk), thereby guiding the sampling trajectory towards highlikelihood regions of the data distribution and reducing offmanifold artifacts (hallucination):

(xk − xk+1)⊤∇x log p(x | tk+1) x=x

max

, where C := ∥xk − xk+1∥2 ≤ δk

xk : C

(11)

k+1

Single-step increment decomposition. For deterministic DDIM/ODE samplers, the single-step score state decomposition can be written as

∆k+1 := x˜k − xk+1 = α˜kϵθ(xk+1,tk+1) + βkxk+1,

(12) with coefficients

√α¯k

√α¯k

α˜k := σk −

√α¯k+1 − 1, with α˜k < 0, βk > 0,

√α¯k+1 σk+1, βk :=

(13)

where α¯ is the standard diffusion cumulative product term. Using the projection operators, which satisfy

##### PM⊥

xk+1 = xk+1, (14) yields the projection-wise identities

xk+1 = 0, PM

k+1

k+1

∆k+1 = α˜kPM⊥

##### PM⊥

ϵθ(xk+1,tk+1), PM

k+1

k+1

∆k+1 = α˜kPM

ϵθ(xk+1,tk+1) + βkxk+1.

k+1

k+1

(15) Substituting Equation (15) into the Equation (7) gives

xTAGk = xk+1 + α˜k PM

+ ηPM⊥

ϵθ(xk+1,tk+1)

k+1

k+1

+ βkxk+1, with η ≥ 1.

(16)

Therefore, the TAG update ∆TAGk+1 can be expressed in terms of the decomposed components of the original up-

date ∆k+1:

+ η PM⊥

∆TAGk+1 = PM

∆k+1. (17) In this way, as visualized in Figure 2, semantic information can be isolated from the update vector via the tangential projection, thereby enabling semantics-aware amplification. To quantify its effect on the log-likelihood, assume the logdensity is smooth (i.e., log p(·|tk+1) is C2 in a neighborhood of xk+1). The first-order Taylor expansion gain for a small TAG update ∆TAGk+1 ∈ Rd is

k+1

k+1

G(η) := ∆TAGk+1 ⊤∇x log p(x | tk+1) x=x

. (18)

k+1

Next, we prove that increasing η provides a monotonic increase in this first-order gain.

Theorem 4.1 (Monotonicity of the First-order Taylor Gain). Assume a deterministic base step with ∆k+1 = α˜kϵθ(xk+1,tk+1) + βkxk+1 and α˜k ≤ 0. Let PM

⪰ 0 be the projectors defined above. For the TAG step ∆TAGk+1 = PM

k+1 ⪰ 0 and PM⊥

k+1

∆k+1 + η PM⊥

k+1

∆k+1, the first-order Taylor gain G(η) := ∆TAGk+1 ⊤∇x log p(x | tk+1) x=x

k+1

satisfies ∂G(η)

k+1

−α˜k σk+1

ϵθ(xk+1,tk+1) 22 ≥ 0,

PM⊥

∂η ≈

k+1

(19) and, in particular,

≥ 0 as α˜k ≤ 0

−σk−+11 · α ˜k(η − 1) · PM⊥

GTAG − Gbase =

ϵθ(xk+1,tk+1) 22 ≥ 0,

k+1

(20) Equality holds iff η = 1 or the tangential component of the score is zero. The proof is provided in Appendix Section A.

Log-likelihood improvements via TAG. We cast inferencetime guidance as maximizing a log-likelihood gain (Equation (11)). TAG simply reweights the update step by amplifying the component that is orthogonal to the current state while leaving the parallel component unchanged. By Theorem 4.1, increasing the orthogonal weight monotonically raises the first-order Taylor gain, so TAG steers the sampler toward higher-density regions of the data manifold, improving image quality.

Avoidance of normal amplification. Amplifying the tangential component monotonically increases the first-order term of a Taylor gain of log p(· | tk+1) (Theorem 4.1), which produces samples with fewer hallucinations. However, amplifying the normal component increases radial contraction and leads to over-smoothing (Figure 5). This radial component of the single-step is aligned with the radial

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

+ TAG #NFEs=50

+ TAG + Normal #NFEs=50

Uncond. #NFEs=50

Uncond. #NFEs=250

Figure 5. Effectiveness of TAG. At 50 NFEs, TAG surpasses the sample quality at 250 NFEs from baseline. In contrast, +Normal causes severe over-smoothing.

part of Tweedie’s correction, which links xk to the posterior mean E[x0|xk] via the score function (Tweedie, 1984; Song et al., 2021b). Formally, rescaling the normal part by a κ (> 1), the radial first–order change is multiplied by κ:

⟨ xk+1,∆(kκ+1) ⟩ = κ⟨ xk+1,∆k+1⟩. (21) Therefore, a value of κ (> 1) excessively strengthens this contraction under the VP/DDIM schedule, leading to oversmoothing. In contrast, tangential scaling preserves the radial first–order term:

⟨ xk+1,∆TAGk+1 ⟩ = ⟨ xk+1,∆k+1⟩. (22) Thus, normal amplification breaks one–step calibration and induces over-smoothing, whereas tangential boosting improves alignment without disturbing the radial schedule.

#### 4.2. TAG for Classifier-Free Guidance

Sections 3 and 4.1 show that the tangential component encodes data-relevant directions and is radius-preserving to first-order; so amplifying it improves image quality by steering updates along data-aligned directions. In CFG (Ho & Salimans, 2021), the guided score combines conditional and unconditional branches

εk = ϵθ(xk,∅) + ω(ϵθ(xk,c) − ϵθ(xk,∅)). (23) Because these two scores follow distinct trajectories, an incoherence between them can arise, and such an effect can degrade generation quality, an issue recently highlighted by Kwon et al. (2025). Motivated by this established score mismatch, and informed by our core intuition that the tangential field encodes data geometry (Equation (1)), we posit that this incoherence is fundamentally tangential in nature; that is, a persistent mismatch exists primarily between the conditional and unconditional tangential components.

Conditional–unconditional tangent reconciliation. Let εu := ϵθ(xk,∅) and εc := ϵθ(xk,c) denote the unconditional/conditional predicted noise, and let gk := εc − εu be the usual CFG residual. We extract its tangential component via the projector PM⊥ (xk) and define the conditionalrelative tangent

gk⊥ := PM⊥ (xk)gk = PM⊥ (xk) εc − εu . (24) We then align the conditional prediction with this tangential residual direction by projecting εc onto span(gk⊥):

⊥ k ⟩

gkalign := ⟨εc,g

gk⊥. (25)

∥gk⊥∥22

Table 1. Quantitative comparison of unconditional generation by SD-series on COCO 2014. Uncond. FID ↓ IS ↑ AES ↑ CMMD ↓

Table 2. Quantitative comparison of conditional generation by SD-series on COCO 2014.

Cond. FID ↓ ImageReward ↑ CLIP↑ SD v1.5 33.49 -0.342 25.00 TAGSDv1.5 26.61 -0.339 25.09 SD v2.1 26.12 0.143 25.35 TAGSDv2.1 21.59 0.424 26.16 SDXL 29.28 0.274 25.41 TAGSDXL 28.53 0.292 25.49 SD3 29.02 1.030 26.39 TAGSD3 27.54 1.043 26.56

SD v1.5 58.41 15.59 5.003 1.069 TAGSDv1.5 46.20 16.77 5.064 0.778 SD v2.1 78.54 12.52 5.299 1.395 TAGSDv2.1 59.94 13.36 5.320 1.122 SDXL 119.14 9.08 5.645 2.474 TAGSDXL 90.71 8.91 5.577 2.201 SD3 84.26 11.53 5.261 1.671 TAGSD3 79.11 11.73 5.365 1.564

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

+TAGBaseline

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

Unconditional Gen. (SD3) Conditional Gen. (SDXL) Plug-and-Play: PAG+SDXL Plug-and-Play: SEG+SDXL

- Figure 6. Qualitative comparison of TAG. Left four columns demonstrate that TAG enhances the detail and coherence of samples from SDXL/3 under both unconditional and conditional generation. Right four columns illustrate its plug-and-play compatibility: TAG can be applied on top of existing methods (e.g., PAG (Ahn et al., 2024), SEG (Hong, 2024)) to further improve their outputs. Finally, we define TAG on CFG as

#### 5.1. Improvements on Stable Diffusion Series

ε˜k = εu + ω gk + η gkalign, (26) where ω is the usual CFG scale and η controls the extra tangential emphasis. Further implementation details are provided in Appendix Section D.

We evaluate TAG across the Stable Diffusion family under both unconditional and conditional generation settings. Table 1 presents unconditional results on widely used backbones. Without requiring additional #NFEs, TAG consistently improves FID and CMMD across all backbones. This improvement extends to the conditional setting: Table 2 reports T2I results on COCO2014, where TAG consistently improves both sample fidelity and prompt alignment across diverse measures. Qualitatively, Figure 6 corroborates these findings: TAG suppresses artifacts and produces more coherent structures, resolving blob-like formations and implausible anatomy (e.g., a woman with three legs), and yielding sharper, cleaner compositions (e.g., the dog-and-cat scene).

### 5. Experiments

General evaluation setup. We apply TAG at inference time to pretrained backbones, including Stable Diffusion v1.5/v2.1/XL (Rombach et al., 2022; Podell et al., 2024) and Stable Diffusion 3 (Esser et al., 2024), which follows a flowmatching formulation (Lipman et al., 2023; Liu et al., 2023). We also use representative guidance methods, PAG (Ahn et al., 2024) and SEG (Hong, 2024), as comparison baselines and plug-and-play modules. We further evaluate TAG with SSG (Zhang et al., 2026) in Appendix Section B.1 to assess its applicability across different guidance schemes.

Compatibility with existing guidance methods. TAG is orthogonal to existing residual-based guidance methods such as PAG and SEG, with an additional SSG compatibility check provided in Appendix Section B.1. Tables 3 and 4 show that stacking TAG on top of these methods further improves sample quality at matched #NFEs, without architectural changes or additional model evaluations. Notably, the gains are consistent across several metrics, especially on human-preference-oriented scores, suggesting that TAG can refine existing guidance outputs without substantially changing the overall distribution. As shown in Figure 6, TAG fixes both an action-binding error (“A girl flying a red kite...”baseline depicts the girl as airborne) and a numeracy error (“A horse standing in a field...”—baseline yields two). Additional qualitative comparisons are provided in Figures 12 to 14 in Appendix Section C.3.

General evaluation metrics and protocol. We report FID (Seitzer, 2020) (distributional similarity to real images) and IS (Salimans et al., 2016) (image quality and diversity) as standard generative metrics, alongside humanaligned metrics that assess distinct dimensions of generation quality: the Aesthetic Score Predictor (Schuhmann et al., 2022) (visual appeal), CMMD (Jayasumana et al., 2024) (distributional similarity in CLIP space), ImageReward (Xu et al., 2023) (overall human preference), and CLIPScore (Hessel et al., 2021) (text–image alignment). We use COCO2014 (Lin et al., 2014) for evaluation. Further details are provided in Appendix Section F.

Table 3. Quantitative comparison with existing guidance methods for unconditional generation on the SD-v1.5 using COCO 2014. Uncond. FID ↓ IS ↑ AES ↑ CMMD ↓ No guidance 58.41 15.59 5.003 1.069 TAG 46.20 16.77 5.064 0.778 PAG (Ahn et al., 2024) 53.72 21.13 5.303 0.723 TAG + PAG 52.61 21.20 5.305 0.701 SEG (Hong, 2024) 47.69 18.50 5.084 0.835 TAG + SEG 42.71 19.45 5.076 0.746

Table 4. Quantitative comparison with existing guidance methods for conditional generation on the SD-XL using COCO 2014. Cond. FID ↓ IR ↑ Pick ↑ CLIP ↑ No CFG 83.74 -0.897 20.19 21.16 TAG 69.87 -0.710 20.35 22.37 CFG + PAG 30.24 0.352 22.13 25.23 TAGcfg + PAG 30.22 0.354 22.13 25.25 CFG + SEG 34.47 0.354 21.97 25.08 TAGcfg + SEG 34.05 0.376 21.99 25.15

Table 5. Compositional faithfulness evaluation on T2I-CompBench. TAG is applied to Stable Diffusion XL on the Spatial and Complex subsets, where structural hallucinations are most common.

Complex-Val300 BLIP-VQA ↑ 2DSpatial ↑ CompCLIP ↑ 3-in-1 ↑ AES ↑ CLIP ↑ IR ↑

Spatial-Val300 2DSpatial ↑ AES ↑ CLIP ↑ IR ↑ SDXL 0.1857 5.779 27.365 0.800 +TAGcfg 0.1980 5.768 27.714 0.911

Method

Method

SDXL 0.4443 0.0243 0.2910 0.3364 5.666 25.975 0.2596 +TAGcfg 0.4650 0.0232 0.2937 0.3472 5.667 26.477 0.3978

#### 5.2. Validation on Hallucination-oriented Benchmark

To further assess whether TAG’s gains reflect improved compositional faithfulness rather than altered perceptual style, we evaluate on T2I-CompBench (Huang et al., 2023), focusing on the ‘Spatial’ and ‘Complex’ subsets where structural hallucinations are most prevalent (e.g., incorrect object placement and multi-object relations). As shown in Ta-

- ble 5, TAG applied to SDXL improves key compositional metrics while maintaining comparable aesthetic and CLIPScores, suggesting that TAG mitigates structural artifacts in hallucination-prone prompts rather than merely shifting perceptual style.

5.3. Comparison with geometry-aware guidance.

A line of work improves T2I generation by modifying the CFG signal. In particular, TCFG (Kwon et al., 2025) and APG (Sadat et al., 2025) exploit geometric structure in the guidance signal to mitigate undesirable effects induced by guidance. To contextualize TAGcfg within this family, we compare it with related geometry-aware CFG methods. Ta-

- ble 6 reports this comparison under a unified evaluation

protocol: TAGcfg achieves the lowest FID among geometryaware CFG variants, although APG yields the highest CLIPScore. These results suggest that tangential trajectory amplification constitutes an effective and distinct form of CFGsignal correction. Qualitative comparisons are provided in Figures 15 and 16 in Appendix Section C.4.

#### 5.4. Improvements on Modern Image & Video Models

Image generation. We further evaluate TAG on QwenImage (Wu et al., 2025), a strong contemporary backbone whose structural artifacts have been largely mitigated by data curation and fine-tuning. As shown in Table 7, TAG improves NR-IQA metrics (Ke et al., 2021; Wang et al., 2023) alongside a marginal FID increase, suggesting that on a backbone already near the reference distribution, tangential amplification primarily refines perceptual detail rather

- Table 6. Quantitative comparison with geometry-aware guidance methods (APG, TCFG) and TAG on COCO 2014. COCO@10K FID ↓ CLIPScore ↑ TCFG (Kwon et al., 2025) 20.72 25.63 APG† (Sadat et al., 2025) 19.52 26.71 TAGcfg 19.29 26.23

†Note. APG primarily targets oversaturation artifacts in the high-CFG regime. We include it to provide a reference point within geometry-aware CFG methods.

- Table 7. TAG on Qwen-Image. TAG improves no-reference perceptual quality metrics on a strong contemporary backbone.

COCO@10K CLIPIQA ↑ MUSIQ ↑ FID ↓ Qwen-I (Wu et al., 2025) 0.4947 67.90 57.53 TAG 0.5051 68.85 59.84

than shifting distributional statistics. This confirms that TAG can serve as a complementary refinement even for well-optimized pipelines, improving perceived visual quality with only a small trade-off in FID on the base model.

Video generation. To validate TAG beyond image generation, we evaluate it on Wan2.2 (Team Wan et al., 2025), using 100 randomly sampled VBench (Huang et al., 2024) prompts. As shown in Table 8, TAG improves over the baseline on most metrics, including dynamic degree, imaging quality, aesthetic quality, overall consistency, and temporal style. Notably, the increased dynamic degree alongside quality gains suggests that TAG supports more dynamic video generation without simply compressing motion. As shown in Figure 7, Wan2.2 occasionally produces structural artifacts in scenes with complex object compositions, whereas TAG reliably preserves the coherence of such scenes. Additional qualitative comparisons are provided in Figure 11 in Appendix Section C.2.

### 6. Discussion

Solver-agnostic improvements across ODE solvers. Table 9 compares TAG with representative stronger samplers for unconditional diffusion sampling on ImageNet using SD v1.4 (Rombach et al., 2022). TAG alone provides a substantial improvement over the baseline and achieves per-

Table 8. Video generation with Wan2.2. TAG is evaluated on 100 VBench prompts.

Visual Quality Consistency Temporal Dyn. Deg.↑ Img. Qual.↑ Aes. Qual.↑ BG Cons.↑ Subj. Cons.↑ Overall Cons.↑ Human Act.↑ Mot. Smooth↑ Temp. Style↑

Method

Wan2.2 0.52 0.7047 0.5647 0.9644 0.9613 0.1796 0.05 0.9871 0.1796 TAG 0.56 0.7066 0.5697 0.9672 0.9625 0.1836 0.07 0.9864 0.1836

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

TAGBaseline

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 7. Qualitative comparison of Wan 2.2 baseline (top) and TAG (bottom) on “A bicycle slowing down to stop.” Six uniformly sampled frames from a 49-frame clip.

- Table 9. Comparison with stronger unconditional samplers and their plug-and-play combination with TAG. All experiments are conducted with Stable Diffusion v1.4 on ImageNet, evaluated by FID@30K and Inception Score (IS). Method FID@30K ↓ IS ↑ Baseline 65.55 14.53 ± 0.32 DPM++ (Lu et al., 2023) 55.40 15.92 ± 0.25 TAG 52.83 16.16 ± 0.31 UniPC (Zhao et al., 2023) 50.82 16.37 ± 0.25

▶ Plug-and-Play

UniPC + TAG 48.34 16.86 ± 0.36 DPM++ + TAG 44.08 17.77 ± 0.36

- Table 10. Computational Cost Analysis. Measurements were conducted using the DeepSpeed Profiler. FLOPs represents the estimated floating-point operations per image. Cost Param. FLOPs VRAM Latency Overhead Baseline 0.86 B 40.99 T 2.61 GB 0.919 s N/A PAG 0.86 B ≈ 82 T 4.95 GB 1.956 s +112.8% TAG 0.86 B ≈ 41 T 2.61 GB 1.005 s +9.4%

formance competitive with widely used sampler variants such as DPM++ (Lu et al., 2023) and UniPC (Zhao et al., 2023). Furthermore, applying TAG on top of these samplers yields additional gains in both FID and IS. Together, these results position TAG as a lightweight refinement of the sampling update that is both competitive as a standalone method and complementary in a plug-and-play manner to stronger solver variants.

Efficiency and overhead. We evaluate the computational implications of TAG by reporting FLOPs, peak memory, and wall-clock inference time. All measurements are obtained on an NVIDIA RTX 4090 using the DeepSpeed Profiler (Rajbhandari et al., 2020). As shown in Table 10, diffusion sampling is largely dominated by the UNet forward pass. PAG (Ahn et al., 2024), which introduces a perturbed attention branch and requires an extra UNet evaluation per sampling step, substantially increasing compute, memory, and latency. In contrast, TAG is applied directly to the solver update computed from the same UNet pass and introduces only lightweight vector projections. Empirically, TAG matches baseline peak memory and adds only marginal inference-time overhead while improving sample quality.

Table 11. Effect of timestep windowing in TAG. TAG is activated only during a timestep window, motivated by the Gaussian annulus theorem. We evaluate FID/IS using 30K ImageNet val samples with Stable Diffusion v1.5, DDIM sampler (Song et al., 2021a).

Uncond. [ηsta, ηend] scale η #NFEs FID ↓ IS ↑ TAGSDv1.5 [400, 0] 1.15 50 69.428 16.104 TAGSDv1.5 [1000,0] 1.15 50 67.805 16.487 TAGSDv1.5 [1000,400] 1.15 50 63.870 17.516

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

baseline η = 1.05 η = 1.1 η = 1.3

Figure 8. Ablation on TAG amplification η. Moderate η improves detail and coherence, while excessive η reduces fidelity.

#### 6.1. η Selection and Adaptive Scaling

The amplification factor η controls the strength of the tangential correction in TAG. As illustrated in Figure 8, moderate amplification improves structural detail and visual coherence, whereas larger values reduce fidelity and produce overly smoothed or distorted outputs.

A simple practical way to mitigate this sensitivity is to restrict TAG to a selected timestep interval, specified by [ηsta,ηend] in Table 11, rather than applying the same amplification over the entire sampling trajectory. In particular, activating TAG mainly in the high-to-intermediate noise regime can avoid over-amplifying late denoising steps, where small tangential perturbations more directly affect visible image details. However, such timestep windowing still requires manual tuning and is model- and samplerdependent, and therefore remains heuristic.

This non-monotone behavior motivates treating η as a sensitivity parameter rather than a simple improvement knob, and highlights the need for adaptive scaling, described below.

Adaptive Scaling for Robust Amplification. A practical challenge in TAG is that the effect of η depends not only on its nominal value, but also on the instantaneous magnitude of the tangential correction. Since ∥gkalign∥ can vary

Table 12. Fixed vs. adaptive scaling of η in TAG. FID@10K on COCO for conditional generation. COCO Baseline η = 1.0 η = 2.0 η = 3.0 Fixed 28.24 25.39 26.61 29.63 Adaptive 28.24 26.34 25.90 25.86

substantially across timesteps, a fixed global η may produce inconsistent correction strengths. Recall the TAG update at step k (Equation (26))

ε˜k = εu + ω gk + η gkalign. (27) To reduce this sensitivity, we normalize the tangential term relative to the original guidance scale:

gˆkalign = ∥gk∥

gkalign. (28)

∥gkalign∥ + δ

where 0 < δ ≪ 1 ensures numerical stability. Under this formulation, η controls the relative amplification of the tangential term with respect to ∥gk∥, making the update robust to timestep-dependent variation in ∥gkalign∥. As shown in Table 12, adaptive scaling maintains stable performance across a wider range of η, whereas fixed scaling degrades rapidly at larger values.

### 7. Conclusion, Limitations, and Implications

This paper offers a geometric view of hallucinations in diffusion models by decomposing each sampling update into normal and tangential components. We find that the tangential component is closely tied to semantic organization, motivating a simple intervention on this direction. Building on this insight, we propose Tangential Amplifying Guidance (TAG), a theoretically grounded, computationally lightweight, architecture-agnostic method that amplifies the tangential component during sampling. TAG steers trajectories toward higher-density regions of the data manifold, producing samples with fewer hallucinations and improved fidelity.

Limitations. Our theoretical analysis establishes local firstorder likelihood gain, but does not fully characterize the behavior of the reverse trajectory. A simple radial–tangential approximation enables efficient guidance, yet may not capture the full geometry of complex data manifolds.

Implications and Future work. Despite these simplifications, our results suggest viewing diffusion guidance as geometric shaping of the sampling trajectory. They also indicate that tangential components can serve as semantic refinement signals, motivating future trajectory-level analyses and lightweight estimates of local manifold geometry.

### Acknowledgments

This work was partly supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00335741), Institute of Informa-

tion & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (RS2025-25442405, Development of a Self-Learning World Model-Based AGI System for Hyperspectral Imaging), and Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism(RS-2024-00345025, International Collaborative Research and Global Talent Development for the Development of Copyright Management and Protection Technologies for Generative AI).

### Impact Statement

This paper presents Tangential Amplifying Guidance (TAG), an inference-time modification to diffusion sampling that improves fidelity with minimal additional computation. TAG may be useful in applications such as creative tools, prototyping, and education, but higher-fidelity generation can also increase risks of misuse, including deceptive content. Since TAG builds on existing models, it may inherit their biases and limitations. Deployments should use appropriate safety, legal, and licensing safeguards.

### References

Ahn, D., Cho, H., Min, J., Jang, W., Kim, J., Kim, S., Park, H. H., Jin, K. H., and Kim, S. Self-rectifying diffusion sampling with perturbed-attention guidance. In European Conference on Computer Vision, pp. 1–17. Springer, 2024.

Aithal, S. K., Maini, P., Lipton, Z. C., and Kolter, J. Z. Understanding hallucinations in diffusion models through mode interpolation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=aNTnHBkw4T.

Anderson, B. D. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313– 326, 1982.

Armandpour, M., Sadeghian, A., Zheng, H., Sadeghian, A., and Zhou, M. Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond, 2023. URL https://arxiv.

org/abs/2304.04968. Blum, A., Hopcroft, J., and Kannan, R. Foundations of Data Science. Cambridge University Press, 2020.

Chen, C. and Mo, J. IQA-PyTorch: Pytorch toolbox for image quality assessment. [Online]. Available: https:// github.com/chaofengc/IQA-PyTorch, 2022.

Chung, H., Kim, J., Mccann, M. T., Klasky, M. L., and Ye, J. C. Diffusion posterior sampling for general noisy

inverse problems. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=OnD9zGAGT0k.

Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Dinh, A.-D., Liu, D., and Xu, C. Representative guidance: Diffusion model sampling with coherence. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=gWgaypDBs8.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

He, Y., Murata, N., Lai, C.-H., Takida, Y., Uesaka, T., Kim, D., Liao, W.-H., Mitsufuji, Y., Kolter, J. Z., Salakhutdinov, R., and Ermon, S. Manifold preserving guided diffusion. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=o3BxOLoxm1.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., and Choi, Y. CLIPScore: A reference-free evaluation metric for image captioning. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7514–7528, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main. 595. URL https://aclanthology.org/2021.

emnlp-main.595/.

Ho, J. and Salimans, T. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. URL https:// openreview.net/forum?id=qw8AKxfYbI.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Hong, S. Smoothed energy guidance: Guiding diffusion models with reduced energy curvature of attention. Advances in Neural Information Processing Systems, 37: 66743–66772, 2024.

Hong, S., Lee, G., Jang, W., and Kim, S. Improving sample quality of diffusion models using self-attention guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7462–7471, 2023.

Huang, K., Sun, K., Xie, E., Li, Z., and Liu, X. T2icompbench: A comprehensive benchmark for open-world compositional text-to-image generation. Advances in Neural Information Processing Systems, 36:78723–78747, 2023.

Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., and Liu, Z. VBench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., and Kumar, S. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9307–9315, June 2024.

Karras, T., Aittala, M., Kynk¨a¨anniemi, T., Lehtinen, J., Aila, T., and Laine, S. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024a.

Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., and Laine, S. Analyzing and improving the training dynamics of diffusion models. In Proc. CVPR, 2024b.

Ke, J., Wang, Q., Wang, Y., Milanfar, P., and Yang, F. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 5148–5157, October 2021.

Kim, K. and Sim, B. PLADIS: Pushing the limits of attention in diffusion models at inference time by leveraging sparsity. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16238–16248, 2025.

Kwon, M., Kim, S., Jeong, J., Hsiao, Y. T., and Uh, Y. TCFG: Tangential damping classifier-free guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2620–2629, June 2025.

Kynk¨a¨anniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., and Lehtinen, J. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems, 37:122458–122483, 2024.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In Fleet, D., Pajdla, T., Schiele, B., and Tuytelaars, T. (eds.), Computer Vision – ECCV 2014, pp. 740–755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.

Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/ forum?id=PqvMRDCJT9t.

Liu, X., Gong, C., and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

net/forum?id=XVjTT1nw5z.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. DPM-solver++: Fast solver for guided sampling of diffusion probabilistic models, 2023. URL https: //openreview.net/forum?id=4vGwQqviud5.

Obukhov, A., Seitzer, M., Wu, P.-W., Zhydenko, S., Kyl, J., and Lin, E. Y.-J. High-fidelity performance metrics for generative models in pytorch, 2020. URL https: //github.com/toshas/torch-fidelity. Version: 0.4.0, DOI: 10.5281/zenodo.3786539.

Okawa, M., Lubana, E. S., Dick, R., and Tanaka, H. Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. Advances in Neural Information Processing Systems, 36:50173–50195, 2023.

Oriyad, A. M., Banayeeanzade, M., Abbasi, R., Rohban, M. H., and Baghshah, M. S. Attention overlap is responsible for the entity missing problem in text-to-image diffusion models! Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https: //openreview.net/forum?id=Xv3ZrFayIO.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Wallach, H., Larochelle, H., Beygelzimer, A., d'Alch´e-Buc, F., Fox, E., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.

cc/paper_files/paper/2019/file/ bdbca288fee7f92f2bfa9f7012727740-Paper. pdf.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=di52zR8xgf.

Rajabi, J., Mehraban, S., Sadat, S., and Taati, B. Token perturbation guidance for diffusion models. In Belgrave, D., Zhang, C., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., and Chen, N. (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 67153–67175. Curran Associates, Inc., 2025.

Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. ZeRO: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’20. IEEE Press, 2020. ISBN 9781728199986.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Sadat, S., Hilliges, O., and Weber, R. M. Eliminating oversaturation and artifacts of high guidance scales in diffusion models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=e2ONKX6qzJ.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., and Chen, X. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C. W., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., Schramowski, P., Kundurthy, S. R., Crowson, K., Schmidt, L., Kaczmarczyk, R., and Jitsev, J. LAION-5b: An open large-scale dataset for training next generation image-text models. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https:

//openreview.net/forum?id=M3Y74vmsMcY.

Seitzer, M. pytorch-fid: FID Score for PyTorch. https:// github.com/mseitzer/pytorch-fid, August

2020. Version 0.3.0.

Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https:// openreview.net/forum?id=St1giarCHLP.

Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum? id=PxTIG12RRHS.

Sun, S., Wei, L., Xing, J., Jia, J., and Tian, Q. Sddm: scoredecomposed diffusion models on manifolds for unpaired image-to-image translation. In International Conference on Machine Learning, pp. 33115–33134. PMLR, 2023.

Team Wan, Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang, X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li, Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.-F., and Liu, Z. Wan: Open and advanced large-scale video generative models, 2025. URL https:

//arxiv.org/abs/2503.20314.

Tweedie, M. C. K. An index which distinguishes between some important exponential families. In Ghosh, J. K. and Roy, J. (eds.), Statistics: Applications and New Directions: Proceedings of the Indian Statistical Institute Golden Jubilee International Conference, pp. 579–604, Calcutta, 1984. Indian Statistical Institute.

Wu, C., Li, J., Zhou, J., Lin, J., Gao, K., Yan, K., Yin, S., Bai, S., Xu, X., Chen, Y., Chen, Y., Tang, Z., Zhang, Z., Wang, Z., Yang, A., Yu, B., Cheng, C., Liu, D., Li, D., Zhang, H., Meng, H., Wei, H., Ni, J., Chen, K., Cao, K., Peng, L., Qu, L., Wu, M., Wang, P., Yu, S., Wen, T., Feng, W., Xu, X., Wang, Y., Zhang, Y., Zhu, Y., Wu, Y., Cai, Y., and Liu, Z. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36: 15903–15935, 2023.

Zhang, W., Liu, Y., Guan, S., Ran, W., Ge, Y., Li, W., and Ma, C. Guiding a diffusion model by swapping its tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14263–14272, June 2026.

Zhao, W., Bai, L., Rao, Y., Zhou, J., and Lu, J. UniPC: A unified predictor-corrector framework for fast sampling of diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https:

//openreview.net/forum?id=hrkmlPhp1u.

Zheng, C. and Lan, Y. Characteristic guidance: Nonlinear correction for diffusion model at large guidance scale. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/ forum?id=eOtjMYdGLt.

Vincent, P. A connection between score matching and denoising autoencoders. Neural Computation, 23(7):1661– 1674, 2011. doi: 10.1162/NECO a 00142.

Voleti, V., Yao, C.-H., Boss, M., Letts, A., Pankratz, D., Tochilkin, D., Laforte, C., Rombach, R., and Jampani, V. SV3D: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion. In European Conference on Computer Vision (ECCV), 2024.

von Platen, P., Patil, S., Lozhkov, A., Cuenca, P., Lambert, N., Rasul, K., Davaadorj, M., Nair, D., Paul, S., Berman, W., Xu, Y., Liu, S., and Wolf, T. Diffusers: State-of-the-art diffusion models. https://github.

com/huggingface/diffusers, 2022.

Wang, J., Chan, K. C., and Loy, C. C. Exploring clip for assessing the look and feel of images. Proceedings of the AAAI Conference on Artificial Intelligence, 37(2):2555–2563, Jun. 2023. doi: 10.1609/aaai.v37i2. 25353. URL https://ojs.aaai.org/index.

php/AAAI/article/view/25353.

### A. Proof & Derivation Proof for Theorem 4.1

Proof. Assume the deterministic base step ∆k+1 = α˜k ϵθ(xk+1,tk+1) + βk xk+1, with α˜k ≤ 0, and let PM

##### ,PM⊥

k+1

k+1

be the orthogonal projectors with PM

xk+1 = 0. Applying the projectors to the base decomposition gives

xk+1 = xk+1 and PM⊥

k+1

k+1

##### PM⊥

∆k+1 = α˜k PM⊥

ϵθ(xk+1,tk+1), (29) PM

k+1

k+1

ϵθ(xk+1,tk+1) + βk xk+1. (30) Therefore, the TAG update rule step is

∆k+1 = α˜k PM

k+1

k+1

+ η PM⊥

∆TAGk+1 = PM

k+1

k+1

∆k+1 = α˜k PM

k+1

+ ηPM⊥

k+1

ϵθ(xk+1,tk+1) + βkxk+1. (31)

The first-order Taylor gain with respect to TAG update at tk+1 is defined as:

G(η) := ∆TAGk+1 ⊤∇x log p(x | tk+1) x=x

k+1

⊤

+ η PM⊥

(32) We analyze this gain by approximating the true score with the model’s score function

∇x log p(x | tk+1) x=x

##### = PM

∆k+1

k+1

k+1

k+1

sθ(xk+1,tk+1) = −σk−+11 ϵθ(xk+1,tk+1), (33) thus:

⊤

+ η PM⊥

∇x log p(x | tk+1) x=x

G(η) = PM

∆k+1

k+1

k+1

k+1

⊤

+ η PM⊥

##### ≈ PM

∆k+1

sθ(xk+1,tk+1)

k+1

k+1

⊤

= −σk−+11 · PM

+ η PM⊥

ϵθ xk+1,tk+1 (34) Substitute Equation (31) into Equation (34), then:

∆k+1

k+1

k+1

⊤

G(η) ≈ −σk−+11 α ˜kPM

ϵθ + βkxk+1 + ηα˜kPM⊥

ϵθ

ϵθ

k+1

k+1

= −σk−+11 α ˜k(PM

ϵθ)⊤ϵθ . (35) Since P and P⊥ are symmetric and idempotent, thus

ϵθ)⊤ϵθ + βkx⊤k+1ϵθ + ηα˜k(PM⊥

k+1

k+1

v⊤Pv = ∥Pv∥22 (36) is established. Therefore,

ϵθ 22 + βkx⊤k+1ϵθ + ηα˜k PM⊥

ϵθ 22 . (37) Differentiating the gain G(η) in Equation (37) with respect to η yields:

G(η) ≈ −σk−+11 α ˜k PM

k+1

k+1

−α˜k σk+1

∂G(η) ∂η ≈

ϵθ(xk+1,tk+1) 22 ≥ 0. (38)

PM⊥

k+1

This derivative is guaranteed to be non-negative, since the DDIM sampler coefficient α˜k ≤ 0 by definition, while σk+1 and the squared L2-norm are strictly non-negative. This proves that the first-order gain G(η) is a monotonically non-decreasing function of η. Consequently, amplifying the tangential component of the update step via TAG is guaranteed to improve the first-order log-likelihood gain compared to the base update step.

##### Analysis on pure TAG gain. Subtracting each gain Gbase ≜ G(η = 1) and GTAG ≜ G(η > 1),

TAG update gain, GTAG

base update gain, Gbase

⊤

⊤

− σk−+11 · ∆TAGk+1

− σk−+11 · ∆k+1

ϵθ xk+1,tk+1 −

ϵθ xk+1,tk+1

= −σk−+11 · ∆TAGk+1 − ∆k+1 ⊤ϵθ xk+1,tk+1

∆k+1 ⊤ϵθ xk+1,tk+1 . (39)

= −σk−+11 · (η − 1)PM⊥

k+1

##### Using ∆k+1 = α˜k ϵθ(xk+1,tk+1) + βk xk+1, PM⊥

be:

k+1

##### PM⊥

∆k+1 = PM⊥

α˜k ϵθ(xk+1,tk+1). (40) Thus, substitute Equation (40) into Equation (39) then:

k+1

k+1

##### GTAG − Gbase = −σk−+11 · α ˜k(η − 1)

scalar

This simplifies to the final quadratic form:

· PM⊥

k+1

ϵθ(xk+1,tk+1) ⊤ϵθ xk+1,tk+1 . (41)

##### ϵθ(xk+1,tk+1) 22, (42)

##### GTAG − Gbase = −σk−+11 · α ˜k(η − 1)

· PM⊥

k+1

≥ 0 as α˜k ≤ 0

This proves that the difference in gain is non-negative for any η ≥ 1. Therefore, the first-order log-likelihood gain of the TAG update is always greater than or equal to that of the base update, with equality holding if and only if η = 1 or the tangential component of the score is zero.

| |
|---|

Table 13. Compatibility with Self-Swap Guidance (SSG). TAG is applied on top of SSG under matched evaluation settings on COCO

2014. TAG consistently improves SSG in both unconditional and conditional generation.

(a) Unconditional generation on SD-v1.5

(b) Conditional generation on SD-XL Cond. FID ↓ IR ↑ Pick ↑ CLIP ↑ CFG + SSG 32.82 0.449 22.40 25.11 TAGcfg + SSG 32.81 0.460 22.40 25.15

Uncond. FID ↓ IS ↑ AES ↑ CMMD ↓ SSG (Zhang et al., 2026) 49.02 18.63 5.186 0.654 TAG + SSG 48.53 19.05 5.187 0.648

### B. Broader Experiments

#### B.1. Application to Self-Swap Guidance (SSG, Section 5.1)

We further evaluate the compatibility of TAG with Self-Swap Guidance (SSG) (Zhang et al., 2026). SSG constructs a perturbed prediction by swapping token and channel positions in self-attention activations, whereas TAG is applied after the scheduler step by decomposing the resulting update into radial and tangential components. Therefore, TAG can be directly stacked on top of SSG without modifying the model architecture or requiring additional denoising evaluations.

In the unconditional setting, TAG improves SSG across all four metrics, reducing FID from 49.02 to 48.53 and CMMD from 0.654 to 0.648. In the conditional setting, TAGcfg also improves FID, ImageReward, and CLIPScore on top of SSG, while maintaining the same PickScore. These results support that TAG is complementary to SSG, as it refines the solver trajectory after SSG guidance rather than replacing the guidance signal itself. Qualitative comparisons are provided in Figure 14.

#### B.2. Unconditional Generation on ImageNet.

We further evaluate TAG on unconditional generation to verify that the proposed method generalizes beyond class-conditional and textconditional settings. Table 14 reports FID-50K on ImageNet 64×64 using EDM2 (Karras et al., 2024b) as the baseline. Applying TAG reduces FID from 11.04 to 10.37, confirming that TAG provides consistent gains even without explicit conditioning signals.

Table 14. Unconditional generation on ImageNet 64×64 (FID-50K). Method [ηsta, ηend] scale η FID↓

EDM2 – – 11.0432 TAG [750, 500] 1.3 10.3741

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

#### B.3. Beyond 2D Image: Image-to-3D shape generation

[Figure 102]

[Figure 103]

[Figure 104]

To check whether TAG can be applied beyond 2D image synthesis, we run a small qualitative test on image-to-3D generation with SV3D (Voleti et al., 2024). We apply TAG as a lightweight modification during SV3D sampling, without changing the model or adding extra evaluations. As shown in Figure 9, TAG yields modest qualitative improvements, with fewer view-dependent artifacts and slightly more consistent structure across viewpoints. A systematic quantitative study for 3D generation is left for future work.

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

###### + TAG

SV3D

GT

Figure 9. Qualitative comparison on SV3D (Voleti et al., 2024). Adding TAG yields modest improvements in cross-view consistency.

### C. Additional Qualitative Results

#### C.1. Image Generation: unconditional

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

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

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

- Stable Diffusion 1.5

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

- Stable Diffusion 2.1

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Stable Diffusion XL

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

- Stable Diffusion 3

- Figure 10. Qualitative results for unconditional generation across backbones. For each model (SD v1.5/2.1/XL (Rombach et al., 2022; Podell et al., 2024), and 3 (Esser et al., 2024)), the top row shows baseline sampling and the bottom row shows +TAG at matched NFEs. TAG yields sharper, more coherent structure with fewer artifacts while preserving diversity.

#### C.2. Video Generation (Section 5.4)

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

TAGBaseline

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

- (a) “A person and a hair drier.”

TAGBaseline

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

- (b) “A person is crawling a baby.”

TAGBaseline

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

- (c) “A tennis racket and a bottle.”

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

TAGBaseline

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

- (d) “Arch.”

TAGBaseline

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

- (e) “Carrousel.”

- Figure 11. Qualitative comparison between the Wan 2.2 (Team Wan et al., 2025) baseline and our TAG model on five prompts. For each prompt, the top row shows the baseline and the bottom row shows TAG (ours). Six frames are uniformly sampled across each 49-frame clip.

#### C.3. Image Generation: Plug-and-Play (for Section 5.1)

PAG (Ahn et al., 2024) PAG + TAGcfg PAG (Ahn et al., 2024) PAG + TAGcfg

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

A man in a suit is resting a bat on his shoulder. A group of skateboarders sitting outside on some concrete.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

A frog is sitting near a bunch of bananas. A young girl with a stuffed toy in a park.

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

A toothbrush holder with one toothbrush in it and two holes for holding other toothbrushes.

A view of a tower from across the street.

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

White dishware and a red table cloth lines a table. A woman standing with a bi-wing airplane on a runway.

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Many zebras graze in a dry field. A laptop with a picture of the earth on its screen while sitting on a surfboard.

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

A blue suitcase sits on the floor in front of 2 pair of shoes. A woman cutting up a large chocolate sheet cake.

- Figure 12. Qualitative comparison between PAG and PAG + TAGcfg. The highlighted words in each prompt indicate target concepts, attributes, or relations that are often weakly represented or missing in the baseline results. PAG + TAGcfg more faithfully reflects these prompt-specific details, such as adding missing objects and correcting actions, while preserving the overall scene structure.

SEG (Hong, 2024) SEG + TAGcfg SEG (Hong, 2024) SEG + TAGcfg

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

A man sitting on a large elephant that is standing in water. A yellow and white cat sitting next to a book on a bed.

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

The man and a young person are flying a kite on the beach. a ski boarder skiing down the side of a mountain.

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

Two cows are in an open field in front of a mountain range. A chick is sitting on a skateboard on a blue sheet.

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

A small child is holding onto lawn decorations. A girl flying a red kite high in the air.

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

... a family is still flying a kite on the beach. person walking on a beach carrying a camera bag

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

a bottle of beer on a wood table with a drink glass A sandwich with chips and drink from panera.

- Figure 13. Qualitative comparison between SEG and SEG + TAGcfg. The highlighted words in each prompt indicate target concepts, attributes, or relations that are often weakly represented or missing in the baseline results. SEG + TAGcfg more faithfully reflects these prompt-specific details, such as adding missing objects and correcting actions, while preserving the overall scene structure.

SSG (Zhang et al., 2026) SSG + TAGcfg SSG (Zhang et al., 2026) SSG + TAGcfg

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

A man holding a baseball bat in his hands. Woman listening on her phone while smoking a cigarette.

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Train traveling through countryside near tall brick structure. A hot dog sitting on top of a bun covered in toppings.

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

###### Two multi coloried cats asleep on a desk. A young girl playing with a play doll while sitting on ...

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

A man on a surfboard riding a wave in swimming trunks. The red clock is in the center of the street.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

A couple of people standing in a room with remotes. a teddy bear next to a picture of a woman

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

A couple walk on the sidewalk past a stop sign. a couple of people skiing down a hill

- Figure 14. Qualitative comparison between SSG and SSG + TAGcfg. The highlighted words in each prompt indicate target concepts, attributes, or relations that are often weakly represented or missing in the baseline results. SSG + TAGcfg more faithfully reflects these prompt-specific details, such as adding missing objects and correcting actions, while preserving the overall scene structure.

#### C.4. Image Generation: Geometric Guidance (for Section 5.3)

TCFG (Kwon et al., 2025) TAGcfg TCFG (Kwon et al., 2025) TAGcfg

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

A green bench sits directly under the Princeton Junction sign. A child eats pizza while on board a ship.

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Two men jumping at the same time for a Frisbee. Man with a hot dog in a paper rapper in his hand.

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

A birthday cake with many candles on it next to a knife. Man lighting candles on a white cake held by a woman.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

A teddy bear has some pizza boxes on it. A red, white and blue fire hydrant on a city sidewalk.

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

A red clock with gold face and a crown on top. A white cat sleeps in a blanketed basket.

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

shirtless man in red swim trunks rides surfboard A person with a remote and holding a dog

- Figure 15. Qualitative comparison between TCFG and TAGcfg. The highlighted words in each prompt indicate target concepts, attributes, or relations that are often weakly represented or missing in the baseline results. TAGcfg more faithfully reflects these prompt-specific details, such as adding missing objects and correcting actions, while preserving the overall scene structure.

APG (Sadat et al., 2025) TAGcfg APG (Sadat et al., 2025) TAGcfg

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

a hot dog with ketchup and mustard and a drink A child eats pizza while on board a ship.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

A couple walk on the sidewalk past a stop sign. A vintage truck rolls down the road ... and a CVS Pharmacy.

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

A horse in a green meadow ... a cloudy grey sky ... A decorative vase placed near oriental wall scrolls.

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

A tabby cat and a black and white cat looking for trouble. A woman in black wetsuit surfing on wave river.

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

A woman smiles brightly while riding a huge elephant. The side of a large black train with a red stripe ...

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

A photo of someone’s sandwich with cheese and bacon. A person in a rain coat is feeding the meter

- Figure 16. Qualitative comparison between APG and TAGcfg. The highlighted words in each prompt indicate target concepts, attributes, or relations that are often weakly represented or missing in the baseline results. TAGcfg more faithfully reflects these prompt-specific details, such as adding missing objects and correcting actions, while preserving the overall scene structure.

### D. Implementation of the Tangential Amplifying Guidance (Section 4.2)

- Algorithm 2 Code: Tangential Amplifying Guidance (TAG)

output = scheduler.step(noise_pred, t, latents, return_dict=False) if apply_tag:

post = latents eta_v, eta_n = t_guidance_scale, 1

v_t = post / (post.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8) latents = output delta = latents - post a = (delta * v_t).sum(dim=(1,2,3), keepdim=True) u_n = a * v_t u_t = delta - u_n latents = post + eta_v * u_t + eta_n * u_n

else:

latents = output

- Algorithm 3 Code: Conditional Tangential Amplifying Guidance (TAGcfg)

def proj_par(z, n): return (z * n).sum(dim=(1,2,3), keepdim=True) * n

def proj(z, v): v = v / (v.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8) return (z * v).sum(dim=(1,2,3), keepdim=True) * v

eps_u, eps_c = HeadToEps(noise_pred, latents, t, scheduler, do_cfg)

- s_u = -eps_u / (sigma + 1e-12)

- s_c = -eps_c / (sigma + 1e-12) n = latents / (latents.norm(p=2, dim=(1,2,3), keepdim=True) + 1e-8) g = s_c - s_u
- t_c = s_c - proj_par(s_c, n)

- t_u = s_u - proj_par(s_u, n) g_aligned = proj(s_c, t_c - t_u)

s_star = s_u + (guidance_scale * g) + (t_guidance_scale * g_aligned) eps = -sigma * s_star

model_out = EpsToHead(eps, latents, t, scheduler) latents = scheduler.step(model_out, t, latents, return_dict=False)

### E. Discussion on Orthogonal and Projection-based Guidance (Section 2)

Several recent works incorporate projections or tangential/orthogonal notions in diffusion guidance. While they share surface-level similarity, they differ in (i) which quantity is decomposed, (ii) how the relevant subspace is defined, and (iii) the problem setting and objective that motivate the projection. We summarize these distinctions below to situate TAG.

#### E.1. Orthogonal and Projection-based Diffusion Guidance

- E.1.1. CFG-SPECIFIC PROJECTIONS FOR ARTIFACT SUPPRESSION

Projected CFG for oversaturation. Sadat et al. (2025) analyze artifacts such as oversaturation at large CFG (Ho & Salimans, 2021) scales. They decompose the CFG update into components parallel and orthogonal to the conditional score, and empirically identify the parallel component as the primary cause of saturation.

Their Adaptive Projected Guidance (APG) therefore attenuates the parallel component while preserving the orthogonal component, and complements this with rescaling and momentum-style heuristics motivated by a gradient-ascent interpretation of CFG. This line of work is (i) tied to the CFG algebra and targets (ii) large-scale saturation artifacts.

SVD-based tangential damping within CFG. Kwon et al. (2025) address conditional-unconditional misalignment in CFG, which can lead to off-manifold samples. They form a score matrix from conditional and unconditional scores and perform SVD, interpreting high singular-value directions as shared, approximately manifold-normal components and low singular-value directions as tangential components.

They observe that the discrepancy between conditional and unconditional scores is concentrated in low-SV (i.e., tangential) directions, and propose filtering the unconditional score by projecting it onto the shared high-SV subspace before applying CFG. This approach modifies only the unconditional term inside CFG rather than the base solver update, and does not study the semantic role of tangential components along the solver trajectory.

- E.1.2. CORRECTION OF CFG DYNAMICS

Nonlinear correction via characteristics. Zheng & Lan (2024) derive guidance from the nonlinear Fokker-Planck dynamics of guided diffusion. They show that standard linear CFG neglects nonlinear correction terms that become important at high guidance scales, causing artifacts.

Their method solves a nonlinear fixed-point / characteristic equation for the corrected guided score. A projection operator can be inserted for numerical regularization, but it is not the conceptual driver of the method; the theoretically ideal operator is the identity. Thus, improvements come from nonlinear correction rather than from projecting tangential/orthogonal components.

- E.1.3. PROMPT INTERACTION AND SEMANTIC DISENTANGLEMENT

Orthogonalizing negative prompts. Armandpour et al. (2023) study failures of negative prompting when negative and positive concepts overlap. They decompose the negative score into components parallel and perpendicular to the positive score direction, discard the parallel (overlapping) negative component, and apply only the perpendicular negative guidance.

The goal is to prevent negative prompts from canceling shared desirable attributes, especially in text-to-image and text-to-3D pipelines. This mechanism concerns prompt interaction effects rather than diffusion-step geometry.

- E.1.4. TASK-SPECIFIC MANIFOLD CONSTRAINTS

Tangent restriction for loss-based guidance. He et al. (2024) focus on training-free, loss-based guidance (e.g., DPS (Chung et al., 2023)-style inverse problems), noting that ambient-space loss gradients may violate data-manifold constraints.

They apply guidance on x0|t and project external guidance gradients onto tangent spaces of the clean-data manifold M0, estimating tangents using an auxiliary pretrained autoencoder. These methods rely on an explicit manifold model and are tailored to observation-conditioned tasks rather than generic unconditional or standard conditional generation.

Refinement–transport decomposition in conditional I2I. In unpaired conditional image-to-image translation, Sun et al. (2023) construct reference-dependent manifolds Mt(y0) induced by a reference image y0. Such a conditional structure is crucial: the authors emphasize (§4, Lemma 1 of (Sun et al., 2023)) that in standard diffusion, where intermediate manifolds

- Table 15. Comparison of projection / tangential guidance methods. Prior works decompose CFG algebra, prompt gradients, or task-specific manifold scores; TAG uniquely decomposes and amplifies the single-step solver update with respect to iso-noise manifolds, without auxiliary models.

Work Quantity Subspace / manifold Projection / filtering Goal Cost APG (Sadat et al., 2025) CFG update ∥ / ⊥ to conditional score

Downweight the ∥ component in CFG

Mitigate oversaturation under high CFG

CFG

Project unconditional score to the high-SV subspace before CFG

Reduce conditional–unconditional mismatch in T2I

TCFG (Kwon et al., 2025)

Conditional / unconditional scores

SVD shared high-SV vs. tangential low-SV subspaces

CFG

Characteristic Guidance (Zheng & Lan, 2024)

Optional stabilizing projection P; ideal case P = I

Nonlinear correction of CFG

Guided-score fixed point None (projection optional)

CFG

Perp-Neg (Armandpour et al., 2023)

Remove the ∥ negative component

Avoid negative / positive prompt overlap

Negative-prompt score ∥ / ⊥ to positive prompt

Multi-prompt

Loss gradient on x0|t / latent

Tangent space of AE-defined M0

Project loss gradient to the tangent space

On-manifold inverse problems

MPGD (He et al., 2024)

AE extra

I2I conditional score / energy

Normal transport with tangential refinement

Unpaired image-to-image translation

SDDM (Sun et al., 2023)

Reference-induced Mt(y0)

Task mods

TAG (ours) Solver step ∆xk ∥ / ⊥ to iso-noise manifold

Amplify the tangential component of ∆xk

Generic sampling refinement

Simple vector calc.

Mk

at adjacent timesteps are coupled, a tangential/normal score split is generally meaningless.

In contrast, Sun et al. (2023) argue that the I2I setting yields compact, well-separated manifolds across time, making the decomposition meaningful; under this specific condition, the tangential component acts as an on-manifold refinement term, while the normal component governs transport between manifolds of adjacent timesteps.

#### E.2. TAG: Intrinsic Tangential Amplification for General Diffusion Sampling

TAG leverages the intrinsic geometry of diffusion trajectories by decomposing the single-step solver update at each state xk into radial (manifold-normal) and tangential components with respect to the iso-noise manifold Mk. We show that this split is stable and meaningful in general diffusion:

- • the tangential update captures semantic refinement along Mk
- • while the radial part is largely unstructured,

and amplifying the tangential component provably promotes local log-likelihood ascent. In contrast to CFG-algebraic projections (Sadat et al., 2025; Kwon et al., 2025), nonlinear CFG corrections (Zheng & Lan, 2024), prompt-specific orthogonalization (Armandpour et al., 2023), or task-/condition-dependent manifold models (Sun et al., 2023; He et al., 2024), TAG operates directly on the base solver update, requires no auxiliary models or extra network passes, and applies uniformly to unconditional and standard conditional generation across modern backbones.

### F. Experimental Details (for Section 5)

#### F.1. Implementation Details

All experiments are implemented in PyTorch 2.5.1 with CUDA 12.1, using the diffusers 0.35.1 library (von Platen et al., 2022). Each pipeline subclasses the corresponding diffusers pipeline and overrides the call method to inject the TAG decomposition logic after the scheduler step. All experiments are conducted on NVIDIA GeForce RTX 4090 GPUs (24GB VRAM each).

#### F.2. Unconditional Generation

We evaluate TAG on four Stable Diffusion variants (SD1.5/2.1 (Rombach et al., 2022), SDXL (Podell et al., 2024), and SD3 (Esser et al., 2024)).

For all unconditional experiments, we use an empty prompt (prompt="") with classifier-free guidance disabled (ω = 0). We generate 30,000 images per setting using integer seeds from 0 to 29,999. The number of denoising steps is fixed at T = 50 for all models. TAG is applied with the radial scaling factor ηr = 1.0 (unchanged) across all experiments; only the tangential scaling factor ηv varies. The TAG application range covers all timesteps (tstart = 1000, tend = 0).

Table 16. TAG hyperparameters for unconditional generation (Table 1 in the main paper). All settings use ηr = 1.0, tstart = 1000, tend = 0, and T = 50 steps.

Model ηv ηr ω tstart tend Steps

SD v1.5 (TAG off) 1.0 0.0 – – 50 TAGSDv1.5 1.15 1.0 0.0 1000 0 50 SD v2.1 (TAG off) 1.0 0.0 – – 50 TAGSDv2.1 1.15 1.0 0.0 1000 0 50 SDXL (TAG off) 1.0 0.0 – – 50 TAGSDXL 1.20 1.0 0.0 1000 0 50 SD3 (TAG off) 1.0 0.0 – – 50 TAGSD3 1.05 1.0 0.0 1000 0 50

Table 16 lists the tangential scaling factor ηv (t lr) used for each model. These values were selected based on preliminary sweeps.

#### F.3. TAG for Classifier-Free Guidance

For conditional (text-to-image) experiments, we use 10,000 captions randomly sampled from the MS-COCO 2014 (Lin et al., 2014) validation set as text prompts. All images are generated with a fixed seed of 0 and CFG scale ω.

#### F.4. Compatibility with Guidance Methods (PAG, SEG, and SSG)

PAG + TAG. Perturbed Attention Guidance (PAG) (Ahn et al., 2024) replaces the selfattention map at designated UNet layers with an identity matrix to create a perturbed prediction, then guides generation away from the perturbed output. We combine PAG with TAG by first applying PAG guidance at the noise-prediction level, then applying the TAG radial–tangential decomposition after the scheduler step. Following the official PAG recommendation, we use pag scale= 3.0 applied to the mid-block (m0) of the UNet.

Table 17. PAG / SEG / SSG compatibility experiment settings (unconditional SD v1.5). SSG uses swap ratio=0.001 applied to all 16 attention layers (d0..d5,m0,u0..u8).

Setting Guidance Scale Layer ηv ηr tstart tend N Baseline – – – – – – – 30K ▶ TAG – – – 1.15 1.0 1000 0 30K ▶ PAG only PAG 3.0 mid (m0) – – – – 30K ▶ SEG only SEG 3.0 mid block – – – – 30K ▶ SSG only SSG 1.0 all attn – – – – 30K Plug-and-Play

▶ PAG + TAG PAG 3.0 mid (m0) 1.15 1.0 750 450 30K ▶ SEG + TAG SEG 3.0 mid block 1.15 1.0 750 450 30K ▶ SSG + TAG SSG 1.0 all attn 1.08 1.0 1000 750 30K

SEG + TAG. Smoothed Energy Guidance (SEG) (Hong, 2024) applies Gaussian blur to the self-attention query projections to create a structurally degraded prediction. We combine SEG with TAG analogously: SEG guidance is applied at the noise-prediction level (seg scale= 3.0, blur sigma= 100.0, mid-block only), followed by TAG decomposition after the scheduler step.

SSG + TAG. Self-Swap Guidance (SSG) (Zhang et al., 2026) stochastically swaps a small fraction of token and channel positions in the self-attention activations to create a structurally perturbed prediction. We combine SSG with TAG analogously: SSG guidance is applied at the noise-prediction level with ssg scale= 1.0 and swap ratio= 0.001 across all 16 UNet self-attention layers (d0..d5, m0, u0..u8), followed by TAG decomposition after the scheduler step.

- Table 17 summarizes the configurations for the PAG, SEG and SSG compatibility experiments. All experiments use unconditional SD v1.5 generation with 30,000 samples and the PNDM scheduler.

#### F.5. Evaluation Metrics

Fr´echet Inception Distance (FID) and Inception Score (IS). We compute FID and IS using torch-fidelity 0.4.0 (Obukhov et al., 2020) with the inception-v3-compat feature extractor (feature layer 2048 for FID, logits unbiased for IS). All generated images are resized to 299 × 299 before feature extraction. For unconditional experiments, we use the full MS-COCO 2014 validation set (40,504 images) as the reference distribution. For conditional experiments, we use a 10,000-image subset of MS-COCO 2014 validation images aligned with the prompt set. Reference statistics are precomputed and cached as .npz files.

Aesthetic Score (AES). We compute the LAION aesthetic score using the ImageReward (Xu et al., 2023) library (RM.load score("Aesthetic")), which employs a linear probe trained on top of CLIP ViT-L/14 features with the SAC+LOGOS+AVA dataset. Scores are computed per-image and averaged over the full set.

CMMD. We compute the CLIP Maximum Mean Discrepancy (CMMD) (Jayasumana et al., 2024) using the clip-mmd 0.0.2 library, which extracts CLIP features from both generated and reference image sets and computes an MMD-based distributional distance. The reference set is the MS-COCO 2014 validation set (40,504 images).

No-Reference Image Quality Assessment (NR-IQA). We additionally report two NR-IQA metrics computed using the pyiqa 0.1.15 library (Chen & Mo, 2022):

- • CLIP-IQA (Wang et al., 2023): CLIP-based perceptual quality score (higher is better).
- • MUSIQ (Ke et al., 2021): Multi-scale image quality transformer trained on KonIQ-10k (higher is better).

CLIPScore. For conditional experiments, we compute CLIPScore using the ImageReward CLIP model (RM.load score("CLIP")), which wraps CLIP ViT-L/14. The raw cosine similarity between image and text embeddings is scaled by 100.

ImageReward. We use the ImageReward-v1.0 model (RM.load("ImageReward-v1.0")) (Xu et al., 2023), which is a BLIP-based reward model fine-tuned on human preference data. For conditional experiments, scores are computed per prompt–image pair and averaged.

#### F.6. Software and Hardware

All generation and evaluation experiments are conducted on NVIDIA GeForce RTX 4090 GPUs (24GB VRAM). Multiple GPUs are used in parallel for generation (data-parallel across seeds) but each individual image is generated on a single GPU without model parallelism.

Package Version PyTorch (Paszke et al., 2019) 2.5.1 (CUDA 12.1) Diffusers (von Platen et al., 2022) 0.35.1 Transformers 4.46.3 Accelerate 1.5.2 torch-fidelity (Obukhov et al., 2020) 0.4.0 pyiqa (Chen & Mo, 2022) 0.1.15 clip-mmd 0.0.2 ImageReward (Xu et al., 2023) 1.5

