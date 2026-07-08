# arXiv:2511.23342v3[cs.CV]29Mar2026

## Overcoming the Curvature Bottleneck in MeanFlow

Xinxi Zhang⋆, Shiwei Tan⋆, Quang Nguyen, Quan Dao, Ligong Han, Xiaoxiao He, Tunyu Zhang, Chengzhi Mao, Dimitris Metaxas, and Vladimir Pavlovic

Rutgers University, Piscataway NJ 08854, USA

Abstract. MeanFlow offers a promising framework for one-step generative modeling by directly learning a mean-velocity field, bypassing expensive numerical integration. However, we find that the highly curved generative trajectories of existing models induce a noisy loss landscape, severely bottlenecking convergence and model quality. We leverage a fundamental geometric principle to overcome this: mean-velocity estimation is drastically simpler along straight paths. Building on this insight, we propose Rectified MeanFlow, a self-distillation approach that learns the mean-velocity field over a straightened velocity field, induced by rectified couplings from a pretrained model. To further promote linearity, we introduce a distance-based truncation heuristic that prunes residual high-curvature pairs. By smoothing the optimization landscape, our method achieves strong one-step generation performance. We improve the FID of baseline MeanFlow models from 30.9 to 8.6 under same training budget, and outperform the recent 2-rectified flow++ by 33.4% in FID while running 26x faster. Our work suggests that the difficulty of one-step flow generation stems partially from the rugged optimization landscapes induced by curved trajectories.

Project Page: https://xinxi-zhang.github.io/WEB_REMF/ Code: https://github.com/Xinxi-Zhang/Re-MeanFlow

Keywords: Image Generation · Efficient Training

### 1 Introduction

Flow models [29,31] and diffusion models [45,49] have become a central paradigm in generative modeling, enabling a wide range of applications across various data domains [13,14,18,41,53,59]. Compared with earlier paradigms such as GANs [12, 23] and Normalizing Flows [40, 57], these models offer stable training and superior fidelity, but at the cost of expensive sampling. The root cause of this inefficiency is the curvature of the generative trajectories induced by the mismatch between the prior and data distributions. In practice, the resulting velocity field can bend sharply, making it difficult to approximate accurately with only a few discretization steps.

⋆ Equal contribution

[Figure 1]

###### 2 X. Zhang et al.

[Figure 2]

[Figure 3]

[Figure 4]

loss

loss

PCA2 PCA1

PCA2 PCA1

###### (a) MeanFlow Loss Landscape (b) Re-MeanFlow (Ours)

(c) Comparison of Convergence (Both initialized from the pretrained EDM2-S)

Loss Landscape

- Fig. 1: Straightening flows smooths the loss landscape for one-step generation. (a) MeanFlow exhibits a sharply peaked and irregular loss landscape, making optimization difficult for learning efficient one-step generators. (b) By learning the mean velocity on a rectified coupling with substantially straighter trajectories, Re-MeanFlow (Ours) yields a much smoother and more regular objective. (c) This improved landscape empirically leads to faster convergence and stronger one-step generation, even when MeanFlow is trained for 2× longer. All plots are from ImageNet 2562, with both MeanFlow and Re-MeanFlow initialized from the same SiT-XL model.

This drawback has motivated works [30,31,36,51] to straighten the underlying trajectories via optimal transport; however, existing methods do not produce sufficiently linear paths, which in turn prevents reliable one-step sampling based on the instantaneous velocity. Another line of work [9, 10, 24, 48, 61] bypasses ODE integration by directly learning time-indexed flow-map predictions. Among these approaches, MeanFlow [10] is particularly appealing: it models the timedependent mean-velocity field, enabling single-step generation without requiring straight trajectories, and achieves strong empirical performance. Despite this advantage, training MeanFlow remains costly and often converges slowly, even when initialized from a pretrained flow model.

In this work, we identify a key bottleneck in mean-velocity learning: the curvature of the underlying generative trajectories. Motivated by this insight, we introduce Rectified MeanFlow (Re-MeanFlow), a lightweight self-distillation approach that trains MeanFlow on rectified couplings, yielding markedly straighter trajectories and a simpler mean-velocity learning problem. Fig. 1 highlights this contrast: optimizing MeanFlow on the original (unrectified) trajectories produces a highly rugged landscape with sharp spikes, while Re-MeanFlow yields a noticeably smoother and better-conditioned surface. This improved conditioning leads to substantially faster convergence and superior one-step generation, even compared against MeanFlow trained with 2× more compute. Importantly, Re-MeanFlow is data-free: it requires only a pretrained flow model and samples from the prior to generate rectified couplings, without access to the original training dataset.

Additionally, we employ a simple distance-based truncation heuristic to further reduce curvature during training. Motivated by the empirical correlation between trajectory curvature and endpoint distance, we discard the top 10%

of couplings ranked by their ℓ2 distance between endpoints. This filter removes residual high-curvature trajectories and consistently improves both training stability and sample quality in our experiments.

Extensive experiments on ImageNet at 642, 2562, and 5122 show that ReMeanFlow consistently outperforms state-of-the-art distillation methods and strong train-from-scratch baselines in both generation quality and training efficiency. In particular, relative to the closely related 2-rectified flow++ [25], ReMeanFlow reduces FID by 33.4% while using only 10% of the compute.

Beyond empirical gains, Re-MeanFlow suggests a more practical training paradigm for few-step generative models. Existing distillation pipelines rely heavily on high-end training GPUs, making hyperparameter tuning and repeated runs prohibitively expensive. In contrast, Re-MeanFlow shifts most computation to an inference-driven reflow stage that can be executed on widely available consumer- or inference-grade accelerators, followed by a lightweight MeanFlow training phase. Overall, the training cost in Re-MeanFlow accounts for only 17% of the total GPU hours used by AYF [42].

### 2 Related Work

We review diffusion and flow matching, emphasizing methods that reduce sampling cost via trajectory straightening or few-step modeling, and summarize recent advances in efficient training.

#### 2.1 Diffusion and Flow Matching.

Diffusion/flow-based generative models [1,16,29,31,45,49,50] learn to reverse a gradual noising process, where the reverse-time dynamics can be formulated as a deterministic probability-flow ODE [20,50]. Although these approaches achieve strong performance, they typically require multi-step numerical integration for sampling [20,33,46,58] due to the high curvature of generative paths.

#### 2.2 Straightening the Generative Trajectories.

The high curvature of diffusion/flow generative trajectories has motivated work that explicitly reduces the transport curvature between noise and data. OTFlow [36] regularizes continuous normalizing flows with OT-inspired objectives to encourage straighter dynamics. OT-CFM [51] and Multisample Flow Matching [39] further reduce curvature by introducing OT-like minibatch couplings, yielding simpler probability paths and lower-variance supervision. However, such minibatch couplings are inherently local and do not directly enforce globally straighter trajectories. In contrast, Rectified Flow [31] achieves global trajectory straightening via reflow, iteratively refining the transport to reduce transport cost and producing near-linear paths that admit accurate few-step simulation. We therefore adopt rectified trajectories as a clean, low-curvature target for MeanFlow training, substantially easing one-step optimization.

#### 2.3 Few-step Diffusion/Flow Models.

Few-step models aim to bypass numerical integration by directly learning large time-step transitions. Consistency Models [11, 32, 47, 48, 54] train networks to produce invariant outputs across different timesteps, enabling direct few-step or even one-step sampling. Flow Map Models [5,9,10] bypass ODE integration by learning the displacement or velocity map over time. Despite their strong empirical results, these few-step paradigms often face stability challenges or high training cost because they must learn mappings along inherently curved trajectories, where supervision is noisy, and optimization is difficult. Recent work has sought to mitigate this issue by simplifying the consistency objective [32] or introducing improved loss functions and normalization strategies [6]. Concurrent work CMT [19] improves the efficiency of few-step models by supervising them with teacher ODE trajectories. Building on these insights, this work aims to combine trajectory simplification with one-step modeling.

#### 2.4 Efficient Training for Diffusion/Flow Models.

Many recent works [27,52,55,56] boost the training efficiency of diffusion models by leveraging external representation learners such as DINO [37]. Orthogonal works [43,60] adopt masked-transformer designs that exploit spatial redundancy by training only on a subset of tokens. ECT [11] enables efficient consistency models through a progressive training strategy that transitions from diffusion to consistency training. Re-MeanFlow contributes in this direction by allowing efficient one-step modeling on a significantly simplified flow path.

### 3 Efficient Mean-Velocity Modeling on Straightened Trajectory

In this section, we first review one-step flow-based generation via MeanFlow

- (Sec. 3.1). We then identify the curvature bottleneck: the standard independent coupling induces highly curved paths that hinder mean-velocity learning
- (Sec. 3.2). Finally, we introduce Re-MeanFlow, which models mean-velocity on rectified, substantially straighter trajectories, and describe our practical training pipeline, including CFG and distance-based truncation (Sec. 3.4).
- 3.1 One-Step Flow-Based Generative Modeling via MeanFlow

Flow-based generative models transport a prior pz (taken as N(0,I) in this work) to the data distribution px by learning a time-dependent velocity field v : [0,1] × Rd → Rd that induces a path of intermediate densities {pt}t∈[0,1] in Rd. Concretely, given a data-noise coupling pxz, let (x,z) ∼ pxz and define the linear interpolation as:

zt = (1 − t)x + tz.

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

x

x

u(zt,r,t)

Curv(r,t) u(zt,r,t) Curv(r,t)

z

z

(a) MeanFlow Models Mean-Velocity on High-Curvature Trajectories

(b) Re-MeanFlow (Ours) Models MeanVelocity on Rectified (Straightened) Trajectories

- Fig. 2: Method. (a) MeanFlow learns the mean-velocity u(zt, r, t) on highly curved trajectories, which makes optimization difficult. In contrast, (b) Re-MeanFlow (ours) models u(zt, r, t) on a rectified coupling that induces substantially straighter generative trajectories, simplifying the underlying vector field and enabling faster and more stable training. Curv(r, t) denotes the curvature proxy measuring trajectory curvature between time points r and t (see Sec. 3.3).

Let pt denote the distribution of zt. The coupling pxz together with the interpolation, therefore, specifies the probability path {pt}t∈[0,1].

The associated velocity field v(zt,t) is defined as the vector field whose continuity equation

∂tpt + ∇ · (ptv(·,t)) = 0 matches this density evolution. Along the interpolation trajectory, the instantaneous velocity is given by

d dt

zt = z − x.

Flow Matching (FM) [29,31] models the time-dependent velocity field with a neural vector field vθ(·,t) and trains it by regressing the trajectory velocity induced by the chosen coupling:

LFM(θ) = E(x,z)∼p

xz, t∼p(t)

d dt

zt − vθ(zt,t)

2

2

(1)

Once vθ is learned, a new sample x can be generated by solving the ODE for z ∼ pz using a numerical solver:

1

vθ(zτ,τ)dτ. (2) MeanFlow [10] instead parameterizes the mean-velocity between two time

xθ(z) = z0 = z −

0

points r < t:

t

1 t − r

u(zt,r,t) ≜

v(zτ,τ)dτ, (3)

r

To train a neural network uθ(zt,r,t) to approximate this mean-velocity, MeanFlow derives an implicit training target that connects the mean-velocity, the instantaneous velocity v(zt,t), and the time derivative of uθ:

xz,(r,t)∼pr,t ∥uθ(zt,r,t) − sg(utgt)∥22 (4)

LMF(θ) = E(x,z)∼p

d dt

with utgt = v(zt,t) − (t − r)

uθ(zt,r,t). (5)

Here, sg(·) denotes the stop-gradient operator. Once uθ is trained, sampling reduces to a single evaluation uθ(z,0,1), avoiding the numerical integration required by Flow Matching (cf. Eq. 2).

#### 3.2 Taming the Curvature Bottleneck

Crucially, the underlying generative trajectories of MeanFlow are dictated by the data-noise coupling distribution pxz. In the absence of prior knowledge about how data and noise should be paired, MeanFlow (and many flow-based methods) adopts the independent coupling distribution p0xz(x,z) = px(x)pz(z), which is known to induce highly curved generative trajectories (Fig. 2a).

We argue that learning the mean-velocity field on such curved trajectories is challenging: curvature amplifies the complexity of the target mean-velocity, which induces a rugged and poorly conditioned loss landscape (visualized in Fig. 1a and Fig. 7 (top)), amplifying the effect of imperfect supervision and slowing down optimization (see Sec. 4.3 for further analysis). To address this bottleneck, we propose Rectified MeanFlow (Re-MeanFlow), which models the mean-velocity field on a rectified coupling that induces substantially straighter trajectories (Fig. 2b). This rectification simplifies the underlying vector field and, correspondingly, yields a markedly smoother and more regular objective (Fig. 1b and Fig. 7 (bottom)), enabling faster convergence and improved one-step generation.

To construct this coupling that induces straighter trajectories, we follow Rectified Flow [31] and perform a single reflow step using a pretrained flow model vϕ trained under the independent coupling. Concretely, we sample z ∼ pz and define x = z − 0 1 vϕ(zτ,τ)dτ, which induces a new coupling distribution p1xz(x,z). We then train Re-MeanFlow by modeling the mean-velocity field on p1xz.

#### 3.3 Estimating the Curvature of the Generative Trajectory

To quantify the trajectory curvature over time pairs (r,t) , we propose an anglebased proxy. Let u˜(zt,r,t) denote the mean-velocity over the interval [r,t]. In practice, we approximate this quantity by integrating a learned velocity field vθ trained under the corresponding coupling:

t

1 t − r

vθ(zτ,τ)dτ, (6)

u˜(zt,r,t) =

r

and let vθ(zt,t) denote the instantaneous velocity at (zt,t). We then define

t∼pt ∠ u ˜(zt,r,t), vθ(zt,t) , (7)

Curv(r,t) = Ez

Algorithm 1 Re-MeanFlow

- 1: Input: pretrained rectified flow vϕ; prior pz; ODE solver SamplePair; time distribution pr,t; guidance strength ω; #pairs N; truncation ratio k%; training iterations T; learning rate η.
- 2: Output: trained Re-MeanFlow model uθ.
- 3: Initialize θ ← InitFrom(ϕ) ▷ Initialize uθ from teacher vϕ

- Stage A: Sample rectified couplings

4: P ← {(xi, zi)}Ni=1 where (x, z) ∼ SamplePair(vϕ, pz) 5: Compute distances di ← ∥xi − zi∥2 for all pairs 6: q ← Percentile(100−k)({di}Ni=1) 7: Pkeep ← {(xi, zi) ∈ P : di ≤ q} ▷ distance-based truncation

- Stage B: Train Re-MeanFlow on Pkeep

8: for s = 1, . . . , T do 9: Sample (x, z) ∼ Pkeep, (r, t) ∼ pr,t

10: zt ← (1 − t)x + tz 11: utgt ← (z − x) − (t − r) dtd uθ(zt, r, t) 12: LMF(θ) ← uθ(zt, r, t) − sg(utgt) 22 13: θ ← θ − η∇θLMF(θ) 14: end for

- Stage C: CFG fine-tuning

- 15: Fine-tune uθ on vωcfg (Eq. 8) using the loop in Stage B.
- 16: Return uθ

where ∠(a,b) = arccos ∥a ⟨a∥ ∥,bb⟩∥ . Intuitively, Curv(r,t) measures directional disagreement between (i) the average transport direction over the interval [r,t], given by u˜(zt,r,t), and (ii) the instantaneous flow direction at time t, given by vθ(zt,t). This proxy serves as a practical diagnostic similar to the ℓ2-based curvature measures used in Rectified Flow [31]; in particular, for perfectly straight trajectories, we have Curv(r,t) = 0 for all valid (r,t). We visualize Curv(r,t) in the ImageNet-5122 over the (r,t)-plane in Fig. 2 and show that the rectified couplings used by Re-MeanFlow exhibit substantially lower trajectory curvature. More results at other resolutions are provided in Appendix C.4.

#### 3.4 Training

Following prior work [25,31], we first construct the rectified coupling distribution p1xz(x,z) by generating N data-noise pairs with a pretrained flow model vϕ, to ensure sufficient coverage of both the data and noise distributions.

Classifier-free guidance. To enable classifier-free guidance (CFG) at inference, Re-MeanFlow (as in MeanFlow) must be trained to predict the CFG meanvelocity field vωcfg:

vωcfg(zt,t | c) ≜ ω v(zt,t | c) + (1 − ω)v(zt,t), (8)

where ω denotes the guidance strength and c is the conditioning signal. In practice, we find it more stable to apply guidance via a two-stage procedure: we first

| |[Figure 9]|
|---|---|
| | |

Curv(0,1)

∥x − z∥2 ∥x − z∥2, (x,z) ∼ px1,z

- Fig. 3: Left: Distance statistics of the data-noise pair distance ∥x − z∥2 under the independent coupling p0xz (red) and the rectified coupling p1xz (green). Right: Histogram of ∥x − z∥2 for rectified pairs (x, z) ∼ p1xz, with bins colored by the curvature proxy (Sec. 3.3). Overall, rectification reduces the average pair distance between data and noise, but a small long-distance tail remains; these long-distance couplings tend to exhibit higher curvature, motivating our distance-based truncation to remove many high-curvature pairs and further improve training stability and performance.

train without guidance to initialize the model, and then perform a brief finetuning stage using the CFG objective. Overall, Re-MeanFlow converges rapidly, typically within a single pass over the generated couplings.

Distance-based truncation. Rectified Flow [31] guarantees that rectification does not increase transport cost: for any convex cost c : Rd→R,

Ep1

xz

[c(x,z)] ≤ Ep0

xz

[c(x,z)]. (9)

We empirically verify this with c(x,z) = ∥x − z∥2 and plot the resulting distance distributions in Fig. 3 (left). As expected, the rectified coupling p1xz is left-shifted relative to the original coupling p0xz, indicating smaller transport distances. Notably, however, p1xz still exhibits a non-negligible long-tail subset of rectified pairs (x,z) ∼ p1xz have unusually large ∥x − z∥2 (above the dashed 90th-percentile threshold). We find that these long-distance pairs are also systematically more curved under our curvature proxy (Sec. 3.3), as shown in Fig. 3 (right). Motivated by this correlation, we introduce a simple distance-based truncation heuristic: we discard rectified couplings whose ∥x − z∥2 exceeds a fixed percentile threshold. In practice, we found that truncating the top 10% largestdistance couplings works robustly across all settings.

Algorithm. The full algorithm of Re-MeanFlow is provided in Alg.1.

- 4 Experiments

##### Datasets. We conduct experiments on ImageNet [7] at 642 resolution in pixel space and at 2562 and 5122 resolutions in latent space [41].

- Table 1: Class-conditional generation on ImageNet. All results use classifier-free guidance (CFG) when supported; “×2” indicates that CFG doubles the effective NFE. † denotes methods initialized from, or directly comparable to, the diffusion backbones also marked with †. (iCT [47] result at 2562 is reported by IMM [61], and ECT/ECD [11] results at 5122 are reported by CMT [19]). Re-MeanFlow (ours) achieves the best FID across all settings, outperforming prior state-of-the-art one-step distillation and training-from-scratch methods.

(a) ImageNet 642

(b) ImageNet 2562

(c) ImageNet 5122

Method NFE FID Diffusion models EDM2-S [22]† 63 × 2 1.58

Method NFE FID Diffusion models

Method NFE FID Diffusion models EDM2-S [22]† 63 × 2 2.23

ADM [8] 250 × 2 10.94 DiT-XL [38] 250 × 2 2.27 SiT-XL [34]† 250 × 2 2.06 SiT-XL + REPA [56] 250 × 2 1.42

+ Autoguidance [21] 63 × 2 1.34 EDM2-XXL [22] 63 × 2 1.81 Few-step models ECT† [11] 1 9.98 ECD† [11] 1 8.47 CMT [19]† 1 3.38 sCT-S [32] 1 10.13 sCD-S [32]† 1 3.07 AYF [42]† 1 3.32 Re-MeanFlow (ours)† 1 3.03

+ Autoguidance [21] 63 × 2 1.01 EDM2-XL [22] 63 × 2 1.33 Few-step models

2-rectified flow++ [25] 1 4.31 iCT [47] 1 4.02

###### Few-step models

ECD-S [11]† 1 3.30 sCD-S [32]† 1 2.97 TCM [26]† 1 2.88 AYF [42]† 1 2.98 Re-MeanFlow (ours)† 1 2.87

iCT [47] 1 34.6 SM [9] 1 10.6 iSM [35] 1 5.27 iMM [61] 1 × 2 7.77 MeanFlow [10] 1 3.43

Re-MeanFlow (ours)† 1 3.41

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

- Fig. 4: Visualization of Re-MeanFlow one-step generation (NFE=1) on ImageNet at 642 (Left), 2562 (Middle), and 5122 (Right).

Settings. We initialize Re-MeanFlow from a pretrained flow or diffusion model. For ImageNet-642 and ImageNet-5122, we initialize Re-MeanFlow from the pretrained EDM2-S model [22]. For ImageNet-2562, we initialize from the pretrained SiT-XL model [34]. To rectify the trajectory, we generate 5M data-noise couplings using the default procedures described in the corresponding papers. During sampling for better synthetic image quality, we apply classifier-free guidance (CFG) [17] for ImageNet-2562 and Autoguidance [21] for ImageNet-642 and 5122. More details on the implementation and the hyperparameter settings are provided in Appendix B.

Baselines. We compare Re-MeanFlow against recent state-of-the-art one-step flow-based methods, selecting baselines with comparable architecture or computational cost to ensure fair comparisons.

|[Figure 16]|[Figure 17]|[Figure 18]|[Figure 19]|[Figure 20]|[Figure 21]|
|---|---|---|---|---|---|

xmul

|[Figure 22]|[Figure 23]|[Figure 24]|[Figure 25]|[Figure 26]|[Figure 27]|
|---|---|---|---|---|---|

xone

|[Figure 28]|[Figure 29]|[Figure 30]|[Figure 31]|[Figure 32]|[Figure 33]|
|---|---|---|---|---|---|

∥xmul − xone∥

xone

###### Fig. 5: Re-MeanFlow yields faithful one-step sampling with significantly less

training. (a) Multi-step samples xmul (reference) obtained by integrating the predicted mean-velocity field; multi-step results from MeanFlow and Re-MeanFlow are visually similar, and we show a mixed set from both methods. (b) MeanFlow one-step samples xone after 20k iterations remain blurry and deviate noticeably from xmul. (c) Re-MeanFlow one-step samples after 10k iterations are already sharp and closely match xmul. (d) Sampling discrepancy distribution ∥xmul − xone∥: Re-MeanFlow is strongly left-shifted, indicating substantially smaller one-step vs. multi-step mismatch. All results are on ImageNet 5122, using the same noise seeds for both methods.

#### 4.1 One-Step Generation Quality

For all experiments, we evaluate image quality using Fréchet Inception Distance (FID) [15] in Tab.1. When comparing with prior methods, we select those with comparable architectures and parameter counts. Across all evaluated resolutions, Re-MeanFlow consistently outperforms state-of-the-art one-step flowbased generation methods. The qualitative results are shown in Fig. 4

On the EDM2-S backbone [22], Re-MeanFlow achieves superior performance at both 642 and 5122 resolutions. On ImageNet-642, Re-MeanFlow outperforms the closely related 2-rectified flow++ [25] by 33.4% in FID, and slightly improves over recent state-of-the-art one-step baselines [26,32,42]. On ImageNet5122, Re-MeanFlow also delivers strong one-step image quality, achieving a 9% FID gain over AYF [42] and outperforming strong consistency distillation methods [19,42].

On the SiT backbone [34] for ImageNet-2562, Re-MeanFlow slightly surpasses MeanFlow [10] despite being trained without real-image supervision and relying solely on synthesized samples. This is noteworthy because training exclusively on self-generated data is known to degrade performance due to self-conditioning effects [2]. We attribute this improvement to the more favorable optimization landscape created by rectified mean-velocity learning: although the supervision is limited to synthetic couplings, rectification produces a smoother and lowervariance trajectory family. As a result, the model converges more reliably toward a high-quality solution.

Sampling Cost Sampling Cost

Training Cost Training Cost x 26.6

x 6.96 x 7.06

x 3.04

x 2.9

x 1.0 x 1.17

x 1.0

Re − MeanFlow (Ours) FID : 2.87

Re − MeanFlow (Ours) FID : 2.87

- Fig. 6: Total compute comparison on ImageNet-642. We compare against the strongest prior methods in our main FID table (Tab. 1) in terms of the quality-efficiency trade-off. Total cost is reported in EFLOPs (left) and GPU hours (right), and each bar is decomposed into training (solid red) and coupling sampling (blue hatched). ReMeanFlow is the baseline (1.0×; numbers denote factors relative to Re-MeanFlow). Across both metrics, Re-MeanFlow achieves the lowest end-to-end compute, significantly outperforming prior approaches even after accounting for coupling sampling.

#### 4.2 Training Efficiency

Compared to MeanFlow. We compare Re-MeanFlow against MeanFlow on ImageNet 5122 without classifier-free guidance (CFG). As shown in Fig. 1, Re-MeanFlow converges substantially faster: even when MeanFlow is trained with 2× the compute budget, its one-step samples remain noticeably blurry, whereas Re-MeanFlow already produces sharp images, with a large FID gap (8.6 vs. 30.9). To diagnose this difference, we contrast one-step samples

xone = z − uθ(z,0,1) against multi-step samples xmul = z − 0 1 uθ(z,τ,τ)dτ obtained by simulating the integral of the instantaneous-velocity (Fig. 5). Both

methods are initialized from the same EDM2-S checkpoint, while MeanFlow is trained with 2× the compute budget. Qualitatively, Re-MeanFlow produces nearly indistinguishable samples under one-step and multi-step generation, indicating that uθ(z,0,1) closely matches the multi-step estimate. In contrast, MeanFlow’s one-step predictions are significantly degraded. Quantitatively, we measure the sampling discrepancy as ∥xmul − xone∥ and plot its distribution in Fig. 5d. Re-MeanFlow exhibits a pronounced left shift, confirming a smaller one-step-to-multi-step error and hence more accurate mean-velocity modeling of the underlying trajectories. We provide additional convergence comparisons with MeanFlow and other related methods in Appendix C.3.

Compared to Other One-Step Distillation Methods. As noted by [25], distillation methods involving sampling rectified couplings can still remain computationally competitive in overall computation cost, even though they require generating additional data-noise couplings.

t = 0.7, r = 0.7 t = 0.7, r = 0.5 t = 0.7, r = 0.3 t = 0.7, r = 0.1

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

- Fig. 7: Loss landscape of LMF on a PCA plane of zt. We visualize LMF as a function of the input zt by evaluating the loss on a 2D grid spanned by the top two PCA directions of a batch of zt samples. We fix t = 0.7 and sweep r ∈ {0.7, 0.5, 0.3, 0.1} (left to right), where larger t − r corresponds to a harder, longer-jump (r, t) regression task. Top (MeanFlow): the landscape becomes increasingly sharp and irregular as t − r grows. Bottom (Re-MeanFlow; ours): the landscape remains substantially smoother and better-conditioned across all r. Overall, learning mean velocity on straighter trajectories yields a markedly smoother objective, enabling more efficient and stable training.

We evaluate Re-MeanFlow by estimating the total computation cost in FLOPs

and GPU hours on ImageNet-642. The protocol for computing these metrics follows [25] and is detailed in the Appendix C.2. We compare against AYF [42], the strongest existing distillation baseline in both quality and efficiency, and against the closely related 2-rectified flow++ [25].

As shown in Fig. 6, Re-MeanFlow achieves the lowest overall compute cost among recent distillation approaches. In terms of GPU hours, Re-MeanFlow is 26.6× faster than 2-rectified flow++, reinforcing the results observed in our controlled efficiency experiments. Even compared to AYF, which does not require coupling generation, Re-MeanFlow remains 2.9× faster.

We also compare estimated FLOPs. Although Re-MeanFlow shows a smaller advantage under this metric, FLOPs alone do not fully reflect practical runtime. A substantial portion of our compute lies in the inference-only coupling sampling stage, which can be executed on widely accessible inference-grade GPUs and runs significantly faster than training workloads with the same FLOP count.

#### 4.3 Loss Landscape

To explain the training speedup enabled by learning mean-velocity on straighter trajectories, we visualize in Fig. 7 the loss landscape of LMF on ImageNet 2562 with respect to the input zt for both MeanFlow and Re-MeanFlow. We fix t = 0.7 and plot the landscape for different values of r. Additional results at other resolutions are provided in Appendix C.1.

- Table 2: Effect of distance-based truncation strength (discarding a top fraction of couplings by ℓ2 endpoint distance). We report FID/IS and Precision/Recall on ImageNet 5122 (top) and 2562 (bottom). Mild truncation consistently improves stability and sample quality (lower FID, higher IS) without sacrificing diversity (comparable Precision/Recall). Notably, even without truncation, Re-MeanFlow matches MeanFlow trained from scratch on real data.

ImageNet 5122 Method FID ↓ IS ↑ Precision ↑ Recall ↑

Re-MeanFlow (no trunc.) 3.50 242.62 0.73 0.54 Re-MeanFlow (top 5%) 3.10 251.84 0.74 0.56 Re-MeanFlow (top 10%) 3.03 262.37 0.75 0.54 Re-MeanFlow (top 15%) 3.19 259.61 0.76 0.53

###### ImageNet 2562

MeanFlow [10] 3.43 247.5 0.73 0.54 Re-MeanFlow (ours; no trunc.) 3.48 243.4 0.72 0.54 Re-MeanFlow (ours; top 10%) 3.41 249.6 0.73 0.54

To construct these visualizations, we perform PCA on a batch of zt samples to obtain the top two principal directions. We then evaluate LMF on a 2D grid spanned by these directions and centered at the sampled zt, and visualize the resulting surfaces. As shown in Fig. 7, Re-MeanFlow consistently exhibits a smoother and better-conditioned landscape. In particular, as t − r grows, the MeanFlow objective becomes increasingly sharp and irregular, whereas Re-MeanFlow remains substantially smoother and more stable.

Unlike conventional loss-landscape studies [28] that perturb model parameters, we perturb zt to directly measure the conditioning of the meanvelocity regression induced by the trajectory geometry. When trajectories are highly curved, the interval mean velocity becomes a rapidly varying, locally unstable function of zt, yielding a rugged loss landscape in zt-space. Such ruggedness indicates that the regression target itself is poorly conditioned, which in turn hampers optimization. In contrast, rectified (straighter) trajectories markedly smooth the landscape as a function of zt, producing a betterconditioned learning problem and enabling more efficient training.

#### 4.4 Distance-based Truncation

- As discussed in Sec. 3.4, we empirically observe that large endpoint distances ∥z − x∥2 are correlated with higher trajectory curvature. We therefore apply a simple top-10% truncation rule, discarding the couplings with the largest ℓ2 endpoint distance. In practice, this filter removes many residual high-curvature pairs, improving training stability and sample quality for Re-MeanFlow.

To study the effect of the truncation ratio, we sweep different truncation rates and report results on ImageNet 5122 and ImageNet 2562 (Tab. 2). We find that moderate truncation consistently improves stability and yields better IS/FID,

Table 3: Ablation study on ImageNet-5122 for key implementation details.

Training configurations FID ↓ Base (Best Setting reported in [10]) 7.81

- (a) + Hyperparameter adjustments 7.22
- (b) + Time embedding change 4.60
- (c) + U-shaped t distribution 3.71
- (d) + Avoid high-variance (r, t) region 3.50

- (e) + Distance-based truncation 3.03

while maintaining diversity as reflected by comparable Recall. In particular, FID improves monotonically within a moderate truncation regime.

Interestingly, even without truncation, Re-MeanFlow already matches the performance of MeanFlow trained directly on real data. This is notable because Re-MeanFlow does not require access to the original dataset: it learns solely from generated coupling pairs. Since self-generated distillation typically incurs discretization error that can degrade performance, we hypothesize that the rectified trajectories substantially simplify the optimization problem, enabling more effective parameter learning and thus competitive (or improved) results.

#### 4.5 Ablation Study

To identify the implementation choices most critical for stable and high-quality training, we conduct an ablation study on ImageNet-5122 in Tab. 3. We summarize the main findings below.

- (a) Hyperparameter adjustments. Since rectified trajectories are substantially straighter, we reduce the normalization strength in MeanFlow’s adaptive loss from 1.0 to 0.5 (Pseudo-Huber style).
- (b) Time embedding change. Following [32,42], we replace the EDM2 time

embedding emb(log σt) with emb(t) to improve the stability of Jacobian-vector product computations.

- (c) U-shaped t distribution. Because rectified trajectories reduce the need for mid-range emphasis, we adopt the U-shaped t distribution from [25] in place of the standard lognormal schedule.
- (d) Avoid high-variance (r,t) region. We observe unusually high variance when t > 0.95 and r < 0.4 (analyzed in the Appendix B.3). Inspired by the truncation strategy in TCM [26], we exclude this time region, which improves FID and accelerates convergence.
- (e) Distance-based truncation. Finally, applying our distance-based truncation heuristic (Sec. 3.4) further improves both efficiency and generation quality.

### 5 Conclusion

We introduced Rectified-MeanFlow, a lightweight, data-free self-distillation framework for one-step generation that addresses a key MeanFlow bottleneck:

learning mean-velocity on highly curved trajectories. Our central geometric takeaway is that mean-velocity estimation is substantially easier on straighter paths. Accordingly, Re-MeanFlow models mean-velocity on rectified couplings generated from a pretrained flow model, yielding a simpler velocity field and a markedly better-conditioned loss landscape; this translates into faster convergence and stronger one-step generation. We further improve robustness with a distance-based truncation rule that removes residual high-curvature couplings. Empirically, on ImageNet at 642, 2562, and 5122, Re-MeanFlow consistently outperforms prior one-step flow distillation methods and strong Rectified Flow baselines, achieving higher sample quality with substantially less training.

### References

- 1. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571 (2022)
- 2. Alemohammad, S., Casco-Rodriguez, J., Luzi, L., Humayun, A.I., Babaei, H., LeJeune, D., Siahkoohi, A., Baraniuk, R.: Self-consuming generative models go mad. In: The Twelfth International Conference on Learning Representations (2023)
- 3. Alemohammad, S., Humayun, A.I., Agarwal, S., Collomosse, J., Baraniuk, R.: Selfimproving diffusion models with synthetic data. arXiv preprint arXiv:2408.16333

(2024)

- 4. Alemohammad, S., Wang, Z., Baraniuk, R.G.: Neon: Negative extrapolation from self-training improves image generation. arXiv preprint arXiv:2510.03597 (2025)
- 5. Boffi, N.M., Albergo, M.S., Vanden-Eijnden, E.: Flow map matching. arXiv preprint arXiv:2406.07507 2 (2024)
- 6. Dao, Q., Doan, K., Liu, D., Le, T., Metaxas, D.: Improved training technique for latent consistency models. arXiv preprint arXiv:2502.01441 (2025)
- 7. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)
- 8. Dhariwal, P., Nichol, A.: Diffusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34 (2021)
- 9. Frans, K., Hafner, D., Levine, S., Abbeel, P.: One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557 (2024)
- 10. Geng, Z., Deng, M., Bai, X., Kolter, J.Z., He, K.: Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447 (2025)
- 11. Geng, Z., Pokle, A., Luo, W., Lin, J., Kolter, J.Z.: Consistency models made easy. arXiv preprint arXiv:2406.14548 (2024)
- 12. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)
- 13. He, X., Dao, Q., Han, L., Wen, S., Bai, M., Liu, D., Zhang, H., Min, M.R., Juefei-Xu, F., Tan, C., et al.: Dice: Discrete inversion enabling controllable editing for multinomial diffusion and masked generative models. arXiv preprint arXiv:2410.08207 (2024)
- 14. He, X., Tan, C., Han, L., Liu, B., Axel, L., Li, K., Metaxas, D.N.: Dmcvr: Morphology-guided diffusion model for 3d cardiac volume reconstruction. In: International conference on medical image computing and computer-assisted intervention. pp. 132–142. Springer Nature Switzerland Cham (2023)

- 15. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)
- 16. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)
- 17. Ho, J., Salimans, T.: Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022)
- 18. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. Advances in neural information processing systems 35, 8633– 8646 (2022)
- 19. Hu, Z., Lai, C.H., Mitsufuji, Y., Ermon, S.: Cmt: Mid-training for efficient learning of consistency, mean flow, and flow map models. arXiv preprint arXiv:2509.24526

(2025)

- 20. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems 35, 26565–26577 (2022)
- 21. Karras, T., Aittala, M., Kynkäänniemi, T., Lehtinen, J., Aila, T., Laine, S.: Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems 37, 52996–53021 (2024)
- 22. Karras, T., Aittala, M., Lehtinen, J., Hellsten, J., Aila, T., Laine, S.: Analyzing and improving the training dynamics of diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24174– 24184 (2024)
- 23. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 4401–4410 (2019)
- 24. Kim, D., Lai, C.H., Liao, W.H., Murata, N., Takida, Y., Uesaka, T., He, Y., Mitsufuji, Y., Ermon, S.: Consistency trajectory models: Learning probability flow ode trajectory of diffusion. arXiv preprint arXiv:2310.02279 (2023)
- 25. Lee, S., Lin, Z., Fanti, G.: Improving the training of rectified flows. Advances in neural information processing systems 37, 63082–63109 (2024)
- 26. Lee, S., Xu, Y., Geffner, T., Fanti, G., Kreis, K., Vahdat, A., Nie, W.: Truncated consistency models. arXiv preprint arXiv:2410.14895 (2024)
- 27. Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., Zheng, L.: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483 (2025)
- 28. Li, H., Xu, Z., Taylor, G., Studer, C., Goldstein, T.: Visualizing the loss landscape of neural nets. Advances in neural information processing systems 31 (2018)
- 29. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022)
- 30. Liu, Q.: Rectified flow: A marginal preserving approach to optimal transport. arXiv preprint arXiv:2209.14577 (2022)
- 31. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022)
- 32. Lu, C., Song, Y.: Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081 (2024)
- 33. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems 35, 5775–5787 (2022)

- 34. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In: European Conference on Computer Vision. pp. 23–40. Springer

(2024)

- 35. Nguyen, A., Nguyen, V., Vu, D., Dao, T., Tran, C., Tran, T., Tran, A.: Improved training technique for shortcut models. arXiv preprint arXiv:2510.21250 (2025)
- 36. Onken, D., Fung, S.W., Li, X., Ruthotto, L.: Ot-flow: Fast and accurate continuous normalizing flows via optimal transport. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 35, pp. 9223–9232 (2021)
- 37. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)
- 38. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023)
- 39. Pooladian, A.A., Ben-Hamu, H., Domingo-Enrich, C., Amos, B., Lipman, Y., Chen, R.T.: Multisample flow matching: Straightening flows with minibatch couplings. arXiv preprint arXiv:2304.14772 (2023)
- 40. Rezende, D., Mohamed, S.: Variational inference with normalizing flows. In: International conference on machine learning. pp. 1530–1538. PMLR (2015)
- 41. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)
- 42. Sabour, A., Fidler, S., Kreis, K.: Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603 (2025)
- 43. Sehwag, V., Kong, X., Li, J., Spranger, M., Lyu, L.: Stretching each dollar: Diffusion training from scratch on a micro-budget. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 28596–28608 (2025)
- 44. Seong, K.S., Kwon, M., Jeong, J., Uh, Y.: Balanced conic rectified flow. arXiv preprint arXiv:2510.25229 (2025)
- 45. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., Ganguli, S.: Deep unsupervised learning using nonequilibrium thermodynamics. In: International conference on machine learning. pp. 2256–2265. pmlr (2015)
- 46. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020)
- 47. Song, Y., Dhariwal, P.: Improved techniques for training consistency models. arXiv preprint arXiv:2310.14189 (2023)
- 48. Song, Y., Dhariwal, P., Chen, M., Sutskever, I.: Consistency models (2023)
- 49. Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems 32 (2019)
- 50. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 (2020)
- 51. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482 (2023)
- 52. Wu, G., Zhang, S., Shi, R., Gao, S., Chen, Z., Wang, L., Chen, Z., Gao, H., Tang, Y., Yang, J., et al.: Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467 (2025)
- 53. Xu, Q., Wang, Z., He, X., Han, L., Tang, R.: Can large vision-language models detect images copyright infringement from genai? arXiv preprint arXiv:2502.16618

(2025)

- 54. Yang, L., Zhang, Z., Zhang, Z., Liu, X., Xu, M., Zhang, W., Meng, C., Ermon, S., Cui, B.: Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398 (2024)
- 55. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15703–15712 (2025)
- 56. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940 (2024)
- 57. Zhai, S., Zhang, R., Nakkiran, P., Berthelot, D., Gu, J., Zheng, H., Chen, T., Bautista, M.A., Jaitly, N., Susskind, J.: Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329 (2024)
- 58. Zhang, Q., Chen, Y.: Fast sampling of diffusion models with exponential integrator. arXiv preprint arXiv:2204.13902 (2022)
- 59. Zhang, X., Wen, S., Han, L., Juefei-Xu, F., Srivastava, A., Huang, J., Pavlovic, V., Wang, H., Tao, M., Metaxas, D.: Soda: Spectral orthogonal decomposition adaptation for diffusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 4665–4682. IEEE (2025)
- 60. Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305 (2023)
- 61. Zhou, L., Ermon, S., Song, J.: Inductive moment matching. arXiv preprint arXiv:2503.07565 (2025)

### Appendix

- A More Discussion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1

- A.1 Limitation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- A.2 Broader Impact . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- A.3 Comparison with One-Step Sampling via Rectified Flow . . . . . . . . 2
- A.4 Comparison with CMT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- B Implementation Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3

- B.1 Pretrained Model Conditioning. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- B.2 Time Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- B.3 Avoiding High-Variance Time Regions. . . . . . . . . . . . . . . . . . . . . . . . 5
- B.4 Training with Guidance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- C More Experiment Details and Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- C.1 Loss Landscape . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- C.2 Computation Estimation of Each Method . . . . . . . . . . . . . . . . . . . . 8
- C.3 Additional Convergence Comparison with Other Related Methods 9
- C.4 Curvature Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10

- D 2D Toy Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- E More Qualitative Results. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- E.1 ImageNet-642 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- E.2 ImageNet-2562 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- E.3 ImageNet-5122 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15

### A More Discussion

In this section, we provide additional analysis and context for our method. We begin by discussing its limitations and then examine the broader impact of our approach. Next, we revisit the theoretical motivation for using a single rectification iteration and explain how our perspective complements existing analyses. Finally, we compare our method with the concurrent CMT method [19].

#### A.1 Limitation

Since our method does not access real data during training and relies entirely on synthetic samples generated by pretrained diffusion or flow models, its performance naturally depends on the quality of these generated couplings. A promising direction is to incorporate real data into the training process, as explored in [25, 44]. Another complementary line of work investigates how to improve generative models using only their own outputs [2–4], which suggests that selfgenerated data can still be leveraged effectively with appropriate regularization. Alternatively, one could improve the synthetic supervision directly by using stronger backbone models to generate the couplings.

#### A.2 Broader Impact

Beyond empirical gains, our method highlights a practical and socially relevant shift in how large generative models can be trained. Traditional few-step or one-step distillation pipelines often concentrate compute in expensive training workloads on high-end accelerators (e.g., A100-class GPUs), which limits accessibility to well-resourced institutions. In contrast, Re-MeanFlow shifts a substantial fraction of computation to an inference-only stage (sampling rectified couplings), followed by a lightweight MeanFlow training phase. Because inference workloads can be executed efficiently on widely available consumer- or inference-grade accelerators, our framework reduces reliance on scarce training GPUs and lowers the barrier to experimentation.

Practically, the sampling process for the rectified couplings is amenable

to pipeline parallelism: coupling generation can run asynchronously alongside training as a data-preparation stream. For implementation convenience, we pre-generate couplings before training in this work; nevertheless, even under this conservative setup, Re-MeanFlow achieves the lowest total compute among prior state-of-the-art distillation methods, underscoring the endto-end efficiency of our framework.

#### A.3 Comparison with One-Step Sampling via Rectified Flow

Rectified Flow [31] supports few-step sampling via an iterative reflow procedure that progressively straightens trajectories by repeatedly training a new flow model on rectified couplings. Liu et al. [31] show that, in the limit, iterating this procedure can yield perfectly straight couplings. In practice, however, achieving near-linear paths for reliable one-step sampling typically requires multiple reflow rounds, which is computationally expensive and can degrade performance due to accumulated approximation errors.

Lee et al. [25] analyze when multiple Reflow iterations are truly necessary for achieving straight trajectories in rectified flows. Their argument centers on how trajectory intersections affect the learned velocity field. Consider two 1-rectified couplings (x′,z′) and (x′′,z′′). A trajectory intersection occurs if there exists t ∈ [0,1] such that

(1 − t)x′ + tz′ = (1 − t)x′′ + tz′′. If such an intersection happens, both trajectories pass through the same intermediate point. Because rectified-flow training regresses the conditional expectation E[x|xt], the model must assign a single velocity to this shared point. As a result, the learned velocity field cannot simultaneously point toward both x′ and x′′, and it instead averages their directions. This averaging effect bends the local velocity field, producing curvature and consequently degrading the accuracy of one-step Euler sampling, which assumes the path to be straight.

To understand how often this phenomenon can happen, Lee et al. [25] show that an intersection implies

1 − t t

z′′ = z′ +

(x′ − x′′).

Under typical training, nearly all noise samples used to form 1-rectified couplings lie in the high-density region of the Gaussian prior. The z′′ required above usually lies far outside that region unless ||x′ −x′′||2 is extremely small or t is very close to 1. Therefore, intersections are statistically rare. Further assuming that the

- 1-rectified flow is approximately L-Lipschitz, ||x′ − x′′||2 ≤ L||z′ − z′′||2, (10)

nearby noise samples cannot map to widely separated data points. Combining the rarity of intersections with this Lipschitz condition, the authors conclude that the optimal 2-rectified flow is nearly straight. Hence, in their view, one additional Reflow step is sufficient, and any remaining performance gap should be attributed primarily to training inefficiency rather than insufficient straightening.

Relation to Our Work. Our focus is complementary. While the above analysis suggests that trajectory intersections are rare for most couplings, our experiments indicate that realistic settings can still contain a small but influential subset of pairs with non-negligible curvature, particularly when the effective Lipschitz constant L is large due to geometric imbalance in the data distribution. As shown in Fig. 2b and Fig. 11, trajectories induced by a once-rectified coupling (i.e., one reflow step) are substantially straighter than those from the independent coupling (Fig. 2a), yet residual curvature remains and can still hinder reliable one-step sampling based on the instantaneous velocity. Re-MeanFlow is designed to be robust in precisely these challenging cases, enabling stable onestep generation even when curvature persists after a single rectification step.

#### A.4 Comparison with CMT

CMT [19] is an important concurrent effort that also leverages synthetic trajectories generated by a pretrained sampler to stabilize few-step flow-map training. Conceptually, CMT introduces a dedicated mid-training stage that learns a full trajectory-to-endpoint mapping from solver-generated paths, which then serves as a trajectory-aligned initialization for a subsequent post-training flow-map stage. In contrast, our method adopts a fundamentally different design: rather than supervising on entire solver trajectories, we distill only the end-point couplings of rectified flows and learn the corresponding mean velocity in a single training stage. This distinction yields a practical advantage: our pipeline avoids the compounded complexity of CMT’s two-stage optimization, which is sensitive to hyperparameters at both stages and substantially more expensive to tune.

### B Implementation Details

In this section, we first describe the conditioning strategy of how Re-MeanFlow utilizes previous pretrained models. We then outline the key design choices during training Re-MeanFlow. We provide the training hyperparameters across all ImageNet resolutions in Tab. 4.

###### Table 4: Training settings of Re-MeanFlow on ImageNet.

| |Resolution 642 2562 5122|
|---|---|
|Training Details| |
|Model Backbone Global Batch size Learning Rate<br><br>Adam β1<br>Adam β2 Model Capacity (Mparams) EMA Rate<br>|EDM2-S [22] SIT-XL [34] EDM2-S [22] 128 128 128 1e-4 1e-4 1e-4 0.9 0.9 0.9 0.99 0.95 0.99 280.2 676.7 280.5 0.9999 0.9999 0.9999<br><br>|
|MeanFlow Setting Details| |
|Ratio of r = t p for Adaptive Weight CFG Effective Scale w′ Avoiding High-variance (t, r) Region<br><br>|0.25 0.25 0.25 0.5 0.5 0.5 Uniform(1.0, 3.0) Uniform(1.0, 3.0) Uniform(1.0, 2.5) t > 0.95 & r < 0.4 t > 0.95 & r < 0.4 t > 0.95 & r < 0.4|
|Sampling Details for Rectified Couplings| |
|Pretrained Model Sampling Number Guidance Method Distance Truncate Strength|EDM2-S [22] SIT-XL [34] EDM2-S [22] 5M 5M 5M Autoguidance [21] CFG [17] Autoguidance [21] Top 10% Top 10% Top 10%<br><br>|

̸

#### B.1 Pretrained Model Conditioning

In Re-MeanFlow, the velocity network is conditioned on two time variables, t and r. We implement this conditioning by learning two separate embeddings, embt(t) and embr(r), and summing them before passing the result to the rest of the network. When initializing from pretrained models, the original networks only contain a single time embedding for t. Replacing this embedding with our two-embedding design requires careful initialization to ensure the model initially behaves like the pretrained flow mode. Specifically, before MeanFlow fine-tuning, we want:

u(xt,t,r) ≈ v(xt,t) (11)

On ImageNet 642 and 5122 , which Re-MeanFlow is initialized from the pretrained EDM2-S model [22], following AYF [42], we perform a short alignment stage in which we train the new embeddings to reproduce the original timeembedding output for the corresponding noise level. Specifically, for EDM2-S the original embedding depends on log σt, where σt = 1−t t. We train the new embeddings via:

Et,r[||embt(t) + embr(r) − embori(log σt)||22], (12)

for 10k iterations with learning rate 1e-3. This process takes only a few minutes. We also convert the original VE diffusion parameterization into the flowmatching setting following [25].

On ImageNet 2562 , we initialize Re-MeanFlow from SiT-XL [34], which is already a flow-based model. In this case, only the additional r-embedding needs

[Figure 38]

[Figure 39]

- Fig. 8: Heatmaps of the mean loss (left) and the standard deviation of the loss (right) for a trained Re-MeanFlow model on ImageNet-5122 under configuration (c).

to be introduced. We simply zero-initialize embr(r) and keep the original SiT time embedding for embt(t).

#### B.2 Time Distribution

We observe a similar loss–time profile to that reported in [25]: the training loss as a function of t closely matches that of 2-rectified flow++. Accordingly, for sampling t we adopt the same U-shaped distribution:

pt(u) ∝ exp(au) + exp(−au), u ∈ [0,1], a = 4.

Following AYF [42], after sampling t we draw the interval length |t−r| from a normal distribution N(Pmean,Pstd) and apply a sigmoid transformation. We use the same parameters as AYF, (Pmean,Pstd) = (−0.8,1.0), which emphasize medium-length intervals and substantially improve stability. As shown in Table 3, this setting yields a noticeable improvement in FID and accelerates convergence.

We also experimented with sampling t uniformly. Despite its simplicity, uniform sampling performs competitively—and in many cases better—than commonly used log-normal time distributions in diffusion and flow-matching models [10,20,22], which prioritize mid-range timesteps to avoid high variance near t ≈ 0 and t ≈ 1. We attribute the strong performance of uniform sampling to the significantly lower variance of rectified trajectories: after one reflow step, early-time and high-noise regions become much more stable, allowing us to allocate more samples to these challenging regimes without the usual degradation observed when training on independent couplings.

#### B.3 Avoiding High-Variance Time Regions

- As discussed in Sec. 4.4, we observe that Re-MeanFlow exhibits unusually high variance when the noise level t is large while the reference time r is close to zero. Intuitively, this corresponds to asking the model to predict a slightly denoised sample (r ∈ [0,0.4]) from an input that remains heavily corrupted (t ≈ 1).

To quantify this effect, we sample 100k pairs of (t,r) uniformly and evaluate a trained Re-MeanFlow model under configuration (c) in Table 3. As shown in Fig. 8, the resulting loss landscape displays a clear spike in both error and variance within this region, often even higher than the loss incurred when predicting directly from noise to a clean target.

Inspired by the truncation strategy in TCM [26], we adopt a simple yet effective rule to avoid this problematic regime: whenever a sampled pair satisfies t > 0.95 and r < 0.4, we set r = 0. Empirically, this improves both training stability and FID. Our hypothesis is that predicting a clean image from pure noise (r = 0, t ≈ 1) is substantially easier than predicting a lightly corrupted target: the latter requires the model to determine which noise components should be preserved, introducing ambiguity and variance at high t. By redirecting training toward these easier high-t targets, the model can allocate more capacity to learning accurate one-step predictions. This modification not only improves FID relative to configuration (c), but also accelerates convergence: in the high-t regime, more updates involve r = 0, allowing the model to refine its one-step outputs more quickly and reliably.

#### B.4 Training with Guidance

Classifier-free guidance (CFG) [17] is widely used to boost the performance of diffusion and flow-based generative models. To incorporate CFG into the MeanFlow stage, we train Re-MeanFlow on the CFG-enhanced velocity field:

vcfg(zt,t | c) ≜ ω v(zt,t | c) + (1 − ω)v(zt,t). (13)

MeanFlow [10] further introduced an improved CFG method that mixes conditional and unconditional mean-velocity predictions:

vcfg(zt,t | c) = ω v(zt,t | c) + κucfg(zt,t,t|c) + (1 − ω + κ)ucfg(zt,t,t). (14)

which is equivalent to using an effective guidance of ω′ = 1−ωκ. We adopt this improved CFG formulation for all experiments.

Empirically, we found that directly training MeanFlow on the CFG field is unstable, consistent with observations in [19]. To mitigate this, we use a simple two-stage strategy: first train uθ on the unconditional flow, then on the CFGmodified flow. Usually, allocating half of the total training budget to each stage provides a good balance between stability and final quality.

We also found that sampling a random CFG scale ω′ from a uniform distribution (rather than fixing it) gives better results. Large values of κ are also important for stable training. In practice, we sample ω′ from a uniform distribution, then set κ = max(1.0,ω′ −1), and finally compute the corresponding value for ω = ω

′

1−κ.

### C More Experiment Details and Results

In this section, we provide additional experimental details, computational analyses, and extended results. We first present loss-landscape visualizations at 642

t = 0.7, r = 0.7 t = 0.7, r = 0.5 t = 0.7, r = 0.3 t = 0.7, r = 0.1

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Re-MeanFlow(Ours)MeanFlow

loss

loss

loss

loss

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

loss

loss

loss

loss

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

- (a) Loss Landscape on ImageNet 642
- (b) Loss Landscape on ImageNet 5122

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

###### Re-MeanFlow(Ours)MeanFlow

loss

loss

loss

loss

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

loss

loss

loss

loss

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

PCA1 PCA2

- Fig. 9: Loss landscape of LMF on a PCA plane of zt for ImageNet 642 (a) and 2562 (b). We evaluate LMF on a 2D PCA grid with t = 0.7 and r ∈ {0.7, 0.5, 0.3, 0.1} (left to right). MeanFlow (top) becomes increasingly sharp and irregular as t − r grows, whereas Re-MeanFlow (bottom) remains substantially smoother and better-conditioned, supporting more stable and efficient training.

and 2562, then describe our protocol for estimating FLOPs and GPU-hours across all methods. Finally, we report additional convergence results that further support the synergy between trajectory rectification and mean-velocity modeling.

#### C.1 Loss Landscape

In Fig. 7, we visualize the loss landscape of LMF with respect to the input zt for MeanFlow and Re-MeanFlow across different (t,r) pairs. Here we report the corresponding results at additional resolutions in Fig. 9, following the same protocol.

Concretely, we fix t = 0.7 and sweep r, and compute PCA directions by collecting 20k samples of zt (equivalently, xt in our implementation). For each resolution, we then evaluate LMF on a 2D grid spanned by the top two PCA

directions using a minibatch of samples. Collecting the data and generating all surfaces for a single resolution takes approximately 3 hours on one A100 GPU. Overall, consistent with Fig. 7, Re-MeanFlow exhibits a substantially smoother and better-conditioned loss landscape, which translates into more efficient and stable training.

#### C.2 Computation Estimation of Each Method

FLOPS Estimation. In Fig. 6 (left), we report efficiency in terms of estimated exaFLOPs (EFLOPs). To ensure comparability, we estimate total training and sampling compute for each method based on their reported FLOPs per forward pass. Specifically, we use the following assumptions:

- – The FLOPs of a forward pass are reported by prior works (e.g., EDM [20]: 100 GFLOPs, EDM2-S [22]: 102 GFLOPs, SiT-XL [34]: 118.64 GFLOPs).
- – The FLOPs of a backward pass are measured empirically and are approximately 2× the cost of a forward pass. (One JVP operation is also counted as a backward pass), say for our example on the first stage where we will perform one forward of the model and one JVP operation and one back propagation with JVP counted as one back propagation we have total flop amount of 1 + (2 × 2) = 1 + 4 forward flops.
- – For training, the total compute is computed as:

Total Train FLOPs =(#iters) × (batch size) × (forward + backward) × (GFLOPs per fwd).

- – For sampling in the reflow process, the total compute is computed as:

Total Sample FLOPs =(#samples) × (#steps) × (forward passes per step) × (GFLOPs per fwd).

- – Example: Re-MeanFlow (Ours) on ImageNet-642. For sampling, we require:

5 × 106 #samples

× 63

steps

× 2

fwd/step (auto-guidance)

× 102

GFLOPs/fwd

≈ 64 Eflops.

For training, we have two stages, with the first stage trained on the original flow and the second stage trained on the CFG velocity field:

50,000

iters

× 128

batch

× (1 + 4)

fwd+back

× 102

GFLOPs/fwd

+ 50,000

iters

× 128

batch

× (3 + 4)

fwd+back

× 102

GFLOPs/fwd

≈ 8 Eflops.

- – Example: AYF [42]. AYF does not require sampling, so only the training computation is considered: 50,000

≈ 8.36 × 1010 GFLOPs ≈ 84 Eflops.

× (4 + 4)

× 102

×2048

batch

GFLOPs/fwd

iters

fwd+back

(a) ImageNet 512 (b) ImageNet 256

- Fig. 10: Additional convergence comparison. FID vs. training time (8×A100 GPU hours) for (a) ImageNet 5122 and (b) ImageNet 2562. We compare Re-MeanFlow against MeanFlow [10] and closely related baselines, including 2-rectified flow++ [25] and MeanFlow trained with mini-batch OT couplings [51] (5122 only). Across resolutions and baselines, Re-MeanFlow converges markedly faster and reaches substantially lower FID.

GPU Hours Estimation. In Fig.6, we also estimate the total GPU hours for AYF [42] and 2-rectified flow++ [25]. For all methods, including ours, we follow the standard convention of computing GPU hours as

GPU Hours = (# of GPUs) × (wall-clock training time).

For example, Re-MeanFlow requires 66 hours of wall-clock time on 8 A100 GPUs, yielding 8 × 66 = 528 GPU hours.

- C.3 Additional Convergence Comparison with Other Related Methods

- In Fig. 1c, we compare the convergence of Re-MeanFlow with MeanFlow [10]. Here, we provide additional convergence results against closely related baselines.

- 2-rectified flow++. We compare against 2-rectified flow++ [25], which improves Rectified Flow but still models the instantaneous velocity on rectified couplings, in contrast to Re-MeanFlow, which models mean-velocity. We observe that one-step sample quality under 2-rectified flow++ improves slowly. We attribute this to the sensitivity of one-step Euler updates to residual curvature: accurate one-step sampling with instantaneous velocity effectively requires near-perfect straightness, since even mild curvature can cause overshooting. Because rectified trajectories are not perfectly linear in practice, this sensitivity compounds and slows convergence. This observation aligns with our efficiency comparison in Sec. 4.2: relative to 2-rectified flow++, Re-MeanFlow achieves a 33.4% lower FID while being 26× faster.

[Figure 52]

[Figure 53]

###### 10 X. Zhang et al.

[Figure 54]

[Figure 55]

Curv(r,t) Curv(r,t)

Curv(r,t) Curv(r,t)

r t r t

r t r t

MeanFlow

###### Re-MeanFlow (Ours)

MeanFlow Re-MeanFlow (Ours)

(a) Curvature Estimation for the Generative Trajectories on ImageNet 642

(b) Curvature Estimation for the Generative Trajectories on ImageNet 2562

- Fig. 11: Curvature estimation of generative trajectories on ImageNet 642 (a) and 2562 (b). Using the proxy in Sec. 3.3, we visualize Curv(r, t) over the (r, t)-plane. Rectified couplings used by Re-MeanFlow induce substantially straighter trajectories (lower curvature) than the independent couplings used by MeanFlow.

MeanFlow with Mini-batch OT couplings. We also train MeanFlow using Mini-batch OT [51] couplings, obtained by locally solving an OT matching within each mini-batch (via an OT server) to pair sampled noises and data points. Results on ImageNet 5122 are shown in Fig. 10a. While mini-batch OT slightly improves over standard MeanFlow, its convergence behavior remains similar and still lags substantially behind Re-MeanFlow. This is consistent with our discussion in Sec. 2.2: Mini-batch OT provides only local transport structure and does not guarantee globally straight trajectories.

Overall, Re-MeanFlow converges substantially faster than these closely related alternatives, highlighting the synergy of modeling mean-velocity on rectified trajectories.

#### C.4 Curvature Estimation

- In Fig. 2, we visualize the curvature proxy on ImageNet 5122. Here we provide the corresponding curvature estimates at additional resolutions in Fig. 11. The same trend holds across settings: compared to independent couplings, the rectified couplings used to train Re-MeanFlow consistently induce substantially lower curvature over the (r,t)-plane.

### D 2D Toy Example

To visualize the synergy of Re-MeanFlow relative to MeanFlow [10] and Rectified Flow [31], we present a 2D toy transport problem in Fig. 12. The task maps a two-mode Gaussian mixture to another two-mode mixture. We compare one-step generation under a fixed budget of 20k training iterations for three approaches: (i) 2-rectified flow, (ii) MeanFlow, and (iii) Re-MeanFlow (ours), which trains a velocity model for 10k iterations to obtain a 1-rectified coupling, followed by 10k iterations of MeanFlow on the resulting couplings.

[Figure 56]

[Figure 57]

[Figure 58]

(a) Linear interplolation of independent couplings

(b) Multi-step sampling on 1-rectified flow

(c) Multi-step sampling on 2-rectified flow

[Figure 59]

[Figure 60]

[Figure 61]

(d) One-step Euler sampling on 2-rectified flow

(e) MeanFlow

(f) Re-MeanFlow (Ours)

- Fig. 12: A 2D Toy Example. We consider a controlled 2D setup where a flow model transports a two-component Gaussian mixture on the left to a mixture on the right. Panels (a-c): (a) Linear interpolation of independently sampled couplings px × pz, which serves as the training signal for the first velocity model. (b) The resulting 1rectified flow learned from these independent couplings; the learned velocity field remains noticeably curved. (c) Using the velocity field from (b), we generate a new set of couplings and train a second velocity model on their linear interpolations, yielding the 2-rectified flow. Panel (d): Due to imperfect straightening, one-step Euler sampling on the 2-rectified flow still yields noticeable outliers. Panel (e): MeanFlow trained directly on independent couplings fails to converge within the training budget because high-variance conditional velocities destabilize learning. Panel (f): Re-MeanFlow combines trajectory rectification with mean-velocity modeling, eliminating most outliers and achieving more accurate one-step generation.

- Fig. 12d: 2-rectified flow. Even after one rectification step, some samples traverse disproportionately long distances toward the denser mode (e.g., due to approximation error or mild mode imbalance). As a result, nearby noise samples can map to widely separated data points, yielding a large effective L in Eq. 10 and inducing early-time curvature near t≈0. The pronounced one-step deviations in

- Fig. 12d illustrate this failure mode: one-step Euler updates using instantaneous velocity can overshoot when trajectories are not perfectly straight. Similar geometric imbalances can arise in real datasets, motivating robustness to residual curvature.
- Fig. 12e: MeanFlow. MeanFlow trained directly on independent couplings struggles to learn a coherent transport map within the same budget, resulting in poor one-step generation.
- Fig. 12f: Re-MeanFlow (ours). Re-MeanFlow achieves more accurate onestep generation with substantially fewer invalid samples. This illustrates

[Figure 62]

[Figure 63]

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

(a) Linear interpolation of independent couplings

(b) Untruncated multi-step sampling with 1-Rectified Flow.

(c) Retained trajectories after truncation (bottom 90%) in (b).

[Figure 64]

[Figure 65]

[Figure 66]

(d) Truncated trajectories (top 10%) removed from (b).

(e) Re-MeanFlow (Ours) trained on the untruncated couplings in (b)

(f) Re-MeanFlow (Ours) trained on the truncated couplings in (c)

- Fig. 13: Distance-based truncation removes high-curvature couplings and improves one-step behavior in a 2D toy task. (a) Independent couplings visualized by linear interpolation between sampled endpoints. (b) Rectified couplings obtained by multi-step sampling with 1-rectified flow, where a small set of long-distance “diagonal” pairings induces high-curvature trajectories. (c) Retained trajectories after truncating the top 10% couplings ranked by endpoint distance ∥x−z∥2 (bottom 90%). (d) Removed trajectories (top 10%) discarded by truncation. (e) Re-MeanFlow trained on the untruncated rectified couplings in (b), exhibiting residual outliers. (f) ReMeanFlow trained on the truncated couplings in (c), yielding much cleaner one-step transport with outliers largely eliminated.

the complementary roles of the two components: rectification reduces curvature enough to make MeanFlow optimization efficient, while mean-velocity modeling removes the requirement of perfectly straight trajectories.

Distance-based Truncation To illustrate the effect of distance-based truncation, we perform an additional toy experiment shown in Fig. 13. We discard the top 10% of couplings ranked by endpoint distance ∥x − z∥2. Fig. 13d visualizes the retained (truncated) trajectories. While this criterion does not guarantee removing all high-curvature paths, it eliminates most of the “diagonal” couplings that exhibit the highest curvature in the toy setting (as analyzed above). Consequently, Re-MeanFlow trained on the truncated couplings (Fig. 13f) achieves even cleaner one-step generation, removing nearly all outliers compared to training on the untruncated couplings (Fig. 13e).

### E More Qualitative Results

In this section, we present additional selected qualitative samples generated by Re-MeanFlow across all ImageNet resolutions.

#### E.1 ImageNet-642

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

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

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

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

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

- Fig. 14: Selected qualitative results for Re-MeanFlow (NFE=1) on ImageNet 642.

#### E.2 ImageNet-2562

[Figure 187]

[Figure 188]

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

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

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

- Fig. 15: Selected qualitative results for Re-MeanFlow (NFE=1) on ImageNet 2562.

#### E.3 ImageNet-5122

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

- Fig. 16: Selected qualitative results for Re-MeanFlow (NFE=1) on ImageNet 5122.

