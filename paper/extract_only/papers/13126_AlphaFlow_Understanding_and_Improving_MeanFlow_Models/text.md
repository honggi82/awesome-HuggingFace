# arXiv:2510.20771v1[cs.CV]23Oct2025

## ALPHAFLOW: UNDERSTANDING AND IMPROVING MEANFLOW MODELS

#### Huijie Zhang 1,2 ∗ Aliaksandr Siarohin1 Willi Menapace1 Michael Vasilkovsky1 Sergey Tulyakov1 Qing Qu2 Ivan Skorokhodov1 1Snap Inc. 2Department of EECS, University of Michigan

[Figure 1]

MeanFlow

- (FID: 2.46)
- (FID: 3.47)

NFE=1NFE=2

| | |
|---|---|

[Figure 2]

| | | |
|---|---|---|

α-Flow

(FID: 2.95)

[Figure 3]

MeanFlow

[Figure 4]

| | | |
|---|---|---|

α-Flow

(FID: 2.34)

Figure 1: Uncurated samples (seeds 1-8) from the DiT-XL/2 model for MeanFlow Geng et al. (2025a) and α-Flow (our proposed method) produced with 1 (upper) and 2 (lower) sampling steps for ImageNet-1K 2562.

ABSTRACT

MeanFlow has recently emerged as a powerful framework for few-step generative modeling trained from scratch, but its success is not yet fully understood. In this work, we show that the MeanFlow objective naturally decomposes into two parts: trajectory flow matching and trajectory consistency. Through gradient analysis, we find that these terms are strongly negatively correlated, causing optimization conflict and slow convergence. Motivated by these insights, we introduce α-Flow, a broad family of objectives that unifies trajectory flow matching, Shortcut Model, and MeanFlow under one formulation. By adopting a curriculum strategy that smoothly anneals from trajectory flow matching to MeanFlow, α-Flow disentangles the conflicting objectives, and achieves better convergence. When trained from scratch on class-conditional ImageNet-1K 256×256 with vanilla DiT backbones, α-Flow consistently outperforms MeanFlow across scales and settings. Our largest α-Flow-XL/2+ model achieves new state-of-the-art results using vanilla DiT backbones, with FID scores of 2.58 (1NFE) and 2.15 (2-NFE). The source code and pre-trained checkpoints are available on https://github.com/snap-research/alphaflow.

1 INTRODUCTION

Diffusion models (Sohl-Dickstein et al., 2015) have emerged as the leading paradigm for generative modeling of visual data (Dhariwal & Nichol, 2021; Rombach et al., 2022; Brooks et al., 2024). However, their widespread use is limited by slow inference, as generating high-fidelity samples

∗Work done during an internship at Snap Inc.

typically requires a large number of denoising steps. This computational bottleneck has spurred extensive research into designing efficient diffusion-based generators that are able to operate in very few steps while preserving high generation quality (Salimans & Ho, 2022; Sauer et al., 2024; Song et al., 2023; Song & Dhariwal, 2024; Lu & Song, 2025; Geng et al., 2025b; Frans et al., 2025; Geng et al., 2025a).

Early attempts reduce the inference time of diffusion models through distilling a pre-trained multistep model into a few-step one (Salimans & Ho, 2022; Sauer et al., 2024). The subsequent development of consistency models (Song et al., 2023; Song & Dhariwal, 2024; Lu & Song, 2025) enabled training from scratch for few-step generative models. However, a significant performance gap still remains between existing few-step and multi-step diffusion models. The recently introduced MeanFlow framework (Geng et al., 2025a) enables more stable training and better classifier-free guidance (Ho & Salimans, 2022) integration, significantly bridging the gap between few-step and multi-step from-scratch trained diffusion models. Despite its practical success, there still lacks a clear understanding of why MeanFlow performs better, which hinders further improvements and the design of stronger few-step models.

In this work, we provide a deeper understanding of why MeanFlow works, revealing that its training objective can be decomposed into two components: trajectory flow matching and trajectory consistency. Our gradient analysis shows that these two components are strongly negatively correlated during training, leading to instability and slow convergence in joint optimization. We further demonstrate that the previous heuristic adoption of border-case flow matching supervision is crucial: it actually acts as a surrogate loss for trajectory flow matching and mitigates gradient conflict. However, over 75% of MeanFlow’s computation is spent on this border-case supervision, which is not its primary focus. This raises an open question: can we design more efficient techniques to optimize MeanFlow objective, without such computational overhead?

Motivated by these observations, we introduce α-Flow, a new broad family of objectives for few-step flow models. This framework unifies trajectory flow matching, Shortcut Models Frans et al. (2025), and MeanFlow under a single unified formulation. By employing a curriculum learning strategy that smoothly transitions from trajectory flow matching to MeanFlow, α-Flow better disentangles the optimization of trajectory flow matching and trajectory consistency, reduces reliance on border-case flow matching supervision, and achieves better convergence.

By training vanilla DiT-(Peebles & Xie, 2023) models from scratch with α-Flow on class-conditional ImageNet-1K 2562, we obtain consistently stronger performance across both small- and large-scale settings compared with MeanFlow, for both one-step and few-step generation. Our largest DiTXL/2+ model establishes new state-of-the-art results among all from-scratch trained models with the vanilla DiT backbone and training pipeline, achieving FID scores of 2.58 (1-NFE) and 2.15 (2-NFE).

- 2 PRELIMINARIES

Diffusion models and flow matching. Diffusion model (Ho et al., 2020; Song & Ermon, 2019; Rombach et al., 2022) define a forward process that progressively adds noise to a data sample x ∼ pdata(x) over a continuous timestep t ∈ [0,1]. Specifically, given training data, the forward process perturbs x into a noisy version zt = βtx + σtϵ where ϵ ∼ N(0,I), βt and σt are pre-defined scheduler parameters that depend on t, such that z0 = x and z1 = ϵ. Flow matching (Liu et al., 2023; Lipman et al., 2023) is a deterministic alternative that defines the forward process as a straightline path between the noise distribution and the data distribution, setting βt = 1 − t and σt = t. A neural network vθ(zt,t) is trained to model the ground-truth vector field dzt/dt along this trajectory zt by minimizing the objective:

##### [||vθ(zt,t) − vt||2] (1)

##### LFM (θ) = Et,x,z

t

where vt ≜ v(zt,t|x) = dzt/dt x = ϵ − x. To generate a new sample, the probability flow ODE (PF-ODE) dz/dt = vθ(zt,t) is solved from t = 1 to t = 0, starting with an initial value z1 ∼ N(0,I).

One primary challenge of diffusion models is the slow sampling speed. To address this, several methods have been proposed to enable high-quality generation with significantly fewer steps.

Consistency model (CM). (Song et al., 2023) enables one-step generation by training a neural network fθ(zt,t) to directly map the noisy input zt to clean samples x. The core idea is to enforce a consistency property at any two nearby timesteps t and s, by minimizing the difference between the model’s output. Depending on the ∆t := t − s, the training objective can be categorized into:

- • Discrete-time Consistency Training (CT) (Geng et al., 2025b; Song et al., 2023; Song & Dhariwal,

2024) minimizes the following discrete time CT loss LCTd

: LCTd

(θ) = Et,s,z

t ∥fθ(zt,t) − fθ− (zs,s)∥22 , (2)

where 0 ≤ s < t ≤ 1, zs = zt − ∆t · v and fθ− := stopgrad(fθ). While smaller values of ∆t reduce the discretization error and improve performance, they might also lead to training instability (Song et al., 2023; Geng et al., 2025b). This necessitates a carefully designed scheduler for ∆t to ensure good performance and stability during training.

- • Continuous-time CT (Lu & Song, 2025; Song et al., 2023) eliminates the discretization error by

the continuous time CT loss LCTc

:

dfθ− (zt,t) dt

fθ⊤(zt,t)

, (3)

(θ) = 2Et,z

LCTc

t

Song et al. (2023) theoretically show that ∇θLCTc

(θ)/∆t. However, estimating dfθ−(zt,t)

(θ) = lim∆t→0 ∇θLCTd

dt relies on the Jacobian-vector product (JVP) operation, which causes potential issues of scalability and efficiency in modern deep learning frameworks (Wang et al., 2025b; Peng et al., 2025).

Consistency trajectory model (CTM). (Kim et al., 2024; Zhou et al., 2025; Frans et al., 2025; Geng et al., 2025a) generalize Consistency Models (CMs) by training a neural network uθ(zt,r,t) to enforce consistency across a trajectory from t to r with 0 ≤ r ≤ t ≤ 1. This allows jumping from any t ∈ (0,1] to any r < t during inference, enabling multi-step generation. To train CTM from scratch:

- • Shortcut model (Frans et al., 2025) enforces consistency by ensuring that a single ”shortcut” step from t to r is consistent with two consecutive shortcut steps of half the size. The training objective is:

LSC(θ) = E

t,r,zt

∥uθ(zt,r,t) − uθ−(zt,s,t)/2 − uθ− (zs,r,s)/2∥22 , (4) where zs = zt − (t − s) · uθ−(zt,s,t) and s = (t + r)/2.

- • MeanFlow (Geng et al., 2025a) trains the model uθ(zt,r,t) to estimate the mean velocity

t r v(zτ,τ)dτ, with training objective given by:

1

- t−r

2

duθ−(zt,r,t) dt

. (5)

LMF(θ) = E

uθ(zt,r,t) − vt + (t − r)

t,r,zt

2

In practice, MeanFlow significantly outperforms other one/few-step diffusion and flow models. Yet, there has been little analysis explaining why it works so effectively. To shed light on this, we analyze MeanFlow training in the next section.

- 3 ANALYZING MEANFLOW TRAINING

An intriguing aspect of MeanFlow is the noise distribution used during training: Geng et al. (2025a) empirically found that the best results are achieved when setting r = t for 75% of the samples. This might look counter-intuitive, since we are interested in learning the average velocity on a [r,t] interval to perform large trajectory leaps during inference, so why spending the majority of the training computation on fitting this border case that corresponds to vanilla flow matching supervision? In this section, we show that the MeanFlow loss on its own can be interpreted as velocity consistency training with extra flow matching supervision, and analyze the interaction of these two objectives.

1.1

SIM

cos(∇LTFM,∇LTCc) cos(∇LFM ,∇LTCc)

0.0

LFM ratio: 0%

LFM ratio: 0%

0.8

LFM ratio: 75%

AveragevalueLTCc

LFM ratio: 75%

AveragevalueLTFM

0.4

0.0

−0.4

−0.8

−0.1 0 100K 200K 300K 400K

1.0

0 100K 200K 300K 400K

0 100K 200K 300K 400K

(c) LTCc under different LFM′ ratios.

(a) Gradient similarity

(b) LTFM under different LFM′ ratios.

- Figure 2: MeanFlow training analysis. (a) Shows the cosine similarity between the gradients of

two loss pairs (∇LTCc

vs. ∇LFM′) throughout training. (b) Evaluated LTFM when MeanFlow trained with 0% and 75% of LFM′. (c) Evaluated LTCc

##### vs. ∇LTFM and ∇LTCc

when MeanFlow trained with 0% and 75% of LFM′.

- 3.1 UNDERSTANDING THE OBJECTIVE

Through algebraic manipulations, the original MeanFlow loss LMF in Eq. (5) can be rewritten into the following equivalent form (see Section D.1):

LMF(θ) = E

t,r,zt

∥uθ(zt,r,t) − vt∥22

Trajectory flow matching LTFM

+ E

t,r,zt

2(t − r) · u⊤θ (zt,r,t)

duθ−(zt,r,t) dt

Trajectory consistency LTCc

+C,

(6)

where C is a constant independent of θ. In this decomposition, the first term LTFM corresponds to a flow matching loss but with an additional modeling input parameter r, so we refer to it as

trajectory flow matching. The second term LTCc

, denoted as trajectory consistency loss, acts as a (t − r)-reweighted continuous consistency loss 1, but also without a boundary condition (Song et al., 2023). This decomposition highlights that the MeanFlow objective can be interpreted as a consistency (trajectory) model with extra flow matching supervision.

An interesting property of this decomposition is that LTCc

does not have any boundary condition. In comparison, Song et al. (2023) enforces such a condition for vanilla consistency models using a z0-prediction parameterization: without it, the model would quickly converge to a trivial solution (e.g., a constant output). In the MeanFlow case, this collapse does not occur, which suggests that LTFM implicitly provides the boundary condition for LTCc

. We believe that the absence of an explicit boundary condition makes LTCc

easier to optimize and gives it a much larger solution space.

Another important observation here is that trajectory flow matching involves random r ⩽ t, which differs from the r = t case used during training by Geng et al. (2025a). To clarify this distinction, we directly compare trajectory flow matching (LTFM) with vanilla flow matching, which we denote as LFM′ when using the u-prediction parameterization:

LTFM ≜ E

t,r,zt

∥uθ(zt,r,t) − vt∥22 , LFM′ ≜ E

t,r,zt|r=t

∥uθ(zt,r,t) − vt∥22 (7)

Here, LTFM arises from the decomposition of the MeanFlow loss, while LFM′ corresponds to the objective used in Geng et al. (2025a) for joint training. From this formulation, several observations

follow. First, LFM′ is a “part” of LTFM, active only on the p(t,r | r = t) slice of the joint distribution p(t,r). Second, if the network is independent of r, then marginalizing out r yields LTFM ≡ LFM′, reducing the objective to vanilla flow matching.

- 3.2 EMPIRICAL ANALYSIS

With the decomposition in Equation (6), how does LFM′ interact with the two decomposed terms? In this section, we analyze the gradients of these losses and examine how extra LFM′ minimization affects LTFM and LTCc

individually. We conduct detailed experiments by training MeanFlow with

1Similarly to the proof in Remark 10 of Song et al. (2023), one can show that this term is equivalent to minimizing the difference between uθ(zt, r, t) and uθ−(zt−∆t, r, t − ∆t) as ∆t → 0.

the DiT-B/2 (Peebles & Xie, 2023) architecture on ImageNet-1K 2562 (Deng et al., 2009) for 400K steps. Additional experiment settings are in Section E.

We first analyze the training dynamics by measuring the cosine similarity between the gradients ∇LTFM and ∇LTCc

during training. As shown in Figure 2a, these two gradients are strongly negatively correlated, with a similarity typically below −0.4. This reveals that optimizing LTFM and LTCc jointly is inherently difficult. We hypothesize this stems from the fact that LTCc

, without any boundary condition, has a very large optimal solution manifold, compared to LTFM whose manifold is very narrow. Thus the optimization process is getting pulled towards the LTCc

manifold, distracting from reaching a narrow intersection.

Given this gradient conflict, the question arises: why does joint training with LFM′ help? We identify two key reasons: First, as a subset of LTFM, LFM′ directly reduces LTFM. This is empirically confirmed in Figure 2b, where allocating 75% of the training budget to LFM′ significantly lowers the overall LTFM compared to pure MeanFlow training. Second, LFM′ applies only at r = t, where LTCc

than the ∇LTFM gradient. This is demonstrated in Figure 2a, which shows that cos(∇LFM′,∇LTCc

= 0. Consequently, the gradient ∇LFM′ interferes less with ∇LTCc

) is consistently higher than cos(∇LTFM,∇LTCc

), that is strongly negative for more than 95% of the training. Surprisingly, LTCc component doesn’t seem to be affected and can even be lower when allocating 75% of the training budget to LFM′, as shown in Figure 2c. Which again hints at the fact that LTCc

is relatively easy to

optimize, even near the LTFM optimum. In conclusion, our analysis reveals three important observations:

▷ LMF can be decomposed into trajectory flow matching LTFM and trajectory consistency LTCc

objectives, whose gradients are strongly negatively correlated during training.

does not have a necessary boundary condition on its own, implying that LTFM serves as an implicit boundary condition for it.

▷ LTCc

▷ LFM′ acts as a surrogate loss for LTFM, but with significantly less gradient conflict with the

Trajectory consistency loss LTCc

.

- 4 α-FLOW MODELS

As we showed in the previous section, the LTFM loss is difficult to optimize jointly with the LTCc

.

While the introduction of the LFM′ loss serves as an effective surrogate for optimizing LTFM, this approach dedicates a significant portion of training to an objective that is not of our primary interest.

This raises a key question: Can we more efficiently optimize LTFM when optimizing LMF without this computational overhead? To answer this, we introduce our α-Flow loss, a new family of training objectives for flow-based models.

- 4.1 α-FLOW: UNIFYING ONE, FEW, AND MANY-STEP FLOW-BASED MODELS Definition 1. The α-Flow loss Lα is defined as:

α−1 · ∥uθ(zt,r,t) − (α · v˜s,t + (1 − α) · uθ−(zs,r,s))∥22 , (8)

Lα(θ) ≜ E

t,r,zt

where t,r ∈ [0,1] is the start and end timestep, s is the intermediate timestep: s = α · r + (1 − α) · t,α ∈ (0,1] is the consistency step ratio, and zs = zt + (t − s) · v˜s,t is the trajectory value at this timestep s. Here, v˜s,t is the “shift velocity” used to estimate the intermediate variable zs from zt.

The α-Flow loss is visualized in Figure 3e. Intuitively, it enforces trajectory consistency between t and r by introducing an additional s, which is an interpolation between t,r with ratio α. More importantly, this definition generalizes previously introduced training objectives such as trajectory flow matching, Shortcut Model training, and MeanFlow training:

Theorem 1. The α-Flow loss unifies flow matching, Shortcut Models, and MeanFlow:

##### ▷ LTFM(θ) = Lα=1(θ) with v˜s,t = vt.

(a) Discrete CT (b) Continuous CT (c) Shortcut Model (d) MeanFlow (e) α-Flow

- Figure 3: Comparison of training trajectories for various few-step diffusion and flow-based models.

▷ LSC(θ) = 12Lα=1/2(θ) with v˜s,t = uθ−(zt,s,t).

▷ ∇θLMF(θ) = ∇θLα→0(θ) with v˜s,t = vt.

Moreover, if one considers a z0-parametrized network uθ(zt,0,t) = (zt − fθ(zt,t))/t = zˆ0, Lα incorporates discrete and continuous consistency training as well. Specifically, with v˜s,t = vt and r ≡ 0:

With uθ(zt,0,t) = (zt − fθ(zt,t))/t = zˆ0, v˜s,t = vt and r ≡ 0:

- Algorithm 1 α-Flow: Training.

# fn(z, r, t): function to predict u # x: training batch, k: training iterations

t, r = sample t r() alpha = sample alpha(k) s = alpha * r + (1 - alpha) * t e = randn like(x)

zt = (1 - t) * x + t * e v = e - x

if alpha == 0: u, dudt = jvp(fn, (zt, r, t), (v, 0, 1)) u_tgt = v - (t - r) * dudt

else : u = fn(zt, r, t) zs = zt - (t - s) * v u_tgt = alpha * v + (1 - alpha) * fn(zs, r, s)

error = u - stopgrad(u_tgt) loss = metric(error)

- Algorithm 2 α-Flow: Curriculum Schedule

(θ) = Lα=δ(θ) for δ ∈ (0,t).

▷ LCTd

(θ) = ∇θLα→0(θ).

▷ ∇θLCTc

This theorem reveals that the ratio α is the key hyperparameter that unifies seemingly different methods, which controls the relative position of the intermediate timestep s within the (r,t) interval. By annealing α from 1 to 0, we obtain a family of models in the interpolation between trajectory flow matching and MeanFlow. Notably, discrete CT is a special case of α-Flow with r ≡ 0. Unlike discrete CT, α-Flow requires no complex timestep partitioning: once t and r are sampled, s is immediately determined with a fixed α.

- 4.2 α-FLOW MODELS

# k_s, k_e: start/end schedule iterations, # gamma: temperature parameter # k: current iteration, eta: clamping value

The α-Flow loss enables a curriculum learning strategy that progressively transitions from the trajectory flow matching to MeanFlow objective. This approach better disentangles the optimization of the trajectory flow matching and consistency losses, could potentially reduce reliance on the flow matching objective, and leads to better convergence. The detailed curriculum learning can be summarized into three phases:

scale = 1 / (k_e - k_s) offset = - (k_s + k_e) / 2 / (k_e - k_s) alpha = 1 - sigmoid((scale * k + offset) * gamma) alpha = 1 if alpha > (1- eta) else (0 if alpha < eta

else alpha)

- • Trajectory flow matching pretraining (α = 1). To speed-up convergence toward narrow LTFM manifold, we prioritize optimizing trajectory flow matching in the early training phase. Additionally, as a low-variance objective, trajectory flow matching quickly establishes a reliable noise-todata mapping, providing a good initialization for subsequent few-step refinement. Notably, this pretraining strategy is aligned with previous diffusion model pretraining strategy applied on consistency model (Geng et al., 2025b), while we start from different motivations and generalize it into the α-Flow framework.
- • α-Flow transition (α ∈ (0,1)). Once the model has a solid foundation, we transition from trajectory flow matching to the MeanFlow objective. We accomplish this with a curriculum learning

- approach where we progressively decrease the consistency step ratio from 1 to 0. This gradual shift is inspired by the discrete CT (Song et al., 2023). It effectively transitions from the “high-bias, low-variance” objective to the “high-variance, low-bias one”, leading to improved convergence.
- • MeanFlow fine-tuning (α → 0). In the final stage, we focus entirely on the MeanFlow training objective. Unlike the original paper, our improved early-stage optimization of trajectory flow matching significantly reduces the need for the flow matching loss (as shown in Table 2 (b)) and achieves significantly better few-step generation quality.

The overall training code of α-Flow is shown in Algorithm 1, where we first sample t,r and obtain the α from the schedule. Based on whether α = 0 or not, α-Flow will use either LMF or Lα to train the model. α-Flow applies the same training details as MeanFlow when training LMF (except a lower ratio of flow matching). Below, we only show the difference: the schedule of α as well as the design space of Lα when α > 0.

Schedule. To schedule the training, we use a sigmoid function, α = Sigmoidk

s⇒ke,γ,η (k), which depends on the training iteration k. The function is defined by its starting and ending iterations, ks,ke, a temperature parameter γ (set to be 25) and a clamping value η. The specific implementation can be found in Algorithm 2. Figure 5 provides a visualization of this scheduler, while Section 5.2 conducts an ablation study over its parameters.

Clamping value. Geng et al. (2025b) show that when ∆t = t − s approaches 0, the performance of few-step CT model will first increase and then decrease. For α-Flow, we observe a similar phenomenon: by training α-Flow with a fixed α, as α approaches 0, the 1-step generation performance will first increase then decrease. Detailed experiments are shown in Table 5 (c). From the experiment, the optimal performance is achieved when α = 5 × 10−3. Thus, we set a clamping value η = 5×10−3 for the schedule. α will be set to 0 when α < η. We also use the same clamping value to set α to 1 when α > 1 − η, as when α is close to 1, LTFM is similar to Lα but more efficient.

Training objective. In the unifying space of α-Flow loss, all other few-step models set v˜s,t = vt except the shortcut model which uses v˜s,t = uθ−(zt,s,t). Additionally, we are interested in seeing whether we need exponential moving average (EMA) for θ−. With ablation study in Table 5 (a), we set v˜s,t = vt and do not use EMA for θ−.

Adaptive loss weight. MeanFlow (Geng et al., 2025a) demonstrates the effectiveness of adaptive loss. Basically, let ||∆||22 denote the squared L2 loss. The adaptive loss weight ω = 1/(||∆||22 + c) where c = 10−3. And the adaptively weighted loss is sg(ω)||∆||22. Theoretically, we derived an equivalent adaptive loss weight ω = α/(||∆||22 + c) for Lα. We defer the derivation in Section G.2. With ablation study in Table 5 (b), we demonstrate the derived adaptive loss weight is better than other loss weights.

Classifier-free guidance (CFG). We apply a similar CFG training strategy as MeanFlow, by setting v˜s,t in Equation (8) with v˜s,t = w · v(zt,t|x) + κ · uθ− (zt,t,t|c) + (1 − w − κ) ·

- uθ− (zt,t,t|∅), where w,κ are the guidance scale, uθ− (·|c), uθ− (·|∅) denotes the class-condition (with class c) and class-unconditional prediction. Detailed settings of w,κ are deferred to Section F.

Sampling. We employ both consistency sampling (Song et al., 2023) and ODE sampling for twostep generation. Implementation details are provided in Algorithm 3. Empirically, we observe that consistency sampling outperforms ODE sampling for larger models with better convergence. Consequently, we adopt ODE sampling for all DiT-B/2 architectures and consistency sampling for all DiT-XL/2 architectures, with additional ablation studies on DiT-XL/2 presented in Figure 4.

- 5 EXPERIMENTS

In this section, we employ α-Flow on real image datasets ImageNet-1K 2562 Deng et al. (2009). We use exactly the same DiT Peebles & Xie (2023) architecture as MeanFlow Geng et al. (2025a). For evaluation, we use Fr´echet Inception Distance (FID) Heusel et al. (2017), Fr´echet DINOv2 Oquab et al. (2023). We evaluate model performance for both 1 and 2 Number of Function Evaluations (NFE=1, NFE=2). We implement our models in the latent space of the Stable Diffusion Variational Autoencoder (SD-VAE) 2. More details on the experiments settings are in Section F.

2The EMA version in https://huggingface.co/stabilityai/sd-vae-ft-mse

NFE 1 NFE 2 FID FDD FID FDD FID†

Method Source Params Epochs

Shortcut-XL/2 Frans et al. (2025) 675M 160 10.60 – – – – IMM-XL/2 Zhou et al. (2025) 676M 3840 8.05 – 3.88 – – MeanFlow-XL/2 Geng et al. (2025a) 676M 240 3.43 – 2.93 – – MeanFlow-XL/2+ Geng et al. (2025a) 676M 1000 – – 2.20 – – FACM-XL/2 Peng et al. (2025) 675M 800 + 250 × 2 – – – – 2.07

FACM-XL/2

675M 120 × 2 9.54 410.4 7.31 362.0 – FACM-XL/2 675M 240 × 2 6.59 327.7 4.73 278.6 – MeanFlow-B/2 131M 240 6.04 312.3 5.17 232.1 – MeanFlow-XL/2 676M 240 3.47 185.8 2.46 108.7 2.26

Our reproduction

α-Flow-B/2

131M 240 5.40 287.1 5.01 231.8 –

α-Flow-XL/2 676M 240 2.95 164.6 2.34 105.7 2.16 α-Flow-XL/2+ 676M 240+60 2.58 148.4 2.15 96.8 1.95

Our methods

Table 1: Class-conditional generation on ImageNet-256×256. The table reports the results for few-step diffusion/flow matching-based methods trained from scratch. ”×2” indicates that FACM requires roughly twice the computation per epoch compared to other methods. For a direct ”epochto-epoch comparison,” α-Flow-XL/2, MeanFlow-XL/2 and FACM-XL/2 are each trained for 240 epochs. α-Flow-XL/2+ is a fine-tuned version of α-Flow-XL/2, trained for extra 60 epochs with a batch size of 1024. † FID scores are evaluated with the balanced class sampling (see Section I).

NFE 1 NFE 2 FID FDD FID FDD Constant0.0 44.4 844.1 42.1 836.3 Trajectory flow matching iterations

Model NFE 1 NFE 2

Schedule

% r = t Schedule FID FDD FID FDD 0% Constant0.0 46.0 879.6 44.3 867.7

Sigmoid0K→400K 40.4 822.5 38.9 811.8 25% Constant0.0 44.4 844.1 42.1 836.3

Sigmoid0K→100K 44.3 860.3 40.8 826.9 Sigmoid50K→150K 44.1 846.8 39.9 811.6 Sigmoid100K→200K 42.4 828.0 38.3 795.4 Sigmoid150K→250K 41.3 818.8 38.1 793.1 Transition iterations

Sigmoid0K→400K 40.0 785.4 37.1 782.9 50% Constant0.0 43.9 844.1 42.1 836.3

Sigmoid0K→400K 40.2 781.0 37.1 775.0 75% Constant0.0 43.1 819.2 38.5 787.6

Sigmoid200K→200K 41.4 794.4 38.8 796.7 Sigmoid150K→250K 41.3 818.8 38.1 793.1 Sigmoid0K→400K 40.0 785.4 37.1 782.9

Sigmoid0K→400K 42.2 810.5 36.2 754.7

(b) Flow matching ratio. Table 2: Ablation study on ImageNet-1K 2562 for α-Flow-B/2.

(a) Consistency step ratio schedule.

- 5.1 COMPARISON WITH BASELINE

In Table 1, we compare α-Flow with previous few-step Diffusion and Flow models, demonstrating its superior performance for 1-NFE and 2-NFE generation. Across models trained for 240 epochs, α-Flow-XL/2 achieves 2.95 FID (164.6 FDD), representing a relative improvement of 15% (12%) over MeanFlow-XL/2 and 70% (60%) over FACM-XL/2. Our best model, α-Flow-XL/2+, sets a new state-of-the-art 1-NFE generation with an impressive FID of 2.58 (148.4 FDD), compared with all the other few-step Diffusion and Flow models trained over the SD-VAE. Furthermore, for 2-NFE generation, α-Flow-XL/2+ achieves 2.15 FID (96.8 FDD), outperforms all these baseline methods. It’s particularly notable that it surpasses FACM-XL/2’s 2.07 FID (achieved with a class-balanced sampling) by reaching 1.95 FID with only 23% of the training epochs. Uncurated samples, shown in Figure 1 and Section K, visually confirm these results. Specifically in Figure 1, α-Flow-XL/2 generates more images with better quality, as highlighted in green.

- 5.2 ABLATION STUDY

Consistency step ratio schedule. In Table 2 (a), we evaluate our α-Flow framework trained with various sigmoid schedules, as visualized in Figure 5. For these experiments, the flow matching ratio is fixed at 25%. We first analyze the impact of the trajectory flow matching pretraining du-

FID MeanFlow DiT-XL/2

MeanFlow DiT-XL/2 (CS)

α-Flow DiT-XL/2

α-Flow DiT-XL/2 (CS)

3.2

α-Flow DiT-XL/2+

α-Flow DiT-XL/2+ (CS)

3.0

2.8

2.6

2.4

- 1.00 0.75 0.50 0.25 0.00 Intermediate Timestep for NFE=2

- 2.2

- Figure 4: Comparing ODE vs consistency sampling for MeanFlow and α-Flow models.

1.00

Constant0.0

Sigmoid0→100K

0.75

Sigmoid50K→150K

αvalue

Sigmoid100K→200K Sigmoid150K→250K

0.50

0.25

0.00

1.00

Sigmoid200K→200K Sigmoid150K→250K Sigmoid0→400K

0.75

αvalue

0.50

0.25

0.00

0 50K 100K 150K 200K 250K 300K 350K 400K

Training Iteration

Figure 5: Visualization of consistency step ratio schedule.

ration. By fixing ke − ks to 100K iterations, we progressively increase ks from 0K to 150K. As the pretraining duration increases, α-Flow’s performance consistently improves across all metrics.

The best-performing schedule, Sigmoid150K→250K, significantly outperforms the baseline MeanFlow (Constant0.0). This suggests that optimizing trajectory flow matching is more crucial than optimizing MeanFlow in the early training stages for achieving superior few-step flow modeling. This finding aligns with our empirical analysis, which shows that because the gradients of the trajectory flow matching and consistency losses conflict, it is more efficient to exclusively optimize the trajectory flow matching objective for faster initial convergence.

Next, we investigate the effect of the transition duration. With the midpoint (ks + ke)/2 fixed at 200K iterations, we vary the total transition iterations from 0 to 400K. Our results indicate that a longer, smoother transition leads to better generation quality. This highlights the importance of gradually reducing the bias of the training objective by smoothly transitioning between trajectory flow matching and MeanFlow.

Flow matching ratio. In Table 2 (b), we compare our α-Flow framework with the MeanFlow baseline across various flow matching ratios (%r = t). Our results show that α-Flow consistently outperforms MeanFlow for all evaluated ratios, confirming the effectiveness of our proposed method. A key finding is that α-Flow achieves its best 1-NFE performance at a relatively low flow matching ratio. Specifically, it reaches the best FID of 40.0 at 25 % of r = t and the best FDD of 781.0 at 50 % of r = t, while MeanFlow requires a higher ratio of 75% to achieve its best FID of 43.1 and FDD of 819.2. This aligns with our motivation: by pretraining on trajectory flow matching, α-Flow is less reliant on the flow matching objective and can focus more on the overall MeanFlow objective, leading to superior one-step generation quality.

Furthermore, we observe that for α-Flow, the flow matching ratio presents a clear trade-off between 1-NFE and 2-NFE performance. For instance, the 75% ratio yields worse NFE=1 but better NFE=2 generation results compared to the 50%-ratio version. This indicates that a higher proportion of flow matching improves the model’s ability to generate images in a slightly higher number of steps.

Sampling. As shown in Figure 4, we compare ODE sampling (solid line) and consistency sampling (dotted line) for 2-NFE generation across different intermediate sampling timesteps, using MeanFlow-XL/2, α-Flow-XL/2, and α-Flow-XL/2+. The results show that consistency sampling yields better generation performance for both α-Flow-XL/2 and α-Flow-XL/2+, achieving the best FID scores of 2.09 at timestep 0.4 and 2.28 at timestep 0.45, respectively. In contrast, ODE sampling performs better for MeanFlow-XL/2, which attains its best FID of 2.39 at timestep 0.35. In Table 1, we select intermediate sampling timesteps that balance FID and FDD; see Table 3 for details.

- 6 CONCLUSION

Our work provided a principled analysis of the MeanFlow framework, analyzing its objective and establishing the necessity of flow matching supervision during training. Motivated by this understanding, we proposed the α-Flow objective as a generalization of MeanFlow loss, allowing us to train consistently stronger few-step image generation models from scratch.

- 7 REPRODUCIBILITY STATEMENT

We are committed to ensuring the reproducibility of our results. To this end, we include all the necessary implementation details in Section F, ensuring that our methodology can be faithfully reproduced. We will publicly release our source training, inference, and evaluation code, as well as the pre-trained checkpoints for ImageNet-1K 2562.

REFERENCES

Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022.

David Berthelot, Arnaud Autef, Jierui Lin, Dian Ang Yap, Shuangfei Zhai, Siyuan Hu, Daniel Zheng, Walter Talbott, and Eric Gu. Tract: Denoising diffusion models with transitive closure time-distillation. arXiv preprint arXiv:2303.04248, 2023.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=OlzB6LnXcS.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025a.

Zhengyang Geng, Ashwini Pokle, Weijian Luo, Justin Lin, and J Zico Kolter. Consistency models made easy. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=xQVxo9dSID.

Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, et al. Splitmeanflow: Interval splitting consistency in few-step generative modeling. arXiv preprint arXiv:2507.16884, 2025.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24174–24184, 2024.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=ymjI8feDTD.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Tuomas Kynk¨a¨anniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. The role of imagenet classes in fr\’echet inception distance. arXiv preprint arXiv:2203.06026, 2022.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=LyJi5ugyJx.

Martin Marek, Sanae Lotfi, Aditya Somasundaram, Andrew Gordon Wilson, and Micah Goldblum. Small batch size training for language models: When vanilla sgd works, and why gradient accumulation is wasteful. arXiv preprint arXiv:2507.07101, 2025.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Flow-anchored consistency models. arXiv preprint arXiv:2507.03738, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103. Springer, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/ forum?id=St1giarCHLP.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=WNzy9bRDvG.

Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32, 2019.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/ forum?id=PxTIG12RRHS.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 32211–32252. PMLR, 23–29 Jul 2023.

Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025.

Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37:83951–84009, 2024.

Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. 2025a.

Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective. arXiv preprint arXiv:2509.04394, 2025b.

Ling Yang, Zixiang Zhang, Zhilong Zhang, Xingchao Liu, Minkai Xu, Wentao Zhang, Chenlin Meng, Stefano Ermon, and Bin Cui. Consistency flow matching: Defining straight flows with velocity consistency. arXiv preprint arXiv:2407.02398, 2024.

Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024a.

Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024b.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=DJSZGGZYVi.

Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/ forum?id=pwNSUo7yUb.

Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

A RELATED WORK

Diffusion Models. Diffusion models have become a dominant paradigm in generative modeling for vision domains (Sohl-Dickstein et al., 2015; Song & Ermon, 2019; Ho et al., 2020; Song et al., 2021a;b; Dhariwal & Nichol, 2021). The classical diffusion framework defines a forward noising process and a corresponding reverse process that a model learns to approximate. Early works such as DDPM (Ho et al., 2020) and score-based generative modeling (Song & Ermon, 2019) demonstrated high-quality image generation, later extended to continuous-time SDEs and ODEs (Song et al., 2021b). (Dhariwal & Nichol, 2021) further improved sample fidelity with larger architectures and classifier guidance. More recently, the community has explored flow-based parameterizations that directly learn continuous velocity fields (Liu et al., 2023; Lipman et al., 2023; Albergo & VandenEijnden, 2022). These flow matching approaches simplify training, unify score- and likelihoodbased models, and are used in large-scale systems such as Stable Diffusion 3 (Esser et al., 2024).

Few-step Diffusion. Despite their quality, diffusion models are computationally expensive due to iterative sampling. A large body of work accelerates sampling to a few steps or even one step. Distillation-based approaches include progressive distillation (Salimans & Ho, 2022; Berthelot et al.,

- 2023), and often incorporate adversarial objectives (Yin et al., 2024b;a; Zhou et al., 2024; Sauer et al., 2024). UCGM (Sun et al., 2025) develops a unified training scheme for multi-step and fewstep diffusion-based methods.

- A closer research direction (which our method follows as well) includes the methods which are trained from scratch and support few- and even one-step generation by design. Consistency Models (CMs) (Song et al., 2023) learn to map noisy inputs directly to clean data by enforcing selfconsistency. Extensions improve stability and scalability (Song & Dhariwal, 2024; Lu & Song, 2025; Geng et al., 2025b). Trajectory-based methods learn the dynamics of the entire denoising process, enabling arbitrary jumps along the diffusion path. PCM (Wang et al., 2024) scale consistency distillation to large scale models and optimize with preselected time intervals. Shortcut diffusion models (Frans et al., 2025) learn direct mappings with shortcut constraints. MeanFlow (Geng et al., 2025a) predicts time-averaged velocities with continuous consistency, while Guo et al. (2025) explore this idea for discrete consistency. Hybrid approaches combine consistency and flow matching: Consistency-FM (Yang et al., 2024) enforces velocity self-consistency, FACM (Peng et al., 2025) anchors consistency to flow objectives, and IMM (Zhou et al., 2025) matches the output distributions via moment matching instead of exact outputs. Consistency Trajectory Models (CTM) (Kim et al.,

2024) generalize consistency training to support transitions between any two timesteps, combining one-step generation with progressive refinement. Transition Models (TiM) (Wang et al., 2025a) derive an exact continuous-time dynamics equation for arbitrary-step transitions. These methods achieve one- to few-step sampling with steadily improving fidelity.

- B LIMITATIONS

- • Our α-Flow loss enables high-quality training of discrete MeanFlow models without requiring JVP computation. However, in practice, the continuous objective (i.e., setting α → 0) remains important, likely due to the bias–variance trade-off inherent in the consistency objective (Song et al., 2023; Song & Dhariwal, 2024).
- • We occasionally observed unstable training in large-scale models with guidance integration, both for the vanilla MeanFlow model and our α-Flow variant. Thus, our framework should not be viewed as a silver bullet for addressing the well-known instability issues of consistency models Geng et al. (2025b).
- • The α-Flow objective uses pure flow matching supervision up to ks iterations, after which the consistency objective is applied. Before this point, the model’s few-step performance is weak, which can make progress harder to monitor.
- • Our gradient analysis provides actionable insights but remains empirical; it does not fully explain, from a theoretical perspective, why flow matching is so critical for consistency.
- • Although we motivate larger batch sizes for fine-tuning by the high variance of the consistency loss, the observed improvements (see Table 4) may instead reflect that small batches are more

sensitive to hyperparameters (Marek et al., 2025), and that beyond a certain size, batch-size scaling exhibits diminishing returns (McCandlish et al., 2018).

- C FAILED EXPERIMENTS

We also wish to share with the community several experiments that did not succeed during the course of this project. Some of these directions were likely underexplored on our side, while others may represent genuine dead-ends. Nevertheless, we believe documenting them may serve as a useful reference for future work.

- • We devoted several weeks to exploring decomposed training of the MeanFlow objective with individually tuned weighting functions for each term, drawing inspiration from EDM Karras et al.

(2022) to map out the design space. Unfortunately, every configuration we attempted produced worse results than the default adaptive loss heuristic, which was a particularly frustrating outcome.

- • Consistency sampling (see Figure 4) did not provide the improvements we had anticipated. Interestingly, the optimal midpoint consistently emerged at ≈ 0.5, which coincides with the default MeanFlow setting. We suspect this effect is related to the training distribution, which has a mode slightly lower 0.5. Following the original work, we employed a logit-normal distribution with location parameter −0.4.
- • We experimented with LoRA fine-tuning and introduced separate prediction heads for vanilla velocity and mean velocity. Neither approach yielded promising results.
- • We conducted roughly 50 ablations on the train-time noise schedule for vanilla MeanFlow models. None resulted in noticeably better performance, even when factorizing the joint distribution p(t,r) into p(t)p(r|t) and exploring alternative supervision distributions for flow matching in parallel.
- • We investigated additional representation alignment losses Yu et al. (2025) with the aim of accelerating convergence in MeanFlow models. However, the observed gains were insufficient to justify the added complexity of the training framework.
- • We also experimented with different EMA schedules, but these attempts did not lead to meaningful improvements.

- D PROOFS OF THINGS

- D.1 LOSS DECOMPOSITION Proof. The MeanFlow loss is given by:

LMF(θ) = Et,r,z

t

uθ(zt,r,t) − vt + (t − r)

duθ−(zt,r,t) dt

2

2

(9)

(unpacking the norm and regrouping terms yields)

= Et,r,z

t ∥uθ(zt,r,t) − vt∥22

LTFM(θ)

+Et,r,z

t

2 · (t − r) · u⊤θ (zt,r,t)

duθ−(zt,r,t) dt

LTCc(θ)

(10)

+ Et,r,z

t −2(t − r) · v⊤(zt,t|x)

duθ−(zt,r,t) dt

+ (t − r)2

duθ−(zt,r,t) dt

2

2 Does not depend on θ

(11)

| |
|---|

- D.2 Lα LOSS UNIFICATION Proof of theorem 1. The proof for flow matching and shortcut models is straightforward. We will

∆t t − r

only show the proof for the third bullet point. For brevity, let’s set ∆t = t − s and α =

.

t − r ∆t · uθ(zt,r,t) −

∆t

##### Lα(θ) = Et,r,z

t − r · vt− t − ∆t − r t − r

t

2

uθ− (zt−∆t,r,t − ∆t)

,

2

- (=i) Et,r,z

t

t − r ∆t · uθ(zt,r,t) −

∆t t − r · vt −

t − ∆t − r t − r ·

uθ− (zt,r,t) −

duθ−(zt,r,t) dt

∆t + O ∆2t

2

2

,

- (=ii) Et,r,z

(12)

t − r ∆t · uθ(zt,r,t) − uθ−(zt,r,t) −

∆t t − r·

t

2

duθ−(zt,r,t) dt − uθ−(zt,r,t) + O ∆2t

vt − (t − r)

,

2

where (i) uses the Taylor expansion over uθ− (zt−∆t,r,t − ∆t):

duθ−(zt,r,t) dt

∆t + O ∆2t ,

uθ− (zt−∆t,r,t − ∆t) = uθ− (zt,r,t) −

duθ−(zt,r,t) dt

and (ii) uses the fact that

∆2t = O ∆2t . Thus, lim

∇θLα(θ) = lim

∇θLα(θ)

α→0

∆t→0

t − r ∆t · ∇⊤θ uθ(zt,r,t) · uθ(zt,r,t) − uθ−(zt,r,t) −

∆t t − r· vt − (t − r)

Et,r,z

2 ·

= lim

t

∆t→0

duθ−(zt,r,t) dt − uθ−(zt,r,t) + O ∆2t ,

t − r ∆t · ∇⊤θ uθ(zt,r,t) · −

∆t

Et,r,z

2 ·

t − r · vt − (t − r)

= lim

t

∆t→0

duθ−(zt,r,t) dt − uθ−(zt,r,t) + O ∆2t ,

duθ−(zt,r,t) dt

t − r ∆t ·

∆t t − r · ∇⊤θ uθ(zt,r,t) · uθ−(zt,r,t) − vt + (t − r)

=Et,r,z

2 ·

,

t

=∇θLMF(θ),

(13)

| |
|---|

Proof of equivalence with consistency model. By setting v˜s,t = vt,r = 0 and uθ(zt,0,t) = (zt − fθ(zt,t))/t, ∆t = t − s and α =

∆t t

, we have:

t ∆t · uθ(zt,r,t) −

∆t

##### Lα(θ) = Et,r,z

t − r · vt− t − ∆t − r t − r

t

2

uθ− (zt−∆t,r,t − ∆t)

,

2

zt − fθ(zt,t)

∆t

t ∆t ·

⇒(i) Et,z

(14)

t −

t · vt− t − ∆t t

t

2

zt−∆t − fθ−(zt−∆t,t − ∆t) t − ∆t

,

2

1 t∆t · ∥fθ(zt,t) − fθ−(zt−∆t,t − ∆t)∥22

(iii=) LCTd

(=ii) Et,z

(θ),

t

where (i) plug in the reparameterization and r = 0, (ii) uses the fact that zt = zt−∆t+∆t·vt. Thus Lα(θ) could be reparameterized to LCTd

1 t∆t

. Since the discrete CT uses timestep partition to determine t and ∆t, the (iii) holds for a special timestep partition when ∆t = α · t given a fixed α. From Theorem 2 in Lipman et al. (2023), because uθ−(zt,r,t) is independent of θ, we have:

(θ) with a loss weighting function

##### LMF(θ) = Et,r,z

t

- uθ(zt,r,t) − vt + (t − r)

duθ−(zt,r,t) dt

2

2

,

= Et,r,z

t

- uθ(zt,r,t) − v(zt,t) + (t − r)

duθ−(zt,r,t) dt

2

2

with C a constant independent of θ. Thus,

+ C,

(15)

∇θLMF(θ) = Et,r,z

duθ−(zt,r,t) dt

2 · ∇⊤θ uθ(zt,r,t) · uθ(zt,r,t) − v(zt,t) + (t − r)

t

zt − fθ(zt,t)

−fθ(zt,t) t ·

⇒(i) Et,z

2 · ∇⊤θ

t − v(zt,t)+ v(zt,t) −

t

dfθ− (zt,t) dt −

(zt − fθ−(zt,t)) t

,

dfθ− (zt,t) dt

1 t · ∇⊤θ fθ(zt,t)

= Et,z

2 ·

= ∇θLCTc

(θ),

t

where (i) plug in the reparameterization and r = 0, and use the fact that:

,

(16)

duθ−(zt,r = 0,t) dt

1 t2

=

dfθ− (zt,t) dt − (zt − fθ−(zt,t)) (17)

t v(zt,t) −

1 t

Thus ∇θLMF(θ) could be reparameterized to LCTc

(θ) with a loss weighting function

.

| |
|---|

- E ANALYSIS DETAILS

The detailed implementation of DiT-B/2 is provided in Table 3, where we adopt the DiT-B/2-non-cfg setting. For loss evaluation, at each checkpoint we use a batch size of 128 and run 1000 iterations to compute the mean loss along with its 5% and 95% percentiles, which are reported in the figure. To measure the cosine similarity between different losses, we calculate ∇LTFM, ∇LFM′, and ∇LTCc

on the same batch and then compute their pairwise cosine similarities. This procedure is also repeated over 1000 iterations to obtain the mean similarity and its 5% and 95% percentiles, as shown in the figure.

- F IMPLEMENTATION DETAILS Implementation details are shown in Table 3.
- G ADDITIONAL EXPERIMENTS

- G.1 ABLATION STUDY OVER BATCH SIZE

Training diffusion/flow-based models can be challenging due to the high variance of their gradients. Past research Zhou et al. (2025); Karras et al. (2022; 2024) often used large batch sizes (1024 or even 4096) to mitigate this issue. In this section, we fine-tune a MeanFlow-XL/2 model (with implementation details in Table 3) for an additional 60 epochs using a large batch size.

Table 3: Configurations on ImageNet 256×256. B/2-non-cfg is our ablation and analysis model in the main text.

Configs DiT-B/2-non-cfg DiT-B/2 DiT-XL/2 DiT-XL/2+ Network Architectures Params (M) 131 131 676 676 FLOPs (G) 23.1 23.1 119.0 119.0 Depth 12 12 28 28 Hidden dim 768 768 1152 1152 Heads 12 12 16 16 Patch size 2×2 2 × 2 2 × 2 2 × 2 Training hyperparameters Training steps 400K 1.2M 1.2M 1.2M Batch size for training 256 256 256 256 Fine-tuning steps – – – 75K Batch size for fine-tuning – – – 1024 Dropout 0.0 Optimizer Adam Kingma & Ba (2014) lr schedule constant lr 0.0001 Adam (β1, β2) (0.9, 0.95) Weight decay 0.0 EMA half-life 6931 Gradient clipping norm 16 Autoencoder used sd-vae-ft-ema α-Flow hyperparameters Ratio of r = t Table 2 (b) 25% 50% 50% (r, t) sampler logitnorm(–0.4, 1.0)

v˜s,t Table 5 (a) vt vt vt Whether to use EMA for uθ− Table 5 (a) No No No Adaptive weight Table 5 (b) ω = α/ ||∆||22 + c

Schedule of α γ 25 25 25 ks Table 2 (b)

0 600K 600K ke 1.2M 1M 1M η Table 5 (c) 5 × 10−3 5 × 10−3 5 × 10−3

CFG training w – 1.0 0.2 0.2 κ – 1.0 0.92 0.92 CFG triggered if t is in – [0.0, 1.0] [0.0, 0.75] [0.0, 0.75] Whether use EMA for CFG – No No No

2-NFE Sampling Method ODE ODE consistency consistency Intermediate timestep 0.5 0.5 0.55 0.5

As shown in Table 4, a batch size of 512 achieved the best 1-NFE FID of 3.05 and FDD of 164.3. A batch size of 1024, however, yielded the best FDD of 93.4. Overall, a batch size of 1024 performed well across all metrics, so we designate this configuration as MeanFlow-XL/2+. The same setting is applied to fine-tune the MeanFlow-XL/2 model, leading to the MeanFlow-XL/2+ results in Table 1. Our proposed α-Flow-XL/2+ model outperforms MeanFlow-XL/2+ in several key metrics: 1-NFE FID (2.58 vs. 3.06), 1-NFE FDD (148.4 vs. 165.7) and 2-NFE FID (2.15 vs. 2.16), only worse in 2-NFE FDD (96.8 vs. 93.4). These results demonstrate the overall effectiveness of our α-Flow method. Notably, the results in Table 4 are obtained using labels sampled from the ImageNet dataset distribution, whereas the results in Table 1 use randomly generated labels. In general, sampling labels from the ImageNet distribution leads to lower FID scores compared to using random labels.

NFE 1 NFE 2 FID FDD FID FDD

Batch Size

Algorithm 3 α-Flow: Sampling

# 1 = t1 > t2 > ... > tN = 0 :sequence of

256 3.13 167.2 2.31 97.1 512 3.05 164.3 2.21 95.2 1024 3.06 165.7 2.16 93.4 2048 3.29 169.6 2.10 96.6 4096 3.13 168.9 2.16 95.1

timesteps z = randn like(x) for n in range(N):

m = n + 1 if consistency sampling:

z = z - tn * fn(z, r=0, t=tn) z = z + tm * randn like(x)

elif ODE sampling: z = z - (tn - tm) * fn(z, r=tm, t=tn)

Table 4: Ablation study over the fine-tuning batch size using the data distribution over class labels.

v˜s,t uθ− FID FDD

uθ− EMA 188.1 1761.6 uθ− Non-EMA 319.0 4009.9

vt EMA 202.8 1832.3 vt Non-EMA 59.2 964.6

(a) Reformulate the training objective.

Loss weight FID FDD ω = 1 59.2 964.6 ω = 1/ ||∆||22 + c 0.5 55.0 918.5 ω = 1/ ||∆||22 + c 52.2 883.6 ω = α/ ||∆||22 + c 49.7 845.2

#### (b) Adaptive loss.

Method FID FDD Shortcut Model 59.8 1017.3

α FID FDD

10−2 49.7 845.2 5 × 10−3 46.2 860.8 2 × 10−3 50.3 833.0 1 × 10−3 57.2 863.7

v˜ = v(zt, t|x) 59.2 964.6 + Adaptive loss 49.7 845.2 + α = 0.005 45.6 857.8 MeanFlow 43.3 822.3

(c) Consistency step ratio.

(d) Overall ablation study. Table 5: Ablation study over α-Flow.

- G.2 ABLATION STUDY OVER α-FLOW DESIGN SPACE

This section contains an ablation study on α-Flow, specifically for α ∈ (0,1). We use a DiTB/2-non-cfg model (see Table 3) that is pre-trained on flow matching for 200k iterations and then fine-tuned on α-Flow for another 200k iterations. Across all experiments, α remains a constant, and the ratio of r = t is 25 %.

Training objective. Here, we set α = 10−2. Table 5(a) shows that the model only converge when v˜s,t was set to vt and without using EMA for uθ−. This is a key difference from Shortcut Models Frans et al. (2025), which set v˜s,t = uθ−. We suspect their objective only works when α is larger (e.g., 0.5).

Adaptive loss. Geng et al. (2025a) uses an adaptive weight: ω = 1/ ||∆||22 + c = 1/(LMF + c). From Equation (12), we could derive limα→0 Lα = αLMF. When α is close 0, we approximate LMF as Lα/α. This gives us a new adaptive weight, ω = 1/(Lα/α + c) ≈ α/(Lα + c) = α/ ||∆||22 + c as both c and α is very small. As shown in Table 5(b), this new weight performs better empirically, especially compared to the original MeanFlow adaptive weight.

Consistency step ratio. Ablating the α in Table 5 (c) reveals that α = 5 × 10−3 to be the optimal consistency step ratio. This value was then used as the clamping value for our schedule. Table 5(d) shows that by combining these improvements, our discrete α-Flow approach significantly reduces the performance gap between Shortcut models and the MeanFlow model.

### H LLM USAGE

As requested by the ICLR 2026 policy3, we disclose the usage of Large Language Models in this section. LLMs were primarily used in two capacities:

- • Coding assistance for experiments. LLMs provided code auto-completion functionality to ease the process of implementing and analyzing the experiments.
- • Writing assistance for paper writing. We used LLMs to assist with grammar and phrasing validation while working on the submission.

### I RANDOM VS BALANCED CLASSES FOR FID COMPUTATION

We treat EDM series (Karras et al., 2022; 2024) as the standard in FID (Heusel et al., 2017) evaluations, which use a randomly sampled class label (from 0 to 999) for each sample in constructing 50,000 synthetic examples with the model. We found a curious way to decrease the FID values by up to 10% by using “balanced” class sampling: instead of using 50,000 independently sampled random classes, one can generate 50 samples for each of 1000 classes. This greatly improves FID results, but not FDD (i.e., Fr´echet Distance in the DINOv2 (Oquab et al., 2023) feature space) or FCD (Kynk¨a¨anniemi et al., 2022) (i.e., Fr´echet Distance in the CLIP-L-based (Radford et al., 2021) feature space).

Since it is not a standard practice in the community, we only report it separately from the random class sampling results and with the appropriate notice. But we emphasize that it might be a more reasonable way to evaluate FID since it reduces the variance (we are less likely to sample an unlucky set of classes). We provide the results for it in Table 6.

NFE 1 NFE 2 FID FDD FCD FID FDD FCD

Method Class sampling Params Epochs

MeanFlow-XL/2∗ Random U[1..1000] 676M 240 3.47 185.8 3.39 2.46 108.7 2.40 α-Flow-XL/2 (ours) Random U[1..1000] 676M 240 2.95 164.6 3.14 2.32 105.7 2.42 α-Flow-XL/2+ (ours) Random U[1..1000] 676M 240+60 2.58 148.4 3.07 2.15 96.8 2.31

MeanFlow-XL/2∗ Balanced 676M 240 3.33 182.8 3.34 2.26 106.1 2.36 α-Flow-XL/2 (ours) Balanced 676M 240 2.81 162.4 3.10 2.16 103.2 2.37 α-Flow-XL/2+ (ours) Balanced 676M 240+60 2.44 147.2 3.04 1.95 94.6 2.30

#### Table 6: Balanced vs random class sampling for FID, FDD and FCD.

It is curious to observe that while it greatly improves FID results, FDD and FCD are barely affected. We believe that this constitutes one more reason for the community to switch from FID to more robust metrics which correlate better with human perception, like FDD and FCD.

3https://iclr.cc/Conferences/2026/AuthorGuide

### J ADDITIONAL EXPLORATION OF THE MEANFLOW LOSS

| | | | | | | | | | | | | | | | | | | | | |L|FM|ra|t|io|=|7|5|%| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

LFM ratio = 0%

LFM ratio = 0%

0.8

0.8

LFM ratio = 75%

∇∇Averagecos() ,LLTCcFM

∇∇Averagecos( ),LLTFMFM

0.4

0.4

0.0

0.0

−0.4

−0.4

−0.8

−0.8

0 100K 200K 300K 400K

0 100K 200K 300K 400K

Training Iteration

Training Iteration

(a) cos (∇LFM′, ∇LTCc)

###### (b) cos (∇LTFM, ∇LFM′)

LFM ratio = 0%

LFM ratio = 0%

0.8

0.8

LFM ratio = 75%

LFM ratio = 75%

∇∇Averagecos(),LLtotalTFM

∇∇Averagecos( ),LLtotalFM

0.4

0.4

0.0

0.0

−0.4

−0.4

−0.8

−0.8

0 100K 200K 300K 400K

0 100K 200K 300K 400K

Training Iteration

Training Iteration

(c) cos (∇LTFM, ∇LMF)

###### (d) cos (∇LFM′, ∇LMF)

LFM ratio = 0%

LFM ratio = 0%

0.8

0.8

LFM ratio = 75%

LFM ratio = 75%

∇∇Averagecos(),LLTCctotal

∇∇Averagecos(),LLTCcTFM

0.4

0.4

0.0

0.0

−0.4

−0.4

−0.8

−0.8

0 100K 200K 300K 400K

0 100K 200K 300K 400K

Training Iteration

Training Iteration

(e) cos (∇LTCc, ∇LMF)

(f) cos (∇LTCc, ∇LTFM)

Figure 6: Average cosine similarities between the gradients of different losses (LTFM,LFM′,LCTc

,LMF) for DiT-B/2 MeanFlow model trained with 0% and 75% of flow matching.

- K ADDITIONAL VISUALIZATIONS

[Figure 5]

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 6]

[Figure 7]

- Figure 7: Uncurated samples (seeds 1-16) for Class 15 (robin) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 8]

[Figure 9]

[Figure 10]

- Figure 8: Uncurated samples (seeds 1-16) for Class 15 (robin) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 12]

[Figure 13]

- Figure 9: Uncurated samples (seeds 1-16) for Class 29 (axolotl) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 10: Uncurated samples (seeds 1-16) for Class 29 (axolotl) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 18]

[Figure 19]

- Figure 11: Uncurated samples (seeds 1-16) for Class 33 (loggerhead) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 20]

[Figure 21]

[Figure 22]

- Figure 12: Uncurated samples (seeds 1-16) for Class 33 (loggerhead) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 24]

[Figure 25]

- Figure 13: Uncurated samples (seeds 1-16) for Class 88 (macaw) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 26]

[Figure 27]

[Figure 28]

- Figure 14: Uncurated samples (seeds 1-16) for Class 88 (macaw) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 30]

[Figure 31]

- Figure 15: Uncurated samples (seeds 1-16) for Class 89 (cockatoo) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 16: Uncurated samples (seeds 1-16) for Class 89 (cockatoo) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 36]

[Figure 37]

- Figure 17: Uncurated samples (seeds 1-16) for Class 127 (white stork) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 38]

[Figure 39]

[Figure 40]

- Figure 18: Uncurated samples (seeds 1-16) for Class 127 (white stork) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 42]

[Figure 43]

- Figure 19: Uncurated samples (seeds 1-16) for Class 279 (arctic fox) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 44]

[Figure 45]

[Figure 46]

- Figure 20: Uncurated samples (seeds 1-16) for Class 279 (arctic fox) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 48]

[Figure 49]

- Figure 21: Uncurated samples (seeds 1-16) for Class 980 (volcano) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 50]

[Figure 51]

[Figure 52]

- Figure 22: Uncurated samples (seeds 1-16) for Class 980 (volcano) for NFE=2.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 54]

[Figure 55]

- Figure 23: Uncurated samples (seeds 1-16) for Class 975 (lakeside) for NFE=1.

MeanFlow-DiT-XL/2-Flow-DiT-XL/2-Flow-DiT-XL/2+αα

[Figure 56]

[Figure 57]

[Figure 58]

- Figure 24: Uncurated samples (seeds 1-16) for Class 975 (lakeside) for NFE=2.

