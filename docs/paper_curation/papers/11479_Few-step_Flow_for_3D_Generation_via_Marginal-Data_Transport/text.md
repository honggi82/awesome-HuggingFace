# arXiv:2509.04406v1[cs.CV]4Sep2025

## Few-step Flow for 3D Generation via Marginal-Data Transport Distillation

Zanwei Zhou1,∗ Taoran Yi2,∗ Jiemin Fang3,† Chen Yang3 Lingxi Xie3 Xinggang Wang2 Wei Shen1,† Qi Tian3

1Shanghai Jiao Tong University 2Huazhong University of Science and Technology 3Huawei Inc.

{sjtu19zzw, wei.shen}@sjtu.edu.cn {taoranyi, xgwang}@hust.edu.cn {jaminfong, chenyang.res, 198808xc}@gmail.com tian.qi1@huawei.com

https://github.com/Zanue/MDT-dist

### Abstract

Flow-based 3D generation models typically require dozens of sampling steps during inference. Though few-step distillation methods, particularly Consistency Models (CMs), have achieved substantial advancements in accelerating 2D diffusion models, they remain under-explored for more complex 3D generation tasks. In this study, we propose a novel framework, MDT-dist, for few-step 3D flow distillation. Our approach is built upon a primary objective: distilling the pretrained model to learn the Marginal-Data Transport. Directly learning this objective needs to integrate the velocity fields, while this integral is intractable to be implemented. Therefore, we propose two optimizable objectives, Velocity Matching (VM) and Velocity Distillation (VD), to equivalently convert the optimization target from the transport level to the velocity and the distribution level respectively. Velocity Matching (VM) learns to stably match the velocity fields between the student and the teacher, but inevitably provides biased gradient estimates. Velocity Distillation (VD) further enhances the optimization process by leveraging the learned velocity fields to perform probability density distillation. When evaluated on the pioneer 3D generation framework TRELLIS, our method reduces sampling steps of each flow transformer from 25 to 1–2, achieving 0.68s (1 step × 2) and 0.94s (2 steps × 2) latency with 9.0× and 6.5× speedup on A800, while preserving high visual and geometric fidelity. Extensive experiments demonstrate that our method significantly outperforms existing CM distillation methods, and enables TRELLIS to achieve superior performance in few-step 3D generation.

∗ Equal contribution. Work done during internship at Huawei. † Corresponding author.

### 1. Introduction

Flow-based 3D generation models [6, 34, 78, 79, 81, 84, 85, 91, 93] have exhibited remarkable abilities in synthesizing intricate 3D representations from image prompts. However, during inference they typically require dozens of iterative sampling steps, posing significant computational barriers to practical applications such as large-scale 3D content generation for embodied intelligence simulation [75] and realtime interactive editing workflows in graphics systems. Although few-step diffusion distillation methods, particularly consistency models (CMs) [43, 66, 74], have achieved substantial advancements in accelerating 2D diffusion models, their extension to the 3D generation area remains underexplored. The only related work recently is FlashVDM [28], which adopts a few-step distillation framework mainly derived from the previous Phased Consistency Models (PCM) method [74].

3D generation presents inherently greater challenges than its 2D counterpart. Unlike 2D images sampled from a continuous color space, 3D representations, e.g., meshes and 3D Gaussians [25], are discrete and sparsely structured in 3D space. 3D models also contain richer geometric and textural details at a finer granularity. Moreover, in latentspace generative frameworks such as Latent Diffusion models (LDM) [58], the dimension of the 3D latent space is typically higher than that of the 2D latent space. These fundamental differences indicate that 3D generation faces more difficulties and challenges than 2D generation, and thus has stricter requirements on few-step acceleration techniques.

To address these challenges, we propose a novel framework, MDT-dist, for few-step 3D flow distillation. Our method is built upon a primary objective: distilling the pretrained 3D flow model to learn the marginal-data transport. CMs have a similar optimization target, but are limited by the consistency constraint which enforces consistency between adjacent time steps to indirectly learn the target. We

instead propose two novel loss functions, Velocity Matching (VM) and Velocity Distillation (VD), to pursue the primary objective in a more direct way. Directly learning the primary objective needs to integrate the velocity fields, but this integral is intractable to be implemented. Therefore, VM and VD equivalently convert it to tractable objectives respectively. In VM, the optimization of the primary objective is converted into optimizing its time derivative, with the error in the primary objective bounded by the error in the VM loss. Specifically, VM learns to stably match the velocity fields between the student and the teacher. However, it inevitably contains a term involving the derivative of the network output. This term cannot be efficiently backpropagated and is therefore detached, leading to biased gradient estimates. VD further enhances the learning of the marginal-data transport by matching the marginal distributions between the student and the teacher. It leverages the velocity fields learned by the student and the teacher as measure to perform probability density distillation. We evaluate our methods on a state-of-the-art 3D generation framework TRELLIS [81]. Our approach reduces the inference steps of each flow transformer from 25 to just 1-2 and the latency from 6.1s to 0.68s and 0.94s on A800, while preserving high visual and geometric fidelity. Extensive experiments demonstrate that our method significantly outperforms existing CM distillation methods, and makes the distilled TRELLIS model surpass FlashVDM [28], enabling fast 3D content generation beneficial for various downstream tasks.

Our contributions are as follows:

- • We develop a novel few-step flow distillation framework MDT-dist for better 3D generation acceleration.
- • We propose two novel optimization objectives, Velocity Matching (VM) and Velocity Distillation (VD), to jointly enable effective few-step distillation.
- • We distilled TRELLIS to achieve a sweet balance between generation speed and quality. Our method reduces sampling steps of each flow transformer from 25 to 1–2, achieving 0.68s (1 step × 2) and 0.94s (2 steps × 2) latency with 9.0× and 6.5× speedup on A800.

- 2. Related Work
- 3D Generation Models. Early 3D generation methods [1, 2, 5, 20–23, 30, 36, 38, 40, 41, 48, 49, 53, 56, 59, 60, 62, 64, 73, 77, 88, 90] are mainly based on 2D diffusion models [58], generating 3D assets by iteratively prompting 2D diffusion models to optimize 3D representations [25, 50]. DreamFusion [54] and Score Jacobian Chaining (SJC) [65] first introduce Score Distillation Sampling (SDS) to generate 3D assets using pretrained 2D diffusion models. ProlificDreamer [76] and other methods [35, 69, 92] further improve SDS to achieve better generation results. Some methods [8, 32, 47, 55, 70, 86, 87] incorporate shape priors

to significantly reduce generation time. Methods [13, 16, 19, 24, 31, 46, 51, 61, 71, 82, 89, 94] like LRM [19] and LGM [71] build native 3D generative models by pretraining on large-scale 3D data, enabling feed-forward generation of 3D assets without optimization. Some native 3D generation methods [4, 6, 7, 9, 33, 34, 78, 79, 81, 84, 85, 91, 93] further introduce flow matching into the 3D generation field. However, generating high-fidelity 3D assets with 3D diffusion models requires a relatively large number of sampling steps during inference, which both increases users’ queuing time and raises computational costs. FlashVDM [28] shortens the generation time through efficient decoder design and few-step distillation, but it can only generate the shape without appearance. We apply our method to TRELLIS [81] to reduce the number of sampling steps for the two stage flow transformers, enabling the fast generation on both shape and appearance.

- 2D Generative Models. 2D generative modeling has progressed from variational autoencoders (VAEs) [27, 57] to generative adversarial networks (GANs) [15]. VAEs map data distributions to latent Gaussian spaces via Evidence Lower Bound (ELBO) optimization but often yield blurry outputs. GANs employ adversarial training to produce high-fidelity images but suffer from instability and mode collapse. Diffusion models have since become a dominant paradigm: denoising diffusion probabilistic models (DDPM) [18] formalize iterative denoising as a Markov process; score-based generative modeling [65] unifies diffusion under the framework of stochastic differential equations (SDEs); flow matching [37, 39, 72] learns velocity fields for direct ODE-based generation and often requires fewer steps than diffusion. An important topic on diffusion models is acceleration by reducing the number of sample steps. Denoising diffusion implicit models (DDIM) [63] accelerate generation through non-Markovian updates. DPM-Solver [44] further reduces sampling steps by employing higher-order ODE solvers. Consistency models (CMs) [66] enable single-step sampling by enforcing consistency across sampling trajectories. Variants such as phased consistency models (PCM) [74] and Trigflow(sCM) [43] improve stability through phased training strategies and continuous-time formulations, respectively. In contrast to consistency models, our method is derived without the consistency constraint and serves as a novel few-step distillation framework.
- 3. Background

#### 3.1. Diffusion Models

Diffusion models [18] learn to generate data by iteratively denoising samples from Gaussian noise distribution. The framework defines a fixed forward diffusion process and

a learned reverse denoising process. Given a data sample x0 ∼ qdata, the forward diffusion process gradually adds the Gaussian noise and produces a series of noised samples {xt}Tt=1, conditioned on the time step t. This induces a sequence of marginal distributions qt(xt), i.e., the noised distribution at time t. The reverse process is parameterized by a neural network with learnable parameters θ, which learns to generate the denoising direction. During the reverse process, the generated marginal distribution ptθ(xt) is expected to be matched with qt(xt).

Flow Matching. Flow matching (FM) [37, 39, 72] learns to map noise to data distribution by estimating a Probability Flow Ordinary Differential Equation (PF-ODE) process. It defines a continuous-time dynamical system with a learnable velocity field vθ(·,t),t ∈ [0,1], which can be used to construct a time-dependent diffeomorphic map ϕt,

d dt

ϕt(x) = vθ(ϕt(x),t), ϕ0(x) = x,

(1)

which subsequently defines a push-forward ϕ∗ transforming a density over time

ptθ(x) = [ϕt]∗p0(x) = p0(ϕ−1(x)) det∇xϕt−1(x) .

(2) In this way, the velocity field vθ(·,t) is said to generate a probability path ptθ. The velocity field is optimized by minimizing the conditional flow matching loss

0,z ∥vθ ((1 − t)x0 + tz,t) − (z − x0)∥2 .

LCFM(θ) = Et,x

(3) At inference, samples are generated from a Gaussian noise sample z ∼ N(0,I) by solving the ODE backward in time:

1

vθ(xτ,τ)dτ. (4)

x0 = z −

0

Notably, the integral of vθ(xt,t) on a time interval [t1,t2] indeed describes the transport from ptθ(xt

) to ptθ(xt

):

2

1

##### xt

1

2 −

= xt

t2

vθ(xτ,τ)dτ. (5)

t1

In practice, the ODE is discretely approximated using a numerical ODE solver such as Euler’s method,

##### xt

1

2 −

= xt

N

,τk)∆τ, (6)

##### vθ(xτ

k

k=1

where t1 = τ1 < τ2 < ··· < τN = t2, xτ

− vθ(xt

= xt

k−1

τk

,τk)∆τ, and ∆τ = τk − τk−1.

τk

#### 3.2. 3D Generation Models

Our method builds upon TRELLIS [81], a recent highquality 3D asset generation framework. In TRELLIS, a 3D asset is implicitly represented by a structured latent variable (SLAT) S, which is composed of sparse voxels and features:

S = {(fi,pi)}Li=1, fi ∈ RC, pi ∈ {0,1,...,N − 1}3,

(7) where pi is the coordinate of the i-th voxel, and fi is the corresponding feature. C denotes the feature dimension, N denotes the voxel resolution, and L denotes the number of active voxels which is much smaller than N3. SLAT can be decoded into different 3D representations such as 3D Gaussians [25], meshes and NeRFs [50]. For generation, two flow transformers are trained to generate coordinates and features of SLAT separately. During inference, the two models both take 25 sampling steps by default.

### 4. Marginal-Data Transport Distillation

Our target is to distill a pretrained 3D flow model into a fewstep generator under the guidance of a pretrained teacher model. Unlike consistency models limited by the consistency constraint, we instead seek a more direct objective.

A straightforward objective is to directly learn the onestep mapping from the noise distribution to the data distribution. However, it has been validated by VAEs [27] and GANs [15] that directly associating the noise distribution to the data distribution faces unstable optimization issues such as posterior collapse [45] and mode collapse [3, 67]. Inspired by the insight of iterative denoising in diffusion models [18], we propose to enhance the objective by learning the mapping from any marginal distribution qt(xt) to the data distribution qdata.

Specifically, in Sec. 4.1 our primary objective is set to be learning the marginal-data transport which directly transports the marginal distribution qt(xt) to the data distribution qdata. Compared with the objective from VAEs and GANs, our primary objective ensures a progressive learning of the noise-data transport, making the optimization process more stable. Unfortunately, simulating this mapping function needs to integrate the velocity fields which is intractable and cannot be directly optimized. Therefore, starting from the primary objective, we develop two optimizable objectives: velocity matching (Sec. 4.2) and velocity distillation (Sec. 4.3). These two objectives equivalently convert the optimization target from the transport level to the velocity level and the distribution level respectively. Velocity matching learns to stably match the velocity fields between the student and the teacher, and velocity distillation further enhances the optimization process by leveraging the learned velocity fields to perform probability density distillation. We combine these two objectives into a final loss

|ℒ!"|
|---|

|ℒ!"|
|---|

|ℒ!#|
|---|

vpre(xtt,tt) Learnable

vpre

|x<br><br>[Figure 1]<br><br>|
|---|

Velocity Matching Velocity Distillation

|[Figure 2]|
|---|

[Figure 3]

|ℒ!"|
|---|

Frozen

|ℒ!#|
|---|

[Figure 4]

ϕθ ϕθ(xtt,tt)

Gaussian noise

x0 t

|ℒ!#| | |
|---|---|---|
| |[Figure 5]<br><br>| |

vpre(xt t′,tt)

vpre

|[Figure 6]|
|---|

[Figure 7]

ϕθ(z′ ,1)

[Figure 8]

|ℒ!"|
|---|

ϕθ ϕθ(xt t′,tt)

x′ x t′

|ℒ!#|
|---|

- Figure 1. The primary objective of our framework is to learn the transport from the marginal distribution to the data distribution. Based

on it we propose two optimization objectives: velocity matching and velocity distillation. Velocity matching directly supervises ϕθ via matching the velocity fields between the student and teacher (Eq. 13), while velocity distillation indirectly supervises ϕθ via matching the marginal distributions between the student and teacher (Eq. 17).

objective is formulated as Lprimary(θ) := min θ

function in Sec. 4.4. We also clarify the differences between our approach and previous methods in Sec. 4.5. Our framework is presented in Fig. 1.

t

Et,x

0,z D tϕθ(xt,t),

vpretrain(xτ,τ)dτ ,

#### 4.1. Primary Objective

0

(10)

Given a noise sample z ∼ N(0,I), our target is to learn a neural network ϕθ to generate x0 ∼ qdata within few-step forwards. Taking one-step generation as an example, ϕθ is expected to satisfy

where vpretrain denotes the velocity fields predicted by the pretrained teacher model.

#### 4.2. Velocity Matching

Note that the primary objective in Eq. 10 cannot be directly optimized, since the integral 0 t vpretrain(xτ,τ)dτ is intractable. We differentiate the objective function with respect to t and turn it to be

0,z [D (ϕθ(z),z − x0)], (8)

Ex

min

θ

where D(·,·) represents a distance metric such as MSE or Kullback-Leibler Divergence, and θ represents learnable model parameters. This formulation aims to transport the noise distribution to the data distribution directly. However, from the experiences of VAEs [27] and GANs [15], it is well known that directly learning data distribution from noise distribution is a challenging task and easily leads to unstable optimization issues such as posterior or mode collapse. Therefore, we instead force ϕθ to learn the transport from the marginal distribution qt(xt) to the data distribution qdata for any diffusion time step t:

0,z [D (uθ(xt,t),vpretrain(xt,t))], (11)

Et,x

min

θ

dϕθ(xt,t) dt

. (12)

uθ(xt,t) = ϕθ(xt,t) + t

Here uθ(xt,t) is the derivative of tϕθ(xt,t) with respect to t, which actually represents the velocity fields. Intuitively,

Eq. 10 supervises ϕθ by the transport from the marginal distribution to the data distribution, and Eq. 11 converts it to be a supervision on its derivative, i.e., the velocity supervision, to learn student velocity fields matched with the teacher velocity fields.

Now Eq. 11 is tractable and can be directly used for supervision. Given a data sample x0 ∼ qdata, we replace D(·,·) with the MSE metric and define our velocity matching loss as

0,z [D (tϕθ(xt,t),xt − x0)], (9)

Et,x

min

θ

where t is used for normalization, xt = (1−t)x0+tz. With the guidance of a well-pretrained 3D flow model, we can approximate qt(xt) using its learned reverse marginal distribution, and approximate the data distribution qdata using the generated teacher distribution. Recall Eq. 5, the transport from ptθ(xt) to p0 is 0 t vθ(xτ,τ)dτ, thus our primary

LVM(θ) := min θ

2

dϕθ(xt,t) dt − vpretrain(xt,t)

Et,x

.

0,z ϕθ(xt,t) + t

(13)

Algorithm 1: MDT-dist Training Procedure

Input: Pretrained flow model vpretrain, dataset qdata Output: Trained generator ϕθ

- 1 // Initialize generator from pretrained model
- 2 ϕθ ← copyWeights(vpretrain)
- 3 while not converge do

- 4 Sample time step t ∼ U(0, 1)
- 5
- 6 // Compute Velocity Matching Loss
- 7 Sample x0 ∼ qdata, z ∼ N(0, I)
- 8 xt ← (1 − t)x0 + tz
- 9 xt−∆t ← xt − vpretrain(xt, t) · ∆t
- 10 uθ(xt, t) ← ϕθ(xt, t) + t · ∆1t · sg (ϕθ(xt, t) − ϕθ(xt−∆t, t − ∆t))

- 11 LVM ← Et,x0,z ∥uθ(xt, t) − vpretrain(xt, t)∥2
- 12
- 13 // Compute Velocity Distillation Loss
- 14 Sample z′, z′′ ∼ N(0, I)
- 15 x′ ← z′ − ϕθ(z′, 1)
- 16 x′t ← (1 − t)x′ + tz′′
- 17 x′t−∆t ← x′t − vpretrain(x′t, t) · ∆t
- 18 uθ(x′t, t) ← ϕθ(x′t, t) + t · ∆1t · sg ϕθ(x′t, t) − ϕθ(x′t−∆t, t − ∆t)

- 19 LVD ← Et,z′,z′′ − uθ(x′t, t) − vpretrain(x′t, t) · ∂x

′t ∂θ

- 20
- 21 // Update model
- 22 L ← LVM + λLVD
- 23 ϕθ ← update(ϕθ, L)
- 24 end while

We provide a detailed proof in Appendix A.1 to demonstrate that, the error in the primary objective Lprimary(θ) is bounded by the error in the velocity matching loss LVM(θ). Therefore, we can effectively learn the primary objective through optimizing the velocity matching loss.

In practice, we accelerate the convergence by discretely approximating the derivative item

dϕθ(xt,t) dt ≈

1 ∆t

(ϕθ(xt,t) − ϕθ(xt−∆t,t − ∆t)),

(14) where xt−∆t = xt − vpretrain(xt,t)∆t, ∆t is a small constant value and we set it to be 1e − 2. The gradient of the

term ϕθ(xt−∆t,t − ∆t) is expected to be detached, since solving it requires computing two-order derivative which is computationally expensive. When training, we stop the gradient of the derivative term dϕθ(xt,t)

dt rather than only ϕθ(xt−∆t,t−∆t). This operation makes ϕθ(xt,t) to learn more consistent with vpretrain(xt,t), achieving a more stable optimization process.

#### 4.3. Velocity Distillation

Though Eq. 13 serves as a tractable objective, we clarify that with only the velocity matching loss the student model ϕθ cannot be optimized well. An inherent flaw in the velocity matching loss is that both the continuous and discrete formulation of ϕθ(xt,t) cannot be back-propagated properly. Therefore, the optimization of Eq. 13 provides a biased gradient estimate.

We revisit our primary objective from the perspective of score distillation [54, 76]. Since the student and teacher model have the same target distribution qdata, we can equivalently convert the constraint on the marginal-data transport into the constraint on the marginal distribution. Then the primary objective turns to minimize the KullbackLeibler divergence between the student marginal ptθ and the marginal qt (note we approximate the real marginal with the teacher marginal):

Et DKL ptθ ∥ qt , (15)

min

θ

which is the objective of our velocity distillation. Specifically, a sample x′ = z′ − ϕθ(z′) is first synthesized from a noise sample z′ ∼ N(0,I). Then x′ is diffused with z′′ ∼ N(0,I) to obtain x′t = (1 − t)x′ + tz′′. The gradient of the training objective can then be written as

∇θEt DKL ptθ ∥ qt

= Et,z′,z′′ ∇θ log ptθ(x′t) − log qt(x′t)

∂x′t ∂θ

log ptθ(x′t) − ∇x′

log qt(x′t)

= Et,z′,z′′ ∇x′

. (16) ProlificDreamer [76] points out that ∇x′

t

t

log ptθ(x′t) and ∇x′

t

log qt(x′t) represent the score [65] of the noisy prediction and the noisy real data respectively. Here we use the student velocity fields uθ(x′t,t) to replace −∇x′

t

log ptθ(x′t), and the teacher velocity fields vpretrain(x′t,t) to replace −∇x′

t

log qt(x′t). A detailed proof of the rationale for this choice can be found in Appendix A.2. Substituted with Eq. 12, the gradient of our velocity distillation loss is formulated as

t

∇θLVD(θ) := Et,z′,z′′ − ϕθ(x′t,t)+t

dϕθ(x′t,t) dt −vpretrain(x′t,t)

∂x′t ∂θ

.

(17)

Intuitively, Eq. 17 performs probability density distillation which serves as a soft supervision. It applies uθ(x′t,t) − vpretrain(x′t,t) as a criterion to measure the discrepancy between the teacher and student marginals, and indirectly optimizes ϕθ through optimizing x′t.

#### 4.4. Optimization

Our velocity matching loss and velocity distillation loss are complementary. While both are d erived from the primary objective, velocity matching provides partially right gradient estimates, improving the performance of ϕθ(·,t) for all the time steps t. Velocity distillation further utilizes the optimized ϕθ(·,t) as a criterion for measuring distribution discrepancy, enhancing the one-step performance of ϕθ. Our

Appearance Geometry

Method Inference Steps Inference Time (s)

FDincep↓ FDdinov2↓ ULIPI↑ LGM* – 5 26.31 322.71 –

3DTopia-XL* 25 5 37.68 437.37 –

Ln3Diff* 250 8 26.61 357.93 – TRELLIS* 25 × 2 – 9.35 67.21 – TRELLIS 25 × 2 6.1 11.80 65.24 39.53

FlashVDM 5 1.30 – – 37.91 Ours

- 1 × 2 0.68 18.09 164.2 36.88

- 2 × 2 0.94 14.16 110.9 39.11

- Table 1. Quantitative comparison on LGM [71], 3DTopia-XL [9], Ln3Diff [29], and our teacher model TRELLIS [81]. * denotes that the metrics are from TRELLIS, which are measured on the subset of the Toys4K dataset [68], and the inference time comes from their original paper. The other reported inference times measured by us are based on evaluations performed on a NVIDIA A800 GPU.

[Figure 9]

[Figure 10]

Condition Images

FlashVDM 5 Steps (1.30s)

TRELLIS 25 Steps×2 (6.1s)

Ours 2 Steps×2 (0.84s)

Ours 1 Step×2 (0.68s)

Geometry Only

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

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- Figure 2. Qualitative results of FlashVDM [28], the teacher model TRELLIS [81], and our method. Since FlashVDM does not generate the appearance of 3D assets, we compare with FlashVDM only on geometry. final loss function is formulated as

is motivated by learning the marginal-data transport, and is designed to distill a pretrained flow model.

LMDT-dist = LVM + λLVD, (18)

where λ is a hyper-parameter and we set it to be 1.0. Algorithm 1 outlines our final training procedure.

#### 4.5. Relation to Prior Work

Velocity matching and velocity distillation are both derived from our primary objective. Velocity matching is related to consistency models [66] and MeanFlow [14], but still has essential differences. Consistency models are derived from the consistency constraint between adjacent time steps. Compared with consistency models, our objective learns ϕθ(xt,t) more consistent with vpretrain(xt,t), leading to more stable optimization. MeanFlow needs two time variables as model input, thus being not suitable for distilling a pretrained 3D flow model. Differently, our method

Velocity distillation is related to Score Distillation Sampling (SDS) [54] and Variational Score Distillation (VSD) [76]. SDS and VSD both try to measure the student and teacher marginals with the score, i.e., the gradient of the log probability density. SDS only uses the added noise as the student score, leading to a single-point Dirac distribution estimation [76]. VSD finetunes an additional diffusion model ϵϕ to learn the student distribution, resulting in additional memory cost and further learning errors. Instead, our method only learns one model to be both the generator and the distribution measure, being both low-cost and accurate.

Condition Images

TRELLIS 2 Steps×2

Ours 2 Steps×2

TRELLIS 1 Step×2

Ours 1 Step×2

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

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

Figure 3. Qualitative results with and without distillation during few-step inference.

Condition CM PCM sCM Ours Images

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

- Figure 4. Qualitative comparison on CM [66], PCM [74], sCM [43] and ours. Our method exhibits the most complete and fine-grained geometric and visual details.

ConditionOursOnlyVMLoss

Images W/oDistillation

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

- Figure 5. Qualitative results of ablation studies on our proposed two loss functions.

### 5. Experiments

Implementation Details. Our training details follow that of TRELLIS. We only train the generation stage of TRELLIS, the flow transformer for structure generation, and the sparse flow transformer for structured latents generation. For each iteration, we randomly select one image from 24 conditional images as the input condition. The conditional images are rendered with different FoVs. We use AdamW [42] as our optimizer. The classifier-free guidance (CFG) [17] in training is set to be 40 for VM and 100 for VD. We train the structure flow transformer for 6k steps with learning rate 4e − 8 and the sparse flow transformer for 4k steps with learning rate 1e − 8, both with batch size 256. We finetune the pretrained 1.1B (Large) image model of TRELLIS. The sampler during inference is the same as TRELLIS.

Datasets. For training, we use around 500K 3D assets from the Objaverse (XL) [11], ABO [10], 3DFUTURE [12], and HSSD datasets [26]. For evaluation, we use all the 3,218 3D assets from the Toys4k dataset [68].

Metrics. For quantitative evaluation, we use the Fr´echet Distance (FD) with different feature extractors. For appear-

##### FDincep↓ FDdinov2↓ ULIPI↑

CM 20.06 194.5 34.62 PCM 19.53 189.5 34.96 sCM 19.29 186.0 35.04 Ours 18.09 164.2 36.88

- Table 2. Quantitative comparison on CM [66], PCM [74], sCM [43] and ours.

LVM LVD FDincep↓ FDdinov2↓ ULIPI↑ × × 20.26 195.6 34.62 ✓ × 18.42 172.0 35.99 ✓ ✓ 18.09 164.2 36.88

- Table 3. Ablation studies on our proposed two loss functions.

ance, we use the Inceptionv3 model [80] (FDincep) and the DINOv2 model [52] (FDdinov2) as feature extractors. Following TRELLIS [81], we render the generated 3D assets at an elevation of 30 degrees and at four azimuth angles (0, 90, 180, and 270 degrees), resulting in rendering four views per asset. The ground truth 3D assets are rendered in the same way, and we compute the FD between the generated and ground truth images. For geometry, we follow FlashVDM [28] and Direct2D-S2 [79] to use ULIP-2 [83] for calculating the similarity between the images and the final mesh. To accurately measure the similarity between the images and the mesh, we use the four rendered views mentioned above, calculate the similarity for each view, and then take the average. For the metric ULIPI in the paper, we multiply the results by 100.

#### 5.1. Quantitative Comparison

Since methods for distilling 3D diffusion models remain scarce, FlashVDM [28] is one of the few. We also compare against non-distilled methods (LGM [71], 3DTopia-XL [9], Ln3Diff [29]) and the teacher model TRELLIS [81]. Apart from LGM, all others are based on 3D diffusion. The results are shown in Table 1.

Non-diffusion-based Method. We compare our method with LGM [71], one of the most influential non-3Ddiffusion-based 3D generation methods. Compared to LGM, our approach distills an end-to-end 3D diffusion model, which has stronger 3D awareness, improves both appearance and geometry quality of the generated 3D assets, and reduces inference time.

Non-distilled 3D Diffusion Methods. We further compare with non-distilled 3D diffusion methods, including 3DTopia-XL [9], Ln3Diff [29], and the teacher model TRELLIS [81]. After distillation, our method reduces the

two 3D flow transformers of TRELLIS from 25 steps each (50 steps total) to 1 and 1 steps (2 steps in total), achieving high-quality 3D generation in just 0.68s. We achieve comparable performance on appearance metrics (FDincep, FDdinov2) and the geometry metric (ULIPI) compared with the teacher model when the number of inference steps is

- 4. Compared to Ln3Diff and 3DTopia-XL [9], our method outperforms it in both appearance and geometry evaluations with fewer than 12× the inference steps and inference time.

Distilled 3D Diffusion Method. Compared to the recent distilled 3D-shape diffusion model FlashVDM [28], we evaluate only its geometry metric (ULIPI), since it does not generate appearance. We use the officially released code of FlashVDM and evaluate it with the default configuration. Our method and FlashVDM complete inference in a similar number of steps. On the geometry metric ULIPI, our method outperforms FlashVDM.

- 5.2. Qualitative Comparison

Qualitative Comparison with Other Methods. We present a visual comparison of the results in Fig. 2. The results show that our method can generate high-quality 3D assets with both geometry and appearance. Since FlashVDM [28] does not generate appearance for 3D assets, our comparison with FlashVDM mainly focuses on geometry. Compared to FlashVDM, our method generates geometry with more details and better consistency with the conditioned images. Compared with the teacher model TRELLIS [81], our method generates comparable 3D assets.

#### 5.3. Comparison with CM methods

We conduct qualitative and quantitative comparison with three CM methods: CM [66], PCM [74], and sCM [43]. The results are shown in Fig. 4 and Table 2. As the basic consistency model, CM performs the worst. Due to the multi-phase design, PCM learns more stably and achieves a better performance than CM. sCM utilizes a continuous format of CM, which eliminates the discretization errors, thereby performing better than the other two CM methods. Derived from a different objective, our method largely outperforms all the CM methods, with more complete and finegrained geometric and visual details.

#### 5.4. Ablation Study

We conduct ablation studies on our proposed two loss functions to validate their effectiveness.

Velocity Matching. The quantitative results of the ablation study on VM are in Table 3. It can be found that directly applying VM to finetune TRELLIS improves both the geometric quality and the visual quality. In Fig. 5, applying

VM significantly reduces the unwanted extra geometry generated by the model, while partially restoring the missing and incomplete geometric structures. Moreover, compared to the generation result of the non-distilled model, the distilled model exhibits enhanced geometric fidelity, particularly in fine structures such as the shape of car windows and the overall body geometry.

Velocity Distillation. The quantitative results of the ablation study on VD are in Table 3. With the help of VM, VD is capable of learning the geometry better. In Fig. 5, VD further eliminates the unwanted extra geometry and addresses the remaining incomplete regions in the geometry.

Joint Optimization. In Table 3, joint optimization with VM and VD achieves the best performance, demonstrating the complementarity between the two loss functions. In Fig. 3, under the joint effect of the two loss functions, both geometry and visual quality are significantly improved compared to the TRELLIS baseline. Our method significantly addresses the remaining incomplete regions in the geometry and improves the details.

### 6. Conclusion

We present a novel framework MDT-dist for few-step flow distillation in 3D generation. By formulating a primary objective as modeling the transport from the marginal distribution to the data distribution, our approach provides a more direct solution to few-step generation, in contrast to consistency models that rely on consistency constraints on the adjacent time steps. To effectively optimize this objective, we introduce two optimization objectives: Velocity Matching (VM), which converts the optimization target from the transport level to the velocity level, enabling tractable and stable matching of velocity fields between the student and the teacher, and Velocity Distillation (VD), which converts the optimization target from the transport level to the distribution level, leveraging the learned velocity fields to perform probability density distillation. When applied to TRELLIS, our method reduces the sampling steps from 25 to 1–2 while preserving high geometric and visual fidelity. Extensive experiments demonstrate that our approach significantly outperforms existing consistency distillation techniques, and achieves new state-of-the-art performance in few-step 3D generation.

Limitations and Future Work. Our method still requires a large amount of conditional images and geometric data to conduct few-step distillation training. While images are relatively easy to collect, high-quality geometric data is much more scarce and expensive, making the few-step 3D generation distillation costly. A possible improvement is to

eliminate the dependence on geometric data and take only conditional images as input, which will reduce the cost and further scale up the distillation by leveraging the abundant online images.

### References

- [1] Mohammadreza Armandpour, Huangjie Zheng, Ali Sadeghian, Amir Sadeghian, and Mingyuan Zhou. Reimagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:2304.04968, 2023. 2
- [2] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and KwanYee K Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models. arXiv preprint arXiv:2304.00916, 2023. 2
- [3] Tong Che, Yanran Li, Athul Jacob, Yoshua Bengio, and Wenjie Li. Mode regularized generative adversarial networks. In International Conference on Learning Representations,

2017. 3

- [4] Jinnan Chen, Lingting Zhu, Zeyu Hu, Shengju Qian, Yugang Chen, Xin Wang, and Gim Hee Lee. Mar-3d: Progressive masked auto-regressor for high-resolution 3d generation. In CVPR, 2025. 2
- [5] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. arXiv preprint arXiv:2303.13873, 2023. 2
- [6] Rui Chen, Jianfeng Zhang, Yixun Liang, Guan Luo, Weiyu Li, Jiarui Liu, Xiu Li, Xiaoxiao Long, Jiashi Feng, and Ping Tan. Dora: Sampling and benchmarking for 3d shape variational auto-encoders. In CVPR, 2025. 1, 2
- [7] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, and XIngang Pan. Sar3d: Autoregressive 3d object generation and understanding via multi-scale 3d vqvae. In CVPR,

2025. 2

- [8] Zilong Chen, Feng Wang, and Huaping Liu. Text-to-3d using gaussian splatting. arXiv preprint arXiv:2309.16585, 2023. 2
- [9] Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling highquality 3d asset generation via primitive diffusion. In CVPR,

2025. 2, 6, 8

- [10] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. CVPR, 2022. 7
- [11] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 7
- [12] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. IJCV, 2021. 7

- [13] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. NeurIPS, 2022. 2
- [14] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 6
- [15] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020. 2, 3, 4
- [16] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023. 2
- [17] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 7
- [18] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 2, 3
- [19] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2
- [20] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. Dreamwaltz: Make a scene with complex 3d animatable avatars. arXiv preprint arXiv:2305.12529, 2023. 2
- [21] Yangyi Huang, Hongwei Yi, Yuliang Xiu, Tingting Liao, Jiaxiang Tang, Deng Cai, and Justus Thies. Tech: Text-guided reconstruction of lifelike clothed humans. arXiv preprint arXiv:2308.08545, 2023.
- [22] Ajay Jain, Ben Mildenhall, Jonathan T Barron, Pieter Abbeel, and Ben Poole. Zero-shot text-guided object generation with dream fields. In CVPR, 2022.
- [23] Ruixiang Jiang, Can Wang, Jingbo Zhang, Menglei Chai, Mingming He, Dongdong Chen, and Jing Liao. Avatarcraft: Transforming text into neural human avatars with parameterized shape and pose control. arXiv preprint arXiv:2303.17606, 2023. 2
- [24] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023. 2
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42

(4), 2023. 1, 2, 3, 14, 15

- [26] Mukul Khanna*, Yongsen Mao*, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat Synthetic Scenes Dataset (HSSD-200): An Analysis of 3D Scene Scale and Realism Tradeoffs for ObjectGoal Navigation. arXiv preprint, 2023. 7
- [27] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 2, 3, 4
- [28] Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Haolin Liu, Fuyun Wang, Huiwen Shi, Xianghui Yang, Qingxiang Lin, Jingwei Huang, Yuhong Liu, et al. Unleashing vecset diffusion model

for fast shape generation. arXiv preprint arXiv:2503.16302,

2025. 1, 2, 6, 8

- [29] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In ECCV, 2024. 6, 8
- [30] Jiabao Lei, Yabin Zhang, Kui Jia, et al. Tango: Text-driven photorealistic and robust 3d stylization via lighting decomposition. NeurIPS, 2022. 2
- [31] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:2311.06214, 2023. 2
- [32] Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:2310.02596, 2023. 2
- [33] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 2
- [34] Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, et al. Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets. arXiv preprint arXiv:2505.07747, 2025. 1, 2
- [35] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching, 2023. 2
- [36] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In CVPR, 2023. 2
- [37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2, 3
- [38] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. arXiv preprint arXiv:2303.11328, 2023. 2
- [39] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 2, 3
- [40] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 2
- [41] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008, 2023. 2
- [42] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 7

- [43] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 1, 2, 7, 8
- [44] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. NeurIPS,

2022. 2

- [45] James Lucas, George Tucker, Roger B Grosse, and Mohammad Norouzi. Don’t blame the elbo! a linear vae perspective on posterior collapse. NeurIPS, 2019. 3
- [46] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. arXiv preprint arXiv:2306.07279, 2023. 2
- [47] Baorui Ma, Haoge Deng, Junsheng Zhou, Yu-Shen Liu, Tiejun Huang, and Xinlong Wang. Geodream: Disentangling 2d and geometric priors for high-fidelity and consistent 3d generation. arXiv preprint arXiv:2311.17971, 2023. 2
- [48] Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In CVPR, 2023. 2
- [49] Oscar Michel, Roi Bar-On, Richard Liu, Sagie Benaim, and Rana Hanocka. Text2mesh: Text-driven neural stylization for meshes. In CVPR, 2022. 2
- [50] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2, 3
- [51] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 2
- [52] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 8
- [53] Yichen Ouyang, Wenhao Chai, Jiayi Ye, Dapeng Tao, Yibing Zhan, and Gaoang Wang. Chasing consistency in text-to-3d generation from a single image. arXiv preprint arXiv:2309.03599, 2023. 2
- [54] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 2, 5, 6
- [55] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to3d. In CVPR, 2024. 2
- [56] Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508, 2023. 2
- [57] Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In ICML. PMLR, 2014. 2

- [58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2
- [59] Hoigi Seo, Hayeon Kim, Gwanghyun Kim, and Se Young Chun. Ditto-nerf: Diffusion-based iterative text to omnidirectional 3d model. arXiv preprint arXiv:2304.02827,

2023. 2

- [60] Junyoung Seo, Wooseok Jang, Min-Seop Kwak, Jaehoon Ko, Hyeonsu Kim, Junho Kim, Jin-Hwa Kim, Jiyoung Lee, and Seungryong Kim. Let 2d diffusion model know 3dconsistency for robust text-to-3d generation. arXiv preprint arXiv:2303.07937, 2023. 2
- [61] Qiuhong Shen, Xuanyu Yi, Zike Wu, Pan Zhou, Hanwang Zhang, Shuicheng Yan, and Xinchao Wang. Gamba: Marry gaussian splatting with mamba for single view 3d reconstruction. arXiv preprint arXiv:2403.18795, 2024. 2
- [62] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 2
- [63] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2
- [64] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint arXiv:2305.11337, 2023. 2
- [65] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2, 5, 13
- [66] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 1, 2, 6, 7, 8
- [67] Akash Srivastava, Lazar Valkov, Chris Russell, Michael U Gutmann, and Charles Sutton. Veegan: Reducing mode collapse in gans using implicit variational learning. NeurIPS,

2017. 3

- [68] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In CVPR, 2021. 6, 7
- [69] Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, and Yebin Liu. Dreamcraft3d: Hierarchical 3d generation with bootstrapped diffusion prior. arXiv preprint arXiv:2310.16818, 2023. 2
- [70] Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. arXiv preprint arXiv:2309.16653,

2023. 2

- [71] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 2, 6, 8
- [72] Alexander Tong, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Kilian Fatras, Guy Wolf, and Yoshua Bengio. Conditional flow matching: Simulation-free dynamic optimal transport. arXiv preprint arXiv:2302.00482, 2(3), 2023. 2, 3

- [73] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:2304.12439, 2023. 2
- [74] Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. NeurIPS, 2024. 1, 2, 7, 8
- [75] Xinjie Wang, Liu Liu, Yu Cao, Ruiqi Wu, Wenkang Qin, Dehui Wang, Wei Sui, and Zhizhong Su. Embodiedgen: Towards a generative 3d world engine for embodied intelligence, 2025. 1
- [76] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2, 5, 6
- [77] Jinbo Wu, Xiaobo Gao, Xing Liu, Zhengyang Shen, Chen Zhao, Haocheng Feng, Jingtuo Liu, and Errui Ding. Hdfusion: Detailed text-to-3d generation leveraging multiple noise estimation. arXiv preprint arXiv:2307.16183, 2023. 2
- [78] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024. 1, 2
- [79] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Philip Torr, Xun Cao, and Yao Yao. Direct3d-s2: Gigascale 3d generation made easy with spatial sparse attention. arXiv preprint arXiv:2505.17412, 2025. 1, 2, 8
- [80] Xiaoling Xia, Cui Xu, and Bing Nan. Inception-v3 for flower classification. In 2017 2nd international conference on image, vision and computing (ICIVC). IEEE, 2017. 8
- [81] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506, 2024. 1, 2, 3, 6, 8
- [82] Yinghao Xu, Zifan Shi, Wang Yifan, Sida Peng, Ceyuan Yang, Yujun Shen, and Wetzstein Gordon. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation. arxiv: 2403.14621, 2024. 2
- [83] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, et al. Ulip-2: Towards scalable multimodal pre-training for 3d understanding. In CVPR,

2024. 8

- [84] Xianghui Yang, Huiwen Shi, Bowen Zhang, Fan Yang, Jiacheng Wang, Hongxu Zhao, Xinhai Liu, Xinzhou Wang, Qingxiang Lin, Jiaao Yu, et al. Hunyuan3d 1.0: A unified framework for text-to-3d and image-to-3d generation. arXiv preprint arXiv:2411.02293, 2024. 1, 2
- [85] Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. arXiv preprint arXiv:2503.22236,

2025. 1, 2

- [86] Taoran Yi, Jiemin Fang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussian splatting with point cloud priors. arXiv preprint arXiv:2310.08529,

2023. 2

- [87] Taoran Yi, Jiemin Fang, Zanwei Zhou, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Xinggang Wang, and Qi Tian. Gaussiandreamerpro: Text to manipulable 3d gaussians with highly enhanced quality. arXiv preprint arXiv:2406.18462, 2024. 2
- [88] Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Kang Du, and Min Zheng. Avatarverse: High-quality & stable 3d avatar creation from text and pose. arXiv preprint arXiv:2308.03610,

2023. 2

- [89] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. ECCV, 2024. 2
- [90] Longwen Zhang, Qiwei Qiu, Hongyang Lin, Qixuan Zhang, Cheng Shi, Wei Yang, Ye Shi, Sibei Yang, Lan Xu, and Jingyi Yu. Dreamface: Progressive generation of animatable 3d faces under text guidance. arXiv preprint arXiv:2304.03117, 2023. 2
- [91] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 2024. 1, 2
- [92] Minda Zhao, Chaoyi Zhao, Xinyue Liang, Lincheng Li, Zeng Zhao, Zhipeng Hu, Changjie Fan, and Xin Yu. Efficientdreamer: High-fidelity and robust 3d creation via orthogonal-view diffusion prior. arXiv preprint arXiv:2308.13223, 2023. 2
- [93] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 1, 2
- [94] Zi-Xin Zou, Zhipeng Yu, Yuan-Chen Guo, Yangguang Li, Ding Liang, Yan-Pei Cao, and Song-Hai Zhang. Triplane meets gaussian splatting: Fast and generalizable singleview 3d reconstruction with transformers. arXiv preprint arXiv:2312.09147, 2023. 2

## Few-step Flow for 3D Generation via Marginal-Data Transport Distillation Supplementary Material

### A. Proofs

#### A.1. Velocity Matching

Theorem 1. Assume the velocity matching loss is uniformly bounded above by a constant M > 0, i.e., for all t ∈ [0,1], x0 ∼ qdata, and z ∼ N(0,I),

∥uθ(xt,t) − vpretrain(xt,t)∥2 ≤ M.

Under this assumption, the primary objective with D(·,·) replaced by MSE metric satisfies the error bound

Lprimary(θ) ≤ M · Et[t2],

where the expectation is over the distribution of t. For a fixed t, the bound simplifies to Lprimary(θ | t) ≤ Mt2.

Proof. Define the learnable student transport function to be Tθ(t) = tϕθ(xt,t), and the target integral Tpretrain(t) =

t 0 vpretrain(xτ,τ)dτ. Note that Tθ(0) = Tpretrain(0) =

0, and their derivatives satisfy dT

θ(t)

dt = uθ(xt,t) and dT pretrain(t)

dt = vpretrain(xt,t). The error for a fixed t, x0, and z is

t

(uθ(xτ,τ) − vpretrain(xτ,τ))dτ

D(Tθ(t),Tpretrain(t)) =

0

2

t

,

=

e(τ)dτ

0

(19) where e(τ) = uθ(xτ,τ) − vpretrain(xτ,τ) satisfies ∥e(τ)∥2 ≤ M for all τ ∈ [0,t] by assumption (with D as squared norm). By the triangle inequality for norms,

√

√

t

t

t

e(τ)dτ ≤

∥e(τ)∥dτ ≤

M dτ = t

M,

0

0

0

(20) since ∥e(τ)∥ ≤

√

M. Squaring both sides gives

2

t

≤ Mt2. (21)

e(τ)dτ

0

Thus, for fixed t,

0,z [D(Tθ(t),Tpretrain(t))] ≤ Mt2,

Lprimary(θ|t) = Ex

(22) as the bound holds uniformly. Taking expectation over t,

Lprimary(θ) = EtEx

0,z [D(Tθ(t),Tpretrain(t))] ≤ Et[Mt2] = MEt[t2].

(23)

| |
|---|

This proof suggests that the error in the primary objective is bounded by the error in the velocity matching loss. Therefore, we can learn the primary objective through optimizing the velocity matching loss.

#### A.2. Velocity Distillation

Score-SDE [65] defines the forward diffusion process by the forward-time SDE:

dxt = f(xt,t)dt + g(t)dwt, (24)

where f : Rd × [0,T] → Rd is a Lipschitz-continuous drift function, g : [0,T] → R>0 is a continuous diffusion coefficient, and wt is a standard d-dimensional Wiener process.

The score function at time t is the gradient of the logmarginal density:

s(x,t) = ∇x log pt(x), (25)

where pt(x) is the marginal density induced by the forward process at time t.

The velocity field arises in the deterministic reformulation of the generative process via the probability flow ODE:

2

dxt dt

- 1

- 2

g(t)2s(xt,t). (26)

= v(xt,t) = f(xt,t) −

Theorem 2. Velocity distillation differs from score distillation by a multiplicative factor 21g(t)2. Proof. We have

∇θLVD(θ|t)

∂x′t ∂θ

= Ez,z′ − uθ(x′t,t) − vpretrain(x′t,t)

- 1

- 2

= Ez,z′ − f(x′t,t) −

g(t)2sθ(xt,t)

∂x′t ∂θ

- 1

- 2

− f(x′t,t) −

g(t)2s′pretrain(x′t,t)

∂x′t ∂θ

- 1

- 2

g(t)2(sθ(x′t,t) − spretrain(x′t,t))

= Ez,z′

- 1

- 2

g(t)2∇θLSD(θ|t).

=

(27)

| |
|---|

This proof implies that, omitting a weight coefficient related to t, velocity distillation is equivalent to score distillation.

### B. More Results

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

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

- Figure 1. Our 3D Gaussian [? ] and mesh generation results with 2 × 2 steps.

- Figure 6. Our 3D Gaussians [25] and mesh generation results. Odd rows and even rows represent samples from 2 steps ×2 and 1 step ×2 during inference, respectively.

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

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

- Figure 2. Our 3D Gaussian [? ] and mesh generation results with 2 × 2 steps.

- Figure 7. Our 3D Gaussians [25] and mesh generation results. Odd rows and even rows represent samples from 2 steps ×2 and 1 step ×2 during inference, respectively.

