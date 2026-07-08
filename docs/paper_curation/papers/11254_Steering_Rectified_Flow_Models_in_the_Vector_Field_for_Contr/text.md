# arXiv:2412.00100v1[cs.CV]27Nov2024

## Steering Rectified Flow Models in the Vector Field for Controlled Image Generation

Maitreya Patel♠, Song Wen♢, Dimitris N. Metaxas♢, Yezhou Yang♠ ♠ Arizona State University ♢Rutgers University

{maitreya.patel, yz.yang}@asu.edu {song.wen, dnm}@rutgers.edu

|[Figure 1]|
|---|

[Figure 2]

[Figure 3]

[Figure 4]

Flux.1DevInstaFlow

A dog wearing glasses. A sleeping dog. A tiger.

|[Figure 5]|
|---|

|[Figure 6]|
|---|

[Figure 7]

[Figure 8]

|[Figure 9]|
|---|

[Figure 10]

Editing

Inpainting

Classifier Guidance

Figure 1. FlowChef steers the trajectory of Rectified Flow Models during inference to tackle linear inverse problems, image editing, and classifier guidance. We extend FlowChef to SOTA models like Flux and InstaFlow, enabling gradient- and inversion-free control for efficient, controlled image generation.

#### Abstract

controlled image generation that, for the first time, simultaneously addresses classifier guidance, linear inverse problems, and image editing without the need for extra training, inversion, or intensive backpropagation. Finally, we perform extensive evaluations and show that FlowChef significantly outperforms baselines in terms of performance, memory, and time requirements, achieving new state-ofthe-art results. Project Page: https://flowchef. github.io.

Diffusion models (DMs) excel in photorealism, image editing, and solving inverse problems, aided by classifierfree guidance and image inversion techniques. However, rectified flow models (RFMs) remain underexplored for these tasks. Existing DM-based methods often require additional training, lack generalization to pretrained latent models, underperform, and demand significant computational resources due to extensive backpropagation through ODE solvers and inversion processes. In this work, we first develop a theoretical and empirical understanding of the vector field dynamics of RFMs in efficiently guiding the denoising trajectory. Our findings reveal that we can navigate the vector field in a deterministic and gradient-free manner. Utilizing this property, we propose FlowChef, which leverages the vector field to steer the denoising trajectory for controlled image generation tasks, facilitated by gradient skipping. FlowChef is a unified framework for

#### 1. Introduction

Recent advances in diffusion models have led to rapid progress in AI generated content (AIGC), particularly in text-to-image (T2I) and text-to-video (T2V) models across various domains such as entertainment, arts, and design [12, 32, 36, 40, 44, 45]. These developments have resulted in remarkable performance in image editing, solving in-

[Figure 11]

∇ 𝐿 ∇ 𝐿

∇ 𝐿 ∇ 𝐿

∇ 𝐿 ∇ 𝐿

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

𝑝 𝑝

𝑝 𝑝

𝑝 𝑝

(a) Trajectory Space of Nonlinear ODE (b) D-Flow Inference-time Trajectory (c) FlowChef Inference-time Trajectory

𝑥 𝑥 Inference trajectory

𝑥 𝑥 forward trajectory inversion

[Figure 17]

[Figure 18]

gradient direction

- Figure 2. Motivation behind FlowChef based on rectified flow models’ trajectory space. Let p1 ∼ N(0, I) and p0 be distributions, with x1 ∼ p1 as initial noise, xref0 as the target sample, xˆ0 as the denoised sample from x1, and xref1 as the specific noise leading to

xref0 . (a) Stochasticity and nonlinear trajectories with crossovers can complicate gradient estimation at each denoising step t. (b) D-Flow (baseline) inference-time trajectory requires the backpropagation through entire denoising steps. (c) Our method FlowChef enables

efficient trajectory steering to guide xt along the trajectory towards xref0 .

verse problems, and personalization. This progress could be attributed to key advances like latent diffusion models (LDMs) [40] and classifier-free guidance (CFG) [16], among other essential components. Despite their applicability to various downstream tasks, these models demand increasing computational resources. For instance, CFG requires additional unconditional training of the model, while traditional classifier guidance necessitates training noiseaware classifiers [11]. Similarly, existing approaches for solving inverse problems often require minutes of computation and additional memory overhead [1, 7, 43, 46, 47]. Moreover, image editing methods typically involve either inversion or explicit training [3, 5, 19]. These limitations can be attributed to the inherent stochasticity of diffusion models, often requiring a higher number of function evaluations (NFEs).

cover that in nonlinear ODEs with stochasticity or trajectory crossovers, error terms emerge that hinder convergence due to inaccuracies in estimating denoised samples or improper gradient approximations (see Figure 2(a)).

Contrary to diffusion models, rectified flow models exhibit straight trajectories and avoid significant trajectory crossovers due to their linear interpolation between noise and data distributions (see Figure 2(b-c)). We theoretically demonstrate and empirically validate that RFMs can achieve higher convergence rates without additional computational overhead by capitalizing on this key property. Building on this understanding, we present FlowChef, that proposes to steer the trajectories towards the target in the vector field by gradient skipping (see Figure 2(c)). This allows us to navigate the vector field in a deterministic manner, akin to a north star guiding sailors across a dark ocean.

However, the recent introduction of flow-based methods [23], especially rectified flow models (RFMs) [22, 24], addresses these limitations to some extent by requiring fewer NFEs, depending on the model considered. Recent works have attempted to solve inverse problems by leveraging this property, focusing mainly on pixel models [1, 28]. While these approaches have improved computational time requirements, they are still not sufficiently efficient, as they require inversion and incur significant memory overhead. As a result, they cannot be extended to large state-of-the-art models like Flux or SD3 [12].

We conduct extensive evaluations of FlowChef across tasks such as pixel-level classifier guidance, image editing, and classifier-guided style transfer. Our results demonstrate that FlowChef not only surpasses baseline methods but does so with greater computational efficiency and without the need for inversion. As illustrated in Figure 1, FlowChef efficiently addresses a variety of tasks. For perspective, FlowChef handles the linear inverse problems within 18 seconds on the latent-space model, while SOTA takes 1-3 minutes per image. Furthermore, we explore its practical applicability to large-scale models (i.e., Flux) to tackle both linear inverse problems and image editing together without inversion and within 30 NFEs at billions of parameter scales. Our key contributions can be summarized as follows:

In this paper, we introduce FlowChef, a novel method that significantly enhances controlled image generation by leveraging the unique characteristics of rectified flow models. We first standardize the objective of controlled synthesis, unifying various downstream tasks within a single framework. By revisiting the ordinary differential equations (ODEs) that govern these models, we analyze their error dynamics both theoretically and empirically. We dis-

- • We develop a unified perspective to study rectified flow models theoretically and empirically for a guided, controlled generation.
- • We introduce FlowChef, the most efficient method

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

0 1 2 3 5 10 20 30

- Figure 3. Illustration of impact of guided control step on Flux.1[Dev] with mean squared error as cost function (L = ||xˆ0 − xref0 ||22). This shows that FlowChef could guide the rectified flow models on the fly without requiring either the gradients through the Flux model or inversion. Importantly, the convergence speed is slowed down for illustration purposes.

to date for guided, controlled generation using RFMs, achieving state-of-the-art performance without requiring inversion or gradient backpropagation through the ODESolver.

• We demonstrate FlowChef’s superior performance across multiple tasks, including linear inverse problems in both pixel and latent spaces, image editing evaluated on the PIE benchmark [19], classifier guidance, and through large-scale human preference studies.

#### 2. Related Works

We provide detailed related works, specially diffusionbased methods and conditional sampling, in the Appendix.

Inverse Problems. This task addresses training-free approaches for solving inverse problems such as in-painting, super resolution, Gaussian de-blurring etc [10]. Since Dhariwal et. al. demonstrated that guiding models with classifiers improves image generation quality [11], much of the current literature focuses on diffusion models, particularly pixel-space models [7, 10, 47, 52]. However, these models face challenges when scaled to latent-space models, as they are incompatible with off-the-shelf pretrained models and require backpropagation through ODESolvers, which can take at least three minutes per image for satisfactory results [10, 41, 43, 46]. Methods such as MPGD [14] attempt to mitigate these issues via manifold correction, but limitations persist, especially with large-scale models. Recent work has extended these approaches to ODEs (e.g., OTODE) and flow models [35]. D-Flow [1], for instance, optimizes initial noise by differentiating through the full trajectory chain; however, this comes with significant resource demands and is not adaptable to state-of-the-art (SOTA) models like Flux or SD3 [12]. In this work, we propose FlowChef, which addresses linear inverse problems in a gradient- and inversion-free manner.

Image Editing. Diffusion-based approaches dominate image editing [17], but they rely heavily on accurate inversion [3, 18, 19, 30]. Although inversion-free diffusion methods are faster, they often lack in edit quality [9, 29, 51, 53].

Despite RFMs being SOTA in text-to-image (T2I) generation, they still lack robust editing capabilities. iRDS [54] presents an inversion strategy for RFMs, especially InstaFlow [25], but it lacks quality and control. Similarly, RectifID [48] offers an optimization-based approach to modify the whole trajectory for personalized T2I generation but performs poorly with InstaFlow like straight models. To the best of our knowledge, we present the first comprehensive solution that enhances RFMs for image editing and extends beyond it that too without significant computational or time overhead.

Concurrent Works. We note two concurrent works: RFInversion [42] and PnP-Flow [28]. RF-Inversion offers an optimal control-based approach for image inversion and editing, whereas our method generalizes across all controlled generation tasks. We demonstrate that inversion is unnecessary, even for RF-Inversion, making RF-Inversion a special case of FlowChef, where starting noise originates from an inverted target image rather than random noise, as in FlowChef. PnP-Flow is an inversion- and gradient-free method for inverse problems, but it leads to over-smoothed results and lacks extensibility to image editing.

#### 3. Preliminaries

Classifier guidance, inversion problems, and image editing involve guiding a model toward a specific target sample or distribution in both pixel and latent spaces. However, these tasks are often treated separately in literature. Here, we present a unified problem formulation to encompass these downstream tasks, with a focus on rectified flow models.

##### 3.1. Problem Formulation

Let uθ : Rd × [0,T] → Rd represent a pretrained flow model estimating the drift v = x1 − x0 from xt. The denoised sample xˆ0 is obtained by integrating the drift uθ over time from t = T to t = 0, starting from xT ∼ p1. With a target sample xref0 , we define a cost function L : Rd ×Rd → R+ that quantifies the cost of aligning xˆ0 with xref0 , yielding the optimization problem:

Method NFEs CG Scale FID (↓) VRAM (↓) Time (↓)

DDIM 50 - 5.39 3.67 14.22 MPGD 50 1 4.24 6.56 25.01 MPGD 50 10 5.46 6.56 25.01

Oursw/ skip grad 50 1 19.28 6.56 24.95 RFPP (2-flow) 2 - 4.56 3.29 0.28 RFPP (2-flow) 15 - 4.29 3.36 2.75 Oursw/ backpropagation 15 5 2.77 17.98 12.79 Oursw/ skip grad 15 50 3.13 6.64 5.85

Table 1. Performance of Various guided sampling methods on ImageNet64x64 with 32 batch size inference on A6000 GPU.

L(ˆxt,xref0 ), (1)

min

{xˆt}Tt=0

where {xˆt}Tt=0 represents the model-generated trajectory from xT to x0. The objective is to find the trajectory that minimizes L, effectively steering the generated sample toward the target. This can be adapted for the denoising stage with either a noise-aware cost function at each timestep t or by estimating x0 to refine the trajectory as needed. The gradient update is given by:

xt ← xt − s · ∇xtL(ˆx0,xref0 ), (2)

where s is guidance scale. This process requires estimating xˆ0, backpropagating gradients through ODESolver (uθ) to adjust xt, and iteratively refining xt−∆t. Additional details on the baseline algorithm is in the Appendix. As it can be observed, this approach depends on accurate xˆ0 estimation and substantial computation to ensure that the trajectory remains on the data manifold.

##### 3.2. Cost Functions

Notably, explicit xref0 is unnecessary and can be approximated with appropriate cost functions depending on the

downstream tasks. Assuming initial Gaussian noise xT leads to xˆ0, the cost function can be defined as:

L(ˆx0,xref0 ) = ||xˆ0 − xref0 ||22. (3)

In inverse problems, let F : Rd → Rn represent a degradation operation (e.g., downsampling for superresolution). We then define:

L(ˆx0,xref0 ) = ||F(ˆx0) − xref0 ||22. (4)

Here, xref0 is a degraded sample, and we guide the model to generate xˆ0 such that its degraded version matches xref0 . For classifier guidance, the cost function can be based on the negative log-likelihood (NLL). Specifically, given a classifier pϕ(c|xˆ0), the cost function is:

L(ˆx0,c) = −log pϕ(c|xˆ0). (5)

Remark 1. Although presented in pixel space, this formulation extends to latent space by introducing a Variational Autoencoder (VAE) encoder (E) and decoder (D).

#### 4. Proposed Method

In this section, we introduce our method, FlowChef, which enables free-form control for rectified flow models by presenting an efficient gradient approximation during guided sampling. We begin by analyzing the error dynamics of general ordinary differential equations (ODEs) and then explain how the inherent properties of rectified flow models mitigate existing approximation issues. Building on these insights, we derive FlowChef, an intuitive yet theoretically grounded approach for free-form controlled image generation applicable to various downstream tasks, including those involving pretrained latent models.

##### 4.1. Error Dynamics of the ODEs

Understanding why existing methods often fail and require computationally intensive strategies is crucial. In ODEbased generative models, guiding the sampling process toward a desired target typically involves computing the gradient of a loss function with respect to the model’s parameters or state variables. As noted in Eq. (2), even though the denoised output can be estimated using xˆ0 ← Sample(xt,uθ(xt,t)), backpropagation through the ODE solver is still necessary to obtain ∇xtL. This raises the question: Why is backpropagation through the ODE solver necessary?

Approximating gradient computations is a common approach to reduce computational overhead [14, 47]. However, in models governed by nonlinear ODEs, unregulated gradient approximations can introduce significant errors into the system dynamics. This issue is formalized in the following proposition:

Proposition 4.1. Let p1 ∼ N(0,I) be the noise distribution and p0 be the data distribution. Let xt denote an intermediate sample obtained from a predefined forward function q as xt = q(x0,x1,t), where x0 ∼ p0 and x1 ∼ p1. Define an ODE sampling process dx(t) = f(xt,t)dt and quadratic L = ||xˆ0 −xref0 ||22, where f : Rd ×[0,T] → Rd is an ODESolver. Then, the error dynamics of ODEs for controlled image generation is governed by:

dE(t) dt

= −4sE(t) + 2e(t)Tϵ(t),

where e(t) = xˆ0−xref0 , E(t) = e(t)Te(t) is the squared error magnitude, s > 0 is the guidance strength, and ϵ(t) represents the accumulated errors due to non-linearity and trajectory crossovers.

The proof of Proposition 4.1 is provided in the Appendix. The term −4sE(t) denotes the exponential decay

of error due to guidance, while 2e(t)⊤ϵ(t) captures the impact of non-linearity and trajectory crossovers. In diffusion models, curved sampling trajectories lead to larger ϵ(t), hindering convergence. In contrast, rectified flow models exhibit straight trajectories with minimal crossovers, causing ϵ(t) to approach zero and allowing error to decrease exponentially.

To validate our findings, we conduct a toy study comparing classifier guidance on two ODE sampling methods using pretrained IDDPM and Rectified Flow++ (RF++) models on the ImageNet 64x64. As reported in Table 1, skipping the gradient in DDIM-based sampling increases the FID score, indicating significant ϵ(t). Conversely, RF++ converges well and improves the FID score. These empirical evidences further bolster our hypothesis that Rectified Flow models observe smooth vector field with the help of Proposition 4.1. Although backpropagating through the ODESolver further improves performance, it incurs higher computational costs as highlighted.

##### 4.2. FlowChef: Steering Within the Vector Field

Rectified flow models inherently allow error dynamics to converge even with gradient approximations due to their straight-line trajectories and smooth vector fields, as discussed previously. Hence, vector field uθ(xt,t) is trained to be smooth, and this smoothness implies that uθ changes gradually w.r.t. xt. We formalize our approach with the following assumptions about the Jacobian of the vector field:

- Assumption 1 (Local Linearity): Within the small neighborhoods around any point xt along the sampling trajectory, the vector field uθ(xt,t) behaves approximately linearly with respect to xt. Doing Taylor series expansion for small perturbations δ, we get:

uθ(xt + δ,t) ≈ uθ(xt,t) + Ju

θ

(xt,t)δ, (6) where Ju

θ

(xt,t) = du

θ(xt,t) dxt is the Jacobian matrix of

uθ with respect to xt.

- Assumption 2 (Constancy of the Jacobian): The Jaco-

bian Ju

(xt,t) varies slowly with respect to xt within these small neighborhoods. Therefore, for small δ, it can be approximated as constant:

θ

(xt,t). (7)

(xt + δ,t) ≈ Ju

Ju

θ

θ

Under these assumptions, we derive the following gradient relationship between ∇xtL and ∇xˆ0L:

Lemma 4.2 (Gradient Relationship). Let uθ : Rd × [0,T] → Rd be the velocity function with the parameter

θ. Then the gradient of the cost function (∇xtL) at any timestep t can be approximated as:

)T∇xˆ0L. (8)

∇xtL = (I + t · Ju

θ

###### Algorithm 1: Proposed FlowChef (generalized).

- 1 Input: Pretrained Rectified-flow model uθ, input noise sample xT ∼ N(0,I), target data sample xref0 , and L cost function.
- 2 for t ∈ {T...0} do

- 3 v ← uθ(xt,t)
- 4 dt ← 1/T
- 5 xt ← xt.require grad (True)

- 6 for N steps do

- 7 xˆ0 ← xt + t · v
- 8 loss ← L(ˆx0,xref0 )
- 9 xt ← Optimize(xt, loss) // Lemma 4.2

- 10 xt−1 ← xt + dt · v // Theorem 4.3

- 11 RETURN x0

Therefore, we get ϵ(t) = t · Ju

(xt,t)T∇xˆ0L. Importantly, when either t → 0 or Ju

θ

varies slowly (Assumption 2), the matrices I + t · Ju

θ

(xt,t) are close to the identity matrix. We further provide empirical evidence about this on pretrained rectified flow models by analyzing the gradients and convergence w.r.t. denoising steps in the Appendix. Where we observe that gradient direction improves linearly and quickly converges to xref0 as t → 0. Under this approximation, the difference between the two error dynamics becomes negligible. Since the ϵ(t) introduces only a small correction, it leads to the convergence in error dynamics as t → 0. Combining the results of Preposition 4.1, Assumption 1 and 2, and Lemma 4.2, we obtain the following theorem with straightforward proof that facilitates the controlled generation for rectified flow models in the most computationally efficient way:

θ

Theorem 4.3. (Informal) Given the above assumption and notations, the update rule for the vector field driven by uθ for the free-form controlled generation is:

xt−∆t = xt + ∆t · uθ(xt,t) − s′∇xˆ0L, (9) where s′ is the guidance scale.

The formal statement and proof are provided in the Appendix. This theorem forms the core of FlowChef, enabling controlled generation efficiently.

Algorithm Overview. Algorithm 1 provides a generalized overview of FlowChef. A key feature of FlowChef is that it starts from any random noise xT ∼ N(0,I) and still converges to the desired distribution or sample without inversion. At each timestep t, we first estimate the xˆ0. Then we calculate the loss L(ˆx0,xref0 ). At last, we directly

PSLD (500 NFEs)

FlowChef (Flux)

FlowChef (InstaFlow)

RectifID D-Flow Degraded

Resample

Reference

###### 14GB / 294 sec 19GB / 181 sec 18GB / 230 sec 33GB / 34 sec 14GB / 18 sec 64GB / 56 sec

|[Figure 28]|
|---|

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

|[Figure 36]|
|---|

|[Figure 37]|
|---|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

|[Figure 44]|
|---|

|[Figure 45]|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 4. Qualitative results on linear inverse problems. All baselines are implemented on stable diffusion v1.5, except FlowChef Flux variant. Results are reported for VRAM and time on an A100 GPU at 512 x 512 resolution, with Flux experiments at 1024 x 1024. Best viewed when zoomed in.

optimize xt using the gradient ∇xˆ0L, as per Lemma 4.2. That’s all we need! We may repeat this optimization N times per denoising step to stabilize gradients and improve convergence, though we found N = 1 sufficient in most cases. Important hyperparameters include the learning rate and total number of function evaluations (NFEs) T. Selecting optimal values for T and the learning rate is crucial to maintain gradients within a suitable range, uphold Jacobian constancy (Assumption 2), and avoid adversarial effects. To illustrate this, we analyze the effects of total FlowChef guidance steps on the Flux model (see Figure 3). Detailed study on this is in Appendix.

- 5. Experiments

extend both FlowChef and the baselines to latent-space models to simulate real-world applications, reporting results on PSNR, SSIM [50], and LPIPS [58] across 200 images from CelebA [26] and AFHQ-Cat [6]. Memory requirements and computation time are also analyzed.

###### 5.1.1. Pixel-space models

As FlowChef requires straightness and no crossovers, we select the Rectified-Flow++ pretrained models [22]. We compare FlowChef with recent flow-based methods OT-ODE [35], D-Flow [1], and PnP-Flow (concurrent work) [28], implementing the former two baselines manually due to lack of open-source access and tuning them for optimal performance. Additionally, we extend two diffusion-based baselines, DPS [7] and FreeDoM [56], for the RFMs. For comparisons, we use the Rectified-Flow++ models that are pretrained on FFHQ (for CelebA) and AFHQ-Cat datasets. Experiments are conducted for 64x64 image resolutions. Hyper-parameters for each method are reported in the Appendix. Our selected tasks include: (1) Box inpainting with 20x20 and 30x30 centered masks, (2) Super-resolution with 2x and 4x scaling factors, and (3) Gaussian deblurring with an 11x11 kernel at intensities of 1.0 and 10.0, with added Gaussian noise at σ = 0.05 for robustness.

We evaluate FlowChef across multiple tasks: (1) Linear inversion problems on pixel- and latent-space models, (2) Image editing, and (3) Classifier-guided style transfer. Overall, FlowChef demonstrates superior performance across all tasks, significantly reducing compute and time costs compared to baselines. Notably, FlowChef extends seamlessly to image editing tasks without inversion or additional memory overhead, allowing it to operate on recent SOTA T2I models, such as Flux, without encountering outof-memory (OOM) errors.

##### 5.1. Linear Inversion Problems

Results. We present the quantitative and qualitative evaluation results in Table 2 and Appendix, respectively. It can be observed that FlowChef significantly improves the per-

We evaluate FlowChef against several baselines on three common linear tasks: box inpainting, super-resolution, and Gaussian deblurring, under varying difficulty levels. We

BoxInpaint Deblurring Super Resolution PSNR (↑) SSIM (↑) LPIPS (↓) PSNR (↑) SSIM (↑) LPIPS (↓) PSNR (↑) SSIM (↑) LPIPS (↓)

Method

Easy Scenarios

Degraded 21.79 74.76 10.92 20.17 54.03 22.20 24.68 77.57 11.67 OT-ODE 19.11 77.86 13.49 21.86 62.51 15.14 21.64 62.23 26.64 PnP-Flow 22.12 68.02 14.70 22.00 65.79 15.95 22.42 68.06 14.91 D-FLow 20.37 70.06 13.67 20.22 61.99 14.51 21.60 69.89 12.29 FreeDoM 20.87 74.79 13.92 20.21 69.73 13.22 21.15 77.54 12.12 DPS 23.61 74.79 9.35 22.49 69.73 10.23 23.94 77.54 8.46

###### FlowChef (ours) 26.32 87.70 3.36 27.69 86.43 2.66 26.00 80.15 4.43

Hard Scenarios

Degraded 18.75 65.12 22.54 16.83 30.02 54.04 20.77 55.85 38.16 OT-ODE 16.37 67.35 19.22 17.89 34.02 29.68 18.19 39.43 36.84 PnP-Flow 20.44 61.96 17.53 19.50 50.54 22.00 21.35 61.78 17.78 D-FLow 18.34 62.62 19.94 16.93 34.13 25.31 20.01 56.46 17.64 FreeDoM 18.88 65.07 16.83 16.50 34.88 18.91 19.58 55.84 14.12 DPS 20.68 65.06 13.06 17.58 34.89 15.86 21.52 55.90 10.31 FlowChef (ours) 21.45 78.75 7.73 20.31 52.73 10.64 21.62 60.33 10.18

- Table 2. Pixel-space model-based evaluations for tackling the linear inverse problems. SSIM & LPIPS results are multiplied by 100.

Method

BoxInpaint Super Resolution Deblurring PSNR (↑) SSIM (↑) LPIPS (↓) PSNR (↑) SSIM (↑) LPIPS (↓) PSNR (↑) SSIM (↑) LPIPS (↓)

Diffusion based methods

Resample 20.12 79.94 19.36 26.91 70.91 30.75 25.27 62.97 41.94 PSLD (500 NFEs) 28.30 93.81 4.49 25.79 65.15 33.27 26.64 65.44 43.10 PSLD (100 NFEs) 26.90 93.13 5.29 21.95 54.67 46.08 21.25 51.62 51.92

Flow based methods

D-Flow 19.68 65.01 27.79 20.23 60.55 50.30 22.42 64.43 53.04 RectifID 23.81 75.13 10.50 10.36 31.55 67.08 10.40 31.16 66.60 FlowChef (InstaFlow) 22.94 73.55 9.94 25.83 64.73 31.38 22.50 47.42 42.54 FlowChef (Flux) 25.74 82.99 9.40 20.25 64.34 41.88 18.98 64.37 53.43

- Table 3. Latent-space model based evaluations for tackling the linear inverse problems. SSIM & LPIPS results are multiplied by 100.

Metric OT-ODE PnP-Flow D-Flow FlowChef

VRAM (GB) 0.70 0.40 6.44 0.43 Time (sec) 10.39 5.23 80.42 4.31

- Table 4. Compute requirement comparisons on a A6000 GPU.

formance on both easy and hard settings across the tasks and all metrics consistently. Notably from Table 4, we find that the FlowChef is also the fastest and most memory efficient. Surprisingly, diffusion-based extended baseline (DPS) significantly outperforms even recent baselines. However, DPS requires backpropagation through billions of parameters of ODESolver. While the concurrent gradientfree work, PnP-Flow, outperforms many other baselines, FlowChef leads the benchmark.

- 5.1.2. Latent-space models.

ple [46] for comparison. We use InstaFlow [25] (Stable Diffusion v1.5 variant) and Flux models as a baseline for flow-based approaches and utilize the original Stable Diffusion v1.5 checkpoint for the diffusion-based baselines. We perform all tasks in 512 x 512 resolution, increasing to 1024 x 1024 for Flux experiments. Our task settings are: (1) Box inpainting with a 128x128 mask, (2) Super-resolution at 4x scaling, and (3) Gaussian deblurring with a 50x50 kernel at intensity 5.0, all without extra Gaussian noise. For consistency, settings are doubled for Flux to a 256x256 mask, 8x super-resolution scaling, and 10.0 deblurring intensity. As VAE encoders add extra unwanted nonlinearity, pixel-level cost functions alone may not be optimal. Hence, we calculate the loss in the latent space only for the box inpainting task (as the degradation function is known with σ = 0), allowing us to extend to image editing later. For superresolution and deblurring, we stick with the pixel-level cost functions. We further detail the task-specific settings and hyperparameters in the Appendix.

Flow-based baselines are not extended to the latent space models as either they are already very computationally heavy or require extra Jacobian calculations to support the non-linearity introduced by the VAE models. We adapt DFlow [1] and RectifID [48] as flow-based baselines, adding diffusion-based baselines PSLD-LDM [43] and Resam-

Results. Quantitative and qualitative results in Figure 4 and Table 3 show that FlowChef achieves SOTA perfor-

FlowChef InfEdit (InstaFlow)

FlowChef (Flux)

Ledits++ DiffEdit

Input

|[Figure 52]|
|---|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

A dog wearing space suit with flowers in mouth.

|[Figure 58]|
|---|

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

###### An illustration of an owl sitting on a branch in a cave.

|[Figure 64]|
|---|

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

A cute little bunny pig with big eyes.

Figure 5. Qualitative results on image editing. As illustrated, our method attains the SOTA performance on comparison inversion-free methods. While FlowChef (Flux) variant achieves better quality and edits.

Preference Scores by Method

| |47.0% 18.0% 35.0%<br><br>40.4% 18.2% 41.4%<br><br>24.2% 26.3% 49.5%<br><br>48.0% 20.0% 32.0%| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

Ledits++

DiffEdit

InfEdit

FlowChef (Flux)

0 20 40 60 80 100

Percentage of Responses

Baseline Tie FlowChef (InstaFlow)

Figure 6. Human preference analysis for image editing.

mance for flow-based methods. However, a huge gap still remains w.r.t. the diffusion-based methods like Resample and PSLD. Notably, these baselines take about 5 minutes and 3 minutes, respectively, per image (see Figure 4), while FlowChef only takes only 18 seconds and less memory (only 14GB). None of the existing flow-based methods can be extended to Flux due to memory constraints. But FlowChef can seamlessly be applied, which further improves the performance. We find that FlowChef (Flux) reduces the artifacts in the images completely but observes the slight degradation in color dynamics. We attribute this

to the observed nonlinearity in the trajectory of Flux (detailed discussion in Appendix).

##### 5.2. Image Editing

We extend FlowChef for image editing on Flux and InstaFlow models, with Algorithm 2 detailing the implementation. This extension reduces FlowChef’s sensitivity to hyper-parameters. Currently, the approach requires a userprovided mask for controlled editing but can be expanded to attention-based techniques. Therefore, we select the baselines that also accept the user-provided mask for holistic comparisons. Due to their optimization constraints, existing baselines for classifier guidance cannot be applied to image editing. For comparison, we use diffusion-based SOTA methods Ledits++ [3] (which requires the inversion), DiffEdit [9] and InfEdit [53], alongside RF-Inversion [42] (the only concurrent flow-based editing framework). We perform large-scale evaluations on PIE-Bench [19]. For fair comparisons, we use PIE-Bench-provided ground truth masks for controlling all editing methods. Additionally, we provide preliminary comparisons with RF-Inversion for “wearing glasses” on randomly selected SFHQ faces [2].

|[Figure 70]|
|---|

|[Figure 71]|
|---|

[Figure 72]

Three pots on top of the table with blue, green, and green colors.

|[Figure 73]|
|---|

|[Figure 74]|
|---|

[Figure 75]

Four people dining at a restaurant and wearing red, blue, yellow, and black hats from left to right.

|[Figure 76]|
|---|

|[Figure 77]|
|---|

[Figure 78]

Four people dining at a restaurant and wearing red, yellow, yellow, and black hats from left to right.

Figure 7. FlowChef (Flux) multi object editing examples.

Method CLIP-I (↑) CLIP-T (↑) VRAM Time FreeDoM 0.5343 0.2541 17GB 80 sec MPGD 0.5285 0.2616 16GB 20 sec RetifID 0.4583 0.1702 18GB 30 sec D-Flow 0.4851 0.2591 23GB 5 sec

FlowChef(10 NFEs) 0.5044 0.2655 2 sec FlowChef(30 NFEs) 0.5301 0.2600 7 sec FlowChef(30 NFEs × 2) 0.5531 0.2478

14GB

12 sec

Table 5. Comparison of Various Classifier Guided Style Transfer.

Results. A human preference evaluation on randomly selected 100 PIE-Bench edits (see Figure 6) shows FlowChef (InstaFlow) outperforming DiffEdit and competing with InfEdit. Although Ledits++ scored highest, it requires inversion, resulting in higher VRAM and time requirements. Importantly, FlowChef on Flux achieves performance comparable to Ledits++ without inversion. Comparisons with RF-Inversion show that FlowChef reduces time by almost 50% without needing inversion and achieves competitive performance, with additional detailed quantitative and qualitative results in the Appendix.

##### 5.3. Classifier Guidance: Style Transfer

We conducted classifier-guided style transfer experiments using 100 randomly selected style reference images paired with 100 random prompts. The objective was to generate stylistic images that align visually with the reference style

RDFS-Rev FlowChef

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

|[Figure 82]|
|---|

An adorable cottage.

|[Figure 83]|
|---|

|[Figure 84]|
|---|

|[Figure 85]|
|---|

|[Figure 86]|
|---|

A steaming mug of hot chocolate with whipped cream.

Figure 8. Extending FlowChef to 3D multiview synthesis.

while adhering to the prompt. A pretrained CLIP model was used for evaluation, and we report both CLIP-T and CLIP-S scores [38]. For baseline comparisons, we included diffusion-based methods FreeDoM and MPGD and flowbased methods D-Flow and RectifID, which were extended for this task. The backbone was fixed to Stable Diffusion v1.5 (SDv1.5), with FlowChef evaluated in its InstaFlow variant to ensure a consistent comparison. Both quantitative and qualitative results are presented in Table 5, demonstrating the effectiveness of FlowChef in this setup.

##### 5.4. Extended Applications

To highlight the versatility and effectiveness of FlowChef, we extended our method to tackle multi-object image editing and 3D multiview generation. Figure 7 demonstrates FlowChef (Flux) performing complex multi-object edits, such as simultaneously modifying two pots and hats. Notably, this capability relies on the base model’s ability to understand textual instructions effectively. FlowChef leverages this strength of Flux, achieving edits without requiring inversion, a significant advantage over traditional methods. In Figure 8, we explore FlowChef’s multiview synthesis capability, inspired by Score Distillation Sampling (SDS) [37]. By incorporating the core idea of FlowChef for model steering into recent work on RFDS [54], we evaluate its effectiveness for 3D view generation. While FlowChef does not improve inference efficiency or reduce cost compared to RFDS-Rev [54], it demonstrates competitive performance in generating highquality multiview outputs. These results underline the adaptability of FlowChef, showcasing its potential for advanced generative tasks such as multi-object editing and 3D synthesis, while maintaining the state-of-the-art quality expected from RFMs.

#### 6. Conclusion

In this work, we introduced FlowChef, a versatile flowbased approach that unifies key tasks in controlled image generation, including linear inverse problems, image editing, and classifier-guided style transfer. Extensive experiments show that FlowChef outperforms baselines across all tasks, achieving state-of-the-art performance with reduced computational cost and memory usage. Notably, FlowChef enables inversion-free editing and scales to SOTA T2I models like Flux without memory issues. Our results demonstrate FlowChef’s adaptability and efficiency, offering a unified solution for both pixel and latent spaces across diverse architectures and practical constraints.

#### Acknowledgments

MP and YY are supported by NSF RI grants #1750082 and #2132724. We thank the Research Computing (RC) at Arizona State University (ASU) and cr8dl.ai for their generous support in providing computing resources. The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers.

#### References

- [1] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In Forty-first International Conference on Machine Learning, 2024. 2, 3, 6, 7, 4
- [2] David Beniaguev. Synthetic faces high quality (sfhq) dataset,

2022. 8

- [3] Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8861–8870, 2024. 2, 3, 8, 4
- [4] Andrew Brock. Large scale gan training for high fidelity natural image synthesis. arXiv preprint arXiv:1809.11096,

2018. 3

- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402, 2023. 2
- [6] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 6
- [7] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. 2, 3, 6, 4
- [8] Hyungjin Chung, Jeongsol Kim, Sehui Kim, and Jong Chul Ye. Parallel diffusion models of operator and image for blind

- inverse problems. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6059–6069, 2023. 4
- [9] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, 2023. 3, 8, 4
- [10] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio. A survey on diffusion models for inverse problems. arXiv preprint arXiv:2410.00083,

2024. 3, 4

- [11] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2, 3, 4
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1, 2, 3, 4
- [13] Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M. Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In The Twelfth International Conference on Learning Representations, 2024. 4
- [14] Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, et al. Manifold preserving guided diffusion. In The Twelfth International Conference on Learning Representations, 2024. 3, 4
- [15] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023. 4
- [16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 2, 4
- [17] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525, 2024. 3
- [18] Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12469– 12478, 2024. 3, 4
- [19] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based editing with 3 lines of code. arXiv preprint arXiv:2310.01506,

2023. 2, 3, 8, 4

- [20] Changhoon Kim, Kyle Min, Maitreya Patel, Sheng Cheng, and Yezhou Yang. Wouaf: Weight modulation for user attribution and fingerprinting in text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8974–8983, 2024. 9
- [21] Changhoon Kim, Kyle Min, and Yezhou Yang. Race: Robust adversarial concept erasure for secure text-to-image diffusion model. arXiv preprint arXiv:2405.16341, 2024. 9

- [22] Sangyun Lee, Zinan Lin, and Giulia Fanti. Improving the training of rectified flows. arXiv preprint arXiv:2405.20320,

2024. 2, 6

- [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. 2, 4
- [24] Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. 2, 4
- [25] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusionbased text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 3, 7, 4
- [26] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015. 6
- [27] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 4
- [28] S´egol`ene Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. arXiv preprint arXiv:2410.02423, 2024. 2, 3, 6, 4
- [29] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022. 3
- [30] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6038–6047, 2023. 3, 4
- [31] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 3
- [32] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept personalized text-to-image diffusion models by leveraging clip latent space. ArXiv, abs/2402.05195, 2024. 1, 4
- [33] Maitreya Patel, Changhoon Kim, Sheng Cheng, Chitta Baral, and Yezhou Yang. Eclipse: A resource-efficient text-toimage prior for image generations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9069–9078, 2024. 4
- [34] Xinyu Peng, Ziyang Zheng, Wenrui Dai, Nuoqian Xiao, Chenglin Li, Junni Zou, and Hongkai Xiong. Improving diffusion models for inverse problems using optimal posterior covariance. In Forty-first International Conference on Machine Learning, 2024. 4

- [35] Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inversion via flows,

2024. 3, 6, 4

- [36] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, et al. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720,

2024. 1

- [37] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In The Eleventh International Conference on Learning Representations, 2023. 9
- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. 9
- [39] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 3

- [40] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 2, 3
- [41] Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9472– 9481, 2024. 3
- [42] Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024. 3, 8
- [43] Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, and Sanjay Shakkottai. Solving linear inverse problems provably via posterior sampling with latent diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 7, 4
- [44] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [45] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, 2023. 1
- [46] Bowen Song, Soo Min Kwon, Zecheng Zhang, Xinyu Hu, Qing Qu, and Liyue Shen. Solving inverse problems with

- latent diffusion models via hard data consistency. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 7, 4
- [47] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023. 2, 3, 4
- [48] Zhicheng Sun, Zhenhao Yang, Yang Jin, Haozhe Chi, Kun Xu, Liwei Chen, Hao Jiang, Yang Song, Kun Gai, and Yadong Mu. Rectifid: Personalizing rectified flow with anchored classifier guidance. arXiv preprint arXiv:2405.14677, 2024. 3, 7
- [49] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 4
- [50] Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4): 600–612, 2004. 6
- [51] Zongze Wu, Nicholas Kolkin, Jonathan Brandt, Richard Zhang, and Eli Shechtman. Turboedit: Instant text-based image editing. arXiv preprint arXiv:2408.08332, 2024. 3, 4
- [52] Zihui Wu, Yu Sun, Yifan Chen, Bingliang Zhang, Yisong Yue, and Katherine L Bouman. Principled probabilistic imaging using diffusion models as plug-and-play priors. arXiv preprint arXiv:2405.18782, 2024. 3
- [53] Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural language. arXiv preprint arXiv:2312.04965, 2023. 3, 8, 4
- [54] Xiaofeng Yang, Cheng Chen, Xulei Yang, Fayao Liu, and Guosheng Lin. Text-to-image rectified flow as plug-and-play priors. arXiv preprint arXiv:2406.03293, 2024. 3, 9
- [55] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024. 4
- [56] Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 23174– 23184, 2023. 6, 4
- [57] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 5907– 5915, 2017. 3
- [58] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 6

## Steering Rectified Flow Models in the Vector Field for Controlled Image Generation

### Supplementary Material

#### 7. Supplementary Overview

This supplementary material contains proofs, detailed results, discussion, and qualitative results:

- • Section 8: Proposition 4.1 proof.
- • Section 9: Theorem 4.3 proof.
- • Section 10: Numerical accuracy analysis.
- • Section 11: Extended related works.
- • Section 12: Empirical study of pixel and latent models.
- • Section 13: Detailed algorithms.
- • Section 14: Experimental setup details.
- • Section 15: RF-Inversion vs. FlowChef.
- • Section 16: Hyperparameter study.
- • Section 17: Qualitative Results.
- • Section 18: Limitations & Future Work

#### 8. Proof of the Proposition

Proposition 4.1. Let p1 ∼ N(0,I) be the noise distribution and p0 be the data distribution. Let xt denote an intermediate sample obtained from a predefined forward function q as xt = q(x0,x1,t), where x0 ∼ p0 and x1 ∼ p1. Define a ODE sampling process dx(t) = f(xt,t)dt and quadratic L = ||xˆ0 − xref0 ||22, where f : Rd × [0,T] → Rd is a nonlinear function parameterized by θ. Then, under Assumption 1, the error dynamics of ODEs for controlled image generation are governed by:

dE(t) dt

= −4sE(t) + 2e(t)Tϵ(t),

- where e(t) is xˆ0−xref0 , E(t) = e(t)Te(t) is the squared

error magnitude, s > 0 is the guidance strength, and ϵ(t) represents the accumulated errors due to non-linearity and trajectory crossovers.

Proof. Consider the sampling process described by the ODE:

dx(t) dt

= f(x(t),t), (10)

- where f(x(t),t) is a nonlinear function often parameter-

ized via neural network θ. To guide the sampling process toward minimizing a loss function L(ˆx0,xref0 ), we can adjust the dynamics by adding the gradient ∇xt

to the vector field (see Eq. 2) as:

dx(t) dt

= f(x(t),t) − s · ∇xtL(ˆx0,xref0 ), (11)

where s is the guidance strength. Let e(t) = xˆ0−xref0 be the error between the estimated and target samples. Since xˆ0(t) = x(t) + t 0 f(x(τ),τ)dτ, differentiating e(t) with respect to t yields:

de(t) dt

dxˆ0(t) dt

(12)

=

dx(t)

dt − f(x(t),t) (13) = −s · ∇xtL(ˆx0,xref0 ). (14)

=

However, this requires the compute-intensive backpropagation through ODESolver. Therefore, it is important to find an approximation of ∇xt

. And the most convenient approximation is: ∇xt ≈ ∇xˆ0

. However, this derivation assumes that the integral t 0 f(x(τ),τ)dτ is well-behaved and that xˆ0(t) depends smoothly on x(t). In the presence of nonlinearity and trajectory crossovers, small changes in x(t) can lead to disproportionately large changes in xˆ0(t), due to the sensitivity of the integral to the path taken. Moreover, potential crossovers in the trajectory mean that the mapping from x(t) to xˆ0(t) is not injective; different trajectories x(t) may lead to the same xˆ0(t) or vice versa. This non-unique mapping complicates the error dynamics because ∇xˆ0L may not provide a consistent or effective direction for updating x(t). Including the effects of nonlinearity and trajectory crossovers, the error dynamics become:

de(t) dt

= −s · ∇xˆ0L(ˆx0,xref0 ) + ϵ(t), (15)

where ϵ(t) represents the errors introduced by the nonlinearity in f(x(t),t) and the sensitivity of xˆ0 to x(t) due to trajectory crossovers. In other words, the approximation error ϵ(t) can be represented as:

ϵ(t) = s · ∇xtL(ˆx0,xref0 ) − ∇xˆtL(ˆx0,xref0 ) . (16)

Assuming a quadratic loss function L = ||xˆ0 − xref0 ||22, we have ∇xˆ0L = 2e(t), leading to:

de(t) dt

= −2se(t) + ϵ(t). (17)

To understand the convergence of the error, we analyze the evolution of the error magnitude E(t) = e(t)Te(t). Differentiating E(t) with respect to time t, we get:

dE(t) dt

d dt

e(t)⊤e(t) (18)

=

de(t) dt

= 2e(t)⊤

(19) = 2e(t)⊤ (−2se(t) + ϵ(t)) (20) = −4se(t)⊤e(t) + 2e(t)⊤ϵ(t) (21) = −4sE(t) + 2e(t)⊤ϵ(t). (22)

This completes the proof.

| |
|---|

Notably, we derive this behavior of the ODE processes under the assumption that the error rate cannot be calculated accurately. This can either come from the incorrect estimation of xˆ0 or the nonlinearity of ODESolver itself. In the next section, we further concretize this with respect to the RFMs.

#### 9. Proof for Theorem

Lemma 4.2 (Gradient Relationship). Let uθ : Rd × [0,T] → Rd be the velocity function with the parameter θ. Then the gradient of the cost function L at any timestep t can be approximated as:

)T∇xˆ0L. (23)

∇xtL = (I + t · Ju

θ

Proof. Leveraging the straight-line trajectories characteristic of rectified flow models, the data sample at t = 0 can be estimated directly from an intermediate state xt:

This completes the proof.

| |
|---|

(Assumption 2), both gradients become approximately similar and ϵ(t) → 0. This guarantees the convergence of the error dynamics as time passes. We further show this behavior of RFMs empirically in Section 12 and show that this remains true for even largescale latent models.

Hence, as either t → 0 or Ju

θ

Theorem 4.3 (Update Rule for Steering the RFMs). Let uθ : Rd × [0,T] → Rd be a velocity field with constant Jacobian Ju

. Define the estimated initial state xˆ0 from an intermediate state xt by

θ

xˆ0 = xt + t · uθ(xt,t).

Consider the quadratic loss function L = ∥xˆ0 − xref0 ∥2, where xref0 is a reference sample. Then, the update rule for controlled generation is given by

xt−∆t = xt + ∆tuθ(xt,t) − s′∇xˆ0L, where:

- • ∇xˆ0L = 2(ˆx0 − xref0 ),
- • s′ ≈ (I + ∆t · Ju

θ

)(I + t · Ju

θ

)⊤,

- • I is the identity matrix.

Proof. By lemma 4.2 and Assumption 2, we can further approximate the Eq. 23:

xˆ0 = xt + t · uθ(xt,t). (24) By differentiating the xˆ0 with respect to xt, we get:

duθ(xt,t) dxt

dxˆ0 dxt

(25)

= I + t ·

(xt,t). (26) Using the chain rule for gradients:

= I + t · Ju

θ

T

dxˆ0 dxt

∇xˆ0L. (27) Substituting the expression for dxˆ0

∇xtL =

dxt , we obtain: ∇xtL = (I + t · Ju

(xt,t))T∇xˆ0L. (28)

θ

According to Assumption 3, due to the constancy of Jacobian, Ju

, for rectified flow models, we can treat it as constant for any time t. Hence, we get our desired approximation:

θ

(xt,t))T∇xˆ0L. (29)

∇xtL = (I + t · Ju

θ

)T∇xˆ0L ≈ KT∇xˆ0L, (30)

∇xtL = (I + t · Ju

θ

where K is the constant matrics as ∆t → 0 and t → 0. Under this formulation, we can perform controlled image generation in three steps:

- Step 1: xˆ0 = xt + t · uθ(xt,t)
- Step 2: xˆt = xt − KT∇xˆ0L
- Step 3: xt−∆t = xˆt + ∆t · uθ(ˆxt,t).

(31)

However, this will require additional forward passes. But according to Assumption 2 if ∆t is sufficiently small, then by Taylor series approximation, we get:

xt−∆t = xt − KT∇xˆ0L + ∆t · uθ xt − KT∇xˆ0L,t

(32)

= xt − KT∇xˆ0L

θ · KT · ∇xˆ0L (33) Now, as Ju

+ ∆t uθ(xt,t) − Ju

is constant w.r.t. ∆t. Hence, we get:

θ

)KT∇xˆ0L + ∆t · uθ(xt,t)

xt−∆t = xt − (I + ∆t · Ju

θ

(34)

= xt + ∆t · uθ(xt,t) − s′∇xˆ0L, (35) where s′ = (I + ∆t · Ju

)KT is constant and it can predetermined.

θ

Hence, this concludes the proof that for appropriate guidance scale s′, we can perform the controlled generation as derived above.

| |
|---|

#### 10. Numerical Accuracy for Model Steering

In our controlled generation framework, we aim to steer the generation process towards a reference sample xref0 by solving the modified ODE:

dx(t) dt

= f(x(t),t) = uθ(x(t),t) − s′∇xˆ0L. (36)

The accuracy of this numerical integration is crucial, as errors can accumulate over time, leading to deviations from the desired trajectory. The smoothness of the modified velocity field f(x(t),t) significantly impacts this accuracy. Specifically, a smaller magnitude of dt d f(x(t),t) reduces local truncation errors. The following Proposition formalizes this relationship, stating that the numerical accuracy improves as dt d f(x(t),t) decreases.

Proposition 10.1. (Informal). Given the prior notations, Assumptions, and Theorem, for any p-th order numerical method solving Eq. (36), the accuracy of the numerical solution increases as the magnitude of dt d f(x(t),t) decreases.

Proof. To analyze the local truncation error, consider the Taylor series expansion of the exact solution around time t when integrating backward in time from t to t − ∆t:

(∆t)2 2

d dt

x(t − ∆t) =x(t) − ∆tf(x(t),t) +

f(x(t),t)

(∆t)3 6

d2 dt2

f(x(t),t) + O (∆t)4 .

−

The numerical method updates the solution using:

xt−∆t = xt + ∆tϕ(xt,t), (37)

where ϕ(xt,t) is the increment function. The local truncation error τ is the difference between the exact solution and the numerical approximation:

τ = x(t − ∆t) − xt−∆t

(∆t)2 2

d dt

= x(t) − ∆tf(x(t),t) +

f(x(t),t)

(∆t)3 6

d2 dt2

f(x(t),t) + O (∆t)4 − [xt + ∆tϕ(xt,t)].

−

The first p-order terms cancel out, and we have:

||τ|| ≤

(∆t)p+1 (p + 2)!

dp+1 dtp+1

f(x(t),t) (38)

According to the Mean Value Theorem, we have

||τ|| ≤ C(∆t)p+1 max

t∈[tn,tn+1]

d dt

f(x(t),t) (39)

where C is a constant depending on the method. The global error e(t) = x(t)−xt accumulates these local errors over the integration interval. Under standard assumptions (e.g., Lipschitz continuity of f), the global error is bounded by:

d dt

∥e(t)∥ ≤ K(∆t)p eL(T−t) − 1 max

f(x(t),t) ,

t∈[0,T]

(40) where K is a constant depending on the Lipschitz con-

stant L of f and the total integration time T.

| |
|---|

As the magnitude of dt d f(xt,t) decreases, both the local truncation error and the global error decrease, enhancing the accuracy of the numerical solution. In the context of controlled generation, ensuring that f(xt,t) changes smoothly over time leads to more accurate integration and better alignment with the reference point xref0 . This insight and prior assumptions require that the guidance scale s′ and δt be sufficiently smaller, where higher NFEs lead to the lower ∆t. Hence, we increase the NFEs significantly to stabilize the steering (see Section 16). By carefully selecting s′, we ensure that the additional term s′ · ∇xˆ0L does not introduce excessive variability into f(x(t),t), maintaining the smoothness necessary for accurate numerical integration.

#### 11. Extended Related Works

Generative Models. Recent advances in generative models, especially diffusion models like Latent Diffusion Model (LDM) [40], GLIDE [31], and DALL-E2 [39], have significantly improved photorealism compared to GAN-based methods such as StackGAN [57] and BigGAN [4]. Pretrained diffusion models have been successfully applied to

diverse tasks, including image editing [15], personalization [32], and style transfer [49], but their inference flexibility remains limited, and they demand substantial resources [13, 33]. Distillation-based strategies like Latent Consistency Models [27] and Distribution Matching Distillation [55] address some limitations but lack control and broader applicability. Rectified Flow Models (RFMs) [23, 24], exemplified by Flux1, SD3 [12], and InstaFlow [25], show promise but face challenges in downstream tasks due to inversion inaccuracies and other limitations. This work addresses these gaps, extending RFMs to downstream tasks in a training-, gradient-, and inversion-free manner.

Conditional Sampling. Song et al. introduced noiseaware classifiers for controlling sampling in diffusion models [11], but these require task-specific training. Classifierfree guidance (CFG) [16] avoids this but necessitates an additional pretraining stage. FreeDoM [56] and MPGD [14] improve sampling control but remain computationally intensive. Initial extensions of conditional sampling to flow models face similar challenges, such as compute-heavy gradient backpropagation and limited applicability to latent space models. Our method, FlowChef, eliminates these issues, seamlessly enabling gradient- and inversionfree conditional sampling in latent-space models.

Inverse Problems. Inverse problems, dominated by diffusion-based methods [10], include pixel-space solutions such as DPS [7], Π-GDM [34], and BlindDPS [8]. PSLD [43] extends support to latent-space models, while manifold-based methods [14, 46] further enhance performance. Flow-based approaches like OT-ODE [35] and DFlow [1] improve speed and quality but remain resourceintensive. Recent advancements like PnP-Flow [28] achieve training- and gradient-free solutions for pixel-space models but face issues like smoothness artifacts. Existing solutions are resource-intensive and unsuitable for large-scale latent models. FlowChef leverages vector field properties of RFMs to enhance performance, generalization, and scalability for state-of-the-art models like Flux.

Image Editing. Image editing typically involves guiding a model to combine a reference image with an editing instruction, often through inversion [15, 18, 19, 30]. Inversion-free methods like DiffEdit [9], InfEdit [53], and TurboEdit [51] are rare, and none apply to flow models. Most state-of-the-art methods rely on cross-attention mechanisms [3, 30], which we do not prioritize. Our approach, FlowChef, introduces the first inversion-free image editing method for RFMs, achieving competitive results with

1https://huggingface.co/black-forest-labs/FLUX. 1-dev

state-of-the-art methods.

#### 12. Empirical Findings

In Section 4, we provided theoretical insights into FlowChef along with an intuitive algorithm. To complement the theory, we conducted an empirical analysis on large-scale RFMs to validate the Assumptions, Propositions, Lemmas, and Theorems presented. The results are summarized in Figure 9.

In Figure 9a, we compare the gradient cosine similarity with and without backpropagation through the ODESolver for InstaFlow and Stable Diffusion v1.5. For all denoising steps, the gradients of SDv1.5 behave nearly randomly, indicating that the stochasticity of the base model significantly impacts gradients, even when using the ODE sampling process during inference. In contrast, for InstaFlow, as denoising progresses (t → 0), gradient alignment improves, supporting our derivation in Lemma 4.2, which states that as t → 0, we achieve ∇xt ≈ ∇xˆ0

.

Further analysis was performed on the Rectified Flow++ model, which is designed for straight trajectories with zero crossovers. As shown in Figure 9b, well-trained models exhibit high gradient similarity even at the initial stages of denoising. However, as illustrated in Figure 9c, during active steering, the gradient direction initially diverges before improving. This behavior is also reflected in the convergence plot in Figure 9d.

We hypothesize that this phenomenon arises due to the proximity to the Gaussian noise space (p1 ∼ N(0,I)), where model steering is more error-prone since minor adjustments can disproportionately affect future trajectories. As denoising progresses and the distribution moves further from the noise (p1), these errors diminish, and convergence is achieved. These observations align well with our theoretical predictions, further reinforcing the validity of FlowChef.

#### 13. Algorithms

This section provides an overview of the algorithms underpinning FlowChef for image editing and its comparison to baseline methods for a comprehensive understanding.

Image Editing. As described in Section 4.2, FlowChef can be easily extended to image editing. Revisiting the core concept, FlowChef modifies random trajectories to align with a target sample. Image editing involves balancing similarity with the target sample while introducing deviations to achieve desired edits.

Figure 3 and Section 16 illustrate how FlowChef progressively transfers characteristics from high-level structure to finer details like color composition. However, editing requirements vary by task. For example, adding an object

InstaFlow

Stable Diffusion v1.5

0.8

CosineSimilarity

0.6

0.4

0.2

0 5 10 15 20 25

Inference Steps (NFEs)

(a) Gradient Similarity in InstaFlow vs. Stable Diffusion v1.5.

1.00

0.95

0.90

CosineSimilarity

0.85

0.80

0.75

0.70

0.65

0 20 40 60 80 100

Inference Steps (NFEs)

(b) Gradient Similarity in Rectified Flow ++ model.

0.9

0.8

CosineSimilarity

0.7

0.6

0.5

0.4

0.3

0 20 40 60 80 100

Guidance Steps / Inference Steps (NFEs)

(c) Gradient Similarity in Rectified Flow ++ during model steering.

1.2

1.0

MeanSquaredError(w.r.t.)rex0

0.8

0.6

0.4

0.2

0.0

0 20 40 60 80 100

Guidance Steps / Inference Steps (NFEs)

(d) Convergence in Rectified Flow ++ during model steering.

Figure 9. Empirical analysis of gradient similarity (a, b, and c) and convergence rate. (a) and (b) analyzes the gradients without model steering. (c) contains the gradient similarity during the active model steering. And (d) shows the trajectory similarity at each timestep t w.r.t. the inversion based trajectory.

###### Algorithm 2: FlowChef vs. Baseline FreeDoM.

- 1 Input: Pretrained Rectified-flow model uθ, input noise sample xT ∼ N(0,I), target data sample xref0 , and L cost function.
- 2 for t ∈ {T...0} do

- 3 v ← uθ(xt,t) dt ← 1/T

- 4 xt ← xt.require grad (True)

- 5 for N steps do

- 6 v ← uθ(xt,t)

- 7 xˆ0 ← xt + t · v
- 8 loss ← L(ˆx0,xref0 )
- 9 ∇xt ← grad(loss, xt) // Compute heavy BP

- 10 xt ← Optimize(xt, loss) // Lemma 4.2

- 11 v ← uθ(xt,t)

- 12 xt−1 ← xt +dt·v // Theorem 4.3

- 13 RETURN x0

benefits from trajectory adjustments earlier in the denoising process, while color changes require gradual learning at later stages. We can optimize parameters for diverse tasks using the generalized FlowChef, as detailed in Algorithm 1.

To simplify the process, we extend FlowChef to support off-the-shelf editing tasks, such as those in the PIEBenchmark, as detailed in Algorithm 2. Assume a non-edit region mask, Medit, derived from cross-attention or human annotation. To steer the trajectory towards the desired edits, we modify the velocity (v) using a classifier-free guidance strategy:

v = vedit + ¬mask · (vedit − vbase) · s, (41)

Algorithm 3: : FlowChef optimized for a wide range of image editing tasks.

- 1 Input: Pretrained Rectified-flow model uθ, input noise sample xT ∼ N(0,I), target data sample

xref0 , cedit is edit prompt, cbase is base prompt, M is user-provided input mask, and L cost function.

- 2 for t ∈ {T...0} do

- 3 dt ← 1/T
- 4 c ← [cedit,cbase]
- 5 v ← uθ(xt,t,c)
- 6 vedit,vbase = v.chunk(2)
- 7 v = vedit + ¬mask · (vedit − vbase) · s
- 8 Medit ← M
- 9 xt ← xt.require grad (true)

- 10 if t < minT then

- 11 for N steps do

- 12 xˆ0 ← xt + t · v
- 13 if t < max full stepsT then

- 14 Medit ← I

- 15 loss ← L(ˆx0,xref0 ) · Medit
- 16 xt ← Optimize(xt, loss) // Lemma 4.2

- 17 xt−1 ← xt + dt · v // Theorem 4.3

- 18 RETURN x0

where vedit corresponds to the edit prompt and vbase to the base (negative) prompt. This adjustment ensures the trajectory reflects the desired edits.

To maintain alignment of non-edited regions with the target sample, we modify the cost function as follows:

L(ˆx0,xref0 ) = ||(ˆx0 − xref0 ) · Medit||22. (42) Preserving the original image structure is crucial for edits such as color or material changes. To achieve this, we

Hyperparameter OT-ODE D-Flow PnP-Flow FlowChef Iterations / NFEs 200 20 50 200 Optimization per iteration 1 - - 1 Optimization per denoising - 50 - Avg. sampling steps - - 5 Guidance scale 1 1 1 500 Cost function L1 L1**2 L1 MSE initial time (1 means noise) 0.8 - - blending strength - 0.05 - inversion × ✓ × × learning rate 1 1 1 1

- Table 6. Hyperparameters for solving inverse problems using pixel-space models.

Hyperparameter D-Flow RectifID FlowChef Iterations / NFEs 10 4 100 Optimization per iteration - - 1 Optimization per denoising 20 400 Blending strength 0.1 - Guidance scale 0.5 0.5 0.5 Cost function MSE MSE MSE Learning rate 0.5 1 0.02 Optimizer Adam SGD Adam loss multiplier (latent/pixel) 0.000001 0.0001 / 100000 0.001/1000 inversion ✓ × ×

- Table 7. Hyperparameters for solving inverse problems using latent-space models (InstaFlow).

introduce the parameter max full steps T, which determines the number of steps that apply full FlowChef guidance with an identity mask. This ensures structural preservation while facilitating edits. Section 16 details a comprehensive reference for hyperparameters.

FlowChef vs. Baseline FreeDoM. Algorithm 2 compares FlowChef to the baseline FreeDoM, a diffusion model method that modifies the score function using a classifier guidance-like approach. FreeDoM requires estimating velocity and calculating gradients (∇xt

) through backpropagation via the ODESolver uθ, as marked in red. In contrast, as highlighted in green, FlowChef eliminates the need for backpropagation while still achieving convergence. This simplification makes FlowChef a more efficient and practical solution without sacrificing performance.

#### 14. Experimental Setup

This section outlines the hyperparameters used for FlowChef and baseline methods in solving inverse problems.

Pixel-Space Models. All evaluations were conducted using the Rectified Flow++ checkpoint. Since public implementations of OT-ODE and D-Flow are unavailable, we implemented these methods manually based on the provided pseudocode and performed hyperparameter tuning to ensure optimal performance. Notably, DPS and FreeDoM hyper-

parameters are the same as the FlowChef. Table 6 provides a detailed overview of the hyperparameters used for each baseline.

Latent-Space Models. For latent-space models, we extended D-Flow to the InstaFlow pretrained model, repurposed RectifID for inverse problems, and fine-tuned the hyperparameters for optimal results. The best-performing hyperparameters for each baseline are listed in Table 7. We utilized their baseline implementations for diffusion modelbased approaches such as Resample and PSLD-LDM, modifying only the number of inference steps. Specifically, we used 100 NFEs for Resample and 100/500 NFEs for PSLD.

#### 15. RF-Inversion vs. FlowChef

In this section, we briefly compare FlowChef with the concurrent work, RF-Inversion, which introduces an inversion strategy for rectified flow models using a linear quadratic regulator perspective from optimal transport, particularly for image editing tasks. RF-Inversion relies on image inversion, significantly increasing compute time—nearly doubling it compared to FlowChef. To evaluate, we conducted a “wearing glasses” editing task using 256 randomly selected SFHQ face images on the Flux.1[dev] model. As shown in Table 9, FlowChef achieves competitive performance in half the time. At a high level, RF-Inversion can be viewed as a special case of FlowChef, where the starting point is an inverted image rather than random noise. We applied a similar editing strategy to both methods for a fair comparison, as outlined in Algorithm 1, using a learning rate of 0.07, 20 optimization steps, and 30 total inference steps. On an A100 GPU, this configuration required approximately 15 seconds per inference. This comparison highlights the efficiency and versatility of FlowChef in handling image editing tasks.

#### 16. Hyper-parameter Study

Figures 10, 11, and 12 present an analysis of the impact of various hyperparameters on steering the InstaFlow model using FlowChef. Figure 10 demonstrates that a lower learning rate combined with a single optimization step is insufficient to effectively steer the model. Optimal performance is achieved with a learning rate of 0.1. Additionally, Figure 11 shows that lower learning rates necessitate more optimization steps to achieve convergence. Finally, Figure 12 illustrates how the denoising trajectory can be controlled by adjusting the learning rate and optimization steps, enabling recovery of the target sample with the desired accuracy. This control is particularly critical for image editing tasks, where striking the right balance between preserving the reference sample and applying the editing prompt is essential. Table 8 further highlights optimal hyperparameter

Model Hyperparameters Chage Object Add Object Remove Object Change Attrbiute Chage Pose Change Color Change Material Change Background Change Style

Learning rate 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.0 0.5 Max setps 50 50 50 50 20 30 50 50 30 Optimization steps 1 1 3 2 2 2 2 4 1 Inference steps 50 50 50 50 50 50 50 50 50 Full source steps 30 30 0 10 10 20 20 0 30 Edit guidance scale 2.0 2.0 2.0 4.5 8.0 8.0 4.0 3.0 6.0

FlowChef (InstaFlow)

Learning rate 0.4 0.5 0.5 0.5 0.5 0.4 0.5 0.5 0.4 Optimization steps 1 1 1 1 1 1 1 1 1 Inference steps 30 30 30 30 30 30 30 30 30 Full source steps 5 5 0 2 5 3 5 0 5 Edit guidance scale 4.5 4.5 4.5 4.5 7.5 10.0 4.5 0.0 10.0

FlowChef (Flux)

- Table 8. Hyperparameter examples for which various editing tasks can be performed (following Algorithm 2). Notably, the FlowChef (Flux) variant can be further optimized for task-specific settings that will follow Algorithm 1 with a careful selection of hyperparameters.

|[Figure 87]|
|---|

lr = 0.01

|[Figure 88]|
|---|

|[Figure 89]|
|---|

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

lr = 0.02 lr = 0.05 lr = 0.1

|[Figure 95]|
|---|

|[Figure 96]|
|---|

reference

Figure 10. Effect of FlowChef learning rate with fixed 20 max steps and one optimization step on InstaFlow.

Methods CLIP-I (↑) CLIP-T (↑) Time (↓) RF-Inversion 0.8573 0.2790 ∼31 sec

FlowChef (ours) 0.8269 0.2828 ∼15 sec

- Table 9. Comparison of FlowChef with concurrent work RFInversion on top of Flux for editing task “wearing glasses”.

hibits difficulties. On the other hand, FlowChef (Flux) achieves superior results, though it replaces a dog with a tiger instead of a lion in one instance. In the final example, both Ledits++ and FlowChef successfully edit long hair into short hair. Importantly, the results in Figure 14 are presented without cherry-picking, using consistent hyperparameters for both baselines and FlowChef. Variability in outcomes may still arise due to random seeds and fine-tuned hyperparameter selection.

settings for image editing tasks, providing valuable guidance for achieving high-quality edits. This study underscores the flexibility of FlowChef in adapting to diverse use cases by tuning these parameters effectively.

Figures 15, 16, 17, 18, 19, and 20 provide pixel-level qualitative results for various inverse problems, spanning inpainting, deblurring, and super-resolution tasks under both easy and hard scenarios. Readers are encouraged to zoom in to inspect these comparisons more closely. For each task, we randomly selected 10 CelebA examples and evaluated various baselines. Across all difficulty levels, FreeDoM, DPS, and PnPFlow demonstrate better performance than D-Flow and OT-ODE. However, FlowChef consistently outperforms all baselines, producing sharp and visually appealing results where other

#### 17. Qualitative Results

Figure 14 showcases additional qualitative examples of image editing tasks. For tasks such as changing materials or removing objects, FlowChef outperforms the baselines significantly. However, some limitations are noted: while FlowChef (InstaFlow) struggles to replace a cat with a tiger, InfEdit handles this task effectively, and Ledits++ ex-

|[Figure 97]|
|---|

|[Figure 98]|
|---|

|[Figure 99]|
|---|

|[Figure 100]|
|---|

|[Figure 101]|
|---|

|[Figure 102]|
|---|

|[Figure 103]|
|---|

|[Figure 104]|
|---|

|[Figure 105]|
|---|

|[Figure 106]|
|---|

|[Figure 107]|
|---|

|[Figure 108]|
|---|

Opt. steps = 1 Opt. steps = 2 Opt. steps = 3 Opt. steps = 4 Opt. steps = 5

reference

Figure 11. Effect of FlowChef optimization steps with fixed 20 max steps and 0.02 learning rate on InstaFlow.

|[Figure 109]|
|---|

|[Figure 110]|
|---|

|[Figure 111]|
|---|

|[Figure 112]|
|---|

|[Figure 113]|
|---|

|[Figure 114]|
|---|

|[Figure 115]|
|---|

|[Figure 116]|
|---|

|[Figure 117]|
|---|

|[Figure 118]|
|---|

lr = 0.01 Opt. steps = 1

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

|[Figure 124]|
|---|

|[Figure 125]|
|---|

|[Figure 126]|
|---|

|[Figure 127]|
|---|

|[Figure 128]|
|---|

lr = 0.01 Opt. steps = 5

|[Figure 129]|
|---|

|[Figure 130]|
|---|

|[Figure 131]|
|---|

|[Figure 132]|
|---|

|[Figure 133]|
|---|

|[Figure 134]|
|---|

|[Figure 135]|
|---|

|[Figure 136]|
|---|

|[Figure 137]|
|---|

|[Figure 138]|
|---|

lr = 0.1 Opt. steps = 1

Max steps = 0

Max steps = 1 Max steps = 2 Max steps = 3 Max steps = 4 Max steps = 5 Max steps = 10 Max steps = 15 Max steps = 20

reference

###### Figure 12. Effect of various FlowChef’s steering parameters with increasing maximum optimization steps on InstaFlow.

|[Figure 139]|
|---|

|[Figure 140]|
|---|

[Figure 141]

[Figure 142]

Deblurring Super-resolution

|[Figure 143]|
|---|

[Figure 144]

|[Figure 145]|
|---|

[Figure 146]

a small white blue lamb standing in the grass.

a green lipstick is being splashed with red powder.

Figure 13. FlowChef (Flux) model failures on inverse problems and image editing.

methods either fail outright or introduce excessive smoothness. Hard scenarios pose challenges for all methods, but FlowChef notably improves performance even under these conditions. While FlowChef shows promise, future

work is needed to address potential adversarial effects and further enhance robustness.

#### 18. Limitations & Future Work

Limitations. While FlowChef represents a significant leap in steering RFMs for controlled generation, it shares some limitations with its baseline counterparts. Hyperparameter tuning remains a challenge, particularly due to differences in trajectory behavior. For instance, while InstaFlow trajectories are relatively linear, Flux.1[Dev] trajectories exhibit non-linearity, necessitating careful tuning. As shown in Figure 7, FlowChef (Flux) faces difficulties in deblurring and super-resolution tasks, which we attribute to the pixel-space loss and non-linear behavior of the VAE model. Importantly, these limitations occur in less than 10% of cases and can often be resolved by simply adjusting the random seed. Furthermore, due to Flux’s lack of true classifier-free guidance (CFG), Algorithm 3 occasionally fails to perfectly execute color changes, some-

times producing the unaltered target image without reflecting the edit (see Figure 7). Despite these minor limitations, FlowChef still delivers state-of-the-art performance, making these challenges opportunities for further refinement rather than fundamental drawbacks.

Future Work. FlowChef opens a promising avenue for steering RFMs effortlessly with guaranteed convergence for controlled image generation. While this work extensively evaluates FlowChef on image generative models, future research should focus on expanding its capabilities to video and 3D generative models, areas that remain largely unexplored. Additionally, the current implementation assumes the availability of human-annotated masks for image editing. Automating this step with advanced attention mechanisms could make FlowChef a fully automated image editing framework. We encourage the research community to build upon this foundation to enhance its accessibility and functionality.

Ethical Concerns. As with all generative models, ethical concerns such as safety, misuse, and copyright issues apply to FlowChef [20, 21]. By enabling controlled generation with state-of-the-art RFMs, FlowChef can be leveraged for beneficial and harmful purposes. To mitigate these risks, future efforts should focus on solutions such as image watermarking, content moderation, and unlearning harmful behaviors. While these issues are not unique to FlowChef, addressing them will be key to ensuring its responsible use.

Ours (Flux)

Ours (InstaFlow)

InfEdit

Input Ledits++ DiffEdit

[Figure 147]

|[Figure 148]|
|---|

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

a colorful wooden bird sitting on a branch with a green background

|[Figure 153]|
|---|

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

A lion in a suit sitting at a table with a laptop.

|[Figure 159]|
|---|

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

A cat tiger sitting next to a mirror

|[Figure 165]|
|---|

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

A dog lion is laying down on a white background.

|[Figure 171]|
|---|

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

a woman with long short hair sitting in the sand at sunset

Figure 14. Qualitative results on image editing. Additional qualitative comparisons of FlowChef with the baselines.

[Figure 177]

Degraded

[Figure 178]

OT-ODE

[Figure 179]

FreeDoM

[Figure 180]

DPS

[Figure 181]

D-Flow

[Figure 182]

PnPFlow

[Figure 183]

FlowChef (ours)

[Figure 184]

Ground Truth

Figure 15. Qualitative examples of various methods for easy box inpainting task on RF++.

[Figure 185]

Degraded

[Figure 186]

OT-ODE

[Figure 187]

FreeDoM

[Figure 188]

DPS

[Figure 189]

D-Flow

[Figure 190]

PnPFlow

[Figure 191]

FlowChef (ours)

[Figure 192]

Ground Truth

Figure 16. Qualitative examples of various methods for hard box inpainting task on RF++.

[Figure 193]

Degraded

[Figure 194]

OT-ODE

[Figure 195]

FreeDoM

[Figure 196]

DPS

[Figure 197]

D-Flow

[Figure 198]

PnPFlow

[Figure 199]

FlowChef (ours)

[Figure 200]

Ground Truth

Figure 17. Qualitative examples of various methods for an easy deblurring task on RF++.

[Figure 201]

Degraded

[Figure 202]

OT-ODE

[Figure 203]

FreeDoM

[Figure 204]

DPS

[Figure 205]

D-Flow

[Figure 206]

PnPFlow

[Figure 207]

FlowChef (ours)

[Figure 208]

Ground Truth

Figure 18. Qualitative examples of various methods for the hard deblurring task on RF++.

[Figure 209]

Degraded

[Figure 210]

OT-ODE

[Figure 211]

FreeDoM

[Figure 212]

DPS

[Figure 213]

D-Flow

[Figure 214]

PnPFlow

[Figure 215]

FlowChef (ours)

[Figure 216]

Ground Truth

Figure 19. Qualitative examples of various methods for an easy super-resolution task on RF++.

[Figure 217]

Degraded

[Figure 218]

OT-ODE

[Figure 219]

FreeDoM

[Figure 220]

DPS

[Figure 221]

D-Flow

[Figure 222]

PnPFlow

[Figure 223]

FlowChef (ours)

[Figure 224]

Ground Truth

Figure 20. Qualitative examples of various methods for the hard super-resolution task on RF++.

