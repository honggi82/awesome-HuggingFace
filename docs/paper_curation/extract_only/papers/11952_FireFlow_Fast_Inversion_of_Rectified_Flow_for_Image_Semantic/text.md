## FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing

# arXiv:2412.07517v1[cs.CV]10Dec2024

##### Yingying Deng* Xiangyu He*1 Changwang Mei1 Peisong Wang1 Fan Tang2

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Source [C]Origami [C]Sculpture [R]Building [R]Running [+]Football [+]Hat

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Source [C]Cartoon [R]Cat [R]Crochet dog [R]Flower rings [R]Smile [+]Makeup

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Source [-]Flower [+]Rainbow Source [R]Chocolate bear Source [+]‘Coca cola’

Figure 1. FireFlow for Image Inversion and Editing in 8 Steps. Our approach achieves outstanding results in semantic image editing and stylization guided by prompts, while maintaining the integrity of the reference content image and avoiding undesired alterations. [+]/[-] means adding or removing contents, [C] indicates changes in visual attributes (style, material, or texture), and [R] denotes content or gesture replacements.

### Abstract

Though Rectified Flows (ReFlows) with distillation offers a promising way for fast sampling, its fast inversion transforms images back to structured noise for recovery and following editing remains unsolved. This paper introduces FireFlow, a simple yet effective zero-shot approach that inherits the startling capacity of ReFlow-based models (such as FLUX) in generation while extending its capabilities to accurate inversion and editing in 8 steps. We first demonstrate that a carefully designed numerical solver is pivotal for ReFlow inversion, enabling accurate inversion

*Equal contribution 1Institute of Automation, Chinese Academy of Sciences, Beijing, China 2Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China. Correspondence to: Xiangyu He <hexiangyu17@mails.ucas.edu.cn>.

preprint

and reconstruction with the precision of a secondorder solver while maintaining the practical efficiency of a first-order Euler method. This solver achieves a 3× runtime speedup compared to state-of-the-art ReFlow inversion and editing techniques, while delivering smaller reconstruction errors and superior editing results in a training-free mode. The code is available at this URL.

### 1. Introduction

The ability to accurately and efficiently invert generative models is critical for enabling applications such as semantic image editing, data reconstruction, and latent space manipulation. Inversion, which involves mapping observed data back to its latent representation, serves as the foundation for fine-grained control over generative processes. Achieving a balance between computational efficiency and numerical accuracy in inversion is particularly challenging for diffusion

models, which rely on iterative processes to bridge data and latent spaces.

Diffusion models have long been regarded as the gold standard for high-quality data generation (Ramesh et al., 2022; Rombach et al., 2022; Podell et al., 2024) and inversion due to their ability to capture complex distributions through stochastic differential equations (SDEs). Their success has often overshadowed deterministic approaches such as Rectified Flow (ReFlow) models (Liu et al., 2023), which replace stochastic sampling with ordinary differential equations (ODEs) for faster and more efficient transformations. Despite skepticism about their capabilities, ReFlow models have demonstrated competitive generative performance, with the FLUX model (Forest) emerging as a leading opensource example. FLUX achieves remarkable instructionfollowing capabilities, challenging the assumption that diffusion models are inherently superior. These advances motivate a closer investigation of ReFlow-based models, particularly in the context of inversion and editing, to develop simple and effective methods.

ReFlow models possess an underutilized advantage: a welltrained ReFlow model learns nearly constant velocity dynamics across the data distribution, ensuring stability and bounded velocity approximation errors. However, existing inversion methods for ReFlow models fail to fully exploit this property. Current approaches rely on generic Euler solvers that prioritize each step’s computational efficiency at the expense of accuracy or incur additional costs to achieve higher precision. As a result, the potential of ReFlow models to deliver fast and accurate inversion remains untapped.

In this work, we introduce a novel numerical solver for the ODEs underlying ReFlow models, addressing the challenges of inversion and editing. Our method achieves second-order precision while retaining the computational cost of a firstorder solver. By reusing intermediate velocity approximations, our approach reduces redundant evaluations, stabilizes the inversion process, and fully leverages the constant velocity property of well-trained ReFlow models. As shown in Table 1, our approach is the first to provide a solver that strikes an optimal trade-off between accuracy and efficiency, enabling ReFlow models to excel in inversion and editing tasks. By combining computational efficiency, numerical robustness, and simplicity, our method offers a scalable solution for real-world tasks requiring high fidelity and realtime performance, advancing the utility of ReFlow-based generative models like FLUX.

### 2. Preliminaries and related works

##### 2.1. Rectified Flow

Rectified Flow (Liu et al., 2023) offers a principled approach for modeling transformations between two distributions, π0

Table 1. Comparison of recent training-free inversion and editing methods based on FLUX, including inversion/denoising steps, NFEs (Number of Function Evaluations) for both inversion and editing, local truncation error orders for solving ODE, and the need for a pre-trained auxiliary model for editing. Our approach offers a simple yet effective solution to address the challenges.

Methods Add-it RF-Solver RF-Inv. Ours Steps 30 15 28 8 NFE 60 60 56 18 Aux. Model ✓ w/o w/o w/o Local Error O(∆t2) O(∆t3) O(∆t2) O(∆t3) (Tewel et al., 2024) for Add-it, (Wang et al., 2024) for RFSolver, (Rout et al., 2024) for RF-Inv.

and π1 , based on empirical observations X0 ∼ π0 and X1 ∼ π1 . The transformation is represented as an ordinary differential equation (ODE) over a continuous time interval t ∈ [0,1] :

dZt = v(Zt,t)dt, (1) where Z0 ∼ π0 is initialized from the source distribution, and Z1 ∼ π1 is generated at the end of the trajectory. The drift v : Rd ×[0,1] → Rd is designed to align the trajectory of the flow with the direction of the linear interpolation path between X0 and X1 . This alignment is achieved by solving the following least squares regression problem:

E

min

v

1

∥(X1 − X0) − vθ(Xt,t)∥22 dt , (2)

0

where Xt = tX1+(1−t)X0 denotes the linear interpolation path between X0 and X1.

Forward process seeks to transform samples X0 ∼ π0 to match the target distribution π1 . A direct parameterization of Xt is given by the linear interpolation Xt = tX1 + (1 − t)X0 , which satisfies the non-causal ODE:

dXt = (X1 − X0)dt. (3)

However, this formulation assumes prior knowledge of X1, rendering it non-causal and unsuitable for practical simulation. By introducing the drift v(Xt,t) , rectified flow causalizes the interpolation process. The drift v is fit to approximate the linear direction X1 − X0 , resulting in the forward ODE with X0 ∼ π0,:

dXt = v(Xt,t)dt, t ∈ [0,1]. (4)

This causalized forward process enables simulation of Zt without requiring access to X1 during intermediate time steps.

Reverse process generates samples from π1 by reversing the learned flow. Starting from X1 ∼ π1 , the reverse ODE is given by negating the drift term:

dXt = −v(Xt,t)dt, t ∈ [1,0]. (5)

This process effectively “undoes” the transformations applied during the forward flow, enabling the generation of X0 that follows the original distribution π0. The reverse process guarantees consistency with the forward dynamics by leveraging the symmetry of the learned drift v.

##### 2.2. Inversion

The inversion of real images into noise feature space, as well as the reconstruction of noise features back to the original real images, is a prominent area of research within diffusion models applied to image editing tasks (Lin et al., 2024; Brack et al., 2024; Miyake et al., 2023; Ju et al., 2024; Zhang et al., 2022; Huberman-Spiegelglas et al., 2024; Cho et al., 2024). The foundational theory of Denoising Diffusion Implicit Models (DDIM) (Song et al., 2021) involves the addition of predicted noise to a fixed noise during the forward process, which can subsequently be mapped to generate an image. However, this approach encounters challenges related to reconstruction bias. To address this issue, Null-Text Inversion (Mokady et al., 2023) optimizes an input null-text embedding, thereby correcting reconstruction errors at each iterative step. Similarly, Prompt-Tuning-Inversion (Dong et al., 2023) refines conditional embeddings to accurately reconstruct the original image. Negative-Prompt-Inversion (Miyake et al., 2023) replaces the null-text embedding with prompt embeddings to expedite the inversion process. Direct-Inversion (Ju et al., 2024) incorporates the inverted noise corresponding to each timestep within the denoising process to mitigate content leakage.

In contrast to the aforementioned Stochastic Differential Equation (SDE)-based formulations, rectified flow models that utilize ordinary differential equations (ODEs) offer a more direct solution pathway. RF Inversion (Rout et al., 2024) employs dynamic optimal control techniques derived from linear quadratic regulators, while RF-Solver (Wang et al., 2024) utilizes Taylor expansion to minimize inversion errors in ODEs. Nevertheless, achieving superior inversion results typically necessitates an increased number of generation steps, which can lead to significant computational time and resource expenditure. In this paper, we propose a fewstep ODE solver designed to balance effective outcomes with high efficiency.

##### 2.3. Editing

Image editing utilizing a pre-trained diffusion model has demonstrated promising results, benefiting from advancements in image inversion and attention manipulation technologies (Hertz et al., 2022; Cao et al., 2023; Meng et al., 2022; Couairon et al., 2023a; Deutch et al., 2024; Xu et al., 2024; Brooks et al., 2023). Training-free editing methods typically employ a dual-network architecture: one network

is dedicated to reconstructing the original image, while the other is focused on editing. The Prompt-to-Prompt (Hertz

- et al., 2022) approach manipulates the cross-attention maps within the editing pipeline by leveraging features from the reconstruction pipeline. The Plug-and-Play (Tumanyan
- et al., 2023a) method substitutes the attention matrices of self-attention blocks in the editing pipeline with those from the reconstruction pipeline. Similarly, MasaCtrl (Cao

- et al., 2023) modifies the Value components of self-attention blocks in the editing pipeline using values derived from the reconstruction pipeline. Additionally, the Add-it (Tewel
- et al., 2024) method utilizes both the Key and Value components of self-attention blocks from the source image to guide the editing process effectively.

3. Motivation

The ReFlow model operates under the simple assumption that Xt evolves linearly between X0 and X1, corresponding to uniform linear motion. Drawing an analogy to physics, it is natural to extend this linear motion to accelerated motion by incorporating an acceleration term:

dvt dt

= a(Xt,t),

dXt dt

= v(Xt,t), (6)

where Xt+1 = Xt + vt∆t + 21at∆t2 , and vt is equivalent to v(Xt,t) for simplicity. Recent works have empirically shown that training-based strategies (Park et al., 2024; Chen et al., 2024) for solving Equation (6) improve coupling preservation and inversion over rectified flow, even with few steps. Moreover, a training-free method (Wang

- et al., 2024) leveraging pre-trained ReFlow models has also demonstrated the utility of the second-order derivative of v in achieving effective inversion, essentially aligning with the principles of accelerated motion.

However, this observation appears counterintuitive. A welltrained ReFlow model, such as FLUX, generally assumes that vt approximates the constant value X1 − X0. Thus, the acceleration term at = dvt/dt theoretically approaches zero, as the learning target for vt is constant.

Connection to High-Order ODE Solvers: Instead of treating at as a continuous term, we reinterpret it through the lens of high-order ODE solvers. Using the finite-difference approximation at = (vt+∆t − vt)/∆t , we can rewrite the equation as:

Xt+1 = Xt + vt∆t +

- 1

- 2

at∆t2 = Xt +

- 1

- 2

(vt + vt+∆t)∆t,

which corresponds to the standard formulation of the secondorder Runge-Kutta method. This high-order approach allows fewer steps (or equivalently, larger step sizes ∆t ) to achieve the same accuracy as Euler’s method, since the global error of a p-th order method scales as O(∆tp).

This enables larger ∆t while maintaining the same error tolerance ϵ. Similarly, if we approximate at using at = (vt+1

2∆t − vt)/(12∆t) , the resulting position update becomes:

2∆t∆t, which corresponds to the standard midpoint method, another second-order ODE solver.

Xt+1 = Xt + vt+1

Impact on ReFlow Inversion: It is well-established that the global error in the forward process of ODE solvers benefits from higher-order methods. Likewise, inversion and reconstruction tasks also exhibit improved performance with high-order solvers, as they better preserve original image details during the inversion of ReFlow models. We formalize this property in the following statement.

Proposition 3.1. Given a p-th order ODE solver and the ODE dXt

dt = vθ(Xt,t), if the dynamics of the reverse pass satisfy dXt

dt = −vθ(Xt,t) which is Lipschitz continuous with constant L. The perturbation ∆T at t = T propagates backward to t = 0. The propagated error satisfies:

###### ∥∆0∥ ≤ e−LT∥∆T∥. (7)

Implication. The inversion error ∥∆T∥ introduced during the p-th order numerical solution propagates into the reverse pass, experiencing a slight reduction scaled by the Lipschitz constant L of the learned drift vθ(Xt,t). Despite this reduction, the overall reconstruction error ∥∆0∥ for the original image remains asymptotically of the same order, O(∆tp), where ∆t represents the integration step size. Consequently, high-order solvers are preferred in ReFlow to achieve accurate inversion and editing with fewer steps.

### 4. Method

Challenges with High-Order Solvers: While the use of high-order solvers is theoretically promising, it fails to yield practical runtime speedups. For a parameterized drift vθ(Xt,t), the runtime is determined by the Number of Function Evaluations (NFEs), i.e., the number of forward passes through the model vθ(Xt,t). High-order solvers require evaluating more points within the interval [t,t + 1], leading to a higher NFE per step, which negates any reduction in the number of steps and fails to improve overall computational efficiency.

For instance, the midpoint method achieves a local error of O(∆t3) and a global error of O(∆t2). Formally, it proceeds as follows:

∆t 2

vθ(Xt,t), (8)

Xt+∆t

= Xt +

2

∆t 2

. (9)

Xt+1 = Xt + ∆t · vθ Xt+∆t

,t +

2

This scheme requires two NFEs per step: one to compute Xt+∆t

,t+ ∆2t , effectively doubling the cost compared to the Euler method. The midpoint method leverages vt+∆t

and another for vθ Xt+∆t

2

2

to provide a more accurate estimate of X

2

t+1−Xt

∆t than vt, which inspires us to seek an alternative with lower computational cost.

A Low-Cost Alternative: The training objective of ReFlow implies that a well-trained model satisfies vθ(Xt,t) ≈ (X1 − X0) for all t. Leveraging this property, the most efficient approach would replace vt with v0, enabling one-step generation as proposed in the original ReFlow method (Liu et al., 2023). However, this simplification makes it difficult to incorporate conditional priors, as multi-step iteration is no longer required.

To maintain a multi-step paradigm, we propose a modified scheme that replaces vt with previous t − 1-step midpoint velocity v(t−1)+∆t

. This approach is formalized as:

rather than vt+∆t

2

2

∆t 2

(10)

vˆθ(Xt,t) := vθ X(t−1)+∆t

,(t − 1) +

2

load from memory

∆t 2

Xˆt+∆t

vˆθ(Xt,t) (11)

:= Xt +

2

∆t 2

Xt+1 = Xt + ∆t · vθ X ˆt+∆t

(12)

,t +

2

run & save to memory

In this scheme, only one NFE is required per step1, matching the computational cost of the Euler method. The key question, then, is whether this scheme retains the second-order accuracy of the original midpoint method.

For the local and global truncation error, we derive that if vθ(Xt,t) is well-trained and varies smoothly with respect to both X and t, the proposed scheme achieves the same truncation error as the standard midpoint method. This ensures that the modified approach retains the benefits of second-order accuracy while operating at the computational cost of a first-order solver.

Proposition 4.1. Let vˆθ(Xt,t) denote the reused velocity approximation in Equation 10, and vθ(Xt,t) denotes the exact velocity at time t. Then, the approximation satisfies the error bound: ||vˆθ(Xt,t) − vθ(Xt,t)|| ≤ O(∆t), under the following conditions:

- 1) Temporal Error: The temporal error is directly proportional to the time step ∆t, stemming from smoothness of vθ(X,t) in the time domain.
- 2) Spatial Error: The spatial error is dominated by O(∆t) ,

1Specifically, we perform two NFEs at t = 0 to initialize the conditions v0 and v0+∆t

for subsequent iterations. Python-style pseudo-code is provided in Sec.D.

2

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

(a) Euler Method (NFE = 20) (b) Midpoint Method (NFE = 20) (c) Ours (NFE = 20)

- Figure 2. Results on 2D synthetic dataset. We evaluate the performance of 2-Rectified Flow using the Euler solver, midpoint solver, and our proposed approach on a 2D synthetic dataset. The source distribution π0 (orange) and the target distribution π1 (green) are parameterized as Gaussian mixture models. For the Euler method, the number of sampling steps is set to N = 20, corresponding to an NFE of 20. Our approach generates samples that align more closely with the target distribution, achieving a better match in density and structure. Additionally, the trajectories of the samples exhibit greater straightness, adhering closely to the ideal of linear motion.

1 2 3 4 5 6 7 8 9

Steps

0.0

0.1

0.2

0.3

0.4

0.5

0.6

Value

Approximation Error Analysis

v v

t

Shaded area represents ±1 standard deviation

1 2 3 4 5 6 7 8 9

Steps

0.00

0.05

0.10

0.15

0.20

0.25

0.30

Value

Approximation Error Analysis

v v

t

Shaded area represents ±1 standard deviation

(a) Inversion and Reconstruction with Step=10

2.5 5.0 7.5 10.0 12.5 15.0 17.5

Steps

0.0

0.1

0.2

0.3

0.4

0.5

Value

Approximation Error Analysis

v v

t

Shaded area represents ±1 standard deviation

2.5 5.0 7.5 10.0 12.5 15.0 17.5

Steps

0.00

0.05

0.10

0.15

0.20

0.25

0.30

Value

Approximation Error Analysis

v v

t

Shaded area represents ±1 standard deviation

(b) Inversion and Reconstruction with Step=20

- Figure 3. Illustrations of the approximation error in velocity (∥vˆθ− vθ∥) as it evolves with inversion steps (left subfigures) and denoising steps (right subfigures), with ∆t included as a reference.

et al., 2023). As shown in Figure 2, the transport trajectories generated by our method are straighter, leading to improved accuracy while maintaining the same NFE as the Euler method, and even surpassing the performance of the standard midpoint method.

Numerical Results and Discussion To empirically validate the theoretical assumption that the reused velocity approximation error ∥vˆθ(Xt,t) − vθ(Xt,t)∥ is bounded by O(∆t), we conducted numerical experiments and analyzed the relationship between the approximation error and the time step size ∆t on FLUX-dev model during inversion and reconstruction. The results are summarized in Figure 3, which depicts the average approximation error across different step sizes, with the shaded area representing ±1 standard deviation.

The data exhibit the following key trend that the approximation error grows approximately linearly with the step size C · ∆t , consistent with the theoretical bound O(∆t). Despite the inherent variability of the error (illustrated by the shaded standard deviation), the magnitude of the error remains well-controlled and stable across most steps, further validating the robustness of the reused velocity approximation in practice.

due to the boundedness of ∂vθ

###### ∂X .

We formally prove in the appendix that when two required conditions are satisfied, leading to the following result: our modified midpoint method achieves the same truncation error as the standard midpoint method under these circumstances. Consequently, it is expected to exhibit smaller overall error while maintaining the same runtime cost as the first-order Euler method.

Image Semantic Editing: To ensure simplicity and fairness in comparison with other methods, we adopt the approach in (Wang et al., 2024), where the value features in self-attention layers during the denoising process are replaced with pre-stored value features generated during the inversion process, serving as a prior. Subsequently, a reference prompt is used as guidance to achieve semantic editing. Leveraging the superior image preservation of our numerical solver, our method does not require careful selection of timesteps or specific blocks for applying the replacements in self-attention layers, as suggested in the original paper. Instead, we uniformly apply this strategy to all self-attention layers solely at the first denoising step, which we find em-

Theorem 4.2. Consider a ReFlow model governed by ODE:

dt = vθ(X,t), where vθ(X,t) is smooth and bounded, and the solution Xt evolves over a time interval [0,T]. The modified midpoint method, defined in Equation (12) , achieves the same global truncation error O(∆t2) as the standard midpoint method, provided the reused velocity satisfies: ||vˆθ(Xt,t) − vθ(Xt,t)|| ≤ O(∆t).

dX

To highlight the advantages of our approach, we conduct experiments on synthetic data following the setup in (Liu

[Figure 28]

Table 2. Quantitive results on Text-to-Image Generation. Methods FLUX-dev RF-Solver Ours

2.73x Speedup

Steps 20 10 10 NFE (↓) 20 20 11 FID (↓) 26.77 25.93 25.16 CLIP Score (↑) 31.44 31.35 31.42 ODE Solver 1st-order 2nd-order 2nd-order

Smallest Reconstruction Error

2.0x Speedup

76% Error Reduction

- Figure 4. Image reconstruction errors versus denoising NFE: Our approach, compared to the first-order vanilla ReFlow inversion and second-order RF-solver, achieves lower reconstruction errors and demonstrates faster convergence with respect to NFE.

pirically effective. The inversion and denoising sampling processes are detailed in Algorithms 1 and 2.

- 5. Experiment

Table 3. Quantitative results for inversion and reconstruction using the FLUX-dev model (excluding the DDIM baseline). NFE includes both inversion and reconstruction function evaluations. Steps or computational costs are kept comparable across comparisons. Reconstruction is performed without leveraging latent features from the inversion process.

Steps NFE↓ LPIPS↓ SSIM↑ PSNR↑

DDIM-Inv. 50 100 0.2342 0.5872 19.72 RF-Solver 30 120 0.2926 0.7078 20.05 ReFlow-Inv. 30 60 0.5044 0.5632 16.57 Ours 30 62 0.1579 0.8160 23.87 RF-Solver 5 20 0.5010 0.5232 14.72 ReFlow-Inv. 9 18 0.8145 0.3828 15.29 Ours 8 18 0.4111 0.5945 16.01

- 5.1. Implementation Details

Baselines: This section compares FireFlow with DM inversion-based editing methods such as Prompt-to-Prompt (Hertz et al., 2022) , MasaCtrl (Cao et al., 2023), Pix2Pixzero (Parmar et al., 2023), Plug-and-Play (Tumanyan et al., 2023a), DiffEdit (Couairon et al., 2023b) and DirectInversion (Ju et al., 2024). We also consider the recent RF inversion methods, such as RF-Inversion (Rout et al., 2024) and RF-Solver (Wang et al., 2024).

fundamental T2I task. Following the setup in RF-solver, we evaluate a randomly selected subset of 10K images from the MSCOCO Caption 2014 validation set (Chen et al., 2015), using the ground-truth captions as reference prompts. The FID and CLIP scores for results generated with a fixed random seed of 1024 are presented in Table 2. In summary, our method delivers superior image quality while maintaining comparable text alignment performance.

Metrics: We evaluate different methods across three aspects: generation quality, text-guided quality, and preservation quality. The Fr´echet Inception Distance (FID) (Heusel et al., 2017) is used to measure image generation quality by comparing the generated images to real ones. A CLIP model (Radford et al., 2021) is used to calculate the similarity between the generated image and the guiding text. To assess the preservation quality of non-edited areas, we use metrics including Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018), Structural Similarity Index Measure (SSIM) (Wang et al., 2004), Peak Signal-to-Noise Ratio (PSNR), and structural distance (Ju et al., 2024).

5.3. Inversion and Reconstruction

Quantitative Comparison: We report the inversion and reconstruction results on the first 1K images from the Densely Captioned Images (DCI) dataset (Urbanek et al., 2024), using the official descriptions as source prompts. The results, shown in Table 3, demonstrate that our approach achieves a significant reduction in reconstruction error, whether compared at the same number of steps (yielding approximately 2× speedup) or at the same computational cost.

Steps: Since the number of inference steps can significantly impact performance, we follow the best settings reported for the RF-Solver to ensure a fair comparison: 10 steps for textto-image generation (T2I) and 30 steps for reconstruction. For editing, RF-Solver varies the number of steps by task, using up to 25 steps. In contrast, we find that our approach achieves comparable or better results using 8 steps. The ablation study is shown in Section E.

Qualitative Comparison: As shown in Figure 5, our approach provides an efficient and effective reconstruction method based on FLUX. The drift from the source image is significantly smaller compared to baseline methods, aligning with the quantitative results.

5.2. Text-to-image Generation

We compare the performance of our method against the vanilla rectified flow and the second-order RF-solver on the

Convergence Rate: We empirically compare the convergence rates of different numerical solvers during reconstruction, as shown in Figure 4. For a fair comparison, we use the demo “boy” image and prompt provided in the RF-solver

Table 4. Compare our approach with other editing methods on PIE-Bench. Method Model

###### Structure Background Preservation CLIP Similarity↑ Steps NFE↓ Distance↓ PSNR↑ SSIM↑ Whole Edited

Prompt2Prompt (Hertz et al., 2022) Diffusion 0.0694 17.87 0.7114 25.01 22.44 50 100 Pix2Pix-Zero (Parmar et al., 2023) Diffusion 0.0617 20.44 0.7467 22.80 20.54 50 100 MasaCtrl (Cao et al., 2023) Diffusion 0.0284 22.17 0.7967 23.96 21.16 50 100 PnP (Tumanyan et al., 2023b) Diffusion 0.0282 22.28 0.7905 25.41 22.55 50 100 PnP-Inv. (Ju et al., 2024) Diffusion 0.0243 22.46 0.7968 25.41 22.62 50 100

RF-Inversion (Rout et al., 2024) ReFlow 0.0406 20.82 0.7192 25.20 22.11 28 56 RF-Solver (Wang et al., 2024) ReFlow 0.0311 22.90 0.8190 26.00 22.88 15 60 Ours ReFlow 0.0283 23.28 0.8282 25.98 22.94 15 32 Ours ReFlow 0.0271 23.03 0.8249 26.02 22.81 8 18

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

NFE=6

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Source Image

‘’A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above.’

NFE=12

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

NFE=18

Prompt

Vanilla ReFlow RF-Inversion RF-Solver Ours

Figure 5. Qualitative results of image reconstruction. Our approach achieves faster convergence and superior reconstruction quality compared to baseline ReFlow methods utilizing the FLUX model. Difference images showing the pixel-wise variations between the source image and the reconstructed images are also provided.

source code. Our approach achieves the lowest reconstruction error with the fastest convergence rate, offering up to 2.7× speedup and over 70% error reduction. Figure 7 further illustrates the results when other approaches are fully converged at NFE = 120.

Table 5. Per-image inference time for ReFlow inversion-based editing measured on an RTX 3090. The baseline is a vanilla ReFlow model utilizing 28 steps for both inversion and denoising.

###### Resolution Time Cost Speedup

Vanilla ReFlow 512 × 512 23.76s 1.0× RF-Inversion 512 × 512 23.36s 1.02× RF-Solver 512 × 512 25.31s 0.94× Ours 512 × 512 7.70s 3.09×

##### 5.4. Inversion-based Semantic Image Editing

Quantitative Comparison: We evaluate prompt-guided editing using the recent PIE-Bench dataset (Ju et al., 2024), which comprises 700 images across 10 types of edits. As shown in Table 4, we compare the editing performance in terms of preservation ability and CLIP similarity. Our method not only competes with but often outperforms other approaches, particularly in CLIP similarity. Notably, our approach achieves high-quality results with relatively few editing steps, demonstrating its efficiency and effectiveness in maintaining the integrity of the original content while producing edits that closely align with the intended modifications.

Vanilla ReFlow 1024 × 1024 72.10s 1.0× RF-Inversion 1024 × 1024 71.35s 1.01× RF-Solver 1024 × 1024 78.80s 0.92× Ours 1024 × 1024 24.52s 2.94×

inconsistencies relative to the source image, as evident in the 3rd and 6th rows of the figure. Additionally, these methods frequently produce invalid edits, as shown in the 7th and 9th rows. While PnP-Inv excels at preserving the structure of the source image, it often fails to effectively apply the desired edits. Rectified flow model-based methods, such as RF-Inversion and RF-Solver, deliver better editing results compared to the aforementioned methods. However, they still face challenges with inconsistencies in non-editing areas. Overall, our method provides a more effective solution to these challenges, achieving superior results in both preservation and editing fidelity.

Qualitative Comparison: We present the visual editing results in Figure 6, which are consistent with our quantitative findings. Our method highlights a fundamental trade-off between minimizing changes to non-editing areas and enhancing the fidelity of the edits. In contrast, methods like P2P, Pix2Pix-Zero, MasaCtrl, and PnP often struggle with

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

[+]biWild[R]Castle[C]Plush[C]Real

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

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

gooserd [R]Child[+]Sun[R]Cat[+]Snow[R]Wave

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

Source Ours RF-Solver RF-Inversion DiffEdit MasaCtrl P2P

PnP-Inv. PnP Pix2Pix-Zero

Figure 6. Comparison with State-of-the-art editing methods.

Inference Speed: In Table 5, we compare the inference time of FireFlow with several recent models. The number of steps is based on those reported in the original papers or provided in open-source code. FireFlow is significantly faster than competing reflow models and does not require an auxiliary model for editing.

key limitations of existing inversion techniques, providing a scalable and efficient solution for tasks such as image reconstruction and semantic editing. Our work highlights the untapped potential of ReFlow-based generative frameworks and establishes a foundation for further advancements in efficient numerical methods for generative ODEs. We also provide a discussion of the limitations in Section F.

### 6. Conclusion

### References

We proposed a novel numerical solver for ReFlow models, achieving second-order precision at the computational cost of a first-order method. By reusing intermediate velocity approximations, our training-free approach fully exploits the nearly constant velocity dynamics of well-trained ReFlow models, minimizing computational overhead while maintaining accuracy and stability. This method addresses

Brack, M., Friedrich, F., Kornmeier, K., Tsaban, L., Schramowski, P., Kersting, K., and Passos, A. LEDITS++: limitless image editing using text-to-image models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 8861–8870. IEEE, 2024.

Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 18392–18402. IEEE, 2023.

Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., and Zheng, Y. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 22560–22570, October 2023.

Chen, T., Gu, J., Dinh, L., Theodorou, E., Susskind, J. M., and Zhai, S. Generative modeling with phase stochastic bridge. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=tUtGjQEDd4.

Chen, X., Fang, H., Lin, T., Vedantam, R., Gupta, S., Doll´ar, P., and Zitnick, C. L. Microsoft COCO captions: Data collection and evaluation server. CoRR, abs/1504.00325, 2015. URL http://arxiv.org/ abs/1504.00325.

Cho, H., Lee, J., Kim, S. B., Oh, T., and Jeong, Y. Noise map guidance: Inversion with spatial context for real image editing. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024.

Couairon, G., Verbeek, J., Schwenk, H., and Cord, M. Diffedit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023a.

Couairon, G., Verbeek, J., Schwenk, H., and Cord, M. Diffedit: Diffusion-based semantic image editing with mask guidance. In The Eleventh International Conference on Learning Representations, 2023b. URL https:// openreview.net/forum?id=3lge0p5o-M-.

Deutch, G., Gal, R., Garibi, D., Patashnik, O., and CohenOr, D. Turboedit: Text-based image editing using fewstep diffusion models. In Igarashi, T., Shamir, A., and Zhang, H. R. (eds.), SIGGRAPH Asia 2024 Conference Papers, SA 2024, Tokyo, Japan, December 3-6, 2024, pp. 41:1–41:12. ACM, 2024.

Dong, W., Xue, S., Duan, X., and Han, S. Prompt tuning inversion for text-driven image editing using diffusion models. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pp. 7396–7406. IEEE, 2023.

Forest, B. Flux. https://github.com/ black-forest-labs/flux. Accessed: [2024.12.8].

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 6626–6637, 2017.

Huberman-Spiegelglas, I., Kulikov, V., and Michaeli, T. An edit friendly DDPM noise space: Inversion and manipulations. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pp. 12469–12478. IEEE, 2024.

Ju, X., Zeng, A., Bian, Y., Liu, S., and Xu, Q. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. International Conference on Learning Representations (ICLR), 2024.

Lin, H., Wang, M., Wang, J., An, W., Chen, Y., Liu, Y., Tian, F., Dai, G., Wang, J., and Wang, Q. Schedule your edit: A simple yet effective diffusion noise schedule for image editing. CoRR, abs/2410.18756, 2024.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https: //openreview.net/forum?id=XVjTT1nw5z.

Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., and Ermon, S. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Miyake, D., Iohara, A., Saito, Y., and Tanaka, T. Negativeprompt inversion: Fast image inversion for editing with text-guided diffusion models. CoRR, abs/2305.16807, 2023.

Mokady, R., Hertz, A., Aberman, K., Pritch, Y., and CohenOr, D. Null-text inversion for editing real images using guided diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 6038– 6047. IEEE, 2023.

Park, D., Lee, S., Kim, S., Lee, T., Hong, Y., and Kim, H. J. Constant acceleration flow. In The Thirty-eighth Annual

Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=hsgNvC5YM9.

Parmar, G., Singh, K. K., Zhang, R., Li, Y., Lu, J., and Zhu, J. Zero-shot image-to-image translation. In Brunvand, E., Sheffer, A., and Wimmer, M. (eds.), ACM SIGGRAPH 2023 Conference Proceedings, SIGGRAPH 2023, Los Angeles, CA, USA, August 6-10, 2023, pp. 11:1–11:11. ACM, 2023. doi: 10.1145/ 3588432.3591513. URL https://doi.org/10.

1145/3588432.3591513.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=di52zR8xgf.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 8748– 8763. PMLR, 2021.

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with CLIP latents. CoRR, abs/2204.06125, 2022. doi: 10.48550/ ARXIV.2204.06125. URL https://doi.org/10.

48550/arXiv.2204.06125.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Rout, L., Chen, Y., Ruiz, N., Caramanis, C., Shakkottai, S., and Chu, W.-S. Semantic image inversion and editing using rectified stochastic differential equations. arXiv preprint arXiv:2410.10792, 2024.

Song, J., Meng, C., and Ermon, S. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

Tewel, Y., Gal, R., Samuel, D., Atzmon, Y., Wolf, L., and Chechik, G. Add-it: Training-free object insertion in images with pretrained diffusion models. arXiv preprint arXiv:2411.07232, 2024.

Tumanyan, N., Geyer, M., Bagon, S., and Dekel, T. Plugand-play diffusion features for text-driven image-toimage translation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 1921–1930. IEEE, 2023a.

Tumanyan, N., Geyer, M., Bagon, S., and Dekel, T. Plugand-play diffusion features for text-driven image-toimage translation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pp. 1921–1930. IEEE, 2023b. doi: 10.1109/CVPR52729.2023.00191. URL https://doi.org/10.1109/CVPR52729.

2023.00191.

Urbanek, J., Bordes, F., Astolfi, P., Williamson, M., Sharma, V., and Romero-Soriano, A. A picture is worth more than 77 text tokens: Evaluating clip-style models on dense captions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26700–26709, June 2024.

Wang, J., Pu, J., Qi, Z., Guo, J., Ma, Y., Huang, N., Chen, Y., Li, X., and Shan, Y. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 13(4): 600–612, 2004.

Xu, S., Huang, Y., Pan, J., Ma, Z., and Chai, J. Inversionfree image editing with natural language. 2024.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pp. 586–595. Computer Vision Foundation / IEEE Computer Society, 2018.

Zhang, Y., Huang, N., Tang, F., Huang, H., Ma, C., Dong, W., and Xu, C. Inversion-based creativity transfer with diffusion models. CoRR, abs/2211.13203, 2022.

### A. The Pseudo-code for Inversion and Editing

- Algorithm 1 Solving ReFlow Inversion ODE

Require: Discretization steps N, reference image X0, prompt embedding network Φ, Flux model v(·, ·, ·; φ), time steps t =

[t0, ..., tN−1]

Ensure: Structured noise X1

- 1: Initialize vt0(Xt0) = v(Xt0, t0, Φ(·); φ) {Run}
- 2: ∆t0 = t1 − t0
- 3: Xt

0+12∆t0 = Xt0 + 12∆t0 · vt0(Xt0)

- 4: Initialize vt

0+21∆t0(Xt

0+12∆t0) = v(Xt

0+21∆t, t0 + 12∆t0, Φ(·); φ) {Run & Save to GPU Memory}

- 5: Xt1 = Xt0 + ∆t0 · vt

0+21∆t0(Xt

0+21∆t0)

- 6: for i = 1 : N − 1 do
- 7: vˆti(Xti) ← vt

i−1+12∆ti−1(Xt

i−1+12∆ti−1) {Load from GPU Memory}

- 8: ∆ti = ti+1 − ti
- 9: Xt

i+21∆ti = Xti + 12∆ti · vˆti(Xti)

- 10: vt

i+21∆ti(Xt

i+12∆ti) = v(Xt

i+21∆ti, ti + 21∆ti, Φ(·); φ) {Run & Save to GPU Memory}

- 11: Xti+1 = Xti + ∆ti · vt

i+12∆ti(Xt

i+21∆ti)

- 12: if i==N − 1 then

- 12: Save Vtinv.N−1 to storage.
- 13: end if
- 14: end for
- 15: return X1, Vtinv.N−1 in Self-attention Layers

- Algorithm 2 Solving ReFlow Denoising ODE (Editing)

Require: Discretization steps N, reference text ”prompt”, structured noise X1, prompt embedding network Φ, Flux model v(·, ·, ·; φ),

time steps t = [tN−1, ..., t0], pre-computed Vtinv.N−1 in Self-attention Layers during inversion Ensure: Edited image X0

- 1: Initialize vtN−1(XtN−1) = v(XtN−1, tN−1, Φ(prompt); φ) {Replace VteditN−1 with Vtinv.N−1 in Self-attention & Run}
- 2: ∆tN−1 = tN−2 − tN−1
- 3: Xt

N−1+12∆tN−1 = XtN−1 + 12∆tN−1 · vtN−1(XtN−1)

- 4: Initialize vt

N−1+12∆tN−1(Xt

N−1+12∆tN−1) = v(Xt

N−1+12∆tN−1, tN−1 + 12∆tN−1, Φ(prompt); φ){Replace VteditN−1 with Vtinv.N−1 in Self-attention & Run & Save to GPU Memory}

- 5: XtN−2 = XtN−1 + ∆tN−1 · vt

N−1+12∆tN−1(Xt

N−1+12∆tN−1)

- 6: for i = N − 2 : 0 do
- 7: vˆti(Xti) ← vt

i+1+12∆ti+1(Xt

i+1+12∆ti+1) {Load from GPU Memory}

- 8: ∆ti = ti−1 − ti
- 9: Xt

i+12∆ti = Xti + 12∆ti · vˆti(Xti)

- 10: vt

i+21∆ti(Xt

i+12∆ti) = v(Xt

i+21∆ti, ti + 21∆ti, Φ(prompt); φ) {Run & Save to GPU Memory}

- 11: Xti−1 = Xti + ∆ti · vt

i+12∆ti(Xt

i+12∆ti)

- 12: end for
- 13: return X0

#### B. Technical Proofs This section provides detailed technical proofs for the theoretical results discussed in this paper.

##### B.1. Proof of Proposition 3.1

Proof. The reverse ODE is given by:

dx dt

= −v(x,t). (13)

Let xTrue(t) be the true solution of the reverse ODE, starting from xTrueT , and xPerturbed(t) be the solution starting from xNumericalT = xTrueT + ∆T. The initial condition difference is:

xPerturbedT (T) − xTrueT (T) = ∆T. (14)

Define the error ∆(t) as the difference between the perturbed and true solutions:

∆(t) = xPerturbed(t) − xTrue(t). (15) The dynamics of ∆(t) follow from the reverse ODE:

dxPerturbed(t) dt −

dxTrue(t) dt

d∆(t) dt

. (16) Substituting the reverse ODE for each term:

=

d∆(t) dt

= −v(xPerturbed(t),t) + v(xTrue(t),t). (17) Using the Lipschitz continuity of v(x,t), we have:

∥v(xPerturbed(t),t) − v(xTrue(t),t)∥ ≤ L∥∆(t)∥, (18) where L is the Lipschitz constant of v(x,t). Thus,

###### d∆(t)

dt ≤ L∥∆(t)∥. (19) Based on the definition of the derivative of a norm: d∥∆(t)∥ dt

d∆(t) dt

d∆(t)

∆(t) ∥∆(t)∥

d∆(t) dt ≤

∆(t) ∥∆(t)∥

dt ≤ L∥∆(t)∥, (20) This can be rewritten as:

·

·

=

=

d∥∆(t)∥ ∥∆(t)∥

≤ Ldt. (21) Integrate both sides from t = T (initial condition) to t = 0 (final condition):

0

0

d∥∆(t)∥ ∥∆(t)∥

Ldt. (22)

≤

T

T

Thus, the inequality becomes:

###### ln∥∆(0)∥ − ln∥∆(T)∥ ≤ −LT. (23) Simplify:

ln∥∆(0)∥ ≤ ln∥∆(T)∥ − LT. (24) Exponentiate both sides to remove the logarithm:

###### ∥∆(0)∥ ≤ ∥∆(T)∥e−LT. (25)

| |
|---|

- B.2. Proof of Proposition 4.1 Proof. Define the time interval between t and t − 1 as ∆t for simplicity, the reused velocity at t is given by:

∆t 2

), (26)

,(t − 1) +

###### vˆθ(Xt,t) = vθ(X(t−1)+∆t

2

###### where X(t−1)+∆t

is computed recursively using:

2

∆t 2

vˆθ(Xt−1,t − 1). (27) The exact velocity at t is:

X(t−1)+∆t

= Xt−1 +

2

###### vθ(Xt,t). (28)

To quantify the difference ||vˆθ(Xt,t) − vθ(Xt,t)||, we expand the reused velocity vˆθ(Xt,t) around the exact velocity vθ(Xt,t). Using a Taylor series expansion, expand vθ(X(t−1)+∆t

###### ,(t − 1) + ∆2t) around Xt and t :

2

∆t 2

∂vθ ∂X

∂vθ ∂t

∆t 2

) + O(∆t2). (29) The temporal difference is:

,(t − 1) +

) ≈ vθ(Xt,t) +

− Xt) +

(−∆t +

vθ(X(t−1)+∆t

(X(t−1)+∆t

2

2

= −∆t 2

∆t 2

. (30) Thus, the temporal contribution to the velocity difference is:

−∆t +

∆t 2

∂vθ ∂t

. (31)

−

The leading term introduces an error of O(∆t). The spatial difference is:

− Xt. (32) Using the recursive relationship:

X(t−1)+∆t

2

∆t 2

vˆθ(Xt−1,t − 1), (33) and the local truncation error of Euler method

X(t−1)+∆t

= Xt−1 +

2

Xt ≈ Xt−1 + ∆t · vθ(Xt−1,t − 1) + O(∆t2), (34) we subtract:

∆t 2

vˆθ(Xt−1,t − 1) + O(∆t2). (35) Simplify:

− Xt = −∆t · vθ(Xt−1,t − 1) +

X(t−1)+∆t

2

∆t 2

(ˆvθ(Xt−1,t − 1) − 2 · vθ(Xt−1,t − 1)) + O(∆t2). (36) Substitute this into the spatial term:

− Xt =

X(t−1)+∆t

2

∂vθ ∂X

∆t 2

∂vθ ∂X

(ˆvθ(Xt−1,t − 1) − 2 · vθ(Xt−1,t − 1)) + O(∆t2). (37) Combine both temporal and spatial difference:

− Xt) =

(X(t−1)+∆t

2

∆t 2

∂vθ ∂t

∂vθ ∂X

∆t 2

∂vθ ∂X

vθ(Xt−1,t−1))+O(∆t2). (38) Let δt = ||vˆθ(Xt,t) − vθ(Xt,t)||. From the above analysis and triangle inequality:

v ˆθ(Xt−1,t−1)−vθ(Xt−1,t−1) −

vˆθ(Xt,t)−vθ(Xt,t) =

(

+

∆t 2

δt−1 + O(∆t). (39) Unfold the recursion:

δt ≤

2

∆t 2 O(∆t) +

∆t 2

O(∆t) + ··· . (40)

δt ≤ O(∆t) +

This is a geometric series with common ratio ∆2t, summing to:

≈ 1 + ∆2t, so:

For small ∆t, 1−1

∆t 2

∞

δt ≤ O(∆t) ·

k=0

∆t 2

k

1 1 − ∆2t

. (41)

= O(∆t) ·

δt = ||vˆθ(Xt,t) − vθ(Xt,t)|| ≤ O(∆t). (42)

| |
|---|

- B.3. Proof of Theorem 4.2 Proof. Define the time interval as ∆t for simplicity, our modified midpoint method updates Xt+1 as:

∆t 2

, (43) where:

Xt+1 = Xt + ∆t · vθ Xt+∆t

,t +

2

∆t 2

vˆθ(Xt,t). (44) Substituting Xt+∆t

Xt+∆t

= Xt +

2

, the velocity term becomes:

2

∆t 2 ≈ vθ(Xt,t) +

∆t 2

∂vθ ∂t

∆t 2

∂vθ ∂X

vˆθ(Xt,t) + O(∆t2). (45) Under the assumption ||vˆθ(Xt,t) − vθ(Xt,t)|| ≤ O(∆t), we write:

vθ Xt+∆t

,t +

+

2

vˆθ(Xt,t) = vθ(Xt,t) + δv, (46) where ||δv|| ≤ O(∆t). Substituting this into the velocity term expansion:

∂vθ ∂X

∂vθ ∂X

∂vθ ∂X

δv. (47)

vˆθ(Xt,t) =

vθ(Xt,t) +

Since ||δv|| ≤ O(∆t), the additional term ∆2t · ∂v

∂X δv contributes an error of O(∆t2), which is of the same order as higher-order terms in the expansion. Thus, the velocity term becomes:

θ

∆t 2

∂vθ ∂t

∆t 2

∂vθ ∂X

∆t 2 ≈ vθ(Xt,t) +

vθ(Xt,t) + O(∆t2). (48) Substituting the expanded velocity back into the update equation:

vθ Xt+∆t

+

,t +

2

∆t 2

∂vθ ∂t

∆t 2

∂vθ ∂X

vθ(Xt,t) + O(∆t2) . (49) Simplify:

Xt+1 = Xt + ∆t · vθ(Xt,t) +

+

∆t2 2

∆t2 2

∂vθ ∂t

∂vθ ∂X

vθ(Xt,t) + O(∆t3). (50) The exact solution to the ODE is:

Xt+1 = Xt + ∆t · vθ(Xt,t) +

+

∆t2 2

∂vθ ∂t

X(t + ∆t) = X(t) + ∆t · vθ(Xt,t) +

∆t2 2

∂vθ ∂X

vθ(Xt,t) + O(∆t3). (51)

+

The modified midpoint method’s update matches the exact solution up to O(∆t2), confirming that the local truncation error remains O(∆t3). Since the local truncation error is O(∆t3), the global error accumulates over O(1/∆t) steps, resulting in O(∆t2).

| |
|---|

### C. Empirical Convergence Rate

In this section, we present the reconstruction errors of various methods with NFE up to 60. Our approach demonstrates rapid convergence, achieving consistently low reconstruction error upon full convergence to the minimum value. In contrast, vanilla ReFlow with the first-order Euler method exhibits a slow convergence rate, while RF-solver with a second-order truncation error shows an error increase after convergence, with optimal performance observed at around 25 steps.

### D. Python-style Pseudo-Code

In this section, we present Python-style pseudo-code to illustrate the core concept of our approach, which is remarkably simple yet delivers promising results. Except for the first iteration, where vˆ is initialized as None, subsequent iterations require only a single function call for model evaluation, matching the computational cost of the original ReFlow model (i.e., the Euler method). However, as shown in Figure 7, the convergence rate of our approach is significantly faster than that of the vanilla first-order method.

[Figure 144]

Figure 7. Visualization of the convergence rate of different order inversion and reconstruction method. With 60 NFE, our approach still enjoys the lowest reconstruction error and the fast convergence speed.

Table 6. Comparison on different editing methods. Results on PIE Bench are reported. Guidance terms indicate the guidance ratio settings used in the FLUX model during the denoising process.

Structure Background Preservation CLIP Similarity↑ Steps NFE↓ Distance↓ PSNR↑ SSIM↑ Whole Edited

Method Guidance

Add Q [1,2,2,...] 0.0590 17.72 0.7340 27.01 23.84 8 18 Add Q + Add K [1,2,2,...] 0.0537 18.35 0.7520 26.82 23.60 8 18 Add Q + Add K + Add V [1,2,2,...] 0.0416 19.63 0.7805 25.95 22.92 8 18

- Add Q [1,1,2,...] 0.0530 18.78 0.7580 26.99 23.85 15 32 Add Q + Add K [1,1,2,...] 0.0486 19.30 0.7721 26.68 23.59 15 32 Replace V [2,...] 0.0271 23.03 0.8249 26.02 22.81 8 18

- Add Q [2,...] 0.0710 16.49 0.7077 27.33 24.09 8 18 Add K [2,...] 0.0738 16.41 0.7066 27.25 24.01 8 18

- 1

|hat_velocity = None for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):<br><br>if hat_velocity is None:<br><br>velocity = model(X, t_curr) else:<br><br>velocity = hat_velocity X_mid = X + (t_prev - t_curr) / 2 * velocity velocity_mid = model(X_mid, t_curr + (t_prev - t_curr) / 2) hat_velocity = velocity_mid X = X + (t_prev - t_curr) * velocity_mid<br><br>return X<br><br>|
|---|

- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11

### E. Ablation Study

Editing Steps We conducted an ablation study by varying the number of editing steps from 2 to 12, as shown in Figure 8. The results reveal that at the 2-step setting, the editing prompts are less effective. However, as the number of steps increases, the editing performance improves significantly. The subject being edited with 8 steps are comparable to those with 10 or 12 steps, indicating that 8 steps are sufficient. Therefore, 8 steps are used in the subsequent experiments.

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

Source 2 steps 4 steps 6 steps 8 steps 10 steps 12 steps

Figure 8. Ablation Study on the Number of Editing Steps.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

(a) Source Image (b) Black Cat (c) Source Image (d) Raising Hands (e) Source Image (f) Storm-trooper

Figure 9. Illustrations on FireFlow Failure Cases.

### F. Limitations

We empirically observe that our approach struggles with editing tasks involving changes to object colors or uncommon scenarios in natural images. As illustrated in Figure 9, the cat’s color remains unsatisfactory after editing. Similarly, in less common scenes, such as when the person’s head is not visible in the image, the editing results are poor.

Another example involves an uncommon description, such as “a [stormtrooper] with blue hair wearing a shirt,” which also yields unexpected results. We attribute these issues to the simplicity of the editing strategy, which involves only replacing the V feature in the self-attention module. This approach appears insufficient for handling such scenarios.

We empirically find that incorporating K feature addition in the self-attention module can resolve these problems. Formally,

Qedit(Kedit + Kinv.) √

Self Attnedit = Softmax(

)Vedit (52)

d

However, this comes at the cost of diminished preservation of the original structure and background details. Details can be found in Figure 10. We also include an quantitative analysis on different editing strategies in Table 6. The results are consistent with the illustrations. It is evident that Equation 52 belongs to the category of “cross-attention,” focusing on merging features from the self-attention module during inversion with the corresponding features generated during the denoising process. This concept has been extensively discussed in diffusion model (DM) editing methods. Table 6 presents only the foundational attempts in this direction. We hope this will inspire further research and advancements in future work.

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

(a) Source Image (b) Black Cat (c) Source Image (d) Raising Hands (e) Source Image (f) Storm-trooper

Figure 10. Illustrations on FireFlow with K feature addition in Self-attention.

