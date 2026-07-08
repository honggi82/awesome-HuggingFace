### DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior

# arXiv:2508.00599v2[cs.CV]4Aug2025

Junzhe Lu1,* Jing Lin2,* Hongkun Dou3 Ailing Zeng4 Yue Deng3 Xian Liu5 Zhongang Cai6 Lei Yang6 Yulun Zhang7 Haoqian Wang1,† Ziwei Liu2,† 1Tsinghua University 2Nanyang Technological University 3Beihang University 4Independent Researcher 5NVIDIA Research 6SenseTime Research 7Shanghai Jiao Tong University https://dposer.github.io/

[Figure 1]

[Figure 2]

[Figure 3]

Novel Poses

Monocular Image Recovered Mesh

Completed Poses

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Partial Poses

[Figure 11]

VPoser

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

DPoser (ours)

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

(a) Pose Generation (b) Human Mesh Recovery

(c)

Multi-hypothesis Pose Completion

Relative Improvements

Figure 1. An overview of DPoser-X’s versatility and performance across multiple pose-related tasks. Built on diffusion models, DPoser-X serves as a robust and adaptable prior for 3D whole-body human pose modeling. Shown are scenarios in (a) pose generation, (b) human mesh recovery, and (c) pose completion. With up to 61% improvement across 8 benchmarks, DPoser-X consistently outstrips existing priors like VPoser [51] and NRDF [24], proving its superiority in tasks involving the human body, hand, and face.

##### Abstract

We present DPoser-X, a diffusion-based prior model for 3D whole-body human poses. Building a versatile and robust full-body human pose prior remains challenging due to the inherent complexity of articulated human poses and the scarcity of high-quality whole-body pose datasets. To address these limitations, we introduce a Diffusion model as body Pose prior (DPoser) and extend it to DPoser-X for expressive whole-body human pose modeling. Our approach unifies various pose-centric tasks as inverse problems, solving them through variational diffusion sampling. To enhance performance on downstream applications, we introduce a novel truncated timestep scheduling method specifically designed for pose data characteristics. We also propose a masked training mechanism that effectively combines whole-body and part-specific datasets, enabling our model to capture interdependencies between body parts while avoiding overfitting to specific actions. Extensive experiments demonstrate DPoser-X’s robustness and versatility across multiple benchmarks for body, hand, face, and full-body pose modeling. Our model consistently outperforms state-of-the-art alternatives, establishing a new benchmark for whole-body human pose prior modeling.

* Equal contribution. † Corresponding authors.

##### 1. Introduction

Human pose modeling is a fundamental research topic with broad applications, ranging from human-robot interaction to augmented and virtual reality experiences. Obtaining plausible and realistic human poses requires learning effective pose distribution from large-scale datasets, which can then serve as priors for downstream tasks like body model fitting, motion capture, and gesture recognition. Previous approaches to human pose prior modeling have primarily adopted techniques such as Gaussian Mixture Models (GMMs) [1], Variational Auto-encoders (VAEs) [51], and Neural Distance Fields (NDFs) [24, 65]. However, each of these approaches faces inherent limitations. GMMs can generate implausible poses due to their unbounded nature, and VAEs enforce a Gaussian prior in the latent that undermines expressiveness. NDFs, while promising for 3D surface modeling, struggle to generalize across the complex, high-dimensional manifolds of human pose. Moreover, due to the scarcity of whole-body pose data, most existing prior models focus on body-only poses, neglecting the wholebody domain which is crucial for detailed human modeling.

To overcome these limitations, recent advances in diffusion models [26, 62] offer a compelling new paradigm. Unlike VAEs with their restrictive latent bottlenecks, diffusion models show greater expressiveness in learning com-

plex distributions. This has led to state-of-the-art results in domains from image synthesis [14, 32] to human motion generation [44, 58] and multi-hypothesis pose estimation [9, 27]. However, these pose-related applications are typically designed for specific generation tasks or tailored to work with conditional inputs. The potential of diffusion models to serve as a universal human pose prior that benefits various pose-related tasks remains largely unexplored.

In this work, we introduce DPoser, a diffusion-based human pose prior that can be seamlessly integrated across diverse pose-related tasks via test-time optimization. Our approach begins with training an unconditional diffusion model. We then formulate various pose-centric tasks as inverse problems, solving them based on variational diffusion sampling [46], where DPoser serves as a regularization component. Furthermore, our investigations reveal that key pose information during diffusion is concentrated in the later timesteps, leading us to develop a novel truncated timestep scheduling strategy to enhance optimization performance. Finally, we extend DPoser to DPoser-X for whole-body pose modeling with a mixed training strategy to address the data scarcity issue. Extensive experiments demonstrate that DPoser-X outshines state-of-the-art pose priors in a range of downstream tasks involving the body, hand, and face. An overview of DPoser-X’s versatility and performance across pose-related tasks is shown in Fig. 1. In summary, our main contributions are as follows:

- • We introduce DPoser, a novel framework based on diffusion models that creates a robust and flexible human pose prior applicable across diverse pose-related tasks.
- • We analyze the impact of diffusion timesteps in the pose domain and propose truncated timestep scheduling for more efficient test-time optimization.
- • We present DPoser-X, the first whole-body pose prior, which incorporates a mixed training strategy that effectively leverages both whole-body and part-only datasets.
- • Extensive experiments demonstrate that DPoser-X outstrips previous pose priors on multiple benchmarks for body, hand, face, and full-body modeling.

##### 2. Methodology

###### 2.1. Preliminary: Diffusion Models

Diffusion models [26, 59, 61, 62] are a class of generative models that operate by reversing a predefined forward noising process. This forward process systematically corrupts data x0 ∼ pdata by adding Gaussian noise over a continuous or discrete time variable t ∈ [0,1]. A noisy sample xt at any given time t is produced according to:

xt = αtx0 + σtϵ, ϵ ∼ N(0,I) (1)

Here, αt and σt are predefined schedule functions such that as t increases from 0 to 1, the data distribution is gradually transformed into a tractable prior, typically an isotropic

Gaussian distribution N(0,I).

The generative aspect lies in learning to reverse this process. This is achieved by training a neural network, typically parameterized as a noise predictor ϵϕ(xt;t), to estimate the noise component ϵ from a given noisy sample xt. The network is optimized using the following L2-loss objective [26], where w(t) is a positive weighting function:

0∼pdata,ϵ∼N(0,I),t∼U[0,1] w(t)||ϵ − ϵϕ(xt;t)||22 , (2)

Ex

Once trained, the model can generate novel data by simulating the reverse diffusion trajectory. This procedure starts with a random sample from the prior, x1 ∼ N(0,I), and iteratively applies the learned denoiser ϵϕ to produce a clean sample x0. In practice, this reverse simulation is implemented by numerical solvers [62], such as the discrete samplers introduced in DDPM [26] and DDIM [60].

###### 2.2. Learning Pose Prior with Diffusion Models

SMPL-based pose representation. To build a flexible 3D human pose prior for the body, we propose to utilize the SMPL body model [43], which can be viewed as a differentiable function [J,V ] = M(θ,β) that maps body joint angles θ ∈ R3×21 and shape parameters β ∈ R10 to mesh vertices V ∈ R3×6890 and joint positions J ∈ R3×22. Our target is to model the distribution of joint angles p(θ).

Training of unconditional diffusion models. To this end, we adopt an unconditional diffusion model to learn the pose representation θ. This approach aligns with a task-agnostic strategy, focusing solely on the distribution of 3D poses. For the diffusion process, we employ the sub-VP SDE parameterization proposed in [62]. Specifically, the coefficients in

Eq. (1) can be obtained as: αt = exp −21 0 t ξ(s)ds and σt = 1 − exp − 0 t ξ(s)ds , where ξ(t) denotes linear scheduled noise scales.

During training, we sample a clean pose θ (also x0) from datasets and introduce noise to generate noisy samples xt according to the forward process (Eq. (1)). Then we apply the objective in Eq. (2) to train the noise predictor ϵϕ(xt;t) with weights w(t) = σt2 as suggested in [62].

###### 2.3. Optimization Leveraging Diffusion Priors

The acquired noise predictor, denoted as ϵϕ, permits the generation of novel poses through various samplers. Yet, integrating diffusion priors into general downstream tasks remains largely unexplored. We address this by reframing pose-related tasks as inverse problems and applying variational diffusion sampling [46] for efficient resolution.

Inverse problem formulation. Given an original signal x0 and a degraded measurement y, a typical inverse problem can be formulated as:

y = A(x0) + n, y,n ∈ Rd, x0 ∈ Rn, (3) where A symbolizes the degradation pattern and n consti-

2

[Figure 27]

[Figure 28]

Degrade

## −

Multi Steps

𝐴𝐴 (x0) y

[Figure 29]

[Figure 30]

Measurement Loss

Gradient Backpropagation

DPoser Loss

Current Pose x0 Optimized Pose

[Figure 31]

2

[Figure 32]

## −

diffuse denoise x0

x0

x𝑡𝑡

- Figure 2. Overview of the DPoser-regularized optimization framework. Task inputs (e.g., 2D keypoints in human mesh recovery) and current poses are used to compute the measurement loss based on the degradation pattern A(·) (e.g., camera projection). Meanwhile, DPoser regularization introduces noise to the current pose and applies a one-step denoiser to compute DPoser loss LDPoser.

tutes noise, assumed to be white Gaussian. x0 refers to pose representations in human models like SMPL [43]. This formulation allows us to approach various pose-centric tasks by adapting A and interpreting y accordingly:

- • Pose completion: Here, A serves as a mask matrix, with y being the observed incomplete pose data.
- • Inverse kinematics & Motion denoising: In this scenario, A applies forward kinematics of human models, treating y as the observed noisy 3D joints.
- • Human mesh recovery: A integrates forward kinematics and perspective camera projection to relate y to 2D joint observations in images (i.e. 2D keypoints).

The aim is to recover the original signal x0 based on the degraded measurement y. Specifically, our objective shifts to sampling from the posterior distribution p(x0 | y).

Solving inverse problems with diffusion models. To simulate this posterior sampling process, we adopt the variational diffusion sampling technique [46]. It employs a variational distribution q (x0 | y) := N(µ,σ2I) and aims to minimize the KL divergence between this variational distribution and the true posterior p(x0 | y). Further, under the assumption of zero variance (σ ≈ 0), the optimization problem of seeking x0 (i.e., µ) is demonstrated to be equivalent to minimizing the following loss function [46, 63]:

L = ∥y − A(x0)∥2 + wt(sg[ϵϕ(xt;t) − ϵ])⊤x0, (4) where the first term represents the task-specific measurement loss, and the second term corresponds to the regularization loss. Here, wt denotes the loss weights, ϵ is the standard Gaussian noise, and sg signifies stopped-gradient. The regularization procedure initiates by selecting a timestep t and perturbs the optimization variable x0 as per Eq. (1), resulting in xt. Then, the gradients [ϵϕ(xt;t)−ϵ] are applied. Introducing DPoser regularization. To shed more light on the working mechanism, we propose an equivalent but more

intuitive regularization term defined as:

LDPoser = wt||x0 − sg[xˆ0(t)]||22,where (5)

xt − σtϵϕ(xt;t) αt

. (6)

xˆ0(t) =

Here, xˆ0(t) functions as one-step denoising estimation using the diffusion model ϵϕ(xt;t), which recovers clean pose from diffused sample xt at any timestep t. The straightforward L2-loss encourages the current pose x0 towards a denoised, plausible pose distribution. Further, it is theoretically consistent with the gradient direction of the regularization loss during variational diffusion sampling in Eq. (4).

Proof: Differentiating Eq. (5) with respect to x0 yields:

∇x0

LDPoser = 2wt(x0 − xˆ0(t))

xt − σtϵϕ(xt;t) αt

xt − σtϵ

αt −

)

= 2wt(

σt αt

(ϵϕ(xt;t) − ϵ) ∝ (ϵϕ(xt;t) − ϵ). (7)

= 2wt

DPoser across pose-related tasks. As a regularization term, DPoser can be combined with task-specific measurement losses in various pose-related tasks. We demonstrate the applications like human mesh recovery (illustrated in Fig. 2) and inverse kinematics. Section G provides test-time details about more domains (e.g., hand and face) and tasks such as pose completion and motion denoising.

Human mesh recovery (HMR) aims to deduce the human pose and shape from monocular images. In this context, we refine the optimization function in SMPLify [1], integrating DPoser as a regularization term, LDPoser, and omitting the original intricate interpenetration error component. The modified objective, engaging both body pose θ and shape β parameters from the SMPL model [43], is defined as:

###### L(θ,β) = LHMR + wβLβ + wθLDPoser. (8)

The re-projection loss LHMR, acting as the measurement loss, is defined by:

###### λiρ ΠC (MJ(θ,β)i) − Jiest , (9)

LHMR =

i∈Joints

where MJ(θ,β) denotes SMPL’s forward kinematics. The camera function ΠC maps 3D joint coordinates into 2D space. Jest refers to the 2D keypoints estimated using offthe-shelf 2D pose estimators, with λi reflecting the confidence score. The Geman-McClure error function (ρ) is employed to assess the discrepancy in 2D joint locations. To avoid unrealistic poses when minimizing re-projection loss, our DPoser regularization LDPoser is introduced on the body pose θ. Moreover, a shape regularization term Lβ = ∥β∥22 is utilized to constrain the body shapes. Their weights are expressed as wθ, and wβ respectively.

Another application is inverse kinematics (IK), which estimates poses from noisy or incomplete 3D joint positions,

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

- (a)
- (b)
- (c)

#### ...

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

N = 10; timesteps: [1.0, 0.9, ..., 0.4, 0.3, 0.2, 0.1]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

N = 5; timesteps: [1.0, 0.8, 0.6, 0.4, 0.2]

Truncation: N = 5; timesteps: [0.5, 0.4, 0.3, 0.2, 0.1]

- Figure 3. Illustration of the rationale behind the proposed truncated timestep scheduling. We employ the DDIM sampler [60] with limited steps and visualize the generated poses. Our observations reveal that pose refinement occurs at later timesteps.

Jobs, where the set of known joints is assumed to be provided. The task-specific measurement loss, LIK, is an L2loss between the model’s 3D joints and the observed ones:

||MJ(θ,β)i − Jiobs||22. (10)

###### LIK =

i∈Known Joints

As with HMR, the full objective combines LIK with our DPoser regularization LDPoser. The inclusion of LDPoser is crucial for ensuring plausible poses, especially when 3D joint observations are sparse or noisy.

Given the structure of LDPoser, selecting the suitable diffusion timestep t is essential in the iterative optimization process. In the subsequent section, we address this by introducing our novel truncated timestep scheduling.

- 2.4. Test-time Truncated Timestep Scheduling In the diffusion process, previous research [5] on images shows that initial timesteps (larger t) correspond to the perceptual content, while later timesteps refine details. Pose data, however, lacks the similar structured layering and spatial redundancy, implying a need for a tailored timestep scheduling approach. As depicted in Fig. 3, we find that pose generation does not benefit from the early timesteps as image generation does. The significant stages of pose refinement occur at smaller t, specifically when t ≤ 0.3. With limited steps, the uniform scheduling, as tested in (b), proves less effective. In contrast, allocating these steps toward the latter end of the diffusion process, as in (c), yields better samples. This indicates that critical pose information is concentrated more heavily in the later timesteps.

Based on the above insights, we propose a shift from standard uniform timestep scheduling [6, 46] to a truncated strategy for pose data. Specifically, the timestep t for each

Algorithm 1 Test-time Optimization with DPoser Require: A trained diffusion model ϵϕ(xt;t), task-specific

loss Ltask, range of diffusion timesteps [tmax,tmin], number of optimization iterations N.

Ensure: Initialization of pose parameters x0

- 1: for iter = 0,1,...,N − 1 do
- 2: t ← tmax − (t

max−tmin)×iter

N−1 ▷ Timestep scheduling

- 3: Sample ϵ ∼ N(0,I)
- 4: xt ← αtx0 + σtϵ ▷ Forward diffusion
- 5: xˆ0(t) ← xt−σtαϵϕ(xt;t)

t

▷ One-step denoiser

- 6: LDPoser ← wt∥x0 − sg[xˆ0(t)]∥22 ▷ Regularization
- 7: Ltotal ← Ltask + LDPoser
- 8: Update x0 via backpropagation on Ltotal
- 9: end for
- 10: return x0

optimization step can be expressed as: t = tmax −

(tmax − tmin) × iter N − 1

. (11)

where N denotes the number of optimization iterations, and iter signifies the current iteration. Note that when the interval [tmax,tmin] is set to ([1.0, 0.0]), this strategy degenerates to the uniform linear timestep schedule used in prior works [6, 46]. This formulation is integral to our optimization framework, which is summarized in Alg. 1. The timestep interval is chosen based on the task’s noise scale (typically [0.15,0.05]). See Section E for details.

###### 2.5. Building Whole-body Pose Prior

To model the whole-body pose, we separately train three part-specific DPoser models to capture the distribution of body pose, hand pose, and facial expressions, following the training mechanism described in Section 2.2. For DPoserhand, we adopt the MANO [56] model with the learned target being hand joint angles θhand ∈ R3×15. For DPoserface, we utilize the FLAME [38] model, where the modeling target θface ∈ R103 includes 100 facial expression coefficients and a 3-dimensional jaw pose. As a baseline, the three models are combined directly, referred to as DPoserX-base. Specifically, it splits the whole-body pose input into body, hand, and face parts, processing each through its respective part-specific model.

DPoser-X-base, however, fails to capture the interactions between different parts, which is crucial for whole-body pose modeling. For instance, in relaxed standing poses, the left and right hand poses are usually mirrored. To address this issue, we introduce a fused module after the DPoserX-base model and train it on whole-body datasets, resulting in DPoser-X-fused. While this model performs well in tasks like completion, it exhibits limited pose diversity, as demonstrated in Section 3.5. This limitation stems from the fact that existing whole-body pose datasets mainly capture spe-

||Body pose<br><br>Left-hand pose<br><br>Right-hand pose Face expression<br><br>|
|---|
<br><br>Whole-body data<br><br>Sample<br><br>Predicted noise<br><br>Supervision<br><br>| | |
|---|---|
| | |
| | |
| | |
| | |
<br><br>𝜀𝜀<br><br>DPoser-X Network<br><br>[Figure 49]<br><br>DPoser-body Network<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]<br><br>DPoser-hand Network<br><br>[Figure 54]<br><br>DPoser-hand Network<br><br>[Figure 55]<br><br>DPoser-face Network<br><br>[Figure 56]<br><br>Fused Module<br><br>[Figure 57]<br><br>Add noise<br><br>|Body pose<br><br>Left-hand pose<br><br>Right-hand pose Face expression<br><br>|
|---|
<br><br>(a)<br><br>||Body pose<br><br>Unavailable data<br><br>Unavailable data<br><br>Unavailable data|
|---|
<br><br>|[Figure 58]| |
|---|---|
|[Figure 59]| |
| |[Figure 60]|
|[Figure 61]| |
| | |
<br><br>Part-only data Predicted noise<br><br>DPoser-X Network<br><br>|Body pose<br><br>No loss computation<br><br>No loss computation<br><br>No loss computation|
|---|
<br><br>|Body pose<br><br>Masked data<br><br>Masked data<br><br>Masked data|
|---|
<br><br>|[Figure 62]| |
|---|---|
|[Figure 63]| |
|[Figure 64]| |
|[Figure 65]| |
| | |
<br><br>Whole-body data Predicted noise<br><br>DPoser-X Network<br><br>|Body pose<br><br>Left-hand pose<br><br>Right-hand pose Face expression<br><br>|
|---|
<br><br>Sample<br><br>𝜀𝜀 Supervision Add noise<br><br>Sample<br><br>𝜀𝜀 Supervision Add noise<br><br>(b)|
|---|---|

- Figure 4. Overview of the DPoser-X methodology. (a) The whole-body network consists of frozen part-only networks, and a fused module trained on whole-body datasets. (b) The mixed training strategy utilizes part-only datasets by applying loss only to available parts. To prevent arbitrary predictions on unavailable parts, the whole-body data is sometimes randomly masked, and loss is applied to all parts.

cific actions (e.g., speech gestures or object-grabbing scenarios), resulting in degraded generalization.

To improve this, we propose a mixed training strategy, which utilizes whole-body, body-only, hand-only, twohands, and face-only datasets, leading to the DPoser-Xmixed model. Specifically, we treat part-only data as incomplete whole-body data, applying loss only to the available parts. In addition, to reduce the data type gap, we randomly mask the whole-body data and apply loss across all parts, forcing the network to predict masked parts. In practice, we implement the unavailable and masked data parts as the mean poses and enable masking at a probability of 20%. This mixed training strategy enables DPoser-X-mixed to maintain whole-body modeling ability while leveraging part-only datasets to enhance generalization.

##### 3. Experiments

In this section, we showcase the robustness and versatility of DPoser-X across a wide range of pose-centric tasks involving the human body, hand, face, and whole-body. Section G and H provide evaluation metrics and complete assessments including but not limited to body pose completion, hand mesh recovery, and face generation.

###### 3.1. Experimental Setup

Implementation details. DPoser-body is trained on the AMASS dataset [45] with the same splits as prior works [51, 65]. The model employs axis-angle representation for joint rotations, which we normalize to have zero mean and unit variance. The architecture consists of a fully connected neural network with about 8.28M parameters. It draws inspiration from GFPose [9] but omits conditional input pathways for our unconditional setting. We train this model for 800,000 iterations using the Adam optimizer with a learning rate of 2 × 10−4 and a batch size of 1280.

DPoser-hand and DPoser-face networks use a similar architecture. DPoser-hand uses the FreiHAND, DexYCB, HO3D, H2O, and ReInterHand datasets [3, 23, 35, 48, 76]. DPoser-face employs WCPA and MICA datasets [31, 75]. The DPoser-X network integrates last-layer features from pre-trained part models using a fully connected neural net-

[Figure 66]

[Figure 67]

[Figure 68]

(a) GAN-S [12] (b) Pose-NDF [65] (c) NRDF [24]

[Figure 69]

[Figure 70]

[Figure 71]

(d) VPoser [51] (e) DPoser (ours) (f) DPoser (ours)*

Figure 5. Qualitative comparison of generated human poses: (e) illustrates naturalistic poses aligned with real-world data, whereas (f) shows poses that, despite superior APD, lack natural appearance. *We use a DDIM sampler [60] with only 10 steps.

Sample source APD ↑ FID ↓ Prec. ↑ Rec. ↑ dNN↓ Real-world [45] 15.44 0.005 0.86 0.90 0.001

GMM [1] 16.28 1.02 0.13 0.34 4.37 VPoser [51] 10.75 0.66 0.29 0.42 3.74 GAN-S [12] 15.68 0.18 0.61 0.41 2.98 Pose-NDF [65] 18.75 5.92 0.02 0.00 9.08 NRDF [24] 22.82 0.64 0.03 0.99 6.69 DPoser 14.28 0.07 0.72 0.80 2.63 DPoser* 19.03 0.58 0.10 0.95 2.95

Table 1. Comparative analysis of pose generation metrics. *Indicates the use of a 10-step DDIM sampler.

work with identical residuals. Whole-body datasets include BEAT2, GRAB, ARCTIC, and EgoBody [16, 41, 64, 71]. After data source weight balancing, based on our mixed training strategy, DPoser-X-mixed is trained on around 65% whole-body, 14% body-only, 12% single-hand, 4% twohand, and 5% face-only data.

Comparison settings. We compare against SOTA pose priors including GMM [1], VPoser [51], GAN-S [12], PoseNDF [65], and NRDF [24]. The above works focus on the body, so we have trained the hand versions of VPoser and NRDF. Since NRDF relies on quaternion representations, VPoser is trained for face and whole-body comparisons. We also include an L2-regularization baseline. See Section G.9 for details of comparative methods implementation.

###### 3.2. Body-only Tasks

Body pose generation. We conduct generation experiments to assess the learned data distribution of pose priors. Since

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Images GMM Pose-NDF VPoser GAN-S DPoser (ours) GT

Figure 6. Visualization of human mesh recovery results (body only) on EHF [51] when fitting from scratch.

Initialization w/o fitting GMM [1] VPoser [51] Pose-NDF [65] NRDF [24] GAN-S [12] DPoser

from scratch 108.57 58.32 58.08 57.87 57.38 57.26 56.05 CLIFF [39] 56.62 51.02 49.39 49.50 49.27 49.58 49.05

Table 2. Performance comparison of human mesh recovery on the EHF dataset [51]. PA-MPJPE is reported as the metric.

generation is not an inverse problem, we employ a standard Euler-Maruyama discretization [62] with 1000 steps for DPoser’s pose generation. As shown in Fig. 5, DPoser generates visually diverse and realistic poses, indicating a well-learned prior distribution. In contrast, VPoser [51] exhibits limited diversity due to its mean-centric nature, constrained by the explicit Gaussian latents. Additionally, both Pose-NDF [65] and NRDF [24] struggle to project pure noise to plausible poses, resulting in unnatural outputs due to their limited generalization capabilities.

In terms of quantitative evaluation (Table 1), Pose-NDF and NRDF show high Average Pairwise Distance (APD) but poor Precision and dNN. The high APD is likely due to exaggerated poses, which should be avoided in pose priors. To verify this, we test a 10-step DDIM sampler [60] named DPoser* that is suboptimal by design. Despite superior APD, DPoser* performs poorly in FID and Precision, indicating that too few steps lead to divergence from the expected distribution. Our findings highlight the need for a balanced evaluation of quantitative and qualitative results.

Human mesh recovery. We probe the efficacy of DPoser in HMR, focusing on estimating human body pose and shape from monocular images. Following Pose-NDF [65], we conduct experiments on the EHF dataset [51] and benchmark our method against existing SOTA priors. Our optimization-based framework incorporates two initialization paradigms: (1) a baseline that utilizes mean poses, and (2) an advanced scheme that employs CLIFF [39], a pre-trained regression-based model tailored for HMR. We further compare with recent generation-based methods GFPose [9] and HuProSO3 [15] in Section H.1.

As presented in Table 2 and Fig. 6, when fitting from scratch, DPoser surpasses established SOTA priors like GAN-S [12] and NRDF [24]. Moreover, DPoser can refine initial pose estimations from SOTA regression-based models like CLIFF [39], aligning better with images.

Motion denoising. Though not initially designed for temporal tasks, DPoser adapts well in motion denoising. The task aims to recover clean body poses from noisy 3D joint

Methods AMASS [45] HPS [22]

w/o prior 24.19 23.67 VPoser [51] 23.42 22.78 Pose-NDF [65] 22.13 21.60 MVAE [40] 26.80 N/A HuMoR [55] 22.69 N/A DPoser 19.87 20.54

- Table 3. Performance metrics (MPJPE) for motion denoising.

Methods Setting

MPJPE ↓ MPVPE ↓ Vis. Occ. All. All.

w/o prior sparse 0.07 14.46 8.98 10.07 L2 prior sparse 0.84 13.84 8.89 8.94 VPoser [51] sparse 0.37 13.10 8.25 8.84 NRDF [24] sparse 0.11 13.92 8.66 9.53 DPoser-hand sparse 0.06 5.15 3.21 3.43

w/o prior fingertip 0.13 4.00 2.89 4.59 L2 prior fingertip 1.02 3.58 2.85 3.15 VPoser [51] fingertip 0.48 4.35 3.25 3.93 NRDF [24] fingertip 0.15 3.95 2.93 3.95 DPoser-hand fingertip 0.07 2.40 1.74 1.99

- Table 4. Quantitative evaluation of hand inverse kinematics on the ReInterhand dataset [48] under various masking settings.

positions in motion sequences. Following Pose-NDF [65] and HuMoR [55], we apply Gaussian noise (standard deviation of 40 mm) to the 60-frame sequences from the AMASS dataset [45]. We also test on the HPS dataset [22] without additional training to validate generalization. As shown in Table 3, DPoser sets a new standard in motion denoising, outperforming even specialized motion priors like HuMoR.

###### 3.3. Hand-only and Face-only Tasks

Hand inverse kinematics. We assess DPoser’s performance in hand inverse kinematics (IK) tasks using the ReInterhand dataset [48], considering various challenging conditions. Table 4 summarizes results across the two settings: sparse (60% keypoints masked) and fingertip (only 5 fingertip keypoints visible). DPoser consistently outperforms baselines, achieving the lowest error. Notably, in the sparse setting, DPoser reduces MPJPE by over 50% compared to other methods, showcasing its robustness in recovering ac-

Methods all side-view

w/o prior 12.97/16.07/13.15 13.15/16.22/13.26 L2 prior 12.21/15.15/12.74 12.29/15.22/12.62 VPoser [51] 12.23/15.13/12.71 12.20/15.35/12.98 DPoser-face 11.68/14.58/12.17 11.77/14.67/12.34

MICA [75] 9.03/11.12/9.24 9.29/11.71/10.04

+ w/o prior 9.01/11.09/9.15 9.47/11.80/9.84

+ L2 prior 9.96/12.37/10.44 10.01/12.49/10.62 + VPoser [51] 9.93/12.34/10.43 10.00/12.50/10.65 + DPoser-face 8.76/10.78/9.00 9.18/11.47/9.73

Table 5. Face reconstruction performance on the NOW benchmark [57]. Results are reported as median/mean/std of MPVPE.

[Figure 79]

[Figure 80]

- (a)
- (b)

Images L2 VPoser DPoser (ours) w/o prior

[Figure 81]

[Figure 82]

Images L2 VPoser DPoser (ours) MICA*

- Figure 7. Visualization of face reconstruction on the NOW benchmark [57]. (a) Fitting from scratch. (b) Initialization using MICA [75]. *MICA predicts only face shape without expressions; translational and global orientation are fitted for visualization.

Methods APDbody ↑ APDhands ↑ FID ↓ Prec. ↑ Rec. ↑

VPoser-X [51] 7.79 1.16 14.52 0.64 0.07 DPoser-X-base 14.45 2.34 90.78 0.00 0.00 DPoser-X-fused 11.78 1.93 3.71 0.32 0.77 DPoser-X-mixed 14.08 2.04 13.97 0.02 0.81

Table 6. Quantitative evaluation of whole-body pose generation.

curate hand poses from limited observations.

Face reconstruction. We evaluate DPoser on face reconstruction tasks using the NOW [57] benchmark. The target is to estimate face shape accurately from a single image. To assess model performance in more challenging scenarios, we collect and test on a side-view subset of NOW. For initialization, in addition to fitting from scratch, we use the SOTA face reconstruction model MICA [75].

As shown in Table 5, when fitting from scratch, DPoser achieves the lowest reconstruction errors across both overall and side-view cases. With MICA initialization, DPoser achieves the best performance, reducing mean error to 8.76 mm. In contrast, due to their mean-centric characteristics, L2 prior and VPoser [51] do not improve on MICA’s results, getting even worse results than the baseline without pose prior. Visualizations in Fig. 7 show DPoser’s ability to reconstruct realistic faces, handling occlusion effectively.

- 3.4. Whole-body Tasks Whole-body pose generation. We evaluate whole-body pose generation in Table 6 for VPoser-X and three DPoserX variants. See Fig. S-18 for visualization results. DPoser-

ARCTIC [16] BEAT2 [41] MPVPE ↓ APD ↑ MPVPE ↓ APD ↑

Methods

VPoser-X [51] 37.34/43.24/4.60 0.59 27.49/35.46/5.06 0.66 DPoser-X 21.81/30.99/6.10 1.24 15.92/25.89/6.04 1.18

- Table 7. Performance metrics (min/mean/std of MPVPE and APD) for whole-body pose completion on multiple datasets.

Methods

PA-MPVPE↓ PA-MPJPE↓ All Hands Face Body

w/o prior 72.94 18.80 11.60 86.40 GMM [1] & L2 prior 67.25 17.93 10.92 79.08 VPoser-X [51] 66.74 17.44 10.99 79.88 DPoser-X 60.98 15.60 9.75 73.00

SMPLer-X [2] 26.15 11.21 2.95 29.31 + w/o prior 26.33 10.34 2.86 28.69 + GMM [1] & L2 prior 25.60 9.99 2.78 28.12 + VPoser-X [51] 25.41 9.50 2.83 28.37 + DPoser-X 24.65 7.33 2.73 27.87

- Table 8. Whole-body mesh recovery results on ARCTIC [16].

X-fused produces more diverse body and hand poses compared to VPoser-X, which prioritizes dataset-consistent realism but has limited diversity. The DPoser-X-base model exhibits the highest diversity but deviates from the wholebody data distribution, as indicated by the high FID. Benefiting our mixed training strategy, DPoser-X-mixed strike a good balance between learning whole-body actions (e.g., expressive grabbing and talking) and preserving generalization on more diverse data sources. For conciseness, in subsequent experiments, we denote our DPoser-X-mixed model simply as DPoser-X when no confusion arises.

Whole-body pose completion. We conduct a challenging pose completion experiment where one hand is masked randomly. This task evaluates the ability to model interdependencies between whole-body parts, particularly between two hands. Given the inherent uncertainties, we obtain 10 hypotheses and evaluate them based on their min/mean/std of errors against the GT. APD across multiple solutions is computed to assess the solution diversity. The testing datasets include ARCTIC [16] and BEAT2 [41], focusing separately on cooperative manipulations and speech gestures involving two hands. As shown in Table 7, DPoserX achieves the lowest min-MPVPE and reflects task uncertainty well with high APD. This indicates that DPoser-X learns the correlation between whole-body parts effectively. Whole-body mesh recovery. We evaluate DPoser-X on the ARCTIC dataset [16] for whole-body mesh recovery. For fitting from scratch, 2D whole-body keypoints are detected by RTMPose [28], while for SMPLer-X [2] initialization, GT 2D keypoints are used. For comparison, we choose VPoser-X and a pose prior baseline that employs GMM [1] for body poses and L2 prior for hands and face.

Results in Table 8 show that DPoser-X outperforms VPoser-X [51] and GMM baselines across all metrics for

[Figure 83]

Images w/o prior VPoser-X GMM & L2 prior DPoser-X (ours) Keypoints GT

- Figure 8. Visualization of whole-body mesh recovery on the ARCTIC dataset [16]. DPoser-X can recover plausible whole-body poses from imperfect 2D keypoints detected by RTMPose [28], while other methods struggle to handle noisy inputs effectively.

Whole-body Mesh Recovery Motion Denoising PA-MPVPEall ↓ PA-MPVPEhands ↓ MPVPE ↓ MPJPE ↓

Scheduling

Random [52] 62.28 16.63 43.33 23.87 Fixed [4] 61.69 15.71 45.69 22.54 Uniform [6, 46] 62.13 17.32 39.72 20.80 Truncated (ours) 60.98 15.60 38.21 19.87

- Table 9. Ablation of timestep scheduling on key pose-related tasks

Methods

ARCTIC [16] BEAT2 [41] MPVPE ↓ APD ↑ MPVPE ↓ APD ↑

DPoser-X-base 25.49/36.94/8.13 1.43 21.98/33.41/8.20 1.37 DPoser-X-fused 21.51/30.37/5.96 1.14 15.51/24.58/6.25 1.12 DPoser-X-mixed 21.81/30.99/6.10 1.24 15.92/25.89/6.04 1.18

- Table 10. Ablation of training strategies for whole-body pose completion. min/mean/std of MPVPE and APD are reported.

Methods

PA-MPVPE↓ PA-MPJPE↓ All Hands Face Body

DPoser-X-base 72.79 17.21 5.41 74.68 DPoser-X-fused 72.06 18.12 5.35 75.27 DPoser-X-mixed 70.91 15.83 5.27 74.33

- Table 11. Ablation of training strategies for whole-body mesh recovery on the Fit3D dataset [19].

the entire body. When used alongside SMPLer-X, DPoserX further refines the initialization, especially on hands, an area often overlooked by the SOTA whole-body mesh recovery models. Qualitative results in Fig. 8 demonstrate that DPoser-X can recover plausible poses from noisy keypoints observations, unlike VPoser-X, which struggles with occlusion and produces unrealistic poses.

- 3.5. Ablation Study We conduct ablation studies to evaluate the effectiveness of the proposed truncated timestep scheduling and mixed training strategy. Experiments on data representations, network settings, and comparisons with other diffusion-based inverse problem solvers are available in Section I and J. Effectiveness of truncated timestep scheduling. We contrast our proposed scheduling strategy against three established methods—random [52], fixed [4], and uniform [6, 46] scheduling. The results in Table 9 demonstrate that our scheduling outperforms existing strategies on all the evaluated tasks. As outlined in Section 2.4, the timestep range should be selected based on the noise characteristic of each task. This actually provides a perspective to explain the per-

formance. The uniform scheduling (i.e., range as [1.0,0.0]) performs poorly on mesh recovery due to the low noise scale of this task. Meanwhile, the fixed scheduling (i.e., tmax = tmin) yields the worst results for motion denoising, since the poses are denoised gradually during optimization and the timestep t should decrease to adapt it.

Advantage of mixed training strategy. The mixed training strategy for DPoser-X combines part-only and wholebody datasets, allowing whole-body modeling and effectively preserving generalization. Beyond the generation experiments in Table 6, we evaluate DPoser-X variants on more downstream tasks. In whole-body pose completion (Table 10), DPoser-X-mixed delivers comparable accuracy to DPoser-X-fused in MPVPE, showing the ability to learn correlations between whole-body parts. In contrast, DPoser-X-base struggles to model such interactions, evidenced by much higher MPVPE. Furthermore, we conduct whole-body mesh recovery experiments on Fit3D [19], which contains challenging sports poses, without additional training. As shown in Table 11, DPoser-X-mixed achieves the best overall PA-MPVPE and outperforms both DPoser-X-fused and DPoser-X-base, especially in handrelated metrics. These results highlight the strength of the mixed training strategy, enabling DPoser-X-mixed to generalize well while maintaining whole-body modeling ability.

##### 4. Conclusion

We present DPoser, an unconditional diffusion-based pose prior model designed to support a wide range of poserelated tasks. DPoser is engineered for versatility, functioning as a simple L2-loss regularizer, and is further enhanced by our novel truncated timestep scheduling for testtime optimization. Unlike prior methods focused solely on the human body, DPoser models are developed for body, hand, and face, demonstrating their efficiency and robustness across various downstream tasks. Additionally, we introduce a mixed training strategy to construct the wholebody model, DPoser-X, which effectively integrates both whole-body and part-only datasets. This approach enables DPoser-X to capture correlations between whole-body parts while maintaining strong generalization. Comprehensive experiments substantiate DPoser-X’s superior performance over existing state-of-the-art pose priors, highlighting its potential for broad application.

##### 5. Acknowledgments

This research was supported by the National Key Research and Development Program of China (Project No. 2022YFB36066) and the Shenzhen Science and Technology Project (Grant Nos. KJZD20240903103210014, JCYJ20220818101004). This work was also supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012, MOE-T2EP20223-0002), and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

##### References

- [1] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In ECCV, 2016. 1, 3, 5, 6, 7, 12, 14, 17, 20, 22
- [2] Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Wang Yanjun, Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, et al. Smpler-x: Scaling up expressive human pose and shape estimation. Advances in Neural Information Processing Systems, 36, 2024. 7, 17
- [3] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, et al. Dexycb: A benchmark for capturing hand grasping of objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9044–9053, 2021. 5, 15
- [4] Hanbyel Cho and Junmo Kim. Generative approach for probabilistic human mesh recovery using diffusion models. In ICCV, 2023. 8, 12
- [5] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models. In CVPR, 2022. 4
- [6] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022. 4, 8, 23, 24
- [7] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. NeurIPS, 2022. 23, 24
- [8] Hyungjin Chung, Jeongsol Kim, Sehui Kim, and Jong Chul Ye. Parallel diffusion models of operator and image for blind inverse problems. In CVPR, 2023. 23
- [9] Hai Ci, Mingdong Wu, Wentao Zhu, Xiaoxuan Ma, Hao Dong, Fangwei Zhong, and Yizhou Wang. Gfpose: Learning 3d human pose prior with gradient fields. In CVPR, 2023. 2, 5, 6, 12, 18
- [10] Radek Danˇeˇcek, Michael J Black, and Timo Bolkart. Emoca: Emotion driven monocular face capture and animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20311–20322, 2022. 20, 21, 33

- [11] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio. A survey on diffusion models for inverse problems. arXiv preprint arXiv:2410.00083,

2024. 13, 24

- [12] Andrey Davydov, Anastasia Remizova, Victor Constantin, Sina Honari, Mathieu Salzmann, and Pascal Fua. Adversarial parametric pose prior. In CVPR, 2022. 5, 6, 12, 13, 14, 17
- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 14
- [14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021. 2, 14
- [15] Olaf D¨unkel, Tim Salzmann, and Florian Pfaff. Normalizing flows on the product space of so (3) manifolds for probabilistic human pose modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2285–2294, 2024. 6, 18
- [16] Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J Black, and Otmar Hilliges. Arctic: A dataset for dexterous bimanual handobject manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12943–12954, 2023. 5, 7, 8, 16, 20
- [17] Haven Feng. Photometric optimization. https : / / github . com / HavenFeng / photometric _ optimization, 2020. 17
- [18] Yao Feng, Haiwen Feng, Michael J Black, and Timo Bolkart. Learning an animatable detailed 3d face model from in-thewild images. ACM Transactions on Graphics (ToG), 40(4): 1–13, 2021. 16
- [19] Mihai Fieraru, Mihai Zanfir, Silviu Cristian Pirlea, Vlad Olaru, and Cristian Sminchisescu. Aifit: Automatic 3d human-interpretable feedback models for fitness training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9919–9928, 2021. 8, 16, 20, 22, 36
- [20] Georgios Georgakis, Ren Li, Srikrishna Karanam, Terrence Chen, Jana Koˇseck´a, and Ziyan Wu. Hierarchical kinematic human mesh recovery. In ECCV, 2020. 12
- [21] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020. 12
- [22] Vladimir Guzov, Aymen Mir, Torsten Sattler, and Gerard Pons-Moll. Human poseitioning system (hps): 3d human pose estimation and self-localization in large scenes from body-mounted sensors. In CVPR, 2021. 6, 14, 19
- [23] Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honnotate: A method for 3d annotation of hand and object poses. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3196–3206, 2020. 5, 15
- [24] Yannan He, Garvita Tiwari, Tolga Birdal, Jan Eric Lenssen, and Gerard Pons-Moll. Nrdf: Neural riemannian distance

- fields for learning articulated pose priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1661–1671, 2024. 1, 5, 6, 12, 17, 19, 20, 21, 29
- [25] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 17
- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 1, 2, 12, 13
- [27] Karl Holmquist and Bastian Wandt. Diffpose: Multihypothesis human pose estimation using diffusion models. In ICCV, 2023. 2, 12
- [28] Tao Jiang, Peng Lu, Li Zhang, Ningsheng Ma, Rui Han, Chengqi Lyu, Yining Li, and Kai Chen. Rtmpose: Realtime multi-person pose estimation based on mmpose. arXiv preprint arXiv:2303.07399, 2023. 7, 8, 20, 21
- [29] Zhongyu Jiang, Zhuoran Zhou, Lei Li, Wenhao Chai, ChengYen Yang, and Jenq-Neng Hwang. Back to optimization: Diffusion-based zero-shot 3d human pose estimation. arXiv preprint arXiv:2307.03833, 2023. 12, 23
- [30] Angjoo Kanazawa, Michael J Black, David W Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, 2018. 12
- [31] Yueying Kao, Bowen Pan, Miao Xu, Jiangjing Lyu, Xiangyu Zhu, Yuanzhang Chang, Xiaobo Li, Zhen Lei, and Zixiong Qin. Single-image 3d face reconstruction under perspective projection. arXiv preprint arXiv:2205.04126, 2022. 5, 15, 17, 20, 21, 22, 32, 33
- [32] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. NeurIPS, 2022. 2
- [33] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. NeurIPS,

2022. 23

- [34] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 12, 18
- [35] Taein Kwon, Bugra Tekin, Jan St¨uhmer, Federica Bogo, and Marc Pollefeys. H2o: Two hands manipulating objects for first person interaction recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10138–10148, 2021. 5, 15
- [36] Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 17
- [37] Lijun Li, Li’an Zhuo, Bang Zhang, Liefeng Bo, and Chen Chen. Diffhand: End-to-end hand mesh reconstruction via diffusion models. arXiv preprint arXiv:2305.13705, 2023. 12
- [38] Tianye Li, Timo Bolkart, Michael J Black, Hao Li, and Javier Romero. Learning a model of facial shape and expression from 4d scans. ACM Trans. Graph., 36(6):194–1, 2017. 4, 15, 16, 17

- [39] Zhihao Li, Jianzhuang Liu, Zhensong Zhang, Songcen Xu, and Youliang Yan. Cliff: Carrying location information in full frames into human pose and shape estimation. In ECCV,

2022. 6, 26

- [40] Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. Character controllers using motion vaes. TOG,

2020. 6, 12

- [41] Haiyang Liu, Zihao Zhu, Giorgio Becherini, Yichen Peng, Mingyang Su, You Zhou, Xuefei Zhe, Naoya Iwamoto, Bo Zheng, and Michael J Black. Emage: Towards unified holistic co-speech gesture generation via expressive masked audio gesture modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1144–1154, 2024. 5, 7, 8, 16
- [42] Qiang Liu and Dilin Wang. Stein variational gradient descent: A general purpose bayesian inference algorithm. NeurIPS, 2016. 24
- [43] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. ACM Transactions on Graphics, 34(6),

2015. 2, 3, 12, 16, 21

- [44] Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, and Heung-Yeung Shum. Humantomato: Text-aligned whole-body motion generation. arXiv preprint arXiv:2310.12978, 2023. 2
- [45] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In ICCV, 2019. 5, 6, 14, 16, 18, 19, 21, 23, 24
- [46] Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. arXiv preprint arXiv:2305.04391,

2023. 2, 3, 4, 8, 14, 23, 24

- [47] Gyeongsik Moon, Hongsuk Choi, and Kyoung Mu Lee. Accurate 3d hand pose estimation for whole-body 3d human mesh estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2308–2317, 2022. 19, 20, 31
- [48] Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe De Bree, et al. A dataset of relighted 3d interacting hands. Advances in Neural Information Processing Systems, 36, 2024. 5, 6, 15, 20, 21
- [49] Lea M¨uller, Vickie Ye, Georgios Pavlakos, Michael Black, and Angjoo Kanazawa. Generative proxemics: A prior for 3d social interaction from images. arXiv preprint arXiv:2306.09337, 2023. 12
- [50] Naoki Murata, Koichi Saito, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Gibbsddrm: A partially collapsed gibbs sampler for solving blind inverse problems with denoising diffusion restoration. arXiv preprint arXiv:2301.12686, 2023. 23
- [51] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In CVPR, 2019. 1, 5, 6, 7, 12, 14, 16, 17, 18, 19, 20, 21, 22, 29

- [52] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 8, 12, 13
- [53] Abhinanda R Punnakkal, Arjun Chandrasekaran, Nikos Athanasiou, Alejandra Quiros-Ramirez, and Michael J Black. Babel: Bodies, action and behavior with english labels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 722–731, 2021. 24
- [54] Zhongwei Qiu, Qiansheng Yang, Jian Wang, Xiyu Wang, Chang Xu, Dongmei Fu, Kun Yao, Junyu Han, Errui Ding, and Jingdong Wang. Learning structure-guided diffusion model for 2d human pose estimation. arXiv preprint arXiv:2306.17074, 2023. 12
- [55] Davis Rempe, Tolga Birdal, Aaron Hertzmann, Jimei Yang, Srinath Sridhar, and Leonidas J Guibas. Humor: 3d human motion model for robust pose estimation. In ICCV, 2021. 6, 12, 21
- [56] Javier Romero, Dimitrios Tzionas, and Michael J Black. Embodied hands: Modeling and capturing hands and bodies together. arXiv preprint arXiv:2201.02610, 2022. 4, 15, 17
- [57] Soubhik Sanyal, Timo Bolkart, Haiwen Feng, and Michael J Black. Learning to regress 3d face shape and expression from an image without 3d supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7763–7772, 2019. 7, 15, 20
- [58] Yonatan Shafir, Guy Tevet, Roy Kapon, and Amit H Bermano. Human motion diffusion as a generative prior. arXiv preprint arXiv:2303.01418, 2023. 2
- [59] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015. 2
- [60] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 2, 4, 5, 6, 12, 21
- [61] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. NeurIPS, 2019. 2, 12
- [62] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 1, 2, 6, 12, 23, 24
- [63] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. NeurIPS, 2021. 3
- [64] Omid Taheri, Nima Ghorbani, Michael J Black, and Dimitrios Tzionas. Grab: A dataset of whole-body human grasping of objects. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 581–600. Springer, 2020. 5, 16
- [65] Garvita Tiwari, Dimitrije Anti´c, Jan Eric Lenssen, Nikolaos Sarafianos, Tony Tung, and Gerard Pons-Moll. Pose-ndf: Modeling human pose manifolds with neural distance fields. In ECCV, 2022. 1, 5, 6, 12, 14, 16, 17, 18, 19
- [66] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 2011. 13

- [67] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In CVPR,

2023. 12, 13

- [68] Jian Wang, Zhe Cao, Diogo Luvizon, Lingjie Liu, Kripasindhu Sarkar, Danhang Tang, Thabo Beeler, and Christian Theobalt. Egocentric whole-body motion capture with fisheyevit and diffusion-based motion refinement. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 777–787, 2024. 12
- [69] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 24
- [70] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. ViTPose: Simple vision transformer baselines for human pose estimation. In Advances in Neural Information Processing Systems, 2022. 24
- [71] Siwei Zhang, Qianli Ma, Yan Zhang, Zhiyin Qian, Taein Kwon, Marc Pollefeys, Federica Bogo, and Siyu Tang. Egobody: Human body shape and motion of interacting people from head-mounted devices. In European conference on computer vision, pages 180–200. Springer, 2022. 5, 16
- [72] Siwei Zhang, Bharat Lal Bhatnagar, Yuanlu Xu, Alexander Winkler, Petr Kadlecek, Siyu Tang, and Federica Bogo. Rohm: Robust human motion reconstruction via diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14606–14617, 2024. 12
- [73] Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation representations in neural networks. In CVPR, 2019. 21
- [74] Joseph Zhu and Peiye Zhuang. Hifa: High-fidelity textto-3d with advanced diffusion guidance. arXiv preprint arXiv:2305.18766, 2023. 13
- [75] Wojciech Zielonka, Timo Bolkart, and Justus Thies. Towards metrical reconstruction of human faces. In European conference on computer vision, pages 250–269. Springer, 2022. 5, 7, 15, 17
- [76] Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox. Freihand: A dataset for markerless capture of hand pose and shape from single rgb images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 813–822, 2019. 5, 14, 19, 21

### Supplementary Material: DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior

This appendix provides comprehensive details to supplement our main work. Section A presents an in-depth discussion of related studies. The parameterization of diffusion models and their connection to score functions are recapped in Section B, followed by the perspective of Score Distillation Sampling (SDS) to understand our DPoser regularization in Section C.

Section D examines the runtime and computational overhead introduced by DPoser, while Section E explores testtime timestep scheduling across both pose and image domains. The datasets used for training and evaluation are detailed in Section F. Section G covers detailed experimental setup including task-specific loss, evaluation metrics used in each task and implementation of comparative methods.

Further evaluations of DPoser-X on additional tasks and datasets are provided in Section H. Section I outlines the training process for DPoser, whereas Section J discusses extended optimization techniques. Section K outlines the method’s limitations, presents failure cases, and suggests avenues for future research. Lastly, Section L showcases additional qualitative results.

##### A. Related Work

###### A.1. Human Pose Priors

Human body models like SMPL [43] and SMPL-X [51] serve as powerful tools for parameterizing both pose and shape, thereby offering a comprehensive framework for describing human gestures. Within the SMPL model, body poses are represented using rotation matrices or joint angles linked to a kinematic skeleton. Adjusting these parameters enables the representation of diverse human actions. Nonetheless, feeding unrealistic poses into these models can result in non-viable human figures, primarily because plausible human poses are confined within a complex, highdimensional manifold due to biomechanical constraints.

Various strategies [1, 12, 24, 51, 65] have been put forward to build human pose priors. Generative frameworks like GMMs, VAEs [34], and Generative Adversarial Networks (GANs) [21] have shown promise in encapsulating the multifaceted pose distribution, facilitating advancements in tasks like human mesh recovery [20, 30]. Further, some studies have delved into conditional pose priors tailored to specific tasks, incorporating extra information such as image features [4, 54], 2D joint coordinates [9], or se-

* Equal contribution. † Corresponding authors.

quences of preceding poses [40, 55]. Our initiative leans towards an unconditional pose prior without relying on additional inputs like images or text, aiming for a versatile application across various pose-related scenarios.

###### A.2. Diffusion Models for Pose-centric Tasks

Diffusion models [26, 60–62] have emerged as powerful tools for capturing intricate data distributions, aligning well with the demands of multi-hypothesis estimation in ambiguous human poses. Notable works include DiffPose [27], which employs a Graph Convolutional Network (GCN) architecture conditioned on 2D pose sequences for 3D pose estimation by learned reverse process (i.e., generation). Similarly, DiffusionPose [54] and GFPose [9] employ the generation-based pipeline but take different approaches in conditioning. Further, ZeDO [29] concentrates on 2D-to3D pose lifting, while Diff-HMR [4] and DiffHand [37] explore estimating SMPL parameters and hand mesh vertices, respectively. EgoWholeBody [68] and RoHM [72] focus on refining noisy motion sequences via diffusion-based generation. BUDDI [49] stands out for using diffusion models to capture the joint distribution of interacting individuals and leveraging SDS loss [52, 67] for optimization during testing phases.

While DPoser shares similar optimization implementation with BUDDI, it sets itself apart by introducing a wider perspective of inverse problems and equipping an innovative timestep scheduling strategy tailored to human poses. Unlike other approaches [9, 27, 29, 54] that primarily focus on 3D location-based representation, DPoser takes on the more demanding task of modeling SMPL-based rotation pose representation. Furthermore, DPoser-X improves whole-body modeling with detailed hand and facial expressions, making it a versatile choice for pose-centric tasks.

##### B. Parameterization of Score-based Diffusion Models

In the seminal work by Song et al. [62], it is demonstrated that both score-based generative models [61] and diffusion probabilistic models [26] can be interpreted as discretized versions of stochastic differential equations (SDEs) defined by score functions. This unification allows the training objective to be interpreted either as learning a time-dependent denoiser or as learning a sequence of score functions that describe increasingly noisy versions of the data.

We begin by revisiting the training objective for scorebased models [61] to elucidate the link with diffusion mod-

els [26]. Consider the transition kernel of the forward diffusion process p0t(xt|x0) = N(xt;αtx0,σt2I). Our goal is to learn score functions ∇xt

log pt(xt) through a neural network sθ(xt;t), by minimizing the L2 loss as follows (we omit the expectation operator for conciseness) :

log pt (xt)||22 . (12) Here, xt = αtx0 + σtϵ, where ϵ ∼ N(0,I).

E w(t)||sθ(xt;t) − ∇xt

Based on denoising score matching [66], we know the minimizing objective Eq. (12) is equivalent to the following tractable term:

log p0t(xt|x0)||22 . (13)

E w(t)||sθ(xt;t) − ∇xt

To link this with the noise predictor ϵθ(xt;t) in diffusion models, we can employ the reparameterization sθ(xt;t) = −ϵ

θ(xt;t)

σt . Then, Eq. (13) can be simplified as follows: w(t)|| −

ϵθ(xt;t) σt − ∇xt

log p0t(xt | x0)||22

(xt − αtx0) σt2 ||22

ϵθ(xt;t) σt

=w(t)|| −

+

ϵθ(xt;t) σt

σtϵ σt2

)||22

=w(t)|| −

+

w(t) σt2 ||ϵθ(xt;t) − ϵ)||22 (14)

=

The resulting form of Eq. (14) aligns precisely with the noise prediction form of diffusion models [26] (refer to Eq. (4) in the main text). This implies that by training ϵθ(xt;t) in a diffusion model context, we simultaneously get a handle on the score function, approximated as ∇xt

θ(xt;t)

log pt(xt) ≈ −ϵ

σt .

##### C. View DPoser as Score Distillation Sampling

Interestingly, the gradient of DPoser (Eq. (10) in the main text) coincides with Score Distillation Sampling (SDS) [52,

- 67], which can be interpreted as aiming to minimize the following KL divergence:

KL p0t (xt | x0) ∥ pSDEt (xt;θ) , (15) where pSDEt (xt;θ) denote the marginal distribution whose score function is estimated by ϵθ(xt;t). For the specific case where t → 0, this term encourages the Dirac distribution δ(x0) (i.e., the optimized variable) to gravitate toward the learned data distribution pSDE0 (x0;θ), while the Gaussian perturbation like Eq. (15) softens the constraint. Building on this understanding, we can borrow advanced techniques from SDS—a rapidly evolving area ripe for methodological innovations [11]. To extend this, we experiment with a multi-step denoising strategy adapted from HiFA [74], substituting our original one-step denoising process. This alternative, however, yields suboptimal results across most evaluation metrics, as demonstrated in Table S-

[Figure 84]

Figure S-1. Visualization of the impact of different timestep values in DPoser regularization. A larger t effectively corrects undesirable poses but may excessively alter well-posed inputs, resulting in plausible yet unrelated poses. Conversely, a smaller t better preserves the original pose but struggles to correct implausible ones.

1. A plausible explanation could be that our proposed truncated timestep scheduling effectively manages low noise levels (i.e., small t), thus negating the need for more denoising steps. In our main experiments, we keep the efficient one-step denoiser.

##### D. Runtime Comparison

Diffusion models generally require iterative steps for gradual denoising, making them less efficient than VAEs and GANs in generation tasks. However, when applied to downstream optimization processes, DPoser introduces minimal additional computational overhead. This is due to two key factors: (1) DPoser regularization involves only a singlestep denoising at each optimization step, and (2) the stopgradient operator ensures that the regularization does not require backpropagation through the trained network.

To assess DPoser’s efficiency, we benchmarked its runtime against various prior models (including a baseline without pose prior) for human mesh recovery across 100 images in a consistent execution environment. As shown in Table S-2, incorporating DPoser results in only a modest (10%) increase in optimization runtime compared to the baseline. In contrast, GAN-S [12] incurs a significant computational cost due to its required GAN-inversion phase, which converts initial poses into their latent representations.

##### E. Analysis of Test-time Timestep Scheduling

During optimization, the selection of timestep is crucial for downstream tasks. As discussed in Section 2.4, the key information of pose data emerges at small t values (t ≤ 0.3), which serves as a coarse range. Moreover, the L2 loss format of our DPoser regularization gives an intuitive view of the impact of timestep. As shown in Fig. S-1, while t is small, since the adding noise and denoising path is short, the denoised pose is close to the origin and the DPoser guidance is weak. Specifically, considering the extreme case where

t → 0, in xˆ0(t) = xt−σtαϵϕ(xt;t)

, the coefficient σt → 0 while αt → 1, causing xˆ0(t) to approach x0, which leads

t

Whole-body Mesh Recovery Body Pose Completion Motion Denoising PA-MPVPE (all) ↓ PA-MPVPE (hands) ↓ MPVPE ↓ APD ↑ MPVPE ↓ MPJPE ↓

Strategy

1 step 60.98 15.60 38.79/78.31/27.13 6.53 38.21 19.87 5 step 61.39 15.70 40.15/85.01/31.96 7.72 40.22 21.21 10 step 61.52 15.74 41.04/87.36/32.51 8.07 40.69 21.34

Table S-1. Ablation of different denoising steps in DPoser’s optimization.

w/o prior GMM [1] VPoser [51] Pose-NDF [65] GAN-S [12] DPoser 15.64 16.14 16.83 21.88 74.60 17.34

- Table S-2. Runtime comparison (in seconds) of different prior models for human mesh recovery on 100 images, evaluated using an RTX 3090Ti GPU.

Noise std [0.15,0.05] [0.2,0.05] [0.2,0.1] [0.25,0.1]

40 mm 19.83 19.87 21.68 22.14 100 mm 36.13 34.15 33.18 33.83

- Table S-3. Ablation of timestep range for motion denoising on the AMASS dataset [45]. MPJPE is reported as the metric.

to a near-zero DPoser loss. On the contrary, suitably large t means strong DPoser guidance and can correct implausible poses better. Thus, we tailor [tmax,tmin] intervals to specific tasks based on their noise scales. To verify this, we conduct ablation of the timestep range on motion denoising. As evidenced in Table S-3, to achieve the best performance, larger t values are required for noisier inputs. Based on the above analyses, we select task-specific timestep intervals [tmax,tmin] as follows: [0.2,0.05] for motion denoising (40 mm noise), [0.15,0.05] for pose completion and inverse kinematics, and [0.12,0.08] for mesh recovery. All the experiments, including body-only, hand-only, face-only, and whole-body, share the same timestep hyperparameters without more tuning.

It is also noteworthy that our truncated timestep scheduling is designed for human poses and does not work well on images. In image domains, the initial timesteps play a crucial role in generating foundational perceptual content. In our study, we employed a 256x256 unconditional diffusion model [14] trained on ImageNet [13] with variational diffusion sampling [46] for image inpainting. This model employs 1000 discrete timesteps during training. We compared standard scheduling (timesteps 990 to 0) with truncated scheduling (timesteps 495 to 0), both using 100 steps. The results, shown in Fig. S-2, indicate that truncation negatively affects image quality. While the standard approach preserved perceptual content, the truncated method produced disjointed patches that were misaligned with the original context. These results affirm that truncated

timestep scheduling excels in pose data where key information emerges in later stages but falls short in image tasks where early timesteps are essential. This scheduling is thus bespoke to the characteristics of human poses and is unsuitable for image processes that rely on the full diffusion timeline for content fidelity.

##### F. Dataset Description

This section provides a detailed overview of the datasets used in our experiments, categorized based on the body part they focus on. We describe each dataset’s specific use case along with the number of samples available for each dataset.

###### F.1. Body-only Dataset

AMASS The AMASS dataset [45] is a large-scale collection of high-quality 3D human body meshes derived from multiple motion capture sources. It provides motion sequences and human poses in a SMPL-based format, covering a broad range of activities such as walking, sitting, dancing, and running. Following the same splits as VPoser [51], and after sampling to de-duplicate the data, we use approximately 55 million body poses in the SMPL-X [51] format to train our DPoser-body model. The test split consists of 54,000 body poses, which are used to evaluate model performance on tasks including body pose completion and motion denoising.

HPS The HPS dataset [22] contains over 300K synchronized RGB images, paired with reference 3D poses and locations, captured from seven people interacting with largescale 3D scenes. The dataset includes motion sequences of various activities such as exercising, reading, eating, lecturing, using a computer, making coffee, and dancing. Following Pose-NDF [65], we use the HPS dataset to evaluate the motion denoising task without training on it. After sampling, we got 350 sequences, each consisting of 60 frames for testing.

###### F.2. Hand-only Dataset

FreiHAND FreiHAND [76] is a large-scale dataset for 3D hand pose estimation, focusing on single-hand poses. It includes 130,240 training samples (4×32560) and 3,960 evaluation samples. Each training hand pose is accompanied by 4 RGB images, providing diverse data for training ro-

[Figure 85]

- (a)
- (b)

[Figure 86]

- Figure S-2. Image inpainting using standard and truncated timestep scheduling. The process evolution is shown over iterations with the middle row depicting the log-magnitude spectrum and the bottom row the phase spectrum. (a) The standard scheduling exhibits cohesive restoration with detail fidelity. (b) The truncated scheduling results in detail-rich patches that are perceptually incongruent with the original image context.

bust models. We use 32,560 hand poses, represented in the MANO format [56], for training the DPoser-hand model. The remaining 3,960 evaluation samples are used to assess hand mesh recovery performance during testing.

DexYCB DexYCB [3] is a dataset for capturing 3D hand poses during hand-object interactions, focusing on singlehand poses. We use only the hand poses for training the DPoser-hand model. The training set includes 407,000 single hand poses.

HO3D HO3D [23] is a dataset that provides 3D annotations for both hand poses and object interactions. Similar to DexYCB, we utilize the hand poses for training the DPoserhand model. The training set contains 83,000 hand poses.

H2O The H2O dataset [35] provides 3D pose annotations for two-hand and object interactions. For the purpose of training DPoser-hand, we use only the right-hand poses. The training set contains 58,000 hand poses.

ReInterHand ReInterHand [48] is a high-quality synthetic dataset designed for 3D hand pose estimation, specifically focusing on interacting hands. It includes annotations for both hands. For training DPoser-hand, we flip the left-hand pose as right-hand to unify the format. The dataset is split into training, validation, and test sets with an 8:1:1 ratio. We use approximately 186,000 hand poses for training, and

23,000 poses for testing. The test set is used to evaluate hand inverse kinematics tasks.

###### F.3. Face-only Dataset

MICA The MICA dataset [75] consists of eight smaller datasets that were unified to represent about 2315 subjects using the FLAME [38] model. It contains only shape geometry. We use the MICA dataset to train the shape component of DPoser-face, focusing on high-quality 3D face shapes.

WCPA WCPA [31] is a large-scale dataset focusing on 3D face reconstruction under perspective projection. It contains 200 subjects and 356,640 training instances, with detailed annotations for facial expressions. We use WCPA to train the expression component of DPoser-face, with 1/10 of the dataset reserved for testing. The test set is used to evaluate face reconstruction, considering both shape and expressions.

NOW NOW [57] is a widely-used benchmark for face reconstruction. It introduces standard evaluation metrics for assessing the accuracy and robustness of 3D face reconstruction methods, especially under variations in viewing angle, lighting, and occlusions. The validation set containing 352 images is employed for our face reconstruction task. We focus on the non-metrical evaluation of face shape, as

the ground truth (GT) only includes shape. As in previous works such as DECA [18], expressions are set to zero in the FLAME [38] model to obtain a neutral face mesh for final evaluation.

###### F.4. Whole-body Dataset

BEAT2 BEAT2 [41] is a holistic co-speech dataset that combines the MoShed SMPL-X [51] body with FLAME head parameters. It refines the modeling of head, neck, and finger movements, providing high-quality 3D motioncaptured data. After de-duplication, we have 1.48 million whole-body poses for training DPoser-X, and the test set contains 172,000 whole-body poses used for whole-body pose completion.

GRAB GRAB [64] is a dataset containing full 3D shape and pose sequences of 10 subjects interacting with 51 everyday objects of varying shapes and sizes. We use this dataset for training DPoser-X, with 391,000 whole-body poses. The data helps train the model for whole-body mesh recovery during grasping actions.

ARCTIC ARCTIC [16] is a dataset focused on two-hand object manipulation, with 2.1 million video frames paired with 3D hand and object meshes. After de-duplication, we have 77,000 whole-body poses for training DPoser-X. The validation set, which contains 10,000 whole-body poses, is used for testing whole-body mesh recovery and pose completion. Note that the face expressions are not annotated, so we set models’ output expressions as zeros for evaluation. Face-related metrics in whole-body mesh recovery are only influenced by the human shape.

EgoBody EgoBody [71] is a large-scale dataset that captures 3D human motions during social interactions in 3D scenes. It provides SMPL-X [51] annotations for 3D wholebody pose, shape, and motion for both the interactee and the camera wearer. The training set contains 38,896 instances of whole-body poses (x2 for each subject), and the test set contains 24,665 instances. The test set is used for the whole-body pose completion task.

Fit3D Fit3D [19] is a dataset with over 3 million images and corresponding 3D human shape and motion capture ground truth data, covering 37 exercises performed by instructors and trainees. We take the subject s04 which consists of 612 images after sampling, for whole-body mesh recovery tasks to test DPoser-X’s generalization, without training the model on this dataset.

EHF EHF [51] is a curated dataset comprising 100 images with pseudo whole-body poses. Following Pose-NDF [65], we use this dataset to evaluate body mesh recovery performance, specifically calculating PA-MPJPE for body joints.

##### G. Experimental Details

In this section, we provide detailed descriptions of the experimental setups and specific loss functions for various

tasks. These tasks include pose completion, motion denoising, inverse kinematics, face reconstruction, hand mesh recovery, and whole-body mesh recovery. In addition, we explain the evaluation metrics used in each task and implementation details of comparative methods.

###### G.1. Pose Completion

For partial observations y, the measurement operator A is modeled as a known mask matrix M ∈ Rd×n. Based on our optimization framework denoted in Alg. 1, we define the task-specific loss, Lcomp, as follows:

Lcomp = ||Mx0 − y||22. (16)

Here, x0 denotes the complete body pose θ we try to recover, where the unseen parts are initialized as random noise. In the following ablated studies, if not specified, the evaluation of the body pose completion is performed using 10 hypotheses on the AMASS dataset [45] with left leg occlusion.

###### G.2. Motion Denoising (Noisy Input) Adhering to Pose-NDF settings [65], we aim to refine noisy

joint positions Jobst over N frames to obtain clean poses θt, initialized from mean poses in SMPL with small noise. We formulate the task-specific loss combining an observation fidelity term Lobs and a temporal consistency term Ltemp:

Lobs =

N−1

||MJ(θt,β0) − Jobst ||22, (17)

t=0

N−1

||MJ(θt−1,β0) − MJ(θt,β0)||22, (18)

Ltemp =

t=1

where MJ denotes the 3D joint positions regressed from SMPL [43] and β0 is the constant mean shape parameters.

###### G.3. Motion Denoising (Partial Input) This task focuses on reconstructing clean poses, θt, from

partially observed joint positions, Jobst , across N frames, employing a known mask matrix to identify visible joints. The optimization objective mirrors that of motion denoising (Section G.2), but incorporates a mask in Eq. (17) to specifically target visible parts, ensuring that only these segments guide the recovery process.

###### G.4. Inverse Kinematics

Inverse kinematics (IK) aims to estimate clean poses from noisy or partially observed 3D joint positions, similar to the motion denoising task. The key difference in the implementation is that the inputs are single-frame data, meaning the temporal consistency term Ltemp is not required.

For inverse kinematics applied to hand poses, we optimize only the hand poses while keeping the hand shape parameters fixed. This simplifies the optimization, focusing

solely on pose adjustments. For the face, we optimize both the face expression and shape parameters, as face-related tasks require accurate modeling of both shape and dynamic expressions. In all cases, we employ a similar optimization framework as in the motion denoising task, using only the fidelity loss for observed 3D joints.

###### G.5. Face Reconstruction

Reconstructing human faces using only 2D keypoints is challenging and typically insufficient for high-quality reconstructions. To address this, we utilize the photometric optimization approach described in [17] to fit a textured FLAME model [38]. The optimization aims to refine the face shape and expression parameters, as well as adjust the appearance and lighting parameters for the face rendering. We use a combination of two key loss functions: the photometric loss (L1-loss between rendered and target images) and the reprojection loss (for 2D face keypoints).

We observe that face shape plays a crucial role in tasks like face reconstruction. To this end, DPoser-face is designed to separately model face shape and expression. Given that these two components (shape and expression) are largely independent, we train the face shape and expression models separately using the WCPA [31] and MICA [75] datasets, respectively. For face-only tasks such as face reconstruction, DPoser regularization is applied to both the face shape and expression models. It is important to note that only the expression component of DPoser-face contributes to the broader DPoser-X framework, with the shape component being reserved for face-specific tasks. For a fair comparison, we implement the same strategy for training the VPoser-face model.

###### G.6. Hand Mesh Recovery

For hand mesh recovery, we optimize the hand poses using the MANO model [56] instead of the SMPL model. Similar to the body mesh recovery task, we employ a reprojection loss based on 2D hand keypoints. In addition to using our DPoser loss for plausible hand poses, we also employ the L2 prior for hand shape, similar to Eq. (11) in the main text, to maintain natural hand geometry.

###### G.7. Whole-body Mesh Recovery

Whole-body mesh recovery shares similarities with body mesh recovery (as discussed in Section 2.5) but additionally incorporates the face and hands into the optimization. The goal is to recover the whole-body poses θ (including body, hands, and face) and shape parameters β by optimizing a reprojection loss based on whole-body 2D keypoints. A distinguishing feature is the inclusion of two root-relative reprojection losses, one for the hands and another for the face, to refine local poses. Specifically, the wrists for hands and the mouth for the face are chosen as the root, and the

root coordinates are subtracted before calculating the reprojection losses. This ensures that the hand and face poses are localized relative to the body, improving the accuracy of hand and facial mesh recovery.

###### G.8. Evaluation Metrics

For comprehensive assessment across various tasks, following recent works like NRDF [24] and SMPLer-X [2], we adopt task-specific metrics:

- • Pose Generation: Diversity and fidelity are evaluated using Average Pairwise Distance (APD) and dNN [24], respectively. dNN measures the distance between the generated pose and its nearest neighbor from the training data. We also report the common metrics for generative models, including FID [25] (distribution similarity), Precision [36] (fidelity), and Recall [36] (diversity).
- • Human Mesh Recovery: Procrustes-aligned Mean PerVertex Position Error (PA-MPJPE) and Procrustesaligned Mean Per-Joint Position Error (PA-MPVPE) measures the accuracy of recovered human meshes.
- • Multi-hypothesis Pose Completion: MPVPE and APD on masked parts across multiple hypotheses measure solution accuracy and diversity, respectively.
- • Motion Denoising & Inverse Kinematics: Both MPJPE and MPVPE are calculated to assess the performance.

All errors are reported in millimeter units.

###### G.9. Implementation of Comparative Methods

In pose generation experiments, we employ standard sampling techniques for generative models, including GMM [1], VPoser (VAE) [51], and GAN-S (GAN) [12]. For Pose-NDF [65] and NRDF [24], we reproduce their projection algorithms using their official repositories. For other tasks during testing, to ensure a fair comparison, we implement all pose priors within the same optimization framework—using identical task-specific loss functions and optimization iterations—while tuning hyperparameters such as loss weights for each method.

VPoser and GAN-S function as pose priors due to their learned meaningful latent representations. We optimize the pose latents for both methods. Given VPoser’s Gaussian assumption, it naturally incorporates L2 regularization on the latent pose [51]. However, we observe that applying spherical loss to the latents of GAN-S [12] degrades human mesh recovery performance. Therefore, we use only GAN-S’s generator for decoding without imposing additional constraints on the pose latents. Both NDF [65] and NRDF [24] directly optimize pose rotation representations by minimizing the predicted distance between the current pose and their learned plausible pose fields. We implement these methods using their official code and model weights. Since GAN-S does not provide pre-trained models, we train it from scratch on the same datasets as our DPoser. Addi-

Methods hypotheses num=1 hypotheses num=10

GFPose [9] 68.64/89.88 62.80/83.39 HuProSO3 [15] 72.00/104.52 57.42/84.21 DPoser (ours) 56.05/79.82 53.28/76.53

- Table S-4. Comparison with generation-based methods on the HMR task using the EHF dataset [51]. We report the minimum PA-MPJPE/MPJPE across multiple hypotheses.

tionally, for hand, face, and whole-body models, we train the comparative methods ourselves.

##### H. Additional Experiments

In this section, we present a series of additional experiments that further demonstrate the efficacy of DPoser-X across various tasks. These experiments cover body mesh recovery, body pose completion, motion denoising, hand/face generation, hand/face inverse kinematics, face reconstruction, and whole-body mesh recovery, with a focus on different input types and datasets.

###### H.1. Body Mesh recovery

In addition to the priors compared in the main text, we evaluate DPoser against two recent state-of-the-art, generationbased methods: GFPose [9] and HuProSO3 [15]. Unlike optimization-based priors, these methods are designed to produce multiple, diverse hypotheses for a given input.

We report the results for the Human Mesh Recovery (HMR) task on the EHF dataset [51] in Table S4. The comparison is conducted with both a single hypothesis (hypotheses num=1) and multiple hypotheses (hypotheses num=10), reporting the minimum PA-MPJPE and MPJPE. The results clearly show that while GFPose and HuProSO3 can generate diverse potential poses, our DPoser achieves significantly higher accuracy (i.e., lower error) in both evaluation settings. This suggests that DPoser provides a more precise and reliable pose prior for this task.

###### H.2. Body Pose Completion

In practical scenarios, HMR algorithms often grapple with occlusions leading to incomplete 3D pose estimates. In this context, the task is to recover full 3D poses from partially observed data, initializing the occluded parts with noise. Our DPoser model is employed to refine these initially implausible poses into feasible ones, utilizing an L2 loss on the visible parts to ensure data consistency. In parallel, we employ a comparable optimization strategy for both PoseNDF [65] and VPoser [51]. As a task-specific baseline, we adapt the original VPoser model into CVPoser by incorporating conditional inputs within its VAE framework [34] for end-to-end training and conditional sampling. The completion experiment is conducted on the AMASS dataset [45] with occlusion of various body parts.

Given the uncertainties in this task, we generate multiple hypotheses and evaluate them using minimum, mean, and standard deviation errors against the ground truth. We calculate APD across solutions to assess diversity. As illustrated in Table S-5, DPoser exhibits superior performance across different occlusion scenarios compared to existing pose priors and even the task-specific CVPoser, highlighting its effectiveness in pose completion. The qualitative evaluations are presented in Fig. S-3. Here, we observe that DPoser can generate a multitude of plausible poses, a capability lacking in VPoser [51]. Pose-NDF [65], meanwhile, struggles with generalizing to unseen noisy poses and making plausible adjustments from the mean pose initialization.

###### H.3. Motion Denoising (Noisy Input)

To further evaluate DPoser’s performance in motion denoising, we extend our analysis to scenarios with varying noise levels. In complement to the results presented in Table 3 of our main text, we conduct an in-depth examination that spans a broader range of noise conditions. The extended results, detailed in Table S-6, showcase DPoser’s exceptional performance against state-of-the-art (SOTA) pose priors, especially under high noise conditions, manifesting DPoser’s resilience to noise.

###### H.4. Motion Denoising (Partial Input)

We next assess the performance of our model in scenarios involving partial input using the AMASS dataset [45]. Two types of occlusions were considered: legs and left arm. The quantitative results of these experiments are presented in Table S-7, while visual examples can be found in Section L. Errors (in cm) are evaluated in terms of MPJPE across visible (Vis.), occluded (Occ.), and all joints, along with MPVPE for all vertices.

In the leg occlusion scenario, where the AMASS dataset primarily consists of straight poses, the lack of diversity allows for reasonable results even without incorporating a pose prior. In this case, the optimization starts from an initial point that closely matches these common poses. However, while VPoser’s mean-centered approach struggles to faithfully replicate visible areas, DPoser accurately handles the visible portions and guides the reconstruction of occluded parts, yielding more realistic results. In contrast, Pose-NDF does not effectively enhance the occluded regions. For left arm occlusions, which involve more varied movements, DPoser markedly surpasses other methods, underlining its adaptability and precision in handling diverse motion patterns.

###### H.5. Hand Pose Generation

We evaluate the generated hand poses based on their diversity and realism. As shown in Table S-8, DPoser produces a strong combination of both, outperforming meth-

Occ. left leg Occ. legs Occ. arms Occ. torso MPVPE ↓ APD ↑ MPVPE ↓ APD ↑ MPVPE ↓ APD ↑ MPVPE ↓ APD ↑

Methods

Pose-NDF [65] (S = 1) 168.61 NAN 169.92 NAN 261.11 NAN 115.03 NAN Pose-NDF (S = 5) 157.62/168.49/7.94 1.95 162.30/169.94/5.54 1.96 254.97/261.01/4.38 1.22 108.07/114.98/4.98 0.93 Pose-NDF (S = 10) 154.21/168.45/8.66 1.95 159.75/169.86/6.12 1.97 252.90/260.94/4.81 1.20 105.87/114.97/5.43 0.93 VPoser [51] (S = 1) 200.23 NAN 221.21 NAN 206.83 NAN 58.66 NAN VPoser (S = 5) 187.38/200.73/10.52 2.38 201.70/221.16/14.57 5.49 191.27/206.55/11.54 4.06 49.88/58.67/6.71 1.59 VPoser (S = 10) 182.31/200.51/12.20 2.41 195.76/221.34/16.40 5.44 186.55/206.72/12.91 4.08 47.31/58.71/7.38 1.56 CVPoser† (S = 10) 113.48/128.04/10.36 1.91 121.00/134.35/10.17 2.43 153.12/162.82/5.58 1.08 45.16/51.23/4.32 0.57

DPoser (S = 1) 78.78 NAN 103.12 NAN 104.59 NAN 44.60 NAN DPoser (S = 5) 46.23/78.13/24.96 6.58 72.37/102.73/23.05 7.72 74.32/105.70/24.15 5.67 27.47/44.63/13.26 2.19 DPoser (S = 10) 38.79/78.31/27.13 6.53 63.65/102.46/25.39 7.75 64.72/104.94/26.44 5.69 22.63/44.60/14.65 2.21

- Table S-5. Performance metrics (min/mean/std of MPVPE and APD) for body pose completion on the AMASS dataset [45] under varying occlusion scenarios. S denotes the number of hypotheses. † Task-specific baseline trained with partial poses as conditional input.

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

VPoser Pose-NDF DPoser (ours) GT

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

- Figure S-3. Visual comparisons of body pose completion. Three hypotheses are drawn for each method. DPoser uniquely offers multiple plausible solutions for partial poses, a scenario where competitors often struggle due to limited generalization.

Methods

AMASS [45] HPS [22] 20mm 100mm 20mm 100mm

w/o prior 15.33 51.48 16.26 50.87 VPoser [51] 15.20 49.10 17.24 46.69 Pose-NDF [65] 13.84 46.10 15.62 47.50 DPoser 13.64 33.18 13.45 35.32

- Table S-6. Performance comparison of motion denoising under varying noise scales. MPJPE is reported afters denoising.

MPJPE MPVPE Vis. Occ. All All

Methods Occlusion

w/o prior legs 0.26 14.72 5.52 5.45 VPoser [51] legs 1.75 14.29 6.31 7.38 Pose-NDF [65] legs 0.25 15.71 5.87 5.64 DPoser legs 0.28 12.24 4.63 3.65

w/o prior left arm 0.26 24.87 4.74 9.91 VPoser [51] left arm 1.21 13.23 3.40 7.68 Pose-NDF [65] left arm 0.25 17.70 3.42 7.86 DPoser left arm 0.27 7.80 1.64 3.81

ods like VPoser [51] and NRDF [24]. Specifically, NRDF shows poor realism, reflected in high FID and dNN scores. VPoser, while achieving moderate precision, suffers from limited diversity, as indicated by its low APD. See Fig. S12 visualization comparison.

###### H.6. Hand Mesh Recovery

To evaluate DPoser’s ability to recover hand meshes, we test its performance on the FreiHAND dataset [76] un-

Table S-7. Comparative analysis of methods for motion denoising with different occlusions (legs or left arm) on the AMASS dataset [45].

der two initialization strategies: mean poses and the Hand4Whole [47] prediction poses. The results, detailed in Table S-9 and visually represented in Section L (Fig. S-15 and Fig. S-14), show DPoser’s superior performance across various metrics and initialization settings.

Methods APD ↑ FID ↓ Prec. ↑ Rec. ↑ dNN ↓

VPoser [51] 1.99 0.21 0.68 0.65 1.85 NRDF [24] 1.76 5.20 0.17 0.65 5.37 DPoser-hand 2.36 0.01 0.82 0.87 1.45

Table S-8. Quantitative evaluation of hand pose generation.

DPoser consistently outperforms competing methods, such as VPoser [51] and NRDF [24], achieving the lowest PA-MPJPE and PA-MPVPE values. For example, when using keypoints detected by RTMPose [28], DPoser reduces PA-MPJPE by 20% compared to VPoser. Moreover, the performance is further enhanced when using Hand4Whole [47] initialization, highlighting DPoser’s ability to refine results from existing SOTA mesh recovery models. In contrast, methods like the L2 prior and VPoser, which rely on mean-centered priors, fail to match the quality of the initializations, producing poorer results. Additionally, DPoser demonstrates significant advantages over NRDF in modeling hand pose distributions, offering more reliable guidance in mesh recovery. By leveraging ground truth (GT) keypoints, DPoser consistently recovers natural hand meshes that align well with observed 2D keypoints.

###### H.7. Hand Inverse Kinematics (Noisy Input)

For hand inverse kinematics, we extend our experiments to noisy settings using the ReInterHand dataset [48]. Table S10 shows that DPoser consistently outperforms alternative methods across different noise levels (2mm, 5mm, 10mm), achieving the lowest MPVPE and MPJPE. While methods like the L2 prior and VPoser [51] perform competitively at lower noise levels, their accuracy deteriorates significantly as noise increases. In contrast, DPoser maintains both stability and precision, showcasing its superior ability to handle noisy input and recover plausible hand poses even under challenging conditions.

###### H.8. Face Generation

We conduct the face generation experiments for the shape and expression separately since they are uncorrelated attrbutes. As detailed in Table S-11, DPoser outperforms VPoser [51] in terms of FID, achieving values of 5.331 for shape and 0.156 for expression, which highlights DPoser’s superior ability to model the distribution of face shapes and expressions. While VPoser achieves higher precision scores, its recall values are considerably lower, indicating a lack of variability in the generated samples. This observation is further corroborated by qualitative results shown in Fig. S-4, which demonstrate DPoser’s ability to generate a wide variety of realistic face shapes and expressions. Compared to VPoser, DPoser captures a broader range of subtle variations, especially in expressions, while maintaining fidelity.

###### H.9. Face Reconstruction

For face reconstruction, along with the NOW beachmark [57], we test on the WCPA [31] dataset, which evaluates both face shape and expression. As shown in Table S-12, DPoser consistently outperforms other methods. It achieves the lowest PA-MPVPE and PA-MPJPE errors across all configurations, with a notable reduction in errors for both overall and side-view cases. When combined with EMOCA [10] initialization, DPoser further refines the reconstruction quality, reducing the mean PA-MPVPE error to 3.10 mmm compared to 3.58 mm for EMOCA alone.

Qualitative visualizations in Fig. S-17 illustrate DPoser’s ability to reconstruct detailed and realistic face meshes, even in challenging scenarios involving variations in sideview poses and complex expressions. While other methods often struggle to generalize across such cases, DPoser remains robust and highly accurate, demonstrating its capability to handle the full diversity of facial shapes and expressions in real-world conditions.

###### H.10. Face Inverse Kinematics

To evaluate DPoser’s robustness in face inverse kinematics, we conduct experiments under various noise levels and occlusion scenarios using the WCPA dataset [31]. The results in Table S-13 demonstrate that DPoser consistently achieves the lowest MPVPE and MPJPE errors across all tested conditions. Notably, DPoser retains its strong performance even under extreme noise conditions, whereas VPoser [51] experiences significant degradation as noise levels increase. Qualitative results, visualized in Fig. S-16, further confirm DPoser’s ability to reconstruct realistic and aligned facial details under noisy and occluded conditions.

###### H.11. Whole-body Mesh Recovery

We extend our evaluation of whole-body mesh recovery to include the Fit3D dataset [19], in addition to the comparative results on ARCTIC [16]. For this evaluation, we compare DPoser-X with VPoser-X [51] and the GMM baseline, which utilizes a Gaussian Mixture Model (GMM) [1] for body poses and an L2 prior for hands and face.

As shown in Table S-14, DPoser-X outperforms both VPoser-X [51] and GMM [1] across most metrics, for both hands and the entire body. However, we observe that the L2-prior baseline performs better than DPoser-X in terms of PA-MPVPE on the face. We attribute this result to the low-resolution images in the Fit3D dataset, where the face is depicted with limited pixel density and the 2D keypoints are less expressive. In this case, the neutral face produced by the L2 prior is more likely to yield better results due to the lack of detailed facial features in the input. Nonetheless, DPoser-X still outperforms other methods in handling the full-body mesh recovery, showing its robustness in both body and hand mesh reconstruction.

Methods PA-MPJPE ↓ PA-MPVPE ↓ F@5 ↑ F@15 ↑

w/o prior 17.71/16.12 18.40/17.04 0.396/0.446 0.875/0.895 L2 prior 12.87/11.49 12.71/11.59 0.512/0.533 0.924/0.927 VPoser [51] 12.31/10.62 12.23/10.91 0.524/0.609 0.931/0.943 NRDF [24] 13.19/11.04 13.39/11.59 0.469/0.554 0.914/0.937 DPoser-hand 10.71/8.68 10.48/8.70 0.574/0.679 0.947/0.963

hand4whole 8.50 7.81 0.651 0.97

+ w/o prior 9.13/6.04 8.97/6.06 0.609/0.749 0.965/0.985 + L2 prior 9.91/7.16 9.69/7.11 0.568/0.686 0.953/0.974 + VPoser [51] 9.13/6.42 9.04/6.55 0.605/0.717 0.964/0.981 + NRDF [24] 9.00/6.15 8.99/6.29 0.595/0.726 0.964/0.983 + DPoser-hand 7.96/5.36 7.69/5.20 0.663/0.793 0.973/0.990

Table S-9. Performance evaluation of hand mesh recovery on the FreiHAND dataset [76]. Results are reported using 2D keypoints detected by RTMPose [28] / ground truth.

2mm 5mm 10mm MPVPE ↓ MPJPE ↓ MPVPE ↓ MPJPE ↓ MPVPE ↓ MPJPE ↓

Methods

No prior 3.95 1.50 5.82 3.46 8.62 5.97 L2 prior 2.10 1.43 4.06 2.92 6.06 4.27 VPoser [51] 2.47 1.36 4.15 2.85 6.32 4.40 NRDF [24] 2.67 1.40 4.57 3.11 7.18 5.06 DPoser-hand 1.71 1.17 3.30 2.39 5.39 3.87

Table S-10. Performance of hand inverse kinematics on the ReInterHand dataset [48] under noisy settings.

Methods FID ↓ Prec. ↑ Rec. ↑ dNN ↓

VPoser [51] (shape) 31.91 0.984 0.105 6.52 DPoser-face (shape) 5.331 0.689 0.396 8.29

VPoser [51] (expression) 0.888 0.993 0.019 0.79 DPoser-face (expression) 0.156 0.818 0.697 1.01

Table S-11. Quantitative evaluation for face generation.

Methods all side-view w/o prior 3.67/4.19 3.77/4.46 L2 prior 3.56/3.90 3.58/4.01 VPoser [51] 3.59/4.01 3.62/4.13 DPoser-face 3.34/3.65 3.32/3.61 EMOCA [10] 3.58/4.07 3.78/4.43 + w/o prior 3.49/3.88 3.92/4.56 + L2 prior 3.49/3.82 3.68/4.28 + VPoser [51] 3.39/3.65 3.56/4.05 + DPoser-face 3.10/3.54 3.16/3.72

Table S-12. Face reconstruction performance (PA-MPVPE/PAMPJPE) on the WCPA dataset [31].

##### I. Ablated DPoser’s Training

This section dissects the impact of different rotation representations and normalization techniques on DPoser’s performance. The ablation of training experiments is conducted for the DPoser-body model trained on AMASS [45]. Initially, we examine axis-angle representation, comparing various normalization strategies: min-max scaling, z-score normalization, and no normalization. Our findings, summarized in Table S-15, indicate that z-score normalization is generally the most effective. Subsequently, using this optimal normalization, we explore 6D rotations [73] as an alternative. As evidenced by Table S-16, axis-angle representation offers superior performance. This preference can be attributed to the effective modeling capabilities of diffusion models, which do not benefit much from a more continuous data representation.

Inspired by HuMoR [55], we experiment with integrating the SMPL body model [43] as a regularization term during training. Alongside the prediction of additive noise, as outlined in Eq. (4) in the main text, we employ a 10-step DDIM sampler [60] to recover a “clean” version of the pose, denoted as x˜0, from the diffused xt. The regularization loss aims to minimize the discrepancy between the original and

[Figure 107]

(a) VPoser

[Figure 108]

(b) DPoser (ours)

Figure S-4. Visualization of face generation results. Top row shows varying face shapes; bottom row shows varying expressions.

1mm Noise 2mm Noise 5mm Noise Half Face Occ. MPVPE ↓ MPJPE ↓ MPVPE ↓ MPJPE ↓ MPVPE ↓ MPJPE ↓ MPVPE ↓ MPJPE ↓

Methods

w/o prior 1.460 0.878 2.230 1.702 4.701 4.028 0.752 0.632 L2 prior 1.121 0.865 1.626 1.288 2.570 2.344 0.698 0.512 VPoser [51] 1.153 0.803 1.688 1.480 2.716 2.688 0.671 0.361 DPoser-face 0.784 0.584 1.098 0.963 1.902 1.936 0.427 0.228

Table S-13. Performance of face inverse kinematics on the WCPA dataset [31] under noisy and occlusion settings.

PA-MPVPE↓ PA-MPJPE↓ All Hands Face Body

Methods

w/o prior 89.72 23.51 7.26 91.18 GMM [1] & L2 prior 86.95 18.22 5.38 83.58 VPoser-X [51] 81.96 17.59 6.37 86.50 DPoser-X 70.91 15.83 5.27 74.33

SMPLerX 25.49 18.89 2.85 28.30 + w/o prior 24.72 11.92 2.78 22.98 + GMM [1] & L2 prior 24.28 11.09 2.58 22.95 + VPoser-X [51] 24.41 10.21 2.65 23.03 + DPoser-X 23.20 8.91 2.62 21.22

Table S-14. Whole-body mesh recovery results on the Fit3d dataset [19].

recovered poses under the SMPL body model M:

###### Lreg = ||MJ(x˜0,β0) − MJ(x0,β0)||22

+ ||MV (x˜0,β0) − MV (x0,β0)||22. (19) Here, β0 represents the mean shape parameters in SMPL.

To account for denoising errors, we scale the regularization loss by log(1 + α

σt ), thereby increasing the weight for samples with smaller t values (less noise).

t

Fig. S-5 visualizes the impact of this regularization on MPJPE during the training, specifically for pose completion tasks with occlusion of both legs. We observe that weighted regularization offers slight performance gains in the early training process, while the absence of weighting introduces instability and deterioration in results. Despite these insights, the computational cost of incorporating the SMPL model—especially for our large batch size of 1280—makes the training approximately 8 times slower. Therefore, we opted not to include this regularization in our main experiments.

We ablate the architectural hyperparameters of DPoserbody and the number of optimization steps on the HMR task, with results shown in Table S-17. Our findings indicate that a more complex architecture (i.e., 4 blocks or a 2048 hidden dimension) does not improve accuracy. Regarding the optimization, increasing the steps to 1000 offers the best accuracy (55.74 PA-MPJPE) but at a high compu-

Body Mesh Recovery Body Pose Completion Motion Denoising PA-MPJPE ↓ MPJPE (S = 10) ↓ MPVPE ↓ MPJPE ↓

Normalization

w/o norm 57.88 45.37/102.28/41.08 44.82 24.04 min-max 59.17 47.41/107.00/43.42 42.70 21.29 z-score 56.49 34.37/72.47/26.32 38.57 20.24

Table S-15. Comparative performance of normalization methods using axis-angle rotation representation across multiple tasks.

Body Mesh Recovery Body Pose Completion Motion Denoising PA-MPJPE ↓ MPJPE (S = 10) ↓ MPVPE ↓ MPJPE ↓

Representation

axis-angle 56.05 34.76/72.41/26.09 38.21 19.87 6D rotations 57.54 40.89/81.43/27.31 38.44 20.12

Table S-16. Comparative performance of rotation representations using z-score normalization across multiple tasks.

- Figure S-5. MPJPE evolution in DPoser training with different regularization loss settings for body pose completion, assessed on AMASS [45] with 10 hypotheses under legs occlusion scenarios.

Steps Blocks Hidden Dim HMR (PA-MPJPE) Runtime (s) 500 2 1024 56.05 17.34 250 2 1024 56.53 8.44 1000 2 1024 55.74 34.11 500 4 1024 56.47 18.72 500 2 2048 56.67 19.12

- Table S-17. Ablation of DPoser’s architecture and optimization steps for the HMR task.

tational cost (34.11s), while 250 steps are fastest but less accurate. Based on this analysis, we adopt the configuration of 500 steps, 2 blocks, and a 1024 hidden dimension for our experiments, as it provides a solid trade-off between accuracy and runtime efficiency.

##### J. Extended DPoser’s Optimization

In addressing pose-centric tasks as inverse problems, we propose a versatile optimization framework, which employs variational diffusion sampling as its foundational approach [46]. Our exploration extends to an array of diffusion-based methodologies for solving these complex inverse problems. Among the techniques considered are ScoreSDE [62], MCG [7], and DPS [6]. These methods augment standard generative processes with observational data, either by employing gradient-based guidance or back-projection techniques. We compare these methods with our DPoser for body pose completion tasks. Our findings, captured in Table S-18, reveal that DPoser outperforms the competitors under most occlusion conditions. Consequently, DPoser emerges not merely as a universally applicable solution to pose-related tasks, but also as an exceptionally efficient one.

It is worth mentioning that methods rooted in generative frameworks [6, 7, 33, 62] can pose challenges for broader applicability in pose-centric tasks. For instance, in blind inverse problems—certain parameters in A (e.g., camera models in HMR) are unknown—generative methods are less straightforward to implement. ZeDO [29], a recent study focusing on the 2D-3D lifting task, adopts the ScoreSDE [62] framework and refines camera translations by solving an optimization sub-problem after each generative step. However, directly porting this strategy to HMR is non-trivial, owing to the added complexity of body shape parameter optimization—a feature currently absent in our DPoser model. Although some state-of-the-art techniques [8, 50] offer solutions by jointly modeling operator A and data distributions, a full-fledged discussion on this subject is beyond this paper’s purview and remains an open question for future work.

Methods Occ. left leg Occ. legs Occ. arms Occ. torso

ScoreSDE [62] 48.73/106.32/41.30 74.68/128.32/37.27 66.89/127.86/48.15 16.69/34.54/12.21 DPS [6] 40.51/104.32/54.57 64.26/113.46/33.71 60.63/119.85/42.78 15.10/33.90/13.27 MCG [7] 49.04/106.37/41.07 74.90/128.53/37.40 66.17/127.72/48.15 16.69/34.66/12.23 DPoser 35.37/74.01/26.47 59.25/96.77/24.55 51.27/81.76/20.04 13.95/28.57/9.85

- Table S-18. Comparative evaluation of diffusion-based solvers for body pose completion on the AMASS dataset [45]. The min/mean/std of MPJPE are reported (hypotheses number S = 10).

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Images Keypoints DPoser-X

- Figure S-6. Failure cases of our method on challenging yoga poses. Inaccuracies in the estimated 2D keypoints (middle column), combined with our model’s limited exposure to such outof-distribution poses during training, lead to flawed 3D mesh reconstructions (right column).

##### K. Limitation and future work

A primary limitation of our work is the dependency on the training data’s distribution. Our body pose prior is trained on the AMASS dataset [45], which, while diverse in common daily actions, contains limited examples of challenging or extreme poses like those found in yoga. This data imbalance leads to two main issues. First, the learned prior is inherently biased towards common standing poses. Second, when confronted with out-of-distribution inputs, as illustrated in Fig. S-6, the prior may offer limited or even incorrect guidance. This problem is often exacerbated by the failure of off-the-shelf 2D keypoint detectors like ViTPose [70] to produce accurate keypoints for such complex images, which in turn misguides the optimization.

Future work could address these data-driven limitations in several ways. To mitigate the action imbalance, techniques like clustering motions with action labels [53] and performing importance sampling during training could be effective. To improve robustness on challenging poses, incorporating more diverse training data and exploring more robust fitting strategies, such as using predicted dense depth maps for supervision, are promising directions.

Our framework also inherits certain limitations from the

variational diffusion sampling [46] process it employs, most notably a tendency towards mode-seeking. For example, minimizing the DPoser regularization loss alone for “generation” results in a high Precision of 0.995 but a low Recall of 0.163. The low recall, compared to standard generative diffusion samplers (see Table 1 in the main text), indicates that the optimization framework captures the primary modes of the data distribution accurately but lacks diversity. To address this, future research could explore techniques like particle-based variational inference [42, 69] to enhance solution diversity. Finally, within the broader context of inverse problems we have framed, a plethora of existing methods [11] could be adapted to leverage our diffusion-based pose prior. Exploring these methods holds great potential for future progress.

##### L. More Qualitative Results

We show more qualitative results for body pose generation (Fig. S-7), body pose completion (Fig. S-8), body mesh recovery (Fig. S-9), motion denoising (Fig. S-10 and Fig. S11), hand generation (Fig S-12), hand inverse kinematics (Fig S-13), hand mesh recovery (Fig S-14 and Fig S15), face inverse kinematics (Fig S-16), face reconstruction (Fig S-17), whole-body pose generation (Fig S-18 and Fig. S-19), whole-body mesh recovery (Fig S-20), wholebody pose completion (Fig S-21 and Fig. S-22).

[Figure 115]

Figure S-7. Visualization of body pose generation. DPoser can generate diverse and realistic body poses.

- (a)

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

VPoser Pose-NDF DPoser (ours) GT

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

- (b)

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

VPoser Pose-NDF DPoser (ours) GT

Figure S-8. Visualization of body pose completion. (a) Left leg under occlusion. (b) Torso under occlusion.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

- (a)

[Figure 161]

GAN-S

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

- (b)

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Images GMM Pose-NDF VPoser GAN-S DPoser (ours) CLIFF

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

- (c)

Images GMM Pose-NDF VPoser DPoser (ours) GT

Figure S-9. Visualization of body mesh recovery. (a) Fitting from scratch. (b) Initialization using the CLIFF [39] prediction results. (c) More results of DPoser optimization with CLIFF initialization on in-the-wild images.

[Figure 192]

Input

[Figure 193]

VPoser

[Figure 194]

- (a)

(a) Gaussian noise with 40 mm standard deviation.

GT

DPoser (Ours)

Pose-NDF

Input

Motion sequence

VPoser

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

- (b)

Pose-NDF

[Figure 200]

DPoser (Ours)

[Figure 201]

GT

Motion sequence

(b) Gaussian noise with 100 mm standard deviation.

Figure S-10. Visualization of motion denoising with noisy observations. We visualize every 20th of the sequence.

[Figure 202]

Input

[Figure 203]

VPoser

[Figure 204]

(a)

Pose-NDF

[Figure 205]

DPoser (Ours)

[Figure 206]

GT

Motion sequence

(a) Legs under occlusion.

[Figure 207]

Input

[Figure 208]

VPoser

[Figure 209]

(b)

Pose-NDF

[Figure 210]

DPoser (Ours)

[Figure 211]

GT

Motion sequence

(b) Left arm under occlusion.

Figure S-11. Visualization of motion denoising with partial observations. We visualize every 20th of the sequence.

[Figure 212]

[Figure 213]

[Figure 214]

(a) DPoser (ours) (b) VPoser (c) NRDF

- Figure S-12. Visualization of hand pose generation. DPoser produces more diverse and realistic hand poses compared to VPoser [51] and NRDF [24].

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

Markers L2 prior VPoser NRDF DPoser (ours) w/o prior GT

[Figure 219]

- a)

[Figure 220]

- b)

[Figure 221]

- c)

[Figure 222]

- d)

- Figure S-13. Visualization of hand inverse kinematics under multiple challenging settings. Comparison across (a) noisy keypoints, (b) fingertip keypoints, (c) partial finger keypoints, and (d) sparse keypoints settings.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Images L2 prior VPoser NRDF DPoser (ours) w/o prior GT

Figure S-14. Visualization of hand mesh recovery with mean pose initialization.

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

Images L2 prior VPoser NRDF DPoser (ours) Hand4Whole GT

Figure S-15. Visualization of hand mesh recovery with Hand4Whole [47] initialization.

[Figure 233]

[Figure 234]

- a)
- b)

Markers L2 prior VPoser w/o prior DPoser (ours) GT

- c)

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

- Figure S-16. Qualitative results of face inverse kinematics on the WCPA dataset [31]. Comparison across (a) 1 mm noise, (b) 5 mm noise, and (c) half-face occlusion. Better zoom in and compare the human eyes and chin.

[Figure 239]

[Figure 240]

- (a)
- (b)

[Figure 241]

Images Keypoints L2 VPoser DPoser (ours) w/o prior GT*

[Figure 242]

[Figure 243]

[Figure 244]

Images Keypoints L2 VPoser DPoser (ours) EMOCA GT*

- Figure S-17. Visualization of face reconstruction results on the WCPA dataset [31]. Comparisons include (a) fitting from scratch and (b) initialization using EMOCA [10] results. *Ground truth lacks global orientation and translational data; these are fitted for visualization.

[Figure 245]

[Figure 246]

(a) VPoser-X (b) DPoser-X-base

[Figure 247]

[Figure 248]

(c) DPoser-X-fused (d) DPoser-X-mixed

- Figure S-18. Visualization of whole-body pose generation. (a) VPoser-X primarily generates standing poses with limited diversity. (b) DPoser-X-base generates diverse samples but lacks realism in hand interactions and facial expressions. (c) DPoser-X-fused produces less diverse samples while maintaining plausible whole-body poses. (d) DPoser-X-mixed achieves a well-balanced trade-off between diversity and realism.

[Figure 249]

- (a) DPoser-X-mixed

[Figure 250]

- (b) DPoser-X-fused

- Figure S-19. Extended visualization of whole-body pose generation. DPoser-X-mixed generates a diverse range of whole-body poses while maintaining realistic hand interactions and facial expressions. In contrast, DPoser-X-fused retains high realism but produces less diverse results.

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

Images w/o prior VPoser-X GMM DPoser-X (ours) GT

Figure S-20. Visualization of whole-body mesh recovery on the Fit3d dataset [19].

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

VPoser-X DPoser-X (ours) GT

Figure S-21. Qualitative comparison of whole-body pose completion. One hand is masked randomly.

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

DPoser-X-base DPoser-X-fused DPoser-X-mixed GT

Figure S-22. Visualization of whole-body pose completion for three DPoser-X variants. One hand is masked randomly.

