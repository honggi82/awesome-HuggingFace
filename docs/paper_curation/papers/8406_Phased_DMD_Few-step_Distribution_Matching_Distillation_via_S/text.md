# arXiv:2510.27684v3[cs.CV]25Mar2026

[Figure 1]

## Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals

Xiangyu Fan1, Zesong Qiu1, Zhuguanyu Wu1, Fanzhou Wang1, Zhiqian Lin1, Tianxiang Ren1, Dahua Lin1, Ruihao Gong1,2, Lei Yang1,

1SenseTime Research, 2Beihang University

### Abstract

Distribution Matching Distillation (DMD) distills score-based generative models into efficient one-step generators, without requiring a one-to-one correspondence with the sampling trajectories of their teachers. Yet, the limited capacity of one-step distilled models compromises generative diversity and degrades performance in complex generative tasks, e.g., generating intricate object motions in text-to-video tasks. Directly extending DMD to multi-step distillation increases memory usage and computational depth, leading to instability and reduced efficiency. While prior works propose stochastic gradient truncation as a potential solution, we observe that it substantially reduces the generative diversity in text-to-image generation and slows motion dynamics in video generation, reducing performance to the level of one-step models. To address these limitations, we propose Phased DMD, a multi-step distillation framework that bridges the idea of phase-wise distillation with Mixture-of-Experts (MoE), reducing learning difficulty while enhancing model capacity. Phased DMD incorporates two key ideas: progressive distribution matching and score matching within subintervals. First, our model divides the SNR range into subintervals, progressively refining the model to higher SNR levels, to better capture complex distributions. Next, to ensure accurate training within each subinterval, we derive rigorous mathematical formulations for the objective. We validate Phased DMD by distilling state-of-the-art image and video generation models, including Qwen-Image-20B and Wan2.2-28B. Experiments demonstrate that Phased DMD enhances motion dynamics, improves visual fidelity in video generation, and increases output diversity in image generation. Our code and models are available at https://x-niper.github.io/projects/Phased-DMD/.

### 1 Introduction

Recently, state-of-the-art (SOTA) diffusion models have made significant progress in image and video generation. In image generation, SOTA models [9, 30, 39, 44] demonstrate precise prompt control, enabling complex text-to-image rendering and accurate layout specification. In video generation, these models [10, 19, 29, 41] exhibit substantial improvements in dynamic scene generation, such as fast-moving objects in sports and complex camera movements like ego-centric videos. However, the increasing parameter sizes and computational demands of base models highlight the importance of accelerating diffusion model sampling.

Several techniques have been proposed to accelerate diffusion models, including classifier-free guidance (CFG) distillation [28], step distillation [6, 7, 14, 21, 22, 25, 26, 32, 37, 42, 47, 51], SVDQuant [20], Mixture-of-Experts (MoE) [1, 5, 41], and parallel computation [4]. Among these, step distillation methods based on Variational Score

With Gradients

Without Gradients

- xt0

GΦ

- xt1

GΦ

- xt2

GΦ

- xt3
- xt4

xt4

- xt0

GΦ

- xt1

GΦ

- xt2

GΦ

- xt3
- xt4

xt4

xt4

xt4

- xt0

GΦ

- xt1

GΦ

- xt2

xt2

- xt0

GΦ1

- xt1
- xt2

- xt0

- GΦ1

xt1

- GΦ2

- xt2

- xt0

GΦ

- xt1

- xt0
- xt1

GΦ

GΦ

GΦ xt0

GΦ

GΦ

GΦ1 Phase 1 Phase 2 Phase 3 Phase 4

- GΦ1

- xt0

GΦ2

- xt1
- xt2

- GΦ1

- xt0

GΦ2

- xt1

GΦ3

- xt2
- xt3

- GΦ1

- xt0

GΦ2

- xt1

GΦ3

- xt2

GΦ4

- xt3
- xt4

GΦ1 xt0

GΦ1

GΦ1

- GΦ1

- xt0

- GΦ1

xt1

- GΦ2

- xt2

GΦ2

- xt3
- xt4

Phase 1 Phase 2 (a) (b) (c) (d)

Stochastic Gradient Truncation

- Figure 1 Schematic diagram of the backward simulation process in (a) Vanilla few-step DMD, (b) DMD2 [47], (c) Phased DMD and (d) Phased DMD with SGTS.

Distillation (VSD) [43], including diff-instruct [26], DMD [47, 48], SID [51], achieve high-quality generation by distilling models into single-step generators. However, single-step distilled models suffer from limited network capacity [21, 22, 47], diminishing output diversity and undermining performance on complex tasks like intricate text rendering and dynamic scene generation.

Few-step distillation has emerged as a solution to this trade-off, effectively balancing computational cost with the preservation of generative quality and diversity [27, 47]. However, as illustrated in Fig. 1a, directly applying VSD to few-step distillation through multi-step generation presents challenges, including increased computational graph depth and higher memory overhead, which hinders its scalability to larger models and video generation tasks. Moreover, the absence of explicit constraints on intermediate generator steps compromises training stability and results in suboptimal performance for few-step models. To address these issues, DMD2 [47] and Self-Forcing [13] introduce a stochastic gradient truncation strategy (SGTS), where few-step backward simulation may terminate at a random step and gradient backpropagation is restricted to the final denoising step (see Fig. 1b). This approach improves training convergence and stability by supervising all intermediate steps while enhancing memory efficiency via gradient detachment for non-final steps. However, SGTS can terminate backward simulation after only one step during training, essentially distilling a one-step generator for that iteration. As a result, the generative diversity and motion dynamics of videos produced by few-step generators trained with SGTS are reduced to levels comparable to those of their one-step generators.

Diffusion theory [36] suggests the existence of infinitely many neural networks serving as score estimators across a range of signal-to-noise ratios (SNR), spanning from zero to infinity. During the generation process, diffusion models exhibit distinct temporal dynamics [1]. In particular, the low-SNR stage emphasizes modeling visual structures and dynamics, while the high-SNR stage focuses on refining visual details. In practice, a single neural network is typically utilized throughout the denoising process, requiring the model to learn and execute a diverse range of denoising tasks simultaneously. Recent studies [1, 5, 41] have integrated a Mixture of Experts (MoE) architecture into diffusion models. By assigning specialized experts to different SNR levels, MoE enhances model capacity and generative performance without increasing inference cost. The performance improvement is particularly pronounced in video generation [41], where the low-SNR expert excels at capturing dynamic content.

In this work, we propose Phased DMD, a novel distillation framework for few-step generation. Our motivation is to achieve scalable distillation for large generative models and video generation tasks by leveraging gradient truncation to manage memory constraints, while simultaneously avoiding the associated one-step degeneration problem to preserve superior generative diversity and motion dynamics. Our method is built upon two key components:

- • Progressive distribution matching: In each phase, a single expert is distilled for a specific SNR subinterval, progressively advancing toward higher SNR levels. The backward simulation terminates at the phase’s corresponding SNR level rather than the clean state, thereby mitigating the one-step degradation associated with SGTS. This framework is conceptually analogous to ProGAN [16], which progressively trains a generator to handle higher resolutions. It is fundamentally distinct from progressive distillation [32], where the objective is to halve the number of sampling steps in each phase.

- • Score matching within SNR subintervals: Since the clean sample is inaccessible in all but the final phase, the training objective must be reformulated to preserve the theoretical integrity of distribution matching. To this end, we derive a revised objective for unbiased score estimation when the clean sample is unavailable (see Sec. 2.2.2).

As illustrated in Fig. 1c, Phased DMD offers several advantages: First, by partitioning SNR into subintervals, the model learns complex data distributions incrementally, improving training stability and generative performance while avoiding degeneration into a one-step generator. Second, each phase involves only a single gradient-recorded sampling step, avoiding additional computational and memory overhead. Third, notably, Phased DMD inherently produces a few-step MoE generative model, irrespective of whether the teacher model employs an MoE architecture. Last, as shown in Fig. 1d, Phased DMD can be combined with SGTS , enabling the training of a 4-step generator within 2 phases, simplifying the overall framework complexity. We validate Phased DMD by distilling SOTA image and video generation models, including Qwen-Image [44] with 20B parameters and Wan2.1/Wan2.2 [41] with 14/28B parameters. Results demonstrate that Phased DMD enhances motion dynamics, improves visual fidelity in video generation, and increases output diversity in image generation.

Our contributions are summarized as follows:

- • Phased DMD: A data-free distillation framework for few-step diffusion models. This framework combines ideas from DMD and MoE, achieving higher performance ceilings while maintaining memory usage similar to single-step distillation.
- • Theoretical Objective: We derive the theoretical training objective for subinterval diffusion models without relying on clean samples. We highlight the necessity of this correctness for DMD distillation.
- • SOTA Performance: Without requiring GAN loss or regression loss, Phased DMD achieves SOTA results on textto-image and text-to-video tasks. To the best of our knowledge, this is the largest reported distillation validation for diffusion models. Experiments show that our method effectively reduces diversity loss while preserving the base models’ key capabilities, including complex text rendering and highly dynamic video generation.

### 2 Method

To clarify the principle of Phased DMD, we begin by introducing the theoretical background and notations related to diffusion models [18, 49], score matching [17, 36], and distribution matching distillation [47, 48]. We explicitly highlight why the principle of DMD is applicable only to score-based generative models. Building on this foundation, we present the motivation behind Phased DMD and explain how it inherently achieves improved generative diversity. Following this, we detail the two key components of Phased DMD: progressive distribution matching and score matching within subintervals.

#### 2.1 Preliminary

##### 2.1.1 Diffusion Models and Score Matching

Consider a continuous-time Gaussian diffusion process defined over the interval 0 ≤ t ≤ 1. The ground-truth distribution is denoted p(x0). For any 0 ≤ t ≤ 1, the forward diffusion process is described by the following conditional distribution:

p(xt|x0) = N(αtx0,σt2I) (1) where αt and σt2 are positive, scalar functions of t. The signal-to-noise ratio (SNR) is defined as SNR(t) = α

2 t

σt2 . It is assumed that SNR(t) is strictly monotonically decreasing over time. No additional constraints are imposed on the relationship between αt and σt, ensuring the notations are compatible with different kinds of diffusion models [11, 17, 31, 35] and flow models [3, 23]. The diffusion process is Markovian [18], meaning that p(xt|xs,x0) = p(xt|xs). For any 0 ≤ s < t ≤ 1, p(xt|xs) is also Gaussian, and can be expressed as:

p(xt|xs) = N(αt|sxs,σt2|sI) (2) where αt|s = α

αs and σt2|s = σt2 − αt2|sσs2. For the marginal distribution of xt, we have the following equivalence: p(xt) = p(xt|x0)p(x0)dx0 = p(xt|xs)p(xs)dxs (3)

t

In the training process, αt and σt are predefined functions of t, while x0 is sampled from the dataset distribution x0 ∼ p(x0). Timestep t is sampled from a predefined distribution over the interval [0, 1], such as a uniform or logit-normal distribution [3], i.e., t ∼ T (0,1). The sample xt is then given by xt = αtx0 + σtϵ, where ϵ ∼ N(0,I). We use t ∼ T and ϵ ∼ N for brevity in later paragraphs unless otherwise specified. Song et al. [36] unified diffusion models under the theoretical framework of score-based generative models and demonstrated that the continuous diffusion process is fundamentally governed by a Stochastic Differential Equation (SDE). Here, we adopt flow velocity prediction as an example and demonstrate its connection to score matching. Let ψ denote a diffusion model. The relationship between flow matching loss and score matching is expressed below.

0,t,ϵ,xt=αtx0+σtϵ[∥ψ(xt,t) − (ϵ − x0)∥2] (4)

Ex

σt2 αt

xt αt

log(p(xt))∥2] (5)

[∥ψ(xt,t) +

)∇xt

= Et,x

+ (σt +

t

Eq. 5 is derived based on the equivalence between denoising score matching (DSM) and explicit score matching (ESM), as originally proven in [40]. In supplementary materials, we provide the detailed derivation of Eq. 5. Additionally, we demonstrate the connection between sample prediction (a.k.a. x-prediction) and score matching in the supplementary materials.

##### 2.1.2 Distribution Matching Distillation

The DMD framework comprises three components: a trainable generator Gϕ, a trainable fake score estimator Fθ, and a frozen pretrained teacher score estimator Tθˆ, parameterized by ϕ, θ and θˆ, respectively. Usually, both ϕ and θ are initialized from θˆ. Formally, the objective of DMD is to minimize the reverse Kullback-Leibler (KL) divergence between the real data distribution preal(x0) and the generated data distribution pfake(x0), the latter produced by Gϕ.

∇ϕDKL(pfake∥preal) = Ez,x

0=Gϕ(z)[(∇x0

log pfake(x0) − ∇x0

log preal(x0))

dG dϕ

] (6)

where z ∼ N(0,I) is a random Gaussian noise input. We use DKL to abbreviate DKL(pfake∥preal) in later paragraphs. To leverage diffusion models as score estimators, the generated samples are diffused and the objective becomes:

dG dϕ

∇ϕDKL ≈ Ez,x

0=Gϕ(z),t,ϵ,xt=αtx0+σtϵ[wt(Tθˆ(xt,t) − Fθ(xt,t))

] (7)

2 t

where wt = α

αtσt+σt2 .

Intuitively, in DMD, the generator Gϕ is optimized to approximate the real data distribution. The fake score estimator Fθ is trained to estimate the score of the generator’s output distribution. The update direction for the generator is then governed by the discrepancy between the teacher score estimator and the fake score estimator. The theoretical foundation of DMD rests on two key assumptions concerning the fake score estimator Fθ.

- A1: Converged Fθ. Analogous to GANs [8], DMD itself is an adversarial training paradigm that proceeds in two stages per iteration. In the fake diffusion optimization stage, Fθ is trained on the generated distribution according to the following loss:

Ez,x

0=Gϕ(z),t,ϵ,xt=αtx0+σtϵ[∥Fθ(xt,t) − (ϵ − x0)∥2] (8) This enables Fθ to function as an effective score estimator for pfake(xt). In the generator optimization stage, Gϕ is updated according to Eq. 7, thereby encouraging the generated distribution to more closely approximate the real data distribution. Theoretically, the convergence of DMD hinges on the convergence of Fθ for each update of Gϕ. In practice, Fθ is updated more frequently than Gϕ, allowing it to accurately estimate the score of the evolving generated distribution [47].

- A2: Unbiased Fθ. Suppose Tθˆ(xt,t) ≈ at∇xt

log(pfake(xt)) + dtxt, where at,bt,ct,dt are scalar functions of t, then the derivation from Eq. 6 to Eq. 7 holds provided at = ct and bt = dt. In DMD, this condition is inherently satisfied, owing to the identical training targets in Eq. 4 and Eq. 8.

log(preal(xt)) + btxt and Fθ(xt,t) ≈ ct∇xt

##### 2.1.3 Few-step Generator

In N-step distillation, a diffusion scheduler S is employed with N + 1 timesteps, denoted by the sequence ⃗t = {t0,t1,t2,...,tN}, where 0 = tN < ti < ti−1 < t0 = 1 for all i ∈ {2,...,N − 1}. The backward simulation [47] begins with xt

= z ∼ N(0,I). The sample x0 is then generated iteratively: for i = 0,1,...,N − 1, xt

= S(Gϕ(xt

0

i+1

,ti,ti+1). Although DMD2 [47] utilizes the LCM [24] scheduler, S can be any stochastic or deterministic solver. Let pipeline(Gϕ,⃗t,z,S) denote this iterative sampling procedure. Eq. 7 is thus adapted as follows:

,ti),xt

i

i

dG dϕ

] (9)

∇ϕDKL ≈ Ez,x

0=pipeline(Gϕ,⃗t,z,S),t,ϵ,xt=αtx0+σtϵ[wt(Tθˆ(xt,t) − Fθ(xt,t))

As illustrated in Fig. 1a, during generator optimization stage, the depth of the computational graph increases linearly with N owing to the few-step inference process, thereby compromising training stability and elevating memory overhead. To mitigate this challenge, DMD2 [47] implemented a stochastic gradient truncation strategy (SGTS), as illustrated in Fig. 1b. Although this strategy was not explicitly named in DMD2, it was later referred to as such in a following work [13]. Under this approach, an index j is sampled uniformly at random from {1,2,...,N} and the corresponding timestep tj is set to 0. The sampling pipeline is then executed solely for the steps i = 0,1,...,j − 1. Crucially, when j = 1, the training iteration collapses to a one-step distillation. Consequently, although SGTS enhances memory efficiency and training stability, it compromises generative diversity, impairs prompt-following capability and yields slower video motion.

#### 2.2 Phased DMD

In contrast to DMD2, which may degenerate into one-step distillation during certain iterations and thereby diminish effective model capacity, Phased DMD circumvents this limitation by partitioning the distillation process into distinct phases and imposing supervision at intermediate timesteps. In every phase except the final one, the generator is optimized to minimize the reverse KL divergence at an intermediate timestep, while the fake diffusion model is updated through score matching over a subinterval of the diffusion process.

##### 2.2.1 Distribution Matching at Intermediate Timesteps

The rationale underlying Phased DMD can be elucidated by revisiting Eq. 9. To sample xt, prior methods [13, 47, 48] first generate x0 and then diffuse it to xt according to Eq. 1. In Phased DMD , the backward simulation is adapted to produce intermediate samples xt

,{t1,t2,...,tk},z,S), where 0 < k ≤ N, rather than x0. For brevity, the backward simulation is henceforth denoted as xt

, i.e., xt

= pipeline(Gϕ

,...,Gϕ

1

k

k

k

= pipelinek(z) in the subsequent discussion. The sample xt

k

is then diffused according to Eq. 2, with s = tk, and t is sampled from the subinterval (tk,1), i.e., t ∼ T (tk,1). The generator optimization objective for the k-th phase is formulated as:

k

dG dϕk

] (10)

∇ϕk

DKL ≈ Ez,x

(Tθˆ(xt,t) − Fθ

tk=pipelinek(z),t,ϵ,xt=αt|tkxtk+σt|tkϵ[wt|t

(xt,t))

i

k

= αtαt|t

where wt|t

αtσt+σt2. Empirically, we observe that sampling t ∼ T (tk,1) rather than t ∼ T (tk,tk−1), aligns more effectively with the progressive architecture of Phased DMD and delivers superior performance. Please refer to the supplementary materials for an ablation study comparing the two sampling ranges.

k

k

As illustrated in Fig. 1c, Phased DMD progressively distills the generator toward higher SNR levels. In each phase k, only a single expert Gϕ

). At the beginning of each phase, the fake diffusion model Fθ

is trained. This expert maps the distribution p(xt

) to p(xt

k−1

k

k

is initialized from the pretrained teacher model Tθˆ rather than from the model of the preceding phase, i.e., Fθ

k

.

k−1

##### 2.2.2 Score Matching Within Subintervals

A central challenge in Phased DMD is that clean samples x0 are inaccessible in all phases except the final one. Consequently, the training objective for the fake diffusion model Fθ

, as presented in Eq. 8, becomes inapplicable. To address this, we derive a training objective based on score matching over a subinterval, which yields unbiased score

k

- 0

- 1

- 2

- 3

- 0

- 1

- 2

- 3

- 0

- 1

- 2

- 3

flow match

SamplingTrajectory

SamplingTrajectory

SamplingTrajectory

- 0.0 0.2 0.4 0.6 0.8 1.0 Timestep

3

2

- 1

- 0.0 0.2 0.4 0.6 0.8 1.0 Timestep

3

2

- 1

- 0.0 0.2 0.4 0.6 0.8 1.0 Timestep

3

2

- 1

flow match

flow match

unbiased flow match

biased flow match

(a)

(b)

###### (c)

- Figure 2 Sampling trajectories of 200 samples in a 1D toy experiment. (a) Training with the full-interval objective (Eq. 4). (b) Training on 0.5 < t < 1 with the correct subinterval objective (Eq. 13). (c) Training on 0.5 < t < 1 with an incorrect target: ∥(ψ(xt, t) − (ϵ − xs)∥2.

estimates. Specifically, a diffusion model ψ trained over the full interval t ∼ T (0,1) via Eq. 4 should instead be optimized over the subinterval t ∼ T (s,1) using the following loss:

Ex

s,t,ϵ,xt=αt|sxs+σt|sϵ[∥ψ(xt,t) − (

αs2σt + αtσs2 αs2σt|s

ϵ −

1 αs

xs)∥2] (11)

For x-prediction diffusion models, as employed in [13, 17, 27, 51], the diffusion model µ should be optimized over the subinterval t ∼ T (s,1) using the following loss:

Ex

s,t,ϵ,xt=αt|sxs+σt|sϵ[∥µ(xt,t) − (

1 αs

xs −

αtσs2 αs2σt|s

ϵ)∥2] (12)

Please refer to the supplementary materials for the detailed derivation. As σt|s → 0 when t → s, the formulation in Eq. 11 encounters singularity and numerical instability. To mitigate this, we apply a clamping function, resulting in the final loss term:

clamp(

1 σt2|s

)∥σt|sψ(xt,t) − (

αs2σt + αtσs2 αs2

ϵ −

σt|s αs

xs)∥2 (13)

Here, clamp(σ12

t|s

) restricts the value within a predefined range [0,10] to prevent overflow.

We design a one-dimensional toy experiment to validate the efficacy of this training objective, wherein x0 takes only four discrete values: {-1, 0, 1, 2}, as shown in Fig. 2. The diffusion model is parameterized by a four-layer MLP with hidden dimension 512. Three models are trained from scratch: (a) one using loss Eq. 4 over the full interval (0,1]; (b) another using loss Eq. 13 over the subinterval (0.5,1]; and (c) a third using loss term ∥ψ(xt,t) − (ϵ − xs)∥2 over the same subinterval. Following training, trajectories are sampled using an Euler solver with 100 steps. The close overlap of the sampling trajectories in Fig. 2b confirms that, within the specified subinterval, the flow model trained with Eq. 13 is equivalent to one trained with the standard objective in Eq. 4. In contrast, Fig. 2c illustrates how an incorrect objective formulation induces biased score estimation, thereby violating the A2 assumption.

In the k-th phase of Phased DMD, the fake diffusion model Fθ

k

is optimized over the subinterval (tk,1] using Eq. 13. As established in the foregoing analysis, Fθ

k

satisfies the A2 assumption. Following DMD2 [47], we perform 5 updates to Fθ

k

for each update to Gϕ

k

.

- 3 Experiments and Results

To evaluate the efficacy of Phased DMD , we conduct experiments across text-to-image (T2I), text-to-video (T2V) and image-to-video (I2V) generation tasks. SOTA base models are employed as teachers, including Wan2.1-T2V-14B,

Wan2.2-T2V-A14B, Wan2.2-I2V-A14B and Qwen-Image-20B. An overview of the experimental configurations is presented in Tab. 1. Due to its substantial computational demands, the vanilla few-step DMD is applied only to the smallest configuration, namely, T2I task using the Wan2.1-T2V-14B base model.

We conduct experiments on 64 GPUs, employing PyTorch FSDP and gradient checkpointing to reduce GPU memory consumption. Context parallelism is applied for T2V and I2V distillation. The following settings are used consistently across all experiments: a batch size of 64; a fake diffusion model learning rate of 4 × 10−7 with full-parameter training; a generator learning rate of 5×10−5 using LoRA [12] with rank = 64 and α = 8. Following prior works [13], AdamW optimizer is employed for both the fake diffusion model and the generator, with hyperparameter β1 = 0,β2 = 0.999. The fake diffusion model is updated 5 times for each generator update. Euler solver is used in backward simulation, due to its simplicity. To align with the two-expert architecture of Wan2.2, we adopt a 4-step, 2-phase configuration for Phased DMD , as illustrated in Fig. 1d. Consequently, each base model is distilled into two expert networks.

To demonstrate that the performance gains arise primarily from our novel distillation paradigm rather than from an increase in trainable parameters, Wan2.2-T2V-A14B is distilled for both T2I and T2V tasks. Wan2.2-T2V-A14B already incorporates an MoE architecture, and both standard DMD and our Phased DMD distill it into two experts, thereby enabling a direct comparison under equivalent parameter budgets.

- Table 1 Overview of the experimental setup. “Vanilla DMD” refers to vanilla few-step DMD shown in Fig. 1a. “Ours” refers to Phased DMD . Some experiments employ mixed data resolutions. The reported values for Frame, Height and Width represent one resolution.

Base Model Task Vanilla DMD DMD2 Ours Timesteps Frame, Height, Width

- Wan2.1-T2V-14B T2I ✓ ✓ ✓ 1000, 938, 833, 625 1, 720, 1280
- Wan2.2-T2V-A14B T2I × ✓ ✓ 1000, 938, 833, 625 1, 720, 1280 Wan2.2-T2V-A14B T2V × ✓ ✓ 1000, 938, 833, 625 81, 720, 1280 Wan2.2-I2V-A14B I2V × ✓ ✓ 1000, 938, 833, 625 81, 720, 1280 Qwen-Image-20B T2I × ✓ ✓ 1000, 900, 750, 500 1, 1382, 1382

- Table 2 Quantitative comparison of video generation performance. “OF” refers to optical flow. “DD” refers to dynamic degree. Phased DMD outperforms DMD2 in motion dynamics (OF, DD) and video quality (FID, FVD) on both T2V and I2V tasks.

T2V I2V OF ↑ DD ↑ FID ↓ FVD ↓ OF ↑ DD ↑ FID ↓ FVD ↓ Base model 10.26 79.55 % 0.0 0.0 9.32 82.27 % 0.0 0.0

Method

DMD2 3.23 65.45 % 55.70 763.1 7.87 80.00 % 18.45 370.0 Phased DMD (Ours) 9.30 82.27 % 47.24 700.9 9.84 83.64 % 17.47 334.7

#### 3.1 Phased DMD in Video Generation

Wan2.2 video generation models exhibit remarkable proficiency in motion dynamics and camera control. However, we observe that DMD2 tends to degrade these qualities, due to the one-step degeneration inherent in SGTS. Phased DMD intrinsically addresses this limitation by explicitly eliminating dependence on x0 in intermediate phases. In the first phase, only the low-SNR expert is activated. As the pretrained low-SNR expert in Wan2.2 is already trained on the low-SNR subinterval, this correspondence better preserves its inherent capabilities. Motion quality is evaluated using 220 text prompts for T2V and 220 image-prompt pairs for I2V, with one video generated per prompt using a fixed seed of 42. The base model is sampled with 40 steps and a guidance scale of 4, while the distilled models utilize 4 steps and a guidance scale of 1. As illustrated in Fig. 4, DMD2 produces slower motion dynamics compared to both the base model and Phased DMD . Similarly, Fig. 5 reveals that DMD2 tends to generate close-up views, whereas Phased DMD and the base model more faithfully adhere to the prompt-specified camera instructions.

[Figure 2]

[Figure 3]

[Figure 4]

(a) base (b) DMD2 (c) Phased DMD

Figure 4 Motion dynamics comparison of video frames generated by (a) Wan2.2-T2V-A14B base model and its distilled versions using (b) DMD2 and (c) Phased DMD. Each video consists of 81 frames and frames with indices {0, 10, ..., 80} are combined as a preview. A portion of the full prompt: “A parkour athlete swiftly runs horizontally along a brick wall in an urban setting. Pushing off powerfully with one foot, they launch themselves explosively into a twisting front flip. ”

[Figure 5]

[Figure 6]

[Figure 7]

(a) base (b) DMD2 (c) Phased DMD

Figure 5 Camera following capability comparison of video frames generated by (a) Wan2.2-T2V-A14B base model and its distilled versions using (b) DMD2 and (c) Phased DMD. Each video consists of 81 frames and frames with indices {0, 10, ..., 80} are combined as a preview. A portion of the full prompt: “The camera starts focused on the skates carving sharp turns on the pavement and tilts up to reveal their entire body leaning into the motion.”

To quantitatively evaluate our method, motion intensity is quantified using mean absolute optical flow computed with UniMatch [45] and dynamic degree metric from VBench [15]. Video visual quality is assessed via FID and FVD [34] computed between the distilled models and the base models. As Tab. 2 shows, Phased DMD yields significantly stronger motion dynamics than DMD2, confirming its superior capacity to preserve the motion dynamics of the base model. Furthermore, the lower FID and FVD values indicate that Phased DMD better maintains the generative quality. We also report results from additional VBench metrics. However, these metrics are less convincing, as they consistently rank the base model lowest, which contradicts human assessment. Further comparative videos and quantitative analyses are provided in the supplementary materials.

#### 3.2 Phased DMD in Image Generation

We apply Phased DMD to distill three base models for text-to-image generation. To evaluate generative diversity, we constructed a test set of 21 prompts, each providing a short image description without detailed specifications. For each prompt, we generated 8 images using random seeds from 0 to 7. The base model was sampled with 40 steps and a guidance scale of 4, while all distilled models used 4 steps and a guidance scale of 1. As shown in Fig. 6b, images generated by the vanilla 4-step DMD model exhibit a loss of fine details. Although the 4-step DMD2 model improves image quality, it does so at the cost of reduced diversity. Fig. 6c reveals that the generated images often converge to a similar close-up view with limited compositional variation across seeds. In contrast, Phased DMD better preserves diversity, producing images with a wider range of natural compositions and lighting conditions, as illustrated in Fig. 6d.

Generative diversity is evaluated using two complementary metrics: (1) the mean pairwise cosine similarity of DINOv3 features [33], where lower values indicate higher diversity, and (2) the mean pairwise LPIPS distance [50], where higher values denote greater diversity. Both metrics are computed across images generated from the same prompt using different seeds. The quantitative results are presented in Tab. 3. As expected, the base models achieve the highest diversity. Phased DMD outperforms both vanilla DMD and DMD2, demonstrating its superior capability for preserving the generative diversity of the original model. The diversity improvement on Qwen-Image is marginal. We argue this stems from the base model’s own limited output diversity.

Qwen-Image is recognized for its faithful adherence to prompts and high-quality text rendering. To evaluate the preservation of these capabilities after distillation, we applied Phased DMD to Qwen-Image and generated images using prompts from its official website [38]. As shown in Fig. 7, the model distilled with Phased DMD exhibits well-preserved capabilities, producing high-quality images with accurate text rendering.

- seed 0

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

- seed 1

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

- seed 2

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

- seed 3 (a) Base (b) Vanilla (c) DMD2 (d) Ours (e) Base (f) Vanilla (g) DMD2 (h) Ours

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Figure 6 Samples (seeds 0-3) from the Wan2.1-T2V-14B base model (40 steps, CFG=4) and its distilled variants (4 steps, CFG=1): (a, e) Base, (b, f) Vanilla few-step DMD, (c, g) DMD2, (d, h) Phased DMD. Left: “A chef meticulously plating a dish.”. Right: “A mother braiding her daughter’s hair, sunlight warming the room.”

#### 3.3 Merit of MoE

Our empirical observations indicate that during distillation, DMD first captures structural information before acquiring finer textural details. Prior to fully learning these textural nuances, generated images and videos often exhibit overly smooth characteristics, such as blurred hair or plastic-like skin textures. Meanwhile, the mode-seeking behavior of the reverse KL divergence causes a decline in generative diversity and motion intensity as training continues. Phased

- Table 3 Two metrics for quantitative diversity evaluation: average pairwise DINOv3 cosine similarity (lower is better) and LPIPS distance (higher is better). Phased DMD outperforms the vanilla DMD and DMD2 in preserving generative diversity of the base models.

Method

Wan2.1-T2V-14B Wan2.2-T2V-A14B Qwen-Image

DINOv3 ↓ LPIPS ↑ DINOv3 ↓ LPIPS ↑ DINOv3 ↓ LPIPS ↑ Base model 0.708 0.607 0.732 0.531 0.907 0.483

Vanilla DMD 0.825 0.522 - - - -

DMD2 0.826 0.521 0.828 0.447 0.941 0.309 Phased DMD (Ours) 0.782 0.544 0.768 0.481 0.958 0.322

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Figure 7 Examples generated by the Qwen-Image distilled with Phased DMD.

DMD mitigates this trade-off by partitioning the distillation process into distinct training phases. In the low-SNR phases, the compositional structure of images and videos is effectively established. In subsequent high-SNR phases, the low-SNR expert is frozen, enabling prolonged training to refine generative quality without compromising the established structural layout. Combined with its inherent avoidance of one-step degeneration, Phased DMD leverages the MoE architecture to enhance fine-grained details while preserving structural diversity and motion intensity. As shown in Fig. 8, extending training of the high-SNR expert primarily influences lighting and textural details, while leaving the overall structural composition of the images intact.

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 8 Samples generated with high-SNR experts from different training stages (top: 100 iterations; bottom: 400 iterations) and a shared low-SNR expert. Each column uses identical prompts and seeds.

- 4 Related Work

Our work is situated within the framework of Variational Score Distillation (VSD) [43]. VSD involves three components: a trainable generator, a fake score estimator, and a pretrained teacher score estimator. The generator is optimized to produce a distribution that approximates the real data distribution. Concurrently, the fake score estimator learns to estimate the score of the generator’s output distribution. The update direction for the generator is then determined by the

discrepancy between the teacher’s score (for the real distribution) and the fake score estimator’s score.

Similar to GANs, the VSD framework is adversarial. The fake score estimator must be precisely optimized to learn the score of the current generated distribution. This accurate estimation is crucial, as it combines with the fixed teacher model (which provides the score for the real data) to produce a correct guidance signal for the generator. This principle explains why DMD2 [47] operates successfully without external real data, in contrast to its predecessor DMD [48].

A key advantage of VSD over GANs for distilling pre-trained diffusion models is initialization. The pre-trained model serves a dual role: it is a powerful multi-step generator and an accurate estimator of the real data distribution’s score. This allows it to effectively initialize all three components in the VSD framework, leading to significantly enhanced training stability.

Several methods are built upon the VSD framework, including Diff-Instruct [26], DMD [47], SID [51], and FGM [14]. The fundamental distinction between these approaches lies in the specific divergence they minimize. DMD, for instance, optimizes the reverse KL divergence between the real and generated distributions. A key advantage of this choice is its computational efficiency compared to alternatives like the Fisher divergence used in SID [51]. Specifically, during generator optimization, DMD does not require gradients to be backpropagated through the fake and teacher score estimators, whereas SID does. This does not imply the two estimators are trainable in this stage for SID, but rather reflects a difference in the computational graph. This property makes DMD more amenable to engineering implementation and scalable to large base models.

Similar to our work, TDM [27] also aimed to extend DMD to few-step distillation. However, our approach differs from TDM in three key aspects: (a) The lack of proper theoretical grounding in TDM renders its fake flow training formulation suboptimal, undermining the foundations of DMD. (b) Our framework inherently produces MoE models for few-step generation. (c) While TDM uses disjoint SNR intervals, our method employs reverse nested intervals, where each interval is a subset of the subsequent one.

### 5 Conclusion and Discussion

Phased DMD primarily enhances structural aspects of generation, such as composition diversity, motion dynamics, and camera control. However, for base models like Qwen-Image, whose outputs are inherently less diverse, the improvement is less pronounced. While this work demonstrates phased distillation within the DMD framework, the approach is generalizable to other objectives like Fisher divergence in SiD [51], which we leave for future exploration. It is conceivable that other methods for enhancing diversity and dynamics, such as incorporating trajectory data pre-generated by the base model, could be integrated. However, this would compromise the data-free advantage central to DMD. While we may explore such directions in the future, this work prioritizes the data-free paradigm.

### References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

- [2] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [3] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.
- [4] Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. arXiv preprint arXiv:2411.01738, 2024.

- [5] Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, et al. Ernie-vilg 2.0: Improving text-to-image diffusion model with knowledge-enhanced mixture-of-denoising-experts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10135–10145, 2023.

- [6] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024.

- [7] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

- [8] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014.
- [9] GoogleAI. Image generation with gemini (aka nano banana), 2025.
- [10] GoogleAI. Generate videos with veo 3 in gemini api, 2025.
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

- [12] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.
- [13] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

- [14] Zemin Huang, Zhengyang Geng, Weijian Luo, and Guo-jun Qi. Flow generator matching. arXiv preprint arXiv:2410.19310, 2024.

- [15] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024.

- [16] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196, 2017.

- [17] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

- [18] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models, 2023.
- [19] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

- [20] Muyang Li*, Yujun Lin*, Zhekai Zhang*, Tianle Cai, Xiuyu Li, Junxian Guo, Enze Xie, Chenlin Meng, Jun-Yan Zhu, and Song Han. Svdquant: Absorbing outliers by low-rank components for 4-bit diffusion models. In The Thirteenth International Conference on Learning Representations, 2025.

- [21] Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.

- [22] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation. arXiv preprint arXiv:2501.08316, 2025.

- [23] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

- [24] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference, 2023.
- [25] Weijian Luo. Diff-instruct++: Training one-step text-to-image generator model to align with human preferences. arXiv preprint arXiv:2410.18881, 2024.

- [26] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36: 76525–76546, 2023.

- [27] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching. arXiv preprint arXiv:2503.06674, 2025.

- [28] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14297–14306, 2023.

- [29] OpenAI. Video generation models as world simulators, 2024.
- [30] OpenAI. Introducing 4o image generation, 2025.
- [31] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023.
- [32] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

- [33] Oriane Sim´eoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timoth´ee Darcet, Th´eo Moutakanni, Leonel Sentana, Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herv´e J´egou, Patrick Labatut, and Piotr Bojanowski. DINOv3, 2025.
- [34] Ivan Skorokhodov, Sergey Tulyakov, and Mohamed Elhoseiny. Stylegan-v: A continuous video generator with the price, image quality and perks of stylegan2. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3626–3636, 2022.

- [35] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022.
- [36] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

- [37] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023.
- [38] Qwen Team. Qwen-image: Crafting with native text rendering, 2025.
- [39] Tencent Hunyuan Team. Hunyuanimage 2.1: An efficient diffusion model for high-resolution (2k) text-to-image generation. https://github.com/Tencent-Hunyuan/HunyuanImage-2.1, 2025.
- [40] Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.

- [41] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

- [42] Fu-Yun Wang, Zhaoyang Huang, Alexander Bergman, Dazhong Shen, Peng Gao, Michael Lingelbach, Keqiang Sun, Weikang Bian, Guanglu Song, Yu Liu, et al. Phased consistency models. Advances in neural information processing systems, 37: 83951–84009, 2024.

- [43] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36: 8406–8441, 2023.

- [44] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025.
- [45] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.

- [46] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025.
- [47] Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024.

- [48] Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation, 2024.
- [49] Pengze Zhang, Hubery Yin, Chen Li, and Xiaohua Xie. Tackling the singularities at the endpoints of time intervals in diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6945–6954, 2024.

- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.

- [51] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

## Appendix

- A Detailed Derivation We show the detailed derivation of Eq. 5 as follows:

Jflowmatch (14)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥ψ(xt,t) − (ϵ − x0)∥2] (15)

= Ex

xt − σtϵ αt

)∥2] (16)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥ψ(xt,t) − (ϵ −

= Ex

1 αt

σt αt

)ϵ∥2] (17)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥ψ(xt,t) +

xt − (1 +

= Ex

σt2 αt

1 αt

log(p(xt|x0))∥2] (18)

0∼p(x0),t∼T ,xt∼p(xt|x0)[∥ψ(xt,t) +

)∇xt

= Ex

xt + (σt +

σt2 αt

1 αt

log(p(xt))∥2] (19)

)∇xt

t∼p(xt)[∥ψ(xt,t) +

xt + (σt +

= Et∼T ,x

In the derivation, we use the score of p(xt|x0), i.e., ∇xt

ϵ, and the equivalence between DSM and

log(p(xt|x0)) = −σ1

t

ESM [40]. We show the detailed derivation of Eq. 11 as follows:

Jsubinterval−flowmatch (20)

σt2 αt

1 αt

log(p(xt))∥2] (21)

t∼p(xt)[∥ψ(xt,t) +

)∇xt

= Et∼T (t;s,1),x

xt + (σt +

σt2 αt

1 αt

log(p(xt|xs))∥2] (22)

)∇xt

s∼p(xs),t∼T (t;s,1),xt∼p(xt|xs)[∥ψ(xt,t) +

xt + (σt +

= Ex

2 t

σt + σ

1 αt

αt σt|s

ϵ∥2] (23)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥ψ(xt,t) +

xt −

= Ex

2 t

σt + σ

αt|sxs + σt|sϵ αt −

αt σt|s

ϵ∥2] (24)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥ψ(xt,t) +

= Ex

αs2σt + αtσs2 αs2σt|s

1 αs

xs)∥2] (25)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥ψ(xt,t) − (

ϵ −

= Ex

The relationship between sample prediction (x-prediction) and score matching is derived as follows:

Jsample (26)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥µ(xt,t) − x0∥2] (27)

= Ex

xt − σtϵ αt ∥2] (28)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥µ(xt,t) −

= Ex

1 αt

σt αt

ϵ∥2] (29)

0∼p(x0),t∼T ,ϵ∼N,xt=αtx0+σtϵ[∥µ(xt,t) −

= Ex

xt +

σt2 αt ∇xt

1 αt

log(p(xt|x0))∥2] (30)

0∼p(x0),t∼T ,xt∼p(xt|x0)[∥µ(xt,t) −

xt −

= Ex

σt2 αt ∇xt

1 αt

log(p(xt))∥2] (31)

t∼p(xt)[∥µ(xt,t) −

xt −

= Et∼T ,x

The training objective for x-prediction diffusion models within a subinterval is as follows:

Jsubinterval−sample (32)

σt2 αt ∇xt

1 αt

log(p(xt))∥2] (33)

t∼p(xt)[∥µ(xt,t) −

xt −

= Et∼T ,x

σt2 αt ∇xt

1 αt

log(p(xt|xs))∥2] (34)

s∼p(xs),t∼T (t;s,1),xt∼p(xt|xs)[∥µ(xt,t) −

xt −

= Ex

σt2 αtσt|s

1 αt

ϵ∥2] (35)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥µ(xt,t) −

= Ex

xt +

σt2 αtσt|s

αt|sxs + σt|sϵ αt

ϵ∥2] (36)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥µ(xt,t) −

= Ex

+

αtσs2 αs2σt|s

1 αs

ϵ)∥2] (37)

s∼p(xs),t∼T (t;s,1),ϵ∼N,xt=αt|sxs+σt|sϵ[∥µ(xt,t) − (

xs −

= Ex

### B Experimental Details

For the Wan2.x base models, distillation on the text-to-image task is conducted at a fixed data resolution of one frame with width 1280 and height 720, i.e., frame = 1,width = 1280,height = 720.

For the Wan2.2-x2V-A14B model, distillation on both the text-to-video and image-to-video tasks employs a mixture of data resolutions: (81,720,1280), (81,1280,720), (81,480,832), (81,832,480), sampled with probabilities 0.1,0.1,0.4,0.4.

For the Qwen-Image model, distillation on the text-to-image task employs a uniform sampling from a set of resolutions: (1,1382,1382), (1,1664,928), (1,928,1664), (1,1472,1104), (1,1104,1472), (1,1584,1056), (1,1056,1584).

A timestep shift of 5 is applied for Wan2.x base models, following the self-forcing approach [13] while a shift of 3 is used for the Qwen-Image base model, based on ComfyUI’s Qwen-Image workflow. The resulting timesteps are provided in Tab. 1. Fig. 9 illustrates how intervals are determined by the steps and the timestep shift value.

Step = 4: [1.00, 0.75, 0.50, 0.25] If shift = 5, [1.00, 0.938, 0.833, 0.625] If shift = 3, [1.00, 0.90, 0.75, 0.50]

(0,1000)

(0,1000)

x0

x0

x0

(625,1000)

- GΦ1 x1000

- GΦ2 x938

- GΦ3 x833

- GΦ4 x625

- GΦ1 x1000

- GΦ1 x938

- GΦ2 x833

- GΦ2 x625

x625

(833,1000)

(833,1000)

- GΦ1 x1000

- GΦ2 x938

- GΦ3 x833

- GΦ1 x938

- GΦ2 x833

x833

x833

x833

(938,1000)

- GΦ1 x1000

- GΦ2 x938

GΦ1 x938

x938

GΦ1 x1000

GΦ1 x1000

GΦ1 x1000

GΦ1 x1000

Phase 1 Phase 2 Phase 3 Phase 4

Phase 1 Phase 2

4-step 4-phase 4-step 2-phase

- Figure 9 The timestep intervals are determined by the steps and the timestep shift value. For Wan-based models, shift = 5 is used, following the Self-Forcing approach. For Qwen-Image-based models, shift = 3 is adopted, based on ComfyUI’s Qwen-Image workflow. The re-noising t range for each phase is represented as a tuple (min step, max step) and t is uniformly sampled before applying the shift. For instance, in a 4-step, 2-phase setting, t is sampled within (0.5, 1) and then apply a shift of 5 during Phase 1.

For Wan2.2 base models, the first training phase exclusively employs the high-noise model. During this phase, the re-noising timestep t is restricted (by torch.clamp) to the range (0.875, 1) for the T2V task and (0.9, 1) for the I2V task, in

accordance with the boundary timestep configuration of the high-noise model. In the second phase, both the high-noise and low-noise models are employed and three components are trainable: the low-noise generator, the high-noise fake model and the low-noise fake model. The choice of high-noise or low-noise teacher and fake model is determined by the values of the re-noising timestep. This training paradigm aligns with our strategy for sampling noise injection timesteps,

- as detailed in Sec. C.2.

### C More Results

#### C.1 Additional Quantitative Evaluation on Video Generation

Table 4 More quantitative comparison on text-to-video generation. The base model is Wan2.2-T2V-A14B.

aesthetic background motion subject temporal

Method

quality consistency smoothness consistency flickering Base model 63.62 % 94.03 % 97.67 % 90.06 % 95.70 %

DMD2 67.02 % (+3.40 %) 95.29 % (+1.26 %) 98.57 % (+0.90 %) 92.93 % (+2.87 %) 97.08 % (+1.38 %) Phased DMD(Ours) 65.73 % (+2.11 %) 94.40 % (+0.37 %) 97.74 % (+0.07 %) 91.15 % (+1.09 %) 95.26 % (-0.44 %)

Table 5 More quantitative comparison on image-to-video generation. The base model is Wan2.2-I2V-A14B.

aesthetic background motion subject temporal

Method

quality consistency smoothness consistency flickering Base model 62.71 % 93.75 % 97.74 % 90.39 % 95.52 %

DMD2 64.14 % (+1.43 %) 94.44 % (+0.69 %) 97.85 % (+0.11 %) 91.84 % (+1.45 %) 95.73 % (+0.21 %) Phased DMD(Ours) 63.91 % (+1.20 %) 94.16 % (+0.41 %) 97.66 % (-0.08 %) 91.40 % (+1.01 %) 95.17 % (-0.35 %)

We present additional quantitative evaluation results in Tab. 4 and Tab. 5. Interestingly, across most evaluation dimensions, the base model using 40 inference steps (80 function evaluations) exhibits the poorest performance. For instance, in text-to-video generation, the base model achieves an aesthetic quality score of only 63.62 %, whereas both distilled variants using 4 steps (4 function evaluations) obtain higher scores. Although the Vbench [15] quantitative metrics suggest that DMD2 achieves the best overall performance while the base model performs worst, human preference ratings show the opposite trend. In the attached video, we compile all 220 generated comparison videos for the T2V task. Note that the video has been heavily compressed to reduce file size. As the video clearly demonstrates, the base model exhibits the highest overall quality in terms of aesthetics and motion dynamics, while Phased DMD preserves the base model’s performance, substantially better than DMD2. We argue that the rankings derived from the quantitative evaluations in Tab. 4 and Tab. 5 are not entirely reliable. Nevertheless, the performance gaps between the distilled models and the base model reveal that Phased DMD produces values more closely aligned with those of the base model, indicating that Phased DMD better preserves the generative distribution of the original base model.

The results presented in Tab. 2 are based on the 4-step 2-phase configuration, as illustrated in Fig. 1d. Tab. 6 demonstrates that the 4-step 4-phase configuration (Fig. 1c) delivers better performance in both motion dynamics and visual quality, albeit at the cost of increased system complexity. This improvement can be attributed to the avoidance of SGTS and the inclusion of more trainable parameters.

#### C.2 Ablation on Diffusion Timestep Subintervals

Empirically, we observe that sampling noise injection timesteps using Reverse Nested Intervals t ∼ T (t;tk,1) outperforms Disjoint Intervals t ∼ T (t;tk,tk−1) in terms of generation quality. Fig. 10 illustrates the results of these two methods in the Wan2.2 T2V distillation task. Specifically, sampling t ∼ T (t;tk,1) yields normal color tones and accurate structures, whereas sampling t ∼ T (t;tk,tk−1) results in low-contrast tones and degraded facial structures.

At the beginning of each phase in Phased DMD, there is a substantial gap between the distribution of samples generated by the few-step generator and the distribution of real samples. The generated samples fall outside the domain of the

Table 6 Quantitative comparison of video generation performance. “OF” refers to optical flow. “DD” refers to dynamic degree. Completely removing SGTS from Phased DMD leads to improved performance.

Method OF ↑ DD ↑ FID ↓ FVD ↓ Base model 10.26 79.55 % 0.0 0.0

DMD2 3.23 65.45 % 55.70 763.1 Ours (4-step 2-phase) 9.30 82.27 % 47.24 700.9 Ours (4-step 4-phase) 9.43 83.18 % 45.40 578.2

[Figure 52]

[Figure 53]

(a) Disjoint Intervals (b) Reverse Nested Intervals

- Figure 10 The effect of noise injection intervals. Luo et al. [27] employs disjoint noise injection timestep intervals for different generation steps, where the intervals do not overlap. In contrast, we adopt reverse nested intervals, where the diffusion timestep interval in each phase terminates at 1.0. Integrating disjoint intervals into Phased DMD leads to unnatural colors and deteriorated facial structures, as illustrated on the left. Conversely, adopting reverse nested intervals yields correct results.

teacher model, leading to inaccurate score estimations. This discrepancy is particularly pronounced in the high-SNR (low-noise level) range, where samples are less corrupted by noise. In contrast, in the low-SNR (high-noise level) range, the diffused generated distribution overlaps more significantly with the diffused real distribution, enabling the teacher model to provide more accurate score estimations. Consequently, noise injection at high-noise levels plays a crucial role in DMD training.

To validate this analysis, we perform ablation studies on vanilla DMD for the Wan2.1 T2I task. Specifically, the diffusion timestep t is fixed at 0.357 for one experiment and at 0.882 for another. Wang et al. [43] has proven that DKL(pfake(xt)∥preal(xt)) = 0 ⇔ DKL(pfake(x0)∥preal(x0)) = 0 for any 0 < t < 1. Thus, both experiments are theoretically valid. However, the experiment with a diffusion timestep t = 0.357 fails to converge, as illustrated in Fig. 11, while the experiment with t = 0.882 demonstrates correct results. This controlled experiment highlights that incorporating high-noise levels is essential for effective DMD training. This observation, to some extent, explains our rationale for adopting Reverse Nested Intervals, wherein the training interval at each stage includes the high-noise range.

[Figure 54]

[Figure 55]

(a) t = 0.357 (b) t = 0.882

- Figure 11 The effect of noise injection timestep in DMD training. In DMD training, noise is injected into the generated samples at a low noise level (left) and a high noise level (right). The training fails to converge correctly when noise is injected exclusively at a low noise level.

#### C.3 Additional Discussion on MoE

Mixture-of-Experts (MoE) architectures are widely employed in large language models [2, 46], where they are typically adapted within the feed-forward network (FFN) layers. In diffusion models, however, MoE is implemented differently. Here, each expert typically features a dense architecture and is assigned to a specific denoising range, allowing it to be optimized for a distinct subset of the generative process. This functional division, which aligns with the different requirements across the denoising trajectory, is illustrated in Fig. 12: the low-SNR (high-noise) stage is critical for modeling global structures and dynamics, whereas the high-SNR (low-noise) stage focuses on refining fine-grained details. During inference, these experts are applied sequentially as the SNR increases throughout the sampling process.

The scaling of diffusion models has recently increased interest in MoE architectures. By dedicating experts to different SNR levels, MoE enhances model capacity and generative quality without a proportional increase in inference cost. This performance gain is particularly pronounced in video generation [41], where a dedicated high-noise expert excels

- at capturing coherent temporal dynamics.

Our training framework, Phased DMD , is naturally compatible with such MoE-based models. For a base model with N experts, the optimal practice is to employ an N-phase training scheme. In the k-th phase, the setup comprises one trainable generator and k trainable fake models. Empirical observations indicate that 4-step sampling represents a performance-efficient balance. Consequently, we posit that a base model architecture with four denoising experts is an effective choice. Given the inherent compatibility between diffusion models and MoE architectures, as well as the demonstrated benefits of specialized experts, we argue that MoE will become an increasingly prevalent choice for image and video generation. Correspondingly, Phased DMD will play a more significant role in the distillation of diffusion models, as it is inherently compatible with MoE-based foundations.

[Figure 56]

[Figure 57]

(a) low-SNR only (b) low-SNR + high-SNR

- Figure 12 Visualization of the functional roles of the low-SNR (high-noise) and high-SNR (low-noise) experts. (a) Video sequences generated by the distilled high-noise expert of Wan2.2-T2V-A14B, evaluated over only the first two denoising steps. (b) Video sequences generated by the combined pipeline of the distilled high-noise and low-noise experts, evaluated over all four denoising steps. A comparison of (a) and (b) demonstrates that the low-SNR expert is responsible for modeling global structure and dynamics, whereas the high-SNR expert refines local details.

