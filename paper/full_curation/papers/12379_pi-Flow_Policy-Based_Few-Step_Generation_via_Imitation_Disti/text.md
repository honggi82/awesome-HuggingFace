## PI-FLOW: POLICY-BASED FEW-STEP GENERATION VIA IMITATION DISTILLATION

Hansheng Chen1 Kai Zhang2 Hao Tan2 Leonidas Guibas1 Gordon Wetzstein1 Sai Bi2 1Stanford University 2Adobe Research https://github.com/Lakonik/piFlow

[Figure 1]

[Figure 2]

[Figure 3]

# arXiv:2510.14974v3[cs.LG]19Feb2026

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Figure 1: High quality 4-NFE text-to-image generations by π-Flow, distilled from FLUX.1-12B (top-right three images) and Qwen-Image-20B (all remaining images). π-Flow preserves the teacher’s coherent structures, fine details (e.g., skin and hair), and accurate text rendering, while avoiding diversity collapse (see Fig. 4 for sample diversity).

ABSTRACT

Few-step diffusion or flow-based generative models typically distill a velocitypredicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality–diversity trade-off. To address this, we propose policy-based flow models (π-Flow). π-Flow modifies the output layer of a student flow model to predict a network-free policy at one timestep. The policy then produces dynamic flow velocities at future substeps with negligible overhead, enabling fast and accurate ODE integration on these substeps without extra network evaluations. To match the policy’s ODE trajectory to the teacher’s, we introduce a novel imitation distillation approach, which matches the policy’s velocity to the teacher’s along the policy’s trajectory using a standard ℓ2 flow matching loss. By simply mimicking the teacher’s behavior, π-Flow enables stable and scalable training and avoids the quality–diversity trade-off. On ImageNet 2562, it attains a 1-NFE FID of 2.85, outperforming previous 1-NFE models of the same DiT architecture. On FLUX.112B and Qwen-Image-20B at 4 NFEs, π-Flow achieves substantially better diversity than state-of-the-art DMD models, while maintaining teacher-level quality.

- 1 INTRODUCTION

Diffusion and flow matching models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song & Ermon, 2019; Lipman et al., 2023; Albergo & Vanden-Eijnden, 2023) have become the dominant method for visual generation, delivering compelling image quality and diversity. However, these models rely on a costly denoising process for inference, which integrates a probability flow ODE (Song et al., 2021) over multiple timesteps, each step requiring a neural network evaluation. Commonly, the inference cost of diffusion models is quantified by the number of function (network) evaluations (NFEs).

To reduce the inference cost, diffusion distillation methods compress a pre-trained multi-step model (the teacher) into a student that requires only one or a few network evaluation steps. Existing distillation approaches avoid ODE integration by taking one or a few shortcut steps that map noise to data, where each shortcut path is predicted by the student network, referred to as a shortcut-predicting model. Learning these shortcuts is a significant challenge because they cannot be directly inferred from the teacher model. This necessitates the use of complex training methods, such as progressive distillation (Salimans & Ho, 2022; Liu et al., 2023; 2024), consistency distillation (Song et al., 2023), and distribution matching (Sauer et al., 2024a; Yin et al., 2024b;a; Salimans et al., 2024). In turn, the sophisticated training often lead to degraded image quality from error accumulation or compromised diversity due to mode collapse.

To sidestep the difficulties in shortcut-predicting distillation, we propose a novel policy-based flow model (π-Flow or pi-Flow) paradigm: given noisy data at one timestep, the student network predicts a network-free policy, which maps new noisy states to their corresponding flow velocities with negligible overhead, allowing fast and accurate ODE integration using multiple substeps of policy velocities instead of network evaluations.

To train the student network, we introduce policy-based imitation distillation (π-ID), a DAggerstyle (Ross et al., 2011) on-policy imitation learning (IL) method. π-ID trains the policy on its own trajectory: at visited states, we query the teacher velocity and match the policy’s output to it, using the teacher’s corrective signal to teach the policy to recover from its own mistakes and reduce error accumulation. Specifically, the matching employs a standard ℓ2 loss aligned with the teacher’s flow matching objective, thus naturally preserving its quality and diversity.

We validate our paradigm with two types of policies: a simple dynamic-xˆ(0t) (DX) policy and an advanced GMFlow policy based on Chen et al. (2025). Experiments show that GMFlow policy

outperforms DX policy and delivers strong ImageNet 2562 FIDs at 1- and 2-NFE generation. To demonstrate its scalability, we distill FLUX.1-12B (Black Forest Labs, 2024b) and Qwen-Image20B (Wu et al., 2025) text-to-image models into 4-NFE π-Flow students, which achieve state-ofthe-art diversity, while maintaining teacher-level quality. We summarize the contributions of this work as follows:

- • We propose π-Flow, a new paradigm that decouples ODE integration substeps from network evaluation steps, enabling both fast generation and straightforward distillation.
- • We introduce π-ID, a novel on-policy IL method for few-step π-Flow distillation, which reduces the training objective to a simple ℓ2 flow matching loss.
- • We demonstrate strong performance and scalability of π-Flow, particularly, its superior diversity and teacher alignment compared to other state-of-the-art 4-NFE text-to-image models.

- 2 PRELIMINARIES

In this section, we briefly introduce flow matching models (Lipman et al., 2023; Liu et al., 2023) and the notations used in this paper.

Let p(x0) denote the (latent) data probability density, where x0 ∈ RD is a data point. A standard flow model defines an interpolation between a data sample and a random Gaussian noise

ϵ ∼ N(0,I), yielding the diffused noisy data xt = αtx0 + σtϵ, where t ∈ (0,1] denotes the diffusion time, and αt = 1 − t,σt = t are the linear flow noise schedule. The optimal transport map across all marginal densities p(xt) = R

#### D N(xt;αtx0,σt2I)p(x0)dx0 can be described by

###### (a) Standard flow model (b) Shortcut-predicting model

###### (c) Policy-based model

- =
- • Many network steps
- • Many integration steps

- • Few network steps
- • Few integration steps

- ≪
- • Few network steps
- • Many integration substeps

=

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

𝒙𝒙𝑡𝑡src

𝒙𝒙𝑡𝑡src

𝒙𝒙𝑡𝑡src

[Figure 16]

[Figure 17]

𝜋𝜋

[Figure 18]

[Figure 19]

𝜋𝜋

𝜋𝜋 𝜋𝜋

𝒙𝒙𝑡𝑡dst

𝒙𝒙𝑡𝑡dst

𝒙𝒙𝑡𝑡dst

Policy

𝜋𝜋 𝒙𝒙𝑡𝑡,𝑡𝑡

Figure 2: Comparison between (a) standard flow model (teacher), (b) shortcut-predicting model, and (c) our policy-based model. The shortcut-predicting model skips all intermediate states, whereas the our-based model retains all intermediate substeps with minimal overhead.

the following probability flow ODE (Song et al., 2021; Liu, 2022):

xt − Ex

xt − RD

x0p(x0|xt)dx0

0∼p(x0|xt)[x0] t

dxt dt

, (1)

= x˙t =

=

t

(xt;αtx0,σt2I)p(x0)

with the denoising posterior p(x0|xt) := N

p(xt) . At test time, the model can generate samples by first initializing the noise x1 ← ϵ and then solving the ODE to obtain limt→0 xt. In practice, flow matching models approximate the ODE velocity dxt

dt using a neural network Gθ(xt,t) with learnable parameters θ, trained using the ℓ2 flow matching loss:

xt − x0 t

- 1

- 2∥u − Gθ(xt,t)∥2 , with sample velocity u :=

. (2)

Lθ = Et,x

0,xt

Since each velocity query requires evaluating the network (Fig. 2 (a)), flow matching models couple sampling efficiency with solver precision. Despite the progress in advanced solvers (Karras et al., 2022; Zhang & Chen, 2023; Lu et al., 2022; 2023; Zhao et al., 2023), high-quality sampling typically requires over 10 steps due to inherent ODE truncation error, making it computationally expensive.

- 3 π-FLOW: POLICY-BASED FEW-STEP GENERATION

In π-Flow, we define the policy as a network-free function π: RD × R → RD that maps a state (xt,t) to a flow velocity. A policy can be network-free if it only needs to describe a single ODE trajectory, which is fully determined by its initial state (xt

,tsrc) with tsrc ≥ t. In this case, the policy for each trajectory must be dynamically predicted by a neural network conditioned on that initial state (xt

src

,tsrc). We therefore adapt a flow model to output not a single velocity, but an entire dynamic policy that governs the full trajectory. Formally, define the policy function space F :=

src

π: RD × R → RD . Then, our goal is to distill a policy generator network Gϕ: RD × R → F with learnable parameters ϕ, such that π(xt,t) = Gϕ(xt

,tsrc)(xt,t).

src

As shown in Fig. 2 (c), π-Flow performs ODE-based denoising from tsrc to tdst via two stages:

- • A policy generation step, which feeds the initial state (xt

src

,tsrc) to the student network Gϕ to produce the policy π, i.e., π ← Gϕ(xt

src

,tsrc).

- • Multiple policy integration substeps, which integrates the ODE by querying the policy velocity

#### + t tdst

over multiple substeps, obtaining a less noisy state by xt

π(xt,t)dt.

dst ← xt

src

src

Unlike previous few-step distillation methods, π-Flow decouples network evaluation steps from ODE integration substeps. This allows it to combine the key advantages of two paradigms: it performs only a few network evaluations for efficient generation, similar to a shortcut-predicting model, while also executing dense integration substeps, just like a standard flow matching teacher. Thanks to its teacher-like ODE integration process, a π-Flow student offers unprecedented advantage in training, as we can now follow well-established imitation learning (IL) approaches to directly match the policy velocity π(xt,t) to the teacher velocity Gθ(xt,t), as discussed later in § 4.

To identify the appropriate function classes of student policies for fast image generation, we need to consider the following requirements:

- • Efficiency. The policy should provide closed-form velocities with minimal overhead, so that rolling out dense (e.g., 100+) substeps incurs negligible cost compared to a network evaluation.

- • Compatibility. The policy should have a compact set of parameters that can be easily predicted by the student Gϕ with standard backbones (e.g., DiT (Peebles & Xie, 2023)).
- • Expressiveness. The policy should be able to approximate a complicated ODE trajectory starting

from a certain initial state xt

src

.

- • Robustness. The policy should be able to handle trajectory variations that arise from perturba-

tions to the initial state xt

. For instance, a suboptimal student network will produce an erroneous mapping from xt

src

to π. This introduces the randomness that the policy needs to accommodate throughout the rollout. Consequently, the policy function should adapt its velocity output to variations in its state input xt, which is a challenging requirement for network-free functions.

src

- 3.1 DYNAMIC-xˆ(0t) POLICY

We introduce a simple baseline policy called dynamic-xˆ(0t) policy (DX policy). DX policy defines π(xt,t) := xt−xˆ

(t) 0

t , where xˆ(0t) approximates the posterior moment Ex

0∼p(x0|xt)[x0] in Eq. (1). Along a fixed trajectory starting from an initial state (xt

src

,tsrc), the posterior moment is only depen-

dent on t. Therefore, we first predict a grid of xˆ(0ti) at N evenly spaced times t1,...,tN ∈ [tdst,tsrc] by a single evaluation of the student network Gϕ(xt

src

,tsrc). This is achieved by expanding the output channels of the student network and performing u-to-x0 reparameterization. Then, for arbitrary t ∈ [tdst,tsrc], we obtain the approximated moment xˆ(0t) by a linear interpolation over the grid.

Apparently, DX policy is fast, compatible, and expressive enough so that any N-step teacher trajectory can be matched with N grid points. However, its robustness is limited because xˆ(0t) is not adaptive to perturbations in xt.

- 3.2 GMFLOW POLICY

For stronger robustness, we incorporate an advanced GMFlow policy based on the closedform GM velocity field in Chen et al. (2025). GMFlow policy expands the network output channels to predict a factorized Gaussian mixture (GM) velocity distribution q(u|xt

) =

src

K k=1 AikN ui;µik,s2I , where Aik ∈ R+, µik ∈ RC, s ∈ R+ are GM parameters predicted by the network, L × C factorizes the data dimension D into sequence length L and channel size C, and K is a hyperparameter specifying the number of mixture components. Intuitively, the student network Gϕ maps the initial state xt

L i=1

to multiple denoising modes that parameterize the GMFlow policy. The policy then enables a closed-form velocity expression at future state (xt,t) for any 0 < t < tsrc (see § F for details). The speed and compatibility of GMFlow has already been discussed in Chen et al. (2025), thus we focus on analyzing its expresiveness and robustness.

src

Expressiveness. With the L × C factorization, each individual C-dimensional GM needs to be expressive enough to approximate a C-dimensional chunk of the teacher trajectory. In § E, we rigorously prove the following theorem, demonstrating GMFlow’s expressiveness.

Theorem 1 (A GMFlow policy with K = N ·C can accurately approximate any N-step trajectory). Given pairwise distinct times t1,...,tN ∈ (0,1] and vectors xt

n ∈ RC for n = 1,...,N, there exists a GM parameterization of p(x0) with N · C components, such that x˙t

, x˙t

n

can be approxi-

n

mated arbitrarily well using Eq (1) at t = tn for every n = 1,...,N. In practice, we can use K ≪ N · C (e.g., K = 8) since the teacher trajectory is mostly smooth. More analysis of GMFlow hyperparameters are presented in § C.1.

Robustness. GMFlow is highly robust against trajectory perturbation due to its probabilistic origin. Unlike DX policy, GMFlow models a fully dynamic denoising posterior (Eq. (23)) dependent on both xt and t. Leveraging its robustness, the policy can be flexibly altered via GM dropout in training (§ 4) and GM temperature in inference (§ B.1), both improving generalization performance.

𝒙𝒙𝑡𝑡src

Algorithm 1: On-policy π-ID.

Input: NFE, teacher Gθ, student Gϕ, condition c

𝐺𝐺𝝓𝝓

𝒙𝒙𝑡𝑡dst

- 1 Sample tsrc from NFE 1 , NFE2 , · · · , 1

- 2 Initialize xtsrc (data-free or data-dependent)
- 3 π ← Gϕ(xtsrc, tsrc, c)
- 4 πD ← stopgrad(π)
- 5 Lϕ ← 0
- 6 for finite samples t ∼ U tsrc − NFE1 , tsrc do

- 7 xt ← xtsrc + t t

src

πD(xt, t) dt

- 8 Lϕ ← Lϕ + 12∥Gθ(xt, t, c) − π(xt, t)∥2

- 9 ϕ ← Adam(ϕ, ∇ϕLϕ) // optimizer step

Policy 𝜋𝜋

stopgrad (+ dropout)

Detached policy 𝜋𝜋D rollout

Teacher velocity 𝐺𝐺𝜽𝜽(𝒙𝒙𝑡𝑡,𝑡𝑡) Policy velocity 𝜋𝜋(𝒙𝒙𝑡𝑡,𝑡𝑡)

Detached policy 𝜋𝜋D

Figure 3: On-policy flow imitation distillation. Intermediate states are sampled along the detached policy rollout, where the loss matches the policy to the teacher.

- 4 π-ID: POLICY-BASED IMITATION DISTILLATION

With the policy rollout sharing the same format as the teacher’s ODE integration, it is straightforward to adopt imitation learning to learn the policy by directly matching the policy’s velocity to the teacher’s velocity. In this section, we introduce a simple policy-based imitation distillation (π-ID) algorithm based on DAgger-style (Ross et al., 2011) on-policy imitation.

On-policy imitation learning is robust to error accumulation since it trains the policy on its own trajectory, allowing the teacher’s corrective signal to steer a deviating trajectory back on track. As shown in Fig. 3 and Algorithm 1, for a time interval from tsrc to tdst (i.e., a 1-NFE segment), we first feed the initial state (xt

src

,tsrc) to the student network Gϕ to obtain the policy π. We then sample an intermediate time t ∈ (tdst,tsrc] and roll out a detached policy πD from tsrc to t using highaccuracy ODE integration (with a small step size of 1/128), yielding an intermediate state xt on the policy trajectory. This state is fed to both the learner policy π and the frozen teacher Gθ, which produce their respective velocities. Finally, we compute a standard ℓ2 flow matching loss between the two velocities, and backpropagate its gradients through the policy π to the student network Gϕ. Because the student forward/backward pass dominates compute while policy and teacher queries are relatively cheap, we may repeat the rollout-and-matching step multiple times for additional teacher supervisions. In practice, we sample two intermediate states per student forward pass.

Data-dependent and data-free π-ID. The initial state xt

src

can be obtained via forward diffusion from real data x0 (data-dependent Algorithm 2), or via π-Flow’s reverse denoising from random noise x1 (data-free Algorithm 3). Both methods have roughly the same computational cost, and comparable performance, as demonstrated in the experiments (§ 5).

Error bounds and convergence. As discussed by Ross et al. (2011), on-policy imitation learning guarantees that the performance of the learned policy is bounded by the teacher’s performance plus an error term that scales as O(nε), where n is the number of substeps and ε is the average imitation error per step, which is strictly better than the O(n2ε) compounding-error behavior of off-policy behavior cloning. Moreover, the sequence of on-policy iterates converges in performance to the best policy in the function class, under the student’s capacity constraint.

- 5 EXPERIMENTS

To demonstrate the versatility of π-Flow, we evaluate it with three distinct image generation models of different scales and architectures: DiT(SiT)-XL/2 (675M) (Peebles & Xie, 2023; Ma et al., 2024; Vaswani et al., 2017) for ImageNet 2562 (Deng et al., 2009) class-conditioned generation, FLUX.1-12B (Black Forest Labs, 2024b) and Qwen-Image-20B (Wu et al., 2025) for text-to-image generation.

- 5.1 IMPLEMENTATION DETAILS

In this subsection, we discuss key implementation details essential to model performance. More training details and hyperparameter choices are presented in § C.

Table 1: 1-NFE generation results of π-Flow with DX and GMFlow policies on ImageNet. Tested after 40K training iterations. FM stands for standard flow matching.

Policy Teacher FID↓ IS↑ Precision↑ Recall↑

DX (N = 10) REPA 4.73 327.6 0.781 0.514 DX (N = 20) REPA 4.44 329.8 0.786 0.531 DX (N = 40) REPA 4.90 321.8 0.778 0.537 GM (K = 8) REPA 3.07 336.9 0.789 0.572 GM (K = 32) REPA 3.08 341.7 0.791 0.562

GM (K = 32) FM 3.65 282.0 0.797 0.533 GM (K = 32) w/o dropout FM 4.14 279.6 0.799 0.525

Table 2: Comparison with previous few-step DiTs on ImageNet.

Model NFE FID↓ iCT 2 20.30 iMM 1×2 7.77 MeanFlow 2 2.20 FACM (REPA) 2 1.52 π-Flow (GM-REPA) 2 1.97

iCT 1 34.24 Shortcut 1 10.60 MeanFlow 1 3.43 π-Flow (GM-FM) 1 3.34 π-Flow (GM-REPA) 1 2.85

GM dropout. Dropout is a widely adopted technique in supervised/imitation learning and reinforcement learning to improve generalization (Srivastava et al., 2014; Cobbe et al., 2019). For the GMFlow policy, we introduce GM dropout in training to stochastically perturb and diversify π-ID rollouts to make the policy more robust to potential trajectory variations. Given the GM mixture weights Aik of the detached policy πD, we sample a binary mask for each component k = 1,··· ,K and multiply it into Aik synchronously across all i = 1,··· ,L. The masked weights are then renormalized and used for the detached rollout. By exploring alternative GM modes, this simple technique improves the policy’s robustness, yielding better FID on ImageNet 2562 (§ 5.2).

Handling FLUX.1 dev teacher. On-policy imitation learning assumes the teacher is robust to outof-distribution (OOD) intermediate states and can steer trajectories back on track. This generally holds for standard flow models with classifier-free guidance (CFG) (Ho & Salimans, 2021), which exhibit error-correction behavior (Chidambaram et al., 2024). However, FLUX.1 dev (Black Forest Labs, 2024b) is a guidance-distilled model without true CFG and is less robust to OOD inputs. To mitigate OOD exposure, we adopt a scheduled trajectory mixing strategy, which rolls out the trajectory using a mixture of teacher and student with a linearly decaying teacher ratio (see § B.2 for details).

- 5.2 IMAGENET DIT

Our study utilizes two pretrained teachers with the same DiT architecture: a standard flow matching (FM) DiT (the baseline in Chen et al. (2025)), and the REPA DiT (Yu et al., 2025). Interval CFG (Kynk¨a¨anniemi et al., 2024) is applied to both teachers to maximize their performance. Each π-Flow student is initialized with the teacher weights and then fully finetuned using the π-ID loss.

Evaluation metrics. We adopt the standard evaluation protocol in ADM (Dhariwal & Nichol, 2021) with the following metrics: Fr´echet Inception Distance (FID) (Heusel et al., 2017), Inception Score (IS), and Precision–Recall (Kynk¨a¨anniemi et al., 2019).

Comparison of DX and GMFlow policies. As shown in Table 1, both policies yield strong 1NFE FIDs after 40k training iterations, with the GMFlow policy consistently outperforming the DX policy by a clear margin. Notably, the DX policy exhibits sensitivity to the hyperparameter N (number of grid points), whereas the GMFlow policy produces consistent results across different values of K (number of Gaussians).

Comparison with prior few-step DiTs. In Table 2, we compare π-Flow (GM policy with K = 32) to prior few-step DiTs on ImageNet 2562: iCT (Song & Dhariwal, 2024), Shortcut models (Frans

- et al., 2025), iMM (Zhou et al., 2025), MeanFlow (Geng et al., 2025), and the concurrent work FACM (Peng et al., 2025). FACM (distilled from REPA) improves MeanFlow with an auxiliary loss and attains a leading 2-NFE FID, though it still relies on the inefficient JVP operation. In contrast, π-Flow uses a minimal training framework with no JVP and adaptive loss scalings, yet still outperforms the original MeanFlow DiT across both 1-NFE and 2-NFE generation.

Ablation study on GM dropout. From the two bottom rows in Table 1 we conclude that our standard implementation with a 0.05 GM dropout rate yields better FID and Recall compared to the setting without dropout, confirming the effectiveness of our GM dropout technique.

Table 3: Quantitative comparisons on COCO-10k dataset and HPSv2 prompt set.

COCO-10k prompts HPSv2 prompts

Model Distill method NFE

###### Data align. Prompt align. Pref. align. Teacher align. Prompt align. Pref. align. FID↓ pFID↓ CLIP↑ VQA↑ HPSv2.1↑ FID↓ pFID↓ CLIP↑ VQA↑ HPSv2.1↑

FLUX.1 dev - 50 27.8 34.9 0.268 0.900 0.309 - - 0.284 0.805 0.314 FLUX Turbo GAN 8 26.7 32.0 0.267 0.900 0.308 13.8 18.5 0.286 0.814 0.313 Hyper-FLUX CD+Re 8 29.8 33.3 0.268 0.894 0.309 15.6 22.2 0.285 0.807 0.315 π-Flow (GM-FLUX) π-ID 8 29.0 35.4 0.268 0.901 0.311 12.6 15.9 0.285 0.810 0.316 SenseFlow (FLUX) VSD+CD+GAN 4 34.1 44.2 0.266 0.879 0.308 23.3 28.2 0.283 0.806 0.318 π-Flow (GM-FLUX) π-ID 4 29.8 36.1 0.269 0.903 0.308 14.3 19.2 0.288 0.816 0.313 π-Flow (GM-FLUX) π-ID (data-free) 4 29.7 36.2 0.269 0.905 0.310 14.4 19.7 0.287 0.813 0.314 Qwen-Image - 50×2 34.1 45.6 0.282 0.936 0.312 - - 0.302 0.872 0.309 Qwen-Image Lightning VSD 4 37.5 51.6 0.280 0.935 0.322 15.6 19.7 0.299 0.867 0.328 π-Flow (GM-Qwen) π-ID 4 36.0 46.1 0.281 0.934 0.314 12.8 16.6 0.300 0.860 0.310 π-Flow (GM-Qwen) π-ID (data-free) 4 36.0 45.7 0.282 0.936 0.315 12.9 16.8 0.301 0.862 0.312

Table 4: Quantitative comparisons on OneIG-Bench (Chang et al., 2025).

Model Distill Method NFE Alignment↑ Text↑ Diversity↑ Style↑ Reasoning↑

FLUX.1 dev - 50 0.790 0.556 0.238 0.370 0.257 FLUX Turbo GAN 8 0.791 0.334 0.234 0.370 0.239 Hyper-FLUX CD+Re 8 0.790 0.530 0.198 0.369 0.254 π-Flow (GM-FLUX) π-ID 8 0.792 0.517 0.234 0.369 0.256 SenseFlow (FLUX) VSD+CD+GAN 4 0.776 0.384 0.151 0.343 0.238 π-Flow (GM-FLUX) π-ID 4 0.799 0.437 0.229 0.360 0.251 π-Flow (GM-FLUX) π-ID (data-free) 4 0.799 0.460 0.224 0.363 0.249 Qwen-Image - 50×2 0.880 0.888 0.194 0.427 0.306 Qwen-Image Lightning VSD 4 0.885 0.923 0.116 0.417 0.311 π-Flow (GM-Qwen) π-ID 4 0.875 0.892 0.180 0.434 0.298 π-Flow (GM-Qwen) π-ID (data-free) 4 0.881 0.890 0.176 0.433 0.300

- 5.3 FLUX.1-12B AND QWEN-IMAGE-20B

For text-to-image generation, we distill the 12B FLUX.1 dev (Black Forest Labs, 2024b) and 20B Qwen-Image (Wu et al., 2025) models into π-Flow students. During student training, we freeze the base parameters inherited from the teacher and finetune only the expanded output layer along with 256-rank LoRA adapters (Hu et al., 2022) on the feed-forward layers. For data-dependent distillation, we prepare 2.3M one-megapixel (1MP) images captioned with Qwen2.5-VL (Bai et al., 2025). In the data-free setting, we use only the generated captions as conditioning inputs while keeping the same 1MP resolution when initializing the noise.

Evaluation protocol. We conduct a comprehensive evaluation on 10242 high-resolution image generation from three distinct prompt sets: (a) 10K captions from the COCO 2014 validation set (Lin et al., 2014), (b) 3200 prompts from the HPSv2 benchmark (Wu et al., 2023), and (c) 1120 prompts from OneIG-Bench (Chang et al., 2025). For the COCO and HPSv2 sets, we report common metrics including FID (Heusel et al., 2017), patch FID (pFID) (Lin et al., 2024a), CLIP similarity (Radford et al., 2021), VQAScore (Lin et al., 2024b), and HPSv2.1 (Wu et al., 2023). On COCO prompts, FIDs are computed against real images, reflecting data alignment. On HPSv2, FIDs are computed against the 50-step teacher generations, reflecting teacher alignment. CLIP and VQAScore measure prompt alignment, while HPSv2 captures human preference alignment. For OneIG-Bench, we adopt its official evaluation protocol and metrics. All quantitative results are presented in Table 3 and 4.

Competitor models. We compare π-Flow against other few-step student models distilled from the same teacher. For FLUX, we compare against: 4-NFE SenseFlow (Ge et al., 2026), primarily leveraging variational score distillation (VSD) (Wang et al., 2023), also known as distribution matching distillation (DMD) (Yin et al., 2024b); 8-NFE Hyper-FLUX (Ren et al., 2024), trained with consistency distillation (CD) (Song et al., 2023) and reward models (Re) (Xu et al., 2023); 8-NFE FLUX Turbo, based on GAN-like adversarial distillation (Goodfellow et al., 2014; Sauer et al., 2024b). For Qwen-Image, we compare with the 4-NFE Qwen-Image Lighting based on VSD (ModelTC, 2025). Note that the 4-NFE FLUX.1 schnell is distilled from the closed-source FLUX.1 pro instead of the

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

With a clear box hovering above showing "System Update: Now supports gardening mode!", generate pixel art of a cheerful robot watering pixelated flowers in a futuristic garden.

𝝅𝝅-Flow (GM-FLUX) 4-NFE FLUX.1 dev 50-NFE

SenseFlow (FLUX) 4-NFE

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

𝝅𝝅-Flow (GM-Qwen) 4-NFE Qwen-Image 50x2-NFE

Qwen-Image Lightning 4-NFE

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Young woman in dynamic neon streetwear poses on bustling Tokyo street at night, surrounded by illuminated signs. Captured in a photograph that features lifelike individuals.

𝝅𝝅-Flow (GM-FLUX) 4-NFE FLUX.1 dev 50-NFE

SenseFlow (FLUX) 4-NFE

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

𝝅𝝅-Flow (GM-Qwen) 4-NFE Qwen-Image 50x2-NFE

Qwen-Image Lightning 4-NFE

The image should capture a lifelike essence, showcasing a young woman gracefully positioned in a golf swing on a bright summer day. Her long hair effortlessly streams behind her, complementing her ideal physique, which is accentuated by form-fitting red leggings adorned with striking geometric patterns. The scene is bathed in vibrant sunlight, creating dynamic shadows as she elegantly executes

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

𝝅𝝅-Flow (GM-FLUX) 4-NFE FLUX.1 dev 50-NFE SenseFlow (FLUX) 4-NFE

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

𝝅𝝅-Flow (GM-Qwen) 4-NFE Qwen-Image 50x2-NFE Qwen-Image Lightning 4-NFE her swing with finesse.

- Figure 4: Images generated from the same batch of initial noise by π-Flows, teachers, and VSD students (SenseFlow, Qwen-Image Lightning). π-Flow models produce diverse structures that closely mirror the teacher’s. In contrast, VSD students tend to repeat the same structure. Notably, SenseFlow mostly generates symmetric images.

publicly available FLUX.1 dev (Black Forest Labs, 2024a), so we do not compare with it directly, but include further discussion in § D.

Strong all-around performance. As shown in Table 3 and Table 4, π-Flow demonstrates strong allaround performance, outperforming other few-step students on roughly 70% of all metrics, without exhibiting obvious weaknesses in any specific area.

Superior diversity and teacher alignment. π-Flow consistently achieves the highest diversity scores and the best teacher-referenced FIDs by clear margins, especially in the 4-NFE setting. These results strongly suggest that π-Flow effectively avoids both diversity collapse and style drift. As a result, most of its scores closely match those of the teacher, with some even slightly surpassing the teacher scores (e.g., prompt alignment and several Qwen-Image OneIG scores). Its strong teacher alignment is also evident in Fig. 4, where π-Flow generates structurally similar images to the teacher’s from the same initial noise.

Comparison with VSD (DMD) students. VSD models are notable for high visual quality, sometimes surpassing teachers in quality and preference metrics. However, they are widely known to

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

𝝅𝝅-Flow (GM-FLUX) 8-NFE FLUX Turbo 8-NFE

- Figure 5: Images generated from the same initial noise by π-Flow and FLUX Turbo. π-Flow renders coherent texts, whereas FLUX Turbo underperforms in text rendering.

[Figure 62]

𝝅𝝅-Flow (GM-FLUX) 8-NFE

| |
|---|

Hyper-FLUX 8-NFE

|| |
|---|
<br><br>[Figure 63]|
|---|

Aoshima's masterpiece depicts a forest illuminated by morning light.

A painting depicting a scenic view of Guangzhou, China as a tourist destination by David Inshaw.

[Figure 64]

|[Figure 65]|
|---|

[Figure 66]

[Figure 67]

|[Figure 68]|
|---|

|[Figure 69]|
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 6: Images generated from the same initial noise by π-Flow and Hyper-FLUX. π-Flow produces notably finer details, as highlighted in the zoomed-in patches.

suffer from mode collapse, as reflected in our experiments: both SenseFlow and Qwen-Image Lightning show significant drops in diversity and FIDs. Visual examples in Fig. 4 further highlight the collapse, where different initial noises produce visually similar images with only minor variations. In contrast, π-Flow maintains high quality and diversity without sacrificing either aspect.

Comparison with other students. FLUX Turbo achieves better data alignment FIDs than the teacher due to GAN training, yet its text rendering performance is significantly weaker, as shown in Fig. 5. Meanwhile, Hyper-FLUX often produces undesirable texture artifacts and fuzzy details, whereas π-Flow achieves superior detail rendering, as shown in Fig. 6.

Data-dependent vs. data-free. As shown in Table 3 and Table 4, data-dependent and data-free π-Flow models achieve nearly identical results. This demonstrates the practicality of π-Flow in scenarios where high-quality data is unavailable.

GMFlow vs. DX policy. Consistent with prior ImageNet findings, the DX policy slightly underperforms comapred to the GMFlow policy (Table 5), highlighting the latter’s superior robustness.

Convergence. Figure 7 illustrates the convergence of π-Flow (GM-Qwen) over training iterations. Both FID and Patch FID scores initially improve rapidly, outperforming Qwen-Image Lightning within the first 400 iterations, and continue to improve steadily thereafter. This contrasts with previous GAN or VSD-based methods that often require frequent checkpointing and cherry-picking (Ge

- et al., 2026), demonstrating the scalability and robustness of our approach.

Inference time. To validate that the policies are indeed fast enough so that the overhead is negligible compared to shortcut-predicting models, we compare the inference times of 4-NFE π-Flow models and Qwen-Image Lightning in Table 6. For π-Flow, each policy generation step (with one network evaluation) is followed by 32 policy integration substeps on average. The results in Table 6 show that 32 policy substeps cost around 15 ms in total, which is only 3% of the network time. Therefore, the overall speed of π-Flow is on par with shortcut-predicting models.

- Table 5: Comparisons between DX and GMFlow policies on text-to-image generation.

Policy Teacher

HPSv2 prompts OneIG-Bench

FID↓ pFID↓ HPSv2.1↑ Text↑ Diversity↑

DX (N = 10) FLUX 14.9 20.9 0.313 0.397 0.225 GM (K = 8) FLUX 14.3 19.2 0.313 0.437 0.229

DX (N = 10) Qwen-Image 12.7 17.0 0.306 0.869 0.185 GM (K = 8) Qwen-Image 12.8 16.6 0.310 0.892 0.180

- Table 6: Per-NFE inference time of π-Flow models and the shortcut-predicting model (Qwen-Image Lightning). Tested on an A100 GPU with 12 CPU cores (3.0 GHz).

Model Network time (sec) Policy time (sec) Qwen-Image Lightning 0.465 -

π-Flow (DX-Qwen) 0.464 0.015 π-Flow (GM-Qwen) 0.465 0.014

| |
|---|

FID

20

Qwen Lightning

Patch FID

| |
|---|

18

Score

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

| |
|---|

16

Qwen Lightning

14

0 2K 4K 6K 8K Iteration

Figure 7: Teacher-referenced FID and Patch FID of GM-Qwen evaluated on HPSv2 prompts.

- 6 RELATED WORK

Prior work on diffusion model distillation primarily focuses on predicting shortcuts towards less noisy states, with training objectives ranging from direct regression to distribution matching.

Early work (Luhman & Luhman, 2021) directly regresses the teacher’s ODE integral in a single step, but suffers from degraded quality, since regressing x0 with an ℓ2 loss tends to produce blurry results. Progressive distillation methods (Salimans & Ho, 2022; Liu et al., 2023; 2024; Frans et al., 2025) make further improvements via a multi-stage process that progressively increases the student’s step size and reduces its NFE by regressing the previous stage’s multi-step outputs with fewer steps, yet this introduces error accumulation.

Consistency-based models (Song et al., 2023; Gu et al., 2023; Kim et al., 2024; Song & Dhariwal, 2024; Geng et al., 2025; Boffi et al., 2025) implicitly impose a velocity-based regression loss, which improves quality compared to x-based regression. However, the flow velocity of a shortcutpredicting student must be constructed implicitly using either inaccurate finite differences or expensive Jacobian–vector products (JVPs). Moreover, their quality is still limited due to accumulation of velocity errors into the integrated shortcut. Therefore, in practice, consistency distillation is often augmented with additional objectives to improve quality (Ren et al., 2024; Zheng et al., 2025), further complicating training.

Conversely, distribution matching approaches (Yin et al., 2024b;a; Sauer et al., 2024b; Zhou et al., 2024; Luo et al., 2024; Salimans et al., 2024; Zhou et al., 2025) adopt score matching and adversarial training to align the student’s output distribution with the teacher’s. The VSD objective achieves superior quality but risks diversity loss due to mode collapse; GAN and SiD objectives balance quality and diversity but can cause style drift. Their common reliance on auxiliary networks introduces additional tuning complexity and may lead to stability issues at scale (Ge et al., 2026).

- 7 CONCLUSION

We introduced policy-based flow models (π-Flow), a novel framework for few-step generation in which the network outputs a fast policy that enables accurate ODE integration via dense substeps reach the denoised state. To distill π-Flow models, we proposed a simple on-policy imitation learning approach that reduces the training objective to a single ℓ2 loss, mitigating error accumulation and quality–diversity trade-offs. Extensive experiments distilling ImageNet DiT, FLUX.1-12B, and Qwen-Image-20B models show that few-step π-Flows consistently attain teacher-level image quality while significantly outperforming competitors in diversity and teacher alignment. π-Flow offers a scalable, principled paradigm for efficient, high-quality generation and opens new directions for future research, such as exploring more robust policy families, improved distillation objectives, and extensions to other applications (e.g., video generation).

Reproducibility statement. To facilitate reproduction, we describe the detailed training procedures in Algorithms 2 and 3, and list all important hyperparameters in §C.

Acknowledgements. This project was partially done while Hansheng Chen was supported by the Qualcomm Innovation Fellowship and partially done while Hansheng Chen was an intern at Adobe Research. We would like to thank Jianming Zhang and Hailin Jin for their great support throughout the internship, and Xingtong Ge for the help in evaluating SenseFlow.

REFERENCES

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In ICLR, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In NeurIPS, NIPS’15, pp. 1171–1179, Cambridge, MA, USA, 2015. MIT Press.

Black Forest Labs. Flux.1 [schnell]. https://huggingface.co/spaces/

black-forest-labs/FLUX.1-schnell, 2024a. Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024b. Nicholas Matthew Boffi, Michael Samuel Albergo, and Eric Vanden-Eijnden. Flow map matching

with stochastic interpolants: A mathematical framework for consistency models. TMLR, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=cqDH0e6ak2.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. In NeurIPS, 2025.

Hansheng Chen, Kai Zhang, Hao Tan, Zexiang Xu, Fujun Luan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. Gaussian mixture flow matching models. In ICML, 2025.

Muthu Chidambaram, Khashayar Gatmiry, Sitan Chen, Holden Lee, and Jianfeng Lu. What does guidance do? a fine-grained analysis in a simple setting. In NeurIPS, 2024. URL https: //openreview.net/forum?id=AdS3H8SaPi.

Karl Cobbe, Oleg Klimov, Chris Hesse, Taehoon Kim, and John Schulman. Quantifying generalization in reinforcement learning. In ICML, 2019.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Tim Dettmers, Mike Lewis, Sam Shleifer, and Luke Zettlemoyer. 8-bit optimizers via block-wise quantization. In ICLR, 2022.

Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat GANs on image synthesis. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), NeurIPS, 2021. URL https://openreview.net/forum?id=AAWuCvzaVt.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In ICLR, 2025. URL https://openreview.net/forum?id=OlzB6LnXcS.

Xingtong Ge, Xin Zhang, Tongda Xu, Yi Zhang, Xinjie Zhang, Yan Wang, and Jun Zhang. Senseflow: Scaling distribution matching for flow-based text-to-image distillation. In ICLR, 2026.

Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In NeurIPS, 2025.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), NeurIPS, volume 27. Curran Associates, Inc., 2014. URL https://proceedings.neurips.cc/paper_files/ paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf.

Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Lingjie Liu, and Joshua M Susskind. Boot: Data-free distillation of denoising diffusion models with bootstrapping. In ICML Workshop, 2023.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS Workshop, 2021. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS,

2020.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In ICLR, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In NeurIPS, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In CVPR, 2024.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, and Stefano Ermon. Consistency trajectory models: Learning probability flow ODE trajectory of diffusion. In ICLR, 2024. URL https://openreview.net/ forum?id=ymjI8feDTD.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2014. Tuomas Kynk¨a¨anniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved

precision and recall metric for assessing generative models. In NeurIPS, 2019.

Tuomas Kynk¨a¨anniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. In NeurIPS, 2024.

Shanchuan Lin, Anran Wang, and Xiao Yang. Sdxl-lightning: Progressive adversarial diffusion distillation, 2024a. URL https://arxiv.org/abs/2402.13929.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C. Lawrence Zitnick. Microsoft coco: Common objects in context. In David Fleet, Tomas Pajdla, Bernt Schiele, and Tinne Tuytelaars (eds.), ECCV, pp. 740–755, Cham, 2014. Springer International Publishing. ISBN 978-3-319-10602-1.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. In ECCV, 2024b.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023. URL https://openreview.net/ forum?id=PqvMRDCJT9t.

Qiang Liu. Rectified flow: A marginal preserving approach to optimal transport, 2022. URL https://arxiv.org/abs/2209.14577.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, and qiang liu. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In ICLR, 2024. URL https:// openreview.net/forum?id=1k4yZbbDqX.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), NeurIPS, 2022. URL https: //openreview.net/forum?id=2uAaGwlP_V.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models, 2023. URL https://arxiv. org/abs/2211.01095.

Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed, 2021. URL https://arxiv.org/abs/2101.02388.

Weijian Luo, Zemin Huang, Zhengyang Geng, J. Zico Kolter, and Guo jun Qi. One-step diffusion distillation through score implicit matching. In NeurIPS, 2024.

Nanye Ma, Mark Goldstein, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In ECCV, 2024.

ModelTC. Qwen-image-lightning. https://github.com/ModelTC/

Qwen-Image-Lightning, 2025. William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. Yansong Peng, Kai Zhu, Yu Liu, Pingyu Wu, Hebei Li, Xiaoyan Sun, and Feng Wu. Flow-anchored

consistency models, 2025. URL https://arxiv.org/abs/2507.03738.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, pp. 8748–8763, 2021.

Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, XING WANG, and Xuefeng Xiao. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS,

2024. URL https://openreview.net/forum?id=O5XbOoi0x3.

Hans Richter. Parameterfreie absch¨atzung und realisierung von erwartungswerten. Bl¨atter der DGVFM, 3(2):147–162, Apr 1957. ISSN 1864-0303. doi: 10.1007/BF02808864. URL https://doi.org/10.1007/BF02808864.

Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Geoffrey Gordon, David Dunson, and Miroslav Dud´ık (eds.), AISTATS, volume 15 of Proceedings of Machine Learning Research, pp. 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr. press/v15/ross11a.html.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. In ICLR, 2022.

Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. NeurIPS, 37:36046–36070, 2024.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, SA ’24, New York, NY, USA, 2024a. Association for Computing Machinery. ISBN 9798400711312. doi: 10.1145/3680528.3687625. URL https://doi.org/10. 1145/3680528.3687625.

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In ECCV, pp. 87–103, 2024b.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, pp. 2256–2265, 2015.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In ICLR,

2024. URL https://openreview.net/forum?id=WNzy9bRDvG. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben

Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.

Dropout: a simple way to prevent neural networks from overfitting. JMLR, 15(1):1929–1958, January 2014. ISSN 1532-4435.

Vladimir Tchakaloff. Formules de cubatures m´ecaniques a` coefficients non n´egatifs. Bulletin des Sciences Math´ematiques, 81(2):123–134, 1957.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), NeurIPS, volume 30. Curran Associates, Inc., 2017.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In NeurIPS, 2023.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/ 2508.02324.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis, 2023. URL https://arxiv.org/abs/2306.09341.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: learning and evaluating human preferences for text-to-image generation. In NeurIPS, pp. 15903–15935, 2023.

Tianwei Yin, Micha¨el Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis. In NeurIPS, 2024a. URL https://openreview.net/forum?id=tQukGCDaNT.

Tianwei Yin, Micha¨el Gharbi, Richard Zhang, Eli Shechtman, Fr´edo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In CVPR, 2024b.

Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In ICLR, 2025.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In ICLR, 2023. URL https://openreview.net/forum?id=Loek7hfb46P.

Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictorcorrector framework for fast sampling of diffusion models. In NeurIPS, 2023.

Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via scoreregularized continuous-time consistency, 2025. URL https://arxiv.org/abs/2510. 08431.

Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. In ICML, 2025. Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity

distillation: exponentially fast distillation of pretrained diffusion models for one-step generation. In ICML, 2024.

Algorithm 2: Data-dependent on-policy π-ID training loop with time shifting.

Input: NFE, teacher Gθ, data–condition distribution p(x0, c), shift m Output: Student Gϕ

- 1 Initialize student params ϕ
- 2 S ← NFE 1 , NFE2 , · · · , 1 // can be adjusted to reduce final step size

- 3 for finite samples x0, c ∼ p(x0, c), ϵ ∼ N(0, I), τ′ ∼ U(0, 1) do

- 4 τsrc ← min{ τsrc | τsrc ∈ S and τsrc ≥ τ′ }
- 5 τdst ← max{ τdst | τdst ∈ S ∪ { 0 } and τdst < τsrc }
- 6 tsrc ← 1+(mmτ−src1)τ

src

// time shifting (Esser et al., 2024)

- 7 xtsrc ← αtsrcx0 + σtsrcϵ
- 8 π ← Gϕ(xtsrc, tsrc, c)
- 9 πD ← stopgrad(π) or πD ← dropout(stopgrad(π))
- 10 Lϕ ← 0
- 11 for finite samples τ ∼ U(τdst, τsrc) do

- 12 t ← 1+(mmτ−1)τ

- 13 xt ← xtsrc + t t

src

πD(xt, t) dt

- 14 Lϕ ← Lϕ + 21∥Gθ(xt, t, c) − π(xt, t)∥2 // can be replaced with Eq. (6)

- 15 ϕ ← Adam(ϕ, ∇ϕLϕ) // optimizer step

Algorithm 3: Data-free on-policy π-ID training loop with time shifting.

Input: NFE, teacher Gθ, condition distribution p(c), shift m Output: Student Gϕ

- 1 Initialize student params ϕ
- 2 for finite samples c ∼ p(c), x1 ∼ N(0, I) do

- 3 τsrc ← 1, tsrc ← 1
- 4 Lϕ ← 0
- 5 while τsrc > 0 do

- 6 τdst ← τsrc − NFE1 // can be adjusted to reduce final step size

- 7 tdst ← 1+(mmτ−dst1)τ

dst

// time shifting (Esser et al., 2024)

- 8 π ← Gϕ(xtsrc, tsrc, c)
- 9 πD ← stopgrad(π) or πD ← dropout(stopgrad(π))
- 10 for finite samples τ ∼ U(τdst, τsrc) do

- 11 t ← 1+(mmτ−1)τ

- 12 xt ← xtsrc + t t

src

πD(xt, t) dt

- 13 Lϕ ← Lϕ + τsrc−2τdst ∥Gθ(xt, t, c) − π(xt, t)∥2 // can be replaced with Eq. (6)

- 14 xtdst ← xtsrc + t tdst

src

πD(xt, t) dt

- 15 τsrc ← τdst, tsrc ← tdst

- 16 ϕ ← Adam(ϕ, ∇ϕLϕ) // optimizer step

- A USE OF LARGE LANGUAGE MODELS

In preparing this manuscript, we used large language models (LLMs) as general-purpose writing assistants for grammar corrections, rephrasing, and clarity/concision edits. All LLM-suggested edits were reviewed and verified by the authors, who take full responsibility for the final manuscript.

- B ADDITIONAL TECHNICAL DETAILS

- B.1 GM TEMPERATURE

Inspired by the temperature parameter in language models, we introduce a similar temperature parameter for the GMFlow policy during inference. Let T > 0 be the temperature parameter. Given a

#### ) = Kk=1 AkN u;µk,s2I , the new GM prob-

C-dimensional GM velocity distribution q(u|xt

src

(a) Off-policy: decay teacher ratio=100%

(c) On-policy: teacher ratio=0%

(b) Mixture of teacher and detached policy

decay

𝒙𝒙𝑡𝑡src

𝒙𝒙𝑡𝑡src

(see Fig. 3)

𝐺𝐺𝝓𝝓 𝒙𝒙𝑡𝑡dst

𝐺𝐺𝝓𝝓 𝒙𝒙𝑡𝑡dst

Policy 𝜋𝜋

Policy 𝜋𝜋

stopgrad (+ dropout)

Detached policy 𝜋𝜋D rollout

Teacher 𝐺𝐺𝜽𝜽 step Policy 𝜋𝜋 rollout

Detached policy 𝜋𝜋D

- Figure 8: Three stages of scheduled trajectory mixing. (a) Off-policy behavior cloning with a teacher ratio of 1. (b) Mixed teacher and detached-policy segments with a decaying teacher ratio. (c) Onpolicy imitation learning with a teacher ratio of 0 (Fig. 3).

ability with temperature T is defined as: qT(u|xt

qT1 (u|xt

#### ) RC qT1 (u|xt

. (3) Although qT(u|xt

src

) :=

src

)du

src

) does not have a general closed-form expression, it can be approximated by the following expression, which works very well as a practical implementation:

src

1 T

K

A

#### N u;µk,s2TI . (4)

k

#### qT(u|xt

) ≈

src

1 zT

K z=1 A

k=1

For the distilled FLUX and Qwen-Image models, we set T = 0.3 for 4-NFE generation and T = 0.7 for 8-NFE generation. An exception is that we do not apply temperature scaling to the final step, as we found this can impair texture details. As shown in Table 7, ablating GM temperature from the 4-NFE GM-FLUX leads to degraded teacher alignment.

- B.2 SCHEDULED TRAJECTORY MIXING FOR GUIDANCE-DISTILLED TEACHERS

To reduce out-of-distribution exposure in imitation learning, scheduled sampling (Bengio et al., 2015) stochastically alternates between expert (teacher) and learner policy during trajectory integration, decaying the expert probability from 1 to 0. However, naively applying it to π-ID is impractical because the teacher flow model Gθ is much slower than the network-free policy πD.

To maintain constant compute throughout training, we introduce a scheduled trajectory mixing strategy. Since the teacher is slow, we fix the total number of teacher queries, allow each query to cover a coarse, longer step initially, and gradually shrink the teacher step size while filling the gaps with the fast policy πD. As shown in Fig. 8 (a), training initially adopts a fully off-policy teacher trajectory (behavior cloning). At the beginning time ta of each teacher step, we roll in the learner policy π, integrate it over the same interval from ta to tb, and match its average velocity to the teacher velocity with the ℓ2 loss:

Lϕ = E

- 1

- 2

Gθ(xt

a

,ta) −

1 tb − ta

tb

ta

π(xt,t)dt

2

. (5)

As training progresses (Fig. 8 (b)), we then mix teacher and detached-policy segments while using the same loss, and linearly decay the teacher ratio—the sum of teacher step lengths divided by the total interval length tsrc − tdst. Finally, when the teacher ratio reaches 0, training reduces to onpolicy π-ID. All teacher step boundaries (starts and ends) are randomly sampled within the interval [tdst,tsrc] under the teacher ratio constraint, so that step sizes and locations vary while the total teacher-covered length follows the current ratio schedule.

We apply scheduled trajectory mixing exclusively when distilling the FLUX.1 dev model, as it lacks real CFG. Since omitting CFG doubles the teacher’s speed, we increase the number of intermediate samples (teacher steps) to 4 accordingly.

- B.3 MICRO-WINDOW VELOCITY MATCHING

For on-policy π-ID, in practice we found that replacing the instantaneous velocity matching loss in Algorithm 1 with a modified average velocity loss over a micro time window generally benefits

- Table 7: Ablation study on 4-NFE π-Flow (GM-FLUX), evaluated on the HPSv2 prompt set using teacher-referenced FID metrics (reflecting teacher alignment).

Micro window

GM temperature

FID↓ pFID↓

✓ ✓ 14.3 19.2 ✓ 14.9 20.1 ✓ 14.6 20.3

FLUX.1 dev 43-NFE

FLUX.1 dev 128-NFE

[Figure 70]

[Figure 71]

| |
|---|

| |
|---|

|[Figure 72]|
|---|

|[Figure 73]|
|---|

A futuristic, sleek sports car with a low, aerodynamic design is shown in motion against a backdrop of a city skyline at sunset. The car features sharp angles, large wheels with orange accents, and a prominent front grille. The cityscape includes tall buildings with illuminated windows, and the sky is painted with hues of orange and blue as the sun sets. The lighting is warm and golden, with the sun setting behind the city, casting a glow over the scene. The car is positioned in the foreground, with the city skyline in the background, creating a sense of depth and movement.

- Figure 9: The 128-NFE FLUX.1 dev often generates blurry images, whereas the 43-NFE FLUX.1 dev reduces the blur and produces sharper edges.

training. The modified loss is defined as:

 1

2 , (6)

t−∆t

1 −∆t

Lϕ = E

Gθ(xt,t) −

π(xt,t)dt

2

t

where ∆t is the window size. We set ∆t = 3/128 (three policy integration steps) for all FLUX.1 and Qwen-Image experiments.

The benefits of micro-window velocity matching are threefold:

- • It generally smooths the training signal, reducing sensitivity to sharp local variations in the teacher trajectory.
- • It stabilizes the less robust DX policy. In the ImageNet experiments, we observe that training with the DX policy diverges without this modification.
- • With ∆t = 3/128, the policy effectively mimics teacher sampling with 1283 ≈ 43 steps instead of 128 steps. For the guidance-distilled FLUX.1 dev model, we observe that the teacher often generates blurry images using 128-step sampling, while 43-step sampling yields sharper results (see Fig. 9). This behavior is inherited by the student, so micro-window velocity matching helps reduce blur.

Table 8: Hyperparameters used in the ImageNet experiments.

1-NFE 2-NFE GM-FM (K = 32)

GM-REPA (K = 32) GM dropout 0.05 0.05 0.05 - - - 0.05 # of intermediate states 2 2 2 2 2 2 2 Window size (raw) ∆τ - - - 10/128 5/128 3/128 Shift m 1.0 1.0 1.0 1.0 1.0 1.0 1.0 Teacher CFG 2.7 3.2 3.2 3.2 3.2 3.2 2.8 Teacher CFG interval t ∈ [0,0.6] t ∈ [0,0.7] t ∈ [0,0.7] t ∈ [0,0.7] t ∈ [0,0.7] t ∈ [0,0.7] t ∈ [0,0.7] Learning rate 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5 Batch size 4096 4096 4096 4096 4096 4096 4096 # of training iterations in Table 2

GM-REPA (K = 8)

GM-REPA (K = 32)

DX-REPA (N = 10)

DX-REPA (N = 20)

DX-REPA (N = 40)

140K - 140K - - - 24K

EMA param γ in Karras et al. (2024)

7.0 7.0 7.0 7.0 7.0 7.0 7.0

Table 9: Hyperparameters used in FLUX and Qwen-Image experiments.

4-NFE 8-NFE GM-FLUX (K = 8)

GM-FLUX (K = 8) GM dropout 0.1 0.1 - - 0.1 GM temperature T 0.3 0.3 - - 0.7 # of intermediate states 4 2 4 2 4 Window size (raw) ∆τ 3/128 3/128 3/128 3/128 3/128 Shift m 3.2 3.2 3.2 3.2 3.2 Final step size scale 0.5 0.5 0.5 0.5 0.5 Teacher CFG 3.5 4.0 3.5 4.0 3.5 Learning rate 1e-4 1e-4 1e-4 1e-4 1e-4 Batch size 256 256 256 256 256 # of training iterations 3K 9K 3K 9K 3K # of decay iterations (§ B.2) 2K - 2K - 2K EMA param γ in Karras et al. (2024)

GM-Qwen (K = 8)

DX-FLUX (N = 10)

DX-Qwen (N = 10)

7.0 7.0 7.0 7.0 7.0

As shown in Table 7, ablating the micro window trick from the 4-NFE GM-FLUX leads to degraded teacher alignment.

- B.4 TIME SAMPLING

For high resolution image generation, Esser et al. (2024) proposed a time shifting mechanism to rescale the noise strength. Let τ be the pre-shift raw time and m be the shift hyperparameter, the

shifted time is defined as t := 1+(mmτ−1)τ . Following this idea, π-ID samples times uniformly in raw-time space and then applies the shift to remap those samples. Detailed time sampling routines are given in Algorithms 2 and 3.

For FLUX.1 and Qwen-Image, we use a fixed shift m = 3.2, which is a rounded approximation of FLUX.1’s official dynamic shift at 1MP resolution. In addition, several diffusion/flow models reduce the noise strength at the final step to improve detail (Karras et al., 2022; Wu et al., 2025). Accordingly, for FLUX.1 and Qwen-Image we halve the final step size (relative to previous steps) in raw-time space.

Table 10: FLUX.1 schnell evaluation results on COCO-10k dataset and HPSv2 prompt set.

COCO-10k prompts HPSv2 prompts

Model Distill method NFE

###### Data align. Prompt align. Pref. align. Prompt align. Pref. align. FID↓ pFID↓ CLIP↑ VQA↑ HPSv2.1↑ CLIP↑ VQA↑ HPSv2.1↑

FLUX.1 schnell GAN 4 21.8 29.1 0.274 0.913 0.297 0.297 0.843 0.301

FLUX.1 schnell 4-NFE

FLUX.1 dev 50-NFE

𝝅𝝅-Flow (GM-FLUX) 4-NFE

[Figure 74]

[Figure 75]

[Figure 76]

The president being abducted by aliens.

[Figure 77]

[Figure 78]

[Figure 79]

A puppy is driving a car in a film still.

[Figure 80]

[Figure 81]

[Figure 82]

Spiderman character in the game Sea of Thieves.

[Figure 83]

[Figure 84]

[Figure 85]

A portrait of a melting skull with intricate abstract details, created by Tooth Wu, Wlop Beeple, and Dan Mumford, using Octane Render.

[Figure 86]

[Figure 87]

[Figure 88]

The image depicts a beautiful goddess of spring wearing a wreath and flowy green skirt, created by artist wlop.

- Figure 10: Typical failure cases of FLUX.1 schnell. For reference, we also show the corresponding FLUX.1 dev and π-Flow results from the same initial noise.

- C ADDITIONAL IMPLEMENTATION DETAILS AND HYPERPARAMETERS

All models are trained with BF16 mixed precision, using the 8-bit Adam optimizer (Kingma & Ba, 2014; Dettmers et al., 2022) without weight decay. For inference, we use EMA weights with a dynamic moment schedule (Karras et al., 2024). Detailed hyperparameter choices are listed in

- Table 8 and 9.

Element-wise factorization 𝐻𝐻 × 𝑊𝑊 × 16 × 1

###### Pixel-wise factorization

Patch-wise factorization 𝐻𝐻/2 × 𝑊𝑊/2 × 64

𝐻𝐻 × 𝑊𝑊 × 16

[Figure 89]

[Figure 90]

[Figure 91]

Fuzzy clay creatures playing a game in a heavily tilted green pasture.

- Figure 11: Comparison of different L × C factorization schemes for the GMFlow policy, evaluated on the toy models after 4000 optimization iterations. The default pixel-wise factorization (C = VAE latent channels) proposed by Chen et al. (2025) produces neutral colors and detailed textures. In contrast, element-wise factorization (C = 1) leads to over-saturated colors, high contrast, and over-smoothed textures, while patch-wise factorization (C = 2 × 2 × VAE latent channels) results in “confetti” artifacts and noisy textures.

C.1 DISCUSSION ON GMFLOW POLICY HYPERPARAMETERS

For the GMFlow policy, we observed that the hyperparameters suggested by Chen et al. (2025) (K = 8, C = VAE latent channel size) generally work well. These parameters play important roles in balancing compatibility, expresiveness, and robustness. A larger K improves expresiveness but impairs compatibility as it may complicate network training. A larger C improves robustness (since GMFlow models correlations within each C-dimensional chunk) but impairs expresiveness (raises the theoretical K = N ·C bound). In addition, improving expresiveness may generally compromise robustness, due to the increased chance of encountering outlier trajectories during inference.

To further justify the design choice of pixel-wise factorization in GMFlow (where the latent grid is factorized by L = H ×W, C = VAE latent channel size), we conduct toy model experiments to test alternative factorizations. In each experiment, we initialize a set of GMFlow parameters Aik,µik,s with K = 32 components, and directly optimize them to overfit the FLUX teacher’s behavior over the entire time domain t ∈ (0,1], using simple π-ID (Algorithm 1) on a fixed initial noise x1 and a fixed text prompt. This setup isolates the inductive bias of the policy itself, without any influence from the student network. As shown in Fig. 11, pixel-wise factorization achieves the best results within 4000 optimization iterations, producing neutral colors and rich textures, and is therefore the most suitable choice for the GMFlow policy.

- D DISCUSSION ON FLUX.1 SCHNELL

The official 4-NFE FLUX.1 schnell model (Black Forest Labs, 2024a) (based on adversarial distillation (Sauer et al., 2024a)) is distilled from the closed-source FLUX.1 pro instead of the publicly available FLUX.1 dev. This makes a direct comparison to the student models in Table 3 inequitable.

For reference, nevertheless, we include the COCO-10k and HPSv2 metrics for FLUX.1 schnell in Table 10. These metrics reveal a trade-off: while FLUX.1 schnell achieves significantly better data and prompt alignment than FLUX.1 dev, its preference alignment is substantially weaker than FLUX.1 dev and all of its students.

To validate this observation, we conducted a human preference study. Our 4-NFE π-Flow (GMFLUX) was compared against FLUX.1 schnell on 200 images generated from HPSv2 prompts. π-Flow was preferred by users 59.5% of the time, aligning with the HPSv2.1 preference metric. Furthermore, qualitative comparisons in Fig. 10 reveals that FLUX.1 schnell is prone to frequent structural errors (e.g., missing/extra/distorted limbs), whereas π-Flow maintains coherent structures.

- E PROOF OF THEOREM 1

We will prove that a GM with N · C components suffices for approximating any N-step trajectory in RC by first establishing Theorem 2, and then applying the Richter–Tchakaloff theorem to show

that a mixture of N · C Dirac deltas satisfy all ODE moment equations, which finally leads to N · C Gaussian components.

- Theorem 2. Given pairwise distinct times t1,...,tN ∈ (0,1] and vectors xt

n ∈ RC for n = 1,...,N, there exists a probability measure p(dx0) on RC, such that Eq (1) holds at t = tn for every n = 1,...,N.

, x˙t

n

- E.1 MOMENT EQUATION For every t ∈ (0,1], the ODE moment equation has the following equivalent forms:

x˙t =

RC

xt − x0 t

p(dx0|xt)

⇔ x˙t

RC

p(dx0|xt) =

RC

xt − x0 t

p(dx0|xt)

⇔

RC

x0 − xt + tx˙t t

p(dx0|xt) = 0

⇔

RC

(x0 − xt + tx˙t)N xt;αtx0,σt2I p(dx0) tp(xt)

= 0

⇔

RC

(x0 − xt + tx˙t)N xt;αtx0,σt2I p(dx0) = 0. (7)

Let g(t,x0) := (x0 −xt +tx˙t)N xt;αtx0,σt2I be a kernel function. The above equation can be written as a multivariate homogeneous Fredholm integral equation of the first kind:

RC

g(t,x0)p(dx0) = 0. (8)

To prove Theorem 2, we need to show that there exists a probability measure p(dx0) on RC that solves the Fredholm equation at t = tn for every n = 1,··· ,N.

- E.2 UNIVARIATE MOMENT EQUATION

To prove the existence of a solution to the multivariate Fredholm equation, we can simplify the proof into a univariate case by showing that an element-wise probability factorization p(dx0) =

C i=1 p(dxi0) exists that solves the Fredholm equation. In this case, Eq. (7) can be written as:

∀i = 1,2,··· ,C,

R

(xi0 − xit + tx˙it)N xit;αtxi0,σt2 p(dxi0)

j̸=i R

N xjt;αtxj0,σt2 p(dxj0) = 0

⇔ ∀i = 1,2,··· ,C,

R

(xi0 − xit + tx˙it)N xit;αtxi0,σt2 p(dxi0) = 0. (9)

To see this, we need to prove that there exists a probability measure p(x0) on R that solves the following univariate Fredholm equation at t = tn for every n = 1,··· ,N:

R

g(t,x0)p(dx0) = 0, (10) where g(t,x0) := (x0 − xt + tx˙t)N xt;αtx0,σt2 is the univariate kernel function.

- E.3 CONVEX COMBINATION

Lemma 1. Define the vector function:

γ : R → RN, γ(x0) = (g(t1,x0),g(t2,x0),··· ,g(tN,x0)). (11) Then, the zero vector lies in the convex hull in RN, i.e.:

0 ∈ conv{ γ(x0) | x0 ∈ R } ∈ RN. (12)

Proof. Define S := conv{ γ(x0) | x0 ∈ R }. Assume for the sake of contradiction that 0 ∈/ S. By the supporting and separating hyperplane theorem, there exists w ̸= 0 ∈ RN, such that:

∀χ ∈ S, ⟨w,χ⟩ ≤ 0. (13) In particular, this implies that:

∀x0 ∈ R, ⟨w,γ(x0)⟩ ≤ 0. (14)

Define h(x0) := ⟨w,γ(x0)⟩ = Nn=1 wng(tn,x0). Recall the definition of g(t,x0) that:

g(t,x0) = (x0 − xt + tx˙t)N xt;αtx0,σt2

(xt − αtx0)2 2σt2

x0 − xt + tx˙t √

. (15)

exp −

=

2πt2

Let n∗ be an index with wn∗ ̸= 0 for which the exponential term above decays the slowest, i.e.:

αt2

αt2

n∗

#### wn∗ ̸= 0 . (16)

n

= min

#### 2σt2

#### 2σt2

n∗

n

2 tn∗

2 tn

2σt2n > α

2 t

2σt2 is monotonic, for every n ̸= n∗ with wn ̸= 0, we have α

Note that since α

. Therefore, as |x0| → ∞, h(x0) is dominated by the n∗-th component, i.e.:

2σt2

n∗

n∗x0)2 2σt2

x0 − xt

n∗ − αt

n∗ + tn∗x˙t

(xt

n∗

#### (1 + O(1)). (17)

exp −

h(x0) = wn∗

2πt2n∗

n∗

Because the term x0−xt

n∗ changes sign between −∞ and +∞, h(x0) takes both positive and negative values. This contradicts the hyperplane implication that h(x0) ≤ 0. Therefore, we conclude that 0 ∈ S.

n∗ +tn∗x˙t

| |
|---|

By Lemma 1 and Carath´eodory’s theorem, the zero vector can be expressed as a convex combination of at most N+1 points on γ(x0). Therefore, there exists a finite-support probability measure p(dx0) consisting of N + 1 Dirac delta components that solves the univariate Fredholm equation at t = tn for every n = 1,··· ,N, completing the proof of Theorem 2.

- E.4 N · C COMPONENTS SUFFICE

Richter’s extension to Tchakaloff’s theorem states as follows.

- Theorem 3 (Richter (1957); Tchakaloff (1957)). Let V be a finite-dimensional space of measurable functions on RC. For some probability measure p(dx0) on RC, define the moment functional:

Λ: V → R, Λ[g] :=

g(x0)p(dx0). (18)

RC

Then there exists a K-atomic measure p∗(dx0) = Kk=1 Akδµ

(dx0) with Ak > 0 and K ≤ dimV such that:

k

∀g ∈ V, Λ[g] =

g(x0)p∗(dx0) =

RC

K

Akg(µk). (19)

k=1

By Theorem 2, we know that for V = span{ gi(tn,x0) | i = 1,··· ,C, n = 1,··· ,N } with the scalar function gi(tn,x0) := (xi0−xit+tx˙it)N xt;αtx0,σt2I , there exists a probability measure p(dx0) such that R

gi(tn,x0)p(dx0) = 0 for every i,n. Then, by the Richter–Tchakaloff theorem, there also exists a K-atomic measure with K ≤ dimV ≤ N · C that satisfies all the moment equations. By taking the upper bound, this implies the existence of an N · C-atomic probability

D

(dx0) with Ak > 0, Nk=1·C Ak = 1 that solves the Fredholm equation (Eq. (8)) at t = tn for every n = 1,··· ,N.

#### measure p∗(dx0) = Nk=1·C Akδµ

k

Finally, since N xt;αtx0,σt2I is continuous, the N · C Dirac deltas in p∗(dx0) can be replaced by a mixture of N · C narrow Gaussians, such that x˙n is approximated arbitrarily well for every n, i.e.:

∀n = 1,··· ,N, lim

n − RD

x0p(dx0|xt

xt

) tn p(dx

n

s→0

0)= Nk=1·C AkN(dx0;µk,s2I)

n − RD

x0p(dx0|xt

xt

) tn p(dx

n

=

0)= Nk=1·C Akδµk(dx0)

(20) This completes the proof of Theorem 1.

= x˙t

n

- F DERIVATION OF CLOSED-FORM GMFLOW VELOCITY

In this section, we provide details regarding the derivation of closed-form GMFlow velocity, which was originally presented by Chen et al. (2025) but not covered in detail.

) = Kk=1 AkN u;µk,s2I with Ak ∈ R+, µk ∈ RC, s ∈ R+, we first convert it into the x0-based parameterization by substituting u = xt

Given the u-based GM prediction q(u|xt

src

src−x0 σtsrc into the

density function, which yields:

#### q(x0|xt

) =

src

K

AkN x0;µxk,s2xI , (21)

k=1

with the new parameters µxk = xt

s. Then, for any t < tsrc and any xt ∈ RC, the denoising posterior at (xt,t) is given by:

#### µk and sx = σt

src − σt

src

src

p(xt|x0) Z · p(xt

), (22)

q(x0|xt) =

q(x0|xt

src

src|x0)

,xt,tsrc,t. Using the definition of forward diffusion p(xt|x0) = N xt;αtx0,σt2I , we have:

where Z is a normalization factor dependent on xt

src

q(x0|xt) = N xt;αtx0,σt2I Z · N xt

q(x0|xt

)

src

#### x0,σt2

#### ;αt

I

src

src

src

2 t

xt, σ

N x0; α1

α2t I Z′ · N x0; α1

t

q(x0|xt

=

)

src

2 tsrc

, σ

### xt

α2tsrc I

src

tsrc

I ζ

ν ζ

1 Z′′ N x0;

q(x0|xt

,

#### ),

=

src

αtxt σt2 −

αt

### xt

where ν =

src

src

,

#### σt2

src

αt2

αt2 σt2 −

src

#### .

ζ =

#### σt2

src

The result can be further simplified into a new GM:

q(x0|xt) =

K

A′kN x0;µ′k,s′2I , (23)

k=1

with the following parameters:

where the new logit a′k is given by:

s2x s2xζ + 1

s′2 =

(24)

s2xν + µxk s2xζ + 1

µ′k =

(25)

expa′k K k=1 expa′

A′k =

, (26)

k

µxk · ν − 21ζµxk

a′k = log Ak +

. (27)

s2xζ + 1

Finally, the closed-form GMFlow velocity at (xt,t) is given by function π:

xt − Ex

0∼q(x0|xt)[x0] t

π: RC × R → RC, π(xt,t) =

xt − Kk=1 A′kµ′k t

. (28)

=

Extension to discrete support. The closed-form GMFlow velocity can also be generalized to discrete support by taking lims

x→0 π(xt,t), which yields the simplified parameters: µ′k = µxk (29) a′k = log Ak + µxk · ν −

- 1

- 2

ζµxk . (30)

- G ADDITIONAL QUALITATIVE RESULTS. We show additional uncurated results of FLUX-based models in Fig. 12 and 13.

SenseFlow (FLUX) 4-NFE

Hyper-FLUX 8-NFE

FLUX Turbo 8-NFE

FLUX.1 dev 50-NFE

𝝅𝝅-Flow (GM-FLUX) 4-NFE

𝝅𝝅-Flow (GM-FLUX) 8-NFE

[Figure 92]

This picture follows the abstract expressionism visual style. A character with short pink hair, blue eyes, wearing a black hoodie, blue neck gaiter, black shorts, and sneakers, stands in a busy street. Vibrant shop signs, lanterns, traffic cones, a fire hydrant, and sunlight create a dynamic atmosphere.

A battle-hardened male warrior stands tall, his blood-drenched face and determined eyes reflecting fierce resolve. He grips an intricately decorated sword, its hilt gleaming in the dim light. The Baroque style fills the scene, with dramatic lighting casting deep shadows on his worn features and intricate armor textures.

Please recreate the extravagant apartment's living room, overlooking the Mediterranean from the desert. The striking surroundings feature succulents and cacti. The furniture is crocheted, including the sofa, walls, curtains, carpet, tables, and wall art, with a desert wildlife view visible through the window, also crocheted.

The bed features a modern, minimalist design with a clean look and low profile. Its headboard is subtly curved and remains within the bed's width without wrapping around the sides, maintaining moderate height for functionality and comfort. Upholstered entirely in textured neutral fabric like light grey or beige, the frame includes headboard, side rails, and footboard without tufting or embellishments. The mattress fits snugly with no visible gap between it and the frame. Short, dark-colored legs subtly contrast with the light upholstery. Bedding consists of neatly arranged white fitted sheets, flat sheets, and pillows. A simple, light-colored wall serves as the neutral background, ensuring the bed remains the image's focal point.

Stunning woman in misty sunlit field, delicate lace rose gold dress flowing in breeze, gazing over shoulder, hair adorned with sunflowers, sunlight highlighting her melancholy face, sorrowful expression, shimmering eyes, leaves swirling around her, overcast sky with lively atmosphere, intricate dress details, realistic style.

Please craft an image of a genuine individual. A young woman, dressed in an authentic cowboy outfit, stands confidently next to a weathered barn. The scene is evocatively captured in sepia tones. In the distance, a vast field stretches out, dotted with tall grasses and dried hay bales, while a vibrant blue sky peeks in from the corner.

In a captivating style reminiscent of animated storytelling, behind prison bars stands a powerful woman, eyes glowing red and intense, with a prominent scar.

Figure 12: An uncurated random batch from the OneIG-Bench prompt set, part A.

SenseFlow (FLUX) 4-NFE

Hyper-FLUX 8-NFE

FLUX Turbo 8-NFE

FLUX.1 dev 50-NFE

𝝅𝝅-Flow (GM-FLUX) 4-NFE

𝝅𝝅-Flow (GM-FLUX) 8-NFE

[Figure 93]

A row of blue buses idles beneath a metal canopy marked "GATE 12 - INTERCITY EXPRESS" at a busy bus terminal. An LED screen scrolls "NEXT DEPARTURE: 15:30." Travelers sip coffee from paper cups, while a nearby newsstand labeled "CITY NEWS" displays headlines and postcards of local landmarks.

Could you explore the differences between nuclear fission and fusion by designing a comparison chart?Tips: Energy release, reaction process, atomic nuclei

Could you show us the educational diagram of nanomaterial structures? Tips: nanomaterials, structure

Green tea ice cubes float in milk, each with a distinct ink pattern. The surface appears smooth and glass-like, enhancing the realistic feel and inviting a tactile experience. The entire scene is depicted in 3d rendering.

Create an image with lifelike depiction of people, featuring a mid-thirties salaryman with thinning jet black hair. He stands confidently with hands on his hips, showcasing a crisp blue suit against the dynamic backdrop of a vibrant Tokyo cityscape.

Beneath a starry sky stretching over a serene desert landscape, the poster title "Celestial Wonders: Stargazing Nights" glows softly in silver. Midway down, text invites "Discover the universe through guided telescope tours and expert talks." Footer information provides "Every Saturday evening at Mirage Dunes Observatory. Tickets available online."

Young woman in wedding attire, ivory skirt, white block heel sandals, delicate train, snug-fitting garment creating stunning silhouette, white mesh top with beadwork and intricate maple leaf pattern, realistic portrayal.

Figure 13: An uncurated random batch from the OneIG-Bench prompt set, part B.

