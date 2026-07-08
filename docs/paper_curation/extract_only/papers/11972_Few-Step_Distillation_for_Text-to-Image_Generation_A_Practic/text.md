# arXiv:2512.13006v1[cs.CV]15Dec2025

Few-Step Distillation for Text-to-Image Generation: A Practical Guide

Yifan Pu∗1,2, Yizeng Han∗2, Zhiwei Tang∗2, Jiasheng Tang†2,3, Fan Wang2, Bohan Zhuang2,4, Gao Huang1 1Tsinghua University 2DAMO Academy, Alibaba Group 3Hupan Lab 4Zhejiang University ∗Equal contribution, †Corresponding author

Diffusion–distillation has dramatically accelerated class-conditional image synthesis, but its applicability to open-ended text-to-image (T2I) generation is still unclear. We present the first systematic study that adapts and compares state-of-the-art distillation techniques on a strong T2I teacher model, FLUX.1-lite. By casting existing methods into a unified framework, we identify the key obstacles that arise when moving from discrete class labels to free-form language prompts. Beyond a thorough methodological analysis, we offer practical guidelines on input scaling, network architecture, and hyperparameters, accompanied by an open-source implementation and pretrained student models. Our findings establish a solid foundation for deploying fast, high-fidelity, and resource-efficient diffusion generators in real-world T2I applications. Code is available on github.com/alibaba-damo-academy/T2I-Distill.

Date: December 16, 2025

1 Introduction

In recent years, large-scale Diffusion Models (DMs) (Ho et al., 2020; Sohl-Dickstein et al., 2015) have achieved unprecedented success in the field of text-to-image synthesis, with generation quality that rivals or even surpasses human creation (Rombach et al., 2022; Saharia et al., 2022). Models such as Flux (Black Forest Labs et al., 2025), Qwen-Image (Wu et al., 2025) and Imagen (Google DeepMind, 2025), trained on massive image-text datasets, can generate high-fidelity, high-resolution images from complex textual descriptions. However, this remarkable performance comes at a significant computational cost. These models rely on an iterative sampling process, progressively converting Gaussian noise into a clear image through hundreds of Network Function Evaluations (NFEs) (Song et al., 2020; Ho et al., 2020). This process is not only computationally intensive but also suffers from high latency, severely hindering its application in scenarios requiring real-time feedback or operating in resource-constrained environments, such as interactive design tools, dynamic game content generation, and augmented reality. Therefore, drastically reducing generation latency while maintaining high-quality output has become a critical challenge in the field of generative AI.

To address the aforementioned challenges, a new research direction focused on few-step generation techniques has emerged (Luhman and Luhman, 2021; Wang et al., 2023; Salimans and Ho, 2022). The core objective is to develop models capable of producing high-quality images in a very small number of NFEs, typically between 1 and 8. This research direction has spurred a variety of innovative methods that, through different technical pathways, attempt to break the trade-off between quality and speed inherent in traditional diffusion models. These methods not only aim to accelerate the inference process of existing models but also explore the fundamental theory of generative models, laying the groundwork for the next generation of efficient generators.

Current mainstream few-step generation methods can be broadly categorized into three major paradigms. This report will conduct an in-depth analysis of one or more representative methods from each paradigm:

Distribution Distillation: This paradigm leverages a few-step and efficient student model to match the output distribution a powerful but slow pre-trained teacher model. The student model aims to simulate the teacher’s output distribution in a minimal number of steps. There are two representative methods in this area:

• Direct Distribution Distillation: Distribution Matching Distillation (DMD) is a method that distills a multistep diffusion model into an efficient, one-step image generator by minimizing the distribution difference

and incorporating a regression loss, while its successor, DMDv2, enhances image quality, training efficiency, and flexibility by replacing the costly regression loss with techniques like a GAN loss. Recent text-to-image model, like Qwen-Image-Lightning (ModelTC, 2024), adopted this technique.

• Adversarial Distribution Distillation: Latent Adversarial Diffusion Distillation (LADD) distills a slow, pretrained latent diffusion model (the teacher model) into a rapid student model capable of generating high-quality images in just one to four steps. The core of this technique (Sauer et al., 2024) is an adversarial game within the latent space, where the student model is trained to produce outputs that fool a discriminator, forcing it to match the teacher’s latent distribution and thereby achieving a massive acceleration in generation speed while maintaining high fidelity. Models like SD3-Turbo (Sauer et al., 2024) and Flux.1 Kontext [dev] (Black Forest Labs et al., 2025) use this technique.

Trajectory-based Distillation: Trajectory distillation is a method for accelerating diffusion models by training a student model to predict an entire segment of the teacher model’s sampling trajectory in fewer steps. Unlike techniques that only match the final output, this approach distills the complete generation process, which allows the student to more faithfully learn complex dynamics like classifier-free guidance and produce high-quality results in very few steps. This report will analyze the latest breakthrough in this area:

- • sCM. Simplified Consistency Models introduces a simplified theoretical framework and a set of practical techniques to stabilize and scale the training of continuous-time consistency models, enabling them to achieve state-of-the-art performance with as few as two sampling steps in ImageNet.
- • IMM. Inductive Moment Matching enforces consistency at the distribution level by using moment matching to ensure that samples generated from different points along a stochastic interpolant path converge to the same target distribution.
- • MeanFlow. Mean-Flow learns the average velocity of a flow field through a novel identity, enabling stable training and achieving state-of-the-art single-function-evaluation performance.

Distillation methods like DMD and LADD have been successfully applied to text-to-image synthesis tasks with significant results. However, the publicly available results for three highly promising new methods—sCM, MeanFlow, and IMM—are primarily focused on smaller-scale, unconditional image generation tasks (e.g., ImageNet). Their performance, adaptability, and potential advantages in the complex, open-domain task of text-to-image synthesis remain unclear. Therefore, this report aims to achieve three main goals:

- 1. Provide a deep theoretical comparison of these cutting-edge methods, dissecting their fundamental differences in core mechanisms, sources of stability, and scalability.
- 2. Propose a detailed, unified, and feasible experimental plan for adapting these methods (especially sCM and MeanFlow) to the text-to-image synthesis domain for rigorous and fair empirical evaluation.
- 3. Deliver a well-engineered, modular, and reproducible codebase that implements the proposed adaptations, training recipes, evaluation pipelines, and baseline models, facilitating fair comparison and enabling the community to reproduce and extend our experiments.

### 2 A brief overview of each method

This chapter will provide a detailed analysis of the three core methods. For each method, we will explain its core mechanism, key innovations, and propose a specific, actionable plan to adapt it to the text-to-image synthesis task. This analysis is not just theoretical but aims to provide a solid foundation for the subsequent experimental design.

- 2.1 stabilized Continuous-time Consistency Models (sCM)

Consistency Models (CMs) aim to learn a function f(xt,t) that can map a sample xt at any noise level t directly to its corresponding clean sample x0 in a single step Song et al. (2023). They can be learned through distillation (Consistency Distillation, CD) or trained from scratch (Consistency Training, CT). However, early CMs, especially in the continuous-time setting, were long plagued by severe training instability Luo et al. (2023). sCM introduces a comprehensive suite of techniques, including the TrigFlow formulation, architectural

improvements, and an adaptively weighted training objective, to resolve the core training instabilities of continuous-time consistency models (CMs).

- 2.2 MeanFlow

MeanFlow takes a distinctly different approach from standard Flow Matching. Standard flow matching models aim to learn the instantaneous velocity v(xt,t) of particles moving along a flow field. MeanFlow, instead, chooses to model a different physical quantity: the average velocity between two time steps t and r.

Beyond standard MeanFlow Training (as shown in algorithm 1 Geng et al. (2025)), we further provide a MeanFlow Distillation algorithm. In this algorithm, we use a flow matching pretrained teacher model to provide the instantaneous velocity in the MeanFlow target. The detailed algorithm is shown in algorithm 2.

Algorithm 1 MeanFlow: Training.

# fn(z, r, t): function to predict u # # x: training batch

- t, r = sample_t_r() e = randn_like(x) z = (1 - t) * x + t * e v = e - x

- u, dudt = jvp(fn, (z, r, t), (v, 0, 1))

u_tgt = v - (t - r) * dudt error = u - stopgrad(u_tgt) loss = metric(error)

Algorithm 2 MeanFlow: Distillation.

# fn(z, r, t): function to predict u # gn(z, t): flow matching teacher # x: training batch

- t, r = sample_t_r() e = randn_like(x) z = (1 - t) * x + t * e v_teacher = gn(z, t)

- u, dudt = jvp(fn, (z, r, t), (v, 0, 1))

u_tgt = v_teacher - (t - r) * dudt error = u - stopgrad(u_tgt) loss = metric(error)

- 2.3 Inductive Moment Matching (IMM)

IMM is a from-scratch training paradigm. Its core idea is to train the model so that the distribution of samples mapped to a target time s is consistent regardless of whether they start from time r or t (with

- s < r < t). This inductive consistency along the time axis ensures convergence of the learned mappings. Instead of KL divergence, IMM uses Maximum Mean Discrepancy (MMD) as a stable, sample-based measure of distributional difference, which implicitly aligns all moments between distributions.

### 3 Methodological Relationships and Comparative Insights

- 3.1 FM v.s. MeanFlow: Flow Matching as a Special Case of MeanFlow (when r ≡ t)

Flow Matching (Algorithm 4) can be understood as a specific instance of the more general MeanFlow framework (Algorithm 3). In the general MeanFlow formulation, the model learns a average velocity vector field that depends on both the current time t and a reference time r. Flow Matching is precisely recovered when we set the reference time r equal to the current time t. In this specialization, the second term of the target vanishes entirely. This simplifies the MeanFlow target to utgt = boldsymbolv. Consequently, the network is trained to predict instantaneous velocity directly. This is exactly the objective of standard Flow Matching, as shown in Algorithm 4. A direct comparison of the pseudocode in Algorithm 3 and Algorithm 4 illustrates this simplification.

##### Algorithm 3 MeanFlow: Training.

##### Algorithm 4 Flow Matching: Training.

# fn(z, r, t): function to predict u # x: training batch

# fn(z, t): function to predict v # x: training batch

- t, r = sample_two_time() e = randn_like(x)

t = sample_one_time() e = randn_like(x)

z = (1 - t) * x + t * e

z = (1 - t) * x + t * e v = e - x

- v = e - x u_pred, dudt = jvp(fn, (z, r, t), (v, 0, 1))

v_pred = fn(z, t) v_tgt = v error = v_pred - v_tgt loss = metric(error)

u_tgt = v - (t - r) * dudt error = u_pred - stopgrad(u_tgt) loss = metric(error)

- 3.2 FM v.s. sCM: Flow Matching and TrigFlow are interconvertible without re-training

Flow Matching and TrigFlow frameworks are mutually convertible at inference time (Zheng et al., 2025; Chen et al., 2025), enabling models trained in one paradigm to be used with samplers from the other without any retraining. Specifically, a pre-trained Flow Matching model, denoted by its velocity field estimator vθ(xt,FM,tFM,y), can be used to denoise a sample xt,Trig from a TrigFlow process. This is achieved by first transforming the TrigFlow state (xt,Trig,tTrig) into its Flow Matching equivalent (xt,FM,tFM). The time variable is mapped to preserve the signal-to-noise ratio (SNR), and the state variable is rescaled accordingly:

tFM =

sin(tTrig) cos(tTrig) + sin(tTrig)

, xt,FM =

1 cos(tTrig) + sin(tTrig) · xt,Trig. (1)

Subsequently, the output of the Flow Matching model is used to construct the optimal TrigFlow estimator Fˆθ, which provides the correct update direction for a TrigFlow-based solver. This relationship is given by:

Fˆθ xt,Trig,tTrig,y =

cos(tTrig) − sin(tTrig) cos(tTrig) + sin(tTrig) · xt,FM +

1 cos(tTrig) + sin(tTrig) · vθ(xt,FM,tFM,y). (2)

Conversely, this mapping is fully and losslessly reversible, allowing a native TrigFlow model Fθ to operate within a Flow Matching sampler. Given a Flow Matching state (xt,FM,tFM), we first map it to the TrigFlow domain using the inverse transformations for time and state:

tTrig = arctan

tFM 1 − tFM

, xt,Trig =

xt,FM

t2FM + (1 − tFM)2

. (3)

The TrigFlow model Fθ processes this transformed input, and its output is then converted back into the velocity field estimate vθ(xt,FM,tFM,y) required by the Flow Matching framework. This output transformation is defined by:

vθ(xt,FM,tFM,y) =

1 t2FM + (1 − tFM)2

Fθ xt,Trig,tTrig,y −

1 − 2tFM t2FM + (1 − tFM)2

xt,FM. (4)

This bidirectional conversion ensures complete interoperability, allowing practitioners to flexibly combine models and samplers from either framework.

- 3.3 sCM v.s. MeanFlow: MeanFlow as a Synchronization Limit of sCM

Under the Flow Matching (FM) parameterization, where the velocity field is directly modeled by a neural network, i.e., vθ(xt,t) = Fθ(xt,t), a direct comparison between the loss gradients of Stochastic Control Matching (sCM) and our proposed MeanFlow (MF) reveals an insightful connection. The respective gradients

with respect to the network parameters θ, denoted as ∇θL′, are given by:

∇θL′sCM = Et,x

t − ∇θFθ(xt,t),vt − Fθ−(xt,t) − t

dFθ dt

∇θL′MF = Et,x

t − ∇θFθ(xt,t),vt − Fθ(xt,t) − t

dFθ− dt

(xt,t) (5)

(xt,t) (6)

A comparison of Eq. (5) and Eq. (6) highlights their nearly identical structure. The sole distinction lies in the regression target provided for the network’s output within the inner product. Specifically, sCM employs a target network Fθ−, parameterized by θ−, which is typically an exponential moving average (EMA) of the online network’s parameters θ. This technique is widely used to stabilize training by providing a more consistent and slowly-evolving target.

In stark contrast, MeanFlow utilizes the current online network Fθ for its entire regression target. Consequently, the MeanFlow gradient is mathematically equivalent to the sCM gradient under the condition that the target network is fully synchronized with the online network at every training step, i.e., by setting θ− = θ. This interpretation frames MeanFlow not as a distinct method, but as a specific, simplified variant of sCM that dispenses with the EMA-based target stabilization. This design choice results in a more up-to-date, self-referential training dynamic, differentiating its behavior from standard sCM.

- 3.4 CM v.s. IMM: Consistency Models as a Special Case of Implicit Moment Matching As shown in the Eq.(12) of the IMM paper (Zhou et al. (2025)), the IMM loss is:

t,x′t,xr,x′r,s,t w(s,t) k ys,t,ys,t′ + k ys,r,ys,r′ − k ys,t,ys,r′ − k ys,t′ ,ys,r , (7)

LIMM(θ) = Ex

−

−

where ys,t = fs,tθ (xt), ys,t′ = fs,tθ (x′t), ys,r = fθ

s,r(x′r), k(·,·) is a kernel function, and w(s,t) is a prior weighting function. As is illustrated in Appendix G of the original paper, when we set xt = x′t, xr = x′r, we have fs,tθ (xt) = fs,tθ (x′t) and fθ

s,r(xr), ys,r′ = fθ

−

−

s,r(x′r) and let k(x,y) = −||x − y||2, which means ys,t = ys,t′

s,r(xr) = fθ

and ys,r = ys,r′ . So k(ys,t,ys,t′ ) = k(fs,tθ (xt),fs,tθ (x′t)) = 0 and k(ys,r,ys,r′ ) = k(fs,rθ (xr),fs,rθ (x′r)) = 0 by definition. The original IMM loss in equation (7) reduce to:

t,xr,s,t w(s,t) − k ys,t,ys,r′ − k ys,t′ ,ys,r . (8)

LIMM(θ) = Ex

−

−

Furthermore, since we use k(x,y) = −||x−y||2, −k(ys,t,ys,r′ ) = ||fs,tθ (xt)−fθ

##### s,r(xr)||2, and −k(ys,t′ ,ys,r) = ||fs,tθ (x′t) − fθ

s,r(x′r)||2 = ||fs,tθ (xt)−fθ

−

−

s,r(xr)||2, a more simplified expression is

s,r(xr)||2 = ||fs,tθ (xt) − fθ

−

t,xr,s,t 2w(s,t) ||fs,tθ (xt) − fθ

s,r(xr)||2 . (9)

LIMM(θ) = Ex

If s is a small positive constant, we further have fs,tθ (xt) ≈ gθ(xt,t) , where gθ(xt,s,t) is the diffusion model parameterized with EDM (Karras et al. (2022)) and we drop s as input. If gθ(xt,t) itself satisfies boundary condition at s = 0, we can directly take s = 0 in which case f0θ,t(xt) = gθ(xt,t). And under these assumptions, and omit the constant 2, our loss becomes

t,xr,t w(t) ||gθ(xt,t) − gθ−(xr,r)||2 , (10)

LIMM(θ) = Ex

which is a discrete-time consistency model(CM) loss using ℓ2 distance. In short, the IMM loss degenerates to the discrete-time Consistency Model (CM) loss when utilizing a single-particle estimate (setting xt = x′t and xr = x′r), employing the negative squared Euclidean distance as the kernel function (i.e., k(x,y) = −∥x−y∥2), and setting the target time s to 0.

### 4 Text-to-Image Adaptation with FLUX.1-lite

In this section, we detail the adaptation of the 8B FLUX.1-lite (Freepik (2024)) model for text-to-image generation. We introduce and evaluate two primary methodologies: an adaptation based on a simplified Consistency Model (sCM) objective and a novel MeanFlow training objective.

- 4.1 Experimental Setup

For all subsequent experiments, we utilize 32 Nvidia H20 GPUs for training. The dataset employed is a proprietary high-quality text-to-image dataset. While the dataset cannot be publicly released, it ensures a consistent and fair comparison across all experiments. We adopt the FLUX.1-lite model with 8 billion parameters as the teacher and aim to distill a few-step student model from it. For sCM, we distill the student model for 3,000 iterations, as we observed no further improvement in the GenEval overall score beyond this point. Conversely, for MeanFlow, we report results at 25,000 iterations, as extending the training duration yielded continuous gains in the GenEval score.

4.2 Methodological Adaptation for Text-to-Image Generation

- 4.2.1 sCM Adaptation

Normalizing DiT timestep inputs stabilizes training. During the training of sCM using FLUX.1-lite, we observed a phenomenon similar to that reported in prior works (Lu and Song (2025); Chen et al. (2025)). Specifically, when the Diffusion Transformer’s timestep input ranges from 0 to 1000, the gradient norm increases continuously during training, eventually leading to collapse. To mitigate this, we first rescaled the diffusion transformer’s timestep to the range [0,1]. We treated the original FLUX.1-lite as the teacher and initialized a student model with an identical structure, configured to accept timesteps in [0,1]. We distilled the teacher into the student using identical images, text prompts, and proportionally scaled timesteps by minimizing the smooth L1 loss between their outputs. This distillation was performed at a resolution of 1024 × 1024 with a total batch size of 128 for 120K iterations. The resulting student model achieved performance comparable to the teacher on the GenEval benchmark, as shown in the first two lines of table 1.

Guiding the student model with the teacher’s velocity prediction. Subsequently, we utilized this timestep-rescaled model (t ∈ [0,1]) to train the sCM. Instead of the standard consistency training paradigm, we employed consistency distillation, which uses the classifier-free guided output of the teacher as the velocity target. The training was conducted at a resolution of 512 with a total batch size of 128 and a learning rate of 1e-6.

- 4.2.2 MeanFlow Adaptation

Dual-timestep input mechanism. As MeanFlow models the average velocity between timesteps r and t, the diffusion transformer requires an additional time input. The standard FLUX.1-lite encodes a single timestep t using sinusoidal embeddings and an MLP projection for AdaLN modulation. We replicate this embedding branch—initialized with the same weights—to process the time differential (t − r). The outputs of the original branch (encoding t) and the cloned branch (encoding t − r) are summed before being passed to the AdaLN layers.

Teacher Distillation for MeanFlow. While MeanFlow can be formulated as a standalone training objective (Algorithm 1), we empirically find it significantly more effective as a distillation technique (Algorithm 2). The primary distinction lies in the target velocity. In the standard training formulation, the target v = e − x serves as an unbiased but highly stochastic estimator of the true vector field, introducing substantial variance into the optimization of u. In contrast, Algorithm 2 employs a pre-trained flow-matching teacher gn to provide the target vteacher = gn(z,t). We utilize the teacher’s instantaneous velocity as a direct guidance signal because the teacher has already converged to the conditional expectation of the vector field, i.e., gn(z,t) ≈ E[v|z]. Consequently, vteacher represents a denoised, deterministic approximation of the optimal transport path. By substituting the noisy raw target with this stable teacher signal, the MeanFlow objective utgt = vteacher − (t − r)dudt focuses purely on rectifying the curvature of the trajectory via the Jacobian correction term, rather than learning the data distribution from scratch.

High-Order Loss Achieves Better Performance. While the original MeanFlow paper Geng et al. (2025) suggests that p = 1 or p = 0.5 are generally optimal in the loss function L = ∥∆∥22γ (where ∆ denotes the regression error), we empirically find that setting γ = 2 yields superior performance. This configuration, which effectively minimizes the fourth power of the norm, imposes a distinct gradient behavior beneficial for our distillation objective. Quantitatively, this adjustment leads to a substantial boost on the vanilla MeanFlow target: the GenEval score increases from 44.04% to 48.65%.

Improved CFG for MeanFlow in Text-to-Image. We compare the vanilla MeanFlow (where κ = 0) with the Improved CFG variant introduced in Appendix B.2 of the MeanFlow paper (Geng et al. (2025)). The improved version incorporates a mixing scale κ to blend class-conditional and unconditional predictions in the regression target, aligning more closely with standard practices in classifier-free guidance. Experiments show that this modification is beneficial for text-to-image generation, boosting the GenEval score from 48.65% to 51.41%.

- 4.3 Quantitative and Qualitative Results

- Table 1 Quantitative comparison on the GenEval benchmark. We evaluate the original FLUX.1-lite, the timestep-rescaled teacher, and the distilled students (sCM and MeanFlow) across varying Numbers of Function Evaluations (NFE). sCM maintains robust performance at NFE=1/2, while MeanFlow requires NFE=4 to match the teacher’s quality.

NFE Single Obj. Two Obj. Counting Colors Position Color Attri. Overall

FLUX.1-lite 28 98.44 65.40 29.69 78.19 13.50 36.25 53.58 Rescaled 28 97.81 63.89 39.06 74.73 12.50 33.50 53.58

sCM 4 97.19 56.82 35.62 76.86 8.75 38.50 52.29 sCM 2 94.69 55.81 38.44 74.20 10.25 43.50 52.81 sCM 1 89.69 36.87 26.56 67.55 6.50 32.50 43.28

MeanFlow 4 96.25 64.14 38.44 69.15 13.00 27.50 51.41 MeanFlow 2 61.88 31.31 18.12 39.89 7.25 18.50 29.49 MeanFlow 1 2.81 0.25 0.31 0.80 0.25 0.25 0.78

- Table 2 Quantitative results on DPG-Bench. This benchmark evaluates the alignment of global structure, entities, attributes, and relations. MeanFlow achieves teacher-level performance at NFE=4, whereas sCM demonstrates superior stability and alignment at lower step counts (NFE ≤ 2).

NFE Global Entity Attribute Relation Other Overall

FLUX.1-lite 28 86.09 88.73 86.27 88.50 80.87 80.20 Rescaled 28 83.93 87.00 86.0 88.86 86.54 79.83

sCM 4 82.21 86.28 85.49 88.41 86.03 77.85 sCM 2 82.22 87.26 85.90 88.66 86.50 79.06 sCM 1 82.33 82.67 85.48 83.66 81.74 75.12

MeanFlow 4 90.83 86.15 87.20 88.27 86.35 80.03 MeanFlow 2 79.81 80.50 83.67 84.13 80.49 71.09 MeanFlow 1 52.44 48.90 59.83 50.10 54.39 27.55

Quantitative Comparison. We report the evaluation results on GenEval (Table 1) and DPG-Bench (Table 2). First, the Rescaled teacher model exhibits performance nearly identical to the original FLUX.1-lite, validating that normalizing the timestep range to [0,1] does not compromise generation quality.

Comparing the distillation methods reveals a distinct trade-off between step efficiency and peak performance. First, sCM excels in extreme few-step regimes. sCM demonstrates remarkable stability at NFE=2 and even NFE=1. On GenEval, sCM (NFE=2) achieves an overall score of 52.81%, effectively matching the teacher’s 53.58%. Even at a single step (NFE=1), it retains a respectable score of 43.28%. A similar trend is observed on DPG-Bench, where sCM maintains high alignment scores across all step counts. We also find that MeanFlow requires sufficient steps but achieves high fidelity. At NFE=4, it outperforms sCM on DPG-Bench (80.03 vs. 77.85) and nearly matches the teacher, suggesting that its trajectory straightening is highly effective when

Flux.1-Lite sCM NFE=28 NFE=1 NFE=2 NFE=4

MeanFlow

NFE=1 NFE=2 NFE=4

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

a photo of a bench

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

a photo of a giraffe

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

a photo of a cake and a zebra

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

a photo of three buses

Figure 1 Qualitative comparison of distillation methods. We visualize samples generated by the teacher (left) and the two student models at NFE={1, 2, 4}. sCM produces structurally coherent images even at a single step. In contrast, MeanFlow exhibits generation collapse at NFE=1 (pure noise) and requires NFE=4 to converge to high-fidelity results.

given sufficient integration steps. However, performance collapses at NFE=1 and NFE=2 (e.g., 0.78% on GenEval at NFE=1), indicating that the learned vector field, while straight, still requires a minimum number of discretization steps to traverse accurately.

Qualitative Visualization. These quantitative findings are visually corroborated in Figure 1. For sCM, as shown in the sCM column, the model produces structurally coherent images even at NFE=1 (e.g., the giraffe and the buses are clearly recognizable). Increasing the steps to 2 or 4 primarily refines high-frequency details and textures. The MeanFlow column illustrates the collapse at NFE=1, resulting in pure noise (gray outputs). At NFE=2, the model begins to form semantic content (e.g., the bench appears), but significant artifacts and noise remain (visible in the giraffe and zebra examples). However, at NFE=4, MeanFlow produces images with exceptional sharpness and correct semantics, often surpassing sCM in fine-grained detail (e.g., the fur texture of the giraffe and the reflection on the buses). In summary, sCM is the optimal choice for real-time applications requiring NFE ≤ 2, while MeanFlow is preferable for scenarios where a slightly higher budget (NFE=4) is acceptable for maximum quality.

- 5 Codebase

Our codebase is built upon the Hugging Face Diffusers library and implements multiple distillation algorithms for accelerating the FLUX.1-lite text-to-image diffusion model. The implementation includes two primary training pipelines: (1) MeanFlow, a trajectory distillation approach utilizing Jacobian-vector products (JVPs) to enforce consistency along the flow path with optional classifier-free guidance integration; and (2) simplified Consistency Matching (sCM), which leverages tangent vector matching with learned per-sample variance through a TrigFlow reparameterization of the FLUX.1-lite model.

The framework employs a teacher-student paradigm based on a modified FLUX.1-lite MMDiT architecture. We utilize DeepSpeed ZeRO with bfloat16 mixed-precision and gradient checkpointing for efficient distributed training. The pipeline performs on-the-fly encoding via the pretrained AutoencoderKL, operating in a 16-channel latent space with 8× downsampling.

References

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M"uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, June 2025.

Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025.

Freepik. Flux.1-lite-8b. https://huggingface.co/Freepik/flux.1-lite-8B, 2024. Hugging Face model repository. Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative

modeling. In NeurIPS, 2025. Google DeepMind. Imagen 4 model card. Model card (PDF), May 2025. https://storage.googleapis.com/ deepmind-media/Model-Cards/Imagen-4-Model-Card.pdf. Published May 20, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 33:6840–6851, 2020. Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative

models. In NeurIPS, 2022. Cheng Lu and Yang Song. Simplifying, stabilizing & scaling continuoustime consistency models. In ICLR, 2025. Eric Luhman and Troy Luhman. Knowledge distillation in iterative generative models for improved sampling speed.

arXiv preprint arXiv:2101.02388, 2021. Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution

images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. ModelTC. Qwen-image-lightning. https://huggingface.co/lightx2v/Qwen-Image-Lightning, 2024. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image

synthesis with latent diffusion models. In CVPR, pages 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, and et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems (NeurIPS), 2022.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. arXiv preprint arXiv:2303.01469, 2023.

Zhendong Wang, Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Diffusion-gan: Training gans with diffusion. In ICLR, 2023.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025.

Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. In ICML, 2025.

## Appendix

### A Proofs

- A.1 Interconversion between TrigFlow and Flow Matching

The following derivation is based on the supplementary materials of sana-sprint Chen et al. (2025), with some modifications made for easier understanding. The TrigFlow framework defines the noisy input sample as:

xt,Trig = cos(tTrig)x0 + sin(tTrig)z. (11)

The Signal-to-Noise Ratios (SNRs) for the Flow Matching and TrigFlow models are defined respectively as:

1 − tFM tFM

SNR(tFM) = (

cos(tTrig) sin(tTrig)

)2, SNR(tTrig) = (

1 tan(tTrig)

)2 = (

)2. (12)

To align the models, we match their SNRs. We seek the corresponding time tFM in the Flow Matching framework that satisfies:

1 − tFM tFM

1 tan(tTrig)

)2. (13)

)2 = (

(

Solving this equation yields the interconversion relationship between tFM and tTrig:

sin(tTrig) sin(tTrig) + cos(tTrig)

tFM 1 − tFM

tFM =

, tTrig = arctan(

From this relationship, we can also derive the following three useful identities:

). (14)

tFM 1 − tFM

cos(tTrig) = cos(arctan(

1 − tFM t2FM + (1 − tFM)2

, (15)

)) =

tFM 1 − tFM

sin(tTrig) = sin(arctan(

tFM t2FM + (1 − tFM)2

, (16)

)) =

1 t2FM + (1 − tFM)2

. (17)

cos(tTrig) + sin(tTrig) =

The conversion relationship between the noisy samples xt,FM and xt,Trig can then be expressed as:

xt,FM = (1 − tFM)x0 + tFMz

1 − tFM t2FM + (1 − tFM)2

tFM t2FM + (1 − tFM)2

= t2FM + (1 − tFM)2 · (

· x0 +

· z),

= t2FM + (1 − tFM)2 · [cos(tTrig)x0 + sin(tTrig)z],

= t2FM + (1 − tFM)2 · xt,Trig,

1 cos(tTrig) + sin(tTrig) · xt,Trig,

=

(18)

This conversion confirms that the samples from both frameworks can be mapped to the same distribution, fulfilling our objective. Our next goal is to determine the optimal estimator for the TrigFlow model, Fθ, based on the optimal estimator of the Flow Matching model, vθ(xt,FM,tFM,y). We begin by considering an ideal scenario assuming sufficient model capacity. In this optimal setting, the flow matching model’s solution is:

v∗(xt,FM,tFM,y) = E[z − x0|xt

FM

,y], (19)

This is the optimal solution as conditional expectation minimizes the Mean Squared Error (MSE) loss. Analogously, the optimal solution for the TrigFlow model is given by:

F∗(xt,Trig,tTrig,y) = E[cos(tTrig)z − sin(tTrig)x0|xt

Trig

,y]. (20)

We now leverage the linearity of conditional expectation to derive the relationship: E[cos(tTrig)z − sin(tTrig)x0|xt

,y].

Trig

1 − tFM t2FM + (1 − tFM)2

tFM t2FM + (1 − tFM)2

=E[

z −

#### x0|xt

#### ,y]

FM

t2FM + (1 − tFM)2 t2FM + (1 − tFM)2

1 − 2tFM t2FM + (1 − tFM)2

E[(1 − tFM) · x0 + tFM · z|xt

E[z − x0|xt

=

,y] +

#### ,y]

FM

FM

t2FM + (1 − tFM)2 t2FM + (1 − tFM)2

1 − 2tFM t2FM + (1 − tFM)2

E[z − x0|xt

#### · xt

=

+

#### ,y]

FM

FM

1 cos(tTrig) + sin(tTrig) · E[z − x0|xt

=[cos(tTrig) − sin(tTrig)] · xt

+

#### ,y]

FM

FM

cos(tTrig) − sin(tTrig) cos(tTrig) + sin(tTrig) · xt,Trig +

1 cos(tTrig) + sin(tTrig) · E[z − x0|xt

,y]

=

FM

(21)

This derivation ultimately yields the following conversion formulas:

cos(tTrig) − sin(tTrig) cos(tTrig) + sin(tTrig) · xt,Trig +

1 cos(tTrig) + sin(tTrig) · v∗(xt,FM,tFM,y), (22)

F∗(xt,Trig,tTrig,y) =

2tFM − 1 t2FM + (1 − tFM)2 · xt,FM +

1 t2FM + (1 − tFM)2

v∗(xt,FM,tFM,y) =

· F∗(xt,Trig,tTrig,y), (23)

- A.2 sCM v.s. MeanFlow: Equivalence of Gradients

This derivation demonstrates that under the Flow Matching parameterization, the training objective gradients for the simplified consistency model (sCM) and a simplified version of Meanflow are structurally equivalent. Instead of the TrigFlow, we use the simpler Flow Matching (FM) parameterization for the time-dependent function fθ(xt,t) = xt−tFθ(xt,t), which satisfies the boundary condition fθ(x0,0) = x0. The time derivative of fθ(xt,t) is:

d dt

dFθ(xt,t) dt

, (24) where vt = dx

fθ(xt,t) = vt − Fθ(xt,t) − t

dt is the velocity of the path xt. Under this parameterization, the loss of a continuous-time consistency model is (as derived in sCM Lu and Song (2025)):

t

dfθ− (xt,t) dt

fθ⊤ (xt,t)

, (25) where θ− denotes the parameters with stopped gradients. This can be written as:

LsCM = Et,x

t

LsCM = Et,x

t

dFθ− dt

fθ(xt,t),vt − Fθ−(xt,t) − t

. (26)

To find the gradient for optimization, we differentiate this loss with respect to θ. Noting that ∇θfθ(xt,t) = −t∇θFθ(xt,t), this yields:

dFθ− dt

. (27)

∇θLsCM = Et,x

t − t∇θFθ(xt,t),vt − Fθ−(xt,t) − t

##### On the other hand, the MeanFlow loss is (as derived in MeanFlow Geng et al. (2025)):

t ∥uθ (zt,r,t) − sg (utgt )∥22 , whereutgt = vt − (t − r)(vt∂zuθ + ∂tuθ).

LMeanFlow(θ) = Et,z

When we set r to 0, it becomes:

(28)

t ∥uθ (zt,t) − sg (utgt )∥22 , whereutgt = vt − t(vt∂zuθ + ∂tuθ).

LMeanFlow(θ) = Et,z

(29)

After unifying the notation, zt ⇒ xt, uθ ⇒ Fθ, and identifying the total derivative term vt∂zuθ + ∂tuθ ⇒ dFθ−(xt,t)

dt (where the stop-gradient operator sg(·) implies the use of θ−), the MeanFlow loss becomes:

LMeanFlow(θ) = Et,x

t

dFθ−(xt,t) dt

Fθ (xt,t) − vt − t

2

2

. (30)

We differentiate the loss with respect to θ. Omitting the constant factor, the gradient becomes:

dFθ− dt

∇θLMeanFlow(θ) = Et,x

t − ∇θFθ(xt,t),vt − Fθ(xt,t) − t

. (31)

Comparing the resulting gradients:

dFθ− dt

∇θLsCM = Et,x

t − t∇θFθ(xt,t),vt − Fθ−(xt,t) − t

, (32)

dFθ− dt

∇θLMeanFlow = Et,x

t − ∇θFθ(xt,t),vt − Fθ(xt,t) − t

. (33)

Since θ− represents the parameters with stopped gradients, numerically Fθ is equal to Fθ−. Consequently, the error signal terms (the second term in the inner product) are numerically identical in both methods. The gradients differ only by the weighting factor t present in sCM.

