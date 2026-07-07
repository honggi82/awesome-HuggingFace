### Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis

Tianhe Wu*13 Ruibin Li*2 Lei Zhang†23 Kede Ma†1 https://github.com/Multimedia-Analytics-Laboratory/dpdmd

# arXiv:2602.03139v2[cs.CV]19May2026

[Figure 1]

[Figure 2]

[Figure 3]

(a) SD3.5-M (60 NFEs) (b) DMD (4 NFEs) (c) DP-DMD (Ours, 4 NFEs)

Figure 1. DP-DMD preserves sample diversity while maintaining competitive perceptual quality. All results are generated under identical text conditioning (“A smiling woman with red hair, green eyes, and dimples.”) with different random initial noise. (a) SD3.5-M (Esser et al., 2024), sampled with 30 steps (i.e., 60 NFEs), serves as the teacher model. (b) DMD (Yin et al., 2024a) and (c) DP-DMD are step-distilled student models, both evaluated using only 4 NFEs.

#### Abstract

objective (e.g., v-prediction) to preserve sample diversity, while the remaining steps are optimized with the standard DMD loss to refine perceptual quality. DP-DMD, with no perceptual or adversarial regularization, no additional modules, and no teacher-generated reference samples, preserves sample diversity while maintaining competitive visual quality under few-step sampling, providing a simple and stable alternative to other DMD variants.

Distribution matching distillation (DMD) facilitates few-step image generation by aligning a distilled student with a reference multi-step teacher. In practice, however, optimizing DMD can reduce sample diversity in few-step synthesis, and existing remedies typically rely on perceptual or adversarial regularization, leading to stability and scalability challenges during training. Here, we describe diversity-preserved DMD (DP-DMD), a role-separated distillation method inspired by the complementary roles of early and late denoising steps. Specifically, the first distillation step is trained with a teacher-derived target-prediction

#### 1. Introduction

Recent years have witnessed rapid progress in generative modeling, particularly with diffusion (Song et al., 2020) and flow-based models (Lipman et al., 2022). These approaches have opened the door to high-quality image and video synthesis (Esser et al., 2024; Wan et al., 2025), supported by advances in large foundation models (Rombach et al., 2022; Labs et al., 2025) and modern optimization techniques (Sutton et al., 1999; Liu et al., 2025b). Despite their

*Equal contribution. 1City University of Hong Kong, 2The Hong Kong Polytechnic University, 3OPPO Research Institute. Correspondence to: Lei Zhang <cslzhang@comp.polyu.edu.hk> and Kede Ma <kede.ma@cityu.edu.hk>.

Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

strong performance, these models typically require solving multi-step dynamical systems during sampling, leading to substantial inference latency and computational cost due to the large number of function evaluations (NFEs).

To reduce sampling cost, prior distillation methods have sought to “compress” a reference multi-step teacher into a few-step student. Early efforts mainly follow a trajectorybased approach (Liu et al., 2023b; Yan et al., 2024), where the student is trained to approximate the teacher’s denoising trajectories using fewer steps. More recently, a complementary line of work has shifted from trajectory imitation to distribution matching, directly aligning the student distribution with that of the teacher (Sauer et al., 2024b; Yin et al., 2024b;a; Liu et al., 2025a). Among these, distribution matching distillation (DMD, Yin et al. 2024a;b) has emerged

- as a particularly effective objective, offering a strong balance between fast sampling and high visual quality.

Nevertheless, DMD suffers from an important limitation: it often reduces sample diversity in few-step generation. As shown in Figure 1, student models optimized with the DMD loss can produce visually plausible but substantially less diverse samples than their multi-step teachers. This behavior is consistent with the mode-seeking tendency of the reverse-KL-style objective underlying DMD, which may weaken coverage of the teacher distribution, especially for lower-probability modes. In practice, this issue becomes more pronounced when aggressive step reduction leaves the student with limited capacity to capture the full diversity of the teacher distribution.

Existing attempts to mitigate this issue typically augment DMD with perceptual or adversarial regularization (Yin et al., 2024a;b; Chadebec et al., 2025; Lu et al., 2025). These regularizers encourage broader coverage of the teacher distribution, often by leveraging additional teacher-generated samples as implicit supervision. However, such designs also introduce notable drawbacks: perceptual losses increase memory and computation overhead1, while adversarial losses often make training less stable and more difficult to scale. These issues motivate a simpler and more stable way to preserve diversity during DMD.

In this paper, we describe diversity-preserved DMD (DPDMD), a role-separated distillation method motivated by a simple observation about the denoising process: early denoising steps mainly determine global image structure and thus play a dominant role in sample diversity, whereas later steps primarily refine local details and perceptual quality (see Figure A in the Appendix). Based on this asymmetry, DP-DMD adopts different optimization objectives to

1Widely used perceptual losses, such as LPIPS (Zhang et al., 2018) and DISTS (Ding et al., 2020), incur non-trivial GPU memory usage and computational cost, especially when applied to high-resolution imagery.

guide distinct distillation steps. The first step is trained with a teacher-derived target-prediction objective (e.g., vprediction) to preserve sample diversity, while the remaining steps are optimized using the standard DMD loss to improve image quality. To enforce this functional separation, gradients from the DMD loss are stopped (i.e., detached) at the first step, preventing the mode-seeking reverse-KL objective from overriding the diversity-preserving supervision.

A key advantage of DP-DMD is its simplicity. The method requires no perceptual or adversarial regularization, no additional modules, or teacher-generated reference images. All computations are performed in latent space, resulting in a simple, stable, and memory-efficient training pipeline. Despite its minimalist design, DP-DMD consistently improves the diversity-quality trade-off in few-step text-to-image generation across diffusion-based and flow-based backbones, supported by subjective verification.

#### 2. Related Work

Research on accelerating diffusion and flow-based generative models can be broadly grouped into two families: trajectory-based distillation, which compresses the teacher’s denoising paths, and distribution-level matching, which aligns the student distribution with that of the teacher.

Trajectory-based distillation. Trajectory-based distillation aims to compress the teacher’s denoising or transport trajectories into a student that requires substantially fewer inference steps. Representative methods include consistency models (Song et al., 2023) and their improved variants (Song & Dhariwal, 2023; Wang et al., 2024; Lu & Song, 2024), which typically select anchor points along the teacher trajectories and train the student to reproduce the corresponding transitions. More recently, MeanFlow (Geng et al., 2025) further investigates trajectory compression by matching average diffusion velocities. Although trajectorybased methods have shown strong acceleration performance, their effectiveness often degrades in extremely low-step regimes, particularly for large-scale pre-trained image and video generators, where faithfully preserving the teacher’s full trajectories becomes increasingly challenging (Zheng et al., 2025). This limitation has motivated alternative approaches that relax strict trajectory imitation and instead pursue distribution-level matching to the teacher.

Distribution-level matching. A complementary line of work distills generative models by aligning the student distribution with that of a pre-trained teacher model. One branch adopts adversarial regularization, either by introducing dedicated discriminators or by repurposing diffusion features to construct generative adversarial network (GAN)like objectives (Sauer et al., 2024a;b; Lin et al., 2024; Zhou et al., 2024a; Chadebec et al., 2025). Although these meth-

ods tend to improve visual sharpness and reduce the number of sampling steps, the added adversarial components often make optimization much more sensitive and less scalable.

Another branch is inspired by distribution-matching ideas originally developed in text-to-3D generation (Poole et al., 2022; Wang et al., 2023a), where optimization is driven by discrepancies between synthesized samples and diffusion model guidance. DMD transfers this principle to image synthesis distillation and achieves a strong balance between efficiency and sample quality (Yin et al., 2024b). Subsequent extensions further improve performance through refined objectives (Zhou et al., 2024b; Luo et al., 2025), guidance strategies (Liu et al., 2025a), or auxiliary training mechanisms (Yin et al., 2024a; Zheng et al., 2025).

Nevertheless, DMD-based methods exhibit a reverse-KLlike mode-seeking tendency, which can reduce support coverage and harm sample diversity under aggressive step reduction. Existing remedies often introduce perceptual or adversarial regularization (Yin et al., 2024b;a; Chadebec et al., 2025), which may increase complexity, destabilize training, and hinder scalability. In contrast, DP-DMD improves sample diversity through step-wise loss design, preserving diversity in the first distillation step and using later steps for standard DMD-based visual quality refinement.

#### 3. Preliminaries

We briefly review flow matching and DMD, which together form the technical foundation of the proposed DP-DMD.

##### 3.1. Flow Matching

Diffusion and flow-based generative models can be formulated in continuous-time ordinary differential equations (Chen et al., 2018; Song et al., 2020). Let x ∼ pdata denote a clean data sample and ϵ ∼ pnoise denote a noise sample, where typically pnoise ≜ N(0,I). A continuous path between data and noise is constructed as zt = atx + btϵ, for t ∈ [0,1], where at and bt are predefined schedules. Following prior work (Liu et al., 2023b; Esser et al., 2024), we adopt the linear schedule:

zt = (1 − t)x + tϵ. (1)

Under this construction, zt evolves from the data distribution at t = 0 to the noise distribution at t = 1.

The corresponding velocity field is given by the time derivative of zt. For the linear path in Equation (1), we have

dzt dt

vt =

= ϵ − x. (2)

Flow matching learns a neural velocity field vθ(zt,t) by regressing it to the target velocity along this path (Lipman

et al., 2022; Geng et al., 2025):

ℓFM(θ) = Et,x,ϵ vθ(zt,t) − (ϵ − x) 2 . (3)

At inference time, generation starts from z1 ∼ pnoise and integrates the learned ODE backward from t = 1 to t = 0: dzt

dt = vθ(zt,t). The resulting terminal state defines the generated sample: xθ = z1 + 1 0 vθ(zt,t)dt.

##### 3.2. DMD Loss

Given a pre-trained multi-step teacher and a few-step student, DMD seeks to align the student distribution pstu with the teacher distribution ptea. Let xθ = gθ(ϵ) denote a sample generated by the student, where ϵ ∼ pnoise is a noise variable and thus xθ ∼ pstu. The DMD loss is defined as

ℓDMD(θ) ≜ DKL pstu(xθ)∥ptea(xθ) . (4)

Directly evaluating this divergence in high-dimensional image space is intractable due to implicit density representations and support mismatch between two distributions (Poole et al., 2023; Wang et al., 2023a). Instead, DMD computes its gradient in a shared perturbed space. Specifically, the student sample xθ is diffused according to Equation (1) to obtain zt, which brings the teacher and student distributions into overlapping support (Yin et al., 2024b). Under this assumption, the gradient of ℓDMD(θ) with respect to the student parameters θ can be written as

θ ∇θxθ ⊺ sstu(zt) − stea(zt) , (5)

∇θℓDMD(θ) = Ex

where sstu = ∇zt

log pstu(zt) and stea = ∇zt

log ptea(zt) are the score functions of the student- and teacher-induced distributions in the perturbed space, respectively.

In practice, the teacher-side score can be obtained from the pre-trained multi-step teacher. By contrast, the score of the student-induced distribution is not explicitly parameterized by the distilled generator and is generally intractable to compute in closed form. Therefore, following prior DMD methods (Yin et al., 2024a), we approximate it using an auxiliary “fake” model2.

#### 4. Proposed Method: DP-DMD

We present DP-DMD, a role-separated distillation method for few-step visual synthesis, motivated by the observation that different portions of the denoising trajectory contribute differently to the final sample. The training pipeline is shown in Figure 2.

2The auxiliary model is initialized in the same way as the teacher with a shared architecture.

-step inference Input

Auxiliary model training Noised Auxiliary pred

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Auxiliary model

Teacher

Teacher

Aux updates per Stu update

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

1-step inference

𝑁 − 1-step inference

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Detaching

Auxiliary

Student

Student

model

[Figure 25]

[Figure 26]

[Figure 27]

First-step pred

- Figure 2. Training pipeline of DP-DMD. The student’s first denoising step is supervised by a teacher-derived intermediate

state through the target-prediction loss ℓDiv. Gradients are detached after this step to prevent the DMD loss from overriding the early diversity signal. The remaining N − 1 steps are trained with the standard DMD loss, where teacher and auxiliary model scores provide distribution-matching guidance for perceptual quality refinement. This role-separated design preserves sample diversity while maintaining high-quality few-step generation.

##### 4.1. Roles of Early and Late Denoising Steps

Diffusion and flow-based generative models exhibit a salient stage-wise denoising behavior (Wang & Vastola, 2023; Liu et al., 2023a; He et al., 2025), which is central to our design.

Early-step sample diversity preservation. Early denoising steps operate at high noise levels and are primarily responsible for establishing the coarse geometric and semantic structure of a sample, including overall composition, coarse geometry, and object presence. As shown in Figure A of the Appendix, variation introduced at this stage is carried forward by the subsequent trajectory and remains visible in the final output. These early decisions therefore play a dominant role in determining sample diversity.

Late-step perceptual quality refinement. In contrast, later denoising steps operate at lower noise levels and mainly refine local visual details, such as texture, colors, contours, and fine appearance. Since the global layout has typically been determined by the previous stage, these steps have a weaker effect on sample diversity and contribute more directly to perceptual quality.

This asymmetry suggests that applying the same distillation loss to all steps, as in standard DMD (Yin et al., 2024a;b), is not ideal for preserving sample diversity under aggressive step reduction. DP-DMD addresses this issue by explicitly separating the roles of the distillation steps: the first step is trained to preserve diverse global structure, and the subsequent steps are trained to improve visual quality.

##### 4.2. Details of DP-DMD

Consider a distilled student model with N inference steps. DP-DMD decomposes the training objective according to the role of each step.

Teacher-derived diversity supervision. We first construct a supervision target for the diversity-preserving step. Starting from the same initial noise ϵ, we run the multi-step teacher for K denoising steps and denote the resulting intermediate latent by zt˜, where t˜is the corresponding continuous-time index. The anchor step K determines which teacher state is used to guide the student’s first step and therefore controls the strength of the diversity-preserving signal.

Under the linear flow path in Equation (1), the teacherderived velocity that transports the initial noise toward zt˜ for the student’s first denoising step is

ϵ − zt˜ 1 − t˜

. (6)

v˜ =

The student predicts a velocity field vθ(ϵ,1) from the initial noise. We supervise this prediction using the diversity loss:

ℓDiv(θ) = Eϵ vθ(ϵ,1) − v˜ 2 . (7)

By aligning the first-step prediction with a teacher-derived intermediate state, this loss encourages the student to retain diverse global structure encoded by the teacher’s early denoising trajectories. Importantly, this supervision is applied only at the first step, where the teacher and student share the same input noise distribution.

Algorithm 1 DP-DMD training for flow-based models

score estimation are optimized with AdamW (Loshchilov & Hutter, 2017) using a learning rate of 10−5. We set the diversity weight to λ = 5 × 10−2, update the auxiliary model every M = 5 student updates, and use K = 5 as the default diversity anchor. Distillation is performed on DiffusionDB prompts (Wang et al., 2023b) for 6 × 103 iterations, with a per-GPU batch size of 4 on 8 NVIDIA A800 GPUs. When comparing DMD variants, we keep the training budget, CFG scale, and inference cost matched.

# x: training data batch # K: diversity anchor step index # t_tilde: anchor diffusion timestep # N: number of student sampling steps # lambda_div: diversity loss weight

eps = randn like(x) z_tilde = rollout teacher(eps, K)

# diversity supervision v_tilde = (eps - z_tilde) / (1 - t_tilde) v1 = v stu(eps, 1) loss_div = l2 loss(v1 - v_tilde)

Evaluation. The evaluation focuses on the central goal of this work: improving sample diversity without sacrificing perceptual quality (or text-image preference). For each prompt, we generate multiple samples from different initial noise and compute feature-space diversity as

# detach after first step z1 = stopgrad(rollout student(eps, 1))

# quality supervision x_theta = rollout student(z1, N - 1) loss_dmd = dmd loss(x_theta)

2 R(R − 1) i, j

cos x(θi),x(θj) , (9)

Diversity = 1 −

loss = loss_dmd + lambda_div * loss_div

DMD-based quality supervision. After the first denoising step, the student is rolled out for the remaining N − 1 steps to obtain the final sample xθ. To maintain a clean separation between diversity preservation and quality refinement, we detach the output of the first step from the computational graph before applying the DMD loss. Consequently, gradients from DMD cannot be propagated back to the diversity-preserving step. The full DP-DMD loss is

###### ℓ(θ) = ℓDMD(θ) + λℓDiv(θ), (8)

where λ controls the trade-off between the two terms. DPDMD therefore requires no perceptual loss, adversarial discriminator, additional modules, or stored teacher-generated reference images. Algorithm 1 summarizes one training iteration for flow-based models.

#### 5. Experiments

In this section, we first describe the training and evaluation protocol. We then compare DP-DMD with regularizationbased DMD variants and existing few-step methods, focusing on how different objectives affect the diversity-quality trade-off. Finally, we conduct ablation studies to analyze the design choices responsible for the observed behavior, along with additional prompt-following evaluation.

##### 5.1. Experimental Setups

Training. We use SD3.5-Medium (Esser et al., 2024) and SDXL (Podell et al., 2023) as representative flow-based and diffusion-based text-to-image teachers. Unless otherwise specified, all models are distilled at 1,024 × 1,024 resolution. The teacher classifier-free guidance (CFG) scale is fixed to 3.5 for SD3.5-M and 8.0 for SDXL. The student generator and the auxiliary “fake” model used for DMD

where R denotes the number of initial noise samples per prompt, and we set R = 9 in all diversity evaluations. We report diversity using DINOv3-ViT-Large (DINO, Sim´eoni et al. 2025) and CLIP-ViT-Large (CLIP, Radford et al. 2021). DINO emphasizes structural variation, while CLIP is more sensitive to semantic variation under the same text condition. Perceptual quality is measured with VisualQuality-R1 (VQR1, Wu et al. 2025) and MANIQA (MIQA, Yang et al.

- 2022); preference alignment is measured with ImageReward (ImgR, Xu et al. 2023) and PickScore (PicS, Kirstain et al.
- 2023). We therefore interpret results jointly rather than treating any single metric as decisive.

##### 5.2. Main Results

Controlled comparison of diversity supervision. We compare DP-DMD with two representative strategies for mitigating diversity degradation in DMD: perceptual regularization with LPIPS and adversarial regularization with a discriminator. All variants in Table 1 use the same backbone, training data, training budget, CFG scale, and NFEs.

Vanilla DMD is a strong few-step baseline because it directly aligns the final student distribution with the teacher through score differences in a perturbed space. However, this objective is reverse-KL-like and therefore tends to be mode-seeking: under a small NFEs, the student can reduce the loss by mapping different noise inputs to a narrower set of high-density teacher modes. This explains its strong quality but reduced sample diversity.

DMD-LPIPS partially alleviates this issue by adding a perceptual feature constraint. Since LPIPS measures samplelevel perceptual similarity rather than distributional support coverage, its gradients are only indirectly related to preserving the branching structure of the denoising trajectory. As a result, while DMD-LPIPS can increase apparent diversity

Table 1. Controlled comparison of DMD variants on Pick-a-Pic (Kirstain et al., 2023) and COCO-10K 2014 (Lin et al., 2014). All variants are evaluated with the same backbone and inference budget. The best and second-best results are highlighted in bold and underline, respectively.

Pick-a-Pic COCO-10K 2014 Method

ImageFree

Diversity Quality Preference Diversity Quality Preference

NFEs

DINO↑ CLIP↑ VQ-R1↑ MIQA↑ ImgR↑ PicS↑ DINO↑ CLIP↑ VQ-R1↑ MIQA↑ ImgR↑ PicS↑ SD3.5-M (CFG=3.5)

Base Model - 60 0.240 0.221 4.657 1.020 1.007 21.80 0.288 0.204 4.636 1.043 0.910 22.31 DMD ✓ 4 0.137 0.133 4.649 1.016 1.189 21.75 0.210 0.154 4.690 1.060 1.053 22.40 DMD-LPIPS ✗ 4 0.169 0.169 4.598 1.005 1.063 21.62 0.204 0.168 4.599 1.012 0.949 22.29 DMD-GAN ✗ 4 0.183 0.162 4.525 0.984 1.033 21.63 0.214 0.174 4.584 0.983 0.751 22.02 DP-DMD ✓ 4 0.179 0.182 4.646 1.017 1.142 21.76 0.250 0.182 4.689 1.032 0.988 22.41 SDXL (CFG=8.0)

Base Model - 100 0.219 0.204 4.675 1.033 1.016 21.96 0.269 0.219 4.637 1.056 0.820 22.54 DMD ✓ 4 0.109 0.133 4.667 0.971 0.951 21.68 0.139 0.143 4.643 0.982 0.712 22.21 DMD-LPIPS ✗ 4 0.136 0.139 4.610 0.976 0.883 21.74 0.181 0.137 4.723 0.984 0.729 22.38 DMD-GAN ✗ 4 0.126 0.124 4.624 1.019 1.036 21.80 0.157 0.117 4.789 1.030 0.801 22.63 DP-DMD ✓ 4 0.173 0.161 4.591 0.954 1.011 21.75 0.204 0.157 4.765 1.041 0.835 22.45

Table 2. Practical comparison of open-source few-step methods.

Pick-a-Pic COCO-10K 2014 Method

ImageFree

Diversity Quality Preference Diversity Quality Preference

NFEs

DINO↑ CLIP↑ VQ-R1↑ MIQA↑ ImgR↑ PicS↑ DINO↑ CLIP↑ VQ-R1↑ MIQA↑ ImgR↑ PicS↑ Base Model - 60 0.230 0.205 4.665 1.042 1.067 21.46 0.278 0.202 4.757 1.065 1.014 22.50 Hyper-SD ✗ 8 0.234 0.225 4.268 0.917 0.808 20.77 0.297 0.236 3.929 0.924 0.661 21.56 Flash ✗ 4 0.184 0.172 4.625 1.023 1.014 21.62 0.229 0.184 4.517 0.998 0.878 22.38 TDM ✓ 4 0.148 0.167 4.675 1.013 1.134 21.21 0.196 0.172 4.617 1.046 0.998 21.49 DP-DMD ✓ 4 0.162 0.181 4.673 1.001 1.128 21.15 0.197 0.174 4.672 1.048 1.034 22.29

in some cases, it does not reliably preserve the underlying branching structure needed for support coverage.

DMD-GAN supplies an adversarial signal that can push samples toward the natural image manifold and sometimes broaden visual variation. Nevertheless, the discriminator introduces a minimax optimization problem, making training less stable, and encouraging artifact amplification and dataset-specific shortcuts. The visual results in Figure 3 show that apparent diversity gains may coincide with degraded quality, which in turn make feature-space diversity scores less reliable.

DP-DMD avoids these drawbacks by changing the allocation of distillation losses rather than adding external regularizers. This simple design improves diversity over vanilla DMD while maintaining competitive quality and preference,

- at essentially no cost. The subjective user study in Section D of the Appendix further supports these findings.

Practical comparison with few-step methods. We further compare DP-DMD with representative open-source fewstep methods in Table 2. This comparison is not strictly controlled because the methods may differ in training data,

teacher configuration, guidance scale, optimization budget, and implementation details. It nevertheless helps assess whether the proposed design remains practical beyond the matched DMD-variant setting.

Hyper-SD (Ren et al., 2024) relies on trajectory-style compression, which preserves variation by referencing intermediate teacher states but may leave limited capacity for final-detail refinement under very small step budgets. Flash Diffusion (Chadebec et al., 2025) uses adversarial distillation, which improves sharpness but inherits the tuning sensitivity. TDM (Luo et al., 2025) matches trajectory distributions and remains image-free, but it does not explicitly reserve the earliest student step for diversity preservation, so global branching and final refinement must still be balanced within the same compressed rollout. By contrast, DP-DMD uses a simpler, image-free decomposition, maintaining competitive sample diversity, perceptual quality, and prompt consistency at low inference cost (see also Figure 4).

5.3. Ablation Studies All ablation studies are conducted on Pick-a-Pic (Kirstain et al., 2023) under the same evaluation protocol.

Half cat, Half woman, A princess, Highly detailed face, ultra-realistic, Jewelry.

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

A cat and a dog sitting on a sofa.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

(a) DMD (b) DMD-LPIPS (c) DMD-GAN (d) DP-DMD

- Figure 3. Visual comparison of diversity supervision strategies. Perceptual and adversarial regularization provide limited or less stable diversity gains and may introduce visual artifacts. In contrast, DP-DMD preserves richer prompt-conditioned variation while maintaining high perceptual quality, yielding a more favorable diversity-quality trade-off.

##### Table 3. Effect of the diversity anchor step K.

|K<br><br>|Diversity DINO↑ CLIP↑|Quality VQ-R1↑ MIQA↑<br><br>|Preference ImgR↑ PicS↑|
|---|---|---|---|
|Base<br><br>|0.137 0.133<br><br>|4.649 1.016|1.189 21.75|
|1 3 5 10 30<br><br>|0.157 0.155 0.175 0.178 0.179 0.182 0.187 0.185 0.187 0.181<br><br>|4.578 0.995 4.648 1.016 4.646 1.017 4.589 1.004 4.602 1.007|1.121 21.62 1.125 21.67 1.142 21.76 1.112 21.78 1.117 21.71<br><br>|

Diversity anchor step K. Table 3 studies where the teacher trajectory should be used to supervise the student’s first distillation step. Adding the teacher-derived target at any tested anchor improves diversity over the DMD baseline, which supports the central premise that direct early-step supervision counteracts the support-narrowing behavior of DMD. Nevertheless, anchors that are too early are close to the noise prior and thus carry only weak semantic information; anchors that are too late mix diversity-relevant structure with teacher-specific refinements that the remaining few student steps may not reproduce reliably. Moderately early anchors offer the most useful compromise.

Diversity weight λ. Table 4 varies the relative strength of the first-step diversity loss. Increasing λ strengthens

##### Table 4. Effect of the diversity weight λ.

|λ|Diversity DINO↑ CLIP↑<br><br>|Quality VQ-R1↑ MIQA↑|Preference ImgR↑ PicS↑<br><br>|
|---|---|---|---|
|Base|0.137 0.133<br><br>|4.649 1.016<br><br>|1.189 21.75|
|0.01 0.05 0.08 0.10|0.170 0.164<br><br>0.175 0.178<br><br>0.176 0.176<br><br>0.177 0.177<br>|4.672 1.003 4.648 1.016 4.662 1.014 4.662 1.001<br><br>|1.136 21.81 1.125 21.67 1.117 21.71 1.056 21.64|

the supervised connection between the initial noise and the teacher’s early trajectory, thereby resisting the tendency of the DMD loss to concentrate probability mass around highdensity, easily reproducible samples. Meanwhile, a large diversity weight can reduce the effective capacity allocated to final-image refinement. Moderate values achieve this separation most effectively.

Gradient stopping. Figure 5 verifies that role separation requires more than only two losses. Without stopping gradients after the first step, the DMD loss back-propagates through the entire student rollout, modifying the very mapping from noise to global structure of the image that the diversity loss is trying to preserve. The training curves show that this conflict appears early: the model quickly improves preference while losing diversity when the DMD

Dark-haired Valerian and redhead Laureline, time and space agents, detailed and realistic painting by Carl Bloch.

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

(a) SD3-M (60 NFEs) (b) Hyper-SD (8 NFEs) (c) Flash (4 NFEs) (d) TDM (4 NFEs) (e) DP-DMD (4 NFEs)

Figure 4. Visual comparison with open-source few-step distillation methods.

[Figure 41]

DMDDMDPreferenceDiversity -一一 w/ow/oDetachingDetachingDiversityPreference -一一 DP-DMDDP-DMDDiversityPreference

0.22

[Figure 42]

1.10

0.20

Preference

1.05

Diversity

/ 1

1 6 1 8

． oo．

4

》

，＂，

1.00

佑

，，

0.95

I

[Figure 43]

11 /

/ I

0.14

0.90

[Figure 44]

## 。

4000

1000 2000 3000 5000 6000

Training

Step

Figure 5. Effect of gradient stopping. Curves are plotted from the 100-th iteration and smoothed by moving average.

Table 5. Prompt-following evaluation on GenEval (Ghosh et al., 2023). We compare DP-DMD with the SD3.5-M teacher in terms of category-wise accuracies, with higher values indicating better prompt following.

|Method<br><br>|Overall<br><br>|Single Obj.<br><br>Two Obj.<br><br>Count. Colors Pos.<br><br>Color Attr.|
|---|---|---|
|SD3.5-M DP-DMD<br><br>|0.66 0.65|0.98 0.78 0.64 0.83 0.18 0.53<br><br>0.99 0.81 0.60 0.81 0.21 0.48<br>|

gradient reaches the first step. Detaching resolves this loss interference: the two parts of the student are optimized for complementary functions rather than competing to control the same parameters through incompatible gradients.

Prompt-following ability. We additionally evaluate compositional prompt following on GenEval (Ghosh et al., 2023). This is to ensure that diversity preservation does not merely introduce random visual variation, but maintains the semantic alignment inherited from the teacher. As shown in Table 5, DP-DMD remains close to the teacher (SD3.5M, Esser et al. 2024) overall and is particularly stable on object-presence and spatial-relation categories.

#### 6. Conclusion and Discussion

We have presented DP-DMD, a simple role-separated distillation method for fast visual synthesis. DP-DMD separates diversity preservation and quality refinement across different denoising steps within the distilled student. The first step is supervised by a teacher-derived target-prediction loss to preserve diverse global modes, while the remaining steps are optimized with the standard DMD loss for perceptual quality refinement. Gradient stopping after the first step further prevents the mode-seeking DMD signal from overriding the diversity-preserving mapping.

DP-DMD currently applies explicit diversity supervision only to the first distillation step and uses a fixed anchor step and loss weight. While this simple design is effective, it may not be optimal when diversity-relevant decisions are distributed across multiple denoising steps, or when difficult prompts, strong guidance, or model-specific dynamics cause later stages to influence global structure. A promising direction is therefore to develop adaptive role separation, where the anchor location, diversity weight, and gradient routing are selected dynamically according to timestep, prompt complexity, teacher uncertainty, or student capacity. Such a design could extend the current first-step supervision into a trajectory-aware objective that preserves global variation while still allowing later steps to specialize in visual quality.

Beyond the reported evaluations, our preliminary studies suggest an additional practical advantage: the role-separated training yields more stable optimization when distilling from checkpoints at different stages of the model development pipeline, including pre-trained, mid-trained, and post-trained models via reinforcement learning. This robustness is important for real deployment scenarios, where foundation models are frequently updated and distillation must remain reliable across heterogeneous checkpoint states. These preliminary findings suggest that DP-DMD may serve not only as a diversity-preserving objective design, but also as a stable training recipe for iterative model development.

#### Acknowledgments

This work was supported in part by the Hong Kong ITC Innovation and Technology Fund (9440379 and 9440390) and the PolyU-OPPO Joint Innovative Research Center.

#### Impact Statement

This work contributes to the development of efficient generative modeling by improving diversity preservation in fewstep model distillation. By enabling faster image synthesis while better maintaining the coverage of the teacher model distribution, the proposed method may support more practical deployment of high-quality text-to-image generators in resource-constrained or latency-sensitive settings.

All datasets used in this study are publicly available and are used in accordance with their respective licenses. As with other advances in image generation, potential downstream risks may include misuse for creating misleading, biased, or harmful visual content. These risks are not unique to the proposed method, but improved sampling efficiency may make responsible deployment practices increasingly important. We therefore encourage future use of this technique together with appropriate safeguards, such as dataset auditing, content moderation, watermarking, provenance tracking, and application-specific risk assessment.

#### References

Chadebec, C., Tasar, O., Benaroche, E., and Aubin, B. Flash diffusion: Accelerating any conditional diffusion model for few steps image generation. In Association for the Advancement of Artificial Intelligence, pp. 15686–15695, 2025.

Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, pp. 6572–6583, 2018.

Ding, K., Ma, K., Wang, S., and Simoncelli, E. P. Image quality assessment: Unifying structure and texture similarity. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(5):2567–2581, 2020.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., M¨uller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow Transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Ghosh, D., Hajishirzi, H., and Schmidt, L. GenEval: An object-focused framework for evaluating text-to-image alignment. In Advances in Neural Information Processing Systems, pp. 52132–52152, 2023.

He, X., Fu, S., Zhao, Y., Li, W., Yang, J., Yin, D., Rao, F., and Zhang, B. TempFlow-GRPO: When timing matters for GRPO in flow models. arXiv preprint arXiv:2508.04324, 2025.

Kirstain, Y., Polyak, A., Singer, U., Matiana, S., Penna, J., and Levy, O. Pick-a-Pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, pp. 36652– 36663, 2023.

Labs, B. F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., et al. FLUX. 1 Kontext: Flow Matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

- Lin, S., Wang, A., and Yang, X. SDXL-Lightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024.
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ar, P., and Zitnick, C. L. Microsoft COCO: Common objects in context. In European Conference on Computer Vision, pp. 740–755, 2014.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. In International Conference on Machine Learning, 2022.

- Liu, D., Gao, P., Liu, D., Du, R., Li, Z., Wu, Q., Jin, X., Cao, S., Zhang, S., Li, H., et al. Decoupled DMD: CFG augmentation as the spear, distribution matching as the shield. arXiv preprint arXiv:2511.22677, 2025a.
- Liu, E., Ning, X., Lin, Z., Yang, H., and Wang, Y. OMSDPM: Optimizing the model schedule for diffusion probabilistic models. In International Conference on Machine Learning, 2023a.

Liu, J., Liu, G., Liang, J., Li, Y., Liu, J., Wang, X., Wan, P., Zhang, D., and Ouyang, W. Flow-GRPO: Training flow matching models via online RL. arXiv preprint arXiv:2505.05470, 2025b.

Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023b.

Geng, Z., Deng, M., Bai, X., Kolter, J. Z., and He, K. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025.

Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In International Conference on Learning Representations, 2017.

Lu, C. and Song, Y. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024.

Lu, Y., Ren, Y., Xia, X., Lin, S., Wang, X., Xiao, X., Ma, A. J., Xie, X., and Lai, J.-H. Adversarial distribution matching for diffusion distillation towards efficient image and video synthesis. In IEEE/CVF International Conference on Computer Vision, pp. 16818–16829, 2025.

Luo, Y., Hu, T., Sun, J., Cai, Y., and Tang, J. Learning fewstep diffusion models by trajectory distribution matching. arXiv preprint arXiv:2503.06674, 2025.

Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M¨uller, J., Penna, J., and Rombach, R. SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. DreamFusion: Text-to-3D using 2D diffusion. arXiv preprint arXiv:2209.14988, 2022.

Poole, B., Jain, A., Barron, J. T., and Mildenhall, B. DreamFusion: Text-to-3D using 2D diffusion. In International Conference on Learning Representations, 2023.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021.

Ren, Y., Xia, X., Lu, Y., Zhang, J., Wu, J., Xie, P., Wang, X., and Xiao, X. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. In Advances in Neural Information Processing Systems, pp. 117340– 117362, 2024.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., and Rombach, R. Fast high-resolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, pp. 1–11, 2024a.

Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. Adversarial diffusion distillation. In European Conference on Computer Vision, pp. 87–103, 2024b.

Sim´eoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. DINOv3. arXiv preprint arXiv:2508.10104, 2025.

Song, Y. and Dhariwal, P. Improved techniques for training consistency models. In International Conference on Learning Representations, 2023.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2020.

Song, Y., Dhariwal, P., Chen, M., and Sutskever, I. Consistency models. In International Conference on Machine Learning, 2023.

Sutton, R. S., Barto, A. G., et al. Reinforcement learning. Journal of Cognitive Neuroscience, 11(1):126–134, 1999.

Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. WAN: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Wang, B. and Vastola, J. J. Diffusion models generate images like painters: an analytical theory of outline first, details later. arXiv preprint arXiv:2303.02490, 2023.

Wang, F.-Y., Huang, Z., Bergman, A., Shen, D., Gao, P., Lingelbach, M., Sun, K., Bian, W., Song, G., Liu, Y., et al. Phased consistency models. In Advances in Neural Information Processing Systems, pp. 83951–84009, 2024.

Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., and Zhu, J. ProlificDreamer: High-fidelity and diverse textto-3D generation with variational score distillation. In Advances in Neural Information Processing Systems, pp. 8406–8441, 2023a.

Wang, Z. J., Montoya, E., Munechika, D., Yang, H., Hoover, B., and Chau, D. H. DiffusionDB: A large-scale prompt gallery dataset for text-to-image generative models. In Association for Computational Linguistics, pp. 893–911, 2023b.

Wu, T., Zou, J., Liang, J., Zhang, L., and Ma, K. VisualQuality-R1: Reasoning-induced image quality assessment via reinforcement learning to rank. In Advances in Neural Information Processing Systems, pp. 88167– 88190, 2025.

Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., and Dong, Y. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, pp. 15903–15935, 2023.

Yan, H., Liu, X., Pan, J., Liew, J. H., Liu, Q., and Feng, J. PeRFlow: Piecewise rectified flow as universal plugand-play accelerator. In Advances in Neural Information Processing Systems, pp. 78630–78652, 2024.

Yang, S., Wu, T., Shi, S., Lao, S., Gong, Y., Cao, M., Wang, J., and Yang, Y. MANIQA: Multi-dimension attention network for no-reference image quality assessment. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, pp. 1191–1200, 2022.

Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., and Freeman, B. Improved distribution matching distillation for fast image synthesis. In Advances in Neural Information Processing Systems, pp. 47455–47487, 2024a.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6613–6623, 2024b.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 586–595, 2018.

Zheng, K., Wang, Y., Ma, Q., Chen, H., Zhang, J., Balaji, Y., Chen, J., Liu, M.-Y., Zhu, J., and Zhang, Q. Large scale diffusion distillation via score-regularized continuoustime consistency. arXiv preprint arXiv:2510.08431, 2025.

Zhou, M., Zheng, H., Gu, Y., Wang, Z., and Huang, H. Adversarial score identity distillation: Rapidly surpassing the teacher in one step. arXiv preprint arXiv:2410.14919, 2024a.

Zhou, M., Zheng, H., Wang, Z., Yin, M., and Huang, H. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In International Conference on Machine Learning, 2024b.

#### Appendix

- A. Derivation of the DMD Gradient For completeness, we derive the DMD gradient used in Equation (5) of the main paper. Let

pθ(x) ≜ pstu(x), q(x) ≜ ptea(x). The DMD loss is the reverse KL divergence:

ℓDMD(θ) = DKL pθ(x)∥q(x) = pθ(x)log

- pθ(x)

- q(x)

dx. (10)

We now differentiate Equation (10) with respect to θ. The interchange between differentiation and integration is valid under standard regularity assumptions, which allows the use of the Leibniz integral rule, or equivalently, the dominated convergence theorem, to write

- pθ(x)

- q(x)

dx. (11) Since the teacher density q(x) does not depend on θ, the integrand can be differentiated as

∇θℓDMD(θ) = ∇θ pθ(x)log

- pθ(x)

- q(x)

- pθ(x)

- q(x)

- pθ(x)

- q(x)

∇θ pθ(x)log

+ pθ(x)∇θ log

= ∇θpθ(x)log

+ pθ(x)∇θpθ(x) pθ(x)

- pθ(x)

- q(x)

= ∇θpθ(x)log

= ∇θpθ(x)(log pθ(x) − log q(x) + 1). (12) Substituting Equation (12) into Equation (11) gives

∇θℓDMD(θ) = ∇θpθ(x)(log pθ(x) − log q(x) + 1) dx. (13)

Following the continuity-equation view of flow-based generative modeling (Lipman et al., 2022), we derive the identity that connects ∇θpθ(x) to the velocity field induced in sample space by an infinitesimal parameter perturbation. Let the student generator be

gθ : RD×1 → RM×1, ϵ ∈ RD×1, xθ = gθ(ϵ) ∈ RM×1,

where θ ∈ RL×1 denotes the student parameters and ϵ ∼ pnoise is sampled from a fixed noise distribution. Thus xθ ∼ pθ. Consider an infinitesimal parameter perturbation along an arbitrary direction v ∈ RL×1:

θη = θ + ηv, xη = gθ

(ϵ). The Jacobian of the generator with respect to the parameters is

η

##### ∂gθ(ϵ)

∂θ ∈ RM×L. Therefore, the infinitesimal motion of the generated sample in x-space is

∇θgθ(ϵ) =

dxη dη η=0

= ∇θgθ+ηv(ϵ)v|η=0 = ∇θgθ(ϵ)v ∈ RM×1. (14)

This perturbation induces a velocity field in sample space,

noise(·|x) [∇θgθ(ϵ)v | gθ(ϵ) = x] ∈ RM×1,

uv(x) = Eϵ∼p

where pnoise(· | x) denotes the conditional distribution of noise variables that generate the sample x. The expectation is needed because gθ need not be injective: multiple noise realizations may map to the same sample. Averaging over this conditional distribution therefore defines the mean sample-space velocity at x under the parameter perturbation v. We

now derive how the density changes under this velocity field. Let φ : RM×1 → R be a smooth test function with compact support. Then ∇xφ(x) ∈ RM×1, and

d dη

d dη

Ex∼p

Eϵ∼p

=

[φ(x)]

[φ(gθ

(ϵ))]

noise

η

θη

η=0

η=0

##### noise ∇xφ gθ(ϵ) ⊺∇θgθ(ϵ)v

= Eϵ∼p

noise(·|x) [∇xφ(x)⊺∇θgθ(ϵ)v]

= Ex∼p

Eϵ∼p

θ

θ ∇xφ(x)⊺Eϵ∼p

= Ex∼p

noise(·|x) [∇θgθ(ϵ)v]

[∇xφ(x)⊺uv(x)]

= Ex∼p

θ

= pθ(x)∇xφ(x)⊺uv(x)dx. (15)

The third equality follows from the law of total expectation and the fourth equality holds because ∇xφ(x) is fixed once we condition on the generated sample location x. Also note that the dimensions in the integrand are pθ(x) ∈ R, ∇xφ(x)⊺ ∈ R1×M, and uv(x) ∈ RM×1, so the integrand is a scalar.

We continue to rewrite the right-hand side of Equation (15). Define the probability flux

##### F(x) = pθ(x)uv(x).

Since x ∈ RM×1 and uv(x) ∈ RM×1, we have F : RM×1 → RM×1. Thus the divergence of F is well defined and is given by

M

∂Fi(x) ∂xi

∇x · F(x) =

.

i=1

##### Using the product rule for divergence, we have

M

∂ ∂xi

∇x · φ(x)F(x) =

φ(x)Fi(x)

i=1

M

∂φ(x) ∂xi

∂Fi(x) ∂xi

=

Fi(x) + φ(x)

i=1

= ∇xφ(x)⊺F(x) + φ(x)∇x · F(x). (16) Rearranging Equation (16) gives

∇xφ(x)⊺F(x) = ∇x · φ(x)F(x) − φ(x)∇x · F(x). (17)

Integrating Equation (17) over a bounded domain Ω ⊂ RM yields

∇xφ(x)⊺F(x)dx =

φ(x)∇x · F(x)dx. (18) By the divergence theorem,

∇x · φ(x)F(x) dx −

Ω

Ω

Ω

φ(x)F(x)⊺n(x)dS,

∇x · φ(x)F(x) dx =

Ω

∂Ω

where n(x) is the outward unit normal on the boundary ∂Ω. The boundary term vanishes and thus can be dropped under the standard assumption that φ has compact support. Hence,

∇xφ(x)⊺F(x)dx = − φ(x)∇x · F(x)dx. (19)

Substituting F(x) = pθ(x)uv(x) into Equation (19), we obtain

∇xφ(x)⊺ (pθ(x)uv(x)) dx = − φ(x)∇x · (pθ(x)uv(x)) dx. (20)

On the other hand, differentiating the same expectation in Equation (15) through the density gives

d dη

Ex∼p

θη

d dη

=

[φ(x)]

η=0

φ(x)pθ+ηv(x)dx

= φ(x)

η=0

d dη

pθ+ηv(x)

η=0

dx. (21)

Comparing Equation (21) with Equation (20) and noting that the equality holds for every smooth compactly supported test function φ, we obtain the weak-form identity

|d dη<br><br>pθ+ηv(x)<br><br>η=0<br><br>= −∇x · (pθ(x)uv(x))|
|---|

. (22)

Equation (22) is the continuity equation induced by the parameter perturbation direction v. Finally, because v ∈ RL×1 is arbitrary, the full parameter gradient can be recovered component-wise. For the j-th parameter coordinate, choose v = ej, where ej is the j-th standard basis vector in RL×1. Then

∂pθ(x) ∂θj

(x) , (23)

= −∇x · pθ(x)ue

j

where

∂gθ(ϵ) ∂θj

gθ(ϵ) = x ∈ RM×1. (24) Thus, the compact notation

(x) = Eϵ∼p

noise(·|x) [∇θgθ(ϵ)ej | gθ(ϵ) = x] = Eϵ∼p

##### ue

noise(·|x)

j

|∇θpθ(x) = −∇x · (pθ(x)Uθ(x))|
|---|

(25)

should be interpreted component-wise: the left-hand side is a vector in RL×1, and the right-hand side collects the L divergence terms induced by the L velocity fields, i.e., Uθ(x) = [ue

(x)] ∈ RM×L. Let

(x),...,ue

1

L

φ(x) = log pθ(x) − log q(x) + 1. (26) Substituting Equations (25) and (26) into Equation (13), we obtain

∇θℓDMD(θ) = − φ(x)∇x · (pθ(x)Uθ(x)) dx. (27) We now apply integration by parts component-wise. For the j-th parameter coordinate, define

(x) ∈ RM×1. Then

##### Vj(x) = pθ(x)ue

j

∂ℓDMD(θ) ∂θj

= −

φ(x)∇x · Vj(x)dx

Ω

[∇x · (φ(x)Vj(x)) − ∇xφ(x)⊺Vj(x)] dx

= −

Ω

φ(x)Vj(x)⊺n(x)dS +

∇xφ(x)⊺Vj(x)dx, (28)

= −

∂Ω

Ω

where the second equality follows from the product rule for divergence in Equation (16) and the third equality follows from the divergence theorem. We once again assume that the boundary term in Equation (28) vanishes. Thus, for the j-th parameter coordinate, we have

∂ℓDMD(θ) ∂θj

(x)⊺∇xφ(x)dx. (29) Substituting the definition of φ(x) into Equation (29) gives

##### = pθ(x)ue

j

∂ℓDMD(θ) ∂θj

(x)⊺∇x (log pθ(x) − log q(x) + 1) dx

##### = pθ(x)ue

j

(x)⊺ (∇x log pθ(x) − ∇x log q(x)) dx, (30)

##### = pθ(x)ue

j

where the constant term disappears because ∇x1 = 0. Define the student and teacher scores as

sstu(x) = ∇x log pθ(x), stea(x) = ∇x log q(x), (31) and define their difference by

∆s(x) = sstu(x) − stea(x). (32) Then Equation (30) can be written as

∂ℓDMD(θ) ∂θj

= Ex∼p

θ

(x)⊺∆s(x) . (33)

##### ue

j

We now express the coordinate-wise velocity through the reparameterized generator using the law of total expectation:

(x)⊺∆s(x)

Ex∼p

##### ue

j

θ

⊺

∂gθ(ϵ) ∂θj

= Ex∼p

Eϵ∼p

∆s(x)

gθ(ϵ) = x

noise(·|x)

θ

⊺

∂gθ(ϵ) ∂θj

= Ex∼p

Eϵ∼p

##### ∆s(gθ(ϵ)) gθ(ϵ) = x

noise(·|x)

θ

⊺

∂gθ(ϵ) ∂θj

∆s(gθ(ϵ)) . (34)

###### = Eϵ∼p

noise

Equivalently, since xθ = gθ(ϵ),

∂ℓDMD(θ) ∂θj

= Ex

θ

⊺

∂xθ ∂θj

∆s(xθ) . (35)

Stacking Equation (35) over j = 1,...,L gives the vector form

∇θℓDMD(θ) = Ex

θ

[(∇θxθ)⊺ ∆s(xθ)]. (36)

In practice, DMD evaluates the score difference in a perturbed space: the generated sample xθ is diffused to zt, and the student and teacher scores are evaluated at zt. With a slight abuse of notation, we write

∆s(zt) = sstu(zt) − stea(zt). (37) Equation (36) then becomes

θ ∇θxθ ⊺ sstu(zt) − stea(zt) , (38) which recovers Equation (5) in the main paper.

∇θℓDMD(θ) = Ex

#### B. Observation of Early and Late Denoising Steps

Figure A visualizes the inference trajectory of SD3.5-M (Esser et al., 2024) under different noise initializations. The results reveal clear stage-wise denoising behavior. At early timesteps, where the latent remains highly noisy, the model already determines the global layout of the image, including overall composition, coarse geometry, and object identity. Differences introduced at this stage are propagated through the remaining trajectories, and lead to distinct final samples under the same text condition. This indicates that early denoising steps are closely tied to sample diversity.

By contrast, later timesteps mainly refine local appearance, such as texture, colors, contours, and fine details. These refinements improve perceptual quality but have a comparatively weaker effect on the global structure of the image. The right panel of Figure A further supports this observation: different noise initializations produce visibly different coarse structure before fine details emerge. This motivates the role-separated design of DP-DMD. The first distillation step is assigned a diversity-preserving objective that follows the teacher’s early trajectories, while the remaining steps are optimized by DMD for quality refinement.

[Figure 45]

[Figure 46]

- Figure A. Progressive denoising dynamics. SD3.5-M exhibits stage-wise generation behavior. The left panel shows one denoising trajectory from early to late timesteps. The right panel compares early denoising states from different noise initializations. Early steps establish diverse global structure, whereas later steps mainly refine local appearance.

#### C. DP-DMD for Diffusion Models

The main paper formulates DP-DMD for flow-based models. For diffusion-based teachers such as SDXL (Podell et al., 2023), which are commonly parameterized by noise prediction, we define the diversity target in the denoised latent space, i.e., the x-prediction space. Starting from the same initial noisy latent zT, we run the teacher for K denoising steps and obtain an intermediate latent zt˜. Given the teacher noise prediction ϵtea(zt˜,t˜), the teacher-implied denoised target is

√1 − αt˜ϵtea(zt˜,t˜) √αt˜

zt˜−

, (39)

x˜ =

where αt˜denotes the cumulative noise-schedule coefficient. This target is the diffusion-model analogue of the teacher-derived intermediate state in Equation (6). For the student, the first-step denoised prediction from zT is

√1 − αT ϵθ(zT,T) √αT

zT −

. (40)

xθ(zT,0) =

The diffusion-model diversity loss is then

T ∥xθ(zT,0) − x˜∥2 . (41)

ℓDiv(θ) = Ez

After this, the first-step output is detached, and the remaining N − 1 student steps are optimized with standard DMD.

#### D. Subjective User Study

We conduct a controlled subjective user study to complement the quantitative results. We randomly sample 50 prompts and recruit 10 participants with experience evaluating image-generation results. For each prompt, outputs from two methods are shown side by side in randomized spatial order under the same text condition and random-seed protocol.

Participants compare the two methods along two criteria: sample diversity and image quality. Diversity emphasizes variation in overall composition, global and local structure, and semantic attributes across multiple samples from the same prompt. Image quality focuses on semantic and statistical fidelity and naturalness. Results are aggregated as pairwise win rates over all prompts and participants.

As shown in Figure B, DP-DMD is preferred over DMD (Yin et al., 2024a), DMD-LPIPS, and DMD-GAN in sample diversity, while maintaining competitive or superior image quality. These results support the central claim that role-separated distillation mitigates diversity collapse at no perceptual cost.

#### E. Additional Qualitative Results

Figure C compares samples generated from identical prompts with different random seeds. DP-DMD produces broader variation in global layout, object appearance, and semantic attributes than other DMD-based baselines, confirming that its diversity gains correspond to meaningful sample variation rather than visual artifacts.

[Figure 47]

[Figure 48]

- Figure B. Subjective user study on sample diversity and image quality. We report pairwise win rates of DP-DMD against DMD (Yin et al., 2024a), DMD-LPIPS, and DMD-GAN, evaluated on 50 prompts and by 10 participants. The dashed line marks the 50% chance-level win rate. DP-DMD is consistently preferred for sample diversity while remaining competitive image quality.

Figure D shows samples generated by DP-DMD at 1,024 × 1,024 resolution using 4 NFEs. The results demonstrate that the proposed diversity-preserving supervision does not prevent the student from producing coherent layouts, realistic appearance, and fine-grained details under few-step inference.

A man with sunglasses.

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

A living room has a couch, a chair, and a fireplace.

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

A very cute dog wearing some type of silly hat.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

A picture of a cute anime girl.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

(a) DMD (b) DMD-LPIPS (c) DMD-GAN (d) DP-DMD

- Figure C. Sample diversity under identical prompts. Images are generated from the same text prompts and using different random seeds. DP-DMD produces richer global and semantic variation than the baselines while requiring only 4 NFEs.

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

###### Figure D. Sample quality of DP-DMD. Images are generated at 1,024 × 1,024 resolution using DP-DMD distilled from SD3.5-M (Esser et al., 2024). All samples are produced with 4 NFEs.

