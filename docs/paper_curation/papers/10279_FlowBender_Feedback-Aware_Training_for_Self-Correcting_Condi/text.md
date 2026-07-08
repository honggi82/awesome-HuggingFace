# arXiv:2606.20404v1[cs.CV]18Jun2026

## FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows

Daniel Gilo1 Sven Elflein2,3,4 Ido Sobol1 Or Litany1,2 1Technion 2NVIDIA 3University of Toronto 4Vector Institute danielgilo@cs.technion.ac.il

##### Abstract

Conditional diffusion and flow models routinely fail to satisfy the very constraints that define their task. For instance, a depth-conditioned model often produces images whose re-extracted depth disagrees with the input, even though the forward operator—the depth predictor defining the constraint—is available during both training and inference. Existing approaches generally fall into two categories: supervised models that treat the conditioning signal as a static cue and ignore alignment information at inference, and guidance-based methods that consult it through hand-tuned linear updates, typically trading fidelity to the condition against the plausibility of the generated sample. We argue that the fundamental gap in both paradigms is that the model is never trained to utilize its own alignment error. We introduce FlowBender, a closed-loop framework that treats this error as a first-class input, training the network to learn a correction policy conditioned on inference-time feedback. At each step, an unguided look-ahead pass estimates the clean signal, a task-specific deviation is computed via the forward operator, and a refinement pass consumes this signal to produce a corrected velocity. We propose several variants of FlowBender, including a gradient-based formulation for differentiable operators and a zero-order variant for non-differentiable settings such as JPEG compression. For efficient sampling, we introduce a prior-step shortcut that enables closed-loop correction at a minimal additional computational cost. Across image-to-image translation, restoration, and 3D mesh texturing, FlowBender consistently outperforms standard supervised baselines, alignmentloss-augmented training, and state-of-the-art inference-time guidance, improving fidelity and plausibility simultaneously rather than trading them against each other. Project page: https://flow-bender.github.io/.

##### 1 Introduction

Diffusion and Flow Matching (FM) models [51, 23, 34] have become the dominant paradigm for generative modeling, and a primary use-case is conditional generation: producing samples x aligned with an external signal y, e.g., a text prompt, a depth map, or a geometric constraint. Effective conditional sampling requires both fidelity to y and plausibility with respect to the target data manifold. Existing methods, however, routinely fail on the first axis: for instance, a ControlNet [69] conditioned on depth or edge maps often produces images whose re-extracted measurements disagree with the input (Fig. 3). This inconsistency occurs even though the forward operator H that relates the sample to the conditioning signal, such as a depth predictor, edge detector, or renderer, is typically available during both training and inference.

This failure mode reveals a missed opportunity. Supervised conditional models [44, 48, 47, 69] treat y as a static cue and operate as open-loop systems at inference: even when the evolving sample drifts from the constraint, and even when the operator H is available to compute a deviation signal,

Preprint.

the network has no built-in mechanism to consult this feedback and adjust its trajectory. The very alignment information that motivates the task is left on the table.

Guidance-based methods [15, 11, 5, 45] do consult H at inference, steering the trajectory using the gradient of a measurement-matching objective. However, these approaches utilize feedback only during inference, through a manually-tuned linear update rule which creates a fundamental train-test discrepancy and a delicate tuning problem: too little guidance fails to satisfy the constraint, too much pushes the trajectory off the data manifold [21, 65]. Figure 1 illustrates both failure modes on a 2D Archimedean spiral: standard conditional generation drifts across class boundaries (c), while inference-time guidance enforces the radial constraint at the cost of missing the data manifold (d).

We argue the fundamental gap in both paradigms is that the model is never trained to utilize its own alignment error. We close this gap by treating this error as a first-class input and training the model to learn a correction policy over it. Concretely, at each sampling step, the model first performs an unguided look-ahead pass to estimate the clean signal, from which we derive a task-specific deviation signal via the operator H. A second refinement pass then consumes this error alongside the standard inputs and emits a corrected update — bending the trajectory toward the conditional manifold (Fig. 1(b), Fig. 2). We refer to our framework as FlowBender – a closed-loop system that self-corrects throughout sampling.

The framework has three notable properties. First, the learned correction is not guidance with better hyperparameters: orthogonal decomposition shows that 80% of the second-pass correction energy lies orthogonal to the gradient, indicating that the model exploits feedback through a non-linear policy that scalar-weighted schemes cannot express (§5.4). Second, FlowBender does not require the error signal to be a gradient: a zero-order variant feeds the raw measurement-space error directly to the network, extending correction to non-differentiable or black-box operators, such as JPEG compression and third-party APIs, where gradient guidance is inapplicable. Third, a prior-step shortcut allows reducing inference cost to as little as N+1 evaluations for N-step sampling, nearly matching open-loop efficiency while retaining corrective benefits (§4.5).

Empirically, FlowBender consistently outperforms standard supervised training, alignment-lossaugmented training [32], and state-of-the-art inference-time guidance [45] across 3D texturing, JPEG restoration, and image translation (super-resolution, depth/edge-to-RGB). Remarkably, while our closed-loop framework is designed to enhance fidelity, it consistently improves plausibility as well (e.g., FID), in direct contrast to the fidelity–plausibility trade-off that manifests in traditional guidance approaches.

Our main contributions are:

- • We present FlowBender, a closed-loop approach for conditional diffusion and FM models that replaces hand-tuned guidance updates with a learned policy over the model’s own alignment error. The framework is architecturally agnostic and integrates with existing adapters such as ControlNet [69] and LoRA [24].
- • We include a zero-order variant that extends learned correction to non-differentiable or black-box operators, a regime where gradient-based guidance is inapplicable.
- • We propose a prior-step shortcut for closed-loop correction at near-open-loop inference cost.
- • We demonstrate simultaneous gains in fidelity and plausibility across image translation, restoration, and 3D texturing; analysis attributes these results to a learned correction policy that utilizes feedback non-linearly, in contrast to the rigid scalar-weighting of traditional guidance.

##### 2 Related Work

Conditional Sampling via Open-Loop Training. Conditional diffusion and flow-matching models typically parameterize a score function or velocity field given a static conditioning signal. For highdimensional domains like images or meshes, foundational models [47, 17, 30, 31] require massive paired data; specific controls (e.g., depth or masks) are thus often added via adapters like ControlNet [69] or LoRA [24]. Although theoretically sampling from the posterior p(x | y) [6], these models often fail to satisfy conditioning constraints in practice [22, 26, 50]. We identify a limitation in treating y as a static hint: the model lacks a mechanism to evaluate or adjust its trajectory even though alignment diagnostics are often available during both training and inference. Self-Conditioning [9] is a related technique that feeds the model’s prior-step sample prediction back into the network as

a) b) c) d)

Ground-Truth FlowBender Standard Conditional FM IT Guidance

- Figure 1: A conditional flow model is trained to sample from the 2D Archimedean spiral distribution shown in (a), which is partitioned by quadrant into four classes representing distinct radius ranges. (b–d) point colors denote the provided condition target classes. (b) FlowBender learns to internalize feedback from a radial guidance signal, achieving faithful alignment with both class constraints and the data manifold. (c) Standard conditional generation often violates class boundaries and misses the target distribution. (d) Inference-time (IT) Guidance satisfies radial constraints but drives samples off-manifold entirely. Insets showcase representative sampling trajectories for a green-class target, focusing on a specific velocity prediction at t = 0.7. The grey arrows (b, d) represent the first-pass or unconditional prediction, the purple arrow is the guidance component, and the green arrow is the final step taken. In (d), guidance dominates, pushing the final sample (large green dot) off-manifold. While the standard model (c) misses both the class and data distribution, our framework (b) learned to aggregate the components to arrive safely at the correct class within the data manifold. See Appendix A.1 for further details.

an additional condition to improve overall sample quality. However, like other recent enhanced training schemes [32, 63, 39, 16], this approach remains fundamentally open-loop: it does not provide the model with an estimate of its deviation with respect to y. By introducing alignment error as an explicit input, we transform the generation process into a closed-loop system capable of active self-correction.

Conditional Sampling via Bayesian Guidance. An alternative paradigm treats conditional sampling as Bayesian posterior inference, decomposing the conditional score into prior and likelihood terms:

log pt(y | xt). (1)

∇xt

log pt(xt | y) = ∇xt

log pt(xt) + ∇xt

While the prior is typically approximated via a pre-trained denoiser, the likelihood term is defined by the intractable integral pt(y | xt) = p(y | x1)p(x1 | xt)dx1. Paradigms for approximating the likelihood vary: Classifier Guidance [15] utilizes time-dependent classifiers, Classifier-Free Guidance (CFG) [22] leverages the difference between conditional and unconditional score estimates, and training-free methods [13, 27, 28, 11, 12, 52, 56, 68, 5, 45, 29] approximate it using distance metrics between the predicted clean signal xˆ1(xt) and the measurement y. These modular approaches rely on heuristic weighting, creating a trade-off between constraint satisfaction and sampling artifacts or divergence [21, 65]. This standard decomposition involves cascading approximations (estimated prior, point-estimate likelihood, and manual weighting), yielding a suboptimal conditional score. Unlike works that learn CFG coefficients [19, 66], we propose a general paradigm where the model utilizes alignment error as a first-class input. This enables the network to learn a complex, non-linear policy that compensates for systematic inaccuracies and more faithfully samples from the posterior.

Learned Iterative Refinement. The paradigm of training neural networks to utilize error signals traces back to learned optimizers, where a network is trained to predict weight update rules based on gradient information from a base model [4, 58, 41, 20, 42]. This approach was adapted for inverse problems to iteratively refine estimates by incorporating measurement-space reconstruction error as input [7, 1, 2, 33, 40]. In computer vision, learned iterative refinement has advanced view synthesis [18] and 3D scene reconstruction [10, 25, 35, 67, 36, 8, 37, 62, 57]. By learning to leverage feedback from their own errors, these methods improve fidelity over feed-forward baselines while exceeding the efficiency of per-scene optimization. To our knowledge, we are the first to adapt this error-feedback paradigm to conditional diffusion and flow models.

Feedback-Aware Training

Feedback Variants

∇ s

grad

### ℒ

First-Order

Pass 1 (Look-ahead)

x

x ℋ y

x

#### v

𝑡

Feedback

s err

ℛ

c

Zero-Order

Pass 2 (Reﬁne)

###### Inference Prior-Step Shortcut

stop-grad

###### s

Fetch

Feedback x Cache

vref

Update

[Figure 1]

x

ℒFA

v

u

𝑡

v ODE Step x

c

- Figure 2: FlowBender overview. (Left) Training follows a two-pass strategy: a look-ahead pass

produces a clean-signal estimate xˆ1 to compute the feedback signal st, which then conditions a second refinement pass. (Top-right) Feedback variants include first-order gradients for differentiable operators and zero-order residuals for non-differentiable or black-box settings. (Bottom-right) At inference, an optional shortcut bypasses the look-ahead pass by fetching xˆ1 from a cached estimate of the previous step, significantly reducing computational overhead.

- 3 Preliminaries: Conditional Flow Matching Models

Flow Matching (FM) [34] models a probability path pt(xt) that interpolates between a noise distribution p0(x0) ∼ N(0,I) and a target data distribution p1(x1). While the term conditional flow matching (CFM) often refers specifically to the training framework where paths are constructed relative to data points x1, we use it here to denote flow models that receive an external task-specific conditioning signal c, e.g., a corrupted image or depth map. The forward process pt(xt | x1) for t ∈ [0,1] is defined by the linear interpolation:

xt = atx1 + σtx0, (2)

where at and σt are schedule coefficients. This framework typically involves training a network vθ(xt,t,c) to approximate the vector field ut(xt | x1) = a˙tx1 + σ˙tx0 via the objective:

LFM = Et,(x

1,c),x0 ∥vθ(xt,t,c) − ut(xt | x1)∥2 . (3)

Notably, vθ provides a principled mechanism to compute a point estimate of the clean signal, xˆ1, at any step t. For instance, in optimal transport FM (at = t,σt = 1 − t), the clean estimate is xˆ1 = xt + (1 − t)vθ.

- 4 Feedback-Aware Conditional Flows

We propose FlowBender, a framework transforming generative sampling into a closed-loop system by learning to integrate alignment feedback (Fig. 2). In the following sections, we investigate various formulations of the feedback signal and describe how the model is trained to internalize them. Pseudocode for training and inference is provided in Algs. 1 and 2 (Appendix).

###### 4.1 Problem Formulation

Conditional generative modeling aims to sample from p(x | c), producing samples that satisfy two simultaneous objectives: plausibility, adhering to the target data manifold, and fidelity, maintaining consistency with the conditioning signal. Given a forward operator H accessible during training and inference, we define the observation y = H(x). The condition c includes y and potentially auxiliary signals such as text. This formulation generalizes diverse tasks; for example, H may represent a renderer for 3D texturing, a depth predictor for depth-to-RGB, or a degradation operator for image restoration. In this context, the objective is to sample from the posterior distribution p(x | c,H).

###### 4.2 The Two-Pass Feedback Loop

The core mechanism of FlowBender relies on feeding a task-specific feedback signal, derived from the model’s current alignment error, back into the network as a first-class input. However, obtaining the clean signal estimate xˆ1 required to compute this feedback st introduces a causal dependency: st must be provided as a model input, yet it depends on xˆ1, which is only available from the model’s output. We resolve this via a two-pass execution strategy where a single network vθ handles both unguided and feedback-aware regimes. In the first pass, we set st = 0 to generate a look-ahead velocity vLA and derive the initial estimate xˆ1. This estimate allows for computing deviations from target constraints to define the feedback signal st. In the second pass, the model consumes st alongside (xt,t,c) to produce the final, refined velocity vref.

###### 4.3 Feedback Design Variants

We propose several formulations for the error signal st. A key design principle is to provide st as an auxiliary input naturally supported by any model, rendering our framework architecture-agnostic.

First-Order Feedback. For differentiable H, we define an alignment loss L(H(x),y) quantifying the discrepancy between prediction and target. The first-order feedback signal is derived from the gradient of this loss. While Bayesian formulations (Eq. 1) require gradients w.r.t. xt, stability and efficiency often favor approximations w.r.t. the estimated clean signal xˆ1 [21, 45, 29]. We therefore evaluate two candidates for sgradt : ∇xtL(H(xˆ1),y) and the “shortcut” ∇xˆ1L(H(xˆ1),y). The latter omits the expensive denoiser Jacobian ∂xˆ1

∂xt , reducing memory overhead. The resulting gradient is concatenated to xt along the channel dimension, providing an explicit direction for error correction. Zero-Order Feedback. We additionally consider a derivative-free feedback signal defined by a measurement-space error operator: serrt = R(H(xˆ1),y). R is task-specific and aligns with the domain of the conditioning signal y, e.g., a pixel-wise residual for images. In this variant, the residual serrt is provided as a conditional input alongside y, forcing the model to learn a mapping from measurement-space errors to signal-space updates. By avoiding gradients through H, this approach improves efficiency and supports non-differentiable operators—such as JPEG compression or physical simulations—where gradient-based guidance is impossible. It also extends to black-box systems where gradients are unavailable, such as proprietary APIs or feature extractors.

Hybrid Feedback. Finally, we consider a composite variant that incorporates both first- and zero-

order feedback. In this configuration, the model receives the gradient sgradt and the residual serrt through their respective input channels.

###### 4.4 Training

We optimize the model parameters θ using a joint training paradigm that supports both unguided and feedback-aware modes. For a pair (x1,c) and timestep t, we generate xt following Eq. 2. First, the model computes a look-ahead estimate xˆ1 by setting the feedback input to zero (st = 0). We then compute the feedback signal st from this estimate (Section 4.3). The training objective is:

LFA = E[∥vθ(xt,t,c,sg[st]) − ut∥2],

where sg[·] denotes the stop-gradient operation. To ensure numerical stability and efficiency, we treat st as a constant input rather than differentiating through the look-ahead pass and operator H. To maintain look-ahead reliability, we randomly replace st with a null vector with probability pun, similar to the training protocol of CFG [22]. This joint training ensures the model remains accurate in the unguided regime, which is essential for generating reliable feedback at inference time.

###### 4.5 Inference

Inference follows the two-pass procedure described in Section 4.2: at each step t, an unguided lookahead pass estimates the clean signal xˆ1 to derive the feedback signal st. A subsequent refinement pass then consumes st to yield the final velocity vref used to advance the ODE. We next describe optional modifications to this procedure that enable stronger alignment and more efficient sampling. Optional CFG. Our approach enables Classifier-Free Guidance (CFG) [22] at zero marginal cost. While our framework aims to move beyond heuristic-based sampling, users desiring manual control

can elegantly leverage the unguided velocity vLA, already computed to derive the feedback signal, as the “unconditional” reference. We define the resulting velocity as: vcfg = w · vref + (1 − w) · vLA, where w modulates alignment strength (see Fig. 5 and Appendix B).

Efficient Sampling via Prior-Step Feedback Approximation. The baseline two-pass execution doubles model evaluations as each step requires both a look-ahead and a refinement pass. To alleviate this, we propose an optional shortcut that exploits the similarity of error signals across subsequent timesteps to approximate the feedback without an additional evaluation. Analysis indicates that the feedback derived from the unguided prediction at t is increasingly correlated with the feedback computed from the guided prediction of the preceding step as the trajectory approaches the clean manifold (see Fig. 6(a–b)). This suggests that the refined prediction from the previous step serves as an effective surrogate for the current unguided clean estimate, particularly in the later stages of generation. We introduce a threshold tthresh ∈ [0,1] to control this approximation; for t > tthresh, we bypass the look-ahead pass by deriving st from the cached estimate xˆprev1 from the prior step (Fig. 2, bottom-right). This threshold enables a controllable trade-off between train-test compatibility and sampling speed. While tthresh = 1 preserves the two-pass logic, tthresh = 0 collapses execution into a single-pass loop after an initial bootstrap step – requiring only N + 1 evaluations for an N-step trajectory, nearly matching vanilla sampling efficiency while retaining closed-loop corrective benefits.

##### 5 Experiments

We evaluate FlowBender by fine-tuning pre-trained models for image-to-image translation and 3D mesh texturing. We conduct these evaluations using latent flow-matching models, as they represent the current state-of-the-art, though our method fits diffusion and signal-space models as well. We compare our closed-loop approach against three established paradigms for conditional sampling:

- 1. Standard Fine-Tuning (Standard FT): The conventional single-pass supervised approach for learning a conditional velocity field. We implement this using ControlNet [69] and LoRA [24], two popular adapters designed to incorporate new conditions into pre-trained models while mitigating catastrophic forgetting.
- 2. Fine-Tuning with Alignment Loss (FT + Lalign): An augmented training strategy that incorporates an explicit measurement-consistency objective during the fine-tuning stage to improve fidelity, as proposed in ControlNet++ [32].
- 3. Inference-Time Guidance (IT Guidance): A test-time approach that applies gradient updates during sampling to enforce conditional consistency without retraining. Specifically, we compare against FlowChef [45], a recent state-of-the-art general guidance framework for FM models, as the representative baseline for this paradigm.

Experimental Protocol. Performance is evaluated across two primary axes: fidelity, defined by the alignment between the projected output H(x) and the target measurement y, and plausibility, which assesses adherence to the target data manifold (e.g., perceptual quality). To ensure a controlled comparison and isolate the impact of our feedback mechanism, primary evaluations utilize an identical number of sampling steps across all methods. We provide more extensive comparisons in Appendix B, including results for baselines with doubled sampling budgets and the incorporation of CFG. As we demonstrate, such standard test-time enhancements do not resolve the fundamental shortcomings of existing paradigms.

###### 5.1 Image-to-Image Translation

We evaluate our framework on four image-to-image translation tasks: super-resolution, depth-toRGB, edge-to-RGB, and JPEG restoration. These represent closed-form, neural-network-based, and non-differentiable forward operators H, respectively. Our setup uses Stable Diffusion 3.5 Large [17] with ControlNet [69] for conditioning. We train on the Unsplash-25K [3] dataset (20k training and 5k test images) and sample using the Euler sampler with 40 steps. The forward operators consist of an 8× downsampling kernel (SR), DepthAnythingV2 [64] (depth), HED [61] (edges), and JPEG compression (σ = 10). To assess fidelity compared to the provided condition image, we report PSNR, SSIM, and LPIPS for restoration tasks; MAE and MSE for edges; and MAE and δ1.25 for depth. Plausibility is measured using FID across all tasks. Since JPEG restoration is non-differentiable, we only compare Standard FT and our zero-order variant for this task, as other baselines and closed-loop variants require a differentiable operator.

Table 1: Image-to-Image Results. Shaded rows denote FlowBender variants. Best results in bold; second best underlined.

Super Resolution Depth-to-RGB Edge-to-RGB

Fidelity Plausibility Fidelity Plausibility Fidelity Plausibility

Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓ MAE ↓ δ1.25 ↑ FID ↓ MAE ↓ MSE ↓ FID ↓ Standard FT 34.35 96.88 0.83 3.93 0.0848 0.7882 18.21 0.0533 0.0137 13.98 FT + Lalign 35.21 97.53 0.79 4.11 0.0834 0.7933 17.57 0.0501 0.0128 14.47 IT Guidance 43.02 98.29 0.65 18.96 0.0866 0.7592 223.54 0.0416 0.0129 97.65 First-order (w.r.t. xt) 36.27 97.10 0.66 4.30 0.0747 0.8268 14.57 0.0456 0.0104 14.81 First-order (w.r.t. xˆ1) 44.07 98.96 0.33 6.45 0.0818 0.7973 15.89 0.0435 0.0123 14.97 Zero-order 39.25 98.18 0.21 3.36 0.0764 0.8187 15.70 0.0460 0.0111 14.29 Combined (w.r.t. xt) 39.95 98.25 0.21 3.40 0.0829 0.7949 15.93 0.0435 0.0097 13.91 Combined (w.r.t. xˆ1) 39.77 98.24 0.23 3.39 0.0783 0.8175 15.39 0.0408 0.0085 13.68

###### Table 2: JPEG Restoration Results.

Fidelity Plausibility Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓

Standard FT 26.29 79.45 22.24 4.35 Zero-order (Ours) 28.86 83.13 16.33 3.80

###### Table 3: Ablation of pun.

Fidelity Plausibility pun PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓

- 0.0 37.38 97.41 0.49 3.89
- 0.1 39.21 97.64 0.43 3.83
- 0.2 38.55 97.57 0.48 3.93
- 0.3 37.55 97.24 0.53 3.92

Tables 1 and 2 summarize the quantitative results across all four tasks. FlowBender’s variants consistently outperform the baselines, yielding significant gains in both fidelity and plausibility. IT Guidance exhibits a strict trade-off between fidelity and plausibility depending on hyperparameter choices. While competitive on fidelity metrics, we find that this is traded off with worse quality of generated images as indicated by higher FID and vice versa. We provide further results and discussion on this trade-off in Appendix B.1. Qualitative examples for edge and depth to RGB are provided in Fig. 3, with restoration tasks examples available in the Appendix.

###### 5.2 3D Mesh Texturing

We evaluate FlowBender on 3D mesh texturing by fine-tuning the TRELLIS-2 [59] texture transformer. Given the 3D geometry and the conditioning image y as inputs, we utilize their corresponding latents as conditions, integrate LoRA adapters [24] into all linear layers and expand the input channels to accommodate the feedback signal st. We focus on first-order feedback, concatenating st with noisy latents, avoiding the complexity of injecting zero-order residuals via DINO-based cross-attention [49]. The forward operator H is defined as the composition of the TRELLIS-2 latent decoder and its differentiable PBR renderer. Training uses 7500 Objaverse assets [14], with evaluation on 100 held-out Objaverse and 100 Toys4K assets [53]. Fidelity is measured via masked PSNR (M.PSNR, excluding background), SSIM, LPIPS, and CLIP-similarity to the conditioning image. Plausibility is assessed by evaluating 50 random renders for each asset (5,000 total images). From these, we compute FID and standard multi-view reconstruction metrics. We adopt the default TRELLIS-2 configuration (n = 12 steps, Euler sampler).

Quantitative results in Table 4 show our framework consistently outperforms all baselines. Notably, the ∇xˆ1

variant achieves the strongest performance, significantly improving over supervised baselines. While IT Guidance enhances fidelity, it fails to match FlowBender in plausibility. Qualitative comparisons (Fig. 4) illustrate that our approach recovers fine-grained details omitted by baselines. Additional visualizations, including multi-view renderings, are provided in the Appendix.

###### 5.3 Ablation Study

We evaluate the impact of key hyperparameters on the super-resolution task.

Inference-time Shortcut. We analyze the efficiency-performance trade-off controlled by tthresh (Section 4.5). As shown in Fig. 6, reducing tthresh decreases the computational cost toward n+1 NFEs, approaching the speed of vanilla sampling at lower threshold values. Although the shortcut introduces a relative performance dip, our approach consistently outperforms the Standard FT baseline even at low tthresh settings. The fidelity gap at tthresh = 0 is especially notable. These results demonstrate that

Ours (Zero-Order)

Ours (First-Order)

Ours (First-Order)

Ours (Zero-Order)

Standard FT FT + ℒalign

Standard FT FT + ℒalign Condition

Condition

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

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

- Figure 3: Qualitative comparisons. (Left) Depth-to-RGB; (right) Edge-to-RGB. Red boxes highlight conditioning inconsistencies.

Condition Image Ours ( x𝟏) Ours (x𝒕) Standard FT FT + ℒalign IT Guidance

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

ObjaverseToys4K

[Figure 27]

| |
|---|

[Figure 28]

| |
|---|

[Figure 29]

| |
|---|

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

| |
|---|

[Figure 40]

| |
|---|

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 4: 3D Texturing Results. Objaverse (rows 1–2) and Toys4K (3–4) assets. The leftmost column provides the input condition image; remaining columns show generated 3D textured assets rendered from corresponding viewpoints. Red boxes highlight conditioning inconsistencies.

Condition 𝑤 = 1 𝑤 = 3 𝑤 = 5

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

GT

- Figure 5: Effect of Optional CFG. Increasing guidance strength w can enhance condition fidelity. Insets show re-extracted edge maps.

[Figure 55]

while our primary approach utilizes a two-pass logic, the framework can provide substantial corrective benefits even with minimal computational overhead compared to standard open-loop sampling.

Null Feedback Probability. The parameter pun allocates the training budget between unguided and feedback-aware iterations. Conceptually, a non-zero pun supports the model’s ability to generate reliable look-ahead estimates, which serve as the foundation for the feedback signal. However, increasing this probability limits the iterations available for learning the second-pass refinement.

- Table 3 shows that pun = 0.1 yields the best results across all metrics, providing the optimal balance for the closed-loop system.

- Table 4: Mesh Texturing Results. Metrics evaluate single-view fidelity to the conditioning image and multi-view (MV) plausibility of the resulting 3D texture across Objaverse and Toys4K datasets.

Shaded rows denote FlowBender variants. Best results in bold; second best underlined.

Fidelity Plausibility Method M. PSNR ↑ SSIM ↑ LPIPS ↓ CLIP ↑ MV-M.PSNR ↑ MV-SSIM ↑ MV-LPIPS ↓ MV-CLIP ↑ FID ↓ Objaverse Dataset

- Pre-Trained 19.20 0.978 0.0221 0.961 19.91 0.987 0.0138 0.961 11.03

- IT Guidance 25.86 0.986 0.0191 0.973 22.80 0.989 0.0123 0.968 9.10 Standard FT 21.91 0.981 0.0189 0.969 23.05 0.990 0.0115 0.970 8.74

- FT + Lalign 22.63 0.983 0.0182 0.970 22.48 0.989 0.0117 0.969 8.94

- Ours (w.r.t. xt) 25.26 0.985 0.0157 0.979 25.91 0.991 0.0105 0.974 7.41

- Ours (w.r.t. xˆ1) 26.39 0.987 0.0140 0.983 26.53 0.992 0.0098 0.977 6.64 Toys4K Dataset

Pre-Trained 20.36 0.984 0.0182 0.965 20.63 0.989 0.0129 0.966 10.66 IT Guidance 26.82 0.990 0.0137 0.982 23.63 0.992 0.0112 0.972 8.43 Standard FT 23.00 0.986 0.0160 0.975 24.38 0.991 0.0114 0.972 8.87 FT + Lalign 23.05 0.986 0.0159 0.976 23.25 0.991 0.0111 0.973 8.30 Ours (w.r.t. xt) 26.54 0.990 0.0129 0.980 27.20 0.993 0.0097 0.977 7.20

- Ours (w.r.t. xˆ1) 27.26 0.991 0.0116 0.984 27.21 0.993 0.0092 0.978 6.66

4.2

e) 2n

a)

b)

c)

d)

gradgrad↑s,scos()t−t1

1.0

80

errerr−↓ ss 1−tt1

4.0

- 0.4

38

↑PSNR

↓NFEs

↓FID

0.5

3.8

60

0.2

36

3.6

0.0

n + 1

3.4

40

0.0

−0.5

34

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

0.00 0.25 0.50 0.75 1.00

t

t

tthresh

tthresh

tthresh

Zero-order First-order Standard FT

- Figure 6: Prior-Step Shortcut Analysis. (a–b) Temporal similarity of feedback signals for zero-order (a) and first-order (b) variants; rising correlation as t → 1 motivates the tthresh-controlled shortcut strategy. (c–d) FID and PSNR vs. tthresh; our method maintains a consistent advantage over Standard FT (dashed) even at low tthresh values. (e) Computational cost (NFEs) as a function of tthresh, where n is the number of sampling steps.

###### 5.4 Is FlowBender Just Gradient Guidance in Disguise?

We investigate whether closed-loop training simply automates hyperparameter tuning for linear guidance. Evaluating 180 velocity predictions across 20 assets in the texturing task (∇xˆ1

variant),

we decompose the learned correction ∆v = vref − vLA relative to the gradient signal sgradt . The squared norm of the parallel component accounts for approximately 20% of the total correction’s energy; the majority of the correcting update resides in the component orthogonal to the gradient. This relationship remains stable across noise levels (t ∈ [0.1,0.9]). While a cosine similarity of

cos(∆v,sgradt ) = 0.42 ± 0.11 confirms the gradient is utilized, the dominant orthogonal component proves that FlowBender internalizes the gradient as a high-dimensional feature to non-linearly bend the flow toward the manifold, transcending the strict additive formulation of traditional Bayesian guidance.

##### 6 Conclusion and Limitations

We presented FlowBender, a framework that addresses the fundamental "open-loop" limitation of existing conditional flow models by treating their own alignment errors as first-class inputs, thus training them to correct (“bend”) their initial predictions. By replacing hand-tuned guidance with a learned, non-linear two-pass correction policy, FlowBender simultaneously enhances both conditional fidelity and sample plausibility. Our approach significantly improves over supervised and inferencetime baselines across a diverse range of tasks, including image-to-image translation, restoration, and 3D mesh texturing.

Despite these gains, certain limitations remain. First, while our prior-step shortcut enables efficient inference, the training phase requires an additional model evaluation per iteration to derive the feedback signal, increasing the computational budget for fine-tuning. Future work could investigate training schemes that directly utilize cached prior-step predictions, potentially restoring singlepass efficiency throughout the entire pipeline. Second, performance sometimes further improves when using CFG alongside FlowBender (Sec. 4.5), suggesting the learned policy has not yet fully internalized the most complex conditioning nuances. We believe that more expressive feedbackintegration architectures and large-scale training are promising directions for closing this gap.

##### 7 Acknowledgments

Or Litany acknowledges support from the Israel Science Foundation (grant 624/25) and the Azrieli Foundation Early Career Faculty Fellowship. This research was also supported in part by an academic gift from Meta. The authors gratefully acknowledge this support. This research was supported by the Council for Higher Education in Israel under the Moonshot Project.

##### References

- [1] Jonas Adler and Ozan Öktem. Solving ill-posed inverse problems using iterative deep neural networks. Inverse Problems, 33(12):124007, 2017.
- [2] Jonas Adler and Ozan Öktem. Learned primal-dual reconstruction. IEEE transactions on medical imaging, 37(6):1322–1332, 2018.
- [3] Zahid Ali, Chesser Luke, and Carbone Timothy. Unsplash. https://github.com/unsplash/ datasets, 2023.
- [4] Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, and Nando De Freitas. Learning to learn by gradient descent by gradient descent. Advances in neural information processing systems, 29, 2016.
- [5] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 843–852, 2023.
- [6] Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, and Christian Etmann. Conditional image generation with score-based diffusion models. arXiv preprint arXiv:2111.13606, 2021.
- [7] Joao Carreira, Pulkit Agrawal, Katerina Fragkiadaki, and Jitendra Malik. Human pose estimation with iterative error feedback. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4733–4742, 2016.
- [8] Tianyu Chen, Wei Xiang, Kang Han, Yu Lu, Di Wu, Gaowen Liu, and Ramana Rao Kompella. Gifsplat: Generative prior-guided iterative feed-forward 3d gaussian splatting from sparse views. arXiv preprint arXiv:2602.22571, 2026.
- [9] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. arXiv preprint arXiv:2208.04202, 2022.
- [10] Yun Chen, Jingkang Wang, Ze Yang, Sivabalan Manivasagam, and Raquel Urtasun. G3r: Gradient guided generalizable reconstruction. In European Conference on Computer Vision, pages 305–323. Springer, 2024.
- [11] Hyungjin Chung, Jeongsol Kim, Michael T Mccann, Marc L Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. arXiv preprint arXiv:2209.14687, 2022.
- [12] Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. Advances in Neural Information Processing Systems, 35:25683–25696, 2022.

- [13] Giannis Daras, Hyungjin Chung, Chieh-Hsin Lai, Yuki Mitsufuji, Jong Chul Ye, Peyman Milanfar, Alexandros G Dimakis, and Mauricio Delbracio. A survey on diffusion models for inverse problems. arXiv preprint arXiv:2410.00083, 2024.
- [14] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36:35799–35813, 2023.
- [15] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.
- [16] Noam Elata, Hyungjin Chung, Jong Chul Ye, Tomer Michaeli, and Michael Elad. Invfusion: Bridging supervised and zero-shot diffusion for inverse problems. arXiv preprint arXiv:2504.01689, 2025.
- [17] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [18] John Flynn, Michael Broxton, Paul Debevec, Matthew DuVall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and Richard Tucker. Deepview: View synthesis with learned gradient descent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2367–2376, 2019.
- [19] Alexandre Galashov, Ashwini Pokle, Arnaud Doucet, Arthur Gretton, Mauricio Delbracio, and Valentin De Bortoli. Learn to guide your diffusion model. arXiv preprint arXiv:2510.00815, 2025.
- [20] James Harrison, Luke Metz, and Jascha Sohl-Dickstein. A closer look at learned optimization: Stability, robustness, and inductive biases. Advances in neural information processing systems, 35:3758–3773, 2022.
- [21] Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, et al. Manifold preserving guided diffusion. arXiv preprint arXiv:2311.16424, 2023.
- [22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.
- [23] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.
- [24] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1

(2):3, 2022.

- [25] Gyeongjin Kang, Seungtae Nam, Seungkwon Yang, Xiangyu Sun, Sameh Khamis, Abdelrahman Mohamed, and Eunbyung Park. ilrm: An iterative large 3d reconstruction model. arXiv preprint arXiv:2507.23277, 2025.
- [26] Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. Advances in Neural Information Processing Systems, 37:52996–53021, 2024.
- [27] Bahjat Kawar, Gregory Vaksman, and Michael Elad. Snips: Solving noisy inverse problems stochastically. Advances in neural information processing systems, 34:21757–21769, 2021.
- [28] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. Advances in neural information processing systems, 35:23593–23606, 2022.
- [29] Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye. Flowdps: Flow-driven posterior sampling for inverse problems. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12328–12337, 2025.

- [30] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.
- [31] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.
- [32] Ming Li, Taojiannan Yang, Huafeng Kuang, Jie Wu, Zhaoning Wang, Xuefeng Xiao, and Chen Chen. Controlnet++: Improving conditional controls with efficient consistency feedback: Project page: liming-ai. github. io/controlnet_plus_plus. In European Conference on Computer Vision, pages 129–147. Springer, 2024.
- [33] Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, and Dieter Fox. Deepim: Deep iterative matching for 6d pose estimation. In Proceedings of the European conference on computer vision (ECCV), pages 683–698, 2018.
- [34] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.
- [35] Yueh-Cheng Liu, Lukas Höllein, Matthias Nießner, and Angela Dai. Quicksplat: Fast 3d surface reconstruction via learned gaussian initialization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 27851–27861, 2025.
- [36] Yueh-Cheng Liu, Jozef Hladk`y, Matthias Nießner, and Angela Dai. Diff3r: Feed-forward 3d gaussian splatting with uncertainty-aware differentiable optimization. arXiv preprint arXiv:2604.01030, 2026.
- [37] Wei Long, Haifeng Wu, Shiyin Jiang, Jinhua Zhang, Xinchun Ji, and Shuhang Gu. Idesplat: Iterative depth probability estimation for generalizable 3d gaussian splatting. arXiv preprint arXiv:2601.03824, 2026.
- [38] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.
- [39] Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski. Readout guidance: Learning control from diffusion features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8217–8227, 2024.
- [40] Wei-Chiu Ma, Shenlong Wang, Jiayuan Gu, Sivabalan Manivasagam, Antonio Torralba, and Raquel Urtasun. Deep feedback inverse problem solver. In European conference on computer vision, pages 229–246. Springer, 2020.
- [41] Luke Metz, Niru Maheswaranathan, C Daniel Freeman, Ben Poole, and Jascha Sohl-Dickstein. Tasks, stability, architecture, and compute: Training more effective learned optimizers, and using them to train themselves. arXiv preprint arXiv:2009.11243, 2020.
- [42] Luke Metz, James Harrison, C Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, et al. Velo: Training versatile learned optimizers by scaling up. arXiv preprint arXiv:2211.09760, 2022.
- [43] Jacob Munkberg, Jon Hasselgren, Tianchang Shen, Jun Gao, Wenzheng Chen, Alex Evans, Thomas Müller, and Sanja Fidler. Extracting triangular 3d models, materials, and lighting from images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8280–8290, 2022.
- [44] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
- [45] Maitreya Patel, Song Wen, Dimitris N Metaxas, and Yezhou Yang. Flowchef: Steering of rectified flow models for controlled generations. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15308–15318, 2025.
- [46] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.

- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.
- [48] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 conference proceedings, pages 1–10, 2022.
- [49] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.
- [50] Ido Sobol, Chenfeng Xu, and Or Litany. Zero-to-hero: Enhancing zero-shot novel view synthesis via attention map filtering. Advances in Neural Information Processing Systems, 37: 30522–30553, 2024.
- [51] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. pmlr, 2015.
- [52] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023.
- [53] Stefan Stojanov, Anh Thai, and James M Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1798–1808, 2021.
- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.
- [55] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. https://github.com/huggingface/ diffusers, 2022.
- [56] Yinhuai Wang, Jiwen Yu, and Jian Zhang. Zero-shot image restoration using denoising diffusion null-space model. arXiv preprint arXiv:2212.00490, 2022.
- [57] Jing Wen, Alexander G Schwing, and Shenlong Wang. Life-gom: Generalizable human rendering with learned iterative feedback over multi-resolution gaussians-on-mesh. In 13th International Conference on Learning Representations, ICLR 2025, pages 40453–40472. International Conference on Learning Representations, ICLR, 2025.
- [58] Olga Wichrowska, Niru Maheswaranathan, Matthew W Hoffman, Sergio Gomez Colmenarejo, Misha Denil, Nando Freitas, and Jascha Sohl-Dickstein. Learned optimizers that scale and generalize. In International conference on machine learning, pages 3751–3760. PMLR, 2017.
- [59] Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, et al. Native and compact structured latents for 3d generation. arXiv preprint arXiv:2512.14692, 2025.
- [60] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.
- [61] Saining Xie and Zhuowen Tu. Holistically-nested edge detection. In Proceedings of the IEEE international conference on computer vision, pages 1395–1403, 2015.
- [62] Haofei Xu, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Resplat: Learning recurrent gaussian splats. arXiv preprint arXiv:2510.08575, 2025.

- [63] Yifeng Xu, Zhenliang He, Shiguang Shan, and Xilin Chen. Ctrlora: An extensible and efficient framework for controllable image generation. arXiv preprint arXiv:2410.09400, 2024.
- [64] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth Anything V2. pages 21875–21911. doi: 10.52202/079017-0688.
- [65] Haotian Ye, Haowei Lin, Jiaqi Han, Minkai Xu, Sheng Liu, Yitao Liang, Jianzhu Ma, James Zou, and Stefano Ermon. Tfg: Unified training-free guidance for diffusion models. Advances in Neural Information Processing Systems, 37:22370–22417, 2024.
- [66] Shai Yehezkel, Omer Dahary, Andrey Voynov, and Daniel Cohen-Or. Navigating with annealing guidance scale in diffusion space. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, pages 1–11, 2025.
- [67] Ahmet Burak Yildirim, Tuna Saygin, Duygu Ceylan, and Aysegul Dundar. Geofusionlrm: Geometry-aware self-correction for consistent 3d reconstruction. In Computer Graphics Forum, page e70325. Wiley Online Library, 2026.
- [68] Bingliang Zhang, Wenda Chu, Julius Berner, Chenlin Meng, Anima Anandkumar, and Yang Song. Improving diffusion inverse problem solving with decoupled noise annealing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 20895–20905, 2025.
- [69] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.

##### A Implementation Details

ALGORITHM 1: FEEDBACK-AWARE TRAINING

Require: Dataset D, model vθ, prob. pun

- 1: while not converged do
- 2: Sample (x1, c) ∼ D where y ∈ c
- 3: x0 ∼ N(0, I), t ∼ U[0, 1]
- 4: xt ← atx1 + σtx0
- 5: vLA ← vθ(xt, t, c, 0) // Pass 1: Unguided Look-ahead
- 6: xˆ1 ← EstimateSample(xt, t, vLA)
- 7: st ← Feedback(xˆ1, y) // See Sec. 4.3
- 8: if rand(0, 1) < pun then // Conditioning Dropout
- 9: sin ← 0
- 10: else
- 11: sin ← stop_grad(st)
- 12: vref ← vθ(xt, t, c, sin) // Pass 2: Refinement
- 13: LFA ← ∥vref − ut∥2
- 14: Update θ via ∇θLFA

ALGORITHM 2: FEEDBACK-AWARE INFERENCE

Require: vθ, condition c (incl. y), tthresh, ODE Solver

- 1: x0 ∼ N(0, I)
- 2: xˆprev1 ← 0
- 3: for i = 0 to N − 1 do
- 4: t ← ti
- 5: if t ≤ tthresh or i = 0 then // Pass 1: Unguided Look-ahead
- 6: vLA ← vθ(xt, t, c, 0)
- 7: xˆ1 ← EstimateSample(xt, t, vLA)
- 8: else // Shortcut: Prior-Step Reuse
- 9: xˆ1 ← xˆprev1
- 10: st ← Feedback(xˆ1, y)
- 11: vref ← vθ(xt, t, c, st) // Pass 2: Refinement
- 12: xˆprev1 ← EstimateSample(xt, ti, vref) // Cache Update
- 13: xt+1 ← Step(xt, vref, t) // Integration
- 14: return xN

###### A.1 2D Toy Experiment

Ground-Truth Distribution. The target distribution is a 2D Archimedean spiral with finite thickness, defined by the curve

r = aθ, θ ∼ Uniform(0,2π), a = 2.0

The finite thickness of is modeled by an isotropic Gaussian spread around each point on the curve (σ = 0.12).

Each point is assigned a quadrant label c ∈ {0,1,2,3} based on which quarter-turn of the spiral it falls in (c = ⌊θ/(π/2)⌋).

Model Architecture. The generative model is an MLP with 3 linear layers, hidden dimension 64, and SiLU activations between layers. The input is a concatenation of the noisy sample xt ∈ R2 with a time embedding. The scalar t ∈ [0,1] is lifted to a 16-dimensional sinusoidal embedding [54]. The class condition is encoded via a learned embedding table that maps the quadrant label c to a vector of the hidden dimension, which is added to the time embedding before being passed to the main network.

Training. All models are trained with the standard conditional flow matching objective [34] for 50,000 steps with a batch size of 1024 and a learning rate of 3 × 10−4 (AdamW optimizer). Training

[Figure 56]

[Figure 57]

[Figure 58]

a) b) c)

- Figure 7: Effect of guidance scale on conditional generation. We compare guidance scales of a)

- 0.5, b) 2.0 (used in Fig. 1), and c) 4.0. Small scale (a) fails to enforce the condition, while large guidance (c) pushes samples off the data manifold entirely. We therefore selected an intermediate scale that provides a reasonable trade-off between fidelity and plausibility.

samples are generated on the fly at each iteration. Prior to training, a z-score normalizer is fitted on 200,000 spiral samples (per-coordinate mean and standard deviation). All inputs and model outputs are normalized using this transformation.

Feedback and Guidance objective The spiral is divided into four quadrants, each assigned a non-overlapping arc-length band. For quadrant c, the target radius interval is:

rminc = cπ, rmaxc = (c + 1)π The loss penalises points outside this band:

L(x,c) = softplus(rminc − r) + softplus(r − rmaxc ), r = |x|2

This is zero (up to softplus smoothing) when r ∈ [rminc ,rmaxc ], and grows linearly outside.

Guidance Scale. As illustrated in Fig. 1, Training-Free Guidance tends to push the final samples off the data manifold. We explore different guidance scales in Fig. 7. Small guidance scales fail to enforce the conditioning signal, while large guidance scales lead to samples that deviate from the target manifold.

###### A.2 Image-to-Image Translation

We use the official Stable Diffusion 3.5 (SD3.5) checkpoint implemented in the diffusers [55] library and adapt their reference script to add and train the ControlNets 1.

The model is conditioned on c = (y,ctext) where y is the image-shaped condition (e.g., edges) and ctext is the text condition. In the following, we omit the dependence on the text condition ctext for clarity.

Zero-order. To inject zero-order information, we expand the input convolution of the ControlNet and concatenate the residual to the condition along the channel dimension. Specifically, the ControlNet module CN computes the condition in the unguided pass from the image condition y as

ˆc = CN([y,0]), y,0 ∈ R3×H×W where [·] denotes concatenation along the channel dimension. We then compute the residual via

vLA = vθ(xt,t,ˆc) xˆ1 = EstimateSample(xt,t,vLA) ˜st = |y − H(xˆ1)|

1https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sd3.py

where H is the composition of the VAE decoder and the pixel-space forward operator. We normalize the feedback signal using

###### st = ˜st/max(|s˜t|)

Finally, the ControlNet encodes the guidance signal ˜c = CN([y,st]) and the refined velocity is

###### vref = vθ(xt,t,˜c).

First-order. For our first-order variant, we double the number of input channels of the input convolution of the DiT to inject the feedback signal. In the unguided pass, the denoising network vθ predicts

vLA = vθ ([xt,0],t,˜c), xt,0 ∈ R16×h×w where h,w are the height and width of the VAE-encoded image latents and ˜c = CN(y). We then compute the guidance signal ˜st = ∇xtL where

xˆ1 = EstimateSample(xt,t,vLA)

L = MSE(y,H(xˆ1)))

We standardize the feedback signal

st = (˜st − mean(s˜t))/std(s˜t) before computing the refined velocity as

###### vref = vθ([xt,st],t,˜c).

Inspired by Patel et al. [45], we additionally explore a variant that uses ˜st = ∇xˆ1L which avoids differentiating through the denoising network vθ.

Training We center crop and resize images of the Unsplash-25K [3] dataset to 1,0242 resolution and create text conditions for images using Florence2-Large [60]. Training uses batch size 16 for 5 epochs, resulting in a total of 6,250 optimizer steps, using the AdamW [38] optimizer with learning rate 10−5, weight decay 0.01, and 500 steps of learning rate warmup. Similar to Zhang et al. [69], the weights corresponding to the additional channels, used to inject feedback information, are zeroinitialized such that the network is slowly adapted to make use of feedback during training. We drop the feedback condition during training with probability pun = 0.1 following our ablation study in the main text (Tab. 3).

We only train newly added parameters, specifically the ControlNet, while keeping weights of the StableDiffusion3.5 base model frozen. In the first-order and combined variants, we additionally unfreeze the input convolution to the DiT [46] to allow the model to make use of the feedback signal that is concatenated to the input latent.

All methods use NVIDIA A100 GPUs for training where zero-order experiments take ≈ 51 GPU hours, first-order experiments take ≈ 61 GPU hours, and Standard FT and FT+Lalign runs take ≈ 38 and ≈ 48 GPU hours, respectively.

Inference. We use 40 Euler steps with full two-pass execution at every step (tthresh = 1). No CFG is applied in the main text comparisons (guidance scale 1.0), consistent with all other evaluated methods.

Baselines. Standard FT. Baseline fine-tuning variants use the vanilla ControlNet setup with zeroconvolutions to slowly expose the network to the condition during training.

FT + Lalign. The variants with Lalign adopt the set up in Li et al. [32] with loss weight λalign = 0.5 for the consistency objective in addition to the standard flow matching loss. We use the same forward operator used in our other experiments to compute the consistency loss. Further, similarly to Li et al. [32], the loss is only applied on timesteps t > tmin such that the consistency loss is not computed on low-quality point estimates during the initial denoising steps. While rectified flow models produce stable estimates relatively early in the denoising chain, we find in Table 5 that higher tmin leads to better results, similar to the findings in Li et al. [32].

IT Guidance. Inference-time guidance methods directly update the evolving sample using gradient descent on the measurement-space distance function L = MSE(H(xˆ1),y). We follow the FlowChef approach [45] to avoid differentiating through the denoising model by computing the gradient w.r.t. xˆ1 instead of xt.

###### A.3 3D Mesh Texturing

Data Preparation. We follow the original TRELLIS-2 data preparation pipeline [59] with two notable modifications. First, conditioning images are rendered using TRELLIS-2’s differentiable PBR renderer rather than Blender. This ensures operator consistency: the ground-truth observation y in the dataset is produced by the exact same forward operator H that computes the feedback term during training and inference. Second, we use a fixed lighting for all assets to aid convergence in our relatively small-data regime.

Architecture. We adapt the TRELLIS-2 texture flow model, a sparse DiT operating on 32-channel PBR texture latents, by attaching LoRA adapters (rank 128, α = 128, dropout 0.05) to all linear layers. To inject the first-order gradient feedback st, we expand the model’s input projection from 2d to 3d channels, where d = 32 is the latent dimension, zero-initializing the additional block so the network initially behaves identically to the unguided baseline. In the look-ahead pass the gradient slot is set to 0; the refinement pass receives st.

Feedback Signal. We use the first-order variant with gradients ˜st taken with respect to either xt or the clean estimate xˆ1. The forward operator H is the composition of the frozen TRELLIS-2 PBR texture decoder and the differentiable split-sum PBR renderer (nvdiffrec [43]), and L is the sum-reduced MSE between the rendered output and the conditioning image. The signal is normalized per-sample by its standard deviation,

###### st = ˜st / std(s˜t),

Training. We fine-tune the LoRA layers and the expanded input layer on 7,500 Objaverse assets, filtered by aesthetic score ≥ 4.5 and a maximum token count of 8,192 sparse voxels. Training runs for 25,000 steps with a batch size of 16. We use AdamW [38] with learning rate 10−4, weight decay 0.01, and β = (0.9, 0.95). Training uses bfloat16 automatic mixed precision with an EMA rate of 0.9999. Gradients are clipped adaptively at the 95th percentile with maximum norm 1.0. The null-feedback probability is pun = 0.1. Image conditioning uses a DINOv3-ViT-L/16 feature extractor [49] at resolution 512. Training takes approximately 34 hours on 4 NVIDIA RTX PRO 6000 GPUs.

Inference. We use the default TRELLIS-2 sampler: 12 Euler steps with full two-pass execution at every step (tthresh = 1). No CFG is applied in the main text comparisons (guidance scale 1.0), consistent with all other evaluated methods.

Baselines. Standard FT uses identical LoRA architecture (rank 128, α = 128, dropout 0.05) and the same training configuration as our method, using the standard flow matching objective. CFG dropout (for results shown in Tables 10, 11) is set to 0.1.

FT + Lalign extends Standard FT with an additional render consistency loss, following the ControlNet++ [32] protocol. The loss weight is λ = 5 × 10−3 and the loss is applied only for t > 0.3, where the clean-signal estimate is sufficiently reliable; all other hyperparameters are shared with Standard FT.

IT Guidance (FlowChef [45]) is applied directly to the pretrained TRELLIS-2 model without any fine-tuning. At each of the 12 ODE steps, one SGD update with learning rate (s′) 0.1 is applied to the noisy latent xt using the MSE render loss gradient, following the official implementation defaults.

Results for baselines with additional configurations, including number of sampling steps, CFG, consistency loss weight λ, and guidance learning rate s′ can be seen in Tables 10, 11.

Table 5: Ablation of tmin parameter in Lalign baseline on depth task. Fidelity Plausibility

tmin MAE ↓ δ1 ↑ FID ↓ 0.8 0.0834 0.7933 17.57 0.5 0.0799 0.7854 21.31

##### B Additional Results

###### B.1 Image-to-Image Translation

We provide additional qualitative comparisons for the JPEG restoration task in Fig. 8, depth-to-RGB in Fig. 9, and super resolution in Fig. 10. Tables with full results and additional settings, such as number of sampling steps, Classifier-Free Guidance, and hyperparameter choices in Table 7, Table 9, Table 8, and Table 6.

Extended Experimental Analysis. While the main text evaluates the configurations marked with (∗), this section provides a comprehensive comparison across varying sampling budgets and guidance strengths.

For open-loop baselines, standard test-time enhancement techniques fail to resolve the fundamental fidelity and plausibility shortcomings. Doubling the sampling budget (2× steps) yields negligible improvements in fidelity metrics. Similarly, applying CFG to these baselines often degrades plausibility (FID) without significantly bridging the fidelity gap. These results confirm that open-loop failures cannot be resolved with standard test-time enhancements.

In contrast, our proposed CFG scheme (Sec. 4.5) often boosts FlowBender’s fidelity. For example, as shown in Table 6, w = 3.0 guidance improves our zero-order variant’s PSNR by 6.0 dB, compared to a negligible 0.3 dB for the baseline.

Inference-time guidance (IT Guidance) shows a strict trade-off between plausibility and fidelity across all tasks depending on hyperparameter choices while our method consistently achieves the best of both worlds.

FlowChef limitations. While Patel et al. [45] show that their inference-time guidance scheme works for simple forward operators such as inpainting and super-resolution, we find that FlowChef does not generalize to complex forward operators such as neural networks. To validate this, we performed a dense sweep over key hyperparameters, including learning rate λ ∈ {1 × 10−5, 2.5 × 10−5, 5×10−5, 7.5×10−5, 1×10−4, 2.5×10−4, 5×10−4, 1×10−3, 5×10−3, 1×10−2, 2× 10−2, 5×10−2, 1×10−1, 5×10−1, 1.0}, the percentage of steps applying guidance (80%,100%), and total number of steps (40,80,100). Despite these extensive attempts, we could not find any setting that resulted in both satisfactory visual quality and adherence to the conditioning for the Edge and Depth tasks. We provide qualitative examples of these failures in Fig. 11.

###### B.2 3D Mesh Texturing

We present extended results in Tables 10 and 11, additional qualitative comparisons in Figure 12, and multi-view visualizations in Figure 13.

The patterns observed in image-to-image tasks (Appendix B.1) persist in the 3D domain (Tables 10 and 11). For open-loop baselines, standard enhancements like doubling sampling steps or applying CFG prove suboptimal or even detrimental; for example, applying CFG (w = 3.0) to Standard FT on Toys4K leads to a 1.57 dB drop in M.PSNR and a significant plausibility collapse (FID increases from 8.87 to 10.30). While our proposed CFG scheme for FlowBender (Sec. 4.5) provides further fidelity and plausibility gains—such as a 1.04 dB M.PSNR and 0.65 FID improvement on Toys4K—we emphasize that FlowBender without any explicit guidance already significantly outperforms baselines using doubled sampling budgets or CFG. This indicates that while FlowBender can effectively leverage optional user-controlled test-time guidance, its primary strength lies in the learned correction policy which internalizes the feedback signal.

- Table 6: Super resolution results. Asterisks (*) denote base configurations compared in the main text. Shaded rows indicate FlowBender variants.

Fidelity Plausibility Method PSNR ↑ SSIM ↑ LPIPS ↓ FID ↓

Standard FT (*) 34.35±2.78 96.88±4.32 0.83±0.43 3.93 + CFG (w = 3.0) 34.65±2.94 97.23±3.29 0.91±0.49 4.09 + CFG (w = 5.0) 34.07±3.02 97.17±2.76 1.15±0.68 4.57

- + 2× steps 34.46±2.74 96.93±3.84 0.86±0.44 4.07

Standard FT + Lalign (*) 35.21±2.80 97.53±2.96 0.79±0.36 4.11

- + 2× steps 35.11±2.81 97.43±2.74 0.83±0.39 4.10

IT Guidance (λ = 0.1) 38.30±2.80 97.20±1.91 1.33±1.46 20.93 IT Guidance (λ = 0.5) (*) 43.02±4.85 98.29±3.19 0.65±2.24 18.96 IT Guidance (λ = 0.5, 2× steps) 46.05±4.91 98.95±2.72 0.27±1.18 18.87

Zero-order (*) 39.25±2.46 98.18±4.06 0.21±0.11 3.36 + CFG (w = 3.0) 45.25±1.35 99.08±1.68 0.13±0.15 3.58 + CFG (w = 5.0) 42.61±1.61 98.70±1.10 0.34±0.38 4.20

First-order (w.r.t. xt) (*) 36.27±3.95 97.10±4.69 0.66±0.39 4.30 First-order (w.r.t. xˆ1) (*) 39.21±3.85 97.64±5.01 0.43±0.30 3.83

+ CFG (w = 3.0) 45.14±2.98 98.95±2.50 0.17±0.17 4.74 + CFG (w = 5.0) 44.07±2.11 98.96±2.04 0.33±0.55 6.45

Combined (w.r.t. xt) (*) 39.95±2.77 98.25±4.18 0.21±0.12 3.40 + CFG (w = 3.0) 43.77±1.40 98.79±2.58 0.15±0.17 3.62 + CFG (w = 5.0) 41.43±1.57 98.25±2.10 0.42±0.49 4.39

Combined (w.r.t. xˆ1) (*) 39.77±2.83 98.24±3.94 0.23±0.13 3.39

- Table 7: Depth-to-RGB generation results. Asterisks (*) denote base configurations compared in the main text. Shaded rows indicate FlowBender variants.

Fidelity Plausibility

Method MAE ↓ δ1 ↑ FID ↓ Standard FT (*) 0.0847±0.0593 0.7886±0.1930 18.21

+ CFG (w = 3.0) 0.0814±0.0394 0.8003±0.1331 14.90 + CFG (w = 5.0) 0.0823±0.0388 0.7959±0.1333 15.77 + 2× steps 0.0844±0.0415 0.7907±0.1381 16.74

FT + Lalign (*) 0.0837±0.0600 0.7930±0.1926 17.57 IT Guidance (λ = 0.0001) 0.1617±0.0787 0.5919±0.1986 23.42 IT Guidance (λ = 0.001) (*) 0.0866±0.0360 0.7592±0.1284 223.54 IT Guidance (λ = 0.01) 0.1385±0.0501 0.6145±0.1378 328.57

Zero-order (*) 0.0763±0.0563 0.8188±0.1843 15.70 First-order (w.r.t. xt) (*) 0.0747±0.0535 0.8268±0.1726 14.57

###### + CFG (w = 3.0) 0.0616±0.0465 0.8705±0.1474 14.62 + CFG (w = 5.0) 0.0534±0.0422 0.8961±0.1280 14.76

First-order (w.r.t. xˆ1) (*) 0.0818±0.0582 0.7973±0.1933 15.89 Combined (w.r.t. xˆ1) (*) 0.0829±0.0610 0.7949±0.1953 15.93 Combined (w.r.t. xt) (*) 0.0783±0.0606 0.8175±0.1878 15.39

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

GT Condition Ours Standard FT

- Figure 8: Qualitative comparison for the JPEG restoration task. Our method reduces color banding quantization artifacts (rows 1-3) and color shifts (rows 4-5).

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

Ours (Zero-Order)

Ours (First-Order)

Standard FT FT + ℒalign

Condition

Figure 9: Qualitative comparison for the depth-to-RGB task.

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

0.3

0.0

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

0.06

0.0

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

0.3

0.0

Ours (Zero-Order)

Ours (First-Order)

GT Standard FT FT + ℒalign

Condition

###### Figure 10: Qualitative comparison for the super-resolution task. Error maps show per-pixel MAE.

- Table 8: JPEG restoration results. Asterisks (*) denote base configurations compared in the main text. Shaded rows indicate FlowBender variants.

Fidelity Plausibility

Method PSNR SSIM LPIPS FID Standard FT (*) 26.29±2.91 79.45±10.59 22.24±5.44 4.35

+ CFG. (w = 3.0) 26.56±3.04 80.32±10.18 21.62±5.48 4.36 + CFG. (w = 5.0) 26.58±3.10 80.85±9.89 21.44±5.47 4.62 + 2× steps 26.30±3.00 79.17±10.54 22.59±5.58 4.30

Zero-order (*) 28.86±4.00 83.13±9.64 16.33±4.59 3.80 + CFG. (w = 3.0) 29.79±4.21 85.58±8.50 14.72±4.93 3.85 + CFG. (w = 5.0) 29.66±4.07 85.83±8.02 16.19±5.95 4.58

###### Table 9: Edge-to-RGB generation results. Asterisks (*) denote base configurations compared in the main text. Shaded rows indicate FlowBender variants.

Fidelity Plausibility

Method Edge MAE ↓ Edge MSE ↓ FID ↓ Standard FT (*) 0.0533±0.0286 0.0137±0.0087 13.98

+ CFG (w = 3.0) 0.0500±0.0286 0.0137±0.0092 12.41 + CFG (w = 5.0) 0.0513±0.0296 0.0150±0.0101 13.64 2× steps 0.0512±0.0285 0.0134±0.0087 13.17

Standard FT + Lalign (*) 0.0501±0.0277 0.0128±0.0087 14.47 IT Guidance (λ = 0.001) 0.0620±0.0359 0.0247±0.0167 23.97 IT Guidance (λ = 0.01) (*) 0.0416±0.0235 0.0129±0.0093 97.65 Zero-order (*) 0.0456±0.0253 0.0104±0.0072 14.81 First-order (w.r.t. xt) (*) 0.0460±0.0265 0.0111±0.0077 14.29 First-order (w.r.t. xˆ1) (*) 0.0435±0.0294 0.0123±0.0089 14.97 Combined (w.r.t. xt) (*) 0.0435±0.0253 0.0097±0.0072 13.91 Combined (w.r.t. xˆ1) (*) 0.0408±0.0232 0.0085±0.0060 13.68

+ CFG (w = 3.0) 0.0322±0.0201 0.0057±0.0053 14.62 + CFG (w = 5.0) 0.0320±0.0220 0.0062±0.0071 17.97

[Figure 147]

[Figure 148]

[Figure 149]

|[Figure 150]<br><br>[Figure 151]|
|---|

|[Figure 152]<br><br>[Figure 153]|
|---|

[Figure 154]

[Figure 155]

[Figure 156]

|[Figure 157]<br><br>[Figure 158]|
|---|

|[Figure 159]<br><br>[Figure 160]|
|---|

Condition lr = 0.00001 lr = 0.001

(a) Depth

[Figure 161]

[Figure 162]

[Figure 163]

|[Figure 164]<br><br>[Figure 165]|
|---|

|[Figure 166]<br><br>[Figure 167]|
|---|

[Figure 168]

[Figure 169]

[Figure 170]

|[Figure 171]<br><br>[Figure 172]|
|---|

|[Figure 173]<br><br>[Figure 174]|
|---|

Condition lr = 0.01 lr = 0.001

(b) Edge

- Figure 11: Qualitative examples of failures of FlowChef for neural-network forward operators. As the adherence to the condition improves, shown as insets, the visual quality degrades significantly.

###### Table 10: Extended Quantitative Results for 3D Texturing (Objaverse). Asterisks (*) denote configurations included in the main text comparisons. Shaded rows indicate FlowBender variants.

Fidelity Plausibility Method M. PSNR ↑ SSIM ↑ LPIPS ↓ CLIP ↑ MV-M.PSNR ↑ MV-SSIM ↑ MV-LPIPS ↓ MV-CLIP ↑ FID ↓ IT Guidance (*) 25.87±3.7 0.986±0.015 0.0191±0.0250 0.973±0.029 22.79±5.7 0.989±0.011 0.0124±0.0113 0.968±0.024 9.10

+ 2× steps 27.05±3.5 0.988±0.013 0.0174±0.0231 0.978±0.026 23.47±5.9 0.990±0.011 0.0122±0.0114 0.969±0.026 8.70

+ s = 0.5 26.77±3.2 0.987±0.015 0.0218±0.0276 0.970±0.035 23.04±6.7 0.989±0.011 0.0136±0.0130 0.966±0.025 10.11 Standard FT (*) 21.91±3.8 0.981±0.020 0.0189±0.0178 0.970±0.022 23.05±6.3 0.990±0.012 0.0116±0.0114 0.970±0.022 8.74

+ CFG (w = 3) 20.41±4.4 0.980±0.023 0.0217±0.0215 0.961±0.030 21.06±5.5 0.988±0.015 0.0134±0.0136 0.963±0.024 10.11 + CFG (w = 5) 19.45±4.8 0.975±0.034 0.0255±0.0273 0.952±0.042 19.83±5.0 0.986±0.020 0.0158±0.0167 0.955±0.029 12.53 + 2× steps 21.68±3.9 0.980±0.021 0.0194±0.0184 0.970±0.022 22.77±6.5 0.989±0.012 0.0120±0.0117 0.969±0.022 8.81

FT + Lalign

λ = 10−3 22.49±3.8 0.983±0.017 0.0182±0.0162 0.971±0.021 22.89±6.4 0.990±0.011 0.0116±0.0110 0.970±0.021 8.79 λ = 5 × 10−3 (*) 22.63±3.6 0.983±0.018 0.0182±0.0160 0.970±0.023 22.48±5.3 0.990±0.011 0.0117±0.0110 0.969±0.021 8.94 λ = 10−2 21.75±4.2 0.981±0.018 0.0193±0.0157 0.969±0.024 21.73±6.0 0.988±0.012 0.0127±0.0112 0.966±0.023 9.64

Ours (w.r.t. xt) (*) 25.26±3.9 0.985±0.017 0.0157±0.0152 0.979±0.017 25.91±7.2 0.991±0.011 0.0105±0.0109 0.974±0.021 7.41 Ours (w.r.t. xˆ1) (*) 26.40±3.8 0.987±0.016 0.0140±0.0142 0.983±0.014 26.53±7.2 0.992±0.010 0.0098±0.0101 0.977±0.019 6.64

+ CFG (w = 3) 27.02±5.1 0.989±0.013 0.0124±0.0128 0.986±0.017 25.85±7.2 0.992±0.010 0.0096±0.0100 0.978±0.018 6.25 + CFG (w = 5) 26.35±5.6 0.989±0.012 0.0139±0.0153 0.983±0.023 24.43±7.2 0.991±0.011 0.0109±0.0111 0.973±0.022 7.35

- Table 11: Extended Quantitative Results for 3D Texturing (Toys4K). Asterisks (*) denote base configurations compared in the main text. Shaded rows indicate FlowBender variants.

Fidelity Plausibility

Method M. PSNR ↑ SSIM ↑ LPIPS ↓ CLIP ↑ MV-M.PSNR ↑ MV-SSIM ↑ MV-LPIPS ↓ MV-CLIP ↑ FID ↓

IT Guidance (*) 26.82±3.2 0.990±0.009 0.0137±0.0136 0.982±0.017 23.63±4.2 0.992±0.008 0.0112±0.0111 0.972±0.020 8.43 + 2× steps 27.95±3.2 0.991±0.008 0.0127±0.0125 0.984±0.016 24.42±4.7 0.992±0.009 0.0111±0.0110 0.973±0.020 7.90 + s = 0.5 28.02±3.4 0.991±0.009 0.0163±0.0195 0.975±0.030 23.94±5.2 0.991±0.008 0.0120±0.0116 0.970±0.020 9.00

Standard FT (*) 23.00±4.2 0.986±0.014 0.0160±0.0164 0.975±0.024 24.38±6.9 0.991±0.010 0.0114±0.0118 0.972±0.021 8.87

+ CFG (w = 3) 21.43±4.4 0.985±0.015 0.0182±0.0177 0.968±0.023 21.26±4.9 0.990±0.010 0.0130±0.0124 0.965±0.023 10.30 + CFG (w = 5) 20.55±4.6 0.984±0.016 0.0210±0.0214 0.963±0.025 20.32±4.9 0.989±0.011 0.0151±0.0149 0.960±0.027 12.41 + 2× steps 22.66±4.3 0.986±0.015 0.0163±0.0161 0.975±0.024 23.04±6.3 0.990±0.010 0.0118±0.0118 0.971±0.020 8.75

FT + Lalign

λ = 10−3 23.01±4.1 0.986±0.014 0.0162±0.0173 0.975±0.023 23.39±5.3 0.991±0.010 0.0114±0.0124 0.972±0.022 7.99 λ = 5 × 10−3 (*) 23.05±4.0 0.986±0.014 0.0159±0.0162 0.976±0.023 23.24±5.0 0.991±0.009 0.0111±0.0112 0.973±0.022 8.30 λ = 10−2 22.79±4.2 0.986±0.015 0.0168±0.0169 0.975±0.024 23.16±5.7 0.990±0.010 0.0119±0.0116 0.971±0.022 8.60

Ours (w.r.t. xt) (*) 26.54±4.2 0.990±0.011 0.0129±0.0140 0.980±0.024 27.20±8.1 0.993±0.008 0.0097±0.0104 0.977±0.020 7.20 Ours (w.r.t. xˆ1) (*) 27.26±4.4 0.991±0.010 0.0116±0.0125 0.984±0.022 27.21±7.5 0.993±0.008 0.0092±0.0100 0.978±0.021 6.66

+ CFG (w = 3) 28.30±4.0 0.993±0.006 0.0099±0.0102 0.990±0.010 26.18±6.7 0.993±0.008 0.0093±0.0102 0.979±0.018 6.01 + CFG (w = 5) 27.44±4.8 0.992±0.007 0.0114±0.0120 0.986±0.017 24.26±6.7 0.992±0.008 0.0109±0.0120 0.974±0.022 7.53

Condition Image Ours ( x𝟏) Ours (x𝒕) Standard FT FT + ℒalign IT Guidance

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

ObjaverseToys4K

| |
|---|

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

| |
|---|

| |
|---|

| |
|---|

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

Figure 12: 3D Mesh Texturing Results (Objaverse, Toys4K).

###### Condition Image Multi-View Renders - Ours ( x𝟏)

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

Figure 13: Multi-view visualizations of textured objects (Objaverse, Toys4K).

