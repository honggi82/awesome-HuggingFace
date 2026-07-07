## WiT: Waypoint Diffusion Transformers via Trajectory Conflict Navigation

##### Hainuo Wang1∗, Mingjia Li1∗, and Xiaojie Guo1†

College of Intelligence and Computing, Tianjin University, Tianjin 300350, China {hainuo, mingjiali}@tju.edu.cn, xj.max.guo@gmail.com

# arXiv:2603.15132v2[cs.CV]26Mar2026

Abstract. While recent Flow Matching models avoid the reconstruction bottlenecks of latent autoencoders by operating directly in pixel space, the lack of semantic continuity in the pixel manifold severely intertwines optimal transport paths. This induces severe trajectory conflicts near intersections, yielding sub-optimal solutions. Rather than bypassing this issue via information-lossy latent representations, we directly untangle the pixel-space trajectories by proposing Waypoint Diffusion Transformers (WiT). WiT factorizes the continuous vector field via intermediate semantic waypoints projected from pre-trained vision models. It effectively disentangles the generation trajectories by breaking the optimal transport into prior-to-waypoint and waypoint-to-pixel segments. Specifically, during the iterative denoising process, a lightweight generator dynamically infers these intermediate waypoints from the current noisy state. They then continuously condition the primary diffusion transformer via the Just-Pixel AdaLN mechanism, steering the evolution towards the next state, ultimately yielding the final RGB pixels. Evaluated on ImageNet 256×256, WiT beats strong pixel-space baselines, accelerating JiT training convergence by 2.2×. Code will be released on our project page.

Keywords: Image Generation · Flow Matching · Trajectory Conflict

### 1 Introduction

Diffusion models [12,37], particularly those formalized through Flow Matching (FM) frameworks [1,24,25] and scaled via Diffusion Transformers (DiT) [26,30], have established a new standard in highly realistic image generation. To mitigate the computational costs, these architectures traditionally operate in latent spaces [4, 31, 34], relying on continuous-valued variational autoencoders (VAEs) [10, 28, 31, 41] to compress raw visual signals. However, this two-stage design inherently introduces an information bottleneck. Consequently, visual tokenizers inevitably discard high-frequency textural details and frequently produce visual artifacts, placing a strict upper bound on overall generation quality [42]. To overcome these limitations, a recent paradigm shift, exemplified by architectures such as JiT [22], advocates for learning continuous vector fields directly in

∗ Equal contribution. † Corresponding author.

[Figure 1]

|[Figure 2]|
|---|

[Figure 3]

[Figure 4]

(c) Waypoint t-SNE

- (e) WiT Training Loss
- (f) WiT Samples Results

|[Figure 5]|
|---|

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(a) FM Trajectory (b) WiT Trajectory

(d) Pixel t-SNE

- Fig. 1: An overview of our Waypoint Diffusion Transformers. (a) and (b) demonstrate the difference in trajectories before/after the waypoint is introduced. In standard pixelspace FM (a), mapping directly to an entangled, non-discriminative pixel manifold (d) induces severe trajectory conflict. With the integration of discriminative semantic waypoints (c), our WiT successfully converts the noise-to-pixel task into two stable, decoupled mappings. By routing the transport path, the generative flow is disentangled, thus mitigating path overlap. Consequently, WiT significantly accelerates convergence compared to baseline (e) while yielding highly realistic generated samples (f).

the original pixel space [6,19,27,44]. By entirely bypassing the visual tokenizer, pixel-space Flow Matching eliminates compression-induced artifacts, offering a direct and theoretically lossless path for preserving fine-grained visual details.

Despite its simplicity, mapping directly from a shared noise distribution to a highly complex, multi-channel pixel distribution presents a formidable optimization challenge, as recent studies suggest that generative models inherently struggle to learn unconstrained, high-dimensional spaces from scratch [3,42]. In the realm of latent diffusion, VA-VAE [42] addresses this optimization dilemma by aligning the VAE’s latent space with pre-trained vision foundation models. This alignment effectively regularizes the target manifold, rendering it more structured, uniform, and semantically discriminative. However, pure pixel-space generation operates under different constraints. Our target manifold (raw pixels) is naturally entangled and inherently non-discriminative (Figure 1(d)). Unlike learnable latent spaces, the pixel domain is locked to universal display standards and cannot be artificially reshaped to disentangle semantics. Consequently, standard pixel-space Flow Matching suffers from severe trajectory conflict [24,25]. Transportation paths destined for visually similar but semantically distinct endpoints lack natural geometric separation, routinely converging in dense local regions of the noise space. Forced to minimize regression loss over overlapping paths, the neural network predicts an averaged velocity field [38]. This manifests as semantic bleeding and slower convergence. Techniques like ClassifierFree Guidance (CFG) [13] dynamically extrapolate the velocity logits using the difference between conditional and unconditional scores. While CFG effectively amplifies class-specific signal magnitudes, it is a post-hoc intervention that does not untangle the underlying spatial overlap of the training trajectories. A question naturally arises: How can we provide clear, semantically separable guidance to a pixel-space vector flow without reverting to black-box latent spaces?

Recognizing that the target pixel space is inherently non-discriminative and resistant to direct regularization, in this paper, we introduce a highly discriminative intermediate waypoint into the generative flow. We propose to explicitly decouple semantic navigation from pixel-level texture generation by reformulating the standard, unconstrained generative trajectory. Specifically, we decompose the challenging mapping between two non-discriminative manifolds (from the isotropic noise prior to the raw pixel distribution) by routing the transport path through a discriminative waypoint. Since the flow tradictory is bijective, this establishes two mathematically stable mappings: an initial mapping from the non-discriminative noise to the discriminative waypoint, followed by a mapping from this discriminative waypoint to the non-discriminative image space. By structuring the continuous vector field around these waypoints, we prevent the flow from collapsing into averaged, conflicting paths. This bipartite regularization not only mitigates severe trajectory conflict but also accelerates training convergence. To construct these robust semantic anchors, we leverage the feature spaces of modern self-supervised vision models [29,35], exploiting their discriminative ability to ground visual subjects within the generative flow.

We implement this concept with WiT (Waypoints Diffusion Transformers), a framework specifically designed to mitigate trajectory conflict in pixel-space Flow Matching. Instead of directly utilizing raw, high-dimensional representations from frozen vision foundation models, we apply Principal Component Analysis (PCA) to project these features onto a compact, low-dimensional semantic manifold. This relieves the burden of significant spatial redundancy and imposes a severe regression burden. By capturing only the principal directions of semantic variance, we extracted discriminative structural cues. Second, we integrate a lightweight waypoint generator into the flow matching pipeline, which is now optimized to reliably infer this condensed semantic waypoint from the noisy distribution at any integration timestep t. Finally, we design the pixel diffusion transformer to be spatially conditioned on these predicted semantic maps via our proposed Just-Pixel AdaLN mechanism. As the noisy state zt evolves, the semantic guidance is naturally and continuously recalibrated, providing a rectifying force that steers the trajectory toward the correct class manifold and away from conflicting zones. As a result, WiT establishes a more effective architecture for pixel-space flow matching. Evaluations on ImageNet 256×256 [7] generation demonstrate that our approach achieves superior boundary clarity and structural consistency compared to previous pixel-based baselines like JiT [22]. Our main contributions can be summarized as follows:

- – We propose the Waypoint Diffusion Transformers (WiT), a novel generative paradigm that mitigates severe trajectory conflict in pixel-space Flow Matching. By anchoring flow trajectories to low-dimensional semantic manifolds, we introduce a decoupled pipeline that isolates semantic navigation from pixel-level generation.
- – We introduce the Just-Pixel AdaLN mechanism. Unlike standard global conditioning, it leverages dynamically predicted semantic waypoints to provide spatially-varying modulation, ensuring semantic guidance.

- – Through extensive experiments on ImageNet 256×256, WiT achieves stateof-the-art performance among purely pixel-space models. Crucially, explicit semantic grounding yields a 2.2× training speedup compared with JiT-L/16.

### 2 Related Work

Diffusion Models and Flow Matching. Score-based diffusion models [12,37] and their continuous-time ODE formulations have established a new paradigm for generative modeling. Early formulations learn a reversed stochastic process by predicting the injected noise (i.e., ϵ-prediction) [12]. Subsequent research revealed that shifting the prediction target to a noised quantity, such as the flow velocity (v-prediction) [32], could alter the optimization landscape and improve generation stability. More recently, Flow Matching [1, 24, 25] has unified these continuous-time processes into a simpler optimal transport framework. By explicitly formulating the mapping between a simple base and the target distribution, FM yields straightened probability flow ODE trajectories, leading to a reduction in steps. Concurrently, the backbone has undergone a significant transition. Diffusion Transformers [30] and Scalable Interpolant Transformers [26] have demonstrated that self-attention can effectively replace traditional dense U-Nets. Building upon these foundations, WiT aims to resolve the optimization instabilities in integrating complex, high-dimensional continuous vector fields.

Generative Modeling in Pixel Space. Generative Adversarial Networks [11, 33] and early Normalizing Flows [9, 17] operate directly in the raw pixel space. However, scaling these early pixel-based approaches to high-resolution synthesis proved computationally prohibitive. Thus, the field experienced a paradigm shift toward latent-space modeling, propelled by VQ-VAE [10] and LDM [31]. These methods compress high-dimensional images into low-dimensional latent manifolds before generation. While this latent compression mitigates computational bottlenecks, it is inherently lossy; it inevitably introduces information bottlenecks, spatial reconstruction artifacts, and a noticeable degradation of textural details. In pursuit of a high-fidelity generation, a recent shift advocates for pure pixel-space modeling [6, 19, 27, 44]. Advances such as SiD2 [15], and PixelFlow [5] demonstrate that scalable large-patch Vision Transformers can now directly model raw pixels without relying on auxiliary tokenizers. However, directly operating in this high-dimensional domain introduces a new bottleneck: according to the manifold assumption, while clean data lies on a low-dimensional manifold, intermediate noisy states inherently span the full high-dimensional space. JiT [22] attempts to mitigate this by x-prediction. However, mapping a highly complex pixel distribution directly from noise severely exacerbates the overlapping of trajectories. WiT embraces the pure pixel-space paradigm but proposes a reorganization to bypass these high-dimensional ambiguities.

Mitigating Optimization Conflict via Representation Alignment. In the conditional Flow Matching regime, we use the neural network to estimate a unified

vector field that transports shared Gaussian noise to thousands of distinct semantic classes simultaneously. Since pixel space is semantically entangled, paths destined for visually similar but semantically distinct endpoints lack natural geometric separation. During intermediate integration phases, these class-conditional optimal transport paths routinely converge or cross. As recently formalized by the optimization dilemma [42], this forces the neural network to minimize the regression loss by predicting an averaged velocity field. Recent literature has also begun exploring the intersection of representation learning and generative diffusion. Methods like REPA [43], REPA-E [20,21], iREPA [36], and RAE [45] attempt to align the internal representations of diffusion transformers with pretrained representation encoders to accelerate convergence. However, these prior methods typically operate within heavily compressed latent spaces or treat representations merely as auxiliary loss supervisions. In stark contrast, WiT explicitly constructs low-dimensional semantic waypoints derived dynamically from these representations and trains a dedicated, lightweight Waypoints DiT to navigate toward them. More importantly, through our proposed Just-Pixel AdaLN mechanism, these predicted waypoints serve as dense, spatially varying conditions that structurally anchor the massive Pixel Space DiT.

### 3 Methodology

In this section, we detail the formulation and architecture of the proposed Waypoint Diffusion Transformers (WiT). We first review the standard pixel-space Flow Matching framework and formalize the trajectory conflict. To resolve these ambiguities, we introduce the construction of low-dimensional semantic waypoints derived from pre-trained vision models. Finally, as illustrated in Figure 2, we present our WiT, detailing how the proposed Just-Pixel AdaLN mechanism modulates the transformer features with spatially-varying semantic guidance, explicitly decoupling semantic navigation from high-realistic pixel generation.

#### 3.1 Pixel-Space Flow Matching and Trajectory Conflict

Following standard Flow Matching frameworks, let x ∈ RH×W×3 denote a clean target image, and ϵ ∼ N(0,I) denote standard Gaussian noise. The intermediate noisy state zt at timestep t ∈ [0,1] is defined as zt = tx + (1 − t)ϵ.

The ground-truth velocity vector field driving the state from noise to data is mathematically given by v = x − ϵ. As exemplified by state-of-the-art pixel models like JiT [22], x-prediction is recommended for pixel space generation, i.e., training a parameterized network Gθ to predict the clean image xˆ directly. From this, the estimated velocity is analytically constructed as:

xˆ − zt 1 − t

. (1)

vˆ =

AdaLN

Just-Pixel AdaLN

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

SemanticWaypoints

TransformerBlock

TransformerBlock

TransformerBlock

TransformerBlock

[Figure 16]

[Figure 17]

[Figure 18]

LinearEmbed

LinearEmbed

LinearEmbed

LinearEmbed

[Figure 19]

[Figure 20]

...

...

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Waypoints Generator

Pixel Space Generator

Tiny (21M)

- Fig. 2: Overview of the WiT architecture. Left: A lightweight Waypoints Generator (21M params) predicts Semantic Waypoints from the noisy state zt. Right: The Pixel Space Generator synthesizes the image, utilizing these predicted waypoints as spatial conditions via the Just-Pixel AdaLN mechanism.

The network is then optimized using a velocity-matching objective (v-loss), which aligns the estimated velocity with the ground-truth vector field:

2

x ˆ − zt 1 − t − (x − ϵ)

Lv = Ex,ϵ,t,y ∥vˆ − v∥22 = Ex,ϵ,t,y

. (2)

2

However, mapping directly from a class-agnostic Gaussian prior to a complex pixel distribution under this objective incurs severe trajectory conflict. Under the MSE objective, the optimal denoiser xˆ∗ at any intermediate timestep t is the conditional expectation of the target data given the noisy observation:

xˆ∗(zt) = E[x|zt]. (3)

The trajectory conflict can be formalized as the irreducible variance of this optimal estimator. Because the pixel space is semantically highly entangled, diverse target images x corresponding to radically different semantic classes share identical dense neighborhoods in the input noise space as t → 0. This ambiguity at coordinate zt can be quantified by the variance of the target distribution:

Var(x|zt) = E ∥x − E[x|zt]∥22 zt . (4)

Attempting to blindly regress divergent endpoints x from overlapping initial states yields an extremely large Var(x|zt). To minimize the regression loss, the neural network is forced to output the averaged state E[x|zt], causing severe gradient interference and limiting convergence.

To resolve this, we hypothesize that explicit semantic grounding can partition the optimal vector field. By introducing a discriminative intermediate semantic waypoint s0, the optimal predictor becomes conditioned on both the noisy state and the semantic topology: xˆ∗WiT(zt,s0) = E[x|zt,s0]. According to the Law of Total Variance, the original trajectory conflict is decomposed as:

Var(x|zt) = Es

0

[Var(x|zt,s0)] + Vars

0

(E[x|zt,s0]). (5)

[Figure 27]

Input Tokens Waypoints

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

| | |
|---|---|
|ML|P|

RMSNorm

Scale, Shif MHA Scale

[Figure 38]

[Figure 39]

RMSNorm

[Figure 40]

[Figure 41]

Scale, Shif

MLP

Scale

(a) Just-Pixel AdaLN (b) Visualization of the Waypoints

- Fig. 3: (a) Just-Pixel AdaLN: The predicted semantic waypoints provide spatially varying modulation. (b) Visualization of the predicted semantic waypoints and intermediate pixel states during inference. Left. The evolving noisy pixel states zt at different integration timesteps. Right. The corresponding spatial semantic waypoints sˆ0 dynamically inferred by our lightweight Waypoints Generator.

(E[x|zt,s0]) is explicitly resolved by predicting s0. As recently formalized by VA-VAE [42], mapping continuous flows from an isotropic noise prior to a highly discriminative, low-dimensional space is inherently more tractable and avoids severe gradient interference. Consequently, the primary pixel generator is only tasked with resolving the residual variance Var(x|zt,s0). Because the semantic waypoint s0 tightly bounds the target manifold to a specific affine subspace, this residual variance is substantially smaller than the unconditioned total variance Var(x|zt). By firmly anchoring the vector field to these semantic guides, generative trajectories are steered to bypass overlapping zones. More details can be found in Section 5.

In our decoupled architecture, the variance component Vars

0

#### 3.2 Constructing Semantic Waypoints

To eliminate the geometric ambiguity of intersecting trajectories, the generative process must be firmly anchored by an intermediate structural guide. We leverage the highly separable representation space of frozen self-supervised vision models, specifically DINOv3 [35], to serve as these ground-truth semantic anchors.

For a given target image x, we extract dense, patch-wise semantic tokens ϕ(x) ∈ RN×D. Because raw DINOv3 features possess a high dimensionality that imposes a severe optimization burden, we construct a compact affine subspace via Principal Component Analysis fitted on the training distribution. Let Ud ∈ RD×d denote the projection matrix for the top d = 64 principal components, and µ be the dataset mean. We define the explicit ground-truth semantic waypoint s0 as:

s0 = (ϕ(x) − µ)Ud ∈ RN×64. (6)

This orthogonal projection constructs a low-dimensional manifold optimized for class separability. By exploiting the intrinsic sparsity and low-rank structure of these feature spaces, we establish a tractable optimization landscape that acts as a direct, structural supervisory signal for our framework.

Lightweight Waypoints Generator. We introduce a lightweight transformer, denoted as Wψ, which operates on the pixel-level noisy observation zt = tx + (1 − t)ϵimg. Conditioned on the timestep t and class label y via standard AdaLN, Wψ is tasked with resolving the clean semantic waypoint sˆ0 = Wψ(zt,t,y) from the high-dimensional pixel noise. To supervise this cross-domain mapping, we establish a parallel probability flow ODE in the semantic space. Let zsem,t = ts0 + (1−t)ϵsem denote the intermediate state on the semantic trajectory, constructed with an independent Gaussian noise ϵsem ∼ N(0,I). The objective is to match the analytically derived semantic velocity vˆsem = (ˆs0 − zsem,t)/max(1 − t,τeps) with the target ground-truth velocity vsem = (s0 − zsem,t)/max(1 − t,τeps). The generator minimizes the following loss:

Lsem = Ex,s

0,ϵimg,ϵsem,t,y

s0 − zsem,t max(1 − t,τeps)

s ˆ0 − zsem,t max(1 − t,τeps) −

2

2

, (7)

where τeps denotes a small positive constant introduced to prevent numerical instability (i.e., division by zero) as t → 1. Given its highly compressed target dimension (d = 64), Wψ requires minimal capacity (e.g., 21M parameters) and serves as an efficient navigator for the primary diffusion process.

#### 3.3 Semantic-Pixel Decoupled Architecture

Rather than enforcing a direct, unconstrained mapping from noise to raw pixels, WiT decomposes the generative process into a decoupled architecture. As shown in Figure 2, the framework consists of a lightweight Waypoints Generator and a primary Pixel Space Generator.

Pixel Space Generator via Just-Pixel AdaLN. Once the semantic waypoint sˆ0 is inferred, it is injected into the primary Pixel Space Generator Gθ. To disentangle the semantic waypoint from pixel-space generation, we propose the Just-Pixel AdaLN mechanism. As shown in Figure 3 (a), unlike standard AdaLN, which modulates tokens uniformly via a globally pooled time-class embedding e(t,y), our mechanism provides spatially-varying guidance. We aggregate the global conditioning and the localized semantic map into a unified spatial condition cs = e(t,y) + Proj(ˆs0), where Proj(·) is a linear projection mapping the 64dimensional sequence to the transformer’s hidden dimension Dh. For the l-th transformer block, given the hidden token sequence hl ∈ RN×D

h, the condition

- Algorithm 1 Training Procedure of WiT

Require: Dataset D, Pre-trained model ϕ, PCA projection Ud, dataset mean µ Require: Waypoints Generator Wψ, Pixel Space Generator Gθ

- 1: Training the Waypoints Generator
- 2: while Wψ has not converged do
- 3: Sample (x, y) ∼ D, t ∼ U[0, 1], and ϵimg, ϵsem ∼ N(0, I)
- 4: s0 ← (ϕ(x) − µ)Ud // Extract ground-truth semantic waypoint
- 5: zt ← tx + (1 − t)ϵimg // Construct noisy pixel state
- 6: zsem,t ← ts0 + (1 − t)ϵsem // Construct noisy semantic state
- 7: sˆ0 ← Wψ(zt, t, y) // Predict clean waypoint from pixel noise
- 8: Lsem ← max(1s ˆ0−z−semt,τ,t

eps) − max(1s0−z−semt,τ,t

eps)

2 2

- 9: Update ψ via gradient descent on Lsem
- 10: end while
- 11: Training the Pixel Space Generator
- 12: Freeze the trained Waypoints Generator Wψ
- 13: while Gθ has not converged do
- 14: Sample (x, y) ∼ D, t ∼ U[0, 1], and ϵimg ∼ N(0, I)
- 15: zt ← tx + (1 − t)ϵimg // Construct noisy pixel state
- 16: sˆ0 ← Wψ(zt, t, y) // Infer semantic condition via frozen Wψ
- 17: xˆ ← Gθ(zt, t, y, sˆ0) // Spatially-conditioned pixel generation
- 18: Limg ← x ˆ1−−ztt − (x − ϵimg)

2 2

- 19: Update θ via gradient descent on Limg
- 20: end while

cs is projected into six spatially-varying modulation parameters to govern both the self-attention and MLP mechanisms:

γl(1),βl(1),αl(1),γl(2),βl(2),αl(2) = Linearl(cs). (8)

Following the AdaLN-Zero formulation, these continuous spatial maps sequentially modulate the normalized features and gate the residual connections:

h˜l = hl + αl(1) ⊙ Attention (1 + γl(1)) ⊙ RMSNorm(hl) + βl(1) , (9) hl+1 = h˜l + αl(2) ⊙ MLP (1 + γl(2)) ⊙ RMSNorm(h˜l) + βl(2) . (10)

By delegating semantic navigation to the waypoints generator, Just-Pixel AdaLN allows the primary transformer to focus entirely on high-realistic spatial generation. Finally, Gθ minimizes the pixel-level velocity-matching objective:

2

x ˆ − zt 1 − t − (x − ϵimg)

. (11)

Limg = Ex,ϵ

img,t,y

2

By explicitly grounding the pixel-level velocity field in a tractable semantic manifold, our WiT significantly enhances optimization stability and spatial realistic without relying on autoencoder-based latent compression. As summarized

- Algorithm 2 Inference Procedure of WiT via Just-Pixel AdaLN

Require: Frozen Waypoints Generator Wψ, Pixel Space Generator Gθ with L blocks Require: Target class y, Integration steps K

- 1: Sample initial pixel noise zt0 ∼ N(0, I)
- 2: Define timestep schedule 0 = t0 < t1 < · · · < tK = 1
- 3: for k = 0, . . . , K − 1 do
- 4: 1. Semantic Waypoint Recalibration
- 5: sˆ0 ← Wψ(ztk, tk, y) // Infer clean semantic waypoint
- 6: 2. Spatial Conditioning via Just-Pixel AdaLN
- 7: cs ← e(tk, y) + Proj(ˆs0) // Aggregate spatial condition
- 8: Initialize hidden token sequence h1 ∈ RN×Dh from ztk
- 9: for l = 1, . . . , L do
- 10: γl(1,2), βl(1,2), αl(1,2) ← Linearl(cs) // Obtain modulation parameters (Eq. 8)
- 11: h˜l ← hl + αl(1) ⊙ Attention (1 + γl(1)) ⊙ RMSNorm(hl) + βl(1)
- 12: hl+1 ← h˜l + αl(2) ⊙ MLP (1 + γl(2)) ⊙ RMSNorm(h˜l) + βl(2)
- 13: end for
- 14: xˆ ← LinearOut(hL+1) // Output predicted clean image
- 15: 3. Vector Field Estimation & ODE Step
- 16: vˆ ← xˆ1−−zttk

k

// Analytically derived velocity (Eq. 1)

- 17: ztk+1 ← ztk + (tk+1 − tk)ˆv // E.g., standard Euler step
- 18: end for
- 19: return Generated clean image ztK ≈ x

in Algorithm 1, we adopt a decoupled two-stage training paradigm. The Waypoints Generator Wψ is first trained to infer clean semantic anchors from pixel noise. Subsequently, Wψ is frozen and embedded within the primary Pixel Space Generator Gθ, providing reliable, spatially-varying semantic conditioning.

During inference, as in Algorithm 2, the generation process starts purely from a class-agnostic noise. At each ODE step, the embedded Wψ dynamically recalibrates the semantic waypoint sˆ0 from the current noisy state zt

. This continually refined semantic blueprint is then projected and aggregated with global embeddings to form the spatial condition cs, which actively modulates the intermediate transformer blocks of Gθ via our Just-Pixel AdaLN mechanism.

k

### 4 Experimental Validation

#### 4.1 Experimental Setup

We conduct experiments on the ImageNet 2012 [7] dataset at 256 × 256 resolution. To fairly evaluate the generative quality, we report the Fréchet Inception Distance (FID-50K) and Inception Score (IS). All pixel-space models are evaluated using the 50-step Heun solver following JiT [22]. The Waypoints Generator Wψ is formulated as a ViT-S/16 configuration, while the primary Pixel Space Generator Gθ maintains parity with JiT-Base and JiT-Large configurations. Before training, we randomly sample 50,000 images from the ImageNet training set

Table 1: Configuration of Experiments for WiT.

WiT-B WiT-L WiT-XL Architecture

depth 12 24 28 hidden dim 768 1024 1152 heads 12 16 16 image size 256 patch size 16 bottleneck 128 dropout 0

###### Training

epochs 200 (ablation), 600 warmup epochs 5 optimizer AdamW, β1 = 0.9, β2 = 0.95 batch size 1024 learning rate 5 × 10−5 learning rate schedule constant weight decay 0 ema decay {0.9996, 0.9999} time sampler logit(t) ∼ N(µ, σ2), µ = −0.8, σ = 0.8 noise scale 1.0 × image_size/256 clip of (1 − t) in division 0.05 class token drop (for CFG) 0.1

###### Sampling

ODE solver Heun ODE steps 50 time steps linear in [0.0, 1.0] CFG scale sweep range [1.0, 4.0] CFG interval [18] [0.1, 1.0] (if used)

to compute the PCA projection matrix, compressing the raw DINOv3 features to a compact dimension of d = 64. During the training stage, the Waypoints Generator Wψ is first optimized for 600 epochs to master semantic velocity matching on the PCA-reduced DINOv3 features. The Pixel Space Generator Gθ is then trained for up to 600 epochs, conditioned on the frozen Exponential Moving Average weights of Wψ. We utilize the AdamW optimizer with a constant learning rate schedule, a base learning rate of 5 × 10−5, and a 5-epoch linear warmup.

To facilitate reproducibility and provide a comprehensive overview of our architectural scaling, Table 1 details the exact hyperparameter configurations for the Waypoint Diffusion Transformers (WiT) across three capacity scales: Base (B), Large (L), and Extra-Large (XL).

In addition, several critical low-level mechanisms are implemented to ensure the stability of pure pixel-space Flow Matching following JiT [22]. First, we adopt a logit-normal distribution (µ = −0.8,σ = 0.8) for sampling the integra-

Table 2: Comprehensive comparison of class-conditional ImageNet 256 × 256.

Method Params Epochs IS ↑ FID-50K ↓ Latent-space Diffusion Models

DiT-XL/2 [30] 675+49M - 278.2 2.27 SiT-XL/2 [26] 675+49M - 277.5 2.06 REPA (SiT-XL/2) [43] 675+49M - 305.7 1.42 LightningDiT-XL/2 [42] 675+49M - 295.3 1.35 DDT-XL/2 [41] 675+49M - 310.6 1.26 RAE (DiTDH-XL/2) [45] 839+415M - 262.6 1.13

Pixel-space Models (Non-diffusion)

JetFormer [39] 2.8B - - 6.64 FractalMAR-H [23] 848M - 348.9 6.15

Pixel-space Diffusion Models

ADM-G [8] 554M - 186.7 4.59 RIN [16] 410M - 182.0 3.42 SiD (UViT/2) [14] 2B - 256.3 2.44 PixelFlow (XL/4) [5] 677M - 282.1 1.98 PixNerd (XL/16) [40] 700M - 297.0 2.15 JiT-H/16 [22] 953M - 303.4 1.86 JiT-G/16 [22] 2B - 292.6 1.82 LF-DIT-L/16 [2] 465M 200 - 2.48

Direct Baselines & Ours

JiT-B/16 [22] 131M 200 - 4.37 WiT-B/16 (Ours) 131M+21M 200 270.7 3.34

JiT-B/16 [22] 131M 600 275.1 3.66 WiT-B/16 (Ours) 131M+21M 600 280.2 3.03 JiT-L/16 [22] 459M 200 - 2.79 WiT-L/16 (Ours) 459M+21M 200 289.1 2.38

JiT-L/16 [22] 459M 600 298.5 2.36 WiT-L/16 (Ours) 459M+21M 265 293.7 2.36 WiT-L/16 (Ours) 459M+21M 600 303.3 2.22 WiT-XL/16 (Ours) 676M+21M 200 288.9 2.16 WiT-XL/16 (Ours) 676M+21M 600 301.0 1.89

tion timestep t during training. This non-uniform sampling strategy deliberately concentrates the training capacity on intermediate noise levels, where the optimal transport paths are most entangled and the trajectory conflict is most severe. Second, to strictly prevent numerical explosion when computing the v-prediction loss as t → 1, we enforce a clipping mechanism that bounds the denominator (1 − t) at a minimum threshold of 0.05. Finally, during the inference stage, we employ a truncated Classifier-Free Guidance (CFG) [13] strategy, parameterized by the CFG interval [18].

#### 4.2 Main Results

Quantitative Results. We compare WiT against a comprehensive set of state-ofthe-art generative models, including leading latent-space diffusion models (e.g., DiT [30], SiT [26]), pixel-space non-diffusion models (e.g., JetFormer [39]), and purely pixel-space diffusion models (e.g., PixelFlow [5], PixNerd [40], and our direct baseline JiT [22]). As shown in Table 2, WiT consistently outperforms its pixel-space counterparts at every comparable stage, highlighting massive improvements in training efficiency and sample realism.

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

Fig. 4: Qualitative Results of WiT-L/16 on ImageNet 256 × 256 [7].

With the B/16 configuration, WiT achieves an FID of 3.34 at just 200 epochs, already surpassing the vanilla JiT [22] trained for 600 epochs (3.66). Extending the training to 600 epochs, WiT reaches a superior FID of 3.03, demonstrating that explicit semantic waypoints significantly accelerate convergence and elevate the performance ceiling of pixel-space modeling. At the L/16 scale, WiT consistently outperforms both the pixel-space baseline (JiT [22]) and the Latent Forcing (LF-DiT [2]) under identical training budgets. With only 265 epochs of training, WiT achieves an FID of 2.36 and a high Inception Score of 293.7. Notably, this matches the performance of the JiT-L baseline at 600 epochs, delivering an impressive 2.27× training speedup. Crucially, when WiT-L/16 is extended to 600 epochs, it achieves an exceptional FID of 2.22 and an IS of 303.3. This milestone not only eclipses its pixel-space counterpart JiT-L/16 (2.36 FID), but also surpasses the heavy latent-space benchmark DiT-XL/2 (2.27 FID). Remarkably, WiT achieves these substantial performance leaps with the negligible computational overhead of a 21M waypoint generator. This confirms that anchoring the vector field in separable semantic manifolds allows pixel-space models to rival VAE-compressed latent models without relying on brute-force parameter scaling. Furthermore, the results demonstrate that our framework scales well with increased model capacity. By scaling the architecture to the Extra-Large (WiTXL/16) configuration, the generative quality is further enhanced, achieving an outstanding FID of 1.89 and an IS of 301.0 after 600 epochs. Notably, breaking the 2.0 FID barrier at the XL scale establishes WiT as a highly competitive stateof-the-art among pure pixel-space generative models (e.g. PixelFlow-XL/4 [5],

Table 3: Ablation studies on WiT-B/16 (200 epochs). We investigate the impact of semantic bottleneck size (d) and the architectural injection method.

Configuration PCA (d) Injection IS ↑ FID ↓ Ablation on Bottleneck Dimension

WiT-B/16 32 Just-Pixel AdaLN 210.40 5.11 WiT-B/16 128 Just-Pixel AdaLN 211.33 4.12

Ablation on Injection Mechanism

WiT-B/16 64 Channel Concat 221.19 3.93 WiT-B/16 64 In-context Concat 238.92 3.63

WiT-B/16 (Ours) 64 Just-Pixel AdaLN 270.73 3.34

PixNerd-XL/16 [40], etc). Moreover, this comprehensive performance successfully surpasses several prominent latent-space diffusion models in both FID and IS (e.g., DiT-XL/2 [30], SiT-XL/2 [26], to name just a few).

Qualitative Results. Figure 4 showcases highly realistic ImageNet 256 × 256 samples generated by WiT-L/16, corroborating our quantitative leaps and highlighting two core advantages. First, WiT exhibits exceptional structural coherence. By utilizing dynamically predicted semantic waypoints as steadfast navigational anchors, animals and complex scenes (e.g., the lion and castle) maintain correct proportions and strict perspectives, avoiding the severe geometric distortions typical of unanchored pixel-space models. Second, operating purely in pixel space preserves pristine, high-frequency micro-textures (e.g., fine owl feathers and intricate butterfly wings) that are often corrupted by VAE-based latent compression. By marrying the structural stability of semantic representations with the uncompressed realism of raw pixels, WiT establishes a highly robust paradigm for photorealistic generation. Finally, to qualitatively demonstrate the structural integrity, visual realism, and diversity of our approach, we provide additional uncurated generated samples in Figure 6, 7, 8, 9, 10, 11, 12, 13.

#### 4.3 Ablation Studies

To validate the effectiveness of WiT’s components and settings, we perform ablations on the semantic waypoint dimensionality, the feature injection mechanism, and the CFG scale during inference with WiT-B/16 under 200 epochs in Table 3.

PCA Dimension d. We evaluate the impact of the semantic waypoint’s information density by varying the number of PCA components d. The dimension d essentially dictates the trade-off between semantic expressiveness and optimization complexity. As shown in Table 3, using an excessively large dimension (d = 128) exacerbates the curse of dimensionality. This makes the waypoint space unnecessarily complex, hindering the predictor’s ability to map smooth trajectories and converge optimally (sub-optimal FID of 4.12). Conversely, extreme compression (d = 32) induces a severe information bottleneck. By inadvertently discarding vital structural variances, it results in semantic under-fitting; the waypoints lose

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

(a) WiT-L, 600 epoch

(b) WiT-B, 600 epoch

(c) WiT-B, 200 epoch

Fig. 5: The impact of CFG on FID and IS. The gold star indicates the minimum FID.

their discriminative power, causing a significant drop in sample quality (FID 5.11) as the network struggles to anchor onto distinct generative modes. We find that d = 64 provides an optimal balance. At this dimension, the PCA projection filters out non-essential noise while strictly preserving the core structural topology. This ensures that the latent representations remain highly clusterable, providing dense and reliable semantic anchors for stable trajectory learning.

Semantic Injection Strategy. We compare three methods for grounding the Pixel Space Generator Gθ with the predicted semantic waypoints: 1) Channel Concat: Concatenating sˆ0 directly to the input pixel noisy patches along the channel dimension; 2) In-context Concat: Appending semantic tokens as an in-context prefix to the Transformer sequence; 3) Just-Pixel AdaLN: Pure localized spatial modulation without invasive sequence concatenation. As reported in Table 3, Channel Concat performs the worst, resulting in an FID of 3.93 and an IS of 221.19. By forcibly fusing highly abstract semantic vectors with raw noisy pixels at the initial projection layer, it creates a representational mismatch that burdens the network’s early optimization. In-context Concat mitigates this mismatch by treating waypoints as separate prefix tokens, allowing the self-attention mechanism to query semantics dynamically. This improves the FID to 3.63 and the IS to 238.92. However, this invasive sequence extension still disrupts the native pixel-to-pixel attention manifold and forces the model to implicitly learn how to route prefix tokens to corresponding local spatial patches. In contrast, our Just-Pixel AdaLN achieves the best performance by a significant margin, securing the lowest FID of 3.34 and a substantially higher IS of 270.73. By injecting semantics through spatially varying affine modulations across intermediate transformer blocks, it avoids polluting the token sequence. This mechanism superiorly preserves the generative model’s internal attention priors while strictly and explicitly enforcing the localized semantic layout at every network depth.

CFG Scale. Finally, we investigate the impact of the CFG scale on generation quality across model capacities and training durations. As in Figure 5, we trace the FID and Inception Score for WiT-L (600 epochs), WiT-B (600 epochs), and WiT-B (200 epochs). Notably, it forms a distinct U-shaped curve, with the optimal point shifting depending on the model’s maturity. The fully trained WiT-L/16 model achieves its optimal FID at a low CFG scale of 2.9. WiT-B at

600 epochs peaks at a CFG of 3.1, while the early-stage WiT-B (200 epochs) relies on a much higher CFG of 3.8 to reach its minimum FID. This demonstrates that as our decoupled architecture is scaled up or trained longer, the model’s inherent semantic mapping capability becomes substantially stronger, thereby reducing the reliance on heavy CFG extrapolation.

### 5 Quantitative Analysis of Trajectory Conflict

Building upon the trajectory conflict formalized in Section 3.1, this section provides a theoretical motivation for search space contraction from a Bayes-risk perspective and empirically validates the reduced trajectory conflict.

The trajectory conflict in pure pixel-space generation stems from an excessively large and unconstrained search space. From a probabilistic perspective, WiT resolves this by introducing an explicit semantic constraint. Rather than modeling the highly entangled marginal distribution p(x|zt) directly, we utilize the semantic prior to model the conditionally constrained distribution p(x,s0|zt). Operationally, this is realized through the injection of intermediate semantic waypoints via our Just-Pixel AdaLN modulation. In terms of optimization, this structural constraint is explicitly enforced by our dual-loss formulation (Lsem and Limg). By satisfying the semantic constraint first, the generative search space for the pixel flow is shrunk. This theoretical reduction in search space directly translates to our observed behavioral advantages, most notably highly stabilized optimal transport paths and a 2.2× acceleration in training convergence.

To theoretically formalize our claim that the semantic constraint drastically shrinks the generative search space, we can analyze the ambiguity of the denoising target at any noisy state zt using the conditional variance of the target distribution. We first consider an oracle setting where the true semantic waypoint s0 is observed. In standard pixel-space x-prediction, the optimal denoiser network minimizes the Mean Squared Error (MSE) and converges to the conditional expectation:

xˆ∗(zt) = Ex∼p(x|z

t)[x]. (12)

The irreducible error of this optimal predictor represents the ambiguity without semantic conditioning, which is given by the trace of the conditional covariance matrix:

Ex ∥x − E[x|zt]∥22 zt . (13) Because the unconstrained pixel manifold is highly entangled, diverse target images map to the same noisy state zt, making Var(x|zt) exceptionally large. This manifests empirically as trajectory conflict. With oracle semantic conditioning, the Bayes-optimal predictor becomes E[x|zt,s0], and its irreducible uncertainty is bounded by:

[Var(x|zt)] = Ez

Estandard = Ez

t

t

t,s0 [Var(x|zt,s0)]. (14) As initially introduced in Equation 5, the total uncertainty in the unconstrained generation can be decomposed according to the Law of Total Variance:

Eoracle = Ez

Var(x|zt) = Es

0|zt [Var(x|zt,s0)] + Vars

0|zt (E[x|zt,s0]). (15)

By taking the expectation over zt on both sides, we obtain the relationship between the optimization burdens of the two paradigms:

0|zt(E[x|zt,s0]) . (16) Since variance is a strictly non-negative quantity, Equation 16 mathemati-

[Var(x|zt)] = Ez

t,s0 [Var(x|zt,s0)] + Ez

Vars

Ez

t

t

cally guarantees the search space contraction under the oracle condition: Eoracle ≤ Estandard. (17)

Therefore, oracle semantic conditioning reduces the Bayes ambiguity of the optimal transport problem. Motivated by this decomposition, our decoupled architecture approximates the oracle regime by explicitly predicting the semantic constraint sˆ0. As corroborated by VA-VAE [42], mapping an isotropic noise prior to a low-dimensional discriminative latent space is easier than a non-discriminative counterpart. Conditioned on this stable prediction, the primary Pixel Space Generator only needs to resolve the substantially reduced residual variance. While not a formal guarantee, this provides a theoretical explanation for how semantic waypoints structurally shrink the generative search space and untangle overlapping trajectories.

While Equation 16 characterizes ambiguity in terms of conditional variance, directly estimating Var(x|zt) in high-dimensional image space is empirically impractical. Accurately computing this quantity requires repeated samples from nearly identical continuous noisy states zt, which is severely hindered by the curse of dimensionality, as well as the handling of an astronomically large conditional covariance matrix. We therefore adopt two inference-time proxies that reflect the directional disagreement and guidance sensitivity of the learned vector field. Following standard v-prediction formulations, the estimated velocity at integration timestep t is defined as:

xˆ − zt max(1 − t,τeps)

. (18)

vˆ =

During Classifier-Free Guidance (CFG), the conditional and unconditional velocities are extrapolated using a guidance scale w:

vˆcfg = vˆuncond + w(ˆvcond − vˆuncond). (19)

To quantify the degree of conflict, we introduce two sample-level metrics measured continuously across the integration steps:

– Pairwise Directional Conflict: This measures the geometric opposition between the vector field conditioned on the target label y and an alternative counterfactual label yalt. We compute the cosine distance:

Cpair(t) = 0.5 · (1 − cos(ˆvcond,vˆalt)). (20)

Higher values indicate severe gradient interference, where paths destined for different semantic endpoints spatially overlap and pull the trajectory in contradictory directions.

Table 4: Quantitative comparison of trajectory conflict between JiT [22] and WiT.

###### Metric Position JiT [22] WiT Difference

Pairwise Conflict Midpoint (t ≈ 0.5) 1.294e-4 8.363e-5 1.55× more stable Pairwise Conflict Maximum Peak 8.532e-3 5.262e-3 1.62× more stable CFG Rel L2 Distance Midpoint (t ≈ 0.5) 1.304e-2 1.159e-2 1.13× more stable

– CFG Relative L2 Distance: This measures the magnitude of divergence be-

tween the conditional and unconditional vector fields:

Crel(t) = ∥vˆcond − vˆuncond∥2 ∥vˆcond∥2

. (21)

We evaluate these metrics over the course of the full generation trajectory using a 50-step Heun solver. For each integration step ti, we compute vˆcond, vˆuncond, and vˆalt, where the counterfactual label is defined as yalt = (y + stride) mod C, with C representing the total number of classes. The metrics are averaged across multiple batches to yield stable trajectory curves over t ∈ [0,1]. We compare our proposed WiT against the direct pixel-space baseline, JiT [22].

Table 4 summarizes the trajectory conflict metrics at critical points during the generative flow: the integration midpoint (where t ≈ 0.5) and the maximum peak conflict observed across the entire timeline. As demonstrated in Table 4, standard pixel-space Flow Matching (like JiT [22]) suffers from overlapping trajectories. By successfully anchoring the generation trajectories to low-dimensional semantic waypoints, WiT structurally untangles these paths, demonstrating approximately 1.62× higher stability in pairwise conflict at the peak integration phase. This validates our theoretical framework: satisfying the explicit semantic constraint narrows the search space, yielding a smoother, highly separable vector field and robust visual structural integrity.

### 6 Conclusion

In this paper, we presented Waypoint Diffusion Transformers (WiT), a novel generative paradigm designed to resolve the severe trajectory conflict inherent in pixel-space Flow Matching. Recognizing that the raw pixel manifold is naturally entangled and resistant to direct regularization, we explicitly decoupled the generative process into semantic navigation and high-realistic texture synthesis. By projecting the discriminative feature space of pre-trained vision models into compact semantic waypoints, WiT successfully factors the complex noiseto-pixel optimal transport path. During integration, a lightweight Waypoints Generator dynamically infers these structural anchors, which subsequently provide spatially-varying guidance to the primary diffusion transformer via our proposed Just-Pixel AdaLN mechanism. Extensive experiments on ImageNet 256×256 demonstrate that WiT achieves a state-of-the-art performance among pure pixel-space architectures, surpassing even heavy VAE-compressed latent models, while delivering a remarkable 2.2× training speedup over JiT [22].

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

[Figure 159]

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

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

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

### Acknowledgements

We would like to thank Qiming Hu for the insightful discussions and feedback. The computational resources of this work was partially supported by TPU Research Cloud (TRC).

### References

- 1. Albergo, M.S., Vanden-Eijnden, E.: Building normalizing flows with stochastic interpolants. In: ICLR (2023)
- 2. Baade, A., Chan, E.R., Sargent, K., Chen, C., Johnson, J., Adeli, E., Fei-Fei, L.: Latent forcing: Reordering the diffusion trajectory for pixel-space image generation. arXiv preprint arXiv:2602.11401 (2026)
- 3. Black Forest Labs: FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison (2025), https://bfl.ai/research/representationcomparison
- 4. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv:2311.15127 (2023)
- 5. Chen, S., Ge, C., Zhang, S., Sun, P., Luo, P.: PixelFlow: Pixel-space generative models with flow. arXiv:2504.07963 (2025)
- 6. Chen, Z., Zhu, J., Chen, X., Zhang, J., Hu, X., Zhao, H., Wang, C., Yang, J., Tai, Y.: Dip: Taming diffusion models in pixel space. arXiv preprint arXiv:2511.18822

(2025)

- 7. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A large-scale hierarchical image database. In: CVPR (2009)
- 8. Dhariwal, P., Nichol, A.: Diffusion models beat GANs on image synthesis (2021)
- 9. Dinh, L., Sohl-Dickstein, J., Bengio, S.: Density estimation using real nvp. In: International Conference on Learning Representations (2017)
- 10. Esser, P., Rombach, R., Ommer, B.: Taming Transformers for high-resolution image synthesis. In: CVPR (2021)
- 11. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial networks. Communications of the ACM 63(11), 139–144 (2020)
- 12. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models (2020)
- 13. Ho, J., Salimans, T.: Classifier-free diffusion guidance. In: NeurIPS Workshops

(2021)

- 14. Hoogeboom, E., Heek, J., Salimans, T.: simple diffusion: End-to-end diffusion for high resolution images. ICML (2023)
- 15. Hoogeboom, E., Mensink, T., Heek, J., Lamerigts, K., Gao, R., Salimans, T.: Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion. In: CVPR (2025)
- 16. Jabri, A., Fleet, D., Chen, T.: Scalable adaptive computation for iterative generation. In: ICML (2023)
- 17. Kingma, D.P., Dhariwal, P.: Glow: Generative flow with invertible 1x1 convolutions. In: Advances in neural information processing systems (2018)
- 18. Kynkäänniemi, T., Aittala, M., Karras, T., Laine, S., Aila, T., Lehtinen, J.: Applying guidance in a limited interval improves sample and distribution quality in diffusion models (2024)

- 19. Lei, J., Liu, K., Berner, J., Yu, H., Zheng, H., Wu, J., Chu, X.: Advancing end-to-end pixel space generative modeling via self-supervised pre-training. arXiv:2510.12586 (2025)
- 20. Leng, X., Singh, J., Hou, Y., Xing, Z., Xie, S., Zheng, L.: Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483 (2025)
- 21. Leng, X., Singh, J., Murdock, R., Smith, E., Li, R., Hou, Y., Xing, Z., Xie, S., Zheng, L.: Family of end-to-end tuned vaes for supercharging t2i diffusion transformers. https://end2end-diffusion.github.io/repa-e-t2i/ (2025)
- 22. Li, T., He, K.: Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720 (2025)
- 23. Li, T., Sun, Q., Fan, L., He, K.: Fractal generative models. arXiv:2502.17437 (2025)
- 24. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: ICLR (2023)
- 25. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. In: ICLR (2023)
- 26. Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E., Xie, S.: SiT: Exploring flow and diffusion-based generative models with scalable interpolant Transformers. In: ECCV (2024)
- 27. Ma, Z., Wei, L., Wang, S., Zhang, S., Tian, Q.: Deco: Frequency-decoupled pixel diffusion for end-to-end image generation. arXiv preprint arXiv:2511.19365 (2025)
- 28. Mentzer, F., Minnen, D., Agustsson, E., Tschannen, M.: Finite scalar quantization: VQ-VAE made simple. In: ICLR (2024)
- 29. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2023)
- 30. Peebles, W., Xie, S.: Scalable diffusion models with Transformers. In: ICCV (2023)
- 31. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: CVPR (2022)
- 32. Salimans, T., Ho, J.: Progressive distillation for fast sampling of diffusion models. In: ICLR (2022)
- 33. Sauer, A., Schwarz, K., Geiger, A.: StyleGAN-XL: Scaling StyleGAN to large diverse datasets. In: SIGGRAPH (2022)
- 34. Shi, M., Wang, H., Zheng, W., Yuan, Z., Wu, X., Wang, X., Wan, P., Zhou, J., Lu, J.: Latent diffusion model without variational autoencoder. arXiv:2510.15301

(2025)

- 35. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: DINOv3 (2025), https://arxiv.org/abs/2508.10104
- 36. Singh, J., Leng, X., Wu, Z., Zheng, L., Zhang, R., Shechtman, E., Xie, S.: What matters for representation alignment: Global information or spatial structure? arXiv preprint arXiv:2512.10794 (2025)
- 37. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic differential equations. In: ICLR

(2021)

- 38. Tong, A., FATRAS, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research

(2024), https://openreview.net/forum?id=CD9Snc73AW, expert Certification

- 39. Tschannen, M., Pinto, A.S., Kolesnikov, A.: JetFormer: an autoregressive generative model of raw images and text. In: ICLR (2025)
- 40. Wang, S., Gao, Z., Zhu, C., Huang, W., Wang, L.: PixNerd: Pixel neural field diffusion. arXiv:2507.23268 (2025)
- 41. Wang, S., Tian, Z., Huang, W., Wang, L.: DDT: Decoupled diffusion Transformer. arXiv:2504.05741 (2025)
- 42. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In: CVPR (2025)
- 43. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training diffusion Transformers is easier than you think. In: ICLR (2025)
- 44. Yu, Y., Xiong, W., Nie, W., Sheng, Y., Liu, S., Luo, J.: Pixeldit: Pixel diffusion transformers for image generation. arXiv preprint arXiv:2511.20645 (2025)
- 45. Zheng, B., Ma, N., Tong, S., Xie, S.: Diffusion Transformers with representation autoencoders. arXiv:2510.11690 (2025)

