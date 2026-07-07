# arXiv:2512.22323v1[cs.CV]26Dec2025

## SpotEdit: Selective Region Editing in Diffusion Transformers

Zhibin Qin1* Zhenxiong Tan1* Zeqing Wang1 Songhua Liu2 Xinchao Wang1† 1National University of Singapore 2Shanghai Jiao Tong University

{e1352224, zhenxiong, zeqing.wang}@u.nus.edu liusonghua@sjtu.edu.cn xinchao@nus.edu.sg

Source Image Results Regenerated region Source Image Results Regenerated region

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

soccerapersontosunfloweradd

topineapplecupasuitcaseadd

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1. Examples of edited images by SpotEdit. The blue area reveals the regenerated region.

### Abstract

### 1. Introduction

Diffusion Transformer models have significantly advanced image editing by encoding conditional images and integrating them into transformer layers. However, most edits involve modifying only small regions, while current methods uniformly process and denoise all tokens at every timestep, causing redundant computation and potentially degrading unchanged areas. This raises a fundamental question: Is it truly necessary to regenerate every region during editing? To address this, we propose SpotEdit, a training-free diffusion editing framework that selectively updates only the modified regions. SpotEdit comprises two key components: SpotSelector identifies stable regions via perceptual similarity and skips their computation by reusing conditional image features; SpotFusion adaptively blends these features with edited tokens through a dynamic fusion mechanism, preserving contextual coherence and editing quality. By reducing unnecessary computation and maintaining high fidelity in unmodified areas, SpotEdit achieves efficient and precise image editing. Project page is available at https://biangbiang0321.github.io/ SpotEdit.github.io/.

*Eqaul contribution. †Corresponding author.

Diffusion models have shown outstanding performance in image generation tasks [10, 16, 31]. By leveraging Diffusion Transformer (DiTs) architectures [28], the generation quality and flexibility have been further enhanced. Building on these advances, encoding condition images and integrating them directly into transformers [36, 37] has become a mainstream technique for image editing [15, 39]. This strategy allows simple yet effective editing without relying on manually provided masks, greatly improving usability in practical applications.

However, in most image editing tasks, only a small region of the image requires modification, while the majority of areas remain unchanged. Yet, existing approaches uniformly follow a full-image regeneration paradigm, indiscriminately denoising every region from random noise, including those that do not need editing. Such uniform processing introduces two prominent drawbacks: First, redundant computations in non-edited regions may inadvertently produce subtle artifacts; second, significant computational resources are wasted by processing unmodified areas. These issues naturally lead us to reconsider the current editing paradigm and pose a critical question: Is it truly necessary to regenerate every region of the image during an editing task?

source image T = 50 T = 48 T = 46 T = 44 T = 0

###### T = 42

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

...

- Figure 2. Reconstruction results at different timesteps.(Totalsteps T=50, seed=42, prompt = “Add a scarf to the dog.”) Each reconstruction

xˆ0 is estimated following the rectified flow formulation: Xˆ0 = xt − t vθ(xt, c, t). It can be observed that some regions become sharp and visually consistent with the original image even at very early stages, while other regions continue to evolve until the final timestep.

To address the above problems, we begin by analyzing the temporal convergence patterns of latent representations during diffusion. Figure 2 reveal that, in partial editing tasks, non-edited regions stabilize quickly, converging at early diffusion timesteps. This observation naturally motivates a more efficient editing strategy: Edit only what needs to be edited.

Guided by this principle, we propose SpotEdit , a mechanism designed to automatically detect stable, non-edited regions and reuse their corresponding condition image latent features without computing them in DiTs, thereby avoiding redundant regeneration computation. Implementing this idea, however, raises two critical challenges: 1) How to efficiently and accurately identify non-edited regions? 2) How to enable models to dynamically focus computation only on the regions requiring modification?

- For challenge 1, we propose SpotSelector, an adaptive

mechanism that dynamically identifies stable regions during diffusion iterations. Specifically, SpotSelector computes a perceptual similarity score for each latent token by measuring the perceptual distance between the reconstructed fully denoised latent and the corresponding condition image latent via VAE decoder layers. Regions whose perceptual distance is below a threshold are automatically classified as non-edited regions. This approach eliminates manual masking and directly leverages the diffusion dynamics observed in our analysis, ensuring that the identified regions align with the model’s generative process.

- For challenge 2, we introduce SpotFusion, a context fu-

sion mechanism that restores missing contextual information by adaptively blending features from the condition image. Leveraging the high feature relationship between nonedited regions and corresponding condition image regions across diffusion steps, SpotFusion dynamically modulates the contribution of the reference based on the current denoising timestep, relying more on the reference early in the process and gradually shifting to the current estimate as generation progresses. This design preserves time coherence in features while avoiding potential boundary arti-

facts, without requiring additional computation for unedited regions.

Experimental results demonstrate that our SpotEdit achieves a speedup of 1.7× for imgEdit-Benchmark[43]and

- 1.9× for PIE-Bench++[12] on the base model FLUX.1Kontext[15], while maintaining quality comparable to the original model. Qualitative results (see Figure 1) further indicate that SpotEdit perfectly preserves non-edited regions and produces clean, localized edits.

Our primary contributions are summarized as follows:

- (i) We propose SpotSelector, a perceptual-similaritybased method for dynamically distinguishing nonedited regions, removing the need for manual masks.
- (ii) We introduce SpotFusion, an adaptive fusion mechanism ensuring temporal coherence and contextual consistency in partially-edited diffusion processes.
- (iii) We demonstrate that our combined framework, SpotEdit, enables selective diffusion-based editing, significantly accelerating inference while preserving the fidelity and quality of edits.

- 2. Related works 2.1. Precise image editing

Image editing has long been a central need in workflows. [29]Following the remarkable breakthroughs of diffusion models [11, 42] in image generation, a growing body of research has focused on adapting these models to serve image editing tasks. Early approaches, such as ControlNet [46], injected external control signals into U-Net [32] to enable robust and controllable editing outcomes.

With the advancement of diffusion models, inversionbased methods [8, 9, 22, 25, 38, 41, 48] have become the mainstream paradigm. These approaches operate by injecting noise into the condition image and then denoising it under the target textual prompt to produce the edited result[40]. However, the pure noise-addition and denoising procedure often leads to unwanted global changes in the image.

To better adhere to localized editing requirements, recent works introduce KV-injection techniques [1, 9, 13, 22, 26, 38, 48] that preserve reference features throughout the denoising process. Inversion KV is injected into the matching timesteps of the denoising phase, effectively transferring the structural and semantic information from the condition image to the edited output [8, 22, 38, 48].

In parallel, another line of work adopts mask-based control mechanisms [34, 35]. These approaches solve the editing task by inpainting and explicitly specify editable regions via a binary mask[7], allowing the denoising process to focus computations only within the masked area while directly retaining unedited pixels from the condition image [3, 36, 37, 45, 48]. Such strategies offer fine-grained control over editing region; however, they limits flexibility and applicability in real-world scenarios.

More recently, a new generation of image editing models [15, 39] have emerged, leveraging Diffusion Transformer architectures [28] to jointly process condition images and noise inputs. This design requires only high-level editing instructions and can directly modify the corresponding semantic regions [2, 8, 27], without relying on any manually provided masks.

Despite the rapid progress of precise image editing, all the mentioned approaches share a fundamental limitation: they regenerate the entire image at every denoising timestep, regardless of which regions actually require modification.

#### 2.2. Acceleration of efficient editing

Existing acceleration techniques for diffusion and diffusion–transformer models mainly operate at the full-token level[23], without distinguishing between edited and nonedited regions. Acceleration methods [4–6, 18, 19, 24, 33, 44] improve efficiency by reusing or approximating intermediate features across timesteps. While effective for generation, these approaches treat all image tokens uniformly, accelerating every spatial position regardless of its semantic importance. However, image editing tasks naturally contain heterogeneous regions: only a small subset of tokens needs modification, whereas the majority should remain unchanged. Quality degradation introduced by aggressive fulltoken acceleration tends to disproportionately affect the semantically important edited regions, leading to visibly inferior results even when background distortion is mild. Consequently, full-token acceleration provides limited practical benefit for editing, as the gain in speed often comes with a noticeable loss in fidelity.

On the other hand, recent token-space strategies such as ToCa [50], DUCA [49], and RAS [21] explore reducing computational redundancy by compacting tokens or adjusting sampling steps based on token saliency. Although these approaches operate at the token level, they still follow the

paradigm of reducing timesteps or compressing token representations, rather than explicitly skipping computation for non-edited regions. As a result, they do not fully exploit the inherent sparsity of image editing tasks and cannot direct computational resources to edited regions alone. Thus, none of the above methods achieve true region-aware acceleration, where the model selectively updates only the edited tokens while bypassing unnecessary computation for preserved regions.

### 3. Preliminary

Flow Matching [17] formulates generative modeling as learning a deterministic flow that continuously transports samples from a source distribution p1, typically Gaussian noise, to a target distribution p0 representing real images. A time-dependent velocity field vθ(X,C,t) defines transformation through the ordinary differential equation

dXt dt

= vθ(Xt,C,t), X1 ∼ p1, X0 ∼ p0, (1)

where C denotes the condition that contains the reference image and the instruction describing the desired edit. During inference, the model generates samples by integrating Eq. 1 backward in time from X1 to X0.

Rectified Flow [20] simplifies this formulation by assuming a linear interpolation between the source and target;

Xt = (1 − t)X0 + tX1, vθ(Xt,C,t) = X1 − X0 (2)

with endpoints sampled as X0 ∼ p0 drawn from the data distribution and X1 ∼ p1 following N(0,I). On this linear path, the instantaneous target velocity becomes ut ≡ X1 − X0, resulting in a constant and easily learnable flow field. For inference, the reverse process is approximated. Xt

i − (ti − ti−1)vθ(Xt

= Xt

,C,ti), i = T,...,1

i−1

i

(3) which progressively transports the initial noise sample X1 to the final image X0 under the guidance of condition C.

### 4. Methodology

As discussed in Section 1, current mainstream instructionbased editing models(FLUX-Kontext[15], Qwen-ImageEditing[39]) regenerate the entire image during the editing task, without classifying whether the regions are semantically meant to be preserved or modified, which not only introduces potential background noises and wastes computation on non-edited regions.

To address these two challenges explicitly, we propose SpotEdit , a training-free framework that efficiently skips redundant computation while preserving high fidelity and editing quality. SpotEdit consists of two parts; (1) SpotSelector for detecting non-edited regions and skipping their

[Figure 20]

###### Condition image

| | | | | | | | |
|---|---|---|---|---|---|---|---|

[Figure 21]

| | | | | | | | |
|---|---|---|---|---|---|---|---|

match

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |
| | |
| | |

[Figure 22]

Decoder

| | |
|---|---|
| | |
| | |
| | |

|DiT|
|---|

SpotStep

Spot

Spot Selector

DiT

Selector DiT

| | | |
|---|---|---|
| | | |
| | | |
| | | |

| |
|---|
| |

SpotEdit step SpotFusion SpotEdit step

SpotFusion

Initial step

Regenerated region token feature fusion

non-edited region token condition image token

reused image token

| |
|---|

SpotSelector

###### SpotFusion

Previous Step Current Step

DiT DiT

[Figure 23]

[Figure 24]

[Figure 25]

|DiT Block|
|---|

|DiT Block|
|---|

SpotFusion

|Query Key/Value<br><br>|
|---|

| | |
|---|---|
| | |

| | |
|---|---|
| | |

L2 Distance

Key/Value Query

cache

| | | | |
|---|---|---|---|

decoder decoder

| | | | |
|---|---|---|---|

| | | | |
|---|---|---|---|

[Figure 26]

SpotFusion

| |
|---|
| |

| | |
|---|---|
| | |

LPIPS-like score

- Figure 3. Overview of SpotEdit. The process consists of three stages: (1) Initial Steps: the model performs standard DiT denoising on all image tokens under the editing instruction, while caching the KV values for Spotfusion. (2) Spot Steps: SpotSelector dynamically identifies regenerated region and non-edited region tokens using LPIPS-like perceptual scores. Non-edited region tokens are skipped by DiT computation, while regenerated region tokens are generated iteratively with SpotFusion, which builds a temporally consistent condition cache by fusing cached non-edited region KV values with condition image KV values. (3) Token Replacement: Regenerated tokens are updated through DiT, and non-edited tokens are directly covered by the corresponding reused tokens before decoding into an image, ensuring background fidelity with reduced computation.

computation, then reusing the original image information; (2) Spotfusion for providing temporally consistent context to regenerated tokens without advancing a full denoising process for the non-edited region.

- 4.1. SpotSelector

And building on this reconstructed Xˆ0, we can decode it into an image.

As shown in Figure 2, diffusion models exhibit a clear coarse to fine reconstruction pattern, where different spatial regions converge at different speeds. Regions that rapidly become sharp and visually identical with the condition image at early timesteps have effectively stabilized, which indicates them to be non-edited areas. In contrast, regions subject to editing continue to evolve throughout the denoising process under the given instruction. A natural implication of this heterogeneous convergence behavior is that the model already exposes which regions are stable and which are still under refinement.

A central question in region-aware editing is determining which tokens truly require modification and which tokens should be preserved. SpotSelector addresses this problem by identifying non-edited regions early in the denoising process and routing them away from full DiT computation. By doing so, it enables SpotEdit to avoid unnecessary updates on stable regions while focusing computation on tokens that actually need to change.

Building on this signal, Spotselector dynamically identifies and separates tokens corresponding to non-edited regions from those that require modification by comparing its early reconstruction Xˆ0 with the condition image latent Y . Tokens that are highly consistent with the condition image and stabilize early are classified as non-edit tokens, while the rest are treated as regenerated tokens. The latter con-

To detect these non-edited regions, we leverage a key property of Rectified Flow: under its linear interpolation dynamics, the latent state at time t admits a closed-form relation to the fully denoised latent state X0.

Xˆ0 = Xt

,C,ti), i = T,...,1 (4)

i − ti × vθ(Xt

i

tinue through the full computation of DiT, while the former are routed away from full sampling updates and reuse the condition image token feature, thereby reducing unnecessary computation. This selective handling preserves nonedited region fidelity and yields substantial efficiency gains. Although early stabilization provides a strong signal for identifying non-edited regions, efficiently and reliably selecting such tokens remains challenging, as distances in latent space do not necessarily reflect human-perceived similarity. To obtain a similarity measure that better aligns with perceptual differences, we introduce a token-level perceptual score inspired by LPIPS[47], computed from VAE[14] decoder activations. Let ϕl(·) denote the feature map extracted from decoder layer l. The perceptual deviation of token i is defined as

2 2

wl ϕ ˆl(Xˆ0)i − ϕˆl(Y )i

, (5)

sLPIPS(i) =

l∈L

where wl are layer weights and hats denote feature normalization. This LPIPS-like formulation maps differences in early decoder features to a token-wise perceptual discrepancy that more faithfully reflects visually meaningful changes.

Having obtained a perceptual token score, we use it to decide whether each token should be actively denoised or directly reused from the reference. Specifically, given a threshold τ, we define the binary routing indicator

rt,i = 1 s(LPIPSt) (i) ≤ τ , Rt = {i : rt,i = 1}, At = {i : rt,i = 0}.

(6)

Tokens in the regenerated set At follow the full reverseintegration update (Eq. (3)) to produce the desired edits. In contrast, tokens in the non-edited set Rt are entirely removed from the DiT computation. This design avoids redundant computation on non-edited regions.

At the final denoising step, we apply a lightweight latent consolidation. Instead of propagating Rt through the network, all non-edited tokens are directly overwritten with their counterparts from the condition image latent before decoding into pixel space. This guarantees that non-edited regions remain visually consistent with the condition image.

#### 4.2. SpotFusion

Routing non-edited tokens out of the DiT computation eliminates redundant updates but also removes their contextual contribution to regenerated regions. Because diffusion transformers rely heavily on cross-token attention to maintain spatial coherence, such naive removal can lead to noticeable degradation in edit quality.

A natural alternative is to cache and reuse the Key–Value (KV) pairs of non-edited tokens or reference-image tokens

during attention. However, this approach introduces a temporal inconsistency: the cached KV pairs represent feature states frozen at the timestep when they were stored, while the hidden states of regenerated tokens continue to evolve throughout denoising.

Dual layer

Single layer

|PC1 (hx)<br><br>PC2 (hx)<br><br>PC1 (hy)<br><br>PC2 (hy)<br><br><br>| | |
|---|---|---|
| | | |

| | | |
|---|---|---|
| | | |

PCAComponentValue

1 0

1 0

t [0, 1]

t [0, 1]

Figure 4. PCA trajectories of hidden states of non-edited tokens across dual stream layers and single stream layers. As denoising progresses, the trajectory of the generated image in the non-edited region (x) gradually approaches that of the condition image (y), indicating that their latent representations progressively align. Eventually, both trajectories overlap, suggesting that unedited regions converge to the same latent subspace, thereby maintaining strong background consistency and semantic preservation.

Unlike language transformers with static embeddings, DiTs maintain drift with timestep, causing cached KV pairs to become increasingly misaligned with the evolving edited-region features. This leads to temporal drift, context mismatch, and statistical discontinuity across layers. As demonstrated in our ablation study (Fig. 6), static KV caching results in clear degradation in editing fidelity.

To design a temporally consistent reuse mechanism, we first examine the temporal behavior of hidden states in nonedited regions. As shown in Fig. 4, the hidden representations of non-edited tokens (x-branch) and condition image tokens (y-branch) follow closely aligned trajectories in principal-component space:

- 1. the reference branch exhibits strong temporal stability,
- 2. non-edited tokens rapidly align with the reference trajectory after an initial transient phase,
- 3. both h(xb,t) and h(yb,t) converge to the same latent subspace as t→0.

These observations suggest that non-edited regions evolve smoothly toward the condition image and therefore admit a consistent feature representation that can be gradually reinforced rather than statically cached.

SpotFusion Motivated by this insight, we propose SpotFusion, a temporally consistent feature-reuse mechanism that blends cached non-edited features with referenceimage features across timesteps. After the initial denoising steps, we cache the KV pairs of both the reference branch and the non-edited region tokens. These KV maps form condition cache reused throughout the denoising process.

To avoid temporal mismatch with the evolving regenerated tokens, SpotFusion smoothly interpolates cached nonedited features toward reference features at every block and timestep. Let h˜(xb,t+1) denote the cached hidden state at block b from step t+1, and h(yb) the corresponding reference feature. For non-edited tokens, the updated hidden state is:

h˜(xb,t) = α(t)h˜(xb,t+1) + (1 − α(t))h(yb), (7)

where α(t) = cos2 π2t gradually shifts the representation from cached features to reference features as denoising pro-

ceeds. This interpolation is applied directly to the KV pairs:

### 5. Experiments

#### 5.1. Settings

Implementation details All experiments are conducted on one single NVIDIA H200 GPU under the environment of CUDA 12.8 and PyTorch 2.9, scheduled with T = 50 steps and a resolution of 1024 × 1024 at random seed 42. SpotEdit employs a SpotSelector threshold τ = 0.2 of token saliency and a token-fusion weighting function α(t) = cos2(πt/2). Condition image token features are cached after the initial stage at t=4 and reused during the remaining timesteps.

Kt,i(b) ← α(t)Kt(+1b) ,i + (1 − α(t))Ky,i(b), (8) Vt,i(b) ← α(t)Vt(+1b),i + (1 − α(t))Vy,i(b). (9)

SpotFusion thus maintains coherent contextual signals for regenerated tokens, ensures temporal consistency in non-edited regions, and eliminates redundant computation associated with reprocessing the reference branch. Together, these advantages allow the model to preserve background fidelity while achieving substantial efficiency gains.

Datasets We evaluate SpotEdit on two image-editing benchmarks: PIE-Bench++ [12]and imgEdit benchmark [43].For PIE-Bench++, we select three subsets related to partial editing tasks: change object, add object, and delete object. For imgEdit benchmark, we focus on the single-turn setting and select the subsets most relevant to partial editing tasks, including adjust, background, extract, remove, replace, add, compose, and action. All images are standardized to 1024 × 1024 resolution.

Partial attention calculation With SpotFusion providing temporally consistent KV caches for non-edited and condition image tokens, The DiT can perform attention using a small set of queries while retaining full spatial context.

During denoising, only the regenerated tokens require forward propagation through the DiT. Thus, the Query set is restricted to the regenerated tokens At together with the instruction–prompt tokens P:

Qactive = [QP, QA

].

t

Other tokens in non-edited regions and the condition image are skipped from computation. However, their contextual influence must remain available, so the Key–Value sets are completed by concatenating the cached features:

, VYC ]. Attention is then computed only on the active queries:

##### , KRC

, KYC ],Vfull = [VP, VA

##### , VRC

Kfull = [KP, KA

t

t

t

t

Attn = softmax

Qactive Kfull⊤

√

d

Vfull. (10)

Here, the non-edited region and condition image token KV pairs (KRC

) and (KYC,VYC) are retrieved directly from the SpotFusion condition cache.

##### ,VRC

t

t

This partial-attention scheme concentrates computation precisely where edits occur, preserving contextual coherence through cached KV maps while avoiding redundant propagation through non-edited and reference regions.

Evaluation metrics We construct a comprehensive evaluation protocol that jointly measures editing quality and computational efficiency. For quality assessment, we adopt four complementary criteria—semantic alignment, structural consistency, perceptual fidelity, and overall editing accuracy. Semantic alignment between the edited results and textual instructions is evaluated using CLIP similarity [30], while PSNR, SSIM, and DISTS quantify structural preservation and perceptual fidelity. Together, these metrics capture how well the edits follow the intended instructions while maintaining consistency in non-edited regions. For efficiency assessment, we additionally report the average inference latency and the relative speedup.

Baselines We compare SpotEdit against a diverse set of representative baselines spanning both cache acceleration and percise editing paradigms. We include cache accelerating methods such as TaylorSeer [19] and TeaCache [18], which improve diffusion inference efficiency through feature reuse, forecasting, or adaptive caching strategies. Also, we evaluate precise-editing approaches such as Follow-Your-Shape [22], which emphasize spatially accurate, mask-free editing by explicitly preserving structural cues during generation. Specifically, we evaluate FollowYourShape under two ControlNet configurations. The “single” setting denotes that the model is driven by a single ControlNet, where we adopt the depth ControlNet as in the original implementation. The “multi” setting instead employs a multi-ControlNet composition, where multiple

###### imgEdit-Benchmark[43] PIE-Bench++[12]

Baselines

CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑ CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑ Original Inference 0.699 0.67 16.40 0.17 1.00× 0.741 0.791 18.76 0.136 1.00×

FollowYourShape[22] (single) 0.686 (↓-0.013) 0.47 (↓-0.20) 11.73 (↓-4.67) 0.27 (↑0.10) 0.33× (↓-0.67) 0.714 (↓-0.27) 0.578 (↓-0.213) 12.91 (↓-5.85) 0.295 (↑0.159) 0.34× (↓-0.66) FollowYourShape[22] (multi) 0.688 (↓-0.011) 0.47 (↓-0.20) 11.47 (↓-4.93) 0.28 (↑0.11) 0.27× (↓-0.73) 0.549 (↓-0.192) 0.583 (↓-0.208) 12.58 (↓-6.18) 0.298 (↑0.162) 0.28× (↓-0.72) TeaCache[18] 0.698 (↓0.001) 0.60 (↓-0.07) 15.02 (↓-1.38) 0.21 (↑0.04) 3.43× (↑2.43) 0.735 (↓-0.006) 0.764 (↓-0.027) 18.89 (↑0.13) 0.144 (↑0.08) 3.59× (↑2.59) TaylorSeer[19] 0.666 (↓-0.033) 0.52 (↓-0.15) 14.36 (↓-2.04) 0.37 (↑0.20) 3.61× (↑2.61) 0.741 (0.00) 0.755 (↓-0.036) 17.81 (↓-0.95) 0.159 (↑0.023) 3.86× (↑2.86) Ours 0.699 (0.00) 0.67 (0.00) 16.45 (↑0.05) 0.16 (↓-0.01) 1.67× (↑0.67) 0.741 (0.00) 0.792 (↑0.01) 18.73 (↓-0.03) 0.136 (0.00) 1.95× (↑0.95)

- Table 1. Comparison of models on imgEdit-Benchmark and PIE-Bench++.

Baselines Adjust Background Extract Remove Replace Add Compose Action Average Original Inference 3.73 3.64 4.09 4.80 4.46 3.78 2.75 4.01 3.91

FollowYourShape[22] (single) 3.27 (↓-0.46) 3.27 (↓-0.37) 3.10 (↓-0.99) 4.67 (↓-0.13) 4.10 (↓-0.36) 3.16 (↓-0.62) 2.32 (↓-0.43) 3.54 (↓-0.47) 3.43 (↓-0.48) FollowYourShape[22] (multi) 3.15 (↓-0.58) 3.31 (↓-0.33) 3.13 (↓-0.96) 4.77 (↓-0.03) 4.00 (↓-0.46) 3.18 (↓-0.60) 2.38 (↓-0.37) 3.36 (↓-0.65) 3.41 (↓-0.50) TaylorSeer[19] 3.51 (↓-0.22) 3.39 (↓-0.25) 3.56 (↓-0.53) 4.66 (↓-0.14) 4.19 (↓-0.27) 3.66 (↓-0.12) 2.41 (↓-0.34) 3.96 (↓-0.05) 3.67 (↓-0.24) TeaCache[18] 3.55 (↓-0.18) 3.41 (↓-0.23) 3.65 (↓-0.44) 4.64 (↓-0.16) 4.19 (↓-0.27) 3.51 (↓-0.27) 2.51 (↓-0.24) 4.17 (↑+0.16) 3.70 (↓-0.21) Ours 3.57 (↓-0.16) 3.43 (↓-0.21) 3.60 (↓-0.49) 4.72 (↓-0.08) 4.41 (↓-0.05) 3.64 (↓-0.14) 2.65 (↓-0.10) 4.14 (↑+0.13) 3.77 (↓-0.14)

- Table 2. VL score comparison of eight subsets of imgEdit-Benchmark.

structural guidance branches are jointly used to provide richer geometric and semantic constraints.

#### 5.2. Results analyse

Quantitative results Table 1 summarizes the results on imgEdit-Benchmark and PIE-Bench++.SpotEdit achieves a highly favorable trade-off between editing fidelity and computational efficiency by explicitly identifying and skipping non-edited regions during denoising. On IMGeditBenchmark, SpotEdit matches or slightly surpasses the original inference results, while achieving a 1.67× speedup. In contrast, cache-based methods such as TaylorSeer obtain higher acceleration (3.61×) but suffer from clear all degradation, while precise-editing methods like FollowYourShape introduce severe distortion to non-edited areas. Similar trends hold on PIE-Bench++, where SpotEdit maintains 18.73 PSNR and 0.792 SSIMc, outperforming all baselines in quality while still reaching 1.95× speedup.

As vision-language (VL) scores across eight instruction types on the IMGedit-Benchmark (Table 2) illustrated, SpotEdit achieves the highest VL score (3.77) among all methods, with strong performance on complex instructions like Replace (4.41) and Compose (2.65), and only a slight drop (-0.14) from original inference. This demonstrates that SpotEdit effectively completes the intended edits while preserving structural consistency. These results show that SpotEdit allocates computation only to regions requiring change, preserving the structure and consistency of unedited areas, validating our core principle—edit what needs to be edited

Qualitative results Figure 5 shows baseline methods struggle to keep the background intact. Our method preserves non-edited regions almost perfectly while delivering

the desired edit. Figure 1 demonstrates that our method works well across diverse editing instructions, and only regenerates the needed region.

#### 5.3. Ablation study

We conduct ablation studies on three key components: Token Fusion, Condition Cache, and Reset mechanism.

Token fusion We compare Spot Fusion with two variants: (1) Naive Skip, which directly discards non-edited tokens without caching, leading to loss of context for regenerated regions; (2) Static Token Fusion, which reuses cached unedited tokens without aligning them with the condition image. As shown in Figure 6, both variants introduce artifacts or inconsistency. In contrast, Spot Fusion preserves background fidelity and edit quality, highlighting the necessity of adaptive token-level blending.

Method CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑

Default 0.741 0.792 18.730 0.136 1.95 w/o Reset 0.738 0.782 17.10 0.154 2.25

Table 3. Ablation on Reset mechanism.

Condition cache We compare SpotEdit’s full condition caching with a variant that only caches non-edited tokens, while recomputing condition image features at each timestep. Our approach caches both condition image and unedited region features, enabling greater reuse and faster inference. Though the uncached variant achieves slightly better PSNR (19.15 vs. 18.73), it is notably slower (1.24× vs. 1.95×). This shows that caching the condition image contributes significantly to generation efficiency without harming visual quality.

###### Source Image

Follow your shape Teacache Taylorsser Ours

###### regenerated region

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

toslicedsteakred

Changed the whole background

Fail to keep color saturation

Perfectly preserve the non-edited region

Fail to keep color

Figure 5. Non-edited preservation comparison across different models. Prior methods either modify unnecessary background regions or distort color consistency, whereas our method preserves non-edited areas faithfully while applying accurate edits.

###### imgEdit-Benchmark[43] PIE-Bench++[12]

Baselines

CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑ CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑

Qwen-Image-Edit 0.688 0.522 14.23 0.22 1.00× 0.724 0.74 18.32 0.16 1.00× Ours 0.686 (↑0.002) 0.524 (↑0.02) 14.24 (↑0.01) 0.21 (↓-0.01) 1.59× (↑0.59) 0.722 (↓0.002) 0.77 (↑0.03) 19.40 (↑1.08) 0.15 (↓-0.01) 1.72× (↑0.72)

Table 4. Comparison between Qwen-Image-Edit and SpotEdit on imgEdit-Benchmark and PIE-Bench++.

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Reference

add sunglass black eyebrow to white

chair to sofa lamp to umbrella

Remove trees Add a girl

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

+ Naive skip

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

+ Static cache

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

+ Spot fusion

Figure 6. The qualitative ablation study on token fusion

Reset mechanism To further ensure numerical stability, we introduce a periodic reset mechanism that forces cached tokens to refresh after an interval. This prevents the accumulation of numerical errors across timesteps. Without Reset, the model gains a slight speedup (from 1.95× to 2.25×) but suffers a clear quality drop: PSNR decreases by 1.6 dB, and DISTS increases by 0.018. This suggests that cumulative cache error gradually deviates unedited regions from their reference alignment. Periodic reset incurs negligible overhead yet effectively stabilizes the denoising trajectory,

Method CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑

Defualt 0.741 0.792 18.730 0.136 1.95 w/o Condition Cache 0.787 0.801 19.155 0.131 1.24

Table 5. Ablation on Condition Cache.

ensuring consistent structure preservation across steps.

#### 5.4. Additional results on Qwen-Image-Edit

As shown in Table 4, when SpotEdit is applied to QwenImage-Edit[39], SpotEdit preserves the original model’s background fidelity perfectly, showing only +0.01 PSNR and -0.01 DISTS difference on imgEdit, while achieving 1.59× acceleration, and on PIE-Bench, it even improves fidelity (+0.03 SSIMc, +1.08 PSNR, -0.01 DISTS) with 1.72× acceleration.

### 6. Conclusion

In this paper, we present SpotEdit, a training-free, regionaware framework that accelerates instruction-based image editing by following a simple principle: edit what needs to be edited. Unlike prior methods that denoise all image tokens uniformly, SpotEdit identifies and isolates edited regions at the token level, avoiding unnecessary computation on unedited areas. With SpotSelector, we indentify and remove non-edited tokens from processing and directly restore them from the condition image. SpotFusion further refines this with consistent feature reuse, ensuring contextual alignment. Extensive experiments show that SpotEdit delivers substantial speedups while preserving editing precision and background consistency.

## SpotEdit: Selective Region Editing in Diffusion Transformers Supplementary Material

### A. Pseudocode of SpotEdit

Algorithm S2: SpotSelector: LPIPS-like Perceptual Scoring

Input : Reconstructed latent Xˆ0 ∈ RN×C, Conditional image latent Y ∈ RN×C, VAE Decoder shallow layers L, Spatial dimensions (H,W), Patch size p

Algorithm S1: SpotEdit: Selective Region Editing with Diffusion Transformers

Input : Diffusion Transformer Φ, Editing Instruction P, Condition Image Latent Y , Initial Noise XT, Total Steps T, Initial Stage Steps Kinit, Threshold τ ,Time Schedule {ti}Ti=0 where tT = 1,t0 = 0

Output: Regenerate region and non-edited region

indices [Ati

,Rti

]

- 1 Fxˆ ← {ϕl(ˆxinput) | l ∈ L};
- 2 Fy ← {ϕl(yinput) | l ∈ L};
- 3 Initialize spatial score map M ← 0;
- 4 for each layer l ∈ L do

- 5 Dl ← ∥Norm(Fxˆ(l)) − Norm(Fy(l))∥22;
- 6 Dlaligned ← Resize(Dl,size ← (H,W));
- 7 M ← M + Dlaligned;
- 8 end
- 9 M ← M/|L|
- 10 Spooled ← AvgPool(M,kernel,stride);
- 11 Stoken ← Flatten(Spooled);
- 12 [Ati

,Rti

] ← Stoken < τ

- 13 return [Ati

Output: Edited Image Img

- 1 Initialize Condition Cache (KY ,VY );
- 2 Initialize Feature Cache (Kcache,Vcache) ← ∅;
- 3 for i ← T,T − 1,...,T − Kinit + 1 do

- 4 vt

i

,(Kcurr,Vcurr) ← Φ(Xt

i

,ti,P,Y );

- 5 Xt

i−1 ← Xt

i − (ti − (ti−1)) · vt

i

;

- 6 Xˆt

i

0 ← Xt

i − ti · vt

i

;

- 7 (Kcache,Vcache) ← (Kcurr,Vcurr);
- 8 end
- 9 for i ← T − Kinit,...,1 do

- 10 [Ati

,Rti

] ← SpotSelector(Xˆt

i

0 ,Y,τ);

- 11 for each transformer block b do

- 12 KR(b)

ti

← α(ti)KR(b)

ti+1

+ (1 − α(ti))KY(b);

- 13 VR(b)

ti

← α(ti)VR(b)

ti+1

+ (1 − α(ti))VY(b);

- 14 end
- 15 Construct Queries Qactive for tokens ∈ Ati
- 16 Qactive ← [QP,QA

ti

]

- 17 Construct Keys and Values
- 18 Kfull ← [KP,KA

ti

,KR

ti

,KY ]

- 19 Vfull ← [VP,VA

ti

,VR

ti

,VY ];

- 20 vt

i

[Ati

] ← Attention(Qactive,Kfull,Vfull); Xt

i−1

[Ati

] ← Xt[Ati

] − (ti − ti−1) · vt

i

[Ati

];

- 21 Xˆt

i

0 ← Xt

i − ti · vt

i

- 22 end
- 23 Identify final non-edited regions Rfinal
- 24 X0final[Afinal] ← X0[Afinal];
- 25 X0final[Rfinal] ← Y [Rfinal];
- 26 Img ← VAE(X0final);
- 27 return Img

,Rti

]

### B. Compatibility with Existing Acceleration Methods

- A key advantage of SpotEdit is that it is orthogonal to existing acceleration techniques for Diffusion Transformers (DiTs). While prior methods accelerate along the temporal, feature, or attention dimensions, SpotEdit accelerates computation along the spatial by skipping non-edited regions. Importantly, the acceleration dimensions targeted by these methods are inherently complementary to the spatial acceleration pursued by SpotEdit. Rather than competing with SpotEdit’s region spotting and token skipping mechanism, they operate along orthogonal axes, enabling their effects to be combined additively for further speed improvements.
- B.1. General compatibility with full-token computation accelerators

Let the set of regenerated tokens selected by SpotSelector be A and non-edited tokens be R.

SpotFusion reconstructs (K,V ) values for computation

of all regenerated tokens A: Kfull = [KP,KA,KR,KY ]Vfull = [VP,VA,VR,VY ] This reconstruction ensures that the edited region forms

a closed and condition-complete subgraph of the DiT model. Thus, any acceleration operator Oacc that assumes full attention context may be applied to the edited region alone:

Oacc(XA) ⊕ XRcache, where ⊕ denotes spatial concatenation.

Since SpotEdit does not change the functional form of the DiT computation and only restricts its spatial domain, it is fully compatible with other temporal accelerators.

#### B.2. Compatibility with TeaCache and TaylorSeer

To further demonstrate that SpotEdit is orthogonal to temporal accelerators, we integrate TeaCache and TaylorSeer into the SpotEdit pipeline. In both cases, the edited region subgraph produced by SpotSelector and reconstructed by SpotFusion forms a self-contained full-token region, ensuring that existing reuse-based accelerators operate without modification.

Formally, for any accelerator Oacc applied on the edited tokens, the composite update takes the unified form:

(X) ← Oacc XA) ⊕ XRcache,

FSpotEdit+Oacc

where the cached non-edited tokens remain valid due to SpotFusion’s full reconstruction of (K,V ).

TeaCache TeaCache performs timestep feature reuse through caching. Since SpotFusion regenerates complete attention states for edited tokens, TeaCache cached residuals remain fully compatible and can be directly reused inside the edited subgraph.

TaylorSeer TaylorSeer approximates residuals via local Taylor-series predictions. Because the edited subgraph satisfies the same continuous-time latent dynamics, the Taylor approximation computed on XA remains valid, while nonedited tokens continue using cached features.

#### B.3. Experimental results

Table S1 and Table S2 report speed and quality metrics for SpotEdit combined with TeaCache and TaylorSeer. Both integrations remain stable and improve efficiency while preserving editing quality.

These results empirically confirm TeaCache and TaylorSeer integrate seamlessly into SpotEdit.

Baselines CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑ Original Inference 0.699 0.67 16.40 0.17 1.00×

TeaCache[18] 0.698 (↓0.001) 0.60 (↓0.07) 15.02 (↓1.38) 0.21 (↑0.04) 3.43× (↑2.43) TaylorSeer[19] 0.666 (↓0.033) 0.52 (↓0.15) 14.36 (↓2.04) 0.37 (↑0.20) 3.61× (↑2.61)

SpotEdit–TeaCache 0.695 (↓0.004) 0.62 (↓0.05) 15.57 (↓0.83) 0.19 (↑0.02) 3.94× (↑2.94) SpotEdit–TaylorSeer 0.698 (↓0.001) 0.61 (↓0.06) 15.50 (↓0.90) 0.19 (↑0.02) 3.85× (↑2.85)

Table S1. Comparison of models on imgEdit-Benchmark.

Baselines CLIP↑ SSIMc↑ PSNR↑ DISTS↓ Speedup↑ Original Inference 0.741 0.791 18.76 0.136 1.00×

TeaCache[18] 0.735 (↓0.006) 0.764 (↓0.027) 18.89 (↑0.13) 0.144 (↑0.008) 3.59× (↑2.59) TaylorSeer[19] 0.741 (0.00) 0.755 (↓0.036) 17.81 (↓0.95) 0.159 (↑0.023) 3.86× (↑2.86)

SpotEdit–TeaCache 0.740 (↓0.0005) 0.797 (↑0.006) 18.98 (↑0.22) 0.133 (↓0.003) 4.28× (↑3.28) SpotEdit–TaylorSeer 0.743 (↑0.002) 0.783 (↓0.008) 18.50 (↓0.26) 0.142 (↑0.006) 4.16× (↑3.16)

Table S2. Comparison of models on PIE-Bench++.

### C. Discussion between ℓ2-distance and LPIPSlike score

We further justify the use of the LPIPS-like score in SpotSelector by analyzing the spectral bias inherent to different similarity metrics.

[Figure 49]

[Figure 50]

Source Image

L2 Distance LPIPS-like Score

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Spaceshiptoeagle

[Figure 57]

[Figure 58]

[Figure 59]

Addflowers

Figure S1. ℓ2 Distance vs. LPIPS-like Score. Low-frequency changes (e.g., brightness) produce overly large ℓ2 responses, while subtle high-frequency texture edits barely affect it. As shown in the first sample, ℓ2 incorrectly preserves the spaceship that should have been removed; in the second sample, it misclassifies background tokens as regenerate tokens, causing unnecessary regeneration and background degradation. LPIPS-like features avoid these failures by operating in a perceptually aligned feature space.

Limitations of latent ℓ2 distance. Latent representations in diffusion models are highly compressed. A pointwise ℓ2 distance is dominated by low-frequency components such as global brightness and color statistics. As noted by Zhang et al. [47], pixel-wise metrics assume independence across dimensions and thus fail to capture structural degradations like blur, which primarily remove highfrequency content but induce only mild ℓ2 deviations. In selective editing, this bias produces two failure modes: (1) low-frequency shifts, such as brightness changes, disproportionately inflate ℓ2 and falsely mark a region as ‘regenerated’, and (2) high-frequency feature changes with similar low-frequency features remain undetected, causing truly edited regions to be misclassified as ‘non-edited’.

Perceptual transferability via the VAE decoder. Although the observations of Zhang et al. were established in the pixel domain, their core principle, deep features reflect perceptual similarity better than raw vectors extends naturally to latent diffusion models. The VAE decoder provides a non-linear mapping from compressed latent space back

to perceptual image manifolds. Its intermediate activation maps recover spatial structure and high-frequency cues that are not explicitly represented in raw latents.

Building on this insight, our LPIPS-like score measures distances between decoder features, analogous to evaluating perceptual differences through VGG features in standard LPIPS. This grants two advantages:

It captures high-frequency discrepancies essential for determining whether a region was genuinely edited. And it ensures alignment between a non-edited region Rt and the condition image Y not only in coarse color tone but also in fine-grained spatial patterns.

By leveraging decoder deep features, LPIPS-like score provides a perceptually faithful descriptor for region stability and editing consistency. This resolves the spectral bias of ℓ2 and enables robust token selection in SpotSelector.

### D. More Visualization Results

As shown in Fig. S2, we provide additional qualitative results of SpotEdit across a wide range of instruction-based editing tasks, including add, action, adjust, background, remove, replace, and extract. These examples further demonstrate the versatility and robustness of our framework in handling diverse semantic manipulations while maintaining high fidelity in non-edited regions.

Source Image Result Regenerated region

Source Image Result Regenerated region

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

hatandscarfAdd

tocarRemovetheRedHouseboythemoonandcloudshandsRraisethegentlemanaman,sittingCleanbicycleAddExtract

BackgroundReplaceRemoveAddAdjustExtractAction

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

singertopottedplantBlueclothesmanRaiseEextracttheSnowarmsWomanbackgroundRemovethe

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

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Figure S2. More results of SpotEdit with various editing tasks

### References

- [1] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18187–18197, 2021. 3
- [2] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions,

2023. 3

- [3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative image transformer, 2022. 3
- [4] Pengtao Chen, Mingzhu Shen, Peng Ye, Jianjian Cao, Chongjun Tu, Christos-Savvas Bouganis, Yiren Zhao, and Tao Chen. $∆$-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers, 2024. 3
- [5] Pengtao Chen, Xianfang Zeng, Maosen Zhao, Mingzhu Shen, Peng Ye, Bangyin Xiang, Zhibo Wang, Wei Cheng, Gang Yu, and Tao Chen. Regione: Adaptive regionaware generation for efficient image editing. arXiv preprint arXiv:2510.25590, 2025.
- [6] Huanpeng Chu, Wei Wu, Guanyu Fen, and Yutao Zhang. Omnicache: A trajectory-oriented global perspective on training-free cache reuse for diffusion transformer models,

2025. 3

- [7] Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusion-based semantic image editing with mask guidance. ArXiv, abs/2210.11427, 2022. 3
- [8] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. 2, 3
- [9] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control, 2022. 2, 3
- [10] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 1
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 2
- [12] Mingzhen Huang, Jialing Cai, Shan Jia, Vishnu Suresh Lokhande, and Siwei Lyu. ParallelEdits: Efficient Multiobject Image Editing. 2, 6, 7, 8
- [13] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models, 2023. 3
- [14] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. 5
- [15] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas M¨uller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space,

2025. 1, 2, 3

- [16] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dock-

- horn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025. 1
- [17] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 3
- [18] Feng Liu, Shiwei Zhang, Xiaofeng Wang, Yujie Wei, Haonan Qiu, Yuzhong Zhao, Yingya Zhang, Qixiang Ye, and Fang Wan. Timestep embedding tells: It’s time to cache for video diffusion model, 2025. 3, 6, 7, 10
- [19] Jiacheng Liu, Chang Zou, Yuanhuiyi Lyu, Junjie Chen, and Linfeng Zhang. From reusing to forecasting: Accelerating diffusion models with taylorseers, 2025. 3, 6, 7, 10
- [20] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. 3
- [21] Ziming Liu, Yifan Yang, Chengruidong Zhang, Yiqi Zhang, Lili Qiu, Yang You, and Yuqing Yang. Region-adaptive sampling for diffusion transformers, 2025. 3
- [22] Zeqian Long, Mingzhe Zheng, Kunyu Feng, Xinhua Zhang, Hongyu Liu, Harry Yang, Linfeng Zhang, Qifeng Chen, and Yue Ma. Follow-your-shape: Shape-aware image editing via trajectory-guided region control, 2025. 2, 3, 6, 7
- [23] Zhaoyang Lyu, Xudong XU, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion models via early stop of the diffusion process, 2022. 3
- [24] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching, 2024. 3
- [25] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations, 2022. 2
- [26] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models, 2022. 3
- [27] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-toimage translation. ACM SIGGRAPH 2023 Conference Proceedings, 2023. 3
- [28] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 1, 3
- [29] Patrick P´erez, Michel Gangnet, and Andrew Blake. Poisson image editing. ACM SIGGRAPH 2003 Papers, 2003. 2
- [30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 6
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models, 2021. 1
- [32] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation,

2015. 2

- [33] Pratheba Selvaraju, Tianyu Ding, Tianyi Chen, Ilya Zharkov, and Luming Liang. Fora: Fast-forward caching in diffusion transformer acceleration, 2024. 3
- [34] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8871–8879, 2023. 3
- [35] Rami Skaik, Leonardo Rossi, Tomaso Fontanini, and Andrea Prati. Mcgm: Mask conditional text-to-image generative model, 2024. 3
- [36] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. 2025. 1, 3
- [37] Zhenxiong Tan, Qiaochu Xue, Xingyi Yang, Songhua Liu, and Xinchao Wang. Ominicontrol2: Efficient conditioning for diffusion transformers. arXiv preprint arXiv:2503.08280,

2025. 1, 3

- [38] Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing, 2025. 2, 3
- [39] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 1, 3, 8

- [40] Zongze Wu, Nicholas Kolkin, Jonathan Brandt, Richard Zhang, and Eli Shechtman. Turboedit: Instant text-based image editing. ArXiv, abs/2408.08332, 2024. 2
- [41] Zexuan Yan, Yue Ma, Chang Zou, Wenteng Chen, Qifeng Chen, and Linfeng Zhang. Eedit: Rethinking the spatial and temporal redundancy for efficient image editing, 2025. 2
- [42] Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and MingHsuan Yang. Diffusion models: A comprehensive survey of methods and applications, 2025. 2
- [43] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A unified image editing dataset and benchmark. arXiv preprint arXiv:2505.20275, 2025. 2, 6, 7, 8
- [44] Zhihang Yuan, Hanling Zhang, Lu Pu, Xuefei Ning, Linfeng Zhang, Tianchen Zhao, Shengen Yan, Guohao Dai, and Yu Wang. DiTFastattn: Attention compression for diffusion transformer models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 3
- [45] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing, 2024. 3
- [46] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2

- [47] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric, 2018. 5, 10
- [48] Tianrui Zhu, Shiyi Zhang, Jiawei Shao, and Yansong Tang. Kv-edit: Training-free image editing for precise background preservation, 2025. 2, 3
- [49] Chang Zou, Evelyn Zhang, Runlin Guo, Haohang Xu, Conghui He, Xuming Hu, and Linfeng Zhang. Accelerating diffusion transformers with dual feature caching. arXiv preprint arXiv:2412.18911, 2024. 3
- [50] Chang Zou, Evelyn Zhang, Runlin Guo, Haohang Xu, Conghui He, Xuming Hu, and Linfeng Zhang. Accelerating diffusion transformers with dual feature caching, 2024. 3

