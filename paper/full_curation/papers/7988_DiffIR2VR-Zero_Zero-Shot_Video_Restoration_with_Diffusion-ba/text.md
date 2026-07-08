# arXiv:2407.01519v5[cs.CV]31Dec2025

## DiffIR2VR-Zero: Zero-Shot Video Restoration with Diffusion-based Image Restoration Models

Chang-Han Yeh1 Hau-Shiang Shiu1 Chin-Yang Lin1 Zhixiang Wang2 Chi-Wei Hsiao3 Ting-Hsuan Chen1 Yu-Lun Liu1

1National Yang Ming Chiao Tung University 2University of Tokyo 3MediaTek Inc.

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

InputResultInputResultInputResult DenoisingSuper-resolutionDepthestimation

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

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Figure 1. Zero-shot temporal-consistent diffusion model for video restoration and beyond. Given a pre-trained diffusion model for single-image restoration, our method generates temporally consistent restored video with fine details without any further training. Our method applies to other video applications, such as depth estimation.

### Abstract

We present DiffIR2VR-Zero, a zero-shot framework that enables any pre-trained image restoration diffusion model to perform high-quality video restoration without additional training. While image diffusion models have shown remarkable restoration capabilities, their direct application to video leads to temporal inconsistencies, and existing video restoration methods require extensive retraining for different

degradation types. Our approach addresses these challenges through two key innovations: a hierarchical latent warping strategy that maintains consistency across both keyframes and local frames, and a hybrid token merging mechanism that adaptively combines optical flow and feature matching. Through extensive experiments, we demonstrate that our method not only maintains the high-quality restoration of base diffusion models but also achieves superior temporal consistency across diverse datasets and degradation

conditions, including challenging scenarios like 8× superresolution and severe noise. Importantly, our framework works with any image restoration diffusion model, providing a versatile solution for video enhancement without taskspecific training or modifications. Please see our project page at jimmycv07.github.io/DiffIR2VR web.

### 1. Introduction

Video restoration — the task of transforming low-quality videos into high-quality ones through denoising, superresolution, and deblurring — remains a significant challenge in computer vision. While diffusion models have recently revolutionized image restoration [24, 39] by generating highly realistic details that surpass traditional regressionbased methods (Fig. 2(a)), extending these advances to video has proven difficult. Current approaches that directly apply image diffusion models frame-by-frame suffer from severe temporal inconsistencies and flickering artifacts (Fig. 2(b)), particularly with Latent Diffusion Models (LDMs).

Existing solutions typically attempt to bridge this gap by fine-tuning image diffusion models with video-specific components like 3D convolution and temporal attention layers. However, these approaches face significant limitations: they require extensive computational resources (e.g., 32 A10080G GPUs for video upscaling [48]), need task-specific retraining, and often struggle to generalize across different degradation types. This creates a pressing need for a more efficient and versatile approach to video restoration.

This paper presents DiffIR2VR-Zero, the first trainingfree framework that enables zero-shot video restoration using any pre-trained image diffusion model. Unlike previous approaches that require extensive fine-tuning or model modification, our method introduces two key innovations that work in synergy: (i) Hierarchical latent warping that maintains consistency at both global and local scales by intelligently propagating latent features between keyframes and neighboring frames. (ii) Hybrid flow-guided spatial-aware token merging that combines optical flow, similarity matching, and spatial information to achieve robust feature correspondence across frames. These components work together to enforce temporal consistency in both latent and token spaces while preserving the high-quality restoration capabilities of the underlying image diffusion model (Fig. 2(c)). Our approach requires no additional training or fine-tuning, making it immediately applicable to any pre-trained image diffusion model.

As demonstrated in Fig. 1, our framework effectively handles a diverse range of restoration tasks, including denoising, super-resolution, and even depth estimation, while maintaining temporal consistency across frames. Extensive experiments show that our method not only achieves stateof-the-art performance in standard scenarios but also excels

LPIPS (↓) / InterLPIPS (↓) 0.354 / 0.012 0.245 / 0.133 0.248 / 0.095

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

|[Figure 35]|
|---|

|[Figure 36]|
|---|

| |
|---|

| |
|---|

CurrentFrameNextFrame

| |
|---|

| |
|---|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

|[Figure 43]|
|---|

|[Figure 44]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

Input frames (a) FMA-Net

(b) DiffBIR (per-frame) (c) Ours

Figure 2. 4× video super-resolution results. (a) Traditional regression-based methods such as FMA-Net [45] are limited to the training data domain and tend to produce blurry results when encountering out-of-domain inputs. (b) Although applying imagebased diffusion models such as DiffBIR [24] to individual frames can generate realistic details, these details often lack consistency across frames. (c) Our method leverages an image diffusion model to restore videos, achieving both realistic and consistent results without any additional training.

in extreme cases (such as 8× super-resolution and high-noise denoising) where traditional methods struggle.

While our work builds upon recent advances in video editing with diffusion models [9, 22], we make several novel contributions that specifically address the challenges of zeroshot video restoration:

- • The first zero-shot framework for adapting any pre-trained image restoration diffusion model to video without additional training, achieving a balance between temporal consistency and detail preservation.
- • A training-free approach that innovatively combines hierarchical latent warping with an improved token merging strategy specifically designed for restoration tasks.
- • State-of-the-art performance across various restoration tasks, demonstrating superior generalization and robustness compared to existing methods, particularly in extreme degradation scenarios.

### 2. Related Work

Video restoration. Video restoration encompasses transforming degraded videos affected by noise, blur, and low resolution into high-quality outputs [3, 4, 15, 20, 45]. Unlike single-image restoration [10], video restoration faces the additional challenge of maintaining temporal consistency across frames. Current approaches primarily rely on motionbased methods using optical flow warping [14, 28, 33] or deformable convolutions [3, 7, 38] to align features temporally. Other methods leverage attention mechanisms [2, 21, 47] to model long-range dependencies across frames, while hybrid approaches [23, 26] combine multiple techniques to handle complex degradations. However, these methods face several critical limitations: they require extensive paired training data [5, 40], assume specific degradation models [19, 21],

[Figure 47]

- Figure 3. Pipeline of our proposed zero-shot video restoration method. We process low-quality (LQ) videos in batches using a diffusion model, with a keyframe randomly sampled within each batch. (a) At the beginning of the diffusion denoising process, hierarchical latent warping provides rough shape guidance both globally, through latent warping between keyframes, and locally, by propagating these latents within the batch. (b) Throughout most of the denoising process, tokens are merged before the self-attention layer. For the downsample blocks, optical flow is used to find the correspondence between tokens, and for the upsample blocks, cosine similarity is utilized. This hybrid flow-guided, spatial-aware token merging accurately identifies correspondences between tokens by leveraging both flow and spatial information, thereby enhancing overall consistency at the token level.

and need retraining for different degradation levels [45]. These constraints significantly limit their real-world applicability and generalization capability.

Diffusion models for image restoration. Recent advances in diffusion models [8, 11, 31] have led to breakthrough improvements in image restoration. Current approaches either train models from scratch [32, 39, 46], adapt pre-trained models through guided sampling [17], or fine-tune frozen models with additional layers [37, 44], as demonstrated by StableSR and DiffBIR [24]. While these methods achieve impressive results for single images, their direct application to video leads to temporal inconsistencies due to the inherent randomness of the diffusion process. Our work uniquely bridges this gap by enabling zero-shot video restoration using these pre-trained image models without any additional training or modification.

Video consistency in diffusion models. Recent works have explored extending image diffusion models to video tasks [12, 13, 16]. Latent-space methods like Rerender-AVideo [43] use warping and interpolation but struggle with detail preservation in restoration. Token-level approaches like VidToMe [22] and TokenFlow [9] often produce oversmoothed results. Our work differs by combining hierarchical latent warping with hybrid correspondence mechanisms, specifically designed for restoration tasks. This enables zeroshot video restoration using any pre-trained image diffusion model, without requiring task-specific training

### 3. Method

Given a low-quality video with n frames y1,y2,...,yn, our goal is to restore it to high-quality x1,x2,...,xn using image-based diffusion models. While direct frame-by-frame application of these models causes temporal inconsistencies due to inherent stochasticity, particularly in extreme degradation cases (Fig. 2 and Fig. 6), our method (Fig. 3) addresses this challenge through two key innovations: Hierarchical Latent Warping (Sec. 3.2) and Hybrid Flow-guided Spatialaware Token Merging (Sec. 3.3). In this section, we first introduce the foundational concepts of diffusion models and video token merging, then detail our novel components and their integration.

#### 3.1. Diffusion Models for Video Editing

The forward process of diffusion models progressively adds noise to a clean image x0 over T steps according to:

xt = √αtxt−1+√1 − αtϵt−1 ⇒ xt = √α¯tx0+√1 − α¯tϵ, (1)

where t ∼ [1,T], ϵt,ϵ ∼ N(0,I), and α¯t = ts=1 αs. A UNet-based denoiser ϵθ learns to estimate and remove this

noise, with the inverse process gradually denoising xt to produce x0 [11, 34].

Recent video editing techniques like VidToMe [22] maintain temporal consistency by merging similar tokens within frame chunks. Given a token chunk T ∈ RB×A×C where A = w × h, tokens are separated into source tokens Tsrc and a target token Ttar. The similarity between tokens is

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Step 1: Keyframe Latent Warping

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Step 10 Step 20 Step 30 Step 40

Figure 5. Token correspondences (cosine similarity and optical flow) across denoising steps. Early on (e.g., step 10), optical flow guides better due to noisy latents. Later (e.g., step 40), similarity and flow focus on different regions, showcasing the benefit of our hybrid approach for effective token merging throughout denoising.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Key frame Key frame Key frame

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

Step2: Latent Warping

- Step 3: Token Merging with Optical Flow
- Step 4: Token Merging with Similarity

Residual Correspondence Block

[Figure 79]

[Figure 80]

batch. This hierarchical approach provides essential shape guidance at both global and local scales, as illustrated in Fig. 4 (upper part). For a given latent xˆit → 0 predicted for the ith keyframe at denoising step t, we first establish global consistency through keyframe warping:

[Figure 81]

[Figure 82]

[Figure 83]

Token Merging

[Figure 84]

[Figure 85]

Self Attention

Correspondence

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Token Unmerging

[Figure 90]

xˆit→0 ← Mji · xˆit→0 + (1 − Mji) · W(ˆxjt→0,fji), (4)

Cross Attention

[Figure 91]

where j = i − 1 represents the previous keyframe. The optical flow field fji and occlusion mask Mji are computed from low-quality frames lqj to lqi using GMFlow [41]. This formulation allows us to blend the original latent features with warped features from the previous keyframe, weighted by the occlusion mask to handle regions where warping may be unreliable.

[Figure 92]

- Figure 4. An illustration of our key modules. Without requiring any training, these modules can achieve coherence across frames by enforcing temporal stability in both latent and token space. Hierarchical latent warping provides global and local shape guidance; Hybrid spatial-aware token merging before the self-attention layer improves temporal consistency by matching similar tokens using optical flow in the down blocks and cosine similarity in the up blocks of the UNet.

Following global alignment, we propagate these warped latents to all frames within the current batch, establishing local consistency. Unlike previous approaches that rely solely on frame-to-frame warping, our hierarchical strategy ensures that corresponding points maintain similar latent representations across both temporal scales from the early stages of the denoising process. This comprehensive approach to temporal consistency proves particularly effective in handling complex motion patterns and maintaining coherent structure across the entire video sequence. To address potential warping errors in severely degraded regions, we incorporate forward-backward consistency checks and selective feature propagation. This makes our approach more robust to optical flow failures and occlusions compared to single-scale warping methods. Our experiments demonstrate that this hierarchical strategy significantly reduces temporal artifacts while preserving fine details in the restored video.

computed as:

Tsrc · Ttar ∥Tsrc∥∥Ttar∥

(s(Tsrc,t)), (2)

s(Tsrc,Ttar) =

, c = max

{t∈Ttar}

where s(·,·) represents cosine similarity and c indicates correspondences. The merging and unmerging operations are defined as:

Tmerge = M(Tsrc,Ttar, c, r), Tunmerge = U(Tmerge, c).

(3)

However, these existing techniques face significant challenges in video restoration. Early-stage denoising produces noisy latents that make traditional similarity measures unreliable, especially in UNet’s downsample blocks (Fig. 5). Additionally, focusing solely on frame-to-frame consistency misses global-local coherence, while aggressive token merging can lead to over-smoothing.

#### 3.3. Hybrid Flow-guided Spatial-aware Token Merging

While latent manipulation effectively maintains consistency in early stages, it can produce blurry results when applied during later denoising stages. To address this limitation, we introduce a hybrid flow-guided spatial-aware token merging approach that operates in the semantically rich token space, achieving both temporal consistency and detail preservation. Flow guidance and spatial-awareness. Our key insight is that different correspondence mechanisms are optimal at different stages of the denoising process. In early stages, when

#### 3.2. Hierarchical Latent Warping

Our key innovation in maintaining temporal consistency begins with a hierarchical latent warping module that operates at two distinct levels in the latent space. The first level handles global consistency across the video by warping between keyframes, while the second level ensures local consistency by propagating these warped latents within each processing

latent representations are noisy, traditional cosine similarity measures become unreliable, particularly in the UNet’s downsample blocks (Fig. 5, top). During these stages, optical flow computed from low-resolution inputs provides more reliable guidance. As denoising progresses (e.g., steps 30-40), flow-based and similarity-based methods often identify complementary correspondences (Fig. 5, bottom), motivating our hybrid approach. For downsample blocks, we employ flow-guided correspondence with a forward-backward consistency check to ensure reliability:

src→tar(X(Tsrc))+ftar→src(X(Tsrc)+fsrc→tar(X(Tsrc)))∥22),

σ = e(−∥f

(5)

where σ represents the confidence score, X(Tsrc) denotes the spatial location of source token Tsrc, and fsrc→tar, ftar→src represent forward and backward optical flows. This confidence score guides the token merging process:

Tmerge = M(Tsrc, Ttar, fsrc→tar, σ, r). (6)

For upsample blocks, we enhance cosine similarity matching with spatial awareness to prevent mismatches in uniform texture regions (e.g., sky, grass). We weight similarity scores based on spatial proximity:

s′ij = sij · e−τ , with τ = ∥X(i) − X(j)∥22 /R , (7)

where X(i), X(j) are token spatial locations and R defines the radius of influence. To prevent padding artifacts, we remove padding before merging and restore it after unmerging. Merging ratio annealing. To maintain high-quality results throughout the denoising process, we implement ratio annealing that gradually reduces the merging ratio:

i − ibeg iend − ibeg

π 2 · max min δ ·

ri = r · cos

##### ,1 ,0 ,

(8) where ibeg and iend define the annealing period, and δ controls the annealing speed. This annealing strategy helps balance temporal consistency with detail preservation, avoiding the over-smoothing common in regression-based methods while maintaining better temporal coherence than per-frame processing (Fig. 2).

#### 3.4. Scheduling

Our method orchestrates different components through the denoising process. In early stages (steps 1-10), hierarchical latent warping establishes global and local consistency by warping between keyframes and propagating features within batches. During the main denoising phase (steps 10-40), we switch to hybrid spatial-aware token merging before each attention layer. This mechanism adapts based on network depth: optical flow guides downsample blocks while similarity matching handles upsample blocks. Throughout the

process, our annealing schedule gradually reduces the merging ratio, starting aggressively for consistency and becoming more conservative to preserve details.

### 4. Experiments

We conduct extensive experiments to evaluate our zero-shot video restoration framework across different tasks, datasets, and degradation levels. We focus particularly on challenging scenarios that test both restoration quality and temporal consistency.

Datasets and evaluation protocol. For video superresolution, we evaluate on three standard benchmarks: REDS4 [27], Vid4 [25], and DAVIS [29]. We test at multiple upscaling factors (×4 and ×8) using the realistic degradation model from RealBasicVSR [5] to simulate real-world conditions. For video denoising, we use REDS30 [27] and Set8 [36], testing across a range of noise levels (std. = 50, 75, 100, 150) as well as random noise in the [50, 100] range to evaluate robustness to varying degradation severity.

Evaluation metrics. Our evaluation considers both perceptual quality and temporal consistency through complementary metrics: (1) For perceptual quality, we use LPIPS for assessing visual realism, alongside traditional PSNR and SSIM metrics. (2) For temporal consistency, we employ warping error (Ewarp), frame interpolation error, and our proposed interpolation LPIPS. The latter metric extends the interpolation error concept from [22] by using LPIPS to better capture perceptual temporal consistency, measuring how well interpolated frames match the actual frames.

Implementation details. We implement our framework using PyTorch and conduct experiments on an NVIDIA RTX 4090 GPU. To demonstrate the versatility of our approach, we apply it to two different image restoration diffusion models: DiffBIR [24] and the SD×4 upscaler [1]. For achieving 8× super-resolution with models limited to 4× upscaling, we cascade the process twice followed by bicubic downsampling. While this approach works well with DiffBIR, we note that memory constraints prevent its application with SDx4 on the REDS dataset due to the larger image sizes involved.

#### 4.1. Comparisons with State-of-the-Art Methods

We conduct comprehensive comparisons with leading methods across different video restoration tasks, examining both traditional learning-based approaches and recent diffusionbased methods.

Video super-resolution. We compare against state-of-the-art methods including BasicVSR++ [4], RVRT [23], and FMANet [45]. Additionally, we evaluate against VidToMe [22] applied to our base models and attempted comparisons with Upscale-A-Video [48], though hardware limitations (48GB A6000 GPU) prevented direct comparison due to memory constraints.

Table 1. Quantitative comparisons of video super-resolution on the DAVIS [30], Vid4 [25] and REDS4 [27] datasets. The best and second performances are marked in red and blue, respectively. Ewarp∗ denotes Ewarp(×10−3) and Einter, LPIPSinter denotes interpolation error and LPIPS. - indicates out-of-memory.

BasicVSR++ [4] RVRT [23] FMA-Net [45] VidToMe [22] SD ×4 [1] DiffBIR [24] (ECCV 2024) Metrics (CVPR 2022) (NeurIPS 2022) (CVPR 2024) (CVPR 2024) Frame Ours (Improve) Frame Ours (Improve)

PSNR ↑ 26.576 26.595 25.215 23.014 23.504 23.843 (+0.339) 23.780 24.182 (+0.402) SSIM ↑ 0.743 0.744 0.727 0.566 0.584 0.618 (+0.034) 0.601 0.621 (+0.020)

DAVIS4×

LPIPS ↓ 0.383 0.388 0.347 0.405 0.277 0.272 (-0.005) 0.264 0.262 (-0.002) Ewarp∗ ↓ 0.090 0.090 0.186 0.520 0.912 0.745 (-0.167) 0.654 0.474 (-0.180) Einter ↓ 9.115 9.135 11.558 13.676 18.125 17.431 (-0.694) 16.529 14.666 (-1.863)

LPIPSinter ↓ 0.058 0.058 0.078 0.329 0.292 0.274 (-0.018) 0.266 0.232 (-0.034)

PSNR ↑ 24.301 24.504 22.690 22.097 20.268 20.519 (+0.251) 21.964 22.331 (+0.367) SSIM ↑ 0.631 0.638 0.594 0.513 0.446 0.424 (-0.022) 0.502 0.519 (+0.017)

DAVIS8×

LPIPS ↓ 0.518 0.560 0.528 0.554 0.470 0.434 (-0.036) 0.362 0.367 (+0.005) Ewarp∗ ↓ 0.132 0.127 0.351 0.440 2.199 1.759 (-0.440) 0.964 0.699 (-0.265) Einter ↓ 9.882 9.725 13.978 12.624 24.496 21.746 (-2.750) 17.981 15.853 (-2.128)

LPIPSinter ↓ 0.088 0.081 0.132 0.388 0.457 0.442 (-0.015) 0.372 0.333 (-0.039)

PSNR ↑ 27.227 27.244 25.829 23.134 24.189 24.226 (+0.037) 24.679 25.118 (+0.439) SSIM ↑ 0.781 0.781 0.761 0.589 0.638 0.641 (+0.003) 0.657 0.683 (+0.026)

REDS44×

LPIPS ↓ 0.369 0.374 0.327 0.357 0.247 0.242 (-0.005) 0.211 0.222 (+0.011) Ewarp∗ ↓ 0.134 0.133 0.392 0.579 0.817 0.811 (-0.006) 0.704 0.499 (-0.205) Einter ↓ 15.799 15.838 19.014 17.869 22.906 22.889 (-0.017) 22.305 20.130 (-2.175)

LPIPSinter ↓ 0.106 0.101 0.133 0.356 0.295 0.281 (-0.014) 0.271 0.221 (-0.050)

PSNR ↑ 26.109 26.226 22.842 21.894 20.601 20.622 (+0.021) 22.479 22.961 (+0.482) SSIM ↑ 0.719 0.726 0.644 0.532 0.519 0.506 (-0.013) 0.559 0.590 (+0.031)

REDS48×

LPIPS ↓ 0.436 0.431 0.423 0.538 0.386 0.367 (-0.019) 0.311 0.306 (-0.005) Ewarp∗ ↓ 0.127 0.129 0.753 0.423 1.928 1.735 (-0.247) 0.828 0.551 (-0.277) Einter ↓ 15.753 15.822 21.519 15.502 26.886 25.503 (-1.383) 21.76 19.382 (-2.378)

LPIPSinter ↓ 0.099 0.099 0.159 0.412 0.370 0.388 (+0.018) 0.351 0.287 (-0.064)

PSNR ↑ 23.579 23.715 21.569 20.520 18.706 18.858 (+0.152) 20.124 20.712 (+0.588) SSIM ↑ 0.616 0.621 0.570 0.483 0.461 0.410 (-0.051) 0.461 0.509 (+0.048)

REDS416×

LPIPS ↓ 0.600 0.596 0.565 0.697 0.612 0.562 (-0.050) 0.446 0.438 (-0.008) Ewarp∗ ↓ 0.084 0.085 0.619 0.296 2.664 2.030 (-0.634) 1.168 0.665 (-0.503) Einter ↓ 14.069 14.267 18.758 12.945 28.478 24.000 (-4.478) 21.33 17.731 (-3.599)

LPIPSinter ↓ 0.088 0.088 0.139 0.417 0.559 0.493 (-0.066) 0.444 0.358 (-0.086)

PSNR ↑ 23.142 23.160 23.209 19.622 20.047 20.134 (+0.087) 20.687 21.226 (+0.539) SSIM ↑ 0.667 0.669 0.679 0.425 0.478 0.473 (-0.005) 0.497 0.525 (+0.028)

Vid44×

LPIPS ↓ 0.418 0.423 0.375 0.491 0.343 0.331 (-0.012) 0.329 0.326 (-0.003) Ewarp∗ ↓ 0.173 0.167 0.203 0.687 1.502 1.397 (-0.105) 1.156 0.677 (-0.479) Einter ↓ 3.398 3.399 4.442 11.754 17.234 16.921 (-0.313) 15.478 11.316 (-4.162)

LPIPSinter ↓ 0.015 0.015 0.026 0.337 0.275 0.271 (-0.004) 0.265 0.198 (-0.067)

PSNR ↑ 21.601 21.707 21.033 18.811 17.813 17.992 (+0.179) 18.636 19.304 (+0.668) SSIM ↑ 0.546 0.552 0.521 0.372 0.345 0.307 (-0.038) 0.367 0.406 (+0.039)

Vid48×

LPIPS ↓ 0.535 0.528 0.514 0.654 0.507 0.484 (-0.023) 0.440 0.435 (-0.005) Ewarp∗ ↓ 0.139 0.151 0.221 0.477 2.523 1.972 (-0.551) 1.524 0.767 (-0.757) Einter ↓ 3.170 3.193 5.269 9.942 22.881 19.970 (-2.911) 18.112 12.281 (-5.831)

LPIPSinter ↓ 0.011 0.011 0.032 0.393 0.423 0.419 (-0.004) 0.395 0.294 (-0.101)

Quantitative results in Tab. 1 reveal several key findings: regression-based methods like FMA-Net struggle with severe degradation and large motion, while VidToMe achieves temporal consistency but produces overly smooth results with poor visual quality. Our method uniquely maintains both the high-quality generation capabilities of the base diffusion model and strong temporal consistency.

Visual comparisons in Fig. 6 further demonstrate these differences. FMA-Net’s results show clear limitations when dealing with domain gaps between training and testing con-

ditions. While per-frame application of DiffBIR [24] and SD×4 upscaler [1] produces sharp details, the results suffer from temporal inconsistencies and jittering. Our approach successfully combines high-fidelity restoration with temporal stability. For comparisons with Upscale-A-Video on their test cases (Fig. 7), our method achieves superior detail preservation, leveraging pre-trained diffusion priors more effectively than their fine-tuning approach.

Video denoising. In denoising experiments in Tab. 2, we observe that while regression models can perform adequately

[Figure 93]

[Figure 94]

[Figure 95]

Table 2. Quantitative comparisons of video denoising of various noise levels on the REDS30 and Set8 [35] dataset. The best and second performances are marked in red and blue, respectively. Ewarp∗ denotes Ewarp(×10−3) and Einter, LPIPSinter denotes interpolation error and LPIPS.

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

(DiffBIR)GTFMA-NetInputBasicVSR++RVRT

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

VidToMe [22] Shift-Net [42] DiffBIR [24] (ECCV 2024) σ Metrics (CVPR 2024) (CVPR 2023) Frame Ours (Improve)

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

VidToMe DiffBIR

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

PSNR ↑ 22.671 21.033 24.585 24.520 (-0.065) SSIM ↑ 0.559 0.381 0.649 0.649 (+0.000)

[Figure 141]

[Figure 142]

[Figure 143]

REDS3075

[Figure 144]

(per-frame) Ours

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

LPIPS ↓ 0.397 0.735 0.276 0.275 (-0.001) Ewarp∗ ↓ 0.727 0.765 0.751 0.706 (-0.045) Einter ↓ 18.440 21.751 21.798 21.166 (-0.632)

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

LPIPSinter ↓ 0.375 0.501 0.275 0.264 (-0.011)

PSNR ↑ 22.588 22.573 24.524 24.534 (+0.010) SSIM ↑ 0.557 0.484 0.648 0.652 (+0.004)

REDS30100

LPIPS ↓ 0.404 0.518 0.275 0.271 (-0.004) Ewarp∗ ↓ 0.733 1.126 0.763 0.696 (-0.067) Einter ↓ 18.370 23.424 21.835 20.639 (-1.196)

- Figure 6. Qualitative comparisons on 4× video super-resolution. As shown in the first row, the low-quality input lacks almost all details. In the zoomed-in patches, our method produces clearer and more consistent results.

[Figure 157]

[Figure 158]

| |
|---|

[Figure 159]

| |
|---|

[Figure 160]

| |
|---|

[Figure 161]

| |
|---|

[Figure 162]

[Figure 163]

| |
|---|

[Figure 164]

| |
|---|

[Figure 165]

Input Ours Upscale-A-Video

[Figure 166]

[Figure 167]

| |
|---|

[Figure 168]

| |
|---|

[Figure 169]

| |
|---|

[Figure 170]

| |
|---|

[Figure 171]

[Figure 172]

| |
|---|

[Figure 173]

| |
|---|

[Figure 174]

Input Ours Upscale-A-Video

- Figure 7. Qualitative comparisons with Upscale-A-Video [48] on 4× video SR.

VidToMe DiffBIR

(per-frame)OursGTShift-NetInput

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

- Figure 8. Video denoising comparisons on the REDS30 [27] dataset. Our method effectively denoises and generates detailed results while maintaining temporal coherence.

LPIPSinter ↓ 0.380 0.375 0.281 0.267 (-0.014)

PSNR ↑ 22.348 21.113 24.579 24.508 (-0.071) SSIM ↑ 0.546 0.386 0.650 0.649 (-0.001)

REDS30random

LPIPS ↓ 0.429 0.728 0.276 0.270 (-0.006) Ewarp∗ ↓ 0.681 1.896 0.755 0.713 (-0.042) Einter ↓ 17.608 27.565 21.743 21.140 (-0.603)

LPIPSinter ↓ 0.384 0.542 0.282 0.272 (-0.010)

PSNR ↑ 21.531 23.433 23.197 23.713 (+0.516) SSIM ↑ 0.501 0.482 0.594 0.630 (+0.036)

Set850

LPIPS ↓ 0.415 0.574 0.261 0.245 (-0.016) Ewarp∗ ↓ 0.911 1.358 1.078 0.747 (-0.331) Einter ↓ 17.217 19.845 19.732 16.814 (-2.918)

LPIPSinter ↓ 0.406 0.432 0.332 0.255 (-0.077)

PSNR ↑ 21.226 18.198 22.519 22.955 (+0.436) SSIM ↑ 0.484 0.281 0.553 0.591 (+0.038)

Set8100

LPIPS ↓ 0.472 0.733 0.338 0.323 (-0.015) Ewarp∗ ↓ 0.918 2.229 1.13 0.802 (-0.328) Einter ↓ 17.367 24.661 20.18 17.444 (-2.736)

LPIPSinter ↓ 0.421 0.619 0.372 0.286 (-0.086)

PSNR ↑ 20.209 16.136 21.005 21.418 (0.413) SSIM ↑ 0.443 0.291 0.486 0.544 (0.058)

Set8150

LPIPS ↓ 0.554 0.729 0.449 0.402 (-0.047) Ewarp∗ ↓ 0.972 4.279 1.207 0.832 (-0.375) Einter ↓ 17.872 22.343 20.729 17.616 (-3.113)

LPIPSinter ↓ 0.470 0.646 0.450 0.331 (-0.119)

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

InputMarigoldOurs

with sufficient batch sizes, our method consistently achieves superior perceptual quality (LPIPS) and maintains this advantage even under severe degradation. As shown in Fig. 8, Shift-Net [20] struggles with out-of-distribution noise levels, and VidToMe produces temporally consistent but detaildeficient results. Per-frame DiffBIR generates high-quality frames but suffers from temporal inconsistencies, particularly noticeable in facial features and moving objects. Our method successfully balances detail preservation with temporal consistency.

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

Figure 9. Applying our techniques to consistent video depth. Integrating our proposed framework into Marigold [18] helps improve the temporal consistency of video depth estimation.

Generalization to other tasks. To demonstrate the broad applicability of our framework, we integrate it with Marigold [18], a state-of-the-art monocular depth estimator based on latent diffusion. Results in Fig. 9 show significant improvements in temporal consistency of depth estimates while maintaining accuracy. This successful adaptation to

depth estimation, alongside our results in super-resolution and denoising, highlights the versatility of our approach. As the field advances and more powerful image models emerge, our framework’s zero-shot nature allows immediate leverage of these improvements across video restoration tasks.

###### Table 3. Ablation studies for 8× VSR with different correspondence matching methods on DAVIS [29] test sets.

Down blocks

Up blocks

Spatialaware

LPIPS ↓ Ewarp∗ ↓ LPIPSinter ↓

Flow Flow – 0.518 1.214 0.563 Cos Cos – 0.390 0.736 0.350 Cos Flow – 0.507 1.049 0.545 Flow Cos – 0.375 0.677 0.347 Flow Cos ✓ 0.367 0.699 0.333

###### Table 4. Ablation studies for 8× VSR with the proposed components applied at different stages of the denoising process on DAVIS [29] test sets. We apply our two proposed components, hierarchical latent warping (HLW) and hybrid spatial-aware token merging (HS-ToMe), at the early, mid, and late denoising stages.

HLW (Sec. 3.2) HS-ToMe (Sec. 3.3) Early Mid Late Early Mid Late LPIPS ↓ Ewarp∗ ↓ LPIPSinter ↓

– – – – – – 0.362 0.964 0.372 ✓ – – ✓ – – 0.368 0.887 0.369 ✓ ✓ – ✓ ✓ ✓ 0.43 0.804 0.383 ✓ ✓ ✓ ✓ ✓ ✓ 0.411 0.704 0.339 ✓ – – ✓ ✓ ✓ 0.367 0.699 0.333

Table 5. Ablation studies on the merging ratio r. Merging ratio r PSNR ↑ SSIM ↑ LPIPS ↓

0.3 22.814 0.483 0.358 0.6 23.308 0.522 0.477 0.9 23.169 0.518 0.478 0.6 → 0 23.143 0.507 0.403 0.9 → 0 (Ours) 23.302 0.518 0.428

#### 4.2. Ablation Study

We conduct extensive ablation studies to validate our design choices and analyze the impact of different components on the final performance. These experiments not only demonstrate the effectiveness of our approach but also provide insights into the interaction between different components.

Correspondence mechanism selection. We first examine different strategies for identifying token correspondences across frames. As shown in Tab. 3, we systematically evaluate combinations of optical flow and cosine similarity at different stages of the UNet architecture. Our hybrid approach—using optical flow in downsample blocks and cosine similarity in upsample blocks—achieves optimal results across all metrics. This validates our insight that different correspondence mechanisms are more effective at different network depths. The addition of spatial awareness further improves performance by preventing mismatches in textureless regions and ensuring locally coherent correspondence matches.

Component scheduling analysis. We analyze the effectiveness of our two key components—hierarchical latent warping (HLW) and hybrid spatial-aware token merging

(HS-ToMe)—when applied at different stages of the denoising process. Tab. 4 shows that applying latent warping during mid or late denoising stages significantly degrades performance. This confirms our hypothesis that latent manipulation is most effective in early stages when establishing coarse structure, while token-level operations are crucial throughout the process for maintaining fine details and temporal consistency.

A particularly interesting finding is that attempting to enforce consistency in later stages through latent warping can actually harm the restoration quality, likely due to the increasing semantic richness of the latent space. This insight guided our design choice to transition from latent-based to tokenbased consistency enforcement as denoising progresses. Extensive temporal profile comparisons and additional ablation results are provided in the supplementary materials.

Token merging ratio r. The effectiveness of our method is significantly influenced by the token merging ratio r, which controls the balance between temporal consistency and detail preservation. We conduct extensive ablation studies on this hyperparameter, comparing our annealing strategy (reducing r from 0.9 to 0) against fixed ratios. Tab. 5 show that higher fixed ratios (0.9, 0.6) tend to improve fidelity metrics (PSNR/SSIM) by enforcing stronger temporal consistency, but at the cost of perceptual quality (higher LPIPS) due to over-smoothing. Conversely, lower fixed ratios (0.3) preserve more details but sacrifice temporal coherence. Our annealing strategy achieves the best overall performance (PSNR: 23.302, SSIM: 0.518, LPIPS: 0.428) by adaptively reducing the merging ratio throughout the denoising process, effectively balancing the trade-off between temporal consistency and detail preservation.

### 5. Conclusion

We have presented DiffIR2VR-Zero, a novel framework enabling zero-shot video restoration using pre-trained image diffusion models without additional training. Our approach combines hierarchical latent warping with hybrid flow-guided token merging to maintain temporal consistency while preserving high-quality restoration capabilities. Extensive experiments demonstrate state-of-the-art performance across various restoration tasks and remarkable robustness to severe degradations.

Limitations. Our framework faces two main limitations: flickering artifacts in dynamic scenes due to LDM decoder sensitivity, and reduced performance under extreme degradation scenarios. Future work will focus on stabilizing decoder output and enhancing degradation handling capabilities. As more powerful image diffusion models emerge, our framework’s zero-shot nature will allow immediate leverage of these advances for improved video restoration.

### References

- [1] Stable diffusion x4 upscaler, 2023. 5, 6, 12
- [2] Jiezhang Cao, Yawei Li, Kai Zhang, and Luc Van Gool. Video super-resolution transformer. arXiv preprint arXiv:2106.06847, 2021. 2
- [3] Kelvin CK Chan, Xintao Wang, Ke Yu, Chao Dong, and Chen Change Loy. Basicvsr: The search for essential components in video super-resolution and beyond. In CVPR, 2021. 2
- [4] Kelvin CK Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Basicvsr++: Improving video superresolution with enhanced propagation and alignment. In CVPR, 2022. 2, 5, 6
- [5] Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, and Chen Change Loy. Investigating tradeoffs in real-world video super-resolution. In CVPR, 2022. 2, 5
- [6] Ziyan Chen, Jingwen He, Xinqi Lin, Yu Qiao, and Chao Dong. Towards real-world video face restoration: A new benchmark. arXiv preprint arXiv:2404.19500, 2024. 11
- [7] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In ICCV, 2017. 2
- [8] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In NeurIPS, 2021. 3
- [9] Michal Geyer, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Tokenflow: Consistent diffusion features for consistent video editing. arXiv preprint arXiv:2307.10373, 2023. 2, 3
- [10] Shi Guo, Zifei Yan, Kai Zhang, Wangmeng Zuo, and Lei Zhang. Toward convolutional blind denoising of real photographs. In CVPR, 2019. 2
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3
- [12] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In NeurIPS, 2022. 3
- [13] Yaosi Hu, Zhenzhong Chen, and Chong Luo. Lamd: Latent motion diffusion for video generation. arXiv preprint arXiv:2304.11603, 2023. 3
- [14] Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Flowformer: A transformer architecture for optical flow. In ECCV, 2022. 2
- [15] Takashi Isobe, Xu Jia, Shuhang Gu, Songjiang Li, Shengjin Wang, and Qi Tian. Video super-resolution with recurrent structure-detail network, 2020. 2
- [16] Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, and Pinar Yanardag. Rave: Randomized noise shuffling for fast and consistent video editing with diffusion models. In CVPR, 2024. 3
- [17] Bahjat Kawar, Michael Elad, Stefano Ermon, and Jiaming Song. Denoising diffusion restoration models. In NeurIPS,

2022. 3

- [18] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 7, 11, 12, 14

- [19] Lingshun Kong, Jiangxin Dong, Jianjun Ge, Mingqiang Li, and Jinshan Pan. Efficient frequency domain-based transformers for high-quality image deblurring. In CVPR, 2023. 2
- [20] Dasong Li, Xiaoyu Shi, Yi Zhang, Ka Chun Cheung, Simon See, Xiaogang Wang, Hongwei Qin, and Hongsheng Li. A simple baseline for video restoration with grouped spatialtemporal shift. In CVPR, 2023. 2, 7
- [21] Wenbo Li, Xin Tao, Taian Guo, Lu Qi, Jiangbo Lu, and Jiaya Jia. Mucan: Multi-correspondence aggregation network for video super-resolution. In ECCV, 2020. 2
- [22] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. In CVPR, 2024. 2, 3, 5, 6, 7, 12
- [23] Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, and Luc V Gool. Recurrent video restoration transformer with guided deformable attention. In NeurIPS, 2022. 2, 5, 6
- [24] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior, 2024. 2, 3, 5, 6, 7, 12
- [25] Ce Liu and Deqing Sun. On bayesian adaptive video super resolution. IEEE TPAMI, 2013. 5, 6
- [26] Yu-Lun Liu, Wei-Sheng Lai, Ming-Hsuan Yang, Yung-Yu Chuang, and Jia-Bin Huang. Hybrid neural fusion for fullframe video stabilization. In ICCV, 2021. 2
- [27] Seungjun Nah, Sungyong Baik, Seokil Hong, Gyeongsik Moon, Sanghyun Son, Radu Timofte, and Kyoung Mu Lee. Ntire 2019 challenge on video deblurring and superresolution: Dataset and study. In CVPRW, 2019. 5, 6, 7
- [28] Jinshan Pan, Haoran Bai, and Jinhui Tang. Cascaded deep video deblurring using temporal sharpness prior. In CVPR,

2020. 2

- [29] F. Perazzi, J. Pont-Tuset, B. McWilliams, L. Van Gool, M. Gross, and A. Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016. 5, 8
- [30] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In CVPR, 2016. 6
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 3
- [32] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image superresolution via iterative refinement. IEEE TPAMI, 2022. 3
- [33] Xiaoyu Shi, Zhaoyang Huang, Weikang Bian, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, and Hongsheng Li. Videoflow: Exploiting temporal cues for multi-frame optical flow estimation. In ICCV, 2023. 2
- [34] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 3

- [35] Matias Tassano, Julie Delon, and Thomas Veit. Dvdnet: A fast network for deep video denoising. In ICIP, 2019. 7

- [36] Matias Tassano, Julie Delon, and Thomas Veit. Fastdvdnet: Towards real-time deep video denoising without flow estimation. In CVPR, 2020. 5
- [37] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. IJCV, 2024. 3
- [38] Xintao Wang, Kelvin CK Chan, Ke Yu, Chao Dong, and Chen Change Loy. Edvr: Video restoration with enhanced deformable convolutional networks. In CVPRW, 2019. 2
- [39] Bin Xia, Yulun Zhang, Shiyin Wang, Yitong Wang, Xinglong Wu, Yapeng Tian, Wenming Yang, and Luc Van Gool. Diffir: Efficient diffusion model for image restoration. In ICCV,

2023. 2, 3

- [40] Liangbin Xie, Xintao Wang, Shuwei Shi, Jinjin Gu, Chao Dong, and Ying Shan. Mitigating artifacts in real-world video super-resolution models. In AAAI, 2023. 2
- [41] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In CVPR, 2022. 4
- [42] Zhaoyi Yan, Xiaoming Li, Mu Li, Wangmeng Zuo, and Shiguang Shan. Shift-net: Image inpainting via deep feature rearrangement. In ECCV, 2018. 7, 12
- [43] Shuai Yang, Yifan Zhou, Ziwei Liu, and Chen Change Loy. Rerender a video: Zero-shot text-guided video-to-video translation. In SIGGRAPH Asia 2023 Conference Papers, 2023. 3
- [44] Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. Pixelaware stable diffusion for realistic image super-resolution and personalized stylization. In ECCV, 2024. 3
- [45] Geunhyuk Youk, Jihyong Oh, and Munchurl Kim. Fma-net: Flow-guided dynamic filtering and iterative feature refinement with multi-attention for joint video super-resolution and deblurring. In CVPR, 2024. 2, 3, 5, 6, 12
- [46] Zongsheng Yue, Jianyi Wang, and Chen Change Loy. Resshift: Efficient diffusion model for image super-resolution by residual shifting. In NeurIPS, 2024. 3
- [47] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In CVPR, 2022. 2
- [48] Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, and Chen Change Loy. Upscale-a-video: Temporal-consistent diffusion model for real-world video super-resolution. In CVPR, 2024. 2, 5, 7, 12

[Figure 226]

(a) Correspondences without spatial-awareness and padding removal

[Figure 227]

(b) Correspondences without spatial-awareness

[Figure 228]

(c) Correspondences with spatial-awareness and padding removal (ours)

Figure 10. Correspondences at denoising step 40 for different settings.

### A. Appendix Section

In this supplementary material, we first provide additional details on the testing datasets and evaluation metrics. Then, we report the computational complexity and inference time comparison. Subsequently, we present more visual comparisons of various methods.

#### A.1. Ablation Studies on Correspondences Identified by Cosine Similarity

Fig. 10 The figure shows the correspondences at denoising step 40 for three scenarios: without spatial awareness and padding removal, without spatial awareness, and with both spatial awareness and padding removal (ours). It is evident that padding values significantly affect the matching quality. However, even after removing padding, many mismatched diagonal lines remain, leading to blurry results. In contrast, our method effectively finds accurate correspondences by leveraging spatial information from the video.

#### A.2. Severe Degradation Scenarios

Our balanced approach proves particularly effective in severe degradation scenarios. For instance, in 8× super-resolution tasks, our method not only avoids artifacts but can even improve visual quality compared to per-frame approaches (Fig. 11). Additionally, in the 4× video face super-resolution

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

SDx4upscalerOurs

OursDiffBIR

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

- Figure 11. Applying our method on DiffBIR and SD ×4 upscaler for 8×SR task. In this case of severe degradation, our method avoids artifacts and outperforms per-frame inference in terms of visual quality.

Input frames FMA-Net DiffBIR (per-frame) Ours

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

|[Figure 245]|
|---|

|[Figure 246]|
|---|

[Figure 247]

[Figure 248]

|[Figure 249]|
|---|

[Figure 250]

|[Figure 251]|
|---|

|[Figure 252]|
|---|

[Figure 253]

[Figure 254]

|[Figure 255]|
|---|

[Figure 256]

Input frames FMA-Net DiffBIR (per-frame) Ours

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

|[Figure 261]|
|---|

|[Figure 262]|
|---|

[Figure 263]

| |
|---|
| |

[Figure 264]

|[Figure 265]|
|---|

[Figure 266]

|[Figure 267]|
|---|

|[Figure 268]|
|---|

[Figure 269]

[Figure 270]

|[Figure 271]|
|---|

[Figure 272]

- Figure 12. Additional qualitative comparisons on 4× video super-resolution. In the zoomed-in patches, our method produces clearer and more consistent results.

dataset [6], our results contain more details compared to FMA-Net and are temporally more consistent than per-frame method DiffBIR as shown in Fig. 14. This underscores the effectiveness of our ratio annealing technique in addressing the over-smoothing tendency while maintaining the benefits of our token merging approach. Additional comparisons on video super-resolution can be found at Fig. 12 and Fig. 13.

Other Video Tasks: Consistent Video Depth. Our zeroshot framework is applicable to any pre-trained image-based diffusion models and could improve the predicted video consistency. Therefore, we integrate our proposed zeroshot framework into a state-of-the-art latent diffusion-based monocular depth estimator: Marigold [18]. Fig. 15 shows that integrating our proposed framework into Marigold helps improve the temporal consistency of video depth estimation.

#### A.3. Computational Complexity

While our method focuses on zero-shot video restoration without additional training, it’s important to consider the

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Table 6. Training time and used devices for different methods.

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

(DiffBIR)GTFMA-NetInput

| |
|---|

| |
|---|

| |
|---|

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

RVRTBasicVSR++

Method Training time GPU specs Shift-Net [42] Not reported 8 NVIDIA A100-32G GPUs FMA-Net [45] Not reported Not reported Upscale-A-Video [48] Not reported 32 NVIDIA A100-80G GPUs Ours No training needed -

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

VidToMe DiffBIR

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

###### Table 7. Inference time different methods.

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

(per-frame) Ours

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

Method Inference time VidToMe [22] 1m 49s FMA-Net [45] 4.7s SDx4 [1] per-frame 41s SDx4 [1] + Ours 1m 7s DiffBIR [24] per-frame 1m 17s DiffBIR [24] + Ours 2m 20s Shift-Net [42] 12.7s Marigold [18] (4-step) + Ours 10s Upscale-a-Video [48] OOM

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

- Figure 13. Additional qualitative comparisons on 8× video super-resolution. As shown in the first row, the low-quality input lacks almost all details. In the zoomed-in patches, our method produces clearer and more consistent results.

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

InputFMA-NetDiffBIROurs

- Figure 14. Additional qualitative comparisons on 4× video face super-resolution.

###### Table 8. Quantitative comparisons of different unmerging methods on Vid4 x4 SR task.

Unmerging Method LPIPS ↓

Averaging 0.337 Replacement 0.329

inference (around 26s for SDx4 and 63s for DiffBIR) while maintaining strong temporal consistency. This is notably more efficient than training-based methods like UpscaleA-Video, which requires 32 A100 GPUs and encounters out-of-memory (OOM) issues even during inference on our test setup. Furthermore, when applied to lightweight models like Marigold with 4-step sampling, our method achieves very fast inference at just 10 seconds total.

computational requirements in comparison to other approaches. Tab. 6 provides an overview of the training time and GPU specifications for different methods, including ours.

As shown in the table, our method stands out by not requiring any training or fine-tuning, which significantly reduces the computational resources needed. This is in stark contrast to other methods that require multiple high-end GPUs and several days of training time. For inference, our method introduces some computational overhead due to the hierarchical latent warping and hybrid token merging processes. However, this overhead is relatively small compared to the resources required for training or fine-tuning video models. Specifically, our method adds only approximately 6 seconds to the inference time of the base image diffusion model per frame.

#### A.5. Additional Ablation Studies

Comparison of temporal profiles. The comparisons in Fig. 16 also indicate that our results are smoother, demonstrating better temporal stability.

Token Unmerging Strategies. We experimented with two unmerging strategies: averaging paired tokens and direct replacement with keyframe tokens. Tab. 8 shows the results of these experiments on the Vid4 x4 SR task. As shown in the table, the replacement method outperforms averaging in terms of LPIPS, indicating better perceptual quality. Our experiments consistently showed that averaging tends to produce blurrier outputs in restoration tasks. Based on these results, we adopted the replacement-based unmerging process in our final model, as it preserves more details and leads to sharper outputs.

#### A.4. Inference Time Comparison

We report the inference time for processing 10 video frames at 854×480 resolution on a single 4090 GPU in Tab. 7. Our method adds a reasonable overhead compared to per-frame

Limitations: Extreme Degradation Extreme degradation (e.g., 32× super-resolution) or overly detailed facial features may yield unsatisfactory results (Fig. 17). However, our framework’s adaptability allows the incorporation of future, more powerful image-based diffusion models. Future improvements will focus on refining keyframe selection, stabilizing decoder output across LDM architectures, and enhancing extreme degradation handling. These aim to improve practical application and mitigate flickering issues inherent in LDM decoders.

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

InputInputInputInputMarigoldMarigoldMarigoldMarigoldOursOursOursOurs

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

- Figure 15. Applying our techniques to consistent video depth. Integrating our proposed framework into Marigold [18] helps improve the temporal consistency of video depth estimation.

𝑥

𝑡

[Figure 437]

[Figure 438]

Input Video

[Figure 439]

[Figure 440]

𝑥

[Figure 441]

Flow + Flow

[Figure 442]

Cosine + Cosine

[Figure 443]

Cosine + Flow

[Figure 444]

Flow + Cosine

[Figure 445]

𝑡

Ours

- Figure 16. Comparison of temporal profile. We examine a row of pixels and track changes over time. The profiles from Flow + Flow and Cosine + Flow methods exhibit noise, indicating flickering artifacts. The Cosine + Cosine method shows smoother profiles but contains some discontinuities. Flow + Cosine demonstrates improved consistency but retains some distortions. Utilizing flow, cosine, and spatial-aware techniques, our method achieves the most seamless and consistent transitions, effectively minimizing artifacts.

[Figure 446]

[Figure 447]

[Figure 448]

(a) DDNM (b) FMA-Net (c) Ours

- Figure 17. Failure case under 32x SR. Most methods fail under this extreme degradation. However, if more powerful image-based diffusion models emerge in the future, our method can be easily adapted, offering greater potential to achieve this task.

