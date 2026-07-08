# arXiv:2511.16301v2[cs.CV]24Nov2025

### Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling

Minseok Seo1 Mark Hamilton2,3 Changick Kim1 1KAIST 2MIT 3Microsoft

minseok.seo@kaist.ac.kr

Medical Autonomous Driving

Natural

[Figure 1]

[Figure 2]

[Figure 3]

|[Figure 4]|
|---|

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

|[Figure 11]|
|---|

[Figure 12]

Ours LR

Ours LR Ours LR

| |
|---|

| |
|---|

| |
|---|

Cartoon

Manufacturing

## Upsample Anything

[Figure 13]

[Figure 14]

[Figure 15]

Ours LR

[Figure 16]

[Figure 17]

[Figure 18]

Ours LR

| |
|---|

| |
|---|

| |
|---|

A universal, training-free upsampler via lightweight test-time optimization

###### Satellite Thermal

###### Biochemical

| |
|---|

| |
|---|

| |
|---|

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Ours LR

[Figure 25]

[Figure 26]

[Figure 27]

Ours LR

Ours LR

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

Figure 1. Our method performs lightweight test-time optimization (≈0.419 s/image) without requiring any dataset-level training.It generalizes seamlessly across domains while maintaining consistent reconstruction quality for every image. (All examples are randomly selected, without cherry-picking.)

##### Abstract

We present Upsample Anything, a lightweight test-time optimization (TTO) framework that restores low-resolution features to high-resolution, pixel-wise outputs without any training. Although Vision Foundation Models demonstrate strong generalization across diverse downstream tasks, their representations are typically downsampled by 14×/16× (e.g., ViT), which limits their direct use in pixellevel applications. Existing feature upsampling approaches depend on dataset-specific retraining or heavy implicit optimization, restricting scalability and generalization. Upsample Anything addresses these issues through a simple per-image optimization that learns an anisotropic Gaussian kernel combining spatial and range cues, effectively bridging Gaussian Splatting and Joint Bilateral Upsampling.

The learned kernel acts as a universal, edge-aware operator that transfers seamlessly across architectures and modalities, enabling precise high-resolution reconstruction of features, depth, or probability maps. It runs in only ≈ 0.419s per 224×224 image and achieves state-of-the-art performance on semantic segmentation, depth estimation, and both depth and probability map upsampling. Project page: https://seominseok0429.github.io/Upsample-Anything/

##### 1. Introduction

Modern computer vision systems for pixel-level prediction tasks such as semantic, instance, and panoptic segmentation [2, 5, 15, 30] or depth estimation [11, 20, 26] often use an encoder–decoder paradigm. The encoder extracts hi-

- Step1. Training Upsampler

[Figure 31]

[Figure 32]

VFM Upsampler

[Figure 33]

[Figure 34]

[Figure 35]

- Step2. Inference (Task-specific inference only)

###### Step1. Test-time Optimization & Inference

[Figure 36]

|≈0.419 s/image|
|---|

Training Objective

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

FeatUp

[Figure 41]

LoftUP

Upsampler

𝑇𝑇𝑂

|𝐹𝐻𝑅|
|---|

Dataset

VFM Upsampler

JAFAR AnyUP

[Figure 42]

[Figure 43]

|𝐼𝐿𝑅|
|---|

|𝐼𝐻𝑅|
|---|

Upsampler

Anything

|𝐷𝐻𝑅|
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

|𝐹𝐿𝑅|
|---|

|𝐼𝐻𝑅|
|---|

|𝐹𝐻𝑅|
|---|

𝐹𝐿𝑅 𝐷𝐿𝑅 𝐿𝐿𝑅

|𝐿𝐻𝑅|
|---|

(a) Dataset-Level Training Method (b) Upsample Anything (Ours)

- Figure 2. Comparison of dataset-level training and our test-time optimization (TTO). (a) Dataset-level methods (FeatUp, LoftUp, JAFAR, AnyUp) require paired training data and handle only 2D feature maps. (b) Our Upsample Anything performs TTO using only one HR image and generalizes to feature, depth, segmentation, and even 3D features.

erarchical features that capture semantic abstraction from the input image, and the decoder reconstructs dense, taskspecific predictions such as class maps, depth, or optical flow at the original spatial resolution.

Recent advances in large-scale self-supervised learning [1, 8, 17, 18, 27, 31] introduce general-purpose encoders called Vision Foundation Models (VFMs) that can serve as universal backbones across diverse downstream tasks. This paradigm shift has led to the emergence of Vision Foundation Models (VFMs) such as DINO [17], CLIP [18], SigLIP [27], and MAE [8], which provide transferable and semantically rich features with minimal task-specific finetuning.

By decoupling the encoder from the downstream task, these foundation models dramatically reduce the data and training costs needed for adaptation while maintaining strong generalization across domains. However, despite these advantages, high-performing pixel-level systems still require large and complex decoders such as DPT [19], UPerNet [24], or SegFormer [25] to recover spatial details from low-resolution features. Foundation features are typically downsampled by a factor of 14-16 in Vision Transformer [4] architectures or equivalently through multiple pooling stages in CNN-based backbones [7, 16]. As a result, they lack fine-grained spatial information, forcing decoders to rely on heavy upsampling networks that are computationally expensive, memory-intensive, and often difficult to generalize to new architectures or resolutions.

To address this resolution gap, a growing line of research has explored feature upsampling methods [3, 6, 10, 22, 23] to restore spatial details in pretrained representations without modifying the encoder. These methods learn an upsampling operator that maps low-resolution foundation features to higher resolutions, effectively bridging the seman-

tic–spatial gap before the downstream decoder. In doing so, they can achieve strong performance across diverse pixellevel tasks even with a single 1×1 convolutional decoder.

Feature upsampling approaches can be broadly categorized into two paradigms depending on how the upsampler is optimized: (a) dataset-level training [3, 6, 10, 22, 23] and (b) test-time optimization (TTO) [6], as illustrated in Fig. 2. In the dataset-level training paradigm, the feature upsampler is trained on a target dataset either by generating pseudo-labels using methods such as SAM [13] for zeroshot supervision [10] or by adopting multi-view training objectives [3, 6, 22, 23].

While this approach can generalize to certain unseen data, it still requires dataset-level training, meaning that the upsampler must be retrained whenever the backbone architecture or target dataset changes. Moreover, due to heavy memory usage, most trained upsamplers can only operate up to 112–224 pixels in resolution. The test-time optimization paradigm, exemplified by methods such as FeatUp (Implicit), avoids dataset-level training by optimizing the feature upsampler directly at inference time for each test image. Although this removes the need for offline training, the per-image optimization is computationally expensive, taking an average of 49 seconds to converge for a 224-sized image.

We propose Upsample Anything, a test-time optimization (TTO) framework for feature upsampling, as illustrated in Fig. 2-(b). Unlike previous methods requiring dataset-level training, it performs lightweight per-image optimization and processes a 224-sized image in only ≈ 0.419 s. Given an input image, Upsample Anything resizes the RGB guidance to match the low-resolution (LR) featuremap size, reconstructs the high-resolution (HR) color image through optimization, and learns pixelwise anisotropic

Gaussian parameters—(σx,σy,θ,σr)—that define a continuous spatial–range splatting kernel. These optimized kernels are then applied to the LR feature maps from a foundation encoder to produce HR feature maps aligned with the original image grid. Although the optimization is guided only by color reconstruction, the learned kernels implicitly capture geometry and semantics. As a result, Upsample Anything not only enhances 2D feature resolution but also generalizes to other pixel- or voxel-level signals (e.g., depth, segmentation, or even 3D representations) without retraining. This property highlights its potential as a unified, lightweight, and resolution-free upsampling operator across 2D and 3D domains. Despite requiring no dataset-level training, it consistently achieves state-of-the-art or nearSOTA performance on multiple pixel-level benchmarks, including semantic segmentation and depth estimation.

tively restores spatial detail from foundation features, it either requires dataset-level training or incurs long per-image optimization.

Several follow-up studies further improved spatial fidelity through learnable upsamplers. LiFT [22] employed a lightweight U-Net-like upsampler with reconstruction loss, LoftUp [10] integrated RGB coordinates via cross-attention with pseudo-groundtruth supervision, and JAFAR [3] introduced joint-attention filtering for semantic–structural alignment. AnyUp [23] proposed resolution-conditioned kernels for scalable upsampling. Despite their strong performance, these methods rely on dataset-level training, which makes them less adaptive to unseen domains. They often generalize reasonably well but still exhibit suboptimal performance when facing novel architectures, resolutions, or out-of-distribution data.

##### 2. Related Works

###### 2.1. Joint Bilateral Upsampling

Joint Bilateral Upsampling (JBU), first introduced by [14], is a classic non-learning, edge-preserving upsampling technique designed to transfer structural details from a highresolution guidance image to a low-resolution signal such as a depth map or label map. Formally, JBU computes each

high-resolution output pixel Fhrˆ [p] as a weighted average of nearby low-resolution pixels Flr[q], as follows:

∥p − q∥2 2σs2

∥I[p] − I[q]∥2 2σr2

1 Zp q∈Ω(p)

Fˆhr[p] =

Flr[q] exp −

exp −

(1)

where I denotes the high-resolution guidance image, and σs, σr control the spatial and range sensitivity, respectively. Here, Zp is a normalization factor ensuring that all weights sum to one, and Ω(p) denotes the spatial neighborhood around pixel p in the low-resolution domain. By coupling spatial proximity and color similarity, JBU preserves edges and fine details while interpolating missing information, enabling high-quality restoration of dense signals without any additional learning. This formulation has inspired numerous modern variants and learnable extensions that generalize bilateral filtering [6] to feature space and neural representations. Although JBU performs upsampling without any training, the resulting quality remains limited due to its fixed, hand-crafted kernel design.

###### 2.2. Feature Upsampling

A pioneering work in feature upsampling is FeatUp [6], which proposed two model-agnostic modules: FeatUp (JBU) and FeatUp (Implicit). The former generalizes Joint Bilateral Upsampling (JBU) [14] to high-dimensional feature space by replacing the fixed Gaussian range kernel with a learnable MLP, while the latter parameterizes high-resolution features as an implicit function Fhr = MLP(x,I(x)) optimized per image. While FeatUp effec-

,

##### 3. Preliminaries

- 2D Gaussian Splatting (2DGS). Recent works extend
- 3D Gaussian Splatting (3DGS) [12] from volumetric radiance fields to 2D image representations [28, 29, 32]. In the image–plane setting, a pixel or small region is represented by a Gaussian kernel

Gi(x) = exp − 12(x − µi)⊤Σ−i 1(x − µi) , (2)

where µi ∈ R2 is the center, Σi ∈ R2×2 is a positive–definite covariance (encoding scale and orientation), and αi is a per–kernel weight. The rendered image (or feature map) is obtained by normalized alpha blending:

I(x) =

i

αiGi(x) j αjGj(x)

, (3)

wi,ci, wi =

with ci the color or feature associated with kernel i. Because all kernels lie on a single 2D plane, rendering reduces to a normalized weighted summation without depth sorting. This process is often described as rasterization/alpha blending and can also be interpreted as a spatially varying anisotropic convolution. This property enables realtime, pose-free optimization directly in 2D image/feature domains.

Relation to Joint Bilateral Upsampling (JBU). JBU [14] computes each high-resolution (HR) output Fhr(p) as a normalized weighted average of low-resolution (LR) samples Flr(q), where the weights depend on both spatial proximity and guidance-image similarity (see Eq. 1). Viewed geometrically, each LR sample q can be regarded as a Gaussian centered at µq=q with an isotropic covariance Σq=σs2I. Under this view, JBU corresponds to a discrete and isotropic instance of a guidance-modulated 2D Gaussian Splatting (2DGS) process, where the range

[Figure 48]

|(𝟑 × 𝐇 × 𝐖)|
|---|

| | | | | | | | |(𝟑|×<br><br>𝐇|×|𝐖<br><br>)| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |𝐬| |𝐬| |
| | | |[Figure 49]<br><br>| | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |
| | | | | | | | | | | | | |

|(𝟑 × 𝐇 × 𝐖)|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

(𝟑 × 𝐇 × 𝐖)

UpsampleAnything

𝜎𝑥, 𝜎𝑦, 𝜃, 𝜎𝑟

𝜇𝑞

Resize

𝜇𝑞−1

𝐼෢𝐻𝑅 𝑝 = ෍

𝑤 𝑝,𝑞 𝐼𝐿𝑅(𝑞)

|𝐼መ𝐻𝑅 𝑝 − 𝐼𝐻𝑅(𝑝)|1

෍

𝑞∈Ω 𝑝

𝑝

###### TTO

|𝐼𝐻𝑅| |
|---|---|
| | |

|𝐼𝐿𝑅|
|---|

𝐼መ𝐻𝑅

𝐼𝐻𝑅

[Figure 53]

[Figure 54]

[Figure 55]

|(𝐂 × 𝐇 × 𝐖)|
|---|

|(𝐂 ×<br><br>𝐇 𝐬<br><br>×<br><br>𝐖 𝐬<br><br>)|
|---|

UpsampleAnything

[Figure 56]

|σ𝑞 = 𝑅(𝜃𝑞)<br><br>𝜎𝑥2(𝑞) 0 0 𝜎𝑦2(𝑞)<br><br>𝑅(𝜃𝑞)T|
|---|

Foundation

|𝐹෠𝐻𝑅 𝑝 =<br><br>σ𝑞∈Ω(𝑝) 𝐹𝐿𝑅 𝑞 exp(−<br><br>1<br>2<br><br><br>𝑥′ 𝑦′ σ𝑞′−1<br><br>𝑥′ 𝑦′<br><br>−<br><br>𝐼𝐻𝑅 𝑝 − 𝐼𝐿𝑅 𝑞 22 2𝜎𝑟2(𝑞)<br><br>)<br><br>σ𝑞′∈Ω(𝑝) exp(−<br><br>1<br>2<br><br><br>𝑥′ 𝑦′ σ𝑞′−1<br><br>𝑥′ 𝑦′<br><br>−<br><br>𝐼𝐻𝑅 𝑝 − 𝐼𝐿𝑅 𝑞′ 22 2𝜎𝑟2(𝑞′)<br><br>)|
|---|

Model

|𝐹෠𝐻𝑅|
|---|

|𝐹𝐿𝑅|
|---|

Rendering

- Figure 3. Overview of Upsample Anything. Given a high-resolution image Ihr, we downsample it to Ilr and optimize GSJBU to reconstruct Ihr, learning per-pixel anisotropic kernels {σx, σy, θ, σr} via test-time optimization (TTO). The learned kernels are then applied to foundation features Flr for rendering the high-resolution features Fhr, achieving pixel-wise anisotropic joint bilateral upsampling.

term is provided by the HR guidance image. In contrast, Upsample Anything assigns per-pixel anisotropic covariances Σp and range scales σr,p through test-time optimization, enabling adaptive fusion across space and range. This pixel-level Gaussian parameterization captures locally varying orientations and scales, making Upsample Anything a continuous, edge-preserving extension of JBU within the 2DGS framework. For the exact derivation and a more formal discussion of how this formulation differs from a simple combination, please refer to Appendix §12.

- 4. Methods

###### 4.2. Algorithm Design

Our design is inspired by classical Joint Bilateral Upsampling (JBU) [14]. The key merit of JBU is transferability: it does not hallucinate new values but instead learns mixing weights that decide how much to blend neighboring samples, which makes it naturally model-/task-agnostic. However, standard JBU in Eq. (1) is limited by global (σs,σr) and isotropic range/spatial kernels, reducing expressivity near complex structures.

Per-pixel anisotropic kernels. To overcome these limits, Upsample Anything assigns a per-LR-pixel anisotropic Gaussian with parameters {σx(q), σy(q), θ(q), σr(q)} for each low-resolution location q. Let the spatial covariance be

###### 4.1. Overview

As illustrated in Fig. 3, our method, Upsample Anything, consists of two stages: (i) test-time optimization (TTO) and (ii) feature rendering. In the TTO stage, Upsample Anything learns per-pixel anisotropic Gaussian parameters {σx,σy,θ,σr} by reconstructing the high-resolution image Ihr from its patch-wise downsampled version Ilr. This process enables each pixel to learn how spatially and photometrically similar neighbors should be blended-effectively discovering local mixing weights that generalize beyond the image domain. Once optimized, these Gaussian kernels are directly transferred to the foundation feature space, where the low-resolution feature map Flr ∈RC×H/s×W/s is splatted to produce the high-resolution feature Fhr ∈RC×H×W using the same learned anisotropic weighting mechanism. Because the splatting weights depend only on spatial–range similarity, this transfer is naturally domain-agnostic, allowing the learned kernels to act as universal upsampling operators. Excluding the feature extraction time of the Vision Foundation Model (VFM), the entire optimization and inference for a 224×224 image takes ≈ 0.419s.

σx2(q) 0 0 σy2(q)

cos θq − sin θq sin θq cos θq

R⊤(θq); R(θq) =

Σq = R(θq)

(4)

For an HR coordinate p, the unnormalized spatial weight

and range (guidance) weight are log wps←q = −21 (p − µq)⊤Σ−q 1(p − µq), (5) log wpr←q = −

∥I(p) − I(q)∥2 2σr2(q)

. (6)

where µq is the LR center projected to HR coordinates and I(·) is the HR guidance image. The final normalized mixing weight is

exp log wps←q + log wpr←q

. (7)

wp←q =

exp log wps←q′ + log wpr←q′

q′∈Ω(p)

Feature rendering (no value synthesis). Given lowresolution features Flr ∈ RC×H

l×Wl and scale s, we render

.

the HR feature Fhr ∈ RC×(sH

l)×(sWl) by pure mixing: Fhr(p) =

wp←q Flr(q).

q∈Ω(p)

This strictly reweights existing LR features (no content generation), hence transfers across backbones and tasks.

Why it generalizes. Unlike feed-forward upsamplers that require dataset-level training [3, 6, 10, 22, 23], Upsample Anything learns only per-image, pixel-wise mixing weights from the HR guidance through a test-time optimization process, and reuses these weights to splat Flr into Fhr. Because the mechanism is based on edge- and range-aware interpolation rather than value synthesis, Upsample Anything is inherently resolution-free, model-agnostic, and robust to unseen domains.

- 4.3. Test-Time Optimization

After defining the Upsample Anything formulation in Sec. 4.2, the next step is to optimize its per-pixel parameters {σx,σy,θ,σr}. Our key idea is inspired by the patchified processing of modern Vision Foundation Models (VFMs): since VFMs downsample images by a fixed stride to extract low-resolution features, we emulate this process during optimization.

Specifically, the high-resolution image Ihr is downsampled to Ilr by bilinear interpolation with a stride s, and the GSJBU parameters are optimized under a reconstruction objective from Ilr back to Ihr:

LTTO = GSJBU(Ilr) − Ihr 1.

This test-time optimization finds image-specific, pixel-wise kernels that best reconstruct the guidance signal.

After the TTO process, the learned kernels are reused to render the high-resolution feature map:

Fhr = GSJBU(Flr; σˆx,σˆy,θ,ˆ σˆr),

where the optimized parameters {σˆx,σˆy,θ,ˆ σˆr} are directly transferred to the foundation feature space to upsample Flr into Fhr.

- 5. Experiments

- 5.1. Experimental Setting

Following prior feature-upscaling works [3, 6, 10, 22, 23], we evaluate Upsample Anything on semantic segmentation (COCO, PASCAL-VOC, ADE20K) and depth estimation (NYUv2). For segmentation, we use a single 1 × 1 convolution head (linear probe), identical to prior settings. For depth, we adopt a DPT-style decoder head, consistent with [10, 23]. To compare backbones, we consider DINOv1, DINOv2, DINOv3, CLIP, and ConvNeXt, covering

|Method|COCO mIoU (↑) Acc. (↑)<br><br>|PASCAL-VOC mIoU (↑) Acc. (↑)<br><br>|ADE20k mIoU (↑) Acc. (↑)|
|---|---|---|---|
|Bilinear FeatUp LoftUp JAFAR AnyUp Upsample Anything<br><br>|60.43 80.18<br><br>60.96 80.65<br>61.08 80.72<br><br><br>60.87 80.51<br>61.25 80.89<br><br><br>61.41 81.34<br><br><br>|81.27 95.96<br><br>81.91 96.27<br><br>81.84 96.33<br><br>82.05 96.21<br><br><br>82.18 96.39<br><br><br>82.22 96.90<br><br><br>|41.48 74.95 41.92 75.41<br><br>41.83 75.36<br><br>41.74 75.22<br><br>42.02 75.63<br><br><br>42.95 76.52<br><br><br>|
|Upsample Anything (prob.)|63.40 83.73<br><br>|84.57 97.42<br><br>|44.29 78.58<br><br>|

Table 1. Comparison of different upsampling methods on COCO, PASCAL-VOC, and ADE20k datasets.

both transformer and convolutional families; unless otherwise specified, the default backbone is DINOv2-S.

Unlike prior approaches restricted to feature maps, Upsample Anything applies to general bilateral upsampling. Accordingly, we further evaluate (i) depth-map upsampling on NYUv2 and Middlebury, and (ii) probability-map upsampling on Cityscapes—each guided by the corresponding high-resolution RGB image.

###### 5.2. Implementation Details

Our Upsample Anything is implemented purely in PyTorch without any chunked or patch-wise processing. All computations are performed in a fully parallel manner over the entire high-resolution grid. Gaussian parameters are initialized as σx = σy = 16.0, σr = 0.12, and θ = 0, and are optimized per-pixel using the Adam optimizer with a learning rate of 1 × 10−3. The model performs test-time optimization for only 50 iterations in total, without any batching or data augmentation. Please refer to the supplementary material for additional implementation details, hyperparameter choices, and ablations.

###### 5.3. quantitative results

Semantic Segmentation. For a fair comparison, we adopt the conventional linear-probe protocol in which prior work fine-tunes only a 1×1 convolutional head for 10 epochs. However, we found that this shallow schedule often undertrains the head. We therefore extend training to 100 epochs and apply a cosine learning-rate schedule to gradually decay the head’s learning rate. Under this setting, our results in Table 1 show a trend that differs from previous reports [3, 6, 10, 22, 23]: although all methods converge quickly, their eventual gains over simple bilinear upsampling are modest when the backbone representation is strong. This raises the question of how much feature upsampling helps semantic segmentation under high-capacity backbones. Nevertheless, our proposed Upsample Anything attains the best accuracy across COCO, PASCALVOC, and ADE20K, with AnyUp consistently second.

Upsample Anything (prob.). In addition to upsampling feature maps, we evaluate a low-compute variant that predicts the segmentation at the feature resolution (no feature upsampling), produces a probabilistic map, and then up-

Depth Estimation Surface Normal Estimation RMSE (↓) δ1 (↑) Mean (↓) Median (↓) <11.25◦ (↑) <22.5◦ (↑) <30◦ (↑)

Method

Bilinear 0.545 0.804 23.8 19.5 33.0 70.0 81.0 FeatUp 0.523 0.810 22.7 18.6 35.5 72.3 83.0 LoftUp 0.796 0.789 28.9 24.1 25.0 58.0 71.0 JAFAR 0.521 0.807 23.2 19.0 34.0 70.8 82.0 AnyUp 0.513 0.817 22.2 18.1 36.8 73.5 84.1 Upsample Anything (Ours) 0.498 0.829 21.5 17.4 38.1 75.2 85.8

Table 2. Comparison of depth and surface normal estimation on the NYUv2 dataset.

samples probabilities to the original image size using our method. Because the logits/probabilities live on a much smaller spatial grid, this pipeline achieves the lowest computational cost yet delivers the highest accuracy in Table 1. This suggests a promising paradigm for segmentation: upsample task probabilities rather than intermediate features.

Depth Estimation We evaluate our method on the NYUv2 dataset using a frozen DINOv2 backbone. Following prior works (AnyUp, LoftUp), we adopt a lightweight DPT-style decoder head for dense prediction (details in Appendix). Unlike the original DPT, our Upsample Anything removes the internal interpolation layers, as the feature maps are already upsampled to high resolution. As shown in Table 2, Upsample Anything achieves the best performance on both depth and surface normal estimation (RMSE 0.498, δ1 0.829, mean 21.5°), indicating that precise feature upsampling is particularly beneficial for geometry-oriented tasks, while LoftUp suffers from domain gaps and fails to generalize. It appears that feature upsampling plays a more critical role in depth and surface normal estimation than in semantic segmentation.

Depth Map Upsampling Unlike feature upsampling tasks, our Upsample Anything can also be applied to other modalities such as raw depth maps. In Table 3, we evaluate Upsample Anything by downsampling high-resolution depth maps to 32×32 and restoring them to 512×512 resolution. This setup shares the same bilateral upsampling pipeline as our feature experiments, except that the lowresolution input is a depth map itself. We compare against the state-of-the-art guided interpolation method (GLU) [21] and the bilinear baseline. As shown in Figure 4, Upsample Anything achieves the best performance on the Middlebury dataset, producing sharper and more consistent structures. In contrast, on the NYUv2 dataset, the bilinear method yields a slightly lower RMSE (0.159 vs. 0.237), likely because the ground-truth depth maps are blurred and contain smoother structures. Nevertheless, the qualitative results suggest that Upsample Anything preserves geometry more effectively, especially for high-frequency and edgedominant regions.

NYUv2 (Depth Up.) Middlebury (Depth Up.)

Method

RMSE (↓) δ1 (↑) RMSE (↓) δ1 (↑) Bilinear 0.167 0.983 0.231 0.962 GLU (Guided Linear Upsample) 0.372 0.841 0.491 0.825 Upsample Anything (Ours) 0.214 0.976 0.209 0.967

Table 3. Comparison on NYUv2 and Middlebury depth upsampling.

[Figure 57]

Ours

Ours

Figure 4. Depth upsampling results on Middlebury (top) and NYUv2 (bottom). 32×32 low-resolution depth maps were upsampled to high resolution using different methods. While Upsample Anything produces sharper and more detailed edges, it still achieves lower RMSE (0.237) than bilinear (0.159) on lowresolution maps. However, in high-resolution depth prediction, Upsample Anything outperforms both qualitatively and quantitatively.

###### 5.4. Qualitative results

Across Different Resolutions. Figure 5 compares AnyUp (previous SOTA) and our Upsample Anything across different input resolutions. As shown, AnyUp performs reasonably well at higher resolutions (e.g., 32×32 and 16×16) but tends to produce over-smoothed regions, as highlighted by the red boxes. In contrast, Upsample Anything maintains sharp boundaries and fine structures even at extremely low resolutions (e.g., 7×7 and 4×4), demonstrating stronger robustness to spatial degradation.

Across Different Backbones. We compare the visual quality of upsampled features produced by AnyUP and our Upsample Anything across various backbone architectures. Given the 224×224 input image, the spatial resolutions of the extracted feature maps differ by model: 7×7 for ConvNeXt, 14×14 for CLIP and DINOv1, and 16×16 for DINOv2 and DINOv3. In this example, IˆHR denotes the reconstructed high-resolution image obtained from a 7×7 lowresolution ILR using Upsample Anything. As shown in Fig. 6, Upsample Anything consistently produces sharper boundaries, finer local structures, and more coherent feature clustering than AnyUP across all backbones. We attribute this advantage to Upsample Anything’s test-time optimization, which adaptively fits Gaussian parameters to each in-

| |
|---|

| |
|---|

| |
|---|

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

| |
|---|

| |
|---|

|𝐼𝐻𝑅(512×512)|
|---|

|𝐹𝐿𝑅(32×32)|
|---|

|𝐹𝐿𝑅(16× 16)|
|---|

AnyUP Ours

AnyUP Ours

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

|𝐹𝐿𝑅(7 × 7)|
|---|

|𝐹𝐿𝑅(4× 4)|
|---|

AnyUP Ours

AnyUP Ours

- Figure 5. Comparison across different resolutions. Qualitative results of AnyUp (previous SOTA) and our Upsample Anything on varying input resolutions.

put image, yielding features that align more precisely with image-level semantics.

Feature Similarity Analysis. Fig. 7 visualizes the feature similarity between two different images that share the same object category. This setting is often adopted in fewshot segmentation tasks to assess the consistency of feature representations. As shown in the figure, both AnyUp and Upsample Anything produce visually plausible feature upsampling results; however, when measuring cosine similarity, AnyUp tends to yield uniformly high similarity across the entire image, indicating a lack of spatial discrimination. In contrast, our Upsample Anything produces sharper and more localized similarity maps, where object boundaries are clearly preserved and distinct regions are well separated. We believe that such discriminative feature behavior suggests the potential of our method for downstream tasks such as few-shot segmentation and category-level feature matching.

###### 5.5. Ablation Study

Resolution–Efficiency Tradeoff We analyze the computational efficiency of our Upsample Anything compared with AnyUp [23] across multiple output resolutions, as summarized in Table 4. AnyUp employs a feature-agnostic layer and local window attention that allow flexible inference across arbitrary encoders and resolutions. While this design yields fast performance at lower resolutions (<256 × 256), the window-based attention and dense similarity computation introduce quadratic growth in both memory and time complexity as the spatial size increases. Consequently, AnyUp suffers from significant GPU memory overhead and fails with out-of-memory (OOM) errors beyond 512 × 512, exposing a scalability bottleneck in its dense attention formulation.

In contrast, our Upsample Anything performs fully parallel anisotropic Gaussian splatting without building dense

Resolution (H×W) AnyUp (trained) / AnyUp (trained) /

Ours (TTA) Time (s) Ours (TTA) Peak Mem (MB)

64×64 0.0025 / 0.0055 53.8 / 358.5 128×128 0.0034 / 0.0150 121.9 / 1320.5 224×224 0.0137 / 0.0419 531.0 / 3969.7 448×448 0.0583 / 0.2398 6184.2 / 15774.8 512×512 0.0893 / 0.3211 10283.8 / 20590.4 896×896 0.5875 / 1.2789 91250.9 / 62985.2

1024×1024 OOM / 1.8083 — / 82255.5 2048×2048 OOM / OOM — / —

Table 4. omparison of inference time and GPU memory usage between AnyUp and Upsample Anything under varying input resolutions. Both methods operate without training, but Upsample Anything performs per-image test-time optimization (TTO), which results in slightly higher computational cost but significantly improved generalization.

pairwise affinity maps, resulting in linear memory growth with respect to the output size. As shown in Table 4, Upsample Anything maintains stable runtime and controlled memory even at 1024 × 1024 resolution—where AnyUp cannot execute—demonstrating its robustness for largescale feature upsampling.

Why Upsample Anything: Design Motivation and Comparative Analysis In our architecture search, we aimed to achieve accurate upsampling within one second while maintaining generalization across domains. Tab. 5 compares different upsampling strategies on PASCAL-VOC and NYUv2 benchmarks. Guided Linear Upsampling (GLU) [21] represents a strong bilateral-filter-based baseline but exhibits unstable optimization due to its unconstrained formulation. We also implemented a 2D Gaussian Splatting (2DGS) [9] baseline by directly optimizing Gaussian kernels for low-tohigh feature interpolation without hierarchical fitting. Although 2DGS can model local structures continuously, its dense Gaussian representation and lack of spatial–range constraints make it computationally heavy and prone to

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

ConvNeXt

###### DINOv1Image

CLIP

|𝐼𝐻𝑅(224,224)|
|---|

|𝐼𝐻𝑅(7,7)|
|---|

|𝐼መ𝐻𝑅(7,7)|
|---|

|𝐹𝐿𝑅(7,7)|
|---|

|𝐹𝐿𝑅(14,14)|
|---|

AnyUP Ours

AnyUP Ours

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

DINOv2

DINOv3

|𝐹𝐿𝑅(14,14)|
|---|

|𝐹𝐿𝑅(14,14)|
|---|

|𝐹𝐿𝑅(16,16)|
|---|

AnyUP Ours

AnyUP Ours

AnyUP Ours

- Figure 6. Visual comparison across different backbones. Given the same 224×224 input, feature maps have varying spatial resolutions (7×7 for ConvNeXt, 14×14 for CLIP and DINOv1, 16×16 for DINOv2/v3). Upsample Anything produces sharper edges, richer textures, and more distinct feature clustering than AnyUP across all backbones, demonstrating its strong adaptability through test-time optimization.

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

Ref Image Target Image LR (CosSim)

AnyUp (CosSim)

Ours (CosSim)

Ref Mask Ground Truth LR (Feature) AnyUp (Feature)

Ours (Feature) Ours (Mask)

[Figure 99]

[Figure 100]

AnyUp (Mask)

- Figure 7. Visualization of feature similarity between a reference and a target image. The feature vector is obtained by averaging the reference features within the reference mask, and cosine similarity is then computed against all feature locations in the target image.

PASCAL-VOC (Segmentation)

Iteration PSNR (↑)

Time (s) mIoU (↑) Acc. (↑)

50 35.33 82.22 96.90 0.041 300 35.59 82.10 96.84 3.397 500 35.60 82.15 96.88 6.161 1000 35.60 82.10 96.82 12.294 5000 35.60 82.17 96.80 61.458

Table 6. Ablation on the number of optimization iterations.

mance on PASCAL-VOC. As shown, PSNR quickly converges to 35.60 after about 500 iterations, indicating that Upsample Anything reaches its optimum very early. Interestingly, the best segmentation accuracy is achieved at only 50 iterations, which also provides the fastest inference time (0.419s). Based on this observation, we adopt 50 iterations as the default setting throughout all experiments.

PASCAL-VOC (Segmentation) NYUv2 (Depth Estimation)

Method Time (s)

mIoU (↑) Acc. (↑) RMSE (↓) δ1 (↑) Bilinear 0.00009 81.27 95.96 0.545 0.804 Guided Linear Upsample 0.00303 80.12 94.73 0.598 0.773 JBU 0.00600 81.65 96.21 0.531 0.812 LIG 481.526 78.54 93.02 0.642 0.741 GSJBU (Ours) 0.4197 82.22 96.90 0.498 0.829

##### 6. Conclusion

Table 5. Comparison of inference time, segmentation, and depth estimation performance across different upsampling methods. GSJBU achieves the best balance between accuracy and scalability.

We introduced Upsample Anything, a unified framework that connects Joint Bilateral Upsampling (JBU) and Gaussian Splatting (GS) under a continuous formulation. It performs lightweight test-time optimization without pretraining or architectural constraints, achieving efficient and robust upsampling across diverse resolutions and domains. It optimizes a 224×224 image in 0.419 seconds while producing significant gains in both feature and depth upsampling. Extensive experiments show that Upsample Anything achieves state-of-the-art performance without any learnable module, serving as a universal, plug-and-play framework that combines the simplicity of JBU with the expressive power of Gaussian representation. Limitation. Despite its generality, Upsample Anything may face challenges under severe occlusions or low-SNR guidance, where optimization becomes unstable. Future work will focus on enhancing the robustness and adaptability of the framework across diverse domains, aiming to make it more resilient under challenging conditions.

over-smoothing during test-time optimization. In contrast, Upsample Anything inherits the spatial–range constraint of classical JBU while leveraging the continuous Gaussian formulation for differentiable optimization. As shown in Tab. 5 Upsample Anything achieves the best balance between speed, convergence, and accuracy, demonstrating that the JBU constraint synergizes effectively with Gaussian optimization for fast and stable test-time refinement.

Impact of Test-Time Optimization Steps We conducted experiments to investigate the trade-off between the number of TTO iterations, inference time, and performance. Table 6 summarizes the results. We measured PSNR between Ilr and Iˆhr, along with downstream segmentation perfor-

### Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling

#### Supplementary Material

##### 7. Semantic Segmentation on Cityscapes

We evaluated feature upsampling and probability-map upsampling on Cityscapes using the official LoftUp segmentation codebase with a stronger training setup that includes 448×448 input resolution, 100 epochs, and a learning-rate scheduler. Under this configuration, all methods, including LoftUp and ours, produced almost the same mIoU as bilinear interpolation, which differs from the improvements reported in the LoftUp paper.

To ensure correctness, we carefully re-examined our implementation through an automated code audit with ChatGPT and a manual review by multiple authors. We found no inconsistencies or bugs.

The quantitative results are summarized in Table 7. Across all methods, including feature-level and probabilitylevel upsampling, the differences remain within a very narrow range. Cityscapes primarily contains large and regular structures, and its annotations are relatively coarse. With a sufficiently trained segmentation head, bilinear interpolation already performs near optimally, leaving little room for additional gains. In contrast, datasets such as COCO, PASCAL-VOC, and ADE20K include many small objects and complex boundaries, where upsampling delivers clear benefits.

|Method|Cityscapes mIoU (↑) Acc. (↑)<br><br>|
|---|---|
|Bilinear FeatUp LoftUp JAFAR AnyUp Upsample Anything<br><br>|57.90 90.59<br><br>57.92 90.61 57.89 90.60<br><br>57.91 90.58<br><br>57.93 90.62<br><br><br>57.92 90.63<br>|
|Upsample Anything (prob.)|56.36 90.05|

- Table 7. Segmentation performance on the Cityscapes dataset using the official LoftUp evaluation pipeline.

##### 8. Details of Probabilistic Map Upsampling in Table 1.

Figure 8-(c) corresponds to the Upsample Anything (prob.) configuration reported in Table 1. In this setting, the segmentation map is predicted from downsampled features using a lightweight 1×1 convolution, followed by our prob-

[Figure 101]

[Figure 102]

Input SegMap

Encoder Decoder

(a) Conventional Segmentation pipeline

[Figure 103]

[Figure 104]

FeatureUpsampler

SegMap

Input

Encoder

Feature Upsampler

- (b) Feature Upsampling for Segmentation Pipeline
- (c) Upsample Anything Segmentation Pipeline

[Figure 105]

[Figure 106]

UpsampleAnything

1x1 Conv

[Figure 107]

Input

Output

Encoder

Figure 8. Comparison of segmentation pipelines. (a) Conventional segmentation pipeline with a Vision Foundation Model encoder and task-specific decoders such as DPT, UPerNet, SegFormer, or Mask2Former. (b) Feature upsampling pipeline using pretrained upsamplers such as FeatUP, LoftUP, JAFAR, or AnyUp, operating on feature maps. (c) Our proposed Upsample Anything, which performs test-time optimization and handles both feature and segmentation upsampling without additional training.

abilistic upsampling to reconstruct high-resolution outputs. This simple setup already shows strong performance. Because the computation is performed on a small feature map, heavier or more complex decoders are expected to be feasible without large computational overhead. This suggests that the proposed segmentation pipeline has further potential, although designing such decoders is beyond the scope of this work.

##### 9. Details of Depth Estimation in Table 2.

We followed the DPT-based depth estimation setup used in prior works [10, 23]. Although DPT includes an internal up-

Algorithm 1 Depth estimation setting used in Table 2. Require: Input image x Ensure: Predicted depth map D

- 1: Extract multi-scale features F using a pretrained Vision Foundation Model encoder (e.g., DINOv2)
- 2: Pass F through the DPT head:
- 3: F1 ← Conv(F, k=3, c→c/2)
- 4: F2 ← Conv(F1, k=3, c/2→32)
- 5: F3 ← ReLU(F2)
- 6: Dinv ← Conv(F3, k=1, 32→1)
- 7: Dinv ← ReLU(Dinv) (if non-negative)
- 8: if invert is True then
- 9: D ← clip(s·D 1

inv+t, 1e−8, ∞)

- 10: else
- 11: D ← Dinv
- 12: end if
- 13: Output the final depth prediction D

sampling, its exact implementation details are not provided in the paper or codebase. Therefore, we reimplemented the head as described in Algorithm 1 and used it consistently for all our depth estimation experiments.

##### 10. From 2D Low-Resolution Feature Maps to 3D High-Resolution Feature Volumes

We extend our test-time optimization (TTO) framework to reconstruct dense 3D feature volumes directly from lowresolution 2D feature maps. Starting with an RGB–Depth (RGB-D) pair, we first downsample the RGB image by a factor of s to simulate a low-resolution feature space. The corresponding high-resolution RGB-D map is used as the guide signal for optimization. During TTO stage, we train only the pixel-wise anisotropic Gaussian kernel parameters (σx,σy,σz,θ,σr) so that the 3D Upsample Anyting can accurately project the low-resolution RGB features to their high-resolution RGB-D counterparts. Once optimized, these learned kernels are frozen and reused in upsample stage to upsample semantic features extracted from 2D LR feature into full 3D feature volumes. This process allows each low-resolution feature token to be expanded not only spatially along the x–y plane, but also along the depth axis z, guided by the HR depth map. The resulting tensor F3D ∈ RD

h×C×Hh×Wh captures the local geometric and appearance-aware structure of the scene. We visualize these 3D feature maps using PCA on each depth slice, revealing how distinct depth layers retain meaningful semantic separation while smoothly transitioning across depth. Figure 9 shows that even without explicit 3D supervision, our Full3DJBU reconstructs volumetric features that align with depth continuity, edges, and object boundaries—demonstrating that our framework can generalize

from 2D low-resolution feature inputs to 3D high-resolution representations at test time.

##### 11. Gaussian Blob Visualization

To better understand what our learned anisotropic kernels capture, we visualized the Gaussian blobs of our model, as shown in Fig. 10. (a) shows the original high-resolution RGB image, and (b) overlays the learned Gaussian blobs on the corresponding low-resolution image. Although the visualization can be difficult to interpret directly, certain spatial regions such as the eyes, nose, and corners exhibit overlapping or consistently oriented blobs, which suggests that nearby kernels capture semantically similar local structures. This indicates that the learned kernels adaptively encode meaningful directional features rather than behaving randomly. However, similar to other methods that rely on Gaussian Splatting or 2DGS, not every blob is fully interpretable, and some visual noise appears due to overparameterization and kernel redundancy.

##### 12. Segment-then-Upsample Pipeline Visualization Results

This section presents the visualization results of our segment-then-upsample pipeline, corresponding to the method in Fig. 8-(c). In this configuration, we perform semantic segmentation on the low-resolution feature maps first and subsequently upsample the segmentation logits by a factor of 16×. Despite the large upsampling ratio, our Upsample Anything produces visually sharp and semantically consistent results, as illustrated in Fig. 11. Compared to conventional bilinear interpolation, the recovered boundaries and fine structures are significantly clearer.

##### 13. Formal Relation Between Joint Bilateral Upsampling and Gaussian Splatting

The purpose of this section is not to claim that Joint Bilateral Upsampling (JBU) and Gaussian Splatting (GS) are mathematically equivalent. Instead, we aim to show why the GS framework provides a useful foundation for our formulation. By reinterpreting JBU through the perspective of GS, we reveal a common idea based on continuous and differentiable Gaussian kernels, which motivates our use of GS-style parameter learning in the Upsample Anything (GSJBU) framework. In short, this section clarifies the conceptual link between the two views and explains why GSbased test-time optimization naturally applies to feature upsampling.

Notation. Let Flr : Q → RC be a low-resolution feature map on a discrete grid Q ⊂ Z2, and let I : Ω → Rd be an HR guidance signal (d=1 for grayscale, d=3 for RGB,

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Image Depth Slices 1 Slices 3 Slices 5 Slices 7

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

Slices 9 Slices 11 Slices 13 Slices 15 Slices 17 Slices 19

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Slices 21 Slices 23 Slices 25 Slices 27 Slices 29 Slices 31

- Figure 9. Visualization of our 3D feature upsampling results. The first two columns show the RGB image and its corresponding depth map. The remaining panels depict representative depth slices from the reconstructed 3D high-resolution feature volume obtained using our Upsample Anything. Each slice is visualized via PCA projection into RGB space. Notice that the recovered 3D feature layers exhibit smooth transitions along the depth axis while preserving fine object boundaries and geometric continuity.

|𝐼𝐻𝑅(224,224)|
|---|
| |
|(a) 𝐼𝑚𝑎𝑔𝑒|

[Figure 126]

[Figure 127]

|(b) 𝐺𝑎𝑢𝑠𝑠𝑖𝑎𝑛 𝐵𝑙𝑜𝑏|
|---|

- Figure 10. Visualization of learned Gaussian blobs. (a) shows the original image and (b) displays the Gaussian blobs overlaid on the low-resolution input. The blobs reveal locally coherent directions and magnitudes, indicating that the learned kernels adapt to the underlying structure of the scene.

###### Joint spatial–range lifting. Define the lifted embedding

x I(x)

ϕ : Ω → R2+d, ϕ(x) :=

,

and the block-diagonal covariance

Λ(σs,σr) := diag σs2I2, σr2Id ∈ R(2+d)×(2+d). For u,v ∈ R2+d, let

GΛ(u,v) := exp − 21 (u − v)⊤Λ−1(u − v) .

Theorem 1 (JBU as a normalized Gaussian mixture in the joint domain). Fix σs>0, σr>0 and let Λ = Λ(σs,σr). Then for any p ∈ Ω,

Flr(q) GΛ ϕ(p), ϕ(q)

Fˆhr(p) = q∈Ω(p)

. (9)

GΛ ϕ(p), ϕ(q)

q∈Ω(p)

In particular, JBU coincides with evaluating a normalized Gaussian mixture in the lifted space R2+d whose centers are {ϕ(q)}q∈Ω(p) and whose (isotropic-by-block) covariance is Λ.

etc.). For p ∈ Ω ⊂ R2, classical JBU is

2

2 2σr2

Flr(q) exp − ∥p−q∥

2σs2 exp − ∥I(p)−I(q)∥

Proof. By construction, ∥ϕ(p) − ϕ(q)∥2Λ−1 = (p − q)⊤(σs−2I2)(p−q)+(I(p)−I(q))⊤(σr−2Id)(I(p)−I(q)). Thus GΛ(ϕ(p), ϕ(q)) = exp − ∥p−q∥

Fˆhr(p) = q∈Ω(p)

.

2

2 2σr2

exp − ∥p−q∥

2σs2 exp − ∥I(p)−I(q)∥

2

2

2σs2 exp − ∥I(p)−I(q)∥

2σr2 , and substituting this identity in (1) yields (9).

q∈Ω(p)

(8)

| |
|---|

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

Image Bilinear Ours GT Image Bilinear Ours GT

- Figure 11. Visualization of the segment-then-upsample pipeline. The segmentation logits are first generated at low resolution and then upsampled by 16× using our method. The results exhibit remarkably sharp object boundaries and preserve semantic coherence, highlighting the effectiveness of our upsampling approach.

Corollary 1 (Discrete GS view in the joint domain). Let µq := ϕ(q) and fq := Flr(q). Then Theorem 1 states that JBU equals

fq exp − 12(ϕ(p) − µq)⊤Λ−1(ϕ(p) − µq)

Fˆhr(p) = q

,

exp − 12(ϕ(p) − µq)⊤Λ−1(ϕ(p) − µq)

q

(10)

i.e., a Gaussian Splatting evaluation in R2+d with fixed block-diagonal covariance and centers on the lifted LR grid.

###### Connection to standard 2D GS. Standard (2D) GS writes, for p ∈ R2,

- αi exp − 12(p − µ˜i)⊤Σ˜−i 1(p − µ˜i) f ˜i

j

- αj exp − 12(p − µ˜j)⊤Σ˜−j 1(p − µ˜j)

F(p) = i

. (11)

The range term in JBU can be absorbed by lifting to the joint domain (Theorem 1), or, equivalently, by keeping the domain 2D and letting the amplitude be query-dependent,

i)∥2

αi(p) = exp − ∥I(p)−I(˜µ

2σr2 . The former is strictly query-independent and thus mathematically cleaner; the latter matches common GS implementations with viewdependent weights.

Theorem 2 (Specialization of GSJBU to JBU (isotropic limit)). Consider the anisotropic per-center model:

fq exp − 21(p − q)⊤Σ−q 1(p − q) βq(p)

F(p) = q

,

exp − 21(p − q)⊤Σ−q 1(p − q) βq(p)

(12)

q

2

βq(p) := exp − ∥I(p)−I(q)∥

2σr2(q) .

If Σq → σs2I2 and σr(q) → σr for all q, then (12) reduces exactly to JBU (1).

Proof. Substitute Σq = σs2I2 and σr(q) = σr into (12) to recover the numerator/denominator of (1).

| |
|---|

Proposition 1 (Discrete-to-continuous convergence). Assume Flr admits a bandlimited (or Lipschitz-continuous) interpolation F˜ : Ω → RC, and let the LR grid spacing be ∆x. Then as ∆x → 0,

F˜(q) GΛ ϕ(p), ϕ(q) (∆x)2 −→

F˜(x) GΛ ϕ(p), ϕ(x) dx,

q

Ω

(13)

and the corresponding normalized ratios converge as well. Hence, discrete JBU converges to its continuous lifteddomain GS counterpart.

Sketch. GΛ(ϕ(p), ϕ(·)) is bounded and continuous for fixed p. Under the stated regularity, Riemann sums converge to the integral

###### Parameter Symbol Default Role Rationale

- Spatial sigma (x) σx init = scale (e.g., 16) Controls major-axis smoothing; receptive-field size Initialized proportional to upsampling factor to provide a wide prior; refined by TTO.

- Spatial sigma (y) σy Same as σx Controls minor-axis smoothing Same reasoning as σx; enables anisotropy to emerge during TTO. Orientation θ 0 Rotation of the anisotropic Gaussian Zero-init avoids directional bias; TTO discovers optimal orientation. Range sigma σr 0.12 Sensitivity to appearance/color similarity Moderate color differences (∆I ≈ 0.2 ∼ 0.3) are significantly downweighted; acts as soft bilateral prior. Support radius (max) Rmax 4–8 Upper bound on spatial Gaussian support Balances context capture and cost (O((2Rmax + 1)2)); too small truncates optimal kernels. Dynamic multiplier αdyn 2.0 Converts σeff to effective support radius Ensures coverage of ∼ 95% Gaussian mass (2σ rule); prevents under-coverage early in TTO. Center mode – nearest Determines LR anchor for each HR pixel Nearest-center alignment improves stability and avoids aliasing for large upsampling factors.

- Table 8. Hyperparameter settings for Upsample Anything. All parameters act as soft priors; the effective kernel shape is governed by test-time optimization of pixelwise anisotropic Gaussians.

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Clean Image LR Feature AnyUP Ours

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

Noise 10% Image LR Feature AnyUP Ours

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

Noise 20% Image LR Feature AnyUP Ours

- Figure 12. Qualitative comparison under low-SNR and noise corruption. From left to right: RGB input, low-resolution feature, upsampled feature by AnyUp, and ours (Upsample Anything). From top to bottom: clean image, 10% noise, and 20% noise. AnyUp remains stable under noise, while our TTO-based method overfits to noisy pixels, revealing its limitation when directly optimizing on corrupted inputs.

and the denominators stay strictly positive (finite kernel mass). The ratio convergence follows by standard arguments (e.g., dominated convergence and continuity of division on R \ {0}).

| |
|---|

Consequences. (i) Equivalence in the joint domain (Thm. 1) shows that JBU is a GS evaluation on (x,I(x)) with block-diagonal covariance. (ii) Anisotropic generalization (12) recovers JBU in the isotropic limit (Thm. 2), and enables per-center covariance learning (our GSJBU). (iii) Discrete-to-continuous consistency (Prop. 1) justifies replacing sums by integrals when refining the sampling grid.

Implementation note. In practice we adopt (12) with test-time optimization of (Σq,σr(q)). For stability, Σq ≻ 0 is parameterized via R(θq)diag(σx2(q),σy2(q))R(θq)⊤ with σx,σy>0.

##### 14. Hyperparameter Table

The hyperparameters in Table 8 function primarily as soft priors for test-time optimization. Since all spatial and range parameters are refined during the 50 optimization steps, the final performance depends only weakly on their initial values. A well-chosen initialization simply accelerates convergence, whereas suboptimal values are eventually corrected by the optimization itself. The table therefore summarizes practical initialization rules rather than strict hyperparameter requirements. These rules are based on the expected receptive-field size, the dynamic range of the guidance image, and the desired locality prior, and they lead to stable and fast convergence.

##### 15. Limitation under Low-SNR or Corrupted Inputs

Although our method performs robustly across diverse datasets and even under moderate perturbations such as those in ImageNet-C, it exhibits a clear limitation when applied to images with extremely low signal-to-noise ratios or severe corruption.

Because our framework performs test-time optimization (TTO) by reconstructing the input image itself, the optimization process inherently assumes that the image contains a clean and reliable signal. When the input is degraded by noise—such as salt-and-pepper artifacts or heavy sensor perturbations—the model tends to overfit to these corruptions rather than recovering the underlying structure. Figure 12 illustrates this effect: the first row shows results on a clean image, while the second and third rows demonstrate increasing corruption levels of 10% and 20%, respectively.

Despite the noise, pretrained Vision Foundation Models still produce reasonable feature embeddings, and AnyUp remains stable by directly upsampling feature maps. In contrast, our TTO-based Upsample Anything reconstructs the noisy signal faithfully, which unintentionally amplifies noise in both the reconstructed RGB and upsampled feature domains.

This limitation is not unique to our method but is common across all TTO-based image restoration approaches that optimize directly on corrupted inputs. While one could incorporate a denoising stage before optimization to alleviate this issue, we consider it outside the current scope.

In summary, AnyUp demonstrates higher robustness under corrupted or low-SNR conditions, whereas our Upsample Anything excels when inputs are visually clean or when handling multi-modal signals such as RGB-D or 3D features.

##### References

- [1] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 2
- [2] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 1
- [3] Paul Couairon, Loick Chambon, Louis Serrano, JeanEmmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Jafar: Jack up any feature at any resolution. arXiv preprint arXiv:2506.11136, 2025. 2, 3, 5
- [4] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 2
- [5] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision, 88(2):303–338, 2010. 1
- [6] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. Featup: A model-agnostic framework for features at any resolution. In The Twelfth International Conference on Learning Representations, 2024. 2, 3, 5
- [7] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016. 2
- [8] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2
- [9] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 7
- [10] Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, and Dan Zhang. Loftup: Learning a coordinatebased feature upsampler for vision foundation models, 2025. 2, 3, 5, 1
- [11] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9492–9502,

2024. 1

- [12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 3

- [13] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 2
- [14] Johannes Kopf, Michael F Cohen, Dani Lischinski, and Matt Uyttendaele. Joint bilateral upsampling. ACM Transactions on Graphics (ToG), 26(3):96–es, 2007. 3, 4
- [15] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 1
- [16] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11976–11986,

2022. 2

- [17] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2
- [18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 2
- [19] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 12179–12188, 2021. 2
- [20] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012. 1
- [21] Shuangbing Song, Fan Zhong, Tianju Wang, Xueying Qin, and Changhe Tu. Guided linear upsampling. ACM Transactions on Graphics (TOG), 42(4):1–12, 2023. 6, 7
- [22] Saksham Suri, Matthew Walmer, Kamal Gupta, and Abhinav Shrivastava. Lift: A surprisingly simple lightweight feature transform for dense vit descriptors. In European Conference on Computer Vision, pages 110–128. Springer, 2024. 2, 3, 5
- [23] Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric Lenssen. Anyup: Universal feature upsampling. arXiv preprint arXiv:2510.12764, 2025. 2, 3, 5, 7, 1
- [24] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV), pages 418–434, 2018. 2
- [25] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer: Simple and

- efficient design for semantic segmentation with transformers. Advances in neural information processing systems, 34: 12077–12090, 2021. 2
- [26] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. Advances in Neural Information Processing Systems, 37:21875–21911, 2024. 1
- [27] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 2
- [28] Xinjie Zhang, Xingtong Ge, Tongda Xu, Dailan He, Yan Wang, Hongwei Qin, Guo Lu, Jing Geng, and Jun Zhang. Gaussianimage: 1000 fps image representation and compression by 2d gaussian splatting. In European Conference on Computer Vision, pages 327–345. Springer, 2024. 3
- [29] Yunxiang Zhang, Bingxuan Li, Alexandr Kuznetsov, Akshay Jindal, Stavros Diolatzis, Kenneth Chen, Anton Sochenov, Anton Kaplanyan, and Qi Sun. Image-gs: Content-adaptive image representation via 2d gaussians. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11,

2025. 3

- [30] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302–321, 2019. 1
- [31] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832,

2021. 2

- [32] Lingting Zhu, Guying Lin, Jinnan Chen, Xinjie Zhang, Zhenchao Jin, Zhao Wang, and Lequan Yu. Large images are gaussians: High-quality large image representation with levels of 2d gaussian splatting. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 10977–10985,

2025. 3

