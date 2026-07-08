# arXiv:2509.19297v3[cs.CV]24Jun2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

September 23, 2025

## VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction

Weijie Wang1,2,∗ Yeqing Chen1,∗ Zeyu Zhang2 Hengyu Liu2,3 Haoxiao Wang1 Zhiyuan Feng4 Wenkang Qin2 Feng Chen5 Jia-Wang Bian6 Zheng Zhu2,‡ Donny Y. Chen7 Bohan Zhuang1,‡

1 Zhejiang University 2 GigaAI 3 The Chinese University of Hong Kong 4 Tsinghua University 5 Adelaide University 6 Nanyang Technological University 7 Monash University

[Figure 8]

[Figure 9]

| |[Figure 10]<br><br>[Figure 11]|[Figure 12]|[Figure 13]|
|---|---|---|---|
|[Figure 14]| |[Figure 15]<br><br>[Figure 16]|[Figure 17]|
| |[Figure 18]<br><br>[Figure 19]| |[Figure 20]<br><br>[Figure 21]|
|[Figure 22]| |[Figure 23]|[Figure 24]|

| |[Figure 25]<br><br>[Figure 26]| |[Figure 27]<br><br>[Figure 28]|
|---|---|---|---|
| |[Figure 29]|[Figure 30]<br><br>[Figure 31]|[Figure 32]|
| |[Figure 33]<br><br>[Figure 34]|[Figure 35]|[Figure 36]|
|[Figure 37]|[Figure 38]|[Figure 39]| |

|[Figure 40]|[Figure 41]| |[Figure 42]<br><br>[Figure 43]|
|---|---|---|---|
| |[Figure 44]<br><br>[Figure 45]|[Figure 46]|[Figure 47]|
| |[Figure 48]<br><br>[Figure 49]| |[Figure 50]<br><br>[Figure 51]|
|[Figure 52]| |[Figure 53]|[Figure 54]|

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

Novel

|[Figure 63]|[Figure 64]|[Figure 65]|[Figure 66]|
|---|---|---|---|
| |[Figure 67]<br><br>[Figure 68]|[Figure 69]|[Figure 70]|
|[Figure 71]|[Figure 72]|[Figure 73]|[Figure 74]|
| |[Figure 75]|[Figure 76]|[Figure 77]|

| |[Figure 78]|[Figure 79]<br><br>[Figure 80]|[Figure 81]|
|---|---|---|---|
| |[Figure 82]<br><br>[Figure 83]|[Figure 84]|[Figure 85]|
|[Figure 86]| |[Figure 87]|[Figure 88]<br><br>[Figure 89]|
| |[Figure 90]|[Figure 91]<br><br>[Figure 92]| |

|[Figure 93]| |[Figure 94]<br><br>[Figure 95]|[Figure 96]|
|---|---|---|---|
| |[Figure 97]<br><br>[Figure 98]|[Figure 99]|[Figure 100]|
|[Figure 101]| |[Figure 102]<br><br>[Figure 103]|[Figure 104]|
|[Figure 105]|[Figure 106]| |[Figure 107]|

Novel

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

View

View

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

x

[Figure 117]

[Figure 118]

|[Figure 119]|[Figure 120]|[Figure 121]|[Figure 122]|
|---|---|---|---|
|[Figure 123]| |[Figure 124]<br><br>[Figure 125]|[Figure 126]|
|[Figure 127]| |[Figure 128]<br><br>[Figure 129]|[Figure 130]|
|[Figure 131]| |[Figure 132]<br><br>[Figure 133]| |

| |[Figure 134]<br><br>[Figure 135]|[Figure 136]|[Figure 137]|
|---|---|---|---|
| |[Figure 138]<br><br>[Figure 139]| |[Figure 140]<br><br>[Figure 141]|
| |[Figure 142]<br><br>[Figure 143]|[Figure 144]|[Figure 145]|
|[Figure 146]|[Figure 147]|[Figure 148]| |

| |[Figure 149]<br><br>[Figure 150]|[Figure 151]|[Figure 152]|
|---|---|---|---|
| |[Figure 153]<br><br>[Figure 154]| |[Figure 155]<br><br>[Figure 156]|
| |[Figure 157]<br><br>[Figure 158]|[Figure 159]|[Figure 160]|
|[Figure 161]|[Figure 162]| |[Figure 163]|

[Figure 164]

Gaussians Gaussian

Gaussians Feature Unprojection

[Figure 165]

[Figure 166]

[Figure 167]

Cross-view

Per-pixel

Voxel-aligned

Rendering

Rendering

Matching

Prediction

Merge

and Refinement

Prediction

(a) Pixel-aligned Feed-forward 3DGS (b) Voxel-aligned Feed-forward 3DGS (Ours)

Figure 1 Comparison between the pixel-aligned feed-forward method and our approach. Pixel-aligned feed-forward 3DGS methods suffer from two primary limitations: 1) 2D feature matching struggles to effectively resolve the multi-view alignment problem, and 2) the Gaussian density is constrained and cannot be adaptively controlled according to scene complexity. We propose VolSplat, a framework that directly regresses Gaussians from 3D features based on a voxel-aligned prediction strategy.

Abstract. Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective solution for novel view synthesis. Existing methods predominantly rely on a pixel-aligned Gaussian prediction paradigm, where each 2D pixel is mapped to a 3D Gaussian. We rethink this widely adopted formulation and identify several inherent limitations: it renders the reconstructed 3D models heavily dependent on the number of input views, leads to view-biased density distributions, and introduces alignment errors, particularly when source views contain occlusions or low texture. To address these challenges, we introduce VolSplat, a new multi-view feed-forward paradigm that replaces pixel alignment with voxel-aligned Gaussians. By directly predicting Gaussians from a predicted 3D voxel grid, it overcomes pixel alignment’s reliance on error-prone 2D feature matching, ensuring robust multi-view consistency. Furthermore, it enables adaptive control over density based on 3D scene complexity, yielding more faithful Gaussians, improved geometric consistency, and enhanced novel-view rendering quality. Experiments on widely used benchmarks demonstrate that VolSplat achieves state-of-the-art performance, while producing more plausible and view-consistent results.

Project Page: lhmd.top/volsplat Correspondence: zhengzhu@ieee.org and bohan.zhuang@zju.edu.cn Conference: The 19th European Conference on Computer Vision (ECCV 2026) Keywords: 3D Gaussians, Feed-Forward Reconstruction, View Synthesis Date: September 23, 2025

∗ Equal contribution. ‡ Corresponding authors.

### 1 Introduction

3D reconstruction is a cornerstone of modern robotics, empowering autonomous systems with the critical ability to perceive, map, and comprehend their physical environment [1], which is fundamental for advanced navigation, object manipulation, and world models. Traditional optimization based approaches, including Neural Radiance Fields (NeRF) [2] and 3D Gaussian Splatting (3DGS) [3], obtain high fidelity results by iteratively enforcing photometric or geometric consistency. These methods achieve excellent accuracy but are computationally intensive and slow to run at inference time. By contrast, feed-forward approaches [4–14] trade per instance optimization for fast learned inference. A single forward pass predicts scene geometry or a 3D representation directly from input images. This speed and simplicity make feed-forward systems attractive for real time applications, large scale datasets, and downstream tasks that require many reconstructions.

Prior feed-forward 3DGS methods [7, 9, 10, 12, 14–16] commonly rely on pixel alignment as their fundamental mechanism for associating image features with pixel aligned Gaussians. In this design, per-pixel features from precomputed image feature maps are unprojected to define the corresponding Gaussians. The prevailing consensus has been to perform fusion directly within

- the 2D feature representation. However, pixel aligned designs inherit two primary limitations. 1) Sampling at discrete pixel locations is sensitive to camera calibration and discretization error, produces inconsistent sampling patterns across views. 2) The rigid pixel-to-Gaussian association enforces a uniform density distribution that ignores scene complexity, leading to redundant primitives in simple regions while failing to capture fine-grained 3D structures.

In this work, we shift the alignment paradigm from pixels to voxels, as illustrated in Fig. 1. Instead of sampling features at projected pixel coordinates, we align and aggregate image features directly into a 3D voxel grid. Multi-view image features are aggregated directly into this 3D voxel space, effectively decoupling feature fusion from the camera view frustums. Within this unified voxel space, we employ a 3D U-Net [17] to reason about scene geometry and appearance volumetrically. Finally, rather than predicting Gaussians per pixel, we predict primitives directly from the refined voxel features, allowing the distribution of 3D Gaussians to be determined by the volumetric structure itself.

There are practical and conceptual advantages to voxel alignment. Specifically, volumetric aggregation reduces floaters and view dependent inconsistency because information from multiple views is fused into a shared 3D container before Gaussian prediction. Simultaneously, operating in a 3D grid enables the use of well studied 3D decoder and regularization strategies, which naturally encode locality and geometrical context. Instead of the integration of auxiliary 3D signals such as depth maps [10] and point clouds [18], our approach naturally resolves spatial ambiguities within the unified voxel space, thereby eliminating the need for such ad hoc priors or auxiliary supervision signals. Furthermore, voxel representations are amenable to modern acceleration strategies such as sparse data structures, making the approach practical at the resolutions required for high quality reconstruction.

In this paper we present a feed-forward three dimensional reconstruction framework built around voxel alignment. As shown in Fig. 2, we first construct 3D feature grids using the extracted 2D image features, then refine the 3D features and use them to predict voxel-aligned Gaussians. We analyze the alignment errors that arise in pixel aligned pipelines and show how voxel alignment reduces these errors both conceptually and empirically. Through systematic experiments on synthetic and real world benchmarks, we demonstrate that voxel aligned feed-forward models achieve more accurate and robust reconstructions than comparable pixel aligned baselines on large-scale benchmarks such as RealEstate10K [19], ScanNet [20] and ACID [21]. Our contributions are as follows:

- • We introduce voxel alignment as a principled alternative to pixel alignment for feed-forward

- 3DGS and present a practical end-to-end framework.
- • We provide an analysis of alignment induced errors in pixel aligned systems and show how volumetric aggregation mitigates these failure modes.
- • Experimental results demonstrate that VolSplat achieves state-of-the-art (SOTA) performance on several large-scale benchmarks.

### 2 Related Work

Novel view synthesis. Traditional approaches to Novel View Synthesis (NVS) primarily rely on geometry-based rendering methods that reconstruct explicit 3D scene geometry from images [22], image-based rendering techniques that interpolate between captured views without full 3D reconstruction [23], and light field rendering that samples and reprojects densely captured rays in space [24]. These methods required either accurate geometric proxies, densely sampled viewpoints, or both to produce convincing visual results, limiting their applicability in real-world scenarios. The emergence of NeRF [2] marked a paradigm shift, significantly improving both rendering quality and robustness over prior methods, which learns a continuous, implicit scene representation by utilizing a MLP to map position and viewing direction to a corresponding color and volume density. While NeRF-based methods [25, 26] require a long training time due to the per-ray rendering. 3DGS [3] and its variants [27–29] have been introduced to represent the 3D scene using a set of anisotropic 3D Gaussians.

##### 3D voxelization. Voxelization, which discretizes 3D space into regular voxel grids, has been a foundational representation in 3D reconstruction and modeling [30]. Prior methods used dense grids for their simplicity, but suffered from high memory costs and poor scalability [31]. To address this, sparse structures like octrees were introduced for more efficient storage and computation [32]. In modern applications, voxels are widely used as input to 3D Convolutional Neural Network (CNN) for tasks such as object detection [33] and semantic segmentation [34, 35]. More recently, voxels are often used as sparse scaffolding rather than as the final representation, supporting more advanced rendering techniques. Representative methods include Plenoxels [36] and K-Planes [37], which optimize voxel-based radiance fields for fast, high-quality rendering, as well as structured strategies such as Scaffold-GS [38] and Octree-GS [39], which leverage voxel grids to organize and accelerate 3DGS.

Feed-forward 3D Gaussian Splatting. Recent developments in feed-forward 3DGS [7, 9, 10, 12, 40– 44] offer a compelling alternative that directly predicts 3D Gaussians from input images in a single forward pass: pixelSplat [7] proposes a two-view feed-forward pipeline that combines epipolar transformers and depth prediction to generate Gaussians. MVSplat [9] introduces a cost-volume-based fusion strategy to enhance multi-view consistency. DepthSplat [10] leverages monocular depth features to improve fine 3D structure reconstruction from sparse views. Follow-up work extends feed-forward 3DGS to more complex scenarios, including pose-free inputs [45, 46], online stream inputs [12], and more dense inputs [11], which can provide priors for the world models [47–49]. While these works adopt a pixel-aligned strategy to predict Gaussian primitives, the pixel-wise formulation struggles to handle multiple input views due to redundancy and inconsistency across pixels. Existing methods attempt to improve the per-pixel strategy by pruning the number of Gaussians [40], token merging [11, 50] and voxel-based fusion [41, 51, 52]. However, these approaches do not fundamentally address the limitations inherent in per-pixel processing. EVolSplat [53] has explored voxel features in autonomous driving scenarios, but it has not been generalized to general scenarios and requires explicit 3D point clouds as intermediate representations. In contrast, our method introduces a voxel-aligned method, which eliminate the need for per-query 2D prediction patterns. This alignment enables more stable multi-view fusion,

[Figure 168]

| |[Figure 169]<br><br>[Figure 170]|[Figure 171]|[Figure 172]| |
|---|---|---|---|---|
| |[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]|[Figure 176]|[Figure 177]<br><br>[Figure 178]|[Figure 179]<br><br>[Figure 180]|
| |[Figure 181]<br><br>[Figure 182]<br><br>[Figure 183]|[Figure 184]<br><br>[Figure 185]|[Figure 186]<br><br>[Figure 187]|[Figure 188]|
|[Figure 189]<br><br>[Figure 190]|[Figure 191]|[Figure 192]|[Figure 193]<br><br>[Figure 194]|[Figure 195]<br><br>[Figure 196]|
|[Figure 197]| |[Figure 198]|[Figure 199]| |

[Figure 200]

|[Figure 201]|[Figure 202]|[Figure 203]|[Figure 204]|
|---|---|---|---|
|[Figure 205]|[Figure 206]| |[Figure 207]<br><br>[Figure 208]|
|[Figure 209]| |[Figure 210]|[Figure 211]<br><br>[Figure 212]|
| |[Figure 213]<br><br>[Figure 214]|[Figure 215]| |

[Figure 216]

[Figure 217]

Mutli-viewTransformerfor2DFeatureExtraction

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

Depth Prediction Module

[Figure 222]

[Figure 223]

| |[Figure 224]<br><br>[Figure 225]| |[Figure 226]<br><br>[Figure 227]| |
|---|---|---|---|---|
|[Figure 228]<br><br>[Figure 229]| |[Figure 230]<br><br>[Figure 231]<br><br>[Figure 232]|[Figure 233]|[Figure 234]<br><br>[Figure 235]|
| |[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]|[Figure 239]<br><br>[Figure 240]|[Figure 241]|[Figure 242]<br><br>[Figure 243]|
|[Figure 244]<br><br>[Figure 245]|[Figure 246]|[Figure 247]|[Figure 248]<br><br>[Figure 249]<br><br>[Figure 250]|[Figure 251]|
|[Figure 252]| |[Figure 253]|[Figure 254]| |

[Figure 255]

|[Figure 256]|[Figure 257]| |[Figure 258]<br><br>[Figure 259]|
|---|---|---|---|
| |[Figure 260]<br><br>[Figure 261]|[Figure 262]|[Figure 263]|
| |[Figure 264]<br><br>[Figure 265]|[Figure 266]|[Figure 267]|
| |[Figure 268]|[Figure 269]<br><br>[Figure 270]| |

Gaussians

Depth Maps

[Figure 271]

[Figure 272]

Voxel-aligned Unproject Prediction

[Figure 273]

[Figure 274]

|[Figure 275]| |[Figure 276]<br><br>[Figure 277]| |[Figure 278]|
|---|---|---|---|---|
|[Figure 279]<br><br>[Figure 280]|[Figure 281]|[Figure 282]|[Figure 283]<br><br>[Figure 284]<br><br>[Figure 285]|[Figure 286]|
|[Figure 287]<br><br>[Figure 288]| |[Figure 289]<br><br>[Figure 290]<br><br>[Figure 291]|[Figure 292]|[Figure 293]<br><br>[Figure 294]|
| |[Figure 295]<br><br>[Figure 296]<br><br>[Figure 297]|[Figure 298]|[Figure 299]<br><br>[Figure 300]<br><br>[Figure 301]|[Figure 302]|
|[Figure 303]| |[Figure 304]|[Figure 305]| |

Sparse 3D Decoder

|[Figure 306]| |[Figure 307]|[Figure 308]<br><br>[Figure 309]|
|---|---|---|---|
| |[Figure 310]<br><br>[Figure 311]| |[Figure 312]<br><br>[Figure 313]|
|[Figure 314]| |[Figure 315]<br><br>[Figure 316]|[Figure 317]|
|[Figure 318]|[Figure 319]| |[Figure 320]|

x

[Figure 321]

[Figure 322]

Input Images

Image Features Cost Volumes Voxel Features Refined Voxel Features

- Figure 2 Overview of VolSplat. Given multi-view images as input, we first extract 2D features for each image using a Transformer-based network and construct per-view cost volumes with plane sweeping. Depth Prediction Module then estimates a depth map for each view, which is used to unproject the 2D features into 3D space to form a voxel feature grid. Subsequently, we employ a sparse 3D decoder (details in Sec. 3.3) to refine these features in 3D space and predict the parameters of a 3D Gaussian for each occupied voxel. Finally, novel views are rendered from the predicted 3D Gaussians. cleaner occlusion handling, and more coherent joint inference of geometry and appearance. 3 Method

#### 3.1 Preliminary and Observation

Feed-forward 3D reconstruction aims to learn a mapping from N input images I = {Ii}Ni=1 where Ii ∈ RH×W×3 and their corresponding camera poses P = {Pi}Ni=1, to a 3D scene representation. In the context of pixel-aligned 3DGS, features are extracted from images and refined by cross-view interaction:

H

p ×Wp ×C (1)

F = {Fi}Ni=1 = h(Φimage(I,P)), Fi ∈ R

where Φimage is a pretrained image encoder. The function h is responsible for processing these features from different viewpoints, with its core purpose being to perform cross-view feature matching and fusion. For pixel-aligned Gaussian prediction, the features must be upsampled to the same resolution as the input image:

Ffull = U(F) ∈ RN×H×W×C, (2)

where Ffull denotes the full-resolution feature maps, U is a feature upsampler such as CNN-based network (in MVSplat [9]) and deconvolution-based network [54] (in DepthSplat [10]). Per-pixel Gaussian predictions are then performed using the upsampled features:

G = {(µi,Σi,αi,ci)}Hi=1×W×N = Ψpred(Ffull,P), (3) where the position of the Gaussians are determined by the predicted depth and pixel location.

While straightforward, this pixel-aligned formulation introduces two critical limitations. First, the geometric accuracy of the reconstruction is critically dependent on the quality of the predicted depth map. After depth unprojecting features into 3D space, the lack of interaction with neighboring points within the 3D space significantly contributes to the generation of floaters.

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

|[Figure 332]|
|---|

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

|[Figure 341]|
|---|

Add

x

x

Sparse Voxel Feature 3D U-Net Refinement Residual Voxel Feature

Refined Voxel Feature

- Figure 3 Architecture of Sparse 3D decoder. Sparse 3D features are fed into a 3D U-Net for processing, which predicts residual features for each voxel. These residual features are then added to the original 3D voxel features to obtain the refined features.

Second, the structure of the 3D representation is rigidly tied to the 2D image grid. The total number of Gaussians is fixed at |S| = H × W × N, which is often suboptimal and cause an over-densification of Gaussians on simple, texture-less surfaces and an insufficient number for representing complex geometry not captured at the pixel level. These observations reveal a fundamental bottleneck and motivate our proposed voxel-aligned framework, designed to decouple

- the 3D representation from the 2D pixel grid.

#### 3.2 3D Feature Construction

Feature extraction and matching. For N input images, we first apply a weight-sharing ResNet [55] backbone to each RGB image to obtain p× downsampled feature maps. These features are then refined with cross-view attention that exchanges information with the two nearest neighboring views. For efficiency, this cross-attention is implemented with the local window attention [56].

p ×Wp ×C) , where C denotes the feature dimension.

H

After this stage we obtain cross-view-aware Transformer features {Fi}Ni=1 (Fi ∈ R

Next, we build per-view cost volumes {Ci}Ni=1 using a plane-sweep strategy [57]. For each view i, we sample D candidate depths {dm}Dm=1, warp the feature from neighboring views to the reference view at each hypothesized depth, and compute pairwise feature similarities [9].These similarities

are aggregated by dot-product matching and stacked along the depth axis to form {Ci}Ni=1, where Ci ∈ R

H

p ×Wp ×D.

To produce robust, multi-view consistent depth estimates, a depth module fuses the monocular features Fmonoi Ni=1 (Fmonoi ∈ R

H p ×Wp ×C) with the cost volume Ci and regresses a dense per-pixel depth map Di ∈ RH×W, which serves as a geometric prior for lifting image features into 3D space. These per-view features Fi and depths Di are used in the next stage to construct 3D point clouds and voxel-based features for volumetric reasoning.

Lifting to 3D feature. Given the predicted depth maps Di and camera parameters, we conveniently aggregate different depth map views by transforming the point clouds into a global coordinate system. First each pixel (u,v) in image space is unprojected to a 3D point in the camera coordinate frame using the camera intrinsics. Then the 3D point is transformed into the world coordinate system via the corresponding extrinsic parameters, including the rotation matrix Ri and translation Ti vector.

  Di(u,v)K−1

  

  

   + Ti. (4)

- u
- v 1

Pworld = Ri Pcam + Ti = Ri

By repeating this process across all views, we obtain a dense |S| = H × W × N point cloud in world space, where each 3D point is associated with its corresponding image feature.

To convert the unstructured dense point cloud P into a structured volumetric representation, we voxelize the points [58]. For each 3D point p = (xp,yp,zp) we compute integer voxel index

(i,j,k) by dividing by the voxel size vs and rounding.

i = rnd

xp vs

, j = rnd

yp vs

, k = rnd

zp vs

, (5)

where rnd(·) denotes rounding to the nearest integer.

Let Si,j,k be the set of all points falling into voxel (i,j,k) and fpbe the image feature corresponding to each point p ∈ Si,j,k The features within this voxel are aggregated via average pooling along the channel dimension, resulting in the voxel feature Vi,j,k:

1 |Si,j,k| p∈S

Vi,j,k =

i,j,k

fp. (6)

#### 3.3 Feature Refinement and 3D Gaussians Prediction

Feature refinement. To improve the spatial consistency and structural fidelity of the voxel representation, we apply an explicit voxel feature refinement stage as shown in Fig. 3. Given an input voxel grid V (with per-voxel feature vectors), a sparse convolutional 3D U-Net [17] R predicts a residual voxel field R:

R = R(V ), Ri ∈ RV×C, (7)

where V denotes the set of occupied voxels and the refined voxel features are obtained by a residual update:

V ′ = V + R, V ′i ∈ RV×C, (8)

The refinement network is implemented with hierarchical sparse 3D convolutional blocks, symmetric encoder-decoder stages, and upsampling layers connected by skip connections. This architecture enables multi-scale fusion of local and global geometric context while keeping computation efficient through sparsity. The residual formulation encourages the network to learn correction terms (fine geometric detail and consistency cues) rather than relearning the entire feature content, which empirically stabilizes training and preserves the coarse voxel information supplied by the lifting stage.

- 3D Gaussians prediction. The output of our network for each voxel v is a set of learnable Gaussian parameters {[¯µj,α¯j,Σj,cj] ∈ R38}. These include the offset of the Gaussian center µ¯j, opacity α¯j, covariance Σj, and spherical harmonic color representation cj. To obtain the final rendering parameters, we apply the following transformations:

µj = r · (σ(¯µj) − 0.5) + Centerj, αj = σ(¯αj), (9)

where µj is the predicted 3D Gaussian center, and Centerj is the centroid of voxel v. We utilize the sigmoid activation σ(·) to restrict the learnable offset within a localized neighborhood. Specifically, the −0.5 shift facilitates symmetrical, bi-directional movement from the voxel center, while r (set to 3× voxel size) acts as a scaling factor to control the effective spatial extent of these refinements.

#### 3.4 Training Objectives

Our network predicts a collection of 3D Gaussians {(µv,αv,Σv,cv)}v∈V. These per-voxel Gaussians are subsequently used to synthesize images at novel camera poses. To ensure a fair comparison and maintain benchmarking consistency with SOTA feed-forward methods, we follow the training protocol established by DepthSplat [10]. Specifically, the network is trained end-to-end using

- Table 1 Quantitative comparisons on RealEstate10K [19]. The top section are pixel-aligned methods, and the middle section are methods that performs post-processing on pixel-aligned Gaussians. All baselines are retrained for fair comparison. Ground truth camera poses are provided for both AnySplat [41] and WorldMirror [52]. “OOM” represents that model cannot infer on a 96G GPU. We compare VolSplat against pixel-aligned and post-processing baselines under 6, 12, and 24 input views. VolSplat consistently achieves the best performance across all metrics and view settings.

6v 12v 24v PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Method

pixelSplat [7] 28.95 0.900 0.163 OOM OOM OOM OOM OOM OOM MVSplat [9] 29.13 0.924 0.091 26.97 0.912 0.101 26.23 0.903 0.108 TranSplat [61] 29.62 0.928 0.084 28.00 0.920 0.089 26.65 0.884 0.115 DepthSplat [10] 30.52 0.931 0.079 28.54 0.919 0.088 26.26 0.880 0.115

GGN [40] 26.68 0.879 0.133 25.83 0.870 0.142 23.86 0.840 0.171 AnySplat [41] 19.05 0.576 0.305 19.86 0.627 0.288 20.21 0.658 0.279 WorldMirror [52] 24.86 0.819 0.079 26.03 0.859 0.072 26.47 0.875 0.071

Ours 31.30 0.941 0.075 29.40 0.928 0.085 27.21 0.896 0.111

ground-truth RGB images as supervision. For a forward pass that renders M novel views, we optimize a combined photometric and perceptual loss:

M

L =

m=1

LMSE(Irender(m) ,Igt(m)) + λLLPIPS(Irender(m) ,Igt(m)) , (10)

where M is the number of novel views rendered in a single pass. Following DepthSplat [10], the weight λ for the perceptual loss LLPIPS is set to 0.05.

### 4 Experiments

#### 4.1 Experimental Setup

Datasets. We train our method using two expansive datasets, RealEstate10K [19] and ScanNet [20], and evaluate its performance on the held-out test splits of both. For RealEstate10K, we adopt the conventional partition of 67,477 training scenes and 7,289 test scenes. As for ScanNet, which consists of 1,513 videos of indoor scenes, we follow past work [12, 59, 60] in using roughly 100 scenes for training and 8 scenes for evaluation. These datasets span a wide variety of environments, including indoor and outdoor real-estate walkthroughs (RealEstate10K), and real-world videos of numerous scenes suitable for indoor robot applications (ScanNet). We resize training and test images to 256 × 256.

Baselines. We benchmark VolSplat against several recent feed-forward methods for sparse-view novel view synthesis, including both pixel-aligned and enhanced pixel-aligned Gaussian splatting approaches. Pixel-aligned methods predict Gaussian parameters on a per-pixel basis in image space before unprojecting to 3D. These include pixelSplat [7], MVSplat [9], FreeSplat [12], TranSplat [61] and DepthSplat [10]. Gaussian Graph Network (GGN) [40], AnySplat [41] and WorldMirror [52] refines the pixel-aligned approach by modeling the relationships between groups of predicted Gaussians across different views while building upon it. In contrast to both pixel-aligned and enhanced pixel-aligned methods, our VolSplat employs a voxel-aligned approach, predicting Gaussian primitives within a 3D voxel grid. This method aggregates multi-view evidence in 3D space, aligning Gaussian predictions to a voxel structure, which facilitates better geometric consistency and efficient redundancy reduction.

Metrics. For quantitative evaluation, we adopt standard image quality metrics commonly used in NVS, including pixel-level PSNR, patch-level SSIM [62],and feature-level LPIPS [63].

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

| |
|---|

| |
|---|

|x|
|---|

[Figure 351]

[Figure 352]

| |
|---|

[Figure 353]

| |
|---|

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

| |
|---|

| |
|---|

| |
|---|

[Figure 364]

[Figure 365]

| |
|---|

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

| |
|---|

| |
|---|

| |
|---|

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

| |
|---|

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

| |
|---|

| |
|---|

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

| |
|---|

| |
|---|

[Figure 406]

[Figure 407]

Inputs AnySplat WorldMirror VolSplat Ground Truth

DepthSplat

GGN

MVSplat

- Figure 4 Qualitative comparison on RealEstate10K [19]. We compare VolSplat against SOTA pixel-aligned baselines under sparse-view inputs. While competing methods often suffer from blurring and geometric distortions in complex environments, VolSplat leverages its voxel-aligned prediction to maintain superior visual fidelity and structural consistency.

Implementation details. We implement VolSplat using PyTorch [64] and optimize the model with the AdamW [65] optimizer and a cosine learning rate schedule. The monocular Vision Transformer backbone is implemented using the xFormers [66] library. For the pre-trained Depth Anything V2 [67] backbone, we use a lower learning rate of 2 × 10−6, while other layers are trained with a learning rate of 2 × 10−4 following DepthSplat [10]. For experiments on the RealEstate10K [19] and ScanNet [20] dataset, we train the model for 150,000 iterations using 4× NVIDIA H20 GPUs with a total batch size of 4. Following the setting of the baseline, we use 256 × 256 as input resolution. In the training stage, the number of input views is set to 6, and we evaluate the model’s performance with same numbers of input views. We will make our codes and pre-trained models publicly available.

#### 4.2 Experimental Results and Analysis

Comparisons with SOTA models. As shown in Tab. 1 and Tab. 2, we report VolSplat’s performance compared to current mainstream pixel-aligned models [7, 9, 10, 12, 61] and their variants [40, 41, 52]. On both the RealEstate10K [19] and ScanNet [20] datasets, VolSplat achieves SOTA results. Notably, since AnySplat [41] and WorldMirror [52] are primarily designed for joint pose-geometry optimization in pose-free settings, they fail to effectively utilize the provided ground truth camera poses, resulting in lower performance compared to other baselines. Our experiments reveal a critical distinction between pixel-aligned and voxel-aligned paradigms. A key observation is that under sparse multi-view settings, all pixel-aligned models exhibit a significant degradation in performance. In contrast, VolSplat demonstrates promising performance to these challenging

[Figure 408]

| |
|---|

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

| |
|---|

| |
|---|

[Figure 415]

[Figure 416]

| |
|---|

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

| |
|---|

[Figure 424]

[Figure 425]

| |
|---|

| |
|---|

| |
|---|

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

| |
|---|

| |
|---|

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

| |
|---|

[Figure 442]

[Figure 443]

Inputs FreeSplat

AnySplat WorldMirror VolSplat Ground Truth

- Figure 5 Qualitative comparison on ScanNet [20]. Compared to recent baseline methods (FreeSplat [12], AnySplat [41], and WorldMirror [52]), VolSplat significantly reduces common floaters and visual artifacts. Our method produces cleaner object boundaries and a more coherent 3D scene reconstruction.

Inputs MVSplat TranSplat DepthSplat GGN VolSplat Ground Truth

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

- Figure 6 Qualitative results on ACID [21]. Despite being trained solely on RealEstate10K [19], VolSplat generalizes exceptionally well to outdoor scenes. Our method reconstructs fine-grained details in natural landscapes and structures, producing photorealistic novel views where baseline methods often fail to maintain visual coherence.

conditions. As illustrated in Fig. 4 and Fig. 5, images rendered by our method are largely free of the common floaters and artifacts that plague competing methods at object boundaries. This visual improvement stems directly from the ability of our model to resolve multi-view alignment issues within its 3D feature representation, resulting in cleaner edges and a more coherent 3D scene reconstruction.

Cross-dataset generalization. We assess the generalization capabilities of our model on unseen outdoor datasets to verify its broad reliability. To this end, we conducted a cross-dataset generalization experiment by taking our model pre-trained on the RealEstate10K [19] dataset and evaluating it directly on the ACID [21] dataset without any fine-tuning. As demonstrated in Tab. 3

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

| |
|---|

| |
|---|

| |
|---|

| |
|---|

DepthSplat

VolSplat (Ours)

- Figure 7 Visualization of Gaussians and density maps. We compare the rendered results and the spatial distribution of Gaussian centers between DepthSplat [10] and VolSplat. DepthSplat is constrained by the pixel grid, resulting in a uniform but redundant distribution regardless of scene content. In contrast, VolSplat adaptively concentrates Gaussians on complex geometric structures (e.g., the washbasin boundaries) while remaining sparse in flat or empty regions, demonstrating a much more efficient and geometry-aware representation.

- Table 2 Quantitative comparisons on ScanNet [20] (6 input views). We compare VolSplat against SOTA methods that rely on post-hoc 3D Gaussian fusion strategies. VolSplat outperforms all baselines.

Method PSNR ↑ SSIM ↑ LPIPS ↓

FreeSplat [12] 27.45 0.829 0.222 FreeSplat++ [68] 27.45 0.829 0.223 AnySplat [41] 19.45 0.626 0.344 WorldMirror [52] 25.83 0.819 0.136

VolSplat 28.41 0.906 0.127

Table 3 Cross-dataset generalization on ACID [21] (6 input views). Models trained on RealEstate10K [19] (indoor scenes) are directly evaluated on ACID [21] (outdoor scenes) without fine-tuning.

Method PSNR ↑ SSIM ↑ LPIPS ↓

MVSplat [9] 28.15 0.841 0.147 TranSplat [61] 28.17 0.842 0.146 DepthSplat [10] 28.37 0.847 0.141 GGN [40] 26.97 0.814 0.196

VolSplat 32.65 0.932 0.092

and Fig. 6, VolSplat maintains significantly higher performance in this zero-shot transfer setting. We attribute this superior generalization to the inherent robustness of our voxel-aligned framework. Pixel-aligned models exhibit a much higher sensitivity to the variations in data complexity and distribution between different datasets. In contrast, VolSplat is less susceptible to these domain shifts.

Analysis of Gaussian density. A fundamental principle of 3D reconstruction is that the complexity of the representation should adapt to the complexity of the scene. Real-world environments contain a mix of simple, planar surfaces and intricate, high-frequency geometric details. An ideal model should allocate its descriptive capacity accordingly. However, pixel-aligned methods are inherently limited in this regard. Their paradigm of predicting one Gaussian per pixel results in a fixed number of primitives, predetermined by the input image resolution (e.g., H × W Gaussians from a reference view), regardless of whether the scene is a simple room or a complex outdoor environment.

In stark contrast, our voxel-aligned framework enables adaptive control over the density of the

- 3D Gaussians. By predicting primitives based on the occupancy of 3D voxel features, VolSplat naturally allocates a higher concentration of Gaussians to regions of high geometric detail while using a sparser representation for simple or empty spaces.

This adaptive capability is quantitatively validated by the results we reported in Tab. 1, Tab. 2 and Tab. 3. Here, we analyze these findings in greater detail. The data shows that pixel-aligned methods consistently generate constant density of Gaussians, irrespective of the scene content. This leads to significant redundancy, as well as an insufficient representational capacity in areas with intricate details. Conversely, Gaussians of VolSplat demonstrate significant variance across different regions, confirming its ability to tailor complexity of the scene, as shown in Fig. 7. Notably, VolSplat often achieves superior rendering quality with a non-uniform set of Gaussians

- Table 4 Analysis of voxel size. “PGS” stands for “average number of per-view Gaussians”. We investigate the impact of voxel resolution on reconstruction quality and efficiency. A voxel size of 0.1 yields the optimal trade-off, achieving the best performance. Notably, further reducing the voxel size to 0.05 degrades performance due to the loss of coherent spatial context in overly sparse grids, while larger voxels fail to capture fine geometric details.

Voxel Size (cm) PSNR ↑ SSIM ↑ LPIPS ↓ PGS Memory(GB) Inference Time(s)

0.05 29.34 0.919 0.092 65415 9.19 0.802 0.1 (default) 29.40 0.928 0.085 60523 9.04 0.768 0.5 27.33 0.899 0.108 59788 8.98 0.744 1 20.78 0.602 0.323 51806 8.74 0.739

- Table 5 Ablation of sparse 3D decoder. “w/ 3D CNN” means replacing the 3D U-Net with a sparse 3D CNN, “w/o residual” means predicting refined voxel feature without residual design, and “w/o decoder” means removing the refinement stage. We validate the necessity of our specific refinement module. Replacing our proposed sparse 3D decoder for feature refinement results in performance drop. This demonstrates that our specific design is essential for refining the voxel features and producing high-quality 3D Gaussians.

Components PSNR ↑ SSIM ↑ LPIPS ↓ Memory(GB) Inference Time(s)

default 29.40 0.928 0.085 9.04 0.768 w/ 3D CNN 28.01 0.919 0.098 9.03 0.705 w/o residual 27.92 0.908 0.101 9.04 0.765 w/o decoder 27.47 0.901 0.102 8.99 0.687

compared to the brute-force density of pixel-aligned approaches.

#### 4.3 Ablation Study

In this section, we study the properties of our key components with 12 input views on the RealEstate10K [19] dataset.

Ablation of Voxel Size. The voxel size is a critical hyperparameter in our framework, as it dictates the resolution of the 3D feature grid. This choice involves a fundamental trade-off between the fidelity of the geometric representation and computational resource consumption. In Tab. 4, we analyze this trade-off by comparing our default setting against configurations with different voxels.

Using a small voxel size increases the granularity of the 3D grid, allowing the model to capture finer geometric details. It comes at a significant cost, substantially increasing memory usage and processing time due to the cubic growth of the voxel volume. Conversely, employing a large voxel size reduces the computational footprint but results in a coarser quantization of the 3D space. Our default configuration strikes an effective balance, achieving SOTA performance while maintaining manageable computational requirements.

Ablation of Model Architecture. Directly predicting Gaussians from the initial unprojected 3D features is less effective, particularly for challenging scenes with complex geometry or sparse viewpoints. To address this, we incorporate a 3D U-Net architecture to refine and enhance this raw feature volume, predicting the residual features. To validate the necessity and efficacy of this design, we conduct an ablation study with three variants: 1) removing the refinement module entirely, 2)directly predicting the refined feature, and 3) replacing the 3D U-Net with a standard 3D CNN.

The results, presented in Tab. 5, confirm our architectural choices. Removing the refinement stage altogether leads to a significant drop in performance, demonstrating that processing the initial voxel features is critical for producing a coherent 3D representation. While substituting our module with a sparse 3D CNN or removing the residual design yields better results than no refinement, it still falls short of the performance of our full model. The multi-scale feature fusion

inherent in the U-Net structure and residual design are crucial for capturing both fine-grained local details and broader spatial context.

### 5 Conclusion

We address the fundamental limitations inherent in the prevailing pixel-aligned paradigm for feed-forward 3D Gaussian Splatting. We identify that existing methods suffer from a rigid coupling of Gaussian density to input image resolution and a high sensitivity to multi-view alignment errors. To overcome these challenges, we introduce VolSplat, a novel framework that fundamentally shifts the reconstruction process from 2D pixels to a 3D voxel-aligned space. By constructing 3D voxel feature and predicting Gaussians directly from this unified representation, our method effectively decouples the 3D scene from the constraints of the input views. This voxel-centric design enables adaptive control over Gaussian density according to scene complexity and inherently resolves alignment ambiguities, leading to more geometrically consistent and faithful reconstructions for downstream tasks.

### References

- [1] Haoxiao Wang, Kaichen Zhou, Binrui Gu, Zhiyuan Feng, Weijie Wang, Peilin Sun, Yicheng Xiao, Jianhua Zhang, and Hao Dong. Transdiff: Diffusion-based method for manipulating transparent objects using a single rgb-d image. In IEEE Int. Conf. Robot. Autom., pages 7277–7283. IEEE, 2025.
- [2] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.
- [3] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.
- [4] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4578–4587, 2021.
- [5] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4690–4699, 2021.
- [6] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Int. Conf. Comput. Vis., pages 14124–14133, 2021.
- [7] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In IEEE Conf. Comput. Vis. Pattern Recog., pages 19457–19467, 2024.
- [8] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In Eur. Conf. Comput. Vis., pages 1–19. Springer, 2024.
- [9] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In Eur. Conf. Comput. Vis., pages 370–386. Springer, 2024.

- [10] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In IEEE Conf. Comput. Vis. Pattern Recog., pages 16453–16463, 2025.
- [11] Weijie Wang, Donny Y Chen, Zeyu Zhang, Duochao Shi, Akide Liu, and Bohan Zhuang. Zpressor: Bottleneck-aware compression for scalable feed-forward 3dgs. arXiv preprint arXiv:2505.23734, 2025.
- [12] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. Adv. Neural Inform. Process. Syst., 37:107326–107349, 2024.
- [13] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. Mvsplat360: Feed-forward 360 scene synthesis from sparse views. Adv. Neural Inform. Process. Syst., 37:107064–107086, 2024.
- [14] Gyeongjin Kang, Seungtae Nam, Xiangyu Sun, Sameh Khamis, Abdelrahman Mohamed, and Eunbyung Park. ilrm: An iterative large 3d reconstruction model. arXiv preprint arXiv:2507.23277, 2025.
- [15] Weijie Wang, Jiagang Zhu, Zeyu Zhang, Xiaofeng Wang, Zheng Zhu, Guosheng Zhao, Chaojun Ni, Haoxiao Wang, Guan Huang, Xinze Chen, et al. Drivegen3d: Boosting feed-forward driving scene generation with efficient video diffusion. arXiv preprint arXiv:2510.15264, 2025.
- [16] Weijie Wang, Qihang Cao, Sensen Gao, Donny Y Chen, Haofei Xu, Wenjing Bian, Songyou Peng, Tat-Jen Cham, Chuanxia Zheng, Andreas Geiger, et al. Feed-forward 3d scene modeling: A problem-driven perspective. arXiv preprint arXiv:2604.14025, 2026.
- [17] Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3d u-net: learning dense volumetric segmentation from sparse annotation. In International conference on medical image computing and computer-assisted intervention, pages 424–432. Springer, 2016.
- [18] Duochao Shi, Weijie Wang, Donny Y. Chen, Zeyu Zhang, Jiawang Bian, Bohan Zhuang, and Chunhua Shen. Revisiting depth representations for feed-forward 3d gaussian splatting. arXiv preprint arXiv:2506.05327, 2025.
- [19] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.
- [20] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5828–5839, 2017.
- [21] Andrew Liu, Richard Tucker, Varun Jampani, Ameesh Makadia, Noah Snavely, and Angjoo Kanazawa. Infinite nature: Perpetual view generation of natural scenes from a single image. In Int. Conf. Comput. Vis., pages 14458–14467, 2021.
- [22] Paul E. Debevec, Camillo J. Taylor, and Jitendra Malik. Modeling and rendering architecture from photographs: a hybrid geometry- and image-based approach. In Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’96, pages 11–20, New York, NY, USA, 1996. Association for Computing Machinery. ISBN

0897917464. doi: 10.1145/237170.237191.

- [23] Dinghuang Ji, Junghyun Kwon, Max McFarland, and Silvio Savarese. Deep view morphing. In IEEE Conf. Comput. Vis. Pattern Recog., pages 2155–2163, 2017.
- [24] Marc Levoy and Pat Hanrahan. Light field rendering. In Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH ’96, pages 31–42, New York, NY, USA, 1996. Association for Computing Machinery. ISBN 0897917464. doi: 10.1145/237170.237199.
- [25] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo MartinBrualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Int. Conf. Comput. Vis., pages 5855–5864, 2021.
- [26] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Int. Conf. Comput. Vis., pages 19697–19705, 2023.
- [27] Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Adv. Neural Inform. Process. Syst., 37:140138–140158, 2024.
- [28] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In Eur. Conf. Comput. Vis., pages 145–163. Springer,

- 2024.

[29] Hengyu Liu, Yuehao Wang, Chenxin Li, Ruisi Cai, Kevin Wang, Wuyang Li, Pavlo Molchanov, Peihao Wang, and Zhangyang Wang. Flexgs: Train once, deploy everywhere with many-in-one flexible 3d gaussian splatting. In IEEE Conf. Comput. Vis. Pattern Recog., pages 16336–16345,

- 2025.

- [30] Donald Meagher. Geometric modeling using octree encoding. Computer graphics and image processing, 19(2):129–147, 1982.
- [31] Arie E Kaufman and Klaus Mueller. Overview of volume rendering. The visualization handbook, 7:127–174, 2005.
- [32] Chamin Hewa Koneputugodage, Yizhak Ben-Shabat, and Stephen Gould. Octree guided unoriented surface reconstruction. In IEEE Conf. Comput. Vis. Pattern Recog., pages 16717–16726, 2023.
- [33] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4490–4499, 2018.
- [34] Gernot Riegler, Ali Osman Ulusoy, and Andreas Geiger. Octnet: Learning deep 3d representations at high resolutions. In IEEE Conf. Comput. Vis. Pattern Recog., pages 3577–3586, 2017.
- [35] Haoxiao Wang, Antao Xiang, Haiyang Sun, Peilin Sun, Changhao Pan, Yifu Chen, Minjie Hong, Weijie Wang, Shuang Chen, Yue Chen, et al. Diffusion model as a generalist segmentation learner. arXiv preprint arXiv:2604.24575, 2026.
- [36] Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5501–5510, 2022.

- [37] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In IEEE Conf. Comput. Vis. Pattern Recog., pages 12479–12488, 2023.
- [38] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In IEEE Conf. Comput. Vis. Pattern Recog., pages 20654–20664, 2024.
- [39] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024.
- [40] Shengjun Zhang, Xin Fei, Fangfu Liu, Haixu Song, and Yueqi Duan. Gaussian graph network: Learning efficient and generalizable gaussian representations from multi-view images. Adv. Neural Inform. Process. Syst., 37:50361–50380, 2024.
- [41] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, Dahua Lin, and Bo Dai. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025.
- [42] Chaojun Ni, Xiaofeng Wang, Zheng Zhu, Weijie Wang, Haoyun Li, Guosheng Zhao, Jie Li, Wenkang Qin, Guan Huang, and Wenjun Mei. Wonderturbo: Generating interactive 3d world in 0.72 seconds. In Int. Conf. Comput. Vis., pages 27423–27434, 2025.
- [43] Ranran Huang and Krystian Mikolajczyk. No pose at all: Self-supervised pose-free 3d gaussian splatting from sparse views. arXiv preprint arXiv: 2508.01171, 2025.
- [44] Weijie Wang, Zimu Li, Jinchuan Shi, Zeyu Zhang, Botao Ye, Marc Pollefeys, Donny Y Chen, and Bohan Zhuang. Trisplat: Simulation-ready feed-forward 3d scene reconstruction. arXiv preprint arXiv:2605.26115, 2026.
- [45] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024.
- [46] Gyeongjin Kang, Jisang Yoo, Jihyeon Park, Seungtae Nam, Hyeonsoo Im, Sangheon Shin, Sangpil Kim, and Eunbyung Park. Selfsplat: Pose-free and 3d prior-free generalizable 3d gaussian splatting. In IEEE Conf. Comput. Vis. Pattern Recog., pages 22012–22022, 2025.
- [47] Cheng Zhang, Hanwen Liang, Donny Y Chen, Qianyi Wu, Konstantinos N Plataniotis, Camilo Cruz Gambardella, and Jianfei Cai. Panflow: Decoupled motion control for panoramic video generation. In AAAI, volume 40, pages 12385–12393, 2026.
- [48] Weijie Wang, Xiaoxuan He, Youping Gu, Yifan Yang, Zeyu Zhang, Yefei He, Yanbo Ding, Xirui Hu, Donny Y Chen, Zhiyuan He, et al. World-r1: Reinforcing 3d constraints for text-to-video generation. arXiv preprint arXiv:2604.24764, 2026.
- [49] Weijie Wang, Haoyu Zhao, Yifan Yang, Feng Chen, Zeyu Zhang, Yefei He, Zicheng Duan, Donny Y Chen, Yuqing Yang, and Bohan Zhuang. Latent spatial memory for video world models. arXiv preprint arXiv:2606.09828, 2026.
- [50] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint arXiv:2410.12781, 2024.

- [51] Yiming Wang, Lucy Chai, Xuan Luo, Michael Niemeyer, Manuel Lagunas, Stephen Lombardi, Siyu Tang, and Tiancheng Sun. Learning efficient fuse-and-refine for feed-forward 3d gaussian splatting. arXiv preprint arXiv:2503.14698, 2025.
- [52] Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan, Yawei Luo, and Chunchao Guo. Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726, 2025.
- [53] Sheng Miao, Jiaxin Huang, Dongfeng Bai, Xu Yan, Hongyu Zhou, Yue Wang, Bingbing Liu, Andreas Geiger, and Yiyi Liao. Evolsplat: Efficient volume-based gaussian splatting for urban view synthesis. In IEEE Conf. Comput. Vis. Pattern Recog., pages 11286–11296, 2025.
- [54] Augustus Odena, Vincent Dumoulin, and Chris Olah. Deconvolution and checkerboard artifacts. Distill, 2016. doi: 10.23915/distill.00003. URL http://distill.pub/2016/ deconv-checkerboard.
- [55] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In IEEE Conf. Comput. Vis. Pattern Recog., pages 770–778, 2016.
- [56] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Int. Conf. Comput. Vis., pages 10012–10022, 2021.
- [57] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying flow, stereo and depth estimation. IEEE Trans. Pattern Anal. Mach. Intell., 45(11):13941–13958, 2023.
- [58] Tai Wang, Xiaohan Mao, Chenming Zhu, Runsen Xu, Ruiyuan Lyu, Peisen Li, Xiao Chen, Wenwei Zhang, Kai Chen, Tianfan Xue, et al. Embodiedscan: A holistic multi-modal 3d perception suite towards embodied ai. In IEEE Conf. Comput. Vis. Pattern Recog., pages 19757–19767, 2024.
- [59] Xiaoshuai Zhang, Sai Bi, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Nerfusion: Fusing radiance fields for large-scale scene reconstruction. In IEEE Conf. Comput. Vis. Pattern Recog., pages 5449–5458, 2022.
- [60] Yiming Gao, Yan-Pei Cao, and Ying Shan. Surfelnerf: Neural surfel radiance fields for online photorealistic reconstruction of indoor scenes. In IEEE Conf. Comput. Vis. Pattern Recog., pages 108–118, 2023.
- [61] Chuanrui Zhang, Yingshuang Zou, Zhuoling Li, Minmin Yi, and Haoqian Wang. Transplat: Generalizable 3d gaussian splatting from sparse multi-view images with transformers. In AAAI, pages 9869–9877, 2025.
- [62] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Trans. Image Process., 13(4):600–612, 2004.
- [63] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conf. Comput. Vis. Pattern Recog., pages 586–595, 2018.
- [64] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Adv. Neural Inform. Process. Syst., 32, 2019.

- [65] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [66] Benjamin Lefaudeux, Francisco Massa, Diana Liskovich, Wenhan Xiong, Vittorio Caggiano, Sean Naren, Min Xu, Jieru Hu, Marta Tintore, Susan Zhang, Patrick Labatut, Daniel Haziza, Luca Wehrstedt, Jeremy Reizenstein, and Grigory Sizov. xformers: A modular and hackable transformer modelling library. https://github.com/facebookresearch/xformers, 2022.
- [67] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024.
- [68] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat++: Generalizable 3d gaussian splatting for efficient indoor scene reconstruction. arXiv preprint arXiv:2503.22986, 2025.
- [69] Christopher Choy, JunYoung Gwak, and Silvio Savarese. 4d spatio-temporal convnets: Minkowski convolutional neural networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 3075–3084, 2019.

### A More Implementation Details

Network architecture. Our framework begins with a 2D feature extraction stage. Following DepthSplat [10], we employ a weight-sharing ResNet [55] backbone to extract multi-scale feature maps from each input view. To enhance multi-view consistency, these features are refined via a cross-view interaction module implemented with local window attention [56], which efficiently aggregates information from neighboring views. Following feature extraction, the architecture proceeds to the multi-view depth prediction module. This module constructs a cost volume using a plane-sweep strategy with 128 inverse-depth candidates per reference view. By performing local neighbor matching on the cost volume, the network predicts robust depth maps that serve as the geometric basis for lifting 2D features into 3D space. The lifted 3D representation is built in world coordinates and processed as a sparse voxel set. Specifically, we employ a sparse data structure where only occupied cells are materialized, avoiding the computational redundancy of a dense global grid. This sparse 3D refinement is efficiently implemented using MinkowskiEngine [69]. In the default configuration, each occupied voxel predicts one Gaussian primitive with a 38-dimensional parameter vector, including opacity, center offset, anisotropic scale, quaternion rotation, and degree-2 spherical-harmonic color coefficients. This design ensures that primitive allocation remains scene-adaptive while preserving stable sparse 3D decoding.

More training details. Optimization uses AdamW [65] with decoupled weight decay, together with two learning-rate groups for the pretrained monocular branch and the remaining trainable parameters, and radient clipping is enabled for stability. During training, each sample uses 6 input views and 8 target views. Two input views are first selected as boundary anchors with a randomly sampled frame gap in a predefined range, and the remaining input views are sampled between these anchors. The anchor-gap range is progressively expanded during early iterations. All experiments are run at 256 × 256 input resolution. Training is performed on RealEstate10K [19] and then fine-tuned on ScanNet [20] from the RealEstate10K checkpoint, while ACID [21] is evaluated in a zero-shot manner. For details on training objectives and weights, please refer to Sec. 3.4.

Evaluation. Evaluation follows a controlled protocol for fair comparison with prior feed-forward 3DGS methods [7, 9, 10, 40, 41, 52, 61]. For each scene, input views are selected using fixed frame-gap rules, and target novel views are chosen from disjoint camera positions that are not included in the inputs. In our setup, each sample is evaluated on 8 target novel views.

Open-source. Our source codes are provided in the supplementary material. We will open-source the complete codebase for VolSplat.

### B More Experimental Analysis

Efficiency Analysis. We evaluate the computational efficiency of our method against state-of-theart baselines on RealEstate10K [19] as illustrated in Tab. A. All metrics utilize a single NVIDIA H20 GPU with 6 input views to ensure same evaluation environment. VolSplat achieves a superior balance between reconstruction quality and resource consumption despite incorporating explicit 3D feature processing. Our approach outperforms all competing approaches while maintaining a competitive inference latency of 0.575s. This runtime remains comparable to leading pixel-aligned approaches [10], which demonstrates that our voxel-aligned architecture introduces negligible overhead relative to the significant performance gains. Furthermore, VolSplat exhibits competitive memory consumption by requiring only 4.65GB of VRAM. This footprint is substantially lower than heavy-weight baselines [7, 52]. Our method remains on par with lightweight alternatives such as MVSplat and ensures practicality for deployment on consumer-grade hardware. While absolute runtime values may vary due to hardware discrepancies compared to original publications, the

Table A Efficiency comparison. All methods are evaluated with 6 input views on RealEstate10K [19] via a single NVIDIA H20 GPU. Our VolSplat tops all image-quality metrics while retaining competitive inference efficiency even though utilizing 3D features. Note that absolute runtimes may differ from original studies due to hardware discrepancies, but relative rankings ensure fair efficiency comparison.

Method PSNR↑ SSIM↑ LPIPS↓ Memory (GB) Infer time (s)

pixelSplat 28.95 0.900 0.163 36.82 0.579 MVSplat 29.13 0.924 0.091 4.70 0.369 TranSplat 29.62 0.928 0.084 3.96 1.002 DepthSplat 30.52 0.931 0.079 8.00 0.513 GGN 26.68 0.879 0.133 4.70 0.377 AnySplat 19.05 0.576 0.305 3.57 0.332 World-Mirror 24.86 0.819 0.079 8.05 0.375 Ours 31.30 0.941 0.075 4.65 0.575

relative rankings presented here reflect a controlled and fair comparison.

### C Limitation and Societal Impacts

Limitation analysis. Our current framework assumes a static scene assumption during the multi-view feature aggregation and voxel construction steps. Consequently, VolSplat struggle to reconstruct dynamic objects or changing environments, as the geometric consistency enforced by our cost volume and sparse 3D U-Net [17] relies on multi-view consistency. Moving elements in the scene can lead to artifacts such as ghosting or blurring in the rendered novel views.

Potential and negative societal impacts. VolSplat significantly advances the capability of feedforward 3D reconstruction by decoupling geometry from input pixel resolution. It enables the generation of high-fidelity 3D assets from sparse multi-view images with superior geometric consistency, positioning VolSplat as a valuable tool for immersive applications such as virtual reality, gaming, and digital twin creation.

While the ability to produce photorealistic 3D reconstructions from limited data is beneficial for content creation, it is important to acknowledge the potential for misuse. The high fidelity of the generated scenes could be exploited to create deepfakes or unauthorized digital replicas of private spaces. Consequently, the deployment of VolSplat in sensitive contexts should be accompanied by robust watermarking techniques or authentication protocols to mitigate the risks associated with the synthesis of misleading or non-consensual 3D content.

### D More Visual Comparisons

This section provides additional qualitative comparison results. We present further visualizations for VolSplat on the RealEstate10K [19] dataset, comparing against SOTA baselines [7, 9, 10, 40, 41, 52, 61]. To illustrate how VolSplat performs with varying numbers of input views, we showcase comparative results across different settings. For the standard setting, comparisons with 6 input views are presented in Fig. A. To demonstrate scalability to denser inputs, visual comparisons between our method and competing baselines are displayed for scenarios with 12 and 24 input views in Fig. B and Fig. C, respectively. The corresponding quantitative results for these multi-view experiments can be found in Tab. 1.

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

Inputs MVSplat DepthSplat GGN AnySplat WorldMirror VolSplat Ground Truth

- Figure A More qualitative comparisons on RealEstate10K [19] under 6 input views.

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

- Inputs MVSplat DepthSplat GGN AnySplat WorldMirror VolSplat Ground Truth
- Figure B More qualitative comparisons on RealEstate10K [19] under 12 input views.

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

- Inputs MVSplat DepthSplat GGN AnySplat WorldMirror VolSplat Ground Truth
- Figure C More qualitative comparisons on RealEstate10K [19] under 24 input views.

