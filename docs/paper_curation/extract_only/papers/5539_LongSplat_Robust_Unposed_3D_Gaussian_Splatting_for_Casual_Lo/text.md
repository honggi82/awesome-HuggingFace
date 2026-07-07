## LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos

# arXiv:2508.14041v1[cs.CV]19Aug2025

Chin-Yang Lin1 Cheng Sun2 Fu-En Yang2 Min-Hung Chen2 Yen-Yu Lin1 Yu-Lun Liu1

1National Yang Ming Chiao Tung University 2NVIDIA Research

Input: casually-captured long video without camera poses

[Figure 1]

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

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Output: high-quality novel view synthesis from jointly optimized camera poses and 3D Gaussian Splatting

Figure 1. LongSplat achieves robust novel view synthesis from casually captured long videos without provided camera poses. Our approach jointly optimizes camera poses and 3D Gaussian Splatting, producing accurate and visually coherent reconstructions even under challenging conditions.

#### Abstract

LongSplat addresses critical challenges in novel view synthesis (NVS) from casually captured long videos characterized by irregular camera motion, unknown camera poses, and expansive scenes. Current methods often suffer from pose drift, inaccurate geometry initialization, and severe memory limitations. To address these issues, we introduce LongSplat, a robust unposed 3D Gaussian Splatting framework featuring: (1) Incremental Joint Optimization that concurrently optimizes camera poses and 3D Gaussians to avoid local minima and ensure global consistency; (2) a robust Pose Estimation Module leveraging learned 3D priors; and (3) an efficient Octree Anchor Formation mechanism that converts dense point clouds into anchors based on spatial density. Extensive experiments on challenging benchmarks demonstrate that LongSplat achieves state-of-the-art results, substantially improving rendering quality, pose accuracy, and computational efficiency compared to prior approaches. Project page: https://linjohnss.github.io/longsplat/

#### 1. Introduction

High-quality 3D reconstruction and novel view synthesis (NVS) are essential for applications such as virtual reality, augmented reality, virtual tourism, and cultural heritage preservation. They also play a crucial role in video editing tasks like stabilization, visual effects, and digital mapping for real estate or pedestrian-level navigation. With the widespread availability of smartphones and action cameras, casually captured videos have emerged as a significant source of 3D content. Unlike professionally acquired datasets, casual videos present challenging characteristics: irregular camera trajectories, long sequences spanning hundreds or thousands of frames, and the absence of reliable camera poses or precise geometric priors.

Addressing novel view synthesis (NVS) from casually captured videos poses two critical challenges: robust camera pose estimation over extended trajectories and efficient representation of large-scale scenes. Traditional methods rely on accurate poses from Structure-from-Motion (SfM) preprocessing, yet as shown in Fig. 2, pipelines like COLMAP [51]

frequently fail in casual settings. COLMAP-free methods, such as CF-3DGS [14], often encounter severe memory constraints, limiting their effectiveness for large-scale scenarios. Similarly, methods like LocalRF [39] struggle with complex camera trajectories, resulting in fragmented reconstructions. Foundation models like MASt3R [27] provide fast initial estimates but suffer inaccuracies and drift in long videos, severely affecting reconstruction quality.

To address these limitations, we introduce LongSplat, a robust unposed 3D Gaussian Splatting (3DGS) [22] framework designed specifically for casual long videos. As illustrated in Fig. 1, LongSplat achieves accurate novel view synthesis without relying on provided camera poses. LongSplat departs from traditional pipelines by jointly optimizing camera poses and 3DGS in a unified framework. It integrates a correspondence-guided Pose Estimation Module with 3DGS geometry and photometric refinements to improve pose accuracy, even under large-scale and unstructured camera motion. Furthermore, an efficient Octree Anchor Formation mechanism converts dense point clouds into anchors based on spatial density, significantly reducing memory usage while retaining detailed scene structures. These components work together in an incremental joint optimization strategy that avoids local minima and ensures global geometric consistency across extensive sequences.

Extensive experiments on challenging datasets, including Tanks and Temples, Free, and Hike datasets, demonstrate that LongSplat consistently outperforms existing approaches, significantly improving rendering quality and pose accuracy. Compared to conventional methods shown in Fig. 2, LongSplat produces clearer and more coherent reconstructions, effectively addressing pose drift and memory limitations and substantially advancing the state-of-the-art. The main contributions of our work are:

- • An incremental joint optimization approach for simultaneous camera pose and 3DGS reconstruction, reducing local minima and ensuring global consistency.
- • A robust pose estimation module leveraging learned 3D priors for accurate camera pose estimation.
- • An adaptive Octree Anchor Formation strategy that significantly reduces memory usage while preserving reconstruction quality.

#### 2. Related Work

Novel View Synthesis. Novel View Synthesis (NVS) generates new perspectives from captured images, evolving from early pixel interpolation methods [8] to depth-based warping techniques [28] and 3D reconstruction-based rendering [6, 12]. Various representations have been explored, including planes [17, 18], meshes [20, 48, 49], point clouds [68, 75], and multi-plane images [29, 58, 77]. Neural Radiance Fields (NeRF) [40] revolutionized photorealistic rendering, with subsequent improvements in anti-

|Fails|
|---|

|OOM|
|---|

[Figure 28]

[Figure 29]

COLMAP LocalRF

CF-3DGS

MASt3R

[Figure 30]

| |
|---|

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Robust render quality and camera pose

Figure 2. Novel view synthesis for casual long videos. Existing methods encounter significant challenges when reconstructing scenes from casually captured long videos: COLMAP [51] fails due to incorrect camera pose estimation, CF-3DGS [14] suffers from out-of-memory issues, LocalRF [39] struggles with complex trajectories, and MASt3R [27]+Scaffold-GS [36] provides inaccurate poses leading to degraded rendering quality. In contrast, LongSplat robustly handles these challenges, yielding accurate camera poses and high-quality novel view synthesis without memory constraints.

aliasing [2–4, 74], reflectance [1, 59], sparse view training [24, 43, 67, 69], faster training [41, 45, 50], and rendering speed [15, 34, 54, 72]. Recent works have extended NeRF to few-shot scenarios without learned priors [32], domain-specific applications such as autonomous driving environments [52], and dynamic scenes with human pose variations [38]. Point-based methods [22, 37, 68, 75], particularly 3D Gaussian Splatting (3DGS) [22], enable real-time rendering through explicit representations. Recent advances have extended 3DGS capabilities to dynamic specular scenes with physically-based rendering [13], developed compression techniques for efficient storage and transmission [73], and improved robustness for unconstrained image scenarios [19]. However, most approaches still rely on pre-computed camera parameters from SfM [16, 32, 42, 51, 56].

Unposed Novel View Synthesis. Recent work has aimed to eliminate camera estimation preprocessing. i-NeRF [71] predicts camera poses using pre-trained NeRF. NeRFmm [65] jointly optimizes NeRF and camera poses for forward-facing scenes, with SiNeRF [66] offering improvements. BARF [31] and GARF [10] address gradient inconsistency through coarse-to-fine positional encoding but require good initialization. Advanced approaches [5, 9, 35, 39] leverage pre-trained networks for geometric priors, with NoPe-NeRF [5] incorporating monocular depth priors and CF-3DGS [14] using progressive optimization. Recent methods have improved robustness in joint optimization of camera poses and scene geometry using decomposed low-rank tensorial representations [7] and dynamic radiance fields [35]. These methods typically assume small pose perturbations [10, 31], limited camera motion [65, 66], or additional priors [5, 11, 21, 39], struggling with challenging trajectories, like Free dataset[26, 44]. Large-scale Novel View Synthesis. Extending NVS to largescale environments introduces memory and computational

(c) Pose Estimation

###### (a) Initialization

###### (b) Global Optimization

###### Loss function

[Figure 35]

Global 3DGS

[Figure 36]

[Figure 37]

|ℒcolor|
|---|

MASt3R

| |
|---|
|[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]|
| |

|[Figure 41]|
|---|

|[Figure 42]|
|---|

[Figure 43]

MASt3R

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Aligned Depth

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

|[Figure 52]|
|---|

|[Figure 53]<br><br>|
|---|

Render RGB GT RGB

[Figure 54]

[Figure 55]

|ℒdepth|
|---|

2D correspondence

[Figure 56]

[Figure 57]

Fallback

[Figure 58]

|PnP + RANSAC<br><br>Pose refinement<br><br>Anchor unprojection<br><br>Failure / Success|
|---|

(d) Local Optimization Local 3DGS

[Figure 59]

Rendered Depth Aligned Depth

[Figure 60]

| | | |
|---|---|---|
|[Figure 61]<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]| | |
|Visibility adapted local window| | |

| |
|---|

|ℒreprojection|
|---|

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

2D

Projected Correspondence

Correspondence

|Octree Anchor Formation|
|---|

- Figure 3. Overview of the LongSplat framework. Given a casually captured long video without known poses, LongSplat incrementally reconstructs the scene through tightly coupled pose estimation and 3D Gaussian Splatting. (a) Initialization converts MASt3R [27] global aligned point cloud into an octree-anchored 3DGS. (b) Global Optimization jointly refines all camera poses and 3D Gaussians for global consistency. (c) Pose estimation estimates each new frame pose via correspondence-guided PnP, applies photometric refinement, and updates octree anchors using unprojected points. If PnP fails, a fallback triggers global re-optimization to recover. (d) Incremental Optimization alternates between Local Optimization within a visibility-adapted window and periodic Global Optimization to propagate consistent updates across frames. (e) All optimization stages leverage a unified objective composed of photometric loss, depth loss, and reprojection loss to ensure accurate geometry and appearance reconstruction.

challenges that NeRF’s implicit global representation struggles with. Recent research employs scene partitioning strategies for managing large scenes [23, 55, 57]. Progressive optimization techniques have been developed for robust view synthesis in large-scale scenes from casually captured videos [39]. At the same time, MVS-based approaches have been enhanced to handle generalizable view synthesis at scale [53]. For indoor environments, methods like GenRC [30] enable room-scale 3D reconstruction from sparse image collections. 3DGS offers explicit representation advantages through Gaussian primitive. VastGaussian [33] divides scenes into separately optimized blocks[22]. Scaffold-GS [36] introduces anchor-based Gaussian representation with fixed-resolution grids, though it requires SfM initialization. Octree-GS [47] implements fixed-level octrees with preset resolutions but similarly depends on SfM. Unlike these approaches, our method dynamically adjusts voxel size based on point cloud density, without dependency on SfM, and addresses unposed, large-scale, casually captured videos through adaptive Octree Anchor Formation.

Casual Long Videos. Casual long videos present unique challenges: free-moving trajectories, lack of pose information, and continuously expanding scenes. LocalRF [39] addresses these through progressive localized field construction but suffers from slow training and fragmentation under irregular camera movements. 3D Foundation Models [60], including DUSt3R [63], MASt3R [27], Fast3r [70], and CUT3R [62], estimate poses and geometry directly but accumulate errors in long sequences. LongSplat treats foundation model outputs as soft priors, jointly optimizing them with 3D Gaussian Splatting while progressively correcting poses and geometry through combined PnP and optimization strategies.

#### 3. Method

LongSplat reconstructs long video sequences with unknown camera poses and unconstrained trajectories through a fully incremental pipeline based on octree-anchored 3D Gaussian Splatting. The process begins with octree anchor formation, where per-frame dense point clouds are structured into an adaptive representation. Next, camera poses are estimated and refined using correspondence-guided initialization and photometric alignment. Finally, the reconstruction alternates between local optimization, which updates Gaussians within a visibility-adapted window, and global refinement, which ensures long-term consistency. This design allows LongSplat to robustly handle long, unconstrained trajectories while adapting to scene complexity and minimizing drift.

##### 3.1. Preliminaries

Gaussian Splatting. 3D Gaussian Splatting (3DGS) [22] represents the scene as a set of 3D Gaussians, each defined by a center µ ∈ R3, a covariance matrix Σ, color, scale, rotation, and opacity. The covariance is factorized into a rotation R ∈ SO(3) and a diagonal scale matrix S, giving:

G(x) = e−12(x−µ)⊤Σ−1(x−µ), Σ = RSS⊤R⊤. (1)

This parameterization allows each Gaussian to adaptively capture local scene geometry.

To render the scene, each Gaussian is projected onto the image plane using the camera pose W, resulting in a 2D Gaussian with covariance Σ2D = JWΣW⊤J⊤, where J is the Jacobian of the projective transformation. The final rendered color and depth are computed via alpha blending:

| | | |
|---|---|---|
|Point cloud density (𝜌)| | |

| | | | |
|---|---|---|---|
| | | | |
| |Split (𝜌 > Threshold)| | |

| | | | |
|---|---|---|---|
| | | | |
|Remove (𝜌 < Threshold)| | | |

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

[Figure 70]

| |
|---|

Octree anchor

Sparse voxelized

Repeat Levels of time

- Figure 4. Visualization of our proposed Octree Anchor Formation strategy. Given an initial sparse voxelized point cloud, we iteratively perform density-guided adaptive voxel splitting and pruning. Voxels with point cloud density (ρ) exceeding a threshold are split, while those with density below the threshold are pruned. Repeated across multiple octree levels, this adaptive octree anchor design significantly reduces memory usage, allowing efficient representation and rendering of large-scale scenes.

- i−1
- j=1

- i−1
- j=1

N

N

1 − αj , D =

1 − αj ,

C =

ci αi

di αi

i=1

i=1

(2)

where ci and αi denote the color and opacity of the i-th Gaussian, respectively. di denotes the depth value along the ray at the Gaussian’s center.

Anchor-based 3D Gaussian Splatting. To enhance memory efficiency and robustness in large scenes, Scaffold-GS [36] introduces the anchor-based 3DGS representation. Instead of directly maintaining individual Gaussians, the scene is first divided into sparse voxels, each acting as an anchor. From each anchor, k Gaussians are initialized with positions relative to the anchor center:

{µ0, µ1, . . . , µk−1} = xv + {O0, O1, . . . , Ok−1} · lv, (3)

where xv denotes the anchor position, {Oi} are offset vectors, and lv is a scaling factor. Each Gaussian’s opacity, rotation, scale, and color are decoded from an anchor feature through lightweight MLPs. For opacity, the formulation is:

{α0, α1, . . . , αk−1} = Fα(fˆv, ∆vc, dˆv), (4)

where Fα is an MLP taking the anchor feature fˆv, the relative view distance ∆vc, and the view direction dˆv as inputs.

Anchor Initialization. In traditional Scaffold-GS, initial anchors are derived from sparse SfM point clouds. Points are voxelized to form anchor centers:

p ϵ ⌋ · ϵ, ∀p ∈ P}, (5)

V = {v | v = ⌊

where P is the SfM point cloud and ϵ is the voxel size. Each anchor holds a local feature, managing its associated Gaussians. This design ensures structured densification and pruning, adapting Gaussian density to scene complexity and improving both memory and rendering efficiency.

##### 3.2. Octree Anchor Formation

In large-scale casual video settings, memory efficiency and scene adaptability are essential. Our Octree Anchor Formation dynamically adjusts spatial resolution based on observed geometry, enabling scalable and redundant-free anchor management. LongSplat constructs structured anchors from MASt3R’s per-frame dense point clouds using an adaptive octree (Fig. 3 (a)). Unlike Scaffold-GS, which relies

[Figure 71]

[Figure 72]

keypoint unproject

| |
|---|

unproject

2D-3D correspondence

[Figure 73]

3DGS

|Octree Anchor Formation|
|---|

rasterize

[Figure 74]

[Figure 75]

Aligned Depth Occlusion Mask

[Figure 76]

|[Figure 77]<br><br>|
|---|

|[Figure 78]<br><br>|
|---|

|[Figure 79]|
|---|

|𝑇𝑖−1|
|---|

|𝑇𝑖|
|---|

|𝑇𝑖|
|---|

[Figure 80]

PnP + RANSAC

|𝑇𝑖|
|---|

(a) PnP initialization

(b) Pose refinement

(c) Anchor unprojection

Figure 5. Detailed illustration of our camera pose estimation. (a) PnP initialization: Given correspondences between the predicted 3D anchor points from frame Ti−1 and the 2D keypoints detected in frame Ti we employ PnP with RANSAC to robustly estimate an initial camera pose. (b) Pose refinement: The estimated pose is further refined by rasterizing the 3DGS scene and iteratively minimizing reprojection error to enhance pose accuracy. (c) Anchor unprojection: Newly observed regions are detected via an occlusion mask, computed by forward-warping the previous frame’s rendered depth. These regions are unprojected into 3D and converted into anchors via Octree Anchor Formation.

on a fixed-resolution grid, we progressively subdivide space based on local point density. Each point cloud P = {pi} is voxelized into a sparse grid at resolution ϵ0. Voxels exceeding a density threshold τsplit split into 8 smaller voxels:

- 1

- 2

ϵl. (6)

ϵl+1 =

This process repeats up to a maximum level L. Low-density voxels (density ρv < τprune) are removed to reduce redundancy (Fig. 4).

Each anchor inherits a spatial scale proportional to its voxel size, ensuring coarse anchors for sparsely observed areas and finer anchors for detailed regions:

sv ∝ ϵv. (7)

To further prevent unnecessary duplication, newly generated anchors are compared to existing ones. If significant spatial overlap exists, the new anchor is discarded. This density-adaptive, duplication-free octree formation ensures compact memory usage while preserving adaptive resolution across scenes.

##### 3.3. Pose Estimation module

Accurate and robust camera pose estimation is essential for consistent reconstruction in unposed long video settings. We estimate each pose using 2D-3D correspondences derived from MASt3R, followed by photometric refinement against the current 3D Gaussian scene to maintain coherence across evolving 3D structures (Fig. 3 (c)).

For each new frame t, MASt3R provides 2D correspon-

dences {(xi,x′i)} between frame t and t − 1, allowing backprojection of matched points xi to 3D via:

Xi = Dt−1(xi) · K−1x˜i. (8)

These 2D-3D correspondences {(x′i,Xi)} are used to solve the initial pose Tt via PnP (Fig. 5 (a)), followed by photometric refinement that minimizes (Fig. 5 (b)):

Local window

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

|𝑇𝑖−2|
|---|

|𝑇𝑖|
|---|

|𝑇𝑖|
|---|

|𝑇𝑖|
|---|

|𝑇𝑖−1|
|---|

|𝑇𝑖−1|
|---|

𝑇𝑖−1

(a) IOU < Threshold (b) IOU > Threshold (c) Local optimize

- Figure 6. Illustration of our Visibility-Adapted Local Window strategy for local optimization. To ensure balanced training of the 3D Gaussians, we dynamically define the optimization window based on anchor visibility overlap. Specifically, we compute the Intersection-over-Union (IoU) of visible anchors between consecutive views. Suppose the visibility IoU is below a certain threshold (a). In that case, the local optimization window is adjusted by removing the earliest frame, iteratively repeating until a suitable window with IoU above the threshold is found (b). This approach ensures balanced training coverage and enhances local reconstruction details during optimization (c).

∥It(p) − Iˆt(p)∥2, (9)

Lphoto =

p∈Ω

where It is the observed frame and Iˆt is the rendering using the current 3DGS. This ensures the pose aligns with the evolving scene.

To correct MASt3R’s depth scale drift, we compute a scale factor sˆt by comparing the rendered depth Dt−1 and MASt3R’s aligned depth Dtalign:

sˆt = ⟨Dt−1, Dtalign⟩ ⟨Dtalign, Dtalign⟩

. (10)

This rescaled depth ensures consistent scale across frames.

As the camera moves, newly visible regions are detected via an occlusion mask Mocc, derived by forward-warping Dt−1 to frame t and comparing it to the rescaled depth DtMASt3R (Fig. 5 (c)). Newly visible pixels are unprojected into 3D using:

pi = Dt,MASt3Rui · K−1ui. (11)

These new points are converted into octree anchors using the Octree Anchor Formation described in Sec. 3.2, with overlapping anchors removed to avoid redundancy (Fig. 5 (c)). This process incrementally expands the scene while maintaining structural regularity.

##### 3.4. Incremental Joint Optimization

To handle casually captured long videos, LongSplat adopts a progressive incremental optimization framework that alternates between per-frame local reconstruction and crossframe global consistency refinement.

Initialization. We begin with a small set of initial frames. Camera poses and dense point clouds for these frames are estimated using MASt3R [27], followed by converting the point cloud into an initial octree-anchored 3DGS using the

proposed Octree Anchor Formation (Fig. 3 (a). When camera intrinsics are unavailable, we directly adopt MASt3R’s estimated focal length.

Global Optimization. After initialization, we jointly optimize all 3D Gaussian parameters and camera poses across all processed frames (Fig. 3 (b)). This global optimization ensures geometric consistency across the entire sequence, reducing accumulated pose drift and local misalignments.

Frame Insertion and Pose Estimation. As new frames arrive, we estimate their poses using the correspondenceguided PnP initialization and refinement strategy described in Sec. 3.3. If PnP fails due to insufficient feature correspondences or poor initialization, we trigger a fallback mechanism that re-optimizes all past frames globally before retrying pose estimation. This iterative fallback enhances robustness under challenging motion or weak texture (Fig. 3 (c)).

Local Optimization with Visibility-Adaptive Window. Once the pose is estimated, we optimize only the Gaussians visible in the new frame’s frustum, while constraining them with observations from nearby frames in a dynamically selected visibility-adapted local window (Fig. 6). Covisibility between frames is measured by:

IoU(t, t′) = |V(t) ∩ V(t′)| |V(t) ∪ V(t′)|

, (12)

where V(t) denotes the set of Gaussians visible in frame t. Frames with covisibility below a threshold τ are excluded from the window. This adaptive mechanism ensures local Gaussians are consistently supervised by reliable multi-view constraints, balancing efficiency and accuracy.

Final Global Refinement. In the final step, a final global refinement jointly optimizes all Gaussians and camera poses over the sequence. This final pass further improves both rendering quality and long-range pose consistency.

Depth and Reprojection Losses. To provide additional supervision in newly revealed regions, where multi-view observations are insufficient, we introduce two regularization terms. A monocular depth loss encourages rendered depth to match MASt3R’s scale-aligned depth prior:

Ldepth = ∥Drendered − DMASt3R∥2. (13)

Additionally, a keypoint reprojection loss enforces alignment between projected 3D keypoints and their 2D observations:

∥π(Xk) − uk∥2, (14)

Lreprojection =

k

where π(·) denotes projection using the current pose.

Total Loss. Throughout the entire incremental reconstruction pipeline, each processed frame is optimized using the following objective:

Ltotal = Lphoto + λdepthLdepth + λreprojectionLreprojection, (15)

This combined loss applies to both local and global optimization stages, ensuring coherent multi-view, robust pose refinement, and stable geometry reconstruction across the evolving scene.

Table 1. Quantitative comparison on the Free dataset [61] across various baseline methods. Methods such as CF-3DGS [14] frequently encounter out-of-memory issues, denoted by “-”. Our method consistently outperforms all baselines across diverse scenes, delivering superior rendering quality and robustness, especially in challenging environments characterized by complex camera trajectories and varied geometric structures. “*”: Initialized with MASt3R poses, then jointly optimized.

COLMAP [51] COLMAP [51] MASt3R [27] MASt3R [27]

CF-3DGS [14] NoPe-NeRF [5] LocalRF [39] Ours

Scenes

+ F2-NeRF [61] + Scaffold-GS [36] + Scaffold-GS [36] + Scaffold-GS [36]* PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Grass 23.44 0.58 0.45 26.75 0.82 0.20 22.65 0.61 0.34 25.06 0.79 0.21 - - - 16.39 0.27 0.81 18.84 0.35 0.60 26.16 0.80 0.22 Hydrant 23.75 0.74 0.28 26.66 0.86 0.12 23.22 0.71 0.21 25.68 0.83 0.12 - - - 17.94 0.43 0.66 19.19 0.48 0.48 24.69 0.79 0.18 Lab 24.34 0.83 0.26 28.27 0.92 0.10 20.66 0.74 0.25 22.42 0.80 0.18 - - - 17.42 0.52 0.63 17.22 0.55 0.47 27.11 0.87 0.15 Pillar 28.05 0.79 0.23 31.75 0.90 0.12 23.95 0.70 0.28 22.88 0.67 0.24 14.55 0.40 0.66 18.88 0.44 0.75 22.98 0.59 0.49 30.44 0.88 0.16 Road 26.03 0.80 0.27 30.45 0.92 0.10 24.23 0.73 0.25 25.05 0.78 0.27 - - - 17.48 0.44 0.79 20.68 0.54 0.56 27.73 0.84 0.20 Sky 25.10 0.86 0.24 28.34 0.92 0.12 23.26 0.80 0.22 25.37 0.88 0.14 - - - 16.18 0.51 0.65 18.76 0.60 0.46 28.07 0.91 0.13 Stair 28.14 0.84 0.22 32.13 0.93 0.10 23.35 0.71 0.30 24.46 0.79 0.28 13.41 0.41 0.63 19.14 0.47 0.69 23.55 0.66 0.38 31.00 0.89 0.16

Avg. 25.55 0.78 0.28 29.19 0.90 0.12 23.05 0.72 0.27 24.42 0.79 0.21 13.98 0.41 0.65 17.63 0.44 0.71 20.17 0.54 0.49 27.88 0.85 0.17

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

|Out of memory|
|---|

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

|Out of memory|
|---|

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

NoPe-NeRF LocalRF CF-3DGS MASt3R + Scaffold-GS MASt3R + Scaffold-GS* Ours Ground-truth

- Figure 7. Qualitative comparison on the Free dataset [61]. We compare our method with state-of-the-art approaches including NoPeNeRF [5], LocalRF [39], CF-3DGS [14], and MASt3R [27] combined with Scaffold-GS [36]. CF-3DGS fails due to memory constraints (OOM), and other baseline methods exhibit artifacts or blurry reconstructions. In contrast, our method produces results closest to the ground truth, demonstrating clearer details, accurate geometry, and visually consistent rendering, particularly under challenging scene structures and complex camera trajectories. “*”: Initialized with MASt3R poses, then jointly optimized.

#### 4. Experiments

##### 4.1. Experimental Setup

Datasets. We evaluate LongSplat on three challenging realworld datasets with varying difficulty levels:

- • Tanks and Temples [25] (Standard): Eight scenes with smooth, forward-facing camera trajectories, evaluated at full resolution. Every 8th frame is used for testing.
- • Free Dataset [61] (Moderate): Seven handheld videos featuring complex, unconstrained trajectories with multiple foreground objects, evaluated at 1/2 resolution. Frequent scene changes make memory-efficient 3D representation essential. Every 8th frame is tested.
- • Hike Dataset [39] (Hard): Long videos with hundreds to thousands of frames, complex trajectories, and detailed geometry, evaluated at 1/4 resolution. The scale and duration demand adaptive memory management. Every 10th frame is used for testing.

Evaluation Metrics. We evaluate novel view synthesis quality using PSNR, SSIM [64], and LPIPS [76]. Pose accuracy is measured with Absolute Trajectory Error (ATE) and Relative Pose Error (RPE), using COLMAP poses as ground truth. We also report model size, training time, and FPS to assess computational efficiency.

Table 2. Quantitative evaluation of camera pose estimation accuracy on the Free dataset [61]. Our method achieves superior performance across most scenes, significantly reducing pose errors compared to state-of-the-art approaches. “*”: Initialized with MASt3R poses, then jointly optimized.

Method RPEt↓ RPEr↓ ATE↓ MASt3R [27] + Scaffold-GS [36] 0.162 0.265 0.013 MASt3R [27] + Scaffold-GS [36]* 0.083 0.176 0.008 CF-3DGS [14] 0.234 3.442 0.022 NoPe-NeRF [5] 6.231 4.822 0.576 LocalRF [39] 0.754 7.086 0.035 Ours 0.028 0.103 0.004

Baselines. We compare LongSplat with COLMAP-based methods (COLMAP [51]+F2-NeRF [61] / 3DGS [22] / Scaffold-GS [36]) and unposed methods (NoPe-NeRF [5], LocalRF [39], CF-3DGS [14]). Additionally, we evaluate a na¨ıve baseline combining MASt3R’s [27] predicted point cloud and poses with Scaffold-GS. During training, camera poses are either fixed (MASt3R + Scaffold-GS) or jointly optimized (MASt3R + Scaffold-GS*).

Implementation Details. We implement LongSplat based on Scaffold-GS [36], using its learning rate schedule and growing/pruning rules. Each anchor emits k Gaussians predicted by a lightweight 2-layer MLP. The initial sparse voxel

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

ATE: 0.624

ATE: 00.006

ATE: 0.010

ATE: 0.021

ATE: 0.010

ATE: 0.002

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

OOM

ATE: 0.807

ATE: 0.031

ATE: 0.010

ATE: 0.010 ATE: 0.002

CF-3DGS MASt3R

Nope-NeRF LocalRF MASt3R

Ours

+ Scaffold-GS

+ Scaffold-GS*

- Figure 8. Visualization of camera trajectories on Free dataset [61]. CF-3DGS [14] encounters OOM and fails for long sequences, whereas our method reliably estimates accurate, stable trajectories, demonstrating superior robustness.

Table 3. Quantitative evaluation of novel view synthesis quality on the Tanks and Temples dataset [25]. Our proposed LongSplat consistently surpasses existing methods across multiple scenes.

Method PSNR↑ SSIM↑ LPIPS↓ RPEt↓ RPEr↓ ATE↓ COLMAP+3DGS [22] 30.21 0.92 0.10 – – – MASt3R [27] + Scaffold-GS [36] 28.67 0.79 0.21 0.166 0.168 0.006 MASt3R [27] + Scaffold-GS [36]* 30.92 0.90 0.13 0.047 0.103 0.005 NoPe-NeRF [5] 26.34 0.74 0.39 0.080 0.038 0.006 CF-3DGS [14] 31.28 0.93 0.09 0.041 0.069 0.004 Ours 32.83 0.94 0.08 00.032 0.068 0.003

[Figure 120]

[Figure 121]

NoPe-NeRF CF-3DGS Ours Ground-truth

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

- Figure 9. Qualitative comparison on the Tanks and Temples dataset [25]. NoPe-NeRF [5] produces visibly blurred results with inaccurate geometries, while CF-3DGS [14], despite better sharpness, fails to reconstruct fine details accurately. In contrast, our LongSplat method achieves superior rendering quality, closely matching the ground truth with sharper textures, more accurate geometry, and consistent lighting.

grid size is 0.1. Camera poses are optimized via a differentiable CUDA-accelerated rasterizer, parameterized with quaternions and translation vectors. We use 400 local, 900 global, and 10,000 refinement iterations, starting with three initial frames. The octree density thresholds for splitting and removal start at 10 and 5, progressively increasing with depth. Visibility IoU threshold is set to 0.2. All experiments are conducted on a single NVIDIA RTX 4090.

##### 4.2. Comparisons

Tanks and Temples. We evaluate LongSplat on the Tanks and Temples dataset [25], a standard benchmark for novel view synthesis. As shown in Tab. 3, LongSplat achieves state-of-the-art rendering quality (avg. PSNR: 32.83 dB) and superior camera pose estimation accuracy (lowest ATE and RPE). Qualitative results in Fig. 9 confirm sharper textures, accurate geometry, and better visual consistency compared to baselines. Please refer to the supplementary material for the full quantitative evaluation table for each scene.

- Table 4. Quantitative evaluation on the Hike dataset [39]. Our method consistently outperforms baselines across diverse scenes with complex trajectories and extended sequences, highlighting LongSplat’s robustness and superior scene representation capability. CF-3DGS [14] encounters OOM in all scenes and is thus omitted.

Method PSNR↑ SSIM↑ LPIPS↓

MASt3R [27] + Scaffold-GS [36] 17.30 0.42 0.52 MASt3R [27] + Scaffold-GS [36]* 17.90 0.44 0.50 LocalRF [39] 23.56 0.68 0.29 Ours 25.39 0.81 0.19

- Table 5. Ablation on training components. Removing pose estimation, global optimization, or local optimization significantly degrades performance, highlighting each module’s importance. Our full method achieves the best rendering quality and pose accuracy.

Method PSNR↑ SSIM↑ LPIPS↓ RPEt↓ RPEr↓ ATE↓

w/o Pose estimation 20.19 0.56 0.51 0.42 2.71 0.71 w/o Global optimization 20.50 0.58 0.41 0.12 0.50 0.01 w/o Local optimization 25.94 0.77 0.28 0.06 0.31 0.01 w/o Refinement 26.08 0.80 0.25 0.04 0.22 0.01 Ours 27.88 0.85 0.17 0.03 0.11 0.00

Free Dataset. We evaluate LongSplat on the challenging Free dataset, achieving superior reconstruction quality as shown in Tab. 1 and Fig. 7. Competing methods like CF3DGS often face OOM issues, while LocalRF produces fragmented geometry and pose drift. Although MASt3R + Scaffold-GS avoids OOM errors, its inaccurate global pose estimates from MASt3R result in blurred renderings and structural distortions. Our method also achieves consistently lower pose errors than baselines, as shown quantitatively in Tab. 2 and visually in Fig. 8.

Hike Dataset. We evaluate LongSplat on the challenging Hike dataset, achieving state-of-the-art reconstruction quality as shown in Tab. 4 and Fig. 10. Competing methods like CF-3DGS often fail due to OOM issue, LocalRF produces lower-quality reconstructions, and MASt3R struggles with long outdoor trajectories, resulting in poor reconstruction quality.

##### 4.3. Ablation Studies

Training Components. To analyze the contribution of each training component, we individually disable them and evaluate performance. As shown in Tab. 5, removing pose estimation severely harms reconstruction quality and increases pose errors (ATE: 0.71). Omitting global or local optimization also reduces performance. Our full method achieves the highest quality and minimal pose errors.

Local Window Sizes. We analyze the effect of local window size on reconstruction and pose accuracy in Tab. 6. Small fixed-size windows (e.g., 1 frame) lack sufficient constraints, causing fragmentation and higher errors. Our visibility-adapted window achieves the best balance, yielding the highest reconstruction quality and lowest pose drift.

Anchor Unprojection Strategies. We compare our adaptive octree anchor formation to (1) per-pixel initialization, (2)

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

MASt3R + Scaffold-GS MASt3R + Scaffold-GS* LocalRF Ours Ground-truth

- Figure 10. Qualitative results on the Hike dataset [39]. Compared to existing methods such as LocalRF [39] and MASt3R [27]+ScaffoldGS [36], our approach significantly improves visual clarity and reconstruction fidelity, accurately capturing complex details and textures in challenging scenes captured during long, casual outdoor trajectories. Notably, our method better preserves structural details and reduces artifacts, demonstrating enhanced robustness and visual quality. “*”: Initialized with MASt3R poses, then jointly optimized.

- Table 6. Ablation on local window sizes. Fixed small windows (e.g., 1-frame or 5-frame) or global optimization degrades reconstruction quality and pose accuracy. Our visibility-adaptive window dynamically selects optimal context, achieving the best balance of local detail and global consistency.

Window size PSNR ↑ SSIM ↑ LPIPS ↓ RPEt↓ RPEr↓ ATE↓

1-frame (Minimum Window) 26.58 0.80 0.23 0.05 0.21 0.01 5-frame (Fixed Window) 26.90 0.82 0.22 0.04 0.18 0.01 All Frames (Global Optimize) 26.15 0.78 0.26 0.06 0.28 0.08 Ours (Visibility-Adaptive) 27.88 0.85 0.17 0.03 0.11 0.00

- Table 7. Ablation on anchor unprojection strategies. Our Adaptive Octree method achieves the best rendering quality and lowest perceptual errors, significantly reducing memory usage (7.92× compression) compared to baselines.

Method PSNR↑ SSIM↑ LPIPS↓ Size (MB)↓ Compress↑

Per-pixel Unprojection (Dense) 22.47 0.69 0.35 799 1.00x Fixed-size Voxel Unprojection 26.99 0.81 0.18 591 1.35x Naive Densification 25.73 0.75 0.31 63 12.66x Ours (Adaptive Octree) 27.88 0.85 0.17 101 7.92x

- Table 8. Comparison of training efficiency on the Free dataset. Our method significantly reduces training time and achieves dramatically higher throughput (FPS) while simultaneously maintaining a compact model size compared to state-of-the-art approaches.

Method FPS ↑ Training time ↓ Size (MB) ↓ NoPe-NeRF [5] 0.29 36 hr 7 LocalRF [39] 1.17 14 hr 1080 CF-3DGS [14] 9.81 2 hr 1966 Ours 281.71 1 hr 101

fixed-resolution voxels, and (3) na¨ıve densification in Tab. 7. Our method achieves superior reconstruction quality with significantly reduced memory usage (7.92× compression).

Training Efficiency. We evaluate the computational efficiency of LongSplat (Tab. 8), which achieves 281.71 FPS and trains in just 1 hour on an NVIDIA RTX 4090, nearly 30× faster than LocalRF. Our method also significantly re-

0.12

0.030

0.12

| |CF-3DGS<br><br>Nope-NeRF<br><br>LocalRF MASt3R MASt3R + Scaffold-GS*<br><br>Ours|
|---|---|
| | |
| | |
| | |
| | |
| | |

| |CF-3DGS<br><br>Nope-NeRF<br><br>LocalRF MASt3R MASt3R + Scaffold-GS*<br><br>Ours|
|---|---|
| | |
| | |
| | |
| | |
| | |

CF-3DGS

Nope-NeRF

0.10

0.025

0.10

LocalRF MASt3R MASt3R + Scaffold-GS*

0.08

0.020

0.08

RPETran

RPERot

Ours

ATE

0.06

0.015

0.06

0.04

0.010

0.04

0.02

0.005

0.02

0.00 Samples (%)

0.000 Samples (%)

0.00 Samples (%)

(a) ATE (b) RPEt (c) RPEr

Figure 11. Robustness analysis on camera pose estimation (Free dataset [61]). We plot cumulative error distributions for ATE, RPE translation, and rotation. Our method consistently achieves lower errors compared to existing methods, demonstrating superior robustness and reduced pose drift.

duces the model size to approximately 101 MB.

Robustness Analysis of Camera Pose Estimation. We further analyze robustness by comparing cumulative error distributions for ATE and RPE (translation and rotation) in Fig. 11. LongSplat achieves consistently lower errors than baselines, effectively minimizing drift and maintaining stable trajectories, highlighting the advantage of our incremental optimization and robust tracking.

#### 5. Conclusion

We present LongSplat, a robust unposed 3D Gaussian Splatting framework for casual long videos. It integrates incremental joint optimization, a robust tracking module, and adaptive octree anchors, significantly improving pose accuracy, reconstruction quality, and memory efficiency. Extensive experiments confirm that LongSplat consistently outperforms state-of-the-art approaches. Future work includes handling dynamic scenes and enhancing pose estimation robustness.

Limitations. LongSplat shares common limitations of unposed reconstruction methods, assuming static scenes and fixed intrinsics, making it unsuitable for dynamic objects or varying focal lengths.

Acknowledgements. This work was supported by NVIDIA Taiwan AI Research & Development Center (TRDC). This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 1122222-E-A49-004-MY2 and 113-2628-E-A49-023-. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

#### References

- [1] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollh¨ofer, Johannes Kopf, Matthew O’Toole, and Changil Kim. Hyperreel: High-fidelity 6-dof video with rayconditioned sampling. In CVPR, 2023. 2
- [2] Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In ICCV, 2021. 2
- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022.
- [4] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. arXiv preprint arXiv:2304.06706, 2023. 2
- [5] Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian, and Victor Adrian Prisacariu. Nope-nerf: Optimising neural radiance field with no pose prior. In CVPR, 2023. 2, 6, 7, 8, 15, 18
- [6] Chris Buehler, Michael Bosse, Leonard McMillan, Steven Gortler, and Michael Cohen. Unstructured lumigraph rendering. In SIGGRAPH, 2001. 2
- [7] Bo-Yu Chen, Wei-Chen Chiu, and Yu-Lun Liu. Improving robustness for joint optimization of camera pose and decomposed low-rank tensorial radiance fields. In AAAI, 2024. 2
- [8] Shenchang Eric Chen and Lance Williams. View interpolation for image synthesis. In SIGGRAPH, 1993. 2
- [9] Zezhou Cheng, Carlos Esteves, Varun Jampani, Abhishek Kar, Subhransu Maji, and Ameesh Makadia. Lu-nerf: Scene and pose estimation by synchronizing local unposed nerfs. arXiv preprint arXiv:2306.05410, 2023. 2
- [10] Shin-Fang Chng, Sameera Ramasinghe, Jamie Sherrah, and Simon Lucey. Garf: Gaussian activated radiance fields for high fidelity reconstruction and pose estimation. arXiv eprints, 2022. 2
- [11] Wenyan Cong, Kevin Wang, Jiahui Lei, Colton Stearns, Yuanhao Cai, Dilin Wang, Rakesh Ranjan, Matt Feiszli, Leonidas Guibas, Zhangyang Wang, Weiyao Wang, and Zhiwen Fan. Videolifter: Lifting videos to 3d with fast hierarchical stereo alignment, 2025. 2
- [12] Paul E Debevec, Camillo J Taylor, and Jitendra Malik. Modeling and rendering architecture from photographs: A hybrid geometry-and image-based approach. In SIGGRAPH, 1996. 2
- [13] Cheng-De Fan, Chen-Wei Chang, Yi-Ruei Liu, Jie-Ying Lee, Jiun-Long Huang, Yu-Chee Tseng, and Yu-Lun Liu. Spectromotion: Dynamic 3d reconstruction of specular scenes. In CVPR, 2025. 2

- [14] Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In CVPR, 2024. 2, 6, 7, 8, 14, 15, 17, 18
- [15] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In ICCV, 2021. 2
- [16] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. 2003. 2
- [17] Derek Hoiem, Alexei A Efros, and Martial Hebert. Automatic photo pop-up. In ACM SIGGRAPH 2005 Papers, 2005. 2
- [18] Youichi Horry, Ken-Ichi Anjyo, and Kiyoshi Arai. Tour into the picture: using a spidery mesh interface to make animation from a single image. In Proceedings of the 24th annual conference on Computer graphics and interactive techniques,

1997. 2

- [19] Hao-Yu Hou, Chia-Chi Hsu, Yu-Chen Huang, Mu-Yi Shen, Wei-Fang Sun, Cheng Sun, Chia-Che Chang, Yu-Lun Liu, and Chun-Yi Lee. 3d gaussian splatting with grouped uncertainty for unconstrained images. In ICASSP, 2025. 2
- [20] Ronghang Hu, Nikhila Ravi, Alex Berg, and Deepak Pathak. Worldsheet: Wrapping the world in a 3d sheet for view synthesis from a single image. In ICCV, 2020. 2
- [21] Bo Ji and Angela Yao. Sfm-free 3d gaussian splatting via hierarchical training. arXiv preprint arXiv:2412.01553, 2024. 2
- [22] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM TOG, 2023. 2, 3, 6, 7, 15
- [23] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM TOG, 2024. 3
- [24] Mijeong Kim, Seonguk Seo, and Bohyung Han. Infonerf: Ray entropy minimization for few-shot neural volume rendering. In CVPR, 2022. 2
- [25] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM TOG, 2017. 6, 7, 15, 18
- [26] Johannes Kopf, Michael F. Cohen, and Richard Szeliski. Firstperson hyper-lapse videos. ACM TOG, 2014. 2
- [27] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In ECCV, 2024. 2, 3, 5, 6, 7, 8, 14, 15, 18, 19, 20
- [28] Du-Hsiu Li, Hsueh-Ming Hang, and Yu-Lun Liu. Virtual view synthesis using backward depth warping algorithm. In PCS, 2013. 2
- [29] Jiaxin Li, Zijian Feng, Qi She, Henghui Ding, Changhu Wang, and Gim Hee Lee. Mine: Towards continuous depth mpi with nerf for novel view synthesis. In ICCV, 2021. 2
- [30] Ming-Feng Li, Yueh-Feng Ku, Hong-Xuan Yen, Chi Liu, YuLun Liu, Albert YC Chen, Cheng-Hao Kuo, and Min Sun. Genrc: Generative 3d room completion from sparse image collections. In ECCV, 2024. 3
- [31] Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In ICCV, 2021. 2

- [32] Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, and Yu-Lun Liu. Frugalnerf: Fast convergence for few-shot novel view synthesis without learned priors. CVPR, 2025. 2
- [33] Jiaqi Lin, Zhihao Li, Xiao Tang, Jianzhuang Liu, Shiyong Liu, Jiayue Liu, Yangdi Lu, Xiaofei Wu, Songcen Xu, Youliang Yan, et al. Vastgaussian: Vast 3d gaussians for large scene reconstruction. In CVPR, 2024. 3
- [34] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. In NeurIPS,

2020. 2

- [35] Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust dynamic radiance fields. In CVPR,

2023. 2

- [36] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In CVPR, 2024. 2, 3, 4, 6, 7, 8, 14, 15, 18, 19, 20
- [37] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713,

- 2023. 2

[38] Caoyuan Ma, Yu-Lun Liu, Zhixiang Wang, Wu Liu, Xinchen Liu, and Zheng Wang. Humannerf-se: A simple yet effective approach to animate humannerf with diverse poses. In CVPR,

- 2024. 2

- [39] Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H Kim, and Johannes Kopf. Progressively optimized local radiance fields for robust view synthesis. In CVPR, 2023. 2, 3, 6, 7, 8, 14, 15, 18, 19, 20
- [40] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 2
- [41] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM TOG, 2022. 2
- [42] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: a versatile and accurate monocular slam system. IEEE transactions on robotics, 2015. 2
- [43] Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In CVPR, 2022. 2
- [44] Yunlong Ran, Yanxu Li, Qi Ye, Yuchi Huo, Zechun Bai, Jiahao Sun, and Jiming Chen. Ct-nerf: Incremental optimizing neural radiance field and poses with complex trajectory. arXiv preprint arXiv:2404.13896, 2024. 2
- [45] Christian Reiser, Songyou Peng, Yiyi Liao, and Andreas Geiger. Kilonerf: Speeding up neural radiance fields with thousands of tiny mlps. In ICCV, 2021. 2
- [46] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, 2021. 12, 13

- [47] Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024. 3
- [48] Gernot Riegler and Vladlen Koltun. Free view synthesis. In ECCV, 2020. 2
- [49] Gernot Riegler and Vladlen Koltun. Stable view synthesis. In CVPR, 2021. 2
- [50] Sara Fridovich-Keil and Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022. 2
- [51] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 1, 2, 6
- [52] Mu-Yi Shen, Chia-Chi Hsu, Hao-Yu Hou, Yu-Chen Huang, Wei-Fang Sun, Chia-Che Chang, Yu-Lun Liu, and Chun-Yi Lee. Driveenv-nerf: Exploration of a nerf-based autonomous driving environment for real-world performance validation. arXiv preprint arXiv:2403.15791, 2024. 2
- [53] Chih-Hai Su, Chih-Yao Hu, Shr-Ruei Tsai, Jie-Ying Lee, Chin-Yang Lin, and Yu-Lun Liu. Boostmvsnerfs: Boosting mvs-based nerfs to generalizable view synthesis in large-scale scenes. In ACM SIGGRAPH 2024 Conference Papers, 2024. 3
- [54] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 2
- [55] Teppei Suzuki. Fed3dgs: Scalable 3d gaussian splatting with federated learning. arXiv preprint arXiv:2403.11460, 2024. 3
- [56] Takafumi Taketomi, Hideaki Uchiyama, and Sei Ikeda. Visual slam algorithms: A survey from 2010 to 2016. IPSJ Transactions on Computer Vision and Applications, 2017. 2
- [57] Matthew Tancik, Vincent Casser, Xinchen Yan, Sabeek Pradhan, Ben Mildenhall, Pratul P Srinivasan, Jonathan T Barron, and Henrik Kretzschmar. Block-nerf: Scalable large scene neural view synthesis. In CVPR, 2022. 3
- [58] Richard Tucker and Noah Snavely. Single-view view synthesis with multiplane images. In CVPR, 2020. 2
- [59] Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In CVPR, 2022. 2
- [60] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024. 3
- [61] Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. F2nerf: Fast neural radiance field training with free camera trajectories. In CVPR, 2023. 6, 7, 8, 15, 17, 18
- [62] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387,

2025. 3

- [63] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 3
- [64] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004. 6

- [65] Zirui Wang, Shangzhe Wu, Weidi Xie, Min Chen, and Victor Adrian Prisacariu. NeRF−−: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064, 2021. 2
- [66] Yitong Xia, Hao Tang, Radu Timofte, and Luc Van Gool. Sinerf: Sinusoidal neural radiance fields for joint pose estimation and scene reconstruction. 2022. 2
- [67] Dejia Xu, Yifan Jiang, Peihao Wang, Zhiwen Fan, Humphrey Shi, and Zhangyang Wang. Sinnerf: Training neural radiance fields on complex scenes from a single image. In ECCV, 2022. 2
- [68] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Pointbased neural radiance fields. In CVPR, 2022. 2
- [69] Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In CVPR, 2023. 2
- [70] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. arXiv preprint arXiv:2501.13928, 2025. 3
- [71] Lin Yen-Chen, Pete Florence, Jonathan T Barron, Alberto Rodriguez, Phillip Isola, and Tsung-Yi Lin. inerf: Inverting neural radiance fields for pose estimation. In IROS, 2021. 2
- [72] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In ICCV, 2021. 2
- [73] Yu-Ting Zhan, Cheng-Yuan Ho, Hebi Yang, Yi-Hsin Chen, Jui Chiu Chiang, Yu-Lun Liu, and Wen-Hsiao Peng. Cat-3dgs: A context-adaptive triplane approach to ratedistortion-optimized 3dgs compression. arXiv preprint arXiv:2503.00357, 2025. 2
- [74] Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv:2010.07492, 2020. 2
- [75] Qiang Zhang, Seung-Hwan Baek, Szymon Rusinkiewicz, and Felix Heide. Differentiable point-based radiance fields for efficient view synthesis. In SIGGRAPH Asia 2022 Conference Papers, 2022. 2
- [76] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [77] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. 2018. 2

#### A. Implementation Details

We implement LongSplat using PyTorch. Our rendering and 3D Gaussian updates are accelerated using CUDA and cuDNN. Camera pose optimization is performed using differentiable rendering, while the PnP initialization leverages OpenCV’s solver with RANSAC. All experiments run on NVIDIA 4090 GPUs.

- A.1. LongSplat Algorithm: Pseudo-Code

The LongSplat pipeline incrementally reconstructs a scene from a casually captured long video, without known poses, by tightly coupling pose estimation and 3D Gaussian Splatting. The workflow can be summarized in the following pseudo-code:

Algorithm 1: LONGSPLAT: Incremental 3DGS

Input: RGB frames {It}Tt=1 Output: 3DGS G, camera poses {Pt}Tt=1 /* Initialization */ (Dt, Ct, Pt) ←

MASt3R Global Alignment(I1...Ninit) OctreeAnchorFormation(G, Dt, Pt) /* Incremental Joint Optimization */ for t ← Ninit to T do

GlobalOptimize(G, {P1..t−1}, Kg) (Dt, Ct) ← MASt3R(It) Pt ← PnP RANSAC(Ct, G) if Pt = FAIL then

fallback to t

end PoseRefine(G, Pt, It) AnchorUnprojection(G, Dt, Pt) W ← VisibilityWindow(t) LocalOptimize(G, {Pk}k∈W, Kℓ)

end /* Final Global Refinement */ GlobalRefinement(G, {P1..T}, Kr) return (G, {Pt}Tt=1)

- B. Additional Experiments

##### B.1. CO3Dv2 Benchmark Evaluation.

We report the results on CO3Dv2 [46] in Fig. 12 and Table 9. LongSplat surpasses CF-3DGS and HT-3DGS in all image and pose metrics, confirming the method’s robustness on this more challenging benchmark.

##### B.2. Comparison between COLMAP and LongSplat on the Hike Dataset

We compare LongSplat with a standard COLMAP-based reconstruction pipeline on our Hike dataset. This dataset poses extreme challenges for incremental SfM due to vegetation

Table 9. Qualtitative comparison on the CO3Dv2 dataset [46]

Dataset Method PSNR ↑ SSIM ↑ LPIPS ↓ ATE↓ RPEt↓ RPEr↓ CO3Dv2

CF-3DGS 26.61 0.79 0.29 0.014 0.218 0.374 HT-3DGS 28.34 0.84 0.30 0.017 0.058 0.314 Ours 32.59 0.91 0.17 0.005 0.023 0.096

Table 10. Pose Accuracy on Hike Dataset.

Hike dataset ATE↓ RPEt↓ RPEr↓

MASt3R + Scaffold-GS 0.006 0.009 0.292 MASt3R + Scaffold-GS* 0.006 0.009 0.221 LocalRF 0.004 0.011 0.211 Ours 0.002 0.003 0.128

Table 11. Qualitative comparison with HT-3DGS.

Dataset Method PSNR ↑ SSIM ↑ LPIPS ↓ ATE↓ RPEt↓ RPEr↓ Success Rate Tanks & Temples

HT-3DGS 33.53 0.96 0.07 0.00 0.04 0.07 8/8 Ours 32.83 0.94 0.08 0.00 0.03 0.07 8/8

HT-3DGS 13.75 0.39 0.65 0.02 0.34 4.41 6/7 Ours 27.88 0.85 0.17 0.00 0.03 0.10 7/7

Free

HT-3DGS OOM OOM OOM OOM OOM OOM 0/12

Hike

Ours 25.39 0.81 0.19 0.00 0.01 0.21 12/12

occlusion, textureless surfaces, and long trajectories. The quantitative results in Table 13 show that LongSplat consistently outperforms COLMAP in both rendering quality and pose estimation accuracy. This highlights the advantage of our octree-anchored Gaussian formulation combined with learned 3D priors.

##### B.3. Pose Accuracy on Hike Dataset.

COLMAP poses are noisy on several Hike videos, so we use the 6 stable sequences (forest2, indoor, university1-4) as references to compute pose accuracy in Table 10. LongSplat achieves the lowest errors, beating all baselines.

##### B.4.ComparisonbetweenHT-3DGSandLongSplat

We report the comparison with HT-3DGS in Table 11 and Fig. 13. HT-3DGS runs only on T&T (33.53 dB), but falls to 13.75 dB on Free and runs OOM on Hike. LongSplat remains stable across all datasets. This confirms our SOTA claim for long, casually captured videos.

##### B.5. Ablation on Using MASt3R Relative Poses

To demonstrate the importance of our proposed pose estimation pipeline, we conduct an ablation replacing LongSplat’s correspondence-guided PnP with directly using MASt3R’s relative pose estimates. As shown in Fig. 14, this leads to degraded novel view synthesis quality and larger pose errors, especially in long sequences. This confirms that raw MASt3R poses alone are insufficient for high-quality incremental reconstruction.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

HT-3DGS Ours HT-3DGS Ours

###### Figure 12. Qualitative comparison on the CO3Dv2 dataset [46]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

HT-3DGS Ours HT-3DGS Ours

Figure 13. Qualitative comparison with HT-3DGS

Table 12. Ablation on training loss.

Method PSNR↑ SSIM↑ LPIPS↓ RPEt↓ RPEr↓ ATE↓ w/o 2d correspondence loss 26.54 0.80 0.24 0.049 0.253 0.007 w/o depth loss 26.74 0.82 0.22 0.076 0.246 0.011 Ours 27.88 0.85 0.17 0.028 0.103 0.004

- B.6. Ablation on training loss

We report the ablation study on training loss in Table 12. Removing individual losses degrades performance. Our full method achieves the best rendering quality and pose accuracy.

- C. Complete Quantitative Evaluation

- C.1. Tanks and Temples

We provide full quantitative results on the Tanks and Temples benchmark in Tabs. 14 and 15. LongSplat consistently outperforms baselines in both rendering quality and pose estimation accuracy, demonstrating its effectiveness even in indoor and urban scenes with varied scales and complexities.

##### C.2. Free dataset

We provide full quantitative results on the Free dataset benchmark in Tab. 16. LongSplat consistently outperforms baselines in both rendering quality and pose estimation accuracy, demonstrating its effectiveness even in indoor and urban scenes with varied scales and complexities.

##### C.3. Hike dataset

Hike dataset benchmark in Tab. 13. LongSplat consistently outperforms baselines in both rendering quality and pose estimation accuracy, demonstrating its effectiveness even

in challenging indoor and urban scenes with varied scales and complexities. Notably, in scenarios where COLMAP fails to reconstruct due to long trajectories or low-texture regions, LongSplat maintains high-quality results, preserving structural details and ensuring stable pose estimation.

#### D. Additional Visual Comparisons

##### D.1. Visual Comparison on Ablation Study

Fig. 15 shows the visual impact of removing key training components. Both trajectory estimation and novel view synthesis degrade severely when global optimization, local optimization, or final refinement is removed, emphasizing their importance.

##### D.2. Additional Trajectory Results

We include additional visualizations of camera trajectories estimated by LongSplat. As shown in Fig. 16, our method reconstructs stable, drift-free trajectories even in long and complex sequences.

##### D.3. Additional Tanks and Temples Results

We provide additional qualitative comparisons on the Tanks and Temples benchmark. LongSplat produces sharper and more visually consistent results across diverse scenes, demonstrating strong generalization across both indoor and outdoor environments.

##### D.4. Additional Free Dataset Results

Additional qualitative comparisons on the Free dataset are shown in Fig. 18. Our method preserves more fine details, produces fewer artifacts, and achieves sharper novel view synthesis than all baselines.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

###### Figure 14. Visual comparisons on ablation MASt3R relative pose.

- Table 13. Quantitative evaluation on the Hike dataset [39]. Our method consistently outperforms baselines across diverse scenes with complex trajectories and extended sequences, highlighting LongSplat’s robustness and superior scene representation capability. CF3DGS [14] encounters OOM in all scenes and is thus omitted.

COLMAP MASt3R [27] MASt3R [27]

LocalRF [39] Ours

Scenes

+ Scaffold-GS [36] + Scaffold-GS [36] + Scaffold-GS [36] PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

- forest1 20.12 0.55 0.44 17.68 0.30 0.64 17.54 0.34 0.55 19.12 0.45 0.41 23.86 0.79 0.21

- forest2 28.35 0.89 0.14 20.91 0.53 0.36 21.11 0.54 0.35 27.23 0.84 0.15 27.87 0.88 0.11

- forest3 - - - 9.54 0.15 0.70 9.62 0.15 0.70 17.05 0.38 0.59 19.59 0.62 0.31

- garden1 20.77 0.67 0.28 13.09 0.23 0.75 14.84 0.27 0.72 22.11 0.66 0.28 24.12 0.80 0.19

- garden2 - - - 13.21 0.19 0.75 15.67 0.26 0.74 23.34 0.61 0.33 24.35 0.74 0.25

- garden3 23.46 0.73 0.23 11.82 0.13 0.64 11.89 0.13 0.64 23.33 0.67 0.27 24.01 0.75 0.23 indoor 28.85 0.90 0.19 23.64 0.81 0.33 24.64 0.83 0.31 30.17 0.91 0.17 30.62 0.92 0.17 playground - - - 19.31 0.49 0.40 19.73 0.52 0.38 22.29 0.63 0.28 24.30 0.78 0.18

- university1 25.36 0.78 0.27 19.38 0.47 0.53 19.62 0.48 0.52 25.22 0.71 0.32 25.50 0.79 0.24

- university2 27.25 0.87 0.13 20.27 0.58 0.36 20.72 0.60 0.35 24.56 0.75 0.23 26.82 0.85 0.15

- university3 26.98 0.89 0.13 18.59 0.51 0.39 19.31 0.57 0.35 23.23 0.73 0.23 25.57 0.86 0.13

- university4 25.03 0.82 0.17 20.23 0.61 0.39 20.13 0.61 0.39 25.08 0.79 0.22 27.00 0.88 0.12 Avg 25.13 0.79 0.22 17.30 0.42 0.52 17.90 0.44 0.50 23.56 0.68 0.29 25.39 0.81 0.19

##### D.5. Additional Hike Dataset Results

Finally, we present more qualitative results on the Hike dataset in Fig. 19, Fig. 20. LongSplat reconstructs complex natural scenes with higher visual quality, capturing vegetation, terrain, and large-scale geometry with remarkable accuracy.

- Table 14. Quantitative evaluation of novel view synthesis quality on the Tanks and Temples dataset [25]. Our proposed LongSplat consistently surpasses existing methods across multiple challenging scenes.

Scenes

COLMAP+3DGS [22] NoPe-NeRF [5] CF-3DGS [14] Ours PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

Church 29.93 0.93 0.09 25.17 0.73 0.39 30.23 0.93 0.11 30.96 0.93 0.10 Barn 31.08 0.95 0.07 26.35 0.69 0.44 31.23 0.90 0.10 32.57 0.92 0.09 Museum 34.47 0.96 0.05 26.77 0.76 0.35 29.91 0.91 0.11 33.78 0.95 0.06 Family 27.93 0.92 0.11 26.01 0.74 0.41 31.27 0.94 0.07 33.67 0.96 0.06 Horse 20.91 0.77 0.23 27.64 0.84 0.26 33.94 0.96 0.05 33.42 0.96 0.06 Ballroom 34.48 0.96 0.04 25.33 0.72 0.38 32.47 0.96 0.07 32.80 0.95 0.06 Francis 32.64 0.92 0.15 29.48 0.80 0.38 32.72 0.91 0.14 33.80 0.92 0.15 Ignatius 30.20 0.93 0.08 23.96 0.61 0.47 28.43 0.90 0.09 31.61 0.94 0.07

Avg. 30.21 0.92 0.10 26.34 0.74 0.39 31.28 0.93 0.09 32.83 0.94 0.08

- Table 15. Quantitative evaluation of camera pose estimation accuracy on the Tanks and Temples dataset [25]. Our method achieves consistently low errors across diverse scenes, outperforming CF-3DGS and NoPe-NeRF, especially in terms of global trajectory accuracy (ATE) and local translation consistency (RPEt).

Scenes

CF-3DGS NoPe-NeRF Ours ATE↓ RPEr↓ RPEt↓ ATE↓ RPEr↓ RPEt↓ ATE↓ RPEr↓ RPEt↓

Church 0.002 0.018 0.008 0.008 0.008 0.034 0.001 0.048 0.011 Barn 0.003 0.034 0.034 0.004 0.032 0.046 0.004 0.061 0.025 Museum 0.005 0.215 0.052 0.020 0.202 0.207 0.001 0.046 0.025 Family 0.002 0.024 0.022 0.001 0.015 0.047 0.002 0.043 0.021 Horse 0.003 0.057 0.112 0.003 0.017 0.179 0.001 0.046 0.086 Ballroom 0.003 0.024 0.037 0.002 0.018 0.041 0.002 0.053 0.021 Francis 0.006 0.154 0.029 0.005 0.009 0.057 0.009 0.213 0.036 Ignatius 0.005 0.032 0.033 0.002 0.005 0.026 0.002 0.034 0.032

Avg. 0.004 0.069 0.041 0.006 0.038 0.080 0.003 0.068 0.032

- Table 16. Quantitative evaluation of camera pose estimation accuracy on the Free dataset [61]. “-” indicates methods that encountered out-of-memory issues. Our method consistently achieves superior performance across most scenes, significantly reducing pose errors compared to state-of-the-art approaches. “*”: Initialized with MASt3R poses, then jointly optimized.

MASt3R [27] MASt3R [27]

CF-3DGS [14] NoPe-NeRF [5] LocalRF [39] Ours

Scenes

+ Scaffold-GS [36] + Scaffold-GS [36]* ATE ↓ RPEr ↓ RPEt ↓ ATE ↓ RPEr ↓ RPEt ↓ ATE ↓ RPEr ↓ RPEt ↓ ATE ↓ RPEr ↓ RPEt ↓ ATE ↓ RPEr ↓ RPEt ↓ ATE ↓ RPEr ↓ RPEt ↓

Grass 0.038 0.554 0.559 0.002 0.152 0.016 - - - 0.431 9.333 3.044 0.056 6.026 0.612 0.000 0.058 0.002 Hydrant 0.013 0.168 0.145 0.013 0.165 0.144 - - - 0.480 4.068 5.844 0.060 8.487 1.068 0.013 0.111 0.069 Lab 0.009 0.294 0.175 0.009 0.265 0.178 - - - 0.533 2.623 5.774 0.041 4.405 1.072 0.004 0.217 0.067 Pillar 0.003 0.225 0.024 0.003 0.199 0.016 0.023 4.744 0.328 0.576 4.176 2.013 0.025 3.553 0.526 0.001 0.066 0.003 Road 0.013 0.153 0.088 0.013 0.159 0.088 - - - 0.584 4.087 6.045 0.023 9.798 0.699 0.005 0.080 0.036 Sky 0.010 0.203 0.091 0.010 0.197 0.090 - - - 0.807 6.661 9.775 0.031 11.075 0.894 0.002 0.114 0.017 Stair 0.006 0.260 0.050 0.006 0.247 0.050 0.021 2.139 0.140 0.624 2.809 11.120 0.008 6.257 0.563 0.000 0.078 0.001

Avg. 0.013 0.265 0.162 0.008 0.198 0.083 0.019 4.365 0.290 0.576 4.822 6.231 0.035 7.086 0.776 0.004 0.103 0.028

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

- Figure 15. Visual comparisons on ablation studies. The top row shows the camera trajectory estimation and novel view synthesis results when different training components are removed, demonstrating the importance of each proposed module. Removing global optimization, local optimization, or final refinement significantly degrades pose accuracy and reconstruction quality. The bottom row evaluates different settings for the visibility-adapted local window size. Too small a window leads to unstable geometry and pose drift, while too large a window dilutes local visibility priors, slowing convergence. LongSplat achieves the best balance using the proposed adaptive window.

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

CF-3DGS

OOM

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

CF-3DGS

OOM

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

CF-3DGS OOM

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

CF-3DGS

OOM

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

CF-3DGS

OOM

[Figure 240]

- Figure 16. Visualization of camera trajectories on Free dataset [61]. CF-3DGS [14] encounters OOM and fails for long sequences, whereas our method reliably estimates accurate, stable trajectories, demonstrating superior robustness.

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

### NoPe-NeRF CF-3DGS Ours Ground-truth

- Figure 17. More Qualitative comparison on the Tanks and Temples dataset [25]. NoPe-NeRF [5] produces visibly blurred results with inaccurate geometries, while CF-3DGS [14], despite better sharpness, fails to reconstruct fine details accurately. In contrast, our LongSplat method achieves superior rendering quality, closely matching the ground truth with sharper textures, more accurate geometry, and consistent lighting.

NoPe-NeRF LocalRF CF-3DGS MASt3R + Scaffold-GS MASt3R + Scaffold-GS* Ours Ground-truth

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

|Out of memory|
|---|

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

|Out of memory|
|---|

[Figure 279]

|[Figure 280]<br><br>Out of memory|
|---|

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

- Figure 18. More Qualitative comparison on the Free dataset [61]. We compare our method with state-of-the-art approaches including NoPe-NeRF [5], LocalRF [39], CF-3DGS [14], and MASt3R [27] combined with Scaffold-GS [36]. CF-3DGS fails due to memory constraints (OOM), and other baseline methods exhibit artifacts or blurry reconstructions. In contrast, our method produces results closest to the ground truth, demonstrating clearer details, accurate geometry, and visually consistent rendering, particularly under challenging scene structures and complex camera trajectories. “*”: Initialized with MASt3R poses, then jointly optimized.

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

MASt3R + Scaffold-GS MASt3R + Scaffold-GS* LocalRF Ours Ground-truth

- Figure 19. Qualitative results on the Hike dataset [39]. Compared to existing methods such as LocalRF [39] and MASt3R [27]+ScaffoldGS [36], our approach significantly improves visual clarity and reconstruction fidelity, accurately capturing complex details and textures in challenging scenes captured during long, casual outdoor trajectories. Notably, our method better preserves structural details and reduces artifacts, demonstrating enhanced robustness and visual quality. “*”: Initialized with MASt3R poses, then jointly optimized.

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

MASt3R + Scaffold-GS MASt3R + Scaffold-GS* LocalRF Ours Ground-truth

- Figure 20. More Qualitative results on the Hike dataset [39]. Compared to existing methods such as LocalRF [39] and MASt3R [27]+Scaffold-GS [36], our approach significantly improves visual clarity and reconstruction fidelity, accurately capturing complex details and textures in challenging scenes captured during long, casual outdoor trajectories. Notably, our method better preserves structural details and reduces artifacts, demonstrating enhanced robustness and visual quality. “*”: Initialized with MASt3R poses, then jointly optimized.

