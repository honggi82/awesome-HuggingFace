###### a). Per-scene Optimization Methods

###### c). Ours: Feed Forward Models with Post Optimization

Images

Pose Optimization

Correspondence

Gaussian Splatting

Images Encoder Decoder

Keypoints&Desc. / Optical Flow …

PNP / ICP / BA…

2D / 3D / MLP …

Gaussian Splatting

###### b). Feed Forward Models

ARTDECO: TOWARDS EFFICIENT AND HIGHFIDELITY ON-THE-FLY 3D RECONSTRUCTION WITH STRUCTURED SCENE REPRESENTATION

Poses Points

Pose Optimization Correspondence 2D / 3D / MLP …

Images

Decoder

Encoder

Gaussian Splatting

Camera/Point Head

DinoV2…

2D / 3D / MLP …

Guanghao Li1,2,3∗ Kerui Ren1,4∗ Linning Xu1,5 Zhewen Zheng1,6 Changjian Jiang1,7 Xin Gao1,2 Bo Dai8 Jian Pu2† Mulin Yu1† Jiangmiao Pang1

1Shanghai Artificial Intelligence Laboratory, 2Fudan University, 3Shanghai Innovation Institute, 4Shanghai Jiao Tong University, 5The Chinese University of Hong Kong, 6Carnegie Mellon University, 7Zhejiang University, 8The University of Hong Kong

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

# arXiv:2510.08551v1[cs.CV]9Oct2025

[Figure 9]

[Figure 10]

[Figure 11]

###### VRNeRF (KITCHEN)

[Figure 12]

[Figure 13]

…

[Figure 14]

[Figure 15]

…

monocular captures

|TIME: 27.1 s PSNR: 28.56 dB MEM: 32.99 MB|
|---|

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Figure 1: ARTDECO delivers high-fidelity, interactive 3D reconstruction from monocular images, combining efficiency with robustness across indoor and outdoor scenes.

ABSTRACT

On-the-fly 3D reconstruction from monocular image sequences is a long-standing challenge in computer vision, critical for applications such as real-to-sim, AR/VR, and robotics. Existing methods face a major tradeoff: per-scene optimization yields high fidelity but is computationally expensive, whereas feed-forward foundation models enable real-time inference but struggle with accuracy and robustness. In this work, we propose ARTDECO, a unified framework that combines the efficiency of feed-forward models with the reliability of SLAM-based pipelines. ARTDECO uses 3D foundation models for pose estimation and point prediction, coupled with a Gaussian decoder that transforms multi-scale features into structured 3D Gaussians. To sustain both fidelity and efficiency at scale, we design a hierarchical Gaussian representation with a LoD-aware rendering strategy, which improves rendering fidelity while reducing redundancy. Experiments on eight diverse indoor and outdoor benchmarks show that ARTDECO delivers interactive performance comparable to SLAM, robustness similar to feed-forward systems, and reconstruction quality close to per-scene optimization, providing a practical path toward on-the-fly digitization of real-world environments with both accurate geometry and high visual fidelity. Explore more demos on our project page: https://city-super.github.io/artdeco/.

∗Equal contribution †Corresponding author.

- 1 INTRODUCTION

High-fidelity 3D reconstruction from monocular image sequences is a long-standing goal in computer vision. Monocular data are relatively inexpensive, ubiquitous, and easy to capture, while accurate 3D scene representations are crucial for downstream applications such as embodied intelligence, AR/VR, or real-to-sim content creation. Recently, 3D Gaussian Splatting (Kerbl et al.,

- 2023a) has emerged as an efficient scene representation with strong empirical results. However, in monocular settings the lack of reliable geometric cues such as ambiguous scale, limited parallax, motion blur, and poor overlap makes it difficult to achieve accuracy, speed, and robustness at the same time. As a consequence, many existing systems optimize one objective at the expense of the others, limiting their practicality for online deployment.

Current 3DGS-based 3D reconstruction methods fall into two paradigms. Per-scene optimization methods (Matsuki et al., 2024; Huang et al., 2024b; Meuleman et al., 2025b) rely on image poses estimated from Structure from Motion (SfM) or Simultaneous Localization and Mapping (SLAM), achieving high accuracy but at the cost of substantial computation, with robustness limited by the fragility of these pipelines. In contrast, recent feed-forward models (Tang et al., 2024; Ye et al., 2024; Jiang et al., 2025b; Zhang et al., 2025) leverage large-scale data to learn monocular priors and directly regress poses and Gaussian primitives with attention, enabling fast and robust inference across diverse scenes but with limited rendering fidelity and weak consistency. This tradeoff highlights the need for approaches that combine the efficiency of feed-forward models with the strengths of per-scene optimization methods to deliver accurate, robust, real-time reconstruction. Beyond the efficiency–accuracy tradeoff, 3DGS pipelines are also highly sensitive to scene scale. As the scene grows, the number of Gaussian primitives required for training and rendering increases rapidly, which reduces efficiency. Prior attempts to address this either apply post-hoc anchor-based pruning, which lowers computation but introduces boundary artifacts and increases memory cost, or add multi-scale Gaussians during training, which mitigates artifacts but lacks explicit structural organization. These limitations underscore the need for a principled and practical level-of-detail (LoD) mechanism in 3DGS.

ARTDECO1 derives its name from a streamlined pipeline that unifies Accurate localization, Robust reconstruction, and Decoder-based rendering, with the aim of enabling on-the-fly 3D scene reconstruction and rendering. The core of ARTDECO is the goal of balancing real-time performance, accuracy, and robustness. It employs feed-forward models as data priors to reduce monocular ambiguities while maintaining the efficiency required for interactive use. To address the global inconsistency often seen in feed-forward approaches, ARTDECO integrates loop detection with lightweight bundle adjustment. Finally, a hierarchical semi-implicit Gaussian structure with LoD-aware densification provides level-of-detail control, helping the system scale without excessive loss of fidelity or efficiency. Together, these components support practical real-time 3D reconstruction across diverse indoor and outdoor settings.

Our main contributions can be summarized as follows:

- • We present ARTDECO, an integrated system that unifies localization, reconstruction, and rendering into a single pipeline, designed to operate robustly across various environments.
- • Notably, we incorporate feed-forward foundation models as modular components for pose estimation, loop closure detection, and dense point prediction. This integration improves localization accuracy and mapping stability while preserving efficiency.
- • We further propose a hierarchical semi-implicit Gaussian representation with a LoD-aware densification strategy, enabling a principled trade-off between reconstruction fidelity and rendering efficiency, critical for large-scale, navigable environments.
- • Extensive indoor and outdoor experiments show that ARTDECO achieves SLAM-level efficiency, feed-forward robustness, and near per-scene optimization quality, validating its effectiveness for practical on-the-fly 3D digitization.

1Beyond the acronym, the name also evokes the Art Deco movement, valued for structure, geometry, and clarity of form. This metaphor reflects our system’s emphasis on structured scene representations.

- 2 RELATED WORK

- 2.1 MULTI-VIEW RECONSTRUCTION AND RENDERING

Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) have attracted significant attention in novel view synthesis (NVS). NeRF and its variants (Barron et al., 2021; 2022; 2023; Zhang et al., 2020; Verbin et al., 2022) model continuous volumetric fields and achieve high-quality image synthesis. However, the reliance on expensive volume rendering and large networks results in long training times and hinders real-time applications. To address these limitations, several works (M¨uller et al., 2022; Xu et al., 2022; Sun et al., 2022) accelerate both training and rendering by introducing hybrid or explicit scene representations. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023b) has shown remarkable progress in high-fidelity reconstruction and real-time rendering by representing scenes with anisotropic Gaussians and leveraging an efficient tile-based rasterizer. Following its introduction, research has largely focused on model compression (Fan et al., 2024; Chen et al., 2025), large-scale scene processing (Ren et al., 2024; Jiang et al., 2025c; Kerbl et al., 2024), and geometry reconstruction (Huang et al., 2024a; Yu et al., 2024c; Gu´edon et al., 2025; Yu et al., 2024a). Despite these advances in novel view synthesis (NVS), most methods assume access to accurate camera poses, typically estimated via Structure-from-Motion (SfM) (Schonberger & Frahm, 2016; Sch¨onberger et al., 2016; Pan et al., 2024), which imposes considerable preprocessing costs for large-scale or in-the-wild captures. To alleviate this reliance, several works (Lin et al., 2021; Fu

- et al., 2024; Lin et al., 2025b) propose joint optimization of camera poses and scene parameters. However, these approaches remain computationally intensive, are sensitive to wide-baseline settings, or still depend on costly post-refinement.

2.2 STREAMING PER-SCENE RECONSTRUCTION

Classical visual SLAM systems (Mur-Artal & Tard´os, 2017; Engel et al., 2017; Teed et al., 2023; Li et al., 2025a) provide online tracking, mapping, and loop closure, but they fall short in producing high-fidelity maps. To overcome this limitation, recent works integrate volumetric rendering techniques into SLAM to enable online NVS. Among these works, NeRF-based SLAM methods (Sucar et al., 2021; Zhu et al., 2022; Zhang et al., 2023; Li et al., 2026) exhibit photorealistic reconstruction but remain computationally expensive due to per-ray volumetric rendering. By contrast, 3DGS has gained traction for SLAM integration thanks to its explicit representation and efficient rendering. Utilizing the differential pipeline of 3DGS, some studies (Matsuki et al., 2024; Hu et al., 2024; Keetha et al., 2024; Yan et al., 2024; Deng et al., 2024; Yu et al., 2025; Li et al., 2025b; Cheng

- et al., 2025; Lin et al., 2025a) directly propagate the gradient from the rendering loss to pose, while others (Yugay et al., 2024; Huang et al., 2024b; Peng et al., 2024b; Ha et al., 2024; Peng et al.,

- 2.3 FEED-FORWARD MODELS

- 2024a; Tianci Wen, 2025; Sandstr¨om et al., 2025) leverage traditional SLAM Modules to provide accurate pose. However, in monocular setting these systems often struggle to balance robustness, accuracy, and runtime efficiency. Recently, On-the-fly NVS (Meuleman et al., 2025b) has shown that GPU-friendly mini-bundle adjustment with incremental 3DGS updates can enable interactive reconstruction, though robustness on casual unposed videos remains limited.

Pretrained on large-scale datasets, recent feed-forward models (Wang et al., 2025a; 2024) reconstruct 3D scenes directly, avoiding per-scene optimization. These approaches can be divided into pose-aware and pose-free methods. Pose-aware models take images together with camera poses as input, enabling rapid reconstruction (Charatan et al., 2024; Chen et al., 2024; Zhang et al., 2024; Jiang et al., 2025a). Pose-free models, in contrast, perform fully end-to-end reconstruction from raw images alone, typically representing scenes with either point maps (Wang et al., 2024; Leroy

- et al., 2024b; Wang et al., 2025a;b; Murai et al., 2024) or 3DGS (Jiang et al., 2025b). Notably, these feed-forward methods offer robustness across diverse scenarios, remove the need for preprocessing, and allow fast inference suitable for interactive use. However, they generally achieve lower accuracy than per-scene optimized methods, and face challenges with maintaining global consistency, handling high-resolution inputs, and processing long sequences.

(b) Backend

(a) Frontend Input frames

Key Frame List

MatchingRatio<𝛕k &PixelDistance<𝛕m

|[Figure 25]<br><br>[Figure 26]<br><br>[Figure 27]<br><br>⋯<br><br>[Figure 28]<br><br>[Figure 29]|
|---|

[Figure 30]

|[Figure 31]<br><br>Common Frame|
|---|

[Figure 32]

[Figure 33]

⋯

Bundle Adjustment Add Key Frame

Case1 : No Loop Closure

MatchingRatio>𝛕k

|[Figure 34]|
|---|

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

|[Figure 39]<br><br>[Figure 40]|
|---|

Add edge

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

Key Frame

###### Matching module

matching module

⋯⋯⋯

Case2 : Loop Closure

|[Figure 45]|
|---|

[Figure 46]

[Figure 47]

[Figure 48]

pcd

|[Figure 49]| |pose &<br><br>[Figure 50]|
|---|---|---|
|MatchingRatio<𝛕k&PixelDistance>𝛕m|[Figure 51]<br><br>Mapper Frame| |

Add edge

Loop Closure

M

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

## ⋯

Loop-detection module

Add long edge

- Figure 2: Frontend and backend modules. (a) Frontend: Images are captured from the scene and streamed into the front-end part. Each incoming frame is aligned with the latest keyframe using a matching module to compute pixel correspondences. Based on the correspondence ratio and pixel displacement, the frame is classified as a keyframe, a mapper frame, or a common frame. The selected frame, along with its pose and point cloud, is then passed to the back-end. (b) Backend: For each new keyframe, a loop-detection module evaluates its similarity with previous keyframes. If a loop is detected, the most relevant candidates are refined and connected in the factor graph; otherwise, the keyframe is linked only to recent frames. Finally, global pose optimization is performed with Gauss–Newton, and other frames are adjusted accordingly. We instantiate the matching module with MASt3R (Leroy et al., 2024b) and the loop-detection module with π3 (Wang et al., 2025b).

- 3 METHOD

We aim to recover a high-fidelity static 3D scene together with the corresponding camera poses from a monocular image sequence. Given a sequence of monocular RGB frames {Ii}Ni=1, with or without known camera intrinsics K ∈ R3×3, we estimate the camera poses {Ri | ti}Ni=1 associated with each image, as well as a set of Gaussian primitives {Gj}Mj=1 that compactly represent the 3D scene. By default, we assume the scene is static and rigid, and that all geometric information is inferred purely from monocular cues without external sensors.

As illustrated in Fig. 2 and 3, ARTDECO processes the sequence in a streaming SLAM-style pipeline consisting of three modules: frontend, backend, and mapping. (1) The frontend estimates relative poses and categorizes frames into common, mapping, or keyframes (Sec. 3.1). (2) The backend refines keyframe poses through loop closure and global bundle adjustment (Sec. 3.2). (3) Finally, image-wise pointmaps initialize 3D Gaussians, which are incrementally optimized in the mapping module (Sec. 3.3).

- 3.1 FRONTEND MODULE

For each input frame, the frontend estimates its pose relative to the latest keyframe and categorizes it as common, mapping, or keyframe. We assume a pinhole camera with fixed intrinsics and a shared optical center. If the focal length is unknown, it is initialized from the first kf GeoCalib (Veicht et al., 2024) estimates and jointly refined during pose estimation.

Pose Estimation. MASt3R (Leroy et al., 2024a) serves as our matching module, a two-view reconstruction and matching prior, to improve camera tracking and focal length estimation. Following MASt3R-SLAM (Murai et al., 2024), we obtain frame-wise pointmaps, their confidence scores, and pixel correspondences between the current frame and the latest keyframe. The 3D points from the current frame are projected into the keyframe image plane, and the relative pose TKC ∈ SIM(3) is estimated by minimizing reprojection residuals with a Gauss–Newton solver. Since MASt3R predictions are less stable near object boundaries, we weight residuals by per-point uncertainty. For each point xc in the current frame, we estimate a local covariance Σc ∈ R3×3 from neighbors within radius δ. We then project Σc to the current keyframe’s measurement space, which is used to filter out unreliable re-projection residuals. Besides, if the focal length is not provided, it is jointly optimized along with the relative pose. Further derivations are given in A.2.

Keyframe Selection. After pose estimation, each frame is categorized as a common frame, mapper frame, or keyframe. A keyframe is created when the number of valid correspondences with the latest keyframe falls below a threshold τk, following standard SLAM practice. Keyframes are passed to the backend for pose refinement and to the mapping module for reconstruction. A mapper frame is selected when the frame provides sufficient parallax for reliable multi-view reconstruction. We compute the pixel displacement between the current frame and the latest keyframe; if the 70th percentile exceeds τm, the frame is promoted to a mapper frame. Mapper frames are first processed by the backend to compute pointmap confidence and are then used in the mapping module to initialize new 3D Gaussians. A common frame does not meet either the keyframe or mapper criteria and is therefore used only to refine existing scene details, without introducing new structure; its role will be further elaborated in later sections.

- 3.2 BACKEND

The backend processes keyframes from the frontend to maintain a globally consistent scene and camera trajectory. For each incoming keyframe, it evaluates correlations with earlier ones, builds a factor graph over the most relevant candidates, and performs global optimization to enforce multiview consistency. In addition, it estimates the confidence of keyframe and mapper pointmaps, which are later used to initialize 3D Gaussian in the mapping module.

Loop Closure and Global Bundle Adjustment. Given a new keyframe, the backend first updates the factor graph by connecting it to related frames, and then performs a PnP-based global bundle adjustment (BA) to refine poses, as illustrated in Fig. 2.(b). If a loop closure is detected, the current keyframe is linked to its three most relevant predecessors; otherwise, it is connected only to the latest keyframe. Loop detection is initially performed using the Aggregated Selective Match Kernel (ASMK). A loop is declared when a previous keyframe has an ASMK score above a threshold τloop and is at least klopp keyframes apart. To increase robustness against weak correspondences and noisy inputs, we further leverage the 3D foundation model π3 (Wang et al., 2025b) as our loop-detection module. Specifically, the current frame and the top Na candidates from ASMK are processed by π3 to produce pointmaps, from which we select the three most geometrically consistent keyframes based on angular error following (Murai et al., 2024). These are then connected to the factor graph, yielding more reliable loop closures and reducing drift. More details can be found in A.5.

Pointmap Confidence. We estimate pointmap confidence using reprojection error rather than relying on the confidence values predicted by MASt3R. When a mapper frame or keyframe is processed, its pointmap is projected onto the Nc previous keyframes with the highest ASMK scores. For each point, we compute the reprojection errors across the Nc keyframes, average them to obtain e¯, and define the confidence score as C = 1 if e¯ ≤ εc, and C = e¯−ε1

c+1 otherwise, where εc is a predefined threshold. This reprojection-based confidence provides a more reliable measure of geometric consistency across frames.

- 3.3 MAPPING MODULE

The mapping module reconstructs the 3D Gaussian scene from incoming frames, their estimated poses, and pointmaps, as illustrated in Fig. 3. Unlike prior 3DGS-based SLAM methods that rely only on keyframes, we leverage all frames to maximize the use of captured information: keyframes and mapper frames introduce new Gaussians, while common frames refine existing ones. This design enriches both visual and geometric details, which are critical not only for accurate reconstruction but also for high-fidelity rendering. Moreover, we introduce a hierarchical Gaussian structure with LoD-aware control. LoD is essential for scalable scene modeling, particularly in large-scale, navigable spaces where SLAM-based applications require consistent detail at varying viewing distances. By combining dense supervision from all frame types with principled level-of-detail management, our mapping module improves fidelity of both reconstructed geometry and rendered views, while maintaining computational efficiency.

Probabilistic Selection for 3D Gaussian Insertion. When a mapper frame or keyframe arrives from the backend, we determine where to initialize new 3D Gaussians. To avoid redundancy, Gaussians are inserted only in regions that require refinement, rather than at every pixel, guided by imagelevel priors inspired by (Meuleman et al., 2025b). We prioritize high-frequency regions and poorly reconstructed areas by computing an insertion probability at each pixel (u,v) using the Laplacian of

[Figure 59]

[Figure 60]

[Figure 61]

Incoming frames (Pos,SHs,opacity,scale,local_feat,vid)

Lgs, Lmono, Lscale

[Figure 62]

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]|
|---|

[Figure 67]

Backend Pose

- LOD 0

[Figure 68]

- LOD 1

[Figure 69]

- LOD 2

[Figure 70]

- LOD 3 Cluster

[Figure 71]

[Figure 72]

[Figure 73]

|[Figure 74]|
|---|

|[Figure 75]|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

|[Figure 79]|
|---|

|[Figure 80]|
|---|

|[Figure 81]|
|---|

[Figure 82]

[Figure 83]

###### ⋯

[Figure 84]

Common Frames

Mapper & Keyframes

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

|[Figure 90]|
|---|

|[Figure 91]|
|---|

|[Figure 92]|
|---|

|[Figure 93]|
|---|

|[Figure 94]|
|---|

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

Feat ← (global_feat, local_feat) Scale ← scale * MLPs (Feat) Rotation ← MLPr (Feat)

[Figure 101]

[Figure 102]

Coarse

[Figure 103]

✕ 1/8

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

✕ 1/4

d > 2*dmax

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

✕ 1/2

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

✕ 1

[Figure 126]

[Figure 127]

Fine

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Multi-resolution images Neural GS rasterization

Laplace norm Add New Gaussians

- Figure 3: Mapping process. When a keyframe or mapper frame arrives from the backend, new Gaussians are added to the scene. Multi-resolution inputs are analyzed with the Laplacian of Gaussian (LoG) operator to identify regions that require refinement, and new Gaussians are initialized at the corresponding monocular depth positions in the current view. Common frames are not used to add Gaussians but contribute through gradient-based refinement. Each primitive stores position, spherical harmonics (SH), base scale, opacity, local feature, dmax, and voxel index vid. For rendering, the dmax attribute determines whether a Gaussian is included at a given viewing distance, enabling consistent level-of-detail control. Gaussian (LoG) operator (Haralock & Shapiro, 1991):

Pa(u,v) = max min(∥∇2(Gσ) ∗ I(u,v)∥,1) − min(∥∇2(Gσ) ∗ I˜(u,v)∥,1),0 , (1)

where I and I˜are the ground-truth and rendered images, and Gσ is a Gaussian kernel with standard deviation σ. A new Gaussian is added when Pa(u,v) exceeds the threshold τa.

Gaussian Primitive Initialization. After identifying candidate pixels, we initialize the corresponding 3D Gaussians. Each Gaussian is parameterized by its center µ, spherical harmonics (SH), opacity α, base scale Sb, individual feature fl, and voxel index vid. The µ and SH0 are initialized from the pointmap and pixel color, while opacity is set to 0.2·C(u,v) to down-weight low-confidence regions, where C(u,v) is the confidence score calculated in backend. Following (Meuleman et al., 2025b; Wu

- et al., 2025; Yu et al., 2024b), the base scale at pixel (u,v) is defined as:

dis′ f

1 2 min(∥(∇2Gσ) ∗ I(u,v)∥,1)

, s′ =

, (2)

Sb =

where di is the distance from the Gaussian center to the camera and f is the focal length. Here, s′ represents an image-space scale, i.e., the expected distance to the nearest neighbor under a local

- 2D Poisson process of intensity min(∥(∇2Gσ) ∗ I(u,v)∥,1), (Clark & Evans, 1954). To ensure smoother reconstruction, we further refine scale and initialize rotation with two MLPs:

S = Sb · MLPs(fr ⊕ fl), R = MLPr(fr ⊕ fl), (3) where ⊕ denotes concatenation, fl is an individual feature initialized as zero, and fr is a region feature encoding local voxel context. We voxelize the 3D space with cell size ϵ; when a new Gaussian is added to a voxel, the corresponding voxel-wise feature is initialized as zero and indexed by vid. This hybrid region–individual design promotes global consistency of the Gaussian field while preserving local distinctiveness.

Levels of Detail Design. To support smooth navigation in large 3D scenes, we organize Gaussians into multiple levels of detail (LoD). Each Gaussian is assigned a level l ∈ N+ with l < L, where level 0 denotes the finest resolution and level L−1 the coarsest. At initialization, a Gaussian at level l corresponds to a patch of 22l pixels in the original image (e.g., level 0 corresponds to one pixel). We progressively downsample the input frame L−1 times and initialize Gaussians from both the downsampled and original images. All Gaussian parameters follow the initialization described earlier, except that (i) the base scale is weighted by 1.42l, and (ii) each Gaussian is assigned a distance-dependent parameter dmax = d · 22l, where d is the distance from the Gaussian center to the camera. During rendering, a Gaussian is included if dr ≤ dmax, excluded if dr > 2dmax, and smoothly faded out for dmax < dr ≤ 2dmax by interpolating its opacity as α = (d − dmax)/dmax. This distance-aware LoD design suppresses flickering and maintains stable rendering quality across scales while preserving efficiency.

Training Strategy. To balance efficiency and reconstruction quality, we adopt a staged training scheme. For streaming input, new Gaussians are initialized and the scene is optimized for K iterations whenever a mapper frame or keyframe arrives, while common frames trigger only K/2 iterations without inserting new Gaussians. Following (Meuleman et al., 2025b; Wu et al., 2025), training frames are sampled with a 0.2 probability from the current frame and 0.8 from past frames to mitigate local overfitting. After processing the sequence in a streaming manner, we run a global optimization over all frames, giving higher sampling probabilities to those with fewer historical updates. Finally, camera poses are optimized jointly with Gaussian parameters, with gradients on positions and rotations propagated to poses, consistent with common practice in on-the-fly reconstruction.

- 4 EXPERIMENTS

- 4.1 EXPERIMENTAL SETUP

Datasets and Metrics. We evaluate on diverse indoor and outdoor benchmarks. Indoor datasets include 11 TUM scenes (Sturm et al., 2012), 14 scenes from ScanNet++ (Yeshwanth et al., 2023), 8 scenes from VR-NeRF (Xu et al., 2023), and 6 scenes from ScanNet (Dai et al., 2017), with sequence lengths ranging from 32–5577 image frames. Outdoor datasets include 8 KITTI scenes (Geiger et al., 2013), 9 Waymo scenes (Sun et al., 2020) (both following S3PO-SLAM (Cheng et al., 2025)), 5 scenes from Fast-livo2 (Zheng et al., 2024), and 1 scene from MatrixCity (Li et al., 2023), with lengths 200–1363 frames per trajectory. Reconstruction is evaluated with PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018); pose accuracy with Absolute Trajectory Error (ATE) RMSE; and system efficiency with FPS.

Baselines. We evaluate against two categories of state-of-the-art methods. For reconstruction quality, we consider 3D Gaussian Splatting approaches, including OnTheFly-NVS (Meuleman et al., 2025a)), LongSplat (Lin et al., 2025a), S3PO-GS (Cheng et al., 2025), SEGS-SLAM (Tianci Wen, 2025), MonoGS (Matsuki et al., 2024)). For pose estimation, in addition to the aforementioned 3DGS-based SLAM methods, we benchmark against several state-of-the-art SLAM systems, including MASt3R-SLAM (Murai et al., 2024), DPV-SLAM (Lipson et al., 2024), DROID-SLAM (Teed & Deng, 2021), and Go-SLAM (Zhang et al., 2023).

Implementation Details. Experiments are run on a desktop with an Intel Core i9-14900K CPU and NVIDIA RTX 4090 GPU. Following standard Novel View Synthesis practices, every 8th frame is held out for evaluation: these frames are excluded from mapping but their poses are estimated and optimized for evaluation. In the frontend, we set τk = max(0.333 · W,30), where W is the image width. In the backend, we use Na = min(23,Nc) candidate keyframes, where Nc is the number of available candidates. If Na < 11, loop-detection modules are disabled, and the top three ASMKscoring keyframes are directly selected to connect in the factor graph. For pointmap confidence, we fix εc = 3. During mapping, 3D Gaussians are organized into 4 LOD levels by setting L = 4.

- 4.2 COMPARISON

Reconstruction Results Analysis. Tab. 1 reports reconstruction results on eight indoor and outdoor benchmarks. Our system achieves state-of-the-art quality, particularly on challenging datasets such as TUM and ScanNet with structural complexity, motion blur, and noise. On higher-quality datasets like VR-NeRF and ScanNet++, where scenes feature diverse multi-scale visuals, all methods improve, yet ARTDECO still delivers the best performance. Outdoor evaluation covers large-scale free-motion captures (Fast-LIVO2) and forward-facing driving datasets (Waymo, KITTI, MatrixCity). ARTDECO consistently outperforms baselines, demonstrating robustness to scale variation. Qualitative comparisons (Fig. 4) show that ARTDECO, enabled by its multi-level Gaussian primitive design, captures fine details, large-scale structures, and high-fidelity geometry within a compact representation. Additional results are provided in Tabs. 4 - 24, Fig.5 in A.6.

Tracking Results Analysis. Tab. 2 summarizes the tracking performance on indoor and outdoor benchmarks. With loop closure and covariance-matrix filtering, ARTDECO achieves markedly higher localization accuracy than other 3DGS-based systems on challenging multi-scale indoor datasets (TUM, ScanNet++). On outdoor datasets such as Waymo, it also delivers competitive performance. Further results on TUM (Second part of Tab. 2) demonstrate that ARTDECO consistently outperforms state-of-the-art non-3DGS SLAM methods, confirming its superior localization

SEGS-SLAM S3PO-GS OnTheFly-NVS LongSplat Ours GT

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

| |
|---|

Figure 4: Qualitative comparisons against popular on-the-fly reconstruction baselines across diverse 3D scene datasets. ARTDECO consistently preserves high-quality rendering details in complex and diverse environments, particularly in the regions highlighted with colored rectangles.

- Table 1: Rendering comparisons against baselines across indoor and outdoor datasets. We report visual quality metrics, average running time.

Indoor-dataset ScanNet++ ScanNet TUM VR-NeRF Training Method PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ Time↓

MonoGS 16.71 0.682 0.600 18.87∗ 0.780∗ 0.629∗ 17.78 0.602 0.573 13.88 0.560 0.420 14.08 min S3PO-GS 22.94 0.820 0.355 20.14 0.797 0.558 19.62 0.656 0.466 12.43 0.642 0.497 41.25 min SEGS-SLAM - - - 19.73∗ 0.839∗ 0.365∗ 19.69∗ 0.743∗ 0.307∗ 31.62∗ 0.896∗ 0.232∗ 10.84 min

OnTheFly-NVS 18.01 0.761 0.386 15.36 0.708 0.494 19.72 0.719 0.380 27.30 0.872 0.310 2.29 min LongSplat 24.94∗ 0.827∗ 0.260∗ 19.27∗ 0.754∗ 0.404∗ 25.09 0.804 0.272 25.74∗ 0.832∗ 0.321∗ 442.96 min

###### Ours 29.12 0.918 0.167 24.10 0.865 0.271 26.18 0.850 0.224 28.57 0.895 0.242 5.33 min

Outdoor-dataset KITTI Waymo Fast-LIVO2 MatrixCity Training Method PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓ Time ↓

MonoGS 14.56 0.489 0.767 19.34 0.752 0.627 18.87 0.598 0.699 19.36 0.593 0.736 16.52 min S3PO-GS 19.97 0.645 0.410 27.28 0.865 0.352 21.51 0.684 0.445 21.76 0.661 0.584 34.89 min SEGS-SLAM 14.03 0.463 0.488 19.01∗ 0.698∗ 0.502∗ 24.58∗ 0.773∗ 0.307∗ 25.57 0.784 0.366 8.75 min OnTheFly-NVS 16.89 0.579 0.471 25.53 0.820 0.360 18.76 0.618 0.497 21.36 0.687 0.451 0.74 min LongSplat 16.86 0.532 0.447 25.61 0.795 0.326 26.37 0.792 0.276 - - - 313.60 min Ours 23.17 0.765 0.299 28.75 0.880 0.276 29.54 0.894 0.158 25.62 0.790 0.327 6.58 min

*: majority of scenes successful; –: majority failed; Only compare fully successful methods.

capability. Per-scene metrics, additional tracking results and qualitative trajectory comparisons are provided in Tabs. 25–29, Figs. 7–8 in A.6.

Runtime Analysis. We compare runtime across 3DGS-based methods on both indoor and outdoor datasets (Tab. 1). ARTDECO runs faster than all except OnTheFly-NVS, with its extra time cost primarily from pose estimation, a trade-off justified by the superior pose accuracy in Tab. 2.

- Table 2: Tracking comparisons. For tracking evaluation, we compare against SLAM- and SFMbased 3D reconstruction methods on indoor and outdoor datasets, as well as state-of-the-art SLAM systems on the TUM dataset (Following MASt3R-SLAM, 9 scenes from TUM fr1). Our method consistently achieves lower ATE RMSE.

Dataset MonoGS S3PO-GS SEGS-SLAM MASt3R-SLAM OnTheFly-NVS LongSplat Ours ScanNet++ 1.217 0.632 0.245 0.025 0.891 0.602 0.018 TUM 0.244 0.117 0.073∗ 0.031 - - 0.025 Waymo 7.370 1.236 - - 3.118 4.956 1.213

Metric ORB-SLAM3 DPV-SLAM++ DROID-SLAM Go-SLAM MASt3R-SLAM Ours ATE RMSE - 0.054 0.038 0.035 0.030 0.028

*: majority of scenes successful; –: majority failed; Only compare fully successful methods.

- Table 3: Quantitative results on ablation studies. We separately listed the rendering metrics and ATE RMSE on ScanNet++ dataset for each ablation described in Sec. 4.3

Front&Backend Full w/ SLAM (MASt3R → π3) w/ Loop (π3 → vggt) w/o loop w/ dense key frame ATE RMSE 0.018 0.374 0.096 0.057 0.094

Mapper Full w/o level-of-detail w/o implicit structure w/o global feat w/o mapper frame w/o common frame

PSNR 29.12 28.13 28.54 27.95 26.38 27.20 SSIM 0.918 0.912 0.914 0.910 0.898 0.904 LPIPS 0.167 0.180 0.175 0.197 0.229 0.211

- 4.3 ABLATION STUDY

Ablation on Localization. We analyze the impact of backbone choice, loop closure, and frame categorization strategy on localization, as summarized in Tab. 3. We first ablate the feed-forward model used in the frontend and backend by replacing MASt3R (pairwise inference) with π3 (multiimage inference). Although π3 is trained on more diverse data, it lacks metric-scale capability and performs worse under varying viewpoints. In contrast, MASt3R better preserves consistent object proportions, resulting in more accurate pose estimation. Next, we ablate the loop-closure module by disabling it, which leads to a significant degradation in localization accuracy. Finally, we ablate the frame categorization strategy. Here, MF denotes mapper frames and KF denotes keyframes. Using both MFs and KFs (track w/ MF&KF) for inference provides additional temporal information, but unexpectedly reduces pose accuracy. This is because 3D foundation models often struggle with small-parallax inputs, producing ghosting and blur that corrupt point clouds and feature correspondences when the input sequence is overly dense.

Ablation on Reconstruction. We further ablate the effects of mapper frames, level-of-detail, and structural Gaussians on reconstruction, as shown in Tab. 3. MFs add richer multi-view constraints, while LoD and structural Gaussians yield more compact, regularized representations, together improving reconstruction fidelity and rendering quality.

- 5 LIMITATIONS

While ARTDECO achieves strong reconstruction and localization, it has several limitations. First, it partly depends on feed-forward 3D foundation models for correspondence and geometry, which, despite enabling fast and scalable inference, reduce robustness under noise, blur, or lighting changes, and suffer when inputs fall outside the training distribution. Second, the system assumes consistent illumination and sufficient parallax; violations such as low-texture surfaces, repetitive structures, or near-degenerate trajectories can cause drift or artifacts. These challenges suggest future work on incorporating uncertainty estimation, adaptive model selection, and stronger priors to improve generalization and reliability in real-world settings.

- 6 CONCLUSION

In this work, we present ARTDECO, a unified framework that advances on-the-fly 3D reconstruction from monocular image sequences. Beyond achieving strong results on standard indoor and outdoor benchmarks, ARTDECO demonstrates that feed-forward priors and structured Gaussian representations can be effectively combined within a single system to deliver both accuracy and efficiency. We see ARTDECO as a step toward practical large-scale deployment of real-to-sim pipelines, with promising applications in AR/VR, robotics, and digital twins.

REFERENCES

Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5855–5864, 2021.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 5470–5479, 2022.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf: Anti-aliased grid-based neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19697–19705, 2023.

David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, 2024.

Yihang Chen, Qianyi Wu, Mengyao Li, Weiyao Lin, Mehrtash Harandi, and Jianfei Cai. Fast feedforward 3d gaussian splatting compression. In The Thirteenth International Conference on Learning Representations, 2025.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, TatJen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. arXiv preprint arXiv:2403.14627, 2024.

Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and Hao Wang. Outdoor monocular slam with global scale-consistent 3d gaussian pointmaps. arXiv preprint arXiv:2507.03737, 2025.

Philip J Clark and Francis C Evans. Distance to nearest neighbor as a measure of spatial relationships in populations. Ecology, 35(4):445–453, 1954.

Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.

Tianchen Deng, Yaohui Chen, Leyan Zhang, Jianfei Yang, Shenghai Yuan, Danwei Wang, and Weidong Chen. Compact 3d gaussian splatting for dense visual slam. arXiv preprint arXiv:2403.11247, 2024.

Jakob Engel, Vladlen Koltun, and Daniel Cremers. Direct sparse odometry. IEEE transactions on pattern analysis and machine intelligence, 40(3):611–625, 2017.

Zhiwen Fan, Kevin Wang, Kairun Wen, Zehao Zhu, Dejia Xu, Zhangyang Wang, et al. Lightgaussian: Unbounded 3d gaussian compression with 15x reduction and 200+ fps. Advances in neural information processing systems, 37:140138–140158, 2024.

Yang Fu, Sifei Liu, Amey Kulkarni, Jan Kautz, Alexei A Efros, and Xiaolong Wang. Colmap-free 3d gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20796–20805, 2024.

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237, 2013.

Antoine Gu´edon, Diego Gomez, Nissim Maruani, Bingchen Gong, George Drettakis, and Maks Ovsjanikov. Milo: Mesh-in-the-loop gaussian splatting for detailed and efficient surface reconstruction. ACM Transactions on Graphics, 2025. URL https://anttwo.github.io/ milo/.

Seongbo Ha, Jiung Yeon, and Hyeonwoo Yu. Rgbd gs-icp slam. In European Conference on Computer Vision, 2024.

Robert M Haralock and Linda G Shapiro. Computer and robot vision. Addison-Wesley Longman Publishing Co., Inc., 1991.

Jiarui Hu, Xianhao Chen, Boyin Feng, Guanglin Li, Liangjing Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Cg-slam: Efficient dense rgb-d slam in a consistent uncertainty-aware 3d gaussian field. In European Conference on Computer Vision, 2024.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024a. doi: 10.1145/3641519.3657428.

Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous localization and photorealistic mapping for monocular, stereo, and rgb-d cameras. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024b.

Hanwen Jiang, Hao Tan, Peng Wang, Haian Jin, Yue Zhao, Sai Bi, Kai Zhang, Fujun Luan, Kalyan Sunkavalli, Qixing Huang, and Georgios Pavlakos. Rayzer: A self-supervised large view synthesis model. 2025a.

Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views. arXiv preprint arXiv:2505.23716, 2025b.

Lihan Jiang, Kerui Ren, Mulin Yu, Linning Xu, Junting Dong, Tao Lu, Feng Zhao, Dahua Lin, and Bo Dai. Horizon-gs: Unified 3d gaussian splatting for large-scale aerial-to-ground scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 26789–26799, June 2025c.

Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, Gengshan Yang, Sebastian Scherer, Deva Ramanan, and Jonathon Luiten. Splatam: Splat, track & map 3d gaussians for dense rgb-d slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023a.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D Gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023b.

Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics, 43(4), July 2024. URL https://repo-sam. inria.fr/fungraph/hierarchical-3d-gaussians/.

Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding Image Matching in 3D with MASt3R, 2024a.

Vincent Leroy, Yohann Cabon, and Jerome Revaud. Grounding image matching in 3d with mast3r,

- 2024b.

Guanghao Li, Yu Cao, Qi Chen, Xin Gao, Yifan Yang, and Jian Pu. Papl-slam: Principal axisanchored monocular point-line slam. IEEE Robotics and Automation Letters, 2025a.

Guanghao Li, Qi Chen, Sijia Hu, Yuxiang Yan, and Jian Pu. Constrained gaussian splatting via implicit tsdf hash grid for dense rgb-d slam. IEEE Transactions on Artificial Intelligence, 2025b.

Guanghao Li, Qi Chen, Yuxiang Yan, and Jian Pu. Ec-slam: Effectively constrained neural rgb-d slam with tsdf hash encoding and joint optimization. Pattern Recognition, 170:112034, 2026.

Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3205–3215, 2023.

Chen-Hsuan Lin, Wei-Chiu Ma, Antonio Torralba, and Simon Lucey. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5741–5751, 2021.

Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. Longsplat: Robust unposed 3d gaussian splatting for casual long videos. arXiv preprint arXiv:2508.14041, 2025a.

Chin-Yang Lin, Cheng Sun, Fu-En Yang, Min-Hung Chen, Yen-Yu Lin, and Yu-Lun Liu. LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos, August 2025b.

Lahav Lipson, Zachary Teed, and Jia Deng. Deep patch visual slam. In European Conference on Computer Vision, pp. 424–440. Springer, 2024.

Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, and George Drettakis. On-thefly reconstruction for large-scale novel view synthesis from unposed images. ACM Transactions on Graphics (TOG), 44(4):1–14, 2025a.

Andreas Meuleman, Ishaan Shah, Alexandre Lanvin, Bernhard Kerbl, and George Drettakis. Onthe-fly Reconstruction for Large-Scale Novel View Synthesis from Unposed Images. ACM Transactions on Graphics, 44(4), 2025b.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022.

Ra´ul Mur-Artal and Juan D. Tard´os. ORB-SLAM2: an open-source SLAM system for monocular, stereo and RGB-D cameras. IEEE Transactions on Robotics, 33(5):1255–1262, 2017. doi: 10. 1109/TRO.2017.2705103.

Riku Murai, Eric Dexheimer, and Andrew J. Davison. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. arXiv preprint, 2024.

Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L Sch¨onberger. Global structure-frommotion revisited. In European Conference on Computer Vision, pp. 58–77. Springer, 2024.

Chensheng Peng, Chenfeng Xu, Yue Wang, Mingyu Ding, Heng Yang, Masayoshi Tomizuka, Kurt Keutzer, Marco Pavone, and Wei Zhan. Q-slam: Quadric representations for monocular slam. arXiv preprint arXiv:2403.08125, 2024a.

Zhexi Peng, Tianjia Shao, Yong Liu, Jingke Zhou, Yin Yang, Jingdong Wang, and Kun Zhou. Rtgslam: Real-time 3d reconstruction at scale using gaussian splatting. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–11, 2024b.

Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octreegs: Towards consistent real-time rendering with lod-structured 3d gaussians. arXiv preprint arXiv:2403.17898, 2024.

Erik Sandstr¨om, Ganlin Zhang, Keisuke Tateno, Michael Oechsle, Michael Niemeyer, Youmin Zhang, Manthan Patel, Luc Van Gool, Martin Oswald, and Federico Tombari. Splat-slam: Globally optimized rgb-only slam with 3d gaussians. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1680–1691, 2025.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016.

Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise View Selection for Unstructured Multi-View Stereo. In European Conference on Computer Vision (ECCV), 2016.

J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pp. 573–580. IEEE, 2012.

Edgar Sucar, Shikun Liu, Joseph Ortiz, and Andrew J Davison. imap: Implicit mapping and positioning in real-time. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6229–6238, 2021.

Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5459–5469, June 2022.

Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2446–2454, 2020.

Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. arXiv preprint arXiv:2412.06974, 2024.

Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569, 2021.

Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual odometry. Advances in Neural Information Processing Systems, 36:39033–39051, 2023.

Yongchun Fang Tianci Wen, Zhiang Liu. Segs-slam: Structure-enhanced 3d gaussian splatting slam with appearance embedding. arXiv preprint arXiv:2501.05242, 2025.

Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. GeoCalib: Single-image Calibration with Geometric Optimization. In ECCV, 2024.

Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5481–5490. IEEE, 2022.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025a.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025b. URL https://arxiv.org/abs/2507.13347.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600– 612, 2004.

Songyin Wu, Zhaoyang Lv, Yufeng Zhu, Duncan Frost, Zhengqin Li, Ling-Qi Yan, Carl Ren, Richard Newcombe, and Zhao Dong. Monocular online reconstruction with enhanced detail preservation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–11, 2025.

Linning Xu, Vasu Agrawal, William Laney, Tony Garcia, Aayush Bansal, Changil Kim, Samuel Rota Bul`o, Lorenzo Porzi, Peter Kontschieder, Aljaˇz Boˇziˇc, Dahua Lin, Michael Zollh¨ofer, and Christian Richardt. VR-NeRF: High-fidelity virtualized walkable spaces. In SIGGRAPH Asia Conference Proceedings, 2023. doi: 10.1145/3610548.3618139. URL https://vr-nerf. github.io.

Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5438–5448, June 2022.

Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting, April 2024.

Botao Ye, Sifei Liu, Haofei Xu, Li Xueting, Marc Pollefeys, Ming-Hsuan Yang, and Peng Songyou. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the International Conference on Computer Vision (ICCV), 2023.

Mulin Yu, Tao Lu, Linning Xu, Lihan Jiang, Yuanbo Xiangli, and Bo Dai. Gsdf: 3dgs meets sdf for improved rendering and reconstruction. arXiv preprint arXiv:2403.16964, 2024a.

Sicheng Yu, Chong Cheng, Yifan Zhou, Xiaojun Yang, and Hao Wang. Rgb-only gaussian splatting slam for unbounded outdoor scenes. arXiv preprint arXiv:2502.15633, 2025.

Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Aliasfree 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19447–19456, 2024b.

Zehao Yu, Torsten Sattler, and Andreas Geiger. Gaussian opacity fields: Efficient adaptive surface reconstruction in unbounded scenes. ACM Transactions on Graphics, 2024c.

Vladimir Yugay, Yue Li, Theo Gevers, and Martin R Oswald. Gaussian-slam: Photo-realistic dense slam with gaussian splatting. arXiv preprint arXiv:2312.10070, 2024.

Kai Zhang, Gernot Riegler, Noah Snavely, and Vladlen Koltun. Nerf++: Analyzing and improving neural radiance fields. arXiv preprint arXiv:2010.07492, 2020.

Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. European Conference on Computer Vision, 2024.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.

Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21936–21947, 2025.

Youmin Zhang, Fabio Tosi, Stefano Mattoccia, and Matteo Poggi. Go-slam: Global optimization for consistent 3d instant reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3727–3737, 2023.

Chunran Zheng, Wei Xu, Zuhao Zou, Tong Hua, Chongjian Yuan, Dongjiao He, Bingyang Zhou, Zheng Liu, Jiarong Lin, Fangcheng Zhu, et al. Fast-livo2: Fast, direct lidar-inertial-visual odometry. IEEE Transactions on Robotics, 2024.

Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, and Marc Pollefeys. Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

A SUPPLEMENTARY MATERIAL

We organize the supplementary material as follows:

- • Sec. A.1 describes implementation & evaluation protocol.
- • Sec. A.2 derives Jacobians and covariance transformation used in the Frontend / Backend.
- • Sec. A.3 presents additional experimental details, including π3-based multi-frame ablations and their observations.
- • Sec. A.4 expands the Frontend implementation, covering correspondence formation, residual weighting, Gauss-Newton updates, and focal optimization under unknown intrinsics.
- • Sec. A.5describes Backend loop-closure detection (ASMK + multi-frame 3D priors) and global bundle adjustment.
- • Sec. A.6 reports extended experiments, per-scene quantitative metrics, and additional qualitative results.

- A.1 MORE IMPLEMENTATION DETAILS

Evaluation protocol. Prior works evaluate reconstruction quality on different subsets of frames, as each method selects its own keyframes for mapping. This inconsistency leads to slight but systematic metric bias. To ensure a fair and reproducible evaluation, we uniformly sample one frame out of every eight across all sequences as evaluation frames. These frames are excluded from mapping supervision and are used only for pose optimization during inference.

We have re-implemented or modified baseline evaluation scripts accordingly so that all reported metrics in Tables are computed under this identical protocol. This guarantees that differences in performance arise solely from algorithmic design rather than evaluation selection bias.

- A.2 JACOBIAN

Let a 3D point in the current camera coordinate system be x = (X,Y,Z)⊤. Its projection to the image plane with log–depth parameterization is defined as

π(X,Y,Z) =

 

fxX/Z + cx fyY/Z + cy log Z

 , (4)

where (fx,fy) are focal lengths and (cx,cy) are principal point offsets. The differential of this mapping governs how small 3D perturbations propagate into pixel-space.

Jacobian with respect to predicted 3D points. The Jacobian of the projection function w.r.t. the

- 3D coordinates is

  

  . (5)

fx Z 0 −f

xX Z2

0 fZy −fZyY2 0 0 Z1

Jπ(x) =

Given the per-point covariance Σc ∈ R3×3 predicted in the current frame (estimated from local neighborhoods as described in Sec. 3.1), the measurement-space covariance after projection to the keyframe view k is

### Σck = Jπ(xck)Rkc Σc Rkc⊤ Jπ(xck)⊤, (6)

where Rkc is the rotation matrix of the relative pose Tkc between the current frame c and keyframe k. We reject a correspondence if det(Σck) > τ, which effectively discards measurements with high geometric uncertainty while remaining rotation-invariant.

Jacobian with respect to focal length When the focal length is unknown or weakly calibrated, we jointly optimize it with the pose parameters. Let a pixel pc = (uc,vc) in the current frame correspond to the 3D point Pc = (Xc,Yc,Zc)⊤. The point reconstructed in normalized camera coordinates is

 

 , (7)

(uc − cx)/f (vc − cy)/f 1

P′c = Zc

and its transformation to the keyframe k is Pk = Tkc P′c with Tkc = [Rkc, tkc] ∈ Sim(3). The projection back to pixel space is

 

 . (8)

fXk/Zk + cx fYk/Zk + cy log Zk

π(Pk) =

The focal length f influences the projection in two ways: (1) Direct effect: scaling of projected pixel coordinates. (2) Indirect effect: change in the reconstructed 3D point P′c itself, which depends on f through normalization.

The corresponding partial derivatives are

 

 , (9)

Xk/Zk Yk/Zk 0

∂(u,v,log Z) ∂f

=

  ,

  

c−cx) f2

−(u

∂P′c ∂f

∂P′c ∂f

∂Pk ∂f

. (10)

−(vcf−2cy) 0

= Zc

= Tkc

Combining both contributions via the chain rule yields the full Jacobian:

  

  

  

  

  

   +

k

Xk Zk

f

Zk 0 − fX

- −(u

c−cx) f2

- −(v

(Zk)2 0 Zf

∂(u,v,log Z) ∂f

∂(u,v,log Z) ∂Pk

∂Pk ∂f

c−cy) f2

k

Y k Zk

Jπ,f(Pk) =

+

=

k − fY

(Zk)2 0 0 Z1

0

0

direct

indirect

k

(11) This joint derivative allows the optimizer to update focal length coherently with the camera pose.

- A.3 ADDITIONAL EXPERIMENTS

Sec. 4.3 of the main paper evaluates how different architectural and procedural choices affect localization and reconstruction. Here we provide detailed background and rationale for one particularly important ablation: replacing the pairwise correspondence model MASt3R with the multi-frame visual-geometry model π3. The goal is to understand whether stronger 3D priors and longer temporal context can further enhance robustness and global consistency in the pipeline.

Background. MASt3R (Leroy et al., 2024b) predicts two-view correspondences and metric pointmaps with well-calibrated scale but limited temporal context. In contrast, π3 (Wang et al.,

- 2025b) is a permutation-equivariant large-scale geometry model trained on multi-image sets. Its potential advantage lies in leveraging more global spatial constraints and recovering denser, smoother pointmaps across frames. However, because π3 is not explicitly metric-aware, integrating it into a streaming SLAM-style system introduces challenges in maintaining scale and temporal consistency. Below we describe how we adapt π3 to our framework and analyze its effect relative to the baseline.

Frontend with π3 Inference and Keyframe Scheduling. In the original frontend, each incoming frame is matched against the latest keyframe using MASt3R. To enable multi-frame inference, we accumulate k consecutive frames before calling π3. The model jointly estimates pointmaps for all k views and provides dense pixel correspondences. After inference, we perform a local bundle adjustment (LBA) in the Sim(3) group among these k frames. Upon completing this process, we follow the keyframe selection strategy described in Sec. 3.1, and the selected keyframes are then passed to the backend.

After the first π3 inference, subsequent updates reuse the l most recent keyframes as temporal anchors and wait until k−l new frames arrive before triggering the next inference batch. This overlap ensures continuity and smooth transition of geometric priors between windows.

Backend with π3-Based Global Optimization. The backend receives keyframes streamed from the frontend. To incorporate π3 priors, it similarly waits until k keyframes are accumulated, then performs a joint π3 inference to obtain multi-view correspondences among the current and historical keyframes. A global bundle adjustment (GBA) then optimizes all selected poses. Compared with the frontend’s local window, the backend’s inference window integrates non-adjacent keyframes filtered by the ASMK module from MASt3R-SLAM (Murai et al., 2024), providing long-range loop-closure evidence while keeping runtime tractable.

Mapper with Keyframe Reception. The mapping module remains unchanged. The mapper simply receives the keyframes sent by the backend and proceeds with the reconstruction following the process described in Sec. 3.3.

- A.4 DETAILS IN FRONTEND: POSE AND FOCAL OPTIMIZATION

The frontend provides the first stage of the streaming pipeline. We now present a detailed formulation of the weighted Gauss–Newton optimization that simultaneously refines pose and focal length when intrinsics are unknown.

For each input frame, MASt3R predicts dense two-view correspondences and associated 3D points between the current frame c and the latest keyframe k. We denote the correspondence set as

### C = {(pmc ,pmk ,Pmc ,Pmk )}Mm=1, (12)

where pmc ,pmk ∈ R2 are pixel coordinates and Pmc ,Pmk ∈ R3 are the corresponding 3D points expressed in the coordinate systems of the current frame and keyframe, respectively. The relative

transform Tkc = (skc,Rkc,tkc) includes scale, rotation, and translation components. Residual formulation. For each correspondence m, we compose the similarity transform and projection directly into the residual:

pmk − pˆmk log Zkm − log Zˆkm

, Pˆmk = skcRkcPmc + tkc, pˆmk = πK(f)(Pˆmk ), (13)

rm =

where πK(f)(·) denotes the pinhole projection with intrinsic matrix K(f). The objective seeks the optimal parameters that minimize the robust weighted energy

Erob = 21

M

r⊤mWmrobrm, Wmrob = ωmWm. (14)

m=1

Weighting and robustness. Each correspondence is assigned a positive-semidefinite weight matrix Wm ∈ R3×3 derived from measurement covariance, balancing pixel and log-depth residuals. To suppress outliers, we apply a Huber kernel:

 

1, sm ≤ δ,

sm = r⊤mWmrm, (15)

ωm =

δ sm + ε



, sm > δ,

with a small ε > 0 for numerical stability.

Linearization and update. Linearizing Eq. 13 about the current estimate yields the normal equations

J⊤mWmrobJm ∆θ =

J⊤mWmrobrm, (16)

m

m

where Jm = ∂rm/∂θ and θ = {Tkc,f}. For pose-only optimization, the dimensionality is d = 7; when focal refinement is enabled, d = 8. The updates are applied through the exponential map in the Sim(3) Lie group:

Tkc ← expSim(3)(∆ξsim)Tkc, f ← f + ∆f, (17) where ∆ξsim ∈ R7 is the minimal increment of the similarity transform.

- A.5 DETAILS IN BACKEND: LOOP CLOSURE AND GLOBAL BUNDLE ADJUSTMENT

The long-term trajectory accuracy requires closing loops and enforcing multi-view consistency across the entire sequence. The backend of ARTDECO detects loop closures, verifies them with 3D priors, and performs global optimization of all keyframe poses in the Sim(3) group.

Hybrid Loop-Closure Detection. Given a new keyframe Kt, the backend first computes its ASMK similarity to all historical keyframes {Kj}. Candidates with a score greater than 0.005 are retained. If the temporal gap between Kt and the nearest candidate exceeds 10 frames, we assume a potential loop and re-rank the top Na candidates by similarity.

Next, we perform π3 inference jointly on the set {Kt} ∪ CNa

to obtain dense pointmaps in a shared coordinate system. For each candidate keyframe Kj, we compute the angular and depth errors between corresponding 3D points Pmt and Pmj . If the ratio of geometrically consistent correspondences exceeds 0.15, the pair (Kt,Kj) is confirmed as a loop closure and added as an edge in the factor graph.

This two-stage scheme achieves a balance between robustness and efficiency: ASMK rapidly filters potential loops, and π3 provides dense geometric validation resilient in practice.

Global Bundle Adjustment. After loop-closure edges are added, we jointly refine all connected keyframe poses {Twi} = {(si,Ri,ti)} ∈ Sim(3) through global bundle adjustment (GBA). For a correspondence m between frames i and j, let Pmi ∈ R3 be the 3D point in frame i and pmj ∈ R2 the observed pixel in frame j. The transformed and projected quantities are

Pˆmw = siRiPmi + ti, Pˆmj = s−j 1R⊤j (Pˆmw − tj), pˆmj = πK(Pˆmj ). (18) We define the residual vector as

p ˆmj − pmj log Zˆjm − log Zjm

, (19)

rm =

which combines pixel reprojection and log-depth errors. The objective function sums all residuals weighted by their confidence ωm:

EGBA =

### ωm r⊤mrm. (20)

i,j,m

For each correspondence, the projection Jacobian with respect to the 3D point Pˆmj = (Xj,Yj,Zj)⊤ is

  

  . (21)

fx Zj 0 −fxZX2j

∂rm ∂Pˆmj

j

−fZyY2j

0 Zfy

=

j

j

### 0 0 Z1

j

Perturbing poses in Sim(3) by left-multiplicative increments δξi,δξj ∈R7 yields

∂Pˆmj ∂δξi

∂Pˆmj ∂δξj

I3 −[Pˆmw ]× 0⊤ Pmw

= −I3 [Pˆmj ]× , (22)

= s−j 1R⊤j

,

where [·]× denotes the skew-symmetric matrix. By the chain rule,

∂Pˆmj ∂δξi

∂rm ∂δξi

∂rm ∂Pˆmj

∂rm ∂δξj

∂rm ∂Pˆmj

=

,

=

∂Pˆmj ∂δξj

. (23)

After global optimization, each keyframe k reprojects its 3D points {Xnk}N

n=1 to all connected keyframes Nk and computes the mean reprojection error

k

1 |Nk| j∈N

∥unj − πK(TjkXnk)∥2. (24)

en =

k

A confidence score is then assigned as

 

1, en ≤ τproj,

(25)

cn =

1 en − τproj + 1



, en > τproj,

and the weighted pairs {(Xnk,cn)} are sent to the mapping thread. This ensures that subsequent Gaussian-primitive updates favor high-confidence geometric regions while down-weighting uncer-

tain areas.

- A.6 MORE EXPERIMENTS

Below, We provide per-scene quantitative results and qualitative comparisons to further support the main paper. As shown in Tab. 4- 29, ARTDECO consistently achieves the highest PSNR and SSIM and the lowest LPIPS across all datasets, validating its strong generalization from vatious type of indoor scenes (TUM (Sturm et al., 2012), ScanNet (Dai et al., 2017), ScanNet++ (Yeshwanth et al., 2023), VR-NeRF (Xu et al., 2023)) to large-scale outdoor captures (Waymo (Sun et al., 2020), FastLIVO2 (Zheng et al., 2024), KITTI (Geiger et al., 2013), MatrixCity (Li et al., 2023)).

We also provide qualitative results related to reconstruction and tracking trajectories, as shown in Fig. 7 6 8.

Table 4: PSNR on the Fast-LIVO2 Dataset

Method CBD Building 01 HKU Campus Red Sculpture Retail Street SYSU MonoGS 19.86 21.70 15.50 18.05 19.23

SEGS-SLAM 26.26 29.55 - 18.49 24.01 S3PO-GS 17.47 25.47 18.89 24.30 21.42

- OnTheFly-NVS 17.79 21.46 17.50 17.67 19.39 LongSplat 29.25 29.45 24.97 23.10 25.07

Ours 31.11 30.89 26.29 29.20 30.22

Table 5: SSIM on the Fast-LIVO2 Dataset

Method CBD Building 01 HKU Campus Red Sculpture Retail Street SYSU MonoGS 0.698 0.608 0.518 0.554 0.610

SEGS-SLAM 0.880 0.847 - 0.589 0.777

S3PO-GS 0.645 0.705 0.620 0.782 0.669 OnTheFly-NVS 0.677 0.628 0.590 0.558 0.635

LongSplat 0.891 0.832 0.786 0.708 0.742 Ours 0.940 0.871 0.861 0.901 0.899

Table 6: LPIPS on the Fast-LIVO2 Dataset

Method CBD Building 01 HKU Campus Red Sculpture Retail Street SYSU MonoGS 0.623 0.687 0.778 0.750 0.658

SEGS-SLAM 0.213 0.205 - 0.516 0.292

S3PO-GS 0.592 0.448 0.505 0.232 0.451 OnTheFly-NVS 0.490 0.470 0.500 0.528 0.498

LongSplat 0.179 0.258 0.304 0.315 0.322 Ours 0.108 0.199 0.205 0.127 0.151

Table 7: PSNR on the TUM Dataset

Method f1 360 f1 desk f1 desk2 f1 floor f1 plant f1 room f1 rpy f1 teddy f1 xyz f2 xyz f3 office MonoGS 16.17 14.86 14.96 20.71 17.46 15.38 16.28 16.50 21.89 21.23 20.10

SEGS-SLAM 19.43 19.81 18.44 21.75 17.33 - 18.44 15.37 20.68 19.54 26.14 S3PO-GS 16.70 20.09 18.52 22.69 18.41 17.14 16.67 19.02 22.78 23.06 20.74

- OnTheFly-NVS 18.44 18.91 18.38 25.43 15.84 17.26 20.86 16.27 25.69 20.04 19.80 LongSplat 27.07 25.35 25.48 28.14 21.27 23.52 22.81 22.36 26.75 27.46 25.80

Ours 26.19 26.04 25.54 29.43 24.06 25.23 24.92 23.30 26.50 29.91 26.92

Table 8: SSIM on the TUM Dataset

Method f1 360 f1 desk f1 desk2 f1 floor f1 plant f1 room f1 rpy f1 teddy f1 xyz f2 xyz f3 office MonoGS 0.583 0.529 0.552 0.586 0.581 0.542 0.575 0.539 0.738 0.698 0.698

SEGS-SLAM 0.751 0.775 0.720 0.752 0.641 - 0.718 0.622 0.821 0.769 0.861 S3PO-GS 0.602 0.680 0.650 0.625 0.616 0.597 0.596 0.614 0.762 0.752 0.723

OnTheFly-NVS 0.725 0.712 0.713 0.783 0.600 0.662 0.755 0.594 0.870 0.730 0.760 LongSplat 0.826 0.833 0.842 0.767 0.695 0.790 0.783 0.736 0.888 0.879 0.799

Ours 0.850 0.861 0.859 0.838 0.797 0.838 0.847 0.768 0.883 0.922 0.882

Table 9: LPIPS on the TUM Dataset

Method f1 360 f1 desk f1 desk2 f1 floor f1 plant f1 room f1 rpy f1 teddy f1 xyz f2 xyz f3 office MonoGS 0.642 0.664 0.661 0.736 0.572 0.679 0.509 0.654 0.326 0.355 0.511

SEGS-SLAM 0.361 0.244 0.358 0.277 0.399 - 0.336 0.415 0.205 0.270 0.200 S3PO-GS 0.551 0.433 0.506 0.634 0.461 0.571 0.507 0.482 0.296 0.270 0.420

OnTheFly-NVS 0.445 0.396 0.404 0.305 0.501 0.440 0.353 0.505 0.199 0.326 0.307 LongSplat 0.324 0.266 0.270 0.305 0.406 0.255 0.267 0.300 0.127 0.152 0.325

###### Ours 0.279 0.220 0.238 0.233 0.263 0.251 0.235 0.298 0.174 0.080 0.191

Table 10: PSNR on the ScanNet Dataset

Method scene0000 00 scene0059 00 scene0106 00 scene0169 00 scene0181 00 scene0207 00 MonoGS - 18.09 18.03 19.71 19.37 19.16

SEGS-SLAM - - 17.92 20.89 21.40 18.69

S3PO-GS 17.98 19.54 20.27 21.24 21.20 20.62 OnTheFly-NVS 14.88 16.33 14.66 16.37 15.47 14.47

LongSplat - 19.52 19.01 19.71 18.94 19.17 Ours 23.28 24.74 26.34 23.07 21.80 25.37

Table 11: SSIM on the ScanNet Dataset

Method scene0000 00 scene0059 00 scene0106 00 scene0169 00 scene0181 00 scene0207 00 MonoGS - 0.733 0.774 0.792 0.823 0.776

SEGS-SLAM - - 0.804 0.850 0.902 0.799

S3PO-GS 0.738 0.769 0.816 0.815 0.845 0.797 OnTheFly-NVS 0.742 0.747 0.667 0.721 0.667 0.705

LongSplat - 0.752 0.764 0.750 0.782 0.721 Ours 0.824 0.863 0.905 0.859 0.889 0.848

Table 12: LPIPS on the ScanNet Dataset

Method scene0000 00 scene0059 00 scene0106 00 scene0169 00 scene0181 00 scene0207 00 MonoGS - 0.713 0.597 0.612 0.578 0.644

SEGS-SLAM - - 0.405 0.352 0.269 0.432

S3PO-GS 0.700 0.518 0.484 0.550 0.513 0.583 OnTheFly-NVS 0.464 0.477 0.554 0.504 0.491 0.474

LongSplat - 0.402 0.384 0.384 0.423 0.427 Ours 0.254 0.279 0.237 0.278 0.288 0.290

Table 13: PSNR on the Waymo Dataset

Method 100613 106762 132384 13476 152706 153495 158686 163453 405841 MonoGS 20.05 20.91 22.71 19.51 21.23 14.15 20.29 19.01 16.19

SEGS-SLAM 20.14 23.60 22.52 - 24.11 21.16 21.22 - 19.01 S3PO-GS 25.36 28.23 27.01 24.85 28.53 26.54 26.03 23.65 27.28

OnTheFly-NVS 26.95 27.21 25.31 24.34 25.45 26.41 26.30 23.97 23.79 LongSplat 24.25 23.70 24.11 24.42 25.66 23.74 24.84 22.69 25.61

##### Ours 27.98 30.84 30.32 28.03 29.80 27.60 26.83 26.91 30.48

Table 14: SSIM on the Waymo Dataset

Method 100613 106762 132384 13476 152706 153495 158686 163453 405841 MonoGS 0.758 0.816 0.855 0.713 0.792 0.672 0.723 0.745 0.695

SEGS-SLAM 0.730 0.805 0.824 - 0.784 0.734 0.677 - 0.698 S3PO-GS 0.828 0.878 0.883 0.778 0.856 0.846 0.819 0.797 0.865

OnTheFly-NVS 0.850 0.854 0.864 0.764 0.795 0.847 0.837 0.785 0.788 LongSplat 0.773 0.766 0.822 0.716 0.778 0.755 0.777 0.732 0.795

##### Ours 0.865 0.906 0.919 0.856 0.882 0.871 0.847 0.866 0.907

Table 15: LPIPS on the Waymo Dataset

Method 100613 106762 132384 13476 152706 153495 158686 163453 405841 MonoGS 0.610 0.534 0.451 0.745 0.666 0.710 0.627 0.664 0.633

SEGS-SLAM 0.484 0.399 0.384 - 0.450 0.472 0.495 - 0.502 S3PO-GS 0.329 0.276 0.275 0.471 0.427 0.379 0.373 0.411 0.352

OnTheFly-NVS 0.328 0.337 0.361 0.378 0.401 0.348 0.307 0.376 0.400 LongSplat 0.356 0.328 0.355 0.354 0.397 0.430 0.321 0.371 0.326

##### Ours 0.308 0.237 0.265 0.267 0.304 0.313 0.283 0.289 0.216

Table 16: PSNR on the VR-NeRF Dataset

Method appartment262 kitchen261 kitchen262 kitchen263 table61 workspace61 workspace62 workspace64 MonoGS 18.43 16.91 11.66 14.47 15.50 14.80 15.12 14.80

###### SEGS-SLAM 26.14 31.81 - 32.65 36.55 30.95 - -

S3PO-GS 28.45 27.98 25.16 18.56 19.36 22.16 23.49 22.17 LongSplat 31.22 27.10 27.76 24.04 24.90 - 22.32 22.84

OnTheFly-NVS 30.52 30.24 27.51 25.16 27.77 22.08 26.04 29.05 Ours 32.98 30.90 30.23 29.04 29.05 24.68 24.63 27.13

Table 17: SSIM on the VR-NeRF Dataset

Method appartment262 kitchen261 kitchen262 kitchen263 table61 workspace61 workspace62 workspace64 MonoGS 0.646 0.627 0.506 0.599 0.595 0.526 0.581 0.522

SEGS-SLAM 0.831 0.910 - 0.883 0.949 0.905 - -

S3PO-GS 0.875 0.880 0.856 0.719 0.737 0.762 0.791 0.758 LongSplat 0.905 0.861 0.888 0.803 0.823 - 0.757 0.787

OnTheFly-NVS 0.912 0.903 0.900 0.847 0.882 0.775 0.855 0.898 Ours 0.937 0.913 0.939 0.913 0.900 0.842 0.833 0.883

Table 18: LPIPS on the VR-NeRF Dataset

Method appartment262 kitchen261 kitchen262 kitchen263 table61 workspace61 workspace62 workspace64 MonoGS 0.631 0.688 0.676 0.662 0.582 0.689 0.721 0.685

###### SEGS-SLAM 0.375 0.218 - 0.192 0.174 0.203 - -

S3PO-GS 0.321 0.302 0.269 0.597 0.474 0.385 0.350 0.386 LongSplat 0.292 0.308 0.265 0.319 0.353 - 0.359 0.352

OnTheFly-NVS 0.302 0.277 0.311 0.311 0.322 0.385 0.307 0.261 Ours 0.201 0.224 0.185 0.198 0.256 0.286 0.314 0.274

Table 19: PSNR on the KITTI Dataset

Method 00 02 03 05 06 07 08 10

MonoGS 16.01 15.08 16.90 15.66 16.38 11.38 13.21 11.82 SEGS-SLAM 12.69 14.88 16.71 14.88 14.65 10.45 14.50 13.51

S3PO-GS 20.77 19.30 20.51 20.73 20.42 20.38 20.27 17.41

- OnTheFly-NVS 15.97 17.00 17.34 18.08 17.29 18.21 16.50 14.73 LongSplat 17.84 14.62 17.97 18.08 18.68 16.14 16.58 14.93

Ours 23.76 22.53 24.54 23.80 23.59 23.92 22.86 20.38

Table 20: SSIM on the KITTI Dataset

Method 00 02 03 05 06 07 08 10

MonoGS 0.568 0.476 0.487 0.491 0.560 0.439 0.480 0.420 SEGS-SLAM 0.454 0.458 0.448 0.470 0.510 0.405 0.493 0.462

S3PO-GS 0.732 0.587 0.581 0.659 0.652 0.715 0.684 0.545 OnTheFly-NVS 0.594 0.538 0.529 0.606 0.583 0.673 0.622 0.483 LongSplat 0.616 0.444 0.482 0.548 0.583 0.557 0.543 0.484 Ours 0.829 0.707 0.745 0.781 0.779 0.827 0.786 0.663

Table 21: LPIPS on the KITTI Dataset

Method 00 02 03 05 06 07 08 10

MonoGS 0.687 0.761 0.735 0.753 0.720 0.826 0.830 0.820 SEGS-SLAM 0.492 0.491 0.465 0.453 0.464 0.552 0.450 0.533

S3PO-GS 0.264 0.459 0.498 0.389 0.401 0.364 0.345 0.563 OnTheFly-NVS 0.461 0.501 0.501 0.455 0.500 0.388 0.422 0.540 LongSplat 0.375 0.531 0.438 0.428 0.394 0.454 0.440 0.517 Ours 0.234 0.375 0.311 0.281 0.288 0.242 0.266 0.394

Table 22: PSNR on the ScanNet++ Dataset

Method 00777c41d4 02f25e5fee 0b031f3119 126d03d821 1cbb105c6a 2284bf5c9d 2d2e873aa0 MonoGS 14.621 20.915 9.488 21.353 23.716 13.049 16.861

SEGS-SLAM - 28.177 - - - - S3PO-GS 21.316 25.158 21.019 23.758 26.614 23.446 23.554

- OnTheFly-NVS 16.270 19.134 21.701 17.572 17.478 18.200 14.173 LongSplat 18.465 26.747 22.484 27.251 28.621 23.348 -

###### Ours 26.543 30.671 27.722 32.796 32.406 30.277 28.137

Method 303745abc7 41eb967018 46001f434d 4808c4a397 546292a9db 712dc47104 7543973e1a MonoGS 14.899 18.011 18.515 21.570 9.464 8.970 22.547

SEGS-SLAM - - - - - - 27.169

S3PO-GS 25.840 21.074 20.288 24.263 20.612 17.843 26.367 OnTheFly-NVS 15.884 16.247 16.426 18.722 24.791 14.056 21.540

LongSplat - - 24.437 - 24.587 - 28.539 Ours 33.056 28.608 21.716 30.101 25.868 27.506 32.308

Table 23: SSIM on the ScaNnet++ Dataset

Method 00777c41d4 02f25e5fee 0b031f3119 126d03d821 1cbb105c6a 2284bf5c9d 2d2e873aa0 MonoGS 0.559 0.784 0.485 0.796 0.833 0.622 0.641

SEGS-SLAM - 0.911 - - - - S3PO-GS 0.695 0.853 0.809 0.828 0.876 0.833 0.830

OnTheFly-NVS 0.583 0.779 0.823 0.753 0.749 0.751 0.705 LongSplat 0.604 0.868 0.816 0.854 0.897 0.819 -

###### Ours 0.855 0.941 0.903 0.942 0.956 0.937 0.918

Method 303745abc7 41eb967018 46001f434d 4808c4a397 546292a9db 712dc47104 7543973e1a MonoGS 0.723 0.737 0.844 0.848 0.244 0.590 0.836

SEGS-SLAM - - - - - - 0.891

S3PO-GS 0.878 0.788 0.861 0.873 0.688 0.784 0.884 OnTheFly-NVS 0.780 0.705 0.817 0.826 0.825 0.732 0.831

LongSplat - - 0.900 - 0.793 - 0.894 Ours 0.959 0.905 0.891 0.938 0.850 0.915 0.950

Table 24: LPIPS on the ScanNet++ Dataset

Method 00777c41d4 02f25e5fee 0b031f3119 126d03d821 1cbb105c6a 2284bf5c9d 2d2e873aa0 MonoGS 0.822 0.459 0.724 0.450 0.370 0.710 0.769

SEGS-SLAM - 0.148 - - - - -

S3PO-GS 0.451 0.243 0.350 0.346 0.264 0.289 0.316 OnTheFly-NVS 0.524 0.352 0.310 0.404 0.409 0.386 0.499 LongSplat 0.436 0.179 0.307 0.242 0.179 0.245 -

###### Ours 0.209 0.122 0.196 0.143 0.123 0.134 0.179

Method 303745abc7 41eb967018 46001f434d 4808c4a397 546292a9db 712dc47104 7543973e1a MonoGS 0.707 0.619 0.485 0.408 0.734 0.707 0.441

SEGS-SLAM - - - - - - 0.216

S3PO-GS 0.259 0.454 0.419 0.300 0.520 0.449 0.303 OnTheFly-NVS 0.382 0.434 0.362 0.331 0.256 0.450 0.305

LongSplat - - 0.265 - 0.286 - 0.200 Ours 0.115 0.165 0.279 0.156 0.205 0.176 0.140

Table 25: Tracking Results on the ScanNet++ Dataset

Method 0077 02f2 0b03 126d 1cbb 2284 2d2e 3037 41eb 4600 4808 5462 712d 7543

OnTheFly 0.804 0.157 0.114 0.227 0.227 0.913 1.438 1.186 1.605 3.932 0.486 0.018 0.458 0.907 LongSplat 1.307 0.041 0.378 0.550 0.023 0.713 - - - 1.943 - 0.408 - 0.058

MonoGS 1.364 0.626 - - 0.294 0.939 2.019 0.922 1.760 3.540 0.432 - 0.853 0.642 S3PO-GS 0.650 0.016 0.093 0.418 0.135 0.441 0.355 0.858 1.674 1.318 0.614 0.909 1.092 0.270

SEGS-SLAM - 0.005 - - - - - - - - - - - 0.485 MASt3R-SLAM 0.121 0.017 0.059 0.122 0.021 0.027 0.835 0.176 0.020 2.379 0.065 0.042 0.021 0.025 loop with vggt 0.010 0.011 0.014 1.176 0.014 0.022 0.010 0.007 0.030 0.075 0.022 0.032 0.013 0.011 Ours 0.009 0.011 0.015 0.019 0.015 0.012 0.010 0.005 0.017 0.060 0.021 0.030 0.014 0.011

Table 26: Tracking Results on the TUM Dataset

Method f1 360 f1 desk f1 desk2 f1 floor f1 plant f1 room f1 rpy f1 teddy f1 xyz f2 xyz f3 office

OnTheFly 0.187 - - - - 0.874 4.136 - 0.278 0.073 1.489 LongSplat 0.133 - - - - 0.712 - - - 0.103 -

MonoGS 0.160 0.034 0.596 0.531 0.077 0.649 0.032 0.512 0.017 0.047 0.033 S3PO-GS 0.093 0.041 0.155 0.228 0.040 0.552 0.053 0.051 0.010 0.016 0.045

SEGS-SLAM 0.154 0.016 0.013 - - - - 0.293 0.009 0.006 0.026 MASt3R-SLAM 0.049 0.016 0.024 0.025 0.020 0.061 0.027 0.041 0.009 0.005 0.031

loop with vggt 0.040 0.016 0.026 0.026 0.017 0.056 0.023 0.047 0.007 0.005 0.024 Ours 0.040 0.016 0.025 0.025 0.016 0.060 0.022 0.045 0.007 0.005 0.019

Table 27: Tracking Results on the KITTI Dataset

Method 00 02 03 05 06 07 08 10 Avg.

OnTheFly 18.297 28.230 14.356 5.027 9.608 1.550 9.836 9.502 12.051 LongSplat 3.047 10.796 9.344 9.482 6.864 2.866 4.480 6.576 6.682

MonoGS 11.868 11.817 11.195 4.623 7.638 4.128 5.864 5.109 7.780 S3PO-GS 1.196 2.961 5.522 1.459 0.721 1.009 2.655 1.927 2.181

SEGS-SLAM 0.561 0.897 0.189 0.967 4.122 0.851 0.477 0.353 1.052 MASt3R-SLAM - 1.756 0.397 0.761 2.279 1.361 - - –

loop with vggt 1.304 2.691 0.442 1.103 0.958 1.257 1.160 1.893 1.351 Ours 1.304 2.691 0.442 1.103 0.958 1.321 1.167 1.893 1.360

Table 28: Tracking Results on the Waymo Dataset

Method 13476 100613 106762 132384 152706 153495 158686 163453 405841 Avg.

OnTheFly 1.687 1.051 2.022 14.395 1.193 2.110 2.206 1.872 1.523 3.118 LongSplat 7.024 10.242 7.506 2.616 4.900 2.625 3.921 3.939 2.729 4.956

MonoGS 3.262 6.706 15.638 11.093 8.286 0.585 6.881 10.669 3.218 7.370 S3PO-GS 1.165 2.471 0.214 1.551 1.144 1.972 0.963 1.085 0.560 1.236

SEGS-SLAM - 0.860 0.696 2.755 1.867 - - - - MASt3R-SLAM - - 2.958 - - 1.569 1.098 2.334 0.873 -

loop with vggt 1.230 0.315 2.762 1.753 0.931 1.578 0.460 1.071 0.816 1.213 Ours 1.229 0.315 2.762 1.753 0.931 1.578 0.460 1.071 0.816 1.213

Table 29: Tracking Results on the TUM Dataset Compared with Non 3DGS SLAM Systems

Method 360 desk desk2 floor plant room rpy teddy xyz Avg. ORB-SLAM3 - 0.017 0.210 - 0.034 - - - 0.009 -

DPV-SLAM++ 0.132 0.018 0.029 0.050 0.022 0.096 0.032 0.098 0.010 0.054 DROID-SLAM 0.111 0.018 0.042 0.021 0.016 0.049 0.026 0.048 0.012 0.038

Go-SLAM 0.089 0.016 0.028 0.025 0.026 0.052 0.019 0.048 0.010 0.035 MASt3R-SLAM 0.049 0.016 0.024 0.025 0.020 0.061 0.027 0.041 0.009 0.030 Ours 0.040 0.016 0.025 0.025 0.016 0.060 0.022 0.045 0.007 0.028

[Figure 164]

[Figure 165]

###### SEGS-SLAM

S3PO-GS

[Figure 166]

[Figure 167]

LongSplat

OnTheFly-NVS

GT

[Figure 168]

[Figure 169]

GT

Ours

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

S3PO-GS SEGSSLAM

OnTheFly -NVS

Ours GT

Figure 5: More Qualitative Reconstruction Results.

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

KITTI-03KITTI-07

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

MonoGS OnTheFly-NVS S3PO-GS SEGS-SLAM Ours

MASt3R-SLAM

- Figure 6: Qualitative Comparison of Trajectories across Different Methods on the KITTI Dataset.

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

###### 02f25e5fee96b7c379eb6183f0657d

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

Loop failed Loop failed

MonoGS OnTheFly-NVS S3PO-GS SEGS-SLAM MASt3R-SLAM Ours

#### Figure 7: Qualitative Comparison of Trajectories across Different Methods on the ScanNet++ Dataset.

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

MonoGS S3PO-GS SEGS-SLAM MASt3R-SLAM Ours Fr1/PlantFr3/OfficeFr1/Desk

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

#### Figure 8: Qualitative Comparison of Trajectories across Different Methods on the TUM Dataset.

