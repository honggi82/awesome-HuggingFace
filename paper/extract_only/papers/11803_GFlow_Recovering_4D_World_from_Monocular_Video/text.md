## GFlow: Recovering 4D World from Monocular Video

### Shizun Wang, Xingyi Yang, Qiuhong Shen, Zhenxiang Jiang, Xinchao Wang*

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Frame 35

National University of Singapore shizun.wang@u.nus.edu, xinchao@nus.edu.sg

teaser

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

# arXiv:2405.18426v2[cs.CV]31Dec2024

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

2D Tracking 3D Tracking

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Still Cluster

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Moving Cluster

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Zero-shot Segmentation

Consistent Depth

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

Novel View Synthesis Scene Editing

A) Monocular Video Input w/o Camera Parameters B) Center points of Reconstructed Gaussians in GFlow

C) Downstream Video Applications

Figure 1: A) Given a monocular video in the wild, B) our proposed GFlow can reconstruct the underlying 4D world, i.e. the dynamic scene represented by 3D Gaussian splatting (Kerbl et al. 2023) and associated camera poses. Within GFlow, the Gaussians are split into still and moving clusters and and are further densified. C) GFlow facilitates a range of applications, including tracking objects in 2D and 3D, segmenting video objects, synthesizing new views, estimating consistent depth and video editing. We encourage readers to visit the anonymous website for more video illustrations.

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

GFlow

[Figure 63]

[Figure 64]

[Figure 65]

Abstract

from causal videos; it naturally enables tracking of points and segmentation of moving objects across frames. Additionally, GFlow estimates the camera poses for each frame, enabling novel view synthesis by changing camera pose. This capability facilitates extensive scene-level or object-level editing, highlighting GFlow’s versatility and effectiveness.

[Figure 66]

Recovering 4D world from monocular video is a crucial yet challenging task. Conventional methods usually rely on the assumptions of multi-view videos, known camera parameters, or static scenes. In this paper, we relax all these constraints and tackle a highly ambitious but practical task: With only one monocular video without camera parameters, we aim to recover the dynamic 3D world alongside the camera poses. To solve this, we introduce GFlow, a new framework that utilizes only 2D priors (depth and optical flow) to lift a video to a 4D scene, as a flow of 3D Gaussians through space and time. GFlow starts by segmenting the video into still and moving parts, then alternates between optimizing camera poses and the dynamics of the 3D Gaussian points. This method ensures consistency among adjacent points and smooth transitions between frames. Since dynamic scenes always continually introduce new visual content, we present prior-driven initialization and pixel-wise densification strategy for Gaussian points to integrate new content. By combining all those techniques, GFlow transcends the boundaries of 4D recovery

Website — https://littlepure2333.github.io/GFlow

### 1 Introduction

The quest for accurate reconstruction of 4D scene from video inputs stands at the forefront of contemporary research in computer vision and graphics. This endeavor is crucial for the advancement of virtual and augmented reality, video analysis, and multimedia applications. The main challenge lies in capturing the transient essence of dynamic scenes and the often absent camera pose information.

Traditional approaches are typically split between two types: the one relies on pre-calibrated camera parameters or multi-view video inputs to reconstruct dynamic scenes (Wu et al. 2023; Luiten et al. 2023; Sun et al. 2024; Bansal et al. 2020; Cao and Johnson 2023; Fridovich-Keil et al. 2023; Li et al. 2022; Lin et al. 2023, 2021b; Pumarola et al. 2021),

*Corresponding Author Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

while the other estimates camera poses from static scenes using multi-view stereo techniques (Bian et al. 2023; Fu et al. 2023; Wang et al. 2023; Lin et al. 2021a; Wang et al.

- 2021; Xia et al. 2022; Sch¨onberger et al. 2016; Schonberger and Frahm 2016; Tian, Du, and Duan 2023; Charatan et al. 2023). This division highlights a missing piece in this field: the challenge of reconstructing dynamic scenes using only a single monocular video without any camera parameters.

Addressing this challenge is particularly difficult due to the inherently ill-posed nature. From a single monocular video, multiple reconstructions might visually appear correct when projected onto the camera view. However, many of these reconstructions fail to conform to the physical laws of the real world. Although NeRF-based (Cao and Johnson 2023; Fridovich-Keil et al. 2023; Shao et al. 2023; Liu et al. 2023) methods attempt to solve this problem, they often yield poor results. This failure is primarily due to their implicit representation, which makes it challenging to accurately enforce physical constraints in the reconstruction.

Recent developments in 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) and its extensions (Wu et al. 2023; Luiten et al. 2023; Yang et al. 2023a,b) into dynamic scenes have emerged as promising alternatives. These techniques have shown promise in handling the complexities associated with the dynamic nature of real-world scenes, as well as the intricacies of camera movement and positioning.Yet, they still operate under the assumption of known camera poses (Sch¨onberger et al. 2016; Schonberger and Frahm 2016).

To transcend these limitations and fully leverage the capabilities of 2D foundation models for dynamic scene reconstruction, we offer a novel insight:

Given 2D factors such as RGB, depth and optical flow from one video, we have enough clues to model the 4D (3D+t) world behind the video.

Leveraging this insight, we introduce GFlow, a novel framework that leverages 3D Gaussian Splatting (Kerbl et al. 2023) to reconstruct the video. It conceptualizes the video content as a dynamic flow of Gaussian points moving through space and time. We simultaneously optimize the flow and the camera poses together, to ensure that the projected video adheres to those 2D factors.

The key to GFlow lies in the alternating optimization of camera poses and dynamic 3D Gaussians. While directly estimating camera poses in dynamic scenes is considered highly challenging, we make it feasible by separating the scene into static and dynamic parts. For the static parts, we optimize the camera poses using reprojection error. In dynamic regions, Gaussian points are first reprojected using the optimized camera poses, then refined based on RGB, depth, and flow priors. This dual optimization ensures that each video frame is rendered accurately, capturing the dynamic nature of the original scene.

Apart from the optimization strategy, we propose two methods to effectively integrate new Gaussian points into the scene and accelerate convergence. The first method, priordriven initialization, sets up initial Gaussian points in plausible 3D geometric positions, based on RGB and depth priors. The second method, pixel-wise densification, involves increasing the number of Gaussian points in regions with large

pixel errors. Together, these strategies contribute to maintaining high fidelity in cross-frame rendering, also ensuring that transitions and movements between frames are smooth.

Beyond dynamic 3D scene recovery, GFlow can also serve as a powerful tool for video processing. It can track points across frames in 3D world coordinates without prior training and segment objects by propagating a given initial mask. Since it employs explicit representation, GFlow can render captivating new views of video scenes by easily changing camera poses and editing objects or entire scenes as desired, showcasing its versatility and power.

To conclude, our contributions are: 1) A novel framework that recovers 4D scenes and associated camera poses from a monocular video. 2) An alternating optimization process that ensures high fidelity and temporally smooth dynamics in 4D scenes. 3) Two new strategies for initializing and densifying Gaussian points in dynamic scenes. 4) Enables new video processing capabilities, including tracking, segmentation, novel view rendering, and editing.

### 2 Related works

- 3D Renderable Representations Static 3D scenes can be recovered as renderable representations from posed multi-view images through differentiable rendering, enabling novel view synthesis. Such 3D renderable representations can be categorized into implicit and explicit representations. Early works in 3D scene reconstruction primarily adopted implicit neural representations (Flynn et al. 2016). The most influential of these, Neural Radiance Fields (NeRFs) (Mildenhall et al. 2021), introduced importance sampling with volumetric ray-marching but relied on a deep multi-layer perceptron, significantly hindering rendering speed. Although follow-up works (M¨uller et al. 2022; Chen et al. 2022) adopted hash grids or structured tensors with smaller MLPs to represent density and appearance, their rendering speed is still constrained by the need to query substantial samples for single ray marching.

In contrast, the explicit category is dominated by differentiable point-based rendering techniques (Yifan et al. 2019; Kerbl et al. 2023). This approach eliminates the need to query samples from deep networks, instead directly fetching attributes from points, which enables a significant speedup compared to implicit neural-based methods. Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) extends points to 3D Gaussians with opacity and spherical harmonics, and introduces tile-based rasterization to achieve realtime rendering speeds. Here, we choose 3DGS as our base representation, as its fast rendering speed and the explicit nature make the reconstructed scene flexible enough for content creation and editing.

- 4D Reconstruction 4D reconstruction from video, also known as dynamic 3D scene reconstruction. Many prior works extended NeRFs to handle dynamic scenes (Park et al. 2021a,b; Pumarola et al. 2021; Li et al. 2023), typically using grids, triplanes, voxels (Cao and Johnson 2023; Fridovich-Keil et al. 2023; Shao et al. 2023; Liu et al. 2023), or learning deformable fields to map a canonical template (Ouyang et al. 2023; Kasten et al. 2021). But the recon-

struction quality is relatively low due to its implicit essence. Recent developments in 3DGS (Kerbl et al. 2023) have set new records in reconstruction quality and rendering speed. Extensions of 3DGS (Wu et al. 2023; Luiten et al. 2023; Yang et al. 2023a,b) have begun exploring dynamic scene reconstruction. However, they still operate under the assumption of a known camera sequence (Sch¨onberger et al. 2016; Schonberger and Frahm 2016).

While almost all previous methods either rely on known camera parameters or multi-view video inputs to reconstruct dynamic scenes (Sun et al. 2024; Bansal et al. 2020; Li et al.

- 2022; Lin et al. 2023, 2021b), or estimate camera poses from static scenes using multi-view stereo techniques (Bian et al.
- 2023; Fu et al. 2023; Wang et al. 2023; Lin et al. 2021a; Wang et al. 2021; Xia et al. 2022; Tian, Du, and Duan

- 2023; Charatan et al. 2023). The key difference between our GFlow and these approaches lies in our ability to recover dynamic scenes from a unposed monocular video. Additionally, some concurrent works (Wang et al. 2024; Stearns et al.
- 2024; Lei et al. 2024; Liu et al. 2024; Kong, Yang, and Wang

2025) also attempt to solve this problem.

### 3 Preliminaries

#### 3.1 3D gaussian splatting

Recently, 3D Gaussian Splatting (3DGS) (Kerbl et al. 2023) exhibits strong performance and efficiency in 3D representation. 3DGS fits a scene as a set of Gaussians {Gi} from multi-view images {Vk} and paired camera poses {Pk} in an optimization pipeline. Adaptive densification and pruning of Gaussians are applied in this iterative optimization to control the total number of Gaussians. Generally, each Gaussian is composed of its center coordinate µ ∈ R3, 3D scale s ∈ R3, opacity α ∈ R, rotation quaternion q ∈ R4, and associated view-dependent color represented as spherical harmonics c ∈ R3(d+1)

2

, where d is the degree of spherical harmonics.

These parameters can be collectively denoted by G, with Gi = {µi,si,αi,qi,ci} denoting the parameters of the i-th Gaussian. The core of 3DGS is its tile-based differentiable rasterization pipeline to achieve real-time optimization and rendering. To render {Gi} into a 2D image, each Gaussian is first projected into the camera coordinate frame given the camera pose Pi to determine the depth of each Gaussian. Then colors, depth, or other attributes in pixel space are rendered in parallel by alpha composition with the depth order of adjacent 3D Gaussians. Specifically, in our formulation, we do not consider view-dependent color variations for simplicity, thus the degree of spherical harmonics is set as d = 0, i.e., only the RGB color c ∈ R3.

#### 3.2 Camera model

To project the 3D point coordinates µ ∈ R3 into the camera view, we use the pinhole camera model. The camera intrinsics is K ∈ R3×3 and the camera extrinsics which define the world-to-camera transformation is E = [R|t] ∈ R3×4. The camera-view 2D coordinates x ∈ R2 are calculated as dh(x) = KEh(µ), where d ∈ R is the depth, and h(·) represents the homogeneous coordinate mapping.

### 4 Methodology

Problem Definition We aim to address a highly challenging and ill-posed problem, which is commonly encountered in real-world scenarios though: Given a sequence of monocular video frames without known camera parameters, the objective is to model the dynamic 3D world and the associated camera poses to represent the video.

Overview To address this problem, we propose GFlow, a framework that represents videos through a flow of 3D Gaussians, as shown in Figure 2. We first preprocess the videos to derive several priors using advanced foundation models. The priors include depth (Leroy, Cabon, and Revaud 2024), optical flow (Xu et al. 2023), and camera intrinsics (Leroy, Cabon, and Revaud 2024), which we believe are the minimum necessary. These priors contribute to good initialization and regularization in the GFlow optimization process. Two novel strategies are devised to effectively deal with the Gaussian points initialization and densification in the dynamic scenes (Sec. 4.1). At the essence of proposed method, GFlow alternately optimizes the camera pose and Gaussian points for each frame in sequential order to reconstruct the 4D world, assisted by movement clustering of Gaussian points (Sec. 4.2).

#### 4.1 Gaussian Points Allocation

This section introduces new strategies for initializing and densifying Gaussian points according to the video content.

Prior-driven Initialization of Gaussians The original 3D Gaussian Splatting (Kerbl et al. 2023) initializes Gaussian points using point clouds derived from Structure-fromMotion (SfM) (Schonberger and Frahm 2016; Sch¨onberger et al. 2016), which are only viable for static scenes with dense views. However, our task involves dynamic scenes that change both spatially and temporally, making SfM infeasible.

To address this, we developed a new method called priordriven initialization for single frames. This method fully utilizes the texture information and depth estimation obtained from the image to initialize the Gaussian points.

Intuitively, image areas with more edges usually indicate more complex textures, so more Gaussian points should be initialized in these areas. Given an image I ∈ RH×W, we extract its texture map T ∈ RH×W the Sobel operator (Kanopoulos, Vasanthavada, and Baker 1988), an edge detection operator. We then normalize this texture map to create a probability map P ∈ RH×W, from which we sample N points to obtain their 2D coordinates {xi}Ni=1.

To obtain their position in the 3D space, we use depth D estimated from off-the-shelf model (Leroy, Cabon, and Revaud 2024), as it can offer strong geometric information. The depth {di}Ni=1 of sampled points can be retrieved from depth map D using 2D coordinates. The 3D center coordinate {µi}Ni=1 of Gaussian points is initialized by unprojecting depth {di}Ni=1 and camera-view 2D coordinates {xi}Ni=1, according to the pinhole model. The scale {si}Ni=1 and color {ci}Ni=1 of the Gaussian points are initialized based on the

[Figure 67]

[Figure 68]

[Figure 69]

{𝐼 }

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

𝐺 𝐺

S M

𝐹

Off-the-shelf Depth,

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

Optical Flow and Camera Intrinsic

𝑃 𝑃

𝐺 𝐺

{𝐷 } {𝐹 }

𝐾

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

B) Movement Clustering A) Prior Association

C1) Camera Optimization

C2) Gaussians Optimization

D) Next frame

- Figure 2: Overview of GFlow. A) Given a monocular video input consisting of image sequence {It}, the associated depth {Dt}, optical flow {Ft} and camera intrinsic K are obtained using off-the-shelf prior. B) For each frame , GFLow first clustering the scene into still part {Gst} and moving part {Gmt }. Then optimization process in GFlow consists of two steps: C1) Only the camera pose Pt is optimized by aligning the appearance, depth and optical flow within the still cluster. C2) Under the optimized

camera pose Pt∗, the Gaussian points {Gt} are optimized and densified based on appearance, depth, optical flow and the two scene clusters. D) The same procedure of steps B, C1, and C2 loops for the next frame. The colorful marks under the dashed line represent the variables involved in the optimization.

#### 4.2 Alternating Gaussian-Camera Optimization

probability values and pixel colors retrieved from the image, respectively.

Once the first frame has been initialized and optimized, for subsequent frames, we adopt a alternating optimization strategy for the camera poses {Pi} and the Gaussians {Gi}.

Pixel-wise Densification of Gaussians 3D Gaussian Splatting (Kerbl et al. 2023), designed for static scenes, uses gradient thresholding to densify Gaussian points; points exceeding a gradient threshold are cloned or split based on their scales. However, this method struggles in dynamic scenes, particularly when camera movements reveal new scene areas where no Gaussian points exist.

Movement Clustering of Gaussian Points In constructing dynamic scenes that include both camera and object movements, treating these scenes as static can lead to inaccuracies and loss of crucial temporal information. To better manage this, we propose a method to cluster the scene into still and moving parts, which will be incorporated in the optimization process.

To address this, we introduce a new pixel-wise densification strategy that leverages image content, specifically targeting areas yet to be fully reconstructed. Our approach utilizes a pixel-wise photometric error map Epho ∈ RH×W and a mask Me ∈ RH×W as the basis for densification. This masked error map is then normalized into a probability map Pe ∈ RH×W. To densify new Gaussian points, the same initialization method described in prior-driven initialization is adopted, with the exception of sampling from Pe ⊙Me. The number of new Gaussian points introduced is proportionate to the mask ratio, ensuring controlled expansion of the point set.

We calculate the epipolar error map based on the optical flow estimated by UniMatch (Xu et al. 2023, 2022). The moving mask Mt at time t is identified by thresholding the epipolar error map. When Gaussian points are initialized or densified, those within Mt are considered as moving points {Gmi }t ⊆ {Gi}t, while others are considered as still points {Gsi}t ⊆ {Gi}t, whose center coordinate {µsi} will stop updating. This simple yet effective movement clustering method enables GFlow to model and track both rigid and non-rigid movement, whether it occurs on objects or other elements like water.

Camera Optimization The camera intrinsic K is estimated by MASt3R (Leroy, Cabon, and Revaud 2024). Between two consecutive frames, only camera extrinsic E is optimizable, all Gaussian points are frozen. The camera extrinsic E = [R|t] consists of a rotation R ∈ SO(3) and a translation t ∈ R3.

There are two scenarios for densification: 1) Before Gaussian optimization, the mask Me only marks new content, which is detected via a forward-backward consistency check using bidirectional flow from advanced optical flow estimators (Xu et al. 2022, 2023). And we set the Pe as uniform probabililty map, to fill new content emerged in a new frame. 2) During Gaussian optimization, Pe is unchanged, and the mask Me is identified by thresholding the error map Epho, ensuring densification occurs at details-lacking area.

For a given frame at time t, we optimize the camera ex-

trinsic Et by minimizing the errors in its photometric appearance, depth and optical flow.

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

- Figure 3: Our GFlow can explicitly model the dynamic 3D scene in the video. Here we show some rendered examples of videos from DAVIS (Perazzi et al. 2016; Pont-Tuset et al. 2017) dataset in the 3D world space.

[Figure 93]

extract the depth from the current frame’s depth map {Dt(x(Gmi,t))}, and project these points from the camera view to world coordinates using Et∗. This step ensures that the moving Gaussian points are accurately positioned and serves as a good initilization for subsequent optimization.

Et∗ = arg min

(λpLpho + λdLdep + λfLflo), (1)

Et

where λp, λd and λf are weighting factors. During this optimization, since the moving part is not contribute to camera pose estimation, we will mask out the moving area according to current and previous moving mask Mt,Mt−1.

Then, the total optimization objectives of Gaussian points contains photometric loss, depth loss, optical flow loss, and additional isotropic loss.

Here, R(·) denotes the 3D Gaussian splatting rendering process for desired output. The photometric loss combines MSE and SSIM (Wang et al. 2004) loss between the rendered image Iˆt = R({Gi}t) and the actual frame image It.

{G∗i }t = arg min

(λpLpho + λdLdep + λfLflo + λiLiso)

{Gi}t

(5)

Where λi is a weight factor. The isotropic loss is calculated as the mean of the standard deviation of the Gaussian points’ 3D scales {si}. In this monocular setting with sparse views, the Gaussians tend to elongate along the view ray direction due to the lack of sufficient multi-view constraints. Applying isotropic loss will encourage the Gaussians to be isotropic, helping to reduce needle-like artifacts.

Lpho = Lmse(Iˆt,It) + Lssim(Iˆt,It) (2)

The depth loss is calculated using the L1 loss between the rendered depth Dˆt = R({Gi}t) and prior depth Dt. To address discrepancies in scale and shift between the rendered and prior depths, we apply a scale and shift-invariant term on the loss, where a and b are optimizable.

N

1 N

std(si) (6)

Liso =

Ldep = |(a ∗ Dˆt + b) − Dt| (3)

i=1

The photometric and isotropic loss is applied to all Gaussian points, whereas the depth and optical flow losses focus specifically on the moving cluster {Gmi }t. This approach ensures tailored optimization that balances the dynamics and stability of Gaussian points in the scene.

The optical flow loss is calculated using the mean squared error (MSE) loss between the movements of Gaussian points in camera view Fˆt and the prior optical flow Ft, to ensure the temporal smoothness of the Gaussian points’ trajectories.

Lflo = Lmse(Fˆt,Ft) (4)

Gaussians Optimization Once the optimized camera extrinsics Et∗ is obtained, we first conduct pre-optimization gaussians relocation for those moving Gaussian points {Gmi }t. Initially, we retrieve the 2D coordinates of moving Gaussian points from the previous frame {x(Gmi,t−1)}. Using these coordinates, we calculate their movement based on the previous frame’s optical flow map {Ft−1(x(Gmi,t−1))} and update their current position: x(Gmi,t) = x(Gmi,t−1) + Ft−1(x(Gmi,t−1)). With the updated coordinates, we then

#### 4.3 Overall pipeline

To conclude, the overall pipeline can be summarized as follows: Given an image sequence {It}Tt=0 of monocular video input, we first utilizes off-the-shelf algorithms (Xu et al. 2023, 2022; Leroy, Cabon, and Revaud 2024) to derive the corresponding depth {Dt}Tt=0, optical flow {Ft}Tt=0 and camera intrinsic K. The initialization of the Gaussians is performed using the prior-driven initialization. Then for each frame It at time t, GFlow first divides the Gaussian points {Gi}t into still cluster {Gsi}t and moving cluster {Gmi }t according to the optical flow. The optimization process then proceeds in two steps. In the first step, only the

Visual results

4DGS Deformable Sprites

CoDeF

RoDynRF

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Ours

Ours

Ours

Ours

- Figure 4: Visual comparison of reconstruction quality on the DAVIS (Perazzi et al. 2016; Pont-Tuset et al. 2017) dataset: CoDef (Ouyang et al. 2023), RoDynRF (Liu et al. 2023), 4DGS (Yang et al. 2024), and Deformable Sprites (Ye et al. 2022) and Ours.

camera extrinsics Et is optimized. This is achieved by aligning the Gaussian points within the still part with the appear-

the 150 and 300 steps in the first frame optimization. For subsequent frames, the densification occurs at the first step with a new content mask applied, and also occurs at the 100 and 200 steps with error-thresholding mask applied. The error threshold in densification is set to 0.01. All experiments are conducted on a single NVIDIA RTX A5000 GPU. Note that the dynamics within each video could be distinct, so for better reconstruction, the hyperparameters could be tuned in practice.

ance It, depth Dt and optical flow Ft. Following that, under the optimized camera extrinsics Et∗, the Gaussian points Gt are further refined using constraints from the RGB It, depth Dt, optical flow Ft, and isotropic loss Liso. Additionally, the Gaussian points are densified using our proposed pixel-wise strategy to incorporate newly visible scene content. After optimizing the current frame, the procedure — movement clustering, camera optimization, and Gaussian point optimization — is repeated for subsequent frames. It is worth noting that in practice, GFlow is highly efficient, taking only 10 to 20 minutes to optimize a video — significantly faster than other methods that typically require more than an hour.

#### 5.1 Evaluation of Reconstruction Quality

Quantitative Results Reconstructing the 4D world, particularly with camera and content movement, is an extremely challenging task. We choose several methods closest to tackle this task as our baseline. Deformable Sprites (Ye et al. 2022) decomposes the video into layers of persistent motion groups, which are then composed to reconstruct the video. RoDynRF (Liu et al. 2023) uses neural voxel radiance fields to model the dynamic scene and camera poses simultaneously. CoDeF (Ouyang et al. 2023) employs implicit content deformation fields to learn a canonical template for modeling monocular video, but it lacks physical interpretability, such as the ability to estimate camera poses. 4DGS (Yang et al. 2024) models the space and time dimensions for dynamic scenes by formulating unbiased 4D Gaussian primitives, though it requires camera poses as input. We use the camera poses estimated by MASt3R as its input. As shown in Table 1, our GFlow demonstrates significant advantages in reconstruction quality. This improvement stems from its explicit representation and tailored optimization process design, which can adapt Gaussian points over time while maintaining visual content coherence.

### 5 Experiments

Dataset and Metrics We conduct experiments on a challenging video dataset, DAVIS (Perazzi et al. 2016; PontTuset et al. 2017) dataset, which contains real-world videos of about 30 ∼ 100 frames with various scenarios and motion dynamics. We report the reconstruction quality results on 30 DAVIS 2017 evaluation videos. For quantitative evaluation of reconstruction quality, we report standard PSNR, SSIM and LPIPS (Zhang et al. 2018) metrics.

Implementation details All image sequences are resized to the shortest side is 480 pixels. The initial number of Gaussian points is set to 50,000. The camera intrinsics K are estimated by MASt3R (Leroy, Cabon, and Revaud 2024). The learning rate for Gaussian optimization is set to 4e-3, and for camera optimization, it is set to 1e-3. The Adam optimizer is used with 500 iterations for Gaussian optimization in the first frame, 150 iterations for camera optimization, and 300 iterations for Gaussian optimization in subsequent frames. The gradient of color is set to zero, enforcing Gaussian points to move rather than lazily changing color. We balance the loss term by setting λp = 1, λd = 0.1, λf = 0.01, and λi = 50. Densification is conducted at

Qualitative Results The visual comparison in Figure 4 shows that CoDeF struggles to reconstruct videos with significant movement due to its reliance on representing a video as a canonical template. RoDynRF has difficulty reconstructing high-quality moving objects. Even with cam-

[Figure 102]

[Figure 103]

DAVIS PSNR↑ SSIM↑ LPIPS↓

Method

Deformable Sprites 22.83 0.6983 0.3014 RoDynRF 24.79 0.7230 0.3940 CoDeF 24.89 0.7703 0.2932 4DGS 24.60 0.7315 0.3710

[Figure 104]

[Figure 105]

GFlow (Ours) 29.74 0.9205 0.1237 GFlow* 29.21 0.9162 0.1320 w/o Ldep 29.30 0.9086 0.1444 w/o Liso 26.32 0.8664 0.2003

A) Point tracking in 2D camera-view B) Point tracking in 3D world-coordinates

Table 1: Reconstruction quality results on DAVIS(Perazzi et al. 2016; Pont-Tuset et al. 2017) dataset. Average PSNR, SSIM and LPIPS scores on all videos are reported.

Figure 5: Point tracking visualization on DAVIS dataset. A) tracking in the 2D camera-view which contains joint motion of camera and content movement. B) tracking in the 3D world-coordinates which only present content movement.

era pose inputs, 4DGS falls short in reconstructing the entire frame image. Additionally, Deformable Sprites can only reconstruct videos at a very low resolution. In contrast, our GFlow can faithfully reconstruct both static and moving content in high quality. The visual illustration in Figure 3 demonstrates the dynamic 3D scene recovered from monocular videos, showcasing the effectiveness of our approach.

motion of both the camera and the content. In contrast, the Gaussian points in GFlow reside in 3D world coordinates, representing only content movement. As a result, some 3D tracking trajectories tend to remain in their original locations, as shown in Figure 5B), because they belong to the static background. These results demonstrate that GFlow can achieve accurate tracking even on water ripples and remains reliable for fast-moving objects and scenes.

#### 5.2 Ablation study

Effect of isotropic loss Since the monocular video input only provides sparse and underconstrained views, traditional 3DGS, which relies on dense multi-view constraints, struggles to achieve good results. The sparse views will result in needle-like artifacts along the camera view ray direction. As shown in Table 1, the isotropic loss helps to overcome these drawbacks and improves the reconstruction quality.

Video Object Segmentation Since GFlow drives the Gaussian points to follow the movement of the visual content, given an initial mask, all Gaussian points within this mask can propagate to subsequent frames. This propagation forms a new mask as a concave hull (Park and Oh 2012) around these points. Notably, this capability is a by-product of GFlow, achieved without extra intended training.

Effect of depth loss Depth loss is used for ensuring a consistent and reasonable 3D geometry structure. Table 1 shows the reconstruction quality will decrease without depth loss.

Video Editing Since explicit representation can be easily edited: Camera-level manipulation: Changing the camera extrinsics can render novel views of dynamic scenes. When combined with camera intrinsics, it can create visual effects like dolly zoom. Object-level editing: With the cluster labels of moving Gaussian points, we can add, remove, resize, or stylize these points, allowing for precise object-level editing. Scene-level editing: Editing can also be applied to the entire scene, enabling the application of visual effects globally, as illustrated in Figure 1.

Effect of optimizing camera pose When preprocessing the monocular video using MASt3R (Leroy, Cabon, and Revaud 2024), it can also estimate camera poses. The results labeled as ‘GFlow*’ in Table 1 show the effect of directly using the camera poses estimated by MASt3R instead of optimizing them. A decrease in quality is observed, demonstrating the necessity of optimizing camera poses. Additionally, GFlow is capable of optimizing camera poses even when most areas in the video are in motion, where MASt3R fails.

#### 5.3 Downstream video applications

Various downstream applications can be extended from our GFlow framework, as it is an explicit representation. We encourage readers to view the videos in the website for more intuitionistic illustration.

Point tracking Due to the nature of GFlow, all Gaussian points can serve as query tracking points, enabling tracking in both 2D and 3D space. The tracking trajectories are illustrated in Figure 5. In conventional 2D tracking, tracking occurs in the camera view, which includes the combined

### 6 Conclusion

We have presented “GFlow”, a novel framework designed to address the challenging task of reconstructing the 4D world from casual monocular videos. Through Gaussian points allocation and alternating camera-Gaussian optimization, GFlow enables the recovery of dynamic scenes alongside camera poses across frames. Further capabilities such as tracking, segmentation, editing, and novel view rendering, highlighting GFlow’s potential to revolutionize video understanding and manipulation.

### Acknowledgments

This project is supported by the National Research Foundation, Singapore, under its Medium Sized Center for Advanced Robotics Technology Innovation, and the Singapore Ministry of Education Academic Research Fund Tier 1 (WBS: A-0009440-01-00).

### References

Bansal, A.; Vo, M.; Sheikh, Y.; Ramanan, D.; and Narasimhan, S. 2020. 4d visualization of dynamic events from unconstrained multi-view videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5366–5375.

Bian, W.; Wang, Z.; Li, K.; Bian, J.-W.; and Prisacariu, V. A. 2023. Nope-nerf: Optimising neural radiance field with no pose prior. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4160–4169.

Butler, D. J.; Wulff, J.; Stanley, G. B.; and Black, M. J. 2012. A naturalistic open source movie for optical flow evaluation. In A. Fitzgibbon et al. (Eds.), ed., European Conf. on Computer Vision (ECCV), Part IV, LNCS 7577, 611–625. Springer-Verlag.

Cao, A.; and Johnson, J. 2023. Hexplane: A fast representation for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 130–141.

Charatan, D.; Li, S.; Tagliasacchi, A.; and Sitzmann, V. 2023. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. arXiv preprint arXiv:2312.12337.

Chen, A.; Xu, Z.; Geiger, A.; Yu, J.; and Su, H. 2022. Tensorf: Tensorial radiance fields. In European conference on computer vision, 333–350. Springer.

Flynn, J.; Neulander, I.; Philbin, J.; and Snavely, N. 2016. Deepstereo: Learning to predict new views from the world’s imagery. In Proceedings of the IEEE conference on computer vision and pattern recognition, 5515–5524.

Fridovich-Keil, S.; Meanti, G.; Warburg, F. R.; Recht, B.; and Kanazawa, A. 2023. K-planes: Explicit radiance fields in space, time, and appearance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12479–12488.

Fu, Y.; Liu, S.; Kulkarni, A.; Kautz, J.; Efros, A. A.; and Wang, X. 2023. Colmap-free 3d gaussian splatting. arXiv preprint arXiv:2312.07504.

Kanopoulos, N.; Vasanthavada, N.; and Baker, R. L. 1988. Design of an image edge detection filter using the Sobel operator. IEEE Journal of solid-state circuits, 23(2): 358–367.

Kasten, Y.; Ofri, D.; Wang, O.; and Dekel, T. 2021. Layered neural atlases for consistent video editing. ACM Transactions on Graphics (TOG), 40(6): 1–12.

Kerbl, B.; Kopanas, G.; Leimk¨uhler, T.; and Drettakis, G. 2023. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4): 1–14.

Kong, H.; Yang, X.; and Wang, X. 2025. Efficient Gaussian Splatting for Monocular Dynamic Scene Rendering via Sparse Time-Variant Attribute Modeling. In Proceedings of the AAAI Conference on Artificial Intelligence.

Kopf, J.; Rong, X.; and Huang, J.-B. 2021. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1611–1621.

Lei, J.; Weng, Y.; Harley, A.; Guibas, L.; and Daniilidis, K. 2024. MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds. arXiv preprint

- arXiv:2405.17421. Leroy, V.; Cabon, Y.; and Revaud, J. 2024. Grounding Image Matching in 3D with MASt3R. arXiv preprint
- arXiv:2406.09756. Li, T.; Slavcheva, M.; Zollhoefer, M.; Green, S.; Lassner, C.; Kim, C.; Schmidt, T.; Lovegrove, S.; Goesele, M.; Newcombe, R.; et al. 2022. Neural 3d video synthesis from multi-view video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5521– 5531. Li, Z.; Wang, Q.; Cole, F.; Tucker, R.; and Snavely, N. 2023. Dynibar: Neural dynamic image-based rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 4273–4284. Lin, C.-H.; Ma, W.-C.; Torralba, A.; and Lucey, S. 2021a. Barf: Bundle-adjusting neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5741–5751. Lin, H.; Peng, S.; Xu, Z.; Xie, T.; He, X.; Bao, H.; and Zhou, X. 2023. High-fidelity and real-time novel view synthesis for dynamic scenes. In SIGGRAPH Asia 2023 Conference Papers, 1–9. Lin, K.-E.; Xiao, L.; Liu, F.; Yang, G.; and Ramamoorthi, R. 2021b. Deep 3d mask volume for view synthesis of dynamic scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 1749–1758. Liu, Q.; Liu, Y.; Wang, J.; Lv, X.; Wang, P.; Wang, W.; and Hou, J. 2024. MoDGS: Dynamic Gaussian Splatting from Causually-captured Monocular Videos. arXiv preprint arXiv:2406.00434. Liu, Y.-L.; Gao, C.; Meuleman, A.; Tseng, H.-Y.; Saraf, A.; Kim, C.; Chuang, Y.-Y.; Kopf, J.; and Huang, J.-B.

2023. Robust dynamic radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13–23.

Luiten, J.; Kopanas, G.; Leibe, B.; and Ramanan, D. 2023. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713.

Mildenhall, B.; Srinivasan, P. P.; Tancik, M.; Barron, J. T.; Ramamoorthi, R.; and Ng, R. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1): 99–106.

M¨uller, T.; Evans, A.; Schied, C.; and Keller, A. 2022. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4): 1– 15.

Ouyang, H.; Wang, Q.; Xiao, Y.; Bai, Q.; Zhang, J.; Zheng, K.; Zhou, X.; Chen, Q.; and Shen, Y. 2023. Codef: Content deformation fields for temporally consistent video processing. arXiv preprint arXiv:2308.07926.

- Park, J.-S.; and Oh, S.-J. 2012. A new concave hull algorithm and concaveness measure for n-dimensional datasets. Journal of Information science and engineering, 28(3): 587– 600.
- Park, K.; Sinha, U.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Seitz, S. M.; and Martin-Brualla, R. 2021a. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 5865–5874.

Park, K.; Sinha, U.; Hedman, P.; Barron, J. T.; Bouaziz, S.; Goldman, D. B.; Martin-Brualla, R.; and Seitz, S. M. 2021b. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. arXiv preprint arXiv:2106.13228.

Perazzi, F.; Pont-Tuset, J.; McWilliams, B.; Van Gool, L.; Gross, M.; and Sorkine-Hornung, A. 2016. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, 724–732.

Pont-Tuset, J.; Perazzi, F.; Caelles, S.; Arbel´aez, P.; SorkineHornung, A.; and Van Gool, L. 2017. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675.

Pumarola, A.; Corona, E.; Pons-Moll, G.; and MorenoNoguer, F. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10318–10327.

Schonberger, J. L.; and Frahm, J.-M. 2016. Structure-frommotion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, 4104–4113.

Sch¨onberger, J. L.; and Frahm, J.-M. 2016. Structure-fromMotion Revisited. In Conference on Computer Vision and Pattern Recognition (CVPR).

Sch¨onberger, J. L.; Zheng, E.; Frahm, J.-M.; and Pollefeys, M. 2016. Pixelwise view selection for unstructured multiview stereo. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, 501–518. Springer.

Shao, R.; Zheng, Z.; Tu, H.; Liu, B.; Zhang, H.; and Liu, Y. 2023. Tensor4d: Efficient neural 4d decomposition for high-fidelity dynamic reconstruction and rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 16632–16642.

Stearns, C.; Harley, A.; Uy, M.; Dubost, F.; Tombari, F.; Wetzstein, G.; and Guibas, L. 2024. Dynamic gaussian marbles for novel view synthesis of casual monocular videos. In SIGGRAPH Asia 2024 Conference Papers, 1–11.

Sturm, J.; Engelhard, N.; Endres, F.; Burgard, W.; and Cremers, D. 2012. A benchmark for the evaluation of RGB-D SLAM systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, 573–580. IEEE.

Sun, J.; Jiao, H.; Li, G.; Zhang, Z.; Zhao, L.; and Xing, W. 2024. 3dgstream: On-the-fly training of 3d gaussians for efficient streaming of photo-realistic free-viewpoint videos. arXiv preprint arXiv:2403.01444.

Teed, Z.; and Deng, J. 2021. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34: 16558–16569.

Tian, F.; Du, S.; and Duan, Y. 2023. Mononerf: Learning a generalizable dynamic radiance field from monocular videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 17903–17913.

Wang, Q.; Ye, V.; Gao, H.; Austin, J.; Li, Z.; and Kanazawa, A. 2024. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764.

Wang, S.; Leroy, V.; Cabon, Y.; Chidlovskii, B.; and Revaud, J. 2023. DUSt3R: Geometric 3D Vision Made Easy. arXiv preprint arXiv:2312.14132.

Wang, Z.; Bovik, A. C.; Sheikh, H. R.; and Simoncelli, E. P. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4): 600–612.

Wang, Z.; Wu, S.; Xie, W.; Chen, M.; and Prisacariu, V. A.

- 2021. NeRF–: Neural radiance fields without known camera parameters. arXiv preprint arXiv:2102.07064.

Wu, G.; Yi, T.; Fang, J.; Xie, L.; Zhang, X.; Wei, W.; Liu, W.; Tian, Q.; and Wang, X. 2023. 4d gaussian splatting for real-time dynamic scene rendering. arXiv preprint arXiv:2310.08528.

Xia, Y.; Tang, H.; Timofte, R.; and Van Gool, L. 2022. Sinerf: Sinusoidal neural radiance fields for joint pose estimation and scene reconstruction. arXiv preprint arXiv:2210.04553.

Xu, H.; Zhang, J.; Cai, J.; Rezatofighi, H.; and Tao, D. 2022. Gmflow: Learning optical flow via global matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 8121–8130.

Xu, H.; Zhang, J.; Cai, J.; Rezatofighi, H.; Yu, F.; Tao, D.; and Geiger, A. 2023. Unifying flow, stereo and depth estimation. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Yang, Z.; Gao, X.; Zhou, W.; Jiao, S.; Zhang, Y.; and Jin, X. 2023a. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. arXiv preprint

- arXiv:2309.13101.

Yang, Z.; Yang, H.; Pan, Z.; and Zhang, L. 2024. Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting. In International Conference on Learning Representations (ICLR).

Yang, Z.; Yang, H.; Pan, Z.; Zhu, X.; and Zhang, L. 2023b. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. arXiv preprint

- arXiv:2310.10642.

Ye, V.; Li, Z.; Tucker, R.; Kanazawa, A.; and Snavely, N.

- 2022. Deformable Sprites for Unsupervised Video Decomposition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Yifan, W.; Serena, F.; Wu, S.; Oztireli,¨ C.; and SorkineHornung, O. 2019. Differentiable surface splatting for pointbased geometry processing. ACM Transactions on Graphics (TOG), 38(6): 1–14.

Zhang, R.; Isola, P.; Efros, A. A.; Shechtman, E.; and Wang, O. 2018. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR.

Zhang, Z.; and Scaramuzza, D. 2018. A tutorial on quantitative trajectory evaluation for visual (-inertial) odometry. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 7244–7251. IEEE.

### A More Implementation Details

Preprocessing We employ UniMatch (Xu et al. 2023,

- 2022) for optical flow estimation. Specifically, we utilize the scale-2 model, which incorporates an additional 6 local regression refinement steps and is trained on a mixture of public datasets, making it well-suited for in-the-wild scenarios. For depth estimation and camera intrinsics, we adopt MASt3R (Leroy, Cabon, and Revaud 2024; Wang et al.
- 2023), performing the estimation at a subsample-2 scale with a shared intrinsic across all frames.

Initialization Once we obtain the probability map P, we normalize all non-zero values. Based on this probability map , the 3D center coordinates of Gaussian points are initialized by unprojecting the depth along camera-view 2D coordinates. The scale of Gaussian points is initialized as the odds of probability, then scaled by a factor of corresponding depth to ensure suitability for the screen size. The color of Gaussian points is initialized using the color retrieved from the image corresponding to the camera-view 2D coordinates. The opacity is initialized as 0.99, and the rotation is randomly initialized.

Once the probability map P is obtained, we normalize all non-zero values. Using this probability map, the 3D center coordinates of the Gaussian points are initialized by unprojecting the depth along the camera-view 2D coordinates. The scale of the Gaussian points is initialized as the odds of the probability and then scaled by a factor of the corresponding depth to ensure suitability for the screen size. The color of the Gaussian points is initialized using the color retrieved from the image at the corresponding camera-view 2D coordinates. The opacity is initialized as 0.99, and the rotation is randomly initialized.

Densification During densification, the number of new Gaussian points is determined by Nden = Rm × Nini, where Rm represents the mask ratio (the ratio of the masked area to the total area of the frame), and Nini denotes the initial number of Gaussian points, which is set to 50,000 in our experiments. There are two types of densification masks: 1) New content mask: Used before Gaussian optimization, this mask is detected through a forward-backward consistency check based on bidirectional optical flow (Xu et al. 2022, 2023). 2) Under-reconstructed mask: Used during Gaussian optimization, this mask is obtained by thresholding the photometric error map Epho with a threshold of 0.01.

Method ATE ↓ RPEt ↓ RPEr ↓

R-CVD 0.360 0.154 3.443 DROID-SLAM 0.175 0.084 1.912 COLMAP Fails 5 out of 14 sequences

NeRF - - 0.433 0.220 3.088 BARF 0.447 0.203 6.353 RoDynRF 0.089 0.073 1.313

##### GFlow 0.124 0.039 0.599

Table 2: Camera pose estimation results on the MPI Sintel dataset (Butler et al. 2012), reporting both Absolute Trajectory Error (ATE) and Relative Pose Error (RPE). The best results are highlighted in bold, while the second-best results are underlined. The methods in the top section can only estimate camera poses, do not reconstruct scene view images.

[Figure 106]

[Figure 107]

cave_2 temple_3

Figure 6: Some challenging cases from the MPI Sintel (Butler et al. 2012) dataset include heavily occluded static backgrounds and significant motion blur, both of which complicate the camera optimization process.

Movement Clustering Since the optical flow constraint operates in 2D space, multiple Gaussian points may exist along the 2D camera view ray in 3D space, especially when occlusion occurs. Therefore, after movement clustering, we freeze all center coordinates µi of static points Gsit to prevent them from being displaced by optical flow. The movement mask is identified by thresholding the epipolar error map, with the threshold set to 0.01.

Camera Optimization When optimizing the camera pose, only the static part serves as a reference, and the moving part does not contribute or even be harmful to pose estimation. Therefore, we need to exclude the moving part, M = Mt ∪ Mt′−1, which is the union of the current moving mask Mt and the previous moving mask Mt′−1 in the new view. The mask Mt′−1 is determined as follows: first, identify the previous moving Gaussian points {Gmi } within the previous moving mask Mt−1 from the previous frame. Then, splat these points {Gmi } using the optimized camera pose E∗ to get the moving part image Iˆm. Finally, threshold the grayscale image of the moving part image grey(Iˆm) with a threshold of 0 to obtain the mask Mt′−1.

Optimization Choices It is worth noting that, although all experiments follow the same hyperparameter settings, the results can be further improved by optimizing these settings for each specific case. This is reasonable because the dynamics and content of videos vary significantly (e.g., Figure 6).

For example, for a video with large camera motion, we

GFlow (Ours) Original frames GFlow*

[Figure 108]

[Figure 109]

[Figure 110]

Frame 0

[Figure 111]

[Figure 112]

[Figure 113]

Frame 20

[Figure 114]

[Figure 115]

[Figure 116]

Frame 40

[Figure 117]

[Figure 118]

Global

- 3D view

GFlow (Ours) GFlow*

Figure 7: 3D point tracking visualization example (blackswan) on the DAVIS (Perazzi et al. 2016; Pont-Tuset et al. 2017) dataset. The colorful dots and trajectories indicates the movement in 3D world coordinates. The red box helps the reader identify the reference plants in the background. The camera is moving to the right, and the black swan is also moving to the right in the video.

can increase the learning rate or extend optimization iterations in the camera optimization process. For videos with clear and simple rigid motions, we can directly use the estimated camera poses (Leroy, Cabon, and Revaud 2024) instead of optimizing them to shorten the overall processing time.

### B More Experiments

#### B.1 Evaluation of Camera Pose Estimation

Dataset and Metrics MPI Sintel (Butler et al. 2012) dataset provides high-quality, synthetic sequences with complex motion, realistic lighting, and challenging visual effects like motion blur and depth of field. Following prior works (Liu et al. 2023), we evaluate 14 sequences with ground-truth camera poses provided. As for camera pose accuracy, we report standard visual odometry metrics (Sturm

et al. 2012; Zhang and Scaramuzza 2018), including the Absolute Trajectory Error (ATE) and Relative Pose Error (RPE) of rotation and translation as in (Bian et al. 2023; Lin et al. 2021a).

Results Our method can reconstruct the 4D world along with the corresponding camera poses. The MPI Sintel dataset presents highly challenging dynamics which makes camera pose estimation more difficult, such as large occlusion by moving objects and severe motion blur. To alleviate these challenges, we combine the strengths of both optimizing camera poses and using estimated camera poses (Leroy, Cabon, and Revaud 2024) as initialization. So we combines the best results from both settings.

In Table 2, we compare the camera pose estimation results with R-CVD (Kopf, Rong, and Huang 2021), DROIDSLAM (Teed and Deng 2021), COLMAP (Sch¨onberger and Frahm 2016), NeRF– (Wang et al. 2021), BARF (Lin

et al. 2021a), and RoDynRF (Liu et al. 2023). Our method, GFlow, achieves comparable or better results in camera pose estimation compared to previous methods, demonstrating its effectiveness.

#### B.2 Ablation study

Effect of optimizing camera pose As described in the main text, we report the reconstruction results between ‘GFlow (ours)’ (optimizing camera pose) and ‘GFlow*’ (directly using the camera poses estimated by MASt3R). Although the numerical differences are not significant, the camera pose estimation and movement are incorrect in the

- 4D world. We show an example in Figure 7. In this video, a black swan is swimming to the right on a river, and the camera is also moving to the right. The tracking trajectories and global 3D view of ‘GFlow (ours)’ show the correct moving direction and camera poses, while ‘GFlow*’ fails. Due to the flowing water, most of the frames are moving, making it difficult for MASt3R to estimate the correct camera poses. Please refer to the supplementary videos for a clearer illustration.

