### EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents

Wenjia Wang1∗ Liang Pan1∗ Huaijin Pi1 Yuke Lou1 Xuqian Ren2 Yifan Wu1 Zhouyingcheng Liao1 Lei Yang3 Rishabh Dabral4 Christian Theobalt4 Taku Komura1

(*: Core contributor.)

1The University of Hong Kong 2Tampere University 3The Chinese University of Hong Kong 4Max-Planck Institute for Informatics

# arXiv:2602.23205v2[cs.CV]2Apr2026

[Figure 1]

[Figure 2]

[Figure 3]

## EmbodMocap

[Figure 4]

[Figure 5]

[Figure 6]

What We Don’t Need

What We Can Get

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

RGBD Frames Camera Params

Multi-View Studio

[Figure 16]

[Figure 17]

Mocap Suit

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Scene Meshes

[Figure 23]

LiDAR Sensor Device Sync.

[Figure 24]

[Figure 25]

Human Motions

View 1 View 2

[Figure 26]

[Figure 27]

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

[Figure 38]

Monocular Human & Scene Reconstruction Physics-based Character Animation Real-world Humanoid Motion Control

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

|[Figure 45]<br><br>[Figure 46]<br><br>|
|---|

|[Figure 47]<br><br>[Figure 48]|
|---|

[Figure 49]

[Figure 50]

[Figure 51]

###### RGBD

|[Figure 52]|
|---|

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

|[Figure 59]|
|---|

𝝅𝟑

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Imitation Policy

[Figure 67]

T

[Figure 68]

[Figure 69]

Skill Policy

Imitation Policy

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

VIMO

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Cameras

|[Figure 82]|
|---|

[Figure 83]

[Figure 84]

[Figure 85]

Imitation Policy

Figure 1. Introducing EmbodMocap, a portable and low-cost system for simultaneous 4D human and scene reconstruction, deployable anywhere using two moving iPhones. The dataset captured by EmbodMocap benefits three crucial embodied AI tasks: monocular human & scene reconstruction, physics-based character animation, and real-world humanoid motion control. Project page.

##### Abstract

consistent capture in everyday environments without static cameras or markers, bridging human motion and scene geometry seamlessly. Compared with optical capture ground truth, we demonstrate that the dual-view setting exhibits a remarkable ability to mitigate depth ambiguity, achieving superior alignment and reconstruction performance over single iphone or monocular models. Based on the collected data, we empower three embodied AI tasks: monocular human-scene-reconstruction, where we fine-tune on feedforward models that output metric-scale, world-space aligned humans and scenes; physics-based character animation, where we prove our data could be used to scale human-object interaction skills and scene-aware motion tracking; and robot motion control, where we train a hu-

Human behaviors in the real world naturally encode rich, long-term contextual information that can be leveraged to train embodied agents for perception, understanding, and acting. However, existing capture systems typically rely on costly studio setups and wearable devices, limiting the large-scale collection of scene-conditioned human motion data in the wild. To address this, we propose EmbodMocap, a portable and affordable data collection pipeline using two moving iPhones. Our key idea is to jointly calibrate dual RGB-D sequences to reconstruct both humans and scenes within a unified metric world coordinate frame. The proposed method allows metric-scale and scene-

manoid robot via sim-to-real RL to replicate human motions depicted in videos. Experimental results validate the effectiveness of our pipeline and its contributions towards advancing embodied AI research.

##### 1. Introduction

Embodied Artificial Intelligence (Embodied AI) aims to build agents that can perceive, understand, and act within real-world environments. Progress in this field relies on datasets that capture both human motion and the surrounding 3D scene, enabling physically grounded perception and action learning. Such scene-aware data allows modeling of realistic human–scene interactions, simulation of lifelike behaviors, and training of humanoids to operate seamlessly in complex environments. They serve as a foundation for advancing embodied reasoning and control across robotics, virtual reality, and computer vision.

However, collecting high-quality human–scene data remains difficult. Precise 3D motion and scene geometry cannot be automatically obtained from internet videos due to occlusions and depth ambiguity. Existing capture systems that provide high-quality human–scene data typically rely on multi-view camera rigs [12, 76], wearable motion suits [23, 36], or LiDAR scanners [7, 20], which are costly, complex, and limited to controlled studio environments. These constraints hinder scalable and scene-aware data acquisition, limiting the ability of embodied AI models to learn from natural human behavior in diverse indoor and outdoor environments.

In this paper, we propose EmbodMocap, an efficient and affordable framework for capturing metrically accurate 4D human and scene using only two iPhones. Our key idea is to jointly calibrate and optimize dual RGB-D inputs to reconstruct both humans and scenes within a unified world coordinate frame. Specifically, we first reconstruct the static scene from a single RGB-D sequence to define the world scale, then capture synchronized dual-view RGB-D videos of human motion, and finally perform geometric alignment and motion optimization to recover worldanchored human poses. In contrast to existing systems that rely on multi-camera rigs or wearable sensors, our approach achieves high-quality, scene-consistent reconstruction using only moving consumer devices. This design enables scalable, in-the-wild data collection that preserves precise human motion and authentic scene context, supporting realistic human–scene interaction modeling for embodied AI research.

Based on the data collected with EmbodMocap, we demonstrate the reliability and versatility of our capture pipeline through three representative applications. The first application verifies geometric consistency, where we finetune reconstruction models to jointly recover humans and

scenes in world coordinates. The second validates physical realism, showing that the captured motions enable scalable training of physics-based character skills and scene-aware motion tracking. The third demonstrates embodied transferability, where our data support humanoid robot training through a sim-to-real motion tracking framework [27, 45]. These results highlight that EmbodMocap enables scalable and physically grounded data acquisition for embodied AI.

In summary, our contributions are:

- • EmbodMocap: A portable capture framework that jointly calibrates and optimizes dual moving RGB-D cameras (iPhones) to reconstruct metrically accurate, world-anchored human motions and static scenes without multi-camera setups, mocap suits, or controlled environments.
- • A multi-modal dataset: A collection of high-quality, scene-aware human motion data captured with EmbodMocap across diverse real-world environments, enabling scalable training for embodied AI.

We validate the effectiveness of our method and dataset through experiments in monocular human-scene reconstruction, physics-based character animation, and sim-toreal humanoid control, demonstrating their utility across key embodied AI tasks.

##### 2. Related Work

Datasets for 4D Human & Scene Capture. Early motion datasets, such as AMASS [11, 37], focus on pure human motion, unifying multiple motion capture sources into a large-scale repository. While invaluable for studying human motion, these datasets lack the 3D scene context essential for understanding human–scene interactions. Recent 4D datasets, like PROX [12], RICH [20], and EgoBody [76], combine scanned 3D scenes with motion capture using multi-view camera systems, while EMDB [23] and SPLOPER4D [7], employ IMUs or electromagnetic sensors for motion recording in large-scale environments. Nymeria [36] extends this further with Project Aria glasses and optical marker-based systems for wide-area motion capture. However, these approaches face notable limitations: marker-based and multi-camera systems are expensive and restricted to small studio environments, while IMU and EM-based methods, though more flexible, require extensive manual alignment and post-processing to synchronize motion with 3D scenes. And the wearable devices will influence the human appearance in RGB images. In contrast, our approach uses minimal equipment, operates in diverse environments without static camera setups, and avoids wearable devices, preserving the naturalness of RGB images for authentic human–scene interaction capture. Table 1 compares these datasets.

Monocular Human & Scene Reconstruction. Early works [4, 9, 22, 25, 43] on RGB-based human mesh re-

Table 1. Comparison of 4D Human & Scene datasets based on different features.

Device Outcome

Datasets Publication

Mocap Suit Scanner Static Cam. Dyna. Cam. Total Cost($) Mesh Dyna.Anno. Outdoor

PROX [12] ICCV2019 - Structure Sensor Kinetic-One - 2K ✓ ✗ ✗ RICH [20] CVPR 2022 - Leica RTC360 6-8×Cameras 1×Camera 20K+ ✓ ✓ ✓ EgoBody [76] ECCV2022 - 1×IPhone 5×Azure Kinect Hololens2 9K ✓ ✓ ✗ SLOPER4D [7] CVPR2023 Noitom PN+NUC11 Ouster-os1 LiDAR - DJI-Action2+TLS 20K ✓ ✓ ✓ EMDB [23] ICCV 2023 EM Sensors - - 1×IPhone 15K ✗ ✓ ✓ Nymeria [36] ECCV2024 2×XSens+Aria Wistband - - 2×Project Aria 60K+ ✗ ✓ ✓ EmbodMocap - - 1×IPhone - 2×IPhone 1K ✓ ✓ ✓

covery focus on reconstructing 3D pose and shape but often ignore scene context [61] or camera information [26,

- 64], leading to inconsistencies under camera motion. Recent methods address this by combining motion cues [75], SLAM or visual odometry [56, 66, 74], and human motion priors [55, 75] to recover global trajectories in world coordinates.

Emerging models move toward jointly reconstructing humans and 3D scenes with spatial intelligence models [62, 63]. For example, HSFM [39] combines Dust3R [63] with multi-view correspondence to jointly recover human meshes, scene point clouds, and camera parameters from multi-cameras. HAMSt3R [50] integrates DensePose [10] and multi-view scene reconstruction in one model, with an optimization to get human poses, while JOSH [30] uses MASt3R-SLAM [40] and joint optimization to achieve globally consistent 4D human-scene reconstructions. Human3R [5] introduces a unified, feed-forward framework for online 4D human-scene reconstruction, jointly recovering multi-person SMPL-X bodies and dense scene point clouds in a global world frame from monocular videos. Crisp [68] presents a contact-guided Real2Sim pipeline that recovers simulatable human motion and scene geometry by fitting compact planar primitives and leveraging humanscene contact cues to hallucinate occluded interaction surfaces. This trend emphasizes the simultaneous prediction of human motion and scene geometry, which further requires multi-model data pairs with high-quality annotations. In our paper, we propose a monocular human & scene reconstruction pipeline combined with 2 feedforward models, and finetuned it on our proposed dataset to prove the efficiency of our paired data.

Training Humanoid from Video Data. Recent advances in physics-based animation and reinforcement learning enable humanoid agents to perform realistic and physically consistent motions using control policies learned from marker-based motion capture data. These methods have shown strong realism in tasks like motion tracking [33, 45], locomotion [34, 46, 47], and human–scene interaction [42,

- 65], and have been extended to real-world applications in motion tracking [16, 18, 21], locomotion [17], and scene interaction [3, 15]. However, marker-based methods require

dedicated studios, expensive hardware, and extensive manual effort, making them costly and hard to scale. Adapting captured motions to new scenes or robot morphologies also demands complex retargeting and re-simulation. To address this, recent works like VideoMimic [2], ASAP [18], and HDMI [69] train humanoid control directly from inthe-wild video data. By using monocular motion capture methods such as TRAM [66] and GVHMR [55], they estimate human motion from videos and retarget it to virtual humanoids for training in physical simulators. This videodriven paradigm leverages diverse real-world data but struggles with capturing complex skills or scene geometries due to occlusion and depth ambiguities. In this paper, we propose a method for high-precision human motion and scene reconstruction that overcomes these limitations.

##### 3. Proposed Capture System

We aim to capture metrically accurate human motion and scene geometry using only two iPhones. As shown in Fig. 2, our capture process consists of four sequential stages that progressively reconstruct and align the scene, cameras, and human motion within a unified world coordinate frame. We first reconstruct a metrically accurate static scene and establish the world reference using a single iPhone RGBD sequence (Sec. 3.1). Then, we use two synchronized iPhones to record dual-view RGB-D videos of human motion and extract per-frame camera poses and human priors with off-the-shelf perception models (Sec. 3.2). Next, we align the dual-view camera trajectories to the reconstructed scene through a combination of COLMAP registration and multi-view geometric optimization (Sec. 3.3). Finally, we refine the SMPL parameters by triangulating dual-view 2D keypoints into 3D space and optimizing human poses and translations in the world coordinate system (Sec. 3.4).

###### 3.1. Stage I: Scene Reconstruction

In this stage, we aim to reconstruct a metrically accurate, Z-up scene mesh that serves as the reference world coordinate system. We first use a single iPhone to capture an RGB-D video of the scene, along with synchronized IMU data. The recorded data are processed by the SpectacularAI

###### Scene Reconstruction Sequence Processing Sequence Calibration Motion Optimization

[Figure 86]

[Figure 87]

###### Triangulation

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

###### COLMAP

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

Bundle

Feature Extraction VocabTree Matching

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Adjustment

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

cameras1

[Figure 118]

Image Registration

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

View1 View2

View Scene

###### View1

[Figure 125]

[Figure 126]

View2

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

SpectacularAI

SpectacularAI

[Figure 134]

###### SpectacularAI

PromptDA

[Figure 135]

cameras2

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

3D Keypoints

View1 View2

|𝑹෡𝒗𝟏,𝒕,𝑻෡𝒗𝟏,𝒕|
|---|

|𝑹𝒗𝟏,𝒕,𝑻𝒗𝟏,𝒕|
|---|

|𝑹𝒗𝟐,𝒕,𝑻𝒗𝟐,𝒕|
|---|

|𝑹෡𝒗𝟐,𝒕,𝑻෡𝒗𝟐,𝒕|
|---|

[Figure 146]

###### SMPLilfy

[Figure 147]

[Figure 148]

|𝑲𝒗𝟏,𝑹𝒗𝟏,𝒕,𝑻𝒗𝟏,𝒕|
|---|

|𝑲𝒗𝟐,𝑹𝒗𝟐,𝒕,𝑻𝒗𝟐,𝒕|
|---|

Calibration

|𝑲𝒔,𝑹𝒔,𝒏,𝑻𝒔,𝒏|
|---|

VITPose VIMO Regularization

Reprojection Smoothness 3D-Keypoints

[Figure 149]

COLMAP

[Figure 150]

[Figure 151]

[Figure 152]

UnProject TSDF

[Figure 153]

Database

|[Figure 154]<br><br>[Figure 155]<br><br>[Figure 156]|
|---|

[Figure 157]

PromptDA SAM2

|[Figure 158]|
|---|

[Figure 159]

[Figure 160]

COLMAP

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

Bundle Adjustment

Point Tracking

Chamfer Distance

Feature Extraction Matching

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Triangulation

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

|[Figure 175]|
|---|

[Figure 176]

[Figure 177]

[Figure 178]

|𝑹𝒗𝒂𝒍𝒊𝟏,𝒕,𝑻𝒗𝒂𝒍𝒊𝟏,𝒕|
|---|

|𝑹𝒗𝒂𝒍𝒊𝟐,𝒕,𝑻𝒗𝒂𝒍𝒊𝟐,𝒕|
|---|

COLMAP Database

Scene & Motion

Scene Mesh

Calibrated Cameras

View1 Information View2 Information

Figure 2. EmbodMocap: We propose an affordable dataset capture and processing system. From left to right, the four stages (StageI to Stage-IV) illustrate our core logic: leveraging high-quality camera matrices provided by SpectacularAI [1] and aligning sequence coordinates to the scene’s world frame. For detailed explanations, please refer to Sec. 3.

SDK (SAI) [1], which automatically selects keyframes according to the accumulated camera translation and estimates corresponding camera parameters (Ks,Rs,n,Ts,n) in Z-up world coordinates with metric scale. These trajectories establish a consistent world frame for all subsequent stages. Based on the recovered poses, we refine the iPhone LiDAR depth maps using PromptDA [28], unproject them into 3D space, and integrate the point clouds through TSDF fusion [6] to obtain a dense and metrically accurate global mesh Mg. Note that the depth maps are truncated based on a threshold determined by the effective range of the iPhone’s depth sensor. Specifically, we use a threshold of 3.5m for indoor scenes and 5m for outdoor scenes. We further apply lightweight post-processing such as outlier removal and small-component filtering to clean the mesh. Finally, we extract SIFT features from the same SAI keyframes and run COLMAP [52] with fixed camera parameters to build a sparse structure database. This database preserves the metric scale and serves as a reference for registering dual-view sequences in later stages.

cameras for each view. Let v denote the view index (v ∈ {v1,v2}), and let t denote index time. For each view independently, SAI provides intrinsics and extrinsics (Kv,Rv,t,Tv,t) for every decoded frame Iv,t in the native coordinate system of that view.

Next, we extract human-related information using several off-the-shelf models: (i) YOLO [57] for person detection and proposal pruning; (ii) ViTPose [72] for 2D human keypoints with confidence scores; (iii) SAM2 [49] for person segmentation masks; (iv) PromptDA [28] to refine dual-view depths; and (v) VIMO [66] for camera space SMPL parameters. Finally, we employ a laser pointer cue for frame-level synchronization between the two camera streams. By identifying the frame index where the laser dot disappears, we temporally align both videos and slice all associated image, depth, and parameter data accordingly. This process yields synchronized dual-view RGB-D sequences with calibrated camera trajectories and per-frame human priors, providing clean inputs for subsequent sequence calibration.

###### 3.2. Stage II: Sequence Processing

After reconstructing the static scene in Stage I, we proceed to capture and process dual-view human motion sequences within the same environment. In this stage, we use two iPhones to record synchronized RGB-D videos of a performer moving inside the reconstructed scene, with each device providing an independent camera coordinate system. The goal is to convert these raw dual-view videos into temporally aligned and metrically consistent per-frame human and camera information, which will serve as the foundation for subsequent calibration and motion optimization.

Firstly, we use SAI to obtain per-frame calibrated

###### 3.3. Stage III: Sequence Calibration

After obtaining the static scene reconstruction in Stage 3.1 and the dual-view camera trajectories in Stage 3.2, the next step is to align all coordinate systems into a unified world frame. At this point, we have three separate coordinate systems: one for the reconstructed scene and two for each iPhone camera trajectory estimated by SAI. Since the dualview coordinate systems differ from the scene coordinate system only by rigid transformations, our goal is to optimize these 2 rigid transformations to unify the dual-view coordinates into the same metric, gravity-aligned world frame. The optimization process is sensitive to the initial values;

therefore, it is necessary to first obtain a good initial estimate for the rigid transformations.

Get Initial Transformation from COLMAP. We register each dual-view sequence to the sparse COLMAP model constructed in Stage 3.1 using the known intrinsics Kv and background-only SIFT features Fv, extracted from images with human regions removed. Matches are established through a trained vocabulary tree [53], and images are registered against the sparse COLMAP model to obtain COLMAP camera poses (Rˆv,t,Tˆv,t) in the same metric, gravity-aligned world coordinates as the scene.

To obtain the initial rigid transformation aligning the SAI camera trajectories Tv,t with their COLMAP counterparts Tˆv,t, we solve for an offset transformation (soff,Roff,Toff) by minimizing:

N

min

soff,Roff,T off

t=1

T ˆv,t − (soffRoffTv,t + Toff) 22, (1)

where N is the number of frames. After centering the trajectories, we solve this minimization problem using singular value decomposition (SVD).

For gravity alignment, Roff is constrained to rotations about the z-axis, ensuring proper alignment of SAI trajectories with the COLMAP coordinate system.

Calibration via Multiple Constraints. While the rigid transformations obtained in the previous step provide coarse alignment between the two camera trajectories and the reconstructed scene, this initialization alone is not sufficient to achieve accurate synchronization and metric consistency. To further refine the calibration, we jointly optimize all alignment parameters by introducing multiple geometric and photometric constraints across views. Specifically, we optimize the per-view global offsets Rvoff (constrained to zaxis rotations) and Tvoff, using the initial alignment as the starting value. The aligned camera extrinsics are:

###### Rv,tali = RvoffRv,t, Tv,tali = RvoffTv,t + Tvoff. (2)

The optimization minimizes a composite loss of point tracking loss, Chamfer distance, and bundle adjustment loss to ensure spatial consistency between views and the global reconstruction.

Lcalib = λtrackLtrack +

λbaLba,v.

λchdChamfer +

v

v

(3) where each loss is defined in the rest of this seciton.

Through VGGT tracking, a subset of keyframes is selected, yielding accurate dual-view pixel tracking results in the human masks region. The tracked human surface 2D

pixel coordinates qv,t(i), along with their corresponding depth

values d(v,ti), are back-projected into the world frame:

Q(v,ti) = d(v,ti)Rv,t⊤aliKv−1

qv,t(i) 1

+ Rv,t⊤aliTv,tali, (4)

To enforce track consistency between views, the following loss is minimized:

1 v,t |Qv,t| t i

Ltrack =

w˜t(i) Q(1i,t) − Q(2i,t) 22, (5)

where Q(1i,t) and Q(2i,t) are the 3D back-projected coordinates of the i-th point from view 1 and view 2, respectively.

The weights w˜t(i) are used to control the contribution of each point based on its tracking confidence. Here w˜t(i) = min(w1(i,t),w2(i,t)) combines the VGGT confidence scores for the same point across views. The Chamfer distance term dChamfer aligns local pointclouds Pv (v ∈ {v1,v2}) with the global reconstruction Pg sampled from Mg in Sec. 3.1, where Pv is obtained by reconstructing the scene using the method from Sec. 3.1 with humans cropped by masks. The Chamfer distance is formally defined as:

1 |Pv| p

dChamfer(Pv,Pg) =

v∈Pv

1 |Pg| p

+

∥pv − pg∥22

min

pg∈Pg

∥pg − pv∥22.

min

pv∈Pv

g∈Pg

(6)

Finally, Lba,v (v ∈ {v1,v2}) ensures reprojection consistency for persistent matches, where the points are obtained from COLMAP image registration:

1 |Mv|

xv,t,j −π(Kv,Rv,tali,Tv,tali,Xj) 22. (7) We solve Eq. (3) using the Adam [24] optimizer with

Lba,v =

(t,j)∈Mv

gradient clipping. For yaw-only updates, Rvoff is parameterized by a single z-axis angle to preserve gravity alignment.

###### 3.4. Stage IV: Motion Optimization

After obtaining calibrated dual-view trajectories and a unified scene coordinate system in Stage 3.3, we further refine the human reconstruction results to achieve accurate and temporally consistent body motions in the world frame. At this stage, both camera poses and scene geometry are fixed, allowing us to focus on optimizing the human parameters. We first triangulate dual-view 2D keypoints into world-space 3D keypoints, which serve as reliable geometric constraints across views. Then, we optimize the SMPL parameters using these triangulated 3D keypoints to recover

precise body poses and translations under the unified world coordinate system.

3D Keypoint Triangulation. To triangulate the 3D keypoints Yt,j from their 2D projections {yv,t,j}, we estimate the 3D position by minimizing the weighted reprojection error across all views:

V

cv,t,j yv,t,j − PvYt,j 22, (8)

min

Yt,j

v=1

The Ltrack effectively stitches the two views together, significantly improving the overall reconstruction performance, making it highly impactful on the final results. The Lkp3d provides 3D joint positions of the human body, and compared to the reprojection loss, it eliminates the issue of depth ambiguity, thus playing a critical role in the overall performance.

Table 2. The performance of different optimization settings.

where Pv = Kv[Rv,t | Tv,t] is the camera projection matrix for the v-th view. The problem can be formulated as a weighted least squares optimization. Using SVD, Yt,j is obtained as the right singular vector corresponding to the smallest singular value of A.

Ltrack Lchamfer Lreproj Lsmooth Lkp3d IoU(%)↑ Reproj↓ Depth↓ Jitter↓ ✗ ✓ ✓ ✓ ✓ 54.3 44.2 2.372 0.0371 ✓ ✗ ✓ ✓ ✓ 72.5 10.9 0.081 0.0131 ✓ ✓ ✗ ✓ ✓ 72.3 11.1 0.079 0.0130 ✓ ✓ ✓ ✗ ✓ 72.1 10.4 0.087 0.0160 ✓ ✓ ✓ ✓ ✗ 59.3 20.4 0.609 0.0126 ✓ ✓ ✓ ✓ ✓ 73.0 9.3 0.078 0.0128

World-Space SMPLify. Start from initial shape β0 and body pose θtb,0 in Sec. 3.2, our World Frame SMPLify [31] jointly optimizes shape β ∈ R10, per-frame pose θt = {θtg,θtb} ∈ R72 and root translation γt ∈ R3 by minimizing:

###### 4.2. Comparison on Capture Methods

Direct comparison in optical mocap studio. To evaluate the accuracy of dual view capture system, we set up furniture in a mocap studio and use a Vicon system to capture ground truth human motion. Two photographers record dual-view videos of the actor with iPhones, while the actor performs basic motions (see Fig. 3, zoom in). We record 5 sequences of one participant with 9420 frames in total. We compare the errors against optical mocap GT of: monocular model GVHMR, our dual-view optimization, and our single-view version (v1 and v2). For the single-view version, we calibrate the actor coordinates to the scene coordinates system using COLMAP and optimize the motion with reprojection, smooth, and prior losses. The optical mocap results are fitted to SMPLX parameters by Mosh [32] and synchronized to dual-view parameters with foot contact keyframs. Results are compared in chunk sizes of 100, 500, and 1000. Our dual-view method outperforms the monocular model and single-view optimization by a large margin. As the chunk length increases, our advantage becomes increasingly evident (see Tab. 3).

LSMPLify = L3D + Lsmooth + Lprior + Lreproj (9)

We use a two-stage optimization phase to ensure the smoothness and alignment with the original dual views. For the first stage, we only fit the body shape and transition, and for the second stage we fit all the parameters.

##### 4. Evaluation

In this section, we aim to prove the effectness of our optimization pipeline. We will first ablate different loss functions of the pipeline in Sec. 4.1, then compare ours with the monocular model, single-view only and optical captured ground truth.

###### 4.1. Ablation Study on Loss Functions

Ablation on dataset optimization. We conduct an ablation study on four core loss functions that significantly influence performance during data optimization, as described in main paper. These loss functions include tracking loss, Chamfer distance, reprojection loss, smoothness loss and kp3d loss. To evaluate the performance under different optimization settings, we employ four metrics. First, IoU(Intersection over Union) measures the overlap between the rendered SMPL mask and the SAM2 [49] mask. Second, Reproj evaluates the pixel error between the reprojected SMPL joints and the 2D keypoints detected by VITPose [72]. Third, Depth error is computed as the mean squared error (MSE) between the rendered depth from SMPL parameters and the sensor depths refined by PromptDA [28]. Finally, Jitter is quantified using the same temporal foot skating metric as MotionVAE [29]. All metrics are averaged across all sequences and views to ensure a robust evaluation.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

(a) Captured view1 (b) Captured view2

[Figure 187]

(c) Optimized view1 (c) Optimized dual-view (c) Optimized view2

Figure 3. Our dual view vs. single view results in optical studio.

The advantage of dual-view over single-view lies in two key aspects: 1) dual-view effectively addresses occlusion

Table 3. Comparision among monocular model, single view optimization, with dual view optimization(ours)

|Method<br><br>|chunk=100|chunk=500<br><br>|chunk=1000<br><br>|RTE↓|
|---|---|---|---|---|
| |WA-MPJPE↓ W-MPJPE↓|WA-MPJPE↓ W-MPJPE↓|WA-MPJPE↓ W-MPJPE↓| |
|GVHMR Single-View V1 Single-View V2 Dual View|66.56 123.44 124.68 218.22 108.31 211.83 56.61 72.86<br><br>|124.61 333.34 233.06 489.11 231.41 357.22 76.90 99.75|179.47 593.79 297.83 768.31 338.42 762.80 119.45 169.11<br><br>|1.85 2.71 3.65 1.13|

and self-occlusion of body joints, 2) it handles the challenging alignment of actor motion coordinates to the scene coordinates. The COLMAP estimates the camera locations for the images but suffers from depth ambiguity in the camera’s facing direction. Using a single iPhone results in large errors in the depth direction. In contrast, using two iPhones enables pixel-wise dense correspondence (see Eq. (5)), which ensures the rigid transformation between the two cameras during the optimization, and resolves the depth ambiguity in each view. This enables a good localization of human trajectories in the scene coordinate system automatically. Our dual view could achieve a calibration accuracy to the scene of about 5cm (human touching table in the figure), while the single view is over 30cm, measured in MeshLab by putting markers on the ground for the actor’s start and end positions.

parameters. A human mask was used to limit supervision to the human region due to our dataset’s smaller range.

Metrics. We evaluate motion and trajectory accuracy on global coordinates using EMDB (subset 2) [23], featuring extended sequences with ground-truth trajectories and meshes. Consistent with prior work [56, 66], each sequence is split into 100-frame chunks, and 3D joint errors are measured using W-MPJPE (aligning the first two frames) and WA-MPJPE (aligning the entire segment), both in millimeters. Additionally, Root Translation Error (RTE) is reported as a percentage (%), normalized by total displacement after rigid alignment (excluding scaling).

Results. We present 3 variants in Tab. 4: the proposed baseline with the original checkpoints from π3 [67] and VIMO [66], fine-tuning only VIMO, and fine-tuning both π3 and VIMO. The results demonstrate that our approach significantly improves the accuracy of VIMO, as we provide paired high-quality real-world RGB sequences and ground truth SMPL parameters. Additionally, leveraging our highquality RGB-D data and camera parameter pairs, π3’s ability to predict in the world coordinate system also shows improvement. Our pipeline shows good performance on largescale real-world videos, see Fig. 4

##### 5. Downstream Tasks

Table 4. Comparison of Finetuned Models on EMDB Benchmarks

In this section, we validate our capture pipeline’s effectiveness across three key applications. In Sec. 5.1, we propose

Finetuned EMDB Pi3 VIMO WA-MPJPE↓ W-MPJPE↓ RTE↓

- ✗ ✗ 83.56 229.04 1.78

- ✗ ✓ 82.89 222.93 1.73

- a monocular human & scene reconstruction pipeline and finetune it with our captured RGBD, cameras, and SMPL annotations. In Sec. 5.2, we train several human-object interaction skills and scene-aware motion tracking with our captured motion & scene. In Sec. 5.3, we train a humanoid in simulator and deploy it to real-world robot.

###### ✓ ✓ 82.21 220.65 1.71

|[Figure 188]|
|---|

[Figure 189]

[Figure 190]

|[Figure 191]|
|---|

###### 5.1. Monocular Human & Scene Reconstruction

Motivation. We propose a data scheme combining RGBD data from dynamic cameras with camera and human motion parameters to train monocular human and scene reconstruction models. As no feedforward model exists, we establish a baseline using π3[67] for SLAM and VIMO[66] for metricscale human motion reconstruction from monocular videos. Implementation. To process long sequences, videos are divided into overlapping chunks, with π3 estimating camera parameters and local point maps per chunk. Adjacent chunks are aligned using Procrustes alignment, and scale/transformations are recursively applied for global consistency. Metric scale is determined as the median ratio of SMPL to π3 depth values. SMPL predictions are then transformed to metric world space. For details, refer to Supp. Mat. We fine-tuned two π3 variants Tab. 4 by adding LoRA [19] layers to the camera and point decoders, supervised with the original π3 loss. For VIMO, we froze the encoder and finetuned the decoder with MSE loss on SMPL

|[Figure 192]|
|---|

|[Figure 193]|
|---|

[Figure 194]

|[Figure 195]|
|---|

|[Figure 196]|
|---|

[Figure 197]

[Figure 198]

|[Figure 199]| |
|---|---|
| | |

|[Figure 200]|
|---|

|[Figure 201]|
|---|

[Figure 202]

|[Figure 203]|
|---|

|[Figure 204]|
|---|

|[Figure 205]|
|---|

[Figure 206]

[Figure 207]

Figure 4. Quality results of proposed 4D Human & Scene Reconstruction pipeline on EMDB dataset.

###### 5.2. Physics-based Character Animation

###### 5.2.1. Human Object Interaction Skill Training

Motivation. We train several human-object interaction skills to demonstrate the physical realism of our approach

Table 5. Comparison of data duration, Success Rate, Contact Error, and APD for different skills among 3 data settings.

and the scalability of our capture framework to new interaction skills. We aim to prove the efficiency and quality superiority of our framework over optical capture and monocular estimation methods.

Task Data Clips Duration (min) Rate (%) ↑ Error (cm) ↓ APD ↑

Optical Mocap 12 1.59 99.9 6.0 20.17 ± 0.19

Implementation. Following [42, 46, 65], we train physical character policies use goal-conditioned reinforcement learning to formulate character control as a Markov Decision Process (MDP) defined by states, actions, transition dynamics, a reward function r, and a discount factor γ. The reward rt ∈ R is calculated by a style reward rtstyle [46] and a task reward rttask. The policies are trained to maximize the expected discounted return: J(π) = Ep(τ|π) Tt=0−1 γtrt , where T is the episode length, γ ∈ [0,1] is the discount factor, and rt is the reward at time step t. We use the widely adopted Proximal Policy Optimization (PPO) algorithm [54] to train the control policy model.

- Ours 1X 12 1.48 99.9 6.7 18.42 ± 0.22

- Ours 2X 24 3.06 99.7 6.8 18.45 ± 0.17

Follow

Ours Full 148 22.43 99.8 6.2 19.69 ± 0.32 Monocular 148 22.43 98.0 7.2 19.85 ± 0.39

Optical Mocap 7 0.28 99.9 2.7 22.03 ± 0.30

- Ours 1X 7 0.54 99.8 1.8 22.77 ± 0.29

- Ours 2X 14 0.97 99.9 1.8 20.72 ± 0.30

Climb

Ours Full 21 1.54 99.9 1.8 22.22 ± 0.27 Monocular 21 1.54 99.2 1.8 21.34 ± 0.38

Optical Mocap 20 4.08 98.0 5.5 16.07 ± 0.39

- Ours 1X 20 2.11 99.8 5.4 14.35 ± 0.27

- Ours 2X 40 4.47 99.9 5.1 14.46 ± 0.24

Sit

Ours Full 80 8.05 99.9 4.7 15.90 ± 0.51 Monocular 80 8.05 98.4 5.7 15.80 ± 0.51

Optical Mocap 10 2.52 89.0 17.5 8.76 ± 0.14

- Ours 1X 10 0.99 85.3 20.2 7.43 ± 0.10

- Ours 2X 20 2.32 86.3 19.8 8.27 ± 0.06

Lie

Ours Full 39 4.25 89.4 18.8 8.57 ± 0.10 Monocular 39 4.25 81.2 21.0 8.14 ± 0.10

###### Ours Full 3 0.26 75.4 16.5 17.58 ± 0.69

Prone

Monocular 3 0.26 71.2 16.5 16.18 ± 0.30 Support

Ours Full 8 0.97 66.0 4.9 21.08 ± 0.59 Monocular 8 0.97 20.6 6.4 20.94 ± 0.48

Following [14, 42, 65], we train a set of human object interaction skills in simulator [38], including follow, climb, sit, and lie. These common interaction skills are designed to guide the character’s root joint to reach specific target positions in 3D environments while maintaining physically realistic and motion divisty. We train these four common skills on 3 different input data: optical captured, which are collected from AMASS [37] and SAMP [13] following TokenHSI [42]; ours, by segmenting the reconstructed motions into skill clips; monocular, by using the motion predicted by GVHMR [55] which is commonly used in humanoid reference motion prediction [18, 69], segmented with the same temporal slices as ours. We also train 2 extra interaction skills which have not been implemented in previous physics-based human object interaction papers: Prone and Support. We will illustrate the observation, reward designs, and the training details of each skill in Supp.Mat.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

OpticalMonocularOurs

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Walk Climb Sit Lie

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

MonocularOurs

Metrics. We follow [13, 70] that uses Success Rate and Contact Error as the main metrics to measure the quality of interactions quantitatively. Success Rate records the percentage of trials that humanoids successfully complete the contact within a certain threshold. We follow [14, 41, 70] in setting the thresholds for various actions: 20cm for Sit, Follow, and Climb; 30cm for Lie and Prone; and 10cm for Support. For Support, the error is defined as the distance from the object surface center to the hand center, while also taking into account the distance between the two feet. Please see details in Supp.Mat. We evaluate motion diversity using Average Pairwise Distance (APD) [8], which measures the average pairwise distance between joint rotations and positions in generated samples. Higher APD values indicate greater diversity.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

Prone Support

Figure 5. Qualitative comparison of 4 basic skills and 2 additional skills.

completion trajectories and motion diversities, which contribute to improve task performance. To prove this, we ablate on skills trained with different data proportions. 1X and 2X indicate the ratio of the number of clips relative to the optical capture data. On the 4 common skills, we observe a general trend where increased data amount leads to improvements in success rate, contact error, and APD metrics.

Results. We can find in Tab. 5, for skills such as Follow, Climb, and Sit, the inherent difficulty is relatively low, and all three data settings achieve good results, very close to 100%. Although the quality of our data is slightly inferior to optically captured data, we provide more variety of task

The success in simulating the 2 extra skills, Prone and Support, demonstrates the versatility of our data collection

pipeline. First, these new skills highlight the ability of our approach to generalize to novel interaction tasks. Second, the Support skill significantly increases the level of difficulty. Unlike other tasks, where a humanoid only needs to walk or offload the full body weight onto furniture surface, Support requires the hands to bear the weight of the body while the feet remain close together, demanding much higher accuracy in reference motion generation. This experiment shows that our approach outperforms monocular estimation methods by a large margin, particularly for high-difficulty interaction skills. The success rate trained on monocular estimated motions degrades to only 20% in Tab. 5. In Sec. 5.2.1, we can see policy trained on motion estimated from monocular models could not perform standard Support skill.

###### 5.2.2. Scene-aware Motion Tracking

Motivation. Recent works [34, 35, 47, 58–60, 73] suggest that solving complex tasks requires pre-training on largescale human motion data via motion tracking objectives, in order to obtain reusable and generalizable skill priors. However, existing motion tracking frameworks are mainly built for human-only [33] or single-object interaction [71] scenarios, primarily because current public datasets are concentrated in these settings. We argue that motion tracking pre-training on diverse 3D scenes is equally important, as it also provides rich priors—such as navigation, interaction, and long-horizon task execution. In this work, we mitigate this gap by: 1) proposing a scene-aware motion tracking framework, and 2) supporting it with high-fidelity paired 3D human-scene data captured by our EmbodMocap system.

Implementation. We extend MimicKit [44] by incorporating the height map into the observation space to achieve scene-aware tracking (details in the Supp. Mat.). For training, we use four 3D scenes, each containing several minutes of motion clips, and train one policy per scene to track all the motion clips in that scene.

Metrics. Policies are evaluated using a success rate metric: an episode is initialized from a random frame and run for 10s, and is considered successful if tracking exceeds 8s. For each scene, 3,072 episodes are used to compute average success, failure rates, and episode length statistics.

Results. The quantitative results in Tab. 6 demonstrate that our data is simulation-ready, enabling the training of sceneaware tracking policies with high success rates. The qualitative results, shown in Fig. 6, further illustrate that the policies not only successfully track the motions but also adapt to subtle imperfections present in the data.

###### 5.3. Real-world Humanoid Robot Control

Motivation. Learning from human videos [2, 48, 69] has emerged as a crucial paradigm for humanoid robots to learn motor skills at scale. In this section, we demonstrate how

Table 6. Quantitative evaluation of scene-aware motion tracking and dataset statistics across four 3D scenes.

Scene Clips Duration (min) Status Rate (%) Eps. Len. (s)

- a 14 12.31

Succ. 87.2 9.97 ± 0.21 Fail. 12.8 3.94 ± 2.10

- b 6 3.62

Succ. 96.7 9.99 ± 0.12

- Fail. 3.3 4.16 ± 2.38

c 12 7.87

Succ. 95.9 9.98 ± 0.17

- Fail. 4.1 5.43 ± 2.18

- d 7 5.06

Succ. 90.4 9.96 ± 0.21 Fail. 9.6 4.44 ± 1.92

EmbodMocap contributes to this paradigm by enabling accurate reconstruction of humans and their interacting 3D environments from videos, while preserving accurate contact information.

Implementation. We capture videos of humans performing ground-contact-rich motions, including locomotion and challenging cartwheels that require precise hand-ground contact. EmbodMocap is then used for real-to-sim reconstruction. The produced motions are used to train a single tracking policy via sim-to-real RL with domain randomization using BeyondMimic [27].

Results. We deploy the policy on a real-world High Torque Hi humanoid robot with 21 joint DoF and a height of 80cm. As shown in Fig. 7, the robot successfully replicates human motions from videos, demonstrating that EmbodMocap produces data of sufficient quality for humanoid robot control.

##### 6. Conclusion

We propose EmbodMocap, a portable and affordable framework for capturing high-quality 4D human & scene data using only two iPhones. Our method enables scalable, metrically accurate reconstruction of human motion and scenes mesh in diverse real-world environments. We directly compare in optical capture studios, and prove the superiority in solving body occlusion and sequence coordinate alignment of our dual view designing. Through downstream applications in monocular human-scene reconstruction, physicsbased character animation, and humanoid robot motion control, we demonstrate the effectiveness and scalability of our approach. By lowering the barrier for embodied AI research, EmbodMocap opens new opportunities for realworld applications.

##### 7. Limitations and Future Work.

Our data collection pipeline encounters limitations in specific scenarios. For example, it fails to record depth when the distance exceeds the range of the iPhone LiDAR sensor (approximately 5 meters). Additionally, it struggles with scenes dominated by moving objects, which degrade the results of the SLAM SDK [1]. Extremely bright lighting

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

(a) Simulation Reference Simulation Reference

(b)

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

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

(c) Simulation Reference

(d) Simulation Reference

- Figure 6. We present qualitative results of scene-aware motion tracking, showing four long-term motion examples in different scenes (a,

b, c, and d), including daily indoor and outdoor interactions such as walking, sitting, lying, stair climbing, and touching. Our motion tracking framework not only accurately tracks the reference motion but also ensures physical realism, resolving subtle issues, such as interpenetration and floating artifacts, present in the reference data (see zoomed-in views on the right).

[Figure 267]

- Figure 7. A real-world humanoid robot imitating human motions depicted in videos.

##### 8. Acknowledge

This work was partially funded by the Innovation and Technology Commission of the HKSAR Government under the ITSP-Platform grants (Ref: ITS/335/23FP, ITS/469/24FP).

We sincerely thank Mr. Xiaohan Ye, Mr. Rui Xu and Mr. Kaiyuan Zheng for volunteering as actors during data collection.

conditions can also cause COLMAP failures, leading to incorrect registration. Future work could integrate more robust structure-from-motion tools, such as H-Loc [51], to improve reliability. Moreover, incorporating automatic synchronization APPs on iPhone could further reduce human effort.

##### References

- [1] Spectacular ai sdk. https://www.spectacularai. com, 2021. 4, 9
- [2] Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, and Angjoo Kanazawa. Visual imitation enables contextual humanoid control. arXiv:2505.03729, 2025. 3, 9
- [3] Qingwei Ben, Feiyu Jia, Jia Zeng, Junting Dong, Dahua Lin, and Jiangmiao Pang. Homie: Humanoid loco-manipulation with isomorphic exoskeleton cockpit. arXiv:2502.13013,

2025. 3

- [4] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J Black. Keep it smpl: Automatic estimation of 3d human pose and shape from a single image. In ECCV, 2016. 2
- [5] Yue Chen, Xingyu Chen, Yuxuan Xue, Anpei Chen, Yuliang Xiu, and Pons-Moll Gerard. Human3r: Everyone everywhere all at once. arXiv:2510.06219, 2025. 3
- [6] Brian Curless and Marc Levoy. A volumetric method for building complex models from range images. In CGIT, 1996. 4
- [7] Yudi Dai, YiTai Lin, XiPing Lin, Chenglu Wen, Lan Xu, Hongwei Yi, Siqi Shen, Yuexin Ma, and Cheng Wang. Sloper4d: A scene-aware dataset for global 4d human pose estimation in urban environments. In CVPR, 2023. 2, 3
- [8] Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, and Wenping Wang. C· ase: Learning conditional adversarial skill embeddings for physics-based characters. In SIGGRAPH, 2023. 8
- [9] Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa, and Jitendra Malik. Humans in 4d: Reconstructing and tracking humans with transformers. In ICCV, 2023. 2
- [10] Rıza Alp G¨uler, Natalia Neverova, and Iasonas Kokkinos. Densepose: Dense human pose estimation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7297–7306, 2018. 3
- [11] F´elix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion in-betweening. 2020. 2
- [12] Mohamed Hassan, Vasileios Choutas, Dimitrios Tzionas, and Michael J Black. Resolving 3d human pose ambiguities with 3d scene constraints. In ICCV, 2019. 2, 3
- [13] Mohamed Hassan, Duygu Ceylan, Ruben Villegas, Jun Saito, Jimei Yang, Yi Zhou, and Michael J Black. Stochastic scene-aware motion prediction. In ICCV, 2021. 8
- [14] Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, and Xue Bin Peng. Synthesizing physical character-scene interactions. In SIGGRAPH, 2023. 8
- [15] Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi. Omnih2o: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. arXiv:2406.08858, 2024. 3
- [16] Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, and Guanya Shi. Learn-

- ing human-to-humanoid real-time whole-body teleoperation. arXiv:2403.04436, 2024. 3
- [17] Tairan He, Wenli Xiao, Toru Lin, Zhengyi Luo, Zhenjia Xu, Zhenyu Jiang, Jan Kautz, Changliu Liu, Guanya Shi, Xiaolong Wang, Linxi Fan, and Yuke Zhu. Hover: Versatile neural whole-body controller for humanoid robots. arXiv:2410.21229, 2024. 3
- [18] Tairan He, Jiawei Gao, Wenli Xiao, Yuanhang Zhang, Zi Wang, Jiashun Wang, Zhengyi Luo, Guanqi He, Nikhil Sobanbabu, Chaoyi Pan, Zeji Yi, Guannan Qu, Kris Kitani, Jessica Hodgins, Linxi ”Jim” Fan, Yuke Zhu, Changliu Liu, and Guanya Shi. Asap: Aligning simulation and real-world physics for learning agile humanoid whole-body skills. arXiv preprint arXiv:2502.01143, 2025. 3, 8
- [19] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 7
- [20] Chun-Hao P Huang, Hongwei Yi, Markus H¨oschle, Matvey Safroshkin, Tsvetelina Alexiadis, Senya Polikovsky, Daniel Scharstein, and Michael J Black. Capturing and inferring dense full-body human-scene contact. In CVPR, 2022. 2, 3
- [21] Mazeyu Ji, Xuanbin Peng, Fangchen Liu, Jialong Li, Ge Yang, Xuxin Cheng, and Xiaolong Wang. Exbody2: Advanced expressive humanoid whole-body control. arXiv:2412.13196, 2024. 3
- [22] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In CVPR, 2018. 2
- [23] Manuel Kaufmann, Jie Song, Chen Guo, Kaiyue Shen, Tianjian Jiang, Chengcheng Tang, Juan Jos´e Z´arate, and Otmar Hilliges. Emdb: The electromagnetic database of global 3d human pose and shape in the wild. In ICCV, 2023. 2, 3, 7
- [24] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv:1412.6980, 2014. 5
- [25] Muhammed Kocabas, Nikos Athanasiou, and Michael J. Black. VIBE: Video inference for human body pose and shape estimation. In CVPR, 2020. 2
- [26] Muhammed Kocabas, Chun-Hao P. Huang, Joachim Tesch, Lea M¨uller, Otmar Hilliges, and Michael J. Black. SPEC: Seeing people in the wild with an estimated camera. In ICCV, 2021. 3
- [27] Qiayuan Liao, Takara E Truong, Xiaoyu Huang, Guy Tevet, Koushil Sreenath, and C Karen Liu. Beyondmimic: From motion tracking to versatile humanoid control via guided diffusion. arXiv preprint arXiv:2508.08241, 2025. 2, 9
- [28] Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, and Bingyi Kang. Prompting depth anything for 4k resolution accurate metric depth estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17070–17080, 2025. 4, 6
- [29] Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. Character controllers using motion vaes. TOG,

2020. 6

- [30] Zhizheng Liu, Joe Lin, Wayne Wu, and Bolei Zhou. Joint optimization for 4d human-scene reconstruction in the wild. arXiv:2501.02158, 2025. 3

- [31] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. Smpl: a skinned multiperson linear model. TOG, 2015. 6
- [32] Matthew M. Loper, Naureen Mahmood, and Michael J. Black. MoSh: Motion and shape capture from sparse markers. ACM Transactions on Graphics, (Proc. SIGGRAPH Asia), 33(6):220:1–220:13, 2014. 6
- [33] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In ICCV, 2023. 3, 9
- [34] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. arXiv:2310.04582, 2023. 3, 9
- [35] Zhengyi Luo, Jinkun Cao, Sammy Christen, Alexander Winkler, Kris Kitani, and Weipeng Xu. Grasping diverse objects with simulated humanoids. arXiv:2407.11385, 2024. 9
- [36] Lingni Ma, Yuting Ye, Fangzhou Hong, Vladimir Guzov, Yifeng Jiang, Rowan Postyeni, Luis Pesqueira, Alexander Gamino, Vijay Baiyya, Hyo Jin Kim, et al. Nymeria: A massive collection of multimodal egocentric daily motion in the wild. In ECCV, 2024. 2, 3
- [37] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In ICCV, 2019. 2, 8
- [38] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. Isaac gym: High performance gpu-based physics simulation for robot learning. arXiv:2108.10470, 2021. 8
- [39] Lea M¨uller, Hongsuk Choi, Anthony Zhang, Brent Yi, Jitendra Malik, and Angjoo Kanazawa. Reconstructing people, places, and cameras. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21948–21958,

2025. 3

- [40] Riku Murai, Eric Dexheimer, and Andrew J Davison. Mast3r-slam: Real-time dense slam with 3d reconstruction priors. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16695–16705, 2025. 3
- [41] Liang Pan, Jingbo Wang, Buzhen Huang, Junyu Zhang, Haofan Wang, Xu Tang, and Yangang Wang. Synthesizing physically plausible human motions in 3d scenes. In 3DV, 2024. 8
- [42] Liang Pan, Zeshi Yang, Zhiyang Dou, Wenjia Wang, Buzhen Huang, Bo Dai, Taku Komura, and Jingbo Wang. Tokenhsi: Unified synthesis of physical human-scene interactions through task tokenization. In CVPR, 2025. 3, 8
- [43] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In CVPR, 2019. 2
- [44] Xue Bin Peng. Mimickit: A reinforcement learning framework for motion imitation and control, 2025. 9
- [45] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. TOG, 2018. 2, 3

- [46] Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. TOG, 2021. 3, 8
- [47] Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. TOG, 2022. 3, 9
- [48] Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, David J Yoon, Ryan Hoque, Lars Paulsen, et al. Humanoid policy˜ human policy. arXiv preprint arXiv:2503.13441, 2025. 9
- [49] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv:2408.00714,

2024. 4, 6

- [50] Sara Rojas, Matthieu Armando, Bernard Ghanem, Philippe Weinzaepfel, Vincent Leroy, and Gregory Rogez. Hamst3r: Human-aware multi-view stereo 3d reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5027–5037, 2025. 3
- [51] Paul-Edouard Sarlin, Cesar Cadena, Roland Siegwart, and Marcin Dymczyk. From coarse to fine: Robust hierarchical localization at large scale. In CVPR, 2019. 10
- [52] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In CVPR, 2016. 4
- [53] Johannes Lutz Sch¨onberger, True Price, Torsten Sattler, JanMichael Frahm, and Marc Pollefeys. A vote-and-verify strategy for fast spatial verification in image retrieval. In ACCV,

2016. 5

- [54] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 8
- [55] Zehong Shen, Huaijin Pi, Yan Xia, Zhi Cen, Sida Peng, Zechen Hu, Hujun Bao, Ruizhen Hu, and Xiaowei Zhou. World-grounded human motion recovery via gravity-view coordinates. In SIGGRAPH Asia, 2024. 3, 8
- [56] Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J Black. Wham: Reconstructing world-grounded humans with accurate 3d motion. In CVPR, 2024. 3, 7
- [57] Juan Terven, Diana-Margarita C´ordova-Esparza, and JulioAlejandro Romero-Gonz´alez. A comprehensive review of yolo architectures in computer vision: From yolov1 to yolov8 and yolo-nas. Machine learning and knowledge extraction, 2023. 4
- [58] Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. Maskedmimic: Unified physics-based character control through masked motion inpainting. TOG, 2024. 9
- [59] Chen Tessler, Yifeng Jiang, Erwin Coumans, Zhengyi Luo, Gal Chechik, and Xue Bin Peng. Maskedmanipulator: Versatile whole-body control for loco-manipulation. arXiv preprint arXiv:2505.19086, 2025.
- [60] Andrea Tirinzoni, Ahmed Touati, Jesse Farebrother, Mateusz Guzek, Anssi Kanervisto, Yingchen Xu, Alessandro Lazaric, and Matteo Pirotta. Zero-shot whole-body humanoid control via behavioral foundation models. In The Thirteenth International Conference on Learning Representations. 9

- [61] Shashank Tripathi, Agniv Chatterjee, Jean-Claude Passy, Hongwei Yi, Dimitrios Tzionas, and Michael J. Black. DECO: Dense estimation of 3D human-scene contact in the wild. In ICCV, 2023. 3
- [62] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 3
- [63] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 3
- [64] Wenjia Wang, Yongtao Ge, Haiyi Mei, Zhongang Cai, Qingping Sun, Yanjun Wang, Chunhua Shen, Lei Yang, and Taku Komura. Zolly: Zoom focal length correctly for perspectivedistorted human mesh reconstruction. In ICCV, 2023. 3
- [65] Wenjia Wang, Liang Pan, Zhiyang Dou, Jidong Mei, Zhouyingcheng Liao, Yuke Lou, Yifan Wu, Lei Yang, Jingbo Wang, and Taku Komura. Sims: Simulating stylized humanscene interactions with retrieval-augmented script generation. ICCV., 2025. 3, 8
- [66] Yufu Wang, Ziyun Wang, Lingjie Liu, and Kostas Daniilidis. Tram: Global trajectory and motion of 3d humans from inthe-wild videos. In ECCV. Springer, 2024. 3, 4, 7
- [67] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025. 7
- [68] Zihan Wang, Jiashun Wang, Jeff Tan, Yiwen Zhao, Jessica K. Hodgins, Shubham Tulsiani, and Deva Ramanan. Contactguided real2sim from monocular video with planar scene primitives. In ICLR, 2026. 3
- [69] Haoyang Weng, Yitang Li, Nikhil Sobanbabu, Zihan Wang, Zhengyi Luo, Tairan He, Deva Ramanan, and Guanya Shi. Hdmi: Learning interactive humanoid whole-body control from human videos. arXiv:2509.16757, 2025. 3, 8, 9
- [70] Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang. Unified human-scene interaction via prompted chain-of-contacts. In ICLR, 2024. 8
- [71] Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, and LiangYan Gui. Intermimic: Towards universal wholebody control for physics-based human-object interactions. arXiv:2502.20390, 2025. 9
- [72] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. NeurIPS, 2022. 4, 6
- [73] Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Moconvq: Unified physicsbased motion control via scalable discrete representations. ACM Transactions on Graphics (TOG), 43(4):1–21, 2024. 9
- [74] Vickie Ye, Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. Decoupling human and camera motion from videos in the wild. In CVPR, 2023. 3
- [75] Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, and Jan Kautz. Glamr: Global occlusion-aware human mesh recovery with dynamic cameras. In CVPR, 2022. 3

[76] Siwei Zhang, Qianli Ma, Yan Zhang, Zhiyin Qian, Taein Kwon, Marc Pollefeys, Federica Bogo, and Siyu Tang. Egobody: Human body shape and motion of interacting people from head-mounted devices. In ECCV. Springer, 2022. 2, 3

### EmbodMocap: In-the-Wild 4D Human-Scene Reconstruction for Embodied Agents

#### Supplementary Material

##### 9. More Details of EmbodMocap

###### 9.1. Capture technique

The primary capture technique involves two photographers, each holding an iPhone in a vertical orientation. The photographers are required to maintain a certain angle relative to each other while following the performer. To achieve optimal triangulation during post-processing, the angle between the two cameras should ideally fall within the range of 60 to 120 degrees.

This configuration not only enhances the accuracy of triangulation but also ensures the capture of the performer from multiple perspectives, providing diverse viewpoint information for keypoint detection. Additionally, the photographers should aim to keep the cameras in motion to dynamically adjust their positions and minimize occlusion caused by objects in the environment.

[Figure 268]

| | | |
|---|---|---|
| | | |

Figure 8. Capture technique.

###### 9.2. Human Labor Analysis

Temporal Synchronization. This step only needs the operator to identify and input the frame indices where the laser pointer’s spot disappears into a .xlsx file. Typically, this process takes only about 1 minute per sequence.

Skill Segmentation. Skill segmentation is only required when training physical interaction skills. The operator annotates each skill’s category, start, and end times based on the video, typically taking 0.5 to 2 minutes per sequence.

Contact Label & Optimization. In the main text, we mention that the alignment between our sequence and the scene coordinate system relies on photometric (COLMAP, pixel tracking) and geometric constraints (chamfer distance). However, this can sometimes result in alignment errors of a few centimeters, primarily due to depth inaccuracies in COLMAP’s sparse keypoints and depth errors

from the iPhone sensor. To address this issue, we propose an optional post-processing solution. During data capture, we place markers in the scene and instruct the performer to begin walking from a designated marker and stop on another at the end of the sequence, standing still on the same marker. Annotating contact frame indices costs 1-2 minutes for each sequence. These markers serve as fixed reference points for alignment. In post-processing, we observe the corresponding marker positions on the reconstructed mesh and record their 3D coordinates, along with the frame indices where the performer stands on the markers. Using this information, we optimize a rigid transformation to align the center of the performer’s feet at the specified frame indices to the 3D coordinates of the markers.

Since SpectacularAI could generate Z-up metric-scaled camera matrices, we define the rigid transformation in the xy-plane, defined by a rotation angle ϕc about the z-axis and a translation Tc. This can be represented by a homogeneous transformation matrix M:

  

  

cos(ϕc) −sin(ϕc) 0 tx sin(ϕc) cos(ϕc) 0 ty

R(ϕc) Tc 0 1

M =

=

0 0 1 tz 0 0 0 1

(10)

This matrix transform the center of lowest point on both feet to match the annotate marker. To robustly solve for the transformation parameters, we employ a gradient descent optimization, constrained by a minimizing a contact loss to match the contact marker:

2

1 Nc i∈C

(V(i)) − c(zi)

(11)

Lcontact =

min

z

For SMPL parameters, the global orientation is updated as θ′g = Rcθg. For translation, the pelvis’s world position is transformed as Pw′ = RcPw + Tc. Re-evaluating the SMPL model with θ′g gives the local pelvis offset Pl′, and the updated translation is γ′ = Pw′ − Pl′.

The updated camera rotation and translation are com-

puted as Rv′ = RvRcT and Tv′ = Tv − RvRcTTc, ensuring alignment and consistency of the scene representation.

##### 10. More Details of Monocular Human-Scene Reconstruction Pipeline

Our monocular reconstruction baseline is a modular pipeline for reconstructing 3D human pose and scene geometry from monocular video, combining two independent

[Figure 269]

[Figure 270]

The point clouds and SMPL global orientation and translation are transformed to the world coordinate system with R,t following the same formula as Sec. 9.2.

##### 11. More Details of Human-Object Interaction Skills

[Figure 271]

###### 11.1. Follow Skill

Definition. The path following task requires the simulated character to move along a predefined 2D trajectory. A trajectory is represented as τ = {xτ0.1,xτ0.2,...,xτT−0.1,xτT}, where xτ0.1 denotes a 2D waypoint at simulation time 0.1s, and T is the episode length. For this task, T is set to 10s. The character is expected to follow the trajectory τ as accurately as possible.

Figure 9. An example in finding the contact marker in software (e.g., Meshlab) and corresponding keyframe index(the frames selected here are just for demo).

Task Observation. At each simulation time step t, the character observes 10 future waypoints sampled over the next 1.0s: {xτt ,xτt+0.1,...,xτt+0.8,xτt+0.9}. These waypoints are sampled at intervals of 0.1s using linear interpolation from the trajectory τ. The 2D coordinates of these waypoints form the task observation gtf ∈ R2×10.

modules: π3 for camera trajectory prediction and scene point cloud reconstruction, and VIMO for SMPL-based human pose estimation. To process long video sequences, π3 divides frames into overlapping chunks, where each chunk independently predicts camera poses Tv ∈ RT×4×4 and local point clouds Plocal ∈ RT×H×W×3. To align these chunks into a global coordinate system, Procrustes analysis is applied to the overlapping regions of adjacent chunks. Given two point clouds X,Y ∈ RN×3, the alignment minimizes the error:

Task Reward. The reward for this task, rtf, is computed based on the distance between the character’s current 2D

root position, xtroot 2d, and the target waypoint, xτt . The reward is defined as:

rtf = exp − 2.0∥xroott 2d − xτt ∥2 . (16)

∥Y − (sRX + t)∥2F, (12)

###### 11.2. Sit Skill

min

s,R,t

Definition. The sitting task requires the character to position its root joint at a target 3D sitting location on an object surface. The target position is defined as 10 cm above the center of the top surface of the chair seat.

where s is the scale, R is the rotation matrix, and t is the translation vector. Using SVD, the optimal alignment parameters are computed as:

Task Observation. The observation gts ∈ R38 includes the 3D target sitting position ∈ R3, the 3D root position ∈ R3, the root rotation ∈ R6, the 2D front-facing direction ∈ R2, and the positions of eight corner points of the object’s bounding box ∈ R3×8.

trace(Yc⊤RXc) trace(Xc⊤Xc)

, t = Y¯ − sRX¯, (13)

R = V SU⊤, s =

where Xc,Yc are the centered point clouds, and V ,U are derived from the SVD of the covariance matrix H = Xc⊤Yc. After chunk alignment, VIMO predicts SMPL parameters (θ,γ,β), where θ ∈ RT×72 represents joint rotations, γ ∈ RT×3 is the root translation, and β ∈ R10 defines body shape. Using a weak perspective camera model, SMPL vertices are projected onto the image plane as:

Task Reward. The sitting task reward rts encourages the character to minimize the distance between its 3D root po-

sition, xroott , and the target sitting position, xtart . It is defined as:

0.7rtnear + 0.3rtfar, ∥xtobj 2d − xroott 2d∥ > 0.5, 0.7rtnear + 0.3, otherwise,

rts =

ximg = sxv + t (14)

(17) where rtfar and rtnear are defined as:

where s is the scaling factor proportional to 1/z. To resolve scale ambiguity, the pipeline estimates a metric scale by matching the predicted depths of SMPL vertices zSMPL (in meters) with the depths of Pi3’s point cloud zPi3 (in arbitrary units) on some sampled points. The scale factor is computed as:

rtfar = exp − 2.0∥1.5 − d∗t · x˙troot 2d∥2 , (18) rtnear = exp − 10.0∥xtart − xroott ∥2 . (19)

Here, xobjt 2d is the 2D position of the object’s root, x˙roott 2d is the 2D linear velocity of the character’s root, and d∗t is a horizontal unit vector pointing from xroott 2d to xtobj 2d.

zπ3 zSMPL

, (15)

s = median

###### 11.3. Climb Skill

Definition. The climbing task requires the character to place its root joint at a target 3D climbing position on a given object. The target position is set 94 cm above the center of the top surface of the object.

Task Observation. The observation gtm ∈ R27 includes the 3D target root position ∈ R3 and the 3D coordinates of eight corner points of the object’s bounding box ∈ R3×8.

Task Reward. The climbing task reward rtm minimizes the 3D distance between the character’s root, xroott , and the target location, xtart . The reward is defined as:

0.5rtnear + 0.2rtfar, ∥xobjt 2d − xtroot 2d∥ > 0.7, 0.5rtnear + 0.2 + 0.3rtfoot, otherwise,

rtm =

(20) where rtnear, rtfar, and rtfoot are defined as:

rtnear = exp − 10.0∥xtart − xroott ∥2 , (21) rtfar = exp − 2.0∥1.5 − d∗t · x˙troot 2d∥2 , (22)

rtfoot = exp − 50.0∥(xtart h − 0.94) − xtfoot h∥2 . (23)

Here, xtart h is the height of the target root position, (xtart h − 0.94) represents the height of the top surface of the target

object in world coordinates, and xfoott h is the mean height of the character’s feet. The reward rtfoot encourages the character to lift its feet and is crucial for successful climbing.

###### 11.4. Lie Skill

Definition. The lying task requires the character to position its root joint at a target 3D lying position on an object, typically centered on the object’s surface. The character must first approach a designated standing point before transitioning into the lying position.

Task Observation. The observation gtl ∈ R38 includes the 3D target lying position ∈ R3, the 3D root position ∈ R3, the root rotation ∈ R6, the 2D front-facing direction ∈ R2, and the positions of eight corner points of the object’s bounding box ∈ R3×8. It also includes the chosen standing point ∈ R3.

Task Reward. The lying reward rtl combines rewards for approaching the standing point and accurately lying down:

rtl =

0.6rtnear + 0.4rtfar, ∥xroott − xtart ∥ > 1.5, rtnear, otherwise.

(24)

The far reward encourages approaching the standing point:

rtfar = 0.5rtwalk + 0.2rtvel + 0.2rtfacing + 0.1rtstand, (25)

where rtwalk rewards walking toward the standing point, rtvel aligns velocity, rtfacing ensures proper facing direction, and rtstand rewards correct height.

The near reward focuses on lying accuracy:

rtnear = 0.5rtpos + 0.3rthead + 0.2rtalignment, (26)

where rtpos minimizes the distance to the target, rthead aligns head height, and rtalignment rewards proper body alignment.

###### 11.5. Prone Skill

Definition. The prone task requires the character to position its root joint at a designated 3D prone position on an object, typically centered on the object’s surface. Unlike the lying task, the character must face downward while maintaining alignment with the target surface.

Task Observation. The observation gtp ∈ R35 includes the 3D target prone position ∈ R3, the 3D root position ∈ R3, the root rotation ∈ R6, the 2D front-facing direction ∈ R2, and the positions of eight corner points of the object’s bounding box ∈ R3×8. These observations help guide the approach and ensure the correct orientation for prone positioning.

Task Reward. The prone reward rtp encourages the character to transition smoothly from moving to a prone position while maintaining proper alignment and facing downward. The reward is defined as:

rtp =

0.7rtnear + 0.3rtfar, ∥xroott − xtart ∥ > 1.5, rtnear, otherwise.

(27)

The far reward encourages approaching the target prone position:

rtfar = 0.5rtwalk + 0.2rtvel + 0.2rtfacing + 0.1rtheight, (28)

where rtwalk rewards moving toward the prone position, rtvel aligns velocity with the direction of motion, rtfacing ensures proper facing direction, and rtheight encourages maintaining an appropriate height during approach.

The near reward focuses on prone accuracy:

rtnear = 0.6rtpos + 0.2rtalignment + 0.2rtface down, (29)

where rtpos minimizes the distance to the prone target, rtalignment ensures proper body alignment with the surface, and rtface down rewards the character for maintaining a facedown orientation.

###### 11.6. Support Skill

Definition. The support task encourages the character to approach a target object and maintain stable interaction by placing its hands on the top surface while keeping stable foot placement and proper posture.

Task Observation. The task observation gtm ∈ R27 consists of the 3D target position of the object’s top surface

center (xot,zto ∈ R3) and the 3D coordinates of the eight corner points of the object’s bounding box (bt ∈ R3×8).

##### 12. More Details of Scene-Aware Imitation Policy

Task Reward. The total reward rtm is defined as:

0.4rtf + 0.6rts, ∥xot − xrt∥ > 1.5, rts, otherwise,

rtm =

(30)

###### 12.1. Representations

Character Proprioception. The state s describes the proprioception of the character’s body, with features consisting of the relative positions of each link with respect to the root (designated to be the pelvis), their rotations expressed in quaternions, and their linear and angular velocities. All features are computed in the character’s local coordinate frame, with the root at the origin and the x-axis along the root link’s facing direction.

rtf = 0.5exp − 0.5∥xot − xrt∥2 (31)

+ 0.5exp − 2.0∥1.5 − d∗t · x˙rt∥2 , (32) rts = 0.3rth + 0.2rtg + 0.15rtt + 0.2rto + 0.15rtz, (33)

where rtf encourages the character to approach the object, and rts combines five components for stable interaction:

rth = 0.6exp − 20∥zth − zto∥2 (34)

Height Map. To perceive the surrounding scene geometry, we utilize a local egocentric height map. This map is structured as an 11×11 grid spanning a 2m×2m area centered on the humanoid, resulting in a sampling interval of 0.2m. The grid is defined within the character’s local coordinate frame; consequently, the sampling points dynamically translate and rotate with the humanoid’s movement and heading, consistently covering the immediate vicinity. The height values at these grid points are queried from a high-resolution underlying scene mesh (0.05m resolution) using nearest-neighbor interpolation.

+ 0.4exp − 5∥xht2 − xot∥2 , (35) rtg = exp − 50∥ztf − zg∥2 , (36) rtt = exp − 10∥xfrt − xflt ∥2 , (37) rto = exp − 2∥1.0 − (−ubt)∥2 , (38) rtz = exp − 10∥ztr − zto∥2 . (39)

Here, xot and xrt denote the 2D positions of the object and the character’s root, while zto and ztr are their respective heights. xht2 and zth represent the 2D position and height of the hands. Similarly, xfrt , xflt , and ztf refer to the 2D positions and height of the feet, zg is the ground height, and −ubt is the vertical component of the body’s up direction.

Target States. The target state qˆ encodes the desired future motion of the character. It is constructed by sampling a short trajectory segment from the dataset spanning three consecutive future time steps: T,T +1, and T +2. For each time step, the state comprises the positions, rotations, linear velocities, and angular velocities of all body links. All features are transformed from the world frame into the simulated character’s local coordinate frame. This local frame is defined with the character’s root located at the origin and the x-axis aligned with the root link’s facing direction.

[Figure 272]

[Figure 273]

Action. Our simulated humanoid is constructed based on the SMPL body model, comprising 23 controllable joints. Each joint possesses 3 degrees of freedom (DoF), and we employ a Proportional-Derivative (PD) controller for each DoF. Consequently, the action a ∈ R69 generated by the policy specifies the target orientations for these PD controllers.

(a) Camera Trajectory Length Distribution.

(b) Human Trajectory Length Distribution.

[Figure 274]

[Figure 275]

###### 12.2. Reward

To encourage the character to closely reproduce the reference motion while maintaining motion naturalness, our reward function rt is composed of two terms: a tracking reward rttrack and a jitter penalty rtsmooth. The tracking reward incentivizes the policy to minimize the kinematic error between the simulated character and the reference motion. The jitter penalty is introduced to suppress abnormal shaking generated when the character interacts with objects, which may be induced by instabilities in the physics simulation. The total reward is defined as:

(c) Scene Mesh Area Distribution. (d) Sequence Length Distribution.

Figure 10. Statistical information of collected dataset.

Evaluation The evaluation of the Support task focuses on the agent’s ability to position its hands on the top surface of the target object and keep its feet close together. The key metric is the combined XY-plane distance and Z-axis deviation between the hands and the object’s top surface. The task is deemed successful if the hands are within predefined thresholds and the feet maintain adequate proximity for stability.

rt = rttrack − rtsmooth. (40)

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

- view1
- view2

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

- view1
- view2

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

[Figure 309]

[Figure 310]

[Figure 311]

- view1
- view2

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

Figure 11. Rendered SMPL and depth images of the captured dataset in camera space.

The tracking reward rttrack is computed as the weighted sum of exponential differences across all humanoid links:

rttrack = wjp exp −100∥pˆt − pt∥2

+ wjr exp −10∥qˆt ⊖ qt∥2

(41)

+ wjv exp −0.1∥vˆt − vt∥2

+ wjω exp −0.1∥ωˆt − ωt∥2 ,

where the equation penalizes the differences in translation p, rotation q, linear velocity v, and angular velocity ω for all rigid body links of the humanoid between the simulation and the reference. The jitter penalty penalizes the magnitude of the difference between consecutive actions, defined as:

rtsmooth = ∥at − at−1∥2, (42)

where at and at−1 denote the action at the current and previous time steps, respectively. By minimizing the rate of

change of the actions, the policy is incentivized to generate continuous and stable control trajectories, thereby reducing jittery behaviors.

##### 13. More Details of Captured Dataset Used in Main Paper

We collected data from 23 scenes, each with a highprecision mesh, 104 sequences, and approximately 200,000 video frames. Each frame is accompanied by corresponding depth maps, segmentation masks, camera trajectories, and human parameters(bounding boxes, 2D keypoints, SMPL parameters).

In Fig. 10a, we present the distribution of camera trajectory lengths, which range from 4 meters to over 30 meters. In Fig. 10b, the human trajectory length distribution is shown, with performers moving between 5 meters and over 30 meters. Figure 10c illustrates the scene mesh area dis-

tribution. Indoor scenes are relatively smaller, ranging from 20 to 90 square meters, while outdoor scenes can be as large as 200 square meters. Finally, in Fig. 10d, we show the sequence length distribution, where most sequences have durations ranging from 30 to 60 seconds.

###### 13.1. Qualitative Demonstrations

We show camera space results in Sec. 11.6 and world space results in Sec. 13.1

|[Figure 317]<br><br>[Figure 318]|
|---|

[Figure 319]

|[Figure 320]<br><br>[Figure 321]|
|---|

[Figure 322]

[Figure 323]

|[Figure 324]<br><br>[Figure 325]|
|---|

|[Figure 326]<br><br>[Figure 327]|
|---|

[Figure 328]

|[Figure 329]<br><br>[Figure 330]|
|---|

|[Figure 331]<br><br>[Figure 332]|
|---|

[Figure 333]

|[Figure 334]<br><br>[Figure 335]|
|---|

[Figure 336]

|[Figure 337]|
|---|

[Figure 338]

[Figure 339]

|[Figure 340]|
|---|

|[Figure 341]<br><br>[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]<br><br>[Figure 345]|
|---|

|[Figure 346]<br><br>[Figure 347]|
|---|

|[Figure 348]<br><br>[Figure 349]|
|---|

|[Figure 350]|
|---|

[Figure 351]

[Figure 352]

(1) Living room1 (2) Living room2 (3) Living room3 (4) Bedroom1

|[Figure 353]<br><br>[Figure 354]|
|---|

[Figure 355]

[Figure 356]

[Figure 357]

|[Figure 358]<br><br>[Figure 359]<br><br>[Figure 360]|
|---|

|[Figure 361]|
|---|

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

|[Figure 366]<br><br>[Figure 367]<br><br>[Figure 368]|
|---|

|[Figure 369]|
|---|

|[Figure 370]|
|---|

|[Figure 371]|
|---|

|[Figure 372]<br><br>[Figure 373]<br><br>[Figure 374]|
|---|

(5) Garden1 (6) Stairs1 (7) Stairs2

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

|[Figure 380]|
|---|

|[Figure 381]|
|---|

|[Figure 382]|
|---|

[Figure 383]

|[Figure 384]|
|---|

|[Figure 385]|
|---|

|[Figure 386]|
|---|

|[Figure 387]|
|---|

(8) Wall1

(9) Garden2

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

|[Figure 392]|
|---|

|[Figure 393]|
|---|

|[Figure 394]|
|---|

|[Figure 395]|
|---|

|[Figure 396]<br><br>[Figure 397]|
|---|

[Figure 398]

|[Figure 399]<br><br>[Figure 400]|
|---|

|[Figure 401]<br><br>[Figure 402]|
|---|

|[Figure 403]<br><br>[Figure 404]|
|---|

(10) Stairs3 (11) Garden3

[Figure 405]

|[Figure 406]|
|---|

|[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

|[Figure 410]|
|---|

(12) Wall2

###### Figure 12. 3D demo of the captured dataset.

