# arXiv:2510.13802v1[cs.CV]15Oct2025

[Figure 1]

### Trace Anything: Representing Any Video in 4D via Trajectory Fields

Xinhang Liu1,2, Yuxi Xiao1,3, Donny Y. Chen1, Jiashi Feng1, Yu-Wing Tai4, Chi-Keung Tang2, Bingyi Kang1

1ByteDance Seed, 2HKUST, 3Zhejiang University, 4Dartmouth College

#### Abstract

Effective spatio-temporal representation is fundamental to modeling, understanding, and predicting dynamics in videos. The atomic unit of a video, the pixel, traces a continuous 3D trajectory over time, serving as the primitive element of dynamics. Based on this principle, we propose representing any video1 as a Trajectory Field: a dense mapping that assigns a continuous 3D trajectory function of time to each pixel in every frame. With this representation, we introduce Trace Anything, a neural network that predicts the entire trajectory field in a single feed-forward pass. Specifically, for each pixel in each frame, our model predicts a set of control points that parameterizes a trajectory (i.e., a B-spline), yielding its 3D position at arbitrary query time instants. We trained the Trace Anything model on large-scale 4D data, including data from our new platform, and our experiments demonstrate that: (i) Trace Anything achieves state-of-the-art performance on our new benchmark for trajectory field estimation and performs competitively on established point-tracking benchmarks; (ii) it offers significant efficiency gains thanks to its one-pass paradigm, without requiring iterative optimization or auxiliary estimators; and (iii) it exhibits emergent abilities, including goal-conditioned manipulation, motion forecasting, and spatio-temporal fusion. We will release the code, the model weights and the data platform.

Correspondence: Bingyi Kang Project Page: trace-anything.github.io

#### 1 Introduction

Understanding dynamic scenes requires more than disjoint reconstruction of 3D space at each time step; it demands modeling how the scene evolves in both space and time. A central challenge toward spatial intelligence is to develop a 4D video representation that captures the underlying spacetime dynamics in a way that is geometrically grounded and scalable. Rather than relying on additional estimators such as depth, flow, or tracking, or on heavy per-scene optimization, we observe that the atomic elements of video, its pixels, naturally trace out 3D trajectories in the world, which acts as the atomic primitive of dynamics.

Recognizing this, we propose Trajectory Fields, a versatile 4D representation for any video that associates each pixel in each frame with a parametric 3D trajectory, as illustrated in Figure 2. Unlike prior 4D reconstruction methods that produce disjoint per-frame point clouds [32, 50, 69] and rely on estimated optical flow or 2D

1Here, “any video” extends beyond monocular videos to include image pairs or even unordered unstructured image collections that capture dynamic scenes.

[Figure 2]

[Figure 3]

[Figure 4]

##### Trace Anything

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

###### Input Frames Trajectory Field

- Figure 1 Any video∗ can be represented in 4D with a Trajectory Field, a dense mapping assigning each pixel in each frame to a parametric 3D trajectory. We propose Trace Anything, a neural network that predicts the trajectory field with a single forward pass.

tracks to build cross-frame correspondences, Trajectory Fields offer a more direct and compact way to model scene dynamics.

Building on this representation, we propose Trace Anything, a feed-forward neural network that estimates trajectory fields directly from video frames. As shown in Figure 1, with a single forward pass over all input frames, it predicts a stack of control point maps for each frame, defining spline-based parametric trajectories for every pixel. This design brings three key advantages: (i) its one-pass scheme eliminates intermediate estimators and iterative global alignment, (ii) it predicts all trajectories (per pixel per frame) jointly in a shared world coordinate system, and (iii) it generalizes across diverse inputs, including monocular videos, image pairs, and unordered photo sets.

[Figure 10]

|[Figure 11]<br><br>[Figure 12]<br><br>|
|---|

……

|[Figure 13]<br><br>[Figure 14]<br><br>|
|---|

|[Figure 15]<br><br>[Figure 16]<br><br>|
|---|

Input Frames

Per-Pixel 3D Parametric Trajectories

Figure 2 Given the input frames (left), a trajectory field represents the video at the atomic level, mapping each pixel in each frame to a 3D trajectory, expressed as a parametric curve (right).

To support training and evaluation at scale, we develop a Blender-based platform featuring diverse environments, moving characters, and camera trajectories. It produces photo-realistic renderings with dense annotations, including 2D/3D trajectories, depth, semantics, flow, and camera poses. From this platform, we release (i) the Trace Anything dataset—10,000+ videos (120 frames each) for training trajectory field estimation models, and (ii) the Trace Anything benchmark—200 curated videos for evaluating models’ ability to capture motion jointly across all frames.

Trained with our new dataset, Trace Anything achieves state-of-the-art results on our trajectory field benchmark and performs competitively on established point tracking benchmarks, while offering significant efficiency

gains. Moreover, our paradigm also reveals new capabilities for spatial reasoning, including motion forecasting, spatio-temporal fusion, and goal-conditioned manipulation.

In summary, our contributions are:

- • We propose Trajectory Fields as an atomic-level and versatile 4D video representation, grounded in a principled formulation.
- • We present Trace Anything, a feed-forward network that predicts trajectory fields without requiring extra estimators or per-scene optimization.
- • We develop a synthetic data platform for large-scale training and benchmarking of trajectory field estimation.
- • Extensive experiments on existing and new benchmarks demonstrate competitive accuracy, faster inference, and new capabilities.

#### 2 Related Work

(Dynamic) 3D scene reconstruction. Reconstructing 3D structure from multi-view images is a long-standing problem in computer vision. Classical Structure-from-Motion (SfM) pipelines [1, 21, 49] proceed in sequential stages: feature extraction, image matching, triangulation, relative pose estimation, and global bundle adjustment. Deep learning has improved individual components [10, 48] yet stage-wise pipelines remain prone to error accumulation. DUSt3R [58] addressed this by directly predicting 3D pointmaps from image pairs. Fast3R [65], VGGT [55], π3 [59] and MapAnything [24] further relaxed the pairwise assumption with all-to-all attention, enabling joint reasoning over all frames and avoiding O(N2) pairwise inference. However, both traditional and learning-based methods generally assume static scenes and sufficient camera baselines, leading to degraded performance in dynamic settings. To handle monocular videos with dynamics, MegaSAM [32] integrates optimization-based SLAM, while Monst3R [69], POMATO [70], Easi3R [5], St4RTrack [15], and Dynamic Point Maps [50] extend DUSt3R-style networks to dynamic scenes. These methods typically generate disjoint per-frame point clouds, relying on optical flow or 2D tracks for cross-frame correspondences, and their pairwise inference often requires costly per-scene optimization for global alignment. In contrast, Trace Anything directly estimates trajectory fields that produce dynamic point clouds with cross-frame correspondences, sharing the feed-forward spirit of Yang et al. [65] and [55] and performing one-pass inference over all frames.

Point tracking. Particle Video [47] first introduced long-range particle trajectories in videos. Early deep learning methods [11, 12, 20] approached this with global matching and local refinement. CoTracker [23] leveraged a transformer-based architecture to enable tracking through occlusions, followed by works [6, 28] that improved efficiency with 4D correlation volumes. CoTracker3 [22] further leveraged unlabeled data to boost performance. 3D point tracking remains comparatively new. OmniMotion [57] addressed the task via test-time optimization, while SpatialTracker [63] introduced the first feed-forward 3D tracker by combining

- 2D tracking with monocular depth priors. DELTA [38] achieved dense 3D tracking using a transformer with upsampling for high-resolution outputs. Concurrently, SpatialTrackerV2 [64] scaled training across real and synthetic data, and St4RTrack [15] and POMATO [70] extended 3D reconstruction models for tracking via joint optimization. Unlike prior approaches, our method bypasses monocular depth estimation and 2D trackers and directly predicts dense 3D trajectories in a feed-forward manner.

4D representations for NVS. A large class of 4D representations has been developed for novel view synthesis (NVS) in dynamic scenes, aiming to deliver immersive effects such as “bullet time.” Since Neural Radiance Fields (NeRF) [37] introduced implicit volumetric representations, many extensions have incorporated temporal dynamics. One class of methods [17, 30, 31, 62] directly conditions the radiance field on time, treating density and color as continuous functions of space and time. Another class [39, 40, 43, 52, 68] maps observations to a canonical space and models dynamics via deformation fields. Grid-based approaches [2, 4, 16, 33, 56] discretize the 4D volume into compact planar factors for efficiency. Also in this line of work, Wang et al. [54] proposed ‘neural trajectory fields’, with a different formulation and purpose than ‘trajectory fields’ in this study. More recently, 3D Gaussian Splatting (3DGS) [25] has been extended to dynamics [29, 35, 61, 66, 67],

[Figure 17]

###### Input Frames Control Point Maps

###### Per-Pixel Trajectories Aggregated 4D Representation

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Curve Evaluation

[Figure 22]

Image Encoder

CP Head

…

[Figure 23]

ℝ × × 

ℝ × × × 

Shared Weights

Shared Weights

[Figure 24]

###### ……

…

[Figure 25]

[Figure 26]

[Figure 27]

Curve Evaluation

Fusion Transformer

Image Encoder

CP Head

…

[Figure 28]

[Figure 29]

ℝ × × 

[Figure 30]

ℝ × × × 

Shared Weights

Shared Weights

…

[Figure 31]

[Figure 32]

[Figure 33]

Curve Evaluation

Image Encoder

CP Head

…

[Figure 34]

[Figure 35]

ℝ × × 

ℝ × × × 

Figure 3 Trace Anything pipeline. Input frames are processed by a geometric backbone consisting of an image encoder and a fusion transformer. The control point head outputs dense control point maps Pi ∈ RD×H×W×3, where P(i,u,vk) is the k-th control point for pixel (u, v) in frame Ii. These define continuous 3D trajectories xi,u,v(t) via cubic B-splines, yielding a 4D reconstruction.

improving rendering quality and speed. These efforts focus on photorealistic appearance and typically assume precomputed camera poses or point clouds. Our work is orthogonal: we propose a geometry-centric paradigm that directly infers trajectory fields from raw videos, emphasizing accurate 3D motion modeling. Integrating NVS with our paradigm, e.g., using trajectory fields to initialize dynamic 3DGS models, is a promising future direction.

#### 3 Method

The atomic elements of video, its pixels, naturally trace out 3D trajectories in the world, forming the primitive units of dynamics. Recognizing this, we aim to model dynamic scenes through trajectory fields, a 4D representation that encodes each pixel in each frame as a continuous 3D trajectory over time. In the following, we first formalize trajectory fields in Section 3.1, then present Trace Anything in Section 3.2, a feed-forward network designed to estimate them, and finally describe the overall training scheme in Section 3.3. In this section, we define a field as a mapping from a domain M to a codomain V , F : M → V , where M may be a discrete or continuous space, and V may represent scalars, vectors, or functions. We provide preliminaries on fields in Section A and on parametric curves in Section B.

###### 3.1 Problem Formulation

We formalize trajectory fields, a 4D representation of dynamic 3D scenes in a video. Let {Ii}Ni=1 be a collection of N RGB frames, where each Ii ∈ R3×H×W captures the scene at different time steps or viewpoints. A trajectory field is defined as

T : [N] × [H] × [W] → C([0,1],R3), (i,u,v)  → xi,u,v(·) (1)

where [N], [H], and [W] denote the discrete sets of frame indices and pixel coordinates, respectively, and xi,u,v : [0,1] → R3 is a continuous 3D trajectory for pixel (u,v) in frame Ii. The domain is M = [N]×[H]×[W], and the codomain is V = C([0,1],R3), the space of continuous functions from [0,1] to R3. Each trajectory xi,u,v(t) is parameterized as a spline-based curve with D control points, defined as

Pi ∈ RD×H×W×3, (2)

where P(i,u,vk) ∈ R3 is the k-th control point for pixel (u,v) in frame Ii, with k ∈ {0,1,...,D − 1}. Given basis functions {ϕk(t)}Dk=0−1, the trajectory is

D−1

P(i,u,vk) ϕk(t). (3)

xi,u,v(t) =

k=0

The form of the basis functions {ϕk(t)}Dk=0−1 depends on the type of parametric curve. In our implementation, we use cubic B-splines with clamped knots as detailed in Section B.

- Figure 2 illustrates this formulation of trajectory fields. For any pixel from any frame, its 3D coordinate at any time t ∈ [0,1] can be obtained with Equation (3). This fundamentally differs from existing 4D reconstruction methods that predict per-frame disjoint point clouds and establish cross-frame correspondences via estimated optical flow or 2D tracks. Ideally, two conditions should hold: (C1) pixels in static regions collapse to degenerate trajectories; (C2) corresponding pixels from different frames map to the same 3D trajectory.

###### 3.2 Network Architecture

Building on the formulation in Section 3.1, we propose a feed-forward network, Trace Anything (Figure 3), which predicts trajectory fields directly from video or unstructured image sets. For each frame, it outputs control point maps defining parametric curves over time, enabling trajectory field estimation in a single pass. This design eliminates reliance on depth or optical flow and avoids per-scene iterative optimization, providing a compact, efficient approach to modeling 4D scenes.

Geometric backbone. We build Trace Anything upon a feed-forward geometric backbone, similar in spirit to recent models [55, 65]. Each frame is first tokenized by an image encoder, followed by a fusion transformer that integrates spatio-temporal context across views through interleaved frame-wise and global attention layers. For sequential video input, we additionally inject temporal index embeddings, while the architecture remains compatible with unordered or unstructured image collections.

Control Point Head. Built on the backbone features, the control point head outputs dense control point maps Pi ∈ RD×H×W×3 for each input frame Ii. Each pixel (u,v) has D control points {P(i,u,vk) }Dk=0−1, compactly parameterizing its 3D trajectory. Predictions are in a shared world coordinate system, with an optional local CP head estimating control points in each frame’s local camera system. The head also predicts per-control-point confidence scores Σ(i,u,vk) for confidence-weighted training and filtering uncertain estimates at inference.

Curve evaluation. Given the predicted control points and basis functions {ϕk(t)}Dk=0−1, continuous trajectories xi,u,v(t) are obtained via Equation (3). At evaluation time, the trajectory can be queried at any t ∈ [0,1]. In particular,

D−1

D−1

P(i,u,vk) · ϕk(0) =∗ P(0)i,u,v, xi,u,v(1) =

P(i,u,vk) · ϕk(1) =∗ P(i,u,vD−1), (4)

xi,u,v(0) =

k=0

k=0

where (∗) holds for cubic B-splines with clamped knots or for Bézier bases. To obtain the 3D coordinates of a pixel from frame i evaluated at the acquisition time of another frame j, we substitute the corresponding temporal parameter tj into its trajectory:

Xi→j(u,v) = xi,u,v(tj). (5)

In most cases, tj is obtained from metadata or frame order. Otherwise, an auxiliary timestamp head predicts normalized timestamps tˆj ∈ [0,1]. As a special case, evaluating each trajectory at frame i’s own acquisition time ti recovers the 3D point map for frame Ii:

Xi(u,v) = xi,u,v(ti). (6)

Trace Anything outputs the trajectory field with a single network inference for all frames, avoiding pairwise inference and subsequent global alignment, while being self-contained and independent of external estimators for monocular depth, optical flow, or 2D tracks.

###### 3.3 Training Scheme

To train Trace Anything, we directly supervise the accuracy of predicted trajectories. Intuitively, a trajectory predicted from frame i should, when evaluated at the timestamp of another frame j, land exactly at its ground-truth 3D location at frame j’s acquisition time.

Trajectory loss. For a pixel (u,v) in frame i, the predicted 3D position evaluated at tj is Xi→j(u,v) (Equation (5)), while the corresponding ground truth is Xgti→j(u,v). We define the loss as

ℓi→j(u,v) = Xi→j(u,v) − Xgti→j(u,v) 22. (7) Confidence adjustment. To account for the varying reliability of predicted trajectories across pixels and control points, we incorporate confidence adjustment. For each control point, the network predicts a scalar confidence

Σˆ(i,u,vk) > 0 alongside its 3D coordinates. The confidence associated with Xi→j(u,v) is then aggregated using the same basis functions as in Equation (3):

D−1

Σˆ(i,u,vk) · ϕk(tj). (8) The final confidence-adjusted loss then becomes

Σˆi→j(u,v) =

k=0

1 |Ω|

Σ ˆi→j(u,v)ℓi→j(u,v) + α log Σˆi→j(u,v) , (9)

Ltraj-conf =

(i,j) (u,v)∈Ω

where Ω denotes valid pixels with ground-truth supervision. This adjustment downweights uncertain predictions while discouraging overconfident ones.

Timestamp supervision. When ground-truth timestamps are available, we directly supervise Timestamp Head with an L1 regression loss:

N

1 N

t ˆi − ti . (10)

Ltime =

i=1

Static regularization. To encourage condition (C1), pixels in static regions should map to overlapped 3D control points. We enforce this by minimizing the variance of their control points:

1 |Ωstatic|

Var {P(i,u,vk) }Dk=0−1 . (11)

Lstatic =

(i,u,v)∈Ωstatic

Rigidity regularization. For pixels segmented as belonging to the same rigid region, their trajectories should preserve internal distances across control points. Equivalently, the pairwise distance between any two pixels p,q within a rigid segment should remain constant across control points. We enforce this by minimizing the variance of their distances:

1 |Ωrigid|

Var ∥P(pk) − P(qk)∥2 Dk=0−1 . (12)

Lrigid =

(p,q)∈Ωrigid

Correspondence regularization. To encourage condition (C2), pixels with known cross-frame correspondences should share identical control points. Let Ωcorr be the set of matched pixels ((i,u,v),(j,u′,v′)). We penalize discrepancies between their control-point sequences:

1 |Ωcorr|

Lcorr =

((i,u,v), (j,u′,v′))∈Ωcorr

D−1

1 D

k=0

P(i,u,vk) − P(j,uk)′,v′

2

2. (13)

Final objective. The overall loss combines the core trajectory supervision with the above regularization terms: L = Ltraj-conf + λtimeLtime + λstaticLstatic + λrigidLrigid + λcorrLcorr. (14)

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Figure 4 Sample renderings from our data platform.

#### 4 Trace Anything Data Platform

Data-driven modeling of dynamic scenes is limited by the lack of large-scale datasets with dense, reliable annotations. Existing synthetic datasets and generators [13, 19, 20, 36, 71] are typically small and biased toward rigid motion, with sparse or short-term annotations, which are insufficient for realistic scene understanding and diverse dynamics. To address this, we develop a scalable 4D Scene Data Platform in Blender that synthesizes photo-realistic dynamic scenes with dense ground-truth annotations.

Trace Anything dataset. Using our platform, we build a dataset whose primary purpose is to train the Trace Anything model on trajectory field estimation. The current release contains about 10K unique scenes, each with 120 annotated frames. The collection spans a wide range of settings and motions, with examples shown in Figure 4. The dataset exhibits diversity across multiple aspects: (i) Environment — diverse indoor and outdoor backgrounds from public asset libraries and procedural generation [44, 45]; (ii) Dynamics articulated human characters and movable objects with both rigid and non-rigid motion; (iii) Camera motion

— smooth trajectories sampled around active regions to mimic natural filming. Rendered RGB videos are paired with per-pixel 2D/3D trajectories, depth maps, camera poses, semantic masks, which facilitate the training scheme introduced in Section 3.3. Since our platform is fully programmable, it can be easily extended with new assets, domains, or annotation modalities to support future research.

Trace Anything benchmark. To evaluate the task of trajectory field estimation, we construct a benchmark consisting of 200 videos, each with 120 frames. A key difference from established point tracking datasets [26] lies in the evaluation protocol: point tracking benchmarks evaluate estimated trajectories only for pixels sampled from the first frame (first-to-all), whereas our benchmark evaluates trajectories for pixels sampled from all frames (all-to-all). This requires models not only to follow motion from a single starting frame, but also to jointly capture dynamics across the entire sequence. In addition, our benchmark provides denser trajectory annotations, covering more pixels per framel, and evaluates in world coordinates, requiring models to reason about both global geometry and motion.

#### 5 Experiments

In this section, we evaluate our method across a series of challenging settings, demonstrating its competitive accuracy, faster inference, and novel capabilities. Please refer to the appendix for additional experimental results, and to our project page for videos and interactive demos.

###### 5.1 Experimental Details

We generate training data using Kubric [19] and our proposed 4D scene data engine. Specifically, we render 20K videos with 24 frames each using Kubric, with half containing continuous camera motion and the other half discrete camera motion, and over 10K videos with 120 frames each from our engine. While Kubric equips models with preliminary ability to capture rigid object motion, it is largely limited to rigid dynamics and

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Input Frames …

… … …

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

Predicted Point Cloud & Trajectories

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Input Frames

… … … …

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

Predicted Point Cloud & Trajectories

- Figure 5 Video-based trajectory field estimation on DAVIS [41]. Trace Anything predicts trajectory fields that can yield dynamic point cloud sequences and dense 3D trajectories, while remaining robust to complex non-rigid motion and occlusions.

textured backgrounds. Our engine complements this with diverse non-rigid object motions and more complex, varied environments.

Our released model uses an image encoder and fusion transformer initialized with Fast3R [65] pretrained weights, while the CP heads are randomly initialized. For the choice of parametric curves, our released model adopts B-splines, as detailed in Section B. All models are trained on images at a resolution of 512 pixels on the longest side, using AdamW [34] with a learning rate of 0.0001 and a cosine annealing schedule.

In the first stage, we train on 20K Kubric videos; in the second stage, we use a mixture of 20K Kubric videos and 10K from our engine. We adopt a batch size of 1, with each batch sampling up to 30 frames. Training takes 7.22 days on 32 NVIDIA A100 80GB GPUs. To enable efficient large-scale training, we leverage FlashAttention [7, 8] for improved time and memory efficiency, and apply DeepSpeed ZeRO Stage 2 [46], which partitions optimizer states, moment estimates, and gradients across machines.

###### 5.2 Trajectory Field Estimation

We present qualitative results of trajectory field estimation on videos and image pairs. Qualitative comparisons are provided in Section C.2.

Video to trajectory field. For computational efficiency, we uniformly subsample long sequences to fewer than 60 frames. Figure 5 shows qualitative results on DAVIS [41], covering diverse dynamic scenes. Our predictions faithfully reconstruct both dynamic and static components of the scene, yielding dense, pixel-level

###### 3D trajectories. These trajectories capture motions ranging from near-rigid transformations, such as a toy train moving along a track, to highly non-rigid deformations, such as humans or animals in motion, while also handling severe occlusions and preserving global scene structure.

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Initial Image & Goal Image

[Figure 70]

[Figure 71]

[Figure 72]

Predicted 3D Trajectories

[Figure 73]

[Figure 74]

[Figure 75]

Projected 2D Trajectories

- Figure 6 Image-pair-based trajectory field estimation (goal-conditioned manipulation) on Bridge [53]. Given an initial and a goal image, Trace Anything predicts a trajectory field that interpolates the 3D motion of both the robot arm and manipulated objects. We further show the projected 2D trajectories (see Section C.1 and Figure A for details).

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

Figure 7 Qualitative results with image pairs as input.

Image pair to trajectory field. Our approach can also infer trajectory fields directly from image pairs, effectively reconstructing the implied spatio-temporal dynamics and interpolating intermediate motion. For this experiment, we use BridgeData V2 [53], a large and diverse dataset of robotic manipulation behaviors. Image pairs are sampled from video sequences with a temporal gap of 10–20 frames. As illustrated in Figure 6 and Figure 7, given an initial image of a scene and a goal image specifying the desired outcome, our model predicts a trajectory field that captures plausible 3D trajectories of both objects and agents involved. These

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

- Figure 8 Trajectory fields and camera poses estimated from an unstructured, unordered image set. No sequence information is provided to the model.

trajectories can also be re-projected with estimated camera poses to yield 2D trajectories (see Section C.1 and Figure A for details). In the context of robot learning, this naturally aligns with goal-conditioned manipulation, where predicted trajectories can be interpreted as feasible robot end-effector motions [3].

Unstructured image set to trajectory field. Beyond videos or image pairs, our method also handles unstructured, unordered image sets, a setting not addressed by prior work. The inputs lack both temporal ordering and continuous camera motion, yet our framework by design can also cope with such challenging cases. As shown in Figure 8, our method predicts plausible trajectory fields and camera poses under these conditions. For clarity, we present the input images in chronological order, although no sequence information is provided to the model.

###### 5.3 Quantitative Evaluation

We quantitatively evaluate trajectory field estimation on the Trace Anything benchmark, introduced in Section 4. In contrast to established point tracking benchmarks, which evaluate trajectories only from the

###### Table 1 Quantitative results on video-based trajectory field estimation. CA is reported in 10−2 and SDD in 10−3. Best in bold, second-best underlined.

Method EPEmix ↓ EPEsta ↓ EPEdyn ↓ CA ↓ SDD↓ Runtime (s)↓ CoTracker3 + VGGT 0.518 0.461 0.555 7.83 1.67 197.4 DELTA 0.404 0.384 0.425 6.24 1.75 231.6 SpaTrackerV2 0.296 0.291 0.366 7.24 1.51 178.4 MonsT3R 0.316 0.258 0.330 8.77 1.74 99.1 St4RTrack 0.278 0.247 0.370 9.37 1.76 22.5 POMATO 0.272 0.254 0.308 6.78 1.44 81.8 Easi3R 0.308 0.302 0.324 5.15 1.55 130.9 Trace Anything 0.234 0.218 0.295 5.09 1.06 2.3

###### Table 2 Quantitative results on image-pair-based trajectory field estimation. CA is reported in 10−2 and SDD in 10−3. Best in bold, second-best underlined.

Method EPEmix ↓ EPEsta ↓ EPEdyn ↓ CA ↓ SDD↓ Runtime (s)↓

SEA-RAFT + VGGT 0.226 0.193 0.427 18.22 0.77 1.91 RAFT-3D 0.281 0.219 0.324 17.50 0.98 0.37 MASt3R 0.220 0.181 0.328 33.99 1.70 2.39 MonST3R 0.206 0.167 0.346 20.10 1.25 2.51 St4RTrack 0.211 0.202 0.325 15.33 0.63 1.39 POMATO 0.181 0.137 0.320 19.58 0.84 4.75 Easi3R 0.284 0.269 0.323 20.41 0.91 5.08

Trace Anything 0.135 0.106 0.304 12.41 0.54 0.20

first frame, our protocol requires all-to-all predictions: every pixel in every frame must be associated with a complete trajectory spanning the entire sequence. Evaluation is conducted in two settings: (i) video-based inference, where models process 30-frame video clips, and (ii) image-pair-based inference, where trajectories are estimated from two frames sampled 5 frames apart. All evaluations were conducted using a single NVIDIA A100 GPU. We present other quantitative results in Section C.3 and ablation study in Section C.4.

Metrics. We evaluate reconstruction accuracy using end-point error (EPE). Specifically, EPEmix, EPEsta, and EPEdyn measure the average 3D end-point error over all points, static points, and dynamic points, respectively. To further verify whether the conditions C1 and C2 introduced in Section 3 are satisfied, we introduce two complementary metrics. Static Degeneracy Deviation (SDD) quantifies the temporal jitter of trajectories in static regions, where smaller values indicate better compliance with C1. Correspondence Agreement (CA) measures how consistently dynamic trajectories are predicted from corresponding pixels in different source frames, with lower values indicating better compliance with C2.

Baselines. For video-based inference, we compare against state-of-the-art dynamic scene reconstruction and point tracking approaches, including CoTracker3 [22] (lifted to 3D using VGGT [55]), DELTA [38], SpaTrackerV2 [64], MonsT3R [69], St4RTrack [15], POMATO [70] and Easi3R [5]. For image-pair inference, we compare against the optical flow method SEA-RAFT [60] (lifted to 3D with VGGT [55]), the scene flow method RAFT-3D [51], and several 3D reconstruction approaches, including MASt3R [27], MonST3R [69], St4RTrack [15], POMATO [70] and Easi3R [5].

Results. Quantitative results are shown in Tables 1 and 2. Trace Anything achieves the best performance across all metrics, substantially reducing end-point errors in both static and dynamic regions while also attaining the lowest SDD and CA, indicating stronger compliance with consistency conditions. Moreover, it runs over an order of magnitude faster than optimization-based approaches, underscoring the advantage of our one-pass design.

Runtime breakdown. As shown in Figure 9, our approach exhibits a total runtime that scales approximately linearly with the number of frames. The fusion transformer is the most time-consuming stage, followed by

[Figure 104]

Input Image Pair Extrapolated Trajectories Input Image Pair Extrapolated Trajectories

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

###### Figure 10 Velocity-based forecasting. Per-pixel trajectories are extrapolated by tangent continuation, with reconstructed trajectories in red and extrapolated ones in blue.

- Figure 9 Stage-wise runtime vs. number of input frames.

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Input Frames

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Canonical Frame

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Trajectories

Figure 11 Spatio-temporal fusion. The trajectory field can be leveraged to fuse observations of the dynamic entity across different frames into a canonical frame.

image encoding and curve evaluation. With single-pass inference and no per-scene optimization or external estimators, it exhibits a clear efficiency advantage, as shown in Tables 1 and 2.

###### 5.4 Emergent Capabilities

Trajectory Field representation and Trace Anything model exhibit emergent capabilities that competing approaches do not support.

Velocity-based forecasting. The trajectory field inherently encodes 3D point velocities, per-pixel trajectories can be extrapolated by tangent continuation, allowing dense motion forecasting without additional predictors, as shown in Figure 10.

Instruction-based forecasting. With natural language instructions as input, we leverage image or video generation models to produce future states conditioned on the instructions, and then apply Trace Anything to lift these forecasts into 2D trajectory fields. In Figure 12, we forecast robot actions conditioned on different instructions. We use Seedance 1.0 [18] to generate videos of future states for different instructions, and then apply Trace Anything to predict the trajectory fields from the generated videos.

“Move forward to the desk.”

“Move forward to the sofa.”

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

Initial Image

[Figure 134]

“Move forward 1 meter, then return to the original position.”

“Move backward diagonally to the right for 3 meters.”

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Figure 12 Instruction-based forecasting. Future states are generated using Seedance 1.0.

Spatio-temporal fusion. In Figure 11, predicted trajectory fields enable dynamic entities observed across multiple frames to be consistently fused back into a common canonical frame. This provides a mechanism for aggregating partial observations over time, overcoming occlusions and view changes by aligning pixels to a common reference.

#### 6 Conclusion

We introduced Trajectory Fields, a 4D representation that encodes each pixel of each frame as a 3D trajectory, and Trace Anything, a feed-forward model that predicts trajectory fields from input frames, eliminating auxiliary estimators and per-scene optimization. To support large-scale learning and evaluation, we developed a synthetic data platform. Experiments show that Trace Anything delivers competitive accuracy and inference efficiency, while exhibiting new capabilities.

#### References

- [1] Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski. Building rome in a day. Communications of the ACM, 54(10):105–112, 2011.

- [2] Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhoefer, Johannes Kopf, Matthew O’Toole, and Changil Kim. HyperReel: High-fidelity 6-DoF video with ray-conditioned sampling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

- [3] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables generalizable robot manipulation. In

European Conference on Computer Vision (ECCV), pages 306–324. Springer, 2024.

- [4] Ang Cao and Justin Johnson. Hexplane: A fast representation for dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 130–141, 2023.

- [5] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [6] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In European Conference on Computer Vision (ECCV), pages 306–325. Springer, 2024.

- [7] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

- [8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems (NeurIPS), 35:16344–16359, 2022.

- [9] Carl de Boor. A Practical Guide to Splines. Springer, 1978.

- [10] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 224–236, 2018.

- [11] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems (NeurIPS), 35:13610–13626, 2022.

- [12] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 10061–10072, 2023.

- [13] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 2758–2766, 2015.

- [14] Gerald Farin. Curves and Surfaces for CAGD: A Practical Guide. Morgan Kaufmann, 2002.

- [15] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [16] Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. K-planes: Explicit radiance fields in space, time, and appearance. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12479–12488, 2023.

- [17] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 5712–5721, 2021.

- [18] Yu Gao, Haoyuan Guo, Tuyen Hoang, Weilin Huang, Lu Jiang, Fangyuan Kong, Huixia Li, Jiashi Li, Liang Li, Xiaojie Li, et al. Seedance 1.0: Exploring the boundaries of video generation models. arXiv preprint arXiv:2506.09113, 2025.

- [19] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam Laradji,

- Hsueh-Ti (Derek) Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, Cengiz Oztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, and Andrea Tagliasacchi. Kubric: a scalable dataset generator. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [20] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In European Conference on Computer Vision (ECCV), pages 59–75. Springer, 2022.

- [21] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003.

- [22] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudo-labelling real videos. arXiv preprint arXiv:2410.11831, 2024.

- [23] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision (ECCV), pages 18–35. Springer, 2024.

- [24] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction, 2025. arXiv preprint arXiv:2509.13414.
- [25] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics (TOG), 42(4):1–14, 2023.

- [26] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Joao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. Advances in Neural Information Processing Systems (NeurIPS), 37:82149–82165, 2024.

- [27] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision (ECCV), pages 71–91. Springer, 2024.

- [28] Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, and Lei Zhang. Taptr: Tracking any point with transformers as detection. In European Conference on Computer Vision (ECCV), pages 57–75. Springer, 2024.

- [29] Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8508–8520, 2024.

- [30] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6498–6508, 2021.

- [31] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6498–6508, 2021.

- [32] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

- [33] Xinhang Liu, Yu-Wing Tai, Chi-Keung Tang, Pedro Miraldo, Suhas Lohit, and Moitreya Chatterjee. Gear-nerf: free-viewpoint rendering and tracking with motion-aware spatio-temporal sampling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19667–19679, 2024.

- [34] Ilya Loshchilov, Frank Hutter, et al. Fixing weight decay regularization in adam. arXiv preprint arXiv:1711.05101, 5(5):5, 2017.

- [35] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3d gaussians: Tracking by persistent dynamic view synthesis. arXiv preprint arXiv:2308.09713, 2023.

- [36] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016.

- [37] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In

European Conference on Computer Vision (ECCV), 2020.

- [38] Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, HsinYing Lee, and Chaoyang Wang. Delta: Dense efficient long-range 3d tracking for any video. International Conference on Learning Representations (ICLR), 2025.

- [39] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 5865–5874, 2021.

- [40] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higher-dimensional representation for topologically varying neural radiance fields. ACM Transactions on Graphics (TOG), 2021.

- [41] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 724–732, 2016.

- [42] Les Piegl and Wayne Tiller. The NURBS Book. Springer, 1997.

- [43] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10318–10327, 2021.

- [44] Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, and Jia Deng. Infinite photorealistic worlds using procedural generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12630–12641, 2023.

- [45] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21783–21794, June 2024.

- [46] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

- [47] Peter Sand and Seth Teller. Particle video: Long-range motion estimation using point trajectories. International Journal of Computer Vision (IJCV), 80(1):72–91, 2008.

- [48] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4938–4947, 2020.

- [49] Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4104–4113, 2016.

- [50] Edgar Sucar, Zihang Lai, Eldar Insafutdinov, and Andrea Vedaldi. Dynamic point maps: A versatile representation for dynamic 3d reconstruction. arXiv preprint arXiv:2503.16318, 2025.

- [51] Zachary Teed and Jia Deng. Raft-3d: Scene flow using rigid-motion embeddings. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8375–8384, 2021.

- [52] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, and Christian Theobalt. Non-rigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 12959–12970, 2021.

- [53] Homer Walke, Kevin Black, Abraham Lee, Moo Jin Kim, Max Du, Chongyi Zheng, Tony Zhao, Philippe HansenEstruch, Quan Vuong, Andre He, Vivek Myers, Kuan Fang, Chelsea Finn, and Sergey Levine. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning (CoRL), 2023.

- [54] Chaoyang Wang, Ben Eckart, Simon Lucey, and Orazio Gallo. Neural trajectory fields for dynamic novel view synthesis. arXiv preprint arXiv:2105.05994, 2021.

- [55] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

- [56] Liao Wang, Jiakai Zhang, Xinhang Liu, Fuqiang Zhao, Yanshun Zhang, Yingliang Zhang, Minye Wu, Jingyi Yu, and Lan Xu. Fourier plenoctrees for dynamic radiance field rendering in real-time. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13524–13534, 2022.

- [57] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 19795–19806, 2023.

- [58] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20697–20709, 2024.

- [59] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π3: Scalable permutation-equivariant visual geometry learning, 2025. URL https://arxiv.org/abs/2507.13347.
- [60] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference on Computer Vision (ECCV), pages 36–54. Springer, 2024.

- [61] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [62] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9421–9431, 2021.

- [63] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20406–20417, 2024.

- [64] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: 3d point tracking made easy. IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [65] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025.

- [66] Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. International Conference on Learning Representations (ICLR), 2023.

- [67] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 20331–20341, 2024.

- [68] Jiakai Zhang, Xinhang Liu, Xinyi Ye, Fuqiang Zhao, Yanshun Zhang, Minye Wu, Yingliang Zhang, Lan Xu, and Jingyi Yu. Editable free-viewpoint video using a layered neural representation. ACM Transactions on Graphics (TOG), 40(4):1–18, 2021.

- [69] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. International Conference on Learning Representations (ICLR), 2025.

- [70] Songyan Zhang, Yongtao Ge, Jinyuan Tian, Guangkai Xu, Hao Chen, Chen Lv, and Chunhua Shen. Pomato: Marrying pointmap matching with temporal motion for dynamic 3d reconstruction. IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

- [71] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 19855–19865, 2023.

## Appendix

#### Contents

- 1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 1
- 2 Related Work . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3 Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 3.1 Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2 Network Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.3 Training Scheme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 4 Trace Anything Data Platform . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5 Experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 5.1 Experimental Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 5.2 Trajectory Field Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 5.3 Quantitative Evaluation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
- 5.4 Emergent Capabilities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12

- 6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13

- A Fields . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- B Parametric Curves . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- C Additional Experimental Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- C.1 2D Trajectories, Dynamic Masks, Scene Flow, and Camera Poses . . . . . . . . . . . . . . . . 21
- C.2 Qualitative Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.3 Additional Quantitative Comparison . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- C.4 Ablation Study . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22

- D Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- E LLM Usage Declarations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23 A Fields

- A field is a mapping defined on a domain M, either a continuous space (e.g., Rn) or a discrete space (e.g., Zn), to a codomain V , which may be a scalar space (e.g., R), a vector space (e.g., R3), or a function space (e.g., C∞(N)). Formally, the field is given by

F : M → V. (15)

For instance, the radiance field introduced in [37] maps a 3D coordinate x ∈ R3 and a viewing direction d ∈ S2 (the unit sphere) to a density σ ∈ R+ and a color c ∈ R3. This is expressed as

R : R3 × S2 → R+ × R3, (x,d)  → (σ,c).

(16)

As discussed in Section 3.1, the trajectory field introduced in this work is defined as T : [N] × [H] × [W] → C([0,1],R3), (i,u,v)  → xi,u,v(·),

(17)

where [N], [H], and [W] denote the discrete sets of frame indices and pixel coordinates, respectively, and xi,u,v : [0,1] → R3 is a continuous 3D trajectory for pixel (u,v) in frame Ii. The domain is M = [N]×[H]×[W], and the codomain is V = C([0,1],R3), the space of continuous functions from [0,1] to R3.

#### B Parametric Curves

Our trajectory field representation assigns each pixel to a 3D trajectory, expressed as a parametric curve. In computer graphics, parametric curves are essential for modeling smooth trajectories and surfaces in applications like geometric design and animation [14, 42]. A spline-based parametric curve x(t) : [0,1] → R3 maps a parameter t ∈ [0,1] to 3D space, defined as

n−1

Pkϕk(t), (18)

x(t) =

k=0

where Pk ∈ R3 are control points and ϕk(t) are basis functions. As a widely used class, Bézier curves use Bernstein polynomials as basis functions. A Bézier curve of degree d with d + 1 control points P0,P1,...,Pd ∈ R3 is defined as

d

d i

ti(1 − t)d−i, (19)

x(t) =

PiBi,d(t), Bi,d(t) =

i=0

where Bi,d(t) are Bernstein polynomials [14]. Bézier curves interpolate the first and last control points but lack local control, as adjusting one control point affects the entire curve.

B-spline curves, in contrast, provide local control through a knot vector that defines the parameter intervals where basis functions are active. A B-spline curve of degree p with control points P0,P1,...,Pn−1 ∈ R3 is defined as

n−1

PiNi,p(t), (20)

x(t) =

i=0

where Ni,p(t) are B-spline basis functions determined by a knot vector via the Cox-de Boor recursion formula [9].

In our implementation of Trace Anything, we employ cubic B-splines (p = 3) with clamped, non-uniform knot vectors to parameterize trajectories xi,u,v(t). Each segment is defined by four control points, corresponding to the cubic degree (p = 3). A trajectory is defined as

xi,u,v(t) =

n−1

P(i,u,vk) Nk,3(t), (21)

k=0

where P(i,u,vk) ∈ R3 are control points indexed by i,u,v, and Nk,3(t) are cubic B-spline basis functions determined by a knot vector t = [t0,t1,...,tm−1]. The basis functions are computed via the Cox-de Boor recursion formula:

 

1 if tk ≤ t < tk+1 for k < n + p, 1 if tk ≤ t ≤ tk+1 for k = n + p, 0 otherwise,

(22)

Nk,0(t) =



tk+p+1 − t tk+p+1 − tk+1

t − tk tk+p − tk

Nk+1,p−1(t), (23)

Nk,p−1(t) +

Nk,p(t) =

for p = 1,2,3, with non-zero denominators assumed. For n = 4,7,10 control points, we define knot vectors with multiplicity 4 at t = 0 and t = 1 to ensure interpolation of the first and last control points (xi,u,v(0) = P(0)i,u,v,

xi,u,v(1) = P(i,u,vn−1)). The knot vectors tn are defined as:

 

[0,0,0,0,1,1,1,1] if n = 4, [0,0,0,0,0.5,0.5,0.5,1,1,1,1] if n = 7, [0,0,0,0,1/3,1/3,1/3,2/3,2/3,2/3,1,1,1,1] if n = 10.

(24)

tn =



Internal knots have multiplicity up to 3, ensuring C0-continuity between segments. Knot differences are precomputed for efficient evaluation. Confidence values are interpolated alongside 3D coordinates P(i,u,vk) using the same basis functions ϕk(t) = Nk,3(t), enabling uncertainty-aware trajectory modeling.

#### C Additional Experimental Results

In this section, we present additional experimental results. Please also refer to the supplementary materials for video results, including the presented features, interactive visualization demos, and qualitative comparisons.

###### C.1 2D Trajectories, Dynamic Masks, Scene Flow, and Camera Poses

The outputs of Trace Anything can naturally yield 2D trajectories, dynamic masks, scene flow, and camera poses.

- 2D trajectories. Given the predicted per-pixel 3D trajectories, and with known or estimated camera parameters, we can project them into the image plane to obtain 2D trajectories. In Figure A, we overlay the projected 2D trajectories on the first input frame. We also demonstrate this feature in Figure 6.

Dynamic masks. Our method effectively disentangles static and dynamic components. After Trace Anything predicts control points, we compute the variance over the control-point set associated with each pixel; thresholding this per-pixel variance yields a dynamic mask that cleanly separates static from dynamic regions, as illustrated in Figure B.

Scene flow. Given an input image pair, the scene flow can be obtained as the difference between the two endpoints of the predicted trajectories. In Figure C, we present a 4D reconstruction together with the estimated scene flow from an image pair in the Spring dataset. To highlight robustness under long-range motion, the two images are chosen from non-consecutive frames.

Camera poses. Since Equation (6) provides a world-coordinate point map for each image, we follow Yang et al. [65] (Sec. 4.2) to estimate focal length, rotation, and translation. Our method handles both continuous camera motion in videos and discrete poses from unordered image sets. As shown in Figure D, it correctly recovers camera motion even in dynamic scenes—for example, forward camera movement with perpendicular object motion, or objects in free fall captured by an unordered image set. In the second example, we present the input images in chronological order, although no temporal information is provided to the model.

C.2 Qualitative Comparison

We provide qualitative comparisons of reconstructed point clouds on DAVIS [41]. As shown in Figure E, our method better preserves fine object details (e.g., the elephant’s tail and the flamingo’s neck), correctly handles complex motion, and disentangles static and dynamic objects. Please refer to the supplementary videos for clearer visual comparisons.

C.3 Additional Quantitative Comparison

Out-of-distribution input. To evaluate our model under out-of-distribution conditions, we construct an additional benchmark from PointOdyssey [71], consisting of 50 videos of 30 frames each. Our model has never been trained or fine-tuned on PointOdyssey. As shown in Table A, our method maintains advantages across all metrics as well as inference efficiency.

- 3D tracking. Although our primary task is trajectory field estimation, our method achieves strong results on

- 3D tracking without task- or dataset-specific fine-tuning. We quantitatively compare against other approaches

Table A Quantitative results on out-of-distribution data. CA is reported in 10−2 and SDD in 10−3. Best in bold, second-best underlined.

Method EPEmix ↓ EPEsta ↓ EPEdyn ↓ CA ↓ SDD↓ Runtime (s)↓

St4RTrack 0.269 0.243 0.325 9.82 1.70 19.9 POMATO 0.344 0.319 0.397 6.24 1.72 84.1 Easi3R 0.368 0.307 0.376 7.10 1.99 125.1

Trace Anything 0.256 0.212 0.319 6.19 1.37 2.3

on the TAPVid-3D [26] benchmark. For each subset of TAPVid-3D (ADT, DriveTrack, and PStudio), we sample 50 videos of 60 frames each, using every other frame as input, and report APD3D (average percent of points within a threshold, measuring spatial accuracy) and AJ (average Jaccard, capturing both spatial and occlusion correctness).

Table B Quantitative results on 3D tracking. Best in bold, second-best underlined.

ADT DriveTrack PStudio Method APD3D ↑ AJ↑ APD3D ↑ AJ↑ APD3D ↑ AJ↑ Runtime (s) ↓ VGGT + CoTracker 8.9 9.7 6.2 5.4 8.6 5.8 172.4 St4RTrack 15.2 13.4 8.5 7.4 7.2 6.9 18.9 POMATO 18.2 13.6 11.3 7.8 12.2 8.3 69.2 SpaTracker 18.3 17.4 16.0 10.1 16.2 10.3 191.1 Trace Anything 20.5 15.6 15.5 9.6 16.3 10.8 2.1

In Table B, we present these quantitative results. Notably, SpaTracker [63] is designed and trained for 3D tracking. Our approach remains competitive, surpassing it on some metrics and running orders of magnitude faster, as SpaTracker is limited to a fixed number of query points per run, whereas our model performs per-pixel tracking in a single forward pass.

###### C.4 Ablation Study

Table C presents ablation studies on our Trace Anything benchmark, evaluating both the choice of geometric backbone and the type of parametric curve. For the geometric backbone, we compare the effect of initializing the image encoder and fusion transformer with different pretrained models, including Fast3R [65], VGGT [55], and “None” (following the Fast3R architecture but with random initialization). For the parametric curve types, we evaluate polynomial curves1 as well as Bézier and B-spline curves with varying numbers of control points.

Table C Ablation study on Trace Anything benchmark. CA is reported in 10−2 and SDD in 10−3. Best in bold, second-best underlined.

Backbone Curve Type EPEmix ↓ EPEsta ↓ EPEdyn ↓ CA ↓ SDD ↓ Runtime (s)↓

None B-Spline (10 control points) 0.472 0.416 0.505 8.17 1.08 2.3 Fast3R Polynomial (degree 3) 0.619 0.582 0.673 9.19 1.10 1.8 Fast3R Bezier (4 control points) 0.299 0.271 0.312 5.08 1.11 1.7 Fast3R Bezier (10 control points) 0.238 0.224 0.319 6.13 1.08 2.5 Fast3R B-Spline (4 control points) 0.281 0.264 0.330 6.01 1.08 1.7 Fast3R B-Spline (7 control points) 0.237 0.229 0.317 5.81 1.11 2.1 Fast3R B-Spline (10 control points) 0.234 0.218 0.295 5.09 1.06 2.3 VGGT B-Spline (10 control points) 0.236 0.221 0.276 6.11 1.07 7.2

1Although our method restricts parametric curves to control-point–based ones such as Bézier and B-spline, we experimented with polynomial curves during the early development phase.

As shown in Table C, polynomial curves underperform because their parameters lack the clear geometric and physical interpretability. In contrast, B-spline curves with ten control points achieve the best overall performance, and accuracy generally improves as the number of control points increases. For the backbone, training without pretrained initialization struggles to converge. Compared with Fast3R, VGGT yields modest gains on certain metrics but incurs substantially higher runtime. Nonetheless, we observe VGGT can be beneficial in settings that demand fine structural detail or involve large-baseline scenarios. Based on these results, we adopt Fast3R with B-spline curves (10 control points) as the default configuration in Trace Anything.

#### D Limitations

Since Trace Anything is trained for trajectory field estimation, we rely on synthetic data to obtain dense annotations. This inevitably introduces a domain gap with real-world scenarios. Incorporating partial annotations from real data may help bridge this gap and represents a promising direction for future work.

Our parametric curve representation, with a limited number of control points, has restricted expressive power for highly complex motions. In such cases, we mitigate the issue by clipping trajectories into fixed window sizes or downsampling frames. However, these strategies may fail in scenarios such as repeated back-and-forth motion, and performance also degrades as the number of frames increases. A more fundamental solution likely requires training with larger-scale datasets with high quality.

As the first attempt at dense per-pixel trajectory field estimation, our approach offers efficiency advantages but may be less precise than sparse 3D tracking methods [63, 64]. Incorporating fine-grained point-level estimation from such methods into our framework could be an interesting direction for future research.

#### E LLM Usage Declarations

We declare that Large Language Models (LLMs) were used in a limited capacity during the preparation of this manuscript. Specifically, LLMs were employed for grammar checking, word choice refinement, and typo correction. All core technical contributions, experimental design, analysis, and conclusions are entirely our own. The use of LLMs did not influence the scientific methodology, result interpretation, or theoretical contributions of this research.

Input Frames Trajectory Field Projected 2D Trajectories

[Figure 143]

[Figure 144]

[Figure 145]

…………

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

Figure A Projected 2D trajectories overlaid on the first input frame.

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

RGB

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Dynamic Mask

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

RGB

Dynamic Mask

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

RGB

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

Dynamic Mask

###### Figure B Dynamic mask estimation.

Input Image Pair Scene Flow

4D Reconstruction Input Image Pair 4D Reconstruction Scene Flow

|[Figure 193]|
|---|

[Figure 194]

|[Figure 195]|
|---|

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

###### Figure C 4D reconstruction and scene flow from a single image pair. From a non-consecutive image pair in the Spring dataset, our method recovers both the 4D reconstruction and the scene flow, with x and z components color-coded for visualization.

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

###### Figure D Estimated camera poses over the 4D reconstruction.

Input Frames … … … … … …

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

MonST3R

[Figure 229]

[Figure 230]

[Figure 231]

St4RTrack

[Figure 232]

[Figure 233]

[Figure 234]

POMATO

[Figure 235]

[Figure 236]

[Figure 237]

Easi3R

[Figure 238]

[Figure 239]

[Figure 240]

Ours

###### Figure E Qualitative comparison on DAVIS [41]. Our method better recovers fine details and handles complex motion while disentangling static and dynamic objects.

