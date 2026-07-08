## Multi-View 3D Point Tracking

Frano Rajiˇc1 Haofei Xu1 Marko Mihajlovic1 Siyuan Li1 Irem Demir1 Emircan G¨undo˘gdu1 Lei Ke2 Sergey Prokudin1,3 Marc Pollefeys1,4 Siyu Tang1

1ETH Z¨urich 2Carnegie Mellon University 3Balgrist University Hospital 4Microsoft

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

# arXiv:2508.21060v1[cs.CV]28Aug2025

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

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

| | |
|---|---|
| | |

[Figure 37]

[Figure 38]

Figure 1. We introduce MVTracker, the first data-driven multi-view 3D point tracker for tracking arbitrary 3D points across multiple cameras. Our method fuses multi-view features into a unified 3D feature point cloud, within which it leverages kNN-based correlation to capture spatiotemporal relationships across views. A transformer then iteratively refines the point tracks, handling occlusions and adapting to varying camera setups without per-sequence optimization. This figure shows results on SelfCap [40] using DUSt3R-based [37] depth.

### Abstract

synthetic multi-view Kubric sequences and evaluate on two real-world benchmarks—Panoptic Studio and DexYCBachieving median trajectory errors of 3.1cm and 2.0cm, respectively. Our method generalizes well to diverse camera setups of 1–8 views with varying vantage points and video lengths of 24–150 frames. By releasing our tracker alongside training and evaluation datasets, we aim to set a new standard for multi-view 3D tracking research and provide a practical tool for real-world applications. Project page: https://ethz-vlg.github.io/mvtracker.

We introduce the first data-driven multi-view 3D point tracker, designed to track arbitrary points in dynamic scenes using multiple camera views. Unlike existing monocular trackers, which struggle with depth ambiguities and occlusion, or prior multi-camera methods that require over 20 cameras and tedious per-sequence optimization, our feedforward model directly predicts 3D correspondences using a practical number of cameras (e.g., four), enabling robust and accurate online tracking. Given known camera poses and either sensor-based or estimated multi-view depth, our tracker fuses multi-view features into a unified point cloud and applies k-nearest-neighbors correlation alongside a transformer-based update to reliably estimate long-range 3D correspondences, even under occlusion. We train on 5K

### 1. Introduction

Tracking arbitrary points in 3D [30] is a fundamental problem in computer vision, with numerous applications in dynamic scene reconstruction [14], robotics [10], and aug-

mented reality [25]. While remarkable progress has been made with the recent advancement of 2D point tracking methods [5, 7–9, 12, 16, 17, 45], they remain inherently limited in modeling 3D-consistent motion due to the fundamental ambiguity of the 3D-to-2D projection process. Thus, high-quality 3D point tracking remains a challenging task.

Scene flow approaches [26, 28, 30, 31] can estimate a dense 3D flow field for every 3D point, but they are usually limited to two consecutive video frames, whereas 3D point tracking operates on long sequences (e.g., tens or even hundreds of video frames). Recently, several methods [18, 24, 38] have been proposed to tackle the 3D point tracking task from monocular videos. However, their performance remains far from satisfactory for real-world applications that demand high quality and robustness, due to the difficulty of estimating 3D from single viewpoints in challenging scenarios such as occlusion and complex motion.

To address these challenges, we adopt a multi-camera setup and develop the first feed-forward model that can efficiently and robustly predict long-range 3D point trajectories from multi-view videos. Unlike previous multi-camera methods [23, 30, 44] that require more than 20 cameras and tedious per-sequence optimization, our approach enables feed-forward 3D point tracking with a practical and flexible number of cameras, such as a set of four camera views with arbitrary viewpoints. Thus, our method strikes a promising balance between accuracy and practicality, making it particularly suitable for real-world applications.

Key to our model is a dynamic fused 3D feature point cloud, constructed by combining the unprojected per-view depth maps, which not only effectively aggregates multiview information to a global scene representation, but also facilitates reliable and efficient correspondence search with the k-nearest-neighbors (kNN) operation. This is different from the triplane representation used in the previous stateof-the-art method SpatialTracker [38], which inevitably suffers from information loss during the triplane splatting process and is therefore less effective in processing varying numbers of input cameras. Moreover, our model performs reasonably well with different sources of depth input, whether from accurate depth sensors or noisy estimates from methods such as DUSt3R [37] and VGGT [33].

Using the fused dynamic point cloud, we retrieve local neighbors for each tracked point using a kNN search and compute multi-scale correlation features that capture both appearance similarity and 3D offset information. A spatiotemporal transformer then iteratively refines each point’s 3D position and appearance over a sliding temporal window. Finally, the outputs from overlapping windows are merged to produce globally consistent 3D point trajectories.

To train our model, we construct a synthetic multi-view dataset using Kubric [11] and simulate 5K sequences. We evaluate the model on two real-world datasets, DexYCB [3]

and Panoptic Studio [15], for which we construct 3D point trajectories by leveraging ground-truth object and hand pose estimation labels in DexYCB and merging existing monocular trajectory labels [18] in Panoptic Studio. We conduct extensive experiments on these multi-view video datasets, which suggest that our model outperforms baseline methods by a significant margin. Our model also performs well with different camera setups, different numbers of cameras, and different sources of depth maps, indicating the robustness of our proposed approach. We release our source code and models alongside the training and evaluation datasets to facilitate future research on multi-view 3D point tracking.

### 2. Related Work

Scene Flow. Scene flow methods [26, 28, 30, 31] are usually designed to estimate the dense 3D motion between two consecutive video frames. Traditional methods [26, 30, 31] try to solve this task with an optimization framework, which is typically slow and requires many cameras (e.g., 31 in [30]) to well-regularize the optimization process. Modern approaches [21, 28, 41] explore data-driven models to directly predict scene flow in a feed-forward manner. Despite recent progress, existing models are limited to two frames and are not able to track long-range correspondences in 3D.

- 2D Point Tracking. Recent years have witnessed significant progress in long-term 2D point tracking [5, 7– 9, 12, 13, 16, 17, 34, 45]. For instance, CoTracker2 [16] leverages self-attention to aggregate spatial context from supporting tracks, while LocoTrack [5] extends traditional

- 2D correlation into bidirectional 4D correlation for more robust local matching. These methods typically estimate long-range point trajectories over an entire video, handling occlusions effectively. In this work, we investigate their extension to 3D, but with a sparse multi-view setup and known camera parameters, rather than monocular input. Moreover, unlike recent trackers [9, 17] that reduce the synthetic-toreal domain gap via self-supervision or pseudolabeling, our approach focuses on direct supervision from synthetic data.
- 3D Point Tracking. 3D point tracking has seen innovative contributions from recent methods [23, 24, 35, 38, 39, 42, 43]. SpatialTracker [38] generalizes point tracking into

- 3D by integrating depth information through a Triplane representation [2] and DELTA [24] introduces a coarse-to-fine approach to estimate dense trajectories across the entire image plane, as opposed to sparse point tracking. Both methods assume monocular input. The concurrent SpatialTrackerV2 [39] and TAPIP3D [43] are also monocular. In a different approach, Dynamic 3DGS [23] leverages 3D Gaussian reconstructions over time to track dense scene elements in 3D and perform accurate 2D and 3D point tracking, but requires a multi-camera setup of 27 cameras in Panoptic Studio [15] in addition to sparse depth from depth sen-

sors for point cloud initialization and segmentation masks for relevant objects. We also use multiple input views but assume fewer cameras (e.g., four), which is more practical for real-world applications. Works such as [6, 27, 35] use monocular tracking priors and our experiments suggest that the multi-view extension of [35] is constrained by this. In contrast, our method directly learns a multi-view tracking prior that is neither Gaussian-based nor optimizationbased. VideoDoodles [42] presents an interactive system that anchors hand-drawn animations to objects within a reconstructed 3D scene, harnessing optical flow and depth maps in a novel 3D point tracking algorithm. However, it is primarily designed for video editing rather than competitive performance on standard benchmarks such as TAPVid-

- 2D [7], whereas we focus on 3D point tracking accuracy.

### 3. Method

Our goal is to perform online 3D point tracking across multiple views, given a set of synchronized RGB frames with known camera parameters. The input consists of video sequences captured from V different camera views, with each frame containing a set of query points that need to be tracked over time. We estimate temporally consistent 3D trajectories for these points while also predicting visibility, handling occlusions, and adapting to scene dynamics. Our method runs online at 7.2 FPS given RGB-D input. For RGB-only input, we rely on external depth estimation (e.g., DUSt3R [37] runs at 0.17 FPS and VGGT [33] at 3.1 FPS). An overview of our method is provided in Fig. 2.

Unlike single-view trackers that compute feature correlation on a 2D grid [5, 7, 8, 12, 16, 17, 24] or a triplane [38], our method leverages a fused multi-view point cloud representation to establish k-nearest neighbors (kNN) feature correlations. While 2D-grid-based correlation can match irrelevant background pixels and triplane-based correlation inherently suffers from information loss, our kNN targets geometrically relevant 3D neighborhoods. This enables more robust tracking by integrating geometric consistency across views and learning motion priors directly from data.

#### 3.1. Problem Formulation

Given a calibrated multi-view video sequence Ivt ∈ RH×W×3 captured by V cameras over T frames, our goal is to track N query points in 3D space. We denote each of the N query points as qn = (tnq , xnq , yqn, zqn) ∈ R4, where tnq denotes the query frame and (xnq , yqn, zqn) its initial 3D location in world coordinates. Besides images and query points, the camera intrinsics Kvt ∈ R3×3 and extrinsics Evt ∈ R4×4 are given for each view v and frame t.

The goal is to predict the 3D trajectory {pnt = (xnt , ytn, ztn) ∈ R3, t ≥ tnq }, along with visibilities {vtn ∈ {0,1}} where vtn = 1 denotes that the point is visible in at least one camera view, and vtn = 0 indicates occlusion in

all views. Note that the query location matches the groundtruth track location at time tnq , i.e. pntn

= (xnq , yqn, zqn).

q

#### 3.2. Point Cloud Encoding of Multi-view Videos

Feature Maps. For each input frame Ivt , we extract ddimensional per-view feature maps Φvt = φ(Ivt ) using a convolutional backbone (same as in [16, 17, 38]). We use a stride factor of k = 4 to downscale the input resolution for computational efficiency, such that Φvt ∈ RHk ×Wk ×d. We also compute the feature maps at S = 4 different scales, i.e., Φv,st ∈ R

k2s−1 ×k2Ws−1 ×d, s = 1,...,S. These are obtained by downscaling the base features (e.g., via average pooling) and will later be used in our pyramid correlation.

H

Fused 3D Feature Point Cloud. We obtain multi-viewconsistent depth maps Dvt ∈ RH×W for each view v and frame t using off-the-shelf depth estimation [23, 33, 37] (see Appendix A). Sensor-based depth (e.g., Kinect) can also be used if available and preferred. Using the depth and camera parameters, we lift each valid pixel (ux,uy) into 3D:

−1

−1

t (ux,uy,1)⊤ · Dvt [uy,ux ] . (1) Invalid pixels are those that do not have a valid depth value due to missing Kinect depth values. For estimated depth, we retain all pixels and train for robustness (instead of thresholding by uncertainty predictions). Each of these lifted points x is associated with its corresponding feature from Φv,st and then fused across views into a single point cloud:

x = Ev

t Kv

Xts = x, Φv,st [uy,ux ] v ∈ {1,...,V },

(ux,uy) ∈ Ωvt , (2) where Ωvt is the set of valid pixels in view v. This unified 3D representation facilitates the computation of spatial correlation across views, enabling multi-view-consistent tracking.

An alternative to our fused point cloud is to combine multi-view features via a single global triplane [38], projecting 3D points onto three orthogonal planes (e.g., XY, YZ, ZX). However, this inevitably suffers from projection collisions: different surfaces map to the same planar coordinates, causing destructive feature averaging. It also requires choosing a fixed bounding region for the scene, which leads to wasted resolution or clipped content for large or uncentered scenes. By contrast, our point cloud preserves features directly in 3D, avoids collisions, and adapts naturally to different scenes, resulting in more robust multi-view tracking.

Track Features. We assign a d-dimensional timedependent appearance feature ftn to each tracked point. These features are initialized by sampling from the fused 3D feature point cloud at the query frame and 3D location:

xn,∗,ϕn,∗ = argmin

∥x − pntq

∥, (3)

n

x, ϕ ∈ X t1q

n

for every track n. We use the sampled feature ϕn,∗ to initialize ftn for all t ≥ tqn. These track features are subsequently

[Figure 39]

→M

RN→T→3 RN→T→128 RN→T

pˆn,mt

pˆn,t 0

pˆn,Mt ω(ˆvtn,M)

[Figure 40]

!pˆn,mt +1 !ftn,m+1

| | |
|---|---|
| | |

ftn,m

ftn,0

vˆtn,0

[Figure 41]

{ ...} → RN→3

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Cn,s,mt → RK→4

[Figure 46]

[Figure 47]

[Figure 48]

… …

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

| |
|---|

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

| |
|---|

[Figure 71]

[Figure 72]

[Figure 73]

K E

| |
|---|

[Figure 74]

[Figure 75]

[Figure 76]

| |
|---|

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

 NN             

t = 1 t = i t = T

Figure 2. MVTracker Pipeline. (a) Given synchronized multi-view RGB videos with known intrinsics and extrinsics, we extract perview feature maps using a CNN-based encoder. (b) We construct a point cloud from estimated or sensor-based depth maps, associating each point with learned feature embeddings (Sec. 3.2). (c) We compute directed kNN-based correlation within the point cloud, capturing spatiotemporal relationships across views (Sec. 3.3). (d) A transformer-based update module iteratively refines point trajectories using self-attention over multi-view feature correlations within a sliding temporal window (Sec. 3.4). (e) The model processes sequences in overlapping sliding windows, producing temporally consistent 3D point trajectories with occlusion-aware visibility predictions (Sec. 3.5). The blue blocks denote trainable neural models. The visualized sequence is from SelfCap [40] and uses DUSt3R [37] to estimate depth.

vector (xk −pˆnt ) to encode direction and distance from pˆnt . This offset helps the model disambiguate different nearest neighbors and enables correlation-based matching in 3D.

used when computing track-centric spatial correlation with nearest neighbors from the fused 3D feature point cloud (see Sec. 3.3), but also refined by the spatiotemporal transformer to capture appearance changes over time (see Sec. 3.4).

#### 3.4. Transformer-Based Iterative Tracking

#### 3.3. Multi-Scale Spatial Correlation

Our transformer refines track estimates over a sliding window of T frames. For each track n at time t, we construct a token Gnt = (η(pˆnt − pˆntq

Instead of computing correlations separately for each view [5, 7, 8, 12, 16, 17, 24] or on an auxiliary triplane [38], we establish correspondences directly in the fused feature point cloud using a multi-scale kNN approach. For each query point, we retrieve its K nearest neighbors at different scales from Xts and compute local kNN feature correlations:

), ftn, Cn,st , vˆtn), where η(·) is a sinusoidal positional encoding. The tokens {Gnt }t,n are passed to a transformer ψ, which applies temporal selfattention across time and cross-attention with a small set of learned virtual tracks to model spatial dependencies, following the design in CoTracker2 [17]. The transformer outputs residual position and feature updates (∆pˆnt ,∆ftn) = ψ(Gnt ), which we apply iteratively for m = 1,...,M:

n

Cn,st = ⟨ftn,ϕk⟩ xk,ϕk ∈ NK pˆnt ,Xts , (4) where ⟨·,·⟩ is the dot product and NK(pˆnt ,Xts) finds the knearest neighbors around the current location estimate pˆnt in the fused 3D feature point cloud. These correlations capture spatial dependencies at multiple scales and are fed into our transformer-based tracking module, allowing it to search increasingly larger 3D neighborhoods. For reference, average neighbor distances at the four scales on Panoptic Studio (made-up of 10m–wide scenes) are 12.5±8.2, 22.4±12.2, 42.7 ± 21.7, and 85.8 ± 44.0cm. At the highest scale, this covers frame-to-frame motion of up to 92 km/h at 30 FPS.

pˆn,mt +1 = pˆn,mt + ∆pˆn,mt +1, (5) ftn,m+1 = ftn,m + ∆ftn,m+1. (6)

After each iteration m, we recompute Cn,s,mt based on the refined position and feature. At the final iteration m = M,

we estimate visibility as vˆtn = σ W ftn,M , where σ(·) is the sigmoid function and W a learned projection matrix.

#### 3.5. Windowed Inference and Unrolled Training

Unlike 2D correlation, where pixel offsets implicitly encode direction, our 3D point cloud representation requires explicit 3D offsets. Concretely, for each neighbor (xk,ϕk) we concatenate the local similarity ⟨ftn,ϕk⟩ with the offset

To process long videos, we adopt a windowed approach as in CoTracker [16]. Let T be the maximum window size; for a longer video of length T′ > T, we divide it into

J overlapping windows of length T, each shifted by T/2 frames. Once the transformer completes M iterative updates for window j, its final track estimates initialize window j + 1, allowing refined trajectories to propagate. Let pˆtn,m,j be the predicted location of query n at time t after iteration m in window j, and vˆtn,j its visibility (predicted only at the final iteration). During training, we unroll these windowed updates so that each iteration learns to correct or refine predictions from previous windows and iterations.

#### 3.6. Supervision

Training is supervised with ground-truth trajectories pnt and visibility labels vtn. The total loss consists of a position loss (Lxyz) and a visibility loss (Lvis), balanced by λvis:

L = Lxyz + λvisLvis. (7)

In particular, the position loss is the weighted ℓ1-norm error between the predicted and the ground-truth 3D positions:

γ M−m JMNT

pˆtn,m,j − ptn,m,j 1 , (8)

Lxyz =

j,m,n,t

where j indexes the windows J, m the iterative updates M, n the trajectories N, t the number of frames T, and γ is a weighting factor used to penalize the later iterations of the iterative refinement more. The visibility loss addresses class imbalance by minimizing the balanced binary cross-entropy (B-BCE) between predicted and ground-truth labels:

1 J T N

B-BCE v ˆtn,j, vtn,j . (9)

Lvis =

j,t,n

### 4. Experiments

Datasets. To train our model, we generate a synthetic multi-view dataset, MV-Kub, using Kubric [11], simulating 5K multi-view video sequences. Since no large-scale datasets exist for multi-view 3D point tracking, we rely on synthetic data for supervised training. For evaluation on the hand dexterity dataset DexYCB [3], we sample 10 scenes, leveraging ground-truth object and hand poses to generate track labels. We also construct a Panoptic Studio evaluation dataset by merging existing monocular labels from TAPVid-

- 3D [18] for a total of 6 scenes. Evaluation query points are randomly sampled from both static and moving surfaces.

All results, unless otherwise stated, are reported assuming the availability of estimated depth [37] for DexYCB, ground-truth optimization-based depth [23] for Panoptic Studio, and simulated depth for MV-Kub. In Appendix C we ablate the impact of depth sources on the performance.

Evaluation Metrics. As no standard metrics exist for multi-view 3D point tracking, we extend those from monocular benchmarks [7], reporting four key metrics: (1) Median Trajectory Error (MTE) for visible points, quantifying spatial accuracy; (2) δavg, the average location accuracy over a set of threshold distances; (3) Occlusion Accuracy (OA),

measuring the model’s binary visibility prediction across views; and (4) Average Jaccard (AJ), which jointly evaluates occlusion and position accuracy. All metrics are computed per track, averaged within a scene, and finally aggregated across all dataset scenes. See Appendix F for details. Training. We train our method on MV-Kub for 200K steps on 8 GH200 chips with 96GB GPU memory over 8 days, with a batch size of 8 and up to 2048 trajectories per sample. Our model is implemented in PyTorch (with Lightning Fabric for multi-node training) and optimized using AdamW [22]. The model operates with a latent feature dimension of d = 128 and is trained on MV-Kub sequences of up to 24 frames using a sliding window of 12 frames. We use six-head self-attention layers with a hidden size of 256. We apply a range of augmentations to improve generalization and robustness. See Appendix C.4 for further details.

Baselines. Since there currently exist no other feedforward multi-view point trackers, we implement a triplanebased baseline by extending SpaTracker [38] to fuse multiview features into a single global triplane, referred to as the “Triplane Baseline” in experiments. We also evaluate against two multi-view methods that require test-time optimization: Shape of Motion [35] and Dynamic 3DGS [23]; please refer to Appendix D for their implementation details.

We further evaluate against monocular 2D and 3D point trackers. For 2D point trackers (CoTracker2 [16], CoTracker3 [17], LocoTrack [5]), we run the models on each camera separately with the query points that are deemed to be best visible from that view (based on their projected depth), then lift the resulting 2D estimates to 3D using the known intrinsics and the same depth used by our model. We similarly run 3D trackers (DELTA [24], SpaTracker [38], SpaTrackerV2 [39], TAPIP3D [43]), but give depth as input to directly get 3D tracks. We finally merge per-camera predictions into the world space using known camera poses.

#### 4.1. Method Comparisons

Tab. 1 compares MVTracker against several baselines on Panoptic Studio, DexYCB, and Multi-View Kubric. Our method consistently achieves higher location accuracy (δavg), better occlusion-aware tracking (AJ), and lower trajectory errors (MTE). This is visually apparent in Fig. 3.

On Panoptic Studio, our tracker obtains an AJ of 86.0 and a δavg of 94.7, substantially outperforming single-view methods such as SpaTracker (61.5AJ) and CoTracker3 (74.5AJ). We observe similar trends on DexYCB, where our method reaches an AJ of 71.6 with a median trajectory error of only 2.0cm, compared to the lower performance of LocoTrack, DELTA, and CoTracker3. This is because 2D point trackers are not robust to lifting their 2D trajectories into 3D estimates based on estimated depth maps. Finally, on the Multi-View Kubric validation set, our model attains an AJ of 81.4 and an exceptionally low MTE of 0.7cm.

Table 1. Quantitative evaluation of multi-view 3D point tracking performance. We report results on Panoptic Studio [18], DexYCB [3], and Multi-View Kubric [11]. Our method outperforms existing approaches across all datasets, achieving the highest accuracy (δavg) and occlusion-aware tracking performance (AJ), while maintaining lower median trajectory error (MTE). Notably, our model surpasses the triplane-based baseline, indicating the effectiveness of kNN-directed correlation for multi-view tracking. We report MTE in centimeters. F denotes that the method is feed-forward, data-driven, and online. M denotes that the method supports fusing multi-view inputs. We use estimated depth [37] for DexYCB, ground-truth optimization-based depth [23] for Panoptic Studio, and simulated depth for Kubric.

Panoptic Studio [18] DexYCB [3] (DUSt3R depth) Multi-View Kubric [11]

Method Train F M

AJ ↑ δavg ↑ OA ↑ MTE ↓ AJ ↑ δavg ↑ OA ↑ MTE ↓ AJ ↑ δavg ↑ OA ↑ MTE ↓ Dynamic 3DGS [23] ✗ ✗ ✓ 66.5 92.4 74.1 3.9 45.7 57.1 81.3 11.3 30.4 48.6 68.3 11.2 Shape of Motion [35] ✗ ✗ ✓ 72.6 89.2 82.1 4.8 36.2 53.6 63.3 8.0 57.8 63.4 86.2 5.3 SceneTracker [32] LSFO ✓ ✗ – 85.1 – 6.3 – 61.1 – 5.7 – 56.9 – 7.8 LocoTrack [5] Kub ✓ ✗ 65.8 79.4 79.6 11.8 27.8 38.6 77.0 22.8 52.5 58.4 76.5 12.9 DELTA [24] Kub ✓ ✗ 68.1 86.3 79.2 5.9 36.8 51.6 61.0 18.3 57.4 68.4 77.9 9.6

[Figure 84]

[Figure 85]

[Figure 86]

- CoTracker2 [16] Kub ✓ ✗ 69.5 83.2 80.7 10.4 28.8 40.5 76.2 20.8 54.6 60.3 79.8 10.4
- CoTracker3 [17] Kub+15k ✓ ✗ 74.5 84.5 86.4 8.6 29.4 40.3 78.6 22.0 55.1 60.6 77.7 11.9 SpaTracker [38] Kub ✓ ✗ 55.5 70.8 81.7 9.9 48.1 62.6 73.9 5.5 43.6 55.1 76.3 5.1 SpaTracker [38] MV-Kub ✓ ✗ 61.5 82.3 75.5 7.3 58.3 72.0 80.2 5.9 65.5 77.6 83.1 2.2 SpaTrackerV2 [39] See [39] ✓ ✗ 75.3 85.1 91.2 6.9 35.5 45.1 91.3 9.8 58.6 69.3 86.3 3.9 TAPIP3D [43] Kub ✓ ✗ 84.3 93.8 91.4 3.1 38.8 50.0 90.1 8.2 72.4 86.5 85.4 1.3 Triplane Baseline MV-Kub ✓ ✓ 65.1 81.8 82.8 7.2 57.5 71.0 81.3 4.3 74.7 85.2 90.4 1.2 MVTracker (ours) MV-Kub ✓ ✓ 86.0 94.7 92.3 3.1 71.6 80.6 91.3 2.0 81.4 90.0 93.7 0.7

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

[Figure 104]

[Figure 105]

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

=100=12=12=1224=12=2412=24=150=24=24=150t=90=150=24=12=24=12tt tttt

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

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

Figure 3. Qualitative comparison of multi-view 3D point tracking. We visualize results from a camera viewpoint not used during inference (and not near the input cameras). Each pair of rows corresponds to two time steps from the same dataset. The leftmost column shows ground-truth 3D trajectories, while remaining columns depict predictions from different methods. Points predicted as occluded are indicated by empty circles. Compared to baselines, our method more accurately maintains correspondences across views and handles occlusions. Please see the supp. video and project page for additional visualizations. These examples correspond to the results in Tab. 1.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

- Table 2. Point Correlation Components. “No offset” omits the offset vector entirely, “Offset + location” concatenates both the offset and the neighbor’s world-space coordinates, and “Offset only” encodes just the relative direction. Including only the offset vector (third row) yields the best overall performance. These experiments were trained for 25% of the total steps and on 8 GPUs.

Multi-View Kubric [11] AJ ↑ δavg ↑ OA ↑ MTE ↓ No offset 21.3 45.3 40.6 15.6 Offset + location 48.7 59.6 68.5 6.8 Offset only 53.6 64.9 73.4 4.3

Compared to optimization-based methods, which require per-sequence training, our approach is fully feed-forward and runs at 7.2 FPS. While Shape of Motion achieves reasonable accuracy, its iterative optimization makes it impractical for large-scale or real-time applications. Dynamic 3DGS further requires dense camera setups, limiting its applicability to real-world scenarios with fewer cameras.

Impact of the kNN-based correlation. Our method’s strong performance is largely due to its kNN-based correlation mechanism. Replacing it with a world-aligned triplane correlation leads to a significant drop in performance (see Tab. 1). Triplane-based methods compress multi-view features onto fixed 2D planes, causing destructive feature collisions when different 3D points from the same or different views project to the same grid cell. This leads to information loss that cannot easily be disentangled, especially as the number of views increases. Moreover, triplanes require placing the 2D planes in a fixed scene-aligned coordinate frame with a pre-defined scale and extent, which is difficult to generalize across diverse camera setups and scene scales. In contrast, our kNN-based correlation operates directly in 3D world space and dynamically selects relevant neighbors, avoiding these issues and enabling more robust tracking.

#### 4.2. Ablations

We further evaluate the impact of the point correlation components, the varying number of input views, and different camera setups, as well as training augmentation strategies.

Point Correlation Components. Tab. 2 analyzes the impact of different components in our correlation module. Unlike 2D correlation, where relative direction is implicitly encoded in the pixel grid, kNN in 3D retrieves neighbors from all directions, making explicit offset information crucial. We compare three variants: (i) no offset vector, (ii) offset vector plus explicit neighbor location, and (iii) offset vector only. We train each variant for 25% of total steps on 8 GPUs. The results suggest that (iii) “offset only” achieves the best performance across metrics, indicating that encoding only the relative offset, without concatenating absolute

80

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | |MVTrac Triplane SpatialT|ker (ours) racker| |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | |CoTrack DELTA LocoTra|er3 ck| |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

70

AverageJaccard(AJ)

60

50

40

30

1 2 3 4 5 6 7 8 Number of Views

Figure 4. Effect of the Number of Input Views on DexYCB [3] (DUSt3R-based depth). MVTracker (blue) consistently improves with more views, reaching an AJ of 79.2 with eight views, indicating the benefit of multi-view information. SpatialTracker (green) and Triplane (orange) show moderate improvements but plateau earlier, while single-view methods such as CoTracker3, DELTA, and LocoTrack (red, purple, and brown) exhibit limited gains.

neighbor locations, yields most effective correspondences.

Effect of the Number of Input Views. Fig. 4 shows how performance changes with the number of input views on DexYCB. Our tracker consistently improves as more views are added, achieving an AJ of 64.0 with a single view, 71.1 with four views, and 79.2 with eight views. This trend highlights the importance of multi-view information in reducing depth ambiguities and improving spatial consistency.

Compared to baseline methods, our approach benefits the most from additional views, thus suggesting better scalability. While SpatialTracker and Triplane show moderate improvements with more views, their performance plateaus earlier, indicating difficulty in fully leveraging additional multi-view information. In contrast, single-view methods like CoTracker3, DELTA, and LocoTrack show limited gains, reinforcing their inability to reliably reconstruct 3D correspondences. Similar trends are observed on Panoptic Studio as well as Multi-View Kubric (see Appendix C.2).

Impact of Camera Setups. Tab. 3 shows the performance (measured in AJ) under different camera configurations on Panoptic Studio and DexYCB. For Panoptic Studio, we selected sets of four cameras from the available 27, ensuring they were either positioned opposite each other (Setup A) or placed nearby with varying baseline sizes (Setups B and C). For DexYCB, we sequentially chose four out of the available eight cameras: views 1–4 for setup A, views 3–6 for setup B, and views 5–8 for setup C. Note that on MultiView Kubric the camera configuration is randomized (with cameras generally oriented toward the scene center), so separate evaluations are unnecessary; the main table (Tab. 1) already reflects a random sampling of camera placements. Our method consistently achieves high AJ across all tested

- Table 3. Impact of Camera Setups. Evaluation of AJ under varied camera setups. For Panoptic Studio [18], we select 4 out of the available 27 views as opposing views (setup A) or nearby views (setups B and C). For DexYCB [3], we form 4-view sets out of the available 8 views: (A) views 1–4, (B) views 3–6, (C) views 5–8.

PStudio [18] DexYCB [3] A B C A B C

Method

Dynamic 3DGS [23] 66.5 50.8 56.6 45.7 – – Shape of Motion [35] 72.6 64.3 66.8 36.2 – – LocoTrack [5] 65.8 57.9 63.7 27.8 40.9 42.9 DELTA [24] 68.1 61.1 65.9 36.5 43.3 47.6

- CoTracker2 [16] 69.5 62.3 66.4 28.8 42.0 44.4
- CoTracker3 [17] 74.5 66.3 70.9 29.4 43.8 46.3 SpaTracker [38] 61.5 54.8 57.8 58.3 57.9 63.8 Triplane Baseline 65.1 59.9 63.5 57.5 62.0 66.3 MVTracker (ours) 86.0 75.7 83.2 71.0 71.2 78.3

setups, indicating robustness to varying camera positions.

Ablation of Training Augmentation. In Appendix C.4, we examine how training augmentations impact performance, focusing on (i) the number of views used during training (ranging from 1 to 8) and (ii) the depth map source (ground truth vs. off-the-shelf estimation). Our results show that training with a variable number of views significantly improves robustness to different camera setups, ensuring better generalization across datasets. Additionally, incorporating both accurate and estimated depth maps helps mitigate domain shifts and enhances adaptability to real-world depth estimation errors. This is particularly noticeable on DexYCB, where DUSt3R depth quality varies significantly.

Overall, these ablations suggest that MVTracker remains robust across different camera configurations, benefits from diverse training conditions, and performs best with kNNbased offset encoding and carefully designed augmentations. Please refer to Appendix C for more ablation experiments and additional discussions. The inference speed measurements in Appendix B show that MVTracker runs at 7.2 FPS when provided with sensor depth, underscoring its suitability for near-real-time and large-scale applications.

### 5. Discussion and Future Work

While our method effectively tracks 3D points across multiple views, it has limitations. The key challenge is its reliance on the quality of sparse-view depth estimation. If no sensor depth is available, the approach assumes a reasonably accurate estimated depth. However, in sparse camera setups, such estimation can be unreliable or fail entirely, making tracking infeasible. While our learned motion priors help mitigate moderate noise or incompleteness, a more principled solution would involve jointly estimating depth and tracking for potential mutual refinement, or developing

a foundation model for 4D reconstruction with integrated tracking capabilities. We view this as a critical future direction for the community. Appendix A provides additional analysis on the impact of depth quality, robustness to noise, comparisons across different estimators, and failure cases.

Another limitation is that our study focuses on tracking within bounded regions where camera overlap is sufficient. Extending to unbounded or outdoor environments presents additional challenges due to limited training data availability, scene scale variation, and less constrained viewpoints.

A related issue is scene normalization: our model is trained on a fixed dataset with randomized but similar scene scales and layouts. To bridge distribution gaps at test time, we currently apply manually or heuristically determined similarity transforms. While this works well on our benchmarks and a few additional qualitative datasets, more principled approaches are needed to support arbitrary new scenes.

Finally, long-term 3D point tracking faces the broader challenge of scarce large-scale real-world training data. Unlike optimization-based approaches, data-driven point trackers require diverse training distributions to generalize. However, most existing datasets are synthetic, limiting robustness. Recent community efforts [9, 17] show promise in leveraging self-supervised learning from real-world videos, which could be instrumental for improving generalization.

### 6. Conclusion

We introduce MVTracker, the first data-driven multi-view 3D point tracker, which combines kNN-based correlation in a fused 3D point cloud with a spatiotemporal transformer to track arbitrary points across multiple views. Our method outperforms single-view, multi-view, and optimizationbased baselines, suggesting strong generalization across diverse camera configurations and occluded environments.

We hope that our work enables real-world use cases in robotics [29], provides a better data-driven multi-view prior for optimization-based reconstruction methods [19, 23, 35], and acts as a point tracking head in the upcoming age of scalable end-to-end 4D reconstruction and tracking models.

### Acknowledgements

We are grateful for insightful discussions with Yiming Wang, Zador Pataki, Luigi Piccinelli, Paul-Edouard Sarlin, and Philipp Lindenberger. We thank Ignacio Rocco and Skanda Koppula for clarifications regarding TAPVid-3D. This study was conducted within the national “Proficiency” research project (No. PFFS-21-19) funded by the Swiss Innovation Agency Innosuisse in 2021 as one of 15 flagship initiatives. This work was additionally supported as part of the Swiss AI initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project IDs A03 and A136 on Alps, enabling large-scale training and evaluation.

### References

- [1] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. ZoeDepth: Zero-shot transfer by combining relative and metric depth. arXiv preprint arXiv:2302.12288, 2023. 5
- [2] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In CVPR, 2022. 2
- [3] Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, and Dieter Fox. DexYCB: A benchmark for capturing hand grasping of objects. In CVPR, 2021. 2, 5, 6, 7, 8, 3
- [4] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3R: Estimating disentangled motion from dust3r without training. In ICCV, 2025. 1
- [5] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In ECCV, 2024. 2, 3, 4, 5, 6, 8
- [6] Wen-Hsuan Chu, Lei Ke, and Katerina Fragkiadaki. DreamScene4D: Dynamic multi-object scene generation from monocular videos. In NeurIPS, 2024. 3
- [7] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. TAP-Vid: A benchmark for tracking any point in a video. In NeurIPS, 2022. 2, 3, 4, 5
- [8] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. ICCV, 2023. 3, 4
- [9] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Jo˜ao Carreira, et al. BootsTAP: Bootstrapped training for tracking-any-point. In ACCV, 2024. 2, 8
- [10] Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Junbo Wang, Haoyi Zhu, and Cewu Lu. RH20T: A robotic dataset for learning diverse skills in one-shot. In RSS 2023 Workshop on Learning for Task and Motion Planning, 2023. 1
- [11] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In CVPR, 2022. 2, 5, 6, 7, 3
- [12] Adam W Harley, Zhaoyuan Fang, and Katerina Fragkiadaki. Particle video revisited: Tracking through occlusions using point trajectories. In ECCV, 2022. 2, 3, 4
- [13] Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, and Leonidas J. Guibas. AllTracker: Efficient dense point tracking at high resolution. In ICCV, 2025. 2
- [14] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4D: Learning how

things move in 3d from internet stereo videos. In CVPR,

2025. 1

- [15] Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh. Panoptic studio: A massively multiview system for social motion capture. In ICCV, 2015. 2
- [16] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker: It is better to track together. In ECCV, 2023. 2, 3, 4, 5, 6, 8, 1
- [17] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. CoTracker3: Simpler and better point tracking by pseudolabelling real videos. In ICCV, 2025. 2, 3, 4, 5, 6, 8, 1
- [18] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Jo˜ao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. TAPVid-3D: A benchmark for tracking any point in 3d. In NeurIPS, 2024. 2, 5, 6, 8, 1, 3, 4
- [19] Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. MoSca: Dynamic gaussian fusion from casual videos via 4d motion scaffolds. In CVPR, 2024. 8
- [20] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. MegaSaM: Accurate, fast and robust structure and motion from casual dynamic videos. In CVPR,

2024. 1, 5

- [21] Haisong Liu, Tao Lu, Yihui Xu, Jia Liu, Wenjie Li, and Lijun Chen. CamLiFlow: bidirectional camera-lidar fusion for joint optical flow and scene flow estimation. In CVPR, 2022. 2
- [22] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. 5
- [23] Jonathon Luiten, Georgios Kopanas, Bastian Leibe, and Deva Ramanan. Dynamic 3D Gaussians: Tracking by persistent dynamic view synthesis. In 3DV, 2024. 2, 3, 5, 6, 8, 1, 4
- [24] Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang. DELTA: Dense efficient long-range 3d tracking for any video. In ICLR, 2024. 2, 3, 4, 5, 6, 8, 1
- [25] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In ICCV, 2023. 2
- [26] Christian Richardt, Hyeongwoo Kim, Levi Valgaerts, and Christian Theobalt. Dense wide-baseline scene flow from two handheld video cameras. In 3DV, 2016. 2
- [27] Jenny Seidenschwarz, Qunjie Zhou, Bardienus Duisterhof, Deva Ramanan, and Laura Leal-Taix´e. DynOMo: Online point tracking by dynamic online monocular gaussian reconstruction. In 3DV, 2025. 3
- [28] Zachary Teed and Jia Deng. RAFT-3D: Scene flow using rigid-motion embeddings. In CVPR, 2021. 2
- [29] Mel Vecerik, Carl Doersch, Yi Yang, Todor Davchev, Yusuf Aytar, Guangyao Zhou, Raia Hadsell, Lourdes Agapito, and

- Jon Scholz. RoboTAP: Tracking arbitrary points for few-shot visual imitation. In ICRA, 2024. 8
- [30] Sundar Vedula, Simon Baker, Peter Rander, Robert Collins, and Takeo Kanade. Three-dimensional scene flow. In ICCV,

1999. 1, 2

- [31] Christoph Vogel, Konrad Schindler, and Stefan Roth. 3d scene flow estimation with a rigid motion prior. In ICCV,

2011. 2

- [32] Bo Wang, Jian Li, Yang Yu, Li Liu, Zhenping Sun, and Dewen Hu. SceneTracker: Long-term scene flow estimation network. IEEE TPAMI, 2025. 6, 1
- [33] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025. 2, 3
- [34] Qianqian Wang, Yen-Yu Chang, Ruojin Cai, Zhengqi Li, Bharath Hariharan, Aleksander Holynski, and Noah Snavely. Tracking everything everywhere all at once. In ICCV, 2023. 2
- [35] Qianqian Wang, Vickie Ye, Hang Gao, Weijia Zeng, Jake Austin, Zhengqi Li, and Angjoo Kanazawa. Shape of Motion: 4d reconstruction from a single video. In ICCV, 2025. 2, 3, 5, 6, 8, 1, 4
- [36] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. MoGe: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In CVPR, 2025. 5
- [37] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. DUSt3R: Geometric 3d vision made easy. In CVPR, 2024. 1, 2, 3, 4, 5, 6
- [38] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. SpatialTracker: Tracking any 2d pixels in 3d space. In CVPR, 2024. 2, 3, 4, 5, 6, 8, 1
- [39] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. SpatialTrackerV2: 3d point tracking made easy. In ICCV, 2025. 2, 5, 6
- [40] Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, and Xiaowei Zhou. Representing long volumetric video with temporal gaussian hierarchy. ACM TOG,

2024. 1, 4

- [41] Gengshan Yang and Deva Ramanan. Upgrading optical flow to 3d scene flow through optical expansion. In CVPR, 2020. 2
- [42] Emilie Yu, Kevin Blackburn-Matzen, Cuong Nguyen, Oliver Wang, Rubaiat Habib Kazi, and Adrien Bousseau. VideoDoodles: Hand-drawn animations on videos with scene-aware canvases. ACM TOG, 2023. 2, 3
- [43] Bowei Zhang, Lei Ke, Adam W Harley, and Katerina Fragkiadaki. TAPIP3D: Tracking any point in persistent 3d geometry. In ICCV, 2025. 2, 5, 6
- [44] Chengwei Zheng, Lixin Xue, Juan Zarate, and Jie Song. GSTAR: Gaussian surface tracking and reconstruction. In CVPR, 2025. 2
- [45] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas. Pointodyssey: A large-scale

synthetic dataset for long-term point tracking. In ICCV,

2023. 2

## Multi-View 3D Point Tracking Supplementary Material

### A. Depth Estimation Analysis

Robustness to depth quality. Our method is designed to be tolerant of moderate noise in estimated depth. As shown in Fig. A.1, tracking performance remains stable under additive Gaussian noise up to σ = 2cm. This robustness stems from extensive training augmentations and from the transformer’s ability to interpolate and re-identify points even when partial depth information is missing or corrupted.

Depth source comparisons. Tab. A.1 summarizes performance across different depth sources, including sensor depth, optimization-based reconstruction, and learned estimators (DUSt3R and VGGT). Our method achieves strong performance with noisy but plausible depth maps from learned estimators, particularly VGGT, but degrades significantly when depth estimation produces misaligned geometry or fails.

DUSt3R alignment and runtime. To align DUSt3R [37] reconstructions to our camera setup, we globally optimize the pairwise maps using its built-in alignment (following Sec. 3.4 in DUSt3R), while keeping our camera intrinsics and extrinsics fixed. This reconstruction step takes 300 iterations and runs at 0.17 FPS. VGGT offers a significantly faster alternative at 3.1 FPS.

Discussion. Sparse-view reconstruction remains an open problem. In our evaluations, DUSt3R was the most reliable method for producing usable depth maps, but even it fails on some scenes. As depth estimation continues to improve, particularly with recent advances in foundation models for geometry (e.g., MegaSAM [20], Easi3R [4]), MVTracker can directly benefit (with or without retraining). Fig. A.2 visualizes our predicted tracks under different depth sources, and Tab. A.1 quantifies performance across depth sources.

Table A.1. AJ for different depth sources. GT*: optimizationbased depth. D: DUSt3R. V: VGGT. S: Sensor. ✗: depth estimation failed.

Panoptic Studio DexYCB MV-Kubric GT* D V S D V GT D V

Method

DELTA 68.1 ✗ 1.8 54.6 36.8 21.1 57.4 25.7 1.9 SpaTracker 61.5 ✗ 10.6 60.9 58.3 33.2 65.5 61.7 3.9 Triplane Baseline 65.1 ✗ 29.7 68.0 57.5 61.8 74.7 70.7 46.7 MVTracker (ours) 86.0 ✗ 47.7 82.7 71.6 68.0 81.4 70.0 48.7

### B. Inference Speed Comparison

We evaluate the runtime of various methods by measuring frames per second (FPS) on a representative test configuration. Tab. B.1 summarizes the results, excluding depth-

80

Ours

70

Triplane

SpaTracker

AJ

60

DELTA

50

40

0 1 2 5 10 20 50 100 200 Depth Noise ( , in cm)

- Figure A.1. Robustness to depth noise N(0, σ2) on MV-Kubric.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

- Figure A.2. Different depth sources along our predicted tracks.

estimation time. For optimization-based methods such as Shape of Motion [35] and Dynamic 3D Gaussians [23], the per-sequence runtime is significantly higher (e.g., about 30 minutes for Shape of Motion and about 50 minutes for Dynamic 3DGS on a single Panoptic Studio [18] sequence). Overall, the optimization-based approaches are orders of magnitude slower, making them less practical for largescale applications and unusable for real-time applications. In contrast, our feed-forward model runs online at a comparable speed to the triplane-based baseline while outperforming it in accuracy, highlighting the efficiency of our kNN correlation in 3D point clouds. Note that we correctly adjust the effective FPS by taking into account the window size for sliding-window-based methods.

Table B.1. Runtime Comparison (FPS). We measure frames per second for each method under a representative test setting, excluding depth-estimation overhead. Optimization-based methods are indicated with placeholders for their much lower throughput.

##### Method FPS

Dynamic 3DGS [23] (optimization-based) Shape of Motion [35] (optimization-based) SpaTracker [38] 1.4 DELTA [24] 1.5 SceneTracker [32] 3.5

- CoTracker2 [16] 5.7 Triplane Baseline 5.8 MVTracker (ours) 7.2

- CoTracker3 [17] 18.4

Table C.1. Impact of Different Depth Sources on DexYCB. We report results using either DUSt3R-estimated [37] or sensorbased depth for all methods. Sensor depth generally improves performance across the board, underscoring the importance of highfidelity depth. Notably, our multi-view tracker benefits the most, indicating that our fused 3D point cloud representation effectively capitalizes on better depth quality.

Method AJ↑ δavg↑ OA↑ MTE↓ DexYCB [3] (DUSt3R depth)

LocoTrack [5] 27.8 38.6 77.0 22.8 DELTA [24] 36.8 51.6 61.0 18.3

- CoTracker2 [16] 28.8 40.5 76.2 20.8
- CoTracker3 [17] 29.4 40.3 78.6 22.0 SpaTracker [38] 58.3 72.0 80.2 5.9 SpaTrackerV2 [39] 35.5 45.1 91.3 9.8 TAPIP3D [43] 38.8 50.0 90.1 8.2 Triplane Baseline 57.5 71.0 81.3 4.3 MVTracker (ours) 71.6 80.6 91.3 2.0

DexYCB [3] (sensor depth)

LocoTrack [5] 55.1 64.6 76.7 27.5 DELTA [24] 54.6 67.8 72.6 42.6

- CoTracker2 [16] 56.4 67.4 76.4 26.4
- CoTracker3 [17] 57.3 68.5 76.7 23.4 SpaTracker [38] 60.9 75.0 82.5 4.0 SpaTrackerV2 [39] 69.2 77.5 91.3 4.2 TAPIP3D [43] 77.2 88.9 88.0 1.3 Triplane Baseline 68.0 79.5 85.9 2.6 MVTracker (ours) 82.7 91.0 91.6 0.8

### C. Ablation Study

#### C.1. Impact of Different Depth Sources

Table C.1 reports the performance of our model and several baselines on DexYCB when using two different depth sources: DUSt3R [37] (estimated) and sensor-based depth. We observe that sensor-based depth leads to improved accuracy for all methods, highlighting the importance of depth quality. Notably, our multi-view tracker benefits substantially from sensor depth, surpassing single-view methods and the triplane baseline by a larger margin. This suggests that our fused 3D point cloud representation is robust to a wide range of depth sources, yet can further leverage higher-fidelity depth to achieve stronger 3D point tracking.

#### C.2. Varying the Number of Input Views

Tab. C.2 shows detailed per-dataset results for a varied number of input views. Our tracker consistently improves as more views are added.

Table C.2. Varying the Number of Input Views. Evaluation of AJ under a varied number of input views.

Number of views 1 2 3 4 5 6 7 8 Panoptic Studio [18]

Method

LocoTrack [5] 54.7 56.4 61.4 65.8 65.8 67.3 66.4 66.8 DELTA [24] 60.9 61.8 63.7 68.1 66.7 67.4 67.6 66.3

- CoTracker2 [16] 59.5 60.7 64.4 69.5 68.8 69.3 68.4 69.9
- CoTracker3 [17] 61.9 65.0 68.5 74.5 74.0 75.2 75.1 76.0 SpaTracker [38] 51.2 52.1 57.5 61.5 60.9 61.4 62.6 63.4 SpaTrackerV2 [39] 57.4 62.7 66.8 72.4 71.8 72.7 72.7 72.9 TAPIP3D [43] 68.8 72.9 78.5 84.3 84.6 85.6 86.0 86.8 Triplane Baseline 61.9 60.9 61.6 65.0 64.0 65.1 66.8 65.6 MVTracker (ours) 66.4 72.2 79.0 86.0 87.4 88.8 89.1 89.9

DexYCB [3] (DUSt3R depth)

LocoTrack [5] 27.9 26.0 28.1 27.8 36.3 34.8 34.7 34.9 DELTA [24] 33.0 34.3 38.0 36.5 37.2 35.4 34.9 35.7

- CoTracker2 [16] 29.8 26.4 29.2 28.8 37.8 36.2 36.0 36.0
- CoTracker3 [17] 28.6 27.0 29.5 29.4 39.1 37.5 37.1 37.3 SpaTracker [38] 60.6 58.4 61.8 58.3 63.2 62.4 62.9 63.4 SpaTrackerV2 [39] 39.8 39.5 36.5 35.5 41.1 37.1 37.0 37.7 TAPIP3D [43] 36.6 35.6 40.5 38.8 57.7 54.2 55.2 56.4 Triplane Baseline 44.0 48.0 56.0 57.6 63.5 64.5 65.5 66.8 MVTracker (ours) 64.0 66.8 73.2 71.1 77.4 76.7 77.3 79.2

Multi-View Kubric [11]

LocoTrack [5] 57.3 56.1 53.8 52.5 52.3 52.1 51.6 51.8 DELTA [24] 65.9 61.9 58.9 57.4 56.3 56.0 54.8 54.1

- CoTracker2 [16] 61.4 59.1 56.1 54.6 53.8 53.7 53.1 52.8
- CoTracker3 [17] 60.1 58.7 56.5 55.1 54.8 54.6 54.2 54.3 SpaTracker [38] 72.8 70.3 67.4 65.5 64.8 64.5 63.1 62.7 SpaTrackerV2 [39] 54.0 57.6 57.6 58.6 58.3 59.1 58.8 59.7 TAPIP3D [43] 76.4 75.4 72.8 72.5 71.7 71.4 70.9 70.6 Triplane Baseline 71.7 73.4 74.1 74.7 75.0 75.2 75.2 75.7 MVTracker (ours) 81.7 81.2 81.4 81.4 81.9 82.1 82.3 82.5

#### C.3. 2D Tracking Accuracy

Tab. C.3 compares the performance of 2D point tracking methods when their outputs are projected into 2D using standard metrics [7]. Although 2D methods do not rely on estimated depth quality, they generally achieve lower location accuracies compared to our multi-view approach, which fuses multi-view data to enhance performance. On DexYCB, we only report results using sensor depth maps to provide a more informative comparison in 2D point tracking. For results with depth estimated by DUSt3R, where single-view methods benefit from not having to handle noisy or incomplete depth maps, see Tab. C.4. Note that such depth noise or incompleteness affects our 2D tracking more than that of SpaTracker, due to SpaTracker’s XYaligned feature plane compared to our fused 3D point cloud.

- Table C.3. 2D Tracking Evaluation. Evaluation of projected 2D tracking performance using standard 2D metrics [7]. Multi-view outputs are projected onto four views and the metrics are averaged across these views. Note that we can only report location accuracies since our multi-view methods report any-view visibility (not per-view).

Method δavg2D↑ δ<2D1↑ δ<2D2↑ δ<2D4↑ δ<2D8↑ δ<2D16↑ Panoptic Studio [18]

LocoTrack [5] 66.7 29.3 51.4 72.6 86.5 93.7 DELTA [24] 70.0 34.8 56.5 76.5 87.6 94.8

- CoTracker2 [16] 64.3 28.0 49.3 69.7 82.9 91.4
- CoTracker3 [17] 70.9 33.1 56.8 78.2 90.5 96.2 SpaTracker [38] 66.4 29.3 51.8 71.8 85.7 93.3 Triplane Baseline 51.9 8.5 24.1 52.0 80.3 94.5 MVTracker (ours) 70.5 26.6 52.9 79.5 94.6 98.9

DexYCB [3] (sensor depth)

LocoTrack [5] 83.7 74.1 78.4 84.0 89.4 92.4 DELTA [24] 84.6 75.7 79.9 85.2 89.4 92.8

- CoTracker2 [16] 83.7 73.7 78.9 84.5 89.3 92.1
- CoTracker3 [17] 84.1 74.8 79.0 84.5 89.4 92.9 SpaTracker [38] 85.8 75.9 80.7 86.1 91.2 94.9 Triplane Baseline 75.9 52.2 70.4 78.9 85.9 92.0 MVTracker (ours) 87.5 73.3 81.2 88.3 95.5 99.2

Multi-View Kubric [11]

LocoTrack [5] 75.3 53.6 66.8 78.0 86.1 91.9 DELTA [24] 78.4 61.9 72.0 80.3 86.5 91.2

- CoTracker2 [16] 72.6 51.8 63.7 74.2 83.1 90.2
- CoTracker3 [17] 74.8 54.2 66.3 76.8 85.4 91.5 SpaTracker [38] 75.8 51.4 65.8 79.3 88.5 94.0 Triplane Baseline 80.6 51.3 72.4 87.3 94.5 97.7 MVTracker (ours) 86.8 61.2 81.7 94.0 97.9 99.2

- Table C.4. Evaluation of projected 2D tracking performance using standard 2D metrics [7] on DexYCB with estimated depths. Single-view methods only require RGB for tracking and do not depend on the quality of the depth for achieving high 2D point tracking performance. However, the multi-view methods face challenges when dealing with the noisy depth estimates on DexYCB.

Method δavg2D↑ δ<2D1↑ δ<2D2↑ δ<2D4↑ δ<2D8↑ δ<2D16↑ DexYCB [3] (DUSt3R depth)

LocoTrack [5] 85.4 75.6 80.0 85.8 91.4 94.4 DELTA [24] 85.6 77.0 81.2 86.0 90.2 93.7

- CoTracker2 [16] 86.2 75.7 81.3 87.1 92.1 95.0
- CoTracker3 [17] 86.7 77.0 81.5 87.2 92.1 95.7 SpaTracker [38] 86.0 76.4 80.8 86.2 91.7 95.1 Triplane Baseline 64.3 31.8 53.6 71.4 79.3 85.4 MVTracker (ours) 71.3 42.1 59.8 76.0 85.2 93.5

- Table C.5. Training Augmentation of the variable number of input views ranging from 1 to 8 (V) and varying depth sources between ground-truth and off-the-shelf depth estimation (D) during training on tracking performance.

V D AJ ↑ δavg ↑ OA ↑ MTE ↓ Panoptic Studio [18]

✓ 80.4 94.4 87.8 3.1 ✓ 80.7 93.3 88.7 3.5 ✓ ✓ 80.7 94.3 88.2 3.2

DexYCB [3]

✓ 49.9 72.4 70.5 3.6 ✓ 62.3 77.1 82.1 3.2 ✓ ✓ 65.2 79.5 82.9 2.3

Multi-View Kubric [11]

✓ 80.2 91.1 91.1 0.7 ✓ 77.2 88.0 91.0 0.8 ✓ ✓ 80.4 90.5 92.1 0.7

- Table C.6. Training Augmentation with evaluation on different numbers of views. We investigate the impact of using a variable number of views ranging from 1 to 8 (V) and varying depth sources between ground-truth and off-the-shelf depth estimation (D) during training on tracking performance. Results are reported in AJ.

V D Number of views

1 2 3 4 5 6 7 8 Panoptic Studio [18]

✓ 69.3 72.0 76.9 80.4 81.4 84.4 84.8 85.1 ✓ 67.0 71.3 77.2 80.7 81.9 84.2 84.9 85.8 ✓ ✓ 68.1 73.1 76.9 80.7 81.5 83.4 84.4 84.3

DexYCB [3] (DUSt3R depth)

✓ 37.6 40.0 48.1 49.9 65.6 67.4 69.6 73.4 ✓ 51.6 54.1 60.7 62.3 68.8 69.9 71.8 74.8 ✓ ✓ 56.3 58.9 64.9 65.2 74.0 75.4 76.3 78.9

Multi-View Kubric [11]

✓ 80.7 79.9 80.1 80.2 80.9 81.4 81.8 81.9 ✓ 75.9 76.3 76.6 77.2 77.2 78.0 78.1 78.4 ✓ ✓ 80.4 79.6 80.0 80.4 81.1 81.6 81.8 82.0

#### C.4. Training Augmentation

Tab. C.5 and C.6 evaluate the impact of training augmentation strategies. We consider two factors: (V) employing a variable number of views during training (ranging from 1 to 8) and (D) varying the source of depth maps (ground-truth versus off-the-shelf depth estimation). The results reveal that combining both variable view and depth augmentations leads to the best performance, especially as measured on DexYCB.

We apply a range of augmentations to improve gener-

Table D.1. Dynamic 3D Gaussians. Novel view synthesis scores for Dynamic 3DGS [23] under varying numbers of cameras on Panoptic Studio [18].

Number of Cameras 27 17 9 4

PSNR 28.5 23.7 17.9 12.6 SSIM 0.91 0.83 0.74 0.52

alization and robustness. These include photometric augmentations such as color jitter, Gaussian blur, and occlusion simulation via erasing and region replacement. Spatial augmentations involve random cropping, padding, flipping, and scaling. Depth perturbations include noise, rescaling, and occlusion-based erasure. Scene-level augmentations transform the world coordinate system through random rotation, translation, and scaling. Camera intrinsics and extrinsics are perturbed to simulate realistic variations. Additionally, we vary the number of tracks per sample, randomly sample the number of input views (between 1 and 6), and use either ground-truth depths or depths from an estimation method to improve model adaptability.

### D. Baselines

Dynamic 3D Gaussians. We use the original training code provided by the authors of the Dynamic 3D Gaussians [23]. The training process utilizes camera intrinsics, extrinsics, segmentation masks, images, and an initial point cloud. This initial point cloud is constructed from depth maps corresponding to selected viewpoints at the initial timestep. To differentiate between static and dynamic points within the point cloud, we leverage the provided segmentation masks.

For tracking, we identify the most influential Gaussian for a given point at time t based on the influence computation described in the Dynamic 3DGS paper. To establish a trajectory, we then track the motion of this Gaussian across all time steps. The visibility of each point at time t is determined by comparing the depth of the Gaussian with the corresponding depth value of the depth map rendered from each camera viewpoint. Trajectories that show sudden discontinuities or have a dominant Gaussian influence near zero for the query point are classified as static to ensure consistency in motion.

To evaluate the model’s performance in novel view synthesis, we conducted experiments on Panoptic Studio. We trained different models using a varying number of cameras. Then, we render images using these trained models from four test cameras. Our results in Tab. D.1 show that with 27 input views, we achieve reconstruction quality comparable to that reported in the original paper. However, as the number of views decreases, reconstruction quality reduces.

Shape of Motion. We adapt the monocular training pipeline of the original Shape of Motion [35] to a multiview setting. The method requires segmentation masks for moving objects, depth maps, long-range 2D tracks, and camera poses. In our multi-view adaptation, we rely on the available depth maps, segmentation masks, and camera poses, while using TAPIR [8] to obtain long-range 2D tracks, as in the original approach.

For both static and dynamic part initialization, we use multi-view depth maps and segmentation masks to construct an initial 3D representation that captures the overall scene structure more effectively than the monocular approach. We follow the original paper’s strategy for choosing the canonical frame and optimizing motion bases. Additionally, we adapt the monocular loss function to a multiview formulation, computing the original loss function separately for each view and averaging the results across all views. This adjustment ensures that the optimization effectively integrates multi-view information.

### E. Evaluation on TAPVid-2D

In Tab. E.1, we evaluate MVTracker on the TAPVid-2D benchmark by projecting our predicted 3D trajectories onto image views. As our method requires depth input, we report results using various monocular depth estimators. However, tracking performance is significantly degraded in this benchmark, which is in part attributable to TAPVid-2D containing outdoor unbounded scenes, our scene normalization not aligning scenes well to the training distribution, and monocular depth estimation frequently failing or flickering.

### F. Evaluation Metrics

There is no established set of metrics for multi-view point tracking in 3D. Thus, we adopt four metrics from prior monocular point-tracking benchmarks [7], extending them to our multi-view 3D setting. These metrics measure the quality of the predicted 3D trajectories (δavg, MTE), the ability to predict whether a point is visible in any of the views (OA), or both simultaneously (AJ). We compute all metrics per-trajectory before averaging across all tracks in a scene, and then across all scenes in a dataset.

Let N be the number of tracked points, T the number of frames, and pˆit (resp. pit) the predicted (resp. ground-truth) 3D location of track i at time t. Similarly, let vˆti ∈ {0,1} and vti ∈ {0,1} be the predicted and ground-truth visibility flags.

Median Trajectory Error (MTE). For each track i, the median trajectory error measures the median distance between predicted and ground-truth locations over timesteps

Table E.1. TAPVid-2D Evaluation. Evaluation results in 2D point tracking on TAPVid-2D [7]. Since our method requires depth as input, we report results for several monocular depth estimators. However, performance is far from satisfactory for most trajectories, which is in part attributable to out-of-distribution outdoor scenes with many far-away points, scenes not being normalized well to the training scale, failures in depth estimation, flickering in the estimated video depth, etc. We report both the original TAPVid-2D metrics as well as the metrics used elsewhere in this paper; the difference is that our metrics take the average of per-trajectory metrics, whereas TAPVid-2D computes each metric over all points, treating all trajectories as one trajectory. All metrics are in pixel scale (e.g., δ<2D4 is the within-aradius-of-4-pixels location accuracy, MTE is in pixels, etc.). Best and second-best results per metric are bold and underlined, respectively.

Method Depth AJ2D AJ2D<1 AJ2D<2 AJ2D<4 AJ2D<8 AJ2D<16 δavg2D δ<2D1 δ<2D2 δ<2D4 δ<2D8 δ<2D16 OA MTE TAPVid-2D Metrics

- MVTracker (ours) ZoeDepth [1] 9.5 1.6 3.9 8.5 14.3 19.4 20.4 3.0 7.5 16.7 29.7 45.3 46.5 –

- MVTracker (ours) MoGe [36] 13.1 2.0 4.4 9.5 19.5 30.0 25.0 3.4 8.0 18.4 37.5 57.6 54.6 – MVTracker (ours) MegaSAM [20] 31.6 5.7 14.4 31.3 47.4 59.1 46.0 11.0 25.3 47.8 66.8 79.1 78.6 – CoTracker3 [17] none required 64.1 28.2 51.3 72.2 82.4 86.4 77.0 42.1 67.1 85.3 93.3 97.0 91.0 –

Our Metrics (where numbers are computed per track and then averaged) MVTracker (ours) ZoeDepth [1] 10.3 1.8 4.4 9.5 15.6 20.4 21.8 3.3 8.4 18.3 31.9 47.3 46.1 59.1

- MVTracker (ours) MoGe [36] 14.6 2.4 5.4 11.6 22.0 31.6 26.7 3.9 9.2 20.5 40.4 59.7 55.1 86.0 MVTracker (ours) MegaSAM [20] 35.0 7.4 17.4 35.7 51.9 62.6 46.4 12.1 26.2 48.1 66.7 78.9 79.8 14.0 CoTracker3 [17] none required 66.7 32.6 56.2 74.9 83.3 86.4 77.2 43.2 67.6 85.3 93.1 96.7 91.7 2.9

where the track is visible:

MTEi = median ∥pˆit − pit∥2 t = 1,...,T, vti = 1 .

(F.1) The overall MTE is the mean across all N tracks:

MTE = N1

N

MTEi. (F.2)

i=1

Occlusion Accuracy (OA). Occlusion accuracy quantifies how well the model predicts visibility:

T

OAi = T1

t=1

OA = N1

1 v ˆti = vti , (F.3)

N

OAi. (F.4)

i=1

Average Location Accuracy (δavg). We evaluate the fraction of points within each of H distance thresholds {x1,...,xH} in centimeters. For a single threshold x, define the following:

δxi = 1 T t=1 vti

T

1 vti = 1 1 ∥pˆit − pit∥2 < x . (F.5)

t=1

We then average over all tracks and thresholds:

δx = N1

N

H

δxi , δavg = H1

i=1

h=1

δx

h

. (F.6)

Average Jaccard (AJ). This metric jointly assesses spatial and occlusion accuracy at each threshold x. Define αt,xi = 1 ∥pˆit − pit∥2 < x . Then, for track i:

T t=1 vtivˆtiαt,xi

AJxi =

.

T t=1 vti + (1 − vti)ˆvti + vtivˆti(1 − αt,xi )

(F.7) Averaging over N tracks yields

AJx = N1

N

H

AJxi, AJ = H1

i=1

h=1

AJx

h

. (F.8)

#### F.1. Evaluation Details

For MV-Kubric and DexYCB, we sample 512 points uniformly from object surfaces and report metrics averaged across static and dynamic points. Panoptic Studio often lacks labeled static objects, so we sample 512 points and compute metrics over all of them. We use four views in the main evaluation tables: for DexYCB, these are the first four calibrated cameras; for Panoptic Studio, we select distant views; and for MV-Kubric, views are randomly sampled per scene. The sensitivity to the selected views is analyzed in Tab. 3. Jaccard and location accuracy thresholds are set to 1/2/5/10/20 cm for DexYCB, 0.65/1.3/2.6/5.2/10.4 simulation-scale-adjusted centimeters for MV-Kubric (1 Kubric unit is about 13 cm before simulation-scale adjustment), and 5/10/20/40 cm for Panoptic Studio.

Panoptic Studio trajectories were generated by merging per-view monocular labels from TAPVid-3D [7], filtered from Dynamic 3DGS [23] 27-view predictions. These la-

bels are often noisy and erroneous, and for example, exhibit point drift (e.g., labels jumping from an elbow to a finger), which is why we use higher evaluation thresholds on this benchmark and recommend interpreting results as well as using the benchmark with caution. MV-Kubric labels are derived from simulated ground-truth trajectories, while DexYCB tracks are extracted from fitted object meshes provided with the dataset. For DexYCB, we sample points on annotated surfaces and track their position using barycentric coordinates of the deforming mesh triangles. Visibility labels on DexYCB can still be noisy due to imperfect Kinect depth maps and segmentation boundaries. The scripts used to process all datasets and generate labels are publicly available in our codebase.

