# arXiv:2603.02573v2[cs.CV]5Mar2026

## Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels

Jiahao Lu1 Jiayi Xu1 Wenbo Hu2† Ruijie Zhu2 Chengfeng Zhao1 Sai-Kit Yeung1 Ying Shan2 Yuan Liu1† 1The Hong Kong University of Science and Technology 2ARC Lab, Tencent PCG

Project page: https://jiah-cloud.github.io/Track4World.github.io/

[Figure 1]

Figure 1. Track4World estimates dense 3D scene flow of every pixel between arbitrary frame pairs from a monocular video in a global feedforward manner, enabling efficient and dense 3D tracking of every pixel in the world-centric coordinate system.

### Abstract

real-world 4D reconstruction tasks.

Estimating the 3D trajectory of every pixel from a monocular video is crucial and promising for a comprehensive understanding of the 3D dynamics of videos. Recent monocular 3D tracking works demonstrate impressive performance, but are limited to either tracking sparse points on the first frame or a slow optimization-based framework for dense tracking. In this paper, we propose a feedforward model, called Track4World, enabling an efficient holistic 3D tracking of every pixel in the world-centric coordinate system. Built on the global 3D scene representation encoded by a VGGT-style ViT, Track4World applies a novel 3D correlation scheme to simultaneously estimate the pixelwise 2D and 3D dense flow between arbitrary frame pairs. The estimated scene flow, along with the reconstructed 3D geometry, enables subsequent efficient 3D tracking of every pixel of this video. Extensive experiments on multiple benchmarks demonstrate that our approach consistently outperforms existing methods in 2D/3D flow estimation and 3D tracking, highlighting its robustness and scalability for

†Corresponding Authors.

### 1. Introduction

Reconstructing 4D dynamics of videos [7, 18, 48, 52, 55, 59, 64, 86, 94, 97, 104] is a task of great significance, with far-reaching applications [5, 24, 29, 99] in robotics, animation production, physical law inference, and other fields. To grasp the holistic dynamics of videos, it is essential to predict the dynamic motion of all pixels in 3D space from monocular video inputs. However, this still remains an extremely challenging problem: monocular geometric reconstruction is inherently ill-posed to recover 3D from singleview observations, and further tracking 3D points at different time steps adds an extra layer of complexity.

While recent advances have addressed isolated aspects of this problem, robust and holistic 3D tracking for every pixel is still an unsolved problem. Some recent trackingbased methods (e.g., St4RTrack [18], SpatialTrackerV2 (STV2) [87], DELTA [59]), extends feedforward 3D geometry estimation [76, 82] with temporal association for 3D tracking. Though achieving impressive performance and accuracy, these methods are restricted to 3D tracking of the

points on the first frame, failing to capture motions for new pixels in subsequent frames. Another work called TrackingWorld [55] enables dense 3D tracking of all frames by fusing multiple modalities, such as 2D flow [34], masks [92], and depth priors [62]. While being effective, this pipeline is computationally expensive and limited by the inability to learn joint spatiotemporal priors, often resulting in temporally inconsistent or suboptimal solutions. Following recent trends in feedforward 3D reconstruction methods, such as DUSt3R [82], VGGT [76], and Pi3 [85], designing an efficient feedforward model for holistic 3D tracking of every pixel in a monocular video becomes a promising direction.

In this work, we propose Track4World, a feedforward framework designed to estimate 3D tracking for every pixel of every frame in a monocular video in the world-centric coordinate system. Following the feedforward design trends, Track4World is built upon the VGGT-like ViT-style 3D geometry reconstruction framework [49, 76, 85] and associates different frames to estimate dense 3D tracking for every pixel in the world-centric coordinate system.

Designing such an association scheme for the 3D tracking of every pixel is computationally expensive. A naive solution is to generalize the 3D tracking methods, Spatial Tracker V2 (STV2) [87] or DELTA [59], to track every pixel. However, explicitly predicting trajectories for every pixel on all frames is computationally prohibitive. The large number of pixels leads to unaffordable memory and computation consumption during both training and inference, with huge redundant 3D trajectories across different frames.

Instead of directly predicting 3D trajectories of all pixels, Track4World resorts to predicting the two-frame scene flows as the representation of the holistic 3D video dynamics. We draw inspiration from efficient pair-wise motion estimation methods, such as St4RTrack [18] and ZeroMSF [48], to estimate the dense scene flow between arbitrary frame pairs. Then, the estimated pair-wise motion can be easily utilized to construct the 3D tracking of arbitrary pixels in a video. By decomposing the continuous tracking problem into pair-wise scene flow estimations, we significantly reduce the computational redundancy, transforming this into a more manageable task.

However, effectively adapting a VGGT-like ViT-style 3D reconstruction framework for dense scene flow estimation remains an open research problem. A straightforward solution might involve appending a motion decoding head to regress 3D scene flow directly [18, 48]. Yet, such implicit regression approaches are typically data-hungry, requiring massive 3D training datasets and heavily parameterized networks. Moreover, they often struggle to capture fine-grained motion details, leading to suboptimal tracking accuracy. Conversely, state-of-the-art 3D tracking methods such as STV2 [87], and 2D optical flow methods like RAFT [72], have demonstrated that constructing correlation

volumes and iteratively updating the motion field yield significantly more precise results. Building on this insight, we take a step back to the effective and accurate correlationbased estimation scheme to incorporate a novel correlationbased iterative 3D scene flow estimation strategy specifically tailored for dense pixel-level prediction within our feedforward framework.

Our proposed iterative correlation scheme differs from all existing correlation-based methods to address the specific challenges of dense 3D tracking.

- (1) Sparse-to-Dense. To manage the computational overhead of all-pixel scene flow estimation, we avoid performing iterative correlation updates on the full original resolution. Instead, we operate on a set of sparse anchor points and subsequently recover the dense motion for the entire image via learned upsampling.
- (2) 2D-to-3D Correlation. We circumvent the expensive 3D spatial correlation (which typically requires k-nearest neighbor searches in 3D space followed by cross-attention operations) used in prior works [87, 94]. We introduce a novel hybrid correlation mechanism that efficiently fuses the 3D geometric feature embeddings from the ViT backbone, utilizing 2D pixel-wise correlations. This design allows us to compute 3D flow updates rapidly without explicitly constructing heavy 3D spatial correlation.
- (3) 2D-3D Joint Supervision. Our 2D-to-3D correlation mechanism inherently supports the dual prediction of 2D and 3D flows. This structural alignment enables a joint supervision strategy that leverages abundant 2D flow datasets to provide auxiliary training signals for the 3D scene flow task. Consequently, we effectively circumvent the severe scarcity of 3D ground-truth annotations, utilizing 2D flow training to significantly enhance the generalization capability of our model.
- (4) Global Scene Flow. Unlike traditional scene flow estimation methods that are limited to adjacent frames, our framework is designed to estimate flows between arbitrary frame pairs within the video sequence, not limiting neighboring frames. By processing the entire video sequence simultaneously, our network leverages global temporal context to resolve local ambiguities, compensating for estimation errors that might occur in isolated frame pairs.

After estimating the accurate pairwise 2D-3D flows for the whole video, Track4World fuses the 3D trajectories in the global coordinate system and then constructs a holistic 3D tracking of every pixel in the world-centric coordinate system. Experiment results demonstrate that our approach delivers accurate, flexible, and comprehensive 3D motion estimates. Extensive experiments demonstrate that Track4World enables robust dense 3D tracking in the world coordinate system, consistently outperforming existing baselines [48, 87, 97].

### 2. Related Work

#### 2.1. Video Geometry Estimation

Video geometry estimation aims to generate temporally consistent point maps from videos. Early works focus on optimization-based refinement, e.g., RobustCVD [40] and CasualSAM [98], which optimize depth and camera parameters using geometric constraints. Recent differentiable SLAM methods, such as MegaSaM [47], Uni4D [92], and ViPE [27], incorporate depth priors and correspondence maps for video-level point map prediction. A second line of research explores data-driven approaches [10, 11, 30, 43, 45, 54, 76, 79, 82, 88, 91, 96, 106], leveraging world-centric geometry for robust, consistent estimation. However, VGGT [76] shows that camera-centric depth and pose often outperform world-centric representations. Consequently, recent methods, including GeometryCrafter [89], Pi3 [85], and MapAnything [37], focus on camera-centric predictions. Our approach follows this trend by employing a camera-centric representation (camera-centric point clouds and camera poses), from which world-centric reconstruction can be directly derived, while jointly estimating geometry and motion.

#### 2.2. Joint Geometry and Motion Estimation

Joint geometry and motion estimation aims to predict both 3D point clouds and motion trajectories from videos. This is inherently more challenging than geometry estimation alone due to the need for consistent spatial-temporal reasoning. Existing approaches generally follow three main directions. First, inspired by 2D point tracking [15, 25, 34, 35, 44, 63], early methods such as SpatialTracker [86] and DELTA [59] represent points in (u,v,d) coordinates, which require known camera intrinsics and can lead to geometric inaccuracies. Later works, e.g., TAPIP3D [94] and SpatialTrackerV2 [87], track directly in (x,y,z) space for improved stability. They rely on explicit 3D spatial correlations, which typically require performing computationally expensive k-nearest neighbor searches in 3D space for each query point, followed by cross-attention with the target points. Second, methods like St4RTrack [18], POMATO [97], Stereo4D [31], and ZeroMSF [48] reconstruct motion by estimating pairwise point maps or scene flow [67, 73, 90] between two frames via an implicit regression approach. Third, optimization-based methods such as TrackingWorld [55] combine multiple cues to recover dense 3D tracking for all pixels, but these are computationally expensive and suboptimal for generalization.

Motivated by these methods, we propose a feed-forward, holistic framework for the dense 3D tracking of every pixel. By leveraging a novel 3D correlation scheme, we simultaneously estimate pixel-wise 2D and 3D dense flow between arbitrary frame pairs. The estimated motion, along with the

reconstructed 3D geometry, enables the subsequent efficient computation of continuous 3D trajectories for every pixel within the global coordinate system.

Concurrent works. Several concurrent works have explored similar directions in joint geometry and motion estimation [2, 36, 52, 56, 69, 95, 105]. Among these, TraceAnything [52], Any4D [36], 4RC [56], DePT3R [2], D4RT [95] and V-DPM [69] also tackle the prediction of motion between timesteps. Track4World introduces two key design choices that distinguish it from these methods. First, we utilize explicit feature correlations to enhance motion prediction accuracy. Second, our novel 2Dlifted correlation architecture alleviates the computational bottleneck inherent in explicit 3D correlation, improving efficiency while enabling the 3D tracking module to benefit from abundant 2D training data.

### 3. Method

#### 3.1. Overview

Given a video sequence of T frames, {Ii ∈ RH×W×3 | i = 1,...,T}, Track4World aims to construct a holistic 3D track for every pixel within a world-centric coordinate system (as illustrated in Fig. 2). To achieve this, rather than treating 3D tracking as an isolated spatial problem, our framework operates through a logical pipeline. First, we extract global scene representations, including geometric features, camera-centric point clouds, and camera poses, using a finetuned geometry encoder initialized from recent state-of-the-art feedforward 3D reconstruction models [49, 80, 85], such as Pi3 or Depth Anything v3 (DA3), (see supplementary material for details). Second, built upon these global representations, our core scene flow decoder predicts dense 3D scene flow between arbitrary pairs of source and target timesteps (i,j). To reduce computational overhead, this decoder operates in a sparse-to-dense manner. Crucially, it introduces a novel 2D-to-3D correlation module that completely avoids expensive 3D spatial correlation, which in turn enables a 2D-3D joint supervision strategy to circumvent the severe scarcity of 3D groundtruth annotations. Finally, Track4World fuses these pairwise 3D scene flows to formulate the ultimate holistic 3D tracks.

#### 3.2. Scene Flow Decoder

Building upon the global scene representations (cameracentric pixel-aligned point clouds Pi, camera poses Ti, and geometry features Fi) extracted by the ViT backbone [49, 80, 85], the scene flow decoder predicts dense 3D scene flow between arbitrary frame pairs (i,j) for the whole video, allowing subsequent 3D tracking in the world-centric coordinate system.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Geometry Feature Context Feature

2D-to-3D Correlation Module

Short-Term Dense Pair

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

###### DenseFlowEstimation

3DFoundationModel

[Figure 15]

ImageEncoder

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

2D-3D Joint Supervision

Long-Term Sparse Pair

(a) Input Video Frames (b) Global Scene Representation (c) On-Demand Arbitrary-Pair Motion

###### (d) World-Centri 3D Dense Tracking

- Figure 2. Overview. Given (a) the input video frames, Track4World first extracts (b) global scene representations (geometric embeddings, point clouds, and camera poses). (c) A sparse-to-dense scene flow decoder then predicts 2D-3D joint flows between arbitrary timesteps, which applies a novel 2D-to-3D correlation scheme to improve efficiency and allows 2D-3D joint supervision. (d) The pairwise flows are ultimately fused to establish holistic world-centric 3D tracking.

[Figure 20]

3D Flow Head

[Figure 21]

[Figure 22]

[Figure 23]

3D Flow Head

[Figure 24]

[Figure 25]

GRU Update

[Figure 26]

[Figure 27]

Joint Supervision Boost

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Our Proposed 2D-to-3D Correlation (Efficient & Boosted by 2D Data)

2D Flow

Target 3D Features

3D KNN Search Source Tokens Target Tokens

Cross Attention

3D Supervision

Previous Dense 3D Correlation (Expensive)

2D Flow Sampled 3D Features

- Figure 3. Comparison of correlation mechanisms. Prior methods rely on explicit k-nearest neighbor searches and crossattention in 3D space, leading to high computational costs. In contrast, our proposed method anchors 3D updates directly to intermediate image-plane correlations. This design significantly improves computational efficiency and allows the 3D tracking module to be effectively boosted by abundant 2D training data.

tracking by the correlation scheme but consume lots of computation in finding k-nearest point in 3D space for correlation as shown in Fig. 3 (top), making these methods limited to tracking only sparse 3D points. To scale this for dense 3D flow estimation, we introduce an efficient 2D-to-3D correlation module that elegantly fuses geometric features with image-plane correlations and lifts it for 3D flow estimation.

In the proposed 2D-3D correlation module, both 2D and 3D flows are estimated with several update iterations to compute the correlations and refine the target positions. In each iteration, the 2D flow and 3D flow are updated sequentially in a coupled manner, as shown in Fig. 3 (bottom). For a given source-target image pair (i,j), the 2D flow is updated first, serving as a reliable spatial basis for the subse-

quent update of the 3D flow. Both 2D flows M(0)2d and 3D flows M(0)3d are initialized to 0 everywhere and updated with the following scheme.

2D Iterative Correlation. We first construct geometric feature correlation volumes Cˆ(i,jt) and semantic feature correlation volumes C˜(i,jt) between the source position on the source image and the neighborhood positions on the target image using the current 2D flow M(2td). At the t-th iteration, a GRU-based operator updates the hidden context feature from Fˆ(it) to Fˆ(it+1), which in turn drives the update of the 2D flow M(2td) and visibility confidence V(t):

##### 3.2.1. Anchor Feature Extraction

To manage the computational overhead of all-pixel flow estimation, we avoid performing iterative correlation updates at the full original resolution. Instead, we operate on a set of sparse anchor points. We begin with augmenting the geometry tokens Fi with temporal context using global self-attentions to yield refined features Fˆi. Concurrently, a lightweight context encoder extracts semantic features F˜i directly from the input images. After that, we downsample the point clouds Pi and both feature maps (Fˆi,F˜i) to 1/8 resolution, establishing a sparse but contextually rich foundation for the flow estimation, which will be upsampled back to the full resolution flow in the end.

Fˆ(it+1) = GRU F ˆ(it), F˜i, Cˆ(i,jt), C˜(i,jt), M(2td), V(t) , (1) M(2td+1) = M(2td) + MLP F ˆ(it+1) , (2)

V(t+1) = V(t) + MLP F ˆ(it+1) . (3) Then, the 2D flow update is used as a basis for 3D flow update as follows.

##### 3.2.2. 2D-to-3D Correlation for Flow Estimation

##### Lifting for 3D Flow Iterative Estimation. With the 2D

flow M(2td) and M(2td+1), we can retrieve the corresponding 2D target positions on timestep t and t + 1. Then, by in-

This section aims to estimate the 3D scene flow on the anchor points using correlation. Previous methods, like STV2 [87] or TAPIP3D [94], demonstrate accurate 3D

terpolating on the extracted global point maps Pj, we can

compute 3D coordinates p(jt) and p(jt+1) for these two target positions correspondingly. Then, p(jt+1)−p(jt) can be an initial estimation of the 3D flow update ∆M(3td). However, this only lifts the 2D flow update to 3D without considering any additional 3D geometric correlation. We further fuse all 3D information together using a 3D flow head H3d to predict the 3D flow update ∆M(3td) by:

∆M(3td) = H3d p(jt),p(jt+1),F(jt),F(jt+1)

###### ,

Lifted Target Samples

Fˆ(it+1),Fi Lifted Source Contexts

,C(3td,i,j) ,M(3td)

Auxiliary Priors

,

(4)

where F(jt) and F(jt+1) denote the interpolated geometric features of Fj at the predicted target positions in two consecutive iterations, providing the 3D shape information

of these two predicted target positions, F(it+1),Fi are the source geometric features on the anchor points. We fur-

ther compute the 3D spatial similarity by correlating the source point cloud warped by the estimated 3D flow M(3td) with the target point cloud. The resulting correlation C(3td,i,j) between the transformed source points and the target geometry provides informative cues for refining the 3D flow. Additionally, we leverage the historical 3D flow M(3td) to construct a trajectory prior M(3td), which constrains the smoothness of the flow update. Then, the 3D flows are updated with M(3td+1) ← M(3td) +∆M(3td). This 2D-to-3D correlation scheme allows us to accurately and iteratively update both

- 2D and 3D flows simultaneously while keeping the computation complexity manageable even for all anchor points from all frames.

##### 3.2.3. Dense Scene Flow Recovery

In the final step, we recover the dense scene flow for the entire image. The low-resolution (1/8) flow maps are upsampled to full resolution via a learned pixel-shuffle operation, guided by decoded contextual weights. To maximize the utility of our 2D-lifted architecture, we introduce a hybrid unprojection scheme. Because the intermediate 2D flow captures image-plane dynamics with high precision, we directly combine it with the Z-axis displacement from the 3D scene flow. Using camera intrinsics derived from the predicted point clouds, we project this combined motion into (x,y,z) space to produce the final refined 3D scene flow. The benefits of this hybrid design are analyzed in Tab. 7.

#### 3.3. Discussion

Computational Efficiency. Prior works [87, 94] heavily rely on explicit 3D spatial correlations, which typically require a computationally expensive k-nearest neighbor (kNN) search in 3D space for each query point, followed

by a cross-attention mechanism with the retrieved target points. In contrast, our 3D flow estimation is fundamentally driven by the proposed 2D-lifted correlation module. This design completely bypasses both the 3D k-NN search and the heavy cross-attention operations, significantly alleviating the computational burden. Specifically, while traditional 3D point-based matching incurs a theoretical complexity of at least O(N log N +N ·k) (or O(N2) for dense global attention) for N points, our warp-sampling mechanism operates with a strict O(N) complexity via direct coordinate-based lookups from the 2D flow.

2D-3D Joint Supervision. This structural alignment naturally supports dual supervision. Unlike prior arts [87, 94] that define correlations strictly in 3D space, our architecture anchors the 3D updates directly to image-plane correlations. This allows us to apply auxiliary training signals from abundant 2D datasets directly to the intermediate 2D motion priors. By leveraging massive 2D data to guide the lifting process, we effectively circumvent the severe scarcity of 3D ground-truth annotations and significantly enhance the generalization capability of our model.

Global Trajectory Inference. Leveraging the predicted arbitrary-pair dense flows, our framework seamlessly generalizes to global 3D tracking. For long-range point tracking, we infer flows from a reference frame to all subsequent frames. This process is refined by a temporal aggregator that applies attention over concatenated flow features to enforce temporal consistency. Conversely, for dense tracking of every pixel, we compute and chain consecutive frameto-frame flows, yielding continuous 3D trajectories in the global coordinate system.

Short-to-Long-Term Supervision. During training, we implement a variable-stride sampling strategy: the model is simultaneously supervised by dense flow ground truth over varying temporal intervals and sparse trajectory annotations across the entire sequence. By optimizing these short- and long-term objectives jointly, the network learns to reconcile local motion precision with global trajectory consistency, constructing a holistic 3D track for every pixel.

### 4. Experiments

Implementation details. Notably, we employ tailored training strategies to fine-tune the base geometric model, leading to refined point map and camera pose estimations as well. Unless specified, all reported results use DA3 [49] as the default backbone initialization. Implementation details about specific fine-tuning strategies are provided in the supplementary materials. Experiment overview. To comprehensively evaluate our framework, we conduct experiments across a variety of tasks, including 2D/3D flow estimation, 2D/3D tracking, point map prediction, and camera pose estimation. Furthermore, we present qualitative results on diverse in-the-wild videos and conduct extensive abla-

|In domain| |
|---|---|
|Kubric-3D val [22] (short) Abs Rel ↓ δ < 1.25 ↑ EPE3D↓ AccS ↑ AccR ↑ EPE2D↓ AccS2D↑ AccR2D↑<br><br>|Kubric-3D val (long) Abs Rel ↓ δ < 1.25 ↑ EPE3D↓ AccS ↑ AccR ↑ EPE2D↓ AccS2D↑ AccR2D↑|

Method

RAFT [72] / / / / / 6.7974 0.7442 0.9018 / / / / / 51.9034 0.5183 0.7042 GMFlowNet [100] / / / / / 7.0390 0.7619 0.9103 / / / / / 51.9049 0.5309 0.7201 SEA-RAFT [84] / / / / / 9.5794 0.7720 0.8947 / / / / / 58.7148 0.5596 0.6825 RAFT-3D [73] 0.0649 0.9344 0.6170 0.0015 0.0078 40.4480 0.0002 0.0015 0.1245 0.8422 1.5652 0.0001 0.0010 83.6966 0.0004 0.0040 OpticalExpansion [90] 0.2170 0.6266 0.2093 0.2890 0.4760 19.6471 0.1183 0.3316 0.2177 0.6316 0.7037 0.1062 0.1903 68.6562 0.0255 0.0874 POMATO [97] 0.1525 0.8329 0.9672 0.0566 0.1696 / / / 0.1761 0.7760 1.6925 0.0148 0.0564 / / / ZeroMSF [48] 0.0860 0.9196 0.3528 0.1867 0.3413 / / / 0.1208 0.8609 1.2182 0.0475 0.0895 / / / Any4D [36] 0.0585 0.9547 0.3908 0.1610 0.2893 / / / 0.1017 0.8770 1.2442 0.0429 0.0855 / / / V-DPM [69] 0.0716 0.9010 0.4087 0.1442 0.2491 / / / 0.1155 0.8205 1.2620 0.0407 0.0803 / / /

Ours 0.0344 0.9719 0.1537 0.5494 0.7460 1.8685 0.8086 0.9309 0.0472 0.9371 0.4808 0.3247 0.5491 15.0906 0.6134 0.7711

Out of domain KITTI [20] BlinkVision [46]

Method

Abs Rel ↓ δ < 1.25 ↑ EPE3D ↓ AccS ↑ AccR ↑ EPE2D ↓ AccS2D ↑ AccR2D ↑ Abs Rel ↓ δ < 1.25 ↑ EPE3D ↓ AccS ↑ AccR ↑ EPE2D ↓ AccS2D ↑ AccR2D ↑

RAFT [72] / / / / / 5.4150 0.6271 0.8068 / / / / / 14.1255 0.5037 0.6953 GMFlowNet [100] / / / / / 4.6977 0.6432 0.8241 / / / / / 12.0176 0.5281 0.7170 SEA-RAFT [84] / / / / / 4.8863 0.6654 0.8297 / / / / / 20.9160 0.5697 0.7186 RAFT-3D [73] 0.1619 0.8413 0.3837 0.0118 0.0678 54.0938 0.0001 0.0007 0.1426 0.8455 0.6690 0.0454 0.1280 85.4975 0.0018 0.0121 OpticalExpansion [90] 0.2764 0.4302 0.2419 0.1553 0.2612 8.8808 0.5446 0.7326 0.3372 0.4099 0.4406 0.2091 0.3116 20.2384 0.4122 0.6139 POMATO [97] 0.2752 0.4359 0.2602 0.1127 0.2156 / / / 0.2089 0.6569 0.4038 0.1522 0.2870 / / / ZeroMSF [48] 0.2064 0.5913 0.1823 0.1695 0.3481 / / / 0.1934 0.6620 0.3937 0.1913 0.2991 / / / Any4D [36] 0.2398 0.4974 0.1856 0.1429 0.2931 / / / 0.2218 0.6125 0.9238 0.1242 0.1818 / / / V-DPM [69] 0.1469 0.7981 0.4462 0.1180 0.1608 / / / 0.2117 0.6449 1.1476 0.1079 0.1547 / / /

Ours 0.0707 0.9570 0.0742 0.6929 0.8238 2.5722 0.6849 0.8769 0.0371 0.9768 0.1135 0.5091 0.7144 7.5632 0.5131 0.7424

- Table 1. Evaluation on scene and optical flow estimation. We evaluate our model on one in-domain dataset: Kubirc-3D val [22] and two out-of-domain datasets: KITTI [20], BlinkVision [46]. Best results are highlighted in darker blue , and second best in lighter blue .

tion studies to validate the effectiveness of each proposed module.

#### 4.1. Scene and Optical Flow Estimation

Evaluation datasets. We evaluate our model on one indomain dataset, Kubric-3D val [22], and two out-of-domain datasets, KITTI [20] and BlinkVision [46]. All three datasets provide ground-truth optical flow and scene flow annotations. For the Kubric-3D dataset, we sample two evaluation settings: a short-range setting where the source and reference frames are 4 frames apart, and a long-range setting where they are 16 frames apart, to assess flow estimation performance under different temporal gaps. For the BlinkVision [46] dataset, the source–target frame pair is randomly sampled.

Evaluation metrics. For scene flow estimation, we follow previous works [23, 48, 51, 73] and report the standard metrics: End Point Error (EPE3D), Accuracy Strict (AccS), and Accuracy Relax (AccR). Additionally, we evaluate the quality of the previous-frame point cloud and the transformed point cloud obtained by applying the predicted scene flow. We adopt the absolute relative error (Abs Rel) and the percentage of inlier points (δ < 1.25) for this assessment. To ensure a fair comparison, both predicted point clouds are aligned to the ground truth by optimizing a shared scale factor and translation. This scale factor is further applied to align the predicted scene flow before computing EPE3D, AccS, and AccR. This alignment procedure follows the method described in MoGe [80]. For optical flow estimation, we follow [72, 84, 100] to report 3 metrics, EPE2D, AccS2D, AccR2D.

Comparison with existing methods. We compare our method against several representative approaches: three optical flow methods (RAFT [72], GMFlowNet [100], and SEA-RAFT [84]), two scene flow methods (RAFT-3D [73] and OpticalExpansion [90]), two joint geometry and scene flow methods (POMATO [97] and ZeroMSF [48]), as well as concurrent works Any4D [36] and V-DPM [69]. For RAFT-3D, we provide the depth and camera parameters predicted by VGGT [76] as input, while OpticalExpansion uses the point clouds predicted by POMATO. Tab. 1 presents a quantitative comparison, demonstrating that our method consistently outperforms these prior works across all four datasets. This validates the effectiveness of the proposed 2D-3D correlation for the flow estimation task.

#### 4.2. 3D Tracking Estimation

Evaluation datasets. Following prior works [59, 86, 87, 97], we evaluate our method on ADT [61], PStudio [32], and DriveTrack [3] benchmarks from the TAPVid-3D [41] dataset, as well as the validation set of the PointOdyssey [102] dataset. Consistent with POMATO [97], we reformulate the datasets by projecting all query points within a temporal window to the first frame. The window sizes are 16 and 50 in the evaluation.

Evaluation metrics. We use the Average Percent Deviation (APD) metric, which measures the percentage of points whose predicted positions fall within a specified threshold relative to the ground-truth depth. APD provides a direct and quantitative assessment of the tracking accuracy.

Comparison with existing methods. We compare our method with representative 3D tracking and geometry/scene

PointOdyssey [102] ADT [61] PStudio [32] DriveTrack [3] Avg.

Method

L-16 L-50 L-16 L-50 L-16 L-50 L-16 L-50 L-16 L-50 Camera coordinate 3D tracking

SpatialTracker∗ [86] 0.3116 0.2977 0.4962 0.4692 0.5390 0.4991 0.2529 0.2502 0.3999 0.3791 DELTA∗ [59] 0.3529 0.3412 0.5116 0.4952 0.5922 0.5533 0.2704 0.2701 0.4317 0.4150 STV2† [87] 0.1864 0.1785 0.2400 0.2330 0.3784 0.3690 0.1711 0.1725 0.2400 0.2383

MASt3R [43] 0.3546 0.3253 0.3368 0.3029 0.3293 0.2956 0.2767 0.2559 0.3244 0.2949 MonST3R [96] 0.3912 0.3860 0.3694 0.3429 0.3511 0.3381 0.3056 0.2787 0.3543 0.3364 POMATO [97] 0.4816 0.4623 0.5338 0.5299 0.5163 0.4726 0.4237 0.4329 0.4888 0.4744 ZeroMSF [48] 0.4214 0.3887 0.5382 0.4635 0.5083 0.4524 0.4448 0.4513 0.4782 0.4390 Ours 0.5397 0.5268 0.6501 0.6091 0.5948 0.5423 0.5003 0.5092 0.5712 0.5469

World coordinate 3D tracking

STV2† [87] 0.1925 0.1763 0.2456 0.2163 0.3790 0.3689 0.1711 0.1725 0.2470 0.2335 POMATO‡ [97] 0.4425 0.3905 0.3611 0.3548 0.5166 0.4713 0.4227 0.4210 0.4357 0.4094 ZeroMSF‡ [48] 0.4053 0.3505 0.4530 0.3563 0.4828 0.4386 0.4474 0.4382 0.4471 0.3959 Any4D [36] 0.4769 0.4174 0.4460 0.3717 0.5707 0.5066 0.5235 0.5079 0.5043 0.4509 V-DPM [69] 0.4848 0.4233 0.4783 0.3759 0.6084 0.5795 0.4854 0.4817 0.5142 0.4668

- Ours 0.5345 0.5162 0.6250 0.5622 0.5946 0.5422 0.5003 0.5087 0.5636 0.5323

- Table 2. 3D tracking estimation. We report the APD metric to evaluate 3D point tracking on the PointOdyssey [102], ADT [61], PStudio [32], and DriveTrack [3] datasets. L-16 and L-50 indicate tracking within the temporal length of 16 and 50 frames, respectively.

* indicates that ground-truth camera intrinsics are used. † indicates that bundle adjustment is not used. ‡ denotes methods using camera poses estimated by VGGT [76].

flow approaches. The baselines include 3D trackers SpatialTracker [86] and DELTA [59] (both using ground-truth intrinsics), as well as STV2 [87]. We further compare against geometry and scene flow methods MASt3R [43], MonST3R [96], POMATO [97], and ZeroMSF [48], along with concurrent works Any4D [36] and V-DPM [69]. We evaluate 3D tracking results in Tab. 2, following the approach described in Sec. 3.3 to achieve dense tracking and perform sampling based on the UV coordinates of the query points. The results demonstrate that our method consistently outperforms most existing 3D tracking and pairwise scene flow approaches across multiple datasets, in both camera-centric and world-centric coordinate systems, highlighting its effectiveness and generalization.

#### 4.3. 2D Tracking Estimation

Evaluation datasets. Following prior works [16, 34, 35], we evaluate our method on three datasets: Kinetics [8], RoboTAP [75], and RGB-Stacking [42]. Kinetics comprises 1,144 YouTube videos from the Kinetics-700–2020 validation set, featuring complex camera motion and cluttered backgrounds, with an average of 26 tracks per video. RoboTAP consists of 265 real-world robotic manipulation videos, with an average duration of 272 frames per video. RGB-Stacking is a synthetic dataset of robotic videos, characterized by numerous texture-less regions that make tracking particularly challenging.

Evaluation metrics. We evaluate tracking performance us-

Kinetics [8] RoboTAP [75] RGB-S [42]

Method

AJ↑ δvisavg↑ OA↑ AJ↑ δvisavg↑ OA↑ AJ↑ δvisavg↑ OA↑ PIPs++ [102] / 63.5 / / 63.0 / / 58.5 / TAPIR [15] 49.6 64.2 85.0 59.6 73.4 87.0 55.5 69.7 88.0 CoTracker [35] 49.6 64.3 83.3 58.6 70.6 87.0 67.4 78.9 85.2 TAPTR [44] 49.0 64.4 85.2 60.1 75.3 86.9 60.8 76.2 87.0 LocoTrack [12] 52.9 66.8 85.3 62.3 76.2 87.1 69.7 83.2 89.5 BootsTAPIR [16] 54.6 68.4 86.5 64.9 80.1 86.3 70.8 83.0 89.9 CoTracker3 [34] 55.8 68.5 88.3 66.4 78.8 90.8 71.7 83.6 91.1

Ours 59.1 71.3 90.6 70.9 81.8 93.3 78.2 88.5 92.3

Table 3. 2D tracking estimation. We evaluate our model on three datasets: Kinetics [8], RoboTAP [75], and RGB-Stacking [42].

ing the TAP-Vid metrics [14]. Specifically, we report Occlusion Accuracy (OA), measuring occlusion prediction as a binary classification; δvisavg, the fraction of visible points tracked within 1, 2, 4, 8, and 16 pixels, averaged across thresholds; and Average Jaccard (AJ), which jointly assesses geometric and occlusion prediction accuracy.

Comparison with existing methods. We compare our method with representative 2D tracking approaches, including CoTracker3 [34], LocoTrack [12], and other state-ofthe-art methods. In Tab. 3, we present the 2D tracking results from our 2D branch. Leveraging valuable geometric cues and joint training on multi-modal, multi-source data, our method achieves performance comparable to existing state-of-the-art 2D tracking approaches.

GMU Kitchen [21] Monkaa [57] Sintel [6] Scannet test [13] Kubric-3D val [22] KITTI [20] Tum [68] Avg. Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑

Method Params

MoGe [80] 314M 0.1728 0.6725 0.2069 0.6317 0.2181 0.6615 0.1194 0.8447 0.0852 0.9294 0.0801 0.9374 0.1563 0.7663 0.1484 0.7776 VGGT [76] 1.26B 0.2530 0.4454 0.2009 0.6678 0.2004 0.7303 0.0763 0.9242 0.0316 0.9733 0.1245 0.8517 0.1278 0.8236 0.1449 0.7738 MoGe-2 [81] 331M 0.0654 0.9391 0.1904 0.6797 0.2058 0.6903 0.0626 0.9673 0.1184 0.8592 0.0776 0.9627 0.1156 0.8694 0.1194 0.8525 MapAnything [37] 563M 0.1090 0.9279 0.2012 0.7379 0.2059 0.7141 0.0463 0.9833 0.0684 0.9475 0.1016 0.9144 0.1132 0.9458 0.1208 0.8816 Pi3 [85] 1.29B 0.0458 0.9338 0.0774 0.9256 0.1489 0.7899 0.0750 0.9532 0.0337 0.9817 0.0866 0.8877 0.0493 0.9680 0.0738 0.9200 DA3 [49] 1.36B 0.0551 0.9343 0.1131 0.8761 0.1575 0.8064 0.0222 0.9917 0.0431 0.9599 0.0404 0.9861 0.0886 0.9515 0.0743 0.9294

- Ours 1.38B 0.0431 0.9341 0.0853 0.9123 0.1261 0.8291 0.0288 0.9872 0.0191 0.9939 0.0268 0.9877 0.0572 0.9638 0.0552 0.9440

Table 4. Evaluation on point map estimation. Results are aligned with the ground truth by optimizing a shared scale factor and shift across the entire video.

#### 4.4. Point Map Estimation

Evaluation datasets. We evaluate our model on seven datasets, covering both real-world and synthetic scenarios. For indoor real-world datasets, we use GMU Kitchen [21], ScanNet test set [13], and TUM [68]. For outdoor realworld data, we use KITTI [20]. For synthetic datasets, we evaluate on Monkaa [57], Sintel [6], and Kubric-3D validation set [22].

Evaluation metrics. Following prior works [26, 54, 96], we first align the estimated point/depth maps with the ground truth using a shared scale and shift before computing the metrics. This alignment procedure follows the approach described in MoGe [80]. We primarily report two metrics: the absolute relative error (Abs Rel) and the percentage of inlier points within a threshold δ < 1.25.

Comparison with existing methods. We compare our method with several representative approaches, including MoGe [80], VGGT [76], MapAnything [37], Pi3 [85], and DA3 [49]. Tab. 4 presents a quantitative comparison against these methods. As shown, our method achieves highly competitive geometry estimation performance due to our tailored training strategies, providing a solid foundation for

- 3D flow estimation.

#### 4.5. Camera Pose Estimation

Evaluation datasets. We evaluate camera pose estimation performance on two widely used benchmarks: Sintel [6] and Bonn [60]. For the Sintel dataset, following prior works [9, 96, 101], we exclude sequences that are static or contain only straight motions, resulting in 14 dynamic sequences for evaluation.

Evaluation metrics. Consistent with existing works [9, 74, 96, 101], we report three standard metrics: ATE ↓ (Absolute Translation Error), RTE ↓ (Relative Translation Error), and RRE ↓ (Relative Rotation Error).

Comparison with existing methods. We compare our approach with several joint depth and pose estimation methods, including Align3R [54], CUT3R [79], VGGT [76], MapAnything [37], Pi3 [85], and DA3 [49], as well as joint depth, pose, and motion estimation methods, such as POMATO [97] and STV2 [87]. Tab. 5 reports the camera pose estimation results. The results demonstrate the effectiveness of our method, particularly among approaches that jointly estimate depth, pose, and motion.

Sintel [6] Bonn [60] ATE↓ RTE↓ RRE↓ ATE↓ RTE↓ RRE↓

Method

Align3R [54] 0.128 0.042 0.432 0.023 0.007 0.620 CUT3R [79] 0.217 0.070 0.636 0.035 0.014 1.212 VGGT [76] 0.167 0.062 0.490 0.051 0.011 1.038 MapAnything [37] 0.227 0.111 2.047 0.026 0.014 0.668 Pi3 [85] 0.088 0.043 0.299 0.012 0.011 0.612 DA3 [49] 0.124 0.061 0.331 0.010 0.011 0.638

POMATO [71] 0.209 0.064 0.694 0.041 0.017 0.832 STV2 [87] 0.133 0.057 0.641 0.019 0.015 0.701 Ours 0.119 0.054 0.309 0.009 0.009 0.604

Table 5. Camera pose estimation. We evaluate our model on two datasets: Sintel [6] and Bonn [60]. ‡ indicates using ground-truth camera intrinsics as input.

#### 4.6. Ablation Study

Effect of different 3D backbone ViT models. Tab. 6 reports the average results obtained using different 3D backbone ViT models as the global geometry encoder initialization. The results demonstrate the flexibility of our framework and show that it remains effective across different backbone choices.

Point Map Scene Flow 3D Tracking

Method Abs Rel ↓ δ < 1.25 ↑ EPE3D ↓ Abs Rel ↓ δ < 1.25 ↑ L-16 ↑ L-50 ↑ Ours (MoGe [80]) 0.0973 0.8921 0.3180 0.0680 0.9479 0.5447 0.5135 Ours (Pi3 [85]) 0.0492 0.9537 0.2569 0.0548 0.9645 0.5734 0.5424 Ours (DA3 [49]) 0.0552 0.9440 0.2056 0.0474 0.9607 0.5712 0.5469

Table 6. Effect of different 3D foundation models.

Ablation study on the scene flow decoder. Tab. 7 validates our key design choices. (1) 2D Supervision: Training solely on 3D datasets (“w/o 2D Supervision”) causes a severe performance collapse (EPE3D 0.6511), underscoring its necessity for guiding 3D flow. (3) Target Lifting: Removing lifted target samples (“w/o Target lifting”) prevents the network from dynamically querying the target space via 2D displacements. The resulting performance drop confirms that mapping 2D matches to 3D residuals is essential for spatial refinement. (4) Iterative Updates: Replacing the iterative refinement mechanism with a direct, singlestep regression (“w/o iterations”) leads to a performance decline. This highlights that iteratively updating the flow field is crucial for achieving highly precise results. (5) Aux-

iliary Priors: Ablating the auxiliary priors (“w/o C(3td,i,j) & M(3td)”) decreases accuracy, verifying their indispensability for global spatial alignment and kinematic smoothness

Method Time (s) ↓ Mem. (GB) ↓ Parm. (M) ↓ POMATO [97] (Dense) 4.8 16 133.64 ZeroMSF [48] (Dense) 8.2 10 153.84 STV2 [87] (Sparse) 5.8 19 65.99 STV2 [87] (Dense) OOM OOM 65.99 Ours w/o 2D-to-3D (Dense) OOM OOM 56.90 Ours (Dense) 3.4 14 26.06

Setting Abs Rel ↓ δ < 1.25 ↑ EPE3D ↓ Supervision w/o 2D Supervision 0.1021 0.9199 0.6511 Decoder Components

w/o Target lifting 0.0534 0.9543 0.3017 w/o iterations 0.0506 0.9559 0.2356

Table 8. Efficiency comparison on 16-frame ADT [61] sequences. Gray rows indicate Out-of-Memory (OOM) failures under dense tracking settings.

w/o C(3td,i,j) & M(3td) 0.0482 0.9591 0.2201 Hybrid Formulation

2D flow + d 0.1304 0.8911 0.8210 Pure 3D flow 0.0552 0.9570 0.2815

T0 T1 T2

Full (Ours) 0.0474 0.9607 0.2056

[Figure 32]

[Figure 33]

[Figure 34]

- First-frame2D

densetracking

irstrae

ensetrain

- First-frame3D

ensetrain

irstrae

Table 7. Ablation study on the scene flow decoder. The results are averaged over all datasets. Best and second best results are highlighted.

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Sceneflowee

across iterations. (4) Hybrid Formulation (introduced in Sec 3.2.3): Our full model (EPE3D 0.2056) significantly outperforms both heuristically lifting 2D flow with depth (“2D flow + d”, 0.8210) and purely regressing 3D flow (“Pure 3D flow”, 0.2854). This proves that explicitly intertwining 2D flow with Z-axis displacement is superior to relying on isolated domains.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

densetracking raieveryie f

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Efficiency of scene flow decoder. Tab. 8 compares the inference time (Time), memory consumption (Mem.), and parameter count (Parm.) on 16-frame sequences from the ADT [61] dataset. First, pairwise scene flow approaches, such as POMATO [97] and ZeroMSF [48], natively support dense tracking by computing flow independently for every pixel. However, unlike our efficient sparse-to-dense formulation, their per-pixel prediction approach incurs significantly higher computational cost, resulting in increased latency without providing a clear advantage in memory efficiency. On the other hand, STV2 [87] relies on a traditional 3D spatial correlation mechanism for iterative trajectory estimation. Its high computational complexity and large parameter count impose a severe bottleneck. Consequently, STV2 is restricted to sparse tracking and suffers from Outof-Memory (OOM) errors when extended to dense tracking scenarios. To explicitly validate that this bottleneck lies in the traditional 3D computation, we replace our proposed

[Figure 50]

[Figure 51]

[Figure 52]

everyfrae Trackingeverypixelof

everyframe

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

WorldCoordinateSystem

enseraetoriesinte

ordoordinateste DenseTrajectoriesinthe

[Figure 58]

[Figure 59]

[Figure 60]

Figure 4. Qualitative results on diverse in-the-wild videos.

- 2D-to-3D correlation module with the aforementioned traditional 3D mechanism (Ours w/o 2D-to-3D). As shown in the gray rows of Tab. 8, this variant similarly crashes with an OOM error under dense tracking settings. Overall, by bypassing the prohibitive cost of explicit 3D correlations, our method achieves superior efficiency in runtime, memory footprint, and model size, all while natively supporting dense tracking and maintaining state-of-the-art performance.

first four illustrate camera-centric perspectives: (1) 1stframe 2D tracking, where consistent colors denote temporal pixel correspondences. (2) Scene flow, contrasting the original RGB point cloud with the flow-transformed geometry (rainbow). (3) 1st-frame 3D tracking, featuring uniformly downsampled camera-centric trajectories for rendering efficiency. (4) All-frame 3D tracking, capturing dense camera-centric trajectories of both existing and newly emerging objects. Distinct from the above, (5) Worldcentric tracking maps these trajectories to a global coordinate system. This effectively decouples camera ego-

#### 4.7. Qualitative Visualization

To demonstrate our method’s effectiveness, Fig. 4 visualizes the tracking and flow results across five aspects. The

motion from object dynamics, yielding spatially stable backgrounds and physically coherent absolute motions for dynamic entities. For dynamic visualizations, please refer to the supplementary video.

### 5. Conclusion

In this paper, we present Track4World, a feedforward foundation model designed for efficient, dense 3D tracking within a world-centric coordinate framework. Leveraging a global 3D scene representation parameterized by a VGGT-style Vision Transformer, our method introduces a novel 3D correlation mechanism to simultaneously estimate dense 2D and 3D scene flow between arbitrary pairs of frames. By tightly coupling this estimated scene flow with the reconstructed 3D geometry, Track4World facilitates highly efficient, pixel-wise 3D trajectory extraction across the entire video. Extensive evaluations across multiple benchmarks demonstrate the exceptional robustness, scalability, and generalization capabilities of our approach, marking a significant step toward holistic 4D reconstruction from in-the-wild monocular videos.

## Track4World: Feedforward World-centric Dense 3D Tracking of All Pixels Supplementary Material

### A. Overview

In this supplementary material, we provide comprehensive details regarding the implementation, training, and evaluation of our proposed method. The document is organized as follows:

- • Sec. B provides the specific implementation details, including training hyperparameters, dataset configurations, and optimization strategies.
- • Sec. D elaborates on the detailed network architecture of the flow estimation module.
- • Sec. E details the unified objective functions designed for joint flow estimation and point tracking.
- • Sec. F describes the optimization procedure and mathematical formulation used for camera pose estimation.
- • Sec. G defines the evaluation metrics used across geometry, flow, tracking, and camera pose tasks to ensure clarity and reproducibility.
- • Sec. H presents additional ablation studies to further validate our design choices.
- • Sec. I showcases extensive qualitative results visualizations on challenging in-the-wild sequences.
- • Sec. K discusses the constraints of the current approach and outlines promising directions for future research.

### B. Implementation Details

Our model is trained in two stages. In the first stage, we focus on geometry estimation, using a diverse set of datasets that provide high-quality depth and pose supervision, including Kubric-3D [22], GTASfM [77], V-KITTI [19], ARKitScenes [4], BlinkVision [46], DynamicStereo [33], TartanAir [83], MVSSynth [28], ScanNetv2 [13], ScanNet++ [93], Spring [58], OmniWorld [103], PointOdyssey [102], Hypersim [66], and IRS [78]. In the second stage, we freeze the geometry estimation module and train the motion estimation module. The training data covers a wide range of motion patterns, including optical flow datasets (AutoFlow [70], FlyingChairs [17], OmniWorld [103], HD1K [39], Spring [58], TartanAir [83], VIPER [65]), 2D point tracking datasets (Kubric [22], Monkaa [57], Driving [57]), scene flow datasets (Kubric-

- 3D [22], V-KITTI [19]), and 3D point tracking datasets (DynamicStereo [33], PointOdyssey [102]).

We train our model on eight 40GB GPUs. In the first stage, we adopt the AdamW [53] optimizer with a StepLR scheduler to train the geometry estimation backbone, using an initial learning rate of 1 × 10−4. This stage runs for 100,000 steps and takes approximately one week. In the second stage, we train the motion estimation module us-

ing the AdamW optimizer with a OneCycleLR scheduler, whose peak learning rate is set to 1 × 10−4. This stage is trained for another 100,000 steps and takes about five days.

### C. Finetuning Geometry Encoder

We employ a backbone-agnostic encoder to process video frames {Ii} into a global scene representation. Leveraging state-of-the-art models [49, 80, 85] as initialization, we fine-tune specific layers to enhance temporal consistency: adding global attention layers and camera pose tokens for monocular models like MoGe [80], or updating intermediate layers for 3D reconstruction models like Pi3 [85] and DA3 [49]. As shown in Tab. 6, we demonstrate the effectiveness of our proposed framework across different geometry backbones.

To address the inherent scale and focal-length ambiguity while ensuring temporal coherence, we compute a videolevel reconstruction loss. We estimate a global optimal scale

- s∗ and translation t∗ to align the predicted point clouds Pi with the ground truth Pˆi:

ℓrecons =

1 |T| i∈T

1 |Mi| j∈M

i

1 zi,j ∥s∗pi,j + t∗ − pˆi,j∥1 ,

(5) where zi,j denotes the ground-truth depth used for reweighting. Furthermore, we enforce global geometric consistency via an affine-invariant pairwise pose loss:

ℓpose =

1 N(N − 1) i̸=j

ℓgeo(Rij,Rˆ ij)

+ λtransℓhuber(˜tij,ˆtij) ,

- (6)

where predicted translations are aligned as ˜tij = s∗tij + t∗ − Rˆ ijt∗. To stabilize convergence, we impose a regularization ℓreg = max(0,Lmean − τ) on the mean point magnitude Lmean. For geometric fidelity, we further introduce a normal consistency loss ℓn and a multi-scale local geometry loss ℓlocal [80]. The overall training objective is:

ℓtotal = ℓrecons+λposeℓpose+λregℓreg+λnℓn+λlocalℓlocal.

- (7)

### D. Model Architecture Details

Recurrent update operator GRU. The update operator GRU refines the motion representations M(2td) and V(t) by integrating correlation cues, recurrent features, semantic context, and temporal motion variation. At iteration

(t), the operator takes as inputs: the 4D correlation volumes Cˆi,j(M(2td)) and C˜i,j(M(2td)), the motion-change gradient ∇(M(2td)−M(2td−1)), the current recurrent feature Fˆ(it), the semantic context feature F˜i, and the current visibility confidence map V(t). First, motion encoders ϕmot extracts correlation-based motion cues from the correlation volumes:

Fcorri,j = ϕmot1 C ˆi,j,∇ M(2td) − M(2td−1) , (8) Fctxi,j = ϕmot2 C ˜i,j,∇ M(2td) − M(2td−1) . (9)

Simultaneously, the recurrent feature and the context feature are projected into a shared latent space via respective functions ψf and ψc:

###### Fˆ(it),corr = ψf(Fˆ(it)), F˜ctxi = ψc(F˜i). (10)

These processed features are then concatenated with the unprocessed inputs, namely the motion-change gradient and the visibility map. This aggregated tensor is fused via a 1×1 convolution ψ to form the initial hidden state H0:

H0 = ψ [Fˆ(it),corr,F˜ctxi ,Fcorri,j ,Fctxi,j,V(t)] . (11)

This initial state H0 is then refined through multi temporal aggregator (TA) layers to do attention along the temporal dimension, as introduced in Sec. 3.4:

Hk+1 = TAk(Hk), k = 0,...,K−1. (12)

The resulting feature HK is passed through a final 1×1 projection ψfinal to produce the updated recurrent feature Fˆ(it+1) for the next iteration (t + 1):

Fˆ(it+1) = ψfinal(HK). (13) Finally, two lightweight MLP heads use this new hidden feature Fˆ(it+1) to predict the updates (∆M(2td) and ∆V(t)) for the 2D motion and visibility fields:

∆M(2td) = MLPmot(Fˆ(it+1)), M(2td+1) = M(2td) + ∆M(2td),

- (14)

∆V(t) = MLPvis(Fˆ(it+1)), V(t+1) = V(t) + ∆V(t).

- (15)

Overall, GRU functions as a recurrent refinement operator that jointly aggregates correlation cues, recurrent and semantic features, temporal motion differences, and visibility confidence. This enables robust and consistent iterative updates of the dense motion and visibility fields.

- 3D flow head H3d. The 3D flow head H3d aggregates all prepared features to predict the 3D flow update ∆M(3td).

Formally, given the lifted source features Fˆ(it+1), Fi, the lifted target point coordinates p(jt), p(jt+1) and features F(jt),F(jt+1), the 3D-to-2D point correlation C(3td,i,j) , and the 3D flow prior M(3td), the head operates as follows:

F′i = ϕ3fd(Fˆ(it+1)), F′i3d = ϕ3sd(Fi),

F′j(t) = ϕ3pd(F(jt)), F′j(t+1) = ϕ3pd(F(jt+1)), M3corrd = ϕ3motd (C(3td,i,j) ,M(3td)),

(16)

where ϕ3fd, ϕ3sd, ϕ3pd, and ϕ3motd denote dedicated 1×1 convolutional projections or motion encoders. These features are

concatenated and fused via a 1×1 convolution to form the initial hidden state:

###### H30d = ψ3d [F′i,F′i3d,F′j(t),F′j(t+1),M3corrd ] . (17)

Temporal refinement is performed using a temporal aggregator applied along the time dimension:

H3kd+1 = TA3d(H3kd), k = 0,...,K−1. (18) After a final 1×1 projection, the refined hidden feature is concatenated with the 3D flow prior p(jt+1) − p(jt):

F3outd = [ψfinal3d (H3Kd),p(jt+1) − p(jt)]. (19)

Finally, a lightweight prediction head MLP3d composed of several convolutional layers outputs the 3D flow update:

∆M(3td) = MLP3d(F3outd), M(3td+1) = M(3td) + ∆M(3td).

(20)

Overall, H3d serves as a 3D motion refinement module that jointly leverages 2D motion features, 3D source information, point correlations, temporal dependencies, and prior flow to produce accurate and temporally consistent 3D flow updates.

### E. Losses for Flow and Tracking Estimation

We observe that both pairwise scene flow and long-term point tracking datasets share a similar data structure, characterized by point-wise correspondences across temporal sequences. Our arbitrary-pair querying mechanism fundamentally supports variable temporal intervals and flexible numbers of frames. Leveraging this structural similarity, we adopt a unified training objective for both tasks, ensuring consistent supervision regardless of the dataset source.

#### E.1. Losses for 2D Branch

For the 2D branch, we align our supervision strategy with state-of-the-art point tracking methods [25, 34, 35]. The overall objective function, ℓtotal, is computed over a sequence of Niter iteratively refined predictions, utilizing an

exponential weighting factor γ (typically set to 0.8) to emphasize the quality of later updates. The total loss is a weighted sum of the trajectory loss (ℓtraj), the visibility loss (ℓvis), and the confidence loss (ℓconf), aggregated across all iterative steps t:

The total loss for the 3D branch, L3totald , is then composed of the 3D trajectory loss (L3trajd) and the scene flow smoothness loss (L3smoothd ). Adhering to the iterative refinement structure of CoTracker [35], the total loss is aggregated across all iterative steps t using an exponential weighting factor γ:

Niter

ℓtotal =

t=1

iter−t ℓ(trajt) + λvisℓ(vist) + λconfℓ(conft) (21)

γN

Note: The weights λvis and λconf are hyperparameters used to balance the contribution of each component. The individual loss components are defined as follows:

##### 1. Trajectory regression loss (ℓtraj). This loss measures

the L1 error between the predicted 2D coordinates M(2td) and the ground truth coordinates Mgt2d. Importantly, it is weighted by the ground truth visibility mask Vgt: points that are visible in the target frame are assigned a full weight of 1, while points that are occluded or invisible receive a smaller weight of 0.2. This design encourages the model to focus more on accurately predicting visible points while still providing a weak supervision signal for invisible points:

ℓ(trajt) = 1(Vgt) + 0.2 · (1 − 1(Vgt)) · M(2td) − Mgt2d 1.

(22)

###### 2. Visibility classification loss (ℓvis). The visibility loss uses Binary Cross Entropy (BCE) to supervise the predicted visibility V′, encouraging the model to correctly classify points as visible or occluded based on the ground truth mask Vgt:

ℓ(vist) = BCE V′(t),Vgt . (23)

###### 3. Confidence loss (ℓconf) Our model predicts a separate confidence score V′′ (track reliability or certainty), this loss supervises it using BCE. The ground truth label is a binary flag indicating whether the corresponding trajectory prediction falls within an acceptable pixel error tolerance δ:

ℓ(conft) = BCE V′′(t),1 M(2td) − Mgt2d

< δ . (24)

2

- E.2. Losses for 3D Branch For the 3D branch, we first apply the predicted 3D flow,

M(3td), to the first-frame camera-centric point cloud P1 to obtain the predicted corresponding points at all other

frames, denoted as Pˆ(3td):

Pˆ(3td) = P1 + M(3td). (25)

Niter

L3totald =

t=1

iter−t L(trajt),3d + λsmoothL(smootht),3d , (26)

γN

where Niter is the total number of update iterations and λsmooth is the hyperparameter balancing the contribution of the smoothness term.

- 1. Trajectory regression loss (ℓ3trajd ). Subsequently, we supervise the full 3D trajectory by adopting a loss function structurally similar to the 2D trajectory loss (Equation 22).

This 3D trajectory loss, ℓ3trajd, computes the L1 difference between the predicted 3D coordinates and the ground truth 3D coordinates, incorporating a robust weighting scheme based on ground truth visibility, Vgt:

ℓ(trajt),3d = 1(Vgt) + 0.2 · (1 − 1(Vgt)) · P ˆ(3td) − Pgt3d 1. (27)

- 2. Scene flow smoothness loss (ℓ3smoothd ). In addition to the trajectory supervision, we incorporate a scene flow smooth-

ness loss(ℓ3smoothd ) to enforce local rigidity and prevent erratic flow predictions across spatially adjacent points. Given the high dimensionality of the point cloud, calculating this loss across all N points is computationally expensive (O(N2)). To address this, we adopt an efficient sampling strategy based on anchor points.

Specifically, we randomly sample M anchor points

(Panchor) from the ground-truth point cloud Pgt3d, where M ≪ N. We then retrieve the corresponding M pre-

dicted flow vectors (Mˆ anchor) and use the spatial structure of Panchor to establish local neighborhoods via KNearest Neighbors (KNN) search. The loss is computed as the squared L2 distance between the flow vector of each anchor point and its K neighbors within the sampled set:

M

2 2

1 M · K

M ˆ (anchor,it) − Mˆ (anchor,jt)

ℓ(smootht),3d =

,

i=1 j∈NiK

(28) where Mˆ (anchor,it) is the predicted scene flow for the i-th anchor point at iteration t, and NiK denotes the set of K nearest neighbors of point i within the sampled M points. The effectiveness of this loss is illustrated in Fig. S1, demonstrating its impact on improving scene flow accuracy.

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

w/o w/

Fig. S1. Effectiveness of ℓ3smoothd .

### F. Test-time Pose Refinement

We refine the per-frame camera poses {πt ∈ SE(3)} from the reconstructed 3D tracks. We begin by computing dynamic masks using VLM [1] and Grounding-SAM [38, 50] to segment foreground dynamic objects. Static 3D tracks are then extracted by retaining only those tracks whose projections fall within the static regions.

Given these static tracks, we reproject each point from timestep t1 to timestep t2 under the current camera poses and optimize the poses using the following projection loss:

N

T

T

πt−1

ℓproj =

πt

Pstatic(i,t1)

2

(29)

1

t1=1

t2=1

i=1

−Pstatic(i,t2)∥22 ,

where πt(·) denotes the transformation at timestep t, Pstatic(i,t) ∈ R3 is the 3D location of the i-th static track at timestep t, and N is the total number of static tracks. To improve computational efficiency, we divide the entire video into C clips and estimate the camera poses of each clip in parallel. After optimizing the intra-clip poses, we estimate inter-clip transformations to merge all clips into a globally consistent camera trajectory. Finally, once the merged global poses are initialized, we refine all camera poses again by optimizing the reprojection loss in Eq. 29 over the entire sequence.

### G. Metric Details

We evaluate our method using a comprehensive set of metrics covering geometry estimation, motion estimation, and camera pose accuracy. Below, we denote the predicted value with (·) and the ground truth with (ˆ·). The set of valid pixels or points is denoted by Ω, and N = |Ω|.

#### G.1. Depth and Geometry Metrics

Abs Rel (Absolute Relative Error): This metric measures the mean absolute relative difference between the predicted depth di and the ground truth depth dˆi. It is defined as:

|dˆi − di| dˆi

1 N i∈Ω

. (30)

Abs Rel =

δ < 1.25 (Threshold Accuracy): This metric measures the percentage of pixels where the ratio between the predicted and ground truth depth is within a threshold of 1.25. It indicates the prevalence of accurate predictions:

1 N i∈Ω

I max

δ < τ =

- d ˆi

- di

di dˆi

,

< τ , (31)

where τ = 1.25, and I(·) is the indicator function.

#### G.2. 3D Scene Flow Metrics

EPE3D (3D End-Point Error): EPE3D measures the average Euclidean distance (in meters) between the predicted 3D scene flow vectors and the ground truth:

1 N i∈Ω ∥pi − pˆi∥2, (32)

EPE3D =

where pi is the predicted flow vector and pˆi is the ground truth.

AccS (Strict Accuracy - 3D): The percentage of points whose EPE3D is strictly within a tight threshold. Following standard scene flow protocols, a point is considered accurate if:

∥pi − pˆi∥2 ∥pˆi∥2

1 N i∈Ω

AccS =

I ∥pi − pˆi∥2 < τ1 ∨

< τ2 .

(33) Typically, τ1 = 0.05m and τ2 = 5%. Note that the relative error is normalized by the ground truth magnitude ∥pˆi∥2.

AccR (Relaxed Accuracy - 3D): Similar to AccS but with relaxed thresholds (typically τ1 = 0.10m and τ2 = 10%) to evaluate robustness against larger motions.

#### G.3. 2D Optical Flow Metrics

EPE2D (2D End-Point Error): The average Euclidean distance (in pixels) between the predicted 2D coordinates ui and ground truth uˆi:

1 N i∈Ω ∥ui − uˆi∥2. (34)

EPE2D =

AccS2D (Strict Accuracy - 2D): The percentage of points with EPE2D smaller than a strict threshold (1 pixel):

1 N i∈Ω

AccS2D =

I(∥ui − uˆi∥2 < τstrict). (35)

AccR2D (Relaxed Accuracy - 2D): The percentage of points satisfying a looser threshold τrelaxed (3 pixels) to measure coarse tracking capability.

#### G.4. Long-term Point Tracking Metrics (TAP-Vid)

APD (Average Position Deviation): This metric measures the percentage of points whose predicted positions fall within a specified threshold relative to the ground-truth depth.

AJ (Average Jaccard): Also known as the ”Position Accuracy” in some benchmarks, it measures the fraction of points that are within a specified distance threshold from the ground truth, averaged over the sequence. It is essentially a survival rate weighted by spatial accuracy.

δavgvis (Average Visible Delta): This metric specifically measures the position accuracy for points that are currently vis-

ible (not occluded). It computes the average percentage of visible points where the prediction error is below a specific threshold δ.

OA (Occlusion Accuracy): OA evaluates the binary classification performance of the visibility estimation head. It is the accuracy of predicting whether a point is occluded or visible:

1 Ntotal i

I(vi = vˆi), (36)

OA =

where vˆi ∈ {0,1} denotes the ground-truth visibility status and vi denotes the prediction.

#### G.5. Camera Pose Metrics

ATE (Absolute Trajectory Error): ATE measures the global consistency of the estimated camera trajectory. It computes the root-mean-square error (RMSE) of the translation difference between the estimated trajectory Test and ground truth Tgt after aligning them via a similarity transformation (Sim3) or rigid transformation (SE3):

ATE =

M

1 M

∥trans(T−gt,j1 STest,j)∥22. (37)

j=1

RTE (Relative Trajectory Error): RTE measures local consistency by averaging the drift over fixed time intervals or distances. It evaluates the error in the relative pose between pairs of frames separated by a fixed lag.

RRE (Relative Rotation Error): RRE measures the geodesic distance between the predicted and ground truth rotation matrices, averaged over all frame pairs. It is defined as:

1 M j

RRE =

arccos

trace(RTgt,jRest,j) − 1 2

. (38)

### H. More Ablation Study

Geometry encoder. We provide additional ablation experiments on the geometry encoder, as summarized in Tab. S1. This ablation is conducted using MoGe [80] as

the geometry backbone initialization. Replacing our affineinvariant loss with a scale-invariant one (“w/ scale-inv”) degrades performance, demonstrating the effectiveness of the affine-invariant formulation. Removing the regularization loss (ℓreg) leads to a catastrophic drop (Abs Rel 0.3530), highlighting its crucial role in stabilizing convergence. Finally, removing the local loss (ℓlocal) also results in a slight degradation (Abs Rel 0.1053), confirming its contribution to refining geometric details.

Setting Abs Rel ↓ δ < 1.25 ↑

w/ scale-inv 0.1060 0.8788 w/o ℓreg 0.3530 0.3547 w/o ℓlocal 0.1053 0.8790 Full (Ours) 0.0973 0.8921

Table S1. Ablation study on geometry encoder. The results are averaged over all datasets. Best results are highlighted in

darker blue , and second best in lighter blue .

Test-time pose refinement. Leveraging long-term correspondences, predicted point clouds, and initial poses, our method supports test-time pose refinement via Bundle Adjustment. Tab. S2 evaluates camera pose accuracy on the Sintel [6] dataset, where “Ours (FF.)” consistently outperforms the geometry-and-motion-based method STV2. Applying our optimization strategy (detailed in Supp. Sec. E) refines these predictions further; this optimized variant, “Ours (FF.+Opt.)”, achieves higher accuracy than MegaSaM [47] at a lower computational cost.

[Figure 65]

Fig. S2. Scene flow Visualization. The deformed point maps (colored ) show that our method produces more temporally consistent geometry and motion.

Method ATE↓ RTE↓ RRE↓ Time (s)

STV2 [87] (FF.) 0.133 0.057 0.641 9 DA3 [49](FF.) 0.124 0.061 0.331 6 Ours (FF.) 0.119 0.054 0.309 6

MegaSaM [47] (Opt.) 0.059 0.027 0.120 310 Ours (FF.+Opt.) 0.045 0.019 0.115 294

Table S2. Efficiency comparison on 16-frame ADT [61] sequences. Gray rows indicate Out-of-Memory (OOM) failures under dense tracking settings. Best and second best results are highlighted.

### I. More Visualization

To further demonstrate the capability of our framework, we provide additional qualitative results. Specifically, we present camera-centric results, including scene flow estimation, dense first-frame 2D tracking, dense first-frame 3D tracking, and per-pixel tracking across all frames (Fig. S2, Fig. S4, Fig. S5, and Fig. S6). We also show worldcentric dense tracking across all frames in Fig. S7. (1) Scene Flow Visualization. We first present pairwise scene flow predictions between two frames. Scene flow characterizes the 3D motion of each point and serves as an essential cue for understanding dynamic geometries. In our visualization, the deformed point maps (colored ) highlight the predicted per-point displacements. Compared with the current state-of-the-art method ZeroMSF [48], our results exhibit smoother spatial transitions and significantly higher temporal consistency in both geometry and motion. This demonstrates that our deformation field models coherent 3D dynamics more effectively, even in complex and rapidly changing scenes. (2) Dense 2D Tracking from the First Frame. The second group shows dense 2D correspondences anchored at the first frame. Pixels with the same color trace the evolution of the same point across time, revealing long-term and fine-grained motion patterns throughout the sequence. (3) Dense 3D Trajectories from the First Frame. We further visualize reconstructed 3D trajectories originating from the first frame. For clarity and rendering efficiency, we uniformly subsample the trajectories while preserving their overall motion structures and temporal smoothness. (4) Dense Per-Pixel Trajectories Across All Frames. We present dense per-pixel trajectories over the full sequence (second row), which consistently track motions even when new objects enter the scene, highlighting the robustness of our method in handling complex dynamic scenarios. (5) Dense Trajectories in the World Coordinate System. Finally, we visualize dense trajectories across all frames after transforming them into a unified global world coordinate system. This world-centric representation effectively decouples camera ego-motion from object dynamics: static background elements remain spatially stable over time, while dynamic objects exhibit coher-

ent and physically meaningful absolute motion trajectories in 3D space. For more intuitive demonstrations and continuous motion effects, we refer the reader to the supplementary video.

### J. Discussion with Concurrent work

MotionCrafter [105] is our concurrent work that also investigates joint reconstruction of geometry and motion. The key difference lies in the modeling paradigm: MotionCrafter explores the potential of diffusion models for joint geometry–motion reconstruction, whereas our method adopts a purely ViT-based feed-forward framework. MotionCrafter primarily focuses on long-sequence motion modeling by predicting motion between consecutive frames (i.e., frame 1 → 2, 2 → 3, ..., N−1 → N). In contrast, our method is more flexible, as it supports on-demand arbitrarypair motion prediction, enabling consistent 4D reasoning across non-consecutive frames. We compare the worldcentric geometric reconstruction results in Tab. S3, and provide qualitative scene flow comparisons in Fig. S3.

Method Sintel GMUKitchen Monkaa Scannet test Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑

MotionCrafter [105] 0.2192 0.6942 0.1303 0.8991 0.1864 0.7837 0.1055 0.9663 Ours 0.1408 0.8155 0.0397 0.9588 0.1503 0.8083 0.0237 0.9931

###### Table S3. Evaluation on world-centric geometric reconstruction.

[Figure 66]

Fig. S3. Scene Flow Visualization. The deformed point maps (shown in solid colors) indicate that our method produces higherquality geometry and motion.

### K. Limitations and Future Work

Despite the effectiveness of our proposed approach, several limitations remain. A primary constraint is the dependence on captured 4D motion datasets, which are laborintensive to acquire and limited in scale. Consequently, the model may struggle to generalize to extreme poses or complex topological changes not represented in the training set. In the future, we plan to explore synthetic data generation pipelines, potentially utilizing generative diffusion models or physics engines, to create large-scale and diverse training samples. Furthermore, we aim to investigate unsupervised or semi-supervised learning schemes to reduce the dependency on labeled data, ultimately pushing the boundaries of

- 4D reconstruction in generalizable settings. References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 4
- [2] Vivek Alumootil, Tuan-Anh Vu, and M Khalid Jawed. Dept3r: Joint dense point tracking and 3d reconstruction of dynamic scenes in a single forward pass. arXiv preprint arXiv:2512.13122, 2025. 3
- [3] Arjun Balasingam, Joseph Chandler, Chenning Li, Zhoutong Zhang, and Hari Balakrishnan. Drivetrack: A benchmark for long-range point tracking in real-world videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22488–22497, 2024. 6, 7
- [4] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, et al. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897,

2021. 1

- [5] Ryan Burgert, Yuancheng Xu, Wenqi Xian, Oliver Pilarski, Pascal Clausen, Mingming He, Li Ma, Yitong Deng, Lingxiao Li, Mohsen Mousavi, Michael Ryoo, Paul Debevec, and Ning Yu. Go-with-the-flow: Motion-controllable video diffusion models using real-time warped noise. In CVPR,

2025. 1

- [6] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In Computer Vision–ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part VI 12, pages 611–

625. Springer, 2012. 8, 5

- [7] Yukang Cao, Jiahao Lu, Zhisheng Huang, Zhuowen Shen, Chengfeng Zhao, Fangzhou Hong, Zhaoxi Chen, Xin Li, Wenping Wang, Yuan Liu, et al. Reconstructing 4d spatial intelligence: A survey. arXiv preprint arXiv:2507.21045,

2025. 1

- [8] Joao Carreira and Andrew Zisserman. Quo vadis, action

- recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 7
- [9] Weirong Chen, Le Chen, Rui Wang, and Marc Pollefeys. Leap-vo: Long-term effective any point tracking for visual odometry. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19844– 19853, 2024. 8
- [10] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391,

2025. 3

- [11] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 3
- [12] Seokju Cho, Jiahui Huang, Jisu Nam, Honggyu An, Seungryong Kim, and Joon-Young Lee. Local all-pair correspondence for point tracking. In European Conference on Computer Vision, pages 306–325. Springer, 2024. 7
- [13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 8, 1
- [14] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022. 7
- [15] Carl Doersch, Yi Yang, Mel Vecerik, Dilara Gokay, Ankush Gupta, Yusuf Aytar, Joao Carreira, and Andrew Zisserman. Tapir: Tracking any point with per-frame initialization and temporal refinement. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10061– 10072, 2023. 3, 7
- [16] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, Joao Carreira, et al. Bootstap: Bootstrapped training for tracking-any-point. In Proceedings of the Asian Conference on Computer Vision, pages 3257–3274, 2024. 7
- [17] Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. Flownet: Learning optical flow with convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 2758–2766, 2015. 1
- [18] Haiwen Feng, Junyi Zhang, Qianqian Wang, Yufei Ye, Pengcheng Yu, Michael J Black, Trevor Darrell, and Angjoo Kanazawa. St4rtrack: Simultaneous 4d reconstruction and tracking in the world. arXiv preprint arXiv:2504.13152, 2025. 1, 2, 3
- [19] Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. Virtual worlds as proxy for multi-object tracking analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4340–4349, 2016. 1
- [20] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The inter-

national journal of robotics research, 32(11):1231–1237,

2013. 6, 8

- [21] Georgios Georgakis, Md Alimoor Reza, Arsalan Mousavian, Phi-Hung Le, and Jana Koˇseck´a. Multiview rgb-d dataset for object instance detection. In 2016 Fourth international conference on 3D vision (3DV), pages 426–434. IEEE, 2016. 8
- [22] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022. 6, 8, 1
- [23] Xiuye Gu, Yijie Wang, Chongruo Wu, Yong Jae Lee, and Panqu Wang. Hplflownet: Hierarchical permutohedral lattice flownet for scene flow estimation on large-scale point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3254–3263,

2019. 6

- [24] Zekai Gu, Rui Yan, Jiahao Lu, Peng Li, Zhiyang Dou, Chenyang Si, Zhen Dong, Qifeng Liu, Cheng Lin, Ziwei Liu, et al. Diffusion as shader: 3d-aware video diffusion for versatile video generation control. arXiv preprint arXiv:2501.03847, 2025. 1
- [25] Adam W Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Suya You, et al. Alltracker: Efficient dense point tracking at high resolution. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5253–5262, 2025. 3, 2
- [26] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. arXiv preprint arXiv:2409.02095,

2024. 8

- [27] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 3
- [28] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. Deepmvs: Learning multiview stereopsis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2821–2830,

2018. 1

- [29] Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, MingYu Liu, Dieter Fox, Kaichun Mo, and Li Fei-Fei. Pointworld: Scaling 3d world models for in-the-wild robotic manipulation. arXiv preprint arXiv:2601.03782, 2026. 1
- [30] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. arXiv preprint arXiv:2504.07961, 2025. 3
- [31] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. CVPR,

2025. 3

- [32] Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh. Panoptic studio: A massively multiview system for social motion capture. In Proceedings of the IEEE international conference on computer vision, pages 3334–3342,

2015. 6, 7

- [33] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13229–13239, 2023. 1
- [34] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudolabelling real videos. arXiv preprint arXiv:2410.11831,

2024. 2, 3, 7

- [35] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker: It is better to track together. In European Conference on Computer Vision, pages 18–35. Springer, 2024. 3, 7, 2
- [36] Jay Karhade, Nikhil Keetha, Yuchen Zhang, Tanisha Gupta, Akash Sharma, Sebastian Scherer, and Deva Ramanan. Any4d: Unified feed-forward metric 4d reconstruction. arXiv preprint arXiv:2512.10935, 2025. 3, 6, 7
- [37] Nikhil Keetha, Norman M¨uller, Johannes Sch¨onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414, 2025. 3, 8
- [38] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 4
- [39] Daniel Kondermann, Rahul Nair, Katrin Honauer, Karsten Krispin, Jonas Andrulis, Alexander Brock, Burkhard Gussefeld, Mohsen Rahimimoghaddam, Sabine Hofmann, Claus Brenner, et al. The hci benchmark suite: Stereo and flow ground truth with uncertainties for urban autonomous driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 19– 28, 2016. 1
- [40] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1611–1621, 2021. 3
- [41] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Joao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. Advances in Neural Information Processing Systems, 37:82149–82165, 2024. 6
- [42] Alex X Lee, Coline Manon Devin, Yuxiang Zhou, Thomas Lampe, Konstantinos Bousmalis, Jost Tobias Springenberg, Arunkumar Byravan, Abbas Abdolmaleki, Nimrod Gileadi,

- David Khosid, et al. Beyond pick-and-place: Tackling robotic stacking of diverse shapes. In 5th Annual Conference on Robot Learning, 2021. 7
- [43] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer,

2024. 3, 7

- [44] Hongyang Li, Hao Zhang, Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, and Lei Zhang. Taptr: Tracking any point with transformers as detection. In European Conference on Computer Vision, pages 57–75. Springer, 2024. 3, 7
- [45] Mengfei Li, Peng Li, Zheng Zhang, Jiahao Lu, Chengfeng Zhao, Wei Xue, Qifeng Liu, Sida Peng, Wenxiao Zhang, Wenhan Luo, et al. Unish: Unifying scene and human reconstruction in a feed-forward pass. arXiv preprint arXiv:2601.01222, 2026. 3
- [46] Yijin Li, Yichen Shen, Zhaoyang Huang, Shuo Chen, Weikang Bian, Xiaoyu Shi, Fu-Yun Wang, Keqiang Sun, Hujun Bao, Zhaopeng Cui, et al. Blinkvision: A benchmark for optical flow, scene flow and point tracking estimation using rgb frames and events. In European conference on computer vision, pages 19–36. Springer, 2024. 6, 1
- [47] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast, and robust structure and motion from casual dynamic videos. arXiv preprint arXiv:2412.04463, 2024. 3, 5, 6
- [48] Yiqing Liang, Abhishek Badki, Hang Su, James Tompkin, and Orazio Gallo. Zero-shot monocular scene flow estimation in the wild. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21031–21044,

2025. 1, 2, 3, 6, 7, 9

- [49] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025. 2, 3, 5, 8, 1, 6
- [50] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European Conference on Computer Vision, pages 38–55. Springer, 2024. 4
- [51] Xingyu Liu, Charles R Qi, and Leonidas J Guibas. Flownet3d: Learning scene flow in 3d point clouds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 529–537, 2019. 6
- [52] Xinhang Liu, Yuxi Xiao, Donny Y Chen, Jiashi Feng, YuWing Tai, Chi-Keung Tang, and Bingyi Kang. Trace anything: Representing any video in 4d via trajectory fields. arXiv preprint arXiv:2510.13802, 2025. 1, 3
- [53] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 1
- [54] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocu-

- lar depth estimation for dynamic videos. arXiv preprint arXiv:2412.03079, 2024. 3, 8
- [55] Jiahao Lu, Weitao Xiong, Jiacheng Deng, Peng Li, Tianyu Huang, Zhiyang Dou, Cheng Lin, Sai-Kit Yeung, and Yuan Liu. Trackingworld: World-centric monocular 3d tracking of almost all pixels. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. 1, 2, 3
- [56] Yihang Luo, Shangchen Zhou, Yushi Lan, Xingang Pan, and Chen Change Loy. 4rc: 4d reconstruction via conditional querying anytime and anywhere. arXiv preprint arXiv:2602.10094, 2026. 3
- [57] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016. 8, 1
- [58] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4981–4991, 2023. 1
- [59] Tuan Duc Ngo, Peiye Zhuang, Chuang Gan, Evangelos Kalogerakis, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang. Delta: Dense efficient long-range 3d tracking for any video. arXiv preprint arXiv:2410.24211,

2024. 1, 2, 3, 6, 7

- [60] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7855–7862. IEEE, 2019. 8
- [61] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20133–20143, 2023. 6, 7, 9
- [62] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. Unidepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10106–10116, 2024. 2
- [63] Jinyuan Qu, Hongyang Li, Shilong Liu, Tianhe Ren, Zhaoyang Zeng, and Lei Zhang. Taptrv3: Spatial and temporal context foster robust tracking of any point in long video. arXiv preprint arXiv:2411.18671, 2024. 3
- [64] Yiming Ren, Chengfeng Zhao, Yannan He, Peishan Cong, Han Liang, Jingyi Yu, Lan Xu, and Yuexin Ma. Lidar-aid inertial poser: Large-scale human motion capture by sparse inertial and lidar sensors. IEEE Transactions on Visualization and Computer Graphics, 29(5):2337–2347, 2023. 1
- [65] Stephan R Richter, Zeeshan Hayder, and Vladlen Koltun. Playing for benchmarks. In Proceedings of the IEEE inter-

national conference on computer vision, pages 2213–2222,

2017. 1

- [66] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021. 1
- [67] Jakob Schmid, Azin Jahedi, Noah Berenguel Senn, and Andr´es Bruhn. Ms-raft-3d: A multi-scale architecture for recurrent image-based scene flow. In 2025 IEEE International Conference on Image Processing (ICIP), pages 1570–1575. IEEE, 2025. 3
- [68] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573–580. IEEE, 2012. 8
- [69] Edgar Sucar, Eldar Insafutdinov, Zihang Lai, and Andrea Vedaldi. V-dpm: 4d video reconstruction with dynamic point maps. arXiv preprint arXiv:2601.09499, 2026. 3, 6, 7
- [70] Deqing Sun, Daniel Vlasic, Charles Herrmann, Varun Jampani, Michael Krainin, Huiwen Chang, Ramin Zabih, William T Freeman, and Ce Liu. Autoflow: Learning a better training set for optical flow. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10093–10102, 2021. 1
- [71] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua Shen, Jiangmiao Pang, et al. Aether: Geometric-aware unified world modeling. arXiv preprint arXiv:2503.18945,

2025. 8

- [72] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs field transforms for optical flow. In Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419. Springer, 2020. 2, 6
- [73] Zachary Teed and Jia Deng. Raft-3d: Scene flow using rigid-motion embeddings. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8375–8384, 2021. 3, 6
- [74] Zachary Teed, Lahav Lipson, and Jia Deng. Deep patch visual odometry. Advances in Neural Information Processing Systems, 36, 2024. 8
- [75] Mel Vecerik, Carl Doersch, Yi Yang, Todor Davchev, Yusuf Aytar, Guangyao Zhou, Raia Hadsell, Lourdes Agapito, and Jon Scholz. Robotap: Tracking arbitrary points for few-shot visual imitation. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 5397–5403. IEEE,

2024. 7

- [76] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651, 2025. 1, 2, 3, 6, 7, 8
- [77] Kaixuan Wang and Shaojie Shen. Flow-motion and depth network for monocular stereo and beyond. IEEE Robotics and Automation Letters, 5(2):3307–3314, 2020. 1

- [78] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. arXiv preprint arXiv:1912.09678, 2019. 1
- [79] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. arXiv preprint arXiv:2501.12387, 2025. 3, 8
- [80] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025. 3, 6, 8, 1, 5
- [81] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details. arXiv preprint arXiv:2507.02546, 2025. 8
- [82] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 1, 2, 3
- [83] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 1
- [84] Yihan Wang, Lahav Lipson, and Jia Deng. Sea-raft: Simple, efficient, accurate raft for optical flow. In European Conference on Computer Vision, pages 36–54. Springer, 2024. 6
- [85] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. pi3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347,

2025. 2, 3, 8, 1

- [86] Yuxi Xiao, Qianqian Wang, Shangzhan Zhang, Nan Xue, Sida Peng, Yujun Shen, and Xiaowei Zhou. Spatialtracker: Tracking any 2d pixels in 3d space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20406–20417, 2024. 1, 3, 6, 7
- [87] Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: Advancing 3d point tracking with explicit camera motion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6726–6737, 2025. 1, 2, 3, 4, 5, 6, 7, 8, 9
- [88] Weitao Xiong, Zhiyuan Yuan, Jiahao Lu, Chengfeng Zhao, Peng Li, and Yuan Liu. Human3r: Incorporating human priors for better 3d dynamic reconstruction from monocular videos. arXiv preprint arXiv:2512.06368, 2025. 3
- [89] Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, SongHai Zhang, and Ying Shan. Geometrycrafter: Consistent

- geometry estimation for open-world videos with diffusion priors. arXiv preprint arXiv:2504.01016, 2025. 3
- [90] Gengshan Yang and Deva Ramanan. Upgrading optical flow to 3d scene flow through optical expansion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1334–1343, 2020. 3, 6
- [91] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935, 2025. 3
- [92] David Yifan Yao, Albert J Zhai, and Shenlong Wang. Uni4d: Unifying visual foundation models for 4d modeling from a single video. arXiv preprint arXiv:2503.21761,

2025. 2, 3

- [93] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 1
- [94] Bowei Zhang, Lei Ke, Adam W Harley, and Katerina Fragkiadaki. Tapip3d: Tracking any point in persistent 3d geometry. arXiv preprint arXiv:2504.14717, 2025. 1, 2, 3, 4, 5
- [95] Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco, Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar, Jo¨elle K Barral, Raia Hadsell, et al. Efficiently reconstructing dynamic scenes one d4rt at a time. arXiv preprint arXiv:2512.08924, 2025. 3
- [96] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. arXiv preprint arXiv:2410.03825, 2024. 3, 7, 8
- [97] Songyan Zhang, Yongtao Ge, Jinyuan Tian, Guangkai Xu, Hao Chen, Chen Lv, and Chunhua Shen. Pomato: Marrying pointmap matching with temporal motion for dynamic 3d reconstruction. arXiv preprint arXiv:2504.05692, 2025. 1, 2, 3, 6, 7, 8, 9
- [98] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision, pages 20–37. Springer, 2022. 3
- [99] Chengfeng Zhao, Juze Zhang, Jiashen Du, Ziwei Shan, Junye Wang, Jingyi Yu, Jingya Wang, and Lan Xu. I’m hoi: Inertia-aware monocular capture of 3d human-object interactions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 729– 741, 2024. 1
- [100] Shiyu Zhao, Long Zhao, Zhixing Zhang, Enyu Zhou, and Dimitris Metaxas. Global matching with overlapping attention for optical flow estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17592–17601, 2022. 6
- [101] Wang Zhao, Shaohui Liu, Hengkai Guo, Wenping Wang, and Yong-Jin Liu. Particlesfm: Exploiting dense point tra-

- jectories for localizing moving cameras in the wild. In European Conference on Computer Vision, pages 523–542. Springer, 2022. 8
- [102] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19855–19865, 2023. 6, 7, 1
- [103] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025. 1
- [104] Ruijie Zhu, Yanzhe Liang, Hanzhi Chang, Jiacheng Deng, Jiahao Lu, Wenfei Yang, Tianzhu Zhang, and Yongdong Zhang. Motiongs: Exploring explicit motion guidance for deformable 3d gaussian splatting. arXiv preprint arXiv:2410.07707, 2024. 1
- [105] Ruijie Zhu, Jiahao Lu, Wenbo Hu, Xiaoguang Han, Jianfei Cai, Ying Shan, and Chuanxia Zheng. Motioncrafter: Dense geometry and motion reconstruction with a 4d vae,

2026. 3, 6

- [106] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539, 2025. 3

[Figure 67]

###### Fig. S4. Visualization of first-frame 2D dense tracking.

[Figure 68]

###### Fig. S5. Visualization of first-frame 3D dense tracking.

[Figure 69]

###### Fig. S6. Visualization of dense per-pixel trajectories across all frames.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

##### Fig. S7. Visualization of the dense trajectories across15 all frames in the global world coordinate system.

