# arXiv:2604.26067v1[cs.CV]28Apr2026

## RADIO-ViPE: Online Tightly Coupled Multi-Modal Fusion for Open-Vocabulary Semantic SLAM in Dynamic Environments

Zaid Nasser1∗, Mikhail Iumanov1∗, Tianhao Li1∗, Maxim Popov1, Jaafar Mahmoud1†, Sergey Kolyubin1

Abstract—We present RADIO-ViPE (Reduce All Domains Into One — Video Pose Engine), an online semantic SLAM system that enables geometry-aware open-vocabulary grounding, associating arbitrary natural language queries with localized 3D regions and objects in dynamic environments. Unlike existing approaches that require calibrated, posed RGB-D input, RADIO-ViPE operates directly on raw monocular RGB video streams, requiring no prior camera intrinsics, depth sensors, or pose initialization. The system tightly couples multimodal embeddings — spanning vision and language — derived from agglomerative foundation models (e.g., RADIO [1]) with geometric scene information. This coupling takes place in initialization, optimization and factor graph connections to improve the consistency of the map from multiple modalities. The optimization is wrapped within adaptive robust kernels, designed to handle both actively moving objects and agentdisplaced scene elements (e.g., furniture rearranged during egocentric session). Experiments demonstrate that RADIO-ViPE achieves state-of-the-art results on the dynamic TUM-RGBD benchmark [2] while maintaining competitive performance against offline open-vocabulary methods that rely on calibrated data and static scene assumptions. RADIO-ViPE bridges a critical gap in real-world deployment, enabling robust openvocabulary semantic grounding for autonomous robotics and unconstrained in-the-wild video streams – Project Page.

I. INTRODUCTION

Knowledge mapping — the process of grounding freeform linguistic concepts onto a 3D geometric representation of the environment — is a fundamental capability for general-purpose robots. Unlike humans, who innately perceive objects not merely as geometric primitives but with rich semantic understanding of their identity, function, and affordances, robots have traditionally relied on purely geometric maps for self-localization and collision avoidance. Recent advances in computer vision, foundation models and large language models have begun to bridge this gap, enabling knowledge-enriched scene representations that support object-level reasoning and open-ended task execution. Openvocabulary semantic perception is a key component here, as it liberates robots from the fixed object taxonomies, allowing flexible, language-driven scene interpretation across diverse, unstructured environments.

A natural source of supervision for such systems is internet-scale video, which offers a much more affordable, scalable, and more diverse alternative to collecting data from curated robot demonstrations, encompassing broad scene

∗ Equal contribution † Corresponding author: jaafar.a.mahmoud@itmo.ru 1 Biomechatronics and Energy-Efficient Robotics (BE2R) Lab, ITMO

University, Saint Petersburg, Russia

[Figure 1]

Fig. 1. RADIO-ViPE: An online, ready-to-deploy semantic SLAM system that ground free-form language queries spatially from uncalibrated, unconstrained monocular RGB video

layouts, object appearances, and human activities across indoor and outdoor settings [3], [4]. However, such data is inherently unstructured and uncalibrated, lacking the depth, pose, and semantic annotations required for most of existing 3D scene understanding methods. In addition, real-world environments introduce dynamic disturbances — from moving agents to quasi-static objects such as furniture and appliances displaced by human activity — that corrupt data associations and degrade localization and mapping if left unaddressed.

These challenges collectively motivate RADIO-ViPE: a calibration-free, online semantic SLAM system that ingests raw, unconstrained monocular RGB video and produces a semantically and geometrically consistent scene representation supporting open-vocabulary language grounding. We build upon ViPE [5] and agglomerative foundation models ([1], [6]) that leverage multi-teacher distillation, and provide a unified student model that retains and improves the distinct capabilities of multiple teachers [1].

The main contributions of this work are stated as follows: 1) Vision-Language-Geometric Fusion via Dense Bun-

dle Adjustment. A novel tightly coupled multi-modal fusion that jointly embeds high-level features from

agglomerative foundation models [1] with geometric constraints directly within a dense bundle adjustment framework. The factor graph is also formulated with both geometric-semantic criteria ensuring consistency of the map built from multiple sources of modalities.

- 2) Temporally Consistent Adaptive Robust Kernel. A dynamic-aware robust optimization scheme that extends adaptive kernel formulations [7], [8] with temporal consistency, jointly reasoning over geometric reprojection error and cross-view semantic embedding discrepancy to suppress the influence of moving and quasi-static scene elements at data association and improve robustness.
- 3) Calibration-Free Online Open-Vocabulary Semantic SLAM. An online, ready-to-deploy semantic SLAM system that unifies vision, language, and geometry into a spatial representation, supporting freeform natural language grounding in 3D directly from uncalibrated, unconstrained monocular RGB video with no depth sensors, pose priors, or category-specific supervision required.

By unifying these concepts, RADIO-ViPE offers a powerful alternative to popular state-of-the-art feedforward SLAM methods [9], [10], [11], excelling particularly in dynamic environments.

II. RELATED WORKS

SLAM systems have evolved from purely geometric pipelines toward rich, semantically-aware scene understanding. We survey existing approaches along five axes localization, mapping, semantic understanding, real-time operation, and dynamic scene handling — and organize them into four categories (Table I).

- A. Geometric SLAM Systems

Early SLAM systems prioritized geometric accuracy at the expense of semantic reasoning. ORB-SLAM3 [12] remains a strong baseline for visual-inertial odometry, offering robust multi-map management and loop closure across diverse sensor configurations, yet provides no semantic interpretation or dynamic object handling. RVWO [7] addresses dynamic objects in wheeled robot settings but lacks semantic awareness. Kimera [13] and RGBDS-SLAM [14] incorporate instancelevel semantics and RGB-D fusion respectively, yet both remain confined to closed-set, pre-defined object categories. SamSLAM [15] advances dynamic handling via categoryagnostic segmentation but cannot associate scene elements with open-vocabulary language descriptions.

- B. Offline Open-Vocabulary Scene Understanding

Foundation models have enabled compelling openvocabulary understanding in 3D scene reconstruction, albeit exclusively in offline settings. BBQ [16] and ConceptGraphs [17] leverage large vision-language models to construct semantically rich 3D scene graphs with natural language grounding. HOV-SG [18] further organizes these into hierarchical spatial-semantic structures. OpenScene [19]

Grounding

Semantics

Odometry

Calib-free

Dynamic

Mapping

Online

Method

ORBS3 [12] ✗ ✗ ✗ ✗ RVWO [7] ✗ ✗ ✗ Kimera [13] Closed ✗ ✗ ✗ SamS [15] Agnostic ✗ ✗ RGBDS [14] Closed ✗ ✗ ✗

BBQ [16] ✗ Open ✗ ✗ ✗ CG [17] ✗ Open ✗ ✗ ✗ HOV-SG [18] ✗ Open ✗ ✗ ✗ OS [19] ✗ Open ✗ ✗ ✗ ✗ OM3D [20] ✗ Open ✗ ✗ ✗ ✗

CLIO [21] Task-driven ✗ ✗ OVO [22] Open ✗ ✗ Rayfronts [23] Open ✗ ✗ ✗

VGGT [10] ✗ ✗ ✗ DUST3R [11] ✗ ✗ ViPE [5] Predefined ✗ ✗

### RADIO-ViPE Open

TABLE I COMPARISON OF STATE-OF-THE-ART SLAM METHODS

Columns indicate: Online operation, Semantic understanding type (Closed=predefined classes, Agnostic=category-agnostic without semantics, Open=open-vocabulary), Geometric Grounding=the ability to textually access and interact with the 3D map, Odometry estimation, Map reconstruction, Dynamic scene handling and calibration requirement.

and OpenMask3D [20] achieve zero-shot 3D segmentation by distilling CLIP features into 3D point representations. Despite strong semantic capabilities, all these methods operate offline, lack integrated odometry, and assume static scenes — significantly constraining their applicability in real robotic deployments.

C. Real-Time Open-Vocabulary SLAM

Recent efforts have sought to reconcile open-vocabulary understanding with real-time operation. CLIO [21] introduces an information-theoretic framework that dynamically clusters 3D primitives according to task-driven language instructions. OVO-SLAM [22] integrates CLIP embeddings within a Gaussian splatting representation for real-time open-vocabulary mapping. RayFronts [23] depends on dense language-aligned features to provide dense voxel-wise embedding of the map. While these systems successfully couple semantic understanding with odometry and mapping, neither provides robustness to dynamic or quasi-static scene disturbances.

D. Feed-Forward SLAM

Feed-forward approaches directly regress 3D geometry from images. DUSt3R [11] reformulates pairwise reconstruction as pointmap regression, removing rigid projective camera assumptions. VGGT-SLAM [10] incrementally aligns

submaps while explicitly optimizing over the SL(4) manifold to resolve the 15-DoF projective ambiguity inherent to uncalibrated monocular reconstruction. ViPE [5] extends this paradigm to in-the-wild video, achieving robust intrinsic and extrinsic estimation with near-metric depth at 3–5 FPS. However, feed-forward systems broadly assume scene rigidity, making them vulnerable to dynamic objects, and lack the high-level semantic representations required for openvocabulary grounding.

Despite substantial progress across these paradigms, no existing system simultaneously achieves real-time operation, open-vocabulary semantic grounding, robust odometry, accurate mapping, and dynamic scene robustness. This integration gap severely limits deployment in complex, real-world environments. RADIO-ViPE addresses this gap with a unified framework that seamlessly combines all these capabilities — supporting applications ranging from autonomous robot navigation and ego-centric [4] or scene-centric [24] video data resourcing.

III. METHODOLOGY

RADIO-ViPE accepts an uncalibrated monocular RGB video stream and unifies camera pose optimization from flow estimation based on [25], depth from monocular foundation depth models [26], [27] that is aligned with consistent 3D SLAM point cloud, and dense high-level visual embeddings from RADIO [1] within a sliding window factor graph optimization framework. The optimization is then wrapped with adaptive kernels [7], [28] to improve robustness in dynamic environments. Altogether the pipeline is operating online at approximately 8–10 FPS.

A. System Overview

The pipeline is based on [5], [25] but with the following adjustments as follows (See Fig. 2):

- 1) Camera Initialization. Camera intrinsics are bootstrapped from uniformly sampled frames using GeoCalib [29], requiring no calibration targets or known camera models, and are subsequently co-optimized within bundle adjustment [5].
- 2) Keyframe Selection. Relative motion between each incoming frame and the most recent keyframe is estimated via weighted dense optical flow [25], [5]. Frames exceeding a predefined motion threshold are designated as keyframes and appended to the factor graph G = (V,E), where V denotes keyframes and E pairwise connections.
- 3) Multi-Modal Feature Extraction. Dense multi-modal embeddings are extracted per keyframe using RADSeg [1], [6] (Sec. III-B), upsampled via bilinear interpolation to (H/8,W/8), and compressed to D=256 dimensions via PCA (Sec. IV) to ensure scalability and efficiency in memory management.
- 4) Depth Estimation. A metric depth map is estimated per keyframe using monocular foundation depth models [27], [26], [30], converted to inverse depth (disparity) for numerical stability, and downsampled by

- a factor of 8 to match the estimated optical flow resolution [5].
- 5) Semantic Flow Initialization. To improve the robustness of optical flow priors [25] in textureless surfaces, we augment the geometric reprojection prior with a semantic correspondence term derived from dense RADIO features. Concretely, for each pixel u in frame i, we compute cosine similarities between the PCA-compressed Radio embedding Zi(u) ∈ RK and pixels embeddings Zj(v), yielding a dense semantic flow field Ωsem(u). The two priors are then fused via per-pixel confidence-based blending:

Ωprior(u) := β Ωprior(u) + 1 − β Ωsem(u), (1)

where the blending weight β balances the photometric flow confidence produced by the flow network [25] against the peak semantic similarity score. The blended initialization replaces Ωprior when constructing the correlation volume for the photometric flow term, leaving the flow network architecture unchanged.

- 6) Bundle Adjustment. Camera intrinsics, poses, and 3D scene structure are jointly refined over all keyframes by minimizing a vision-language-geometric energy function (Sec. III-C). All terms are evaluated on downsampled images ˜I ∈ RH×W×3, H=h/8, W=w/8, to ensure compatibility with the estimated optical flow [5]. Factor graph connectivity is augmented beyond geometric proximity by embedding-based co-visibility: a global descriptor per keyframe is obtained by mean-

pooling its RADSeg embeddings and ℓ2-normalizing the result, yielding a compact representation that is free to compute as the embeddings are already resident in the keyframe buffer. Incoming keyframes are matched against all non-recent keyframes (excluding the most recent τ frames to suppress trivially adjacent matches) via a single cosine-similarity query; pairs whose similarity exceeds a threshold η are linked by bidirectional edges injected into the factor graph.

- 7) Non-Keyframe Pose Estimation. Each non-keyframe is connected to its two nearest keyframes via unidirectional edges, and its pose is recovered through photometric alignment, bypassing per-frame depth estimation. All non-keyframe poses are estimated in parallel [5].
- 8) Open-Vocabulary Grounding. Real-time OpenVocabulary capabilities are achieved via decoding the compressed Radio features of 3D points and then projecting them into SigLip [31] latent space for matching with the text query embedding

B. Dense Visual Features Extraction

We present a dense spatial embedding pipeline that integrates and adapts existing components to achieve robust feature representation tailored for dynamic environments. Our approach employs RADSeg [6] as the core dense embedding model, while we use its agglomerative capabili-

[Figure 2]

Fig. 2. RADIO-ViPE pipeline

ties—grounded in RADIO [1]—to generate language-aligned features within the SigLIP [31] embedding space.

Furthermore, we operate RADSeg using a sliding-window approach, performing inference over overlapping image regions and subsequently refining the aggregated feature map through a self-attention mechanism. It allowed us to provide an optimal balance between spatial discriminability and semantic language alignment.

Unlike conventional approaches that operate on the language-aligned space, we perform PCA directly on the encoder feature space, thereby preserving structural integrity that is essential for spatial reasoning. To ensure a representative PCA mapping, we exploit the initialization period of our Bundle Adjustment (BA) optimization. By waiting until a sufficient buffer of keyframes is collected before computing the PCA components, we achieve a robust representation that balances computational efficiency with feature expressivity.

- C. Joint Bundle Adjustment

Let Ti ∈ SE(3) denote the world-to-camera pose for frame i, and Kq = [fx,fy,cx,cy]⊤ the camera intrinsics. RADIO-ViPE formulates dense SLAM as a factor graph optimization, jointly refining poses {Ti}, dense disparity maps {di}, and intrinsics {Kq} by minimizing a weighted vision-language-geometric energy function.

1) Dense Photometric Flow Term: Following DROIDSLAM [25], geometric consistency is enforced via dense optical flow constraints. For each edge (i,j) in G, pixels u in frame i are projected into frame j as:

µij = Πj TjT−i 1 ◦ Π−i 1(u, di(u)) , (2) where Πq and Π−q 1 denote the projection and unprojection functions under intrinsics Kq. An optical flow network [25] predicts a residual dense flow field Ωij ∈ RH×W×2 alongside per-pixel confidence weights w(u) conditioned on the relative poses of the edge’s frames. The prior flow estimate Ωpriorij = µij −u initializes a correlation volume for iterative refinement. The photometric term is then:

2

w(u) · Ωpriorij − Ωij(u)

, (3)

Ephoto =

u

which directly penalizes discrepancies between geometric reprojections and learned dense correspondences.

2) RADIO Embedding Similarity Term: To incorporate semantic consistency into the optimization, we introduce a novel embedding similarity term that enforces crossview feature alignment under geometric constraints, enabling RADIO-ViPE to leverage the rich multi-modal representations of RADIO [1], [6] directly within bundle adjustment.

For each edge (i,j), let Zi,Zj ∈ RK×H×W denote the PCA-compressed dense embedding maps. Each source pixel u ∈ Ii is projected into the target frame using current pose and depth estimates to locate its corresponding pixel v = Pi,j(u) ∈ Ij, then the target embedding is recovered via bilinear interpolation:

wlm(Pi,j(u)) · Zjlm (4)

Zˆj(Pi,j(u)) =

(l,m) ∈ N(Pi,j(u))

where N(Pi,j(u)) denotes the four nearest integer neighbours of Pi,j(u) = (u,v), with bilinear weights:

wlm(Pi,j(u)) ∈ αβ, α(1−β), (1−α)β, (1−α)(1−β) , α = u − ⌊u⌋, β = v − ⌊v⌋

After ℓ2-normalizing both embeddings, the cosine similarity between the source and projected target is:

Zi(u)⊤ Zˆj(Pi,j(u)) ∥Zi(u)∥ · ∥Zˆj(Pi,j(u))∥

, (5)

csij(u) =

The embedding residual is casted in photometric form to maintain consistency with the flow term:

rembed(u) = λembed 2 1 − csij(u) , (6)

with λembed=2, yielding residuals of comparable magnitude to intensity differences in classical photometric SLAM. The full embedding similarity term is:

Eembed =

u

w(u) · rembed2 (u) (7)

where w(u) incorporates photometric flow confidence.

- D. Temporally Consistent Adaptive Robust Kernel

One of the core contribution of our method is introducing adaptive robust kernels to improve SLAM robustness in dynamic environments. Since a pairwise kernel shaped solely by the cosine similarity csij(u) across a single edge (i,j) cannot distinguish genuinely static surfaces, movable objects, and actively moving agents — reliable classification requires aggregating evidence across multiple edges temporally.

1) Temporal Stability Field: Let N(i) denote all keyframes connected to keyframe i in G. Using the peredge cosine similarities csij(u) defined in Sec. III-C-2, we compute the temporal mean and variance at each pixel:

1 |N(i)|

csij(u),

cs¯ i(u) =

j∈N(i)

1 |N(i)|

csij(u) − cs¯ i(u) 2

σi2(u) =

j∈N(i)

The temporal stability field is then defined as:

Si(u) = cs¯ i(u) · 1 − σi2(u) ∈ [0, 1] (8)

Si(u) ≈ 1 indicates a genuinely static surface — consistently high similarity with low temporal variance. Si(u) ≈ 0 flags either active motion (low mean) or a displaced object (high variance), correctly identifying both as unreliable.

- 2) Three-Regime Barron Shape Mapping: We classify

pixels in real-world video to 3 physically distinct categories that require different loss regimes: genuinely static, movable objects by agent (e.g. in ego-centric session), and actively moving agents (e.g. people). So, we map Si(u) to the shape parameter α of the general Barron loss ρα [8], which recovers ℓ2 (α=2), Huber (α=1), and Cauchy (α→0) as special cases. We assign α via a differentiable piecewiselinear mapping:

 

2, Si(u) ≥ θs 1 + Si(u) − θm

, θm ≤ Si(u) < θs

αi(u) =

θs − θm

αdyn + Si(u) θm



1 − αdyn , Si(u) < θm

(9) with αdyn≤0, θs=0.75, and θm=0.35. The three regimes apply the correct loss family to each category, while keeping the shape parameter α continuous and differentiable. We then rewrite (3) as follows:

Ephotoark =

wark(Ephoto(u),αi) · Ephoto(u), (10)

u

1 max(r,ε)

∂ ρα(r) ∂r

, (11)

wark(r,α) =

where ε prevents division by zero, and ρα is the barron function [8](Fig. 3). The adaptive robust kernel thus jointly down-weights low-confidence and semantically dynamic regions via w and αij(u) respectively, while preserving full ℓ2 influence on static scene structure (See Fig. 3).

[Figure 3]

Fig. 3. Adaptive robust kernels based on Barron’s general loss [8]. The shape parameter α is governed by the temporal stability field Si(u), transitioning from ℓ2 (static surfaces) through Huber (movable objects) to Cauchy (actively moving agents).

E. Objective for Tightly-Coupled Optimization

Finally, we perform optimization with Gauss–Newton method to jointly calculate camera poses {Ti}, disparities {di}, and intrinsics {Kq} by minimizing the following complete objective function:

Etotal = γphotoEphotoarc + γembedEembed + Ereg | (12) Ereg(di) = αdisp

∥di(u) − dpriori (u)∥2, (13)

u

where Ereg is the regularization term with disparity dpriori from the foundation depth model and the regularization

weight αdisp (for experiments we set it to 1.0) that is introduced to stabilize depth estimates and incorporate prior knowledge from foundation depth models similarly to [5]. This term prevents drift while allowing the optimization to refine depth estimates based on multi-view consistency.

TABLE II SLAM PERFORMANCE COMPARISON ON TUM-RGBD IN CM (ATE ↓)

fr3/w/static

fr3/s/static

fr3/w/rpy

fr3/w/xyz

fr3/s/rpy

fr3/s/xyz

fr3/w/hs

Average

fr3/s/hs

Method

Dyna-SLAM [32] 1.64 3.54 2.96 0.68 1.27 – 1.86 – 2.00 DLD-SLAM [33] 1.85 4.24 2.19 0.56 – – – – 2.21 V3D-SLAM [34] 1.53 7.81 2.29 0.65 0.87 1.69 1.47 0.58 2.10 DGS-SLAM [35] 4.10 – 5.50 0.60 – – 4.40 – 3.65 RoDyn-SLAM [36] 8.30 – 5.60 1.70 – – 2.70 – 4.58 DynaMON [37] 1.4 3.9 2.0 1.4 0.9 2.1 1.9 0.5 1.76 ViPE (SAM) [5] 2.4 3.46 2.52 0.54 1.43 3.80 2.57 0.6 2.17 RADIO-ViPE 1.90 3.50 3.10 0.55 1.15 2.72 1.60 0.53 1.90 RADIO-ViPEark 1.55 3.39 1.96 0.50 0.98 2.65 1.44 0.56 1.63

Best Second best Third best – Not reported

TABLE III QUANTITATIVE COMPARISON ONREPLICA [24] FOLLOWING [23]. THE RIGHTMOST 4 COLUMNS INDICATE WHETHER A METHOD SUPPORTS ONLINE INFERENCE AND OPERATES WITHOUT CALIBRATION, DEPTH, OR POSE INPUTS.

Without Background With Background Calib Depth Pose Methods mIoU↑ f-mIoU↑ Acc↑ mIoU↑ f-mIoU↑ Acc↑ Online Free Free Free

ConceptFusion [38] 21.07 31.51 35.65 20.38 35.75 41.58 ✗ ✗ ✗ ✗ ConceptGraphs [17] 11.63 16.61 19.80 11.72 21.35 28.28 ✗ ✗ ✗ ✗ HOV-SG [18] 16.93 31.45 34.74 19.29 30.64 35.17 ✗ ✗ ✗ ✗ NACLIP-3D [39] 20.37 35.08 47.47 15.30 16.98 26.23 ✗ ✗ ✗ ✗ Trident-3D [40] 21.30 43.34 54.79 20.63 38.53 50.31 ✗ ✗ ✗ ✗ RayFronts [23] 39.37 62.03 68.80 27.73 43.37 54.45 ✗ ✗ ✗

RADIO-ViPEGT 29.51 52.24 59.80 28.19 54.44 65.21 ✗ ✗ ✗ RADIO-ViPE 24.25 50.63 59.25 19.00 37.13 48.38

Best Second best Third best

IV. EXPERIMENTS

We evaluate RADIO-ViPE on open-vocabulary semantic segmentation on Replica [24] and robustness in dynamic environments on TUM-RGBD [2]. We denote the variant using only the embedding error term as RADIO-ViPE, and the full pipeline with adaptive kernel tuning as RADIO-ViPEark. All experiments are conducted on Intel Xeon Gold 5320 CPU 2.20 GHz with NVIDIA GeForce RTX 4090.

- A. SLAM Performance

Our embedding error term and adaptive robust kernel improve SLAM in dynamic environments by compensating for or attenuating incorrect data associations caused by object motion. Experiments on the dynamic TUM RGB-D [2] sequences show best average ATE, outperforming existing dynamic SLAM methods. Unlike ViPE [5], which relies on foundation models (Grounding DINO [41], SAM [42]) and manual class specification for dynamic masking, our pipeline achieves better performance with substantially fewer resources.

- B. 3D Semantic Segmentation

A core feature of our approach is the ability to construct a semantically-aware map during real-time operation. To assess this capability, we evaluate our semantic segmentation performance on the Replica dataset [24]. We adopt D=256 as the default PCA encoded dimension of RADIO features, offering an effective trade-off between memory efficiency and segmentation performance (Fig. 4). Unlike RayFronts [23], ConceptFusion [38], ConceptGraphs [17], HOV-SG [18], NACLIP-3D [39], and Trident-3D [40], RADIO-ViPE operates without known camera parameters, poses, point cloud, or depth. Furthermore, except for RayFronts [23], none of these works support realtime operation;

We follow the evaluation protocol of RayFronts [23] on eight Replica scenes (office0--4, room0--2); results are reported in Table III. Our method ranks in the top-

- 3 among all compared approaches. The key observation is

the small gap between RADIO-ViPE and RADIO-ViPEGT: removing ground-truth depth, pose, and calibration costs only

[Figure 4]

- Fig. 4. Ablation study on RADIO PCA feature dimensionality for semantic mapping on the Replica dataset. D=256 (—•—) closely matches the full-dimensional baseline (∆mIoU < 1%).

[Figure 5]

- Fig. 5. Quantitative results of RADIO-ViPE on Replica with different text queries.

∼1–2% in f-mIoU and Acc (without-background setting), confirming that our method retains most of its accuracy without any geometric supervision. Performance degrades more noticeably in the with-background setting, which we attribute to the difficulty of segmenting structural classes (e.g., wall, floor); we consider this a current limitation.

V. CONCLUSION

RADIO-ViPE is a robust, online semantic SLAM system that operates without calibration and handles dynamic scenes. Leveraging RADSeg features, it achieves competitive performance on TUM-RGBD [2] and ranks top-3 on Replica despite PCA-based memory compression—demonstrating a strong performance-to-efficiency trade-off suitable for realtime robotics, AR/VR, and unconstrained in-the-wild video.

REFERENCES

- [1] M. Ranzinger and G. e. a. Heinrich, “Am-radio: Agglomerative vision foundation model reduce all domains into one,” in IEEE/CVF Conference on Computer Vision and Pattern Recognition, June 2024, pp. 12490–12500.
- [2] J. Sturm and e. a. N. Engelhard, “A benchmark for the evaluation of rgb-d slam systems,” in The International Conference on Intelligent Robot Systems, October 2012.
- [3] K. Grauman, A. Westbury, and E. a. Byrne, “Ego4d: Around the World in 3,000 Hours of Egocentric Video,” in IEEE/CVF Computer Vision and Pattern Recognition (CVPR), 2022.
- [4] X. Pan and e. a. Charron, Nicholas, “Aria digital twin: A new benchmark dataset for egocentric 3d machine perception,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2023, pp. 20133–20143.
- [5] J. Huang and e. a. Zhou, “Vipe: Video pose engine for 3d geometric perception,” in NVIDIA Research Whitepapers arXiv:2508.10934, 2025.
- [6] O. Alama and e. a. Jariwala, Darshil, “Radseg: Unleashing parameter and compute efficient zero-shot open-vocabulary segmentation using agglomerative models,” arXiv preprint arXiv:2511.19704, 2025.
- [7] J. Mahmoud, A. Penkovskiy, H. T. Long Vuong, A. Burkov, and S. Kolyubin, “Rvwo: A robust visual-wheel slam system for mobile robots in dynamic environments,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2023, pp. 3468–3474.
- [8] J. T. Barron, “A general and adaptive robust loss function,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 4326–4334.
- [9] R. Murai, E. Dexheimer, and A. J. Davison, “Mast3r-slam: Realtime dense slam with 3d reconstruction priors,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 16695–16705.
- [10] D. Maggio, H. Lim, and L. Carlone, “Vggt-slam: Dense rgb slam optimized on the sl (4) manifold,” arXiv preprint arXiv:2505.12549, 2025.
- [11] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “Dust3r: Geometric 3d vision made easy,” in CVPR, 2024.
- [12] E. a. Campos, Carlos, “Orb-slam3: An accurate open-source library for visual, visual–inertial, and multimap slam,” IEEE Transactions on Robotics, vol. 37, no. 6, p. 1874–1890, Dec. 2021.
- [13] A. Rosinol and e. a. Abate, Marcus, “Kimera: an open-source library for real-time metric-semantic localization and mapping,” in 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020, pp. 1689–1696.
- [14] Z. C. et al, “Rgbds-slam: A rgb-d semantic dense slam based on 3d multi level pyramid gaussian splatting,” 2024.
- [15] X. Chen, T. Wang, H. Mai, and L. Yang, “Samslam: A visual slam based on segment anything model for dynamic environment,” in 2024 8th International Conference on Robotics, Control and Automation (ICRCA), 2024, pp. 91–97.
- [16] S. Linok, T. Zemskova, S. Ladanova, R. Titkov, D. Yudin, M. Monastyrny, and A. Valenkov, “Beyond bare queries: Openvocabulary object grounding with 3d scene graph,” 2024.
- [17] Q. Gu and E. a. Alihusein Kuwajerwala, “Conceptgraphs: Openvocabulary 3d scene graphs for perception and planning,” 2023.
- [18] A. Werby and e. a. Huang, Chenguang, “Hierarchical open-vocabulary 3d scene graphs for language-grounded robot navigation,” in Robotics: Science and Systems XX, ser. RSS2024. Robotics: Science and Systems Foundation, July 2024.
- [19] S. Peng, K. Genova, C. M. Jiang, A. Tagliasacchi, M. Pollefeys, and T. Funkhouser, “Openscene: 3d scene understanding with open vocabularies,” 2023.

- [20] A. Takmaz, E. Fedele, R. W. Sumner, M. Pollefeys, F. Tombari, and F. Engelmann, “Openmask3d: Open-vocabulary 3d instance segmentation,” 2023.
- [21] D. Maggio, Y. Chang, N. Hughes, M. Trang, D. Griffith, C. Dougherty, E. Cristofalo, L. Schmid, and L. Carlone, “Clio: Real-time task-driven open-set 3d scene graphs,” 2024.
- [22] T. B. Martins, M. R. Oswald, and J. Civera, “Ovo-slam: Openvocabulary online simultaneous localization and mapping,” 2024.
- [23] O. Alama and e. a. Bhattacharya, Avigyan, “Rayfronts: Open-set semantic ray frontiers for online scene understanding and exploration,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2025, pp. 5930–5937.
- [24] J. Straub and E. a. Thomas Whelan, “The replica dataset: A digital replica of indoor spaces,” 2019.
- [25] Z. Teed and J. Deng, “DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras,” Advances in neural information processing systems, 2021.
- [26] L. e. a. Piccinelli, “Unidepth: Universal monocular metric depth estimation,” in 2024 (CVPR), 2024, pp. 10106–10116.
- [27] R. e. a. Wang, “Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision,” in (CVPR), 2025, pp. 5261–5271.
- [28] N. e. a. Chebrolu, “Adaptive robust kernels for non-linear least squares problems,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2240–2247, 2021.
- [29] A. V. et al, “GeoCalib: Single-image Calibration with Geometric Optimization,” in ECCV, 2024.
- [30] M. Hu, W. Yin, and e. a. Zhang, “Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
- [31] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11975–11986.
- [32] B. Bescos, C. Campos, J. D. Tard´os, and J. Neira, “Dynaslam ii: Tightly-coupled multi-object tracking and slam,” 2020. [Online]. Available: https://arxiv.org/abs/2010.07820
- [33] H. Yu, Q. Wang, C. Yan, Y. Feng, Y. Sun, and L. Li, “Dld-slam: Rgbd visual simultaneous localisation and mapping in indoor dynamic environments based on deep learning,” Remote Sensing, vol. 16, no. 2, 2024.
- [34] T. Dang, K. Nguyen, and M. Huber, “V3d-slam: Robust rgb-d slam in dynamic environments with 3d semantic geometry voting,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7847–7853.
- [35] L. Yan, X. Hu, and e. a. Zhao, “Dgs-slam: A fast and robust rgbd slam in dynamic environments combined by geometric and semantic information,” Remote Sensing, vol. 14, no. 3, p. 795, 2022.
- [36] H. Jiang, Y. Xu, K. Li, J. Feng, and L. Zhang, “Rodyn-slam: Robust dynamic dense rgb-d slam with neural radiance fields,” 2024.
- [37] N. Schischka and e. a. Schieber, “Dynamon: Motion-aware fast and robust camera localization for dynamic neural radiance fields,” IEEE Robotics and Automation Letters, pp. 1–8, 2024.
- [38] e. a. Jatavallabhula, “Conceptfusion: Open-set multimodal 3d mapping,” Robotics: Science and Systems (RSS), 2023.
- [39] S. Hajimiri, I. Ben Ayed, and J. Dolz, “Pay attention to your neighbours: Training-free open-vocabulary semantic segmentation,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2025.
- [40] Y. Shi, M. Dong, and C. Xu, “Harnessing vision foundation models for high-performance, training-free open vocabulary segmentation,” arXiv preprint arXiv:2411.09219, 2024.
- [41] S. Liu, Z. Zeng, and e. a. Ren, “Grounding dino: Marrying dino with grounded pre-training for open-set object detection,” in European conference on computer vision. Springer, 2024, pp. 38–55.
- [42] E. a. Alexander Kirillov, Eric Mintun, “Segment anything,” 2023.

