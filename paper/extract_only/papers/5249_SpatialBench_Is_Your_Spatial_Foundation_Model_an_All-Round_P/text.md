[Figure 1]

[Figure 2]

# SpatialBench

### Is Your Spatial Foundation Model an All-Round Player?

###### Haosong Peng1★ Hao Li23★‡ ★ Jiaqi Chen3★ Yuhao Pan1★ Runmao Yao2 ★ Yalun Dai2 Fushuo Huo4 Fangzhou Hong2 ★ Zhaoxi Chen2 ★ Haozhao Wang5 Dingwen Zhang3 Ziwei Liu2 ★ Wenchao Xu1†

1Hong Kong University of Science and Technology 2Nanyang Technological University ★Ropedia 3Northwestern Polytechnical University 4Southeast University 5Huazhong University of Science and Technology

## arXiv:2605.27367v2[cs.CV]29May2026

Project Page: ropedia.github.io/SpatialBench Dataset: ropedia-ai/DA-Next-5M Code: github.com/Ropedia/SpatialBench Model: ropedia-ai/DA-Next

[Figure 3]

[Figure 4]

[Figure 5]

###### Why a new benchmark?

###### What SpatialBench Reveals?

[Figure 6]

[Figure 7]

Spatial Intelligence operates from diverse inputs with various domains.

[Figure 8]

Full Context Attention Sets the accuracy ceiling on highmemory GPUs.

- 1

- 2

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### 19 Datasets 546 Scenes 41 Models 6 Paradigms

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Bounded Mem. Modeling enables longsequence reconstruction on limited GPU memory.

Room

###### Deterministic Multi-Density Sampling

[Figure 18]

[Figure 19]

[Figure 20]

Single Sparse Medium Dense

[Figure 21]

Roaming

[Figure 22]

(a) For Model Design

(b) For Data Curation

[Figure 23]

###### Diversity, Fairness, and Reproducibility

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

###### Compared w/ Existing Bench

E3D-Bench Ours

Driving

[Figure 29]

- 2 6

1 4

- 3 5

Broader Model Paradigms

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Reproducible protocol

Random Fixed

[Figure 34]

Sequence-Density Scaling

Ego-View

[Figure 35]

3 4

DATA quality v.s. Quantity Training data quality outweighs data volume.

OOD in Ego / Embodied Domain match matters more than dataset count.

More Test Domains

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Domain Converge

[Figure 41]

[Figure 42]

Room

Roaming Driving Ego-View Wrist-View

###### DA-Next & DA-Next-5M Dataset

Wrist-View

Task Suites

[Figure 43]

[Figure 44]

[Figure 45]

5.5M photorealistic, high quality frames

[Figure 46]

Depth Pose Prior-Inputs Point Cloud Trajectory

Fixed Indexes

Various Domains

[Figure 47]

[Figure 48]

Model Paradigms

across 22K egocentric and wrist-view scenes

[Figure 49]

[Figure 50]

Optimization End-to-End SLAM Online Chunk-Wise TTT

Figure 1 | SpatialBench provides a reproducible, cross-paradigm benchmark spanning 19 datasets, 546 scenes, 41 models, and 6 paradigms under deterministic multi-density sampling. Our analysis reveals insights on model design, domain generalization, data curation, and beyond, complemented by DA-Next and DA-Next-5M to address the embodied domain gap.

### Abstract

[Figure 51]

While spatial foundation models have demonstrated impressive performance on standard datasets, a critical question remains: are they truly all-round players capable of generalizing robustly across diverse downstream tasks, arbitrary viewpoints, shifting scene domains, varying input densities, and specific hardware constraints? Answering this overarching question requires a holistic assessment, yet current models are

★ Core contributors. ‡ Project lead. † Corresponding author.

mainly evaluated on specific domains for which they were specifically designed or trained. Such evaluations are intrinsically limited by narrow paradigm coverage, limited scene domains, and arbitrary frame sampling, making it fundamentally difficult to assess their true generalization capabilities. To address this gap, we present SpatialBench, a cross-paradigm, domain-diverse benchmark for spatial foundation models with deterministic sampling. SpatialBench features unprecedented scale and rigorous deterministic design, comprising 19 datasets and 546 scenes across 5 diverse spatial domains. It comprehensively evaluates 41 models across 6 paradigms on 5 task suites under 4 different input density settings. Our extensive evaluation reveals that current models are not yet all-round players, and uncovers crucial insights for future advancement. Specifically, we demonstrate that full-context attention maximizes accuracy while bounded-memory strategies unlock long-sequence scalability. Moreover, our empirical evaluations in challenging embodied and egocentric tasks demonstrate that strict domain alignment and high data quality are far more critical to performance than simple dataset scaling. Furthermore, to address the largest data gap identified in our analysis, we go beyond evaluation by introducing a large-scale dataset, DA-Next-5M, and a strong baseline model, DA-Next, pushing the boundaries of spatial representation learning.

#### 1. Introduction

Spatial foundation models have already been widely deployed across robotics [117, 56, 128], AR/VR [119], autonomous driving [15], and embodied AI [41, 130]. This extensive adoption is driven by their remarkable ability to recover accurate 3D structures from mere images or videos, establishing them as general-purpose visual geometry backbones for spatial intelligence. However, operating in these real-world applications is inherently chaotic and far more demanding than standard reconstruction benchmarks. To truly support these downstream tasks, a robust model must maintain its reliability when confronting unpredictable scene domain shifts, highly variable sparse-to-dense input regimes, and strict hardware memory constraints. This raises the central question of this work: if spatial foundation models are expected to support general-purpose spatial intelligence, can they truly serve as robust all-round players across the diverse conditions of the 3D world?

However, existing evaluations fall short in several critical aspects. First, they cover only a narrow slice of today’s model paradigms. Spatial foundation models now span feed-forward [94, 42, 54], optimization-based [99, 48, 125], streaming [136, 46, 13], SLAM-based [62, 66], chunkbased [23], and test-time training (TTT) [16, 113, 126] approaches, yet most benchmarks evaluate only one or a few of them under separate protocols. Second, current comparisons are often not standardized. Even when papers report results on the same dataset, they may use different scene splits, private subsets, frame indices, temporal windows, or input densities, making direct comparison ambiguous. Third, existing protocols rarely expose how models scale with sequence density. For example, a model that works well on sparse image sets may fail on dense long videos because of memory growth, accumulated drift, or degraded global consistency; conversely, bounded-memory methods (e.g., online, chunk-wise, and TTT) may be undervalued when evaluation is restricted to short sequences. Finally, test domains remain too limited for assessing real-world spatial intelligence. Standard indoor or object-centric reconstruction datasets do not capture the diversity of robotics, autonomous driving, egocentric perception, and

wrist-mounted manipulation settings. These gaps motivate a benchmark that is cross-paradigm, deterministic, density-aware, and domain-diverse: one that can fairly compare models, reveal how performance changes from sparse views to dense streams, and diagnose where current spatial foundation models succeed or fail.

To address the aforementioned challenges, we introduce SpatialBench. By incorporating a deterministic density-aware protocol, broad domain diversity, and cross-paradigm comparisons, this comprehensive benchmark serves as a beacon to guide and verify spatial foundation models toward becoming true all-round players. SpatialBench is built around three core design principles: (1) Deterministic Multi-Density Evaluation Protocol. To systematically assess model robustness across varying input scales, SpatialBench adopts a deterministic sampling strategy to precompute frame indices across 4 distinct density regimes: single-frame, sparse, medium, and dense. By evaluating each scene under these standardized configurations across several key metrics, our protocol ensures both a comprehensive understanding of model performance and full reproducibility across different paradigms. (2) Broad Domain Coverage Across 19 Datasets. SpatialBench aggregates 19 datasets and 546 scenes in total, spanning a comprehensive range of conditions, including indoor and outdoor environments, static and dynamic scenes, real-world and synthetic data, and diverse viewpoint types. Each scene is annotated with orthogonal tags along these axes, enabling fine-grained cross-domain filtering and aggregation, supporting over 100 distinct evaluation configurations that far exceed any existing benchmark. (3) Comprehensive and Cross-Paradigm Model Comparison. SpatialBench provides unified adapters for 31 state-ofthe-art models and 41 variants in total, spanning all six reconstruction paradigms: optimizationbased, end-to-end feed-forward, online streaming, chunk-wise, TTT-based, and SLAM-based systems. All methods are evaluated under a unified protocol, enabling fair and direct comparison across several geometric tasks, including depth and camera pose estimation, reconstruction, prior-enhanced geometry prediction, and trajectory estimation.

We further conduct extensive analysis experiments on SpatialBench, revealing several key insights: (1) Full-context attention defines the accuracy upper bound, with globally coupled feed-forward models consistently outperforming bounded-memory approaches under the same input budget. (2) Bounded-memory models unlock long-horizon scalability, enabling continuous reconstruction beyond the memory limits of full-context models, at the cost of geometry estimation accuracy. (3) Data quality outweighs data volume, as carefully curated pseudo-GT supervision consistently outperforms larger but noisier training mixtures. (4) Egocentric and wrist-view domains remain the dominant OOD failure modes, exposing a field-level gap that cannot be addressed by scaling existing training mixtures alone.

To further address the gap in egocentric and wrist-view domains, we curate DA-Next-5M, a dataset comprising 22K scenes with 5.5M frames of 3D data in total from egocentric and robot wrist-view sources. We train our proposed Depth-Anything-Next (DA-Next) on DA-Next-5M, establishing a strong domain-specific baseline for these underexplored viewpoints. The key contributions of our work are summarized as follows.

- • SpatialBench is the first standardized benchmark for comprehensive evaluation of 3D spatial foundation models on several geometry tasks, aggregating 19 diverse datasets and 546 scenes, and providing unified adapters for 32 methods and 41 variants across all six paradigms.
- • Through extensive experiments on SpatialBench, we conduct a comprehensive crossparadigm analysis and derive key insights into model robustness, domain generalization, and input-density scaling behavior, highlighting promising directions for future research.

- • Experiments show that DA-Next achieves substantial gains over DA3-Giant: +47%/+59% in depth estimation and +3.1%/+5.5% in pose estimation on sparse/medium inputs, demonstrating that targeted in-domain data curation effectively closes the embodied domain gap.

#### 2. Related Work

Spatial foundation models for visual geometry. Recent advances in visual geometry have shifted 3D reconstruction from optimization-heavy pipelines toward spatial foundation models that directly infer scene geometry, camera parameters, and point cloud from images. Early influential systems such as DUSt3R [99] and MASt3R [48] reformulate geometric reconstruction as dense pointmap prediction, substantially simplifying pose-free 3D reconstruction and 3D-grounded image matching. Although these methods still rely on global alignment or optimization-based post-processing, they establish a strong foundation for subsequent feedforward reconstruction models. Building on this direction, end-to-end feed-forward methods aim to recover visual geometry in a single network pass. VGGT [94] predicts camera parameters, depth maps, pointmaps, and point tracks in a unified transformer framework, while Fast3R [116] scales feed-forward reconstruction to large unordered image collections and FastVGGT [82] accelerates VGGT-style inference without retraining. MUSt3R [9] extends stereo-style reconstruction to multi-view settings, and MapAnything [42] supports flexible geometric inputs such as poses, depths, intrinsics, and partial reconstructions for universal metric 3D reconstruction. More model families further expand this paradigm: OmniVGGT [71] incorporates omni-modality prior for reconstruction; 𝜋3 removes the dependence on reference frames through a fully permutation-equivariant architecture, predicting affine-invariant camera poses, and scale-invariant local pointmaps; AMB3R [93] introduces a backend module for more accurate metric-scale reconstruction; DA3 and its variants [54] recover consistent 3D geometry from arbitrary visual inputs across multiple model scales; and WorldMirror [59] explores any-prior prompting to unify diverse 3D representations. Together, these feed-forward spatial foundation models demonstrate strong reconstruction capability on bounded image sets, but their performance can degrade when applied to long videos, streaming inputs, or large-scale scenes where memory, consistency, and drift become critical. Moreover, processing long sequences with these models incurs prohibitive GPU memory consumption and increased inference latency.

Long-sequence, online, and test-time training models. To handle realistic video streams, recent work has extended spatial foundation models from bounded image sets to online, chunkwise, SLAM-based, and test-time adaptive settings. Online and streaming methods maintain temporal or spatial memory, recurrent states, or compact historical context as new frames arrive. Spann3R [92] introduces spatial memory for incremental 3D reconstruction, CUT3R [98] uses a persistent recurrent state for continuous 3D perception, and MonST3R [125] extends DUSt3R-style reconstruction to dynamic scenes with motion. Point3R [107] employs explicit spatial pointer memory for streaming reconstruction, while Stream3R [46], StreamVGGT [136], Page4D [133], InfiniteVGGT [123], WinT3R [52], LongStream [17], and LingBot-Map [13] investigate different memory mechanisms, window designs, causal attention strategies, and long-horizon update rules for scalable online geometry estimation. Another line processes long videos in chunks and then aligns local reconstructions into a global scene. VGGT-Long [23], 𝜋3-Long [23], and DA3-Streaming [23] follow this chunk-wise strategy to extend powerful feed-forward backbones or model variants to kilometer-scale or long-sequence reconstruction. In parallel, SLAM-based systems such as MASt3R-SLAM [66] and VGGT-SLAM [62] combine learned 3D priors with classical mapping and tracking components to improve real-time dense reconstruction. Finally, test-time training methods, including TTT3R [16], Scal3R [113],

[Figure 52]

- Figure 2 | Overview of SpatialBench. (Left) Distribution of all scene categories and their corresponding counts. (Right) Data sources and the median number of frames per scene under sparse, medium, and dense input settings. The number on each circle indicates the number of scenes.

ZipMap [39] and LoGeR [126], adapt the model or scene representation during inference to improve large-scale consistency and reduce drift. These methods reveal an emerging trend: the central challenge of spatial foundation models is no longer only accurate single-shot reconstruction, but also scalable memory management, temporal consistency, dynamic-scene robustness, and long-range geometric alignment under realistic visual streams.

Related benchmarks for visual geometry. Several recent efforts have introduced systematic benchmarks for 3D reconstruction and visual geometry. Robust MVD [81] focuses on cross-dataset generalization for multi-view depth estimation, while E3D-Bench [19] provides a broader evaluation covering depth, reconstruction, pose estimation, and novel-view synthesis. In addition, several model works construct their own evaluation protocols, including DA3 [54], 𝜋3 [101], and MapAnything [42]. Among these, E3D-Bench is the most comprehensive standalone effort, supporting cross-method comparison across multiple tasks. However, it does not provide comparisons across various domains and paradigms. The remaining model-specific suites are largely tied to individual model studies and lack a unified protocol for controlled cross-paradigm comparison. In contrast, SpatialBench provides a standalone, deterministic, and tag-aware benchmark that enables systematic analysis across diverse input densities, viewpoint types, scene dynamics, and foundation model paradigms.

#### 3. SpatialBench Design

SpatialBench is built upon a large-scale collection of heterogeneous 3D vision datasets, covering a diverse spectrum of scene categories, capture conditions, and viewpoint configurations. Fig. 2 provides an overview of SpatialBench: the left panel shows the breakdown of scene categories and their corresponding counts, and the right panel reports the data sources alongside the median number of frames per scene across different settings. This multi-dimensional design allows SpatialBench to evaluate model capabilities across a wide range of conditions in a principled and systematic way.

Mask Propagate

[Figure 53]

[Figure 54]

[Figure 55]

| | |[Figure 56]|[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]<br><br>[Figure 61]| |
|---|---|---|---|---|

|[Figure 62]|
|---|

|[Figure 63]|
|---|

|[Figure 64]|
|---|

|[Figure 65]|
|---|

SAM 3

|[Figure 66]|
|---|

|[Figure 67]|
|---|

|[Figure 68]|
|---|

|[Figure 69]|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Depth&Photometric

[Figure 75]

[Figure 76]

High-confidence Depths

BundleAdjustment

[Figure 77]

[Figure 78]

RGBs

[Figure 79]

MapAnything

Refined Camera Poses

Baseline

Intrinsic

[Figure 80]

…

[Figure 81]

[Figure 82]

Depth Prior

Left Views

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

###### S2M2

Init Cam Poses

…

…

[Figure 91]

Right Views Metric Left Depths

Aligned Point Cloud

- Figure 3 | DROID data curation pipeline. We obtain metric depth from stereo sequences via S2M2 [65], with initial camera poses estimated by MapAnything [42]. Gripper and contact affordance masks are segmented using SAM3. Then the camera poses are refined via bundle adjustment using the RGBs, initial camera poses, and masks.

###### 3.1. Data Collection and Curation

SpatialBench unifies heterogeneous 3D vision datasets under a common, deterministic evaluation protocol. Raw datasets are first normalized into a shared per-scene representation comprising RGB frames, metric depth maps, camera-to-world poses, and camera intrinsics, and are subsequently curated into a fixed set of evaluation scene indices. Each scene index is stored as a JSON record that specifies, for every (scene, view-density) pair, the exact frame indices to be consumed by a method. By decoupling data ingestion from evaluation, this design ensures that all methods are assessed on identical inputs and that results remain fully reproducible across repeated runs.

We aggregate 19 publicly available real-world and synthetic datasets, spanning the principal axes relevant to modern 3D perception: environment (indoor/outdoor), dynamics (static/dynamic), viewpoint (normal/egocentric/wrist), and data type (real/synthetic). For example, the whole dataset can be classified into four distinct subsets according to the dynamics and data type axes: static-real, static-synthetic, dynamic-real, and dynamic-synthetic: (1) Static-real. 7Scenes [83], DTU [37], NRGBD [4], Scannet++ [121], Tanks & Temples [44], and ETH3D [80] provide high-quality ground-truth geometry under static conditions, covering settings from close-range tabletop scans to large-scale outdoor architecture. (2) Static-synthetic. Hiroom [54]. (3) Dynamic-real. TUM-Dynamic [86], DROID [43], Xperience [78], Waymo [87], and KITTIOdometry [26] capture dynamic indoor activities as well as street-scale driving scenarios. (4) Dynamic-synthetic. ADT [69], RLBench [36] with Colosseum [72], RoboTwin [14], Robolab [118], Virtual KITTI 2 [8], and OmniWorld-Game [135] provide dense photorealistic sequences for dynamic and robotic settings that are otherwise costly to acquire in the real world. We also collect a Single-frame Mixture including Lingbot-Depth [88] and all the above datasets, which contributes one-shot rgb/depth/intrinsic triplets, used exclusively in the monocular depth evaluation. We refer the reader to Tab. 4 in Appendix B.1 for a complete overview of all datasets included in SpatialBench.

To obtain high-quality, depth-consistent real-world wrist-view sequences, we design a dedicated data curation pipeline for the DROID [43] dataset, as illustrated in Fig. 3. We feed stereo video sequences into the S2M2 [65] stereo depth estimation model to obtain per-frame metric depth for the left image, with unreliable points filtered out via confidence thresholding. The

resulting image sequence and metric depth maps are then passed to MapAnything [42] to obtain initial camera poses. In parallel, we apply SAM3 [10] to segment dynamic regions, including the gripper and objects it interacts with, on a set of keyframes, and propagate the masks to the full sequence. These masks exclude dynamic foreground regions from the Bundle Adjustment optimization, which assumes a static scene background. Finally, leveraging the initial camera poses along with RGB images and the obtained masks, we perform depth & photometric bundle adjustment to refine the camera poses, yielding globally aligned point clouds. Other data pipeline and implementation details are provided in Appendix A.

###### 3.2. Multi-density Evaluation Regimes

A central principle of our benchmark is that each method is evaluated across multiple temporal resolutions on the same scene, rather than on arbitrarily truncated clips. From each curated scene, we generate four parallel entries corresponding to distinct view-density regimes: SINGLE, SPARSE, MEDIUM, and DENSE. These regimes are designed to probe complementary failure modes: SINGLE isolates monocular depth priors; SPARSE stresses wide-baseline reconstruction from unordered views; MEDIUM reflects the moderate-overlap inputs typical of SfM and SLAM; and DENSE evaluates long-horizon online estimation. Because all four regimes are derived from the same underlying scene, scene difficulty and density-related difficulty can be disentangled.

SINGLE. For each scene, we fix a single deterministic frame index that is consistent across all evaluations. This ensures that frame selection is reproducible across machines and independent of wall-clock time.

SPARSE. Sparse-view selection is formulated as a weighted set-cover problem over the scene’s 3D voxel support. Let V denote the set of all voxels in the scene, and let F denote the candidate frames. Each frame 𝑓 ∈ F covers a subset of voxels 𝑉𝑓 ⊆ V. We greedily select frames to maximize cumulative voxel coverage until a small frame budget 𝐾 is reached. This deterministic procedure promotes viewpoint diversity and is robust to variations in trajectory speed, producing a compact set of views that jointly covers the scene rather than merely temporally distant frames. The full selection objective is given in Appendix B.2.

MEDIUM. The medium regime retains the set-cover formulation of the sparse regime but favors view overlap over diversity. Let V be the coarsened voxel set and F the candidate frames. The selected frame set S is obtained by greedily maximizing voxel coverage, subject to a length-adaptive frame budget. This procedure encourages mid-range overlap among views rather than extreme viewpoint diversity. The corresponding budgeted objective is detailed in Appendix B.2.

DENSE. The dense regime targets the opposite end of the spectrum: an online, long-horizon setting in which a method must ingest essentially every frame of a trajectory and reconstruct temporally coherent geometry. Accordingly, the goal is not to select views in the set-cover sense, but to preserve temporal continuity while avoiding the trivial inflation of evaluation cost caused by near-duplicate consecutive frames. In practice, however, processing arbitrarily long trajectories can exceed the memory limits of some methods; therefore, we impose a maximum frame budget to ensure that the dense evaluation remains feasible across all methods.

###### 3.3. Evaluated Models

In this work, we evaluate 31 methods with 41 model variants in total. We categorize the evaluated methods into six paradigms: 1) Optimization-based methods, including DUSt3R [99] and MASt3R [48]; 2) End-to-End Feed-Forward methods, including VGGT [94], Fast3R [116],

FastVGGT [82], MUSt3R [9], MapAnything [42], OmniVGGT [71], 𝜋3 [101], AMB3R [93], DepthAnything3 [54], WorldMirror [59], and VGGT-Omega [95]; 3) Online/Streaming methods, including Spann3r [92], CUT3R [98], MonST3R [125], Point3R [107], Stream3R [46], StreamVGGT [136], PAGE4D [133], InfiniteVGGT [123], WinT3R [52], LongStream [17] and LingBot-Map [13]; 4) Chunk-based methods, including VGGT-Long [23], Pi-Long [23] and DA3Streaming [23]; 5) SLAM-based methods, including MASt3R-SLAM [66] and VGGT-SLAM [61]; and 6) Test-Time Training methods, including TTT3R [16], Scal3R [113] and LoGeR [126]. All evaluated models are listed in Table 1.

###### 3.4. Task Description and Metrics

To comprehensively assess the capabilities of each model, we design five general evaluation tasks across different settings as follows. Appendix B.4 reports complete metric definitions.

Camera Pose Estimation. Given an input image sequence, we evaluate pairwise camera geometry using Relative Rotation Accuracy (RRA) and Relative Translation Accuracy (RTA), following Wang et al. [94]. We use RAcc𝑥 and TAcc𝑥 to denote the fraction of pairs with angular errors below threshold 𝑥, and AUC𝑥 as the area under the joint accuracy curve of RRA and RTA up to threshold 𝑥.

Camera Trajectory Estimation. For continuous image sequences, i.e., our medium and dense input settings, we additionally compute Absolute Trajectory Error (ATE), Relative Translation Error (RPE𝑡), and Relative Rotation Error (RPE𝑟), all evaluated after global Sim(3) alignment with the ground-truth trajectory, following Teed and Deng [89].

Depth Estimation. For single/multi-view depth estimation, we compute AbsRel, SqRel, RMSE, LogRMSE, and threshold (inlier, 𝛿𝜏) metrics over all valid pixels with respect to the groundtruth depth, where the inlier ratio indicates the percentage of pixels with correct predictions. Predicted depths are aligned to the ground truth via median alignment by default. For models with metric-scale prediction capability, we additionally report AbsRel both before and after alignment.

Dense-View Reconstruction. We evaluate scene-level 3D reconstruction on a subset of scenes. Given the ground-truth and reconstructed point sets, we compute Accuracy and Completeness, and report the F-score (harmonic mean of Precision and Recall) and the Overall score, defined as (Accuracy + Completeness) / 2, following Lin et al. [54].

Prior-Enhanced Prediction. This task targets methods that accept auxiliary prior inputs (e.g., MapAnything, OmniVGGT [42, 71]). We evaluate their performance under two settings: with all ground-truth depth priors provided, and with all ground-truth camera pose priors provided.

#### 4. Depth-Anything-Next

To fill the gap of 3D foundation models in the egocentric and wrist-view domains, we introduce Depth-Anything-Next and our DA-Next-5M dataset in this section.

###### 4.1. DA-Next-5M Dataset

We curate DA-Next-5M, a large-scale 3D dataset comprising 5.5M high-quality frames across 22K scenes, primarily collected from egocentric and wrist-view perspectives. This category of views presents unique challenges, including high motion dynamics, frequent occlusions, and ultra-close-range capture, making it a critical data paradigm for embodied intelligence

ADT HOI4D RoboLab RLBench RoboTwin Colosseum

Xperience

|[Figure 92]<br><br>[Figure 93]|
|---|

|[Figure 94]<br><br>[Figure 95]|
|---|

|[Figure 96]<br><br>[Figure 97]|
|---|

|[Figure 98]<br><br>[Figure 99]|
|---|

|[Figure 100]<br><br>[Figure 101]|
|---|

|[Figure 102]<br><br>[Figure 103]|
|---|

|[Figure 104]<br><br>[Figure 105]|
|---|

|[Figure 106]<br><br>[Figure 107]|
|---|

|[Figure 108]<br><br>[Figure 109]|
|---|

|[Figure 110]<br><br>[Figure 111]|
|---|

|[Figure 112]<br><br>[Figure 113]|
|---|

|[Figure 114]<br><br>[Figure 115]|
|---|

|[Figure 116]<br><br>[Figure 117]|
|---|

|[Figure 118]<br><br>[Figure 119]|
|---|

Figure 4 | DA-Next-5M Data Samples. The dataset showcases a diverse array of assets and episodes.

applications. Tab. 7 presents the statistics of DA-Next-5M, where all datasets provide image sequences, metric depths, camera intrinsics, and extrinsics. For simulation-based datasets, extensive domain randomization is applied to various factors, including background appearance, object size, color, and wrist camera placement. Please refer to Appendix C for the data collection pipeline.

###### 4.2. Model Architecture

[Figure 120]

[Figure 121]

Depth-Anything-Next builds upon the architecture of DA3 [54], while extending it with the capability to predict absolute scale in an endto-end manner. Fig. 5 illustrates the overall architecture of DA-Next. Specifically, DA-Next takes a sequence of frames I = {𝐼𝑖}𝑖𝑁=1 and auxiliary camera information C = {𝐶𝑖}𝑖𝑁=1 as input, where each 𝐶𝑖 = {𝐾𝑖,𝐺𝑖} comprises the intrinsics 𝐾𝑖 ∈ R3×3 and the pose 𝐺𝑖 ∈ SE(3). All the frames are first patchified into patch tokens e𝑓. The patch tokens from all frames are then concatenated with camera tokens e𝑐 and scale tokens e𝑠, and jointly processed by the transformer encoder E: (eˆ𝑐,eˆ𝑠,eˆ 𝑓) = E(e𝑐,e𝑠,e𝑓). Here, e𝑐 represents GT camera tokens when auxiliary camera information is available, and defaults to learnable camera tokens otherwise. DA-Next adopts pure transformer blocks as its encoder, where frame-wise self-attention and global self-attention alternate throughout the layers. After being processed by the encoder (𝐿 layers), the patch tokens eˆ 𝑓 and camera tokens eˆ𝑐 are fed into the Dual-DPT heads to produce the final depth map Dˆ and ray map Rˆ predictions, while the scale tokens eˆ𝑠 are passed through a lightweight MLP to regress a scalar scale factor 𝑆ˆ. Finally, the scene point cloud is reconstructed from the predicted depth map Dˆ , ray map Rˆ , and scale factor 𝑆ˆ.

Transformer Blocks

[Figure 122]

[Figure 123]

|[Figure 124]<br><br>[Figure 125]|
|---|

| |
|---|
| |
| |
|[Figure 126]<br><br>[Figure 127]<br><br>[Figure 128]<br><br>[Figure 129]|
|Depth|
|Depth|
| |

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Dual-DPT Head Scale

[Figure 135]

[Figure 136]

Global-Attention Block

Frame-Attention Block

…

Image Patch Embed

|[Figure 137]<br><br>[Figure 138]|
|---|

[Figure 139]

| |
|---|
|Ray|
|Ray|
| |
|Ray|
|Ray|
| |

or

|[Figure 140]<br><br>[Figure 141]|
|---|

…

[Figure 142]

[Figure 143]

Head

|[Figure 144]<br><br>[Figure 145]|
|---|

[Figure 146]

…

[Figure 147]

[Figure 148]

Scene Scale

x L times

[Figure 149]

[Figure 150]

Points

GT Camera Token (If available) Scale Token

Camera Poses

Figure 5 | Overview of Depth-Anything-Next. The model incorporates additional scale tokens to learn the scene-level metric scale. Optionally, camera pose information can be embedded as GT camera tokens to serve as auxiliary input to provide geometric guidance.

###### 4.3. Implementation Details

Training Objectives. Following DA3 [54], our training objective is a multi-task loss comprising five components: depths {Dˆ ,D}, depth gradients {∇Dˆ , ∇D}, ray maps {Rˆ ,R}, points {Pˆ,P}, and scale {𝑆ˆ,𝑆} supervision. The total loss is defined as L = Ldepth + 𝛼Lgrad + Lray + Lpmap + Lscale. Prior to loss computation, all ground-truth signals are canonicalized into the first-camera

[Figure 151]

[Figure 152]

Figure 6 | Operating snapshot. Memory, depth error, and inference time at 𝑁 = 800.

Figure 7 | Memory scaling. Peak GPU memory versus input sequence length.

coordinate frame and then normalized by a per-scene scale factor 𝑆, defined as the mean ℓ2 norm of the valid ground-truth world points P. We divide P, D, and the camera translations by 𝑆, and use 𝑆 itself as the regression target of the scale head, so that 𝑆ˆ recovers the absolute metric scale to which the other predictions are invariant. All loss terms are based on the ℓ1 norm with 𝛼 = 1, where the scale loss is defined as: Lscale = ∥ 𝑓log(𝑆ˆ) − 𝑓log(𝑆)∥1 and 𝑓log : x → (x/∥x∥) · log(1+ ∥x∥).

Datasets. We train our model mainly on DA-Next-5M. To mitigate potential generalization degradation, we retain data from 11 general 3D datasets for joint training, including Mapfree [2], Hypersim [76], Infinigen [73], etc. The full datasheet and dataset mixture can be found in the

- Appendix D.

Training Details. Our DA-Next architecture follows DA3-Giant [54] with 𝐿 = 41 Transformer blocks and is initialized by pre-trained weights. During training, each batch incorporates ground-truth camera information as auxiliary input with probability 𝑝 = 20%. The training runs end-to-end on 4 NVIDIA H200 GPUs over seven days. We provide complete implementation details in Appendix D.

#### 5. Findings: How to Train Your Best Spatial Foundation Models?

Tab. 1 presents the main results on SpatialBench, reporting depth, camera pose, trajectory, and point cloud metrics across single-frame, sparse, medium, dense, and average settings for 41 models spanning six reconstruction paradigms. We refer the reader to Appendix F for per-regime sub-tables and per-dataset metric breakdowns.

Full-Context Attention Sets the Accuracy Upper Bound on High-Memory GPUs. Fig. 6 compares representative models at a fixed sequence length of 𝑁 = 800, where all compared methods successfully complete the input. Under this shared input budget, full-context feedforward models occupy the strongest accuracy region of the memory–accuracy Pareto plot. In particular, DA3-Giant and 𝜋3 achieve the lowest depth errors among the compared paradigms, outperforming streaming and online-map variants in reconstruction accuracy. This indicates that globally coupled attention over the input sequence remains a highly effective representation mechanism for geometric reasoning. By jointly resolving cross-view correspondences and enforcing scene-level consistency, full-context models provide stronger reconstruction accuracy and generalization under the same budget.

Table 1 | Main Results on SpatialBench. We report performance across four input settings: Single Frame, Sparse, Medium, Dense, and their Average across all settings. Time is the per-sequence inference time in seconds reported in the sparse regime, averaged over all scenes. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Outof-memory (OOM, > 140G) and Timeout (T.O, > 4h per scene) cells are shaded light red. Within each sub-category, the bold value marks the in-group best.

Single Frame Sparse Medium Dense Average

#Params (M)

Time (s)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.17 7.59 0.385 0.257 0.498 0.276 0.448 1.691 0.343 OOM OOM OOM OOM (0.306) (0.473) (1.691) (0.343) MASt3R 688.64 8.17 0.456 0.209 0.568 0.259 0.522 1.911 0.370 OOM OOM OOM OOM (0.308) (0.545) (1.911) (0.370)

###### End-to-End Feed-Forward

VGGT 1256.54 0.40 0.184 0.105 0.700 0.125 0.687 0.727 0.661 OOM OOM OOM OOM (0.138) (0.694) (0.727) (0.661) Fast3R 647.55 0.90 0.350 0.260 0.392 0.255 0.386 6.582 0.300 0.331 0.232 13.68 0.224 0.299 0.337 10.13 0.262 FastVGGT 1157.94 0.24 0.183 0.113 0.631 0.105 0.662 0.738 0.576 0.120 0.588 19.23 0.479 0.130 0.627 9.984 0.527 MUSt3R 423.43 0.96 0.429 0.165 0.614 0.162 0.643 3.097 0.507 T.O T.O T.O T.O (0.252) (0.629) (3.097) (0.507) MapAnything 1228.49 0.22 0.451 0.153 0.579 0.146 0.579 1.737 0.420 OOM OOM OOM OOM (0.250) (0.579) (1.737) (0.420) OmniVGGT 1217.49 0.22 0.188 0.117 0.665 0.111 0.665 1.491 0.595 OOM OOM OOM OOM (0.139) (0.665) (1.491) (0.595) 𝜋3 958.70 0.20 0.478 0.092 0.742 0.082 0.749 0.565 0.649 0.109 0.524 16.39 0.332 0.190 0.672 8.478 0.491 𝜋3-X 1360.03 0.24 0.371 0.084 0.741 0.078 0.744 0.369 0.658 OOM OOM OOM OOM (0.178) (0.742) (0.369) (0.658) AMB3R 1563.12 0.53 0.466 0.088 0.739 0.085 0.727 0.645 0.554 OOM OOM OOM OOM (0.213) (0.733) (0.645) (0.554) DA3-Small 34.30 0.39 0.385 0.191 0.476 0.176 0.479 4.850 0.432 0.208 0.368 28.12 0.325 0.240 0.441 16.48 0.379 DA3-Base 135.37 0.40 0.349 0.159 0.566 0.142 0.562 3.865 0.515 0.166 0.436 26.35 0.399 0.204 0.521 15.11 0.457 DA3-Large 410.94 0.41 0.333 0.128 0.688 0.105 0.701 2.722 0.626 OOM OOM OOM OOM (0.189) (0.694) (2.722) (0.626) DA3-Giant 1355.67 0.47 0.368 0.095 0.785 0.086 0.776 1.161 0.742 OOM OOM OOM OOM (0.183) (0.780) (1.161) (0.742) DA3-Nested 1689.85 0.52 0.364 0.106 0.779 0.086 0.770 1.980 0.737 OOM OOM OOM OOM (0.185) (0.774) (1.980) (0.737) WorldMirror 1263.34 0.22 0.349 0.139 0.660 0.118 0.674 1.357 0.575 OOM OOM OOM OOM (0.202) (0.667) (1.357) (0.575) VGGT-Omega 1143.81 0.48 0.516 0.077 0.803 0.067 0.795 0.659 0.706 – – – – (0.220) (0.799) (0.659) (0.706)

0.166

0.050

0.809

0.035

0.819

1.442

0.727

DA-Next † (Ours) 1303.76 0.50

OOM OOM OOM OOM (0.084) (0.814) (1.442) (0.727)

(−54.9%)

(−47.4%)

(+3.1%)

(−59.3%)

(+5.5%)

(+24.2%)

(−2.0%)

###### Online

Spann3r224 658.69 0.55 0.370 0.274 0.329 0.252 0.361 4.312 0.254 0.315 0.246 26.48 0.159 0.303 0.312 15.4 0.207 CUT3R 793.31 0.41 0.247 0.196 0.519 0.189 0.469 2.676 0.286 0.260 0.165 25.54 0.109 0.223 0.384 14.11 0.197 MonST3R 571.17 20.81 0.309 0.227 0.269 0.241 0.195 2.234 0.081 OOM OOM OOM OOM (0.259) (0.232) (2.234) (0.081) Point3R 828.01 1.05 0.379 0.221 0.339 0.228 0.303 6.512 0.211 0.285 0.212 28.09 0.139 0.278 0.285 17.3 0.175 Stream3R-S 1190.60 0.62 0.409 0.114 0.603 0.204 0.427 5.717 0.348 OOM OOM OOM OOM (0.242) (0.515) (5.717) (0.348) Stream3R-W 1190.60 0.62 0.409 0.117 0.597 0.240 0.364 6.756 0.323 OOM OOM OOM OOM (0.255) (0.480) (6.756) (0.323) StreamVGGT 1256.54 0.85 0.219 0.154 0.598 0.171 0.562 4.940 0.397 0.198 0.413 26.9 0.251 0.185 0.524 15.92 0.324 Page4D 1256.81 0.56 0.228 0.112 0.608 0.107 0.618 0.855 0.423 OOM OOM OOM OOM (0.149) (0.613) (0.855) (0.423) InfiniteVGGT 1256.54 0.46 0.217 0.154 0.596 0.170 0.563 4.964 0.402 0.197 0.416 27.01 0.254 0.184 0.525 15.99 0.328 Wint3R 749.46 0.41 0.619 0.157 0.499 0.144 0.444 3.944 0.401 0.234 0.202 27.8 0.114 0.288 0.382 15.87 0.258 LongStream-B 1190.60 0.59 0.523 0.153 0.549 0.224 0.455 0.925 0.135 0.269 0.294 5.766 0.083 0.292 0.433 3.345 0.109 LongStream-S 1190.60 0.83 0.523 0.151 0.543 0.166 0.385 1.188 0.126 0.279 0.218 10.08 0.083 0.280 0.382 5.634 0.105 LingbotMap∗-W 1157.94 0.30 0.333 0.138 0.650 0.114 0.641 0.509 0.362 0.167 0.553 4.694 0.352 0.188 0.615 2.602 0.357 LingbotMap∗-S 1157.94 0.33 0.333 0.138 0.650 0.114 0.647 0.508 0.411 0.139 0.627 3.470 0.472 0.181 0.641 1.989 0.442

###### Chunk-wise

VGGT-Long 1256.54 0.20 0.184 0.105 0.700 0.131 0.679 0.512 0.633 0.222 0.507 8.467 0.467 0.161 0.629 4.490 0.550 𝜋3-Long 958.70 0.23 0.478 0.092 0.742 0.097 0.740 0.465 0.590 0.216 0.614 4.021 0.251 0.221 0.699 2.243 0.420 DA3-Streaming 1355.67 0.51 0.368 0.095 0.785 0.091 0.767 0.563 0.725 0.245 0.546 8.575 0.516 0.200 0.699 4.569 0.621

###### SLAM-based

MASt3R-SLAM 688.64 3.04 0.348 0.336 0.190 0.348 0.262 6.075 0.130 0.404 0.311 25.7 0.121 0.359 0.254 15.89 0.126 VGGT-SLAM 1256.54 0.57 0.184 0.105 0.700 0.129 0.645 0.686 0.610 0.211 0.441 9.069 0.384 0.157 0.595 4.878 0.497

###### Test-Time Training

TTT3R 793.31 0.61 0.247 0.202 0.469 0.179 0.493 2.343 0.294 0.222 0.321 21.07 0.173 0.212 0.428 11.71 0.233 Scal3R 1266.14 2.32 0.227 0.114 0.732 0.147 0.670 0.400 0.671 0.244 0.480 2.396 0.498 0.183 0.627 1.398 0.585 LoGeR 1254.62 0.26 0.251 0.095 0.687 0.113 0.693 0.591 0.504 0.197 0.552 5.217 0.335 0.164 0.644 2.904 0.419 LoGeR∗ 1254.60 0.30 0.200 0.077 0.708 0.083 0.714 0.566 0.574 0.156 0.598 4.598 0.421 0.129 0.673 2.582 0.497

“S” denotes stream, “B” denotes batch, and “W” denotes window. LingbotMap∗ indicates the best checkpoint is selected in each regime. DA-Next †: Numbers in (parentheses) below each DA-Next entry give the relative gap to DA3-Giant. Values in parentheses in the Average column indicate that the method runs OOM on the dense regime, and the average is therefore computed over fewer settings. Such methods are excluded from per-column rankings in the Average column. Note that DA-Next is excluded from the per-column rankings.

Takeaway: Under the same input budget, the full-context attention models still define the accuracy upper bound, suggesting that globally coupled representations remain crucial for high-fidelity 3D geometry estimation and reconstruction.

Bounded-Memory Modeling Enables Long-Sequence Reconstruction on Limited GPUs. The accuracy advantage of full-context models comes with a clear physical constraint. As shown in Fig. 7, their GPU memory consumption grows rapidly with sequence length and eventually reaches the out-of-memory boundary on dense long-horizon inputs. Streaming, online, chunk-

[Figure 153]

[Figure 154]

Figure 8 | Training coverage. Dataset count, parameter scale, and SpatialBench accuracy.

Figure 9 | Domain-level OOD severity. Mean AUC@30 grouped by evaluation domain.

wise, and TTT variants exhibit a different scaling behavior: by restricting the active context through causal updates, sliding windows, or chunked inference, they maintain substantially flatter memory curves and can continue processing sequences that full-context models cannot complete under the same hardware budget. As sequence length grows, these methods show stronger trajectory-level consistency (lower ATE), reflecting their ability to maintain long-range geometric alignment. However, their depth estimation accuracy remains below that of fullcontext models across all input regimes, indicating that bounded-memory inference trades pairwise geometric precision for scalability. Thus, these two model families occupy complementary operating regimes: full-context models are preferable when accuracy on bounded inputs is the priority, while bounded-memory methods are better suited for long-horizon or resource-constrained deployment.

Takeaway: Streaming, chunk-wise, and TTT models trade part of the full-context accuracy advantage for bounded-memory inference, enabling continuous long-horizon reconstruction under realistic hardware constraints.

Training Data Volume Correlates with Performance, but Data Quality is the a Critical Factor. Tab. 5 summarizes the training datasets used across all evaluated methods, showing considerable variation in both dataset count and domain coverage. As shown in Fig. 8, there is a clear positive correlation between the number of training datasets and composite benchmark score within the end-to-end and online paradigms: methods trained on more diverse data sources consistently achieve higher performance. However, data volume alone does not tell the full story. Within comparable dataset counts, data quality plays a more decisive role than sheer dataset quantity. A representative example is DA3, which leverages synthetic datasets to train a teacher model and subsequently refines noisy depth estimates to generate high-quality pseudo-GT supervision. This careful curation strategy enables DA3 to achieve the highest composite scores among feed-forward models, despite not relying on the largest training corpus.

Takeaway: Training data volume and performance are positively correlated, but data quality is the more decisive factor: carefully curated, high-quality pseudo-GT supervision consistently outperforms larger but noisier training mixtures under comparable dataset scales.

Egocentric-View and Wrist-View Remain the Dominant OOD Failure Modes. The most severe generalization gap exposed by SpatialBench is not on standard indoor reconstruction datasets, but on embodied-view domains. As shown in Fig. 9, the cross-method average remains relatively strong on indoor datasets, while performance drops sharply on ego-view and especially wrist-

view sequences. This failure is not caused by a single weak model: the OOD curve averages over the full evaluated method pool, indicating a field-level limitation. The training-data analysis explains this behavior. Current pre-training mixtures heavily cover standard indoor and outdoor reconstruction data, but real robot wrist-view data is systematically absent, and real egocentric coverage is sparse.

To address this gap, DA-Next is trained on DA-Next-5M, which explicitly incorporates egocentric and wrist-view data into the training mixture. As shown in Tab. 1, DA-Next achieves consistent and substantial improvements over DA3-Giant across depth and camera pose metrics: depth AbsRel improves by 47% from 0.095 to 0.050 on sparse inputs and 59% from 0.086 to 0.035 on medium inputs, while AUC@30 improves by +3.1% and +5.5% respectively. These gains confirm that targeted in-domain data curation is an effective and practical strategy for closing the OOD gap.

Takeaway: The largest remaining data gap is domain diversity, not only data volume: ego-view and wrist-view data are underrepresented in current training mixtures and produce the strongest OOD failures on SpatialBench. DA-Next, trained on DA-Next-5M, directly addresses this gap.

###### Is Test-Time Training a Free Lunch?

Table 2 | TTT vs. Base model across input length regimes. For each (Base model, TTT) pair, we report aggregate camerapose AUC@30 (↑) and global trajectory ATE (↓), formatted as base → TTT. Green marks improvement, red marks regression. † Vanilla VGGT OOMs under Dense.

Tab. 2 contrasts three pairs of feedforward 3D foundation models with their test-time training (TTT) descendants: CUT3R [98] / TTT3R [16], VGGT [94] / Scal3R [113], and Pi3 [101] / LoGeR [126] across Sparse, Medium, and Dense, summarized by aggregate camera-pose AUC@30 (↑) and global trajectory ATE (↓). Specifically, the TTT methods are designed to digest sequences longer than the context their base models were trained on. For example, Scal3R inserts chunk-wise Global Context Memory into VGGT, and LoGeR augments Pi3 with a parametric TTT memory plus sliding-window attention. Therefore, the natural prior is that TTT should pay off most when sequences are long, and least when they are short and discontinuous. The Dense block confirms this expectation. TTT3R nearly doubles AUC@30 (0.165→0.321, +95%) and cuts ATE by 17.5%; LoGeR lifts AUC@30 by 14.1% and reduces ATE by 72% over Pi3; Scal3R is the only VGGT-family model that runs at all, since vanilla VGGT OOMs under thousand-frame inputs. In the Medium setting, inputs contain a mix of short and moderately long segments, which limits the benefit of TTT. Gains become inconsistent across metrics: AUC@30 improves only for TTT3R (+5.1%) while regressing for Scal3R (−2.5%) and LoGeR (−4.7%); ATE narrows for TTT3R (−12.7%) and Scal3R (−45.2%) but stays flat for LoGeR (0.57→0.57). In the Sparse setting, inputs consist of only 4–15 frames with large baselines and limited temporal continuity, making it difficult for TTT methods to perform effective test-time updates. As a result, the end-to-end base models outperform their TTT counterparts on AUC@30 in two of the three pairs, with only Scal3R holding even (+4.7%).

Regime Pair AUC@30 ↑ ATE ↓

CUT3R / TTT3R 0.519 → 0.470 VGGT / Scal3R 0.700 → 0.732 Pi3 / LoGeR 0.742 → 0.708 —

Sparse

CUT3R / TTT3R 0.469 → 0.493 2.68 → 2.34 VGGT / Scal3R 0.687 → 0.670 0.73 → 0.40 Pi3 / LoGeR 0.749 → 0.714 0.57 → 0.57

Medium

CUT3R / TTT3R 0.165 → 0.321 25.5 → 21.1 VGGT† / Scal3R N/A → 0.480 N/A → 2.40 Pi3 / LoGeR 0.524 → 0.598 16.4 → 4.60

Dense

Takeaway: TTT’s gains are concentrated on dense long sequences, where it consistently improves both pairwise camera pose accuracy and global trajectory consistency over the base models, which confirms

Table 3 | Effect of GT Priors across Sparse / Medium. We inject ground-truth depth and/or camera (pose + intrinsic) priors for each prior-aware model. Trajectory and point-cloud metrics are not reported for the sparse regime.

Aux. Prior Depth Camera Trajectory PointCloud Depth Camera AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ RAcc3 ↑ RAcc5 ↑ TAcc3 ↑ TAcc5 ↑ AUC@5↑ AUC@15↑ AUC@30↑ ATE↓ RPE𝑡 ↓ RPE𝑟 ↓ F-Score↑ Overall↓

Method

###### Sparse

- ✗ ✗ 0.095 0.107 0.608 0.563 0.689 0.821 0.791 0.870 0.587 0.682 0.525 0.699 0.785 – – – – –

- ✗ ✓ 0.078 0.100 0.586 0.635 0.749 0.859 0.968 0.981 0.965 0.989 0.918 0.969 0.984 – – – – –

DA3-Giant

- ✗ ✗ 0.153 1.079 1.337 0.361 0.512 0.701 0.608 0.762 0.300 0.423 0.244 0.446 0.579 – – – – –

✓ ✗ 0.029 0.048 0.415 0.726 0.845 0.940 0.651 0.783 0.404 0.507 0.317 0.547 0.674 – – – – –

- ✗ ✓ 0.143 1.172 1.192 0.427 0.580 0.755 1.000 1.000 0.779 0.894 0.741 0.883 0.934 – – – – –

MapAnything

###### ✓ ✓ 0.020 0.040 0.368 0.861 0.906 0.949 1.000 1.000 0.956 0.981 0.913 0.966 0.982 – – – – –

- ✗ ✗ 0.117 0.119 0.669 0.534 0.671 0.810 0.620 0.746 0.416 0.507 0.332 0.537 0.665 – – – – –

✓ ✗ 0.023 0.061 0.479 0.840 0.913 0.963 0.613 0.723 0.385 0.479 0.313 0.505 0.631 – – – – –

- ✗ ✓ 0.115 0.117 0.665 0.545 0.680 0.814 0.896 0.953 0.819 0.896 0.697 0.858 0.921 – – – – –

OmniVGGT

###### ✓ ✓ 0.021 0.044 0.427 0.850 0.919 0.969 0.907 0.959 0.833 0.917 0.715 0.876 0.934 – – – – –

- ✗ ✗ 0.084 0.084 0.599 0.576 0.710 0.833 0.756 0.837 0.491 0.595 0.427 0.627 0.741 – – – – –

✓ ✗ 0.009 0.017 0.255 0.959 0.979 0.992 0.787 0.860 0.540 0.645 0.476 0.667 0.769 – – – – –

- ✗ ✓ 0.080 0.082 0.589 0.640 0.757 0.858 0.936 0.979 0.715 0.834 0.636 0.827 0.901 – – – – –

𝜋3-X

###### ✓ ✓ 0.009 0.017 0.255 0.960 0.980 0.992 0.940 0.981 0.758 0.848 0.679 0.843 0.908 – – – – –

- ✗ ✗ 0.139 0.165 0.803 0.443 0.585 0.747 0.701 0.796 0.400 0.507 0.328 0.537 0.666 – – – – –

✓ ✗ 0.081 0.243 0.933 0.510 0.643 0.789 0.695 0.808 0.377 0.507 0.329 0.544 0.672 – – – – –

- ✗ ✓ 0.127 0.158 0.759 0.559 0.688 0.811 0.812 0.899 0.629 0.762 0.534 0.762 0.863 – – – – –

WorldMirror

✓ ✓ 0.058 0.282 0.945 0.674 0.772 0.868 0.826 0.911 0.620 0.753 0.533 0.764 0.864 – – – – –

Medium

- ✗ ✗ 0.086 0.088 0.578 0.572 0.686 0.812 0.750 0.834 0.587 0.663 0.532 0.684 0.776 1.161 0.284 2.275 0.742 0.073

- ✗ ✓ 0.078 0.111 0.579 0.644 0.756 0.862 0.984 0.992 0.970 0.987 0.951 0.978 0.987 0.000 0.000 0.002 0.772 0.062

DA3-Giant

- ✗ ✗ 0.146 0.490 1.052 0.347 0.491 0.681 0.563 0.702 0.312 0.419 0.254 0.451 0.579 1.737 0.533 2.852 0.420 0.114

✓ ✗ 0.032 0.037 0.380 0.683 0.811 0.933 0.597 0.733 0.393 0.499 0.316 0.529 0.664 2.053 0.512 2.521 0.539 0.086

- ✗ ✓ 0.126 0.225 0.893 0.429 0.567 0.737 0.998 1.000 0.786 0.881 0.740 0.879 0.930 0.051 0.029 0.270 0.521 0.108

MapAnything

###### ✓ ✓ 0.019 0.034 0.356 0.857 0.910 0.955 1.000 1.000 0.940 0.969 0.896 0.953 0.972 0.042 0.023 0.216 0.773 0.056

- ✗ ✗ 0.111 0.096 0.649 0.518 0.645 0.780 0.609 0.726 0.426 0.527 0.340 0.542 0.665 1.491 0.355 2.768 0.595 0.104

✓ ✗ 0.023 0.056 0.454 0.838 0.914 0.966 0.602 0.727 0.418 0.520 0.325 0.531 0.663 1.904 0.404 2.949 0.582 0.094

- ✗ ✓ 0.108 0.106 0.668 0.521 0.650 0.786 0.870 0.936 0.792 0.878 0.664 0.834 0.905 0.513 0.111 0.589 0.673 0.073

OmniVGGT

###### ✓ ✓ 0.023 0.064 0.479 0.839 0.910 0.961 0.870 0.930 0.804 0.895 0.671 0.843 0.910 0.639 0.133 0.576 0.693 0.068

- ✗ ✗ 0.078 0.061 0.538 0.582 0.712 0.831 0.741 0.827 0.536 0.628 0.463 0.644 0.744 0.369 0.108 1.459 0.658 0.074

✓ ✗ 0.008 0.013 0.239 0.969 0.984 0.993 0.761 0.844 0.547 0.638 0.478 0.659 0.763 0.478 0.129 1.277 0.667 0.074

- ✗ ✓ 0.070 0.060 0.536 0.652 0.759 0.856 0.912 0.960 0.721 0.827 0.641 0.818 0.889 0.373 0.075 0.453 0.703 0.067

𝜋3-X

###### ✓ ✓ 0.008 0.013 0.238 0.969 0.984 0.993 0.914 0.958 0.747 0.840 0.677 0.829 0.890 0.112 0.040 0.461 0.748 0.057

- ✗ ✗ 0.118 0.129 0.745 0.446 0.574 0.721 0.676 0.777 0.419 0.535 0.354 0.557 0.674 1.356 0.342 2.046 0.576 0.090

✓ ✗ 0.082 0.132 0.726 0.512 0.638 0.782 0.694 0.797 0.420 0.540 0.354 0.564 0.685 1.797 0.479 1.974 0.643 0.073

- ✗ ✓ 0.101 0.117 0.700 0.572 0.682 0.790 0.766 0.863 0.621 0.744 0.528 0.740 0.838 0.276 0.093 0.937 0.664 0.077

WorldMirror

###### ✓ ✓ 0.057 0.126 0.688 0.668 0.755 0.851 0.793 0.872 0.633 0.754 0.542 0.752 0.845 0.313 0.107 0.894 0.755 0.058

that TTT is engineered as a length-generalization mechanism rather than as a universal free lunch. Can Injecting GT Priors Drive Performance to a Perfect Level?

Tab. 3 presents an ablation study on the effect of injecting ground-truth depth and camera pose priors for five prior-aware models (DA3-Giant, MapAnything, OmniVGGT, 𝜋3-X, and WorldMirror) across sparse and medium input settings, evaluated on depth, camera, trajectory, and point cloud metrics. All prior-aware models benefit from depth prior injection to varying degrees, with depth metrics approaching near-GT-level accuracy across the board. However, the effect of camera pose priors is more nuanced. DA3-Giant and MapAnything exhibit strong prior adherence: injecting GT camera poses leads to highly consistent pose predictions, with AUC@15 maintained above 90% under all settings. In contrast, OmniVGGT, 𝜋3-X, and WorldMirror show moderate improvements but fail to fully conform to the injected camera poses under challenging conditions (see Fig. 19 in Appendix. B.6), partially overriding them with their own predictions.

Takeaway: Injecting GT depth priors consistently drives depth estimation to near-perfect accuracy, yet camera pose priors yield inconsistent gains. Some models partially override the injected poses with their own predictions and fail under challenging conditions.

Fig. 10 presents a qualitative comparison against representative baselines, where DA-Next

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Multi-view Input

MASt3r Cut3r

MapAnything

OmniVGGT

Fast3r

[Figure 160]

|[Figure 161]|
|---|
| |
|[Figure 162]|
| |
|[Figure 163]|
| |
|[Figure 164]|
| |
|[Figure 165]|

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

DA-Next

VGGT-SLAM Pi3-X

GT Point Cloud

Lingbot-Map

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Multi-view Input

WorldMirror Cut3r

Loger Stream3r

Scal3r

[Figure 174]

|[Figure 175]|
|---|
| |
|[Figure 176]|
| |
|[Figure 177]|
| |
|[Figure 178]|
| |
|[Figure 179]|

|[Figure 180]|
|---|
| |
|[Figure 181]|
| |
|[Figure 182]|
| |
|[Figure 183]|
| |
|[Figure 184]|

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

DA-Next GT Point Cloud

VGGT-Long DA3-Giant

Lingbot-Map

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

Multi-view Input

AMB3R

Page4D LongStream Wint3r

Monst3r

| |
|---|
|[Figure 195]<br><br>[Figure 196]|
|[Figure 197]|
|[Figure 198]|
|[Figure 199]|
| |
|[Figure 200]<br><br>[Figure 201]|
| |

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

DA-Next GT Point Cloud

MUSt3R

FastVGGT

TTT3R

Figure 10 | Qualitative comparison of multi-view 3D reconstruction on SpatialBench.

yields a more accurate camera trajectory and sharper geometry under challenging viewpoints. Due to space limitations, we provide additional visualizations in Appendix B.6 and findings in

- Appendix E.

#### 6. Conclusion

We presented SpatialBench, a comprehensive, reproducible, and cross-paradigm benchmark for evaluating spatial foundation models across diverse domains, input densities, and reconstruction task suites. Through extensive experiments on 41 models across 6 paradigms, SpatialBench reveals that current spatial foundation models are not yet all-round players, exposing critical gaps in domain generalization, input-density robustness. To address the most significant data gap identified, we introduced DA-Next-5M, a large-scale egocentric and wrist-view dataset, and trained DA-Next as a strong baseline. We hope SpatialBench serves as a rigorous foundation for future research toward more generalizable and robust 3D foundation models.

#### References

- [1] Manuel López Antequera, Pau Gargallo, Markus Hofinger, Samuel Rota Bulo, Yubin Kuang, and Peter Kontschieder. 2020. Mapillary planet-scale depth dataset. In European Conference on Computer Vision, pages 589–604. Springer.
- [2] Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Áron Monszpart, Victor Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. 2022. Map-free visual relocalization: Metric pose relative to a single image. In Computer Vision - ECCV 2022 - 17th European Conference, Tel Aviv, Israel, October 23-27, 2022, Proceedings, Part I, volume 13661 of Lecture Notes in Computer Science, pages 690–708. Springer.
- [3] Armen Avetisyan, Christopher Xie, Henry Howard-Jenkins, Tsun-Yi Yang, Samir Aroudj, Suvam Patra, Fuyang Zhang, Duncan Frost, Luke Holland, Campbell Orme, and 1 others. 2024. Scenescript: Reconstructing scenes with an autoregressive structured language model. In European Conference on Computer Vision, pages 247–263. Springer.
- [4] Dejan Azinovic´, Ricardo Martin-Brualla, Dan B Goldman, Matthias Nießner, and Justus Thies. 2022. Neural rgb-d surface reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6290–6301.
- [5] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and 1 others. 2021. Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. arXiv preprint arXiv:2111.08897.
- [6] Zuria Bauer, Francisco Gomez-Donoso, Edmanuel Cruz, Sergio Orts-Escolano, and Miguel Cazorla. 2019. Uasol, a large-scale high-resolution outdoor stereo dataset. Scientific data, 6(1):162.
- [7] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. 2023. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8726–8737.
- [8] Yohann Cabon, Naila Murray, and Martin Humenberger. 2020. Virtual KITTI 2. CoRR, abs/2001.10773.
- [9] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. 2025. Must3r: Multi-view network for stereo 3d reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1050–1060.
- [10] Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, and 19 others. 2025. Sam 3: Segment anything with concepts. Preprint, arXiv:2511.16719.
- [11] Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Niessner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. 2017. Matterport3d: Learning from rgb-d data in indoor environments. arXiv preprint arXiv:1709.06158.
- [12] Ming-Fang Chang, John Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and 1 others. 2019. Argoverse: 3d tracking and forecasting with rich maps. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8748–8757.
- [13] Lin-Zhuo Chen, Jian Gao, Yihang Chen, Ka Leong Cheng, Yipengjing Sun, Liangxiao Hu, Nan Xue, Xing Zhu, Yujun Shen, Yao Yao, and Yinghao Xu. 2026. Geometric context transformer for streaming 3d reconstruction. arXiv preprint arXiv:2604.14141.
- [14] Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, and 1 others. 2025. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088.
- [15] Xiaoxue Chen, Ziyi Xiong, Yuantao Chen, Gen Li, Nan Wang, Hongcheng Luo, Long Chen, Haiyang Sun, Bing Wang, Guang Chen, and 1 others. 2025. Dggt: Feedforward 4d reconstruction of dynamic driving scenes using unposed images. arXiv preprint arXiv:2512.03004.
- [16] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. 2025. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645.
- [17] Chong Cheng, Xianda Chen, Tao Xie, Wei Yin, Weiqiang Ren, Qian Zhang, Xiaoyang Guo, and Hao Wang. 2026. Longstream: Long-sequence streaming autoregressive visual geometry. Preprint, arXiv:2602.13172.
- [18] Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. 2021. Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes. arXiv preprint arXiv:2110.11590.

- [19] Wenyan Cong, Yiqing Liang, Yancheng Zhang, Ziyi Yang, Yan Wang, Boris Ivanovic, Marco Pavone, Chen Chen, Zhangyang Wang, and Zhiwen Fan. 2025. E3d-bench: A benchmark for end-to-end 3d geometric foundation models. arXiv preprint arXiv:2506.01933.
- [20] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. 2016. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223.
- [21] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias Nießner. 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In 2017 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, Honolulu, HI, USA, July 21-26, 2017, pages 2432–2443. IEEE Computer Society.
- [22] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153.
- [23] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. 2025. Vggt-long: Chunk it, loop it, align it – pushing vggt’s limits on kilometer-scale long rgb sequences. Preprint, arXiv:2507.16443.
- [24] Michael Fonder and Marc Van Droogenbroeck. 2019. Mid-air: A multi-modal dataset for extremely low altitude drone flights. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 0–0.
- [25] Mathias Gehrig, Willem Aarents, Daniel Gehrig, and Davide Scaramuzza. 2021. Dsec: A stereo event camera dataset for driving scenarios. IEEE Robotics and Automation Letters, 6(3):4947–4954.
- [26] Andreas Geiger, Philip Lenz, and Raquel Urtasun. 2012. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE conference on computer vision and pattern recognition, pages 3354–3361. IEEE.
- [27] Yotam Gil, Shay Elmalem, Harel Haim, Emanuel Marom, and Raja Giryes. 2021. Online training of stereo self-calibration using monocular depth estimation. IEEE Transactions on Computational Imaging, 7:812–823.
- [28] Jose L Gómez, Manuel Silva, Antonio Seoane, Agnès Borrás, Mario Noriega, Germán Ros, Jose A Iglesias-Guitian, and Antonio M López. 2025. All for one, and one for all: Urbansyn dataset, the third musketeer of synthetic driving scenes. Neurocomputing, 637:130038.
- [29] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, and 1 others. 2022. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761.
- [30] Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos, and Adrien Gaidon. 2020. 3d packing for selfsupervised monocular depth estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2485–2494.
- [31] John Houston, Guido Zuidhof, Luca Bergamini, Yawei Ye, Long Chen, Ashesh Jain, Sammy Omari, Vladimir Iglovikov, and Peter Ondruska. 2021. One thousand and one hours: Self-driving motion prediction dataset. In Conference on Robot Learning, pages 409–418. PMLR.
- [32] Yuan-Ting Hu, Jiahong Wang, Raymond A Yeh, and Alexander G Schwing. 2021. Sail-vos 3d: A synthetic dataset and baselines for object detection and 3d mesh reconstruction from video data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1418–1428.
- [33] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, and Sanja Fidler.

2025. Vipe: Video pose engine for 3d geometric perception. In NVIDIA Research Whitepapers arXiv:2508.10934.

- [34] Po-Han Huang, Kevin Matzen, Johannes Kopf, Narendra Ahuja, and Jia-Bin Huang. 2018. Deepmvs: Learning multi-view stereopsis. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 2821–2830. Computer Vision Foundation / IEEE Computer Society.
- [35] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan

Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, and 1 others. 2025. 𝜋0.5: A vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054.

- [36] Stephen James, Zicong Ma, David Rovick Arrojo, and Andrew J Davison. 2020. Rlbench: The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026.
- [37] Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. 2014. Large scale multi-view stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 406–413.

- [38] Hanwen Jiang, Zexiang Xu, Desai Xie, Ziwen Chen, Haian Jin, Fujun Luan, Zhixin Shu, Kai Zhang, Sai Bi, Xin Sun, and 1 others. 2025. Megasynth: Scaling up 3d scene reconstruction with synthesized data. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16441–16452.
- [39] Haian Jin, Rundi Wu, Tianyuan Zhang, Ruiqi Gao, Jonathan T. Barron, Noah Snavely, and Aleksander Holynski.

2026. ZipMap: Linear-time stateful 3d reconstruction via test-time training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

- [40] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht.

2023. Dynamicstereo: Consistent dynamic depth from stereo videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13229–13239.

- [41] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. 2025. Egomimic: Scaling imitation learning via egocentric video. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pages 13226–13233. IEEE.
- [42] Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulò, Christian Richardt, Deva Ramanan, Sebastian Scherer, and Peter Kontschieder. 2026. MapAnything: Universal feed-forward metric 3D reconstruction. In International Conference on 3D Vision (3DV). IEEE.
- [43] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, and 1 others. 2024. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945.
- [44] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. 2017. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4).
- [45] Anastasiia Kornilova, Marsel Faizullin, Konstantin Pakulev, Andrey Sadkov, Denis Kukushkin, Azat Akhmetyanov, Timur Akhtyamov, Hekmat Taherinejad, and Gonzalo Ferrer. 2022. Smartportraits: Depth powered handheld smartphone dataset of human portraits for state estimation, reconstruction and synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21318–21329.
- [46] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. 2026. STream3R: Scalable sequential 3D reconstruction with causal transformer. In ICLR.
- [47] Hoang-An Le, Thomas Mensink, Partha Das, Sezer Karaoglu, and Theo Gevers. 2021. Eden: Multimodal synthetic dataset of enclosed garden scenes. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1579–1589.
- [48] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. 2024. Grounding image matching in 3d with mast3r. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXII, volume 15130 of Lecture Notes in Computer Science, pages 71–91. Springer.
- [49] Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai. 2023. Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3205–3215.
- [50] Zhengqi Li and Noah Snavely. 2018. Megadepth: Learning single-view depth prediction from internet photos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2041–2050.
- [51] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. 2025. MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- [52] Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, and Tong He. 2025. Wint3r: Window-based streaming reconstruction with camera token pool. Preprint, arXiv:2509.05296.
- [53] Yiyi Liao, Jun Xie, and Andreas Geiger. 2022. Kitti-360: A novel dataset and benchmarks for urban scene understanding in 2d and 3d. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(3):3292–3310.
- [54] Haotong Lin, Sili Chen, Junhao Liew, Donny Y Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang. 2025. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647.
- [55] Haotong Lin, Sida Peng, Jingxiao Chen, Songyou Peng, Jiaming Sun, Minghuan Liu, Hujun Bao, Jiashi Feng, Xiaowei Zhou, and Bingyi Kang. 2025. Prompting depth anything for 4k resolution accurate metric depth estimation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 17070–17080.

- [56] Tao Lin, Gen Li, Yilei Zhong, Yanwen Zou, Yuxin Du, Jiting Liu, Encheng Gu, and Bo Zhao. 2025. Evo-0: Vision-language-action model with implicit spatial understanding. arXiv preprint arXiv:2507.00416.
- [57] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, and 1 others. 2024. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169.
- [58] Minghuan Liu, Zhengbang Zhu, Xiaoshen Han, Peng Hu, Haotong Lin, Xinyao Li, Jingxiao Chen, Jiafeng Xu, Yichu Yang, Yunfeng Lin, Xinghang Li, Yong Yu, Weinan Zhang, Tao Kong, and Bingyi Kang. 2025. Manipulation as in simulation: Enabling accurate geometry perception in robots. arXiv preprint.
- [59] Yifan Liu, Zhiyuan Min, Zhenwei Wang, Junta Wu, Tengfei Wang, Yixuan Yuan, Yawei Luo, and Chunchao Guo.

2025. Worldmirror: Universal 3d world reconstruction with any-prior prompting. arXiv preprint arXiv:2510.10726.

- [60] Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, and Li Yi. 2022. Hoi4d: A 4d egocentric dataset for category-level human-object interaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21013–21022.
- [61] Dominic Maggio and Luca Carlone. 2026. Vggt-slam 2.0: Real-time dense feed-forward scene reconstruction. arXiv preprint arXiv:2601.19887.
- [62] Dominic Maggio, Hyungtae Lim, and Luca Carlone. 2025. Vggt-slam: Dense rgb slam optimized on the sl (4) manifold. Advances in Neural Information Processing Systems, 39.
- [63] John McCormac, Ankur Handa, Stefan Leutenegger, and Andrew J Davison. 2016. Scenenet rgb-d: 5m photorealistic images of synthetic indoor trajectories with ground truth. arXiv preprint arXiv:1612.05079.
- [64] Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andrés Bruhn. 2023. Spring: A highresolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4981–4991.
- [65] Junhong Min, Youngpil Jeon, Jimin Kim, and Minyong Choi. 2025. S2M2: Scalable stereo matching model for reliable depth estimation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
- [66] Riku Murai, Eric Dexheimer, and Andrew J. Davison. 2025. MASt3R-SLAM: Real-time dense SLAM with 3D reconstruction priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
- [67] Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 2019. 3d ken burns effect from a single image. ACM Transactions on Graphics (ToG), 38(6):1–15.
- [68] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, and 7 others. 2024. Dinov2: Learning robust visual features without supervision. Trans. Mach. Learn. Res., 2024.
- [69] Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. 2023. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20133–20143.
- [70] Manthan Patel, Fan Yang, Yuheng Qiu, Cesar Cadena, Sebastian Scherer, Marco Hutter, and Wenshan Wang. 2025. Tartanground: A large-scale dataset for ground robot perception and navigation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 20524–20531. IEEE.
- [71] Haosong Peng, Hao Li, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi, Zhengshen Zhang, Yufeng Zhan, Junfei Zhang, Wenchao Xu, and 1 others. 2025. Omnivggt: Omni-modality driven visual geometry grounded transformer. arXiv preprint arXiv:2511.10560.
- [72] Wilbert Pumacay, Ishika Singh, Jiafei Duan, Ranjay Krishna, Jesse Thomason, and Dieter Fox. 2024. The colosseum: A benchmark for evaluating generalization for robotic manipulation. arXiv preprint arXiv:2402.08191.
- [73] Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, and 1 others. 2024. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21783–21794.
- [74] Santhosh K. Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alexander Clegg, John M. Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X. Chang, Manolis Savva, Yili Zhao, and Dhruv Batra. 2021. Habitat-matterport 3d dataset (HM3D): 1000 large-scale 3d environments for embodied AI. CoRR, abs/2109.08238.

- [75] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny.

2021. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10901–10911.

- [76] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Ángel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. 2021. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021, pages 10892–10902. IEEE.
- [77] Eric Rohmer, Surya PN Singh, and Marc Freese. 2013. V-rep: A versatile and scalable robot simulation framework. In 2013 IEEE/RSJ international conference on intelligent robots and systems, pages 1321–1326. IEEE.
- [78] Ropedia. 2026. Xperience-10m: A large-scale egocentric multimodal dataset with structured 3d/4d annotations. Dataset.
- [79] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, and 1 others. 2019. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339–9347.
- [80] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. 2017. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269.
- [81] Philipp Schröppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. 2022. A benchmark and a baseline for robust multi-view depth estimation. In 2022 International Conference on 3D Vision (3DV), pages 637–645. IEEE.
- [82] You Shen, Zhipeng Zhang, Yansong Qu, and Liujuan Cao. 2025. Fastvggt: Training-free acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560.
- [83] Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi, Antonio Criminisi, and Andrew Fitzgibbon. 2013. Scene coordinate regression forests for camera relocalization in rgb-d images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2930–2937.
- [84] Samarth Sinha, Roman Shapovalov, Jeremy Reizenstein, Ignacio Rocco, Natalia Neverova, Andrea Vedaldi, and David Novotny. 2023. Common pets in 3d: Dynamic new-view synthesis of real-life deformable categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4881–4891.
- [85] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul MurArtal, Carl Ren, Shobhit Verma, and 1 others. 2019. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797.
- [86] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers. 2012. A benchmark for the evaluation of rgb-d slam systems. In Proc. of the International Conference on Intelligent Robot Systems (IROS).
- [87] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, and 4 others. 2020. Scalability in perception for autonomous driving: Waymo open dataset. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 2443–2451. Computer Vision Foundation / IEEE.
- [88] Bin Tan, Changjiang Sun, Xiage Qin, Hanat Adai, Zelin Fu, Tianxiang Zhou, Han Zhang, Yinghao Xu, Xing Zhu, Yujun Shen, and Nan Xue. 2026. Masked depth modeling for spatial perception. arXiv preprint arXiv:2601.17895.
- [89] Zachary Teed and Jia Deng. 2021. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569.
- [90] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. 2021. Smd-nets: Stereo mixture density networks. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 8942–8952. Computer Vision Foundation / IEEE.
- [91] Basile Van Hoorick, Rundi Wu, Ege Ozguroglu, Kyle Sargent, Ruoshi Liu, Pavel Tokmakov, Achal Dave, Changxi Zheng, and Carl Vondrick. 2024. Generative camera dolly: Extreme monocular dynamic novel view synthesis. In European Conference on Computer Vision, pages 313–331. Springer.
- [92] Hengyi Wang and Lourdes Agapito. 2024. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061.
- [93] Hengyi Wang and Lourdes Agapito. 2025. Amb3r: Accurate feed-forward metric-scale 3d reconstruction with backend. arXiv preprint arXiv:2511.20343.

- [94] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotný. 2025. VGGT: visual geometry grounded transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 5294–5306. Computer Vision Foundation / IEEE.
- [95] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. 2026. Vggt-𝜔. Preprint, arXiv:2605.15195.
- [96] Kaixuan Wang and Shaojie Shen. 2020. Flow-motion and depth network for monocular stereo and beyond. IEEE Robotics and Automation Letters, 5(2):3307–3314.
- [97] Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. 2021. Irs: A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface normal estimation. In 2021 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE.
- [98] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. 2025. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 10510–10522.
- [99] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jérôme Revaud. 2024. Dust3r: Geometric 3d vision made easy. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 20697–20709. IEEE.
- [100] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian A. Scherer. 2020. Tartanair: A dataset to push the limits of visual SLAM. In IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2020, Las Vegas, NV, USA, October 24, 2020 - January 24, 2021, pages 4909–4916. IEEE.
- [101] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. 2025. 𝜋3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347.
- [102] Yihan Wang and Jia Deng. 2026. Waft-stereo: Warping-alone field transforms for stereo matching. arXiv preprint arXiv:2603.24836.
- [103] Zehan Wang, Siyu Chen, Lihe Yang, Jialei Wang, Ziang Zhang, Hengshuang Zhao, and Zhou Zhao. 2025. Depth anything with any prior. Preprint, arXiv:2505.10565.
- [104] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz, Orazio Gallo, and Stan Birchfield. 2025. Foundationstereo: Zero-shot stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5249–5260.
- [105] Magnus Wrenninge and Jonas Unger. 2018. Synscapes: A photorealistic synthetic dataset for street scene parsing. arXiv preprint arXiv:1810.08705.
- [106] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, and 1 others. 2023. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 803–814.
- [107] Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. 2025. Point3r: Streaming 3d reconstruction with explicit spatial pointer memory. arXiv preprint arXiv:2507.02863.
- [108] Fei Xia, Amir R Zamir, Zhiyang He, Alexander Sax, Jitendra Malik, and Silvio Savarese. 2018. Gibson env: Real-world perception for embodied agents. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 9068–9079.
- [109] Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. 2024. Rgbd objects in the wild: Scaling real-world 3d object learning from rgb-d videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22378–22389.
- [110] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21469–21480.
- [111] Pengchuan Xiao, Zhenlei Shao, Steven Hao, Zishuo Zhang, Xiaolin Chai, Judy Jiao, Zesong Li, Jian Wu, Kai Sun, Kun Jiang, and 1 others. 2021. Pandaset: Advanced sensor suite dataset for autonomous driving. In 2021 IEEE international intelligent transportation systems conference (ITSC), pages 3095–3101. IEEE.
- [112] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. 2021. Segformer: Simple and efficient design for semantic segmentation with transformers. In Neural Information Processing Systems (NeurIPS).

- [113] Tao Xie, Peishan Yang, Yudong Jin, Yingfeng Cai, Wei Yin, Weiqiang Ren, Qian Zhang, Wei Hua, Sida Peng, Xiaoyang Guo, and 1 others. 2026. Scal3r: Scalable test-time training for large-scale 3d reconstruction. arXiv preprint arXiv:2604.08542.
- [114] Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, and Xin Yang. 2025. Igev++: Iterative multi-range geometry encoding volumes for stereo matching. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- [115] Guorun Yang, Xiao Song, Chaoqin Huang, Zhidong Deng, Jianping Shi, and Bolei Zhou. 2019. Drivingstereo: A large-scale dataset for stereo matching in autonomous driving scenarios. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 899–908.
- [116] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. 2025. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 21924–21935. Computer Vision Foundation / IEEE.
- [117] Sizhe Yang, Linning Xu, Hao Li, Juncheng Mu, Jia Zeng, Dahua Lin, and Jiangmiao Pang. 2026. Robo3r: Enhancing robotic manipulation with accurate feed-forward 3d reconstruction. arXiv preprint arXiv:2602.10101.
- [118] Xuning Yang, Rishit Dagli, Alex Zook, Hugo Hadfield, Ankit Goyal, Stan Birchfield, Fabio Ramos, and Jonathan Tremblay. 2026. Robolab: A high-fidelity simulation benchmark for analysis of task generalist policies. arXiv preprint arXiv:2604.09860.
- [119] Yuxue Yang, Lue Fan, Ziqi Shi, Junran Peng, Feng Wang, and Zhaoxiang Zhang. 2026. Neoverse: Enhancing 4d world model with in-the-wild monocular videos. arXiv preprint arXiv:2601.00393.
- [120] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. 2020. Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2020, Seattle, WA, USA, June 13-19, 2020, pages 1787–1796. Computer Vision Foundation / IEEE.
- [121] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. 2023. Scannet++: A high-fidelity dataset of 3d indoor scenes. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 12–22. IEEE.
- [122] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Chenming Zhu, Zhangyang Xiong, Tianyou Liang, and 1 others. 2023. Mvimgnet: A large-scale dataset of multi-view images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9150–9161.
- [123] Shuai Yuan, Yantai Yang, Xiaotian Yang, Xupeng Zhang, Zhonghao Zhao, Lingming Zhang, and Zhipeng Zhang.

2026. Infinitevggt: Visual geometry grounded transformer for endless streams.

- [124] Amir R Zamir, Alexander Sax, William Shen, Leonidas J Guibas, Jitendra Malik, and Silvio Savarese. 2018. Taskonomy: Disentangling task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3712–3722.
- [125] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. 2025. Monst3r: A simple approach for estimating geometry in the presence of motion. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.
- [126] Junyi Zhang, Charles Herrmann, Junhwa Hur, Chen Sun, Ming-Hsuan Yang, Forrester Cole, Trevor Darrell, and Deqing Sun. 2026. Loger: Long-context geometric reconstruction with hybrid memory. arXiv preprint arXiv:2603.03269.
- [127] Yibo Zhang, Li Zhang, Rui Ma, and Nan Cao. 2025. Texverse: A universe of 3d objects with high-resolution textures. arXiv preprint arXiv:2508.10868.
- [128] Zhengshen Zhang, Hao Li, Yalun Dai, Zhengbang Zhu, Lei Zhou, Chenchen Liu, Dong Wang, Francis EH Tay, Sijin Chen, Ziwei Liu, and 1 others. 2025. From spatial to actions: Grounding vision-language-action model in spatial foundation priors. arXiv preprint arXiv:2510.17439.
- [129] Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. 2020. Structured3d: A large photo-realistic dataset for structured 3d modeling. In European Conference on Computer Vision, pages 519–535. Springer.
- [130] Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, and 1 others. 2026. Egoscale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710.

- [131] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. 2023. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19855–19865.
- [132] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. 2019. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision, 127(3):302– 321.
- [133] Kaichen Zhou, Yuhan Wang, Grace Chen, Xinhai Chang, Gaspard Beaudouin, Fangneng Zhan, Paul Pu Liang, and Mengyu Wang. 2025. Page-4d: Disentangled pose and geometry estimation for 4d perception. arXiv e-prints, pages arXiv–2510.
- [134] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817.
- [135] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, and 1 others. 2025. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201.
- [136] Dong Zhuo, Wenzhao Zheng, Jiahe Guo, Yuqi Wu, Jie Zhou, and Jiwen Lu. 2025. Streaming 4d visual geometry transformer. arXiv preprint arXiv:2507.11539.

#### Appendix Contents

- A SpatialBench Data Curation Pipeline 25

- A.1 DROID Curation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- A.2 Xperience Curation Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.3 Other Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- A.4 Depth Map Post-Processing Pipeline . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26

- B Benchmark Details 27

- B.1 Benchmark Composition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.2 Multi-density Evaluation Regime Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- B.3 General Setting of the Benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- B.4 Evaluation Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29

- B.4.1 Camera Pose Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- B.4.2 Camera Trajectory Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- B.4.3 Depth Estimation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- B.4.4 Dense-View Reconstruction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31

- B.5 Evaluated Models Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32

- B.5.1 Optimization-based Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- B.5.2 End-to-End Feed-Forward Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- B.5.3 Online / Streaming Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- B.5.4 Chunk-based Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- B.5.5 SLAM-based Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- B.5.6 Test-Time Training Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

- B.6 SpatialBench Visualizations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
- B.7 Evaluated Models and Training Datasets in SpatialBench . . . . . . . . . . . . . . . . . . . . . . . . . 36

- C The Collection of DA-Next-5M 46
- D Detail of DA-Next 47

- D.1 Model Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- D.2 Training Objective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
- D.3 Camera Conditioning Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- D.4 Training Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- D.5 Frame Sampling Strategy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- D.6 Training Configuration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

- E Additional Findings 51

- E.1 Do More Input Frames Always Lead to Better Results? . . . . . . . . . . . . . . . . . . . . . . . . . . 51
- E.2 How to Select the Right Spatial Foundation Model for Your Task? . . . . . . . . . . . . . . . . . . . . 51

- F The Complete SpatialBench Results 57
- G Limitations 81

[Figure 207]

[Figure 208]

[Figure 209]

###### Multi-view Inputs OmniWorld DA-Next-5M DA-Next-5M

w/ Init Cam Poses w/ Refined Cam Poses

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

###### Fri_Jul_21_12:36:41_2023 Multi-view Inputs OmniWorld DA-Next-5M DA-Next-5M

[Figure 217]

w/ Init Cam Poses w/ Refined Cam Poses

| |
|---|
|[Figure 218]<br><br>[Figure 219]|
| |
|[Figure 220]|
| |

| |
|---|
|[Figure 221]<br><br>[Figure 222]|
| |
|[Figure 223]|
| |

###### Mon_May_22_12:08:07_2023 Multi-view Inputs OmniWorld DA-Next-5M DA-Next-5M

[Figure 224]

[Figure 225]

w/ Init Cam Poses w/ Refined Cam Poses

[Figure 226]

|[Figure 227]|
|---|
| |
| |
|[Figure 228]|
|[Figure 229]|

|[Figure 230]|
|---|
| |
| |
|[Figure 231]|
|[Figure 232]|

Mon_May_22_12:08:07_2023

Figure 11 | Visual comparison on the DROID dataset. We compare the annotations from OmniWorld against our reconstructed point clouds using initial poses and refined poses, respectively.

#### A. SpatialBench Data Curation Pipeline

In this section, we detail the data processing pipeline in SpatialBench, covering the postprocessing pipelines for Xperience (Sec. A.2), DROID (Sec. A.1), and our collected simulation datasets (Sec. A.3), as well as a unified depth map post-processing pipeline (Sec. A.4) applied to selected datasets to ensure annotation quality.

###### A.1. DROID Curation Pipeline

Accurate geometric annotations are critical for reliable benchmark evaluation. However, the raw DROID [43] data is highly noisy with unreliable camera calibration, precluding direct use of the original annotations. We first investigate learning-based annotation solutions, including Mega-SaM [51] and VIPE [33] for visual SLAM, and Camera Depth Model [58], PriorDA [103], Lingbot-Depth [88], and PromptDA [55] for depth completion. However, none of these methods are trained on wrist-view domains, and we observe significant failure cases under the closerange, high-motion conditions characteristic of this setting.

Since DROID provides synchronized stereo pairs, we turn to stereo-based depth estimation

- as a more reliable annotation source. We test S2M2 [65], FoundationStereo [104], IGEV [114], DynamicStereo [40], and WAFT-Stereo [102], ultimately selecting S2M2 for its superior performance on wrist-view sequences. Specifically, we randomly select a subset of DROID sequences and feed the left-right stereo image pairs from each wrist-view sequence into S2M2 to obtain disparity maps. Metric depth maps for the left eye are then computed using the ZED camera intrinsics and baseline, with a confidence threshold of 0.999 applied to mask out unreliable depth predictions. The masked depth maps and camera intrinsics are subsequently injected as priors into MapAnything [42] to obtain initial camera poses. We then employ SAM3 [10] to annotate

masks for moving objects on selected keyframes, which are propagated to the full sequence. Finally, leveraging the initial camera poses, RGB images, and propagated masks, we perform Bundle Adjustment with both depth and photometric losses to refine the camera parameters and obtain globally aligned point clouds. Sequences with poor temporal depth consistency are filtered out by computing the mean per-pixel reprojection depth error between adjacent frames. Fig. 11 presents a comparison between our DROID data curation pipeline and the OmniWorld data processing approach. OmniWorld annotates wrist-view depth using FoundationStereo and relies on the dataset’s original inaccurate camera poses, resulting in unreliable depth estimates and misaligned point clouds across frames (e.g., incorrect depth on the marker pen on the table in Mon_May_22_12:08:07_2023). In contrast, our annotation pipeline produces temporally consistent depth estimates with well-preserved cross-frame continuity.

We also provide a gallery of the DROID [43] dataset we processed in Fig. 12 and 13. For each scene, we visualize the first frame of the RGB sequence overlaid with the SAM3 segmentation mask (top-left), the corresponding depth map (bottom-left), and the reconstructed point cloud (right).

###### A.2. Xperience Curation Pipeline

Xperience [78] consists of egocentric sequences captured via a head-mounted device during human activity, recorded by four fisheye cameras. We randomly sample a subset of the original data for processing. We obtain camera poses using a SLAM system built upon VIPE [33], and estimate metric depth maps and their confidence maps from rectified stereo image pairs using FoundationStereo [104]. Fig. 14 presents sample visualizations of Xperience sequences included in SpatialBench.

###### A.3. Other Data Collection

We collect robot manipulation sequences from four simulators: RLBench [36], Robo Colosseum [72], RoboTwin [14], and RoboLab [118]. We refer the reader to Sec. C for details on the data collection procedure.

###### A.4. Depth Map Post-Processing Pipeline

Raw depth maps collected from sensors or simulation engines often contain systematic artifacts that degrade geometric evaluation quality. We apply a unified five-stage cleaning pipeline to all real-world datasets in SpatialBench, producing a per-frame binary validity mask that downstream evaluation uses to ignore unreliable pixels.

- Stage 1: Depth Range Clipping. Depth values are first converted to meters and then clipped

to a scene-appropriate valid range [𝑑min, 𝑑max]. Pixels falling outside this range, including nearfield sensor noise and far-field sensor dropout are marked invalid. The range bounds are set per dataset according to the typical operating distance of the capture platform (e.g., [0.05m, 5m] for egocentric indoor scenes and [0m, 3m] for close-range wrist-view sequences).

- Stage 2: Flying Point Removal. Flying points are erroneous depth values that appear at

depth discontinuity boundaries, caused by stereo mismatches or sensor mixed-pixel effects. We detect them by computing the spatial gradient of the depth map and identifying pixels where the gradient magnitude is disproportionately large relative to the local depth value, which indicates a sharp, physically implausible depth jump. Detected discontinuity pixels and a small surrounding neighborhood (controlled by an erosion radius) are masked out, preventing

boundary artifacts from contaminating geometric metrics.

- Stage 3: Edge-Aware Bilateral Filtering. Surviving valid pixels are smoothed using a joint

bilateral filter guided by the corresponding RGB image. The filter suppresses high-frequency depth noise while preserving genuine depth edges aligned with visible object boundaries. Only already-valid pixels are updated; no hole-filling is performed, ensuring that masked regions remain invalid after filtering.

- Stage 4: Small Isolated Region Removal. After filtering, the valid-pixel mask may contain

small, disconnected clusters of surviving pixels that are physically implausible as independent surface patches. We apply connected-component analysis on the binary valid mask and remove any component whose area falls below a minimum threshold, eliminating residual noise fragments that escaped the earlier stages.

- Stage 5: Sky Mask. For outdoor datasets (e.g., OmniWorld [135], Lingbot-Depth [88]), depth

sensors and stereo algorithms systematically fail on sky regions, producing either missing values or extreme outliers at near-infinite range. We generate a per-frame sky mask using a pretrained semantic segmentation model (SegFormer [112] fine-tuned on ADE20K [132]) and set all sky-classified pixels to invalid regardless of their depth values. This prevents sky regions from inflating depth error metrics or introducing spurious points at the horizon in point cloud evaluation.

#### B. Benchmark Details

In this section, we provide a comprehensive description of SpatialBench. We first detail the composition of our dataset collection (Sec. B.2), covering the 19 source datasets and the three multi-density evaluation regimes (Sparse, Medium, and Dense). We then describe the hardware configuration and general experimental settings (Sec. B.4), followed by the complete mathematical definitions of all evaluation metrics used to assess camera pose estimation and depth prediction quality. Next, we report the implementation details and hyperparameters for each evaluated method, organized by category: optimization-based, feed-forward, streaming, chunk-based, SLAM-based, and test-time training approaches. Finally, we present qualitative visualizations of prior-enhanced models and summarize the training datasets used by all compared methods.

###### B.1. Benchmark Composition

Our benchmark aggregates 19 publicly available source datasets, spanning indoor and outdoor scenes, static reconstructions and dynamic sequences, and three viewpoint regimes (third-person, egocentric, and wrist-mounted), captured from both real sensors and simulation. A complete summary of these datasets, together with their per-scene attribute tags, the supported input regimes (Single Frame, Sparse, Medium, Dense), and the number of evaluated scenes and frames, is provided in Table 4. In total, the benchmark contains 546 scenes and 72,540 evaluation frames.

###### B.2. Multi-density Evaluation Regime Details

SPARSE REGIME. Sparse-view selection is implemented as a deterministic set-cover procedure. Let V denote the voxel support of a scene and F the set of candidate frames. Each frame 𝑓 ∈ F

Table 4 | Source Datasets and Scene Attributes Used in Our Benchmark. We summarize the datasets together with their per-dataset attribute tags. The four #Scenes sub-columns report the number of scenes evaluated under each input regime (Single Frame, Sparse, Medium, Dense); “–” marks regimes in which a dataset does not participate. #Frames is the total frame count actually consumed across all regimes.

#Scenes

# Dataset Environment Dynamics View Type Source

#Frames Single Sparse Medium Dense

- 1 7-Scenes [83] Indoor Static Normal Real 7 7 7 7 7,242

- 2 ADT [69] Indoor Dynamic Egocentric Simulation 4 4 4 4 2,112

- 3 DROID [43] Indoor Dynamic Wrist Real 16 16 16 16 3,440

- 4 DTU [37] Indoor Static Normal Real 13 13 13 – 431

- 5 ETH3D [80] Both Static Normal Real 8 8 8 – 150

- 6 Hiroom [54] Indoor Static Normal Simulation 6 6 6 – 90

- 7 KITTI-Odometry [26] Outdoor Dynamic Normal Real – – – 11 12,973

- 8 LingBot-Depth [88] Both Static Mixed Mixed 50 – – – 50

- 9 NRGBD [4] Indoor Static Normal Real 8 8 8 8 10,945

- 10 OmniWorld [135] Outdoor Dynamic Normal Simulation 5 5 5 4 2,033

- 11 RLBench [36] Indoor Dynamic Wrist Simulation 10 10 10 8 1,887

- 12 Robolab [118] Indoor Dynamic Wrist Simulation 8 8 8 8 9,360

- 13 RoboTwin [14] Indoor Dynamic Wrist Simulation 8 8 8 7 1,695

- 14 ScanNet++ [121] Indoor Static Normal Real 11 11 11 11 4,137

- 15 Tanks and Temples [44] Both Static Normal Real 4 4 4 4 934

- 16 TUM [86] Indoor Dynamic Normal Real 6 6 6 6 6,657

- 17 V-KITTI [8] Outdoor Dynamic Normal Simulation 5 5 5 5 2,851

- 18 Waymo [87] Outdoor Dynamic Normal Real 8 8 8 8 2,464

- 19 Xperience [78] Indoor Dynamic Egocentric Real 2 2 2 2 3,089 Total 179 129 129 109 72,540

covers a subset of voxels 𝑉𝑓 ⊆ V. Given a frame budget 𝐾, the selected frame set is defined as

S = arg max

𝑆⊆F, |𝑆|≤𝐾

𝑉𝑓 .

𝑓 ∈𝑆

In practice, we optimize this objective greedily by repeatedly selecting the frame that provides the largest marginal increase in voxel coverage.

MEDIUM REGIME. The medium regime uses the same coverage objective but applies a length-adaptive frame budget. For a scene with 𝑁 frames, the selected set satisfies

S = arg max

𝑉𝑓 ,

𝑆⊆F, 𝐹min(𝑁)≤|𝑆|≤𝐹max(𝑁) 𝑓 ∈𝑆

where 𝐹min(𝑁) and 𝐹max(𝑁) define the admissible range of selected frames. This prevents over-pruning short sequences and over-sampling long ones while preserving moderate view overlap.

DENSE REGIME. For dense evaluation, the goal is to preserve temporal continuity while bounding evaluation cost. Given a scene with 𝑁 frames and a target maximum budget of 𝑇 = 500 frames, we retain all frames when 𝑁 ≤ 𝑇. Otherwise, we subsample the trajectory with a stride

𝑁 𝑇

,

𝑠 =

which yields approximately 𝑇 evenly spaced frames per scene. Datasets with too few valid frames or too few eligible trajectories after filtering are not included in the DENSE regime. Accordingly, missing dense entries in the per-dataset results tables indicate unavailable evaluation indices rather than failed method runs.

###### B.3. General Setting of the Benchmark

All experiments are run on a single workstation with the following configuration: 2× Intel Xeon Platinum 8580 processors (60 cores / 120 threads per socket, 240 logical cores in total), 2 TB of system memory, and 8× NVIDIA H200 GPUs, each with 141 GB of memory and pairwise connected through 18-link NVLink (NV18). The system runs Ubuntu 22.04 with CUDA 12.8 and NVIDIA driver 570.148.08. For all methods, we prioritize using the resolution recommended by each method for inference. Specifically, for methods that only support a fixed resolution (e.g., Spann3R at 224 × 224), we resize the image such that the width matches the target resolution while preserving the original aspect ratio, followed by a center crop. For all other methods, inference is performed at a resolution of 512 or 518 pixels, depending on each model’s patch stride (16 or 14, respectively).

###### B.4. Evaluation Metrics

We provide the complete mathematical definitions of all evaluation metrics used in SpatialBench. Throughout, 𝑁 denotes the number of input frames in a scene, indexed by 𝑖 ∈ {1, . . . , 𝑁}.

Each camera pose 𝐺𝑖 ∈ 𝑆𝐸(3) is a world-to-camera transformation, decomposed as 𝐺𝑖 = [𝑅𝑖 | 𝑡𝑖] with rotation 𝑅𝑖 ∈ 𝑆𝑂(3) and translation 𝑡𝑖 ∈ R3, so that the camera centre in world coordinates is c𝑖 = −𝑅⊤

⊤𝑡∗

𝑖 𝑡𝑖. Ground-truth quantities carry a superscript star (e.g. 𝑅∗

𝑖 , 𝑡∗

𝑖 , c∗

𝑖 = −𝑅∗

𝑖 ) and predicted quantities carry a hat (e.g. 𝑅ˆ𝑖, 𝑡ˆ𝑖, cˆ𝑖 = −𝑅ˆ⊤

𝑖

𝑖 𝑡ˆ𝑖). 𝐷𝑝 and 𝐷ˆ 𝑝 denote the ground-truth and predicted depth at pixel 𝑝, and P𝑝, Pˆ 𝑝 the corresponding 3D world points.

###### B.4.1. Camera Pose Estimation

Camera pose estimation is evaluated via pairwise relative geometry across all |P| ordered pairs P = {(𝑖, 𝑗) | 1≤𝑖< 𝑗≤𝑁}.

Relative rotation error. The relative rotation from camera 𝑖 to camera 𝑗 is

𝑅∗𝑖𝑗 = 𝑅∗𝑗 𝑅∗𝑖⊤, 𝑅ˆ𝑖𝑗 = 𝑅ˆ 𝑗 𝑅ˆ⊤𝑖 . (1) The pairwise rotation error is the geodesic distance on 𝑆𝑂(3):

⊤𝑅ˆ𝑖𝑗 − 1

tr 𝑅∗

𝑖𝑗

2 ∈ [0◦, 180◦]. (2)

𝑒𝑖𝑗𝑅 = arccos

The Relative Rotation Accuracy (RAcc) at threshold 𝑥 is the fraction of pairs whose rotation error falls below 𝑥:

∑︁

1 |P|

1 𝑒𝑖𝑗𝑅 < 𝑥 . (3)

RAcc𝑥 =

(𝑖,𝑗)∈P

Relative translation error. Since the predicted trajectory may carry an arbitrary global scale, translation quality is measured by the angular deviation between the translation components of the predicted and ground-truth pairwise relative poses, with the 180◦ direction ambiguity folded out. Let

𝑡𝑖𝑗∗ = (𝐺𝑖∗)−1𝐺∗𝑗 trans = 𝑅∗⊤𝑖 𝑡∗𝑗 − 𝑡𝑖∗ , 𝑡ˆ𝑖𝑗 = 𝐺 ˆ𝑖−1𝐺ˆ𝑗 trans = 𝑅ˆ⊤𝑖 𝑡 ˆ𝑗 − 𝑡ˆ𝑖 (4) denote the translation components of the relative SE(3) poses, expressed in camera 𝑖’s local frame, with corresponding unit directions

𝑡∗

𝑡ˆ𝑖𝑗 ∥𝑡ˆ𝑖𝑗∥2

𝑖𝑗

𝜏∗𝑖𝑗 =

, 𝜏ˆ𝑖𝑗 =

. (5)

###### ∥𝑡∗

𝑖𝑗∥2

The pairwise translation error is

𝑒𝑡𝑖𝑗 = arccos 𝜏∗𝑖𝑗 · 𝜏ˆ𝑖𝑗 ∈ [0◦, 90◦], (6)

where the absolute value absorbs the inherent 180◦ direction ambiguity under unknown global scale. The Relative Translation Accuracy (TAcc) at threshold 𝑥 is

∑︁

1 |P|

1 𝑒𝑡𝑖𝑗 < 𝑥 . (7)

TAcc𝑥 =

(𝑖,𝑗)∈P

AUC. The joint accuracy curve measures the fraction of pairs for which both rotation and translation errors are below threshold 𝑥:

∑︁

1 |P|

1 max 𝑒𝑖𝑗𝑅, 𝑒𝑡𝑖𝑗 < 𝑥 . (8)

Acc𝑥 =

(𝑖,𝑗)∈P

The Area Under the Curve up to a maximum threshold 𝑥max is

###### ∫ 𝑥

1

max

Acc𝑥 d𝑥, (9) approximated in practice by linear interpolation over a uniform grid of thresholds 𝑥 ∈ [0◦, 𝑥max].

AUC𝑥max =

𝑥max

0

###### B.4.2. Camera Trajectory Estimation

For continuous image sequences (medium and dense regimes), the predicted trajectory {𝑅ˆ𝑖,cˆ𝑖}𝑖𝑁=1 is first aligned to the ground truth {𝑅∗

###### 𝑖 ,c∗

𝑖 }𝑖𝑁=1 via a global Sim(3) transformation. Specifically, we solve for scale 𝑠∗ > 0, rotation 𝑅¯∗ ∈ 𝑆𝑂(3), and translation ¯t∗ ∈ R3 by minimising

###### ∑︁𝑁

𝑠 𝑅¯ cˆ𝑖 + t¯ − c∗𝑖 22, (10)

min

𝑠, 𝑅¯,¯t

𝑖=1

yielding scale-aligned camera centres c˜𝑖 = 𝑠∗𝑅¯∗cˆ𝑖 + ¯t∗ and correspondingly aligned rotations 𝑅˜𝑖 = 𝑅¯∗𝑅ˆ𝑖. Let 𝑇˜𝑖 ∈ 𝑆𝐸(3) denote the resulting aligned camera pose.

###### Absolute Trajectory Error (ATE).

###### ∑︁𝑁

1

2 2 . (11)

###### c ˜𝑖 − c∗

ATE =

𝑖

𝑁

𝑖=1

Relative Pose Error. For consecutive frame pairs with temporal displacement Δ = 1, the ground-truth and aligned-predicted relative SE(3) motions are

𝛿𝑇𝑖∗ = (𝐺𝑖∗)−1 𝐺𝑖∗+1, 𝛿𝑇˜𝑖 = 𝑇˜𝑖−1 𝑇˜𝑖+1, (12) 𝛿𝑅∗𝑖 = 𝑅∗𝑖+1 𝑅∗𝑖⊤, 𝛿𝑅˜𝑖 = 𝑅˜𝑖+1 𝑅˜⊤𝑖 , (13)

𝑖 )−1 𝛿𝑇˜𝑖 ∈ 𝑆𝐸(3). The Relative Translation Error and Relative Rotation Error, each averaged over all 𝑁−1 consecutive windows, are

and the per-step pose-error matrix is 𝐸𝑖 = (𝛿𝑇∗

𝑁∑︁−1

1 𝑁 − 1

RPE𝑡 =

[𝐸𝑖]trans 2, (14)

𝑖=1

𝑁∑︁−1

⊤𝛿𝑅˜𝑖 − 1

tr 𝛿𝑅∗

1 𝑁 − 1

𝑖

RPE𝑟 =

arccos

, (15)

2

𝑖=1

where [𝐸𝑖]trans denotes the translation component of 𝐸𝑖. This follows the standard evo-library RPE definition adopted by the DROID-SLAM evaluation protocol. ATE and RPE𝑡 are reported in metres; RPE𝑟 is in degrees.

###### B.4.3. Depth Estimation

Let Ω𝐷 = {𝑝 | 𝐷𝑝 > 0} denote the set of pixels with valid ground-truth depth in a given frame. For models that do not produce metric-scale output, the predicted depth 𝐷ˆ 𝑝 is first rescaled by the per-frame median alignment:

𝐷ˆ 𝑝 ← 𝑠 · 𝐷ˆ 𝑝, 𝑠 = median

𝑝∈Ω𝐷

𝐷𝑝 𝐷ˆ 𝑝

. (16)

All depth metrics reported in this paper are computed after median-scale alignment, unless otherwise specified.

Absolute Relative Error (AbsRel).

∑︁

1 |Ω𝐷|

AbsRel =

𝑝∈Ω𝐷

𝐷𝑝 − 𝐷ˆ 𝑝 𝐷𝑝

. (17)

Squared Relative Error (SqRel).

SqRel =

∑︁

𝐷𝑝 − 𝐷ˆ 𝑝 2 𝐷𝑝

1 |Ω𝐷|

. (18)

𝑝∈Ω𝐷

Root Mean Squared Error (RMSE).

RMSE = √︄ 1

∑︁

𝐷𝑝 − 𝐷ˆ 𝑝 2 . (19)

|Ω𝐷|

𝑝∈Ω𝐷

Log-scale RMSE (LogRMSE).

LogRMSE = √︄ 1

∑︁

log 𝐷𝑝 − log 𝐷ˆ 𝑝 2 . (20)

|Ω𝐷|

𝑝∈Ω𝐷

Threshold Inlier Ratio (𝛿𝜏).

1 |Ω𝐷|

𝛿𝜏 =

𝑝 ∈ Ω𝐷 max

𝐷𝑝 𝐷ˆ 𝑝

,

𝐷ˆ 𝑝 𝐷𝑝

< 𝜏 . (21)

We report 𝛿1.03, 𝛿1.05, and 𝛿1.10, following the RMVD evaluation protocol of Schröppel et al. [81]. For AbsRel, SqRel, RMSE, and LogRMSE, lower is better; for 𝛿𝜏, higher is better.

###### B.4.4. Dense-View Reconstruction

Scene-level reconstruction quality is evaluated by comparing the predicted and groundtruth point clouds on ScanNet++ [121], NRGBD [4], DTU [37], Hiroom [54], and 7-Scenes [83], following the protocol of DA3-Bench [54].

The ground-truth point cloud P∗ is obtained by uniformly sampling 106 points from the GT mesh, which is reconstructed via offline TSDF fusion of all available RGB-D frames. The predicted point cloud Pˆ is obtained via online TSDF fusion of the model’s per-frame predicted depth maps and camera poses, from which a triangle mesh is extracted and uniformly resampled to 106 points.

Before computing metrics, both point clouds undergo dataset-specific preprocessing. For ScanNet++, NRGBD, and DTU, Pˆ is cropped to the axis-aligned bounding box of P∗ inflated by 0.1m to exclude out-of-range predictions. No cropping is applied for 7-Scenes and Hiroom. Both point clouds are then voxel-downsampled with dataset-specific voxel sizes: 0.02m for ScanNet++ and NRGBD, 0.01m for DTU, and 4/512m (≈7.8mm) for 7-Scenes and Hiroom.

We report two complementary families of Chamfer-based metrics. Threshold-based Precision and Recall.

∑︁

1 |P|ˆ

P ˆ𝑚 − P∗𝑛 2 < 𝑑𝜏 , (22)

Precision =

1 min

P∗𝑛∈P∗

Pˆ𝑚∈Pˆ

∑︁

1 |P∗|

P∗𝑛 − Pˆ𝑚 2 < 𝑑𝜏 . (23)

Recall =

1 min

Pˆ𝑚∈Pˆ

P∗𝑛∈P∗

Precision is the fraction of predicted points within 𝑑𝜏 of any GT point. Recall is the fraction of GT points covered by the prediction within 𝑑𝜏. Both are reported as percentages.

###### Mean Chamfer Accuracy and Completeness (metres).

∑︁

1 |P|ˆ

P ˆ𝑚 − P∗𝑛 2, (24)

min

Acc =

P∗𝑛∈P∗

Pˆ𝑚∈Pˆ

∑︁

1 |P∗|

P∗𝑛 − Pˆ𝑚 2, (25)

Comp =

min

Pˆ𝑚∈Pˆ

P∗𝑛∈P∗

which measures the mean nearest-neighbor distances in metres. F-score and Overall.

2 · Precision · Recall

Precision + Recall (%), (26) Overall =

F-score =

Acc + Comp

2 (m). (27) F-score is the harmonic mean of threshold-based precision and recall. Overall is the mean of the two Chamfer distances, reported in metres. The distance threshold is set to 𝑑𝜏 = 0.05m across all datasets.

###### B.5. Evaluated Models Details

We group the 32 methods (41 variants) into six paradigms. For each model, we briefly summarize its core idea, and then list the key inference-time configuration adopted in our benchmark. Unless otherwise noted, every model is driven by a unified ModelAdapter interface that feeds a complete scene (RGB tensors, GT intrinsics used only for evaluation) to the model and collects pred_depth, pred_pose (cam-to-world), and optionally pred_pointcloud/pred_confidence; depth is aligned to GT by median scaling before computing metrics unless the model produces metric depth.

###### B.5.1. Optimization-based Methods

DUSt3R [99] regresses per-pair dense point maps in a canonical coordinate frame, and recovers a globally consistent reconstruction by a gradient-based global alignment over all image pairs. In our benchmark, the ViT-Large 512 checkpoint (naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt) is used, with niter=300, schedule=cosine, lr=0.01 for the global aligner, and inputs resized to width 512 aligned to a stride of 16.

MASt3R [48] extends DUSt3R with a dense matching head that produces robust 2D correspondences, enabling metric-scale recovery through a sparse global optimizer. We use the metric checkpoint naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric with niter=300, schedule=linear, lr=0.01, input width 512 (align 16), and median depth scaling for final evaluation.

###### B.5.2. End-to-End Feed-Forward Methods

VGGT [94] is a single forward model using a unified transformer that jointly predicts camera poses, depth maps, and point clouds for an arbitrary set of views. We run facebook/VGGT-1B

- at the native 518-pixel resolution; no extra iterative refinement is applied.

VGGT-Omega [95] We use official implementation code and VGGT-Omega-1B-512 checkpoint and run VGGT-Omega-1B at the native 512-pixel resolution.

Fast3R [116] performs one-shot multi-view prediction for up to a thousand views, using factorized global attention to scale VGGT-style reconstruction to long sequences. We use jedyang97/Fast3R_ViT_Large_512 with input width 512 (align 16) and niter_PnP=100 for the auxiliary PnP-RANSAC pose estimation.

FastVGGT [82] accelerates VGGT by a training-free token-merging scheme that drops redundant cross-view tokens inside the transformer. In our runs (checkpoint checkpoints/fastvggt/model_tracker_fixed_e20.pt) we set merging=0 (baseline layer) with merge_ratio=0.9 and enable_point=false to save memory.

MUSt3R [9] couples DUSt3R-style pair prediction with a multi-view registrar that produces a single globally aligned reconstruction without per-scene optimization. We use MUSt3R_512.pth at width 512 (align 16), with preserve_gpu_mem=true and num_refinements_iterations=1.

MapAnything [42] is a universal feed-forward reconstructor that can ingest any combination of images, poses, intrinsics and depth priors and output metric-scale dense geometry. We evaluate MapAnything on the facebook/map-anything checkpoint with memory_efficient_inference=true and use_amp=true.

OmniVGGT [71] augments VGGT with a Geo-Adapter so that any subset of camera poses with intrinsics and depths can be injected as priors. We run Livioni/OmniVGGT in its prior-free configuration for the main comparison, and additionally use its prior-injection branch in Table 3.

Pi3 (𝜋3) [101] removes the reference-frame bias of VGGT-style models with a permutationequivariant design, predicting depth and cameras in a symmetric manner across views. We evaluate yyfz233/Pi3 and its metric variant yyfz233/Pi3X. Pi3X additionally predicts metricscale depth and is evaluated without scale alignment. Pi3X also supports causal prior information injection, which is also used in Table 3.

AMB3R [93] introduces an attention-based multi-branch backbone that outputs metric-

##### scale depth and poses in one forward pass. We use checkpoints/amb3r/amb3r.pt with depth_alignment=none and data_type=bf16.

Depth Anything 3 (DA3) [54] is a scalable 3D foundation model based on transformers trained on large-scale data. We benchmark four sizes: DA3Small (depth-anything/DA3-SMALL), DA3-Base (depth-anything/DA3-BASE), DA3-Large 1.1 (depth-anything/DA3-LARGE-1.1) and DA3-Giant 1.1 (depth-anything/DA3-GIANT-1.1), all with reference-view strategy ref_view_strategy=first for fair comparison. The two-branch variant DA3Nested (depth-anything/DA3NESTED-GIANT-LARGE-1.1) outputs metric-scale depth by combining an anyview Giant branch with a metric Large branch.

WorldMirror [59] is a unified feed-forward reconstruction model that can flexibly condition on (pose, depth, intrinsic) priors. We use tencent/HunyuanWorld-Mirror with cond_flags=[0,0,0] (no prior) for the main benchmark.

###### B.5.3. Online / Streaming Methods

Spann3R [92] builds an external spatial memory on top of DUSt3R so that each incoming frame can be incrementally fused into a shared canonical frame. We use spann3r_101.pth at a resolution of 224 (align 16) with inference_mode=online, use_feat=false and focal_mode=weiszfeld.

CUT3R [98] performs continuous updating of a persistent internal scene state, regressing dense geometry frame-by-frame. We use the default CUT3R checkpoint at width 512 (align 16) with focal_mode=weiszfeld for focal estimation.

MonST3R [125] specializes DUSt3R for dynamic scenes by injecting temporal smoothness, optical-flow, and translation losses into the global optimizer. We use Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt at width 512 (align 16) with niter=300, schedule=linear, lr=0.01, batch_size=16, winsize=5, scenegraph_type=swinstride, flow_loss_weight=0.01, flow_loss_threshold=25, flow_loss_start_iter=0.1, sam2_mask_refine=true, batchify=true (temporal_smoothing_weight=0.01, translation_weight=1.0, shared_focal=true).

Point3R [107] maintains an explicit online 3D point memory that grows as new frames arrive, enabling streaming reconstruction with constant-time per-frame cost. We use point3r_512.pth at width 512 (align 16) with focal_mode=weiszfeld.

Stream3R [46] formulates multi-view reconstruction as causal next-token prediction over pointmaps, enabling an efficient StreamVGGT-style decoder. We use yslan/STream3R checkpoint and implement stream and window variants using default settings.

StreamVGGT [136] is a streaming counterpart of VGGT with temporally causal cross-view attention and KV-caching. We use lch01/StreamVGGT in the default online mode.

PAGE4D [133] extends VGGT to 4D reconstruction by modeling both static scene geometry and dynamic motion components. We use checkpoints/page4d/checkpoint_nomask.pt in its default configuration.

InfiniteVGGT [123] augments StreamVGGT with a rolling KV-cache, so that arbitrarily long sequences can be processed with bounded memory. We use lch01/StreamVGGT as the backbone, image width 518 (align 14) and total_budget=1,200,000 KV-tokens.

WinT3R [52] employs a sliding-window transformer with a camera-token pool

##### to capture long-range geometry cheaply. We use lizizun/WinT3R at width 512 (align 16) with inference_mode=online, window_size=4, state_size=1024 and ret_first_pred=false.

LongStream [17] builds on a VGGT backbone with a keyframe-driven long-context memory module to target long-horizon streaming reconstruction. We use NicolasCC/LongStream at width 518 (align 14) under streaming_mode=causal with keyframe_stride=8, keyframe_mode=fixed and rel_pose_num_iterations=4, and evaluate two inference variants: batch-refresh and streaming-refresh with a sliding window of size 48. The refresh parameter is set to 3 for batch-refresh and 7 for streaming-refresh. In addition, we disable the scale token prediction of LongStream, as we observe that enabling metric scale prediction introduces erroneous outliers in depth estimation metrics on our benchmark, despite the model natively supporting metric-scale output.

LingBot-Map [13] is an online mapping model designed for streaming data, supporting both per-frame streaming and window-based inference with KV-cache sliding windows. We evaluate two variants: LingBot-Map (windowed) with mode=windowed, window_size=64, overlap_size=16, num_scale_frames=8, kv_cache_sliding_window=64 and camera_num_iterations=4; and LingBot-Map (streaming) with adaptive keyframe_interval, kv_cache_sliding_window=64, kv_cache_scale_frames=8 and SDPA attention (use_sdpa=false, enable_3d_rope=true). Both variants use image_size=518, patch_size=14 and enable_point=false. The medium regime runs use the long-context checkpoint lingbot-map-long.pt, while the sparse and single-frame settings use the short-context lingbot-map.pt checkpoint.

###### B.5.4. Chunk-based Methods

VGGT-Long [23] scales VGGT to long sequences by overlapping-chunk inference and cross-chunk Sim(3) alignment. We use facebook/VGGT-1B as the backbone with chunk_size=60 and overlap=30. The chunked streaming variant DA3-Streaming uses backbone depth-anything/DA3-GIANT-1.1 with chunk_size=60, overlap=30 and ref_view_strategy=first for long sequences. The long-sequence variant Pi-Long (built on the Pi3 backbone) also uses chunk_size=60, overlap=30.

###### B.5.5. SLAM-based Methods

MASt3R-SLAM [66] integrates MASt3R’s metric dense matcher into a SLAM pipeline that jointly estimates camera trajectory and metric geometry. We use MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth in uncalibrated mode (use_calib=false), with explicit overrides img_size=512 and c_conf_threshold=1.5 (point-cloud filtering); the remaining tracking hyper-parameters use the official config/base.yaml defaults (C_conf=0.0, Q_conf=1.5, min_match_frac=0.05, max_iters=50).

VGGT-SLAM [61] turns VGGT into a SLAM system by constructing overlapping submaps and performing Sim(3) cross-submap alignment with loop closures. We use facebook/VGGT-1B with submap_size=32, overlapping_window_size=3, max_loops=0 (no loop closure), conf_threshold=25.0, lc_thres=0.95 and use_optical_flow=true.

###### B.5.6. Test-Time Training Methods

TTT3R [16] performs per-sequence test-time training on top of a CUT3R-style backbone, finetuning a small set of parameters to adapt to each scene. We use cut3r_512_dpt_4_64.pth at width 512 (align 16) with focal_mode=weiszfeld.

Scal3R [113] introduces a scalable pipeline that processes long videos in blocks with loop-level test-time refinement. We use xbillowy/Scal3R at width 518 (align 14) with block_size=60, overlap_size=30 (20 for dense), loop_size=20, use_xyz_align=1 and test_use_amp=false.

LoGeR [126] is a long-context hybrid-memory geometry model that combines test-time training with sliding-window attention. We evaluate the LoGeR and LoGeR∗ variants from Junyi42/LoGeR at width 518 (align 14) with window_size=32, overlap_size=3 and reset_every=1 (TTT-state reset across windows). For LoGeR we set se3=false, sim3=false; for LoGeR∗ we set se3=true, sim3=false.

###### B.6. SpatialBench Visualizations

Qualitative Comparison on Representative SpatialBench Cases. Fig. 15 visualizes four representative cases from SpatialBench, covering short indoor reconstruction, dense outdoor driving, dense indoor long-horizon reconstruction, and wrist-view out-of-distribution input. Each case shows the sampled input views, point-cloud reconstructions from representative methods, and the corresponding depth and camera metrics.

Qualitative Comparison on Prior-Enhanced Models. Fig. 16, Fig. 17, and Fig. 18 present qualitative comparisons of prior-aware models under joint camera and depth prior injection, camera-only prior injection, and depth-only prior injection settings, respectively.

Bad Cases For Prior-Enhanced Models. Fig. 19 illustrates representative failure cases of prioraware models under challenging conditions, including object-centric scenes, extreme no-overlap view, and out-of-distribution wrist-view sequences. MapAnything fails on object-centric scenes when GT camera priors are injected. WorldMirror fails on out-of-distribution scenes such as wrist-view sequences, even with both camera and depth priors provided. OmniVGGT breaks down under extreme no-overlap conditions.

###### B.7. Evaluated Models and Training Datasets in SpatialBench

In this subsection, we summarize the training data and annotations used by all evaluated methods. Tab. 5 presents the training datasets of all compared methods, excluding training-free and chunk-wise methods, as they do not involve explicit training. Evaluation-only, qualitativeonly, and runtime benchmark datasets are not marked. ✓ indicates datasets directly used for model training. ✓ indicates that the method is initialized from a pretrained checkpoint whose backbone has been trained on the corresponding dataset. Tab. 6 presents the properties of each dataset, including scene type, real-world vs. synthetic domain, the presence of dynamic content, and public license information.

[Figure 233]

[Figure 234]

- (5) Sat_Aug_19_11:09:16_2023

|[Figure 235]<br><br>[Figure 236]|
|---|

|[Figure 237]<br><br>[Figure 238]|
|---|

[Figure 239]

[Figure 240]

- (6) Sat_Jul__8_10:28:16_2023

|[Figure 241]<br><br>[Figure 242]|
|---|

|[Figure 243]<br><br>[Figure 244]|
|---|

[Figure 245]

[Figure 246]

|[Figure 247]<br><br>[Figure 248]|
|---|

|[Figure 249]<br><br>[Figure 250]|
|---|

- (7) Sun_Jun__4_17:55:44_2023

[Figure 251]

[Figure 252]

|[Figure 253]<br><br>[Figure 254]|
|---|

|[Figure 255]<br><br>[Figure 256]|
|---|

- (8) Thu_Jul_13_11:12:31_2023

- (1) Fri_Jul_14_15:19:25_2023

|[Figure 257]<br><br>[Figure 258]|
|---|

|[Figure 259]<br><br>[Figure 260]|
|---|

- (2) Fri_Jul_21_12:36:41_2023

|[Figure 261]<br><br>[Figure 262]|
|---|

|[Figure 263]<br><br>[Figure 264]|
|---|

[Figure 265]

[Figure 266]

|[Figure 267]<br><br>[Figure 268]|
|---|

|[Figure 269]<br><br>[Figure 270]|
|---|

[Figure 271]

[Figure 272]

- (3) Mon_Jul_31_20:04:41_2023

|[Figure 273]<br><br>[Figure 274]|
|---|

|[Figure 275]<br><br>[Figure 276]|
|---|

- (4) Mon_May_22_12:08:07_2023

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

###### Figure 12 | DROID Gallery 1.

- (13) Tue_Jun_27_00:45:19_2023

|[Figure 281]<br><br>[Figure 282]|
|---|

|[Figure 283]<br><br>[Figure 284]|
|---|

[Figure 285]

[Figure 286]

- (14) Tue_May__9_04:20:30_2023

|[Figure 287]<br><br>[Figure 288]|
|---|

|[Figure 289]<br><br>[Figure 290]|
|---|

[Figure 291]

[Figure 292]

|[Figure 293]<br><br>[Figure 294]|
|---|

|[Figure 295]<br><br>[Figure 296]|
|---|

- (15) Tue_Sep__5_09:01:57_2023

[Figure 297]

[Figure 298]

(9) Thu_May_18_20:23:52_2023

|[Figure 299]<br><br>[Figure 300]|
|---|

|[Figure 301]<br><br>[Figure 302]|
|---|

[Figure 303]

[Figure 304]

|[Figure 305]<br><br>[Figure 306]|
|---|

|[Figure 307]<br><br>[Figure 308]|
|---|

- (16) Wed_May_17_21:25:32_2023

- (10) Tue_Jun__6_00:44:37_2023

|[Figure 309]<br><br>[Figure 310]|
|---|

|[Figure 311]<br><br>[Figure 312]|
|---|

[Figure 313]

[Figure 314]

|[Figure 315]<br><br>[Figure 316]|
|---|

|[Figure 317]<br><br>[Figure 318]|
|---|

- (11) Tue_Jun__6_00:55:14_2023

[Figure 319]

[Figure 320]

|[Figure 321]<br><br>[Figure 322]|
|---|

|[Figure 323]<br><br>[Figure 324]|
|---|

- (12) Tue_Jun_20_03:32:23_2023

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

###### Figure 13 | DROID Gallery 2.

[Figure 330]

###### Figure 14 | Visualization of Xperience Samples.

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

DA3-Giant CUT3R

Multi-view Input

MapAnything

|[Figure 337]|
|---|
| |
| |
|[Figure 338]<br><br>[Figure 339]|
| |
| |
| |
|[Figure 340]|
|[Figure 341]|

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

LingBot-Map-W

pi3 GT Point Cloud

[Figure 347]

[Figure 348]

[Figure 349]

CUT3R

Multi-view Input

DA3-Giant MapAnything

[Figure 350]

[Figure 351]

[Figure 352]

| |
|---|
|[Figure 353]<br><br>[Figure 354]|
| |
|[Figure 355]|
| |
| |
| |
|[Figure 356]<br><br>[Figure 357]|
| |

LingBot-Map-W

pi3 GT Point Cloud

[Figure 358]

[Figure 359]

[Figure 360]

CUT3R

Multi-view Input

DA3-Giant MapAnything

[Figure 361]

[Figure 362]

|[Figure 363]|
|---|
| |
|[Figure 364]|
| |
|[Figure 365]|
| |
|[Figure 366]|
| |
|[Figure 367]|

|[Figure 368]|
|---|
| |
|[Figure 369]|
| |
|[Figure 370]|
| |
|[Figure 371]|
| |
|[Figure 372]|

[Figure 373]

LingBot-Map-W

pi3 GT Point Cloud

[Figure 374]

[Figure 375]

[Figure 376]

CUT3R

Multi-view Input

DA3-Giant MapAnything

[Figure 377]

[Figure 378]

[Figure 379]

| |
|---|
|[Figure 380]<br><br>[Figure 381]|
| |
| |
| |
|[Figure 382]|
| |
|[Figure 383]<br><br>[Figure 384]|
| |

LingBot-Map-W

pi3 GT Point Cloud

[Figure 385]

[Figure 386]

[Figure 387]

- Figure 15 | Representative benchmark cases. We show input views, per-method point-cloud reconstructions, and depth/camera metrics for four representative scenes spanning short indoor reconstruction, dense outdoor driving, dense indoor long-horizon reconstruction, and wrist-view OOD input.

[Figure 388]

[Figure 389]

[Figure 390]

###### robotwin_franka-panda1_stack_blocks_two_episode44_sparse

DA3 MapAnything OmniVGGT

Multi-view Input

|[Figure 391]|
|---|
| |
| |
|[Figure 392]|
| |
|[Figure 393]<br><br>[Figure 394]|
| |
| |
|[Figure 395]|

|[Figure 396]<br><br>[Figure 397]|
|---|

[Figure 398]

[Figure 399]

[Figure 400]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 401]

[Figure 402]

[Figure 403]

DA3 MapAnything OmniVGGT

Multi-view Input

###### Tanks_and_temples_Barn_sparse

|[Figure 404]|
|---|

|[Figure 405]|
|---|

|[Figure 406]<br><br>[Figure 407]|
|---|

|[Figure 408]|
|---|

|[Figure 409]|
|---|

[Figure 410]

[Figure 411]

[Figure 412]

|[Figure 413]|
|---|

|[Figure 414]|
|---|

WorldMirror

Pi3-X Ground Truth Point Cloud

|[Figure 415]|
|---|

|[Figure 416]|
|---|

|[Figure 417]|
|---|

|[Figure 418]|
|---|

[Figure 419]

[Figure 420]

[Figure 421]

###### droid_Sat_Aug_19_11:09:16_2023_17368348_mediu m

DA3 MapAnything OmniVGGT

Multi-view Input

|[Figure 422]|
|---|

|[Figure 423]|
|---|

|[Figure 424]<br><br>[Figure 425]|
|---|

|[Figure 426]|
|---|

|[Figure 427]|
|---|

[Figure 428]

[Figure 429]

|[Figure 430]|
|---|

|[Figure 431]|
|---|

[Figure 432]

WorldMirror

Pi3-X Ground Truth Point Cloud

|[Figure 433]|
|---|

|[Figure 434]|
|---|

|[Figure 435]|
|---|

|[Figure 436]|
|---|

|[Figure 437]|
|---|

|[Figure 438]|
|---|

[Figure 439]

[Figure 440]

[Figure 441]

OmniVGGT

Multi-view Input

DA3 MapAnything

###### omniworld_51af3ddc9546_medium

|[Figure 442]|
|---|

|[Figure 443]|
|---|

|[Figure 444]<br><br>[Figure 445]|
|---|

|[Figure 446]|
|---|

|[Figure 447]|
|---|

|[Figure 448]|
|---|

|[Figure 449]|
|---|

[Figure 450]

[Figure 451]

[Figure 452]

WorldMirror

Pi3-X Ground Truth Point Cloud

|[Figure 453]|
|---|

|[Figure 454]|
|---|

|[Figure 455]|
|---|

|[Figure 456]|
|---|

|[Figure 457]|
|---|

|[Figure 458]|
|---|

[Figure 459]

[Figure 460]

[Figure 461]

waymo_segment5200186706748209867_80_000_100_0_dense

DA3 MapAnything OmniVGGT

Multi-view Input

|[Figure 462]|
|---|

|[Figure 463]|
|---|

|[Figure 464]<br><br>[Figure 465]|
|---|

|[Figure 466]|
|---|

|[Figure 467]|
|---|

[Figure 468]

[Figure 469]

|[Figure 470]|
|---|

|[Figure 471]|
|---|

[Figure 472]

WorldMirror

Pi3-X

Ground Truth Point Cloud

|[Figure 473]|
|---|

|[Figure 474]|
|---|

|[Figure 475]|
|---|

|[Figure 476]|
|---|

- Figure 16 | Qualitative Comparison on Prior-Enhanced Models with Auxiliary Camera and Depth Information. We visualize the reconstruction quality of DA3-Giant, MapAnything, OmniVGGT, Pi3, and WorldMirror under the setting where both camera and depth prior information are provided as auxiliary inputs. For each scene, the right panel reports the depth AbsRel and camera AUC@30 metrics for each method.

[Figure 477]

[Figure 478]

[Figure 479]

DA3 MapAnything OmniVGGT

Multi-view Input

rlbench_take_off_weighing_scales_episode0013_sparse

|[Figure 480]|
|---|
|[Figure 481]|
| |
|[Figure 482]|
| |
|[Figure 483]|
| |
| |
|[Figure 484]|

|[Figure 485]<br><br>[Figure 486]|
|---|

[Figure 487]

[Figure 488]

[Figure 489]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 490]

[Figure 491]

[Figure 492]

DA3 MapAnything OmniVGGT

Multi-view Input

###### eth3d_courtyard_sparse

| |
|---|
|[Figure 493]<br><br>[Figure 494]|
| |
|[Figure 495]|
| |
| |
| |
|[Figure 496]<br><br>[Figure 497]|
| |

|[Figure 498]<br><br>[Figure 499]|
|---|

[Figure 500]

[Figure 501]

[Figure 502]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 503]

[Figure 504]

[Figure 505]

DA3 MapAnything OmniVGGT

Multi-view Input

###### ropedia_slab_01_sparse

|[Figure 506]|
|---|
| |
|[Figure 507]|
| |
|[Figure 508]|
| |
|[Figure 509]|
| |
|[Figure 510]|

|[Figure 511]|
|---|
| |
|[Figure 512]|
| |
|[Figure 513]|
| |
|[Figure 514]|
| |
|[Figure 515]|

|[Figure 516]<br><br>[Figure 517]|
|---|

[Figure 518]

[Figure 519]

[Figure 520]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 521]

[Figure 522]

[Figure 523]

OmniVGGT

Multi-view Input

DA3 MapAnything

###### scannetpp_578511c8a9_sparse

| |
|---|
|[Figure 524]<br><br>[Figure 525]|
| |
|[Figure 526]|
| |
| |
| |
|[Figure 527]<br><br>[Figure 528]|
| |

|[Figure 529]<br><br>[Figure 530]|
|---|

[Figure 531]

[Figure 532]

[Figure 533]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 534]

[Figure 535]

[Figure 536]

DA3 MapAnything OmniVGGT

Multi-view Input

vkitti_scene20_sunset_sparse

|[Figure 537]|
|---|

|[Figure 538]<br><br>[Figure 539]|
|---|

|[Figure 540]|
|---|

|[Figure 541]|
|---|

[Figure 542]

[Figure 543]

[Figure 544]

|[Figure 545]|
|---|

WorldMirror

Pi3-X

Ground Truth Point Cloud

|[Figure 546]|
|---|

|[Figure 547]|
|---|

- Figure 17 | Qualitative Comparison on Prior-Enhanced Models with Auxiliary Camera Information. We visualize the reconstruction quality of DA3-Giant, MapAnything, OmniVGGT, Pi3, and WorldMirror under the setting where camera prior information is provided as auxiliary inputs. For each scene, the right panel reports the depth AbsRel and camera AUC@30 metrics for each method.

[Figure 548]

[Figure 549]

[Figure 550]

###### robolab_ClearOrganicObjectsTask_bg_aerodynamic s_workshop_run_0_demo_0_sparse

DA3 MapAnything OmniVGGT

Multi-view Input

|[Figure 551]|
|---|

|[Figure 552]<br><br>[Figure 553]|
|---|

[Figure 554]

[Figure 555]

[Figure 556]

|[Figure 557]|
|---|

WorldMirror

Pi3-X Ground Truth Point Cloud

|[Figure 558]|
|---|

[Figure 559]

[Figure 560]

[Figure 561]

DA3 MapAnything OmniVGGT

Multi-view Input

###### hiroom_828738_cam_sampled_08_sparse

| |
|---|
|[Figure 562]|
|[Figure 563]|
| |
|[Figure 564]|
|[Figure 565]|
| |

|[Figure 566]<br><br>[Figure 567]|
|---|

[Figure 568]

[Figure 569]

[Figure 570]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 571]

[Figure 572]

[Figure 573]

###### adt_Apartment_release_meal_skeleton_seq135_M12 92_medium

DA3 MapAnything OmniVGGT

Multi-view Input

|[Figure 574]|
|---|
|[Figure 575]|
|[Figure 576]|
|[Figure 577]|

|[Figure 578]|
|---|
|[Figure 579]|
|[Figure 580]|
|[Figure 581]|

|[Figure 582]<br><br>[Figure 583]|
|---|

[Figure 584]

[Figure 585]

[Figure 586]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 587]

[Figure 588]

[Figure 589]

###### tum_rgbd_dataset_freiburg3_long_office_household _0_sparse

OmniVGGT

Multi-view Input

DA3 MapAnything

| |
|---|
|[Figure 590]<br><br>[Figure 591]|
| |
| |
| |
|[Figure 592]|
| |
|[Figure 593]<br><br>[Figure 594]|
| |

|[Figure 595]<br><br>[Figure 596]|
|---|

[Figure 597]

[Figure 598]

[Figure 599]

WorldMirror

Pi3-X Ground Truth Point Cloud

[Figure 600]

[Figure 601]

[Figure 602]

lingbot_RobbyReal_00009_07_overpass_passage_si ngle

DA3 MapAnything OmniVGGT

Single-view Input

|[Figure 603]<br><br>[Figure 604]|
|---|

[Figure 605]

[Figure 606]

[Figure 607]

|[Figure 608]|
|---|

WorldMirror

Pi3-X

Ground Truth Point Cloud

- Figure 18 | Qualitative Comparison on Prior-Enhanced Models with Auxiliary Depth Information. We visualize the reconstruction quality of DA3-Giant, MapAnything, OmniVGGT, Pi3, and WorldMirror under the setting where depth prior information is provided as auxiliary inputs. For each scene, the right panel reports the depth AbsRel and camera AUC@30 metrics for each method.

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

Multi-view Input Ground Truth Point Cloud

MapAnything MapAnything MapAnything

|[Figure 613]|
|---|

|[Figure 614]|
|---|

|[Figure 615]|
|---|

|[Figure 616]|
|---|

|[Figure 617]|
|---|

|[Figure 618]|
|---|

|[Figure 619]|
|---|

|[Figure 620]|
|---|

|[Figure 621]|
|---|

|[Figure 622]|
|---|

|[Figure 623]|
|---|

|[Figure 624]|
|---|

[Figure 625]

[Figure 626]

[Figure 627]

|[Figure 628]|
|---|

|[Figure 629]|
|---|

|[Figure 630]|
|---|

|[Figure 631]|
|---|

wo/ Cam. & Dep. w/ 100% Cam. w/ 100% Cam. & 100% Dep. Multi-view Input WorldMirror WorldMirror WorldMirror

[Figure 632]

Ground Truth Point Cloud

|[Figure 633]|[Figure 634]|[Figure 635]|[Figure 636]|
|---|---|---|---|

|[Figure 637]|[Figure 638]|[Figure 639]|[Figure 640]|
|---|---|---|---|

|[Figure 641]|[Figure 642]|[Figure 643]|[Figure 644]|
|---|---|---|---|

|[Figure 645]|[Figure 646]|[Figure 647]|[Figure 648]|
|---|---|---|---|

[Figure 649]

wo/ Cam. & Dep. w/ 100% Cam. w/ 100% Cam. & 100% Dep.

[Figure 650]

[Figure 651]

[Figure 652]

Multi-view Input OmniVGGT OmniVGGT OmniVGGT

Ground Truth Point Cloud

|[Figure 653]| | |[Figure 654]| |[Figure 655]<br><br>[Figure 656]| |
|---|---|---|---|---|---|---|

| |[Figure 657]| |[Figure 658]<br><br>[Figure 659]| |[Figure 660]| |
|---|---|---|---|---|---|---|

wo/ Cam. & Dep. w/ 100% Dep. w/ 100% Cam. & 100% Dep.

- Figure 19 | Failure Cases on Challenging Scenes. We visualize failure cases of MapAnything, WorldMirror, and OmniVGGT across three challenging scenes.

- Table 5 | Dataset usage across all methods. ✓ indicates datasets directly used for model training, finetuning, or teacher/student training; ✓ indicates datasets inherited from a pretrained checkpoint whose backbone has been trained on the corresponding dataset. The last row reports total used (trained): the first number counts all datasets either directly used by the method or inherited through its pretrained checkpoint, while the number in parentheses counts only datasets used to train or fine-tune the method itself. Identical values, e.g., 42 (42), indicate that no additional datasets are inherited. Evaluation-only, qualitative-only, and runtime benchmark datasets are not marked.

Dataset End-to-end Online / streaming Test-Time Training

MapAnything

StreamVGGT

LingBot-map LongStream MonST3RPAGE-4D Point3R Spann3R STream3R

WorldMirror

OmniVGGT

MASt3R MUSt3R

WinT3R TTT3R Scal3R LoGeR

DUSt3R Fast3R

AMB3R

CUT3R

3 VGGT

DA3

𝜋

CO3D [75] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ RealEstate10K [134] ✓ ✓ ScanNet [21] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ScanNet++ [121] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ARKitScenes [5] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Habitat [79] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ BlendedMVS [120] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ MegaDepth [50] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ DL3DV [57] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ MapFree [2] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ WildRGBD [109] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Waymo [87] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ VirtualKITTI [8] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ TartanAir [100] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Unreal4K [90] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ MVS-Synth [34] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ HyperSim [76] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Mapillary [1] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ASE [3] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ADT [69] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ PointOdyssey [131] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Spring [64] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Dynamic Replica [40] ✓ ✓ ✓ ✓ ✓ ✓ ParallelDomain-4D [91] ✓ SAIL-VOS 3D [32] ✓ OmniObject3D [106] ✓ ✓ ✓ ✓ ✓ ✓ ✓ StaticThings3D [81] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ TartanGround [70] ✓ ✓ Matterport3D [11] ✓ ✓ ✓ ✓ ✓ ✓ BEDLAM [7] ✓ ✓ ✓ UASOL [6] ✓ ✓ ✓ ✓ MVImgNet [122] ✓ ✓ ✓ CoP3D [84] ✓ ✓ ✓ EDEN [47] ✓ ✓ ✓ ✓ IRS [97] ✓ ✓ ✓ Synscapes [105] ✓ ✓ ✓ 3D Ken Burns [67] ✓ ✓ ✓ SmartPortraits [45] ✓ ✓ UrbanSyn [28] ✓ ✓ ✓ ✓ HOI4D [60] ✓ ✓ ✓ Kubric [29] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Replica [85] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Objaverse [22] ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ Texverse [127] ✓ GTA-SfM [96] ✓ ✓ ✓ ✓ ✓ ✓ SceneNet RGB-D [63] ✓ ✓ ✓ MatrixCity [49] ✓ ✓ ✓ ✓ ✓ ✓ Mid-Air [24] ✓ ✓ ✓ KITTI-360 [53] ✓ Gibson [108] ✓ HM3D [74] ✓ Taskonomy [124] ✓ ✓ ✓ ✓ ✓ MegaSynth [38] ✓ OmniWorld [135] ✓ ✓ Trellis [110] ✓ TauAgent [27] ✓ Structured3D [129] ✓ DIML Outdoor [18] ✓ DDAD [30] ✓ Argoverse [12] ✓ Lyft [31] ✓ PandaSet [111] ✓ DSEC [25] ✓ Driving Stereo [115] ✓ Cityscapes [20] ✓

#Datasets: total used (trained) 22 (11) 42 (42) 8 (8) 10 (8) 13 (13) 14 (14) 13 (13) 27 (19) 24 (14) 17 (17) 23 (15) 32 (32) 30 (30) 19 (18) 11 (4) 20 (6) 16 (14) 9 (6) 36 (29) 21 (13) 15 (12) 32 (0) 18 (18) 29 (13)

- Table 6 | Dataset licenses and basic properties. We summarize the scene type, real/synthetic domain, dynamic content, and public license or access terms for each dataset. “Non-Commercial” denotes noncommercial use, “Terms” denotes dataset-specific access terms, and “Not specified” indicates that no clear standalone public dataset license was identified.

Idx. Dataset Scene Type Real/Synth. Dynamic License / Access Terms

- 1 CO3D [75] Object Real Static CC BY-NC 4.0
- 2 RealEstate10K [134] Indoor+Outdoor Real Static CC BY 4.0
- 3 ScanNet [21] Indoor Real Static Non-Commercial
- 4 ScanNet++ [121] Indoor Real Static Non-Commercial
- 5 ARKitScenes [5] Indoor Real Static Non-Commercial
- 6 Habitat [79] Indoor Real Static Habitat dataset-specific terms
- 7 BlendedMVS [120] Indoor+Outdoor Synth. Static CC BY 4.0
- 8 MegaDepth [50] Outdoor Real Static CC BY 4.0 + source image licenses
- 9 DL3DV [57] Indoor+Outdoor Real Mostly Static DL3DV Terms
- 10 MapFree [2] Outdoor Real Static Non-Commercial
- 11 WildRGBD [109] Object Real Static MIT
- 12 Waymo [87] Outdoor Real Dynamic Non-Commercial
- 13 VirtualKITTI [8] Outdoor Synth. Dynamic CC BY-NC-SA 3.0
- 14 TartanAir [100] Indoor+Outdoor Synth. Dynamic CC BY 4.0
- 15 Unreal4K [90] Indoor+Outdoor Synth. Static MIT
- 16 MVS-Synth [34] Outdoor Synth. Mostly Static Non-Commercial
- 17 HyperSim [76] Indoor Synth. Static CC BY-SA 3.0
- 18 Mapillary [1] Outdoor Real Static CC BY-NC-SA
- 19 ASE [3] Indoor Synth. Static Non-Commercial
- 20 ADT [69] Indoor Real Dynamic Non-Commercial
- 21 PointOdyssey [131] Indoor+Outdoor Synth. Dynamic CC BY-NC-SA 4.0
- 22 Spring [64] Outdoor Synth. Dynamic CC BY 4.0
- 23 Dynamic Replica [40] Indoor Synth. Dynamic Non-Commercial
- 24 ParallelDomain-4D [91] Outdoor Synth. Dynamic CC BY-NC 4.0
- 25 SAIL-VOS 3D [32] Indoor+Outdoor Synth. Dynamic Non-Commercial
- 26 OmniObject3D [106] Object Real Static CC BY 4.0
- 27 StaticThings3D [81] Object Synth. Static Non-Commercial
- 28 TartanGround [70] Outdoor Synth. Static CC BY 4.0
- 29 Matterport3D [11] Indoor Real Static Non-Commercial
- 30 BEDLAM [7] Human-centric Synth. Dynamic Non-Commercial
- 31 UASOL [6] Outdoor Real Static CC BY-NC-SA 3.0
- 32 MVImgNet [122] Object Real Static MVImgNet Terms
- 33 CoP3D [84] Object Real Dynamic CC BY-NC 4.0
- 34 EDEN [47] Outdoor Synth. Static Research Use
- 35 IRS [97] Indoor Synth. Static Apache 2.0
- 36 Synscapes [105] Outdoor Synth. Dynamic Non-Commercial
- 37 3D Ken Burns [67] Indoor+Outdoor Synth. Static CC BY-NC-SA 4.0
- 38 SmartPortraits [45] Human-centric Real Dynamic Academic Use
- 39 UrbanSyn [28] Outdoor Synth. Dynamic CC BY-SA 4.0
- 40 HOI4D [60] Indoor Real Dynamic CC BY-NC 4.0
- 41 Kubric [29] Object Synth. Dynamic Apache 2.0
- 42 Replica [85] Indoor Real Static Research Use
- 43 Objaverse [22] Object Synth. Static Per-object CC licenses
- 44 TexVerse [127] Object Synth. Static Per-object CC licenses
- 45 GTA-SfM [96] Outdoor Synth. Static Not specified
- 46 SceneNet RGB-D [63] Indoor Synth. Static Non-Commercial
- 47 MatrixCity [49] Outdoor Synth. Static CC BY-NC 4.0
- 48 Mid-Air [24] Outdoor Synth. Static CC BY-NC-SA 4.0
- 49 KITTI-360 [53] Outdoor Real Dynamic CC BY-NC-SA 3.0
- 50 Gibson [108] Indoor Real Static Non-Commercial
- 51 HM3D [74] Indoor Real Static Non-Commercial
- 52 Taskonomy [124] Indoor Real Static Non-Commercial
- 53 MegaSynth [38] Indoor Synth. Static CC BY-NC-SA 4.0
- 54 OmniWorld [135] Outdoor Synth. Dynamic CC BY-NC-SA 4.0
- 55 Trellis [110] Object Synth. Static MIT
- 56 TauAgent [27] Indoor+Outdoor Synth. Dynamic Not specified
- 57 Structured3D [129] Indoor Synth. Static Structured3D Terms
- 58 DIML Outdoor [18] Outdoor Real Mostly Static Non-Commercial
- 59 DDAD [30] Outdoor Real Dynamic CC BY-NC-SA 4.0
- 60 Argoverse [12] Outdoor Real Dynamic CC BY-NC-SA 4.0
- 61 Lyft [31] Outdoor Real Dynamic CC BY-NC-SA 4.0
- 62 PandaSet [111] Outdoor Real Dynamic CC BY 4.0
- 63 DSEC [25] Outdoor Real Dynamic CC BY-SA 4.0
- 64 Driving Stereo [115] Outdoor Real Dynamic MIT
- 65 Cityscapes [20] Outdoor Real Dynamic Non-Commercial

#### C. The Collection of DA-Next-5M

This section details the data collection pipeline of DA-Next-5M. DA-Next-5M consists of 22K sequences with 5.5M frames in total. Each scene in the dataset contains an image sequence I ∈ R𝑁×𝐻×𝑊×3, depth maps D ∈ R𝐻×𝑊×𝑁, camera intrinsics K ∈ R3×3, and camera extrinsics Gc2w ∈ SE(3) for each frame. Fig. 4 showcases data samples from DA-Next-5M. Note that all evaluation scenes in SpatialBench are held out from the training splits of their respective datasets.

###### Table 7 | DA-Next-5M dataset statistics.

###### Id Dataset View R/S Fr. Sc.

- 1 Xperience [78] Ego Real 400K 100
- 2 Aria Digital Twin [69] Ego Syn. 86K 232
- 3 Colosseum [72] Wrist Syn. 334K 1,837
- 4 HOI4D [60] Ego Real 739K 2,466
- 5 RLBench [36] Wrist Syn. 1.2M 5,120
- 6 Robolab [118] Wrist Syn. 158K 132
- 7 RoboTwin [14] Wrist Syn. 2.6M 11,923

Aria Digital Twin (ADT). ADT [69] is an egocentric RGB-D dataset captured with Meta Project Aria smart glasses in two instrumented indoor environments. It comprises 200 sequences of natural daily activities performed by 9 participants interacting with 398 object instances (344 static, 74 dynamic). We process the raw data into 232 sequences with 86K frames in total at a resolution of 512 × 512 using official preprocessing scripts. We observe scattered outlier points along object boundaries in scene visualizations and apply a unified depth post-processing step to refine the depth maps accordingly.

HOI4D. HOI4D [60] is a large-scale 4D egocentric dataset for category-level human-object interaction, containing 2.4M RGB-D frames across 4,000 sequences collected by 9 participants interacting with 800 object instances from 16 categories in 610 distinct indoor rooms. Each frame provides RGB and depth from a head-mounted RGBD camera with a resolution of 1920 × 1080, together with camera 6-DoF poses, 3D hand poses, category-level object poses, panoptic segmentation, and reconstructed scene point clouds. We directly use a subset of the raw HOI4D data (2.5K scenes and 739K frames) with only depth post-processing filtering (Appendix A.4) applied.

RLBench. RLBench [36] is a large-scale robot learning benchmark and simulation environment built on CoppeliaSim [77], featuring more than 100 distinct hand-designed manipulation tasks ranging from simple reaching to long-horizon multi-stage operations. Each task provides an unlimited supply of expert demonstrations via built-in motion planners. Visual observations include RGB, depth, and segmentation from both an over-the-shoulder stereo camera and an eye-in-hand monocular camera, with pixel-accurate synthetic depth and ground-truth 6-DoF camera poses available at every frame. Since the original simulation’s wrist-view camera does not capture the gripper, we adjust the wrist camera pose relative to the end-effector and increase the resolution to 1280 × 720 to broaden the field of view. We collect around 50 episodes per task using motion planners with random seeds across 103 tasks to form our dataset, with 1.2M frames in total.

Robo Colosseum. The Robo Colosseum [72] is a robotic manipulation benchmark built on top of RLBench that introduces systematic environmental perturbations for evaluating generalization. It provides 20 diverse manipulation tasks evaluated across 14 perturbation axes, including object color, texture, and size, background and tabletop appearance, lighting conditions, number of distractors, physical properties, and camera pose, with each perturbation applied independently and in combination. RGB, depth, and camera pose annotations are identical in format to RLBench. We similarly adjust the initial wrist camera placement and set the rendering resolution to 1280 × 720. We collect 100 episodes per task

with random seeds across 19 tasks, which is 1.8K episodes with 334K frames. Domain randomization is configured as follows: the number of distractor objects is set to 5–8; camera pose perturbations follow euler_range = [[−0.05,−0.05,−0.05], [0.05,0.05,0.05]] and position_range = [[0,0,0], [0,0,0.3]]; surface colors (e.g., table and objects) are randomized within color_range = [[0.25,0.25,0.25], [1.0,1.0,1.0]]; lighting color is randomized within [[0,0,0], [0.5,0.5,0.5]]; object scale is perturbed in the range [0.8, 1.0]; and texture randomization is enabled for objects, the table surface, and the background.

RoboTwin. RoboTwin [14] is a large-scale simulation-based dataset and benchmark for bimanual robotic manipulation, built on top of the RoboTwin Object Dataset (RoboTwin-OD) comprising 731 object instances across 147 categories with rich manipulation annotations (grasp points, functional points, object axes) and diverse language descriptions. It spans 50 dual-arm collaborative manipulation tasks executed across 5 heterogeneous robot embodiments. For DA-Next-5M, we collect data using five robot embodiments: Aloha-AgileX, ARX-X5, Franka, Piper, and UR5 across 50 bimanual tasks, which is 11K scenes with 2.6M frames in total. The wrist camera resolution is set to 1280 × 720 with a field of view of 60◦, and approximately 25 episodes are collected per task. Domain randomization is enabled with random backgrounds, cluttered table setups, a clean background rate of 0.2, random table height variation of 0.1, and random lighting. The wrist camera position is randomly switched between two fixed placements across episodes. For each episode, the image sequence from either the left or right arm is randomly selected for collection. The RoboTwin dataset has 12K episodes, comprising 2.6M frames in total.

Robolab. Robolab [118] is a high-fidelity simulation benchmarking framework for taskgeneralist robotic policies, developed by NVIDIA and built on Isaac Sim. It introduces the Robolab-120 benchmark comprising 120 tasks organized along three competency axes: visual, procedural, and relational, each at three difficulty levels, with LLM-enabled generation of novel scenes and tasks. Scenes feature photorealistic assets and physically accurate simulation, with ground-truth RGB, depth, and 6-DoF camera poses from wrist-mounted and external cameras. For DA-Next-5M, we use 𝜋0.5 [35] as the trajectory generator to collect data across 108 tasks under 2 background settings, yielding a total of 158K frames at a resolution of 1280 × 720.

#### D. Detail of DA-Next

###### D.1. Model Architecture

DA-Next is built upon the Giant variant of Depth-Anything-3 [54]. The backbone adopts the DINOv2 [68] ViT-Giant (vitg) architecture, in which Alternating Attention, QK-Norm, and Rotary Position Embedding (RoPE) are enabled starting from the 13th layer, while the cat_token and scale_token are retained. Multi-scale features are extracted from the 19th, 27th, 33rd, and 39th layers. The depth and ray prediction heads are instantiated as a DualDPT module with an input dimension of 3072, a feature dimension of 256, output channels of [256,512,1024,1024], and an output dimension of 2. The Scale Head is implemented as a 3-layer MLP (input 1536, hidden 1024, ReLU activation, Softplus output), and the Camera Encoder has an output dimension of 1536. All weights are initialized from the officially released da3-giant-1.1 checkpoint.

###### D.2. Training Objective

In DA-Next, all input images I = {𝐼𝑖}𝑖𝑁=1, together with the available camera parameters C = {𝐶𝑖}𝑖𝑁=1 = {𝐾𝑖,𝐺𝑖}𝑖𝑁=1 (if provided), are fed into the network G, which predicts the complete

ray maps Rˆ , depth maps Dˆ , scale 𝑆ˆ, and their corresponding confidence maps Yˆ 𝑑,Yˆ𝑟 in an end-to-end manner:

G (I,C) = (Rˆ ,Dˆ ,Yˆ 𝑑,Yˆ𝑟,𝑆ˆ). (28) The total training objective is a weighted sum of five task-specific ℓ1 terms,

L = Ldepth + 𝛼 Lgrad + Lray + Lpmap + Lscale, (29)

with 𝛼 = 1. Let 𝑚𝑝 ∈{0,1} denote the per-pixel validity mask of the ground truth and Ω = {𝑝 | 𝑚𝑝 = 1} the set of valid pixels across all 𝑁 views, with |Ω| its cardinality. We detail each term below.

Confidence-weighted regression. Following Wang et al. [99], Leroy et al. [48], each dense prediction is supervised with a confidence-weighted ℓ1 regression term. Given a prediction 𝑥ˆ, its confidence 𝑦ˆ, and the ground truth 𝑥, the per-pixel regression and confidence terms are

ℓreg(𝑥ˆ, 𝑥 ; 𝑝) = ∥𝑥ˆ𝑝 − 𝑥𝑝∥1, ℓconf(𝑥ˆ, 𝑥, 𝑦ˆ ; 𝑝) = 𝛾 𝑦ˆ𝑝 ℓreg(𝑥ˆ, 𝑥 ; 𝑝) − 𝛽 log 𝑦ˆ𝑝, (30) where 𝛾 and 𝛽 balance the data term and the confidence regularizer that prevents 𝑦ˆ →0. Depth loss. For the predicted depth map Dˆ with confidence Yˆ 𝑑, the depth term aggregates the confidence-weighted and plain ℓ1 terms over valid pixels:

Ldepth(Dˆ ,D;Yˆ 𝑑) = |Ω1| ∑︁ 𝑝∈Ω

𝛾𝑌ˆ𝑑,𝑝 𝐷 ˆ 𝑝 − 𝐷𝑝 − 𝛽 log𝑌ˆ𝑑,𝑝 + 𝐷 ˆ 𝑝 − 𝐷𝑝 . (31)

Depth-gradient loss. To preserve sharp edges while enforcing smoothness on planar regions, we penalize the discrepancy between the depth gradients of the prediction and the ground truth:

###### Lgrad(Dˆ ,D) = ∥∇𝑥Dˆ − ∇𝑥D∥1 + ∥∇𝑦Dˆ − ∇𝑦D∥1, (32)

where ∇𝑥 and ∇𝑦 are the horizontal and vertical finite difference operators. In practice, this term is evaluated in a multi-scale manner over 𝐽 dyadic scales (stride 2𝑗, 𝑗 = 0, . . . , 𝐽−1) and averaged.

Ray loss. The ray map Rˆ ∈ R𝑁×𝐻×𝑊×6 encodes per-pixel camera origins oˆ and viewing directions dˆ. With its confidence Yˆ𝑟, the ray loss mirrors the depth term:

Lray(Rˆ ,R;Yˆ𝑟) = |Ω1| ∑︁ 𝑝∈Ω

𝛾𝑌ˆ𝑟,𝑝 ∥Rˆ 𝑝 − R𝑝∥1 − 𝛽 log𝑌ˆ𝑟,𝑝 + ∥Rˆ 𝑝 − R𝑝∥1 . (33)

Point-map loss. We recover the 3D world points by back-projecting depth along the predicted rays, Pˆ = Dˆ ⊙ dˆ + oˆ, and supervise them with a masked ℓ1 against the ground-truth world points P:

∑︁

1 |Ω|

Lpmap(Dˆ ⊙ dˆ + oˆ, P) =

P ˆ 𝑝 − P𝑝 1. (34)

𝑝∈Ω

Scale loss. The predicted global scale 𝑆ˆ is supervised against the ground-truth scale factor 𝑆 via a log-space ℓ1 term,

Lscale(𝑆ˆ,𝑆) = 𝑓log(𝑆ˆ) − 𝑓log(𝑆) 1, 𝑓log : x ↦→ ∥xx∥ log(1 + ∥x∥), (35)

which compresses the dynamic range of absolute scale and yields stable gradients across scenes of drastically different sizes.

Ground-truth preprocessing and scale target. To remove the inherent scene-scale ambiguity across heterogeneous training data and keep the magnitudes of different modalities consistent, all ground-truth signals are canonicalized in a two-step procedure before loss computation. (i) Coordinate canonicalization. The first frame is taken as the reference: we apply the cam-to-world transform of the first camera to all extrinsics, and accordingly transform the ground-truth world points P into the first-camera coordinate frame. (ii) Scale normalization. We then compute the per-scene scale factor as the mean ℓ2 norm of the valid reprojected world points,

∑︁

1 |Ω|

∥P𝑝∥2, (36)

𝑆 =

𝑝∈Ω

and divide the ground-truth world points, depths and camera translations by 𝑆:

P←P/𝑆, D←D/𝑆, t←t/𝑆. (37)

The resulting scalar 𝑆 is also kept as the regression target of the scale head, so that 𝑆ˆ in Lscale learns to recover the absolute metric scale that the other dense predictions are invariant to. During inference, the predicted 𝑆ˆ is multiplied back to lift the normalized geometry into its metric space. We set 𝛾 = 1.0 and 𝛽 = 0.2 for all confidence-weighted terms, use 𝐽 = 3 scales for the multi-scale gradient loss, and apply quantile filtering with 𝜏 = 0.98 to suppress outliers.

###### D.3. Camera Conditioning Training

We adopt a stochastic pose-conditioning scheme at training time: for each mini-batch we flip a Bernoulli coin with probability 𝑝 ∈ [0,1], and inject the ground-truth camera as an auxiliary input only when the coin turns up, i.e.

𝑢 ∼ Bernoulli(𝑝), (Rˆ ,Dˆ ,Yˆ 𝑑,Yˆ𝑟,𝑆ˆ) = G(I,G˜ ,K), 𝑢 = 1, G(I), 𝑢 = 0,

(38)

where G˜ denotes the canonicalized ground-truth extrinsics described below, and K = {𝐾𝑖}𝑖𝑁=1 the intrinsics. When 𝑢 = 0, the encoder instead receives a learnable placeholder camera token

e𝑐; when 𝑢 = 1, (G˜ ,K) is tokenized and replaces e𝑐, so that the encoder attends to geometrically grounded pose cues.

Extrinsic Normalization for Conditioning. The extrinsics fed into the network are preprocessed independently from the supervision pipeline of Eq. (36), so as to remain agnostic to the absolute metric scale. Specifically, given the world-to-camera matrices {𝐺𝑖}𝑖𝑁=1, we (i) right-multiply by 𝐺−1

1 so that the first view becomes the canonical frame, and (ii) divide the translation components by the median camera-centre distance 𝑑¯:

𝐺˜𝑖 = 𝐺𝑖 𝐺1−1, 𝐺˜𝑖[:3,3] ← 𝐺˜𝑖[:3,3] /max(𝑑¯, 𝜖), (39) with 𝑑¯ = median𝑖 ∥c𝑖∥2 and c𝑖 the camera centre extracted from 𝐺˜−1

𝑖 . This keeps the conditioning signal in a scale-invariant canonical frame and prevents trivial leakage of the ground-truth scale through the pose input (which is instead supervised via Lscale).

###### D.4. Training Datasets

We train DA-Next model using images from 18 datasets, including: DA-Next-5M (7 datasets in total), HyperSim [76], Infinigen [73], Spring [64], MapFree [2], Matterport 3D [74], MVSSynth [34], ScanNet++ [121], TartanAir [100], Unreal 4K [90], Virtual KITTI [8], Waymo [87].

- Table 8 | Training Datasets Statistics. Each training epoch is composed of 18 datasets, and their dataset mixture ratios are reported as the training prob.

Cat. Id. Dataset Scene R/S Dyn. # Fr. Met. Prob.

- 1 HyperSim [76] Indoor Syn. Stat. 70K ✓ 2.8
- 2 Infinigen [73] Indoor Syn. Stat. 142K ✓ 2.5
- 3 MapFree [2] Outdoor Real Stat. 2.6M ✓ 6.5
- 4 Matterport 3D [74] Indoor Real Stat. 1.9M ✓ 7.3
- 5 MVS-Synth [34] Outdoor Syn. Stat. 12K ✓ 0.4
- 6 ScanNet++ [121] Indoor Real Stat. 7.8M ✓ 4.1
- 7 Spring [64] Outdoor Syn. Dyn. 5K ✓ 2.3
- 8 Tartanair [100] Mixed Syn. Stat. 3M ✓ 6.8
- 9 Unreal 4K [90] Outdoor Syn. Stat. 16K ✓ 0.4
- 10 Vkitti [8] Outdoor Syn. Dyn. 42K ✓ 1.5
- 11 Waymo [87] Outdoor Real Dyn. 7.9M ✓ 5.1

DA-Next-5M

- 12 Aria Digital Twin [69] Indoor Syn. Dyn. 87K ✓ 8.0
- 13 Colosseum [72] Indoor Syn. Dyn. 334K ✓ 5.5
- 14 HOI4D [60] Indoor Real Dyn. 739K ✓ 11.1
- 15 RLBench [36] Indoor Syn. Dyn. 1.2M ✓ 5.5
- 16 Robolab [118] Indoor Syn. Dyn. 158K ✓ 11.1
- 17 RoboTwin [14] Indoor Syn. Dyn. 1.8M ✓ 11.1
- 18 Xperience [78] Mixed Real Dyn. 400K ✓ 8.0

General3DDatasets

These datasets cover normal, egocentric, and wrist view perspectives, both synthetic and real-world content, indoor and outdoor environments, as well as static and dynamic scenes. Such a diverse composition preserves a strong generalization capability for DA-Next. Table 8 summarizes the statistics of the datasets we used. In each epoch, we sample a fixed total number of samples from the training datasets, with their mixture ratio indicated by the “Training Prob” column in the table.

###### D.5. Frame Sampling Strategy

For every batch, we select between 2 and 18 frames from multiple random training scenes while maintaining a constant total of 18 frames within each batch. We sample each batch of images based on the Euclidean distance to the camera pose. For each frame, all other frames are ranked according to their pose distance, and the top 𝑁 (𝑁 depends on the frame density in each dataset) closest frames are selected as its valid range. Then, for each sequence, we randomly choose one frame as the anchor frame and sample the remaining frames from its valid range.

###### D.6. Training Configuration

Our DA-Next architecture follows DA3-Giant [54] with 𝐿 = 40 Transformer blocks and is initialized by pre-trained weights. During training, each batch incorporates ground-truth camera information as auxiliary input with probability 𝑝 = 20%. The training runs end-to-end on 4 NVIDIA H200 GPUs over seven days. Training is conducted in BF16 mixed precision with a fixed random seed of 42 for 10 epochs in total. The per-device batch size is 18 frames, and gradient accumulation is performed every 2 steps. During training, the backbone, depth and ray prediction head, camera encoder, and scale head are jointly optimized, whereas the camera decoder is kept frozen (we use the ray map branch). Since the 3D Gaussian Splatting branch is not involved in this work, its corresponding GS Head is also frozen. Gradient checkpointing is further enabled to reduce GPU memory consumption. We employ the AdamW optimizer with 𝛽1 = 0.9, 𝛽2 = 0.95, 𝜖 = 1 × 10−8, and a weight decay of 0.01. A layer-wise learning rate scheme is adopted: 1 × 10−5 for the backbone, 2 × 10−5 for the prediction head, camera encoder. The learning rate follows a cosine decay schedule with linear warm-up, where the warm-up stage

lasts 5,000 steps and the minimum learning rate factor is 0.1. Gradient clipping is applied with a maximum norm of 1.0. To enhance robustness across varying aspect ratios, multi-scale training is performed over 17 resolutions, where the width is fixed at 504 and the height ranges from 280 to 504 with a step of 14.

#### E. Additional Findings

###### E.1. Do More Input Frames Always Lead to Better Results?

A natural assumption in 3D reconstruction is that more input frames yield better geometric estimates. However, Tab. 1 reveals a more nuanced picture. Under the sparse regime, where inputs consist of only a few frames with large inter-frame baselines and limited overlap, most models struggle with geometric estimation due to insufficient visual correspondence. Increasing the number of input frames from sparse to medium consistently improves both depth and camera pose metrics across paradigms, confirming that a moderate level of multi-view overlap is beneficial. However, further extending to the dense regime does not uniformly bring additional gains. For many feed-forward models, dense inputs introduce redundant or near-duplicate frames that add memory pressure without providing useful new geometric constraints, leading to stagnating or even degraded performance on reconstruction. This non-monotonic behavior suggests that the relationship between input density and reconstruction quality is task- and model-dependent, rather than strictly increasing.

Takeaway: For bounded 3D reconstruction tasks, there exists an optimal input density range that maximizes reconstruction quality. Too few frames leave geometric constraints underspecified, while excessive inputs can introduce redundancy and hurt performance, making careful input density selection as important as model choice.

###### E.2. How to Select the Right Spatial Foundation Model for Your Task?

Fig. 20 presents the overall ranking of the best-performing methods from each paradigm (FeedForward, Chunk-wise, Online, and TTT) across different domain subsets of SpatialBench. The overall ranking is computed from the weighted Average column of each per-domain sub-table, aggregating four complementary metrics: AbsRel, AUC@30, ATE, and F-Score. We partition SpatialBench into five domain groups: G1: Indoor, G2: Outdoor, G3: Driving, G4: Wrist-view, and G5: Ego-view. The colored cells report the ranking of each method on each domain, while the scatter points indicate whether the method’s training data includes in-domain samples from the corresponding group. Fig. 21, Fig. 22, and Fig. 23 present the per-domain ranking breakdowns under the sparse, medium, and dense input settings, respectively.

Fig. 20 reveals that model rankings are highly inconsistent across domain groups and input density settings. This inconsistency stems from two factors: the training data coverage of each method, and their architectural design choices that favor different operating regimes (e.g., long-sequence streaming methods naturally excel on dense, extended trajectories but may underperform on sparse multi-view inputs). This finding gives a guideline for downstream deployment: selecting a spatial foundation model for a target application requires verifying not only its general benchmark performance, but also whether its training mixture covers the relevant domain. Furthermore, for practitioners preparing fine-tuning datasets, these results suggest that domain match matters more than dataset volume: prioritizing data from the target domain, even in modest quantities, is likely to yield greater gains than simply increasing the total number of training scenes from unrelated distributions.

###### Takeaway: No single model dominates across all domains and input regimes. Domain coverage in training data is the primary predictor of per-domain ranking, and should be the first consideration when selecting or fine-tuning a spatial foundation model for a specific downstream application.

[Figure 661]

- Figure 20 | Training Data Domain Coverage vs. Overall Ranking Across Domain Groups. We compare the best-performing methods from each paradigm on five domain groups in SpatialBench: Indoor, Outdoor, Autonomous Driving, Wrist-view, and Ego-view, where 𝑁 denotes the number of datasets in each group. Colored cells report the per-domain ranking of each method, while scatter points indicate whether the corresponding training set includes in-domain data from that group.

[Figure 662]

- Figure 21 | Training Data Domain Coverage vs. Overall Ranking Across Domain Groups (Sparse). We compare the best-performing methods from each paradigm on five domain groups in SpatialBench under the sparse regime, where 𝑁 denotes the number of datasets in each group. Colored cells report the per-domain ranking of each method, while scatter points indicate whether the corresponding training set includes in-domain data from that group.

[Figure 663]

- Figure 22 | Training Data Domain Coverage vs. Overall Ranking Across Domain Groups (Medium). We compare the best-performing methods from each paradigm on five domain groups in SpatialBench under the medium regime, where 𝑁 denotes the number of datasets in each group. Colored cells report the per-domain ranking of each method, while scatter points indicate whether the corresponding training set includes in-domain data from that group.

[Figure 664]

- Figure 23 | Training Data Domain Coverage vs. Overall Ranking Across Domain Groups (Dense). We compare the best-performing methods from each paradigm on five domain groups in SpatialBench under the dense regime, where 𝑁 denotes the number of datasets in each group. Colored cells report the per-domain ranking of each method, while scatter points indicate whether the corresponding training set includes in-domain data from that group. Empty cells indicate that the method runs out of memory under the corresponding input regime.

#### F. The Complete SpatialBench Results

Per-regime Breakdowns of the Main Table. We present detailed per-regime results for the single-frame, sparse, medium, and dense input settings in Tab. 9, Tab. 10, Tab. 11, and Tab. 12, respectively, complementing the aggregated results in Tab. 1.

Performance Comparison on Each Dataset. We present per-dataset metric breakdowns for all evaluated methods in this section.

- • 7-Scenes. Per-dataset results on 7-Scenes are reported in Table 13.
- • ADT. Per-dataset results on Aria Digital Twin (ADT) are reported in Table 14.
- • DROID. Per-dataset results on DROID are reported in Table 15.
- • DTU. Per-dataset results on DTU are reported in Table 16.
- • ETH3D. Per-dataset results on ETH3D are reported in Table 17.
- • HiRoom. Per-dataset results on HiRoom are reported in Table 18.
- • KITTI Odometry. Per-dataset results on KITTI Odometry are reported in Table 19.
- • Lingbot-Depth. Per-dataset results on Lingbot are reported in Table 20.
- • NRGBD. Per-dataset results on NRGBD are reported in Table 21.
- • OmniWorld. Per-dataset results on OmniWorld are reported in Table 22.
- • RLBench. Per-dataset results on RLBench are reported in Table 23.
- • Robolab. Per-dataset results on Robolab are reported in Table 24.
- • RoboTwin. Per-dataset results on RoboTwin are reported in Table 25.
- • Xperience. Per-dataset results on Xperience are reported in Table 26.
- • ScanNet++. Per-dataset results on ScanNet++ are reported in Table 27.
- • Tanks and Temples. Per-dataset results on Tanks and Temples are reported in Table 28.
- • TUM RGB-D. Per-dataset results on TUM RGB-D are reported in Table 29.
- • Virtual KITTI. Per-dataset results on Virtual KITTI are reported in Table 30.
- • Waymo. Per-dataset results on Waymo are reported in Table 31.

Metric Depth Evaluation. Tab. 32 compares the native metric-depth prediction quality of all six metric-scale-capable methods on SpatialBench, evaluated without median/scale alignment across single-frame, sparse, and medium input settings.

- Table 9 | Detailed Results on the Single-Frame Setting. The best, second-best, and third-best results in each column are highlighted in deep blue , medium blue , and light blue , respectively. Out-of-memory cells are shaded light red . Within each sub-category, the bold value marks the in-group best.

Depth AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ Optimization-based

#Params (M)

Method

DUSt3R 571.17 0.385 5.337 0.992 0.391 0.524 0.685 MASt3R 688.64 0.456 10.14 0.973 0.373 0.508 0.684

###### End-to-End Feed-Forward

VGGT 1256.54 0.184 0.779 0.608 0.517 0.639 0.773 Fast3R 647.55 0.350 3.454 1.139 0.339 0.463 0.631 FastVGGT 1157.94 0.183 0.761 0.608 0.517 0.640 0.773 MUSt3R 423.43 0.429 8.637 0.934 0.393 0.524 0.699 MapAnything 1228.49 0.451 11.99 0.854 0.441 0.583 0.747 OmniVGGT 1217.49 0.188 0.795 0.620 0.525 0.649 0.791 𝜋3 958.70 0.478 16.85 0.983 0.542 0.665 0.794 𝜋3-X 1360.03 0.371 8.834 0.631 0.522 0.651 0.802 AMB3R 1563.12 0.466 15.32 0.658 0.544 0.671 0.802 DA3-Small 34.30 0.385 6.358 0.948 0.300 0.431 0.625 DA3-Base 135.37 0.349 5.281 0.842 0.359 0.501 0.691 DA3-Large 410.94 0.333 5.649 0.763 0.449 0.579 0.742 DA3-Giant 1355.67 0.368 7.494 0.724 0.504 0.628 0.771 DA3-Nested 1689.85 0.364 7.358 0.800 0.500 0.633 0.778 WorldMirror 1263.34 0.349 7.695 0.707 0.485 0.619 0.776 VGGT-Omega 1143.81 0.516 20.72 1.086 0.552 0.671 0.812 DANext† (Ours) 1303.76 0.166 0.985 0.646 0.511 0.636 0.802

###### Online

Spann3r224 658.69 0.370 12.39 2.345 0.288 0.417 0.596 CUT3R 793.31 0.247 1.356 0.712 0.409 0.555 0.720 MonST3R 571.17 0.309 2.500 0.999 0.348 0.470 0.633 Point3R 828.01 0.379 6.806 0.783 0.400 0.534 0.703 Stream3R-S 1190.60 0.409 11.3 0.672 0.564 0.691 0.811 Stream3R-W 1190.60 0.409 11.3 0.672 0.564 0.691 0.811 StreamVGGT 1256.54 0.219 1.696 0.583 0.523 0.646 0.784 Page4D 1256.81 0.228 1.977 0.636 0.467 0.606 0.762 InfiniteVGGT 1256.54 0.217 1.643 0.583 0.524 0.645 0.783 Wint3R 749.46 0.619 25.47 1.053 0.434 0.555 0.719 LongStream-B 1190.60 0.523 17.1 0.682 0.458 0.597 0.770 LongStream-S 1190.60 0.523 17.1 0.682 0.458 0.597 0.770 LingbotMap∗-W 1157.94 0.333 4.470 0.755 0.437 0.572 0.741 LingbotMap∗-S 1157.94 0.333 4.470 0.755 0.437 0.572 0.741

###### Chunk-wise

VGGT-Long 1256.54 0.184 0.779 0.608 0.517 0.639 0.773 𝜋3-Long 958.70 0.478 16.85 0.983 0.542 0.665 0.794 DA3-Streaming 1355.67 0.368 7.494 0.724 0.504 0.628 0.771

###### SLAM-based

MASt3R-SLAM 688.64 0.348 1.553 1.237 0.165 0.268 0.471 VGGT-SLAM 1256.54 0.184 0.779 0.608 0.517 0.639 0.773

###### Test-Time Training

TTT3R 793.31 0.247 1.357 0.712 0.409 0.555 0.720 Scal3R 1266.14 0.227 1.583 0.613 0.519 0.644 0.775 LoGeR 1254.62 0.251 2.718 0.635 0.527 0.666 0.800 LoGeR∗ 1254.60 0.200 1.320 0.637 0.540 0.668 0.803

- Table 10 | Detailed Results on the Sparse Setting. The best, second-best, and third-best results in each column are highlighted in deep blue , medium blue , and light blue , respectively. Out-of-memory cells are shaded light red . Within each sub-category, the bold value marks the in-group best.

Depth Camera

#Params (M)

Method

AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ RAcc3 ↑ RAcc5 ↑ TAcc3 ↑ TAcc5 ↑ AUC@5↑ AUC@15↑ AUC@30↑

###### Optimization-based

DUSt3R 571.17 0.257 2.693 1.567 0.295 0.420 0.600 0.527 0.654 0.242 0.350 0.184 0.374 0.498 MASt3R 688.64 0.209 0.606 1.246 0.309 0.456 0.639 0.587 0.701 0.301 0.411 0.244 0.438 0.568

###### End-to-End Feed-Forward

VGGT 1256.54 0.105 0.105 0.673 0.498 0.627 0.778 0.734 0.833 0.454 0.550 0.405 0.585 0.700 Fast3R 647.55 0.260 0.760 1.576 0.228 0.340 0.508 0.347 0.486 0.156 0.235 0.109 0.264 0.392 FastVGGT 1157.94 0.113 0.098 0.666 0.457 0.590 0.752 0.642 0.773 0.384 0.472 0.314 0.503 0.631 MUSt3R 423.43 0.165 0.276 0.999 0.366 0.502 0.686 0.622 0.726 0.343 0.435 0.273 0.482 0.614 MapAnything 1228.49 0.153 1.079 1.337 0.361 0.512 0.701 0.608 0.762 0.300 0.423 0.244 0.446 0.579 OmniVGGT 1217.49 0.117 0.119 0.669 0.534 0.671 0.810 0.620 0.746 0.416 0.507 0.332 0.537 0.665 𝜋3 958.70 0.092 0.258 0.855 0.547 0.686 0.826 0.769 0.854 0.474 0.586 0.412 0.619 0.742 𝜋3-X 1360.03 0.084 0.084 0.599 0.576 0.710 0.833 0.756 0.837 0.491 0.595 0.427 0.627 0.741 AMB3R 1563.12 0.088 0.083 0.617 0.534 0.668 0.812 0.743 0.844 0.475 0.591 0.412 0.619 0.739 DA3-Small 34.30 0.191 0.270 1.057 0.308 0.440 0.633 0.392 0.568 0.172 0.300 0.127 0.336 0.476 DA3-Base 135.37 0.159 0.210 0.937 0.386 0.520 0.690 0.528 0.678 0.292 0.399 0.222 0.427 0.566 DA3-Large 410.94 0.128 0.198 0.839 0.470 0.601 0.753 0.662 0.776 0.443 0.528 0.361 0.559 0.688 DA3-Giant 1355.67 0.095 0.107 0.608 0.563 0.689 0.821 0.792 0.871 0.586 0.682 0.525 0.699 0.785 DA3-Nested 1689.85 0.106 0.199 0.805 0.564 0.692 0.814 0.784 0.855 0.577 0.682 0.519 0.691 0.779 WorldMirror 1263.34 0.139 0.162 0.798 0.440 0.583 0.748 0.697 0.793 0.394 0.506 0.320 0.532 0.660 VGGT-Omega 1143.81 0.077 0.362 1.089 0.556 0.702 0.843 0.808 0.892 0.585 0.678 0.504 0.695 0.803 DANext† (Ours) 1303.76 0.050 0.073 0.554 0.647 0.772 0.893 0.815 0.900 0.587 0.718 0.518 0.718 0.809

###### Online

Spann3r224 658.69 0.274 212.3 10.47 0.209 0.317 0.496 0.265 0.423 0.080 0.163 0.052 0.198 0.329 CUT3R 793.31 0.196 0.218 0.917 0.300 0.445 0.638 0.478 0.637 0.223 0.356 0.177 0.388 0.519 MonST3R 571.17 0.227 0.399 1.338 0.220 0.332 0.521 0.243 0.311 0.107 0.177 0.069 0.172 0.269 Point3R 828.01 0.221 0.280 1.044 0.251 0.379 0.559 0.194 0.372 0.058 0.123 0.028 0.177 0.339 Stream3R-S 1190.60 0.114 0.120 0.705 0.450 0.597 0.765 0.618 0.774 0.351 0.437 0.278 0.471 0.603 Stream3R-W 1190.60 0.117 0.131 0.739 0.442 0.584 0.753 0.610 0.762 0.344 0.430 0.272 0.464 0.597 StreamVGGT 1256.54 0.154 0.362 1.243 0.314 0.424 0.609 0.638 0.771 0.339 0.435 0.283 0.472 0.598 Page4D 1256.81 0.112 0.089 0.646 0.438 0.587 0.758 0.551 0.730 0.295 0.428 0.224 0.456 0.608 InfiniteVGGT 1256.54 0.154 0.363 1.245 0.314 0.423 0.610 0.639 0.767 0.346 0.435 0.286 0.471 0.596 Wint3R 749.46 0.157 0.232 0.909 0.399 0.532 0.702 0.438 0.600 0.214 0.334 0.157 0.366 0.499 LongStream-B 1190.60 0.153 0.152 0.771 0.380 0.526 0.712 0.502 0.678 0.263 0.383 0.197 0.417 0.549 LongStream-S 1190.60 0.151 0.144 0.754 0.383 0.531 0.716 0.501 0.677 0.258 0.377 0.196 0.413 0.543 LingbotMap∗-W 1157.94 0.138 0.138 0.759 0.376 0.536 0.730 0.696 0.811 0.359 0.469 0.303 0.520 0.650 LingbotMap∗-S 1157.94 0.138 0.138 0.759 0.376 0.536 0.730 0.696 0.811 0.359 0.469 0.303 0.520 0.650

###### Chunk-wise

VGGT-Long 1256.54 0.105 0.105 0.673 0.498 0.627 0.778 0.734 0.833 0.454 0.550 0.405 0.585 0.700 𝜋3-Long 958.70 0.092 0.258 0.855 0.547 0.686 0.826 0.769 0.854 0.474 0.586 0.412 0.619 0.742 DA3-Streaming 1355.67 0.095 0.107 0.608 0.563 0.689 0.821 0.790 0.871 0.588 0.682 0.525 0.699 0.785

###### SLAM-based

MASt3R-SLAM 688.64 0.336 1.715 2.282 0.120 0.194 0.347 0.317 0.401 0.066 0.095 0.047 0.118 0.190 VGGT-SLAM 1256.54 0.105 0.105 0.673 0.498 0.627 0.778 0.734 0.833 0.454 0.550 0.405 0.585 0.700

###### Test-Time Training

TTT3R 793.31 0.202 0.257 0.966 0.282 0.428 0.628 0.452 0.591 0.197 0.302 0.153 0.342 0.469 Scal3R 1266.14 0.114 0.091 0.593 0.535 0.669 0.810 0.734 0.833 0.484 0.581 0.418 0.610 0.732 LoGeR 1254.62 0.095 0.077 0.569 0.546 0.700 0.833 0.695 0.821 0.411 0.538 0.348 0.564 0.687 LoGeR∗ 1254.60 0.077 0.071 0.558 0.581 0.725 0.854 0.715 0.824 0.424 0.547 0.353 0.576 0.708

- Table 11 | Detailed Results on the Medium Setting. The best, second-best, and third-best results in each column are highlighted in deep blue , medium blue , and light blue , respectively. Out-of-memory cells are shaded light red . Within each sub-category, the bold value marks the in-group best.

Depth Camera Trajectory PointCloud

#Params (M)

Method

AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ RAcc3 ↑ RAcc5 ↑ TAcc3 ↑ TAcc5 ↑ AUC@5↑ AUC@15↑ AUC@30↑ ATE↓ RPE𝑡 ↓ RPE𝑟 ↓ F-Score↑ Overall↓

###### Optimization-based

DUSt3R 571.17 0.276 0.860 1.447 0.247 0.361 0.529 0.402 0.543 0.172 0.273 0.130 0.315 0.448 1.691 0.386 4.096 0.343 0.142 MASt3R 688.64 0.259 1.871 1.898 0.259 0.379 0.554 0.477 0.615 0.230 0.344 0.176 0.381 0.522 1.911 0.236 2.997 0.370 0.382

###### End-to-End Feed-Forward

VGGT 1256.54 0.125 0.602 0.688 0.499 0.613 0.747 0.695 0.782 0.484 0.562 0.432 0.586 0.687 0.727 0.216 2.202 0.661 0.087 Fast3R 647.55 0.255 0.567 1.539 0.241 0.344 0.496 0.296 0.422 0.164 0.248 0.117 0.272 0.386 6.582 2.055 13.65 0.300 0.210 FastVGGT 1157.94 0.105 0.086 0.611 0.479 0.597 0.737 0.678 0.771 0.448 0.542 0.379 0.554 0.662 0.738 0.259 3.260 0.576 0.121 MUSt3R 423.43 0.162 0.327 0.966 0.388 0.522 0.696 0.618 0.734 0.361 0.475 0.296 0.507 0.643 3.097 0.613 2.686 0.507 0.230 MapAnything 1228.49 0.146 0.490 1.052 0.347 0.491 0.681 0.563 0.702 0.312 0.419 0.254 0.451 0.579 1.737 0.533 2.852 0.420 0.114 OmniVGGT 1217.49 0.111 0.096 0.649 0.518 0.645 0.780 0.609 0.726 0.426 0.527 0.340 0.542 0.665 1.491 0.355 2.768 0.595 0.104 𝜋3 958.70 0.082 0.229 0.822 0.563 0.689 0.814 0.742 0.830 0.501 0.613 0.433 0.636 0.749 0.565 0.128 1.240 0.649 0.080 𝜋3-X 1360.03 0.078 0.061 0.538 0.582 0.712 0.831 0.741 0.827 0.536 0.628 0.463 0.644 0.744 0.369 0.108 1.459 0.658 0.074 AMB3R 1563.12 0.085 0.074 0.580 0.539 0.661 0.795 0.704 0.799 0.496 0.588 0.429 0.613 0.727 0.645 0.223 1.881 0.554 0.123 DA3-Small 34.30 0.176 0.226 1.008 0.312 0.435 0.617 0.370 0.535 0.192 0.301 0.136 0.338 0.479 4.850 1.356 5.534 0.432 0.123 DA3-Base 135.37 0.142 0.171 0.884 0.395 0.521 0.683 0.494 0.626 0.289 0.407 0.227 0.430 0.562 3.865 0.976 3.928 0.515 0.100 DA3-Large 410.94 0.105 0.155 0.779 0.498 0.618 0.767 0.655 0.758 0.487 0.575 0.411 0.588 0.701 2.722 0.460 2.800 0.626 0.130 DA3-Giant 1355.67 0.086 0.088 0.578 0.572 0.686 0.812 0.749 0.834 0.587 0.663 0.532 0.684 0.776 1.161 0.285 2.253 0.742 0.073 DA3-Nested 1689.85 0.086 0.182 0.801 0.577 0.690 0.810 0.742 0.833 0.569 0.658 0.510 0.676 0.770 1.980 0.394 2.708 0.737 0.073 WorldMirror 1263.34 0.118 0.126 0.741 0.445 0.573 0.721 0.674 0.775 0.417 0.533 0.352 0.556 0.674 1.357 0.347 2.056 0.575 0.090 VGGT-Omega 1143.81 0.067 0.304 1.049 0.589 0.721 0.845 0.787 0.870 0.589 0.692 0.512 0.700 0.795 0.659 0.160 1.369 0.706 0.078 DANext† (Ours) 1303.76 0.035 0.062 0.520 0.715 0.830 0.928 0.806 0.880 0.630 0.733 0.553 0.733 0.819 1.442 0.251 1.602 0.727 0.072

###### Online

Spann3r224 658.69 0.252 145 9.194 0.208 0.315 0.486 0.198 0.358 0.085 0.160 0.049 0.205 0.361 4.312 1.118 6.784 0.254 0.249 CUT3R 793.31 0.189 0.423 1.095 0.271 0.396 0.576 0.335 0.506 0.160 0.273 0.110 0.316 0.469 2.676 0.373 4.271 0.286 0.385 MonST3R 571.17 0.241 0.595 1.551 0.162 0.252 0.413 0.161 0.221 0.037 0.078 0.026 0.100 0.195 2.234 0.526 12.99 0.081 0.502 Point3R 828.01 0.228 0.261 1.044 0.247 0.368 0.551 0.178 0.340 0.055 0.116 0.027 0.161 0.303 6.512 0.847 8.545 0.211 0.213 Stream3R-S 1190.60 0.204 0.725 1.414 0.295 0.395 0.544 0.392 0.514 0.223 0.299 0.178 0.317 0.427 5.717 0.621 5.650 0.348 0.594 Stream3R-W 1190.60 0.240 1.099 1.717 0.260 0.350 0.500 0.330 0.441 0.173 0.251 0.136 0.265 0.364 6.756 0.896 7.827 0.323 0.565 StreamVGGT 1256.54 0.171 0.509 1.458 0.271 0.372 0.539 0.566 0.679 0.305 0.405 0.251 0.437 0.562 4.940 0.526 3.820 0.397 0.154 Page4D 1256.81 0.107 0.079 0.613 0.454 0.589 0.748 0.531 0.712 0.334 0.464 0.246 0.479 0.618 0.855 0.255 2.574 0.423 0.118 InfiniteVGGT 1256.54 0.170 0.504 1.453 0.271 0.372 0.539 0.566 0.679 0.304 0.407 0.251 0.439 0.563 4.964 0.536 3.908 0.402 0.151 Wint3R 749.46 0.144 0.174 0.856 0.356 0.494 0.677 0.336 0.508 0.157 0.267 0.108 0.303 0.444 3.944 0.625 3.351 0.401 0.201 LongStream-B 1190.60 0.224 0.182 0.854 0.213 0.324 0.504 0.376 0.550 0.165 0.270 0.118 0.315 0.455 0.925 0.282 4.074 0.135 0.303 LongStream-S 1190.60 0.166 0.147 0.796 0.288 0.418 0.609 0.266 0.412 0.128 0.217 0.085 0.249 0.385 1.188 0.345 4.945 0.126 0.345 LingbotMap∗-W 1157.94 0.114 0.177 0.799 0.385 0.535 0.725 0.610 0.745 0.349 0.471 0.290 0.502 0.641 0.509 0.176 1.896 0.362 0.196 LingbotMap∗-S 1157.94 0.114 0.179 0.807 0.405 0.549 0.730 0.621 0.753 0.369 0.479 0.304 0.511 0.647 0.508 0.193 2.025 0.411 0.175

###### Chunk-wise

VGGT-Long 1256.54 0.131 0.601 0.679 0.479 0.593 0.735 0.673 0.773 0.465 0.548 0.410 0.573 0.679 0.512 0.164 2.199 0.633 0.090 𝜋3-Long 958.70 0.097 0.248 0.897 0.446 0.583 0.742 0.734 0.825 0.501 0.609 0.426 0.626 0.740 0.465 0.106 1.149 0.590 0.087 DA3-Streaming 1355.67 0.091 0.110 0.618 0.560 0.674 0.794 0.743 0.828 0.578 0.656 0.520 0.675 0.767 0.563 0.135 1.922 0.725 0.074

###### SLAM-based

MASt3R-SLAM 688.64 0.348 1.504 2.278 0.102 0.169 0.312 0.321 0.438 0.084 0.135 0.064 0.168 0.262 6.075 0.682 10.98 0.130 0.870 VGGT-SLAM 1256.54 0.129 0.212 0.691 0.448 0.562 0.703 0.617 0.723 0.416 0.502 0.363 0.531 0.645 0.686 0.145 2.186 0.610 0.091

###### Test-Time Training

TTT3R 793.31 0.179 0.242 0.970 0.295 0.424 0.604 0.412 0.573 0.188 0.307 0.142 0.351 0.493 2.343 0.462 5.740 0.294 0.373 Scal3R 1266.14 0.147 0.170 0.818 0.327 0.423 0.602 0.629 0.753 0.435 0.535 0.349 0.546 0.670 0.400 0.154 1.713 0.671 0.201 LoGeR 1254.62 0.113 0.065 0.542 0.435 0.576 0.741 0.690 0.793 0.405 0.540 0.340 0.564 0.693 0.591 0.123 1.254 0.504 0.096 LoGeR∗ 1254.60 0.083 0.057 0.515 0.523 0.667 0.799 0.704 0.801 0.452 0.569 0.373 0.589 0.714 0.566 0.097 1.080 0.574 0.086

- Table 12 | Detailed Results on the Dense Regime. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red. Within each sub-category, the bold value marks the in-group best.

Depth Camera Trajectory PointCloud

#Params (M)

Method

AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ RAcc3 ↑ RAcc5 ↑ TAcc3 ↑ TAcc5 ↑ AUC@5↑ AUC@15↑ AUC@30↑ ATE↓ RPE𝑡 ↓ RPE𝑟 ↓ F-Score↑ Overall↓

###### Optimization-based

DUSt3R 571.17 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM MASt3R 688.64 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM

###### End-to-End Feed-Forward

VGGT 1256.54 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM Fast3R 647.55 0.331 0.814 2.162 0.152 0.232 0.376 0.191 0.304 0.071 0.116 0.047 0.135 0.232 13.68 3.064 12.04 0.224 0.289 FastVGGT 1157.94 0.120 0.102 0.685 0.421 0.552 0.704 0.609 0.716 0.368 0.456 0.306 0.473 0.588 19.23 1.145 1.899 0.479 0.185 MUSt3R 423.43 T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O T.O MapAnything 1228.49 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OmniVGGT 1217.49 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM 𝜋3 958.70 0.109 0.268 1.008 0.403 0.553 0.738 0.528 0.627 0.300 0.379 0.254 0.410 0.524 16.39 1.132 2.237 0.332 0.757 𝜋3-X 1360.03 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM AMB3R 1563.12 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM DA3-Small 34.30 0.208 0.280 1.226 0.249 0.368 0.556 0.297 0.435 0.130 0.215 0.088 0.243 0.368 28.12 3.996 5.154 0.325 0.187 DA3-Base 135.37 0.166 0.220 1.093 0.327 0.454 0.632 0.378 0.509 0.198 0.285 0.151 0.310 0.436 26.35 3.818 6.089 0.399 0.146 DA3-Large 410.94 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM DA3-Giant 1355.67 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM DA3-Nested 1689.85 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM WorldMirror 1263.34 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM DA-Next 1303.76 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM

###### Online

Spann3r 658.69 0.315 97.78 5.152 0.136 0.217 0.373 0.143 0.260 0.050 0.099 0.023 0.125 0.246 26.48 3.257 5.064 0.159 0.322 CUT3R 793.31 0.260 0.497 1.346 0.192 0.290 0.458 0.130 0.200 0.043 0.079 0.023 0.080 0.165 25.54 0.484 1.180 0.109 0.497 MonST3R 571.17 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM Point3R 828.01 0.285 1.066 1.450 0.230 0.339 0.504 0.148 0.263 0.035 0.080 0.015 0.104 0.212 28.09 1.286 3.057 0.139 0.299 Stream3R-S 1190.60 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM Stream3R-W 1190.60 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM StreamVGGT 1256.54 0.198 0.590 1.715 0.179 0.279 0.460 0.414 0.565 0.152 0.244 0.111 0.278 0.413 26.9 1.719 1.785 0.251 0.197 Page4D 1256.81 OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM OOM InfiniteVGGT 1256.54 0.197 0.585 1.703 0.179 0.280 0.461 0.416 0.566 0.154 0.246 0.112 0.280 0.416 27.01 1.717 1.750 0.254 0.197 Wint3R 749.46 0.234 0.280 1.215 0.179 0.278 0.466 0.167 0.262 0.042 0.083 0.021 0.098 0.202 27.8 0.874 1.213 0.114 0.475 LongStream-B 1190.60 0.269 0.225 1.013 0.158 0.245 0.411 0.165 0.273 0.097 0.158 0.039 0.149 0.294 5.766 0.105 0.702 0.083 0.410 LongStream-S 1190.60 0.279 0.230 0.998 0.170 0.261 0.426 0.130 0.210 0.076 0.126 0.026 0.107 0.218 10.08 0.186 1.038 0.083 0.438 LingbotMap∗-W 1157.94 0.167 0.211 0.976 0.265 0.392 0.572 0.524 0.678 0.294 0.393 0.229 0.414 0.553 4.694 0.098 0.502 0.352 0.383 LingbotMap∗-S 1157.94 0.139 0.209 0.958 0.384 0.516 0.677 0.602 0.724 0.382 0.477 0.308 0.496 0.627 3.470 0.328 0.749 0.472 0.296

###### Chunk-wise

VGGT-Long 1256.54 0.222 1.006 0.986 0.310 0.428 0.582 0.464 0.593 0.257 0.347 0.211 0.379 0.507 8.467 0.149 0.693 0.467 0.142 𝜋3-Long 958.70 0.216 0.375 1.464 0.121 0.203 0.363 0.595 0.741 0.321 0.437 0.253 0.469 0.614 4.021 0.093 0.396 0.251 0.223 DA3-Streaming 1355.67 0.245 22.56 2.475 0.379 0.502 0.666 0.513 0.625 0.331 0.405 0.277 0.427 0.546 8.575 0.119 0.588 0.516 0.162

###### SLAM-based

MASt3R-SLAM 688.64 0.404 2.171 2.920 0.086 0.140 0.265 0.357 0.475 0.117 0.180 0.088 0.207 0.311 25.7 0.413 1.983 0.121 0.493 VGGT-SLAM 1256.54 0.211 0.366 1.051 0.266 0.373 0.530 0.381 0.510 0.206 0.284 0.159 0.309 0.441 9.069 0.152 0.626 0.384 0.160

###### Test-Time Training

TTT3R 793.31 0.222 0.353 1.222 0.220 0.331 0.507 0.221 0.351 0.102 0.172 0.064 0.193 0.321 21.07 0.569 1.234 0.173 0.283 Scal3R 1266.14 0.244 0.280 1.127 0.124 0.197 0.353 0.407 0.552 0.247 0.335 0.161 0.328 0.480 2.396 0.111 0.864 0.498 0.142 LoGeR 1254.62 0.197 0.112 0.741 0.225 0.345 0.524 0.535 0.677 0.249 0.345 0.197 0.391 0.552 5.217 0.090 0.385 0.335 0.165 LoGeR∗ 1254.60 0.156 0.086 0.684 0.304 0.435 0.611 0.590 0.725 0.317 0.411 0.256 0.446 0.598 4.598 0.077 0.347 0.421 0.145

- Table 13 | Per-Dataset Results on 7Scenes. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.2 0.084 0.075 0.641 0.083 0.591 0.107 0.281 OOM OOM OOM OOM (0.079) (0.616) (0.107) (0.281) MASt3R 688.6 0.111 0.085 0.591 0.105 0.614 0.121 0.294 OOM OOM OOM OOM (0.095) (0.603) (0.121) (0.294)

###### End-to-End Feed-Forward

VGGT 1257 0.074 0.068 0.715 0.064 0.777 0.070 0.463 OOM OOM OOM OOM (0.066) (0.746) (0.070) (0.463) Fast3R 647.5 0.104 0.218 0.428 0.087 0.603 0.201 0.216 0.124 0.341 0.321 0.075 0.143 0.457 0.261 0.146 FastVGGT 1158 0.075 0.072 0.604 0.062 0.771 0.071 0.431 0.066 0.732 0.115 0.401 0.067 0.703 0.093 0.416 MUSt3R 423.4 0.072 0.077 0.557 0.063 0.748 0.077 0.383 T.O T.O T.O T.O (0.070) (0.652) (0.077) (0.383) MapAnything 1228 0.076 0.073 0.642 0.066 0.732 0.072 0.378 OOM OOM OOM OOM (0.070) (0.687) (0.072) (0.378) OmniVGGT 1217 0.062 0.063 0.673 0.058 0.753 0.081 0.469 OOM OOM OOM OOM (0.061) (0.713) (0.081) (0.469) 𝜋3 958.7 0.065 0.063 0.663 0.059 0.769 0.063 0.393 0.074 0.161 0.427 0.052 0.065 0.531 0.245 0.222 𝜋3-X 1360 0.069 0.064 0.693 0.061 0.776 0.063 0.442 OOM OOM OOM OOM (0.062) (0.735) (0.063) (0.442) AMB3R 1563 0.063 0.060 0.680 0.058 0.782 0.063 0.490 OOM OOM OOM OOM (0.059) (0.731) (0.063) (0.490) DA3-Small 34.3 0.090 0.082 0.549 0.067 0.699 0.096 0.404 0.070 0.671 0.103 0.399 0.073 0.640 0.099 0.401 DA3-Base 135.4 0.088 0.075 0.675 0.061 0.743 0.079 0.395 0.060 0.735 0.086 0.405 0.065 0.718 0.082 0.400 DA3-Large 410.9 0.077 0.074 0.760 0.059 0.789 0.062 0.464 OOM OOM OOM OOM (0.066) (0.774) (0.062) (0.464) DA3-Giant 1356 0.066 0.066 0.780 0.058 0.790 0.059 0.439 OOM OOM OOM OOM (0.062) (0.785) (0.059) (0.439) DA3-Nested 1690 0.067 0.064 0.784 0.058 0.790 0.060 0.435 OOM OOM OOM OOM (0.061) (0.787) (0.060) (0.435) WorldMirror 1263 0.068 0.066 0.695 0.059 0.770 0.068 0.463 OOM OOM OOM OOM (0.063) (0.733) (0.068) (0.463) VGGT-Omega 1144 0.059 0.056 0.808 0.049 0.858 0.035 0.582 – – – – (0.052) (0.833) (0.035) (0.582) DA-Next (Ours) 1304 0.066 0.063 0.744 0.059 0.790 0.063 0.446 OOM OOM OOM OOM (0.063) (0.767) (0.063) (0.446)

###### Online

Spann3r224 658.7 0.076 0.096 0.429 0.086 0.477 0.195 0.147 0.102 0.452 0.224 0.138 0.095 0.452 0.209 0.143 CUT3R 793.3 0.072 0.073 0.640 0.081 0.648 0.123 0.318 0.105 0.069 0.498 0.055 0.087 0.452 0.311 0.187 MonST3R 571.2 0.104 0.097 0.270 0.092 0.345 0.192 0.151 OOM OOM OOM OOM (0.095) (0.307) (0.192) (0.151) Point3R 828 0.072 0.080 0.537 0.083 0.588 0.120 0.225 0.093 0.427 0.239 0.094 0.086 0.517 0.179 0.159 Stream3R-S 1191 0.079 0.078 0.614 0.437 0.236 0.499 0.040 OOM OOM OOM OOM (0.257) (0.425) (0.499) (0.040) Stream3R-W 1191 0.079 0.078 0.614 0.308 0.214 0.537 0.048 OOM OOM OOM OOM (0.193) (0.414) (0.537) (0.048) StreamVGGT 1257 0.069 0.081 0.598 0.073 0.740 0.086 0.307 0.085 0.659 0.130 0.278 0.080 0.666 0.108 0.292 Page4D 1257 0.069 0.065 0.632 0.062 0.743 0.079 0.373 OOM OOM OOM OOM (0.064) (0.687) (0.079) (0.373) InfiniteVGGT 1257 0.069 0.081 0.600 0.073 0.741 0.086 0.310 0.085 0.658 0.131 0.283 0.080 0.666 0.109 0.297 Wint3R 749.5 0.070 0.075 0.598 0.065 0.676 0.097 0.358 0.106 0.241 0.318 0.054 0.082 0.505 0.207 0.206 LongStream-B 1191 0.057 0.067 0.605 0.087 0.683 0.094 0.100 0.087 0.362 0.151 0.060 0.081 0.550 0.123 0.080 LongStream-S 1191 0.057 0.067 0.605 0.071 0.442 0.150 0.040 0.085 0.293 0.180 0.042 0.075 0.447 0.165 0.041 LingbotMap∗-W 1158 0.076 0.069 0.702 0.085 0.758 0.070 0.340 0.111 0.674 0.107 0.220 0.088 0.711 0.088 0.280 LingbotMap∗-S 1158 0.076 0.069 0.702 0.081 0.770 0.068 0.375 0.092 0.769 0.069 0.300 0.081 0.747 0.069 0.337

###### Chunk-wise

VGGT-Long 1257 0.074 0.068 0.715 0.065 0.777 0.063 0.420 0.078 0.645 0.101 0.332 0.070 0.712 0.082 0.376 𝜋3-Long 958.7 0.065 0.063 0.663 0.078 0.782 0.058 0.310 0.118 0.679 0.091 0.213 0.086 0.708 0.074 0.261 DA3-Streaming 1356 0.066 0.066 0.780 0.058 0.788 0.059 0.439 0.066 0.675 0.084 0.336 0.063 0.748 0.071 0.388

###### SLAM-based

MASt3R-SLAM 688.6 0.141 0.185 0.193 0.181 0.662 0.095 0.209 0.196 0.648 0.099 0.178 0.187 0.501 0.097 0.194 VGGT-SLAM 1257 0.074 0.068 0.715 0.069 0.733 0.075 0.382 0.084 0.576 0.125 0.270 0.074 0.675 0.100 0.326

###### Test-Time Training

TTT3R 793.3 0.072 0.077 0.552 0.070 0.701 0.097 0.339 0.097 0.304 0.333 0.151 0.081 0.519 0.215 0.245 Scal3R 1266 0.073 0.069 0.668 0.133 0.688 0.068 0.402 0.140 0.531 0.097 0.288 0.114 0.629 0.083 0.345 LoGeR 1255 0.066 0.062 0.686 0.081 0.758 0.079 0.282 0.106 0.651 0.116 0.238 0.083 0.698 0.097 0.260 LoGeR∗ 1255 0.070 0.060 0.735 0.061 0.765 0.063 0.455 0.079 0.701 0.074 0.370 0.067 0.734 0.069 0.413

- Table 14 | Per-Dataset Results on ADT. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.160 0.117 0.882 0.120 0.762 0.117 OOM OOM OOM (0.119) (0.822) (0.117) MASt3R 688.6 0.155 0.079 0.905 0.103 0.861 0.065 OOM OOM OOM (0.091) (0.883) (0.065)

###### End-to-End Feed-Forward

VGGT 1257 0.133 0.072 0.787 0.073 0.840 0.122 OOM OOM OOM (0.072) (0.813) (0.122) Fast3R 647.5 0.172 0.147 0.597 0.189 0.611 0.710 0.220 0.380 1.032 0.185 0.529 0.871 FastVGGT 1158 0.133 0.064 0.776 0.068 0.873 0.097 0.065 0.825 0.083 0.066 0.825 0.090 MUSt3R 423.4 0.168 0.074 0.915 0.126 0.941 0.029 T.O T.O T.O (0.100) (0.928) (0.029) MapAnything 1228 0.108 0.066 0.833 0.073 0.867 0.093 OOM OOM OOM (0.069) (0.850) (0.093) OmniVGGT 1217 0.114 0.078 0.741 0.081 0.742 0.254 OOM OOM OOM (0.080) (0.741) (0.254) 𝜋3 958.7 0.048 0.045 0.954 0.041 0.957 0.025 0.042 0.877 0.042 0.043 0.929 0.033 𝜋3-X 1360 0.080 0.046 0.954 0.041 0.965 0.021 OOM OOM OOM (0.044) (0.959) (0.021) AMB3R 1563 0.081 0.045 0.905 0.045 0.934 0.035 OOM OOM OOM (0.045) (0.920) (0.035) DA3-Small 34.3 0.135 0.105 0.571 0.097 0.598 0.326 0.092 0.640 0.255 0.098 0.603 0.290 DA3-Base 135.4 0.159 0.090 0.583 0.092 0.632 0.275 0.091 0.661 0.242 0.091 0.625 0.259 DA3-Large 410.9 0.127 0.072 0.726 0.059 0.783 0.130 OOM OOM OOM (0.066) (0.755) (0.130) DA3-Giant 1356 0.095 0.063 0.775 0.057 0.851 0.101 OOM OOM OOM (0.060) (0.813) (0.101) DA3-Nested 1690 0.089 0.064 0.754 0.059 0.850 0.098 OOM OOM OOM (0.061) (0.802) (0.098) WorldMirror 1263 0.105 0.078 0.807 0.089 0.871 0.078 OOM OOM OOM (0.084) (0.839) (0.078) VGGT-Omega 1144 0.081 0.064 0.851 0.042 0.937 0.039 – – – (0.053) (0.894) (0.039) DA-Next (Ours) 1304 0.010 0.017 0.963 0.013 0.980 0.013 OOM OOM OOM (0.013) (0.972) (0.013)

###### Online

Spann3r224 658.7 0.230 0.110 0.731 0.095 0.807 0.122 0.116 0.699 0.202 0.107 0.746 0.162 CUT3R 793.3 0.130 0.106 0.657 0.131 0.576 0.272 0.159 0.255 0.968 0.132 0.496 0.620 MonST3R 571.2 0.212 0.153 0.146 0.182 0.093 0.631 OOM OOM OOM (0.168) (0.119) (0.631) Point3R 828 0.142 0.140 0.421 0.143 0.494 0.320 0.159 0.397 0.420 0.147 0.437 0.370 Stream3R-S 1191 0.112 0.075 0.732 0.449 0.060 1.494 OOM OOM OOM (0.262) (0.396) (1.494) Stream3R-W 1191 0.112 0.078 0.720 0.655 0.054 1.512 OOM OOM OOM (0.367) (0.387) (1.512) StreamVGGT 1257 0.124 0.117 0.722 0.100 0.779 0.208 0.113 0.672 0.229 0.110 0.725 0.219 Page4D 1257 0.163 0.091 0.657 0.079 0.677 0.199 OOM OOM OOM (0.085) (0.667) (0.199) InfiniteVGGT 1257 0.124 0.117 0.719 0.100 0.779 0.209 0.114 0.673 0.229 0.111 0.724 0.219 Wint3R 749.5 0.136 0.089 0.529 0.079 0.731 0.137 0.094 0.496 0.375 0.087 0.585 0.256 LongStream-B 1191 0.056 0.068 0.776 0.148 0.529 0.266 0.138 0.327 0.281 0.118 0.544 0.274 LongStream-S 1191 0.056 0.068 0.776 0.141 0.238 0.512 0.152 0.145 0.565 0.120 0.386 0.539 LingbotMap∗-W 1158 0.199 0.070 0.915 0.071 0.909 0.046 0.068 0.728 0.069 0.069 0.851 0.057 LingbotMap∗-S 1158 0.199 0.070 0.915 0.067 0.919 0.038 0.058 0.835 0.044 0.065 0.890 0.041

###### Chunk-wise

VGGT-Long 1257 0.133 0.072 0.787 0.079 0.792 0.139 0.109 0.574 0.219 0.087 0.718 0.179 𝜋3-Long 958.7 0.048 0.045 0.954 0.115 0.953 0.022 0.135 0.835 0.045 0.098 0.914 0.033 DA3-Streaming 1356 0.095 0.063 0.775 0.059 0.837 0.102 0.071 0.633 0.162 0.064 0.748 0.132

###### SLAM-based

MASt3R-SLAM 688.6 0.425 0.319 0.118 0.354 0.481 0.544 0.371 0.747 0.076 0.348 0.449 0.310 VGGT-SLAM 1257 0.133 0.072 0.787 0.103 0.653 0.208 0.111 0.529 0.233 0.095 0.656 0.221

###### Test-Time Training

TTT3R 793.3 0.130 0.130 0.531 0.121 0.589 0.445 0.136 0.481 0.383 0.129 0.534 0.414 Scal3R 1266 0.114 0.072 0.903 0.168 0.784 0.027 0.178 0.589 0.042 0.139 0.759 0.035 LoGeR 1255 0.095 0.053 0.904 0.101 0.831 0.095 0.106 0.682 0.111 0.087 0.806 0.103 LoGeR∗ 1255 0.069 0.046 0.933 0.054 0.904 0.045 0.065 0.816 0.039 0.055 0.884 0.042

- Table 15 | Per-Dataset Results on DROID. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.218 0.301 0.177 0.305 0.124 0.080 OOM OOM OOM (0.303) (0.151) (0.080) MASt3R 688.6 0.167 0.203 0.269 0.244 0.188 0.072 OOM OOM OOM (0.224) (0.229) (0.072)

###### End-to-End Feed-Forward

VGGT 1257 0.121 0.111 0.525 0.131 0.450 0.025 OOM OOM OOM (0.121) (0.487) (0.025) Fast3R 647.5 0.199 0.320 0.200 0.282 0.150 0.074 0.303 0.134 0.076 0.301 0.161 0.075 FastVGGT 1158 0.121 0.123 0.506 0.125 0.443 0.028 0.129 0.447 0.026 0.126 0.465 0.027 MUSt3R 423.4 0.185 0.190 0.412 0.176 0.389 0.037 T.O T.O T.O (0.183) (0.401) (0.037) MapAnything 1228 0.100 0.167 0.266 0.144 0.272 0.044 OOM OOM OOM (0.155) (0.269) (0.044) OmniVGGT 1217 0.136 0.262 0.551 0.121 0.509 0.020 OOM OOM OOM (0.192) (0.530) (0.020) 𝜋3 958.7 0.106 0.151 0.534 0.105 0.522 0.022 0.105 0.508 0.022 0.120 0.521 0.022 𝜋3-X 1360 0.100 0.103 0.607 0.094 0.563 0.019 OOM OOM OOM (0.099) (0.585) (0.019) AMB3R 1563 0.098 0.091 0.561 0.080 0.481 0.024 OOM OOM OOM (0.085) (0.521) (0.024) DA3-Small 34.3 0.198 0.238 0.168 0.219 0.185 0.052 0.224 0.162 0.058 0.227 0.172 0.055 DA3-Base 135.4 0.158 0.190 0.294 0.159 0.252 0.040 0.161 0.237 0.042 0.170 0.261 0.041 DA3-Large 410.9 0.136 0.162 0.523 0.095 0.477 0.020 OOM OOM OOM (0.129) (0.500) (0.020) DA3-Giant 1356 0.114 0.125 0.642 0.092 0.593 0.016 OOM OOM OOM (0.108) (0.618) (0.016) DA3-Nested 1690 0.123 0.146 0.675 0.110 0.578 0.018 OOM OOM OOM (0.128) (0.627) (0.018) WorldMirror 1263 0.125 0.256 0.539 0.125 0.477 0.026 OOM OOM OOM (0.190) (0.508) (0.026) VGGT-Omega 1144 0.073 0.091 0.621 0.059 0.533 0.021 – – – (0.075) (0.577) (0.021) DA-Next (Ours) 1304 0.098 0.090 0.627 0.044 0.582 0.018 OOM OOM OOM (0.077) (0.605) (0.018)

###### Online

Spann3r224 658.7 0.189 0.246 0.087 0.259 0.103 0.076 0.326 0.077 0.091 0.277 0.089 0.084 CUT3R 793.3 0.190 0.253 0.305 0.294 0.199 0.057 0.399 0.066 0.086 0.315 0.190 0.071 MonST3R 571.2 0.195 0.204 0.183 0.269 0.168 0.073 OOM OOM OOM (0.237) (0.176) (0.073) Point3R 828 0.213 0.369 0.167 0.353 0.094 0.081 0.348 0.077 0.082 0.357 0.113 0.081 Stream3R-S 1191 0.093 0.112 0.467 0.139 0.339 0.050 OOM OOM OOM (0.125) (0.403) (0.050) Stream3R-W 1191 0.093 0.112 0.467 0.220 0.217 0.086 OOM OOM OOM (0.166) (0.342) (0.086) StreamVGGT 1257 0.105 0.130 0.488 0.161 0.380 0.038 0.177 0.327 0.048 0.156 0.398 0.043 Page4D 1257 0.135 0.127 0.322 0.113 0.345 0.028 OOM OOM OOM (0.120) (0.334) (0.028) InfiniteVGGT 1257 0.105 0.131 0.491 0.162 0.379 0.038 0.178 0.328 0.048 0.157 0.400 0.043 Wint3R 749.5 0.189 0.187 0.262 0.179 0.159 0.059 0.321 0.058 0.084 0.229 0.160 0.072 LongStream-B 1191 0.176 0.306 0.367 0.559 0.174 0.070 0.481 0.204 0.067 0.449 0.249 0.069 LongStream-S 1191 0.176 0.306 0.368 0.302 0.217 0.061 0.512 0.167 0.062 0.373 0.251 0.062 LingbotMap∗-W 1158 0.142 0.176 0.308 0.145 0.391 0.033 0.209 0.269 0.048 0.177 0.322 0.041 LingbotMap∗-S 1158 0.142 0.176 0.308 0.145 0.376 0.035 0.163 0.323 0.036 0.161 0.336 0.036

###### Chunk-wise

VGGT-Long 1257 0.121 0.111 0.525 0.135 0.435 0.028 0.200 0.284 0.045 0.149 0.415 0.036 𝜋3-Long 958.7 0.106 0.151 0.534 0.113 0.515 0.022 0.232 0.352 0.036 0.166 0.467 0.029 DA3-Streaming 1356 0.114 0.125 0.642 0.094 0.567 0.018 0.136 0.414 0.031 0.118 0.541 0.025

###### SLAM-based

MASt3R-SLAM 688.6 0.261 0.342 0.239 0.392 0.126 0.078 0.378 0.108 0.076 0.370 0.158 0.077 VGGT-SLAM 1257 0.121 0.111 0.525 0.141 0.383 0.034 0.232 0.179 0.059 0.161 0.362 0.047

###### Test-Time Training

TTT3R 793.3 0.190 0.263 0.268 0.253 0.244 0.043 0.304 0.194 0.062 0.274 0.235 0.053 Scal3R 1266 0.159 0.252 0.571 0.186 0.495 0.022 0.302 0.279 0.035 0.247 0.448 0.028 LoGeR 1255 0.102 0.140 0.490 0.142 0.445 0.032 0.269 0.394 0.033 0.183 0.443 0.032 LoGeR∗ 1255 0.095 0.120 0.518 0.105 0.437 0.028 0.214 0.387 0.036 0.146 0.447 0.032

- Table 16 | Per-Dataset Results on DTU. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. This dataset is not evaluated under the Dense regime, so the corresponding cells are marked as “-”. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.2 0.049 0.064 0.627 0.077 0.613 0.045 0.385 - - - - 0.070 0.620 0.045 0.385 MASt3R 688.6 0.078 0.069 0.723 0.065 0.686 0.034 0.436 - - - - 0.067 0.704 0.034 0.436

###### End-to-End Feed-Forward

VGGT 1257 0.009 0.008 0.997 0.005 0.995 0.002 0.797 - - - - 0.007 0.996 0.002 0.797 Fast3R 647.5 0.059 0.074 0.764 0.064 0.716 0.038 0.318 - - - - 0.069 0.740 0.038 0.318 FastVGGT 1158 0.009 0.009 0.968 0.007 0.960 0.005 0.809 - - - - 0.008 0.964 0.005 0.809 MUSt3R 423.4 0.059 0.058 0.797 0.042 0.823 0.023 0.468 - - - - 0.050 0.810 0.023 0.468 MapAnything 1228 0.090 0.084 0.624 0.083 0.621 0.044 0.290 - - - - 0.084 0.622 0.044 0.290 OmniVGGT 1217 0.022 0.013 0.962 0.012 0.962 0.005 0.735 - - - - 0.012 0.962 0.005 0.735 𝜋3 958.7 0.027 0.016 0.945 0.013 0.950 0.006 0.669 - - - - 0.015 0.947 0.006 0.669 𝜋3-X 1360 0.027 0.018 0.943 0.014 0.958 0.005 0.649 - - - - 0.016 0.951 0.005 0.649 AMB3R 1563 0.012 0.009 0.994 0.007 0.990 0.003 0.581 - - - - 0.008 0.992 0.003 0.581 DA3-Small 34.3 0.069 0.034 0.831 0.028 0.798 0.019 0.625 - - - - 0.031 0.814 0.019 0.625 DA3-Base 135.4 0.055 0.021 0.898 0.017 0.898 0.010 0.689 - - - - 0.019 0.898 0.010 0.689 DA3-Large 410.9 0.035 0.017 0.972 0.012 0.966 0.004 0.727 - - - - 0.015 0.969 0.004 0.727 DA3-Giant 1356 0.033 0.014 0.992 0.011 0.987 0.002 0.747 - - - - 0.013 0.990 0.002 0.747 DA3-Nested 1690 0.028 0.019 0.993 0.013 0.986 0.002 0.739 - - - - 0.016 0.989 0.002 0.739 WorldMirror 1263 0.039 0.024 0.898 0.021 0.899 0.011 0.571 - - - - 0.022 0.898 0.011 0.571 VGGT-Omega 1144 0.018 0.014 0.989 0.011 0.973 0.003 0.679 – – – – (0.012) (0.981) (0.003) (0.679) DA-Next (Ours) 1304 0.121 0.054 0.898 0.020 0.901 0.011 0.643 - - - - (0.065) (0.900) (0.011) (0.643)

###### Online

Spann3r224 658.7 0.040 0.047 0.602 0.044 0.634 0.043 0.473 - - - - 0.045 0.618 0.043 0.473 CUT3R 793.3 0.050 0.056 0.763 0.053 0.724 0.028 0.375 - - - - 0.054 0.744 0.028 0.375 MonST3R 571.2 0.104 0.101 0.470 0.123 0.014 0.214 0.007 - - - - 0.112 0.242 0.214 0.007 Point3R 828 0.038 0.061 0.596 0.063 0.499 0.048 0.239 - - - - 0.062 0.547 0.048 0.239 Stream3R-S 1191 0.010 0.011 0.965 0.010 0.937 0.008 0.726 - - - - 0.011 0.951 0.008 0.726 Stream3R-W 1191 0.010 0.011 0.965 0.011 0.896 0.013 0.720 - - - - 0.011 0.931 0.013 0.720 StreamVGGT 1257 0.012 0.012 0.971 0.011 0.963 0.005 0.795 - - - - 0.012 0.967 0.005 0.795 Page4D 1257 0.019 0.022 0.877 0.015 0.860 0.011 0.568 - - - - 0.018 0.869 0.011 0.568 InfiniteVGGT 1257 0.011 0.012 0.972 0.011 0.962 0.006 0.804 - - - - 0.012 0.967 0.006 0.804 Wint3R 749.5 0.033 0.028 0.791 0.026 0.785 0.020 0.610 - - - - 0.027 0.788 0.020 0.610 LongStream-B 1191 0.037 0.038 0.851 0.062 0.715 0.031 0.247 - - - - 0.050 0.783 0.031 0.247 LongStream-S 1191 0.037 0.038 0.851 0.032 0.707 0.030 0.267 - - - - 0.035 0.779 0.030 0.267 LingbotMap∗-W 1158 0.058 0.048 0.776 0.048 0.719 0.032 0.300 - - - - 0.048 0.748 0.032 0.300 LingbotMap∗-S 1158 0.058 0.048 0.776 0.048 0.719 0.032 0.300 - - - - 0.048 0.748 0.032 0.300

###### Chunk-wise

VGGT-Long 1257 0.009 0.008 0.997 0.005 0.995 0.002 0.797 - - - - 0.007 0.996 0.002 0.797 𝜋3-Long 958.7 0.027 0.016 0.945 0.013 0.950 0.006 0.669 - - - - 0.015 0.947 0.006 0.669 DA3-Streaming 1356 0.033 0.014 0.993 0.011 0.987 0.002 0.746 - - - - 0.013 0.990 0.002 0.746

###### SLAM-based

MASt3R-SLAM 688.6 0.126 0.120 0.251 0.140 0.320 0.093 0.189 - - - - 0.130 0.285 0.093 0.189 VGGT-SLAM 1257 0.009 0.008 0.997 0.005 0.995 0.002 0.797 - - - - 0.007 0.996 0.002 0.797

###### Test-Time Training

TTT3R 793.3 0.050 0.059 0.696 0.055 0.702 0.042 0.345 - - - - 0.057 0.699 0.042 0.345 Scal3R 1266 0.010 0.008 0.991 0.007 0.981 0.002 0.773 - - - - 0.008 0.986 0.002 0.773 LoGeR 1255 0.031 0.031 0.890 0.020 0.897 0.011 0.639 - - - - 0.026 0.893 0.011 0.639 LoGeR∗ 1255 0.038 0.021 0.888 0.018 0.889 0.011 0.658 - - - - 0.020 0.889 0.011 0.658

- Table 17 | Per-Dataset Results on ETH3D. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. This dataset is not evaluated under the Dense regime, so the corresponding cells are marked as “-”. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.046 0.099 0.482 0.084 0.528 1.297 - - - 0.092 0.505 1.297 MASt3R 688.6 0.051 0.086 0.690 0.057 0.793 0.597 - - - 0.071 0.742 0.597

###### End-to-End Feed-Forward

VGGT 1257 0.033 0.050 0.686 0.039 0.715 0.460 - - - 0.044 0.701 0.460 Fast3R 647.5 0.083 0.163 0.267 0.208 0.285 2.747 - - - 0.186 0.276 2.747 FastVGGT 1158 0.034 0.055 0.568 0.053 0.601 0.791 - - - 0.054 0.584 0.791 MUSt3R 423.4 0.037 0.071 0.587 0.056 0.762 0.554 - - - 0.063 0.675 0.554 MapAnything 1228 0.038 0.047 0.670 0.048 0.736 0.314 - - - 0.048 0.703 0.314 OmniVGGT 1217 0.026 0.042 0.556 0.039 0.638 0.952 - - - 0.040 0.597 0.952 𝜋3 958.7 0.027 0.034 0.706 0.030 0.784 0.199 - - - 0.032 0.745 0.199 𝜋3-X 1360 0.028 0.032 0.692 0.030 0.779 0.207 - - - 0.031 0.735 0.207 AMB3R 1563 0.022 0.050 0.684 0.037 0.779 0.280 - - - 0.043 0.731 0.280 DA3-Small 34.3 0.063 0.108 0.484 0.102 0.511 1.120 - - - 0.105 0.498 1.120 DA3-Base 135.4 0.045 0.080 0.655 0.068 0.677 0.685 - - - 0.074 0.666 0.685 DA3-Large 410.9 0.042 0.056 0.689 0.041 0.795 0.552 - - - 0.049 0.742 0.552 DA3-Giant 1356 0.031 0.035 0.802 0.028 0.880 0.196 - - - 0.031 0.841 0.196 DA3-Nested 1690 0.041 0.027 0.805 0.025 0.878 0.192 - - - 0.026 0.842 0.192 WorldMirror 1263 0.031 0.040 0.709 0.039 0.785 1.411 - - - 0.040 0.747 1.411 VGGT-Omega 1144 0.019 0.023 0.863 0.021 0.899 0.126 – – – (0.022) (0.881) (0.126) DA-Next (Ours) 1304 0.030 0.037 0.765 0.025 0.845 0.222 - - - (0.031) (0.805) (0.222)

###### Online

Spann3r224 658.7 0.156 0.256 0.256 0.264 0.256 2.833 - - - 0.260 0.256 2.833 CUT3R 793.3 0.040 0.100 0.428 0.104 0.449 1.354 - - - 0.102 0.439 1.354 MonST3R 571.2 0.051 0.131 0.207 0.196 0.078 2.126 - - - 0.163 0.142 2.126 Point3R 828 0.046 0.173 0.223 0.177 0.181 2.042 - - - 0.175 0.202 2.042 Stream3R-S 1191 0.023 0.043 0.455 0.064 0.518 1.241 - - - 0.053 0.486 1.241 Stream3R-W 1191 0.023 0.043 0.455 0.069 0.493 1.290 - - - 0.056 0.474 1.290 StreamVGGT 1257 0.031 0.092 0.473 0.138 0.443 1.126 - - - 0.115 0.458 1.126 Page4D 1257 0.031 0.046 0.527 0.050 0.565 0.696 - - - 0.048 0.546 0.696 InfiniteVGGT 1257 0.031 0.092 0.470 0.138 0.442 1.129 - - - 0.115 0.456 1.129 Wint3R 749.5 0.043 0.093 0.416 0.093 0.321 1.556 - - - 0.093 0.369 1.556 LongStream-B 1191 0.031 0.071 0.472 0.081 0.389 1.802 - - - 0.076 0.431 1.802 LongStream-S 1191 0.031 0.071 0.472 0.081 0.385 1.798 - - - 0.076 0.429 1.798 LingbotMap∗-W 1158 0.038 0.052 0.711 0.054 0.673 1.293 - - - 0.053 0.692 1.293 LingbotMap∗-S 1158 0.038 0.052 0.711 0.054 0.673 1.293 - - - 0.053 0.692 1.293

###### Chunk-wise

VGGT-Long 1257 0.033 0.050 0.686 0.039 0.715 0.460 - - - 0.044 0.701 0.460 𝜋3-Long 958.7 0.027 0.034 0.706 0.030 0.784 0.199 - - - 0.032 0.745 0.199 DA3-Streaming 1356 0.031 0.035 0.803 0.028 0.880 0.196 - - - 0.031 0.841 0.196

###### SLAM-based

MASt3R-SLAM 688.6 0.104 0.165 0.145 0.165 0.098 2.659 - - - 0.165 0.121 2.659 VGGT-SLAM 1257 0.033 0.050 0.686 0.039 0.715 0.460 - - - 0.044 0.701 0.460

###### Test-Time Training

TTT3R 793.3 0.040 0.114 0.450 0.100 0.395 2.009 - - - 0.107 0.422 2.009 Scal3R 1266 0.027 0.035 0.717 0.033 0.772 0.234 - - - 0.034 0.744 0.234 LoGeR 1255 0.032 0.035 0.717 0.026 0.793 0.217 - - - 0.030 0.755 0.217 LoGeR∗ 1255 0.030 0.031 0.730 0.025 0.828 0.176 - - - 0.028 0.779 0.176

- Table 18 | Per-Dataset Results on Hiroom. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. This dataset is not evaluated under the Dense regime, so the corresponding cells are marked as “-”. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.2 0.024 0.028 0.786 0.031 0.853 0.079 0.289 - - - - 0.030 0.820 0.079 0.289 MASt3R 688.6 0.056 0.075 0.879 0.051 0.904 0.054 0.306 - - - - 0.063 0.891 0.054 0.306

###### End-to-End Feed-Forward

VGGT 1257 0.024 0.020 0.792 0.017 0.848 0.091 0.558 - - - - 0.018 0.820 0.091 0.558 Fast3R 647.5 0.038 0.079 0.646 0.050 0.729 0.222 0.256 - - - - 0.065 0.688 0.222 0.256 FastVGGT 1158 0.024 0.024 0.621 0.024 0.752 0.224 0.345 - - - - 0.024 0.687 0.224 0.345 MUSt3R 423.4 0.033 0.021 0.933 0.020 0.942 0.037 0.651 - - - - 0.021 0.938 0.037 0.651 MapAnything 1228 0.021 0.028 0.890 0.027 0.898 0.063 0.584 - - - - 0.028 0.894 0.063 0.584 OmniVGGT 1217 0.023 0.025 0.811 0.021 0.778 0.137 0.387 - - - - 0.023 0.795 0.137 0.387 𝜋3 958.7 0.020 0.022 0.932 0.013 0.948 0.033 0.679 - - - - 0.017 0.940 0.033 0.679 𝜋3-X 1360 0.021 0.018 0.919 0.013 0.955 0.028 0.651 - - - - 0.015 0.937 0.028 0.651 AMB3R 1563 0.027 0.020 0.871 0.021 0.897 0.057 0.390 - - - - 0.021 0.884 0.057 0.390 DA3-Small 34.3 0.039 0.042 0.650 0.044 0.801 0.103 0.363 - - - - 0.043 0.725 0.103 0.363 DA3-Base 135.4 0.036 0.032 0.848 0.029 0.869 0.086 0.454 - - - - 0.031 0.859 0.086 0.454 DA3-Large 410.9 0.020 0.017 0.947 0.014 0.972 0.027 0.629 - - - - 0.016 0.959 0.027 0.629 DA3-Giant 1356 0.017 0.009 0.966 0.005 0.996 0.008 0.960 - - - - 0.007 0.981 0.008 0.960 DA3-Nested 1690 0.020 0.010 0.971 0.006 0.994 0.009 0.947 - - - - 0.008 0.983 0.009 0.947 WorldMirror 1263 0.026 0.030 0.912 0.025 0.923 0.047 0.474 - - - - 0.027 0.918 0.047 0.474 VGGT-Omega 1144 0.008 0.015 0.985 0.009 0.980 0.017 0.868 – – – – (0.012) (0.983) (0.017) (0.868) DA-Next (Ours) 1304 0.018 0.010 0.982 0.007 0.990 0.011 0.948 - - - - (0.012) (0.986) (0.011) (0.948)

###### Online

Spann3r224 658.7 0.041 0.079 0.594 0.071 0.659 0.328 0.214 - - - - 0.075 0.627 0.328 0.214 CUT3R 793.3 0.035 0.082 0.531 0.076 0.634 0.359 0.143 - - - - 0.079 0.583 0.359 0.143 MonST3R 571.2 0.033 0.094 0.327 0.070 0.167 0.702 0.050 - - - - 0.082 0.247 0.702 0.050 Point3R 828 0.038 0.088 0.417 0.074 0.422 0.456 0.138 - - - - 0.081 0.419 0.456 0.138 Stream3R-S 1191 0.025 0.030 0.781 0.030 0.777 0.158 0.269 - - - - 0.030 0.779 0.158 0.269 Stream3R-W 1191 0.025 0.030 0.781 0.030 0.771 0.159 0.256 - - - - 0.030 0.776 0.159 0.256 StreamVGGT 1257 0.024 0.076 0.736 0.057 0.688 0.274 0.079 - - - - 0.066 0.712 0.274 0.079 Page4D 1257 0.028 0.029 0.736 0.025 0.858 0.076 0.314 - - - - 0.027 0.797 0.076 0.314 InfiniteVGGT 1257 0.023 0.075 0.731 0.057 0.687 0.274 0.078 - - - - 0.066 0.709 0.274 0.078 Wint3R 749.5 0.028 0.035 0.728 0.034 0.648 0.268 0.252 - - - - 0.035 0.688 0.268 0.252 LongStream-B 1191 0.019 0.033 0.819 0.030 0.759 0.224 0.052 - - - - 0.031 0.789 0.224 0.052 LongStream-S 1191 0.019 0.033 0.819 0.030 0.759 0.224 0.053 - - - - 0.031 0.789 0.224 0.053 LingbotMap∗-W 1158 0.027 0.034 0.870 0.036 0.769 0.127 0.278 - - - - 0.035 0.820 0.127 0.278 LingbotMap∗-S 1158 0.027 0.034 0.870 0.036 0.769 0.127 0.278 - - - - 0.035 0.820 0.127 0.278

###### Chunk-wise

VGGT-Long 1257 0.024 0.020 0.792 0.017 0.848 0.091 0.559 - - - - 0.018 0.820 0.091 0.559 𝜋3-Long 958.7 0.020 0.022 0.932 0.013 0.948 0.033 0.679 - - - - 0.017 0.940 0.033 0.679 DA3-Streaming 1356 0.017 0.009 0.966 0.005 0.996 0.008 0.960 - - - - 0.007 0.981 0.008 0.960

###### SLAM-based

MASt3R-SLAM 688.6 0.159 0.202 0.194 0.171 0.238 0.845 0.021 - - - - 0.187 0.216 0.845 0.021 VGGT-SLAM 1257 0.024 0.020 0.792 0.017 0.848 0.091 0.558 - - - - 0.018 0.820 0.091 0.558

###### Test-Time Training

TTT3R 793.3 0.035 0.086 0.486 0.080 0.608 0.380 0.136 - - - - 0.083 0.547 0.380 0.136 Scal3R 1266 0.022 0.019 0.859 0.014 0.958 0.034 0.651 - - - - 0.016 0.909 0.034 0.651 LoGeR 1255 0.025 0.018 0.836 0.017 0.915 0.046 0.342 - - - - 0.017 0.875 0.046 0.342 LoGeR∗ 1255 0.023 0.020 0.856 0.019 0.908 0.049 0.402 - - - - 0.019 0.882 0.049 0.402

- Table 19 | Per-Dataset Results on KITTI-Odometry. Only metrics available for the Dense regime are reported. The best, second-best, and third-best results in each column are highlighted in deep blue ,

medium blue , and light blue , respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red ; methods without dense-regime results use “–”. Within each sub-category, the bold value

marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Camera Trajectory

#Params (M)

Method

AUC@30↑ ATE↓ RPE𝑡 ↓ RPE𝑟 ↓ Optimization-based

DUSt3R 571.2 OOM OOM OOM OOM MASt3R 688.6 OOM OOM OOM OOM

###### End-to-End Feed-Forward

VGGT 1257 OOM OOM OOM OOM Fast3R 647.5 0.000 134.5 29.12 27.34 FastVGGT 1158 0.307 181.9 9.827 3.338 MUSt3R 423.4 T.O T.O T.O T.O MapAnything 1228 OOM OOM OOM OOM OmniVGGT 1217 OOM OOM OOM OOM 𝜋3 958.7 0.412 153 10.07 1.369 𝜋3-X 1360 OOM OOM OOM OOM AMB3R 1563 OOM OOM OOM OOM DA3-Small 34.3 0.089 220.3 29 17.12 DA3-Base 135.4 0.067 211.7 30.08 40.69 DA3-Large 410.9 OOM OOM OOM OOM DA3-Giant 1356 OOM OOM OOM OOM DA3-Nested 1690 OOM OOM OOM OOM WorldMirror 1263 OOM OOM OOM OOM VGGT-Omega 1144 – – – – DA-Next (Ours) 1304 OOM OOM OOM OOM

###### Online

Spann3r224 658.7 0.036 214.9 23.33 18.99 CUT3R 793.3 0.065 206.6 2.763 1.830 MonST3R 571.2 OOM OOM OOM OOM Point3R 828 0.051 206.5 7.671 4.421 Stream3R-S 1191 OOM OOM OOM OOM Stream3R-W 1191 OOM OOM OOM OOM StreamVGGT 1257 0.168 206.2 14 4.174 Page4D 1257 OOM OOM OOM OOM InfiniteVGGT 1257 0.169 206.7 13.87 4.164 Wint3R 749.5 0.074 211.3 5.458 1.624 LongStream-B 1191 0.317 47.33 0.348 0.331 LongStream-S 1191 0.211 88.6 0.885 0.620 LingbotMap∗-W 1158 0.615 41.16 0.368 0.286 LingbotMap∗-S 1158 0.729 29.31 2.440 0.663

###### Chunk-wise

VGGT-Long 1257 0.465 76.14 0.798 0.468 𝜋3-Long 958.7 0.702 31.91 0.362 0.181 DA3-Streaming 1356 0.221 79.26 0.633 0.689

###### SLAM-based

MASt3R-SLAM 688.6 0.158 196.5 1.737 0.561 VGGT-SLAM 1257 0.436 80.06 0.607 0.226

###### Test-Time Training

TTT3R 793.3 0.086 182.1 3.824 2.146 Scal3R 1266 0.655 18.17 0.457 0.649 LoGeR 1255 0.457 45.03 0.382 0.238 LoGeR∗ 1255 0.456 39.62 0.286 0.220

- Table 20 | Per-Dataset Results on Lingbot-Depth. Only metrics available for the Single Frame regime are reported. The best, second-best, and third-best results in each column are highlighted in deep blue ,

medium blue , and light blue , respectively. Within each sub-category, the bold value marks the ingroup best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Depth AbsRel↓ SqRel↓ RMSE↓ 𝛿1.03 ↑ 𝛿1.05 ↑ 𝛿1.10 ↑ Optimization-based

#Params (M)

Method

DUSt3R 571.2 0.949 18.34 0.836 0.308 0.439 0.636 MASt3R 688.6 1.227 35.58 0.911 0.349 0.485 0.682

###### End-to-End Feed-Forward

VGGT 1257 0.428 2.572 0.692 0.387 0.519 0.709 Fast3R 647.5 0.804 11.42 0.826 0.285 0.390 0.543 FastVGGT 1158 0.424 2.505 0.692 0.388 0.520 0.710 MUSt3R 423.4 1.110 30.18 0.831 0.368 0.484 0.681 MapAnything 1228 1.326 42.33 0.784 0.423 0.571 0.749 OmniVGGT 1217 0.419 2.593 0.614 0.428 0.557 0.751 𝜋3 958.7 1.485 59.37 0.793 0.480 0.622 0.771 𝜋3-X 1360 1.118 31.36 0.674 0.461 0.592 0.780 AMB3R 1563 1.453 54.6 0.762 0.439 0.572 0.750 DA3-Small 34.3 0.973 22.16 0.796 0.240 0.359 0.578 DA3-Base 135.4 0.875 18.38 0.661 0.338 0.480 0.694 DA3-Large 410.9 0.881 19.73 0.628 0.382 0.534 0.726 DA3-Giant 1356 1.034 26.41 0.693 0.440 0.586 0.744 DA3-Nested 1690 1.015 25.81 0.707 0.448 0.594 0.758 WorldMirror 1263 0.962 27.24 0.681 0.434 0.567 0.746 VGGT-Omega 1144 1.647 73.16 0.889 0.450 0.582 0.768 DA-Next (Ours) 1304 0.405 3.183 0.611 0.399 0.529 0.715

###### Online

Spann3r224 658.7 0.843 13.87 1.055 0.203 0.321 0.509 CUT3R 793.3 0.543 4.487 0.611 0.347 0.482 0.677 MonST3R 571.2 0.660 8.235 0.792 0.332 0.455 0.636 Point3R 828 0.988 23.95 0.766 0.300 0.422 0.632 Stream3R-S 1191 1.228 40.11 0.741 0.474 0.615 0.773 Stream3R-W 1191 1.228 40.11 0.741 0.474 0.615 0.773 StreamVGGT 1257 0.553 5.858 0.611 0.399 0.528 0.727 Page4D 1257 0.566 6.851 0.734 0.355 0.474 0.667 InfiniteVGGT 1257 0.547 5.667 0.610 0.399 0.528 0.727 Wint3R 749.5 1.869 90.54 1.231 0.385 0.502 0.682 LongStream-B 1191 1.572 60.93 0.831 0.372 0.510 0.722 LongStream-S 1191 1.572 60.93 0.831 0.372 0.510 0.722 LingbotMap∗-W 1158 0.883 15.64 0.768 0.385 0.516 0.693 LingbotMap∗-S 1158 0.883 15.64 0.768 0.385 0.516 0.693

###### Chunk-wise

VGGT-Long 1257 0.428 2.572 0.692 0.387 0.519 0.709 𝜋3-Long 958.7 1.485 59.37 0.793 0.480 0.622 0.771 DA3-Streaming 1356 1.034 26.41 0.693 0.440 0.586 0.744

###### SLAM-based

MASt3R-SLAM 688.6 0.619 4.389 0.784 0.150 0.241 0.438 VGGT-SLAM 1257 0.428 2.572 0.692 0.387 0.519 0.709

###### Test-Time Training

TTT3R 793.3 0.543 4.488 0.611 0.347 0.482 0.677 Scal3R 1266 0.546 5.438 0.682 0.424 0.555 0.733 LoGeR 1255 0.662 9.432 0.617 0.465 0.607 0.764 LoGeR∗ 1255 0.506 4.440 0.607 0.487 0.605 0.769

- Table 21 | Per-Dataset Results on NRGBD. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.2 0.028 0.038 0.843 0.036 0.783 0.265 0.327 OOM OOM OOM OOM (0.037) (0.813) (0.265) (0.327) MASt3R 688.6 0.036 0.043 0.891 0.048 0.778 0.205 0.290 OOM OOM OOM OOM (0.046) (0.834) (0.205) (0.290)

###### End-to-End Feed-Forward

VGGT 1257 0.024 0.016 0.926 0.012 0.956 0.039 0.696 OOM OOM OOM OOM (0.014) (0.941) (0.039) (0.696) Fast3R 647.5 0.042 0.073 0.779 0.041 0.858 0.102 0.345 0.039 0.592 0.157 0.351 0.051 0.743 0.130 0.348 FastVGGT 1158 0.026 0.023 0.888 0.014 0.947 0.044 0.569 0.019 0.831 0.162 0.470 0.019 0.889 0.103 0.519 MUSt3R 423.4 0.024 0.026 0.912 0.021 0.932 0.053 0.569 T.O T.O T.O T.O (0.024) (0.922) (0.053) (0.569) MapAnything 1228 0.026 0.035 0.872 0.031 0.886 0.093 0.437 OOM OOM OOM OOM (0.033) (0.879) (0.093) (0.437) OmniVGGT 1217 0.016 0.020 0.913 0.013 0.948 0.039 0.728 OOM OOM OOM OOM (0.016) (0.931) (0.039) (0.728) 𝜋3 958.7 0.021 0.017 0.930 0.012 0.958 0.030 0.735 0.074 0.158 1.158 0.039 0.034 0.682 0.594 0.387 𝜋3-X 1360 0.023 0.016 0.940 0.012 0.962 0.029 0.770 OOM OOM OOM OOM (0.014) (0.951) (0.029) (0.770) AMB3R 1563 0.022 0.024 0.933 0.015 0.953 0.038 0.688 OOM OOM OOM OOM (0.020) (0.943) (0.038) (0.688) DA3-Small 34.3 0.043 0.034 0.822 0.031 0.888 0.080 0.428 0.032 0.861 0.094 0.412 0.033 0.857 0.087 0.420 DA3-Base 135.4 0.031 0.024 0.891 0.020 0.937 0.048 0.561 0.019 0.929 0.055 0.467 0.021 0.919 0.051 0.514 DA3-Large 410.9 0.024 0.019 0.918 0.014 0.962 0.030 0.624 OOM OOM OOM OOM (0.017) (0.940) (0.030) (0.624) DA3-Giant 1356 0.026 0.009 0.944 0.009 0.971 0.025 0.811 OOM OOM OOM OOM (0.009) (0.957) (0.025) (0.811) DA3-Nested 1690 0.023 0.013 0.942 0.010 0.971 0.025 0.811 OOM OOM OOM OOM (0.011) (0.957) (0.025) (0.811) WorldMirror 1263 0.020 0.019 0.916 0.014 0.948 0.041 0.603 OOM OOM OOM OOM (0.017) (0.932) (0.041) (0.603) VGGT-Omega 1144 0.008 0.010 0.935 0.007 0.961 0.031 0.732 – – – – (0.008) (0.948) (0.031) (0.732) DA-Next (Ours) 1304 0.028 0.014 0.943 0.011 0.966 0.028 0.837 OOM OOM OOM OOM (0.018) (0.955) (0.028) (0.837)

###### Online

Spann3r224 658.7 0.040 0.083 0.600 0.053 0.711 0.241 0.149 0.067 0.687 0.258 0.185 0.068 0.666 0.250 0.167 CUT3R 793.3 0.044 0.044 0.847 0.056 0.738 0.208 0.250 0.089 0.085 1.308 0.074 0.063 0.557 0.758 0.162 MonST3R 571.2 0.026 0.064 0.260 0.050 0.377 0.574 0.151 OOM OOM OOM OOM (0.057) (0.318) (0.574) (0.151) Point3R 828 0.024 0.055 0.582 0.048 0.724 0.191 0.238 0.061 0.618 0.430 0.149 0.054 0.641 0.311 0.193 Stream3R-S 1191 0.024 0.031 0.893 0.105 0.384 1.039 0.100 OOM OOM OOM OOM (0.068) (0.639) (1.039) (0.100) Stream3R-W 1191 0.024 0.031 0.895 0.251 0.354 1.170 0.088 OOM OOM OOM OOM (0.141) (0.625) (1.170) (0.088) StreamVGGT 1257 0.026 0.046 0.871 0.051 0.890 0.136 0.196 0.051 0.739 0.241 0.211 0.049 0.834 0.188 0.204 Page4D 1257 0.030 0.029 0.877 0.024 0.898 0.063 0.364 OOM OOM OOM OOM (0.026) (0.887) (0.063) (0.364) InfiniteVGGT 1257 0.026 0.046 0.871 0.051 0.889 0.138 0.194 0.051 0.740 0.239 0.211 0.049 0.833 0.189 0.202 Wint3R 749.5 0.023 0.032 0.779 0.027 0.796 0.254 0.472 0.083 0.264 0.750 0.091 0.047 0.613 0.502 0.281 LongStream-B 1191 0.025 0.031 0.876 0.082 0.656 0.376 0.060 0.091 0.277 0.692 0.053 0.068 0.603 0.534 0.057 LongStream-S 1191 0.025 0.031 0.876 0.058 0.392 0.685 0.060 0.090 0.169 0.849 0.050 0.060 0.479 0.767 0.055 LingbotMap∗-W 1158 0.028 0.030 0.885 0.027 0.911 0.085 0.364 0.049 0.790 0.176 0.288 0.035 0.862 0.130 0.326 LingbotMap∗-S 1158 0.028 0.030 0.885 0.022 0.929 0.055 0.524 0.021 0.934 0.055 0.494 0.024 0.916 0.055 0.509

###### Chunk-wise

VGGT-Long 1257 0.024 0.016 0.926 0.013 0.955 0.038 0.658 0.024 0.851 0.174 0.481 0.018 0.911 0.106 0.570 𝜋3-Long 958.7 0.021 0.017 0.930 0.027 0.952 0.032 0.568 0.143 0.850 0.147 0.160 0.062 0.911 0.090 0.364 DA3-Streaming 1356 0.026 0.009 0.944 0.009 0.967 0.027 0.790 0.023 0.871 0.170 0.551 0.014 0.927 0.098 0.671

###### SLAM-based

MASt3R-SLAM 688.6 0.096 0.103 0.258 0.111 0.674 0.482 0.134 0.116 0.791 0.230 0.130 0.110 0.574 0.356 0.132 VGGT-SLAM 1257 0.024 0.016 0.926 0.015 0.941 0.047 0.647 0.027 0.837 0.163 0.410 0.019 0.901 0.105 0.528

###### Test-Time Training

TTT3R 793.3 0.044 0.048 0.800 0.047 0.846 0.114 0.333 0.070 0.302 0.686 0.095 0.055 0.649 0.400 0.214 Scal3R 1266 0.024 0.016 0.928 0.147 0.798 0.037 0.709 0.172 0.622 0.174 0.472 0.112 0.783 0.105 0.590 LoGeR 1255 0.020 0.021 0.912 0.022 0.922 0.067 0.502 0.055 0.731 0.215 0.268 0.033 0.855 0.141 0.385 LoGeR∗ 1255 0.015 0.013 0.935 0.017 0.938 0.056 0.593 0.052 0.828 0.187 0.289 0.028 0.900 0.121 0.441

- Table 22 | Per-Dataset Results on OmniWorld. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.211 0.305 0.601 0.371 0.550 2.484 OOM OOM OOM (0.338) (0.575) (2.484) MASt3R 688.6 0.216 0.247 0.598 0.248 0.595 2.923 OOM OOM OOM (0.247) (0.596) (2.923)

###### End-to-End Feed-Forward

VGGT 1257 0.147 0.170 0.774 0.176 0.752 3.763 OOM OOM OOM (0.173) (0.763) (3.763) Fast3R 647.5 0.246 0.352 0.449 0.348 0.337 7.692 0.438 0.165 12.5 0.380 0.317 10.1 FastVGGT 1158 0.148 0.176 0.672 0.166 0.739 3.956 0.161 0.735 4.926 0.168 0.715 4.441 MUSt3R 423.4 0.215 0.205 0.762 0.291 0.724 1.362 T.O T.O T.O (0.248) (0.743) (1.362) MapAnything 1228 0.144 0.122 0.824 0.125 0.802 2.256 OOM OOM OOM (0.123) (0.813) (2.256) OmniVGGT 1217 0.143 0.125 0.713 0.140 0.691 2.765 OOM OOM OOM (0.132) (0.702) (2.765) 𝜋3 958.7 0.123 0.042 0.956 0.037 0.962 0.270 0.037 0.959 0.286 0.039 0.959 0.278 𝜋3-X 1360 0.125 0.041 0.972 0.035 0.965 0.217 OOM OOM OOM (0.038) (0.969) (0.217) AMB3R 1563 0.147 0.124 0.876 0.124 0.880 0.837 OOM OOM OOM (0.124) (0.878) (0.837) DA3-Small 34.3 0.223 0.211 0.531 0.227 0.476 5.359 0.261 0.462 7.659 0.233 0.490 6.509 DA3-Base 135.4 0.195 0.155 0.613 0.170 0.640 4.013 0.183 0.583 4.623 0.169 0.612 4.318 DA3-Large 410.9 0.110 0.074 0.956 0.069 0.959 0.436 OOM OOM OOM (0.071) (0.958) (0.436) DA3-Giant 1356 0.114 0.050 0.978 0.046 0.982 0.235 OOM OOM OOM (0.048) (0.980) (0.235) DA3-Nested 1690 0.106 0.050 0.981 0.044 0.985 0.186 OOM OOM OOM (0.047) (0.983) (0.186) WorldMirror 1263 0.180 0.173 0.733 0.167 0.787 1.836 OOM OOM OOM (0.170) (0.760) (1.836) VGGT-Omega 1144 0.061 0.050 0.969 0.047 0.960 0.299 – – – (0.048) (0.965) (0.299) DA-Next (Ours) 1304 0.135 0.057 0.973 0.049 0.974 0.278 OOM OOM OOM (0.080) (0.973) (0.278)

###### Online

Spann3r224 658.7 0.269 0.385 0.434 0.394 0.367 5.645 0.417 0.259 8.475 0.399 0.353 7.060 CUT3R 793.3 0.172 0.216 0.726 0.312 0.608 2.828 0.389 0.373 9.160 0.306 0.569 5.994 MonST3R 571.2 0.200 0.223 0.394 0.242 0.495 3.075 OOM OOM OOM (0.233) (0.445) (3.075) Point3R 828 0.200 0.246 0.260 0.262 0.261 7.981 0.339 0.092 9.878 0.282 0.205 8.929 Stream3R-S 1191 0.160 0.146 0.637 0.184 0.643 6.196 OOM OOM OOM (0.165) (0.640) (6.196) Stream3R-W 1191 0.160 0.152 0.592 0.215 0.450 11.25 OOM OOM OOM (0.183) (0.521) (11.25) StreamVGGT 1257 0.158 0.298 0.622 0.382 0.578 6.522 0.376 0.383 9.389 0.352 0.528 7.956 Page4D 1257 0.148 0.135 0.720 0.147 0.740 4.709 OOM OOM OOM (0.141) (0.730) (4.709) InfiniteVGGT 1257 0.160 0.299 0.623 0.381 0.586 6.464 0.379 0.387 9.373 0.353 0.532 7.919 Wint3R 749.5 0.137 0.066 0.747 0.072 0.685 1.833 0.144 0.394 6.080 0.094 0.609 3.956 LongStream-B 1191 0.153 0.185 0.420 0.247 0.780 1.356 0.224 0.650 2.156 0.219 0.617 1.756 LongStream-S 1191 0.153 0.165 0.417 0.199 0.617 2.763 0.232 0.502 2.440 0.199 0.512 2.602 LingbotMap∗-W 1158 0.131 0.080 0.948 0.094 0.936 0.402 0.117 0.906 0.881 0.097 0.930 0.642 LingbotMap∗-S 1158 0.131 0.080 0.948 0.090 0.923 0.357 0.095 0.918 0.588 0.088 0.930 0.473

###### Chunk-wise

VGGT-Long 1257 0.147 0.170 0.774 0.170 0.814 2.701 0.349 0.613 5.409 0.230 0.734 4.055 𝜋3-Long 958.7 0.123 0.042 0.956 0.082 0.957 0.288 0.337 0.909 0.739 0.154 0.940 0.513 DA3-Streaming 1356 0.114 0.050 0.978 0.046 0.981 0.243 0.051 0.956 0.289 0.049 0.972 0.266

###### SLAM-based

MASt3R-SLAM 688.6 0.281 0.331 0.342 0.280 0.610 3.683 0.344 0.616 3.616 0.318 0.523 3.649 VGGT-SLAM 1257 0.147 0.170 0.774 0.186 0.765 4.607 0.371 0.433 6.222 0.243 0.657 5.415

###### Test-Time Training

TTT3R 793.3 0.172 0.236 0.506 0.249 0.645 3.182 0.284 0.631 4.303 0.257 0.594 3.742 Scal3R 1266 0.142 0.116 0.848 0.137 0.830 0.812 0.405 0.682 1.359 0.219 0.787 1.085 LoGeR 1255 0.131 0.039 0.936 0.048 0.960 0.311 0.087 0.939 0.605 0.058 0.945 0.458 LoGeR∗ 1255 0.130 0.041 0.942 0.033 0.971 0.238 0.074 0.939 0.602 0.050 0.951 0.420

- Table 23 | Per-Dataset Results on RLBench. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.437 0.830 0.181 0.746 0.148 0.154 OOM OOM OOM (0.788) (0.165) (0.154) MASt3R 688.6 0.308 0.419 0.216 0.790 0.133 0.150 OOM OOM OOM (0.605) (0.174) (0.150)

###### End-to-End Feed-Forward

VGGT 1257 0.190 0.296 0.302 0.608 0.177 0.133 OOM OOM OOM (0.452) (0.239) (0.133) Fast3R 647.5 0.367 0.432 0.147 0.630 0.082 0.150 0.722 0.055 0.155 0.595 0.095 0.153 FastVGGT 1158 0.192 0.304 0.172 0.351 0.139 0.152 0.368 0.158 0.126 0.341 0.156 0.139 MUSt3R 423.4 0.394 0.382 0.267 0.480 0.308 0.056 T.O T.O T.O (0.431) (0.287) (0.056) MapAnything 1228 0.281 0.328 0.284 0.443 0.176 0.101 OOM OOM OOM (0.386) (0.230) (0.101) OmniVGGT 1217 0.300 0.283 0.521 0.306 0.323 0.104 OOM OOM OOM (0.295) (0.422) (0.104) 𝜋3 958.7 0.201 0.205 0.559 0.204 0.433 0.061 0.210 0.439 0.062 0.207 0.477 0.062 𝜋3-X 1360 0.205 0.260 0.440 0.212 0.385 0.072 OOM OOM OOM (0.236) (0.413) (0.072) AMB3R 1563 0.204 0.229 0.525 0.256 0.389 0.086 OOM OOM OOM (0.243) (0.457) (0.086) DA3-Small 34.3 0.350 0.435 0.283 0.482 0.183 0.088 0.536 0.140 0.088 0.484 0.202 0.088 DA3-Base 135.4 0.410 0.406 0.285 0.400 0.217 0.077 0.438 0.157 0.078 0.415 0.220 0.078 DA3-Large 410.9 0.364 0.336 0.431 0.353 0.315 0.078 OOM OOM OOM (0.344) (0.373) (0.078) DA3-Giant 1356 0.355 0.270 0.565 0.250 0.471 0.058 OOM OOM OOM (0.260) (0.518) (0.058) DA3-Nested 1690 0.385 0.311 0.560 0.241 0.521 0.051 OOM OOM OOM (0.276) (0.541) (0.051) WorldMirror 1263 0.322 0.329 0.318 0.340 0.245 0.116 OOM OOM OOM (0.335) (0.282) (0.116) VGGT-Omega 1144 0.193 0.194 0.596 0.182 0.589 0.043 – – – (0.188) (0.592) (0.043) DA-Next (Ours) 1304 0.088 0.046 0.893 0.037 0.807 0.019 OOM OOM OOM (0.057) (0.850) (0.019)

###### Online

Spann3r224 658.7 0.444 0.434 0.131 0.618 0.129 0.108 0.813 0.045 0.167 0.622 0.102 0.138 CUT3R 793.3 0.329 0.543 0.187 0.440 0.171 0.102 0.557 0.051 0.148 0.513 0.137 0.125 MonST3R 571.2 0.393 0.553 0.080 0.476 0.147 0.125 OOM OOM OOM (0.515) (0.113) (0.125) Point3R 828 0.348 0.503 0.164 0.678 0.044 0.183 0.795 0.036 0.181 0.659 0.081 0.182 Stream3R-S 1191 0.262 0.326 0.134 0.352 0.099 0.153 OOM OOM OOM (0.339) (0.116) (0.153) Stream3R-W 1191 0.262 0.326 0.134 0.484 0.064 0.198 OOM OOM OOM (0.405) (0.099) (0.198) StreamVGGT 1257 0.239 0.316 0.151 0.358 0.132 0.162 0.379 0.119 0.163 0.351 0.134 0.162 Page4D 1257 0.219 0.338 0.330 0.327 0.211 0.113 OOM OOM OOM (0.333) (0.271) (0.113) InfiniteVGGT 1257 0.239 0.315 0.155 0.358 0.132 0.162 0.378 0.118 0.163 0.350 0.135 0.162 Wint3R 749.5 0.364 0.383 0.201 0.398 0.142 0.107 0.584 0.076 0.154 0.455 0.140 0.131 LongStream-B 1191 0.385 0.332 0.121 0.438 0.097 0.112 0.668 0.080 0.102 0.479 0.099 0.107 LongStream-S 1191 0.385 0.331 0.121 0.363 0.081 0.110 0.750 0.063 0.113 0.482 0.088 0.112 LingbotMap∗-W 1158 0.300 0.339 0.387 0.282 0.257 0.091 0.364 0.269 0.095 0.328 0.304 0.093 LingbotMap∗-S 1158 0.300 0.339 0.387 0.282 0.257 0.091 0.324 0.183 0.114 0.315 0.276 0.103

###### Chunk-wise

VGGT-Long 1257 0.190 0.296 0.302 0.609 0.178 0.136 0.599 0.177 0.113 0.501 0.219 0.124 𝜋3-Long 958.7 0.201 0.205 0.559 0.206 0.433 0.061 0.350 0.375 0.069 0.254 0.456 0.065 DA3-Streaming 1356 0.355 0.270 0.566 0.256 0.472 0.062 0.549 0.311 0.102 0.359 0.450 0.082

###### SLAM-based

MASt3R-SLAM 688.6 0.410 0.594 0.106 0.773 0.148 0.139 0.859 0.201 0.107 0.742 0.152 0.123 VGGT-SLAM 1257 0.190 0.296 0.302 0.432 0.182 0.135 0.350 0.120 0.108 0.359 0.201 0.121

###### Test-Time Training

TTT3R 793.3 0.329 0.559 0.156 0.493 0.158 0.106 0.506 0.112 0.113 0.519 0.142 0.110 Scal3R 1266 0.238 0.233 0.524 0.282 0.347 0.089 0.386 0.248 0.073 0.300 0.373 0.081 LoGeR 1255 0.235 0.202 0.428 0.321 0.452 0.052 0.433 0.435 0.066 0.319 0.438 0.059 LoGeR∗ 1255 0.210 0.171 0.523 0.202 0.525 0.040 0.349 0.465 0.077 0.241 0.505 0.059

- Table 24 | Per-Dataset Results on Robolab. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.371 0.495 0.274 0.512 0.132 0.117 OOM OOM OOM (0.504) (0.203) (0.117) MASt3R 688.6 0.386 0.515 0.429 0.463 0.242 0.103 OOM OOM OOM (0.489) (0.336) (0.103)

###### End-to-End Feed-Forward

VGGT 1257 0.174 0.201 0.566 0.177 0.504 0.035 OOM OOM OOM (0.189) (0.535) (0.035) Fast3R 647.5 0.340 0.412 0.174 0.403 0.176 0.085 0.480 0.099 0.103 0.432 0.150 0.094 FastVGGT 1158 0.171 0.215 0.453 0.175 0.483 0.038 0.180 0.468 0.041 0.190 0.468 0.040 MUSt3R 423.4 0.491 0.457 0.376 0.262 0.505 0.036 T.O T.O T.O (0.359) (0.440) (0.036) MapAnything 1228 0.194 0.381 0.221 0.321 0.179 0.069 OOM OOM OOM (0.351) (0.200) (0.069) OmniVGGT 1217 0.157 0.185 0.601 0.162 0.562 0.030 OOM OOM OOM (0.173) (0.582) (0.030) 𝜋3 958.7 0.168 0.171 0.663 0.150 0.742 0.018 0.183 0.458 0.049 0.168 0.621 0.033 𝜋3-X 1360 0.157 0.155 0.635 0.137 0.682 0.024 OOM OOM OOM (0.146) (0.658) (0.024) AMB3R 1563 0.213 0.239 0.639 0.188 0.582 0.032 OOM OOM OOM (0.213) (0.611) (0.032) DA3-Small 34.3 0.338 0.490 0.198 0.389 0.134 0.082 0.402 0.100 0.093 0.427 0.144 0.087 DA3-Base 135.4 0.278 0.326 0.352 0.273 0.225 0.067 0.276 0.177 0.073 0.292 0.251 0.070 DA3-Large 410.9 0.243 0.237 0.447 0.184 0.469 0.036 OOM OOM OOM (0.210) (0.458) (0.036) DA3-Giant 1356 0.132 0.174 0.719 0.170 0.671 0.027 OOM OOM OOM (0.172) (0.695) (0.027) DA3-Nested 1690 0.148 0.194 0.716 0.137 0.721 0.018 OOM OOM OOM (0.166) (0.718) (0.018) WorldMirror 1263 0.271 0.311 0.409 0.275 0.405 0.050 OOM OOM OOM (0.293) (0.407) (0.050) VGGT-Omega 1144 0.207 0.147 0.742 0.149 0.811 0.012 – – – (0.148) (0.777) (0.012) DA-Next (Ours) 1304 0.025 0.030 0.802 0.018 0.916 0.006 OOM OOM OOM (0.024) (0.859) (0.006)

###### Online

Spann3r224 658.7 0.380 0.485 0.041 0.399 0.154 0.079 0.434 0.076 0.108 0.439 0.090 0.094 CUT3R 793.3 0.274 0.451 0.287 0.404 0.122 0.090 0.449 0.029 0.123 0.434 0.146 0.107 MonST3R 571.2 0.381 0.501 0.152 0.416 0.124 0.114 OOM OOM OOM (0.458) (0.138) (0.114) Point3R 828 0.424 0.469 0.139 0.416 0.100 0.114 0.434 0.053 0.129 0.440 0.097 0.121 Stream3R-S 1191 0.196 0.270 0.424 0.244 0.260 0.092 OOM OOM OOM (0.257) (0.342) (0.092) Stream3R-W 1191 0.196 0.271 0.421 0.348 0.117 0.137 OOM OOM OOM (0.310) (0.269) (0.137) StreamVGGT 1257 0.170 0.266 0.392 0.240 0.335 0.060 0.254 0.175 0.092 0.253 0.301 0.076 Page4D 1257 0.189 0.259 0.494 0.222 0.389 0.051 OOM OOM OOM (0.241) (0.441) (0.051) InfiniteVGGT 1257 0.169 0.265 0.390 0.239 0.339 0.060 0.252 0.175 0.090 0.252 0.301 0.075 Wint3R 749.5 0.319 0.333 0.292 0.268 0.169 0.077 0.364 0.020 0.119 0.322 0.161 0.098 LongStream-B 1191 0.216 0.270 0.412 0.360 0.165 0.087 0.358 0.062 0.100 0.329 0.213 0.093 LongStream-S 1191 0.216 0.270 0.411 0.326 0.129 0.095 0.358 0.038 0.103 0.318 0.193 0.099 LingbotMap∗-W 1158 0.202 0.327 0.394 0.201 0.360 0.044 0.323 0.200 0.077 0.284 0.318 0.060 LingbotMap∗-S 1158 0.202 0.327 0.394 0.208 0.412 0.043 0.230 0.359 0.047 0.255 0.388 0.045

###### Chunk-wise

VGGT-Long 1257 0.174 0.201 0.566 0.192 0.506 0.049 0.345 0.278 0.083 0.246 0.450 0.066 𝜋3-Long 958.7 0.168 0.171 0.663 0.185 0.639 0.042 0.301 0.378 0.075 0.219 0.560 0.058 DA3-Streaming 1356 0.132 0.174 0.720 0.183 0.578 0.037 0.223 0.325 0.074 0.193 0.541 0.056

###### SLAM-based

MASt3R-SLAM 688.6 0.473 0.547 0.118 0.471 0.147 0.110 0.459 0.181 0.094 0.492 0.149 0.102 VGGT-SLAM 1257 0.174 0.201 0.566 0.209 0.436 0.062 0.320 0.239 0.081 0.244 0.414 0.072

###### Test-Time Training

TTT3R 793.3 0.274 0.464 0.278 0.367 0.212 0.082 0.412 0.075 0.109 0.415 0.189 0.096 Scal3R 1266 0.189 0.216 0.711 0.308 0.413 0.052 0.329 0.188 0.079 0.284 0.437 0.066 LoGeR 1255 0.190 0.220 0.586 0.301 0.448 0.066 0.452 0.223 0.100 0.324 0.419 0.083 LoGeR∗ 1255 0.142 0.130 0.617 0.206 0.490 0.057 0.325 0.240 0.096 0.220 0.449 0.077

- Table 25 | Per-Dataset Results on RoboTwin. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.342 0.728 0.061 0.928 0.067 0.120 OOM OOM OOM (0.828) (0.064) (0.120) MASt3R 688.6 0.388 0.756 0.190 0.695 0.132 0.107 OOM OOM OOM (0.725) (0.161) (0.107)

###### End-to-End Feed-Forward

VGGT 1257 0.218 0.369 0.253 0.323 0.105 0.093 OOM OOM OOM (0.346) (0.179) (0.093) Fast3R 647.5 0.351 0.698 0.079 0.580 0.030 0.109 0.624 0.045 0.110 0.634 0.051 0.110 FastVGGT 1158 0.218 0.416 0.185 0.317 0.100 0.085 0.300 0.088 0.090 0.345 0.124 0.088 MUSt3R 423.4 0.366 0.507 0.126 0.473 0.137 0.093 T.O T.O T.O (0.490) (0.131) (0.093) MapAnything 1228 0.232 0.415 0.142 0.410 0.124 0.096 OOM OOM OOM (0.412) (0.133) (0.096) OmniVGGT 1217 0.212 0.333 0.090 0.495 0.070 0.106 OOM OOM OOM (0.414) (0.080) (0.106) 𝜋3 958.7 0.181 0.309 0.482 0.317 0.250 0.064 0.313 0.263 0.066 0.313 0.332 0.065 𝜋3-X 1360 0.170 0.271 0.345 0.310 0.090 0.084 OOM OOM OOM (0.291) (0.217) (0.084) AMB3R 1563 0.184 0.252 0.454 0.273 0.223 0.068 OOM OOM OOM (0.263) (0.339) (0.068) DA3-Small 34.3 0.310 0.575 0.060 0.485 0.106 0.087 0.494 0.108 0.091 0.518 0.091 0.089 DA3-Base 135.4 0.285 0.565 0.095 0.476 0.133 0.084 0.484 0.144 0.091 0.509 0.124 0.088 DA3-Large 410.9 0.255 0.426 0.285 0.348 0.210 0.070 OOM OOM OOM (0.387) (0.248) (0.070) DA3-Giant 1356 0.287 0.320 0.441 0.315 0.253 0.071 OOM OOM OOM (0.318) (0.347) (0.071) DA3-Nested 1690 0.259 0.317 0.385 0.280 0.263 0.069 OOM OOM OOM (0.298) (0.324) (0.069) WorldMirror 1263 0.205 0.363 0.179 0.344 0.121 0.095 OOM OOM OOM (0.353) (0.150) (0.095) VGGT-Omega 1144 0.182 0.217 0.529 0.189 0.372 0.058 – – – (0.203) (0.450) (0.058) DA-Next (Ours) 1304 0.072 0.068 0.765 0.045 0.541 0.036 OOM OOM OOM (0.062) (0.653) (0.036)

###### Online

Spann3r224 658.7 0.307 0.612 0.085 0.421 0.067 0.099 0.460 0.072 0.103 0.498 0.075 0.101 CUT3R 793.3 0.320 0.617 0.043 0.501 0.064 0.101 0.539 0.032 0.104 0.552 0.046 0.102 MonST3R 571.2 0.361 0.627 0.027 0.427 0.061 0.127 OOM OOM OOM (0.527) (0.044) (0.127) Point3R 828 0.295 0.583 0.017 0.591 0.045 0.109 0.613 0.053 0.114 0.596 0.038 0.112 Stream3R-S 1191 0.216 0.354 0.289 0.449 0.106 0.091 OOM OOM OOM (0.401) (0.198) (0.091) Stream3R-W 1191 0.216 0.354 0.289 0.484 0.079 0.138 OOM OOM OOM (0.419) (0.184) (0.138) StreamVGGT 1257 0.203 0.372 0.155 0.391 0.128 0.085 0.369 0.116 0.081 0.377 0.133 0.083 Page4D 1257 0.224 0.326 0.122 0.347 0.082 0.092 OOM OOM OOM (0.337) (0.102) (0.092) InfiniteVGGT 1257 0.203 0.369 0.152 0.392 0.131 0.085 0.369 0.121 0.080 0.377 0.135 0.082 Wint3R 749.5 0.245 0.569 0.106 0.438 0.111 0.085 0.517 0.078 0.095 0.508 0.099 0.090 LongStream-B 1191 0.243 0.462 0.199 0.512 0.098 0.098 0.444 0.070 0.092 0.473 0.122 0.095 LongStream-S 1191 0.243 0.462 0.199 0.420 0.085 0.097 0.448 0.106 0.083 0.443 0.130 0.090 LingbotMap∗-W 1158 0.303 0.509 0.262 0.336 0.201 0.082 0.354 0.191 0.088 0.400 0.218 0.085 LingbotMap∗-S 1158 0.303 0.509 0.262 0.361 0.202 0.081 0.429 0.199 0.084 0.433 0.221 0.082

###### Chunk-wise

VGGT-Long 1257 0.218 0.369 0.253 0.334 0.088 0.093 0.415 0.100 0.096 0.373 0.147 0.095 𝜋3-Long 958.7 0.181 0.309 0.482 0.309 0.238 0.068 0.439 0.243 0.076 0.353 0.321 0.072 DA3-Streaming 1356 0.287 0.320 0.442 0.310 0.257 0.074 0.461 0.189 0.098 0.364 0.296 0.086

###### SLAM-based

MASt3R-SLAM 688.6 0.504 0.937 0.109 0.806 0.094 0.100 0.854 0.151 0.101 0.866 0.118 0.101 VGGT-SLAM 1257 0.218 0.369 0.253 0.423 0.110 0.085 0.426 0.110 0.099 0.406 0.158 0.092

###### Test-Time Training

TTT3R 793.3 0.320 0.606 0.051 0.480 0.070 0.103 0.502 0.062 0.097 0.529 0.061 0.100 Scal3R 1266 0.193 0.350 0.280 0.309 0.147 0.083 0.391 0.132 0.085 0.350 0.186 0.084 LoGeR 1255 0.223 0.350 0.195 0.354 0.160 0.070 0.466 0.181 0.083 0.390 0.178 0.076 LoGeR∗ 1255 0.191 0.259 0.233 0.271 0.173 0.070 0.430 0.203 0.077 0.320 0.203 0.074

- Table 26 | Per-Dataset Results on Xperience. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.054 0.092 0.471 0.240 0.079 4.312 OOM OOM OOM (0.166) (0.275) (4.312) MASt3R 688.6 0.087 0.068 0.694 0.261 0.202 4.373 OOM OOM OOM (0.164) (0.448) (4.373)

###### End-to-End Feed-Forward

VGGT 1257 0.126 0.051 0.070 0.078 0.047 7.239 OOM OOM OOM (0.065) (0.059) (7.239) Fast3R 647.5 0.085 0.359 0.005 0.468 0.001 7.571 – – – 0.413 0.003 7.571 FastVGGT 1158 0.128 0.048 0.073 0.076 0.036 7.024 0.113 0.062 7.943 0.079 0.057 7.483 MUSt3R 423.4 0.058 0.059 0.496 0.111 0.282 5.645 T.O T.O T.O (0.085) (0.389) (5.645) MapAnything 1228 0.062 0.040 0.617 0.098 0.138 5.422 OOM OOM OOM (0.069) (0.377) (5.422) OmniVGGT 1217 0.056 0.033 0.042 0.072 0.042 6.971 OOM OOM OOM (0.052) (0.042) (6.971) 𝜋3 958.7 0.133 0.031 0.127 0.050 0.130 7.219 0.265 0.009 7.460 0.115 0.089 7.340 𝜋3-X 1360 0.089 0.029 0.421 0.046 0.022 6.461 OOM OOM OOM (0.038) (0.221) (6.461) AMB3R 1563 0.104 0.041 0.022 0.069 0.045 7.102 OOM OOM OOM (0.055) (0.034) (7.102) DA3-Small 34.3 0.112 0.061 0.011 0.094 0.025 6.493 0.104 0.041 6.741 0.086 0.026 6.617 DA3-Base 135.4 0.103 0.053 0.071 0.078 0.053 5.507 0.088 0.081 5.624 0.073 0.068 5.565 DA3-Large 410.9 0.099 0.064 0.103 0.073 0.116 5.989 OOM OOM OOM (0.068) (0.109) (5.989) DA3-Giant 1356 0.162 0.049 0.291 0.057 0.193 4.895 OOM OOM OOM (0.053) (0.242) (4.895) DA3-Nested 1690 0.180 0.044 0.201 0.051 0.185 6.236 OOM OOM OOM (0.047) (0.193) (6.236) WorldMirror 1263 0.076 0.048 0.197 0.091 0.040 7.172 OOM OOM OOM (0.069) (0.118) (7.172) VGGT-Omega 1144 0.070 0.038 0.528 0.069 0.275 5.317 – – – (0.054) (0.401) (5.317) DA-Next (Ours) 1304 0.064 0.037 0.703 0.046 0.385 4.508 OOM OOM OOM (0.041) (0.544) (4.508)

###### Online

Spann3r224 658.7 0.156 0.292 0.066 0.221 0.012 5.472 0.171 0.216 4.852 0.228 0.098 5.162 CUT3R 793.3 0.148 0.074 0.124 0.114 0.027 4.663 0.316 0.029 7.244 0.168 0.060 5.954 MonST3R 571.2 0.191 0.116 0.021 0.283 0.007 5.104 OOM OOM OOM (0.200) (0.014) (5.104) Point3R 828 0.114 0.074 0.031 0.156 0.017 6.804 0.234 0.029 7.762 0.154 0.025 7.283 Stream3R-S 1191 0.083 0.055 0.046 0.419 0.005 7.469 OOM OOM OOM (0.237) (0.025) (7.469) Stream3R-W 1191 0.083 0.056 0.047 0.407 0.006 7.484 OOM OOM OOM (0.231) (0.027) (7.484) StreamVGGT 1257 0.108 0.131 0.054 0.159 0.033 7.306 0.185 0.053 8.103 0.158 0.047 7.705 Page4D 1257 0.134 0.049 0.075 0.073 0.032 7.165 OOM OOM OOM (0.061) (0.054) (7.165) InfiniteVGGT 1257 0.109 0.131 0.051 0.159 0.033 7.317 0.185 0.054 8.087 0.159 0.046 7.702 Wint3R 749.5 0.053 0.054 0.039 0.074 0.065 6.512 0.290 0.059 7.698 0.139 0.054 7.105 LongStream-B 1191 0.142 0.079 0.029 0.118 0.004 6.245 0.259 0.027 2.884 0.152 0.020 4.565 LongStream-S 1191 0.142 0.075 0.028 0.137 0.004 6.278 0.223 0.020 6.397 0.145 0.017 6.337 LingbotMap∗-W 1158 0.060 0.050 0.197 0.086 0.299 2.164 0.157 0.193 1.458 0.098 0.230 1.811 LingbotMap∗-S 1158 0.060 0.050 0.197 0.097 0.270 2.445 0.100 0.451 3.616 0.082 0.306 3.031

###### Chunk-wise

VGGT-Long 1257 0.126 0.051 0.070 0.244 0.067 7.389 1.390 0.075 5.495 0.562 0.071 6.442 𝜋3-Long 958.7 0.133 0.031 0.127 0.091 0.134 6.415 0.465 0.809 0.215 0.196 0.357 3.315 DA3-Streaming 1356 0.162 0.049 0.285 0.175 0.173 5.291 0.474 0.319 2.817 0.233 0.259 4.054

###### SLAM-based

MASt3R-SLAM 688.6 0.206 0.144 0.000 0.201 0.001 6.986 0.279 0.423 4.841 0.208 0.141 5.913 VGGT-SLAM 1257 0.126 0.051 0.070 0.302 0.044 7.735 0.973 0.051 5.342 0.442 0.055 6.538

###### Test-Time Training

TTT3R 793.3 0.148 0.064 0.107 0.102 0.061 7.159 0.143 0.119 5.582 0.103 0.096 6.371 Scal3R 1266 0.256 0.045 0.056 0.192 0.010 7.411 0.462 0.195 3.739 0.233 0.087 5.575 LoGeR 1255 0.122 0.040 0.036 0.076 0.043 4.821 0.158 0.400 1.088 0.092 0.160 2.954 LoGeR∗ 1255 0.113 0.034 0.071 0.066 0.093 7.053 0.090 0.583 0.608 0.064 0.249 3.830

- Table 27 | Per-Dataset Results on Scannet++. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑ AbsRel↓ AUC@30↑ ATE↓ F-Score↑

###### Optimization-based

DUSt3R 571.2 0.038 0.064 0.762 0.057 0.713 0.392 0.374 OOM OOM OOM OOM (0.060) (0.738) (0.392) (0.374) MASt3R 688.6 0.061 0.069 0.858 0.073 0.757 0.184 0.431 OOM OOM OOM OOM (0.071) (0.807) (0.184) (0.431)

###### End-to-End Feed-Forward

VGGT 1257 0.038 0.049 0.901 0.043 0.942 0.067 0.657 OOM OOM OOM OOM (0.046) (0.921) (0.067) (0.657) Fast3R 647.5 0.046 0.102 0.567 0.095 0.619 0.555 0.325 0.112 0.599 0.639 0.308 0.103 0.595 0.597 0.316 FastVGGT 1158 0.038 0.052 0.838 0.043 0.922 0.079 0.524 0.040 0.915 0.104 0.536 0.045 0.892 0.092 0.530 MUSt3R 423.4 0.039 0.044 0.881 0.042 0.921 0.077 0.506 T.O T.O T.O T.O (0.043) (0.901) (0.077) (0.506) MapAnything 1228 0.041 0.039 0.854 0.039 0.897 0.100 0.498 OOM OOM OOM OOM (0.039) (0.875) (0.100) (0.498) OmniVGGT 1217 0.047 0.033 0.728 0.039 0.875 0.162 0.526 OOM OOM OOM OOM (0.036) (0.801) (0.162) (0.526) 𝜋3 958.7 0.035 0.031 0.901 0.031 0.952 0.044 0.709 0.030 0.953 0.058 0.723 0.031 0.935 0.051 0.716 𝜋3-X 1360 0.036 0.031 0.903 0.031 0.951 0.039 0.728 OOM OOM OOM OOM (0.031) (0.927) (0.039) (0.728) AMB3R 1563 0.035 0.036 0.871 0.035 0.929 0.084 0.557 OOM OOM OOM OOM (0.036) (0.900) (0.084) (0.557) DA3-Small 34.3 0.073 0.075 0.677 0.071 0.614 0.616 0.262 0.073 0.533 0.787 0.216 0.073 0.608 0.701 0.239 DA3-Base 135.4 0.058 0.056 0.795 0.048 0.799 0.215 0.388 0.048 0.759 0.288 0.346 0.050 0.785 0.251 0.367 DA3-Large 410.9 0.036 0.039 0.899 0.035 0.941 0.062 0.609 OOM OOM OOM OOM (0.037) (0.920) (0.062) (0.609) DA3-Giant 1356 0.035 0.032 0.959 0.031 0.983 0.020 0.761 OOM OOM OOM OOM (0.031) (0.971) (0.020) (0.761) DA3-Nested 1690 0.035 0.031 0.958 0.031 0.983 0.020 0.758 OOM OOM OOM OOM (0.031) (0.970) (0.020) (0.758) WorldMirror 1263 0.037 0.037 0.889 0.033 0.932 0.070 0.686 OOM OOM OOM OOM (0.035) (0.911) (0.070) (0.686) VGGT-Omega 1144 0.035 0.042 0.946 0.041 0.969 0.036 0.709 – – – – (0.042) (0.958) (0.036) (0.709) DA-Next (Ours) 1304 0.039 0.029 0.949 0.026 0.976 0.025 0.805 OOM OOM OOM OOM (0.032) (0.962) (0.025) (0.805)

###### Online

Spann3r224 658.7 0.072 0.175 0.337 0.123 0.431 0.731 0.161 0.130 0.361 0.925 0.154 0.143 0.376 0.828 0.158 CUT3R 793.3 0.042 0.069 0.717 0.070 0.673 0.506 0.264 0.083 0.275 1.132 0.169 0.074 0.555 0.819 0.217 MonST3R 571.2 0.049 0.110 0.232 0.157 0.022 1.189 0.084 OOM OOM OOM OOM (0.133) (0.127) (1.189) (0.084) Point3R 828 0.040 0.061 0.445 0.062 0.459 0.528 0.189 0.067 0.412 0.597 0.159 0.063 0.438 0.562 0.174 Stream3R-S 1191 0.032 0.045 0.836 0.226 0.514 1.088 0.291 OOM OOM OOM OOM (0.136) (0.675) (1.088) (0.291) Stream3R-W 1191 0.032 0.046 0.828 0.217 0.437 1.299 0.236 OOM OOM OOM OOM (0.131) (0.633) (1.299) (0.236) StreamVGGT 1257 0.038 0.077 0.837 0.088 0.740 0.500 0.302 0.087 0.670 0.622 0.263 0.084 0.749 0.561 0.283 Page4D 1257 0.044 0.054 0.812 0.050 0.867 0.130 0.385 OOM OOM OOM OOM (0.052) (0.840) (0.130) (0.385) InfiniteVGGT 1257 0.035 0.078 0.811 0.087 0.759 0.477 0.313 0.085 0.676 0.602 0.267 0.083 0.749 0.540 0.290 Wint3R 749.5 0.049 0.057 0.565 0.058 0.442 0.920 0.213 0.067 0.344 0.982 0.169 0.061 0.450 0.951 0.191 LongStream-B 1191 0.047 0.062 0.574 0.110 0.309 1.044 0.123 0.125 0.280 1.018 0.119 0.099 0.388 1.031 0.121 LongStream-S 1191 0.047 0.062 0.574 0.089 0.158 1.313 0.104 0.118 0.055 1.351 0.133 0.090 0.262 1.332 0.119 LingbotMap∗-W 1158 0.042 0.042 0.906 0.046 0.869 0.165 0.494 0.044 0.875 0.124 0.481 0.044 0.884 0.145 0.488 LingbotMap∗-S 1158 0.042 0.042 0.906 0.040 0.900 0.082 0.556 0.038 0.912 0.088 0.565 0.040 0.906 0.085 0.561

###### Chunk-wise

VGGT-Long 1257 0.038 0.049 0.901 0.046 0.911 0.126 0.597 0.045 0.893 0.111 0.543 0.047 0.902 0.119 0.570 𝜋3-Long 958.7 0.035 0.031 0.901 0.043 0.946 0.050 0.643 0.102 0.929 0.085 0.342 0.059 0.925 0.067 0.493 DA3-Streaming 1356 0.035 0.032 0.958 0.032 0.956 0.084 0.707 0.054 0.817 0.381 0.606 0.039 0.910 0.233 0.656

###### SLAM-based

MASt3R-SLAM 688.6 0.165 0.205 0.096 0.199 0.093 1.638 0.043 0.188 0.164 1.500 0.074 0.197 0.118 1.569 0.059 VGGT-SLAM 1257 0.038 0.049 0.901 0.048 0.884 0.134 0.538 0.053 0.759 0.275 0.437 0.050 0.848 0.204 0.487

###### Test-Time Training

TTT3R 793.3 0.042 0.071 0.634 0.067 0.637 0.662 0.261 0.071 0.679 0.484 0.244 0.069 0.650 0.573 0.253 Scal3R 1266 0.065 0.043 0.891 0.097 0.843 0.056 0.707 0.144 0.728 0.092 0.651 0.095 0.821 0.074 0.679 LoGeR 1255 0.038 0.033 0.908 0.040 0.908 0.077 0.574 0.058 0.826 0.182 0.447 0.043 0.881 0.129 0.510 LoGeR∗ 1255 0.035 0.030 0.900 0.034 0.923 0.073 0.632 0.040 0.865 0.143 0.548 0.035 0.896 0.108 0.590

- Table 28 | Per-Dataset Results on Tanks and Temples. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.045 0.097 0.368 0.132 0.333 11.93 OOM OOM OOM (0.114) (0.351) (11.93) MASt3R 688.6 0.043 0.074 0.548 0.113 0.578 7.868 OOM OOM OOM (0.094) (0.563) (7.868)

###### End-to-End Feed-Forward

VGGT 1257 0.022 0.039 0.712 0.038 0.780 4.238 OOM OOM OOM (0.038) (0.746) (4.238) Fast3R 647.5 0.045 0.182 0.303 0.181 0.342 11.37 0.172 0.354 14.09 0.178 0.333 12.73 FastVGGT 1158 0.022 0.043 0.507 0.039 0.748 4.854 0.027 0.832 3.614 0.036 0.695 4.234 MUSt3R 423.4 0.054 0.070 0.670 0.064 0.729 5.223 T.O T.O T.O (0.067) (0.699) (5.223) MapAnything 1228 0.027 0.054 0.604 0.045 0.696 4.929 OOM OOM OOM (0.050) (0.650) (4.929) OmniVGGT 1217 0.023 0.052 0.419 0.051 0.498 11.06 OOM OOM OOM (0.051) (0.458) (11.06) 𝜋3 958.7 0.026 0.053 0.626 0.043 0.732 4.345 0.030 0.813 7.929 0.042 0.724 6.137 𝜋3-X 1360 0.023 0.048 0.479 0.038 0.730 3.709 OOM OOM OOM (0.043) (0.605) (3.709) AMB3R 1563 0.021 0.040 0.561 0.039 0.779 7.555 OOM OOM OOM (0.039) (0.670) (7.555) DA3-Small 34.3 0.093 0.075 0.373 0.069 0.457 9.981 0.072 0.499 10.56 0.072 0.443 10.27 DA3-Base 135.4 0.066 0.073 0.385 0.060 0.613 8.527 0.060 0.624 11.09 0.064 0.541 9.807 DA3-Large 410.9 0.034 0.076 0.468 0.063 0.713 6.316 OOM OOM OOM (0.069) (0.590) (6.316) DA3-Giant 1356 0.023 0.041 0.624 0.034 0.751 6.912 OOM OOM OOM (0.037) (0.687) (6.912) DA3-Nested 1690 0.022 0.043 0.625 0.042 0.688 7.066 OOM OOM OOM (0.043) (0.657) (7.066) WorldMirror 1263 0.031 0.061 0.389 0.055 0.682 7.159 OOM OOM OOM (0.058) (0.535) (7.159) VGGT-Omega 1144 0.028 0.041 0.593 0.037 0.629 10.21 – – – (0.039) (0.611) (10.21) DA-Next (Ours) 1304 0.030 0.040 0.444 0.035 0.813 7.456 OOM OOM OOM (0.035) (0.628) (7.456)

###### Online

Spann3r224 658.7 0.142 0.677 0.260 0.517 0.316 11.5 0.410 0.261 13.84 0.535 0.279 12.67 CUT3R 793.3 0.040 0.105 0.335 0.150 0.535 9.351 0.153 0.385 10.4 0.136 0.418 9.877 MonST3R 571.2 0.058 0.108 0.088 0.179 0.029 10.42 OOM OOM OOM (0.143) (0.059) (10.42) Point3R 828 0.038 0.284 0.244 0.218 0.226 13.01 0.593 0.229 14 0.365 0.233 13.51 Stream3R-S 1191 0.022 0.050 0.366 0.081 0.488 10.99 OOM OOM OOM (0.065) (0.427) (10.99) Stream3R-W 1191 0.022 0.065 0.373 0.090 0.411 13.03 OOM OOM OOM (0.078) (0.392) (13.03) StreamVGGT 1257 0.023 0.100 0.378 0.085 0.510 11.51 0.085 0.590 8.610 0.090 0.493 10.06 Page4D 1257 0.029 0.036 0.695 0.038 0.730 4.257 OOM OOM OOM (0.037) (0.712) (4.257) InfiniteVGGT 1257 0.026 0.099 0.412 0.087 0.507 11.88 0.089 0.615 9.897 0.092 0.511 10.89 Wint3R 749.5 0.037 0.094 0.344 0.104 0.448 10.57 0.092 0.361 14.37 0.097 0.384 12.47 LongStream-B 1191 0.034 0.097 0.374 0.171 0.327 11.43 0.191 0.284 12.92 0.153 0.329 12.18 LongStream-S 1191 0.034 0.068 0.368 0.141 0.320 11.48 0.166 0.108 12 0.125 0.265 11.74 LingbotMap∗-W 1158 0.031 0.060 0.532 0.058 0.647 7.800 0.060 0.636 7.732 0.059 0.605 7.766 LingbotMap∗-S 1158 0.031 0.060 0.532 0.057 0.667 7.531 0.054 0.740 7.282 0.057 0.646 7.406

###### Chunk-wise

VGGT-Long 1257 0.022 0.039 0.712 0.038 0.779 4.242 0.044 0.586 6.455 0.040 0.692 5.349 𝜋3-Long 958.7 0.026 0.053 0.626 0.047 0.732 4.331 0.134 0.698 6.858 0.078 0.685 5.594 DA3-Streaming 1356 0.023 0.041 0.624 0.034 0.750 6.913 2.078 0.537 9.225 0.718 0.637 8.069

###### SLAM-based

MASt3R-SLAM 688.6 0.074 0.173 0.052 0.178 0.049 12.21 0.196 0.172 15.25 0.182 0.091 13.73 VGGT-SLAM 1257 0.022 0.039 0.712 0.051 0.698 4.195 0.139 0.491 8.844 0.076 0.633 6.520

###### Test-Time Training

TTT3R 793.3 0.040 0.109 0.373 0.119 0.413 11.95 0.180 0.567 11.93 0.136 0.451 11.94 Scal3R 1266 0.027 0.041 0.531 0.056 0.629 4.730 0.234 0.504 6.711 0.110 0.555 5.720 LoGeR 1255 0.031 0.044 0.413 0.066 0.650 8.799 0.135 0.478 10.81 0.082 0.513 9.803 LoGeR∗ 1255 0.034 0.053 0.482 0.061 0.671 8.614 0.087 0.536 9.533 0.067 0.563 9.073

- Table 29 | Per-Dataset Results on TUM. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.235 0.123 0.418 0.237 0.251 0.208 OOM OOM OOM (0.180) (0.335) (0.208) MASt3R 688.6 0.223 0.125 0.472 0.201 0.376 0.168 OOM OOM OOM (0.163) (0.424) (0.168)

###### End-to-End Feed-Forward

VGGT 1257 0.120 0.070 0.728 0.075 0.836 0.017 OOM OOM OOM (0.072) (0.782) (0.017) Fast3R 647.5 0.350 0.258 0.323 0.237 0.507 0.176 0.258 0.097 0.127 0.251 0.309 0.152 FastVGGT 1158 0.119 0.075 0.703 0.074 0.812 0.023 0.070 0.821 0.022 0.073 0.778 0.022 MUSt3R 423.4 0.237 0.124 0.627 0.194 0.714 0.032 T.O T.O T.O (0.159) (0.670) (0.032) MapAnything 1228 0.175 0.089 0.542 0.092 0.656 0.062 OOM OOM OOM (0.090) (0.599) (0.062) OmniVGGT 1217 0.140 0.063 0.727 0.072 0.786 0.057 OOM OOM OOM (0.068) (0.756) (0.057) 𝜋3 958.7 0.089 0.046 0.700 0.045 0.806 0.023 0.057 0.161 0.454 0.049 0.556 0.238 𝜋3-X 1360 0.084 0.046 0.730 0.045 0.833 0.018 OOM OOM OOM (0.046) (0.781) (0.018) AMB3R 1563 0.087 0.056 0.690 0.065 0.734 0.033 OOM OOM OOM (0.061) (0.712) (0.033) DA3-Small 34.3 0.180 0.134 0.546 0.116 0.632 0.092 0.105 0.616 0.151 0.118 0.598 0.121 DA3-Base 135.4 0.181 0.112 0.561 0.109 0.703 0.075 0.094 0.697 0.106 0.105 0.654 0.091 DA3-Large 410.9 0.160 0.089 0.643 0.081 0.811 0.038 OOM OOM OOM (0.085) (0.727) (0.038) DA3-Giant 1356 0.189 0.088 0.747 0.079 0.865 0.014 OOM OOM OOM (0.084) (0.806) (0.014) DA3-Nested 1690 0.180 0.088 0.769 0.080 0.854 0.015 OOM OOM OOM (0.084) (0.812) (0.015) WorldMirror 1263 0.121 0.082 0.694 0.090 0.795 0.023 OOM OOM OOM (0.086) (0.744) (0.023) VGGT-Omega 1144 0.066 0.042 0.821 0.040 0.899 0.013 – – – (0.041) (0.860) (0.013) DA-Next (Ours) 1304 0.151 0.057 0.697 0.048 0.813 0.016 OOM OOM OOM (0.085) (0.755) (0.016)

###### Online

Spann3r224 658.7 0.195 0.220 0.239 0.133 0.391 0.241 0.130 0.250 0.432 0.161 0.293 0.336 CUT3R 793.3 0.154 0.102 0.634 0.088 0.551 0.088 0.105 0.115 0.478 0.099 0.433 0.283 MonST3R 571.2 0.237 0.127 0.342 0.235 0.141 0.303 OOM OOM OOM (0.181) (0.241) (0.303) Point3R 828 0.153 0.134 0.410 0.110 0.421 0.145 0.114 0.218 0.292 0.119 0.350 0.218 Stream3R-S 1191 0.106 0.069 0.591 0.262 0.208 0.519 OOM OOM OOM (0.166) (0.400) (0.519) Stream3R-W 1191 0.106 0.069 0.591 0.221 0.221 0.553 OOM OOM OOM (0.145) (0.406) (0.553) StreamVGGT 1257 0.106 0.087 0.621 0.094 0.699 0.052 0.108 0.553 0.115 0.097 0.624 0.084 Page4D 1257 0.098 0.053 0.634 0.045 0.726 0.033 OOM OOM OOM (0.049) (0.680) (0.033) InfiniteVGGT 1257 0.106 0.087 0.601 0.093 0.701 0.052 0.108 0.554 0.117 0.096 0.619 0.084 Wint3R 749.5 0.147 0.102 0.588 0.108 0.489 0.165 0.126 0.163 0.301 0.112 0.413 0.233 LongStream-B 1191 0.108 0.064 0.573 0.145 0.576 0.104 0.183 0.188 0.262 0.131 0.446 0.183 LongStream-S 1191 0.108 0.064 0.573 0.108 0.448 0.179 0.181 0.146 0.271 0.118 0.389 0.225 LingbotMap∗-W 1158 0.167 0.066 0.655 0.116 0.671 0.048 0.207 0.378 0.265 0.130 0.568 0.157 LingbotMap∗-S 1158 0.167 0.066 0.655 0.104 0.677 0.048 0.095 0.658 0.051 0.088 0.663 0.049

###### Chunk-wise

VGGT-Long 1257 0.120 0.070 0.728 0.080 0.757 0.036 0.227 0.266 0.311 0.126 0.584 0.174 𝜋3-Long 958.7 0.089 0.046 0.700 0.057 0.784 0.025 0.172 0.441 0.146 0.091 0.642 0.086 DA3-Streaming 1356 0.189 0.088 0.747 0.081 0.822 0.019 0.107 0.429 0.144 0.092 0.666 0.081

###### SLAM-based

MASt3R-SLAM 688.6 0.230 0.232 0.129 0.337 0.306 0.361 0.323 0.426 0.125 0.297 0.287 0.243 VGGT-SLAM 1257 0.120 0.070 0.728 0.085 0.653 0.056 0.303 0.196 0.352 0.152 0.526 0.204

###### Test-Time Training

TTT3R 793.3 0.154 0.097 0.613 0.084 0.651 0.058 0.090 0.257 0.325 0.090 0.507 0.192 Scal3R 1266 0.186 0.070 0.668 0.126 0.680 0.054 0.217 0.197 0.260 0.138 0.515 0.157 LoGeR 1255 0.085 0.045 0.746 0.068 0.690 0.064 0.103 0.411 0.142 0.072 0.616 0.103 LoGeR∗ 1255 0.079 0.045 0.717 0.053 0.761 0.035 0.074 0.657 0.061 0.057 0.712 0.048

- Table 30 | Per-Dataset Results on Vkitti. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.215 0.286 0.668 0.326 0.702 16.67 OOM OOM OOM (0.306) (0.685) (16.67) MASt3R 688.6 0.114 0.210 0.386 0.504 0.677 21.24 OOM OOM OOM (0.357) (0.532) (21.24)

###### End-to-End Feed-Forward

VGGT 1257 0.037 0.055 0.922 0.046 0.921 5.778 OOM OOM OOM (0.051) (0.922) (5.778) Fast3R 647.5 0.228 0.424 0.115 0.303 0.135 72.45 0.324 0.146 75.05 0.350 0.132 73.75 FastVGGT 1158 0.038 0.062 0.918 0.048 0.918 4.304 0.047 0.784 6.673 0.052 0.873 5.488 MUSt3R 423.4 0.117 0.108 0.634 0.094 0.583 47.54 T.O T.O T.O (0.101) (0.608) (47.54) MapAnything 1228 0.120 0.113 0.690 0.106 0.681 28.69 OOM OOM OOM (0.109) (0.685) (28.69) OmniVGGT 1217 0.045 0.058 0.816 0.070 0.795 12.71 OOM OOM OOM (0.064) (0.806) (12.71) 𝜋3 958.7 0.112 0.082 0.843 0.076 0.898 5.051 0.074 0.804 5.849 0.077 0.848 5.450 𝜋3-X 1360 0.062 0.064 0.882 0.056 0.903 1.783 OOM OOM OOM (0.060) (0.892) (1.783) AMB3R 1563 0.045 0.061 0.950 0.050 0.921 4.534 OOM OOM OOM (0.056) (0.936) (4.534) DA3-Small 34.3 0.119 0.122 0.448 0.120 0.458 55.2 0.123 0.397 59.85 0.122 0.434 57.52 DA3-Base 135.4 0.142 0.105 0.550 0.105 0.485 45.83 0.103 0.440 56.79 0.104 0.492 51.31 DA3-Large 410.9 0.085 0.079 0.655 0.075 0.649 44.85 OOM OOM OOM (0.077) (0.652) (44.85) DA3-Giant 1356 0.072 0.064 0.787 0.061 0.749 20.83 OOM OOM OOM (0.063) (0.768) (20.83) DA3-Nested 1690 0.071 0.072 0.767 0.072 0.655 37.2 OOM OOM OOM (0.072) (0.711) (37.2) WorldMirror 1263 0.081 0.104 0.734 0.099 0.773 13.65 OOM OOM OOM (0.101) (0.754) (13.65) VGGT-Omega 1144 0.110 0.100 0.793 0.094 0.787 4.123 – – – (0.097) (0.790) (4.123) DA-Next (Ours) 1304 0.083 0.073 0.742 0.073 0.691 26.178 OOM OOM OOM (0.077) (0.717) (26.178)

###### Online

Spann3r224 658.7 0.327 0.453 0.366 0.443 0.346 38.77 0.497 0.278 33.67 0.464 0.330 36.22 CUT3R 793.3 0.070 0.075 0.650 0.069 0.588 41.24 0.063 0.349 58.95 0.069 0.529 50.1 MonST3R 571.2 0.254 0.216 0.485 0.391 0.478 17.98 OOM OOM OOM (0.304) (0.482) (17.98) Point3R 828 0.064 0.073 0.348 0.067 0.244 70.09 0.062 0.249 60.33 0.067 0.280 65.21 Stream3R-S 1191 0.053 0.092 0.708 0.128 0.584 51.36 OOM OOM OOM (0.110) (0.646) (51.36) Stream3R-W 1191 0.053 0.126 0.666 0.133 0.479 63.01 OOM OOM OOM (0.130) (0.572) (63.01) StreamVGGT 1257 0.054 0.322 0.808 0.280 0.602 58.14 0.227 0.417 65.29 0.276 0.609 61.71 Page4D 1257 0.043 0.061 0.823 0.053 0.837 7.484 OOM OOM OOM (0.057) (0.830) (7.484) InfiniteVGGT 1257 0.054 0.322 0.809 0.281 0.588 59.06 0.223 0.415 65.02 0.275 0.604 62.04 Wint3R 749.5 0.126 0.142 0.688 0.110 0.505 48.24 0.146 0.317 74.98 0.133 0.503 61.61 LongStream-B 1191 0.055 0.073 0.822 0.069 0.875 2.261 0.069 0.585 2.175 0.070 0.760 2.218 LongStream-S 1191 0.055 0.073 0.787 0.074 0.805 3.625 0.072 0.557 2.488 0.073 0.716 3.057 LingbotMap∗-W 1158 0.075 0.079 0.884 0.060 0.927 1.310 0.063 0.815 1.637 0.067 0.875 1.473 LingbotMap∗-S 1158 0.075 0.079 0.884 0.059 0.923 1.464 0.055 0.827 1.396 0.064 0.878 1.430

###### Chunk-wise

VGGT-Long 1257 0.037 0.055 0.922 0.041 0.959 0.719 0.044 0.853 1.324 0.047 0.911 1.021 𝜋3-Long 958.7 0.112 0.082 0.843 0.095 0.930 2.012 0.159 0.722 6.218 0.112 0.831 4.115 DA3-Streaming 1356 0.072 0.064 0.787 0.116 0.897 5.066 0.052 0.779 1.386 0.077 0.821 3.226

###### SLAM-based

MASt3R-SLAM 688.6 0.253 0.459 0.165 0.492 0.152 77.07 0.566 0.139 68.34 0.506 0.152 72.71 VGGT-SLAM 1257 0.037 0.055 0.922 0.054 0.956 2.172 0.045 0.847 1.076 0.051 0.909 1.624

###### Test-Time Training

TTT3R 793.3 0.070 0.073 0.541 0.069 0.655 32.52 0.064 0.405 33 0.068 0.534 32.76 Scal3R 1266 0.036 0.048 0.928 0.087 0.905 0.437 0.089 0.804 0.650 0.075 0.879 0.543 LoGeR 1255 0.054 0.065 0.751 0.061 0.819 3.494 0.058 0.707 2.756 0.061 0.759 3.125 LoGeR∗ 1255 0.054 0.068 0.715 0.056 0.846 2.877 0.052 0.771 2.949 0.059 0.778 2.913

- Table 31 | Per-Dataset Results on Waymo. Performance across different input regimes: Single Frame, Sparse, Medium, Dense, and the Average. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively. Out-of-memory (OOM) and Timeout (T.O) cells are shaded light red; Average values for those rows are wrapped in parentheses and excluded from per-column ranking. Within each sub-category, the bold value marks the in-group best. Note that DA-Next (Ours) is excluded from the per-column rankings.

Single Frame Sparse Medium Dense Average

#Params (M)

Method

AbsRel↓ AbsRel↓ AUC@30↑ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓ AbsRel↓ AUC@30↑ ATE↓

###### Optimization-based

DUSt3R 571.2 0.125 0.270 0.725 0.256 0.601 5.117 OOM OOM OOM (0.263) (0.663) (5.117) MASt3R 688.6 0.119 0.224 0.751 0.334 0.682 8.506 OOM OOM OOM (0.279) (0.716) (8.506)

###### End-to-End Feed-Forward

VGGT 1257 0.053 0.048 0.975 0.035 0.958 0.683 OOM OOM OOM (0.042) (0.966) (0.683) Fast3R 647.5 0.159 0.226 0.508 0.285 0.230 43.43 0.274 0.248 47.63 0.262 0.329 45.53 FastVGGT 1158 0.053 0.049 0.962 0.037 0.946 0.944 0.036 0.946 0.776 0.041 0.951 0.860 MUSt3R 423.4 0.125 0.128 0.792 0.147 0.648 14.2 T.O T.O T.O (0.138) (0.720) (14.2) MapAnything 1228 0.110 0.293 0.905 0.137 0.863 3.648 OOM OOM OOM (0.215) (0.884) (3.648) OmniVGGT 1217 0.055 0.054 0.951 0.044 0.923 5.228 OOM OOM OOM (0.049) (0.937) (5.228) 𝜋3 958.7 0.101 0.086 0.887 0.069 0.883 1.200 0.068 0.866 1.207 0.074 0.879 1.203 𝜋3-X 1360 0.060 0.058 0.972 0.044 0.968 0.601 OOM OOM OOM (0.051) (0.970) (0.601) AMB3R 1563 0.054 0.047 0.975 0.037 0.956 0.666 OOM OOM OOM (0.042) (0.965) (0.666) DA3-Small 34.3 0.144 0.179 0.671 0.145 0.537 30.89 0.141 0.532 30.19 0.155 0.580 30.54 DA3-Base 135.4 0.112 0.147 0.754 0.114 0.580 23.83 0.112 0.543 22.18 0.124 0.626 23.01 DA3-Large 410.9 0.115 0.164 0.862 0.117 0.851 9.856 OOM OOM OOM (0.140) (0.856) (9.856) DA3-Giant 1356 0.083 0.078 0.985 0.067 0.986 0.303 OOM OOM OOM (0.072) (0.985) (0.303) DA3-Nested 1690 0.099 0.125 0.905 0.094 0.903 2.907 OOM OOM OOM (0.110) (0.904) (2.907) WorldMirror 1263 0.068 0.104 0.902 0.075 0.875 4.761 OOM OOM OOM (0.090) (0.888) (4.761) VGGT-Omega 1144 0.088 0.088 0.972 0.073 0.949 0.975 – – – (0.081) (0.960) (0.975) DA-Next (Ours) 1304 0.064 0.059 0.943 0.052 0.960 0.884 OOM OOM OOM (0.058) (0.952) (0.884)

###### Online

Spann3r224 658.7 0.178 0.335 0.526 0.288 0.431 29.38 0.294 0.360 29.13 0.306 0.439 29.25 CUT3R 793.3 0.057 0.072 0.865 0.062 0.709 6.451 0.061 0.573 10.89 0.065 0.716 8.670 MonST3R 571.2 0.092 0.232 0.730 0.296 0.726 9.918 OOM OOM OOM (0.264) (0.728) (9.918) Point3R 828 0.062 0.074 0.577 0.062 0.331 43.64 0.062 0.282 44.75 0.066 0.396 44.19 Stream3R-S 1191 0.059 0.069 0.879 0.240 0.573 42.25 OOM OOM OOM (0.155) (0.726) (42.25) Stream3R-W 1191 0.059 0.071 0.850 0.205 0.514 47.43 OOM OOM OOM (0.138) (0.682) (47.43) StreamVGGT 1257 0.050 0.205 0.856 0.297 0.674 28.85 0.308 0.600 29.34 0.270 0.710 29.09 Page4D 1257 0.050 0.048 0.954 0.041 0.947 0.696 OOM OOM OOM (0.045) (0.950) (0.696) InfiniteVGGT 1257 0.050 0.204 0.856 0.293 0.676 28.54 0.305 0.603 29.63 0.268 0.712 29.09 Wint3R 749.5 0.130 0.124 0.765 0.125 0.597 21.38 0.131 0.458 25.96 0.127 0.607 23.67 LongStream-B 1191 0.057 0.065 0.895 0.057 0.906 0.786 0.058 0.840 0.896 0.060 0.880 0.841 LongStream-S 1191 0.057 0.064 0.839 0.056 0.855 2.367 0.053 0.806 1.323 0.058 0.833 1.845 LingbotMap∗-W 1158 0.073 0.086 0.950 0.060 0.967 0.509 0.063 0.954 0.644 0.069 0.957 0.576 LingbotMap∗-S 1158 0.073 0.086 0.950 0.060 0.965 0.645 0.058 0.945 0.659 0.068 0.954 0.652

###### Chunk-wise

VGGT-Long 1257 0.053 0.048 0.975 0.047 0.930 0.889 0.059 0.878 1.408 0.051 0.927 1.148 𝜋3-Long 958.7 0.101 0.086 0.887 0.117 0.865 1.642 0.111 0.812 2.409 0.105 0.855 2.026 DA3-Streaming 1356 0.083 0.078 0.985 0.065 0.981 0.286 0.062 0.967 0.233 0.068 0.977 0.260

###### SLAM-based

MASt3R-SLAM 688.6 0.216 0.494 0.478 0.477 0.392 32.29 0.468 0.422 23.64 0.480 0.430 27.96 VGGT-SLAM 1257 0.053 0.048 0.975 0.061 0.841 1.431 0.095 0.797 2.519 0.068 0.871 1.975

###### Test-Time Training

TTT3R 793.3 0.057 0.072 0.815 0.061 0.847 3.565 0.061 0.813 4.001 0.065 0.825 3.783 Scal3R 1266 0.059 0.049 0.968 0.180 0.870 0.673 0.088 0.856 1.382 0.106 0.898 1.027 LoGeR 1255 0.058 0.057 0.944 0.051 0.966 0.681 0.043 0.953 0.428 0.050 0.954 0.555 LoGeR∗ 1255 0.054 0.053 0.949 0.039 0.976 0.391 0.037 0.965 0.275 0.043 0.963 0.333

- Table 32 | Metric-Depth Comparison on SpatialBench. We compare the native metric-depth predictions (no median/scale alignment) of all 6 methods on SpatialBench that emit metric-scale depth, across the Single Frame, Sparse, and Medium settings, together with the Average across all three settings. The best, second-best, and third-best results in each column are highlighted in deep blue, medium blue, and light blue, respectively.

Single Frame Sparse Medium Average

#Params (M)

Method

AbsRel↓ SqRel↓ RMSE↓ 𝛿1.25 ↑ AbsRel↓ SqRel↓ RMSE↓ 𝛿1.25 ↑ AbsRel↓ SqRel↓ RMSE↓ 𝛿1.25 ↑ AbsRel↓ SqRel↓ RMSE↓ 𝛿1.25 ↑ DA3-Nested 1689.85 1.599 16.57 1.477 0.507 0.870 5.295 1.569 0.617 1.188 19.64 2.000 0.552 1.219 13.83 1.682 0.559 MapAnything 1228.49 2.476 31.96 2.440 0.365 2.368 23.54 2.592 0.437 3.103 27.87 2.374 0.407 2.649 27.79 2.469 0.403 AMB3R 1563.12 2.235 48.59 1.473 0.388 1.133 4.934 1.360 0.519 1.407 3.990 1.242 0.521 1.592 19.17 1.358 0.476 𝜋3-X 1360.03 3.187 13.87 1.842 0.400 1.771 9.361 1.779 0.461 1.957 6.269 1.705 0.449 2.305 9.832 1.775 0.436 DANext 1303.76 1.484 3.369 3.595 0.070 0.713 3.280 4.235 0.117 0.713 3.350 4.261 0.116 0.970 3.333 4.030 0.101 MASt3R-SLAM 688.64 2.672 5.479 2.962 0.231 1.570 4.043 3.468 0.191 2.264 4.992 3.555 0.209 2.169 4.838 3.329 0.210

#### G. Limitations

We report the limitations of SpatialBench in this section.

Evaluation Cost. Evaluating 41 models across 100+ scenes per model under the dense regime is time-consuming. This can be mitigated by distributing evaluation across multiple GPUs in parallel.

Memory Constraints. All evaluations are conducted on H200 GPUs with 141GB VRAM. We have not tested models under larger memory configurations such as B100 or B200, and performance or behavior may differ on such hardware.

Hyperparameter Selection. We acknowledge that some evaluated methods may require task- or scene-specific hyperparameter tuning to achieve optimal performance. However, such tuning falls outside the scope of this benchmark. We follow the recommended configurations provided in each method’s official codebase to ensure a fair and consistent comparison across all methods.

Expanding Method Coverage. As of the submission deadline, a number of newly released models continue to be open-sourced. Therefore, we cannot guarantee complete coverage of all existing methods. We commit to continuously integrating and evaluating new methods as they become available.

