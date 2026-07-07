#### SpatialVID: A Large-Scale Video Dataset with Spatial Annotations

### arXiv:2509.09676v2[cs.CV]18Dec2025

Jiahao Wang1† Yufeng Yuan1† Rujie Zheng1† Youtian Lin1 Jian Gao1 Lin-Zhuo Chen1 Yajie Bao1 Yi Zhang1 Chang Zeng1 Yanxi Zhou1 Xiao-Xiao Long1 Hao Zhu1 Zhaoxiang Zhang2 Xun Cao1 Yao Yao1‡

1Nanjing University 2Institute of Automation, Chinese Academy of Sciences †Equal contribution ‡Corresponding author

[Figure 1]

## 2.7M

e Video Length

e

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

··· ···

[Figure 13]

7089h

DCylnipasmic

Raw Video

[Figure 14]

[Figure 15]

[Figure 16]

Clip

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

··· ···

i i

[Figure 21]

[Figure 22]

[Figure 23]

Frames

··· ···

Camera Pose

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|
|---|

|[Figure 29]|
|---|

|[Figure 30]|
|---|

|W|
|---|

|[Figure 31]|
|---|

Depth

··· ···

| |
|---|

| |
|---|

|S|
|---|

|A|
|---|

|D|
|---|

[Figure 32]

[Figure 33]

[Figure 34]

Dynamic Mask

Motion Instructions

··· ···

Structured Caption:

Category Tags: Rural (Traditional Village Street), Bright, Daytime, Sunny, Sparse

Motion Trends: forward translate, left translate

Scene Description: ··· depicts a charming street in a traditional Swiss village, lined with wooden houses adorned with vibrant flower boxes. Two women are present: one stands near the end of the street, while the other walks a dog, ···

Camera Description: ··· glides steadily forward, translating through the scene with a smooth, consistent pace. As it moves, it subtly shifts left, revealing more of the village street and its wooden houses. The motion remains steady throughout ···

# 127M a

o Captoion Words

Shot Summary: The camera smoothly advances down a cobbled path, flanked by charming wooden houses bursting with flowers. As it moves left, the view expands, revealing more of the village and the distant snow-capped mountains. The serene, sunlit scene unfolds with quiet elegance, capturing the essence of a timeless Alpine retreat.

720M

Images Annotated

nntatee

Figure 1. We introduce SpatialVID, a large-scale video dataset with explicit spatial annotations including camera poses, depth maps, structured captions and serialized motion instructions. The dataset consists of 7,089 hours of real-world dynamic scenes. An exemplar of our SpatialVID is shown on the right.

###### Abstract

Significant progress has been made in spatial intelligence, spanning both spatial reconstruction and world exploration. However, the scalability and real-world fidelity of current models remain severely constrained by the scarcity of largescale, high-quality training data. While several datasets provide camera pose information, they are typically limited in scale, diversity, and annotation richness, particularly for real-world dynamic scenes with ground-truth camera motion. To this end, we collect SpatialVID, a dataset consisting of a large corpus of in-the-wild videos with diverse scenes, camera movements and dense 3D annotations such as per-frame camera poses, depth, and motion instructions. Specifically, we collect more than 21,000 hours of raw video, and process

them into 2.7 million clips through a hierarchical filtering pipeline, totaling 7,089 hours of dynamic content. A subsequent annotation pipeline enriches these clips with detailed spatial and semantic information, including camera poses, depth maps, dynamic masks, structured captions, and serialized motion instructions. Analysis of SpatialVID’s data statistics reveals a richness and diversity that directly foster improved model generalization and performance, establishing it as a key asset for the video and 3D vision research community. Through extensive validation experiments, we demonstrate SpatialVID’s effectiveness across tasks such as controllable video generation, world simulation and geometric reconstruction, providing a strong foundation for spatial intelligence research.

###### 1. Introduction

Perceiving, reasoning about, and interacting with the 3D world are fundamental components of artificial general intelligence. Recent advances in intrinsic 3D modeling have enabled the generation of spatially consistent and navigable environments directly from text or image prompts, bridging reconstruction and world simulation [5]. These models embody the dual objectives of understanding the current physical world and imagining plausible future ones.

The field of 3D scene understanding has evolved from optimization-based methods to scalable, data-driven representations of geometry and motion. Classical approaches such as Structure-from-Motion (SfM) [45] and neural MultiView Stereo (MVS) [67] laid the foundation for geometric reasoning. Subsequently, large-scale neural models have recently emerged: the LRM series [18, 62, 75] learns to reconstruct high-quality 3D objects from single images or text prompts; DUSt3R series [9, 58] achieve robust multi-view matching; VGGT [55] directly predicts key 3D attributes such as camera poses and point clouds in a feed-forward manner. Collectively, these models exemplify a shift toward scalable, data-driven reconstruction and synthesis with reduced reliance on explicit geometry. Despite these advancements, building large-scale 3D datasets remains challenging due to costly data acquisition and the reliance on accurate annotation pipelines [15, 43, 68]. In contrast, videos are abundant and naturally encode spatial, temporal, and semantic cues, offering a scalable foundation for 3D learning at scale. Leveraging such data presents a promising path toward high-fidelity reconstruction and dynamic world simulation.

Beyond reconstruction, video generation has emerged as a key capability for world modeling, serving as a simulator to represent and predict physical dynamics. Recent models including UNet-based diffusion methods like Stable Video Diffusion (SVD) [7], DiT-based architectures such as Sora [38], HunyuanVideo [25], and CogVideoX [66], as well

- as autoregressive approaches [6] enable high-fidelity video synthesis. To achieve controllable and physically grounded generation, recent efforts extend to object motion [69], camera trajectory control [17, 61], and 3D signal integration [72]. These capabilities represent crucial steps toward physically grounded video simulation and align with recent world models such as Cosmos [1], HunyuanWorld [52], and Genie3 [5]. Despite this progress, current datasets still lack detailed finegrained semantic and spatial metadata, limiting their capacity to support physically grounded video synthesis.

Current data falls into two disjoint categories. Large-scale video datasets provide rich semantics but lack explicit 3D information [11, 37], offering no geometric ground truth and forcing models to learn spatial relations implicitly from pixels. Conversely, spatial datasets such as CO3D [42], RealEstate10K [82], and Tartanair [59] provide accurate geometry and camera parameters but remain small, object-

centric, or synthetic. This division hampers progress toward spatiotemporally coherent world simulators and underscores the need for a dataset that bridges scene reconstruction and world simulation, closing the gap between semantic diversity without geometry and geometric precision without semantics. To bridge the gap between dynamic videos and spatial understanding, we introduce SpatialVID (Fig. 1), a large-scale multimodal dataset that connects raw pixels to the physical world. Starting from over 21,000 hours of internet video manually screened for diversity and motion richness, we apply a hierarchical filtering pipeline to obtain 7,089 hours of high-quality 720P clips with reliable camera motion. From this corpus, we derive SpatialVID-HQ, a 1,111-hour balanced subset optimized for robust training and evaluation. Through extensive experiments, we demonstrate that SpatialVID advances tasks such as controllable video generation, world simulation and geometric reconstruction, providing a strong foundation for spatial intelligence research.

To our knowledge, SpatialVID is the largest dataset of dynamic videos with explicit geometric annotations and makes three primary contributions:

- • Manually Screened Videos with Camera Motions: SpatialVID is built from 21,000+ hours of internet videos manually screened for rich scene motion. This motionfirst curation and processing yield diverse, high-quality clips for training spatially aware models.
- • Comprehensive Geometric Annotations: Each clip includes camera poses and depth maps generated via an adjusted pipeline [30], providing explicit 3D grounding and motion dynamics absent in prior video datasets.
- • Spatially-Aware Captions and Motion Instructions: Structured captions describe scene content, camera motion, and semantic attributes (e.g., weather, lighting, time), while motion instructions derived from trajectories offer precise supervision for navigation and control tasks.

###### 2. Related Work

Scene Reconstruction Traditionally, scene reconstruction has followed two main trajectories: Simultaneous Localization and Mapping (SLAM) and Structure-from-Motion (SfM). Classical systems such as ORB-SLAM [36] and COLMAP [45] achieve accurate geometry estimation and real-time tracking but depend heavily on handcrafted features, limiting robustness under challenging conditions. To overcome these constraints, learnable Multi-View Stereo (MVS) methods have emerged, with MVSNet [67] introducing deep cost-volume reasoning and Transformer-based models such as DUSt3R [58], MASt3R [27], and their successors [9, 34, 64] achieving robust feed-forward multi-view matching. Recent frameworks like VGGT [55] further integrate these advances into end-to-end 3D reconstruction pipelines. Extending this trend to dynamic environments,

- Table 1. Comparisons with previous datasets with spatial information. SpatialVID is a million-level, dynamic and open-scenario high-quality video dataset with rich annotated geometric and semantic information. Syn. denotes synthetic data; Sta. and Dyn. indicate static and dynamic scenes. In Geometry Info. column, C. denotes camera, D. denotes depth or point cloud.

Dataset Domain Real/Syn. Dyn./Sta. Geometry Info. Caption # Video Clips # Frames BlendedMVS [68] Open Syn. Sta. C. D. Label 113 (Scenes) 17K Multi-Cam Video [4] Open Syn. Dyn. C. Label 136K ∼11.02M PointOdyssey [79] Walk Syn. Dyn. C. D. N/A 159 200K Camerabench [32] Open Real&Syn. Dyn. N/A Label, Short 3,381 ScanNet [14] Indoor Real Sta. C. D. N/A 1500 (Scenes) 2.50M MVImgNet [73] Object-Centric Real Sta. C. Label 219,188 6.50M CO3Dv2 [42] Object-Centric Real Sta. C. Label 19K 1.50M DL3DV [33] Open Real Sta. N/A Label 10,510 512M WebVi3D [35] Open Real Sta. N/A - 15.99M 320M Dynpose100k [44] Open Real Dyn. C. Short 100,131 6.81M CamVid-30K [77] Open Real Dyn. C. N/A 30K Stereo4d [22] Fisheye Real Dyn. C. D. Short 110K 10M RealEstate10K [82] Indoor Real Sta. C. N/A ∼80K ∼10M Waymo [49] Drive Real Dyn. C. N/A 1150 230K Map-free [3] Object-Centric Real Sta. C. N/A 655 (Scenes) ∼560K Princeton365 [24] open Real Dyn. C. D. N/A 365 ∼1.19M SpatialVID Open Real Dyn. C. D. Structured Caption 2.71M 127.60M SpatialVID-HQ Open Real Dyn. C. D. Structured Caption 0.37M 20.63M

recent DUSt3R variants [12, 57, 74] incorporate motion awareness extend the capabilities of DUSt3R to incorporate dynamic elements, while dense approaches including CasualSAM [76] and MegaSaM [30] leverage optical flow or SLAM backbones [53] for robust tracking. Given its demonstrated robustness in unconstrained, in-the-wild videos, we utilize an enhanced version of MegaSaM in our work to generate the initial geometric annotations for our dataset.

World Simulator The concept of a world simulator lies

- at the core of spatial intelligence, encompassing the ability to perceive, simulate, and interact with dynamic environments. Recent progress in video generation has made this increasingly feasible. Early progress was driven by UNetbased video diffusion models such as Stable Video Diffusion (SVD) [7], followed by DiT-based architectures [25, 38, 66], which significantly advance fidelity, scalability, and temporal consistency. In parallel, autoregressive approaches [6, 70] remain competitive for generating high-quality, long-form videos. Achieving effective world simulation further requires controllability over 3D structure and motion. Models such as DragNUWA [69], CameraCtrl [17], and MotionCtrl [61] enable explicit manipulation of objects and camera trajectories, while GameFactory [71] introduces action-level control for interactive synthesis. Incorporating 3D data further enhances spatial coherence, as demonstrated by ViewCrafter [72], which integrates point clouds into video generation. These controlled and geometry-aware paradigms contribute to the development of world models such as Cosmos Predictor [1], HunyuanWorld [52], and Genie3 [5], aiming to simulate spatiotemporal dynamics and support interactive exploration within complex virtual environments.

Datasets with Spatial Annotations High-quality spatial datasets are crucial for advancing world reconstruction and exploration, ideally combining 3D geometry, dynamics, and semantic richness. Existing efforts fall into three main categories. Synthetic datasets such as Multi-Cam Video [4], virtualKITTI [8], and BlendedMVS [68] offer precise geometry information but demand heavy engineering, limiting scalability. Real-world datasets like CO3DV2 [42] and RealEstate10K [82] rely on SfM or SLAM-based annotation for efficiency, often yield sparse trajectories and struggle under dynamic motion. Video-mined datasets including CamVid-30K [77] and DynPose100K [44] provide largescale semantic or motion cues but often lack fine-grained geometry or diversity. Recent datasets including CameraBench [32], VLM4D [81], and the concurrent Sekai [31] improve geometric–semantic integration but remain limited in semantic richness, motion coverage, or scale. To address these limitations, we introduce SpatialVID, a large-scale realworld dataset of diverse dynamic scenes with detailed geometric and semantic annotations. As shown in Tab. 1, SpatialVID outperforms prior datasets in scale and annotation quality, providing a rich foundation for spatially grounded representation learning and world simulation research.

###### 3. SpatialVID Curation

As illustrated in Fig. 2, the SpatialVID curation pipeline comprises three main stages: filtering, annotation, and sampling. Sec. 3.1 covers video collection, format unification, and clip segmentation; Sec. 3.2 introduces multi-dimensional filtering based on four key metrics to retain diverse, motion-rich clips; Sec. 3.3 describes geometric annotation via camera

Filter Pipeline

Annotation Pipeline

Sample Pipeline

9216 GPU·H

[Figure 35]

[Figure 36]

MegaSaM Sam2

[Figure 37]

[Figure 38]

Clip Frames

[Figure 39]

[Figure 40]

| | |
|---|---|
|[Figure 41]| |
|[Figure 42]| |
|[Figure 43]| |
| | |
|[Figure 44]| |
|[Figure 45]| |

[Figure 46]

[Figure 47]

3840 GPU·H

69120 GPU·H

Raw Videos

[Figure 48]

Motion Probability

H.265Recoding

SpatialVID

Preprocessing

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

7088.76 hours

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

2715740 clips

|[Figure 58]|
|---|

|[Figure 59]|
|---|

|[Figure 60]|
|---|

|W|
|---|

|[Figure 61]|
|---|

[Figure 62]

[Figure 63]

|S|
|---|

|A|
|---|

|D|
|---|

Dynamic Mask

Motion Instructions

Sample

[Figure 64]

Depth

Camera Pose

6M Clips

4.5B Tokens

[Figure 65]

Hierarchical

[Figure 66]

Structured Caption

Filtering

3

Camera Descriptions

Scene Descriptions

Optimized Captions

Combine

9B Tokens

Category Tags Shot Summary

[Figure 67]

SpatialVID-HQ

1111.55 hours

Origin Captions

365362 clips

3M Clips

- Figure 2. Overview of the curation pipeline. The pipeline comprises three stages: filtering, annotation, and sampling. We start from manually collected web videos with notable camera motion. In the filtering stage, raw videos are hierarchically preprocessed and filtered. The annotation stage adds geometric and semantic labels and derives motion instructions from camera poses. The sampling stage then balances clips by motion and category to form a high-quality subset (SpatialVID-HQ) with well-distributed classes for downstream tasks.

poses and depth maps; Sec. 3.4 details motion instruction generation; and Sec. 3.5 outlines structured captions integrating scene, motion, and semantic attributes.

###### 3.1. Data Collection and Preprocessing

Source Data. While large-scale video datasets such as Panda70M [11] and MiraData [23] offer valuable benchmarks, they remain inadequate for our needs due to limited viewpoints and motion diversity. Processing the Panda70M validation split through our pipeline retained only about 10% of clips that met our quality standards, yielding insufficient data for large-scale reconstruction (Sec. 4.2).

To obtain richer and more varied data, we turned to YouTube, leveraging its vast and heterogeneous collection of high-resolution videos. We queried motion-related keywords (walk, tour, drone, etc.) to capture videos with smooth and diverse camera trajectories. Candidate videos were then manually screened for compatibility with the MegaSaM reconstruction pipeline. We excluded broken videos and videos with titles containing inappropriate terms (e.g., "Panoramic camera"). Footage with heavy occlusion or intrusive overlays (e.g., logos, subtitles) was also discarded.

This process yielded a curated corpus with stable motion, rich parallax, and detailed textures, ideal for 3D reconstruction. In total, we collected 33,443 YouTube videos (21,789 hours), covering a broad range of motion types, camera trajectories, and scene categories. Refer to Supplementary Materials for additional details about video statistics.

Data Preprocessing. We segment the collected long-form videos into 3–15 s clips using a modified PySceneDetect [10].

To better handle aesthetic transitions such as fades, we adjust sensitivity thresholds and adopt an interval-based multiframe comparison strategy, significantly improving segmentation accuracy and efficiency. Given the variability in encoding formats and resolutions, all clips are then standardized to H.265-encoded MP4 at 1280 × 720 to ensure consistency and compatibility across the pipeline. After preprocessing, this yields over 7 million distinct video clips.

###### 3.2. Video Quality Filtering

We filter all candidate videos using four key metrics: aesthetic quality, motion intensity, text interference, and luminance. This stage ensures that only clips with rich motion and clear visuals are retained, improving the reliability of downstream camera pose estimation. Given the large scale of our collection, such filtering is crucial for maintaining dataset quality and suitability for training and evaluation.

Specifically, we adopt a CLIP+MLP aesthetic predictor [47] to remove visually unappealing clips. Luminance filtering discards overexposed or underexposed samples to ensure consistent brightness. PaddleOCR [13] is used to detect and eliminate clips with excessive text based on the text-area ratio. Finally, motion filtering leverages lightweight VMAF metric [80] to retain videos with sufficient motion diversity for stable reconstruction. Representative filtering examples are provided in Supplementary Materials.

###### 3.3. Geometry Information Annotation

We employ MegaSaM [30] as our primary camera estimator, chosen for its strong balance between accuracy and efficiency

[Figure 68]

[Figure 69]

Scene Description

- 2. LLM Orig.

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

| |
|---|

Captions

Orig. Camera Description

| |
|---|

| |
|---|

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Camera Pose

Clip Frames

1. VLM

Scene Keywords

Figure 3. Structured caption generation. A VLM produces initial motion and scene descriptions, which the LLM refines using camera poses to yield structured captions.

compared to existing methods [46, 53, 55, 64, 74]. Comparisons of several methods are presented in Supplementary Materials. Our annotation pipeline is poised to improve by integrating future estimators with enhanced performance (e.g., ViPE [20]).

While effective, MegaSaM faces challenges under extreme conditions, such as dominant moving objects, collinear motion, or reliance on external monocular depth. To address these issues, we replace its original depth module with UniDepth v2 [39] and Depth Anything v2 [65], significantly improving depth accuracy and robustness. Furthermore, we first obtain candidate regions via adaptive thresholding and contour detection, then sample anchor points from them and use these as SAM2 [41] prompts to extract dynamic masks. From these refined masks, we compute dynamic ratio that quantifies the proportion of dynamic regions per frame.

To ensure physically plausible trajectories, we employ an acceleration-based detector to identify abrupt, non-physical motion fluctuations. Additionally, we introduce three metrics for quantitative camera motion analysis: (1) MoveDist, measuring total trajectory length; (2) RotAngle, capturing cumulative camera rotation; and (3) TrajTurns, estimating the number of significant directional changes.

- 3.4. Motion Instruction Decomposition

[Figure 80]

[Figure 81]

[Figure 82]

Category Tags

Scene Type & Secondary Tag

[Figure 83]

Opt. Camera Description

Scene Abstract

Light Intensity

Weather Type Shot Summary Crowd Density

Motion Trends

Time of Day

Given the importance of motion instructions for controllable and semantically grounded learning in models such as Hunyuan-GameCraft [29], our dataset explicitly incorporates them to enable interpretable motion understanding. We derive these instructions from estimated camera pose sequences, where relative translations and rotations between consecutive frames capture the camera’s motion dynamics. To enhance robustness, temporal smoothing filters are ap-

[Figure 84]

[Figure 85]

[Figure 86]

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

4

3

1

2

[Figure 97]

[Figure 98]

2

1

Cameraa e a Posee

3

4

Orig. Camera Description Opt. Camera Description

··· camera then pans slightly to the right, revealing more of the street and the people walking along it. The dolly forward continues at a steady pace, maintaining ···

··· translates forward along the cobblestone street, maintaining a steady pace. It subtly shifts left as it moves, revealing more of the village scene and the people ···

Figure 4. Effect of spatial enhancement. After applying spatial enhancement, the LLM corrected the incorrect direction (right) output by the VLM to left.

plied to suppress jitter and noise prior to instruction extraction. Perceptible motion segments are then identified through magnitude-based thresholding, generating instructions only when pose variations exceed predefined limits to avoid trivial movements. Finally, motion signals are mapped to a controlled vocabulary of cinematographic terms [32], such as dolly in (forward translation), pan left (horizontal rotation), and truck right (lateral translation). Visualized annotation results are provided in Supplementary Materials. This standardized decomposition ensures clarity, consistency, and semantic usability for downstream training.

###### 3.5. Semantic Information Annotation

Multimodal models have made remarkable strides in bridging vision and language, employing diverse strategies for video caption generation. Early efforts such as VATEX [60] rely on manual annotation to ensure linguistic precision but lack scalability. More recent datasets [11, 37, 50] leverage multimodal large language models (MLLMs) for automatic caption generation, striking a balance between efficiency and semantic relevance. Others adopt structured or multi-stage pipelines [23, 56] to improve vision–text alignment, while OpenHumanVid [28] enhances caption reliability through LLM-based voting and reformatting. Despite these advances, vision–language models (VLMs) such as Gemini [51] remain limited in spatial reasoning, often overlooking geometric and camera-related details (Fig. 4). Recent works like CameraBench [32], VLM4D [81], and 3D LLM-Mem [19] highlight these shortcomings and explore early attempts to integrate depth and pose awareness.

Building upon these insights, we propose a structured caption generation framework that unifies VLM and LLM capabilities while explicitly incorporating camera poses (Fig. 3),

[Figure 99]

[Figure 100]

[Figure 101]

(b) Luminance Distribution (c) Motion Distribution

(a) Aesthetics Distribution

>2 turns 1%

[Figure 102]

2 turns 2%

[Figure 103]

1 turn 12.2%

0 turn 1.1%

>2 turns 1.1%

Panda-70M >21.6%turns

2 turns 7%

2 turns 14.2%

unqualified 83.7%

0 turn 30.7%

SpatialVID HQ

0 turn 52.9%

SpatialVID

1 turn 39%

1 turn 53.5%

(d) RotAngle Distribution (e) TrajTurns Distribution () MoveDist Distribution

- Figure 5. Dataset quality comparison. Comparison of SpatialVID, its balanced subset (SpatialVID-HQ), and the Panda70M-test set processed with the same pipeline. Histograms and KDE curves reveal distribution patterns across quality metrics, showing that SpatialVIDHQ achieves consistently superior quality, validating our manual collection, filtering, and sampling process.

enabling captions that are both semantically expressive and spatially grounded. Our pipeline operates in two stages: (1) visual parsing, where Gemini-2.0-Flash analyzes sampled frames to produce initial scene and motion descriptions; and (2) language refinement, where Qwen3-30B-A3B [63] refines these outputs using camera pose priors to correct motion directions and ensure spatial coherence. The refined captions integrate scene semantics, camera motion, and multilevel attributes (e.g., scene type, lighting, weather), forming a hierarchical textual representation that captures spatial, temporal, and cinematic structure. Further implementation details are provided in Supplementary Materials.

###### 4. Dataset Analysis

###### 4.1. Data Sampling

We aim to curate clips that maximize both quality and diversity. To this end, we first tighten thresholds across core quality metrics to elevate visual standards, and then balance semantic tags and trajectory statistics to ensure diversity. The resulting collection, SpatialVID-HQ, comprises over one thousand hours of high-quality spatial videos.

###### 4.2. Comparison with Panda-70M

Panda-70M is a large-scale video dataset designed to support research on vision-language and video understanding. However, it exhibits several quality limitations: many clips are static, contain flickering artifacts, and include captions that lack motion descriptions. To demonstrate the advantages of our dataset, we perform a systematic comparison between SpatialVID and Panda-70M, as illustrated in Fig. 5.

Across key video-quality metrics including Aesthetics (Fig. 5a), Luminance (Fig. 5b), and Motion (Fig. 5c), both SpatialVID and its high-quality subset SpatialVID-HQ show more compact distributions, indicating greater consistency and higher average quality. Regarding camera motion statistics, Panda-70M is dominated by static viewpoints, as seen in the distributions of camera rotation (Fig. 5d) and translation distance (Fig. 5f). In contrast, SpatialVID achieves a balanced and realistic distribution of camera movements. Finally, Fig. 5e shows the distribution of trajectory turns (Arc Nums): over 80% of Panda-70M videos cannot be reconstructed by MegaSaM due to insufficient motion, whereas SpatialVID-HQ deliberately increases the proportion of clips with curved or turning trajectories, providing a richer and more diverse motion profile.

###### 5. SpatialVID Validation Tasks

In this section, we primarily evaluate the effectiveness of the SpatialVID dataset on camera-controlled video generation, which serves as our main task to assess spatially consistent visual synthesis. In addition, we conduct large-scale spatial reconstruction and pose estimation experiments as complementary validations, providing further evidence of the dataset’s quality and geometric fidelity.

###### 5.1. Camera-Controlled Video Generation

Baseline. We build upon the camera-control injection mechanism introduced in ReCamMaster [4] (excluding intrinsic parameters), adopting the Wan2.2 architecture [54] as the base model and T5 [40] as the text encoder. For each frame,

- Table 2. Quantitative comparison of camera-controlled video generation performance across different training datasets on Sekai-Real [31], RealEstate10K[82], and SpatialVID benchmarks.

Camera Accuracy Visual Quality VBench TransErr↓ RotErr↓ CamMC↓ CLIP-T↑ CLIP-F ↑

Benchmark Training Data

Subject Consistency

Background Consistency

Motion Smoothness

Aesthetic Quality

Imaging Quality

RE10K 7.46 1.15 7.91 30.38 99.53 96.45 94.52 99.05 55.99 72.71 Sekai-Real 8.50 1.86 9.35 30.51 99.20 97.09 94.21 98.57 52.17 72.02

RE10K

###### SpatialVID-HQ 7.42 0.99 7.72 30.54 99.59 98.04 95.41 98.77 56.26 75.68

RE10K 8.17 1.51 8.78 34.97 99.35 96.25 93.24 99.12 54.91 71.27

Sekai

Sekai-Real 6.49 1.47 7.12 33.98 98.57 93.25 91.24 98.06 52.53 72.32 SpatialVID-HQ 6.04 1.43 6.70 35.19 99.28 96.39 93.49 98.88 54.14 73.13

RE10K 5.16 4.07 8.59 30.22 99.02 94.31 92.70 98.68 55.00 66.23 Sekai-Real 5.63 4.70 9.39 30.25 98.13 91.63 91.93 97.97 52.90 66.14

SpatialVID

###### SpatialVID-HQ 4.33 3.81 7.57 30.26 98.69 94.89 93.23 98.18 55.11 70.38

[Figure 104]

[Figure 105]

The camera surges ahead along the winding highway, its path weaving subtly to the right as it passes beneath towering overpasses. The scene unfolds in a controlled, industrial setting, where nature is tamed by concrete and steel, all bathed in the muted light of an overcast day.

The camera moves smoothly forward along a snow-laden road, flanked by towering trees draped in white. As the path bends gently to the left, the scene unfolds in a peaceful stillness, capturing the quiet beauty of a remote winter landscape.

Ours-HQConditionSekaiRE10K

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

- Figure 6. Different training datasets performance on SpatialVID. Under identical training settings, models trained on our dataset produce videos with more consistent appearance and significantly improved camera controllability.

camera extrinsics are processed by a learnable encoder Ec that projects them into the video token dimension. The encoded features are then fused with visual tokens and injected into each transformer block, enabling precise, framelevel camera control. We train models on Sekai-Real [31], RealEstate10K, and our SpatialVID-HQ subset under identical training configurations. Each model is fine-tuned on 32 H20 GPUs for 2 days and detailed training setup is provided in Supplementary Material.

Metrics. Following prior works [17, 78], we evaluate camera controllability using TransErr, RotErr, and CamMC, computed on poses estimated by MegaSaM [30]. For a fair and comprehensive comparison, all methods are evaluated on 100 randomly sampled sequences from the RealEstate10K test set [82], 300 sequences from the Sekai-Real test subset [31], and 500 unseen sequences from the SpatialVID dataset. These samples cover diverse subjects and a wide range of camera trajectories. Reported results are averaged across all sequences. Following ReCamMaster [4], we additionally report frame–text similarity (CLIP-T) and interframe temporal consistency (CLIP-F) [26], together with results on the widely used VBench metrics [21].

Results and Analysis. Quantitative results for cameracontrolled video generation in Tab. 2 compare models

trained on different datasets across different benchmarks, all evaluated with camera pose and text as inputs. The analysis reveals that the model trained on SpatialVID-HQ consistently achieves the highest accuracy in camera control across all three benchmarks, demonstrating the high quality of the camera pose annotations and the strong generalization capability of our dataset. Additionally, models trained on SpatialVID-HQ attain the highest CLIP-T, reflecting enhanced visual fidelity and text-video alignment. It further shows stable improvements across all VBench metrics, particularly in Imaging Quality. Qualitative comparisons in Fig. 6 show that videos generated from models trained on SpatialVID-HQ exhibit greater realism and temporal consistency, benefiting from the dataset’s diverse trajectories, high-quality videos, and consistent text annotations.

###### 5.2. Novel View Synthesis

Baseline. Recent large reconstruction models for novel view synthesis, such as GS-LRM [75] and Long-LRM [83], have demonstrated strong scene-level 3D reconstruction capabilities. We adopt GS-LRM as our baseline framework, which combines a Transformer-based architecture with 3D Gaussian rendering for high-fidelity scene recovery. For fairness, GS-LRM is trained separately on RealEstate10K

- Table 3. Comparison of Original and Fine-tuned Models for Camera Pose Estimation on Sintel [2], TUM-dynamics [48], and Dycheck [16].

Method

Sintel TUM-dynamics Dycheck

ATE↓ RPE trans↓ RPE rot↓ ATE↓ RPE trans↓ RPE rot↓ ATE↓ RPE trans↓ RPE rot↓

CUT3R 0.210 0.070 0.637 0.049 0.015 0.449 0.020 0.011 1.275 CUT3R (Fine-tuned) 0.210 0.069 0.619 0.040 0.013 0.395 0.019 0.011 1.184

VGGT 0.134 0.079 0.501 0.015 0.013 0.352 0.005 0.008 1.087 VGGT (Fine-tuned) 0.148 0.075 0.462 0.013 0.011 0.312 0.005 0.009 1.090

- Table 4. Comparison of GS-LRM across different training datasets on DL3DV [33] and our SpatialVID.

Training Data

DL3DV SpatialVID

PSNR ↑ SSIM ↑ LPIPS ↓ PSNR ↑ SSIM ↑ LPIPS ↓

RE10K 27.01 0.889 0.132 24.13 0.774 0.222 SpatialVID 27.80 0.892 0.116 24.97 0.790 0.203

and a SpatialVID-HQ subset under identical configurations and data volumes. The SpatialVID-HQ subset is curated to match the number of clips in RealEstate10K, prioritizing static scenes for consistency. Each model is finetuned on 8 A6000 GPUs for 3 days and detailed training setup is provided in Supplementary Materials.

Metrics. We follow GS-LRM [75] and evaluate novel-view synthesis using PSNR, SSIM, and LPIPS. Experiments are conducted on 500 DL3DV [33] and 500 unseen SpatialVID sequences, with 2 reference and 4 target views per sequence, rendered at 360 × 640 for a fair comparison.

Results and Analysis. As shown in Tab. 4, the model trained on the SpatialVID subset consistently outperforms RealEstate10K across all the metrics. These results highlight the strong generalization capability of SpatialVID, with superior performance even on the predominantly outdoor DL3DV scenes. This suggests that the diversity and quality of SpatialVID data lead to more robust scene understanding and higher-quality novel view synthesis. Further results are provided in Supplementary Materials.

- 5.3. Geometric Prediction

dynamics [48], and Dycheck [16], all containing non-rigid and dynamic objects to ensure comprehensive coverage of different data types and scene characteristics. Following previous works [57, 74], we report ATE, RPE-rot and RPEtrans computed over estimated camera trajectories.

Results and Analysis. Both CUT3R and VGGT are trained on multiple 3D-annotated datasets and already achieve strong results, though most training data are synthetic. SpatialVID, covering diverse real-world scenes, complements these sources and improves generalization to realistic conditions. As shown in Tab. 3, fine-tuning on SpatialVID yields notable gains for both models on the TUM-dynamics benchmark. On the synthetic Sintel benchmark, both models show improvements, except for a slight ATE regression for VGGT. The Dycheck benchmark, with high dynamics and complex camera trajectories from handheld jitter, remains challenging. After fine-tuning, CUT3R’s performance is marginally improved, while VGGT, near its ceiling, exhibits only minor fluctuations, reflecting its strong pre-trained capability.

###### 6. Discussion and Conclusion

Limitations. Our annotation pipeline inherits failure modes from MegaSaM and can degrade under extreme scenarios (e.g., object-dominated frames, varying focal lengths, and severe radial distortion), constraining its broader application. Additionally, the predicted camera poses exhibit non-metric properties in specific scenarios, and masks derived from motion probabilities yield suboptimal performance in complex scenes. We expect that the development of advanced video pose estimator (e.g., ViPE [20]) would mitigate these issues in the future.

Baselines. For camera pose estimation task, we adopt CUT3R [57] and VGGT [55] as representative baselines, both demonstrating strong performance in geometric and structural reconstruction. Each model is initialized from its official pre-trained weights and fine-tuned under comparable settings. CUT3R is further fine-tuned on SpatialVID to assess pose consistency improvements, while VGGT is trained on its original data both with and without SpatialVID-HQ for fair comparison. Each model is fine-tuned on 8 H20 GPUs for 2 days and detailed training setup is provided in Supplementary Materials.

Conclusion. We introduce SpatialVID, a large-scale video dataset spanning diverse real-world scenes, with tightly aligned semantic and geometric annotations. Our procedural pipeline distills in-the-wild videos into clips annotated with camera motion, depth, and structured, motion-aware scene descriptions. By unifying explicit 3D motion control with rich textual semantics, SpatialVIDprovides strong 3D inductive biases for physically grounded video generation, dynamic scene synthesis, and spatial intelligence. Extensive experiments demonstrate its effectiveness across multiple tasks, establishing a robust foundation for future research.

Metrics. For geometric prediction we focus on camerapose estimation in dynamic scenes. We evaluate models on three diverse benchmarks including Sintel [2], TUM-

###### References

- [1] Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575,

2025. 2, 3

- [2] Sarah Alnegheimish, Dongyu Liu, Carles Sala, Laure BertiEquille, and Kalyan Veeramachaneni. Sintel: A machine learning framework to extract insights from signals. In Proceedings of the 2022 International Conference on Management of Data, pages 1855–1865, 2022. 8
- [3] Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Áron Monszpart, Victor Adrian Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. Map-free visual relocalization: Metric pose relative to a single image. In ECCV, 2022. 3
- [4] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647,

2025. 3, 6, 7, 19

- [5] Philip J. Ball, Jakob Bauer, Frank Belletti, Bethanie Brownfield, Ariel Ephrat, Shlomi Fruchter, Agrim Gupta, Kristian Holsheimer, Aleksander Holynski, Jiri Hron, and et al. Genie 3: A new frontier for world models. 2025. 2, 3
- [6] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471,

2024. 2, 3

- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3
- [8] Yohann Cabon, Naila Murray, and Martin Humenberger. Virtual kitti 2. arXiv preprint arXiv:2001.10773, 2020. 3
- [9] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1050–1060, 2025. 2
- [10] Brandon Castellano. Pyscenedetect. 4
- [11] Tsai-Shien Chen, Aliaksandr Siarohin, Willi Menapace, Ekaterina Deyneka, Hsiang-wei Chao, Byung Eun Jeon, Yuwei Fang, Hsin-Ying Lee, Jian Ren, Ming-Hsuan Yang, et al. Panda-70m: Captioning 70m videos with multiple crossmodality teachers. arXiv preprint arXiv:2402.19479, 2024. 2, 4, 5
- [12] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. arXiv preprint arXiv:2503.24391,

2025. 3

- [13] Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025. 4, 13

- [14] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 3
- [15] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023. 2
- [16] Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, and Angjoo Kanazawa. Monocular dynamic view synthesis: A reality check. Advances in Neural Information Processing Systems, 35:33768–33780, 2022. 8
- [17] Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024. 2, 3, 7, 18
- [18] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2
- [19] Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, and Kai-Wei Chang. 3dllm-mem: Long-term spatial-temporal memory for embodied 3d large language model. arXiv preprint arXiv:2505.22657, 2025. 5
- [20] Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, et al. Vipe: Video pose engine for 3d geometric perception. arXiv preprint arXiv:2508.10934, 2025. 5, 8
- [21] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 7
- [22] Linyi Jin, Richard Tucker, Zhengqi Li, David Fouhey, Noah Snavely, and Aleksander Holynski. Stereo4d: Learning how things move in 3d from internet stereo videos. arXiv preprint arXiv:2412.09621, 2024. 3
- [23] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions, 2024. 4, 5
- [24] Karhan Kayan, Stamatis Alexandropoulos, Rishabh Jain, Yiming Zuo, Erich Liang, and Jia Deng. Princeton365: A diverse dataset with accurate camera pose, 2025. 3
- [25] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 2, 3
- [26] Zhengfei Kuang, Shengqu Cai, Hao He, Yinghao Xu, Hongsheng Li, Leonidas J Guibas, and Gordon Wetzstein. Collaborative video diffusion: Consistent multi-video generation with

- camera control. Advances in Neural Information Processing Systems, 37:16240–16271, 2024. 7
- [27] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024. 2
- [28] Hui Li, Mingwang Xu, Yun Zhan, Shan Mu, Jiaye Li, Kaihui Cheng, Yuxuan Chen, Tan Chen, Mao Ye, Jingdong Wang, et al. Openhumanvid: A large-scale high-quality dataset for enhancing human-centric video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7752–7762, 2025. 5
- [29] Jiaqi Li, Junshu Tang, Zhiyong Xu, Longhuang Wu, Yuan Zhou, Shuai Shao, Tianbao Yu, Zhiguo Cao, and Qinglin Lu. Hunyuan-gamecraft: High-dynamic interactive game video generation with hybrid history condition, 2025. 5
- [30] Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, and Noah Snavely. Megasam: Accurate, fast and robust structure and motion from casual dynamic videos. In arxiv,

2024. 2, 3, 4, 7

- [31] Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, et al. Sekai: A video dataset towards world exploration. arXiv preprint arXiv:2506.15675, 2025. 3, 7, 18
- [32] Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, et al. Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376, 2025. 3, 5
- [33] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learningbased 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160– 22169, 2024. 3, 8
- [34] Jiahao Lu, Tianyu Huang, Peng Li, Zhiyang Dou, Cheng Lin, Zhiming Cui, Zhen Dong, Sai-Kit Yeung, Wenping Wang, and Yuan Liu. Align3r: Aligned monocular depth estimation for dynamic videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 22820–22830,

2025. 2

- [35] Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. You see it, you got it: Learning 3d creation on pose-free videos at scale. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 2016–2029, 2025. 3
- [36] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: A versatile and accurate monocular slam system. IEEE transactions on robotics, 31(5):1147–1163,

2015. 2

- [37] Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-tovideo generation. arXiv preprint arXiv:2407.02371, 2024. 2, 5
- [38] OpenAI. Video generation models as world simulators, 2024. Accessed: 2024-02-15. 2, 3

- [39] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. UniDepthV2: Universal monocular metric depth estimation made simpler, 2025. 5
- [40] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020. 6
- [41] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 5
- [42] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10901–10911, 2021. 2, 3
- [43] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In Proceedings of the IEEE/CVF international conference on computer vision, pages 10912–10922, 2021. 2
- [44] Chris Rockwell, Joseph Tung, Tsung-Yi Lin, Ming-Yu Liu, David F Fouhey, and Chen-Hsuan Lin. Dynamic camera poses and where to find them. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 12444– 12455, 2025. 3
- [45] Johannes L. Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),

2016. 2

- [46] Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 5, 13
- [47] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 4, 13
- [48] Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 573–580. IEEE, 2012. 8
- [49] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, et al. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2446–2454, 2020. 3

- [50] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, and Hao Li. Vidgen-1m: A large-scale dataset for text-to-video generation. arXiv preprint arXiv:2408.02629, 2024. 5
- [51] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 5, 13
- [52] HunyuanWorld Team, Zhenwei Wang, Yuhao Liu, Junta Wu, Zixiao Gu, Haoyuan Wang, Xuhui Zuo, Tianyu Huang, Wenhuan Li, Sheng Zhang, et al. Hunyuanworld 1.0: Generating immersive, explorable, and interactive 3d worlds from words or pixels. arXiv preprint arXiv:2507.21809, 2025. 2, 3
- [53] Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras. Advances in neural information processing systems, 34:16558–16569, 2021. 3, 5, 13
- [54] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 6, 18
- [55] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2, 5, 8, 13, 25
- [56] Qiuheng Wang, Yukai Shi, Jiarong Ou, Rui Chen, Ke Lin, Jiahao Wang, Boyuan Jiang, Haotian Yang, Mingwu Zheng, Xin Tao, et al. Koala-36m: A large-scale video dataset improving consistency between fine-grained conditions and video content. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8428–8437, 2025. 5
- [57] Qianqian Wang*, Yifei Zhang*, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 3, 8, 25
- [58] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697–20709, 2024. 2
- [59] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 2
- [60] Xin Wang, Jiawei Wu, Junkun Chen, Lei Li, Yuan-Fang Wang, and William Yang Wang. Vatex: A large-scale, high-quality multilingual dataset for video-and-language research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4581–4591, 2019. 5
- [61] Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 2, 3

- [62] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality meshes. arXiv preprint arXiv:2404.12385, 2024. 2
- [63] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6, 13
- [64] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935,

2025. 2, 5, 13

- [65] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024. 5
- [66] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 2, 3
- [67] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767–783, 2018. 2
- [68] Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan. Blendedmvs: A largescale dataset for generalized multi-view stereo networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1790–1799, 2020. 2, 3
- [69] Shengming Yin, Chenfei Wu, Jian Liang, Jie Shi, Houqiang Li, Gong Ming, and Nan Duan. Dragnuwa: Fine-grained control in video generation by integrating text, image, and trajectory. arXiv preprint arXiv:2308.08089, 2023. 2, 3
- [70] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141,

2025. 3

- [71] Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325,

2025. 3

- [72] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2, 3
- [73] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. Mvimgnet: A large-scale dataset of multi-view images. In CVPR, 2023. 3
- [74] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and Ming-Hsuan Yang. Monst3r: A simple approach for estimating geometry

in the presence of motion. arXiv preprint arxiv:2410.03825,

2024. 3, 5, 8, 13

- [75] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19. Springer, 2024. 2, 7, 8, 25
- [76] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In European Conference on Computer Vision, pages 20–37. Springer, 2022. 3
- [77] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, and Lijuan Wang. Genxd: Generating any 3d and 4d scenes. arXiv preprint arXiv:2411.02319, 2024. 3
- [78] Guangcong Zheng, Teng Li, Rui Jiang, Yehao Lu, Tao Wu, and Xi Li. Cami2v: Camera-controlled image-to-video diffusion model. arXiv preprint arXiv:2410.15957, 2024. 7
- [79] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In ICCV, 2023. 3
- [80] Li Zhi, Aaron Anne, Katsavounidis Ioannis, Moorthy Anush, and Manohara Megha. Toward a practical perceptual video quality metric, 2016. 4, 13
- [81] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Eric Xin Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. In Proceedings of the IEEE/CVF international conference on computer vision, 2025. 3, 5
- [82] Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018. 2, 3, 7, 18, 25
- [83] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025. 7

#### SpatialVID: A Large-Scale Video Dataset with Spatial Annotations Supplementary Material

###### Overview

This supplementary material provides extended details and additional results complementing the main paper. We elaborate on the data curation pipeline (Sec. A), present in-depth analyses of dataset statistics and semantic properties (Sec. B), and describe additional validation tasks and implementation details (Sec. C). An overview of the video filtering and annotation process is shown in Fig. 7.

###### A. Details of Our Curation Pipeline

Fig. 7 presents the overall data flow of our video curation process, through which the raw videos are progressively refined into the final SpatialVID dataset. The pipeline integrates automatic quality filtering, geometric annotation, and semantic captioning. Figure 8 further illustrates the duration and quantity distribution of the collected raw data, covering diverse indoor and outdoor scenes such as walking, train rides, and drone flights.

- A.1. Score Filtering

To ensure data quality, we apply four complementary filters based on aesthetics, luminance, text content, and motion intensity. These filters remove visually poor or unsuitable clips before any geometric or semantic processing, significantly improving the robustness of downstream pose estimation.

Aesthetic Filtering. To quantitatively assess visual appeal, we use a CLIP + MLP aesthetic score predictor [47]. The model assigns a score from 0 to 10, with higher values indicating better quality. For each video clip, the score is averaged across the first, middle, and last frames. Clips with an average score below 4.0 are considered insufficiently appealing and discarded Fig. 9a).

Luminance Filtering. Luminance is calculated for the first, middle, and last frames using the standard formula L = 0.2126R + 0.7152G + 0.0722B, where R, G, and

###### B are the respective channel values. Clips with average luminance outside the range [20, 140], either too dark or too bright, are excluded, ensuring that only clips with proper exposure are retained (Fig. 9b).

OCR filter. For text detection, we use the latest release of PaddleOCR [13], which offers high accuracy and robust multilingual support. We processe the first, middle, and last frames of each clip to detect text regions, computing the ratio of text area to frame size. Clips where the text area exceeded 30% are removed, as these are considered informational rather than visual (Fig. 9c).

Motion Filtering. We use lightweight VMAF [80],

which is integrated into FFmpeg and outputs a valid motion score ranging from 2.0 to 14.0 (Fig. 9d).

###### A.2. Geometry Pipeline

We employ MegaSaM as our primary geometric reconstruction engine. As shown in Fig. 10, it achieves superior accuracy and robustness compared with DROID-SLAM [53], COLMAP [46], and Fast3R [64], while being faster than MonST3R [74]. Unlike VGGT [55], MegaSaM also maintains stability in feature-sparse scenes. We utilize three trajectory statistics for quality assessment:

Move Distance (MoveDist). Total camera travel distance, computed as the sum of Euclidean distances between consecutive camera centers.

Rotation Angle (RotAngle). Cumulative angular displacement across frames, capturing the extent of viewpoint change.

Trajectory Turns (TrajTurns). Number of significant turns, estimated by counting local extrema in the sequence of orientation angles relative to a start–end reference direction.

###### A.3. Caption Pipeline

The captioning process integrates visual-language reasoning and structured text generation in two stages.

- Stage 1: Visual Parsing. We use Gemini-2.0-flash [51] to analyze sampled frames (1 fps), producing an initial description of camera motion and a summary of scene layout. The prompting format is illustrated in Fig. 11.
- Stage 2: Language Refinement. The outputs, along with calibrated camera poses, are then refined using Qwen3-30BA3B [63]. This stage yields (1) concise scene abstracts, (2) immersive shot-level narratives, and (3) structured semantic tags describing scene type, lighting, weather, crowd density, and time of day. Additionally, Motion Trends labels (e.g., pan, dolly, rotate, steady) summarize camera dynamics. Distributions of these tags are shown in Fig. 14. The prompt template used for the large-language refinement is shown in Fig. 12.

###### A.4. Instruction Examples

Examples of motion instructions are illustrated in Fig. 13. Each instruction corresponds to a specific type of camera movement derived from pose dynamics, providing an interpretable bridge between geometric motion and textual representation. The visualizations demonstrate how camera translations and rotations are systematically mapped to human-readable motion terms, ensuring clarity and consistency across clips.

[Figure 130]

SpatialVID-HQ

SpatialVID

Filtered Clips

Other SpatialVID

Pose Wrong

Clips

Pose Estimation Failed

Aes Filter

Raw

Luminance Filter

OCR Filter

Motion Filter

Broken Videos Title Filter

Size Miss Match

- Figure 7. The data flow of video filtering. Raw videos are first pre-filtered to exclude content with quality defects, incorrect dimensions, or irrelevant titles. The remaining videos are segmented into clips, which are then ranked via a hierarchical scoring strategy integrating aesthetics metrics, luminance, OCR, and motion values. High-scoring clips undergo a dual annotation pipeline to capture both spatial structure and semantic information, yielding the final SpatialVID dataset. This pipeline is also employed to curate a high-quality subset (SpatialVID-HQ) with a more balanced category distribution.

29966 Videos

Video Count

Walk 32.2%

Train 3.8%

House Tour 31.3%

Boat 3%

Drive 15.3%

Roller Coaster 2.5%

Drone 9.8%

Ride 1.7%

Others 0.4%

Others 0.2%

Walk 43.5%

Train 5.8%

Boat 2.1%

Drive 22%

House Tour 12.8%

Roller Coaster 0.3%

Drone 11.4%

Ride 1.8%

19982.653 Hours

Video Duration

- Figure 8. Statistics of pre-filtered videos (aka. the “Clips” in the Fig. 7). The left panel shows the quantity distribution of raw videos, while the right panel presents the duration distribution. These charts illustrate the variety of shooting contexts, including indoor (house tour) and outdoor (walking, train, drone, etc.) scenarios, demonstrating broad coverage of different shooting carriers and environments.

###### B. Details of Dataset Analysis

This section provides extended analyses of the SpatialVID dataset, focusing on semantic composition, caption statistics, and qualitative examples. We examine the distributions of camera motion and scene attributes, analyze caption diversity, and visualize representative samples from the dataset.

###### B.1. Semantic Analysis

Camera Motion Distribution. Fig. 14 shows the distribution of camera motion directions across SpatialVID and its high-quality subset, SpatialVID-HQ. The original dataset contains a wide range of motion types, including forward, lateral, and rotational movements, but their distribution is not well-balanced. In contrast, SpatialVID-HQ displays a

|[Figure 131]<br><br>3.27|
|---|

|[Figure 132]<br><br>3.414.0000<br><br>|
|---|

|[Figure 133]<br><br>4.00|
|---|

|[Figure 134]<br><br>4.21|
|---|

|[Figure 135]<br><br>4.86|
|---|

|[Figure 136]<br><br>5.5155.515<br><br>|
|---|

- (a) Aesthetics Filtering

|[Figure 137]<br><br>8.73|
|---|

|[Figure 138]<br><br>60.08|
|---|

|[Figure 139]<br><br>132.53|
|---|

|[Figure 140]<br><br>210.24|
|---|

|[Figure 141]<br><br>92.59|
|---|

|[Figure 142]<br><br>20.13|
|---|

- (b) Luminance Filtering

|[Figure 143]<br><br>0.74|
|---|

|[Figure 144]<br><br>0.49|
|---|

|[Figure 145]<br><br>0.33|
|---|

|[Figure 146]<br><br>0.22|
|---|

|[Figure 147]<br><br>0.12|
|---|

|[Figure 148]<br><br>0.05|
|---|

(c) OCR Filtering

|[Figure 149]<br><br>[Figure 150]<br><br>[Figure 151]<br><br>[Figure 152]<br><br>[Figure 153]<br><br>[Figure 154]<br><br>0.38|
|---|

|4.084.08<br><br>[Figure 155]<br><br>[Figure 156]<br><br>[Figure 157]<br><br>[Figure 158]<br><br>[Figure 159]<br><br>[Figure 160]|
|---|

|[Figure 161]<br><br>[Figure 162]<br><br>[Figure 163]<br><br>[Figure 164]<br><br>[Figure 165]<br><br>[Figure 166]<br><br>9.02|
|---|

[Figure 167]

|[Figure 168]<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>[Figure 172]<br><br>13.96|
|---|

|[Figure 173]<br><br>[Figure 174]<br><br>[Figure 175]<br><br>[Figure 176]<br><br>[Figure 177]<br><br>[Figure 178]<br><br>19.78|
|---|

(d) Motion Filtering

- Figure 9. Video filtering strategies. Videos are filtered based on various quality criteria (Aesthetics, Luminance, OCR, and Motion). The number in the bottom-right corner of each clip represents its score for the corresponding quality filter. Clips with green boxes are retained, while those with red boxes are discarded due to scores below the threshold.

more balanced distribution, mitigating bias toward any single motion direction and offering improved diversity for motion-conditioned generation or control tasks.

Caption Length and Enrichment. We provide multilevel captions for each video, including motion-oriented descriptions, concise scene summaries, and immersive narratives. To evaluate caption quality, we compare the length distributions of original and enhanced captions (Fig. 15). Both motion and scene captions show notable length increases after refinement, reflecting richer context and more detailed spatial reasoning introduced by our LLM-based generation process.

Hierarchical Scene Tags. Fig. 16 visualizes the struc-

tured semantic attributes extracted from enhanced captions. The sunburst chart summarizes the distribution of five primary attributes—weather, time of day, crowd density, lighting, and scene type. The hierarchical scene-type branch covers fine-grained subcategories such as street, park, interior, and vehicle. The accompanying word cloud, shaped into the SpatialVID logo, highlights the dataset’s emphasis on spatial and motion-oriented vocabulary, with frequently occurring terms like motion, forward, and left.

Multi-Level Caption Design. SpatialVID delivers a versatile caption suite suitable for various research needs: (1) OptCamMotion provides concise, machine-friendly kinematic instructions, reducing average caption length

[Figure 179]

[Figure 180]

COLMAP DROID-SLAM Fast3R MonST3R VGGT MegaSaM

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

- Figure 10. Comparison of MegaSaM with other SLAM/3D reconstruction methods. We visualize the trajectories predicted by six representative methods. The color order ROYGBV corresponds to the progression from the initial to the final time step.

You are given a sequence of video frames in chronological order. Analyze them carefully and generate two distinct captions based on the following instructions:

1. Camera Motion Caption: From the perspective of the camera operator, describe the entire motion trajectory of the camera throughout the clip using precise cinematography terminology (e.g., static, pan, tilt, dolly, handheld, crane, aerial, zoom, etc.). Do NOT assume the camera starts in a "static" position just because it appears stationary in the first frame.Only describe the camera as stationary if there is no visual change across multiple consecutive frames. Instead, focus on changes between frames to infer movement. Describe motion state transitions, not frame-by-frame repetition (e.g., do not say “the camera moves forward again” if it’s continuous). For example:

- - Starting with a dolly forward along a straight path,
- - Then transitioning into a slow right-hand pan,
- - Or shifting from handheld walking movement to a stationary pivot tilt. Include brief environmental context where relevant to clarify direction or intent (e.g., "The camera dollies forward through a narrow alleyway, then smoothly turns left at the intersection"). Keep the final caption concise, between 50–100 words, focused only on motion and its evolution over time.

- 2. Scene Description: Provide a rich, holistic description of the visual content. Include:
- - Main subjects and dynamic objects: who or what is present, and what they are doing (e.g., a cyclist rides past from left to right, a group of people gather near a bench),
- - Background/environment: setting (urban street, forest trail, indoor space), notable landmarks or structures,
- - Lighting and atmosphere: time of day, weather conditions, mood (e.g., golden-hour lighting, overcast sky casting soft shadows, neon-lit nighttime scene),
- - Overall tone or emotion conveyed by the scene.

Avoid focusing on individual frames—describe the general impression and ongoing activity across the entire clip. Aim for around 100 words, balancing detail and conciseness.

Output Format: Do not include any explanations or extra text before or after your response.

Begin directly with:

- 1. Camera Motion Caption: ... followed by
- 2. Scene Description: ...

- Figure 11. Visual-language model (VLM) prompt. The template guides Gemini-2.0-flash to describe both the visual content and coarse camera motion in natural language, enabling structured parsing of dynamic scenes.

from 62.5 to 50.3 words for clean motion supervision. (2) SceneSummary offers compact high-level context with an average of 28.6 words. (3) ShotImmersion integrates

scene semantics and camera motion into rich narratives averaging 89.7 words, supporting reasoning-intensive tasks such as video understanding and story grounding.

You are given a video sequence with camera trajectory data representing the camera's movement through a scene.

- - Key architectural elements
- - Overall atmosphere/style
- - Notable design features Target Length: About 50 words

The data consists of: Camera Motion Caption: A basic description of how the camera moves. Scene Description: A detailed visual summary of the environment. Camera position data: Three lines, representing the sequence of the camera's x-coordinate, y-coordinate, and z-coordinate. These values are derived from normalized 3D pose data using the following formula: poses = np.round(poses / (max_value min_value) / min_abs_value).astype(int); Each value is then multiplied by 1,000,000 and rounded to the nearest integer. Motion intensity: An integer that indicates the level of camera movement, where a value of 0 means the camera is static, 1 indicates slight movement, and 2 or higher represents normal or noticeable motion. In tasks such as Optimized Camera Motion Caption and Main Motion Trend Summary, this intensity value should be used to qualify the degree of motion described — for example, using "slight forward translate" when the intensity is 1.

- 3. Main Motion Trend Summary Summarize the general movement using only 1–3 short motion phrases , depending on how many are clearly present. Focus strictly on major, sustained movements — ignore minor fluctuations or brief directional changes. If only one or two movements dominate, list only those. Use directional translation terms (e.g., forward translate, left translate, upward drift)
- 4. Scene Keywords Extract up to 4 keywords summarizing the key aspects of the scene. Include one term that broadly describes the scene type. Use nouns/noun phrases related to weather, place, time, lighting, scene type. Avoid adjectives/gerunds except for weather. Example: sunset, foggy, marketplace, city street, village
- 5. Immersive Shot Summary Blend Optimized Camera Motion Caption and Scene Description evenly — do not focus more on the camera or the scene alone. Describe the visuals as if someone is watching a moving image unfold. Use descriptive, cinematic language that evokes imagery and emotion. Keep it concise but expressive — suitable for use in scripts, storyboards, or AI video/image generation. Target Length: 50–100 words

Your Tasks:

- 1. Optimized Camera Motion Caption Generate a refined motion caption **from the perspective of the camera itself**, using only the **camera position data** to determine movement direction and dynamics. Use the following rules to interpret motion:

- x increasing: camera moves right

- x decreasing: camera moves left
- y increasing: camera moves down

y decreasing: camera moves up z increasing: camera moves forward

- z decreasing: camera moves backward

Analyze the full trajectory over time to capture acceleration, deceleration, or steady motion. Integrate scene context but prioritize accuracy based on numerical data. Avoid vague phrases like "zoom out" unless it's clearly due to focal length change — here, use translation terms instead. If motion intensity is 0, describe the fixed viewpoint and what the camera observes from that vantage point, incorporating compositional or environmental elements from the original caption. If intensity is 1, reflect subtle movement in the description (e.g., "slight right translate") without exaggerating the motion. For both cases, preserve visual context while aligning with the actual movement level. Avoid mentioning data analysis or detection explicitly — let the description itself reflect the motion state. Target Length: 50–100 words

- 2. Scene Abstract Caption Provide a single-sentence summary that captures:

Given Information: [VQA Captions]

Camera Position Data: [Camera Poses]

Output Format:

- 1. Camera Motion Caption: [From the perspective of the camera holder, with the camera as the subject. Combine camera pose information to describe]
- 2. Scene Abstract Caption: [A concise one sentence summary of the scene]
- 3. Main Motion Trend Summary: [keywords separated by commas, e.g., forward translate, downward tilt]
- 4. Scene Keywords: [word1, word2, word3, ...] (max 5 words)
- 5. Immersive Shot Summary

- Figure 12. Large-language model (LLM) refinement prompt. This instruction template conditions Qwen3-30B-A3B to generate coherent, attribute-rich captions aligned with the extracted spatial and motion cues.

Overall, this structured annotation design ensures both interpretability and flexibility, enabling downstream applications ranging from camera control to multimodal spatial reasoning.

###### B.2. Examples of SpatialVID

We present qualitative examples in Fig. 17. The selected clips illustrate diverse motion trajectories, scene contexts, and annotation richness. Each sample contains synchronized geometry, caption, and metadata, demonstrating the spatial and semantic consistency maintained throughout the dataset.

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

|[Figure 230]|
|---|

|[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|W|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|W|
|---|

|[Figure 238]|
|---|

|[Figure 239]|
|---|

|W|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|W|
|---|

|[Figure 242]|
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

|S|
|---|

|S|
|---|

|S|
|---|

|S|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]|
|---|

|[Figure 255]|
|---|

| |
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|W|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|W|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|W|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|W|
|---|

|[Figure 266]|
|---|

| |
|---|

| |
|---|

| |
|---|

|S|
|---|

|S|
|---|

|S|
|---|

|S|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

|A|
|---|

|D|
|---|

- Figure 13. Examples of motion instructions. Keyboard-style icons denote camera motion directions. The cluster on the left corresponds to translations: W and S indicate forward and backward movement, A and D indicate left and right movement, and ↑, ↓ represent vertical movement. The cluster on the right corresponds to rotations: arrows denote pitch (∧, ∨) and yaw (<, >), while circular arrows indicate roll (⟲, ⟳). These visual cues intuitively describe camera operations, linking geometric motion to semantic labels.

Forward Right Upward + Forward Right + Forward Left + Upward + Forward Static

Downward + Forward + Left

Right + Left + Forward

Upward + Forward + Right

Left + Forward

Side + Forward

Left

Others

SpatialVID

SpatialVID HQ

8.7%

33.3%

18.9%

3% 1.8%

22.4%

2.8%

6.2%

2.5%

13.7%

1.4%

1.4%

3.6% 2.5%

21.7% 6.4% 3.9%

3.8%

19.5%

2%

- 1.6%
- 2.1%

1.7%

12.3%

1.4% 1.4%

- Figure 14. Distribution of camera motion directions. The donut charts show the distribution of camera motion directions for the SpatialVID (left) and HQ SpatialVID (right) datasets. The original SpatialVID dataset exhibits a wide range of motion patterns. In contrast, the HQ SpatialVID dataset features a more balanced distribution, addressing the overrepresentation of any single motion direction.

###### C. Validation Tasks

This section presents the implementation details and qualitative results of validation tasks conducted using the SpatialVID dataset. We focus on three representative paradigms, camera-controlled video generation, spatial reconstruction, and view-consistent rendering, to demonstrate the utility and quality of SpatialVID annotations.

###### C.1. Implementation Details

Camera-Controlled Video Generation. Experiments are conducted on Sekai-Real [31], RealEstate10K [82], and our SpatialVID-HQ dataset. Since RealEstate10K does not in-

clude textual annotations, we adopt the text captions provided by CameraCtrl [17]. For fair comparison, we follow the original train/test split for RealEstate10K, randomly sample 10K training clips from Sekai-Real, and use all high-quality clips from SpatialVID-HQ. For SpatialVIDHQ, the training captions are derived from the Immersive Shot Summary component of our structured annotations. The DiT-based models are initialized from the publicly released TI2V-Wan2.2-5B checkpoint [54], ensuring consistent architecture and capacity across datasets. During training, the camera encoder, projector, and self-attention modules are learnable, while all remaining components are frozen. LIn-

[Figure 267]

[Figure 268]

(a) Motion caption length distribution (b) Scene caption length distribution

- Figure 15. Statistical analysis of the caption data. Fig. (a) and Fig. (b) show the length distributions for motion and scene captions, respectively, comparing the original captions to our enhanced versions. A significant increase in caption length is evident for both types after enhancement.

[Figure 269]

Scene Type

urban

40%

urban

city street

highway

others

desert foresttrail

others valley

mountain road

others kitchen

livingroom

others marina

yachtdeck others

natural landscape

20.5%

15.5%

13%

11%

rural

interior

waterfront

bright

58%

dim/dark

42%

deserted

43%

sparse

29%

crowded

10%

moderate

9.5%

unknown

8.5%

daytime

42%

unknown 16%

night

13%

cloudy

30%

sunny

25%

rainy

15%

unknown

10%

foggy

10%

10%

snowy

dawn morning

12%

dusk evening 17%

Weather

iT

em

f o aD y

Crowd Density

Lighting

(a) Scene tags distribution (b) World cloud

[Figure 270]

- Figure 16. (a) Distribution of scene tags. The sunburst chart shows the distribution of categorical tags across five primary attributes: weather, time of day, crowd density, lighting, and scene type. The scene type attribute is hierarchical, with sub-categories for more detailed classification. The width of each sector reflects the prevalence of the corresponding tag in the dataset. (b) Word cloud. A word cloud shaped into the SpatialVID logo, generated from the enhanced captions. The size of each word corresponds to its frequency in the corpus. Key terms such as motion, forward, left, and right emphasize the dataset’s focus on describing camera movement and spatial dynamics.

put frames are resized to 382 × 480 with a sequence length of 81 frames. We train for 20K steps using a global batch size of 32, the AdamW optimizer with an initial learning

- rate of 1 × 10−5, a cosine decay schedule, and a warmup period of 2K steps. Unless otherwise specified, camera conditioning follows the injection scheme introduced in ReCamMaster [4]. For each frame, the 3×4 camera extrinsic

matrix (12 parameters) is passed through a learnable linear layer cam_encoder ∈ R12×d to project it into the feature dimension d. The resulting embedding is combined with visual tokens through a lightweight per-block projector (Rd×d) initialized as an identity mapping to preserve the pretrained feature scale.

CUT3R Fine-tuning. We follow the official fine-tuning

Video Dynamic Mask Depth Structured Caption Camera Pose

Category Tags: Nature (Mountain Road), Dim, Daytime, Cloudy, Sparse

[Figure 271]

[Figure 272]

[Figure 273]

Motion Trends: forward translate

Scene Summary: A winding mountain road cuts through dense forests and industrial structures, framed by an overcast sky that enhances the quiet, contemplative mood of the scene.

[Figure 274]

Scene Description: A road winds through a mountainous landscape, flanked by dense forests and a large concrete retaining wall. A few cars travel along the road. A building with a crane stands to the side, smoke rising from its chimney. The sky is overcast, casting a muted light over the scene. The overall atmosphere is calm and somewhat industrial, with the natural beauty of the mountains contrasting with the man-made structures. The scene evokes a sense of travel and the integration of industry within a scenic environment.

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

Shot Summary: The camera glides forward along a winding mountain road, revealing dense forests and towering concrete walls. A building with a crane comes into view, its chimney releasing smoke into the muted light. As the shot continues, the road passes under a bridge, blending natural beauty with industrial elements in a serene, flowing journey.

[Figure 281]

[Figure 282]

[Figure 283]

Camera Description: The camera steadily translates forward along a straight path, maintaining a consistent direction and speed. It glides smoothly over the road, passing a building and then emerging beneath a bridge, with the landscape unfolding in a continuous, unbroken motion.

Category Tags: Waterfront (Coastal Church), Bright, Daytime, Sunny, Sparse

[Figure 284]

[Figure 285]

[Figure 286]

Motion Trends: backward translate, upward translate, forward translate

Scene Summary: A serene stone church stands on a cliff, overlooking a turquoise ocean and a quiet town, bathed in warm, natural light that enhances its isolated beauty.

[Figure 287]

Scene Description: A small, stone church with a steeple stands prominently on a grassy cliff overlooking a beach and the ocean. Several people are gathered near the edge of the cliff, enjoying the view. A small town is visible in the background, nestled along the coastline. The scene is bathed in warm, natural light, with the grass and sky appearing in shades of green and blue and the ocean in a turquoise hue. The overall atmosphere is peaceful and serene, highlighting the church's isolated beauty against the backdrop of the sea.

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Shot Summary: The camera drifts back and up, revealing a tranquil stone church perched on a cliff, framed by a turquoise sea and a clear blue sky. The peaceful scene unfolds as the camera pulls away, capturing the quiet majesty of the coastline and the distant town below.

[Figure 294]

[Figure 295]

[Figure 296]

Camera Description: The camera glides backward and upward, steadily translating away from the cliffside church. As it moves, it shifts from a high vantage point to a more distant perspective, maintaining a smooth, continuous motion that emphasizes the vastness of the coastal landscape.

Category Tags: Interior (Car Showroom), Bright, Daytime, Sparse

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

Motion Trends: forward translate

Scene Summary: A sleek, modern car showroom features vibrant BMW models, masked visitors, and digital displays under bright, contemporary lighting, exuding luxury and sophistication.

Scene Description: The scene opens with a vibrant, abstract art display before transitioning into a sleek, modern car showroom. Several BMW vehicles are displayed, including a blue BMW 3 Series, a white BMW 5 Series, and a dark SUV. People are seen walking around, some wearing masks, observing the cars. Large screens display promotional content. The showroom is brightly lit with a warm, contemporary atmosphere. The overall tone is upscale and sophisticated, showcasing luxury automobiles in a inviting environment.

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Shot Summary: The camera glides forward through the polished showroom, revealing a blue BMW 3 Series, a white 5 Series, and a dark SUV, as masked visitors walk among them. Digital screens flash promotional content, and the clean, well-lit space radiates an air of elegance and innovation.

[Figure 309]

[Figure 310]

[Figure 311]

Camera Description: The camera smoothly translates forward through the showroom, maintaining a steady pace as it moves from the abstract display into the luxury space. Slight lateral shifts suggest minor adjustments in framing, but the primary motion is a consistent forward dolly, revealing BMW models and visitors in an upscale environment.

[Figure 312]

Category Tags: Urban (Shopping Street), Dim, Night, Moderate

[Figure 313]

[Figure 314]

[Figure 315]

Motion Trends: forward translate, left translate

Scene Summary: A serene, glass-roofed shopping street in Japan transitions into a narrow, dimly lit alley lined with closed shops, evoking a calm, nocturnal urban atmosphere.

[Figure 316]

Scene Description: The scene depicts a quiet, covered shopping street in Japan, with a few pedestrians walking away from the camera. The street is lined with shops and covered by a glass roof. The camera moves forward, exiting the covered street and entering a narrow street lined with closed shops and restaurants. The lighting is dim, suggesting it is late evening or night. A parked van sits on the right side of the street. The overall tone is calm and peaceful, with a hint of urban solitude.

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

Shot Summary: The camera glides forward through the glass-covered street, revealing a quiet, pedestrian-filled path under a soft, ambient glow. As it exits into the narrow alley, the dim light casts long shadows, capturing the solitude and stillness of the urban landscape.

[Figure 323]

[Figure 324]

[Figure 325]

Camera Description: The camera steadily moves forward through the covered street, gliding smoothly as it exits into the narrow alley. The motion is consistent and unbroken, with no lateral or vertical shifts, maintaining a steady, immersive perspective of the quiet, dimly lit urban environment.

###### Figure 17. Sample videos from SpatialVID. Each example includes synchronized geometry, captions, and spatial annotations. The dataset encompasses diverse environments and camera motions, highlighting its broad coverage and multimodal consistency.

[Figure 326]

The camera glides forward along a mossy path, flanked by towering trees and vibrant ferns. The still water of the pond mirrors the green canopy above, while the soft light bathes the scene in a warm, dreamlike glow, drawing the viewer deeper into the quiet wilderness.

SekaiSekaiSekaiOurs-HQOurs-HQOurs-HQRE10KRE10KRE10KConditionConditionCondition

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

The camera glides forward along a sun-dappled street, its path curving subtly to the right. Colorful buildings rise on either side, their hues softened by the gray sky, while the distant silhouette of a church looms in the background, capturing the peaceful charm of a small town.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

The camera glides forward along the winding track, capturing a serene landscape. Trees and distant structures blur past as the sky shines with a natural blue hue, evoking a quiet, contemplative journey through the countryside.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

###### Figure 18. Different training datasets performance on SpatialVID samples.

[Figure 374]

SekaiSekaiSekaiOurs-HQOurs-HQOurs-HQRE10KRE10KRE10KConditionConditionCondition

an aerial view of a large house with a swimming pool

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

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

a backyard with a wooden deck and patio furniture

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

a dining room table with chairs and a painting on the wall

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

###### Figure 19. Different training datasets performance on RealEstate10K samples.

[Figure 422]

The video begins with a serene and slightly overcast scene of a narrow residential street, likely in an urban area. The street is wet from recent rain, reflecting the surroundings and creating a glossy sheen on the pavement. On either side of the street, there are modern, multi-story buildings with a mix of architectural styles. Some buildings have balconies adorned with greenery, while others feature more minimalist designs. As the camera moves forward, it ···

SekaiSekaiSekaiConditionConditionCondition

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

RE10KRE10KRE10K

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

Ours-HQOurs-HQOurs-HQ

[Figure 438]

The video captures a vibrant and atmospheric night scene in a bustling urban alleyway, likely in Japan, as suggested by the signage and architectural style. The narrow street is lined with various small shops and restaurants, each adorned with colorful neon signs and lanterns that illuminate the area with a warm, inviting glow. As the camera moves forward, it follows a person riding a bicycle down the alley. The cyclist is dressed in dark clothing and appears to be ...

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

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

The video captures a bustling urban street scene from a first-person perspective, likely taken during the evening as the sky is dim and artificial lights illuminate the surroundings. The viewer is walking along a sidewalk that is lined with tall buildings, some of which have illuminated windows, suggesting they are office or residential spaces. On the left side of the frame, there is a yellow delivery truck parked on the street, partially obstructing the view. Behind it, a red truck ...

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

Sekai

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

RE10K

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

Ours-HQ

###### Figure 20. Different training datasets performance on Sekai-Real samples.

|Input Images|GT & Outputs<br><br>|
|---|---|
|[Figure 470]<br><br>[Figure 471]<br><br>|[Figure 472]<br><br>[Figure 473]<br><br>[Figure 474]<br><br>[Figure 475]<br><br>[Figure 476]<br><br>[Figure 477]<br><br>[Figure 478]<br><br>[Figure 479]<br><br>[Figure 480]<br><br>[Figure 481]<br><br>[Figure 482]<br><br>[Figure 483]<br><br>GTR10LKSVID|
|[Figure 484]<br><br>[Figure 485]<br><br>|[Figure 486]<br><br>[Figure 487]<br><br>[Figure 488]<br><br>[Figure 489]<br><br>[Figure 490]<br><br>[Figure 491]<br><br>[Figure 492]<br><br>[Figure 493]<br><br>[Figure 494]<br><br>[Figure 495]<br><br>[Figure 496]<br><br>[Figure 497]<br><br>GTR10LKSVID|
|[Figure 498]<br><br>[Figure 499]|[Figure 500]<br><br>[Figure 501]<br><br>[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]<br><br>[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>[Figure 512]<br><br>GTR10LKSVID|
|[Figure 513]<br><br>[Figure 514]<br><br>|[Figure 515]<br><br>[Figure 516]<br><br>[Figure 517]<br><br>[Figure 518]<br><br>[Figure 519]<br><br>[Figure 520]<br><br>[Figure 521]<br><br>[Figure 522]<br><br>[Figure 523]<br><br>GTR10LKSVID|

##### GTR10LKSVID

[Figure 524]

[Figure 525]

Figure 21. GS-LRM qualitive comparison on SpatialVID.

protocol of CUT3R [57] and initialize from the publicly released cut3r_512_dpt_4_64 checkpoint. Training is performed with a global batch size of 64 using the AdamW optimizer with a learning rate of 1 × 10−6, a weight decay of 0.05, and a total of 6,500 iterations. We fine-tune on long video sequences ranging from 4 to 64 frames, with each frame resized such that the longer side does not exceed 512 pixels. During training, the encoder is kept frozen while the decoder and output heads are updated to better align CUT3R’s geometric predictions with the spatially consistent captions and camera poses provided by SpatialVID.

VGGT Fine-tuning. We fine-tune VGGT [55] on SpatialVID-HQ together with most of its original training data. For fair comparison, an additional model is fine-tuned using the same original data without SpatialVID-HQ. We initialize from the publicly released VGGT checkpoint and follow the original training strategy, keeping the DINO backbone frozen to ensure consistent architecture and capacity. Training is performed for 40K steps, using the AdamW optimizer with an initial learning rate of 2 × 10−4. Unless otherwise specified, all remaining hyperparameters follow the default VGGT configuration.

Large Reconstruction Models. All experiments are conducted on RealEstate10K [82] and our SpatialVIDHQ dataset. For fair comparison, both datasets are trained with an equal amount of 60K video clips. In each epoch, random image pairs are sampled, using two views as input and four intermediate views as supervision. All models are trained from scratch to ensure consistent architecture and capacity. Following the GS-LRM [75] training protocol, we adopt a two-stage schedule: the first stage trains at a resolution of 180 × 320 for 15K steps, followed by a highresolution stage at 360 × 640 for an additional 45K steps, totaling 60K steps. Training is performed with a global batch size of 32 using the AdamW optimizer, an initial learning

- rate of 2 × 10−5, a cosine decay schedule, and 2K warmup steps. Unless otherwise specified, the total loss combines pixel-level, perceptual, and depth-smoothness objectives: Ltotal = λ1Lmse + λ2Llpips + λ3Lreg, where λ1 = 1.0, λ2 = 0.5, and λ3 = 0.25. The regularization term Lreg encourages depth smoothness by penalizing abrupt depth discontinuities.

###### C.2. Qualitative Results

We provide additional qualitative comparisons of Cameracontrolled video generation and Novel View Synthesis. The camera-controlled video generation results (Fig. 18, Fig. 19, Fig. 20) show that the model trained on SpatialVID-HQ precisely follows complex camera trajectories while maintaining realistic spatial continuity and dynamic visual coherence. Moreover, the model demonstrates improved prompt understanding, enabling the generation of more accurate and visually convincing environmental details such as trees and

decorations. The novel view synthesis results ( Fig. 21) highlight how SpatialVID supports robust geometry learning, maintaining consistent spatial layouts and detailed texture synthesis across diverse motion trajectories.

By integrating explicit 3D motion control with rich textual semantics, SpatialVIDendows physically grounded video generation, dynamic scene synthesis, and spatial intelligence tasks with robust 3D inductive biases. Comprehensive experimental results validate SpatialVID’s effectiveness across diverse tasks, laying a solid foundation for research in the field of spatial intelligence.

