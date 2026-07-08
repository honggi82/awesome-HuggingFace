## PE3R: Perception-Efficient 3D Reconstruction

Jie Hu, Shizun Wang, Xinchao Wang* xML Lab, National University of Singapore

# arXiv:2503.07507v2[cs.CV]26Mar2026

### Abstract

Recent advances in 2D-to-3D perception have enabled the recovery of 3D scene semantics from unposed images. However, prevailing methods often suffer from limited generalization, reliance on per-scene optimization, and semantic inconsistencies across viewpoints. To address these limitations, we introduce PE3R, a tuning-free framework for efficient and generalizable 3D semantic reconstruction. By integrating multi-view geometry with 2D semantic priors in a feed-forward pipeline, PE3R achieves zero-shot generalization across diverse scenes and object categories without any scene-specific fine-tuning. Extensive evaluations on openvocabulary segmentation and multi-view depth estimation show that PE3R not only achieves up to 9× faster inference but also sets new state-of-the-art accuracy in both semantic and geometric metrics. Our approach paves the way for scalable, language-driven 3D scene understanding. Code is available at github.com/hujiecpp/PE3R.

### 1. Introduction

While machine vision has achieved remarkable success in 2D perception tasks, human perception is inherently threedimensional, seamlessly integrating multiple viewpoints to construct a coherent understanding of the world [2, 43, 54]. This fundamental gap motivates a central challenge in computer vision: can we harness powerful 2D perception models to reconstruct and comprehend 3D scenes without relying on 3D supervision? The problem is twofold: it requires not only estimating accurate geometry from unposed images but also lifting 2D semantic information into a consistent 3D representation.

Recent methods have made significant strides in 2Dto-3D perception, recovering both geometry and semantics from unposed multi-view images [5, 6, 15, 17, 18, 20, 23, 27, 32, 33, 45–47, 56, 62]. Despite this progress, these approaches often face a trilemma: they struggle with scene generalization, fail to maintain semantic consistency across views, and suffer from high computational costs. For instance, Neural Radiance Fields (NeRF) [29] and 3D Gaus-

*Corresponding author: xinchao@nus.edu.sg

sian Splatting (3DGS) [19] can produce high-quality reconstructions but typically require per-scene training and involve heavy semantic lifting, making them impractical for real-time or large-scale applications.

To overcome these limitations, we introduce PE3R, a tuning-free framework for efficient, accurate, and generalizable 3D semantic reconstruction. Inspired by recent advances in feed-forward geometry estimation [24, 50, 52], PE3R employs a feed-forward pipeline that eliminates scene-specific fine-tuning while enabling rapid inference. As illustrated in Fig. 1, our framework integrates three carefully designed modules: (1) Pixel Embedding Disambiguation ensures semantic consistency across views and object hierarchies; (2) Semantic Point Cloud Reconstruction fuses geometry with dense 2D semantics to refine 3D structure; and (3) Global View Perception enables open-vocabulary object retrieval through language grounding. This cohesive design empowers robust zero-shot generalization to previously unseen environments.

We conduct extensive evaluation across diverse benchmarks, including Mip-NeRF360 [3], Replica [44], ScanNet++ [57] for open-vocabulary segmentation, and KITTI [14], ScanNet [9], DTU [1], ETH3D [39], Tanks and Temples [22] for multi-view depth estimation. Results demonstrate that PE3R achieves up to 9× faster 3D semantic reconstruction than prior methods while advancing in both semantic fidelity and geometric precision.

Our main contributions are summarized as follows:

- • We propose PE3R, a fast and scalable framework for 3D semantic reconstruction that operates directly on unposed 2D images, requiring no 3D supervision, per-scene finetuning, or camera parameters.
- • We introduce three novel modules that address key challenges in 2D-to-3D perception: pixel embedding disambiguation for cross-view consistency, semantic point cloud reconstruction for geometry-semantics fusion, and global view perception for open-vocabulary interaction.
- • Through comprehensive evaluations on multiple benchmarks, we demonstrate state-of-the-art performance, effective zero-shot generalization, and practical scalability. Code is provided to ensure reproducibility.

2D Images 3D Perception

2D Images 3D Perception

2D Images 3D Perception

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

Shelf Bottle with blue logo

Blue mailbox

Signage

Big Ben Houses of Parliament

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

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

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

Eyeglasses Desk lamp

Black seat

White cars

Westminster Hall Christmas tree

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

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

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

[Figure 87]

Scissors Frying pan

Table with white legs White sofa

Sydney opera house Bay

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

[Figure 120]

Counting cup Washing machine

Bicycle Blue box

Ship Skyscrapers

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

Nespresso Android doll

Flowers Statue

Leaning Tower

Dome Baptistery

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

Paper box Telephone

Trees Bench

Cathedral

Glass lawns

(a) Indoor Scenes

(b) Outdoor Scenes

(c) Wild Scenes

- Figure 1. Examples of Perception-Efficient 3D Reconstruction. Our framework reconstructs 3D semantic scenes from unposed RGB images and supports open-vocabulary interaction, without any training or per-scene optimization. It is efficient in two critical ways: (1) Input efficiency: no depth maps or camera parameters are needed; (2) Inference efficiency: significant speed gains on semantic reconstruction compared to prior pipelines. These properties make PE3R practical for real-world deployment in large-scale and time-sensitive scenarios.

### 2. Related Work

- 2D-to-3D Reconstruction. Significant progress has been made in reconstructing 3D scenes from 2D images, with many methods achieving high surface fidelity and geometric coherence. A number of approaches leverage signed distance functions (SDFs) combined with volume rendering for accurate mesh extraction [13, 16, 28, 51, 53]. Neural Radiance Fields (NeRF) [29] popularized the use of implicit neural representations for view synthesis, while 3D Gaussian Splatting (3DGS) [19] later introduced an explicit, point-based alternative that enables highly efficient rendering. To overcome the dependency on known camera parameters, several recent works operate directly on unposed images. Among them, DUSt3R [52] predicts 3D structure from image sets. MASt3R [24] extends this direction by incorporating keypoint correspondences, and MASt3RSfM [10] further unifies structure-from-motion and multiview stereo within a transformer framework. Hybrid techniques such as Vggsfm [49] combine traditional geometry with learned features to improve robustness. Scalable systems like Spann3R [48], FASt3R [55], and VGGT [50] integrate attention mechanisms or visibility reasoning to enhance multi-view consistency and computational efficiency. While these methods achieve impressive geometric results, they typically lack integrated semantic understanding or require task-specific adaptation. PE3R builds upon these geometric foundations but moves beyond them by incorporating semantic priors, enabling efficient and context-aware
- 3D scene understanding without per-scene training. 2D-to-3D Perception. A parallel line of research seeks to lift 2D semantic information into 3D reconstructions. Early works such as GNeRF [7] and InPlace [61] embed-

ded 2D masks into NeRF representations to enable 3Daware segmentation with 2D supervision. Subsequent efforts extended NeRF to support 3D instance segmentation, using either supervised [4, 26, 42] or unsupervised [30, 58] learning paradigms. The introduction of the Segment Anything Model (SAM) [21] inspired a range of SAMguided 3D lifting pipelines, including SA3D [5] and Feature3DGS [62], which incorporate prompt-driven segmentation into 3D scene representations. Several other methods directly project SAM masks into 3D space, such as SAGA [6], Gaussian Grouping [56], SAGS [18], ClickGaussian [8], and FlashSplat [41]. The advent of visionlanguage models has further enabled open-vocabulary 3D segmentation. For example, LangSplat [33] aligns CLIP embeddings with 3D Gaussians, while GOI [34] enforces multi-view consistency through text-image alignment. The Large Spatial Model (LSM) [12] performs open-vocabulary semantic segmentation by lifting features from a pre-trained 2D model (LSeg) and aggregating them into a unified 3D representation. A fundamental limitation shared by most of these methods is their reliance on scene-specific training, iterative semantic optimization, or manual heuristics, which constrains their generalization and practical efficiency. In contrast, PE3R supports scalable, zero-shot 3D semantic reconstruction through a cohesive feed-forward pipeline that jointly optimizes geometry and semantics, eliminating the need for 3D supervision or per-scene training.

2D Foundational Models. The advancement of large-scale pre-trained models has been pivotal for generalizable visual perception. CLIP [35] established a shared imagetext embedding space through contrastive learning, with SigLIP [59] offering a more efficient variant. SAM [21] enables promptable segmentation of arbitrary image re-

[Figure 181]

[Figure 182]

Pixel Embedding Disambiguation Global View Perception Television

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Text Queries

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Segmentation

Image Encoder

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Text Encoder

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

Table

[Figure 205]

Multi-Level Identity

|[Figure 206]<br><br>[Figure 207]<br><br>Area-Moving Aggregation| |
|---|---|
| | |

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

Text Embeddings

[Figure 213]

[Figure 214]

Tracking

[Figure 215]

[Figure 216]

|[Figure 217]<br><br>[Figure 218]<br><br>Global Similarity Normalization|
|---|

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

Slippers

[Figure 224]

Pixel Embeddings

Cross-View Consistency

Point Embeddings

Semantic Field Reconstruction

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

Keyboard

[Figure 232]

[Figure 233]

[Figure 234]

|[Figure 235]<br><br>[Figure 236]<br><br>Semantic-Guided Refinement| |
|---|---|
| | |

Feed-Forward Prediction

2D Images

3D Point Cloud

3D Semantic Field

3D Perception

- Figure 2. The PE3R Framework. Our pipeline comprises three core stages: (1) Pixel Embedding Disambiguation: Images are decomposed into hierarchical masks (via SAM/SAM2), which are encoded by CLIP and aggregated to yield consistent per-pixel embeddings. (2) Semantic Point Cloud Reconstruction: A feed-forward model (DUSt3R) generates 3D point clouds, which are then refined using semantic cues. (3) Global View Perception: Text embeddings are compared against the 3D point features via global similarity normalization, enabling open-vocabulary semantic localization.

gions, with SAM2 [36] enhancing cross-view and temporal consistency. DINOv2 [31] delivers high-quality visual features through self-supervised learning, and Grounding DINO [25] facilitates open-set object detection using language-based supervision. PE3R strategically leverages these foundational models not merely as feature extractors, but as core components within a novel pipeline designed to generate robust, language-grounded 3D semantic representations, all without requiring any 3D model fine-tuning.

- 3. Method

challenge of 3D semantic reconstruction from unposed images. (1) Pixel Embedding Disambiguation: This stage focuses on deriving view-consistent and semantically coherent embeddings from the input images. Each image is first decomposed into hierarchical object masks using SAM [21] or SAM2 [36]. These masks are encoded by a vision-language model (e.g., CLIP [35]) into a shared semantic space. The core of this module lies in an areaweighted spherical interpolation strategy that aggregates region features across different views and hierarchical levels. This process resolves semantic ambiguities caused by occlusions or nested objects, yielding dense, per-pixel semantic embeddings that are consistent across images. (2) Semantic Point Cloud Reconstruction: Building upon the disambiguated 2D embeddings, this stage transitions to 3D geometric reconstruction. A feed-forward model, such as DUSt3R [52], is leveraged to generate an initial dense 3D point cloud from the multi-view images. The key innovation here is the semantic-guided refinement of this raw geometry. The point cloud is refined by leveraging the previously computed semantic embeddings to filter spatial outliers and enhance coherence, resulting in a unified reconstruction that is both geometrically accurate and semantically labeled. (3) Global View Perception: The final stage enables open-vocabulary interaction with the reconstructed 3D scene. A user’s natural language query is first encoded into a textual embedding using CLIP’s text encoder. This query embedding is then compared against the semantic features of every 3D point via cosine similarity. To ensure a consistent and interpretable retrieval process across all views, a global min-max normalization is applied to the similarity scores. 3D points whose normalized similarity scores exceed a predefined threshold are retrieved and

#### 3.1. Problem Formulation

We address the challenging problem of reconstructing a semantically rich 3D scene representation from a collection of 2D images captured in unconstrained, in-the-wild settings. The core difficulty stems from the absence of any geometric supervision of semantics. To bridge this gap, we aim to reconstruct a 3D semantic point cloud directly from a set of unposed 2D images, capturing both the spatial structure and high-level semantic content of the scene. Our framework is designed to operate without any 3D semantic annotations, camera intrinsics, or extrinsics, ensuring scalability and practicality in unconstrained environments. The reconstructed scene supports natural language interaction, allowing users to issue open-ended text queries (e.g., “black chair”, “white bottle”) to retrieve and localize corresponding semantic entities in 3D, thereby enabling interactive and human-aligned perception.

#### 3.2. PE3R Framework

As illustrated in Fig. 2, the PE3R pipeline is architected around three cohesive stages that collectively address the

precisely localized, thereby facilitating real-time, languagegrounded understanding and exploration of the scene.

#### 3.3. Pixel Embedding Disambiguation

Establishing a consistent semantic embedding space across multiple views is foundational for robust 3D reconstruction. This process, however, is inherently challenged by two types of ambiguity: (i) object-level ambiguity, where overlapping or nested structures (e.g., a cup on a table) create uncertain semantic associations; and (ii) viewpoint-level ambiguity, where occlusions or perspective changes lead to inconsistent semantics across different viewpoints. To resolve these issues, we construct dense per-pixel embeddings that are both discriminative at the object level and consistent across all views. In our formulation, an “object” is defined as a collection of hierarchically related masks with substantial spatial overlap. Specifically, a smaller mask is grouped into a larger object if their Intersection over Union (IoU) exceeds a threshold of 0.9. This simple yet effective heuristic allows us to associate fine-grained parts (e.g., chair legs) with their corresponding whole-object regions, without requiring any external supervision or predefined labels.

Image Embedding Extraction. Given n input images X1,...,Xn ∈ R3×H×W, we employ SAM [21] and SAM2 [36] to decompose each image into a hierarchy of object masks with consistent object indices across views. This yields a collection of masked image regions M1,...,Mn ∈ Rm×3×H×W, where m is the number of masks per image. A vision-language encoder Fimg(·) is used to extract each masked region into the semantic space:

F1,F2,...,Fn = N(Fimg(M1,M2,...,Mn)), (1)

where Fi ∈ Rm×d denotes the L2-normalized (N(·)) embedding matrix for the i-th image.

Area-Weighted Interpolation. To resolve semantic ambiguities and align features across both views and hierarchy levels, we introduce a geometry-aware aggregation strategy. Our intuition is that masks with larger visible areas typically provide more stable and reliable semantic features, and thus should exert a greater influence on the aggregated representation. Given two unit embeddings FA and FB with corresponding areas areaA and areaB, we define the area ratio t = area

areaA+areaB and compute the aggregated embedding as:

B

sin((1 − t)θ) sin(θ)

FˆB = aFA + bFB,a =

,b =

sin(tθ) sin(θ)

, (2)

where θ is the angle between FA and FB. This formulation possesses two key properties that underpin its effectiveness:

Proposition 1. Vector Normalization: The interpolated vector FˆB preserves unit norm, ensuring it remains within the original semantic embedding space.

Dataset Method mIoU mPA mP

LERF [20] 0.2698 0.8183 0.6553 F-3DGS [62] 0.3889 0.8279 0.7085 GS Grouping [56] 0.4410 0.7586 0.7611 LangSplat [33] 0.5545 0.8071 0.8600

Mip.

OpenNeRF [11] 0.5734 0.8345 0.8959 LSM [12] 0.6254 0.8887 0.9192 GOI [34] 0.8646 0.9569 0.9362

##### PE3R, ours 0.8951 0.9617 0.9726

LERF [20] 0.2815 0.7071 0.6602 F-3DGS [62] 0.4480 0.7901 0.7310 GS Grouping [56] 0.4170 0.7370 0.7276 LangSplat [33] 0.4703 0.7694 0.7604

Rep.

OpenNeRF [11] 0.5014 0.7915 0.7831 LSM [12] 0.5614 0.8113 0.7961 GOI [34] 0.6169 0.8367 0.8088

##### PE3R, ours 0.6531 0.8377 0.8444

Table 1. 2D-to-3D Open-Vocabulary Segmentation results on small-scale Mip-NeRF360 (Mip.) and Replica (Rep.) datasets.

Proposition 2. Semantic Guidance: If FA has a higher similarity to a reference semantic vector FC than FB does, then FˆB will also exhibit a higher similarity to FC than FB does. This steers the aggregated representation toward more semantically meaningful directions.

These properties guarantee that our interpolation strategy is both geometrically sound and semantically informative.

Pixel Embedding Ensemble. To enhance semantic coherence, we perform a two-stage ensemble that operates both within and across views. (1) Within-view aggregation: For each image, object masks are processed in descending order of area. The embeddings of smaller masks (likely object parts) are iteratively aggregated via area-weighted interpolation with those of larger masks (likely whole objects), promoting semantic consistency across the hierarchical part-whole structure within a single view. (2) Crossview aggregation: We seed the SAM2 tracker on the first view using SAM (segment-everything) masks; for each subsequent view, SAM2 propagates tracked masks and SAM proposes all masks again. Any SAM mask with IoU<0.1 to all current SAM2 masks is treated as newly discovered/redetected and inserted as a new tracking prompt (improving recall under tracking loss). When association is unreliable, we skip cross-view fusion and fall back to withinview disambiguation. This two-level ensemble produces a set of stable and semantically coherent object embeddings that are consistent across both spatial hierarchies and multiple views. The final object-level embeddings are then projected back into the pixel space, yielding dense per-pixel semantic maps E1,E2,...,En ∈ RH×W×d.

#### 3.4. Semantic Point Cloud Reconstruction

Building upon the disambiguated pixel embeddings, this module focuses on reconstructing a high-quality 3D semantic point cloud. We leverage recent feed-forward 3D pointmap predictors, such as DUSt3R [52], to directly estimate per-pixel 3D coordinates (or pointmaps) from the

Method Preprocess Training Total LERF [20] 3mins 40mins 43mins F-3DGS [62] 25mins 623mins 648mins GS Grouping [56] 27mins 138mins 165mins

LangSplat [33] 50mins 99mins 149mins

GOI [34] 8mins 37mins 45mins PE3R, ours 5mins - 5mins

- Table 2. Runtime Comparison for 3D Semantic Reconstruction on Mipnerf360.

Method mIoU mPA mP

LERF [20] Embedding 0.1824 0.6024 0.5873 GOI [34] Embedding 0.2101 0.6216 0.6013 LSM [12] Embedding 0.2124 0.6346 0.6114

PE3R Embedding (ours) 0.2248 0.6542 0.6315

- Table 3. 2D-to-3D Open-Vocabulary Segmentation results on the large-scale ScanNet++ dataset.

multi-view images:

P1,P2,...,Pn = Fpts(X1,X2,...,Xn), (3)

where each Pi ∈ RH×W×3 represents the 3D coordinates (x,y,z) for every pixel in the i-th view. While this feedforward paradigm is fast, the initial pointmaps are often contaminated by spatial noise due to visual artifacts like occlusions, reflections, and transparent surfaces. To overcome this inherent limitation, we introduce a semantic-guided refinement process that utilizes the consistent embeddings to enforce both geometric and semantic coherence.

Anomaly Point Detection. The core premise of our detection strategy is that pixels belonging to the semantic region should form a spatially coherent cluster in 3D. We thus identify outliers by evaluating the local 3D consistency within semantically homogeneous areas. For each pixel Pi,j, we compute its average Euclidean distance to neighboring pixels within a k × k window that share the same semantic label:

Li,j = dx,dy I(Mi,j,Mi+dx,j+dy) · D(Pi,j,Pi+dx,j+dy) dx,dy I(Mi,j,Mi+dx,j+dy)

,

(4) where dx,dy ∈ [−⌊k/2⌋,⌊k/2⌋], I(·,·) is an indicator function that evaluates to 1 if two pixels share the same instance-track index, and D(·,·) denotes the Euclidean distance in 3D space. This operation yields a semantic-aware distance map Li ∈ RH×W for each view. Pixels associated with distance values exceeding a predefined threshold after normalization are flagged as spatial anomalies and subsequently excluded, resulting in a significantly cleaner intermediate point cloud.

Semantic-Guided Refinement. Merely filtering outliers, however, does not actively correct the underlying erroneous estimates. Instead of applying computationally expensive post-hoc geometric regularization (e.g., least-squares fit-

ting), we propose an efficient image-space smoothing strategy that tackles the problem at its root. For each detected anomaly, we adjust its RGB value y by blending it with the average color x of its surrounding semantic region:

yˆ = α · x + (1 − α) · y. (5)

Here, the blending factor α ∈ [0,1] controls the strength of semantic smoothing, effectively suppressing textural noise that misleads the geometric predictor while preserving genuine structural edges. The smoothed image is then fed back into the pointmap predictor Fpts, yielding a refined 3D estimate that is more spatially coherent and better aligned with the semantic structure of the scene. Finally, a global alignment step synchronizes all refined pointmaps Pˆi into a unified coordinate frame, ensuring multi-view consistency. The output is a fused semantic point cloud where each point is associated with a 3D coordinate and its corresponding semantic embedding, forming a robust foundation for language-grounded interaction.

#### 3.5. Global View Perception

The final stage of PE3R leverages the unified semantic point cloud to enable intuitive, open-vocabulary interaction with the reconstructed 3D semantic scene. Given a user-provided text query, we first encode it into a semantic embedding using a language encoder Ftxt(·) (e.g., CLIP [35]):

T = Ftxt(text), (6)

where T ∈ Rd resides in the shared image-language embedding space. To identify corresponding regions in 3D, we compute the cosine similarity between the query embedding T and the dense per-pixel semantic embeddings E1,E2,...,En from all input views:

[S1,S2,...,Sn] = D(T,[E1,E2,...,En]), (7)

where D(·) denotes the cosine similarity function. To ensure consistency in retrieval scores across different viewpoints, we apply a global min–max normalization to all similarity values. Specifically, we concatenate the similarity scores from every view and scale them globally into the range [0,1]. This provides a unified and interpretable similarity measure that is invariant to view-specific variations. A predefined threshold is then applied to the normalized scores to select the most relevant 3D points from the corresponding refined pointmaps P1,P2,...,Pn. Points whose normalized similarity scores exceed this threshold are considered semantically aligned with the text query and are retained for 3D localization. This approach enables natural language-based interaction with the reconstructed 3D scene, supporting flexible object retrieval and semantic scene understanding without the need for predefined category labels or model retraining.

|KITTI|ScanNet<br><br>|ETH3D|DTU<br><br>|T&T|Ave.|
|---|---|---|---|---|---|
|rel↓ τ ↑|rel↓ τ ↑<br><br>|rel↓ τ ↑<br><br>|rel↓ τ ↑<br><br>|rel↓ τ ↑|rel↓ τ ↑|

Methods

- (a)

COLMAP [37] 12.0 58.2 14.6 34.2 16.4 55.1 0.7 96.5 2.7 95.0 9.3 67.8 COLMAP Dense [38] 26.9 52.7 38.0 22.5 89.8 23.2 20.8 69.3 25.7 76.4 40.2 48.8

- (b)

DUSt3R [52] 9.1 39.5 4.9 60.2 2.9 76.9 3.5 69.3 3.2 76.7 4.7 64.5 Fast3R [55] - - 6.3 50.3 4.7 62.7 3.9 62.6 4.4 64.0 - -

DUSt3R [52]† 11.0 33.2 4.8 60.3 3.1 74.5 2.7 75.7 2.9 78.5 4.9 64.4 MASt3R [24]† 36.9 5.4 22.0 9.6 27.9 9.9 13.6 13.7 22.1 14.6 24.5 10.6 VGGT [50] 9.4 41.3 4.4 65.0 1.8 86.3 1.1 94.3 2.1 85.2 3.8 74.4

- (c)

PE3R, with DUSt3R 9.4 48.6 5.5 55.1 2.3 82.0 3.2 69.1 2.1 85.3 4.5 68.0 PE3R, with VGGT 9.2 43.8 4.3 65.8 1.7 86.6 1.1 94.4 2.0 85.4 3.7 75.2

- Table 4. Multi-view Depth Estimation on RobustMVD. Comparison under: (a) classical methods, (b) feed-forward methods, and (c) semantic reconstruction. PE3R achieves the best average performance, demonstrating that semantic guidance generally enhances geometric accuracy. † indicates our implementation. Metrics are indicated by relative error (rel ↓) and inlier ratio (τ ↑).

### 4. Experiments

#### 4.1. Experimental Details

Datasets and Tasks. We evaluate PE3R on two core tasks to comprehensively assess its capabilities: open-vocabulary 2D-to-3D segmentation and multi-view depth estimation. For open-vocabulary segmentation, we employ three benchmarks: Mip-NeRF360 [3] and Replica [44] are used to evaluate performance in relatively compact, object-centric scenes, using the open-vocabulary labels from GOI [34]. ScanNet++ [57] is adopted to test generalization in largescale, complex indoor environments with diverse layouts and object categories. For multi-view depth estimation, we follow established benchmarks to assess geometric accuracy across a wide range of scenarios. The evaluated datasets include KITTI [14] (outdoor driving), ScanNet [9] (indoor scenes), DTU [1] (controlled object-level scans), ETH3D [39] (high-resolution scenes), and Tanks and Temples (T&T) [22] (large-scale outdoor). This selection ensures a comprehensive evaluation across both indoor and outdoor environments.

Implementation Details. PE3R is built upon several powerful, off-the-shelf models. We employ MobileSAMv2 [60] for efficient image segmentation, SAM2 [36] for obtaining temporally consistent mask tracks, and DUSt3R [52] as our core feed-forward 3D pointmap predictor. We use SAM for per-image “segment-everything” masks, and SAM2 for instance tracking. Concretely, we initialize SAM2 on the first image by running SAM and using the resulting masks as prompts to seed the SAM2 tracker. For each subsequent image, we (i) run SAM2 to propagate tracked masks, then (ii) run SAM to re-detect all objects; any SAM mask with IoU<0.1 to all current SAM2 masks is treated as newly discovered/re-detected and added as a new instance to the tracker, recovering missed/lost tracks under large viewpoint changes. All experiments were conducted on a single server equipped with an NVIDIA A100 GPU and an Intel i7 CPU. Evaluation Protocol. Our evaluation protocol is designed to be fair, reproducible, and aligned with standard prac-

tices in the field. For open-vocabulary segmentation, we report the standard metrics of mean Intersection over Union (mIoU), mean Pixel Accuracy (mPA), and mean Precision (mP). The evaluation procedure is tailored to dataset characteristics to ensure optimal and fair benchmarking. On MipNeRF360 and Replica, we follow the protocol established by GOI [34]. We perform a holistic 3D semantic reconstruction of the entire scene. Open-vocabulary queries, labeled by GOI, are then executed in this unified 3D space. The resulting 3D segmentation is projected back onto the original 2D image planes for quantitative comparison against the

- 2D ground-truth masks. On the larger ScanNet++ dataset, we process data in snippets of four consecutive frames to reduce the computational load, a common practice for scaling 3D methods to large scenes. This approach effectively highlights the benefit of integrating semantic features into
- 3D reconstruction. For a direct and fair comparison, we integrate the semantic embeddings from baseline methods, LERF [20], LSM [12], and GOI [34], directly into our pipeline. This is done by replacing only the pixel-level semantic embeddings while keeping the geometry backbone and the entire evaluation procedure identical. The final evaluation is conducted by projecting the 3D segmentation results back to 2D for comparison with ground-truth annotations. This controlled setup ensures that performance differences are attributable solely to the quality of the semantic embeddings. We report the total pipeline processing time (preprocessing and inference) to assess computational efficiency. For multi-view depth estimation, we adhere to the official protocol of the RobustMVD benchmark [40]. For each test sample, we use source views selected via the quasi-optimal strategy. Performance is measured using Absolute Relative Error (Rel) ↓ and the Inlier Ratio (τ) ↑ at a threshold of 1.03. Results are reported for each individual dataset as well as the overall average across all benchmarks.

#### 4.2. Main Results

2D-to-3D Open-Vocabulary Segmentation. We first evaluate PE3R on the Mip-NeRF360 and Replica datasets,

Method mIoU mPA mP

w/o Multi-Level Disam. 0.1624 0.5892 0.5623 w/o Cross-View Disam. 0.1895 0.6012 0.5923

w/o Global MinMax Norm. 0.2035 0.6253 0.6186 PE3R (full) 0.2248 0.6542 0.6315

- Table 5. Effect of Disambiguation Modules for 2D-to-3D openvocabulary segmentation on ScanNet++.

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

- (a)
- (b)

[Figure 243]

[Figure 244]

Flowerpot on table Pink flower Basin on table

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

(b)

Flowerpot on table Pink flower Basin on table

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

- (a)
- (b)

[Figure 257]

[Figure 258]

Espresso machine Brew head Drip tray

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

Espresso machine Brew head Drip tray

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

(a)

[Figure 271]

[Figure 272]

A box of donuts Paper boxes Chocolate donut

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

A box of donuts Paper boxes Chocolate donut

(a) w/o Multi-Level Disambiguation (b) w/ Multi-Level Disambiguation

Figure 3. Effect of Multi-Level Disambiguation. Without this module, the model fails to associate object parts with the whole, leading to fragmented semantics.

which provide a standard benchmark for comparing with state-of-the-art NeRF- and 3DGS-based methods. As summarized in Table 1, PE3R consistently surpasses all previous approaches, establishing a new state-of-the-art across all metrics (mIoU, mPA, mP). A pivotal advantage of our framework is its exceptional computational efficiency. As evidenced by Table 2, PE3R completes the full semantic reconstruction pipeline in merely 5 minutes. This represents up to a 9× speedup over optimization-based methods. To assess generalization, we evaluate PE3R on the large-scale and complex ScanNet++ benchmark. Given the scalability constraints of methods requiring per-scene optimization, we conduct a direct and fair comparison by integrating the 2D semantic features from LERF, LSM, and GOI into our pipeline under identical settings. As reported in Table 3, PE3R achieves superior segmentation accuracy, confirming its ability to maintain high performance while scaling effectively to challenging, large-scale environments. Qualitative results in Fig. 1 further corroborate these findings, showcasing robust and precise segmentation across diverse indoor, outdoor, and in-the-wild scenes.

Multi-view Depth Estimation. We further evaluate the ge-

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

Bicycle Chair

- (a) w/o Cross-View Disambiguation
- (b) w/ Cross-View Disambiguation

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

Bicycle Chair

- Figure 4. Effect of Cross-View Disambiguation. In the absence of cross-view alignment, the same object is inconsistently labeled across viewpoints.

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

Remote control Table Sofa Black box

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

- (a) w/o Global Min-Max Normalization
- (b) w/ Global Min-Max Normalization

Remote control Table Sofa Black box

- Figure 5. Effect of Global Min-Max Normalization. Without global normalization, similarity scores become uncalibrated across views, impairing retrieval reliability.

ometric fidelity of PE3R on the task of multi-view depth estimation. For a comprehensive analysis, Table 4 organizes methods into three categories: (a) classical 3D reconstruction approaches, (b) modern feed-forward methods, and (c) our full PE3R framework with semantic-guided reconstruction. PE3R demonstrates highly competitive performance across a wide spectrum of datasets, achieving the best overall average on standard metrics (Rel and τ). This indicates that the integration of semantic priors generally enhances the robustness of the geometric reconstruction. An in-depth analysis reveals that the benefits of semantic guidance are scene-dependent. While PE3R shows notable improvements on most benchmarks, its performance on ScanNet and DTU is marginally behind the DUSt3R baseline. We attribute this to the characteristic challenges of these datasets, such as low semantic diversity, repetitive textures, and limited viewpoint coverage, where semantic cues can introduce ambiguity rather than resolve it. This nuanced performance profile highlights a key insight: semantic guidance is most beneficial in semantically rich and visually distinct environments. Nevertheless, the improvement in the overall benchmark average strongly validates that our semantic augmentation strategy provides a net positive effect on reconstruction quality.

Multi-Level Disam. Cross-View Disam. mIoU mPA mP

Method rel↓ τ ↑ Runtime ∆ w/o Semantic Recon. 4.9 64.4 10.40s -

averaged averaged 0.1994 0.6001 0.5985 averaged area-slerp 0.2158 0.6141 0.6015 area-slerp averaged 0.2058 0.6041 0.6003

PE3R (full) 4.5 68.0 11.19s +0.79s

conf-weighted conf-weighted 0.1532 0.5723 0.5435 conf-weighted conf-slerp 0.1743 0.5842 0.5554

Table 7. Ablation on Semantic Point Cloud Reconstruction. Removing this module degrades geometric quality.

conf-slerp conf-weighted 0.1721 0.5813 0.5493 conf-slerp conf-slerp 0.1998 0.5991 0.5918

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

|[Figure 323]<br><br>[Figure 324]|
|---|

|[Figure 325]<br><br>[Figure 326]|
|---|

area-slerp area-slerp 0.2248 0.6542 0.6315

- Table 6. Comparison of Feature Aggregation Strategies on ScanNet++.

[Figure 327]

[Figure 328]

[Figure 329]

|[Figure 330]<br><br>[Figure 331]|
|---|

|[Figure 332]<br><br>[Figure 333]|
|---|

[Figure 334]

#### 4.3. Ablation Studies

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

To provide a comprehensive analysis of PE3R’s design, we present key ablation studies here, with additional experiments detailed in the supplementary material. All evaluations are conducted on both open-vocabulary segmentation and multi-view depth estimation to quantitatively assess the contribution of each component.

|[Figure 339]<br><br>[Figure 340]|
|---|

|[Figure 341]<br><br>[Figure 342]|
|---|

[Figure 343]

[Figure 344]

|[Figure 345]<br><br>[Figure 346]|
|---|

|[Figure 347]<br><br>[Figure 348]|
|---|

[Figure 349]

[Figure 350]

Effect of Disambiguation Modules. We first analyze the three core modules responsible for establishing a consistent semantic space: Multi-Level Disambiguation, CrossView Disambiguation, and Global Min-Max Normalization. Quantitative results on ScanNet++ (Table 5) demonstrate that ablating any module causes a performance drop, with the most significant degradation occurring when MultiLevel Disambiguation is removed. This underscores its critical role in resolving hierarchical ambiguities and maintaining part-whole consistency. Qualitative results in Figs. 3–

(a) w/o Semantic Field Reconstruction (b) w/ Semantic Field Reconstruction

Figure 6. Effect of Semantic Point Cloud Reconstruction. Removing this module increases spatial noise and reduces objectlevel coherence in the reconstructed point cloud, highlighting its importance for both geometric and semantic fidelity.

Effect of Semantic Reconstruction Module. We further ablate the Semantic Point Cloud Reconstruction module to evaluate its impact on geometric accuracy. The results in Table 7 show that this module brings a noticeable improvement in depth estimation metrics with only a minimal computational overhead. Qualitatively, Fig. 6 demonstrates that semantic-guided refinement effectively suppresses spatial outliers caused by challenging conditions like reflections and transparency, resulting in cleaner and more structurally coherent point clouds. This confirms that semantic cues actively contribute to enhancing the underlying geometry, beyond merely providing labels.

- 5 offer further insight. As shown in Fig. 3, the absence of Multi-Level Disambiguation leads to fragmented semantics. While parts (e.g., “Drip tray”) may be detected, the model fails to associate them with the whole object (e.g., “Espresso machine”). Our module enables coherent hierarchical understanding. Fig. 4 illustrates that without CrossView Disambiguation, the same object (e.g., “Bicycle” or “Chair”) is inconsistently labeled across viewpoints due to occlusion or perspective changes. Integrating this module restores semantic coherence throughout the 3D scene. Fig. 5 reveals that removing Global Min-Max Normalization results in uncalibrated similarity scores across views, impairing retrieval reliability. This step ensures a consistent and interpretable similarity range, which is crucial for robust open-vocabulary localization. To validate our specific design choice for feature aggregation, we compare our geometry-aware interpolation against naive uniform averaging and confidence-weighted baselines using DUSt3R’s per-point confidence (conf-avg and conf-slerp). As shown in Table 6, uniform averaging leads to consistently lower mIoU and less accurate retrieval, while our area-weighted slerp outperforms both confidence-weighted variants on ScanNet++. This confirms that our area-weighted, spherical interpolation is essential for generating discriminative and reliable semantic embeddings.

### 5. Conclusion

We presented PE3R, a framework that establishes a new paradigm for efficient and generalizable 3D semantic reconstruction from unposed images. By integrating pixel embedding disambiguation, semantic point cloud reconstruction, and global view perception into a cohesive pipeline, PE3R achieves robust zero-shot generalization across diverse scenes without any scene-specific fine-tuning. Extensive evaluations across diverse benchmarks demonstrate that PE3R achieves state-of-the-art accuracy with good efficiency. We believe PE3R paves the way for more scalable and practical 3D vision systems, with promising applications in robotics, AR/VR, and autonomous navigation.

Acknowledgements. This project is supported by the National Research Foundation, Singapore, under its Medium Sized Center for Advanced Robotics Technology Innovation.

### References

- [1] Henrik Aanæs, Rasmus Ramsbøl Jensen, George Vogiatzis, Engin Tola, and Anders Bjorholm Dahl. Large-scale data for multiple-view stereopsis. International Journal of Computer Vision, 120:153–168, 2016. 1, 6
- [2] Vladislav Ayzenberg and Marlene Behrmann. Development of visual object recognition. Nature Reviews Psychology, 3

(2):73–90, 2024. 1

- [3] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5470–5479, 2022. 1, 6
- [4] Yash Bhalgat, Iro Laina, Joao F Henriques, Andrew Zisserman, and Andrea Vedaldi. Contrastive lift: 3d object instance segmentation by slow-fast contrastive fusion. arXiv preprint arXiv:2306.04633, 2023. 2
- [5] Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian, et al. Segment anything in 3d with nerfs. Advances in Neural Information Processing Systems, 36:25971–25990, 2023. 1, 2
- [6] Jiazhong Cen, Jiemin Fang, Chen Yang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. Segment any 3d gaussians, 2024. 1, 2
- [7] Hanlin Chen, Chen Li, Mengqi Guo, Zhiwen Yan, and Gim Hee Lee. Gnesf: Generalizable neural semantic fields. Advances in Neural Information Processing Systems, 36: 36553–36565, 2023. 2
- [8] Seokhun Choi, Hyeonseop Song, Jaechul Kim, Taehyeong Kim, and Hoseok Do. Click-gaussian: Interactive segmentation to any 3d gaussians. arXiv preprint arXiv:2407.11793,

2024. 2

- [9] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 1, 6
- [10] Bardienus Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. Mast3rsfm: a fully-integrated solution for unconstrained structurefrom-motion. arXiv preprint arXiv:2409.19152, 2024. 2
- [11] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views. In International Conference on Learning Representations, 2024. 4
- [12] Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to semantic 3d. Advances in Neural Information Processing Systems, 37:40212–40229,

2024. 2, 4, 5, 6

- [13] Qiancheng Fu, Qingshan Xu, Yew Soon Ong, and Wenbing Tao. Geo-neus: Geometry-consistent neural implicit surfaces learning for multi-view reconstruction. Advances in Neural Information Processing Systems, 35:3403–3416, 2022. 2
- [14] Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237,

2013. 1, 6

- [15] Rahul Goel, Dhawal Sirikonda, Saurabh Saini, and PJ Narayanan. Interactive segmentation of radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4201–4211, 2023. 1
- [16] Jianfei Guo, Nianchen Deng, Xinyang Li, Yeqi Bai, Botian Shi, Chiyu Wang, Chenjing Ding, Dongliang Wang, and Yikang Li. Streetsurf: Extending multi-view implicit surface reconstruction to street views. arXiv preprint arXiv:2306.04988, 2023. 2
- [17] Yansong Guo, Jie Hu, Yansong Qu, and Liujuan Cao. Wildseg3d: Segment any 3d objects in the wild from 2d images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5166–5176, 2025. 1
- [18] Xu Hu, Yuxi Wang, Lue Fan, Junsong Fan, Junran Peng, Zhen Lei, Qing Li, and Zhaoxiang Zhang. Semantic anything in 3d gaussians. arXiv preprint arXiv:2401.17857, 2024. 1, 2
- [19] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 1, 2

- [20] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19729–19739,

2023. 1, 4, 5, 6

- [21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023. 2, 3, 4
- [22] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36

(4):1–13, 2017. 1, 6

- [23] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. Advances in Neural Information Processing Systems, 35:23311–23330, 2022. 1
- [24] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. arXiv preprint arXiv:2406.09756, 2024. 1, 2, 6
- [25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023. 3

- [26] Yichen Liu, Benran Hu, Junkai Huang, Yu-Wing Tai, and Chi-Keung Tang. Instance neural radiance field. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 787–796, 2023. 2
- [27] Yichen Liu, Benran Hu, Chi-Keung Tang, and Yu-Wing Tai. Sanerf-hq: Segment anything for nerf in high quality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3216–3226, 2024. 1
- [28] Xiaoxiao Long, Cheng Lin, Peng Wang, Taku Komura, and Wenping Wang. Sparseneus: Fast generalizable neural surface reconstruction from sparse views. In European Conference on Computer Vision, pages 210–227. Springer, 2022. 2
- [29] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 1, 2
- [30] Michael Niemeyer and Andreas Geiger. Giraffe: Representing scenes as compositional generative neural feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11453–11464, 2021. 2
- [31] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 3
- [32] Songyou Peng, Kyle Genova, Chiyu Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas Funkhouser, et al. Openscene: 3d scene understanding with open vocabularies. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 815–824, 2023. 1
- [33] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024. 1, 2, 4, 5
- [34] Yansong Qu, Shaohui Dai, Xinyang Li, Jianghang Lin, Liujuan Cao, Shengchuan Zhang, and Rongrong Ji. Goi: Find 3d gaussians of interest with an optimizable open-vocabulary semantic-space hyperplane. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5328–5337,

2024. 2, 4, 5, 6

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 2, 3, 5
- [36] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 3, 4, 6
- [37] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE con-

- ference on computer vision and pattern recognition, pages 4104–4113, 2016. 6
- [38] Johannes L Sch¨onberger, Enliang Zheng, Jan-Michael Frahm, and Marc Pollefeys. Pixelwise view selection for unstructured multi-view stereo. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 501–518. Springer, 2016. 6
- [39] Thomas Schops, Johannes L Schonberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3260–3269, 2017. 1, 6
- [40] Philipp Schr¨oppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. A benchmark and a baseline for robust multiview depth estimation. In Proceedings of the International Conference on 3D Vision (3DV), 2022. 6
- [41] Qiuhong Shen, Xingyi Yang, and Xinchao Wang. Flashsplat: 2d to 3d gaussian splatting segmentation solved optimally. arXiv preprint arXiv:2409.08270, 2024. 2
- [42] Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bul´o, Norman M¨uller, Matthias Nießner, Angela Dai, and Peter Kontschieder. Panoptic lifting for 3d scene understanding with neural fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9043–9052, 2023. 2
- [43] Pawan Sinha and Tomaso Poggio. Role of learning in threedimensional form perception. Nature, 384(6608):460–463,

1996. 1

- [44] Julian Straub, Thomas Whelan, Lingni Ma, Yufan Chen, Erik Wijmans, Simon Green, Jakob J Engel, Raul Mur-Artal, Carl Ren, Shobhit Verma, et al. The replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797,

2019. 1, 6

- [45] Ay¸ca Takmaz, Elisabetta Fedele, Robert W Sumner, Marc Pollefeys, Federico Tombari, and Francis Engelmann. Openmask3d: Open-vocabulary 3d instance segmentation. arXiv preprint arXiv:2306.13631, 2023. 1
- [46] Songlin Tang, Wenjie Pei, Xin Tao, Tanghui Jia, Guangming Lu, and Yu-Wing Tai. Scene-generalizable interactive segmentation of radiance fields. In Proceedings of the 31st ACM International Conference on Multimedia, pages 6744–6755, 2023.
- [47] Vadim Tschernezki, Iro Laina, Diane Larlus, and Andrea Vedaldi. Neural feature fusion fields: 3d distillation of self-supervised 2d image representations. In 2022 International Conference on 3D Vision (3DV), pages 443–453. IEEE, 2022. 1
- [48] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061, 2024. 2
- [49] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 21686–21697, 2024. 2
- [50] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt:

- Visual geometry grounded transformer. arXiv preprint arXiv:2503.11651, 2025. 1, 2, 6
- [51] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. arXiv preprint arXiv:2106.10689, 2021. 2
- [52] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 1, 2, 3, 4, 6
- [53] Yiqun Wang, Ivan Skorokhodov, and Peter Wonka. Hf-neus: Improved surface reconstruction using high-frequency details. Advances in Neural Information Processing Systems, 35:1966–1978, 2022. 2
- [54] Andrew E Welchman, Arne Deubelius, Verena Conrad, Heinrich H B¨ulthoff, and Zoe Kourtzi. 3d shape perception from combined depth cues in human visual cortex. Nature neuroscience, 8(6):820–827, 2005. 1
- [55] Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025. 2, 6
- [56] Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. arXiv preprint arXiv:2312.00732, 2023. 1, 2, 4, 5
- [57] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12–22, 2023. 1, 6
- [58] Hong-Xing Yu, Leonidas J Guibas, and Jiajun Wu. Unsupervised discovery of object radiance fields. arXiv preprint arXiv:2107.07905, 2021. 2
- [59] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 2
- [60] Chaoning Zhang, Dongshen Han, Sheng Zheng, Jinwoo Choi, Tae-Ho Kim, and Choong Seon Hong. Mobilesamv2: Faster segment anything to everything. arXiv preprint arXiv:2312.09579, 2023. 6
- [61] Shuaifeng Zhi, Tristan Laidlow, Stefan Leutenegger, and Andrew J Davison. In-place scene labelling and understanding with implicit scene representation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15838–15847, 2021. 2
- [62] Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21676–21685, 2024. 1, 2, 4, 5

## PE3R: Perception-Efficient 3D Reconstruction Supplementary Material

### 6. Proof of Interpolation Properties

This section provides the complete proofs for the two key properties of our area-weighted interpolation strategy, which underpin the effectiveness of the Pixel Embedding Disambiguation described in Sec. 3.3 of the main text.

- Proposition 1. Vector Normalization: The interpolated vec-

tor FˆB preserves unit norm, ensuring it remains within the original semantic embedding space. Proof. The norm of FˆB is given by:

∥FˆB∥2 = ∥aFA + bFB∥2 . (8) Expanding this expression:

∥FˆB∥2 = (aFA + bFB) · (aFA + bFB)

= a2∥FA∥2 + 2abFA · FB + b2∥FB∥2.

(9)

Since FA and FB are unit vectors:

∥FA∥ = 1,∥FB∥ = 1,FA · FB = cos(θ). (10) Substituting these values, we get:

∥FˆB∥2 =

1 sin2(θ)

(sin2((1 − t)θ) + sin2(tθ)

+ 2sin((1 − t)θ)sin(tθ)cos(θ)).

(11)

Using trigonometric identities:

sin2(θ) = sin2((1 − t)θ) + sin2(tθ)

+ 2sin((1 − t)θ)sin(tθ)cos(θ),

(12)

we find that:

∥FˆB∥2 = sin2(θ)/sin2(θ) = 1. (13) Thus, FˆB is confirmed to be a unit vector. □

- Proposition 2. Semantic Guidance: If FA has a higher similarity to a reference semantic vector FC than FB does, then FˆB will also exhibit a higher similarity to FC than FB does. This steers the aggregated representation toward more semantically meaningful directions. Proof. The cosine similarity between FˆB and FC is:

FˆB · FC = a(FA · FC) + b(FB · FC). (14) Since FA · FC > FB · FC, we have:

a(FA · FC) + b(FB · FC) > (a + b)(FB · FC). (15)

Threshold rel↓ τ ↑

- 0.001 6.1 49.3

- 0.002 5.9 50.3

- 0.003 5.5 55.1

- 0.004 5.6 53.6

- 0.005 5.8 52.8

Table 8. Anomaly point selection with varying thresholds on ScanNet. The optimal value of 0.003 is used as the default in all main experiments.

Using trigonometric properties:

sin((1 − t)θ) sin(θ)

+

a + b =

we conclude:

sin(tθ) sin(θ)

= 1, (16)

FˆB · FC = a(FA · FC) + b(FB · FC) > FB · FC. (17)

This confirms that FˆB semantically integrates information from both FA and FB, steering the aggregated representation toward more meaningful directions. □

### 7. Extended Ablation Studies

This section provides extended ablation studies that complement the analysis in Sec. 4.3 of the main paper, offering further insights into the design choices and parameter sensitivity of PE3R.

#### 7.1. Anomaly Point Selection

The accuracy of our anomaly point detection mechanism is validated through both empirical and theoretical analysis.

Empirical Validation. We determine the anomaly threshold by statistically analyzing the mean 3D distance between spatially consistent points across semantic regions. This threshold effectively separates normal points from anomalies. Table 8 presents results under different thresholds on ScanNet, with the optimal performance observed at a threshold of 0.003.

Theoretical Validation. While anomaly detection could theoretically be approached through parametric modeling of intra-object distributions (e.g., using Gaussian estimation or least-squares fitting), such methods become computationally prohibitive in complex, large-scale scenes. Our empirical thresholding strategy provides a practical and scalable alternative that maintains high performance without introducing significant computational overhead.

Window Size avg. rel↓ avg. τ ↑

3 × 3 4.9 65.5 5 × 5 4.5 68.0 7 × 7 4.8 66.1

- Table 9. Effect of sliding window size on anomaly detection performance across all depth estimation benchmarks.

a b avg. rel↓ avg. τ ↑

0.00 1.00 5.3 60.2 0.10 0.90 4.5 68.0 0.20 0.80 4.7 62.3 0.50 0.50 4.9 61.6 0.80 0.20 6.1 59.4

- 0.90 0.10 6.5 57.5

- 1.00 0.00 10.2 50.2

- Table 10. Effect of semantic-aware smoothing parameter α on reconstruction performance. Moderate smoothing (α = 0.1) achieves the best results.

#### 7.2. Sliding Window Size Analysis

The size of the sliding window used for local semanticaware distance computation significantly impacts both reconstruction quality and computational efficiency. As demonstrated in Table 9, we evaluate three window sizes on the multi-view depth estimation task. A 5 × 5 window achieves the optimal trade-off, capturing sufficient local context for robust anomaly detection without introducing excessive computational cost or over-smoothing fine geometric details.

#### 7.3. RGB Image Smoothing Analysis

To enhance robustness in challenging regions characterized by reflections, transparency, or complex textures, we apply semantic-aware smoothing to the input images prior to pointmap estimation. The smoothing operation is defined as yˆ = α · x + (1 − α) · y, where x is the mean RGB value of the semantic region, y is the original pixel value, and α controls the smoothing strength.

- As shown in Table 10, moderate smoothing (α = 0.1)

provides the optimal balance between noise suppression and detail preservation. Lower values (α ≤ 0.1) insufficiently address visual artifacts, while higher values (α ≥ 0.2) oversmooth genuine structural details, ultimately degrading geometric accuracy.

7.4. Iterative Refinement Analysis

We further investigate the effect of repeated refinement iterations in the semantic point cloud reconstruction module.

- As shown in Table 11, while a single iteration provides substantial improvement, additional iterations yield diminishing returns with increased computational cost. This suggests that our method achieves effective refinement in a sin-

Iterations 1 2 3 4 avg. rel↓ 4.5 4.6 4.6 4.6 avg. τ ↑ 68.0 67.9 67.9 67.9

Table 11. Effect of iterative refinement cycles on reconstruction performance (Mip-NeRF360). A single iteration achieves optimal performance.

gle pass, maintaining the efficiency advantages of the overall feed-forward pipeline.

