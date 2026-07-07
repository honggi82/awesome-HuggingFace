## MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction

Haitian Li* , Haozhe Xie* , Junxiang Xu , Beichen Wen , Fangzhou Hong , and Ziwei Liu

S-Lab, Nanyang Technological University, 637335 Singapore https://lihaitian.com/MonoArt

# arXiv:2603.19231v1[cs.CV]19Mar2026

###### Baselines MonoArt

Input

0.8 0.7 0.6 0.5 0.4 0.3 0.2

|· MonoArt| |
|---|---|
|· MonoArt<br><br>| |
|· SINGAPO<br><br>·|·<br><br>PhysXAny<br><br>·|
|PhysXGen|ArtAny|
| | |
|· URDFo|rmer|
| | |

（Ours）

SINGAPO ArtAny PhysXAny

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

F-score

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

No Output

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

No Output

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

10 100 1000

No Output

Inference Time (seconds)

Fig. 1: (Left) Qualitative results of SINGAPO [24], Articulate-Anything (ArtAny) [18], PhysX-Anything (PhysXAny) [4], and MonoArt on diverse objects. (Right) F-score vs. inference time on the PartNet-Mobility [51] test set. Circles indicate models evaluated on 7 categories, while triangles denote models supporting all 46 categories.

Abstract. Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that MonoArt achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

Keywords: Monocular Articulated 3D Reconstruction · Progressive Structural Reasoning · Kinematic Estimation

* Equal Contribution Corresponding Author

### 1 Introduction

Reconstructing articulated 3D objects from visual observations remains a fundamental challenge in computer vision and graphics. Such objects (e.g., laptops, cabinets) are ubiquitous in daily environments, and generating high-fidelity, interactable 3D assets at scale is increasingly vital for applications in robotics [1,

##### 2, 55] and scene synthesis [49, 53, 54, 58]. Despite recent progress in 3D object generation [10,17,46,52], producing high-quality articulated assets still requires extensive manual effort. Unlike rigid reconstruction, articulated reconstruction demands understanding both an object’s structural composition and the kinematic relationships among its parts.

Articulated 3D object modeling has attracted increasing attention in recent years, driving significant advances in reconstruction. Existing approaches can be broadly classified into three categories. Most existing methods [13,25,34,48,50] rely on multiple images captured from videos or frame sequences indicating different articulation states (e.g., open and closed). While achieving higher reconstruction accuracy, they require multiple motion states of the same object, which are not always readily available in practice. To relax this constraint, subsequent approaches [8,18,24] take a single image as input but reconstruct 3D objects by retrieving and assembling parts from pre-built asset libraries, often leading to texture misalignment and geometric inaccuracies. Very recent works move beyond retrieval and attempt single-image reconstruction of articulated objects by generating auxiliary videos [31], leveraging vision–language models [4], or relying on predefined motion directions to infer articulation cues [12, 21]. While these approaches demonstrate the potential of single-image articulation modeling, the former is complex and computationally expensive, whereas the latter depends on handcrafted priors that limit generalization. This reliance on external cues reflects the lack of intrinsic 3D understanding, making it difficult to infer part composition and spatial relationships directly from a single image.

To address these limitations, we propose MonoArt, an end-to-end framework for monocular articulated 3D reconstruction grounded in progressive structural reasoning. Instead of directly regressing articulation parameters from image features, MonoArt progressively constructs structured part representations and lifts them into motion-aware embeddings, enabling stable and interpretable kinematic prediction. Specifically, TRELLIS-based 3D Generator first produces a canonical 3D shape from the input image, providing a stable geometric foundation. Part-aware Semantic Reasoner then lifts geometry-aligned point features into globally contextualized part-level embeddings through triplane aggregation and transformer-based refinement under 3D structural supervision. These part-aware features are subsequently processed by a Dual-Query Motion Decoder, which decouples spatial motion anchors and semantic part representations, and performs iterative refinement to reason about componentlevel motion patterns. Finally, Kinematic Estimator predicts part-level articulation parameters, including the part centroid, mask, joint type, axis, origin, and motion limits, and infers the kinematic tree structure, yielding a coherent articulated 3D reconstruction. As illustrated in Figure 1, MonoArt achieves both

higher F-score and substantially lower inference time than existing approaches, demonstrating that explicit structural priors enable not only more accurate but also more efficient articulated reconstruction.

The contributions are summarized as follows:

- – We show that embedding 3D structural priors simplifies single-image articulated reconstruction by eliminating reliance on video generation, handcrafted motion templates, or vision–language priors, enabling reliable part decomposition and articulation reasoning.
- – We propose MonoArt, a unified end-to-end framework that progressively reasons from geometry to kinematics, disentangling shape recovery, partaware encoding, motion decoding, and kinematic regression for stable and physically meaningful articulation inference.
- – MonoArt achieves state-of-the-art performance on PartNet-Mobility in both geometric and articulation metrics while significantly reducing inference time. The framework further generalizes to robotic manipulation and indoor 3D reconstruction tasks.

### 2 Related Work

Articulated Object Modeling. Articulated object modeling aims to recover object geometry together with part structure and motion from visual observations. Early multi-view methods [44, 48] adopt neural implicit representations to model category-level articulated shapes as deformations of a canonical template, enabling view synthesis but without explicit part or kinematic modeling. Later works introduce explicit part decomposition and motion reasoning. PARIS [25] aligns reconstructions across articulation states to estimate rigid parts and transformations. DTA [50] and ArticulatedGS [13] jointly model geometry, segmentation, and articulation parameters from multi-view RGB-D or Gaussian Splatting, producing motion-ready digital twins with high geometric fidelity. More recent approaches reduce input requirements by leveraging generative or language priors. FreeArt3D [7] optimizes geometry from sparse articulated views, while SINGAPO [24], NAP [19], and MeshArt [11] predict articulation trees and synthesize articulated parts. Articulate-Anything [18] formulates the task as vision–language reasoning to infer symbolic part hierarchies, and PhysX-Anything [4] further incorporates VLM priors to predict physically plausible structures and interactions. These methods improve scalability and controllability, often at the expense of precise instance-level reconstruction.

##### 3D Part Segmentation. 3D part segmentation aims to decompose objects into semantic parts. Early learning-based methods [14, 22, 23, 30, 38–40, 61] focus on fully supervised point- or mesh-level classification using curated 3D datasets [6,9,35,60]. While effective, these approaches are limited by the scale and diversity of part annotations. To improve open-world generalization and enable zero-shot capabilities, recent methods leverage 2D foundation vision models [16,20,29,37,41]. PartSLIP [28], PartSLIP++ [62], and ZeroPS [57] transfer image–language and segmentation priors to 3D via multi-view reasoning or

prompt-based inference, while PartDistill [47], SaMesh [45], and SAMPart3D [59] distill 2D foundation features into geometric representations. More recently, scaling-based approaches train feed-forward 3D segmentation models with largescale part annotations. Find3D [33] leverages foundation models for pseudo-label generation, PartField [26] learns ambiguity-aware continuous feature fields, and P3-SAM [32] and PartSAM [63] demonstrate that large-scale part supervision yields strong point-wise representations for part prompting.

### 3 Our Approach

#### 3.1 Overview

MonoArt reconstructs articulated 3D objects from a single image by progressively transforming visual observations into geometry-aware, part-aware, and motion-aware representations. As shown in Fig. 2, the framework consists of four components: TRELLIS-based 3D Generator (Sec. 3.2), Part-Aware Semantic Reasoner (Sec. 3.3), Dual-Query Motion Decoder (Sec. 3.4), and Kinematic Estimator (Sec. 3.5).

Given an input image I, TRELLIS-based 3D Generator reconstructs a canonical geometry O using a frozen TRELLIS backbone and produces geometryaligned latent features Z. Based on (O,Z), Part-Aware Semantic Reasoner derives part-aware features H that encode explicit part decomposition guided by 3D structural annotations. Then, Dual-Query Motion Decoder initializes position and content queries (Q0p,Q0c) from (H,Fgeo) and iteratively refines them to obtain motion-aware representations (QLp ,QLc ), jointly reasoning about part localization and motion semantics. Finally, Kinematic Estimator transforms the refined queries into explicit articulation parameters, namely part masks mm, motion types mt, motion origins mo, motion axes ma, and motion limits ml, while inferring the kinematic hierarchy to produce an articulated 3D representation.

#### 3.2 TRELLIS-based 3D Generator

Given a single RGB image I, MonoArt first reconstructs a canonical 3D object geometry using TRELLIS [52] as a frozen 3D generation backbone. TRELLIS predicts a structured sparse voxel latent representation Z ∈ RN

z×Nz×Nz×d1, where Nz denotes the voxel grid resolution and each active voxel stores a d1dimensional feature while empty regions remain inactive, forming a spatially structured sparse volume. The latent Z is subsequently decoded by the mesh decoder of TRELLIS to produce an explicit 3D mesh O, which serves as the canonical geometry for downstream part reasoning and articulation inference.

#### 3.3 Part-Aware Semantic Reasoner

Our goal is to derive part-aware point features H from the canonical geometry O and the sparse voxel latent Z predicted by TRELLIS-based 3D generator, so that downstream modules can reason about object parts and their motions.

Part-aware Semantic Reasoner

TRELLIS-based

###### Kinematic Estimator

- 3D Generator

[Figure 31]

𝐐𝐐p𝐿𝐿 𝐐𝐐𝑐𝑐𝐿𝐿 𝐇𝐇

PartCont.Trans.

Tri-linearInterp.

[Figure 32]

TriplaneProj.

[Figure 33]

[Figure 34]

𝐎𝐎

Articulate Param. Regressor

𝐙𝐙

Type 𝐦𝐦𝑡𝑡 Axis 𝐦𝐦a Limit 𝐦𝐦l

𝐅𝐅geo

𝐅𝐅tri

𝐇𝐇

Input Image 𝐈𝐈

[Figure 35]

Origin 𝐦𝐦o

Mask 𝐦𝐦m

Outputs

Dual-Query Motion Decoder

𝐒𝐒𝑝𝑝𝐿𝐿

TRELLIS

Refinement Block × 𝐿𝐿

𝐐𝐐𝑝𝑝𝑙𝑙−1

Dual-queryInit.

Position Query

𝐐𝐐p0 𝐐𝐐c0

MLPMLP

𝐇𝐇

𝐐𝐐𝑝𝑝𝐿𝐿

CrossAttn.

SelfAttn.

Pairwise Affinity

𝚫𝚫𝑙𝑙𝑝𝑝

Content Query

[Figure 36]

[Figure 37]

Parent Assignment

𝐅𝐅geo

CLIP

𝐐𝐐𝑐𝑐𝐿𝐿

Kinematic Tree Predictor

𝚫𝚫𝑙𝑙𝑐𝑐

𝐐𝐐𝑐𝑐𝑙𝑙−1

Mesh 𝐎𝐎

Latent 𝐙𝐙 S𝑜𝑜 Object Class 𝐒𝐒𝑝𝑝𝐿𝐿 Part Logits

Kinematic Tree

𝐇𝐇

- Fig. 2: Overview of MonoArt. TRELLIS-based 3D Generator reconstructs a canonical shape from a single image. Part-Aware Semantic Reasoner derives tri-plane-based part embeddings. Dual-Query Motion Decoder performs iterative motion reasoning, and Kinematic Estimator predicts part-level articulation parameters (motion type, origin, axis, limits) and infers the kinematic tree structure. Note that “Attn.”, “Interp.”, “Proj.”, “Cont.”, “Trans.”, and “Init.” represent “Attention”, “Interpolation”, “Projection”, “Contrast”, “Transformer”, and “Initialization”, respectively. ⊕ and ⊗ denote elementwise addition and matrix multiplication, respectively.

Tri-linear Interpolation. We sample M point sets {pm}Mm=1 on the surface of O, where pm ∈ R3. Given the sparse voxel feature volume Z, we obtain point-aligned features by tri-linear interpolation over the neighboring voxels:

fm = TrilinearInterp(Z,pm), Fgeo = {fm}Mm=1. (1)

Here, each point pm is first mapped to its corresponding continuous voxel coordinate in the Nz × Nz × Nz grid. The feature fm ∈ Rd

1 is then computed as the weighted combination of the eight neighboring voxel features according to standard tri-linear interpolation [56]. This converts the discrete voxel latent Z into continuous surface-aligned features Fgeo ∈ RM×d

1.

Triplane Projection. To incorporate global spatial context while preserving geometric structure, we project the geometry-aligned point features Fgeo onto three orthogonal planes (XY, YZ, ZX) following the triplane formulation [5]. Specifically, each surface point pm is orthographically projected onto the three planes according to its 3D coordinates, and its feature fm is accumulated to the corresponding 2D grid locations via bilinear interpolation. This yields three feature maps of resolution Nt × Nt, forming Ftri ∈ R3×N

t×Nt×d1.

Part Contrast Transformer. The triplane features Ftri are flattened into a token sequence T ∈ R3N

2

t ×d1 and processed by a multi-layer self-attention Transformer to capture global interactions across planes, producing refined tokens T′. The refined tokens are reshaped back into triplane feature maps F′tri ∈ R3×N

t×Nt×d1. For each surface point pm, we project its 3D coordinate onto the three orthogonal planes and query the refined triplane feature maps via a

tri-plane sampling operation. This yields the updated point embedding

hm = MLP(TriQuery(F′tri,pm)), H = {hm}Mm=1, (2)

where TriQuery projects pm onto the XY/YZ/ZX planes, bilinearly samples each plane feature map, and concatenates the three sampled features. Then, the MLP then lifts the aggregated feature from dimension d1 to d2. These embeddings encode part-aware structural representations and are supervised by 3D part annotations using triplet loss [43].

#### 3.4 Dual-Query Motion Decoder

Articulated reasoning requires jointly modeling what constitutes a movable part and where its motion is spatially anchored. To this end, we adopt a dual-query formulation that disentangles semantic representation and geometric localization via two complementary query types: a content query Qc ∈ RN

q×d2 encoding part semantics, and a position query Qp ∈ RN

q×3 representing spatial motion anchors, where Nq denotes the number of dual queries. The dual queries are initialized from global object context and subsequently refined through L stacked refinement blocks in an iterative manner.

Dual-query Initialization. Given the part-aware point embeddings H and geometry-aligned features Fgeo, we first aggregate global object context by applying global pooling over H and Fgeo, followed by feature concatenation to obtain an object-level representation. This aggregated feature is projected to generate the initial content queries Q0c ∈ RN

q×3, together with an auxiliary object-category prediction So.

q×d2 and position queries Q0p ∈ RN

Refinement Block. The refinement block iteratively updates the dual queries over L layers to progressively refine articulation hypotheses. Given the initialized position query Q0p ∈ RN

q×3 and content query Q0c, each layer first applies selfattention to model inter-part interactions, followed by cross-attention where the queries attend to the visual feature H to retrieve image evidence. For the l-th layer, the position query Qlp represents spatial motion anchors that localize candidate articulation points. It is updated via a residual scheme Qlp = Qlp−1 + ∆lp, allowing progressive refinement of spatial motion anchors. In parallel, the content query is also updated in a residual manner Qlc = Qlc−1 + ∆lc, where ∆lc captures semantic refinement derived from self- and cross-attention. Based on the refined content queries, we predict per-query part class logits Slp, which provide structural supervision and are further used to retrieve semantic prototypes from frozen CLIP text embeddings. The retrieved prototypes are fused into the content branch to enhance part-level semantic consistency.

Query Confidence Estimation. Since the number of dual queries Nq defines an upper bound on articulated components, some queries may correspond to invalid part hypotheses. To address this, we predict a confidence score ci ∈ [0,1] from the refined content embedding QLc,i of the i-th query, indicating the reliability of its part hypothesis. During training, confidence supervision is derived

from Hungarian matching between predicted part masks and ground-truth components. Matched queries are assigned confidence targets proportional to their mask overlap, while unmatched queries are treated as null hypotheses with zero confidence. At inference time, queries with confidence below a threshold are discarded, allowing the model to automatically determine the number of parts.

#### 3.5 Kinematic Estimator

Given QLp , QLc , part logits SLp , and point embedding H, Kinematic Estimator predicts articulation parameters and a kinematic tree.

#### Articulate Parameter Regressor. The refined dual queries are QLp ∈ RN

q×3 and QLc ∈ RN

q×d2, where QLp is interpreted as part centroid, i.e., the spatial center of an articulated component. Part mask is obtained by query–point matching. Specifically, we compute the affinity between content queries and point features H × RM×d

2: mm = QLc H⊤, mm ∈ RN

q×M. (3)

Each row of mm indicates the soft assignment of surface points to a queryinduced articulated part.

Joint parameters are regressed from query-level representations. We integrate QLp , QLc , and part-level features H into a unified per-query representation, which is processed by lightweight MLP heads to predict physically interpretable joint parameters:

#### – Joint type mt ∈ RN

q×Nt, where Nt denotes the number of predefined motion categories (e.g., fixed, revolute, prismatic, and continuous);

#### – Joint axis ma ∈ RN

q×3, representing the unit direction vector of the predicted joint axis;

#### – Joint pivot mo ∈ RN

q×3, denoting the pivot point through which the motion axis passes;

#### – Joint limits ml ∈ RN

q×2, parameterized by a center and a symmetric span.

Kinematic Tree Predictor. The articulated objects are modeled as treestructured kinematic graphs. We use i ∈ {1,...,Nq} to index queries. The part logits are given by SLp ∈ RN

q×Nc, and the predicted category distribution of part i is denoted as si ∈ RN

c.

Pairwise affinity is computed to model potential parent–child relations among predicted parts. For each ordered pair (i,j), we compute a semantic attachment score using a learnable compatibility matrix C ∈ RN

c×Nc:

Si,j = s⊤i Csj. (4)

This bilinear formulation captures category-level attachment priors in a datadriven manner. To obtain the probability that part j serves as the parent of part i, we normalize the scores over all candidate parents:

P(j|i) = Softmax(Si,j). (5)

Parent assignment selects, for each part i, the parent with the highest attachment probability P(j|i), optionally including a learnable root node as a candidate parent. To ensure a valid kinematic hierarchy, we enforce a single-root, cycle-free constraint during structure construction.

### 4 Experiments

#### 4.1 Evaluation Protocol

Dataset. We use PartNet-Mobility [51] as the benchmark, which contains approximately 2K articulated objects with part-level geometry and joint annotations, covering fixed, prismatic, revolute, and continuous joints. We adopt two evaluation splits: (1) a 7-category setting following SINGAPO [24] (Storage, Table, Refrigerator, Dishwasher, Oven, Washer, and Microwave), and (2) a full 46-category setting following PhysX-Anything [4] to evaluate large-scale multiclass generalization.

Metrics. Following FreeArt3D [7], we evaluate both motion-aware geometric reconstruction quality and kinematic prediction accuracy.

Geometric reconstruction quality includes four metrics: Chamfer Distance (CD), F-Score, PSNR, and CLIP similarity. For each shape, we uniformly sample six articulation states along the predicted motion range and generate the corresponding meshes using the estimated joint parameters. All metrics are computed at each state and averaged across states. Predicted meshes are aligned with ground-truth meshes using [27] prior to evaluation. CD and F-Score (threshold 0.05) are computed on 100k uniformly sampled surface points per aligned mesh. For appearance, we render both predicted and ground-truth meshes at the six articulation states, sampling 10 random viewpoints per state from a unit sphere (60 images in total), and compute the average PSNR and CLIP similarity [41] using ViT-L/14@336px.

Kinematic prediction accuracy includes three metrics: Type Accuracy, Axis Direction Error, and Pivot Distance Error. Predicted parts are first matched to ground-truth parts with bipartite matching to establish one-to-one correspondence. Type Accuracy measures joint type classification correctness. Axis Direction Error eaxis is defined as the angular deviation between predicted and ground-truth motion axes ap and ag for both revolute and prismatic joints:

ap · ag ∥ap∥2∥ag∥2

,arccos −ap · ag ∥ap∥2∥ag∥2

. (6)

eaxis = min arccos

Pivot Distance Error epivot measures the distance between the predicted joint pivot op to the ground-truth pivot og:

epivot = |(op − og) · (ap × ag)| |ap × ag|

. (7)

All geometric quantities are evaluated in normalized object coordinates.

- Table 1: Quantitative comparison on the PartNet-Mobility dataset. CD is scaled by ×10−2. Type Acc. (%) denotes joint type classification accuracy. Axis Err. (rad) represents joint axis direction error.. Pivot Err. is joint pivot distance error in normalized object coordinates. Best results are highlighted in bold.

Geometry Kinematics

Method

CD ↓ F-Score ↑ PSNR ↑ CLIP ↑ Type Acc. ↑ Axis Err. ↓ Pivot Err. ↓ Partial 7 classes

URDFormer [8] 4.73 0.275 12.43 0.845 35.22 1.324 0.404 SINGAPO [24] 1.26 0.572 15.22 0.870 77.12 0.493 0.201

- MonoArt 0.77 0.728 17.55 0.926 88.26 0.209 0.085

All 46 classes

ArtAny [18] 2.07 0.514 16.44 0.866 43.32 0.440 0.347 PhysXGen [3] 3.06 0.501 16.38 0.859 46.82 0.941 0.208 PhysXAny [4] 1.88 0.531 17.07 0.880 63.35 0.289 0.173

- MonoArt 1.25 0.670 18.55 0.907 67.47 0.423 0.108

#### 4.2 Implementation Details

Hyperparameters. We sample M = 100,000 surface points from the reconstructed mesh for part-aware reasoning. The structured voxel latent resolution is Nz = 64. The tri-plane resolution is Nt = 128 with feature dimension d1 = 8, and the lifted point embedding dimension is d2 = 448. We use Nq = 100 dual queries and a 6-layer dual-query motion decoder (L = 6).

Training Procedure. We adopt a four-phase training strategy: 1) Warm up Part-Aware Semantic Reasoner with triplet supervision to learn motion-aware part embeddings. 2) Freeze Part-Aware Semantic Reasoner and train the dualquery initialization branch using object-category supervision. 3) Jointly optimize Part-Aware Semantic Reasoner, Dual-query Motion Decoder, and Articulation Parameter Regressor. 4) Freeze all preceding modules and train Kinematic Tree Predictor to model parent–child relations. Additional training hyperparameters and loss configurations are provided in the Appendix.

#### 4.3 Main Results

We compare MonoArt with state-of-the-art monocular articulated object reconstruction methods, including URDFormer [8], SINGAPO [24], ArticulateAnything [18], PhysXGen [3], and PhysXAnything [4]. For retrieval-based baselines (Articulate-Anything and SINGAPO), we remove ground-truth test shapes from their retrieval database to ensure fair comparison and avoid data leakage.

PartNet-Mobility Benchmark. On PartNet-Mobility [51], MonoArt achieves the best overall performance in both geometry reconstruction and kinematic prediction. For the partial 7 classes, it substantially outperforms prior methods with clear margins in reconstruction quality and articulation estimation, particularly showing large gains in joint type accuracy and significant reductions in axis and

[Figure 38]

###### Input SINGAPO ArtAny PhysXAny MonoArt

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

[Figure 61]

[Figure 62]

No Output

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

No Output

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

No Output

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

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

- Fig. 3: Qualitative results on the test set of PartNet-Mobility. ArtAny and PhysXAny denote Articulate-Anything and PhysXAnything, respectively. For each object, we show the reconstructed geometry under two sampled articulated states.

pivot errors. On the full 46 classes, MonoArt remains consistently superior, delivering the strongest overall reconstruction fidelity while achieving the highest joint type accuracy and the lowest pivot error. Notably, it reduces pivot error by more than 40%, demonstrating robust structural reasoning across diverse articulated categories. Qualitative results in Fig. 3 further show that MonoArt produces more faithful geometry and more accurate joint predictions, leading to more plausible articulated motion.

Real-World Generalization. To evaluate real-world generalization, we collect approximately 100 in-the-wild images from the Internet using category keywords, covering common everyday articulated objects. As shown in Fig. 4, MonoArt produces coherent geometry and plausible articulation across diverse real-world

###### Input SINGAPO ArtAny PhysXAny MonoArt

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

No Output

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

No Output

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

No Output

- Fig. 4: Qualitative results on in-the-wild images. ArtAny and PhysXAny denote Articulate-Anything and PhysXAnything, respectively. For each object, we show the reconstructed geometry under two sampled articulated states.

appearances, despite being trained primarily on synthetic data. We conduct a user study with 20 participants to evaluate geometric and kinematic quality of the generated articulations. Participants rate rendered videos on two 1–5 scales. Our method achieves the highest scores (4.63 / 4.37), outperforming PhysXAnything (3.34 / 3.12), SINGAPO (2.55 / 2.87), Articulate-Anything (2.72 /

- 2.60), PhysXGen (2.53 / 2.46), and URDFormer (1.37 / 1.49).

#### 4.4 Ablation Study

We perform ablations on MonoArt to study key design choices in Part-Aware Semantic Reasoner, Dual-Query Motion Decoder, and Kinematic Estimator. All experiments are conducted on PartNet-Mobility (46 classes), evaluating their impact on geometry and kinematics.

Part-Aware Semantic Reasoner. As shown in Table 2, we ablate the design of the Part-Aware Semantic Reasoner. Removing the reasoner significantly degrades both geometry and kinematics, especially joint type accuracy and pivot prediction, confirming its critical role in motion-aware part reasoning. In this variant, voxel features and lifted point embeddings are directly fused via trilinear interpolation followed by an MLP. To make the learned feature H part-aware, we supervise the reasoner with a triplet loss that enforces part-level feature separation. Replacing it with cross-entropy supervision leads to clear performance drops, and removing the loss further degrades results. Triplet supervision consistently achieves the best performance across all metrics, highlighting the importance for discriminative and motion-consistent part representations.

Dual-Query Motion Decoder. As shown in Table 3, we analyze key design choices of the Dual-Query Motion Decoder. Disabling Dual-Query Initialization (DQI) and randomly initializing Q0p and Q0c degrades both geometry and kinematics, showing the importance of informed query initialization. Applying residual updates to only one branch is suboptimal, while updating both position and content queries achieves the best performance, highlighting the need for

- Table 2: Ablation of Part-Aware Semantic Reasoner. CD is scaled by ×10−2. Type Acc. (%) is joint classification accuracy. Axis Err. (rad) and Pivot Err. denote axis direction and pivot distance errors. Note that “CE” and “Tri.” denote “Cross-Entropy” and “Triplet”, respectively.

Enabled Loss

Geometry Kinematics CD ↓ F-Score ↑ PSNR ↑ Type Acc. ↑ Axis Err. ↓ Pivot Err. ↓

✗ ✗ 1.74 0.626 17.96 24.72 0.549 0.237 ✓ ✗ 1.63 0.643 17.74 41.60 0.922 0.323 ✓ CE 1.49 0.648 17.71 57.74 1.029 0.302

✓ Tri. 1.25 0.670 18.55 67.47 0.423 0.108

- Table 3: Ablation of Dual-Query Motion Decoder. CD is scaled by ×10−2. Type Acc. (%) is joint classification accuracy. Axis Err. (rad) and Pivot Err. denote axis direction and pivot distance errors. Note that “DQI.” indicates whether DualQuery Initialization is enabled. “Res.” denotes residual updates in the position and/or content query branches. L is the number of refinement blocks.

DQI. Res. L

Geometry Kinematics CD ↓ F-Score ↑ PSNR ↑ Type Acc. ↑ Axis Err. ↓ Pivot Err. ↓

✗ Both 6 1.67 0.622 18.11 44.06 0.472 0.329 ✓ Qlp 6 1.73 0.640 17.81 66.41 0.523 0.181 ✓ Qlc 6 1.29 0.663 17.80 60.88 0.506 0.184

- ✓ Both 0 1.70 0.652 17.94 62.65 0.640 0.186
- ✓ Both 1 1.71 0.652 17.91 63.12 0.608 0.189

✓ Both 3 1.67 0.655 17.88 66.38 0.524 0.157 ✓ Both 6 1.25 0.670 18.55 67.47 0.423 0.108 ✓ Both 9 1.59 0.659 17.95 66.81 0.475 0.161

- Table 4: Ablation of Kinematic Estimator. CD is scaled by ×10−2. Type Acc. (%) is joint classification accuracy. Axis Err. (rad) and Pivot Err. denote axis direction and pivot distance errors.

Geometry Kinematics CD ↓ F-Score ↑ PSNR ↑ Type Acc. ↑ Axis Err. ↓ Pivot Err. ↓

QLp H

✗ ✓ 1.65 0.652 17.92 67.20 0.499 0.191 ✓ ✗ 2.35 0.573 18.11 27.14 0.882 0.283

###### ✓ ✓ 1.25 0.670 18.55 67.47 0.423 0.108

joint refinement. Increasing the number of refinement layers improves results up to L = 6, while a deeper model (e.g., L = 9) leads to performance degradation.

Kinematic Estimator. As shown in Table 4, we ablate the Kinematic Estimator. We regress the joint origin using a residual formulation, mo = QLp + ∆o. Directly predicting mo without this anchor degrades both geometry and kine-

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

- Fig. 5: Robot manipulation with generated articulated objects. MonoArt reconstructions are directly imported into IsaacSim for contact-rich interaction.

[Figure 133]

Input Object Instances Articulated Scene

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

- Fig. 6: Articulated scene reconstruction. MonoArt augments SAM3D [42] with object-level articulation recovery to produce articulated, operable 3D scenes.

matics, showing the benefit of centroid-based residual prediction. Excluding the point embedding H also leads to significant performance drops, indicating its importance for parameter regression. Using both QLp -based residual prediction and H achieves the best results.

#### 4.5 Applications

Robot Manipulation. MonoArt extends beyond static 3D reconstruction by inferring articulation axes, joint types, and motion limits that are directly usable for robotic control. These structured kinematic priors convert monocular observations into actionable parameters, enabling motion reasoning and interaction planning without manual joint annotation. To evaluate downstream utility, we import the articulated objects reconstructed from in-the-wild images in Fig. 4 into IsaacSim [36]. As shown in Fig. 5, the simulation-ready assets can be directly manipulated by a Franka robot arm for contact-rich tasks such as grasping and opening, without additional modeling. This demonstrates a practical realto-sim pipeline that produces physically plausible articulated models for robotic interaction and policy learning.

Articulated Scene Reconstruction. Methods such as MIDI [15] and SAM 3D [42] provide static scene reconstructions with per-object masks and 6D poses.

Building on these outputs, we reconstruct each masked object instance with MonoArt to recover both geometry and articulation parameters. The reconstructed articulated objects are then placed back into the scene using their estimated 6D poses, yielding a coherent articulated scene without additional manual modeling. As shown in Fig. 6, this simple object-level augmentation converts rigid scene reconstructions into functionally operable environments, where articulated objects preserve their kinematic structure while remaining consistent with the global layout.

#### 4.6 Discussion

Runtime. All timings are measured on a single NVIDIA A6000 GPU, excluding I/O time, and averaged over 100 runs. As shown in Fig. 1, recent methods require 229.9s (Articulate-Anything [18]), 256.8s (PhysXAnything [4]), 31.6s (PhysXGen [3]), 34.1s (URDFormer [8]), and 19.6s (SINGAPO [24]) per instance. In comparison, MonoArt requires 20.5 seconds per instance. Among this, 18.2s is spent on TRELLIS-based 3D reconstruction, while articulation reasoning and post-processing introduce only marginal overhead.

Limitations. While MonoArt demonstrates strong performance in articulated object reconstruction and motion reasoning, several limitations remain. MonoArt can struggle with very small parts attached to large objects (e.g., tiny buttons). Due to uniform point sampling over the entire shape, such components may receive only sparse coverage, making their features less distinctive and more prone to over-smoothing. Consequently, articulations under extreme scale imbalance can be difficult to reliably segment and parameterize. MonoArt also relies on learned structural priors over part–whole relationships, which may not fully generalize to objects with novel topologies or uncommon articulation patterns. For such unseen object configurations, the predicted motion parameters (MonoArt, axes or ranges) can be less accurate, even when part segmentation remains reasonable.

### 5 Conclusion

In this work, we present MonoArt, a unified framework for monocular articulated 3D reconstruction grounded in progressive structural reasoning. Instead of depending on multi-view supervision, retrieval libraries, or auxiliary video synthesis, MonoArt formulates monocular articulated reconstruction as a progressive structural reasoning process. By explicitly modeling geometry, part structure, and motion in a unified framework, it achieves accurate and efficient articulation inference without handcrafted motion priors or external pipelines. Extensive experiments demonstrate that MonoArt achieves state-ofthe-art performance in both reconstruction accuracy and inference speed on PartNet-Mobility. Beyond object-level reconstruction, the framework generalizes effectively to robotic manipulation and articulated scene reconstruction, highlighting its practical applicability.

### Acknowledgements

This study is supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOET2EP20221-0012, MOE-T2EP20223-0002), and by cash and in-kind contributions from NTU S-Lab and industry partner(s).

### References

- 1. Black, K., Brown, N., Driess, D., Esmail, A., Equi, M., Finn, C., Fusai, N., Groom, L., Hausman, K., Ichter, B., Jakubczak, S., Jones, T., Ke, L., Levine, S., Li-Bell, A., Mothukuri, M., Nair, S., Pertsch, K., Shi, L.X., Tanner, J., Vuong, Q., Walling,

A., Wang, H., Zhilinsky, U.: π0: A vision-language-action flow model for general robot control. In: RSS (2025) 2

- 2. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., Ibarz, J., Ichter, B., Irpan, A., Jackson, T., Jesmonth, S., Joshi, N.J., Julian, R., Kalashnikov, D., Kuang, Y., Leal,

I., Lee, K., Levine, S., Lu, Y., Malla, U., Manjunath, D., et al.: RT-1: robotics transformer for real-world control at scale. In: RSS (2023) 2

- 3. Cao, Z., Chen, Z., Pan, L., Liu, Z.: PhysX-3D: physical-grounded 3D asset generation. In: NeurIPS (2025) 9, 14
- 4. Cao, Z., Hong, F., Chen, Z., Pan, L., Liu, Z.: Physx-Anything: Simulation-ready physical 3D assets from single image. In: CVPR (2026) 1, 2, 3, 8, 9, 14, 24, 25
- 5. Chan, E.R., Lin, C.Z., Chan, M.A., Nagano, K., Pan, B., Mello, S.D., Gallo, O., Guibas, L.J., Tremblay, J., Khamis, S., Karras, T., Wetzstein, G.: Efficient geometry-aware 3D generative adversarial networks. In: CVPR (2022) 5
- 6. Chang, A.X., Funkhouser, T.A., Guibas, L.J., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., Xiao, J., Yi, L., Yu, F.: ShapeNet: An information-rich 3D model repository. arXiv 1512.03012 (2015) 3
- 7. Chen, C., Liu, I., Wei, X., Su, H., Liu, M.: FreeArt3D: Training-free articulated object generation using 3D diffusion. In: SIGGRAPH Asia (2025) 3, 8
- 8. Chen, Q., Walsman, A., Memmel, M., Mo, K., Fang, A., Fox, D., Gupta, A.: URDFormer: A pipeline for constructing articulated simulation environments from real-world images. In: RSS (2024) 2, 9, 14
- 9. Chen, X., Golovinskiy, A., Funkhouser, T.A.: A benchmark for 3D mesh segmentation. ACM TOG 28(3), 73 (2009) 3
- 10. Chen, Z., Tang, J., Dong, Y., Cao, Z., Hong, F., Lan, Y., Wang, T., Xie, H., Wu, T., Saito, S., Pan, L., Lin, D., Liu, Z.: 3DTopia-XL: scaling high-quality 3D asset generation via primitive diffusion. In: CVPR (2025) 2
- 11. Gao, D., Siddiqui, Y., Li, L., Dai, A.: MeshArt: Generating articulated meshes with structure-guided transformers. In: CVPR (2025) 3
- 12. Gao, M., Pan, Y., Gao, H., Zhang, Z., Li, W., Dong, H., Tang, H., Yi, L., Zhao, H.: PartRM: modeling part-level dynamics with large cross-state reconstruction model. In: CVPR (2025) 2
- 13. Guo, J., Xin, Y., Liu, G., Xu, K., Liu, L., Hu, R.: ArticulatedGS: self-supervised digital twin modeling of articulated objects using 3D gaussian splatting. In: CVPR

(2025) 2, 3

- 14. Hanocka, R., Hertz, A., Fish, N., Giryes, R., Fleishman, S., Cohen-Or, D.: MeshCNN: a network with an edge. ACM TOG 38(4), 90:1–90:12 (2019) 3

- 15. Huang, Z., Guo, Y., An, X., Yang, Y., Li, Y., Zou, Z., Liang, D., Liu, X., Cao, Y., Sheng, L.: MIDI: multi-instance diffusion for single image to 3D scene generation. In: CVPR (2025) 13
- 16. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A.C., Lo, W., Dollár, P., Girshick, R.B.: Segment anything. In: ICCV (2023) 3
- 17. Lai, Z., Zhao, Y., Liu, H., Zhao, Z., Lin, Q., Shi, H., Yang, X., Yang, M., Yang, S., Feng, Y., Zhang, S., Huang, X., Luo, D., Yang, F., Yang, F., Wang, L., Liu, S., Tang, Y., Cai, Y., He, Z., Liu, T., Liu, Y., Jiang, J., Linus, Huang, J., Guo, C.: Hunyuan3D 2.5: towards high-fidelity 3D assets generation with ultimate details. arXiv 2506.16504 (2025) 2
- 18. Le, L., Xie, J., Liang, W., Wang, H., Yang, Y., Ma, Y.J., Vedder, K., Krishna, A., Jayaraman, D., Eaton, E.: Articulate-Anything: automatic modeling of articulated objects via a vision-language foundation model. In: ICLR (2025) 1, 2, 3, 9, 14, 24, 25
- 19. Lei, J., Deng, C., Shen, W.B., Guibas, L.J., Daniilidis, K.: NAP: neural 3D articulated object prior. In: NeurIPS (2023) 3
- 20. Li, L.H., Zhang, P., Zhang, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J., Chang, K., Gao, J.: Grounded language-image pre-training. In: CVPR (2022) 3
- 21. Li, R., Zheng, C., Rupprecht, C., Vedaldi, A.: Puppet-Master: scaling interactive video generation as a motion prior for part-level dynamics. In: ICCV (2025) 2
- 22. Li, X., Yang, J., Zhang, F.: Laplacian mesh transformer: Dual attention and topology aware network for 3D mesh classification and segmentation. In: ECCV (2022) 3
- 23. Li, Y., Bu, R., Sun, M., Wu, W., Di, X., Chen, B.: PointCNN: Convolution on x-transformed points. In: NeurIPS (2018) 3
- 24. Liu, J., Iliash, D., Chang, A.X., Savva, M., Amiri, A.M.: SINGAPO: single image controlled generation of articulated parts in objects. In: ICLR (2025) 1, 2, 3, 8, 9, 14, 24, 25
- 25. Liu, J., Mahdavi-Amiri, A., Savva, M.: PARIS: part-level reconstruction and motion analysis for articulated objects. In: ICCV (2023) 2, 3
- 26. Liu, M., Uy, M.A., Xiang, D., Su, H., Fidler, S., Sharp, N., Gao, J.: PARTFIELD: learning 3D feature fields for part segmentation and beyond. In: ICCV (2025) 4
- 27. Liu, M., Shi, R., Chen, L., Zhang, Z., Xu, C., Wei, X., Chen, H., Zeng, C., Gu, J., Su, H.: One-2-3-45++: Fast single image to 3D objects with consistent multi-view generation and 3D diffusion. In: CVPR (2024) 8
- 28. Liu, M., Zhu, Y., Cai, H., Han, S., Ling, Z., Porikli, F., Su, H.: PartSLIP: Low-shot part segmentation for 3D point clouds via pretrained image-language models. In: CVPR (2023) 3
- 29. Liu, X., Sun, X., Xie, H., Li, Z., Li, R., Zhang, S.: Multi-view consistent 3D panoptic scene understanding. In: AAAI (2025) 3
- 30. Liu, X., Xie, H., Zhang, S., Yao, H., Ji, R., Nie, L., Tao, D.: 2D semantic-guided semantic scene completion. IJCV 133(3), 1306–1325 (2025) 3
- 31. Lu, R., Liu, Y., Tang, J., Ni, J., Wang, Y., Wan, D., Zeng, G., Chen, Y., Huang, S.: DreamArt: generating interactable articulated objects from a single image. In: SIGGRAPH Asia (2025) 2
- 32. Ma, C., Li, Y., Yan, X., Xu, J., Yang, Y., Wang, C., Zhao, Z., Guo, Y., Chen, Z., Guo, C.: P3-SAM: native 3D part segmentation. arXiv 2509.06784 (2025) 4
- 33. Ma, Z., Yue, Y., Gkioxari, G.: Find any part in 3D. In: ICCV (2025) 4

- 34. Mandi, Z., Weng, Y., Bauer, D., Song, S.: Real2Code: reconstruct articulated objects via code generation. In: ICLR (2025) 2
- 35. Mo, K., Zhu, S., Chang, A.X., Yi, L., Tripathi, S., Guibas, L.J., Su, H.: PartNet: A large-scale benchmark for fine-grained and hierarchical part-level 3D object understanding. In: CVPR (2019) 3
- 36. NVIDIA: Isaac Lab: A GPU-accelerated simulation framework for multi-modal robot learning. arXiv 2511.04831 (2025) 13
- 37. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P., Li, S., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jégou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. TMLR 2024 (2024) 3
- 38. Qi, C.R., Su, H., Mo, K., Guibas, L.J.: PointNet: Deep learning on point sets for 3D classification and segmentation. In: CVPR (2017) 3
- 39. Qi, C.R., Yi, L., Su, H., Guibas, L.J.: PointNet++: Deep hierarchical feature learning on point sets in a metric space. In: NIPS (2017) 3
- 40. Qian, G., Li, Y., Peng, H., Mai, J., Hammoud, H., Elhoseiny, M., Ghanem, B.: PointNeXt: Revisiting pointnet++ with improved training and scaling strategies. In: NeurIPS (2022) 3
- 41. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021) 3, 8
- 42. SAM, Chen, X., Chu, F., Gleize, P., Liang, K.J., Sax, A., Tang, H., Wang, W., Guo, M., Hardin, T., Li, X., Lin, A., Liu, J., Ma, Z., Sagar, A., Song, B., Wang, X., Yang, J., Zhang, B., Dollár, P., Gkioxari, G., Feiszli, M., Malik, J.: SAM 3D: 3Dfy anything in images. arXiv 2511.16624 (2025) 13
- 43. Schroff, F., Kalenichenko, D., Philbin, J.: FaceNet: A unified embedding for face recognition and clustering. In: CVPR (2015) 6
- 44. Song, C., Wei, J., Foo, C.S., Lin, G., Liu, F.: REACTO: reconstructing articulated objects from a single video. In: CVPR (2024) 3
- 45. Tang, G., Zhao, W., Ford, L., Ben-Haim, D., Zhang, P.: Segment any mesh: Zero-shot mesh part segmentation via lifting segment anything 2 to 3D. arXiv 2408.13679 (2024) 4
- 46. Tang, J., Ren, J., Zhou, H., Liu, Z., Zeng, G.: DreamGaussian: generative gaussian splatting for efficient 3D content creation. In: ICLR (2024) 2
- 47. Umam, A., Yang, C., Chen, M., Chuang, J., Lin, Y.: PartDistill: 3D shape part segmentation by vision-language model distillation. In: CVPR (2024) 4
- 48. Wei, F., Chabra, R., Ma, L., Lassner, C., Zollhöfer, M., Rusinkiewicz, S., Sweeney, C., Newcombe, R.A., Slavcheva, M.: Self-supervised neural articulated shape and appearance models. In: CVPR (2022) 2, 3
- 49. Wen, B., Xie, H., Chen, Z., Hong, F., Liu, Z.: 3D scene generation: A survey. arXiv 2505.05474 (2025) 2
- 50. Weng, Y., Wen, B., Tremblay, J., Blukis, V., Fox, D., Guibas, L.J., Birchfield, S.: Neural implicit representation for building digital twins of unknown articulated objects. In: CVPR (2024) 2, 3
- 51. Xiang, F., Qin, Y., Mo, K., Xia, Y., Zhu, H., Liu, F., Liu, M., Jiang, H., Yuan, Y., Wang, H., Yi, L., Chang, A.X., Guibas, L.J., Su, H.: SAPIEN: A simulated part-based interactive environment. In: CVPR (2020) 1, 8, 9
- 52. Xiang, J., Lv, Z., Xu, S., Deng, Y., Wang, R., Zhang, B., Chen, D., Tong, X., Yang, J.: Structured 3D latents for scalable and versatile 3D generation. In: CVPR (2025) 2, 4

- 53. Xie, H., Chen, Z., Hong, F., Liu, Z.: CityDreamer: compositional generative model of unbounded 3D cities. In: CVPR (2024) 2
- 54. Xie, H., Chen, Z., Hong, F., Liu, Z.: Compositional generative model of unbounded 4D cities. IEEE TPAMI 48(1), 312–328 (2026) 2
- 55. Xie, H., Wen, B., Zheng, J., Chen, Z., Hong, F., Diao, H., Liu, Z.: DynamicVLA: A vision-language-action model for dynamic object manipulation. arXiv 2601.22153

(2026) 2

- 56. Xie, H., Yao, H., Zhou, S., Mao, J., Zhang, S., Sun, W.: GRNet: gridding residual network for dense point cloud completion. In: ECCV (2020) 5
- 57. Xue, Y., Chen, N., Liu, J., Sun, W.: ZeroPS: High-quality cross-modal knowledge transfer for zero-shot 3D part segmentation. In: 3DV (2025) 3
- 58. Yang, Y., Jia, B., Zhi, P., Huang, S.: PhyScene: physically interactable 3D scene synthesis for embodied AI. In: CVPR (2024) 2
- 59. Yang, Y., Huang, Y., Guo, Y., Lu, L., Wu, X., Lam, E.Y., Cao, Y., Liu, X.: SAMPart3D: Segment any part in 3D objects. arXiv 2411.07184 (2024) 4
- 60. Yi, L., Kim, V.G., Ceylan, D., Shen, I., Yan, M., Su, H., Lu, C., Huang, Q., Sheffer, A., Guibas, L.J.: A scalable active framework for region annotation in 3D shape collections. ACM TOG 35(6), 210:1–210:12 (2016) 3
- 61. Zhao, H., Jiang, L., Jia, J., Torr, P.H.S., Koltun, V.: Point transformer. In: ICCV

(2021) 3

- 62. Zhou, Y., Gu, J., Li, X., Liu, M., Fang, Y., Su, H.: PartSLIP++: Enhancing lowshot 3D part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv 2312.03015 (2023) 3
- 63. Zhu, Z., Wan, L., Xu, R., Zhang, Y., Chen, H., Dou, Z., Lin, C., Liu, Y., Wei, M.: PartSAM: A scalable promptable part segmentation model trained on native 3D data. In: ICLR (2026) 4

### A Implementation Details

#### A.1 Loss Functions

We supervise the model with five training objectives: a triplet loss ℓtriplet for the Part-Aware Semantic Reasoner, a mask loss ℓmask and a confidence loss ℓscore for Dual-Query Motion Decoder, a motion loss ℓmotion for articulation parameter regression, and a structure loss ℓstruct for kinematic tree prediction.

To supervise the fixed set of predicted queries, we establish a one-to-one correspondence between predicted queries and ground-truth articulated parts using Hungarian bipartite matching. The matching cost is computed from mask similarity between the predicted part mask and the ground-truth part mask, using a weighted combination of binary cross-entropy and Dice cost. The resulting assignment identifies the queries matched to ground-truth articulated parts, while the remaining queries are treated as null predictions.

Part-Aware Semantic Reasoner. The Part-Aware Semantic Reasoner is supervised by a triplet contrastive loss to learn discriminative motion-aware part embeddings. For a triplet (ha,hb,hc), where (ha,hb) belong to the same articulated part and hc belongs to a different part, the triplet loss is defined as

- 1

- 2

ℓtriplet = −

sab sab + sac

sab sab + sbc

log

+ log

, (8)

where the pairwise similarity score is defined as

sij = exp

cos(hi,hj) τ

, (9)

and τ is a learnable temperature parameter. Dual-Query Motion Decoder. The Dual-Query Motion Decoder is supervised by three terms: a mask loss for part segmentation, a confidence loss for query reliability estimation, and an auxiliary object-category classification loss for query initialization. Mask loss is defined for each query matched to a ground-truth part, where the predicted mask mq is supervised against its matched ground-truth mask mgtq using focal and Dice losses:

ℓmask = λfocalℓfocal + λdiceℓdice. (10)

Confidence loss is applied to the predicted confidence score cˆq, which indicates the reliability of the predicted part mask. The supervision target is defined as the mask IoU between the predicted mask and the matched ground-truth mask:

uq = IoU(mq,mgtq ). (11) Following the Quality Focal Loss formulation, the confidence loss is

ℓscore = |σ(ˆcq) − uq|β · BCE(ˆcq,uq), (12)

where σ(·) denotes the sigmoid function. Queries not matched to any groundtruth part are assigned zero targets. During inference, queries with predicted confidence scores below 0.5 are filtered out, and the remaining queries are retained as valid articulated part hypotheses.

Auxiliary object-category classification is introduced to supervise the dualquery initialization branch:

ℓobj = CE(yˆobj,yobjgt ). (13) This auxiliary supervision stabilizes dual-query initialization in early training.

Articulation Parameter Regressor. The Articulation Parameter Regressor is supervised by a motion loss that includes joint type, axis direction, motion origin, and motion limit regression:

ℓmotion = λtℓtype + λdℓdir + λoℓorigin + λlℓlimit. (14) where, the joint type loss is

ℓtype = CE(ˆtq,tgtq ). (15) The axis direction loss is defined using unsigned cosine similarity:

The motion origin loss is

ℓdir = 1 −

a ˆq ∥aˆq∥

agtq ∥agtq ∥

·

. (16)

ℓorigin = ∥oˆq − ogtq ∥1. (17)

For joints with bounded motion ranges, we use a center–span parameterization. Let lmin and lmax denote the lower and upper motion bounds, and define

lmax − lmin 2

lmin + lmax 2

. (18)

c =

, s =

The network predicts cˆq and sˆq, and the corresponding loss is ℓlimit = ∥cˆq − cgtq ∥1 + ∥sˆq − sgtq ∥1. (19)

Kinematic Tree Predictor. The Kinematic Tree Predictor is supervised by a structure loss for parent prediction, computed over queries matched to groundtruth articulated parts:

1 Nmatch q matched

CE(ˆsq,pgtq ), (20)

ℓstruct =

where Nmatch denotes the number of queries matched to ground-truth articulated parts, ˆsq denotes the predicted parent probability distribution, and pgtq is the ground-truth parent index.

Enabled 

Enabled ✓

Enabled ✓

Enabled ✓

Input Loss Tri

Loss CE

Loss 

Loss 

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

- Fig. 7: Visualization of point features H from the Part-Aware Semantic Reasoner. Points are colored by PCA projection of the features. “Enabled” indicates whether the reasoner is used, and “Loss” specifies the supervision type. Note that “CE” and “Tri.” denote “Cross-Entropy” and “Triplet”, respectively.

#### A.2 Training Details

Training Procedure. Training is performed on four NVIDIA A100 GPUs with a batch size of 1 per GPU (effective batch size of 4) and requires approximately six days. The training procedure consists of four stages.

- Stage I warms up the Part-Aware Semantic Reasoner for 100 epochs using only the triplet loss while freezing the TRELLIS backbone and all downstream modules:

- ℓstage1 = ℓtriplet. (21)

Stage II freezes the semantic reasoner and trains the dual-query initialization branch for 20 epochs with object-category supervision:

- ℓstage2 = ℓobj. (22)

- Stage III jointly optimizes the Part-Aware Semantic Reasoner, Dual-Query Motion Decoder, and Articulation Parameter Regressor for 100 epochs using

ℓstage3 = λtripletℓtriplet + λmaskℓmask + λscoreℓscore + λmotionℓmotion. (23) The weight of ℓmotion is linearly increased during the first 40 epochs.

- Stage IV freezes all preceding modules and trains the Kinematic Tree Predictor for 30 epochs using

ℓstage4 = ℓstruct. (24)

Loss Weights. Unless otherwise specified, we use λtriplet = 0.2, λmask = 1.0, λscore = 1.0, and λmotion = 1.0. For mask supervision, we use λfocal = 1.0 and λdice = 1.0 with focal parameter γ = 2. For confidence score supervision, we use β = 2. For motion regression, we use λt = 1.0, λd = 1.0, λo = 1.0, and λl = 1.0. For Hungarian matching, we use equal weights for BCE and Dice costs.

Input SINGAPO Articulate-Anything

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

GT PhysX-Anything MonoArt

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Input SINGAPO Articulate-Anything

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

GT PhysX-Anything MonoArt

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

N/A

- Fig. 8: Representative Failure Cases. (Top) an object containing an extremely small component (printer button). (Bottom) an unseen articulated object (bicycle).

Optimizer. All stages use the AdamW optimizer with weight decay 0.01. Mixed-precision training (FP16) is used to reduce memory usage and accelerate training. Gradients are clipped with a maximum norm of 0.1.

Learning Rate Schedule. The learning rate is warmed up for 10 epochs from 1% of the base rate (5 × 10−5), followed by cosine annealing to 10−6.

### B Additional Ablation Study Results

#### B.1 Part-Aware Semantic Reasoner

- Fig. 7 qualitatively complements the ablation results in Table 2 by visualizing the learned point features under different variants of Part-Aware Semantic Reasoner (PASR). Without PASR, the features show limited part discrimination. Adding PASR improves structural organization, while different supervision strategies further affect feature separability. Cross-entropy supervision yields partially separated clusters, whereas triplet supervision produces compact and well-separated part features that better align with articulated components.

- C More Discussion on Limitations
- Fig. 8 presents two representative failure cases, revealing key challenges in monocular articulated reconstruction. 1) Extremely small components (e.g., the printer button) are difficult to capture under uniform sampling and limited spatial resolution, resulting in inaccurate part identification and placement. 2) Unseen articulated categories (e.g., bicycle) lead to incorrect articulation recovery due to the large domain gap from the training data.

### D More Qualitative Results

We present additional qualitative comparisons on articulated object reconstruction in Fig. 9 and 10. Across diverse objects and motion ranges, MonoArt reconstructs more consistent geometry and more plausible articulated motion than prior methods.

Input Motion States Articulation

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

MonoArtPhysXAnyPhysXAnySINGAPOSINGAPOMonoArtArtAnyArtAny

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

[Figure 229]

[Figure 230]

[Figure 231]

Input Motion States Articulation

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

PhysXAnyPhysXAnySINGAPOSINGAPOMonoArtMonoArtArtAnyArtAny

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

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

No Output

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

