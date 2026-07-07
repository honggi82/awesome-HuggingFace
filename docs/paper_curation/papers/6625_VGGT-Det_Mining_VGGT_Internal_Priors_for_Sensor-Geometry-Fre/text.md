# arXiv:2603.00912v1[cs.CV]1Mar2026

### VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection

Yang Cao1* Feize Wu1,3∗† Dave Zhenyu Chen2 Yingji Zhong1 Lanqing Hong2 Dan Xu1‡ 1Hong Kong University of Science and Technology 2Huawei 3Sun Yat-Sen University

##### Abstract

Current multi-view indoor 3D object detectors rely on sensor geometry that is costly to obtain—i.e., precisely calibrated multi-view camera poses—to fuse multi-view information into a global scene representation, limiting deployment in real-world scenes. We target a more practical setting: Sensor-Geometry-Free (SG-Free) multi-view indoor 3D object detection, where there are no sensor-provided geometric inputs (multi-view poses or depth). Recent Visual Geometry Grounded Transformer (VGGT) shows that strong 3D cues can be inferred directly from images. Building on this insight, we present VGGT-Det, the first framework tailored for SG-Free multi-view indoor 3D object detection. Rather than merely consuming VGGT predictions, our method integrates VGGT encoder into a transformer-based pipeline. To effectively leverage both the semantic and geometric priors from inside VGGT, we introduce two novel key components: (i) Attention-Guided Query Generation (AG): exploits VGGT attention maps as semantic priors to initialize object queries, improving localization by focusing on object regions while preserving global spatial structure; (ii) Query-Driven Feature Aggregation (QD): a learnable See-Query interacts with object queries to ‘see’ what they need, and then dynamically aggregates multi-level geometric features across VGGT layers that progressively lift 2D features into 3D. Experiments show that VGGT-Det significantly surpasses the best-performing method in the SG-Free setting by 4.4 and 8.6 mAP@0.25 on ScanNet and ARKitScenes, respectively. Ablation study shows that VGGT’s internally learned semantic and geometric priors can be effectively leveraged by our AG and QD. Source code and pre-trained models are available at the GitHub project page.

##### 1. Introduction

Multi-view indoor 3D object detection is a fundamental task with broad applications in robotics and augmented reality.

*Equal contribution †Work done during an internship at HKUST ‡Corresponding author: danxu@cse.ust.hk

[Figure 1]

[Figure 2]

No Multi-view Poses or Depth from Sensor

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

More Practical

[Figure 7]

[Figure 8]

Mining VGGT internal priors effectively

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Semantic Prior Geometry Prior

VGGT Encoder

VGGT-Det Decoder

[Figure 18]

SG-Free Detection Results

46.9

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

42.5

41.2

mAP(%)

[Figure 24]

[Figure 25]

40.6

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

35.2

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

ImVoxelNet FCAF3D NeRF-Det MVSDet VGGT-Det

Figure 1. To achieve and improve Sensor-Geometry-Free (SG-Free) multi-view indoor 3D object detection, VGGT-Det effectively leverages the internal semantic and geometric priors from VGGT [41], rather than merely consuming its predictions. VGGT-Det significantly surpasses competitive methods in the SG-Free setting.

Current methods [4, 32, 36, 46, 47] predominantly rely on sensor-derived geometric inputs, i.e., precisely calibrated multi-view camera poses or depth. Although these methods achieve strong performance, their reliance on expensive and often inaccessible sensor-derived geometric inputs [3, 5] severely limits scalability and real-world deployment.

We instead consider a more practical setting: performing indoor 3D detection from multi-view images without sensor-derived geometric inputs. We refer to this as SensorGeometry-Free (SG-Free) multi-view indoor 3D object detection. This setting is highly challenging because it eliminates the sensor-derived multi-view camera poses and depth. Recent advances in feed-forward 3D reconstruction have shown that 3D structure can be inferred directly from unposed 2D images [15, 37, 41, 43, 49, 50]. These develop-

ments open new opportunities for SG-Free multi-view indoor 3D object detection. Beyond providing low-cost estimates of scene geometry from unposed images, the feed-forward models also encode strong learned priors that improve 3D reasoning capabilities that are crucial for reliable detection.

In this paper, we present VGGT-Det, a framework for Sensor-Geometry-Free (SG-Free) multi-view indoor 3D object detection, built upon the Visual Geometry Grounded Transformer (VGGT) [41], a representative feed-forward 3D reconstruction model. As shown in Fig. 1, rather than merely consuming VGGT’s predictions, our method integrates a pretrained VGGT encoder in a transformer-based pipeline. To effectively leverage the learned knowledge from inside VGGT, we conduct an in-depth analysis of its intermediate representations. We find that the attention maps in the VGGT encoder capture rich semantic information despite the model not being explicitly trained for semantics. Motivated by this interesting observation, we propose Attention-Guided Query Generation (AG), which exploits VGGT attention maps to initialize and steer object queries, improving localization by focusing on semantic regions while preserving global spatial structure. Furthermore, since VGGT progressively lifts 2D features into 3D across its layers—each encoding distinct levels of geometric abstraction—we introduce QueryDriven Feature Aggregation (QD). This module incorporates a learnable See-Query that interacts with object queries to ‘see’ what information is required and then dynamically aggregates multi-level geometric features from VGGT layers, effectively capturing hierarchical representations. Together, these designs unlock more of VGGT’s potential for SGFree multi-view indoor 3D detection. Extensive experiments show that VGGT-Det significantly outperforms strong methods in the SG-Free setting, and ablations confirm that the proposed AG and QD effectively leverage the learned priors inside VGGT. Our contributions are summarized as follows:

- • We introduce a more practical setting—Sensor-GeometryFree (SG-Free) multi-view indoor 3D detection—which removes the need for sensor-derived geometric inputs (i.e., multi-view camera poses or depth maps). To the best of our knowledge, this is the first work to explicitly target this challenging setting. Building on this setting, we introduce VGGT-Det, the first transformer-based SG-Free 3D object detection framework.
- • We propose Attention-Guided Query Generation (AG), which leverages semantic priors encoded in VGGT encoder attention to initialize object queries. These queries concentrate on object regions while preserving the global spatial structure, improving the localization.
- • We develop Query-Driven Feature Aggregation (QD), which introduces a learnable See-Query that interacts with object queries to ‘see’ their needs and dynamically aggregates multi-level geometric features accordingly.
- • With the proposed approach, VGGT-Det achieves signifi-

cant gains, consistently outperforming the best-performing method in the SG-Free setting by 4.4 and 8.6 mAP@0.25 points on ScanNet and ARKitScenes, respectively.

##### 2. Related Work

Multi-view Outdoor 3D Object Detection. Current outdoor multi-view 3D object detection [7, 10, 42, 45] can be broadly divided into two lines. The first line projects 3D queries to 2D images for aggregating visual knowledge. Inspired by DETR [6], DETR3D [44] uses sparse 3D object queries to index multi-view 2D features through camera parameters. The PETR series [24, 25] introduce a position embedding transformation that encodes 3D coordinate information into multi-view image features. The Sparse4D series [19–21] and SparseBEV [23] perform temporally aware detection by exploring sparse strategies to aggregate multi-frame features. BEVFormer [18] introduces a unified BEV representation learned with spatiotemporal transformers. The second line constructs BEV representations by lifting 2D features into 3D space. LSS [31] introduces an end-to-end multi-view architecture that ‘lifts’ each image to a feature frustum and ‘splats’ it into a BEV grid to learn a robust scene representation. BEVDet [13] utilizes standard modules but boosts performance via tailored data augmentation and improved NMS. BEVDepth [17] and BEVStereo [16] address the depth bottleneck using explicit depth supervision. Taking a step further from the above methods, CorrBEV [48] tackles occlusion through elegant design choices, such as auxiliary visual and language prototypes. Our work focuses on indoor detection. Unlike outdoor settings where cameras are rigidly mounted on vehicles, indoor cameras are usually handheld or frequently repositioned. This makes obtaining reliable sensor poses both costly and often inaccessible, motivating the introduction of the SG-Free setting for indoor scenes.

Multi-view Indoor 3D Object Detection. Present multiview indoor 3D object detection methods handle 3D object detection from images with multi-view poses [4, 36, 46, 47]. Notably, recent advances in Multi-View Stereo [51, 52], Neural Radiance Fields (NeRF) [29], and 3D Gaussian Splatting (3DGS) [14] have enhanced geometry recovery from posed multi-view images, enabling effective 3D detection that leverages 3D clues of voxel [12, 46] and splats [4]. ImVoxelNet [34] aggregates 2D features from multi-view images into 3D voxel volumes via unprojection in an end-toend manner. NeRF-Det [46] integrates multi-view geometric constraints derived from the NeRF module into the 3D object detection pipeline. MVSDet [47] replaces NeRF with plane sweep and uses probabilistic sampling with soft weighting to regularize depth, enabling accurate detection. Existing methods rely on sensor-derived geometric inputs (e.g., multi-view poses or depth), which severely limits their applicability in the real world. To address this, we tackle Sensor-GeometryFree (SG-Free) multi-view indoor 3D object detection and

###### Attention-Guided Query Generation Decoder Block

Object Queries

Updated Queries

Attn. + λ · Dis. Max

Attach to Points

[Figure 38]

High

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

DetectionHead

[Figure 51]

[Figure 52]

[Figure 53]

Cross-Attention

[Figure 54]

[Figure 55]

Self-Attention

[Figure 56]

[Figure 57]

Init.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

•••

Low

Selected Points

Select MaxPriority Point

Attention Maps

Compute Distances

Point Clouds

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

× K

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

Unposed Images

[Figure 83]

[Figure 84]

DPT Head Multi-Level VGGT Encoder Features

[Figure 85]

•••

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

See-Query

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

×

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

MLP

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Feature Weights

Aggregated Features

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Query-Driven Feature Aggregation

[Figure 126]

[Figure 127]

[Figure 128]

× L times

[Figure 129]

[Figure 130]

- Figure 2. Overview of the proposed Sensor-Geometry-Free (SG-Free) multi-view indoor 3D object detection framework, VGGT-Det. It is built upon the VGGT encoder [41], which extracts 3D-aware features from multi-view images. The decoder processes a set of object queries that cross-attend to the extracted features, and iteratively updates the queries for final detection. To effectively leverage both the semantic and geometric priors from inside VGGT, we carefully design two key components: Attention-Guided Query Generation (AG) and Query-Driven Feature Aggregation (QD). AG utilizes the semantic priors from the VGGT encoder’s attention to generate object queries, enabling these queries to focus on object regions while preserving the global spatial structure. Besides, QD introduces a learnable See-Query, which interacts with object queries via self-attention to ‘see’ their needs and dynamically aggregates multi-level geometric features accordingly.

propose VGGT-Det, which utilizes Attention-Guided Query Generation and Query-Driven Feature Aggregation to effectively mine internal semantic and geometric priors from VGGT rather than merely consuming its predictions.

(i.e., multi-view poses and depth). In the following, we will first introduce our basic backbone in Sec. 3.2, which is a transformer-based pipeline integrating the VGGT encoder [41]. To effectively leverage both the semantic and geometric priors from inside VGGT, we introduce two novel key components: (i) Attention-Guided Query Generation (AG) is proposed to effectively utilize the semantic prior from VGGT encoder attention, detailed in Sec. 3.3. (ii) Query-Driven Feature Aggregation (QD) is designed to adaptively aggregates multi-level geometric features according to the needs of object queries, which is introduced in Sec. 3.4.

Generalizable 3D Reconstruction. Given unposed multiview images, 3D reconstruction [11, 15] targets estimating geometry and camera poses of the images. The traditional pipeline for 3D reconstruction involves several sub-tasks, including keypoint detection [9, 28], matching [2, 22, 35, 38], multi-view stereo [40, 51–53], etc. Recently, DUSt3R [43] revolutionizes the paradigm and uses a single network to directly estimate the geometry of a target scene. MASt3R [15] inherits the paradigm with an auxiliary prediction head for the matching task. Both DUSt3R and MASt3R are limited to pair-wise inputs, which restricts context reasoning and leads to repeated network forward passes and time-consuming global coordinate alignment. Fast3R [50] improves performance by enabling a single forward pass for long sequences and eliminating the need for coordinate alignment. In addition to point maps, VGGT [41] predicts camera poses and other 3D-relevant attributes. In this paper, we present VGGT-Det, which effectively leverages priors within VGGT to advance SG-Free multi-view indoor 3D object detection.

###### 3.2. Basic Backbone

We design an encoder-decoder transformer [39] architecture as our backbone. Given a set of multi-view images {I1,I2,...,IV }, we employ the VGGT encoder [41] as the feature extractor to extract 3D-aware features. For each view, the VGGT encoder outputs a sequence of tokens, where each token corresponds to a spatial region or a patch of the image. The extracted tokens for all views are denoted as {T1,T2,...,TV }, where Ti ∈ RM×C is the token sequence for the i-th view, with M tokens per view and C as the token dimension. These tokens are concatenated along the token dimension to form a unified token representation:

##### 3. Method

Tconcat = [T1;T2;...;TV ] ∈ R(V ·M)×C. (1)

###### 3.1. Overview

Inspired by 3DETR [30], we adopt non-parametric query sampling to initialize object queries. Specifically, we apply farthest point sampling (FPS) to the point clouds Ppred predicted by VGGT. FPS selects a subset of K points {p1,p2,...,pK} ⊂ Ppred, where K is the number of object

An overview of our proposed sensor-geometry-free 3D detection framework is presented in Fig. 2. VGGT-Det takes multi-view images as inputs and outputs 3D detection results, which eliminates sensor-provided geometric inputs

[Figure 131]

[Figure 132]

[Figure 133]

queries falling in background regions rather than actual objects. Such misplacement of queries fundamentally compromises detection efficacy, as background-located queries provide minimal useful information for detection. Interestingly, we observed that the attention in the VGGT encoder inherently captures rich semantic information, as shown in the left column of Fig. 4, even though VGGT is not trained on semantic tasks. The observation motivates us to propose the Attention-Guided Query Generation (AG) that leverages the attention maps from the VGGT encoder to encourage object queries to locate at semantic object regions. The proposed AG not only leverages the semantic guidance from the VGGT encoder, but also considers the spatial diversity to make the sampled points fully cover the entire 3D space.

High

[Figure 134]

[Figure 135]

[Figure 136]

Select the First Point

[Figure 137]

[Figure 138]

Low

Attention Maps

[Figure 139]

[Figure 140]

Select Next Point,

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Normalize Attention

Compute Distances

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Attach to Points

[Figure 153]

Update Minimal Distances

Highest Attention, Firstly Selected

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Normalize Distances

[Figure 158]

Compute Priority

[Figure 159]

[Figure 160]

[Figure 161]

Formally, the point clouds reconstructed by VGGT are denoted as P ∈ RN×3, where N is the number of points, and each point is represented by its 3D coordinates. We denote the attention weights from the VGGT encoder as A ∈ RN, where each weight indicates the semantic importance of a point. The goal is to sample K query points {P1,P2,...,PK} from P, guided by both semantic information from the attention weights and spatial distribution. To ensure numerical stability and compatibility with distancebased features, the attention weights are normalized:

[Figure 162]

Selected Select Next Point,

[Figure 163]

[Figure 164]

Select the Max Priority Point

[Figure 165]

[Figure 166]

- Figure 3. Computation flow of Attention-Guided Query Generation.

queries. These sampled points are encoded with positional embeddings Epos ∈ RK×C to initialize object queries:

Q0 = Embed({p1,p2,...,pK}) + Epos. (2)

The object queries Q0 are then passed through a series of L transformer decoder layers. In each decoder layer, the queries first undergo query-to-query self-attention to exchange information among themselves as:

A − min(A) max(A) − min(A) + ϵ

, (6)

Anorm =

where ϵ is a small constant for numerical stability. Anorm ∈ RN. The index of the first query point I[1] is selected as the point with the highest attention score:

Qselfl = SelfAttention(Ql−1), (3)

where Ql−1 is the input to the l-th decoder layer, and Qselfl is the output after self-attention. Next, the queries attend to

I[1] = arg maxAnorm. (7)

the unified token representation Tconcat via cross-attention:

Following points are selected iteratively by a fused priority that balances semantic attention and spatial dispersion:

Ql = CrossAttention(Qselfl ,Tconcat). (4)

After passing through all L decoder layers, the updated object queries QL ∈ RK×C are fed into a detection head to generate object-level predictions, including class labels cˆ ∈ RK and bounding boxes bˆ ∈ RK×7:

Priority = Anorm + λdist · Dnorm, (8)

where λdist ∈ [0,1] is a trade-off coefficient that controls the relative influence of spatial dispersion, and Dnorm is the normalized minimum Euclidean distance between point P and the already-sampled set {PI[1],...,PI[k−1]}. Specifically, we let Dmin denote the minimum Euclidean distance from P to all the previously selected points:

{cˆ,bˆ} = DetectionHead(QL). (5)

By concatenating tokens across views, the model enables object queries to directly interact with the multi-view token representation. This pipeline effectively leverages the 3Daware features extracted by the VGGT encoder and dynamically refines object queries through the transformer decoder, enabling the sensor-geometry-free 3D object detection.

∥P − PI[j]∥2, (9)

Dmin = min

j∈{1,...,k−1}

where Dmin ∈ RN. The normalized distance is defined as:

###### 3.3. Attention-Guided Query Generation

Dmin − min(Dmin) max(Dmin) − min(Dmin) + ϵ

, (10)

Dnorm =

Since the point clouds predicted by VGGT [41] are dense reconstruction of the scene, it does not distinguish between object and background regions. Therefore, simply applying the FPS to uniformly select queries from the predicted point clouds may result in a significant number of object

where Dnorm ∈ RN. At each iteration, the point with the highest priority is subsequently selected:

Priority, (11)

I[k] = arg max

i/∈S

[Figure 167]

[Figure 168]

|AttentionWeight<br><br>High<br><br>Low<br><br>Next Selected Point<br><br>Selected Point<br><br>High Attention But Short Distance<br><br>Long Distance But Low Attention<br><br>Next Selected Point<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>Selected Point|
|---|

|[Figure 172]|
|---|

|[Figure 173]|
|---|

[Figure 174]

| |
|---|

[Figure 175]

AttentionWeight

|[Figure 176]|
|---|

|[Figure 177]|
|---|

| |
|---|

vs.

Attention From VGGT (w/ AG) (w/o AG)

Attention-Guided Query Generation

- Figure 4. The proposed Attention-Guided Query Generation (AG) is inspired by the interesting observation: attention maps from the VGGT encoder [41] exhibit a strong correlation with semantic content, even though VGGT is not explicitly trained for semantic tasks. For example, in the left column, object regions tend to receive higher attention weights. In the middle column, AG samples from VGGT-predicted point clouds under the guidance of attention weights and point distribution information. In the right column, compared to farthest point sampling without guidance (red points), the points sampled by AG (green points) are more concentrated in object regions (labeled by green boxes), resulting in more green points than red points in those areas. For clarity, we recommend viewing the figure in color and zooming in.

where S = {I[1],I[2],...,I[k − 1]} represents the index set of samples selected in previous iterations. In this step, as illustrated in the middle column Fig. 4, points with high attention scores and greater distances are selected. This design encourages object queries to focus on semantically meaningful object regions while preserving spatial diversity, ensuring coverage of the entire 3D space. The computation flow is shown in Fig. 3, which presents a step-by-step implementation of our proposed Attention-Guided Query Generation (AG). As shown in the right column of Fig. 4, while FPS tends to distribute sampling points indiscriminately across both foreground and background regions, AG strategically concentrates sampling queries on object regions by leveraging the internal semantic priors from VGGT, facilitating more effective 3D object detection.

followed by a softmax normalization, producing attention weights w ∈ RL over the L feature maps:

L

w = Softmax(MLP(qsee)), wi ≥ 0,

wi = 1. (12)

i=1

The aggregated feature Fagg is then computed as a weighted sum of the multi-level geometric encoded features:

L

wi · Fi, (13)

Fagg =

i=1

where Fagg ∈ RN×C serves as the key and value inputs to the decoder’s cross-attention block. Then, we present how the See-Query interacts with object queries and is updated. The See-Query token qsee plays a dual role in the decoder. First, it is concatenated with the object queries {q1,q2,...,qK}, forming a unified query set jointly for decoding:

###### 3.4. Query-Driven Feature Aggregation

Recognizing that the VGGT encoder [41] progressively transforms 2D image features into 3D representations across its layers, with each stage capturing distinct levels of geometric information, we propose a Query-Driven Feature Aggregation (QD) strategy. This approach introduces a learnable See-Query, which interacts with object queries to ‘see’ their requirements and dynamically aggregates multi-level features accordingly. Formally, let the VGGT encoder produce a set of feature maps from L intermediate layers, denoted as {F1,F2,...,FL}, where Fi ∈ RN×C is the feature map from the i-th layer, with N spatial elements and C feature dimensions. To aggregate these multi-level geometric features, we introduce a learnable See-Query token qsee ∈ RC, shared across all decoder layers. Firstly, we present how the See-Query aggregates multi-level features. The See-Query token is first transformed by a multi-layer perceptron (MLP)

Qinput = [qsee,q1,q2,...,qK] ∈ R(K+1)×C. (14) This unified query set undergoes a self-attention operation

to exchange information among the queries:

Qself = SelfAttention(Qinput), (15)

where Qself ∈ R(K+1)×C. Through this step, the See-Query interacts with the object queries to ‘see’ what they need, and then accordingly enriches them with relevant global multiscale context from Fagg. Specifically, the See-Query and object queries participate in the cross-attention operation, attending to the aggregated geometric encoder features Fagg:

Qcross = CrossAttention(Qself,Fagg), (16)

where Qcross ∈ R(K+1)×C. This generates updated representations for the See-Query and object queries based on

Methods cab bed chair sofa tabl door wind bkshf pic cntr ImVoxelNet [34] 20.2 76.3 45.8 60.5 36.0 16.3 09.6 17.9 00.3 23.3 FCAF3D [33] 28.1 81.0 47.2 70.7 40.9 17.1 13.2 22.0 00.7 45.6 NeRF-Det [46] 24.9 85.3 54.4 64.3 42.2 24.3 12.7 31.0 01.0 39.3 MVSDet [47] 28.0 80.6 55.5 65.3 43.2 25.2 09.9 29.4 01.1 43.4 VGGT-Det (Basic backbone) 20.1 85.5 48.4 78.5 37.0 13.2 09.4 22.1 00.8 40.4 VGGT-Det (Basic backbone+AG) 25.9 84.7 53.7 73.8 46.3 17.6 13.7 26.4 02.2 40.5 VGGT-Det (Basic backbone+AG+QD) 27.8 82.0 61.9 74.0 50.3 18.3 17.6 36.9 02.6 40.3 Methods desk curt fridg showr toil sink bath ofurn mAP@0.25

ImVoxelNet [34] 53.1 08.1 29.4 35.0 83.2 43.1 61.3 14.4 35.2 FCAF3D [33] 65.4 08.3 32.3 39.1 83.6 46.9 72.7 15.6 40.6 NeRF-Det [46] 62.6 17.3 41.4 32.8 88.4 39.7 58.7 21.0 41.2 MVSDet [47] 63.8 20.4 39.2 36.7 89.7 44.2 68.4 20.2 42.5 VGGT-Det (Basic backbone) 68.3 19.3 35.5 41.4 89.5 39.7 76.8 20.0 41.4 VGGT-Det (Basic backbone+AG) 66.5 18.7 41.5 45.2 84.6 45.3 84.0 24.3 44.2 VGGT-Det (Basic backbone+AG+QD) 66.9 31.4 49.0 32.9 90.3 50.8 83.2 27.6 46.9 (+4.4)

- Table 1. Comparison of mAP@0.25 across different methods on ScanNet. To achieve Sensor-Geometry-Free (SG-Free) setting and ensure a fair comparison, FCAF3D is retrained with point clouds predicted by VGGT. ImVoxelNet, NeRF-Det and MVSDet are retrained with multi-view camera poses predicted by VGGT. Our method significantly outperforms the competitive methods.

the hierarchical encoder features. After the cross-attention operation, the updated See-Query token q(seel) and the object queries are passed to the next decoder layer, where the process repeats starting from Eqn. (12). The iterative refinement of the See-Query enables dynamic, context-aware guidance of multi-level geometric feature aggregation across decoder layers, facilitating fine-grained 3D detection.

##### 4. Experiments

###### 4.1. Experimental Setup

Datasets and Settings. We evaluate our method on the widely-used ScanNet [8] and ARKitScenes [1]. They cover diverse indoor environments like homes, offices, and classrooms. ScanNet includes 18 standard object categories. ARKitScenes consists of 17 categories. For evaluation, we report mAP@0.25, where the Mean Average Precision (mAP) is computed to assess the model’s detection performance across all categories comprehensively.

Implementation Details. The pretrained VGGT encoder is frozen during training to preserve its original capabilities. In our experiments, we use 256 object queries. The model is optimized using the AdamW optimizer [26] with an initial learning rate of 2.5 × 10−4 and a weight decay of 1 × 10−4. To ensure stable training, gradient clipping is applied with a maximum norm of 35 and a norm type of 2. For learning rate scheduling, we adopt a cosine annealing strategy [27], where the learning rate gradually decays to 1 × 10−6, facilitating smooth convergence and mitigating instability caused by abrupt changes in the learning rate. The training loss setting follows 3DETR [30]. In our basic backbone, we employ a naive encoded feature aggregation method, where the encoded features from the 4th, 11th, 17th, and 23rd layers (following the VGGT architecture) are sequentially queried

in the decoder. These features, arranged from shallow to deep, are queried step by step to progressively refine the object representations. We train our models on 8×H800 GPUs, requiring approximately two days to complete.

###### 4.2. Method Analysis

In this section, we present an ablation study to thoroughly analyze the contributions of our proposed designs. We progressively incorporate each design into our basic backbone to evaluate their individual effectiveness.

Effect of Attention-Guided Query Generation. As shown in Tab. 2a, our Attention-Guided Query Generation (the second row) significantly improves the performance of the basic backbone (the first row), achieving a +2.8 point gain. This demonstrates the superiority of our Attention-Guided Query Generation compared to the FPS Sampling Generation. The gains suggest that our Attention-Guided Query Generation effectively leverages the internal semantic priors of VGGT, guiding the object queries to focus on object regions while preserving the global spatial structure of the scene.

Effect of Query-Driven Feature Aggregation. Considering both the first and second rows in Tab. 2a, which utilize the naive encoded feature aggregation strategy, the encoded features from the 4th, 11th, 17th, and 23rd layers (following the default setting of VGGT [41]) are sequentially queried in the decoder. This straightforward strategy aggregates multilevel geometric features in a simple manner. In contrast, our proposed Query-Driven Feature Aggregation adaptively aggregates multi-level geometric features through interactions with object queries. This method achieves a notable improvement, increasing mAP@0.25 by 2.7 points, showcasing the effectiveness of the See-Query mechanism. By interacting with object queries across different decoder layers, the SeeQuery learns to identify which levels of encoded features are

Methods mAP

###### Number mAP

###### Methods mAP

###### Methods Time (s) Mem (GB) mAP

BB 41.4 BB+AG 44.2 BB+AG+QD 46.9

40 45.3 60 46.2 80 46.9

Vanilla-v1 37.5 Vanilla-v2 44.2 QD (Ours) 46.9

VGGT 0.68 11.71 w/ MVSDet [47] 0.21 13.81 39.7 w/ Ours 0.23 3.57 45.3

(a)

(b)

(c)

(d)

- Table 2. Ablation studies: (a) Effectiveness of the proposed designs. ‘BB’ denotes our basic backbone; (b) Performance across different numbers of input frames; (c) Comparison of different feature aggregation methods; (d) Time and memory analysis. VGGT computation is in the first row, while the additional time and memory consumption of each method are shown in subsequent rows.

GIoU Validation Loss Backbone

Mean Validation Loss AG+QD

used 40 input frames on a 1×H800 GPU, with evaluation performed on ScanNet. The table breaks down the time required to process a 3D scene. For methods like MVSDet [47] that are not inherently sensor-geometry-free (SG-Free), we adapt them to an SG-Free setting by using predictions (e.g., multi-view poses) from VGGT as input for training and evaluation. Thus, for MVSDet, the time and memory overhead related to the shared VGGT components are included in the analysis. VGGT computation is listed in the first row of the table, while the additional time and memory consumption of each method are shown in subsequent rows. Our method ‘w/ Ours’ achieve comparable time consumption to ‘w/ MVSDet’ while significantly reducing memory usage. For instance, ‘w/ Ours’ consumes only 3.57 GB of memory compared to 13.81 GB for ‘w/ MVSDet’. Moreover, our method achieves a significant 5.6 points performance improvement with 40 input frames.

- 0.7 2.0
- 1.1 4.0

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

AG

AG

3.5

1.0

0.9

3.0

0.8

2.5

50 100 150 50 100 150

0 0

Epoch

Epoch

- Figure 5. Visualization of validation losses. In the left subfigure, after applying AG, the GIoU loss is significantly lower than that of the baseline backbone, indicating that AG effectively improves object localization during training. In the right subfigure, as See-Query progressively learns, within a few epochs, to interact with object queries and to aggregate encoded geometric features effectively, the loss for ‘AG+QD’ becomes significantly lower than that for ‘AG’, highlighting the effectiveness of the proposed QD strategy.

Performance Varies with the Number of Frames. We conduct an ablation study on ScanNet by feeding different numbers of frames into our model during evaluation, as shown in the Tab. 2b. The results demonstrate that more frames provide richer information, enabling the model to achieve higher performance. Performance saturates around 80 frames, suggesting that 80 frames are sufficient to capture most of the information needed to represent a 3D scene.

most relevant at each stage, leading to precise aggregation and significant performance gains.

Validating the Motivation Through Loss Dynamics. To deeply validate the motivation behind our technical contributions, we analyze the validation loss dynamics during training. Specifically, we evaluate the effects of AttentionGuided Query Generation (‘AG’) and its combination with Query-Driven Feature Aggregation (‘AG+QD’). As shown in the left subfigure of Fig. 5, the GIoU loss with ‘AG’ is consistently lower than that of the baseline backbone. This supports our motivation that attention maps effectively help object queries focus on semantic regions, thereby improving object localization accuracy. In the right subfigure, we further examine how Query-Driven Feature Aggregation (‘AG+QD’) complements ‘AG’. During the initial epochs, the losses of ‘AG’ and ‘AG+QD’ are similar, as the See-Query module is still learning to interact with object queries. However, after a few epochs, the loss for ‘AG+QD’ becomes significantly lower than that for ‘AG’. This demonstrates that the See-Query module learns to effectively aggregate encoded geometric features. These validation loss dynamics provide strong evidence validating our motivation and confirm the effectiveness of both ‘AG’ and ‘AG+QD’ in improving the model’s training.

Performance under Different Feature Aggregation Methods. We analyze the performance of our Query-Driven Feature Aggregation by comparing it with alternative feature aggregation strategies. The results are summarized in Tab. 2c. We consider three feature aggregation methods. (i) In the ‘Vanilla-v1’ method, only the encoded features from the last layer of the VGGT encoder [41] are fed into the decoder. (ii) In the ‘Vanilla-v2’ method, the encoded features from four specific layers of the VGGT encoder—namely, the 4th, 11th, 17th, and 23rd layers (following VGGT [41])—are sequentially fed into the decoder. (iii) Our Query-Driven Feature Aggregation (‘QD’ in Tab. 2c) employs a See-Query to adaptively interact with object queries across decoder layers, thereby determining and aggregating the most relevant encoded geometric features. As shown in Tab. 2c, ‘Vanilla-v2’ achieves better performance than ‘Vanilla-v1’, demonstrating that incorporating multi-level features provides richer information. Furthermore, our Query-Driven Feature Aggregation outperforms ‘Vanilla-v2’ by 2.7 points.

Time and Memory Analysis. We conducted the time and memory analysis in Tab. 2d. For fairness, all experiments

Ground Truth MVSDet VGGT-Det (Ours) Ground Truth MVSDet VGGT-Det (Ours)

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

Figure 6. Qualitative comparison. Compared with MVSDet [47], our VGGT-Det detects more objects with higher precision.

[Figure 194]

[Figure 195]

Methods cab bed chair sofa tabl sink washer toil bath ovn refg shelf stove tvm stl fire dshw mAP@0.25

ImVoxelNet 06.3 42.4 04.4 26.7 03.4 05.9 23.5 20.9 47.1 09.1 15.0 04.4 00.0 00.0 00.1 0.00 00.8 12.4 NeRF-Det 13.3 51.7 08.8 28.1 08.2 04.5 35.7 42.4 71.7 14.8 19.4 06.6 00.1 00.0 01.3 0.00 00.2 18.1 MVSDet 10.9 53.0 09.0 34.3 07.7 07.4 40.1 43.8 69.8 16.3 25.5 05.7 00.5 00.0 01.2 0.00 03.7 19.4 VGGT-Det 14.6 58.8 15.2 48.2 26.3 09.6 56.4 55.1 64.9 30.9 57.7 09.1 00.0 00.0 10.9 12.5 05.9 28.0 (+8.6)

- Table 3. Comparison of mAP@0.25 on ARKitScenes. To achieve Sensor-Geometry-Free (SG-Free) setting and ensure a fair comparison, ImVoxelNet [34], NeRF-Det [46] and MVSDet [47] are retrained with multi-view camera poses predicted by VGGT. Methods lacking both results and code support for ARKitScenes (e.g., FCAF3D) are not included. Our method significantly outperforms the competitive methods.

This improvement highlights the effectiveness of our design.

By interacting with object queries, the See-Query learns to adaptively aggregate the most relevant geometric features, leading to significant performance gains.

- 4.3. Comparison with Alternatives

We also compare against competitive multi-view indoor 3D object detectors [46, 47] on ARKitScenes [1], as shown in Tab. 3. VGGT-Det surpasses the state-of-the-art MVSDet [47] by 8.6 points, further demonstrating the superiority of our method. We also find that objects in a few categories, e.g. TV, are typically small, thin, or embedded within cabinetry or walls. Without accurate sensor geometry as prior information, such instances are highly prone to localization errors, leading to universally low performance across all the existing models. Note that to adapt the competitive methods to the same SG-Free setting for a fair comparison, we use the predicted multi-view poses from VGGT to retrain the models of ImVoxelNet, NeRF-Det and MVSDet.

Quantitative Comparison. Tab. 1 presents the quantitative comparison of mAP@0.25 across different methods on the ScanNet dataset [8]. Notably, our proposed VGGTDet achieves significant performance improvements over the baseline and the competitive multi-view indoor 3D object detection methods [32, 46, 47]. Specifically, by progressively incorporating the Attention-Guided Query Generation (AG) and Query-Driven Feature Aggregation (QD), our method demonstrates a consistent upward trend in overall mAP, culminating in a final result of 46.9, which is +4.4 higher than MVSDet [47]. This highlights the effectiveness of our method. Note that to adapt the competitive methods to the same SG-Free setting for a fair comparison, we use the predicted multi-view poses from VGGT to retrain the models of ImVoxelNet, NeRF-Det and MVSDet. To provide a more comprehensive analysis, we additionally compare with a representative point cloud–based method FCAF3D [33]. To adapt it to the same SG-Free setting for a fair comparison, we use the predicted point clouds from VGGT [41] to retrain the models of FCAF3D. VGGT-Det significantly outperforms FCAF3D by 6.3 points. In summary, the superior performance of VGGT-Det validates the proposed design’s ability to effectively leverage VGGT’s internally learned priors.

Qualitative Comparison. The qualitative comparison results are shown in Fig. 6. These results highlight the clear advantages of our VGGT-Det: it consistently detects more objects than the state-of-the-art MVSDet [47] and achieves higher detection precision, benefiting from the effective utilization of internal VGGT priors by our AG and QD modules.

##### 5. Conclusion

In this work, we introduce a more practical setting: SensorGeometry-Free 3D object detection, which removes the dependency on sensor-derived geometric inputs. To tackle this setting, we leverage the VGGT encoder to build a transformer-based backbone. Then, we propose two key innovations: Attention-Guided Query Generation and QueryDriven Feature Aggregation. The former utilizes the internal

semantic priors from VGGT’s attention mechanisms to generate object queries that effectively focus on object regions while maintaining the global spatial structure. The latter adaptively aggregates multi-level geometric features by dynamically interacting with object queries. Extensive experiments show that our VGGT-Det significantly outperforms strong methods in the SG-Free setting on both ScanNet and ARKitScenes. Ablations confirm that our AG and QD effectively leverage the learned priors from pretrained VGGT.

##### References

- [1] Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer, Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. Arkitscenes - a diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data. In NeurIPS, 2021. 6, 8
- [2] Eric Brachmann and Carsten Rother. Neural-guided ransac: Learning where to sample model hypotheses. In ICCV, 2019. 3
- [3] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Coda: Collaborative novel box discovery and cross-modal alignment for open-vocabulary 3d object detection. In NeurIPS, 2023. 1
- [4] Yang Cao, Yuanliang Jv, and Dan Xu. 3dgs-det: Empower 3d gaussian splatting with boundary guidance and box-focused sampling for 3d object detection. arXiv preprint arXiv:2410.01647, 2024. 1, 2
- [5] Yang Cao, Yihan Zeng, Hang Xu, and Dan Xu. Collaborative novel object discovery and box-guided cross-modal alignment for open-vocabulary 3d object detection. TPAMI, 2025. 1
- [6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. Endto-end object detection with transformers. In ECCV, 2020. 2
- [7] Dian Chen, Jie Li, Vitor Guizilini, Rares Andrei Ambrus, and Adrien Gaidon. Viewpoint equivariance for multi-view 3d object detection. In CVPR, 2023. 2
- [8] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richlyannotated 3D reconstructions of indoor scenes. In CVPR,

2017. 6, 8, 2

- [9] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superpoint: Self-supervised interest point detection and description. In CVPRW, 2018. 3
- [10] Chengjian Feng, Zequn Jie, Yujie Zhong, Xiangxiang Chu, and Lin Ma. Aedet: Azimuth-invariant multi-view 3d object detection. In CVPR, 2023. 2
- [11] Richard Hartley and Andrew Zisserman. Multiple view geometry in computer vision. Cambridge university press, 2003. 3
- [12] Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, and Chi-Keung Tang. Nerf-rpn: A general framework for object detection in nerfs. In CVPR, 2023. 2
- [13] Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, and Dalong Du. Bevdet: High-performance multi-camera 3d object detection in bird-eye-view. arXiv preprint arXiv:2112.11790,

2021. 2

- [14] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14,

2023. 2

- [15] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r. In ECCV, 2024. 1, 3
- [16] Yinhao Li, Han Bao, Zheng Ge, Jinrong Yang, Jianjian Sun, and Zeming Li. Bevstereo: Enhancing depth estimation in multi-view 3d object detection with temporal stereo. In AAAI,

2023. 2

- [17] Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, and Zeming Li. Bevdepth: Acquisition of reliable depth for multi-view 3d object detection. In AAAI, 2023. 2
- [18] Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, and Jifeng Dai. Bevformer: learning bird’s-eye-view representation from lidar-camera via spatiotemporal transformers. TPAMI, 2024. 2
- [19] Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4d: Multi-view 3d object detection with sparse spatial-temporal fusion. arXiv preprint arXiv:2211.10581, 2022. 2
- [20] Xuewu Lin, Tianwei Lin, Zixiang Pei, Lichao Huang, and Zhizhong Su. Sparse4d v2: Recurrent temporal fusion with sparse model. arXiv preprint arXiv:2305.14018, 2023.
- [21] Xuewu Lin, Zixiang Pei, Tianwei Lin, Lichao Huang, and Zhizhong Su. Sparse4d v3: Advancing end-to-end 3d detection and tracking. arXiv preprint arXiv:2311.11722, 2023. 2
- [22] Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys. Lightglue: Local feature matching at light speed. In ICCV, 2023. 3
- [23] Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, and Limin Wang. Sparsebev: High-performance sparse 3d object detection from multi-camera videos. In ICCV, 2023. 2
- [24] Yingfei Liu, Tiancai Wang, Xiangyu Zhang, and Jian Sun. Petr: Position embedding transformation for multi-view 3d object detection. In ECCV, 2022. 2
- [25] Yingfei Liu, Junjie Yan, Fan Jia, Shuailin Li, Aqi Gao, Tiancai Wang, and Xiangyu Zhang. Petrv2: A unified framework for 3d perception from multi-camera images. In ICCV, 2023. 2
- [26] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [27] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017. 6
- [28] David G Lowe. Distinctive image features from scaleinvariant keypoints. IJCV, 60:91–110, 2004. 3
- [29] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020. 2
- [30] Ishan Misra, Rohit Girdhar, and Armand Joulin. An end-toend transformer model for 3d object detection. In ICCV, 2021. 3, 6
- [31] Jonah Philion and Sanja Fidler. Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d. In ECCV, 2020. 2

- [32] Charles R Qi, Xinlei Chen, Or Litany, and Leonidas J Guibas. Imvotenet: Boosting 3D object detection in point clouds with image votes. In CVPR, 2020. 1, 8, 2
- [33] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Fcaf3d: Fully convolutional anchor-free 3d object detection. In ECCV, 2022. 6, 8, 1
- [34] Danila Rukhovich, Anna Vorontsova, and Anton Konushin. Imvoxelnet: Image to voxels projection for monocular and multi-view general-purpose 3d object detection. In WACV,

2022. 2, 6, 8

- [35] Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue: Learning feature matching with graph neural networks. In CVPR, 2020. 3
- [36] Guanlin Shen, Jingwei Huang, Zhihua Hu, and Bin Wang. Cn-rma: Combined network with ray marching aggregation for 3d indoor object detection from multi-view images. In CVPR, 2024. 1, 2
- [37] Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot gaussian splatting from uncalibrated image pairs. arXiv preprint arXiv:2408.13912, 2024. 1
- [38] Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local feature matching with transformers. In CVPR, 2021. 3
- [39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NeurIPS, 2017. 3
- [40] Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Pablo Speciale, and Marc Pollefeys. Patchmatchnet: Learned multiview patchmatch stereo. In CVPR, 2021. 3
- [41] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 1, 2, 3, 4, 5, 6, 7, 8
- [42] Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In ICCV, 2023. 2
- [43] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 1, 3
- [44] Yue Wang, Vitor Campagnolo Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. Detr3d: 3d object detection from multi-view images via 3d-to-2d queries. In CoRL, pages 180–191. PMLR, 2022. 2
- [45] Kaixin Xiong, Shi Gong, Xiaoqing Ye, Xiao Tan, Ji Wan, Errui Ding, Jingdong Wang, and Xiang Bai. Cape: Camera view position embedding for multi-view 3d object detection. In CVPR, 2023. 2
- [46] Chenfeng Xu, Bichen Wu, Ji Hou, Sam Tsai, Ruilong Li, Jialiang Wang, Wei Zhan, Zijian He, Peter Vajda, Kurt Keutzer, et al. Nerf-det: Learning geometry-aware volumetric representation for multi-view 3d object detection. In ICCV,

2023. 1, 2, 6, 8

- [47] Yating Xu, Chen Li, and Gim Hee Lee. Mvsdet: Multi-view indoor 3d object detection via efficient plane sweeps. In NeurIPS, 2024. 1, 2, 6, 7, 8, 4

- [48] Ziteng Xue, Mingzhe Guo, Heng Fan, Shihui Zhang, and Zhipeng Zhang. Corrbev: Multi-view 3d object detection by correlation learning with multi-modal prototypes. In CVPR,

2025. 2

- [49] Chi Yan, Delin Qu, Dan Xu, Bin Zhao, Zhigang Wang, Dong Wang, and Xuelong Li. Gs-slam: Dense visual slam with 3d gaussian splatting. In CVPR, 2024. 1
- [50] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. arXiv preprint arXiv:2501.13928, 2025. 1, 3
- [51] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In ECCV, 2018. 2, 3
- [52] Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. Recurrent mvsnet for high-resolution multi-view stereo depth inference. In CVPR, 2019. 2
- [53] Jingyang Zhang, Yao Yao, Shiwei Li, Zixin Luo, and Tian Fang. Visibility-aware multi-view stereo network. arXiv preprint arXiv:2008.07928, 2020. 3

### VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection

#### Supplementary Material

##### 6. More Ablation Studies

Performance under different λdist. In this subsection, we study the impact of varying λdist in Eqn. (8) on the performance of our proposed Attention-Guided Query Generation (AG). As shown in Tab. 4, the best performance in our evaluation is obtained around λdist = 0.8. Intuitively, λdist modulates the balance between attention guidance and spatial dispersion. When λdist is too small, the model relies heavily on attention guidance, leading to excessive focus on specific high-attention regions while neglecting the global spatial structure. When λdist is too large, the spatial dispersion term dominates, reducing the influence of attention guidance. In summary, λdist controls an intuitive balance between attention guidance and spatial dispersion. The observed optimal performance around λdist = 0.8 in our ablations aligns well with our goal of initializing queries that focus on semantic regions while preserving the global spatial structure.

Methods mAP@0.25

w/ AG (λdist=0.9) 44.9 w/ AG (λdist=0.8) 46.9 w/ AG (λdist=0.5) 40.7

Table 4. Performance under different λdist in Eqn. (8).

The impact of varying levels of noise. In this subsection, we evaluate the robustness of our method to noise in VGGTpredicted point clouds. Because the quality of the pretrained VGGT [41] outputs is not directly controllable, we introduce controlled noise by adding Gaussian noise to the VGGT point clouds. Specifically, the noise is sampled from a normal distribution n ∼ N(0,σ2), where σ represents the standard deviation of the noise. To ensure precise and scalable noise intensities, σ is defined as: σ = range · noise_level, where range = max(P) − min(P) represents the value range of the point clouds, and noise_level ∈ [0,1] is a userdefined parameter controlling the noise intensity. The noisy point cloud is generated as: P′ = P + N, where N is the noise matrix. This method enables fine-grained noise control, facilitating precise evaluation of noise effects. We evaluate VGGT-Det on ScanNet using noisy point clouds. To provide a fair comparison, we also evaluate the representative point cloud–based method FCAF3D [33], which likewise takes noisy point clouds as input. The results are summarized in Tab. 5. As expected, the performance of both methods degrades as the noise level increases, confirming the adverse impact of noise on detection. However, FCAF3D starts to degrade significantly at a noise level of 0.001 and drops to 0.0

Noise Level FCAF3D VGGT-Det (Ours)

0.30 0.0 34.1 0.20 0.0 39.4 0.10 0.0 44.5 0.01 18.7 47.0 0.005 33.4 47.0 0.001 39.7 46.9 0.000 40.6 46.9

Table 5. Robustness under different noise levels: comparison between FCAF3D [33] and our VGGT-Det.

mAP@0.25 at 0.1, whereas VGGT-Det maintains robustness and only starts degrading at 0.1, achieving 34.1 mAP@0.25 even at 0.3. At the same noise level of 0.01, VGGT-Det significantly outperforms FCAF3D by 28.3 points (47.0 vs. 18.7), demonstrating its superior robustness.

This robustness stems from our Attention-Guided Query Generation, which efficiently leverages VGGT’s internal semantic priors to generate object queries that focus on semantically relevant regions, as illustrated in Fig. 7. By reducing reliance on the precise geometric details of the predicted point clouds from VGGT, this mechanism effectively mitigates the impact of noise. In contrast, FCAF3D relies heavily on the geometric integrity of the input point clouds to extract features, making it significantly more susceptible to noise, as evidenced by its rapid performance degradation in Tab. 5. Performance across different numbers of input frames. For a fair comparison, we follow the standard practice in indoor multi-view 3D object detection (e.g., the official MVSDet implementation) by using 80 input frames. Nevertheless, our approach is not restricted to this configuration. Across different numbers of input frames, it consistently surpasses the strongest competing method, MVSDet, as shown in Tab. 6.

# of Frames 20 40 60 80 100 MVSDet [47] 35.9 39.7 40.9 42.5 42.5 Ours 42.6 45.3 46.2 46.9 47.3

Table 6. Performance across different numbers of input frames

Performance gap to methods that utilize sensor geometry. In Tab. 8, M1, M2, and M3 correspond to ImVoxelNet [32], NeRF-Det [46], and MVSDet [47], respectively, while “+SG” indicates methods that leverage sensor geometry. Our method achieves the highest performance in both settings—SG-Free and SG-Based. A noticeable gap remains between the two, which is expected since sensor-provided ge-

ometric priors offer strong cues for detection. Nevertheless, we regard the SG-Free setting as an important and practical regime, given that sensor geometry is often unavailable or impractical in real-world scenarios.

Method M1 M2 M3 Ours mAP 35.2 41.2 42.5 46.9 Method M1+SG M2+SG M3+SG Ours+SG mAP 48.1 49.5 56.2 58.8

- Table 7. Performance gap to methods that leverage sensor geometry.

More methods in efficiency analysis. To ensure fair comparison under the same SG-Free setting, we train and evaluate SG-Free variants of all models, rather than their original sensor-geometry-dependent versions. Specifically, multi-view RGB images are first processed by VGGT to estimate camera poses, which are then provided to each detector. Consequently, the SG-Free pipeline inherently includes VGGT’s runtime and memory costs, ensuring fairness in comparison. In Tab. 8, M1, M2, and M3 represent ImVoxelNet [32], NeRF-Det [46], and MVSDet [47], respectively. As shown, our designs (+Ours) achieve inference time comparable to the strongest competitor (+MVSDet), while attaining the lowest GPU memory usage among all methods.

Method VGGT +M1 +M2 +M3 +Ours

T (s) 0.68 0.09 0.07 0.21 0.23 M (GB) 11.71 12.95 6.76 13.81 3.57

- Table 8. Efficiency comparison with different mothods in SG-Free setting with 40 input frames.

Impact of view number on efficiency. As shown in Tab. 9, the time and GPU memory cost increase with more views, which is a common challenge in multi-view 3D detection.

# of Frames 20 40 60 80 100 T (s) 0.49 0.91 1.48 2.20 3.06 M (GB) 14.46 15.28 17.50 19.48 21.66

Table 9. Efficiency comparison under different input numbers.

Visualization analysis. To further study the effectiveness of our Attention-Guided Query Generation, we conduct a visualization analysis of both the attention maps and the positions of the generated object queries. As shown in Fig. 7, the 2nd column illustrates the attention maps, which clearly highlight the object regions. This observation supports our motivation: attention can provide semantic guidance for query generation. In the 3rd column, we compare the generated object query positions. The red points represent the object queries generated without attention guidance, while the green points correspond to the queries generated by our Attention-Guided Query Generation. Notably, within the

object regions (highlighted by green boxes), our method generates significantly more object queries (green points) compared to the baseline (red points). This demonstrates that our Attention-Guided Query Generation effectively focuses on object regions, leading to a more accurate query distribution. The effectiveness of our approach is further reflected in an improvement of 2.8 points in mAP@0.25, as reported in Tab. 2a of the main paper. This result highlights the direct contribution of Attention-Guided Query Generation to the overall performance.

##### 7. Comparison with Alternatives

Qualitative Comparison. We provide a qualitative comparison with the best-performing competitive method, MVSDet [47]. To adapt to the Sensor-Geometry-Free (SG-Free) setting and ensure a fair comparison, MVSDet is trained and tested using multi-view poses predicted by VGGT [41]. As shown in Fig. 8, our VGGT-Det detects more objects with higher accuracy, which is consistent with the significant performance improvement of 4.4 points reported in Tab. 1 of the main paper. The higher performance benefits from the effective utilization of internal VGGT priors by the proposed Attention-Guided Query Generation (AG) and Query-Driven Feature Aggregation (QD) modules.

##### 8. Training and Testing Time

All the ablation experiments are conducted on eight H800 GPUs. Training our model on the ScanNet dataset [8] takes approximately 2 days to complete. For testing, the entire ScanNet testing set can be processed in about 1 minute.

##### 9. Limitation and Future work

While VGGT-Det achieves significant improvements over strong alternative methods, several limitations remain for further exploration. Across current SG-free pipelines, VGGT incurs noticeable runtime and memory overhead. Besides, because VGGT produces normalized predictions, the scales from datasets are utilized to denormalize predictions of VGGT in all the current SG-free pipelines. Looking ahead, introducing a lighter VGGT-like model with metric-scale predictions could further advance this direction.

[Figure 196]

Input Images Attention Visualization

(w/ AG) vs. (w/o AG)

|[Figure 197]|
|---|

|[Figure 198]|
|---|

|[Figure 199]|
|---|

| |
|---|

[Figure 200]

[Figure 201]

| |
|---|

|[Figure 202]|
|---|

|[Figure 203]|
|---|

|[Figure 204]|
|---|

| |
|---|

[Figure 205]

|[Figure 206]|
|---|

|[Figure 207]|
|---|

|[Figure 208]|
|---|

| |
|---|

[Figure 209]

|[Figure 210]|
|---|

|[Figure 211]|
|---|

|[Figure 212]|
|---|

| |
|---|

[Figure 213]

[Figure 214]

|[Figure 215]|
|---|

|[Figure 216]|
|---|

[Figure 217]

- Figure 7. Visualization of attention and generated object query positions. Compared to farthest point sampling without guidance (red points), the points sampled by AG (green points) are more concentrated in object regions (labeled by green boxes), resulting in more green points than red points in those areas. For clarity, we recommend viewing the figure in color and zooming in.

## Ground Truth MVSDet VGGT-Det (Ours)

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

[Figure 232]

- Figure 8. Qualitative comparison with MVSDet [47]. To achieve the Sensor-Geometry-Free (SG-Free) setting and ensure a fair comparison, MVSDet is trained with multi-view poses predicted by VGGT [41]. The mesh here is not utilized in the methods and is only for visualization.

