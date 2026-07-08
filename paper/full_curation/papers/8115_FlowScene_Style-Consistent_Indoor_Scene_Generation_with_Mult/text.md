## FlowScene: Style-Consistent Indoor Scene Generation with Multimodal Graph Rectified Flow

Zhifei Yang1∗, Guangyao Zhai2∗, Keyang Lu1, YuYang Yin3, Chao Zhang4†, Zhen Xiao1†, Jieyi Long5, Nassir Navab2, Yikai Wang6

1School of Computer Science, Peking University 2Technical University of Munich 3Beijing Jiaotong University 4Beijing Digital Native Digital City Research Center 5Theta Labs, Inc. 6Beijing Normal University

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(Objects) I want a double bed, a wardrobe, a closet, a nightstand, a table, and a chair to generate a scene.

# arXiv:2603.19598v1[cs.CV]20Mar2026

Left of; Bigger than

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

LLM Layout

Left of; Close by

[Figure 14]

(Relations) The bed is at the left of the wardrobe and nightstand, and at the right of the table… The chair is at behind the table…

Left of

[Figure 15]

Left of

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

VLM

Shape

Bed

Right of

Behind

[Figure 24]

Object Selection Relation Selection

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|
|---|

Front of

Right of

Behind

|[Figure 29]|
|---|

[Figure 30]

[Figure 31]

|[Figure 32]|
|---|

[Figure 33]

[Figure 34]

Bigger than

Chair

Right of; Bigger than

GUI

Texture

…

[Figure 35]

[Figure 36]

Left of

[Figure 37]

[Figure 38]

…

|[Figure 39]|
|---|

[Figure 40]

|[Figure 41]|
|---|

|[Figure 42]<br><br>[Figure 43]|
|---|

Behind

Close by

Taller than

[Figure 44]

[Figure 45]

A. Diverse Input Graph Constructor B. Multimodal Graph FlowScene C. Scene Generation

Figure 1: Scene Generation from Diverse Input. The prospective system, powered by FlowScene, supports the generation of style-consistent 3D scenes from multi-source descriptions, including text input, GUI selections, and mixed information. Users can flexibly specify object categories and, if desired, provide their detailed RGB images in arbitrary views, along with their optional inter-object relationships (A). These descriptions are parsed into a multimodal graph by modern LLMs or VLMs as graph constructors (B). Based on this graph, FlowScene generates complete textured scenes, comprising both spatial scene layout and individual posed objects (C).

#### Abstract

#### 1 Introduction

Scene generation has extensive industrial applications, demanding both high realism and precise control over geometry and appearance. Language-driven retrieval methods compose plausible scenes from a large object database, but overlook object-level control and often fail to enforce scene-level style coherence. Graph-based formulations offer higher controllability over objects and inform holistic consistency by explicitly modeling relations, yet existing methods struggle to produce high-fidelity textured results, thereby limiting their practical utility. We present FlowScene, a tri-branch scene generative model conditioned on multimodal graphs that collaboratively generates scene layouts, object shapes, and object textures. At its core lies a tight-coupled rectified flow model that exchanges object information during generation, enabling collaborative reasoning across the graph. This enables fine-grained control of objects’ shapes, textures, and relations while enforcing scene-level style coherence across structure and appearance. Extensive experiments show that FlowScene outperforms both language-conditioned and graph-conditioned baselines in terms of generation realism, style consistency, and alignment with human preferences.

Scene generation from user prompts supports a wide spectrum of applications in manufacturing and interior design [72], VR/AR content creation [3], autonomous driving [52], and robotics [28]. These settings demand high realism and precise control over geometry and appearance, allowing users to specify object categories, semantic and spatial relations, and desired individual appearances. The scene generator must follow these instructions while preserving scene-level style consistency across object structure and visual appearance. A prospective workflow is able to transfer flexible and diverse inputs into a high-quality scene (cf. Figure 1 A and C).

Training-free approaches [59, 80] that rely directly on large models fail to meet this goal. They typically retrieve well-designed meshes to compose scenes from coarse language commands, yet provide limited per-object control, overlook inter-object relations, and rarely maintain scene-level style consistency. This leads to scale, topology, and appearance mismatches in the final scenes. For more interactive workflows, scene-graph–based methods [69, 81, 84, 85] explicitly model objects and their relations, enabling strong perobject controllability, better handling of relational constraints, and improved structural consistency. However, these methods fail to generate textured scenes in an end-to-end manner, which results in low-fidelity generation and limits their utility in downstream tasks.

∗ Equal contribution. † Corresponding authors.

Simultaneously achieving high fidelity and maintaining style consistency across objects under flexible control is non-trivial, especially when some objects lack visual or textual cues. In such cases, the model must aggregate scene-level context to guide the generation of their geometry and appearance. In this paper, we introduce FlowScene for compositional scene generation that maintains high fidelity and style consistency. As shown in Figure 1 B-C, FlowScene accepts a multimodal scene graph in which each node can fuse textual and visual information of the object, and employs three branches to generate scene layouts, object shapes, and object textures, respectively. At the core of each branch is Multimodal Graph Rectified Flow, which tightly couples exchanges of node information. Specifically, every node carries denoising states matched to each branch, including bounding boxes for the layout branch, voxelized latents for the shape branch, and structured latents for the texture branch. Throughout the denoising steps, nodes interact iteratively with each other along graph edges, ensuring progressively refined results and yielding fine-grained control over per-object appearance, inter-object spatial relations, and cross-object style consistency across structure and texture. Textured shapes are finally populated into the generated layout to produce the full scene.

We train FlowScene on 3D-FRONT [16] with SG-FRONT [85]. Experimentalperformanceshowsthat our results outperform languagedriven synthesis [59, 80] and graph-conditioned generation [81] in realism, style consistency, and human preference. Moreover, FlowScene significantly accelerates the graph-conditioned generation process while enhancing both the quality of individual objects and the overall performance of holistic scene generation. For the long-horizon application scope, we apply FlowScene as a robust backend that initiates scene creation based on user-provided sentences, or interactive selections of objects and relationships, or both. Our contributions are summarized as follows:

- • We present FlowScene for generating high-fidelity 3D scenes from multimodal graphs, supporting fine-grained control in object-level appearance and scene-level style consistency across structure and texture.
- • We detail the core of FlowScene as Multimodal Graph Rectified Flow. It exchanges node information during sampling to satisfy both individual and holistic conditions, and achieves faster generation than the previous diffusion-based mechanism.
- • We show stronger performance than competitors in generation realism, style consistency, and human preference alignment. We provide a workflow of how FlowScene facilitates scene generation from diverse input sources.

#### 2 Related Work

Scene Graph and Applications. Scene graphs provide a symbolic representation of a scene as a graph with object nodes and directed edges encoding inter-object relations. They can be constructed from multiple modalities, including text [87], 2D images [54], and 3D geometry [33], and even 4D spatio-temporal data [77], enabling rich spatial and temporal understanding. Early works established scene graphs for retrieval and reasoning [34], and subsequent research has leveraged them across a wide range of downstream tasks, including visual retrieval [13, 14, 31, 66], question answering [63],

controllable image synthesis [70], video synthesis [7, 82], 3D scene understanding [42], and 3D scene synthesis [81]. Scene graphs also support embodied applications such as manipulation planning [83] and instruction-conditioned navigation [45, 56]. In this work, we adopt a multimodal graph formulation that integrates textual and visual information at the node level [81].

Rectified Flow and Applications. Rectified flow and flow matching [41] have emerged as a strong alternative to diffusion-based generators [24], leveraging straight-line supervision with deterministic ODE sampling to reduce training variance and enable few-step generation. At scale, rectified flow transformers achieve the best image quality with competitive speed and model scaling behavior [12], and multi-rate designs extend these benefits to video by improving temporal coherence and long-horizon efficiency [29]. Efficient ODE solvers further amplify these gains [43], while related fast-sampling paradigms provide complementary perspectives on one/few-step generation [58]. Building on these advances, we adopt rectified flow as our backbone for 3D scene generation when coupled with the graph representation.

3D Scene Synthesis. 3D scene synthesis supports robotics and AR [47, 61] and is commonly driven by text [18, 44, 46, 53, 62], graphs [19, 84, 85], or images [27, 39]. Text-conditioned methods leverage LLM priors to generate layouts or full scenes [37, 49, 57, 80], but often yield incomplete or ambiguous spatial structure without strong visual grounding [2, 5, 15, 79]. Graph-conditioned pipelines encode objects and relations to improve coherence and control [9, 38], including relation conditioning [84, 85], with hierarchical extensions [60]. Image-based approaches exploit visual priors [21, 25, 65, 68], but fixed viewpoints limit holistic 3D reasoning and introduce cross-view inconsistencies [8, 59]. Along the modeling axis, works span autoregressive generators [51, 67, 88], layout/shape priors [10, 11, 32, 74, 75, 86], and diffusion-style objectives [25, 48, 71, 78]. Despite progress, current systems still do not unify multimodal inputs while offering both geometric and appearance control and reliable relation compliance, which is the goal of FlowScene in this paper.

#### 3 Preliminary

Rectified Flow. Rectifiedflow[41]is thestraight-lineinstantiation of flow matching models [40], which learns a time-dependent velocity field 𝑣𝜃 (𝑑,𝑡), 𝑡 ∈ [0, 1], such that the ODE 𝑑𝑡 = 𝑣𝜃 (𝑑𝑡,𝑡) transports the data distribution 𝑝data to a simple prior 𝑝1(e.g., N(0,𝐼)), where 𝑑 ∈ R𝑛 denotes the data variable. Training uses linear paths between 𝑑0 from the data and 𝑑1 from the prior, with identity path parameterization 𝑑𝑡 = (1 − 𝑡)𝑑0 + 𝑡𝑑1. The target velocity is constant along each path to be 𝑑1 − 𝑑0. We optimize the model by least-squares regression to these targets,

LRF(𝜃) = E𝑑0,𝑑1,𝑡 𝑣𝜃 (𝑑𝑡,𝑡) − (𝑑1 − 𝑑0) 22 , (1)

where 𝑑0 ∼𝑝data, 𝑑1 ∼𝑝1, and 𝑡 is sampled from a LogNormal(1,1) derived schedule on [0, 1]. During sampling, we draw𝑑1∼𝑝1 and integrate the reverse-time ODE 𝑑𝑡 = −𝑣𝜃 (𝑑𝑡,𝑡) from𝑡=1 to 0. Optional conditioning on side information 𝑐 (e.g., text, camera, geometry) is handled by using 𝑣𝜃 (𝑑,𝑡 | 𝑐) and forming the same straight-line targets with 𝑐 included; the objective keeps the same form.

Algorithm 1 Multimodal Graph Rectified Flow–Training Input: Graph GM, D0 = {𝑑0𝑖 }𝑖𝑁=1, noise sampler for D1

Left of; Taller than

[Figure 46]

###### D𝑡 VM E

###### GM

[Figure 47]

[Figure 48]

Night stand

- 1: while not converged do
- 2: Sample batch (GM, D0)
- 3: Calculate D1 based on D0 ⊲ D1 subject to N(0, 1)
- 4: Sample 𝑡 ∼ LogNormal(1, 1)
- 5: D𝑡 ← (1 − 𝑡)D0 + 𝑡D1
- 6: 𝑣target ← D1 − D0 ⊲ rectified target velocity
- 7: C𝑡 ← InfoExchangeUnit(GM, D𝑡,𝑡)
- 8: 𝑣ˆ ← ΘD(D𝑡, C𝑡,𝑡)
- 9: LGRF ← ∥𝑣ˆ − 𝑣target∥22 ⊲ Eq. (3)
- 10: Update ΘD, InfoExchangeUnit via backprop
- 11: end while

[Figure 49]

Stand on

[Figure 50]

Projector

[Figure 51]

[Figure 52]

Left of; Same style as Right of; Bigger than

|Concat| |
|---|---|
| | |

[Figure 53]

Floor

[Figure 54]

Stand on

[Figure 55]

[Figure 56]

Triplet-GCN

ΩG

[Figure 57]

[Figure 58]

Stand on

[Figure 59]

[Figure 60]

Right of; Smaller than

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

D𝑡 InfoExchangeUnit

×𝐿

#### 4 Method

C𝑡

We first illustrate the proposed graph rectified flow as a common graph-conditioned generation backbone. Then we delve into the details of how we integrate it into FlowScene.

- Figure 2: Multimodal Graph Processing. Given a graph GM (upper left), the graph network (right) first projects multimodal features of nodes VM, followed by message passing and feature aggregation along the edges E connecting VM. Within the denoising process of a generative model, it acts as an InfoExchangeUnit to incorporate temporal denoising

#### 4.1 Multimodal Graph Rectified Flow

Our formulation is essentially a tightly coupled rectified flow designed for multiple content generation. Original rectified flow [41] is applied to single-content generation, while our variant operates on all contents jointly to achieve high single-content quality and inter-content consistency.

data D𝑡, which is projected and fused with the node features (lower left). In this way, the network yields node-wise conditioning C𝑡 while respecting the global constraints imposed by the graph structure (bottom).

Training. The overall procedure is summarized in 1. We define target data samples of 𝑁 nodes as D0 = {𝑑0𝑖 | 𝑖 ∈ {1, . . ., 𝑁}} and a known prior as D1. In the forward process, we deploy an 𝑁-thread linear interpolation D𝑡 = (1 − 𝑡)D0 + 𝑡D1, which plays the role of the “reference” trajectories, analogous to the straight-line path connecting the source node data and the global priors, typically being white noise. The backward process is represented as a constant vector field 𝑣(D𝑡, C𝑡,𝑡) = D1−D0. Here, C𝑡 = {𝑐𝑡𝑖 | 𝑖 ∈ {1, . . ., 𝑁}} are the time-dependent, tightly coupled conditions, which exchange inter-object information to guide the denoising process. To obtain C𝑡, as shown in Figure 2, we adapt the triplet-GCN (2) to an InfoExchangeUnit fed on GM and D𝑡 to perform feature aggregation for timestep-wise condition: Each node now stacks the feature 𝑣𝑖𝑚 and the noisy 𝑑𝑡𝑖 as 𝛿𝑖𝑑 = 𝑣𝑖𝑚 ⊕ Projector(𝑑𝑡𝑖), and thus GM is augmented to GMD, which is processed by (2) to reflect the global data constraints inside the conditions C𝑡 = ΩG(GMD). Therefore, a denoiser ΘD is trained to approximate the vector field 𝑣 by minimizing the objective:

Multimodal Graph. A scene graph G = (V, E) represents an environment with object nodes and their relationships [6]. The node set V = {𝑣𝑖 | 𝑖 = 1, . . ., 𝑁} carries categorical embeddings 𝑝𝑖, and the edge set E = {𝑒𝑖→𝑗 | 𝑖, 𝑗 ∈ {1, . . ., 𝑁}, 𝑖 ≠ 𝑗}, where |E| = 𝑀, carries predicate labels𝜏𝑖→𝑗, indicating relations from𝑣𝑖 to 𝑣𝑗. To capture richer cues than semantics alone, a multimodal scene graph is introduced by [81] as GM = (VM, E), and its𝑖-th node𝑣𝑖𝑚 = [𝑝𝑖,𝑢𝑖, 𝑓𝑖] aggregates a learnable embedding 𝑝𝑖 with foundation text features 𝑢𝑖 (e.g., CLIP [55]), or foundation visual features 𝑓𝑖 (e.g., CLIP/DINOv2 [50] from the object image) or both. Missing modalities are zero-padded to a common format (e.g., [𝑝𝑖,𝑢𝑖, 𝑓𝑖] ↦→ [𝑝𝑖,𝑢𝑖, 0] or [0, 0, 𝑓𝑖]). GM thereby supports nodes that are (i) textonly, (ii) image-only, or (iii) multimodal, allowing the graph to unify language-only inputs (parsed by an LLM), GUI/VLM-derived visual nodes, or their mixtures, as shown in Figure 1 A-B.

GM can be encoded by a 𝐿-layer triplet Graph Convolutional Network (triplet-GCN) [30] ΩG for message passing and aggregation. Let the node features after the 𝑙-th layer be 𝛿𝑖(𝑙) with 𝛿𝑖(0) = 𝑣𝑖𝑚. Each layer updates via

###### LGRF = ED,C,𝑡 ∥ΘD(D𝑡, C𝑡,𝑡) − 𝑣∥22 . (3)

The architecture of ΘD can be various, and we adopt the rectified flow transformers [73].

(𝛾𝑖(𝑙),𝜏𝑖(→𝑙+𝑗1),𝛾𝑗(𝑙)) = MLP 𝛿𝑖(𝑙), 𝜏𝑖(→𝑙)𝑗, 𝛿𝑗(𝑙) , (2a) 𝛿𝑖(𝑙+1) =𝛾𝑖(𝑙) + MLP Avg 𝛾𝑗(𝑙) | 𝑗 ∈ NG(𝑖) , (2b)

Inference. At inferencetime,the process starts from D1 ∼ N(0, 1).

The model then integrates the learned conditional vector field, in order to evolve D𝑡 from D1 back toward the target data distribution D0. Formally, the trajectory follows ΔD𝑡 = −ΘD D𝑡, C𝑡,𝑡 Δ𝑡, integrated from 𝑡 = 1 to 𝑡 = 0 along 𝐾 steps. Owing to the near-linear paths induced by rectified flow training, this integration requires only a small number of steps in practice; see 2.

where NG(𝑖) are neighbors of 𝑣𝑖𝑚 and Avg is mean pooling. (2a) facilitates message passing between connected nodes, while (2b)

performs feature aggregation for each node.

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

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

| | | |
|---|---|---|
| | | |
| | | |

| | | |
|---|---|---|
| | | |
| | | |

ΘS Φ𝐷

ΘT Ψ𝐷

S1 S

X X1𝑒 X𝑒

O

B. Shape Branch C. Texture Branch

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

ShapeExchangeUnit

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

| |
|---|
| |
| |
| |
| |
| |
| |

[Figure 102]

[Figure 103]

[Figure 104]

TextureExchangeUnit

ΘB

[Figure 105]

LayoutExchangeUnit

B1 B

Noise Anchoring

A. Layout Branch Generated Layout Generated Scene

- Figure 3: FlowScene Pipeline. Conditioned on GM in Figure 2, FlowScene stacks three coordinated branches to generate a scene. A. Layout Branch generates the scene layout from noise B1, where the x-axis (red) points to the right and the z-axis (blue) points to the front. The LayoutExchangeUnit iteratively applies temporal layout constraints to the generation process by conditioning ΘB. B. Shape Branch is parallel to the layout branch. It exchanges shape information among nodes according to GM through the ShapeExchangeUnit. At inference, we sample noise S1, and generate latent shape codes S via ΘS, and decode them into voxelized shapes X using Φ𝐷. C. Texture Branch is subsequent to the shape branch. It anchors Gaussian noise onto X

to form X1𝑒, which is then denoised by ΘT into structured latents X𝑒. Finally, Ψ𝐷 decodes them into objects O, which are scaled and placed into the generated layout B to complete the scene.

Algorithm 2 Multimodal Graph Rectified Flow–Inference Input: Graph GM, sample steps 𝐾

Shape Branch. Following previous work [73, 85], training data for this branch are prepared by voxelizing objects into a sparse structure X = {𝑥𝑖 | 𝑖 ∈ {1, . . ., 𝑁}}. We use a shape VQ-VAE Φ(𝐸,𝐷) [64], which encodes X into compact latent code S = {𝑠𝑖 | 𝑖 ∈ {1, . . ., 𝑁}}. We show the procedure in Appendix A, Figure 6 A–B. Since |𝑠𝑖| ≪ |𝑥𝑖|, Φ provides computationally efficient modeling. Similar to the layout branch, this branch is trained independently with Eq. (3), with noise iteratively added in the forward process and mappings D𝑡 ↦→ S𝑡, GMD ↦→ GMS, and ΘD ↦→ ΘS. During denoising, the InfoExchangeUnit is specialized into a ShapeExchangeUnit, which exchanges shape information according to GMS to ensure consistent generation compliant with high-level edges (e.g., the same style as). The branch ultimately generates S via ΘS, and decodes them into shapes X = Φ𝐷(S).

- 1: Initialize D1 from noise N(0, 1)
- 2: Δ𝑡 ← 1/𝐾
- 3: for 𝑘 = 𝐾 − 1 down to 0 do ⊲ from 𝑡 = 1 to 𝑡 = 0
- 4: 𝑡 ← (𝑘 + 1)Δ𝑡
- 5: C𝑡 ← InfoExchangeUnit(GM, D𝑡,𝑡)
- 6: D𝑡−Δ𝑡 ← D𝑡 − ΘD(D𝑡, C𝑡, 𝑡) · Δ𝑡
- 7: end for
- 8: return D0

#### 4.2 Tri-branch Generation

Building on the proposed graph flow model, we develop FlowScene for scene generation, as illustrated in Figure 3. The framework consists of three branches responsible for generating the scene layout, object shapes, and object textures. The final scene is synthesized by populating the generated layouts with textured object shapes.

Texture Branch. The texture branch is subordinate to the shape branch, as an object’s texture is anchored to its geometry. Nevertheless, it is trained independently using the same rectified flow objective in Eq. (3), with branch-specific denoiser and exchange unit. Following Trellis [73], we render each object from random spherical views, and extract features with a pre-trained DINOv2 encoder [50]. Each voxel in 𝑥𝑖 projects onto the multiview features, which are averaged into 𝑥𝑖𝑓 . A texture VQ-VAE Ψ(𝐸,𝐷) learns to reconstruct the object, modeling a structured latent 𝑥𝑖𝑒 (cf. Appendix A, Figure 6 A–C). Thus, D0 maps to X𝑒 = {𝑥𝑖𝑒 | 𝑖 ∈ {1, . . ., 𝑁}}. Unlike the other branches, here noise is iteratively added only to the features of X𝑒, while the geometric structure remains unchanged. Note that X𝑒 shares the same geometry as X. Mappings follow GMD ↦→ GMT and ΘD ↦→ ΘT with InfoExchangeUnit specialized into a TextureExchangeUnit. At inference, the branch takes decoded shapes X and initializes X1𝑒 by anchoring Gaussian noise to the structured latent while keeping the geometric structure

Layout Branch. We represent the scene layout using 3D object bounding boxes, as drawn in Figure 3. Each bounding box b𝑖 is defined by its normalized location t𝑖 ∈ R3, size s𝑖 ∈ R3, and rotation angle 𝛼𝑖 expressed in sine–cosine form [cos(𝛼𝑖), sin(𝛼𝑖)]. Thus, the complete layout is represented as B = {[t𝑖, s𝑖, cos(𝛼𝑖), sin(𝛼𝑖)] | 𝑖 ∈ {1, . . ., 𝑁}}. This branch is trained independently to generate B by optimizing Eq. (3), with the denoising data following D𝑡 ↦→ B𝑡 and thus the data graph transforming into a layout-enhanced graph GMD ↦→ GMB. In the forward process, Gaussian noise is iteratively added until B becomes white noise as B1. The denoiser ΘD ↦→ ΘB is then conditioned on a LayoutExchangeUnit, specialized from the InfoExchangeUnit which extracts global layout constraints from GMB.

fixed. During the denoising process, the TextureExchangeUnit exchanges texture information among nodes to form X𝑒, resulting in style-consistent generation O. This procedure is particularly important for text-only nodes, whose appearances are primarily inferred from the exchanged information.

#### 5 Experiments

We first describe the experimental setup, including the dataset, baselines within different technical routes, metrics with a perceptual study for each aspect of generation quality, and implementation details. We then present quantitative and qualitative results against these baselines, followed by an ablation study on the architecture and input to assess their effect of generation fidelity and style consistency.

#### 5.1 Experimental Settings

Datasets. We conductexperimentson theSG-FRONT dataset [85] and 3D-FRONT dataset [16]. SG-FRONT encompasses about 45K object instances and 15 categories of relationships across bedrooms, dining rooms, and living rooms. Following the protocol [81], we obtain a multimodal graph where each node selectively includes textual and visual modalities.

Baselines. We compare FlowScene to two categories of methods: (i) Training-free, language-based methods for object retrieval: Holodeck [80] and LayoutVLM [59], which leverage large language and vision–language models [1, 36] to compose 3D scenes from text prompts. We obtain the prompts by translating scene graph labels to sentences. (ii) Graph-conditioned generative models trained under the same protocol for both object retrieval and generation: CommonScenes [85], which generates scenes from graphs using a VAE for layout and latent diffusion for shapes; EchoScene [84], a dualbranch diffusion model for layout and shape; MMGDreamer [81], which leverages a multimodal graph to emphasize geometry control; and all models are trained under the same protocol.

Metrics. We evaluate along four dimensions. (i) For generation realism, we measure scene-level fidelity with Fréchet Inception Distance (FID) [23], FIDCLIP [35], and Kernel Inception Distance (KID) [4] computed on top-down renderings of generation to measure similarity to ground truth, and object-level fidelity with Minimum Matching Distance (MMD), Coverage (COV), and 1-Nearest Neighbor Accuracy (1-NNA) [76]. (ii) For generation controllability, we report CLIPScore measuring the adherence between top-down renderings and user instructions [22], FPVScore on multiple egocentric views [26], and the graph-constraint satisfaction rate following prior work [81, 84]. (iii) For style consistency, we again adopt FPVScore and adapt its criteria to additionally focus on the geometric structure and visual appearance. (iv) Finally, we report the inference time for evaluating the generation efficiency compared to other graph-based methods.

Perceptual Study. We conduct a perceptual study based on ratings from 25 participants across 20 scenes to evaluate human preference alignment, including prompt adherence (PA), layout correctness (LC), visual quality (VQ), style consistency (SC), and overall preference (OP). Details are provided in Appendix F.

Implementation Details. We perform all experiments on a single NVIDIA A100 GPU with 80 GB of memory. The training of the three branches—shape, layout, and texture—is independently optimized using AdamW with an initial learning rate of 1e-4 and a batch size of 196. The shape and layout branches use a flow transformer, while the texture branch employs a sparse flow transformer, both following [73]. The sampling step 𝐾 is 25. More implementation details are provided in Appendix D.

#### 5.2 Quantitative results

Scene-Level Realism. We compare FlowScene with graph-based baselines [81, 84, 85] in two settings: First, baselines operate in retrieval mode, composing textured objects into 3D scenes, and we evaluate FlowScene with its texture branch. Second, all methods generate both layouts and shapes without textures, as baselines are not able to produce textures. This fairly assesses holistic geometric fidelity only. As shown in Table 1, FlowScene consistently outperforms all methods in the retrieval setting, even though others retrieve well-designed meshes. For bedroom in particular, FlowScene improves over MMGDreamer by reducing FID by 7.37, FIDCLIP by 1.34, and KID by 0.20, which indicates a closer match to the realscene distribution. A similar pattern appears in the full generation setting, where FlowScene achieves the lowest FID and KID among all baselines. In the perceptual study, as shown in Table 3, LC and VQ also indicate that FlowScene exhibits the best alignment with human preference.

Object-Level Realism. Object-level metrics focus on fine-grained geometric accuracy and diversity rather than the holistic scene fidelity. As shown in Table 2, FlowScene achieves better MMD and COV across most object categories. Specifically, for nightstands and lamps, MMD decreases by 43.90% and 41.76%, while COV increases by 45.43% and 40.30%, respectively, over MMGDreamer [81]. The 1-NNA results further show that FlowScene matches the shape distribution more closely than competing methods. Overall, these results demonstrate the superior realism and generative quality of objects produced by FlowScene.

Generation Controllability. Controllability measures how well generated scenes follow the given conditions or instructions. Since all methods except MMGDreamer and FlowScene take only textual inputs, we construct text-only scene graphs for all methods to ensure a fair comparison and evaluate vision–text consistency with CLIPScore [22] and FPVScore [26]. CLIPScore is computed on top-down renderings against a single coarse sentence that encodes all semantic and spatial information, which is widely used but relatively rough. FPVScore is a more recent approach that utilizes egocentric views with detailed prompts to provide fine-grained measurements of spatial and semantic adherence. As shown in Table 3, FlowScene attains the highest CLIPScore across all room types, and outperforms all baselines on the PA of FPVScore and perceptual study, indicating strong alignment with textual descriptions. We additionally provide the measurements of graph constraints in Appendix B.

Style Consistency. Style consistency is harder to quantify than fidelity or controllability, since it depends on subjective judgments about whether objects share a consistent geometry and appearance

- Table 1: Scene-Level Fidelity. This is evaluated by comparing top-down renderings of generated and real scenes at 2562 resolution

using FID, FIDCLIP, and KID (×0.001), following [84] (lower is better). * denotes their retrieval branches, which generate layouts and retrieve well-designed meshes from 3D-FUTURE [17]. The best results are in bold.

Method

Bedroom Living room Dining room FID FIDCLIP KID FID FIDCLIP KID FID FIDCLIP KID

CommonScenes∗ [85] 49.52 5.56 2.20 78.19 7.59 6.12 69.89 7.88 4.78 EchoScene∗ [84] 44.10 4.49 0.30 75.35 7.37 5.59 67.52 7.27 3.15 MMGDreamer∗[81] 42.38 4.48 -0.14 74.14 7.31 5.62 67.97 7.39 2.93 FlowScene (Ours) 35.01 3.14 -0.34 70.34 7.18 5.11 60.26 6.72 0.07

CommonScenes [85] 57.68 4.86 6.59 80.99 7.05 6.39 65.71 7.04 5.47 EchoScene [84] 48.85 4.26 1.77 75.95 6.73 0.60 62.85 6.28 1.72 MMGDreamer [81] 45.75 3.84 1.72 68.94 6.19 0.40 55.17 5.86 0.05 FlowScene (Ours, w/o TB) 38.87 3.56 -0.54 67.83 6.11 0.31 54.38 5.94 0.02

- Table 2: Object-Level Fidelity. Quantitative evaluation of generated object quality and diversity using MMD (×0.01), COV, and 1-NNA metrics across different categories. The best results are highlighted in bold.

Method Metric Bed N.stand Ward. Chair Table Cabinet Lamp Shelf Sofa TV stand CommonScenes [85]

MMD (↓)

0.49 0.92 0.54 0.99 1.91 0.96 1.50 2.73 0.57 0.29 EchoScene [84] 0.37 0.75 0.39 0.62 1.47 0.83 0.66 2.52 0.48 0.35 MMGDreamer [81] 0.22 0.41 0.24 0.35 0.55 0.71 0.34 1.58 0.43 0.24 FlowScene (Ours) 0.17 0.23 0.20 0.32 0.64 0.60 0.20 1.53 0.48 0.13

CommonScenes [85]

COV (%,↑)

24.07 24.17 26.62 26.72 40.52 28.45 36.21 40.00 28.45 33.62 EchoScene [84] 39.51 25.59 37.07 17.25 35.05 43.21 33.33 50.00 41.94 40.70 MMGDreamer [81] 42.59 30.81 44.44 19.95 44.12 49.38 40.56 70.00 47.31 45.35 FlowScene (Ours) 69.54 56.46 66.42 50.90 64.10 70.89 67.94 90.00 61.36 66.67

CommonScenes [85]

1-NNA (%,↓)

85.49 95.26 88.13 86.21 75.00 80.17 71.55 66.67 85.34 78.88 EchoScene [84] 72.84 91.00 81.90 92.67 75.74 69.14 78.90 35.00 69.35 78.49 MMGDreamer [81] 69.44 90.52 74.81 89.56 68.85 68.35 72.38 30.00 62.37 73.26 FlowScene (Ours) 45.51 59.19 50.19 79.60 54.83 59.12 53.22 10.00 52.25 54.44

- Table 3: Scene Perception Evaluation. For FPVScore, we report the average ranking across all scenes, where 1.00 is the worst and 5.00 is the best. For the perceptual study, scores range from 1.00 to 10.00. Each score is averaged across all participants and scenes. Both FPVScore and the perceptual study benchmark prompt adherence (PA), layout correctness (LC), visual quality (VQ), style consistency (SC), and overall preference (OP). The best results are highlighted in bold. * denotes the retrieval mode.

Method CLIPScore↑ FPVScore↑ Perceptual Study↑

Bedroom Living room Dining room PA LC VQ SC OP PA LC VQ SC OP

Echoscene∗ [84] 0.1837 0.1799 0.1553 3.03 3.07 2.82 3.03 3.01 7.21 6.83 6.86 7.12 7.24 MMGDreamer∗ [81] 0.1957 0.1877 0.1626 3.12 3.31 3.08 3.25 3.27 7.47 7.12 7.25 7.31 7.42 Holodeck [80] 0.1912 0.2116 0.1968 2.79 2.73 2.71 2.70 2.70 7.16 6.61 7.43 6.72 6.64 LayoutVLM [59] 0.2045 0.2056 0.1947 2.36 2.44 2.15 2.23 2.31 6.52 6.33 7.21 6.43 6.32 FlowScene (Ours) 0.2386 0.2246 0.2092 3.35 3.54 3.83 3.85 3.76 7.97 7.65 8.46 8.72 8.57

within a scene. To obtain a more reliable and comprehensive measurement, we extend FPVScore with a new criterion that focuses specifically on object structure and visual appearance, as illustrated in Appendix E. Benefiting from multiple continuous views covering the whole scene, this criterion is designed to more effectively capture whether objects match the intended style at both the geometric

and textural levels, rather than only matching semantics. As shown in Table 3, FlowScene outperforms all baselines on both VQ and SC, reflecting its strong ability to generate scenes that are both high-quality and consistent in cross-object style. The perceptual study on SC leads to the same conclusion, further supporting the effectiveness of our style modeling.

LivingRoomDiningRoomBedroom

Input Graph EchoScene† [84] MMGDreamer [81] FlowScene w/o TB FlowScene

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

|Wardrobe<br><br>[Figure 110]<br><br>[Figure 111]<br><br>TV stand<br><br>taller than<br><br>front<br><br>front right<br><br>[Figure 112]<br><br>same<br><br>material as<br><br>left<br><br>above Lamp above<br><br>|
|---|

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

|[Figure 117]<br><br>[Figure 118]<br><br>front<br><br>close by Chair<br><br>Chair<br><br>same super<br><br>category as<br><br>right<br><br>Chair<br><br>Chair<br><br>close by<br><br>[Figure 119]<br><br>Chair<br><br>Chair<br><br>[Figure 120]<br><br>right<br><br>close by<br><br>left<br><br>right<br><br>above<br><br>same<br><br>style as|
|---|

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

|[Figure 125]<br><br>Table<br><br>front<br><br>Cabinet<br><br>right<br><br>[Figure 126]<br><br>above<br><br>TV stand<br><br>front<br><br>[Figure 127]<br><br>right<br><br>taller<br><br>than Chair<br><br>same<br><br>style as<br><br>[Figure 128]<br><br>left<br><br>above|
|---|

###### Figure 4: Qualitative comparison. The first column shows the input multimodal graph with key scene edges. Red rectangles show inconsistencies and artifacts, while green rectangles highlight consistent regions. †Text-only scene graphs are given.

[Figure 129]

[Figure 130]

[Figure 131]

“Six wooden dining chairs are arranged around a dining table with a square wood top. A black pendant lamp is hung above the dining

table. Behind the dining table, a console table is set up. The dining chairs are positioned to the left and right of the pendant lamp, as well as in front and behind the dining table and console table.”

[Figure 132]

[Figure 133]

[Figure 134]

“Position a double bed in the room, with a gray

ceiling lamp hanging above it. To the left of the bed, arrange two nightstands with drawers.

Behind the bed, place a wardrobe with a door. The ceiling lamp should be positioned in front of the wardrobe.”

Instructions Holodeck [80] LayoutVLM [59] FlowScene (Ours)

###### Figure 5: Qualitative comparison with training-free, language-based methods. The grey boxes display the input textual instructions derived from text-only scene graphs, while FlowScene takes the text-only graphs as input.

Generation Efficiency. We measure the inference speed for evaluating efficiency, averaging over 10 scenes. All methods are conductedonasingle NVIDIAA100GPU. As shown inTable4,FlowScene demonstrates significantly faster speed in parallelly generating layout and shape, achieving an inference time of 6.83 seconds, which is 84.93% faster than MMGDreamer. Even though the texture branch is enabled, FlowScene still maintains the fastest inference time, with a result of 37.38 seconds.

#### 5.3 Qualitative results

We perform qualitative evaluation across different room types, as shown in Figure 4. FlowScene demonstrates superior style consistency, particularly in the dining room, where the generated chairs maintain a consistent appearance and texture, even as text-only nodes, while other methods fail to generate textures. This emphasizes FlowScene’s ability to generate both geometrically accurate and visually consistent scenes, outperforming baseline methods

###### Table 4: Inference Time Comparison.

Method Layout Shape Texture Time (s)

EchoScene [84] ✓ ✓ – 44.27 MMGDreamer [81] ✓ ✓ – 45.34

FlowScene (Ours, w/o TB) ✓ ✓ ✗ 6.83 FlowScene (Ours) ✓ ✓ ✓ 37.38

- Table 5: Ablation Study on the InfoExchangeUnit across different branches, including the LayoutExchangeUnit (LEU), ShapeExchangeUnit (SEU), and TextureExchangeUnit (TEU).

LEU SEU TEU FID FIDCLIP KID 50.83 4.55 7.32 ✓ 40.55 3.48 2.58 ✓ 44.73 4.42 4.97

✓ 43.82 3.86 3.68 ✓ ✓ 36.81 2.92 1.31 ✓ ✓ 36.30 2.90 0.77

✓ ✓ 39.35 3.22 2.08 ✓ ✓ ✓ 32.76 2.86 0.38

- Table 6: Ablation studies on different generation backbones.

Shape Layout

FID FIDCLIP KID Diffusion Flow Diffusion Flow

✓ ✓ 35.23 3.26 1.78 ✓ ✓ 34.96 2.90 1.03

✓ ✓ 33.03 2.94 0.62 ✓ ✓ 32.76 2.86 0.38

- Table 7: Ablation studies on the ratio of node modalities.

Modality FPVScore↑

Visual Textual LC VQ SC OP

10% 90% 3.28 3.37 3.42 3.26 40% 60% 3.45 3.86 3.89 3.67 50% 50% 3.42 3.88 3.91 3.64 60% 40% 3.36 3.88 3.91 3.62 90% 10% 3.28 3.78 3.82 3.59

#### 5.4 Ablation Study

Comparison on InfoExchangeUnit. As shown in Table 5, we ablate different combinations of LayoutExchangeUnit (LEU), ShapeExchangeUnit (SEU), and TextureExchangeUnit (TEU). Without any exchange unit, the model obtains an FID of 50.83. Enabling a single exchange unit consistently improves performance, with LEU giving the largest gain, highlighting the importance of layout-level information exchange for spatial coherence. Combining exchange units further improves results, especially LEU with TEU, which reduces KID to 0.77 and indicates better appearance consistency across objects. The best performance is achieved when all three units are enabled, yielding the lowest FID, FIDCLIP, and KID.

Comparison on Different Backbones. We steadily replace the diffusion backbone in the shape and layout branches with our graph flow. As shown in Table 6, switching only the layout branch from diffusion to flow brings a modest gain in scene fidelity. In contrast, switching only the shape branch yields a larger improvement, with FID dropping from 35.23 to 33.03. Using graph flow in both branches yields the best results, indicating that most of the benefit comes from the shape branch, while the layout branch provides complementary gains.

Comparison on Different Modality Ratios. We evaluate how robust style consistency and visual quality are to different ratios of visual and textual information in the multimodal graph. Starting from 10% of nodes with visual inputs, we gradually increase the ratio to 90%. As shown in Table 7, style consistency, measured by SC, remains stable at 3.79 with a variance of 0.03, and the accuracy-related metrics LC, VQ, and OP also stay stable across all ratios.

#### 6 Discussion

Experiments have shown that FlowScene serves as a robust scene generator driven by multimodal input. To broaden its use, we design an application interface in Appendix G that illustrates how FlowScene supports scene creation from diverse input sources and interactive user operations. While our current experiments are conducted on a single dataset, the scalability of FlowScene to larger and more diverse environments remains to be further explored, this does not diminish the generality and promise of the proposed interaction framework. Future directions include extending FlowScene to larger and outdoor scene, and integrating more deeply with interactive design and planning tools.

like MMGDreamer. Furthermore, Figure 5 highlights FlowScene’s improved controllability compared with the language-driven methods Holodeck [80] and LayoutVLM [59]. To achieve a fair comparison, we condition FlowScene on the text-only scene graphs. FlowScene enables more controllable instruction-guided generation, whereas baselines either exhibit style inconsistencies or fail to satisfy layout constraints accurately. This confirms that FlowScene not only enforces holistic coherence across multiple objects but also bridges abstract language with precise 3D generation, surpassing graph-condition or pure language-based approaches. We provide additional qualitative results in Appendix C.

#### 7 Conclusion

We introduced FlowScene, a tri-branch generator conditioned on multimodal graphs that generates scene layouts, object shapes, and object textures while preserving scene-level style consistency across structure and appearance. Each branch is backboned by a Multimodal Graph Rectified Flow module that exchanges object information during sampling, which enables fine-grained control over per-object appearance and relations. This module also improves efficiency compared with diffusion-based pipelines. Experiments show consistent gains in realism, controllability, style consistency, and human preference.

#### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Rio Aguina-Kang, Maxim Gumin, Do Heon Han, Stewart Morris, Seung Jean Yoo, Aditya Ganeshan, R Kenny Jones, Qiuhong Anna Wei, Kailiang Fu, and Daniel Ritchie. 2024. Open-universe indoor scene generation using llm program synthesis and uncurated object databases. arXiv preprint arXiv:2403.09675 (2024).
- [3] Miguel Angel Bautista, Pengsheng Guo, Samira Abnar, Walter Talbott, Alexander Toshev, Zhuoyuan Chen, Laurent Dinh, Shuangfei Zhai, Hanlin Goh, Daniel Ulbricht, et al. 2022. Gaudi: A neural architect for immersive 3d scene generation. Advances in Neural Information Processing Systems 35 (2022), 25102–25116.
- [4] Mikołaj Bińkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton.

2018. Demystifying mmd gans. arXiv preprint arXiv:1801.01401 (2018).

- [5] Ata Çelen, Guo Han, Konrad Schindler, Luc Van Gool, Iro Armeni, Anton Obukhov, and Xi Wang. 2024. I-design: Personalized llm interior designer. In European Conference on Computer Vision (ECCV) Workshop.
- [6] Xiaojun Chang, Pengzhen Ren, Pengfei Xu, Zhihui Li, Xiaojiang Chen, and Alex Hauptmann. 2021. A comprehensive survey of scene graphs: Generation and application. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 1

(2021), 1–26.

- [7] Yuren Cong, Jinhui Yi, Bodo Rosenhahn, and Michael Ying Yang. 2023. Ssgvs: Semantic scene graph-to-video synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2555–2565.
- [8] Wei Deng, Mengshi Qi, and Huadong Ma. 2025. Global-Local Tree Search in VLMs for 3D Indoor Scene Generation. arXiv preprint arXiv:2503.18476 (2025).
- [9] Helisa Dhamo, Fabian Manhardt, Nassir Navab, and Federico Tombari. 2021. Graph-to-3d: End-to-end generation and manipulation of 3d scenes using scene graphs. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 16352–16361.
- [10] Francis Engelmann, Konstantinos Rematas, Bastian Leibe, and Vittorio Ferrari.

2021. From points to multi-object 3D reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4588–4597.

- [11] Dave Epstein, Ben Poole, Ben Mildenhall, Alexei A Efros, and Aleksander Holynski. 2024. Disentangled 3d scene generation with layout learning. arXiv preprint arXiv:2402.16936 (2024).
- [12] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Fortyfirst international conference on machine learning.
- [13] Han Fang, Zhifei Yang, Yuhan Wei, Xianghao Zang, Chao Ban, Zerun Feng, Zhongjiang He, Yongxiang Li, and Hao Sun. 2023. Alignment and generation adapter for efficient video-text understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2791–2797.
- [14] Han Fang, Zhifei Yang, Xianghao Zang, Chao Ban, Zhongjiang He, Hao Sun, and Lanxiang Zhou. 2023. Mask to reconstruct: Cooperative semantics completion for video-text retrieval. In Proceedings of the 31st ACM International Conference on Multimedia. 3847–3856.
- [15] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2024. Layoutgpt: Compositional visual planning and generation with large language models. Advances in Neural Information Processing Systems 36 (2024).
- [16] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 2021. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 10933–10942.
- [17] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 2021. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision 129 (2021), 3313–3337.
- [18] Rao Fu, Zehao Wen, Zichen Liu, and Srinath Sridhar. 2024. Anyhome: Openvocabulary generation of structured and textured 3d homes. In ECCV.
- [19] Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, and Bernhard Schölkopf.

2024. Graphdreamer: Compositional 3d scene synthesis from scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21295–21304.

- [20] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025).
- [21] Takayuki Hara and Tatsuya Harada. 2024. MaGRITTe: Manipulative and Generative 3D Realization from Image, Topview and Text. arXiv preprint arXiv:2404.00345

(2024).

- [22] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi.

2021. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing. 7514–7528.

- [23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30

(2017).

- [24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems 33 (2020), 6840–6851.
- [25] Lukas Höllein, Ang Cao, Andrew Owens, Justin Johnson, and Matthias Nießner.

2023. Text2room: Extracting textured 3d meshes from 2d text-to-image models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7909– 7920.

- [26] Rui Huang, Guangyao Zhai, Zuria Bauer, Marc Pollefeys, Federico Tombari, Leonidas Guibas, Gao Huang, and Francis Engelmann. 2025. Video Perception Models for 3D Scene Synthesis. arXiv preprint arXiv:2506.20601 (2025).
- [27] Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. 2024. MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation. arXiv preprint arXiv:2412.03558 (2024).
- [28] Hanxiao Jiang, Binghao Huang, Ruihai Wu, Zhuoran Li, Shubham Garg, Hooshang Nayyeri, Shenlong Wang, and Yunzhu Li. 2024. Roboexp: Actionconditioned scene graph via interactive exploration for robotic manipulation. arXiv preprint arXiv:2402.15487 (2024).
- [29] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. 2024. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954

(2024).

- [30] Justin Johnson, Agrim Gupta, and Li Fei-Fei. 2018. Image generation from scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1219–1228.
- [31] Justin Johnson, Ranjay Krishna, Michael Stark, Li-Jia Li, David Shamma, Michael Bernstein, and Li Fei-Fei. 2015. Image retrieval using scene graphs. In Proceedings of the IEEE conference on computer vision and pattern recognition. 3668–3678.
- [32] Akash Abdu Jyothi, Thibaut Durand, Jiawei He, Leonid Sigal, and Greg Mori. 2019. Layoutvae: Stochastic scene layout generation from a label set. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9895–9904.
- [33] Sebastian Koch, Pedro Hermosilla, Narunas Vaskevicius, Mirco Colosi, and Timo Ropinski. 2024. Sgrec3d: Self-supervised 3d scene graph learning via objectlevel scene reconstruction. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 3404–3414.
- [34] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al.

2017. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision 123 (2017), 32–73.

- [35] Tuomas Kynkäänniemi, Tero Karras, Miika Aittala, Timo Aila, and Jaakko Lehtinen. 2022. The role of imagenet classes in fr\’echet inception distance. arXiv preprint arXiv:2203.06026 (2022).
- [36] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2024. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024).
- [37] Haoran Li, Haolin Shi, Wenli Zhang, Wenjun Wu, Yong Liao, Lin Wang, Lik-hang Lee, and Peng Yuan Zhou. 2024. Dreamscene: 3d gaussian-based text-to-3d scene generation via formation pattern sampling. In European Conference on Computer Vision. Springer, 214–230.
- [38] Chenguo Lin and Yadong Mu. 2024. Instructscene: Instruction-driven 3d indoor scene synthesis with semantic graph prior. arXiv preprint arXiv:2402.04717

(2024).

- [39] Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Zhaoshuo Li. 2025. Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation. arXiv preprint arXiv:2505.02836 (2025).
- [40] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.

2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747

(2022).

- [41] Xingchao Liu, Chengyue Gong, and Qiang Liu. 2022. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022).
- [42] Samuel Looper, Javier Rodriguez-Puigvert, Roland Siegwart, Cesar Cadena, and Lukas Schmid. 2022. 3d vsg: Long-term semantic scene change prediction through 3d variable scene graphs. arXiv preprint arXiv:2209.07896 (2022).
- [43] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu.

2022. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in neural information processing systems 35 (2022), 5775–5787.

- [44] Keyang Lu, Sifan Zhou, Hongbin Xu, Gang Xu, Zhifei Yang, Yikai Wang, Zhen Xiao, Jieyi Long, and Ming Li. 2025. Yo’City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion. arXiv preprint arXiv:2511.18734 (2025).
- [45] Ruifei Ma, Yifan Xu, Yuze Li, Yanping Fang, Zhifei Yang, Jiaxing Qi, Xinyu Zhao, and Chao Zhang. 2025. CTSG: Integrating Context and Way Topology Into Scene

- Graph for Zero-shot Navigation. In 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 18934–18941.
- [46] Yikun Ma, Dandan Zhan, and Zhi Jin. 2024. Fastscene: Text-driven fast 3d indoor scene generation via panoramic gaussian splatting. arXiv preprint arXiv:2405.05768 (2024).
- [47] Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox. 2023. MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations.
- [48] Quan Meng, Lei Li, Matthias Nießner, and Angela Dai. 2024. Lt3sd: Latent trees for 3d scene diffusion. arXiv preprint arXiv:2409.08215 (2024).
- [49] Başak Melis Öcal, Maxim Tatarchenko, Sezer Karaoğlu, and Theo Gevers. 2024. Sceneteller: Language-to-3d scene generation. In European Conference on Computer Vision. Springer, 362–378.
- [50] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin ElNouby, et al. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023).
- [51] Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, and Sanja Fidler. 2021. Atiss: Autoregressive transformers for indoor scene synthesis. Advances in Neural Information Processing Systems 34 (2021), 12013–12026.
- [52] Ethan Pronovost, Meghana Reddy Ganesina, Noureldin Hendy, Zeyu Wang, Andres Morales, Kai Wang, and Nick Roy. 2023. Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion. In NeurIPS.
- [53] Hou In Derek Pun, Hou In Ivan Tam, Austin T Wang, Xiaoliang Huo, Angel X Chang, and Manolis Savva. 2025. HSM: Hierarchical Scene Motifs for Multi-Scale Indoor Scene Generation. arXiv preprint arXiv:2503.16848 (2025).
- [54] Mengshi Qi, Weijian Li, Zhengyuan Yang, Yunhong Wang, and Jiebo Luo. 2019. Attentive relational networks for mapping images to scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3957– 3966.
- [55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.

2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763.

- [56] Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, and Niko Suenderhauf. 2023. Sayplan: Grounding large language models using 3d scene graphs for scalable robot task planning. arXiv preprint arXiv:2307.06135 (2023).
- [57] Liangchen Song, Liangliang Cao, Hongyu Xu, Kai Kang, Feng Tang, Junsong Yuan, and Yang Zhao. 2023. Roomdreamer: Text-driven 3d indoor scene synthesis with coherent geometry and texture. arXiv preprint arXiv:2305.11337 (2023).
- [58] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency Models. In International Conference on Machine Learning. PMLR, 32211–32252.
- [59] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. 2025. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference. 29469–29478.
- [60] Weilin Sun, Xinran Li, Manyi Li, Kai Xu, Xiangxu Meng, and Lei Meng. 2025. Hierarchically-Structured Open-Vocabulary Indoor Scene Synthesis with Pretrained Large Language Model. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 7122–7130.
- [61] Tomu Tahara, Takashi Seno, Gaku Narita, and Tomoya Ishikawa. 2020. Retargetable AR: Context-aware augmented reality in indoor scenes based on 3D scene graph. In 2020 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct).
- [62] Jiapeng Tang, Yinyu Nie, Lev Markhasin, Angela Dai, Justus Thies, and Matthias Nießner. 2023. Diffuscene: Scene graph denoising diffusion probabilistic model for generative indoor scene synthesis. arXiv preprint arXiv:2303.14207 2, 3 (2023).
- [63] Damien Teney, Lingqiao Liu, and Anton van Den Hengel. 2017. Graph-structured representations for visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1–9.
- [64] Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning. Advances in neural information processing systems 30 (2017).
- [65] Guangcong Wang, Peng Wang, Zhaoxi Chen, Wenping Wang, Chen Change Loy, and Ziwei Liu. 2023. Perf: Panoramic neural radiance field from a single panorama. arXiv preprint arXiv:2310.16831 (2023).
- [66] Jianhui Wang, Zhifei Yang, Yangfan He, Huixiong Zhang, Yuxuan Chen, and Jingwei Huang. 2025. Mari: Material retrieval integration across domains. In Proceedings of the Computer Vision and Pattern Recognition Conference. 5814– 5823.
- [67] Xinpeng Wang, Chandan Yeshwanth, and Matthias Nießner. 2021. Sceneformer: Indoor scene generation with transformers. In 2021 International Conference on 3D Vision (3DV). IEEE, 106–115.
- [68] Yian Wang, Xiaowen Qiu, Jiageng Liu, Zhehuan Chen, Jiting Cai, Yufei Wang, Tsun-Hsuan Johnson Wang, Zhou Xian, and Chuang Gan. 2025. Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting.

(2025).

- [69] Yao Wei, Martin Renqiang Min, George Vosselman, Li Erran Li, and Michael Ying Yang. 2025. Planner3D: LLM-enhanced Graph Prior Meets 3D Indoor Scene Explicit Regularization. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025), 1–17. doi:10.1109/TPAMI.2025.3602216
- [70] Yang Wu, Pengxu Wei, and Liang Lin. 2023. Scene graph to image synthesis via knowledge consensus. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 2856–2865.
- [71] Zhennan Wu, Yang Li, Han Yan, Taizhang Shang, Weixuan Sun, Senbo Wang, Ruikai Cui, Weizhe Liu, Hiroyuki Sato, Hongdong Li, et al. 2024. Blockfusion: Expandable 3d scene generation using latent tri-plane extrapolation. ACM Transactions on Graphics (ToG) 43, 4 (2024), 1–17.
- [72] Xiao Xia, Dan Zhang, Zibo Liao, Zhenyu Hou, Tianrui Sun, Jing Li, Ling Fu, and Yuxiao Dong. 2024. Scenegenagent: Precise industrial scene generation with coding agent. arXiv preprint arXiv:2410.21909 (2024).
- [73] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2024. Structured 3d latents for scalable and versatile 3d generation. arXiv preprint arXiv:2412.01506 (2024).
- [74] Yinghao Xu, Menglei Chai, Zifan Shi, Sida Peng, Ivan Skorokhodov, Aliaksandr Siarohin, Ceyuan Yang, Yujun Shen, Hsin-Ying Lee, Bolei Zhou, et al. 2023. Discoscene: Spatially disentangled generative radiance fields for controllable 3d-aware scene synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 4402–4412.
- [75] Han Yan, Yang Li, Zhennan Wu, Shenzhou Chen, Weixuan Sun, Taizhang Shang, Weizhe Liu, Tian Chen, Xiaqiang Dai, Chao Ma, et al. 2024. Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane. arXiv preprint arXiv:2403.16210 (2024).
- [76] Guandao Yang, Xun Huang, Zekun Hao, Ming-Yu Liu, Serge Belongie, and Bharath Hariharan. 2019. Pointflow: 3d point cloud generation with continuous normalizing flows. In Proceedings of the IEEE/CVF international conference on computer vision. 4541–4550.
- [77] Jingkang Yang, Jun Cen, Wenxuan Peng, Shuai Liu, Fangzhou Hong, Xiangtai Li, Kaiyang Zhou, Qifeng Chen, and Ziwei Liu. 2023. 4d panoptic scene graph generation. Advances in Neural Information Processing Systems 36 (2023), 69692– 69705.
- [78] Yandan Yang, Baoxiong Jia, Peiyuan Zhi, and Siyuan Huang. 2024. Physcene: Physically interactable 3d scene synthesis for embodied ai. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16262–16272.
- [79] Yixuan Yang, Junru Lu, Zixiang Zhao, Zhen Luo, James JQ Yu, Victor Sanchez, and Feng Zheng. 2024. LLplace: The 3D Indoor Scene Layout Generation and Editing via Large Language Model. arXiv preprint arXiv:2406.03866 (2024).
- [80] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. 2024. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 16227– 16237.
- [81] Zhifei Yang, Keyang Lu, Chao Zhang, Jiaxing Qi, Hanqi Jiang, Ruifei Ma, Shenglin Yin, Yifan Xu, Mingzhe Xing, Zhen Xiao, et al. 2025. MMGDreamer: MixedModality Graph for Geometry-Controllable 3D Indoor Scene Generation. arXiv preprint arXiv:2502.05874 (2025).
- [82] Zhoufaran Yang, Yan Shu, Jing Wang, Zhifei Yang, Yan Zhang, Yu Li, Keyang Lu, Gangyan Zeng, Shaohui Liu, Yu Zhou, et al. 2025. Vidtext: Towards comprehensive evaluation for video text understanding. arXiv preprint arXiv:2505.22810

(2025).

- [83] Guangyao Zhai, Xiaoni Cai, Dianye Huang, Yan Di, Fabian Manhardt, Federico Tombari, Nassir Navab, and Benjamin Busam. 2024. Sg-bot: Object rearrangement via coarse-to-fine robotic imagination on scene graphs. In 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 4303–4310.
- [84] Guangyao Zhai, Evin Pınar Örnek, Dave Zhenyu Chen, Ruotong Liao, Yan Di, Nassir Navab, Federico Tombari, and Benjamin Busam. 2024. EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion. arXiv preprint arXiv:2405.00915 (2024).
- [85] Guangyao Zhai, Evin Pınar Örnek, Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. 2024. Commonscenes: Generating commonsense 3d indoor scenes with scene graphs. Advances in Neural Information Processing Systems 36 (2024).
- [86] Yunfan Zhang, Hong Huang, Zhiwei Xiong, Zhiqi Shen, Guosheng Lin, Hao Wang, and Nicholas Vun. 2024. Style-Consistent 3D Indoor Scene Synthesis with Decoupled Objects. arXiv preprint arXiv:2401.13203 (2024).
- [87] Chengyang Zhao, Yikang Shen, Zhenfang Chen, Mingyu Ding, and Chuang Gan.

2023. Textpsg: Panoptic scene graph generation from textual descriptions. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 2839– 2850.

- [88] Yiqun Zhao, Zibo Zhao, Jing Li, Sixun Dong, and Shenghua Gao. 2024. Roomdesigner: Encoding anchor-latents for style-consistent and shape-compatible indoor scene generation. In 2024 International Conference on 3D Vision (3DV). IEEE, 1413–1423.

Supplementary Material Overview.

- • Object preprocessing pipeline (Appendix A)
- • Additional experimental results (Appendix B)
- • Qualitative comparisons (Appendix C)
- • Implementation details (Appendix D)
- • FPVScore criteria and perceptual study protocol (Appendix E, Appendix F)
- • Applications: language-driven and GUI-driven interfaces (Appendix G)
- • Limitations and failure cases (Appendix H)

#### A Object Processing

We show the object processing stages in Figure 6. Object 𝑜𝑖 is voxelized into 𝑥𝑖 (Figure 6 A.1), on which we train Φ(𝐸,𝐷) to obtain the latent shape code 𝑠𝑖 (Figure 6 B). In parallel, we render multi-view images of 𝑜𝑖 and extract DINOv2 features, which are reprojected onto 𝑥𝑖 and averaged across views to yield feature voxels 𝑥𝑖𝑓 (Figure 6 A.2). We then train Ψ(𝐸,𝐷), encoding 𝑥𝑖𝑓 into 𝑥𝑖𝑒 and decoding it to the original object 𝑜𝑖 (Figure 6 C).

#### B Additional Results

SceneGraphConstraints. Followingpreviousgraph-basedworks [81,

84, 85], we assess the model’s ability to maintain semantic consistency between the input graph and the generated 3D scene across three distinct modes: Relationship Change, Node Addition, and Generation Only. As shown in Table 8, even though the results are close to being saturated, FlowScene still achieves the best performances in the evaluation of most constraints, and exhibits robust performance in preserving spatial consistency during edge manipulations. For example, in the Relationship Change mode, FlowScene effectively adjusts object positions to satisfy updated constraints, achieving the best performance on complex relations, such as close by. Similarly, in the Node Addition mode, FlowScene demonstrates precise context-aware placement for newly added objects. It outperforms MMGDreamer on the challenging symmetrical metric. This indicates that our graph rectified flow mechanism can flexibly propagate local updates to the holistic scene structure.

Inter-Shape Consistency. To quantify the geometric coherence of identical object shapes, the Chamfer Distance (CD) is computed between pairs linked by the same as relationship, where lower scores denote higher shape consistency. As shown in Table 9, FlowScene achieves the state-of-the-art performance, consistently yielding the lowest CD values across all room types and object categories. In the Bedroom scenario, a striking improvement is observed in the “Lamp" category. Previous diffusion-based methods, such as EchoScene, struggle significantly, and FlowScene achieves a remarkable CD, surpassing MMGDreamer by an order of magnitude in the meantime. This demonstrates exceptional stability in generating consistent fine-grained structures. A similar trend persists in the Living Room, where our model reduces the geometric discrepancy for “Tables" and significantly outperforms CommonScenes. Furthermore, in the Dining Room, FlowScene achieves the lowest CD for chairs, demonstrating that it successfully synchronizes geometric latents across spatially distinct nodes to ensure holistic consistency.

Ablation on Different Input Views. We investigate the impact of visual input variability by training FlowScene with either singleview or multi-view images. As detailed in Table 10, while the model trained strictly on single-view images achieves the lowest FID under the matched testing condition, it fails to generalize to multi-view inputs, suffering a significant performance drop (FID increases by 7.97). In contrast, training with multi-view images yields remarkable robustness. The model exhibits negligible performance variance when switching from multi-view to single-view testing. This suggests that leveraging multi-view images during training encourages the model to learn view-invariant 3D representations, enabling it to adapt flexibly to varying visual conditions at inference without retraining.

Ablation on the Mask Ratio of Relations. We conduct relation masking on the scene graphs of the SG-Front dataset [85] to analyze how graph relation density affects scene generation fidelity, as summarized in Table 7. As the mask ratio decreases, performance consistently improves, indicating that denser inter-object relations provide stronger structural guidance for scene generation. Notably, even with a moderate mask ratio of 50%, FlowScene still outperforms the baseline without relation masking (0%∗ from MMGDreamer [81]), demonstrating robustness to partial relation removal and the effectiveness of leveraging even sparse relational information. Overall, these results highlight the importance of inter-object relations in enabling the graph-rectified flow to effectively propagate contextual information and enforce scene-level consistency across both structure and appearance during scene generation.

#### C Qualitative Results

Scene-Level Generation. We conduct a qualitative comparison to demonstrate FlowScene’s superiority in both consistency and controllability. As shown in Figure 7, compared to graph-conditioned baselines, our method excels in style consistency and layout plausibility. For instance, in the Dining Room scenario, while baselines often produce disparate object styles or chaotic placements, FlowScene generates a set of chairs with unified appearance and texture, faithfully adhering to satisfying the same style as constraint.

Object-Level Generation. We evaluate the fidelity of individual object generation from single-view inputs. As visualized in Figure 8, the baseline MMGDreamer tends to produce over-smoothed and distorted geometries, failing to capture complex topological structures. For instance, the generated bookshelf appears to be melted with irregular compartments, and the desk lacks distinct boundaries for its drawers. In contrast, FlowScene reconstructs precise geometries characterized by sharp edges and fine-grained details, alongside high-fidelity textures, faithfully preserving the structural integrity of the input images. As evidenced by examples of the wardrobe and bed, our method accurately renders intricate visual attributes, such as wood grain patterns and fabric folds. Consequently, FlowScene generates 3D objects that surpass the baselines in both geometric precision and appearance quality.

#### D Implementation Details

[Figure 135]

A. Object Preprocessing

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

DINOv2

[Figure 146]

[Figure 147]

A.1 Voxelization

A.2 Fusion

[Figure 148]

[Figure 149]

𝑜𝑖 𝑥𝑖 𝑥𝑖𝑓

B. Shape VQ-VAE C. Texture VQ-VAE

[Figure 150]

𝑥𝑖 𝑥𝑖𝑓 𝑥𝑖𝑒

[Figure 151]

Φ𝐸

| | | | |
|---|---|---|---|
| |𝑠| | |
| |𝑖| | |
| | | | |

Ψ𝐸 Ψ𝐷

Φ𝐷

Figure 6: Data Preparation.

- Table 8: Scene graph constraints. Comparisons are across three modes: Relationship Change, Node Addition, and Generation Only (listed top to bottom). The best results are highlighted in bold.

Method Mode

left/ right

front/ behind

smaller/ larger

taller/ shorter

close by symmetrical Graph-to-3D [9]

Change

0.91 0.92 0.86 0.89 0.69 0.46 CommonScenes [85] 0.91 0.92 0.86 0.91 0.69 0.59 EchoScene [84] 0.94 0.96 0.92 0.93 0.74 0.50 MMGDreamer [81] 0.95 0.96 0.93 0.93 0.71 0.53 FlowScene (Ours) 0.95 0.96 0.93 0.94 0.75 0.58

Graph-to-3D [9]

Addition

0.94 0.95 0.91 0.93 0.63 0.47 CommonScenes [85] 0.95 0.95 0.91 0.95 0.70 0.61 EchoScene [84] 0.98 0.99 0.96 0.96 0.76 0.49 MMGDreamer [81] 0.98 1.00 0.97 0.97 0.80 0.61 FlowScene (Ours) 0.98 1.00 0.97 0.98 0.77 0.68

Graph-to-3D [9]

None

0.98 0.99 0.97 0.95 0.74 0.57 CommonScenes [85] 0.98 1.00 0.97 0.95 0.77 0.60 EchoScene [84] 0.98 0.99 0.96 0.96 0.74 0.55 MMGDreamer [81] 0.98 0.99 0.97 0.96 0.76 0.62 FlowScene (Ours) 0.98 1.00 0.97 0.97 0.76 0.58

- Table 9: Inter-shape consistency. The Chamfer Distance (CD, ×0.001) is computed between object pairs linked by the same as relationship. Lower values indicate a higher degree of shape similarity between paired instances.

Bedroom Living room Dining room N.stand Lamp Table Lamp Chair Table

Method

CommonScenes [85] 2.69 - 11.75 - 1.96 9.04 EchoScene [84] 1.68 30.07 3.02 10.06 1.75 1.26 MMGDreamer[81] 1.33 2.29 2.44 0.18 0.23 0.83 FlowScene (Ours) 0.46 0.19 1.61 0.14 0.21 0.37

Software and Hardware. All experiments are conducted on a computing system configured with an AMD EPYC 7702 64-core processor and 8×NVIDIA A100 GPUs, as detailed in Table 12. Note that FlowScene is able to be trained with only one GPU. The system

runs Ubuntu 22.04 LTS with CUDA 12.8, NVIDIA Driver 570.133.20, and uses Python 3.10.0 with PyTorch 2.1.0 as the machine learning framework. For rendering 3D indoor scenes, we employ Blender 4.1 with the CYCLES engine. The rendering settings include a Noise

###### Table 10: Ablation studies of different input views.

Training Testing

FID FIDCLIP KID Multi Single Multi Single

✓ ✓ 33.77 2.88 0.73 ✓ ✓ 33.79 2.97 0.99

✓ ✓ 40.73 4.04 3.79 ✓ ✓ 32.76 2.86 0.38

###### Table 11: Ablation Study on the mask ratio of relation. * indicates results from MMGDreamer.

Mask Ratio FID FIDCLIP KID 20% 34.40 2.83 0.43 50% 38.94 3.22 0.96 80% 43.54 3.78 2.19

0%∗ 40.89 3.36 2.05

Table 12: System and Software Configuration.

System & Hardware Overview

CPU AMD EPYC 7702 64-Core GPU 8 ×NVIDIA A100 GPU Memory 1T RAM Operating System Ubuntu 22.04 LTS CUDA Version 12.8 NVIDIA Driver 570.133.20 ML Framework Python 3.10.0 Pytorch 2.1.0

###### GPU Specifications

CUDA Cores 6912 Memory Capacity 80GB

Threshold of 0.001, a max samples step of 300, and RGBA color mode with a 16 color depth. All rendering parameters are kept consistent across qualitative comparisons.

Dataset Details. We conduct experiments on the publicly available SG-FRONT dataset [85], extended scene graph annotations of the 3D-FRONT dataset [16]. SG-FRONT encompasses 15 relationship types across three categories: spatial (e.g., left/right of, bigger/smaller than, taller/shorter than), support (e.g., above, standing on), and style (e.g., the same material as, the same style as). These diverse relationships allow for modeling spatial and semantic interactions among objects. In total, there are 4041 bedroom scenes, 900 dining room scenes, and 813 living room scenes. We follow the data split used in MMGDreamer [81]. Unless otherwise specified, all ablation studies are evaluated on the full test set aggregated across all room types, rather than reported separately for each room category. For the training-free language-based methods, we construct input instructions from scene graphs based on Instructscene [38].

Model Architectures. All three branches adopt Transformer-based flow matching backbones similar to Trellis [73], and incorporate a triplet-GCN [30] as the core of the InfoExchangeUnit to enable iterative message passing. As illustrated in Figure 2, each branch instantiates a modality-specific unit variant with a dedicated projector. All InfoExchangeUnits use a 5-layer triplet-GCN backbone (𝐿 = 5) with MLP hidden dimension 𝑑hidden = 256. Graph convolutional parameters are not shared across branches. In all branches, modality-specific denoising states are first projected to a unified dimension𝑑gcn = 64, and then concatenated with graph node features before being processed by the triplet-GCN. The denoisers of layout and shape branches Θ(B,S) share a lightweight backbone configuration with 16 Transformer blocks, a hidden dimension of 512, and 16 attention heads. The layout branch employs the LayoutExchangeUnit, where bounding box attributes are projected to 𝑑gcn = 64 via a linear layer, and then concatenated with node features and processed by a triplet-GCN to produce node-wise layout conditioning. The shape branch operates on a 163 voxel grid and employs the ShapeExchangeUnit. Its projector consists of two 3D convolution layers with channels 32 → 64, followed by downsampling with stride 2 and stride 4, flattening, and a linear projection to 𝑑gcn = 64. The projected shape embeddings are concatenated with graph node features to generate shape conditioning. The texture branch ΘT captures high-frequency details at 643 resolution using a Sparse Transformer backbone wrapped with Sparse ResNet blocks, and instantiates the TextureExchangeUnit. Its sparse projector uses sparse 3D convolutions with channels 32 → 64, followed by sparse downsampling with factors 2 and 4, sparse-to-dense flattening, and a linear projection to 𝑑gcn = 64. The projected texture embeddings are concatenated with graph node features for graph-based texture conditioning.

VQ-VAE Details for Shape and Texture Branches. Following Trellis [73], both the shape and texture branches employ VQ-VAE models to encode high-dimensional structured representations into compact discrete latents. Shape VQ-VAE (Φ). The shape branch adopts a 3D convolutional VQ-VAE to encode sparse voxelized geometry into discrete latent codes, as illustrated in Figure 6 B. The encoder Φ𝐸 consists of a hierarchy of 3D convolutional and residual blocks with channel sizes 32, 128, 512, mapping input voxel grids to a latent embedding space of dimension 8. We use a codebook of size 1024 for vector quantization. We adopt the standard VQ objective with an exponential moving average (EMA) codebook update and set the commitment loss weight to 0.25. The decoder Φ𝐷 mirrors the encoder with reversed channel ordering 512, 128, 32 to reconstruct voxelized geometry from the discrete latent codes. Texture VQ-VAE (Ψ). The texture branch employs a structured latent VQ-VAE to encode multi-view appearance features anchored to object geometry, as illustrated in Figure 6 C. Following Trellis, multi-view image features are first extracted using a frozen DINOv2 encoder and aggregated into voxel-aligned feature volumes. The encoder Ψ𝐸 encodes these structured feature volumes into discrete latents of dimension 8 with a codebook size of 1024, using the same EMA-based vector quantization scheme and commitment weight of 0.25. The decoder Ψ𝐷 is implemented as a high-capacity transformer-based mesh and appearance decoder, parameterized

Bedroom Dining Room Living Room

|Cabinet<br><br>right<br><br>above<br><br>[Figure 152]<br><br>[Figure 153]<br><br>Shorter than<br><br>Nightstand<br><br>left<br><br>[Figure 154]<br><br>[Figure 155]<br><br>right<br><br>same<br><br>style as<br><br>above|
|---|

|[Figure 156]<br><br>Chair<br><br>Chair<br><br>Chair<br><br>Chair<br><br>[Figure 157]<br><br>[Figure 158]<br><br>close by close by<br><br>front<br><br>same style as<br><br>symmetrical to<br><br>taller than<br><br>same style as left<br><br>close<br><br>by<br><br>right|
|---|

|[Figure 159]<br><br>[Figure 160]<br><br>Table<br><br>close by<br><br>left Table taller<br><br>[Figure 161]<br><br>than<br><br>front above<br><br>|
|---|

[Figure 162]

[Figure 163]

[Figure 164]

FlowScene(Ours)EchoScene[84]MMGDreamer[81]CommonScenes[85]*,†*,†*

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Figure 7: Qualitative comparison with graph-conditioned generative models. The top row illustrates the input graphs, highlighting key semantic and spatial edges. In the generated scenes, red rectangles indicate generation failures in baselines (e.g., collisions), whereas green rectangles mark consistent regions in our results. * denotes the retrieval mode. †Text-only scene graphs are given.

with 12 attention blocks, 12 attention heads, and a model dimension of 768, to reconstruct high-resolution texture representations conditioned on the discrete texture latents.

Training Setup and Inference. We train the three branches of FlowScene independently using separate denoisers and exchange

[Figure 174]

Bed Bookshelf Wardrobe Desk

[Figure 175]

[Figure 176]

[Figure 177]

NodeImageFlowScene(w/oTB)MMGDreamer[81]FlowScene(Ours)

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

###### Figure 8: Qualitative comparison on object generation with other methods.

units, each optimized with the objective LGRF. Each branch is optimized for 82k steps with the AdamW optimizer. A batch size of 196 is applied uniformly across all three branches. We employ a piecewise constant learning rate schedule, initialized at 1 × 10−4 and decayed to 5 × 10−5 and 1 × 10−5 at 35k and 70k steps, respectively.To encode multimodal node information, we utilize a pre-trained DINOv2 encoder [50] for visual features and a frozen CLIP ViT-B/32 encoder [55] for textual descriptions. Additionally, to foster robustness against diverse user inputs, we randomly mask 20% of the textual and visual modalities within the graph nodes during both training and inference. For sampling, we adopt an Eulerbased ODE solver enhanced with interval Classifier-Free Guidance (CFG), following Trellis [73]. Each branch performs inference using Algorithm 2. We utilize 𝐾 = 25 inference steps with a CFG strength of 5.0.

Input Graph Failure Case

[Figure 190]

|[Figure 191]<br><br>short than<br><br>Nightstand<br><br>same<br><br>material as<br><br>left<br><br>[Figure 192]<br><br>[Figure 193]<br><br>[Figure 194]<br><br>left<br><br>bigger than<br><br>short<br><br>than<br><br>behind<br><br>right<br><br>[Figure 195]|
|---|

| |
|---|

Figure 9: Failure case. The left panel shows the input multimodal scene graph, while the right panel shows the generated failure case. Red cross marks indicate removed relationships.

#### E FPVScore Criteria Details

We implement FPVScore [26] using the GPT-4o model with the prompt template illustrated in Figure 10, performing a joint comparative evaluation of five distinct methods based on a shared text description. We present the LLM with 12 static multi-view images rendered from evenly spaced viewpoints around the scene, providing a 360-degree visual coverage. Each method is evaluated using

the same set of viewpoints to ensure fair comparison. We extend the standard criteria of Prompt Adherence (PA) and Layout Correctness (LC) by introducing Visual Quality (VQ) to assess visual fidelity and texture realism, and Style Consistency (SC) to evaluate the coherence of materials and design themes, specifically enforcing material uniformity across identical object instances. Finally, the model analyzes these visual inputs and assigns a rank from 1 (worst)

to 5 (best) for each method across all five dimensions, including Overall Preference (OP).

#### F Perceptual Study Details

We conducted a perceptual study involving 25 voluntary participants across 20 distinct scenes using a custom web-based interface. Upon entry, participants were briefed on the detailed instructions and evaluation criteria via the landing page, as shown in Figure 11. Subsequently, they navigated to the evaluation page in Figure 12 to assess the results. For each scene, outputs from five different methods were presented in a randomized order to prevent bias. Participants rated each method independently using a scale from 1.00 to 10.00 across five key metrics.

#### G Application Interface

We demonstrate the whole workflow with two application modes.(i) In a language-driven mode, users provide natural-language descriptions of objects and their spatial and semantic relations, optionally augmented with reference object images as visual appearances. As shown in Figure 13, an LLM such as DeepSeek [20] or GPT4o [1] can be employed to parse the text into a scene graph, which is then converted into a multimodal graph by attaching the uploaded images to the corresponding nodes. FlowScene runs as a backend on this graph and generates a high-fidelity textured 3D scene that follows the described configuration. (ii) In an interactive GUI mode, also illustrated in Figure 14, users first choose a

room type, after which the interface presents object candidates with images from multiple viewpoints. Users can select objects by clicking on these images or by specifying textual descriptions when they do not wish to fix the visual appearance. They can then define optional relations between object pairs. The GUI composes these choices into a multimodal graph, and again, FlowScene finally generates a 3D scene consistent with this graph. We provide two videos named language-to-scene.mov and gui-to-scene.mov in the Supplementary Material zip files. From a long-horizon perspective, FlowScene with scalable training on more data could generate textured objects from arbitrary given object images, allowing users to upload their desired images.

#### H Limitations

Our current implementation focuses primarily on synthetic indoor environments (e.g., 3D-FRONT [16]) and has not yet been evaluated on outdoor scenes. Additionally, the generation quality is intrinsically dependent on the precision of the input multimodal graph; consequently, inaccuracies or ambiguity from upstream graph constructors (LLMs/VLMs) may propagate to the final scene structure. For example, when the input image is captured from a limited or biased viewpoint, the model may generate imperfect object shapes. Moreover, missing key spatial relationship edges in the graph may reduce layout constraints, which can result in object interpenetration. We present such failure cases in Figure 9. Future work will aim to extend the model’s applicability to broader domains and enhance its robustness against noisy graph inputs.

FPVScore Prompt: Task: Compare the room layout rationality and overall performance of five methods, all generated from the same text description. From top to bottom, each row shows 12 static views sampled around the generated scene, providing a 360-degree multi-view coverage for each method. Decide which method performs best according to the criteria below. Text Description: {text_description}

Instructions:

- 1. Prompt Adherence (PA) Does the generated scene accurately reflect the text description? Check whether all described objects are present and correctly represented.
- 2. Layout Correctness (LC) Is the room design physically plausible and functional? Evaluate if the layout supports practical use, space efficiency, and proper object functionality. Consider object positions, orientations, and user convenience.
- 3. Visual Quality (VQ) Assess the visual realism and rendering quality of the generated scene. Consider texture fidelity, lighting realism, material smoothness, and absence of artifacts or distortions.
- 4. Style Consistency (SC) Evaluate whether the overall aesthetic style is coherent across the scene. Check if color palettes, material usage, and design patterns are consistent and match the intended scene theme.
- 5. Overall Preference (OP) Does the room layout look natural, appealing, and harmonious as a whole? Consider the holistic impression combining semantics, layout, visuals, and style.

Evaluation process: Carefully examine the multi-view images of all five 3D scenes. Focus on one criterion at a time and make independent judgments for each.

Output format: Provide a clear, concise analysis for each criterion. After the analyses, assign ranks (1˘5) to each method per criterion (1 = worst, 5 = best). Summarize your final ranking in the format: 〈rank for criterion 1 〉〈rank for criterion 2 〉〈rank for criterion 3 〉〈rank for criterion 4 〉〈rank for criterion 5 〉 for each method.

Example: Analysis:

- 1. Prompt Adherence: The first one ...; The second one ...; The third one ...; The fourth one ...; The fifth one ...
- 2. Layout Correctness: The first one ...; The second one ...; The third one ...; The fourth one ...; The fifth one ...
- 3. Visual Quality: The first one ...; The second one ...; The third one ...; The fourth one ...; The fifth one ...
- 4. Style Consistency: The first one ...; The second one ...; The third one ...; The fourth one ...; The fifth one ...
- 5. Overall Preference: The first one ...; The second one ...; The third one ...; The fourth one ...; The fifth one ...

Final answer: The first one: x x x x x The second one: x x x x x The third one: x x x x x The fourth one: x x x x x The fifth one: x x x x x

###### Figure 10: Prompt template for evaluating FPVScore [26] with GPT-4o [1].

### 📋 3D Scene Generation Perceptual Study

Evaluation Interface

###### 📘 Study Description

Thank you for participating in this academic perceptual study. In this experiment, you will evaluate different 3D scene generation methods. For each test case, you will be shown five scene results generated using different methods, all conditioned on the same text prompt.

###### 📝 Instructions

Please observe all images carefully before rating

All five evaluation metrics use a 1.00–10.00 scale (1.00 = very poor, 10.00 = excellent)

Rate each metric independently without comparing across metrics

Zoom in if necessary for better visibility

Estimated completion time: 10–15 minutes Your responses will be used for research purposes only. Thank you for contributing to the improvement of 3D scene generation technology.

###### Evaluation Metrics

PA — Prompt Adherence LC — Layout Correctness VQ — Visual Quality

SC — Style Consistency

OP — Overall Preference

Participant ID / Name (Optional)

⬅ Back Next ➡

###### Figure 11: Perceptual Study Interface: Instructions and Metrics. This introductory page is presented to participants at the start of the perceptual study. It outlines the study objectives, provides detailed instructions for viewing and rating, and clearly defines the five evaluation metrics (PA, LC, VQ, SC, OP).

##### Case 1

📌 Prompt: The room features a grey double bed with yellow pillows, flanked by two gold studded nightstands. One nightstand is positioned behind the bed, while the other is to the left of a wardrobe. The bed is also situated to the left of a lounge chair with a pillow. Two black wardrobes with doors and drawers are strategically placed in the room; one is behind the lounge chair, another is to the right of the bed. The wardrobes are also positioned in relation to the black ceiling lamp with hanging wires, with one wardrobe to its right and another in front of it.

###### Generated Scene Results

Method 1 Method 2 Method 3 Method 4 Method 5

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

###### Please Rate Each Method

Prompt Adherence — Does the generated scene accurately reflect the text description?

Method 1 Method 2 Method 3 Method 4 Method 5

1 ↺

1 ↺

1 ↺

1 ↺

1 ↺

1 10

1 10

1 10

1 10

1 10

Layout Correctness — Are the object placements physically plausible and functionally reasonable?

Method 1 Method 2 Method 3 Method 4 Method 5

1 ↺

1 ↺

1 ↺

1 ↺

1 ↺

1 10

1 10

1 10

1 10

1 10

Visual Quality — Does the generated scene appear visually realistic and high-quality?

Method 1 Method 2 Method 3 Method 4 Method 5

1 ↺

1 ↺

1 ↺

1 ↺

1 ↺

1 10

1 10

1 10

1 10

1 10

Style Consistency — Are colors, materials, and overall visual style consistent and unified throughout the scene?

Method 1 Method 2 Method 3 Method 4 Method 5

1 ↺

1 ↺

1 ↺

1 ↺

1 ↺

1 10

1 10

1 10

1 10

1 10

Overall Preference — Does the room look natural, appealing, and harmonious as a whole?

Method 1 Method 2 Method 3 Method 4 Method 5

1 ↺

1 ↺

1 ↺

1 ↺

1 ↺

1 10

1 10

1 10

1 10

1 10

⬅ Back Next ➡

###### Figure 12: Perceptual Study Interface: Example Page. Participants are presented with a detailed text prompt (top) and the corresponding generated 3D scenes from five different methods. Users interactively rate each method using sliding scales (1.00-10.00) across the five defined criteria, allowing for a direct and comprehensive comparison of generation quality.

[Figure 201]

###### FlowScene — Multimodal Indoor Scene Generation

###### Demo: Diverse Input → Multimodal Graph → Scene Generation.

Text + Image Input GUI Input

Fine-grained Control

Multimodal Graph

[Figure 202]

Specify objects with detailed text descriptions or reference images.

Graph Visualization

Global Scene Description

A bedroom with a double bed and two nightstands. The double bed is centered, positioned to the right of one nightstand and to the left of the other, indicating that the bed is ﬂanked by the two nightstands which are placed close by. These two nightstands share the same material and style. Regarding dimensions the double bed is the largest object; it is bigger and

- Object 1 ▼

[Figure 203]

- Object 2 ▼

[Figure 204]

- Object 3 ▼

Name (ID)

bed1

Ref Image (Optional)

[Figure 205]

Step 2: Generate 3D Scene

Generated 3D Scene

[Figure 206]

Name (ID)

3D Viewer

- nightstand1

- nightstand2

Ref Image (Optional)

Name (ID)

Ref Image (Optional)

Drop Image Here - or Click to Upload

+ Add Object

AI Relation Parsing

[Figure 207]

Parse

Select LLM

Relations from GPT-4o Text

Current Relations (Editable)

bed1, right of, nightstand1 bed1, left of, nightstand2 bed1, close by, nightstand1 bed1, close by, nightstand2 nightstand1, same material as, nightstand2 nightstand1, same style as, nightstand2 bed1, bigger than, nightstand1 bed1, bigger than, nightstand2 bed1, taller than, nightstand1 bed1, taller than, nightstand2 (Parsed by GPT-4o)

[Figure 208]

Step 1: Generate Graph

###### Figure 13: The Interface of Language Description with Images.

[Figure 209]

###### FlowScene — Multimodal Indoor Scene Generation

###### Demo: Diverse Input → Multimodal Graph → Scene Generation.

Text + Image Input GUI Input

###### Interactive Layout

2. Visualize & Generate

[Figure 210]

###### Quickly compose scenes using the object palette. Click multiple times to add multiple instances (e.g., nightstand1, nightstand2).

Multimodal Graph

Select Room Type

Switching room type updates the palette below.

Bedroom Living Room Dining Room

Object Palette (Click to Add)

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

Assets

Double Bed Wardrobe Nightstand Desk

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

Step 2: Generate 3D Scene

Cabinet Table Stool Lamp

[Figure 220]

3D Scene Result

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Shelf Chair Nightstand Double Bed

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Dressing Ta… Bookshelf Double Bed Stool

Current Objects

bed1 (Double Bed) nightstand1 (Nightstand) table1 (Table) nightstand2 (Nightstand) stool1 (Stool)

Build Relations

Subject Relation Target

bed1 close by nightstand1

[Figure 229]

Add Relation

Current Relations (Editable)

table1, left of, bed1 stool1, front of, nightstand2 nightstand1, right of, table1 table1, front of, bed1 nightstand2, right of, table1 nightstand1, same style as, nightstand2 bed1, right of, stool1 table1, left of, stool1 bed1, bigger than, nightstand2 bed1, close by, nightstand1

[Figure 230]

[Figure 231]

Clear All Step 1: Generate Graph

###### Figure 14: The Interface of GUI.

