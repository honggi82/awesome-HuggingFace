## HoloPart: Generative 3D Part Amodal Segmentation

Yunhan Yang1 Yuan-Chen Guo2 Yukun Huang1 Zi-Xin Zou2 Zhipeng Yu2 Yangguang Li2 Yan-Pei Cao2 Xihui Liu1

1 The University of Hong Kong 2 VAST Project Page: https://vast-ai-research.github.io/HoloPart

# arXiv:2504.07943v1[cs.CV]10Apr2025

###### 3D Part Amodal Segmentation

###### Applications

###### Generalization

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

(a) (b)

3D Part Segmentation 3D Part Shape Completion

Geometry Editing Material Editing

Figure 1. Demonstration of the difference between (a) 3D part segmentation and (b) 3D part amodal segmentation. 3D part amodal segmentation decomposes the 3D shape into complete semantic parts rather than broken surface patches, facilitating various downstream applications. In this paper, we propose a solution by performing 3D part shape completion on incomplete part segments.

### Abstract

opening new avenues for applications in geometry editing, animation, and material assignment.

3D part amodal segmentation—decomposing a 3D shape into complete, semantically meaningful parts, even when occluded—is a challenging but crucial task for 3D content creation and understanding. Existing 3D part segmentation methods only identify visible surface patches, limiting their utility. Inspired by 2D amodal segmentation, we introduce this novel task to the 3D domain and propose a practical, two-stage approach, addressing the key challenges of inferring occluded 3D geometry, maintaining global shape consistency, and handling diverse shapes with limited training data. First, we leverage existing 3D part segmentation to obtain initial, incomplete part segments. Second, we introduce HoloPart, a novel diffusion-based model, to complete these segments into full 3D parts. HoloPart utilizes a specialized architecture with local attention to capture finegrained part geometry and global shape context attention to ensure overall shape consistency. We introduce new benchmarks based on the ABO and PartObjaverse-Tiny datasets and demonstrate that HoloPart significantly outperforms state-of-the-art shape completion methods. By incorporating HoloPart with existing segmentation techniques, we achieve promising results on 3D part amodal segmentation,

### 1. Introduction

3D part segmentation [1, 23, 35, 56, 61, 69, 70] is an active research area. Given a 3D shape represented as a polygonal mesh or point cloud, 3D part segmentation groups its elements (vertices or points) into semantic parts. This is particularly valuable for shapes produced by photogrammetry or 3D generative models [18, 37, 38, 46, 63, 66], which are often one-piece and difficult to deal with for downstream applications. However, part segmentation has limitations. It produces surface patches rather than “complete parts” of the 3D shape like is shown in Figure 1 (a), where the segmented parts are broken. This may suffice for perception tasks but falls short for content creation scenarios where complete part geometry is required for geometry editing, animation, and material assignment. A similar challenge has been learned in 2D for many years, through the research area of 2D amodal segmentation. Numerous previous works [13, 20, 22, 32, 44, 49, 53, 62, 71] have explored the 2D amodal segmentation task, yet there remains a lack of related research for 3D shapes.

To address this, we introduce the task of 3D part amodal

: Corresponding authors.

segmentation. This task aims to separate a 3D shape into its complete semantic parts, emulating how human artists model complex 3D assets. Figure 1 (b) shows the expected output of 3D part amodal segmentation, where segmented parts are complete. However, extending the concept of amodal segmentation to 3D shapes introduces significant, non-trivial complexities that cannot be directly addressed by existing 2D or 3D techniques. 3D part amodal segmentation requires: (1) Inferring Occluded Geometry: Accurately reconstructing the 3D geometry of parts that are partially or completely hidden. (2) Maintaining Global Shape Consistency: Ensuring the completed parts are geometrically and semantically consistent with the entire 3D shape. (3) Handling Diverse Shapes and Parts: Generalizing to a wide variety of object categories and part types, while leveraging a limited amount of part-specific training data.

Recognizing the inherent difficulty of end-to-end learning for this task, we propose a practical and effective twostage approach. The first stage, part segmentation, has been widely studied, and we leverage an existing state-of-the-art method [61] to obtain initial, incomplete part segmentations (surface patches). The second stage, and the core of our contribution, is 3D part shape completion given segmentation masks. This is the most challenging aspect, requiring us to address the complexities outlined above. Previous 3D shape completion methods [6, 8, 52] focus on completing entire objects, often struggling with large missing regions or complex part structures. They also do not address the specific problem of completing individual parts within a larger shape while ensuring consistency with the overall structure.

We introduce HoloPart, a novel diffusion-based model specifically designed for 3D part shape completion. Given an incomplete part segment, HoloPart doesn’t just “fill in the hole”. It leverages a learned understanding of 3D shape priors to generate a complete and plausible 3D geometry, even for complex parts with significant occlusions. To achieve this, we first utilize the strong 3D generative prior learned from a large-scale dataset of general 3D shapes. We then adapt this prior to the part completion task using a curated, albeit limited, dataset of part-whole pairs, enabling effective learning despite data scarcity. Motivated by the need to balance local details and global context, HoloPart incorporates two key components: (1) a local attention design that focuses on capturing the fine-grained geometric details of the input part, and (2) a shape context-aware attention mechanism that effectively injects both local and global information to the diffusion model.

To facilitate future research, we propose evaluation benchmarks on the ABO [9] and PartObjaverse-Tiny [61] datasets. Extensive experiments demonstrate that HoloPart significantly outperforms existing shape completion approaches. Furthermore, by chaining HoloPart with off-theshelf 3D part segmentation, we achieve superior results on

the full 3D part amodal segmentation task. In summary, we make the following contributions:

- • We formally introduce the task of 3D part amodal segmentation, which separates a 3D shape into multiple semantic parts with complete geometry. This is a critical yet unexplored problem in 3D shape understanding, and provide two new benchmarks (based on ABO and PartObjaverse-Tiny) to facilitate research in this area.
- • We propose HoloPart, a novel diffusion-based model for 3D part shape completion. HoloPart features a dual attention mechanism (local attention for fine-grained details and context-aware attention for overall consistency) and leverages a learned 3D generative prior to overcome limitations imposed by scarce training data.
- • We demonstrate that HoloPart significantly outperforms existing shape completion methods on the challenging part completion subtask and achieves superior results when integrated with existing segmentation techniques for the full 3D part amodal segmentation task, showcasing its practical applicability and potential for various downstream applications.

- 2. Related Work
- 3D Part Segmentation. 3D Part Segmentation seeks to decompose 3D objects into meaningful, semantic parts, a long-standing challenge in 3D computer vision. Earlier studies [31, 47, 48, 50, 67] largely focused on developing network architectures optimized to learn rich 3D representations. These methods generally rely on fully supervised training, which requires extensive, labor-intensive 3D part annotations. Constrained by the limited scale and diversity of available 3D part datasets [3, 41], these approaches often face challenges in open-world scenarios. To enable open-world 3D part segmentation, recent methods [1, 23, 34, 35, 56–58, 60, 61, 69] leverage 2D foundation models such as SAM [25], GLIP [28] and CLIP [51]. These approaches first segment 2D renderings of 3D objects and then develop methods to project these 2D masks onto 3D surfaces. However, due to occlusions, these methods can only segment the visible surface areas of 3D objects, resulting in incomplete segmentations that are challenging to directly apply in downstream tasks. In this work, we advance 3D part segmentation by introducing 3D part amodal segmentation, enabling the completion of segmented parts beyond visible surfaces.

3D Shape Completion. 3D shape completion is a postprocessing step that restores missing regions, primarily focusing on whole shape reconstruction. Traditional methods like Laplacian hole filling [42] and Poisson surface reconstruction [21] address small gaps and geometric primitives. With the growth of 3D data, retrieval-based methods [55] have been developed to find and retrieve shapes that best match incomplete inputs from a predefined dataset. Along-

side these, learning-based methods [14, 43] predict complete shapes from partial inputs, aiming to minimize the difference to ground-truth shapes. Notable works include 3D-EPN [11] and Scan2Mesh [10], which use encoderdecoder architectures. PatchComplete [52] further enhances completion performance by incorporating multiresolution patch priors, especially for unseen categories. The rise of generative models such as GANs [16], Autoencoders [24], and Diffusion models [17] has led to methods like DiffComplete [8] and SC-Diff [15], which generate diverse and plausible 3D shapes from partial inputs. These models offer flexibility and creative freedom in shape generation. Furthermore, methods like DiffComplete [8], SCDiff [15], and others [5, 40, 65] leverage these advances for more robust shape completion. Additionally, PartGen [4] investigates part completion through the use of a multi-view diffusion model.

3D Shape Diffusion. Various strategies have been proposed to address the challenges associated with directly training a 3D diffusion model for shape generation, primarily due to the lack of a straightforward 3D representation suitable for diffusion. Several studies [7, 11, 19, 26, 27, 29, 30, 54, 59, 64, 66, 68] leverage Variational Autoencoders (VAEs) to encode 3D shapes into a latent space, enabling a diffusion model to operate on this latent representation for 3D shape generation. For instance, Shap-E [11] encodes a point cloud and an image of a 3D shape into an implicit latent space using a transformer-based VAE, enabling subsequent reconstruction as a Neural Radiance Field (NeRF). 3DShape2VecSet [63] employs cross-attention mechanisms to encode 3D shapes into latent representations that can be decoded through neural networks. Michelangelo [68] further aligns the 3D shape latent space with the CLIP [51] feature space, enhancing the correspondence between shapes, text, and images. CLAY [66] trains a large-scale 3D diffusion model on an extensive dataset, implementing a hierarchical training approach that achieves remarkable results.

### 3. 3D Part Amodal Segmentation

We formally introduce the task of 3D part amodal segmentation. Given a 3D shape m, the goal is to decompose m into a set of complete semantic parts, denoted as {p1,p2,...,pn}, where each pi represents a geometrically and semantically meaningful region of the shape, including any occluded portions. This is in contrast to standard 3D part segmentation, which only identifies visible surface patches. The completed parts should adhere to the following constraints:

- 1. Completeness: Each pi should represent the entire geometry of the part, even if portions are occluded in the input shape m.
- 2. Geometric Consistency: The geometry of each pi should be plausible and consistent with the visible por-

tions of the part and the overall shape m.

3. Semantic Consistency: Each pi should correspond to a semantically meaningful part (e.g., a wheel, a handle).

As discussed in the Introduction, this task presents significant challenges, including inferring occluded geometry, maintaining global shape consistency, and generalizing across diverse shapes and parts, all with limited training data. To address these challenges, we propose a two-stage approach:

- 1. Part Segmentation: We first obtain an initial part segmentation of the input shape m. This provides us with a set of surface patches, each corresponding to a (po-

tentially occluded) semantic segments {s1,s2,...,sn}. For this stage, we leverage SAMPart3D [61], although our framework is compatible with other 3D part segmentation techniques.

- 2. Part Completion: This is the core technical contribu-

tion of our work. Given an incomplete part segment si, our goal is to generate the corresponding complete part pi. This requires inferring the missing geometry of the occluded regions while maintaining geometric and semantic consistency. We address this challenge with our HoloPart model, described in the following sections. The remainder of this section details our approach, beginning with the object-level pretraining used to establish a strong 3D generative prior (Section 3.1), followed by the key designs of the HoloPart model (Section 3.2), and finally the data curation process (Section 3.3). The overall pipeline of HoloPart is shown in Figure 2.

- 3.1. Object-level Pretraining

Due to the scarcity of 3D data with complete part annotations, we first pretrain a 3D generative model on a largescale dataset of whole 3D shapes. This pretraining allows us to learn a generalizable representation of the 3D shape and capture semantic correspondences between different parts, which is crucial for the subsequent part completion stage.

Variational Autoencoder (VAE). We adopt the VAE module design as described in 3DShape2VecSet [63] and CLAY [66]. This design embeds the input point cloud X ∈ RN×3 sampled from a complete mesh, into a set of latent vectors using a learnable embedding function combined with a cross-attention encoding module:

z = E(X) = CrossAttn(PosEmb(X0),PosEmb(X)), (1)

where X0 represents subsampled point cloud from X via furthest point sampling, i.e. X0 = FPS(X) ∈ RM×3. The VAE’s decoder, composed of several self-attention layers and a cross-attention layer, processes these latent codes along with a list of query points q in 3D space, to produce the occupancy logits of these positions:

D(z,q) = CrossAttn(PosEmb(q),SelfAttn(z)). (2)

###### Context-Aware Attention

[Figure 15]

###### ×N 3D Shape Diffusion

[Figure 16]

|CrossAttn|
|---|

|SelfAttn|
|---|

|SelfAttn|Context Latents|
|---|---|
| | |

[Figure 17]

CrossAttn

#### …

SelfAttn

SelfAttn

Q

| | |Self| |Cross| |Cross|…|Self| |Cross| |Cross| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |Attn| |Attn| |Attn| |Attn| |Attn| |Attn| | |

CrossAttn

CrossAttn

CrossAttn

CrossAttn

SelfAttn

SelfAttn

Segmented Part

Decoder

KV

KV

[Figure 18]

|CrossAttn|
|---|

|SelfAttn|
|---|

|SelfAttn|
|---|

Local Latents

CrossAttn

#### …

SelfAttn

SelfAttn

Q

Noise Latents

Whole Shape w/ Segmentation Mask

Shape Latents

Complete Part

Scaled Part Local Attention

Figure 2. An overview of the HoloPart model design. Given a whole 3D shape and a corresponding surface segmentation mask, HoloPart encodes these inputs into latent tokens, using context-aware attention to capture global shape context and local attention to capture local part detailed features and position mapping. These tokens are used as conditions and injected into the part diffusion model via crossattention respectively. During training, noise is added to complete 3D parts, and the model learns to denoise them and recover the original complete part.

3D Shape Diffusion. Our diffusion denoising network vθ is built upon a series of diffusion transformer (DiT) blocks [30, 45, 59, 66, 68]. In line with the approach of Rectified Flows (RFs) [2, 33, 36], our diffusion model is trained in a compressed latent space to map samples from the gaussian distribution ϵ ∼ N(0,I) to the distribution of 3D shapes. The forward process is defined using a linear interpolation between the original shape and noise, represented as:

cl = C(S0,S) = CrossAttn(PosEmb(S0),PosEmb(S)),

(6) where S represents the sampled point cloud on the surface of the incomplete part mesh, and S0 denotes the subsampled point cloud from S via furthest point sampling. X represents the sampled point cloud on the overall shape. Here, M is a binary mask used to highlight the segmented area on the entire mesh, and ## represents concatenation.

We further finetune the shape diffusion model into a part diffusion model by incorporating our designed local and context-aware attention. The part diffusion model is trained in a compressed latent space to transform noise ϵ ∼ N(0,I) into the distribution of 3D part shapes. The objective function for part latent diffusion is defined as follows:

zt = (1 − t)z0 + tϵ, (3) where 0 ≤ t < 1000 is the diffusion timestep, z0 represents the original 3D shape, and zt is progressively noised version of the 3D shape at time t. Our goal is to solve the following flow matching objective:

Ez∈E(X),t,ϵ∼N(0,I) ∥vθ(zt,t,g) − (ϵ − z0)∥22 , (4)

Ez∈E(K),t,ϵ∼N(0,I) ∥vθ(zt,t,co,cl) − (ϵ − z0)∥22 , (7)

where g is the image conditioning feature [59] derived from the rendering of 3D shape during the pretraining stage.

where K represents the sampled point cloud from the complete part meshes. Following [68], we apply classifier-free guidance (CFG) by randomly setting the conditional information to a zero vector randomly. Once the denoising network vθ is trained, the function f can generate mˆ p by iterative denoising. The resulting latent embedding is then decoded into 3D space occupancy and the mesh is extracted from the part region using the marching cubes [39].

##### 3.2. Context-aware Part Completion

Given a pair consisting of a whole mesh x and a part segment mask si on the surface from 3D segmentation models as a prompt, we aim to leverage the learned understanding of 3D shape priors to generate a complete and plausible 3D geometry pi. To preserve local details and capture global context, we incorporate two key mechanisms into our pretrained model: local attention and shape context-aware attention. The incomplete part first performs cross-attention with the global shape to learn the contextual shape for completion. Next, the incomplete part is normalized to [−1,1] and undergoes cross-attention with subsampled points, enabling the model to learn both local details and the new position. Specifically, the context-aware attention and local attention can be expressed as follows:

##### 3.3. Data Curation

We process data from two 3D datasets: ABO [9] and Objaverse [12]. For the ABO dataset, which contains part ground truths, we directly use this information to generate whole-part pair data. In contrast, filtering valid part data from Objaverse is challenging due to the absence of part annotations, and the abundance of scanned objects and low-quality models. To address this, we first filter out all scanned objects and select 180k high-quality 3D shapes from the original 800,000 available models. We then develop a set of filtering rules to extract 3D objects with

co = C(S0,X)

(5)

= CrossAttn(PosEmb(S0),PosEmb(X##M)),

Ours w/o C-A w C-A

P/C D/C F/V

bed 0.093 0.061 0.023 0.032 0.020 table 0.081 0.068 0.030 0.042 0.018 lamp 0.170 0.084 0.044 0.036 0.031 chair 0.121 0.107 0.045 0.035 0.030

Chamfer ↓

mean (instance) 0.122 0.087 0.037 0.036 0.026

mean (category) 0.116 0.080 0.035 0.036 0.025

bed 0.148 0.266 0.695 0.792 0.833 table 0.180 0.248 0.652 0.791 0.838 lamp 0.155 0.238 0.479 0.677 0.697 chair 0.156 0.214 0.490 0.695 0.718

IoU ↑

mean (instance) 0.159 0.235 0.565 0.733 0.764

mean (category) 0.160 0.241 0.580 0.739 0.771

bed 0.244 0.412 0.802 0.864 0.896 table 0.291 0.390 0.758 0.844 0.890 lamp 0.244 0.374 0.610 0.769 0.789 chair 0.262 0.342 0.631 0.800 0.817

F-Score ↑

mean (instance) 0.259 0.371 0.689 0.816 0.843

mean (category) 0.260 0.380 0.700 0.819 0.848 Success ↑ mean (instance) 0.822 0.824 0.976 0.987 0.994

Table 1. 3D part amodal completion results of PatchComplete (P/C), DiffComplete (D/C), Finetune-VAE (F/V), Ours (w/o Context-attention), Ours (with Context-attention), on ABO, reported in Chamfer Distance, IoU, F-Score and Success Rate.

a reasonable part-wise semantic distribution from 3D asset datasets, including Mesh Count Restriction, Connected Component Analysis and Volume Distribution Optimization. Further details are provided in the supplementary.

To train the conditional part diffusion model f, we develop a data creation pipeline to generate whole-part pair datasets. First, all component parts are merged to form the complete 3D mesh. Next, several rays are sampled from different angles to determine the visibility of each face, and any invisible faces are removed. To handle non-watertight meshes, we compute the Unsigned Distance Field (UDF) of the 3D mesh and then obtain the processed whole 3D mesh using the marching cubes algorithm. We apply a similar process to each individual 3D part to generate the corresponding complete 3D part mesh. Finally, we assign part labels to each face of the whole mesh by finding the nearest part face, which provides surface segment masks {si}.

### 4. Experiments

##### 4.1. Experimental Setup

Datasets and Benchmarks. We propose two benchmarks based on two 3D shape datasets: ABO [9] and PartObjaverse-Tiny [61], to evaluate the 3D amodal completion task. The ABO dataset contains high-quality 3D models of real-world household objects, covering four categories: bed, table, lamp, and chair, all with detailed part annotations. For training, we use 20,000 parts, and for evaluation, we use 60 shapes containing a total of 1,000 parts. Objaverse [12] is a large-scale 3D dataset comprising over 800,000 3D shapes. PartObjaverse-Tiny is a curated subset of Objaverse, consisting of 200 objects (with 3,000 parts in total) with fine-grained part annotations. These 200 ob-

jects are distributed across eight categories: Human-Shape (29), Animals (23), Daily-Use (25), Buildings && Outdoor (25), Transportation (38), Plants (18), Food (8), and Electronics (34). We process 160,000 parts from Objaverse to create our training set, while PartObjaverse-Tiny serves as our evaluation set. We use our data-processing method to prepare two evaluation datasets, selecting only valid parts for our benchmarks. We further incorporate SAMPart3D to evaluate the 3D amodal segmentation task, with the details provided in the supplementary material.

Baselines. We compare our methods against state-of-the-art shape completion models, PatchComplete [52], DiffComplete [8] and SDFusion [6] using our proposed benchmarks. We train all baselines on our processed ABO and Objaverse datasets using the official implementations. To adapt to the data requirements of these models, we generated voxel grids with SDF values from our processed meshes. Additionally, our VAE model also uses 3D encoder-decoder architectures for 3D shape compression and reconstruction. Thus, we directly fine-tune the VAE on our parts dataset for part completion, serving as a baseline method.

Metrics. To evaluate the quality of predicted part shape geometry, we use three metrics: L1 Chamfer Distance (CD) Intersection over Union (IoU), and F-Score, comparing the predicted and ground truth part shapes. We sample 500k points on both the predicted and the group truth part meshes to capture detailed geometry information, used for the CD calculation. To compute IoU and F-Score, we generate voxel grids of size 643 with occupancy values based on the sampled points. Since the baseline methods are sometimes unable to reconstruct effective meshes, we calculate CD, IoU, and F-Score only for the successfully reconstructed meshes. Additionally, we report the reconstruction success ratio to quantify the reliability of each method.

##### 4.2. Main Results

ABO. We compare our method with PatchComplete [52], DiffComplete [8] and our fintuned VAE on the ABO dataset. Quantitative results are presented in Table 1, with qualitative comparisons illustrated in Figure 3. When dealing with parts containing large missing areas, PartComplete struggles to generate a plausible shape. PatchComplete and DiffComplete often fail to reconstruct small or thin structures, such as the bed sheets or the connections of the lamp in Figure 3. Although the finetuned VAE can reconstruct parts that have substantial visible areas, it performs poorly when completing regions with little visibility, such as the bedstead or the interior of the chair, as shown in Figure 3. In contrast, our method consistently generates high-quality, coherent parts and significantly outperforms the other approaches in both quantitative and qualitative evaluations.

PartObjaverse-Tiny. We also compare our method with PatchComplete, DiffComplete, and our finetuned VAE

Overall Mesh & Segmented Parts PatchComplete DiffComplete Finetune-VAE Ours GT

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

Figure 3. Qualitative comparison with PatchComplete, DiffComplete and Finetune-VAE on the ABO dataset.

[Figure 77]

Overall Mesh & Segmented Parts

PatchComplete DiffComplete Finetune-VAE Ours GT

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

Figure 4. Qualitative comparison with PatchComplete, DiffComplete and Finetune-VAE on the PartObjaverse-Tiny dataset.

Method Overall Human Animals Daily Buildings Transports Plants Food Electronics PatchComplete 0.144 0.150 0.165 0.141 0.173 0.147 0.110 0.118 0.147

DiffComplete 0.133 0.130 0.144 0.127 0.145 0.136 0.129 0.128 0.125

SDFusion 0.137 0.135 0.162 0.146 0.162 0.144 0.104 0.105 0.134 Finetune-VAE 0.064 0.064 0.067 0.075 0.064 0.076 0.049 0.041 0.073

Chamfer ↓

Ours w/o Local 0.057 0.061 0.083 0.051 0.047 0.075 0.045 0.037 0.057 Ours w/o Context 0.055 0.059 0.076 0.044 0.047 0.053 0.042 0.039 0.056

Ours 0.034 0.034 0.042 0.032 0.032 0.037 0.029 0.029 0.041 PatchComplete 0.137 0.129 0.147 0.132 0.116 0.129 0.152 0.156 0.138

DiffComplete 0.142 0.149 0.139 0.142 0.124 0.139 0.153 0.134 0.157

SDFusion 0.235 0.214 0.237 0.229 0.202 0.198 0.265 0.294 0.242 Finetune-VAE 0.502 0.460 0.464 0.503 0.513 0.468 0.536 0.583 0.490

IoU ↑

Ours w/o Local 0.618 0.582 0.574 0.618 0.634 0.591 0.673 0.677 0.594 Ours w/o Context 0.553 0.535 0.518 0.579 0.593 0.553 0.590 0.609 0.538

Ours 0.688 0.675 0.667 0.699 0.714 0.687 0.709 0.710 0.648 PatchComplete 0.232 0.221 0.246 0.224 0.197 0.220 0.254 0.261 0.233

DiffComplete 0.239 0.250 0.235 0.238 0.212 0.234 0.254 0.225 0.262

SDFusion 0.365 0.340 0.368 0.357 0.318 0.316 0.403 0.442 0.374 Finetune-VAE 0.638 0.600 0.613 0.638 0.646 0.596 0.672 0.718 0.623

F-Score ↑

Ours w/o Local 0.741 0.715 0.706 0.743 0.750 0.713 0.786 0.796 0.719 Ours w/o Context 0.691 0.679 0.663 0.716 0.722 0.688 0.727 0.743 0.676

Ours 0.801 0.794 0.788 0.809 0.818 0.798 0.817 0.820 0.767

Table 2. 3D part amodal completion results on PartObjaverse-Tiny, reported in Chamfer Distance, IoU, F-Score and Success Rate.

[Figure 212]

Generated Mesh Surface Segments Complete Parts Generated Mesh Surface Segments Complete Parts

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

- Figure 5. Our method seamlessly integrates with existing zero-shot 3D part segmentation models, enabling effective zero-shot 3D part amodal segmentation.

on the PartObjaverse-Tiny dataset. The shapes in the PartObjaverse-Tiny dataset are more complex and diverse, making part completion more challenging. We calculate the Chamfer Distance, IoU, F-Score, and Reconstruction Success rate for each method, with the quantitative comparison shown in Table 2. Our method consistently outperforms the others, even on this challenging dataset. As shown in Figure 4, our approach effectively completes intricate details, such as the eyeball, strawberry, and features on the house, which the other methods fail to achieve.

Zero-shot Generalization. By leveraging pretraining on the large-scale Objaverse dataset and finetuning on processed parts data, our model is capable of zero-shot amodal segmentation. To demonstrate the generalization capabilities of our model in a challenging zero-shot setting, we present 3D part amodal sementation results on generated meshes. As shown in Figure 5, we first apply SAMPart3D [61] to segment the surfaces of 3D shapes, and then use our model to generate complete and consistent parts.

S = 1.5 S = 3.5 S = 5 S = 7.5 Chamfer ↓ 0.059 0.057 0.058 0.089

IoU ↑ 0.590 0.618 0.614 0.514 F-Score ↑ 0.718 0.741 0.738 0.641 Success ↑ 0.995 0.997 0.996 0.997

Table 3. Ablation study of different guidance scale for diffusion sampling on the PartObjaverse-Tiny dataset.

##### 4.3. Ablation Analysis

Necessity of Context-Aware Attention. The contextaware attention is crucial for completing invisible areas of parts and ensuring the consistency of generated components. To demonstrate this, we replace the context-aware attention block with a local-condition block and train the model. The quantitative comparison shown in Table 1 and Table 2 demonstrates the significance of context-aware attention. The qualitative analysis is provided in the supplementary material.

###### Necessity of Local Attention. Local attention is crucial for

(a) Geometry Editing (b) Geometry Processing

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

Part-Aware w/o Part-Aware

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

|[Figure 236]|
|---|

|[Figure 237]|
|---|

[Figure 238]

[Figure 239]

[Figure 240]

(c) Material Editing

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

- Figure 6. 3D part amodal segmentation is capable of numerous downstream applications, such as Geometry Editing, Geometry Processing, Material Editing and Animation.

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

Overall Shape Part Shapes Overall Shape Part Shapes Overall Shape Part Shapes

- Figure 7. Geometry Super-resolution. By representing a part with the same number of tokens as the overall object, we can achieve geometry super-resolution.

maintaining details and mapping positions. We perform an ablation study on the local attention module and present the quantitative comparison in Table 2, highlighting the necessity of our local attention design.

These operations would be difficult to achieve with traditional 3D part segmentation techniques. Additionally, we showcase an example of a geometry processing application in Figure 6 (b). With our completed parts, we achieve more reasonable remeshing results. Additionally, by integrating with existing 3D part segmentation methods, our model can serve as a powerful data creation tool for training part-aware generative models or part editing models.

Effect of Guidance Scale. We find that the guidance scale significantly impacts the quality of our generated shapes. We evaluate four different guidance scales (1.5, 3.5, 5, and 7) on the PartObjaverse-Tiny dataset, with the results presented in Table 3. A small guidance scale leads to insufficient control, while an excessively large guidance scale results in the failure of shape reconstruction from latent fields. We find a scale of 3.5 provides the optimal balance.

Our model also has the potential for Geometric Superresolution. By representing a part with the same number of tokens as the overall object, we can fully preserve and generate the details of the part. A comparison with the overall shape, reconstructed using the same number of tokens by VAE, is shown in Figure 7.

##### 4.4. Application

Our model is capable of completing high-quality parts across a variety of 3D shapes, thereby enabling numerous downstream applications such as geometry editing, material assignment and animation. We demonstrate the application of geometry editing in Figures 1 and 6 (a), and material assignment in Figures 1 and 6 (c). For example, in the case of the car model, we perform 3D part amodal segmentation, then modify the sizes of the front and rear wheels, increase the number of jars, and expand the car’s width in Blender. Afterward, we assign unique textures to each part and enable the wheels and steering wheel to move. The video demo is included in the supplementary material.

### 5. Conclusion

This paper introduces 3D part amodal segmentation, a novel task that addresses a key limitation in 3D content generation. We decompose the problem into subtasks, focusing on 3D part shape completion, and propose a diffusion-based approach with local and context-aware attention mechanisms to ensure coherent part completion. We establish evaluation benchmarks on the ABO and PartObjaverse-Tiny datasets, demonstrating that our method significantly outperforms prior shape completion approaches. Our compre-

hensive evaluations and application demonstrations validate the effectiveness of our approach and establish a foundation for future research in this emerging field.

### References

- [1] Ahmed Abdelreheem, Ivan Skorokhodov, Maks Ovsjanikov, and Peter Wonka. Satr: Zero-shot semantic segmentation of 3d shapes. In ICCV, 2023. 1, 2
- [2] Michael S Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. arXiv preprint arXiv:2209.15571, 2022. 4
- [3] Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d model repository. arXiv:1512.03012,

2015. 2

- [4] Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models. arXiv preprint arXiv:2412.18608, 2024. 3
- [5] Xuelin Chen, Baoquan Chen, and Niloy J Mitra. Unpaired point cloud completion on real scans using adversarial training. arXiv preprint arXiv:1904.00069, 2019. 3
- [6] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In CVPR, 2023. 2, 5
- [7] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-sdf: Conditional generative modeling of signed distance functions. In ICCV, 2023. 3
- [8] Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Nießner, Chi-Wing Fu, and Jiaya Jia. Diffcomplete: Diffusion-based generative 3d shape completion. NeurIPS,

2024. 2, 3, 5

- [9] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In CVPR, 2022. 2, 4, 5, 1
- [10] Angela Dai and Matthias Nießner. Scan2mesh: From unstructured range scans to 3d meshes. In CVPR, 2019. 3
- [11] Angela Dai, Charles Ruizhongtai Qi, and Matthias Nießner. Shape completion using 3d-encoder-predictor cnns and shape synthesis. In CVPR, 2017. 3
- [12] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In CVPR, 2023. 4, 5, 1
- [13] Kiana Ehsani, Roozbeh Mottaghi, and Ali Farhadi. Segan: Segmenting and generating the invisible. In CVPR, 2018. 1
- [14] Michael Firman, Oisin Mac Aodha, Simon Julier, and Gabriel J Brostow. Structured prediction of unobserved voxels from a single depth image. In CVPR, 2016. 3
- [15] Juan D Galvis, Xingxing Zuo, Simon Schaefer, and Stefan Leutengger. Sc-diff: 3d shape completion with latent diffusion models. arXiv preprint arXiv:2403.12470, 2024. 3

- [16] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 2020. 3
- [17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020. 3
- [18] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 1
- [19] Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. Neural wavelet-domain diffusion for 3d shape generation. In SIGGRAPH Asia 2022 Conference Papers, 2022. 3
- [20] Abhishek Kar, Shubham Tulsiani, Joao Carreira, and Jitendra Malik. Amodal completion and size constancy in natural scenes. In ICCV, 2015. 1
- [21] Michael Kazhdan, Matthew Bolitho, and Hugues Hoppe. Poisson surface reconstruction. In Proceedings of the fourth Eurographics symposium on Geometry processing, 2006. 2
- [22] Lei Ke, Yu-Wing Tai, and Chi-Keung Tang. Deep occlusionaware instance segmentation with overlapping bilayers. In CVPR, 2021. 1
- [23] Hyunjin Kim and Minhyuk Sung. Partstad: 2d-to-3d part segmentation task adaptation. arXiv:2401.05906, 2024. 1, 2
- [24] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 3
- [25] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 2
- [26] Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. Salad: Part-level latent diffusion for 3d shape generation and manipulation. In ICCV, 2023. 3
- [27] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In ECCV. Springer, 2025. 3
- [28] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In CVPR, 2022. 2
- [29] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusionsdf: Text-to-shape via voxelized diffusion. In CVPR, 2023.

- 3

[30] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 3,

- 4

- [31] Yangyan Li, Rui Bu, Mingchao Sun, Wei Wu, Xinhan Di, and Baoquan Chen. Pointcnn: Convolution on x-transformed points. In NeurIPS, 2018. 2
- [32] Huan Ling, David Acuna, Karsten Kreis, Seung Wook Kim, and Sanja Fidler. Variational amodal object completion. Advances in Neural Information Processing Systems, 2020. 1
- [33] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4

- [34] Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. Part123: part-aware 3d reconstruction from a single-view image. In ACM SIGGRAPH 2024 Conference Papers, 2024. 2
- [35] Minghua Liu, Yinhao Zhu, Hong Cai, Shizhong Han, Zhan Ling, Fatih Porikli, and Hao Su. Partslip: Low-shot part segmentation for 3d point clouds via pretrained image-language models. In CVPR, 2023. 1, 2
- [36] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022. 4
- [37] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 1
- [38] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In CVPR,

2024. 1

- [39] William E Lorensen and Harvey E Cline. Marching cubes: A high resolution 3d surface construction algorithm. In Seminal graphics: pioneering efforts that shaped the field. 1998. 4
- [40] Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, and Shubham Tulsiani. Autosdf: Shape priors for 3d completion, reconstruction and generation. In CVPR, 2022. 3
- [41] Kaichun Mo, Shilin Zhu, Angel X Chang, Li Yi, Subarna Tripathi, Leonidas J Guibas, and Hao Su. Partnet: A largescale benchmark for fine-grained and hierarchical part-level 3d object understanding. In CVPR, 2019. 2
- [42] Andrew Nealen, Takeo Igarashi, Olga Sorkine, and Marc Alexa. Laplacian mesh optimization. In Proceedings of the 4th international conference on Computer graphics and interactive techniques in Australasia and Southeast Asia, 2006. 2
- [43] Duc Thanh Nguyen, Binh-Son Hua, Khoi Tran, Quang-Hieu Pham, and Sai-Kit Yeung. A field model for repairing 3d shapes. In CVPR, 2016. 3
- [44] Ege Ozguroglu, Ruoshi Liu, D´ıdac Sur´ıs, Dian Chen, Achal Dave, Pavel Tokmakov, and Carl Vondrick. pix2gestalt: Amodal segmentation by synthesizing wholes. In CVPR. IEEE Computer Society, 2024. 1
- [45] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 4
- [46] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 1
- [47] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In CVPR, 2017. 2
- [48] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In NeurIPS, 2017. 2
- [49] Lu Qi, Li Jiang, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Amodal instance segmentation with kins dataset. In CVPR,

2019. 1

- [50] Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. In NeurIPS, 2022. 2
- [51] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2, 3
- [52] Yuchen Rao, Yinyu Nie, and Angela Dai. Patchcomplete: Learning multi-resolution patch priors for 3d shape completion on unseen categories. NeurIPS, 2022. 2, 3, 5
- [53] N Dinesh Reddy, Robert Tamburo, and Srinivasa G Narasimhan. Walt: Watch and learn 2d amodal representation from time-lapse imagery. In CVPR, 2022. 1
- [54] Jaehyeok Shim, Changwoo Kang, and Kyungdon Joo. Diffusion-based signed distance fields for 3d shape generation. In CVPR, 2023. 3
- [55] Minhyuk Sung, Vladimir G Kim, Roland Angst, and Leonidas Guibas. Data-driven structural priors for shape completion. ACM Transactions on Graphics (TOG), 2015. 2
- [56] George Tang, William Zhao, Logan Ford, David Benhaim, and Paul Zhang. Segment any mesh: Zero-shot mesh part segmentation via lifting segment anything 2 to 3d. arXiv:2408.13679, 2024. 1, 2
- [57] Anh Thai, Weiyao Wang, Hao Tang, Stefan Stojanov, Matt Feiszli, and James M Rehg. 3x2: 3d object part segmentation by 2d semantic correspondences. arXiv preprint arXiv:2407.09648, 2024.
- [58] Ardian Umam, Cheng-Kun Yang, Min-Hung Chen, JenHui Chuang, and Yen-Yu Lin. Partdistill: 3d shape part segmentation by vision-language model distillation. arXiv:2312.04016, 2023. 2
- [59] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024. 3, 4
- [60] Yuheng Xue, Nenglun Chen, Jun Liu, and Wenyun Sun. Zerops: High-quality cross-modal knowledge transfer for zeroshot 3d part segmentation. arXiv:2311.14262, 2023. 2
- [61] Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao, and Xihui Liu. Sampart3d: Segment any part in 3d objects, 2024. 1, 2, 3, 5, 7
- [62] Xiaohang Zhan, Xingang Pan, Bo Dai, Ziwei Liu, Dahua Lin, and Chen Change Loy. Self-supervised scene deocclusion. In CVPR, 2020. 1
- [63] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions on Graphics (TOG), 2023. 1, 3
- [64] Bowen Zhang, Tianyu Yang, Yu Li, Lei Zhang, and Xi Zhao. Compress3d: a compressed latent space for 3d generation from a single image. In ECCV. Springer, 2025. 3
- [65] Junzhe Zhang, Xinyi Chen, Zhongang Cai, Liang Pan, Haiyu Zhao, Shuai Yi, Chai Kiat Yeo, Bo Dai, and Chen Change

- Loy. Unsupervised 3d shape completion through gan inversion. In CVPR, 2021. 3
- [66] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 2024. 1, 3, 4
- [67] Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip HS Torr, and Vladlen Koltun. Point transformer. In ICCV, 2021. 2
- [68] Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation. Advances in Neural Information Processing Systems, 2024. 3, 4
- [69] Ziming Zhong, Yanyu Xu, Jing Li, Jiale Xu, Zhengxin Li, Chaohui Yu, and Shenghua Gao. Meshsegmenter: Zero-shot mesh semantic segmentation via texture synthesis. In ECCV. Springer, 2024. 1, 2
- [70] Yuchen Zhou, Jiayuan Gu, Xuanlin Li, Minghua Liu, Yunhao Fang, and Hao Su. Partslip++: Enhancing low-shot 3d part segmentation via multi-view instance segmentation and maximum likelihood estimation. arXiv:2312.03015, 2023. 1
- [71] Yan Zhu, Yuandong Tian, Dimitris Metaxas, and Piotr Doll´ar. Semantic amodal segmentation. In CVPR, 2017. 1

## HoloPart: Generative 3D Part Amodal Segmentation Supplementary Material

### 6. Supplementary Material

##### 6.1. Implementation Details

The VAE consists of 24 transformer blocks, with 8 blocks functioning as the encoder and the remaining 16 as the decoder. The part diffusion model consists of 10 DiT layers with a hidden size of 2048, and the context-aware attention block consists of 8 self-attention blocks. To balance effectiveness with training efficiency, we set the token number for our part diffusion to 512. The latent tokens, encoded by the context-aware attention block, have a dimension of (512, 512), which are integrated into the part diffusion model via cross-attention. We fine-tune the part diffusion model using the ABO [9] dataset with 4 RTX 4090 GPUs for approximately two days, using the Objaverse [12] dataset with 8 A100 GPUs for around four days.

We set the learning rate to 1e-4 for both the pretraining and finetuning stages, using the AdamW optimizer. During training, as illustrated in Figure 2, we sample 20,480 points from the overall shape, which serve as the keys and values, while 512 points are sampled from each segmented part to serve as the query. This results in the context latent dimensions being (512, 512). For each point, we use the position embedding concatenated with a normal value as the input feature. After passing through the denoising UNet, we obtain shape latents of dimensions (512, 2048), representing the complete part’s shape. Subsequently, we use the 3D spatial points to query these shape latents and employ a local marching cubes algorithm to reconstruct the complete part mesh. The local bounding box is set to be 1.3 times the size of the segmented part’s bounding box to ensure complete mesh extraction.

##### 6.2. Data Curation Details

We develop a set of filtering rules to extract 3D objects with a reasonable part-wise semantic distribution from 3D asset datasets. The specific rules are as follows:

- • Mesh Count Restriction: We select only 3D objects with a mesh count within a specific range (2 to 15) to avoid objects that are either too simple or too complex (such as scenes or architectural models). The example data filtered out by this rule is shown in Figure 9 (a).
- • Connected Component Analysis: For each object, we render both frontal and side views of all parts and calculate the number of connected components in the 2D images. We then compute the average number of connected components per object, as well as the top three average values. An empirical threshold (85% of the connected component distribution) is used to filter out objects with

severe fragmentation or excessive floating parts (floaters). The example data filtered out by this rule is shown in Figure 9 (b).

• Volume Distribution Optimization: We analyze the volume distribution among different parts and ensure a balanced composition by removing or merging small floating parts and filtering out objects where a single part dominates excessively (e.g., cases where the alpha channel of the rendered image overlaps with the model rendering by up to 90%). The example data filtered out by this rule is shown in Figure 9 (c).

##### 6.3. Amodal Segmentation Results

To evaluate the amodal segmentation task, we further incorporate SAMPart3D and completion methods to perform amodal segmentation on the PartObjaverse-Tiny dataset. The quantitative comparison is presented in Table 4.

##### 6.4. More Ablation Analysis

Semantic and Instance Part Completion. Traditionally, segmentation definitions fall into two categories: semantic segmentation and instance segmentation. Similarly, we process our 3D parts from the ABO dataset according to these two settings. For example, in the semantic part completion setting, we consider all four chair legs as a single part, whereas in the instance part completion setting, they are treated as four separate parts. Our model is capable of handling both settings effectively. We train on the mixed dataset and present the completion results for a single bed using the same model weight, as shown in Figure 8.

Necessity of Context-Aware Attention. To emphasize the importance of our proposed context-aware attention block, we provide both quantitative analysis (refer to Section 4.3) and qualitative comparisons. As shown in Figure 10, the absence of context-aware attention results in a lack of guidance for completing individual parts, leading to inconsistent and lower-quality completion outcomes.

Qualitative Comparison of Different Guidance Scales. In Section 4.3, we provide a quantitative analysis of various guidance scales. Additionally, We illustrate the qualitative comparison of different guidance scales in Figure 11. Our findings indicate that excessively large or small guidance scales can adversely impact the final completion results. Through experimentation, we identify 3.5 as an optimal value for achieving balanced outcomes.

Learning Rate Setting. During the fine-tuning stage, we experiment with a weighted learning rate approach, where the parameters of the denoising U-Net are set to 0.1 times that of the context-aware attention block. However, we ob-

Method Overall Hum Ani Dai Bui Tra Pla Food Ele SDFusion 0.264 0.241 0.232 0.282 0.365 0.323 0.230 0.185 0.254

PatchComplete 0.289 0.267 0.258 0.295 0.382 0.314 0.247 0.231 0.291 DiffComplete 0.231 0.197 0.193 0.252 0.307 0.264 0.206 0.198 0.235 Finetune-VAE 0.178 0.138 0.114 0.202 0.279 0.213 0.140 0.141 0.198

Chamfer ↓

Ours 0.134 0.094 0.086 0.155 0.210 0.144 0.109 0.110 0.162 SDFusion 0.169 0.159 0.191 0.161 0.124 0.117 0.201 0.234 0.168

PatchComplete 0.086 0.079 0.097 0.079 0.076 0.076 0.105 0.091 0.084 DiffComplete 0.102 0.115 0.121 0.093 0.073 0.087 0.122 0.109 0.098 Finetune-VAE 0.347 0.370 0.406 0.313 0.299 0.277 0.412 0.381 0.320

IoU ↑

Ours 0.455 0.508 0.513 0.415 0.360 0.379 0.522 0.529 0.416 SDFuison 0.273 0.263 0.306 0.260 0.208 0.198 0.316 0.364 0.271

PatchComplete 0.149 0.139 0.168 0.138 0.133 0.134 0.179 0.157 0.147 DiffComplete 0.177 0.198 0.206 0.162 0.129 0.153 0.206 0.189 0.170 Finetune-VAE 0.473 0.507 0.543 0.433 0.417 0.395 0.540 0.513 0.439

F-Score ↑

###### Ours 0.570 0.626 0.628 0.529 0.477 0.497 0.627 0.645 0.533

###### Table 4. 3D part amodal segmentation results on PartObjaverse-Tiny, reported in Chamfer Distance, IoU, F-Score and Success Rate.

[Figure 257]

##### 6.6. More Results on PartObjaverse-Tiny

[Figure 258]

Overall Mesh Semantic Parts Instance Parts

[Figure 259]

[Figure 260]

We present more qualitative results on the PartObjaverseTiny dataset in Figures 14 and 15. Our method can effectively complete the details of parts and maintain overall consistency, which other methods cannot achieve.

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

##### 6.7. Limitations and Future Works

The outcome of HoloPart is influenced by the quality of input surface masks. Unreasonable or low-quality masks may lead to incomplete results. Therefore, a better approach moving forward would be to use our method to generate a large number of 3D part-aware shapes, which can then be used to train part-aware generation models.

- Figure 8. Ablation study of semantic and instance part completion.

serve that this approach results in unstable training and negatively impacts the final outcomes. We present the comparison of generated parts with different learning rate training setting in Figure 11.

##### 6.5. More Results of 3D Part Amodal Segmentation

In Figure 13, we showcase additional examples of 3D part amodal segmentation applied to generated meshes from 3D generation models. Initially, we employ SAMPart3D [61] to segment the generated meshes, resulting in several surface masks. Subsequently, our model completes each segmented part, enabling the reconstruction of a consistent overall mesh by merging the completed parts. For instance, as demonstrated in Figure 13, our model effectively completes intricate components such as glasses, hats, and headsets from the generated meshes. This capability supports a variety of downstream tasks, including geometry editing, geometry processing, and material editing.

(a)

[Figure 270]

(b)

[Figure 271]

[Figure 272]

(c)

[Figure 273]

Figure 9. Examples of data filtered out by rules.

W Context-Aware W/o Context-Aware

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Figure 10. The absence of context-aware attention leads to a lack of guidance for completing individual components, resulting in inconsistent and lower-quality outcomes.

Guidance Scale: 1.5 Guidance Scale: 3.5 Guidance Scale: 5.0 Guidance Scale: 7.5

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

Figure 11. Visualization of generated parts across different guidance scales.

Same Learning Rate Weighted Learning Rate

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Figure 12. Qualitative comparison of different learning rate settings.

Generated Mesh Surface Segments Complete Parts Merged Parts

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

Figure 13. More Results of 3D Part Amodal Segmentation.

Overall Mesh & Segmented Parts PatchComplete DiffComplete Finetune-VAE Ours GT

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

[Figure 374]

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

Figure 14. More qualitative results on the PartObjaverse-Tiny dataset.

Overall Mesh & Segmented Parts PatchComplete DiffComplete Finetune-VAE Ours GT

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

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

[Figure 422]

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

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

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

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

Figure 15. More qualitative results on the PartObjaverse-Tiny dataset.

