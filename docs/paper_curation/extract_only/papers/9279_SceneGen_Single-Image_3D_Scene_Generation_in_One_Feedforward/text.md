### SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass

Yanxu Meng∗, Haoning Wu∗, Ya Zhang, Weidi Xie School of Artificial Intelligence, Shanghai Jiao Tong University

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

## arXiv:2508.15769v2[cs.CV]9Dec2025

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

Figure 1. Overview. Our proposed SceneGen framework takes a single scene image and its corresponding object masks as inputs, and efficiently generates multiple 3D assets with coherent geometry, texture, and spatial arrangement in a single feedforward pass.

#### Abstract

Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen’s direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architecture yields improved generation performance when multiple images are provided; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robustness of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: https://mengmouxu.github.io/SceneGen.

3D content generation has recently attracted significant research interest, driven by its critical applications in VR/AR and embodied AI. In this work, we tackle the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for extra optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module.

“Everything you can imagine is real.”

—— Pablo Picasso

*: These authors contribute equally to this work.

#### 1. Introduction

The growing demand for immersive digital environments in applications such as virtual/augmented reality (VR/AR) and embodied AI has spurred significant advancements in 3D content generation [6, 8, 9, 13, 14, 28]. While early efforts primarily focus on synthesizing individual 3D assets [33, 59, 69], recent research attention has shifted to the more challenging task of 3D scene generation. Generating realistic 3D scenes [5, 11, 15, 16, 19, 64, 65], whether conditioned on input text or images, requires synthesizing multiple assets with accurate geometry, texture, and spatial relationships. This challenge fundamentally hinges on two key capabilities: (i) 3D asset generation for creating plausible asset geometric topologies from limited textual or visual input; and (ii) spatial arrangement for managing inter-object spatial relationships to ensure physical plausibility, such as support, occlusion, and other interactions among assets.

In general, existing works fall into two paradigms: (i) retrieval-based methods [12, 38, 49, 61] typically employ LLMs for layout planning and retrieve matching 3D assets from existing libraries to assemble scenes. Though straightforward, their flexibility is constrained by the coverage of available asset libraries; (ii) two-stage approaches [18, 35, 62] first synthesize individual 3D assets and then employ vision-language models (VLMs) or optimization techniques to refine scene structure and spatial arrangement. While more flexible, their reliance on iterative optimization inevitably leads to inefficiency and error accumulation. The most relevant works to ours are PartCrafter [34] and MIDI [23], which generate parts or multiple assets from a single image. However, they still suffer from limited synthesis fidelity and inaccurate spatial relations among assets.

To tackle the aforementioned challenges, we propose SceneGen, a novel 3D scene generation model designed to simultaneously generate multiple assets, including their geometry, texture, and spatial positions, from a single scene image in a single feedforward pass (Figure 1). Concretely, our framework builds upon an existing single-asset generation model [59] and incorporates three key modules: feature extraction, feature aggregation, and output.

Specifically, the feature extraction module first strategically leverages off-the-shelf visual [42] and geometric [53] encoders to extract both asset-level and scene-level representations. Subsequently, our proposed feature aggregation module, composed of local and global attention blocks, effectively integrates these visual and geometric features while facilitating inter-asset interactions during generation to ensure geometrically plausible topologies. Finally, leveraging this comprehensive scene context, the output module can directly decode the generated latent features into the assets’ relative position, geometry, and texture via a position head and a pre-trained structure decoder.

Moreover, despite being trained exclusively on single-

image samples, SceneGen exhibits remarkable generalization to multi-image input scenarios, yielding even better generation quality, which primarily stems from our dedicated architectural design. To ensure a comprehensive and reliable evaluation of SceneGen, we systematically adopt multiple metrics focusing on both geometric and visual quality. Both quantitative and qualitative results demonstrate that our proposed SceneGen significantly outperforms previous methods in terms of generation quality and efficiency, which can generate a textured scene with four assets in approximately 2 minutes on a single A100 GPU.

The rest of this paper is organized as follows: Sec. 2 provides a comprehensive review and discussion of related literature. Sec. 3 elaborates on our proposed SceneGen framework. Sec. 4 presents extensive quantitative and qualitative evaluations. Finally, Sec. 5 concludes with key insights and contributions. To our knowledge, SceneGen is the first 3D scene generation model capable of simultaneously synthesizing geometry, texture, and relative positions of multiple 3D assets in a single feedforward pass, without requiring per-scene optimization. We believe this work will inspire future advances in high-quality, efficient 3D content generation and facilitate diverse downstream applications.

- 2. Related Work
- 3D visual perception. Extensive research has advanced 3D visual perception, where traditional methods like SfM [48, 52] rely on computationally intensive optimization for 3D reconstruction. Notably, emerging feedforward methods [3, 27, 30, 51, 53–55, 68, 71] have demonstrated efficient 3D perception, with DUSt3R [55] pioneering this trend and VGGT [53] establishing a minimalist yet powerful paradigm that distills geometric priors from large-scale data without explicit 3D inductive biases or optimizations.

3D asset synthesis. Typically, 3D asset synthesis aims to generate object-centric geometry and texture from text or image inputs. The recent success of diffusion models [22] in

- 2D generation [37, 43, 47, 56, 57] has inspired the development of learning-based, scalable 3D content [6, 8, 9, 13, 14, 28] generation, which produce 3D asset in various representations, including explicit forms such as point clouds [40], voxels [25, 41], and SDFs [4, 32], as well as implicit ones like 3D Gaussians [20, 67] and NeRFs [1, 31, 60]. Subsequent advances leverage VAEs [29] for compressing 3D geometry or textures [33, 59, 69] and adopt hybrid meshtexture pipelines [24, 26, 58, 66], with TRELLIS [59] demonstrating scalable, high-fidelity generation via structured latents. Nevertheless, these methods remain restricted to single-asset synthesis and fundamentally lack the capability to model complex multi-asset scenes.
- 3D scene generation. Beyond single-asset synthesis, 3D scene generation is more challenging yet valuable, aiming

##### 3.1. Problem Formulation

[Figure 111]

[Figure 112]

|[Figure 113]<br><br>[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Our proposed SceneGen is a single-stage feedforward 3D scene generation model (GScene), which takes a scene image (IScene) containing N objects and corresponding masks ({mi}Ni=1) as input, simultaneously generating 3D asset structure and texture representations ({Si}Ni=1), and their relative positions ({Pi}Ni=1), formulated as:

(a) Scene image segmentation

Assets Latent

Pos. Tokens

Pre-process Modules

VLM

Florence-2 Depth Model …

Multi-Instance Structure Generation Module

ObjectN

###### Object1

[Figure 121]

###### …

{(Si,Pi)}Ni=1 = GScene(IScene,{mi}Ni=1)

[Figure 122]

Position Head

Decoder

3D Instance Generation

3D Instance Retrieval

Multi-Instance 3D Asset Generation

Textured Assets

Relative Positions

Here, the position of each asset relative to a pre-selected query asset, is denoted as Pi = [ti,qi,si] ∈ R8, comprising ti ∈ R3 (translation), qi ∈ R4 (rotation quaternion), and si ∈ R1 (scale factor). By default, we select the asset with i = 1 as the query asset, with its parameters fixed as: tquery = [0,0,0], qquery = [1,0,0,0], squery = 1.

[Figure 123]

Scene Composition

Point Cloud Alignment

Post-optimization Steps

VLM Scene Graph Render Projection

Physical Constrain …

(b) Two-stage approaches (c) Single-image to 3D scene

###### (d) Our SceneGen paradigm

- Figure 2. 3D Scene Generation. (a) Existing methods typically require segmenting target objects from the scene image; (b) Twostage methods like CAST [62] sequentially retrieve or generate individual assets, then assemble them via post-processing; (c) Methods such as MIDI [23] directly generate multiple assets from a single image, but suffer from blurry details and unreasonable spatial layouts; (d) In contrast, our SceneGen jointly synthesizes the geometry, texture, and spatial positions of multiple assets in a single feedforward pass, producing plausible 3D scenes.

##### 3.2. SceneGen

SceneGen framework (GScene) comprises three key stages: (i) feature extraction, employing a scene visual encoder (ΦV ) and a scene geometric encoder (ΦG) to extract visual and structural features within the scene, implemented using pre-trained DINOv2 [42] and VGGT [53], respectively; (ii) feature aggregation, comprising M DiT [43] blocks, each integrating a local attention block, a global attention block, and a feedforward network; and (iii) output module, which introduces a position head (Ψpos) for predicting the spatial locations of assets and adopts off-theshelf sparse structure (SS) and structured latents (SLAT) decoders [59] for decoding scene geometry structures. By integrating these complementary modules, our SceneGen effectively captures both local asset-level and global scenelevel features, enabling it to simultaneously generate multiple 3D assets and predict their relative positions.

to produce multiple coordinated, physically plausible assets within a scene. Prior text-based approaches primarily leverage LLMs for layout planning [12, 38, 49, 61] and retrieve suitable assets from existing libraries. Subsequent image-based methods employ segmentation [5, 10, 15, 19, 64], scene graphs [11, 16, 65] and depth/point cloud alignment [10, 50, 62] to assist in multi-asset generation and arrangement. As depicted in Figure 2 (b), recent optimization-based methods [18, 35, 62] adopt VLMs for post-processing, refining scene structures via image- or text-guided adjustments, but inevitably suffer from inefficiency. Other works (Figure 2 (c)), such as MIDI [23] and PartCrafter [34], explore scene generation conditioned on a single image, but inherently sacrifice reconstruction fidelity due to their reliance on canonical-space representations. To overcome these limitations, our proposed SceneGen uniquely integrates asset-level and scene-level features, enabling robust and efficient 3D scene generation.

Feature extraction. SceneGen starts with extracting both local and global features from a given scene image (Iscene) with the visual encoder (ΦV ) and geometric encoder (ΦG). Specifically, for each object with its corresponding segmentation mask (mi), we obtain four complementary feature representations: (i) the object’s individual visual features (FiV ); (ii) the visual features of its mask (Fimask); (iii) scene global visual features (FglobalV ); and (iv) the global geometric features (Fglobalgeo ), formulated as:

#### 3. Method

FiV = ΦV (Iscene ⊗ mi), Fimask = ΦV (mi), FglobalV = ΦV (Iscene), Fglobalgeo = ΦG(Iscene)

In this work, we present SceneGen, designed to jointly perform 3D asset generation within scenes and predict relative spatial positions among assets. Here, we first formally describe our problem formulation in Sec. 3.1; followed by elaboration on our model architecture and training method in Sec. 3.2 and Sec. 3.3, respectively; finally, we extend SceneGen to multi-view input scenarios in Sec. 3.4.

Here, ⊗ denotes pixel-wise multiplication. These features are then concatenated along the sequence length dimension into a unified scene context (Fiscene), formulated as:

Fiscene = [FiV ;Fimask;FglobalV ;Fglobalgeo ]

𝒕

###### Timestep Embedding

(iii) Output Module

Pos. Tokens

|[Figure 124]|
|---|

|[Figure 125]|
|---|

Unpatchify

|[Figure 126]|
|---|

|[Figure 127]|
|---|

Patchify

Asset Self-Attn. (AS)

Asset Cross-Attn. (AC)

Scene Self-Attn. (SS)

Scene Cross-Attn. (SC)

Linear

Linear

[Figure 128]

SS Decoder

FFN

Position Embed.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

noisy sparse structure latents 𝐱

###### Local Attention Global Attention

Position Head

[Figure 134]

SLAT Decoder

Pos. Tokens

Ψ

[Figure 135]

[Figure 136]

ℱ

(ii) Feature Aggregation

𝑴× DiT Blocks

# Scene

ℱ ℱ

ℱ ℱ

Query Pos.

Query Asset

Visual Encoder

Geometric Encoder

- Asset 2 Pos.

- Asset 3 Pos.

Asset 2

[Figure 137]

[Figure 138]

Φ

Φ

(i) Feature Extraction

[Figure 139]

Asset 3

Conditions

Asset 4 Pos.

Asset 4

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

|[Figure 143]|
|---|

[Figure 144]

: position encoding

: concatenate : frozen parameters : trainable parameters

|[Figure 145]|
|---|

|[Figure 146]|
|---|

|[Figure 147]|
|---|

|[Figure 148]|
|---|

[Figure 149]

[Figure 150]

scene image 𝑰

3D scene containing multiple 3D assets object masks 𝒎

segmented objects

- Figure 3. Architecture Overview. SceneGen takes a single scene image with multiple objects and corresponding segmentation masks as input. A pre-trained local attention block first refines the texture of each asset. Then, our introduced global attention block integrates asset-level and scene-level features extracted by dedicated visual and geometric encoders. Finally, two off-the-shelf structure decoders and our position head decode these latent features into multiple 3D assets with geometry, texture, and relative spatial positions.

Feature aggregation. To integrate the extracted scene context features (Fiscene), SceneGen employs a feature aggregation module, that enables the simultaneous generation of multiple 3D assets. This module comprises a local attention block that refines details of individual assets, a global attention block that incorporates scene context to facilitate interasset interactions, and a feedforward network. Concretely, the local attention blocks and feedforward networks are initialised from pre-trained weights from TRELLIS [59], a flow-matching [36] model designed to synthesize 3D content from noisy sparse structure latents. For clarity and conciseness, given the sparse structure latents ({xi}Ni=1, where each xi ∈ RT×C) of N objects in a scene, we denote the standard attention mechanism as Attention(Q,K,V), and elaborate on a single DiT block as follows.

Similar to [53], we initialize one learnable position token (pi) and four register tokens (ri) [7] for refined features of each object (xACi ), denoted as: xˆi = [pi;ri;xACi ], where [·;·] refers to concatenation along the token length dimension. Notably, we assign a unique position token (pquery) and register tokens (rquery) to the query asset, while adopting shared position token (pi) and register tokens (ri) for other assets. For each asset feature (xˆi ∈ RT×C), we concatenate them along the token sequence dimension to form a unified scene representation (X ∈ R(N·T)×C), which is processed by our scene-level self-attention layer, resulting in updated tokens of each asset ({xSSi }Ni=1), formulated as: {xSSi }Ni=1 = Attention(X,X,X)

Through this process, intra-asset and inter-asset information aggregation establishes essential shape and position awareness for coherent multi-asset generation. We then employ scene-level cross-attention to integrate multiple preextracted scene-aware features, thus incorporating 3D geometric context. The features of each asset are updated into geometry-aware representations ({xSCi }Ni=1), denoted as:

The local attention block aims to enhance details of individual objects through asset-level self-attention (AS) and cross-attention (AC). To be specific, it focuses on fusing the latent features of each object (xi) with their corresponding visual features (FiV ) to yield refined representations of each object (xACi ), which can be formulated as:

xASi = Attention(xi,xi,xi) xACi = Attention(xASi ,FiV ,FiV )

xSCi = Attention(xSSi ,Fiscene,Fiscene)

This preserves object-specific details while integrating global geometric constraints, which effectively addresses occlusion challenges and enables geometric refinement.

To establish inter-dependencies among 3D assets, we propose a global attention block, comprising scene-level self-attention (SS) and cross-attention (SC), which capture inter-object relationships and integrate scene geometry, respectively. Consequently, this design ensures physically plausible spatial arrangements of generated assets.

Output module. After passing through M DiT blocks, we obtain the updated position tokens ({pˆ}Ni=1) and latent features ({x˜}Ni=1) of each generated asset, which are subsequently decoded into their relative spatial positions and

detailed 3D representations (structure and texture), respectively. For relative positions, we concatenate the position tokens of all non-query assets, which are then decoded into corresponding 8D position vectors ({Pˆ i}Ni=2) by our proposed position head (Ψpos), comprising four self-attention layers and a linear layer, denoted as:

{Pˆ i}Ni=2 = {[ˆti,qˆi,sˆi]}Ni=2 = Ψpos({pˆi}Ni=2)

Here, each vector (Pˆ i) represents an asset’s spatial position (translation, rotation, and scale) relative to the preselected query asset (i = 1). Furthermore, the latent features can be directly decoded into the geometry and texture of each asset ({Sˆ}Ni=1) using off-the-shelf sparse structure generator (GS) and structured latents generator (GL) from TRELLIS [59], formulated as:

{Sˆ}Ni=1 = GL(GS({x˜}Ni=1))

##### 3.3. Training

During training, only the global attention blocks, learnable position tokens, and position head are optimized, with all other parameters frozen to facilitate efficient training, as depicted in Figure 3. The technical details regarding training data and loss function designs are presented below.

Training data. Our SceneGen model is trained on the 3DFUTURE [14] dataset, containing photorealistic scene renderings with instance masks and asset annotations. This dataset comprises 12K training scenes and 4.8K test scenes, each featuring a scene image with one or multiple objects. To better capture inter-object spatial relationships, we augment the training set by iteratively designating each asset as the query asset while randomly permuting the remaining assets, which expands the effective training samples to 30K. Training objectives. Our SceneGen model is trained endto-end using a composite loss function (L) comprising three key components: (i) the average conditional flow matching [36] loss (Lcfm), applied to each generated asset for supervising asset generation; (ii) the position loss (Lpos) for maintaining accurate relative spatial arrangements among assets; and (iii) the voxel-space collision loss (Lcoll) for enforcing physically plausible object placement. The overall objective function (L) combines these components with a weighting factor (λ), which can be formulated as:

L = Lcfm + λ(Lpos + Lcoll)

Concretely, the flow matching loss establishes straight probability paths between distributions via linear interpolation: xi(t) = (1−t)x0i+tϵ, where ϵ ∼ N(0,I), t ∈ [0,1], and x0i denotes the noise-free sparse structure latents for each of the N assets. The conditional flow matching objective (Lcfm) learns a parameterized function vθ to approximate the ve-

locity field (v(xi(t),t) = ∇txi(t)), represented as:

1 N

Lcfm(θ) =

N

Et,ϵ∥vθ(xi(t),t) − (ϵ − x0i)∥22

i=1

The position loss (Lpos) adopts a µ-weighted Huber loss (∥· ∥δP

) between the predicted positions (Pˆi = [ˆti,qˆi,sˆi]) for all non-query assets (i ∈ [2,...,N]) and their ground truth (Pi = [ti,qi,si]), denoted as:

N

(µt∥(ˆti − ti)/dscene∥δP

Lpos =

i=2

+µq∥qˆi − qi∥δP

+ µs∥sˆi − si∥δP

)

Here, the translation error component is normalized by the scene scale (dscene) of each sample to mitigate numerical instability caused by varying query asset selections. This stabilizes translation loss during training while improving generalization across distinct query asset configurations.

The collision loss (Lcoll) quantifies surface collision in a 64 × 64 × 64 voxel grid (V ). Specifically, the predicted sparse structure latents (x˜i) are decoded into point clouds ({pi}Li=1) via a pre-trained sparse structure decoder from TRELLIS [59], then transformed using predicted pose parameters (Pˆ i) and voxelized into V . The collision loss is defined as the ratio of overlapping surface voxels to all surface voxels, using the Huber loss (∥ · ∥δC

), denoted as:

I[V i > 1] i I[V i > 0]∥δC

= ∥ i

Lcoll = ∥IoUscene∥δC

Ideally, IoUscene = 0 indicates there are no asset collisions.

##### 3.4. Extension to Multi-view Inputs

Despite being trained exclusively on single-image samples, our model exhibits inherent multi-view compatibility, enabled by its flexible feature extraction and conditioning strategy. Given a scene with K input views ({Ikscene}Kk=1), the visual features (FVk ) for each view are extracted independently via the visual encoder (ΦV ), while the geometric features are derived from a unified scene representations encoded by aggregating information across all views using the geometric encoder (ΦG), denoted as:

Fgeok = ΦG({Ijscene}Kj=1)[k]

The final asset positions are determined by averaging the predictions across all views. Experimental results (detailed in Sec. 4.3) indicate that this multi-view inference scheme improves generation quality by leveraging better geometric understanding, despite the model having never been explicitly fine-tuned on such multi-view inputs.

Visual Metrics Inference CD-S↓ CD-O↓ F-Score-S↑ F-Score-O↑ IoU-B↑ PSNR↑ SSIM↑ LPIPS↓ FID↓ CLIP-S↑ DINO-S↑ Time (s)

Geometric Metrics Image

Instance Specific

Method

Category

— — — — — —

Scene GT-Render

###### 7.2

PartCrafter [34] 0.2027 — 40.43 — —

— — — — — DepR [72] 0.0518 0.0862 63.02 47.66 0.2989

— — — — — —

Scene GT-Render

11.6

— — — — — Gen3DSR [10] 0.0521 0.0935 61.26 41.26 0.2978

15.92 0.8885 0.1730 63.95 0.8059 0.4334

Scene GT-Render

179.0

15.43 0.8899 0.1660 78.26 0.7950 0.4416 MIDI∗ [23] 0.0501 0.0602 68.74 61.04 0.2493

16.93 0.8814 0.1778 22.75 0.8711 0.6892

Scene GT-Render

42.5

###### 15.45 0.8814 0.1711 28.26 0.8706 0.7034 SceneGen 0.0118 0.0138 90.60 89.73 0.5818

###### 16.76 0.8903 0.1417 19.59 0.9152 0.8322

Scene GT-Render

26.0

###### 17.59 0.8991 0.1234 12.34 0.9236 0.8702

- Table 1. Quantitative Comparisons on the 3D-FUTURE Test Set. We evaluate the geometric structure using scene-level Chamfer Distance (CD-S) and F-Score (F-Score-S), object-level Chamfer Distance (CD-O) and F-Score (F-Score-O), and volumetric IoU of object bounding boxes (IoU-B). For visual quality, CLIP-S and DINO-S represent CLIP and DINOv2 image-to-image similarity, respectively. We report the time cost for generating a single asset on a single A100 GPU, and ∗ indicates adopting MV-Adapter [24] for texture rendering.

#### 4. Experiments

This section starts with the experimental settings in Sec. 4.1, followed by comprehensive quantitative and qualitative evaluations in Sec. 4.2 and Sec. 4.3, respectively. Finally, we conduct ablation studies in Sec. 4.4.

##### 4.1. Experimental Settings

Implementation details. All experiments are conducted on 8× NVIDIA A100 GPUs, where we train SceneGen for 240 epochs using the AdamW [39] optimizer with a learning rate of 5×10−5 and a batch size of 8. The weighting factor λ decays dynamically within [0.2,1] using a decay factor of 0.99, and the thresholds of Huber loss δP and δC are set to 0.02 and 0.05, respectively. To handle varying numbers of assets across training scenes, each training step dynamically samples scenes containing identical asset counts. During inference, we adopt 25 sampling steps with the classifierfree guidance (CFG) weight set to w = 5.0.

Evaluation metrics. We assess the generated 3D scenes from both geometric and visual perspectives. For geometry, we reconstruct point clouds from the synthesized asset surfaces and align them with the ground truth using FilterReg [17] for faster and more accurate registration than traditional Iterative Closest Point (ICP [2]). We then compute commonly used point cloud metrics, Chamfer Distance (CD) and F-Score, at both scene and object levels, as well as the volumetric IoU of asset bounding boxes.

For visual quality, we focus on the scene texture rendering. Specifically, after alignment with the ground truth point cloud, we render the predicted scenes with Blender from the original input camera viewpoint. We consider two types of ground truth: (i) instance-masked scene images extracted using corresponding object masks, and (ii) images rendered from ground truth assets at the same viewpoint (excluding ambient lighting). We compare our rendered results with both types of ground truth using PSNR, SSIM, LPIPS [70],

FID [21], CLIP [44] similarity, and DINOv2 [42] similarity to assess the texture quality of generated assets. Regarding efficiency, we report the inference time cost for synthesizing a single 3D asset on a single A100 GPU. More details will be included in Sec. C.2 of the Appendix.

Baselines. We compare SceneGen with representative 3D scene generation methods, including PartCrafter [34], DepR [72], Gen3DSR [10], and MIDI [23], using their pre-trained models. Specifically, we adopt object masks to specify generation targets for all baselines except for PartCrafter, which does not support mask-based control. Instead, we directly provide PartCrafter with extracted objects and the number of assets as input. Moreover, as PartCrafter and DepR do not offer code for texture rendering, our evaluation of these methods focuses on geometric quality, while visual quality is compared with Gen3DSR and MIDI (relying on MV-Adapter [24] for texture synthesis).

Benchmarks. All evaluations are conducted on the 3DFUTURE [14] test set, comprising 4.8K scenes. Each scene contains a photorealistic rendered image with one or more objects and corresponding segmentation masks as input.

##### 4.2. Quantitative Results

As presented in Table 1, we draw the following key observations: (i) geometric quality: SceneGen consistently outperforms existing methods across all scene-level and assetlevel metrics. This stems from its joint integration of local asset features and global scene context during generation. The interactions among multiple assets facilitate the model in producing physically plausible geometric structures, while the position head further improves the structural realism by explicitly predicting spatial arrangements. (ii) visual quality: SceneGen can render high-quality textures for generated 3D assets without relying on any external texture generation models. Moreover, whether using masked scene images or ground-truth renderings as references, our

###### Scene Image GT Assets PartCrafter DepR Gen3DSR MIDI SceneGen (Ours)

||[Figure 151]<br><br>[Figure 152]|
|---|
<br><br>|[Figure 153]|
|---|
<br><br>|[Figure 154]|
|---|
<br><br>|[Figure 155]|
|---|
<br><br>|[Figure 156]|
|---|
<br><br>|[Figure 157]|
|---|
<br><br>|[Figure 158]|
|---|
<br><br>|[Figure 159]<br><br>[Figure 160]|
|---|
<br><br>|[Figure 161]|
|---|
<br><br>|[Figure 162]|
|---|
<br><br>|[Figure 163]|
|---|
<br><br>|[Figure 164]|
|---|
<br><br>|[Figure 165]|
|---|
<br><br>|[Figure 166]|
|---|
<br><br>|[Figure 167]<br><br>[Figure 168]|
|---|
<br><br>|[Figure 169]|
|---|
<br><br>|[Figure 170]|
|---|
<br><br>|[Figure 171]|
|---|
<br><br>|[Figure 172]|
|---|
<br><br>|[Figure 173]|
|---|
<br><br>|[Figure 174]|
|---|
|
|---|

||[Figure 175]<br><br>[Figure 176]|
|---|
<br><br>|[Figure 177]|
|---|
<br><br>|[Figure 178]|
|---|
<br><br>|[Figure 179]|
|---|
<br><br>|[Figure 180]|
|---|
<br><br>|[Figure 181]|
|---|
<br><br>|[Figure 182]|
|---|
<br><br>|[Figure 183]<br><br>[Figure 184]|
|---|
<br><br>|[Figure 185]|
|---|
<br><br>|[Figure 186]|
|---|
<br><br>|[Figure 187]|
|---|
<br><br>|[Figure 188]|
|---|
<br><br>|Not<br><br>Available|
|---|
<br><br>Not<br><br>Available|
|---|

- Figure 4. Qualitative Comparisons on the 3D FUTURE Test Set and ScanNet++. Our proposed SceneGen is capable of generating physically plausible 3D scenes featuring complete structures, detailed textures, and precise spatial relationships, demonstrating superior performance over prior methods in terms of both geometric accuracy and visual quality on both the synthetic and real-world datasets.

method consistently achieves the best performance across all metrics. This indicates that our synthesized assets are spatially closer to the ground truth while maintaining superior texture fidelity. and (iii) efficiency: While PartCrafter demonstrates a clear advantage in inference speed, it suffers from limited generation quality and controllability. In contrast, SceneGen achieves both superior quality and a strong balance between quality and efficiency, synthesizing a 3D scene containing four assets with geometry and textures within 2 minutes on a single A100 GPU.

on both the 3D FUTURE [14] test set and in-the-wild ScnaNet++ [63], where they still struggle with 3D scene generation: PartCrafter lacks controllability over the generated targets and often mistakenly merges distinct assets, while both PartCrafter and DepR are limited to geometry generation and cannot render textures. More critically, all these methods exhibit difficulties in accurately understanding the spatial relationships among assets. In contrast, our proposed SceneGen precisely predicts the spatial relationships among assets and synthesizes multiple 3D assets with accurate geometry and high-quality textures, without relying on any additional tools or optimizations.

In addition, while the baseline methods, e.g., PartCrafter, DepR, and MIDI have been trained on 3D-FRONT [13], which may overlap with our test data, our SceneGen still consistently outperforms them across all metrics, further demonstrating its effectiveness and superiority.

Extension to multi-image inputs. Benefiting from our architecture design, SceneGen can seamlessly handle multiimage inputs after being trained exclusively on singleimage samples. Given the lack of suitable datasets for quantitative evaluation, we qualitatively assess the impact of multi-image inputs by randomly sampling several scenes from ScanNet++ [63] and employing SAM2 [46] to obtain

##### 4.3. Qualitative Results

Comparisons with baselines. As illustrated in Figure 4, we qualitatively compare SceneGen with existing baselines

Visual Metrics CD-S↓ CD-O↓ F-Score-S↑ F-Score-O↑ IoU-B↑ PSNR↑ SSIM↑ LPIPS↓ FID↓ CLIP-S↑ DINO-S↑

Geometric Metrics Image

Fglobalgeo FglobalV Fimask ASS

Category

- 16.76 0.8903 0.1417 19.59 0.9152 0.8322

- 17.59 0.8991 0.1234 12.34 0.9236 0.8702

Scene GT-Render

0.0118 0.0138 90.60 89.73 0.5818

- 15.89 0.8845 0.1574 20.21 0.9049 0.8063

- 16.27 0.8918 0.1421 15.36 0.9125 0.8420

Scene GT-Render

0.0183 0.0266 83.33 74.71 0.4805

15.56 0.8806 0.1655 20.68 0.8980 0.7850 15.86 0.8873 0.1511 16.62 0.9046 0.8187

Scene GT-Render

0.0250 0.0286 79.08 73.46 0.4253

15.30 0.8773 0.1730 21.12 0.8932 0.7737 15.55 0.8837 0.1591 17.45 0.9000 0.8076

Scene GT-Render

0.0310 0.0290 75.20 73.17 0.3825

13.32 0.8418 0.2329 27.56 0.8399 0.6059 13.39 0.8464 0.2217 28.61 0.8440 0.6362

Scene GT-Render

0.0764 0.0352 54.21 70.55 0.1705

- Table 2. Ablations on SceneGen Variants. We progressively remove global geometric features (Fglobalgeo ), global visual features (FglobalV ), mask visual features (Fimask), and substitute the scene-level self-attention (ASS) to validate each component’s contribution to SceneGen.

[Figure 189]

[Figure 190]

sessing both the geometric and visual quality of synthesized scenes. Concretely, we investigate the impact of gradually removing global geometric features (Fglobalgeo ), global visual features (FglobalV ), mask visual features (Fimask), as well as substituting the scene-level self-attention block (ASS) with a simple asset-level self-attention block (AAS). As depicted in Table 2, we have the following observations: (i) Removing any of the aforementioned components degrades the overall performance, confirming their necessity in SceneGen; (ii) The geometric features primarily affect the structure of synthesized scenes, while the visual features further impact the visual quality; and (iii) The absence of scenelevel self-attention blocks eliminates inter-asset interactions during generation, leading to notable performance declines across all metrics. These results strongly demonstrate the necessity and effectiveness of our proposed feature extraction and aggregation modules for SceneGen.

Single Image Input

|[Figure 191]| |
|---|---|
| | |
|[Figure 192]<br><br>[Figure 193]| |

[Figure 194]

| |
|---|

[Figure 195]

|[Figure 196]<br><br>|
|---|

[Figure 197]

[Figure 198]

| |
|---|

[Figure 199]

[Figure 200]

Multi-Image Input

|[Figure 201]<br><br>[Figure 202]<br><br>[Figure 203]<br><br>[Figure 204]<br><br>[Figure 205]<br><br>[Figure 206]|
|---|

|[Figure 207]<br><br>[Figure 208]<br><br>[Figure 209]<br><br>[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]|
|---|

#### 5. Conclusion

In this paper, we present SceneGen, a novel framework that takes a single scene image and target object masks as input to simultaneously synthesize multiple 3D assets with structure, texture, and relative spatial positions in a single feedforward pass. Specifically, we incorporate dedicated visual and geometric encoders to extract both asset-level and scene-level features, which are effectively fused with our proposed feature aggregation module. Notably, through our meticulous design, SceneGen can naturally generalize to multi-image inputs and achieve even better generation fidelity. Quantitative and qualitative evaluations demonstrate that SceneGen produces physically plausible and mutually consistent 3D assets, significantly outperforming previous methods in terms of generation quality and efficiency.

- Figure 5. Qualitative Results with Multi-view Inputs. SceneGen can directly handle multi-view inputs in ScanNet++ and even achieves better generation quality, especially accurate structure.

segmentation masks of corresponding objects. As depicted in Figure 5, compared to single-image inputs, incorporating multi-view images leads to 3D assets with more complete geometry and finer texture details. This illustrates that SceneGen can adaptively integrate complementary information from multiple views to produce higher-quality 3D scenes, further validating its practicality and scalability. More qualitative results will be included in Sec C.1 of the Appendix.

#### Acknowledgments

##### 4.4. Ablation Studies

Weidi would like to acknowledge the funding from Scientific Research Innovation Capability Support Project for Young Faculty (ZY-GXQNJSKYCXNLZCXM-I22).

To validate the efficacy of our modules, we conduct comprehensive evaluations on several variants of SceneGen, as-

#### References

- [1] Titas Anciukeviˇcius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J Mitra, and Paul Guerrero. Renderdiffusion: Image diffusion for 3d reconstruction, inpainting and generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [2] PJ Besl and Neil D McKay. A method for registration of 3-d shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1992. 6, 14
- [3] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Easi3r: Estimating disentangled motion from dust3r without training. In Proceedings of the International Conference on Computer Vision, 2025. 2
- [4] Yen-Chi Cheng, Hsin-Ying Lee, Sergey Tulyakov, Alexander G Schwing, and Liang-Yan Gui. Sdfusion: Multimodal 3d shape completion, reconstruction, and generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [5] Tao Chu, Pan Zhang, Qiong Liu, and Jiaqi Wang. Buol: A bottom-up framework with occupancy-aware lifting for panoptic 3d scene reconstruction from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2, 3
- [6] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, et al. Abo: Dataset and benchmarks for real-world 3d object understanding. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022. 2
- [7] Timoth´ee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In Proceedings of the International Conference on Learning Representations, 2024. 4
- [8] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Conference on Neural Information Processing Systems, 2023. 2
- [9] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [10] Andreea Dogaru, Mert Ozer,¨ and Bernhard Egger. Gen3dsr: Generalizable 3d scene reconstruction via divide and conquer from a single view. In International Conference on 3D Vision, 2025. 3, 6, 16
- [11] Wenqi Dong, Bangbang Yang, Zesong Yang, Yuan Li, Tao Hu, Hujun Bao, Yuewen Ma, and Zhaopeng Cui. Hiscene: creating hierarchical 3d scenes with isometric view generation. In ACM Multimedia, 2025. 2, 3
- [12] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and generation with large language models. In Conference on Neural Information Processing Systems, 2023. 2, 3

- [13] Huan Fu, Bowen Cai, Lin Gao, Ling-Xiao Zhang, Jiaming Wang, Cao Li, Qixun Zeng, Chengyue Sun, Rongfei Jia, Binqiang Zhao, et al. 3d-front: 3d furnished rooms with layouts and semantics. In Proceedings of the International Conference on Computer Vision, 2021. 2, 7
- [14] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision, 2021. 2, 5, 6, 7, 13, 16
- [15] Daoyi Gao, D´avid Rozenberszki, Stefan Leutenegger, and Angela Dai. Diffcad: Weakly-supervised probabilistic cad model retrieval and alignment from an rgb image. ACM Transactions On Graphics, 2024. 2, 3
- [16] Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, and Bernhard Sch¨olkopf. Graphdreamer: Compositional 3d scene synthesis from scene graphs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2, 3
- [17] Wei Gao and Russ Tedrake. Filterreg: Robust and efficient probabilistic point-set registration using gaussian filter and twist parameterization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019. 6, 15
- [18] Zeqi Gu, Yin Cui, Zhaoshuo Li, Fangyin Wei, Yunhao Ge, Jinwei Gu, Ming-Yu Liu, Abe Davis, and Yifan Ding. Artiscene: Language-driven artistic 3d scene generation through image intermediary. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,

2025. 2, 3

- [19] Can G¨umeli, Angela Dai, and Matthias Nießner. Roca: Robust cad model retrieval and alignment from a single image. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022. 2, 3
- [20] Xianglong He, Junyi Chen, Sida Peng, Di Huang, Yangguang Li, Xiaoshui Huang, Chun Yuan, Wanli Ouyang, and Tong He. Gvgen: Text-to-3d generation with volumetric representation. In Proceedings of the European Conference on Computer Vision, 2024. 2
- [21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Conference on Neural Information Processing Systems, 2017. 6
- [22] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Conference on Neural Information Processing Systems, 2020. 2
- [23] Zehuan Huang, Yuan-Chen Guo, Xingqiao An, Yunhan Yang, Yangguang Li, Zi-Xin Zou, Ding Liang, Xihui Liu, Yan-Pei Cao, and Lu Sheng. Midi: Multi-instance diffusion for single image to 3d scene generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 6, 14, 16
- [24] Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. In Proceedings of the International Conference on Computer Vision, 2025. 2, 6

- [25] Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. Neural wavelet-domain diffusion for 3d shape generation. In ACM SIGGRAPH Asia Conference, 2022. 2
- [26] Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, et al. Hunyuan3d 2.1: From images to highfidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025. 2
- [27] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. In Proceedings of the International Conference on Computer Vision, 2025. 2
- [28] Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2
- [29] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations, 2014. 2
- [30] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In Proceedings of the European Conference on Computer Vision, 2024. 2
- [31] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In Proceedings of the International Conference on Learning Representations, 2024. 2
- [32] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusionsdf: Text-to-shape via voxelized diffusion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [33] Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025. 2
- [34] Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. Partcrafter: Structured 3d mesh generation via compositional latent diffusion transformers. In Conference on Neural Information Processing Systems, 2025. 2, 3, 6, 16
- [35] Lu Ling, Chen-Hsuan Lin, Tsung-Yi Lin, Yifan Ding, Yu Zeng, Yichen Sheng, Yunhao Ge, Ming-Yu Liu, Aniket Bera, and Zhaoshuo Li. Scenethesis: A language and vision agentic framework for 3d scene generation. arXiv preprint arXiv:2505.02836, 2025. 2, 3
- [36] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In Proceedings of the International Conference on Learning Representations, 2023. 4, 5, 13
- [37] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm - open-ended visual storytelling via latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2

- [38] Xinhang Liu, Yu-Wing Tai, and Chi-Keung Tang. Agentic 3d scene generation with spatially contextualized vlms. arXiv preprint arXiv:2505.20129, 2025. 2, 3
- [39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In Proceedings of the International Conference on Learning Representations, 2019. 6
- [40] Shitong Luo and Wei Hu. Diffusion probabilistic models for 3d point cloud generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021. 2
- [41] Norman M¨uller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. Diffrf: Rendering-guided 3d radiance field diffusion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 2
- [42] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 2, 3, 6, 13
- [43] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the International Conference on Computer Vision, 2023. 2, 3
- [44] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021. 6
- [45] Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021. 13
- [46] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, et al. Sam 2: Segment anything in images and videos. In Proceedings of the International Conference on Learning Representations, 2025. 7, 17
- [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2022. 2
- [48] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016. 2
- [49] Fan-Yun Sun, Weiyu Liu, Siyi Gu, Dylan Lim, Goutam Bhat, Federico Tombari, Manling Li, Nick Haber, and Jiajun Wu. Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2, 3
- [50] Xiang Tang, Ruotong Li, and Xiaopeng Fan. Towards geometric and textural consistency 3d scene generation via single image-guided model generation and layout optimization. arXiv preprint arXiv:2507.14841, 2025. 3
- [51] Aether Team, Haoyi Zhu, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Chunhua

- Shen, Jiangmiao Pang, et al. Aether: Geometric-aware unified world modeling. In Proceedings of the International Conference on Computer Vision, 2025. 2
- [52] Jianyuan Wang, Nikita Karaev, Christian Rupprecht, and David Novotny. Vggsfm: Visual geometry grounded deep structure from motion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2
- [53] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 4, 13, 14
- [54] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025.
- [55] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2
- [56] Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. In Winter Conference on Applications of Computer Vision, 2025. 2
- [57] Haoning Wu, Ziheng Zhao, Ya Zhang, Weidi Xie, and Yanfeng Wang. Mrgen: Diffusion-based controllable data engine for mri segmentation towards unannotated modalities. In Proceedings of the International Conference on Computer Vision, 2025. 2
- [58] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. In Conference on Neural Information Processing Systems,

2024. 2

- [59] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2, 3, 4, 5, 13
- [60] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan Shi, Kalyan Sunkavalli, et al. Dmv3d: Denoising multi-view diffusion using 3d large reconstruction model. In Proceedings of the International Conference on Learning Representations, 2024. 2
- [61] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2024. 2, 3
- [62] Kaixin Yao, Longwen Zhang, Xinhao Yan, Yan Zeng, Qixuan Zhang, Wei Yang, Lan Xu, Jiayuan Gu, and Jingyi Yu. Cast: Component-aligned 3d scene reconstruction from an rgb image. In ACM SIGGRAPH Conference, 2025. 2, 3
- [63] Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A high-fidelity dataset of 3d

- indoor scenes. In Proceedings of the International Conference on Computer Vision, 2023. 7, 14
- [64] Huangyue Yu, Baoxiong Jia, Yixin Chen, Yandan Yang, Puhao Li, Rongpeng Su, Jiaxin Li, Qing Li, Wei Liang, Song-Chun Zhu, et al. Metascenes: Towards automated replica creation for real-world 3d scans. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2, 3
- [65] Guangyao Zhai, Evin Pınar Ornek,¨ Shun-Cheng Wu, Yan Di, Federico Tombari, Nassir Navab, and Benjamin Busam. Commonscenes: Generating commonsense 3d indoor scenes with scene graph diffusion. In Conference on Neural Information Processing Systems, 2023. 2, 3
- [66] Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics, 2023. 2
- [67] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang, Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo. Gaussiancube: A structured and explicit radiance representation for 3d generative modeling. In Conference on Neural Information Processing Systems, 2024. 2
- [68] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In Proceedings of the International Conference on Learning Representations, 2025. 2
- [69] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions On Graphics,

2024. 2

- [70] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. 6
- [71] Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou, Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2025. 2
- [72] Qingcheng Zhao, Xiang Zhang, Haiyang Xu, Zeyuan Chen, Jianwen Xie, Yuan Gao, and Zhuowen Tu. Depr: Depth guided single-view scene reconstruction with instance-level diffusion. In Proceedings of the International Conference on Computer Vision, 2025. 6, 16

### SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass Appendix

#### Contents

- 1. Introduction 2
- 2. Related Work 2
- 3. Method 3

- 3.1. Problem Formulation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.2. SceneGen . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.3. Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.4. Extension to Multi-view Inputs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 4. Experiments 6

- 4.1. Experimental Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.2. Quantitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.3. Qualitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.4. Ablation Studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5. Conclusion 8

- A. Preliminaries on 3D Foundation Models 13
- B. More Details about Training Data 13
- C. More Implementation Details 13

- C.1. Extension to Multi-image Inputs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- C.2. Evaluation Protocols . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- D. More Visualizations 16
- E. Limitations & Future Works 17

- E.1. Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- E.2. Future Works . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

#### A. Preliminaries on 3D Foundation Models

Given the inherent challenges of directly generating a 3D scene with multiple 3D assets from a single image, SceneGen aims to fully leverage the visual and geometric priors embedded in state-of-the-art 3D foundation models. Therefore, we build our model based on TRELLIS [59], and adopt DINOv2 [42] and VGGT [53] as our visual and geometric encoders, respectively. In the following, we provide a detailed introduction to TRELLIS and VGGT to better illustrate their roles.

TRELLIS. For a 3D asset (O), TRELLIS encodes its geometry and appearance into a unified representation (z), denoted as: z = {(zi,pi)}Li=1. Here, pi ∈ {0,1,...,D − 1}3 denotes the positional index of an active voxel in the 3D grid intersecting the surface of O, and zi ∈ RC represents the local latent feature attached to the corresponding voxel, with D and L representing the 3D grid resolution and the total number of active voxels, respectively.

The generation process adopts two cascaded rectified flow models: the sparse structure generator (GS) synthesizes the sparse voxel structure {pi}Li=1, encoding geometric priors by predicting its low-resolution feature gird (S); while the structured latents generator (GL) generates texture and appearance features {zi}Li=1 conditioned on {pi}. Both models are optimized via the conditional flow matching (CFM) [36] objective, which establishes straight probability paths between distributions through linear interpolation: x(t) = (1−t)x0+tϵ, where x0 denotes data samples, ϵ ∼ N(0,I), and t ∈ [0,1]. The velocity field (v(x,t) = ∇tx) governs the reverse process, with the CFM objective formulated as:

0,ϵ∥vθ(x,t) − (ϵ − x0)∥22

Lcfm(θ) = Et,x

Notably, the sparse structured generator (GS) learns rich geometric priors from large-scale 3D data, effectively capturing both object geometries and spatial relationships, thus delivering essential asset-level understanding capabilities. Our SceneGen is compatible with both the sparse structured generator and structured latents generator, thus can sequentially employ them to decode synthesized latent features into the geometry and texture of 3D assets.

VGGT. Trained on large-scale 3D annotated data, VGGT can extract 3D scene features through a purely feedforward network without explicit 3D inductive biases. For single or multi-view RGB inputs ({Ii}si=1), its aggregator derives scene geometric features ({Figeo}si=1), represented as:

{Figeo}si=1 = {[FGgeo,FIgeo]}si=1 = VGGT({Ii}si=1)

Here, FGgeo and FIgeo denote features extracted by global self-attention and local self-attention layers, respectively. These features are efficiently decoded by lightweight DPT layers [45] into depth maps, point maps, and tracks, validating their rich

scene geometric representation capacity.

By integrating these complementary strengths, our SceneGen effectively captures both local asset-level and global scenelevel features from the input image, achieving robust performance on the challenging 3D scene generation task.

#### B. More Details about Training Data

We train SceneGen on the 3D-FUTURE [14] dataset, leveraging its rich textures and diverse lighting conditions to simulate real-world environments and thereby enhance the model’s generalization ability. Additionally, to ensure the model robustly learns the relative spatial relationships among multiple assets, we further scale up training data through data augmentation. Specifically, for a scene with N objects, we iteratively select each asset as the query asset and randomly shuffle the remaining ones during training. Considering GPU memory constraints, we set the maximum number of assets per scene to N′ = 7 on a single A100 GPU. For samples containing more than N′ assets, we randomly select a subset of N′ assets for training. Furthermore, following TRELLIS [59], we apply its aesthetic score filtering criterion to exclude assets with aesthetic scores below 4.5, thereby ensuring high data quality. The distribution of asset counts across training scenes is illustrated in Figure 6.

#### C. More Implementation Details

This section provides a holistic explanation of implementation details discussed in the paper. Concretely, Sec. C.1 describes the specific strategies applied to extend SceneGen to multi-image inputs; and Sec. C.2 elaborates on our evaluation protocols.

##### C.1. Extension to Multi-image Inputs

While SceneGen is primarily designed for 3D scene generation based on a single scene image and trained exclusively on single-view images, it can be seamlessly adapted to multi-view inputs during inference with no need for additional training or fine-tuning. Concretely, during inference, our model can take images of the same scene from multiple viewpoints, along

[Figure 213]

Figure 6. Distribution of Asset Counts in our Training Data and Original 3D-FUTURE.

with their corresponding objects and instance masks, as input. Within our SceneGen, the geometric encoder (ΦG) (an offthe-shelf VGGT [53] aggregator) integrates geometric information across different viewpoints to produce better geometric representations for each perspective, thereby enabling SceneGen to synthesize more accurate geometric structures. Finally, we predict the relative positions among different assets from each viewpoint and use the mean of these predictions across all views as the final spatial position output. It is important to note that, to ensure correctness throughout the inference process, the input order of assets and their segmentation masks must remain consistent across all viewpoints.

Given the current lack of training and quantitative evaluation data for multi-view 3D scene generation, this work presents qualitative results on scenes sampled from ScanNet++ [63] to demonstrate the scalability of SceneGen, and leaves the construction of suitable multi-view datasets and evaluation methods for future work.

Method Alignment CD-S↓ CD-S 1↓ CD-S 2↓ F-Score-S↑ IoU-B↑

ICP 0.1697 0.0653 0.1044 41.64 0.1232 FilterReg 0.0501 0.0278 0.0223 68.74 0.2493

MIDI [23]

ICP 0.0310 0.0121 0.0189 83.74 0.5103 FilterReg 0.0118 0.0052 0.0066 90.60 0.5818

SceneGen (Ours)

Table 3. Geometric Metrics Comparisons on Different Point Cloud Alignment Methods.

##### C.2. Evaluation Protocols

Geometric metrics. Following previous work [23], we conduct geometry evaluation in normalized 3D space (also referred to as canonical space, i.e., x,y,z ∈ [−1,1]), where the ground truth and the synthesized query asset are first rigidly aligned using point cloud registration algorithms. Unlike MIDI [23], which relies on the traditional Iterative Closest Point (ICP [2]) method

###### Scene Image Instance-masked Image GT Render Pred Render (Ours)

|[Figure 214]|
|---|

|[Figure 215]|
|---|

|[Figure 216]|
|---|

|[Figure 217]|
|---|

|[Figure 218]|
|---|

|[Figure 219]|
|---|

|[Figure 220]|
|---|

|[Figure 221]|
|---|

|[Figure 222]|
|---|

|[Figure 223]|
|---|

|[Figure 224]|
|---|

|[Figure 225]|
|---|

|[Figure 226]|
|---|

|[Figure 227]|
|---|

|[Figure 228]|
|---|

|[Figure 229]|
|---|

Figure 7. Examples of Visual Metrics Evaluation Protocols. Here, we present two complementary types of ground truth: instancemasked images may introduce slight differences due to potential occlusions, while GT-render images lack scene-level illumination.

prone to suboptimal alignment results, we employ FilterReg [17], a faster and more robust point cloud alignment approach. As presented in Table 3, both MIDI and SceneGen achieve better overall performance when aligned via FilterReg, demonstrating the reliability of this alignment method compared to traditional ICP. Moreover, under both alignment strategies, SceneGen consistently outperforms MIDI, indicating that explicitly predicting the spatial positions among assets enables SceneGen to more accurately model the relationships among distinct 3D assets within the scene.

Visual metrics. Beyond the commonly used geometric evaluations described above, we also consider several visual metrics

Scene Image GT Assets PartCrafter DepR Gen3DSR MIDI SceneGen (Ours)

|[Figure 230]<br><br>[Figure 231]|
|---|

|[Figure 232]|
|---|

|[Figure 233]|
|---|

|[Figure 234]|
|---|

|[Figure 235]|
|---|

|[Figure 236]|
|---|

|[Figure 237]|
|---|

|[Figure 238]<br><br>[Figure 239]|
|---|

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

|[Figure 243]|
|---|

|[Figure 244]|
|---|

|[Figure 245]|
|---|

|[Figure 246]<br><br>[Figure 247]|
|---|

|[Figure 248]|
|---|

|[Figure 249]|
|---|

|[Figure 250]|
|---|

|[Figure 251]|
|---|

|[Figure 252]|
|---|

|[Figure 253]|
|---|

|[Figure 254]<br><br>[Figure 255]|
|---|

|[Figure 256]|
|---|

|[Figure 257]|
|---|

|[Figure 258]|
|---|

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

[Figure 262]

|[Figure 263]<br><br>[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

|[Figure 267]|
|---|

| |
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

Figure 8. More Qualitative Comparisons on the 3D FUTURE Test Set.

to assess the visual quality of generated scenes. Concretely, after aligning the synthesized point clouds with the ground truth scenes, we use Blender to render them with the identical camera parameters. The rendered images are then compared with two types of ground truth to compute perceptual metrics that reflect the visual quality of synthesized scenes. As illustrated in Figure 7, these include: (i) instance-masked scene images, which are extracted using the corresponding object masks, where the occlusion relationships between assets introduce differences relative to predicted renderings; and (ii) GT-Render images, which are rendered from the ground truth assets at the same viewpoint using Blender, but lack scene-level illumination and complete textures, resulting in textural discrepancies compared to predicted scenes. Thus, by computing visual metrics against both types of ground truth, we provide a complementary evaluation of the visual quality of synthesized scenes.

Efficiency. To ensure a fair comparison across all methods, we report the average inference time over 500 trials of synthesizing scenes with a single asset on a single A100 GPU. Notably, our proposed SceneGen can directly generate 3D scenes containing 4 assets in a single feedforward pass within 2 minutes on the same hardware, eliminating the need for timeconsuming sequential generation of individual 3D assets.

#### D. More Visualizations

This section presents additional qualitative results on the 3D-FUTURE [14] test set, offering a detailed comparison between our SceneGen and representative baselines. As depicted in Figure 8, we have the following observations: (i) PartCrafter [34] frequently suffers from missing or mixed-up assets due to its inability to control generation via object masks, despite already taking segmented objects and asset counts as input; (ii) Both PartCrafter and DepR [72] can only generate scene geometry without rendering texture details; and (iii) All baseline methods (PartCrafter [34], DepR [72], Gen3DSR [10], and MIDI [23])

share the common limitation of incorrect spatial relationships among synthesized assets. In contrast, our SceneGen fully integrates visual and geometric features within the scene to enable mutual influence among multiple assets during generation, producing 3D scenes with physically plausible geometry and high-quality texture details.

#### E. Limitations & Future Works

##### E.1. Limitations

While our SceneGen demonstrates superior performance in 3D scene generation, it is not without its limitations.

Limited to Indoor Generation. While SceneGen demonstrates better texture generation and generalization capabilities compared to previous methods that rely on canonical representations, the narrow training data distribution limits its ability to generalize to non-indoor scenes, restricting its generalization to a broader range of environments.

Asset Collisions and Overlaps. Although SceneGen can generate multiple 3D assets and relative spatial positions in a single feedforward pass, without relying on complex post-processing, it does not always handle contact relationships among objects, occasionally leading to asset overlaps or geometric inconsistencies. This is mainly because our single-stage framework does not explicitly enforce strict spatial or physical constraints among objects.

Reliance on Segmentation Masks. SceneGen inherently requires segmentation masks of the target objects as input. In our current framework, we leverage either ground truth masks or masks pre-extracted by an off-the-shelf SAM 2 [46]. This reliance limits the flexibility of applying SceneGen directly to in-the-wild data to some extent and may potentially result in a lack of robustness against low-quality segmentation masks.

##### E.2. Future Works

To address the aforementioned limitations of SceneGen, we propose several directions for future improvement: (i) Constructing larger-scale 3D scene generation datasets that cover more diverse indoor and outdoor scenarios, to address biases in training data distribution and improve the generalization ability of models; (ii) Building suitable multi-view scene generation datasets to expand the application scope and practical potential of existing models; (iii) Incorporating explicit physical priors or constraints to facilitate the model to better learn complex interactions among objects; and (iv) Introducing an additional object segmentation module into the current framework or natively integrating the capability to segment assets from scenes.

