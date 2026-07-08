# arXiv:2411.05738v2[cs.CV]5Mar2025

## StdGEN: Semantic-Decomposed 3D Character Generation from Single Images

Yuze He1,2† Yanning Zhou1* Wang Zhao2 Zhongkai Wu1 Kaiwen Xiao1 Wei Yang1 Yong-Jin Liu2* Xiao Han1

1 Tencent AI Lab 2 Tsinghua University

[Figure 1]

Figure 1. Our StdGEN generates high-quality, decomposed 3D characters from a single reference image.

### Abstract

We present StdGEN, an innovative pipeline for generating semantically decomposed high-quality 3D characters from single images, enabling broad applications in virtual reality, gaming, and filmmaking, etc. Unlike previous methods which struggle with limited decomposability, unsatisfactory quality, and long optimization times, StdGEN features decomposability, effectiveness and efficiency; i.e., it generates intricately detailed 3D characters with separated semantic components such as the body, clothes, and hair, in three minutes. At the core of StdGEN is our proposed Semantic-aware Large Reconstruction Model (S-LRM), a transformer-based generalizable model that jointly reconstructs geometry, color and semantics from multi-view images in a feed-forward manner. A differentiable multilayer semantic surface extraction scheme is introduced to acquire meshes from hybrid implicit fields reconstructed by our S-LRM. Additionally, a specialized efficient multi-

*Corresponding Authors: amandayzhou@tencent.com, liuyongjin@tsinghua.edu.cn

†Work done during an internship at Tencent AI Lab.

view diffusion model and an iterative multi-layer surface refinement module are integrated into the pipeline to facilitate high-quality, decomposable 3D character generation. Extensive experiments demonstrate our state-of-theart performance in 3D anime character generation, surpassing existing baselines by a significant margin in geometry, texture and decomposability. StdGEN offers ready-touse semantic-decomposed 3D characters and enables flexible customization for a wide range of applications. Project page: https://stdgen.github.io

### 1. Introduction

Generating high-quality 3D characters from single images has widespread applications in virtual reality, video games, filmmaking, etc. Beyond automatically creating a complete 3D character, there is an increasing demand for the ability to produce decomposable characters, where distinct semantic components like the body, clothes, and hair are disentangled. This decomposition allows for much easier editing, control, and animation of characters, greatly enhancing their usability across various downstream applications.

However, creating such decomposable characters from single images is challenging, as each component may face issues such as occlusion, ambiguity, and inconsistencies in their interactions with other components. Existing methods for decomposable avatar generation primarily focus on realistic clothed human models, exploring disentangled 3D parametric [56], explicit [37, 38], or implicit [11, 14, 18, 58] representations alongside various optimization techniques. These optimization approaches often employ score distillation loss [40] to leverage 2D generative priors, which leads to prolonged optimization times and the generation of coarse, high-contrast textures. Additionally, the dependence on parametric human models, such as SMPL-X [32], is inadequate for virtual characters, which often exhibit exaggerated body proportions and complex clothing designs.

To address these limitations, CharacterGen [39] was developed to efficiently generate characters from single images using a multi-view diffusion model and large reconstruction model [15]. Despite showing impressive generation capabilities in various posed images, CharacterGen can only produce holistic avatars in watertight meshes with no decomposability. These meshes require significant manual labor to separate, edit, or animate, limiting their applicability. Moreover, the quality of the generated meshes is often unsatisfactory, particularly in finer details such as the character’s face and clothing, as shown in Fig. 5. Therefore, efficiently generating high-quality, decomposable 3D characters remains an open challenge.

In this work, we propose StdGEN, an efficient and effective pipeline to generate decomposable, high-quality, Apose 3D characters from single images of any pose. At its core is a novel Semantic-aware Large Reconstruction Model (S-LRM), which innovatively introduces semantic attributes to the original Large Reconstruction Model [15], enabling efficient reconstruction of unified geometry, color and semantic fields. Moreover, a differentiable multi-layer semantic surface extraction scheme is proposed to extract decomposed semantic 3D surfaces from reconstructed implicit fields, empowering joint end-to-end training with explicit meshes. Together, S-LRM can efficiently generate semantic-decomposed, complete and consistent 3D surfaces from multi-view input images in a feed-forward manner.

Built upon S-LRM, StdGEN comprises three key stages: multi-view diffusion, feed-forward reconstruction, and mesh refinement, each designed with technical innovations to enable effective decomposition. Specifically, given a single reference image with an arbitrary pose, a specialized efficient multi-view diffusion model first generates multiple A-pose RGB images and normal maps at different camera views. Next, the generated images are processed by our novel S-LRM, to reconstruct geometry, color, and semantics in a feed-forward manner. Once the coarse decomposed surface generation is obtained, we further refine the mesh

quality through a proposed iterative multi-layer mesh refinement method, using diffusion-generated 2D images and normal maps as guidance. As a result, StdGEN can generate high-quality decomposable 3D characters from a single reference image within minutes.

To facilitate the semantic decomposed training and testing, we further developed Anime3D++ dataset based on [39] , focusing on anime characters due to the wide internet presence, with finely annotated multi-view multi-pose semantic parts. Our experiments demonstrate that StdGEN surpasses all existing baselines, achieving state-of-the-art performance in both arbitrary and A-pose 3D character generation. Moreover, StdGEN’s decomposable generation capability enables flexible customization, which can benefit numerous downstream applications. In summary, this work makes the following contributions:

- • We introduce a novel Semantic-aware Large Reconstruction Model (S-LRM), which jointly models geometry, color, and semantic information with efficient tri-plane feed-forward inference. An effective differentiable surface extraction scheme is proposed to facilitate the explicit multi-layer semantic surface reconstruction.
- • Building upon (1) our S-LRM, (2) an efficient multi-view diffusion model and (3) iterative multi-layer mesh refinement, we present StdGEN, an efficient pipeline for highquality decomposed 3D character generation, which for the first time enables semantic-decomposed 3D character creation from arbitrary-posed single images in minutes.
- • Our method demonstrates superior performance over existing baselines, with its decomposed modeling unlocking potential for various downstream applications.

### 2. Related Works

#### 2.1. 3D Generation

To circumvent the need for extensive 3D assets during training, several approaches suggest lifting powerful 2D pre-trained diffusion models [10, 35, 45, 46] for 3D generation. The earliest works [40, 55] incorporate a pretrained 2D diffusion model for probability density distillation using Score Distillation Sampling (SDS). These approaches gradually optimize a randomly initialized radiance field [1, 5, 52] with volume rendering, making it timeconsuming to generate an object. Later research continues to enhance the aesthetics and accuracy of 3D content generation [7, 28, 48, 54, 60] and further investigate different application scenarios [12, 44, 47, 51]. However, relying solely on 2D priors for 3D generation often leads to poor geometry representation, e.g., multi-faced Janus problem, due to the challenges in controlling precise viewpoints through text prompts. The large-scale 3D datasets, e.g. Objaverse [9], unlock the possibility of imposing 3D priors to the model. Several works utilize view-consistent im-

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

|Semantic-EquivalentNeRF/SDF|
|---|

|TriplaneDecoder|
|---|

|Multi-layerRefinement|
|---|

|Color| |
|---|---|
| | |

|ViTEncoder|
|---|

[Figure 12]

|CanonicalizationDiffusion|
|---|

|Multi-ViewDiffusion|
|---|

Semantic-EquivalentNeRF/SDF

|Density| |
|---|---|
| | |

CanonicalizationDiffusion

Multi-layerRefinement

Multi-ViewDiffusion

TriplaneDecoder

Semantics

ViTEncoder

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

Decode

Triplane

Multi-view RGBs and Normals

Semantic-aware Large Reconstruction Model (S-LRM)

Coarse Decomposed Results

Refined Decomposed Results

Input Image

Figure 2. The overview of our StdGEN pipeline. Starting from a single reference image, our method utilizes diffusion models to generate multi-view RGB and normal maps, followed by S-LRM to obtain the color/density and semantic field for 3D reconstruction. Semantic decomposition and part-wise refinement are then applied to produce the final result.

line of work leverages 3D-aware GANs to model the distribution of digital humans [2, 14, 21, 36, 71]. Recently the SDS-based methods have shown the possibility of generating a variety of stylized characters [3, 11, 19, 22, 56], yet it suffers from the long optimization times and the difficulty of meticulous style control. Frankenstein [65] concentrates on producing decomposed, textureless 3D meshes based on 2D layouts, restricting the potential for achieving highfidelity reconstruction from the reference image. CharacterGen [39] calibrates input poses to canonical multiview images via an image-conditioned multi-view diffusion model, followed by LRM for 3D character reconstruction and multi-view texture back projection, but still exhibits limited geometry and texture quality. Our approach, in contrast, employs a semantic-aware, feed-forward paradigm that generates high-quality, decomposable characters using only one forward pass from an arbitrary reference image, providing significant efficiency and quality improvement.

ages to fine-tune the diffusion model. Zero-1-to-3 [29] integrates 3D priors into 2D stable diffusion by fine-tuning the pre-trained model for novel view synthesis (NVS). To further enhance the multi-view consistency, several recent works [20, 30, 31, 50] propose synchronously generating multi-view images in a single generation process and achieving constraints in 3D place through feature interaction in attention mechanism. Besides, the 3D native generation method shows powerful geometric generation ability [27, 33, 73]. However, the ability of these methods to follow instructions is typically moderate; therefore, they face challenges in achieving the desired outcomes in scenarios requiring precise restoration of reference images, e.g., 3D character generation.

#### 2.2. Large Reconstruction Model

Large Reconstruction Model (LRM) [15] leverages the transformer-based model to map the single image feature to implicit tri-plane representation. Instant3D [25] extends LRM by feeding multi-view images instead of a single image. LGM [53], GRM [68] and GS-LRM [72] replace the 3D representation to 3D Gaussians, embracing its efficiency in rendering and low memory consumption. InstantMesh [63] and CRM [61] explicitly model the geometry by equipping the generative pipeline with FlexiCubes [49], achieving high-quality surface extraction and high rendering speed. The following works further explored applying advanced model architecture [70] or 3D presentation [6, 8], aiming to improve the efficiency, realism, and generalization of reconstruction. Integrating with multi-view diffusion models, these LRMs can achieve textto-3D generation or single image-to-3D generation. Yet all these methods typically produce holistic models. In contrast, our method generates semantically decomposed characters, making downstream processing such as editing and animation much more efficient.

### 3. Method

The proposed StdGEN begins with multi-view canonical character generation (Sec. 3.1) from a reference image. To reconstruct a decomposable 3D character from multi-view images, we extend the LRM with a semantic field, enabling semantic-based layered generation (Sec. 3.2). Finally, a multi-level refinement process is designed to enhance the results, improving the geometric structure and providing more detailed textures (Sec. 3.3). An overview of the StdGEN pipeline is shown in Fig. 2.

#### 3.1. Multi-view Generation and Canonicalization

Given a reference character image, our objective is to generate a set of multi-view images and corresponding normal maps that maintain 3D consistency. This process involves two steps: Canonicalizing an arbitrary reference image into an A-pose character and generating multi-view RGBs and normals from the A-pose representation.

#### 2.3. 3D Character Generation

Arbitrary Reference to A-pose Conversion. Directly reconstructing a 3D character model in arbitrary poses can be affected by self-occlusion from different viewpoints. There-

3D character generation is a challenging problem due to its high precision requirements and the scarcity of data. One

Image Tokens Triplane Decoder

Triplane Tokens

fore, we design to first canonicalize the character in 2D space. We select the A-pose as the target representation, as it is more conducive to subsequent multi-view generation, reconstruction, and applications like rigging. Following previous works [17, 39], we employ the Stable Diffusion [45] model, augmented with a ReferenceNet, to translate a 2D arbitrary posed reference image to A-pose image. The ReferenceNet helps to preserve the reference identity in the resulting image. We refer to [39] for more details.

[Figure 20]

|ViTEncoder<br><br>[Figure 21]|
|---|

|ReferenceImage|
|---|

|Modulation|
|---|

|Self-Attn|
|---|
|LoRA|

|MLP|
|---|

|Modulation|
|---|

|Cross-Attn|
|---|
|LoRA|

|Modulation|
|---|

ReferenceImage

Cross-Attn

ViTEncoder

Self-Attn

Modulation

Modulation

Modulation

MLP

LoRA

LoRA

Triplane

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]<br><br>[Figure 29]| |
|---|---|

|SDF / Color Decoder<br><br>[Figure 30]|
|---|

Decomposition

[Figure 31]

[Figure 32]

|Semantic Decoder<br><br>[Figure 33]|
|---|

Decomposed Results Results with Semantics

Multi-view RGBs and Normals Generation. This stage provides comprehensive information for the subsequent S-LRM, with multi-view consistent normals crucial for capturing rich surface details in mesh refinement rather than relying on independent normal predictions. We adapt Era3D [26] to generate high-resolution RGBs and normals simultaneously. Using memory-efficient rowwise attention across views and between RGB and normal maps, we ensure fine-grained spatial consistency and enable higher output resolutions. Specifically, it takes a single A-pose image as input, producing orthographic projections of six viewpoints (elevation 0◦, azimuth −90◦,−45◦,0◦,45◦,90◦,180◦). For sharper character details, we increased the resolution through a progressive training approach, ultimately reaching 1024.

Figure 3. Demonstration of the structure and intermediate outputs of our semantic-aware large reconstruction model (S-LRM).

decoded SDF grid, and use direct mesh rasterization to render images. However, to obtain semantic-decomposed surface reconstruction, both NeRF and SDF implicit representations must be capable of rendering distinct semantic layers into images or extracting separate semantic surfaces using FlexiCubes in a differentiable manner. To achieve that, a novel semantic-equivalent NeRF/ SDF is proposed to extract character parts by specific semantics.

Semantic-equivalent NeRF and SDF. NeRF represents a 3D scene by spatial-variant volume densities with colors*. We extend it with a semantic field, and model them as a learnable function FΘ that takes sampled point location x = (x;y;z) as inputs, and outputs color c, density σ and semantic distribution s as: (σ,c,s) = FΘ(x).

Compared with CharacterGen [39], our choice can simultaneously generate high-resolution, multi-view consistent normal maps for mesh refinement. Besides, the twostep design allows for improved editing in the 2D A-pose space, facilitating the generation of decomposed characters for enhanced 3D editing applications.

To render per-pixel color Cˆ(r), a series of 3D points are sampled along the ray r, and the pixel color is computed by integrating the sampled densities σi and colors ci using the volume rendering equation with:

#### 3.2. Semantic-aware Large Reconstruction Model

- i−1
- j=1

N

Cˆ(r) =

(1 − αj) (1)

Tiαici, Ti =

Once obtaining multi-view images, [39, 63] use transformer based sparse-view Large Reconstruction Model (LRM) to reconstruct a holistic 3D mesh. In contrast, we aim to generate characters with decomposed components, including the minimal-clothed human model, external clothing, and hair, to produce 3D character models that are not only visually accurate but also functionally versatile for various uses in 3D games and animation pipelines. To achieve this goal, we proposed the Semantic-aware Large Reconstruction Model (S-LRM), which extends NeRF/SDF to simultaneously encode semantics, appearance and geometry information in a feed-forward manner.

i=1

where αi = (1 − exp(−σiδi)), δi = ti+1 − ti is the alpha value of samples and distance between adjacent samples.

Given the probability ps,i of semantic s at location i, the pixel color Cˆs(r) under semantic s can be calculated as:

- i−1
- j=1

N

Cˆs(r) =

(1 − αjps,j) (2)

Ts,ips,iαici, Ts,i =

i=1

If the probability of a certain semantic at a given location is zero, it should be considered fully transparent under the current semantic category. Furthermore, given that a position is known to be opaque, the probability of the current semantic should be linear to the final equivalent transparency.

As shown in Fig. 2, the S-LRM takes multi-view images into tokens and then feeds into a transformer-based imageto-triplane decoder to output a triplane representation. The triplane features are decoded to obtain semantic, color, density/SDF information. To enhance the training efficiency and reconstruction quality, following InstantMesh [63], we first utilize triplane NeRF representation to render 2D images with volume rendering for training. We then switch to FlexiCubes [49] to extract explicit mesh from triplane-

Unlike NeRF, SDF does not incorporate the concept of transparency. Instead, positive/negative values represent points outside/inside the surface. Consequently, semantic probabilities cannot be directly applied to SDF for the mesh

*We ignore the view-dependent effects to simplify the discussion.

Selected RED as current NeRF semantic

No Semantic Selected

Selected GREEN as current NeRF semantic

under occlusion; 3D supervision would be effective but often too resource-intensive. To address this, we propose an effective supervision that jointly learns semantics and colors, enabling the acquisition of a 3D semantic field and internal character information using only 2D supervision.

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

Red

[Figure 59]

[Figure 60]

[Figure 61]

Red

>0

>0

>0

| |
|---|
|[Figure 62]|

|[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>①<br><br>②|
|---|
|[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>④<br><br>③|

|[Figure 73]<br><br>[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]<br><br>[Figure 77]<br><br>①<br><br>②|
|---|
|[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>④<br><br>③|

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

- Stage 1: Training on NeRF with Single-layer Semantics. In this initial stage, we train on the triplane NeRF representation. We initialize the model with the pre-trained InstantNeRF, training the newly added LoRA in all attention blocks’ linear layers and the newly introduced semantic decoder. We train it under the image, mask and semantic loss:

Sˆ(r) =

N

i=1

Tipiαi, Lsem =

k

CE(Sˆk,Skgt) (4) L1 = Lmse + λlpipsLlpips + λmaskLmask + λsemLsem (5)

Sˆ is the semantic map calculated by the probabilities pi from semantic decoder’s output through a softmax layer.

Sˆk,Skgt denotes the k-th view of rendered and ground-truth semantic maps, and CE denotes the cross-entropy function.

- Stage 2: Training on NeRF with Multi-layer Semantics. Having learned robust surface semantic information in the first stage, we aim to learn the 3D character’s internal semantic and color information. We hierarchically supervise from outside to inside according to the spatial relationship of different semantic parts, by masking specific semantics during rendering and supervising with corresponding 2D ground truth. Assuming we aim to preserve a set of seman-

tics {Ps}, we can render the image and semantic map under current conditions as follows:

CˆP(r) =

N

i=1

TP,iαici

s∈P

ps,i, (6)

SˆP(r) =

N

i=1

TP,iαipi

s∈P

ps,i, (7)

where TP,i =

- i−1
- j=1

(1 − αj

s∈P

ps,j), (8)

The loss function is defined as:

L2 = Lmse,P + λlpipsLlpips,P + λmaskLmask,P

+ λsem

k

CE(SˆP,k,SP,kgt ) (9)

This decomposed training approach enables our S-LRM to simultaneously learn color and semantic information for the surface and the object’s interior, thus achieving feedforward 3D content decomposition and reconstruction.

- Stage 3: Training on Mesh with Multi-layer Semantics. We switch to mesh representation [49] for efficient highresolution training. We then extract the equivalent SDF via:

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

SDF

SDF

SDF

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Green

Green

[Figure 100]

[Figure 101]

[Figure 102]

<0

<0

<0

[Figure 103]

[Figure 104]

Selected GREEN as current SDF semantic

No semantic Selected

Selected RED as current SDF semantic

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Figure 4. Our semantic-equivalent NeRF and SDF extraction scheme (shown in yellow color).

part extraction. Upon analysis, the extraction of a semanticequivalent SDF should adhere to the following principles:

- 1. The zero value of the original SDF serves as a hard constraint. When the original SDF is positive, the equivalent SDF should also be positive;
- 2. When the original SDF is negative, the equivalent SDF should be zero at the boundaries where the maximum of relevant semantics transits;
- 3. At locations where the original SDF equals zero, but the probability of the current semantic is not the highest among all semantics, the equivalent SDF should not only maintain its sign but also be greater than zero. Based on these principles, we propose the following for-

mula for constructing the equivalent SDF:

##### pi,r) − pi,s), (3)

fi,s = max(fi,(max

r̸=s

Where fi, fi,s are original SDF and equivalent SDF of semantic s at location i, respectively. Fig. 4 illustrates our method’s scheme. For red semantics, only region 3 is selected, as regions 1, 2 (SDF>0) and region 4 (non-red) are discarded. Similarly, when the green is chosen, region 4 is correctly extracted. This formulation ensures correct decomposition by specific semantics and is fully compatible with subsequent FlexiCubes mesh extraction. In this way, we can differentiably extract multi-layer semantic surfaces from S-LRM’s outputs, greatly facilitating the LRM training and downstream optimization.

Model Structure. Our S-LRM’s core structure is derived from InstantMesh [63] with a ViT encoder, an image-totriplane transformer, and the feature decoder. Several enhancements are made to this foundation to better suit our objectives, with the semantic decoder being a key addition. This component mirrors the structure of the density/color decoder and is designed to generate the semantic field in 3D space. For memory efficient training, we incorporate LoRA [16, 41] structures into all linear layers inside both self-attention and cross-attention blocks.

Multiple semantic level supervision. Current LRMs typically rely solely on 2D supervision, which limits their ability to generate information about objects’ internal structures

pi,s)), (10)

pi,s − max

fi,P = max(fi,(max

s∈P

s/∈P

A-pose Conditioned Input Arbitrary-pose Conditioned Input SSIM↑ LPIPS↓ FID↓ CLIP Similarity↑ SSIM↑ LPIPS↓ FID↓ CLIP Similarity↑

SyncDreamer [30] 0.870 0.183 0.223 0.864 0.845 0.217 0.328 0.839 Zero-1-to-3 [29] 0.865 0.172 0.500 0.885 0.842 0.209 0.481 0.878 Era3D [26] 0.876 0.144 0.095 0.908 0.842 0.195 0.094 0.900 CharacterGen [39] 0.886 0.119 0.063 0.928 0.871 0.139 0.056 0.919 Ours 0.958 0.038 0.004 0.941 0.920 0.071 0.014 0.935

Multi-view Comparisons in 2D

Magic123 [42] 0.886 0.142 0.192 0.887 0.849 0.197 0.256 0.862 ImageDream [57] 0.856 0.171 0.846 0.836 0.823 0.218 0.875 0.818 OpenLRM [13] 0.889 0.151 0.406 0.878 0.863 0.191 0.707 0.844 LGM [53] 0.876 0.151 0.282 0.902 0.838 0.203 0.480 0.884 InstantMesh [63] 0.888 0.126 0.107 0.906 0.846 0.202 0.285 0.886 Unique3D [62] 0.889 0.136 0.030 0.919 0.856 0.190 0.042 0.903 CharacterGen [39] 0.880 0.124 0.081 0.905 0.869 0.134 0.119 0.901 Ours 0.937 0.066 0.010 0.941 0.916 0.084 0.011 0.936

Character Comparisons in 3D

Table 1. Quantitative comparison of A-pose and arbitrary pose inputs for 2D multi-view generation and 3D character generation.

Subsequently, we input the equivalent SDF into FlexiCubes to obtain the mesh, render the image and semantic map, and supervise using the following loss function:

MPgt ⊗ 1 − NˆP,k · NP,kgt

L3 = L2 + λnormal

k

MPgt ⊗ D ˆP,k − DP,kgt

+ λdevLdev (11)

+ λdepth

1

k

where DˆP,k, NˆP,k, denotes the rendered depth and normal; DP,kgt , NP,kgt and MPgt denote the ground truth depth, normal, and mask of the k-th view under semantic set P, respectively; Ldev denotes the deviation loss of FlexiCubes.

#### 3.3. Multi-layer Refinement

Given the limitations in geometric and texture detail achievable by large reconstruction models, further optimization of the mesh post-reconstruction is necessary. Recent methods [27, 62] utilizing high-resolution normal maps for mesh optimization have shown promising results, albeit primarily for holistic meshes. We propose an iterative optimization mechanism for multi-layer mesh refinement.

To ensure thorough optimization at each level, we employ a staged approach following our S-LRM: Initially, we extract different parts of the mesh by specifying various semantics and optimize only the base minimal-clothed human model; upon completion, we overlay the clothing mesh, fixing the base and optimizing solely the clothing component; finally, we add the hair mesh, fixing the previous two layers and optimizing only the hair. The optimization process is guided by the multi-view normal maps generated earlier through diffusion models, each optimization step involves a differentiable rendering of the mesh to compute losses and gradients, followed by vertex adjustments based on these gradients, and re-mesh operations including edge collapse, split and flips. The loss function is defined as follows:

||Mˆk − Mkpred||22 + λcolLcol

Lr1 = λ′mask

k

+ λ′normalMkpred ⊗

||Nˆk − Nkpred||22 (12)

k

where Mˆk, Nˆk are rendered masks and normal maps, Mkpred, Nkpred are diffusion-generated masks and normal maps under k-th view, respectively. Lcol is the collision loss modified from [38] to ensure outer-layer mesh in the outer normal direction of inner-layer mesh:

n

1 n

max((vj − vi) · nj,0)3 (13)

Lcol =

i=1

where vi represents the i-th vertex of the outer-layer mesh, vj is its nearest neighbor of vi on the inner-layer mesh, and nj denotes the normal vector associated with vj. Upon completing the optimization process, the mesh undergoes an additional ExplicitTarget Optimization phase, similar to that employed in Unique3D [62]. This stage aims to eliminate multi-view inconsistencies and further refine the geometry. Finally, the optimized meshes are colorized by the back projection of the multi-view images.

### 4. Experiments

#### 4.1. Anime3D++ Dataset

We develop the Anime3D++ Dataset to meet the quality and decomposability requirements based on the original Anime3D [39] Dataset. We initially gathered around 14,000 3D anime character models from VRoid-Hub, then filtered these down to 10,811 high-quality models, standardized in A-pose with arms angled 45° downward. Renderings include RGB images, depth, normal, and semantic maps for layered supervision. Each character is rendered in full, only a base minimal-clothed human model and a base human model plus clothing. Please refer to the supplementary materials (Sec. E) for more details.

#### 4.2. Results and Comparisons

Given that current methods cannot generate layered 3D models from a single arbitrary character image, we compare our non-layered generation results with other methods on the test split of our Anime3D++ dataset. We further decoupled the pose canonicalization component to ensure fairness, conducting both 2D and 3D comparisons for arbitrary

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

Reference Image Ours CharacterGen Unique3D InstantMesh

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

Figure 5. Qualitative comparisons on geometry and appearance of generated 3D characters.

pose and A-pose character reference inputs. When using A-pose inputs, all methods are compared against A-pose ground truth. For arbitrary pose inputs, we follow CharacterGen’s settings, comparing our method and CharacterGen (both capable of canonicalization) against the A-pose ground truth, while other methods are compared under the original pose ground truth. Generation quality and fidelity are evaluated using SSIM [59], LPIPS [74], and FID. We also compute CLIP [43] cosine similarity between the front ground-truth image and the multi-views or 3D renderings.

ically exhibit blurred geometry and suffer from the Janus Problem, and feed-forward methods generally lack geometric and texture precision. Unique3D achieves high metrics due to high-resolution supervision but suffers from unstable mesh initialization, impacting visual quality. CharacterGen demonstrates a notable advantage to other methods in arbitrary pose settings but the advantage diminishes in Apose, showing effective pose canonicalization but limited reconstruction ability. In contrast, our method consistently achieves superior results across all cases.

Quantitative Results. For 2D multi-view generation results, we compare our method with Zero-1-to-3 [29], SyncDreamer [30], Era3D [26], and CharacterGen [39]. For 3D Character Generation results, we compare with Magic123 [42], ImageDream [57] (SDS-based optimization); OpenLRM [13, 15], LGM [53], InstantMesh [63] (feed-forward methods); Unique3D [62] (direct mesh reconstruction); and CharacterGen. 3D Results are uniformly rendered as eight equidistant images at elevation=0 and aligned using horizontal mask registration.

Qualitative Results. Fig. 5 reveals several limitations across different methods. InstantMesh’s results were constrained by grid resolution, leading to insufficient texture precision. While Unique3D achieved higher resolution, its heavy reliance on depth-based mesh initialization made it susceptible to geometric collapse under inaccurate depth estimations. CharacterGen exhibited low geometric and texture fidelity despite employing multi-view back-projection, and frequently produced visually disruptive black artifacts. In contrast, our method demonstrated superior geometric accuracy and texture fidelity performance.

As shown in Tab. 1, Our method performs better in both standard and arbitrary pose settings. Existing 2D multiview generation methods often struggle to preserve adequate geometric and appearance information in generated images. Among 3D methods, SDS-based approaches typ-

Decomposed Results. Fig. 6 illustrates our method’s capability to reconstruct decomposed characters from single reference images, organized from left to right as follows: the input reference image, the reconstructed 3D character (ap-

[Figure 179]

Reference Image

Generated 3D Character

Decomposed Results

Geometry & Cross-sections

[Figure 180]

[Figure 181]

see supplementary material (Sec. A) for details.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

#### 4.3. Applications

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

3D Editing in 2D Domain

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Front capturing

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Our Recon Pipeline

or Using reference image

[Figure 209]

[Figure 210]

“A girl with blue skirt, anime style”

[Figure 211]

[Figure 212]

Figure 6. Decomposed outputs of our method, presented in texture, mesh, and cross-section.

“A girl with short brown hair, anime style”

Figure 7. Our pipeline enables diverse 3D editing using only text prompts, masks, and in-painting diffusion in the 2D domain.

Overall Quality

Geometric Quality

Texture Quality

Method

Fidelity

CharacterGen [39] 4.0% 4.5% 6.2% 4.2% InstantMesh [63] 1.8% 2.0% 1.8% 2.5% Unique3D [62] 11.8% 18.3% 10.5% 17.0% Ours 82.4% 75.2% 81.5% 76.3%

Our method’s capability to generate decomposed characters in A-pose not only facilitates simpler rigging but also enables 3D editing through user-specified prompts and inpainting masks in the 2D domain. As illustrated in Fig. 7, our framework allows for effortless character customization, including outfit changes and hairstyle modifications. The process requires minimal user input: the original reference image or a front-view capture, a roughly sketched in-painting mask, and a text prompt. Utilizing existing inpainting models (HD-Painter [34] in this case), we achieve the desired 2D edits. Subsequently, these 2D modifications are translated into the 3D domain through our reconstruction and optimization pipeline. During the multi-layer refinement stage, the relevant mesh components are selectively replaced with the edited versions while preserving the unmodified parts. This bridges the gap between 2D image editing and 3D character customization and streamlines the 3D character modification process. We further provide animation comparisons in supplementary material (Sec. B).

Table 2. User study results comparing different methods across four dimensions. The values represent the percentage of times each method was selected as the best in the respective dimension.

plied shading for better visualization), the semantically decomposed components and geometries. The reconstructed meshes demonstrate high geometric precision, while the semantic decomposition is also accurate, successfully decoupling the base human model, clothing, and hair. This level of decomposition represents a significant advancement in character reconstruction from single images. The rightmost figure presents a cross-sectional view of the clothing, revealing that our reconstructed clothing is entirely internally hollow. This substantially enhances the potential for integration with downstream applications like realistic physics simulations and easier rigging.

User Study. To comprehensively evaluate our method’s performance, we conducted a user study involving 28 volunteers. The study utilized 16 randomly selected 3D meshes and their corresponding reference images, drawn from both web-collected images and the Anime3D++ test split. Participants were asked to compare the randomly shuffled results of four approaches based on overall quality, fidelity, geometric quality, and texture quality. For each comparison group, participants identified the best result in each dimension. Tab. 2 shows that our method achieved superior performance across all dimensions, demonstrating its effectiveness in generating high-quality 3D characters.

### 5. Conclusion

In this paper, we introduce StdGEN, an innovative pipeline for generating semantic decomposed high-quality 3D characters from single images. A novel semantic-aware large reconstruction model is proposed to jointly reconstruct geometry, color and semantics, together with an efficient 2D diffusion model and iterative multi-layer refinement module to enable the high-quality generation from arbitrary-posed single image. StdGEN achieves superior performance over existing baselines in terms of geometry, texture and decomposability, and offers ready-to-use, semantic-decomposed, customizable 3D characters, demonstrating significant potential for various applications.

Ablation Study. We further conduct ablation studies on character decomposition and multi-layer refinement. Please

Acknowledgement. This work was partially supported by Beijing Science and Technology plan project (Z231100005923029), Beijing Natural Science Foundation (L247007), and the Natural Science Foundation of China (62461160309, 62332019).

### References

- [1] Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5470–5479, 2022. 2
- [2] Alexander W. Bergman, Petr Kellnhofer, Wang Yifan, Eric R. Chan, David B. Lindell, and Gordon Wetzstein. Generative neural articulated radiance fields, 2023. 3
- [3] Yukang Cao, Yan-Pei Cao, Kai Han, Ying Shan, and KwanYee K. Wong. Dreamavatar: Text-and-shape guided 3d human avatar generation via diffusion models, 2023. 3
- [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 4
- [5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European Conference on Computer Vision, pages 333–350. Springer,

2022. 2

- [6] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields, 2024. 3
- [7] Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for highquality text-to-3d content creation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 22246–22256, 2023. 2
- [8] Ruikai Cui, Xibin Song, Weixuan Sun, Senbo Wang, Weizhe Liu, Shenzhou Chen, Taizhang Shang, Yang Li, Nick Barnes, Hongdong Li, and Pan Ji. Lam3d: Large image-point-cloud alignment model for 3d reconstruction from single image,

2024. 3

- [9] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-xl: A universe of 10m+ 3d objects, 2023. 2, 5
- [10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 2
- [11] Junting Dong, Qi Fang, Zehuan Huang, Xudong Xu, Jingbo Wang, Sida Peng, and Bo Dai. Tela: Text to layer-wise 3d clothed human generation. arXiv preprint arXiv:2404.16748, 2024. 2, 3, 4
- [12] Ayaan Haque, Matthew Tancik, Alexei A Efros, Aleksander Holynski, and Angjoo Kanazawa. Instruct-nerf2nerf:

- Editing 3d scenes with instructions. arXiv preprint arXiv:2303.12789, 2023. 2
- [13] Zexin He and Tengfei Wang. Openlrm: Open-source large reconstruction models. https://github.com/ 3DTopia/OpenLRM, 2023. 6, 7
- [14] Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, and Ziwei Liu. Eva3d: Compositional 3d human generation from 2d image collections. arXiv preprint arXiv:2210.04888,

2022. 2, 3

- [15] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2, 3, 7
- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5, 3
- [17] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 4
- [18] Shuo Huang, Zongxin Yang, Liangting Li, Yi Yang, and Jia Jia. Avatarfusion: Zero-shot generation of clothingdecoupled 3d avatars using 2d diffusion. In Proceedings of the 31st ACM International Conference on Multimedia, pages 5734–5745, 2023. 2
- [19] Yukun Huang, Jianan Wang, Ailing Zeng, He Cao, Xianbiao Qi, Yukai Shi, Zheng-Jun Zha, and Lei Zhang. Dreamwaltz: Make a scene with complex 3d animatable avatars, 2023. 3
- [20] Zehuan Huang, Hao Wen, Junting Dong, Yaohui Wang, Yangguang Li, Xinyuan Chen, Yan-Pei Cao, Ding Liang, Yu Qiao, Bo Dai, et al. Epidiff: Enhancing multi-view synthesis via localized epipolar-constrained diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9784–9794, 2024. 3
- [21] Suyi Jiang, Haoran Jiang, Ziyu Wang, Haimin Luo, Wenzheng Chen, and Lan Xu. Humangen: Generating human radiance fields with explicit priors, 2022. 3
- [22] Taeksoo Kim, Byungjun Kim, Shunsuke Saito, and Hanbyul Joo. Gala: Generating animatable layered assets from a single scan. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1535– 1545, 2024. 3, 4
- [23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 5
- [24] Nikos Kolotouros, Thiemo Alldieck, Enric Corona, Eduard Gabriel Bazavan, and Cristian Sminchisescu. Instant 3d human avatar generation using image diffusion models. In European Conference on Computer Vision, pages 177–195. Springer, 2024. 5
- [25] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. In

- The Twelfth International Conference on Learning Representations. 3
- [26] Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang Zhang, Wenhan Luo, Ping Tan, et al. Era3d: High-resolution multiview diffusion using efficient row-wise attention. arXiv preprint arXiv:2405.11616, 2024. 4, 6, 7
- [27] Weiyu Li, Jiarui Liu, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman: High-fidelity mesh generation with 3d native generation and interactive geometry refiner. arXiv preprint arXiv:2405.14979, 2024. 3, 6
- [28] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2
- [29] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9298–9309, 2023. 3, 6, 7
- [30] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. The Twelfth International Conference on Learning Representations, 2024. 3, 6, 7
- [31] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9970–9980, 2024. 3
- [32] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. Smpl: A skinned multiperson linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2, pages 851–866. 2023. 2
- [33] Yuanxun Lu, Jingyang Zhang, Shiwei Li, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, Xun Cao, and Yao Yao. Direct2. 5: Diverse text-to-3d generation via multi-view 2.5 d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8744– 8753, 2024. 3
- [34] Hayk Manukyan, Andranik Sargsyan, Barsegh Atanyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Hd-painter: high-resolution and prompt-faithful text-guided image inpainting with diffusion models. arXiv preprint arXiv:2312.14091, 2023. 8
- [35] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [36] Atsuhiro Noguchi, Xiao Sun, Stephen Lin, and Tatsuya Harada. Unsupervised learning of efficient geometry-aware neural articulated representations, 2022. 3
- [37] Panwang Pan, Zhuo Su, Chenguo Lin, Zhen Fan, Yongjie Zhang, Zeming Li, Tingting Shen, Yadong Mu, and Yebin

- Liu. Humansplat: Generalizable single-image human gaussian splatting with structure priors. arXiv preprint arXiv:2406.12459, 2024. 2
- [38] Bo Peng, Yunfan Tao, Haoyu Zhan, Yudong Guo, and Juyong Zhang. Pica: Physics-integrated clothed avatar. arXiv preprint arXiv:2407.05324, 2024. 2, 6
- [39] Hao-Yang Peng, Jia-Peng Zhang, Meng-Hao Guo, Yan-Pei Cao, and Shi-Min Hu. Charactergen: Efficient 3d character generation from single images with multi-view pose canonicalization. ACM Transactions on Graphics (TOG), 43(4): 1–13, 2024. 2, 3, 4, 6, 7, 8, 1
- [40] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion, 2022. 2, 4
- [41] Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv preprint arXiv:2407.06191, 2024. 5
- [42] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843,

2023. 6, 7

- [43] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021. 7, 4
- [44] Amit Raj, Srinivas Kaza, Ben Poole, Michael Niemeyer, Nataniel Ruiz, Ben Mildenhall, Shiran Zada, Kfir Aberman, Michael Rubinstein, Jonathan Barron, et al. Dreambooth3d: Subject-driven text-to-3d generation. arXiv preprint arXiv:2303.13508, 2023. 2
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [46] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022. 2
- [47] Ruizhi Shao, Jingxiang Sun, Cheng Peng, Zerong Zheng, Boyao Zhou, Hongwen Zhang, and Yebin Liu. Control4d: Dynamic portrait editing by learning 4d gan from 2d diffusion-based editor. arXiv preprint arXiv:2305.20082,

2023. 2

- [48] Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, and Sanja Fidler. Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. In Advances in Neural Information Processing Systems (NeurIPS), 2021. 2
- [49] Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler,

- Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. ACM Trans. Graph., 42(4):37–1, 2023. 3, 4, 5, 1
- [50] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In The Twelfth International Conference on Learning Representations. 3
- [51] Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, et al. Text-to-4d dynamic scene generation. arXiv preprint arXiv:2301.11280,

2023. 2

- [52] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5459– 5469, 2022. 2
- [53] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. arXiv preprint arXiv:2402.05054, 2024. 3, 6, 7
- [54] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:2304.12439, 2023. 2
- [55] Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation, 2022. 2
- [56] Jionghao Wang, Yuan Liu, Zhiyang Dou, Zhengming Yu, Yongqing Liang, Xin Li, Wenping Wang, Rong Xie, and Li Song. Disentangled clothed avatar generation from text descriptions. arXiv preprint arXiv:2312.05295, 2023. 2, 3
- [57] Peng Wang and Yichun Shi. Imagedream: Image-prompt multi-view diffusion for 3d generation. arXiv preprint arXiv:2312.02201, 2023. 6, 7
- [58] Yi Wang, Jian Ma, Ruizhi Shao, Qiao Feng, Yu-Kun Lai, and Kun Li. Humancoser: Layered 3d human generation via semantic-aware diffusion model. arXiv preprint arXiv:2408.11357, 2024. 2
- [59] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004. 7
- [60] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:2305.16213, 2023. 2
- [61] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. Crm: Single image to 3d textured mesh with convolutional reconstruction model. arXiv preprint arXiv:2403.05034, 2024. 3
- [62] Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024. 6, 7, 8, 4
- [63] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d

- mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024. 3, 4, 5, 6, 7, 8
- [64] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 2
- [65] Han Yan, Yang Li, Zhennan Wu, Shenzhou Chen, Weixuan Sun, Taizhang Shang, Weizhe Liu, Tian Chen, Xiaqiang Dai, Chao Ma, et al. Frankenstein: Generating semanticcompositional 3d scenes in one tri-plane. arXiv preprint arXiv:2403.16210, 2024. 3, 4
- [66] Xueting Yang, Yihao Luo, Yuliang Xiu, Wei Wang, Hao Xu, and Zhaoxin Fan. D-if: Uncertainty-aware human digitization via implicit distribution field. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9122–9132, 2023. 5
- [67] Yifan Yang, Dong Liu, Shuhai Zhang, Zeshuai Deng, Zixiong Huang, and Mingkui Tan. Hilo: Detailed and robust 3d clothed human reconstruction with high-and lowfrequency information of parametric models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10671–10681, 2024. 5
- [68] Xu Yinghao, Shi Zifan, Yifan Wang, Chen Hansheng, Yang Ceyuan, Peng Sida, Shen Yujun, and Wetzstein Gordon. Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation, 2024. 3
- [69] Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. Function4d: Real-time human volumetric capture from very sparse consumer rgbd sensors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5746–5756, 2021. 5
- [70] Chubin Zhang, Hongliang Song, Yi Wei, Yu Chen, Jiwen Lu, and Yansong Tang. Geolrm: Geometry-aware large reconstruction model for high-quality 3d gaussian generation,

2024. 3

- [71] Jianfeng Zhang, Zihang Jiang, Dingdong Yang, Hongyi Xu, Yichun Shi, Guoxian Song, Zhongcong Xu, Xinchao Wang, and Jiashi Feng. Avatargen: a 3d generative model for animatable human avatars, 2022. 3
- [72] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. European Conference on Computer Vision, 2024. 3
- [73] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 3
- [74] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 7

## StdGEN: Semantic-Decomposed 3D Character Generation from Single Images Supplementary Material

### A. Ablation Study

[Figure 213]

[Figure 214]

Non-Decomposed

[Figure 215]

[Figure 216]

[Figure 217]

Decomposed

Ours w/o Decomposition Ours w/ Decomposition

Figure 8. Ablation study on character decomposition.

Character Decomposition. To demonstrate the decomposition capabilities of our core S-LRM and its impact on the results, we compared our method with a direct refinement approach that does not employ semantic decomposition. The visual comparison in Fig. 8 reveals that without decomposition, the results exhibit a fusion of hair, clothing, and the base human model, significantly limiting their potential for downstream applications. In contrast, our method successfully separates these components while maintaining high mesh precision, showcasing the effectiveness of our semantic decomposing approach.

Figure 10. Rigging and animation comparisons on 3D character generation. Our method demonstrates superior performance in human perception and physical characteristics.

poses various mesh components with correct geometry and shape, validating the capabilities of our S-LRM. However, the precision is limited due to the inherent characteristics of FlexiCubes [49] and memory constraints. Post-refinement, we observe a substantial enhancement in precision while maintaining the overall structure. This improvement underscores the effectiveness of our multi-layer refinement process in preserving the structure of the decomposed components while significantly elevating the geometric accuracy and overall quality of the reconstructed character.

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

| |
|---|

[Figure 223]

| |
|---|

| |
|---|

| |
|---|

### B. More Applications

[Figure 224]

[Figure 225]

| |
|---|

| |
|---|

Compared to other 3D character generation methods, our decomposed generation in A-pose is more suitable for downstream animation and 3D applications. In Fig. 10, we rig the 3D character generated by our approach and by CharacterGen [39] for comparison. Without decomposition, the hair and clothing stick together and are attached to the base human model. In contrast, our approach maintains separated parts, aligning more closely with natural perception. Additionally, the non-decomposed nature leads to inaccurate deformations and physical characteristics during movement, which our method effectively avoids.

Before Multi-layer Refine After Multi-layer Refine

Figure 9. Ablation study on multi-layer refinement. Zoom in for better details.

Multi-layer Refinement. We further illustrate the distinction between the direct output of our S-LRM and the results after multi-layer refinement in Fig. 9. The pre-optimization results demonstrate that our S-LRM successfully decom-

### C. Time Breakdown Analysis

Process Time (s) Canonicalization Diffusion 7 Multi-view Diffusion 29 S-LRM 12 Refinement

- - Single-layer Settings 18
- - Multi-layer Settings 117

Table 3. Time breakdown for each processing step.

In Tab. 3, we present the time breakdown of different components in our method. Creating a single-layered 3D character takes only about 1 minute while generating a decomposed, multi-layered 3D character requires less than 3 minutes. Our S-LRM is relatively efficient, with minimal additional time and memory overhead compared to InstantMesh’s LRM.

### D. More Quantitative Results

3D Semantic Metrics. To demonstrate our 3D semantic decomposition capability, we extracted separate meshes for three semantic categories (hair, cloth, and base human model) and rendered masks from eight different views for comparison with ground truth. In the arbitrary pose setting, we achieved IoU scores of 0.73 for hair, 0.86 for cloth, and

- 0.88 for the base human model. These results demonstrate effective semantic decomposition, particularly considering the significant challenges in single-image-based 3D reconstruction, such as spatial ambiguity and occlusion. 3D Geometric Metrics. Additional comparisons in the arbitrary pose setting are presented in Tab. 4, where our method demonstrates superior performance on both the Chamfer distance and the F-score, showing a more precise prediction of 3D geometry.

Metric Unique3D CharacterGen Ours Chamfer Distance ↓ 0.109 0.035 0.023 F-Score ↑ 0.137 0.465 0.664

Table 4. 3D metric comparison (0.01 for f-score threshold).

Ablation Study on Refinement Stage. We conduct experiments without refinement in Tab. 5 under arbitrary pose setting, where we only apply color back-projection to the mesh generated by S-LRM. The results indicate that our method outperforms CharacterGen and Unique3D even without refinement, demonstrating that our S-LRM is effective even without the refinement step.

Method SSIM ↑ LPIPS ↓ FID ↓ CLIP Similarity ↑

Unique3D 0.856 0.190 0.042 0.903 CharacterGen 0.869 0.134 0.119 0.901 Ours (w/o refine) 0.912 0.094 0.026 0.933 Ours 0.916 0.084 0.011 0.936

Table 5. Ablation study on refinement stage.

### E. Dataset Construction

Semantic Definition. Unlike general 3D models, 3D character modeling typically involves multiple components rather than a single entity. This segmentation is essential for downstream applications such as rigging and physical simulation, which often require the manipulation of distinct parts. Conventional reconstruction models only capture the character’s surface appearance, lacking internal information and the ability to decompose the model, which limitation severely restricts subsequent applications. After considering both practical applications and data composition, we have categorized character composition into three semantic categories: base minimal-clothed human model, clothing, and hair (specifically, shoes and underwear are classified as part of the base human model, considering downstream applications and data characteristics). By incorporating these semantic categories into our reconstruction process, we aim to produce 3D character models that are not only visually accurate but also functionally versatile for various uses in 3D game and animation pipelines. Note that our method supports the learning and extracting of an arbitrary number of semantic categories.

Data Cleaning. We begin by filtering out data that cannot be layered according to semantic structure. Using multiple prompts combined with ImageReward [64], we remove low-quality or malicious data with low scores. Additionally, we identify instances where semantic information predictions are inaccurate, then manually review and remove data that appears semantically incorrect to human perception. Since the base human model in the original dataset can occasionally contain defects, we apply a connectivity check on the front rendering of every base human model. Examples lacking connectivity are only used to supervise either the complete model or the base human model with clothing, but not the base human model alone.

Rendering Settings. Our rendering process goes beyond standard image generation, incorporating depth, normal, and semantic maps. 3D Character models are adjusted to an A-pose configuration, with arms rotated 45 degrees downward from the horizontal position. To facilitate multilayered reconstruction supervision, we rendered the complete model and two additional configurations: the base minimal-clothed human model alone, and the base model with clothing. The rendering includes eight views at 45degree azimuth intervals with zero elevation, supplemented

by top-down and bottom-up views. We enriched the dataset with five close-up facial views and 20 random viewpoints. To enhance the training of our diffusion model, we implemented data augmentation on varying arm angles.

We use the orthographic camera for all renderings. For non-close-up views of the character, after normalizing the character to fit within a unit cube, we set the ortho scale to

- 1.2. For close-up views of the face, we locate the 3D center position and bounding box of the facial semantics, aligning the camera’s center projection with the 3D center of the face and setting the ortho scale to 1.2 times the bounding box size. We render five facial close-up views at elevation = 0° and azimuth angles of {−90◦,−45◦,0◦,45◦,90◦}. For inputs to the canonicalization diffusion model, we add outline and shading with a 50% probability. Rim lighting, shading, and outlines (except on the face) are consistently removed

- to supervise diffusion and S-LRM outputs. Semantic maps are rendered by modifying non-transparent regions of the texture map assigned to specific semantic parts. Multi-layer Settings. We provide three different rendering levels: the complete model, the base human model only, and the base human model with clothes, each generated by selectively removing specific semantic elements. For supervising S-LRM, these correspond to (1) no semantic masking, (2) masking of hair and clothing, and (3) masking of hair only. By mixing these levels of 2D supervision, we can train S-LRM to reconstruct multi-layered density, color, and semantic information automatically.

### F. Details of Loss Functions

In this section, we provide a detailed description of each loss component in our framework. Lmse is the commonly used mean squared error loss defined as:

Lmse =

k

I ˆk − Ikgt

2 2

(14)

where Iˆk, Ikgt denotes the k-th view of rendered images and ground-truth images, respectively.

Llpips is the perceptual loss defined as

Llpips =

k

τ ϕ(Iˆk),ϕ(Ikgt) (15)

where ϕ is the VGG feature extractor, τ transforms deep embedding to a scalar LPIPS score.

Lmask is the mask loss defined as

2 2

M ˆk − Mkgt

(16)

Lmask =

k

where Mˆk and Mkgt denote the rendered non-transparent mask, and ground-truth masks, respectively.

The deviation loss Ldev penalizes the Euclidean distances between each dual vertex v and the edge crossings ue ∈ Nv that bound its primal face, encourages vertices to center within their cells and allowing flexibility for connectivity adaptation:

MAD[{|v − ue|2 : ue ∈ Nv}] (17)

Ldev =

v∈V

where | · |2 is the Euclidean distance, MAD(Y ) =

- 1

|Y | y∈Y |y − mean(Y )| denotes the mean absolute deviation. We use the same approach as the implementation of InstantMesh [63], applying L2 regularization with a weight of 0.1 to the FlexiCubes weights.

During S-LRM training, we specifically incorporated facial semantics as an additional component to enhance the training process and facilitate potential applications. In subsequent stages, facial semantics were treated as an integral part of the base human model. For the semantic crossentropy loss Lsem, we empirically assigned weights to four semantic categories - hair (1.255), face (1.758), base human model (0.913), and cloth (0.650) - based on their respective rendering proportions in the dataset to optimize semantic learning.

G. Implementation Details

We divide our Anime3D++ dataset into a training and testing set in a 99:1 ratio. We first train the canonicalization diffusion model at a 512 resolution with a learning rate of 5e-5, then reduce it to 1e-5 as we progressively increase the resolution to 768 and 1024. Similarly, the multi-view diffusion model is trained at a constant learning rate of 5e-5 while scaling from 512 to 768 and finally to 1024 resolution. We use LoRA with a 128-rank for S-LRM, a learning rate of 4e-5, and three supervision stages with rendering resolutions of 192, 144, and 512. The loss function parameters are set as λlpips,λmask,λsem,λdepth,λnormal,λdev =

- 2.0,1.0,1.0,0.5,0.2,0.5. For multi-layer refinement, we

set λ′mask,λ′normal,λcol = 1.0,1.0,100.0, and we further extract the coarse hair mask, applying additional normal and mask loss for hair refinement with a weight of 1 and 10.

Detailed Structure of S-LRM. Following InstantMesh [63], our S-LRM adopts 6 RGB images generated by multi-view diffusion in a resolution of 320 × 320 as model input. In the training stage 3 (training on meshes with multi-layer semantics), we set the sampling grid size for FlexiCubes extraction to 100 × 100 × 150, with dimensions scaled to 0.7 × 0.7 × 1.05 of the triplane size, and the centers of both are aligned. We integrate the LoRA [16] structure into the S-LRM transformer, modifying both the self-attention and cross-attention modules. For self-attention, where q, k, and v values are produced by shared linear layers, we substitute all input and output

linear layers with LoRA structures. In cross-attention, where q, k, and v are produced through separate linear layers, we replace the linear layers for q, k, v, and the outputs with LoRA structures. The specifics are as follows:

hi = W0i + ∆Wtpi x = W0ix + Btpi Aitpx (18) Here, i denotes the i-th transformer layer. In self-attention,

- tp represents the linear projection for inputs and outputs, while in cross-attention, tp denotes the linear projections for q, k, v, and outputs. During the training process, the DINO [4] encoder is kept frozen while the feature decoder (including color/density decoder and semantic decoder) remains trainable. In the triplane transformer, the positional embeddings, de-convolution layers, and all LoRA layers are set as learnable parameters, while all other layers are frozen. Detailed Settings of Canonicalization Diffusion. Our canonicalization diffusion model comprises a U-Net and a ReferenceNet with an identical architecture, both networks are initialized with the weights from Stable Diffusion 2.1. The U-Net takes the CLIP-encoded features of the reference image as input for the encoder-hidden states. ReferenceNet, on the other hand, receives the image latents of the reference image (encoded by a VAE without added noise) as input, along with the features of a fixed text prompt, “high quality, best quality,” encoded by CLIP [43], which are fed into the encoder-hidden states. A cross-attention operation is applied for each corresponding layer pair in the U-Net and ReferenceNet, using the current U-Net layer as the query and the corresponding ReferenceNet layer as the key and value. This cross-attention mechanism transfers the detailed information of the reference image into the U-Net. Detailed Settings of Multi-View Diffusion. Building upon Era3D’s [26] multi-view model, we start training from its inherited weights. We concatenate the noisy VAE latent and reference image VAE latent as input to the U-Net. For each view’s color and normal output, we specify fixed prompts (”a rendering image of 3D models, {view} view, {color/normal} map”), which are encoded through CLIP and fed into the U-Net’s encoder hidden states. For U-Net’s class labels, we reserve the first 1024 dimensions for the CLIP embedding of the reference image, and replace the noise level embedding in the latter 1024 dimensions with a level switcher. This level switcher uses different one-hot vectors for three distinct rendering levels to support the specific semantic combinations in the diffusion output, serving as supervision signals for multi-layer refinement. Since the previous canonicalization diffusion step already ensures that the output A-pose character reference image has elevation=0 and is orthographic, we do not employ the regression loss from Era3D. Additionally, we fix the noise level of the image VAE latent to 0 to achieve optimal fidelity. Details of Color Back-projection. We employ a multiview projection method similar to Unique3D [62] to back-

project the texture onto the 3D character model. For each vertex v that is visible in at least one view, we calculate its final color C(v) in the 3D mesh using the following formula:

wi(nv · di)2cv,i wi(nv · di)2

(19)

C(v) =

i∈I

where cv,i represents the color corresponding to v in the i-th view texture; nv and ni is the vertex normal of v and the view direction of the i-th view respectively; wi is the projection weight of the i-th view, and I is the set of views where v is visible. In practice, we assign wi values of {2.0,0.5,0.0,1.0,0.0,0.5} for views at azimuths of {0◦,45◦,90◦,180◦,270◦,315◦} respectively. For vertices that are not visible in any view, we treat the 3D mesh as a graph composed of vertices and edges, and iteratively perform convolution and mean pooling to transfer colors from vertices with determined colors to those without, until all vertices obtain their colors.

Pre- and Post-dilation of Mesh. To better solve the problematic intersections among meshes in the multi-layer refinement stage, we introduce a ”dilation” process applied both before and after optimization. This process constructs an approximate ”flow field” based on the original positions of the inner and outer layer meshes.

For each vertex on the outside mesh, we utilize a kdtree to query its nearest vertex neighbors of the fixed inside mesh. The movement range is then smoothly weighted based on the exponential inverse distance from these neighbors, with distant points remaining stationary. This approach creates a ”dilation” effect, ensuring that when inner layers are moved outward to resolve intersections, the outer layers follow suit in a natural, gradual manner.

### H. Discussions

Comparison with other decomposition methods. We note that some works have also applied the concept of decomposition, while they differ from our method in problem definition and scope. GALA [22] and TELA [11] use realworld 3D mesh scans and text as input respectively, employing Score Distillation Sampling (SDS) [40] with classspecific text prompts and pose control for layered 3D avatar generation. Frankenstein [65] takes a textureless, 2D semantic layout as input and generates semantic-decomposed 3D meshes (also textureless) through triplane diffusion. In contrast, our method accepts RGB reference images of arbitrary characters and generates 3D character meshes that faithfully preserve the reference texture while enabling semantic decomposition in a feed-forward manner.

Regarding specific decomposition techniques, GALA and TELA use SDS and different prompts to optimize both individual parts and the whole iteratively, typically requiring hours or more for a single case; Frankenstein outputs

separate SDFs for each semantic class, training and inferring on datasets with specific semantics, while our method treats geometry and semantics information independently, enabling greater compatibility and scalability. Our method can extract equivalent surfaces by specifying any combination of semantics, while maintaining compatibility with general datasets like Objaverse [9] and preserving LRM’s general performance. It also has the potential to achieve semantic decomposition for multiple data types through a single LRM paired with multiple semantic decoders. Moreover, when adding new semantic classes, our approach can inherit geometric priors, making fine-tuning more efficient.

[Figure 226]

[Figure 227]

Reference AvatarPopUp Ours

[Figure 228]

[Figure 229]

“Wearing a purple shirt and grey jacket”

Figure 13. 3D editing comparison with AvatarPopUp.

[Figure 230]

Semantic definition and possible improvements. Our Anime3D++ dataset adopts the VRoid-Hub data standard, which is designed to align with VR/game requirements, particularly for animation and collision detection. Following this standard, close-fitting garments are classified as part of base human model, while outerwear (e.g., pants, skirts, hoodies with long sleeves) is categorized as clothing. Although our results currently support relatively few semantic categories due to the limitations in datasets, our method is general and the results demonstrate the feasibility of semantic awareness generation. In future work, our framework can be easily extended to support fine-grained semantic decomposition by incorporating Segment Anything Model (SAM) [23] to generate detailed semantic labels for S-LRM training.

[Figure 231]

Reference Image D-IF HiLo Ours

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

Design choice of Semantic-Equivalent SDF. Here we discuss why we don’t assign a dedicated decoder for each semantic class to predict their SDFs separately. First, this approach would not effectively utilize the prior knowledge from the NeRF training stage. The SDF information predicted by these decoders would differ significantly from what was learned in the previous stage, with each decoder only retaining ”its own” portion of the whole original SDF. The semantic information learned during the last stage would also become unusable. In contrast, our method almost completely inherits the prior knowledge from the NeRF training stage and smoothly transitions to the SDF stage without any modifications to the network architecture.

- Figure 11. Comparison on THuman 2.0 dataset.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Reference Image Results Reference Image Results

- Figure 12. Our result on furry and 2.5D style images.

Non-anime style results. Our method demonstrates generalization across diverse character types, as shown in Figs. 11 and 12. For realistic style, we compare with D-IF [66] and HiLo [67] on THuman 2.0 dataset [69]. While the results show slight stylistic bias inherently from the Anime3D++ training data (e.g., slim faces), our method is general with robustness, canonicalization, and decomposition capabilities even without real-human training data. To further show our method’s 3D editing and decomposing ability, we also directly compare the editing case in AvatarPopUp [24] (Fig. 13). Our approach shows similar effectiveness on realhuman examples as AvatarPopUp and offers semantic decomposition and style flexibility capabilities.

Additionally, this alternative approach would suffer from poor scalability - adding a new semantic class would require adding and training a new triplane feature decoder nearly from scratch and modifying the network structure, while our method does not require relearning geometric information when adding or removing semantics. It would also increase computational and memory costs during both training and inference. Furthermore, without cross-semantic constraints, the surfaces extracted from SDFs of different semantic classes might intersect, which contradicts our requirements. Another issue is that such a representation would not be unified - we could only extract a surface for each individual semantic class, but not for the entire char-

[Figure 252]

- Figure 14. More visualizations on semantic-decomposed 3D generations.

acter or multiple selected semantic classes simultaneously. In contrast, our solution can extract equivalent surfaces for any combination of selected semantic classes with only one decoder employed.

Limitations. We note several limitations that leave room for future work: (1) Following InstantMesh [63], our SLRM produces triplanes with the resolution of 64 × 64. After switching to SDF training, the FlexiCubes sampling uses a grid size of only 150 height and 100 width. This resolution may constrain and limit further improvement in the results. (2) While our pipeline enables high-quality generation by the high-resolution diffusion output up to a resolution of 1024, this also slows the overall generation speed, presenting a trade-off. Our S-LRM requires only about 10 seconds for one inference, with the majority of time con-

sumed in the diffusion and refinement, suggesting room for further optimization. (3) The restricted style and category diversity in the training data affects its ability to handle inputs that deviate significantly from the human-centric categories (e.g., animals or general 3D objects). Despite such challenges, our framework is extensible, allowing further improvements through tools like SAM for semantic labeling or SMPL label transfer for human datasets.

### I. More Visualizations

We demonstrate more visualizations of semanticdecomposed 3D results (Fig. 14), outputs of canonicalization (Fig. 16) and multi-view (Fig. 15) diffusion model, comparisons with other methods (Fig. 17, 18) and multi-view renderings (Fig. 19).

[Figure 253]

###### Figure 15. Visualizations of the 2D multi-view diffusion model results.

[Figure 254]

###### Figure 16. Visualizations of the 2D canonicalization diffusion model results.

[Figure 255]

###### Figure 17. More qualitative comparisons of 3D character generations (#1).

[Figure 256]

###### Figure 18. More qualitative comparisons of 3D character generations (#2).

[Figure 257]

###### Figure 19. More qualitative comparisons of 3D character generations (multi-view renderings).

