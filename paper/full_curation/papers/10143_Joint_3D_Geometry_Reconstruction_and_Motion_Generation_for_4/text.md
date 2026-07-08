# arXiv:2512.05044v1[cs.CV]4Dec2025

## Joint 3D Geometry Reconstruction and Motion Generation for 4D Synthesis from a Single Image

Yanran Zhang∗,1 Ziyi Wang∗,1 Wenzhao Zheng†,1 Zheng Zhu2 Jie Zhou1 Jiwen Lu1 1 Department of Automation, Tsinghua University, China 2 GigaAI https://ivg-yanranzhang.github.io/MoRe4D/

###### Input Image Viewpoint

4D Representation

Input Image

3D Representation 4D Representation

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

Gen

Gen

Recon

Recon

[Figure 33]

[Figure 34]

Temporal Motion Generation

4D Spatial Reconstruction

Single-View 3D Reconstruction

Temporal Motion Generation

[Figure 35]

[Figure 36]

Time

(a) Generation → Reconstruction

(b) Reconstruction → Generation: Vanilla

A furry monster's hand reaches for a candle flame.

Render

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

Recon

[Figure 53]

[Figure 54]

[Figure 55]

Recon Gen

Geometric Prior

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

Motion

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### Input Single Image

4D Scene Representation

Joint Model

Motion Prior

(c) Reconstruction → Generation: MoRe4D

Figure 1. MoRe4D for 4D synthesis from a single image. Most existing paradigms either suffer from geometric inconsistencies (generatethen-reconstruct) or are constrained by animating a pre-determined static geometry (vanilla reconstruct-then-generate). Our MoRe4D advances by tightly coupling geometric modeling and motion generation, effectively achieving consistent 4D motion and geometry.

### Abstract

Generating interactive and dynamic 4D scenes from a single static image remains a core challenge. Most existing generate-then-reconstruct and reconstruct-thengenerate methods decouple geometry from motion, causing spatiotemporal inconsistencies and poor generalization. To address these, we extend the reconstruct-thengenerate framework to jointly perform Motion generation and geometric Reconstruction for 4D Synthesis (MoRe4D). We first introduce TrajScene-60K, a large-scale dataset of 60,000 video samples with dense point trajectories, addressing the scarcity of high-quality 4D scene data. Based on this, we propose a diffusion-based 4D Scene Trajectory Generator (4D-STraG) to jointly generate geometrically consistent and motion-plausible 4D point trajectories. To leverage single-view priors, we design a depth-guided

*Equal contributions. †Project leader.

motion normalization strategy and a motion-aware module for effective geometry and dynamics integration. We then propose a 4D View Synthesis Module (4D-ViSM) to render videos with arbitrary camera trajectories from 4D point track representations. Experiments show that MoRe4D generates high-quality 4D scenes with multi-view consistency and rich dynamic details from a single image. Code: https://github.com/Zhangyr2022/MoRe4D.

### 1. Introduction

4D scene generation seeks to reconstruct comprehensive spatiotemporal representations capturing both explicit 3D geometry and complex temporal dynamics. Generating such dynamically rich 4D scenes from a single static image remains a fundamental challenge [40], as it requires recovering complete spatiotemporal information from inherently limited 2D observations. Achieving this capabil-

ity would largely benefit applications such as virtual reality, augmented reality [20, 39], and immersive content creation.

Although recent video generation models can produce realistic dynamic content, they generally lack an explicit understanding of 3D structure, leading to inconsistencies across views and failing to capture physically plausible motion. Generate-then-reconstruct methods [35, 49, 58, 61] use powerful video models to synthesize multi-view videos before reconstructing a 4D representation. This paradigm benefits from the high-fidelity and rich dynamics offered by existing video generation models. However, video models struggle to maintain strict geometric consistency across the generated views, leading to significant artifacts and structural collapse during the subsequent 3D reconstruction phase. Consequently, the alternative paradigm, reconstructthen-generate [8, 26, 28, 46, 63], has emerged as another promising direction. By first establishing a static 3D structure, this approach provides a robust geometric foundation for subsequent motion generation. Still, by decoupling static reconstruction from motion generation, they discard the rich dynamic potential latent in the source image. As a result, they are restricted to modeling physically plausible and externally constrained motions (e.g., swinging) and struggle to generate large-scale and self-initiated movements that originate from the scene or objects themselves.

To address these challenges, we propose an advanced reconstruct-then-generate framework that tightly couples motion generation with geometric reconstruction, as illustrated in Figure 1. To achieve joint modeling of motion and geometry, we propose to leverage dense point track as the 4D representation, which is directly predicted from the single input image with an integrated model. To facilitate training of the 3D motion generation model, we first built TrajScene-60K, a large-scale dataset of 60,000 samples with 4D point trajectories, tackling data scarcity of largescale, high-quality 4D scene data with complex dynamics. We then introduce the 4D Scene Trajectory Generator (4DSTraG), a novel diffusion model that, starting from an initial 3D reconstruction, uniquely enables the joint prediction of geometry and motion generation. Unlike prior works that treat these as separate steps, 4D-STraG operationalizes them within a unified denoising process, producing coherent 4D point trajectories whose motion is intrinsically consistent with the evolving 3D structure. To fully leverage priors from the input image, we further incorporate a depth-guided motion normalization method to enhance geometric awareness, and a Motion Perception Module (MPM) to leverage plausible motion priors. Finally, the generated 4D point cloud trajectories are rendered into high-fidelity dynamic videos from arbitrary novel viewpoints using our 4D View Synthesis Module (4D-ViSM), completing the full pipeline from a single image to a coherent 4D scene.

Experimental results of both quantitative and qualita-

tive comparisons demonstrate that our method consistently outperforms existing baselines, producing 4D scenes with more pronounced dynamics, stronger three-dimensional motion consistency, and superior numerical performance. In addition to overall comparisons, we conduct detailed visual ablation studies, which confirm the effectiveness of the key modules in our framework, highlighting how each component contributes to generating coherent and physically plausible 4D motion.

### 2. Related Work

Video Generation. From early VAE [3, 4, 15, 21, 52] and GAN [12, 51, 53] frameworks to modern large-scale diffusion models [23, 30, 48, 50, 54] trained on extensive video data, the realism and resolution of generated videos have been revolutionized. In addition to text control generation, researchers have introduced various guidance strategies for video generation, such as structure-based [37, 60], imagebased [10, 14], and temporal controls [47, 57]. However, most of them are fundamentally limited as they operate in 2D pixel space, lacking explicit 3D scene modeling.

Novel-View Synthesis (NVS). 3D reconstruction-based methods reconstruct a 3D or 4D representation [9, 56, 65, 66], ensuring strong geometric consistency but often requiring costly optimization and suffering from artifacts. Generation-based methods use pretrained video diffusion models conditioned on camera trajectories to synthesize novel views [5, 17, 34, 41, 62]. While leveraging strong generative priors, they often exhibit inconsistencies and object drift under large viewpoint changes or complex scenes.

4D Generation. 4D generation aims to produce temporally evolving 3D representations from texts or sparse images. Existing methods largely follow two decoupled paradigms: generate-then-reconstruct, which first synthesizes videos and then reconstructs 4D (e.g., L4GM [45], 4Real [61], CAT4D [58], DimensionX [49], Free4D [35], SV4D [59]), and reconstruct-then-generate, which builds static 3D assets before animating them (e.g., Animate124 [63], Animate3D [26], GS-DiT [8], Jin et al. [28], 4D-fy [4], Gen3C[46]). While effective, the former suffers from video–geometry mismatch, and the latter is often objectcentric and limited in scene complexity. Concurrently, 4DNeX [11] concatenates RGB-XYZ, in contrast to our method, which advances the reconstruct-then-generate paradigm by jointly modeling motion and geometry in a coupled manner and enables consistent 4D synthesis.

### 3. Dataset Curation – TrajScene-60K

Training our 4D generation model requires a dataset with three components: 4D point trajectories, viewpoint-specific observations, and semantic descriptions of 4D scenes. Given the scarcity of high-quality, large-scale annotated

[Figure 75]

(a) High-Quality Dynamic Video Curation and Description

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Fidelity

A red fox with...... sniffing the ground and foraging in

[Figure 80]

[Figure 81]

[Figure 82]

Dynamics

VLM Captioning

Description Filtering

slow motion....

High Quality

[Figure 83]

- (b) Dense Point Track Extraction
- (c) Quality Filtering and Cleaning (d) Viewpoint-Specific Rendering

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

Accurate Depth

[Figure 100]

[Figure 101]

Coherent Tracks

Plausible Motion

Excluded Sample Rendered Scene

Point Track Inpainting Mask

- Figure 2. TrajScene-60K curation pipeline. We curate videos from WebVid-10M, filtered via VLMs for structured motion and countable entities. Dense 4D point tracks are extracted and refined via depth filtering and Gaussian Splatting, producing 60K high-quality 4D scenes.

### 4. Proposed Approach

data, particularly for scene-level videos with significant motion and complex dynamics, we introduce TrajScene-60K, a comprehensive dataset designed to support robust 4D modeling of dynamic scenes. A detailed comparison with existing datasets is provided in the supplementary material.

##### 4.1. Problem Definition

Given a single input image I ∈ RH×W×3 and its description C, our goal is to reconstruct a physically plausible and temporally consistent 4D scene. We represent the scene as a point cloud sequence P ∈ RT×N×3, capturing the 3D geometry and motion trajectories of N = H × W points over T frames. From this representation, we render dynamic scene videos V ∈ RT×H

As shown in Figure 2, we construct our dataset from the high-quality WebVid-10M corpus [7], extracting about 200,000 samples. To ensure data quality, we design an automated filtering pipeline with large language models: CogVLM2 [24] generates a caption C for each video, and DeepSeek-V3 [33] evaluates these captions to retain videos with countable entities undergoing self-initiated motion, while discarding those dominated by unstructured dynamics (e.g., crowd behaviors, wind-driven oscillations, or background jitter). This process yields samples with wellstructured and quantifiable motion targets, enabling the learning of clear and interpretable dynamic representations.

′×W′×3 from arbitrary novel viewpoints or along arbitrary camera paths to the gap between a static image and full multi-view 4D dynamic generation.

Unlike prior decoupled paradigms that suffer from error accumulation, our approach tightly integrates motion generation with geometric reconstruction. This ensures that the inferred dynamics are not only plausible but also intrinsically consistent with the evolving 3D structure. By cooptimizing structure and motion, we produce a more coherent and stable 4D representation. To systematically achieve these objectives, we first design 4D Scene Trajectory Generator (4D-STraG), a joint diffusion model that simultaneously reconstructs and generates spatiotemporal point trajectories from the initial image. Then we present in detail the 4D View Synthesis Module (4D-ViSM), which leverages the reconstructed 4D representation to enable highquality video generation under arbitrary camera motion. The overview illustration is shown in Figure 3.

To extract 4D dense point track data from videos, we employ the DELTA [42] model. The model takes RGB video sequences V ∈ RT×H×W×3 as input and utilizes monocular depth estimation methods to obtain depth maps D ∈ RT×H×W, subsequently estimating occlusion-aware 4D trajectories P ∈ RT×H×W×4. Each 4D vector pt,u,v = (ut,vt,dt,ot) represents the 3D position and occlusion status in frame t corresponding to the pixel located at (u,v) in the first frame. After obtaining 4D point cloud scene data, we perform quality filtering to remove samples with significant depth estimation errors, anomalous depth values, or missing values, and produce 60,000 high-quality samples. We then render these 4D scenes into videos from the original camera viewpoint using Gaussian Splatting [29]. Inpainting masks are also generated for the void regions.

##### 4.2. 4D Scene Trajectory Generator (4D-STraG)

To leverage the rich motion priors and structural understanding capabilities, we build upon Wan 2.1 [54] and finetune its spatiotemporal VAE and DiT modules separately.

###### RGB Motion Map

Depth-Guided Motion Dense Point Track Normalization

[Figure 102]

[Figure 103]

[Figure 104]

Sec. 4.2: 4D Scene Trajectory Generator

[Figure 105]

[Figure 106]

Training Only Training & Inference

[Figure 107]

[Figure 108]

Trajectory Encoder

[Figure 109]

 =  Objective

[Figure 110]

[Figure 111]

[Figure 112]

###### 4D Scene

[Figure 113]

[Figure 114]

VAEEncoder

[Figure 115]

∆ 

∆ 

∆ 

(

,

,

)

[Figure 116]

VAEDecoder

 = 

 = 

 = 

[Figure 117]

[Figure 118]

Trajectory Decoder

[Figure 119]

[Figure 120]

DiT Blocks

[Figure 121]

Tracking model

###### Depth Estimation

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

t

###### MPM

Any Camera Trajectory

Depth Image

Ground Truth Video

Input Single Image Backprojection

Render

###### Novel-View Videos

###### Input Image

Motion-aware Features

[Figure 132]

DiT Blocks

[Figure 133]

[Figure 134]

Rendered Scene

FFN

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

MAdaNorm

[Figure 146]

2, 2

VAEDecoder

VAEEncoder

[Figure 147]

[Figure 148]

C

Concat

Token-wise Alignment

Cross-Attention

[Figure 149]

[Figure 150]

[Figure 151]

Motion Feature Extrator

[Figure 152]

Normalization

[Figure 153]

DiT Blocks

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Self-Attention MAdaNorm

[Figure 158]

[Figure 159]

1, 1

Inpainting Mask

[Figure 160]

LoRA

Motion Perception Module

Sec. 4.3: 4D View Synthesis Module

x N

- Figure 3. Pipeline of MoRe4D. Top: The 4D Scene Trajectory Generator (Sec. 4.2), a Diffusion Transformer, jointly generates geometry and motion. Bottom-Left: The Motion Perception Module (MPM) identifies potential motion regions and semantic structure from the input image. Bottom-Right: The 4D View Synthesis Module (Sec. 4.3) renders the output into novel-view videos.

Point Trajectory Initialization and Normalization. To enhance training stability and ensure compatibility with generative model scales, we only use the 4D-STraG to predict relative motion ∆Pt = Pt−P0 = {[∆xt,∆yt,∆zt]}, where t ∈ [0,T] and P0 denotes coordinates in the first frame. We further normalize ∆Pt to avoid an unlimited data value range. Based on the observation that a small 3D movement in a nearby object causes a large displacement on the 2D image plane, whereas the same movement in a distant object appears minuscule, we propose a Depth-Guided Motion Normalization Strategy. It normalizes the absolute motion of each point relative to its viewing frustum at the initial depth. By doing so, we transform raw motion into a scale-invariant representation, ensuring perceptual consistency across different distances and producing a more uniform data distribution to learn from.

Specifically, given focal lengths fx,fy and image dimensions W ×H, we introduce the scaling factors αx = fx/W and αy = fy/H. Geometrically, z/αx and z/αy correspond to the width and height of the viewing frustum at depth z, respectively. We normalize motion quantities by

the viewing frustum size at depth z = P(0z):

αy · ∆yt z

αx · ∆xt z

, ∆˜yt =

, ∆˜zt =

∆˜xt =

∆zt z

. (1)

This depth-dependent normalization achieves scale invariance across different depth ranges, enabling our diffusion model to effectively learn motion patterns without being biased by the absolute spatial position of points. During inference, we use UniDepthv2 [44] to estimate depth, ensuring consistency with the DELTA tracking model setup. The relative motion maps are de-normalized and fused with the initial point cloud to form a 4D scene representation.

Model Pipeline. During training, our model takes an image–caption pair as input and learns a diffusion model to predict pixel-level point trajectories across frames. To adapt the generative backbone, we first finetune a motion-sensitive VAE capable of handling trajectory signals. Specifically, the relative point displacement ∆Pt is transformed into an RGB motion map by a lightweight Trajectory Encoder, where spatial movements are represented as color variations while the shape and appearance remain static as the first frame. Correspondingly, a Trajectory Decoder is appended after the VAE decoder, ensuring accurate recovery of point trajectories from the RGB motion map.

After finetuning the VAE, we adapt the Diffusion Transformer (DiT) to handle latents encoded from the RGB motion map. We explicitly inject strong geometric priors into the model, thereby enhancing its generative capability.

Specifically, the depth information from the initial frame is encoded into a latent representation using the VAE encoder. This provides the model with robust structural priors and geometric cues, significantly improving its ability to reason about scene layout and object relationships. The image, noise, and depth latents are concatenated along the feature dimension to form the final input for the DiT:

zcombined = Concat(zimage,znoise,zdepth). (2)

The DiT model is trained using flow matching [32], which learns deterministic flows from noise to data distributions by minimizing the error between predicted and true flow fields, enabling accurate modeling of pixel-level motion. The objective function is defined as:

0,x1 |vθ(t,xt) − (x1 − x0)|2 , (3)

Lfm = Et,x

where x0 and x1 represent the initial and target states, respectively, xt denotes the interpolated state at time t, and vθ is the flow vector predicted by the model.

Motion Perception Module (MPM). A significant challenge lies in adapting the powerful motion priors learned by existing image-to-video diffusion models. These models excel at generating dynamic, moving scenes from a static image. Our objective, however, is to leverage this pretrained temporal knowledge for a distinct task: generating structurally static videos that exhibit dynamic color variations. This creates a core conflict, as the model must learn to translate its ingrained understanding of physical displacement into a new domain of temporal color evolution.

To resolve this conflict and effectively guide the model’s adaptation, we introduce the Motion Perception Module (MPM). The MPM is designed to identify semantic regions within the static scene that are plausible candidates for these color dynamics. We first employ a pretrained motion feature extractor, OmniMAE [18], to derive motion-aware patch-level features S from the input static image. To embed motion information into the diffusion process, we introduce Motion-aware Adaptive Normalization (MAdaNorm). While inspired by conditional injection mechanisms like AdaIN [43], our proposed module is specifically designed for 4D motion generation. Unlike global conditioning approaches, MAdaNorm performs token-wise modulation driven by motion-aware features, enabling fine-grained control over temporal coherence in a geometrically consistent manner. Specifically, after spatial alignment of motion features S by resizing them to match the DiT token sequence length, token-wise adaptive parameters are generated through linear layers. For intermediate features Fit ∈ RN×d in the i-th DiT block, the process is as follows:

α1,α2,β1,β2 = Linear(S), (4) F′ = Attn γ1α1 ⊙ LN(Fit) + γ1β1 , (5) F′′ = MLP(γ2α2 ⊙ LN(F′) + γ2β2), (6)

where α1,α2,β1,β2 ∈ RN×d are token-wise scaling and bias parameters, γ1,γ2 ∈ Rd is a learnable global gating coefficient, and ⊙ denotes token-wise multiplication. Experimental results show that MPM significantly improves the fidelity of motion trajectory modeling and enhances the spatio-temporal consistency of generated 4D content.

- 4.3. 4D View Synthesis Module (4D-ViSM)

After obtaining a dense 4D point cloud representation, we propose the 4D-ViSM to achieve novel view video synthesis along arbitrary camera trajectories. The original point cloud may not fully cover image regions in novel views, which results in holes in the rendered output. We thereby leverage generative models to complete these missing regions, ensuring visual coherence and plausibility.

Considering the excellent performance of Wan2.1 [54] in video generation tasks, we also choose to finetune this model to build our 4D-ViSM. The training data also come from our TrajScene-60K dataset, including rendered videos, corresponding occlusion masks, ground truth videos, and captions. During training, we follow the Wan2.1 mask processing strategy, setting the mask value to 0.5 for regions without projected points in each frame. Through finetuning, our model achieves high-quality and visually consistent novel view video synthesis.

- 5. Experiments

- 5.1. Experimental Setup

Implementation Details. Trained on our TrajScene-60K dataset, our model generates videos at a resolution of 512× 368 with a length of 49 frames. For the 4D-STraG module, we performed full-parameter training based on Wan2.114B [1, 54]. The trajectory encoder and decoder are both shallow ResNets. We first trained its tracking components and VAE decoder for 5k steps, then its DiT for 2k steps using OmniMAE features for motion conditioning. The 4DViSM module finetuned Wan2.1-14B [2, 54] with LoRA for 10k steps. We used AdamW [36] with a learning rate of 2×10−5. All experiments ran on four NVIDIA H20 GPUs.

- 5.2. Quantitative Results

While 4D geometric metrics are valuable for evaluating 4D consistency, their high computational cost and incompatibility with many image-to-4D methods make them prohibitive for the main comparison. Following prior 4D generation works, we thus primarily use VBench [25] for a fair and extensive video-quality comparison against a wider range of baselines. A more detailed 4D geometric analysis is provided in the supplementary material.

We compare it with leading 4D generation methods: 4Real, GenXD, DimensionX, Gen3C, and Free4D. Following the evaluation protocol in Free4D [35], we structured

- Table 1. Quantitative comparison on VBench. Higher values are better. The best results in each comparison group are marked in bold.

Exp.

Subject Background Motion Dynamic Aesthetic Imaging No. Consistency Consistency Smoothness Degree Quality Quality

Model / Metrics

- I

4Real [61] 0.9329 0.9709 0.9664 0.7708 0.4938 0.5095 MoRe4D(Ours) 0.8752 0.9364 0.9682 1.0000 0.5613 0.6230

- II

GenXD [64] 0.8042 0.8789 0.9030 1.0000 0.4077 0.5209 DimensionX [49] 0.7553 0.8481 0.9827 1.0000 0.4634 0.5545 MoRe4D(Ours) 0.8241 0.9044 0.9760 0.9500 0.4820 0.5828

- III

Free4D [35] 0.7899 0.8883 0.9797 1.0000 0.3607 0.3562 Gen3C [46] 0.8112 0.8871 0.9845 0.9940 0.3812 0.4814 MoRe4D(Ours) 0.8339 0.9065 0.9773 0.9000 0.4820 0.5939

###### Input

Generated Video

|[Figure 161]|[Figure 162]|[Figure 163]|[Figure 164]|[Figure 165]|
|---|---|---|---|---|

[Figure 166]

A Bactrian camel strides across the sandy terrain.

|[Figure 167]|[Figure 168]|[Figure 169]|[Figure 170]|[Figure 171]|
|---|---|---|---|---|

[Figure 172]

Trajectory 1

|[Figure 173]|[Figure 174]|[Figure 175]|[Figure 176]|[Figure 177]|
|---|---|---|---|---|

Trajectory 2

[Figure 178]

|[Figure 179]|[Figure 180]|[Figure 181]|[Figure 182]| |[Figure 183]|
|---|---|---|---|---|---|

[Figure 184]

A rhino takes small steps in the woods.

[Figure 185]

|[Figure 186]|[Figure 187]|[Figure 188]| |[Figure 189]<br><br>[Figure 190]| |
|---|---|---|---|---|---|

Trajectory 1

Trajectory 2

[Figure 191]

|[Figure 192]| |[Figure 193]|[Figure 194]|[Figure 195]|[Figure 196]|
|---|---|---|---|---|---|

- Figure 4. Qualitative results of our model. The first row shows the 4D point cloud generated by our 4D-STraG. The second and third rows show the videos rendered by our 4D-ViSM under two distinct, user-defined camera trajectories.

our comparisons into three groups based on model availability and technical constraints, with each group evaluated under appropriate camera trajectories. In Group I, we compared against the closed-source 4Real using simple trajectories, as we could only access their official demonstration videos for evaluation. In Group II, we benchmarked against

single-image 3D reconstruction models (GenXD and the open-source S-Director from DimensionX) using moderate trajectories (90° leftward rotations around the center), constrained by their limited trajectory support in open-source implementations. For Group III, we evaluated against other 4D generation models using complex trajectories, includ-

###### Input Generated Videos

|[Figure 197]|[Figure 198]|[Figure 199]|[Figure 200]|[Figure 201]|
|---|---|---|---|---|

[Figure 202]

4Real

|[Figure 203]|[Figure 204]|[Figure 205]|[Figure 206]|[Figure 207]|
|---|---|---|---|---|

A humanoid robot skillfully plays violin.

Ours

|[Figure 208]|[Figure 209]|[Figure 210]|[Figure 211]|[Figure 212]|
|---|---|---|---|---|

[Figure 213]

DimensionX

A mixologist crafts a cocktail at home.

|[Figure 214]|[Figure 215]|[Figure 216]|[Figure 217]|[Figure 218]|
|---|---|---|---|---|

Ours

|[Figure 219]|[Figure 220]|[Figure 221]|[Figure 222]|[Figure 223]|
|---|---|---|---|---|

[Figure 224]

Gen3C

A surfer conquers a towering, dark wave.

|[Figure 225]|[Figure 226]|[Figure 227]|[Figure 228]|[Figure 229]|
|---|---|---|---|---|

Ours

|[Figure 230]|[Figure 231]| | |[Figure 232]| |[Figure 233]|[Figure 234]|
|---|---|---|---|---|---|---|---|

[Figure 235]

Free4D

|[Figure 236]|[Figure 237]|[Figure 238]|[Figure 239]|[Figure 240]|
|---|---|---|---|---|

A couple rests intimately under patterned bedding.

Ours

- Figure 5. Qualitative comparison with baseline methods. For each sample, the first row shows the baseline results while the second row presents our MoRe4D results. The first column displays the input image and text prompt.

ing upward, forward, leftward, rightward, and downward movements, to assess performance under challenging camera motions. For Groups II and III, we used 200 held-out samples from WebVid-10M that do not overlap with our TrajScene-60k training subset to ensure fair evaluation in diverse scenarios. As shown in Table 1, MoRe4D consistently achieved promising results across groups. Notably, it demonstrated superior Dynamics, Aesthetic, and Imaging Quality compared to 4Real, outperformed 3D reconstruction baselines in consistency and visual quality, and maintained a pronounced lead in Aesthetic and Imaging Quality against 4D generation models under complex trajectories. Furthermore, consistent improvements across all groups demonstrate our robust generalization.

##### 5.3. Qualitative Results

For qualitative validation of our model effectiveness, we conduct comparative analyses in Figure B and Figure 5. Figure B presents two representative scenes, with the first row displaying the point cloud outputs of 4D-STraG. It can be observed that the 4D point clouds maintain structural consistency and exhibit reasonable motion over time, with high detail completeness. For each scene, we define two camera trajectories and perform rendering using 4D-ViSM. The results demonstrate that the rendered images not only preserve the geometric accuracy of the original structure but also effectively align with the camera trajectories, reflecting strong visual consistency. Figure 5 further compares the vi-

###### Input 4D Point Cloud from 4D-STraG Generated Video from 4D-ViSM

|[Figure 241]|
|---|
|[Figure 242]|

|[Figure 243]|
|---|
|[Figure 244]|

|[Figure 245]|
|---|
|[Figure 246]|

[Figure 247]

|[Figure 248]|
|---|
|[Figure 249]|

|[Figure 250]|
|---|
|[Figure 251]|

|[Figure 252]|
|---|
|[Figure 253]|

W/o DepthGuided Norm

Breakdancer performs dynamic moves for an urban crowd.

MoRe4D

Generated Video from 4D-ViSM

Point Trajectory from 4D-STraG

[Figure 254]

|[Figure 255]|
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

W/o MPM

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

|[Figure 264]|
|---|

|[Figure 265]|
|---|

|[Figure 266]|
|---|

A hiker strides forward on a mountain path.

MoRe4D

|[Figure 267]|
|---|

|[Figure 268]|
|---|

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|

[Figure 273]

W/o Depth Latents

|[Figure 274]|
|---|

|[Figure 275]|
|---|

|[Figure 276]|
|---|

|[Figure 277]|
|---|

|[Figure 278]|
|---|

|[Figure 279]|
|---|

A joyful girl dances with a spinning skirt.

MoRe4D

- Figure 6. Ablation studies on normalization methods and module components. (Rows 1-2) Depth-guided motion normalization stabilizes 4D point cloud generation. (Rows 3-6) Removing the MPM module reduces motion magnitude while excluding depth guidance breaks structural motion consistency, validating our design choices.

- Table 2. Ablation Study. We evaluate the contribution of each component. The best results are marked in bold.

ally demonstrate these findings. Baseline min-max normalization (rows 1-2) yields unstable point clouds with exaggerated movements, particularly in scenes with large depth variations. Our depth-guided approach effectively resolves this issue. Further visualizations (rows 3-6) show that without MPM or depth guidance, the model fails to preserve object structure or maintain motion consistency across different parts, confirming the critical role of depth in ensuring structurally coherent point trajectories.

Variant Consistency↑ Dynamic↑ Aesthetic↑

w/o Depth-Guided Norm 0.8604 0.8850 0.4672 w/o MPM 0.8650 0.8500 0.4806 w/o Depth Latents 0.8567 0.8500 0.4738 MoRe4D 0.8702 0.9000 0.4820

sual outcomes of several existing methods, including 4Real, DimensionX, Gen3C, and Free4D. In comparison to these approaches, our method generates more diverse and realistic motion, validating the advantage of the “joint reconstruction and generation” strategy in ensuring both structural rationality and motion plausibility for 4D generation.

### 6. Conclusion

We proposed MoRe4D, a unified framework for singleimage 4D generation that tightly couples motion and geometry, overcoming the inherent weaknesses of decoupled paradigms. To support training, we constructed TrajScene60K, a large-scale dataset with dense 4D trajectories, and introduced the 4D-STraG diffusion model with depthguided motion normalization and motion perception priors for joint inference. Experiments show that MoRe4D achieves superior geometric consistency, dynamic realism, and visual fidelity compared to existing approaches. This work establishes a new technical pathway for 4D synthesis from minimal input, and points toward future exploration of more universal dynamic priors and lightweight 4D representations for practical deployment.

##### 5.4. Ablation Studies

Table 2 provides quantitative analysis of each component’s contribution on VBench under the Group III setting of Table 1. Removing MPM reduces the Dynamic score from 0.9 to 0.85, confirming its necessity for inferring motion magnitude. Excluding depth latents degrades both Consistency and Dynamic scores, as the model struggles to maintain motion coherence. Without depth-guided normalization, Consistency drops to 0.8604 and Aesthetic quality decreases to 0.4672. Our full MoRe4D latents achieve optimal performance across all metrics, validating the synergistic contribution of each component. Row 1-6 in Figure 6 visu-

Table A. Quantitative evaluation on 4D Generation Consistency via VLM-based assessment. Higher values are better. The best results in each comparison group are marked in bold.

Exp.

3D Geometric Temporal Texture Subject Identity Motion Geometry Background Stability Average No. Consistency Stability Preservation Coupling Consistency Score

Model / Metrics

- I

4Real [61] 3.04 3.49 4.11 2.64 3.49 3.35 MoRe4D (Ours) 3.46 3.88 4.42 3.25 3.96 3.80

- II

GenXD [64] 2.43 2.66 3.09 2.38 2.65 2.64 DimensionX [49] 2.58 2.74 3.35 2.53 2.79 2.80 MoRe4D (Ours) 3.53 3.92 4.33 3.45 3.93 3.83

- III

Free4D [35] 1.13 1.30 1.37 1.17 1.17 1.23 Gen3C [46] 1.99 2.26 2.54 2.01 2.31 2.22 MoRe4D (Ours) 3.47 3.88 4.25 3.35 3.85 3.76

### Appendix

In this Appendix, we provide:

- • Additional experiment analysis (Section A).
- • Comprehensive ablation studies (Section B).
- • Extended visualization results (Section C).
- • Dataset curation and comparison details (Section D).
- • Experimental settings (Section E).
- • Implementation details (Section F).
- • Discussion on applications and future work (Section G). A. More Experiment Analysis

##### A.1. 4D Consistency Analysis

While VBench offers valuable insights into general video quality, such as consistency, smoothness, and aesthetics, it does not directly penalize 3D-consistency artifacts, such as geometric drift or texture swimming, which are critical challenges in single-view 4D generation. We acknowledge that established 4D metrics exist for tasks like point cloud tracking, including 3D Average Jaccard (AJ), Average 3D Position Accuracy (APD3D), and 3D endpoint error (EPE). However, these metrics presuppose the existence of a single ground-truth trajectory to measure against. Our image-to4D task is fundamentally different and ill-posed for such direct error calculation, as a static image can plausibly evolve into an infinite number of possible dynamic sequences.

To address this evaluation gap, we leverage a VisionLanguage Model (VLM) for systematic quality assessment. We uniformly sample 8 frames from each video generated under the settings of Table 1 in the main paper and feed them to Qwen2.5-VL-72B-Instruct [6], which rates the sequence on a 1-5 scale across five critical dimensions:

- • 3D Geometric Consistency: assesses 3D structure coherence during camera motion.
- • Temporal Texture Stability: evaluates texture stability without flickering or sliding.
- • Subject Identity Preservation: ensures the main subject retains identity and shape.

Table B. Comparative runtime analysis of 4D generation methods. All timings of open-source methods are measured on single NVIDIA A100 GPU averaged over 100 samples.

Method Year Resolution Frames Time

4Dfy [4] CVPR’24 256 × 256 − 10h 4Real [61] NeurIPS’24 256 × 144 8 1.5h Gen3C [46] CVPR’25 1280 × 704 121 50min GenXD [64] ICLR’25 256 × 256 12 2min Free4D [35] ICCV’25 512 × 368 16 30min

MoRe4D − 512 × 368 49 6min

- • Motion-Geometry Coupling: checks motion alignment with 3D geometry and physics.
- • Background Stability: measures background static behavior without distortion.

Table A shows that our method consistently achieves higher scores across all metrics and comparison groups. The significant lead in Motion-Geometry Coupling validates the ability to maintain structural integrity during motion. Additionally, MoRe4D demonstrates strong performance in 3D Geometric Consistency and Temporal Texture Stability, indicating robust spatial coherence and temporal stability. The consistent superiority across all three experiment groups further confirms the generalizability of our method. This comprehensive VLM-based evaluation confirms the superior 4D consistency of our approach, capturing nuanced perceptual quality aspects that current automated metrics fail to measure.

##### A.2. Runtime Analysis

As shown in Table B, we compare runtime performance with existing 4D generation methods on a single NVIDIA A100 GPU (averaged over 100 samples). DimensionX [49] is excluded due to unavailable runtime details and limited code. Among other baselines, only 4Real [61] reports inference time; others are evaluated using official implementations. Our method achieves competitive efficiency: both

###### Input Method

###### 4D Represnetation

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

MoRe4D (Ours)

A coastal rescue: multiple rescue boats racing, people jumping into the water, helicopter lowering.

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

Wan2.1-I2V Generation

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

+ DELTA Tracking

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

+ VGGT Reconstruction

###### Input Method 4D Represnetation

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

MoRe4D (Ours)

An outdoor music festival crowd doing synchronized jumping and crowd-surfing under strobe lights.

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

Wan2.1-I2V Generation

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

+ DELTA Tracking

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

+ VGGT Reconstruction

- Figure A. Two examples comparing our joint 4D-STraG framework (top row) against sequential pipelines. For each sample, rows 2-4 show results from a cascaded approach: Wan2.1-I2V video generation, followed by DELTA tracking or VGGT reconstruction. Our method yields superior spatio-temporal coherence, while sequential approaches exhibit fragmentation from error accumulation. All samples are consistently rendered from the fixed camera viewpoint for fair comparison.

4D-STraG and 4D-ViSM complete generation in 3 minutes each. This is notable given our higher resolution (512×368) and longer sequences (49 frames) versus most baselines. As Table B shows, our approach balances quality and efficiency effectively. While GenXD [64] is faster (2 minutes), it uses lower resolution (256×256) and shorter sequences (12 frames). Free4D [35], at similar resolution, requires 30 minutes, which is five times longer than ours.

### B. More Ablation Analysis

##### B.1. Joint vs. Sequential 4D Generation

To validate the effectiveness of our joint generation paradigm, we conduct an ablation study against alternative sequential pipelines that decompose the task into two stages: first generating a video from the source image, then performing 3D reconstruction or tracking as post-

#### Input Generated Video

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

A brown bear walking across rocky terrain.

- Trajectory 1
- Trajectory 2
- Trajectory 3

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

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

A windsurfer glides across the vast blue ocean.

- Trajectory 1
- Trajectory 2
- Trajectory 3

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

- Figure B. Qualitative results of our model. We visualize the generated results for two more samples. The first row shows the 4D point cloud generated by our 4D-STraG. The second, third and fourth rows show the videos rendered by our 4D-ViSM under three distinct, user-defined camera trajectories.

processing to obtain 4D representations. Specifically, we construct two baseline pipelines: (1) video generation using Wan2.1-I2V-14B [54] followed by DELTA [42], the same dense point tracking model used in our dataset construction pipeline; and (2) the same video generation followed by VGGT[55], a state-of-the-art feed-forward 3D transformer

designed for robust 4D scene reconstruction.

As illustrated in Figure A, where results are visualized under a fixed rendering viewpoint to reveal geometric consistency, our joint framework demonstrates substantially superior spatio-temporal coherence. The generated point clouds maintain structurally coherent geometry and physi-

Table C. Ablation study on MAdaNorm. We evaluate the contribution of each component. The best results are marked in bold.

Variant Consistency↑ Dynamic↑ Aesthetic↑

w/o MPM 0.8650 0.8500 0.4806 w/o Patch-level Features 0.8743 0.8840 0.4754 MoRe4D 0.8702 0.9000 0.4820

cally plausible motion throughout the sequence. In contrast, sequential pipelines exhibit notable failure modes: stationary background regions exhibit spurious motion (e.g., drifting water and crowds), while dynamic foreground objects suffer from geometric fragmentation and discontinuous trajectories (e.g., unrealistic body part detachment). These artifacts stem from spatial inconsistencies introduced during video generation—since standard video diffusion models like Wan-I2V lack explicit 3D geometric constraints, the synthesized videos contain frame-to-frame appearance variations, motion blur, and perspective inconsistencies that violate rigid scene structure. When fed into subsequent reconstruction or tracking modules, these visual inconsistencies are incorrectly interpreted as 3D motion or deformation, causing severe error accumulation.

Our method fundamentally mitigates these issues by unifying geometry, appearance, and trajectory generation within a single diffusion backbone. This enables bidirectional mutual regularization: geometric constraints guide video synthesis toward spatial consistency, while visual features inform motion prediction for temporal coherence. The diffusion process learns to co-evolve appearance and geometry jointly, ensuring every frame is both visually plausible and geometrically self-consistent under the fixed viewpoint. This prevents the error accumulation plaguing sequential approaches, ensuring robust and high-fidelity 4D content creation with coherent structure, realistic appearance, and accurate motion dynamics.

##### B.2. MAdaNorm Module

To further analyze the MAdaNorm module, we conducted additional ablation studies in Table C under the same settings of Table 2 in the main paper. We evaluate a “w/o Patch-level Features” variant, which replaces our Motion Perception Module (MPM) with a simpler conditioning mechanism using only OmniMAE [18] global ‘[CLS]’ token. While this simpler design slightly improves consistency, it degrades both dynamic and aesthetic quality, suggesting a global feature alone is insufficient for capturing the fine-grained details needed for high-quality 4D generation. In contrast, our full MoRe4D model, leveraging MPM to process rich patch-level features, achieves a superior balance. It significantly outperforms the ablated variants in dynamic and aesthetic scores (0.9000 and 0.4820, respectively). This highlights that aggregating multi-view patch features is crucial for generating temporally coherent, dynamic, and aesthetically pleasing 4D content.

### C. More Visualization Results

- C.1. Qualitative Results of MoRe4D

- Figure B presents two samples of the visualizations for MoRe4D. They are evaluated under distinct user-defined camera trajectories: up-down movement, top-down viewing, forward-backward movement, and left-hand circling. These trajectories are designed to provide comprehensive multi-perspective observations. The rendered videos demonstrate that our approach consistently produces temporally coherent motion and maintains high-fidelity visual details across all viewpoints. The high-quality 4D point clouds produced by 4D-STraG provide a robust geometric foundation for rendering. The results further demonstrate the strong capability of our 4D-ViSM renderer in synthesizing consistent, high-fidelity videos under diverse camera paths, confirming its flexibility and generalization.

C.2. Qualitative Results of Motion-Sensitive VAE

- Figure C shows the reconstruction results using the finetuned VAE on a subset of samples during inference. The reconstructed points are rendered from the original camera viewpoint, demonstrating that they effectively preserve the structural and motion consistency with the input. More specifically, the model exhibits robust representation learning capabilities, enabling effective extraction of latent space representations that accurately capture both geometric and dynamic properties of the observed scenes.
- D. Dataset Curation

- D.1. Video Filtering and Annotation

We initiated our process with approximately 200,000 video candidates from the WebVid-10M [7] dataset. To automate the selection of content suitable for 4D dynamic scene modeling, we implemented a two-stage pipeline. First, we employed the multimodal large language model CogVLM2 [24] to generate a detailed English caption for each video. Subsequently, these captions were fed into the DeepSeek-V3 [33] model, which used a carefully designed prompt to assess content suitability. The core evaluation criteria were: (1) the presence of one or more clearly countable entities, and (2) the exhibition of self-initiated, non-rigid, or articulated motion, as opposed to random movements driven by external forces (e.g., wind, water) or dominant camera motion. This process effectively filtered out videos featuring unstructured dynamics, such as crowd movements, water ripples, or swaying foliage. Our prompt is shown in Figure D.

- D.2. 4D Trajectory Quality Control

After extracting the raw 4D trajectories, we applied a strict quality filtering process to eliminate samples compromised

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Ground Truth

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

Rendered Output

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

Ground Truth

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

Rendered Output

- Figure C. Qualitative results of Motion-Sensitive VAE. The figure presents two examples. For each, the top row shows pseudo-GT from the original camera view, while the bottom is the corresponding VAE rendering at the same pose. The close visual alignment demonstrates our VAE’s high-fidelity reconstruction capability.

[Figure 390]

Identify whether the caption describes countable moving entities. An entity is considered moving if:

- 1) it moves by itself,
- 2) part of it moves, or
- 3) it appears to move due to camera movement. However, uncountable phenomena (like waterfalls, rivers, particles, light beams, text animations, crowds too numerous to count) do NOT qualify as countable moving entities. Let me think through this step by step:

- 1. What entities are described in the caption?
- 2. Is any entity or part of an entity moving (or does it appear to move due to camera movement)?
- 3. Can these moving entities be counted, or are they uncountable phenomena?
- 4. Based on this analysis, determine if there are countable moving entities.

Your conclusion: After analyzing the caption step by step, I determine the answer is {result: <flag>}, where <flag> is true or false.

Figure D. Our input prompt used to query the model for identifying and counting self-initiated and articulated motion.

Table D. Comprehensive comparison of 3D scene understanding datasets. Our TrajScene-60K dataset significantly surpasses existing benchmarks in scale, providing orders of magnitude more frames and 3D point annotations, while offering dense tracking and depth information across diverse real-world indoor and outdoor environments.

Dataset Year Frames Resolution 3D Points Dense Track Depth Caption Description

FlyingThings3D [38] CVPR’15 2.6 × 104 960 × 540 3 × 108 ✓ ✓ × Synthetic, object-level ScanNet [13] CVPR’17 2.5 × 106 1296 × 968 0 × ✓ × Real, indoor scenes Kubric [19] CVPR’22 Flexible 256 × 256 Flexible ✓ × × Synthetic, object-level TAPVid [16] NeurIPS’22 3 × 105 Multiple 0 ✓ × × Real + synthetic videos TAPVid-3D [31] NeurIPS’24 5 × 105 Multiple 1 × 106 × ✓ × Real-world scenes

TrajScene-60K - 3 × 106 596 × 336 1.2 × 1010 ✓ ✓ ✓ Real, indoor/outdoor

by depth estimation failures or extreme motion. The specific criteria were as follows: (1) We removed samples where a significant portion of point cloud trajectories contained invalid or anomalous depth values (e.g., near-infinite or zero) at any timestep. (2) We discarded samples exhibiting an excessively large standard deviation in scene depth, which indicates potential errors in the global depth estimation. (3) We performed a scale consistency check. Since uniformly scaling a point cloud should yield an identical rendering from the original camera perspective, we removed samples where this transformation resulted in significant visual changes, flagging them as geometrically inconsistent. This comprehensive filtering pipeline yielded the final 60,000 high-quality samples that constitute the TrajScene-60K dataset.

worlds.

##### D.4. Dataset Bias and Release

TrajScene-60K is built from WebVid-10M [7] using LLM/VLM-based caption filtering, and therefore inherits both the demographic and content biases of the source videos and the filtering models. For example, certain object categories, body types, or geographic regions may be under-represented, and the motion statistics may be skewed towards popular online content. While our work focuses on the technical aspects of 4D scene generation, we acknowledge these biases and will release the dataset and filtering prompts to facilitate transparency and future audits. We also encourage downstream users to carefully consider fairness and representational biases when deploying models trained on TrajScene-60K.

##### D.3. Comparison with Previous Datasets

Our TrajScene-60K dataset fills a critical gap in the landscape of resources for 4D dynamic scene understanding and generation. As summarized in Table D, existing datasets are often limited in scale, realism, or the richness of provided annotations. For instance, while synthetic datasets like FlyingThings3D [38] and Kubric [19] offer precise groundtruth motion and geometry, their domain gap with realworld imagery hinders generalization. Real-world datasets like ScanNet [13] provide dense 3D reconstructions but are static and lack temporal dynamics. Recent video tracking benchmarks like TAPVid [16] and TAPVid-3D [31] offer real-world point trajectories but are orders of magnitude smaller in scale and do not provide semantic descriptions crucial for conditional generation.

### E. More Experimental Settings E.1. Baseline Introduction

We compare our method with the following leading methods focusing on 4D generation from a single image.

4Real: [61] generates photorealistic 4D scenes using video diffusion models trained on real data. It first creates a reference video, then learns its 3D canonical representation and temporal deformations, avoiding synthetic 3D priors.

GenXD: [64] introduces a real-world 4D dataset and multiview temporal modules to disentangle camera and object motion, enabling joint learning from 3D/4D data for arbitrary scene generation.

Gen3C: [46] guides video generation via a 3D cache from predicted point clouds, ensuring precise camera control and 3D consistency by focusing generation on unobserved regions.

TrajScene-60K distinguishes itself by combining largescale, real-world video content with dense, occlusion-aware 4D trajectories and high-quality text captions. It provides over 3 million frames and approximately 12 billion 3D point annotations, significantly surpassing predecessors in volume. Crucially, we provide dense 4D tracking, perframe depth, and language descriptions for dynamic scenes across diverse indoor and outdoor environments. This unique combination of scale, realism, and multi-modal annotation makes it an enabling resource for training and evaluating complex 4D scene generation or tracking models, facilitating a more holistic understanding of dynamic 3D

DimensionX: [49] reconstructs 3D/4D scenes from a single image by decoupling spatio-temporal factors in a controllable video diffusion framework, allowing precise manipulation of structure and dynamics.

Free4D: [35] proposes a tuning-free framework that distills foundation models to generate consistent 4D scenes from one image, enabling strong generalization without expensive training.

Among these, 4Real, GenXD, DimensionX, and Free4D follow a generate-then-reconstruct pipeline, while Gen3C adopts a reconstruct-then-generate strategy.

##### E.2. VBench Metrics

In the main paper, we selected VBench [25] for quantitative evaluation as it offers a standardized, efficient framework for assessing perceptual video quality across diverse baselines, whereas 4D geometric metrics are computationally intensive and often incompatible with many baseline methods. VBench comprehensively evaluates six critical dimensions:

- • Subject Consistency: Measures identity and appearance coherence of the primary subject over time.
- • Background Consistency: Assesses temporal stability and coherence of background elements.
- • Motion Smoothness: Evaluates the naturalness and fluidity of object and camera motion.
- • Dynamic Degree: Quantifies the intensity and extent of dynamic changes in the scene.
- • Aesthetic Quality: Judges overall visual appeal, composition, and stylistic merit.
- • Imaging Quality: Rates technical image attributes like sharpness, noise, and artifacts.

This multi-dimensional analysis ensures a fair and holistic comparison of performance on 4D generation from a single image.

### F. Implementation Details F.1. Motion-Sensitive VAE Architecture

Before training the DiT in the 4D-STraG module, we finetune a specialized, motion-sensitive VAE to effectively adapt our generative backbone for trajectory synthesis. This VAE is designed to process and reconstruct trajectory information encoded as RGB-like motion maps. Inspired by Geo4D [27], the architecture of both the VAE encoder and decoder is intentionally kept shallow to preserve finegrained motion details. The Trajectory Encoder and Decoder that process these motion maps are constructed as shallow ResNets [22]. This minimalist design ensures that the VAE learns a compact and efficient latent space for motion patterns without aggressively downsampling the spatial features, which is critical for the precise reconstruction of trajectories by the Trajectory Decoder. When finetuning Motion-Sensitive VAE, we trained the Trajectory Encoder, Trajectory Decoder, and the VAE Decoder, while freezing the VAE Encoder. This approach allows the model to fully adapt to the motion input while preserving the integrity of the pre-trained visual representations.

##### F.2. Details of 4D-ViSM Architecture

Our 4D-ViSM model adapts a pre-trained video Diffusion Transformer (DiT) for dynamic video inpainting, specifically to fill holes in novel-view videos rendered from our 4D representation. During fine-tuning, the model is conditioned on the incomplete video. For each step, the rendered video with holes is encoded into a latent representation zrendered, and its binary occlusion mask is downsampled to mlatent. The denoising network’s input is a channel-wise concatenation of the noisy latent zt, zrendered, and mlatent, formulated as

z = concat([zt,zrendered,mlatent]).

This explicitly provides the model with the known visual context and the missing regions, enabling coherent video completion.

##### F.3. Inference Pipeline

Given a single image and a text prompt, our model first acquires geometric information by estimating depth through the 4D-STraG module. Specifically, we utilize the UniDepthv2 [44] model to infer depth information, ensuring consistency with the estimation method used for the DELTA tracking model during training. The VAE then encodes the image and depth map into a latent space, while MPM extracts motion features. Subsequently, DiT generates latent representations, which are decoded into relative motion latents. These latents are de-normalized by reversing our depth-guided motion normalization strategy and then fused with the initial point cloud’s spatial coordinates to construct a 4D scene representation. Finally, this representation allows for rendering from arbitrary camera poses, and our 4D-ViSM synthesizes a spatio-temporally consistent 4D video aligned with the desired camera trajectory.

### G. Applications and Future Work

Our image-to-4D generation framework opens up numerous possibilities across various domains. It can serve as a powerful tool for digital content creation, enabling artists and designers to bring static images to life as dynamic 3D assets for films, games, and advertising. Furthermore, this technology holds significant potential for populating virtual and augmented reality (VR/AR) environments with dynamic objects, creating more immersive and interactive digital experiences.

Looking ahead, we identify two primary directions for future research. First, we plan to explore more deeply unified architectures that further entangle motion and appearance generation, potentially within a single, non-autoregressive model, to achieve even greater spatio-temporal consistency. Second, a critical challenge in this field is the lack of standardized evaluation protocols. We aim to develop novel metrics specifically

designed to quantify the 4D consistency of generative models, providing a more reliable and automated way to measure geometric and textural stability over time.

### References

- [1] Alibaba PAI Team. Wan2.1-Fun-14B-Control. https:// huggingface.co/alibaba-pai/Wan2.1-Fun14B-Control, 2024. 5
- [2] Alibaba PAI Team. Wan2.1-Fun-14B-InP. https:// huggingface.co/alibaba-pai/Wan2.1-Fun14B-InP, 2024. 5
- [3] Mohammad Babaeizadeh, Mohammad Taghi Saffar, Suraj Nair, Sergey Levine, Chelsea Finn, and Dumitru Erhan. Fitvid: Overfitting in pixel-level video prediction. arXiv preprint arXiv:2106.13195, 2021. 2
- [4] Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4D-fy: Text-to-4d generation using hybrid score distillation sampling. In CVPR, pages 7996–8006, 2024. 2
- [5] Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. arXiv preprint arXiv:2503.11647, 2025. 2
- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2
- [7] Max Bain, Arsha Nagrani, G¨ul Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In ICCV, pages 1728–1738, 2021. 3, 5, 7
- [8] Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yijin Li, FuYun Wang, and Hongsheng Li. Gs-dit: Advancing video generation with pseudo 4d gaussian fields through efficient dense 3d point tracking. arXiv preprint arXiv:2501.02690,

2025. 2

- [9] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, pages 19457–19467, 2024. 2
- [10] Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, et al. Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512, 2023. 2
- [11] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4dnex: Feed-forward 4d generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025. 2
- [12] Aidan Clark, Jeff Donahue, and Karen Simonyan. Adversarial video generation on complex datasets. arXiv preprint arXiv:1907.06571, 2019. 2
- [13] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet:

- Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839, 2017. 7
- [14] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169,

2024. 2

- [15] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In ICML, pages 1174–1183. PMLR,

2018. 2

- [16] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. NeurIPS, 35:13610–13626, 2022. 7
- [17] Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314, 2024. 2
- [18] Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Omnimae: Single model masked pretraining on images and videos. In CVPR, pages 10406–10417, 2023. 5
- [19] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In CVPR, pages 3749–3761,

2022. 7

- [20] Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, and Otmar Hilliges. Vid2avatar: 3d avatar reconstruction from videos in the wild via self-supervised scene decomposition. In CVPR, pages 12858–12868, 2023. 2
- [21] Jiawei He, Andreas Lehrmann, Joseph Marino, Greg Mori, and Leonid Sigal. Probabilistic video generation using holistic attribute control. In ECCV, pages 452–467, 2018. 2
- [22] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016. 8
- [23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. NeurIPS, 35:8633–8646, 2022. 2
- [24] Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024. 3, 5
- [25] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, pages 21807–21818, 2024. 5, 8
- [26] Yanqin Jiang, Chaohui Yu, Chenjie Cao, Fan Wang, Weiming Hu, and Jin Gao. Animate3d: Animating any 3d model with multi-view video diffusion. NeurIPS, 37:125879– 125906, 2024. 2

- [27] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. arXiv preprint arXiv:2504.07961, 2025. 8
- [28] In-Hwan Jin, Haesoo Choo, Seong-Hun Jeong, Park Heemoon, Junghwan Kim, Oh-joon Kwon, and Kyeongbo Kong. Optimizing 4d gaussians for dynamic scene video from single landscape images. In ICLR, 2025. 2
- [29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 3

- [30] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In ICCV, pages 15954–15964, 2023. 2
- [31] Skanda Koppula, Ignacio Rocco, Yi Yang, Joe Heyward, Joao Carreira, Andrew Zisserman, Gabriel Brostow, and Carl Doersch. Tapvid-3d: A benchmark for tracking any point in 3d. NeurIPS, 37:82149–82165, 2024. 7
- [32] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023. 5
- [33] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 3, 5
- [34] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. arXiv preprint arXiv:2408.16767, 2024. 2
- [35] Tianqi Liu, Zihao Huang, Zhaoxi Chen, Guangcong Wang, Shoukang Hu, Liao Shen, Huiqiang Sun, Zhiguo Cao, Wei Li, and Ziwei Liu. Free4d: Tuning-free 4d scene generation with spatial-temporal consistency. arXiv preprint arXiv:2503.20785, 2025. 2, 5, 6, 3, 7
- [36] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR. 5
- [37] Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Poseguided text-to-video generation using pose-free videos. In AAAI, pages 4117–4125, 2024. 2
- [38] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer, Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In CVPR, pages 4040–4048, 2016. 7
- [39] Qiaowei Miao, Jinsheng Quan, Kehan Li, and Yawei Luo. Pla4d: Pixel-level alignments for text-to-4d gaussian splatting. arXiv preprint arXiv:2405.19957, 2024. 2
- [40] Qiaowei Miao, Kehan Li, Jinsheng Quan, Zhiyuan Min, Shaojie Ma, Yichao Xu, Yi Yang, Ping Liu, and Yawei Luo. Advances in 4d generation: A survey. arXiv preprint arXiv:2503.14501, 2025. 1
- [41] Norman M¨uller, Katja Schwarz, Barbara R¨ossle, Lorenzo Porzi, Samuel Rota Bulo, Matthias Nießner, and Peter

- Kontschieder. Multidiff: Consistent novel view synthesis from a single image. In CVPR, pages 10258–10268, 2024. 2
- [42] Tuan Duc Ngo, Peiye Zhuang, Evangelos Kalogerakis, Chuang Gan, Sergey Tulyakov, Hsin-Ying Lee, and Chaoyang Wang. Delta: Dense efficient long-range 3d tracking for any video. In ICLR, 2025. 3, 4
- [43] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205,

2023. 5

- [44] Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. arXiv preprint arXiv:2502.20110, 2025. 4, 8
- [45] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828–56858, 2025. 2
- [46] Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas M¨uller, Alexander Keller, Sanja Fidler, and Jun Gao. Gen3c: 3d-informed world-consistent video generation with precise camera control. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 6121–6132, 2025. 2, 6, 7
- [47] Xiaoyu Shi, Zhaoyang Huang, Fu-Yun Wang, Weikang Bian, Dasong Li, Yi Zhang, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, et al. Motion-i2v: Consistent and controllable image-to-video generation with explicit motion modeling. In ACM SIGGRAPH, pages 1–11, 2024. 2
- [48] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. In ICLR, 2023. 2
- [49] Wenqiang Sun, Shuo Chen, Fangfu Liu, Zilong Chen, Yueqi Duan, Jun Zhang, and Yikai Wang. Dimensionx: Create any 3d and 4d scenes from a single image with controllable video diffusion. arXiv preprint arXiv:2411.04928, 2024. 2, 6, 7
- [50] KLING AI Team. Kling image-to-video model, 2024. 2
- [51] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In CVPR, pages 1526–1535, 2018. 2
- [52] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. NeurIPS, 30, 2017. 2
- [53] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Generating videos with scene dynamics. 2016. 2
- [54] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 3, 5, 4
- [55] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 4

- [56] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, pages 20310–20320, 2024. 2
- [57] Jianzong Wu, Xiangtai Li, Yanhong Zeng, Jiangning Zhang, Qianyu Zhou, Yining Li, Yunhai Tong, and Kai Chen. Motionbooth: Motion-aware customized text-to-video generation. NeurIPS, 37:34322–34348, 2024. 2
- [58] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In CVPR, pages 26057–26068, 2025. 2
- [59] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation with multi-frame and multi-view consistency. arXiv preprint arXiv:2407.17470, 2024. 2
- [60] Jinbo Xing, Menghan Xia, Yuxin Liu, Yuechen Zhang, Yong Zhang, Yingqing He, Hanyuan Liu, Haoxin Chen, Xiaodong Cun, Xintao Wang, et al. Make-your-video: Customized video generation using textual and structural guidance. IEEE TVCG, 31(2):1526–1541, 2024. 2
- [61] Heng Yu, Chaoyang Wang, Peiye Zhuang, Willi Menapace, Aliaksandr Siarohin, Junli Cao, L´aszl´o Jeni, Sergey Tulyakov, and Hsin-Ying Lee. 4Real: Towards photorealistic 4d scene generation via video diffusion models. NeurIPS, 37:45256–45280, 2024. 2, 6, 7
- [62] Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048, 2024. 2
- [63] Yuyang Zhao, Zhiwen Yan, Enze Xie, Lanqing Hong, Zhenguo Li, and Gim Hee Lee. Animate124: Animating one image to 4d dynamic scene. arXiv preprint arXiv:2311.14603,

2023. 2

- [64] Yuyang Zhao, Chung-Ching Lin, Kevin Lin, Zhiwen Yan, Linjie Li, Zhengyuan Yang, Jianfeng Wang, Gim Hee Lee, and Lijuan Wang. GenXD: Generating Any 3D and 4D Scenes. In ICLR, 2025. 6, 2, 3, 7
- [65] Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, and Jiangbo Lu. Nerflix: High-quality neural view synthesis by learning a degradation-driven interviewpoint mixer. In CVPR, pages 12363–12374, 2023. 2
- [66] Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In ECCV, pages 145–163. Springer, 2024. 2

