# arXiv:2510.22706v3[cs.CV]31Oct2025

## IGGT: INSTANCE-GROUNDED GEOMETRY TRANSFORMER FOR SEMANTIC 3D RECONSTRUCTION

#### Hao Li1,2,3, Zhengyu Zou1, Fangfu Liu4, Xuanyang Zhang3∗, Fangzhou Hong2, Yukang Cao2, Yushi Lan2, Manyuan Zhang5, Gang Yu3, Dingwen Zhang1 , Ziwei Liu2 1NWPU 2S-Lab, NTU 3StepFun, Inc. 4THU 5MMLab, CUHK

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

3D Reconstruction of the Scene

3D PCA Visualization of the Scene

[Figure 44]

[Figure 45]

Various Downstream Applications

InsScene-15K Dataset

[Figure 46]

[Figure 47]

[Figure 48]

Geometry Segmentthesofa nearest to newspaper in the scene.

[Figure 49]

Instance Grounded Transformer

[Figure 50]

“yellow counter”

“cabinet”

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

### 15K Scenes 200M Imgs

[Figure 73]

The Masks as follow:

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

In total

High-Quality Synthesis

High-Quality RGBD Scan

Scalable Web-Captured

Instance Spatial Tracking

Open-Vocab Segmentation

Scene Grounding

Figure 1: IGGT: building upon our curated large-scale dataset InsScene-15K, we propose a novel end-to-end framework that enables geometric reconstruction and contextual understanding in a unified representation. This paradigm facilitates a wide range of applications, including spatial tracking, 2D / 3D open-vocabulary segmentation, and scene grounding.

ABSTRACT

Humans naturally perceive the geometric structure and semantic content of a 3D world as intertwined dimensions, enabling coherent and accurate understanding of complex scenes. However, most prior approaches prioritize training large geometry models for low-level 3D reconstruction and treat high-level spatial understanding in isolation, overlooking the crucial interplay between these two fundamental aspects of 3D-scene analysis, thereby limiting generalization and leading to poor performance in downstream 3D understanding tasks. Recent attempts have mitigated this issue by simply aligning 3D models with specific language models, thus restricting perception to the aligned model’s capacity and limiting adaptability to downstream tasks. In this paper, we propose InstanceGrounded Geometry Transformer (IGGT), an end-to-end large unified transformer to unify the knowledge for both spatial reconstruction and instance-level contextual understanding. Specifically, we design a 3D-Consistent Contrastive Learning strategy that guides IGGT to encode a unified representation with geometric structures and instance-grounded clustering through only 2D visual inputs. This representation supports consistent lifting of 2D visual inputs into a coherent 3D scene with explicitly distinct object instances. To facilitate this task, we further construct InsScene-15K, a large-scale dataset with high-quality RGB images, poses, depth maps, and 3D-consistent instance-level mask annotations with a novel data curation pipeline. Unlike previous methods that bound with a specific language model, we introduce an Instance-Grounded Scene Understanding paradigm, where instance masks serve as the bridge connecting our

∗Project Leader. Corresponding Authors.

unified representation with diverse Visual Language Models (VLMs) in a plugand-play manner, substantially expanding downstream understanding capabilities. Extensive experiments on instance spatial tracking, open-vocabulary segmentation, and QA scene grounding demonstrate that IGGT outperforms state-ofthe-art methods in both quality and consistency for semantic 3D reconstruction. https://github.com/lifuguan/IGGT_official.

- 1 INTRODUCTION

A foundational goal in the pursuit of spatial intelligence (Yang et al., 2025) is to build representations that mirror human understanding—capturing both the precise geometric structure and rich semantic content of a scene from visual sensory inputs such as RGB images. Such representations are vital for enabling downstream tasks like robotic manipulation (Qu et al., 2025), AR / VR (Jiang et al., 2025), and planning (Zhang et al., 2024).

Previous methods (Zust et al., 2025; Fan et al., 2024; Sun et al., 2025) tackle this challenge through a fragmented paradigm, decoupling 3D geometric reconstruction and high-level semantic understanding into isolated tasks. Typically, they first leverage geometry-focused techniques (e.g., Multi-View Stereo (MVS) methods (Sch¨onberger et al., 2016; Sch¨onberger & Frahm, 2016) or off-the-shelf large Image-to-3D models (Wang et al., 2024; 2025)) to predict low-level 3D structures, followed by vision-language models (VLMs) (Bai et al., 2023; 2025) or 2D segmentation models (Cheng et al., 2022) to perform high-level semantic segmentation tasks. However, these disjointed approaches are inherently flawed, as they propagate errors between stages and fail to leverage the mutual context between shape and identity, preventing them from enhancing each other’s capabilities and hindering their ability to support model reconstruction.

Recently emerged methods (Fan et al., 2024; Sun et al., 2025) attempt to bridge this gap by aligning spatial models with specific VLM (Li et al., 2022). However, these approaches suffer from three critical limitations. First, since 3D geometry contains low-level, fine-grained structural signals, forcing a strict alignment with high-level textual concepts can over-smooth the representation, degrading high-frequency geometric details and undermining multi-view consistency. Second, this tight coupling to a specific VLM architecture inherently restricts the performance to the base model (e.g., LSeg (Li et al., 2022)) and prevents the integration of newer, more powerful foundation models (e.g., CLIP (Radford et al., 2021), SigLIP (Tschannen et al., 2025)). Third, since these VLMs (Li et al., 2022; Ghiasi et al., 2022) are mainly trained on 2D image–text pairs, their aligned features often fail to distinguish objects within the same semantic category, which significantly limits more downstream applications (e.g., , 3D instance-consistent tracking under large viewpoint changes and spatial QA when interfaced with VLMs).

To address this, we propose Instance-Grounded Geometry Transformer (IGGT), a novel end-to-end framework that unifies the representation for spatial reconstruction and contextual understanding. Instead of simply aligning geometry with language features, our key idea is to couple both factors by joint training and encourage the model to autonomously learn the relationship between 3D instancelevel semantics and their geometric structures, yielding mutual improvements in contextual understanding and geometry reconstruction. Specifically, 1) we employ a large Unified Transformer to encode multi-view images into unified token representations of the 3D scene, which are decoded by a Geometry Head and an Instance Head into geometric point maps and an instance clustering fields, respectively. 2) we employ a cross-modal fusion block with a window-shifted attention mechanism, enabling the Instance Head to leverage fine-grained geometric features at pixel level to enhance its spatial awareness. 3) To further improve multi-view consistency of the instance fields, we design a 3D-consistent contrastive learning strategy that guides IGGT to learn both geometric structures and instance-grounded clustering features. As instance-level geometry-semantics aligned annotations remain scarce in the community, we facilitate this task by presenting a large-scale dataset coined InsScene-15K, a meticulously constructed dataset comprising high-quality RGB images, poses, depth maps, and 3D-consistent instance masks.

One more thing, after training the full model (i.e., IGGT), we design an Instance-Grounded Scene Understanding strategy, where instance masks serve as the bridge connecting IGGT with diverse VLMs. Such a paradigm not only enables the seamless, plug-and-play integration of various visionlanguage models (VLMs) such as CLIP and SigLIP to lift downstream task performance, but also

Yes

Propagate Masks Forward

Video Frames Input

Frame 0: Initial Mask Proposals

Frame t: New Mask Proposals

Multi-View Consistent Object Segmentation Masks

...

...

...

[Figure 92]

[Figure 93]

[Figure 94]

Uncovered Area > Threshold?

[Figure 95]

[Figure 96]

[Figure 97]

...

...

...

...

[Figure 98]

[Figure 99]

- (a) Video Captured Mask Annotation Pipeline
- (b) RGBD Captured Mask Refinement Pipeline

Update Masks

[Figure 100]

Mask Match & Merge

No

...

GT Projected Object Masks

GT 3D Ann.

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Bi-directional Propagation

2K

ScenePart-15K

Match SAM masks with GT IDs

Refined Object Masks

- 1K
- 2K

Initial Mask Proposals

RGB Image

Yes Uncovered No Area >

[Figure 107]

Re10K

ScenePart-15K Data Distribution (Scene)

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

Infinigen

[Figure 112]

Threshold?

Aria Syn. Env.

10K

Merge SAM masks with same IDs

Scannet++

- Figure 2: Data Curation Pipeline. Our data is collected from various sources and then annotated by a novel data engine driven by SAM2 (Ravi et al., 2024). (a) For video captured scenes (i.e., RE10k (Zhou et al., 2018)), we annotate them through a customized SAM2 video dense prediction pipeline. (b) For RGBD-scan scenes (e.g., ScanNet++ (Yeshwanth et al., 2023)), we regenerate dense mask annotations for each image and align them with the projected coarse GT masks.

extends to Large Multimodal Models (LMMs) (Bai et al., 2023; 2025), unlocking more sophisticated scene understanding and a broader spectrum of applications like scene grounding.

We validate our framework through extensive experiments on diverse downstream tasks (e.g., spatial tracking segmentation, open-vocabulary segmentation, and scene grounding), demonstrating its superiority over state-of-the-art methods in both task performance and 3D scene coherence.

2 INSSCENE-15K DATASET

We construct the InsScene-15K dataset (in Sec. 2), where each scene includes corresponding RGB images, depth maps, poses, and 3D-consistent instance segmentation masks. To maintain consistency, we ensure that each instance retains a unique ID across all views.

Our data curation pipeline systematically integrates three distinct categories of data to ensure comprehensiveness and diversity, as illustrated in Fig. 2: 1) synthesis (Aria (Pan et al., 2023), Infinigen (Raistrick et al., 2024)); 2) Video captured (RE10K (Zhou et al., 2018)); 3) RGBD captured (Scannet++ (Yeshwanth et al., 2023)). For synthetic datasets (e.g., Aria and Infinigen), we simultaneously generate the RGB image, depth map, camera pose, and object-level segmentation masks for each rendered view. Since the simulation environment provides perfectly accurate 2D groundtruth masks (in Fig. 3 (a)), we use them directly without any post-processing. Moreover, regarding real-world scenarios, we propose a novel data curation pipeline that includes multi-view mask annotation and refinement stages, driven by SAM2 (Ravi et al., 2024). Specifically, for real-world video-captured scenes such as RE10K (Zhou et al., 2018) (Fig. 2a), our method first employs SAM to generate dense mask proposals on the initial frame. These proposals are then used as prompts

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

RGB Image Vanilla GT Our Refined

(c) RGBD-Scan Scene

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

RGB Image Gen. Masks

(b) Web-Captured Scene

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

RGB Image Our Refined

(a) Synthesis Scene

- Figure 3: Visualization of mask annotations from three different sources. For the RGBD-scan scene, we additionally compare the vanilla ground-truth masks from ScanNet++ (Yeshwanth et al., 2023) with our refined annotations, along with their corresponding matched IDs and mIoU scores.

for the SAM2 video object segmenter to propagate masks temporally throughout the sequence. To handle new objects and mitigate drift, we adopt an iterative strategy that designates a new keyframe whenever the unsegmented area increases, where SAM is reapplied to discover objects in the uncovered regions. After processing the entire video, a final bi-directional propagation pass ensures high temporal consistency across object tracks. This curation strategy provides scalable and diverse annotations that enhance the generalization ability of our model.

For challenging datasets with large-scale camera motion but coarse 3D annotations such as ScanNet++ (Yeshwanth et al., 2023), we first project the 3D annotations into 2D to obtain initial imagelevel object masks. While this guarantees multi-view consistency of object IDs, the masks are often coarse and imprecise. To improve their quality, we use SAM2 to generate fine-grained initial mask proposals that are accurate in shape but lack identity information. These proposals are then aligned with the projected ground-truth masks to assign consistent object IDs (Fig. 3 (c)), and proposals belonging to the same ID are merged into complete masks. The process is iteratively refined until all image regions are covered. This pipeline (Fig. 2 (b)) achieves both multi-view ID consistency and shape-accurate annotations, substantially improving 2D mask quality for real-world scenarios.

- 3 METHODOLOGY

- 3.1 OVERVIEW

Our method consists of two main phases. Firstly, we propose IGGT (in Sec. 3.2), a unified foundation model that simultaneously predicts instance-discriminative features at the spatial level and performs 3D reconstruction through 3D-consistent contrastive learning on large-scale datasets. Secondly, we propose an instance-grounded scene understanding strategy (Sec. 3.3). This strategy employs unsupervised clustering to partition the scene into instances by grouping the predicted features into masks with consistent instance IDs. These masks are then used to guide state-of-the-art vision-language models (VLMs, e.g., CLIP, OpenSeg) and large multimodal models (LMMs, e.g., GPT-4o, Qwen2.5-VL) to perform open-vocabulary scene querying and grounding tasks.

- 3.2 ARCHITECTURE OF IGGT

As illustrated in Fig. 4, given N input images {Ii ∈ RH×W×3}Ni=1, we aim to forge a unified representation, enabling comprehensive 3D reconstruction and understanding in a mutually reinforcing

manner. Specifically, we propose IGGT F, which predicts camera parameters ti, depth map Di, point map Pi, and 3D-consistent, instance-level feature maps Si in a feed-forward manner:

F : {Ii}Ni=1  → (ti, Di, Pi, Si)Ni=1. (1)

Our IGGT consists of three parts: 1) a Large Unified Transformer to capture Unified Token Representation from multiple images; 2) two Downstream Heads with a Cross-Modal Fusion Block to simultaneously predict geometric structures and corresponding instance features via a mutual enhancement pattern; 3) a 3D consistent supervision to empower the model to construct 3D-consistent feature fields.

Large Unified Transformer. We follow VGGT to construct a 1B parameter large unified Transformer, designed to encode the multi-view images {Ii}Ni=1 into a set of powerful unified token representations {Ti ∈ RM×D}Ni=1, where M denotes the numbers of the tokens for each image and D is the dimension of the token. Our large Unified Transformer first adopts pretrained DINOv2 (Oquab

- et al., 2023) to extract patch-level image tokens. To support arbitrary multi-view inputs while maintaining permutation equivariance, a learnable camera token is concatenated to each view’s token sequences. Subsequently, 24 blocks of intra-view self-attention and global-view cross-attention are

applied to transform the image tokens into unified tokens {Ti}Ni=1, capturing both local and global context, which enables a holistic and globally consistent understanding of the 3D scene.

Downstream Heads and Cross-Modal Fusion Block. We employ two downstream branches—Geometry Head and Instance Head—to decode the unified tokens Ti} into geometric and instance features, respectively. The Geometry Head, inheriting its design from VGGT, is composed of three distinct modules: a camera predictor, a depth predictor, and a point predictor. The camera predictor is tasked with regressing camera parameters, including extrinsics and intrinsics, from camera-specific tokens. For dense prediction, the depth and point predictors employ a DPTlike architecture (Ranftl et al., 2021). This architecture reconstructs a hierarchical geometric feature

###### Instance-Grounded Scene Understanding

Pred. Depth

x L times

Geometry Head

[Figure 134]

[Figure 135]

[Figure 136]

AdaptorAdaptor

[Figure 137]

DINODINODINO

[Figure 138]

###### Instance Spatial Tracking

[Figure 139]

Cross-Attention Block

Self-Attention Block

Open-Vocab Semantic Segmentation

Clus. Masks

Large Unified Transformer

[Figure 140]

Cross-Modal Fusion

[Figure 141]

[Figure 142]

[Figure 143]

###### Geo. Recon. 3D Ins. Recon. QA Scene Grounding

[Figure 144]

Pred. Feats GT Ins Masks

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

MVC-Loss

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

###### InsScene-15K

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

Instance Head

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

###### Open-Vocab Segmentation QA Scene Grounding

###### Cross-Modal Fusion

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

Geo. Latent

[Figure 217]

[Figure 218]

[Figure 219]

“Please segment the white backet near to the sink.”

V

[Figure 220]

OpenSeg

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

Window Shifted Attention

[Figure 225]

CLIP

Ins. Latent

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

K

[Figure 232]

Qwen-VL 2.5

Ins. Latent

Q

Cluster Masks CLIP Features “Sink” Cluster Masks Highlighted Ins.

- Figure 4: Overview of IGGT. Given input images, our method encodes them into a series of Unified Token Representations, which are then processed by the Geometry Head and the Instance Head to produce high-quality geometric reconstructions and instance-grounded clusterings simultaneously. In the end, we introduce Instance-Grounded Scene Understanding to perform multiple applications.

Fipt = {Fi,pt(l)}4l=1 from the unified tokens through progressive upsampling and multi-scale fusion network Φpt(·). Similar to this dense prediction paradigm, our Instance Head Φins(·) also adopts a DPT-like architecture to perform dense instance features Fiins = {Fi,ins(l)}4l=1:

{Fipt} = Φpt({Ti}), {Fiins} = Φins({Ti}). (2)

Moreover, to enhance the fine-grained spatial awareness of the instance head, we propose a crossmodal fusion block Fwin(·), which utilizes a sliding window cross attention to embed spatial structure into the instance representation, making them more sensitive to object boundaries and spatial layouts while avoiding the quadratic complexity of global attention:

Fˆi,ins(l) = Fi,ins(l) + Fwin(Q = Fi,ins(l), K = Fi,pt(l), V = Fi,pt(l)). (3)

After that, we concatenate all refined instance features {Fˆi,ins(l)} and map them through a conventional

- 3 × 3 convolutional layer to 8 dimensional instance features Oins ∈ RN×8×H×W. 3D-Consistent Contrastive Supervision. We enforce 3D consistency on the instance features

Oins ∈ RN×8×H×W by applying a multi-view contrastive loss Lmvc, which is designed to pull features from the same 3D instance together across views while pushing features from different instances apart. Given a set of sampled pixels P, the loss is formulated as:

max(0, M − d(fpi, fpj)) (4)

Lmvc = λpull ·

d(fpi, fpj) + λpush ·

pi,pj∈P m(pi)=m(pj)

pi,pj∈P m(pi)̸=m(pj)

Here, d(·,·) is the L2 distance between normalized features, m(pi) is the instance ID of pixel pi. The coefficients λpull and λpush balance the pulling and pushing terms, while M is a margin hyperparameter that controls the discriminative between different instances. This objective structures the instance representations according to the 3D scene geometry, improving generalization. Overall, we train the whole model in a multi-task loss:

Loverall = Lpose + Ldepth + Lpmap + Lmvc, (5)

where geometry supervision terms pose Lpose, depth Ldepth, and point map Lpmap are followed by the training paradigm of VGGT, which is used to supervise the outputs of the geometry head.

- 3.3 INSTANCE-GROUNDED SCENE UNDERSTANDING

Unlike prior approaches that are tightly coupled with a specific language model (e.g., for OpenVocabulary Segmentation) and thus limited to a single type of task, we decouple our framework from specific language models and propose a novel Instance-Grounded Scene Understanding strategy to support a broad range of downstream tasks. As shown in Tab. 1, our method is the only one that simultaneously enables spatial tracking, image-to-3D reconstruction, and scene understanding, while achieving state-of-the-art performance across all tasks.

- Table 1: Quantitative Results on Scannet (Dai et al., 2017). Here we showcase the capability overview and report the spatial track quality, reconstruction accuracy, and 2D / 3D open-vocabulary semantic segmentation accuracy. The bold denotes the best results.

Model Capability Spatial Track Recon. Metric Open-Vocab. Semantic Segment Recon. Understand Track T-mIoU↑ T-SR↑ Abs. Rel↓ τ↑ 2D mIoU↑ 2D mAcc↑ 3D mIoU↑

LSeg ✗ ✓ ✗ - - - - 58.11 65.76 OpenSeg ✗ ✓ ✗ - - - - 42.33 68.06 NeRF-DFF ✓ ✓ ✗ - - 7.99 36.53 45.40 65.29 12.29 Feature-3DGS ✓ ✓ ✗ - - 6.48 41.63 57.69 63.26 23.42 LSM (2 Views) ✓ ✓ ✗ - - 4.22 58.65 53.07 53.86 LSM (Multi-Views) ✓ ✓ ✗ - - 3.17 64.81 53.40 59.50 35.37 SpaTracker+SAM ✗ ✗ ✓ 26.43 38.57 - - - - SAM2* ✗ ✓ ✓ 53.74 71.25 - - - - VGGT ✓ ✗ ✗ - - 1.84 83.60 - - Ours ✓ ✓ ✓ 69.41 98.66 1.90 83.71 60.46 81.84 39.68

- Table 2: Quantitative Results on Scannet++ (Yeshwanth et al., 2023). Here we report the spatial track quality, reconstruction accuracy, and 2D / 3D open-vocabulary semantic segmentation accuracy.

Model Spatial Track Recon. Metric Open-Vocab. Semantic Segment

T-mIoU↑ T-SR↑ Abs. Rel↓ τ↑ 2D mIoU↑ 2D mAcc↑ 3D mIoU↑

LSeg - - - - 22.61 34.42 OpenSeg - - - - 13.92 48.13 Feature-3DGS - - 5.92 41.64 22.47 33.14 10.59 LSM (2 Views) - - 4.22 74.02 17.76 26.95 LSM (Multi-Views) - - 2.96 83.28 17.88 27.84 15.17 SpaTracker+SAM 16.15 23.68 - - - - SAM2* 44.16 57.89 - - - - VGGT - - 2.75 85.41 - - Ours 73.02 98.90 2.61 85.66 31.31 70.78 20.14

Instance Spatial Tracking. Specifically, inspired by SAMPart3D (Liu et al., 2025), we apply the density-based clustering algorithm HDBSCAN (McInnes et al., 2017) that gathers multi-view 2D instance features {Oiins} into K distinct clusters, where each cluster represents a unique object instance present in the scene. Then we re-project the assigned cluster labels to their corresponding pixel locations produces a set of 3D-consistent 2D instance masks {Mi,kins}Kk=1. Such a paradigm enables dense tracking and segmentation of specific instances across multi-view images by leveraging explicit 3D priors, in stark contrast to existing methods that are either limited to discriminating category-level features or lose targets during significant camera motion.

Open-Vocabulary Semantic Segmentation. These 3D-consistent instance masks serve as effective prompts for any off-the-shelf VLMs (Radford et al., 2021; Ghiasi et al., 2022), enabling them to perform robust open-vocabulary semantic segmentation by assigning a semantic category to each mask-defined region. Here we take OpenSeg (Ghiasi et al., 2022) as an example. It first produces

image-wise features {Filang ∈ RD×H×W}Ni=1, which considers contextual information to enable accurate visual-language alignment of the features. We then aggregate the features within each 2D

instance mask {flangk ∈ RD}Ki=1 via average mask pooling, yielding a compact representation for each instance. This step not only integrates the mask priors into the visual-language space, but also

sharpens object boundaries and captures fine-grained local category cues, making the subsequent semantic assignment more accurate and robust.

QA Scene Grounding. Unlike prior methods that directly align 3D features with language embeddings, our approach offers greater flexibility by decoupling instance clusterings, which can then interact with LMMs (Bai et al., 2025; Team et al., 2023) to support object-centric QA in 3D scenes. Concretely, as shown in Fig. 4, given N views, we highlight the image regions corresponding to the same instance k with masks {Mi,kins}Ni=1 (rendered in red), and query the LMM with yes/no questions to verify object consistency across views. Finally, we aggregate all positive (“yes”) responses and concatenate the corresponding masks to form the final segmentation output.

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

1st Frame N-2th Frame N-1th Frame Nth Frame

1st Frame N-2th Frame N-1th Frame Nth Frame Recon. Scene: 0a76e06478

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

SpaTrck+SAMSAM2*GT MasksiGGT(Our)

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

[Figure 258]

Recon. Scene: scene0758_00

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

- Figure 5: Qualitative results on Instance Spatial Tracking. We present two example scenes from ScanNet (Dai et al., 2017) and ScanNet++ (Yeshwanth et al., 2023), and compare our method with SAM2* and SpaTracker+SAM. All instances are visualized with distinct IDs and colors for clarity.

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

Images Ins. PCA Masks ScannetID:Scene0708_00 Images Ins. PCA Masks

[Figure 276]

[Figure 277]

[Figure 278]

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

Scannet++ ID: f6659a3107

- Figure 6: We visualize our 3D-consistent PCA results with corresponding clustered masks derived from instance-grounded features. Similar colors in PCA indicate higher feature similarity between instances. For clustered masks, the same object instance shares the same color across multi-views.

- 4 EXPERIMENTS

Evaluation Details. We conduct comprehensive experiments on the ScanNet (Dai et al., 2017) and ScanNet++ (Yeshwanth et al., 2023) datasets. From each dataset, we randomly select 10 scenes and sample 8-10 images per scene, with the selection strategy designed to maximize spatial coverage of the scene while preserving sufficient overlap to ensure cross-view consistency. (a) For Instance Spatial Tracking evaluation, we evaluate tracking performance using Temporal mIoU (TmIoU) and Temporal Success Rate (T-SR). T-mIoU measures the segmentation accuracy of the same object across different views, while T-SR assesses whether the object is successfully tracked in every view. (b) For Open-Vocabulary Segmentation evaluation, we follow LangSplat (Qin et al., 2024) and LangSurf (Li et al., 2024b), which adopt mIoU and mAcc to measure 2D segmentation accuracy. In addition, we evaluate the 3D mIoU metric by aligning the reconstructed scene with the ground-truth point cloud. (c) For Reconstruction evaluation, we follow LSM (Fan et al., 2024) and VGGT (Wang et al., 2025) that utilize Absolute Relative Error (Abs. Rel) and Inlier Ratio (τ) with a threshold of 1.03 to assess each scene. The details of these metrics are shown in the appendix.

Evaluation of Instance Spatial Tracking. To comprehensively evaluate the tracking quality of our proposed method and competing approaches, particularly under large viewpoint changes with multiple objects, we manually annotate a subset of objects across several scenes with precise ground-truth labels (more visualization in the Appendix). For baseline methods, we modify SAM2 (Ravi et al., 2024) to support dense segmentation and tracking under multi-view inputs, denoted as SAM2*. In addition, we integrate SAM into SpaTrackerV2 (Xiao et al., 2025), where tracking points are used as prompts to perform dense segmentation. Tab. 1 and Tab. 2 present the quantitative results, demonstrating the significant superiority of our method. By leveraging implicit 3D reasoning, our approach successfully distinguishes object identities to achieve nearly 100% T-SR accuracy. In contrast, baseline methods fail at this crucial task, yielding a T-mIoU below 30%, whereas our approach surpasses 60%. This performance gap is visually demonstrated in Fig. 5, where our method successfully tracks and segments the chair under large camera motions, while competing methods lose the track.

###### RGB Image GT Label IGGT (Ours) LSM (Multi-View) LSM (2 View) Feature-3DGS LSeg OpenSeg

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

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

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

Figure 7: Qualitative Results of 2D Open-Vocabulary Segmentation on Scannet and Scannet++.

Furthermore, we provide additional visualizations of our 3D-consistent instance features using Principal Component Analysis (PCA), along with their corresponding clustered masks, as shown in Fig. 6. As illustrated, IGGT produces 3D-consistent instance-grounded features that remain discriminative across multiple views: multiple instances with the same category exhibit similar yet distinguishable colors in the PCA space. This property serves as a crucial foundation for the Instance Spatial Tracking task, as it enables consistent tracking and segmentation of individual objects even under large motions and in the presence of many similar instances.

Evaluation of Open-Vocabulary Segmentation. We compare our method with other Image-to-3D feedforward method (Fan et al., 2024), per-scene optimized methods (Zhou et al., 2024; Kobayashi et al., 2022), and 2D methods (Ghiasi et al., 2022; Li et al., 2022) on both Scannet and Scanent++ datasets. The results are reported in Tab. 1 and Tab. 2. On ScanNet++, our method achieves leading performance, surpassing other approaches by 8.34% in mIoU for segmentation and 7.88% in mAcc for object localization.

This performance improvement is attributed to our method’s superior multi-view consistency, which helps correct object recognition errors caused by incomplete views, as illustrated in Fig. 7, where the sink is difficult to identify due to limited viewpoint coverage. On the other hand, we also evaluate the accuracy of depth estimation on multi-view inputs. The results show that our method is on par with VGGT on ScanNet, and outperforms VGGT on ScanNet++ by 0.14 in Abs. Rel and 0.25 in τ, benefiting from the mutual enhancement of semantics and geometry achieved through joint training. This performance improvement is further demonstrated in 3D segmentation (see Tab. 1 and Tab. 2), where our method outperforms previous approaches by 4.31% and 4.97% in terms of 3D mIoU. As shown in Fig. 8, our method achieves superior 3D semantic representations while also maintaining better segmentation consistency in the same regions.

###### GT RGB Mesh GT Semantic Mesh Pred. LSM-MV Mesh Pred. iGGT Mesh (Ours)

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

Figure 8: Visualization of 3D Open-Vocab. Segmentation.

Applications of QA Scene Grounding. We present the QA application results in Fig. 9 on the Teatime scene from the LERF-OVS (Kerr et al., 2023) dataset, and compare our approach against the state-of-the-art Gemini 2.5 Pro (Comanici et al., 2025). As shown, our instancegrounded querying fully leverages the reasoning capacity of LMMs, achieving accurate segmentation for complex prompts and superior multi-view consistency compared to existing unified generation–understanding models, thereby enabling more complex QA tasks in 3D scenes.

Ablation Study. Here, we showcase the training curve of our IGGT in Fig. 11. Without the crossmodal fusion model, the instance head struggles to capture high-resolution geometric information, resulting in more difficult convergence, as reflected in the sharpness of the chair’s edges in the PCA visualization. We also conduct ablations on integrating different VLMs into our method (e.g., LSeg (Li et al., 2022), CLIP (Radford et al., 2021), OpenSeg (Ghiasi et al., 2022)). As shown in the

[Figure 348]

[Figure 349]

Images

LSM Ours w/ LSeg Ours w/ CLIP

Ours w/OpenSeg

Please segment the animal sitting on the chair? Do the highlighted red areas across these multiple views match what is being asked for? Answer only "yes" or "no".

Please segment the animal Queried Multi-View Masks sitting on the chair and highlight the segmented objects in red colors across all provided views.

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

###### DALL-E“”

###### ID = 1 ID = 2 ID = N

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

…

3D Visualization

Ottolegnghi“”

Here are the segmented images highlighted in red:

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

Cabinet“”

Yes Yes No

(a) Vanilla Gemini 2.5 Pro (b) Ours Instance-Grounded Scene Understanding + Qwen-VL 2.5

Figure 9: Applications of QA Scene Understanding compared with vanilla Gemini 2.5 Pro Comanici et al. (2025) model.

##### Figure 10: Visualization of our method using different VLMs.

[Figure 401]

w/ Multi-Modal Fusion

[Figure 402]

Table 3: Integration with Different VLMs.

Scannet Scannet++

w/o Multi-Modal Fusion

Method

[Figure 403]

mIoU↑ mAcc↑ mIoU↑ mAcc↑

Ours w/ Lseg 60.46 81.84 22.72 63.56 Ours w/ CLIP 49.36 62.68 21.52 61.36 Ours w/ OpenSeg 58.12 78.75 31.31 70.78

RGB Images

[Figure 404]

Figure 11: Ablation on Cross-Modal Fusion.

table, LSeg and OpenSeg, with better global context representation, achieve higher accuracy in handling background classes (e.g., cabinet). In contrast, CLIP, with superior text alignment capabilities, performs better on complex categories, such as ‘DALL-E’ and ‘Ottolegnghi’ shown in Fig. 10. This further demonstrates the flexibility of our method in utilizing different VLMs to achieve improved text query performance.

- 4.1 RELATED WORK

Spatial Foundation Models. Image-to-3D reconstruction has evolved from early SfM pipelines like COLMAP (Schonberger & Frahm, 2016), which estimate camera poses and sparse point clouds, to more advanced methods like 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) for efficient novel view synthesis. Scene Representation Transformers (Sajjadi et al., 2022) represent images as latent tokens, enabling view synthesis without accurate poses, but still struggle with explicit geometry and generalization. DUSt3R (Wang et al., 2024) improves upon this by directly regressing dense point maps from unposed image pairs, while VGGT (Wang et al., 2025) scales this approach to multiple images with competitive accuracy. However, these methods remain focused on geometric reconstruction, often neglecting higher-level scene understanding.

3D Scene Understanding. Integrating semantics into 3D reconstruction is vital for scene understanding. Methods like LangSplat (Qin et al., 2024) inject vision-language features into 3D Gaussian Splatting, enabling semantic reasoning, but typically require dense multi-view inputs and per-scene optimization. Approaches such as Panst3R (Zust et al., 2025) and DUSt3R (Wang et al., 2024) attempt feed-forward scene understanding, but decouple geometry and semantics, limiting mutual benefits. Methods like LSM (Fan et al., 2024) and Uni3R (Sun et al., 2025) align spatial models with vision-language models (e.g., LSeg (Li et al., 2022)), but face limitations in integrating stronger VLMs and struggle with fine-grained, instance-level queries in complex scenes.

- 5 CONCLUSION

In this paper, we introduce IGGT, a novel end-to-end framework that unifies the representation for both spatial reconstruction and contextual understanding in a 3D scene. The key to our success is that we couple geometric and instance-level semantic features by joint training and unleash the potential of a unified large transformer to achieve mutual improvements in contextual understanding and geometry reconstruction. To facilitate this task, we further present a large scale dataset

called InsScene-15K, including high-quality RGB images, poses, depth maps, and 3D-consistent instance masks. Moreover, our proposed instance-grounded scene understanding strategy enables IGGT with plug-and-play integration of various VLMs and LMMs, unlocking a broader range of applications. Extensive experiments demonstrate the superiority of our IGGT over the latest stateof-the-art methods in terms of high task performance and 3D coherence. We believe that IGGT provides a promising research direction for crafting and understanding intricate 3D worlds jointly and will inspire more works in the future.

- 6 ETHICS STATEMENT

This work focuses on improving spatial reconstruction and understanding. While our model is trained on self-annotated datasets based on standard open-source images and tested in controlled settings, we acknowledge that any AI system may potentially exhibit biases or produce unexpected behaviors. Our research is intended for academic exploration only, and we emphasize that any such outcomes do not reflect the views of the authors. We support the development of AI technologies that are ethical, safe, and aligned with societal values.

- 7 REPRODUCIBILITY STATEMENT All code and model checkpoints will be publicly released to ensure reproducibility.

REFERENCES

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Danpeng Chen, Hai Li, Weicai Ye, Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, and Guofeng Zhang. Pgsr: Planar-based gaussian splatting for efficient and high-fidelity surface reconstruction. IEEE Transactions on Visualization and Computer Graphics, 2024.

Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Maskedattention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1290–1299, 2022.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5828–5839, 2017.

Zhiwen Fan, Jian Zhang, Wenyan Cong, Peihao Wang, Renjie Li, Kairun Wen, Shijie Zhou, Achuta Kadambi, Zhangyang Wang, Danfei Xu, et al. Large spatial model: End-to-end unposed images to semantic 3d. Advances in neural information processing systems, 37:40212–40229, 2024.

Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Scaling open-vocabulary image segmentation with image-level labels. In European conference on computer vision, pp. 540–557. Springer, 2022.

Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pp. 1–11, 2024.

Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 19729–19739, 2023.

Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing nerf for editing via feature field distillation. Advances in neural information processing systems, 35:23311–23330, 2022.

Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Ren´e Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.

Hao Li, Yuanyuan Gao, Chenming Wu, Dingwen Zhang, Yalun Dai, Chen Zhao, Haocheng Feng, Errui Ding, Jingdong Wang, and Junwei Han. Ggrt: Towards pose-free generalizable 3d gaussian splatting in real-time. In European Conference on Computer Vision, pp. 325–341. Springer, 2024a.

Hao Li, Roy Qin, Zhengyu Zou, Diqi He, Bohan Li, Bingquan Dai, Dingewn Zhang, and Junwei Han. Langsurf: Language-embedded surface gaussians for 3d scene understanding. arXiv preprint arXiv:2412.17635, 2024b.

Minghua Liu, Mikaela Angelina Uy, Donglai Xiang, Hao Su, Sanja Fidler, Nicholas Sharp, and Jun Gao. Partfield: Learning 3d feature fields for part segmentation and beyond. arXiv preprint arXiv:2504.11451, 2025.

Leland McInnes, John Healy, Steve Astels, et al. hdbscan: Hierarchical density based clustering. J. Open Source Softw., 2(11):205, 2017.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria digital twin: A new benchmark dataset for egocentric 3d machine perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20133–20143, 2023.

Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20051–20060, 2024.

Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visuallanguage-action model. arXiv preprint arXiv:2501.15830, 2025.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen indoors: Photorealistic indoor scenes using procedural generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 21783–21794, June 2024.

Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 12179–12188, 2021.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Doll´ar, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. URL https://arxiv.org/abs/2408.00714.

Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Luˇci´c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6229–6238, 2022.

Johannes L Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4104–4113, 2016.

Johannes Lutz Sch¨onberger and Jan-Michael Frahm. Structure-from-motion revisited. In Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Johannes Lutz Sch¨onberger, Enliang Zheng, Marc Pollefeys, and Jan-Michael Frahm. Pixelwise view selection for unstructured multi-view stereo. In European Conference on Computer Vision (ECCV), 2016.

Oriane Sim´eoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Micha¨el Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Xiangyu Sun, Liu Liu, Seungtae Nam, Gyeongjin Kang, Wei Sui, Zhizhong Su, Wenyu Liu, Xinggang Wang, Eunbyung Park, et al. Uni3r: Unified 3d reconstruction and semantic understanding via generalizable gaussian splatting from unposed multi-view images. arXiv preprint arXiv:2508.03643, 2025.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5294–5306, 2025.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20697–20709, 2024.

Yuxi Xiao, Jianyuan Wang, Nan Xue, Nikita Karaev, Yuri Makarov, Bingyi Kang, Xing Zhu, Hujun Bao, Yujun Shen, and Xiaowei Zhou. Spatialtrackerv2: 3d point tracking made easy. arXiv preprint arXiv:2507.12462, 2025.

Jihan Yang, Shusheng Yang, Anjali W Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643, 2025.

Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pp. 767–783, 2018.

Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 12–22, 2023.

Yue Zhang, Ziqiao Ma, Jialu Li, Yanyuan Qiao, Zun Wang, Joyce Chai, Qi Wu, Mohit Bansal, and Parisa Kordjamshidi. Vision-and-language navigation today and tomorrow: A survey in the era of foundation models. arXiv preprint arXiv:2407.07035, 2024.

Shijie Zhou, Haoran Chang, Sicheng Jiang, Zhiwen Fan, Zehao Zhu, Dejia Xu, Pradyumna Chari, Suya You, Zhangyang Wang, and Achuta Kadambi. Feature 3dgs: Supercharging 3d gaussian splatting to enable distilled feature fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21676–21685, 2024.

Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817, 2018.

Lojze Zust, Yohann Cabon, Juliette Marrie, Leonid Antsfeld, Boris Chidlovskii, Jerome Revaud, and Gabriela Csurka. Panst3r: Multi-view consistent panoptic segmentation. arXiv preprint arXiv:2506.21348, 2025.

A APPENDIX

- A.1 USE OF LARGE LANGUAGE MODELS

Large Language Models (LLMs) are used exclusively for minor grammar corrections and stylistic polishing of the manuscript. They are not involved in the design of the methodology, execution of experiments, analysis of results, or any other aspect of the scientific contribution.

- A.2 RELATED WORK

Spatial Foundation Model Image-to-3D reconstruction is a long-standing problem in computer vision. Early pipelines such as COLMAP (Schonberger & Frahm, 2016) and related SfM methods estimate camera poses and sparse point clouds, often followed by multi-view stereo (MVS) to obtain dense geometry (Yao et al., 2018). Building on such SfM-based initialization, 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) introduced a highly efficient representation for photorealistic novel view synthesis, inspiring reconstruction-oriented extensions (Chen et al., 2024; Huang et al., 2024). To reduce the reliance on accurate calibration, Scene Representation Transformers (Sajjadi et al., 2022; Li et al., 2024a) represent multiple images as latent scene tokens, enabling novel view synthesis under uncertain or missing poses, though they still struggle to produce explicit geometry and generalize reliably. DUSt3R (Wang et al., 2024) takes a further step by directly regressing dense point maps from unposed image pairs, achieving pixel-aligned geometry without SfM initialization. In contrast, VGGT (Wang et al., 2025) scales this paradigm to dozens to hundreds of images in a single feed-forward pass, jointly predicting cameras, depth, point maps, and tracks with competitive accuracy to optimization-based pipelines. Despite these advances, these methods remain focused on low-level geometric reconstruction while overlooking higher-level scene understanding.

- 3D Scene Understanding Integrating semantics into 3D reconstructions is crucial for higher-level scene understanding tasks. Recent efforts (Zhou et al., 2024; Li et al., 2024b; Qin et al., 2024) like LangSplat (Qin et al., 2024) inject vision-language features (e.g., CLIP (Radford et al., 2021)) into 3D Gaussian Splatting, enabling semantic reasoning over reconstructed scenes. However, these methods typically require dense multi-view inputs and per-scene optimization, which hinders scalability. More generalizable approaches like Panst3R (Zust et al., 2025) build on DUSt3R (Wang

- et al., 2024) to achieve feed-forward 3D scene understanding directly from posed or unposed images. Yet, they often decouple reconstruction from understanding and freeze the geometry module, which restricts mutual benefits between the two and leads to suboptimal semantic grounding. Parallel attempts such as LSM (Fan et al., 2024) and Uni3R (Sun et al., 2025) seek to bridge geometry and semantics by aligning spatial models with specific vision-language models (e.g., LSeg (Li et al., 2022)), but this tight coupling has two key drawbacks: (1) it prevents seamless integration of stronger VLMs (Tschannen et al., 2025; Sim´eoni et al., 2025) as they emerge, thereby constraining text query performance; (2) the alignment is typically at the category level rather than instancelevel, so these methods struggle with fine-grained, object-centric QA in scenes that contain multiple similar instances.

To address this problem, our proposed framework, IGGT, addresses these limitations by learning a unified representation for both reconstruction and understanding. Instead of tightly coupling with a single VLM, we introduce an instance-grounded paradigm where instance masks serve as a bridge to connect with diverse VLMs and Large Multimodal Models (LMMs) in a plug-and-play manner, substantially expanding downstream capabilities.

- A.3 TRAINING DETAILS

Our model is initialized with weights from VGGT (Wang et al., 2025) and fine-tuned on the InsScene-15K dataset, which contains 15,000 scenes. Training is performed on 8 NVIDIA A800 GPUs for 2 days using the AdamW optimizer. The learning rate is set to 1 × 10−6 for the large unified Transformer backbone and 1 × 10−5 for both the geometry and instance heads. For each training batch, we randomly sample 1–12 frames from a randomly selected scene, yielding a total of 24 images per batch. For hyper-parameter settings, we set λpull = 2.0, λpull = 1.0 and M = 1.0.

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

RGB Image

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

GT Labels

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

Our Results

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

RGB Image

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

GT Labels

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

Our Results

Figure 12: Visualization of our manually annotated tracking GT and our tracking results.

- A.4 METRICS FOR DIFFERENT TASKS

Instance Spatial Tracking. For the Instance Spatial Tracking task, we evaluate tracking performance using Temporal mIoU (T-mIoU) and Temporal Success Rate (T-SR). Given an object o and

its predicted masks {Mˆto}Tt=1 across T views with corresponding ground-truth masks {Mto}Tt=1, T-mIoU is defined as

T

|Mˆto ∩ Mto| |Mˆto ∪ Mto|

1 T

T-mIoU(o) =

.

t=1

T-SR evaluates whether the object is successfully tracked across all views, and is defined as T-SR(o) = ⊮ ∀t ∈ {1,...,T}, |Mˆto| > 0 ,

where ⊮[·] denotes the indicator function. The final scores are averaged over all objects in the dataset.

RGB 3D Points

Semantic 3D Points (Floor) Voxelization of Semantic 3D Points Voxel Comparison for 3D mIoU Evaluation

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

GT

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

Ours

- Figure 13: Visualization of the pipeline from RGB 3D points to semantic labeling, voxelization, and voxel comparison for 3D mIoU.

GT Ours LSM(Multi-Views) Feature-3DGS

Semantic3DPointsRGB3DPoints

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

- Figure 14: We visualize the RGB and semantic 3D points of the ground truth, IGGT(Ours), LSM(Multi-Views), and Feature-3DGS.

3D Semantic Segmentation mIoU. To evaluate 3D semantic segmentation, we first obtain the RGB 3D points from per-image point maps and align them with the ground truth. Next, we assign semantic labels to the corresponding 3D points based on the results of 2D open-vocabulary segmentation. These labeled 3D points are subsequently voxelized, and the 3D mIoU is computed based on the voxel representation. Fig. 13 illustrates the overall pipeline. Additionally, Fig. 14 presents qualitative results of 3D open-vocabulary segmentation. For LSM, based on its two-view input, we apply the global alignment strategy of Dust3R to optimize the point maps across all views. For Feature3DGS, the ground-truth point maps are used as the initial input. However, due to the sparsity of input views, its reconstruction quality remains limited.

###### RGB Images on Scannet++ Vanilla 2D GT Masks InsScene-15K Masks (Ours)

[Figure 488]

[Figure 489]

Figure 15: Comparison between vanilla Scannet++ GT masks and our refined results.

- A.5 ADDITION VISUALIZATION OF OUR INSSCENE-15K DATASET

Fig. 15 presents the vanilla masks and the refined counterparts, together with the IDs that establish the correspondence between them. The refined masks contain fewer unannotated regions and align more closely with the actual objects. Training with these high-quality instance-level masks facilitates more accurate instance-level segmentation and tracking.

- A.6 DECLARATION OF LLM USAGE

In the preparation of this work, the authors used LLM (e.g., GPT-4) in order to improve the readability and language of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

- A.7 LIMITATION

Our method adopts an unsupervised clustering strategy on the proposed Instance-Grounded Clustering for post-processing. As a result, the accuracy of object boundaries in the clustered masks cannot yet rival that of state-of-the-art segmentation models (e.g., SAM2 (Ravi et al., 2024)). Future work may integrate stronger DETR-based (Cheng et al., 2022) instance heads and larger annotated datasets to improve segmentation accuracy.

