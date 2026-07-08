### One4D: Unified 4D Generation and Reconstruction via Decoupled LoRA Control

# arXiv:2511.18922v1[cs.CV]24Nov2025

Zhenxing Mi, Yuxin Wang, Dan Xu The Hong Kong University of Science and Technology (HKUST)

zmiaa@connect.ust.hk, ywangom@connect.ust.hk, danxu@cse.ust.hk

[Figure 1]

[Figure 2]

Single image

“A vibrant aquarium filled with several goldfish, swim freely …”

[Figure 3]

[Figure 4]

[Figure 5]

###### Sparse frames

“A black swan gracefully swims in a pond, its dark plumage …”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Full video

“A gray Cooper navigates a roundabout in a European city …”

[Figure 11]

[Figure 12]

[Figure 13]

Input 4D point clouds & cameras RGB & Depth time

Figure 1. One4D supports single-image-to-4D generation, sparse-frame-to-4D generation, and full-video reconstruction in a single model. It outputs synchronized RGB frames and pointmaps, visualized as 4D point clouds with cameras, and RGB-depth sequences.

##### Abstract

We present One4D, a unified framework for 4D generation and reconstruction that produces dynamic 4D content as synchronized RGB frames and pointmaps. By consistently handling varying sparsities of conditioning frames through a Unified Masked Conditioning (UMC) mechanism, One4D can seamlessly transition between 4D generation from a single image, 4D reconstruction from a full video, and mixed generation and reconstruction from sparse frames. Our framework adapts a powerful video generation model for joint RGB and pointmap generation, with carefully designed network architectures. The commonly used diffusion finetuning strategies for depthmap or pointmap reconstruction often fail on joint RGB and pointmap generation, quickly degrading the base video model. To address this challenge, we introduce Decoupled LoRA Control

(DLC), which employs two modality-specific LoRA adapters to form decoupled computation branches for RGB frames and pointmaps, connected by lightweight, zero-initialized control links that gradually learn mutual pixel-level consistency. Trained on a mixture of synthetic and real 4D datasets under modest computational budgets, One4D produces high-quality RGB frames and accurate pointmaps across both generation and reconstruction tasks. This work represents a step toward general, high-quality geometrybased 4D world modeling using video diffusion models. Project page: https://mizhenxing.github.io/One4D.

##### 1. Introduction

Simulating the dynamics of the physical world has progressed rapidly with video diffusion and flow-matching models [2, 3, 17, 19, 36, 39, 53, 63]. Recent open systems

such as Wan [39], HunyuanVideo [17], and Cosmos [23] demonstrate remarkable visual fidelity and strong understanding of real-world dynamics. However, these foundation models operate purely in RGB space while lacking explicit geometry. Augmenting them with accurate geometry generation is a key step toward downstream worldsimulation tasks such as spatial reasoning [50].

Concurrently, scalable 3D/4D foundation models have advanced rapidly. Dust3R [43] introduces a pointmap representation encoding both geometry and camera information, enabling efficient feedforward 3D reconstruction. Monst3R [56] and VGGT [40], etc., show that pointmaps are effective for static 3D and dynamic 4D reconstruction. Meanwhile, several works extend video or multi-view diffusion models with 6D video representations (RGB+XYZ) for 4D reconstruction [14], 3D world generation [59], 3D generation and reconstruction [33], and 4D generation [5]. However, these methods typically specialize in either reconstruction or generation, or are restricted to static 3D scenes.

In this paper, we take a step further and propose One4D, a unified 4D generation and reconstruction framework that significantly enhances the fidelity of joint RGB-geometry modeling through innovative network designs.

We represent each 4D scene as RGB frames and pointmaps due to their flexibility and scalability, where pointmaps are 3-channel 2D (XYZ) videos analogous to RGB videos. Prior work [5, 14, 33, 59] shows that modern video VAEs can effectively encode and decode pointmaps. A central challenge in joint RGB-geometry modeling is to fully exploit the strong priors of a pretrained video diffusion model to generate distinct modalities. Existing diffusionbased geometry methods [14, 15, 59] typically couple RGB and geometry through channel-wise concatenation, which in our setting causes severe cross-modal interference and rapid degradation under low-resource fine-tuning.

To address this, we introduce the Decoupled LoRA Control (DLC), an efficient adaptation design guided by three goals: (i) preserve the base model’s strong video priors under low-resource finetuning; (ii) decouple RGB and geometry generation to reduce mutual interference; (iii) enable sufficient cross-modal communication for pixel-level consistency. Concretely, DLC attaches two decoupled and modality-specific LoRA adapters to the video backbone, one for RGB and the other for geometry. The adapters share the frozen base parameters but not the base forward computation, forming two fully decoupled computation branches. The RGB branch maintains pretrained video quality, while the geometry branch adapts to the “geometry video” distribution. To synchronize the two modalities, we add zero-initialized control links [57] between a few corresponding layers across branches. These links gradually learn to transmit information for precise pixel-wise alignment. In practice, DLC prevents mode collapse and yields

accurate geometry without sacrificing RGB fidelity.

To unify 4D generation and reconstruction tasks, we further propose Unified Masked Conditioning (UMC). UMC packs different condition types into a single conditioning video that differs in the sparsity of observed frames, filling unobserved frames with zeros. A single-image (with text) corresponds to pure generation. A set of sparse frames corresponds to mixed generation and reconstruction, and a full video corresponds to pure reconstruction. The conditioning video is fed to the diffusion backbone to guide the network to generate missing RGB frames and full pointmaps. With UMC, One4D transitions seamlessly between generation and reconstruction without any architectural changes.

We train One4D on a curated mixture of synthetic and real 4D datasets, combining accurate synthetic geometry and in-the-wild appearance. The resulting model achieves generalizable, high-quality 4D generation and reconstruction in a single framework. Our main contributions are:

- • We introduce One4D, a unified 4D framework that bridges 4D generation and 4D reconstruction within a single video diffusion model, achieving strong performance across diverse dynamic scenarios.
- • We propose Decoupled LoRA Control (DLC), which uses modality-specific LoRA branches and zero-initialized control links to preserve video priors while enabling accurate, consistent joint RGB-geometry generation.
- • We design Unified Masked Conditioning (UMC), a single conditioning interface that supports various 4D generation and reconstruction tasks without model changes.

##### 2. Related Work

Video generation models. Video generation models [10, 17, 19, 23, 24, 39, 53, 63] have advanced rapidly with diffusion [11, 26, 28, 30, 31] and flow-matching models [6, 20]. Modern systems typically combine 3D causal VAE [2, 16, 46, 55] with DiTs [26] to model spatiotemporal dynamics. Recent open-weight models such as Wan [39], HunyuanVideo [17], and Cosmos [23] demonstrate strong world modeling abilities, suggesting possibilities for geometry generation. Built upon powerful base textto-video models, image-to-video systems such as Wan [39] and LongCat-Video [37] unify image-to-video, frame interpolation, and video continuation via frame inpainting within a single architecture. One4D follows a similar spirit to unify the 4D generation and reconstruction tasks, with specific designs for joint RGB and geometry generation.

Scalable geometry modeling. DUSt3R [43] introduces a paired pointmap representation that encodes both geometry and camera information in a 2D map, enabling scalable 3D reconstruction with transformers. Subsequent works [14, 34, 40, 42, 51, 56] extend this representation to multiview and dynamic reconstruction. Our method leverages these advances by adopting pointmaps as the geometry

[Figure 14]

[Figure 15]

LoRA

LoRA

###### ©

[Figure 16]

[Figure 17]

[Figure 18]

LoRA

[Figure 19]

[Figure 20]

Pixel-wise

DiTlayer

DiTlayer

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Diffusion model

⨁

[Figure 26]

Zero-initialized Control Link

RGB

RGB & XYZ

(a) Channel-wise concat.

[Figure 27]

Zero-initialized Control Link

©

[Figure 28]

[Figure 29]

DiTlayer

DiTlayer

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Diffusion

model ⨁

Pixel-wise

[Figure 35]

LoRA

XYZ

RGB XYZ

[Figure 36]

[Figure 37]

LoRA

LoRA

(b) Spatial-wise concat.

(c) Decoupled LoRA Control (Ours)

Figure 2. Architecture comparison for joint RGB and geometry modeling. (a) Channel-wise and (b) spatial-wise concatenation feed RGB and XYZ into a single diffusion model with a shared LoRA branch. (c) Our Decoupled LoRA Control (DLC) employs two modality-specific LoRA branches with zero-initialized control links, achieving decoupled yet controlled RGB–XYZ joint generation. © denotes concatenation and ⊕ denotes pixel-wise addition.

representation for joint RGB–geometry generation.

- 3D and 4D generation. Leveraging strong 2D diffusion models [7, 28] for 3D/4D advances rapidly. Several methods repurpose diffusion models for reconstructing depth and normal maps [15, 21], conditioning noisy geometry with clean RGB latents. Other works use diffusion models as multiview generators conditioned on cameras, such as CAT3D [8], Bolt3D [33], and Stable virtual camera [64], and extend this to multi-view videos for 4D generation [47, 49]. Their support for a single or multiple views with camera poses as input is also a multitasking design. Some other works focus on 3D object generation from images or text [27, 48, 58, 61].

Several methods model geometry and camera poses jointly via pointmaps, raymaps, or depth maps [5, 14, 22, 33, 59]. WVD [59] generates 6D (RGB+XYZ) static scenes from a single image using channel-wise concatenation of RGB and pointmaps, while 4DNeX [5] generates dynamic scenes via spatial-wise concatenation. Our method also adopts the RGB+XYZ representation, but differs in two key aspects. (1) We unify dynamic 4D generation and reconstruction within a single model. (2) We introduce Decoupled LoRA Control (DLC), which achieves substantially higher quality and geometry accuracy than channel-wise or spatial-wise concatenation for coupling RGB and geometry.

##### 3. Method

###### 3.1. Overview

One4D extends a flow-matching video generation model [6, 20, 39] for joint RGB and geometry generation. As shown in Figure 4, given input images and a text prompt, the model generates synchronized RGB frames Xrgb ∈ R3×F×H×W and pointmaps Xxyz ∈ R3×F×H×W, where each pointmap pixel stores the 3D coordinates of the corresponding RGB

[Figure 38]

[Figure 39]

[Figure 40]

RGBXYZDepth

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Channel-wise Concat.

Spatial-wise Concat. 4DNeX

Decoupled LoRA Control One4D (Ours)

Figure 3. Comparison of architectures for joint RGB–geometry generation. Our Decoupled LoRA Control produces cleaner RGB and sharper, more consistent XYZ and depth than channel-wise and spatial-wise concatenation, while channel-wise concatenation severely degrades both appearance and geometry.

pixel. F denotes the number of frames and H ×W the spatial resolution. At each training step, RGB and pointmaps are encoded by a video VAE into latents zrgb,zxyz ∈ Rc×f×h×w. We then sample a timestep t ∈ [0,1] and Gaussian noise ϵtrgb,ϵtxyz ∼ N(0,I), , and construct noisy latents using the Rectified Flow formulation [6, 20]:

ztrgb = tzrgb + (1 − t)ϵtrgb, (1) ztxyz = tzxyz + (1 − t)ϵtxyz. (2)

The noisy latents and conditioning inputs are fed into DiTs [26] to predict velocities supervised by:

vrgbt = zrgb − ϵrgb (3) vxyzt = zxyz − ϵxyz, (4)

and we train with a mean-squared error loss between predicted and ground-truth velocities for both modalities.

Based on this structure, we introduce Decoupled LoRA Control (DLC) for stable, modality-specific adaptation with pixel-level consistency between RGB and geometry, and Unified Masked Conditioning (UMC) to handle 4D generation and reconstruction in a single model. After generation, camera poses and depthmaps are recovered from the pointmaps via a lightweight global optimization [14, 43].

###### 3.2. Decoupled LoRA Control

Previous diffusion-based reconstruction methods, such as Marigold [15] and Geo4D [14], generate depthmaps or pointmaps conditioned on clean RGB inputs. They typically concatenate clean RGB latents with noisy geometry latents channel-wise, which works when RGB is fixed.

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

###### Sparse

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

Single UMC

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

###### Full

or

or

Frames Masks

Frames Masks

Frames Masks

[Figure 108]

|VAE| |
|---|---|
| | |
|padding<br><br>| |

| | |
|---|---|
| | |
| | |

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Video V

LoRA

LoRA

LoRA

LoRA

[Figure 113]

Noise

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

VideoVAE

[Figure 126]

[Figure 127]

[Figure 128]

VideoVAEVideoVAE

[Figure 129]

[Figure 130]

[Figure 131]

DiTlayer

DiTlayer

DiTlayerDiTlayer

DiTlayer

Pixel-wise

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

…

[Figure 142]

[Figure 143]

…

©

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

⨁

⨁

[Figure 150]

Channel-wise

Zero-initialized Control Link

RGB

RGB latents

[Figure 151]

[Figure 152]

[Figure 153]

Zero-initialized Control Link

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

VideoVAE

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

Channel-wise

DiTlayer

DiTlayer

DiTlayer

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

…

…

[Figure 180]

[Figure 181]

## ©

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

⨁

#### ⨁

Pixel-wise

[Figure 188]

[Figure 189]

Noise

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

XYZ latents

XYZ

LoRA

LoRA

LoRA

LoRA

Zero pad

[Figure 194]

[Figure 195]

Text encoder DiT with DLC

Text prompt

- Figure 4. Overview of the One4D framework. Unified Masked Conditioning (UMC) packs single-image, sparse-frame, and full-video inputs into a masked conditioning video. RGB and XYZ videos are encoded into latent spaces via video VAEs, and the conditioning latents are concatenated only with noisy RGB latents. These RGB and XYZ latents are then processed by a DiT backbone with Decoupled LoRA Control (DLC). DLC employs modality-specific LoRA branches to decouple computation, and zero-initialized cross-modal control links to learn pixel-wise consistency. The denoised RGB and XYZ latents are finally decoded into RGB frames and pointmaps.

However, in the joint generation setting, both RGB and geometry latents are noisy, and such direct coupling leads to severe quality degradation. WVD [59] still adopts channelwise concatenation for joint RGB and XYZ generation, but is trained only on static scenes and requires extreme compute (over 1M steps on 64 A100s). As shown in Figure 3, under moderate compute, we observe that both channelwise and spatial-wise concatenation [5] induce premature and excessive interaction between modalities, degrading RGB quality or preventing high-quality geometry learning.

To address these challenges, we propose the Decoupled LoRA Control (DLC), which decouples RGB and XYZ computation to minimize cross-modal interference, while introducing pixel-wise cross-modal communication in a gradually learnable manner.

Decoupled computation. DLC’s first principle is decoupling. We maintain two computation branches for RGB and XYZ tokens, each equipped with its own LoRA adapter, so the model can adapt to each modality without mutual degradation. For a submodule with LoRA in DiTs, the modalityspecific computation can be written as:

z′rgb = DiTSubmoduleWithRGBLoRA(zrgb), (5) z′xyz = DiTSubmoduleWithXYZLoRA(zxyz) (6)

Implementing the decoupled computation by simply duplicating all DiT parameters and attaching distinct LoRAs to each copy is infeasible for large-scale base models (14B parameters in our case). Instead, we share the frozen base

parameters between modalities and add two separate LoRA adapters on each DiT submodule, while keeping the forward computation disjoint.

###### z′rgb = DiTSubmodule(zrgb) + RGBLoRA(zrgb), (7) z′xyz = DiTSubmodule(zxyz) + XYZLoRA(zxyz) (8)

Each DiT submodule is evaluated once per modality, reusing weights but keeping computations decoupled. This drastically reduces memory usage compared to duplicating parameters, making finetuning feasible for large models. The decoupled design of DLC preserves pretrained video priors in RGB branch while allowing XYZ branch to adapt to pointmap generation, achieving high fidelity in both modalities without cross-modal degradation.

Control links. DLC’s second principle is control. For 4D generation, RGB and geometry outputs must be pixel-wise consistent. Channel-wise concatenation enforces strong coupling but harms video quality, while spatial-wise concatenation relies on non-pixel-aligned attention interactions that are relatively weak, as shown in Figure 3. To obtain strong yet controllable cross-modal consistency, we introduce lightweight control links to connect the two branches.

Let the DiT backbone have N layers, and let {Li

m}, with 1 ≤ i1 < ··· < im ≤ N denote a subset of m layers where we insert cross-modal control links. At a linked layer l, features of one modality are up-

,...,Li

1

dated by features from the other modality:

zˆ(rgbl) = z(rgbl) + ZCLrgb←xyz z(xyzl) , zˆ(xyzl) = z(xyzl) + ZCLxyz←rgb z(rgbl) ,

(9)

where ZCLrgb←xyz and ZCLxyz←rgb are zero-initialized linear control links. The updated features zˆ(rgbl) are zˆ(xyzl) then passed to the subsequent DiT layers. Zero initialization [57] keeps the two branches fully independent at the start of training, preserving the pretrained video priors and the links gradually learn pixel-level alignment between RGB and geometry. In practice, we link only a small subset of layers.

Compared to concatenation-based coupling, DLC provides a controlled, sparsely inserted, and gradually learned pathway for cross-modal communication. As shown in Figure 3, it achieves stronger pixel-wise consistency while maintaining high RGB fidelity and geometric accuracy. It also avoids the token-doubling issue of spatial-wise concatenation within a single attention operation, thereby reducing memory and compute cost.

###### 3.3. Unified Masked Conditioning

We introduce Unified Masked Conditioning to express single-image, sparse-frame, and full-video conditioning within a single interface. Prior video generation models such as Wan [39] and LongCat-Video [37] unify different video generation tasks via frame inpainting. We extend this idea to unified 4D generation and reconstruction, where RGB and geometry are modeled jointly.

Condition construction. We assemble the available image conditions into a conditioning video Xc ∈ RC×F×H×W, matching the RGB shape, and fill unobserved frames with zeros. Xc is encoded by the video VAE into zc ∈ Rc×f×h×w, which is concatenated channel-wise with the RGB latents zrgb. We also build a binary mask Mc ∈ R1×F×H×W [39] indicating observed vs. unobserved frames, reshape it to latent resolution Mc ∈ Rc

m×f×h×w, and concatenate it with RGB latents. The DiT input is:

###### zinput = Concat(zrgb,zc,Mc) (10)

Given zc and Mc, the model will generate missing RGB frames and the full set of pointmaps. This unified construction handles single-image, sparse-frame, and full-video conditioning without changing the architecture.

Controlling geometry. Since all XYZ frames are always generated, we do not feed zc or Mc directly to the geometry branch, avoiding conditioning artifacts in geometry. Instead, conditioning signals reach the geometry branch through DLC control links from the RGB branch, allowing the geometry branch to focus on accurate 3D structure.

With the above designs, UMC makes One4D seamlessly switch between 4D generation and reconstruction tasks within one unified model.

###### 3.4. Post-Optimization

After 4D generation, we apply a simple global optimization to recover camera parameters and depth maps from the generated pointmaps, following MonST3R [56] and Geo4D [14]. Given the N generated point maps {Xˆ i}Ni=1, each Xˆ i ∈ RH×W×3, we recover {Ki, Ri, oi, Di}Ni=1, where Ki is the intrinsic matrix, Ri is the world-to-camera rotation matrix, oi is the camera center in the global frame, and Di is the depth map of frame i.

For each frame i and pixel (u,v), the corresponding 3D point in the global reference frame is parameterized as:

Xiuv = Ri⊤ Duvi Ki−1(u,v,1)⊤ + oi, (11) where Duvi is the depth value at pixel (u,v) in frame i.

The predicted point maps Xˆ i is treated as the obser-

vations of Xi and they are aligned by the loss: Lp = N i=1 u,v Xiuv − Xˆ iuv

. We minimize Lp with respect

1

to {Ki,Ri,oi,Di}Ni=1, yielding globally consistent camera parameters and depth maps. We further regularize the camera trajectory by a temporally smooth loss [14, 56]:

N−1

Ri⊤Ri+1 − I

+ oi+1 − oi 2 .

Ls(R,o) =

f

i=1

(12) The final post-optimization objective is a weighted combination of the point-map alignment and trajectory smoothness losses: Lall = α1Lp + α2Ls.

##### 4. Experiments 4.1. Implementation Details

Datasets. We construct 4D training datasets from both synthetic and real videos. We first collect dynamic synthetic 4D datasets OmniWorld-Game [65], BEDLAM [1], PointOdyssey [62], TarTanAir [44], which provide accurate geometry data. To increase dataset scale and diversity, we further annotate real-world videos in SpatialVID [41] with pseudo geometry using Geo4D [14], covering diverse inthe-wild dynamic scenes. Long videos are clipped into segments of about 81 frames and captioned with Gemini-2.0Flash [9]. In total, we obtain about 17k synthetic and 17k real-world clips, with roughly 2M frames. Given camera parameters, depth maps are lifted to 3D pointmaps using the first frame as the global coordinate frame, and pointmaps are normalized to [−1,1] before video-VAE encoding.

Training. One4D is built on Wan2.1-Fun-V1.1-14BInP [35], a community finetuned version of Wan2.1-I2V14B [38] for video inpainting. We apply LoRA with rank 64 to all linear layers in the DiT for both RGB and XYZ branches, with 685M parameters. We add DLC control links to five DiT layers, introducing 250.7M parameters.

RGB Depth 4D point clouds & cameras

RGB Depth 4D point clouds & cameras

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

4DNeX One4D (Ours)

- Figure 5. Single-image-to-4D generation comparison between 4DNeX [5] and our One4D. Compared to 4DNeX, One4D produces more dynamic and realistic videos, sharper and cleaner depth, and more complete, coherent 4D point clouds with cameras.

Table 2. VBench [13] for video quality. One4D significantly improves motion dynamics and aesthetic quality over 4DNeX, while maintaining comparable image-to-video (I2V) consistency.

Table 1. User study comparing 4DNeX [5] and One4D for 4D generation. Percentages indicate user preference. Our model shows a clear overall advantage, especially on geometry-related criteria.

Method Consistency ↑ Dynamic ↑ Aesthetic ↑ Depthmap ↑ 4D ↑

Method Dynamic ↑ I2V consistency ↑ Aesthetic ↑

4DNeX [5] 21.0% 16.7% 17.7% 11.7% 10.0% One4D (Ours) 78.9% 83.3% 82.3% 88.3% 90.0%

4DNeX [5] 25.6% 98.7% 61.9% One4D (Ours) 55.7% 97.8% 63.8%

Overall, our model has 935.7M trainable parameters. Training is done on 8 NVIDIA H800 GPUs with batch size 1 per GPU and gradient accumulation of 4, for 5500 steps at a learning rate of 1 × 10−4. We randomly switch among tasks by varying the number of masked frames. The task sampling ratios are 0.35 for single-image, 0.30 for sparseframe, and 0.35 for full-video input. The maximum number of training frames is 81, at a resolution of 352 × 624.

a user study, and qualitative comparisons.

As shown in Table 1, One4D is preferred over 4DNeX across image-to-video consistency, motion dynamics, aesthetics, and geometry-related criteria such as depth quality and overall 4D coherence. VBench scores in Table 2 further show that One4D produces stronger, more natural motion while maintaining comparable image-to-video consistency. These results support the effectiveness of our decoupled design, which learns accurate geometry without sacrificing video quality.

Inference. At inference, we use 50 flow-matching steps with a classifier-free guidance scale 6.0. The model jointly generates RGB frames and corresponding pointmaps (XYZ). From the pointmaps, we derive depth maps and estimate camera trajectories via post-optimization. For visualization, pointmaps (XYZ) are interpreted as RGB images. Depth maps are mapped to three-channel images. We use Viser [54] to visualize 4D point clouds together with their camera trajectories, typically subsampling frames with a temporal stride to better show motion.

- Figure 5 qualitatively compares RGB frames, 4D point

clouds, and depth maps generated by the two methods. One4D generates finer geometric details, more accurate depth, and richer motion, whereas 4DNeX’s spatial-wise concatenation struggles to produce fine-grained geometry, as also illustrated in Figure 3. After lifting to 4D, One4D yields coherent scenes with stable backgrounds and naturally moving foreground objects, while 4DNeX often shows limited dynamics. Additional single-image-to-4D results in

- Figure 6 demonstrate that One4D produces high-quality geometry and realistic 4D structure across diverse indoor, outdoor, and static, dynamic scenarios.

###### 4.2. 4D Generation

We compare One4D with the recent 4D generation method

- 4DNeX [5], which relies on spatial-wise concatenation. Both models take a single image and a text prompt as input. We evaluate the generated results using VBench [13],

These high-quality RGB videos and fine-grained, pixel-

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

A group of statues … Hot air balloons flying … A living room … The pyramids of giza …

- Figure 6. Additional single-image-to-4D generation results from One4D. It can generate coherent 4D geometry for various types of scenes.

Table 3. Trained as a unified generation–reconstruction model (G&R), One4D outperforms reconstruction-only (R) pointmapbased methods such as MonST3R and CUT3R, and remains reasonably close to the reconstruction-only Geo4D-ref, demonstrating effective geometry reconstruction within a unified architecture.

Sintel [4] Bonn [25] Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑

Method Task

Marigold [15] R 0.532 51.5 0.091 93.1 Depth-Anything [52] R 0.367 55.4 0.106 92.1 NVDS [45] R 0.408 48.3 0.167 76.6 ChronoDepth [29] R 0.687 48.6 0.100 91.1 DepthCrafter [12] R 0.270 69.7 0.071 97.2 Robust-CVD [18] R 0.703 47.8 - CasualSAM [60] R 0.387 54.7 0.169 73.7 MonST3R [56] R 0.335 58.5 0.063 96.4 CUT3R [42] R 0.311 62.0 0.070 96.7 Geo4D-ref [14] R 0.205 73.5 0.059 97.2

One4D (Ours) G&R 0.273 70.4 0.092 93.7

wise aligned geometry validate the design of Decoupled LoRA Control (DLC), which preserves the strong generative priors of the base model while learning accurate geometry and maintaining pixel-wise consistency.

###### 4.3. 4D Reconstruction

We evaluate 4D reconstruction in both full-video and sparse-frame settings on several benchmarks to assess our unified generation–reconstruction framework. Sintel [4] provides synthetic video with accurate ground-truth depthmaps, about 50 frames per video. Bonn [25] contains real dynamic indoor scenes, about 110 frames per video. TUM-dynamics[32] contains dynamic scenes sampled to 30 frames for each video. Our evaluation setting closely follows MonST3R[56] and Geo4D[14].

We compared depth and camera trajectory derived from our generated pointmaps to reconstruction-only baselines. Marigold [15] and Depth-Anything-V2 [52] are singleimage depth methods. NVDS [45], ChronoDepth [29] and DepthCrafter [12] operate on videos. In addition, RobustCVD [18], CasualSAM [60], MonST3R [56], CUT3R [42], Geo4D [14] jointly reconstruct video depth maps and camera poses. Although Geo4D [14] is used to annotate our real-world training data, we still report its numbers (Geo4DRef) as a strong reconstruction-only reference.

For depth evaluation, we use Sintel [4] and Bonn [25], align predicted depths to the ground truth, and report the

- Table 4. Camera trajectory accuracy. The Task column distinguishes reconstruction-only (R) methods from our unified generation–reconstruction model (G&R). One4D achieves camera accuracy comparable to strong reconstruction-only baselines, indicating our unified 4D model performs competitive camera estimation.

Method Task

Sintel [4] TUM-dynamics [32] ATE↓ RPE-T↓ RPE-R↓ ATE↓ RPE-T↓ RPE-R↓

Robust-CVD [18] R 0.360 0.154 3.443 0.153 0.026 3.528 CasualSAM [60] R 0.141 0.035 0.615 0.071 0.010 1.712 MonST3R [56] R 0.108 0.042 0.732 0.063 0.009 1.217 CUT3R [42] R 0.208 0.062 0.610 0.046 0.014 0.446 Geo4D-ref [14] R 0.185 0.063 0.547 0.073 0.020 0.635

One4D (Ours) G&R 0.213 0.057 0.818 0.129 0.022 1.447

- Table 5. Depth accuracy for sparse-frame-to-4D generation on Sintel and Bonn. Using only a fraction of frames (Sparsity), One4D remains competitive and degrades gracefully, indicating strong geometry generation from sparse observations.

Sintel [4] Bonn [25] Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑

Sparsity

0.50 0.314 70.3 0.094 93.5 0.25 0.443 67.7 0.094 93.3 0.10 0.453 64.0 0.099 92.9 0.05 0.641 57.6 0.151 87.2 0.04 - - 0.191 82.5 0.03 - - 0.277 71.1

Full Model 0.273 70.4 0.092 93.7

absolute relative error (Abs Rel) and the percentage of inlier points with δ < 1.25. For camera evaluation, we use Sintel[4] and TUM-dynamics[32], and report Absolute Trajectory Error (ATE), Relative Pose Error in translation (RPE-T), and Relative Pose Error in rotation (RPE-R).

Full-video-to-4D. Table 3 reports full-video depth accuracy. Among pointmap-based methods, One4D clearly outperforms MonST3R [56] and CUT3R [42] on Sintel (70.4% vs 58.5 / 62.0 on δ < 1.25), and remains close to the reconstruction-only Geo4D-ref [14], even though those models are trained purely for reconstruction and One4D is trained once for both 4D generation and reconstruction. This indicates that our Decoupled LoRA Control and Unified Masked Conditioning allow a single generative model to recover highly accurate geometry. Results for CUT3R are obtained using the official Geo4D evaluation scripts. Results of other baselines are taken from MonST3R [56] and Geo4D [14] papers.

Camera accuracy in Table 4 shows that One4D achieves

RGB Depth

of frames, One4D is almost as accurate as in the full-video setting on both datasets. Even at sparsity 0.10, degradation is modest. Moreover, under extreme sparsity (e.g., 0.05 or 0.03), the model still produces reasonable geometry from only 2 or 3 frames, typically just the first and last frames. This is also partly because these boundary frames already capture sufficient background information.

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

Overall, the sparse-frame evaluation shows that our model can reliably generate unobserved RGB frames and corresponding 4D structure from very sparse observations, complementing the single-image-to-4D generation results and highlighting the strength of our unified generation–reconstruction design.

CUT3R MonST3R Geo4D One4D (Ours)

###### 4.4. Additional Ablation Study

- Figure 7. Qualitative full-video 4D reconstruction comparison with CUT3R [42], MonST3R [56], and Geo4D [14]. The proposed One4D recovers sharper object boundaries and more accurate depth, especially on thin structures and challenging geometry.

Beyond our main generation and reconstruction experiments, we further ablate two factors of One4D on Sintel and Bonn, including the classifier-free guidance (CFG) scale and the number of training steps. Results are summarized in Tables 5 and 6.

Table 6. Ablation on CFG scale and training steps for depth reconstruction. Our model is robust to CFG choice and attains good accuracy with few training steps, improving with longer training.

We use CFG = 6 as the default setting in all main experiments. As shown in Table 6, using CFG = 4 or CFG = 5 yields very similar depth accuracy, indicating that One4D is robust to the choice of guidance scale. This robustness simplifies deployment in real-world applications.

Sintel [4] Bonn [25] Abs Rel ↓ δ < 1.25 ↑ Abs Rel ↓ δ < 1.25 ↑

Setting

- CFG=4 0.257 71.5 0.092 94.0

- CFG=5 0.259 70.9 0.090 94.1

We also study the effect of training steps by training models with 1K, 3K, and 5.5K optimization steps. Even with only 1k steps, One4D already achieves reasonable accuracy, and at 3k steps, it is close to the full model. Compared to WVD [59], which requires around 1M steps over two weeks on 64 A100 GPUs with channel-wise concatenation, our decoupled design adapts the base video model with much fewer training steps. This efficiency supports our claim that the proposed architecture preserves and effectively leverages pretrained video priors. As the number of training steps increases, performance generally improves, indicating promising potential for further gains when scaling to larger datasets and more computation.

Step=1000 0.331 65.4 0.114 88.9 Step=3000 0.284 68.1 0.097 91.8

One4D (Ours) 0.273 70.4 0.092 93.7

competitive ATE and RPE scores within the same range of Geo4D-ref and other reconstruction baselines, confirming that our pointmaps are sufficiently accurate to support robust camera estimation. Qualitative results in Figure 7 further show that One4D can recover fine geometric structures (e.g., bamboo leaves, ropes) and robustly handles challenging scenes such as dense bamboo forests.

Overall, the strong depth and camera reconstruction results across Sintel, Bonn, and TUM-dynamics highlight that our designs of DLC and UMC enable One4D to effectively reconstruct 4D geometry and camera trajectories, even though we train it as a single unified model for both generation and reconstruction.

##### 5. Conclusion

We presented One4D, a unified 4D framework that bridges 4D generation and 4D reconstruction within a single video diffusion model. We introduced Decoupled LoRA Control for robust joint RGB and geometry modeling. We proposed Unified Masked Conditioning, a simple conditioning scheme that seamlessly handles pure generation, mixed generation–reconstruction, and pure reconstruction without modifying the architecture. Trained on a curated mixture of synthetic and real 4D data, One4D achieves generalizable, high-quality 4D results and takes a step toward geometryaware world simulation with video foundation models.

Sparse-frame-to-4D. In the sparse-frame-to-4D setting, we keep the first frame and last frame of each video and uniformly sample additional frames in between, with the total number of observed frames controlled by a sparsity ratio. The model must generate all missing RGB frames and the complete pointmap sequence.

Table 5 reports depth accuracy on Sintel and Bonn with different spatial ratios. Remarkably, with only 50% or 25%

##### References

- [1] Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In CVPR, 2023. 5
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 2
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024. 1
- [4] Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. A naturalistic open source movie for optical flow evaluation. In ECCV, 2012. 7, 8
- [5] Zhaoxi Chen, Tianqi Liu, Long Zhuo, Jiawei Ren, Zeng Tao, He Zhu, Fangzhou Hong, Liang Pan, and Ziwei Liu. 4dnex: Feed-forward 4d generative modeling made easy. arXiv preprint arXiv:2508.13154, 2025. 2, 3, 4, 6
- [6] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 2, 3
- [7] Black Forest. Flux. https://github.com/blackforest-labs/flux, 2024. GitHub repository. 3
- [8] Ruiqi Gao, Aleksander Hoły´nski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. Cat3d: create anything in 3d with multi-view diffusion models. In NeurIPS, 2024. 3
- [9] Demis Hassabis, Koray Kavukcuoglu, and the Gemini Team. Introducing gemini 2.0: our new ai model for the agentic era. Online at https : / / blog.google/technology/google-deepmind/ google-gemini-ai-update-december-2024/,

2024. 5

- [10] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. arXiv preprint arXiv:2211.13221,

2022. 2

- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 2
- [12] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan. Depthcrafter: Generating consistent long depth sequences for open-world videos. In CVPR, 2025. 7
- [13] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In CVPR, 2024. 6
- [14] Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, and Andrea Vedaldi. Geo4d: Leveraging video generators for geometric 4d scene reconstruction. In ICCV, 2025. 2, 3, 5, 7, 8
- [15] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurpos-

- ing diffusion-based image generators for monocular depth estimation. In CVPR, 2024. 2, 3, 7
- [16] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014. 2
- [17] Weijie Kong, Qi Tian, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 1, 2
- [18] Johannes Kopf, Xuejian Rong, and Jia-Bin Huang. Robust consistent video depth estimation. In CVPR, 2021. 7
- [19] Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131, 2024. 1, 2
- [20] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2, 3
- [21] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In CVPR,

2024. 3

- [22] Yuanxun Lu, Jingyang Zhang, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Long Quan, Xun Cao, Yao Yao, and Shiwei Li. Matrix3d: Large photogrammetry model all-in-one. In CVPR, 2025. 3
- [23] NVIDIA, :, Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, et al. Cosmos world foundation model platform for physical ai, 2025. 2
- [24] OpenAI. Video generation models as world simulators. https : / / openai . com / index / video generation-models-as-world-simulators/,

2024. 2

- [25] Emanuele Palazzolo, Jens Behley, Philipp Lottes, Philippe Giguere, and Cyrill Stachniss. Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In IROS, 2019. 7, 8
- [26] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023. 2, 3
- [27] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In ICLR,

2023. 3

- [28] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3
- [29] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang, Yujun Shen, Vitor Guizilini, Yue Wang, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors. In CVPR, 2025. 7
- [30] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2020. 2
- [31] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021. 2
- [32] J¨urgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, and Daniel Cremers. A benchmark for the evaluation of rgb-d slam systems. In IROS, 2012. 7

- [33] Stanislaw Szymanowicz, Jason Y. Zhang, Pratul Srinivasan, Ruiqi Gao, Arthur Brussee, Aleksander Holynski, Ricardo Martin-Brualla, Jonathan T. Barron, and Philipp Henzler. Bolt3D: Generating 3D Scenes in Seconds. ICCV, 2025. 2, 3
- [34] Zhenggang Tang, Yuchen Fan, Dilin Wang, Hongyu Xu, Rakesh Ranjan, Alexander Schwing, and Zhicheng Yan. Mv-dust3r+: Single-stage scene reconstruction from sparse views in 2 seconds. In CVPR, 2025. 2
- [35] Alibaba PAI Team. Wan2.1-Fun-V1.1-14B-InP. https:// huggingface.co/alibaba-pai/Wan2.1-FunV1.1-14B-InP, 2024. 5
- [36] Genmo Team. Mochi 1. https://github.com/ genmoai/models, 2024. 1
- [37] Meituan LongCat Team. Longcat-video technical report,

2025. 2, 5

- [38] Wan-AI Team. Wan2.1-I2V-14B-480P. https:// huggingface.co/Wan-AI/Wan2.1-I2V- 14B480P, 2024. 5
- [39] Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 1, 2, 3, 5
- [40] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In CVPR, 2025. 2
- [41] Jiahao Wang, Yufeng Yuan, Rujie Zheng, Youtian Lin, Jian Gao, Lin-Zhuo Chen, Yajie Bao, Yi Zhang, Chang Zeng, Yanxi Zhou, et al. Spatialvid: A large-scale video dataset with spatial annotations. arXiv preprint arXiv:2509.09676,

2025. 5

- [42] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In CVPR, 2025. 2, 7, 8
- [43] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In CVPR, 2024. 2, 3
- [44] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In IROS, 2020. 5
- [45] Yiran Wang, Min Shi, Jiaqi Li, Zihao Huang, Zhiguo Cao, Jianming Zhang, Ke Xian, and Guosheng Lin. Neural video depth stabilizer. In ICCV, 2023. 7
- [46] Pingyu Wu, Kai Zhu, Yu Liu, Liming Zhao, Wei Zhai, Yang Cao, and Zheng-Jun Zha. Improved video vae for latent video diffusion model. In CVPR, 2025. 2
- [47] Rundi Wu, Ruiqi Gao, Ben Poole, Alex Trevithick, Changxi Zheng, Jonathan T Barron, and Aleksander Holynski. Cat4d: Create anything in 4d with multi-view video diffusion models. In CVPR, 2025. 3
- [48] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. In NeurIPS, 2024. 3
- [49] Yiming Xie, Chun-Han Yao, Vikram Voleti, Huaizu Jiang, and Varun Jampani. Sv4d: Dynamic 3d content generation

- with multi-frame and multi-view consistency. In ICLR, 2025. 3
- [50] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv preprint arXiv:2412.14171, 2024. 2
- [51] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In CVPR, 2025. 2
- [52] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In NeurIPS, 2024. 7
- [53] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2
- [54] Brent Yi, Chung Min Kim, Justin Kerr, Gina Wu, Rebecca Feng, Anthony Zhang, Jonas Kulhanek, Hongsuk Choi, Yi Ma, Matthew Tancik, et al. Viser: Imperative, web-based 3d visualization in python. arXiv preprint arXiv:2507.22885,

2025. 6

- [55] Lijun Yu, Jos´e Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. In ICLR, 2024. 2
- [56] Junyi Zhang, Charles Herrmann, Junhwa Hur, Varun Jampani, Trevor Darrell, Forrester Cole, Deqing Sun, and MingHsuan Yang. Monst3r: A simple approach for estimating geometry in the presence of motion. In ICLR, 2025. 2, 5, 7, 8
- [57] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 5
- [58] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. TOG, pages 1–20, 2024. 3
- [59] Qihang Zhang, Shuangfei Zhai, Miguel Angel´ Bautista, Kevin Miao, Alexander Toshev, Joshua Susskind, and Jiatao Gu. World-consistent video diffusion with explicit 3d modeling. In CVPR, 2025. 2, 3, 4, 8
- [60] Zhoutong Zhang, Forrester Cole, Zhengqi Li, Michael Rubinstein, Noah Snavely, and William T Freeman. Structure and motion from casual videos. In ECCV, 2022. 7
- [61] Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025. 3
- [62] Yang Zheng, Adam W Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In CVPR,

2023. 5

- [63] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang

- You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024. 1, 2
- [64] Jensen Zhou, Hang Gao, Vikram Voleti, Aaryaman Vasishta, Chun-Han Yao, Mark Boss, Philip Torr, Christian Rupprecht, and Varun Jampani. Stable virtual camera: Generative view synthesis with diffusion models. arXiv preprint arXiv:2503.14489, 2025. 3
- [65] Yang Zhou, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Haoyu Guo, Zizun Li, Kaijing Ma, Xinyue Li, Yating Wang, Haoyi Zhu, et al. Omniworld: A multi-domain and multi-modal dataset for 4d world modeling. arXiv preprint arXiv:2509.12201, 2025. 5

